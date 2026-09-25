from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import healpy as hp
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats


RAW_SAMPLE_RATE = 4096.0
MODEL_SAMPLE_RATE = 2048.0
MODEL_WINDOW_SECONDS = 24.0
PAD_SECONDS = 1.0
PADDED_WINDOW_SECONDS = MODEL_WINDOW_SECONDS + 2.0 * PAD_SECONDS
MODEL_SAMPLES = int(MODEL_SAMPLE_RATE * MODEL_WINDOW_SECONDS)
RAW_MODEL_SAMPLES = int(RAW_SAMPLE_RATE * MODEL_WINDOW_SECONDS)
RAW_PADDED_SAMPLES = int(RAW_SAMPLE_RATE * PADDED_WINDOW_SECONDS)
PSD_SECONDS = 256.0
SECONDS_PER_DAY = 86400.0


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def robust_scale_channels(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    med = np.median(arr, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(arr - med), axis=-1, keepdims=True)
    std = np.std(arr, axis=-1, keepdims=True)
    scale = np.where(mad > 1e-12, mad, std)
    scale = np.maximum(scale, np.maximum(np.abs(med) * 1e-6, 1e-30))
    return ((arr - med) / scale).astype(np.float32)


def estimate_psd(reference: np.ndarray, fs: float = RAW_SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for each detector channel using finite off-source strain."""
    x = np.asarray(reference, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected [detector,time], got {x.shape}")
    if np.mean(np.isfinite(x)) < 0.999:
        raise ValueError("PSD reference contains non-finite samples")
    nperseg = int(round(8.0 * fs))
    noverlap = nperseg // 2
    freq = None
    rows = []
    for channel in x:
        f, p = signal.welch(
            channel,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        floor = max(float(np.nanmedian(p)) * 1e-12, np.finfo(np.float64).tiny)
        rows.append(np.maximum(p, floor))
        freq = f
    return np.asarray(freq, dtype=np.float64), np.asarray(rows, dtype=np.float64)


def optimal_network_snr(
    strain: np.ndarray,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    fs: float = RAW_SAMPLE_RATE,
    f_low: float = 20.0,
    f_high: float = 1024.0,
) -> float:
    """Discrete matched-filter optimal network SNR for a two-channel signal."""
    h = np.asarray(strain, dtype=np.float64)
    n = h.shape[-1]
    dt = 1.0 / fs
    freq = np.fft.rfftfreq(n, d=dt)
    df = fs / n
    total = 0.0
    for detector in range(h.shape[0]):
        hf = np.fft.rfft(h[detector]) * dt
        sn = np.interp(freq, psd_freq, psd[detector])
        keep = (freq >= f_low) & (freq <= min(f_high, fs / 2.0))
        total += 4.0 * float(np.sum((np.abs(hf[keep]) ** 2) / np.maximum(sn[keep], 1e-60)) * df)
    return float(math.sqrt(max(total, 0.0)))


def scale_to_network_snr(
    strain: np.ndarray,
    target_snr: float,
    psd_freq: np.ndarray,
    psd: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    current = optimal_network_snr(strain, psd_freq, psd)
    if not np.isfinite(current) or current <= 0.0:
        raise ValueError("Clean signal has zero or invalid PSD-weighted SNR")
    factor = float(target_snr) / current
    scaled = np.asarray(strain, dtype=np.float64) * factor
    recovered = optimal_network_snr(scaled, psd_freq, psd)
    return scaled.astype(np.float32), factor, recovered


def whiten_with_psd(
    x: np.ndarray,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    fs: float = RAW_SAMPLE_RATE,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[-1]
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    # The small taper suppresses FFT boundary leakage without attenuating the
    # merger, which is 1.25 s before the padded window end.
    taper = signal.windows.tukey(n, alpha=0.02)
    rows = []
    for detector, channel in enumerate(arr):
        spectrum = np.fft.rfft((channel - np.mean(channel)) * taper)
        sn = np.interp(freq, psd_freq, psd[detector])
        white = np.fft.irfft(spectrum / np.sqrt(np.maximum(sn, 1e-60)), n=n)
        rows.append(white)
    return np.asarray(rows, dtype=np.float64)


def preprocess_24s(
    padded_26s: np.ndarray,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    fs: float = RAW_SAMPLE_RATE,
    band_low_hz: float = 40.0,
    band_high_hz: float = 580.0,
) -> np.ndarray:
    """Whiten, physical-Hz bandpass, anti-alias resample, and crop 24 s."""
    x = np.asarray(padded_26s, dtype=np.float64)
    if x.shape != (2, RAW_PADDED_SAMPLES):
        raise ValueError(f"Expected (2,{RAW_PADDED_SAMPLES}), got {x.shape}")
    white = whiten_with_psd(x, psd_freq, psd, fs=fs)
    sos = signal.butter(6, [band_low_hz, band_high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, white, axis=-1)
    down = signal.resample_poly(filtered, up=1, down=2, axis=-1, window=("kaiser", 8.6))
    pad = int(round(PAD_SECONDS * MODEL_SAMPLE_RATE))
    cropped = down[..., pad : pad + MODEL_SAMPLES]
    if cropped.shape != (2, MODEL_SAMPLES):
        raise RuntimeError(f"Unexpected prepared shape {cropped.shape}")
    return robust_scale_channels(cropped)


def embed_signal_in_padded_window(signal_24s: np.ndarray) -> np.ndarray:
    signal_24s = np.asarray(signal_24s, dtype=np.float32)
    if signal_24s.shape != (2, RAW_MODEL_SAMPLES):
        raise ValueError(f"Expected (2,{RAW_MODEL_SAMPLES}), got {signal_24s.shape}")
    out = np.zeros((2, RAW_PADDED_SAMPLES), dtype=np.float32)
    pad = int(round(PAD_SECONDS * RAW_SAMPLE_RATE))
    out[..., pad : pad + RAW_MODEL_SAMPLES] = signal_24s
    return out


def effective_rank(vectors: np.ndarray) -> float:
    x = np.asarray(vectors, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    variance = singular * singular
    denom = float(np.sum(variance * variance))
    return float(np.sum(variance) ** 2 / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class LiveSegment:
    start: float
    end: float
    run: str
    weight_per_second: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def sample_live_times(segments: list[LiveSegment], size: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray([s.duration * s.weight_per_second for s in segments], dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Live-segment weights are invalid")
    choice = rng.choice(len(segments), size=int(size), p=weights / weights.sum())
    u = rng.random(int(size))
    starts = np.asarray([segments[i].start for i in choice], dtype=np.float64)
    widths = np.asarray([segments[i].duration for i in choice], dtype=np.float64)
    return starts + u * widths


def in_live_segments(values: np.ndarray, segments: list[LiveSegment]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    starts = np.asarray([s.start for s in segments], dtype=np.float64)
    ends = np.asarray([s.end for s in segments], dtype=np.float64)
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    idx = np.searchsorted(starts, values, side="right") - 1
    valid = idx >= 0
    clipped = np.clip(idx, 0, len(ends) - 1)
    valid &= values <= ends[clipped]
    return valid


def draw_detected_lens_delays(
    raw_delays_days: np.ndarray,
    segments: list[LiveSegment],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    delays = np.asarray(raw_delays_days, dtype=np.float64)
    delays = delays[np.isfinite(delays) & (delays > 0)]
    accepted: list[np.ndarray] = []
    remaining = int(size)
    attempts = 0
    while remaining > 0 and attempts < 100:
        batch = max(10_000, 4 * remaining)
        selected = rng.choice(delays, size=batch, replace=True)
        first = sample_live_times(segments, batch, rng)
        keep = in_live_segments(first + selected * SECONDS_PER_DAY, segments)
        if np.any(keep):
            take = selected[keep][:remaining]
            accepted.append(take)
            remaining -= len(take)
        attempts += 1
    if remaining > 0:
        raise RuntimeError(f"Could not draw enough exposure-conditioned lens delays; missing {remaining}")
    return np.concatenate(accepted).astype(np.float64)


def draw_detected_lens_times(
    raw_delays_days: np.ndarray,
    segments: list[LiveSegment],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    delays = np.asarray(raw_delays_days, dtype=np.float64)
    for _ in range(10_000):
        delay = float(rng.choice(delays))
        first = float(sample_live_times(segments, 1, rng)[0])
        second = first + delay * SECONDS_PER_DAY
        if bool(in_live_segments(np.asarray([second]), segments)[0]):
            return first, second, delay
    raise RuntimeError("Unable to place both lensed images in live exposure")


def draw_null_delays(segments: list[LiveSegment], size: int, rng: np.random.Generator) -> np.ndarray:
    a = sample_live_times(segments, int(size), rng)
    b = sample_live_times(segments, int(size), rng)
    values = np.abs(a - b) / SECONDS_PER_DAY
    return values[values > 0]


def _kde_density_grid(values: np.ndarray, grid: np.ndarray, bandwidth_scale: float = 1.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) > 30_000:
        # Deterministic quantile thinning preserves tails better than a random subset.
        q = np.linspace(0.0, 1.0, 30_000, endpoint=False) + 0.5 / 30_000
        x = np.quantile(x, q)
    kde = stats.gaussian_kde(x)
    kde.set_bandwidth(kde.factor * float(bandwidth_scale))
    density = np.maximum(kde(grid), np.finfo(np.float64).tiny)
    density /= np.trapz(density, grid)
    return density


def fit_time_likelihood_ratio(
    lens_delays_days: np.ndarray,
    null_delays_days: np.ndarray,
    grid_size: int = 2048,
    bandwidth_scale: float = 1.0,
) -> dict[str, Any]:
    lens = np.log10(np.maximum(np.asarray(lens_delays_days, dtype=np.float64), 1e-12))
    null = np.log10(np.maximum(np.asarray(null_delays_days, dtype=np.float64), 1e-12))
    combined = np.concatenate([lens[np.isfinite(lens)], null[np.isfinite(null)]])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    margin = max(0.05, 0.02 * (hi - lo))
    grid = np.linspace(lo - margin, hi + margin, int(grid_size))
    p_lens = _kde_density_grid(lens, grid, bandwidth_scale)
    p_null = _kde_density_grid(null, grid, bandwidth_scale)
    log_lr = np.log(p_lens) - np.log(p_null)
    return {
        "log10_delay_grid": grid,
        "p_lens_log10": p_lens,
        "p_null_log10": p_null,
        "log_likelihood_ratio": log_lr,
        "lens_samples": int(len(lens)),
        "null_samples": int(len(null)),
        "bandwidth_scale": float(bandwidth_scale),
        "estimator": "Gaussian KDE in log10(delta_t_days); identical Jacobian cancels in density ratio",
    }


def apply_time_likelihood_ratio(delays_days: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    values = np.log10(np.maximum(np.asarray(delays_days, dtype=np.float64), 1e-12))
    grid = np.asarray(calibration["log10_delay_grid"], dtype=np.float64)
    score = np.asarray(calibration["log_likelihood_ratio"], dtype=np.float64)
    return np.interp(values, grid, score, left=score[0], right=score[-1]).astype(np.float32)


def fit_score_likelihood_ratio(
    signal_values: np.ndarray,
    null_values: np.ndarray,
    grid_size: int = 1024,
    bandwidth_scale: float = 1.0,
) -> dict[str, Any]:
    signal_values = np.asarray(signal_values, dtype=np.float64)
    null_values = np.asarray(null_values, dtype=np.float64)
    merged = np.concatenate([signal_values[np.isfinite(signal_values)], null_values[np.isfinite(null_values)]])
    lo, hi = float(np.min(merged)), float(np.max(merged))
    margin = max(1e-5, 0.02 * (hi - lo))
    grid = np.linspace(lo - margin, hi + margin, int(grid_size))
    p_signal = _kde_density_grid(signal_values, grid, bandwidth_scale)
    p_null = _kde_density_grid(null_values, grid, bandwidth_scale)
    return {
        "score_grid": grid,
        "p_signal": p_signal,
        "p_null": p_null,
        "log_likelihood_ratio": np.log(p_signal) - np.log(p_null),
        "signal_samples": int(np.isfinite(signal_values).sum()),
        "null_samples": int(np.isfinite(null_values).sum()),
        "bandwidth_scale": float(bandwidth_scale),
    }


def apply_score_likelihood_ratio(values: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    grid = np.asarray(calibration["score_grid"], dtype=np.float64)
    score = np.asarray(calibration["log_likelihood_ratio"], dtype=np.float64)
    return np.interp(x, grid, score, left=score[0], right=score[-1]).astype(np.float32)


def sky_log_bayes_factor_from_maps(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(maps, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    raw = p @ p.T
    npix = p.shape[1]
    bayes = npix * raw
    norms = np.linalg.norm(p, axis=1)
    cosine = raw / np.maximum(norms[:, None] * norms[None, :], 1e-300)
    return np.log(np.maximum(bayes, 1e-300)), raw, cosine


def rotation_matrix_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.asarray([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.asarray([0.0, 1.0, 0.0])
        axis -= np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        return axis_angle_matrix(axis, math.pi)
    vx = np.asarray([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    u = np.asarray(axis, dtype=np.float64)
    u /= np.linalg.norm(u)
    ux = np.asarray([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) * math.cos(angle) + (1.0 - math.cos(angle)) * np.outer(u, u) + math.sin(angle) * ux


def rotate_probability_map_to_true_position(
    probability: np.ndarray,
    true_ra: float,
    true_dec: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Rotate a real posterior template so a posterior draw equals true sky."""
    source_map = np.asarray(probability, dtype=np.float64)
    source_map = np.clip(source_map, 0.0, None)
    source_map /= source_map.sum()
    nside = hp.npix2nside(len(source_map))
    anchor_pixel = int(rng.choice(len(source_map), p=source_map))
    source = np.asarray(hp.pix2vec(nside, anchor_pixel, nest=False), dtype=np.float64)
    target = np.asarray(
        [math.cos(true_dec) * math.cos(true_ra), math.cos(true_dec) * math.sin(true_ra), math.sin(true_dec)],
        dtype=np.float64,
    )
    rotation = axis_angle_matrix(target, float(rng.uniform(0.0, 2.0 * math.pi))) @ rotation_matrix_between(source, target)
    out_vectors = np.asarray(hp.pix2vec(nside, np.arange(len(source_map)), nest=False), dtype=np.float64)
    in_vectors = rotation.T @ out_vectors
    theta, phi = hp.vec2ang(in_vectors.T)
    rotated = hp.get_interp_val(source_map, theta, phi, nest=False)
    rotated = np.clip(rotated, 0.0, None)
    rotated /= np.maximum(rotated.sum(), 1e-300)
    return rotated.astype(np.float32), anchor_pixel


def posterior_area90(probability: np.ndarray) -> float:
    p = np.asarray(probability, dtype=np.float64)
    order = np.argsort(p)[::-1]
    n90 = int(np.searchsorted(np.cumsum(p[order]), 0.9, side="left")) + 1
    return float(n90 * hp.nside2pixarea(hp.npix2nside(len(p)), degrees=True))


def percentile_interval(values: Iterable[float], qs: tuple[float, float] = (0.025, 0.975)) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(x, qs[0])), float(np.quantile(x, qs[1]))
