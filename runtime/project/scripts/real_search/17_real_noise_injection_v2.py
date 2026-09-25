from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from gwosc.datasets import run_segment
from gwosc.timeline import get_segments
from sklearn.metrics import average_precision_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matchgw.config import MatchRunConfig  # noqa: E402
from matchgw.aux_priors.observed_sky import a90_to_sigma_rad, sample_observed_sky_center  # noqa: E402
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices  # noqa: E402
from matchgw.matching import similarity_matrix  # noqa: E402
from matchgw.pipeline import embed_eval, run_train_eval  # noqa: E402
from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    across_seed_summary,
    bootstrap_system_ci,
    metric_rows,
    seed_everything,
    split_integrity_audit,
    write_json,
)
from scripts.sky.sky_posterior_overlap import gaussian_log_cosine_overlap_pairs  # noqa: E402


O3_SOURCE = REPO_ROOT / "runs" / "real_gwtc_lensing_search_20260625"
O4_SOURCE = REPO_ROOT / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"
CLEAN_SOURCE = O3_SOURCE / "data" / "real_noise_injections" / "matchroots" / "LIGO"
DEFAULT_OUT = REPO_ROOT / "results" / "real_noise_injection_v2_20260719"
DEFAULT_PACKAGE = REPO_ROOT / "packages" / "real_noise_injection_v2_20260719.tar.gz"
DEFAULT_SEEDS = (202607191, 202607192, 202607193)
FAMILIES = ("SIS", "PM")
FULL_WINDOW = 98_304
MODEL_TAIL = 8_192
SAMPLE_RATE = 4096.0
SECONDS_PER_DAY = 86400.0
SNR_MIN = 8.0
SNR_MAX = 60.0
RHO_REF = 12.0
WEIGHT_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
STRAIN_CACHES: dict[str, "LruStrainCache"] = {}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def prepare_seed_layout(seed_dir: Path, source_run: Path) -> None:
    for name in ("event_manifest.csv", "strain_gwosc_download_manifest.csv"):
        ensure_link(seed_dir / "data" / name, source_run / "data" / name)
    ensure_link(seed_dir / "data" / "real_strain", source_run / "data" / "real_strain")
    ensure_link(
        seed_dir / "features" / "real_pair_observable_features.parquet",
        source_run / "features" / "real_pair_observable_features.parquet",
    )
    inj = seed_dir / "data" / "real_noise_injections"
    inj.mkdir(parents=True, exist_ok=True)
    ensure_link(
        inj / "offsource_noise_segments.parquet",
        source_run / "data" / "real_noise_injections" / "offsource_noise_segments.parquet",
    )
    (seed_dir / "results").mkdir(parents=True, exist_ok=True)
    (seed_dir / "logs").mkdir(parents=True, exist_ok=True)
    (seed_dir / "waveform_gate").mkdir(parents=True, exist_ok=True)


def zscore_channelwise(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mean) * 1e-6, 1e-30)
    return ((x - mean) / np.maximum(std, floor)).astype(np.float32, copy=False)


class LruStrainCache:
    def __init__(self, source_run: Path, max_files: int = 512):
        self.source_run = source_run
        self.max_files = max_files
        self.cache: OrderedDict[str, tuple[np.ndarray, float, float]] = OrderedDict()

    def get(self, rel_path: str) -> tuple[np.ndarray, float, float]:
        if rel_path in self.cache:
            value = self.cache.pop(rel_path)
            self.cache[rel_path] = value
            return value
        path = self.source_run / rel_path
        with h5py.File(path, "r") as h5:
            strain = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
            start = float(np.asarray(h5["meta/GPSstart"][()]).item())
            duration = float(np.asarray(h5["meta/Duration"][()]).item())
        value = (np.nan_to_num(strain), start, duration)
        self.cache[rel_path] = value
        while len(self.cache) > self.max_files:
            self.cache.popitem(last=False)
        return value


def draw_full_noise(
    cache: LruStrainCache,
    segments: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    segment = segments.iloc[int(rng.integers(0, len(segments)))]
    latest = float(segment.segment_end) - FULL_WINDOW / SAMPLE_RATE
    start = float(rng.uniform(float(segment.segment_start), latest))
    channels = []
    for detector in ("H1", "L1"):
        data, gps_start, duration = cache.get(str(segment[f"{detector}_path"]))
        rate = len(data) / duration
        if abs(rate - SAMPLE_RATE) > 1e-3:
            raise ValueError(f"Unexpected {detector} sample rate {rate}")
        i0 = int(round((start - gps_start) * rate))
        i0 = max(0, min(i0, len(data) - FULL_WINDOW))
        channels.append(data[i0 : i0 + FULL_WINDOW])
    return np.stack(channels).astype(np.float32), {
        "noise_event": str(segment.event_name),
        "noise_segment_kind": str(segment.segment_kind),
        "noise_start_gps": start,
    }


def _intersect_segments(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(float(a[i][0]), float(b[j][0]))
        hi = min(float(a[i][1]), float(b[j][1]))
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _trim_segments(segments: list[tuple[float, float]], margin: float = 32.0) -> list[tuple[float, float]]:
    return [(a + margin, b - margin) for a, b in segments if b - a > 2.0 * margin]


def joint_h1l1_live_segments(deployment: str, cache_path: Path) -> list[tuple[float, float]]:
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        return list(frame[["start_gps", "end_gps"]].itertuples(index=False, name=None))
    run_names = ("O3a", "O3b") if deployment == "GWTC3" else ("O4a",)
    joint: list[tuple[float, float]] = []
    for run_name in run_names:
        start, end = run_segment(run_name)
        h1 = sorted((float(a), float(b)) for a, b in get_segments("H1_DATA", int(start), int(end)))
        l1 = sorted((float(a), float(b)) for a, b in get_segments("L1_DATA", int(start), int(end)))
        joint.extend(_intersect_segments(h1, l1))
    joint = _trim_segments(sorted(joint), margin=32.0)
    if not joint:
        raise RuntimeError(f"No H1-L1 live segments found for {deployment}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(joint, columns=["start_gps", "end_gps"]).assign(
        duration_s=lambda x: x.end_gps - x.start_gps,
        deployment=deployment,
    ).to_csv(cache_path, index=False)
    return joint


def _shift_overlap_segments(live: list[tuple[float, float]], delay_s: float) -> list[tuple[float, float]]:
    shifted = [(a - delay_s, b - delay_s) for a, b in live]
    return _intersect_segments(live, shifted)


def _segment_duration(segments: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, b - a) for a, b in segments))


def sample_from_segments(segments: list[tuple[float, float]], rng: np.random.Generator) -> float:
    durations = np.asarray([b - a for a, b in segments], dtype=np.float64)
    probs = durations / durations.sum()
    idx = int(rng.choice(len(segments), p=probs))
    a, b = segments[idx]
    return float(rng.uniform(a, b))


def liao_delay_ratio_samples() -> tuple[np.ndarray, np.ndarray, str]:
    module = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")
    cfg = module.LIAO_PRIOR_CONFIG["LIGO"]
    delays, ratios = module.extract_liao_delay_snr_pairs(cfg["image_csv"], cfg["snr_threshold"])
    delays = np.asarray(delays, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    keep = np.isfinite(delays) & (delays > 0.0) & np.isfinite(ratios) & (ratios >= 1.0)
    return delays[keep], ratios[keep], str(cfg["label"])


def exposure_conditioned_delay_ratio(
    live: list[tuple[float, float]],
    delays_days: np.ndarray,
    ratios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exposure = np.zeros(len(delays_days), dtype=np.float64)
    for idx, delay_days in enumerate(delays_days):
        exposure[idx] = _segment_duration(_shift_overlap_segments(live, float(delay_days) * SECONDS_PER_DAY))
    keep = exposure > 0.0
    if int(keep.sum()) < 20:
        raise RuntimeError("Too few GW-LMC delay samples remain after run exposure conditioning")
    return delays_days[keep], ratios[keep], exposure[keep] / exposure[keep].sum()


def calibrate_sky_model(source_run: Path) -> dict[str, float]:
    audit = pd.read_csv(source_run / "features" / "real_sky_map_audit.csv")
    events = pd.read_csv(source_run / "data" / "event_manifest.csv")
    primary = events[events["include_in_primary_search"] == True].copy()
    merged = audit.merge(primary[["event_name", "network_snr"]], on="event_name", how="inner")
    area = merged["map_area90_deg2"].to_numpy(dtype=np.float64)
    snr = merged["network_snr"].to_numpy(dtype=np.float64)
    keep = np.isfinite(area) & (area > 0.0) & np.isfinite(snr) & (snr >= SNR_MIN)
    area = area[keep]
    snr = snr[keep]
    if len(area) < 10:
        raise RuntimeError(f"Too few real sky maps to calibrate {source_run}")
    scaled = area * (snr / RHO_REF) ** 2
    a90_ref = float(np.median(scaled))
    residual = np.log(area / np.maximum(a90_ref * (RHO_REF / snr) ** 2, 1e-12))
    med = float(np.median(residual))
    sigma = float(1.4826 * np.median(np.abs(residual - med)))
    return {
        "a90_ref_deg2": a90_ref,
        "rho_ref": RHO_REF,
        "lognormal_sigma": max(0.15, sigma),
        "clip_min_deg2": float(max(1.0, np.quantile(area, 0.02))),
        "clip_max_deg2": float(np.quantile(area, 0.98)),
        "calibration_events": int(len(area)),
        "real_a90_median_deg2": float(np.median(area)),
        "real_a90_p90_deg2": float(np.quantile(area, 0.9)),
    }


def draw_truncated_snr(rng: np.random.Generator, minimum: float = SNR_MIN, maximum: float = SNR_MAX) -> float:
    minimum = float(minimum)
    maximum = float(max(maximum, minimum + 1e-6))
    # Detected Euclidean sources approximately follow p(rho) proportional to rho^-4.
    a = minimum ** -3
    b = maximum ** -3
    return float((a - rng.random() * (a - b)) ** (-1.0 / 3.0))


def draw_snr_pair(ratio: float, rng: np.random.Generator) -> tuple[float, float]:
    ratio = float(np.clip(ratio, 1.0, SNR_MAX / SNR_MIN))
    bright = draw_truncated_snr(rng, minimum=SNR_MIN * ratio, maximum=SNR_MAX)
    faint = bright / ratio
    return (bright, faint) if rng.random() < 0.5 else (faint, bright)


def inject_and_crop(clean: np.ndarray, noise: np.ndarray, target_snr: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    noise_full = zscore_channelwise(noise)
    signal = np.nan_to_num(np.asarray(clean, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    signal -= signal.mean(axis=-1, keepdims=True)
    norm = float(np.sqrt(np.sum(signal * signal)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Clean waveform has zero or invalid norm")
    signal = (signal * (float(target_snr) / norm)).astype(np.float32)
    mixed = noise_full + signal
    tail_fraction = float(np.sum(signal[..., -MODEL_TAIL:] ** 2) / max(np.sum(signal ** 2), 1e-30))
    return signal[..., -MODEL_TAIL:], mixed[..., -MODEL_TAIL:].astype(np.float32), {
        "target_network_snr": float(target_snr),
        "full_window_signal_l2": float(np.sqrt(np.sum(signal ** 2))),
        "model_tail_signal_l2": float(np.sqrt(np.sum(signal[..., -MODEL_TAIL:] ** 2))),
        "model_tail_snr2_fraction": tail_fraction,
        "mixed_tail_std": float(np.std(mixed[..., -MODEL_TAIL:])),
    }


def materialize_compact_dataset(
    seed_dir: Path,
    source_run: Path,
    deployment: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    marker = seed_dir / "data" / "real_noise_injections" / "compact_dataset_summary.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    segments = pd.read_parquet(source_run / "data" / "real_noise_injections" / "offsource_noise_segments.parquet")
    if samples > 600:
        raise ValueError("The fixed mainline SIS/PM clean waveform bank contains 600 systems per family")
    cache_key = str(source_run.resolve())
    cache = STRAIN_CACHES.setdefault(cache_key, LruStrainCache(source_run))
    root = seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    live_path = seed_dir / "data" / "real_noise_injections" / f"{deployment.lower()}_h1l1_live_segments.csv"
    live = joint_h1l1_live_segments(deployment, live_path)
    delay_days_raw, ratios_raw, delay_label = liao_delay_ratio_samples()
    delay_days, ratios, delay_prob = exposure_conditioned_delay_ratio(live, delay_days_raw, ratios_raw)
    exposure_delay_sample = rng.choice(delay_days, size=100_000, replace=True, p=delay_prob).astype(np.float32)
    np.save(seed_dir / "data" / "real_noise_injections" / "exposure_conditioned_delay_prior_days.npy", exposure_delay_sample)
    sky_cal = calibrate_sky_model(source_run)
    write_json(seed_dir / "data" / "real_noise_injections" / "sky_localization_calibration.json", sky_cal)
    metadata: list[dict[str, Any]] = []
    for family in FAMILIES:
        source = CLEAN_SOURCE / f"{family}_data_0222"
        clean1 = np.load(source / f"{family}_h_strain_1.npy", mmap_mode="r")[:samples]
        clean2 = np.load(source / f"{family}_h_strain_2.npy", mmap_mode="r")[:samples]
        out = root / f"{family}_data_0222"
        out.mkdir(parents=True, exist_ok=True)
        h1 = np.empty((samples, 2, MODEL_TAIL), dtype=np.float32)
        h2 = np.empty_like(h1)
        d1 = np.empty_like(h1)
        d2 = np.empty_like(h1)
        for idx in range(samples):
            prior_idx = int(rng.choice(len(delay_days), p=delay_prob))
            delay = float(delay_days[prior_idx])
            ratio = float(ratios[prior_idx])
            allowed = _shift_overlap_segments(live, delay * SECONDS_PER_DAY)
            gps1 = sample_from_segments(allowed, rng)
            gps2 = gps1 + delay * SECONDS_PER_DAY
            snr1, snr2 = draw_snr_pair(ratio, rng)
            ra_true = float(rng.uniform(0.0, 2.0 * math.pi))
            dec_true = float(math.asin(rng.uniform(-1.0, 1.0)))
            noise1, info1 = draw_full_noise(cache, segments, rng)
            noise2, info2 = draw_full_noise(cache, segments, rng)
            h1[idx], d1[idx], audit1 = inject_and_crop(clean1[idx], noise1, snr1)
            h2[idx], d2[idx], audit2 = inject_and_crop(clean2[idx], noise2, snr2)
            metadata.append(
                {
                    "family": family,
                    "sample_index": idx,
                    "gps_image1": gps1,
                    "gps_image2": gps2,
                    "delay_days": delay,
                    "snr_ratio_prior": ratio,
                    "target_snr_image1": snr1,
                    "target_snr_image2": snr2,
                    "ra_true": ra_true,
                    "dec_true": dec_true,
                    "noise_event_a": info1["noise_event"],
                    "noise_event_b": info2["noise_event"],
                    "noise_start_gps_a": info1["noise_start_gps"],
                    "noise_start_gps_b": info2["noise_start_gps"],
                    "tail_snr2_fraction_a": audit1["model_tail_snr2_fraction"],
                    "tail_snr2_fraction_b": audit2["model_tail_snr2_fraction"],
                    "mixed_tail_std_a": audit1["mixed_tail_std"],
                    "mixed_tail_std_b": audit2["mixed_tail_std"],
                }
            )
        np.save(out / f"{family}_h_strain_1.npy", h1)
        np.save(out / f"{family}_h_strain_2.npy", h2)
        np.save(out / f"{family}_data_strain_1.npy", d1)
        np.save(out / f"{family}_data_strain_2.npy", d2)
        fam_meta = [row for row in metadata if row["family"] == family]
        np.save(out / f"{family}_optimal_SNR_network_1.npy", np.asarray([x["target_snr_image1"] for x in fam_meta], dtype=np.float32))
        np.save(out / f"{family}_optimal_SNR_network_2.npy", np.asarray([x["target_snr_image2"] for x in fam_meta], dtype=np.float32))
        del h1, h2, d1, d2
        gc.collect()

    source = CLEAN_SOURCE / "Unlensed_data_0222" / "unlensed_h_strain.npy"
    clean = np.load(source, mmap_mode="r")[:samples]
    out = root / "Unlensed_data_0222"
    out.mkdir(parents=True, exist_ok=True)
    h = np.empty((samples, 2, MODEL_TAIL), dtype=np.float32)
    d = np.empty_like(h)
    for idx in range(samples):
        gps = sample_from_segments(live, rng)
        snr = draw_truncated_snr(rng)
        ra_true = float(rng.uniform(0.0, 2.0 * math.pi))
        dec_true = float(math.asin(rng.uniform(-1.0, 1.0)))
        noise, info = draw_full_noise(cache, segments, rng)
        h[idx], d[idx], audit = inject_and_crop(clean[idx], noise, snr)
        metadata.append(
            {
                "family": "unlensed",
                "sample_index": idx,
                "gps_image1": gps,
                "gps_image2": np.nan,
                "delay_days": np.nan,
                "snr_ratio_prior": np.nan,
                "target_snr_image1": snr,
                "target_snr_image2": np.nan,
                "ra_true": ra_true,
                "dec_true": dec_true,
                "noise_event_a": info["noise_event"],
                "noise_event_b": "",
                "noise_start_gps_a": info["noise_start_gps"],
                "noise_start_gps_b": np.nan,
                "tail_snr2_fraction_a": audit["model_tail_snr2_fraction"],
                "tail_snr2_fraction_b": np.nan,
                "mixed_tail_std_a": audit["mixed_tail_std"],
                "mixed_tail_std_b": np.nan,
            }
        )
    np.save(out / "unlensed_h_strain.npy", h)
    np.save(out / "unlensed_data_strain.npy", d)
    unlensed_meta = [row for row in metadata if row["family"] == "unlensed"]
    np.save(out / "unlensed_optimal_SNR_network.npy", np.asarray([x["target_snr_image1"] for x in unlensed_meta], dtype=np.float32))
    metadata_frame = pd.DataFrame(metadata)
    metadata_frame.to_parquet(
        seed_dir / "data" / "real_noise_injections" / "compact_injection_metadata.parquet", index=False
    )
    payload = {
        "protocol_version": "real_noise_injection_v2_run_matched_exposure_snr_sky",
        "deployment": deployment,
        "seed": int(seed),
        "samples_per_family": int(samples),
        "source_noise_run": str(source_run),
        "clean_signal_source": str(CLEAN_SOURCE),
        "clean_signal_policy": "Reuse the fixed mainline SIS/PM clean waveform bank so the ablation changes only real-noise/SNR/time/sky deployment conditions.",
        "full_injection_window_samples": FULL_WINDOW,
        "cached_model_tail_samples": MODEL_TAIL,
        "snr_calibration": "Full-window clean waveform L2 norm after channel demeaning is scaled to a target network SNR draw; this is an empirical whitened-SNR calibration, not a search-pipeline recovered SNR.",
        "snr_min": SNR_MIN,
        "snr_max": SNR_MAX,
        "delay_prior": delay_label,
        "delay_prior_raw_count": int(len(delay_days_raw)),
        "delay_prior_exposure_conditioned_count": int(len(delay_days)),
        "h1l1_live_segment_count": int(len(live)),
        "h1l1_live_time_days": _segment_duration(live) / SECONDS_PER_DAY,
        "sky_calibration": sky_cal,
        "tail_snr2_fraction_median": float(np.nanmedian(pd.concat([
            metadata_frame["tail_snr2_fraction_a"], metadata_frame["tail_snr2_fraction_b"]
        ], ignore_index=True))),
        "equivalence_note": "Injection is performed on the full 24 s window; only the exact 8192-sample tail consumed by the encoder is persisted.",
        "paired_noise_segments": int(len(segments)),
        "metadata_rows": int(len(metadata)),
        "known_limitations": [
            "The fixed clean SIS/PM bank is retained rather than regenerated from a new source-population posterior.",
            "Synthetic localization uses a run-calibrated Gaussian posterior surrogate, not BAYESTAR/PE for each injection.",
            "Off-source data exclude known-event +/-128 s windows but are not BayesWave-cleaned.",
        ],
    }
    write_json(marker, payload)
    return payload


def train_family(seed_dir: Path, family: str, seed: int, samples: int, epochs: int) -> dict[str, Any]:
    out = seed_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean"
    summary = out / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    cfg = MatchRunConfig(
        data_root=seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        model_type=family,
        data_mode="noisy",
        out_dir=out,
        backbone="inceptiontime",
        lensed_limit=samples,
        unlensed_limit=samples,
        epochs=epochs,
        batch_size=min(64, max(8, int(round(1.4 * samples)))),
        eval_batch_size=256,
        target_len=8192,
        stride=2,
        preprocess="bandpass",
        bandpass_low=40,
        bandpass_high=580,
        use_pure_aux=True,
        aug_flip=True,
        aug_roll=16,
        aug_scale=0.1,
        aug_noise=0.02,
        candidate_topk=10,
        seed=seed + (0 if family == "SIS" else 1000),
        num_workers=0,
        pin_memory=False,
        amp=False,
        export_candidates=False,
    )
    seed_everything(cfg.seed)
    return run_train_eval(cfg, cpu=False)


def synthetic_observables_from_metadata(
    ds_meta: list[dict[str, Any]],
    family: str,
    metadata: pd.DataFrame,
    sky_cal: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + (0 if family == "SIS" else 100_000))
    family_rows = metadata[metadata["family"] == family].set_index("sample_index")
    unlensed_rows = metadata[metadata["family"] == "unlensed"].set_index("sample_index")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(ds_meta):
        source_index = int(item["source_index"])
        tag = str(item["tag"])
        source = unlensed_rows.loc[source_index] if tag == "U" else family_rows.loc[source_index]
        image = 2 if tag == "L2" else 1
        snr = float(source[f"target_snr_image{image}"])
        gps = float(source[f"gps_image{image}"])
        a90 = float(
            sky_cal["a90_ref_deg2"]
            * (sky_cal["rho_ref"] / max(snr, 1.0)) ** 2
            * rng.lognormal(0.0, sky_cal["lognormal_sigma"])
        )
        a90 = float(np.clip(a90, sky_cal["clip_min_deg2"], sky_cal["clip_max_deg2"]))
        sigma = float(a90_to_sigma_rad(np.asarray([a90]))[0])
        ra_true = float(source["ra_true"])
        dec_true = float(source["dec_true"])
        ra_obs, dec_obs = sample_observed_sky_center(
            np.asarray([ra_true]), np.asarray([dec_true]), np.asarray([sigma]), rng,
            sampling="tangent_2d_gaussian",
        )
        rows.append(
            {
                "idx": idx,
                "pair_id": int(item["pair_id"]),
                "tag": tag,
                "source_index": source_index,
                "gps_obs": gps,
                "snr": snr,
                "ra_true": ra_true,
                "dec_true": dec_true,
                "ra_obs": float(ra_obs[0]),
                "dec_obs": float(dec_obs[0]),
                "sky_area90_deg2": a90,
                "sky_sigma_rad": sigma,
            }
        )
    return pd.DataFrame(rows)


def build_split_pair_table(
    deploy,
    seed_dir: Path,
    family: str,
    epochs: int,
    samples: int,
    split: str,
    sky_cal: dict[str, float],
    observable_seed: int,
) -> pd.DataFrame:
    checkpoint = seed_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "model.pt"
    model, cfg = deploy.load_model_from_checkpoint(
        checkpoint,
        seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        samples,
        False,
    )
    cfg.model_type = family
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits["lensed"][split], splits["unlensed"][split], cfg)
    embeddings = embed_eval(model, ds, cfg, cpu=False)
    waveform = similarity_matrix(embeddings)
    gt = ground_truth_partner(ds.meta)
    metadata = pd.read_parquet(seed_dir / "data" / "real_noise_injections" / "compact_injection_metadata.parquet")
    events = synthetic_observables_from_metadata(ds.meta, family, metadata, sky_cal, observable_seed)
    events.to_parquet(seed_dir / "results" / f"{family.lower()}_{split}_synthetic_event_observables.parquet", index=False)
    ii, jj = np.triu_indices(len(events), k=1)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    delta_days = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    sky_score, theta, norm_sep = gaussian_log_cosine_overlap_pairs(
        events["ra_obs"].to_numpy(dtype=np.float64)[ii],
        events["dec_obs"].to_numpy(dtype=np.float64)[ii],
        events["sky_sigma_rad"].to_numpy(dtype=np.float64)[ii],
        events["ra_obs"].to_numpy(dtype=np.float64)[jj],
        events["dec_obs"].to_numpy(dtype=np.float64)[jj],
        events["sky_sigma_rad"].to_numpy(dtype=np.float64)[jj],
    )
    return pd.DataFrame(
        {
            "family": family,
            "split": split,
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "is_true_pair": (gt[ii] == jj).astype(np.int8),
            "waveform_score": waveform[ii, jj].astype(np.float32),
            "time_score": np.zeros(len(ii), dtype=np.float32),
            "sky_score": sky_score.astype(np.float32),
            "delta_t_days": delta_days,
            "sky_ang_sep_rad": theta.astype(np.float32),
            "sky_norm_sep": norm_sep.astype(np.float32),
            "event_count": int(len(events)),
        }
    )


def fit_time_lr(validation: pd.DataFrame, signal_prior_days: np.ndarray, n_bins: int = 80) -> dict[str, Any]:
    background = validation.loc[validation["is_true_pair"] == 0, "delta_t_days"].to_numpy(dtype=np.float64)
    signal = np.asarray(signal_prior_days, dtype=np.float64)
    signal = signal[np.isfinite(signal) & (signal > 0.0)]
    background = background[np.isfinite(background) & (background > 0.0)]
    merged = np.concatenate([signal, background])
    lo = max(1e-6, float(np.min(merged)))
    hi = max(float(np.max(merged)), lo * 1.01)
    edges = np.linspace(math.log10(lo), math.log10(hi) + 1e-9, n_bins + 1)
    hs, _ = np.histogram(np.log10(signal), bins=edges)
    hb, _ = np.histogram(np.log10(background), bins=edges)
    ps = (hs + 1.0) / (hs.sum() + len(hs))
    pb = (hb + 1.0) / (hb.sum() + len(hb))
    return {
        "log10_edges": edges,
        "log_lr": np.log(ps) - np.log(pb),
        "signal_count": int(len(signal)),
        "validation_background_count": int(len(background)),
        "pseudocount": 1.0,
    }


def apply_time_lr(frame: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    values = np.log10(np.maximum(out["delta_t_days"].to_numpy(dtype=np.float64), 1e-12))
    edges = np.asarray(calibration["log10_edges"], dtype=np.float64)
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    out["time_score"] = np.asarray(calibration["log_lr"], dtype=np.float64)[index].astype(np.float32)
    return out


def query_ranks_from_pairs(deploy, pairs: pd.DataFrame, weights: dict[str, float], deployment: str, seed: int, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, frame in pairs.groupby("family", sort=True):
        n = int(frame["event_count"].iloc[0])
        label = np.zeros((n, n), dtype=bool)
        ii = frame["idx_i"].to_numpy(dtype=np.int32)
        jj = frame["idx_j"].to_numpy(dtype=np.int32)
        truth = frame["is_true_pair"].to_numpy(dtype=bool)
        label[ii, jj] = truth
        label[jj, ii] = truth
        score = np.zeros((n, n), dtype=np.float32)
        for channel in ("waveform", "time", "sky"):
            score += float(weights.get(channel, 0.0)) * deploy.row_z_neutral(
                deploy.matrix_from_pairs(frame, n, f"{channel}_score")
            )
        np.fill_diagonal(score, -np.inf)
        for query in np.where(label.any(axis=1))[0]:
            partner = int(np.flatnonzero(label[query])[0])
            rank = int(1 + np.sum(score[query] > score[query, partner]))
            system = f"{family}:{min(query, partner)}-{max(query, partner)}"
            rows.append(
                {
                    "deployment": deployment,
                    "seed": int(seed),
                    "method": method,
                    "query_index": int(query),
                    "partner_index": partner,
                    "family": str(family).upper(),
                    "system_id": system,
                    "query_tag": "query",
                    "query_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def select_weights_v2(
    deploy,
    validation: pd.DataFrame,
    require_all_positive: bool = False,
    force_waveform_zero: bool = False,
    require_time_sky_positive: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for ww, wt, ws in itertools.product(WEIGHT_GRID, repeat=3):
        if ww == wt == ws == 0.0:
            continue
        if force_waveform_zero and ww != 0.0:
            continue
        if require_all_positive and min(ww, wt, ws) <= 0.0:
            continue
        if require_time_sky_positive and min(wt, ws) <= 0.0:
            continue
        weights = {"waveform": float(ww), "time": float(wt), "sky": float(ws)}
        metrics = deploy.validation_metrics_for_weights(validation, weights)
        rows.append({**weights, **metrics, "weight_l2": float(math.sqrt(ww * ww + wt * wt + ws * ws))})
    grid = pd.DataFrame(rows).sort_values(
        ["macro_r_at_10", "macro_r_at_5", "macro_r_at_1", "weight_l2"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if grid.empty:
        raise RuntimeError("Weight grid is empty")
    best = grid.iloc[0]
    return {name: float(best[name]) for name in ("waveform", "time", "sky")}, grid


def score_matrix_for_weights(deploy, frame: pd.DataFrame, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    n = int(frame["event_count"].iloc[0])
    score = np.zeros((n, n), dtype=np.float32)
    for channel in ("waveform", "time", "sky"):
        score += float(weights.get(channel, 0.0)) * deploy.row_z_neutral(
            deploy.matrix_from_pairs(frame, n, f"{channel}_score")
        )
    label = np.zeros((n, n), dtype=bool)
    ii = frame["idx_i"].to_numpy(dtype=np.int32)
    jj = frame["idx_j"].to_numpy(dtype=np.int32)
    truth = frame["is_true_pair"].to_numpy(dtype=bool)
    label[ii, jj] = truth
    label[jj, ii] = truth
    np.fill_diagonal(score, -np.inf)
    return score, label


def pair_level_metrics(deploy, pairs: pd.DataFrame, methods: dict[str, dict[str, float]], deployment: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, weights in methods.items():
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for _, frame in pairs.groupby("family", sort=True):
            matrix, label = score_matrix_for_weights(deploy, frame, weights)
            ii, jj = np.triu_indices(len(matrix), k=1)
            all_scores.append(np.maximum(matrix[ii, jj], matrix[jj, ii]))
            all_labels.append(label[ii, jj].astype(np.int8))
        score = np.concatenate(all_scores)
        label = np.concatenate(all_labels)
        order = np.argsort(-score, kind="stable")
        sorted_label = label[order]
        tp = np.cumsum(sorted_label)
        fp = np.cumsum(1 - sorted_label)
        n_true = int(label.sum())
        n_false = int(len(label) - n_true)
        base = n_true / len(label)
        rows.append({
            "deployment": deployment,
            "seed": int(seed),
            "method": method,
            "metric": "average_precision",
            "value": float(average_precision_score(label, score)),
            "n_true_pairs": n_true,
            "n_false_pairs": n_false,
            "base_positive_rate": base,
        })
        for target_recall in (0.5, 0.9):
            idx = int(np.searchsorted(tp / max(n_true, 1), target_recall, side="left"))
            idx = min(idx, len(tp) - 1)
            precision = float(tp[idx] / max(tp[idx] + fp[idx], 1))
            rows.append({
                "deployment": deployment,
                "seed": int(seed),
                "method": method,
                "metric": f"at_recall_{target_recall:g}",
                "value": precision,
                "recall": float(tp[idx] / max(n_true, 1)),
                "precision": precision,
                "true_pairs": int(tp[idx]),
                "false_pairs": int(fp[idx]),
                "n_true_pairs": n_true,
                "n_false_pairs": n_false,
                "base_positive_rate": base,
            })
        false_budget = max(1, int(math.floor(1e-5 * n_false)))
        valid = np.flatnonzero(fp <= false_budget)
        idx = int(valid[-1]) if len(valid) else 0
        rows.append({
            "deployment": deployment,
            "seed": int(seed),
            "method": method,
            "metric": "at_false_pair_rate_1e-5",
            "value": float(tp[idx] / max(n_true, 1)),
            "recall": float(tp[idx] / max(n_true, 1)),
            "precision": float(tp[idx] / max(tp[idx] + fp[idx], 1)),
            "true_pairs": int(tp[idx]),
            "false_pairs": int(fp[idx]),
            "n_true_pairs": n_true,
            "n_false_pairs": n_false,
            "base_positive_rate": base,
        })
    return pd.DataFrame(rows)


def embed_real_events(seed_dir: Path, epochs: int) -> None:
    summary = seed_dir / "features" / "real_waveform_embedding_summary.json"
    if summary.exists():
        return
    embed = load_module(f"embed_real_{seed_dir.name}", REPO_ROOT / "scripts" / "real_search" / "06_embed_real_gwtc_events.py")
    old_argv = sys.argv
    try:
        sys.argv = ["06_embed_real_gwtc_events.py", "--run-dir", str(seed_dir), "--epochs", str(epochs)]
        embed.main()
    finally:
        sys.argv = old_argv


def real_catalog_scores(
    deploy,
    unified,
    seed_dir: Path,
    source_run: Path,
    methods: dict[str, dict[str, float]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    observable = pd.read_parquet(source_run / "features" / "real_pair_observable_features.parquet")
    waveform = pd.read_parquet(seed_dir / "features" / "real_waveform_similarity.parquet")
    primary = unified.primary_manifest(seed_dir)
    real = observable.merge(
        waveform[["idx_i", "idx_j", "waveform_available", "waveform_score", "waveform_score_sis", "waveform_score_pm"]],
        on=["idx_i", "idx_j"],
        how="left",
    )
    object_class = (
        primary["object_class"].fillna("unknown").astype(str).to_numpy()
        if "object_class" in primary.columns
        else np.asarray(["BBH"] * len(primary), dtype=object)
    )
    event_ood = (
        primary["is_ood_for_bbh_encoder"].fillna(False).astype(bool).to_numpy()
        if "is_ood_for_bbh_encoder" in primary.columns
        else np.zeros(len(primary), dtype=bool)
    )
    ii = real["idx_i"].to_numpy(dtype=np.int32)
    jj = real["idx_j"].to_numpy(dtype=np.int32)
    real["object_class_i"] = object_class[ii]
    real["object_class_j"] = object_class[jj]
    real["pair_has_ood"] = event_ood[ii] | event_ood[jj]
    real["waveform_available"] = real["waveform_available"].fillna(False).astype(bool)
    real["full_waveform_scored"] = real["waveform_available"] & (~real["pair_has_ood"])
    real["raw_waveform_score"] = real["waveform_score"]
    real.loc[~real["full_waveform_scored"], "waveform_score"] = np.nan
    outputs: dict[str, pd.DataFrame] = {}
    for method, method_weights in methods.items():
        scored = deploy.add_real_scores(real, method_weights, method)
        scored.to_parquet(seed_dir / "results" / f"real_pair_scores_{method}.parquet", index=False)
        deploy.shortlist(scored, 100).to_csv(seed_dir / "results" / f"candidate_shortlist_{method}.csv", index=False)
        outputs[method] = scored
    wf = real.loc[real["full_waveform_scored"], "raw_waveform_score"].dropna()
    audit = {
        "n_events": int(len(primary)),
        "n_pairs": int(len(real)),
        "n_full_waveform_scored_pairs": int(real["full_waveform_scored"].sum()),
        "waveform_score_min": float(wf.min()) if len(wf) else None,
        "waveform_score_max": float(wf.max()) if len(wf) else None,
        "waveform_score_std": float(wf.std()) if len(wf) else None,
        "waveform_score_unique": int(wf.nunique()) if len(wf) else 0,
    }
    return outputs, audit


def run_seed(
    deployment: str,
    source_run: Path,
    seed: int,
    out_root: Path,
    samples: int,
    epochs: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    seed_dir = out_root / deployment.lower() / f"seed_{seed}"
    complete = seed_dir / "seed_summary.json"
    if complete.exists():
        print(f"{deployment} seed {seed}: complete, reusing", flush=True)
        return json.loads(complete.read_text(encoding="utf-8"))
    started = time.perf_counter()
    prepare_seed_layout(seed_dir, source_run)
    dataset = materialize_compact_dataset(seed_dir, source_run, deployment, seed + 100, samples)
    split_sets: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        split_cfg = MatchRunConfig(
            data_root=seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
            model_type=family,
            data_mode="noisy",
            lensed_limit=samples,
            unlensed_limit=samples,
            seed=seed + (0 if family == "SIS" else 1000),
        )
        parts = split_indices(samples, samples, split_cfg)
        split_sets[f"{family}_lensed"] = parts["lensed"]
        split_sets[f"{family}_unlensed"] = parts["unlensed"]
    split_audit = split_integrity_audit(split_sets)
    if not split_audit["all_disjoint"]:
        raise RuntimeError(f"{deployment} train/validation/test system split leakage detected")
    write_json(seed_dir / "results" / "split_integrity_audit.json", split_audit)
    for family in FAMILIES:
        print(f"{deployment} seed {seed}: train {family}", flush=True)
        train_family(seed_dir, family, seed, samples, epochs)
    deploy = load_module(
        f"deploy_uncertainty_{deployment}_{seed}",
        REPO_ROOT / "scripts" / "real_search" / "15_gwtc34_real_deployment.py",
    )
    unified = load_module(
        f"unified_uncertainty_{deployment}_{seed}",
        REPO_ROOT / "scripts" / "experiments" / "102_unified_sky_formal_pipeline.py",
    )
    sky_cal = json.loads(
        (seed_dir / "data" / "real_noise_injections" / "sky_localization_calibration.json").read_text(encoding="utf-8")
    )
    val = pd.concat(
        [
            build_split_pair_table(deploy, seed_dir, family, epochs, samples, "val", sky_cal, seed + 2000)
            for family in FAMILIES
        ],
        ignore_index=True,
    )
    test = pd.concat(
        [
            build_split_pair_table(deploy, seed_dir, family, epochs, samples, "test", sky_cal, seed + 3000)
            for family in FAMILIES
        ],
        ignore_index=True,
    )
    signal_prior = np.load(seed_dir / "data" / "real_noise_injections" / "exposure_conditioned_delay_prior_days.npy")
    time_calibration = fit_time_lr(val, signal_prior)
    val = apply_time_lr(val, time_calibration)
    test = apply_time_lr(test, time_calibration)
    write_json(
        seed_dir / "results" / "time_delay_lr_calibration.json",
        {
            "fit_split": "validation false pairs only",
            "signal_prior": "GW-LMC LIGO delay prior conditioned on actual H1-L1 live-time autocorrelation",
            "signal_count": time_calibration["signal_count"],
            "validation_background_count": time_calibration["validation_background_count"],
            "pseudocount": time_calibration["pseudocount"],
            "log10_edges": np.asarray(time_calibration["log10_edges"]).tolist(),
            "log_lr": np.asarray(time_calibration["log_lr"]).tolist(),
        },
    )
    val.to_parquet(seed_dir / "results" / "fusion_validation_pairs.parquet", index=False)
    test.to_parquet(seed_dir / "results" / "fusion_heldout_test_pairs.parquet", index=False)
    unconstrained_weights, grid = select_weights_v2(deploy, val, require_all_positive=False)
    positive_weights, positive_grid = select_weights_v2(deploy, val, require_all_positive=True)
    baseline_weights, baseline_grid = select_weights_v2(
        deploy, val, force_waveform_zero=True, require_time_sky_positive=True
    )
    equal_weights = {"waveform": 1.0, "time": 1.0, "sky": 1.0}
    grid.to_csv(seed_dir / "results" / "fusion_weight_grid_unconstrained.csv", index=False)
    positive_grid.to_csv(seed_dir / "results" / "fusion_weight_grid_strictly_positive.csv", index=False)
    baseline_grid.to_csv(seed_dir / "results" / "fusion_weight_grid_time_sky.csv", index=False)
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "time_sky_positive": baseline_weights,
        "waveform_time_sky_unconstrained": unconstrained_weights,
        "waveform_time_sky_positive": positive_weights,
        "waveform_time_sky_equal": equal_weights,
    }
    query = pd.concat(
        [query_ranks_from_pairs(deploy, test, method_weights, deployment, seed, method) for method, method_weights in methods.items()],
        ignore_index=True,
    )
    query.to_parquet(seed_dir / "results" / "heldout_test_query_ranks.parquet", index=False)
    point = metric_rows(query)
    point.to_csv(seed_dir / "results" / "heldout_test_retrieval_metrics.csv", index=False)
    waveform_rows = point[point["method"] == "waveform_only"].set_index("subset")
    family_r10 = [float(waveform_rows.loc[family, "r_at_10"]) for family in FAMILIES]
    gate = {
        "metric_split": "held-out test",
        "waveform_only_r_at_1": float(waveform_rows.loc["overall", "r_at_1"]),
        "waveform_only_r_at_10": float(waveform_rows.loc["overall", "r_at_10"]),
        "sis_r_at_10": family_r10[0],
        "pm_r_at_10": family_r10[1],
        "macro_family_r_at_10": float(np.mean(family_r10)),
        "min_family_r_at_10": float(np.min(family_r10)),
        "pass_rule": "macro_family_r_at_10 > 0.6 and min_family_r_at_10 > 0.5",
        "passed": bool(np.mean(family_r10) > 0.6 and np.min(family_r10) > 0.5),
    }
    write_json(seed_dir / "results" / "waveform_gate1_metrics.json", gate)
    pair_metrics = pair_level_metrics(deploy, test, methods, deployment, seed)
    pair_metrics.to_csv(seed_dir / "results" / "heldout_test_pair_level_metrics.csv", index=False)
    bootstrap = bootstrap_system_ci(query, draws=bootstrap_draws, seed=seed + 4000)
    bootstrap.to_csv(seed_dir / "results" / "heldout_test_bootstrap_95ci.csv", index=False)
    write_json(
        seed_dir / "results" / "selected_weights.json",
        {
            "selection_split": "validation",
            "evaluation_split": "held-out test",
            "unconstrained_waveform_time_sky": unconstrained_weights,
            "strictly_positive_waveform_time_sky": positive_weights,
            "equal_waveform_time_sky": equal_weights,
            "strictly_positive_time_sky": baseline_weights,
            "primary_policy": "strictly_positive_waveform_time_sky; all three weights were constrained >0 before held-out test evaluation",
            "unconstrained_validation_best": grid.iloc[0].to_dict(),
            "strictly_positive_validation_best": positive_grid.iloc[0].to_dict(),
            "heldout_test_metrics": point.to_dict(orient="records"),
        },
    )

    embed_real_events(seed_dir, epochs)
    real_methods = {
        "waveform_time_sky_positive": positive_weights,
        "waveform_time_sky_unconstrained": unconstrained_weights,
        "waveform_time_sky_equal": equal_weights,
        "time_sky_positive": baseline_weights,
    }
    real_outputs, waveform_audit = real_catalog_scores(deploy, unified, seed_dir, source_run, real_methods)
    selected = real_outputs["waveform_time_sky_positive"]
    top = selected.iloc[0]
    historical = {
        method: unified.safe_rank(scores, "GW170104", "GW170814")
        for method, scores in real_outputs.items()
    } if deployment == "GWTC3" else {method: {"found": False} for method in real_outputs}
    summary = {
        "deployment": deployment,
        "seed": int(seed),
        "status": "complete",
        "samples_per_family": int(samples),
        "epochs": int(epochs),
        "noise_realization_seed": int(seed + 100),
        "validation_observable_seed": int(seed + 2000),
        "test_observable_seed": int(seed + 3000),
        "selected_weights_primary_positive": positive_weights,
        "selected_weights_unconstrained": unconstrained_weights,
        "equal_weights": equal_weights,
        "time_sky_weights": baseline_weights,
        "validation_metrics_unconstrained": grid.iloc[0].to_dict(),
        "validation_metrics_positive": positive_grid.iloc[0].to_dict(),
        "heldout_test_metrics": point.to_dict(orient="records"),
        "waveform_gate1": gate,
        "heldout_test_pair_level_metrics": pair_metrics.to_dict(orient="records"),
        "dataset": dataset,
        "real_waveform_audit": waveform_audit,
        "top_pair": {
            "event_i": str(top.event_i),
            "event_j": str(top.event_j),
            "rank": int(top["rank"]),
            "final_score": float(top.final_score),
            "waveform_available": bool(top.waveform_available),
            "full_waveform_scored": bool(top.full_waveform_scored),
            "pair_has_ood": bool(top.pair_has_ood),
        },
        "gw170104_gw170814_by_method": historical,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(complete, summary)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_deployment(out_root: Path, deployment: str, seeds: list[int]) -> None:
    root = out_root / deployment.lower()
    query = pd.concat(
        [pd.read_parquet(root / f"seed_{seed}" / "results" / "heldout_test_query_ranks.parquet") for seed in seeds],
        ignore_index=True,
    )
    query.to_parquet(root / "heldout_test_query_ranks_all_seeds.parquet", index=False)
    per_seed = metric_rows(query)
    per_seed.to_csv(root / "heldout_test_retrieval_metrics_per_seed.csv", index=False)
    across_seed_summary(per_seed).to_csv(root / "heldout_test_retrieval_metrics_across_seed_summary.csv", index=False)
    boot = pd.concat(
        [pd.read_csv(root / f"seed_{seed}" / "results" / "heldout_test_bootstrap_95ci.csv") for seed in seeds],
        ignore_index=True,
    )
    boot.to_csv(root / "heldout_test_bootstrap_95ci_per_seed.csv", index=False)
    pair_metrics = pd.concat(
        [pd.read_csv(root / f"seed_{seed}" / "results" / "heldout_test_pair_level_metrics.csv") for seed in seeds],
        ignore_index=True,
    )
    pair_metrics.to_csv(root / "heldout_test_pair_level_metrics_per_seed.csv", index=False)
    pair_summary = (
        pair_metrics.groupby(["deployment", "method", "metric"], as_index=False)
        .agg(value_mean=("value", "mean"), value_std=("value", "std"), seeds=("seed", "nunique"))
    )
    pair_summary.to_csv(root / "heldout_test_pair_level_metrics_across_seed_summary.csv", index=False)

    weight_rows = []
    for seed in seeds:
        payload = json.loads((root / f"seed_{seed}" / "results" / "selected_weights.json").read_text(encoding="utf-8"))
        for method in (
            "unconstrained_waveform_time_sky",
            "strictly_positive_waveform_time_sky",
            "equal_waveform_time_sky",
            "strictly_positive_time_sky",
        ):
            weight_rows.append({"deployment": deployment, "seed": int(seed), "method": method, **payload[method]})
    pd.DataFrame(weight_rows).to_csv(root / "fusion_weights_per_seed.csv", index=False)

    ranking = []
    for seed in seeds:
        for method in (
            "waveform_time_sky_positive",
            "waveform_time_sky_unconstrained",
            "waveform_time_sky_equal",
            "time_sky_positive",
        ):
            scores = pd.read_parquet(root / f"seed_{seed}" / "results" / f"real_pair_scores_{method}.parquet")
            scores = scores[["event_i", "event_j", "rank", "final_score", "waveform_available", "full_waveform_scored", "pair_has_ood"]].copy()
            scores["seed"] = int(seed)
            scores["method"] = method
            scores["pair_key"] = scores.apply(lambda r: "--".join(sorted((str(r.event_i), str(r.event_j)))), axis=1)
            ranking.append(scores)
    all_ranks = pd.concat(ranking, ignore_index=True)
    all_ranks.to_parquet(root / "real_candidate_ranks_all_seeds.parquet", index=False)
    stability = (
        all_ranks.groupby(["method", "pair_key"], as_index=False)
        .agg(
            event_i=("event_i", "first"),
            event_j=("event_j", "first"),
            median_rank=("rank", "median"),
            q25_rank=("rank", lambda x: float(np.quantile(x, 0.25))),
            q75_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            top10_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 10))),
            top20_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 20))),
            full_waveform_scored=("full_waveform_scored", "all"),
            pair_has_ood=("pair_has_ood", "any"),
        )
        .sort_values(["method", "median_rank", "q75_rank", "min_rank"])
        .reset_index(drop=True)
    )
    stability.to_csv(root / "real_candidate_rank_stability.csv", index=False)
    write_json(
        root / "uncertainty_summary.json",
        {
            "deployment": deployment,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "fusion_weight_selection": "validation-only for each seed",
            "split_policy": "independent source-system split per seed; both images of each injected doublet remain in the same split",
            "reported_retrieval_split": "held-out real-noise synthetic injection test",
            "real_catalog_interpretation": "rank stability only; no confirmed real positive pairs",
            "most_stable_positive_three_channel_candidates": stability[
                stability["method"] == "waveform_time_sky_positive"
            ].head(20).to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--deployments", nargs="+", choices=["GWTC3", "GWTC4"], default=["GWTC3", "GWTC4"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    if not args.materialize_only and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; refusing to launch the formal O3/O4 multi-seed retraining on CPU")
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_config.json",
        {
            "deployments": args.deployments,
            "seeds": args.seeds,
            "samples_per_family": args.samples_per_family,
            "epochs": args.epochs,
            "bootstrap_draws": args.bootstrap_draws,
            "gwtc3_source": O3_SOURCE,
            "gwtc4_source": O4_SOURCE,
            "main_channels": ["waveform", "time-delay", "unified posterior-overlap sky"],
            "snr_policy": "audit-only",
            "primary_fusion_policy": "validation-selected strictly positive waveform/time/sky weights",
            "comparison_fusion_policies": ["unconstrained validation selection", "equal weights", "time+sky positive baseline"],
            "unordered_pair_score": "max(directed_i_to_j, directed_j_to_i)",
        },
    )
    sources = {"GWTC3": O3_SOURCE, "GWTC4": O4_SOURCE}
    if args.materialize_only:
        for deployment in args.deployments:
            for seed in args.seeds:
                seed_dir = args.out_root / deployment.lower() / f"seed_{int(seed)}"
                prepare_seed_layout(seed_dir, sources[deployment])
                payload = materialize_compact_dataset(
                    seed_dir, sources[deployment], deployment, int(seed) + 100, args.samples_per_family
                )
                print(json.dumps({"deployment": deployment, **payload}, indent=2), flush=True)
        return
    for deployment in args.deployments:
        completed: list[int] = []
        for seed in args.seeds:
            run_seed(
                deployment,
                sources[deployment],
                int(seed),
                args.out_root,
                args.samples_per_family,
                args.epochs,
                args.bootstrap_draws,
            )
            completed.append(int(seed))
            aggregate_deployment(args.out_root, deployment, completed)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
