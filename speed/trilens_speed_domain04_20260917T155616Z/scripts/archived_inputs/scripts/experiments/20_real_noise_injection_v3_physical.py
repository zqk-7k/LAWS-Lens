from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import pandas as pd
import torch
from gwosc.datasets import run_segment
from gwosc.timeline import get_segments
from sklearn.metrics import average_precision_score


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from matchgw.config import MatchRunConfig  # noqa: E402
from matchgw.data import (  # noqa: E402
    EvaluationSet,
    ground_truth_partner,
    load_match_arrays,
    peak_flip_channels,
    split_indices,
    zscore_channels,
)
from matchgw.matching import similarity_matrix  # noqa: E402
from matchgw.pipeline import build_model, embed_eval, run_train_eval  # noqa: E402
from scripts.real_search.common import read_probability_map  # noqa: E402
from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    across_seed_summary,
    bootstrap_system_ci,
    metric_rows,
    seed_everything,
    split_integrity_audit,
)
from scripts.real_search.physical_common import (  # noqa: E402
    MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    PAD_SECONDS,
    PSD_SECONDS,
    RAW_MODEL_SAMPLES,
    RAW_PADDED_SAMPLES,
    RAW_SAMPLE_RATE,
    SECONDS_PER_DAY,
    LiveSegment,
    apply_score_likelihood_ratio,
    apply_time_likelihood_ratio,
    draw_detected_lens_delays,
    draw_null_delays,
    effective_rank,
    embed_signal_in_padded_window,
    estimate_psd,
    fit_score_likelihood_ratio,
    fit_time_likelihood_ratio,
    in_live_segments,
    optimal_network_snr,
    posterior_area90,
    preprocess_24s,
    rotate_probability_map_to_true_position,
    sample_live_times,
    scale_to_network_snr,
    sky_log_bayes_factor_from_maps,
    write_json,
)


O3_SOURCE = REPO / "runs" / "real_gwtc_lensing_search_20260625"
O4_SOURCE = REPO / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"
CLEAN_SOURCE = O3_SOURCE / "data" / "real_noise_injections" / "matchroots" / "LIGO"
DEFAULT_OUT = REPO / "results" / "real_noise_injection_v3_physical_20260721"
DEFAULT_SEEDS = (202607211, 202607212, 202607213)
FAMILIES = ("SIS", "PM")
DETECTORS = ("H1", "L1")
SNR_MIN = 8.0
SNR_MAX = 60.0
WEIGHT_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SKY_NSIDE = 64
NOISE_BANK_SIZE = 64
NOISE_REFERENCE_SAMPLES = int(PSD_SECONDS * RAW_SAMPLE_RATE)
RUNS = {
    "GWTC3": ("O1", "O2", "O3a", "O3b"),
    "GWTC4": ("O4a",),
}
SOURCES = {"GWTC3": O3_SOURCE, "GWTC4": O4_SOURCE}


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
    for rel in ("data/event_manifest.csv", "data/strain_gwosc_download_manifest.csv"):
        ensure_link(seed_dir / rel, source_run / rel)
    ensure_link(seed_dir / "data" / "real_strain", source_run / "data" / "real_strain")
    ensure_link(
        seed_dir / "features" / "source_real_sky_overlap.parquet",
        source_run / "features" / "real_sky_overlap.parquet",
    )
    for name in ("data", "features", "results", "logs", "waveform_gate"):
        (seed_dir / name).mkdir(parents=True, exist_ok=True)


def primary_manifest(source_run: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_run / "data" / "event_manifest.csv")
    mask = frame["include_in_primary_search"].fillna(False).astype(bool)
    mask &= frame["sky_map_available"].fillna(False).astype(bool)
    return frame.loc[mask].sort_values("gps_time").reset_index(drop=True)


def infer_run(gps: float) -> str:
    for name in ("O1", "O2", "O3a", "O3b", "O4a"):
        start, end = run_segment(name)
        if float(start) <= gps <= float(end):
            return name
    return "outside_run"


def _intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((float(lo), float(hi)))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def build_live_schedule(deployment: str, source_run: Path, shared_dir: Path) -> list[LiveSegment]:
    path = shared_dir / "h1l1_live_schedule.csv"
    if path.exists():
        frame = pd.read_csv(path)
        return [LiveSegment(float(r.start_gps), float(r.end_gps), str(r.run), float(r.weight_per_second)) for r in frame.itertuples()]

    manifest = primary_manifest(source_run)
    counts: dict[str, int] = {}
    for gps in manifest["gps_time"].to_numpy(dtype=float):
        name = infer_run(float(gps))
        counts[name] = counts.get(name, 0) + 1

    raw_by_run: dict[str, list[tuple[float, float]]] = {}
    duration_by_run: dict[str, float] = {}
    for name in RUNS[deployment]:
        start, end = run_segment(name)
        h1 = sorted((float(a), float(b)) for a, b in get_segments("H1_DATA", int(start), int(end)))
        l1 = sorted((float(a), float(b)) for a, b in get_segments("L1_DATA", int(start), int(end)))
        joint = [(a + 32.0, b - 32.0) for a, b in _intersect(h1, l1) if b - a > 64.0]
        raw_by_run[name] = joint
        duration_by_run[name] = float(sum(b - a for a, b in joint))

    rows = []
    result: list[LiveSegment] = []
    for name, segments in raw_by_run.items():
        rate = max(counts.get(name, 0), 1) / max(duration_by_run[name], 1.0)
        for start, end in segments:
            result.append(LiveSegment(start, end, name, rate))
            rows.append(
                {
                    "run": name,
                    "start_gps": start,
                    "end_gps": end,
                    "duration_s": end - start,
                    "primary_events_in_run": counts.get(name, 0),
                    "joint_live_duration_s": duration_by_run[name],
                    "weight_per_second": rate,
                }
            )
    shared_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return result


class HdfCache:
    def __init__(self, source_run: Path, max_files: int = 8):
        self.source_run = source_run
        self.max_files = int(max_files)
        self.cache: dict[str, tuple[np.ndarray, float, float]] = {}
        self.order: list[str] = []

    def get(self, rel: str) -> tuple[np.ndarray, float, float]:
        if rel in self.cache:
            self.order.remove(rel)
            self.order.append(rel)
            return self.cache[rel]
        path = self.source_run / rel
        with h5py.File(path, "r") as h5:
            x = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
            start = float(np.asarray(h5["meta/GPSstart"][()]).item())
            duration = float(np.asarray(h5["meta/Duration"][()]).item())
        self.cache[rel] = (x, start, duration)
        self.order.append(rel)
        while len(self.order) > self.max_files:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return self.cache[rel]


def build_noise_bank(source_run: Path, shared_dir: Path, seed: int) -> dict[str, Any]:
    marker = shared_dir / "noise_bank_summary.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    source = pd.read_parquet(source_run / "data" / "real_noise_injections" / "offsource_noise_segments.parquet")
    source = source[(source.segment_end - source.segment_start) >= PSD_SECONDS + 2.0].reset_index(drop=True)
    if source.empty:
        raise RuntimeError("No sufficiently long H1/L1 off-source segments")
    rng = np.random.default_rng(seed)
    cache = HdfCache(source_run)
    references: list[np.ndarray] = []
    psds: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    frequency: np.ndarray | None = None
    attempts = 0
    while len(references) < NOISE_BANK_SIZE and attempts < NOISE_BANK_SIZE * 30:
        attempts += 1
        row = source.iloc[int(rng.integers(0, len(source)))]
        latest = float(row.segment_end) - PSD_SECONDS
        start = float(rng.uniform(float(row.segment_start), latest))
        channels = []
        ok = True
        for detector in DETECTORS:
            data, gps_start, duration = cache.get(str(row[f"{detector}_path"]))
            fs = len(data) / duration
            i0 = int(round((start - gps_start) * fs))
            part = data[i0 : i0 + NOISE_REFERENCE_SAMPLES]
            if len(part) != NOISE_REFERENCE_SAMPLES or np.mean(np.isfinite(part)) < 0.999:
                ok = False
                break
            channels.append(part)
        if not ok:
            continue
        reference = np.stack(channels).astype(np.float32)
        f, p = estimate_psd(reference)
        references.append(reference)
        psds.append(p.astype(np.float64))
        frequency = f
        rows.append(
            {
                "bank_index": len(references) - 1,
                "source_event": str(row.event_name),
                "segment_kind": str(row.segment_kind),
                "reference_start_gps": start,
                "reference_duration_s": PSD_SECONDS,
                "finite_fraction": float(np.mean(np.isfinite(reference))),
            }
        )
    if len(references) < max(16, NOISE_BANK_SIZE // 2):
        raise RuntimeError(f"Only {len(references)} valid PSD/noise references")
    shared_dir.mkdir(parents=True, exist_ok=True)
    np.save(shared_dir / "noise_reference_bank.npy", np.asarray(references, dtype=np.float32))
    np.save(shared_dir / "noise_psd_bank.npy", np.asarray(psds, dtype=np.float64))
    np.save(shared_dir / "noise_psd_frequency.npy", np.asarray(frequency, dtype=np.float64))
    pd.DataFrame(rows).to_csv(shared_dir / "noise_bank_manifest.csv", index=False)
    payload = {
        "references": len(references),
        "reference_seconds": PSD_SECONDS,
        "sample_rate_hz": RAW_SAMPLE_RATE,
        "psd_method": "Welch, 8 s Hann segments, 50% overlap, off-source H1/L1 public strain",
    }
    write_json(marker, payload)
    return payload


def draw_noise_window(
    references: np.ndarray,
    psds: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    bank_index = int(rng.integers(0, len(references)))
    latest = references.shape[-1] - RAW_PADDED_SAMPLES
    offset = int(rng.integers(0, latest + 1))
    window = np.asarray(references[bank_index, :, offset : offset + RAW_PADDED_SAMPLES], dtype=np.float32)
    return window, np.asarray(psds[bank_index], dtype=np.float64), bank_index, offset


def draw_truncated_snr(rng: np.random.Generator, minimum: float = SNR_MIN, maximum: float = SNR_MAX) -> float:
    a, b = minimum ** -3, maximum ** -3
    return float((a - rng.random() * (a - b)) ** (-1.0 / 3.0))


def draw_snr_pair(ratio: float, rng: np.random.Generator) -> tuple[float, float]:
    ratio = float(np.clip(ratio, 1.0, SNR_MAX / SNR_MIN))
    bright = draw_truncated_snr(rng, SNR_MIN * ratio, SNR_MAX)
    faint = bright / ratio
    return (bright, faint) if rng.random() < 0.5 else (faint, bright)


def liao_delay_ratio_samples() -> tuple[np.ndarray, np.ndarray, str]:
    module = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")
    image_csv = Path(
        "/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR8/"
        "BBH_2.5PLUS_Any_Detected_SNR8_ImageParams.csv"
    )
    delays, ratios = module.extract_liao_delay_snr_pairs(image_csv, 8.0)
    delays = np.asarray(delays, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    keep = np.isfinite(delays) & (delays > 0) & np.isfinite(ratios) & (ratios >= 1)
    return delays[keep], ratios[keep], "GW-LMC 2.5PLUS BBH: at least two images with simulated SNR >= 8"


def draw_lensed_timing(
    delays: np.ndarray,
    ratios: np.ndarray,
    schedule: list[LiveSegment],
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    for _ in range(20_000):
        idx = int(rng.integers(0, len(delays)))
        delay = float(delays[idx])
        first = float(sample_live_times(schedule, 1, rng)[0])
        second = first + delay * SECONDS_PER_DAY
        if bool(in_live_segments(np.asarray([second]), schedule)[0]):
            return first, second, delay, float(ratios[idx])
    raise RuntimeError("Unable to place lensed timing in run exposure")


def build_time_calibration(
    shared_dir: Path,
    schedule: list[LiveSegment],
    raw_delays: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    path = shared_dir / "time_delay_likelihood_ratio.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    lens = draw_detected_lens_delays(raw_delays, schedule, 100_000, rng)
    null = draw_null_delays(schedule, 250_000, rng)
    calibration = fit_time_likelihood_ratio(lens, null, grid_size=2048)
    calibration.update(
        {
            "definition": "log p(delta_t|lensed,run exposure) - log p(delta_t|null,run exposure)",
            "signal_source": "GW-LMC 2.5PLUS LIGO BBH detected-image delay samples",
            "exposure_conditioning": "Both images must fall in actual public H1-L1 joint DATA segments",
            "null_definition": "Absolute separation of two independent event times drawn from the same run exposure and run-specific observed-event intensity",
            "fit_policy": "Frozen before validation/test/real-catalog scoring",
        }
    )
    write_json(path, calibration)
    np.save(shared_dir / "time_delay_lensed_samples_days.npy", lens.astype(np.float32))
    np.save(shared_dir / "time_delay_null_samples_days.npy", null.astype(np.float32))
    return calibration


def build_sky_template_library(source_run: Path, shared_dir: Path) -> dict[str, Any]:
    map_path = shared_dir / f"real_pe_sky_templates_nside{SKY_NSIDE}.npy"
    manifest_path = shared_dir / "real_pe_sky_template_manifest.csv"
    summary_path = shared_dir / "real_pe_sky_template_summary.json"
    if map_path.exists() and manifest_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    maps: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    manifest = primary_manifest(source_run)
    for map_number, row in enumerate(manifest.itertuples(index=False), start=1):
        print(f"sky template {map_number}/{len(manifest)}: {row.event_name}", flush=True)
        series = pd.Series(row._asdict())
        try:
            probability, meta = read_probability_map(series, target_nside=SKY_NSIDE)
        except Exception as exc:
            rows.append({"event_name": str(row.event_name), "usable": False, "error": str(exc)})
            continue
        maps.append(probability.astype(np.float32))
        rows.append(
            {
                "event_name": str(row.event_name),
                "usable": True,
                "template_index": len(maps) - 1,
                "network_snr": float(getattr(row, "network_snr", np.nan)),
                "area90_deg2": posterior_area90(probability),
                **meta,
            }
        )
    if len(maps) < 10:
        raise RuntimeError(f"Only {len(maps)} usable real PE sky posterior templates")
    shared_dir.mkdir(parents=True, exist_ok=True)
    np.save(map_path, np.stack(maps).astype(np.float32))
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    usable = pd.DataFrame(rows).query("usable == True")
    payload = {
        "n_templates": int(len(maps)),
        "nside": int(SKY_NSIDE),
        "npix": int(hp.nside2npix(SKY_NSIDE)),
        "construction": "Real run-matched PE sky posterior templates; each synthetic event independently draws a template, anchors a posterior draw at the simulated true direction, and randomizes orientation about that direction.",
        "score_definition": "log B_sky = log(Npix * sum_k P_i,k P_j,k), applied identically to synthetic validation/test and real PE maps.",
        "area90_median_deg2": float(usable["area90_deg2"].median()),
        "area90_p90_deg2": float(usable["area90_deg2"].quantile(0.9)),
    }
    write_json(summary_path, payload)
    return payload


def _choose_sky_template(
    snr: float,
    template_snr: np.ndarray,
    rng: np.random.Generator,
    neighbours: int = 8,
) -> int:
    finite = np.isfinite(template_snr) & (template_snr > 0)
    if not np.any(finite):
        return int(rng.integers(0, len(template_snr)))
    candidates = np.flatnonzero(finite)
    distance = np.abs(np.log(np.maximum(template_snr[candidates], 1e-3)) - math.log(max(snr, 1e-3)))
    nearest = candidates[np.argsort(distance)[: min(int(neighbours), len(candidates))]]
    return int(rng.choice(nearest))


def synthetic_event_sky_maps(
    ds_meta: list[dict[str, Any]],
    family: str,
    metadata: pd.DataFrame,
    shared_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    templates = np.load(shared_dir / f"real_pe_sky_templates_nside{SKY_NSIDE}.npy", mmap_mode="r")
    template_frame = pd.read_csv(shared_dir / "real_pe_sky_template_manifest.csv")
    template_frame = template_frame[template_frame["usable"] == True].sort_values("template_index")
    template_snr = template_frame["network_snr"].to_numpy(dtype=np.float64)
    template_names = template_frame["event_name"].astype(str).to_numpy()
    template_area = template_frame["area90_deg2"].to_numpy(dtype=np.float64)
    fam = metadata[metadata["family"] == family].set_index("sample_index")
    unlensed = metadata[metadata["family"] == "unlensed"].set_index("sample_index")
    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    maps: list[np.ndarray] = []
    for idx, item in enumerate(ds_meta):
        source_index = int(item["source_index"])
        tag = str(item["tag"])
        row = unlensed.loc[source_index] if tag == "U" else fam.loc[source_index]
        image = 2 if tag == "L2" else 1
        snr = float(row[f"target_snr_image{image}"])
        gps = float(row[f"gps_image{image}"])
        template_index = _choose_sky_template(snr, template_snr, rng)
        rotated, anchor = rotate_probability_map_to_true_position(
            np.asarray(templates[template_index]), float(row.ra_true), float(row.dec_true), rng
        )
        maps.append(rotated)
        events.append(
            {
                "idx": idx,
                "pair_id": int(item["pair_id"]),
                "tag": tag,
                "source_index": source_index,
                "gps_obs": gps,
                "snr": snr,
                "ra_true": float(row.ra_true),
                "dec_true": float(row.dec_true),
                "sky_template_index": template_index,
                "sky_template_event": template_names[template_index],
                "sky_template_area90_deg2": float(template_area[template_index]),
                "sky_anchor_pixel": int(anchor),
            }
        )
    return pd.DataFrame(events), np.stack(maps).astype(np.float32)


def _preprocess_injection(
    clean: np.ndarray,
    noise: np.ndarray,
    psd_frequency: np.ndarray,
    psd: np.ndarray,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    signal = np.asarray(clean, dtype=np.float32)
    if signal.shape != (2, RAW_MODEL_SAMPLES) or np.mean(np.isfinite(signal)) < 0.999:
        raise ValueError(f"Invalid clean H1/L1 signal shape/finite fraction: {signal.shape}")
    signal = np.nan_to_num(signal, copy=True)
    scaled, scale_factor, recovered = scale_to_network_snr(signal, target_snr, psd_frequency, psd)
    padded_signal = embed_signal_in_padded_window(scaled)
    mixed = np.asarray(noise, dtype=np.float32) + padded_signal
    prepared_mixed = preprocess_24s(mixed, psd_frequency, psd)
    prepared_clean = preprocess_24s(padded_signal, psd_frequency, psd)
    return prepared_clean, prepared_mixed, {
        "target_network_snr": float(target_snr),
        "recovered_optimal_network_snr": float(recovered),
        "physical_strain_scale_factor": float(scale_factor),
        "prepared_mixed_std": float(np.std(prepared_mixed)),
        "prepared_clean_std": float(np.std(prepared_clean)),
    }


def materialize_compact_dataset(
    seed_dir: Path,
    shared_dir: Path,
    deployment: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    base = seed_dir / "data" / "real_noise_injections"
    marker = base / "compact_dataset_summary.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if samples > 600:
        raise ValueError("The fixed clean H1/L1 waveform bank contains 600 systems per family")
    rng = np.random.default_rng(seed)
    references = np.load(shared_dir / "noise_reference_bank.npy", mmap_mode="r")
    psds = np.load(shared_dir / "noise_psd_bank.npy", mmap_mode="r")
    psd_frequency = np.load(shared_dir / "noise_psd_frequency.npy")
    schedule = build_live_schedule(deployment, SOURCES[deployment], shared_dir)
    delays, ratios, delay_label = liao_delay_ratio_samples()
    root = base / "matchroots" / "LIGO"
    metadata: list[dict[str, Any]] = []

    def prepare_one(clean: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        noise, psd, bank_index, offset = draw_noise_window(references, psds, rng)
        pure, mixed, audit = _preprocess_injection(clean, noise, psd_frequency, psd, target)
        return pure, mixed, {**audit, "noise_bank_index": bank_index, "noise_offset_samples": offset}

    for family in FAMILIES:
        source = CLEAN_SOURCE / f"{family}_data_0222"
        clean1 = np.load(source / f"{family}_h_strain_1.npy", mmap_mode="r")[:samples]
        clean2 = np.load(source / f"{family}_h_strain_2.npy", mmap_mode="r")[:samples]
        out = root / f"{family}_data_0222"
        out.mkdir(parents=True, exist_ok=True)
        pure1 = np.empty((samples, 2, MODEL_SAMPLES), dtype=np.float32)
        pure2 = np.empty_like(pure1)
        noisy1 = np.empty_like(pure1)
        noisy2 = np.empty_like(pure1)
        for idx in range(samples):
            gps1, gps2, delay, ratio = draw_lensed_timing(delays, ratios, schedule, rng)
            snr1, snr2 = draw_snr_pair(ratio, rng)
            ra_true = float(rng.uniform(0.0, 2.0 * math.pi))
            dec_true = float(math.asin(rng.uniform(-1.0, 1.0)))
            pure1[idx], noisy1[idx], audit1 = prepare_one(clean1[idx], snr1)
            pure2[idx], noisy2[idx], audit2 = prepare_one(clean2[idx], snr2)
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
                    **{f"image1_{k}": v for k, v in audit1.items()},
                    **{f"image2_{k}": v for k, v in audit2.items()},
                }
            )
        np.save(out / f"{family}_h_strain_1.npy", pure1)
        np.save(out / f"{family}_h_strain_2.npy", pure2)
        np.save(out / f"{family}_data_strain_1.npy", noisy1)
        np.save(out / f"{family}_data_strain_2.npy", noisy2)
        rows = [x for x in metadata if x["family"] == family]
        np.save(out / f"{family}_optimal_SNR_network_1.npy", np.asarray([x["target_snr_image1"] for x in rows], dtype=np.float32))
        np.save(out / f"{family}_optimal_SNR_network_2.npy", np.asarray([x["target_snr_image2"] for x in rows], dtype=np.float32))
        del pure1, pure2, noisy1, noisy2
        gc.collect()

    clean = np.load(CLEAN_SOURCE / "Unlensed_data_0222" / "unlensed_h_strain.npy", mmap_mode="r")[:samples]
    out = root / "Unlensed_data_0222"
    out.mkdir(parents=True, exist_ok=True)
    pure = np.empty((samples, 2, MODEL_SAMPLES), dtype=np.float32)
    noisy = np.empty_like(pure)
    for idx in range(samples):
        gps = float(sample_live_times(schedule, 1, rng)[0])
        snr = draw_truncated_snr(rng)
        ra_true = float(rng.uniform(0.0, 2.0 * math.pi))
        dec_true = float(math.asin(rng.uniform(-1.0, 1.0)))
        pure[idx], noisy[idx], audit = prepare_one(clean[idx], snr)
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
                **{f"image1_{k}": v for k, v in audit.items()},
            }
        )
    np.save(out / "unlensed_h_strain.npy", pure)
    np.save(out / "unlensed_data_strain.npy", noisy)
    np.save(out / "unlensed_optimal_SNR_network.npy", np.asarray([x["target_snr_image1"] for x in metadata if x["family"] == "unlensed"], dtype=np.float32))
    frame = pd.DataFrame(metadata)
    base.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(base / "compact_injection_metadata.parquet", index=False)
    payload = {
        "protocol_version": "real_noise_injection_v3_physical_full24",
        "deployment": deployment,
        "seed": int(seed),
        "samples_per_family": int(samples),
        "prepared_shape": [2, MODEL_SAMPLES],
        "prepared_sample_rate_hz": MODEL_SAMPLE_RATE,
        "prepared_duration_s": MODEL_SAMPLES / MODEL_SAMPLE_RATE,
        "preprocessing": "Per-segment off-source Welch PSD whitening; Tukey taper; 40-580 Hz physical Butterworth bandpass; anti-aliased 4096->2048 Hz resampling; full 24 s crop; robust per-channel scale.",
        "snr_definition": "PSD-weighted optimal H1-L1 network SNR before injection",
        "snr_range": [SNR_MIN, SNR_MAX],
        "delay_prior": delay_label,
        "timing": "Run-exposure conditioned; both images fall in public H1-L1 joint DATA segments",
        "sky": "Run-matched empirical PE posterior-template bootstrap, generated at evaluation time",
        "noise_references": int(len(references)),
        "metadata_rows": int(len(frame)),
    }
    write_json(marker, payload)
    del pure, noisy
    gc.collect()
    return payload


def training_config(seed_dir: Path, family: str, seed: int, samples: int, epochs: int, batch_size: int) -> MatchRunConfig:
    return MatchRunConfig(
        data_root=seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        model_type=family,
        data_mode="noisy",
        out_dir=seed_dir / "waveform_gate" / f"{family.lower()}_physical_full24_ep{epochs}",
        backbone="inceptiontime",
        lensed_limit=samples,
        unlensed_limit=samples,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=max(4, min(32, batch_size * 2)),
        target_len=MODEL_SAMPLES,
        stride=1,
        preprocess="none",
        use_pure_aux=True,
        aug_flip=True,
        aug_roll=128,
        aug_scale=0.1,
        aug_noise=0.01,
        candidate_topk=10,
        seed=seed + (0 if family == "SIS" else 1000),
        num_workers=0,
        pin_memory=True,
        amp=True,
        amp_dtype="bf16",
        export_candidates=False,
    )


def train_family(seed_dir: Path, family: str, seed: int, samples: int, epochs: int, batch_size: int) -> dict[str, Any]:
    cfg = training_config(seed_dir, family, seed, samples, epochs, batch_size)
    summary = cfg.out_dir / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    seed_everything(cfg.seed)
    return run_train_eval(cfg, cpu=False)


def load_checkpoint_model(path: Path, data_root: Path, samples: int, family: str):
    deploy = load_module(f"deploy_loader_{path.parent.name}_{family}", REPO / "scripts" / "real_search" / "15_gwtc34_real_deployment.py")
    model, cfg = deploy.load_model_from_checkpoint(path, data_root, samples, False)
    cfg.model_type = family
    cfg.eval_batch_size = min(int(cfg.eval_batch_size), 32)
    return model, cfg


def build_split_pair_table(
    seed_dir: Path,
    shared_dir: Path,
    family: str,
    epochs: int,
    samples: int,
    split: str,
    time_calibration: dict[str, Any],
    observable_seed: int,
) -> pd.DataFrame:
    output = seed_dir / "results" / f"{family.lower()}_{split}_physical_pair_features.parquet"
    if output.exists():
        return pd.read_parquet(output)
    root = seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    own_cfg = training_config(seed_dir, family, observable_seed - (2000 if split == "val" else 3000), samples, epochs, 4)
    arrays = load_match_arrays(own_cfg)
    parts = split_indices(len(arrays.l1), len(arrays.unlensed), own_cfg)
    ds = EvaluationSet(arrays, parts["lensed"][split], parts["unlensed"][split], own_cfg)
    raw_scores: dict[str, np.ndarray] = {}
    embeddings: dict[str, np.ndarray] = {}
    for encoder_family in FAMILIES:
        checkpoint = seed_dir / "waveform_gate" / f"{encoder_family.lower()}_physical_full24_ep{epochs}" / "model.pt"
        model, cfg = load_checkpoint_model(checkpoint, root, samples, family)
        emb = embed_eval(model, ds, cfg, cpu=False)
        embeddings[encoder_family] = emb
        raw_scores[encoder_family] = similarity_matrix(emb)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gt = ground_truth_partner(ds.meta)
    metadata = pd.read_parquet(seed_dir / "data" / "real_noise_injections" / "compact_injection_metadata.parquet")
    events, maps = synthetic_event_sky_maps(ds.meta, family, metadata, shared_dir, observable_seed)
    events.to_parquet(seed_dir / "results" / f"{family.lower()}_{split}_synthetic_event_observables.parquet", index=False)
    np.save(seed_dir / "results" / f"{family.lower()}_{split}_synthetic_sky_posteriors_nside{SKY_NSIDE}.npy", maps)
    for encoder_family, emb in embeddings.items():
        np.save(seed_dir / "results" / f"{family.lower()}_{split}_{encoder_family.lower()}_embeddings.npy", emb)
    sky_log_bf, sky_raw, sky_cosine = sky_log_bayes_factor_from_maps(maps)
    ii, jj = np.triu_indices(len(events), k=1)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    delta = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    frame = pd.DataFrame(
        {
            "family": family,
            "split": split,
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "is_true_pair": (gt[ii] == jj).astype(np.int8),
            "waveform_raw_sis": raw_scores["SIS"][ii, jj].astype(np.float32),
            "waveform_raw_pm": raw_scores["PM"][ii, jj].astype(np.float32),
            "delta_t_days": delta.astype(np.float64),
            "time_score": apply_time_likelihood_ratio(delta, time_calibration),
            "sky_score": sky_log_bf[ii, jj].astype(np.float32),
            "sky_bayes_factor": np.exp(np.clip(sky_log_bf[ii, jj], -80, 80)).astype(np.float64),
            "sky_raw_overlap": sky_raw[ii, jj].astype(np.float64),
            "sky_cosine_overlap": sky_cosine[ii, jj].astype(np.float32),
            "event_count": int(len(events)),
        }
    )
    frame.to_parquet(output, index=False)
    return frame


def fit_waveform_calibrations(validation: pd.DataFrame) -> dict[str, Any]:
    calibrations: dict[str, Any] = {}
    null_mask = validation["is_true_pair"].to_numpy(dtype=np.int8) == 0
    for family in FAMILIES:
        column = f"waveform_raw_{family.lower()}"
        signal_mask = (validation["is_true_pair"].to_numpy(dtype=np.int8) == 1) & (
            validation["family"].astype(str).str.upper().to_numpy() == family
        )
        calibrations[family] = fit_score_likelihood_ratio(
            validation.loc[signal_mask, column].to_numpy(dtype=np.float64),
            validation.loc[null_mask, column].to_numpy(dtype=np.float64),
            grid_size=1024,
        )
        calibrations[family]["signal_definition"] = f"true held-out validation {family} companion pairs"
        calibrations[family]["null_definition"] = "all non-companion pairs in both validation family catalogs"
    return calibrations


def apply_waveform_calibrations(frame: pd.DataFrame, calibrations: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    sis = apply_score_likelihood_ratio(out["waveform_raw_sis"].to_numpy(), calibrations["SIS"])
    pm = apply_score_likelihood_ratio(out["waveform_raw_pm"].to_numpy(), calibrations["PM"])
    out["waveform_log_bf_sis"] = sis
    out["waveform_log_bf_pm"] = pm
    out["waveform_score"] = (np.logaddexp(sis, pm) - math.log(2.0)).astype(np.float32)
    return out


def score_vector(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        float(weights["waveform"]) * frame["waveform_score"].to_numpy(dtype=np.float64)
        + float(weights["time"]) * frame["time_score"].to_numpy(dtype=np.float64)
        + float(weights["sky"]) * frame["sky_score"].to_numpy(dtype=np.float64)
    )


def retrieval_rows(frame: pd.DataFrame, weights: dict[str, float], deployment: str, seed: int, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, subset in frame.groupby("family", sort=True):
        n = int(subset["event_count"].iloc[0])
        score = np.full((n, n), -np.inf, dtype=np.float64)
        truth = np.zeros((n, n), dtype=bool)
        ii = subset["idx_i"].to_numpy(dtype=np.int32)
        jj = subset["idx_j"].to_numpy(dtype=np.int32)
        values = score_vector(subset, weights)
        score[ii, jj] = values
        score[jj, ii] = values
        labels = subset["is_true_pair"].to_numpy(dtype=bool)
        truth[ii, jj] = labels
        truth[jj, ii] = labels
        for query in np.flatnonzero(truth.any(axis=1)):
            partner = int(np.flatnonzero(truth[query])[0])
            rank = int(1 + np.sum(score[query] > score[query, partner]))
            rows.append(
                {
                    "deployment": deployment,
                    "seed": int(seed),
                    "method": method,
                    "query_index": int(query),
                    "partner_index": partner,
                    "family": str(family).upper(),
                    "system_id": f"{family}:{min(query, partner)}-{max(query, partner)}",
                    "query_tag": "directed_query",
                    "query_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def retrieval_metric_dict(frame: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    ranks = retrieval_rows(frame, weights, "validation", 0, "grid")
    out: dict[str, float] = {}
    family_rows = []
    for family, sub in ranks.groupby("family"):
        r = sub["query_rank"].to_numpy(dtype=np.int32)
        metrics = {
            "r_at_1": float(np.mean(r <= 1)),
            "r_at_5": float(np.mean(r <= 5)),
            "r_at_10": float(np.mean(r <= 10)),
            "median_rank": float(np.median(r)),
        }
        family_rows.append(metrics)
        for key, value in metrics.items():
            out[f"{family.lower()}_{key}"] = value
    for key in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
        out[f"macro_{key}"] = float(np.mean([x[key] for x in family_rows]))
    return out


def pair_metric_dict(frame: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    score = score_vector(frame, weights)
    label = frame["is_true_pair"].to_numpy(dtype=np.int8)
    order = np.argsort(-score, kind="stable")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_true = max(int(label.sum()), 1)
    idx50 = min(int(np.searchsorted(tp / n_true, 0.5, side="left")), len(tp) - 1)
    idx90 = min(int(np.searchsorted(tp / n_true, 0.9, side="left")), len(tp) - 1)
    return {
        "average_precision": float(average_precision_score(label, score)),
        "precision_at_recall_0p5": float(tp[idx50] / max(tp[idx50] + fp[idx50], 1)),
        "false_at_recall_0p5": int(fp[idx50]),
        "precision_at_recall_0p9": float(tp[idx90] / max(tp[idx90] + fp[idx90], 1)),
        "false_at_recall_0p9": int(fp[idx90]),
    }


def select_weights_v3(
    validation: pd.DataFrame,
    require_all_positive: bool = False,
    force_waveform_zero: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for ww, wt, ws in itertools.product(WEIGHT_GRID, repeat=3):
        if ww == wt == ws == 0:
            continue
        if force_waveform_zero and ww != 0:
            continue
        if require_all_positive and min(ww, wt, ws) <= 0:
            continue
        if force_waveform_zero and min(wt, ws) <= 0:
            continue
        weights = {"waveform": ww, "time": wt, "sky": ws}
        retrieval = retrieval_metric_dict(validation, weights)
        rows.append({**weights, **retrieval, "weight_l2": float(math.sqrt(ww * ww + wt * wt + ws * ws))})
    grid = pd.DataFrame(rows)
    best_r10 = float(grid["macro_r_at_10"].max())
    eligible_index = grid.index[grid["macro_r_at_10"] >= best_r10 - 0.005]
    for idx in eligible_index:
        weights = {name: float(grid.loc[idx, name]) for name in ("waveform", "time", "sky")}
        for key, value in pair_metric_dict(validation, weights).items():
            grid.loc[idx, key] = value
    grid["average_precision"] = grid["average_precision"].fillna(-np.inf)
    grid["precision_at_recall_0p5"] = grid["precision_at_recall_0p5"].fillna(-np.inf)
    grid = grid.sort_values(
        ["macro_r_at_10", "average_precision", "precision_at_recall_0p5", "macro_r_at_1", "weight_l2"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return {name: float(best[name]) for name in ("waveform", "time", "sky")}, grid


def pair_level_metric_rows(frame: pd.DataFrame, methods: dict[str, dict[str, float]], deployment: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = frame["is_true_pair"].to_numpy(dtype=np.int8)
    n_true = int(labels.sum())
    n_false = int(len(labels) - n_true)
    for method, weights in methods.items():
        scores = score_vector(frame, weights)
        order = np.argsort(-scores, kind="stable")
        y = labels[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        rows.append({"deployment": deployment, "seed": seed, "method": method, "metric": "average_precision", "value": float(average_precision_score(labels, scores)), "n_true_pairs": n_true, "n_false_pairs": n_false})
        for target in (0.5, 0.9):
            idx = min(int(np.searchsorted(tp / max(n_true, 1), target, side="left")), len(tp) - 1)
            rows.append({"deployment": deployment, "seed": seed, "method": method, "metric": f"at_recall_{target:g}", "value": float(tp[idx] / max(tp[idx] + fp[idx], 1)), "recall": float(tp[idx] / max(n_true, 1)), "precision": float(tp[idx] / max(tp[idx] + fp[idx], 1)), "true_pairs": int(tp[idx]), "false_pairs": int(fp[idx]), "n_true_pairs": n_true, "n_false_pairs": n_false})
        budget = max(1, int(math.floor(1e-5 * n_false)))
        valid = np.flatnonzero(fp <= budget)
        idx = int(valid[-1]) if len(valid) else 0
        rows.append({"deployment": deployment, "seed": seed, "method": method, "metric": "at_false_pair_rate_1e-5", "value": float(tp[idx] / max(n_true, 1)), "recall": float(tp[idx] / max(n_true, 1)), "precision": float(tp[idx] / max(tp[idx] + fp[idx], 1)), "true_pairs": int(tp[idx]), "false_pairs": int(fp[idx]), "n_true_pairs": n_true, "n_false_pairs": n_false})
    return pd.DataFrame(rows)


def _finite_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in changes.reshape(-1, 2)]


def _extract_channel_window(data: np.ndarray, gps_start: float, gps: float, samples: int, start_offset_s: float) -> np.ndarray | None:
    i0 = int(round((gps + start_offset_s - gps_start) * RAW_SAMPLE_RATE))
    part = np.asarray(data[i0 : i0 + samples], dtype=np.float32)
    if len(part) != samples or np.mean(np.isfinite(part)) < 0.999:
        return None
    return part


def _extract_psd_reference(data: np.ndarray, gps_start: float, gps: float) -> tuple[np.ndarray | None, float | None]:
    valid = np.isfinite(data)
    exclusion_lo = int(round((gps - 128.0 - gps_start) * RAW_SAMPLE_RATE))
    exclusion_hi = int(round((gps + 128.0 - gps_start) * RAW_SAMPLE_RATE))
    candidates: list[tuple[float, int]] = []
    for lo, hi in _finite_blocks(valid):
        for a, b in ((lo, min(hi, exclusion_lo)), (max(lo, exclusion_hi), hi)):
            if b - a < NOISE_REFERENCE_SAMPLES:
                continue
            midpoint = 0.5 * (a + b)
            event_index = (gps - gps_start) * RAW_SAMPLE_RATE
            start = int(np.clip(event_index - NOISE_REFERENCE_SAMPLES / 2, a, b - NOISE_REFERENCE_SAMPLES))
            candidates.append((abs(midpoint - event_index), start))
    if not candidates:
        return None, None
    start = min(candidates)[1]
    return np.asarray(data[start : start + NOISE_REFERENCE_SAMPLES], dtype=np.float32), float(gps_start + start / RAW_SAMPLE_RATE)


def build_real_preprocessed_inputs(
    deployment: str,
    source_run: Path,
    shared_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    array_path = shared_dir / "real_event_preprocessed_full24.npy"
    audit_path = shared_dir / "real_event_preprocessing_audit.csv"
    if array_path.exists() and audit_path.exists():
        return np.load(array_path, mmap_mode="r"), pd.read_csv(audit_path)
    manifest = primary_manifest(source_run)
    strain = pd.read_csv(source_run / "data" / "strain_gwosc_download_manifest.csv")
    cache = HdfCache(source_run, max_files=6)
    inputs = np.zeros((len(manifest), 2, MODEL_SAMPLES), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for idx, event in manifest.iterrows():
        channels: list[np.ndarray] = []
        references: list[np.ndarray] = []
        detector_rows: list[dict[str, Any]] = []
        reason = ""
        for detector in DETECTORS:
            matches = strain[(strain["event_name"].astype(str) == str(event.event_name)) & (strain["detector"].astype(str) == detector)]
            if matches.empty:
                reason = f"missing_{detector}_manifest"
                break
            rel = str(matches.iloc[0]["local_path"])
            try:
                data, gps_start, duration = cache.get(rel)
            except Exception as exc:
                reason = f"{detector}_read_error:{exc}"
                break
            sample_rate = len(data) / duration
            if abs(sample_rate - RAW_SAMPLE_RATE) > 1e-3:
                reason = f"{detector}_sample_rate_{sample_rate:g}"
                break
            onsource = _extract_channel_window(data, gps_start, float(event.gps_time), RAW_PADDED_SAMPLES, -24.75)
            reference, reference_start = _extract_psd_reference(data, gps_start, float(event.gps_time))
            detector_rows.append(
                {
                    "detector": detector,
                    "path": rel,
                    "file_duration_s": duration,
                    "file_finite_fraction": float(np.mean(np.isfinite(data))),
                    "onsource_finite": onsource is not None,
                    "psd_reference_start_gps": reference_start,
                }
            )
            if onsource is None:
                reason = f"{detector}_onsource_nonfinite_or_short"
                break
            if reference is None:
                reason = f"{detector}_no_256s_finite_offsource_psd"
                break
            channels.append(onsource)
            references.append(reference)
        available = len(channels) == 2 and len(references) == 2
        if available:
            frequency, psd = estimate_psd(np.stack(references))
            try:
                inputs[idx] = preprocess_24s(np.stack(channels), frequency, psd)
            except Exception as exc:
                available = False
                reason = f"preprocess_error:{exc}"
        object_class = str(event.get("object_class", "BBH")) if isinstance(event, pd.Series) else "BBH"
        ood = bool(event.get("is_ood_for_bbh_encoder", False)) if isinstance(event, pd.Series) else False
        rows.append(
            {
                "idx": int(idx),
                "event_name": str(event.event_name),
                "gps_time": float(event.gps_time),
                "strict_h1l1_preprocessing_pass": bool(available),
                "failure_reason": reason,
                "object_class": object_class,
                "is_ood_for_bbh_encoder": ood,
                "prepared_std_h1": float(np.std(inputs[idx, 0])) if available else np.nan,
                "prepared_std_l1": float(np.std(inputs[idx, 1])) if available else np.nan,
                "detector_audit": json.dumps(detector_rows),
            }
        )
    shared_dir.mkdir(parents=True, exist_ok=True)
    np.save(array_path, inputs)
    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False)
    write_json(
        shared_dir / "real_event_preprocessing_summary.json",
        {
            "deployment": deployment,
            "n_primary_events": int(len(audit)),
            "n_strict_h1l1_preprocessing_pass": int(audit["strict_h1l1_preprocessing_pass"].sum()),
            "n_strict_h1l1_bbh": int((audit["strict_h1l1_preprocessing_pass"] & ~audit["is_ood_for_bbh_encoder"]).sum()),
            "onsource_window": "[gps-24.75 s, gps+1.25 s] before preprocessing; central crop is [gps-23.75 s, gps+0.25 s]",
            "psd_reference": "Nearest finite 256 s segment outside the event +/-128 s exclusion interval, independently for H1 and L1",
            "silent_zero_fill": False,
        },
    )
    return inputs, audit


@torch.no_grad()
def embed_preprocessed_inputs(model: torch.nn.Module, values: np.ndarray, batch_size: int = 16) -> np.ndarray:
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(values), batch_size):
        prepared = []
        for sample in np.asarray(values[start : start + batch_size], dtype=np.float32):
            prepared.append(zscore_channels(peak_flip_channels(sample)))
        tensor = torch.from_numpy(np.stack(prepared)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            chunks.append(model(tensor).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def embed_real_events_v3(
    deployment: str,
    seed_dir: Path,
    shared_dir: Path,
    epochs: int,
    samples: int,
    calibrations: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb_path = seed_dir / "features" / "real_waveform_embeddings_physical.parquet"
    pair_path = seed_dir / "features" / "real_waveform_similarity_physical.parquet"
    if emb_path.exists() and pair_path.exists():
        return pd.read_parquet(emb_path), pd.read_parquet(pair_path)
    inputs, audit = build_real_preprocessed_inputs(deployment, SOURCES[deployment], shared_dir)
    available = audit["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    root = seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    embedding_by_family: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        checkpoint = seed_dir / "waveform_gate" / f"{family.lower()}_physical_full24_ep{epochs}" / "model.pt"
        model, _ = load_checkpoint_model(checkpoint, root, samples, family)
        z = np.full((len(audit), 128), np.nan, dtype=np.float32)
        z[available] = embed_preprocessed_inputs(model, np.asarray(inputs[available]), batch_size=8)
        embedding_by_family[family] = z
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    emb = audit.copy()
    for family, values in embedding_by_family.items():
        emb[f"{family.lower()}_embedding"] = [row.tolist() if np.isfinite(row).all() else None for row in values]
    emb.to_parquet(emb_path, index=False)
    ii, jj = np.triu_indices(len(emb), k=1)
    both = available[ii] & available[jj]
    raw_sis = np.full(len(ii), np.nan, dtype=np.float32)
    raw_pm = np.full(len(ii), np.nan, dtype=np.float32)
    raw_sis[both] = np.sum(embedding_by_family["SIS"][ii[both]] * embedding_by_family["SIS"][jj[both]], axis=1)
    raw_pm[both] = np.sum(embedding_by_family["PM"][ii[both]] * embedding_by_family["PM"][jj[both]], axis=1)
    sis = np.zeros(len(ii), dtype=np.float32)
    pm = np.zeros(len(ii), dtype=np.float32)
    sis[both] = apply_score_likelihood_ratio(raw_sis[both], calibrations["SIS"])
    pm[both] = apply_score_likelihood_ratio(raw_pm[both], calibrations["PM"])
    mixture = np.zeros(len(ii), dtype=np.float32)
    mixture[both] = (np.logaddexp(sis[both], pm[both]) - math.log(2.0)).astype(np.float32)
    pairs = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": emb["event_name"].to_numpy()[ii],
            "event_j": emb["event_name"].to_numpy()[jj],
            "waveform_available": both,
            "waveform_raw_sis": raw_sis,
            "waveform_raw_pm": raw_pm,
            "waveform_log_bf_sis": sis,
            "waveform_log_bf_pm": pm,
            "waveform_score": mixture,
        }
    )
    pairs.to_parquet(pair_path, index=False)
    valid_embeddings = np.concatenate([embedding_by_family[f][available] for f in FAMILIES], axis=0)
    valid_scores = mixture[both]
    write_json(
        seed_dir / "results" / "real_waveform_deployment_audit_physical.json",
        {
            "deployment": deployment,
            "strict_h1l1_events": int(available.sum()),
            "strict_h1l1_pairs": int(both.sum()),
            "embedding_effective_rank_combined": effective_rank(valid_embeddings),
            "embedding_effective_rank_sis": effective_rank(embedding_by_family["SIS"][available]),
            "embedding_effective_rank_pm": effective_rank(embedding_by_family["PM"][available]),
            "waveform_score_min": float(np.min(valid_scores)) if len(valid_scores) else None,
            "waveform_score_max": float(np.max(valid_scores)) if len(valid_scores) else None,
            "waveform_score_std": float(np.std(valid_scores)) if len(valid_scores) else None,
            "waveform_score_unique_rounded_1e6": int(len(np.unique(np.round(valid_scores, 6)))) if len(valid_scores) else 0,
            "constant_score_failure": bool(len(valid_scores) == 0 or np.std(valid_scores) < 1e-6),
            "silent_zero_fill": False,
        },
    )
    return emb, pairs


def _rank_real_pairs(frame: pd.DataFrame, weights: dict[str, float], method: str) -> pd.DataFrame:
    out = frame.copy()
    out["waveform_contribution"] = float(weights["waveform"]) * out["waveform_score"].to_numpy(dtype=np.float64)
    out["time_contribution"] = float(weights["time"]) * out["time_score"].to_numpy(dtype=np.float64)
    out["sky_contribution"] = float(weights["sky"]) * out["sky_score"].to_numpy(dtype=np.float64)
    out["final_score"] = out[["waveform_contribution", "time_contribution", "sky_contribution"]].sum(axis=1)
    out["final_score_i_to_j"] = out["final_score"]
    out["final_score_j_to_i"] = out["final_score"]
    out["unordered_max_score"] = out["final_score"]
    out["unordered_mean_score"] = out["final_score"]
    out["method"] = method
    out = out.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def real_catalog_scores_v3(
    deployment: str,
    seed_dir: Path,
    shared_dir: Path,
    methods: dict[str, dict[str, float]],
    epochs: int,
    samples: int,
    waveform_calibrations: dict[str, Any],
    time_calibration: dict[str, Any],
) -> dict[str, Any]:
    source_run = SOURCES[deployment]
    primary = primary_manifest(source_run)
    embedding, waveform = embed_real_events_v3(
        deployment, seed_dir, shared_dir, epochs, samples, waveform_calibrations
    )
    sky = pd.read_parquet(source_run / "features" / "real_sky_overlap.parquet")
    expected_i = primary["event_name"].astype(str).to_numpy()[sky["idx_i"].to_numpy(dtype=np.int32)]
    expected_j = primary["event_name"].astype(str).to_numpy()[sky["idx_j"].to_numpy(dtype=np.int32)]
    if not np.array_equal(expected_i, sky["event_i"].astype(str).to_numpy()) or not np.array_equal(expected_j, sky["event_j"].astype(str).to_numpy()):
        raise RuntimeError(f"{deployment} real sky-pair indices do not match the primary manifest")
    real = sky.merge(waveform, on=["idx_i", "idx_j", "event_i", "event_j"], how="left", validate="one_to_one")
    ii = real["idx_i"].to_numpy(dtype=np.int32)
    jj = real["idx_j"].to_numpy(dtype=np.int32)
    gps = primary["gps_time"].to_numpy(dtype=np.float64)
    real["delta_t_days"] = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    real["time_score"] = apply_time_likelihood_ratio(real["delta_t_days"].to_numpy(), time_calibration)
    nside = real["common_nside"].to_numpy(dtype=np.int64)
    npix = 12 * nside * nside
    real["sky_bayes_factor"] = npix * real["raw_posterior_overlap"].to_numpy(dtype=np.float64)
    real["sky_score"] = np.log(np.maximum(real["sky_bayes_factor"], 1e-300))
    real["sky_log_cosine_overlap"] = np.log(np.maximum(real["cosine_overlap"].to_numpy(dtype=np.float64), 1e-300))
    event_available = embedding["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    event_ood = embedding["is_ood_for_bbh_encoder"].to_numpy(dtype=bool)
    real["waveform_available"] = event_available[ii] & event_available[jj]
    real["pair_has_ood"] = event_ood[ii] | event_ood[jj]
    real["strict_h1l1_bbh_pair"] = real["waveform_available"] & ~real["pair_has_ood"]
    real["waveform_score"] = real["waveform_score"].fillna(0.0)
    real.loc[~real["waveform_available"], "waveform_score"] = 0.0
    object_class = embedding["object_class"].astype(str).to_numpy()
    real["object_class_i"] = object_class[ii]
    real["object_class_j"] = object_class[jj]
    network_snr = primary["network_snr"].to_numpy(dtype=np.float64)
    real["network_snr_i"] = network_snr[ii]
    real["network_snr_j"] = network_snr[jj]
    detectors = primary["detectors_available"].astype(str).to_numpy() if "detectors_available" in primary else np.asarray([""] * len(primary))
    real["detectors_i"] = detectors[ii]
    real["detectors_j"] = detectors[jj]
    outputs: dict[str, Any] = {}
    for method, weights in methods.items():
        full = _rank_real_pairs(real, weights, method)
        strict = _rank_real_pairs(real[real["strict_h1l1_bbh_pair"]].copy(), weights, method)
        full.to_parquet(seed_dir / "results" / f"real_pair_scores_all_catalog_{method}.parquet", index=False)
        strict.to_parquet(seed_dir / "results" / f"real_pair_scores_strict_h1l1_bbh_{method}.parquet", index=False)
        full.head(100).to_csv(seed_dir / "results" / f"candidate_shortlist_all_catalog_{method}.csv", index=False)
        strict.head(100).to_csv(seed_dir / "results" / f"candidate_shortlist_strict_h1l1_bbh_{method}.csv", index=False)
        outputs[method] = {
            "all": full,
            "strict": strict,
            "top_all": full.iloc[0].to_dict() if len(full) else {},
            "top_strict": strict.iloc[0].to_dict() if len(strict) else {},
        }
    return outputs


def _historical_pair_rank(frame: pd.DataFrame, event_a: str, event_b: str) -> dict[str, Any]:
    match = frame[((frame["event_i"] == event_a) & (frame["event_j"] == event_b)) | ((frame["event_i"] == event_b) & (frame["event_j"] == event_a))]
    if match.empty:
        return {"found": False}
    row = match.iloc[0]
    return {"found": True, "rank": int(row["rank"]), "final_score": float(row["final_score"]), "strict_h1l1_bbh_pair": bool(row["strict_h1l1_bbh_pair"])}


def run_seed(
    deployment: str,
    seed: int,
    out_root: Path,
    samples: int,
    epochs: int,
    batch_size: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    seed_dir = out_root / deployment.lower() / f"seed_{seed}"
    shared_dir = out_root / deployment.lower() / "shared"
    complete = seed_dir / "seed_summary.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    started = time.perf_counter()
    prepare_seed_layout(seed_dir, SOURCES[deployment])
    schedule = build_live_schedule(deployment, SOURCES[deployment], shared_dir)
    build_noise_bank(SOURCES[deployment], shared_dir, 20260721 + (0 if deployment == "GWTC3" else 1000))
    build_sky_template_library(SOURCES[deployment], shared_dir)
    raw_delays, _, delay_label = liao_delay_ratio_samples()
    time_calibration = build_time_calibration(shared_dir, schedule, raw_delays, 20260722 + (0 if deployment == "GWTC3" else 1000))
    dataset = materialize_compact_dataset(seed_dir, shared_dir, deployment, seed + 100, samples)

    split_sets: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        cfg = training_config(seed_dir, family, seed, samples, epochs, batch_size)
        parts = split_indices(samples, samples, cfg)
        split_sets[f"{family}_lensed"] = parts["lensed"]
        split_sets[f"{family}_unlensed"] = parts["unlensed"]
    split_audit = split_integrity_audit(split_sets)
    if not split_audit["all_disjoint"]:
        raise RuntimeError("Source-system train/validation/test leakage detected")
    write_json(seed_dir / "results" / "split_integrity_audit.json", split_audit)

    for family in FAMILIES:
        print(f"{deployment} seed {seed}: training {family} full-24-s encoder", flush=True)
        train_family(seed_dir, family, seed, samples, epochs, batch_size)

    validation = pd.concat(
        [build_split_pair_table(seed_dir, shared_dir, family, epochs, samples, "val", time_calibration, seed + 2000) for family in FAMILIES],
        ignore_index=True,
    )
    test = pd.concat(
        [build_split_pair_table(seed_dir, shared_dir, family, epochs, samples, "test", time_calibration, seed + 3000) for family in FAMILIES],
        ignore_index=True,
    )
    waveform_calibrations = fit_waveform_calibrations(validation)
    validation = apply_waveform_calibrations(validation, waveform_calibrations)
    test = apply_waveform_calibrations(test, waveform_calibrations)
    write_json(seed_dir / "results" / "waveform_score_likelihood_ratio_calibration.json", waveform_calibrations)
    validation.to_parquet(seed_dir / "results" / "fusion_validation_pairs_physical.parquet", index=False)
    test.to_parquet(seed_dir / "results" / "fusion_heldout_test_pairs_physical.parquet", index=False)

    unconstrained, grid_unconstrained = select_weights_v3(validation)
    positive, grid_positive = select_weights_v3(validation, require_all_positive=True)
    time_sky, grid_time_sky = select_weights_v3(validation, force_waveform_zero=True)
    grid_unconstrained.to_csv(seed_dir / "results" / "fusion_weight_grid_unconstrained.csv", index=False)
    grid_positive.to_csv(seed_dir / "results" / "fusion_weight_grid_strictly_positive.csv", index=False)
    grid_time_sky.to_csv(seed_dir / "results" / "fusion_weight_grid_time_sky.csv", index=False)
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_bayes_factor_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "time_sky_validation_selected": time_sky,
        "three_channel_unconstrained": unconstrained,
        "three_channel_strictly_positive": positive,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
    }
    query = pd.concat([retrieval_rows(test, weights, deployment, seed, method) for method, weights in methods.items()], ignore_index=True)
    query.to_parquet(seed_dir / "results" / "heldout_test_query_ranks.parquet", index=False)
    points = metric_rows(query)
    points.to_csv(seed_dir / "results" / "heldout_test_retrieval_metrics.csv", index=False)
    pair_metrics = pair_level_metric_rows(test, methods, deployment, seed)
    pair_metrics.to_csv(seed_dir / "results" / "heldout_test_pair_level_metrics.csv", index=False)
    bootstrap = bootstrap_system_ci(query, draws=bootstrap_draws, seed=seed + 4000)
    bootstrap.to_csv(seed_dir / "results" / "heldout_test_bootstrap_95ci.csv", index=False)
    wf = points[points["method"] == "waveform_only"].set_index("subset")
    family_r10 = [float(wf.loc[family, "r_at_10"]) for family in FAMILIES]
    gate = {
        "split": "held-out test",
        "overall_r_at_1": float(wf.loc["overall", "r_at_1"]),
        "overall_r_at_10": float(wf.loc["overall", "r_at_10"]),
        "sis_r_at_10": family_r10[0],
        "pm_r_at_10": family_r10[1],
        "macro_family_r_at_10": float(np.mean(family_r10)),
        "min_family_r_at_10": float(np.min(family_r10)),
        "pass_rule": "macro family R@10 > 0.6 and minimum-family R@10 > 0.5",
        "passed": bool(np.mean(family_r10) > 0.6 and np.min(family_r10) > 0.5),
    }
    write_json(seed_dir / "results" / "waveform_gate1_metrics.json", gate)
    write_json(
        seed_dir / "results" / "selected_weights.json",
        {
            "selection_split": "held-out validation systems only",
            "test_split": "held-out test systems only",
            "selection_objective": "maximize macro R@10; within 0.005 choose higher pair AUPRC, precision at 50% recall, macro R@1, then smaller L2 norm",
            "three_channel_unconstrained": unconstrained,
            "three_channel_strictly_positive": positive,
            "time_sky_validation_selected": time_sky,
            "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
        },
    )
    real_methods = {
        "three_channel_unconstrained": unconstrained,
        "three_channel_strictly_positive": positive,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
        "time_sky_validation_selected": time_sky,
    }
    real = real_catalog_scores_v3(deployment, seed_dir, shared_dir, real_methods, epochs, samples, waveform_calibrations, time_calibration)
    audit = json.loads((seed_dir / "results" / "real_waveform_deployment_audit_physical.json").read_text(encoding="utf-8"))
    deployment_pass = bool(gate["passed"] and not audit["constant_score_failure"] and audit["embedding_effective_rank_combined"] >= 5.0)
    primary_method = "three_channel_unconstrained" if unconstrained["waveform"] > 0 and deployment_pass else "three_channel_strictly_positive"
    selected_strict = real[primary_method]["strict"]
    summary = {
        "deployment": deployment,
        "seed": int(seed),
        "status": "complete",
        "samples_per_family": int(samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "delay_prior": delay_label,
        "dataset": dataset,
        "waveform_gate1": gate,
        "real_deployment_audit": audit,
        "deployment_passed": deployment_pass,
        "primary_method_policy": primary_method,
        "weights": {"unconstrained": unconstrained, "strictly_positive": positive, "time_sky": time_sky},
        "strict_primary_top_pair": selected_strict.iloc[0].to_dict() if len(selected_strict) else {},
        "gw170104_gw170814": _historical_pair_rank(real[primary_method]["all"], "GW170104", "GW170814") if deployment == "GWTC3" else {"found": False},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(complete, summary)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_deployment(out_root: Path, deployment: str, seeds: list[int]) -> None:
    root = out_root / deployment.lower()
    query = pd.concat([pd.read_parquet(root / f"seed_{seed}" / "results" / "heldout_test_query_ranks.parquet") for seed in seeds], ignore_index=True)
    query.to_parquet(root / "heldout_test_query_ranks_all_seeds.parquet", index=False)
    per_seed = metric_rows(query)
    per_seed.to_csv(root / "heldout_test_retrieval_metrics_per_seed.csv", index=False)
    across_seed_summary(per_seed).to_csv(root / "heldout_test_retrieval_metrics_across_seed_summary.csv", index=False)
    bootstrap = pd.concat([pd.read_csv(root / f"seed_{seed}" / "results" / "heldout_test_bootstrap_95ci.csv") for seed in seeds], ignore_index=True)
    bootstrap.to_csv(root / "heldout_test_bootstrap_95ci_per_seed.csv", index=False)
    pair = pd.concat([pd.read_csv(root / f"seed_{seed}" / "results" / "heldout_test_pair_level_metrics.csv") for seed in seeds], ignore_index=True)
    pair.to_csv(root / "heldout_test_pair_level_metrics_per_seed.csv", index=False)
    pair.groupby(["deployment", "method", "metric"], as_index=False).agg(
        value_mean=("value", "mean"), value_std=("value", "std"), seeds=("seed", "nunique")
    ).to_csv(root / "heldout_test_pair_level_metrics_across_seed_summary.csv", index=False)
    weight_rows = []
    seed_summaries = []
    for seed in seeds:
        selected = json.loads((root / f"seed_{seed}" / "results" / "selected_weights.json").read_text(encoding="utf-8"))
        summary = json.loads((root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
        seed_summaries.append(summary)
        for method in ("three_channel_unconstrained", "three_channel_strictly_positive", "time_sky_validation_selected", "three_channel_equal_evidence"):
            weight_rows.append({"deployment": deployment, "seed": seed, "method": method, **selected[method]})
    pd.DataFrame(weight_rows).to_csv(root / "fusion_weights_per_seed.csv", index=False)

    rank_frames = []
    for seed in seeds:
        for scope in ("strict_h1l1_bbh", "all_catalog"):
            for method in ("three_channel_unconstrained", "three_channel_strictly_positive", "three_channel_equal_evidence", "time_sky_validation_selected"):
                frame = pd.read_parquet(root / f"seed_{seed}" / "results" / f"real_pair_scores_{scope}_{method}.parquet")
                keep = frame[["event_i", "event_j", "rank", "final_score", "waveform_available", "strict_h1l1_bbh_pair", "pair_has_ood"]].copy()
                keep["seed"] = seed
                keep["scope"] = scope
                keep["method"] = method
                keep["pair_key"] = keep.apply(lambda r: "--".join(sorted((str(r.event_i), str(r.event_j)))), axis=1)
                rank_frames.append(keep)
    all_ranks = pd.concat(rank_frames, ignore_index=True)
    all_ranks.to_parquet(root / "real_candidate_ranks_all_seeds.parquet", index=False)
    stability = all_ranks.groupby(["scope", "method", "pair_key"], as_index=False).agg(
        event_i=("event_i", "first"),
        event_j=("event_j", "first"),
        median_rank=("rank", "median"),
        q25_rank=("rank", lambda x: float(np.quantile(x, 0.25))),
        q75_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
        min_rank=("rank", "min"),
        max_rank=("rank", "max"),
        top10_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 10))),
        top20_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 20))),
        waveform_available=("waveform_available", "all"),
        pair_has_ood=("pair_has_ood", "any"),
    ).sort_values(["scope", "method", "median_rank", "q75_rank", "min_rank"])
    stability.to_csv(root / "real_candidate_rank_stability.csv", index=False)
    write_json(
        root / "deployment_summary.json",
        {
            "deployment": deployment,
            "seeds": seeds,
            "all_seed_deployments_passed": bool(all(x["deployment_passed"] for x in seed_summaries)),
            "gate1": [x["waveform_gate1"] for x in seed_summaries],
            "real_deployment_audits": [x["real_deployment_audit"] for x in seed_summaries],
            "strict_catalog_interpretation": "Strict complete H1-L1, non-OOD BBH event pairs; no silent zero filling.",
            "all_catalog_interpretation": "Supplementary catalog; missing waveform evidence is neutral log evidence zero and explicitly flagged.",
            "real_candidate_interpretation": "Catalog triage ranks only; no lensing detection claim and PE consistency is evaluated independently downstream.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--deployments", nargs="+", choices=["GWTC3", "GWTC4"], default=["GWTC3", "GWTC4"])
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    if not args.materialize_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing formal full-24-s multi-seed training")
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_config.json",
        {
            "protocol": "real_noise_injection_v3_physical_full24",
            "deployments": args.deployments,
            "seeds": args.seeds,
            "samples_per_family": args.samples_per_family,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "bootstrap_draws": args.bootstrap_draws,
            "channels": ["calibrated waveform log evidence", "run-conditioned time-delay log likelihood ratio", "HEALPix common-source sky log Bayes factor"],
            "snr_amplitude_in_final_score": False,
            "primary_real_catalog": "strict complete H1-L1 and non-OOD BBH subset",
            "all_catalog_missing_waveform": "neutral log evidence 0, supplementary only",
            "validation_test_policy": "source-system-disjoint; fusion selected on validation only; held-out test used once",
            "real_catalog_not_used_for_tuning": True,
            "pair_direction_policy": "All corrected evidence channels are symmetric, so directed max and mean are identical; both audit columns are retained.",
        },
    )
    for deployment in args.deployments:
        shared = args.out_root / deployment.lower() / "shared"
        schedule = build_live_schedule(deployment, SOURCES[deployment], shared)
        build_noise_bank(SOURCES[deployment], shared, 20260721 + (0 if deployment == "GWTC3" else 1000))
        build_sky_template_library(SOURCES[deployment], shared)
        raw_delays, _, _ = liao_delay_ratio_samples()
        build_time_calibration(shared, schedule, raw_delays, 20260722 + (0 if deployment == "GWTC3" else 1000))
        completed = []
        for seed in args.seeds:
            seed_dir = args.out_root / deployment.lower() / f"seed_{seed}"
            prepare_seed_layout(seed_dir, SOURCES[deployment])
            if args.materialize_only:
                materialize_compact_dataset(seed_dir, shared, deployment, seed + 100, args.samples_per_family)
            else:
                run_seed(deployment, seed, args.out_root, args.samples_per_family, args.epochs, args.batch_size, args.bootstrap_draws)
                completed.append(seed)
                aggregate_deployment(args.out_root, deployment, completed)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
