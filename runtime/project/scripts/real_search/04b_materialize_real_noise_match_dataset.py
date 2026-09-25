from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import default_run_dir, ensure_run_dirs, utc_now, write_json


SOURCE_ROOT = Path("data_generation/ligo_et_snr_like_2000_matchroots/LIGO")
TARGET_LEN = 98304
SAMPLE_RATE = 4096.0
CHANNELS = ("H1", "L1")


def zscore_channelwise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mu) * 1e-6, 1e-30)
    return ((x - mu) / np.maximum(sd, floor)).astype(np.float32, copy=False)


def load_strain_file(path: Path) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as h5:
        strain = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
        strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
        gps_start = float(np.asarray(h5["meta/GPSstart"][()]).item())
        duration = float(np.asarray(h5["meta/Duration"][()]).item())
    return strain, gps_start, duration


class NoiseCache:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.cache: dict[str, tuple[np.ndarray, float, float]] = {}

    def get(self, rel_path: str) -> tuple[np.ndarray, float, float]:
        if rel_path not in self.cache:
            self.cache[rel_path] = load_strain_file(self.run_dir / rel_path)
        return self.cache[rel_path]


def paired_noise_segments(run_dir: Path) -> pd.DataFrame:
    inj_dir = run_dir / "data" / "real_noise_injections"
    offsource_path = inj_dir / "offsource_noise_segments.parquet"
    meta_path = inj_dir / "metadata.parquet"
    if offsource_path.exists():
        existing = pd.read_parquet(offsource_path)
        if {"H1_path", "L1_path", "segment_start", "segment_end"}.issubset(existing.columns):
            return existing.reset_index(drop=True)
    meta = pd.read_parquet(meta_path)
    rows = []
    for (event, kind), group in meta.groupby(["event_name", "segment_kind"]):
        dets = {row.detector: row for row in group.itertuples(index=False)}
        if not all(det in dets for det in CHANNELS):
            continue
        start = max(float(dets["H1"].segment_start), float(dets["L1"].segment_start))
        end = min(float(dets["H1"].segment_end), float(dets["L1"].segment_end))
        if end - start < 24.0:
            continue
        rows.append({
            "event_name": event,
            "segment_kind": kind,
            "segment_start": start,
            "segment_end": end,
            "usable_duration_s": end - start,
            "H1_path": dets["H1"].strain_path,
            "L1_path": dets["L1"].strain_path,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No paired H1/L1 off-source noise segments available")
    return out.reset_index(drop=True)


def draw_noise_pair(cache: NoiseCache, segments: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    seg = segments.iloc[int(rng.integers(0, len(segments)))]
    latest_start = float(seg.segment_end) - TARGET_LEN / SAMPLE_RATE
    t0 = float(rng.uniform(float(seg.segment_start), latest_start))
    chans = []
    for det in CHANNELS:
        data, gps_start, _ = cache.get(str(seg[f"{det}_path"]))
        i0 = int(round((t0 - gps_start) * SAMPLE_RATE))
        i0 = max(0, min(i0, len(data) - TARGET_LEN))
        chans.append(data[i0:i0 + TARGET_LEN])
    noise = np.stack(chans, axis=0)
    return zscore_channelwise(noise), {
        "noise_event": str(seg.event_name),
        "noise_segment_kind": str(seg.segment_kind),
        "noise_start_gps": float(t0),
    }


def load_signal_array(path: Path, n: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.shape[-1] != TARGET_LEN:
        raise ValueError(f"{path} has length {arr.shape[-1]}, expected {TARGET_LEN}")
    return arr[:n]


def inject_signal(signal: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    clean = zscore_channelwise(signal.astype(np.float32, copy=False))
    scale = float(rng.lognormal(mean=-0.05, sigma=0.35))
    mixed = zscore_channelwise(noise + scale * clean)
    return mixed.astype(np.float32, copy=False), scale


def build_family(
    family: str,
    source_dir: Path,
    out_dir: Path,
    n: int,
    segments: pd.DataFrame,
    cache: NoiseCache,
    rng: np.random.Generator,
) -> list[dict]:
    sig1 = load_signal_array(source_dir / f"{family}_h_strain_1.npy", n)
    sig2 = load_signal_array(source_dir / f"{family}_h_strain_2.npy", n)
    data1 = np.empty((n, 2, TARGET_LEN), dtype=np.float32)
    data2 = np.empty((n, 2, TARGET_LEN), dtype=np.float32)
    h1 = np.empty_like(data1)
    h2 = np.empty_like(data2)
    rows = []
    for idx in range(n):
        noise_a, meta_a = draw_noise_pair(cache, segments, rng)
        noise_b, meta_b = draw_noise_pair(cache, segments, rng)
        h1[idx] = zscore_channelwise(sig1[idx].astype(np.float32, copy=False))
        h2[idx] = zscore_channelwise(sig2[idx].astype(np.float32, copy=False))
        data1[idx], scale_a = inject_signal(h1[idx], noise_a, rng)
        data2[idx], scale_b = inject_signal(h2[idx], noise_b, rng)
        rows.append({
            "family": family,
            "sample_index": idx,
            "pair_id": f"{family}_{idx:05d}",
            "split": "train" if idx < int(0.7 * n) else ("val" if idx < int(0.85 * n) else "test"),
            "signal_source": "ligo_et_snr_like_2000 clean h_strain",
            "noise_source_a": meta_a["noise_event"],
            "noise_source_b": meta_b["noise_event"],
            "noise_segment_kind_a": meta_a["noise_segment_kind"],
            "noise_segment_kind_b": meta_b["noise_segment_kind"],
            "noise_start_gps_a": meta_a["noise_start_gps"],
            "noise_start_gps_b": meta_b["noise_start_gps"],
            "signal_scale_a": scale_a,
            "signal_scale_b": scale_b,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{family}_data_strain_1.npy", data1)
    np.save(out_dir / f"{family}_data_strain_2.npy", data2)
    np.save(out_dir / f"{family}_h_strain_1.npy", h1)
    np.save(out_dir / f"{family}_h_strain_2.npy", h2)
    np.save(out_dir / f"{family}_optimal_SNR_network_1.npy", np.ones(n, dtype=np.float32))
    np.save(out_dir / f"{family}_optimal_SNR_network_2.npy", np.ones(n, dtype=np.float32))
    return rows


def build_unlensed(
    source_dir: Path,
    out_dir: Path,
    n: int,
    segments: pd.DataFrame,
    cache: NoiseCache,
    rng: np.random.Generator,
) -> list[dict]:
    sig = load_signal_array(source_dir / "unlensed_h_strain.npy", n)
    data = np.empty((n, 2, TARGET_LEN), dtype=np.float32)
    clean_arr = np.empty_like(data)
    rows = []
    for idx in range(n):
        noise, meta = draw_noise_pair(cache, segments, rng)
        clean = zscore_channelwise(sig[idx].astype(np.float32, copy=False))
        mixed, scale = inject_signal(clean, noise, rng)
        clean_arr[idx] = clean
        data[idx] = mixed
        rows.append({
            "family": "unlensed",
            "sample_index": idx,
            "pair_id": f"unlensed_{idx:05d}",
            "split": "train" if idx < int(0.7 * n) else ("val" if idx < int(0.85 * n) else "test"),
            "signal_source": "ligo_et_snr_like_2000 clean h_strain",
            "noise_source_a": meta["noise_event"],
            "noise_source_b": "",
            "noise_segment_kind_a": meta["noise_segment_kind"],
            "noise_segment_kind_b": "",
            "noise_start_gps_a": meta["noise_start_gps"],
            "noise_start_gps_b": np.nan,
            "signal_scale_a": scale,
            "signal_scale_b": np.nan,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "unlensed_data_strain.npy", data)
    np.save(out_dir / "unlensed_h_strain.npy", clean_arr)
    np.save(out_dir / "unlensed_optimal_SNR_network.npy", np.ones(n, dtype=np.float32))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    match_root = run_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    segments = paired_noise_segments(run_dir)
    cache = NoiseCache(run_dir)
    rng = np.random.default_rng(args.seed)
    rows = []
    rows.extend(build_family("SIS", SOURCE_ROOT / "SIS_data_0222", match_root / "SIS_data_0222", args.samples_per_family, segments, cache, rng))
    rows.extend(build_family("PM", SOURCE_ROOT / "PM_data_0222", match_root / "PM_data_0222", args.samples_per_family, segments, cache, rng))
    rows.extend(build_unlensed(SOURCE_ROOT / "Unlensed_data_0222", match_root / "Unlensed_data_0222", args.samples_per_family, segments, cache, rng))
    meta = pd.DataFrame(rows)
    meta.to_parquet(run_dir / "data" / "real_noise_injections" / "metadata.parquet", index=False)
    segments.to_parquet(run_dir / "data" / "real_noise_injections" / "offsource_noise_segments.parquet", index=False)
    summary = {
        "generated_at_utc": utc_now(),
        "match_root": str(match_root),
        "samples_per_family": int(args.samples_per_family),
        "target_len": TARGET_LEN,
        "sample_rate_hz": SAMPLE_RATE,
        "detectors": list(CHANNELS),
        "paired_noise_segments": int(len(segments)),
        "metadata_rows": int(len(meta)),
        "split_counts": meta["split"].value_counts().to_dict(),
        "method_note": "Clean SIS/PM/unlensed simulated strain is z-scored channelwise and injected into real GWOSC off-source H1/L1 strain segments. This is a domain-matched Gate-1 dataset, not a population-rate injection campaign.",
    }
    write_json(run_dir / "data" / "real_noise_injections" / "materialized_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
