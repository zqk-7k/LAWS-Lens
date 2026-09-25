from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import (
    SECONDS_PER_DAY,
    default_run_dir,
    empirical_hist_lr,
    ensure_run_dirs,
    pair_indices,
    utc_now,
    write_json,
)


def load_primary_events(run_dir: Path) -> pd.DataFrame:
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    events = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)].copy()
    return events.sort_values("gps_time").reset_index(drop=True)


def liao_delay_snr_samples() -> tuple[np.ndarray, np.ndarray]:
    liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")
    cfg = liao.LIAO_PRIOR_CONFIG["LIGO"]
    delays, ratios = liao.extract_liao_delay_snr_pairs(cfg["image_csv"], cfg["snr_threshold"])
    delays = np.asarray(delays, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    mask = np.isfinite(delays) & np.isfinite(ratios) & (delays > 0) & (ratios > 0)
    if mask.sum() < 10:
        raise RuntimeError("Liao delay/SNR-ratio prior did not yield enough usable samples")
    return delays[mask], ratios[mask]


def log_bins_for(values: np.ndarray, signal: np.ndarray, min_floor: float, n_bins: int) -> np.ndarray:
    merged = np.concatenate([np.asarray(values, dtype=np.float64), np.asarray(signal, dtype=np.float64)])
    merged = merged[np.isfinite(merged) & (merged > 0)]
    lo = max(float(np.nanmin(merged)), min_floor)
    hi = max(float(np.nanmax(merged)), lo * 1.01)
    return np.geomspace(lo, hi * 1.001, n_bins + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--time-bins", type=int, default=80)
    parser.add_argument("--snr-bins", type=int, default=60)
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    events = load_primary_events(run_dir)
    sky = pd.read_parquet(run_dir / "features" / "real_sky_overlap.parquet")
    n = len(events)
    rows, cols = pair_indices(n)
    if len(sky) != len(rows):
        raise RuntimeError(f"Sky-overlap row mismatch: sky={len(sky)} expected={len(rows)}")

    gps = events["gps_time"].to_numpy(dtype=np.float64)
    snr = events["network_snr"].to_numpy(dtype=np.float64)
    delta_t_days = np.abs(gps[rows] - gps[cols]) / SECONDS_PER_DAY
    snr_ratio = np.maximum(snr[rows], snr[cols]) / np.maximum(np.minimum(snr[rows], snr[cols]), 1e-6)

    lensed_delay_days, lensed_snr_ratio = liao_delay_snr_samples()
    time_bins = log_bins_for(delta_t_days, lensed_delay_days, min_floor=1e-4, n_bins=args.time_bins)
    snr_bins = log_bins_for(snr_ratio, lensed_snr_ratio, min_floor=1.0, n_bins=args.snr_bins)

    time_score = empirical_hist_lr(delta_t_days, lensed_delay_days, delta_t_days, time_bins)
    snr_score = empirical_hist_lr(snr_ratio, lensed_snr_ratio, snr_ratio, snr_bins)
    sky_score = sky["sky_log_cosine_overlap"].to_numpy(dtype=np.float64).astype(np.float32)

    out = pd.DataFrame({
        "idx_i": rows.astype(np.int32),
        "idx_j": cols.astype(np.int32),
        "event_i": events["event_name"].to_numpy()[rows],
        "event_j": events["event_name"].to_numpy()[cols],
        "run_i": events["run"].to_numpy()[rows],
        "run_j": events["run"].to_numpy()[cols],
        "catalog_i": events["catalog"].to_numpy()[rows],
        "catalog_j": events["catalog"].to_numpy()[cols],
        "gps_i": gps[rows],
        "gps_j": gps[cols],
        "delta_t_days": delta_t_days.astype(np.float64),
        "network_snr_i": snr[rows],
        "network_snr_j": snr[cols],
        "snr_ratio": snr_ratio.astype(np.float64),
        "time_score": time_score,
        "snr_score": snr_score,
        "delay_snr_lr_score": (time_score + snr_score).astype(np.float32),
        "sky_overlap": sky["raw_posterior_overlap"].to_numpy(dtype=np.float64),
        "sky_cosine_overlap": sky["cosine_overlap"].to_numpy(dtype=np.float64),
        "sky_score": sky_score,
        "angular_sep_map_deg": sky["angular_sep_map_deg"].to_numpy(dtype=np.float64),
        "detectors_i": events["detectors_available"].fillna("").to_numpy()[rows],
        "detectors_j": events["detectors_available"].fillna("").to_numpy()[cols],
    })
    out_path = run_dir / "features" / "real_pair_observable_features.parquet"
    out.to_parquet(out_path, index=False)
    out.to_parquet(run_dir / "real_pair_features.parquet", index=False)

    calib = {
        "generated_at_utc": utc_now(),
        "n_events": int(n),
        "n_pairs": int(len(out)),
        "time_lr": {
            "signal_prior": "Liao LIGO simulated lensed image delay samples from existing project prior",
            "background": "all unordered primary real-catalog event pairs",
            "bins": time_bins.tolist(),
            "lensed_samples": int(len(lensed_delay_days)),
        },
        "snr_lr": {
            "signal_prior": "Liao LIGO lensed image SNR-ratio samples from existing project prior",
            "background": "all unordered primary real-catalog event SNR ratios",
            "bins": snr_bins.tolist(),
            "lensed_samples": int(len(lensed_snr_ratio)),
        },
        "sky_score": "log(cosine overlap) from real HEALPix posterior maps; no A90 surrogate",
        "output": str(out_path),
    }
    write_json(run_dir / "features" / "observable_feature_calibration.json", calib)
    print(json.dumps({
        "output": str(out_path),
        "n_events": int(n),
        "n_pairs": int(len(out)),
        "time_score_quantiles": {str(q): float(out["time_score"].quantile(q)) for q in [0.0, 0.5, 0.9, 0.99, 1.0]},
        "sky_score_quantiles": {str(q): float(out["sky_score"].quantile(q)) for q in [0.0, 0.5, 0.9, 0.99, 1.0]},
        "snr_score_quantiles": {str(q): float(out["snr_score"].quantile(q)) for q in [0.0, 0.5, 0.9, 0.99, 1.0]},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

