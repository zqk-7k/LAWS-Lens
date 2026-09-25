from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.aux_priors import a90_to_sigma_rad, observed_sky_pair_features
from scripts.gwtc import config

liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")


IMPORT_LIST = [
    "matchgw.aux_priors.a90_to_sigma_rad",
    "matchgw.aux_priors.observed_sky_pair_features",
    "scripts.experiments.88_liao_realistic_p1_p2_rerank.fit_time_lr_from_liao",
    "scripts.experiments.88_liao_realistic_p1_p2_rerank.time_lr_score_matrix",
    "scripts.experiments.88_liao_realistic_p1_p2_rerank.raw_snr_ratio_score_matrix",
]


def row_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values, dtype=np.float64)
    mu = float(values[finite].mean())
    sd = float(values[finite].std())
    return (values - mu) / max(sd, 1e-8)


def load_observables(catalog: str) -> pd.DataFrame:
    path = config.GWTC3_OBSERVABLES_CSV if catalog == "gwtc3" else config.GWTC5_OBSERVABLES_CSV
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ["event_name", "gps_trigger_time", "ra_median", "dec_median", "sky_area_90_deg2", "network_snr"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def build_features(catalog: str) -> pd.DataFrame:
    obs = load_observables(catalog)
    n = len(obs)
    time_obs = pd.DataFrame({
        "trigger_time_obs": obs["gps_trigger_time"].to_numpy(dtype=np.float64),
        "snr": obs["network_snr"].to_numpy(dtype=np.float64),
    })
    sky_obs = pd.DataFrame({
        "ra_obs": obs["ra_median"].to_numpy(dtype=np.float64),
        "dec_obs": obs["dec_median"].to_numpy(dtype=np.float64),
        "sky_area90_deg2": obs["sky_area_90_deg2"].to_numpy(dtype=np.float64),
    })
    sky_obs["sky_sigma_rad"] = a90_to_sigma_rad(sky_obs["sky_area90_deg2"].to_numpy(dtype=np.float64))

    gt_none = np.full(n, -1, dtype=np.int32)
    prior = liao.fit_time_lr_from_liao("LIGO", time_obs, gt_none)
    time_score = liao.time_lr_score_matrix(time_obs, prior)
    snr_score = liao.raw_snr_ratio_score_matrix(time_obs)
    sky = observed_sky_pair_features(sky_obs)

    rows, cols = np.triu_indices(n, k=1)
    delta_t_days = np.abs(time_obs["trigger_time_obs"].to_numpy()[rows] - time_obs["trigger_time_obs"].to_numpy()[cols]) / 86400.0
    out = pd.DataFrame({
        "event_i": obs["event_name"].to_numpy()[rows],
        "event_j": obs["event_name"].to_numpy()[cols],
        "delta_t_days": delta_t_days,
        "time_score": time_score[rows, cols],
        "sky_norm_sep": sky["sky_norm_sep"][rows, cols],
        "sky_step_weight": sky["sky_step_weight"][rows, cols],
        "sky_log_overlap": sky["sky_log_overlap"][rows, cols],
        "snr_ratio": snr_score[rows, cols],
        "ang_sep_deg": np.degrees(sky["sky_sep_obs"][rows, cols]),
    })
    out["combined_time_sky_z"] = row_z(out["time_score"].to_numpy()) + row_z(out["sky_log_overlap"].to_numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", choices=["gwtc3", "gwtc5"], required=True)
    args = parser.parse_args()

    print("Reused imports:", ", ".join(IMPORT_LIST), flush=True)
    start = time.perf_counter()
    features = build_features(args.catalog)
    elapsed = time.perf_counter() - start
    out_path = config.DATA_DIR / f"{args.catalog}_pair_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(features)} wall_time_s={elapsed:.2f}", flush=True)
    print("10 most sky-consistent pairs:", flush=True)
    print(
        features.sort_values(["sky_norm_sep", "ang_sep_deg"], ascending=[True, True])
        .head(10)[["event_i", "event_j", "sky_norm_sep", "ang_sep_deg", "sky_log_overlap"]]
        .to_string(index=False),
        flush=True,
    )
    print("10 highest combined time+sky pairs:", flush=True)
    print(
        features.sort_values("combined_time_sky_z", ascending=False)
        .head(10)[["event_i", "event_j", "delta_t_days", "time_score", "sky_norm_sep", "combined_time_sky_z"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
