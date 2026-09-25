from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import default_run_dir, ensure_run_dirs, row_z_matrix, utc_now, write_json


OBS_CHANNELS = ("time_score", "sky_score", "snr_score")


def build_matrix(features: pd.DataFrame, n: int, column: str) -> np.ndarray:
    mat = np.full((n, n), np.nan, dtype=np.float64)
    i = features["idx_i"].to_numpy(dtype=np.int32)
    j = features["idx_j"].to_numpy(dtype=np.int32)
    v = features[column].to_numpy(dtype=np.float64)
    mat[i, j] = v
    mat[j, i] = v
    return mat


def attach_rowz(out: pd.DataFrame, features: pd.DataFrame, n: int, column: str) -> np.ndarray:
    mat = build_matrix(features, n, column)
    z = row_z_matrix(mat)
    i = features["idx_i"].to_numpy(dtype=np.int32)
    j = features["idx_j"].to_numpy(dtype=np.int32)
    out[f"{column}_rowz_i_to_j"] = z[i, j]
    out[f"{column}_rowz_j_to_i"] = z[j, i]
    out[f"{column}_rowz_mean"] = 0.5 * (z[i, j] + z[j, i])
    return z


def detector_coverage(row: pd.Series) -> str:
    return f"{row.get('detectors_i', '')} | {row.get('detectors_j', '')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--use-waveform", action="store_true", help="Use exploratory waveform score in final fusion. Default keeps weight zero.")
    parser.add_argument("--waveform-weight", type=float, default=1.0)
    parser.add_argument("--shortlist-size", type=int, default=100)
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    features = pd.read_parquet(run_dir / "features" / "real_pair_observable_features.parquet")
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)].copy()
    primary = primary.sort_values("gps_time").reset_index(drop=True)
    n = len(primary)

    out = features.copy()
    z = {name: attach_rowz(out, features, n, name) for name in OBS_CHANNELS}
    i = features["idx_i"].to_numpy(dtype=np.int32)
    j = features["idx_j"].to_numpy(dtype=np.int32)
    directed_obs = sum(z[name] for name in OBS_CHANNELS)
    out["observable_score_i_to_j"] = directed_obs[i, j]
    out["observable_score_j_to_i"] = directed_obs[j, i]
    out["observable_score_mean"] = 0.5 * (out["observable_score_i_to_j"] + out["observable_score_j_to_i"])
    out["observable_score_max"] = np.maximum(out["observable_score_i_to_j"], out["observable_score_j_to_i"])

    waveform_weight = 0.0
    waveform_reason = "waveform weight forced to 0 by default; real-event waveform score is exploratory unless explicitly enabled"
    sim_path = run_dir / "features" / "real_waveform_similarity.parquet"
    if sim_path.exists():
        sim = pd.read_parquet(sim_path)
        keep = ["idx_i", "idx_j", "waveform_available", "waveform_score", "waveform_score_sis", "waveform_score_pm"]
        out = out.merge(sim[keep], on=["idx_i", "idx_j"], how="left")
        if bool(args.use_waveform):
            valid = out["waveform_score"].notna()
            wf_features = out[["idx_i", "idx_j", "waveform_score"]].copy()
            wf_features["waveform_score"] = wf_features["waveform_score"].where(valid, np.nan)
            wf_z = attach_rowz(out, wf_features, n, "waveform_score")
            directed_obs = directed_obs + float(args.waveform_weight) * wf_z
            waveform_weight = float(args.waveform_weight)
            waveform_reason = "waveform score included because --use-waveform was provided"
    else:
        out["waveform_available"] = False
        out["waveform_score"] = np.nan
        out["waveform_score_sis"] = np.nan
        out["waveform_score_pm"] = np.nan
        waveform_reason = "no real_waveform_similarity.parquet found"

    out["final_score_i_to_j"] = directed_obs[i, j]
    out["final_score_j_to_i"] = directed_obs[j, i]
    out["final_score_mean"] = 0.5 * (out["final_score_i_to_j"] + out["final_score_j_to_i"])
    out["final_score"] = np.maximum(out["final_score_i_to_j"], out["final_score_j_to_i"])
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["empirical_catalog_tail_p"] = out["rank"] / float(len(out))

    out.to_parquet(run_dir / "results" / "real_pair_scores.parquet", index=False)
    out.to_parquet(run_dir / "real_pair_scores.parquet", index=False)

    cols = [
        "rank", "event_i", "event_j", "final_score", "observable_score_max",
        "waveform_score", "waveform_available",
        "time_score", "sky_score", "snr_score",
        "time_score_rowz_mean", "sky_score_rowz_mean", "snr_score_rowz_mean",
        "delta_t_days", "sky_cosine_overlap", "sky_overlap", "angular_sep_map_deg",
        "snr_ratio", "network_snr_i", "network_snr_j",
        "detectors_i", "detectors_j", "empirical_catalog_tail_p",
    ]
    shortlist = out.head(int(args.shortlist_size))[cols].copy()
    shortlist["detector_coverage"] = shortlist.apply(detector_coverage, axis=1)
    shortlist["notes"] = "candidate shortlist for Bayesian follow-up; no lensing claim"
    shortlist.to_csv(run_dir / "results" / "candidate_shortlist.csv", index=False)
    shortlist.to_csv(run_dir / "candidate_shortlist.csv", index=False)

    weights = {
        "generated_at_utc": utc_now(),
        "fusion": "real catalog fusion",
        "weights": {"time_score": 1.0, "sky_score": 1.0, "snr_score": 1.0, "waveform_score": waveform_weight},
        "uses_et3_weights": False,
        "uses_test_label_tuning": False,
        "waveform_policy": waveform_reason,
        "final_score": "max directed sum of row-standardized channel scores",
    }
    gate_path = run_dir / "results" / "waveform_gate1.json"
    if gate_path.exists():
        with gate_path.open("r", encoding="utf-8") as f:
            weights["waveform_gate1"] = json.load(f)
    write_json(run_dir / "results" / "channel_weights.json", weights)

    summary = {
        "generated_at_utc": utc_now(),
        "n_events": int(n),
        "n_pairs": int(len(out)),
        "n_waveform_pairs_available": int(out["waveform_available"].fillna(False).sum()),
        "waveform_weight": waveform_weight,
        "top20": shortlist.head(20)[["rank", "event_i", "event_j", "final_score", "delta_t_days", "sky_cosine_overlap", "snr_ratio"]].to_dict(orient="records"),
    }
    write_json(run_dir / "results" / "final_fusion_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
