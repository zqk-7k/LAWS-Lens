from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import default_run_dir, ensure_run_dirs, row_z_matrix, utc_now, write_json


CHANNELS = ("time_score", "sky_score", "snr_score")
LVK_CHECK_PAIRS = [
    ("GW170104", "GW170814", "LVK-reported historical lensing-candidate pair; not confirmed as lensed"),
]


def build_channel_matrix(features: pd.DataFrame, n: int, column: str) -> np.ndarray:
    mat = np.full((n, n), np.nan, dtype=np.float64)
    i = features["idx_i"].to_numpy(dtype=np.int32)
    j = features["idx_j"].to_numpy(dtype=np.int32)
    v = features[column].to_numpy(dtype=np.float64)
    mat[i, j] = v
    mat[j, i] = v
    return mat


def score_pairs(features: pd.DataFrame, n: int) -> tuple[pd.DataFrame, dict[str, float]]:
    matrices = {name: build_channel_matrix(features, n, name) for name in CHANNELS}
    z = {name: row_z_matrix(mat) for name, mat in matrices.items()}
    directed = sum(z[name] for name in CHANNELS)
    i = features["idx_i"].to_numpy(dtype=np.int32)
    j = features["idx_j"].to_numpy(dtype=np.int32)
    out = features.copy()
    for name in CHANNELS:
        out[f"{name}_rowz_i_to_j"] = z[name][i, j]
        out[f"{name}_rowz_j_to_i"] = z[name][j, i]
        out[f"{name}_rowz_mean"] = 0.5 * (z[name][i, j] + z[name][j, i])
    out["observable_score_i_to_j"] = directed[i, j]
    out["observable_score_j_to_i"] = directed[j, i]
    out["observable_score_mean"] = 0.5 * (out["observable_score_i_to_j"] + out["observable_score_j_to_i"])
    out["observable_score_max"] = np.maximum(out["observable_score_i_to_j"], out["observable_score_j_to_i"])
    out["final_score"] = out["observable_score_max"]
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["empirical_catalog_tail_p"] = out["rank"] / float(len(out))
    weights = {name: 1.0 for name in CHANNELS}
    return out, weights


def detector_coverage(row: pd.Series) -> str:
    left = str(row.get("detectors_i", ""))
    right = str(row.get("detectors_j", ""))
    return f"{left} | {right}"


def make_shortlist(scores: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    cols = [
        "rank", "event_i", "event_j", "final_score",
        "time_score", "sky_score", "snr_score",
        "time_score_rowz_mean", "sky_score_rowz_mean", "snr_score_rowz_mean",
        "delta_t_days", "sky_cosine_overlap", "sky_overlap", "angular_sep_map_deg",
        "snr_ratio", "network_snr_i", "network_snr_j",
        "detectors_i", "detectors_j", "empirical_catalog_tail_p",
    ]
    out = scores.head(n)[cols].copy()
    out["detector_coverage"] = out.apply(detector_coverage, axis=1)
    out["notes"] = "candidate shortlist for Bayesian follow-up; no lensing claim"
    return out


def crosscheck(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b, note in LVK_CHECK_PAIRS:
        mask = ((scores["event_i"] == a) & (scores["event_j"] == b)) | ((scores["event_i"] == b) & (scores["event_j"] == a))
        if mask.any():
            row = scores.loc[mask].iloc[0]
            rows.append({
                "event_i": a,
                "event_j": b,
                "rank": int(row["rank"]),
                "final_score": float(row["final_score"]),
                "time_score": float(row["time_score"]),
                "sky_score": float(row["sky_score"]),
                "snr_score": float(row["snr_score"]),
                "delta_t_days": float(row["delta_t_days"]),
                "sky_cosine_overlap": float(row["sky_cosine_overlap"]),
                "angular_sep_map_deg": float(row["angular_sep_map_deg"]),
                "snr_ratio": float(row["snr_ratio"]),
                "empirical_catalog_tail_p": float(row["empirical_catalog_tail_p"]),
                "lvk_context": note,
            })
        else:
            rows.append({"event_i": a, "event_j": b, "rank": -1, "lvk_context": note})
    return pd.DataFrame(rows)


def save_figures(run_dir: Path, scores: pd.DataFrame, shortlist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(scores["final_score"], bins=55, color="#475467", alpha=0.88)
    ax.set_xlabel("observable-only final score", fontweight="bold")
    ax.set_ylabel("unordered event pairs", fontweight="bold")
    ax.set_title("Primary GWTC real-catalog observable score background")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(run_dir / "figures" / f"observable_score_hist.{ext}", dpi=300)
    plt.close(fig)

    top = shortlist.head(20).iloc[::-1]
    labels = [f"{r.event_i}-{r.event_j}" for r in top.itertuples()]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.barh(y, top["time_score_rowz_mean"], color="#2E90FA", label="time z")
    ax.barh(y, top["sky_score_rowz_mean"], left=top["time_score_rowz_mean"], color="#12B76A", label="sky z")
    left = top["time_score_rowz_mean"].to_numpy() + top["sky_score_rowz_mean"].to_numpy()
    ax.barh(y, top["snr_score_rowz_mean"], left=left, color="#F79009", label="SNR z")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("row-standardized channel contribution", fontweight="bold")
    ax.set_title("Top observable candidates for follow-up")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(run_dir / "figures" / f"top_candidate_forest.{ext}", dpi=300)
    plt.close(fig)


def write_report(run_dir: Path, events: pd.DataFrame, scores: pd.DataFrame, shortlist: pd.DataFrame, cross: pd.DataFrame) -> None:
    strain = pd.read_csv(run_dir / "data" / "strain_manifest.csv")
    sky = pd.read_csv(run_dir / "data" / "skymap_manifest.csv")
    primary_events = events[events["include_in_primary_search"] == True]
    h1l1 = strain[(strain["detector"].isin(["H1", "L1"])) & (strain["exists"] == True)]
    h1l1_events = h1l1.groupby("event_name")["detector"].nunique()
    h1l1_count = int((h1l1_events >= 2).sum())
    gw170104_rank = int(cross.iloc[0]["rank"]) if not cross.empty and "rank" in cross else -1
    lines = [
        "# Real GWTC Lensing Search Phase A-D Observable Baseline",
        "",
        f"Generated at: {utc_now()}",
        "",
        "This is a candidate-shortlist workflow for Bayesian follow-up. It does not claim that any real GWTC pair is lensed.",
        "",
        "## Data Scope",
        "",
        f"- Manifest events total: {len(events)}",
        f"- Primary scored events: {len(primary_events)}",
        f"- Primary HEALPix sky maps available: {int(sky[sky['include_in_primary_search'] == True]['sky_map_available'].sum())}",
        f"- Events with any local strain file: {int(strain.groupby('event_name')['exists'].any().sum())}",
        f"- Events with local H1+L1 strain files: {h1l1_count}",
        "- Primary scoring scope: O1-O3 confident BBH events with GWTC PE HDF5 HEALPix skymaps.",
        "- GWTC-5/O4 search rows are listed in the manifest as an extension, but not included in this primary score because they use search skymaps rather than full PE skymaps.",
        "",
        "## Scoring",
        "",
        "- time_score: empirical log likelihood ratio using the project Liao LIGO lensed-delay prior versus all real-catalog unordered pair delays.",
        "- sky_score: log cosine overlap of real HEALPix sky posterior maps.",
        "- snr_score: empirical log likelihood ratio using Liao lensed SNR-ratio prior versus all real-catalog unordered SNR ratios.",
        "- final_score: equal-weight sum of row-standardized time, sky, and SNR scores, aggregated by max directed score.",
        "",
        "## Top 20 Candidate Shortlist",
        "",
        shortlist.head(20)[["rank", "event_i", "event_j", "final_score", "delta_t_days", "sky_cosine_overlap", "snr_ratio", "empirical_catalog_tail_p"]].to_markdown(index=False),
        "",
        "## LVK Cross-check",
        "",
        cross.to_markdown(index=False),
        "",
        f"GW170104--GW170814 rank: {gw170104_rank}",
        "",
        "## Waveform Gate Readiness",
        "",
        "The observable-only A-D products are complete. Domain-matched waveform training is not ready yet because only a small subset of local on-source strain files is currently present; Phase E must first build/download off-source real-noise segments and a real-noise injection dataset.",
    ]
    (run_dir / "real_search_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    en = [
        "# Real GWTC Lensing Search Phase A-D Observable Baseline",
        "",
        "This run produces a candidate shortlist for Bayesian follow-up only; it does not claim a real lensed GW detection.",
        "",
        f"Primary scored events: {len(primary_events)}",
        f"Unordered primary pairs: {len(scores)}",
        f"GW170104--GW170814 rank: {gw170104_rank}",
        "",
        "Primary scope: O1-O3 confident BBH events with GWTC PE HDF5 HEALPix sky maps. GWTC-5/O4 search-map rows are manifest-only extension entries in this Phase A-D run.",
        "",
        "Outputs are under `data/`, `features/`, `results/`, and `figures/` in this run directory.",
    ]
    (run_dir / "real_search_report_en.md").write_text("\n".join(en) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)].copy()
    primary = primary.sort_values("gps_time").reset_index(drop=True)
    features = pd.read_parquet(run_dir / "features" / "real_pair_observable_features.parquet")
    scores, weights = score_pairs(features, len(primary))

    scores.to_parquet(run_dir / "results" / "observable_pair_scores.parquet", index=False)
    scores.to_parquet(run_dir / "results" / "real_pair_scores.parquet", index=False)
    scores.to_parquet(run_dir / "real_pair_scores.parquet", index=False)

    shortlist = make_shortlist(scores, n=100)
    shortlist.to_csv(run_dir / "results" / "observable_candidate_shortlist.csv", index=False)
    shortlist.to_csv(run_dir / "results" / "candidate_shortlist.csv", index=False)
    shortlist.to_csv(run_dir / "candidate_shortlist.csv", index=False)

    cross = crosscheck(scores)
    cross.to_csv(run_dir / "results" / "candidate_crosscheck_lvk.csv", index=False)

    quantiles = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    bg = pd.DataFrame({
        "metric": ["final_score_quantile"] * len(quantiles),
        "quantile": quantiles,
        "value": [float(scores["final_score"].quantile(q)) for q in quantiles],
        "n_pairs": len(scores),
    })
    bg.to_csv(run_dir / "results" / "background_null_summary.csv", index=False)

    weight_payload = {
        "generated_at_utc": utc_now(),
        "fusion": "observable-only equal-weight row-standardized channels",
        "weights": weights,
        "uses_test_label_tuning": False,
        "uses_et3_weights": False,
        "waveform_weight": 0.0,
    }
    write_json(run_dir / "results" / "channel_weights.json", weight_payload)

    save_figures(run_dir, scores, shortlist)
    write_report(run_dir, events, scores, shortlist, cross)

    reproduce = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "RUN_DIR=\"${1:-runs/real_gwtc_lensing_search_20260625}\"",
        "/root/miniconda3/bin/python scripts/real_search/00_build_event_manifest.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/01_build_healpix_overlap.py --run-dir \"$RUN_DIR\" --nside 512",
        "/root/miniconda3/bin/python scripts/real_search/02_build_observable_features.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/03_score_observable_baseline.py --run-dir \"$RUN_DIR\"",
    ]) + "\n"
    (run_dir / "reproduce.sh").write_text(reproduce, encoding="utf-8")
    (run_dir / "reproduce.sh").chmod(0o755)

    summary = {
        "generated_at_utc": utc_now(),
        "n_manifest_events": int(len(events)),
        "n_primary_events": int(len(primary)),
        "n_pairs": int(len(scores)),
        "top20_path": str(run_dir / "results" / "observable_candidate_shortlist.csv"),
        "gw170104_gw170814_rank": int(cross.iloc[0]["rank"]) if not cross.empty else -1,
        "top20": shortlist.head(20)[["rank", "event_i", "event_j", "final_score", "delta_t_days", "sky_cosine_overlap", "snr_ratio"]].to_dict(orient="records"),
    }
    write_json(run_dir / "results" / "phase_a_to_d_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

