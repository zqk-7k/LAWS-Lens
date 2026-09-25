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

from scripts.real_search.common import default_run_dir, ensure_run_dirs, utc_now, write_json


LVK_PAIRS = [
    ("GW170104", "GW170814", "LVK literature candidate pair; not confirmed as lensed"),
]


def find_pair(scores: pd.DataFrame, a: str, b: str) -> dict:
    mask = ((scores["event_i"] == a) & (scores["event_j"] == b)) | ((scores["event_i"] == b) & (scores["event_j"] == a))
    if not mask.any():
        return {"event_i": a, "event_j": b, "rank": -1, "status": "not_in_primary_pair_table"}
    row = scores.loc[mask].iloc[0]
    return {
        "event_i": a,
        "event_j": b,
        "rank": int(row["rank"]),
        "final_score": float(row["final_score"]),
        "observable_score_max": float(row.get("observable_score_max", np.nan)),
        "waveform_score": float(row.get("waveform_score", np.nan)) if pd.notna(row.get("waveform_score", np.nan)) else np.nan,
        "time_score": float(row["time_score"]),
        "sky_score": float(row["sky_score"]),
        "snr_score": float(row["snr_score"]),
        "delta_t_days": float(row["delta_t_days"]),
        "sky_cosine_overlap": float(row["sky_cosine_overlap"]),
        "angular_sep_map_deg": float(row["angular_sep_map_deg"]),
        "snr_ratio": float(row["snr_ratio"]),
        "empirical_catalog_tail_p": float(row["empirical_catalog_tail_p"]),
        "status": "ranked",
    }


def save_context_figure(run_dir: Path, scores: pd.DataFrame, cross: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(scores["final_score"], bins=60, color="#667085", alpha=0.85)
    for row in cross.itertuples(index=False):
        if getattr(row, "rank", -1) > 0 and pd.notna(getattr(row, "final_score", np.nan)):
            ax.axvline(row.final_score, color="#D92D20", lw=1.4, label=f"{row.event_i}-{row.event_j}")
    ax.set_xlabel("final score", fontweight="bold")
    ax.set_ylabel("real-catalog unordered pairs", fontweight="bold")
    ax.set_title("LVK candidate-pair rank context")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / "figures" / f"lvk_candidate_rank_context.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    args = parser.parse_args()
    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    scores = pd.read_parquet(run_dir / "results" / "real_pair_scores.parquet")
    rows = []
    for a, b, note in LVK_PAIRS:
        row = find_pair(scores, a, b)
        row["lvk_context"] = note
        row["interpretation"] = "candidate shortlist context only; not a lensing claim"
        rows.append(row)
    cross = pd.DataFrame(rows)
    cross.to_csv(run_dir / "results" / "candidate_crosscheck_lvk.csv", index=False)
    cross.to_csv(run_dir / "candidate_crosscheck_lvk.csv", index=False)
    save_context_figure(run_dir, scores, cross)
    summary = {"generated_at_utc": utc_now(), "rows": int(len(cross)), "output": str(run_dir / "results" / "candidate_crosscheck_lvk.csv")}
    write_json(run_dir / "results" / "lvk_crosscheck_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
