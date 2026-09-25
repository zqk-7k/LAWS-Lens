from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.gwtc import config


CANDIDATE_A = "GW170104"
CANDIDATE_B = "GW170814"
CAMPAILLA_REFERENCE_KEPT = 185
CAMPAILLA_REFERENCE_TOTAL = 3655


def load_features(catalog: str) -> pd.DataFrame:
    path = config.DATA_DIR / f"{catalog}_pair_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def candidate_row(features: pd.DataFrame) -> pd.Series:
    mask = (
        ((features["event_i"] == CANDIDATE_A) & (features["event_j"] == CANDIDATE_B))
        | ((features["event_i"] == CANDIDATE_B) & (features["event_j"] == CANDIDATE_A))
    )
    if not mask.any():
        raise RuntimeError(f"Missing candidate pair {CANDIDATE_A}-{CANDIDATE_B}")
    return features[mask].iloc[0]


def descending_rank(values: pd.Series, score: float) -> int:
    return int((values > score).sum() + 1)


def ascending_rank(values: pd.Series, score: float) -> int:
    return int((values < score).sum() + 1)


def candidate_rank_summary(features: pd.DataFrame) -> dict:
    row = candidate_row(features)
    total = len(features)
    summary = {
        "pair": f"{CANDIDATE_A}-{CANDIDATE_B}",
        "total_pairs": total,
        "sky_step_score": float(row["sky_step_weight"]),
        "sky_step_rank": descending_rank(features["sky_step_weight"], float(row["sky_step_weight"])),
        "sky_log_overlap": float(row["sky_log_overlap"]),
        "sky_log_overlap_rank": descending_rank(features["sky_log_overlap"], float(row["sky_log_overlap"])),
        "sky_norm_sep": float(row["sky_norm_sep"]),
        "sky_norm_sep_rank": ascending_rank(features["sky_norm_sep"], float(row["sky_norm_sep"])),
        "time_score": float(row["time_score"]),
        "time_rank": descending_rank(features["time_score"], float(row["time_score"])),
        "combined_score": float(row["combined_time_sky_z"]),
        "combined_rank": descending_rank(features["combined_time_sky_z"], float(row["combined_time_sky_z"])),
        "delta_t_days": float(row["delta_t_days"]),
        "ang_sep_deg": float(row["ang_sep_deg"]),
    }
    sky_high = summary["sky_step_rank"] <= max(1, int(0.1 * total))
    time_low = summary["time_rank"] > int(0.5 * total)
    summary["narrative"] = (
        "PASS" if sky_high and time_low else "EXPLAIN"
    )
    summary["narrative_text"] = (
        "sky-only ranks high and time-only ranks low"
        if sky_high and time_low
        else "time-only ranks low as expected, but this observable-only sky-center/A90 proxy does not rank the pair sky-high"
    )
    return summary


def shortlist_table(features: pd.DataFrame) -> dict:
    scores = features["combined_time_sky_z"].to_numpy(dtype=float)
    total = len(scores)
    reference_fraction = CAMPAILLA_REFERENCE_KEPT / CAMPAILLA_REFERENCE_TOTAL
    top_equiv = int(round(reference_fraction * total))
    sorted_scores = np.sort(scores)[::-1]
    threshold_equiv = float(sorted_scores[max(0, min(total - 1, top_equiv - 1))])
    threshold_185 = float(sorted_scores[max(0, min(total - 1, CAMPAILLA_REFERENCE_KEPT - 1))])
    return {
        "total_pairs": total,
        "reference_fraction": reference_fraction,
        "equivalent_top_count": top_equiv,
        "equivalent_threshold": threshold_equiv,
        "top185_threshold": threshold_185,
        "top185_fraction": CAMPAILLA_REFERENCE_KEPT / total,
    }


def save_figures(gwtc3: pd.DataFrame, gwtc5: pd.DataFrame | None, summary: dict) -> None:
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["sky_step", "time", "combined"]
    ranks = [summary["sky_step_rank"], summary["time_rank"], summary["combined_rank"]]
    totals = [summary["total_pairs"]] * 3
    plt.figure(figsize=(6.0, 3.5))
    plt.bar(labels, [r / t for r, t in zip(ranks, totals)], color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylabel("fractional rank (lower is better)")
    plt.title("GW170104-GW170814 ranks")
    for idx, rank in enumerate(ranks):
        plt.text(idx, rank / totals[idx] + 0.02, f"{rank}/{totals[idx]}", ha="center", fontsize=8)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(config.FIGURE_DIR / f"fig_gwtc_candidate_ranks.{ext}", dpi=400)
    plt.close()

    plt.figure(figsize=(6.0, 3.7))
    for label, features in [("GWTC-3 PE-supported", gwtc3), ("GWTC-5 strict BBH", gwtc5)]:
        if features is None:
            continue
        scores = np.sort(features["combined_time_sky_z"].to_numpy(dtype=float))[::-1]
        kept_fraction = np.arange(1, len(scores) + 1) / len(scores)
        plt.plot(scores, kept_fraction, label=f"{label} ({len(scores)} pairs)")
    ref = CAMPAILLA_REFERENCE_KEPT / CAMPAILLA_REFERENCE_TOTAL
    plt.axhline(ref, color="black", linestyle="--", linewidth=1.0, label="Campailla 185/3655")
    plt.xlabel("combined time+sky score threshold")
    plt.ylabel("fraction of pairs retained")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(config.FIGURE_DIR / f"fig_gwtc_null_threshold.{ext}", dpi=400)
    plt.close()

    plt.figure(figsize=(6.0, 3.7))
    plt.hist(gwtc3["combined_time_sky_z"], bins=45, color="#4c78a8", alpha=0.78)
    plt.axvline(summary["combined_score"], color="#e45756", linewidth=1.5, label=CANDIDATE_A + "-" + CANDIDATE_B)
    plt.xlabel("combined time+sky score")
    plt.ylabel("pair count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(config.FIGURE_DIR / f"fig_gwtc_score_hist.{ext}", dpi=400)
    plt.close()


def write_results(summary: dict, gwtc3_shortlist: dict, gwtc5_shortlist: dict | None) -> None:
    lines = [
        "# GWTC real-data LensGraph physical reranker results",
        "",
        "## Candidate reproduction",
        "",
        f"Candidate pair: `{summary['pair']}`",
        "",
        "| score | value | rank | total pairs |",
        "| --- | ---: | ---: | ---: |",
        f"| sky_step_weight | {summary['sky_step_score']:.4f} | {summary['sky_step_rank']} | {summary['total_pairs']} |",
        f"| sky_log_overlap | {summary['sky_log_overlap']:.4f} | {summary['sky_log_overlap_rank']} | {summary['total_pairs']} |",
        f"| sky_norm_sep (ascending) | {summary['sky_norm_sep']:.4f} | {summary['sky_norm_sep_rank']} | {summary['total_pairs']} |",
        f"| time_score | {summary['time_score']:.4f} | {summary['time_rank']} | {summary['total_pairs']} |",
        f"| combined_time_sky_z | {summary['combined_score']:.4f} | {summary['combined_rank']} | {summary['total_pairs']} |",
        "",
        f"Physical narrative check: **{summary['narrative']}** - {summary['narrative_text']}. "
        f"The observed delay is {summary['delta_t_days']:.2f} days and the Gaussian sky-center separation is {summary['ang_sep_deg']:.2f} deg.",
        "",
        "This means the current real-data observable-only layer correctly down-weights the long time delay, but it does not by itself reproduce the literature's parameter-consistency statement for GW170104-GW170814. That statement should be tested with posterior/parameter-overlap features such as the optional Prompt 5 baseline.",
        "",
        "## Null catalog shortlist",
        "",
        "| catalog | pairs | Campailla fraction equivalent count | equivalent threshold | top-185 fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| GWTC-3 PE-supported | {gwtc3_shortlist['total_pairs']} | {gwtc3_shortlist['equivalent_top_count']} | {gwtc3_shortlist['equivalent_threshold']:.4f} | {gwtc3_shortlist['top185_fraction']:.4f} |",
    ]
    if gwtc5_shortlist is not None:
        lines.append(
            f"| GWTC-5 strict BBH | {gwtc5_shortlist['total_pairs']} | {gwtc5_shortlist['equivalent_top_count']} | {gwtc5_shortlist['equivalent_threshold']:.4f} | {gwtc5_shortlist['top185_fraction']:.4f} |"
        )
    lines += [
        "",
        "Figures written under `figures_gwtc/`: `fig_gwtc_candidate_ranks`, `fig_gwtc_null_threshold`, and `fig_gwtc_score_hist` as PNG and PDF.",
    ]
    (config.SCRIPT_DIR / "RESULTS_gwtc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    gwtc3 = load_features("gwtc3")
    gwtc5 = load_features("gwtc5") if (config.DATA_DIR / "gwtc5_pair_features.csv").exists() else None
    summary = candidate_rank_summary(gwtc3)
    gwtc3_shortlist = shortlist_table(gwtc3)
    gwtc5_shortlist = shortlist_table(gwtc5) if gwtc5 is not None else None
    save_figures(gwtc3, gwtc5, summary)
    write_results(summary, gwtc3_shortlist, gwtc5_shortlist)
    print(summary)
    print(f"Wrote {config.SCRIPT_DIR / 'RESULTS_gwtc.md'}")
    print(f"Wrote figures to {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
