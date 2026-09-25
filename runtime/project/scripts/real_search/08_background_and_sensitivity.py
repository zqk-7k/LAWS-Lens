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


def load_gate_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results" / "waveform_gate1_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_background_figure(run_dir: Path, scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(scores["final_score"], bins=60, color="#344054", alpha=0.9)
    for k in (20, 100):
        if len(scores) >= k:
            thr = float(scores.nlargest(k, "final_score")["final_score"].min())
            ax.axvline(thr, ls="--", lw=1.2, label=f"top {k} threshold")
    ax.set_xlabel("final score", fontweight="bold")
    ax.set_ylabel("real-catalog unordered pairs", fontweight="bold")
    ax.set_title("Empirical real-catalog null score distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / "figures" / f"background_score_distribution.{ext}", dpi=300)
    plt.close(fig)


def save_injection_figure(run_dir: Path, gate: pd.DataFrame) -> None:
    if gate.empty:
        return
    test = gate[gate["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    x = np.arange(len(test))
    ax.bar(x - 0.22, test["r_at_1"], width=0.22, label="R@1", color="#2E90FA")
    ax.bar(x, test["r_at_5"], width=0.22, label="R@5", color="#12B76A")
    ax.bar(x + 0.22, test["r_at_10"], width=0.22, label="R@10", color="#F79009")
    ax.set_xticks(x)
    ax.set_xticklabels(test["family"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("held-out real-noise injection retrieval", fontweight="bold")
    ax.set_title("Gate-1 waveform injection recovery")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / "figures" / f"injection_efficiency_curves.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    args = parser.parse_args()
    run_dir = args.run_dir
    ensure_run_dirs(run_dir)

    scores = pd.read_parquet(run_dir / "results" / "real_pair_scores.parquet")
    scores = scores.sort_values("final_score", ascending=False).reset_index(drop=True)
    quantiles = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    rows = []
    for q in quantiles:
        rows.append({
            "summary_type": "empirical_real_catalog_null",
            "metric": "final_score_quantile",
            "quantile": q,
            "value": float(scores["final_score"].quantile(q)),
            "n_pairs": int(len(scores)),
            "note": "All real-catalog unordered pairs are treated as null for ranking context; this is not an LVK FAR.",
        })
    for k in [1, 5, 10, 20, 50, 100]:
        if len(scores) >= k:
            rows.append({
                "summary_type": "empirical_real_catalog_null",
                "metric": f"top_{k}_tail_probability",
                "quantile": np.nan,
                "value": float(k / len(scores)),
                "n_pairs": int(len(scores)),
                "note": "Catalog-level empirical tail probability.",
            })
    bg = pd.DataFrame(rows)
    bg.to_csv(run_dir / "results" / "background_null_summary.csv", index=False)

    gate = load_gate_metrics(run_dir)
    inj_rows = []
    if not gate.empty:
        for row in gate.itertuples(index=False):
            inj_rows.append({
                "experiment": "Gate-1 waveform held-out real-noise injection retrieval",
                "family": row.family,
                "split": row.split,
                "r_at_1": row.r_at_1,
                "r_at_5": row.r_at_5,
                "r_at_10": row.r_at_10,
                "candidate_pair_recall_top10": row.candidate_pair_recall_top10,
                "precision_at_tuned_pair_output": row.pair_precision_at_tuned,
                "recall_at_tuned_pair_output": row.pair_recall_at_tuned,
                "note": "This is a domain-matched waveform Gate-1 injection test, not a full population sensitivity campaign.",
            })
    inj = pd.DataFrame(inj_rows)
    inj.to_csv(run_dir / "results" / "injection_recovery_summary.csv", index=False)
    inj.to_csv(run_dir / "injection_recovery_summary.csv", index=False)

    save_background_figure(run_dir, scores)
    save_injection_figure(run_dir, gate)
    summary = {
        "generated_at_utc": utc_now(),
        "background_rows": int(len(bg)),
        "injection_rows": int(len(inj)),
        "background_output": str(run_dir / "results" / "background_null_summary.csv"),
        "injection_output": str(run_dir / "results" / "injection_recovery_summary.csv"),
        "limitation": "Background is empirical catalog context only; injection sensitivity is Gate-1 waveform retrieval, not an LVK full FAR or full PE follow-up.",
    }
    write_json(run_dir / "results" / "background_and_sensitivity_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
