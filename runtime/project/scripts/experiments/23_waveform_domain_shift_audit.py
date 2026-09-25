#!/usr/bin/env python3
"""Audit validation-to-real waveform deployment shift for the v3 experiment.

The real catalogs are overwhelmingly null pairs.  This audit is frozen before
examining real PE follow-up results and never changes fusion weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.real_search.physical_common import write_json


DEFAULT_ROOT = REPO / "results" / "real_noise_injection_v3_physical_20260721"
SEEDS = (202607211, 202607212, 202607213)
DEPLOYMENTS = {"gwtc3": "GWTC-3 (O1-O3)", "gwtc4": "GWTC-4.1 O4a"}
SCORES = (
    ("waveform_raw_sis", "SIS-encoder cosine"),
    ("waveform_raw_pm", "PM-encoder cosine"),
    ("waveform_score", "Calibrated family-mixture log evidence"),
)


def quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    q = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {"q01": float(q[0]), "q05": float(q[1]), "median": float(q[2]), "q95": float(q[3]), "q99": float(q[4])}


def seed_audit(root: Path, deployment: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_root = root / deployment / f"seed_{seed}"
    validation = pd.read_parquet(seed_root / "results" / "fusion_validation_pairs_physical.parquet")
    real = pd.read_parquet(seed_root / "features" / "real_waveform_similarity_physical.parquet")
    real = real[real["waveform_available"].astype(bool)].copy()
    embedding_audit = json.loads((seed_root / "results" / "real_waveform_deployment_audit_physical.json").read_text())
    rows: list[dict[str, Any]] = []
    catastrophic = []
    for column, label in SCORES:
        if column == "waveform_raw_sis":
            signal_mask = (validation["is_true_pair"] == 1) & (validation["family"].str.upper() == "SIS")
        elif column == "waveform_raw_pm":
            signal_mask = (validation["is_true_pair"] == 1) & (validation["family"].str.upper() == "PM")
        else:
            signal_mask = validation["is_true_pair"] == 1
        null = validation.loc[validation["is_true_pair"] == 0, column].to_numpy(dtype=float)
        true = validation.loc[signal_mask, column].to_numpy(dtype=float)
        observed = real[column].to_numpy(dtype=float)
        q_null, q_true, q_real = quantiles(null), quantiles(true), quantiles(observed)
        catastrophic_shift = bool(q_real["median"] >= q_true["median"])
        catastrophic.append(catastrophic_shift)
        rows.append(
            {
                "deployment": deployment,
                "seed": seed,
                "score": column,
                "score_label": label,
                **{f"validation_null_{k}": v for k, v in q_null.items()},
                **{f"validation_true_{k}": v for k, v in q_true.items()},
                **{f"real_{k}": v for k, v in q_real.items()},
                "fraction_real_above_validation_true_q05": float(np.mean(observed >= q_true["q05"])),
                "fraction_real_above_validation_true_median": float(np.mean(observed >= q_true["median"])),
                "ks_real_vs_validation_null": float(stats.ks_2samp(observed, null).statistic),
                "ks_real_vs_validation_true": float(stats.ks_2samp(observed, true).statistic),
                "catastrophic_shift_real_median_ge_true_median": catastrophic_shift,
                "n_validation_null": int(np.isfinite(null).sum()),
                "n_validation_true": int(np.isfinite(true).sum()),
                "n_real_pairs": int(np.isfinite(observed).sum()),
            }
        )
    pass_noncollapse = bool(
        not embedding_audit["constant_score_failure"]
        and float(embedding_audit["embedding_effective_rank_combined"]) >= 5.0
    )
    summary = {
        "deployment": deployment,
        "seed": seed,
        "noncollapse_audit_passed": pass_noncollapse,
        "catastrophic_distribution_shift": bool(any(catastrophic)),
        "deployment_waveform_audit_passed": bool(pass_noncollapse and not any(catastrophic)),
        "catastrophic_rule": (
            "Fail if the median real-catalog score reaches or exceeds the held-out validation true-pair median "
            "for either raw family encoder or the calibrated mixture. The real catalog is overwhelmingly null, "
            "so such behavior invalidates the likelihood-ratio transfer; no PE result is used by this rule."
        ),
        "embedding_audit": embedding_audit,
    }
    return rows, summary


def make_figure(root: Path, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    for row_index, (deployment, label) in enumerate(DEPLOYMENTS.items()):
        seed_root = root / deployment / f"seed_{SEEDS[0]}"
        validation = pd.read_parquet(seed_root / "results" / "fusion_validation_pairs_physical.parquet")
        real = pd.read_parquet(seed_root / "features" / "real_waveform_similarity_physical.parquet")
        real = real[real["waveform_available"].astype(bool)]
        for column_index, (column, score_label) in enumerate(SCORES):
            ax = axes[row_index, column_index]
            if column == "waveform_raw_sis":
                signal_mask = (validation["is_true_pair"] == 1) & (validation["family"].str.upper() == "SIS")
            elif column == "waveform_raw_pm":
                signal_mask = (validation["is_true_pair"] == 1) & (validation["family"].str.upper() == "PM")
            else:
                signal_mask = validation["is_true_pair"] == 1
            groups = [
                (validation.loc[validation["is_true_pair"] == 0, column].to_numpy(float), "Validation null", "#9E9E9E"),
                (validation.loc[signal_mask, column].to_numpy(float), "Validation true", "#D62828"),
                (real[column].to_numpy(float), "Real catalog", "#277DA1"),
            ]
            merged = np.concatenate([g[0][np.isfinite(g[0])] for g in groups])
            lo, hi = np.quantile(merged, [0.002, 0.998])
            bins = np.linspace(lo, hi, 55)
            for values, group_label, color in groups:
                ax.hist(values, bins=bins, density=True, histtype="step", lw=1.2, label=group_label, color=color)
            ax.set_yscale("log")
            ax.set_xlabel(score_label)
            ax.set_ylabel("Density")
            ax.set_title(label)
            if row_index == 0 and column_index == 2:
                ax.legend(fontsize=7)
    for panel, ax in zip("abcdef", axes.ravel()):
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.grid(alpha=0.12)
    fig.suptitle("Waveform deployment shift audit (first seed shown)", fontweight="bold", fontsize=13)
    fig.savefig(out / "fig_waveform_validation_real_domain_shift.pdf", dpi=300)
    fig.savefig(out / "fig_waveform_validation_real_domain_shift.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    out = args.root / "waveform_domain_shift_audit"
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_summaries = [], []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            rows, summary = seed_audit(args.root, deployment, seed)
            all_rows.extend(rows)
            all_summaries.append(summary)
    pd.DataFrame(all_rows).to_csv(out / "waveform_validation_real_distribution_audit.csv", index=False)
    per_seed = pd.DataFrame(
        [
            {
                "deployment": x["deployment"],
                "seed": x["seed"],
                "noncollapse_audit_passed": x["noncollapse_audit_passed"],
                "catastrophic_distribution_shift": x["catastrophic_distribution_shift"],
                "deployment_waveform_audit_passed": x["deployment_waveform_audit_passed"],
            }
            for x in all_summaries
        ]
    )
    per_seed.to_csv(out / "waveform_deployment_pass_per_seed.csv", index=False)
    summary = {
        "per_seed": all_summaries,
        "catalog_all_seeds_passed": {
            deployment: bool(
                per_seed.loc[per_seed["deployment"] == deployment, "deployment_waveform_audit_passed"].all()
            )
            for deployment in DEPLOYMENTS
        },
        "pe_used_for_this_audit": False,
        "weights_changed_by_this_audit": False,
    }
    write_json(out / "waveform_domain_shift_summary.json", summary)
    make_figure(args.root, out)
    print(json.dumps(summary["catalog_all_seeds_passed"], indent=2))


if __name__ == "__main__":
    main()
