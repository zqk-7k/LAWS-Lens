#!/usr/bin/env python3
"""Finalize, report, plot, and package the waveform-domain exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/gw-catalog")
BASELINE = PROJECT / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
MODEL_SEEDS = (202609031, 202609032, 202609033)
EVALUATION_SEEDS = (202607241, 202607242, 202607243)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def table_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "（无可用结果）"
    shown = frame.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    try:
        return shown.to_markdown(index=False)
    except ImportError:
        return "```text\n" + shown.to_string(index=False) + "\n```"


def collect_stage_contracts(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "contracts").glob("G*.json")):
        payload = read_json(path)
        rows.append({
            "contract": path.name,
            "passed": payload.get("passed"),
            "decision": payload.get("decision", payload.get("status", "")),
            "timestamp_utc": payload.get("timestamp_utc", ""),
        })
    frame = pd.DataFrame(rows)
    write_csv(root / "tables/STAGE_GATE_LEDGER.csv", frame)
    return frame


def collect_model_ledger(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "models").glob("**/summary.json")):
        payload = read_json(path)
        best = payload.get("best_validation", {})
        rows.append({
            "relative_path": str(path.parent.relative_to(root)),
            "deployment": payload.get("deployment"),
            "seed": payload.get("seed"),
            "windows_seconds": "+".join(map(str, payload.get("windows_seconds", []))),
            "hard_negatives": payload.get("hard_negatives"),
            "uncertainty_head": payload.get("uncertainty_head"),
            "embedding_regularization": payload.get("embedding_regularization"),
            "epochs": payload.get("epochs"),
            "validation_r_at_10": best.get("composite_r_at_10"),
            "validation_pair_auprc": best.get("composite_pair_auprc"),
            "validation_logmc_mae": best.get("logmc_mae"),
            "validation_effective_rank": best.get("embedding_effective_rank"),
            "validation_top100_catastrophic_fraction": best.get("top100_false_catastrophic_mass_fraction"),
        })
    frame = pd.DataFrame(rows)
    write_csv(root / "manifests/EXPERIMENT_LEDGER.csv", frame)
    return frame


def summarize_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(BASELINE / "results/locked_test_retrieval_metrics_per_seed.csv")
    keep = base.loc[base.method.isin(["waveform_only", "BAYESTAR_C_fixed"])].copy()
    keep["provenance"] = "frozen BAYESTAR baseline"
    frames = [keep]
    wf_path = root / "results/locked_test/WAVEFORM_METRICS_PER_SEED.csv"
    if wf_path.exists():
        new = pd.read_csv(wf_path)
        new = new.rename(columns={"evaluation_seed": "seed"})
        new["provenance"] = "new one-time locked test"
        frames.append(new)
    fusion_path = root / "results/locked_test/FUSION_METRICS_PER_SEED.csv"
    if fusion_path.exists():
        fusion = pd.read_csv(fusion_path).rename(columns={"evaluation_seed": "seed"})
        fusion["provenance"] = "new one-time locked test"
        frames.append(fusion)
    per_seed = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "deployment", "method", "seed", "model_seed", "provenance", "overall_r_at_1",
        "overall_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9",
    ]
    per_seed = per_seed[[column for column in columns if column in per_seed.columns]]
    write_csv(root / "tables/LOCKED_TEST_ALL_METHODS_PER_SEED.csv", per_seed)
    numeric = [column for column in ("overall_r_at_1", "overall_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9") if column in per_seed]
    summary_rows = []
    for (deployment, method), part in per_seed.groupby(["deployment", "method"], sort=False):
        row: dict[str, Any] = {"deployment": deployment, "method": method, "n_seeds": len(part)}
        for column in numeric:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_sd"] = float(part[column].std(ddof=1)) if len(part) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_csv(root / "tables/LOCKED_TEST_ALL_METHODS_SUMMARY.csv", summary)
    return per_seed, summary


def preserve_baseline_pe_official(root: Path) -> None:
    source = BASELINE / "results/real_PE_official_budget_summary.csv"
    if source.exists():
        shutil.copy2(source, root / "tables/BASELINE_C_FIXED_PE_OFFICIAL_BUDGET_SUMMARY.csv")
    budget_rows = []
    known_pairs = {
        "GW190924_021846--GW191105_143521",
        "GW190412--GW191204_171526",
        "GW190924_021846--GW190930_133541",
    }
    failure_rows = []
    official_rows = []
    for deployment in ("gwtc3", "gwtc4"):
        source = BASELINE / f"results/{deployment}/real_consensus_with_pe_official_C_fixed.parquet"
        if not source.exists():
            continue
        frame = pd.read_parquet(source).sort_values("consensus_rank", kind="stable")
        top100 = frame.head(100).copy()
        write_csv(root / f"tables/{deployment}_BASELINE_C_FIXED_TOP100_WITH_PE_OFFICIAL.csv", top100)
        official_rows.append({
            "deployment": deployment,
            "method": "C_fixed_frozen_baseline",
            "n_top100": len(top100),
            "official_frontend_overlap": int(top100.get("official_frontend_overlap", pd.Series(False, index=top100.index)).fillna(False).astype(bool).sum()),
            "public_hanabi_table_overlap": int(top100.get("public_hanabi_table_overlap", pd.Series(False, index=top100.index)).fillna(False).astype(bool).sum()),
            "official_po_ml_fpp_available": int(pd.to_numeric(top100.get("official_po_ml_fpp"), errors="coerce").notna().sum()),
            "official_po_phazap_fpp_available": int(pd.to_numeric(top100.get("official_po_phazap_fpp"), errors="coerce").notna().sum()),
            "scope_note": "Frozen local cross-check only; unavailable official numeric fields remain NA",
        })
        for budget in (10, 20, 50, 100):
            part = top100.head(budget)
            mc_bc = pd.to_numeric(part.get("chirp_mass_bhattacharyya_coefficient"), errors="coerce")
            mc_d = pd.to_numeric(part.get("chirp_mass_standardized_posterior_distance"), errors="coerce")
            dmax = pd.to_numeric(part.get("max_standardized_posterior_distance"), errors="coerce")
            budget_rows.append({
                "deployment": deployment,
                "method": "C_fixed_frozen_baseline",
                "audit_scope": "baseline_only_new_model_not_ranked_after_G7_failure",
                "budget": budget,
                "n_pairs": len(part),
                "chirp_mass_BC_ge_0p5": int((mc_bc >= 0.5).sum()),
                "median_chirp_mass_BC": float(mc_bc.median()),
                "chirp_mass_D_le_3": int((mc_d <= 3).sum()),
                "Dmax_le_3": int((dmax <= 3).sum()),
                "catastrophic_chirp_mass_count": int(((mc_bc < 0.1) | (mc_d > 5)).sum()),
                "official_frontend_overlap": int(part.get("official_frontend_overlap", pd.Series(False, index=part.index)).fillna(False).astype(bool).sum()),
                "public_hanabi_table_overlap": int(part.get("public_hanabi_table_overlap", pd.Series(False, index=part.index)).fillna(False).astype(bool).sum()),
            })
        selected = frame.loc[frame.pair_key.astype(str).isin(known_pairs)].copy()
        for row in selected.to_dict("records"):
            failure_rows.append({
                "deployment": deployment,
                "pair_key": row.get("pair_key"),
                "baseline_consensus_rank": row.get("consensus_rank"),
                "baseline_waveform_score": row.get("waveform_score_mean"),
                "baseline_embedding_cosine": row.get("embedding_cosine_mean", np.nan),
                "chirp_mass_BC": row.get("chirp_mass_bhattacharyya_coefficient"),
                "chirp_mass_D": row.get("chirp_mass_standardized_posterior_distance"),
                "Dmax": row.get("max_standardized_posterior_distance"),
                "official_po_ml_fpp": row.get("official_po_ml_fpp"),
                "official_po_phazap_fpp": row.get("official_po_phazap_fpp"),
                "official_tier": row.get("official_tier"),
                "official_fast_golum_stage": row.get("official_fast_golum_stage"),
                "official_hanabi_stage": row.get("official_hanabi_stage"),
                "new_model_rank": np.nan,
                "new_model_waveform_score": np.nan,
                "new_model_audit_status": "NOT_RUN_G7_FAILED_BEFORE_REAL_CATALOG",
            })
    if budget_rows:
        write_csv(root / "tables/REAL_PE_OFFICIAL_BUDGET_SUMMARY.csv", pd.DataFrame(budget_rows))
    if failure_rows:
        write_csv(root / "tables/KNOWN_FAILURE_PAIR_AUDIT.csv", pd.DataFrame(failure_rows))
    if official_rows:
        write_csv(root / "tables/OFFICIAL_STAGE_AVAILABILITY_SUMMARY.csv", pd.DataFrame(official_rows))


def companion_ranks(frame: pd.DataFrame, score_column: str) -> dict[tuple[int, int], int]:
    scores = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=float)
    idx_i = frame.idx_i.to_numpy(dtype=int)
    idx_j = frame.idx_j.to_numpy(dtype=int)
    ranks: dict[tuple[int, int], int] = {}
    for row_index in np.flatnonzero(frame.is_true_pair.astype(bool).to_numpy()):
        for query, companion in ((idx_i[row_index], idx_j[row_index]), (idx_j[row_index], idx_i[row_index])):
            incident = ((idx_i == query) | (idx_j == query)) & np.isfinite(scores)
            ranks[(query, companion)] = 1 + int(np.sum(scores[incident] > scores[row_index]))
    return ranks


def write_query_rank_audit(root: Path) -> None:
    rows = []
    for deployment in ("gwtc3", "gwtc4"):
        for model_seed, evaluation_seed in zip(MODEL_SEEDS, EVALUATION_SEEDS, strict=True):
            new_path = root / f"results/locked_test/{deployment}/seed_{model_seed}/pair_scores_new_waveform.parquet"
            if not new_path.exists():
                continue
            new = pd.read_parquet(new_path)
            old = pd.read_parquet(
                BASELINE / f"results/{deployment}/seed_{evaluation_seed}/test_pair_scores_bayestar_sky.parquet"
            )
            if not (np.array_equal(new.idx_i, old.idx_i) and np.array_equal(new.idx_j, old.idx_j)):
                raise RuntimeError(f"Locked-test pair mismatch in query-rank audit: {deployment}/{model_seed}")
            new_ranks = companion_ranks(new, "waveform_score")
            old_ranks = companion_ranks(old, "waveform_score")
            true_rows = new.loc[new.is_true_pair.astype(bool)]
            for pair in true_rows.itertuples(index=False):
                for query, companion in ((int(pair.idx_i), int(pair.idx_j)), (int(pair.idx_j), int(pair.idx_i))):
                    rows.append({
                        "deployment": deployment,
                        "model_seed": model_seed,
                        "evaluation_seed": evaluation_seed,
                        "family": pair.true_pair_family,
                        "query_idx": query,
                        "companion_idx": companion,
                        "new_rank": new_ranks[(query, companion)],
                        "baseline_rank": old_ranks[(query, companion)],
                        "delta_rank_new_minus_baseline": new_ranks[(query, companion)] - old_ranks[(query, companion)],
                    })
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_csv(root / "tables/NEW_VS_BASELINE_WAVEFORM_QUERY_RANKS.csv", frame)
    summary_rows = []
    for keys, part in frame.groupby(["deployment", "model_seed", "evaluation_seed", "family"], sort=False):
        summary_rows.append({
            "deployment": keys[0], "model_seed": keys[1], "evaluation_seed": keys[2], "family": keys[3],
            "n_queries": len(part),
            "new_R_at_1": float((part.new_rank <= 1).mean()),
            "baseline_R_at_1": float((part.baseline_rank <= 1).mean()),
            "new_R_at_10": float((part.new_rank <= 10).mean()),
            "baseline_R_at_10": float((part.baseline_rank <= 10).mean()),
            "median_delta_rank": float(part.delta_rank_new_minus_baseline.median()),
        })
    write_csv(root / "tables/NEW_VS_BASELINE_WAVEFORM_QUERY_RANKS_SUMMARY.csv", pd.DataFrame(summary_rows))


def verify_historical_hashes(root: Path) -> pd.DataFrame:
    before = pd.read_csv(root / "manifests/HISTORICAL_HASHES_BEFORE.csv")
    rows = []
    for row in before.itertuples(index=False):
        path = Path(row.path)
        current = sha256_file(path) if path.exists() else "MISSING"
        rows.append({"path": row.path, "sha256_before": row.sha256_before, "sha256_after": current, "unchanged": current == row.sha256_before})
    frame = pd.DataFrame(rows)
    write_csv(root / "manifests/HISTORICAL_HASHES_AFTER.csv", frame)
    if not frame.unchanged.all():
        raise RuntimeError("A protected historical input changed")
    return frame


def make_figures(root: Path, per_seed: pd.DataFrame, metric_summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    colors = {"gwtc3": "#0072B2", "gwtc4": "#D55E00"}
    labels = {"gwtc3": "O3", "gwtc4": "O4a"}
    methods = list(dict.fromkeys(per_seed.method.astype(str)))
    x = np.arange(len(methods))
    for ax, metric, title in (
        (axes[0, 0], "overall_r_at_10", "Companion retrieval"),
        (axes[0, 1], "average_precision", "Pair-level precision-recall"),
    ):
        if metric not in per_seed:
            ax.set_axis_off()
            continue
        for offset, deployment in zip((-0.12, 0.12), ("gwtc3", "gwtc4")):
            means, stds = [], []
            for method in methods:
                values = per_seed.loc[per_seed.deployment.eq(deployment) & per_seed.method.eq(method), metric].dropna()
                means.append(values.mean() if len(values) else np.nan)
                stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
                if len(values):
                    ax.scatter(np.full(len(values), methods.index(method) + offset), values, color=colors[deployment], s=12, alpha=0.65, zorder=3)
            ax.errorbar(x + offset, means, yerr=stds, color=colors[deployment], marker="o", ms=3, lw=1, capsize=2, label=labels[deployment])
        ax.set_xticks(x, [name.replace("BAYESTAR_", "").replace("new_", "") for name in methods], rotation=25, ha="right")
        ax.set_ylabel("R@10" if metric == "overall_r_at_10" else "Pair AUPRC")
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(loc="best")

    budget_path = root / "tables/REAL_PE_OFFICIAL_BUDGET_SUMMARY.csv"
    if budget_path.exists():
        budget = pd.read_csv(budget_path)
        for deployment in ("gwtc3", "gwtc4"):
            part = budget.loc[budget.deployment.eq(deployment)]
            axes[1, 0].plot(part.budget, part.chirp_mass_D_le_3 / part.n_pairs, marker="o", color=colors[deployment], label=labels[deployment])
        axes[1, 0].set(xlabel="Candidate budget", ylabel="Fraction with $D_{\\mathcal{M}_c}\\leq3$", title="Frozen C-fixed PE audit")
        axes[1, 0].grid(alpha=0.2, linewidth=0.5); axes[1, 0].legend(loc="best")
    else:
        axes[1, 0].text(0.5, 0.5, "New real-catalog audit not authorized\nby the locked-test Gate", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_axis_off()

    plotted = False
    for deployment in ("gwtc3", "gwtc4"):
        path = root / f"results/real_{deployment}/consensus_all_pairs_with_pe_official.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).head(100)
        xcol = "waveform_score_mean" if "waveform_score_mean" in frame else "waveform_score"
        ycol = "chirp_mass_standardized_posterior_distance"
        if xcol in frame and ycol in frame:
            axes[1, 1].scatter(frame[xcol], frame[ycol], s=14, alpha=0.65, color=colors[deployment], label=labels[deployment])
            plotted = True
    if plotted:
        axes[1, 1].axhline(3, color="black", ls="--", lw=0.8)
        axes[1, 1].axhline(5, color="#CC3311", ls=":", lw=0.8)
        axes[1, 1].set(xlabel="New waveform score", ylabel="$D_{\\mathcal{M}_c}$", title="Real Top-100 waveform versus PE")
        axes[1, 1].grid(alpha=0.2, linewidth=0.5); axes[1, 1].legend(loc="best")
    else:
        axes[1, 1].text(0.5, 0.5, "No new real-catalog ranking", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_axis_off()
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.suptitle("Waveform-domain multiscale exploration", fontsize=10, fontweight="bold")
    fig.savefig(root / "figures/fig_waveform_domain_multiscale_summary.pdf", bbox_inches="tight")
    fig.savefig(root / "figures/fig_waveform_domain_multiscale_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic_figures(root: Path, per_seed: pd.DataFrame) -> None:
    colors = {"gwtc3": "#0072B2", "gwtc4": "#D55E00"}
    labels = {"gwtc3": "O3", "gwtc4": "O4a"}
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    baseline = per_seed.loc[per_seed.method.eq("waveform_only")]
    new = per_seed.loc[per_seed.method.eq("new_waveform_only")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    panels = (
        ("overall_r_at_10", "R@10", False),
        ("average_precision", "Pair AUPRC", False),
        ("false_at_recall_0p5", "False pairs at 50% recall", True),
        ("false_at_recall_0p9", "False pairs at 90% recall", True),
    )
    for ax, (metric, ylabel, log_scale) in zip(axes.flat, panels):
        for deployment in ("gwtc3", "gwtc4"):
            old_part = baseline.loc[baseline.deployment.eq(deployment)].sort_values("seed")
            new_part = new.loc[new.deployment.eq(deployment)].sort_values("seed")
            merged = old_part[["seed", metric]].merge(new_part[["seed", metric]], on="seed", suffixes=("_old", "_new"))
            for row in merged.itertuples(index=False):
                ax.plot([0, 1], [getattr(row, metric + "_old"), getattr(row, metric + "_new")],
                        color=colors[deployment], alpha=0.45, lw=0.8)
            ax.scatter(np.zeros(len(merged)), merged[metric + "_old"], color=colors[deployment], s=18, label=labels[deployment])
            ax.scatter(np.ones(len(merged)), merged[metric + "_new"], color=colors[deployment], s=18)
        ax.set_xticks([0, 1], ["Frozen baseline", "New model"])
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.2, linewidth=0.5)
    axes[0, 0].legend(loc="best")
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.suptitle("One-time locked-test comparison", fontsize=10, fontweight="bold")
    fig.savefig(root / "figures/fig_locked_test_retrieval_false_burden.pdf", bbox_inches="tight")
    fig.savefig(root / "figures/fig_locked_test_retrieval_false_burden.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    g5_path = root / "tables/G5_VALIDATION_METRICS_PER_SEED.csv"
    if g5_path.exists():
        g5 = pd.read_csv(g5_path)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
        for deployment in ("gwtc3", "gwtc4"):
            part = g5.loc[g5.deployment.eq(deployment)].sort_values("model_seed")
            x = np.arange(len(part))
            axes[0, 0].plot(x, part.logmc_mae, marker="o", color=colors[deployment], label=labels[deployment])
            axes[0, 1].plot(x, part.embedding_effective_rank, marker="o", color=colors[deployment])
            axes[1, 0].plot(x, part.top100_false_catastrophic_mass_fraction, marker="o", color=colors[deployment])
            mass_cols = [f"mass_bin_{index}_r_at_10" for index in range(5)]
            axes[1, 1].plot(np.arange(5), part[mass_cols].mean(axis=0), marker="o", color=colors[deployment])
        axes[0, 0].set(ylabel="Validation log-$\\mathcal{M}_c$ MAE", xlabel="Training seed index")
        axes[0, 1].set(ylabel="Embedding effective rank", xlabel="Training seed index")
        axes[1, 0].set(ylabel="Catastrophic fraction in false Top-100", xlabel="Training seed index")
        axes[1, 1].set(ylabel="Validation R@10", xlabel="Chirp-mass stratum", xticks=np.arange(5))
        axes[0, 0].legend(loc="best")
        for label, ax in zip("abcd", axes.flat):
            ax.grid(alpha=0.2, linewidth=0.5)
            ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
        fig.suptitle("Validation chirp-mass and embedding audit", fontsize=10, fontweight="bold")
        fig.savefig(root / "figures/fig_validation_chirpmass_embedding_audit.pdf", bbox_inches="tight")
        fig.savefig(root / "figures/fig_validation_chirpmass_embedding_audit.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    any_real = False
    for deployment in ("gwtc3", "gwtc4"):
        path = root / f"tables/{deployment}_BASELINE_C_FIXED_TOP100_WITH_PE_OFFICIAL.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path)
        score = pd.to_numeric(part.waveform_score_mean, errors="coerce")
        bc = pd.to_numeric(part.chirp_mass_bhattacharyya_coefficient, errors="coerce")
        distance = pd.to_numeric(part.chirp_mass_standardized_posterior_distance, errors="coerce")
        axes[0, 0].scatter(score, bc, s=14, alpha=0.6, color=colors[deployment], label=labels[deployment])
        axes[0, 1].scatter(score, distance, s=14, alpha=0.6, color=colors[deployment])
        budgets = np.arange(1, len(part) + 1)
        axes[1, 0].plot(budgets, np.cumsum(bc >= 0.5) / budgets, color=colors[deployment])
        axes[1, 1].plot(budgets, np.cumsum(distance <= 3) / budgets, color=colors[deployment])
        any_real = True
    if any_real:
        axes[0, 0].set(xlabel="Frozen C-fixed waveform score", ylabel="$BC_{\\mathcal{M}_c}$")
        axes[0, 1].set(xlabel="Frozen C-fixed waveform score", ylabel="$D_{\\mathcal{M}_c}$")
        axes[0, 1].axhline(3, color="black", ls="--", lw=0.8)
        axes[1, 0].set(xlabel="Frozen candidate budget", ylabel="Fraction with $BC_{\\mathcal{M}_c}\\geq0.5$")
        axes[1, 1].set(xlabel="Frozen candidate budget", ylabel="Fraction with $D_{\\mathcal{M}_c}\\leq3$")
        axes[0, 0].legend(loc="best")
        for label, ax in zip("abcd", axes.flat):
            ax.grid(alpha=0.2, linewidth=0.5)
            ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
        fig.suptitle("Frozen C-fixed real-catalog PE audit (new model not ranked)", fontsize=10, fontweight="bold")
        fig.savefig(root / "figures/fig_baseline_C_fixed_waveform_PE_audit.pdf", bbox_inches="tight")
        fig.savefig(root / "figures/fig_baseline_C_fixed_waveform_PE_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(root: Path, stage: pd.DataFrame, ledger: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = read_json(root / "contracts/G4_ARCHITECTURE_SELECTED.json")
    g7 = read_json(root / "contracts/G7_LOCKED_TEST_COMPLETE.json")
    g8 = read_json(root / "contracts/G8_FUSION_COMPLETE.json")
    data_rows = []
    path = root / "tables/G1_DATA_STRATA_SUMMARY.csv"
    if path.exists():
        data_rows = pd.read_csv(path)
    pe_path = root / "tables/REAL_PE_OFFICIAL_BUDGET_SUMMARY.csv"
    pe = pd.read_csv(pe_path) if pe_path.exists() else pd.DataFrame()
    failures_path = root / "tables/KNOWN_FAILURE_PAIR_AUDIT.csv"
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    metric_columns = [column for column in (
        "deployment", "method", "overall_r_at_1_mean", "overall_r_at_10_mean",
        "average_precision_mean", "false_at_recall_0p5_mean", "false_at_recall_0p9_mean"
    ) if column in metrics]
    text = f"""# O3/O4a 波形域多尺度探索实验完整报告

**最终状态：** `{FINAL_STATUS}`  
**实验目录：** `{root}`  
**基线：** `{BASELINE}`

## 1. 科学边界

本轮只探索 waveform 通道。BAYESTAR `Nside=512` 天空图、`Z_sky`、一维
`Z_time`、真实目录范围以及 C-fixed 历史结果均被冻结。本轮结果是
lens-candidate refinement 的方法学审计，不是透镜探测或确认；真实 PE 和
官方阶段未参与选模。若 G7 失败，本节只复制冻结 C-fixed 的 PE/官方阶段
审计，绝不把它冒充为新模型结果。

## 2. 基线与不可变性

基线交付包 SHA-256 与合同值一致，并复现 O3/O4a waveform-only 及 C-fixed
三通道指纹。受保护输入在结束时逐文件复算哈希，详见
`manifests/HISTORICAL_HASHES_AFTER.csv`。

## 3. 数据构建

新训练库使用 O3/O4a 各自的公开 GWOSC H1/L1 off-source strain 与局部 PSD。
每个物理源产生 24 s、4096 Hz 的 detector response，随后沿用冻结的 Tukey
taper、40--580 Hz 物理频带、PSD whitening 和抗混叠 4096→2048 Hz 预处理。
短分支始终是末尾 2 s/4096 点；8 s 或 16 s 分支经抗混叠重采样到 4096 点。

训练 proposal 在 detector-frame chirp mass 与 network-SNR 区间内做覆盖均衡，
其用途是 representation learning，不代表天体物理人口先验。validation 和
locked test 仍是冻结 BAYESTAR 人口级评估目录。

{table_markdown(pd.DataFrame(data_rows), 3)}

首次 G1 因无效 off-source 窗口及 global GW-LMC origin 重合被 Gate 停止；修复
后先做有限值/PSD 拒绝抽样，并排除冻结 validation/test 的所有 origin ID。
subhalo 未泄漏且可观测的透镜环境数量有限，因此允许未泄漏 lens environment
重复，但每个 waveform source 具有独立 detector-frame masses 和唯一父 UID。
这项环境多样性限制不被隐藏。

## 4. 阶段 Gate

{table_markdown(stage, 4)}

## 5. 模型消融

{table_markdown(ledger, 4)}

窗口、hard negative、chirp-mass uncertainty head 和 embedding regularization
均只依据开发 validation 晋级。G6 才在冻结 BAYESTAR validation 上分别拟合
O3 与 O4a waveform likelihood-ratio；G7 对唯一配置一次性打开 locked test。

## 6. 注入 locked-test 结果

{table_markdown(metrics[metric_columns] if metric_columns else metrics, 4)}

`R@K` 是 directed companion retrieval；Pair AUPRC 与 F50/F90 使用无序 pair。
这些指标来自注入目录，不应解释成真实 GWTC 的透镜率或显著性。

## 7. 融合结论

G7：`{g7.get('decision', '尚未运行')}`。  
G8：`{g8.get('decision', '未获授权或尚未运行')}`。  
选择配置：`{selected.get('selected_config_id', '尚未选择')}`。  
最终真实目录方案：`{g8.get('preferred_frozen_scheme', '无新方案')}`。

若 G7 未通过，协议禁止用失败的新 waveform 模型重排真实目录；交付包仍保存
冻结 C-fixed 的 PE/官方阶段表。若 G7 通过，则 G9 表是配置冻结后的一次只读
外部审计，不能反向证明模型选择正确。

## 8. 真实 PE 与官方阶段

下表的 `audit_scope` 明确给出来源。当前 G7 未通过时，它只描述冻结的
C-fixed 基线；失败的新 waveform 模型没有被授权接回融合或重排真实目录。

{table_markdown(pe, 3)}

已知失败 pair：

{table_markdown(failures, 4)}

`BC_Mc` 与 `D_Mc` 是公开 PE 的描述性一致性审计。它们不是训练标签，也不是
第四个检索通道。官方 PO/ML、PO/Phazap、Tier、Fast-GOLUM 或 Hanabi 字段仅
复用冻结的本地交叉核对；缺失值保持缺失，不做推断或补写。

## 9. 可否替换基线

本报告不自动采纳任何新结果。只有 O3/O4a waveform locked test、三个 seed、
三通道融合及假对负担同时满足预注册 guardrails，才可提交作者考虑建立新的
确认版本。无论数值如何，本轮均不覆盖 v9.3/v9.4/C/C-fixed，不修改论文。

## 10. 文件导航

- 所有注入指标：`tables/LOCKED_TEST_ALL_METHODS_PER_SEED.csv`
- 注入汇总：`tables/LOCKED_TEST_ALL_METHODS_SUMMARY.csv`
- 逐 query 新旧 rank：`tables/NEW_VS_BASELINE_WAVEFORM_QUERY_RANKS.csv`
- PE/官方预算：`tables/REAL_PE_OFFICIAL_BUDGET_SUMMARY.csv`
- 官方字段可用性：`tables/OFFICIAL_STAGE_AVAILABILITY_SUMMARY.csv`
- 冻结 C-fixed O3/O4a Top-100：`tables/gwtc3_BASELINE_C_FIXED_TOP100_WITH_PE_OFFICIAL.csv`、`tables/gwtc4_BASELINE_C_FIXED_TOP100_WITH_PE_OFFICIAL.csv`
- 已知失败 pair：`tables/KNOWN_FAILURE_PAIR_AUDIT.csv`
- 配置与 Gate：`contracts/`、`configs/`
- 可复现代码：`scripts/`
- 图：`figures/fig_waveform_domain_multiscale_summary.pdf`
"""
    (root / "reports/FINAL_WAVEFORM_EXPERIMENT_REPORT_CN.md").write_text(text, encoding="utf-8")
    (root / "reports/STAGE_GATE_REPORT_CN.md").write_text(
        "# 阶段 Gate 报告\n\n" + table_markdown(stage, 4) + f"\n\n最终状态：`{FINAL_STATUS}`\n",
        encoding="utf-8",
    )
    (root / "README_CN.md").write_text(
        "# Waveform-domain multiscale exploration\n\n"
        "完整中文报告见 `reports/FINAL_WAVEFORM_EXPERIMENT_REPORT_CN.md`。"
        f"本轮状态为 `{FINAL_STATUS}`，不得自动替换历史结果。\n",
        encoding="utf-8",
    )


def compact_files(root: Path) -> list[Path]:
    included = []
    allowed_roots = ("contracts", "configs", "manifests", "tables", "figures", "reports", "scripts", "logs")
    for name in allowed_roots:
        included.extend(path for path in (root / name).glob("**/*") if path.is_file())
    included.extend(path for path in (root / "results").glob("**/*") if path.is_file() and path.suffix in {".csv", ".json", ".parquet"})
    for path in (root / "models/final").glob("**/validation_selected_model.pt") if (root / "models/final").exists() else []:
        included.append(path)
    included.extend(path for path in (root / "failed_attempts").glob("**/*") if path.is_file() and path.suffix in {".json", ".log", ".txt"})
    if (root / "README_CN.md").exists():
        included.append(root / "README_CN.md")
    return sorted(set(included))


def package(root: Path, package_dir: Path) -> tuple[Path, str]:
    package_dir.mkdir(parents=True, exist_ok=True)
    files = [path for path in compact_files(root) if path.name != "OUTPUT_SHA256.csv"]
    manifest = pd.DataFrame({
        "relative_path": [str(path.relative_to(root)) for path in files],
        "size_bytes": [path.stat().st_size for path in files],
        "sha256": [sha256_file(path) for path in files],
    })
    write_csv(root / "manifests/OUTPUT_SHA256.csv", manifest)
    files = compact_files(root)
    target = package_dir / f"{root.name}_deliverables.tar.gz"
    with tarfile.open(target, "w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
    digest = sha256_file(target)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=PROJECT / "packages")
    args = parser.parse_args()
    root = args.root
    preserve_baseline_pe_official(root)
    write_query_rank_audit(root)
    stage = collect_stage_contracts(root)
    ledger = collect_model_ledger(root)
    per_seed, metric_summary = summarize_metrics(root)
    verify_historical_hashes(root)
    make_figures(root, per_seed, metric_summary)
    make_diagnostic_figures(root, per_seed)
    write_report(root, stage, ledger, metric_summary)
    write_json(root / "contracts/FINAL_STATUS.json", {
        "status": FINAL_STATUS,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "historical_outputs_overwritten": False,
        "paper_modified": False,
        "new_real_catalog_audit_available": (root / "contracts/G9_REAL_READONLY_AUDIT_COMPLETE.json").exists(),
    })
    target, digest = package(root, args.package_dir)
    print(json.dumps({"package": str(target), "sha256": digest, "status": FINAL_STATUS}, indent=2))


if __name__ == "__main__":
    main()
