#!/usr/bin/env python3
"""Independent PE follow-up audit for the frozen v7 real-catalog ranking."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
DEFAULT_OUT = DEFAULT_INPUT / "pe_followup"
DEFAULT_PACKAGE = REPO / "packages/real_noise_injection_v7_peak2s_formal_20260722_deliverables.tar.gz"
DEFAULT_SEEDS = (202607241, 202607242, 202607243)
EXTERNAL_CANDIDATE_PAIRS = (
    ("GW200210_035448", "GW200210_100022", "Barsode et al. 2026 PO2.0 rank 1; IAS-only events"),
    ("GW190425_133124", "GW190605_025957", "Barsode et al. 2026 PO2.0 rank 2; includes IAS-only event"),
    ("GW230707_124047", "GW230708_230935", "Barsode et al. 2026 PO2.0 rank 3"),
    ("GW230922_040658", "GW231005_021030", "Barsode et al. 2026 PO2.0 rank 4"),
    ("GW190425_133124", "GW190426_190642", "Barsode et al. 2026 PO2.0 rank 5; includes IAS-only event"),
    ("GW170809_082821", "GW170814_103043", "Hannuksela et al. historical candidate"),
    ("GW170104", "GW170814", "historical O2 candidate requested in this project"),
    ("GW191103_012549", "GW191105_143521", "O3 posterior-overlap follow-up candidate"),
)


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = module_from(
    REPO / "scripts/experiments/32_physical_deployment_v4_pe_audit.py",
    "v4_pe_helpers_for_v7",
)


def consensus_scores(
    input_root: Path,
    deployment: str,
    seeds: list[int],
    method: str,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = (
            input_root
            / deployment
            / f"seed_{seed}/results/real_pair_scores_strict_h1l1_bbh_{method}_v7.parquet"
        )
        frame = pd.read_parquet(path).copy()
        frame["seed"] = int(seed)
        frame["pair_key"] = frame.apply(
            lambda row: "--".join(sorted((str(row.event_i), str(row.event_j)))), axis=1
        )
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    first_columns = [
        name
        for name in (
            "event_i",
            "event_j",
            "idx_i",
            "idx_j",
            "waveform_available",
            "strict_h1l1_bbh_pair",
            "pair_has_ood",
            "sky_bayes_factor",
            "cosine_overlap",
            "angular_sep_map_deg",
            "common_nside",
        )
        if name in merged.columns
    ]
    aggregation: dict[str, Any] = {
        "median_seed_rank": ("rank", "median"),
        "q75_seed_rank": ("rank", lambda values: float(np.quantile(values, 0.75))),
        "best_seed_rank": ("rank", "min"),
        "top10_seed_frequency": ("rank", lambda values: float(np.mean(np.asarray(values) <= 10))),
        "final_score": ("final_score", "median"),
        "waveform_score": ("waveform_score", "median"),
        "time_score": ("time_score", "median"),
        "sky_score": ("sky_score", "median"),
    }
    for contribution in (
        "waveform_contribution",
        "time_contribution",
        "sky_contribution",
    ):
        if contribution in merged.columns:
            aggregation[contribution] = (contribution, "median")
    for name in first_columns:
        aggregation[name] = (name, "first")
    consensus = merged.groupby("pair_key", as_index=False).agg(**aggregation)
    consensus = consensus.sort_values(
        ["median_seed_rank", "q75_seed_rank", "best_seed_rank", "final_score"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    consensus["rank"] = np.arange(1, len(consensus) + 1, dtype=np.int32)
    consensus["median_rank"] = consensus["median_seed_rank"]
    consensus["consensus_method"] = method
    return consensus


def frozen_consensus_method(input_root: Path, deployment: str, seeds: list[int]) -> tuple[str, dict[str, Any]]:
    summaries = [
        json.loads(
            (
                input_root
                / deployment
                / f"seed_{seed}/seed_summary_v7.json"
            ).read_text(encoding="utf-8")
        )
        for seed in seeds
    ]
    waveform_seed_pass = [bool(item["waveform_eligible_for_primary"]) for item in summaries]
    # The v7 research question explicitly evaluates a nonzero contribution
    # from every predeclared channel. The weights are still selected only on
    # synthetic validation systems. Gate failures remain visible diagnostics;
    # they are never hidden by silently relabelling a time+sky fallback as a
    # three-channel result.
    method = "candidate_three_channel_strictly_positive"
    reason = (
        "Predeclared v7 three-channel analysis. All weights are constrained "
        "to be positive and selected on synthetic validation systems only; "
        "per-seed waveform/deployment Gate results are reported separately."
    )
    return method, {
        "method": method,
        "reason": reason,
        "waveform_eligible_by_seed": {
            str(seed): passed for seed, passed in zip(seeds, waveform_seed_pass)
        },
    }


def add_area_columns(scores: pd.DataFrame, template_manifest: Path) -> pd.DataFrame:
    templates = pd.read_csv(template_manifest)
    templates = templates[templates["usable"] == True].set_index("event_name")  # noqa: E712
    area = templates["area90_deg2"].to_dict()
    out = scores.copy()
    out["sky_area90_i_deg2"] = out["event_i"].map(area)
    out["sky_area90_j_deg2"] = out["event_j"].map(area)
    return out


def event_name_matches(actual: str, requested: str) -> bool:
    """Match legacy short O1/O2 names to full GPS-suffixed names."""
    actual, requested = str(actual), str(requested)
    return actual == requested or actual.startswith(requested + "_") or requested.startswith(actual + "_")


def external_candidate_crosscheck(
    deployment_key: str,
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for event_a, event_b, source in EXTERNAL_CANDIDATE_PAIRS:
        def locate(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[
                frame.apply(
                    lambda row: (
                        event_name_matches(str(row.event_i), event_a)
                        and event_name_matches(str(row.event_j), event_b)
                    )
                    or (
                        event_name_matches(str(row.event_i), event_b)
                        and event_name_matches(str(row.event_j), event_a)
                    ),
                    axis=1,
                )
            ]

        primary_match = locate(primary)
        comparison_match = locate(comparison)
        diagnostic_match = locate(diagnostics)
        record: dict[str, Any] = {
            "deployment": deployment_key,
            "requested_event_i": event_a,
            "requested_event_j": event_b,
            "literature_context": source,
            "present_in_strict_catalog": bool(len(primary_match)),
            "primary_rank": int(primary_match.iloc[0]["rank"]) if len(primary_match) else np.nan,
            "unconstrained_three_channel_rank": int(comparison_match.iloc[0]["rank"])
            if len(comparison_match)
            else np.nan,
        }
        if len(primary_match):
            record["matched_event_i"] = str(primary_match.iloc[0]["event_i"])
            record["matched_event_j"] = str(primary_match.iloc[0]["event_j"])
        if len(diagnostic_match):
            for name in (
                "chirp_mass_standardized_posterior_distance",
                "mass_ratio_standardized_posterior_distance",
                "chi_eff_standardized_posterior_distance",
                "intrinsic_3sigma_consistent",
            ):
                if name in diagnostic_match:
                    record[name] = diagnostic_match.iloc[0][name]
        rows.append(record)
    return pd.DataFrame(rows)


def consensus_event_predictions(input_root: Path, deployment: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = input_root / deployment / f"seed_{seed}/features/real_waveform_embeddings_unified_v7.parquet"
        frame = pd.read_parquet(path)
        keep = [
            "event_name",
            "waveform_pred_chirp_mass_detector",
            "waveform_pred_mass_ratio",
            "strict_h1l1_preprocessing_pass",
        ]
        current = frame[keep].copy()
        current["seed"] = int(seed)
        rows.append(current)
    merged = pd.concat(rows, ignore_index=True)
    return merged.groupby("event_name", as_index=False).agg(
        waveform_pred_chirp_mass_detector=("waveform_pred_chirp_mass_detector", "median"),
        waveform_pred_mass_ratio=("waveform_pred_mass_ratio", "median"),
        waveform_prediction_seed_std_mc=("waveform_pred_chirp_mass_detector", "std"),
        waveform_prediction_seed_std_q=("waveform_pred_mass_ratio", "std"),
        strict_h1l1_all_seeds=("strict_h1l1_preprocessing_pass", "all"),
    )


def plot_top5_waveform_prediction_vs_pe(
    deployment: Any,
    top: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    lookup = predictions.set_index("event_name")
    fig, axes = base.plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    labels = []
    for pair_index, pair in enumerate(top.head(5).itertuples(index=False)):
        labels.append(f"{int(pair.rank)}")
        for event_offset, event in enumerate((str(pair.event_i), str(pair.event_j))):
            x = pair_index + (-0.13 if event_offset == 0 else 0.13)
            color = "#277DA1" if event_offset == 0 else "#F94144"
            for axis, parameter, prediction_column in (
                (axes[0], "chirp_mass", "waveform_pred_chirp_mass_detector"),
                (axes[1], "mass_ratio", "waveform_pred_mass_ratio"),
            ):
                posterior = cache[event]["posterior"][parameter]
                q16, median, q84 = np.quantile(posterior, [0.16, 0.5, 0.84])
                axis.errorbar(
                    x,
                    median,
                    yerr=[[median - q16], [q84 - median]],
                    fmt="o",
                    color=color,
                    ms=4,
                    capsize=2,
                    alpha=0.85,
                )
                if event in lookup.index:
                    axis.scatter(
                        x,
                        float(lookup.loc[event, prediction_column]),
                        marker="x",
                        color="black",
                        s=30,
                        linewidth=1.0,
                        zorder=4,
                    )
    axes[0].set_ylabel(r"Detector-frame $\mathcal{M}_c\ (M_\odot)$")
    axes[1].set_ylabel(r"Mass ratio $q$")
    for axis in axes:
        axis.set_xticks(np.arange(5), labels)
        axis.set_xlabel("Consensus pair rank")
        axis.grid(alpha=0.15)
    axes[0].plot([], [], "o", color="#277DA1", label="event i PE median / 68% CI")
    axes[0].plot([], [], "o", color="#F94144", label="event j PE median / 68% CI")
    axes[0].plot([], [], "x", color="black", label="waveform-head prediction")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{deployment.label}: waveform-derived intrinsic prediction versus independent PE", fontweight="bold")
    stem = figure_dir / f"fig_top5_waveform_prediction_vs_pe_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    base.plt.close(fig)


def plot_v7_top5_posteriors(
    deployment: Any,
    top: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    figure_dir: Path,
) -> None:
    labels = {
        "chirp_mass": r"$\mathcal{M}_{c,\rm det}\ (M_\odot)$",
        "mass_ratio": r"$q$",
        "chi_eff": r"$\chi_{\rm eff}$",
    }
    fig, axes = base.plt.subplots(5, 3, figsize=(11.0, 11.5), constrained_layout=True)
    for row_index, pair in enumerate(top.head(5).itertuples(index=False)):
        for col_index, parameter in enumerate(base.PARAMETERS):
            axis = axes[row_index, col_index]
            x = cache[str(pair.event_i)]["posterior"][parameter]
            y = cache[str(pair.event_j)]["posterior"][parameter]
            lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
            hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
            grid = np.linspace(lo, hi, 500)
            axis.plot(grid, stats.gaussian_kde(x)(grid), color="#277DA1", lw=1.0, label=str(pair.event_i))
            axis.plot(grid, stats.gaussian_kde(y)(grid), color="#F94144", lw=1.0, label=str(pair.event_j))
            metric = base.one_dimensional_pair_metrics(x, y)
            axis.text(
                0.02,
                0.96,
                f"D={metric['standardized_posterior_distance']:.2f}\nBC={metric['bhattacharyya_coefficient']:.2g}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
            axis.set_xlabel(labels[parameter])
            if col_index == 0:
                axis.set_ylabel(f"Consensus rank {int(pair.rank)}\nDensity")
            if row_index == 0 and col_index == 2:
                axis.legend(fontsize=6, loc="upper right")
    fig.suptitle(
        f"{deployment.label}: intrinsic-parameter consistency of the top five triage pairs",
        fontweight="bold",
    )
    stem = figure_dir / f"fig_top5_pe_consistency_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    base.plt.close(fig)


def plot_v7_waveform_inputs(
    input_root: Path,
    deployment: Any,
    top: pd.DataFrame,
    figure_dir: Path,
) -> None:
    values = np.load(
        input_root / deployment.key / "shared/real_event_preprocessed_full24.npy",
        mmap_mode="r",
    )
    audit = pd.read_csv(
        input_root / deployment.key / "shared/real_event_preprocessing_audit.csv"
    )
    event_index = audit.set_index("event_name")["idx"].to_dict()
    sample_count = 4096
    sample_rate = 2048.0
    time_axis = np.arange(sample_count) / sample_rate - 1.75
    fig, axes = base.plt.subplots(5, 2, figsize=(12.0, 10.5), sharex=True, constrained_layout=True)
    for row_index, pair in enumerate(top.head(5).itertuples(index=False)):
        for detector, axis in enumerate(axes[row_index]):
            x = np.asarray(values[int(event_index[str(pair.event_i)]), detector, -sample_count:])
            y = np.asarray(values[int(event_index[str(pair.event_j)]), detector, -sample_count:])
            axis.plot(time_axis, x, color="#277DA1", lw=0.45, alpha=0.8, label=str(pair.event_i))
            axis.plot(time_axis, y, color="#F94144", lw=0.45, alpha=0.72, label=str(pair.event_j))
            axis.axvline(0, color="black", ls="--", lw=0.6)
            axis.set_title(f"Consensus rank {int(pair.rank)}, {'H1' if detector == 0 else 'L1'}")
            if detector == 0:
                axis.set_ylabel("Whitened model input")
            if row_index == 4:
                axis.set_xlabel("Time from event GPS (s)")
            if row_index == 0 and detector == 1:
                axis.legend(fontsize=6, loc="upper right")
    fig.suptitle(
        f"{deployment.label}: exact 2 s / 4096-sample waveform inputs for top pairs",
        fontweight="bold",
    )
    stem = figure_dir / f"fig_top5_corrected_waveform_inputs_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    base.plt.close(fig)


def event_prediction_pe_audit(
    predictions: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare strain-derived point predictions with independent PE summaries."""
    rows = []
    for item in predictions.itertuples(index=False):
        event = str(item.event_name)
        if event not in cache:
            continue
        record: dict[str, Any] = {"event_name": event}
        for parameter, prediction_name in (
            ("chirp_mass", "waveform_pred_chirp_mass_detector"),
            ("mass_ratio", "waveform_pred_mass_ratio"),
        ):
            posterior = np.asarray(cache[event]["posterior"].get(parameter, []), dtype=np.float64)
            if len(posterior) < 50:
                continue
            q16, median, q84 = np.quantile(posterior, [0.16, 0.5, 0.84])
            sigma = max(float(0.5 * (q84 - q16)), 1e-12)
            prediction = float(getattr(item, prediction_name))
            record.update(
                {
                    f"{parameter}_waveform_prediction": prediction,
                    f"{parameter}_pe_median": float(median),
                    f"{parameter}_pe_q16": float(q16),
                    f"{parameter}_pe_q84": float(q84),
                    f"{parameter}_prediction_pe_distance_sigma": abs(prediction - median) / sigma,
                }
            )
        rows.append(record)
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for parameter in ("chirp_mass", "mass_ratio"):
        x = frame[f"{parameter}_waveform_prediction"].to_numpy(dtype=np.float64)
        y = frame[f"{parameter}_pe_median"].to_numpy(dtype=np.float64)
        d = frame[f"{parameter}_prediction_pe_distance_sigma"].to_numpy(dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        summary[f"{parameter}_events"] = int(np.sum(keep))
        summary[f"{parameter}_spearman_prediction_vs_pe"] = float(
            stats.spearmanr(x[keep], y[keep]).statistic
        )
        summary[f"{parameter}_median_prediction_pe_distance_sigma"] = float(
            np.median(d[keep])
        )
    return frame, summary


def plot_event_prediction_pe_audit(
    deployment: Any,
    frame: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(9.5, 4.1), constrained_layout=True)
    for axis, parameter, label in (
        (axes[0], "chirp_mass", r"Detector-frame $\mathcal{M}_c\ (M_\odot)$"),
        (axes[1], "mass_ratio", r"Mass ratio $q$"),
    ):
        x = frame[f"{parameter}_pe_median"].to_numpy(dtype=np.float64)
        y = frame[f"{parameter}_waveform_prediction"].to_numpy(dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(y)
        lo = min(float(np.min(x[keep])), float(np.min(y[keep])))
        hi = max(float(np.max(x[keep])), float(np.max(y[keep])))
        axis.scatter(x[keep], y[keep], s=16, alpha=0.65, color="#277DA1")
        axis.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8)
        rho = stats.spearmanr(x[keep], y[keep]).statistic
        axis.text(0.04, 0.95, rf"$\rho_s={rho:.2f}$", transform=axis.transAxes, va="top")
        axis.set_xlabel(f"Independent PE median: {label}")
        axis.set_ylabel(f"Waveform-head prediction: {label}")
        axis.grid(alpha=0.15)
    fig.suptitle(f"{deployment.label}: real-event waveform deployment audit", fontweight="bold")
    stem = figure_dir / f"fig_real_waveform_prediction_pe_audit_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    base.plt.close(fig)


def plot_top5_score_contributions(
    deployment: Any,
    scores: pd.DataFrame,
    figure_dir: Path,
    suffix: str,
    title_suffix: str,
) -> None:
    """Show the signed, already-weighted evidence behind the top five ranks."""
    top = scores.head(5).copy()
    labels = [f"{row.event_i}\n{row.event_j}" for row in top.itertuples(index=False)]
    x = np.arange(len(top), dtype=np.float64)
    width = 0.24
    with base.plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    ):
        fig, ax = base.plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        for offset, column, label, color in (
            (-width, "waveform_contribution", "Waveform", "#4C78A8"),
            (0.0, "time_contribution", "Time delay", "#F2CF5B"),
            (width, "sky_contribution", "Sky Bayes factor", "#59A14F"),
        ):
            values = (
                top[column].to_numpy(dtype=np.float64)
                if column in top
                else np.zeros(len(top), dtype=np.float64)
            )
            ax.bar(x + offset, values, width=width, label=label, color=color)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.set_xticks(x, labels, rotation=28, ha="right", fontsize=7)
        ax.set_ylabel("Weighted log-evidence contribution")
        ax.set_title(f"{deployment.label}: top-five channel contributions ({title_suffix})")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        ax.grid(axis="y", alpha=0.18)
        stem = figure_dir / f"fig_top5_score_contributions_{deployment.key}_{suffix}"
        fig.savefig(stem.with_suffix(".pdf"), dpi=300)
        fig.savefig(stem.with_suffix(".png"), dpi=240)
        base.plt.close(fig)


def chirp_mass_rank_enrichment(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Quantify enrichment of independent detector-frame Mc agreement."""
    column = "chirp_mass_standardized_posterior_distance"
    valid = diagnostics[np.isfinite(diagnostics[column])].sort_values("rank").copy()
    rows: list[dict[str, Any]] = []
    population = len(valid)
    for threshold in (1.0, 2.0):
        successes = int((valid[column] <= threshold).sum())
        base_rate = successes / population if population else np.nan
        for requested_top_n in (5, 10, 20, 50):
            top_n = min(requested_top_n, population)
            observed = int((valid.head(top_n)[column] <= threshold).sum())
            precision = observed / top_n if top_n else np.nan
            rows.append(
                {
                    "d_mc_threshold": threshold,
                    "top_n": top_n,
                    "consistent_pairs": observed,
                    "precision": precision,
                    "catalog_consistency_rate": base_rate,
                    "enrichment_over_catalog": (
                        precision / base_rate
                        if base_rate and np.isfinite(base_rate)
                        else np.nan
                    ),
                    "hypergeometric_one_sided_p_audit_only": (
                        float(stats.hypergeom.sf(observed - 1, population, successes, top_n))
                        if population and successes and top_n
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def heldout_retrieval_audit(
    input_root: Path,
    out: Path,
    figure_dir: Path,
    seeds: list[int],
) -> dict[str, Any]:
    methods = (
        "waveform_only",
        "time_only",
        "sky_bayes_factor_only",
        "time_sky_candidate_selected",
        "candidate_three_channel_unconstrained",
        "candidate_three_channel_strictly_positive",
    )
    frames = []
    gate_rows = []
    weight_rows = []
    for deployment in ("gwtc3", "gwtc4"):
        path = input_root / deployment / "heldout_test_retrieval_metrics_per_seed_v7.csv"
        current = pd.read_csv(path)
        current = current[
            (current["subset"] == "overall") & current["method"].isin(methods)
        ].copy()
        frames.append(current)
        for seed in seeds:
            seed_root = input_root / deployment / f"seed_{seed}"
            summary = json.loads((seed_root / "seed_summary_v7.json").read_text(encoding="utf-8"))
            validation_gate = summary["validation_gate"]
            deployment_gate = summary["real_waveform_deployment_audit"]
            gate_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "validation_discrimination_passed": validation_gate["statistical_discrimination_passed"],
                    "validation_incremental_utility_passed": validation_gate["incremental_utility_passed"],
                    "real_effective_rank": deployment_gate["real_embedding_effective_rank"],
                    "validation_effective_rank_q01": deployment_gate["validation_effective_rank_reference"]["q01"],
                    "real_rank_q01_gate_passed": deployment_gate["effective_rank_passed"],
                    "waveform_score_std": deployment_gate["waveform_score_std"],
                    "waveform_score_unique": deployment_gate["waveform_score_unique_rounded_1e6"],
                }
            )
            weights = summary["weights"]["candidate_three_channel_strictly_positive"]
            weight_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    **{f"weight_{name}": float(value) for name, value in weights.items()},
                }
            )
    per_seed = pd.concat(frames, ignore_index=True)
    summary = (
        per_seed.groupby(["deployment", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            r_at_1_mean=("r_at_1", "mean"),
            r_at_1_sd=("r_at_1", "std"),
            r_at_10_mean=("r_at_10", "mean"),
            r_at_10_sd=("r_at_10", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_sd=("median_rank", "std"),
        )
    )
    per_seed.to_csv(out / "heldout_retrieval_selected_methods_per_seed.csv", index=False)
    summary.to_csv(out / "heldout_retrieval_selected_methods_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(out / "waveform_gate_and_deployment_audit_by_seed.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "strictly_positive_weights_by_seed.csv", index=False)

    labels = {
        "waveform_only": "Waveform",
        "time_only": "Time",
        "sky_bayes_factor_only": "Sky BF",
        "time_sky_candidate_selected": "Time + sky",
        "candidate_three_channel_unconstrained": "3-ch free",
        "candidate_three_channel_strictly_positive": "3-ch positive",
    }
    colors = {"r_at_1": "#4C78A8", "r_at_10": "#E45756"}
    with base.plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = base.plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True, constrained_layout=True)
        rng = np.random.default_rng(20260722)
        for axis, deployment in zip(axes, ("GWTC3", "GWTC4")):
            frame = per_seed[per_seed["deployment"] == deployment]
            x = np.arange(len(methods), dtype=np.float64)
            width = 0.34
            for offset, metric in ((-width / 2, "r_at_1"), (width / 2, "r_at_10")):
                means = [frame.loc[frame.method == method, metric].mean() for method in methods]
                axis.bar(x + offset, means, width=width, color=colors[metric], alpha=0.78, label=metric.replace("r_at_", "R@"))
                for method_index, method in enumerate(methods):
                    values = frame.loc[frame.method == method, metric].to_numpy(dtype=np.float64)
                    jitter = rng.uniform(-0.035, 0.035, size=len(values))
                    axis.scatter(
                        np.full(len(values), x[method_index] + offset) + jitter,
                        values,
                        s=15,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.55,
                        zorder=3,
                    )
            axis.set_xticks(x, [labels[method] for method in methods], rotation=28, ha="right")
            axis.set_title("GWTC-3 / O3" if deployment == "GWTC3" else "GWTC-4.1 / O4a")
            axis.set_ylim(0, 0.62)
            axis.grid(axis="y", alpha=0.18)
        axes[0].set_ylabel("Held-out companion recall")
        axes[1].legend(frameon=False, loc="upper right")
        fig.suptitle("Run-matched real-noise injections: three formal training seeds", fontweight="bold")
        stem = figure_dir / "fig_v7_heldout_retrieval_multiseed"
        fig.savefig(stem.with_suffix(".pdf"), dpi=300)
        fig.savefig(stem.with_suffix(".png"), dpi=240)
        base.plt.close(fig)
    return {
        "selected_method_summary": summary.to_dict(orient="records"),
        "gate_by_seed": gate_rows,
        "strictly_positive_weights_by_seed": weight_rows,
    }


def audit_deployment(
    key: str,
    deployment: Any,
    input_root: Path,
    out: Path,
    figure_dir: Path,
    seeds: list[int],
) -> dict[str, Any]:
    cache, pe_manifest = base.load_catalog_pe(deployment)
    pe_manifest.to_csv(out / f"{key}_pe_manifest.csv", index=False)
    primary_method, primary_policy = frozen_consensus_method(input_root, key, seeds)
    scores = consensus_scores(input_root, key, seeds, primary_method)
    predictions = consensus_event_predictions(input_root, key, seeds)
    predictions.to_csv(out / f"{key}_consensus_event_waveform_predictions.csv", index=False)
    event_audit, event_audit_summary = event_prediction_pe_audit(predictions, cache)
    event_audit.to_csv(out / f"{key}_event_waveform_prediction_pe_audit.csv", index=False)
    scores = add_area_columns(
        scores,
        input_root / key / "shared/real_pe_sky_template_manifest.csv",
    )
    scores.to_parquet(out / f"{key}_consensus_all_strict_pair_scores.parquet", index=False)
    diagnostics_path = out / f"{key}_all_strict_pair_pe_diagnostics.parquet"
    if diagnostics_path.exists():
        diagnostics = pd.read_parquet(diagnostics_path)
    else:
        diagnostics = base.all_pair_pe_metrics(scores, cache)
        diagnostics.to_parquet(diagnostics_path, index=False)
    enrichment_path = out / f"{key}_pe_consistency_rank_enrichment.csv"
    if enrichment_path.exists():
        enrichment = pd.read_csv(enrichment_path)
    else:
        enrichment = base.pe_rank_enrichment(diagnostics)
        enrichment.to_csv(enrichment_path, index=False)
    mc_enrichment_path = out / f"{key}_chirp_mass_rank_enrichment.csv"
    if mc_enrichment_path.exists():
        mc_enrichment = pd.read_csv(mc_enrichment_path)
    else:
        mc_enrichment = chirp_mass_rank_enrichment(diagnostics)
        mc_enrichment.to_csv(mc_enrichment_path, index=False)

    top50 = scores.head(50).copy()
    top50.to_csv(out / f"{key}_consensus_top50.csv", index=False)
    shared_path = out / f"{key}_consensus_top50_shared_source_evidence.jsonl"
    if shared_path.exists():
        shared = pd.read_json(shared_path, orient="records", lines=True)
    else:
        shared = base.add_shared_evidence_to_candidates(top50, cache)
        shared.to_json(shared_path, orient="records", lines=True)
    one_d = diagnostics.assign(
        pair_key=diagnostics.apply(
            lambda row: "--".join(sorted((str(row.event_i), str(row.event_j)))), axis=1
        )
    )
    followup = top50.merge(
        one_d,
        on=["event_i", "event_j"],
        how="left",
        suffixes=("", "_pe"),
    ).merge(shared, on=["event_i", "event_j"], how="left", suffixes=("", "_shared"))
    followup["pe_followup_screen_pass"] = (
        (followup["chirp_mass_standardized_posterior_distance"] <= 2.0)
        & (followup["mass_ratio_standardized_posterior_distance"] <= 3.0)
        & (followup["chi_eff_standardized_posterior_distance"] <= 3.0)
        & followup["available"].fillna(False)
        & (followup["min_directional_log_evidence_all_bandwidths"].fillna(-np.inf) > 0)
    )
    followup["pe_followup_strict_mc_screen_pass"] = (
        followup["pe_followup_screen_pass"]
        & (followup["chirp_mass_standardized_posterior_distance"] <= 1.0)
    )
    followup.to_csv(out / f"{key}_consensus_top50_pe_followup.csv", index=False)
    pe_shortlist = followup[followup["pe_followup_screen_pass"]].copy()
    pe_shortlist_by_retrieval = pe_shortlist.sort_values(
        "rank", ascending=True, kind="stable"
    ).reset_index(drop=True)
    pe_shortlist_by_retrieval.insert(
        0, "pe_followup_rank", np.arange(1, len(pe_shortlist_by_retrieval) + 1)
    )
    pe_shortlist_by_retrieval = pe_shortlist_by_retrieval.rename(
        columns={"rank": "retrieval_rank"}
    )
    pe_shortlist_by_retrieval.to_csv(
        out / f"{key}_pe_confirmed_followup_shortlist_by_retrieval_rank.csv",
        index=False,
    )
    pe_shortlist = pe_shortlist.sort_values(
        ["min_directional_log_evidence_all_bandwidths", "rank"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    pe_shortlist.insert(0, "pe_followup_rank", np.arange(1, len(pe_shortlist) + 1))
    pe_shortlist = pe_shortlist.rename(columns={"rank": "retrieval_rank"})
    pe_shortlist.to_csv(out / f"{key}_pe_consistent_followup_shortlist.csv", index=False)
    top50.to_csv(out / f"{key}_raw_three_channel_triage_shortlist.csv", index=False)

    # Independently audit the validation-selected unconstrained three-channel
    # ranking. This does not change the predeclared positive-weight primary,
    # and PE is never used to select either ranking.
    comparison_method = "candidate_three_channel_unconstrained"
    comparison_scores = consensus_scores(input_root, key, seeds, comparison_method)
    comparison_scores = add_area_columns(
        comparison_scores,
        input_root / key / "shared/real_pe_sky_template_manifest.csv",
    )
    comparison_scores.to_parquet(
        out / f"{key}_unconstrained_three_channel_all_strict_pair_scores.parquet",
        index=False,
    )
    rankings_identical = bool(
        np.array_equal(scores["pair_key"].to_numpy(), comparison_scores["pair_key"].to_numpy())
        and np.allclose(
            scores["final_score"].to_numpy(dtype=np.float64),
            comparison_scores["final_score"].to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if rankings_identical:
        comparison_diagnostics = diagnostics.copy()
        comparison_enrichment = enrichment.copy()
        comparison_mc_enrichment = mc_enrichment.copy()
    else:
        comparison_diagnostics = base.all_pair_pe_metrics(comparison_scores, cache)
        comparison_enrichment = base.pe_rank_enrichment(comparison_diagnostics)
        comparison_mc_enrichment = chirp_mass_rank_enrichment(comparison_diagnostics)
    comparison_diagnostics.to_parquet(
        out / f"{key}_unconstrained_three_channel_all_strict_pair_pe_diagnostics.parquet",
        index=False,
    )
    comparison_enrichment.to_csv(
        out / f"{key}_unconstrained_three_channel_pe_consistency_rank_enrichment.csv",
        index=False,
    )
    comparison_mc_enrichment.to_csv(
        out / f"{key}_unconstrained_three_channel_chirp_mass_rank_enrichment.csv",
        index=False,
    )
    comparison_top50 = comparison_scores.head(50).copy()
    comparison_top50.to_csv(
        out / f"{key}_unconstrained_three_channel_consensus_top50.csv",
        index=False,
    )
    comparison_shared = (
        shared.copy()
        if rankings_identical
        else base.add_shared_evidence_to_candidates(comparison_top50, cache)
    )
    comparison_shared.to_json(
        out / f"{key}_unconstrained_three_channel_consensus_top50_shared_source_evidence.jsonl",
        orient="records",
        lines=True,
    )
    comparison_one_d = comparison_diagnostics.assign(
        pair_key=comparison_diagnostics.apply(
            lambda row: "--".join(sorted((str(row.event_i), str(row.event_j)))), axis=1
        )
    )
    comparison_followup = comparison_top50.merge(
        comparison_one_d,
        on=["event_i", "event_j"],
        how="left",
        suffixes=("", "_pe"),
    ).merge(
        comparison_shared,
        on=["event_i", "event_j"],
        how="left",
        suffixes=("", "_shared"),
    )
    comparison_followup["pe_followup_screen_pass"] = (
        (comparison_followup["chirp_mass_standardized_posterior_distance"] <= 2.0)
        & (comparison_followup["mass_ratio_standardized_posterior_distance"] <= 3.0)
        & (comparison_followup["chi_eff_standardized_posterior_distance"] <= 3.0)
        & comparison_followup["available"].fillna(False)
        & (
            comparison_followup["min_directional_log_evidence_all_bandwidths"].fillna(-np.inf)
            > 0
        )
    )
    comparison_followup.to_csv(
        out / f"{key}_unconstrained_three_channel_consensus_top50_pe_followup.csv",
        index=False,
    )
    external_crosscheck = external_candidate_crosscheck(
        key, scores, comparison_scores, diagnostics
    )
    external_crosscheck.to_csv(
        out / f"{key}_external_candidate_crosscheck_v7.csv", index=False
    )

    plot_v7_top5_posteriors(deployment, top50, cache, figure_dir)
    plot_v7_waveform_inputs(input_root, deployment, top50, figure_dir)
    base.plot_score_vs_pe(deployment, diagnostics, figure_dir)
    base.plot_top5_sky_audit(deployment, diagnostics, figure_dir)
    plot_top5_waveform_prediction_vs_pe(deployment, top50, cache, predictions, figure_dir)
    plot_event_prediction_pe_audit(deployment, event_audit, figure_dir)
    plot_top5_score_contributions(
        deployment,
        scores,
        figure_dir,
        "primary",
        "frozen primary policy",
    )
    plot_top5_score_contributions(
        deployment,
        comparison_scores,
        figure_dir,
        "unconstrained_three_channel",
        "unconstrained three-channel comparison",
    )

    valid = diagnostics[np.isfinite(diagnostics["chirp_mass_standardized_posterior_distance"])].copy()
    rho_final = stats.spearmanr(
        valid["final_score"], -valid["chirp_mass_standardized_posterior_distance"]
    ).statistic
    rho_waveform = stats.spearmanr(
        valid["waveform_score"], -valid["chirp_mass_standardized_posterior_distance"]
    ).statistic
    top10_row = enrichment.loc[enrichment["top_n"] == min(10, len(valid))].iloc[0]
    comparison_valid = comparison_diagnostics[
        np.isfinite(comparison_diagnostics["chirp_mass_standardized_posterior_distance"])
    ].copy()
    comparison_top10_row = comparison_enrichment.loc[
        comparison_enrichment["top_n"] == min(10, len(comparison_valid))
    ].iloc[0]
    comparison_rho_final = stats.spearmanr(
        comparison_valid["final_score"],
        -comparison_valid["chirp_mass_standardized_posterior_distance"],
    ).statistic
    historical = scores[
        (
            (scores["event_i"].astype(str) == "GW170104")
            & (scores["event_j"].astype(str) == "GW170814")
        )
        | (
            (scores["event_i"].astype(str) == "GW170814")
            & (scores["event_j"].astype(str) == "GW170104")
        )
    ]
    top_columns = [
        name
        for name in (
            "rank",
            "event_i",
            "event_j",
            "final_score",
            "waveform_score",
            "time_score",
            "sky_score",
            "waveform_contribution",
            "time_contribution",
            "sky_contribution",
            "sky_bayes_factor",
            "cosine_overlap",
        )
        if name in scores.columns
    ]
    return {
        "label": deployment.label,
        "primary_consensus_policy": primary_policy,
        "event_waveform_prediction_pe_audit": event_audit_summary,
        "seeds": seeds,
        "strict_pairs": int(len(scores)),
        "pe_available_events": int(pe_manifest["pe_available"].sum()),
        "top10_3sigma_pass": int(diagnostics.head(10)["intrinsic_3sigma_consistent"].sum()),
        "top10_dmc_le_1": int(
            (diagnostics.head(10)["chirp_mass_standardized_posterior_distance"] <= 1.0).sum()
        ),
        "top10_dmc_le_2": int(
            (diagnostics.head(10)["chirp_mass_standardized_posterior_distance"] <= 2.0).sum()
        ),
        "chirp_mass_rank_enrichment": mc_enrichment.to_dict(orient="records"),
        "top10_pe_enrichment": float(top10_row["enrichment_over_catalog"]),
        "top10_pe_enrichment_p": float(top10_row["hypergeometric_one_sided_p"]),
        "spearman_final_score_vs_negative_dmc": float(rho_final),
        "spearman_waveform_score_vs_negative_dmc": float(rho_waveform),
        "consensus_top50_pe_followup_count": int(followup["pe_followup_screen_pass"].sum()),
        "consensus_top50_strict_mc_followup_count": int(
            followup["pe_followup_strict_mc_screen_pass"].sum()
        ),
        "consensus_top10_triage_pairs": scores.head(10)[top_columns].to_dict(orient="records"),
        "gw170104_gw170814": (
            {"found": True, **historical.iloc[0][top_columns].to_dict()}
            if len(historical)
            else {"found": False}
        ),
        "top_pe_followup_candidates": pe_shortlist.head(20).to_dict(orient="records"),
        "unconstrained_three_channel_sensitivity": {
            "method": comparison_method,
            "is_primary_method": bool(primary_method == comparison_method),
            "pe_not_used_for_selection": True,
            "top10_3sigma_pass": int(
                comparison_diagnostics.head(10)["intrinsic_3sigma_consistent"].sum()
            ),
            "top10_pe_enrichment": float(comparison_top10_row["enrichment_over_catalog"]),
            "top10_pe_enrichment_p": float(
                comparison_top10_row["hypergeometric_one_sided_p"]
            ),
            "spearman_final_score_vs_negative_dmc": float(comparison_rho_final),
            "consensus_top50_pe_followup_count": int(
                comparison_followup["pe_followup_screen_pass"].sum()
            ),
            "consensus_top10_triage_pairs": comparison_scores.head(10)[top_columns].to_dict(
                orient="records"
            ),
            "chirp_mass_rank_enrichment": comparison_mc_enrichment.to_dict(orient="records"),
        },
        "external_candidate_crosscheck": external_crosscheck.to_dict(orient="records"),
    }


def report_cn(summary: dict[str, Any]) -> str:
    lines = [
        "# v7 真实目录三通道检索与独立 PE 验收",
        "",
        "本报告以预先冻结的严格正权重三通道排名作为 v7 主分析，并另报 validation-selected unconstrained 三通道和 time+sky 对照。真实 PE 没有参与 encoder、waveform calibration、融合权重或主方法选择。",
        "",
        "高 retrieval rank 不等于透镜探测。只有排名后仍通过内禀参数一致性审计的 pair 才进入 coherent Bayesian lensing follow-up。",
        "",
        "## Held-out real-noise injection 检索",
        "",
    ]
    retrieval_rows = summary["heldout_retrieval_audit"]["selected_method_summary"]
    for deployment, label in (("GWTC3", "GWTC-3 / O3"), ("GWTC4", "GWTC-4.1 / O4a")):
        lines.append(f"### {label}")
        lines.append("")
        for method in (
            "waveform_only",
            "time_only",
            "sky_bayes_factor_only",
            "time_sky_candidate_selected",
            "candidate_three_channel_strictly_positive",
        ):
            row = next(
                item
                for item in retrieval_rows
                if item["deployment"] == deployment and item["method"] == method
            )
            lines.append(
                f"- {method}: R@1={row['r_at_1_mean']:.3f} +/- {row['r_at_1_sd']:.3f}, "
                f"R@10={row['r_at_10_mean']:.3f} +/- {row['r_at_10_sd']:.3f}。"
            )
        lines.append("")
    for key in ("gwtc3", "gwtc4"):
        item = summary[key]
        lines.extend(
            [
                f"## {item['label']}",
                "",
                f"- 冻结主排序方法：{item['primary_consensus_policy']['method']}",
                f"- 主排序选择原因：{item['primary_consensus_policy']['reason']}",
                f"- 严格双探测器 BBH pair 数：{item['strict_pairs']}",
                f"- top-10 中 D(Mc,q,chi_eff) 均不超过 3 的 pair：{item['top10_3sigma_pass']} / 10",
                f"- top-10 中 D_Mc <= 1：{item['top10_dmc_le_1']} / 10；D_Mc <= 2：{item['top10_dmc_le_2']} / 10",
                f"- top-10 PE-consistency enrichment：{item['top10_pe_enrichment']:.3f}",
                f"- enrichment 的单侧超几何 p 值（仅作审计，不是 detection p-value）：{item['top10_pe_enrichment_p']:.3g}",
                f"- final score 与 -D_Mc 的 Spearman 相关：{item['spearman_final_score_vs_negative_dmc']:.3f}",
                f"- waveform score 与 -D_Mc 的 Spearman 相关：{item['spearman_waveform_score_vs_negative_dmc']:.3f}",
                f"- waveform head 与 PE 啁啾质量中位数的事件级 Spearman 相关：{item['event_waveform_prediction_pe_audit']['chirp_mass_spearman_prediction_vs_pe']:.3f}",
                f"- consensus top-50 中通过预定 PE follow-up screen 的数量：{item['consensus_top50_pe_followup_count']}",
                f"- unconstrained 三通道对照 top-10 PE enrichment：{item['unconstrained_three_channel_sensitivity']['top10_pe_enrichment']:.3f}",
                f"- unconstrained 三通道对照 top-50 通过 PE screen：{item['unconstrained_three_channel_sensitivity']['consensus_top50_pe_followup_count']}",
                "",
            ]
        )
        for candidate in item["external_candidate_crosscheck"]:
            if candidate["present_in_strict_catalog"]:
                lines.append(
                    f"- 外部交叉核对 {candidate['requested_event_i']}--{candidate['requested_event_j']}："
                    f"主 rank={int(candidate['primary_rank'])}，unconstrained 三通道 rank={int(candidate['unconstrained_three_channel_rank'])}。"
                )
        lines.append("")
    lines.extend(
        [
            "## 解释规则",
            "",
            "如果 top pairs 没有相对于全目录产生 PE-consistency enrichment，结论必须写成：当前三通道部署没有产生通过 PE 一致性的可信 follow-up candidate。不得使用这些真实候选反向调整三通道权重。",
            "",
            "三维 shared-source evidence 使用 native-prior-corrected symmetric KDE 近似并扫描带宽；它是 follow-up diagnostic，不代替 GOLUM/hanabi 的完整 coherent lensing Bayes factor。",
            "",
        ]
    )
    return "\n".join(lines)


def report_en(summary: dict[str, Any]) -> str:
    lines = [
        "# v7 real-catalog three-channel retrieval and independent PE audit",
        "",
        "The predeclared v7 primary analysis uses validation-selected, strictly-positive waveform, time-delay, and sky weights. Unconstrained three-channel and time+sky rankings are reported as comparisons. Real-event PE intrinsic parameters are not used to train the encoder, calibrate channels, select fusion weights, or choose the primary method.",
        "",
        "A high retrieval rank is not a lensing detection. Only pairs that also pass the independent intrinsic-parameter screen are retained as candidates for coherent Bayesian follow-up.",
        "",
        "## Held-out real-noise injection retrieval",
        "",
    ]
    retrieval_rows = summary["heldout_retrieval_audit"]["selected_method_summary"]
    for deployment, label in (("GWTC3", "GWTC-3 / O3"), ("GWTC4", "GWTC-4.1 / O4a")):
        lines.append(f"### {label}")
        lines.append("")
        for method in (
            "waveform_only",
            "time_only",
            "sky_bayes_factor_only",
            "time_sky_candidate_selected",
            "candidate_three_channel_strictly_positive",
        ):
            row = next(
                item
                for item in retrieval_rows
                if item["deployment"] == deployment and item["method"] == method
            )
            lines.append(
                f"- {method}: R@1={row['r_at_1_mean']:.3f} +/- {row['r_at_1_sd']:.3f}; "
                f"R@10={row['r_at_10_mean']:.3f} +/- {row['r_at_10_sd']:.3f}."
            )
        lines.append("")
    for key in ("gwtc3", "gwtc4"):
        item = summary[key]
        lines.extend(
            [
                f"## {item['label']}",
                "",
                f"- Frozen primary ranking method: {item['primary_consensus_policy']['method']}",
                f"- Primary-policy reason: {item['primary_consensus_policy']['reason']}",
                f"- Strict two-detector BBH pairs: {item['strict_pairs']}",
                f"- Top-10 pairs with D(Mc,q,chi_eff) <= 3: {item['top10_3sigma_pass']} / 10",
                f"- Top-10 pairs with D_Mc <= 1: {item['top10_dmc_le_1']} / 10; D_Mc <= 2: {item['top10_dmc_le_2']} / 10",
                f"- Top-10 PE-consistency enrichment: {item['top10_pe_enrichment']:.3f}",
                f"- One-sided hypergeometric audit p-value: {item['top10_pe_enrichment_p']:.3g}",
                f"- Spearman(final score, -D_Mc): {item['spearman_final_score_vs_negative_dmc']:.3f}",
                f"- Spearman(waveform score, -D_Mc): {item['spearman_waveform_score_vs_negative_dmc']:.3f}",
                f"- Event-level Spearman(waveform-head Mc, PE-median Mc): {item['event_waveform_prediction_pe_audit']['chirp_mass_spearman_prediction_vs_pe']:.3f}",
                f"- PE-screened pairs in the consensus top 50: {item['consensus_top50_pe_followup_count']}",
                f"- Unconstrained three-channel comparison top-10 PE enrichment: {item['unconstrained_three_channel_sensitivity']['top10_pe_enrichment']:.3f}",
                f"- Unconstrained three-channel comparison PE-screened pairs in top 50: {item['unconstrained_three_channel_sensitivity']['consensus_top50_pe_followup_count']}",
                "",
            ]
        )
        for candidate in item["external_candidate_crosscheck"]:
            if candidate["present_in_strict_catalog"]:
                lines.append(
                    f"- External cross-check {candidate['requested_event_i']}--{candidate['requested_event_j']}: "
                    f"primary rank={int(candidate['primary_rank'])}, unconstrained three-channel rank={int(candidate['unconstrained_three_channel_rank'])}."
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "If the top triage pairs are not enriched in PE consistency, this deployment produced no physically credible follow-up candidate. Real pairs are never used to retune the triage score until agreement appears.",
            "",
            "The three-dimensional shared-source evidence is a symmetrized, native-prior-corrected KDE diagnostic with a bandwidth scan. It does not replace a coherent GOLUM/hanabi lensing Bayes factor.",
            "",
        ]
    )
    return "\n".join(lines)


def package(input_root: Path, out: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    include_scripts = [
        REPO / "scripts/real_search/34_generate_physical_h1l1_source_bank.py",
        REPO / "scripts/real_search/35_real_noise_injection_v5_physical_source.py",
        REPO / "scripts/real_search/36_materialize_multinoise_v5.py",
        REPO / "scripts/real_search/37_unified_intrinsic_multitask_pilot.py",
        REPO / "scripts/real_search/45_real_noise_injection_v7_peak2s.py",
        REPO / "scripts/real_search/46_physical_deployment_v7_pe_audit.py",
        REPO / "scripts/real_search/40_select_waveform_pilot_validation.py",
        REPO / "scripts/real_search/unified_v7_common.py",
        REPO / "scripts/real_search/physical_common.py",
        REPO / "scripts/real_search/reproduce_v7_peak2s.sh",
        REPO / "docs/methods/real_noise_v7_peak2s_protocol_cn.md",
        REPO / "results/real_noise_injection_v7_peak2s_20260722/development/frozen_selection_v7_peak2s.json",
        REPO / "results/real_noise_injection_v7_peak2s_20260722/development/backbone_validation_comparison_v7_peak2s.csv",
    ]
    with tarfile.open(package_path, "w:gz") as archive:
        for path in include_scripts:
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(REPO)))
        root_level_names = {
            "run_config_v7.json",
            "heldout_test_retrieval_metrics_per_seed_v7.csv",
            "heldout_test_retrieval_metrics_across_seed_v7.csv",
            "heldout_test_pair_level_metrics_per_seed_v7.csv",
            "heldout_test_pair_level_metrics_across_seed_v7.csv",
            "heldout_test_bootstrap_95ci_per_seed_v7.csv",
        }
        per_seed_names = {
            "seed_summary_v7.json",
            "split_integrity_audit_v7.json",
            "waveform_channel_calibration_v7.json",
            "waveform_gate_validation_v7.json",
            "selected_weights_v7.json",
            "real_waveform_deployment_audit_unified_v7.json",
            "heldout_test_retrieval_metrics_v7.csv",
            "heldout_test_pair_level_metrics_v7.csv",
            "heldout_test_bootstrap_95ci_v7.csv",
            "candidate_shortlist_strict_primary_v7.csv",
            "candidate_shortlist_all_primary_v7.csv",
            "real_waveform_embeddings_unified_v7.parquet",
            "real_waveform_similarity_unified_v7.parquet",
        }
        real_score_marker = "real_pair_scores_strict_h1l1_bbh_"
        shortlist_marker = "candidate_shortlist_strict_h1l1_bbh_"
        for path in input_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(input_root)
            include = (
                path.name in root_level_names
                or path.name in per_seed_names
                or real_score_marker in path.name
                or shortlist_marker in path.name
            )
            if not include:
                continue
            archive.add(path, arcname=str(relative))
        for path in out.rglob("*"):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(input_root)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    args.input_root = args.input_root.resolve()
    args.out = args.out.resolve()
    args.package = args.package.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out / "figures"
    figure_dir.mkdir(exist_ok=True)
    base.set_style()
    retrieval_audit = heldout_retrieval_audit(
        args.input_root, args.out, figure_dir, args.seeds
    )
    summary = {
        key: audit_deployment(
            key, deployment, args.input_root, args.out, figure_dir, args.seeds
        )
        for key, deployment in base.DEPLOYMENTS.items()
    }
    summary["methodology"] = {
        "triage_channels": [
            "calibrated waveform evidence",
            "run-conditioned time-delay likelihood ratio",
            "HEALPix common-source sky Bayes factor",
        ],
        "real_pe_used_for_retrieval_tuning": False,
        "distance_included": False,
        "detection_claim": False,
        "screen": "D_Mc_det <= 2, D_q <= 3, D_chi_eff <= 3, and both directional native-prior-corrected shared-source log evidences > 0 at every predeclared KDE bandwidth; D_Mc_det <= 1 is reported as a stricter sensitivity screen",
    }
    summary["heldout_retrieval_audit"] = retrieval_audit
    base.write_json(args.out / "pe_followup_summary_v7.json", summary)
    cn = report_cn(summary)
    (args.out / "physical_deployment_v7_pe_audit_report_cn.md").write_text(cn, encoding="utf-8")
    (args.out / "physical_deployment_v7_pe_audit_report_en.md").write_text(
        report_en(summary), encoding="utf-8"
    )
    package(args.input_root, args.out, args.package)
    print(json.dumps({"status": "complete", "out": str(args.out), "package": str(args.package)}, indent=2))


if __name__ == "__main__":
    main()
