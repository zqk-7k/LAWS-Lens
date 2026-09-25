#!/usr/bin/env python3
"""Independent PE follow-up audit for the frozen v6 real-catalog ranking."""

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
DEFAULT_INPUT = REPO / "results/real_noise_injection_v6_unified_physics_20260721"
DEFAULT_OUT = DEFAULT_INPUT / "pe_followup"
DEFAULT_PACKAGE = REPO / "packages/real_noise_injection_v6_unified_physics_20260721_deliverables.tar.gz"
DEFAULT_SEEDS = (202607231, 202607232, 202607233)
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
    "v4_pe_helpers_for_v6",
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
            / f"seed_{seed}/results/real_pair_scores_strict_h1l1_bbh_{method}_v6.parquet"
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
                / f"seed_{seed}/seed_summary_v6.json"
            ).read_text(encoding="utf-8")
        )
        for seed in seeds
    ]
    waveform_seed_pass = [bool(item["waveform_eligible_for_primary"]) for item in summaries]
    if all(waveform_seed_pass):
        method = "candidate_three_channel_unconstrained"
        reason = "All formal seeds passed the frozen waveform validation/deployment gates."
    else:
        method = "time_sky_candidate_selected"
        reason = "At least one formal seed failed a frozen waveform gate; the common cross-seed primary falls back to time+sky."
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
    forced: pd.DataFrame,
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
        forced_match = locate(forced)
        diagnostic_match = locate(diagnostics)
        record: dict[str, Any] = {
            "deployment": deployment_key,
            "requested_event_i": event_a,
            "requested_event_j": event_b,
            "literature_context": source,
            "present_in_strict_catalog": bool(len(primary_match)),
            "primary_rank": int(primary_match.iloc[0]["rank"]) if len(primary_match) else np.nan,
            "strictly_positive_three_channel_rank": int(forced_match.iloc[0]["rank"]) if len(forced_match) else np.nan,
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
        path = input_root / deployment / f"seed_{seed}/features/real_waveform_embeddings_unified_v6.parquet"
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
    diagnostics = base.all_pair_pe_metrics(scores, cache)
    diagnostics.to_parquet(out / f"{key}_all_strict_pair_pe_diagnostics.parquet", index=False)
    enrichment = base.pe_rank_enrichment(diagnostics)
    enrichment.to_csv(out / f"{key}_pe_consistency_rank_enrichment.csv", index=False)

    top50 = scores.head(50).copy()
    top50.to_csv(out / f"{key}_consensus_top50.csv", index=False)
    shared = base.add_shared_evidence_to_candidates(top50, cache)
    shared.to_json(
        out / f"{key}_consensus_top50_shared_source_evidence.jsonl",
        orient="records",
        lines=True,
    )
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
        followup["intrinsic_3sigma_consistent"].fillna(False)
        & followup["available"].fillna(False)
        & (followup["min_directional_log_evidence_all_bandwidths"].fillna(-np.inf) > 0)
    )
    followup.to_csv(out / f"{key}_consensus_top50_pe_followup.csv", index=False)
    pe_shortlist = followup[followup["pe_followup_screen_pass"]].copy()
    pe_shortlist = pe_shortlist.sort_values(
        ["min_directional_log_evidence_all_bandwidths", "rank"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    pe_shortlist.insert(0, "pe_followup_rank", np.arange(1, len(pe_shortlist) + 1))
    pe_shortlist = pe_shortlist.rename(columns={"rank": "retrieval_rank"})
    pe_shortlist.to_csv(out / f"{key}_pe_consistent_followup_shortlist.csv", index=False)

    # Independently audit the predeclared strictly-positive three-channel
    # sensitivity ranking when the primary policy falls back to time+sky.  This
    # does not change the primary method and PE is never used to select either
    # ranking; it exposes whether adding waveform evidence improves or worsens
    # physical consistency on the real catalog.
    forced_method = "candidate_three_channel_strictly_positive"
    forced_scores = consensus_scores(input_root, key, seeds, forced_method)
    forced_scores = add_area_columns(
        forced_scores,
        input_root / key / "shared/real_pe_sky_template_manifest.csv",
    )
    forced_scores.to_parquet(
        out / f"{key}_forced_three_channel_all_strict_pair_scores.parquet",
        index=False,
    )
    forced_diagnostics = base.all_pair_pe_metrics(forced_scores, cache)
    forced_diagnostics.to_parquet(
        out / f"{key}_forced_three_channel_all_strict_pair_pe_diagnostics.parquet",
        index=False,
    )
    forced_enrichment = base.pe_rank_enrichment(forced_diagnostics)
    forced_enrichment.to_csv(
        out / f"{key}_forced_three_channel_pe_consistency_rank_enrichment.csv",
        index=False,
    )
    forced_top50 = forced_scores.head(50).copy()
    forced_top50.to_csv(
        out / f"{key}_forced_three_channel_consensus_top50.csv",
        index=False,
    )
    forced_shared = base.add_shared_evidence_to_candidates(forced_top50, cache)
    forced_shared.to_json(
        out / f"{key}_forced_three_channel_consensus_top50_shared_source_evidence.jsonl",
        orient="records",
        lines=True,
    )
    forced_one_d = forced_diagnostics.assign(
        pair_key=forced_diagnostics.apply(
            lambda row: "--".join(sorted((str(row.event_i), str(row.event_j)))), axis=1
        )
    )
    forced_followup = forced_top50.merge(
        forced_one_d,
        on=["event_i", "event_j"],
        how="left",
        suffixes=("", "_pe"),
    ).merge(
        forced_shared,
        on=["event_i", "event_j"],
        how="left",
        suffixes=("", "_shared"),
    )
    forced_followup["pe_followup_screen_pass"] = (
        forced_followup["intrinsic_3sigma_consistent"].fillna(False)
        & forced_followup["available"].fillna(False)
        & (
            forced_followup["min_directional_log_evidence_all_bandwidths"].fillna(-np.inf)
            > 0
        )
    )
    forced_followup.to_csv(
        out / f"{key}_forced_three_channel_consensus_top50_pe_followup.csv",
        index=False,
    )
    external_crosscheck = external_candidate_crosscheck(
        key, scores, forced_scores, diagnostics
    )
    external_crosscheck.to_csv(
        out / f"{key}_external_candidate_crosscheck_v6.csv", index=False
    )

    base.plot_top5_posteriors(deployment, top50, cache, figure_dir)
    base.plot_waveform_inputs(input_root, deployment, top50, figure_dir)
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
        forced_scores,
        figure_dir,
        "forced_three_channel",
        "strictly positive three-channel sensitivity",
    )

    valid = diagnostics[np.isfinite(diagnostics["chirp_mass_standardized_posterior_distance"])].copy()
    rho_final = stats.spearmanr(
        valid["final_score"], -valid["chirp_mass_standardized_posterior_distance"]
    ).statistic
    rho_waveform = stats.spearmanr(
        valid["waveform_score"], -valid["chirp_mass_standardized_posterior_distance"]
    ).statistic
    top10_row = enrichment.loc[enrichment["top_n"] == min(10, len(valid))].iloc[0]
    forced_valid = forced_diagnostics[
        np.isfinite(forced_diagnostics["chirp_mass_standardized_posterior_distance"])
    ].copy()
    forced_top10_row = forced_enrichment.loc[
        forced_enrichment["top_n"] == min(10, len(forced_valid))
    ].iloc[0]
    forced_rho_final = stats.spearmanr(
        forced_valid["final_score"],
        -forced_valid["chirp_mass_standardized_posterior_distance"],
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
        "top10_pe_enrichment": float(top10_row["enrichment_over_catalog"]),
        "top10_pe_enrichment_p": float(top10_row["hypergeometric_one_sided_p"]),
        "spearman_final_score_vs_negative_dmc": float(rho_final),
        "spearman_waveform_score_vs_negative_dmc": float(rho_waveform),
        "consensus_top50_pe_followup_count": int(followup["pe_followup_screen_pass"].sum()),
        "consensus_top10_triage_pairs": scores.head(10)[top_columns].to_dict(orient="records"),
        "gw170104_gw170814": (
            {"found": True, **historical.iloc[0][top_columns].to_dict()}
            if len(historical)
            else {"found": False}
        ),
        "top_pe_followup_candidates": pe_shortlist.head(20).to_dict(orient="records"),
        "forced_three_channel_sensitivity": {
            "method": forced_method,
            "is_primary_method": bool(primary_method == forced_method),
            "pe_not_used_for_selection": True,
            "top10_3sigma_pass": int(
                forced_diagnostics.head(10)["intrinsic_3sigma_consistent"].sum()
            ),
            "top10_pe_enrichment": float(forced_top10_row["enrichment_over_catalog"]),
            "top10_pe_enrichment_p": float(
                forced_top10_row["hypergeometric_one_sided_p"]
            ),
            "spearman_final_score_vs_negative_dmc": float(forced_rho_final),
            "consensus_top50_pe_followup_count": int(
                forced_followup["pe_followup_screen_pass"].sum()
            ),
            "consensus_top10_triage_pairs": forced_scores.head(10)[top_columns].to_dict(
                orient="records"
            ),
        },
        "external_candidate_crosscheck": external_crosscheck.to_dict(orient="records"),
    }


def report_cn(summary: dict[str, Any]) -> str:
    lines = [
        "# v6 真实目录三通道检索与独立 PE 验收",
        "",
        "本报告按预定 Gate 在冻结三通道排名与 time+sky 回退排名之间确定主结果，并始终另报严格正权重三通道敏感性结果。真实 PE 没有参与 encoder、waveform calibration、融合权重或主方法选择。",
        "",
        "高 retrieval rank 不等于透镜探测。只有排名后仍通过内禀参数一致性审计的 pair 才进入 coherent Bayesian lensing follow-up。",
        "",
    ]
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
                f"- top-10 PE-consistency enrichment：{item['top10_pe_enrichment']:.3f}",
                f"- enrichment 的单侧超几何 p 值（仅作审计，不是 detection p-value）：{item['top10_pe_enrichment_p']:.3g}",
                f"- final score 与 -D_Mc 的 Spearman 相关：{item['spearman_final_score_vs_negative_dmc']:.3f}",
                f"- waveform score 与 -D_Mc 的 Spearman 相关：{item['spearman_waveform_score_vs_negative_dmc']:.3f}",
                f"- waveform head 与 PE 啁啾质量中位数的事件级 Spearman 相关：{item['event_waveform_prediction_pe_audit']['chirp_mass_spearman_prediction_vs_pe']:.3f}",
                f"- consensus top-50 中通过预定 PE follow-up screen 的数量：{item['consensus_top50_pe_followup_count']}",
                f"- 严格正权重三通道补充排名 top-10 PE enrichment：{item['forced_three_channel_sensitivity']['top10_pe_enrichment']:.3f}",
                f"- 严格正权重三通道补充排名 top-50 通过 PE screen：{item['forced_three_channel_sensitivity']['consensus_top50_pe_followup_count']}",
                "",
            ]
        )
        for candidate in item["external_candidate_crosscheck"]:
            if candidate["present_in_strict_catalog"]:
                lines.append(
                    f"- 外部交叉核对 {candidate['requested_event_i']}--{candidate['requested_event_j']}："
                    f"主 rank={int(candidate['primary_rank'])}，严格正权重三通道 rank={int(candidate['strictly_positive_three_channel_rank'])}。"
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
        "# v6 real-catalog three-channel retrieval and independent PE audit",
        "",
        "The predeclared Gate selects either the frozen three-channel ranking or the time+sky fallback as the primary result; a strictly-positive three-channel sensitivity ranking is always reported separately. Real-event PE intrinsic parameters are not used to train the encoder, calibrate channels, select fusion weights, or choose the primary method.",
        "",
        "A high retrieval rank is not a lensing detection. Only pairs that also pass the independent intrinsic-parameter screen are retained as candidates for coherent Bayesian follow-up.",
        "",
    ]
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
                f"- Top-10 PE-consistency enrichment: {item['top10_pe_enrichment']:.3f}",
                f"- One-sided hypergeometric audit p-value: {item['top10_pe_enrichment_p']:.3g}",
                f"- Spearman(final score, -D_Mc): {item['spearman_final_score_vs_negative_dmc']:.3f}",
                f"- Spearman(waveform score, -D_Mc): {item['spearman_waveform_score_vs_negative_dmc']:.3f}",
                f"- Event-level Spearman(waveform-head Mc, PE-median Mc): {item['event_waveform_prediction_pe_audit']['chirp_mass_spearman_prediction_vs_pe']:.3f}",
                f"- PE-screened pairs in the consensus top 50: {item['consensus_top50_pe_followup_count']}",
                f"- Strictly-positive three-channel sensitivity top-10 PE enrichment: {item['forced_three_channel_sensitivity']['top10_pe_enrichment']:.3f}",
                f"- Strictly-positive three-channel sensitivity PE-screened pairs in top 50: {item['forced_three_channel_sensitivity']['consensus_top50_pe_followup_count']}",
                "",
            ]
        )
        for candidate in item["external_candidate_crosscheck"]:
            if candidate["present_in_strict_catalog"]:
                lines.append(
                    f"- External cross-check {candidate['requested_event_i']}--{candidate['requested_event_j']}: "
                    f"primary rank={int(candidate['primary_rank'])}, strictly-positive three-channel rank={int(candidate['strictly_positive_three_channel_rank'])}."
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
        REPO / "scripts/real_search/38_real_noise_injection_v6_unified_physics.py",
        REPO / "scripts/real_search/39_physical_deployment_v6_pe_audit.py",
        REPO / "scripts/real_search/40_select_waveform_pilot_validation.py",
        REPO / "scripts/real_search/42_prebuild_v6_training_data.py",
        REPO / "scripts/real_search/unified_v6_common.py",
        REPO / "scripts/real_search/physical_common.py",
        REPO / "docs/methods/real_noise_v6_unified_physics_protocol_cn.md",
    ]
    with tarfile.open(package_path, "w:gz") as archive:
        for path in include_scripts:
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(REPO)))
        for path in input_root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in {"matchroots", "waveform_gate"} for part in path.parts):
                continue
            if path.suffix in {".npy", ".h5", ".hdf5"}:
                continue
            archive.add(path, arcname=str(path.relative_to(REPO)))
        for path in out.rglob("*"):
            if path.is_file() and not str(path).startswith(str(input_root)):
                archive.add(path, arcname=str(path.relative_to(REPO)))


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
        "screen": "D <= 3 for Mc_det, q, chi_eff and both directional native-prior-corrected shared-source log evidences > 0 at every predeclared KDE bandwidth",
    }
    base.write_json(args.out / "pe_followup_summary_v6.json", summary)
    cn = report_cn(summary)
    (args.out / "physical_deployment_v6_pe_audit_report_cn.md").write_text(cn, encoding="utf-8")
    (args.out / "physical_deployment_v6_pe_audit_report_en.md").write_text(
        report_en(summary), encoding="utf-8"
    )
    package(args.input_root, args.out, args.package)
    print(json.dumps({"status": "complete", "out": str(args.out), "package": str(args.package)}, indent=2))


if __name__ == "__main__":
    main()
