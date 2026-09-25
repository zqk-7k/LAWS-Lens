#!/usr/bin/env python3
"""Aggregate, audit, plot, document, and package unified sky-v8 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import pyarrow.parquet as pq
from scipy.stats import hypergeom


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/unified_sky_v8_20260724"
V7_GWTC = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
V7_ET = REPO / "results/et3_v7_aligned_20260723/formal"
PE_ROOT = V7_GWTC / "pe_followup"
PACKAGE_PATH = REPO / "packages/unified_sky_v8_20260724_deliverables.tar.gz"
GWTC_SEEDS = (202607241, 202607242, 202607243)
ET_SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)

DEPLOYMENT_LABELS = {
    "ET3": "ET-3",
    "gwtc3": "GWTC-3/O3",
    "gwtc4": "GWTC-4.1/O4a",
}
METHODS = {
    "waveform": {
        "ET3": "waveform_only",
        "gwtc3": "waveform_only",
        "gwtc4": "waveform_only",
    },
    "time": {
        "ET3": "time_only",
        "gwtc3": "time_only",
        "gwtc4": "time_only",
    },
    "sky": {
        "ET3": "sky_only",
        "gwtc3": "sky_only",
        "gwtc4": "sky_only",
    },
    "time+sky": {
        "ET3": "time_sky_validation_selected",
        "gwtc3": "time_sky_retrieval_selected",
        "gwtc4": "time_sky_retrieval_selected",
    },
    "three-channel": {
        "ET3": "three_channel_strict_positive",
        "gwtc3": "retrieval_three_channel_strict_positive",
        "gwtc4": "retrieval_three_channel_strict_positive",
    },
}
PALETTE = {
    "waveform": "#2C6E9B",
    "time": "#D5842F",
    "sky": "#47956F",
    "time+sky": "#8B78A5",
    "three-channel": "#B4433C",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def pair_key(frame: pd.DataFrame) -> np.ndarray:
    left = frame["event_i"].astype(str).to_numpy()
    right = frame["event_j"].astype(str).to_numpy()
    return np.where(left < right, left + "||" + right, right + "||" + left)


def load_retrieval(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    et = pd.read_csv(root / "et3/et3_retrieval_metrics_per_seed_v8.csv")
    et = et[et["deployment"] == "ET3"].copy()
    gwtc = []
    for deployment in ("gwtc3", "gwtc4"):
        frame = pd.read_csv(
            root / deployment / "heldout_test_retrieval_metrics_per_seed_v8.csv"
        )
        gwtc.append(frame)
    raw = pd.concat([et, *gwtc], ignore_index=True)
    rows = []
    for deployment in ("ET3", "gwtc3", "gwtc4"):
        part = raw[(raw["deployment"] == deployment) & (raw["subset"] == "overall")]
        for canonical, mapping in METHODS.items():
            selected = part[part["method"] == mapping[deployment]].copy()
            if selected.empty:
                raise RuntimeError(f"Missing {deployment}/{mapping[deployment]}")
            selected["canonical_method"] = canonical
            rows.append(selected)
    canonical = pd.concat(rows, ignore_index=True)
    summary = (
        canonical.groupby(["deployment", "canonical_method"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_queries=("n_queries", "first"),
            r_at_1_mean=("r_at_1", "mean"),
            r_at_1_std=("r_at_1", "std"),
            r_at_5_mean=("r_at_5", "mean"),
            r_at_5_std=("r_at_5", "std"),
            r_at_10_mean=("r_at_10", "mean"),
            r_at_10_std=("r_at_10", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
        )
    )
    canonical.to_csv(root / "unified_retrieval_metrics_per_seed_v8.csv", index=False)
    summary.to_csv(root / "unified_retrieval_metrics_summary_v8.csv", index=False)
    return canonical, summary


def make_v7_v8_comparison(
    root: Path,
    v8_summary: pd.DataFrame,
) -> pd.DataFrame:
    v7_frames = []
    et = pd.read_csv(V7_ET / "et3_retrieval_metrics_per_seed.csv")
    et = et[(et["deployment"] == "ET3") & (et["subset"] == "overall")].copy()
    v7_frames.append(et)
    for deployment in ("gwtc3", "gwtc4"):
        frame = pd.read_csv(
            V7_GWTC / deployment / "heldout_test_retrieval_metrics_per_seed_v7.csv"
        )
        frame = frame[frame["subset"] == "overall"].copy()
        frame["deployment"] = deployment
        v7_frames.append(frame)
    v7 = pd.concat(v7_frames, ignore_index=True)
    v7_methods = {
        "ET3": {
            "sky": "sky_only",
            "three-channel": "three_channel_strict_positive",
        },
        "gwtc3": {
            "sky": "sky_bayes_factor_only",
            "three-channel": "retrieval_three_channel_strictly_positive",
        },
        "gwtc4": {
            "sky": "sky_bayes_factor_only",
            "three-channel": "retrieval_three_channel_strictly_positive",
        },
    }
    rows = []
    for deployment in ("ET3", "gwtc3", "gwtc4"):
        for canonical in ("sky", "three-channel"):
            old = v7[
                (v7["deployment"] == deployment)
                & (v7["method"] == v7_methods[deployment][canonical])
            ]
            new = v8_summary[
                (v8_summary["deployment"] == deployment)
                & (v8_summary["canonical_method"] == canonical)
            ].iloc[0]
            rows.append(
                {
                    "deployment": deployment,
                    "canonical_method": canonical,
                    "v7_r_at_1_mean": float(old["r_at_1"].mean()),
                    "v7_r_at_1_std": float(old["r_at_1"].std(ddof=1)),
                    "v8_r_at_1_mean": float(new["r_at_1_mean"]),
                    "v8_r_at_1_std": float(new["r_at_1_std"]),
                    "v7_r_at_10_mean": float(old["r_at_10"].mean()),
                    "v7_r_at_10_std": float(old["r_at_10"].std(ddof=1)),
                    "v8_r_at_10_mean": float(new["r_at_10_mean"]),
                    "v8_r_at_10_std": float(new["r_at_10_std"]),
                    "r_at_10_change_v8_minus_v7": float(
                        new["r_at_10_mean"] - old["r_at_10"].mean()
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(root / "v7_v8_retrieval_comparison.csv", index=False)
    return output


def load_pair_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    et_path = root / "et3/et3_pair_level_metrics_per_seed_v8.csv"
    if et_path.exists():
        et = pd.read_csv(et_path)
        et["deployment"] = "ET3"
        frames.append(et)
    for deployment in ("gwtc3", "gwtc4"):
        frame = pd.read_csv(
            root / deployment / "heldout_test_pair_level_metrics_per_seed_v8.csv"
        )
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    keep = []
    for deployment in ("ET3", "gwtc3", "gwtc4"):
        method = METHODS["three-channel"][deployment]
        keep.append(
            raw[
                (raw["deployment"] == deployment)
                & (raw["method"] == method)
            ]
        )
    selected = pd.concat(keep, ignore_index=True)
    metrics = [
        column
        for column in selected.columns
        if column
        not in {
            "deployment",
            "method",
            "unordered_score_rule",
            "seed",
        }
        and pd.api.types.is_numeric_dtype(selected[column])
    ]
    rows = []
    for deployment, part in selected.groupby("deployment", sort=False):
        row = {"deployment": deployment, "n_seeds": part["seed"].nunique()}
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    selected.to_csv(root / "unified_pair_metrics_per_seed_v8.csv", index=False)
    summary.to_csv(root / "unified_pair_metrics_summary_v8.csv", index=False)
    return selected, summary


def aggregate_et_operating_points(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect full-catalog ET operating points without resampling negatives."""
    frames = []
    for path in sorted(
        (root / "et3").glob(
            "seed_*/pair_level_operating_points_*_v8.csv"
        )
    ):
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(path.parent.name.removeprefix("seed_")))
        frame.insert(
            1,
            "method",
            path.stem.removeprefix("pair_level_operating_points_").removesuffix(
                "_v8"
            ),
        )
        frames.append(frame)
    if not frames:
        raise RuntimeError("No ET pair-level operating-point tables found")
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed.to_csv(
        root / "et3_pair_level_operating_points_per_seed_v8.csv", index=False
    )
    metrics = [
        "threshold",
        "false_pairs",
        "true_pairs",
        "fpr",
        "recall",
        "precision",
        "sis_recall",
        "pm_recall",
    ]
    rows = []
    for keys, part in per_seed.groupby(
        ["method", "operating_point", "target"], sort=False
    ):
        row = {
            "method": keys[0],
            "operating_point": keys[1],
            "target": float(keys[2]),
            "n_seeds": int(part["seed"].nunique()),
        }
        for metric in metrics:
            values = part[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_q75"] = float(values.quantile(0.75))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        root / "et3_pair_level_operating_points_summary_v8.csv", index=False
    )
    return per_seed, summary


def load_weights(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    et = pd.read_csv(root / "et3/et3_selected_weights_per_seed_v8.csv")
    et["deployment"] = "ET3"
    frames.append(et)
    for deployment in ("gwtc3", "gwtc4"):
        frame = pd.read_csv(root / deployment / "selected_weights_per_seed_v8.csv")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    rows = []
    for deployment in ("ET3", "gwtc3", "gwtc4"):
        method = METHODS["three-channel"][deployment]
        part = raw[
            (raw["deployment"] == deployment) & (raw["method"] == method)
        ].copy()
        for _, item in part.iterrows():
            rows.append(
                {
                    "deployment": deployment,
                    "seed": int(item["seed"]),
                    "selection_method": method,
                    "waveform_weight": float(item["waveform"]),
                    "time_weight": float(item["time"]),
                    "sky_weight": float(item["sky"]),
                }
            )
    selected = pd.DataFrame(rows)
    summary = (
        selected.groupby("deployment", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            waveform_weight_mean=("waveform_weight", "mean"),
            waveform_weight_std=("waveform_weight", "std"),
            time_weight_mean=("time_weight", "mean"),
            time_weight_std=("time_weight", "std"),
            sky_weight_mean=("sky_weight", "mean"),
            sky_weight_std=("sky_weight", "std"),
        )
    )
    selected.to_csv(root / "unified_selected_weights_per_seed_v8.csv", index=False)
    summary.to_csv(root / "unified_selected_weights_summary_v8.csv", index=False)
    return selected, summary


def sky_calibration_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for deployment, seeds, directory in (
        ("ET3", ET_SEEDS, "et3"),
        ("gwtc3", GWTC_SEEDS, "gwtc3"),
        ("gwtc4", GWTC_SEEDS, "gwtc4"),
    ):
        for seed in seeds:
            payload = json.loads(
                (root / directory / f"seed_{seed}/sky_calibration_v8.json").read_text()
            )
            validation = payload["validation"]
            coefficients = dict(
                zip(payload["feature_columns"], payload["coefficients"])
            )
            rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "n_validation_pairs": validation["n_pairs"],
                    "n_validation_true_pairs": validation["n_true_pairs"],
                    "n_validation_false_pairs": validation["n_false_pairs"],
                    "roc_auc": validation["roc_auc"],
                    "average_precision": validation["average_precision"],
                    "true_score_median": validation["true_score_median"],
                    "null_score_median": validation["null_score_median"],
                    "true_pair_o90_nonzero_retention": validation[
                        "true_pair_o90_nonzero_retention"
                    ],
                    "o90_hard_veto_enabled": payload["o90_hard_veto"]["enabled"],
                    "coefficient_log_bf": coefficients[
                        "sky_log_bayes_factor_raw"
                    ],
                    "coefficient_o90": coefficients["sky_o90"],
                    "coefficient_cross_hpd": coefficients["sky_cross_hpd"],
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("deployment", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            true_score_median_mean=("true_score_median", "mean"),
            null_score_median_mean=("null_score_median", "mean"),
            o90_retention_mean=("true_pair_o90_nonzero_retention", "mean"),
            hard_veto_enabled_seeds=("o90_hard_veto_enabled", "sum"),
        )
    )
    raw.to_csv(root / "sky_calibration_per_seed_v8.csv", index=False)
    summary.to_csv(root / "sky_calibration_summary_v8.csv", index=False)
    return raw, summary


def posterior_map_audit_summary(root: Path) -> pd.DataFrame:
    rows = []
    for seed in ET_SEEDS:
        for split in ("validation", "test"):
            frame = pd.read_parquet(
                root / f"et3/seed_{seed}/{split}_event_sky_diagnostics_v8.parquet"
            )
            rows.append(
                {
                    "deployment": "ET3",
                    "seed": seed,
                    "sample": split,
                    "n_events": len(frame),
                    "area90_q10_deg2": frame["area90_deg2"].quantile(0.1),
                    "area90_median_deg2": frame["area90_deg2"].median(),
                    "area90_q90_deg2": frame["area90_deg2"].quantile(0.9),
                    "true_sky_90_coverage": frame["true_sky_inside_90"].mean(),
                    "posterior_source": (
                        "GWFAST local Fisher width plus eight ET-symmetry modes"
                    ),
                }
            )
    for deployment, seeds in (("gwtc3", GWTC_SEEDS), ("gwtc4", GWTC_SEEDS)):
        for seed in seeds:
            for sample, filename in (
                (
                    "synthetic_validation",
                    "synthetic_validation_event_sky_diagnostics_v8.parquet",
                ),
                (
                    "synthetic_test",
                    "synthetic_test_event_sky_diagnostics_v8.parquet",
                ),
                ("real_catalog", "real_event_sky_diagnostics_v8.parquet"),
            ):
                frame = pd.read_parquet(root / deployment / f"seed_{seed}" / filename)
                rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "sample": sample,
                        "n_events": len(frame),
                        "area90_q10_deg2": frame["area90_deg2"].quantile(0.1),
                        "area90_median_deg2": frame["area90_deg2"].median(),
                        "area90_q90_deg2": frame["area90_deg2"].quantile(0.9),
                        "true_sky_90_coverage": math.nan,
                        "posterior_source": (
                            "public PE HEALPix map"
                            if sample == "real_catalog"
                            else "independent public-PE map template realization"
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    output.to_csv(root / "posterior_map_audit_summary_v8.csv", index=False)
    return output


def merge_pe_audit(root: Path, deployment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    consensus = pd.read_parquet(root / deployment / "real_candidate_consensus_v8.parquet")
    pe = pd.read_parquet(
        PE_ROOT / f"{deployment}_all_strict_pair_pe_diagnostics.parquet"
    )
    consensus["pair_key"] = pair_key(consensus)
    pe["pair_key"] = pair_key(pe)
    pe_columns = [
        "pair_key",
        "pe_available",
        "chirp_mass_median_i",
        "chirp_mass_median_j",
        "chirp_mass_sigma_i_q16_q84",
        "chirp_mass_sigma_j_q16_q84",
        "chirp_mass_standardized_posterior_distance",
        "chirp_mass_wasserstein_distance",
        "chirp_mass_normalized_wasserstein_distance",
        "chirp_mass_bhattacharyya_coefficient",
        "mass_ratio_median_i",
        "mass_ratio_median_j",
        "mass_ratio_sigma_i_q16_q84",
        "mass_ratio_sigma_j_q16_q84",
        "mass_ratio_standardized_posterior_distance",
        "mass_ratio_wasserstein_distance",
        "mass_ratio_normalized_wasserstein_distance",
        "mass_ratio_bhattacharyya_coefficient",
        "chi_eff_median_i",
        "chi_eff_median_j",
        "chi_eff_sigma_i_q16_q84",
        "chi_eff_sigma_j_q16_q84",
        "chi_eff_standardized_posterior_distance",
        "chi_eff_wasserstein_distance",
        "chi_eff_normalized_wasserstein_distance",
        "chi_eff_bhattacharyya_coefficient",
        "max_standardized_posterior_distance",
        "intrinsic_3sigma_consistent",
    ]
    merged = consensus.merge(
        pe[pe_columns],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    merged["deployment"] = deployment
    merged.to_parquet(
        root / deployment / "real_candidate_consensus_with_pe_v8.parquet",
        index=False,
    )
    merged.head(20).to_csv(
        root / deployment / "real_candidate_top20_with_pe_v8.csv",
        index=False,
    )
    rows = []
    overall_rate = float(merged["intrinsic_3sigma_consistent"].mean())
    for top_n in (5, 10, 20, 50, 100):
        part = merged.head(top_n)
        available = part["pe_available"].fillna(False)
        passed = part["intrinsic_3sigma_consistent"].fillna(False)
        n_available = int(available.sum())
        n_passed = int(passed.sum())
        rate = n_passed / n_available if n_available else math.nan
        rows.append(
            {
                "deployment": deployment,
                "top_n": top_n,
                "n_pe_available": n_available,
                "n_intrinsic_3sigma_consistent": n_passed,
                "consistency_fraction": rate,
                "all_pair_consistency_fraction": overall_rate,
                "enrichment_over_all_pairs": (
                    rate / overall_rate if overall_rate > 0 else math.nan
                ),
                "hypergeometric_enrichment_p_one_sided": float(
                    hypergeom.sf(
                        n_passed - 1,
                        len(merged),
                        int(
                            merged["intrinsic_3sigma_consistent"]
                            .fillna(False)
                            .sum()
                        ),
                        n_available,
                    )
                )
                if n_available
                else math.nan,
                "median_max_standardized_distance": float(
                    part["max_standardized_posterior_distance"].median()
                ),
            }
        )
    return merged, pd.DataFrame(rows)


def historical_ranks(root: Path) -> pd.DataFrame:
    checks = {
        "gwtc3": [
            ("GW170104", "GW170814", "historical lensing pair audit"),
            ("GW170809", "GW170814", "catalog coincidence audit"),
        ],
        "gwtc4": [
            (
                "GW230707_124047",
                "GW230708_230935",
                "pre-specified O4a pair audit",
            ),
        ],
    }
    rows = []
    for deployment, pairs in checks.items():
        frame = pd.read_parquet(
            root / deployment / "real_candidate_consensus_with_pe_v8.parquet"
        )
        for event_i, event_j, label in pairs:
            selected = frame[
                ((frame["event_i"] == event_i) & (frame["event_j"] == event_j))
                | ((frame["event_i"] == event_j) & (frame["event_j"] == event_i))
            ]
            row = {
                "deployment": deployment,
                "event_i": event_i,
                "event_j": event_j,
                "audit_label": label,
                "present_in_strict_catalog": not selected.empty,
            }
            if not selected.empty:
                item = selected.iloc[0]
                row.update(
                    {
                        "consensus_rank": int(item["consensus_rank"]),
                        "rank_mean": float(item["rank_mean"]),
                        "rank_min": int(item["rank_min"]),
                        "rank_max": int(item["rank_max"]),
                        "intrinsic_3sigma_consistent": bool(
                            item["intrinsic_3sigma_consistent"]
                        ),
                        "max_standardized_posterior_distance": float(
                            item["max_standardized_posterior_distance"]
                        ),
                    }
                )
            rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(root / "historical_candidate_rank_audit_v8.csv", index=False)
    return output


def real_catalog_scope(root: Path) -> pd.DataFrame:
    rows = []
    for deployment, seed in (("gwtc3", GWTC_SEEDS[0]), ("gwtc4", GWTC_SEEDS[0])):
        summary = json.loads(
            (root / deployment / f"seed_{seed}/seed_summary_v8.json").read_text()
        )
        strict = pd.read_parquet(
            root / deployment / "real_candidate_consensus_v8.parquet",
            columns=["event_i", "event_j"],
        )
        strict_events = pd.unique(
            pd.concat([strict["event_i"], strict["event_j"]], ignore_index=True)
        )
        rows.append(
            {
                "deployment": deployment,
                "all_catalog_events": int(summary["n_real_events"]),
                "all_catalog_unordered_pairs": int(summary["n_real_pairs"]),
                "strict_h1l1_non_ood_bbh_events": int(len(strict_events)),
                "strict_h1l1_non_ood_bbh_unordered_pairs": int(len(strict)),
                "primary_candidate_method_seed_example": summary["primary_method"],
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(root / "real_catalog_scope_v8.csv", index=False)
    return output


def collect_sky_samples(root: Path) -> pd.DataFrame:
    frames = []
    rng = np.random.default_rng(20260724)
    for deployment, seed in (
        ("ET3", ET_SEEDS[0]),
        ("gwtc3", GWTC_SEEDS[0]),
        ("gwtc4", GWTC_SEEDS[0]),
    ):
        if deployment == "ET3":
            frame = pd.read_parquet(
                root / f"et3/seed_{seed}/pair_diagnostics_sample_v8.parquet"
            )
        else:
            frame = pd.read_parquet(
                root
                / f"{deployment}/seed_{seed}/fusion_heldout_test_pairs_v8.parquet"
            )
        label = frame["is_true_pair"].to_numpy(dtype=bool)
        for is_true, name, maximum in ((True, "true", 5000), (False, "false", 20000)):
            indices = np.flatnonzero(label == is_true)
            if len(indices) > maximum:
                indices = rng.choice(indices, size=maximum, replace=False)
            frames.append(
                pd.DataFrame(
                    {
                        "deployment": deployment,
                        "pair_class": name,
                        "sky_score": frame.iloc[indices]["sky_score"].to_numpy(
                            dtype=np.float64
                        ),
                    }
                )
            )
    output = pd.concat(frames, ignore_index=True)
    output.to_parquet(root / "sky_score_distribution_sample_v8.parquet", index=False)
    return output


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def plot_main(
    root: Path,
    retrieval: pd.DataFrame,
    sky_samples: pd.DataFrame,
    pe_tables: dict[str, pd.DataFrame],
) -> None:
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    deployment_order = ["ET3", "gwtc3", "gwtc4"]
    method_order = list(METHODS)
    x = np.arange(len(deployment_order))
    width = 0.15
    jitter_rng = np.random.default_rng(711)
    for axis, metric, title, label in (
        (axes[0, 0], "r_at_1", "First-rank companion retrieval", "a"),
        (axes[0, 1], "r_at_10", "Top-ten companion retrieval", "b"),
    ):
        for method_index, method in enumerate(method_order):
            position = x + (method_index - 2) * width
            means = []
            standard_deviations = []
            for deployment in deployment_order:
                values = retrieval[
                    (retrieval["deployment"] == deployment)
                    & (retrieval["canonical_method"] == method)
                ][metric].to_numpy(dtype=np.float64)
                means.append(values.mean())
                standard_deviations.append(values.std(ddof=1))
                axis.scatter(
                    np.full(len(values), position[deployment_order.index(deployment)])
                    + jitter_rng.normal(0, 0.012, len(values)),
                    values,
                    s=10,
                    facecolor="white",
                    edgecolor=PALETTE[method],
                    linewidth=0.7,
                    zorder=4,
                )
            axis.bar(
                position,
                means,
                width=width * 0.9,
                color=PALETTE[method],
                alpha=0.82,
                label=method,
                zorder=2,
            )
            axis.errorbar(
                position,
                means,
                yerr=standard_deviations,
                fmt="none",
                color="#222222",
                capsize=2,
                linewidth=0.7,
                zorder=3,
            )
        axis.set_xticks(x, [DEPLOYMENT_LABELS[item] for item in deployment_order])
        axis.set_ylabel(title.split()[0] + " recall")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        axis.set_title(title)
        panel_label(axis, label)
    axes[0, 1].legend(
        loc="upper center",
        bbox_to_anchor=(-0.08, 1.25),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    axis = axes[1, 0]
    offsets = {"true": -0.18, "false": 0.18}
    colors = {"true": "#B4433C", "false": "#777777"}
    for deployment_index, deployment in enumerate(deployment_order):
        for pair_class in ("false", "true"):
            values = sky_samples[
                (sky_samples["deployment"] == deployment)
                & (sky_samples["pair_class"] == pair_class)
            ]["sky_score"].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            finite = np.clip(finite, np.quantile(finite, 0.005), np.quantile(finite, 0.995))
            if np.ptp(finite) < 1e-8:
                finite = finite + np.linspace(-1e-6, 1e-6, len(finite))
            violin = axis.violinplot(
                finite,
                positions=[deployment_index + offsets[pair_class]],
                widths=0.32,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[pair_class])
                body.set_edgecolor(colors[pair_class])
                body.set_alpha(0.55)
            violin["cmedians"].set_color("#111111")
            violin["cmedians"].set_linewidth(0.8)
    axis.scatter([], [], color=colors["false"], alpha=0.55, label="false pair")
    axis.scatter([], [], color=colors["true"], alpha=0.55, label="true companion")
    axis.set_xticks(x, [DEPLOYMENT_LABELS[item] for item in deployment_order])
    axis.set_ylabel("Validation-calibrated sky score")
    axis.set_title("Held-out sky-channel distributions")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "c")

    axis = axes[1, 1]
    top_values = np.arange(1, 101)
    for deployment, color in (("gwtc3", "#2C6E9B"), ("gwtc4", "#B4433C")):
        table = pe_tables[deployment].head(100)
        cumulative = (
            table["intrinsic_3sigma_consistent"].fillna(False).astype(int).cumsum()
            / np.arange(1, len(table) + 1)
        )
        all_rate = pe_tables[deployment]["intrinsic_3sigma_consistent"].mean()
        axis.plot(
            top_values[: len(cumulative)],
            cumulative,
            color=color,
            linewidth=1.6,
            label=DEPLOYMENT_LABELS[deployment],
        )
        axis.axhline(all_rate, color=color, linewidth=0.7, linestyle=":", alpha=0.8)
    axis.axhline(1.0, color="#BBBBBB", linewidth=0.6)
    axis.set_xlim(1, 100)
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Consensus top-N pairs")
    axis.set_ylabel("Cumulative PE 3σ-consistency fraction")
    axis.set_title("Post-retrieval intrinsic-PE audit")
    axis.grid(color="#D8D8D8", linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, loc="lower right")
    panel_label(axis, "d")

    for suffix in ("pdf", "png"):
        figure.savefig(
            root / f"figures/fig_unified_sky_v8_main.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_candidate_diagnostics(
    root: Path,
    pe_tables: dict[str, pd.DataFrame],
) -> None:
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    for row, deployment in enumerate(("gwtc3", "gwtc4")):
        table = pe_tables[deployment].head(10).copy()
        labels = [
            f"{left.replace('GW', '')}\n{right.replace('GW', '')}"
            for left, right in zip(table["event_i"], table["event_j"])
        ]
        axis = axes[row, 0]
        contributions = table[
            ["waveform_score_mean", "time_score_mean", "sky_score_mean"]
        ].to_numpy(dtype=np.float64)
        bottom = np.zeros(len(table))
        for index, (name, color) in enumerate(
            (
                ("waveform", PALETTE["waveform"]),
                ("time", PALETTE["time"]),
                ("sky", PALETTE["sky"]),
            )
        ):
            values = contributions[:, index]
            positive = np.maximum(values, 0)
            axis.bar(
                np.arange(len(table)),
                positive,
                bottom=bottom,
                color=color,
                width=0.78,
                label=name if row == 0 else None,
            )
            bottom += positive
            negative = np.minimum(values, 0)
            axis.bar(
                np.arange(len(table)),
                negative,
                color=color,
                width=0.78,
                alpha=0.45,
            )
        axis.axhline(0, color="#333333", linewidth=0.7)
        axis.set_xticks(np.arange(len(table)), labels, rotation=90)
        axis.set_ylabel("Mean channel evidence")
        axis.set_title(f"{DEPLOYMENT_LABELS[deployment]} consensus top 10")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
        panel_label(axis, "a" if row == 0 else "c")

        axis = axes[row, 1]
        distances = table[
            [
                "chirp_mass_standardized_posterior_distance",
                "mass_ratio_standardized_posterior_distance",
                "chi_eff_standardized_posterior_distance",
            ]
        ].to_numpy(dtype=np.float64).T
        image = axis.imshow(
            np.clip(distances, 0, 6),
            aspect="auto",
            cmap="magma_r",
            vmin=0,
            vmax=6,
        )
        axis.set_yticks(
            [0, 1, 2],
            [r"$D_{\mathcal{M}_c}$", r"$D_q$", r"$D_{\chi_{\rm eff}}$"],
        )
        axis.set_xticks(np.arange(len(table)), np.arange(1, len(table) + 1))
        axis.set_xlabel("Consensus rank")
        axis.set_title("Independent PE consistency (clipped at 6σ)")
        for y in range(3):
            for x_index in range(len(table)):
                value = distances[y, x_index]
                text = ">6" if value > 6 else f"{value:.1f}"
                axis.text(
                    x_index,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if min(value, 6) > 3.0 else "black",
                )
        panel_label(axis, "b" if row == 0 else "d")
    axes[0, 0].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(1.08, 1.30),
    )
    colorbar = figure.colorbar(image, ax=axes[:, 1], shrink=0.85, pad=0.02)
    colorbar.set_label("Standardized posterior distance")
    for suffix in ("pdf", "png"):
        figure.savefig(
            root / f"figures/fig_unified_sky_v8_real_candidate_audit.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_revision_diagnostics(
    root: Path,
    comparison: pd.DataFrame,
) -> None:
    setup_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), constrained_layout=True)

    axis = axes[0]
    rows = comparison.copy()
    rows["label"] = (
        rows["deployment"].map(DEPLOYMENT_LABELS)
        + "\n"
        + rows["canonical_method"].str.replace("three-channel", "3-channel")
    )
    positions = np.arange(len(rows))
    axis.bar(
        positions - 0.18,
        rows["v7_r_at_10_mean"],
        width=0.35,
        color="#A7A7A7",
        label="v7",
    )
    axis.bar(
        positions + 0.18,
        rows["v8_r_at_10_mean"],
        width=0.35,
        color="#2C6E9B",
        label="v8",
    )
    axis.errorbar(
        positions - 0.18,
        rows["v7_r_at_10_mean"],
        yerr=rows["v7_r_at_10_std"],
        fmt="none",
        color="#222222",
        capsize=2,
        linewidth=0.7,
    )
    axis.errorbar(
        positions + 0.18,
        rows["v8_r_at_10_mean"],
        yerr=rows["v8_r_at_10_std"],
        fmt="none",
        color="#222222",
        capsize=2,
        linewidth=0.7,
    )
    axis.set_xticks(positions, rows["label"], rotation=45, ha="right")
    axis.set_ylabel("R@10")
    axis.set_ylim(0, 1.03)
    axis.set_title("Effect of revised sky model")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "a")

    axis = axes[1]
    area_sources = {
        "ET v7 Gaussian": pd.read_parquet(
            V7_ET / f"seed_{ET_SEEDS[0]}/test_observed_sky_events.parquet"
        )["sky_area90_deg2"].to_numpy(dtype=np.float64),
        "ET v8 multimodal": pd.read_parquet(
            root / f"et3/seed_{ET_SEEDS[0]}/test_event_sky_diagnostics_v8.parquet"
        )["area90_deg2"].to_numpy(dtype=np.float64),
        "GWTC-3 PE": pd.read_parquet(
            root
            / f"gwtc3/seed_{GWTC_SEEDS[0]}/real_event_sky_diagnostics_v8.parquet"
        )["area90_deg2"].to_numpy(dtype=np.float64),
        "O4a PE": pd.read_parquet(
            root
            / f"gwtc4/seed_{GWTC_SEEDS[0]}/real_event_sky_diagnostics_v8.parquet"
        )["area90_deg2"].to_numpy(dtype=np.float64),
    }
    colors = ["#999999", "#B4433C", "#2C6E9B", "#47956F"]
    for (name, values), color in zip(area_sources.items(), colors):
        values = np.sort(values[np.isfinite(values) & (values > 0)])
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axis.plot(values, cumulative, color=color, linewidth=1.4, label=name)
    axis.set_xscale("log")
    axis.set_xlabel(r"90% sky area (deg$^2$)")
    axis.set_ylabel("Event CDF")
    axis.set_ylim(0, 1.02)
    axis.set_title("Posterior-area audit")
    axis.grid(color="#D8D8D8", linewidth=0.5)
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    panel_label(axis, "b")

    axis = axes[2]
    calibration_rows = []
    for deployment, seeds in (
        ("ET3", ET_SEEDS),
        ("gwtc3", GWTC_SEEDS),
        ("gwtc4", GWTC_SEEDS),
    ):
        for seed in seeds:
            path = (
                root
                / ("et3" if deployment == "ET3" else deployment)
                / f"seed_{seed}/sky_calibration_v8.json"
            )
            payload = json.loads(path.read_text())
            calibration_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "auc": payload["validation"]["roc_auc"],
                    "retention": payload["validation"][
                        "true_pair_o90_nonzero_retention"
                    ],
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    for index, deployment in enumerate(("ET3", "gwtc3", "gwtc4")):
        values = calibration[calibration["deployment"] == deployment]["auc"]
        axis.bar(
            index,
            values.mean(),
            color=["#B4433C", "#2C6E9B", "#47956F"][index],
            alpha=0.75,
            width=0.62,
        )
        axis.scatter(
            np.full(len(values), index),
            values,
            facecolor="white",
            edgecolor="#222222",
            s=18,
            linewidth=0.7,
            zorder=3,
        )
    axis.axhline(0.5, color="#666666", linestyle=":", linewidth=0.8)
    axis.set_xticks(
        range(3),
        [DEPLOYMENT_LABELS[item] for item in ("ET3", "gwtc3", "gwtc4")],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Validation sky ROC AUC")
    axis.set_ylim(0.45, 1.01)
    axis.set_title("Sky discrimination without test tuning")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    panel_label(axis, "c")

    for suffix in ("pdf", "png"):
        figure.savefig(
            root / f"figures/fig_unified_sky_v8_revision_diagnostics.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(figure)


def format_result_table(summary: pd.DataFrame, metric: str) -> str:
    lines = [
        "| Deployment | Waveform | Time | Sky | Time+sky | Three-channel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for deployment in ("ET3", "gwtc3", "gwtc4"):
        values = {}
        for method in METHODS:
            row = summary[
                (summary["deployment"] == deployment)
                & (summary["canonical_method"] == method)
            ].iloc[0]
            values[method] = (
                f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            )
        lines.append(
            f"| {DEPLOYMENT_LABELS[deployment]} | {values['waveform']} | "
            f"{values['time']} | {values['sky']} | {values['time+sky']} | "
            f"{values['three-channel']} |"
        )
    return "\n".join(lines)


def format_top_candidates(frame: pd.DataFrame, top_n: int = 10) -> str:
    lines = [
        "| Rank | Event i | Event j | Mean rank | Waveform | Time | Sky | "
        r"$D_{\max}$ | PE 3σ |",
        "|---:|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, row in frame.head(top_n).iterrows():
        lines.append(
            f"| {int(row['consensus_rank'])} | {row['event_i']} | {row['event_j']} | "
            f"{row['rank_mean']:.1f} | {row['waveform_score_mean']:.3f} | "
            f"{row['time_score_mean']:.3f} | {row['sky_score_mean']:.3f} | "
            f"{row['max_standardized_posterior_distance']:.2f} | "
            f"{'yes' if row['intrinsic_3sigma_consistent'] else 'no'} |"
        )
    return "\n".join(lines)


def format_version_comparison(frame: pd.DataFrame) -> str:
    lines = [
        "| Deployment | Method | v7 R@10 | v8 R@10 | Change |",
        "|---|---|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {DEPLOYMENT_LABELS[row.deployment]} | {row.canonical_method} | "
            f"{row.v7_r_at_10_mean:.3f} ± {row.v7_r_at_10_std:.3f} | "
            f"{row.v8_r_at_10_mean:.3f} ± {row.v8_r_at_10_std:.3f} | "
            f"{row.r_at_10_change_v8_minus_v7:+.3f} |"
        )
    return "\n".join(lines)


def make_reports(
    root: Path,
    retrieval_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    weight_summary: pd.DataFrame,
    sky_summary: pd.DataFrame,
    version_comparison: pd.DataFrame,
    catalog_scope: pd.DataFrame,
    pe_tables: dict[str, pd.DataFrame],
    pe_summary: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    r1_table = format_result_table(retrieval_summary, "r_at_1")
    r10_table = format_result_table(retrieval_summary, "r_at_10")
    comparison_table = format_version_comparison(version_comparison)
    scope_lines = [
        "| Deployment | All events | All pairs | Strict H1L1 non-OOD BBH events | Strict pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in catalog_scope.itertuples():
        scope_lines.append(
            f"| {DEPLOYMENT_LABELS[row.deployment]} | {row.all_catalog_events} | "
            f"{row.all_catalog_unordered_pairs:,} | "
            f"{row.strict_h1l1_non_ood_bbh_events} | "
            f"{row.strict_h1l1_non_ood_bbh_unordered_pairs:,} |"
        )
    scope_table = "\n".join(scope_lines)
    sky_lines = [
        "| Deployment | Validation sky ROC AUC | Validation sky AP | O90 retention |",
        "|---|---:|---:|---:|",
    ]
    for row in sky_summary.itertuples():
        sky_lines.append(
            f"| {DEPLOYMENT_LABELS[row.deployment]} | "
            f"{row.roc_auc_mean:.3f} ± {row.roc_auc_std:.3f} | "
            f"{row.average_precision_mean:.4f} ± {row.average_precision_std:.4f} | "
            f"{row.o90_retention_mean:.3f} |"
        )
    sky_table = "\n".join(sky_lines)
    pe_text = "\n".join(
        f"- {DEPLOYMENT_LABELS[row.deployment]} top-{int(row.top_n)}: "
        f"{int(row.n_intrinsic_3sigma_consistent)}/{int(row.n_pe_available)} "
        f"通过 3σ screen；全严格 pair 基线为 "
        f"{row.all_pair_consistency_fraction:.3f}；富集倍数 "
        f"{row.enrichment_over_all_pairs:.2f}；单侧超几何 p="
        f"{row.hypergeometric_enrichment_p_one_sided:.3g}。"
        for row in pe_summary[pe_summary["top_n"].isin([5, 10])].itertuples()
    )
    operating = pd.read_csv(
        root / "et3_pair_level_operating_points_summary_v8.csv"
    )
    operating = operating[
        operating["method"] == "three_channel_strict_positive"
    ].copy()
    operating_lines = [
        "| Operating point | False pairs | True pairs | Precision | Recall | SIS recall | PM recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    selected_operating = pd.concat(
        [
            operating[
                (operating["operating_point"] == "fpr")
                & operating["target"].isin([1e-4, 1e-5])
            ],
            operating[
                (operating["operating_point"] == "recall")
                & operating["target"].isin([0.5, 0.9])
            ],
        ],
        ignore_index=True,
    )
    for row in selected_operating.itertuples():
        label = (
            f"FPR={row.target:.0e}"
            if row.operating_point == "fpr"
            else f"recall={row.target:.1f}"
        )
        operating_lines.append(
            f"| {label} | {row.false_pairs_mean:,.1f} ± "
            f"{row.false_pairs_std:,.1f} | {row.true_pairs_mean:,.1f} ± "
            f"{row.true_pairs_std:,.1f} | {row.precision_mean:.3f} ± "
            f"{row.precision_std:.3f} | {row.recall_mean:.3f} ± "
            f"{row.recall_std:.3f} | {row.sis_recall_mean:.3f} ± "
            f"{row.sis_recall_std:.3f} | {row.pm_recall_mean:.3f} ± "
            f"{row.pm_recall_std:.3f} |"
        )
    operating_table = "\n".join(operating_lines)
    fisher_audit = json.loads(
        (root / "shared/et3_gwfast_fisher_localization_audit.json").read_text()
    )
    finite = fisher_audit["finite_event_count"]
    total = fisher_audit["event_count"]
    reconstructed = fisher_audit["mass_label_reconstruction"][
        "reconstructed_event_count"
    ]
    resolution_audit = json.loads(
        (
            root / "shared/et3_fisher_frequency_resolution_audit.json"
        ).read_text()
    )
    cn = """# 统一天空后验检索实验 v8：完整方法、结果与审计

## 1. 研究目的与结论边界

本实验把 ET-3、GWTC-3.0/O3 和 GWTC-4.1/O4a 放入同一个
`waveform + time-delay + sky-localization` 方法框架。这里的“统一”是指：

1. 三个部署都把每个事件的天空信息表示为归一化 HEALPix 后验；
2. 三个部署都使用相同的 pair-level 原始统计量和同一段实现；
3. 每个部署只在自己的 synthetic validation systems 上冻结天空校准和融合权重；
4. test labels、真实 GWTC 排名和 PE follow-up 均不参与调参。

ET 与 GWTC 的 posterior **来源不能强行相同**：GWTC 使用公开 PE sky map，
ET 尚无真实观测，因此使用探测器响应推导的后验代理。二者从 posterior
开始的计算完全相同。本结果是 catalog-level companion retrieval 和
candidate shortlist，不是透镜探测，也不提供 LVK 意义上的 FAR。

## 2. v7 中需要修复的问题

v7 的 GWTC sky 通道已使用 common-source Bayes factor，但 ET 使用单峰、
偏窄的 Gaussian 定位代理。后者没有体现单台三角形 ET 在短信号下的多峰
天空简并，导致 ET true/false sky score 近乎完美分离，不能与真实 GWTC
做可信比较。v8 不提高 GWTC 分数来追求好看指标，而是把 ET 模拟改得更
保守，并让两个域共享统计量、校准边界和代码实现。

__VERSION_COMPARISON__

表中的下降尤其是 ET sky-only 的预期修正：它反映取消过强单峰定位假设，
不能解释成 waveform encoder 退化。GWTC 变化主要来自加入 O90/cross-HPD
和 validation-frozen 单调校准。

## 3. 固定数据、划分与复用边界

- ET-3：复用 v7 五个正式训练 seed 的 source-system split、waveform
  embeddings、intrinsic-head predictions、waveform evidence calibration
  和冻结 time-delay lookup。
- GWTC-3/O3、GWTC-4.1/O4a：复用 v7 各三个正式 seed 的 run-matched
  H1/L1 encoder、严格 2 s/4096 点输入、waveform evidence、run-exposure
  conditioned time evidence、synthetic validation/test catalogs 和真实
  GWTC embeddings。
- 一个透镜系统的两幅像始终处于同一个 split。encoder 只用 train；
  waveform/sky calibration 与 fusion weights 只用 validation；test 只评价。
- v8 **没有重训 waveform encoder，也没有改变 time score**；只重建天空
  posterior、天空 pair 特征、天空校准、融合权重和最终排名。这样能把变化
  明确归因于统一天空方案。

真实目录范围：

__CATALOG_SCOPE__

## 4. Waveform 通道

每个事件先由 domain-matched encoder 得到 L2 归一化 embedding
`z_i`，并预测 detector-frame `log Mc` 与 `logit(q)`。pair-local 原始量为：

```text
cosine_ij = z_i dot z_j
dMc_ij    = -abs(logMc_hat_i - logMc_hat_j)
dq_ij     = -abs(logitq_hat_i - logitq_hat_j)
```

这些量只读取事件 i 和 j。随后使用 validation true/false pairs 冻结全局
中心、尺度、组合系数及 log-likelihood-ratio 标定，得到

\\[
s_{{\\rm w},ij}=\\log\\frac{{p(u_{{ij}}\\mid L,\\mathrm{{val}})}}
{{p(u_{{ij}}\\mid N,\\mathrm{{val}})}}.
\\]

真实 PE 质量参数不进入 waveform score。它们只在最终排序后做独立审计。

## 5. Time-delay 通道

pair-local 时间差为

\\[
\\Delta t_{{ij}}=|t_i-t_j|/86400\\quad\\mathrm{{days}}.
\\]

冻结 time evidence 为

\\[
s_{{\\rm t},ij}=\\log\\frac{{p(\\Delta t_{{ij}}\\mid L,E)}}
{{p(\\Delta t_{{ij}}\\mid N,E)}} ,
\\]

其中透镜时延 population 来自 GW-LMC，`E` 是对应部署的冻结观测日历或
exposure；null 由同一 exposure 下的随机非伴随事件构成。原始时间差只用
两个事件，但从时间差到 evidence 的 lookup 使用外部透镜 population 和
validation 前冻结的观测模型。真实目录不重新拟合 lookup。

## 6. ET 后验如何生成

### 6.1 局部宽度

对五个 seed 的 validation/test 事件取并集，共 __TOTAL_EVENTS__ 个独立
event keys。用 GWFAST 1.1.2、IMRPhenomD、ET-D ASD、5 Hz 下限、地球自转
和一个三角形 ET 站点计算 Fisher matrix。然后按目录已有 network SNR
缩放 Fisher 信息。__FINITE_EVENTS__/__TOTAL_EVENTS__ 个事件通过有限性审计；病态矩阵
被显式记录，不会静默产生极窄 posterior。

v7 的 event table 对 unlensed 事件有意不保存监督用的 `target_logmc` 和
`target_logitq`。Fisher 计算不能把这些空值当物理参数，因此 v8 从冻结
`source_samples.csv` 的 source-frame masses 和 luminosity distance 出发，
使用与原生成器相同的 Planck15 关系重建 detector-frame 质量，共
__RECONSTRUCTED_EVENTS__ 个 event keys。对所有已有 lensed 标签的交叉检查
达到 `max |delta logMc| < 2.5e-7`、`max |delta logitq| < 4e-7` 后才用于正式
重算。

数值积分分辨率另用固定 SNR-quantile 样本比较 `res=100` 与 `res=400`：
log-area Spearman 为 __FISHER_RES_SPEARMAN__，面积 log10 比值的中位绝对值为
__FISHER_RES_MEDIAN_ERROR__，全天空分类一致率为 __FISHER_RES_FULL_AGREEMENT__。
按预定收敛规则判定为 **__FISHER_RES_STATUS__**。

Fisher 只提供一个局部模式的 90% 面积。Santoliquido et al. 指出单台
三角形 ET 的 Fisher 近似会漏掉全局多峰结构，因此这里不把 Fisher ellipse
直接当完整 posterior。

### 6.2 八重天空简并

把主模式转换到 ET 站点的水平坐标 `(altitude, azimuth)`，加入短持续信号
下三角形 ET 响应的八个对称模式：高度镜像以及方位的 π、±π/2 旋转，再
转回 RA/Dec。八个模式等权，每个模式用球面 von Mises-Fisher 核表示。
观测主中心从以真实天空为中心、Fisher 宽度决定的分布中随机抽取；ranking
代码从不直接读取 true RA/Dec。

因此 ET posterior 的正式名称是：
`GWFAST-local-width, ET-symmetry multimodal posterior surrogate`。
它比旧单峰 Gaussian 合理，但仍不是 full Bayesian/NPE posterior，结果必须
按敏感性实验解释。

## 7. GWTC 后验如何获得

- synthetic real-noise injection：从对应 run 的真实公开 PE sky maps 中按
  SNR/质量条件抽取模板；透镜双像共享真实天空，但每幅像使用独立模板和
  独立观测噪声 realization，再旋转到共同真实方向。
- real catalog deployment：直接读取该真实事件公开 PE release 的 HEALPix
  posterior。
- 所有图降到 common `nside=32`，使用 RING ordering，非负化并归一化，
  保证 `sum_k P_i[k]=1`。

这种做法保留真实后验的宽度、多峰和不规则形状，不把真实位置强行放到
posterior 峰值。

## 8. 统一 pair-level sky 统计

### 8.1 Common-source sky Bayes factor

对各向同性先验 `pi(Omega)=1/(4 pi)`：

\\[
B_{{\\rm sky},ij}
=\\int\\frac{{p_i(\\Omega)p_j(\\Omega)}}{{\\pi(\\Omega)}}d\\Omega
=N_{{\\rm pix}}\\sum_k P_{{ik}}P_{{jk}},
\\qquad
f_1=\\log B_{{\\rm sky},ij}.
\\]

`B=1` 表示不比均匀天空更有信息；大于 1 支持共同方向；小于 1 反对。
这保留绝对定位面积信息，避免两个近均匀宽图因形状相似而得到高 cosine。

### 8.2 90% credible-region overlap

令 `C_i^90` 为事件 i 的 90% HPD 像素集合：

\\[
f_2=O_{{90}}=
\\frac{{|C_i^{{90}}\\cap C_j^{{90}}|}}
{{\\min(|C_i^{{90}}|,|C_j^{{90}}|)}}.
\\]

若 validation true pairs 中 `O90>0` 的保留率至少 0.99，则启用预先规定的
`O90=0` hard veto。该过滤形式与 LVK O3/O4a lensing search 的非零 90%
天空重叠筛选一致；是否启用只由 validation 决定。

### 8.3 Symmetric cross-HPD

设 `H_q(Omega_p^MAP)` 是在 q 的 posterior 中、概率密度不低于 q 在
p 的 MAP 位置处密度的累计后验质量。使用 Wong et al. 的对称量：

\\[
f_3=\\max[1-H_q(\\Omega_p^{{MAP}}),\\,
          1-H_p(\\Omega_q^{{MAP}})].
\\]

三个特征都定义为数值越大、共同天空相容性越强。

## 9. 天空证据校准

原始 `log B_sky` 是物理 Bayes factor；`O90` 和 cross-HPD 是互补过滤统计。
为避免凭经验手写它们的权重，每个 deployment/seed 在 synthetic validation
pairs 上拟合单调 logistic：

\\[
s_{{\\rm sky},ij}
=b+\\beta_1 z(f_1)+\\beta_2 z(f_2)+\\beta_3 z(f_3),
\\qquad \\beta_m\\ge0.
\\]

- true 和 false 类总权重各为 0.5，避免极端 class imbalance 淹没真对；
- `z` 的中心和尺度只来自 validation；
- L2 penalty 为 1；
- 非负系数防止有限样本学出“重叠越大反而越不相容”的非物理符号；
- 拟合输出是 validation-calibrated density-ratio surrogate，不冒充精确
  Bayes factor；
- test labels 和真实候选排名完全不参与。

天空矩阵与 waveform/time 一样，是每对事件一个对称分数矩阵。它不是
额外后处理标签。

__SKY_CALIBRATION_TABLE__

ROC AUC 接近 0.5 表示相应 posterior 本身几乎没有 pair 判别信息；校准器
不会利用 test catalog 把这种弱信息人为拉大。

## 10. 三通道融合与权重

最终：

\\[
S_{{ij}}=w_{{\\rm w}}s_{{\\rm w},ij}
        +w_{{\\rm t}}s_{{\\rm t},ij}
        +w_{{\\rm s}}s_{{\\rm sky},ij}.
\\]

三个通道已由 validation 冻结为 evidence-like score，因此不再在 test 或
真实目录内按行 z-score。每个 seed 分别在 validation 上搜索：

- waveform-only、time-only、sky-only；
- validation-selected time+sky baseline；
- unconstrained three-channel（允许诊断性零权重）；
- strict-positive three-channel（三个权重均大于 0）；
- equal-evidence `(1,1,1)` sensitivity。

正文同框结果使用 strict-positive、retrieval-objective 权重；candidate-
objective 权重用于真实 shortlist，并单独保存。真实 PE 结果不能反向调权。

## 11. 检索与 pair-level 评价

对每个 lensed-image query，排除自匹配后按 `S_ij` 排候选；伴随像位于前
K 即计入 R@K。同一系统的两个 directed queries 一起进入 system-level
bootstrap。无序 pair-level PR 使用

\\[
S_{{\\{{i,j\\}}}}=\\max(S_{{ij}},S_{{ji}}).
\\]

v8 三个证据矩阵对称，max 与 mean 仅有浮点差异；仍保留 max 以匹配论文
既定口径。完整 pair metrics 包含 AUPRC、50%/90% recall 的 false pair
数量及 false-pair-rate `1e-5` operating point。

ET-3 的 full-pair 统计使用全部 40,495,500 个无序 pair（3,000 true，
40,492,500 false），没有抽样负例。五个训练 seed 的关键 operating points：

__ET_OPERATING_POINTS__

## 12. Companion retrieval 结果

### R@1

__R1_TABLE__

### R@10

__R10_TABLE__

这些数值是训练 seed 间 mean ± sample SD。ET 五个 seed、GWTC 每个部署
三个 seed 均另存 system-level 10,000 次 bootstrap CI；同一透镜系统的
两个 directed queries 成组抽样，overall 指标按 SIS/PM 分层。bootstrap
量化固定模型与有限 test systems 的抽样波动，seed 间 SD 则量化独立训练
运行差异，二者不能混为一谈。结果应按部署难度解读，不能把 ET posterior
surrogate 结果当真实 ET 性能预测。

## 13. 真实 GWTC shortlist 与 PE 独立审计

真实目录主 shortlist 使用 strict H1+L1、非 OOD BBH subset。每个 seed
使用其 validation-selected candidate three-channel 权重；consensus rank
按三个 seed 的平均 rank 排序。PE 不参与检索，只用于检验 shortlist 是否
富集具有相容内禀参数的 pair。

采用标准化后验距离：

\\[
D_x=
\\frac{{|\\mathrm{{median}}(x_i)-\\mathrm{{median}}(x_j)|}}
{{\\sqrt{{\\sigma_i^2+\\sigma_j^2}}}},
\\quad x\\in(\\mathcal M_{{c,det}},q,\\chi_{{eff}}).
\\]

预定 screen 为三项均 `D_x<=3`。距离、Wasserstein 和 Bhattacharyya 系数
均保存；luminosity distance 不进入 screen，因为透镜放大率会改变表观距离。

__PE_TEXT__

这表示前列 pair 对 PE 一致性有富集，但 **不等于 lensing detection**。
超几何 p 只检验“top-N 是否富集通过 screen 的 pair”，不是单个候选的
lensing p-value，也没有计入搜索 trials 或完整 astrophysical background。
通过 screen 的 pair 仍需 joint Bayesian lensing/unlensed model comparison。
未通过的 pair 也不得用于反向调整三通道模型。

### GWTC-3/O3 consensus top 10

__GWTC3_TOP__

### GWTC-4.1/O4a consensus top 10

__GWTC4_TOP__

## 14. 信息边界

| 计算 | pair-local 信息 | validation/population 信息 | test/real catalog 信息 |
|---|---|---|---|
| waveform raw cosine/预测质量差 | 事件 i,j 的 waveform 输出 | 无 | 无 |
| waveform evidence calibration | pair-local raw features | validation true/false pairs | 无 |
| raw time difference | 事件 i,j 的 GPS time | 无 | 无 |
| time likelihood ratio | pair-local delay | GW-LMC + frozen exposure/null | 无 |
| raw sky BF/O90/cross-HPD | 事件 i,j 的 posterior maps | uniform sky prior | 无 |
| calibrated sky score | pair-local sky features | synthetic validation pairs | 无 |
| fusion weights | 三个 frozen scores | validation systems | 无 |
| R@K/AUPRC | 已冻结 score | 无 | 只用于评价 |
| PE consistency | 排名后的事件对 | 无 | 只用于独立审计 |

## 15. 已知限制

1. ET v8 posterior 是 response-derived surrogate，不是逐事件 full Bayesian PE。
2. GWTC sky-only 判别力仍弱；这是宽真实 posterior 的结果，不应通过目录内
   标准化人为放大。
3. GWTC 的注入性能不等于真实目录 precision；真实目录没有已知 lens labels。
4. PE 3σ screen 是 follow-up audit，不是 Bayes factor 或 detection statistic。
5. candidate rank 不是 p-value；本实验未替代 LVK 完整 background/FAR。
6. ET 与 H1/L1 使用 domain-matched waveform encoder；“统一框架”不等于
   共享 checkpoint。

## 16. 复现

```bash
cd /root/autodl-tmp/gw-catalog
bash results/unified_sky_v8_20260724/reproduce_unified_sky_v8.sh
```

逐 seed 数据、weight grids、calibration JSON、pair diagnostics、图和日志均
保存在本目录。交付包不包含原始 strain、完整 PE 文件和大型 posterior map
数组，但包含复现脚本与其路径清单。

## 17. 参考依据

1. Haris et al., *Identifying strongly lensed gravitational wave signals
   from binary black hole mergers*, arXiv:1807.07062.
2. Wong et al., *Using overlap of sky localization probability maps for
   filtering potentially lensed pairs of gravitational-wave signals*,
   arXiv:2112.05932.
3. Iacovelli et al., *GWFAST: A Fisher Information Matrix Python Code for
   Third-generation Gravitational-wave Detectors*, arXiv:2207.06910.
4. Santoliquido et al., *Fast and accurate parameter estimation of
   high-redshift sources with the Einstein Telescope*, arXiv:2504.21087.
5. LVK, *Search for gravitational-lensing signatures in the full third
   observing run*, arXiv:2304.08393.
6. LVK, *GWTC-4.0: Searches for Gravitational Wave Lensing Signatures*,
   LIGO-P2500419-v10.
7. Garrón & Keitel, *Waveform systematics in identifying strongly
   gravitationally lensed gravitational waves: Posterior overlap method*,
   arXiv:2306.12908.
"""
    cn = cn.replace("{{", "{").replace("}}", "}")
    cn = (
        cn.replace("__TOTAL_EVENTS__", f"{total:,}")
        .replace("__FINITE_EVENTS__", f"{finite:,}")
        .replace("__RECONSTRUCTED_EVENTS__", f"{reconstructed:,}")
        .replace(
            "__FISHER_RES_SPEARMAN__",
            f"{resolution_audit['spearman_log_area']:.4f}",
        )
        .replace(
            "__FISHER_RES_MEDIAN_ERROR__",
            f"{resolution_audit['median_abs_log10_area_ratio']:.4f}",
        )
        .replace(
            "__FISHER_RES_FULL_AGREEMENT__",
            f"{resolution_audit['full_sky_classification_agreement']:.3f}",
        )
        .replace(
            "__FISHER_RES_STATUS__",
            "通过" if resolution_audit["accepted"] else "未通过",
        )
        .replace("__R1_TABLE__", r1_table)
        .replace("__R10_TABLE__", r10_table)
        .replace("__VERSION_COMPARISON__", comparison_table)
        .replace("__CATALOG_SCOPE__", scope_table)
        .replace("__SKY_CALIBRATION_TABLE__", sky_table)
        .replace("__ET_OPERATING_POINTS__", operating_table)
        .replace("__PE_TEXT__", pe_text)
        .replace("__GWTC3_TOP__", format_top_candidates(pe_tables["gwtc3"]))
        .replace("__GWTC4_TOP__", format_top_candidates(pe_tables["gwtc4"]))
    )
    (root / "unified_sky_v8_method_and_results_cn.md").write_text(
        cn, encoding="utf-8"
    )

    english = """# Unified posterior-sky retrieval v8: method and results

## Scope

This experiment aligns ET-3, GWTC-3/O3, and GWTC-4.1/O4a under the same
three-channel retrieval semantics: calibrated waveform evidence,
exposure-conditioned time-delay evidence, and event-level posterior-sky
evidence. It is a candidate-generation and companion-retrieval study, not a
lensing detection or an LVK-equivalent false-alarm analysis.

## What changed from v7

The v7 waveform encoders, system splits, waveform calibrations, and frozen
time-delay likelihood ratios are unchanged. Sky posteriors, sky pair
statistics, validation-only sky calibration, fusion weights, and final ranks
are recomputed. GWTC uses public PE HEALPix maps. ET uses a response-derived
surrogate: a GWFAST/IMRPhenomD local Fisher width, ET-D sensitivity and Earth
motion, plus the eight published short-signal degeneracy modes of a
co-located triangular ET. It is explicitly not full Bayesian ET PE.

## Unified sky statistic

All event maps are non-negative, normalized probability-mass vectors on a
RING-ordered nside=32 grid. For each pair:

\\[
B_{{sky}}=N_{{pix}}\\sum_k P_{{ik}}P_{{jk}},\\quad
f_1=\\log B_{{sky}},
\\]

\\[
f_2=\\frac{{|C_i^{{90}}\\cap C_j^{{90}}|}}
{{\\min(|C_i^{{90}}|,|C_j^{{90}}|)}},
\\]

and `f3` is the symmetric cross-HPD statistic of Wong et al. A class-balanced,
non-negative-coefficient logistic calibration is fitted on synthetic
validation pairs only:

\\[
s_{{sky}}=b+\\sum_m\\beta_m z_{{val}}(f_m),\\qquad\\beta_m\\ge0.
\\]

The calibrated output is a density-ratio surrogate, not an exact Bayes
factor. A non-overlapping-90%-region veto is enabled only when validation
true-pair retention is at least 99%. Test labels and real-catalog ranks are
never used for fitting.

__SKY_CALIBRATION_TABLE__

## v7 to v8 retrieval comparison

__VERSION_COMPARISON__

## Fusion

\\[
S_{{ij}}=w_w s_{{w,ij}}+w_t s_{{t,ij}}+w_s s_{{sky,ij}}.
\\]

Weights are selected separately for each deployment and training seed using
held-out validation systems. The main cross-deployment comparison uses the
predeclared strict-positive three-channel retrieval solution. The
unconstrained, time+sky, and equal-evidence solutions are retained as
diagnostics. There is no real-catalog row standardization.

## R@1

__R1_TABLE__

## R@10

__R10_TABLE__

Values are mean ± sample SD across five ET seeds and three seeds per GWTC
deployment. Each seed also has a 10,000-draw, family-stratified,
system-cluster bootstrap; the two directed queries from one lensed system are
resampled together.

ET full-pair metrics use all 40,495,500 unordered pairs rather than sampled
negatives. Key operating points across the five training seeds are:

__ET_OPERATING_POINTS__

## Real-catalog PE audit

PE is not a fourth retrieval channel. After consensus ranking, detector-frame
chirp mass, mass ratio, and effective-spin posteriors are audited with

\\[
D_x=\\frac{{|\\mathrm{{median}}(x_i)-\\mathrm{{median}}(x_j)|}}
{{\\sqrt{{\\sigma_i^2+\\sigma_j^2}}}}.
\\]

A predeclared screen requires all three distances to be at most 3. The tables
below are candidate shortlists for Bayesian follow-up, not detections.

__CATALOG_SCOPE__

### GWTC-3/O3 top 10

__GWTC3_TOP__

### GWTC-4.1/O4a top 10

__GWTC4_TOP__

## Limitations

ET sky maps remain posterior surrogates; GWTC sky-only discrimination is weak
because many public posteriors are broad; injection retrieval does not
measure real-catalog precision; a 3-sigma PE screen is not a lensing Bayes
factor; catalog rank is not a p-value.

## References

Haris et al. (arXiv:1807.07062); Wong et al. (arXiv:2112.05932);
Iacovelli et al. (arXiv:2207.06910); Santoliquido et al.
(arXiv:2504.21087); LVK O3 lensing search (arXiv:2304.08393);
LVK O4a lensing search (LIGO-P2500419-v10); Garrón & Keitel
(arXiv:2306.12908).
"""
    english = english.replace("{{", "{").replace("}}", "}")
    english = (
        english.replace("__R1_TABLE__", r1_table)
        .replace("__R10_TABLE__", r10_table)
        .replace("__VERSION_COMPARISON__", comparison_table)
        .replace("__CATALOG_SCOPE__", scope_table)
        .replace("__SKY_CALIBRATION_TABLE__", sky_table)
        .replace("__ET_OPERATING_POINTS__", operating_table)
        .replace("__GWTC3_TOP__", format_top_candidates(pe_tables["gwtc3"]))
        .replace("__GWTC4_TOP__", format_top_candidates(pe_tables["gwtc4"]))
    )
    (root / "unified_sky_v8_method_and_results_en.md").write_text(
        english, encoding="utf-8"
    )
    readme = """# Unified sky v8 current results

Authoritative result directory: `results/unified_sky_v8_20260724`.

Use `unified_sky_v8_method_and_results_cn.md` for the complete Chinese
method/results description and `unified_sky_v8_method_and_results_en.md` for
the English summary. Primary plots are under `figures/`.

This release supersedes only the sky-posterior construction, sky
calibration, fusion weights, and ranks from v7. It deliberately reuses the
frozen v7 waveform and time-delay evidence so the effect of the revised sky
model is identifiable. Results are retrieval/candidate-generation evidence,
not lensing detections.
"""
    (root / "README_CURRENT_RESULTS.md").write_text(readme, encoding="utf-8")


def make_reproduce(root: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gw-catalog
OUT=results/unified_sky_v8_20260724
PY=/root/miniconda3/bin/python
GWPY=.venv_sky_v8/bin/python

for shard in 0 1 2 3; do
  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    "$GWPY" scripts/experiments/113_et3_sky_v8_gwfast_fisher.py \
    --output-root "$OUT" --num-shards 4 --shard-index "$shard" \
    --chunk-size 500 --frequency-resolution 100 &
done
wait
"$GWPY" scripts/experiments/113_et3_sky_v8_gwfast_fisher.py \
  --output-root "$OUT" --num-shards 4 --merge-shards
"$PY" scripts/experiments/114_gwtc_sky_v8_pipeline.py \
  --output-root "$OUT" --deployment all
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/experiments/115_et3_sky_v8_pipeline.py \
  --output-root "$OUT"
"$PY" scripts/experiments/116_unified_sky_v8_report.py \
  --output-root "$OUT"
"""
    path = root / "reproduce_unified_sky_v8.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def make_summary_json(
    root: Path,
    retrieval_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    weight_summary: pd.DataFrame,
    sky_summary: pd.DataFrame,
    posterior_audit: pd.DataFrame,
    pe_summary: pd.DataFrame,
    history: pd.DataFrame,
    catalog_scope: pd.DataFrame,
) -> None:
    gates = {}
    for deployment in ("gwtc3", "gwtc4"):
        frame = pd.read_csv(root / deployment / "waveform_gate_per_seed_v8.csv")
        gates[deployment] = {
            "passed_seeds": int(frame["passed_for_primary_deployment"].sum()),
            "total_seeds": int(len(frame)),
            "all_passed": bool(frame["passed_for_primary_deployment"].all()),
        }
    payload = {
        "version": "unified_sky_v8",
        "status": "complete",
        "scientific_scope": (
            "Companion retrieval and candidate-generation evidence; not a "
            "lensing detection or LVK-equivalent FAR."
        ),
        "retrieval_summary": retrieval_summary.to_dict(orient="records"),
        "pair_level_summary": pair_summary.to_dict(orient="records"),
        "et3_pair_level_operating_points": pd.read_csv(
            root / "et3_pair_level_operating_points_summary_v8.csv"
        ).to_dict(orient="records"),
        "strict_positive_weight_summary": weight_summary.to_dict(orient="records"),
        "sky_calibration_summary": sky_summary.to_dict(orient="records"),
        "posterior_map_audit": posterior_audit.to_dict(orient="records"),
        "gwtc_waveform_gate": gates,
        "real_catalog_scope": catalog_scope.to_dict(orient="records"),
        "real_candidate_pe_enrichment": pe_summary.to_dict(orient="records"),
        "historical_pair_rank_audit": history.to_dict(orient="records"),
        "key_interpretation": [
            "GWTC and ET use the same HEALPix pair-statistic implementation.",
            (
                "ET uses a GWFAST-local-width, eight-mode response-derived "
                "posterior surrogate rather than the optimistic v7 Gaussian."
            ),
            (
                "Real PE is excluded from retrieval tuning and used only for "
                "post-retrieval physical-consistency auditing."
            ),
        ],
    }
    write_json(root / "unified_sky_v8_summary.json", payload)


def build_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.name == "data_inventory_v8.csv":
            continue
        row_count: int | None = None
        if path.suffix == ".parquet":
            try:
                row_count = int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                row_count = None
        elif path.suffix == ".csv":
            with path.open("rb") as handle:
                row_count = max(sum(chunk.count(b"\n") for chunk in iter(
                    lambda: handle.read(1024 * 1024), b""
                )) - 1, 0)
        rows.append(
            {
                "relative_path": str(relative),
                "suffix": path.suffix,
                "size_bytes": path.stat().st_size,
                "row_count_if_tabular": row_count,
                "included_in_portable_package": not (
                    path.suffix == ".npy"
                    or "validation_sky_calibration_pairs" in path.name
                    or "fusion_validation_pairs" in path.name
                    or "fusion_heldout_test_pairs" in path.name
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(root / "data_inventory_v8.csv", index=False)
    return output


def package(root: Path, package_path: Path) -> dict[str, Any]:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    include_root_files = [
        "README_CURRENT_RESULTS.md",
        "unified_sky_v8_method_and_results_cn.md",
        "unified_sky_v8_method_and_results_en.md",
        "run_config_v8.json",
        "unified_sky_v8_summary.json",
        "data_inventory_v8.csv",
        "reproduce_unified_sky_v8.sh",
        "unified_retrieval_metrics_per_seed_v8.csv",
        "unified_retrieval_metrics_summary_v8.csv",
        "unified_pair_metrics_per_seed_v8.csv",
        "unified_pair_metrics_summary_v8.csv",
        "et3_pair_level_operating_points_per_seed_v8.csv",
        "et3_pair_level_operating_points_summary_v8.csv",
        "unified_selected_weights_per_seed_v8.csv",
        "unified_selected_weights_summary_v8.csv",
        "sky_calibration_per_seed_v8.csv",
        "sky_calibration_summary_v8.csv",
        "posterior_map_audit_summary_v8.csv",
        "v7_v8_retrieval_comparison.csv",
        "real_candidate_pe_enrichment_v8.csv",
        "real_catalog_scope_v8.csv",
        "historical_candidate_rank_audit_v8.csv",
        "sky_score_distribution_sample_v8.parquet",
    ]
    include_patterns = [
        "figures/*",
        "shared/*audit*.json",
        "shared/*audit*.csv",
        "gwtc3/*metrics*csv",
        "gwtc3/*weights*csv",
        "gwtc3/*gate*csv",
        "gwtc3/*bootstrap*csv",
        "gwtc3/real_candidate_consensus_with_pe_v8.parquet",
        "gwtc3/real_candidate_top20_with_pe_v8.csv",
        "gwtc3/real_candidate_consensus_top100_v8.csv",
        "gwtc3/seed_*/sky_calibration_v8.json",
        "gwtc3/seed_*/selected_weights_v8.json",
        "gwtc3/seed_*/heldout_test_retrieval_metrics_v8.csv",
        "gwtc3/seed_*/heldout_test_pair_level_metrics_v8.csv",
        "gwtc3/seed_*/heldout_test_bootstrap_95ci_v8.csv",
        "gwtc3/seed_*/heldout_test_query_ranks_v8.parquet",
        "gwtc3/seed_*/real_pair_features_unified_sky_v8.parquet",
        "gwtc3/seed_*/real_pair_scores_all_catalog_candidate_three_channel_unconstrained_v8.parquet",
        "gwtc3/seed_*/real_pair_scores_strict_h1l1_bbh_candidate_three_channel_unconstrained_v8.parquet",
        "gwtc3/seed_*/real_pair_scores_all_catalog_time_sky_candidate_selected_v8.parquet",
        "gwtc3/seed_*/real_pair_scores_strict_h1l1_bbh_time_sky_candidate_selected_v8.parquet",
        "gwtc4/*metrics*csv",
        "gwtc4/*weights*csv",
        "gwtc4/*gate*csv",
        "gwtc4/*bootstrap*csv",
        "gwtc4/real_candidate_consensus_with_pe_v8.parquet",
        "gwtc4/real_candidate_top20_with_pe_v8.csv",
        "gwtc4/real_candidate_consensus_top100_v8.csv",
        "gwtc4/seed_*/sky_calibration_v8.json",
        "gwtc4/seed_*/selected_weights_v8.json",
        "gwtc4/seed_*/heldout_test_retrieval_metrics_v8.csv",
        "gwtc4/seed_*/heldout_test_pair_level_metrics_v8.csv",
        "gwtc4/seed_*/heldout_test_bootstrap_95ci_v8.csv",
        "gwtc4/seed_*/heldout_test_query_ranks_v8.parquet",
        "gwtc4/seed_*/real_pair_features_unified_sky_v8.parquet",
        "gwtc4/seed_*/real_pair_scores_all_catalog_candidate_three_channel_unconstrained_v8.parquet",
        "gwtc4/seed_*/real_pair_scores_strict_h1l1_bbh_candidate_three_channel_unconstrained_v8.parquet",
        "gwtc4/seed_*/real_pair_scores_all_catalog_time_sky_candidate_selected_v8.parquet",
        "gwtc4/seed_*/real_pair_scores_strict_h1l1_bbh_time_sky_candidate_selected_v8.parquet",
        "et3/*metrics*csv",
        "et3/*weights*csv",
        "et3/*bootstrap*csv",
        "et3/seed_*/sky_calibration_v8.json",
        "et3/seed_*/selected_weights_v8.json",
        "et3/seed_*/retrieval_metrics_v8.csv",
        "et3/seed_*/query_ranks_v8.parquet",
        "et3/seed_*/bootstrap_95ci_v8.csv",
        "et3/seed_*/pair_level_metrics_*_v8.json",
        "et3/seed_*/pair_level_operating_points_*_v8.csv",
        "et3/seed_*/pair_diagnostics_sample_v8.parquet",
        "logs/*.log",
    ]
    paths = []
    for relative in include_root_files:
        path = root / relative
        if path.exists():
            paths.append(path)
    for pattern in include_patterns:
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    script_paths = [
        REPO / "scripts/real_search/common.py",
        REPO / "scripts/real_search/physical_common.py",
        REPO / "scripts/real_search/unified_v7_common.py",
        REPO / "scripts/real_search/unified_sky_v8.py",
        REPO / "scripts/real_search/et_sky_v8.py",
        REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py",
        REPO / "scripts/experiments/110_et3_v7_aligned_pipeline.py",
        REPO / "scripts/experiments/113_et3_sky_v8_gwfast_fisher.py",
        REPO / "scripts/experiments/114_gwtc_sky_v8_pipeline.py",
        REPO / "scripts/experiments/115_et3_sky_v8_pipeline.py",
        REPO / "scripts/experiments/116_unified_sky_v8_report.py",
        REPO / "scripts/experiments/117_et3_fisher_resolution_audit.py",
    ]
    paths.extend(path for path in script_paths if path.exists())
    provenance_paths = [
        V7_GWTC / "run_config_v7.json",
        V7_ET / "run_config.json",
        REPO / "docs/methods/et3_gwtc_unified_v7_method_20260723_cn.md",
    ]
    paths.extend(path for path in provenance_paths if path.exists())
    unique = sorted(set(paths))
    with tarfile.open(package_path, "w:gz") as archive:
        for path in unique:
            if path.is_relative_to(root):
                target = Path("unified_sky_v8_20260724") / path.relative_to(root)
            else:
                target = Path("unified_sky_v8_20260724") / path.relative_to(REPO)
            archive.add(path, arcname=str(target))
    digest = hashlib.sha256(package_path.read_bytes()).hexdigest()
    package_path.with_suffix(package_path.suffix + ".sha256").write_text(
        f"{digest}  {package_path.name}\n",
        encoding="ascii",
    )
    return {
        "package_path": str(package_path),
        "file_count": len(unique),
        "size_bytes": package_path.stat().st_size,
        "sha256": digest,
        "excluded": [
            "raw GWOSC strain",
            "complete public PE posterior files",
            "large ET/GWTC posterior-map arrays",
            "encoder checkpoints and full embedding matrices already frozen in v7",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--package-path", type=Path, default=PACKAGE_PATH)
    args = parser.parse_args()
    root = args.output_root
    (root / "figures").mkdir(parents=True, exist_ok=True)

    retrieval, retrieval_summary = load_retrieval(root)
    version_comparison = make_v7_v8_comparison(root, retrieval_summary)
    _, pair_summary = load_pair_metrics(root)
    aggregate_et_operating_points(root)
    _, weight_summary = load_weights(root)
    _, sky_summary = sky_calibration_summary(root)
    posterior_audit = posterior_map_audit_summary(root)
    pe_tables = {}
    pe_rows = []
    for deployment in ("gwtc3", "gwtc4"):
        pe_tables[deployment], summary = merge_pe_audit(root, deployment)
        pe_rows.append(summary)
    pe_summary = pd.concat(pe_rows, ignore_index=True)
    pe_summary.to_csv(root / "real_candidate_pe_enrichment_v8.csv", index=False)
    history = historical_ranks(root)
    catalog_scope = real_catalog_scope(root)
    sky_samples = collect_sky_samples(root)
    plot_main(root, retrieval, sky_samples, pe_tables)
    plot_candidate_diagnostics(root, pe_tables)
    plot_revision_diagnostics(root, version_comparison)

    configuration = {
        "version": "unified_sky_v8",
        "date": "2026-07-24",
        "status": "complete",
        "waveform_and_time_base": "frozen v7 evidence",
        "changed_components": [
            "ET posterior construction",
            "unified HEALPix pair statistics",
            "validation-only sky calibration",
            "validation-only fusion weights",
            "test metrics and real ranks",
        ],
        "common_nside": 32,
        "sky_features": [
            "log common-source sky Bayes factor",
            "90%-credible-region overlap",
            "symmetric cross-HPD",
        ],
        "sky_calibration": {
            "model": "class-balanced monotone logistic",
            "l2_penalty": 1.0,
            "fit_split": "synthetic validation systems only",
        },
        "et_posterior": {
            "local_width": "GWFAST 1.1.2 IMRPhenomD Fisher",
            "detector": "one triangular ET-D site, ET1/ET2/ET3 response",
            "fmin_hz": 5,
            "earth_motion": True,
            "global_modes": 8,
            "label": "posterior surrogate, not full PE",
        },
        "gwtc_seeds": list(GWTC_SEEDS),
        "et_seeds": list(ET_SEEDS),
        "unordered_pair_rule": "max(S_ij,S_ji)",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json(root / "run_config_v8.json", configuration)
    make_reproduce(root)
    make_reports(
        root,
        retrieval_summary,
        pair_summary,
        weight_summary,
        sky_summary,
        version_comparison,
        catalog_scope,
        pe_tables,
        pe_summary,
        history,
    )
    make_summary_json(
        root,
        retrieval_summary,
        pair_summary,
        weight_summary,
        sky_summary,
        posterior_audit,
        pe_summary,
        history,
        catalog_scope,
    )
    build_inventory(root)
    package_audit = package(root, args.package_path)
    write_json(root / "package_audit_v8.json", package_audit)
    print(json.dumps(package_audit, indent=2))


if __name__ == "__main__":
    main()
