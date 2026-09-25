#!/usr/bin/env python3
"""Build the integrated ET-3 + GWTC-3 + GWTC-4.1 result delivery.

This script does not retrain models or rescore catalogs. It assembles the
authoritative ET moderate-sky v8.2 results, GWTC unified-sky v8.1 results, and
the independently recomputed GWTC3/GWTC4 Top-B truth accounting. It also
creates cross-deployment tables, figures, reports, and a reproducibility
inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/et3_gwtc34_complete_20260726"
ET_SOURCE = REPO / "results/et3_moderate_sky_gwtc3_topb_20260726"
GWTC_SOURCE = REPO / "results/unified_sky_v81_20260725"
METHOD_DOCUMENT = (
    REPO / "docs/methods/complete_et3_gwtc34_method_20260726_cn.md"
)

DEPLOYMENT_LABELS = {
    "ET3": "ET-3 moderate",
    "gwtc3": "GWTC-3.0 / O3",
    "gwtc4": "GWTC-4.1 / O4a",
}
METHOD_ORDER = (
    "waveform_only",
    "time_only",
    "sky_only",
    "time_sky",
    "three_channel",
)
METHOD_LABELS = {
    "waveform_only": "Waveform",
    "time_only": "Time delay",
    "sky_only": "Sky",
    "time_sky": "Time + sky",
    "three_channel": "Three channel",
}
COLORS = {
    "ET3": "#2f658f",
    "gwtc3": "#b15b3a",
    "gwtc4": "#558a54",
}


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, dirs_exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
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
            "savefig.dpi": 300,
        }
    )


def normalize_et_retrieval() -> pd.DataFrame:
    path = ET_SOURCE / "et3/moderate/retrieval_metrics_per_seed_v82.csv"
    frame = pd.read_csv(path)
    method_map = {
        "waveform_only": "waveform_only",
        "time_only": "time_only",
        "sky_only": "sky_only",
        "time_sky_validation_selected": "time_sky",
        "three_channel_strict_positive": "three_channel",
    }
    frame = frame[
        frame["method"].isin(method_map) & frame["subset"].eq("overall")
    ].copy()
    frame["deployment"] = "ET3"
    frame["deployment_label"] = DEPLOYMENT_LABELS["ET3"]
    frame["source_method"] = frame["method"]
    frame["method"] = frame["method"].map(method_map)
    frame["selection_objective"] = "retrieval"
    return frame[
        [
            "deployment",
            "deployment_label",
            "seed",
            "method",
            "source_method",
            "selection_objective",
            "n_queries",
            "n_systems",
            "r_at_1",
            "r_at_5",
            "r_at_10",
            "r_at_50",
            "median_rank",
        ]
    ]


def normalize_gwtc_retrieval(deployment: str) -> pd.DataFrame:
    path = (
        GWTC_SOURCE
        / deployment
        / "heldout_test_retrieval_metrics_per_seed_v81.csv"
    )
    frame = pd.read_csv(path)
    method_map = {
        "waveform_only": "waveform_only",
        "time_only": "time_only",
        "sky_only": "sky_only",
        "time_sky_retrieval_selected": "time_sky",
        "retrieval_three_channel_strict_positive": "three_channel",
    }
    frame = frame[
        frame["method"].isin(method_map) & frame["subset"].eq("overall")
    ].copy()
    frame["deployment"] = deployment
    frame["deployment_label"] = DEPLOYMENT_LABELS[deployment]
    frame["source_method"] = frame["method"]
    frame["method"] = frame["method"].map(method_map)
    frame["selection_objective"] = "retrieval"
    return frame[
        [
            "deployment",
            "deployment_label",
            "seed",
            "method",
            "source_method",
            "selection_objective",
            "n_queries",
            "r_at_1",
            "r_at_5",
            "r_at_10",
            "median_rank",
        ]
    ]


def summarize_retrieval(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in per_seed.groupby(
        ["deployment", "deployment_label", "method"],
        sort=False,
    ):
        deployment, deployment_label, method = key
        row: dict[str, Any] = {
            "deployment": deployment,
            "deployment_label": deployment_label,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_seeds": int(part["seed"].nunique()),
            "n_queries": int(part["n_queries"].iloc[0]),
        }
        for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
            values = part[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    output = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    deployment_order = {"ET3": 0, "gwtc3": 1, "gwtc4": 2}
    output["_deployment_order"] = output["deployment"].map(deployment_order)
    output["_method_order"] = output["method"].map(order)
    return output.sort_values(
        ["_deployment_order", "_method_order"]
    ).drop(columns=["_deployment_order", "_method_order"])


def selected_weight_table() -> pd.DataFrame:
    frames = []
    et = pd.read_csv(
        ET_SOURCE / "et3/moderate/selected_weights_per_seed_v82.csv"
    )
    et.insert(0, "deployment", "ET3")
    frames.append(et)
    for deployment in ("gwtc3", "gwtc4"):
        frames.append(
            pd.read_csv(
                GWTC_SOURCE
                / deployment
                / "selected_weights_per_seed_v81.csv"
            )
        )
    output = pd.concat(frames, ignore_index=True)
    output.insert(
        1,
        "deployment_label",
        output["deployment"].map(DEPLOYMENT_LABELS),
    )
    return output


def copy_authoritative_results(output: Path) -> None:
    et_target = output / "et3_moderate"
    copy_tree(ET_SOURCE / "et3", et_target / "et3")
    for name in (
        "README_RESULTS.md",
        "data_inventory.csv",
        "et3_moderate_sky_gwtc3_topb_report_cn.md",
        "et3_moderate_sky_gwtc3_topb_report_en.md",
        "et3_moderate_sky_pair_geometry_sample.parquet",
        "et3_moderate_sky_pair_geometry_summary.csv",
        "et3_moderate_vs_v81.csv",
        "et3_scenario_sensitivity_v82.csv",
        "experiment_summary.json",
        "run_config.json",
    ):
        copy_file(ET_SOURCE / name, et_target / name)
    copy_tree(ET_SOURCE / "figures", et_target / "figures")

    gwtc_target = output / "gwtc_v81"
    for deployment in ("gwtc3", "gwtc4"):
        copy_tree(GWTC_SOURCE / deployment, gwtc_target / deployment)
    for name in (
        "README_CURRENT_RESULTS.md",
        "data_inventory_v81.csv",
        "deployment_data_summary_v81.csv",
        "historical_candidate_rank_audit_v81.csv",
        "posterior_map_audit_summary_v81.csv",
        "real_candidate_pe_enrichment_v81.csv",
        "real_catalog_scope_v81.csv",
        "run_config_v81.json",
        "sky_score_validation_per_seed_v81.csv",
        "sky_score_validation_summary_v81.csv",
        "unified_pair_metrics_per_seed_v81.csv",
        "unified_pair_metrics_summary_v81.csv",
        "unified_retrieval_metrics_per_seed_v81.csv",
        "unified_retrieval_metrics_summary_v81.csv",
        "unified_selected_weights_per_seed_v81.csv",
        "unified_selected_weights_summary_v81.csv",
        "unified_sky_v81_method_and_results_cn.md",
        "unified_sky_v81_method_and_results_en.md",
        "unified_sky_v81_summary.json",
        "v7_v81_retrieval_comparison.csv",
        "v8_v81_retrieval_comparison.csv",
    ):
        copy_file(GWTC_SOURCE / name, gwtc_target / name)
    copy_tree(GWTC_SOURCE / "figures", gwtc_target / "figures")


def copy_code(output: Path) -> None:
    experiment_names = (
        "20_real_noise_injection_v3_physical.py",
        "109_materialize_et3_v7_preprocessing.py",
        "110_et3_v7_aligned_pipeline.py",
        "118_et3_sky_v81_gwfast_covariance.py",
        "119_et3_sky_v81_posterior_bank.py",
        "120_et3_sky_v81_pipeline.py",
        "121_gwtc_sky_v81_pipeline.py",
        "122_unified_sky_v81_report.py",
        "124_et3_moderate_observed_sky.py",
        "127_gwtc34_real_noise_injection_topb.py",
        "128_build_et3_gwtc34_complete_delivery.py",
    )
    real_search_names = (
        "34_generate_physical_h1l1_source_bank.py",
        "35_real_noise_injection_v5_physical_source.py",
        "36_materialize_multinoise_v5.py",
        "37_unified_intrinsic_multitask_pilot.py",
        "38_real_noise_injection_v6_unified_physics.py",
        "39_physical_deployment_v6_pe_audit.py",
        "40_select_waveform_pilot_validation.py",
        "42_prebuild_v6_training_data.py",
        "43_make_v6_deliverables.py",
        "44_et3_sky_bayes_factor_refusion.py",
        "45_real_noise_injection_v7_peak2s.py",
        "46_physical_deployment_v7_pe_audit.py",
        "et_sky_v8.py",
        "et_sky_v81.py",
        "physical_common.py",
        "physical_source_v5_common.py",
        "sky_channel_reference.py",
        "unified_sky_v8.py",
        "unified_sky_v81.py",
        "unified_v6_common.py",
        "unified_v7_common.py",
    )
    for name in experiment_names:
        copy_file(
            REPO / "scripts/experiments" / name,
            output / "scripts/experiments" / name,
        )
    for name in real_search_names:
        copy_file(
            REPO / "scripts/real_search" / name,
            output / "scripts/real_search" / name,
        )
    copy_file(
        REPO / "scripts/real_search/reproduce_v7_peak2s.sh",
        output / "scripts/real_search/reproduce_v7_peak2s.sh",
    )


def build_summary_tables(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_dir = output / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    retrieval_per_seed = pd.concat(
        [
            normalize_et_retrieval(),
            normalize_gwtc_retrieval("gwtc3"),
            normalize_gwtc_retrieval("gwtc4"),
        ],
        ignore_index=True,
    )
    retrieval_summary = summarize_retrieval(retrieval_per_seed)
    retrieval_per_seed.to_csv(
        summary_dir / "et3_gwtc34_retrieval_per_seed.csv",
        index=False,
    )
    retrieval_summary.to_csv(
        summary_dir / "et3_gwtc34_retrieval_summary.csv",
        index=False,
    )
    selected_weight_table().to_csv(
        summary_dir / "et3_gwtc34_selected_weights_per_seed.csv",
        index=False,
    )

    candidate_frames = []
    for deployment in ("gwtc3", "gwtc4"):
        frame = pd.read_csv(
            GWTC_SOURCE / deployment / "real_candidate_top20_with_pe_v81.csv"
        )
        candidate_frames.append(frame)
        frame.head(10).to_csv(
            summary_dir / f"{deployment}_real_candidate_top10_with_pe.csv",
            index=False,
        )
    pd.concat(candidate_frames, ignore_index=True).to_csv(
        summary_dir / "gwtc34_real_candidate_top20_with_pe.csv",
        index=False,
    )

    for name in (
        "deployment_data_summary_v81.csv",
        "real_catalog_scope_v81.csv",
        "real_candidate_pe_enrichment_v81.csv",
        "historical_candidate_rank_audit_v81.csv",
        "posterior_map_audit_summary_v81.csv",
    ):
        copy_file(GWTC_SOURCE / name, summary_dir / name)

    topb = pd.read_csv(
        output / "gwtc_topb/gwtc34_topb_true_false_summary.csv"
    )
    primary_topb = topb[
        topb["method"].eq("candidate_three_channel_strict_positive")
    ].copy()
    primary_topb[
        primary_topb["budget_type"].eq("fixed_count")
    ].to_csv(
        summary_dir / "gwtc34_primary_topb_fixed_count_summary.csv",
        index=False,
    )
    primary_topb[
        primary_topb["budget_type"].eq("top_pair_fraction")
    ].to_csv(
        summary_dir / "gwtc34_primary_topb_fraction_summary.csv",
        index=False,
    )
    topb_per_seed = pd.read_csv(
        output / "gwtc_topb/gwtc34_topb_true_false_per_seed.csv"
    )
    topb_per_seed[
        topb_per_seed["method"].eq(
            "candidate_three_channel_strict_positive"
        )
    ].to_csv(
        summary_dir / "gwtc34_primary_topb_per_seed.csv",
        index=False,
    )
    return retrieval_per_seed, topb


def plot_complete(
    output: Path,
    retrieval_per_seed: pd.DataFrame,
    topb_summary: pd.DataFrame,
) -> None:
    style()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))

    # a: Cross-deployment directed companion R@10.
    axis = axes[0, 0]
    x = np.arange(len(METHOD_ORDER))
    offsets = {"ET3": -0.18, "gwtc3": 0.0, "gwtc4": 0.18}
    for deployment in ("ET3", "gwtc3", "gwtc4"):
        part = retrieval_per_seed[
            retrieval_per_seed["deployment"].eq(deployment)
        ]
        means = []
        stds = []
        for method in METHOD_ORDER:
            values = part.loc[part["method"].eq(method), "r_at_10"].to_numpy()
            means.append(values.mean())
            stds.append(values.std(ddof=1))
            jitter = np.linspace(-0.035, 0.035, len(values))
            axis.scatter(
                x[METHOD_ORDER.index(method)]
                + offsets[deployment]
                + jitter,
                values,
                s=9,
                facecolors="white",
                edgecolors=COLORS[deployment],
                linewidths=0.6,
                zorder=4,
            )
        axis.errorbar(
            x + offsets[deployment],
            means,
            yerr=stds,
            marker="o",
            markersize=4,
            linewidth=1.1,
            capsize=2,
            color=COLORS[deployment],
            label=DEPLOYMENT_LABELS[deployment],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [METHOD_LABELS[name] for name in METHOD_ORDER],
        rotation=23,
        ha="right",
    )
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel("Directed companion R@10")
    axis.set_title("Held-out companion retrieval")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.18)

    primary = topb_summary[
        topb_summary["method"].eq(
            "candidate_three_channel_strict_positive"
        )
        & topb_summary["budget_type"].eq("fixed_count")
    ].copy()
    primary = primary[primary["budget_pairs"].le(1000)]

    # b: Global Top-B precision.
    axis = axes[0, 1]
    for deployment in ("gwtc3", "gwtc4"):
        part = primary[primary["deployment"].eq(deployment)].sort_values(
            "budget_pairs"
        )
        axis.errorbar(
            part["budget_pairs"],
            part["precision_mean"],
            yerr=part["precision_std"],
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=COLORS[deployment],
            label=DEPLOYMENT_LABELS[deployment],
        )
    base_rate = float(primary["base_positive_rate"].iloc[0])
    axis.axhline(
        base_rate,
        color="black",
        linestyle=":",
        linewidth=0.9,
        label=f"Base rate {base_rate:.4f}",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Global shortlist budget B")
    axis.set_ylabel("Precision@B")
    axis.set_title("Injection-test Top-B precision")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(alpha=0.18, which="both")

    # c: Global Top-B recall.
    axis = axes[1, 0]
    for deployment in ("gwtc3", "gwtc4"):
        part = primary[primary["deployment"].eq(deployment)].sort_values(
            "budget_pairs"
        )
        axis.errorbar(
            part["budget_pairs"],
            part["recall_mean"],
            yerr=part["recall_std"],
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=COLORS[deployment],
            label=DEPLOYMENT_LABELS[deployment],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Global shortlist budget B")
    axis.set_ylabel("Recall@B")
    axis.set_ylim(0.0, 0.36)
    axis.set_title("Injection-test Top-B recall")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(alpha=0.18, which="both")

    # d: Post-ranking PE consistency in real catalogs.
    axis = axes[1, 1]
    enrichment = pd.read_csv(
        GWTC_SOURCE / "real_candidate_pe_enrichment_v81.csv"
    )
    for deployment in ("gwtc3", "gwtc4"):
        part = enrichment[enrichment["deployment"].eq(deployment)]
        axis.plot(
            part["top_n"],
            part["consistency_fraction"],
            marker="o",
            markersize=4,
            linewidth=1.2,
            color=COLORS[deployment],
            label=DEPLOYMENT_LABELS[deployment],
        )
        baseline = float(part["all_pair_consistency_fraction"].iloc[0])
        axis.axhline(
            baseline,
            color=COLORS[deployment],
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
        )
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Top-N real catalog pairs")
    axis.set_ylabel("Fraction passing intrinsic PE screen")
    axis.set_title("Real-catalog PE audit (not truth)")
    axis.legend(frameon=False, loc="lower left")
    axis.grid(alpha=0.18, which="both")

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.14,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(figures / "fig_et3_gwtc34_complete_results.pdf")
    fig.savefig(figures / "fig_et3_gwtc34_complete_results.png")
    plt.close(fig)


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: list[str] | None = None,
    formats: dict[str, str] | None = None,
) -> str:
    formats = formats or {}
    labels = labels or columns
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formats:
                value = formats[column].format(value)
            elif isinstance(value, (float, np.floating)):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_cn(
    output: Path,
    retrieval_summary: pd.DataFrame,
    topb_summary: pd.DataFrame,
) -> None:
    retrieval = retrieval_summary.copy()
    retrieval["R1"] = retrieval.apply(
        lambda row: f"{row.r_at_1_mean:.4f} +/- {row.r_at_1_std:.4f}",
        axis=1,
    )
    retrieval["R10"] = retrieval.apply(
        lambda row: f"{row.r_at_10_mean:.4f} +/- {row.r_at_10_std:.4f}",
        axis=1,
    )
    retrieval_table = markdown_table(
        retrieval,
        ["deployment_label", "method_label", "R1", "R10"],
        ["部署", "方法", "R@1", "R@10"],
    )

    primary = topb_summary[
        topb_summary["method"].eq(
            "candidate_three_channel_strict_positive"
        )
        & topb_summary["budget_type"].eq("fixed_count")
        & topb_summary["budget_pairs"].isin([5, 10, 20, 50, 100, 200, 500])
    ].copy()
    topb_table = markdown_table(
        primary,
        [
            "deployment_label",
            "budget_pairs",
            "true_recovered_mean",
            "false_candidates_mean",
            "precision_mean",
            "recall_mean",
            "enrichment_over_base_rate_mean",
        ],
        ["部署", "B", "TP", "FP", "Precision", "Recall", "Enrichment"],
        {
            "true_recovered_mean": "{:.2f}",
            "false_candidates_mean": "{:.2f}",
            "precision_mean": "{:.4f}",
            "recall_mean": "{:.4f}",
            "enrichment_over_base_rate_mean": "{:.1f}x",
        },
    )

    enrichment = pd.read_csv(
        GWTC_SOURCE / "real_candidate_pe_enrichment_v81.csv"
    )
    enrichment = enrichment[enrichment["top_n"].isin([5, 10, 20])]
    enrichment["deployment_label"] = enrichment["deployment"].map(
        DEPLOYMENT_LABELS
    )
    pe_table = markdown_table(
        enrichment,
        [
            "deployment_label",
            "top_n",
            "n_intrinsic_3sigma_consistent",
            "consistency_fraction",
            "enrichment_over_all_pairs",
        ],
        ["部署", "Top-N", "PE pass", "Pass fraction", "Enrichment"],
        {
            "consistency_fraction": "{:.3f}",
            "enrichment_over_all_pairs": "{:.2f}x",
        },
    )

    content = f"""# ET-3 + GWTC-3.0 + GWTC-4.1 完整结果报告

## 1. 最简结论

- ET-3 使用中等信息量 observed-sky 模拟；GWTC-3/4.1 真实目录继续使用真实
  PE HEALPix sky posterior。
- 三者的主 sky pair score 都是
  `log(N_pix * sum(P_i * P_j))`，并继续与 waveform、time 构成三个分数矩阵。
- GWTC-3 和 GWTC-4.1 均完成 held-out real-noise injection Top-B 真/假对统计。
- 真实目录没有真值标签；真实 Top-N 只报告候选排名和 PE screen，不能写成真/假对。

## 2. Directed companion retrieval

这里使用每个 seed 在 validation systems 上冻结的 retrieval-selected
strict-positive 三通道权重。误差为训练/realization seed 间标准差。

{retrieval_table}

## 3. Global Top-B true/false accounting

每个 GWTC injection test catalog 有 450 events、180 true companion pairs、
100,845 false pairs，base positive rate 为 0.00178174。这里使用
candidate-selected strict-positive weights。

{topb_table}

Top-B 的结论是“富集但仍含大量假对”。例如 Top-50：

- GWTC-3/O3：平均 11 TP + 39 FP；
- GWTC-4.1/O4a：平均 8 TP + 42 FP。

## 4. 真实目录候选与 PE 独立审计

真实目录范围：

- GWTC-3：63 events / 1,953 all pairs；严格 H1L1 non-OOD BBH 为
  53 events / 1,378 pairs；
- GWTC-4.1/O4a：84 events / 3,486 all pairs；严格 subset 为
  74 events / 2,701 pairs。

{pe_table}

PE pass 表示 detector-frame chirp mass、mass ratio、chi_eff 的最大标准化
posterior distance不超过 3；它不是 lensing detection。

真实 Top-10 事件对和逐参数审计位于：

- `summary/gwtc3_real_candidate_top10_with_pe.csv`
- `summary/gwtc4_real_candidate_top10_with_pe.csv`

## 5. ET moderate-sky 主结果

- sky-only R@10 = 0.3730 +/- 0.0086；
- time+sky R@10 = 0.9065 +/- 0.0062；
- three-channel R@10 = 0.9975 +/- 0.0013；
- SIS three-channel R@10 = 0.9955；
- PM three-channel R@10 = 0.9995；
- full 40,495,500-pair AUPRC = 0.8557 +/- 0.0329。

该结果必须与 informative/conservative 及 secondary-mode sensitivity 一起
展示；不能把 moderate A90 当成真实 ET full-PE 预言。

## 6. 文件结构

- `complete_et3_gwtc34_method_20260726_cn.md`：完整方法、公式与限制；
- `et3_moderate/`：ET moderate 主结果和 sensitivity；
- `gwtc_v81/gwtc3/`：GWTC-3 注入、真实候选和 PE 结果；
- `gwtc_v81/gwtc4/`：GWTC-4.1/O4a 对应结果；
- `gwtc_topb/`：两套 GWTC Top-B per-seed 和 summary；
- `summary/`：跨部署统一表；
- `figures/fig_et3_gwtc34_complete_results.pdf`：整合图；
- `scripts/`：生成、训练、统一 sky、Top-B 和报告代码。

## 7. 论文表述边界

建议正文表述：

> Under a controlled moderate-information ET sky-localization scenario,
> validation-selected waveform, time-delay and common-source sky evidence
> yielded complementary companion retrieval. In run-matched O3 and O4a
> real-noise injection tests, the same three-channel candidate-generation
> framework enriched true companions within fixed follow-up budgets, while
> retaining non-negligible false-candidate burdens.

不要写：

- 在真实 GWTC 中检测到透镜；
- 真实 Top-10 有多少“真透镜”；
- ET moderate surrogate 等同 full Bayesian PE；
- empirical catalog rank 是 FAR 或 p-value。
"""
    (output / "complete_results_report_cn.md").write_text(
        content,
        encoding="utf-8",
    )

    english = """# Integrated ET-3 and GWTC-3/4.1 result summary

ET-3 uses the predeclared moderate-information observed-sky surrogate. The
real GWTC-3 and GWTC-4.1 catalogs continue to use their public PE HEALPix sky
posteriors. All deployments use the same common-source statistic,
`sky_score = log(N_pix * sum(P_i * P_j))`.

The GWTC Top-B TP/FP counts refer only to held-out synthetic lensed systems
injected into run-matched real off-source noise. Real catalog pairs have no
known lensing labels; their tables are candidate shortlists for Bayesian
follow-up, accompanied by an independent intrinsic-PE consistency audit.

See `complete_et3_gwtc34_method_20260726_cn.md` for the full formulas,
calibration boundaries, split definitions, results, and limitations.
"""
    (output / "complete_results_report_en.md").write_text(
        english,
        encoding="utf-8",
    )


def write_reproduce(output: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/gw-catalog
PY=/root/miniconda3/bin/python

cd "$REPO"

# ET moderate sky: run each frozen waveform seed.
for seed in 202607251 202607252 202607253 202607254 202607255; do
  "$PY" scripts/experiments/124_et3_moderate_observed_sky.py \
    --scenario moderate --seed "$seed"
done

# GWTC v8.1 sky rescore: models, splits, waveform and time calibrations remain
# frozen from v7. Run both O3 and O4a for each formal seed.
for seed in 202607241 202607242 202607243; do
  "$PY" scripts/experiments/121_gwtc_sky_v81_pipeline.py \
    --deployment all --seed "$seed"
done

# Global unordered-pair Top-B accounting for both held-out injection tests.
"$PY" scripts/experiments/127_gwtc34_real_noise_injection_topb.py

# Rebuild the integrated tables, figures, reports and inventory.
"$PY" scripts/experiments/128_build_et3_gwtc34_complete_delivery.py
"""
    path = output / "reproduce.sh"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def file_inventory(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        rows.append(
            {
                "relative_path": str(path.relative_to(output)),
                "size_bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def write_readme(output: Path, retrieval_summary: pd.DataFrame) -> None:
    payload = {
        "release_id": "et3_moderate_v82_plus_gwtc_v81_20260726",
        "status": "complete",
        "et3_sky": "moderate-information simulated observed-sky posterior",
        "gwtc_real_sky": "public PE HEALPix posterior",
        "common_sky_score": "log(N_pix * sum(P_i * P_j))",
        "primary_channels": ["waveform", "time_delay", "sky_localization"],
        "snr_in_final_score": False,
        "topb_truth_scope": "held-out real-noise injection tests only",
        "real_catalog_truth_labels_available": False,
        "retrieval_summary": retrieval_summary.to_dict(orient="records"),
    }
    write_json(output / "complete_release_summary.json", payload)

    readme = """# Current integrated result package

This directory combines:

1. ET-3 moderate-information observed-sky v8.2 results;
2. GWTC-3.0/O3 unified-sky v8.1 injection and real-catalog results;
3. GWTC-4.1/O4a unified-sky v8.1 injection and real-catalog results;
4. global Top-B TP/FP accounting for both held-out injection tests;
5. source scripts, formulas, figures, reports, and SHA-256 inventory.

Read in this order:

1. `complete_et3_gwtc34_method_20260726_cn.md`
2. `complete_results_report_cn.md`
3. `summary/et3_gwtc34_retrieval_summary.csv`
4. `gwtc_topb/gwtc34_topb_true_false_summary.csv`
5. `summary/gwtc34_real_candidate_top20_with_pe.csv`

Important: TP/FP labels exist only in synthetic injection tests. Real GWTC
catalog outputs are candidate shortlists for Bayesian follow-up, not lensing
detections.

Excluded because of size/licensing scope: raw GWOSC strain, complete public PE
source files, waveform checkpoints, and large reusable posterior banks. Their
derived pair tables, manifests/audits, metrics and code are included.
"""
    (output / "README_CURRENT_RESULTS.md").write_text(
        readme,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Regenerate summaries/reports without copying component trees.",
    )
    args = parser.parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_copy:
        copy_authoritative_results(output)
        copy_code(output)
    copy_file(
        METHOD_DOCUMENT,
        output / "complete_et3_gwtc34_method_20260726_cn.md",
    )

    retrieval_per_seed, topb_summary = build_summary_tables(output)
    retrieval_summary = summarize_retrieval(retrieval_per_seed)
    plot_complete(output, retrieval_per_seed, topb_summary)
    report_cn(output, retrieval_summary, topb_summary)
    write_reproduce(output)
    write_readme(output, retrieval_summary)

    # Inventory is generated last and intentionally excludes itself.
    inventory = file_inventory(output)
    inventory.to_csv(output / "data_and_code_inventory.csv", index=False)
    write_json(
        output / "delivery_audit.json",
        {
            "status": "complete",
            "files_before_inventory": int(len(inventory)),
            "bytes_before_inventory": int(inventory["size_bytes"].sum()),
            "required_paths_present": {
                "method_document": (
                    output / "complete_et3_gwtc34_method_20260726_cn.md"
                ).exists(),
                "et3_results": (
                    output
                    / "et3_moderate/et3/moderate/retrieval_metrics_summary_v82.csv"
                ).exists(),
                "gwtc3_results": (
                    output
                    / "gwtc_v81/gwtc3/heldout_test_retrieval_metrics_per_seed_v81.csv"
                ).exists(),
                "gwtc4_results": (
                    output
                    / "gwtc_v81/gwtc4/heldout_test_retrieval_metrics_per_seed_v81.csv"
                ).exists(),
                "gwtc34_topb": (
                    output / "gwtc_topb/gwtc34_topb_true_false_summary.csv"
                ).exists(),
                "complete_figure": (
                    output / "figures/fig_et3_gwtc34_complete_results.pdf"
                ).exists(),
            },
        },
    )


if __name__ == "__main__":
    main()
