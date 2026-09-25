#!/usr/bin/env python3
"""Summarize the frozen ET-3/GWTC v7-aligned experiments.

This script does not train, score, or select any model.  It reads the already
frozen ET formal runs, GWTC v7 runs, and the matched-SNR waveform control, then
creates comparison tables, a paper-style diagnostic figure, and a Chinese
results report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ET = REPO / "results/et3_v7_aligned_20260723"
DEFAULT_GWTC = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
DEFAULT_MATCHED = REPO / "results/et_gwtc_snr_matched_control_20260723"
DEFAULT_OUT = REPO / "results/et3_gwtc_unified_v7_comparison_20260723"
RAW_ET = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")

METHOD_ORDER = (
    "waveform",
    "time",
    "sky",
    "time+sky",
    "three-channel",
)
COLORS = {
    "waveform": "#2674A6",
    "time": "#C44E52",
    "sky": "#4C956C",
    "time+sky": "#8A6BBE",
    "three-channel": "#D18424",
}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def normalized_retrieval(et_root: Path, gwtc_root: Path) -> pd.DataFrame:
    et = pd.read_csv(require(et_root / "formal/et3_retrieval_metrics_per_seed.csv"))
    et = et[et["subset"].astype(str).str.lower() == "overall"].copy()
    et_methods = {
        "waveform_only": "waveform",
        "time_only": "time",
        "sky_only": "sky",
        "time_sky_validation_selected": "time+sky",
        "three_channel_strict_positive": "three-channel",
    }
    et = et[et.method.isin(et_methods)].copy()
    et["method_display"] = et.method.map(et_methods)
    et["deployment_display"] = "ET-3"
    frames = [et]
    gwtc_methods = {
        "waveform_only": "waveform",
        "time_only": "time",
        "sky_bayes_factor_only": "sky",
        "time_sky_retrieval_selected": "time+sky",
        "retrieval_three_channel_strictly_positive": "three-channel",
    }
    for deployment, display in (("gwtc3", "GWTC-3/O3"), ("gwtc4", "GWTC-4.1/O4a")):
        frame = pd.read_csv(
            require(
                gwtc_root
                / deployment
                / "heldout_test_retrieval_metrics_per_seed_v7.csv"
            )
        )
        frame = frame[
            (frame["subset"].astype(str).str.lower() == "overall")
            & frame.method.isin(gwtc_methods)
        ].copy()
        frame["method_display"] = frame.method.map(gwtc_methods)
        frame["deployment_display"] = display
        frames.append(frame)
    columns = [
        "deployment_display",
        "seed",
        "method",
        "method_display",
        "r_at_1",
        "r_at_5",
        "r_at_10",
        "median_rank",
        "n_queries",
    ]
    return pd.concat([frame[columns] for frame in frames], ignore_index=True)


def retrieval_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in per_seed.groupby(
        ["deployment_display", "method_display"], sort=False
    ):
        for metric in ("r_at_1", "r_at_10", "median_rank"):
            values = frame[metric].to_numpy(np.float64)
            rows.append(
                {
                    "deployment": key[0],
                    "method": key[1],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def pair_min_snr_samples(gwtc_root: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    et_values = []
    for family in ("SIS", "PM"):
        root = RAW_ET / f"{family}_data_0222"
        first = np.load(
            root / f"{family}_optimal_SNR_network_1.npy", mmap_mode="r"
        )
        second = np.load(
            root / f"{family}_optimal_SNR_network_2.npy", mmap_mode="r"
        )
        et_values.append(np.minimum(np.asarray(first), np.asarray(second)))
    values["ET-3 original"] = np.concatenate(et_values).astype(np.float64)
    for deployment, label in (("gwtc3", "O3 injection target"), ("gwtc4", "O4a injection target")):
        path = (
            gwtc_root
            / deployment
            / "seed_202607241/data/real_noise_injections/compact_injection_metadata.parquet"
        )
        frame = pd.read_parquet(require(path))
        frame = frame[frame.family.isin(["SIS", "PM"])]
        values[label] = np.minimum(
            frame.target_snr_image1.to_numpy(np.float64),
            frame.target_snr_image2.to_numpy(np.float64),
        )
    return values


def snr_summary(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, values in samples.items():
        rows.append(
            {
                "sample": name,
                "n_systems": len(values),
                "min": float(np.min(values)),
                "q10": float(np.quantile(values, 0.10)),
                "median": float(np.median(values)),
                "q90": float(np.quantile(values, 0.90)),
                "fraction_lt_8": float(np.mean(values < 8)),
                "fraction_8_12": float(np.mean((values >= 8) & (values < 12))),
                "fraction_12_20": float(np.mean((values >= 12) & (values < 20))),
                "fraction_ge_20": float(np.mean(values >= 20)),
            }
        )
    return pd.DataFrame(rows)


def matched_rows(matched_root: Path) -> pd.DataFrame:
    return pd.read_csv(
        require(
            matched_root
            / "et_o3_o4_snr_matched_waveform_comparison_per_seed.csv"
        )
    )


def matched_summary(rows: pd.DataFrame) -> pd.DataFrame:
    overall = rows[
        (rows["subset"].astype(str).str.lower() == "overall")
        & rows["r_at_10"].notna()
    ]
    records = []
    for deployment, frame in overall.groupby("deployment", sort=False):
        values = frame["r_at_10"].to_numpy(np.float64)
        records.append(
            {
                "deployment": deployment,
                "n_seeds": len(values),
                "waveform_r_at_10_mean": float(values.mean()),
                "waveform_r_at_10_std": (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ),
                "waveform_r_at_10_median": float(np.median(values)),
            }
        )
    return pd.DataFrame(records)


def nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.labelweight": "bold",
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(
    development: pd.DataFrame,
    retrieval: pd.DataFrame,
    snr_values: dict[str, np.ndarray],
    matched: pd.DataFrame,
    output: Path,
) -> None:
    nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.7), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(development))
    width = 0.34
    ax.bar(
        x - width / 2,
        development.waveform_r_at_1,
        width,
        color="#9CC9E2",
        edgecolor="black",
        linewidth=0.5,
        label="Waveform R@1",
    )
    ax.bar(
        x + width / 2,
        development.waveform_r_at_10,
        width,
        color="#2674A6",
        edgecolor="black",
        linewidth=0.5,
        label="Waveform R@10",
    )
    ax.set_xticks(x, development.preprocessing)
    ax.set_ylabel("Validation recall\n(truncated axis)")
    ax.set_ylim(0.48, 0.84)
    ax.set_title("a  Frozen preprocessing selection", loc="left")
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[0, 1]
    deployments = ("ET-3", "GWTC-3/O3", "GWTC-4.1/O4a")
    base_x = np.arange(len(deployments))
    offsets = np.linspace(-0.28, 0.28, len(METHOD_ORDER))
    for offset, method in zip(offsets, METHOD_ORDER):
        means = []
        stds = []
        for deployment in deployments:
            values = retrieval[
                (retrieval.deployment_display == deployment)
                & (retrieval.method_display == method)
            ].r_at_10.to_numpy(np.float64)
            means.append(values.mean())
            stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
            jitter = np.linspace(-0.018, 0.018, len(values))
            ax.scatter(
                np.full(len(values), base_x[deployments.index(deployment)] + offset)
                + jitter,
                values,
                s=10,
                color=COLORS[method],
                alpha=0.75,
                zorder=3,
            )
        ax.errorbar(
            base_x + offset,
            means,
            yerr=stds,
            fmt="o",
            ms=4,
            capsize=2,
            color=COLORS[method],
            label=method,
        )
    ax.set_xticks(base_x, deployments, rotation=12, ha="right")
    ax.set_ylabel("Held-out test R@10")
    ax.set_ylim(0, 1.03)
    ax.set_title("b  Unified evidence channels", loc="left")
    ax.legend(frameon=False, ncol=2, loc="lower left")

    ax = axes[1, 0]
    snr_colors = ("#222222", "#2674A6", "#C44E52")
    for (name, values), color in zip(snr_values.items(), snr_colors):
        ordered = np.sort(values)
        y = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, y, lw=1.5, color=color, label=name)
    ax.axvline(8, color="0.55", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(3, max(300, max(float(np.quantile(v, 0.995)) for v in snr_values.values())))
    ax.set_xlabel("Minimum image network SNR")
    ax.set_ylabel("System ECDF")
    ax.set_title("c  Original SNR-domain mismatch", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    keep = matched[
        (matched["subset"].astype(str).str.lower() == "overall")
        & matched["r_at_10"].notna()
    ].copy()
    labels = [
        "ET at O3 SNR",
        "O3 v7",
        "ET at O4a SNR",
        "O4a v7",
    ]
    deployment_map = {
        "ET_GWTC3_SNR_MATCHED": "ET at O3 SNR",
        "GWTC3": "O3 v7",
        "ET_GWTC4_SNR_MATCHED": "ET at O4a SNR",
        "GWTC4": "O4a v7",
    }
    keep["display"] = keep.deployment.map(deployment_map)
    for index, label in enumerate(labels):
        values = keep.loc[keep.display == label, "r_at_10"].to_numpy(np.float64)
        jitter = np.linspace(-0.06, 0.06, len(values))
        color = "#2674A6" if label.startswith("ET") else "#C44E52"
        ax.scatter(np.full(len(values), index) + jitter, values, s=20, color=color)
        ax.errorbar(
            index,
            values.mean(),
            yerr=values.std(ddof=1) if len(values) > 1 else 0,
            fmt="_",
            ms=16,
            capsize=3,
            color="black",
        )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("Waveform-only R@10\n(truncated axis)")
    ax.set_ylim(0.12, 0.38)
    ax.set_title("d  Matched-SNR control", loc="left")

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "fig_et3_gwtc_unified_v7_comparison.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_et3_gwtc_unified_v7_comparison.png", bbox_inches="tight")
    plt.close(fig)


def metric_value(summary: pd.DataFrame, deployment: str, method: str, metric: str) -> tuple[float, float]:
    row = summary[
        (summary.deployment == deployment)
        & (summary.method == method)
        & (summary.metric == metric)
    ].iloc[0]
    return float(row["mean"]), float(row["std"])


def make_report(
    output: Path,
    selected_preprocessing: dict,
    summary: pd.DataFrame,
    snr: pd.DataFrame,
    matched_overall: pd.DataFrame,
    et_retrieval_across: pd.DataFrame,
    et_pair_across: pd.DataFrame,
    et_operating: pd.DataFrame,
    et_weights: pd.DataFrame,
) -> None:
    lines = [
        "# ET-3 / GWTC 统一 v7 方法重跑报告",
        "",
        "## 结论边界",
        "",
        "本报告比较的是同一套 waveform + time-delay + common-source sky "
        "evidence 框架。ET-3、O3 和 O4a 使用各自 domain-matched encoder；"
        "这不是共享 checkpoint 实验，也不是对真实透镜的发现声明。",
        "",
        "## 冻结的预处理",
        "",
        f"- validation-only 选择：`{selected_preprocessing['selected_preprocessing']}`；",
        "- 两种候选均使用 [GPS-1.75 s, GPS+0.25 s]、2048 Hz、4096 点、"
        "物理 40--580 Hz；",
        "- 正式五种子运行前已冻结，test 未参与选择。",
        "",
        "## Held-out test 主要结果",
        "",
        "| 部署 | 方法 | R@1 mean ± SD | R@10 mean ± SD |",
        "|---|---|---:|---:|",
    ]
    for deployment in ("ET-3", "GWTC-3/O3", "GWTC-4.1/O4a"):
        for method in METHOD_ORDER:
            r1 = metric_value(summary, deployment, method, "r_at_1")
            r10 = metric_value(summary, deployment, method, "r_at_10")
            lines.append(
                f"| {deployment} | {method} | {r1[0]:.4f} ± {r1[1]:.4f} | "
                f"{r10[0]:.4f} ± {r10[1]:.4f} |"
            )
    lines.extend(
        [
            "",
            "ET 与 GWTC 的绝对结果不能只归因于 detector generation：原 ET "
            "目录的 network SNR 明显更高，且 ET、H1/L1 的响应与噪声域不同。",
            "",
            "## SNR 分布审计",
            "",
            snr.to_markdown(index=False, floatfmt=".4g"),
            "",
            "## Matched-SNR waveform control",
            "",
            "该控制把 ET clean detector strain 缩放到每个 O3/O4a v7 seed "
            "实际使用的目标 SNR，并加入独立 ET whitened noise。它隔离 SNR "
            "影响，但不消除 source population、detector response 和真实非平稳 "
            "H1/L1 noise 的差异。",
            "",
            matched_overall.to_markdown(index=False, floatfmt=".4g"),
            "",
            "逐 seed、SIS 和 PM 明细保存在 "
            "`matched_snr_waveform_comparison_per_seed.csv`。",
            "",
            "## ET-3 分族、不确定度与 pair-level 负担",
            "",
            et_retrieval_across[
                (et_retrieval_across["method"] == "three_channel_strict_positive")
                & et_retrieval_across["metric"].isin(["r_at_1", "r_at_10"])
            ][
                [
                    "subset",
                    "metric",
                    "n_seeds",
                    "mean",
                    "std",
                    "ci95_low_t",
                    "ci95_high_t",
                ]
            ].to_markdown(index=False, floatfmt=".5g"),
            "",
            et_pair_across[
                (et_pair_across["method"] == "three_channel_strict_positive")
                & et_pair_across["metric"].isin(["roc_auc", "average_precision"])
            ][["metric", "n_seeds", "mean", "std", "median", "q25", "q75"]]
            .to_markdown(index=False, floatfmt=".5g"),
            "",
            "在完整 40,495,500 个 unordered pairs 上，固定 FPR=1e-5 的逐 seed "
            "误配负担如下：",
            "",
            et_operating[
                (et_operating["method"] == "three_channel_strict_positive")
                & (et_operating["operating_point"] == "fpr")
                & np.isclose(et_operating["target"], 1e-5)
            ][
                [
                    "seed",
                    "false_pairs",
                    "true_pairs",
                    "recall",
                    "precision",
                    "sis_recall",
                    "pm_recall",
                ]
            ]
            .assign(seed=lambda frame: frame["seed"].astype("int64").astype(str))
            .to_markdown(index=False, floatfmt=".5g"),
            "",
            "五个正式 seed 的 validation-selected 权重汇总：",
            "",
            et_weights[
                et_weights["method"].isin(
                    [
                        "time_sky_validation_selected",
                        "three_channel_strict_positive",
                        "three_channel_unconstrained",
                    ]
                )
            ].to_markdown(index=False, floatfmt=".5g"),
            "",
            "## z-score 与总体信息的使用边界",
            "",
            "1. 单事件：每个 detector channel 的 4096 点独立做均值/方差归一化；"
            "不读取其他事件。",
            "2. 事件对：embedding cosine、waveform-head 参数差、时间差和 sky "
            "Bayes factor 只读取当前两个事件。",
            "3. validation population：waveform 特征的全局中心/尺度、waveform "
            "likelihood-ratio 和融合权重由全部 validation systems/pairs 冻结。",
            "4. 外部 population/calendar：time score 使用 GW-LMC lensed-delay "
            "population 和对应部署的冻结观测日历/null population。",
            "5. test/real catalog：只用于把冻结分数排序并计算 R@K、PR/AUPRC；"
            "不重新计算 z 标尺、time lookup 或融合权重。",
            "6. 最终融合没有 query-wise row z-score。三个 evidence 均为对称 "
            "pair score，因此 unordered max 与 mean 在数值上相同。",
            "",
            "完整公式与依据见 "
            "`docs/methods/et3_gwtc_unified_v7_method_20260723_cn.md`。",
        ]
    )
    (output / "et3_gwtc_unified_v7_results_report_cn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def information_scope_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "operation": "per-channel waveform z normalization",
                "single_event": True,
                "event_pair": False,
                "validation_or_population": False,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "mean/std from the 4096 samples of one event and one detector channel",
            },
            {
                "operation": "embedding and waveform-head inference",
                "single_event": True,
                "event_pair": False,
                "validation_or_population": False,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "one event enters the frozen encoder",
            },
            {
                "operation": "waveform raw pair features",
                "single_event": False,
                "event_pair": True,
                "validation_or_population": False,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "embedding cosine and waveform-derived intrinsic differences for i,j",
            },
            {
                "operation": "waveform feature z calibration",
                "single_event": False,
                "event_pair": False,
                "validation_or_population": True,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "one global center/scale per feature from validation pairs; never row-wise",
            },
            {
                "operation": "waveform log-likelihood ratio",
                "single_event": False,
                "event_pair": True,
                "validation_or_population": True,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "pair composite evaluated in validation-fitted true/false score densities",
            },
            {
                "operation": "raw time difference",
                "single_event": False,
                "event_pair": True,
                "validation_or_population": False,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "absolute GPS difference of i,j",
            },
            {
                "operation": "time-delay log-likelihood ratio",
                "single_event": False,
                "event_pair": True,
                "validation_or_population": True,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "GW-LMC lensed-delay population plus deployment-specific frozen calendar/null",
            },
            {
                "operation": "common-source sky Bayes factor",
                "single_event": False,
                "event_pair": True,
                "validation_or_population": True,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "two event posteriors plus frozen isotropic sky prior; no catalog row z-score",
            },
            {
                "operation": "fusion weight selection",
                "single_event": False,
                "event_pair": False,
                "validation_or_population": True,
                "test_or_real_catalog": False,
                "frozen_before_test": True,
                "detail": "held-out validation systems only",
            },
            {
                "operation": "query ranking and pair PR/AUPRC",
                "single_event": False,
                "event_pair": True,
                "validation_or_population": False,
                "test_or_real_catalog": True,
                "frozen_before_test": False,
                "detail": "whole catalog is used only to compare frozen pair scores for evaluation",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--et-root", type=Path, default=DEFAULT_ET)
    parser.add_argument("--gwtc-root", type=Path, default=DEFAULT_GWTC)
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(
        require(
            args.et_root
            / "development/preprocessing_comparison_validation_only.csv"
        )
    )
    frozen = json.loads(
        require(
            args.et_root / "development/frozen_preprocessing_selection.json"
        ).read_text(encoding="utf-8")
    )
    retrieval = normalized_retrieval(args.et_root, args.gwtc_root)
    summary = retrieval_summary(retrieval)
    values = pair_min_snr_samples(args.gwtc_root)
    snr = snr_summary(values)
    matched = matched_rows(args.matched_root)
    matched_overall = matched_summary(matched)
    formal = args.et_root / "formal"
    et_retrieval_across = pd.read_csv(
        require(formal / "et3_retrieval_metrics_across_seed_summary.csv")
    )
    et_pair_across = pd.read_csv(
        require(formal / "et3_pair_level_metrics_across_seed_summary.csv")
    )
    et_operating = pd.read_csv(
        require(formal / "et3_pair_level_operating_points_per_seed.csv")
    )
    et_weights = pd.read_csv(
        require(formal / "et3_selected_weights_summary.csv")
    )

    development.to_csv(
        args.out_root / "preprocessing_comparison_validation_only.csv", index=False
    )
    retrieval.to_csv(
        args.out_root / "unified_retrieval_metrics_per_seed.csv", index=False
    )
    summary.to_csv(
        args.out_root / "unified_retrieval_metrics_summary.csv", index=False
    )
    snr.to_csv(args.out_root / "snr_distribution_summary.csv", index=False)
    matched.to_csv(
        args.out_root / "matched_snr_waveform_comparison_per_seed.csv", index=False
    )
    matched_overall.to_csv(
        args.out_root / "matched_snr_waveform_comparison_summary.csv", index=False
    )
    et_retrieval_across.to_csv(
        args.out_root / "et3_retrieval_metrics_across_seed_summary.csv", index=False
    )
    et_pair_across.to_csv(
        args.out_root / "et3_pair_level_metrics_across_seed_summary.csv", index=False
    )
    et_operating.to_csv(
        args.out_root / "et3_pair_level_operating_points_per_seed.csv", index=False
    )
    et_weights.to_csv(
        args.out_root / "et3_selected_weights_summary.csv", index=False
    )
    information_scope_table().to_csv(
        args.out_root / "score_information_scope.csv", index=False
    )
    plot_figure(development, retrieval, values, matched, args.out_root / "figures")
    make_report(
        args.out_root,
        frozen,
        summary,
        snr,
        matched_overall,
        et_retrieval_across,
        et_pair_across,
        et_operating,
        et_weights,
    )
    save_json(
        args.out_root / "summary_manifest.json",
        {
            "status": "complete",
            "et_root": str(args.et_root),
            "gwtc_root": str(args.gwtc_root),
            "matched_snr_root": str(args.matched_root),
            "selected_preprocessing": frozen["selected_preprocessing"],
            "primary_comparison_method": "validation retrieval-selected strict-positive waveform/time/sky evidence",
            "final_row_standardization": False,
            "test_or_real_catalog_used_for_calibration": False,
        },
    )


if __name__ == "__main__":
    main()
