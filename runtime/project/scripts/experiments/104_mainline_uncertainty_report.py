from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.mainline_uncertainty_common import write_json  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "results" / "mainline_uncertainty_20260713"
DEFAULT_PACKAGE = REPO_ROOT / "packages" / "mainline_uncertainty_20260713.tar.gz"


LABELS = {
    "waveform_only": "Waveform",
    "time_sky": "Validation-selected time/sky",
    "waveform_time_sky": "Validation-selected fusion",
    "time_delay_only": "Time delay",
    "unified_sky_only": "Unified sky",
}


def register_times_new_roman() -> None:
    font_root = Path("/usr/share/fonts/truetype/msttcorefonts")
    for path in font_root.glob("Times_New_Roman*.ttf"):
        font_manager.fontManager.addfont(path)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def metric_value(summary: pd.DataFrame, deployment: str, method: str, subset: str, metric: str) -> pd.Series:
    rows = summary[
        (summary["deployment"] == deployment)
        & (summary["method"] == method)
        & (summary["subset"] == subset)
        & (summary["metric"] == metric)
    ]
    if rows.empty:
        raise KeyError((deployment, method, subset, metric))
    return rows.iloc[0]


def seed_scatter(ax, values: np.ndarray, x: float, color: str, width: float = 0.08) -> None:
    jitter = np.linspace(-width, width, len(values)) if len(values) > 1 else np.asarray([0.0])
    ax.scatter(np.full(len(values), x) + jitter, values, color=color, edgecolor="white", linewidth=0.7, s=38, zorder=4)


def make_figure(root: Path) -> tuple[Path, Path]:
    register_times_new_roman()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    et_seed = pd.read_csv(require(root / "et3" / "et3_retrieval_metrics_per_seed.csv"))
    et_summary = pd.read_csv(require(root / "et3" / "et3_retrieval_metrics_across_seed_summary.csv"))
    o3_seed = pd.read_csv(require(root / "gwtc" / "gwtc3" / "heldout_test_retrieval_metrics_per_seed.csv"))
    o3_summary = pd.read_csv(require(root / "gwtc" / "gwtc3" / "heldout_test_retrieval_metrics_across_seed_summary.csv"))
    o4_seed = pd.read_csv(require(root / "gwtc" / "gwtc4" / "heldout_test_retrieval_metrics_per_seed.csv"))
    o4_summary = pd.read_csv(require(root / "gwtc" / "gwtc4" / "heldout_test_retrieval_metrics_across_seed_summary.csv"))
    pair_seed = pd.read_csv(require(root / "et3" / "et3_pair_level_metrics_per_seed.csv"))
    pair_seed = pair_seed[pair_seed["method"] == "waveform_time_sky"].copy()
    o3_stability = pd.read_csv(require(root / "gwtc" / "gwtc3" / "real_candidate_rank_stability.csv"))
    o4_stability = pd.read_csv(require(root / "gwtc" / "gwtc4" / "real_candidate_rank_stability.csv"))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    colors = {"waveform_only": "#4C78A8", "time_sky": "#72B7B2", "waveform_time_sky": "#E45756"}

    ax = axes[0, 0]
    methods = ["waveform_only", "time_sky", "waveform_time_sky"]
    width = 0.34
    for method_index, method in enumerate(methods):
        for metric_index, metric in enumerate(("r_at_1", "r_at_10")):
            x = metric_index + (method_index - 1) * width
            row = metric_value(et_summary, "ET3", method, "overall", metric)
            vals = et_seed[(et_seed.method == method) & (et_seed.subset == "overall")][metric].to_numpy()
            ax.errorbar(x, row["mean"], yerr=row["std"], fmt="o", color=colors[method], capsize=4, ms=7)
            seed_scatter(ax, vals, x, colors[method], width=0.045)
    ax.set_xticks([0, 1], ["R@1", "R@10"])
    ax.set_ylabel("Companion retrieval recall")
    ax.set_ylim(0, 1.03)
    ax.set_title("a  ET-3 test catalog (5 seeds)", loc="left")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m], markeredgecolor="none", label=LABELS[m]) for m in methods]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    deployments = [("GWTC3", o3_seed, o3_summary), ("GWTC4", o4_seed, o4_summary)]
    xbase = np.arange(2)
    for method_index, method in enumerate(methods):
        for dep_index, (deployment, seed_df, summary_df) in enumerate(deployments):
            x = dep_index + (method_index - 1) * width
            row = metric_value(summary_df, deployment, method, "overall", "r_at_10")
            vals = seed_df[(seed_df.method == method) & (seed_df.subset == "overall")]["r_at_10"].to_numpy()
            ax.errorbar(x, row["mean"], yerr=row["std"], fmt="o", color=colors[method], capsize=4, ms=7)
            seed_scatter(ax, vals, x, colors[method], width=0.045)
    ax.set_xticks(xbase, ["GWTC-3 / O3", "GWTC-4.1 / O4a"])
    ax.set_ylabel("Held-out injection R@10")
    ax.set_ylim(0, 1.03)
    ax.set_title("b  Run-matched real-noise tests (3 seeds)", loc="left")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    ap = pair_seed["average_precision"].to_numpy(dtype=float)
    auc = pair_seed["roc_auc"].to_numpy(dtype=float)
    x = np.arange(len(ap))
    ax.plot(x, ap, "o-", color="#E45756", label="AUPRC")
    ax.plot(x, auc, "s--", color="#4C78A8", label="ROC AUC")
    ax.axhline(float(pair_seed["base_positive_rate"].iloc[0]), color="black", lw=1, ls=":", label="Pair base rate")
    ax.set_xticks(x, [f"Seed {i + 1}" for i in x])
    ax.set_yscale("log")
    ax.set_ylabel("Pair-level metric (log scale)")
    ax.set_title("c  Full ET-3 unordered-pair diagnostics", loc="left")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    for offset, (label, stability, color) in enumerate(
        [("GWTC-3", o3_stability, "#4C78A8"), ("GWTC-4.1", o4_stability, "#F58518")]
    ):
        top = stability.head(5).copy().iloc[::-1]
        y = np.arange(len(top)) + offset * 0.12
        xerr = np.vstack([top["median_rank"] - top["q25_rank"], top["q75_rank"] - top["median_rank"]])
        ax.errorbar(top["median_rank"], y, xerr=xerr, fmt="o", color=color, capsize=3, label=label)
    ax.set_yticks(np.arange(5), ["5", "4", "3", "2", "1"])
    ax.set_xlabel("Median real-catalog rank across seeds")
    ax.set_ylabel("Stability-selected candidate")
    ax.set_xscale("log")
    ax.set_title("d  Real-catalog shortlist stability", loc="left")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Uncertainty of validation-selected catalog retrieval", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    pdf = fig_dir / "fig_mainline_uncertainty.pdf"
    png = fig_dir / "fig_mainline_uncertainty.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def method_label(deployment: str, method: str) -> str:
    if deployment == "ET3" and method == "waveform_time_sky":
        return "Waveform + time + sky"
    return LABELS[method]


def compact_table(summary: pd.DataFrame, deployment: str) -> list[dict[str, Any]]:
    rows = []
    for method in ("waveform_only", "time_sky", "waveform_time_sky"):
        row = {"method": method_label(deployment, method)}
        for metric in ("r_at_1", "r_at_10"):
            value = metric_value(summary, deployment, method, "overall", metric)
            row[metric] = f"{value['mean']:.4f} ± {value['std']:.4f} [{value['ci95_low_t']:.4f}, {value['ci95_high_t']:.4f}]"
        rows.append(row)
    return rows


def selected_weight_table(root: Path) -> str:
    rows: list[dict[str, Any]] = []
    deployments = [
        ("ET-3", root / "et3", False),
        ("GWTC-3/O3", root / "gwtc" / "gwtc3", True),
        ("GWTC-4.1/O4a", root / "gwtc" / "gwtc4", True),
    ]
    for deployment, base, has_results_dir in deployments:
        for seed_dir in sorted(base.glob("seed_*")):
            path = seed_dir / ("results/selected_weights.json" if has_results_dir else "selected_weights.json")
            payload = json.loads(require(path).read_text(encoding="utf-8"))
            if has_results_dir:
                weights = payload["waveform_time_sky"]
            else:
                weights = payload["waveform_time_sky"]["selected_weights"]
            rows.append(
                {
                    "deployment": deployment,
                    "seed": seed_dir.name.removeprefix("seed_"),
                    "waveform": weights.get("waveform", 0.0),
                    "time": weights.get("time", 0.0),
                    "sky": weights.get("sky", 0.0),
                    "selection": "validation only",
                }
            )
    return md_table(rows, ["deployment", "seed", "waveform", "time", "sky", "selection"])


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(root: Path, package: Path) -> None:
    et = pd.read_csv(root / "et3" / "et3_retrieval_metrics_across_seed_summary.csv")
    o3 = pd.read_csv(root / "gwtc" / "gwtc3" / "heldout_test_retrieval_metrics_across_seed_summary.csv")
    o4 = pd.read_csv(root / "gwtc" / "gwtc4" / "heldout_test_retrieval_metrics_across_seed_summary.csv")
    pair = pd.read_csv(root / "et3" / "et3_pair_level_metrics_across_seed_summary.csv")
    pair_ops = pd.read_csv(root / "et3" / "et3_pair_level_operating_points_across_seed_summary.csv")
    fixed_bootstrap = pd.read_csv(
        root / "existing_model_bootstrap" / "existing_model_system_bootstrap_95ci.csv"
    )
    seed_bootstrap = pd.concat(
        [
            pd.read_csv(root / "et3" / "et3_bootstrap_95ci_per_seed.csv"),
            pd.read_csv(root / "gwtc" / "gwtc3" / "heldout_test_bootstrap_95ci_per_seed.csv"),
            pd.read_csv(root / "gwtc" / "gwtc4" / "heldout_test_bootstrap_95ci_per_seed.csv"),
        ],
        ignore_index=True,
    )
    o3_stability = pd.read_csv(root / "gwtc" / "gwtc3" / "real_candidate_rank_stability.csv")
    o4_stability = pd.read_csv(root / "gwtc" / "gwtc4" / "real_candidate_rank_stability.csv")
    et_table = md_table(compact_table(et, "ET3"), ["method", "r_at_1", "r_at_10"])
    o3_table = md_table(compact_table(o3, "GWTC3"), ["method", "r_at_1", "r_at_10"])
    o4_table = md_table(compact_table(o4, "GWTC4"), ["method", "r_at_1", "r_at_10"])
    weights_table = selected_weight_table(root)
    pair_primary = pair[pair["method"] == "waveform_time_sky"]
    pair_rows = pair_primary[pair_primary.metric.isin(["roc_auc", "average_precision", "base_positive_rate"])][
        ["metric", "mean", "std", "ci95_low_t", "ci95_high_t"]
    ].to_dict(orient="records")
    pair_table = md_table(pair_rows, ["metric", "mean", "std", "ci95_low_t", "ci95_high_t"])
    pair_sensitivity_rows = pair[pair.metric.isin(["average_precision"])][
        ["method", "metric", "mean", "std", "ci95_low_t", "ci95_high_t"]
    ].to_dict(orient="records")
    pair_sensitivity_table = md_table(
        pair_sensitivity_rows, ["method", "metric", "mean", "std", "ci95_low_t", "ci95_high_t"]
    )
    requested = pair_ops[
        (pair_ops["method"] == "waveform_time_sky")
        & (
            ((pair_ops["operating_point"] == "recall") & pair_ops["target"].isin([0.5, 0.9]))
            | ((pair_ops["operating_point"] == "fpr") & np.isclose(pair_ops["target"], 1e-5))
        )
        & pair_ops["metric"].isin(["false_pairs", "true_pairs", "precision", "recall"])
    ]
    pair_operating_table = md_table(
        requested[["operating_point", "target", "metric", "mean", "std", "median", "q25", "q75"]]
        .to_dict(orient="records"),
        ["operating_point", "target", "metric", "mean", "std", "median", "q25", "q75"],
    )
    fixed_primary = fixed_bootstrap[
        (fixed_bootstrap["method"] == "waveform_time_sky")
        & (fixed_bootstrap["subset"] == "overall")
    ][["deployment", "metric", "point_estimate", "ci95_low", "ci95_high", "n_systems", "n_queries"]]
    fixed_bootstrap_table = md_table(
        fixed_primary.to_dict(orient="records"),
        ["deployment", "metric", "point_estimate", "ci95_low", "ci95_high", "n_systems", "n_queries"],
    )
    seed_bootstrap_primary = seed_bootstrap[
        (seed_bootstrap["method"] == "waveform_time_sky")
        & (seed_bootstrap["subset"] == "overall")
    ][["deployment", "seed", "metric", "point_estimate", "ci95_low", "ci95_high", "n_systems"]]
    seed_bootstrap_table = md_table(
        seed_bootstrap_primary.to_dict(orient="records"),
        ["deployment", "seed", "metric", "point_estimate", "ci95_low", "ci95_high", "n_systems"],
    )
    top3 = o3_stability.iloc[0].to_dict()
    top4 = o4_stability.iloc[0].to_dict()

    cn = f"""# 主流程不确定性实验报告

## 统计口径

- ET-3 使用最新 unified posterior-overlap 方案，5 个独立训练 seed。
- GWTC-3/O3 和 GWTC-4.1/O4a 使用各自 run-matched real-noise injection encoder，各 3 个 seed。
- 每个 seed 独立改变 system-level split、模型初始化、训练增强、O3/O4 real-noise realization 和 synthetic sky realization；同一双像系统的两个 directed queries 始终保留在同一 split。
- 融合权重只在该 seed 的 validation set 选择，R@1/R@10 在 held-out test 上报告。
- bootstrap 以 lensed system 为 cluster，并按 SIS/PM 分层；它反映有限测试系统波动。
- seed 间均值、标准差和 Student-t 95% CI 反映训练及 realization 波动。

## Validation-selected 融合权重

{weights_table}

ET-3 的最终方法在五个 seed 中均保留 waveform、time 和 sky。O3/O4a 的候选权重网格同样允许三个通道进入，但本次各自的三个 validation split 都把 time 权重选择为 0；因此这六个 held-out real-noise 结果的实际最终组成是 waveform + unified sky，而不是三个正权重通道。表中的 `waveform_time_sky` 是方法族内部标识，不能解释为 time 在这些 seed 中产生了正贡献。O3/O4a 的 time/sky 对照也都选择 (waveform,time,sky)=(0,0,4)，实际是 validation-selected sky-only 对照。

O3 与 O4a 使用相同的 synthetic source systems、split seeds 和 observable seeds，作为隔离 O3/O4 real-noise domain 差异的 paired control；两者分别注入不同 run 的 off-source noise 并独立训练 encoder。因此相同的 time/sky 对照数值是设计结果，不能当作两个独立 observable realization 的重复证据。

## 固定现有模型的 system-level bootstrap

该层不重新训练；它回答固定模型与固定 held-out catalog 条件下，有限测试系统带来的抽样不确定度。每个双像系统的两个 directed queries 一起重采样，SIS/PM 分层，10,000 次。

{fixed_bootstrap_table}

## 各训练 seed 的 system-level bootstrap

以下区间是在每个独立训练 seed 的 held-out test 内进行的有限系统 bootstrap；跨 seed 的 mean ± SD 见后续主表。

{seed_bootstrap_table}

## ET-3

{et_table}

ET-3 pair-level 指标来自每个 seed 的全部 40,495,500 个 unordered pairs。正文口径使用两个 directed score 的最大值；SI 同时保存均值聚合敏感性结果。没有抽样 false pairs。只保存 AUPRC/AUC、50%/90% recall 工作点和 FPR=1e-5 等固定工作点，不保存五份完整 pair 表。

{pair_table}

主 `max` 与 SI `mean` 聚合的 AUPRC 敏感性：

{pair_sensitivity_table}

正文 `max` 口径的指定工作点（跨训练 seed）：

{pair_operating_table}

## GWTC-3/O3 held-out real-noise injections

{o3_table}

这里的“validation-selected fusion”在三个 seed 中分别选择 (waveform, time, sky)=(2,0,4)、(4,0,1)、(2,0,4)。

## GWTC-4.1/O4a held-out real-noise injections

{o4_table}

这里的“validation-selected fusion”在三个 seed 中均选择 (waveform, time, sky)=(1,0,4)。O4a 的跨 seed 方差明显大于 ET-3，尤其不能把单个 seed 的结果写成稳定的 run-to-run 性能。

## 真实目录候选稳定性

- GWTC-3 最稳定的高位 pair：{top3['event_i']} -- {top3['event_j']}，median rank={top3['median_rank']:.1f}，IQR=[{top3['q25_rank']:.1f}, {top3['q75_rank']:.1f}]，top-10 frequency={top3['top10_frequency']:.2f}。
- GWTC-4.1 最稳定的高位 pair：{top4['event_i']} -- {top4['event_j']}，median rank={top4['median_rank']:.1f}，IQR=[{top4['q25_rank']:.1f}, {top4['q75_rank']:.1f}]，top-10 frequency={top4['top10_frequency']:.2f}。

真实 GWTC 目录没有 confirmed lensed positives，因此真实目录只报告 rank stability，不报告真实 recall、p-value 或 detection significance。所有高位 pair 都只是 Bayesian follow-up candidate shortlist。

## 文件

- 图：`{root / 'figures' / 'fig_mainline_uncertainty.pdf'}`
- 结果目录：`{root}`
- 打包文件：`{package}`
"""
    en = f"""# Mainline uncertainty experiment report

## Statistical design

ET-3 uses the latest unified posterior-overlap method with five independent training seeds. GWTC-3/O3 and GWTC-4.1/O4a use separate run-matched real-noise injection encoders with three seeds each. Fusion weights are selected only on each seed's validation set, while retrieval is reported on held-out tests. System-clustered, SIS/PM-stratified bootstrap intervals quantify finite-test-sample variation; across-seed Student-t intervals quantify training and realization variation.

## Validation-selected fusion weights

{weights_table}

All five ET-3 seeds retain waveform, time, and sky in the final model. The O3 and O4a grids also allow all three channels, but every one of the six real-noise validation splits selects zero time weight. The realized O3/O4a final models are therefore waveform + unified sky, not three-positive-channel fusions. The internal `waveform_time_sky` identifier denotes the candidate method family and must not be interpreted as evidence of a positive time-channel contribution in these seeds. Their time/sky controls also select (waveform,time,sky)=(0,0,4), making them validation-selected sky-only controls in practice.

O3 and O4a use matched synthetic source systems, split seeds, and observable seeds as a paired control that isolates the O3/O4 real-noise domain. Injections use different run-specific off-source noise and the encoders are trained separately. Consequently, identical time/sky control values are expected by design and are not independent observable-realization replications.

## Fixed-model system bootstrap

This layer performs no retraining. Both directed queries from a lensed doublet are resampled together, stratified by SIS/PM, for 10,000 draws.

{fixed_bootstrap_table}

## Per-training-seed system bootstrap

{seed_bootstrap_table}

## ET-3

{et_table}

The pair-level analysis uses all 40,495,500 unordered ET-3 test-catalog pairs per seed. The primary unordered-pair score is the maximum of the two directed scores; mean aggregation is retained as an SI sensitivity analysis. Full pair tables are not retained; AUC, AUPRC, recall=0.5/0.9 and fixed-FPR operating points are retained.

{pair_table}

Max-versus-mean aggregation sensitivity:

{pair_sensitivity_table}

Requested operating points under the primary max rule:

{pair_operating_table}

## GWTC-3/O3 held-out real-noise injections

{o3_table}

The three selected (waveform, time, sky) weights are (2,0,4), (4,0,1), and (2,0,4).

## GWTC-4.1/O4a held-out real-noise injections

{o4_table}

All three selected weights are (1,0,4). O4a has materially greater across-seed variation than ET-3 and should not be summarized by a single-seed estimate.

## Real-catalog rank stability

The most stable high-ranked GWTC-3 pair is {top3['event_i']} -- {top3['event_j']} (median rank {top3['median_rank']:.1f}, IQR {top3['q25_rank']:.1f}-{top3['q75_rank']:.1f}). The corresponding GWTC-4.1 pair is {top4['event_i']} -- {top4['event_j']} (median rank {top4['median_rank']:.1f}, IQR {top4['q25_rank']:.1f}-{top4['q75_rank']:.1f}).

The real catalogs contain no confirmed lensed positives. These are candidate shortlists for Bayesian follow-up, not detections, p-values, or empirical real-catalog recall measurements.
"""
    (root / "mainline_uncertainty_report_cn.md").write_text(cn, encoding="utf-8")
    (root / "mainline_uncertainty_report_en.md").write_text(en, encoding="utf-8")


def write_reproduce(root: Path) -> Path:
    path = root / "reproduce.sh"
    text = """#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gw-catalog
/root/miniconda3/bin/python scripts/experiments/103_et3_unified_sky_uncertainty.py
/root/miniconda3/bin/python scripts/experiments/106_existing_mainline_system_bootstrap.py
/root/miniconda3/bin/python scripts/real_search/16_gwtc_unified_sky_uncertainty.py
/root/miniconda3/bin/python scripts/experiments/104_mainline_uncertainty_report.py
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def package_outputs(root: Path, package: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    scripts = [
        REPO_ROOT / "scripts" / "experiments" / "mainline_uncertainty_common.py",
        REPO_ROOT / "scripts" / "experiments" / "103_et3_unified_sky_uncertainty.py",
        REPO_ROOT / "scripts" / "real_search" / "16_gwtc_unified_sky_uncertainty.py",
        REPO_ROOT / "scripts" / "experiments" / "104_mainline_uncertainty_report.py",
        REPO_ROOT / "scripts" / "experiments" / "106_existing_mainline_system_bootstrap.py",
        REPO_ROOT / "scripts" / "experiments" / "run_mainline_uncertainty_20260713.sh",
        REPO_ROOT / "docs" / "methods" / "mainline_uncertainty_protocol_20260713_cn.md",
    ]
    excluded_parts = {"matchroots", "__pycache__"}
    excluded_names = {"val_scores.npy", "test_scores.npy", "val_embeddings.npy", "test_embeddings.npy"}
    with tarfile.open(package, "w:gz") as tar:
        for script in scripts:
            tar.add(script, arcname=str(script.relative_to(REPO_ROOT)))
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root)
            if any(part in excluded_parts for part in rel.parts) or path.name in excluded_names:
                continue
            tar.add(path, arcname=str(Path("results/mainline_uncertainty_20260713") / rel))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    pdf, png = make_figure(args.root)
    write_reproduce(args.root)
    write_reports(args.root, args.package)
    summary = {
        "status": "complete",
        "root": str(args.root),
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "report_cn": str(args.root / "mainline_uncertainty_report_cn.md"),
        "report_en": str(args.root / "mainline_uncertainty_report_en.md"),
        "package": str(args.package),
    }
    write_json(args.root / "mainline_uncertainty_delivery_summary.json", summary)
    package_outputs(args.root, args.package)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
