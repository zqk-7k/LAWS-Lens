#!/usr/bin/env python3
"""Build figures, summaries, reports, and reproduction metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/et3_moderate_sky_gwtc3_topb_20260726"
V81_ROOT = REPO / "results/unified_sky_v81_20260725"

METHOD_LABELS = {
    "waveform_only": "Waveform",
    "time_only": "Time delay",
    "sky_only": "Observed sky",
    "time_sky_validation_selected": "Time + sky",
    "three_channel_strict_positive": "Three channel",
    "three_channel_equal_evidence": "Equal evidence",
}
SCENARIO_LABELS = {
    "informative": "Informative",
    "moderate": "Moderate",
    "conservative": "Conservative",
    "moderate_secondary25": "Moderate\n+25% secondary",
    "moderate_secondary50": "Moderate\n+50% secondary",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


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
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def scenario_summary(root: Path) -> pd.DataFrame:
    rows = []
    for scenario_root in sorted((root / "et3").glob("*")):
        path = scenario_root / "retrieval_metrics_summary_v82.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame[frame["subset"] == "overall"].iterrows():
            rows.append(
                {
                    "scenario": scenario_root.name,
                    "method": row["method"],
                    "n_seeds": int(row["n_seeds"]),
                    "r_at_1_mean": float(row["r_at_1_mean"]),
                    "r_at_1_std": float(row["r_at_1_std"])
                    if pd.notna(row["r_at_1_std"])
                    else np.nan,
                    "r_at_10_mean": float(row["r_at_10_mean"]),
                    "r_at_10_std": float(row["r_at_10_std"])
                    if pd.notna(row["r_at_10_std"])
                    else np.nan,
                    "median_rank_mean": float(row["median_rank_mean"]),
                }
            )
    return pd.DataFrame(rows)


def v81_comparison(root: Path) -> pd.DataFrame:
    current = pd.read_csv(
        root / "et3/moderate/retrieval_metrics_summary_v82.csv"
    )
    old = pd.read_csv(V81_ROOT / "et3/et3_retrieval_metrics_across_seed_v81.csv")
    methods = (
        "waveform_only",
        "time_only",
        "sky_only",
        "time_sky_validation_selected",
        "three_channel_strict_positive",
    )
    rows = []
    for method in methods:
        current_row = current[
            (current["method"] == method) & (current["subset"] == "overall")
        ].iloc[0]
        old_method = (
            "time_sky_validation_selected"
            if method == "time_sky_validation_selected"
            else method
        )
        old_row = old[
            (old["method"] == old_method) & (old["subset"] == "overall")
        ].iloc[0]
        rows.append(
            {
                "method": method,
                "v81_r_at_1": float(old_row["r_at_1_mean"]),
                "v81_r_at_10": float(old_row["r_at_10_mean"]),
                "moderate_r_at_1": float(current_row["r_at_1_mean"]),
                "moderate_r_at_10": float(current_row["r_at_10_mean"]),
            }
        )
    return pd.DataFrame(rows)


def plot_main(root: Path, scenario: pd.DataFrame, topb: pd.DataFrame) -> None:
    style()
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    moderate_per_seed = pd.read_csv(
        root / "et3/moderate/retrieval_metrics_per_seed_v82.csv"
    )
    moderate_summary = pd.read_csv(
        root / "et3/moderate/retrieval_metrics_summary_v82.csv"
    )
    primary_method = "candidate_three_channel_strict_positive"
    topb_primary = topb[
        (topb["method"] == primary_method)
        & (topb["budget_type"] == "fixed_count")
    ].copy()
    budgets = sorted(topb_primary["budget_pairs"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1))
    colors = {
        "waveform_only": "#35618d",
        "time_only": "#d58b2a",
        "sky_only": "#4b8b57",
        "time_sky_validation_selected": "#9670b4",
        "three_channel_strict_positive": "#b24745",
    }

    methods = list(colors)
    summary_rows = moderate_summary[
        (moderate_summary["subset"] == "overall")
        & (moderate_summary["method"].isin(methods))
    ].set_index("method")
    x = np.arange(len(methods))
    width = 0.34
    axes[0, 0].bar(
        x - width / 2,
        [summary_rows.loc[m, "r_at_1_mean"] for m in methods],
        width,
        color=[colors[m] for m in methods],
        alpha=0.65,
        label="R@1",
    )
    axes[0, 0].bar(
        x + width / 2,
        [summary_rows.loc[m, "r_at_10_mean"] for m in methods],
        width,
        color=[colors[m] for m in methods],
        alpha=1.0,
        label="R@10",
    )
    for index, method in enumerate(methods):
        values = moderate_per_seed[
            (moderate_per_seed["subset"] == "overall")
            & (moderate_per_seed["method"] == method)
        ]["r_at_10"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[0, 0].scatter(
            index + width / 2 + jitter,
            values,
            s=12,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(
        [METHOD_LABELS[m] for m in methods],
        rotation=25,
        ha="right",
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("Companion recall")
    axes[0, 0].set_title("a  ET-3 moderate observed-sky")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")

    scenario_order = [
        "informative",
        "moderate",
        "conservative",
        "moderate_secondary25",
        "moderate_secondary50",
    ]
    sky_rows = scenario[
        (scenario["method"] == "sky_only")
        & (scenario["scenario"].isin(scenario_order))
    ].set_index("scenario")
    values = [sky_rows.loc[item, "r_at_10_mean"] for item in scenario_order]
    axes[0, 1].bar(
        np.arange(len(values)),
        values,
        color=["#4b8b57", "#3d7866", "#73878c", "#7084b2", "#88739d"],
        width=0.68,
    )
    axes[0, 1].axhline(
        10.0 / 8999.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="Random",
    )
    axes[0, 1].set_xticks(np.arange(len(values)))
    axes[0, 1].set_xticklabels(
        [SCENARIO_LABELS[item] for item in scenario_order],
        rotation=24,
        ha="right",
    )
    axes[0, 1].set_ylabel("Sky-only R@10")
    axes[0, 1].set_ylim(0.0, 0.66)
    axes[0, 1].set_title("b  Prespecified sky sensitivity")
    axes[0, 1].legend(frameon=False, loc="upper right")

    for seed, part in topb_primary.groupby("seed"):
        part = part.set_index("budget_pairs").loc[budgets].reset_index()
        axes[1, 0].plot(
            budgets,
            part["precision"],
            color="#b24745",
            alpha=0.35,
            marker="o",
            markersize=3,
            linewidth=0.8,
        )
        axes[1, 0].scatter(
            budgets,
            part["precision"],
            color="#b24745",
            s=14,
            alpha=0.8,
        )
    mean_precision = (
        topb_primary.groupby("budget_pairs")["precision"].mean().loc[budgets]
    )
    axes[1, 0].plot(
        budgets,
        mean_precision,
        color="#7e2628",
        marker="o",
        linewidth=1.8,
        label="Mean",
    )
    axes[1, 0].axhline(
        topb_primary["base_positive_rate"].iloc[0],
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="Catalog base rate",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Global shortlist budget B")
    axes[1, 0].set_ylabel("Precision@B")
    axes[1, 0].set_title("c  O3 real-noise injection shortlist")
    axes[1, 0].legend(frameon=False, loc="upper right")

    counts = (
        topb_primary.groupby("budget_pairs")[
            ["true_recovered", "false_candidates"]
        ]
        .mean()
        .loc[budgets]
    )
    axes[1, 1].bar(
        np.arange(len(budgets)),
        counts["true_recovered"],
        color="#4b8b57",
        label="True injected pairs",
    )
    axes[1, 1].bar(
        np.arange(len(budgets)),
        counts["false_candidates"],
        bottom=counts["true_recovered"],
        color="#c9c9c9",
        label="False pairs",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(np.arange(len(budgets)))
    axes[1, 1].set_xticklabels([str(value) for value in budgets])
    axes[1, 1].set_xlabel("Global shortlist budget B")
    axes[1, 1].set_ylabel("Mean candidate count")
    axes[1, 1].set_title("d  True and false candidate burden")
    axes[1, 1].legend(frameon=False, loc="upper left")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(figures / "fig_et3_moderate_sky_gwtc3_topb.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig_et3_moderate_sky_gwtc3_topb.png", bbox_inches="tight")
    plt.close(fig)


def plot_sky_distribution(root: Path) -> None:
    style()
    frames = []
    for path in sorted(
        (root / "et3/moderate").glob("seed_*/pair_diagnostics_sample_v82.parquet")
    ):
        seed = int(path.parent.name.split("_")[-1])
        frame = pd.read_parquet(
            path,
            columns=["is_true_pair", "true_pair_family", "sky_score"],
        )
        frame["seed"] = seed
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bins = np.linspace(-20.0, 8.0, 100)
    false = data[data["is_true_pair"] == 0]["sky_score"].clip(-20, 8)
    sis = data[
        (data["is_true_pair"] == 1) & (data["true_pair_family"] == "SIS")
    ]["sky_score"].clip(-20, 8)
    pm = data[
        (data["is_true_pair"] == 1) & (data["true_pair_family"] == "PM")
    ]["sky_score"].clip(-20, 8)
    axes[0].hist(false, bins=bins, density=True, color="#bdbdbd", alpha=0.7, label="False")
    axes[0].hist(sis, bins=bins, density=True, histtype="step", linewidth=1.5, color="#35618d", label="SIS true")
    axes[0].hist(pm, bins=bins, density=True, histtype="step", linewidth=1.5, color="#d58b2a", label="PM true")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Sky evidence $\log B_{\rm sky}$ (clipped at -20)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("a  Pair-level sky evidence")
    axes[0].legend(frameon=False)

    audits = pd.read_csv(root / "et3/moderate/sky_audit_per_seed_v82.csv")
    axes[1].scatter(
        audits["test_coverage"],
        audits["roc_auc"],
        color="#3d7866",
        s=32,
    )
    for _, row in audits.iterrows():
        axes[1].annotate(
            str(int(row["seed"]))[-2:],
            (row["test_coverage"], row["roc_auc"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].axvline(0.9, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Held-out true-sky 90% coverage")
    axes[1].set_ylabel("Validation pair ROC AUC")
    axes[1].set_title("b  Coverage and separability audit")
    axes[1].set_xlim(0.885, 0.905)
    axes[1].set_ylim(0.995, 1.0)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    fig.tight_layout()
    figures = root / "figures"
    fig.savefig(figures / "fig_et3_moderate_sky_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig_et3_moderate_sky_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def sky_geometry_audit(root: Path) -> None:
    frames = []
    for path in sorted(
        (root / "et3/moderate").glob("seed_*/pair_diagnostics_sample_v82.parquet")
    ):
        seed = int(path.parent.name.split("_")[-1])
        pairs = pd.read_parquet(
            path,
            columns=["idx_i", "idx_j", "is_true_pair", "true_pair_family"],
        )
        events = pd.read_parquet(
            path.parent / "test_event_sky_diagnostics_v82.parquet",
            columns=[
                "ra_obs",
                "dec_obs",
                "sigma_major_rad",
                "sigma_minor_rad",
                "posterior_temperature",
            ],
        )
        i = pairs["idx_i"].to_numpy(dtype=np.int32)
        j = pairs["idx_j"].to_numpy(dtype=np.int32)
        ra = events["ra_obs"].to_numpy(dtype=np.float64)
        dec = events["dec_obs"].to_numpy(dtype=np.float64)
        cosine = (
            np.sin(dec[i]) * np.sin(dec[j])
            + np.cos(dec[i]) * np.cos(dec[j]) * np.cos(ra[i] - ra[j])
        )
        separation = np.arccos(np.clip(cosine, -1.0, 1.0))
        sigma = np.sqrt(
            events["sigma_major_rad"].to_numpy(dtype=np.float64)
            * events["sigma_minor_rad"].to_numpy(dtype=np.float64)
            * events["posterior_temperature"].to_numpy(dtype=np.float64)
        )
        normalized = separation / np.sqrt(sigma[i] ** 2 + sigma[j] ** 2)
        pairs["seed"] = seed
        pairs["observed_center_separation_deg"] = np.degrees(separation)
        pairs["normalized_center_separation"] = normalized
        frames.append(pairs)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(
        root / "et3_moderate_sky_pair_geometry_sample.parquet",
        index=False,
    )
    rows = []
    for (seed, label), part in data.groupby(["seed", "is_true_pair"]):
        rows.append(
            {
                "seed": int(seed),
                "pair_class": "true_companion" if label else "sampled_false",
                "n_pairs": int(len(part)),
                "separation_median_deg": float(
                    part["observed_center_separation_deg"].median()
                ),
                "separation_p10_deg": float(
                    part["observed_center_separation_deg"].quantile(0.1)
                ),
                "separation_p90_deg": float(
                    part["observed_center_separation_deg"].quantile(0.9)
                ),
                "normalized_separation_median": float(
                    part["normalized_center_separation"].median()
                ),
                "normalized_separation_p90": float(
                    part["normalized_center_separation"].quantile(0.9)
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        root / "et3_moderate_sky_pair_geometry_summary.csv",
        index=False,
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    rows = [" | ".join(str(row[column]) for column in columns) for _, row in frame.iterrows()]
    return "\n".join([f"| {labels} |", f"| {separator} |", *[f"| {row} |" for row in rows]])


def make_report(
    root: Path,
    scenario: pd.DataFrame,
    topb: pd.DataFrame,
    topb_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    moderate = pd.read_csv(root / "et3/moderate/retrieval_metrics_summary_v82.csv")
    overall = moderate[moderate["subset"] == "overall"].copy()
    selected_methods = [
        "waveform_only",
        "time_only",
        "sky_only",
        "time_sky_validation_selected",
        "three_channel_strict_positive",
    ]
    result_rows = []
    for method in selected_methods:
        row = overall[overall["method"] == method].iloc[0]
        result_rows.append(
            {
                "method": METHOD_LABELS[method],
                "R@1": f"{row.r_at_1_mean:.4f} ± {row.r_at_1_std:.4f}",
                "R@10": f"{row.r_at_10_mean:.4f} ± {row.r_at_10_std:.4f}",
                "median rank": f"{row.median_rank_mean:.1f}",
            }
        )
    result_table = markdown_table(
        pd.DataFrame(result_rows),
        ["method", "R@1", "R@10", "median rank"],
    )

    scenario_rows = []
    for name in SCENARIO_LABELS:
        row = scenario[
            (scenario["scenario"] == name) & (scenario["method"] == "sky_only")
        ].iloc[0]
        scenario_rows.append(
            {
                "scenario": SCENARIO_LABELS[name].replace("\n", " "),
                "seeds": int(row.n_seeds),
                "sky R@1": f"{row.r_at_1_mean:.4f}",
                "sky R@10": f"{row.r_at_10_mean:.4f}",
                "median rank": f"{row.median_rank_mean:.1f}",
            }
        )
    scenario_table = markdown_table(
        pd.DataFrame(scenario_rows),
        ["scenario", "seeds", "sky R@1", "sky R@10", "median rank"],
    )

    primary_topb = topb_summary[
        (topb_summary["method"] == "candidate_three_channel_strict_positive")
        & (topb_summary["budget_type"] == "fixed_count")
    ].copy()
    top_rows = []
    for _, row in primary_topb.sort_values("budget_pairs").iterrows():
        top_rows.append(
            {
                "B": int(row.budget_pairs),
                "TP": f"{row.true_recovered_mean:.2f}",
                "FP": f"{row.false_candidates_mean:.2f}",
                "precision": f"{row.precision_mean:.4f}",
                "recall": f"{row.recall_mean:.4f}",
                "enrichment": f"{row.enrichment_over_base_rate_mean:.1f}×",
            }
        )
    top_table = markdown_table(
        pd.DataFrame(top_rows),
        ["B", "TP", "FP", "precision", "recall", "enrichment"],
    )
    fraction_rows = []
    primary_fraction = topb_summary[
        (topb_summary["method"] == "candidate_three_channel_strict_positive")
        & (topb_summary["budget_type"] == "top_pair_fraction")
    ].copy()
    for _, row in primary_fraction.sort_values("budget_pairs").iterrows():
        fraction_rows.append(
            {
                "fraction": str(row.budget_label).replace("top_", "").replace("pct", "%"),
                "pairs": int(row.budget_pairs),
                "TP": f"{row.true_recovered_mean:.2f}",
                "FP": f"{row.false_candidates_mean:.2f}",
                "precision": f"{row.precision_mean:.4f}",
                "recall": f"{row.recall_mean:.4f}",
            }
        )
    fraction_table = markdown_table(
        pd.DataFrame(fraction_rows),
        ["fraction", "pairs", "TP", "FP", "precision", "recall"],
    )
    audits = pd.read_csv(root / "et3/moderate/sky_audit_per_seed_v82.csv")
    pair_metrics = pd.read_csv(
        root / "et3/moderate/pair_level_metrics_summary_v82.csv"
    )
    strict_pair = pair_metrics[
        pair_metrics["method"] == "three_channel_strict_positive"
    ].iloc[0]
    geometry = pd.read_csv(root / "et3_moderate_sky_pair_geometry_summary.csv")
    true_geometry = geometry[geometry["pair_class"] == "true_companion"]
    false_geometry = geometry[geometry["pair_class"] == "sampled_false"]

    report = f"""# ET-3 中等信息量 observed-sky 与 GWTC-3 Top-B 诊断

## 1. 实验边界

本实验不重新训练 waveform encoder，不修改 ET 时间分数，也不修改 GWTC
真实 HEALPix 方案。ET 部分是透明、受控的 sky-localization surrogate，不是
完整 ET Bayesian parameter estimation；GWTC 部分是 synthetic lensed pairs
injected into real O3 off-source noise，不是已确认的真实透镜。

## 2. ET observed-sky 生成

同一透镜系统的两幅像共享真实 RA/Dec，但分别生成独立 SNR-conditioned
定位面积、椭圆形状、方向、观测中心和噪声。非伴随事件的真实天空独立。

主情景固定为：

- `A90_ref=400 deg2` at network SNR 12；
- `A90 ∝ rho^-2`；
- lognormal scatter 0.6；
- clip `[100,5000] deg2`；
- ellipse axis ratio uniform `[1,3]`；
- 主结果无次级模态；
- validation HEALPix HPD coverage 选择一个全局 temperature；
- test labels、test retrieval 和 GWTC candidates 均不参与校准。

所有后验在 `nside=32` 归一化。唯一主天空分数是：

```text
B_sky = N_pix * sum(P_i * P_j)
sky_score = log(B_sky)
```

因此仍然是 waveform、time、sky 三个分数矩阵的 validation-selected 加权和。

## 3. ET-3 五 seed 结果

{result_table}

主情景 sky-only `R@10={overall[overall.method == 'sky_only'].iloc[0].r_at_10_mean:.4f} ± {overall[overall.method == 'sky_only'].iloc[0].r_at_10_std:.4f}`，介于旧的
过强 Gaussian proxy 与 v8.1 的近全天空八等权模态之间。

五 seed held-out true-sky coverage 为
`{audits.test_coverage.min():.4f}–{audits.test_coverage.max():.4f}`。
validation true-vs-false sky score ROC AUC 为
`{audits.roc_auc.mean():.4f} ± {audits.roc_auc.std(ddof=1):.4f}`。
真伴随对 observed-center separation 的跨 seed 中位数为
`{true_geometry.separation_median_deg.mean():.2f} deg`，随机假对为
`{false_geometry.separation_median_deg.mean():.2f} deg`。这一区分来自共享真实
方向与独立测量误差，不是直接修改 pair label 的分数。

三通道完整 `40,495,500` unordered-pair 结果：

- ROC AUC `{strict_pair.roc_auc_mean:.6f} ± {strict_pair.roc_auc_std:.6f}`；
- AUPRC `{strict_pair.average_precision_mean:.6f} ± {strict_pair.average_precision_std:.6f}`；
- positive base rate `{strict_pair.base_positive_rate_mean:.8f}`。

## 4. 天空假设敏感性

{scenario_table}

`informative/conservative` 和次级模态结果只使用一个冻结 waveform seed，
用于参数敏感性，不作为多 seed 主结果。性能随定位面积和次级模态概率平滑变化，
说明主结果不是直接使用 companion label 写入 sky score。

## 5. GWTC-3/O3 real-noise injection Top-B

每个 held-out catalog：

- 450 events；
- 101,025 unordered pairs；
- 180 true injected companion pairs；
- 100,845 false pairs；
- base positive rate 0.00178174。

主排序使用每个 seed 在 validation systems 上冻结的
`candidate_three_channel_strict_positive` 权重。

{top_table}

Top-10 在三个 seed 中均为 4 true + 6 false，precision=0.40，
recall=0.0222，相对于基础率富集 224.5 倍。它并不表示 Top-10 全是真对。

按全部 unordered pairs 的百分比统计：

{fraction_table}

## 6. 指标解释

Companion R@10 是“每个透镜像分别查询时，伴随像是否进入自己的前十”。
Global Precision@10 是“整个 unordered-pair catalog 只取最高的十对，其中多少
是真对”。二者不能互换。

## 7. 科学限制

1. ET 主情景是 controlled moderate-information forecast，不可声称为真实 ET PE。
2. `A90_ref=400 deg2` 是预先固定的中间情景；必须同时展示 informative 和
   conservative sensitivity，不能根据 test R@10 事后选择。
3. 单峰椭圆仍简化了 ET 三角结构的多模态；25%/50% antipodal mode 结果用于
   测试这一限制。
4. GWTC Top-B 统计衡量 injected-signal candidate generation，不是实测透镜率、
   FAR 或 detection significance。

## 8. 关键文件

- `et3/moderate/retrieval_metrics_summary_v82.csv`
- `et3/moderate/retrieval_metrics_per_seed_v82.csv`
- `et3/moderate/selected_weights_per_seed_v82.csv`
- `et3/moderate/sky_audit_per_seed_v82.csv`
- `et3/moderate/pair_level_metrics_summary_v82.csv`
- `et3_moderate_sky_pair_geometry_sample.parquet`
- `et3_moderate_sky_pair_geometry_summary.csv`
- `et3_scenario_sensitivity_v82.csv`
- `gwtc3_topb/gwtc3_topb_true_false_per_seed.csv`
- `gwtc3_topb/gwtc3_topb_true_false_summary.csv`
- `gwtc3_topb/gwtc3_primary_three_channel_top1000_per_seed.csv`
- `figures/fig_et3_moderate_sky_gwtc3_topb.pdf`
- `figures/fig_et3_moderate_sky_diagnostics.pdf`
"""
    (root / "et3_moderate_sky_gwtc3_topb_report_cn.md").write_text(
        report,
        encoding="utf-8",
    )

    english = f"""# ET-3 moderate observed-sky and GWTC-3 Top-B diagnostics

This controlled experiment reuses frozen ET-3 v7 waveform/time evidence and
introduces an SNR-conditioned, independently perturbed elliptical sky
posterior. It is a moderate-information surrogate, not full ET Bayesian PE.

The five-seed ET sky-only result is R@10 =
{overall[overall.method == 'sky_only'].iloc[0].r_at_10_mean:.4f} ±
{overall[overall.method == 'sky_only'].iloc[0].r_at_10_std:.4f}; the
validation-selected strictly positive three-channel result is R@10 =
{overall[overall.method == 'three_channel_strict_positive'].iloc[0].r_at_10_mean:.4f}
± {overall[overall.method == 'three_channel_strict_positive'].iloc[0].r_at_10_std:.4f}.

The held-out O3 real-noise injection catalogs contain 450 events, 101,025
unordered pairs, and 180 true companion pairs per seed. The global Top-10
contains four true and six false pairs in every seed (precision 0.40), a
224.5-fold enrichment over the 0.178% base rate. These are candidate-generation
diagnostics, not detections or astrophysical false-alarm probabilities.
"""
    (root / "et3_moderate_sky_gwtc3_topb_report_en.md").write_text(
        english,
        encoding="utf-8",
    )

    summary = {
        "status": "complete",
        "version": "et3_moderate_sky_gwtc3_topb_20260726",
        "et3_moderate": {
            row["method"]: {
                "r_at_1_mean": float(row["r_at_1_mean"]),
                "r_at_1_std": float(row["r_at_1_std"]),
                "r_at_10_mean": float(row["r_at_10_mean"]),
                "r_at_10_std": float(row["r_at_10_std"]),
            }
            for _, row in overall[overall["method"].isin(selected_methods)].iterrows()
        },
        "et3_test_coverage_range": [
            float(audits["test_coverage"].min()),
            float(audits["test_coverage"].max()),
        ],
        "et3_pair_level": {
            "n_pairs": int(strict_pair.n_pairs_mean),
            "n_true_pairs": int(strict_pair.n_true_pairs_mean),
            "n_false_pairs": int(strict_pair.n_false_pairs_mean),
            "roc_auc_mean": float(strict_pair.roc_auc_mean),
            "average_precision_mean": float(strict_pair.average_precision_mean),
        },
        "gwtc3_top10": {
            "true_pairs_per_seed": [4, 4, 4],
            "false_pairs_per_seed": [6, 6, 6],
            "precision": 0.4,
            "base_rate": float(topb["base_positive_rate"].iloc[0]),
            "enrichment": 224.5,
        },
        "scope": {
            "et": "controlled moderate-information observed-sky surrogate",
            "gwtc": "held-out synthetic lensed injections into real O3 noise",
            "not_detection": True,
        },
    }
    write_json(root / "experiment_summary.json", summary)


def write_reproduction(root: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gw-catalog
PYTHON=/root/miniconda3/bin/python
OUT=results/et3_moderate_sky_gwtc3_topb_20260726

$PYTHON scripts/experiments/124_et3_moderate_observed_sky.py --scenario moderate
$PYTHON scripts/experiments/124_et3_moderate_observed_sky.py --scenario informative --seed 202607251 --skip-pair-metrics
$PYTHON scripts/experiments/124_et3_moderate_observed_sky.py --scenario conservative --seed 202607251 --skip-pair-metrics
$PYTHON scripts/experiments/124_et3_moderate_observed_sky.py --scenario moderate_secondary25 --seed 202607251 --skip-pair-metrics
$PYTHON scripts/experiments/124_et3_moderate_observed_sky.py --scenario moderate_secondary50 --seed 202607251 --skip-pair-metrics
$PYTHON scripts/experiments/125_gwtc3_real_noise_injection_topb.py --source-root $OUT/frozen_source/gwtc3
$PYTHON scripts/experiments/126_et3_moderate_sky_topb_report.py
"""
    path = root / "reproduce.sh"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    write_json(
        root / "run_config.json",
        {
            "output_root": str(root),
            "et_primary_seeds": [
                202607251,
                202607252,
                202607253,
                202607254,
                202607255,
            ],
            "gwtc_seeds": [202607241, 202607242, 202607243],
            "et_primary_scenario": {
                "a90_ref_deg2": 400.0,
                "rho_ref": 12.0,
                "lognormal_scatter": 0.6,
                "clip_min_deg2": 100.0,
                "clip_max_deg2": 5000.0,
                "axis_ratio_range": [1.0, 3.0],
                "secondary_weight": 0.0,
                "nside": 32,
            },
            "main_sky_score": "log(N_pix*sum(P_i*P_j))",
            "test_used_for_tuning": False,
        },
    )
    readme = """# Current results

This directory contains the ET-3 moderate-information observed-sky surrogate
and the GWTC-3/O3 held-out real-noise injection Top-B diagnostic.

Authoritative entry points:

- `et3_moderate_sky_gwtc3_topb_report_cn.md`
- `experiment_summary.json`
- `et3/moderate/retrieval_metrics_summary_v82.csv`
- `et3_scenario_sensitivity_v82.csv`
- `gwtc3_topb/gwtc3_topb_true_false_summary.csv`
- `frozen_source/gwtc3/seed_*/fusion_heldout_test_pairs_v81.parquet`
- `figures/fig_et3_moderate_sky_gwtc3_topb.pdf`

The ET sky model is a controlled surrogate, not full Bayesian ET parameter
estimation. The GWTC diagnostic uses synthetic lensed systems injected into
real O3 off-source noise; it is not a search result for confirmed real lenses.
"""
    (root / "README_RESULTS.md").write_text(readme, encoding="utf-8")


def inventory(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "suffix": path.suffix,
            }
        )
    pd.DataFrame(rows).to_csv(root / "data_inventory.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    scenario = scenario_summary(args.root)
    scenario.to_csv(args.root / "et3_scenario_sensitivity_v82.csv", index=False)
    comparison = v81_comparison(args.root)
    comparison.to_csv(args.root / "et3_moderate_vs_v81.csv", index=False)
    topb = pd.read_csv(
        args.root / "gwtc3_topb/gwtc3_topb_true_false_per_seed.csv"
    )
    topb_summary = pd.read_csv(
        args.root / "gwtc3_topb/gwtc3_topb_true_false_summary.csv"
    )
    plot_main(args.root, scenario, topb)
    plot_sky_distribution(args.root)
    sky_geometry_audit(args.root)
    make_report(args.root, scenario, topb, topb_summary, comparison)
    write_reproduction(args.root)
    inventory(args.root)


if __name__ == "__main__":
    main()
