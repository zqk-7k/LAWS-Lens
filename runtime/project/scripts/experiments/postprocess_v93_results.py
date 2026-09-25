#!/usr/bin/env python3
"""Post-process the GWTC sky-resolution v9.3 rerun.

PE posterior consistency is attached only after the validation-selected
ranking has been frozen.  It is never used to train, calibrate, select fusion
weights, or reorder candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = REPO / "results/gwtc_sky_resolution_v93_20260730"
DEFAULT_PE = REPO / "results/unified_sky_v81_20260725"
DEPLOYMENTS = ("gwtc3", "gwtc4")
SEEDS = (202607241, 202607242, 202607243)
NSIDES = (32, 64, 128, 256, 512, 1024)
FAILED_O3_PAIRS = (
    ("GW190924_021846", "GW200202_154313"),
    ("GW190707_093326", "GW190828_065509"),
    ("GW190728_064510", "GW190828_065509"),
)


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def pair_key(left: str, right: str) -> str:
    return "||".join(sorted((str(left), str(right))))


def numeric_summary(
    frame: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column not in groups
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for key, part in frame.groupby(groups, sort=False):
        values_key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, values_key))
        row["n_seeds"] = int(part["seed"].nunique()) if "seed" in part else len(part)
        for column in numeric:
            values = part[column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            row[f"{column}_std"] = (
                float(finite.std(ddof=1))
                if len(finite) > 1
                else (0.0 if len(finite) == 1 else np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_retrieval(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for deployment in DEPLOYMENTS:
        path = (
            result_root
            / deployment
            / "heldout_test_retrieval_metrics_per_seed_v81.csv"
        )
        frames.append(pd.read_csv(path))
    per_seed = pd.concat(frames, ignore_index=True)
    summary = numeric_summary(
        per_seed,
        ["deployment", "method", "subset"],
    )
    per_seed.to_csv(
        result_root / "retrieval_metrics_per_seed_v93.csv",
        index=False,
    )
    summary.to_csv(
        result_root / "retrieval_metrics_summary_v93.csv",
        index=False,
    )
    return per_seed, summary


def compare_with_v81(
    result_root: Path,
    v81_root: Path,
    new_per_seed: pd.DataFrame,
) -> pd.DataFrame:
    old_frames = [
        pd.read_csv(
            v81_root
            / deployment
            / "heldout_test_retrieval_metrics_per_seed_v81.csv"
        )
        for deployment in DEPLOYMENTS
    ]
    old = pd.concat(old_frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        methods = sorted(
            set(
                new_per_seed.loc[
                    new_per_seed["deployment"] == deployment,
                    "method",
                ]
            )
        )
        for method in methods:
            for subset in ("overall", "sis", "pm"):
                before = old[
                    (old["deployment"] == deployment)
                    & (old["method"] == method)
                    & (old["subset"] == subset)
                ]
                after = new_per_seed[
                    (new_per_seed["deployment"] == deployment)
                    & (new_per_seed["method"] == method)
                    & (new_per_seed["subset"] == subset)
                ]
                if before.empty or after.empty:
                    continue
                row: dict[str, Any] = {
                    "deployment": deployment,
                    "method": method,
                    "subset": subset,
                }
                for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
                    old_mean = float(before[metric].mean())
                    new_mean = float(after[metric].mean())
                    row[f"v81_nside32_{metric}"] = old_mean
                    row[f"v93_{metric}"] = new_mean
                    row[f"delta_{metric}"] = new_mean - old_mean
                rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        result_root / "v81_nside32_vs_v93_comparison.csv",
        index=False,
    )
    return comparison


def contribution_consensus(
    result_root: Path,
    deployment: str,
) -> pd.DataFrame:
    root = result_root / deployment
    stability = pd.read_parquet(
        root / "real_candidate_rank_stability_v81.parquet"
    )
    weights = pd.read_csv(root / "selected_weights_per_seed_v81.csv")
    weights = weights[
        weights["method"] == "candidate_three_channel_strict_positive"
    ][["seed", "waveform", "time", "sky"]]
    stability = stability.merge(
        weights,
        on="seed",
        how="left",
        validate="many_to_one",
    )
    stability["waveform_contribution"] = (
        stability["waveform_score"] * stability["waveform"]
    )
    stability["time_contribution"] = (
        stability["time_score"] * stability["time"]
    )
    stability["sky_contribution"] = (
        stability["sky_score"] * stability["sky"]
    )
    consensus = (
        stability.groupby(["event_i", "event_j"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            rank_min=("rank", "min"),
            rank_max=("rank", "max"),
            final_score_mean=("final_score", "mean"),
            waveform_score_mean=("waveform_score", "mean"),
            time_score_mean=("time_score", "mean"),
            sky_score_mean=("sky_score", "mean"),
            waveform_contribution_mean=("waveform_contribution", "mean"),
            time_contribution_mean=("time_contribution", "mean"),
            sky_contribution_mean=("sky_contribution", "mean"),
        )
        .sort_values(["rank_mean", "rank_max"], kind="stable")
        .reset_index(drop=True)
    )
    consensus.insert(
        0,
        "consensus_rank",
        np.arange(1, len(consensus) + 1),
    )
    consensus["pair_key"] = [
        pair_key(left, right)
        for left, right in zip(consensus["event_i"], consensus["event_j"])
    ]
    return consensus


def pe_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "consensus_rank",
        "event_i",
        "event_j",
        "seeds",
        "rank_mean",
        "rank_std",
        "rank_min",
        "rank_max",
        "final_score_mean",
        "waveform_score_mean",
        "time_score_mean",
        "sky_score_mean",
        "deployment",
    }
    return [column for column in frame.columns if column not in excluded]


def attach_pe_and_resolution(
    result_root: Path,
    pe_root: Path,
    deployment: str,
) -> pd.DataFrame:
    consensus = contribution_consensus(result_root, deployment)
    pe = pd.read_parquet(
        pe_root
        / deployment
        / "real_candidate_consensus_with_pe_v81.parquet"
    )
    convergence = pd.read_parquet(
        result_root
        / deployment
        / "real_sky_resolution_convergence_all_pairs_v93.parquet"
    )
    pe_payload = pe[pe_columns(pe)].copy()
    if "pair_key" not in pe_payload:
        pe_payload["pair_key"] = [
            pair_key(left, right)
            for left, right in zip(pe["event_i"], pe["event_j"])
        ]
    convergence_columns = [
        "pair_key",
        *[f"sky_log_bf_nside{nside}" for nside in NSIDES],
        "positive_at_all_high_resolutions",
        "negative_at_all_high_resolutions",
        "high_resolution_sign_stable",
        "sign_flip_nside32_1024",
        "abs_delta_nside512_1024",
    ]
    merged = (
        consensus.merge(
            pe_payload,
            on="pair_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            convergence[convergence_columns],
            on="pair_key",
            how="left",
            validate="one_to_one",
        )
    )
    merged["deployment"] = deployment
    merged.to_parquet(
        result_root
        / deployment
        / "real_candidate_consensus_with_pe_v93.parquet",
        index=False,
    )
    merged.head(100).to_csv(
        result_root
        / deployment
        / "real_candidate_top100_resolution_audit_v93.csv",
        index=False,
    )
    return merged


def pe_budget_summary(
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deployment, frame in candidates.items():
        for budget in (5, 10, 20, 50, 100):
            part = frame.head(budget)
            available = part["pe_available"].fillna(False).astype(bool)
            passed = (
                part["max_standardized_posterior_distance"].le(3.0)
                & available
            )
            rows.append(
                {
                    "deployment": deployment,
                    "top_budget": budget,
                    "n_pairs": int(len(part)),
                    "n_pe_available": int(available.sum()),
                    "n_dmax_le_3": int(passed.sum()),
                    "fraction_dmax_le_3_among_pe": (
                        float(passed.sum() / available.sum())
                        if available.sum()
                        else np.nan
                    ),
                    "n_sign_flip_nside32_1024": int(
                        part["sign_flip_nside32_1024"].sum()
                    ),
                    "n_high_resolution_sign_unstable": int(
                        (~part["high_resolution_sign_stable"]).sum()
                    ),
                    "n_positive_at_all_high_resolutions": int(
                        part["positive_at_all_high_resolutions"].sum()
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    return summary


def old_failed_pair_audit(
    result_root: Path,
    pe_root: Path,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    old = pd.read_parquet(
        pe_root / "gwtc3/real_candidate_consensus_with_pe_v81.parquet"
    ).copy()
    if "pair_key" not in old:
        old["pair_key"] = [
            pair_key(left, right)
            for left, right in zip(old["event_i"], old["event_j"])
        ]
    new = candidates.set_index("pair_key")
    old = old.set_index("pair_key")
    rows: list[dict[str, Any]] = []
    for left, right in FAILED_O3_PAIRS:
        key = pair_key(left, right)
        old_row = old.loc[key]
        new_row = new.loc[key]
        row: dict[str, Any] = {
            "event_i": left,
            "event_j": right,
            "old_consensus_rank_nside32": int(old_row["consensus_rank"]),
            "new_consensus_rank_nside1024": int(new_row["consensus_rank"]),
            "old_sky_log_bf_nside32_consensus_mean": float(
                old_row["sky_score_mean"]
            ),
            "new_sky_log_bf_nside1024_consensus_mean": float(
                new_row["sky_score_mean"]
            ),
            "max_standardized_posterior_distance": float(
                new_row["max_standardized_posterior_distance"]
            ),
            "intrinsic_3sigma_consistent": bool(
                new_row["intrinsic_3sigma_consistent"]
            ),
        }
        for nside in NSIDES:
            row[f"sky_log_bf_nside{nside}"] = float(
                new_row[f"sky_log_bf_nside{nside}"]
            )
        row["high_resolution_sign_stable"] = bool(
            new_row["high_resolution_sign_stable"]
        )
        row["abs_delta_nside512_1024"] = float(
            new_row["abs_delta_nside512_1024"]
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(
        result_root / "gwtc3/old_failed_pairs_v93_audit.csv",
        index=False,
    )
    return output


def historical_candidate_audit(
    result_root: Path,
    pe_root: Path,
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    source = pd.read_csv(pe_root / "historical_candidate_rank_audit_v81.csv")
    rows: list[dict[str, Any]] = []
    for _, item in source.iterrows():
        deployment = str(item["deployment"])
        key = pair_key(item["event_i"], item["event_j"])
        lookup = candidates[deployment].set_index("pair_key")
        row: dict[str, Any] = {
            "deployment": deployment,
            "event_i": item["event_i"],
            "event_j": item["event_j"],
            "audit_label": item["audit_label"],
            "old_v81_consensus_rank": item["consensus_rank"],
            "present_in_new_strict_catalog": key in lookup.index,
        }
        if key in lookup.index:
            match = lookup.loc[key]
            row.update(
                {
                    "new_v93_consensus_rank": int(match["consensus_rank"]),
                    "new_rank_mean": float(match["rank_mean"]),
                    "new_rank_min": int(match["rank_min"]),
                    "new_rank_max": int(match["rank_max"]),
                    "sky_log_bf_nside1024": float(
                        match["sky_log_bf_nside1024"]
                    ),
                    "positive_at_all_high_resolutions": bool(
                        match["positive_at_all_high_resolutions"]
                    ),
                    "max_standardized_posterior_distance": float(
                        match["max_standardized_posterior_distance"]
                    ),
                    "intrinsic_3sigma_consistent": bool(
                        match["intrinsic_3sigma_consistent"]
                    ),
                }
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(
        result_root / "historical_candidate_rank_audit_v93.csv",
        index=False,
    )
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(
                    "nan" if not np.isfinite(value) else f"{float(value):.4f}"
                )
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_figure(
    result_root: Path,
    per_seed: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    failed: pd.DataFrame,
) -> None:
    figures = result_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    methods = [
        ("waveform_only", "Waveform", "#4C78A8"),
        ("time_only", "Time", "#F2CF5B"),
        ("sky_only", "Sky", "#59A14F"),
        (
            "retrieval_three_channel_unconstrained",
            "Three-channel",
            "#E45756",
        ),
    ]
    x = np.arange(2)
    width = 0.18
    for index, (method, label, color) in enumerate(methods):
        means = []
        errors = []
        for deployment in DEPLOYMENTS:
            part = per_seed[
                (per_seed["deployment"] == deployment)
                & (per_seed["method"] == method)
                & (per_seed["subset"] == "overall")
            ]
            means.append(part["r_at_10"].mean())
            errors.append(part["r_at_10"].std(ddof=1))
        axes[0].bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=errors,
            capsize=2,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, ["GWTC-3/O3", "GWTC-4.1/O4a"])
    axes[0].set_ylabel("Held-out R@10")
    axes[0].set_title("a  Injection retrieval", loc="left")
    axes[0].legend(frameon=False, fontsize=7)

    for row_index, row in failed.iterrows():
        axes[1].plot(
            NSIDES,
            [row[f"sky_log_bf_nside{nside}"] for nside in NSIDES],
            marker="o",
            label=f"Old rank {int(row['old_consensus_rank_nside32'])}",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(NSIDES, [str(value) for value in NSIDES])
    axes[1].set_xlabel("Common HEALPix Nside")
    axes[1].set_ylabel(r"$\log B_{\rm sky}$")
    axes[1].set_title("b  O3 failed-pair convergence", loc="left")
    axes[1].legend(frameon=False, fontsize=7)

    colors = {"gwtc3": "#4C78A8", "gwtc4": "#E45756"}
    for deployment in DEPLOYMENTS:
        top = candidates[deployment].head(10)
        axes[2].scatter(
            top["consensus_rank"],
            top["max_standardized_posterior_distance"],
            color=colors[deployment],
            label=("GWTC-3/O3" if deployment == "gwtc3" else "GWTC-4.1/O4a"),
            alpha=0.85,
        )
    axes[2].axhline(3.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Consensus candidate rank")
    axes[2].set_ylabel(r"PE $D_{\max}$")
    axes[2].set_title("c  Frozen PE audit", loc="left")
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=8)
    fig.tight_layout()
    fig.savefig(
        figures / "fig_gwtc_sky_resolution_v93.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        figures / "fig_gwtc_sky_resolution_v93.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_report(
    result_root: Path,
    retrieval_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    pe_budget: pd.DataFrame,
    failed: pd.DataFrame,
    historical: pd.DataFrame,
) -> None:
    method_order = [
        "waveform_only",
        "time_only",
        "sky_only",
        "retrieval_three_channel_unconstrained",
        "retrieval_three_channel_strict_positive",
    ]
    metric_rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        for method in method_order:
            row = retrieval_summary[
                (retrieval_summary["deployment"] == deployment)
                & (retrieval_summary["method"] == method)
                & (retrieval_summary["subset"] == "overall")
            ]
            if row.empty:
                continue
            item = row.iloc[0]
            metric_rows.append(
                {
                    "deployment": deployment,
                    "method": method,
                    "R@1 mean": item["r_at_1_mean"],
                    "R@1 SD": item["r_at_1_std"],
                    "R@10 mean": item["r_at_10_mean"],
                    "R@10 SD": item["r_at_10_std"],
                    "median rank": item["median_rank_mean"],
                }
            )
    metrics = pd.DataFrame(metric_rows)
    comparison_report = comparison[
        (comparison["subset"] == "overall")
        & (
            comparison["method"].isin(
                [
                    "sky_only",
                    "retrieval_three_channel_unconstrained",
                    "retrieval_three_channel_strict_positive",
                ]
            )
        )
    ][
        [
            "deployment",
            "method",
            "v81_nside32_r_at_1",
            "v93_r_at_1",
            "delta_r_at_1",
            "v81_nside32_r_at_10",
            "v93_r_at_10",
            "delta_r_at_10",
        ]
    ]

    weight_rows = []
    for deployment in DEPLOYMENTS:
        weights = pd.read_csv(
            result_root
            / deployment
            / "selected_weights_per_seed_v81.csv"
        )
        weights = weights[
            weights["method"].isin(
                [
                    "retrieval_three_channel_unconstrained",
                    "candidate_three_channel_strict_positive",
                ]
            )
        ]
        weight_rows.append(weights)
    weights = pd.concat(weight_rows, ignore_index=True)

    top_frames = []
    for deployment, frame in candidates.items():
        top = frame.head(10)[
            [
                "consensus_rank",
                "event_i",
                "event_j",
                "rank_mean",
                "waveform_score_mean",
                "time_score_mean",
                "sky_score_mean",
                "waveform_contribution_mean",
                "time_contribution_mean",
                "sky_contribution_mean",
                "max_standardized_posterior_distance",
                "intrinsic_3sigma_consistent",
                "positive_at_all_high_resolutions",
                "high_resolution_sign_stable",
                "abs_delta_nside512_1024",
            ]
        ].copy()
        top.insert(0, "deployment", deployment)
        top_frames.append(top)
    top10 = pd.concat(top_frames, ignore_index=True)
    top10.to_csv(
        result_root / "real_candidate_top10_v93.csv",
        index=False,
    )

    resolution_rows = []
    for deployment in DEPLOYMENTS:
        payload = json.loads(
            (
                result_root
                / deployment
                / "real_sky_resolution_convergence_summary_v93.json"
            ).read_text(encoding="utf-8")
        )
        resolution_rows.append(
            {
                "deployment": deployment,
                "n_events": payload["n_events"],
                "n_pairs": payload["n_unordered_pairs"],
                "32-to-1024 sign flips": payload[
                    "sign_flips_nside32_to_1024"
                ],
                "high-res sign unstable": payload[
                    "high_resolution_sign_unstable"
                ],
                "median |512-1024|": payload[
                    "abs_delta_nside512_1024"
                ]["median"],
                "q99 |512-1024|": payload[
                    "abs_delta_nside512_1024"
                ]["q99"],
            }
        )
    resolution = pd.DataFrame(resolution_rows)

    report = f"""# O3/O4a 天空分辨率 v9.3 重跑报告

## 1. 冻结边界

本轮只改变天空图的离散分辨率：

- O3/O4a synthetic real-noise injection：保持原生 `Nside=64`；
- 真实 GWTC PE sky map：正式排名统一到 `Nside=1024`；
- 真实 pair 额外计算 `32,64,128,256,512,1024` 收敛审计。

waveform encoder、waveform/time calibration、系统划分、三个训练 seed、融合
网格、validation-only 选择规则、strict H1-L1 BBH 范围和 PE 指标定义均未
改变。ET-3 未运行。

主天空分数始终为：

\\[
Z_{{\\mathrm{{sky}},ij}}=\\log\\left[N_{{\\rm pix}}
\\sum_k P_i(k)P_j(k)\\right].
\\]

## 2. Held-out injection retrieval

{markdown_table(metrics)}

waveform-only 和 time-only 必须与 v8.1 逐 seed 完全一致；对应自动审计见
`frozen_waveform_time_reproduction_v93.csv`。

## 3. Validation 选择的权重

{markdown_table(weights)}

所有权重只由 synthetic validation systems 选择。真实排名、PE 和 held-out
test 均未参与选权。

## 4. 相对 Nside=32 v8.1 的变化

{markdown_table(comparison_report)}

该表比较的是各自 validation-only 重新选权后的 held-out 结果，不是固定权重
下只替换一个 test 分数。

## 5. 真实天空分辨率收敛

{markdown_table(resolution)}

`high-res sign unstable` 表示 256、512、1024 的符号未保持一致。这类 pair
不能描述为具有稳健共同天空支持。

## 6. PE budget audit

{markdown_table(pe_budget)}

PE 在排名冻结后附加，未用于重排。`Dmax<=3` 是后续物理一致性审计，不是
第四个检索通道。

## 7. 原 O3 失败候选

{markdown_table(failed)}

## 8. 预先指定的历史候选

{markdown_table(historical)}

这些 pair 在查看 v9.3 结果前已经固定；新 rank 不是显著性或 p-value。

## 9. 新真实目录 Top-10

{markdown_table(top10)}

这些结果是 catalog coincidences / Bayesian follow-up shortlist，不是
lensing detections。高 rank、正 sky Bayes factor 或 PE 3-sigma screen
中的任意一个都不能单独构成探测显著性。
"""
    (result_root / "gwtc_sky_resolution_v93_report_cn.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--pe-root", type=Path, default=DEFAULT_PE)
    args = parser.parse_args()

    contract = json.loads(
        (args.result_root / "analysis_contract_v93.json").read_text(
            encoding="utf-8"
        )
    )
    if contract.get("et3_rerun") is not False:
        raise RuntimeError("The result contract does not explicitly exclude ET-3")
    per_seed, summary = aggregate_retrieval(args.result_root)
    comparison = compare_with_v81(
        args.result_root,
        args.pe_root,
        per_seed,
    )
    candidates = {
        deployment: attach_pe_and_resolution(
            args.result_root,
            args.pe_root,
            deployment,
        )
        for deployment in DEPLOYMENTS
    }
    budgets = pe_budget_summary(candidates)
    budgets.to_csv(
        args.result_root / "real_candidate_pe_budget_summary_v93.csv",
        index=False,
    )
    failed = old_failed_pair_audit(
        args.result_root,
        args.pe_root,
        candidates["gwtc3"],
    )
    historical = historical_candidate_audit(
        args.result_root,
        args.pe_root,
        candidates,
    )
    make_figure(args.result_root, per_seed, candidates, failed)
    make_report(
        args.result_root,
        summary,
        comparison,
        candidates,
        budgets,
        failed,
        historical,
    )

    core = {}
    for deployment in DEPLOYMENTS:
        rows = summary[
            (summary["deployment"] == deployment)
            & (summary["subset"] == "overall")
            & (
                summary["method"].isin(
                    [
                        "waveform_only",
                        "time_only",
                        "sky_only",
                        "retrieval_three_channel_unconstrained",
                    ]
                )
            )
        ]
        core[deployment] = {
            str(row["method"]): {
                "r_at_1_mean": float(row["r_at_1_mean"]),
                "r_at_1_std": float(row["r_at_1_std"]),
                "r_at_10_mean": float(row["r_at_10_mean"]),
                "r_at_10_std": float(row["r_at_10_std"]),
            }
            for _, row in rows.iterrows()
        }
        top10 = candidates[deployment].head(10)
        core[deployment]["real_candidate_audit"] = {
            "top10_dmax_le_3": int(
                top10["max_standardized_posterior_distance"].le(3.0).sum()
            ),
            "top10_positive_at_all_high_resolutions": int(
                top10["positive_at_all_high_resolutions"].sum()
            ),
            "top10_high_resolution_sign_unstable": int(
                (~top10["high_resolution_sign_stable"]).sum()
            ),
        }
    write_json(
        args.result_root / "final_audit_summary_v93.json",
        {
            "status": "complete",
            "version": "gwtc_sky_resolution_v93",
            "et3_rerun": False,
            "injection_nside": 64,
            "real_ranking_nside": 1024,
            "pe_used_for_ranking_or_weight_selection": False,
            "v81_comparison_file": "v81_nside32_vs_v93_comparison.csv",
            "historical_candidate_file": (
                "historical_candidate_rank_audit_v93.csv"
            ),
            "core_results": core,
            "interpretation": (
                "Candidate shortlist for Bayesian follow-up; not a lensing "
                "detection or significance claim."
            ),
        },
    )
    print("[complete] v9.3 PE and convergence post-processing finished")


if __name__ == "__main__":
    main()
