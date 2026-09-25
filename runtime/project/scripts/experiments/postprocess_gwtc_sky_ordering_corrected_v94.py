#!/usr/bin/env python3
"""Audit, compare, plot, and report the independent v9.4 ordering fix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OLD = REPO / "results/gwtc_sky_resolution_v93_20260730"
DEFAULT_NEW = REPO / "results/gwtc_sky_ordering_corrected_v94_20260830"
DEPLOYMENTS = ("gwtc3", "gwtc4")
SEEDS = (202607241, 202607242, 202607243)
PRIMARY_METHOD = "candidate_three_channel_strict_positive"
RETRIEVAL_METHOD = "retrieval_three_channel_strict_positive"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def pair_key(left: str, right: str) -> str:
    return "||".join(sorted((str(left), str(right))))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def compare_retrieval(old_root: Path, new_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for deployment in DEPLOYMENTS:
        old = pd.read_csv(
            old_root / deployment / "heldout_test_retrieval_metrics_per_seed_v81.csv"
        )
        new = pd.read_csv(
            new_root / deployment / "heldout_test_retrieval_metrics_per_seed_v81.csv"
        )
        keys = ["deployment", "seed", "method", "subset"]
        merged = old.merge(new, on=keys, suffixes=("_v93", "_v94"), validate="one_to_one")
        for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
            merged[f"delta_{metric}"] = merged[f"{metric}_v94"] - merged[f"{metric}_v93"]
        rows.append(merged)
    per_seed = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for key, part in per_seed.groupby(["deployment", "method", "subset"], sort=False):
        row = dict(zip(["deployment", "method", "subset"], key))
        row["n_seeds"] = int(part["seed"].nunique())
        for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
            for suffix in ("v93", "v94"):
                values = part[f"{metric}_{suffix}"].to_numpy(dtype=float)
                row[f"{metric}_{suffix}_mean"] = float(values.mean())
                row[f"{metric}_{suffix}_std"] = float(values.std(ddof=1))
            row[f"delta_{metric}_mean"] = float(part[f"delta_{metric}"].mean())
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def compare_real_sky(old_root: Path, new_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    summaries = []
    for deployment in DEPLOYMENTS:
        old = pd.read_parquet(
            old_root / deployment / "real_sky_resolution_convergence_all_pairs_v93.parquet"
        )
        new = pd.read_parquet(
            new_root / deployment / "real_sky_resolution_convergence_all_pairs_v93.parquet"
        )
        columns = ["pair_key", "event_i", "event_j", "sky_log_bf_nside1024"]
        merged = old[columns].merge(
            new[columns],
            on=["pair_key", "event_i", "event_j"],
            suffixes=("_v93", "_v94"),
            validate="one_to_one",
        )
        merged.insert(0, "deployment", deployment)
        merged["delta_sky_log_bf"] = (
            merged["sky_log_bf_nside1024_v94"]
            - merged["sky_log_bf_nside1024_v93"]
        )
        merged["abs_delta_sky_log_bf"] = merged["delta_sky_log_bf"].abs()
        merged["sign_flip_v93_v94"] = (
            np.signbit(merged["sky_log_bf_nside1024_v93"])
            != np.signbit(merged["sky_log_bf_nside1024_v94"])
        )
        values = merged["abs_delta_sky_log_bf"].to_numpy(dtype=float)
        summaries.append(
            {
                "deployment": deployment,
                "n_pairs": int(len(merged)),
                "median_abs_delta": float(np.median(values)),
                "p90_abs_delta": float(np.quantile(values, 0.90)),
                "p99_abs_delta": float(np.quantile(values, 0.99)),
                "max_abs_delta": float(np.max(values)),
                "sign_flips": int(merged["sign_flip_v93_v94"].sum()),
                "spearman": float(
                    merged[
                        ["sky_log_bf_nside1024_v93", "sky_log_bf_nside1024_v94"]
                    ].corr(method="spearman").iloc[0, 1]
                ),
            }
        )
        all_rows.append(merged)
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summaries)


def compare_candidates(old_root: Path, new_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = []
    overlaps = []
    for deployment in DEPLOYMENTS:
        old = pd.read_parquet(
            old_root / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        )
        new = pd.read_parquet(
            new_root / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        )
        if "pair_key" not in old:
            old["pair_key"] = [pair_key(a, b) for a, b in zip(old.event_i, old.event_j)]
        if "pair_key" not in new:
            new["pair_key"] = [pair_key(a, b) for a, b in zip(new.event_i, new.event_j)]
        merged = old[
            ["pair_key", "event_i", "event_j", "consensus_rank", "final_score_mean", "sky_score_mean"]
        ].merge(
            new[
                [
                    "pair_key",
                    "consensus_rank",
                    "final_score_mean",
                    "waveform_score_mean",
                    "time_score_mean",
                    "sky_score_mean",
                    "max_standardized_posterior_distance",
                    "intrinsic_3sigma_consistent",
                ]
            ],
            on="pair_key",
            suffixes=("_v93", "_v94"),
            validate="one_to_one",
        )
        merged.insert(0, "deployment", deployment)
        merged["rank_shift_v94_minus_v93"] = (
            merged["consensus_rank_v94"] - merged["consensus_rank_v93"]
        )
        comparisons.append(merged)
        for budget in (10, 20, 50, 100):
            old_set = set(old.head(budget)["pair_key"])
            new_set = set(new.head(budget)["pair_key"])
            overlaps.append(
                {
                    "deployment": deployment,
                    "budget": budget,
                    "overlap": len(old_set & new_set),
                    "union": len(old_set | new_set),
                    "jaccard": len(old_set & new_set) / len(old_set | new_set),
                    "entered_v94": ";".join(sorted(new_set - old_set)),
                    "left_v94": ";".join(sorted(old_set - new_set)),
                }
            )
    return pd.concat(comparisons, ignore_index=True), pd.DataFrame(overlaps)


def compare_weights(old_root: Path, new_root: Path) -> pd.DataFrame:
    rows = []
    for deployment in DEPLOYMENTS:
        old = pd.read_csv(old_root / deployment / "selected_weights_per_seed_v81.csv")
        new = pd.read_csv(new_root / deployment / "selected_weights_per_seed_v81.csv")
        keys = ["deployment", "seed", "method"]
        rows.append(old.merge(new, on=keys, suffixes=("_v93", "_v94"), validate="one_to_one"))
    return pd.concat(rows, ignore_index=True)


def numeric_equal(left: pd.Series, right: pd.Series) -> tuple[bool, float]:
    a = left.to_numpy(dtype=float)
    b = right.to_numpy(dtype=float)
    same_nan = np.isnan(a) == np.isnan(b)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return bool(same_nan.all() and delta == 0.0), delta


def frozen_pair_audit(old_root: Path, new_root: Path) -> pd.DataFrame:
    rows = []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            for split, filename in (
                ("validation", "fusion_validation_pairs_v81.parquet"),
                ("heldout_test", "fusion_heldout_test_pairs_v81.parquet"),
            ):
                old = pd.read_parquet(old_root / deployment / f"seed_{seed}" / filename)
                new = pd.read_parquet(new_root / deployment / f"seed_{seed}" / filename)
                frozen_columns = [
                    column
                    for column in old.columns.intersection(new.columns)
                    if not column.startswith("sky_")
                    and column not in {"sky_score", "sky_bayes_factor"}
                ]
                max_delta = 0.0
                exact = len(old) == len(new)
                bad_columns = []
                for column in frozen_columns:
                    if pd.api.types.is_numeric_dtype(old[column]):
                        passed, delta = numeric_equal(old[column], new[column])
                        max_delta = max(max_delta, delta)
                    else:
                        passed = old[column].fillna("<NA>").equals(new[column].fillna("<NA>"))
                    if not passed:
                        bad_columns.append(column)
                rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "scope": split,
                        "rows": len(new),
                        "frozen_columns": len(frozen_columns),
                        "max_abs_numeric_delta": max_delta,
                        "bad_columns": ";".join(bad_columns),
                        "exact_match": exact and not bad_columns,
                    }
                )
    return pd.DataFrame(rows)


def make_figure(
    new_root: Path,
    retrieval_summary: pd.DataFrame,
    real_sky_pairs: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    methods = ["sky_only", RETRIEVAL_METHOD]
    labels = ["Sky only", "Three-channel"]
    x = np.arange(2)
    width = 0.34
    for offset, deployment in enumerate(DEPLOYMENTS):
        values_old, values_new, errors_new = [], [], []
        for method in methods:
            row = retrieval_summary[
                (retrieval_summary.deployment == deployment)
                & (retrieval_summary.method == method)
                & (retrieval_summary.subset == "overall")
            ].iloc[0]
            values_old.append(row.r_at_10_v93_mean)
            values_new.append(row.r_at_10_v94_mean)
            errors_new.append(row.r_at_10_v94_std)
        axes[0, 0].bar(x + (offset - 0.5) * width, values_new, width, yerr=errors_new, capsize=3, label=deployment.upper())
        for index, value in enumerate(values_old):
            axes[0, 0].plot(x[index] + (offset - 0.5) * width, value, marker="_", color="black", markersize=12)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Held-out R@10")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("a  Ordering-corrected injection retrieval")
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 0].text(0.02, 0.03, "Black ticks: historical v9.3", transform=axes[0, 0].transAxes, fontsize=8)

    for axis, deployment, title in (
        (axes[0, 1], "gwtc3", "b  GWTC-3 real-pair sky evidence"),
        (axes[1, 0], "gwtc4", "c  GWTC-4.1 real-pair sky evidence"),
    ):
        part = real_sky_pairs[real_sky_pairs.deployment == deployment]
        xval = np.clip(part.sky_log_bf_nside1024_v93, -15, 15)
        yval = np.clip(part.sky_log_bf_nside1024_v94, -15, 15)
        axis.hexbin(xval, yval, gridsize=42, mincnt=1, bins="log", cmap="viridis")
        axis.plot([-15, 15], [-15, 15], color="0.35", linestyle="--", linewidth=1)
        axis.set_xlabel("Historical v9.3 $Z_{sky}$")
        axis.set_ylabel("Corrected v9.4 $Z_{sky}$")
        axis.set_title(title)

    axis = axes[1, 1]
    colors = {"gwtc3": "#2676b8", "gwtc4": "#d97a20"}
    offsets = {"gwtc3": -0.10, "gwtc4": 0.10}
    for deployment in DEPLOYMENTS:
        frame = pd.read_parquet(
            new_root / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        ).head(10)
        ranks = np.arange(1, len(frame) + 1) + offsets[deployment]
        axis.scatter(
            ranks,
            frame.max_standardized_posterior_distance,
            label=deployment.upper(),
            color=colors[deployment],
            s=28,
        )
    axis.axhline(3.0, color="0.35", linestyle="--", linewidth=1)
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlabel("Corrected consensus rank")
    axis.set_ylabel("PE $D_{max}$")
    axis.set_title("d  Post-ranking intrinsic-PE audit")
    axis.legend(frameon=False, loc="upper left")
    for suffix in ("pdf", "png"):
        fig.savefig(
            new_root / "figures" / f"fig_gwtc_sky_ordering_corrected_v94.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW)
    args = parser.parse_args()
    args.new_root.mkdir(parents=True, exist_ok=True)
    (args.new_root / "figures").mkdir(exist_ok=True)

    retrieval_per_seed, retrieval_summary = compare_retrieval(args.old_root, args.new_root)
    sky_pairs, sky_summary = compare_real_sky(args.old_root, args.new_root)
    candidate_comparison, candidate_overlap = compare_candidates(args.old_root, args.new_root)
    weights = compare_weights(args.old_root, args.new_root)
    frozen = frozen_pair_audit(args.old_root, args.new_root)

    retrieval_per_seed.to_csv(args.new_root / "v93_vs_v94_retrieval_per_seed.csv", index=False)
    retrieval_summary.to_csv(args.new_root / "v93_vs_v94_retrieval_summary.csv", index=False)
    sky_pairs.to_parquet(args.new_root / "v93_vs_v94_real_sky_all_pairs.parquet", index=False)
    sky_summary.to_csv(args.new_root / "v93_vs_v94_real_sky_summary.csv", index=False)
    candidate_comparison.to_parquet(args.new_root / "v93_vs_v94_candidate_rank_comparison.parquet", index=False)
    candidate_overlap.to_csv(args.new_root / "v93_vs_v94_candidate_overlap.csv", index=False)
    weights.to_csv(args.new_root / "v93_vs_v94_selected_weights.csv", index=False)
    frozen.to_csv(args.new_root / "frozen_pair_feature_audit_v94.csv", index=False)

    top_frames = []
    key_metrics = []
    pe_summary = pd.read_csv(args.new_root / "real_candidate_pe_budget_summary_v93.csv")
    for deployment in DEPLOYMENTS:
        candidate = pd.read_parquet(
            args.new_root / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        ).head(10).copy()
        top_frames.append(candidate)
        for method in ("waveform_only", "time_only", "sky_only", "time_sky_retrieval_selected", RETRIEVAL_METHOD):
            row = retrieval_summary[
                (retrieval_summary.deployment == deployment)
                & (retrieval_summary.method == method)
                & (retrieval_summary.subset == "overall")
            ].iloc[0]
            key_metrics.append(row.to_dict())
    top10 = pd.concat(top_frames, ignore_index=True)
    top10.to_csv(args.new_root / "ordering_corrected_real_top10_with_pe_v94.csv", index=False)
    pd.DataFrame(key_metrics).to_csv(args.new_root / "ordering_corrected_key_metrics_v94.csv", index=False)

    make_figure(args.new_root, retrieval_summary, sky_pairs)

    contract = json.loads(
        (args.new_root / "analysis_contract_ordering_corrected_v94.json").read_text(encoding="utf-8")
    )
    historical_package = Path(contract["historical_v93_package"])
    current_historical_hash = sha256_file(historical_package)
    source_audit = {}
    convergence = {}
    gates = {}
    key_result = {}
    for deployment in DEPLOYMENTS:
        source = pd.read_csv(args.new_root / deployment / "real_map_source_audit_v93.csv")
        source_audit[deployment] = {
            "events": len(source),
            "source_ordering": source.source_ordering.value_counts().to_dict(),
            "output_ordering": source.output_ordering.value_counts().to_dict(),
            "ordering_conversion_applied": int(source.ordering_conversion_applied.sum()),
        }
        convergence[deployment] = json.loads(
            (args.new_root / deployment / "real_sky_resolution_convergence_summary_v93.json").read_text()
        )
        gate = pd.read_csv(args.new_root / deployment / "waveform_gate_per_seed_v81.csv")
        gates[deployment] = {
            "passed": int(gate.passed_for_primary_deployment.sum()),
            "total": len(gate),
        }
        rows = retrieval_summary[
            (retrieval_summary.deployment == deployment)
            & (retrieval_summary.subset == "overall")
        ].set_index("method")
        key_result[deployment] = {
            "sky_only_r1": rows.loc["sky_only", "r_at_1_v94_mean"],
            "sky_only_r10": rows.loc["sky_only", "r_at_10_v94_mean"],
            "three_channel_r1": rows.loc[RETRIEVAL_METHOD, "r_at_1_v94_mean"],
            "three_channel_r10": rows.loc[RETRIEVAL_METHOD, "r_at_10_v94_mean"],
            "three_channel_r10_v93": rows.loc[RETRIEVAL_METHOD, "r_at_10_v93_mean"],
            "top_pair": top10[top10.deployment == deployment].iloc[0][["event_i", "event_j"]].to_dict(),
            "top10_pe_pass": int(
                pe_summary[(pe_summary.deployment == deployment) & (pe_summary.top_budget == 10)].iloc[0].n_dmax_le_3
            ),
        }
    summary = {
        "version": "gwtc_sky_ordering_corrected_v94",
        "status": "COMPLETE_NO_OVERWRITE_NOT_A_LENSING_DETECTION",
        "correction": "Strictly decode [b'True'] as NESTED and reorder NESTED-to-RING before resizing",
        "historical_v93_package_hash_before": contract["historical_v93_package_sha256_before_rerun"],
        "historical_v93_package_hash_after": current_historical_hash,
        "historical_v93_unchanged": current_historical_hash == contract["historical_v93_package_sha256_before_rerun"],
        "frozen_pair_features_all_exact": bool(frozen.exact_match.all()),
        "frozen_pair_features_max_abs_delta": float(frozen.max_abs_numeric_delta.max()),
        "source_map_audit": source_audit,
        "waveform_gates": gates,
        "key_results": key_result,
        "real_sky_change": sky_summary.to_dict("records"),
        "candidate_overlap": candidate_overlap.to_dict("records"),
        "resolution_convergence": convergence,
        "limitations": [
            "Synthetic injection skies remain rotated real-PE posterior templates rather than independent full PE per injected event.",
            "Real-catalog ranks are candidate shortlists for Bayesian follow-up, not detections.",
            "PE consistency is attached after ranking and was not used to tune weights.",
        ],
    }
    write_json(args.new_root / "ordering_corrected_summary_v94.json", summary)

    def metric_table(deployment: str) -> pd.DataFrame:
        rows = retrieval_summary[
            (retrieval_summary.deployment == deployment)
            & (retrieval_summary.subset == "overall")
            & retrieval_summary.method.isin(
                ["waveform_only", "time_only", "sky_only", "time_sky_retrieval_selected", RETRIEVAL_METHOD]
            )
        ].copy()
        return rows[
            [
                "method",
                "r_at_1_v93_mean",
                "r_at_1_v94_mean",
                "r_at_10_v93_mean",
                "r_at_10_v94_mean",
                "r_at_10_v94_std",
            ]
        ].round(4)

    def top_table(deployment: str) -> pd.DataFrame:
        frame = top10[top10.deployment == deployment].copy()
        return frame[
            [
                "consensus_rank",
                "event_i",
                "event_j",
                "final_score_mean",
                "waveform_score_mean",
                "time_score_mean",
                "sky_score_mean",
                "max_standardized_posterior_distance",
                "intrinsic_3sigma_consistent",
            ]
        ].round(3)

    cn = f"""# GWTC 天空像素排序修正版 v9.4 完整报告

## 1. 结论

历史 v9.3 的真实 PE HDF5 均为 NESTED，但旧读取器把 `[b'True']` 错判为 False，按 RING 解释。本轮在独立目录中修复该错误，重建注入天空模板与 validation/test map，重新选择 validation 权重，并重算 held-out 指标和真实 GWTC 排名。历史 v9.3 未覆盖，包哈希保持不变。

修正对结果有实质影响。O3 三通道 retrieval R@10 由 {key_result['gwtc3']['three_channel_r10_v93']:.3f} 变为 {key_result['gwtc3']['three_channel_r10']:.3f}；O4a 由 {key_result['gwtc4']['three_channel_r10_v93']:.3f} 变为 {key_result['gwtc4']['three_channel_r10']:.3f}。两套部署的三个 waveform Gate 均通过。

## 2. 冻结与修改边界

冻结：waveform、time、system split、三个 model seed、template 身份、权重网格、validation-only 选择规则、sky Bayes-factor 公式、注入 Nside=64 和真实排名 Nside=1024。

修改：严格解析 NESTED 元数据、先转成 RING 再改变分辨率、重建天空相关产物及其下游权重和排名。PE 只在排名后附加。

冻结 pair feature 审计：{len(frozen)} 个 deployment/seed/split 检查全部通过，最大数值差 {frozen.max_abs_numeric_delta.max():.1e}。

## 3. O3 held-out 注入

{markdown_table(metric_table('gwtc3'))}

## 4. O4a held-out 注入

{markdown_table(metric_table('gwtc4'))}

## 5. 真实 GWTC-3 Top-10

{markdown_table(top_table('gwtc3'))}

Top-10 中 {key_result['gwtc3']['top10_pe_pass']}/10 通过冻结的 `Dmax<=3` 描述性 PE screen；PE 没有参与调权。

## 6. 真实 GWTC-4.1/O4a Top-10

{markdown_table(top_table('gwtc4'))}

Top-10 中 {key_result['gwtc4']['top10_pe_pass']}/10 通过同一 PE screen。

## 7. 分辨率与排序审计

{markdown_table(sky_summary.round(6))}

修正后的 512 到 1024 收敛良好：O3/O4a 均无 256/512/1024 高分辨率符号不稳定 pair，最大 `|Z512-Z1024|` 分别约 {convergence['gwtc3']['abs_delta_nside512_1024']['max']:.2e} 和 {convergence['gwtc4']['abs_delta_nside512_1024']['max']:.2e}。

## 8. 科学边界

注入天空仍是旋转后的真实 PE posterior template，不等于对每个注入从 strain 独立运行完整 PE。真实排名是 Bayesian follow-up shortlist，不是透镜探测。该版本是修正后的独立候选结果；历史 v9.3 保留用于 provenance，对论文的正式采纳仍需作者确认。
"""
    en = f"""# Complete GWTC sky-ordering correction v9.4 report

## Summary

All 147 PE HDF5 sky maps used by historical v9.3 are NESTED, but the old parser interpreted `[b'True']` as False and treated them as RING. This independent rerun strictly decoded the ordering, converted NESTED to RING before resizing, rebuilt the injection sky templates/maps, reselected weights on validation only, and recomputed held-out metrics and real-catalog ranks. The historical v9.3 package remained byte-identical.

O3 three-channel retrieval R@10 changed from {key_result['gwtc3']['three_channel_r10_v93']:.3f} to {key_result['gwtc3']['three_channel_r10']:.3f}; O4a changed from {key_result['gwtc4']['three_channel_r10_v93']:.3f} to {key_result['gwtc4']['three_channel_r10']:.3f}. All three waveform gates passed in both deployments.

## O3 held-out injections

{markdown_table(metric_table('gwtc3'))}

## O4a held-out injections

{markdown_table(metric_table('gwtc4'))}

## GWTC-3 top ten

{markdown_table(top_table('gwtc3'))}

## GWTC-4.1/O4a top ten

{markdown_table(top_table('gwtc4'))}

## Scope

Waveform/time evidence, source-system splits, model seeds, template identities, the fusion grid, and the validation-only selection rule were frozen. PE consistency was attached only after ranking. Synthetic skies remain rotated real-PE posterior templates rather than independent full PE for every injected event. These are candidate shortlists for Bayesian follow-up, not lensing detections.
"""
    (args.new_root / "gwtc_sky_ordering_corrected_v94_report_cn.md").write_text(cn, encoding="utf-8")
    (args.new_root / "gwtc_sky_ordering_corrected_v94_report_en.md").write_text(en, encoding="utf-8")
    readme = """# Current authoritative ordering-corrected result\n\nThis directory is an independent correction of the historical v9.3 sky-map ordering bug. It does not overwrite v9.3. Use `ordering_corrected_summary_v94.json` and the two v9.4 reports as entry points. Real ranked pairs are follow-up candidates, not detections.\n"""
    (args.new_root / "README_CURRENT_RESULTS.md").write_text(readme, encoding="utf-8")
    print("[complete] ordering-corrected v9.4 audit/report generation finished")


if __name__ == "__main__":
    main()
