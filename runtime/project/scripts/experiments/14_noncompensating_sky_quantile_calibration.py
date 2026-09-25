#!/usr/bin/env python3
"""Exploratory target-background calibration for real-GWTC sky scores.

The calibration is deliberately non-compensating: it can remove positive sky
evidence that is excessive relative to the injection-validation null, but it
can never increase a pair's sky score. Waveform/time scores, fusion weights,
splits, labels, and injection retrieval metrics remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROJECT = Path("/root/autodl-tmp/gw-catalog")
DEFAULT_STAGE3 = Path(
    "results/sky_background_fast_results_v10_20260823_20260823T155259Z/"
    "stage3_nside512_map_pilot"
)
DEFAULT_V93 = Path("results/gwtc_sky_resolution_v93_20260730")
DEFAULT_AUDIT = Path(
    "results/unified_sky_v81_20260725/historical_candidate_rank_audit_v81.csv"
)
SEED = 202607241
PE_COLUMNS = [
    "pe_available",
    "chirp_mass_standardized_posterior_distance",
    "mass_ratio_standardized_posterior_distance",
    "chi_eff_standardized_posterior_distance",
    "max_standardized_posterior_distance",
    "intrinsic_3sigma_consistent",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def pair_key(left: object, right: object) -> str:
    return "||".join(sorted((str(left), str(right))))


def resolve(project: Path, path: Path) -> Path:
    return path if path.is_absolute() else project / path


def leave_two_events_out_quantile_map(
    real: pd.DataFrame, injection_null: np.ndarray
) -> pd.DataFrame:
    """Map each real score to the same quantile of the injection null.

    The real-catalog empirical CDF for pair (i,j) excludes every pair touching
    either i or j. This prevents a candidate and its incident edges from
    calibrating their own score.
    """

    output = real.copy()
    raw = output["sky_score"].to_numpy(dtype=np.float64)
    left = output["event_i"].astype(str).to_numpy()
    right = output["event_j"].astype(str).to_numpy()
    target = np.sort(np.asarray(injection_null, dtype=np.float64))
    quantiles = np.empty(len(output), dtype=np.float64)
    mapped = np.empty(len(output), dtype=np.float64)
    reference_sizes = np.empty(len(output), dtype=np.int32)
    for index, (event_i, event_j, score) in enumerate(zip(left, right, raw)):
        keep = (
            (left != event_i)
            & (right != event_i)
            & (left != event_j)
            & (right != event_j)
        )
        reference = np.sort(raw[keep])
        if len(reference) < 100:
            raise RuntimeError("Insufficient leave-two-events-out real null")
        lo = np.searchsorted(reference, score, side="left")
        hi = np.searchsorted(reference, score, side="right")
        midrank = 0.5 * (lo + hi)
        quantile = (midrank + 0.5) / (len(reference) + 1.0)
        quantile = float(np.clip(quantile, 0.5 / (len(reference) + 1.0), 1.0 - 0.5 / (len(reference) + 1.0)))
        quantiles[index] = quantile
        mapped[index] = float(np.quantile(target, quantile, method="linear"))
        reference_sizes[index] = len(reference)
    output["sky_score_raw_log_bf"] = raw
    output["real_null_loo_quantile"] = quantiles
    output["sky_score_full_quantile_map_diagnostic"] = mapped
    output["sky_score_noncompensating"] = np.minimum(raw, mapped)
    output["sky_score_removed"] = raw - output["sky_score_noncompensating"]
    output["sky_score_was_compressed"] = output["sky_score_removed"] > 1e-12
    output["real_null_loo_reference_pairs"] = reference_sizes
    return output


def rerank(
    calibrated: pd.DataFrame, weights: dict[str, float], method: str
) -> pd.DataFrame:
    output = calibrated.copy()
    output["waveform_contribution"] = (
        float(weights["waveform"]) * output["waveform_score"]
    )
    output["time_contribution"] = float(weights["time"]) * output["time_score"]
    output["sky_contribution"] = (
        float(weights["sky"]) * output["sky_score_noncompensating"]
    )
    output["final_score"] = output[
        ["waveform_contribution", "time_contribution", "sky_contribution"]
    ].sum(axis=1)
    output["method"] = method
    output = output.sort_values(
        ["final_score", "pair_key"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1, dtype=np.int32)
    return output


def attach_pe(frame: pd.DataFrame, pe_path: Path) -> pd.DataFrame:
    pe = pd.read_parquet(pe_path).copy()
    pe["canonical_pair_key"] = [
        pair_key(left, right) for left, right in zip(pe.event_i, pe.event_j)
    ]
    keep = ["canonical_pair_key"] + [c for c in PE_COLUMNS if c in pe]
    pe = pe[keep].drop_duplicates("canonical_pair_key")
    output = frame.copy()
    output["canonical_pair_key"] = [
        pair_key(left, right) for left, right in zip(output.event_i, output.event_j)
    ]
    return output.merge(pe, on="canonical_pair_key", how="left", validate="one_to_one")


def pe_budget_summary(
    raw: pd.DataFrame, calibrated: pd.DataFrame, deployment: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, frame in (("raw", raw), ("noncompensating_quantile", calibrated)):
        for budget in (10, 20, 50, 100):
            subset = frame.loc[frame["rank"] <= budget]
            available = subset.loc[subset.pe_available.fillna(False).astype(bool)]
            passed = available.loc[
                available.intrinsic_3sigma_consistent.fillna(False).astype(bool)
            ]
            rows.append(
                {
                    "deployment": deployment,
                    "arm": arm,
                    "top_budget": budget,
                    "n_pe_available": len(available),
                    "n_intrinsic_3sigma_consistent": len(passed),
                    "pe_consistent_fraction": (
                        len(passed) / len(available) if len(available) else np.nan
                    ),
                    "median_dmax": (
                        float(available.max_standardized_posterior_distance.median())
                        if len(available)
                        else np.nan
                    ),
                    "max_dmax": (
                        float(available.max_standardized_posterior_distance.max())
                        if len(available)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def distribution_summary(
    injection: np.ndarray,
    raw: np.ndarray,
    full: np.ndarray,
    noncompensating: np.ndarray,
    deployment: str,
) -> pd.DataFrame:
    rows = []
    quantiles = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
    for name, values in (
        ("injection_validation_null", injection),
        ("real_raw", raw),
        ("real_full_quantile_map_diagnostic", full),
        ("real_noncompensating_quantile", noncompensating),
    ):
        for quantile in quantiles:
            rows.append(
                {
                    "deployment": deployment,
                    "distribution": name,
                    "quantile": quantile,
                    "sky_score": float(np.quantile(values, quantile)),
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def rank_stability(
    raw: pd.DataFrame, calibrated: pd.DataFrame, deployment: str
) -> pd.DataFrame:
    raw_rank = raw.set_index("pair_key")["rank"]
    cal_rank = calibrated.set_index("pair_key")["rank"]
    rows = []
    for budget in (10, 20, 50, 100):
        raw_set = set(raw_rank.loc[raw_rank <= budget].index)
        cal_set = set(cal_rank.loc[cal_rank <= budget].index)
        intersection = len(raw_set & cal_set)
        union = len(raw_set | cal_set)
        rows.append(
            {
                "deployment": deployment,
                "top_budget": budget,
                "overlap_count": intersection,
                "jaccard": intersection / union if union else np.nan,
                "n_moved_in": len(cal_set - raw_set),
                "n_moved_out": len(raw_set - cal_set),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    distributions: pd.DataFrame,
    pe_summary: pd.DataFrame,
    official: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    colors = {
        "injection_validation_null": "#222222",
        "real_raw": "#D95F02",
        "real_noncompensating_quantile": "#2878B5",
    }
    labels = {
        "injection_validation_null": "Injection validation null",
        "real_raw": "Real raw",
        "real_noncompensating_quantile": "Real calibrated",
    }
    for axis, deployment in zip(axes[0], ("gwtc3", "gwtc4")):
        subset = distributions.loc[
            distributions.deployment.eq(deployment)
            & distributions.distribution.isin(colors)
            & distributions["quantile"].lt(1.0)
        ]
        for name, group in subset.groupby("distribution", sort=False):
            axis.plot(
                group["quantile"],
                group.sky_score,
                marker="o",
                markersize=3,
                color=colors[name],
                label=labels[name],
            )
        axis.set_xlabel("Quantile")
        axis.set_ylabel("Sky ranking score")
        axis.set_title(deployment.upper())
        axis.legend(frameon=False, fontsize=7)

    for deployment, color in (("gwtc3", "#2878B5"), ("gwtc4", "#D95F02")):
        subset = pe_summary.loc[
            pe_summary.deployment.eq(deployment)
            & pe_summary.arm.eq("noncompensating_quantile")
        ]
        raw = pe_summary.loc[
            pe_summary.deployment.eq(deployment) & pe_summary.arm.eq("raw")
        ]
        axes[1, 0].plot(
            raw.top_budget,
            raw.pe_consistent_fraction,
            linestyle="--",
            marker="o",
            color=color,
            alpha=0.6,
            label=f"{deployment.upper()} raw",
        )
        axes[1, 0].plot(
            subset.top_budget,
            subset.pe_consistent_fraction,
            linestyle="-",
            marker="o",
            color=color,
            label=f"{deployment.upper()} calibrated",
        )
    axes[1, 0].set_xlabel("Real-catalog Top-B")
    axes[1, 0].set_ylabel(r"PE-consistent fraction ($D_{max}\leq3$)")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(frameon=False, fontsize=6.8)

    y = np.arange(len(official))
    width = 0.36
    axes[1, 1].barh(
        y - width / 2,
        official.raw_rank,
        height=width,
        color="0.65",
        label="Raw",
    )
    axes[1, 1].barh(
        y + width / 2,
        official.calibrated_rank,
        height=width,
        color="#2878B5",
        label="Calibrated",
    )
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels(
        [f"{a}--{b}" for a, b in zip(official.event_i, official.event_j)],
        fontsize=6.5,
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Candidate rank")
    axes[1, 1].legend(frameon=False)
    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontweight="bold")
    figure.savefig(output / "figures/fig_noncompensating_sky_quantile_calibration.pdf")
    figure.savefig(
        output / "figures/fig_noncompensating_sky_quantile_calibration.png",
        dpi=300,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--stage3", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--v93", type=Path, default=DEFAULT_V93)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    stage3 = resolve(project, args.stage3).resolve()
    v93 = resolve(project, args.v93).resolve()
    audit_path = resolve(project, args.audit).resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for directory in ("contracts", "tables", "figures", "scripts", "manifest"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    contract = {
        "schema": "v10.5-noncompensating-sky-quantile-calibration-v1",
        "generated_at_utc": utc_now(),
        "status": "EXPLORATORY_NO_ADOPTION",
        "analysis_nside": 512,
        "ranking_seed": SEED,
        "calibration": "leave-two-events-out real-null percentile mapped to injection-validation null percentile",
        "noncompensating_rule": "min(raw_sky_log_bf, mapped_injection_null_quantile)",
        "real_null_proxy": "all strict H1L1 BBH unordered pairs within each run",
        "injection_null": "synthetic validation non-companion pairs within the same run",
        "waveform_score_frozen": True,
        "time_score_frozen": True,
        "fusion_weights_frozen": True,
        "injection_recall_frozen": True,
        "pe_used_for_calibration": False,
        "candidate_labels_used_for_calibration": False,
        "warning": "calibrated sky score is a ranking score, not a physical log Bayes factor",
        "terminal_status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "contracts/NONCOMPENSATING_SKY_QUANTILE_CONTRACT.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    distribution_frames = []
    pe_frames = []
    stability_frames = []
    official_rows = []
    summary: dict[str, object] = {"contract": contract, "deployments": {}}
    frozen_audit = pd.read_csv(audit_path)

    for deployment in ("gwtc3", "gwtc4"):
        seed_root = stage3 / deployment / f"seed_{SEED}"
        validation = pd.read_parquet(
            seed_root / "synthetic_validation_pairs_nside512_map_matched.parquet"
        )
        injection_null = validation.loc[
            validation.is_true_pair.eq(0), "sky_score"
        ].to_numpy(dtype=np.float64)
        real = pd.read_parquet(
            seed_root / "real_pair_features_nside512_map_matched_pilot.parquet"
        )
        real = real.loc[real.strict_h1l1_bbh_pair].copy().reset_index(drop=True)
        calibrated_base = leave_two_events_out_quantile_map(real, injection_null)
        weights = json.loads(
            (seed_root / "selected_weights_nside512_map_matched.json").read_text()
        )["methods"]

        candidate = rerank(
            calibrated_base,
            weights["candidate_three_channel_strict_positive"],
            "candidate_three_channel_noncompensating_sky_quantile",
        )
        retrieval = rerank(
            calibrated_base,
            weights["retrieval_three_channel_strict_positive"],
            "retrieval_three_channel_noncompensating_sky_quantile",
        )
        candidate.to_parquet(
            output / "tables" / f"{deployment}_real_pair_scores_candidate_noncompensating.parquet",
            index=False,
        )
        retrieval.to_parquet(
            output / "tables" / f"{deployment}_real_pair_scores_retrieval_noncompensating.parquet",
            index=False,
        )

        raw = pd.read_parquet(
            seed_root
            / "real_strict_pair_scores_candidate_three_channel_strict_positive.parquet"
        )
        raw["canonical_pair_key"] = [
            pair_key(left, right) for left, right in zip(raw.event_i, raw.event_j)
        ]
        candidate["canonical_pair_key"] = [
            pair_key(left, right)
            for left, right in zip(candidate.event_i, candidate.event_j)
        ]
        rank_compare = raw[
            ["canonical_pair_key", "event_i", "event_j", "rank", "final_score", "sky_score"]
        ].rename(
            columns={
                "rank": "raw_rank",
                "final_score": "raw_final_score",
                "sky_score": "raw_sky_score",
            }
        ).merge(
            candidate[
                [
                    "canonical_pair_key",
                    "rank",
                    "final_score",
                    "sky_score_noncompensating",
                    "sky_score_full_quantile_map_diagnostic",
                    "sky_score_removed",
                    "sky_score_was_compressed",
                ]
            ].rename(
                columns={"rank": "calibrated_rank", "final_score": "calibrated_final_score"}
            ),
            on="canonical_pair_key",
            how="inner",
            validate="one_to_one",
        )
        rank_compare["rank_change_calibrated_minus_raw"] = (
            rank_compare.calibrated_rank - rank_compare.raw_rank
        )
        write_csv(rank_compare, output / "tables" / f"{deployment}_rank_change_all_pairs.csv")

        pe_path = v93 / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        raw_pe = attach_pe(raw, pe_path)
        candidate_pe = attach_pe(candidate, pe_path)
        write_csv(
            candidate_pe.head(100),
            output / "tables" / f"{deployment}_top100_noncompensating_with_pe.csv",
        )
        candidate_pe.head(100).to_parquet(
            output / "tables" / f"{deployment}_top100_noncompensating_with_pe.parquet",
            index=False,
        )
        write_csv(
            candidate_pe.head(10),
            output / "tables" / f"{deployment}_top10_noncompensating_with_pe.csv",
        )
        pe_frames.append(pe_budget_summary(raw_pe, candidate_pe, deployment))
        stability_frames.append(rank_stability(raw, candidate, deployment))
        distribution_frames.append(
            distribution_summary(
                injection_null,
                calibrated_base.sky_score_raw_log_bf.to_numpy(),
                calibrated_base.sky_score_full_quantile_map_diagnostic.to_numpy(),
                calibrated_base.sky_score_noncompensating.to_numpy(),
                deployment,
            )
        )

        audit = frozen_audit.loc[frozen_audit.deployment.eq(deployment)].copy()
        audit["canonical_pair_key"] = [
            pair_key(left, right) for left, right in zip(audit.event_i, audit.event_j)
        ]
        audit = audit.drop(
            columns=[column for column in PE_COLUMNS if column in audit.columns],
            errors="ignore",
        )
        audit = audit.merge(
            rank_compare,
            on=["canonical_pair_key", "event_i", "event_j"],
            how="left",
            validate="one_to_one",
        )
        pe_keep = ["canonical_pair_key"] + [c for c in PE_COLUMNS if c in candidate_pe]
        audit = audit.merge(
            candidate_pe[pe_keep].drop_duplicates("canonical_pair_key"),
            on="canonical_pair_key",
            how="left",
            validate="one_to_one",
        )
        official_rows.append(audit)

        test_metrics = pd.read_csv(seed_root / "retrieval_metrics.csv")
        test_metrics = test_metrics.loc[test_metrics.split.eq("test")]
        write_csv(
            test_metrics,
            output / "tables" / f"{deployment}_heldout_retrieval_unchanged.csv",
        )
        summary["deployments"][deployment] = {
            "n_injection_validation_null_pairs": len(injection_null),
            "n_real_strict_pairs": len(real),
            "fraction_real_pairs_sky_compressed": float(
                calibrated_base.sky_score_was_compressed.mean()
            ),
            "raw_sky_q99": float(np.quantile(real.sky_score, 0.99)),
            "calibrated_sky_q99": float(
                np.quantile(calibrated_base.sky_score_noncompensating, 0.99)
            ),
            "injection_null_sky_q99": float(np.quantile(injection_null, 0.99)),
            "raw_top_pair": f"{raw.iloc[0].event_i}--{raw.iloc[0].event_j}",
            "calibrated_top_pair": f"{candidate.iloc[0].event_i}--{candidate.iloc[0].event_j}",
            "weights": weights["candidate_three_channel_strict_positive"],
            "heldout_recall_changed": False,
        }

    distributions = pd.concat(distribution_frames, ignore_index=True)
    pe_summary = pd.concat(pe_frames, ignore_index=True)
    stability = pd.concat(stability_frames, ignore_index=True)
    official = pd.concat(official_rows, ignore_index=True)
    write_csv(distributions, output / "tables/sky_distribution_quantiles.csv")
    write_csv(pe_summary, output / "tables/pe_budget_raw_vs_calibrated.csv")
    write_csv(stability, output / "tables/rank_stability_raw_vs_calibrated.csv")
    write_csv(official, output / "tables/frozen_candidate_rank_raw_vs_calibrated.csv")
    summary["pe_budget"] = pe_summary.to_dict(orient="records")
    summary["rank_stability"] = stability.to_dict(orient="records")
    summary["frozen_candidates"] = official[
        [
            "deployment",
            "event_i",
            "event_j",
            "raw_rank",
            "calibrated_rank",
            "max_standardized_posterior_distance",
            "intrinsic_3sigma_consistent",
        ]
    ].to_dict(orient="records")
    summary["terminal_status"] = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
    (output / "noncompensating_sky_quantile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    make_figure(distributions, pe_summary, official, output)

    lines = [
        "# 真实 GWTC 天空高分尾的非补偿分位数校准",
        "",
        f"生成时间：{utc_now()}",
        "",
        "> 本轮是独立探索性 sensitivity arm，不覆盖 v9.3 或 v10 快速基线。校准只减弱过大的真实天空正证据，不增加任何 pair 的天空分。",
        "",
        "## 1. 方法",
        "",
        "对真实 pair `(i,j)`，使用排除所有包含事件 i 或 j 的真实 strict-pair 背景计算其百分位，再映射到同一 run 注入 validation 非伴随背景的相同百分位。正式探索 arm 使用：",
        "",
        "```text",
        "Z_cal = min(Z_raw, injection_null_quantile(real_null_LOO_percentile(Z_raw)))",
        "```",
        "",
        "硬截断到样本最大值没有采用，因为最大值随样本量剧烈变化。完整 quantile map 仅保存为诊断；它可能抬高负分，所以不参与主探索排名。校准后的量是 ranking score，不再解释为物理 log Bayes factor。",
        "",
        "## 2. 分布变化",
        "",
        "| deployment | injection q99 | real raw q99 | real calibrated q99 | compressed fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for deployment in ("gwtc3", "gwtc4"):
        item = summary["deployments"][deployment]
        lines.append(
            f"| {deployment.upper()} | {item['injection_null_sky_q99']:.3f} | {item['raw_sky_q99']:.3f} | {item['calibrated_sky_q99']:.3f} | {item['fraction_real_pairs_sky_compressed']:.3f} |"
        )
    lines += [
        "",
        "## 3. PE Top-B 对照",
        "",
        "| deployment | arm | Top-B | PE consistent | fraction | median Dmax | max Dmax |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pe_summary.itertuples():
        lines.append(
            f"| {row.deployment.upper()} | {row.arm} | {row.top_budget} | {row.n_intrinsic_3sigma_consistent}/{row.n_pe_available} | {row.pe_consistent_fraction:.3f} | {row.median_dmax:.3f} | {row.max_dmax:.3f} |"
        )
    lines += [
        "",
        "## 4. 冻结审计 pair",
        "",
        "| deployment | pair | raw rank | calibrated rank | Dmax | PE consistent |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in official.itertuples():
        lines.append(
            f"| {str(row.deployment).upper()} | {row.event_i}--{row.event_j} | {int(row.raw_rank)} | {int(row.calibrated_rank)} | {row.max_standardized_posterior_distance:.3f} | {'是' if bool(row.intrinsic_3sigma_consistent) else '否'} |"
        )
    lines += [
        "",
        "## 5. 科学边界",
        "",
        "- waveform、time、融合权重和 held-out injection recall 全部冻结；recall 数字因此严格不变。",
        "- 真实目录几乎全为 null 的假设用于无标签背景校准；事件共享导致 pair 并非独立。正式采用前需要 event/catalog-block bootstrap 和多 seed。",
        "- PE 与历史候选标签只在排名后附加，没有参与校准规则。",
        "- 本结果不能解释为 Bayes factor、p-value 或透镜探测。",
        "",
        "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`",
    ]
    (output / "NONCOMPENSATING_SKY_QUANTILE_REPORT_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    shutil.copy2(Path(__file__).resolve(), output / "scripts" / Path(__file__).name)
    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "files_sha256.csv":
            manifest_rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(pd.DataFrame(manifest_rows), output / "manifest/files_sha256.csv")

    package = project / "packages" / f"{output.name}.tar.gz"
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    package_hash = sha256(package)
    package.with_suffix(package.suffix + ".sha256").write_text(
        f"{package_hash}  {package.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {"output": str(output), "package": str(package), "sha256": package_hash},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
