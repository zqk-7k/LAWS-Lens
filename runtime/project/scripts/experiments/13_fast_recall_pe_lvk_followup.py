#!/usr/bin/env python3
"""Attach frozen PE and historical-candidate audits to the v10.3 fast pilot.

This is deliberately post-ranking.  Public PE quantities and literature/audit
pair labels are never used to select templates, fusion weights, or ranks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_FAST = Path(
    "results/sky_background_fast_results_v10_20260823_20260823T155259Z"
)
DEFAULT_V93 = Path("results/gwtc_sky_resolution_v93_20260730")
DEFAULT_AUDIT = Path(
    "results/unified_sky_v81_20260725/historical_candidate_rank_audit_v81.csv"
)

METHOD_FILES = {
    "waveform_only": "real_strict_pair_scores_waveform_only.parquet",
    "time_only": "real_strict_pair_scores_time_only.parquet",
    "sky_only": "real_strict_pair_scores_sky_only.parquet",
    "retrieval_three_channel_strict_positive": (
        "real_strict_pair_scores_retrieval_three_channel_strict_positive.parquet"
    ),
    "candidate_three_channel_strict_positive": (
        "real_strict_pair_scores_candidate_three_channel_strict_positive.parquet"
    ),
}

PE_COLUMNS = [
    "pe_available",
    "chirp_mass_median_i",
    "chirp_mass_median_j",
    "chirp_mass_standardized_posterior_distance",
    "chirp_mass_wasserstein_distance",
    "chirp_mass_bhattacharyya_coefficient",
    "mass_ratio_median_i",
    "mass_ratio_median_j",
    "mass_ratio_standardized_posterior_distance",
    "mass_ratio_wasserstein_distance",
    "mass_ratio_bhattacharyya_coefficient",
    "chi_eff_median_i",
    "chi_eff_median_j",
    "chi_eff_standardized_posterior_distance",
    "chi_eff_wasserstein_distance",
    "chi_eff_bhattacharyya_coefficient",
    "max_standardized_posterior_distance",
    "intrinsic_3sigma_consistent",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pair_key(left: object, right: object) -> str:
    return "||".join(sorted((str(left), str(right))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def method_rank_table(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["canonical_pair_key"] = [
        pair_key(left, right) for left, right in zip(frame.event_i, frame.event_j)
    ]
    if "final_score" not in frame:
        raise KeyError(f"final_score missing from {path}")
    frame = frame.sort_values(
        ["final_score", "canonical_pair_key"], ascending=[False, True]
    ).reset_index(drop=True)
    frame[f"{method}_rank"] = np.arange(1, len(frame) + 1)
    frame[f"{method}_score"] = frame.final_score
    keep = [
        "canonical_pair_key",
        f"{method}_rank",
        f"{method}_score",
    ]
    if method == "candidate_three_channel_strict_positive":
        for column in (
            "waveform_score",
            "time_score",
            "sky_score",
            "sky_log_bf_nside512",
            "waveform_contribution",
            "time_contribution",
            "sky_contribution",
            "delta_t_days",
        ):
            if column in frame:
                keep.append(column)
    return frame[keep]


def pe_budget_summary(frame: pd.DataFrame, deployment: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for budget in (10, 20, 50, 100):
        subset = frame[frame["rank"] <= budget]
        available = subset[subset.pe_available.fillna(False).astype(bool)]
        passed = available[
            available.intrinsic_3sigma_consistent.fillna(False).astype(bool)
        ]
        rows.append(
            {
                "deployment": deployment,
                "top_budget": budget,
                "n_pairs": len(subset),
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
                "median_chirp_mass_distance": (
                    float(
                        available.chirp_mass_standardized_posterior_distance.median()
                    )
                    if len(available)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    recall: pd.DataFrame,
    pe_budget: pd.DataFrame,
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
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    shown = [
        "waveform_only",
        "time_only",
        "sky_only",
        "retrieval_three_channel_strict_positive",
    ]
    labels = ["Waveform", "Time", "Sky", "Three-channel"]
    colors = {"gwtc3": "#2878B5", "gwtc4": "#D95F02"}
    width = 0.36
    x = np.arange(len(shown))
    for offset, deployment in zip((-width / 2, width / 2), ("gwtc3", "gwtc4")):
        values = []
        for method in shown:
            row = recall[
                recall.deployment.eq(deployment) & recall.method.eq(method)
            ]
            values.append(float(row.r_at_10.iloc[0]))
        axes[0].bar(
            x + offset,
            values,
            width,
            label=deployment.upper(),
            color=colors[deployment],
            alpha=0.88,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=24, ha="right")
    axes[0].set_ylabel("Held-out companion R@10")
    axes[0].set_ylim(0, 0.58)
    axes[0].legend(frameon=False)

    for deployment, group in pe_budget.groupby("deployment"):
        axes[1].plot(
            group.top_budget,
            group.pe_consistent_fraction,
            marker="o",
            linewidth=1.5,
            color=colors[deployment],
            label=deployment.upper(),
        )
    axes[1].set_xlabel("Real-catalog Top-B")
    axes[1].set_ylabel(r"PE-consistent fraction ($D_{\max}\leq3$)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, loc="best")

    names = [f"{a}\n{b}" for a, b in zip(official.event_i, official.event_j)]
    y = np.arange(len(official))
    axes[2].barh(
        y,
        official.candidate_three_channel_strict_positive_rank,
        color=[colors.get(str(x), "0.5") for x in official.deployment],
        alpha=0.88,
    )
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names, fontsize=6.6)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Current pilot candidate rank")
    axes[2].set_xscale("log")
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.15, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.savefig(output / "figures/fig_fast_recall_pe_lvk_followup.pdf")
    figure.savefig(output / "figures/fig_fast_recall_pe_lvk_followup.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--fast-root", type=Path, default=DEFAULT_FAST)
    parser.add_argument("--v93-root", type=Path, default=DEFAULT_V93)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    fast_root = (project / args.fast_root).resolve() if not args.fast_root.is_absolute() else args.fast_root
    v93_root = (project / args.v93_root).resolve() if not args.v93_root.is_absolute() else args.v93_root
    audit_path = (project / args.audit).resolve() if not args.audit.is_absolute() else args.audit
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for name in ("tables", "figures", "scripts", "manifest"):
        (output / name).mkdir(parents=True, exist_ok=True)

    stage3 = fast_root / "stage3_nside512_map_pilot"
    recall_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    pe_budget_frames: list[pd.DataFrame] = []
    top_frames: dict[str, pd.DataFrame] = {}

    for deployment in ("gwtc3", "gwtc4"):
        seed_root = stage3 / deployment / "seed_202607241"
        retrieval = pd.read_csv(seed_root / "retrieval_metrics.csv")
        recall_frames.append(retrieval[retrieval.split.eq("test")].copy())
        pair_metrics = pd.read_csv(seed_root / "pair_metrics.csv")
        pair_frames.append(pair_metrics[pair_metrics.split.eq("test")].copy())

        top = pd.read_csv(
            seed_root
            / "real_strict_candidate_top100_candidate_three_channel_strict_positive.csv"
        )
        top["canonical_pair_key"] = [
            pair_key(left, right) for left, right in zip(top.event_i, top.event_j)
        ]
        pe = pd.read_parquet(v93_root / deployment / "real_candidate_consensus_with_pe_v93.parquet")
        pe["canonical_pair_key"] = [
            pair_key(left, right) for left, right in zip(pe.event_i, pe.event_j)
        ]
        keep = ["canonical_pair_key", "consensus_rank"] + [
            column for column in PE_COLUMNS if column in pe
        ]
        pe = pe[keep].drop_duplicates("canonical_pair_key").rename(
            columns={"consensus_rank": "historical_v93_consensus_rank"}
        )
        merged = top.merge(pe, on="canonical_pair_key", how="left", validate="one_to_one")
        merged["deployment"] = deployment
        top_frames[deployment] = merged
        write_csv(merged, output / "tables" / f"{deployment}_top100_with_pe.csv")
        merged.to_parquet(output / "tables" / f"{deployment}_top100_with_pe.parquet", index=False)
        write_csv(
            merged[merged["rank"] <= 10],
            output / "tables" / f"{deployment}_top10_with_pe.csv",
        )
        pe_budget_frames.append(pe_budget_summary(merged, deployment))

    recall = pd.concat(recall_frames, ignore_index=True)
    pair_metrics = pd.concat(pair_frames, ignore_index=True)
    pe_budget = pd.concat(pe_budget_frames, ignore_index=True)
    write_csv(recall, output / "tables/heldout_test_retrieval_metrics.csv")
    write_csv(pair_metrics, output / "tables/heldout_test_pair_metrics.csv")
    write_csv(pe_budget, output / "tables/real_candidate_pe_budget_summary.csv")

    official = pd.read_csv(audit_path).copy()
    official["canonical_pair_key"] = [
        pair_key(left, right) for left, right in zip(official.event_i, official.event_j)
    ]
    official["provenance_class"] = official.audit_label.map(
        {
            "historical lensing pair audit": "LVK_literature_candidate_not_confirmed",
            "catalog coincidence audit": "catalog_coincidence_audit_not_official_confirmation",
            "pre-specified O4a pair audit": "project_prespecified_O4a_audit",
        }
    ).fillna("frozen_audit_pair")
    joined: list[pd.DataFrame] = []
    for deployment, group in official.groupby("deployment", sort=False):
        seed_root = stage3 / deployment / "seed_202607241"
        current = group.copy()
        for method, filename in METHOD_FILES.items():
            current = current.merge(
                method_rank_table(seed_root / filename, method),
                on="canonical_pair_key",
                how="left",
                validate="one_to_one",
            )
        pe_source = pd.read_parquet(
            v93_root / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        )
        pe_source["canonical_pair_key"] = [
            pair_key(left, right)
            for left, right in zip(pe_source.event_i, pe_source.event_j)
        ]
        pe_keep = ["canonical_pair_key"] + [
            column for column in PE_COLUMNS if column in pe_source
        ]
        current = current.drop(
            columns=[
                column
                for column in (
                    "intrinsic_3sigma_consistent",
                    "max_standardized_posterior_distance",
                )
                if column in current
            ]
        ).merge(
            pe_source[pe_keep].drop_duplicates("canonical_pair_key"),
            on="canonical_pair_key",
            how="left",
            validate="one_to_one",
        )
        joined.append(current)
    official_current = pd.concat(joined, ignore_index=True)
    write_csv(official_current, output / "tables/frozen_lvk_historical_candidate_comparison.csv")

    make_figure(
        recall[(recall.subset == "overall")],
        pe_budget,
        official_current,
        output,
    )

    key_metrics: dict[str, object] = {}
    for deployment in ("gwtc3", "gwtc4"):
        subset = recall[(recall.deployment == deployment) & (recall.subset == "overall")]
        key_metrics[deployment] = {
            row.method: {
                "r_at_1": float(row.r_at_1),
                "r_at_5": float(row.r_at_5),
                "r_at_10": float(row.r_at_10),
                "median_rank": float(row.median_rank),
            }
            for row in subset.itertuples()
        }
    summary = {
        "schema": "v10.4-fast-recall-pe-lvk-followup-v1",
        "generated_at_utc": utc_now(),
        "source_fast_root": str(fast_root),
        "source_v93_pe_root": str(v93_root),
        "ranking_seed": 202607241,
        "pe_attached_after_ranking": True,
        "pe_used_for_weight_or_rank_selection": False,
        "official_labels_used_for_weight_or_rank_selection": False,
        "key_retrieval_metrics": key_metrics,
        "pe_budget": pe_budget.to_dict(orient="records"),
        "frozen_audit_pairs": official_current[
            [
                "deployment",
                "event_i",
                "event_j",
                "provenance_class",
                "candidate_three_channel_strict_positive_rank",
                "max_standardized_posterior_distance",
                "intrinsic_3sigma_consistent",
            ]
        ].to_dict(orient="records"),
        "limitations": [
            "single model seed exploratory map-matched pilot",
            "synthetic sky uses public-PE-template surrogate rather than full homogeneous PE",
            "candidate ranks are descriptive and are not p-values or detections",
        ],
        "terminal_status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "fast_recall_pe_lvk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    overall = recall[recall.subset.eq("overall")]
    display_methods = [
        "waveform_only",
        "time_only",
        "sky_only",
        "retrieval_three_channel_strict_positive",
        "candidate_three_channel_strict_positive",
    ]
    report = [
        "# v10 快速召回、PE 与历史/LVK 候选对照",
        "",
        f"生成时间：{utc_now()}",
        "",
        "> 本结果来自 Nside=512 map-matched 单-seed探索性 pilot。PE 与候选标签均在排名冻结后附加，不参与权重选择或重排；结果不是透镜探测，也不替代 v9.3。",
        "",
        "## 1. 快速主结论",
        "",
        "历史 O3/O4a 注入天空 null 与真实 GWTC 公开 PE null 不同域；同在 Nside=64 下比较时真实 q99 正尾仍分别膨胀 11.097 和 14.024 倍。A90/KL/SNR 匹配只能解释部分差异。因此，旧注入 validation 选择出的天空权重部署到真实目录存在外推风险。",
        "",
        "## 2. Held-out companion retrieval",
        "",
        "| deployment | method | R@1 | R@5 | R@10 | median rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for deployment in ("gwtc3", "gwtc4"):
        for method in display_methods:
            row = overall[(overall.deployment == deployment) & (overall.method == method)].iloc[0]
            report.append(
                f"| {deployment.upper()} | {method} | {row.r_at_1:.4f} | {row.r_at_5:.4f} | {row.r_at_10:.4f} | {row.median_rank:.1f} |"
            )
    report += [
        "",
        "## 3. 真实目录 Top-B 的 PE 后验一致性",
        "",
        "PE screen 使用冻结的公开 posterior 指标，主摘要为 $D_{max}\\leq3$。它只是排名后物理一致性审计，不是第四个检索通道。",
        "",
        "| deployment | Top-B | PE可用 | 3sigma一致 | 比例 | median Dmax | max Dmax |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pe_budget.itertuples():
        report.append(
            f"| {row.deployment.upper()} | {row.top_budget} | {row.n_pe_available} | {row.n_intrinsic_3sigma_consistent} | {row.pe_consistent_fraction:.3f} | {row.median_dmax:.3f} | {row.max_dmax:.3f} |"
        )
    report += [
        "",
        "## 4. 冻结历史/LVK审计 pair",
        "",
        "| deployment | pair | provenance | 当前三通道rank | waveform rank | time rank | sky rank | Dmax | 3sigma一致 |",
        "|---|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in official_current.itertuples():
        report.append(
            f"| {str(row.deployment).upper()} | {row.event_i}--{row.event_j} | {row.provenance_class} | {int(row.candidate_three_channel_strict_positive_rank)} | {int(row.waveform_only_rank)} | {int(row.time_only_rank)} | {int(row.sky_only_rank)} | {row.max_standardized_posterior_distance:.3f} | {'是' if bool(row.intrinsic_3sigma_consistent) else '否'} |"
        )
    report += [
        "",
        "其中只有 GW170104--GW170814 明确标为 LVK 文献候选对且未被确认；其余两对是冻结的目录/项目审计对，不能称为 LVK 已确认候选。",
        "",
        "## 5. 限制",
        "",
        "- 该结果只有一个 model seed，不能给出训练种子不确定度。",
        "- synthetic sky 仍是公开 PE map template surrogate，不是从每条注入 strain 完整重做同质 PE。",
        "- 正在运行的 12-case 快速相干天空 PE 将独立检验 A90、真值覆盖和分辨率稳定性；不会反馈修改本结果。",
        "",
        "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`",
    ]
    (output / "FAST_RECALL_PE_LVK_REPORT_CN.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    script_target = output / "scripts" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), script_target)
    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and "manifest/files_sha256.csv" not in str(path):
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
    package_sha = sha256(package)
    (package.with_suffix(package.suffix + ".sha256")).write_text(
        f"{package_sha}  {package.name}\n", encoding="ascii"
    )
    print(json.dumps({"output": str(output), "package": str(package), "sha256": package_sha}, indent=2))


if __name__ == "__main__":
    main()
