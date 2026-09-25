#!/usr/bin/env python3
"""Audit injected true-pair, injected-null, and real-catalog sky scores."""

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
from scipy.stats import rankdata


PROJECT = Path("/root/autodl-tmp/gw-catalog")
STAGE3 = Path(
    "results/sky_background_fast_results_v10_20260823_20260823T155259Z/"
    "stage3_nside512_map_pilot"
)
SEED = 202607241


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def auc_score(labels: np.ndarray, values: np.ndarray) -> float:
    labels = labels.astype(bool)
    ranks = rankdata(values, method="average")
    n_true = int(labels.sum())
    n_null = int((~labels).sum())
    return float(
        (ranks[labels].sum() - n_true * (n_true + 1) / 2) / (n_true * n_null)
    )


def distribution_row(
    deployment: str,
    split: str,
    subset: str,
    values: np.ndarray,
    null_q99: float,
    auc: float | None,
) -> dict[str, object]:
    return {
        "deployment": deployment,
        "split": split,
        "subset": subset,
        "n": len(values),
        "min": float(np.min(values)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "fraction_positive": float(np.mean(values > 0)),
        "fraction_above_injection_null_q99": float(np.mean(values > null_q99)),
        "true_vs_null_roc_auc": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--stage3", type=Path, default=STAGE3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    stage3 = args.stage3 if args.stage3.is_absolute() else project / args.stage3
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for name in ("tables", "figures", "scripts", "manifest"):
        (output / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    plot_data: dict[str, dict[str, np.ndarray]] = {}
    for deployment in ("gwtc3", "gwtc4"):
        seed_root = stage3 / deployment / f"seed_{SEED}"
        for split in ("validation", "test"):
            frame = pd.read_parquet(
                seed_root / f"synthetic_{split}_pairs_nside512_map_matched.parquet"
            )
            labels = frame.is_true_pair.to_numpy(dtype=bool)
            values = frame.sky_score.to_numpy(dtype=np.float64)
            null_q99 = float(np.quantile(values[~labels], 0.99))
            auc = auc_score(labels, values)
            masks = {
                "injection_null": ~labels,
                "injection_true_all": labels,
                "injection_true_sis": labels
                & frame.true_pair_family.astype(str).str.lower().eq("sis").to_numpy(),
                "injection_true_pm": labels
                & frame.true_pair_family.astype(str).str.lower().eq("pm").to_numpy(),
            }
            for subset, mask in masks.items():
                rows.append(
                    distribution_row(
                        deployment,
                        split,
                        subset,
                        values[mask],
                        null_q99,
                        auc if subset == "injection_true_all" else None,
                    )
                )
            if split == "test":
                plot_data[deployment] = {
                    "null": values[~labels],
                    "true": values[labels],
                }

        test = pd.read_parquet(
            seed_root / "synthetic_test_pairs_nside512_map_matched.parquet"
        )
        true = test.loc[test.is_true_pair.eq(1), "sky_score"].to_numpy(dtype=np.float64)
        real = pd.read_parquet(
            seed_root / "real_pair_features_nside512_map_matched_pilot.parquet"
        )
        real = real.loc[real.strict_h1l1_bbh_pair, "sky_score"].to_numpy(dtype=np.float64)
        comparisons.append(
            {
                "deployment": deployment,
                "n_injection_true_pairs": len(true),
                "injection_true_median": float(np.median(true)),
                "injection_true_q99": float(np.quantile(true, 0.99)),
                "injection_true_max": float(np.max(true)),
                "n_real_strict_pairs": len(real),
                "real_strict_q99": float(np.quantile(real, 0.99)),
                "n_real_pairs_above_injection_true_max": int(np.sum(real > np.max(true))),
                "fraction_real_pairs_above_injection_true_max": float(
                    np.mean(real > np.max(true))
                ),
            }
        )
        plot_data[deployment]["real"] = real

    distributions = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    write_csv(distributions, output / "tables/injection_true_null_sky_distribution.csv")
    write_csv(comparison, output / "tables/real_vs_injection_true_tail.csv")

    inventory = pd.DataFrame(
        [
            {
                "category": "completed_ranking",
                "scheme": "historical_v9.3_raw_log_bf",
                "status": "historical baseline",
                "uses_true_pair_distribution": "indirectly in validation weight selection",
                "uses_unlensed_background": "indirectly in validation metrics",
            },
            {
                "category": "completed_ranking",
                "scheme": "Nside512_map_matched_raw_three_channel",
                "status": "exploratory result",
                "uses_true_pair_distribution": "indirectly in validation weight selection",
                "uses_unlensed_background": "indirectly in validation metrics",
            },
            {
                "category": "completed_ranking",
                "scheme": "noncompensating_real_to_injection_null_quantile",
                "status": "sensitivity result; no adoption",
                "uses_true_pair_distribution": "no",
                "uses_unlensed_background": "explicitly; injection null plus real null proxy",
            },
            {
                "category": "running_sky_validation",
                "scheme": "12_case_rapid_coherent_sky_PE",
                "status": "running",
                "uses_true_pair_distribution": "selected injected companion cases only",
                "uses_unlensed_background": "selected unlensed cases only",
            },
            {
                "category": "running_sky_validation",
                "scheme": "4_case_strict_full_PE",
                "status": "running",
                "uses_true_pair_distribution": "selected injected companion cases only",
                "uses_unlensed_background": "not a catalog calibrator",
            },
            {
                "category": "planned_formal_ranking",
                "scheme": "one_stage_eta_0_0.25_0.5_1.0",
                "status": "four planned arms; not run",
                "uses_true_pair_distribution": "validation only",
                "uses_unlensed_background": "explicit conditional calibrator planned",
            },
            {
                "category": "planned_formal_ranking",
                "scheme": "two_stage_wt_only",
                "status": "planned; not run",
                "uses_true_pair_distribution": "validation only",
                "uses_unlensed_background": "explicit conditional FPP/LLR planned",
            },
            {
                "category": "planned_formal_ranking",
                "scheme": "two_stage_full",
                "status": "planned; not run",
                "uses_true_pair_distribution": "validation only",
                "uses_unlensed_background": "explicit conditional FPP/LLR planned",
            },
        ]
    )
    write_csv(inventory, output / "tables/scheme_inventory.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelweight": "bold",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    bins = np.linspace(-2.5, 3.5, 100)
    rng = np.random.default_rng(20260824)
    for axis, deployment in zip(axes, ("gwtc3", "gwtc4")):
        data = plot_data[deployment]
        null = data["null"]
        if len(null) > 20000:
            null = rng.choice(null, 20000, replace=False)
        axis.hist(null, bins=bins, density=True, histtype="step", linewidth=1.2, color="0.2", label="Injection null")
        axis.hist(data["true"], bins=bins, density=True, histtype="step", linewidth=1.7, color="#2878B5", label="Injection true")
        axis.hist(data["real"], bins=bins, density=True, histtype="step", linewidth=1.2, color="#D95F02", label="Real strict pairs")
        axis.set_title(deployment.upper())
        axis.set_xlabel(r"$Z_{sky}=\log B_{sky}$")
        axis.set_ylabel("Density")
        axis.legend(frameon=False, fontsize=7)
    figure.savefig(output / "figures/fig_true_null_real_sky_distribution.pdf")
    figure.savefig(output / "figures/fig_true_null_real_sky_distribution.png", dpi=300)
    plt.close(figure)

    test_rows = distributions.loc[
        distributions.split.eq("test")
        & distributions.subset.isin(["injection_null", "injection_true_all"])
    ]
    lines = [
        "# 注入真对、无透镜背景与真实目录天空分数审计",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "> 真实 GWTC 没有已确认的透镜真对。本报告中的真对仅指 held-out synthetic companion pairs；真实 strict pairs 是低透镜率下的背景代理，不是真值标签。",
        "",
        "## Held-out test 分布",
        "",
        "| deployment | subset | n | median | q90 | q99 | max | positive | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows.itertuples():
        auc = "" if pd.isna(row.true_vs_null_roc_auc) else f"{row.true_vs_null_roc_auc:.3f}"
        lines.append(
            f"| {row.deployment.upper()} | {row.subset} | {row.n} | {row.median:.3f} | {row.q90:.3f} | {row.q99:.3f} | {row.max:.3f} | {row.fraction_positive:.3f} | {auc} |"
        )
    lines += [
        "",
        "## 真实目录异常正尾",
        "",
        "| deployment | injection true max | real q99 | real pairs above true max | fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.deployment.upper()} | {row.injection_true_max:.3f} | {row.real_strict_q99:.3f} | {row.n_real_pairs_above_injection_true_max} | {row.fraction_real_pairs_above_injection_true_max:.3f} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "原始 log B_sky 的真对和 null 分布高度重叠，O3/O4a held-out AUC 仅约 0.66/0.60。真实目录还出现大量超过注入真对最大值的高分 pair。因此必须显式利用无透镜背景控制候选尾部，而不能把高 log B_sky 直接解释为可靠透镜支持。",
        "",
        "优先方案是按 run、detector network、A90/KL 和地图形态条件化的 null-tail FPP 或 sky likelihood ratio，并用 source/event-disjoint cross-fitting。当前非补偿分位数 arm 已显式使用 null，但没有利用真对密度；原始 v9.3 和 Nside512 加权和只在 validation 调权时间接使用 null。",
        "",
        "当前已有三个可排名结果（一个历史基线、两个新探索 arm），另有两个 PE 管线正在验证。正式 v10 的四个 one-stage eta arms 与两个 two-stage arms尚未运行。",
        "",
        "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`",
    ]
    (output / "TRUE_NULL_SKY_DISTRIBUTION_AUDIT_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    summary = {
        "schema": "true-null-sky-distribution-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_nside": 512,
        "seed": SEED,
        "real_catalog_has_confirmed_true_lens_pairs": False,
        "comparison": comparison.to_dict(orient="records"),
        "completed_ranking_results": 3,
        "running_pe_validation_pipelines": 2,
        "planned_formal_ranking_arms_not_run": 6,
        "terminal_status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "true_null_sky_distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(Path(__file__).resolve(), output / "scripts" / Path(__file__).name)
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "files_sha256.csv":
            manifest.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(pd.DataFrame(manifest), output / "manifest/files_sha256.csv")
    package = project / "packages" / f"{output.name}.tar.gz"
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    digest = sha256(package)
    package.with_suffix(package.suffix + ".sha256").write_text(
        f"{digest}  {package.name}\n", encoding="ascii"
    )
    print(json.dumps({"output": str(output), "package": str(package), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
