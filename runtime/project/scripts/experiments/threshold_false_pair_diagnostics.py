#!/usr/bin/env python3
"""Threshold-level false-pair diagnostics for catalog lensing search.

This script does not retrain models. It reads existing pair-level score files
and converts ranking scores into thresholded candidate decisions:

  score >= threshold  -> candidate for follow-up
  score < threshold   -> rejected as non-candidate

Outputs separate diagnostics for:
  1. constructed ET-3 full pair catalog with true/false labels;
  2. constructed GWTC real-background + injected lensed pairs;
  3. real GWTC-3/GWTC-4 null/background catalogs, where all pairs are treated
     as background coincidences and no detection claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


FPR_TARGETS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
RECALL_TARGETS = [0.1, 0.5, 0.9]
TOPK_TARGETS = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]
REAL_TOPK_TARGETS = [1, 5, 10, 20, 50, 100, 200, 500]
REAL_TAIL_FRACTIONS = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]


def repo_default() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def threshold_for_topk(scores: np.ndarray, k: int) -> float:
    scores = finite_scores(scores)
    k = int(min(max(k, 1), len(scores)))
    return float(np.partition(scores, len(scores) - k)[len(scores) - k])


def threshold_for_false_rate(false_scores: np.ndarray, target_fpr: float) -> float:
    false_scores = finite_scores(false_scores)
    if len(false_scores) == 0:
        return float("nan")
    q = max(0.0, min(1.0, 1.0 - float(target_fpr)))
    return float(np.quantile(false_scores, q, method="higher"))


def threshold_for_recall(true_scores: np.ndarray, target_recall: float) -> float:
    true_scores = finite_scores(true_scores)
    if len(true_scores) == 0:
        return float("nan")
    q = max(0.0, min(1.0, 1.0 - float(target_recall)))
    return float(np.quantile(true_scores, q, method="lower"))


def labeled_metrics_at_threshold(
    true_scores: np.ndarray,
    false_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    true_scores = finite_scores(true_scores)
    false_scores = finite_scores(false_scores)
    tp = int(np.sum(true_scores >= threshold))
    fp = int(np.sum(false_scores >= threshold))
    n_true = int(len(true_scores))
    n_false = int(len(false_scores))
    candidates = tp + fp
    precision = float(tp / candidates) if candidates else 0.0
    recall = float(tp / n_true) if n_true else float("nan")
    fpr = float(fp / n_false) if n_false else float("nan")
    return {
        "threshold": float(threshold),
        "n_true_pairs": n_true,
        "n_false_pairs": n_false,
        "true_pair_count_above_threshold": tp,
        "false_pair_count_above_threshold": fp,
        "candidate_count_above_threshold": candidates,
        "recall": recall,
        "precision": precision,
        "false_pair_exceedance_rate": fpr,
        "false_per_true_recovered": float(fp / tp) if tp else float("inf"),
    }


def unlabeled_null_metrics_at_threshold(scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    scores = finite_scores(scores)
    n = int(len(scores))
    count = int(np.sum(scores >= threshold))
    return {
        "threshold": float(threshold),
        "n_null_pairs": n,
        "candidate_count_above_threshold": count,
        "null_pair_exceedance_rate": float(count / n) if n else float("nan"),
    }


def quantile_payload(scores: np.ndarray) -> dict[str, float]:
    scores = finite_scores(scores)
    qs = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0]
    return {str(q): float(np.quantile(scores, q)) for q in qs if len(scores)}


def load_et3_scores(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    print(f"Loading ET-3 pair scores from {path}", flush=True)
    df = pd.read_parquet(path, columns=["is_true_pair", "final_score"])
    labels = df["is_true_pair"].to_numpy(dtype=bool)
    scores = df["final_score"].to_numpy(dtype=np.float64)
    true_scores = scores[labels]
    false_scores = scores[~labels]

    # Load only true-pair lens family rows for family recall diagnostics.
    dataset = ds.dataset(path)
    table = dataset.to_table(
        columns=["lens_type", "final_score"],
        filter=ds.field("is_true_pair") == 1,
    )
    true_family = table.to_pandas()
    return true_scores, false_scores, true_family


def build_labeled_threshold_table(
    dataset_name: str,
    score_name: str,
    true_scores: np.ndarray,
    false_scores: np.ndarray,
    all_scores: np.ndarray | None = None,
    include_large_topk: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for target in FPR_TARGETS:
        threshold = threshold_for_false_rate(false_scores, target)
        rows.append(
            {
                "dataset": dataset_name,
                "score_name": score_name,
                "threshold_source": "target_false_pair_exceedance_rate",
                "target_value": target,
                **labeled_metrics_at_threshold(true_scores, false_scores, threshold),
            }
        )
    for target in RECALL_TARGETS:
        threshold = threshold_for_recall(true_scores, target)
        rows.append(
            {
                "dataset": dataset_name,
                "score_name": score_name,
                "threshold_source": "target_recall",
                "target_value": target,
                **labeled_metrics_at_threshold(true_scores, false_scores, threshold),
            }
        )
    if all_scores is not None:
        topks = TOPK_TARGETS if include_large_topk else REAL_TOPK_TARGETS
        for k in topks:
            if k <= len(all_scores):
                threshold = threshold_for_topk(all_scores, k)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "score_name": score_name,
                        "threshold_source": "top_k_candidates",
                        "target_value": k,
                        **labeled_metrics_at_threshold(true_scores, false_scores, threshold),
                    }
                )
    return pd.DataFrame(rows)


def et3_family_recall_table(true_family: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fam_scores = {
        str(fam): sub["final_score"].to_numpy(dtype=np.float64)
        for fam, sub in true_family.groupby("lens_type")
    }
    for _, row in thresholds.iterrows():
        threshold = float(row["threshold"])
        for fam, scores in fam_scores.items():
            rows.append(
                {
                    "dataset": row["dataset"],
                    "threshold_source": row["threshold_source"],
                    "target_value": row["target_value"],
                    "threshold": threshold,
                    "lens_type": fam,
                    "n_true_pairs": int(len(scores)),
                    "true_pair_count_above_threshold": int(np.sum(scores >= threshold)),
                    "recall": float(np.mean(scores >= threshold)) if len(scores) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def gwtc_injection_diagnostics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Loading GWTC injection pair diagnostics from {path}", flush=True)
    df = pd.read_csv(path)
    score_col = "combined_time_sky"
    true_scores = df.loc[df["is_true_pair"].astype(bool), score_col].to_numpy(dtype=np.float64)
    false_scores = df.loc[~df["is_true_pair"].astype(bool), score_col].to_numpy(dtype=np.float64)
    all_scores = df[score_col].to_numpy(dtype=np.float64)
    table = build_labeled_threshold_table(
        "gwtc3_real_background_plus_synthetic_injections",
        score_col,
        true_scores,
        false_scores,
        all_scores=all_scores,
        include_large_topk=False,
    )

    by_seed_rows = []
    # Use thresholds selected on pooled seeds, then report per-seed variability.
    chosen = table[table["threshold_source"].isin(["target_false_pair_exceedance_rate", "target_recall"])].copy()
    for seed, sub in df.groupby("seed"):
        ts = sub.loc[sub["is_true_pair"].astype(bool), score_col].to_numpy(dtype=np.float64)
        fs = sub.loc[~sub["is_true_pair"].astype(bool), score_col].to_numpy(dtype=np.float64)
        for _, th in chosen.iterrows():
            by_seed_rows.append(
                {
                    "dataset": "gwtc3_real_background_plus_synthetic_injections",
                    "seed": int(seed),
                    "score_name": score_col,
                    "threshold_source": th["threshold_source"],
                    "target_value": th["target_value"],
                    **labeled_metrics_at_threshold(ts, fs, float(th["threshold"])),
                }
            )

    class_rows = []
    for _, th in chosen.iterrows():
        false = df[~df["is_true_pair"].astype(bool)].copy()
        false["above_threshold"] = false[score_col] >= float(th["threshold"])
        for pair_class, sub in false.groupby("pair_class"):
            class_rows.append(
                {
                    "dataset": "gwtc3_real_background_plus_synthetic_injections",
                    "score_name": score_col,
                    "threshold_source": th["threshold_source"],
                    "target_value": th["target_value"],
                    "threshold": float(th["threshold"]),
                    "pair_class": pair_class,
                    "n_false_pairs": int(len(sub)),
                    "false_pair_count_above_threshold": int(sub["above_threshold"].sum()),
                    "false_pair_exceedance_rate": float(sub["above_threshold"].mean()) if len(sub) else float("nan"),
                }
            )
    return table, pd.DataFrame(by_seed_rows), pd.DataFrame(class_rows)


def real_catalog_tables(real_paths: dict[str, Path], et_thresholds: Iterable[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cross_rows = []
    for name, path in real_paths.items():
        print(f"Loading real null catalog scores for {name}: {path}", flush=True)
        df = pd.read_parquet(path, columns=["final_score", "event_i", "event_j", "rank"])
        scores = df["final_score"].to_numpy(dtype=np.float64)
        n = len(scores)
        for k in REAL_TOPK_TARGETS:
            if k <= n:
                threshold = threshold_for_topk(scores, k)
                rows.append(
                    {
                        "dataset": name,
                        "score_name": "final_score",
                        "threshold_source": "top_k_candidates_within_real_catalog",
                        "target_value": k,
                        **unlabeled_null_metrics_at_threshold(scores, threshold),
                    }
                )
        for frac in REAL_TAIL_FRACTIONS:
            k = max(1, int(math.ceil(frac * n)))
            threshold = threshold_for_topk(scores, k)
            rows.append(
                {
                    "dataset": name,
                    "score_name": "final_score",
                    "threshold_source": "catalog_tail_fraction_within_real_catalog",
                    "target_value": frac,
                    **unlabeled_null_metrics_at_threshold(scores, threshold),
                }
            )
        for threshold in et_thresholds:
            cross_rows.append(
                {
                    "dataset": name,
                    "score_name": "final_score",
                    "threshold_source": "et3_synthetic_score_threshold_applied_without_recalibration",
                    "target_value": float(threshold),
                    **unlabeled_null_metrics_at_threshold(scores, float(threshold)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(cross_rows)


def write_report(
    path: Path,
    summary: dict,
    et_table: pd.DataFrame,
    inj_table: pd.DataFrame,
    real_table: pd.DataFrame,
) -> None:
    def md(df: pd.DataFrame, n: int = 12) -> str:
        show = df.head(n).copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "inf" if np.isinf(x) else f"{x:.6g}")
        return show.to_markdown(index=False)

    text = f"""# 阈值化误配率诊断

本报告把 catalog ranking score 转换成阈值决策：

```text
score >= threshold  -> candidate for Bayesian follow-up
score < threshold   -> rejected as non-candidate
```

核心目的是补充 R@K 召回率之外的 false-pair burden。对构造目录，真对和假对标签已知，因此可以同时报告 recall、precision 和 false-pair exceedance rate。对真实 GWTC catalog，不声称存在真实透镜对，所有 pair 均按 null/background 处理，只报告超过阈值的候选数量和目录尾部比例。

## 数据集摘要

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

## 1. Constructed ET-3 full pair catalog

ET-3 构造目录包含完整 unordered pair 空间。这里的 false-pair exceedance rate 是在去掉真实透镜 companion pair 后，只对 false pairs 计算：

```text
false_pair_exceedance_rate = # false pairs with score >= threshold / # all false pairs
```

示例阈值表：

{md(et_table)}

## 2. GWTC real background + synthetic injected pairs

该构造数据把真实 GWTC 背景事件作为 null catalog，再注入 synthetic lensed pairs。它用于估计真实背景下的 candidate burden，但 injected pairs 仍是 synthetic，不能解释为真实透镜发现。

示例阈值表：

{md(inj_table)}

## 3. Real GWTC-3/GWTC-4 null catalogs

真实 GWTC 部署没有已知真透镜标签。默认口径是：所有真实事件对均为 background/null coincidence。阈值以上的 pair 只能叫 candidate shortlist for Bayesian follow-up。

示例阈值表：

{md(real_table)}

## 推荐论文表述

可以写：

> We report retrieval recall for known injected/simulated lensed pairs and separately quantify false-association burden by thresholding the final pair score. For labeled synthetic catalogs, false-pair exceedance is computed after removing true companion pairs. For real GWTC deployments, all catalog pairs are treated as a null/background population; threshold exceedances define a candidate shortlist for Bayesian follow-up and are not interpreted as detections.

不要写：

> thresholded GWTC pairs are confirmed lenses.

## 解释

- R@K 回答“真透镜 pair 能否被找回”。
- false-pair exceedance 回答“如果设置阈值，会留下多少假候选”。
- 真实 GWTC 只能报告 null/background exceedance 和 shortlist rank，不能计算真实 precision 或 FDR。
"""
    path.write_text(text, encoding="utf-8")


def make_figure(
    out_pdf: Path,
    out_png: Path,
    et_false_scores: np.ndarray,
    et_true_scores: np.ndarray,
    et_table: pd.DataFrame,
    inj_table: pd.DataFrame,
    real_table: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    # Panel A: false-only exceedance curve.
    false_sorted = np.sort(finite_scores(et_false_scores))
    m = len(false_sorted)
    idx = np.unique(np.linspace(0, m - 1, min(2500, m)).astype(int))
    thresholds = false_sorted[idx]
    exceed = (m - idx) / m
    axes[0].plot(thresholds, exceed, color="#344054", lw=1.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("ET-3 final score threshold")
    axes[0].set_ylabel("False-pair exceedance rate")
    axes[0].set_title("A. False-only tail")
    axes[0].grid(True, alpha=0.25)

    # Panel B: recall vs false candidate count for selected ET thresholds.
    sub = et_table[et_table["threshold_source"] == "target_false_pair_exceedance_rate"].copy()
    axes[1].plot(
        sub["false_pair_count_above_threshold"],
        sub["recall"],
        marker="o",
        color="#175CD3",
        lw=1.6,
        label="ET-3 thresholds",
    )
    for _, row in sub.iterrows():
        axes[1].annotate(
            f"{row['target_value']:.0e}",
            (row["false_pair_count_above_threshold"], row["recall"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("False candidates above threshold")
    axes[1].set_ylabel("True-pair recall")
    axes[1].set_title("B. Recall vs false burden")
    axes[1].grid(True, alpha=0.25)

    # Panel C: real null catalog tail counts.
    sub = real_table[real_table["threshold_source"] == "catalog_tail_fraction_within_real_catalog"].copy()
    for name, g in sub.groupby("dataset"):
        g = g.sort_values("target_value")
        axes[2].plot(
            g["target_value"],
            g["candidate_count_above_threshold"],
            marker="o",
            lw=1.6,
            label=name,
        )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Real-catalog tail fraction")
    axes[2].set_ylabel("Null pairs retained")
    axes[2].set_title("C. Real GWTC null shortlist size")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def package_outputs(package_path: Path, files: list[Path], base_dir: Path) -> None:
    ensure(package_path.parent)
    with tarfile.open(package_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(base_dir)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_default())
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir or (repo / "results" / "threshold_false_pair_diagnostics_20260701")
    ensure(out_dir)
    ensure(out_dir / "figures")
    ensure(repo / "packages")

    et_path = repo / "results" / "pair_level_full_curves" / "et3_pair_scores_full.parquet"
    inj_path = repo / "data" / "gwtc_injection_pair_diagnostics.csv"
    real_paths = {
        "real_gwtc3_null_catalog": repo / "runs" / "real_gwtc_lensing_search_20260625" / "results" / "real_pair_scores_waveform_time_sky.parquet",
        "real_gwtc4p1_null_catalog": repo / "runs" / "real_gwtc34_lensing_search_20260629_full_o4" / "results" / "gwtc4_pair_scores_waveform_time_sky.parquet",
    }

    true_scores, false_scores, true_family = load_et3_scores(et_path)
    et_all_scores = np.concatenate([true_scores, false_scores])
    et_table = build_labeled_threshold_table(
        "constructed_et3_full_pair_catalog",
        "final_score",
        true_scores,
        false_scores,
        all_scores=et_all_scores,
        include_large_topk=True,
    )
    et_table.to_csv(out_dir / "constructed_et3_threshold_diagnostics.csv", index=False)
    et_table[et_table["threshold_source"] == "target_false_pair_exceedance_rate"].to_csv(
        out_dir / "constructed_et3_false_only_thresholds.csv",
        index=False,
    )
    et_family = et3_family_recall_table(true_family, et_table)
    et_family.to_csv(out_dir / "constructed_et3_family_recall_by_threshold.csv", index=False)

    inj_table, inj_seed, inj_class = gwtc_injection_diagnostics(inj_path)
    inj_table.to_csv(out_dir / "gwtc_injection_threshold_diagnostics.csv", index=False)
    inj_seed.to_csv(out_dir / "gwtc_injection_threshold_diagnostics_by_seed.csv", index=False)
    inj_class.to_csv(out_dir / "gwtc_injection_false_by_pair_class.csv", index=False)

    et_fpr_thresholds = et_table.loc[
        et_table["threshold_source"] == "target_false_pair_exceedance_rate",
        "threshold",
    ].to_numpy(dtype=np.float64)
    real_table, real_cross = real_catalog_tables(real_paths, et_fpr_thresholds)
    real_table.to_csv(out_dir / "real_gwtc_null_threshold_diagnostics.csv", index=False)
    real_cross.to_csv(out_dir / "real_gwtc_cross_applied_et3_score_thresholds.csv", index=False)

    summary = {
        "constructed_et3": {
            "n_true_pairs": int(len(true_scores)),
            "n_false_pairs": int(len(false_scores)),
            "n_total_pairs": int(len(true_scores) + len(false_scores)),
            "true_score_quantiles": quantile_payload(true_scores),
            "false_score_quantiles": quantile_payload(false_scores),
        },
        "gwtc_injection": {
            "path": str(inj_path.relative_to(repo)),
            "n_rows": int(pd.read_csv(inj_path, usecols=["is_true_pair"]).shape[0]),
            "n_true_rows": int(pd.read_csv(inj_path, usecols=["is_true_pair"])["is_true_pair"].astype(bool).sum()),
        },
        "real_catalogs": {
            name: {
                "path": str(path.relative_to(repo)),
                "n_pairs": int(pd.read_parquet(path, columns=["final_score"]).shape[0]),
                "score_quantiles": quantile_payload(pd.read_parquet(path, columns=["final_score"])["final_score"].to_numpy(dtype=np.float64)),
            }
            for name, path in real_paths.items()
        },
        "interpretation": {
            "labeled_false_rate": "computed after removing true companion pairs",
            "real_catalog_rate": "all real pairs are treated as null/background; exceedances are candidate shortlist entries, not detections",
            "cross_applied_et3_thresholds": "not a calibrated real-catalog significance; included only as a score-scale diagnostic",
        },
    }
    (out_dir / "threshold_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figure(
        out_dir / "figures" / "fig_threshold_false_pair_burden.pdf",
        out_dir / "figures" / "fig_threshold_false_pair_burden.png",
        false_scores,
        true_scores,
        et_table,
        inj_table,
        real_table,
    )

    write_report(
        out_dir / "threshold_false_pair_diagnostics_cn.md",
        summary,
        et_table,
        inj_table,
        real_table,
    )

    files = [
        out_dir / "constructed_et3_threshold_diagnostics.csv",
        out_dir / "constructed_et3_false_only_thresholds.csv",
        out_dir / "constructed_et3_family_recall_by_threshold.csv",
        out_dir / "gwtc_injection_threshold_diagnostics.csv",
        out_dir / "gwtc_injection_threshold_diagnostics_by_seed.csv",
        out_dir / "gwtc_injection_false_by_pair_class.csv",
        out_dir / "real_gwtc_null_threshold_diagnostics.csv",
        out_dir / "real_gwtc_cross_applied_et3_score_thresholds.csv",
        out_dir / "threshold_diagnostics_summary.json",
        out_dir / "threshold_false_pair_diagnostics_cn.md",
        out_dir / "figures" / "fig_threshold_false_pair_burden.pdf",
        out_dir / "figures" / "fig_threshold_false_pair_burden.png",
        repo / "scripts" / "experiments" / "threshold_false_pair_diagnostics.py",
    ]
    package_path = repo / "packages" / "threshold_false_pair_diagnostics_20260701.tar.gz"
    package_outputs(package_path, files, repo)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "package": str(package_path),
                "et3_n_true_pairs": int(len(true_scores)),
                "et3_n_false_pairs": int(len(false_scores)),
                "real_tables": {k: str(v) for k, v in real_paths.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
