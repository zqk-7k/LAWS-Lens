#!/usr/bin/env python3
"""SNR-stratified diagnostics for catalog-level lensing search.

The main split is:
  - low_snr_lt8: SNR < 8
  - high_snr_ge8: SNR >= 8

For simulated ET-3, where true companion pairs are known, this script reports
query-level retrieval metrics and threshold-level false-pair burden by SNR.

For real GWTC-3/GWTC-4, where no true lenses are claimed, it reports null
catalog candidate composition by pair minimum SNR.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SNR_CUT = 8.0
METHODS = ["waveform_score", "time_score", "sky_score", "final_score"]
FPR_TARGETS = [1e-2, 1e-3, 1e-4, 1e-5]
RECALL_TARGETS = [0.1, 0.5, 0.9]
REAL_TOPK = [1, 5, 10, 20, 50, 100, 200]
REAL_TAIL_FRACTIONS = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]


def repo_default() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def snr_group(values: np.ndarray, cut: float = SNR_CUT) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values < cut, "low_snr_lt8", "high_snr_ge8")


def retrieval_metrics(ranks: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.float64)
    ranks = ranks[np.isfinite(ranks)]
    if len(ranks) == 0:
        return {
            "n_queries": 0,
            "recall_at_1": float("nan"),
            "recall_at_5": float("nan"),
            "recall_at_10": float("nan"),
            "recall_at_50": float("nan"),
            "median_rank": float("nan"),
            "mean_rank": float("nan"),
            "mrr": float("nan"),
        }
    return {
        "n_queries": int(len(ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "recall_at_50": float(np.mean(ranks <= 50)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def add_metric_rows(rows: list[dict], df: pd.DataFrame, method: str, grouping_kind: str, group_col: str) -> None:
    for subset in ["all", "SIS", "PM"]:
        sub0 = df if subset == "all" else df[df["lens_type"] == subset]
        for group in ["all", "low_snr_lt8", "high_snr_ge8"]:
            sub = sub0 if group == "all" else sub0[sub0[group_col] == group]
            rows.append(
                {
                    "method": method,
                    "grouping_kind": grouping_kind,
                    "snr_group": group,
                    "lens_subset": subset,
                    **retrieval_metrics(sub["rank"].to_numpy(dtype=np.float64)),
                }
            )


def load_true_pairs(pair_path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        pair_path,
        filters=[("is_true_pair", "=", 1)],
        columns=["pair_i", "pair_j", "event_i", "event_j", "lens_type"],
    )


def compute_directed_ranks_by_method(
    pair_path: Path,
    true_pairs: pd.DataFrame,
    event_snr: np.ndarray,
    n_events: int,
    method: str,
) -> pd.DataFrame:
    print(f"Computing ET-3 directed ranks for {method}", flush=True)
    df = pd.read_parquet(pair_path, columns=["pair_i", "pair_j", method])
    ii = df["pair_i"].to_numpy(dtype=np.int32)
    jj = df["pair_j"].to_numpy(dtype=np.int32)
    score = df[method].to_numpy(dtype=np.float32)
    mat = np.full((n_events, n_events), -np.inf, dtype=np.float32)
    mat[ii, jj] = score
    mat[jj, ii] = score
    np.fill_diagonal(mat, -np.inf)
    del df, ii, jj, score
    gc.collect()

    rows = []
    for row in true_pairs.itertuples(index=False):
        a = int(row.pair_i)
        b = int(row.pair_j)
        s_ab = float(mat[a, b])
        rank_ab = int(1 + np.sum(mat[a] > s_ab))
        s_ba = float(mat[b, a])
        rank_ba = int(1 + np.sum(mat[b] > s_ba))
        min_snr = float(min(event_snr[a], event_snr[b]))
        max_snr = float(max(event_snr[a], event_snr[b]))
        rows.append(
            {
                "method": method,
                "query_idx": a,
                "partner_idx": b,
                "query_snr": float(event_snr[a]),
                "partner_snr": float(event_snr[b]),
                "pair_min_snr": min_snr,
                "pair_max_snr": max_snr,
                "query_snr_group": "low_snr_lt8" if event_snr[a] < SNR_CUT else "high_snr_ge8",
                "pair_min_snr_group": "low_snr_lt8" if min_snr < SNR_CUT else "high_snr_ge8",
                "lens_type": row.lens_type,
                "rank": rank_ab,
                "true_score": s_ab,
            }
        )
        rows.append(
            {
                "method": method,
                "query_idx": b,
                "partner_idx": a,
                "query_snr": float(event_snr[b]),
                "partner_snr": float(event_snr[a]),
                "pair_min_snr": min_snr,
                "pair_max_snr": max_snr,
                "query_snr_group": "low_snr_lt8" if event_snr[b] < SNR_CUT else "high_snr_ge8",
                "pair_min_snr_group": "low_snr_lt8" if min_snr < SNR_CUT else "high_snr_ge8",
                "lens_type": row.lens_type,
                "rank": rank_ba,
                "true_score": s_ba,
            }
        )
    del mat
    gc.collect()
    return pd.DataFrame(rows)


def labeled_threshold_metrics(true_scores: np.ndarray, false_scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    true_scores = np.asarray(true_scores, dtype=np.float64)
    false_scores = np.asarray(false_scores, dtype=np.float64)
    true_scores = true_scores[np.isfinite(true_scores)]
    false_scores = false_scores[np.isfinite(false_scores)]
    tp = int(np.sum(true_scores >= threshold))
    fp = int(np.sum(false_scores >= threshold))
    n_true = int(len(true_scores))
    n_false = int(len(false_scores))
    cand = tp + fp
    return {
        "threshold": float(threshold),
        "n_true_pairs": n_true,
        "n_false_pairs": n_false,
        "true_pair_count_above_threshold": tp,
        "false_pair_count_above_threshold": fp,
        "candidate_count_above_threshold": cand,
        "recall": float(tp / n_true) if n_true else float("nan"),
        "precision": float(tp / cand) if cand else float("nan"),
        "false_pair_exceedance_rate": float(fp / n_false) if n_false else float("nan"),
    }


def threshold_for_false_rate(false_scores: np.ndarray, fpr: float) -> float:
    false_scores = np.asarray(false_scores, dtype=np.float64)
    false_scores = false_scores[np.isfinite(false_scores)]
    if len(false_scores) == 0:
        return float("nan")
    return float(np.quantile(false_scores, 1.0 - fpr, method="higher"))


def threshold_for_recall(true_scores: np.ndarray, recall: float) -> float:
    true_scores = np.asarray(true_scores, dtype=np.float64)
    true_scores = true_scores[np.isfinite(true_scores)]
    if len(true_scores) == 0:
        return float("nan")
    return float(np.quantile(true_scores, 1.0 - recall, method="lower"))


def et3_threshold_by_snr(pair_path: Path, event_snr: np.ndarray) -> pd.DataFrame:
    print("Computing ET-3 final_score threshold diagnostics by pair_min_snr", flush=True)
    df = pd.read_parquet(pair_path, columns=["pair_i", "pair_j", "is_true_pair", "final_score"])
    ii = df["pair_i"].to_numpy(dtype=np.int32)
    jj = df["pair_j"].to_numpy(dtype=np.int32)
    min_snr = np.minimum(event_snr[ii], event_snr[jj])
    df["pair_min_snr"] = min_snr
    df["pair_min_snr_group"] = snr_group(min_snr)
    rows = []
    for group in ["all", "low_snr_lt8", "high_snr_ge8"]:
        sub = df if group == "all" else df[df["pair_min_snr_group"] == group]
        true_scores = sub.loc[sub["is_true_pair"].astype(bool), "final_score"].to_numpy(dtype=np.float64)
        false_scores = sub.loc[~sub["is_true_pair"].astype(bool), "final_score"].to_numpy(dtype=np.float64)
        for fpr in FPR_TARGETS:
            th = threshold_for_false_rate(false_scores, fpr)
            rows.append(
                {
                    "dataset": "constructed_et3_full_pair_catalog",
                    "score_name": "final_score",
                    "snr_group": group,
                    "threshold_source": "target_false_pair_exceedance_rate_within_snr_group",
                    "target_value": fpr,
                    **labeled_threshold_metrics(true_scores, false_scores, th),
                }
            )
        for rec in RECALL_TARGETS:
            th = threshold_for_recall(true_scores, rec)
            rows.append(
                {
                    "dataset": "constructed_et3_full_pair_catalog",
                    "score_name": "final_score",
                    "snr_group": group,
                    "threshold_source": "target_recall_within_snr_group",
                    "target_value": rec,
                    **labeled_threshold_metrics(true_scores, false_scores, th),
                }
            )
    return pd.DataFrame(rows)


def real_null_snr_tables(real_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    threshold_rows = []
    top_rows = []
    for name, path in real_paths.items():
        print(f"Computing real null SNR diagnostics for {name}", flush=True)
        df = pd.read_parquet(path)
        df["pair_min_snr"] = np.minimum(df["network_snr_i"].astype(float), df["network_snr_j"].astype(float))
        df["pair_max_snr"] = np.maximum(df["network_snr_i"].astype(float), df["network_snr_j"].astype(float))
        df["pair_min_snr_group"] = snr_group(df["pair_min_snr"].to_numpy(dtype=np.float64))
        for group in ["all", "low_snr_lt8", "high_snr_ge8"]:
            sub = df if group == "all" else df[df["pair_min_snr_group"] == group]
            summary_rows.append(
                {
                    "dataset": name,
                    "snr_group": group,
                    "n_null_pairs": int(len(sub)),
                    "final_score_min": float(sub["final_score"].min()) if len(sub) else float("nan"),
                    "final_score_median": float(sub["final_score"].median()) if len(sub) else float("nan"),
                    "final_score_p90": float(sub["final_score"].quantile(0.90)) if len(sub) else float("nan"),
                    "final_score_p99": float(sub["final_score"].quantile(0.99)) if len(sub) else float("nan"),
                    "final_score_max": float(sub["final_score"].max()) if len(sub) else float("nan"),
                }
            )
            scores = sub["final_score"].to_numpy(dtype=np.float64)
            scores = scores[np.isfinite(scores)]
            for frac in REAL_TAIL_FRACTIONS:
                if len(scores) == 0:
                    continue
                k = max(1, int(math.ceil(frac * len(scores))))
                th = float(np.partition(scores, len(scores) - k)[len(scores) - k])
                threshold_rows.append(
                    {
                        "dataset": name,
                        "snr_group": group,
                        "threshold_source": "catalog_tail_fraction_within_snr_group",
                        "target_value": frac,
                        "threshold": th,
                        "n_null_pairs": int(len(scores)),
                        "candidate_count_above_threshold": int(np.sum(scores >= th)),
                        "null_pair_exceedance_rate": float(np.mean(scores >= th)),
                    }
                )
        ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        for k in REAL_TOPK:
            sub = ranked.head(min(k, len(ranked)))
            for group, g in sub.groupby("pair_min_snr_group"):
                top_rows.append(
                    {
                        "dataset": name,
                        "top_k": int(k),
                        "snr_group": group,
                        "candidate_count_in_top_k": int(len(g)),
                        "fraction_of_top_k": float(len(g) / len(sub)) if len(sub) else float("nan"),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows), pd.DataFrame(top_rows)


def write_report(path: Path, summary: dict, et_metrics: pd.DataFrame, et_thresholds: pd.DataFrame, real_summary: pd.DataFrame, real_top: pd.DataFrame) -> None:
    def md(df: pd.DataFrame, n: int = 14) -> str:
        show = df.head(n).copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.6g}")
        return show.to_markdown(index=False)

    final_pair = et_metrics[
        (et_metrics["method"] == "final_score")
        & (et_metrics["grouping_kind"] == "pair_min_snr")
        & (et_metrics["lens_subset"] == "all")
    ][["snr_group", "n_queries", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "median_rank"]]
    final_query = et_metrics[
        (et_metrics["method"] == "final_score")
        & (et_metrics["grouping_kind"] == "query_snr")
        & (et_metrics["lens_subset"] == "all")
    ][["snr_group", "n_queries", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "median_rank"]]

    text = f"""# SNR<8 / SNR>=8 分层诊断

本报告把当前 catalog-level lensing search 结果按 SNR 分层，重点区分：

```text
low_snr_lt8  : SNR < 8
high_snr_ge8: SNR >= 8
```

对 ET-3 构造目录，有真实 companion pair，因此可以报告 R@K。对真实 GWTC-3/GWTC-4，没有真实透镜标签，因此只报告 null/background candidate 分布，不能解释为 detection 或真实 FDR。

## 数据摘要

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```

## ET-3 final_score: 按 pair_min_snr 分层

`pair_min_snr` 是真实 pair 两幅像中较弱一幅的 SNR，反映 pair 是否受弱像限制。

{md(final_pair)}

## ET-3 final_score: 按 query_snr 分层

`query_snr` 是 directed retrieval 中当前 query image 的 SNR。

{md(final_query)}

## ET-3 final_score 阈值误配率: 按 pair_min_snr 分层

低 SNR 组真对数量很小，因此阈值结果只作 sanity check，不应作为稳定统计结论。

{md(et_thresholds)}

## 真实 GWTC null catalog: 按 pair_min_snr 分层

真实 GWTC 默认所有 pair 视为 null/background。这里统计的是不同 SNR 组在 final_score 分布和 top-k candidate 中的占比。

{md(real_summary)}

Top-k 组成：

{md(real_top)}

## 论文建议

可以写：

> We further stratified retrieval and candidate-burden diagnostics by SNR. In the ET-3 synthetic catalog, almost all true lensed pairs have both images above SNR 8, so the SNR<8 stratum contains too few true pairs for stable recall estimates. For real GWTC deployments, SNR stratification is reported only as a null-catalog audit of candidate composition and is not interpreted as a detection significance.

不要写：

> SNR<8 组已经被充分验证。

因为当前 ET-3 SNR<8 true-pair 样本极少。
"""
    path.write_text(text, encoding="utf-8")


def make_figures(out_dir: Path, et_metrics: pd.DataFrame, real_top: pd.DataFrame, event_summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    sub = et_metrics[
        (et_metrics["grouping_kind"] == "pair_min_snr")
        & (et_metrics["lens_subset"] == "all")
        & (et_metrics["snr_group"].isin(["low_snr_lt8", "high_snr_ge8"]))
    ].copy()
    methods = ["waveform_score", "time_score", "sky_score", "final_score"]
    x = np.arange(len(methods))
    width = 0.35
    for offset, group, color in [(-width / 2, "low_snr_lt8", "#D92D20"), (width / 2, "high_snr_ge8", "#175CD3")]:
        vals = [float(sub[(sub["method"] == m) & (sub["snr_group"] == group)]["recall_at_10"].iloc[0]) if len(sub[(sub["method"] == m) & (sub["snr_group"] == group)]) else np.nan for m in methods]
        axes[0].bar(x + offset, vals, width, label=group, color=color, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["waveform", "time", "sky", "final"], rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("ET-3 Recall@10")
    axes[0].set_title("A. Recall by pair min-SNR")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    event_counts = event_summary[event_summary["row_type"] == "et3_event_snr_group"].copy()
    axes[1].bar(event_counts["snr_group"], event_counts["count"], color=["#D92D20", "#175CD3"], alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ET-3 event count")
    axes[1].set_title("B. ET-3 SNR group sizes")
    axes[1].grid(True, axis="y", alpha=0.25)

    sub = real_top[(real_top["top_k"].isin([10, 20, 50, 100]))].copy()
    for name, g0 in sub.groupby("dataset"):
        lows = []
        ks = []
        for k, g in g0.groupby("top_k"):
            ks.append(k)
            low = g[g["snr_group"] == "low_snr_lt8"]["candidate_count_in_top_k"].sum()
            total = g["candidate_count_in_top_k"].sum()
            lows.append(low / total if total else 0.0)
        axes[2].plot(ks, lows, marker="o", label=name)
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("Top-K real candidates")
    axes[2].set_ylabel("Fraction with min SNR < 8")
    axes[2].set_title("C. Real top-K low-SNR composition")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig_snr_stratified_diagnostics.pdf", dpi=300)
    fig.savefig(out_dir / "figures" / "fig_snr_stratified_diagnostics.png", dpi=300)
    plt.close(fig)


def package_outputs(package_path: Path, repo: Path, files: list[Path]) -> None:
    with tarfile.open(package_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(repo)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_default())
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out_dir = args.out_dir or (repo / "results" / "snr_stratified_diagnostics_20260702")
    ensure(out_dir)
    ensure(out_dir / "figures")
    ensure(repo / "packages")

    pair_path = repo / "results" / "pair_level_full_curves" / "et3_pair_scores_full.parquet"
    meta_path = repo / "runs" / "p2a_ann_et3_noisy_meta_20260618" / "catalog_meta.csv"
    meta = pd.read_csv(meta_path).sort_values("event_id").reset_index(drop=True)
    event_snr = meta["network_snr"].to_numpy(dtype=np.float64)
    n_events = len(meta)
    true_pairs = load_true_pairs(pair_path)
    true_pairs["pair_min_snr"] = np.minimum(event_snr[true_pairs["pair_i"].to_numpy(dtype=np.int32)], event_snr[true_pairs["pair_j"].to_numpy(dtype=np.int32)])
    true_pairs["pair_min_snr_group"] = snr_group(true_pairs["pair_min_snr"].to_numpy(dtype=np.float64))

    directed_all = []
    metric_rows: list[dict] = []
    for method in METHODS:
        directed = compute_directed_ranks_by_method(pair_path, true_pairs, event_snr, n_events, method)
        directed_all.append(directed)
        add_metric_rows(metric_rows, directed, method, "query_snr", "query_snr_group")
        add_metric_rows(metric_rows, directed, method, "pair_min_snr", "pair_min_snr_group")
    directed_df = pd.concat(directed_all, ignore_index=True)
    et_metrics = pd.DataFrame(metric_rows)
    directed_df.to_parquet(out_dir / "constructed_et3_directed_true_pair_ranks_by_snr.parquet", index=False)
    et_metrics.to_csv(out_dir / "constructed_et3_retrieval_metrics_by_snr.csv", index=False)

    et_threshold = et3_threshold_by_snr(pair_path, event_snr)
    et_threshold.to_csv(out_dir / "constructed_et3_threshold_metrics_by_snr.csv", index=False)

    real_paths = {
        "real_gwtc3_null_catalog": repo / "runs" / "real_gwtc_lensing_search_20260625" / "results" / "real_pair_scores_waveform_time_sky.parquet",
        "real_gwtc4p1_null_catalog": repo / "runs" / "real_gwtc34_lensing_search_20260629_full_o4" / "results" / "gwtc4_pair_scores_waveform_time_sky.parquet",
    }
    real_summary, real_thresholds, real_top = real_null_snr_tables(real_paths)
    real_summary.to_csv(out_dir / "real_gwtc_null_pair_score_summary_by_snr.csv", index=False)
    real_thresholds.to_csv(out_dir / "real_gwtc_null_thresholds_by_snr.csv", index=False)
    real_top.to_csv(out_dir / "real_gwtc_topk_candidate_composition_by_snr.csv", index=False)

    event_rows = []
    for group in ["low_snr_lt8", "high_snr_ge8"]:
        event_rows.append({"row_type": "et3_event_snr_group", "snr_group": group, "count": int(np.sum(snr_group(event_snr) == group))})
    for group in ["low_snr_lt8", "high_snr_ge8"]:
        event_rows.append({"row_type": "et3_true_pair_min_snr_group", "snr_group": group, "count": int(np.sum(true_pairs["pair_min_snr_group"] == group))})
    for fam, sub in meta.groupby("family"):
        for group in ["low_snr_lt8", "high_snr_ge8"]:
            event_rows.append({"row_type": f"et3_event_snr_group_{fam}", "snr_group": group, "count": int(np.sum(snr_group(sub["network_snr"].to_numpy(dtype=np.float64)) == group))})
    event_summary = pd.DataFrame(event_rows)
    event_summary.to_csv(out_dir / "snr_group_counts.csv", index=False)

    summary = {
        "snr_cut": SNR_CUT,
        "et3": {
            "n_events": int(n_events),
            "n_events_snr_lt8": int(np.sum(event_snr < SNR_CUT)),
            "n_events_snr_ge8": int(np.sum(event_snr >= SNR_CUT)),
            "n_true_pairs": int(len(true_pairs)),
            "n_true_pairs_min_snr_lt8": int(np.sum(true_pairs["pair_min_snr"].to_numpy(dtype=np.float64) < SNR_CUT)),
            "n_true_pairs_min_snr_ge8": int(np.sum(true_pairs["pair_min_snr"].to_numpy(dtype=np.float64) >= SNR_CUT)),
            "snr_quantiles": {str(q): float(np.quantile(event_snr, q)) for q in [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0]},
        },
        "real_catalogs": {
            row["dataset"] + ":" + row["snr_group"]: int(row["n_null_pairs"]) for _, row in real_summary.iterrows()
        },
        "notes": [
            "ET-3 SNR<8 true-pair group is very small; recall estimates are unstable and should be reported as an audit only.",
            "Real GWTC has no confirmed true lens labels here; SNR-stratified real results are null-catalog candidate composition, not recall or detection significance.",
            "GWTC injection pair diagnostics currently contain snr_ratio but not absolute injected-event SNR, so absolute SNR<8/>=8 injection stratification is not reported here.",
        ],
    }
    (out_dir / "snr_stratified_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(out_dir / "snr_stratified_diagnostics_cn.md", summary, et_metrics, et_threshold, real_summary, real_top)
    make_figures(out_dir, et_metrics, real_top, event_summary)

    package_path = repo / "packages" / "snr_stratified_diagnostics_20260702.tar.gz"
    files = [
        out_dir / "constructed_et3_directed_true_pair_ranks_by_snr.parquet",
        out_dir / "constructed_et3_retrieval_metrics_by_snr.csv",
        out_dir / "constructed_et3_threshold_metrics_by_snr.csv",
        out_dir / "real_gwtc_null_pair_score_summary_by_snr.csv",
        out_dir / "real_gwtc_null_thresholds_by_snr.csv",
        out_dir / "real_gwtc_topk_candidate_composition_by_snr.csv",
        out_dir / "snr_group_counts.csv",
        out_dir / "snr_stratified_summary.json",
        out_dir / "snr_stratified_diagnostics_cn.md",
        out_dir / "figures" / "fig_snr_stratified_diagnostics.pdf",
        out_dir / "figures" / "fig_snr_stratified_diagnostics.png",
        repo / "scripts" / "experiments" / "snr_stratified_diagnostics.py",
    ]
    package_outputs(package_path, repo, files)

    print(json.dumps({"out_dir": str(out_dir), "package": str(package_path), **summary["et3"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
