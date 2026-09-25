from __future__ import annotations

import json
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t


RECALL_K = (1, 5, 10, 50)


def split_integrity_audit(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Verify system-index partitions are disjoint and record reproducible hashes."""
    result: dict[str, Any] = {"all_disjoint": True, "groups": {}}
    for group, parts in splits.items():
        arrays = {name: np.asarray(values, dtype=np.int64) for name, values in parts.items()}
        names = sorted(arrays)
        overlaps = {
            f"{a}__{b}": int(np.intersect1d(arrays[a], arrays[b]).size)
            for pos, a in enumerate(names)
            for b in names[pos + 1 :]
        }
        disjoint = all(value == 0 for value in overlaps.values())
        result["all_disjoint"] = bool(result["all_disjoint"] and disjoint)
        result["groups"][group] = {
            "counts": {name: int(len(arrays[name])) for name in names},
            "pairwise_overlap_counts": overlaps,
            "disjoint": disjoint,
            "index_hashes_sha256": {
                name: hashlib.sha256(np.sort(arrays[name]).tobytes()).hexdigest() for name in names
            },
        }
    return result


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ranks_from_components(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
    gt: np.ndarray,
    meta: list[dict[str, Any]],
    deployment: str,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    query = np.where(np.asarray(gt) >= 0)[0].astype(np.int32)
    if not len(query):
        raise RuntimeError("No lensed queries")
    score = None
    for channel, weight in weights.items():
        if float(weight) == 0.0:
            continue
        part = np.asarray(components[channel][query], dtype=np.float32) * float(weight)
        score = part if score is None else score + part
    if score is None:
        raise ValueError("All channel weights are zero")
    score[np.arange(len(query)), query] = -np.inf
    partner = np.asarray(gt[query], dtype=np.int32)
    true_score = score[np.arange(len(query)), partner]
    ranks = (1 + np.sum(score > true_score[:, None], axis=1)).astype(np.int32)
    rows = []
    for local, q in enumerate(query):
        item = meta[int(q)]
        family = str(item.get("family", "unknown")).upper()
        source = item.get("source_index", item.get("pair_id", int(min(q, partner[local]))))
        rows.append(
            {
                "deployment": deployment,
                "seed": int(seed),
                "method": method,
                "query_index": int(q),
                "partner_index": int(partner[local]),
                "family": family,
                "system_id": f"{family}:{source}",
                "query_tag": str(item.get("tag", "query")),
                "query_rank": int(ranks[local]),
            }
        )
    return pd.DataFrame(rows), score


def metric_rows(query_ranks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["deployment", "seed", "method"]
    for key, frame in query_ranks.groupby(keys, sort=False):
        subsets: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
        subsets.extend((family, part) for family, part in frame.groupby("family", sort=True))
        for subset, part in subsets:
            ranks = part["query_rank"].to_numpy(dtype=np.int32)
            row: dict[str, Any] = dict(zip(keys, key))
            row.update(
                {
                    "subset": subset,
                    "n_queries": int(len(ranks)),
                    "n_systems": int(part["system_id"].nunique()),
                    "median_rank": float(np.median(ranks)),
                }
            )
            for k in RECALL_K:
                row[f"r_at_{k}"] = float(np.mean(ranks <= k))
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty(draws, dtype=np.float64)
    chunk = max(1, min(draws, 2_000_000 // max(len(values), 1)))
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        out[start:stop] = values[idx].mean(axis=1)
    return out


def bootstrap_system_ci(
    query_ranks: pd.DataFrame,
    draws: int = 10_000,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Cluster bootstrap by lens system, stratified by family for overall rows."""
    rows: list[dict[str, Any]] = []
    group_keys = ["deployment", "seed", "method"]
    for group_index, (key, frame) in enumerate(query_ranks.groupby(group_keys, sort=False)):
        subsets = ["overall", *sorted(frame["family"].unique().tolist())]
        for subset_index, subset in enumerate(subsets):
            part = frame if subset == "overall" else frame[frame["family"] == subset]
            for k in (1, 10):
                system = (
                    part.assign(hit=(part["query_rank"] <= k).astype(float))
                    .groupby(["family", "system_id"], as_index=False)["hit"]
                    .mean()
                )
                rng = np.random.default_rng(seed + group_index * 1009 + subset_index * 97 + k)
                if subset == "overall":
                    samples = []
                    weights = []
                    for _, family_systems in system.groupby("family", sort=True):
                        vals = family_systems["hit"].to_numpy(dtype=np.float64)
                        samples.append(_bootstrap_mean(vals, draws, rng))
                        weights.append(len(vals))
                    boot = np.average(np.stack(samples), axis=0, weights=np.asarray(weights))
                else:
                    boot = _bootstrap_mean(system["hit"].to_numpy(dtype=np.float64), draws, rng)
                point = float(np.mean(part["query_rank"].to_numpy(dtype=np.int32) <= k))
                rows.append(
                    {
                        "deployment": key[0],
                        "seed": int(key[1]),
                        "method": key[2],
                        "subset": subset,
                        "metric": f"r_at_{k}",
                        "point_estimate": point,
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "bootstrap_draws": int(draws),
                        "bootstrap_unit": "lensed_system_stratified_by_family",
                        "n_systems": int(system["system_id"].nunique()),
                        "n_queries": int(len(part)),
                    }
                )
    return pd.DataFrame(rows)


def across_seed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"deployment", "seed", "method", "subset", "n_queries", "n_systems"}
    metric_cols = [c for c in per_seed.columns if c not in id_cols]
    rows: list[dict[str, Any]] = []
    for key, frame in per_seed.groupby(["deployment", "method", "subset"], sort=False):
        for metric in metric_cols:
            values = frame[metric].dropna().to_numpy(dtype=np.float64)
            if not len(values):
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            ci_low = mean - half
            ci_high = mean + half
            if metric.startswith("r_at_") or metric in {
                "roc_auc",
                "average_precision",
                "base_positive_rate",
                "precision",
                "recall",
                "fpr",
            }:
                ci_low = max(0.0, ci_low)
                ci_high = min(1.0, ci_high)
            rows.append(
                {
                    "deployment": key[0],
                    "method": key[1],
                    "subset": key[2],
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": mean,
                    "std": std,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "ci95_low_t": ci_low,
                    "ci95_high_t": ci_high,
                    "ci_method": "two-sided Student-t across independent seeds; bounded metrics clipped to [0,1]",
                }
            )
    return pd.DataFrame(rows)


def full_unordered_pair_metrics(
    directed_score: np.ndarray,
    gt: np.ndarray,
    meta: list[dict[str, Any]],
    fpr_targets: Iterable[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    recall_targets: Iterable[float] = (0.1, 0.5, 0.9),
    aggregation: str = "max",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Exact unordered-pair metrics from directed row-standardized scores."""
    n = int(len(gt))
    ii, jj = np.triu_indices(n, k=1)
    score_i_to_j = directed_score[ii, jj]
    score_j_to_i = directed_score[jj, ii]
    if aggregation == "max":
        scores = np.maximum(score_i_to_j, score_j_to_i).astype(np.float32)
    elif aggregation == "mean":
        scores = (0.5 * (score_i_to_j + score_j_to_i)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported unordered-pair aggregation: {aggregation!r}")
    labels = (np.asarray(gt[ii], dtype=np.int32) == jj).astype(np.int8)
    n_pos = int(labels.sum())
    n_total = int(len(labels))
    n_neg = n_total - n_pos
    if n_pos == 0:
        raise RuntimeError("No true unordered pairs")

    positive_family = np.zeros(n_total, dtype=np.int8)
    pos_idx = np.flatnonzero(labels)
    for p in pos_idx:
        family = str(meta[int(ii[p])].get("family", "")).upper()
        positive_family[p] = 1 if family == "SIS" else (2 if family == "PM" else 0)

    order = np.argsort(scores, kind="quicksort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_family = positive_family[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), n_total - 1]
    cum_tp_all = np.cumsum(sorted_labels, dtype=np.int64)
    cum_fp_all = np.cumsum(1 - sorted_labels, dtype=np.int64)
    cum_sis_all = np.cumsum(sorted_family == 1, dtype=np.int64)
    cum_pm_all = np.cumsum(sorted_family == 2, dtype=np.int64)
    tp = cum_tp_all[ends]
    fp = cum_fp_all[ends]
    sis_tp = cum_sis_all[ends]
    pm_tp = cum_pm_all[ends]
    thresholds = sorted_scores[ends]
    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)
    precision = tp / np.maximum(tp + fp, 1)
    delta_tp = np.diff(np.r_[0, tp])
    average_precision = float(np.sum((delta_tp / float(n_pos)) * precision))
    roc_auc = float(np.trapz(np.r_[0.0, tpr, 1.0], np.r_[0.0, fpr, 1.0]))

    n_sis = int(np.sum(positive_family == 1))
    n_pm = int(np.sum(positive_family == 2))
    operating: list[dict[str, Any]] = []
    for target in fpr_targets:
        budget = int(math.floor(float(target) * n_neg))
        eligible = np.flatnonzero(fp <= budget)
        if len(eligible):
            ix = int(eligible[-1])
            row = {
                "operating_point": "fpr",
                "target": float(target),
                "threshold": float(thresholds[ix]),
                "false_pairs": int(fp[ix]),
                "true_pairs": int(tp[ix]),
                "fpr": float(fpr[ix]),
                "recall": float(tpr[ix]),
                "precision": float(precision[ix]),
                "sis_recall": float(sis_tp[ix] / n_sis) if n_sis else None,
                "pm_recall": float(pm_tp[ix] / n_pm) if n_pm else None,
            }
        else:
            row = {
                "operating_point": "fpr",
                "target": float(target),
                "threshold": float("inf"),
                "false_pairs": 0,
                "true_pairs": 0,
                "fpr": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "sis_recall": 0.0,
                "pm_recall": 0.0,
            }
        operating.append(row)
    for target in recall_targets:
        ix = int(np.searchsorted(tpr, float(target), side="left"))
        ix = min(ix, len(tpr) - 1)
        operating.append(
            {
                "operating_point": "recall",
                "target": float(target),
                "threshold": float(thresholds[ix]),
                "false_pairs": int(fp[ix]),
                "true_pairs": int(tp[ix]),
                "fpr": float(fpr[ix]),
                "recall": float(tpr[ix]),
                "precision": float(precision[ix]),
                "sis_recall": float(sis_tp[ix] / n_sis) if n_sis else None,
                "pm_recall": float(pm_tp[ix] / n_pm) if n_pm else None,
            }
        )

    summary = {
        "n_events": n,
        "n_pairs": n_total,
        "n_true_pairs": n_pos,
        "n_false_pairs": n_neg,
        "base_positive_rate": float(n_pos / n_total),
        "unordered_score_aggregation": aggregation,
        "unordered_score_rule": (
            "max of the two directed row-standardized final scores"
            if aggregation == "max"
            else "mean of the two directed row-standardized final scores"
        ),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "n_true_sis": n_sis,
        "n_true_pm": n_pm,
    }
    return summary, pd.DataFrame(operating)
