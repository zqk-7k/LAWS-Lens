#!/usr/bin/env python3
"""Top-B true/false pair accounting for held-out O3 real-noise injections.

This is a global unordered-pair shortlist diagnostic.  It is distinct from
directed companion R@K.  The score weights are read from each seed's frozen
validation selection; the held-out test labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "results/unified_sky_v81_20260725/gwtc3"
DEFAULT_OUTPUT = (
    REPO / "results/et3_moderate_sky_gwtc3_topb_20260726/gwtc3_topb"
)
SEEDS = (202607241, 202607242, 202607243)
COUNT_BUDGETS = (10, 20, 50, 100, 200, 500, 1000)
PAIR_FRACTIONS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
METHODS = (
    "candidate_three_channel_strict_positive",
    "retrieval_three_channel_strict_positive",
    "waveform_only",
    "time_only",
    "sky_only",
    "time_sky_candidate_selected",
    "three_channel_equal_evidence",
)


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


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = success / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def score_pairs(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        float(weights["waveform"]) * frame["waveform_score"].to_numpy(np.float64)
        + float(weights["time"]) * frame["time_score"].to_numpy(np.float64)
        + float(weights["sky"]) * frame["sky_score"].to_numpy(np.float64)
    )


def metric_row(
    ordered: pd.DataFrame,
    *,
    seed: int,
    method: str,
    weights: dict[str, float],
    budget_type: str,
    budget_label: str,
    budget_pairs: int,
    n_true: int,
    base_rate: float,
) -> dict[str, Any]:
    selected = ordered.head(budget_pairs)
    true_recovered = int(selected["is_true_pair"].sum())
    false_candidates = int(len(selected) - true_recovered)
    precision = true_recovered / max(len(selected), 1)
    recall = true_recovered / max(n_true, 1)
    low, high = wilson_interval(true_recovered, len(selected))
    return {
        "seed": int(seed),
        "method": method,
        "budget_type": budget_type,
        "budget_label": budget_label,
        "budget_pairs": int(len(selected)),
        "waveform_weight": float(weights["waveform"]),
        "time_weight": float(weights["time"]),
        "sky_weight": float(weights["sky"]),
        "n_catalog_events": 450,
        "n_unordered_pairs": int(len(ordered)),
        "n_true_pairs": int(n_true),
        "n_false_pairs": int(len(ordered) - n_true),
        "base_positive_rate": float(base_rate),
        "true_recovered": true_recovered,
        "false_candidates": false_candidates,
        "precision": float(precision),
        "precision_wilson95_low_descriptive": float(low),
        "precision_wilson95_high_descriptive": float(high),
        "recall": float(recall),
        "false_discovery_rate": float(1.0 - precision),
        "enrichment_over_base_rate": float(precision / max(base_rate, 1e-300)),
        "score_threshold": float(selected["final_score"].iloc[-1]),
    }


def run_seed(
    seed: int,
    source_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    seed_root = source_root / f"seed_{seed}"
    pair_path = seed_root / "fusion_heldout_test_pairs_v81.parquet"
    weight_path = seed_root / "selected_weights_v81.json"
    pairs = pd.read_parquet(pair_path)
    if not np.all(pairs["idx_i"].to_numpy() < pairs["idx_j"].to_numpy()):
        raise RuntimeError("Held-out pair table is not unordered i<j")
    if pairs.duplicated(["idx_i", "idx_j"]).any():
        raise RuntimeError("Duplicate unordered pairs found")
    n_pairs = len(pairs)
    n_true = int(pairs["is_true_pair"].sum())
    n_false = n_pairs - n_true
    event_count = int(pairs["event_count"].iloc[0])
    expected_pairs = event_count * (event_count - 1) // 2
    if n_pairs != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} pairs, found {n_pairs}")
    base_rate = n_true / n_pairs
    weights_all = json.loads(weight_path.read_text(encoding="utf-8"))["methods"]

    rows: list[dict[str, Any]] = []
    shortlist_frames: list[pd.DataFrame] = []
    null_rows: list[dict[str, Any]] = []
    for method in METHODS:
        weights = weights_all[method]
        scored = pairs.copy()
        scored["final_score"] = score_pairs(scored, weights)
        scored = scored.sort_values(
            ["final_score", "idx_i", "idx_j"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        scored.insert(0, "global_rank", np.arange(1, len(scored) + 1))
        scored.insert(0, "method", method)
        scored.insert(0, "seed", int(seed))

        for budget in COUNT_BUDGETS:
            rows.append(
                metric_row(
                    scored,
                    seed=seed,
                    method=method,
                    weights=weights,
                    budget_type="fixed_count",
                    budget_label=f"top_{budget}",
                    budget_pairs=min(budget, n_pairs),
                    n_true=n_true,
                    base_rate=base_rate,
                )
            )
        for fraction in PAIR_FRACTIONS:
            budget = max(1, int(math.ceil(fraction * n_pairs)))
            rows.append(
                metric_row(
                    scored,
                    seed=seed,
                    method=method,
                    weights=weights,
                    budget_type="top_pair_fraction",
                    budget_label=f"top_{100.0 * fraction:g}pct",
                    budget_pairs=budget,
                    n_true=n_true,
                    base_rate=base_rate,
                )
            )

        keep = 1000 if method == "candidate_three_channel_strict_positive" else 100
        shortlist = scored.head(keep).copy()
        shortlist["waveform_weight"] = float(weights["waveform"])
        shortlist["time_weight"] = float(weights["time"])
        shortlist["sky_weight"] = float(weights["sky"])
        shortlist_frames.append(shortlist)

        false_scores = scored.loc[scored["is_true_pair"] == 0, "final_score"]
        true_scores = scored.loc[scored["is_true_pair"] == 1, "final_score"]
        null_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "n_true_pairs": n_true,
                "n_false_pairs": n_false,
                "true_score_median": float(true_scores.median()),
                "true_score_p90": float(true_scores.quantile(0.9)),
                "false_score_median": float(false_scores.median()),
                "false_score_p90": float(false_scores.quantile(0.9)),
                "false_score_p99": float(false_scores.quantile(0.99)),
                "false_score_p999": float(false_scores.quantile(0.999)),
                "maximum_false_score": float(false_scores.max()),
            }
        )
    return rows, shortlist_frames, null_rows


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "true_recovered",
        "false_candidates",
        "precision",
        "recall",
        "false_discovery_rate",
        "enrichment_over_base_rate",
        "score_threshold",
    )
    rows = []
    groups = ["method", "budget_type", "budget_label", "budget_pairs"]
    for key, part in per_seed.groupby(groups, sort=False):
        row = dict(zip(groups, key))
        row["n_seeds"] = int(part["seed"].nunique())
        row["base_positive_rate"] = float(part["base_positive_rate"].mean())
        for column in metric_columns:
            values = part[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1))
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{column}_q75"] = float(np.quantile(values, 0.75))
            row[f"{column}_min"] = float(np.min(values))
            row[f"{column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_shortlists: list[pd.DataFrame] = []
    all_null: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows, shortlists, null_rows = run_seed(
            seed,
            args.source_root,
            args.output_root,
        )
        all_rows.extend(rows)
        all_shortlists.extend(shortlists)
        all_null.extend(null_rows)

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(
        args.output_root / "gwtc3_topb_true_false_per_seed.csv",
        index=False,
    )
    summary = aggregate(per_seed)
    summary.to_csv(
        args.output_root / "gwtc3_topb_true_false_summary.csv",
        index=False,
    )
    shortlists = pd.concat(all_shortlists, ignore_index=True)
    shortlists.to_parquet(
        args.output_root / "gwtc3_top_pair_shortlists_by_method.parquet",
        index=False,
    )
    primary = shortlists[
        shortlists["method"] == "candidate_three_channel_strict_positive"
    ].copy()
    primary.to_csv(
        args.output_root / "gwtc3_primary_three_channel_top1000_per_seed.csv",
        index=False,
    )
    pd.DataFrame(all_null).to_csv(
        args.output_root / "gwtc3_false_score_null_audit.csv",
        index=False,
    )

    top10 = per_seed[
        (per_seed["method"] == "candidate_three_channel_strict_positive")
        & (per_seed["budget_label"] == "top_10")
    ]
    payload = {
        "status": "complete",
        "scope": (
            "held-out synthetic lensed pairs injected into real O3 noise; "
            "not confirmed lenses in the real GWTC-3 catalog"
        ),
        "score_scope": "all unordered i<j pairs; self-pairs excluded",
        "score_weights": "frozen per seed on synthetic validation systems",
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "n_events_per_seed": int(per_seed["n_catalog_events"].iloc[0]),
        "n_pairs_per_seed": int(per_seed["n_unordered_pairs"].iloc[0]),
        "n_true_pairs_per_seed": int(per_seed["n_true_pairs"].iloc[0]),
        "n_false_pairs_per_seed": int(per_seed["n_false_pairs"].iloc[0]),
        "base_positive_rate": float(per_seed["base_positive_rate"].iloc[0]),
        "primary_top10": {
            "true_recovered_per_seed": top10["true_recovered"].astype(int).tolist(),
            "false_candidates_per_seed": top10[
                "false_candidates"
            ].astype(int).tolist(),
            "precision_mean": float(top10["precision"].mean()),
            "recall_mean": float(top10["recall"].mean()),
            "enrichment_mean": float(
                top10["enrichment_over_base_rate"].mean()
            ),
        },
        "important_distinction": (
            "Global precision@B ranks the whole unordered pair catalog once; "
            "directed companion R@K ranks candidates separately for every query."
        ),
    }
    write_json(args.output_root / "gwtc3_topb_summary.json", payload)


if __name__ == "__main__":
    main()
