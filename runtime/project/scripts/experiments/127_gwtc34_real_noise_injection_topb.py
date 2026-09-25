#!/usr/bin/env python3
"""Global Top-B true/false accounting for held-out GWTC3/O4a injections.

The input tables contain every unordered pair (i < j) from held-out synthetic
lensed systems injected into run-matched real off-source detector noise.  Pair
labels are used only after ranking.  Fusion weights are frozen independently
for every training seed using that seed's validation split.

This diagnostic is not a truth assessment of pairs in the real GWTC catalogs.
Real catalog pairs have no lensing labels and require Bayesian follow-up.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "results/unified_sky_v81_20260725"
DEFAULT_OUTPUT = REPO / "results/et3_gwtc34_complete_20260726/gwtc_topb"
DEFAULT_DEPLOYMENTS = ("gwtc3", "gwtc4")
COUNT_BUDGETS = (5, 10, 20, 50, 100, 200, 500, 1000)
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
DEPLOYMENT_LABELS = {
    "gwtc3": "GWTC-3.0 / O3",
    "gwtc4": "GWTC-4.1 / O4a",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def wilson_interval(
    success: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
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


def discover_seeds(deployment_root: Path) -> tuple[int, ...]:
    seeds = []
    for seed_root in sorted(deployment_root.glob("seed_*")):
        pair_path = seed_root / "fusion_heldout_test_pairs_v81.parquet"
        weight_path = seed_root / "selected_weights_v81.json"
        if pair_path.exists() and weight_path.exists():
            seeds.append(int(seed_root.name.removeprefix("seed_")))
    if not seeds:
        raise FileNotFoundError(
            f"No complete seed directories found below {deployment_root}"
        )
    return tuple(seeds)


def score_pairs(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    values = (
        float(weights["waveform"])
        * frame["waveform_score"].to_numpy(np.float64)
        + float(weights["time"]) * frame["time_score"].to_numpy(np.float64)
        + float(weights["sky"]) * frame["sky_score"].to_numpy(np.float64)
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite final scores found")
    return values


def metric_row(
    ordered: pd.DataFrame,
    *,
    deployment: str,
    seed: int,
    method: str,
    weights: dict[str, float],
    event_count: int,
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
        "deployment": deployment,
        "deployment_label": DEPLOYMENT_LABELS[deployment],
        "seed": int(seed),
        "method": method,
        "budget_type": budget_type,
        "budget_label": budget_label,
        "budget_pairs": int(len(selected)),
        "waveform_weight": float(weights["waveform"]),
        "time_weight": float(weights["time"]),
        "sky_weight": float(weights["sky"]),
        "n_catalog_events": int(event_count),
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
    deployment: str,
    seed: int,
    deployment_root: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    seed_root = deployment_root / f"seed_{seed}"
    pairs = pd.read_parquet(
        seed_root / "fusion_heldout_test_pairs_v81.parquet"
    )
    required = {
        "idx_i",
        "idx_j",
        "is_true_pair",
        "event_count",
        "waveform_score",
        "time_score",
        "sky_score",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise RuntimeError(f"Missing pair-table columns: {missing}")
    if not np.all(pairs["idx_i"].to_numpy() < pairs["idx_j"].to_numpy()):
        raise RuntimeError("Held-out pair table is not unordered i < j")
    if pairs.duplicated(["idx_i", "idx_j"]).any():
        raise RuntimeError("Duplicate unordered pairs found")

    n_pairs = len(pairs)
    n_true = int(pairs["is_true_pair"].sum())
    n_false = n_pairs - n_true
    event_count_values = pairs["event_count"].unique()
    if len(event_count_values) != 1:
        raise RuntimeError(f"Ambiguous event counts: {event_count_values}")
    event_count = int(event_count_values[0])
    expected_pairs = event_count * (event_count - 1) // 2
    if n_pairs != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} pairs, found {n_pairs}")
    base_rate = n_true / n_pairs

    weight_path = seed_root / "selected_weights_v81.json"
    weights_all = json.loads(weight_path.read_text(encoding="utf-8"))[
        "methods"
    ]
    absent_methods = sorted(set(METHODS) - set(weights_all))
    if absent_methods:
        raise RuntimeError(f"Missing frozen methods in {weight_path}: {absent_methods}")

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
        scored.insert(0, "deployment", deployment)

        for budget in COUNT_BUDGETS:
            rows.append(
                metric_row(
                    scored,
                    deployment=deployment,
                    seed=seed,
                    method=method,
                    weights=weights,
                    event_count=event_count,
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
                    deployment=deployment,
                    seed=seed,
                    method=method,
                    weights=weights,
                    event_count=event_count,
                    budget_type="top_pair_fraction",
                    budget_label=f"top_{100.0 * fraction:g}pct",
                    budget_pairs=budget,
                    n_true=n_true,
                    base_rate=base_rate,
                )
            )

        keep = (
            1000
            if method == "candidate_three_channel_strict_positive"
            else 100
        )
        shortlist = scored.head(keep).copy()
        shortlist["waveform_weight"] = float(weights["waveform"])
        shortlist["time_weight"] = float(weights["time"])
        shortlist["sky_weight"] = float(weights["sky"])
        shortlist_frames.append(shortlist)

        false_scores = scored.loc[~scored["is_true_pair"].astype(bool), "final_score"]
        true_scores = scored.loc[scored["is_true_pair"].astype(bool), "final_score"]
        null_rows.append(
            {
                "deployment": deployment,
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
    groups = [
        "deployment",
        "deployment_label",
        "method",
        "budget_type",
        "budget_label",
        "budget_pairs",
    ]
    for key, part in per_seed.groupby(groups, sort=False):
        row = dict(zip(groups, key))
        row["n_seeds"] = int(part["seed"].nunique())
        row["n_catalog_events"] = int(part["n_catalog_events"].iloc[0])
        row["n_unordered_pairs"] = int(part["n_unordered_pairs"].iloc[0])
        row["n_true_pairs"] = int(part["n_true_pairs"].iloc[0])
        row["n_false_pairs"] = int(part["n_false_pairs"].iloc[0])
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


def deployment_payload(
    deployment: str,
    seeds: Iterable[int],
    per_seed: pd.DataFrame,
) -> dict[str, Any]:
    primary = per_seed[
        per_seed["method"].eq("candidate_three_channel_strict_positive")
        & per_seed["budget_type"].eq("fixed_count")
    ].copy()
    fixed_budget = {}
    for label, part in primary.groupby("budget_label", sort=False):
        fixed_budget[label] = {
            "budget_pairs": int(part["budget_pairs"].iloc[0]),
            "true_recovered_per_seed": part["true_recovered"].astype(int).tolist(),
            "false_candidates_per_seed": part["false_candidates"].astype(int).tolist(),
            "true_recovered_mean": float(part["true_recovered"].mean()),
            "false_candidates_mean": float(part["false_candidates"].mean()),
            "precision_mean": float(part["precision"].mean()),
            "recall_mean": float(part["recall"].mean()),
            "enrichment_mean": float(part["enrichment_over_base_rate"].mean()),
        }
    first = per_seed.iloc[0]
    return {
        "status": "complete",
        "deployment": deployment,
        "deployment_label": DEPLOYMENT_LABELS[deployment],
        "scope": (
            "held-out synthetic lensed pairs injected into run-matched real "
            "off-source noise; not confirmed lenses in the real GWTC catalog"
        ),
        "score_scope": "all unordered i<j pairs; self-pairs excluded",
        "score_weights": "frozen per seed on synthetic validation systems",
        "primary_topb_method": "candidate_three_channel_strict_positive",
        "n_seeds": len(tuple(seeds)),
        "seeds": list(seeds),
        "n_events_per_seed": int(first["n_catalog_events"]),
        "n_pairs_per_seed": int(first["n_unordered_pairs"]),
        "n_true_pairs_per_seed": int(first["n_true_pairs"]),
        "n_false_pairs_per_seed": int(first["n_false_pairs"]),
        "base_positive_rate": float(first["base_positive_rate"]),
        "fixed_budget": fixed_budget,
        "important_distinction": (
            "Global precision@B ranks the whole unordered pair catalog once; "
            "directed companion R@K ranks candidates separately for every query."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--deployments",
        nargs="+",
        choices=tuple(DEPLOYMENT_LABELS),
        default=list(DEFAULT_DEPLOYMENTS),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    combined_rows: list[pd.DataFrame] = []
    combined_summaries: list[pd.DataFrame] = []
    combined_null: list[pd.DataFrame] = []
    payloads: dict[str, Any] = {}

    for deployment in args.deployments:
        deployment_root = args.source_root / deployment
        output = args.output_root / deployment
        output.mkdir(parents=True, exist_ok=True)
        seeds = discover_seeds(deployment_root)

        all_rows: list[dict[str, Any]] = []
        all_shortlists: list[pd.DataFrame] = []
        all_null: list[dict[str, Any]] = []
        for seed in seeds:
            rows, shortlists, null_rows = run_seed(
                deployment,
                seed,
                deployment_root,
            )
            all_rows.extend(rows)
            all_shortlists.extend(shortlists)
            all_null.extend(null_rows)

        per_seed = pd.DataFrame(all_rows)
        summary = aggregate(per_seed)
        shortlists = pd.concat(all_shortlists, ignore_index=True)
        null_audit = pd.DataFrame(all_null)
        prefix = deployment

        per_seed.to_csv(
            output / f"{prefix}_topb_true_false_per_seed.csv",
            index=False,
        )
        summary.to_csv(
            output / f"{prefix}_topb_true_false_summary.csv",
            index=False,
        )
        shortlists.to_parquet(
            output / f"{prefix}_top_pair_shortlists_by_method.parquet",
            index=False,
        )
        primary = shortlists[
            shortlists["method"].eq(
                "candidate_three_channel_strict_positive"
            )
        ]
        primary.to_csv(
            output / f"{prefix}_primary_three_channel_top1000_per_seed.csv",
            index=False,
        )
        null_audit.to_csv(
            output / f"{prefix}_false_score_null_audit.csv",
            index=False,
        )

        payload = deployment_payload(deployment, seeds, per_seed)
        write_json(output / f"{prefix}_topb_summary.json", payload)
        payloads[deployment] = payload
        combined_rows.append(per_seed)
        combined_summaries.append(summary)
        combined_null.append(null_audit)

    pd.concat(combined_rows, ignore_index=True).to_csv(
        args.output_root / "gwtc34_topb_true_false_per_seed.csv",
        index=False,
    )
    pd.concat(combined_summaries, ignore_index=True).to_csv(
        args.output_root / "gwtc34_topb_true_false_summary.csv",
        index=False,
    )
    pd.concat(combined_null, ignore_index=True).to_csv(
        args.output_root / "gwtc34_false_score_null_audit.csv",
        index=False,
    )
    write_json(
        args.output_root / "gwtc34_topb_summary.json",
        {
            "status": "complete",
            "deployments": payloads,
            "truth_label_boundary": (
                "True/false counts are valid only for held-out injection tests. "
                "Real GWTC catalog candidates do not have known lensing labels."
            ),
        },
    )
    write_json(
        args.output_root / "topb_method_contract.json",
        {
            "pair_definition": "one row per unordered pair with idx_i < idx_j",
            "self_pairs": "excluded",
            "primary_method": "candidate_three_channel_strict_positive",
            "input_scores": ["waveform_score", "time_score", "sky_score"],
            "fusion": "validation-frozen weighted sum for each training seed",
            "test_labels": "used only to count TP/FP after ranking",
            "warning": (
                "Do not apply true/false labels to real GWTC catalog pairs; "
                "those tables are candidate shortlists for Bayesian follow-up."
            ),
        },
    )


if __name__ == "__main__":
    main()
