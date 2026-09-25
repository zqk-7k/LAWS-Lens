#!/usr/bin/env python3
"""Re-score the ET-3 v7 catalogs with response-derived sky-v8 posteriors.

The five v7 waveform seeds, event splits, embeddings, waveform calibration,
and frozen ET time-delay likelihood ratio are reused unchanged.  This script
generates new eight-mode ET sky posterior surrogates, fits the sky calibration
on validation systems only, reselects fusion weights on validation, and
evaluates the untouched test systems.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.et_sky_v8 import generate_et_posterior_maps
from scripts.real_search.unified_sky_v8 import (
    SKY_FEATURE_COLUMNS,
    apply_sky_calibration_to_matrices,
    fit_sky_calibration,
    pair_statistic_matrices,
    write_json,
)


V7_ET = REPO / "results/et3_v7_aligned_20260723/formal"
DEFAULT_OUTPUT = REPO / "results/unified_sky_v8_20260724"
SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
SKY_NSIDE = 32


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


etv7 = load_module(
    REPO / "scripts/experiments/110_et3_v7_aligned_pipeline.py",
    "et3_v7_for_sky_v8",
)


@dataclass
class CatalogProxy:
    frame: pd.DataFrame
    partner: np.ndarray
    meta: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.frame)


def build_catalog(frame: pd.DataFrame) -> CatalogProxy:
    partner = np.full(len(frame), -1, dtype=np.int32)
    for _, part in frame[frame["family"].isin(etv7.FAMILIES)].groupby(
        "system_id",
        sort=False,
    ):
        indices = part.index.to_numpy(dtype=np.int32)
        if len(indices) != 2:
            raise RuntimeError(f"Lensed system has {len(indices)} events")
        partner[indices[0]] = indices[1]
        partner[indices[1]] = indices[0]
    return CatalogProxy(
        frame=frame,
        partner=partner,
        meta=frame.to_dict(orient="records"),
    )


def matrix_pair_frame(
    matrices: dict[str, np.ndarray],
    diagnostics: pd.DataFrame,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
) -> pd.DataFrame:
    idx_i = np.asarray(idx_i, dtype=np.int32)
    idx_j = np.asarray(idx_j, dtype=np.int32)
    frame = pd.DataFrame({"idx_i": idx_i, "idx_j": idx_j})
    for name in SKY_FEATURE_COLUMNS:
        frame[name] = matrices[name][idx_i, idx_j]
    area = diagnostics["area90_deg2"].to_numpy(dtype=np.float64)
    kl = diagnostics["kl_to_uniform_nats"].to_numpy(dtype=np.float64)
    frame["sky_area90_i_deg2"] = area[idx_i]
    frame["sky_area90_j_deg2"] = area[idx_j]
    frame["sky_area90_ratio"] = np.maximum(area[idx_i], area[idx_j]) / np.maximum(
        np.minimum(area[idx_i], area[idx_j]), 1e-12
    )
    frame["sky_kl_i_nats"] = kl[idx_i]
    frame["sky_kl_j_nats"] = kl[idx_j]
    frame["sky_min_kl_nats"] = np.minimum(kl[idx_i], kl[idx_j])
    return frame


def load_waveform_and_time(
    seed: int,
    split: str,
    catalog: CatalogProxy,
) -> tuple[np.ndarray, np.ndarray]:
    seed_dir = V7_ET / f"seed_{seed}"
    embedding = np.load(seed_dir / f"{split}_embeddings.npy")
    prediction = np.load(seed_dir / f"{split}_waveform_intrinsic_predictions.npy")
    waveform_config = json.loads(
        (seed_dir / "waveform_evidence_calibration.json").read_text(encoding="utf-8")
    )
    waveform = etv7.apply_waveform_calibration(
        embedding,
        prediction,
        waveform_config,
    )
    time_calibration = json.loads(
        (V7_ET / "shared/et_time_delay_likelihood_ratio.json").read_text(
            encoding="utf-8"
        )
    )
    time_score = etv7.time_score_matrix(catalog, time_calibration)
    return waveform, time_score


def generate_split_sky(
    seed: int,
    split: str,
    events: pd.DataFrame,
    fisher: pd.DataFrame,
    output_seed: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    maps, diagnostics, audit = generate_et_posterior_maps(
        events,
        fisher,
        seed=seed + (41000 if split == "validation" else 51000),
        nside=SKY_NSIDE,
        mode_count=8,
    )
    np.save(
        output_seed / f"{split}_et_sky_posteriors_nside{SKY_NSIDE}_v8.npy",
        maps.astype(np.float16),
    )
    diagnostics.to_parquet(
        output_seed / f"{split}_event_sky_diagnostics_v8.parquet",
        index=False,
    )
    write_json(output_seed / f"{split}_event_sky_audit_v8.json", audit)
    matrices, map_diagnostics = pair_statistic_matrices(
        maps,
        block_size=512,
    )
    # The diagnostics are independently recomputed by the matrix routine; this
    # assertion catches ordering or normalization mistakes.
    if not np.allclose(
        diagnostics["area90_deg2"],
        map_diagnostics["area90_deg2"],
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError("ET map diagnostics changed during pair scoring")
    return matrices, diagnostics


def retrieval_tables(
    components: dict[str, np.ndarray],
    catalog: CatalogProxy,
    methods: dict[str, dict[str, float]],
    seed: int,
    deployment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    query_frames = []
    metric_rows = []
    selected_scores: dict[str, np.ndarray] = {}
    for method, weights in methods.items():
        score = etv7.combine(components, weights)
        _, ranks = etv7.retrieval_from_matrix(score, catalog.partner, catalog.meta)
        ranks.insert(0, "method", method)
        ranks.insert(0, "seed", int(seed))
        ranks.insert(0, "deployment", deployment)
        query_frames.append(ranks)
        for subset in ("overall", *etv7.FAMILIES):
            part = ranks if subset == "overall" else ranks[ranks["family"] == subset]
            values = part["query_rank"].to_numpy(dtype=np.int32)
            metric_rows.append(
                {
                    "deployment": deployment,
                    "seed": int(seed),
                    "method": method,
                    "subset": subset,
                    "n_queries": int(len(values)),
                    "n_systems": int(part["system_id"].nunique()),
                    "r_at_1": float(np.mean(values <= 1)),
                    "r_at_5": float(np.mean(values <= 5)),
                    "r_at_10": float(np.mean(values <= 10)),
                    "r_at_50": float(np.mean(values <= 50)),
                    "median_rank": float(np.median(values)),
                }
            )
        if method in {
            "three_channel_unconstrained",
            "three_channel_strict_positive",
        }:
            selected_scores[method] = score
    return (
        pd.concat(query_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
        selected_scores,
    )


def run_seed(
    seed: int,
    output_root: Path,
    fisher: pd.DataFrame,
    pair_metrics: bool,
) -> dict[str, Any]:
    source_seed = V7_ET / f"seed_{seed}"
    output_seed = output_root / "et3" / f"seed_{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)
    validation_events = pd.read_parquet(source_seed / "validation_event_catalog.parquet")
    test_events = pd.read_parquet(source_seed / "test_event_catalog.parquet")
    validation = build_catalog(validation_events)
    test = build_catalog(test_events)

    val_waveform, val_time = load_waveform_and_time(seed, "validation", validation)
    test_waveform, test_time = load_waveform_and_time(seed, "test", test)
    val_sky_matrices, val_sky_events = generate_split_sky(
        seed,
        "validation",
        validation_events,
        fisher,
        output_seed,
    )
    test_sky_matrices, test_sky_events = generate_split_sky(
        seed,
        "test",
        test_events,
        fisher,
        output_seed,
    )

    val_true_i, val_true_j = etv7.true_pairs(validation.partner)
    val_false_i, val_false_j = etv7.sample_false_pairs(
        validation.partner,
        1_000_000,
        seed + 61000,
    )
    fit_i = np.concatenate([val_true_i, val_false_i])
    fit_j = np.concatenate([val_true_j, val_false_j])
    fit_labels = np.concatenate(
        [
            np.ones(len(val_true_i), dtype=np.int8),
            np.zeros(len(val_false_i), dtype=np.int8),
        ]
    )
    fit_frame = matrix_pair_frame(
        val_sky_matrices,
        val_sky_events,
        fit_i,
        fit_j,
    )
    fit_frame["is_true_pair"] = fit_labels
    fit_frame["sampling_strategy"] = np.concatenate(
        [
            np.full(len(val_true_i), "all_true_pairs", dtype=object),
            np.full(len(val_false_i), "fixed_random_false_pairs", dtype=object),
        ]
    )
    fit_frame.to_parquet(
        output_seed / "validation_sky_calibration_pairs_v8.parquet",
        index=False,
    )
    sky_calibration = fit_sky_calibration(
        fit_frame,
        fit_labels,
        deployment="ET3",
        seed=seed,
    )
    write_json(output_seed / "sky_calibration_v8.json", sky_calibration)
    val_sky = apply_sky_calibration_to_matrices(
        val_sky_matrices,
        sky_calibration,
    )
    test_sky = apply_sky_calibration_to_matrices(
        test_sky_matrices,
        sky_calibration,
    )
    del val_sky_matrices, test_sky_matrices

    val_components = {
        "waveform": val_waveform,
        "time": val_time,
        "sky": val_sky,
    }
    test_components = {
        "waveform": test_waveform,
        "time": test_time,
        "sky": test_sky,
    }
    selected, grids = etv7.select_fusion_modes(
        val_components,
        validation,
        seed + 62000,
    )
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "time_sky_validation_selected": selected["time_sky"],
        "three_channel_unconstrained": selected["unconstrained"],
        "three_channel_strict_positive": selected["strict_positive"],
        "three_channel_equal_evidence": {
            "waveform": 1.0,
            "time": 1.0,
            "sky": 1.0,
        },
    }
    write_json(
        output_seed / "selected_weights_v8.json",
        {
            "methods": methods,
            "selection_split": "validation systems only",
            "test_used_for_selection": False,
            "fusion_channel_standardization": (
                "none; waveform/time are v7 validation-frozen evidence and "
                "sky is the v8 validation-frozen calibrated logit"
            ),
        },
    )
    for name, grid in grids.items():
        grid.to_csv(output_seed / f"fusion_grid_{name}_v8.csv", index=False)

    validation_query, validation_metrics, _ = retrieval_tables(
        val_components,
        validation,
        methods,
        seed,
        "ET3-validation",
    )
    validation_query.to_parquet(
        output_seed / "validation_query_ranks_v8.parquet",
        index=False,
    )
    validation_metrics.to_csv(
        output_seed / "validation_retrieval_metrics_v8.csv",
        index=False,
    )
    query, metrics, score_cache = retrieval_tables(
        test_components,
        test,
        methods,
        seed,
        "ET3",
    )
    query.to_parquet(output_seed / "query_ranks_v8.parquet", index=False)
    metrics.to_csv(output_seed / "retrieval_metrics_v8.csv", index=False)
    bootstrap = etv7.bootstrap_system_ci(query, draws=10000, seed=seed + 63000)
    bootstrap.to_csv(output_seed / "bootstrap_95ci_v8.csv", index=False)

    pair_payload = {}
    if pair_metrics:
        for method, score in score_cache.items():
            summary, operating = etv7.full_unordered_pair_metrics(
                score,
                test.partner,
                test.meta,
                aggregation="max",
            )
            summary.update(
                {
                    "deployment": "ET3",
                    "seed": int(seed),
                    "method": method,
                    "unordered_score_rule": (
                        "max(S_ij,S_ji); all v8 channels are symmetric, so "
                        "max equals mean up to floating-point precision"
                    ),
                }
            )
            pair_payload[method] = summary
            write_json(
                output_seed / f"pair_level_metrics_{method}_v8.json",
                summary,
            )
            operating.to_csv(
                output_seed / f"pair_level_operating_points_{method}_v8.csv",
                index=False,
            )

    true_i, true_j = etv7.true_pairs(test.partner)
    false_i, false_j = etv7.sample_false_pairs(
        test.partner,
        200_000,
        seed + 64000,
    )
    pair_i = np.concatenate([true_i, false_i])
    pair_j = np.concatenate([true_j, false_j])
    diagnostics = pd.DataFrame(
        {
            "idx_i": pair_i,
            "idx_j": pair_j,
            "is_true_pair": np.concatenate(
                [
                    np.ones(len(true_i), dtype=np.int8),
                    np.zeros(len(false_i), dtype=np.int8),
                ]
            ),
            "waveform_score": test_waveform[pair_i, pair_j],
            "time_score": test_time[pair_i, pair_j],
            "sky_score": test_sky[pair_i, pair_j],
            "strict_positive_score": score_cache[
                "three_channel_strict_positive"
            ][pair_i, pair_j],
            "unconstrained_score": score_cache[
                "three_channel_unconstrained"
            ][pair_i, pair_j],
            "sampling_strategy": np.concatenate(
                [
                    np.full(len(true_i), "all_true_pairs", dtype=object),
                    np.full(len(false_i), "fixed_random_false_pairs", dtype=object),
                ]
            ),
        }
    )
    event_family = test_events["family"].astype(str).to_numpy()
    diagnostics["true_pair_family"] = np.where(
        diagnostics["is_true_pair"].to_numpy(dtype=bool),
        event_family[pair_i],
        "background",
    )
    diagnostics.to_parquet(
        output_seed / "pair_diagnostics_sample_v8.parquet",
        index=False,
    )
    summary = {
        "status": "complete",
        "version": "unified_sky_v8",
        "seed": int(seed),
        "n_validation_events": int(len(validation)),
        "n_test_events": int(len(test)),
        "n_test_queries": int(np.sum(test.partner >= 0)),
        "n_test_true_pairs": int(len(true_i)),
        "n_test_unordered_pairs": int(len(test) * (len(test) - 1) // 2),
        "selected_weights": methods,
        "sky_calibration_validation": sky_calibration["validation"],
        "validation_true_sky_90_coverage": float(
            val_sky_events["true_sky_inside_90"].mean()
        ),
        "test_true_sky_90_coverage": float(
            test_sky_events["true_sky_inside_90"].mean()
        ),
        "pair_level": pair_payload,
    }
    write_json(output_seed / "seed_summary_v8.json", summary)
    return summary


def aggregate(output_root: Path, seeds: tuple[int, ...]) -> None:
    root = output_root / "et3"
    metrics = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/retrieval_metrics_v8.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    metrics.to_csv(root / "et3_retrieval_metrics_per_seed_v8.csv", index=False)
    summary = (
        metrics.groupby(["method", "subset"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_queries=("n_queries", "first"),
            r_at_1_mean=("r_at_1", "mean"),
            r_at_1_std=("r_at_1", "std"),
            r_at_5_mean=("r_at_5", "mean"),
            r_at_5_std=("r_at_5", "std"),
            r_at_10_mean=("r_at_10", "mean"),
            r_at_10_std=("r_at_10", "std"),
            r_at_50_mean=("r_at_50", "mean"),
            r_at_50_std=("r_at_50", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
        )
    )
    summary.to_csv(root / "et3_retrieval_metrics_across_seed_v8.csv", index=False)
    bootstrap = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/bootstrap_95ci_v8.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(root / "et3_bootstrap_95ci_per_seed_v8.csv", index=False)
    weight_rows = []
    pair_rows = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        selected = json.loads((seed_root / "selected_weights_v8.json").read_text())
        for method, weights in selected["methods"].items():
            weight_rows.append({"seed": seed, "method": method, **weights})
        for method in (
            "three_channel_unconstrained",
            "three_channel_strict_positive",
        ):
            path = seed_root / f"pair_level_metrics_{method}_v8.json"
            if path.exists():
                pair_rows.append(json.loads(path.read_text()))
    pd.DataFrame(weight_rows).to_csv(
        root / "et3_selected_weights_per_seed_v8.csv",
        index=False,
    )
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        pairs.to_csv(root / "et3_pair_level_metrics_per_seed_v8.csv", index=False)
        numeric = [
            column
            for column in pairs.columns
            if column not in {"deployment", "method", "unordered_score_rule"}
            and pd.api.types.is_numeric_dtype(pairs[column])
        ]
        aggregate_rows = []
        for method, part in pairs.groupby("method"):
            row = {"method": method, "n_seeds": len(part)}
            for column in numeric:
                row[f"{column}_mean"] = float(part[column].mean())
                row[f"{column}_std"] = float(part[column].std(ddof=1))
            aggregate_rows.append(row)
        pd.DataFrame(aggregate_rows).to_csv(
            root / "et3_pair_level_metrics_across_seed_v8.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-pair-metrics", action="store_true")
    args = parser.parse_args()
    fisher_path = (
        args.output_root
        / "shared/et3_gwfast_fisher_localization_events.parquet"
    )
    if not fisher_path.exists():
        raise FileNotFoundError(
            f"Run 113_et3_sky_v8_gwfast_fisher.py first: {fisher_path}"
        )
    fisher = pd.read_parquet(fisher_path)
    seeds = (args.seed,) if args.seed is not None else SEEDS
    for seed in seeds:
        run_seed(
            seed,
            args.output_root,
            fisher,
            pair_metrics=not args.skip_pair_metrics,
        )
    if args.seed is None:
        aggregate(args.output_root, SEEDS)


if __name__ == "__main__":
    main()
