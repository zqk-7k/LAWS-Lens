#!/usr/bin/env python3
"""Re-score GWTC-3/O3 and GWTC-4.1/O4a with unified sky-v8.1.

Waveform encoders, waveform calibration, time-delay calibration, event splits,
and real strain embeddings are reused unchanged from v7.  Only the sky
posterior statistic, validation-selected fusion weights, and downstream
rankings are recomputed.  The sole main sky score is raw log B_sky.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.common import read_probability_map
from scripts.real_search.unified_sky_v81 import (
    MAIN_SKY_COLUMN,
    attach_main_sky_score,
    change_nside_probability_mass,
    pair_feature_frame,
    score_contract,
    write_json,
)
from scripts.real_search import unified_v7_common as v7
from scripts.experiments.mainline_uncertainty_common import bootstrap_system_ci


V7_ROOT = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
DEFAULT_OUTPUT = REPO / "results/unified_sky_v81_20260725"
DEPLOYMENTS = {
    "gwtc3": {
        "source_key": "GWTC3",
        "seeds": (202607241, 202607242, 202607243),
    },
    "gwtc4": {
        "source_key": "GWTC4",
        "seeds": (202607241, 202607242, 202607243),
    },
}
COMMON_NSIDE = 32


def _method_weights(validation: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    selected: dict[str, dict[str, float]] = {}
    grids: dict[str, pd.DataFrame] = {}
    for objective in ("retrieval", "candidate"):
        time_sky, grid = v7.select_fusion_weights(
            validation,
            waveform_zero=True,
            objective=objective,
        )
        selected[f"time_sky_{objective}_selected"] = time_sky
        grids[f"time_sky_{objective}"] = grid
        unconstrained, grid = v7.select_fusion_weights(
            validation,
            objective=objective,
        )
        selected[f"{objective}_three_channel_unconstrained"] = unconstrained
        grids[f"{objective}_three_channel_unconstrained"] = grid
        positive, grid = v7.select_fusion_weights(
            validation,
            require_all_positive=True,
            objective=objective,
        )
        selected[f"{objective}_three_channel_strict_positive"] = positive
        grids[f"{objective}_three_channel_strict_positive"] = grid
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        **selected,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
    }
    return methods, grids


def _load_pair_sky(
    seed_dir: Path,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_dir = seed_dir / "results"
    maps = change_nside_probability_mass(
        np.load(
            result_dir / f"mixed_{split}_synthetic_sky_posteriors_nside64_v7.npy",
            mmap_mode="r",
        ),
        COMMON_NSIDE,
    )
    base_name = (
        "fusion_validation_pairs_v7.parquet"
        if split == "val"
        else "fusion_heldout_test_pairs_v7.parquet"
    )
    base = pd.read_parquet(result_dir / base_name)
    sky, diagnostics = pair_feature_frame(
        maps,
        idx_i=base["idx_i"].to_numpy(dtype=np.int32),
        idx_j=base["idx_j"].to_numpy(dtype=np.int32),
    )
    return sky, diagnostics


def _load_real_maps(source_key: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    primary = v7.v3.primary_manifest(v7.v3.SOURCES[source_key])
    maps = []
    audits = []
    for index, row in primary.iterrows():
        probability, metadata = read_probability_map(row, target_nside=COMMON_NSIDE)
        maps.append(probability)
        audits.append(
            {
                "idx": int(index),
                "event_name": str(row["event_name"]),
                **metadata,
            }
        )
    return primary, np.stack(maps).astype(np.float32), pd.DataFrame(audits)


def _base_real_pairs(seed_dir: Path) -> pd.DataFrame:
    path = (
        seed_dir
        / "results/real_pair_scores_all_catalog_three_channel_equal_evidence_v7.parquet"
    )
    frame = pd.read_parquet(path)
    drop = [
        "raw_posterior_overlap",
        "cosine_overlap",
        "angular_sep_map_deg",
        "common_nside",
        "waveform_contribution",
        "time_contribution",
        "sky_contribution",
        "final_score",
        "final_score_i_to_j",
        "final_score_j_to_i",
        "unordered_max_score",
        "unordered_mean_score",
        "method",
        "rank",
    ]
    return frame.drop(columns=drop, errors="ignore")


def _rank_real(
    seed_dir: Path,
    output_seed: Path,
    methods: dict[str, dict[str, float]],
    primary: pd.DataFrame,
    maps: np.ndarray,
) -> dict[str, dict[str, pd.DataFrame]]:
    base = _base_real_pairs(seed_dir)
    sky, event_diagnostics = pair_feature_frame(
        maps,
        idx_i=base["idx_i"].to_numpy(dtype=np.int32),
        idx_j=base["idx_j"].to_numpy(dtype=np.int32),
    )
    real = attach_main_sky_score(base, sky)
    event_diagnostics["event_name"] = primary["event_name"].astype(str).to_numpy()
    event_diagnostics.to_parquet(
        output_seed / "real_event_sky_diagnostics_v81.parquet",
        index=False,
    )
    real.to_parquet(output_seed / "real_pair_features_unified_sky_v81.parquet", index=False)
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for method, weights in methods.items():
        all_catalog = v7._rank_real_pairs(real, weights, method)
        strict = v7._rank_real_pairs(
            real[real["strict_h1l1_bbh_pair"]].copy(),
            weights,
            method,
        )
        all_catalog.to_parquet(
            output_seed / f"real_pair_scores_all_catalog_{method}_v81.parquet",
            index=False,
        )
        strict.to_parquet(
            output_seed / f"real_pair_scores_strict_h1l1_bbh_{method}_v81.parquet",
            index=False,
        )
        all_catalog.head(100).to_csv(
            output_seed / f"candidate_shortlist_all_catalog_{method}_v81.csv",
            index=False,
        )
        strict.head(100).to_csv(
            output_seed / f"candidate_shortlist_strict_h1l1_bbh_{method}_v81.csv",
            index=False,
        )
        outputs[method] = {"all": all_catalog, "strict": strict}
    return outputs


def run_seed(
    deployment: str,
    source_key: str,
    seed: int,
    output_root: Path,
    primary: pd.DataFrame,
    real_maps: np.ndarray,
    real_map_audit: pd.DataFrame,
) -> dict[str, Any]:
    source_seed = V7_ROOT / deployment / f"seed_{seed}"
    output_seed = output_root / deployment / f"seed_{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)

    val_sky, val_event_sky = _load_pair_sky(source_seed, "val")
    test_sky, test_event_sky = _load_pair_sky(source_seed, "test")
    validation_base = pd.read_parquet(
        source_seed / "results/fusion_validation_pairs_v7.parquet"
    )
    test_base = pd.read_parquet(
        source_seed / "results/fusion_heldout_test_pairs_v7.parquet"
    )
    validation_labels = validation_base["is_true_pair"].to_numpy(dtype=np.int8)
    validation_raw = val_sky[MAIN_SKY_COLUMN].to_numpy(dtype=np.float64)
    sky_validation = {
        "score": "raw log B_sky",
        "n_pairs": int(len(validation_labels)),
        "n_true_pairs": int(validation_labels.sum()),
        "n_false_pairs": int((validation_labels == 0).sum()),
        "roc_auc": float(roc_auc_score(validation_labels, validation_raw)),
        "average_precision": float(
            average_precision_score(validation_labels, validation_raw)
        ),
        "true_score_median": float(
            np.median(validation_raw[validation_labels == 1])
        ),
        "null_score_median": float(
            np.median(validation_raw[validation_labels == 0])
        ),
        "validation_fitted_sky_composite": False,
    }
    contract = score_contract()
    contract["deployment"] = deployment
    contract["seed"] = int(seed)
    contract["validation"] = sky_validation
    write_json(output_seed / "sky_score_contract_v81.json", contract)
    val_event_sky.to_parquet(
        output_seed / "synthetic_validation_event_sky_diagnostics_v81.parquet",
        index=False,
    )
    test_event_sky.to_parquet(
        output_seed / "synthetic_test_event_sky_diagnostics_v81.parquet",
        index=False,
    )

    validation = attach_main_sky_score(validation_base, val_sky)
    heldout = attach_main_sky_score(test_base, test_sky)
    validation.to_parquet(
        output_seed / "fusion_validation_pairs_v81.parquet",
        index=False,
    )
    heldout.to_parquet(
        output_seed / "fusion_heldout_test_pairs_v81.parquet",
        index=False,
    )

    methods, grids = _method_weights(validation)
    write_json(
        output_seed / "selected_weights_v81.json",
        {
            "selection_split": "synthetic validation systems only",
            "channel_standardization": (
                "none; waveform/time are v7 validation-frozen evidence and "
                "sky is the raw common-source log B_sky"
            ),
            "sky_diagnostics_not_used_for_ranking": [
                "sky_o90",
                "sky_cross_hpd",
            ],
            "methods": methods,
        },
    )
    for name, grid in grids.items():
        grid.to_csv(output_seed / f"fusion_weight_grid_{name}_v81.csv", index=False)

    validation_query, validation_metrics, validation_pair = v7.evaluation_tables(
        validation,
        methods,
        deployment,
        seed,
    )
    test_query, test_metrics, test_pair = v7.evaluation_tables(
        heldout,
        methods,
        deployment,
        seed,
    )
    validation_query.to_parquet(
        output_seed / "validation_query_ranks_v81.parquet",
        index=False,
    )
    validation_metrics.to_csv(
        output_seed / "validation_retrieval_metrics_v81.csv",
        index=False,
    )
    validation_pair.to_csv(
        output_seed / "validation_pair_level_metrics_v81.csv",
        index=False,
    )
    test_query.to_parquet(output_seed / "heldout_test_query_ranks_v81.parquet", index=False)
    test_metrics.to_csv(output_seed / "heldout_test_retrieval_metrics_v81.csv", index=False)
    test_pair.to_csv(output_seed / "heldout_test_pair_level_metrics_v81.csv", index=False)
    bootstrap = bootstrap_system_ci(
        test_query,
        draws=10_000,
        seed=seed + 64_000,
    )
    bootstrap.to_csv(
        output_seed / "heldout_test_bootstrap_95ci_v81.csv",
        index=False,
    )

    gate = v7.waveform_gate(
        validation,
        methods["retrieval_three_channel_unconstrained"],
        methods["time_sky_retrieval_selected"],
        seed=seed,
    )
    write_json(output_seed / "waveform_gate_validation_v81.json", gate)
    real_outputs = _rank_real(
        source_seed,
        output_seed,
        methods,
        primary,
        real_maps,
    )
    real_map_audit.to_csv(output_seed / "real_map_source_audit_v81.csv", index=False)

    primary_method = (
        "candidate_three_channel_unconstrained"
        if gate["passed_for_primary_deployment"]
        else "time_sky_candidate_selected"
    )
    primary_all = real_outputs[primary_method]["all"]
    primary_strict = real_outputs[primary_method]["strict"]
    primary_all.head(100).to_csv(
        output_seed / "candidate_shortlist_all_primary_v81.csv",
        index=False,
    )
    primary_strict.head(100).to_csv(
        output_seed / "candidate_shortlist_strict_primary_v81.csv",
        index=False,
    )
    summary = {
        "version": "unified_sky_v81",
        "deployment": deployment,
        "seed": int(seed),
        "primary_method": primary_method,
        "waveform_gate_passed": bool(gate["passed_for_primary_deployment"]),
        "n_real_events": int(len(primary)),
        "n_real_pairs": int(len(primary_all)),
        "n_strict_h1l1_bbh_pairs": int(len(primary_strict)),
        "top_all_pair": primary_all.iloc[0][
            ["event_i", "event_j", "final_score", "waveform_available", "pair_has_ood"]
        ].to_dict(),
        "top_strict_pair": primary_strict.iloc[0][
            ["event_i", "event_j", "final_score", "waveform_available", "pair_has_ood"]
        ].to_dict(),
        "sky_validation": sky_validation,
        "selected_weights": methods,
    }
    write_json(output_seed / "seed_summary_v81.json", summary)
    return summary


def _numeric_summary(
    frame: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column not in groups and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = []
    for key, part in frame.groupby(groups, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_values))
        row["n_seeds"] = len(part)
        for column in numeric:
            values = part[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_min"] = float(np.nanmin(values))
            row[f"{column}_max"] = float(np.nanmax(values))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(deployment: str, seeds: tuple[int, ...], output_root: Path) -> None:
    root = output_root / deployment
    retrieval = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/heldout_test_retrieval_metrics_v81.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    pairs = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/heldout_test_pair_level_metrics_v81.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    bootstrap = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/heldout_test_bootstrap_95ci_v81.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    retrieval.to_csv(root / "heldout_test_retrieval_metrics_per_seed_v81.csv", index=False)
    pairs.to_csv(root / "heldout_test_pair_level_metrics_per_seed_v81.csv", index=False)
    bootstrap.to_csv(
        root / "heldout_test_bootstrap_95ci_per_seed_v81.csv",
        index=False,
    )
    _numeric_summary(retrieval, ["deployment", "method", "subset"]).to_csv(
        root / "heldout_test_retrieval_metrics_across_seed_v81.csv",
        index=False,
    )
    _numeric_summary(pairs, ["deployment", "method"]).to_csv(
        root / "heldout_test_pair_level_metrics_across_seed_v81.csv",
        index=False,
    )

    weight_rows = []
    gate_rows = []
    real_rows: dict[str, list[pd.DataFrame]] = {
        "candidate_three_channel_strict_positive": [],
        "candidate_three_channel_unconstrained": [],
        "time_sky_candidate_selected": [],
        "primary_gate_selected": [],
    }
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        selected = json.loads((seed_root / "selected_weights_v81.json").read_text())
        for method, weights in selected["methods"].items():
            weight_rows.append({"deployment": deployment, "seed": seed, "method": method, **weights})
        gate = json.loads((seed_root / "waveform_gate_validation_v81.json").read_text())
        gate_rows.append(
            {
                "deployment": deployment,
                "seed": seed,
                "passed_for_primary_deployment": gate["passed_for_primary_deployment"],
                "statistical_discrimination_passed": gate["statistical_discrimination_passed"],
                "incremental_utility_passed": gate["incremental_utility_passed"],
            }
        )
        summary = json.loads((seed_root / "seed_summary_v81.json").read_text())
        methods_for_consensus = {
            "candidate_three_channel_strict_positive": (
                "candidate_three_channel_strict_positive"
            ),
            "candidate_three_channel_unconstrained": (
                "candidate_three_channel_unconstrained"
            ),
            "time_sky_candidate_selected": "time_sky_candidate_selected",
            "primary_gate_selected": summary["primary_method"],
        }
        for label, method in methods_for_consensus.items():
            scored = pd.read_parquet(
                seed_root
                / f"real_pair_scores_strict_h1l1_bbh_{method}_v81.parquet"
            )
            real_rows[label].append(
                scored[
                [
                    "event_i",
                    "event_j",
                    "rank",
                    "final_score",
                    "waveform_score",
                    "time_score",
                    "sky_score",
                ]
                ].assign(seed=seed, method=method)
            )
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(root / "selected_weights_per_seed_v81.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(root / "waveform_gate_per_seed_v81.csv", index=False)

    for label, frames in real_rows.items():
        rank_stability = pd.concat(frames, ignore_index=True)
        consensus = (
            rank_stability.groupby(["event_i", "event_j"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                rank_mean=("rank", "mean"),
                rank_std=("rank", "std"),
                rank_min=("rank", "min"),
                rank_max=("rank", "max"),
                final_score_mean=("final_score", "mean"),
                waveform_score_mean=("waveform_score", "mean"),
                time_score_mean=("time_score", "mean"),
                sky_score_mean=("sky_score", "mean"),
            )
            .sort_values(["rank_mean", "rank_max"], kind="stable")
            .reset_index(drop=True)
        )
        consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
        rank_stability.to_parquet(
            root / f"real_candidate_rank_stability_{label}_v81.parquet",
            index=False,
        )
        consensus.to_parquet(
            root / f"real_candidate_consensus_{label}_v81.parquet",
            index=False,
        )
        consensus.head(100).to_csv(
            root / f"real_candidate_consensus_top100_{label}_v81.csv",
            index=False,
        )
        if label == "candidate_three_channel_strict_positive":
            rank_stability.to_parquet(
                root / "real_candidate_rank_stability_v81.parquet",
                index=False,
            )
            consensus.to_parquet(
                root / "real_candidate_consensus_v81.parquet",
                index=False,
            )
            consensus.head(100).to_csv(
                root / "real_candidate_consensus_top100_v81.csv",
                index=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deployment", choices=["gwtc3", "gwtc4", "all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    deployments = DEPLOYMENTS if args.deployment == "all" else {args.deployment: DEPLOYMENTS[args.deployment]}
    for deployment, config in deployments.items():
        primary, real_maps, real_map_audit = _load_real_maps(config["source_key"])
        seeds = (args.seed,) if args.seed is not None else config["seeds"]
        for seed in seeds:
            run_seed(
                deployment,
                config["source_key"],
                seed,
                args.output_root,
                primary,
                real_maps,
                real_map_audit,
            )
        if args.seed is None:
            aggregate(deployment, config["seeds"], args.output_root)


if __name__ == "__main__":
    main()
