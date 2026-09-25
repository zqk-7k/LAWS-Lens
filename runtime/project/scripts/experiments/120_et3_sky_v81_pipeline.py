#!/usr/bin/env python3
"""Re-score the ET-3 v7 catalogs with response-derived sky-v8.1 posteriors.

The five v7 waveform seeds, event splits, embeddings, waveform calibration,
and frozen ET time-delay likelihood ratio are reused unchanged.  This script
loads the shared v8.1 posterior bank, uses raw log B_sky as the sole main sky
score, reselects fusion weights on validation, and evaluates untouched test
systems.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.unified_sky_v81 import (
    MAIN_SKY_COLUMN,
    SKY_FEATURE_COLUMNS,
    main_sky_matrix,
    normalize_probability_maps,
    pair_statistic_matrices,
    write_json,
)


V7_ET = REPO / "results/et3_v7_aligned_20260723/formal"
DEFAULT_OUTPUT = REPO / "results/unified_sky_v81_20260725"
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
    "et3_v7_for_sky_v81",
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


def _event_keys(events: pd.DataFrame) -> pd.Series:
    return (
        events["family"].astype(str)
        + ":"
        + events["source_index"].astype(str)
        + ":"
        + events["image"].astype(str)
    )


def _bank_indices(
    events: pd.DataFrame,
    bank_index: pd.DataFrame,
) -> np.ndarray:
    lookup = bank_index.set_index("event_key")["bank_index"]
    keys = _event_keys(events)
    indices = keys.map(lookup)
    if indices.isna().any():
        missing = keys.loc[indices.isna()].tolist()[:10]
        raise RuntimeError(f"ET posterior bank is missing events: {missing}")
    return indices.to_numpy(dtype=np.int32)


def _tempered_probability_maps(
    maps: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply validation-frozen posterior temperature.

    For a Gaussian posterior, ``p**(1/T)`` multiplies its covariance by T.
    This supplies the global coverage inflation permitted by the frozen
    protocol while preserving every mode location, mode ordering, and
    event-specific shape.
    """
    values = np.asarray(maps, dtype=np.float32)
    if not np.isclose(temperature, 1.0):
        values = np.power(
            np.maximum(values, 0.0),
            np.float32(1.0 / float(temperature)),
        ).astype(np.float32, copy=False)
    return normalize_probability_maps(values)


def _prepare_hpd_order(
    maps: np.ndarray,
    true_pixels: np.ndarray,
    *,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze the exact HPD tie ordering used by ``credible_levels``."""
    probability = normalize_probability_maps(maps)
    sorted_probability = np.empty_like(probability, dtype=np.float32)
    true_rank = np.empty(len(probability), dtype=np.int32)
    for start in range(0, len(probability), block_size):
        stop = min(start + block_size, len(probability))
        block = probability[start:stop]
        order = np.argsort(-block, axis=1)
        sorted_probability[start:stop] = np.take_along_axis(
            block,
            order,
            axis=1,
        )
        true_rank[start:stop] = np.argmax(
            order == true_pixels[start:stop, None],
            axis=1,
        )
    return sorted_probability, true_rank


def _true_hpd_credible_mass(
    maps: np.ndarray,
    true_pixels: np.ndarray,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Return each true pixel's HPD mass using the production tie ordering."""
    probability = normalize_probability_maps(maps)
    output = np.empty(len(probability), dtype=np.float64)
    for start in range(0, len(probability), block_size):
        stop = min(start + block_size, len(probability))
        block = probability[start:stop]
        order = np.argsort(-block, axis=1)
        sorted_probability = np.take_along_axis(block, order, axis=1)
        true_rank = np.argmax(
            order == true_pixels[start:stop, None],
            axis=1,
        )
        cumulative = np.cumsum(
            sorted_probability,
            axis=1,
            dtype=np.float64,
        )
        rows = np.arange(stop - start)
        output[start:stop] = cumulative[rows, true_rank] / np.maximum(
            cumulative[:, -1],
            1e-30,
        )
    return output


def _coverage_from_sorted_probability(
    sorted_probability: torch.Tensor,
    true_rank: torch.Tensor,
    temperature: float,
) -> float:
    """Evaluate coverage with the same deterministic HPD pixel ordering."""
    exponent = 1.0 / float(temperature)
    if np.isclose(temperature, 1.0):
        powered = sorted_probability
    else:
        powered = torch.clamp(sorted_probability, min=0.0).pow(exponent)
    cumulative = torch.cumsum(powered, dim=1, dtype=torch.float64)
    rows = torch.arange(len(powered), device=powered.device)
    credible_mass = cumulative[rows, true_rank] / torch.clamp(
        cumulative[:, -1],
        min=1e-30,
    )
    coverage = float(
        torch.mean((credible_mass <= (0.9 + 1e-7)).to(torch.float32)).item()
    )
    del cumulative, rows, credible_mass
    if powered is not sorted_probability:
        del powered
    return coverage


def select_validation_posterior_temperature(
    events: pd.DataFrame,
    posterior_bank: np.ndarray,
    bank_index: pd.DataFrame,
    *,
    target_coverage: float = 0.9,
) -> tuple[float, dict[str, Any]]:
    """Select one global temperature from validation true-sky coverage only."""
    indices = _bank_indices(events, bank_index)
    maps = np.asarray(posterior_bank[indices], dtype=np.float32)
    true_pixels = hp.ang2pix(
        SKY_NSIDE,
        math.pi / 2.0 - events["dec"].to_numpy(dtype=np.float64),
        events["ra"].to_numpy(dtype=np.float64),
    )
    sorted_probability, true_rank = _prepare_hpd_order(maps, true_pixels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sorted_probability_t = torch.as_tensor(
        sorted_probability,
        dtype=torch.float32,
        device=device,
    )
    true_rank_t = torch.as_tensor(
        true_rank,
        dtype=torch.long,
        device=device,
    )

    def evaluate(temperature: float) -> float:
        return _coverage_from_sorted_probability(
            sorted_probability_t,
            true_rank_t,
            temperature,
        )

    base_coverage = evaluate(1.0)
    if base_coverage >= target_coverage:
        selected = 1.0
        selected_coverage = base_coverage
    else:
        low = 1.0
        high = 1.25
        high_coverage = evaluate(high)
        while high_coverage < target_coverage and high < 8.0:
            low = high
            high = min(2.0 * high, 8.0)
            high_coverage = evaluate(high)
        candidates = [(low, evaluate(low))]
        candidates.append((high, high_coverage))
        for _ in range(12):
            middle = 0.5 * (low + high)
            middle_coverage = evaluate(middle)
            candidates.append((middle, middle_coverage))
            if middle_coverage < target_coverage:
                low = middle
            else:
                high = middle
        selected, selected_coverage = min(
            candidates,
            key=lambda item: (
                abs(item[1] - target_coverage),
                item[0],
            ),
        )
    del sorted_probability_t, true_rank_t, sorted_probability, true_rank, maps
    if device.type == "cuda":
        torch.cuda.empty_cache()
    audit = {
        "selection_split": "validation events only",
        "test_events_used": False,
        "target_true_sky_90_coverage": float(target_coverage),
        "base_temperature": 1.0,
        "base_true_sky_90_coverage": float(base_coverage),
        "selected_temperature": float(selected),
        "selected_validation_true_sky_90_coverage": float(selected_coverage),
        "interpretation": (
            "p_calibrated(Omega) is proportional to "
            "p_base(Omega)**(1/T); for a Gaussian this inflates covariance by T"
        ),
        "event_count": int(len(events)),
    }
    return float(selected), audit


def generate_split_sky(
    split: str,
    events: pd.DataFrame,
    posterior_bank: np.ndarray,
    bank_index: pd.DataFrame,
    bank_diagnostics: pd.DataFrame,
    output_seed: Path,
    posterior_temperature: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    keyed = events.copy()
    keyed["event_key"] = _event_keys(keyed)
    indices = _bank_indices(keyed, bank_index)
    maps = _tempered_probability_maps(
        posterior_bank[indices],
        posterior_temperature,
    )
    pd.DataFrame(
        {
            "catalog_index": np.arange(len(events), dtype=np.int32),
            "bank_index": indices,
            "event_key": keyed["event_key"].astype(str),
        }
    ).to_parquet(
        output_seed / f"{split}_posterior_bank_indices_v81.parquet",
        index=False,
    )
    matrices, map_diagnostics = pair_statistic_matrices(
        maps,
        block_size=512,
    )
    shared_diagnostics = (
        bank_diagnostics.set_index("bank_index")
        .loc[indices]
        .reset_index(drop=False)
    )
    for column in shared_diagnostics.columns:
        if column not in map_diagnostics.columns:
            map_diagnostics[column] = shared_diagnostics[column].to_numpy()
    true_pixels = hp.ang2pix(
        SKY_NSIDE,
        math.pi / 2.0 - events["dec"].to_numpy(dtype=np.float64),
        events["ra"].to_numpy(dtype=np.float64),
    )
    true_credible_level = _true_hpd_credible_mass(maps, true_pixels)
    map_diagnostics["true_sky_credible_level"] = true_credible_level
    map_diagnostics["true_sky_inside_90"] = (
        true_credible_level <= (0.9 + 1e-7)
    )
    map_diagnostics["event_key"] = keyed["event_key"].astype(str).to_numpy()
    map_diagnostics.to_parquet(
        output_seed / f"{split}_event_sky_diagnostics_v81.parquet",
        index=False,
    )
    write_json(
        output_seed / f"{split}_event_sky_audit_v81.json",
        {
            "version": "unified_sky_v81",
            "split": split,
            "n_events": int(len(events)),
            "nside": SKY_NSIDE,
            "posterior_source": "shared deterministic ET v8.1 bank",
            "validation_frozen_posterior_temperature": float(
                posterior_temperature
            ),
            "main_sky_score": "raw log B_sky",
        },
    )
    return matrices, map_diagnostics


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
    posterior_bank: np.ndarray,
    bank_index: pd.DataFrame,
    bank_diagnostics: pd.DataFrame,
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
    posterior_temperature, temperature_audit = (
        select_validation_posterior_temperature(
            validation_events,
            posterior_bank,
            bank_index,
        )
    )
    write_json(
        output_seed / "posterior_temperature_calibration_v81.json",
        temperature_audit,
    )
    val_sky_matrices, val_sky_events = generate_split_sky(
        "validation",
        validation_events,
        posterior_bank,
        bank_index,
        bank_diagnostics,
        output_seed,
        posterior_temperature,
    )
    test_sky_matrices, test_sky_events = generate_split_sky(
        "test",
        test_events,
        posterior_bank,
        bank_index,
        bank_diagnostics,
        output_seed,
        posterior_temperature,
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
        output_seed / "validation_sky_diagnostic_pairs_v81.parquet",
        index=False,
    )
    validation_raw = fit_frame[MAIN_SKY_COLUMN].to_numpy(dtype=np.float64)
    sky_validation = {
        "score": "raw log B_sky",
        "n_pairs": int(len(fit_labels)),
        "n_true_pairs": int(fit_labels.sum()),
        "n_false_pairs": int((fit_labels == 0).sum()),
        "roc_auc": float(roc_auc_score(fit_labels, validation_raw)),
        "average_precision": float(
            average_precision_score(fit_labels, validation_raw)
        ),
        "true_score_median": float(np.median(validation_raw[fit_labels == 1])),
        "null_score_median": float(np.median(validation_raw[fit_labels == 0])),
        "validation_fitted_sky_composite": False,
    }
    write_json(
        output_seed / "sky_score_contract_v81.json",
        {
            "version": "unified_sky_v81",
            "definition": "sky_score=log(N_pix*sum_k(P_i[k]*P_j[k]))",
            "diagnostic_only": ["sky_o90", "sky_cross_hpd"],
            "validation": sky_validation,
            "test_labels_used": False,
        },
    )
    val_sky = main_sky_matrix(val_sky_matrices)
    test_sky = main_sky_matrix(test_sky_matrices)
    del val_sky_matrices

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
        output_seed / "selected_weights_v81.json",
        {
            "methods": methods,
            "selection_split": "validation systems only",
            "test_used_for_selection": False,
            "fusion_channel_standardization": (
                "none; waveform/time are v7 validation-frozen evidence and "
                "sky is the raw common-source log B_sky"
            ),
            "validation_frozen_et_posterior_temperature": float(
                posterior_temperature
            ),
            "sky_diagnostics_not_used_for_ranking": [
                "sky_o90",
                "sky_cross_hpd",
            ],
        },
    )
    for name, grid in grids.items():
        grid.to_csv(output_seed / f"fusion_grid_{name}_v81.csv", index=False)

    validation_query, validation_metrics, _ = retrieval_tables(
        val_components,
        validation,
        methods,
        seed,
        "ET3-validation",
    )
    validation_query.to_parquet(
        output_seed / "validation_query_ranks_v81.parquet",
        index=False,
    )
    validation_metrics.to_csv(
        output_seed / "validation_retrieval_metrics_v81.csv",
        index=False,
    )
    query, metrics, score_cache = retrieval_tables(
        test_components,
        test,
        methods,
        seed,
        "ET3",
    )
    query.to_parquet(output_seed / "query_ranks_v81.parquet", index=False)
    metrics.to_csv(output_seed / "retrieval_metrics_v81.csv", index=False)
    bootstrap = etv7.bootstrap_system_ci(query, draws=10000, seed=seed + 63000)
    bootstrap.to_csv(output_seed / "bootstrap_95ci_v81.csv", index=False)

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
                        "max(S_ij,S_ji); all v8.1 channels are symmetric, so "
                        "max equals mean up to floating-point precision"
                    ),
                }
            )
            pair_payload[method] = summary
            write_json(
                output_seed / f"pair_level_metrics_{method}_v81.json",
                summary,
            )
            operating.to_csv(
                output_seed / f"pair_level_operating_points_{method}_v81.csv",
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
    test_area90 = test_sky_events["area90_deg2"].to_numpy(dtype=np.float64)
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
            "sky_log_bayes_factor_raw": test_sky_matrices[
                "sky_log_bayes_factor_raw"
            ][pair_i, pair_j],
            "sky_o90_diagnostic_only": test_sky_matrices["sky_o90"][
                pair_i,
                pair_j,
            ],
            "sky_cross_hpd_diagnostic_only": test_sky_matrices[
                "sky_cross_hpd"
            ][pair_i, pair_j],
            "sky_area90_i_deg2": test_area90[pair_i],
            "sky_area90_j_deg2": test_area90[pair_j],
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
        output_seed / "pair_diagnostics_sample_v81.parquet",
        index=False,
    )
    del test_sky_matrices
    summary = {
        "status": "complete",
        "version": "unified_sky_v81",
        "seed": int(seed),
        "n_validation_events": int(len(validation)),
        "n_test_events": int(len(test)),
        "n_test_queries": int(np.sum(test.partner >= 0)),
        "n_test_true_pairs": int(len(true_i)),
        "n_test_unordered_pairs": int(len(test) * (len(test) - 1) // 2),
        "selected_weights": methods,
        "posterior_temperature_calibration": temperature_audit,
        "sky_validation": sky_validation,
        "validation_true_sky_90_coverage": float(
            val_sky_events["true_sky_inside_90"].mean()
        ),
        "test_true_sky_90_coverage": float(
            test_sky_events["true_sky_inside_90"].mean()
        ),
        "pair_level": pair_payload,
    }
    write_json(output_seed / "seed_summary_v81.json", summary)
    return summary


def aggregate(output_root: Path, seeds: tuple[int, ...]) -> None:
    root = output_root / "et3"
    metrics = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/retrieval_metrics_v81.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    metrics.to_csv(root / "et3_retrieval_metrics_per_seed_v81.csv", index=False)
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
    summary.to_csv(root / "et3_retrieval_metrics_across_seed_v81.csv", index=False)
    bootstrap = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/bootstrap_95ci_v81.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(root / "et3_bootstrap_95ci_per_seed_v81.csv", index=False)
    weight_rows = []
    pair_rows = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        selected = json.loads((seed_root / "selected_weights_v81.json").read_text())
        for method, weights in selected["methods"].items():
            weight_rows.append({"seed": seed, "method": method, **weights})
        for method in (
            "three_channel_unconstrained",
            "three_channel_strict_positive",
        ):
            path = seed_root / f"pair_level_metrics_{method}_v81.json"
            if path.exists():
                pair_rows.append(json.loads(path.read_text()))
    pd.DataFrame(weight_rows).to_csv(
        root / "et3_selected_weights_per_seed_v81.csv",
        index=False,
    )
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        pairs.to_csv(root / "et3_pair_level_metrics_per_seed_v81.csv", index=False)
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
            root / "et3_pair_level_metrics_across_seed_v81.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-pair-metrics", action="store_true")
    args = parser.parse_args()
    shared = args.output_root / "shared"
    bank_path = shared / f"et3_posterior_bank_nside{SKY_NSIDE}_v81.npy"
    index_path = shared / "et3_posterior_bank_index_v81.parquet"
    diagnostics_path = shared / "et3_posterior_bank_diagnostics_v81.parquet"
    for path in (bank_path, index_path, diagnostics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Run 119_et3_sky_v81_posterior_bank.py first: {path}"
            )
    posterior_bank = np.load(bank_path, mmap_mode="r")
    bank_index = pd.read_parquet(index_path)
    bank_diagnostics = pd.read_parquet(diagnostics_path)
    seeds = (args.seed,) if args.seed is not None else SEEDS
    for seed in seeds:
        run_seed(
            seed,
            args.output_root,
            posterior_bank,
            bank_index,
            bank_diagnostics,
            pair_metrics=not args.skip_pair_metrics,
        )
    if args.seed is None:
        aggregate(args.output_root, SEEDS)


if __name__ == "__main__":
    main()
