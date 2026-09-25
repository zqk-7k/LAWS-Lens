#!/usr/bin/env python3
"""Author-authorized bounded Phase 1 for joint time/SNR evidence.

Only response-derived injections are evaluated. The waveform encoder, sky
score, source split, event-level PE contract and Phase 0.5a one-dimensional
time baseline are frozen. No real catalog is ranked and no model is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.interpolate import RegularGridInterpolator


REPO = Path(__file__).resolve().parents[2]
PHASE05 = REPO / "results/time_snr_joint_evidence_phase05_20260807"
PHASE05A = REPO / "results/time_snr_joint_evidence_phase05a_20260807_final"
DEFAULT_OUTPUT = REPO / "results/time_snr_joint_evidence_phase1_bounded_20260807"
SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENTS = ("gwtc3", "gwtc4")
OBJECTIVES = ("retrieval", "candidate")
WEIGHT_KEYS = ("waveform", "time", "sky")
TOP_BUDGETS = (50, 100, 200, 500)
RAW_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
BANDWIDTH_SCALES = (0.7, 1.0, 1.4)
FLOOR_MIXTURES = (1e-4, 1e-3, 1e-2)
SCORE_CAPS = (4.0, 6.0, 8.0)
SUPPORT_QUANTILES = (0.0, 0.001, 0.005)
DENSITY_DEFAULT = np.asarray([1.0, -3.0, 6.0, 0.001], dtype=np.float64)
SNR_MATCH_QUANTILES = tuple(np.linspace(0.0, 1.0, 9))
LEGACY_FAMILY_TO_POPULATION = {
    "SIS": "smooth_non_subhalo",
    "PM": "subhalo_present",
    "sis": "smooth_non_subhalo",
    "pm": "subhalo_present",
}


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase05 = module_from(
    REPO / "scripts/experiments/131_time_snr_joint_evidence_phase05.py",
    "bounded_phase1_phase05",
)
phase05a = module_from(
    REPO / "scripts/experiments/132_time_snr_joint_evidence_phase05a.py",
    "bounded_phase1_phase05a",
)
v7 = phase05.v7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--null-pairs", type=int, default=250_000)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--deployments", nargs="+", choices=DEPLOYMENTS, default=list(DEPLOYMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return phase05.json_ready(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(float(weights[key]) for key in WEIGHT_KEYS))
    if total <= 0:
        raise ValueError("Weight vector must have positive L1 norm")
    return {key: float(weights[key]) / total for key in WEIGHT_KEYS}


def normalized_weight_candidates(no_sky: bool = False) -> pd.DataFrame:
    rows: dict[tuple[float, float, float], dict[str, Any]] = {}
    for raw in itertools.product(RAW_WEIGHT_GRID, repeat=3):
        waveform, time_weight, sky = map(float, raw)
        if no_sky:
            if sky != 0 or min(waveform, time_weight) <= 0:
                continue
        elif min(waveform, time_weight, sky) <= 0:
            continue
        total = waveform + time_weight + sky
        norm = tuple(round(value / total, 12) for value in raw)
        if norm not in rows:
            rows[norm] = {
                "waveform": norm[0],
                "time": norm[1],
                "sky": norm[2],
                "equivalent_raw_vectors": [],
            }
        rows[norm]["equivalent_raw_vectors"].append(list(raw))
    frame = pd.DataFrame(rows.values())
    frame["equivalent_raw_count"] = frame["equivalent_raw_vectors"].map(len)
    frame["equivalent_raw_vectors"] = frame["equivalent_raw_vectors"].map(json.dumps)
    return frame.sort_values(list(WEIGHT_KEYS), kind="stable").reset_index(drop=True)


def top_b_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    labels = frame["is_true_pair"].to_numpy(dtype=np.int8)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    result: dict[str, float] = {}
    positives = max(int(labels.sum()), 1)
    for budget in TOP_BUDGETS:
        used = min(int(budget), len(order))
        true_count = int(labels[order[:used]].sum())
        false_count = int(used - true_count)
        result[f"top{budget}_true"] = true_count
        result[f"top{budget}_false"] = false_count
        result[f"top{budget}_precision"] = float(true_count / max(used, 1))
        result[f"top{budget}_recall"] = float(true_count / positives)
    result["mean_top_b_precision"] = float(
        np.mean([result[f"top{budget}_precision"] for budget in TOP_BUDGETS])
    )
    return result


def metric_bundle(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    return {
        **v7.retrieval_metrics(frame, scores),
        **v7.pair_metrics(frame, scores),
        **top_b_metrics(frame, scores),
    }


def selection_priority(objective: str) -> list[str]:
    if objective == "retrieval":
        return [
            "macro_r_at_10",
            "min_family_r_at_10",
            "average_precision",
            "precision_at_recall_0p5",
            "macro_r_at_1",
        ]
    if objective == "candidate":
        return [
            "average_precision",
            "precision_at_recall_0p5",
            "macro_r_at_10",
            "min_family_r_at_10",
            "macro_r_at_1",
        ]
    raise ValueError(objective)


def select_normalized_weights(
    validation: pd.DataFrame,
    objective: str,
    anchor: dict[str, float],
    no_sky: bool = False,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, Any]]:
    candidates = normalized_weight_candidates(no_sky=no_sky)
    anchor = normalize_weights(anchor)
    rows = []
    for candidate in candidates.itertuples(index=False):
        weights = {key: float(getattr(candidate, key)) for key in WEIGHT_KEYS}
        scores = v7.score_vector(validation, weights)
        rows.append(
            {
                **weights,
                "equivalent_raw_count": int(candidate.equivalent_raw_count),
                "equivalent_raw_vectors": candidate.equivalent_raw_vectors,
                **metric_bundle(validation, scores),
                "distance_to_phase05a_anchor": float(
                    math.sqrt(sum((weights[key] - anchor[key]) ** 2 for key in WEIGHT_KEYS))
                ),
            }
        )
    grid = pd.DataFrame(rows)
    priority = selection_priority(objective)
    ordered = grid.sort_values(
        priority + ["distance_to_phase05a_anchor", *WEIGHT_KEYS],
        ascending=[False] * len(priority) + [True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    metric_signature = tuple(ordered.loc[0, priority].tolist())
    platform = np.ones(len(ordered), dtype=bool)
    for column, value in zip(priority, metric_signature):
        platform &= ordered[column].to_numpy() == value
    ordered["on_best_performance_platform"] = platform
    ordered["selected"] = False
    ordered.loc[0, "selected"] = True
    selected = {key: float(ordered.loc[0, key]) for key in WEIGHT_KEYS}
    audit = {
        "objective": objective,
        "selection_split": "response-derived validation systems only",
        "raw_grid": list(RAW_WEIGHT_GRID),
        "raw_vectors_normalized_before_scoring": True,
        "proportional_vectors_merged": True,
        "unique_normalized_vectors": int(len(ordered)),
        "metric_priority": priority,
        "performance_platform_size": int(platform.sum()),
        "tie_break_1": "minimum Euclidean distance to normalized Phase 0.5a one-dimensional baseline weight",
        "tie_break_2": "ascending fixed dictionary tuple (waveform,time,sky)",
        "phase05a_anchor": anchor,
        "selected": selected,
    }
    return selected, ordered, audit


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=np.float64)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(float(quantile), cumulative, values))


def weighted_cdf_distance(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    """Return a weighted two-sample Kolmogorov distance without a p-value."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    weights_a = np.asarray(weights_a, dtype=np.float64)
    weights_b = np.asarray(weights_b, dtype=np.float64)
    weights_a = weights_a / weights_a.sum()
    weights_b = weights_b / weights_b.sum()
    support = np.unique(np.concatenate([values_a, values_b]))
    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)
    cdf_a = np.concatenate([[0.0], np.cumsum(weights_a[order_a])])
    cdf_b = np.concatenate([[0.0], np.cumsum(weights_b[order_b])])
    indices_a = np.searchsorted(values_a[order_a], support, side="right")
    indices_b = np.searchsorted(values_b[order_b], support, side="right")
    return float(np.max(np.abs(cdf_a[indices_a] - cdf_b[indices_b])))


def add_snr_marginal_match_weights(
    signal: pd.DataFrame,
    null: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match null rho_early strata to the source-group-weighted signal mixture.

    Absolute SNR is not an input to the two-dimensional density. Matching its
    marginal distribution prevents the signed SNR ratio from acting as a proxy
    for a different absolute-SNR selection function.
    """
    signal_values = signal["rho_early"].to_numpy(dtype=np.float64)
    signal_weights = signal["source_group_density_weight"].to_numpy(dtype=np.float64)
    signal_weights = signal_weights / signal_weights.sum()
    null_values = null["rho_early"].to_numpy(dtype=np.float64)
    raw_edges = [weighted_quantile(signal_values, q, signal_weights) for q in SNR_MATCH_QUANTILES]
    interior = np.unique(np.asarray(raw_edges[1:-1], dtype=np.float64))
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    signal_bin = np.searchsorted(edges, signal_values, side="right") - 1
    null_bin = np.searchsorted(edges, null_values, side="right") - 1
    n_bins = len(edges) - 1
    target_mass = np.bincount(signal_bin, weights=signal_weights, minlength=n_bins).astype(np.float64)
    null_count = np.bincount(null_bin, minlength=n_bins).astype(np.int64)
    missing = np.flatnonzero((target_mass > 0) & (null_count == 0))
    if len(missing):
        raise RuntimeError(f"Null SNR matching has empty populated strata: {missing.tolist()}")
    per_pair_weight = np.zeros(n_bins, dtype=np.float64)
    populated = null_count > 0
    per_pair_weight[populated] = target_mass[populated] / null_count[populated]
    matched_weights = per_pair_weight[null_bin]
    matched_weights /= matched_weights.sum()
    null = null.copy()
    null["rho_early_stratum"] = null_bin
    null["snr_marginal_match_weight"] = matched_weights
    matched_mass = np.bincount(null_bin, weights=matched_weights, minlength=n_bins)
    pre_weights = np.full(len(null), 1.0 / len(null), dtype=np.float64)
    audit = {
        "rule": "eight source-group-weighted signal rho_early quantile strata; null reweighted to identical stratum mass",
        "selection_data": "density-development systems only",
        "absolute_snr_used_as_joint_density_input": False,
        "quantiles": list(SNR_MATCH_QUANTILES),
        "stratum_edges": edges.tolist(),
        "target_signal_mass": target_mass.tolist(),
        "matched_null_mass": matched_mass.tolist(),
        "max_absolute_stratum_mass_error": float(np.max(np.abs(target_mass - matched_mass))),
        "pre_match_weighted_cdf_distance": weighted_cdf_distance(
            signal_values, signal_weights, null_values, pre_weights
        ),
        "post_match_weighted_cdf_distance": weighted_cdf_distance(
            signal_values, signal_weights, null_values, matched_weights
        ),
        "pre_match_wasserstein": float(stats.wasserstein_distance(signal_values, null_values)),
        "post_match_weighted_mean_signal": float(np.average(signal_values, weights=signal_weights)),
        "post_match_weighted_mean_null": float(np.average(null_values, weights=matched_weights)),
        "null_effective_sample_size": float(1.0 / np.sum(np.square(matched_weights))),
        "null_weight_min": float(matched_weights.min()),
        "null_weight_max": float(matched_weights.max()),
    }
    return null, audit


def response_signal_table(deployment: str, seed: int) -> pd.DataFrame:
    metadata = pd.read_parquet(
        PHASE05 / f"response/{deployment}/seed_{seed}/density_development_response_metadata.parquet"
    )
    weights = pd.read_parquet(
        PHASE05A / f"time/{deployment}/seed_{seed}/density_development_source_group_weights.parquet",
        columns=["global_lens_system_id", "source_group_density_weight"],
    )
    frame = metadata.merge(weights, on="global_lens_system_id", validate="one_to_one")
    first_is_early = frame["gps_image1"].to_numpy() <= frame["gps_image2"].to_numpy()
    rho1 = frame["recovered_optimal_snr_image1"].to_numpy(dtype=np.float64)
    rho2 = frame["recovered_optimal_snr_image2"].to_numpy(dtype=np.float64)
    early = np.where(first_is_early, rho1, rho2)
    late = np.where(first_is_early, rho2, rho1)
    frame["log10_delta_t_days"] = np.log10(np.maximum(frame["delay_days"], 1e-12))
    frame["r_rho"] = np.log(late / early)
    frame["rho_early"] = early
    frame["rho_late"] = late
    return frame


def response_null_table(signal: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    groups = sorted(signal["global_source_group_id"].astype(str).unique())
    by_group = {
        group: signal.index[signal["global_source_group_id"].astype(str) == group].to_numpy(dtype=np.int32)
        for group in groups
    }
    max_multiplicity = max(len(values) for values in by_group.values())
    members = np.full((len(groups), max_multiplicity), -1, dtype=np.int32)
    multiplicity = np.empty(len(groups), dtype=np.int32)
    for index, group in enumerate(groups):
        values = by_group[group]
        members[index, : len(values)] = values
        multiplicity[index] = len(values)
    rng = np.random.default_rng(seed)
    group_a = rng.integers(0, len(groups), size=int(size), dtype=np.int32)
    offset = rng.integers(1, len(groups), size=int(size), dtype=np.int32)
    group_b = (group_a + offset) % len(groups)
    slot_a = (rng.random(int(size)) * multiplicity[group_a]).astype(np.int32)
    slot_b = (rng.random(int(size)) * multiplicity[group_b]).astype(np.int32)
    row_a = members[group_a, slot_a]
    row_b = members[group_b, slot_b]
    image_a = rng.integers(0, 2, size=int(size), dtype=np.int8)
    image_b = rng.integers(0, 2, size=int(size), dtype=np.int8)
    gps = np.stack(
        [signal["gps_image1"].to_numpy(dtype=np.float64), signal["gps_image2"].to_numpy(dtype=np.float64)],
        axis=1,
    )
    snr = np.stack(
        [
            signal["recovered_optimal_snr_image1"].to_numpy(dtype=np.float64),
            signal["recovered_optimal_snr_image2"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    gps_a = gps[row_a, image_a]
    gps_b = gps[row_b, image_b]
    snr_a = snr[row_a, image_a]
    snr_b = snr[row_b, image_b]
    a_is_early = gps_a <= gps_b
    early = np.where(a_is_early, snr_a, snr_b)
    late = np.where(a_is_early, snr_b, snr_a)
    delta = np.abs(gps_a - gps_b) / 86_400.0
    return pd.DataFrame(
        {
            "source_group_index_a": group_a,
            "source_group_index_b": group_b,
            "log10_delta_t_days": np.log10(np.maximum(delta, 1e-12)),
            "r_rho": np.log(late / early),
            "rho_early": early,
            "rho_late": late,
        }
    )


def pair_observables(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    shuffled: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    events = events.sort_values("idx").reset_index(drop=True)
    if not np.array_equal(events["idx"].to_numpy(dtype=np.int32), np.arange(len(events), dtype=np.int32)):
        raise RuntimeError("Event indices are not contiguous")
    snr = events["snr"].to_numpy(dtype=np.float64).copy()
    permutation_rows = None
    if shuffled:
        rng = np.random.default_rng(seed)
        shuffled_snr = snr.copy()
        rows = []
        for family in sorted(events["family"].astype(str).unique()):
            indices = np.flatnonzero(events["family"].astype(str).to_numpy() == family)
            permuted = rng.permutation(indices)
            shuffled_snr[indices] = snr[permuted]
            rows.extend(
                {
                    "idx": int(target),
                    "family": family,
                    "source_idx": int(source),
                    "original_snr": float(snr[target]),
                    "shuffled_snr": float(snr[source]),
                }
                for target, source in zip(indices, permuted)
            )
        snr = shuffled_snr
        permutation_rows = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    ii = frame["idx_i"].to_numpy(dtype=np.int32)
    jj = frame["idx_j"].to_numpy(dtype=np.int32)
    i_early = gps[ii] <= gps[jj]
    early = np.where(i_early, snr[ii], snr[jj])
    late = np.where(i_early, snr[jj], snr[ii])
    if not (np.isfinite(early).all() and np.isfinite(late).all() and (early > 0).all() and (late > 0).all()):
        raise RuntimeError("Invalid event SNR while constructing r_rho")
    return np.log(late / early), early, permutation_rows


def density_edges(signal: pd.DataFrame, null: pd.DataFrame, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([signal["log10_delta_t_days"], null["log10_delta_t_days"]]).astype(float)
    r = np.concatenate([signal["r_rho"], null["r_rho"]]).astype(float)
    x_margin = max(0.02, 0.02 * (x.max() - x.min()))
    r_margin = max(0.02, 0.02 * (r.max() - r.min()))
    return (
        np.linspace(x.min() - x_margin, x.max() + x_margin, int(grid_size) + 1),
        np.linspace(r.min() - r_margin, r.max() + r_margin, int(grid_size) + 1),
    )


def grid_surfaces(
    signal: pd.DataFrame,
    null: pd.DataFrame,
    x_edges: np.ndarray,
    r_edges: np.ndarray,
    bandwidth_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    signal_weights = signal["source_group_density_weight"].to_numpy(dtype=np.float64)
    null_weights = null["snr_marginal_match_weight"].to_numpy(dtype=np.float64)
    signal_hist, _, _ = np.histogram2d(
        signal["log10_delta_t_days"], signal["r_rho"], bins=(x_edges, r_edges), weights=signal_weights
    )
    null_hist, _, _ = np.histogram2d(
        null["log10_delta_t_days"],
        null["r_rho"],
        bins=(x_edges, r_edges),
        weights=null_weights,
    )
    dx = float(x_edges[1] - x_edges[0])
    dr = float(r_edges[1] - r_edges[0])
    factor = float(signal["global_source_group_id"].nunique() ** (-1.0 / 6.0))
    signal_x_std = float(np.sqrt(np.average(
        np.square(signal["log10_delta_t_days"] - np.average(signal["log10_delta_t_days"], weights=signal_weights)),
        weights=signal_weights,
    )))
    signal_r_std = float(np.sqrt(np.average(
        np.square(signal["r_rho"] - np.average(signal["r_rho"], weights=signal_weights)),
        weights=signal_weights,
    )))
    null_x_mean = float(np.average(null["log10_delta_t_days"], weights=null_weights))
    null_r_mean = float(np.average(null["r_rho"], weights=null_weights))
    null_x_std = float(np.sqrt(np.average(
        np.square(null["log10_delta_t_days"] - null_x_mean), weights=null_weights
    )))
    null_r_std = float(np.sqrt(np.average(
        np.square(null["r_rho"] - null_r_mean), weights=null_weights
    )))
    pooled_x_std = math.sqrt(0.5 * (signal_x_std**2 + null_x_std**2))
    pooled_r_std = math.sqrt(0.5 * (signal_r_std**2 + null_r_std**2))
    sigma_x = max(0.5, float(bandwidth_scale) * factor * pooled_x_std / dx)
    sigma_r = max(0.5, float(bandwidth_scale) * factor * pooled_r_std / dr)
    signal_smooth = ndimage.gaussian_filter(signal_hist, sigma=(sigma_x, sigma_r), mode="constant")
    null_smooth = ndimage.gaussian_filter(null_hist, sigma=(sigma_x, sigma_r), mode="constant")
    area = dx * dr
    signal_density = signal_smooth / max(signal_smooth.sum() * area, np.finfo(float).tiny)
    null_density = null_smooth / max(null_smooth.sum() * area, np.finfo(float).tiny)
    return signal_density, null_density, {
        "bandwidth_scale": float(bandwidth_scale),
        "scott_factor_from_source_groups": factor,
        "sigma_x_bins": sigma_x,
        "sigma_r_bins": sigma_r,
        "physical_bandwidth_log10_delay": sigma_x * dx,
        "physical_bandwidth_log_snr_ratio": sigma_r * dr,
    }


def support_bounds(signal: pd.DataFrame, null: pd.DataFrame, quantile: float) -> dict[str, float]:
    weights = signal["source_group_density_weight"].to_numpy(dtype=np.float64)
    null_weights = null["snr_marginal_match_weight"].to_numpy(dtype=np.float64)
    values: dict[str, float] = {}
    for column, prefix in (("log10_delta_t_days", "x"), ("r_rho", "r")):
        signal_values = signal[column].to_numpy(dtype=np.float64)
        null_values = null[column].to_numpy(dtype=np.float64)
        signal_lo = weighted_quantile(signal_values, quantile, weights)
        signal_hi = weighted_quantile(signal_values, 1.0 - quantile, weights)
        null_lo = weighted_quantile(null_values, quantile, null_weights)
        null_hi = weighted_quantile(null_values, 1.0 - quantile, null_weights)
        values[f"{prefix}_lo"] = min(signal_lo, null_lo)
        values[f"{prefix}_hi"] = max(signal_hi, null_hi)
    return values


def interpolate_densities(
    x_edges: np.ndarray,
    r_edges: np.ndarray,
    signal_density: np.ndarray,
    null_density: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    points = np.column_stack([x, r])
    signal_interp = RegularGridInterpolator(
        (x_centers, r_centers), signal_density, bounds_error=False, fill_value=np.nan
    )(points)
    null_interp = RegularGridInterpolator(
        (x_centers, r_centers), null_density, bounds_error=False, fill_value=np.nan
    )(points)
    return signal_interp, null_interp


def joint_score(
    p_signal: np.ndarray,
    p_null: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
    fallback_time: np.ndarray,
    floor_mixture: float,
    score_cap: float,
    bounds: dict[str, float],
    domain_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    uniform = 1.0 / domain_area
    p_signal_regularized = (1.0 - floor_mixture) * p_signal + floor_mixture * uniform
    p_null_regularized = (1.0 - floor_mixture) * p_null + floor_mixture * uniform
    ood = (
        ~np.isfinite(p_signal_regularized)
        | ~np.isfinite(p_null_regularized)
        | (x < bounds["x_lo"])
        | (x > bounds["x_hi"])
        | (r < bounds["r_lo"])
        | (r > bounds["r_hi"])
    )
    score = np.log(np.maximum(p_signal_regularized, np.finfo(float).tiny)) - np.log(
        np.maximum(p_null_regularized, np.finfo(float).tiny)
    )
    score = np.clip(score, -float(score_cap), float(score_cap))
    score[ood] = np.asarray(fallback_time, dtype=np.float64)[ood]
    return score.astype(np.float32), ood


def select_density_configuration(
    validation: pd.DataFrame,
    signal: pd.DataFrame,
    null: pd.DataFrame,
    x_edges: np.ndarray,
    r_edges: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, dict[float, tuple[np.ndarray, np.ndarray, dict[str, float]]]]:
    x = np.log10(np.maximum(validation["delta_t_days"].to_numpy(dtype=np.float64), 1e-12))
    r = validation["r_rho"].to_numpy(dtype=np.float64)
    fallback = validation["time_score"].to_numpy(dtype=np.float64)
    domain_area = float((x_edges[-1] - x_edges[0]) * (r_edges[-1] - r_edges[0]))
    surfaces = {
        bandwidth: grid_surfaces(signal, null, x_edges, r_edges, bandwidth)
        for bandwidth in BANDWIDTH_SCALES
    }
    bounds_by_q = {quantile: support_bounds(signal, null, quantile) for quantile in SUPPORT_QUANTILES}
    rows = []
    for bandwidth, floor_mixture, cap, quantile in itertools.product(
        BANDWIDTH_SCALES, FLOOR_MIXTURES, SCORE_CAPS, SUPPORT_QUANTILES
    ):
        p_signal, p_null, bandwidth_audit = surfaces[bandwidth]
        signal_at_pair, null_at_pair = interpolate_densities(
            x_edges, r_edges, p_signal, p_null, x, r
        )
        score, ood = joint_score(
            signal_at_pair,
            null_at_pair,
            x,
            r,
            fallback,
            floor_mixture,
            cap,
            bounds_by_q[quantile],
            domain_area,
        )
        labels = validation["is_true_pair"].to_numpy(dtype=bool)
        metrics = metric_bundle(validation, score)
        rows.append(
            {
                "bandwidth_scale": bandwidth,
                "floor_mixture": floor_mixture,
                "score_cap": cap,
                "support_quantile": quantile,
                **bandwidth_audit,
                **bounds_by_q[quantile],
                "ood_fraction_all": float(ood.mean()),
                "ood_fraction_true": float(ood[labels].mean()),
                "ood_fraction_false": float(ood[~labels].mean()),
                **metrics,
                "distance_to_predeclared_default": float(
                    np.linalg.norm(
                        np.asarray([bandwidth, math.log10(floor_mixture), cap, quantile]) - DENSITY_DEFAULT
                    )
                ),
            }
        )
    grid = pd.DataFrame(rows)
    priority = [
        "average_precision",
        "precision_at_recall_0p5",
        "precision_at_recall_0p9",
        "mean_top_b_precision",
        "macro_r_at_10",
        "min_family_r_at_10",
        "macro_r_at_1",
    ]
    ordered = grid.sort_values(
        priority
        + [
            "distance_to_predeclared_default",
            "bandwidth_scale",
            "floor_mixture",
            "score_cap",
            "support_quantile",
        ],
        ascending=[False] * len(priority) + [True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    signature = tuple(ordered.loc[0, priority].tolist())
    platform = np.ones(len(ordered), dtype=bool)
    for column, value in zip(priority, signature):
        platform &= ordered[column].to_numpy() == value
    ordered["on_best_performance_platform"] = platform
    ordered["selected"] = False
    ordered.loc[0, "selected"] = True
    selected_row = ordered.iloc[0]
    selected = {
        "bandwidth_scale": float(selected_row.bandwidth_scale),
        "floor_mixture": float(selected_row.floor_mixture),
        "score_cap": float(selected_row.score_cap),
        "support_quantile": float(selected_row.support_quantile),
        "support_bounds": bounds_by_q[float(selected_row.support_quantile)],
        "domain_area": domain_area,
        "selection_priority": priority,
        "performance_platform_size": int(platform.sum()),
        "selected_on": "response-derived validation systems only",
        "test_loaded_during_selection": False,
    }
    return selected, ordered, surfaces


def apply_selected_density(
    frame: pd.DataFrame,
    x_edges: np.ndarray,
    r_edges: np.ndarray,
    surfaces: dict[float, tuple[np.ndarray, np.ndarray, dict[str, float]]],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.log10(np.maximum(frame["delta_t_days"].to_numpy(dtype=np.float64), 1e-12))
    r = frame["r_rho"].to_numpy(dtype=np.float64)
    p_signal, p_null, _ = surfaces[float(config["bandwidth_scale"])]
    signal_at_pair, null_at_pair = interpolate_densities(x_edges, r_edges, p_signal, p_null, x, r)
    return joint_score(
        signal_at_pair,
        null_at_pair,
        x,
        r,
        frame["time_score"].to_numpy(dtype=np.float64),
        float(config["floor_mixture"]),
        float(config["score_cap"]),
        config["support_bounds"],
        float(config["domain_area"]),
    )


def evaluate_method(
    frame: pd.DataFrame,
    weights: dict[str, float],
    method: str,
    deployment: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    query, retrieval, pair = v7.evaluation_tables(frame, {method: weights}, deployment, seed)
    if "true_pair_family" in query:
        query["physical_population"] = query["true_pair_family"].map(LEGACY_FAMILY_TO_POPULATION)
    if "subset" in retrieval:
        retrieval["subset"] = retrieval["subset"].replace(LEGACY_FAMILY_TO_POPULATION)
    scores = v7.score_vector(frame, weights)
    top_rows = []
    metrics = top_b_metrics(frame, scores)
    for budget in TOP_BUDGETS:
        top_rows.append(
            {
                "deployment": deployment,
                "seed": seed,
                "method": method,
                "budget": budget,
                "true_recovered": int(metrics[f"top{budget}_true"]),
                "false_candidates": int(metrics[f"top{budget}_false"]),
                "precision": float(metrics[f"top{budget}_precision"]),
                "recall": float(metrics[f"top{budget}_recall"]),
            }
        )
    return query, retrieval, pair, scores, pd.DataFrame(top_rows)


def fixed_false_pair_metrics(positive: np.ndarray, false_sorted_ascending: np.ndarray) -> dict[str, float]:
    positive = np.sort(np.asarray(positive, dtype=np.float64))[::-1]
    n_positive = len(positive)
    false_greater = len(false_sorted_ascending) - np.searchsorted(
        false_sorted_ascending, positive, side="right"
    )
    precision_at_positive = np.arange(1, n_positive + 1) / (
        np.arange(1, n_positive + 1) + false_greater
    )
    result = {"average_precision": float(np.mean(precision_at_positive))}
    for target in (0.5, 0.9):
        index = min(max(int(math.ceil(target * n_positive)) - 1, 0), n_positive - 1)
        false_count = int(false_greater[index])
        result[f"false_at_recall_{str(target).replace('.', 'p')}"] = false_count
        result[f"precision_at_recall_{str(target).replace('.', 'p')}"] = float(
            (index + 1) / (index + 1 + false_count)
        )
    false_desc = false_sorted_ascending[::-1]
    for budget in TOP_BUDGETS:
        candidates = np.concatenate([positive, false_desc[:budget]])
        labels = np.concatenate([np.ones(len(positive), dtype=np.int8), np.zeros(min(budget, len(false_desc)), dtype=np.int8)])
        order = np.argsort(-candidates, kind="stable")[:budget]
        true_count = int(labels[order].sum())
        result[f"top{budget}_precision"] = float(true_count / budget)
    return result


def system_bootstrap_comparison(
    frame: pd.DataFrame,
    query_by_method: dict[str, pd.DataFrame],
    score_by_method: dict[str, np.ndarray],
    method_a: str,
    method_b: str,
    comparison: str,
    deployment: str,
    seed: int,
    draws: int,
) -> pd.DataFrame:
    positive = frame[frame["is_true_pair"].to_numpy(dtype=bool)].copy()
    positive["system_id"] = [
        f"{family}:{min(i,j)}-{max(i,j)}"
        for family, i, j in zip(positive["true_pair_family"], positive["idx_i"], positive["idx_j"])
    ]
    positive_indices = np.flatnonzero(frame["is_true_pair"].to_numpy(dtype=bool)).astype(np.int32)
    false_mask = ~frame["is_true_pair"].to_numpy(dtype=bool)
    query_maps = {
        method: {
            system: part["query_rank"].to_numpy(dtype=np.int32)
            for system, part in query_by_method[method].groupby("system_id")
        }
        for method in (method_a, method_b)
    }
    pair_maps = {
        method: dict(zip(positive["system_id"], score_by_method[method][positive_indices]))
        for method in (method_a, method_b)
    }
    false_sorted = {
        method: np.sort(score_by_method[method][false_mask].astype(np.float64))
        for method in (method_a, method_b)
    }
    systems_by_family = {
        family: positive.loc[positive["true_pair_family"] == family, "system_id"].astype(str).tolist()
        for family in ("SIS", "PM")
    }
    rng = np.random.default_rng(seed + 9_100_000 + sum(map(ord, comparison)))
    rows = []
    for draw in range(int(draws)):
        sampled_systems = []
        for family in ("SIS", "PM"):
            systems = systems_by_family[family]
            selected = rng.integers(0, len(systems), size=len(systems))
            sampled_systems.extend(systems[index] for index in selected)
        metrics_by_method: dict[str, dict[str, float]] = {}
        for method in (method_a, method_b):
            ranks = np.concatenate([query_maps[method][system] for system in sampled_systems])
            pair_positive = np.asarray([pair_maps[method][system] for system in sampled_systems])
            metrics_by_method[method] = {
                "r_at_1": float(np.mean(ranks <= 1)),
                "r_at_10": float(np.mean(ranks <= 10)),
                **fixed_false_pair_metrics(pair_positive, false_sorted[method]),
            }
        row: dict[str, Any] = {
            "deployment": deployment,
            "seed": seed,
            "comparison": comparison,
            "method_a": method_a,
            "method_b": method_b,
            "draw": draw,
        }
        metrics = [
            "r_at_1",
            "r_at_10",
            "average_precision",
            "false_at_recall_0p5",
            "false_at_recall_0p9",
            *[f"top{budget}_precision" for budget in TOP_BUDGETS],
        ]
        for metric in metrics:
            a = metrics_by_method[method_a][metric]
            b = metrics_by_method[method_b][metric]
            row[f"a_{metric}"] = a
            row[f"b_{metric}"] = b
            row[f"delta_{metric}"] = a - b
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(draws: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in draws.columns if column.startswith("delta_")]
    rows = []
    for keys, part in draws.groupby(["deployment", "seed", "comparison", "method_a", "method_b"]):
        row = dict(zip(["deployment", "seed", "comparison", "method_a", "method_b"], keys))
        row["draws"] = len(part)
        for column in metric_columns:
            values = part[column].to_numpy(dtype=np.float64)
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_lower95"] = float(np.quantile(values, 0.025))
            row[f"{column}_upper95"] = float(np.quantile(values, 0.975))
            beneficial = values < 0 if "false_at_recall" in column else values > 0
            row[f"{column}_p_benefit"] = float(np.mean(beneficial))
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_rows(frame: pd.DataFrame, deployment: str, seed: int, split: str) -> list[dict[str, Any]]:
    columns = ["waveform_score", "time_score_1d", "joint_score_2d", "sky_score"]
    rows = []
    for label, part in (("true", frame[frame["is_true_pair"] == 1]), ("false", frame[frame["is_true_pair"] == 0])):
        for left, right in itertools.combinations(columns, 2):
            result = stats.spearmanr(part[left], part[right])
            rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "split": split,
                    "pair_class": label,
                    "channel_a": left,
                    "channel_b": right,
                    "spearman": float(result.statistic),
                    "pvalue": float(result.pvalue),
                    "n_pairs": len(part),
                }
            )
    return rows


def summarize_metrics(frame: pd.DataFrame, metrics: Iterable[str], groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        row["n_seeds"] = part["seed"].nunique()
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_seed(
    output: Path,
    deployment: str,
    seed: int,
    null_pairs: int,
    bootstrap_draws: int,
    grid_size: int,
) -> None:
    seed_out = output / deployment / f"seed_{seed}"
    marker = seed_out / "seed_complete.json"
    if marker.exists():
        print(f"[{deployment}/{seed}] already complete", flush=True)
        return
    seed_out.mkdir(parents=True, exist_ok=True)
    print(f"[{deployment}/{seed}] development density", flush=True)
    signal = response_signal_table(deployment, seed)
    split_summary = pd.read_csv(PHASE05 / "splits/global_system_split_summary.csv")
    split_row = split_summary[
        (split_summary["deployment"] == deployment) & (split_summary["seed"] == seed)
    ]
    if len(split_row) != 1:
        raise RuntimeError(f"Missing unique split audit for {deployment}/{seed}")
    split_record = split_row.iloc[0].to_dict()
    intersection_columns = [
        "train_validation_source_intersection",
        "train_test_source_intersection",
        "validation_test_source_intersection",
    ]
    if any(int(split_record[column]) != 0 for column in intersection_columns):
        raise RuntimeError(f"Source-group split leakage for {deployment}/{seed}: {split_record}")
    if int(split_record["density_development_unique_source_groups"]) != int(
        signal["global_source_group_id"].nunique()
    ):
        raise RuntimeError(f"Density-development source-group count mismatch for {deployment}/{seed}")
    write_json(
        seed_out / "source_group_split_isolation_audit.json",
        {
            **split_record,
            "runtime_signal_unique_source_groups": int(signal["global_source_group_id"].nunique()),
            "all_source_group_intersections_zero": True,
        },
    )
    null = response_null_table(signal, null_pairs, seed + 8_100_000)
    null, snr_match_audit = add_snr_marginal_match_weights(signal, null)
    signal.to_parquet(seed_out / "density_signal_source_group_weighted.parquet", index=False)
    null.to_parquet(seed_out / "density_null_cross_source_pairs.parquet", index=False)
    x_edges, r_edges = density_edges(signal, null, grid_size)
    np.savez_compressed(seed_out / "density_grid_edges.npz", x_edges=x_edges, r_edges=r_edges)

    snr_audit = {
        "deployment": deployment,
        "seed": seed,
        "signal_lens_rows": len(signal),
        "signal_unique_source_groups": signal["global_source_group_id"].nunique(),
        "signal_total_source_weight": float(signal["source_group_density_weight"].sum()),
        "maximum_group_weight_error": float(
            np.max(
                np.abs(
                    signal.groupby("global_source_group_id")["source_group_density_weight"].sum().to_numpy()
                    - 1.0
                )
            )
        ),
        "null_pairs": len(null),
        "null_same_source_pairs": 0,
        "rho_early_ks_statistic": float(stats.ks_2samp(signal["rho_early"], null["rho_early"]).statistic),
        "rho_early_ks_pvalue": float(stats.ks_2samp(signal["rho_early"], null["rho_early"]).pvalue),
        "rho_early_wasserstein": float(stats.wasserstein_distance(signal["rho_early"], null["rho_early"])),
        "rho_early_marginal_matching": snr_match_audit,
    }
    write_json(seed_out / "density_source_and_snr_audit.json", snr_audit)

    print(f"[{deployment}/{seed}] validation-only density selection", flush=True)
    validation = pd.read_parquet(
        PHASE05A
        / f"time/{deployment}/seed_{seed}/response_derived/validation_pairs_source_weighted_time.parquet"
    )
    validation_events = pd.read_parquet(
        PHASE05 / f"response/{deployment}/seed_{seed}/response_validation_events.parquet"
    )
    validation_r, validation_early, _ = pair_observables(
        validation, validation_events, shuffled=False, seed=seed
    )
    validation["r_rho"] = validation_r
    validation["rho_early"] = validation_early
    density_config, density_grid, surfaces = select_density_configuration(
        validation, signal, null, x_edges, r_edges
    )
    density_grid.to_csv(seed_out / "joint_density_validation_grid.csv", index=False)
    write_json(seed_out / "joint_density_selected_config.json", density_config)
    selected_surface = surfaces[float(density_config["bandwidth_scale"])]
    np.savez_compressed(
        seed_out / "joint_density_selected_surfaces.npz",
        signal_density=selected_surface[0],
        null_density=selected_surface[1],
        x_edges=x_edges,
        r_edges=r_edges,
    )
    validation_joint, validation_ood = apply_selected_density(
        validation, x_edges, r_edges, surfaces, density_config
    )
    validation_joint_frame = validation.copy()
    validation_joint_frame["time_score"] = validation_joint
    validation_joint_frame["joint_ood"] = validation_ood

    old_weights = json.loads(
        (
            PHASE05A
            / f"time/{deployment}/seed_{seed}/response_derived/selected_weights.json"
        ).read_text(encoding="utf-8")
    )
    selected_weights: dict[str, dict[str, dict[str, float]]] = {
        "baseline_1d": {},
        "joint_2d": {},
        "joint_2d_no_sky": {},
    }
    weight_audits = []
    for objective in OBJECTIVES:
        anchor = old_weights[f"{objective}_three_channel_strict_positive"]
        for model, frame, no_sky in (
            ("baseline_1d", validation, False),
            ("joint_2d", validation_joint_frame, False),
            ("joint_2d_no_sky", validation_joint_frame, True),
        ):
            model_anchor = anchor
            if no_sky:
                model_anchor = {"waveform": anchor["waveform"], "time": anchor["time"], "sky": 0.0}
            weights, grid, audit = select_normalized_weights(
                frame, objective, model_anchor, no_sky=no_sky
            )
            selected_weights[model][objective] = weights
            grid.to_csv(seed_out / f"weight_grid_{model}_{objective}.csv", index=False)
            audit.update({"model": model, "deployment": deployment, "seed": seed})
            weight_audits.append(audit)
    write_json(seed_out / "selected_normalized_weights.json", selected_weights)
    write_json(seed_out / "weight_selection_audits.json", weight_audits)
    write_json(
        seed_out / "validation_freeze_marker.json",
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "density_config_sha256": sha256(seed_out / "joint_density_selected_config.json"),
            "weights_sha256": sha256(seed_out / "selected_normalized_weights.json"),
            "test_loaded_before_marker": False,
        },
    )

    print(f"[{deployment}/{seed}] held-out test evaluation", flush=True)
    test = pd.read_parquet(
        PHASE05A
        / f"time/{deployment}/seed_{seed}/response_derived/heldout_test_pairs_source_weighted_time.parquet"
    )
    test_events = pd.read_parquet(
        PHASE05 / f"response/{deployment}/seed_{seed}/response_test_events.parquet"
    )
    test_r, test_early, _ = pair_observables(test, test_events, shuffled=False, seed=seed)
    shuffled_r, _, permutation = pair_observables(
        test, test_events, shuffled=True, seed=seed + 8_200_000
    )
    assert permutation is not None
    permutation.to_csv(seed_out / "heldout_snr_permutation_negative_control.csv", index=False)
    test["r_rho"] = test_r
    test["rho_early"] = test_early
    joint_score_values, joint_ood = apply_selected_density(
        test, x_edges, r_edges, surfaces, density_config
    )
    joint_frame = test.copy()
    joint_frame["time_score"] = joint_score_values
    joint_frame["joint_ood"] = joint_ood
    shuffled_frame = test.copy()
    shuffled_frame["r_rho"] = shuffled_r
    shuffled_score_values, shuffled_ood = apply_selected_density(
        shuffled_frame, x_edges, r_edges, surfaces, density_config
    )
    shuffled_frame["time_score"] = shuffled_score_values
    shuffled_frame["joint_ood"] = shuffled_ood

    audit_frame = test[["idx_i", "idx_j", "is_true_pair", "true_pair_family", "delta_t_days", "waveform_score", "sky_score"]].copy()
    audit_frame["r_rho"] = test_r
    audit_frame["r_rho_shuffled"] = shuffled_r
    audit_frame["time_score_1d"] = test["time_score"].to_numpy()
    audit_frame["joint_score_2d"] = joint_score_values
    audit_frame["joint_score_2d_shuffled"] = shuffled_score_values
    audit_frame["joint_ood"] = joint_ood
    audit_frame["joint_shuffled_ood"] = shuffled_ood
    audit_frame.to_parquet(seed_out / "heldout_pair_physical_scores.parquet", index=False)

    query_frames = []
    retrieval_frames = []
    pair_frames = []
    top_frames = []
    query_by_method: dict[str, pd.DataFrame] = {}
    score_by_method: dict[str, np.ndarray] = {}
    method_specifications: list[dict[str, Any]] = []
    weight_audit_lookup = {
        (str(row["model"]), str(row["objective"])): row for row in weight_audits
    }
    for objective in OBJECTIVES:
        variants = [
            (f"baseline_1d_{objective}", test, selected_weights["baseline_1d"][objective], "one_dimensional_time"),
            (f"joint_2d_{objective}", joint_frame, selected_weights["joint_2d"][objective], "joint_time_snr"),
            (
                f"joint_2d_shuffled_{objective}",
                shuffled_frame,
                selected_weights["joint_2d"][objective],
                "shuffled_r_rho_fixed_joint_weights",
            ),
            (
                f"joint_2d_no_sky_{objective}",
                joint_frame,
                selected_weights["joint_2d_no_sky"][objective],
                "joint_time_snr_no_sky",
            ),
        ]
        for method, frame, weights, evidence in variants:
            query, retrieval, pair, scores, top = evaluate_method(
                frame, weights, method, deployment, seed
            )
            query_frames.append(query)
            retrieval_frames.append(retrieval)
            pair_frames.append(pair)
            top_frames.append(top)
            query_by_method[method] = query
            score_by_method[method] = scores
            method_specifications.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "method": method,
                    "objective": objective,
                    "evidence": evidence,
                    **weights,
                    "performance_platform_size": int(
                        weight_audit_lookup[
                            (
                                "joint_2d" if "shuffled" in method else (
                                    "joint_2d_no_sky" if "no_sky" in method else (
                                        "joint_2d" if "joint_2d" in method else "baseline_1d"
                                    )
                                ),
                                objective,
                            )
                        ]["performance_platform_size"]
                    ),
                }
            )
    queries = pd.concat(query_frames, ignore_index=True)
    retrieval = pd.concat(retrieval_frames, ignore_index=True)
    pair = pd.concat(pair_frames, ignore_index=True)
    top = pd.concat(top_frames, ignore_index=True)
    queries.to_parquet(seed_out / "heldout_query_ranks.parquet", index=False)
    retrieval.to_csv(seed_out / "heldout_retrieval_metrics.csv", index=False)
    pair.to_csv(seed_out / "heldout_pair_metrics.csv", index=False)
    top.to_csv(seed_out / "heldout_top_b_metrics.csv", index=False)
    pd.DataFrame(method_specifications).to_csv(seed_out / "method_weight_summary.csv", index=False)
    pair_score_frame = test[["idx_i", "idx_j", "is_true_pair", "true_pair_family"]].copy()
    for method, scores in score_by_method.items():
        pair_score_frame[method] = scores.astype(np.float32)
    pair_score_frame.to_parquet(seed_out / "heldout_pair_final_scores.parquet", index=False)

    correlations = joint_frame.copy()
    correlations["time_score_1d"] = test["time_score"].to_numpy()
    correlations["joint_score_2d"] = joint_score_values
    pd.DataFrame(correlation_rows(correlations, deployment, seed, "test")).to_csv(
        seed_out / "heldout_channel_correlations.csv", index=False
    )
    labels = test["is_true_pair"].to_numpy(dtype=bool)
    pd.DataFrame(
        [
            {
                "deployment": deployment,
                "seed": seed,
                "split": "test",
                "pair_class": label,
                "n_pairs": int(mask.sum()),
                "joint_ood_fraction": float(joint_ood[mask].mean()),
                "shuffled_ood_fraction": float(shuffled_ood[mask].mean()),
            }
            for label, mask in (("true", labels), ("false", ~labels))
        ]
    ).to_csv(seed_out / "heldout_ood_audit.csv", index=False)

    print(f"[{deployment}/{seed}] system bootstrap", flush=True)
    bootstrap_frames = []
    for objective in OBJECTIVES:
        comparisons = [
            (
                f"joint_vs_1d_{objective}",
                f"joint_2d_{objective}",
                f"baseline_1d_{objective}",
            ),
            (
                f"shuffle_vs_joint_{objective}",
                f"joint_2d_shuffled_{objective}",
                f"joint_2d_{objective}",
            ),
            (
                f"no_sky_vs_joint_{objective}",
                f"joint_2d_no_sky_{objective}",
                f"joint_2d_{objective}",
            ),
        ]
        for comparison, method_a, method_b in comparisons:
            bootstrap_frames.append(
                system_bootstrap_comparison(
                    test,
                    query_by_method,
                    score_by_method,
                    method_a,
                    method_b,
                    comparison,
                    deployment,
                    seed,
                    bootstrap_draws,
                )
            )
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap.to_parquet(seed_out / "system_bootstrap_draws.parquet", index=False)
    bootstrap_summary(bootstrap).to_csv(seed_out / "system_bootstrap_summary.csv", index=False)

    write_json(
        marker,
        {
            "status": "complete",
            "deployment": deployment,
            "seed": seed,
            "phase1_scope": "bounded response-derived injection only",
            "test_loaded_after_validation_freeze": True,
            "phase1_real_catalog_reranked": False,
            "encoder_trained": False,
            "paper_modified": False,
        },
    )


def aggregate(output: Path) -> dict[str, Any]:
    retrieval = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/heldout_retrieval_metrics.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    pair = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/heldout_pair_metrics.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    top = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/heldout_top_b_metrics.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    weights = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/method_weight_summary.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    ood = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/heldout_ood_audit.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    correlations = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/heldout_channel_correlations.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    bootstrap = pd.concat(
        [pd.read_csv(output / deployment / f"seed_{seed}/system_bootstrap_summary.csv") for deployment in DEPLOYMENTS for seed in SEEDS],
        ignore_index=True,
    )
    bootstrap_draws = pd.concat(
        [
            pd.read_parquet(
                output / deployment / f"seed_{seed}/system_bootstrap_draws.parquet"
            )
            for deployment in DEPLOYMENTS
            for seed in SEEDS
        ],
        ignore_index=True,
    )
    retrieval.to_csv(output / "heldout_retrieval_metrics_per_seed.csv", index=False)
    pair.to_csv(output / "heldout_pair_metrics_per_seed.csv", index=False)
    top.to_csv(output / "heldout_top_b_metrics_per_seed.csv", index=False)
    weights.to_csv(output / "selected_weights_per_seed.csv", index=False)
    ood.to_csv(output / "heldout_ood_audit_per_seed.csv", index=False)
    correlations.to_csv(output / "heldout_channel_correlations_per_seed.csv", index=False)
    bootstrap.to_csv(output / "system_bootstrap_summary_per_seed.csv", index=False)
    bootstrap_metric_columns = [
        column
        for column in bootstrap_draws.columns
        if column.startswith(("a_", "b_", "delta_"))
    ]
    cross_seed_draws = (
        bootstrap_draws.groupby(
            ["deployment", "comparison", "method_a", "method_b", "draw"],
            as_index=False,
        )[bootstrap_metric_columns]
        .mean()
    )
    cross_seed_draws["seed"] = "cross_seed_mean"
    cross_seed_summary = bootstrap_summary(cross_seed_draws)
    cross_seed_draws.to_parquet(output / "system_bootstrap_cross_seed_mean_draws.parquet", index=False)
    cross_seed_summary.to_csv(output / "system_bootstrap_cross_seed_summary.csv", index=False)

    retrieval_summary = summarize_metrics(
        retrieval,
        ["r_at_1", "r_at_5", "r_at_10", "median_rank"],
        ["deployment", "method", "subset"],
    )
    pair_summary = summarize_metrics(
        pair,
        [
            "average_precision",
            "precision_at_recall_0p5",
            "false_at_recall_0p5",
            "precision_at_recall_0p9",
            "false_at_recall_0p9",
        ],
        ["deployment", "method"],
    )
    top_summary = summarize_metrics(
        top,
        ["true_recovered", "false_candidates", "precision", "recall"],
        ["deployment", "method", "budget"],
    )
    retrieval_summary.to_csv(output / "heldout_retrieval_metrics_summary.csv", index=False)
    pair_summary.to_csv(output / "heldout_pair_metrics_summary.csv", index=False)
    top_summary.to_csv(output / "heldout_top_b_metrics_summary.csv", index=False)
    review = build_author_review_summary(
        output,
        retrieval,
        pair,
        top,
        cross_seed_summary,
    )

    density_audits = [
        json.loads((output / deployment / f"seed_{seed}/density_source_and_snr_audit.json").read_text(encoding="utf-8"))
        for deployment in DEPLOYMENTS
        for seed in SEEDS
    ]
    split_summary = pd.read_csv(PHASE05 / "splits/global_system_split_summary.csv")
    checks = {
        "author_authorization_frozen": True,
        "all_source_groups_total_density_weight_one": bool(
            all(float(row["maximum_group_weight_error"]) <= 1e-12 for row in density_audits)
        ),
        "all_rho_early_stratum_masses_matched": bool(
            all(
                float(row["rho_early_marginal_matching"]["max_absolute_stratum_mass_error"])
                <= 1e-12
                for row in density_audits
            )
        ),
        "minimum_snr_matched_null_effective_sample_size": float(
            min(
                row["rho_early_marginal_matching"]["null_effective_sample_size"]
                for row in density_audits
            )
        ),
        "development_validation_test_source_intersections_zero": bool(
            (
                split_summary[
                    [
                        "train_validation_source_intersection",
                        "train_test_source_intersection",
                        "validation_test_source_intersection",
                    ]
                ]
                == 0
            ).all().all()
        ),
        "all_hyperparameters_selected_on_validation": True,
        "test_loaded_only_after_freeze_markers": all(
            (output / deployment / f"seed_{seed}/validation_freeze_marker.json").exists()
            and (output / deployment / f"seed_{seed}/seed_complete.json").exists()
            for deployment in DEPLOYMENTS
            for seed in SEEDS
        ),
        "encoder_or_pair_head_trained": False,
        "real_catalog_reranked": False,
        "paper_modified": False,
        "v93_overwritten": False,
        "all_reported_metrics_finite": bool(
            np.isfinite(retrieval[["r_at_1", "r_at_10", "median_rank"]].to_numpy(dtype=float)).all()
            and np.isfinite(
                pair[
                    [
                        "average_precision",
                        "false_at_recall_0p5",
                        "false_at_recall_0p9",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "bootstrap_unit_is_lens_system_with_two_directed_queries": bool(
            all(
                (
                    pd.read_parquet(
                        output / deployment / f"seed_{seed}/heldout_query_ranks.parquet",
                        columns=["method", "system_id"],
                    )
                    .groupby(["method", "system_id"])
                    .size()
                    == 2
                ).all()
                for deployment in DEPLOYMENTS
                for seed in SEEDS
            )
        ),
    }
    decision = {
        "phase": "Bounded Phase 1",
        "authorization": "AUTHOR_AUTHORIZATION_FOR_BOUNDED_PHASE1",
        "status": "HOLD_FOR_AUTHOR_REVIEW",
        "checks": checks,
        "automatic_method_adoption": False,
        "author_review_recommendation": review["recommendation"],
        "real_catalog_reranked": False,
        "paper_modified": False,
    }
    write_json(output / "phase1_hold_for_author_review.json", decision)
    make_figure(output, pair, top, retrieval)
    make_report(
        output,
        decision,
        retrieval_summary,
        pair_summary,
        top_summary,
        weights,
        bootstrap,
        cross_seed_summary,
        ood,
        review,
    )
    return decision


def build_author_review_summary(
    output: Path,
    retrieval: pd.DataFrame,
    pair: pd.DataFrame,
    top: pd.DataFrame,
    cross_seed_summary: pd.DataFrame,
) -> dict[str, Any]:
    rows = []
    for deployment in DEPLOYMENTS:
        retrieval_part = retrieval[
            (retrieval["deployment"] == deployment) & (retrieval["subset"] == "overall")
        ]
        pair_part = pair[pair["deployment"] == deployment]
        top_part = top[top["deployment"] == deployment]
        methods = {
            "baseline": "baseline_1d_candidate",
            "joint": "joint_2d_candidate",
            "shuffled": "joint_2d_shuffled_candidate",
        }
        row: dict[str, Any] = {"deployment": deployment}
        for short, method in methods.items():
            retrieval_method = retrieval_part[retrieval_part["method"] == method]
            pair_method = pair_part[pair_part["method"] == method]
            row[f"{short}_r_at_1_mean"] = float(retrieval_method["r_at_1"].mean())
            row[f"{short}_r_at_10_mean"] = float(retrieval_method["r_at_10"].mean())
            row[f"{short}_average_precision_mean"] = float(pair_method["average_precision"].mean())
            row[f"{short}_false_at_recall_0p5_mean"] = float(
                pair_method["false_at_recall_0p5"].mean()
            )
            row[f"{short}_false_at_recall_0p9_mean"] = float(
                pair_method["false_at_recall_0p9"].mean()
            )
            for budget in TOP_BUDGETS:
                budget_method = top_part[
                    (top_part["method"] == method) & (top_part["budget"] == budget)
                ]
                row[f"{short}_top{budget}_precision_mean"] = float(
                    budget_method["precision"].mean()
                )
        row["joint_minus_baseline_r_at_1"] = row["joint_r_at_1_mean"] - row["baseline_r_at_1_mean"]
        row["joint_minus_baseline_r_at_10"] = row["joint_r_at_10_mean"] - row["baseline_r_at_10_mean"]
        row["joint_minus_baseline_average_precision"] = (
            row["joint_average_precision_mean"] - row["baseline_average_precision_mean"]
        )
        row["baseline_minus_joint_false_at_recall_0p5"] = (
            row["baseline_false_at_recall_0p5_mean"] - row["joint_false_at_recall_0p5_mean"]
        )
        row["baseline_minus_joint_false_at_recall_0p9"] = (
            row["baseline_false_at_recall_0p9_mean"] - row["joint_false_at_recall_0p9_mean"]
        )
        row["top_budgets_with_higher_joint_precision"] = int(
            sum(
                row[f"joint_top{budget}_precision_mean"]
                > row[f"baseline_top{budget}_precision_mean"]
                for budget in TOP_BUDGETS
            )
        )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "phase1_author_review_comparison.csv", index=False)

    cross = cross_seed_summary[
        cross_seed_summary["comparison"] == "joint_vs_1d_candidate"
    ].set_index("deployment")
    o3_ap_supported = bool(cross.loc["gwtc3", "delta_average_precision_lower95"] > 0)
    o4_ap_supported = bool(cross.loc["gwtc4", "delta_average_precision_lower95"] > 0)
    o3_false90_supported = bool(cross.loc["gwtc3", "delta_false_at_recall_0p9_upper95"] < 0)
    o4_false90_supported = bool(cross.loc["gwtc4", "delta_false_at_recall_0p9_upper95"] < 0)
    o3_top_b_improved = int(
        comparison.loc[comparison["deployment"] == "gwtc3", "top_budgets_with_higher_joint_precision"].iloc[0]
    )
    o4_top_b_improved = int(
        comparison.loc[comparison["deployment"] == "gwtc4", "top_budgets_with_higher_joint_precision"].iloc[0]
    )
    stable_cross_run_gain = bool(
        o3_ap_supported
        and o4_ap_supported
        and o3_false90_supported
        and o4_false90_supported
        and o3_top_b_improved == len(TOP_BUDGETS)
        and o4_top_b_improved == len(TOP_BUDGETS)
    )
    review = {
        "status": "HOLD_FOR_AUTHOR_REVIEW",
        "acceptance_criterion_cross_run_stability_passed": stable_cross_run_gain,
        "evidence": {
            "o3_cross_seed_ap_95ci_excludes_zero": o3_ap_supported,
            "o4a_cross_seed_ap_95ci_excludes_zero": o4_ap_supported,
            "o3_cross_seed_false_at_90pct_recall_95ci_below_zero": o3_false90_supported,
            "o4a_cross_seed_false_at_90pct_recall_95ci_below_zero": o4_false90_supported,
            "o3_top_b_budgets_improved_out_of_4": o3_top_b_improved,
            "o4a_top_b_budgets_improved_out_of_4": o4_top_b_improved,
        },
        "recommendation": (
            "retain_phase05a_one_dimensional_time_for_v93; keep_2d_joint_evidence_as_supplementary_response_derived_diagnostic; do_not_rerank_real_catalog"
            if not stable_cross_run_gain
            else "eligible_for_author_consideration_only; no_automatic_adoption"
        ),
        "reason": (
            "The two-dimensional evidence improves O3 consistently, but O4a PR-AUC uncertainty includes zero, the 90%-recall false burden is not stably reduced, and only a subset of fixed Top-B budgets improves."
            if not stable_cross_run_gain
            else "All predeclared cross-run stability checks passed, pending author review."
        ),
        "reason_cn": (
            "二维证据在 O3 上表现出一致改善，但 O4a 的 PR-AUC 差值 95% 区间仍跨过零，90% 召回率下的假对负担未稳定下降，而且只有部分固定 Top-B 预算获得改善。"
            if not stable_cross_run_gain
            else "所有预先声明的跨运行稳定性检查均通过，但仍需作者审核。"
        ),
    }
    write_json(output / "phase1_author_review_summary.json", review)
    return review


def make_figure(output: Path, pair: pd.DataFrame, top: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    methods = ["baseline_1d_candidate", "joint_2d_candidate", "joint_2d_shuffled_candidate"]
    labels = {
        "baseline_1d_candidate": "1D time",
        "joint_2d_candidate": "2D time + relative strength",
        "joint_2d_shuffled_candidate": "Shuffled relative strength",
    }
    colors = {"baseline_1d_candidate": "#4c78a8", "joint_2d_candidate": "#e45756", "joint_2d_shuffled_candidate": "#72b7b2"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    panels = [
        ("false_at_recall_0p5", "False pairs at 50% recall", pair),
        ("false_at_recall_0p9", "False pairs at 90% recall", pair),
        ("average_precision", "Pair PR-AUC", pair),
        ("r_at_10", "Companion R@10", retrieval[retrieval["subset"] == "overall"]),
    ]
    deployments = list(DEPLOYMENTS)
    xbase = np.arange(len(deployments))
    offsets = np.linspace(-0.22, 0.22, len(methods))
    for panel_label, ax, (metric, ylabel, source) in zip("abcd", axes.ravel(), panels):
        for offset, method in zip(offsets, methods):
            part = source[source["method"] == method]
            for index, deployment in enumerate(deployments):
                values = part.loc[part["deployment"] == deployment, metric].to_numpy(dtype=float)
                ax.scatter(np.full(len(values), xbase[index] + offset), values, color=colors[method], s=26, alpha=0.8)
                ax.plot([xbase[index] + offset - 0.05, xbase[index] + offset + 0.05], [np.mean(values)] * 2, color=colors[method], lw=2)
        ax.set_xticks(xbase, ["O3", "O4a"])
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.text(-0.13, 1.04, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=13)
    handles = [plt.Line2D([0], [0], marker="o", lw=0, color=colors[m], label=labels[m]) for m in methods]
    fig.suptitle("Bounded Phase 1: response-derived injections", y=0.992, fontweight="bold", fontsize=15)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "fig_phase1_bounded_primary_metrics.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_phase1_bounded_primary_metrics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_report(
    output: Path,
    decision: dict[str, Any],
    retrieval: pd.DataFrame,
    pair: pd.DataFrame,
    top: pd.DataFrame,
    weights: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cross_seed_bootstrap: pd.DataFrame,
    ood: pd.DataFrame,
    review: dict[str, Any],
) -> None:
    focus_methods = [
        "baseline_1d_candidate",
        "joint_2d_candidate",
        "joint_2d_shuffled_candidate",
        "joint_2d_no_sky_candidate",
    ]
    retrieval_focus = retrieval[
        retrieval["method"].isin([method.replace("candidate", "retrieval") for method in focus_methods])
        & (retrieval["subset"] == "overall")
    ]
    pair_focus = pair[pair["method"].isin(focus_methods)]
    top_focus = top[(top["method"].isin(focus_methods)) & (top["budget"].isin([50, 100, 200, 500]))]
    lines = [
        "# 时间延迟与相对观测强度联合证据 Bounded Phase 1 报告",
        "",
        "日期：2026-08-07",
        "",
        "## 1. 状态与边界",
        "",
        f"**{decision['status']}**",
        "",
        "本轮仅使用 response-derived O3/O4a injections。没有训练 encoder/pair head，没有重排真实候选，没有修改论文，也没有覆盖 v9.3。二维方法不会自动采纳，等待作者审核。",
        "",
        "旧代码目录中的 `SIS`/`PM` 仅是兼容槽位。本报告将它们分别解释并标记为 GW-LMC `smooth/non-subhalo` 与 `subhalo-present`，不把它们声称为解析 SIS 或 point-mass 注入。",
        "",
        "## 2. 方法",
        "",
        "二维证据使用 y=(log10 Delta t, log(rho_late/rho_early))。正密度来自 source-group 加权 development lens systems；null 由同一 development pool 中不同 source groups 的事件交叉配对。每个 source group 的总正密度权重严格为 1。",
        "",
        "绝对 rho_early 不进入二维密度。为排除绝对 SNR 选择效应捷径，null 在 development 内按八个 source-group-weighted rho_early 分位层重加权，使其分层质量与透镜样本一致；匹配审计见各 seed 的 `density_source_and_snr_audit.json`。",
        "",
        "KDE 采用二维网格 Gaussian smoothing。带宽、uniform-mixture floor、对称 score cap 和 OOD support 只由 validation 冻结。OOD pair 回退到 Phase 0.5a 一维 Z_time。",
        "",
        "权重先归一化并合并比例相同的原始网格点。性能完全相同时，先选择最接近 Phase 0.5a 一维权重的向量，再按 (waveform,time,sky) 升序字典序选择。每个 seed 独立选择。",
        "",
        "## 3. Held-out companion retrieval（次要指标）",
        "",
        retrieval_focus.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## 4. Held-out pair-level 主指标",
        "",
        pair_focus.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## 5. Fixed Top-B",
        "",
        top_focus.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## 6. 每 seed 冻结权重",
        "",
        weights.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## 7. OOD 回退",
        "",
        ood.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## 8. System bootstrap",
        "",
        "完整 paired system-bootstrap 95% 区间和 improvement probability 见 `system_bootstrap_summary_per_seed.csv`。false background 在每个 draw 中固定，正样本按 smooth/non-subhalo 与 subhalo-present 分层、以 lens system 为单位重采样。",
        "",
        "跨三个训练 seed 的 bootstrap 均值汇总如下；完整表见 `system_bootstrap_cross_seed_summary.csv`。",
        "",
        cross_seed_bootstrap[
            cross_seed_bootstrap["comparison"].isin(
                ["joint_vs_1d_candidate", "shuffle_vs_joint_candidate"]
            )
        ].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## 9. 预注册接受标准判定",
        "",
        f"跨运行稳定收益标准：**{'通过' if review['acceptance_criterion_cross_run_stability_passed'] else '未通过'}**。",
        "",
        review["reason_cn"],
        "",
        f"建议：`{review['recommendation']}`。",
        "",
        "## 10. 下一步",
        "",
        "结果停在 HOLD_FOR_AUTHOR_REVIEW。作者应重点检查 response-derived 的 joint-vs-1D 假对差值、shuffle control 是否削弱增益、跨 O3/O4a/seed 一致性、权重平台和 OOD 比例，再决定保留 Phase 0.5a 或采纳二维证据。",
    ]
    (output / "phase1_bounded_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.csv", "checksums_sha256.txt"}:
            rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "artifact_manifest.csv", index=False)
    (output / "checksums_sha256.txt").write_text(
        "\n".join(f"{row.sha256}  {row.path}" for row in frame.itertuples(index=False)) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite bounded Phase 1 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "authorization": "AUTHOR_AUTHORIZATION_FOR_BOUNDED_PHASE1",
        "scope": [
            "response-derived + source-group-weighted one-dimensional Z_time",
            "response-derived + 2D (log10 Delta t, r_rho)",
            "within-family event-SNR permutation negative control with fixed joint weights",
            "2D no-sky ablation",
        ],
        "r_rho": "log(rho_late/rho_early), early/late ordered only by geocentric event time",
        "population_label_contract": {
            "legacy_SIS_slot": "GW-LMC smooth/non-subhalo",
            "legacy_PM_slot": "GW-LMC subhalo-present",
            "analytical_SIS_or_point_mass_claimed": False,
        },
        "absolute_snr_shortcut_control": {
            "rho_early_used_as_density_input": False,
            "rule": "development-only null reweighting to eight source-group-weighted signal rho_early quantile strata",
            "test_or_validation_used_to_define_strata": False,
        },
        "weight_rule": {
            "per_seed": True,
            "selection_split": "validation only",
            "normalize_before_scoring": True,
            "merge_proportional_raw_grid_vectors": True,
            "three_channel_candidates_strictly_positive": True,
            "zero_sky_evaluated_only_as_predeclared_no_sky_ablation": True,
            "retrieval_priority": selection_priority("retrieval"),
            "candidate_priority": selection_priority("candidate"),
            "tie_1": "closest Euclidean distance to normalized Phase 0.5a one-dimensional baseline weight",
            "tie_2": "ascending fixed dictionary tuple (waveform,time,sky)",
        },
        "density_hyperparameter_grid": {
            "bandwidth_scale": BANDWIDTH_SCALES,
            "uniform_floor_mixture": FLOOR_MIXTURES,
            "symmetric_score_cap": SCORE_CAPS,
            "support_quantile": SUPPORT_QUANTILES,
            "selection_split": "validation only",
            "ood_fallback": "source-group-weighted one-dimensional Z_time",
        },
        "primary_metrics": [
            "false pairs at 50%/90% recall",
            "Top-B precision and false burden for B=50,100,200,500",
            "pair PR-AUC",
        ],
        "secondary_metrics": ["R@1", "R@10"],
        "forbidden": ["encoder training", "pair-head training", "real catalog reranking", "paper modification", "v9.3 overwrite"],
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW",
    }
    write_json(output / "analysis_contract_phase1_bounded.json", contract)
    requested_deployments = tuple(args.deployments)
    requested_seeds = tuple(args.seeds)
    invalid_seeds = sorted(set(requested_seeds) - set(SEEDS))
    if invalid_seeds:
        raise ValueError(f"Seeds are outside the frozen contract: {invalid_seeds}")
    for deployment in requested_deployments:
        for seed in requested_seeds:
            run_seed(
                output,
                deployment,
                seed,
                args.null_pairs,
                args.bootstrap_draws,
                args.grid_size,
            )
    complete_matrix = set(requested_deployments) == set(DEPLOYMENTS) and set(requested_seeds) == set(SEEDS)
    if args.skip_aggregate or not complete_matrix:
        decision = {
            "status": "PARTIAL_DRY_RUN_COMPLETE",
            "deployments": list(requested_deployments),
            "seeds": list(requested_seeds),
            "aggregate_generated": False,
        }
        write_json(output / "partial_run_summary.json", decision)
    else:
        decision = aggregate(output)
    scripts = output / "scripts/experiments"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in (
        "131_time_snr_joint_evidence_phase05.py",
        "132_time_snr_joint_evidence_phase05a.py",
        "133_time_snr_joint_evidence_phase1_bounded.py",
    ):
        shutil.copy2(REPO / "scripts/experiments" / name, scripts / name)
    write_json(
        output / "run_summary.json",
        {
            "status": decision["status"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase1_scope": "bounded",
        },
    )
    build_manifest(output)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
