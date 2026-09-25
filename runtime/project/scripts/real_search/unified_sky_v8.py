#!/usr/bin/env python3
"""Unified posterior-map sky evidence for ET and real GWTC catalogs.

The public interface deliberately separates three layers:

1. event-level HEALPix posterior normalization and diagnostics;
2. physically interpretable pair statistics (common-source Bayes factor,
   90%-credible-region overlap, and cross-HPD);
3. a validation-frozen, class-balanced logistic calibration of those pair
   statistics.

The same functions are used for synthetic ET posteriors, real-noise
injection posteriors, and real GWTC PE sky maps.  No test labels or real
catalog ranks are used while fitting the calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import healpy as hp
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score


SKY_FEATURE_COLUMNS = (
    "sky_log_bayes_factor_raw",
    "sky_o90",
    "sky_cross_hpd",
)
FULL_SKY_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2


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
    raise TypeError(f"Cannot serialize {type(value)!r}")


def normalize_probability_maps(maps: np.ndarray) -> np.ndarray:
    """Return finite, non-negative HEALPix probability-mass vectors."""
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [event, pixel] maps, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = values.sum(axis=1, keepdims=True)
    invalid = total[:, 0] <= 0.0
    if np.any(invalid):
        values[invalid] = 1.0
        total = values.sum(axis=1, keepdims=True)
    return (values / total).astype(np.float32)


def change_nside_probability_mass(maps: np.ndarray, nside_out: int) -> np.ndarray:
    """Change HEALPix resolution while preserving probability mass."""
    values = normalize_probability_maps(maps)
    nside_in = hp.npix2nside(values.shape[1])
    if nside_in == nside_out:
        return values
    output = np.empty((len(values), hp.nside2npix(nside_out)), dtype=np.float32)
    for index, probability in enumerate(values):
        # hp.ud_grade averages child pixels. power=-2 converts the average back
        # to probability mass when degrading a RING-ordered map.
        output[index] = hp.ud_grade(
            probability,
            nside_out=nside_out,
            order_in="RING",
            order_out="RING",
            power=-2,
        )
    return normalize_probability_maps(output)


def credible_levels(maps: np.ndarray) -> np.ndarray:
    """Return each pixel's highest-posterior-density credible level."""
    probability = normalize_probability_maps(maps)
    order = np.argsort(-probability, axis=1)
    sorted_probability = np.take_along_axis(probability, order, axis=1)
    cumulative = np.cumsum(sorted_probability, axis=1, dtype=np.float64)
    levels = np.empty_like(probability, dtype=np.float32)
    np.put_along_axis(levels, order, cumulative.astype(np.float32), axis=1)
    return np.minimum(levels, 1.0)


def event_map_diagnostics(
    maps: np.ndarray,
    *,
    event_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute event-level map information and reusable HPD arrays."""
    probability = normalize_probability_maps(maps)
    levels = credible_levels(probability)
    mask90 = levels <= (0.9 + 1e-7)
    # Always retain the MAP pixel, including very coarse maps.
    map_pixel = np.argmax(probability, axis=1).astype(np.int32)
    mask90[np.arange(len(probability)), map_pixel] = True
    npix = probability.shape[1]
    pixel_area = FULL_SKY_DEG2 / npix
    probability64 = probability.astype(np.float64)
    positive = probability64 > 0
    entropy = -np.sum(
        np.where(
            positive,
            probability64 * np.log(np.maximum(probability64, 1e-300)),
            0.0,
        ),
        axis=1,
    )
    kl_uniform = math.log(npix) - entropy
    names = (
        list(event_names)
        if event_names is not None
        else [f"event_{index}" for index in range(len(probability))]
    )
    frame = pd.DataFrame(
        {
            "idx": np.arange(len(probability), dtype=np.int32),
            "event_name": names,
            "nside": hp.npix2nside(npix),
            "map_sum": probability.sum(axis=1),
            "map_min": probability.min(axis=1),
            "map_max": probability.max(axis=1),
            "map_pixel": map_pixel,
            "area90_deg2": mask90.sum(axis=1) * pixel_area,
            "entropy_nats": entropy,
            "kl_to_uniform_nats": kl_uniform,
            "effective_area_deg2": np.exp(entropy) * pixel_area,
        }
    )
    return frame, levels, mask90


def _pair_statistics_block(
    probability: torch.Tensor,
    levels: torch.Tensor,
    mask90: torch.Tensor,
    map_pixel: torch.Tensor,
    area90_pixels: torch.Tensor,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_probability = probability[start:stop]
    raw = block_probability @ probability.T
    log_bf = torch.log(torch.clamp(raw * probability.shape[1], min=1e-30))

    intersection = mask90[start:stop] @ mask90.T
    denominator = torch.minimum(
        area90_pixels[start:stop, None],
        area90_pixels[None, :],
    )
    o90 = intersection / torch.clamp(denominator, min=1.0)

    # H(p,q) is q's HPD mass above q evaluated at p's MAP.  The symmetric
    # statistic of Wong et al. is max(1-H(p,q), 1-H(q,p)).
    q_at_p_map = levels[:, map_pixel[start:stop]].T
    p_at_q_map = levels[start:stop][:, map_pixel]
    cross_hpd = torch.maximum(1.0 - q_at_p_map, 1.0 - p_at_q_map)
    return (
        log_bf.float().cpu().numpy(),
        o90.float().cpu().numpy(),
        cross_hpd.float().cpu().numpy(),
    )


def pair_statistic_matrices(
    maps: np.ndarray,
    *,
    device: str | None = None,
    block_size: int = 1024,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Compute all symmetric pair statistics in GPU/CPU blocks."""
    probability = normalize_probability_maps(maps)
    diagnostics, levels_np, mask90_np = event_map_diagnostics(probability)
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(chosen)
    probability_t = torch.as_tensor(probability, dtype=torch.float32, device=torch_device)
    levels_t = torch.as_tensor(levels_np, dtype=torch.float32, device=torch_device)
    mask90_t = torch.as_tensor(mask90_np, dtype=torch.float32, device=torch_device)
    map_pixel_t = torch.as_tensor(
        diagnostics["map_pixel"].to_numpy(np.int64),
        dtype=torch.long,
        device=torch_device,
    )
    area90_t = mask90_t.sum(dim=1)

    n = len(probability)
    matrices = {
        name: np.empty((n, n), dtype=np.float32)
        for name in SKY_FEATURE_COLUMNS
    }
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        log_bf, o90, cross_hpd = _pair_statistics_block(
            probability_t,
            levels_t,
            mask90_t,
            map_pixel_t,
            area90_t,
            start,
            stop,
        )
        matrices["sky_log_bayes_factor_raw"][start:stop] = log_bf
        matrices["sky_o90"][start:stop] = o90
        matrices["sky_cross_hpd"][start:stop] = cross_hpd
    for matrix in matrices.values():
        matrix[:] = 0.5 * (matrix + matrix.T)
    return matrices, diagnostics


def pair_feature_frame(
    maps: np.ndarray,
    *,
    idx_i: np.ndarray | None = None,
    idx_j: np.ndarray | None = None,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unordered pair features and event-map diagnostics."""
    matrices, diagnostics = pair_statistic_matrices(maps, device=device)
    if idx_i is None or idx_j is None:
        idx_i, idx_j = np.triu_indices(len(maps), k=1)
    idx_i = np.asarray(idx_i, dtype=np.int32)
    idx_j = np.asarray(idx_j, dtype=np.int32)
    frame = pd.DataFrame({"idx_i": idx_i, "idx_j": idx_j})
    for name, matrix in matrices.items():
        frame[name] = matrix[idx_i, idx_j]
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
    return frame, diagnostics


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    n_positive = max(int(labels.sum()), 1)
    n_negative = max(int((labels == 0).sum()), 1)
    output = np.empty(len(labels), dtype=np.float64)
    output[labels == 1] = 0.5 / n_positive
    output[labels == 0] = 0.5 / n_negative
    return output


def fit_sky_calibration(
    frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    deployment: str,
    seed: int,
    l2_penalty: float = 1.0,
) -> dict[str, Any]:
    """Fit a monotone, class-balanced validation-only sky logit.

    Coefficients are constrained non-negative because each input statistic is
    defined so that larger values mean stronger common-sky consistency.  This
    prevents a finite validation sample from learning an unphysical sign.
    """
    labels = np.asarray(labels, dtype=np.int8)
    x = frame.loc[:, SKY_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-8)
    z = (x - center) / scale
    weights = _balanced_weights(labels)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficient = parameters[1:]
        logit = intercept + z @ coefficient
        # Stable logistic negative log likelihood.
        loss_rows = np.logaddexp(0.0, logit) - labels * logit
        loss = float(np.sum(weights * loss_rows))
        loss += 0.5 * l2_penalty * float(coefficient @ coefficient)
        residual = weights * (1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40))) - labels)
        gradient = np.concatenate(
            [
                [np.sum(residual)],
                z.T @ residual + l2_penalty * coefficient,
            ]
        )
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(1 + len(SKY_FEATURE_COLUMNS), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None)] + [(0.0, None)] * len(SKY_FEATURE_COLUMNS),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    parameters = fit.x
    score = parameters[0] + z @ parameters[1:]
    true = score[labels == 1]
    null = score[labels == 0]
    true_o90 = frame.loc[labels == 1, "sky_o90"].to_numpy(dtype=np.float64)
    o90_retention = float(np.mean(true_o90 > 0.0)) if len(true_o90) else 0.0
    hard_veto = bool(o90_retention >= 0.99)
    return {
        "version": "unified_sky_v8",
        "deployment": deployment,
        "seed": int(seed),
        "definition": (
            "Class-balanced monotone logistic density-ratio surrogate fitted "
            "only on synthetic validation pairs."
        ),
        "raw_common_source_bayes_factor": (
            "B_sky=N_pix*sum_k(P_i[k]*P_j[k]) under a uniform sky prior."
        ),
        "feature_columns": list(SKY_FEATURE_COLUMNS),
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(parameters[0]),
        "coefficients": parameters[1:].tolist(),
        "coefficient_constraints": "non-negative for all consistency features",
        "l2_penalty": float(l2_penalty),
        "class_weighting": "0.5 total positive weight and 0.5 total null weight",
        "fit_success": bool(fit.success),
        "fit_message": str(fit.message),
        "selection_split": "synthetic validation systems only",
        "test_labels_used": False,
        "real_catalog_ranks_used": False,
        "validation": {
            "n_pairs": int(len(labels)),
            "n_true_pairs": int(labels.sum()),
            "n_false_pairs": int((labels == 0).sum()),
            "roc_auc": float(roc_auc_score(labels, score)),
            "average_precision": float(average_precision_score(labels, score)),
            "true_score_median": float(np.median(true)),
            "null_score_median": float(np.median(null)),
            "true_pair_o90_nonzero_retention": o90_retention,
        },
        "o90_hard_veto": {
            "enabled": hard_veto,
            "rule": "enabled only if validation true-pair retention for O90>0 is >=0.99",
            "validation_retention": o90_retention,
            "veto_score": -30.0,
        },
    }


def apply_sky_calibration(
    frame: pd.DataFrame,
    calibration: dict[str, Any],
) -> np.ndarray:
    columns = calibration["feature_columns"]
    x = frame.loc[:, columns].to_numpy(dtype=np.float64)
    center = np.asarray(calibration["feature_center"], dtype=np.float64)
    scale = np.asarray(calibration["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(calibration["coefficients"], dtype=np.float64)
    score = float(calibration["intercept"]) + ((x - center) / scale) @ coefficient
    gate = calibration["o90_hard_veto"]
    if gate["enabled"]:
        score = np.where(
            frame["sky_o90"].to_numpy(dtype=np.float64) > 0.0,
            score,
            float(gate["veto_score"]),
        )
    return score.astype(np.float32)


def apply_sky_calibration_to_matrices(
    matrices: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:
    center = np.asarray(calibration["feature_center"], dtype=np.float32)
    scale = np.asarray(calibration["feature_scale"], dtype=np.float32)
    coefficient = np.asarray(calibration["coefficients"], dtype=np.float32)
    score = np.full_like(matrices[SKY_FEATURE_COLUMNS[0]], calibration["intercept"])
    for index, name in enumerate(calibration["feature_columns"]):
        score += coefficient[index] * (matrices[name] - center[index]) / scale[index]
    gate = calibration["o90_hard_veto"]
    if gate["enabled"]:
        score[matrices["sky_o90"] <= 0.0] = float(gate["veto_score"])
    np.fill_diagonal(score, -np.inf)
    return score.astype(np.float32)


def attach_sky_features(
    base: pd.DataFrame,
    sky: pd.DataFrame,
    calibration: dict[str, Any],
) -> pd.DataFrame:
    """Replace v7 sky columns while preserving waveform/time evidence."""
    keys = ["idx_i", "idx_j"]
    payload = sky.copy()
    payload["sky_score"] = apply_sky_calibration(payload, calibration)
    output = base.drop(
        columns=[
            column
            for column in base.columns
            if column.startswith("sky_")
            or column in {"sky_score", "sky_bayes_factor"}
        ],
        errors="ignore",
    ).merge(payload, on=keys, how="left", validate="one_to_one")
    output["sky_bayes_factor"] = np.exp(
        np.clip(output["sky_log_bayes_factor_raw"], -80, 80)
    )
    return output
