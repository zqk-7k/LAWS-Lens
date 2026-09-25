from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8
SECONDS_PER_DAY = 86400.0


def angular_sep_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def unit_from_radec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float64)


def observed_sky_pair_features(sky_obs: pd.DataFrame, chunk_rows: int = 48) -> dict[str, np.ndarray]:
    """Build pairwise observed-sky features from ra_obs/dec_obs/A90-derived sigma.

    The input is an observed posterior summary, not true sky. Returned matrices are
    full catalog pair features and keep the diagonal at -inf for scorer safety.
    """
    vec = unit_from_radec(sky_obs["ra_obs"].to_numpy(dtype=np.float64), sky_obs["dec_obs"].to_numpy(dtype=np.float64))
    sigma = sky_obs["sky_sigma_rad"].to_numpy(dtype=np.float64)
    n = len(sky_obs)
    sep = np.empty((n, n), dtype=np.float32)
    norm_sep = np.empty((n, n), dtype=np.float32)
    step = np.empty((n, n), dtype=np.float32)
    gaussian = np.empty((n, n), dtype=np.float32)
    log_overlap = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n))
        theta = angular_sep_unit(vec[rows, None, :], vec[None, :, :])
        sigma_ij = np.sqrt(sigma[rows, None] ** 2 + sigma[None, :] ** 2)
        d_sky = theta / np.maximum(sigma_ij, EPS)
        st = np.full(d_sky.shape, -0.5, dtype=np.float32)
        st[d_sky <= 3.03] = 0.1
        st[d_sky <= 2.15] = 0.5
        st[d_sky <= 1.18] = 1.0
        var = np.maximum(sigma_ij ** 2, EPS)
        sep[rows] = theta.astype(np.float32)
        norm_sep[rows] = d_sky.astype(np.float32)
        step[rows] = st
        gaussian[rows] = np.exp(-0.5 * d_sky * d_sky).astype(np.float32)
        log_overlap[rows] = (-np.log(2.0 * np.pi * var) - (theta * theta) / (2.0 * var)).astype(np.float32)
    for matrix in (sep, norm_sep, step, gaussian, log_overlap):
        np.fill_diagonal(matrix, -np.inf)
    return {
        "sky_sep_obs": sep,
        "sky_norm_sep": norm_sep,
        "sky_step_weight": step,
        "sky_gaussian_weight": gaussian,
        "sky_log_overlap": log_overlap,
    }


def time_step_score_matrix(time_obs: pd.DataFrame, lensed_days: np.ndarray, chunk_rows: int = 48) -> np.ndarray:
    """Quantile step baseline derived from the lensed delay distribution."""
    q01, q05, q10, q90, q95, q99 = np.percentile(lensed_days, [1, 5, 10, 90, 95, 99])
    t = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    n = len(t)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n))
        dt_days = np.abs(t[rows, None] - t[None, :]) / SECONDS_PER_DAY
        st = np.full(dt_days.shape, -0.5, dtype=np.float32)
        st[(dt_days >= q01) & (dt_days <= q99)] = 0.1
        st[(dt_days >= q05) & (dt_days <= q95)] = 0.5
        st[(dt_days >= q10) & (dt_days <= q90)] = 1.0
        out[rows] = st
    np.fill_diagonal(out, -np.inf)
    return out


def rank_feature_matrices(waveform_score: np.ndarray, chunk_rows: int = 48) -> dict[str, np.ndarray]:
    """Create rank, reciprocal-rank, and margin features from waveform scores."""
    n = waveform_score.shape[0]
    rank = np.empty((n, n), dtype=np.float32)
    reciprocal = np.empty((n, n), dtype=np.float32)
    margin = np.empty((n, n), dtype=np.float32)
    masked = waveform_score.astype(np.float32).copy()
    np.fill_diagonal(masked, -np.inf)
    for start in range(0, n, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n))
        block = masked[rows]
        order = np.argsort(-block, axis=1)
        local_rank = np.empty_like(order, dtype=np.float32)
        local_rank[np.arange(order.shape[0])[:, None], order] = np.arange(1, n + 1, dtype=np.float32)
        top1 = block[np.arange(order.shape[0]), order[:, 0]][:, None]
        rank[rows] = local_rank
        reciprocal[rows] = 1.0 / local_rank
        margin[rows] = block - top1
    rank_score = -np.log1p(rank)
    for matrix in (rank_score, reciprocal, margin):
        np.fill_diagonal(matrix, -np.inf)
    return {
        "waveform_rank": rank_score,
        "waveform_reciprocal_rank": reciprocal,
        "waveform_margin": margin,
    }
