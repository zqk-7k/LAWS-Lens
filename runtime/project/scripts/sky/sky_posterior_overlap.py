from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS = 1e-300


def angular_separation_rad(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
) -> np.ndarray:
    ra1 = np.asarray(ra1, dtype=np.float64)
    dec1 = np.asarray(dec1, dtype=np.float64)
    ra2 = np.asarray(ra2, dtype=np.float64)
    dec2 = np.asarray(dec2, dtype=np.float64)
    sin1 = np.sin(dec1)
    sin2 = np.sin(dec2)
    cos1 = np.cos(dec1)
    cos2 = np.cos(dec2)
    cos_dra = np.cos(ra1 - ra2)
    cos_theta = np.clip(sin1 * sin2 + cos1 * cos2 * cos_dra, -1.0, 1.0)
    return np.arccos(cos_theta)


def gaussian_log_cosine_overlap(
    theta_rad: np.ndarray,
    sigma_i_rad: np.ndarray,
    sigma_j_rad: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Log cosine overlap for two locally Gaussian sky posteriors.

    For p_i(x)=N(mu_i, sigma_i^2 I) and p_j(x)=N(mu_j, sigma_j^2 I),
    cosine overlap is

      C_ij = 2 sigma_i sigma_j / (sigma_i^2 + sigma_j^2)
             * exp[-theta_ij^2 / (2 (sigma_i^2 + sigma_j^2))].

    This is the tangent-plane analytic equivalent of the HEALPix posterior
    cosine overlap used for real GWTC maps.
    """

    theta = np.asarray(theta_rad, dtype=np.float64)
    si = np.maximum(np.asarray(sigma_i_rad, dtype=np.float64), 1e-12)
    sj = np.maximum(np.asarray(sigma_j_rad, dtype=np.float64), 1e-12)
    var_sum = si * si + sj * sj
    width_term = np.log(np.maximum(2.0 * si * sj / var_sum, eps))
    sep_term = -theta * theta / (2.0 * var_sum)
    return (width_term + sep_term).astype(np.float32)


def gaussian_log_cosine_overlap_pairs(
    ra_i: np.ndarray,
    dec_i: np.ndarray,
    sigma_i_rad: np.ndarray,
    ra_j: np.ndarray,
    dec_j: np.ndarray,
    sigma_j_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = angular_separation_rad(ra_i, dec_i, ra_j, dec_j)
    sigma_pair = np.sqrt(np.asarray(sigma_i_rad, dtype=np.float64) ** 2 + np.asarray(sigma_j_rad, dtype=np.float64) ** 2)
    norm_sep = theta / np.maximum(sigma_pair, 1e-12)
    log_cos = gaussian_log_cosine_overlap(theta, sigma_i_rad, sigma_j_rad)
    return log_cos, theta.astype(np.float32), norm_sep.astype(np.float32)


def gaussian_log_cosine_overlap_matrix(
    sky_obs: pd.DataFrame,
    chunk_rows: int = 128,
    diagonal: float = 0.0,
) -> np.ndarray:
    required = {"ra_obs", "dec_obs", "sky_sigma_rad"}
    missing = sorted(required.difference(sky_obs.columns))
    if missing:
        raise KeyError(f"missing observed-sky columns: {missing}")

    ra = sky_obs["ra_obs"].to_numpy(dtype=np.float64)
    dec = sky_obs["dec_obs"].to_numpy(dtype=np.float64)
    sigma = np.maximum(sky_obs["sky_sigma_rad"].to_numpy(dtype=np.float64), 1e-12)
    n = len(sky_obs)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        rows = slice(start, stop)
        theta = angular_separation_rad(ra[rows, None], dec[rows, None], ra[None, :], dec[None, :])
        out[rows] = gaussian_log_cosine_overlap(theta, sigma[rows, None], sigma[None, :])
    np.fill_diagonal(out, float(diagonal))
    return out


def a90_to_sigma_rad(a90_deg2: np.ndarray) -> np.ndarray:
    a90_rad2 = np.asarray(a90_deg2, dtype=np.float64) * (math.pi / 180.0) ** 2
    return np.sqrt(a90_rad2 / (2.0 * math.pi * math.log(10.0))).astype(np.float64)
