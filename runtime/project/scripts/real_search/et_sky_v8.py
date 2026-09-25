#!/usr/bin/env python3
"""ET response-derived, eight-mode HEALPix posterior surrogate.

The local width comes from GWFAST.  The global mode locations implement the
short-signal symmetries of a triangular, co-located ET observatory in the
horizontal frame:

* altitude reflection;
* azimuth rotations by pi and +/-pi/2.

This is more conservative than a single Gaussian posterior and is explicitly
labeled a surrogate.  It is not a replacement for full Bayesian ET parameter
estimation.
"""

from __future__ import annotations

import math
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from scripts.real_search.unified_sky_v8 import (
    FULL_SKY_DEG2,
    credible_levels,
    event_map_diagnostics,
    normalize_probability_maps,
)


ET_LATITUDE_RAD = math.radians(40.516666666666666)
ET_LONGITUDE_RAD = math.radians(9.416666666666666)


def equatorial_to_horizontal(
    ra: np.ndarray,
    dec: np.ndarray,
    local_sidereal_time: np.ndarray,
    latitude: float = ET_LATITUDE_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RA/Dec to altitude/azimuth (azimuth from North to East)."""
    hour_angle = np.asarray(local_sidereal_time) - np.asarray(ra)
    sin_altitude = (
        np.sin(dec) * math.sin(latitude)
        + np.cos(dec) * math.cos(latitude) * np.cos(hour_angle)
    )
    altitude = np.arcsin(np.clip(sin_altitude, -1.0, 1.0))
    cos_altitude = np.maximum(np.cos(altitude), 1e-12)
    sin_azimuth = -np.sin(hour_angle) * np.cos(dec) / cos_altitude
    cos_azimuth = (
        np.sin(dec) - np.sin(altitude) * math.sin(latitude)
    ) / (cos_altitude * math.cos(latitude))
    azimuth = np.mod(np.arctan2(sin_azimuth, cos_azimuth), 2.0 * math.pi)
    return altitude, azimuth


def horizontal_to_equatorial(
    altitude: np.ndarray,
    azimuth: np.ndarray,
    local_sidereal_time: np.ndarray,
    latitude: float = ET_LATITUDE_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert altitude/azimuth back to RA/Dec."""
    sin_dec = (
        np.sin(altitude) * math.sin(latitude)
        + np.cos(altitude) * math.cos(latitude) * np.cos(azimuth)
    )
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    cos_dec = np.maximum(np.cos(dec), 1e-12)
    sin_hour_angle = -np.sin(azimuth) * np.cos(altitude) / cos_dec
    cos_hour_angle = (
        np.sin(altitude) - math.sin(latitude) * np.sin(dec)
    ) / (math.cos(latitude) * cos_dec)
    hour_angle = np.arctan2(sin_hour_angle, cos_hour_angle)
    ra = np.mod(np.asarray(local_sidereal_time) - hour_angle, 2.0 * math.pi)
    return ra, dec


def _vmf_kappa_from_area90(area90_deg2: float) -> float:
    """Concentration of a spherical von Mises-Fisher 90% cap."""
    area = float(np.clip(area90_deg2, 1e-6, 0.9 * FULL_SKY_DEG2))
    area_sr = area * (math.pi / 180.0) ** 2
    a = area_sr / (2.0 * math.pi)
    if area >= 0.899999 * FULL_SKY_DEG2:
        return 0.0

    def cdf_minus_target(kappa: float) -> float:
        numerator = -math.expm1(-kappa * a)
        denominator = -math.expm1(-2.0 * kappa)
        return numerator / denominator - 0.9

    lower = 1e-8
    upper = max(10.0, 10.0 * math.log(10.0) / max(a, 1e-12))
    while cdf_minus_target(upper) < 0.0:
        upper *= 2.0
    return float(brentq(cdf_minus_target, lower, upper, maxiter=100))


def kappa_from_area90(area90_deg2: np.ndarray) -> np.ndarray:
    values = np.asarray(area90_deg2, dtype=np.float64)
    return np.asarray([_vmf_kappa_from_area90(value) for value in values])


def _unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    )


def _ra_dec(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    ra = np.mod(np.arctan2(unit[..., 1], unit[..., 0]), 2.0 * math.pi)
    dec = np.arcsin(np.clip(unit[..., 2], -1.0, 1.0))
    return ra, dec


def sample_vmf_centers(
    true_ra: np.ndarray,
    true_dec: np.ndarray,
    kappa: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample observed mode centers from an isotropic spherical likelihood."""
    mean = _unit_vectors(true_ra, true_dec)
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(mean), 1))
    near_pole = np.abs(mean[:, 2]) > 0.9
    reference[near_pole] = np.array([1.0, 0.0, 0.0])
    first = np.cross(reference, mean)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = np.cross(mean, first)

    uniform = rng.random(len(mean))
    cosine = np.empty(len(mean), dtype=np.float64)
    weak = kappa < 1e-6
    cosine[weak] = 2.0 * uniform[weak] - 1.0
    strong = ~weak
    # Stable inverse CDF for a 3-D von Mises-Fisher distribution.
    cosine[strong] = 1.0 + np.log(
        uniform[strong]
        + (1.0 - uniform[strong]) * np.exp(-2.0 * kappa[strong])
    ) / kappa[strong]
    cosine = np.clip(cosine, -1.0, 1.0)
    sine = np.sqrt(np.maximum(1.0 - cosine**2, 0.0))
    phase = rng.uniform(0.0, 2.0 * math.pi, len(mean))
    sampled = (
        cosine[:, None] * mean
        + sine[:, None]
        * (
            np.cos(phase)[:, None] * first
            + np.sin(phase)[:, None] * second
        )
    )
    return _ra_dec(sampled)


def eight_mode_centers(
    primary_ra: np.ndarray,
    primary_dec: np.ndarray,
    local_sidereal_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the published eight ET-triangle short-signal sky modes."""
    altitude, azimuth = equatorial_to_horizontal(
        primary_ra,
        primary_dec,
        local_sidereal_time,
    )
    mode_altitude = np.column_stack(
        [
            altitude,
            -altitude,
            -altitude,
            altitude,
            altitude,
            altitude,
            -altitude,
            -altitude,
        ]
    )
    mode_azimuth = np.column_stack(
        [
            azimuth,
            azimuth,
            azimuth + math.pi,
            azimuth + math.pi / 2.0,
            azimuth + math.pi,
            azimuth - math.pi / 2.0,
            azimuth + math.pi / 2.0,
            azimuth - math.pi / 2.0,
        ]
    )
    lst = np.repeat(local_sidereal_time[:, None], 8, axis=1)
    return horizontal_to_equatorial(
        mode_altitude,
        np.mod(mode_azimuth, 2.0 * math.pi),
        lst,
    )


def generate_et_posterior_maps(
    events: pd.DataFrame,
    fisher: pd.DataFrame,
    *,
    seed: int,
    nside: int = 32,
    mode_count: int = 8,
    chunk_size: int = 128,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Generate calibrated-noise, multimodal ET posterior surrogates."""
    if mode_count not in (1, 4, 8):
        raise ValueError("mode_count must be 1, 4, or 8")
    frame = events.copy()
    frame["event_key"] = (
        frame["family"].astype(str)
        + ":"
        + frame["source_index"].astype(str)
        + ":"
        + frame["image"].astype(str)
    )
    columns = [
        "event_key",
        "fisher_local_area90_deg2_raw",
        "fisher_local_area90_deg2_capped",
        "fisher_inversion_error",
        "fisher_log10_condition",
        "inspiral_duration_from_5hz_s_audit",
        "gmst_rad",
        "fisher_finite",
    ]
    merged = frame.merge(
        fisher[columns],
        on="event_key",
        how="left",
        validate="many_to_one",
    )
    finite = merged["fisher_finite"].fillna(False).to_numpy(dtype=bool)
    local_area = merged["fisher_local_area90_deg2_capped"].to_numpy(dtype=np.float64)
    local_area[~finite] = FULL_SKY_DEG2
    local_area = np.clip(local_area, 1e-4, FULL_SKY_DEG2)
    kappa = kappa_from_area90(local_area)
    rng = np.random.default_rng(seed)
    primary_ra, primary_dec = sample_vmf_centers(
        merged["ra"].to_numpy(dtype=np.float64),
        merged["dec"].to_numpy(dtype=np.float64),
        kappa,
        rng,
    )
    local_sidereal_time = np.mod(
        merged["gmst_rad"].to_numpy(dtype=np.float64) + ET_LONGITUDE_RAD,
        2.0 * math.pi,
    )
    all_ra, all_dec = eight_mode_centers(
        primary_ra,
        primary_dec,
        local_sidereal_time,
    )
    if mode_count == 1:
        all_ra = all_ra[:, :1]
        all_dec = all_dec[:, :1]
    elif mode_count == 4:
        # Precession can remove the altitude-reflection branch, leaving the
        # four azimuthal modes discussed by Santoliquido et al.
        all_ra = all_ra[:, [0, 3, 4, 5]]
        all_dec = all_dec[:, [0, 3, 4, 5]]

    centers = _unit_vectors(all_ra.reshape(-1), all_dec.reshape(-1)).reshape(
        len(merged), mode_count, 3
    )
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    pixels = _unit_vectors(phi, math.pi / 2.0 - theta).T
    maps = np.empty((len(merged), hp.nside2npix(nside)), dtype=np.float32)
    for start in range(0, len(merged), chunk_size):
        stop = min(start + chunk_size, len(merged))
        dot = np.einsum("bmc,cp->bmp", centers[start:stop], pixels, optimize=True)
        log_density = logsumexp(
            kappa[start:stop, None, None] * dot,
            axis=1,
        ) - math.log(mode_count)
        log_density -= np.max(log_density, axis=1, keepdims=True)
        probability = np.exp(log_density)
        maps[start:stop] = probability / probability.sum(axis=1, keepdims=True)
    maps = normalize_probability_maps(maps)

    diagnostics, levels, _ = event_map_diagnostics(
        maps,
        event_names=merged["event_key"].astype(str),
    )
    true_pixel = hp.ang2pix(
        nside,
        math.pi / 2.0 - merged["dec"].to_numpy(dtype=np.float64),
        merged["ra"].to_numpy(dtype=np.float64),
    )
    true_credible_level = levels[np.arange(len(merged)), true_pixel]
    diagnostics["true_sky_credible_level"] = true_credible_level
    diagnostics["true_sky_inside_90"] = true_credible_level <= 0.9
    diagnostics["fisher_local_area90_deg2"] = local_area
    diagnostics["vmf_kappa"] = kappa
    diagnostics["mode_count"] = mode_count
    diagnostics["snr"] = merged["snr"].to_numpy(dtype=np.float64)
    diagnostics["family"] = merged["family"].astype(str).to_numpy()
    diagnostics["source_index"] = merged["source_index"].to_numpy(dtype=np.int64)
    diagnostics["image"] = merged["image"].to_numpy(dtype=np.int8)
    diagnostics["gps"] = merged["gps"].to_numpy(dtype=np.float64)
    diagnostics["fisher_inversion_error"] = merged[
        "fisher_inversion_error"
    ].to_numpy(dtype=np.float64)
    diagnostics["fisher_log10_condition"] = merged[
        "fisher_log10_condition"
    ].to_numpy(dtype=np.float64)
    diagnostics["inspiral_duration_from_5hz_s_audit"] = merged[
        "inspiral_duration_from_5hz_s_audit"
    ].to_numpy(dtype=np.float64)
    audit = {
        "seed": int(seed),
        "n_events": int(len(merged)),
        "nside": int(nside),
        "mode_count": int(mode_count),
        "posterior_label": "GWFAST-local-width, ET-symmetry multimodal posterior surrogate",
        "fisher_missing_or_nonfinite": int((~finite).sum()),
        "frequentist_true_sky_90_coverage": float(
            diagnostics["true_sky_inside_90"].mean()
        ),
        "generated_area90_quantiles_deg2": {
            str(quantile): float(diagnostics["area90_deg2"].quantile(quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "local_fisher_area90_quantiles_deg2": {
            str(quantile): float(
                diagnostics["fisher_local_area90_deg2"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "scientific_scope": (
            "The local information is response-derived with GWFAST. The "
            "eight global modes implement published ET-triangle symmetries. "
            "This remains an approximate posterior surrogate, not full PE."
        ),
    }
    return maps, diagnostics, audit
