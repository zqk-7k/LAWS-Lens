#!/usr/bin/env python3
"""ET response-derived, elliptical multimodal HEALPix posterior surrogate.

The full local two-dimensional angular covariance comes from GWFAST.  Global
mode locations implement the short-signal symmetries of a triangular,
co-located ET observatory in the horizontal frame.  Event-level mode weights
come from a compressed ET1/ET2/ET3 response likelihood generated upstream.

* altitude reflection;
* azimuth rotations by pi and +/-pi/2.

The output is explicitly a posterior surrogate.  It is not a replacement for
full Bayesian ET parameter estimation.
"""

from __future__ import annotations

import math
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
import torch

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


def _tangent_basis(
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = _unit_vectors(ra.reshape(-1), dec.reshape(-1)).reshape(
        *ra.shape,
        3,
    )
    east = np.stack(
        (
            -np.sin(ra),
            np.cos(ra),
            np.zeros_like(ra),
        ),
        axis=-1,
    )
    north = np.stack(
        (
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        ),
        axis=-1,
    )
    return center, east, north


def _exp_map(
    ra: np.ndarray,
    dec: np.ndarray,
    east_offset: float,
    north_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    center, east, north = _tangent_basis(ra, dec)
    radius = math.hypot(east_offset, north_offset)
    if radius <= 1e-15:
        return np.asarray(ra).copy(), np.asarray(dec).copy()
    direction = (
        east_offset * east + north_offset * north
    ) / radius
    displaced = math.cos(radius) * center + math.sin(radius) * direction
    return _ra_dec(displaced)


def _log_map_offsets(
    center_ra: np.ndarray,
    center_dec: np.ndarray,
    target_ra: np.ndarray,
    target_dec: np.ndarray,
) -> np.ndarray:
    center, east, north = _tangent_basis(center_ra, center_dec)
    target = _unit_vectors(target_ra.reshape(-1), target_dec.reshape(-1)).reshape(
        *target_ra.shape,
        3,
    )
    dot = np.clip(np.sum(center * target, axis=-1), -1.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sqrt(np.maximum(1.0 - dot**2, 1e-24))
    scale = angle / sine
    return np.stack(
        (
            scale * np.sum(target * east, axis=-1),
            scale * np.sum(target * north, axis=-1),
        ),
        axis=-1,
    )


def _transport_mode_covariances(
    primary_ra: np.ndarray,
    primary_dec: np.ndarray,
    local_sidereal_time: np.ndarray,
    covariance: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Transport the local ellipse through each spherical symmetry map."""
    base_ra, base_dec = eight_mode_centers(
        primary_ra,
        primary_dec,
        local_sidereal_time,
    )
    jacobian = np.empty((len(primary_ra), 8, 2, 2), dtype=np.float64)
    for dimension, displacement in enumerate(
        ((epsilon, 0.0), (0.0, epsilon))
    ):
        plus_ra, plus_dec = _exp_map(
            primary_ra,
            primary_dec,
            displacement[0],
            displacement[1],
        )
        minus_ra, minus_dec = _exp_map(
            primary_ra,
            primary_dec,
            -displacement[0],
            -displacement[1],
        )
        plus_mode_ra, plus_mode_dec = eight_mode_centers(
            plus_ra,
            plus_dec,
            local_sidereal_time,
        )
        minus_mode_ra, minus_mode_dec = eight_mode_centers(
            minus_ra,
            minus_dec,
            local_sidereal_time,
        )
        plus_offset = _log_map_offsets(
            base_ra,
            base_dec,
            plus_mode_ra,
            plus_mode_dec,
        )
        minus_offset = _log_map_offsets(
            base_ra,
            base_dec,
            minus_mode_ra,
            minus_mode_dec,
        )
        jacobian[..., dimension] = (
            plus_offset - minus_offset
        ) / (2.0 * epsilon)
    transported = np.einsum(
        "bmik,bkl,bmjl->bmij",
        jacobian,
        covariance,
        jacobian,
        optimize=True,
    )
    transported = 0.5 * (
        transported + np.swapaxes(transported, -1, -2)
    )
    for event_index in range(len(transported)):
        for mode_index in range(8):
            matrix = transported[event_index, mode_index]
            try:
                eigenvalue, eigenvector = np.linalg.eigh(matrix)
            except np.linalg.LinAlgError:
                eigenvalue = np.linalg.eigvalsh(covariance[event_index])
                eigenvector = np.eye(2)
            eigenvalue = np.clip(eigenvalue, 1e-10, math.pi**2)
            transported[event_index, mode_index] = (
                eigenvector @ np.diag(eigenvalue) @ eigenvector.T
            )
    return transported


def _render_elliptical_mixture(
    mode_ra: np.ndarray,
    mode_dec: np.ndarray,
    mode_covariance: np.ndarray,
    mode_weight: np.ndarray,
    *,
    nside: int,
    device: str | None,
    chunk_size: int,
) -> np.ndarray:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected_device)
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    pixel_vectors = _unit_vectors(phi, math.pi / 2.0 - theta)
    maps = np.empty(
        (len(mode_ra), hp.nside2npix(nside)),
        dtype=np.float32,
    )
    pixel_t = torch.as_tensor(
        pixel_vectors,
        dtype=torch.float32,
        device=torch_device,
    )
    for start in range(0, len(mode_ra), chunk_size):
        stop = min(start + chunk_size, len(mode_ra))
        center, east, north = _tangent_basis(
            mode_ra[start:stop],
            mode_dec[start:stop],
        )
        center_t = torch.as_tensor(center, dtype=torch.float32, device=torch_device)
        east_t = torch.as_tensor(east, dtype=torch.float32, device=torch_device)
        north_t = torch.as_tensor(north, dtype=torch.float32, device=torch_device)
        dot = torch.einsum("bmc,pc->bmp", center_t, pixel_t).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        sine = torch.sqrt(torch.clamp(1.0 - dot**2, min=1e-12))
        scale = angle / sine
        east_offset = scale * torch.einsum("bmc,pc->bmp", east_t, pixel_t)
        north_offset = scale * torch.einsum("bmc,pc->bmp", north_t, pixel_t)

        covariance_t = torch.as_tensor(
            mode_covariance[start:stop],
            dtype=torch.float32,
            device=torch_device,
        )
        inverse = torch.linalg.inv(covariance_t)
        determinant = torch.linalg.det(covariance_t).clamp_min(1e-20)
        quadratic = (
            inverse[..., 0, 0, None] * east_offset**2
            + 2.0
            * inverse[..., 0, 1, None]
            * east_offset
            * north_offset
            + inverse[..., 1, 1, None] * north_offset**2
        )
        weight_t = torch.as_tensor(
            np.maximum(mode_weight[start:stop], 1e-30),
            dtype=torch.float32,
            device=torch_device,
        )
        log_component = (
            torch.log(weight_t)[..., None]
            - 0.5 * torch.log(determinant)[..., None]
            - 0.5 * quadratic
        )
        log_density = torch.logsumexp(log_component, dim=1)
        log_density -= torch.amax(log_density, dim=1, keepdim=True)
        probability = torch.exp(log_density)
        probability /= probability.sum(dim=1, keepdim=True)
        maps[start:stop] = probability.cpu().numpy()
        del (
            center_t,
            east_t,
            north_t,
            dot,
            angle,
            sine,
            scale,
            east_offset,
            north_offset,
            covariance_t,
            inverse,
            determinant,
            quadratic,
            weight_t,
            log_component,
            log_density,
            probability,
        )
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return normalize_probability_maps(maps)


def generate_et_posterior_maps(
    events: pd.DataFrame,
    fisher: pd.DataFrame,
    *,
    seed: int,
    nside: int = 32,
    mode_count: int = 8,
    chunk_size: int = 32,
    device: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Generate deterministic, response-weighted ET posterior surrogates."""
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
    mode_weight_columns = [f"et_mode_weight_{index}" for index in range(8)]
    columns = [
        "event_key",
        "fisher_local_area90_deg2_raw",
        "fisher_local_area90_deg2_capped",
        "fisher_cov_east_east_rad2",
        "fisher_cov_east_north_rad2",
        "fisher_cov_north_north_rad2",
        "observed_primary_ra_rad",
        "observed_primary_dec_rad",
        "fisher_inversion_error",
        "fisher_log10_condition",
        "inspiral_duration_from_5hz_s_audit",
        "gmst_rad",
        "fisher_finite",
        "et_mode_effective_count",
        "et_mode_entropy_nats",
        "et_mode_response_failed",
        *mode_weight_columns,
    ]
    merged = frame.merge(
        fisher[columns],
        on="event_key",
        how="left",
        validate="many_to_one",
    )
    finite = merged["fisher_finite"].fillna(False).to_numpy(dtype=bool)
    if not finite.all():
        raise RuntimeError(
            f"Missing/non-finite Fisher rows for {(~finite).sum()} events"
        )
    local_covariance = np.empty((len(merged), 2, 2), dtype=np.float64)
    local_covariance[:, 0, 0] = merged[
        "fisher_cov_east_east_rad2"
    ].to_numpy(dtype=np.float64)
    local_covariance[:, 0, 1] = merged[
        "fisher_cov_east_north_rad2"
    ].to_numpy(dtype=np.float64)
    local_covariance[:, 1, 0] = local_covariance[:, 0, 1]
    local_covariance[:, 1, 1] = merged[
        "fisher_cov_north_north_rad2"
    ].to_numpy(dtype=np.float64)
    primary_ra = merged["observed_primary_ra_rad"].to_numpy(dtype=np.float64)
    primary_dec = merged["observed_primary_dec_rad"].to_numpy(dtype=np.float64)
    local_sidereal_time = np.mod(
        merged["gmst_rad"].to_numpy(dtype=np.float64) + ET_LONGITUDE_RAD,
        2.0 * math.pi,
    )
    all_ra, all_dec = eight_mode_centers(
        primary_ra,
        primary_dec,
        local_sidereal_time,
    )
    all_covariance = _transport_mode_covariances(
        primary_ra,
        primary_dec,
        local_sidereal_time,
        local_covariance,
    )
    all_weight = merged[mode_weight_columns].to_numpy(dtype=np.float64)
    if mode_count == 1:
        selected_modes = [0]
    elif mode_count == 4:
        selected_modes = [0, 3, 4, 5]
    else:
        selected_modes = list(range(8))
    all_ra = all_ra[:, selected_modes]
    all_dec = all_dec[:, selected_modes]
    all_covariance = all_covariance[:, selected_modes]
    all_weight = all_weight[:, selected_modes]
    all_weight /= np.maximum(all_weight.sum(axis=1, keepdims=True), 1e-30)
    maps = _render_elliptical_mixture(
        all_ra,
        all_dec,
        all_covariance,
        all_weight,
        nside=nside,
        device=device,
        chunk_size=chunk_size,
    )

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
    diagnostics["fisher_local_area90_deg2"] = merged[
        "fisher_local_area90_deg2_capped"
    ].to_numpy(dtype=np.float64)
    diagnostics["mode_count"] = mode_count
    diagnostics["mode_effective_count"] = merged[
        "et_mode_effective_count"
    ].to_numpy(dtype=np.float64)
    diagnostics["mode_entropy_nats"] = merged[
        "et_mode_entropy_nats"
    ].to_numpy(dtype=np.float64)
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
        "seed_argument_for_provenance_only": int(seed),
        "event_realization": (
            "deterministic per event_key; no lens label, test rank, or "
            "training seed is used"
        ),
        "n_events": int(len(merged)),
        "nside": int(nside),
        "mode_count": int(mode_count),
        "posterior_label": (
            "GWFAST 2D Fisher covariance plus response-weighted ET symmetry "
            "modes; approximate posterior surrogate"
        ),
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
        "mode_effective_count_quantiles": {
            str(quantile): float(
                diagnostics["mode_effective_count"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "scientific_scope": (
            "The local covariance and mode weights are response-derived. "
            "The eight global modes implement published ET-triangle "
            "symmetries. This remains an approximate posterior surrogate, "
            "not full Bayesian PE."
        ),
    }
    return maps, diagnostics, audit
