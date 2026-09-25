#!/usr/bin/env python3
"""Build response-derived ET sky posteriors for unified-sky v8.1.

Run this script with the isolated ``.venv_sky_v8`` interpreter.  It uses the
public GWFAST implementation, the ET-D ASD, the triangular ET response, Earth
rotation, and the same standard angular priors shown in the GWFAST tutorial.

Unlike v8, v8.1 preserves the full two-dimensional local angular covariance.
It also assigns event-level weights to the eight short-signal ET-triangle
symmetry modes from a noise-perturbed, compressed ET1/ET2/ET3 response
likelihood.  These quantities are used to construct a reproducible posterior
surrogate downstream.  They are not full Bayesian parameter estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from astropy.cosmology import Planck15


# GWFAST 1.1.2 still imports the pre-SciPy-1.14 aliases.
if not hasattr(integrate, "cumtrapz"):
    integrate.cumtrapz = integrate.cumulative_trapezoid
if not hasattr(integrate, "simps"):
    integrate.simps = integrate.simpson

from gwfast import fisherTools, gwfastGlobals, gwfastUtils, signal, waveforms


REPO = Path(__file__).resolve().parents[2]
V7_ET = REPO / "results/et3_v7_aligned_20260723/formal"
OUTPUT_ROOT = REPO / "results/unified_sky_v81_20260725"
ET_SOURCE_ROOT = Path(
    "/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root"
)
SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
FULL_SKY_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2
FAMILY_SOURCE_DIR = {
    "SIS": "SIS_data_0222",
    "PM": "PM_data_0222",
    "unlensed": "Unlensed_data_0222",
}
ET_LATITUDE_RAD = math.radians(40.516666666666666)
ET_LONGITUDE_RAD = math.radians(9.416666666666666)
MODE_RESPONSE_FREQUENCIES = 12
MODE_RESPONSE_SEED_SALT = "unified-sky-v81-et-response-20260725"
CHI2_90_2D = -2.0 * math.log(0.1)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stable_seed(event_key: str, purpose: str) -> int:
    digest = hashlib.blake2b(
        f"{MODE_RESPONSE_SEED_SALT}:{purpose}:{event_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        )
    )


def _ra_dec(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = vectors / np.maximum(
        np.linalg.norm(vectors, axis=-1, keepdims=True),
        1e-15,
    )
    return (
        np.mod(np.arctan2(unit[..., 1], unit[..., 0]), 2.0 * math.pi),
        np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)),
    )


def _horizontal_coordinates(
    ra: np.ndarray,
    dec: np.ndarray,
    local_sidereal_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hour_angle = np.asarray(local_sidereal_time) - np.asarray(ra)
    sin_altitude = (
        np.sin(dec) * math.sin(ET_LATITUDE_RAD)
        + np.cos(dec) * math.cos(ET_LATITUDE_RAD) * np.cos(hour_angle)
    )
    altitude = np.arcsin(np.clip(sin_altitude, -1.0, 1.0))
    cos_altitude = np.maximum(np.cos(altitude), 1e-12)
    sin_azimuth = -np.sin(hour_angle) * np.cos(dec) / cos_altitude
    cos_azimuth = (
        np.sin(dec) - np.sin(altitude) * math.sin(ET_LATITUDE_RAD)
    ) / (cos_altitude * math.cos(ET_LATITUDE_RAD))
    return altitude, np.mod(
        np.arctan2(sin_azimuth, cos_azimuth),
        2.0 * math.pi,
    )


def _equatorial_coordinates(
    altitude: np.ndarray,
    azimuth: np.ndarray,
    local_sidereal_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sin_dec = (
        np.sin(altitude) * math.sin(ET_LATITUDE_RAD)
        + np.cos(altitude) * math.cos(ET_LATITUDE_RAD) * np.cos(azimuth)
    )
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    cos_dec = np.maximum(np.cos(dec), 1e-12)
    sin_hour_angle = -np.sin(azimuth) * np.cos(altitude) / cos_dec
    cos_hour_angle = (
        np.sin(altitude) - math.sin(ET_LATITUDE_RAD) * np.sin(dec)
    ) / (math.cos(ET_LATITUDE_RAD) * cos_dec)
    hour_angle = np.arctan2(sin_hour_angle, cos_hour_angle)
    return (
        np.mod(np.asarray(local_sidereal_time) - hour_angle, 2.0 * math.pi),
        dec,
    )


def _eight_mode_centers(
    ra: np.ndarray,
    dec: np.ndarray,
    local_sidereal_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    altitude, azimuth = _horizontal_coordinates(
        ra,
        dec,
        local_sidereal_time,
    )
    mode_altitude = np.column_stack(
        (
            altitude,
            -altitude,
            -altitude,
            altitude,
            altitude,
            altitude,
            -altitude,
            -altitude,
        )
    )
    mode_azimuth = np.column_stack(
        (
            azimuth,
            azimuth,
            azimuth + math.pi,
            azimuth + math.pi / 2.0,
            azimuth + math.pi,
            azimuth - math.pi / 2.0,
            azimuth + math.pi / 2.0,
            azimuth - math.pi / 2.0,
        )
    )
    repeated_lst = np.repeat(
        np.asarray(local_sidereal_time)[:, None],
        8,
        axis=1,
    )
    return _equatorial_coordinates(
        mode_altitude,
        np.mod(mode_azimuth, 2.0 * math.pi),
        repeated_lst,
    )


def _regularize_tangent_covariance(
    covariance: np.ndarray,
    theta: float,
    area90_deg2: float,
) -> tuple[np.ndarray, bool]:
    """Return a finite east/north covariance with the reported Fisher area.

    The determinant is set by the 90% Fisher area.  The Fisher axis ratio and
    orientation are retained when finite, with only a numerical cap that
    prevents a local tangent ellipse from spanning more than a hemisphere.
    """
    sin_theta = max(abs(math.sin(theta)), 1e-8)
    angular = np.asarray(
        [
            [
                sin_theta**2 * covariance[1, 1],
                -sin_theta * covariance[1, 0],
            ],
            [
                -sin_theta * covariance[0, 1],
                covariance[0, 0],
            ],
        ],
        dtype=np.float64,
    )
    fallback = not np.isfinite(angular).all()
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (angular + angular.T)
        )
    except np.linalg.LinAlgError:
        eigenvalues = np.ones(2, dtype=np.float64)
        eigenvectors = np.eye(2, dtype=np.float64)
        fallback = True
    if np.any(eigenvalues <= 0.0) or not np.isfinite(eigenvalues).all():
        eigenvalues = np.ones(2, dtype=np.float64)
        eigenvectors = np.eye(2, dtype=np.float64)
        fallback = True

    area_sr = np.clip(
        area90_deg2 * (math.pi / 180.0) ** 2,
        1e-10,
        4.0 * math.pi,
    )
    determinant_root = area_sr / (math.pi * CHI2_90_2D)
    axis_ratio = float(
        np.clip(eigenvalues.max() / max(eigenvalues.min(), 1e-30), 1.0, 1e8)
    )
    # Keep the major-axis sigma within pi radians; the posterior generator
    # uses a spherical log map, so a larger local sigma has no interpretation.
    maximum_ratio = max((math.pi**2 / determinant_root) ** 2, 1.0)
    axis_ratio = min(axis_ratio, maximum_ratio)
    root_ratio = math.sqrt(axis_ratio)
    regularized_values = np.asarray(
        [
            determinant_root / root_ratio,
            determinant_root * root_ratio,
        ]
    )
    regularized = eigenvectors @ np.diag(regularized_values) @ eigenvectors.T
    return 0.5 * (regularized + regularized.T), fallback


def _sample_observed_center(
    event_key: str,
    true_ra: float,
    true_dec: float,
    covariance_en: np.ndarray,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(_stable_seed(event_key, "local-center"))
    offset = rng.multivariate_normal(np.zeros(2), covariance_en)
    radius = float(np.linalg.norm(offset))
    if radius > math.pi:
        offset *= math.pi / radius
        radius = math.pi
    center = _unit_vectors(
        np.asarray([true_ra]),
        np.asarray([true_dec]),
    )[0]
    east = np.asarray([-math.sin(true_ra), math.cos(true_ra), 0.0])
    north = np.asarray(
        [
            -math.sin(true_dec) * math.cos(true_ra),
            -math.sin(true_dec) * math.sin(true_ra),
            math.cos(true_dec),
        ]
    )
    if radius <= 1e-15:
        observed = center
    else:
        direction = (offset[0] * east + offset[1] * north) / radius
        observed = math.cos(radius) * center + math.sin(radius) * direction
    ra, dec = _ra_dec(observed[None, :])
    return float(ra[0]), float(dec[0]), float(offset[0]), float(offset[1])


def _event_dictionary(
    parameters: dict[str, np.ndarray],
    index: int,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([np.asarray(value).reshape(-1)[index]], dtype=np.float64)
        for name, value in parameters.items()
    }


def _event_mode_weights(
    detector: signal.GWSignal,
    waveform: waveforms.IMRPhenomD,
    event_key: str,
    parameters: dict[str, np.ndarray],
    index: int,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compressed response likelihood for the eight ET symmetry modes.

    The statistic is the coherent projection energy after maximizing over two
    complex polarization amplitudes.  It uses only ET1/ET2/ET3 responses,
    Earth rotation, the event's observed network SNR, and independent Gaussian
    measurement noise.  No companion label or retrieval rank enters here.
    """
    event = _event_dictionary(parameters, index)
    try:
        f_cut = float(np.asarray(waveform.fcut(**event)).reshape(-1)[0])
    except Exception:
        f_cut = 256.0
    f_high = float(np.clip(0.98 * f_cut, 6.0, 512.0))
    frequencies = np.geomspace(5.0, f_high, MODE_RESPONSE_FREQUENCIES)
    frequency_width = np.gradient(frequencies)
    psd = np.interp(
        frequencies,
        np.asarray(detector.strainFreq, dtype=np.float64),
        np.asarray(detector.noiseCurve, dtype=np.float64),
    )
    frequency_weight = (
        frequencies ** (-7.0 / 6.0)
        * np.sqrt(np.maximum(frequency_width, 1e-12))
        / np.sqrt(np.maximum(psd, 1e-60))
    )
    frequency_weight /= max(np.linalg.norm(frequency_weight), 1e-15)

    true_theta = float(event["theta"][0])
    true_phi = float(event["phi"][0])
    true_iota = float(event["iota"][0])
    true_psi = float(event["psi"][0])
    t_coal = float(event["tcoal"][0])
    try:
        tau = np.asarray(
            waveform.tau_star(frequencies, **event),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        tau = np.zeros_like(frequencies)
    tau = np.maximum(tau, 0.0)
    time_base = t_coal - tau / 86400.0

    local_sidereal_time = np.asarray(
        [2.0 * math.pi * t_coal + ET_LONGITUDE_RAD]
    )
    mode_ra, mode_dec = _eight_mode_centers(
        np.asarray([true_phi]),
        np.asarray([math.pi / 2.0 - true_theta]),
        local_sidereal_time,
    )

    rotations = (0.0, 60.0, 120.0)
    true_rows = []
    for rotation in rotations:
        reference_delay = float(
            np.asarray(
                detector._DeltLoc(true_theta, true_phi, t_coal),
                dtype=np.float64,
            )
        )
        location_delay = np.asarray(
            detector._DeltLoc(true_theta, true_phi, time_base),
            dtype=np.float64,
        )
        response_time = time_base + location_delay / 86400.0
        f_plus, f_cross = detector._PatternFunction(
            true_theta,
            true_phi,
            response_time,
            true_psi,
            rot=rotation,
        )
        # A constant geocentre-to-site delay is degenerate with coalescence
        # time and must not be used to break a single-site sky symmetry.  Only
        # the frequency-dependent change caused by Earth rotation remains.
        phase = np.exp(
            1j
            * 2.0
            * math.pi
            * frequencies
            * (location_delay - reference_delay)
        )
        response = (
            0.5 * (1.0 + math.cos(true_iota) ** 2)
            * np.asarray(f_plus)
            + 1j * math.cos(true_iota) * np.asarray(f_cross)
        ) * phase
        true_rows.append(frequency_weight * response)
    true_signal = np.stack(true_rows, axis=1).reshape(-1)
    norm = float(np.linalg.norm(true_signal))
    if not np.isfinite(norm) or norm <= 1e-12:
        return (
            np.full(8, 1.0 / 8.0),
            np.zeros(8, dtype=np.float64),
            True,
        )
    true_signal *= float(max(target_snr, 1.0)) / norm
    rng = np.random.default_rng(_stable_seed(event_key, "mode-response"))
    noise = (
        rng.normal(size=true_signal.shape)
        + 1j * rng.normal(size=true_signal.shape)
    ) / math.sqrt(2.0)
    observed = true_signal + noise

    log_likelihood = np.empty(8, dtype=np.float64)
    failed = False
    for mode_index in range(8):
        theta = math.pi / 2.0 - float(mode_dec[0, mode_index])
        phi = float(mode_ra[0, mode_index])
        design_rows = []
        for rotation in rotations:
            reference_delay = float(
                np.asarray(
                    detector._DeltLoc(theta, phi, t_coal),
                    dtype=np.float64,
                )
            )
            location_delay = np.asarray(
                detector._DeltLoc(theta, phi, time_base),
                dtype=np.float64,
            )
            response_time = time_base + location_delay / 86400.0
            f_plus, f_cross = detector._PatternFunction(
                theta,
                phi,
                response_time,
                0.0,
                rot=rotation,
            )
            phase = np.exp(
                1j
                * 2.0
                * math.pi
                * frequencies
                * (location_delay - reference_delay)
            )
            design_rows.append(
                np.column_stack(
                    (
                        frequency_weight * np.asarray(f_plus) * phase,
                        1j
                        * frequency_weight
                        * np.asarray(f_cross)
                        * phase,
                    )
                )
            )
        design = np.stack(design_rows, axis=1).reshape(-1, 2)
        gram = design.conj().T @ design
        ridge = max(float(np.trace(gram).real), 1.0) * 1e-10
        gram += ridge * np.eye(2)
        projection = design.conj().T @ observed
        try:
            energy = float(
                np.real(projection.conj().T @ np.linalg.solve(gram, projection))
            )
        except np.linalg.LinAlgError:
            energy = 0.0
            failed = True
        log_likelihood[mode_index] = 0.5 * max(energy, 0.0)

    shifted = log_likelihood - np.max(log_likelihood)
    weights = np.exp(np.clip(shifted, -80.0, 0.0))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights[:] = 1.0 / 8.0
        failed = True
    else:
        weights /= total
    return weights, log_likelihood - np.max(log_likelihood), failed


def load_union() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        seed_dir = V7_ET / f"seed_{seed}"
        for split in ("validation", "test"):
            frame = pd.read_parquet(seed_dir / f"{split}_event_catalog.parquet")
            frame["source_seed"] = seed
            frame["source_split"] = split
            rows.append(frame)
    union = pd.concat(rows, ignore_index=True)
    union["event_key"] = (
        union["family"].astype(str)
        + ":"
        + union["source_index"].astype(str)
        + ":"
        + union["image"].astype(str)
    )
    checks = (
        union.groupby("event_key", sort=False)
        .agg(
            gps_nunique=("gps", "nunique"),
            snr_nunique=("snr", "nunique"),
            ra_nunique=("ra", "nunique"),
            dec_nunique=("dec", "nunique"),
        )
        .reset_index()
    )
    if (checks.iloc[:, 1:] > 1).any().any():
        raise RuntimeError("An ET event_key maps to inconsistent physical values")
    return union.drop_duplicates("event_key", keep="first").reset_index(drop=True)


def load_source_tables() -> dict[str, pd.DataFrame]:
    return {
        family: pd.read_csv(ET_SOURCE_ROOT / directory / "source_samples.csv")
        for family, directory in FAMILY_SOURCE_DIR.items()
    }


def source_columns(
    frame: pd.DataFrame,
    source_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "luminosity_distance",
        "theta_jn",
        "psi",
        "phase",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
    ):
        output[column] = np.nan
    output["redshift_planck15_reconstructed"] = np.nan
    output["detector_logmc_reconstructed"] = np.nan
    output["logitq_reconstructed"] = np.nan
    redshift_grid = np.linspace(0.0, 5.0, 250_001)
    luminosity_distance_grid = Planck15.luminosity_distance(redshift_grid).value
    for family, table in source_tables.items():
        keep = output["family"].astype(str) == family
        indices = output.loc[keep, "source_index"].to_numpy(dtype=np.int64)
        selected = table.iloc[indices]
        for column in (
            "luminosity_distance",
            "theta_jn",
            "psi",
            "phase",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
        ):
            output.loc[keep, column] = selected[column].to_numpy(dtype=np.float64)
        distance = selected["luminosity_distance"].to_numpy(dtype=np.float64)
        redshift = np.interp(
            distance,
            luminosity_distance_grid,
            redshift_grid,
        )
        mass_1 = selected["mass_1_source"].to_numpy(dtype=np.float64)
        mass_2 = selected["mass_2_source"].to_numpy(dtype=np.float64)
        mass_high = np.maximum(mass_1, mass_2)
        mass_low = np.minimum(mass_1, mass_2)
        chirp_mass_source = (
            (mass_high * mass_low) ** (3.0 / 5.0)
            / (mass_high + mass_low) ** (1.0 / 5.0)
        )
        detector_logmc = np.log(chirp_mass_source * (1.0 + redshift))
        # Match detector_frame_targets() in the frozen ET-v7 data loader.
        q = np.clip(mass_low / mass_high, 1e-4, 1.0 - 1e-4)
        logit_q = np.log(q / (1.0 - q))
        output.loc[keep, "redshift_planck15_reconstructed"] = redshift
        output.loc[keep, "detector_logmc_reconstructed"] = detector_logmc
        output.loc[keep, "logitq_reconstructed"] = logit_q
    if output[
        [
            "luminosity_distance",
            "theta_jn",
            "psi",
            "phase",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
        ]
    ].isna().any().any():
        raise RuntimeError("Missing source parameters for ET Fisher calculation")
    existing = output["target_logmc"].notna()
    if existing.any():
        max_logmc_difference = float(
            np.max(
                np.abs(
                    output.loc[existing, "target_logmc"]
                    - output.loc[existing, "detector_logmc_reconstructed"]
                )
            )
        )
        max_logitq_difference = float(
            np.max(
                np.abs(
                    output.loc[existing, "target_logitq"]
                    - output.loc[existing, "logitq_reconstructed"]
                )
            )
        )
        if max_logmc_difference > 1e-4 or max_logitq_difference > 1e-4:
            raise RuntimeError(
                "Reconstructed source masses disagree with lensed event labels: "
                f"dlogMc={max_logmc_difference}, dlogitq={max_logitq_difference}"
            )
    missing = output["target_logmc"].isna() | output["target_logitq"].isna()
    output.loc[missing, "target_logmc"] = output.loc[
        missing, "detector_logmc_reconstructed"
    ]
    output.loc[missing, "target_logitq"] = output.loc[
        missing, "logitq_reconstructed"
    ]
    output["mass_labels_reconstructed_from_source_table"] = missing
    return output


def event_parameters(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    logit_q = frame["target_logitq"].to_numpy(dtype=np.float64)
    q = 1.0 / (1.0 + np.exp(-np.clip(logit_q, -40, 40)))
    return {
        "Mc": np.exp(frame["target_logmc"].to_numpy(dtype=np.float64)),
        "eta": q / (1.0 + q) ** 2,
        "dL": frame["luminosity_distance"].to_numpy(dtype=np.float64) / 1000.0,
        "theta": math.pi / 2.0 - frame["dec"].to_numpy(dtype=np.float64),
        "phi": frame["ra"].to_numpy(dtype=np.float64),
        "iota": frame["theta_jn"].to_numpy(dtype=np.float64),
        "psi": frame["psi"].to_numpy(dtype=np.float64),
        "tcoal": np.asarray(
            gwfastUtils.GPSt_to_LMST(
                frame["gps"].to_numpy(dtype=np.float64),
                lat=0.0,
                long=0.0,
            )
        ),
        "Phicoal": frame["phase"].to_numpy(dtype=np.float64),
        "chi1z": frame["a_1"].to_numpy(dtype=np.float64)
        * np.cos(frame["tilt_1"].to_numpy(dtype=np.float64)),
        "chi2z": frame["a_2"].to_numpy(dtype=np.float64)
        * np.cos(frame["tilt_2"].to_numpy(dtype=np.float64)),
    }


def inspiral_duration_from_5hz(mc_detector: np.ndarray) -> np.ndarray:
    """Leading-order time from 5 Hz to merger, used only as an audit column."""
    solar_mass_seconds = 4.925490947e-6
    mc_seconds = np.maximum(mc_detector * solar_mass_seconds, 1e-12)
    return (
        5.0
        / 256.0
        * mc_seconds ** (-5.0 / 3.0)
        * (math.pi * 5.0) ** (-8.0 / 3.0)
    )


def build_signal() -> tuple[signal.GWSignal, waveforms.IMRPhenomD]:
    waveform = waveforms.IMRPhenomD()
    detector = signal.GWSignal(
        waveform,
        psd_path=str(Path(gwfastGlobals.detPath) / "ET-0000A-18.txt"),
        detector_shape="T",
        det_lat=40.516666666666666,
        det_long=9.416666666666666,
        det_xax=0.0,
        verbose=False,
        is_ASD=True,
        useEarthMotion=True,
        fmin=5.0,
        compute2arms=False,
    )
    return detector, waveform


def run_chunk(
    detector: signal.GWSignal,
    waveform: waveforms.IMRPhenomD,
    frame: pd.DataFrame,
    resolution: int,
) -> pd.DataFrame:
    parameters = event_parameters(frame)
    model_snr = np.asarray(detector.SNRInteg(parameters, res=resolution), dtype=np.float64)
    fisher = np.asarray(
        detector.FisherMatr(parameters, res=resolution),
        dtype=np.longdouble,
    )
    target_snr = frame["snr"].to_numpy(dtype=np.float64)
    scale = (target_snr / np.maximum(model_snr, 1e-12)) ** 2
    fisher *= scale[None, None, :]

    fixed, parameter_numbers = fisherTools.fixParams(
        fisher,
        waveform.ParNums,
        ["chi1z", "chi2z"],
    )
    angles = ["theta", "phi", "iota", "psi", "Phicoal"]
    # This is the same weak bounded-angle Gaussian prior demonstrated in the
    # official GWFAST tutorial, not a fitted constant from the present data.
    prior_precision = np.repeat(1.0 / (2.0 * math.pi**2), len(angles))
    with_prior = fisherTools.addPrior(
        fixed,
        prior_precision,
        parameter_numbers,
        angles,
    )
    covariance, inversion_error = fisherTools.CovMatr(
        with_prior,
        invMethodIn="svd",
        truncate=True,
        verbose=False,
    )
    area = fisherTools.compute_localization_region(
        covariance,
        parameter_numbers,
        parameters["theta"],
        perc_level=90,
        units="SqDeg",
    )
    theta_index = parameter_numbers["theta"]
    phi_index = parameter_numbers["phi"]
    capped_area = np.clip(
        np.asarray(area, dtype=np.float64),
        0.0,
        FULL_SKY_DEG2,
    )
    covariance_en = np.empty((len(frame), 2, 2), dtype=np.float64)
    covariance_fallback = np.zeros(len(frame), dtype=bool)
    observed_ra = np.empty(len(frame), dtype=np.float64)
    observed_dec = np.empty(len(frame), dtype=np.float64)
    observed_offset_east = np.empty(len(frame), dtype=np.float64)
    observed_offset_north = np.empty(len(frame), dtype=np.float64)
    mode_weights = np.empty((len(frame), 8), dtype=np.float64)
    mode_log_likelihood_relative = np.empty((len(frame), 8), dtype=np.float64)
    mode_response_failed = np.zeros(len(frame), dtype=bool)
    event_keys = frame["event_key"].astype(str).to_numpy()
    for index, event_key in enumerate(event_keys):
        angular_covariance = np.asarray(
            [
                [
                    covariance[theta_index, theta_index, index],
                    covariance[theta_index, phi_index, index],
                ],
                [
                    covariance[phi_index, theta_index, index],
                    covariance[phi_index, phi_index, index],
                ],
            ],
            dtype=np.float64,
        )
        covariance_en[index], covariance_fallback[index] = (
            _regularize_tangent_covariance(
                angular_covariance,
                float(parameters["theta"][index]),
                float(capped_area[index]),
            )
        )
        (
            observed_ra[index],
            observed_dec[index],
            observed_offset_east[index],
            observed_offset_north[index],
        ) = _sample_observed_center(
            event_key,
            float(parameters["phi"][index]),
            math.pi / 2.0 - float(parameters["theta"][index]),
            covariance_en[index],
        )
        (
            mode_weights[index],
            mode_log_likelihood_relative[index],
            mode_response_failed[index],
        ) = _event_mode_weights(
            detector,
            waveform,
            event_key,
            parameters,
            index,
            float(target_snr[index]),
        )
    condition_values = []
    for index in range(len(frame)):
        matrix = np.asarray(with_prior[:, :, index], dtype=np.float64)
        if not np.isfinite(matrix).all():
            condition_values.append(np.inf)
            continue
        try:
            condition_values.append(float(np.linalg.cond(matrix)))
        except np.linalg.LinAlgError:
            condition_values.append(np.inf)
    condition = np.asarray(condition_values, dtype=np.float64)
    output = frame[
        [
            "event_key",
            "family",
            "source_index",
            "system_id",
            "tag",
            "image",
            "snr",
            "gps",
            "ra",
            "dec",
            "target_logmc",
            "target_logitq",
            "redshift_planck15_reconstructed",
            "mass_labels_reconstructed_from_source_table",
        ]
    ].copy()
    output["gwfast_model_snr_before_catalog_rescale"] = model_snr
    output["fisher_snr_rescale"] = np.sqrt(scale)
    output["gmst_rad"] = np.mod(
        2.0 * math.pi * np.asarray(parameters["tcoal"], dtype=np.float64),
        2.0 * math.pi,
    )
    output["fisher_local_area90_deg2_raw"] = np.asarray(area, dtype=np.float64)
    output["fisher_local_area90_deg2_capped"] = capped_area
    output["fisher_cov_east_east_rad2"] = covariance_en[:, 0, 0]
    output["fisher_cov_east_north_rad2"] = covariance_en[:, 0, 1]
    output["fisher_cov_north_north_rad2"] = covariance_en[:, 1, 1]
    output["fisher_covariance_regularization_fallback"] = covariance_fallback
    output["observed_primary_ra_rad"] = observed_ra
    output["observed_primary_dec_rad"] = observed_dec
    output["observed_center_offset_east_rad"] = observed_offset_east
    output["observed_center_offset_north_rad"] = observed_offset_north
    for mode_index in range(8):
        output[f"et_mode_weight_{mode_index}"] = mode_weights[:, mode_index]
        output[f"et_mode_log_likelihood_relative_{mode_index}"] = (
            mode_log_likelihood_relative[:, mode_index]
        )
    output["et_mode_effective_count"] = 1.0 / np.maximum(
        np.sum(mode_weights**2, axis=1),
        1e-12,
    )
    output["et_mode_entropy_nats"] = -np.sum(
        mode_weights * np.log(np.maximum(mode_weights, 1e-30)),
        axis=1,
    )
    output["et_mode_response_failed"] = mode_response_failed
    output["fisher_inversion_error"] = np.asarray(inversion_error, dtype=np.float64)
    output["fisher_log10_condition"] = np.log10(np.maximum(condition, 1.0))
    output["inspiral_duration_from_5hz_s_audit"] = inspiral_duration_from_5hz(
        parameters["Mc"]
    )
    output["fisher_finite"] = np.isfinite(
        output[
            [
                "fisher_local_area90_deg2_raw",
                "fisher_inversion_error",
                "fisher_log10_condition",
                "fisher_cov_east_east_rad2",
                "fisher_cov_north_north_rad2",
            ]
        ]
    ).all(axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--frequency-resolution", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shared = args.output_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    canonical_path = shared / "et3_gwfast_sky_v81_events.parquet"
    if args.merge_shards:
        parts = [
            shared
            / f"et3_gwfast_sky_v81_part_{index:02d}_of_{args.num_shards:02d}.parquet"
            for index in range(args.num_shards)
        ]
        missing = [str(path) for path in parts if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Fisher shards: {missing}")
        merged = (
            pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
            .drop_duplicates("event_key")
            .sort_values("event_key")
            .reset_index(drop=True)
        )
        expected = len(load_union())
        if len(merged) != expected:
            raise RuntimeError(f"Merged {len(merged)} Fisher rows, expected {expected}")
        merged.to_parquet(canonical_path, index=False)
        write_json(
            shared / "et3_gwfast_sky_v81_merge_audit.json",
            {
                "num_shards": args.num_shards,
                "expected_events": expected,
                "merged_events": len(merged),
                "unique_event_keys": int(merged["event_key"].nunique()),
                "complete": True,
            },
        )
        finite = merged[merged["fisher_finite"]]
        write_json(
            shared / "et3_gwfast_sky_v81_audit.json",
            {
                "version": "unified_sky_v81",
                "event_count": int(len(merged)),
                "finite_event_count": int(len(finite)),
                "gwfast_version": "1.1.2",
                "waveform": "IMRPhenomD aligned-spin Fisher",
                "detector": "one triangular ET-D site with ET1/ET2/ET3 responses",
                "et_asd": "GWFAST psds/ET-0000A-18.txt",
                "fmin_hz": 5.0,
                "earth_motion": True,
                "fisher_resolution": int(args.frequency_resolution),
                "fixed_parameters": ["chi1z", "chi2z"],
                "angular_prior_precision": 1.0 / (2.0 * math.pi**2),
                "catalog_snr_rescaling": True,
                "mass_label_reconstruction": {
                    "cosmology": "Planck15",
                    "reason": (
                        "v7 event catalogs intentionally omit supervised "
                        "intrinsic targets for unlensed events"
                    ),
                    "reconstructed_event_count": int(
                        merged[
                            "mass_labels_reconstructed_from_source_table"
                        ].sum()
                    ),
                    "validated_against_existing_lensed_labels": True,
                },
                "area90_quantiles_deg2": {
                    str(quantile): float(
                        finite["fisher_local_area90_deg2_capped"].quantile(quantile)
                    )
                    for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                },
                "fraction_local_area_at_full_sky_cap": float(
                    np.mean(
                        finite["fisher_local_area90_deg2_raw"] >= FULL_SKY_DEG2
                    )
                ),
                "mode_effective_count_quantiles": {
                    str(quantile): float(
                        finite["et_mode_effective_count"].quantile(quantile)
                    )
                    for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
                },
                "mode_response_failed_count": int(
                    merged["et_mode_response_failed"].sum()
                ),
                "scope_warning": (
                    "The local covariance is Fisher-derived. Event-level mode "
                    "weights use a compressed ET response likelihood. The "
                    "result remains a posterior surrogate, not full PE."
                ),
            },
        )
        print(f"Merged {len(merged)} rows into {canonical_path}")
        return

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    output_path = (
        canonical_path
        if args.num_shards == 1
        else shared
        / (
            f"et3_gwfast_sky_v81_part_{args.shard_index:02d}"
            f"_of_{args.num_shards:02d}.parquet"
        )
    )
    audit_name = (
        "et3_gwfast_sky_v81_audit.json"
        if args.num_shards == 1
        else (
            f"et3_gwfast_sky_v81_audit_part_{args.shard_index:02d}"
            f"_of_{args.num_shards:02d}.json"
        )
    )
    audit_path = shared / audit_name
    if (
        output_path.exists()
        and audit_path.exists()
        and not args.force
        and args.limit is None
    ):
        print(f"Reusing {output_path}")
        return

    union = source_columns(load_union(), load_source_tables())
    if args.limit is not None:
        union = union.iloc[: args.limit].copy()
    if args.num_shards > 1:
        shard_indices = np.array_split(np.arange(len(union)), args.num_shards)[
            args.shard_index
        ]
        union = union.iloc[shard_indices].reset_index(drop=True)
    detector, waveform = build_signal()
    chunks = []
    resume_from = 0
    if output_path.exists() and not args.force and args.limit is None:
        existing = pd.read_parquet(output_path)
        resume_from = len(existing)
        if resume_from > len(union):
            raise RuntimeError(
                f"Partial output has {resume_from} rows for a {len(union)}-row shard"
            )
        chunks.append(existing)
        print(f"Resuming {output_path} from row {resume_from}", flush=True)
    started = time.time()
    for start in range(resume_from, len(union), args.chunk_size):
        stop = min(start + args.chunk_size, len(union))
        chunk_started = time.time()
        chunk = run_chunk(
            detector,
            waveform,
            union.iloc[start:stop].copy(),
            args.frequency_resolution,
        )
        chunks.append(chunk)
        partial = pd.concat(chunks, ignore_index=True)
        partial.to_parquet(output_path, index=False)
        print(
            f"ET Fisher {stop}/{len(union)} "
            f"chunk={time.time()-chunk_started:.1f}s total={time.time()-started:.1f}s",
            flush=True,
        )

    result = pd.concat(chunks, ignore_index=True)
    finite = result[result["fisher_finite"]]
    audit = {
        "version": "unified_sky_v81",
        "event_count": int(len(result)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "finite_event_count": int(len(finite)),
        "gwfast_version": "1.1.2",
        "waveform": "IMRPhenomD aligned-spin Fisher",
        "detector": "one triangular ET-D site with ET1/ET2/ET3 responses",
        "et_asd": "GWFAST psds/ET-0000A-18.txt",
        "fmin_hz": 5.0,
        "earth_motion": True,
        "fisher_resolution": int(args.frequency_resolution),
        "fixed_parameters": ["chi1z", "chi2z"],
        "angular_prior_precision": 1.0 / (2.0 * math.pi**2),
        "catalog_snr_rescaling": True,
        "area90_quantiles_deg2": {
            str(quantile): float(
                finite["fisher_local_area90_deg2_capped"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "fraction_local_area_at_full_sky_cap": float(
            np.mean(finite["fisher_local_area90_deg2_raw"] >= FULL_SKY_DEG2)
        ),
        "mode_effective_count_quantiles": {
            str(quantile): float(
                finite["et_mode_effective_count"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "mode_response_failed_count": int(
            result["et_mode_response_failed"].sum()
        ),
        "scope_warning": (
            "The local covariance is Fisher-derived. Event-level mode weights "
            "use a compressed ET response likelihood. The result remains a "
            "posterior surrogate, not full PE."
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
