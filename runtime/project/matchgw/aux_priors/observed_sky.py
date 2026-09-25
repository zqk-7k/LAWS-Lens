from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

EPS = 1e-8
DEG2_TO_SR_EQUIV = (np.pi / 180.0) ** 2


@dataclass(frozen=True)
class DetectorSkyScenario:
    name: str
    label: str
    ifos: tuple[str, ...] | None
    n_ifos: int | None
    is_multisite: bool
    is_network: bool
    snr_for_sky: str
    a90_ref_deg2: float
    rho_ref: float
    clip_min_deg2: float
    clip_max_deg2: float
    lognormal_sigma: float
    sky_model: str = "detector_dependent_A90_approximation"
    uses_h1l1_timing: bool = False
    uses_antenna_pattern_localization: bool = False
    uses_healpix_skymap: bool = False


DETECTOR_SKY_SCENARIOS: dict[str, DetectorSkyScenario] = {
    "ET_SINGLE": DetectorSkyScenario(
        name="ET_SINGLE",
        label="ET-like single-interferometer A90 approximation",
        ifos=("ET",),
        n_ifos=1,
        is_multisite=False,
        is_network=False,
        snr_for_sky="network",
        a90_ref_deg2=300.0,
        rho_ref=12.0,
        clip_min_deg2=50.0,
        clip_max_deg2=2000.0,
        lognormal_sigma=0.35,
    ),
    "ET_TRIANGLE": DetectorSkyScenario(
        name="ET_TRIANGLE",
        label="ET three-arm network-SNR A90 approximation",
        ifos=("ET1", "ET2", "ET3"),
        n_ifos=3,
        is_multisite=False,
        is_network=True,
        snr_for_sky="network",
        a90_ref_deg2=100.0,
        rho_ref=12.0,
        clip_min_deg2=20.0,
        clip_max_deg2=1000.0,
        lognormal_sigma=0.35,
    ),
    "LIGO_HL": DetectorSkyScenario(
        name="LIGO_HL",
        label="LIGO H1+L1 network A90 approximation",
        ifos=("H1", "L1"),
        n_ifos=2,
        is_multisite=True,
        is_network=True,
        snr_for_sky="network",
        a90_ref_deg2=100.0,
        rho_ref=12.0,
        clip_min_deg2=10.0,
        clip_max_deg2=500.0,
        lognormal_sigma=0.35,
    ),
    "FIXED_A90_100": DetectorSkyScenario(
        name="FIXED_A90_100",
        label="Fixed A90=100 deg2 ablation",
        ifos=None,
        n_ifos=None,
        is_multisite=False,
        is_network=True,
        snr_for_sky="network",
        a90_ref_deg2=100.0,
        rho_ref=12.0,
        clip_min_deg2=10.0,
        clip_max_deg2=2000.0,
        lognormal_sigma=0.35,
        sky_model="fixed_A90_ablation",
    ),
    "FIXED_A90_300": DetectorSkyScenario(
        name="FIXED_A90_300",
        label="Fixed A90=300 deg2 ablation",
        ifos=None,
        n_ifos=None,
        is_multisite=False,
        is_network=True,
        snr_for_sky="network",
        a90_ref_deg2=300.0,
        rho_ref=12.0,
        clip_min_deg2=10.0,
        clip_max_deg2=2000.0,
        lognormal_sigma=0.35,
        sky_model="fixed_A90_ablation",
    ),
}

DETECTOR_TO_DEFAULT_SCENARIO = {
    "ET": "ET_SINGLE",
    "ET3": "ET_TRIANGLE",
    "LIGO": "LIGO_HL",
}


def scenario_for_detector(detector: str, scenario_name: str | None = None) -> DetectorSkyScenario:
    key = scenario_name or DETECTOR_TO_DEFAULT_SCENARIO.get(detector.upper(), detector.upper())
    if key not in DETECTOR_SKY_SCENARIOS:
        raise KeyError(f"unknown detector sky scenario: {key}")
    return DETECTOR_SKY_SCENARIOS[key]


def with_a90_ref(scenario: DetectorSkyScenario, a90_ref_deg2: float | None) -> DetectorSkyScenario:
    if a90_ref_deg2 is None:
        return scenario
    label = f"{scenario.label} sweep A90={float(a90_ref_deg2):g} deg2"
    return replace(scenario, a90_ref_deg2=float(a90_ref_deg2), label=label)


def unit_from_radec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float64)


def radec_from_unit(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), EPS)
    ra = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2.0 * np.pi)
    dec = np.arcsin(np.clip(vec[:, 2], -1.0, 1.0))
    return ra.astype(np.float64), dec.astype(np.float64)


def tangent_basis(true_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref = np.tile(z_axis, (len(true_vec), 1))
    near_pole = np.abs(true_vec @ z_axis) > 0.95
    ref[near_pole] = x_axis
    e1 = np.cross(ref, true_vec)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), EPS)
    e2 = np.cross(true_vec, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), EPS)
    return e1, e2


def a90_to_sigma_rad(a90_deg2: np.ndarray) -> np.ndarray:
    a90_rad2 = np.asarray(a90_deg2, dtype=np.float64) * DEG2_TO_SR_EQUIV
    return np.sqrt(a90_rad2 / (2.0 * np.pi * np.log(10.0))).astype(np.float64)


def select_snr_for_sky(time_obs: pd.DataFrame, scenario: DetectorSkyScenario) -> np.ndarray:
    mode = scenario.snr_for_sky
    if mode in {"network", "network_snr"}:
        return time_obs["snr"].to_numpy(dtype=np.float64)
    if mode in {"max_single", "max_single_snr"}:
        cols = [c for c in time_obs.columns if c.startswith("snr_") and c != "snr"]
        if not cols:
            raise KeyError("single-detector SNR columns are unavailable for max_single_snr")
        return time_obs[cols].to_numpy(dtype=np.float64).max(axis=1)
    if mode in {"mean_single", "mean_single_snr"}:
        cols = [c for c in time_obs.columns if c.startswith("snr_") and c != "snr"]
        if not cols:
            raise KeyError("single-detector SNR columns are unavailable for mean_single_snr")
        return time_obs[cols].to_numpy(dtype=np.float64).mean(axis=1)
    raise ValueError(f"unsupported snr_for_sky: {mode}")


def compute_a90_from_snr(snr: np.ndarray, scenario: DetectorSkyScenario, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.lognormal(mean=0.0, sigma=scenario.lognormal_sigma, size=len(snr))
    a90 = scenario.a90_ref_deg2 * (scenario.rho_ref / np.maximum(snr, 1.0)) ** 2 * jitter
    return np.clip(a90, scenario.clip_min_deg2, scenario.clip_max_deg2).astype(np.float64)


def sample_observed_sky_center(
    ra_true: np.ndarray,
    dec_true: np.ndarray,
    sigma_rad: np.ndarray,
    rng: np.random.Generator,
    sampling: str = "tangent_2d_gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    true_vec = unit_from_radec(ra_true, dec_true)
    if sampling == "old_radial_normal":
        noise = rng.normal(size=true_vec.shape)
        noise -= np.sum(noise * true_vec, axis=1, keepdims=True) * true_vec
        noise /= np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), EPS)
        radial = rng.normal(loc=0.0, scale=sigma_rad)
        obs_vec = true_vec * np.cos(radial)[:, None] + noise * np.sin(radial)[:, None]
    elif sampling == "tangent_2d_gaussian":
        e1, e2 = tangent_basis(true_vec)
        dx = rng.normal(0.0, sigma_rad)
        dy = rng.normal(0.0, sigma_rad)
        offset = dx[:, None] * e1 + dy[:, None] * e2
        obs_vec = true_vec + offset
    else:
        raise ValueError(f"unsupported sky sampling mode: {sampling}")
    obs_vec /= np.maximum(np.linalg.norm(obs_vec, axis=1, keepdims=True), EPS)
    return radec_from_unit(obs_vec)


def build_observed_sky_table(
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    scenario: DetectorSkyScenario,
    rng: np.random.Generator,
    sampling: str = "tangent_2d_gaussian",
) -> pd.DataFrame:
    snr_for_sky = select_snr_for_sky(time_obs, scenario)
    a90 = compute_a90_from_snr(snr_for_sky, scenario, rng)
    sigma = a90_to_sigma_rad(a90)
    ra_true = raw_obs["ra"].to_numpy(dtype=np.float64)
    dec_true = raw_obs["dec"].to_numpy(dtype=np.float64)
    ra_obs, dec_obs = sample_observed_sky_center(ra_true, dec_true, sigma, rng, sampling=sampling)
    return pd.DataFrame({
        "event_id": np.arange(len(raw_obs), dtype=np.int64),
        "scenario": scenario.name,
        "sky_model": scenario.sky_model,
        "sky_sampling": sampling,
        "snr_for_sky_mode": scenario.snr_for_sky,
        "snr_for_sky": snr_for_sky,
        "a90_ref_deg2": float(scenario.a90_ref_deg2),
        "ra_true": ra_true,
        "dec_true": dec_true,
        "ra_obs": ra_obs,
        "dec_obs": dec_obs,
        "sky_area90_deg2": a90,
        "sky_sigma_rad": sigma,
        "uses_h1l1_timing": bool(scenario.uses_h1l1_timing),
        "uses_antenna_pattern_localization": bool(scenario.uses_antenna_pattern_localization),
        "uses_healpix_skymap": bool(scenario.uses_healpix_skymap),
    })


def public_observed_sky_features(sky_obs: pd.DataFrame) -> pd.DataFrame:
    allowed = ["ra_obs", "dec_obs", "sky_area90_deg2", "sky_sigma_rad"]
    return sky_obs[allowed].copy()
