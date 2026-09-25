#!/usr/bin/env python3
"""ET-3 moderate-information observed-sky surrogate experiment.

This experiment deliberately sits between two unsuitable extremes:

* an unrealistically narrow, effectively oracle single-Gaussian sky proxy;
* the v8.1 eight-nearly-equal-mode surrogate whose maps are almost all-sky.

Each event receives an independent, SNR-conditioned localization measurement.
Companion images share only the physical source direction.  They do not share
an observed center, localization area, ellipse, or noise realization.

The main pair score remains the same common-source statistic used for GWTC:

    B_sky = N_pix * sum_k P_i[k] P_j[k]
    sky_score = log(B_sky)

Waveform evidence, time evidence, catalog splits, and labels are frozen from
the ET-3 v7 experiment.  Fusion weights are selected on validation systems.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.unified_sky_v81 import write_json


BASE_PIPELINE = REPO / "scripts/experiments/120_et3_sky_v81_pipeline.py"
DEFAULT_OUTPUT = REPO / "results/et3_moderate_sky_gwtc3_topb_20260726"
PRIMARY_SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
NSIDE = 32
CHI2_2D_90 = 2.0 * math.log(10.0)
FULL_SKY_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


base = load_module(BASE_PIPELINE, "et3_v81_base_for_moderate_sky")
etv7 = base.etv7


@dataclass(frozen=True)
class SkyScenario:
    scenario_id: str
    a90_ref_deg2: float
    rho_ref: float
    lognormal_scatter: float
    clip_min_deg2: float
    clip_max_deg2: float
    axis_ratio_min: float = 1.0
    axis_ratio_max: float = 3.0
    secondary_antipodal_weight: float = 0.0


SCENARIOS = {
    "informative": SkyScenario(
        "informative",
        a90_ref_deg2=200.0,
        rho_ref=12.0,
        lognormal_scatter=0.6,
        clip_min_deg2=50.0,
        clip_max_deg2=2000.0,
    ),
    "moderate": SkyScenario(
        "moderate",
        a90_ref_deg2=400.0,
        rho_ref=12.0,
        lognormal_scatter=0.6,
        clip_min_deg2=100.0,
        clip_max_deg2=5000.0,
    ),
    "conservative": SkyScenario(
        "conservative",
        a90_ref_deg2=800.0,
        rho_ref=12.0,
        lognormal_scatter=0.6,
        clip_min_deg2=200.0,
        clip_max_deg2=10000.0,
    ),
    "moderate_secondary25": SkyScenario(
        "moderate_secondary25",
        a90_ref_deg2=400.0,
        rho_ref=12.0,
        lognormal_scatter=0.6,
        clip_min_deg2=100.0,
        clip_max_deg2=5000.0,
        secondary_antipodal_weight=0.25,
    ),
    "moderate_secondary50": SkyScenario(
        "moderate_secondary50",
        a90_ref_deg2=400.0,
        rho_ref=12.0,
        lognormal_scatter=0.6,
        clip_min_deg2=100.0,
        clip_max_deg2=5000.0,
        secondary_antipodal_weight=0.50,
    ),
}


def event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["family"].astype(str)
        + ":"
        + frame["source_index"].astype(str)
        + ":"
        + frame["image"].astype(str)
    )


def stable_rng(key: str, realization_seed: int, scenario_id: str) -> np.random.Generator:
    payload = f"{scenario_id}|{realization_seed}|{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little", signed=False))


def unit_from_radec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    )


def radec_from_unit(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-15)
    ra = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * math.pi)
    dec = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    return ra, dec


def tangent_basis(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(unit), 1))
    near_pole = np.abs(unit[:, 2]) > 0.95
    reference[near_pole] = np.array([1.0, 0.0, 0.0])
    east = np.cross(reference, unit)
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-15)
    north = np.cross(unit, east)
    north /= np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1e-15)
    return east, north


def exp_map(
    center: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
) -> np.ndarray:
    offset = offset_x[:, None] * basis_x + offset_y[:, None] * basis_y
    radius = np.linalg.norm(offset, axis=1)
    direction = offset / np.maximum(radius[:, None], 1e-15)
    output = (
        np.cos(radius)[:, None] * center
        + np.sin(radius)[:, None] * direction
    )
    zero = radius <= 1e-15
    output[zero] = center[zero]
    return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-15)


def build_event_parameters(
    events: pd.DataFrame,
    scenario: SkyScenario,
    realization_seed: int,
) -> pd.DataFrame:
    keys = event_key(events).astype(str).tolist()
    n_events = len(events)
    jitter = np.empty(n_events)
    axis_ratio = np.empty(n_events)
    orientation = np.empty(n_events)
    gaussian_x = np.empty(n_events)
    gaussian_y = np.empty(n_events)
    for index, key in enumerate(keys):
        rng = stable_rng(key, realization_seed, scenario.scenario_id)
        jitter[index] = rng.lognormal(0.0, scenario.lognormal_scatter)
        axis_ratio[index] = rng.uniform(
            scenario.axis_ratio_min,
            scenario.axis_ratio_max,
        )
        orientation[index] = rng.uniform(0.0, 2.0 * math.pi)
        gaussian_x[index], gaussian_y[index] = rng.normal(size=2)

    snr = events["snr"].to_numpy(dtype=np.float64)
    a90 = (
        scenario.a90_ref_deg2
        * (scenario.rho_ref / np.maximum(snr, 1.0)) ** 2
        * jitter
    )
    a90 = np.clip(a90, scenario.clip_min_deg2, scenario.clip_max_deg2)
    area_sr = a90 * (math.pi / 180.0) ** 2
    sigma_geo = np.sqrt(area_sr / (math.pi * CHI2_2D_90))
    sigma_major = sigma_geo * np.sqrt(axis_ratio)
    sigma_minor = sigma_geo / np.sqrt(axis_ratio)
    offset_major = gaussian_x * sigma_major
    offset_minor = gaussian_y * sigma_minor

    true_ra = events["ra"].to_numpy(dtype=np.float64)
    true_dec = events["dec"].to_numpy(dtype=np.float64)
    true_unit = unit_from_radec(true_ra, true_dec)
    east, north = tangent_basis(true_unit)
    cos_o = np.cos(orientation)[:, None]
    sin_o = np.sin(orientation)[:, None]
    major_basis = cos_o * east + sin_o * north
    minor_basis = -sin_o * east + cos_o * north
    observed_unit = exp_map(
        true_unit,
        major_basis,
        minor_basis,
        offset_major,
        offset_minor,
    )
    observed_ra, observed_dec = radec_from_unit(observed_unit)
    mahalanobis2 = gaussian_x**2 + gaussian_y**2

    return pd.DataFrame(
        {
            "event_index": np.arange(n_events, dtype=np.int32),
            "event_key": keys,
            "family": events["family"].astype(str).to_numpy(),
            "system_id": events["system_id"].astype(str).to_numpy(),
            "image": events["image"].astype(str).to_numpy(),
            "snr": snr,
            "ra_true": true_ra,
            "dec_true": true_dec,
            "ra_obs": observed_ra,
            "dec_obs": observed_dec,
            "a90_model_deg2": a90,
            "axis_ratio": axis_ratio,
            "ellipse_orientation_rad": orientation,
            "sigma_major_rad": sigma_major,
            "sigma_minor_rad": sigma_minor,
            "center_error_mahalanobis2": mahalanobis2,
            "inside_nominal_90": mahalanobis2 <= CHI2_2D_90,
            "secondary_antipodal_weight": scenario.secondary_antipodal_weight,
        }
    )


def validation_temperature(
    parameters: pd.DataFrame,
    *,
    calibration_events: int = 2048,
    iterations: int = 8,
) -> tuple[float, dict[str, Any]]:
    """Select one global temperature from validation-map HPD coverage.

    HEALPix discretization and an optional secondary mode make the continuous
    chi-square calibration slightly inaccurate.  We therefore freeze T using
    only a deterministic validation-event subset and the actual HEALPix maps.
    No pair labels or retrieval metrics enter this calibration.
    """
    sample_indices = np.unique(
        np.linspace(
            0,
            len(parameters) - 1,
            min(calibration_events, len(parameters)),
        ).astype(np.int32)
    )
    sample = parameters.iloc[sample_indices].reset_index(drop=True)
    true_pixel = hp.ang2pix(
        NSIDE,
        math.pi / 2.0 - sample["dec_true"].to_numpy(dtype=np.float64),
        sample["ra_true"].to_numpy(dtype=np.float64),
    )

    def coverage_at(temperature: float) -> float:
        maps = generate_probability_maps(sample, temperature)
        probability_at_truth = maps[np.arange(len(maps)), true_pixel]
        credible_mass = np.sum(
            maps * (maps >= probability_at_truth[:, None]),
            axis=1,
            dtype=np.float64,
        )
        return float(np.mean(credible_mass <= (0.9 + 1e-7)))

    low, high = 0.5, 2.5
    records = []
    candidates = []
    for _ in range(iterations):
        temperature = 0.5 * (low + high)
        coverage = coverage_at(temperature)
        records.append(
            {
                "temperature": float(temperature),
                "healpix_hpd90_coverage": float(coverage),
            }
        )
        candidates.append((abs(coverage - 0.9), temperature, coverage))
        if coverage < 0.9:
            low = temperature
        else:
            high = temperature
    _, temperature, coverage = min(candidates)
    return float(temperature), {
        "calibration_split": "deterministic validation-event subset only",
        "definition": "temperature selected by actual HEALPix HPD90 coverage",
        "n_calibration_events": int(len(sample)),
        "iterations": int(iterations),
        "temperature": float(temperature),
        "validation_subset_coverage_after_temperature": float(coverage),
        "calibration_trace": records,
        "test_labels_used": False,
        "pair_labels_used": False,
        "retrieval_metrics_used": False,
    }


def pixel_vectors(nside: int) -> np.ndarray:
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return np.column_stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    ).astype(np.float64)


def _elliptical_mode(
    center: np.ndarray,
    major_basis: np.ndarray,
    minor_basis: np.ndarray,
    sigma_major: np.ndarray,
    sigma_minor: np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    z = np.clip(center @ pixels.T, -1.0, 1.0)
    tangent_x = major_basis @ pixels.T
    tangent_y = minor_basis @ pixels.T
    angle = np.arccos(z)
    sine = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    scale = angle / np.maximum(sine, 1e-12)
    local_x = tangent_x * scale
    local_y = tangent_y * scale
    quadratic = (
        (local_x / sigma_major[:, None]) ** 2
        + (local_y / sigma_minor[:, None]) ** 2
    )
    log_probability = -0.5 * quadratic
    log_probability -= np.max(log_probability, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-300)
    return probability


def generate_probability_maps(
    parameters: pd.DataFrame,
    temperature: float,
    *,
    nside: int = NSIDE,
    chunk_size: int = 64,
) -> np.ndarray:
    pixels = pixel_vectors(nside)
    output = np.empty(
        (len(parameters), hp.nside2npix(nside)),
        dtype=np.float32,
    )
    observed = unit_from_radec(
        parameters["ra_obs"].to_numpy(dtype=np.float64),
        parameters["dec_obs"].to_numpy(dtype=np.float64),
    )
    east, north = tangent_basis(observed)
    orientation = parameters["ellipse_orientation_rad"].to_numpy(dtype=np.float64)
    cos_o = np.cos(orientation)[:, None]
    sin_o = np.sin(orientation)[:, None]
    major_basis = cos_o * east + sin_o * north
    minor_basis = -sin_o * east + cos_o * north
    sigma_major = (
        parameters["sigma_major_rad"].to_numpy(dtype=np.float64)
        * math.sqrt(temperature)
    )
    sigma_minor = (
        parameters["sigma_minor_rad"].to_numpy(dtype=np.float64)
        * math.sqrt(temperature)
    )
    secondary_weight = float(
        parameters["secondary_antipodal_weight"].iloc[0]
    )

    for start in range(0, len(parameters), chunk_size):
        stop = min(start + chunk_size, len(parameters))
        main = _elliptical_mode(
            observed[start:stop],
            major_basis[start:stop],
            minor_basis[start:stop],
            sigma_major[start:stop],
            sigma_minor[start:stop],
            pixels,
        )
        if secondary_weight > 0.0:
            secondary = _elliptical_mode(
                -observed[start:stop],
                major_basis[start:stop],
                -minor_basis[start:stop],
                sigma_major[start:stop],
                sigma_minor[start:stop],
                pixels,
            )
            main = (1.0 - secondary_weight) * main + secondary_weight * secondary
        main /= np.maximum(main.sum(axis=1, keepdims=True), 1e-300)
        output[start:stop] = main.astype(np.float32)
    return output


def map_diagnostics(
    maps: np.ndarray,
    parameters: pd.DataFrame,
    temperature: float,
    nside: int = NSIDE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability64 = maps.astype(np.float64)
    entropy = -np.sum(
        probability64 * np.log(np.maximum(probability64, 1e-300)),
        axis=1,
    )
    true_pixel = hp.ang2pix(
        nside,
        math.pi / 2.0 - parameters["dec_true"].to_numpy(dtype=np.float64),
        parameters["ra_true"].to_numpy(dtype=np.float64),
    )
    true_probability = maps[np.arange(len(maps)), true_pixel]
    true_credible_mass = np.sum(
        maps * (maps >= true_probability[:, None]),
        axis=1,
        dtype=np.float64,
    )
    diagnostics = parameters.copy()
    diagnostics["posterior_temperature"] = temperature
    diagnostics["a90_model_calibrated_deg2"] = (
        diagnostics["a90_model_deg2"] * temperature
    )
    diagnostics["map_sum"] = maps.sum(axis=1)
    diagnostics["map_max"] = maps.max(axis=1)
    diagnostics["entropy_nats"] = entropy
    diagnostics["effective_area_deg2"] = (
        np.exp(entropy) * FULL_SKY_DEG2 / maps.shape[1]
    )
    diagnostics["true_sky_credible_mass"] = true_credible_mass
    diagnostics["true_sky_inside_90_healpix"] = true_credible_mass <= (0.9 + 1e-7)

    sample_indices = np.unique(
        np.linspace(0, len(maps) - 1, min(256, len(maps))).astype(np.int32)
    )
    sample_rows = []
    pixel_area = FULL_SKY_DEG2 / maps.shape[1]
    for index in sample_indices:
        values = np.sort(maps[index].astype(np.float64))[::-1]
        count = int(np.searchsorted(np.cumsum(values), 0.9, side="left") + 1)
        sample_rows.append(
            {
                "event_index": int(index),
                "event_key": str(parameters.iloc[index]["event_key"]),
                "model_a90_calibrated_deg2": float(
                    parameters.iloc[index]["a90_model_deg2"] * temperature
                ),
                "healpix_a90_deg2": float(count * pixel_area),
            }
        )
    return diagnostics, pd.DataFrame(sample_rows)


def log_bayes_factor_matrix(
    maps: np.ndarray,
    *,
    block_size: int = 512,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability = torch.as_tensor(maps, dtype=torch.float32, device=device)
    output = np.empty((len(maps), len(maps)), dtype=np.float32)
    for start in range(0, len(maps), block_size):
        stop = min(start + block_size, len(maps))
        overlap = probability[start:stop] @ probability.T
        score = torch.log(torch.clamp(overlap * maps.shape[1], min=1e-30))
        output[start:stop] = score.cpu().numpy()
    output[:] = 0.5 * (output + output.T)
    np.fill_diagonal(output, -np.inf)
    del probability
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def build_split_sky(
    split: str,
    events: pd.DataFrame,
    scenario: SkyScenario,
    realization_seed: int,
    temperature: float,
    output_seed: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    parameters = build_event_parameters(events, scenario, realization_seed)
    maps = generate_probability_maps(parameters, temperature)
    diagnostics, area_sample = map_diagnostics(maps, parameters, temperature)
    diagnostics.to_parquet(
        output_seed / f"{split}_event_sky_diagnostics_v82.parquet",
        index=False,
    )
    area_sample.to_csv(
        output_seed / f"{split}_healpix_area90_sample_v82.csv",
        index=False,
    )
    score = log_bayes_factor_matrix(maps)
    write_json(
        output_seed / f"{split}_sky_audit_v82.json",
        {
            "split": split,
            "scenario": asdict(scenario),
            "n_events": int(len(events)),
            "nside": NSIDE,
            "main_score": "log(N_pix*sum(P_i*P_j))",
            "posterior_temperature": float(temperature),
            "map_sum_min": float(maps.sum(axis=1).min()),
            "map_sum_max": float(maps.sum(axis=1).max()),
            "healpix_true_sky_90_coverage": float(
                diagnostics["true_sky_inside_90_healpix"].mean()
            ),
            "model_a90_median_deg2": float(
                diagnostics["a90_model_calibrated_deg2"].median()
            ),
            "model_a90_p90_deg2": float(
                diagnostics["a90_model_calibrated_deg2"].quantile(0.9)
            ),
        },
    )
    del maps
    return score, diagnostics


def run_seed(
    seed: int,
    scenario: SkyScenario,
    output_root: Path,
    *,
    pair_metrics: bool,
) -> dict[str, Any]:
    output_seed = output_root / "et3" / scenario.scenario_id / f"seed_{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)
    source_seed = base.V7_ET / f"seed_{seed}"
    validation_events = pd.read_parquet(source_seed / "validation_event_catalog.parquet")
    test_events = pd.read_parquet(source_seed / "test_event_catalog.parquet")
    validation = base.build_catalog(validation_events)
    test = base.build_catalog(test_events)
    val_waveform, val_time = base.load_waveform_and_time(
        seed,
        "validation",
        validation,
    )
    test_waveform, test_time = base.load_waveform_and_time(seed, "test", test)

    val_parameters = build_event_parameters(
        validation_events,
        scenario,
        seed + 82000,
    )
    temperature, temperature_audit = validation_temperature(val_parameters)
    write_json(
        output_seed / "posterior_temperature_calibration_v82.json",
        temperature_audit,
    )
    val_sky, val_sky_events = build_split_sky(
        "validation",
        validation_events,
        scenario,
        seed + 82000,
        temperature,
        output_seed,
    )
    test_sky, test_sky_events = build_split_sky(
        "test",
        test_events,
        scenario,
        seed + 83000,
        temperature,
        output_seed,
    )

    val_true_i, val_true_j = etv7.true_pairs(validation.partner)
    val_false_i, val_false_j = etv7.sample_false_pairs(
        validation.partner,
        1_000_000,
        seed + 84000,
    )
    labels = np.concatenate(
        [
            np.ones(len(val_true_i), dtype=np.int8),
            np.zeros(len(val_false_i), dtype=np.int8),
        ]
    )
    sky_values = np.concatenate(
        [
            val_sky[val_true_i, val_true_j],
            val_sky[val_false_i, val_false_j],
        ]
    )
    sky_validation = {
        "n_true_pairs": int(labels.sum()),
        "n_false_pairs": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, sky_values)),
        "average_precision": float(average_precision_score(labels, sky_values)),
        "true_score_median": float(np.median(sky_values[labels == 1])),
        "false_score_median": float(np.median(sky_values[labels == 0])),
    }

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
        seed + 85000,
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
        output_seed / "selected_weights_v82.json",
        {
            "scenario": asdict(scenario),
            "methods": methods,
            "selection_split": "validation systems only",
            "test_used_for_selection": False,
            "channel_standardization": (
                "none; waveform/time use frozen v7 evidence and sky uses raw log B_sky"
            ),
        },
    )
    for name, grid in grids.items():
        grid.to_csv(output_seed / f"fusion_grid_{name}_v82.csv", index=False)

    validation_query, validation_metrics, _ = base.retrieval_tables(
        val_components,
        validation,
        methods,
        seed,
        f"ET3-{scenario.scenario_id}-validation",
    )
    validation_query.to_parquet(
        output_seed / "validation_query_ranks_v82.parquet",
        index=False,
    )
    validation_metrics.to_csv(
        output_seed / "validation_retrieval_metrics_v82.csv",
        index=False,
    )
    query, metrics, score_cache = base.retrieval_tables(
        test_components,
        test,
        methods,
        seed,
        f"ET3-{scenario.scenario_id}",
    )
    query.to_parquet(output_seed / "query_ranks_v82.parquet", index=False)
    metrics.to_csv(output_seed / "retrieval_metrics_v82.csv", index=False)
    bootstrap = etv7.bootstrap_system_ci(query, draws=10000, seed=seed + 86000)
    bootstrap.to_csv(output_seed / "bootstrap_95ci_v82.csv", index=False)

    pair_payload: dict[str, Any] = {}
    if pair_metrics:
        for method in (
            "three_channel_strict_positive",
            "three_channel_unconstrained",
        ):
            summary, operating = etv7.full_unordered_pair_metrics(
                score_cache[method],
                test.partner,
                test.meta,
                aggregation="max",
            )
            summary.update(
                {
                    "deployment": "ET3",
                    "scenario": scenario.scenario_id,
                    "seed": int(seed),
                    "method": method,
                    "unordered_score_rule": "max; all component matrices are symmetric",
                }
            )
            pair_payload[method] = summary
            write_json(
                output_seed / f"pair_level_metrics_{method}_v82.json",
                summary,
            )
            operating.to_csv(
                output_seed / f"pair_level_operating_points_{method}_v82.csv",
                index=False,
            )

    true_i, true_j = etv7.true_pairs(test.partner)
    false_i, false_j = etv7.sample_false_pairs(
        test.partner,
        200_000,
        seed + 87000,
    )
    pair_i = np.concatenate([true_i, false_i])
    pair_j = np.concatenate([true_j, false_j])
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
            "strict_positive_score": score_cache[
                "three_channel_strict_positive"
            ][pair_i, pair_j],
            "unconstrained_score": score_cache[
                "three_channel_unconstrained"
            ][pair_i, pair_j],
            "sky_area90_i_deg2": test_sky_events[
                "a90_model_calibrated_deg2"
            ].to_numpy()[pair_i],
            "sky_area90_j_deg2": test_sky_events[
                "a90_model_calibrated_deg2"
            ].to_numpy()[pair_j],
            "sampling_strategy": np.concatenate(
                [
                    np.full(len(true_i), "all_true_pairs", dtype=object),
                    np.full(len(false_i), "fixed_random_false_pairs", dtype=object),
                ]
            ),
        }
    )
    family = test_events["family"].astype(str).to_numpy()
    diagnostics["true_pair_family"] = np.where(
        diagnostics["is_true_pair"].to_numpy(dtype=bool),
        family[pair_i],
        "background",
    )
    diagnostics.to_parquet(
        output_seed / "pair_diagnostics_sample_v82.parquet",
        index=False,
    )

    summary = {
        "status": "complete",
        "version": "et3_moderate_observed_sky_v82",
        "scenario": asdict(scenario),
        "seed": int(seed),
        "n_validation_events": int(len(validation)),
        "n_test_events": int(len(test)),
        "n_test_queries": int(np.sum(test.partner >= 0)),
        "n_test_true_pairs": int(len(true_i)),
        "n_test_unordered_pairs": int(len(test) * (len(test) - 1) // 2),
        "temperature_calibration": temperature_audit,
        "validation_sky_pair_separability": sky_validation,
        "validation_true_sky_90_coverage": float(
            val_sky_events["true_sky_inside_90_healpix"].mean()
        ),
        "test_true_sky_90_coverage": float(
            test_sky_events["true_sky_inside_90_healpix"].mean()
        ),
        "selected_weights": methods,
        "pair_level": pair_payload,
    }
    write_json(output_seed / "seed_summary_v82.json", summary)
    return summary


def aggregate(
    output_root: Path,
    scenario: SkyScenario,
    seeds: tuple[int, ...],
) -> None:
    root = output_root / "et3" / scenario.scenario_id
    metrics = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/retrieval_metrics_v82.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    metrics.to_csv(root / "retrieval_metrics_per_seed_v82.csv", index=False)
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
    summary.to_csv(root / "retrieval_metrics_summary_v82.csv", index=False)
    bootstrap = pd.concat(
        [
            pd.read_csv(root / f"seed_{seed}/bootstrap_95ci_v82.csv")
            for seed in seeds
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(root / "bootstrap_95ci_per_seed_v82.csv", index=False)
    weights = []
    pair_rows = []
    audits = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        selected = json.loads(
            (seed_root / "selected_weights_v82.json").read_text(encoding="utf-8")
        )
        for method, value in selected["methods"].items():
            weights.append(
                {
                    "seed": seed,
                    "method": method,
                    **value,
                }
            )
        seed_summary = json.loads(
            (seed_root / "seed_summary_v82.json").read_text(encoding="utf-8")
        )
        audits.append(
            {
                "seed": seed,
                "validation_coverage": seed_summary[
                    "validation_true_sky_90_coverage"
                ],
                "test_coverage": seed_summary["test_true_sky_90_coverage"],
                "temperature": seed_summary["temperature_calibration"]["temperature"],
                **seed_summary["validation_sky_pair_separability"],
            }
        )
        for method in (
            "three_channel_strict_positive",
            "three_channel_unconstrained",
        ):
            path = seed_root / f"pair_level_metrics_{method}_v82.json"
            if path.exists():
                pair_rows.append(json.loads(path.read_text(encoding="utf-8")))
    pd.DataFrame(weights).to_csv(
        root / "selected_weights_per_seed_v82.csv",
        index=False,
    )
    pd.DataFrame(audits).to_csv(root / "sky_audit_per_seed_v82.csv", index=False)
    if pair_rows:
        frame = pd.DataFrame(pair_rows)
        frame.to_csv(root / "pair_level_metrics_per_seed_v82.csv", index=False)
        numeric = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
            and column not in {"seed"}
        ]
        rows = []
        for method, part in frame.groupby("method"):
            row: dict[str, Any] = {"method": method, "n_seeds": len(part)}
            for column in numeric:
                row[f"{column}_mean"] = float(part[column].mean())
                row[f"{column}_std"] = float(part[column].std(ddof=1))
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            root / "pair_level_metrics_summary_v82.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="moderate")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-pair-metrics", action="store_true")
    args = parser.parse_args()
    scenario = SCENARIOS[args.scenario]
    seeds = (args.seed,) if args.seed is not None else PRIMARY_SEEDS
    for seed in seeds:
        run_seed(
            seed,
            scenario,
            args.output_root,
            pair_metrics=not args.skip_pair_metrics,
        )
    aggregate(args.output_root, scenario, seeds)


if __name__ == "__main__":
    main()
