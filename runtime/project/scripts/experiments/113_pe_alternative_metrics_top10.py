#!/usr/bin/env python3
"""Compute sample-level PE consistency diagnostics for current GWTC top-10 pairs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, jensenshannon, pdist


REPO = Path("/root/autodl-tmp/gw-catalog")
INPUT = (
    REPO
    / "results/et3_gwtc34_complete_20260726/summary/gwtc34_real_candidate_top20_with_pe.csv"
)
OUT_DIR = REPO / "results/pe_metric_crosscheck_20260726"
OUT_CSV = OUT_DIR / "gwtc34_top10_pe_alternative_metrics.csv"
OUT_JSONL = OUT_DIR / "gwtc34_top10_shared_source_evidence.jsonl"

PARAMETERS = ("chirp_mass", "mass_ratio", "chi_eff")
SAMPLE_LIMIT = 1_200
PROJECTION_COUNT = 128


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = module_from(
    REPO / "scripts/experiments/32_physical_deployment_v4_pe_audit.py",
    "pe_metric_helpers",
)


def seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)


def deterministic_sample(values: np.ndarray, count: int, key: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= count:
        return values
    rng = np.random.default_rng(seed_for(key))
    return values[np.sort(rng.choice(len(values), size=count, replace=False))]


def histogram_metrics(x: np.ndarray, y: np.ndarray, bins: int = 512) -> dict[str, float]:
    lo = min(float(np.quantile(x, 0.001)), float(np.quantile(y, 0.001)))
    hi = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        equal = float(np.isclose(np.median(x), np.median(y)))
        return {
            "overlap_coefficient": equal,
            "bhattacharyya_coefficient": equal,
            "hellinger_distance": float(math.sqrt(max(0.0, 1.0 - equal))),
            "jensen_shannon_distance": float(1.0 - equal),
        }
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges)
    py, _ = np.histogram(y, bins=edges)
    px = px.astype(np.float64)
    py = py.astype(np.float64)
    px /= max(px.sum(), 1.0)
    py /= max(py.sum(), 1.0)
    overlap = float(np.minimum(px, py).sum())
    bc = float(np.sqrt(px * py).sum())
    return {
        "overlap_coefficient": overlap,
        "bhattacharyya_coefficient": bc,
        "hellinger_distance": float(math.sqrt(max(0.0, 1.0 - bc))),
        "jensen_shannon_distance": float(jensenshannon(px, py, base=2.0)),
    }


def one_dimensional_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    hist = histogram_metrics(x, y)
    sigma_x = base.posterior_sigma(x)
    sigma_y = base.posterior_sigma(y)
    scale = math.sqrt(sigma_x * sigma_x + sigma_y * sigma_y)
    return {
        **hist,
        "ks_statistic": float(stats.ks_2samp(x, y, method="asymp").statistic),
        "wasserstein_distance": float(stats.wasserstein_distance(x, y)),
        "normalized_wasserstein_distance": float(
            stats.wasserstein_distance(x, y) / max(scale, 1e-12)
        ),
    }


def transformed_samples(record: dict, count: int, key: str) -> np.ndarray:
    values = base.transform_theta(record["posterior"])
    return deterministic_sample(values, count, key)


def regularized_covariance(values: np.ndarray) -> np.ndarray:
    covariance = np.cov(values, rowvar=False)
    ridge = max(float(np.trace(covariance)) / covariance.shape[0], 1e-8) * 1e-6
    return covariance + ridge * np.eye(covariance.shape[0])


def gaussian_bhattacharyya(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = regularized_covariance(x)
    cov_y = regularized_covariance(y)
    cov = 0.5 * (cov_x + cov_y)
    delta = mu_x - mu_y
    inverse = np.linalg.pinv(cov)
    term_center = 0.125 * float(delta @ inverse @ delta)
    sign, logdet = np.linalg.slogdet(cov)
    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    if min(sign, sign_x, sign_y) <= 0:
        return np.nan, np.nan
    term_shape = 0.5 * (logdet - 0.5 * (logdet_x + logdet_y))
    distance = max(0.0, term_center + term_shape)
    return float(math.exp(-distance)), float(distance)


def joint_metrics(a: dict, b: dict, pair_key: str) -> dict[str, float]:
    x = transformed_samples(a, SAMPLE_LIMIT, pair_key + ":a")
    y = transformed_samples(b, SAMPLE_LIMIT, pair_key + ":b")

    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    covariance_sum = regularized_covariance(x) + regularized_covariance(y)
    delta = mu_x - mu_y
    mahalanobis = float(math.sqrt(max(0.0, delta @ np.linalg.pinv(covariance_sum) @ delta)))

    pooled = np.vstack([x, y])
    center = np.median(pooled, axis=0)
    scale = np.std(pooled, axis=0)
    scale = np.maximum(scale, 1e-8)
    xz = (x - center) / scale
    yz = (y - center) / scale

    rng = np.random.default_rng(seed_for(pair_key + ":projections"))
    directions = rng.normal(size=(PROJECTION_COUNT, xz.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sliced_wasserstein = float(
        np.mean(
            [
                stats.wasserstein_distance(xz @ direction, yz @ direction)
                for direction in directions
            ]
        )
    )

    dxy = cdist(xz, yz)
    dxx = pdist(xz)
    dyy = pdist(yz)
    energy_squared = max(
        0.0,
        2.0 * float(dxy.mean()) - float(dxx.mean()) - float(dyy.mean()),
    )
    energy_distance = float(math.sqrt(energy_squared))

    median_sample = deterministic_sample(
        np.vstack([xz, yz]), min(600, len(xz) + len(yz)), pair_key + ":median"
    )
    squared_distances = pdist(median_sample, metric="sqeuclidean")
    median_squared_distance = float(np.median(squared_distances[squared_distances > 0]))
    gamma = 1.0 / max(2.0 * median_squared_distance, 1e-12)
    kxx = np.exp(-gamma * cdist(xz, xz, metric="sqeuclidean"))
    kyy = np.exp(-gamma * cdist(yz, yz, metric="sqeuclidean"))
    kxy = np.exp(-gamma * cdist(xz, yz, metric="sqeuclidean"))
    mmd_squared = max(0.0, float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean()))

    gaussian_bc, gaussian_bd = gaussian_bhattacharyya(xz, yz)
    return {
        "joint_mahalanobis_distance": mahalanobis,
        "joint_gaussian_bhattacharyya_coefficient": gaussian_bc,
        "joint_gaussian_bhattacharyya_distance": gaussian_bd,
        "joint_sliced_wasserstein_distance": sliced_wasserstein,
        "joint_energy_distance": energy_distance,
        "joint_rbf_mmd": float(math.sqrt(mmd_squared)),
        "joint_sample_count_each": int(min(len(x), len(y))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(INPUT)
    candidates = candidates.loc[candidates["consensus_rank"] <= 10].copy()

    caches = {}
    for deployment in ("gwtc3", "gwtc4"):
        cache, _ = base.load_catalog_pe(base.DEPLOYMENTS[deployment])
        caches[deployment] = cache

    rows: list[dict] = []
    shared_records: list[dict] = []
    for pair in candidates.itertuples(index=False):
        deployment = str(pair.deployment)
        event_i = str(pair.event_i)
        event_j = str(pair.event_j)
        pair_key = "--".join(sorted((event_i, event_j)))
        cache = caches[deployment]
        a = cache[event_i]
        b = cache[event_j]
        row = {
            "deployment": deployment,
            "deployment_label": {
                "gwtc3": "GWTC-3 / O3",
                "gwtc4": "GWTC-4.1 / O4a",
            }[deployment],
            "consensus_rank": int(pair.consensus_rank),
            "event_i": event_i,
            "event_j": event_j,
            "pair_key": pair_key,
        }
        for parameter in PARAMETERS:
            metrics = one_dimensional_metrics(
                np.asarray(a["posterior"][parameter], dtype=np.float64),
                np.asarray(b["posterior"][parameter], dtype=np.float64),
            )
            for name, value in metrics.items():
                row[f"{parameter}_{name}"] = value

        for name, value in joint_metrics(a, b, pair_key).items():
            row[name] = value

        shared = base.shared_source_overlap_evidence(a, b)
        shared_records.append(
            {
                "deployment": deployment,
                "consensus_rank": int(pair.consensus_rank),
                "event_i": event_i,
                "event_j": event_j,
                "pair_key": pair_key,
                **shared,
            }
        )
        row["shared_source_evidence_available"] = bool(shared.get("available", False))
        for name in (
            "symmetric_log_evidence",
            "symmetric_log_evidence_min_bandwidth_scan",
            "min_directional_log_evidence_all_bandwidths",
            "max_directional_log_evidence_disagreement",
        ):
            row[f"shared_source_{name}"] = shared.get(name, np.nan)
        rows.append(row)
        print(f"{deployment} rank {int(pair.consensus_rank)} complete", flush=True)

    result = pd.DataFrame(rows).sort_values(["deployment", "consensus_rank"])
    for metric in (
        "overlap_coefficient",
        "bhattacharyya_coefficient",
        "hellinger_distance",
        "jensen_shannon_distance",
        "ks_statistic",
        "normalized_wasserstein_distance",
    ):
        columns = [f"{parameter}_{metric}" for parameter in PARAMETERS]
        aggregate = "min" if metric in {"overlap_coefficient", "bhattacharyya_coefficient"} else "max"
        result[f"intrinsic_{metric}_{aggregate}"] = getattr(result[columns], aggregate)(axis=1)

    result.to_csv(OUT_CSV, index=False)
    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for record in shared_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(OUT_CSV)
    print(OUT_JSONL)


if __name__ == "__main__":
    main()
