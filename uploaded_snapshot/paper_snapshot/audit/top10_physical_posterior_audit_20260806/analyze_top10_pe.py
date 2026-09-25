#!/usr/bin/env python3
"""Expanded posterior and amplitude audit for the displayed v9.3 Top-10 pairs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path(__file__).resolve().parent
CURRENT = ROOT / "_ovl_sync_6a0019996cb0d99d490519e9/source_data/current_v93"
V93 = ROOT / "sky_resolution_v93_20260730/server_pull/gwtc_sky_resolution_v93_20260730"
O3_MANIFEST = (
    ROOT
    / "overleaf-project/real_gwtc_lensing_search_20260625_scaleaware_waveform_passed_clean_current"
    / "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv"
)
O4_MANIFEST = (
    ROOT
    / "server_pull/real_noise_injection_v6_20260721/extracted/source_manifests"
    / "real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest.csv"
)
PE_ROOT = AUDIT / "pe_files"
OUTPUT = AUDIT / "top10_expanded_pe_audit.csv"
OUTPUT_LONG = AUDIT / "top10_posterior_intervals_long.csv"

MAX_SAMPLES = 30_000
RATIO_SAMPLES = 30_000
PARAMETERS = (
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "mass_1",
    "mass_2",
    "chi_p",
    "theta_jn",
    "luminosity_distance",
    "network_optimal_snr",
)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def posterior_sigma(values: np.ndarray) -> float:
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def bc_histogram(x: np.ndarray, y: np.ndarray, bins: int = 384) -> float:
    lo = min(float(np.quantile(x, 0.001)), float(np.quantile(y, 0.001)))
    hi = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.isclose(np.median(x), np.median(y)))
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges)
    py, _ = np.histogram(y, bins=edges)
    px = px.astype(float) / max(float(px.sum()), 1.0)
    py = py.astype(float) / max(float(py.sum()), 1.0)
    return float(np.sqrt(px * py).sum())


def parameter_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    med_i, med_j = float(np.median(x)), float(np.median(y))
    q05_i, q95_i = np.quantile(x, [0.05, 0.95])
    q05_j, q95_j = np.quantile(y, [0.05, 0.95])
    sig_i, sig_j = posterior_sigma(x), posterior_sigma(y)
    scale = math.sqrt(sig_i * sig_i + sig_j * sig_j)
    return {
        "median_i": med_i,
        "q05_i": float(q05_i),
        "q95_i": float(q95_i),
        "median_j": med_j,
        "q05_j": float(q05_j),
        "q95_j": float(q95_j),
        "standardized_median_distance": abs(med_i - med_j) / max(scale, 1e-12),
        "bhattacharyya_coefficient": bc_histogram(x, y),
    }


def find_pe_file(deployment: str, event: str) -> Path:
    candidates = sorted((PE_ROOT / deployment).glob(f"*{event}*PEDataRelease*"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one PE file for {deployment} {event}; found {candidates}")
    return candidates[0]


def read_samples(path: Path, group_name: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with h5py.File(path, "r") as handle:
        if group_name not in handle:
            raise KeyError(f"{group_name} absent in {path.name}; groups={list(handle.keys())}")
        dataset = handle[f"{group_name}/posterior_samples"]
        names = set(dataset.dtype.names or ())
        count = len(dataset)
        index = np.linspace(0, count - 1, min(count, MAX_SAMPLES), dtype=np.int64)
        samples: dict[str, np.ndarray] = {}
        for parameter in PARAMETERS:
            alias = parameter
            if parameter == "theta_jn" and alias not in names and "iota" in names:
                alias = "iota"
            if alias not in names:
                continue
            values = np.asarray(dataset[alias][index], dtype=float)
            samples[parameter] = values[np.isfinite(values)]
        metadata = {
            "posterior_samples_total": int(count),
            "max_log_likelihood": float(np.nanmax(dataset["log_likelihood"]))
            if "log_likelihood" in names
            else np.nan,
        }
    return samples, metadata


def draw_distance_ratio(
    distance_i: np.ndarray, distance_j: np.ndarray, pair_key: str
) -> dict[str, float]:
    rng = np.random.default_rng(stable_seed(pair_key + ":distance_ratio"))
    n = min(RATIO_SAMPLES, len(distance_i), len(distance_j))
    i = rng.choice(len(distance_i), n, replace=len(distance_i) < n)
    j = rng.choice(len(distance_j), n, replace=len(distance_j) < n)
    amp_i_over_j = distance_j[j] / np.maximum(distance_i[i], 1e-12)
    mu_i_over_j = amp_i_over_j**2
    q05, median, q95 = np.quantile(mu_i_over_j, [0.05, 0.5, 0.95])
    amp_q05, amp_median, amp_q95 = np.quantile(amp_i_over_j, [0.05, 0.5, 0.95])
    return {
        "apparent_amplitude_ratio_i_over_j_q05": float(amp_q05),
        "apparent_amplitude_ratio_i_over_j_median": float(amp_median),
        "apparent_amplitude_ratio_i_over_j_q95": float(amp_q95),
        "apparent_mu_ratio_i_over_j_q05": float(q05),
        "apparent_mu_ratio_i_over_j_median": float(median),
        "apparent_mu_ratio_i_over_j_q95": float(q95),
        "prob_mu_ratio_within_25pct_of_unity": float(
            np.mean((mu_i_over_j >= 0.8) & (mu_i_over_j <= 1.25))
        ),
    }


def main() -> None:
    candidates = pd.read_csv(CURRENT / "fig4_selected_seed_candidates.csv")
    manifests = {
        "gwtc3": pd.read_csv(O3_MANIFEST).set_index("event_name", drop=False),
        "gwtc4": pd.read_csv(O4_MANIFEST).set_index("event_name", drop=False),
    }
    consensus = {
        deployment: pd.read_parquet(
            V93 / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        )
        for deployment in ("gwtc3", "gwtc4")
    }

    cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, float]]] = {}
    rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for pair in candidates.itertuples(index=False):
        deployment = str(pair.deployment)
        event_i, event_j = str(pair.event_i), str(pair.event_j)
        manifest = manifests[deployment]
        mi, mj = manifest.loc[event_i], manifest.loc[event_j]
        group_column = "sky_map_internal_group" if deployment == "gwtc3" else "sky_map_group"

        for event, manifest_row in ((event_i, mi), (event_j, mj)):
            key = (deployment, event)
            if key not in cache:
                cache[key] = read_samples(find_pe_file(deployment, event), str(manifest_row[group_column]))

        si, meta_i = cache[(deployment, event_i)]
        sj, meta_j = cache[(deployment, event_j)]
        pair_key = f"{event_i}--{event_j}"
        row: dict[str, object] = {
            "deployment": deployment,
            "rank": int(pair.rank),
            "event_i": event_i,
            "event_j": event_j,
            "pair_key": pair_key,
            "selected_seed": int(pair.selected_seed),
            "final_score": float(pair.final_score),
            "waveform_score": float(pair.waveform_score),
            "time_score": float(pair.time_score),
            "sky_score": float(pair.sky_score),
            "current_max_standardized_posterior_distance": float(
                pair.max_standardized_posterior_distance
            ),
            "current_intrinsic_3sigma_consistent": bool(
                pair.intrinsic_3sigma_consistent
            ),
            "waveform_contribution": 0.5 * float(pair.waveform_score),
            "time_contribution": 0.25 * float(pair.time_score),
            "sky_contribution": 0.5 * float(pair.sky_score),
            "delay_days": abs(float(mi.gps_time) - float(mj.gps_time)) / 86400.0,
            "catalog_network_snr_i": float(mi.network_snr),
            "catalog_network_snr_j": float(mj.network_snr),
            "catalog_network_snr_ratio_i_over_j": float(mi.network_snr) / float(mj.network_snr),
            "catalog_network_snr_squared_ratio_i_over_j": (
                float(mi.network_snr) / float(mj.network_snr)
            ) ** 2,
            "pe_group_i": str(mi[group_column]),
            "pe_group_j": str(mj[group_column]),
            "posterior_samples_total_i": meta_i["posterior_samples_total"],
            "posterior_samples_total_j": meta_j["posterior_samples_total"],
        }
        for parameter in PARAMETERS:
            if parameter not in si or parameter not in sj:
                continue
            summary = parameter_summary(si[parameter], sj[parameter])
            for name, value in summary.items():
                row[f"{parameter}_{name}"] = value
            long_rows.append(
                {
                    "deployment": deployment,
                    "rank": int(pair.rank),
                    "event_i": event_i,
                    "event_j": event_j,
                    "parameter": parameter,
                    **summary,
                }
            )

        row.update(
            draw_distance_ratio(
                si["luminosity_distance"], sj["luminosity_distance"], pair_key
            )
        )

        match = consensus[deployment]
        match = match[
            ((match.event_i == event_i) & (match.event_j == event_j))
            | ((match.event_i == event_j) & (match.event_j == event_i))
        ]
        if len(match) == 1:
            record = match.iloc[0]
            for name in (
                "sky_log_bf_nside32",
                "sky_log_bf_nside64",
                "sky_log_bf_nside128",
                "sky_log_bf_nside256",
                "sky_log_bf_nside512",
                "sky_log_bf_nside1024",
                "high_resolution_sign_stable",
                "sign_flip_nside32_1024",
            ):
                row[name] = record[name]
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["deployment", "rank"])
    result["core_min_bhattacharyya_coefficient"] = result[
        [
            "chirp_mass_bhattacharyya_coefficient",
            "mass_ratio_bhattacharyya_coefficient",
            "chi_eff_bhattacharyya_coefficient",
        ]
    ].min(axis=1)
    result["extended_intrinsic_min_bhattacharyya_coefficient"] = result[
        [
            "chirp_mass_bhattacharyya_coefficient",
            "mass_ratio_bhattacharyya_coefficient",
            "chi_eff_bhattacharyya_coefficient",
            "mass_1_bhattacharyya_coefficient",
            "mass_2_bhattacharyya_coefficient",
            "chi_p_bhattacharyya_coefficient",
        ]
    ].min(axis=1)
    result.to_csv(OUTPUT, index=False)
    pd.DataFrame(long_rows).sort_values(
        ["deployment", "rank", "parameter"]
    ).to_csv(OUTPUT_LONG, index=False)
    print(OUTPUT)
    print(OUTPUT_LONG)
    print(result[["deployment", "rank", "event_i", "event_j", "delay_days", "final_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
