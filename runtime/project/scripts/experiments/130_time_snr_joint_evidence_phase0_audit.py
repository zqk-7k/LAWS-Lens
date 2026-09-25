#!/usr/bin/env python3
"""Phase-0 provenance audit for time-delay/SNR joint evidence.

This script is intentionally read-only with respect to all existing v9.3 products.
It does not fit a new density, tune weights, or rerank real candidates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


AUDIT_DATE = "2026-08-07"
DEFAULT_RESULT = "results/time_snr_joint_evidence_phase0_20260807"
SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENTS = {
    "gwtc3": {
        "label": "GWTC-3.0/O3",
        "manifest": "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv",
        "preferred_group_column": "sky_map_internal_group",
        "v7_root": "results/real_noise_injection_v7_peak2s_formal_20260722/gwtc3",
        "source_root": "results/real_noise_injection_v5_physical_source_20260721/gwtc3/shared/physical_h1l1_source_bank",
        "v93_root": "results/gwtc_sky_resolution_v93_20260730/gwtc3",
    },
    "gwtc4": {
        "label": "GWTC-4.1/O4a",
        "manifest": "runs/real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest.csv",
        "preferred_group_column": "sky_map_group",
        "v7_root": "results/real_noise_injection_v7_peak2s_formal_20260722/gwtc4",
        "source_root": "results/real_noise_injection_v5_physical_source_20260721/gwtc4/shared/physical_h1l1_source_bank",
        "v93_root": "results/gwtc_sky_resolution_v93_20260730/gwtc4",
    },
}

GW_LMC_ROOT = Path("/root/autodl-tmp/GW-LMC/2.5PLUS/BBH")
GW_LMC_SNR1_IMAGE = GW_LMC_ROOT / "Any_Detected_SNR1/BBH_2.5PLUS_Any_Detected_SNR1_ImageParams.csv"
GW_LMC_SNR1_SOURCE = GW_LMC_ROOT / "Any_Detected_SNR1/BBH_2.5PLUS_Any_Detected_SNR1_SourceParams.csv"
GW_LMC_SNR8_IMAGE = GW_LMC_ROOT / "Any_Detected_SNR8/BBH_2.5PLUS_Any_Detected_SNR8_ImageParams.csv"
GW_LMC_SNR8_SOURCE = GW_LMC_ROOT / "Any_Detected_SNR8/BBH_2.5PLUS_Any_Detected_SNR8_SourceParams.csv"

ET_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_RESULT))
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    return array[np.isfinite(array)]


def quantiles(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = finite(values)
    if not len(array):
        return {"n": 0, "min": None, "q05": None, "median": None, "q95": None, "max": None}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def correlation(x: Iterable[Any], y: Iterable[Any]) -> dict[str, float | int | None]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[keep], ya[keep]
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return {"n": int(len(xa)), "pearson": None, "spearman": None}
    return {
        "n": int(len(xa)),
        "pearson": float(pearsonr(xa, ya).statistic),
        "spearman": float(spearmanr(xa, ya).statistic),
    }


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_numeric_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []
    try:
        return [float(item) for item in raw]
    except (TypeError, ValueError):
        return []


def extract_gwlmc_detectable_pairs(image_csv: Path, threshold: float = 8.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = pd.read_csv(image_csv)
    for _, row in frame.iterrows():
        delays = parse_numeric_list(row["img_delays_days"])
        snrs = parse_numeric_list(row["img_snrs"])
        if len(delays) != len(snrs) or len(delays) < 2:
            continue
        kept = [index for index, snr in enumerate(snrs) if snr >= threshold]
        for image_a, image_b in itertools.combinations(kept, 2):
            if delays[image_a] <= delays[image_b]:
                early, late = image_a, image_b
            else:
                early, late = image_b, image_a
            delta = abs(delays[late] - delays[early])
            if not np.isfinite(delta) or delta <= 0:
                continue
            snr_early = max(snrs[early], np.finfo(float).tiny)
            snr_late = max(snrs[late], np.finfo(float).tiny)
            rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "image_early": int(early),
                    "image_late": int(late),
                    "delta_t_days": float(delta),
                    "snr_early_proposal": float(snr_early),
                    "snr_late_proposal": float(snr_late),
                    "r_rho_signed": float(np.log(snr_late / snr_early)),
                    "snr_ratio_unsigned": float(max(snr_early, snr_late) / min(snr_early, snr_late)),
                }
            )
    return pd.DataFrame(rows)


def audit_et_snr() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("SIS", "image1", ET_ROOT / "SIS_data_0222/SIS_optimal_SNR_single_1.npy", ET_ROOT / "SIS_data_0222/SIS_optimal_SNR_network_1.npy"),
        ("SIS", "image2", ET_ROOT / "SIS_data_0222/SIS_optimal_SNR_single_2.npy", ET_ROOT / "SIS_data_0222/SIS_optimal_SNR_network_2.npy"),
        ("PM", "image1", ET_ROOT / "PM_data_0222/PM_optimal_SNR_single_1.npy", ET_ROOT / "PM_data_0222/PM_optimal_SNR_network_1.npy"),
        ("PM", "image2", ET_ROOT / "PM_data_0222/PM_optimal_SNR_single_2.npy", ET_ROOT / "PM_data_0222/PM_optimal_SNR_network_2.npy"),
        ("unlensed", "single", ET_ROOT / "Unlensed_data_0222/unlensed_optimal_SNR_single.npy", ET_ROOT / "Unlensed_data_0222/unlensed_optimal_SNR_network.npy"),
    ]
    max_error = 0.0
    for family, image, component_path, network_path in specs:
        components = np.asarray(np.load(component_path), dtype=np.float64)
        network = np.asarray(np.load(network_path), dtype=np.float64).reshape(-1)
        recomputed = np.sqrt(np.sum(np.square(components), axis=1))
        error = np.abs(recomputed - network)
        max_error = max(max_error, float(np.max(error)))
        stats = quantiles(network)
        rows.append(
            {
                "dataset": "ET-3",
                "family": family,
                "image": image,
                "n_events": len(network),
                "n_subinterferometers": int(components.shape[1]),
                "network_snr_median": stats["median"],
                "network_snr_q05": stats["q05"],
                "network_snr_q95": stats["q95"],
                "quadrature_max_abs_error": float(np.max(error)),
                "single_snr_path": str(component_path),
                "network_snr_path": str(network_path),
            }
        )
    summary = {
        "definition": "Bilby PSD-weighted optimal SNR for the three ET triangular sub-interferometers, combined in quadrature",
        "observable_class": "simulation-truth optimal SNR; not a recovered search statistic",
        "three_subinterferometer_check_max_abs_error": max_error,
        "generator_family": "analytic SIS and analytic point-mass ET simulations",
        "network": "ET triangular detector represented by three sub-interferometers",
    }
    return pd.DataFrame(rows), summary


def find_posterior_datasets(handle: h5py.File) -> list[str]:
    candidates: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset) or not name.endswith("posterior_samples"):
            return
        names = set(obj.dtype.names or ())
        if {"network_matched_filter_snr", "network_optimal_snr"}.intersection(names):
            candidates.append(name)

    handle.visititems(visitor)
    return candidates


def choose_posterior_dataset(candidates: list[str], preferred_group: str) -> str | None:
    preferred = f"{preferred_group}/posterior_samples" if preferred_group else ""
    if preferred in candidates:
        return preferred
    priorities = ("Mixed+XO4a", "Mixed", "IMRPhenomXO4a", "IMRPhenomXPHM")
    for token in priorities:
        matches = sorted(path for path in candidates if token in path)
        if matches:
            return matches[0]
    return sorted(candidates)[0] if candidates else None


def read_pe_snr(path: Path, preferred_group: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pe_file_exists": path.is_file(),
        "pe_group_requested": preferred_group,
        "pe_dataset_used": None,
        "pe_sample_count": 0,
        "pe_network_matched_filter_snr_median": np.nan,
        "pe_network_matched_filter_snr_q05": np.nan,
        "pe_network_matched_filter_snr_q95": np.nan,
        "pe_network_optimal_snr_median": np.nan,
        "pe_network_optimal_snr_q05": np.nan,
        "pe_network_optimal_snr_q95": np.nan,
        "pe_read_error": "",
    }
    if not path.is_file():
        result["pe_read_error"] = "missing_file"
        return result
    try:
        with h5py.File(path, "r") as handle:
            dataset_name = choose_posterior_dataset(find_posterior_datasets(handle), preferred_group)
            if dataset_name is None:
                result["pe_read_error"] = "no_posterior_dataset_with_snr"
                return result
            dataset = handle[dataset_name]
            result["pe_dataset_used"] = dataset_name
            result["pe_sample_count"] = int(dataset.shape[0])
            names = set(dataset.dtype.names or ())
            for field, prefix in (
                ("network_matched_filter_snr", "pe_network_matched_filter_snr"),
                ("network_optimal_snr", "pe_network_optimal_snr"),
            ):
                if field not in names:
                    continue
                values = finite(np.asarray(dataset[field]))
                if not len(values):
                    continue
                result[f"{prefix}_median"] = float(np.median(values))
                result[f"{prefix}_q05"] = float(np.quantile(values, 0.05))
                result[f"{prefix}_q95"] = float(np.quantile(values, 0.95))
    except Exception as exc:  # Keep a complete per-event failure audit.
        result["pe_read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def strict_real_events(v93_root: Path) -> set[str]:
    path = v93_root / f"seed_{SEEDS[0]}/real_pair_features_unified_sky_v81.parquet"
    pairs = pd.read_parquet(path, columns=["event_i", "event_j", "strict_h1l1_bbh_pair"])
    strict = pairs[as_bool(pairs["strict_h1l1_bbh_pair"])]
    return set(strict["event_i"].astype(str)).union(strict["event_j"].astype(str))


def pair_log_ratios(events: pd.DataFrame, value_column: str) -> pd.DataFrame:
    frame = events[["event_name", "gps_time", value_column]].copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame[np.isfinite(frame[value_column]) & (frame[value_column] > 0)].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(frame)), 2):
        left, right = frame.iloc[i], frame.iloc[j]
        early, late = (left, right) if left["gps_time"] <= right["gps_time"] else (right, left)
        rows.append(
            {
                "event_early": early["event_name"],
                "event_late": late["event_name"],
                "pair_key": "--".join(sorted((str(left["event_name"]), str(right["event_name"])))),
                value_column: float(np.log(float(late[value_column]) / float(early[value_column]))),
            }
        )
    return pd.DataFrame(rows)


def audit_real_snr(root: Path, deployment: str, config: dict[str, str]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = resolve(root, config["manifest"])
    manifest = pd.read_csv(manifest_path)
    primary = manifest[as_bool(manifest["include_in_primary_search"])].copy()
    strict_names = strict_real_events(resolve(root, config["v93_root"]))
    primary["strict_h1l1_bbh_event"] = primary["event_name"].astype(str).isin(strict_names)
    group_column = config["preferred_group_column"]
    rows: list[dict[str, Any]] = []
    for _, event in primary.iterrows():
        pe_path = resolve(root, str(event.get("sky_map_path", "")))
        preferred_group = str(event.get(group_column, "")) if pd.notna(event.get(group_column)) else ""
        pe = read_pe_snr(pe_path, preferred_group)
        rows.append(
            {
                "deployment": deployment,
                "deployment_label": config["label"],
                "event_name": str(event["event_name"]),
                "gps_time": float(event["gps_time"]),
                "manifest_network_snr": float(event["network_snr"]) if pd.notna(event["network_snr"]) else np.nan,
                "manifest_network_snr_definition": "GWOSC event API/allevents network_matched_filter_snr",
                "strict_h1l1_bbh_event": bool(event["strict_h1l1_bbh_event"]),
                "pe_path": str(pe_path),
                **pe,
            }
        )
    audit = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for scope, subset in (
        ("primary_catalog", audit),
        ("strict_h1l1_bbh", audit[audit["strict_h1l1_bbh_event"]]),
    ):
        n = len(subset)
        matched = subset["pe_network_matched_filter_snr_median"]
        optimal = subset["pe_network_optimal_snr_median"]
        manifest_snr = subset["manifest_network_snr"]
        summaries.append(
            {
                "deployment": deployment,
                "scope": scope,
                "n_events": n,
                "pe_matched_filter_coverage": float(np.isfinite(matched).sum() / n) if n else np.nan,
                "pe_optimal_coverage": float(np.isfinite(optimal).sum() / n) if n else np.nan,
                "manifest_vs_pe_matched_pearson": correlation(manifest_snr, matched)["pearson"],
                "manifest_vs_pe_matched_spearman": correlation(manifest_snr, matched)["spearman"],
                "manifest_vs_pe_optimal_pearson": correlation(manifest_snr, optimal)["pearson"],
                "manifest_vs_pe_optimal_spearman": correlation(manifest_snr, optimal)["spearman"],
                "pe_matched_vs_optimal_pearson": correlation(matched, optimal)["pearson"],
                "pe_matched_vs_optimal_spearman": correlation(matched, optimal)["spearman"],
                "median_pe_optimal_over_matched": float(np.nanmedian(optimal / matched)),
                "p90_abs_log_pe_optimal_over_matched": float(np.nanquantile(np.abs(np.log(optimal / matched)), 0.9)),
                "pe_read_failures": int((subset["pe_read_error"].astype(str) != "").sum()),
            }
        )

    strict = audit[audit["strict_h1l1_bbh_event"]].copy()
    ratio_frames = []
    for column in (
        "manifest_network_snr",
        "pe_network_matched_filter_snr_median",
        "pe_network_optimal_snr_median",
    ):
        ratio_frames.append(pair_log_ratios(strict, column))
    ratios = ratio_frames[0]
    for extra in ratio_frames[1:]:
        ratios = ratios.merge(extra[["pair_key", extra.columns[-1]]], on="pair_key", how="inner")
    pair_sensitivity: dict[str, Any] = {"n_pairs_complete": int(len(ratios))}
    comparisons = [
        ("manifest_vs_pe_matched", "manifest_network_snr", "pe_network_matched_filter_snr_median"),
        ("manifest_vs_pe_optimal", "manifest_network_snr", "pe_network_optimal_snr_median"),
        ("pe_matched_vs_pe_optimal", "pe_network_matched_filter_snr_median", "pe_network_optimal_snr_median"),
    ]
    for label, left, right in comparisons:
        delta = np.abs(ratios[left].to_numpy() - ratios[right].to_numpy()) if len(ratios) else np.array([])
        corr = correlation(ratios[left], ratios[right]) if len(ratios) else {"n": 0, "pearson": None, "spearman": None}
        pair_sensitivity[label] = {
            **corr,
            "median_abs_delta_log_ratio": float(np.median(delta)) if len(delta) else None,
            "p90_abs_delta_log_ratio": float(np.quantile(delta, 0.9)) if len(delta) else None,
            "max_abs_delta_log_ratio": float(np.max(delta)) if len(delta) else None,
        }
    pair_sensitivity["definition_judgement"] = (
        "Manifest/PE matched-filter SNR and PE optimal SNR are correlated but are not definition-identical. "
        "Current injections use clean-signal PSD-optimal SNR; current real ranking tables carry catalog matched-filter SNR."
    )
    return audit, summaries, pair_sensitivity


def audit_injection_snr(root: Path, deployment: str, config: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    sampled_rows: list[pd.DataFrame] = []
    manifest = pd.read_csv(resolve(root, config["manifest"]))
    empirical = np.sort(pd.to_numeric(manifest.loc[as_bool(manifest["include_in_primary_search"]), "network_snr"], errors="coerce").dropna().to_numpy())
    for seed in SEEDS:
        path = resolve(root, config["v7_root"]) / f"seed_{seed}/data/real_noise_injections/compact_injection_metadata.parquet"
        frame = pd.read_parquet(path)
        lensed = frame[frame["family"].isin(["SIS", "PM"])].copy()
        t1 = lensed["image1_target_network_snr"].to_numpy(dtype=float)
        t2 = lensed["image2_target_network_snr"].to_numpy(dtype=float)
        r1 = lensed["image1_recovered_optimal_network_snr"].to_numpy(dtype=float)
        r2 = lensed["image2_recovered_optimal_network_snr"].to_numpy(dtype=float)
        prior = lensed["snr_ratio_prior"].to_numpy(dtype=float)
        realized_unsigned = np.maximum(t1, t2) / np.minimum(t1, t2)
        faint = np.minimum(t1, t2)
        nearest_empirical = np.min(np.abs(faint[:, None] - empirical[None, :]), axis=1)
        summary_rows.append(
            {
                "deployment": deployment,
                "seed": seed,
                "n_lensed_systems": len(lensed),
                "n_smooth_non_subhalo": int((lensed["family"] == "SIS").sum()),
                "n_subhalo_present": int((lensed["family"] == "PM").sum()),
                "max_abs_target_minus_recovered_optimal_snr": float(max(np.max(np.abs(t1 - r1)), np.max(np.abs(t2 - r2)))),
                "max_abs_realized_unsigned_ratio_minus_gwlmc_proposal_ratio": float(np.max(np.abs(realized_unsigned - prior))),
                "fraction_faint_target_exactly_from_real_manifest_snr": float(np.mean(nearest_empirical < 1e-9)),
                "target_snr_min": float(min(np.min(t1), np.min(t2))),
                "target_snr_median": float(np.median(np.concatenate([t1, t2]))),
                "target_snr_max": float(max(np.max(t1), np.max(t2))),
                "proposal_ratio_median": float(np.median(prior)),
                "proposal_ratio_q95": float(np.quantile(prior, 0.95)),
                "snr_definition": "PSD-weighted optimal H1-L1 network SNR of the known clean injected signal",
                "construction": "fainter target sampled from real catalog matched-filter SNR; brighter target fixed by same GW-LMC proposal ratio; each image independently rescaled",
            }
        )
        sample = lensed[["family", "sample_index", "gwlmc_row", "gps_image1", "gps_image2", "snr_ratio_prior", "image1_target_network_snr", "image2_target_network_snr", "image1_recovered_optimal_network_snr", "image2_recovered_optimal_network_snr"]].copy()
        sample.insert(0, "seed", seed)
        sample.insert(0, "deployment", deployment)
        sample["r_rho_signed_target"] = np.log(sample["image2_target_network_snr"] / sample["image1_target_network_snr"])
        sampled_rows.append(sample)
    return pd.DataFrame(summary_rows), pd.concat(sampled_rows, ignore_index=True)


def load_source_mapping(source_root: Path) -> pd.DataFrame:
    frames = []
    for family, directory, physical in (
        ("SIS", "SIS_data_0222", "gwlmc_smooth_non_subhalo"),
        ("PM", "PM_data_0222", "gwlmc_subhalo_present"),
    ):
        path = source_root / directory / "physical_source_pair_metadata.parquet"
        frame = pd.read_parquet(path, columns=["pair_id", "gwlmc_row", "gwlmc_event_id", "physical_lens_group"])
        frame = frame.rename(columns={"pair_id": "source_index"})
        frame["family"] = family
        frame["expected_physical_lens_group"] = physical
        frame["source_metadata_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def split_system_ids(root: Path, deployment: str, config: dict[str, str], prior_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = load_source_mapping(resolve(root, config["source_root"]))
    membership_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = resolve(root, config["v7_root"]) / f"seed_{seed}/results"
        split_sets: dict[str, set[tuple[str, int]]] = {}
        for split in ("validation", "test"):
            short = "val" if split == "validation" else "test"
            events = pd.read_parquet(seed_root / f"mixed_{short}_synthetic_events_v7.parquet", columns=["family", "source_index"])
            events = events[events["family"].isin(["SIS", "PM"])]
            keys = set(zip(events["family"].astype(str), events["source_index"].astype(int)))
            split_sets[split] = keys
        all_keys = set(zip(mapping["family"].astype(str), mapping["source_index"].astype(int)))
        split_sets["train"] = all_keys - split_sets["validation"] - split_sets["test"]
        for split, keys in split_sets.items():
            selected = mapping[
                [(str(family), int(index)) in keys for family, index in zip(mapping["family"], mapping["source_index"])]
            ].copy()
            selected.insert(0, "split", split)
            selected.insert(0, "seed", seed)
            selected.insert(0, "deployment", deployment)
            selected["in_current_time_prior_source"] = selected["gwlmc_event_id"].astype(int).isin(prior_ids)
            membership_rows.append(selected)
            audit_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "split": split,
                    "n_systems": len(selected),
                    "n_unique_gwlmc_event_ids": selected["gwlmc_event_id"].nunique(),
                    "n_systems_overlapping_current_time_prior": int(selected["in_current_time_prior_source"].sum()),
                    "fraction_overlapping_current_time_prior": float(selected["in_current_time_prior_source"].mean()),
                }
            )
        for left, right in itertools.combinations(("train", "validation", "test"), 2):
            left_ids = set(mapping[[key in split_sets[left] for key in zip(mapping["family"], mapping["source_index"])]]["gwlmc_event_id"].astype(int))
            right_ids = set(mapping[[key in split_sets[right] for key in zip(mapping["family"], mapping["source_index"])]]["gwlmc_event_id"].astype(int))
            audit_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "split": f"intersection_{left}_{right}",
                    "n_systems": len(left_ids & right_ids),
                    "n_unique_gwlmc_event_ids": len(left_ids & right_ids),
                    "n_systems_overlapping_current_time_prior": np.nan,
                    "fraction_overlapping_current_time_prior": np.nan,
                }
            )
    return pd.concat(membership_rows, ignore_index=True), pd.DataFrame(audit_rows)


def audit_time_prior(root: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    pairs = extract_gwlmc_detectable_pairs(GW_LMC_SNR8_IMAGE, threshold=8.0)
    prior_ids = set(pairs["event_id"].astype(int))
    source8 = pd.read_csv(GW_LMC_SNR8_SOURCE)
    source1 = pd.read_csv(GW_LMC_SNR1_SOURCE)
    merged = source8.merge(source1, on="event_id", suffixes=("_snr8", "_snr1"), how="left", indicator=True)
    comparison_columns = ["m1_det", "m2_det", "z_source", "dl_source", "gps_time", "ra", "dec"]
    max_differences = {
        column: float(np.nanmax(np.abs(merged[f"{column}_snr8"] - merged[f"{column}_snr1"])))
        for column in comparison_columns
    }
    system_counts = pairs.groupby("event_id").size().rename("n_detectable_pairs").reset_index()
    time_summary = {
        "source": "GW-LMC 2.5PLUS BBH Any_Detected_SNR8",
        "snr_threshold": 8.0,
        "source_catalog_rows": int(len(source8)),
        "eligible_independent_lens_systems": int(pairs["event_id"].nunique()),
        "raw_detectable_image_pairs": int(len(pairs)),
        "systems_with_multiple_pair_rows": int((system_counts["n_detectable_pairs"] > 1).sum()),
        "largest_pair_multiplicity_per_system": int(system_counts["n_detectable_pairs"].max()),
        "delay_days": quantiles(pairs["delta_t_days"]),
        "signed_log_snr_ratio": quantiles(pairs["r_rho_signed"]),
        "unsigned_snr_ratio": quantiles(pairs["snr_ratio_unsigned"]),
        "kde_or_lookup_monte_carlo_draws": 100000,
        "independence_statement": "The 100,000 exposure-conditioned/bootstrap draws are not independent lens systems.",
        "snr8_ids_found_in_snr1_source_catalog": int((merged["_merge"] == "both").sum()),
        "snr1_snr8_source_parameter_max_abs_differences": max_differences,
        "image_csv": str(GW_LMC_SNR8_IMAGE),
        "source_csv": str(GW_LMC_SNR8_SOURCE),
    }
    memberships = []
    split_audits = []
    for deployment, config in DEPLOYMENTS.items():
        membership, audit = split_system_ids(root, deployment, config, prior_ids)
        memberships.append(membership)
        split_audits.append(audit)
    return pairs, time_summary, pd.concat(memberships, ignore_index=True), pd.concat(split_audits, ignore_index=True)


def provenance_inventory(root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "ET-3",
                "field_or_quantity": "optimal_SNR_network",
                "definition": "sqrt(sum over ET1/ET2/ET3 of Bilby optimal_snr_squared)",
                "source": str(ET_ROOT),
                "noise_or_response_conditioning": "ET simulation PSD/response at generated event parameters",
                "equivalent_to_current_real_manifest_snr": False,
                "role": "simulation observable/audit candidate",
            },
            {
                "dataset": "O3 real-noise injection",
                "field_or_quantity": "image*_recovered_optimal_network_snr",
                "definition": "PSD-weighted H1-L1 optimal SNR of known clean injected signal, 20-1024 Hz",
                "source": str(resolve(root, DEPLOYMENTS["gwtc3"]["v7_root"])),
                "noise_or_response_conditioning": "run-matched off-source PSD; target imposed by independent image rescaling",
                "equivalent_to_current_real_manifest_snr": False,
                "role": "closed-loop injection observable",
            },
            {
                "dataset": "O4a real-noise injection",
                "field_or_quantity": "image*_recovered_optimal_network_snr",
                "definition": "PSD-weighted H1-L1 optimal SNR of known clean injected signal, 20-1024 Hz",
                "source": str(resolve(root, DEPLOYMENTS["gwtc4"]["v7_root"])),
                "noise_or_response_conditioning": "run-matched off-source PSD; target imposed by independent image rescaling",
                "equivalent_to_current_real_manifest_snr": False,
                "role": "closed-loop injection observable",
            },
            {
                "dataset": "GWTC-3.0 real catalog",
                "field_or_quantity": "network_snr",
                "definition": "GWOSC allevents/API network_matched_filter_snr",
                "source": str(resolve(root, DEPLOYMENTS["gwtc3"]["manifest"])),
                "noise_or_response_conditioning": "search/catalog statistic; pipeline-dependent",
                "equivalent_to_current_real_manifest_snr": True,
                "role": "current real catalog audit column",
            },
            {
                "dataset": "GWTC-4.1/O4a real catalog",
                "field_or_quantity": "network_snr",
                "definition": "GWOSC event API network_matched_filter_snr",
                "source": str(resolve(root, DEPLOYMENTS["gwtc4"]["manifest"])),
                "noise_or_response_conditioning": "search/catalog statistic; pipeline-dependent",
                "equivalent_to_current_real_manifest_snr": True,
                "role": "current real catalog audit column",
            },
            {
                "dataset": "GWTC PE posterior",
                "field_or_quantity": "network_optimal_snr",
                "definition": "posterior-derived model optimal network SNR in the frozen preferred PE group",
                "source": "per-event public PE HDF5 listed by each manifest",
                "noise_or_response_conditioning": "waveform-model and PE dependent",
                "equivalent_to_current_real_manifest_snr": False,
                "role": "candidate definition-aligned alternative requiring explicit freeze",
            },
        ]
    )


def family_label_audit(root: Path) -> pd.DataFrame:
    rows = [
        {
            "scope": "ET-3",
            "legacy_family_label": "SIS",
            "physical_interpretation": "analytic singular isothermal sphere lens simulation",
            "may_be_called_SIS_in_results": True,
        },
        {
            "scope": "ET-3",
            "legacy_family_label": "PM",
            "physical_interpretation": "analytic point-mass lens simulation",
            "may_be_called_SIS_in_results": False,
        },
    ]
    for deployment, config in DEPLOYMENTS.items():
        mapping = load_source_mapping(resolve(root, config["source_root"]))
        for family, expected in (("SIS", "gwlmc_smooth_non_subhalo"), ("PM", "gwlmc_subhalo_present")):
            subset = mapping[mapping["family"] == family]
            rows.append(
                {
                    "scope": deployment,
                    "legacy_family_label": family,
                    "physical_interpretation": "GW-LMC smooth/non-subhalo" if family == "SIS" else "GW-LMC subhalo-present",
                    "may_be_called_SIS_in_results": False,
                    "n_systems": len(subset),
                    "metadata_matches_expected_mapping": bool((subset["physical_lens_group"] == expected).all()),
                }
            )
    return pd.DataFrame(rows)


def build_decision(
    real_summaries: pd.DataFrame,
    injection_summary: pd.DataFrame,
    split_audit: pd.DataFrame,
    time_summary: dict[str, Any],
) -> dict[str, Any]:
    strict = real_summaries[real_summaries["scope"] == "strict_h1l1_bbh"]
    pe_optimal_complete = bool((strict["pe_optimal_coverage"] >= 0.95).all())
    split_intersections = split_audit[split_audit["split"].str.startswith("intersection_")]
    split_isolation_ok = bool((split_intersections["n_unique_gwlmc_event_ids"] == 0).all())
    heldout = split_audit[split_audit["split"].isin(["validation", "test"])]
    prior_overlap_count = int(heldout["n_systems_overlapping_current_time_prior"].sum())
    closed_loop_exact = bool(
        (injection_summary["max_abs_target_minus_recovered_optimal_snr"] < 1e-8).all()
        and (injection_summary["max_abs_realized_unsigned_ratio_minus_gwlmc_proposal_ratio"] < 1e-8).all()
    )
    return {
        "phase": "Phase 0 only",
        "overall_phase1_status": "HOLD_FOR_AUTHOR_DECISION",
        "real_catalog_current_manifest_snr": {
            "status": "NO_GO_FOR_FORMAL_REAL_RERANK",
            "reason": (
                "The current real manifests use recovered/search network matched-filter SNR, while current injections use "
                "the PSD-optimal SNR of a known clean signal. Equal field names or numerical correlation do not make these estimands identical."
            ),
        },
        "real_catalog_pe_optimal_snr_alternative": {
            "status": "CONDITIONAL_CANDIDATE_REQUIRING_AUTHOR_FREEZE" if pe_optimal_complete else "NO_GO_INCOMPLETE_COVERAGE",
            "strict_catalog_coverage_at_least_95pct": pe_optimal_complete,
            "reason": (
                "Public PE files expose posterior network_optimal_snr, which is closer in definition to injection optimal SNR, "
                "but it is waveform-model/PE dependent and was not the pre-existing catalog observable. Its group choice and use must be frozen before Phase 1."
            ),
        },
        "current_closed_loop_injection": {
            "status": "DIAGNOSTIC_ONLY",
            "exact_target_and_proposal_ratio_closure_confirmed": closed_loop_exact,
            "reason": "Each image was independently rescaled, so the realized unsigned SNR ratio reproduces the GW-LMC proposal ratio by construction.",
        },
        "response_derived_injection": {
            "status": "REQUIRED_BUT_NOT_YET_BUILT",
            "reason": "A fair primary test needs one shared source-distance/amplitude factor per system and a ratio emerging from magnification, response, arrival time, and local PSD.",
        },
        "current_encoder_split_isolation": {
            "status": "PASS" if split_isolation_ok else "FAIL",
            "zero_train_validation_test_event_id_intersections": split_isolation_ok,
        },
        "current_time_prior_system_isolation": {
            "status": "FAIL" if prior_overlap_count else "PASS",
            "heldout_membership_overlap_count_across_seed_tables": prior_overlap_count,
            "reason": (
                "Any overlap means the external one-dimensional prior includes the same GW-LMC system IDs used in held-out injections; "
                "a new joint density must exclude validation/test IDs at system level."
            ),
        },
        "time_prior_independent_sample_base": {
            "independent_lens_systems": time_summary["eligible_independent_lens_systems"],
            "detectable_image_pairs": time_summary["raw_detectable_image_pairs"],
            "bootstrap_draws_are_independent": False,
        },
        "authorized_actions_completed": [
            "SNR provenance inventory",
            "real PE SNR coverage and sensitivity audit",
            "closed-loop injection construction audit",
            "GW-LMC train/validation/test system-ID isolation audit",
            "current time-prior source overlap audit",
            "legacy family-name audit",
        ],
        "actions_not_run": [
            "new two-dimensional density fitting",
            "response-derived injection generation",
            "weight selection",
            "held-out retrieval comparison",
            "real catalog reranking",
            "paper modification",
        ],
    }


def make_report(
    out: Path,
    et_summary: dict[str, Any],
    real_summary: pd.DataFrame,
    pair_sensitivity: dict[str, Any],
    injection_summary: pd.DataFrame,
    time_summary: dict[str, Any],
    split_audit: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    lines = [
        "# 时间延迟与相对观测强度联合证据：Phase 0 provenance 审计",
        "",
        f"日期：{AUDIT_DATE}",
        "",
        "## 1. 本轮边界",
        "",
        "本轮严格停在 Phase 0。没有拟合二维密度、没有修改融合权重、没有重排真实候选、没有重训编码器，也没有覆盖 v9.3。所有输出都在新的独立目录中。",
        "",
        "## 2. 结论先行",
        "",
        f"总体状态：**{decision['overall_phase1_status']}**。",
        "",
        "当前 manifest 中的真实 GWTC `network_snr` **不能直接通过正式真实目录接入门槛**。它是搜索/目录给出的 network matched-filter SNR；O3/O4a 注入中的量是对已知干净注入信号按局部 PSD 计算的 optimal H1-L1 network SNR。二者相关不等于定义相同。",
        "",
        "公开 PE HDF5 中存在 `network_optimal_snr`，它可作为更接近注入定义的备选观测量；但它依赖 PE 波形模型和冻结的 posterior group。是否将其设为 Phase 1 的正式真实目录量，需要作者明确冻结，不能由本次脚本自动决定。",
        "",
        "当前逐像缩放注入被确认是 closed-loop control：较暗像 SNR 从真实目录经验分布抽取，较亮像由同一 GW-LMC proposal ratio 固定，然后两幅像分别缩放。它适合检验实现是否闭合，不足以支持真实推广结论。",
        "",
        "## 3. 三类 SNR 的物理定义",
        "",
        "| 数据 | 当前量 | 实际含义 | 可否直接共用 |",
        "|---|---|---|---|",
        "| ET-3 | optimal network SNR | Bilby 计算的 ET 三个子干涉仪 optimal SNR 的平方和开根号 | 否，探测器情景不同 |",
        "| O3/O4a 注入 | recovered optimal H1-L1 network SNR | 已知干净注入信号在局部 PSD 下的 optimal SNR | 仅适用于注入 |",
        "| 真实 GWTC manifest | network matched-filter SNR | 搜索/目录 pipeline 的 recovered matched-filter statistic | 不能直接当作注入 optimal SNR |",
        "| 真实 GWTC PE | posterior network optimal SNR | 给定 PE 波形模型的 posterior-derived optimal SNR | 条件性备选，需冻结 |",
        "",
        f"ET 三子干涉仪 network SNR 的重算最大误差为 `{et_summary['three_subinterferometer_check_max_abs_error']:.3g}`，因此 ET 文件内部定义一致。",
        "",
        "## 4. 真实目录 PE SNR 覆盖与敏感性",
        "",
        real_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    for deployment, payload in pair_sensitivity.items():
        lines.extend(
            [
                f"### {deployment}",
                "",
                f"严格 H1-L1 BBH 中具有三种 SNR 的完整 unordered pairs：{payload['n_pairs_complete']}。",
                "",
                "signed `log(rho_late/rho_early)` 的定义敏感性保存在 `real_gwtc_pair_snr_ratio_sensitivity.json`。这项比较只说明改用 PE optimal SNR 会怎样改变 pair observable，不把相关性当作定义等价证明。",
                "",
            ]
        )
    lines.extend(
        [
            "## 5. 当前注入的 closed-loop 程度",
            "",
            injection_summary.to_markdown(index=False, floatfmt=".6g"),
            "",
            "`target - recovered optimal SNR` 与 `realized unsigned ratio - GW-LMC proposal ratio` 若接近机器精度，说明该 ratio 是生成规则的一部分，而不是独立测试中由响应和噪声自然恢复出来的量。",
            "",
            "## 6. 时间先验的独立样本基数",
            "",
            f"GW-LMC Any_Detected_SNR8 表共有 {time_summary['source_catalog_rows']} 行，但满足至少两幅像 SNR>=8 且形成有效 delay 的只有 **{time_summary['eligible_independent_lens_systems']} 个独立 lens systems**，产生 **{time_summary['raw_detectable_image_pairs']} 个 image pairs**。100,000 次 exposure-conditioned/KDE 抽样只是这些有限系统的重采样，不是 100,000 个独立系统。",
            "",
            "## 7. 系统级隔离",
            "",
            split_audit.to_markdown(index=False, floatfmt=".4f"),
            "",
            "编码器 train/validation/test 的 `gwlmc_event_id` 交叉应为零。另一方面，若 validation/test 与当前全局时间先验源 ID 有交集，则当前 prior 不能直接用于新二维方法；新 density-development 集必须排除所有 validation/test system IDs。",
            "",
            "## 8. Family 名称",
            "",
            "ET-3 的 SIS/PM 是解析 SIS 与 point-mass。O3/O4a 的 `SIS_data_0222` 和 `PM_data_0222` 只是旧接口目录名，分别表示 GW-LMC smooth/non-subhalo 与 subhalo-present，不能在新增实验中称为解析 SIS/PM。详表见 `family_label_audit.csv`。",
            "",
            "## 9. Go/no-go 决定",
            "",
            "1. **当前 manifest SNR 用于真实重排：NO-GO。** 物理 estimand 与注入 optimal SNR 不同。",
            "2. **PE posterior optimal SNR：条件性候选。** 只有作者冻结 posterior group、覆盖范围和用途后才能进入 Phase 1。",
            "3. **当前逐像缩放注入：仅 closed-loop diagnostic。**",
            "4. **response-derived injection：正式主实验必需，但本轮未启动。**",
            "5. **Phase 1：等待作者决定。** 本审计没有自动跨越该门槛。",
            "",
            "## 10. 下一步若获批准",
            "",
            "先冻结真实目录使用 manifest matched-filter SNR 还是 PE optimal SNR；随后从 system-disjoint development systems 拟合二维 `(log10 Delta t, log(rho_late/rho_early))` 密度，并重建每个系统只有一个共同幅度因子的 response-derived injections。只有其在固定召回率或固定 Top-B 下稳定降低假对，才允许替换 v9.3 时间通道。",
            "",
            "## 11. 文件",
            "",
            "机器可读决定见 `phase0_go_no_go.json`，完整 provenance 见 CSV/JSON 表，复现命令见 `README.md`。",
        ]
    )
    (out / "phase0_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_checksums(out: Path) -> None:
    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "checksums_sha256.txt")
    content = "\n".join(f"{sha256(path)}  {path.relative_to(out)}" for path in files) + "\n"
    (out / "checksums_sha256.txt").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = resolve(root, args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        et_rows, et_summary = audit_et_snr()
        et_rows.to_csv(staging / "et3_network_snr_audit.csv", index=False)
        write_json(staging / "et3_network_snr_summary.json", et_summary)

        real_event_frames = []
        real_summary_rows: list[dict[str, Any]] = []
        pair_sensitivity: dict[str, Any] = {}
        injection_summaries = []
        injection_samples = []
        for deployment, config in DEPLOYMENTS.items():
            real_events, summaries, sensitivity = audit_real_snr(root, deployment, config)
            real_event_frames.append(real_events)
            real_summary_rows.extend(summaries)
            pair_sensitivity[deployment] = sensitivity
            injection_summary, injection_sample = audit_injection_snr(root, deployment, config)
            injection_summaries.append(injection_summary)
            injection_samples.append(injection_sample)

        real_events = pd.concat(real_event_frames, ignore_index=True)
        real_summary = pd.DataFrame(real_summary_rows)
        injection_summary = pd.concat(injection_summaries, ignore_index=True)
        injection_sample = pd.concat(injection_samples, ignore_index=True)
        real_events.to_csv(staging / "real_gwtc_snr_event_audit.csv", index=False)
        real_summary.to_csv(staging / "real_gwtc_snr_consistency_summary.csv", index=False)
        write_json(staging / "real_gwtc_pair_snr_ratio_sensitivity.json", pair_sensitivity)
        injection_summary.to_csv(staging / "injection_snr_audit_per_seed.csv", index=False)
        injection_sample.to_parquet(staging / "injection_snr_provenance_rows.parquet", index=False)

        prior_pairs, time_summary, membership, split_audit = audit_time_prior(root)
        prior_pairs.to_csv(staging / "gwlmc_snr8_detectable_delay_ratio_pairs.csv", index=False)
        write_json(staging / "time_prior_source_audit.json", time_summary)
        membership.to_parquet(staging / "gwlmc_system_split_membership.parquet", index=False)
        split_audit.to_csv(staging / "gwlmc_system_split_audit.csv", index=False)

        inventory = provenance_inventory(root)
        inventory.to_csv(staging / "snr_provenance_inventory.csv", index=False)
        families = family_label_audit(root)
        families.to_csv(staging / "family_label_audit.csv", index=False)

        decision = build_decision(real_summary, injection_summary, split_audit, time_summary)
        write_json(staging / "phase0_go_no_go.json", decision)
        contract = {
            "audit_name": "time-delay and relative observed strength Phase-0 provenance audit",
            "audit_date": AUDIT_DATE,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root),
            "baseline": "GWTC sky-resolution v9.3 plus frozen ET-3 inputs",
            "phase_completed": 0,
            "read_only_existing_results": True,
            "new_density_fitted": False,
            "real_candidates_reranked": False,
            "encoder_retrained": False,
            "paper_modified": False,
            "seeds_audited": list(SEEDS),
            "author_confirmation_required_before_phase1": True,
            "input_hashes": {
                str(path): sha256(path)
                for path in (GW_LMC_SNR1_IMAGE, GW_LMC_SNR1_SOURCE, GW_LMC_SNR8_IMAGE, GW_LMC_SNR8_SOURCE)
            },
        }
        write_json(staging / "analysis_contract_phase0.json", contract)
        make_report(staging, et_summary, real_summary, pair_sensitivity, injection_summary, time_summary, split_audit, decision)

        readme = f"""# Phase 0 SNR provenance audit

This directory is independent of v9.3 and contains no reranked candidate result.

Run from the repository root:

```bash
/root/miniconda3/bin/python scripts/experiments/130_time_snr_joint_evidence_phase0_audit.py \\
  --root /root/autodl-tmp/gw-catalog \\
  --output {DEFAULT_RESULT}
```

The script refuses to overwrite an existing output directory. Read
`phase0_report_cn.md` and `phase0_go_no_go.json` first.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        script_target = staging / "scripts/experiments"
        script_target.mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve(), script_target / Path(__file__).name)
        build_checksums(staging)
        os.replace(staging, output)
    except Exception:
        print(f"Audit failed; partial output retained at {staging}", file=sys.stderr)
        raise
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
