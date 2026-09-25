#!/usr/bin/env python3
"""Bounded Phase 0.5 for the time-delay/relative-strength study.

The script creates a new result tree. It never overwrites v9.3, never fits the
two-dimensional time/SNR density, and never reranks the real GWTC catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ["GW_WAVEFORM_INPUT_SAMPLES"] = "4096"

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from matchgw.matching import similarity_matrix
from scripts.real_search import physical_common as physical
from scripts.real_search import unified_v7_common as v7


DATE = "2026-08-07"
SEEDS = (202607241, 202607242, 202607243)
DEFAULT_OUTPUT = REPO / "results/time_snr_joint_evidence_phase05_20260807"
V7_ROOT = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
V93_ROOT = REPO / "results/gwtc_sky_resolution_v93_20260730"
SOURCE_ROOT = REPO / "results/real_noise_injection_v5_physical_source_20260721"
PAIR_ROWS = {
    "validation": "fusion_validation_pairs_v81.parquet",
    "test": "fusion_heldout_test_pairs_v81.parquet",
}
EVENT_ROWS = {
    "validation": "mixed_val_synthetic_events_v7.parquet",
    "test": "mixed_test_synthetic_events_v7.parquet",
}
MAP_ROWS = {
    "validation": "mixed_val_synthetic_sky_posteriors_nside64_v7.npy",
    "test": "mixed_test_synthetic_sky_posteriors_nside64_v7.npy",
}
DEPLOYMENTS = {
    "gwtc3": {
        "label": "GWTC-3.0/O3",
        "source_key": "GWTC3",
        "manifest": REPO / "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv",
        "group_column": "sky_map_internal_group",
    },
    "gwtc4": {
        "label": "GWTC-4.1/O4a",
        "source_key": "GWTC4",
        "manifest": REPO / "runs/real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest.csv",
        "group_column": "sky_map_group",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("all", "contract", "splits-time", "response", "report"),
        default="all",
    )
    parser.add_argument("--null-samples", type=int, default=250_000)
    return parser.parse_args()


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = module_from(REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py", "phase05_v3")
pilot = module_from(REPO / "scripts/real_search/37_unified_intrinsic_multitask_pilot.py", "phase05_pilot")


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
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def quantiles(values: Iterable[Any]) -> dict[str, Any]:
    array = finite(values)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def posterior_datasets(handle: h5py.File) -> list[str]:
    rows: list[str] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.endswith("posterior_samples"):
            if "network_optimal_snr" in set(obj.dtype.names or ()):
                rows.append(name)

    handle.visititems(visit)
    return sorted(rows)


def group_class_compatible(group: str, object_class: str) -> bool:
    is_nsbh = "NSBH" in group.upper() or "NRTIDAL" in group.upper()
    return is_nsbh if object_class != "BBH" else not is_nsbh


def choose_frozen_group(groups: list[str], requested: str, object_class: str) -> tuple[str | None, str]:
    requested_path = f"{requested}/posterior_samples" if requested else ""
    if requested_path in groups and group_class_compatible(requested_path, object_class):
        return requested_path, "exact_manifest_group"
    if object_class == "BBH":
        priority = ("Mixed+XO4a", "C01:Mixed/", "IMRPhenomXO4a", "IMRPhenomXPHM", "SEOBNRv5PHM", "SEOBNRv4PHM")
    else:
        priority = ("Mixed:NSBH:LowSpin", "IMRPhenomNSBH:LowSpin", "NRTidalv2_NSBH:LowSpin", "Mixed:NSBH:HighSpin", "IMRPhenomNSBH:HighSpin")
    compatible = [group for group in groups if group_class_compatible(group, object_class)]
    for token in priority:
        matches = sorted(group for group in compatible if token in group)
        if matches:
            return matches[0], f"frozen_class_priority:{token}"
    return (compatible[0], "frozen_lexical_class_fallback") if compatible else (None, "no_class_compatible_group")


def old_strict_event_names(deployment: str) -> set[str]:
    path = V93_ROOT / deployment / f"seed_{SEEDS[0]}/real_pair_features_unified_sky_v81.parquet"
    frame = pd.read_parquet(path, columns=["event_i", "event_j", "strict_h1l1_bbh_pair"])
    strict = frame[as_bool(frame["strict_h1l1_bbh_pair"])]
    return set(strict["event_i"].astype(str)).union(strict["event_j"].astype(str))


def object_class(deployment: str, row: pd.Series) -> str:
    if deployment == "gwtc4":
        value = str(row.get("object_class", "BBH"))
        return "BBH" if value == "BBH" else "NSBH_or_low_mass_OOD"
    mass_2 = float(row.get("mass_2", np.nan))
    if str(row["event_name"]) == "GW191219_163120" or (np.isfinite(mass_2) and mass_2 < 3.0):
        return "NSBH_or_low_mass_OOD"
    return "BBH"


def freeze_pe_contract(output: Path) -> pd.DataFrame:
    target = output / "contract/pe_network_optimal_snr_event_contract.csv"
    if target.exists():
        return pd.read_csv(target)
    rows: list[dict[str, Any]] = []
    for deployment, config in DEPLOYMENTS.items():
        manifest = pd.read_csv(config["manifest"])
        manifest = manifest[as_bool(manifest["include_in_primary_search"])].copy()
        old_strict = old_strict_event_names(deployment)
        for _, event in manifest.iterrows():
            classification = object_class(deployment, event)
            path = Path(str(event["sky_map_path"]))
            path = path if path.is_absolute() else REPO / path
            requested = str(event.get(config["group_column"], ""))
            record: dict[str, Any] = {
                "deployment": deployment,
                "event_name": str(event["event_name"]),
                "gps_time": float(event["gps_time"]),
                "object_class_phase05": classification,
                "pe_path": str(path),
                "requested_manifest_group": requested,
                "selected_posterior_dataset": "",
                "selection_reason": "",
                "network_optimal_snr_median": np.nan,
                "network_optimal_snr_q05": np.nan,
                "network_optimal_snr_q95": np.nan,
                "posterior_samples": 0,
                "old_v93_strict_event": str(event["event_name"]) in old_strict,
                "strict_h1l1_bbh_phase05": str(event["event_name"]) in old_strict and classification == "BBH",
                "error": "",
            }
            try:
                with h5py.File(path, "r") as handle:
                    selected, reason = choose_frozen_group(posterior_datasets(handle), requested, classification)
                    record["selection_reason"] = reason
                    if selected is None:
                        raise RuntimeError("No class-compatible posterior group with network_optimal_snr")
                    dataset = handle[selected]
                    values = finite(dataset["network_optimal_snr"])
                    values = values[values > 0]
                    if not len(values):
                        raise RuntimeError("Selected posterior has no finite positive network_optimal_snr")
                    record.update(
                        {
                            "selected_posterior_dataset": selected,
                            "network_optimal_snr_median": float(np.median(values)),
                            "network_optimal_snr_q05": float(np.quantile(values, 0.05)),
                            "network_optimal_snr_q95": float(np.quantile(values, 0.95)),
                            "posterior_samples": int(len(values)),
                        }
                    )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)
    frame = pd.DataFrame(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    gw191219 = frame[frame["event_name"] == "GW191219_163120"].iloc[0]
    summary = {
        "observable": "PE posterior network_optimal_snr",
        "summary_rule": "median of finite positive posterior samples",
        "uncertainty_audit": "q05 and q95 retained per event",
        "pair_observable": "r_rho = log(median_rho_late / median_rho_early), ordered by catalog geocentric GPS time",
        "group_rule": "Use exact manifest group only when it contains network_optimal_snr and matches the frozen BBH/NSBH class; otherwise use the predeclared within-class priority. Never cross class silently.",
        "strict_scope": "BBH and strict H1-L1 waveform availability; NSBH/low-mass OOD excluded",
        "coverage": {
            deployment: {
                "primary_events": int(len(part)),
                "strict_events": int(part["strict_h1l1_bbh_phase05"].sum()),
                "strict_finite_snr": int(np.isfinite(part.loc[part["strict_h1l1_bbh_phase05"], "network_optimal_snr_median"]).sum()),
            }
            for deployment, part in frame.groupby("deployment")
        },
        "gw191219_163120": gw191219.to_dict(),
        "gw191219_conclusion": "The event has a ~1.17 Msun secondary and NSBH PE groups. It is removed from the strict BBH scope and retained only as an OOD audit event.",
    }
    write_json(output / "contract/pe_network_optimal_snr_contract.json", summary)
    return frame


def source_mapping(deployment: str) -> pd.DataFrame:
    root = SOURCE_ROOT / deployment / "shared/physical_h1l1_source_bank"
    rows = []
    for family, relative, index_column, role in (
        ("SIS", "SIS_data_0222/physical_source_pair_metadata.parquet", "pair_id", "lensed"),
        ("PM", "PM_data_0222/physical_source_pair_metadata.parquet", "pair_id", "lensed"),
        ("unlensed", "Unlensed_data_0222/physical_unlensed_source_metadata.parquet", "sample_index", "unlensed"),
    ):
        frame = pd.read_parquet(root / relative)
        keep = [index_column, "gwlmc_row", "gwlmc_event_id"]
        if role == "lensed":
            keep += ["physical_lens_group", "delay_days", "gps_image1", "gps_image2", "proposal_snr_image1", "proposal_snr_image2", "proposal_snr_ratio"]
        else:
            keep += ["gps"]
        frame = frame[keep].rename(columns={index_column: "source_index"})
        frame["family"] = family
        frame["role"] = role
        frame["global_source_group_id"] = frame["gwlmc_event_id"].map(lambda value: f"GW-LMC-source:{int(value)}")
        frame["global_lens_system_id"] = frame["gwlmc_row"].map(lambda value: f"GW-LMC-lens-row:{int(value)}") if role == "lensed" else ""
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def existing_event_table(deployment: str, seed: int, split: str) -> pd.DataFrame:
    return pd.read_parquet(V7_ROOT / deployment / f"seed_{seed}/results/{EVENT_ROWS[split]}")


def build_split_contract(output: Path) -> pd.DataFrame:
    target = output / "splits/global_system_split_membership.parquet"
    if target.exists():
        return pd.read_parquet(target)
    all_rows = []
    summaries = []
    for deployment in DEPLOYMENTS:
        mapping = source_mapping(deployment)
        all_keys = set(zip(mapping["family"].astype(str), mapping["source_index"].astype(int)))
        for seed in SEEDS:
            split_keys: dict[str, set[tuple[str, int]]] = {}
            for split in ("validation", "test"):
                events = existing_event_table(deployment, seed, split)
                split_keys[split] = set(zip(events["family"].astype(str), events["source_index"].astype(int)))
            split_keys["train"] = all_keys - split_keys["validation"] - split_keys["test"]
            membership = mapping.copy()
            membership["original_split"] = [
                next(split for split in ("train", "validation", "test") if (str(family), int(index)) in split_keys[split])
                for family, index in zip(membership["family"], membership["source_index"])
            ]
            by_source = membership.groupby("global_source_group_id")["original_split"].agg(lambda values: set(values))
            corrected = []
            reasons = []
            for row in membership.itertuples(index=False):
                occupied = by_source[row.global_source_group_id]
                if row.original_split == "train":
                    corrected.append("train")
                    reasons.append("retained_train")
                elif "train" in occupied:
                    corrected.append("excluded")
                    reasons.append("source_group_present_in_train")
                elif row.original_split == "validation":
                    corrected.append("validation")
                    reasons.append("retained_validation")
                elif "validation" in occupied:
                    corrected.append("excluded")
                    reasons.append("source_group_present_in_validation")
                else:
                    corrected.append("test")
                    reasons.append("retained_test")
            membership["corrected_split"] = corrected
            membership["correction_reason"] = reasons
            membership.insert(0, "seed", seed)
            membership.insert(0, "deployment", deployment)
            all_rows.append(membership)
            retained = membership[membership["corrected_split"] != "excluded"]
            intersections = {}
            for left, right in itertools.combinations(("train", "validation", "test"), 2):
                left_ids = set(retained.loc[retained["corrected_split"] == left, "global_source_group_id"])
                right_ids = set(retained.loc[retained["corrected_split"] == right, "global_source_group_id"])
                intersections[f"{left}_{right}"] = len(left_ids & right_ids)
            summaries.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "train_rows": int((membership["corrected_split"] == "train").sum()),
                    "validation_rows": int((membership["corrected_split"] == "validation").sum()),
                    "test_rows": int((membership["corrected_split"] == "test").sum()),
                    "excluded_rows": int((membership["corrected_split"] == "excluded").sum()),
                    "density_development_lens_systems": int(((membership["corrected_split"] == "train") & (membership["role"] == "lensed")).sum()),
                    "density_development_unique_source_groups": int(membership.loc[(membership["corrected_split"] == "train") & (membership["role"] == "lensed"), "global_source_group_id"].nunique()),
                    "train_validation_source_intersection": intersections["train_validation"],
                    "train_test_source_intersection": intersections["train_test"],
                    "validation_test_source_intersection": intersections["validation_test"],
                }
            )
    result = pd.concat(all_rows, ignore_index=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    pd.DataFrame(summaries).to_csv(output / "splits/global_system_split_summary.csv", index=False)
    leak = result[(result["deployment"] == "gwtc4") & (result["seed"] == 202607241) & (result["gwlmc_event_id"] == 1568)]
    leak.to_csv(output / "splits/o4a_seed202607241_source1568_resolution.csv", index=False)
    return result


def method_weights(validation: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    selected: dict[str, dict[str, float]] = {}
    grids: dict[str, pd.DataFrame] = {}
    for objective in ("retrieval", "candidate"):
        time_sky, grid = v7.select_fusion_weights(validation, waveform_zero=True, objective=objective)
        selected[f"time_sky_{objective}_selected"] = time_sky
        grids[f"time_sky_{objective}"] = grid
        free, grid = v7.select_fusion_weights(validation, objective=objective)
        selected[f"{objective}_three_channel_unconstrained"] = free
        grids[f"{objective}_three_channel_unconstrained"] = grid
        positive, grid = v7.select_fusion_weights(validation, require_all_positive=True, objective=objective)
        selected[f"{objective}_three_channel_strict_positive"] = positive
        grids[f"{objective}_three_channel_strict_positive"] = grid
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        **selected,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
    }
    return methods, grids


def corrected_event_indices(membership: pd.DataFrame, deployment: str, seed: int, split: str) -> set[tuple[str, int]]:
    subset = membership[
        (membership["deployment"] == deployment)
        & (membership["seed"] == seed)
        & (membership["corrected_split"] == split)
    ]
    return set(zip(subset["family"].astype(str), subset["source_index"].astype(int)))


def filter_pair_frame(frame: pd.DataFrame, events: pd.DataFrame, keep_keys: set[tuple[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    event_keep = np.asarray(
        [(str(family), int(index)) in keep_keys for family, index in zip(events["family"], events["source_index"])],
        dtype=bool,
    )
    old_indices = np.flatnonzero(event_keep)
    remap = np.full(len(events), -1, dtype=np.int32)
    remap[old_indices] = np.arange(len(old_indices), dtype=np.int32)
    ii = frame["idx_i"].to_numpy(dtype=np.int32)
    jj = frame["idx_j"].to_numpy(dtype=np.int32)
    keep_pair = event_keep[ii] & event_keep[jj]
    filtered = frame.loc[keep_pair].copy()
    filtered["idx_i"] = remap[filtered["idx_i"].to_numpy(dtype=np.int32)]
    filtered["idx_j"] = remap[filtered["idx_j"].to_numpy(dtype=np.int32)]
    filtered["event_count"] = int(len(old_indices))
    filtered_events = events.iloc[old_indices].copy().reset_index(drop=True)
    filtered_events["idx"] = np.arange(len(filtered_events), dtype=np.int32)
    return filtered.reset_index(drop=True), filtered_events, old_indices


def fit_time_calibrations(output: Path, membership: pd.DataFrame, null_samples: int) -> None:
    marker = output / "time/corrected_time_calibration_summary.csv"
    if marker.exists():
        return
    rows = []
    baseline_retrieval = []
    baseline_pair = []
    for deployment, config in DEPLOYMENTS.items():
        schedule = v3.build_live_schedule(
            config["source_key"],
            v3.SOURCES[config["source_key"]],
            SOURCE_ROOT / deployment / "shared",
        )
        for seed in SEEDS:
            seed_out = output / "time" / deployment / f"seed_{seed}"
            seed_out.mkdir(parents=True, exist_ok=True)
            dev = membership[
                (membership["deployment"] == deployment)
                & (membership["seed"] == seed)
                & (membership["corrected_split"] == "train")
                & (membership["role"] == "lensed")
            ].drop_duplicates("global_lens_system_id")
            delays = dev["delay_days"].to_numpy(dtype=np.float64)
            rng = np.random.default_rng(seed + 500_000)
            null = physical.draw_null_delays(schedule, null_samples, rng)
            calibration = physical.fit_time_likelihood_ratio(delays, null, grid_size=2048, bandwidth_scale=1.0)
            calibration.update(
                {
                    "definition": "Leakage-free log p(delta_t|lensed,run) - log p(delta_t|null,run)",
                    "density_development_source": "Corrected train split of the run-matched GW-LMC physical source bank",
                    "global_grouping": "All realizations sharing gwlmc_event_id are kept in one split; each gwlmc_row contributes once",
                    "validation_or_test_systems_used": False,
                    "bootstrap_expansion_used": False,
                    "bandwidth_scale_frozen": 1.0,
                }
            )
            write_json(seed_out / "time_likelihood_ratio.json", calibration)
            np.save(seed_out / "density_development_delays_days.npy", delays.astype(np.float32))
            np.save(seed_out / "null_delays_days.npy", null.astype(np.float32))
            rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "lens_systems": len(delays),
                    "unique_source_groups": dev["global_source_group_id"].nunique(),
                    "smooth_non_subhalo": int((dev["family"] == "SIS").sum()),
                    "subhalo_present": int((dev["family"] == "PM").sum()),
                    "null_pairs": len(null),
                    "delay_median_days": float(np.median(delays)),
                    "delay_q95_days": float(np.quantile(delays, 0.95)),
                }
            )

            split_frames: dict[str, pd.DataFrame] = {}
            for split in ("validation", "test"):
                events = existing_event_table(deployment, seed, split)
                base = pd.read_parquet(V93_ROOT / deployment / f"seed_{seed}/{PAIR_ROWS[split]}")
                keep_keys = corrected_event_indices(membership, deployment, seed, split)
                frame, kept_events, _ = filter_pair_frame(base, events, keep_keys)
                frame["time_score"] = physical.apply_time_likelihood_ratio(frame["delta_t_days"], calibration)
                frame.to_parquet(seed_out / f"closed_loop_{split}_pairs_corrected_time.parquet", index=False)
                kept_events.to_parquet(seed_out / f"closed_loop_{split}_events_corrected_split.parquet", index=False)
                split_frames[split] = frame
            methods, grids = method_weights(split_frames["validation"])
            write_json(seed_out / "closed_loop_selected_weights.json", methods)
            for name, grid in grids.items():
                grid.to_csv(seed_out / f"closed_loop_weight_grid_{name}.csv", index=False)
            _, retrieval, pair = v7.evaluation_tables(split_frames["test"], methods, deployment, seed)
            retrieval["protocol"] = "closed_loop_corrected_time_phase05"
            pair["protocol"] = "closed_loop_corrected_time_phase05"
            retrieval.to_csv(seed_out / "closed_loop_heldout_retrieval_metrics.csv", index=False)
            pair.to_csv(seed_out / "closed_loop_heldout_pair_metrics.csv", index=False)
            baseline_retrieval.append(retrieval)
            baseline_pair.append(pair)
    pd.DataFrame(rows).to_csv(marker, index=False)
    pd.concat(baseline_retrieval, ignore_index=True).to_csv(output / "time/closed_loop_corrected_time_retrieval_per_seed.csv", index=False)
    pd.concat(baseline_pair, ignore_index=True).to_csv(output / "time/closed_loop_corrected_time_pair_metrics_per_seed.csv", index=False)


@dataclass
class ResponseResources:
    deployment: str
    references: np.ndarray
    psds: np.ndarray
    psd_frequency: np.ndarray
    noise_manifest: pd.DataFrame
    clean: dict[str, tuple[np.ndarray, np.ndarray]]


def load_response_resources(deployment: str) -> ResponseResources:
    shared = SOURCE_ROOT / deployment / "shared"
    bank = shared / "physical_h1l1_source_bank"
    clean = {}
    for family in ("SIS", "PM"):
        root = bank / f"{family}_data_0222"
        clean[family] = (
            np.load(root / f"{family}_h_strain_1.npy", mmap_mode="r"),
            np.load(root / f"{family}_h_strain_2.npy", mmap_mode="r"),
        )
    return ResponseResources(
        deployment=deployment,
        references=np.load(shared / "noise_reference_bank.npy", mmap_mode="r"),
        psds=np.load(shared / "noise_psd_bank.npy", mmap_mode="r"),
        psd_frequency=np.load(shared / "noise_psd_frequency.npy"),
        noise_manifest=pd.read_csv(shared / "noise_bank_manifest.csv"),
        clean=clean,
    )


def choose_psd_index(resources: ResponseResources, gps: float, rng: np.random.Generator, neighbours: int = 8) -> int:
    starts = resources.noise_manifest["reference_start_gps"].to_numpy(dtype=np.float64)
    nearest = np.argsort(np.abs(starts - float(gps)))[: min(neighbours, len(starts))]
    return int(rng.choice(nearest))


def noise_window(resources: ResponseResources, bank_index: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    samples = physical.RAW_PADDED_SAMPLES
    width = resources.references.shape[-1]
    offset = int(rng.integers(0, width - samples + 1))
    return np.asarray(resources.references[bank_index, :, offset : offset + samples], dtype=np.float32), offset


def empirical_optimal_snrs(pe_contract: pd.DataFrame, deployment: str) -> np.ndarray:
    subset = pe_contract[
        (pe_contract["deployment"] == deployment)
        & as_bool(pe_contract["strict_h1l1_bbh_phase05"])
    ]
    values = finite(subset["network_optimal_snr_median"])
    values = values[(values >= 8.0) & (values <= 60.0)]
    if len(values) < 20:
        raise RuntimeError(f"Too few frozen real optimal SNRs for {deployment}: {len(values)}")
    return values


def response_system(
    resources: ResponseResources,
    row: pd.Series,
    empirical_snr: np.ndarray,
    rng: np.random.Generator,
    materialize_waveform: bool,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    family = str(row["family"])
    index = int(row["source_index"])
    clean1 = np.asarray(resources.clean[family][0][index], dtype=np.float32)
    clean2 = np.asarray(resources.clean[family][1][index], dtype=np.float32)
    bank1 = choose_psd_index(resources, float(row["gps_image1"]), rng)
    bank2 = choose_psd_index(resources, float(row["gps_image2"]), rng)
    psd1 = np.asarray(resources.psds[bank1], dtype=np.float64)
    psd2 = np.asarray(resources.psds[bank2], dtype=np.float64)
    raw1 = physical.optimal_network_snr(clean1, resources.psd_frequency, psd1)
    raw2 = physical.optimal_network_snr(clean2, resources.psd_frequency, psd2)
    if min(raw1, raw2) <= 0 or not np.isfinite([raw1, raw2]).all():
        raise RuntimeError(f"Invalid raw SNR for {family}:{index}")
    target_faint = float(rng.choice(empirical_snr))
    common_scale = target_faint / min(raw1, raw2)
    scaled1 = clean1 * common_scale
    scaled2 = clean2 * common_scale
    recovered1 = physical.optimal_network_snr(scaled1, resources.psd_frequency, psd1)
    recovered2 = physical.optimal_network_snr(scaled2, resources.psd_frequency, psd2)
    proposal_signed = math.log(float(row["proposal_snr_image2"]) / float(row["proposal_snr_image1"]))
    response_signed = math.log(recovered2 / recovered1)
    record = {
        "deployment": resources.deployment,
        "family": family,
        "physical_lens_group": str(row["physical_lens_group"]),
        "source_index": index,
        "gwlmc_row": int(row["gwlmc_row"]),
        "gwlmc_event_id": int(row["gwlmc_event_id"]),
        "global_source_group_id": str(row["global_source_group_id"]),
        "global_lens_system_id": str(row["global_lens_system_id"]),
        "gps_image1": float(row["gps_image1"]),
        "gps_image2": float(row["gps_image2"]),
        "delay_days": float(row["delay_days"]),
        "psd_bank_index_image1": bank1,
        "psd_bank_index_image2": bank2,
        "raw_optimal_snr_image1": raw1,
        "raw_optimal_snr_image2": raw2,
        "target_faint_optimal_snr": target_faint,
        "common_system_scale_factor": common_scale,
        "recovered_optimal_snr_image1": recovered1,
        "recovered_optimal_snr_image2": recovered2,
        "proposal_signed_log_snr_ratio": proposal_signed,
        "response_signed_log_snr_ratio": response_signed,
        "response_unsigned_snr_ratio": max(recovered1, recovered2) / min(recovered1, recovered2),
        "common_scale_for_both_images": True,
    }
    if not materialize_waveform:
        return record, None, None
    noise1, offset1 = noise_window(resources, bank1, rng)
    noise2, offset2 = noise_window(resources, bank2, rng)
    padded1 = physical.embed_signal_in_padded_window(scaled1)
    padded2 = physical.embed_signal_in_padded_window(scaled2)
    mixed1 = physical.preprocess_24s(noise1 + padded1, resources.psd_frequency, psd1)
    mixed2 = physical.preprocess_24s(noise2 + padded2, resources.psd_frequency, psd2)
    record["noise_offset_samples_image1"] = offset1
    record["noise_offset_samples_image2"] = offset2
    record["prepared_waveform_samples"] = 4096
    return record, mixed1[..., -4096:].astype(np.float32), mixed2[..., -4096:].astype(np.float32)


class WaveformDataset(Dataset):
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(pilot.prepare(self.values[index], None, False))


def checkpoint_path(deployment: str, seed: int) -> Path:
    matches = list((V7_ROOT / deployment / f"seed_{seed}/waveform_gate").glob("unified_*_v7/validation_selected_model.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen checkpoint for {deployment}/{seed}, got {matches}")
    return matches[0]


def true_pair_labels(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(len(events), k=1)
    family_i = events["family"].astype(str).to_numpy()[ii]
    family_j = events["family"].astype(str).to_numpy()[jj]
    pair_i = events["pair_id"].to_numpy(dtype=np.int64)[ii]
    pair_j = events["pair_id"].to_numpy(dtype=np.int64)[jj]
    tag_i = events["tag"].astype(str).to_numpy()[ii]
    tag_j = events["tag"].astype(str).to_numpy()[jj]
    labels = (family_i == family_j) & (family_i != "unlensed") & (pair_i == pair_j) & (tag_i != tag_j)
    return ii, jj, labels


def build_response_catalog(
    output: Path,
    deployment: str,
    seed: int,
    split: str,
    membership: pd.DataFrame,
    pe_contract: pd.DataFrame,
    resources: ResponseResources,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    seed_out = output / "response" / deployment / f"seed_{seed}"
    event_path = seed_out / f"response_{split}_events.parquet"
    waveform_path = seed_out / f"response_{split}_waveforms_peak2s.npy"
    map_path = seed_out / f"response_{split}_sky_posteriors_nside64.npy"
    metadata_path = seed_out / f"response_{split}_system_metadata.parquet"
    if all(path.exists() for path in (event_path, waveform_path, map_path, metadata_path)):
        return pd.read_parquet(event_path), np.load(waveform_path, mmap_mode="r"), np.load(map_path, mmap_mode="r"), pd.read_parquet(metadata_path)
    original_events = existing_event_table(deployment, seed, split)
    keep_keys = corrected_event_indices(membership, deployment, seed, split)
    keep = np.asarray(
        [(str(family), int(index)) in keep_keys for family, index in zip(original_events["family"], original_events["source_index"])],
        dtype=bool,
    )
    old_indices = np.flatnonzero(keep)
    events = original_events.iloc[old_indices].copy().reset_index(drop=True)
    events["idx"] = np.arange(len(events), dtype=np.int32)
    original_maps = np.load(V7_ROOT / deployment / f"seed_{seed}/results/{MAP_ROWS[split]}", mmap_mode="r")
    maps = np.asarray(original_maps[old_indices], dtype=np.float32)

    mapping = membership[
        (membership["deployment"] == deployment)
        & (membership["seed"] == seed)
        & (membership["corrected_split"] == split)
    ].set_index(["family", "source_index"])
    empirical = empirical_optimal_snrs(pe_contract, deployment)
    rng = np.random.default_rng(seed + (710_000 if split == "validation" else 720_000))
    response_waveforms: dict[tuple[str, str, int], np.ndarray] = {}
    system_rows = []
    for family in ("SIS", "PM"):
        source_indices = sorted(events.loc[events["family"] == family, "source_index"].astype(int).unique())
        for number, source_index in enumerate(source_indices, start=1):
            row = mapping.loc[(family, source_index)].copy()
            # The lookup keys are removed from the Series by set_index(), but the
            # response builder needs them to select the frozen physical waveforms.
            row["family"] = family
            row["source_index"] = source_index
            record, first, second = response_system(resources, row, empirical, rng, materialize_waveform=True)
            assert first is not None and second is not None
            response_waveforms[(family, "L1", source_index)] = first
            response_waveforms[(family, "L2", source_index)] = second
            record["split"] = split
            system_rows.append(record)
            if number % 50 == 0:
                print(f"[{deployment}/{seed}/{split}] {family} response systems {number}/{len(source_indices)}", flush=True)

    original_data = v7.load_mixed_data(
        V7_ROOT / deployment / f"seed_{seed}",
        SOURCE_ROOT / deployment / "shared/physical_h1l1_source_bank",
        seed,
        600,
    )
    waveforms = np.empty((len(events), 2, 4096), dtype=np.float32)
    response_meta = pd.DataFrame(system_rows)
    by_system = response_meta.set_index(["family", "source_index"])
    for index, event in events.iterrows():
        family = str(event["family"])
        source_index = int(event["source_index"])
        tag = str(event["tag"])
        if family == "unlensed":
            waveforms[index] = np.asarray(original_data["SIS"].arrays.unlensed[source_index, :, -4096:], dtype=np.float32)
            continue
        waveforms[index] = response_waveforms[(family, tag, source_index)]
        image = 2 if tag == "L2" else 1
        events.loc[index, "snr"] = float(by_system.loc[(family, source_index), f"recovered_optimal_snr_image{image}"])
    seed_out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(event_path, index=False)
    np.save(waveform_path, waveforms)
    np.save(map_path, maps)
    response_meta.to_parquet(metadata_path, index=False)
    return events, waveforms, maps, response_meta


def raw_response_pair_table(
    events: pd.DataFrame,
    waveforms: np.ndarray,
    maps: np.ndarray,
    checkpoint: Path,
    calibration: dict[str, Any],
) -> pd.DataFrame:
    model, payload = v7.load_unified_model(checkpoint)
    embeddings, predictions = v7.embed_catalog(model, WaveformDataset(np.asarray(waveforms)), batch_size=24)
    del model
    torch.cuda.empty_cache()
    waveform = similarity_matrix(embeddings)
    sky_log_bf, sky_raw, sky_cosine = physical.sky_log_bayes_factor_from_maps(np.asarray(maps))
    ii, jj, labels = true_pair_labels(events)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    delta = np.abs(gps[ii] - gps[jj]) / physical.SECONDS_PER_DAY
    pair_family = np.where(labels, events["family"].astype(str).to_numpy()[ii], "background")
    frame = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "is_true_pair": labels.astype(np.int8),
            "true_pair_family": pair_family,
            "waveform_embedding_cosine": waveform[ii, jj].astype(np.float32),
            "waveform_pred_logmc_std_i": predictions[ii, 0].astype(np.float32),
            "waveform_pred_logmc_std_j": predictions[jj, 0].astype(np.float32),
            "waveform_pred_logitq_std_i": predictions[ii, 1].astype(np.float32),
            "waveform_pred_logitq_std_j": predictions[jj, 1].astype(np.float32),
            "waveform_abs_delta_logmc_std": np.abs(predictions[ii, 0] - predictions[jj, 0]).astype(np.float32),
            "waveform_abs_delta_logitq_std": np.abs(predictions[ii, 1] - predictions[jj, 1]).astype(np.float32),
            "delta_t_days": delta,
            "time_score": physical.apply_time_likelihood_ratio(delta, calibration),
            "sky_score": sky_log_bf[ii, jj].astype(np.float32),
            "sky_bayes_factor": np.exp(np.clip(sky_log_bf[ii, jj], -80, 80)),
            "sky_raw_overlap": sky_raw[ii, jj].astype(np.float64),
            "sky_cosine_overlap": sky_cosine[ii, jj].astype(np.float32),
            "event_count": int(len(events)),
        }
    )
    frame.attrs["checkpoint"] = str(checkpoint)
    frame.attrs["target_mean"] = json_ready(payload.get("target_mean"))
    frame.attrs["target_std"] = json_ready(payload.get("target_std"))
    return frame


def generate_density_development(
    output: Path,
    deployment: str,
    seed: int,
    membership: pd.DataFrame,
    pe_contract: pd.DataFrame,
    resources: ResponseResources,
) -> pd.DataFrame:
    path = output / "response" / deployment / f"seed_{seed}/density_development_response_metadata.parquet"
    if path.exists():
        return pd.read_parquet(path)
    dev = membership[
        (membership["deployment"] == deployment)
        & (membership["seed"] == seed)
        & (membership["corrected_split"] == "train")
        & (membership["role"] == "lensed")
    ].drop_duplicates("global_lens_system_id")
    empirical = empirical_optimal_snrs(pe_contract, deployment)
    rng = np.random.default_rng(seed + 700_000)
    rows = []
    for number, (_, row) in enumerate(dev.iterrows(), start=1):
        record, _, _ = response_system(resources, row, empirical, rng, materialize_waveform=False)
        record["split"] = "density_development"
        rows.append(record)
        if number % 200 == 0:
            print(f"[{deployment}/{seed}] density response systems {number}/{len(dev)}", flush=True)
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def response_audit_rows(frame: pd.DataFrame, deployment: str, seed: int, split: str) -> dict[str, Any]:
    proposal = frame["proposal_signed_log_snr_ratio"].to_numpy(dtype=np.float64)
    response = frame["response_signed_log_snr_ratio"].to_numpy(dtype=np.float64)
    correlation = float(spearmanr(proposal, response).statistic) if len(frame) >= 3 else np.nan
    return {
        "deployment": deployment,
        "seed": seed,
        "split": split,
        "n_lens_systems": len(frame),
        "n_unique_source_groups": frame["global_source_group_id"].nunique(),
        "smooth_non_subhalo": int((frame["family"] == "SIS").sum()),
        "subhalo_present": int((frame["family"] == "PM").sum()),
        "minimum_faint_snr": float(np.minimum(frame["recovered_optimal_snr_image1"], frame["recovered_optimal_snr_image2"]).min()),
        "fraction_bright_snr_above_60": float((np.maximum(frame["recovered_optimal_snr_image1"], frame["recovered_optimal_snr_image2"]) > 60).mean()),
        "proposal_response_signed_ratio_spearman": correlation,
        "median_abs_response_minus_proposal_log_ratio": float(np.median(np.abs(response - proposal))),
        "max_abs_response_minus_proposal_log_ratio": float(np.max(np.abs(response - proposal))),
        "all_rows_use_one_common_scale": bool(frame["common_scale_for_both_images"].all()),
    }


def run_response(output: Path, membership: pd.DataFrame, pe_contract: pd.DataFrame) -> None:
    marker = output / "response/response_derived_audit_summary.csv"
    if marker.exists():
        return
    audit_rows = []
    retrieval_rows = []
    pair_rows = []
    for deployment in DEPLOYMENTS:
        resources = load_response_resources(deployment)
        for seed in SEEDS:
            density = generate_density_development(output, deployment, seed, membership, pe_contract, resources)
            audit_rows.append(response_audit_rows(density, deployment, seed, "density_development"))
            calibration = json.loads((output / f"time/{deployment}/seed_{seed}/time_likelihood_ratio.json").read_text(encoding="utf-8"))
            split_raw = {}
            for split in ("validation", "test"):
                events, waveforms, maps, system_meta = build_response_catalog(
                    output, deployment, seed, split, membership, pe_contract, resources
                )
                audit_rows.append(response_audit_rows(system_meta, deployment, seed, split))
                raw_path = output / f"response/{deployment}/seed_{seed}/response_{split}_pairs_raw.parquet"
                if raw_path.exists():
                    raw = pd.read_parquet(raw_path)
                else:
                    raw = raw_response_pair_table(events, waveforms, maps, checkpoint_path(deployment, seed), calibration)
                    raw.to_parquet(raw_path, index=False)
                split_raw[split] = raw
            waveform_config, waveform_grid = v7.fit_waveform_channel(split_raw["validation"], allow_q_feature=True)
            validation = v7.apply_waveform_channel(split_raw["validation"], waveform_config)
            test = v7.apply_waveform_channel(split_raw["test"], waveform_config)
            seed_out = output / "response" / deployment / f"seed_{seed}"
            write_json(seed_out / "response_waveform_channel_calibration.json", waveform_config)
            waveform_grid.to_csv(seed_out / "response_waveform_grid_validation.csv", index=False)
            validation.to_parquet(seed_out / "response_validation_pairs_calibrated.parquet", index=False)
            test.to_parquet(seed_out / "response_heldout_test_pairs_calibrated.parquet", index=False)
            methods, grids = method_weights(validation)
            write_json(seed_out / "response_selected_weights.json", methods)
            for name, grid in grids.items():
                grid.to_csv(seed_out / f"response_weight_grid_{name}.csv", index=False)
            _, retrieval, pair = v7.evaluation_tables(test, methods, deployment, seed)
            retrieval["protocol"] = "response_derived_phase05"
            pair["protocol"] = "response_derived_phase05"
            retrieval.to_csv(seed_out / "response_heldout_retrieval_metrics.csv", index=False)
            pair.to_csv(seed_out / "response_heldout_pair_metrics.csv", index=False)
            retrieval_rows.append(retrieval)
            pair_rows.append(pair)
    pd.DataFrame(audit_rows).to_csv(marker, index=False)
    pd.concat(retrieval_rows, ignore_index=True).to_csv(output / "response/response_heldout_retrieval_per_seed.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(output / "response/response_heldout_pair_metrics_per_seed.csv", index=False)


def summarize_metrics(frame: pd.DataFrame, metrics: list[str], groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        row["n_seeds"] = part["seed"].nunique()
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def finalize(output: Path) -> dict[str, Any]:
    (output / "summary").mkdir(parents=True, exist_ok=True)
    pe = pd.read_csv(output / "contract/pe_network_optimal_snr_event_contract.csv")
    split = pd.read_csv(output / "splits/global_system_split_summary.csv")
    time_summary = pd.read_csv(output / "time/corrected_time_calibration_summary.csv")
    response_audit = pd.read_csv(output / "response/response_derived_audit_summary.csv")
    closed_retrieval = pd.read_csv(output / "time/closed_loop_corrected_time_retrieval_per_seed.csv")
    response_retrieval = pd.read_csv(output / "response/response_heldout_retrieval_per_seed.csv")
    closed_pair = pd.read_csv(output / "time/closed_loop_corrected_time_pair_metrics_per_seed.csv")
    response_pair = pd.read_csv(output / "response/response_heldout_pair_metrics_per_seed.csv")
    retrieval_summary = pd.concat(
        [
            summarize_metrics(closed_retrieval, ["r_at_1", "r_at_10", "median_rank"], ["protocol", "deployment", "method", "subset"]),
            summarize_metrics(response_retrieval, ["r_at_1", "r_at_10", "median_rank"], ["protocol", "deployment", "method", "subset"]),
        ],
        ignore_index=True,
    )
    pair_summary = pd.concat(
        [
            summarize_metrics(closed_pair, ["average_precision", "false_at_recall_0p5", "false_at_recall_0p9"], ["protocol", "deployment", "method"]),
            summarize_metrics(response_pair, ["average_precision", "false_at_recall_0p5", "false_at_recall_0p9"], ["protocol", "deployment", "method"]),
        ],
        ignore_index=True,
    )
    retrieval_summary.to_csv(output / "summary/retrieval_metrics_summary.csv", index=False)
    pair_summary.to_csv(output / "summary/pair_metrics_summary.csv", index=False)

    strict = pe[as_bool(pe["strict_h1l1_bbh_phase05"])]
    pe_pass = bool(strict["error"].fillna("").eq("").all() and np.isfinite(strict["network_optimal_snr_median"]).all())
    split_pass = bool(
        (
            split[
                ["train_validation_source_intersection", "train_test_source_intersection", "validation_test_source_intersection"]
            ]
            == 0
        ).all().all()
    )
    density_pass = bool(
        (time_summary["lens_systems"] >= 500).all()
        and (time_summary["smooth_non_subhalo"] >= 200).all()
        and (time_summary["subhalo_present"] >= 200).all()
    )
    response_eval = response_audit[response_audit["split"].isin(["validation", "test"])]
    response_pass = bool(
        response_eval["all_rows_use_one_common_scale"].all()
        and (response_eval["minimum_faint_snr"] >= 8.0 - 1e-6).all()
        and (response_eval["n_lens_systems"] >= 170).all()
        and (response_eval["median_abs_response_minus_proposal_log_ratio"] > 0.02).all()
    )
    time_pass = bool(np.isfinite(time_summary["delay_median_days"]).all())
    checks = {
        "pe_contract_and_strict_scope": pe_pass,
        "global_source_group_split_isolation": split_pass,
        "density_development_population": density_pass,
        "leakage_free_one_dimensional_time_baseline": time_pass,
        "response_derived_common_scale_injections": response_pass,
    }
    go = all(checks.values())
    decision = {
        "phase": "Phase 0.5",
        "checks": checks,
        "decision": "GO_FOR_AUTHOR_REVIEW_BEFORE_BOUNDED_PHASE1" if go else "NO_GO_REMEDIATE_PHASE05",
        "phase1_started": False,
        "real_catalog_reranked": False,
        "two_dimensional_density_fitted": False,
        "paper_modified": False,
        "gw191219_163120": "Excluded from strict BBH scope; retained as NSBH/low-mass OOD audit only.",
        "source1568": "The O4a seed 202607241 PM test realization is removed because the same global source ID is present in train; no post-hoc replacement was made.",
        "author_action": "Review and explicitly approve the frozen PE-optimal-SNR contract and Phase 0.5 evidence before any two-dimensional Phase 1 fit.",
    }
    write_json(output / "summary/phase05_go_no_go.json", decision)
    make_report(output, decision, pe, split, time_summary, response_audit, retrieval_summary, pair_summary)
    return decision


def make_report(
    output: Path,
    decision: dict[str, Any],
    pe: pd.DataFrame,
    split: pd.DataFrame,
    time_summary: pd.DataFrame,
    response_audit: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> None:
    focus_methods = ["waveform_only", "time_only", "sky_only", "retrieval_three_channel_strict_positive"]
    response_focus = retrieval_summary[
        (retrieval_summary["protocol"] == "response_derived_phase05")
        & (retrieval_summary["subset"] == "overall")
        & (retrieval_summary["method"].isin(focus_methods))
    ]
    lines = [
        "# 时间延迟与相对观测强度联合证据 Phase 0.5 报告",
        "",
        f"日期：{DATE}",
        "",
        "## 1. 边界",
        "",
        "本轮冻结真实 PE optimal-SNR 契约、修复全局 source-group 切分、重建一维时间基线，并生成 response-derived 注入。没有拟合二维联合密度、没有重排真实候选、没有修改 v9.3 或论文。",
        "",
        "## 2. 决定",
        "",
        f"**{decision['decision']}**",
        "",
        pd.DataFrame([{"check": key, "passed": value} for key, value in decision["checks"].items()]).to_markdown(index=False),
        "",
        "## 3. 冻结的真实 SNR 规则",
        "",
        "每个事件使用预先确定、与其 BBH/NSBH class 一致的 PE posterior group。观测量是该 group 中有限正值 `network_optimal_snr` 的 posterior median；q05/q95 只作不确定度审计。pair 的 `r_rho` 按 catalog GPS 排 early/late 后计算 `log(rho_late/rho_early)`。",
        "",
        pe.groupby(["deployment", "object_class_phase05", "strict_h1l1_bbh_phase05"]).size().rename("events").reset_index().to_markdown(index=False),
        "",
        "GW191219_163120 的次级质量约 1.17 Msun，且公开 PE 含明确 NSBH groups。旧 manifest 把它写成 BBH 是 scope bug；Phase 0.5 将其排除出 strict BBH，但保留为 OOD audit。",
        "",
        "## 4. 全局切分",
        "",
        split.to_markdown(index=False),
        "",
        "分组键使用 GW-LMC `event_id`，lens realization 使用 `gwlmc_row`。同一 source 的所有 realization 不得跨 train/validation/test。为保持冻结 encoder，冲突 held-out realization 被删除而不是用 train source 事后替换。",
        "",
        "## 5. 无泄漏一维时间先验",
        "",
        time_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "每个 development lens row 只贡献一次，没有将 bootstrap draws 计作独立系统；null 来自同一 run live schedule。固定 bandwidth scale=1，未查看 test 选择带宽。",
        "",
        "## 6. Response-derived 注入",
        "",
        response_audit.to_markdown(index=False, floatfmt=".4g"),
        "",
        "两幅像的物理 H1/L1 strain 已包含 GW-LMC magnification、Morse phase 和各自到达时刻的 detector response。脚本先用各自 PSD 得到未缩放 optimal SNR，再对两像施加完全相同的 common factor；最终 ratio 因此由 magnification、response 和 PSD 自然产生。共同 factor 只改变共享源距离，不逐像指定 ratio。",
        "",
        "## 7. Response-derived held-out 基线",
        "",
        response_focus.to_markdown(index=False, floatfmt=".4f"),
        "",
        "这些指标只用于确认修正后的 v9.3 三通道基线可运行，不是二维联合证据的结果。二维模型仍未拟合。完整 retrieval/pair 指标见 `summary/`。",
        "",
        "## 8. 下一决策点",
        "",
        "只有作者审核 `summary/phase05_go_no_go.json`、PE group contract、response ratio audit 和 held-out 基线后，才能另行授权 Phase 1。Phase 1 仍须只在 development/validation 上冻结二维密度、支持域和权重。",
    ]
    (output / "phase05_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.csv", "checksums_sha256.txt"}:
            rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "artifact_manifest.csv", index=False)
    (output / "checksums_sha256.txt").write_text(
        "\n".join(f"{row.sha256}  {row.path}" for row in frame.itertuples(index=False)) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.stage == "all" and output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 0.5 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(
        output / "analysis_contract_phase05.json",
        {
            "phase": "0.5",
            "date": DATE,
            "baseline": "GWTC sky-resolution v9.3",
            "encoder_retrained": False,
            "waveform_and_sky_definition_frozen": True,
            "two_dimensional_density_fitted": False,
            "real_catalog_reranked": False,
            "v93_overwritten": False,
            "seeds": list(SEEDS),
            "response_amplitude_rule": "one common multiplicative strain factor per two-image system",
        },
    )
    pe = freeze_pe_contract(output)
    if args.stage == "contract":
        return 0
    membership = build_split_contract(output)
    fit_time_calibrations(output, membership, args.null_samples)
    if args.stage == "splits-time":
        return 0
    run_response(output, membership, pe)
    if args.stage == "response":
        return 0
    decision = finalize(output)
    script_dir = output / "scripts/experiments"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), script_dir / Path(__file__).name)
    write_json(
        output / "run_summary.json",
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "decision": decision["decision"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    build_manifest(output)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
