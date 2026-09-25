#!/usr/bin/env python3
"""Run only G-1/G0/G0.5 planning for the Nside=512 v10 addendum.

This script is deliberately incapable of launching injections, PE, calibration,
candidate reranking, or G1-G5. Dense HEALPix maps exist only in process memory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENTS = {
    "GWTC3_O3": {
        "short": "gwtc3",
        "source_key": "GWTC3",
        "real_run": "runs/real_gwtc_lensing_search_20260625",
        "source_bank": "results/real_noise_injection_v5_physical_source_20260721/gwtc3/shared/physical_h1l1_source_bank",
    },
    "GWTC4P1_O4A": {
        "short": "gwtc4",
        "source_key": "GWTC4",
        "real_run": "runs/real_gwtc34_lensing_search_20260629_full_o4",
        "source_bank": "results/real_noise_injection_v5_physical_source_20260721/gwtc4/shared/physical_h1l1_source_bank",
    },
}
AUDIT_NSIDES = (256, 512, 1024)
PILOT_EVENTS_PER_DEPLOYMENT = 50
PAIR_HASH_SALT = "v10-nside512-g05-pilot-v1"
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def deterministic_order(deployment: str, names: list[str]) -> list[str]:
    return sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{PAIR_HASH_SALT}|{deployment}|{name}".encode("utf-8")
        ).hexdigest(),
    )


def strict_event_names(v93_root: Path, short: str) -> tuple[list[str], dict[str, Any]]:
    per_seed: dict[str, list[str]] = {}
    for seed in MODEL_SEEDS:
        path = (
            v93_root
            / short
            / f"seed_{seed}"
            / "real_pair_scores_all_catalog_three_channel_equal_evidence_v81.parquet"
        )
        frame = pd.read_parquet(path, columns=["event_i", "event_j", "strict_h1l1_bbh_pair"])
        strict = frame[frame["strict_h1l1_bbh_pair"].astype(bool)]
        names = sorted(set(strict.event_i.astype(str)) | set(strict.event_j.astype(str)))
        per_seed[str(seed)] = names
    first = per_seed[str(MODEL_SEEDS[0])]
    identical = all(names == first for names in per_seed.values())
    if not identical:
        raise RuntimeError(f"Strict event UID set differs by model seed for {short}")
    return first, {
        "per_seed_unique_event_count": {seed: len(names) for seed, names in per_seed.items()},
        "sets_identical_across_model_seeds": identical,
        "event_uid_sha256": hashlib.sha256("\n".join(first).encode("utf-8")).hexdigest(),
    }


def source_dataset_storage(project: Path, common: Any, row: pd.Series) -> dict[str, Any]:
    path = project / str(row["sky_map_path"])
    suffix = path.suffix.lower()
    result: dict[str, Any] = {
        "source_path": str(path),
        "source_file_bytes": path.stat().st_size,
        "source_file_sha256": sha256(path),
        "native_moc_available": False,
        "native_moc_bytes": None,
    }
    if suffix in {".h5", ".hdf5"}:
        group = common.choose_h5_skymap_group(path)
        if group is None:
            raise RuntimeError(f"No skymap group in {path}")
        with h5py.File(path, "r") as handle:
            dataset = handle[f"{group}/skymap/data"]
            result.update(
                {
                    "source_group": group,
                    "native_map_logical_bytes": int(dataset.nbytes),
                    "native_map_storage_bytes": int(dataset.id.get_storage_size()),
                    "native_map_shape": list(dataset.shape),
                    "native_map_dtype": str(dataset.dtype),
                    "native_product_kind": "fixed_order_healpix_in_pe_hdf5",
                }
            )
    else:
        result.update(
            {
                "source_group": "",
                "native_map_logical_bytes": path.stat().st_size,
                "native_map_storage_bytes": path.stat().st_size,
                "native_map_shape": None,
                "native_map_dtype": None,
                "native_product_kind": "fits_or_moc_requires_runtime_inspection",
            }
        )
    return result


def map_diagnostics(probability: np.ndarray, nside: int) -> dict[str, Any]:
    prob = np.asarray(probability, dtype=np.float64)
    total = float(prob.sum(dtype=np.float64))
    prob = prob / total
    positive = prob[prob > 0]
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64))
    kl_uniform = float(math.log(len(prob)) - entropy)
    ordered = np.sort(prob)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    n50 = int(np.searchsorted(cumulative, 0.50, side="left") + 1)
    n90 = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
    pixel_area = float(hp.nside2pixarea(nside, degrees=True))
    return {
        "normalization_before_renorm": total,
        "normalization_abs_error": abs(total - 1.0),
        "nonzero_pixels": int(np.count_nonzero(prob)),
        "nonzero_fraction": float(np.count_nonzero(prob) / len(prob)),
        "a50_deg2": n50 * pixel_area,
        "a90_deg2": n90 * pixel_area,
        "entropy_nats": entropy,
        "kl_from_uniform_nats": kl_uniform,
        "hpd_component_count": None,
        "hpd_component_count_status": "DEFERRED_UNTIL_FROZEN_HOMOGENEOUS_PE_PIPELINE",
    }


def overlap_matrix_float64(maps: np.ndarray) -> np.ndarray:
    values = np.asarray(maps, dtype=np.float64)
    totals = values.sum(axis=1, dtype=np.float64)
    values /= totals[:, None]
    raw = values @ values.T
    return np.log(np.maximum(raw * values.shape[1], np.finfo(np.float64).tiny))


def run_map_pilot(
    project: Path,
    output: Path,
    common: Any,
    v93_module: Any,
    v93_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    pair_tables: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    uid_audit: dict[str, Any] = {}
    peak_rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    for deployment, cfg in DEPLOYMENTS.items():
        strict_names, seed_audit = strict_event_names(v93_root, cfg["short"])
        uid_audit[deployment] = seed_audit
        primary = v93_module.v7.v3.primary_manifest(v93_module.v7.v3.SOURCES[cfg["source_key"]])
        primary = primary[primary.event_name.astype(str).isin(strict_names)].copy()
        if len(primary) != len(strict_names):
            missing = sorted(set(strict_names) - set(primary.event_name.astype(str)))
            raise RuntimeError(f"Missing primary rows for {deployment}: {missing}")
        order = deterministic_order(deployment, strict_names)
        selected_names = order[: min(PILOT_EVENTS_PER_DEPLOYMENT, len(order))]
        selected = primary.set_index(primary.event_name.astype(str)).loc[selected_names].reset_index(drop=True)
        uid_audit[deployment].update(
            {
                "strict_events": len(strict_names),
                "pilot_events": len(selected),
                "pilot_event_names": selected_names,
                "pilot_selection_rule": f"first {len(selected_names)} by SHA256({PAIR_HASH_SALT}|deployment|event_name)",
            }
        )

        source_records: dict[str, dict[str, Any]] = {}
        for _, row in selected.iterrows():
            source_records[str(row.event_name)] = source_dataset_storage(project, common, row)

        scores_by_nside: dict[int, np.ndarray] = {}
        diagnostics_by_event: dict[str, dict[str, Any]] = {}
        for nside in AUDIT_NSIDES:
            maps: list[np.ndarray] = []
            load_seconds: list[float] = []
            metadata_rows: list[dict[str, Any]] = []
            stage_start = time.perf_counter()
            for _, row in selected.iterrows():
                start = time.perf_counter()
                probability, metadata = common.read_probability_map(row, target_nside=nside)
                load_seconds.append(time.perf_counter() - start)
                maps.append(probability)
                metadata_rows.append(metadata)
                if nside == 512:
                    diagnostics_by_event[str(row.event_name)] = {
                        **map_diagnostics(probability, nside),
                        **metadata,
                    }
            stack_start = time.perf_counter()
            stack = np.stack(maps).astype(np.float32, copy=False)
            stack_seconds = time.perf_counter() - stack_start
            score_start = time.perf_counter()
            score_matrix = overlap_matrix_float64(stack)
            score_seconds = time.perf_counter() - score_start
            scores_by_nside[nside] = score_matrix.copy()
            stage_seconds = time.perf_counter() - stage_start
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            runtime_rows.append(
                {
                    "deployment": deployment,
                    "nside": nside,
                    "n_events": len(selected),
                    "npix": hp.nside2npix(nside),
                    "dense_float32_bytes_per_event": hp.nside2npix(nside) * 4,
                    "median_read_rasterize_seconds_per_event": float(np.median(load_seconds)),
                    "p90_read_rasterize_seconds_per_event": float(np.quantile(load_seconds, 0.90)),
                    "total_read_rasterize_seconds": float(np.sum(load_seconds)),
                    "stack_seconds": stack_seconds,
                    "all_pair_overlap_seconds": score_seconds,
                    "stage_wall_seconds": stage_seconds,
                    "process_peak_rss_bytes": int(peak_rss),
                    "dense_persisted": False,
                    "accumulation_dtype": "float64",
                }
            )
            del stack, score_matrix, maps
            gc.collect()

        names = selected.event_name.astype(str).tolist()
        ii, jj = np.triu_indices(len(names), k=1)
        pair = pd.DataFrame(
            {
                "deployment": deployment,
                "event_i": np.asarray(names, dtype=object)[ii],
                "event_j": np.asarray(names, dtype=object)[jj],
            }
        )
        for nside in AUDIT_NSIDES:
            pair[f"z_sky_nside{nside}"] = scores_by_nside[nside][ii, jj]
        pair["abs_delta_256_512"] = np.abs(pair.z_sky_nside256 - pair.z_sky_nside512)
        pair["abs_delta_512_1024"] = np.abs(pair.z_sky_nside512 - pair.z_sky_nside1024)
        pair["sign_flip_256_512"] = np.sign(pair.z_sky_nside256) != np.sign(pair.z_sky_nside512)
        pair["sign_flip_512_1024"] = np.sign(pair.z_sky_nside512) != np.sign(pair.z_sky_nside1024)
        pair["pair_selection_hash"] = [
            hashlib.sha256(
                f"{PAIR_HASH_SALT}|{deployment}|{a}|{b}".encode("utf-8")
            ).hexdigest()
            for a, b in zip(pair.event_i, pair.event_j)
        ]
        pair_tables.append(pair)

        for _, row in selected.iterrows():
            name = str(row.event_name)
            source = source_records[name]
            diag = diagnostics_by_event[name]
            event_rows.append(
                {
                    "deployment": deployment,
                    "event_name": name,
                    "event_uid": f"{deployment}:{name}",
                    "map_role": "planning_real_public_pe",
                    "native_nside": diag["source_nside"],
                    "analysis_nside": 512,
                    "source_ordering": diag["source_ordering"],
                    "analysis_ordering": "RING",
                    "coordinate_frame": "ICRS/equatorial",
                    "probability_convention": "per-pixel probability mass",
                    "source_format": diag["source_format"],
                    "source_group": diag["source_group"],
                    "pe_pipeline_config_hash": sha256(project / "scripts/real_search/common.py"),
                    **source,
                    **{k: v for k, v in diag.items() if k not in {"source_nside", "source_ordering", "source_format", "source_group"}},
                }
            )
        del scores_by_nside
        gc.collect()

    events = pd.DataFrame(event_rows)
    pairs = pd.concat(pair_tables, ignore_index=True)
    runtimes = pd.DataFrame(runtime_rows)
    peak_rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    summary = {
        "pilot_events": int(len(events)),
        "pilot_events_by_deployment": events.groupby("deployment").size().to_dict(),
        "pilot_pairs": int(len(pairs)),
        "event_uid_audit": uid_audit,
        "process_peak_rss_before_bytes": int(peak_rss_start),
        "process_peak_rss_after_bytes": int(peak_rss_end),
        "dense_maps_persisted": False,
        "homogeneous_sky_pe_runtime_measured": False,
        "homogeneous_sky_pe_runtime_reason": "No author-frozen executable PE pipeline shared by new injections and real GWTC exists; map read/rasterization timing is reported separately and is not PE runtime.",
    }
    return events, pairs, runtimes, summary


def convergence_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for deployment, group in pairs.groupby("deployment", sort=True):
        for low, high in ((256, 512), (512, 1024)):
            delta = np.abs(group[f"z_sky_nside{low}"] - group[f"z_sky_nside{high}"]).to_numpy(float)
            flip = (
                np.sign(group[f"z_sky_nside{low}"])
                != np.sign(group[f"z_sky_nside{high}"])
            ).to_numpy(bool)
            rho = group[[f"z_sky_nside{low}", f"z_sky_nside{high}"]].corr(method="spearman").iloc[0, 1]
            rows.append(
                {
                    "deployment": deployment,
                    "comparison": f"{low}_to_{high}",
                    "n_events": len(set(group.event_i) | set(group.event_j)),
                    "n_correlated_pairs": len(group),
                    "median_abs_delta": float(np.quantile(delta, 0.50)),
                    "p90_abs_delta": float(np.quantile(delta, 0.90)),
                    "p95_abs_delta": float(np.quantile(delta, 0.95)),
                    "p99_abs_delta": float(np.quantile(delta, 0.99)),
                    "max_abs_delta": float(np.max(delta)),
                    "sign_flip_count": int(flip.sum()),
                    "sign_flip_pair_fraction_descriptive": float(flip.mean()),
                    "source_or_catalog_block_ucb95": None,
                    "source_or_catalog_block_ucb_status": "NOT_IDENTIFIABLE_FROM_SHARED_EVENT_REAL_CATALOG_PAIRS",
                    "spearman": float(rho),
                    "descriptive_starting_thresholds_pass": bool(
                        low == 512
                        and float(np.quantile(delta, 0.99)) <= 0.05
                        and float(np.max(delta)) <= 0.20
                        and int(flip.sum()) == 0
                    ),
                    "formal_gate_status": "PLANNING_ONLY_NOT_G1",
                }
            )
    return pd.DataFrame(rows)


def resource_scenarios(required_pairs: int, real_events: int) -> pd.DataFrame:
    rows = []
    for name, factor, interpretation in (
        ("minimum_exact_gate_target", 1.0, "Exact finite-sample target before attrition"),
        ("recommended_25pct_attrition_reserve", 1.25, "Planning reserve for failed PE/maps and stratification"),
        ("maximum_2x_sensitivity_envelope", 2.0, "Upper planning envelope; not author-approved sample size"),
    ):
        pairs_per_deployment = int(math.ceil(required_pairs * factor))
        null_events_total = 2 * pairs_per_deployment * len(DEPLOYMENTS)
        fit_tune_lens_events = int(math.ceil(4000 * factor))
        locked_events = int(math.ceil(900 * factor))
        total = null_events_total + fit_tune_lens_events + locked_events + real_events
        rows.append(
            {
                "scenario": name,
                "multiplier": factor,
                "independent_null_pairs_per_deployment_major_stratum": pairs_per_deployment,
                "null_endpoint_pe_events_total": null_events_total,
                "fit_tune_lens_pe_events": fit_tune_lens_events,
                "exploratory_holdout_pe_events": locked_events,
                "strict_real_pe_events": real_events,
                "total_pe_or_map_events": total,
                "dense_nside512_gib_if_improperly_persisted": total * hp.nside2npix(512) * 4 / 2**30,
                "dense_persistence_allowed": False,
                "interpretation": interpretation,
                "pe_cpu_or_gpu_hours": None,
                "pe_runtime_status": "UNKNOWN_UNTIL_FROZEN_HOMOGENEOUS_PE_PILOT",
            }
        )
    return pd.DataFrame(rows)


def make_figure(output: Path, support: pd.DataFrame, convergence: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(support))
    axes[0].bar(x - 0.18, support.required_independent_audit_pairs, 0.36, color="#333333", label="Required")
    axes[0].bar(x + 0.18, support.available_event_disjoint_null_pairs, 0.36, color="#2A788E", label="Available")
    axes[0].set_xticks(x, support.deployment)
    axes[0].set_ylabel("Independent audit pairs")
    axes[0].set_title("Current null support")
    axes[0].legend(frameon=False)

    high = convergence[convergence.comparison == "512_to_1024"]
    axes[1].bar(np.arange(len(high)), high.p99_abs_delta, color="#D1495B")
    axes[1].axhline(0.05, color="#222222", linestyle="--", linewidth=1, label="Starting P99 gate")
    axes[1].set_xticks(np.arange(len(high)), high.deployment)
    axes[1].set_ylabel(r"P99 $|Z_{512}-Z_{1024}|$ (nats)")
    axes[1].set_title("Planning-map convergence")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    output = args.output_root.resolve()
    protocol = args.protocol.resolve()
    addendum = args.addendum.resolve()
    if not output.is_dir():
        raise FileNotFoundError("Run G-1 schema/template generator before this script")

    legacy = load_module(
        project / "scripts/experiments/01_05_author_override_inventory_power.py",
        "g05_legacy_helpers_nside512",
    )
    common = load_module(project / "scripts/real_search/common.py", "real_search_common_nside512")
    v93_module = load_module(
        project / "scripts/experiments/run_gwtc_sky_resolution_v93.py",
        "gwtc_v93_nside512_planning",
    )
    v93_root = project / "results/gwtc_sky_resolution_v93_20260730"

    contracts = output / "contracts"
    shutil.copy2(addendum, contracts / "NSIDE512_PROTOCOL_ADDENDUM.md")
    os.chmod(contracts / "NSIDE512_PROTOCOL_ADDENDUM.md", 0o444)
    copied_protocol = output / "contracts/protocol" / protocol.name
    if copied_protocol.exists():
        os.chmod(copied_protocol, 0o444)

    disk = shutil.disk_usage(project)
    analysis_contract = {
        "schema": "sky-background-v10-nside512-planning-contract-v1",
        "created_at_utc": utc_now(),
        "lifecycle_status": "PLANNING_ONLY_NOT_ANALYSIS_CONFIG_FINAL",
        "approved_for": "G_MINUS1_G0_G05_PLANNING_ONLY",
        "analysis_nside": 512,
        "coarse_audit_nside": 256,
        "convergence_reference_nside": 1024,
        "dense_persistence": False,
        "probability_storage_dtype": "float32_transient",
        "normalization_and_overlap_dtype": "float64",
        "active_working_set_limit_gib": 70,
        "minimum_free_space_gib": 20,
        "pair_audit_selection": {
            "hash_salt": PAIR_HASH_SALT,
            "required_1024_sets": [
                "all_real_strict_scope_pairs",
                "all_injection_true_companions",
                "prefrozen_source_noise_disjoint_random_null_pairs",
                "nside512_null_positive_tail_1pct",
                "sign_flips_256_to_512",
                "abs_delta_256_512_above_frozen_threshold",
            ],
        },
        "starting_convergence_gate": {
            "p99_abs_delta_512_1024_max_nats": 0.05,
            "max_abs_delta_512_1024_nats": 0.20,
            "sign_flip_source_catalog_block_ucb95_max": 0.001,
            "o3_o4a_must_pass_separately": True,
        },
        "future_final_contract_required": {
            "filename": "ANALYSIS_CONFIG_FINAL.json",
            "approved_for": "G1_TO_G5_EXPLORATORY",
            "must_be_author_signed_after_g05_review": True,
        },
        "prohibited_now": [
            "G1_TO_G5",
            "NEW_INJECTION_PE",
            "CALIBRATOR_FIT",
            "REAL_CATALOG_RERANK",
            "V93_OVERWRITE",
            "PAPER_OR_GIT_OR_OVERLEAF_MODIFICATION",
        ],
        "physical_disk_free_gib_at_start": disk.free / 2**30,
        "protocol_sha256": sha256(protocol),
        "addendum_sha256": sha256(addendum),
    }
    write_json(contracts / "NSIDE512_ANALYSIS_CONTRACT.json", analysis_contract)

    authorization = {
        "recorded_at_utc": utc_now(),
        "source": "explicit author instruction in current conversation",
        "authorized_scope": ["G_MINUS1", "G0", "G0.5", "100_TO_300_EVENT_PLANNING_PILOT"],
        "not_authorized": ["G1_TO_G5", "EXPENSIVE_HOMOGENEOUS_PE", "REAL_RERANK", "PAPER_UPDATE"],
        "final_analysis_config_signed": False,
    }
    write_json(contracts / "AUTHOR_INSTRUCTION_G0_G05_ONLY.json", authorization)

    gw = legacy.eligible_gwlmc_systems()
    eligible_frame = gw.pop("eligible_frame")
    inventories: dict[str, Any] = {}
    noise_tables = []
    support_rows = []
    strict_total = 0
    for deployment, cfg in DEPLOYMENTS.items():
        run_root = project / cfg["real_run"]
        noise_frame, noise_summary = legacy.noise_inventory(project, run_root)
        noise_frame.insert(0, "deployment", deployment)
        noise_tables.append(noise_frame)
        strict_names, seed_audit = strict_event_names(v93_root, cfg["short"])
        strict_total += len(strict_names)
        bank = legacy.source_bank_audit(project, project / cfg["source_bank"], eligible_frame)
        inventories[deployment] = {
            "strict_scope": {
                "event_count": len(strict_names),
                "unordered_pairs": len(strict_names) * (len(strict_names) - 1) // 2,
                **seed_audit,
            },
            "source_bank": bank,
            "noise": noise_summary,
        }
    noise_table = pd.concat(noise_tables, ignore_index=True)
    noise_table.to_csv(output / "tables/G0_OFFSOURCE_NOISE_BLOCK_INVENTORY.csv", index=False, encoding="utf-8-sig")

    events, pairs, runtimes, pilot_summary = run_map_pilot(project, output, common, v93_module, v93_root)
    convergence = convergence_summary(pairs)
    events.to_csv(output / "tables/G05_NSIDE512_EVENT_PILOT.csv", index=False, encoding="utf-8-sig")
    pairs.to_parquet(output / "tables/G05_NSIDE512_PAIR_CONVERGENCE_PILOT.parquet", index=False)
    convergence.to_csv(output / "tables/G05_NSIDE512_CONVERGENCE_SUMMARY.csv", index=False, encoding="utf-8-sig")
    runtimes.to_csv(output / "tables/G05_NSIDE512_RUNTIME_MEMORY_PILOT.csv", index=False, encoding="utf-8-sig")

    req_1 = legacy.required_n_for_probable_ucb(0.01, 0.015, 0.80)
    req_5 = legacy.required_n_for_probable_ucb(0.05, 0.06, 0.80)
    required_pairs = max(req_1["n"], req_5["n"])
    for deployment, inventory in inventories.items():
        available = inventory["noise"]["maximum_event_disjoint_null_pairs_from_current_files"]
        support_rows.append(
            {
                "deployment": deployment,
                "required_independent_audit_pairs": required_pairs,
                "available_event_disjoint_null_pairs": available,
                "support_fraction": available / required_pairs,
                "passes": available >= required_pairs,
                "additional_pairs_required": max(required_pairs - available, 0),
                "additional_endpoint_noise_blocks_required": 2 * max(required_pairs - available, 0),
            }
        )
    support = pd.DataFrame(support_rows)
    support.to_csv(output / "tables/G05_FINITE_SAMPLE_SUPPORT.csv", index=False, encoding="utf-8-sig")
    scenarios = resource_scenarios(required_pairs, strict_total)
    scenarios.to_csv(output / "tables/G05_RESOURCE_SCENARIOS.csv", index=False, encoding="utf-8-sig")

    map_source_summary = (
        events.groupby(["deployment", "native_product_kind", "native_nside"], dropna=False)
        .agg(
            events=("event_name", "size"),
            median_native_storage_bytes=("native_map_storage_bytes", "median"),
            p90_native_storage_bytes=("native_map_storage_bytes", lambda x: np.quantile(x, 0.9)),
            moc_available=("native_moc_available", "max"),
        )
        .reset_index()
    )
    map_source_summary.to_csv(output / "tables/G0_PE_MOC_SOURCE_INVENTORY.csv", index=False, encoding="utf-8-sig")

    g0_inventory = {
        "schema": "sky-background-v10-nside512-g0-inventory-v1",
        "status": "G0_INVENTORY_COMPLETE",
        "generated_at_utc": utc_now(),
        "global_gwlmc_inventory": gw,
        "deployments": inventories,
        "pilot": pilot_summary,
        "pe_and_map_pipeline": {
            "real_public_products": "fixed-order HEALPix maps in public PE HDF5",
            "native_moc_available_in_pilot": bool(events.native_moc_available.any()),
            "new_injection_homogeneous_pe_pipeline": "NOT_AVAILABLE_AS_AUTHOR_FROZEN_EXECUTABLE_PIPELINE",
            "same_pipeline_for_new_injection_and_real": False,
            "historical_injection_maps": "Nside=64 rotated empirical templates; provenance only; forbidden as v10 formal Nside=512 calibration",
            "full_pe_runtime_measured": False,
        },
    }
    write_json(output / "G0_SOURCE_NOISE_INVENTORY.json", g0_inventory)

    nside512_pilot = convergence[convergence.comparison == "512_to_1024"]
    descriptive_convergence_all = bool(nside512_pilot.descriptive_starting_thresholds_pass.all())
    checks = {
        "g0_source_inventory_complete": True,
        "g0_noise_inventory_complete": True,
        "strict_event_uid_sets_identical_across_model_seeds": all(
            x["strict_scope"]["sets_identical_across_model_seeds"] for x in inventories.values()
        ),
        "planning_pilot_100_to_300_events_complete": 100 <= len(events) <= 300,
        "streamed_nside512_dense_persistence_disabled": True,
        "active_working_set_below_70_gib": runtimes.process_peak_rss_bytes.max() <= 70 * 2**30,
        "disk_safety_above_20_gib": shutil.disk_usage(project).free >= 20 * 2**30,
        "homogeneous_real_and_new_injection_pe_pipeline_available": False,
        "o3_independent_null_support": bool(support.loc[support.deployment == "GWTC3_O3", "passes"].iloc[0]),
        "o4a_independent_null_support": bool(support.loc[support.deployment == "GWTC4P1_O4A", "passes"].iloc[0]),
        "pilot_real_map_nside512_descriptive_convergence": descriptive_convergence_all,
        "formal_source_catalog_block_signflip_ucb_evaluable_at_g05": False,
        "full_homogeneous_pe_runtime_benchmarked": False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    storage_projection = {
        "dense_float32_bytes_per_map": {str(n): hp.nside2npix(n) * 4 for n in AUDIT_NSIDES},
        "minimum_total_events_historical_projection": int(scenarios.iloc[0].total_pe_or_map_events),
        "improper_full_dense_nside512_persistence_gib": float(scenarios.iloc[0].dense_nside512_gib_if_improperly_persisted),
        "dense_persistence": False,
        "measured_peak_rss_gib": float(runtimes.process_peak_rss_bytes.max() / 2**30),
        "active_working_set_limit_gib": 70,
        "minimum_free_space_gib": 20,
        "free_space_after_pilot_gib": float(shutil.disk_usage(project).free / 2**30),
        "storage_blocker_resolved_by_streaming_design": True,
        "native_moc_available": bool(events.native_moc_available.any()),
        "native_product_note": "Current public PE pilot uses compressed fixed-order HEALPix datasets, not native MOC. MOC production remains a future pipeline requirement, not an assumed compression ratio.",
    }
    recommendation = {
        "schema": "sky-background-v10-nside512-g05-recommendation-v1",
        "generated_at_utc": utc_now(),
        "status": "NO_GO_OR_REDESIGN",
        "final_state": FINAL_STATUS,
        "sample_size": {
            "nominal_1pct_ucb_1p5pct_power80": req_1,
            "nominal_5pct_ucb_6pct_power80": req_5,
            "binding_independent_audit_pairs_per_deployment_major_stratum": required_pairs,
            "statistical_unit": "source/noise-disjoint event pair; model seeds do not multiply n",
        },
        "checks": checks,
        "blocking_reasons": blockers,
        "storage": storage_projection,
        "pilot_convergence": convergence.to_dict(orient="records"),
        "decision": "DO_NOT_START_G1_TO_G5. Storage is now feasible with no dense persistence, but homogeneous PE provenance, independent null support, formal block-level convergence, and PE runtime remain unresolved.",
        "next_author_decision": "Review G0.5; either acquire/freeze a homogeneous PE pipeline and more independent off-source blocks, or redesign finite-sample Gate E. Do not sign ANALYSIS_CONFIG_FINAL until blockers are resolved.",
    }
    write_json(output / "G05_POWER_SUPPORT_RECOMMENDATION.json", recommendation)

    initial_audit_path = output / "provenance/G_MINUS1_CONFIG_AUDIT.json"
    initial_audit = json.loads(initial_audit_path.read_text(encoding="utf-8"))
    initial_hold = output / "HOLD_FOR_AUTHOR_SIGNATURE"
    if initial_hold.exists():
        initial_hold.replace(output / "provenance/G_MINUS1_INITIAL_HOLD_FOR_AUTHOR_SIGNATURE")
    gminus = {
        **initial_audit,
        "status": "G_MINUS1_TEMPLATES_COMPLETE_G0_G05_EXPLICITLY_AUTHORIZED_ONLY",
        "formal_detached_signature_gate_pass": False,
        "planning_authorization": authorization,
        "nside512_contract": str(contracts / "NSIDE512_ANALYSIS_CONTRACT.json"),
        "nside512_contract_sha256": sha256(contracts / "NSIDE512_ANALYSIS_CONTRACT.json"),
        "g1_to_g5_authorized": False,
    }
    write_json(output / "G_MINUS1_CONFIG_AUDIT.json", gminus)
    write_json(initial_audit_path, gminus)

    make_figure(output / "figures/fig_g05_nside512_feasibility", support, convergence)

    report_header = (
        "本轮为Nside=512统一分辨率的v10探索实验。historical v9.3真实GWTC "
        "Nside=1024、历史O3/O4a注入Nside=64和ET-3 Nside=128结果均未覆盖。"
        "Nside=1024仅作为本轮高分辨率收敛参考。本轮结果未经新的单一方案确认性"
        "locked test，不得写入论文或替代v9.3。"
    )
    conv_lines = []
    for row in convergence.itertuples(index=False):
        conv_lines.append(
            f"- {row.deployment} {row.comparison}: P99={row.p99_abs_delta:.6g}, "
            f"max={row.max_abs_delta:.6g}, sign flips={row.sign_flip_count}/{row.n_correlated_pairs}, "
            f"Spearman={row.spearman:.6f}."
        )
    report = f"""# G0.5 Nside=512 存储与收敛规划报告

> {report_header}

生成时间：{utc_now()}

## 结论

本轮完成 G-1 模板、G0 inventory 和 {len(events)} 个真实 public-PE planning
事件的 256/512/1024 临时 rasterization pilot。没有生成新 injection PE，
没有拟合 calibrator，没有重排真实候选，也没有启动 G1-G5。

磁盘问题已通过“不永久保存 dense maps”的实现原则解除：本次 dense maps
只存在于内存，峰值 RSS 为 {runtimes.process_peak_rss_bytes.max()/2**30:.2f} GiB，
低于 70 GiB 活动工作集上限；当前剩余空间约
{shutil.disk_usage(project).free/2**30:.1f} GiB。

G0.5 科学结论仍为 **NO_GO_OR_REDESIGN**，原因不是磁盘，而是：

1. 尚无新 injection 与真实 GWTC 共用的、作者冻结且可执行的同质天空 PE 管线；
2. O3/O4a 当前可构造的独立 null pairs 仅为
   {int(support.iloc[0].available_event_disjoint_null_pairs)} 和
   {int(support.iloc[1].available_event_disjoint_null_pairs)}，低于每个主要层
   {required_pairs} 的 finite-sample 目标；
3. 真实 catalog pair 共享事件，不能用普通 pair 数计算所要求的 source/catalog-
   block Clopper-Pearson UCB；
4. 同质 PE 尚未实现，因此 PE runtime 不能用“读取公开地图的时间”替代。

## 分辨率 planning pilot

{chr(10).join(conv_lines)}

这些是 planning real-map 的描述性数值，不是 G1 正式 Gate。若 P99、最大差值
或符号翻转提示风险，后续不得调宽容差；应先建立同质 PE 管线，再按原合同在
O3/O4a 的独立层上正式验证。

## 存储实现

- Nside=256/512/1024 dense float32 单图理论体积分别为
  {hp.nside2npix(256)*4/2**20:.1f}、{hp.nside2npix(512)*4/2**20:.1f}、
  {hp.nside2npix(1024)*4/2**20:.1f} MiB；
- 正式概率归一化与 overlap 使用 float64；
- 本 pilot 没有保存任何 dense map；
- 100 个 pilot 事件的公开 PE 产品为固定阶 HEALPix HDF5，不是原生 MOC；
- 将 {int(scenarios.iloc[0].total_pe_or_map_events):,} 张 Nside=512 dense map
  全部落盘仍会占约 {scenarios.iloc[0].dense_nside512_gib_if_improperly_persisted:.1f}
  GiB，因此依旧禁止；生产实现应按事件/小批次流式 rasterize，只保存 pair score
  和审计字段。

## 资源方案

- 最小：每 deployment × major stratum {required_pairs} 个独立 null audit pairs；
- 推荐：预留 25% PE/map 失败与分层损耗；
- 最大 planning envelope：最小规模的 2 倍，只用于预算上界，未获作者批准；
- PE CPU/GPU 时目前必须记为未知，直到同质 PE pipeline 冻结并完成小型 pilot。

## 下一步

不得签署或运行 G1-G5，除非先补齐同质 PE 管线、足量 source/noise-disjoint
off-source blocks，并能在 pilot 上测得真实 PE 时间与 MOC/压缩产物大小。之后
作者需审核本包并另行签署 `ANALYSIS_CONFIG_FINAL.json`。

## 固定状态

`{FINAL_STATUS}`
"""
    (output / "G05_NSIDE512_STORAGE_AND_CONVERGENCE_PLAN_CN.md").write_text(report, encoding="utf-8")
    (output / "reports/G05_NSIDE512_STORAGE_AND_CONVERGENCE_PLAN_CN.md").write_text(report, encoding="utf-8")

    write_json(
        output / "STATUS.json",
        {
            "run_id": output.name,
            "status": FINAL_STATUS,
            "scientific_decision": "NO_GO_OR_REDESIGN",
            "updated_at_utc": utc_now(),
            "g1_to_g5_started": False,
            "real_catalog_reranked": False,
            "v93_modified": False,
            "paper_modified": False,
        },
    )
    (output / FINAL_STATUS).write_text(utc_now() + "\n", encoding="ascii")

    index = f"""# Results index

- `G_MINUS1_CONFIG_AUDIT.json`
- `G0_SOURCE_NOISE_INVENTORY.json`
- `G05_POWER_SUPPORT_RECOMMENDATION.json`
- `G05_NSIDE512_STORAGE_AND_CONVERGENCE_PLAN_CN.md`
- `contracts/NSIDE512_PROTOCOL_ADDENDUM.md`
- `contracts/NSIDE512_ANALYSIS_CONTRACT.json`
- `tables/G05_NSIDE512_EVENT_PILOT.csv`
- `tables/G05_NSIDE512_PAIR_CONVERGENCE_PILOT.parquet`
- `tables/G05_NSIDE512_CONVERGENCE_SUMMARY.csv`
- `tables/G05_NSIDE512_RUNTIME_MEMORY_PILOT.csv`
- `tables/G05_FINITE_SAMPLE_SUPPORT.csv`
- `tables/G05_RESOURCE_SCENARIOS.csv`
- `figures/fig_g05_nside512_feasibility.pdf` and `.png`

Final status: `{FINAL_STATUS}`.
"""
    (output / "RESULTS_INDEX_CN.md").write_text(index, encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), output / "scripts" / Path(__file__).name)
    manifest_rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and "package" not in p.parts):
        if path.name == "SHA256SUMS":
            continue
        manifest_rows.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(
        output / "provenance/FILE_MANIFEST.json",
        {"schema": "sky-background-v10-nside512-file-manifest-v1", "files": manifest_rows},
    )
    checksum_lines = [f"{row['sha256']}  {row['path']}" for row in manifest_rows]
    (output / "provenance/SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    package = project / "packages" / f"{output.name}_G05_HOLD.tar.gz"
    package.parent.mkdir(parents=True, exist_ok=True)
    if package.exists():
        raise FileExistsError(package)
    with tarfile.open(package, "w:gz") as archive:
        archive.add(output, arcname=output.name, recursive=True)
    package_hash = sha256(package)
    package.with_suffix(package.suffix + ".sha256").write_text(
        f"{package_hash}  {package.name}\n", encoding="ascii"
    )
    write_json(
        output / "provenance/PACKAGE.json",
        {"path": str(package), "bytes": package.stat().st_size, "sha256": package_hash},
    )
    print(json.dumps({"output": str(output), "package": str(package), "sha256": package_hash, "status": FINAL_STATUS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
