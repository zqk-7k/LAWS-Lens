#!/usr/bin/env python3
"""Run the author-overridden G0/G0.5 inventory and feasibility audit.

This script intentionally does not generate injections, run PE, inspect a new
locked sample, or rerank the real catalog.  It implements the scientific
go/no-go checks that precede those operations in the 2026-08-22 protocol.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEPLOYMENTS = {
    "GWTC3_O3": {
        "short": "gwtc3",
        "real_run": "runs/real_gwtc_lensing_search_20260625",
        "source_bank": "results/real_noise_injection_v5_physical_source_20260721/gwtc3/shared/physical_h1l1_source_bank",
    },
    "GWTC4P1_O4A": {
        "short": "gwtc4",
        "real_run": "runs/real_gwtc34_lensing_search_20260629_full_o4",
        "source_bank": "results/real_noise_injection_v5_physical_source_20260721/gwtc4/shared/physical_h1l1_source_bank",
    },
}

V93_ROOT = Path("results/gwtc_sky_resolution_v93_20260730")
V7_ROOT = Path("results/real_noise_injection_v7_peak2s_formal_20260722")
GW_LMC_ROOT = Path("/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1")
MODEL_SEEDS = (202607241, 202607242, 202607243)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return []
    return [float(x) for x in parsed]


def eligible_gwlmc_systems() -> dict[str, Any]:
    image = pd.read_csv(GW_LMC_ROOT / "BBH_2.5PLUS_Any_Detected_SNR1_ImageParams.csv")
    lens = pd.read_csv(GW_LMC_ROOT / "BBH_2.5PLUS_Any_Detected_SNR1_LensParams.csv")
    rows = []
    for idx, (im, le) in enumerate(zip(image.itertuples(index=False), lens.itertuples(index=False))):
        snr = parse_list(im.img_snrs)
        delay = parse_list(im.img_delays_days)
        detected = [i for i, x in enumerate(snr) if np.isfinite(x) and x >= 1.0]
        if len(detected) < 2 or len(delay) != len(snr):
            continue
        selected = sorted(detected, key=lambda i: snr[i], reverse=True)[:2]
        ratio = max(snr[i] for i in selected) / max(min(snr[i] for i in selected), 1e-12)
        dt = abs(delay[selected[0]] - delay[selected[1]])
        if not np.isfinite(ratio) or ratio > 4.0 or not np.isfinite(dt) or dt <= 0:
            continue
        rows.append(
            {
                "gwlmc_row": idx,
                "gwlmc_event_id": int(im.event_id),
                "physical_lens_group": "subhalo_present" if bool(le.is_subhalo) else "smooth_non_subhalo",
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "catalog_rows": int(len(image)),
        "eligible_systems": int(len(frame)),
        "eligible_unique_event_ids": int(frame.gwlmc_event_id.nunique()),
        "by_physical_lens_group": frame.physical_lens_group.value_counts().to_dict(),
        "eligible_frame": frame,
    }


def resolve_manifest_path(project: Path, run_root: Path, raw: Any) -> Path | None:
    if pd.isna(raw):
        return None
    path = Path(str(raw))
    candidates = [path] if path.is_absolute() else [run_root / path, project / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def dq_good_seconds(path: Path) -> tuple[int, np.ndarray]:
    with h5py.File(path, "r") as handle:
        start = int(handle["meta/GPSstart"][()])
        mask = np.asarray(handle["quality/simple/DQmask"][:], dtype=np.uint32)
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["quality/simple/DQShortnames"][:]]
    required = ["DATA", "CBC_CAT2"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing DQ bits {missing} in {path}")
    good = np.ones(len(mask), dtype=bool)
    for name in required:
        bit = names.index(name)
        good &= (mask & np.uint32(1 << bit)) != 0
    return start, good


def count_fixed_blocks(good: np.ndarray, block_seconds: int) -> int:
    count = 0
    index = 0
    size = len(good)
    while index + block_seconds <= size:
        if bool(np.all(good[index : index + block_seconds])):
            count += 1
            index += block_seconds
        else:
            bad = np.flatnonzero(~good[index : index + block_seconds])
            index += int(bad[0]) + 1
    return count


def noise_inventory(project: Path, run_root: Path, block_seconds: int = 320) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = run_root / "data/strain_gwosc_download_manifest.csv"
    events_path = run_root / "data/event_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    event_times = pd.read_csv(events_path)
    gps_col = "gps_time" if "gps_time" in event_times else "GPS"
    all_event_gps = event_times[gps_col].dropna().astype(float).to_numpy()

    if "gps_start" in manifest:
        start_col = "gps_start"
    else:
        start_col = "gwosc_start"
    records = []
    grouped = manifest.dropna(subset=[start_col]).groupby(start_col, sort=True)
    for start_value, group in grouped:
        detector_paths = {}
        for row in group.itertuples(index=False):
            detector = str(getattr(row, "detector"))
            if detector not in {"H1", "L1"}:
                continue
            path = resolve_manifest_path(project, run_root, getattr(row, "local_path"))
            if path is not None and path.exists():
                detector_paths[detector] = path
        if set(detector_paths) != {"H1", "L1"}:
            continue
        try:
            start_h, good_h = dq_good_seconds(detector_paths["H1"])
            start_l, good_l = dq_good_seconds(detector_paths["L1"])
            if start_h != start_l or len(good_h) != len(good_l):
                raise ValueError("H1/L1 DQ grids differ")
            good = good_h & good_l
            seconds = start_h + np.arange(len(good), dtype=np.float64)
            for gps in all_event_gps:
                good[np.abs(seconds - gps) <= 128.0] = False
            records.append(
                {
                    "gps_start": start_h,
                    "duration_seconds": len(good),
                    "joint_good_seconds_after_event_veto": int(good.sum()),
                    "independent_block_seconds": block_seconds,
                    "independent_blocks": count_fixed_blocks(good, block_seconds),
                    "h1_path": str(detector_paths["H1"]),
                    "l1_path": str(detector_paths["L1"]),
                    "status": "usable",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "gps_start": float(start_value),
                    "duration_seconds": 0,
                    "joint_good_seconds_after_event_veto": 0,
                    "independent_block_seconds": block_seconds,
                    "independent_blocks": 0,
                    "h1_path": str(detector_paths["H1"]),
                    "l1_path": str(detector_paths["L1"]),
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            )
    frame = pd.DataFrame(records)
    usable = frame[frame.status == "usable"] if len(frame) else frame
    return frame, {
        "strain_manifest": str(manifest_path),
        "strain_manifest_sha256": sha256(manifest_path),
        "h1l1_file_groups": int(len(frame)),
        "usable_h1l1_file_groups": int(len(usable)),
        "joint_good_seconds_after_event_veto": int(usable.joint_good_seconds_after_event_veto.sum()) if len(usable) else 0,
        "conservative_independent_block_seconds": block_seconds,
        "conservative_independent_noise_blocks": int(usable.independent_blocks.sum()) if len(usable) else 0,
        "maximum_event_disjoint_null_pairs_from_current_files": int(usable.independent_blocks.sum() // 2) if len(usable) else 0,
        "dq_rule": "H1 and L1 DATA + CBC_CAT2 at one-second resolution; exclude every catalog event +/-128 s; greedy non-overlapping 320 s blocks",
    }


def one_sided_cp_upper(x: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(stats.beta.ppf(confidence, x + 1, n - x))


def required_n_for_probable_ucb(true_rate: float, target_ucb: float, power: float = 0.8, max_n: int = 20000) -> dict[str, Any]:
    for n in range(1, max_n + 1):
        # CP upper limits are monotone in the observed failure count.  The
        # beta/binomial identity places the boundary near the 5% lower tail
        # under target_ucb, so only a few exact beta evaluations are needed.
        candidate = int(stats.binom.ppf(0.05, n, target_ucb))
        candidate = min(max(candidate, 0), n)
        while candidate >= 0 and one_sided_cp_upper(candidate, n) > target_ucb:
            candidate -= 1
        while candidate + 1 <= n and one_sided_cp_upper(candidate + 1, n) <= target_ucb:
            candidate += 1
        probability = float(stats.binom.cdf(candidate, n, true_rate)) if candidate >= 0 else 0.0
        if probability >= power:
            return {"n": n, "achieved_probability": probability}
    raise RuntimeError("required sample size exceeds search range")


def required_zero_failure_n(target_ucb: float) -> int:
    n = 1
    while one_sided_cp_upper(0, n) > target_ucb:
        n += 1
    return n


def strict_real_event_count(v93: Path, short: str) -> int:
    frame = pd.read_parquet(v93 / short / "seed_202607241/real_pair_features_unified_sky_v81.parquet")
    strict = frame[frame.strict_h1l1_bbh_pair.astype(bool)]
    return int(len(set(strict.event_i.astype(str)) | set(strict.event_j.astype(str))))


def source_bank_audit(project: Path, source_bank: Path, eligible: pd.DataFrame) -> dict[str, Any]:
    records = {}
    used_ids = set()
    paths = {
        "smooth_non_subhalo": source_bank / "SIS_data_0222/physical_source_pair_metadata.parquet",
        "subhalo_present": source_bank / "PM_data_0222/physical_source_pair_metadata.parquet",
        "unlensed": source_bank / "Unlensed_data_0222/physical_unlensed_source_metadata.parquet",
    }
    for label, path in paths.items():
        frame = pd.read_parquet(path)
        ids = set(frame.gwlmc_event_id.astype(int))
        used_ids |= ids
        records[label] = {
            "rows": int(len(frame)),
            "unique_gwlmc_event_ids": int(len(ids)),
            "path": str(path),
            "sha256": sha256(path),
        }
    eligible_ids = set(eligible.gwlmc_event_id.astype(int))
    return {
        "source_bank_summary": str(source_bank / "physical_source_bank_summary.json"),
        "source_bank_summary_sha256": sha256(source_bank / "physical_source_bank_summary.json"),
        "existing_bank": records,
        "all_existing_bank_unique_event_ids": int(len(used_ids)),
        "eligible_lensed_event_ids_not_in_existing_bank": int(len(eligible_ids - used_ids)),
        "warning": "The existing 600-per-slot bank was materialized in every v7 seed and is not globally new for v10.",
    }


def baseline_metrics(v93: Path) -> pd.DataFrame:
    summary = json.loads((v93 / "final_audit_summary_v93.json").read_text(encoding="utf-8"))
    rows = []
    for dep in ("gwtc3", "gwtc4"):
        core = summary["core_results"][dep]
        for method in ("waveform_only", "time_only", "sky_only", "retrieval_three_channel_unconstrained"):
            item = core[method]
            rows.append(
                {
                    "deployment": dep,
                    "method": method,
                    "r_at_1_mean": item["r_at_1_mean"],
                    "r_at_1_std": item["r_at_1_std"],
                    "r_at_10_mean": item["r_at_10_mean"],
                    "r_at_10_std": item["r_at_10_std"],
                    "status": "historical_v9.3_context_only",
                }
            )
    return pd.DataFrame(rows)


def environment_contract(project: Path) -> dict[str, Any]:
    def command(args: list[str]) -> str:
        return subprocess.check_output(args, text=True).strip()

    disk = shutil.disk_usage(project)
    try:
        gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
    except Exception as exc:
        gpu = f"unavailable:{type(exc).__name__}"
    return {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "python": sys.version,
        "git_commit": command(["git", "-C", str(project), "rev-parse", "HEAD"]),
        "cpu_count": os.cpu_count(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": gpu,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "healpy": __import__("healpy").__version__,
            "bilby": __import__("bilby").__version__,
        },
    }


def make_figure(out: Path, feasibility: pd.DataFrame, storage: dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    labels = feasibility.deployment.tolist()
    required = feasibility.required_independent_audit_pairs.to_numpy()
    available = feasibility.available_event_disjoint_null_pairs.to_numpy()
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, required, width, color="#222222", label="Required")
    axes[0].bar(x + width / 2, available, width, color="#2A788E", label="Available")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Independent audit pairs")
    axes[0].set_title("Finite-sample calibration support")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)

    names = ["Dense Nside=1024\nmaps", "Free disk"]
    values = [storage["dense_map_storage_gib"], storage["disk_free_gib"]]
    axes[1].bar(names, values, color=["#D1495B", "#2A788E"])
    axes[1].set_ylabel("Storage (GiB)")
    axes[1].set_title("Minimum projected map storage")
    axes[1].spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(values):
        axes[1].text(idx, value, f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    protocol = args.protocol.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")

    for rel in [
        "contracts", "provenance", "tables", "figures", "reports", "logs", "scripts", "package",
        "calibration/one_stage", "calibration/two_stage", "injections/o3", "injections/o4a",
        "scores/baseline_v93_same_domain", "scores/one_stage_eta_000", "scores/one_stage_eta_025",
        "scores/one_stage_eta_050", "scores/one_stage_eta_100", "scores/two_stage_wt_only",
        "scores/two_stage_full", "real_catalog/descriptive_only", "checkpoints",
    ]:
        (output / rel).mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol, output / "contracts" / protocol.name)

    override = {
        "schema": "sky-background-v10-author-direct-execution-override-v1",
        "recorded_at_utc": utc_now(),
        "author_instruction": "跳过这些需要签署的地方，直接开始做。按照我的说法来，直接对着方案做出来结果。",
        "administrative_signature_gates_waived": True,
        "scientific_quality_and_feasibility_gates_waived": False,
        "approved_scope": ["G0", "G0.5", "G1_TO_G5_ONLY_IF_G0.5_PASS"],
        "no_overwrite": True,
        "no_paper_modification": True,
    }
    write_json(output / "contracts/AUTHOR_DIRECT_EXECUTION_OVERRIDE.json", override)
    env = environment_contract(project)
    write_json(output / "provenance/ENVIRONMENT.json", env)

    v93 = project / V93_ROOT
    v7 = project / V7_ROOT
    gw = eligible_gwlmc_systems()
    eligible_frame = gw.pop("eligible_frame")
    inventories = {}
    noise_rows = []
    support_rows = []
    strict_counts = {}
    for deployment, cfg in DEPLOYMENTS.items():
        short = cfg["short"]
        run_root = project / cfg["real_run"]
        source_bank = project / cfg["source_bank"]
        noise_frame, noise_summary = noise_inventory(project, run_root)
        noise_frame.insert(0, "deployment", deployment)
        noise_rows.append(noise_frame)
        bank = source_bank_audit(project, source_bank, eligible_frame)
        strict_count = strict_real_event_count(v93, short)
        strict_counts[deployment] = strict_count
        inventories[deployment] = {
            "source_bank": bank,
            "noise": noise_summary,
            "strict_real_catalog_events": strict_count,
            "strict_real_catalog_pairs": strict_count * (strict_count - 1) // 2,
            "v93_seed_pair_tables": {
                str(seed): {
                    "validation": str(v93 / short / f"seed_{seed}/fusion_validation_pairs_v81.parquet"),
                    "heldout_test": str(v93 / short / f"seed_{seed}/fusion_heldout_test_pairs_v81.parquet"),
                }
                for seed in MODEL_SEEDS
            },
        }

    noise_table = pd.concat(noise_rows, ignore_index=True)
    noise_table.to_csv(output / "tables/G0_OFFSOURCE_NOISE_BLOCK_INVENTORY.csv", index=False, encoding="utf-8-sig")
    write_json(
        output / "G0_SOURCE_NOISE_INVENTORY.json",
        {
            "schema": "sky-background-v10-g0-source-noise-inventory-v1",
            "status": "G0_INVENTORY_COMPLETE",
            "generated_at_utc": utc_now(),
            "global_gwlmc_inventory": gw,
            "deployments": inventories,
            "homogeneous_pe_pipeline": {
                "status": "NOT_AVAILABLE_AS_FROZEN_EXECUTABLE_PIPELINE",
                "bilby_installed": True,
                "real_maps": "public PE posterior maps represented at Nside=1024",
                "injection_maps": "v9.3 Nside=64 empirical real-PE templates rotated to injection truth",
                "same_pipeline_for_real_and_injection": False,
                "evidence": [
                    str(project / "scripts/experiments/20_real_noise_injection_v3_physical.py"),
                    str(v93 / "analysis_contract_v93.json"),
                ],
            },
        },
    )

    req_1 = required_n_for_probable_ucb(0.01, 0.015, 0.80)
    req_5 = required_n_for_probable_ucb(0.05, 0.06, 0.80)
    zero_failure = {str(x): required_zero_failure_n(x) for x in (0.05, 0.01, 0.001)}
    required_pairs = max(req_1["n"], req_5["n"])
    for deployment, inv in inventories.items():
        available_pairs = inv["noise"]["maximum_event_disjoint_null_pairs_from_current_files"]
        support_rows.append(
            {
                "deployment": deployment,
                "required_independent_audit_pairs": required_pairs,
                "available_event_disjoint_null_pairs": available_pairs,
                "support_fraction": available_pairs / required_pairs,
                "passes_null_audit_support": available_pairs >= required_pairs,
                "current_v93_validation_events": 450,
                "current_v93_max_event_disjoint_pairs": 225,
                "required_independent_endpoint_events": 2 * required_pairs,
            }
        )
    support = pd.DataFrame(support_rows)
    support.to_csv(output / "tables/G05_FINITE_SAMPLE_SUPPORT.csv", index=False, encoding="utf-8-sig")

    minimum_null_pe_events = 2 * required_pairs * len(DEPLOYMENTS)
    minimum_lens_pe_events = 2 * 500 * 2 * len(DEPLOYMENTS)  # fit+tune, two images
    locked_pe_events = 450 * len(DEPLOYMENTS)
    real_pe_events = sum(strict_counts.values())
    minimum_total_pe_events = minimum_null_pe_events + minimum_lens_pe_events + locked_pe_events + real_pe_events
    bytes_per_dense_map = 12 * 1024 * 1024 * 4
    dense_storage = minimum_total_pe_events * bytes_per_dense_map
    storage = {
        "minimum_null_pe_events": minimum_null_pe_events,
        "minimum_fit_tune_lens_pe_events": minimum_lens_pe_events,
        "locked_pe_events": locked_pe_events,
        "strict_real_pe_events": real_pe_events,
        "minimum_total_pe_events": minimum_total_pe_events,
        "bytes_per_dense_float32_nside1024_map": bytes_per_dense_map,
        "dense_map_storage_bytes": dense_storage,
        "dense_map_storage_gib": dense_storage / 2**30,
        "disk_free_bytes": env["disk_free_bytes"],
        "disk_free_gib": env["disk_free_bytes"] / 2**30,
        "dense_maps_fit_in_current_free_disk": dense_storage <= 0.8 * env["disk_free_bytes"],
        "note": "Sparse/MOC storage could reduce this projection, but no frozen homogeneous PE/MOC pipeline exists in the project.",
    }

    checks = {
        "g0_source_inventory_complete": True,
        "g0_noise_inventory_complete": True,
        "homogeneous_real_and_injection_pe_pipeline_available": False,
        "o3_independent_null_support": bool(support.loc[support.deployment == "GWTC3_O3", "passes_null_audit_support"].iloc[0]),
        "o4a_independent_null_support": bool(support.loc[support.deployment == "GWTC4P1_O4A", "passes_null_audit_support"].iloc[0]),
        "dense_nside1024_storage_within_80pct_free_disk": bool(storage["dense_maps_fit_in_current_free_disk"]),
        "old_rotated_templates_permitted_as_formal_v10_calibration": False,
    }
    pass_all = all(checks.values())
    recommendation = {
        "schema": "sky-background-v10-g05-power-support-recommendation-v1",
        "generated_at_utc": utc_now(),
        "status": "PROSPECTIVE_FEASIBILITY_PASS" if pass_all else "NO_GO_OR_REDESIGN",
        "author_signature_gate_waived": True,
        "scientific_gate_waived": False,
        "sample_size_calculations": {
            "nominal_1pct_ucb_1p5pct_power80": req_1,
            "nominal_5pct_ucb_6pct_power80": req_5,
            "binding_required_independent_audit_pairs_per_deployment_major_stratum": required_pairs,
            "zero_failure_minimum_n_by_ucb": zero_failure,
            "method": "Exact binomial probability with one-sided 95% Clopper-Pearson upper bounds",
        },
        "support_by_deployment": support.to_dict(orient="records"),
        "pe_and_storage_projection": storage,
        "checks": checks,
        "decision": (
            "G1_TO_G5_AUTHORIZED_BY_FEASIBILITY"
            if pass_all
            else "DO_NOT_START_G1_TO_G5; retain v9.3 and redesign homogeneous PE/null acquisition"
        ),
        "blocking_reasons": [name for name, passed in checks.items() if not passed],
    }
    write_json(output / "G05_POWER_SUPPORT_RECOMMENDATION.json", recommendation)

    metrics = baseline_metrics(v93)
    metrics.to_csv(output / "tables/V93_HISTORICAL_BASELINE_CONTEXT.csv", index=False, encoding="utf-8-sig")
    make_figure(output / "figures/fig_g05_feasibility", support, storage)

    status = "HOLD_NO_GO_OR_REDESIGN" if not pass_all else "G05_PASS_READY_FOR_G1"
    write_json(
        output / "STATUS.json",
        {
            "run_id": output.name,
            "status": status,
            "updated_at_utc": utc_now(),
            "science_computation_performed": False,
            "real_catalog_reranked": False,
            "v93_modified": False,
            "next_action": recommendation["decision"],
        },
    )
    sentinel = output / ("HOLD_NO_GO_OR_REDESIGN" if not pass_all else "G05_PASS_READY_FOR_G1")
    sentinel.write_text(status + "\n", encoding="ascii")

    report = f"""# 真实运行期匹配天空背景 v10：G0/G0.5 作者直授权审计

生成时间：{utc_now()}

## 结论

作者已明确授权跳过行政签署步骤，但没有放弃科学质量门槛。G0 source/noise
库存已完成，G0.5 的结论为 **{recommendation['status']}**。因此本 run
{'没有启动 G1--G5，也没有重排真实候选。' if not pass_all else '可进入 G1。'}

## 关键原因

1. 当前 v9.3 注入天空图来自真实 PE 模板旋转；真实目录来自公开 PE posterior。
   两者不是协议要求的同一冻结 PE 管线，不能作为正式 v10 null calibration。
2. nominal 1%/5% FPR Gate 的 binding 要求为每个 deployment × major stratum
   至少 **{required_pairs:,}** 个独立 audit pairs。
3. 当前下载文件经 H1/L1 DATA+CBC_CAT2、已知事件 ±128 s 排除和 320 s
   独立块划分后，O3/O4a 最多提供的 event-disjoint null pairs 分别为
   **{int(support.iloc[0].available_event_disjoint_null_pairs):,}** 和
   **{int(support.iloc[1].available_event_disjoint_null_pairs):,}**。
4. 最低同质 PE 事件投影为 **{minimum_total_pe_events:,}**；若把每张 Nside=1024
   map 保存为 dense float32，单 maps 即约 **{storage['dense_map_storage_gib']:.1f} GiB**，
   当前空闲约 **{storage['disk_free_gib']:.1f} GiB**。

## 本轮做了什么

- 审计 GW-LMC source bank、全局 event ID 和剩余候选 source；
- 逐个读取现有 H1/L1 HDF5 的真实 DQ mask，不只看 manifest 状态；
- 要求 DATA 与 CBC_CAT2 同时通过，并排除所有已知事件 ±128 s；
- 计算保守、互不重叠的 320 s noise blocks；
- 用 exact binomial 与 Clopper--Pearson 上界重算 Gate E 样本量；
- 核对 v9.3 的注入/真实天空图 provenance、严格真实目录规模和历史指标；
- 只读检查输入，未覆盖 v9.3、未修改论文、未查看新 locked sample。

## 解释边界

跳过签名只能解除行政 HOLD，不能把模板旋转改名为同质 PE，也不能把相关
pairs 当成独立 audit units。若忽略这些门槛直接给出 R@10 或真实 Top-10，
结果将违反用户指定协议的核心科学目的。

## 可执行的重新设计

1. 冻结并实现一套真实事件与注入事件完全相同的 BAYESTAR-to-BAYESTAR 或
   sky-only Bilby pipeline；不能混用公开 full-PE map 与旋转模板。
2. 增加 O3 的 source/noise-disjoint off-source 数据块；O4a 也按相同规则补齐。
3. 以 sparse MOC 或分块 overlap 代替逐事件 dense Nside=1024 数组。
4. 获得每个主要层至少 {required_pairs:,} 个独立 calibration audit pairs 后，
   新建 run ID，再执行 G1--G5。
"""
    (output / "reports/G0_G05_AUTHOR_OVERRIDE_REPORT_CN.md").write_text(report, encoding="utf-8")
    index = """# Results Index

- `G0_SOURCE_NOISE_INVENTORY.json`: source/noise/PE provenance inventory.
- `G05_POWER_SUPPORT_RECOMMENDATION.json`: prospective finite-sample and resource decision.
- `tables/G0_OFFSOURCE_NOISE_BLOCK_INVENTORY.csv`: per-HDF5 H1/L1 DQ block counts.
- `tables/G05_FINITE_SAMPLE_SUPPORT.csv`: required versus available independent audit units.
- `tables/V93_HISTORICAL_BASELINE_CONTEXT.csv`: read-only v9.3 context, not a new result.
- `figures/fig_g05_feasibility.pdf` and `.png`: feasibility summary.
- `reports/G0_G05_AUTHOR_OVERRIDE_REPORT_CN.md`: complete Chinese report.
"""
    (output / "RESULTS_INDEX_CN.md").write_text(index, encoding="utf-8")
    return 0 if pass_all else 21


if __name__ == "__main__":
    raise SystemExit(main())
