#!/usr/bin/env python3
"""Execute G0-G9 for the waveform-domain multiscale exploration.

The script enforces validation-only choices and opens the frozen BAYESTAR
locked test once. Historical result trees are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path("/root/autodl-tmp/gw-catalog")
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE = PROJECT / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
BASELINE_PACKAGE = PROJECT / "packages/bayestar_injection_sky_pe_20260901_20260901T102000Z_deliverables.tar.gz"
BASELINE_SHA256 = "3ea3cf13992a5154e0dd757ab6087b73344cf28eb0ccf5e4a8a903b0e83850af"
INPUT_ROOT = PROJECT / "results/real_noise_injection_v7_peak2s_formal_20260722"
SOURCE_ROOT = PROJECT / "results/real_noise_injection_v5_physical_source_20260721"
BASE_SCRIPT = PROJECT / "scripts/experiments/gwtc_c_scheme_ordering_confirmation_20260831.py"
MODEL_SEEDS = (202609031, 202609032, 202609033)
EVALUATION_SEEDS = (202607241, 202607242, 202607243)
SEED_MAP = dict(zip(MODEL_SEEDS, EVALUATION_SEEDS, strict=True))
DEPLOYMENTS = ("gwtc3", "gwtc4")
WINDOWS = ((2,), (2, 8), (2, 16))
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
R10_MARGIN = 0.02
AUPRC_MARGIN = 0.005


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SCRIPT, "waveform_experiment_c_base")
TRAIN = load_module(SCRIPT_DIR / "waveform_multiscale_train.py", "waveform_multiscale_train_runtime")
PHYS = load_module(PROJECT / "scripts/real_search/physical_common.py", "waveform_experiment_physical")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def initialize(root: Path) -> None:
    if root.exists() and not (root / "contracts/BASELINE_IDENTITY.json").exists():
        raise RuntimeError(f"Refusing ambiguous existing directory: {root}")
    for name in ("contracts", "manifests", "configs", "models", "results", "tables", "figures", "reports", "logs", "scripts", "cache", "data"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for script in ("waveform_multiscale_data.py", "waveform_multiscale_train.py", "waveform_domain_multiscale_exploratory.py"):
        shutil.copy2(SCRIPT_DIR / script, root / "scripts" / script)


def g0(root: Path) -> None:
    marker = root / "contracts/G0_BASELINE_REPRODUCED.json"
    if marker.exists():
        return
    if not BASELINE.is_dir() or not BASELINE_PACKAGE.is_file():
        raise RuntimeError("Required BAYESTAR baseline is missing")
    actual_hash = sha256_file(BASELINE_PACKAGE)
    if actual_hash != BASELINE_SHA256:
        raise RuntimeError(f"Baseline package SHA mismatch: {actual_hash}")
    summary_path = BASELINE / "results/locked_test_retrieval_metrics_summary.csv"
    metrics = pd.read_csv(summary_path)
    expected = {
        ("gwtc3", "waveform_only"): 0.555,
        ("gwtc4", "waveform_only"): 0.474,
        ("gwtc3", "BAYESTAR_C_fixed"): 0.865,
        ("gwtc4", "BAYESTAR_C_fixed"): 0.777,
    }
    observed = {}
    for key, target in expected.items():
        row = metrics.loc[metrics.deployment.eq(key[0]) & metrics.method.eq(key[1])]
        if len(row) != 1:
            raise RuntimeError(f"Missing baseline fingerprint row {key}")
        value = float(row.overall_r_at_10_mean.iloc[0])
        observed[f"{key[0]}:{key[1]}"] = value
        if abs(value - target) > 0.001:
            raise RuntimeError(f"Baseline metric mismatch for {key}: {value} vs {target}")
    protected = [
        BASELINE_PACKAGE,
        summary_path,
        BASELINE / "results/locked_test_retrieval_metrics_per_seed.csv",
        BASELINE / "contracts/ANALYSIS_CONTRACT.json",
        BASELINE / "contracts/selected_config.json",
    ]
    identity = {
        "schema": "waveform-domain-multiscale-baseline-v1",
        "created_utc": utc_stamp(),
        "baseline_result_root": BASELINE,
        "baseline_package": BASELINE_PACKAGE,
        "expected_package_sha256": BASELINE_SHA256,
        "actual_package_sha256": actual_hash,
        "fingerprint_metrics": observed,
        "status": "reproduced",
    }
    write_json(root / "contracts/BASELINE_IDENTITY.json", identity)
    hashes = pd.DataFrame({"path": [str(p) for p in protected], "sha256_before": [sha256_file(p) for p in protected]})
    write_csv(root / "manifests/HISTORICAL_HASHES_BEFORE.csv", hashes)
    gate = {
        "schema": "waveform-domain-multiscale-gates-v1",
        "frozen_before_new_training": True,
        "window_gate": {
            "per_run_r10_noninferiority_margin": R10_MARGIN,
            "per_run_auprc_noninferiority_margin": AUPRC_MARGIN,
            "plateau_tolerance": {"r_at_10": 0.01, "auprc": 0.005, "low_mass_r_at_10": 0.02},
        },
        "hard_negative_gate": {
            "r10_margin": R10_MARGIN, "auprc_margin": AUPRC_MARGIN,
            "required_relative_reduction_top100_catastrophic_mass_false": 0.10,
        },
        "uncertainty_head_gate": {
            "r10_margin": R10_MARGIN, "auprc_margin": AUPRC_MARGIN,
            "required_mean_logmc_mae_reduction": 0.05, "acceptable_90pct_coverage": [0.75, 0.98],
        },
        "embedding_gate": {
            "r10_margin": R10_MARGIN, "auprc_margin": AUPRC_MARGIN,
            "required_mean_effective_rank_gain": 0.10,
        },
        "locked_test_gate": {
            "mean_r10_margin": R10_MARGIN, "mean_auprc_margin": AUPRC_MARGIN,
            "minimum_noninferior_seeds_per_run": 2, "maximum_mean_false_burden_multiplier": 1.10,
        },
        "real_PE_or_official_used_for_selection": False,
        "final_status": FINAL_STATUS,
    }
    write_json(root / "contracts/GATE_CONTRACT.json", gate)
    write_json(root / "contracts/PREPROCESSING_CONTRACT.json", {
        "frozen_base_preprocessing": "off-source PSD whitening; Tukey taper; 40-580 Hz physical bandpass; anti-aliased 4096-to-2048 Hz resampling",
        "short_view": "last 2 s, 4096 points",
        "long_views": "last 8 or 16 s, anti-aliased resample to 4096 points; short 2 s branch retained",
        "branch_fusion": "L2-normalized mean of shared-weight InceptionAttention embeddings",
    })
    write_json(marker, {"passed": True, "timestamp_utc": utc_stamp(), "observed": observed})


def run_checked(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_stamp()}] {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def g1(root: Path) -> None:
    marker = root / "contracts/G1_DATA_COMPLETE.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("audit_version") == "global-source-noise-v3":
        return
    run_checked(
        [sys.executable, str(root / "scripts/waveform_multiscale_data.py"), "--root", str(root), "--deployment", "all"],
        root / "logs/G1_data.log",
    )
    rows = []
    for deployment in DEPLOYMENTS:
        data = root / "data/development" / deployment
        split = json.loads((data / "DATA_SPLIT_CONTRACT.json").read_text(encoding="utf-8"))
        noise = json.loads((root / "data/noise_banks" / deployment / "noise_partition_summary.json").read_text(encoding="utf-8"))
        source_summary = json.loads((root / "data/source_banks" / deployment / "physical_source_bank_summary.json").read_text(encoding="utf-8"))
        if (
            split["source_parent_overlap_count"]
            or source_summary["selected_frozen_origin_overlap"]
            or noise["historical_parent_event_overlap"]
            or noise["train_validation_parent_event_overlap"]
        ):
            raise RuntimeError(f"G1 leakage audit failed for {deployment}")
        for family in ("sis", "pm"):
            train = pd.read_parquet(data / f"{family}_train_metadata.parquet")
            val = pd.read_parquet(data / f"{family}_validation_metadata.parquet")
            source_meta = pd.read_parquet(
                root / "data/source_banks" / deployment / f"{family.upper()}_data_0222/physical_source_pair_metadata.parquet"
            ).reset_index(drop=True)
            origin_lookup = source_meta.gwlmc_origin_row.astype(int)
            train_origins = train.source_index.astype(int).map(origin_lookup)
            validation_origins = val.source_index.astype(int).map(origin_lookup)
            if train_origins.isna().any() or validation_origins.isna().any():
                raise RuntimeError(f"Missing lens-environment origin mapping for {deployment}/{family}")
            if train.waveform_parent_uid.nunique() != train.source_index.nunique():
                raise RuntimeError(f"Non-unique training waveform parent for {deployment}/{family}")
            if val.waveform_parent_uid.nunique() != val.source_index.nunique():
                raise RuntimeError(f"Non-unique validation waveform parent for {deployment}/{family}")
            parent_overlap = set(train.waveform_parent_uid) & set(val.waveform_parent_uid)
            if parent_overlap:
                raise RuntimeError(f"Training/validation waveform parent overlap for {deployment}/{family}")
            rows.append({
                "deployment": deployment, "family": family.upper(),
                "train_rows": len(train), "train_unique_sources": train.source_index.nunique(),
                "validation_rows": len(val), "validation_unique_sources": val.source_index.nunique(),
                "train_unique_lens_environment_origins": train_origins.nunique(),
                "validation_unique_lens_environment_origins": validation_origins.nunique(),
                "train_validation_lens_environment_origin_overlap": len(set(train_origins) & set(validation_origins)),
                "train_validation_waveform_parent_overlap": len(parent_overlap),
                "frozen_validation_test_origin_overlap": source_summary["selected_frozen_origin_overlap"],
                "train_mc_min": train.chirp_mass_detector.min(), "train_mc_median": train.chirp_mass_detector.median(),
                "train_mc_max": train.chirp_mass_detector.max(), "train_snr_min": min(train.target_snr_a.min(), train.target_snr_b.min()),
                "train_snr_max": max(train.target_snr_a.max(), train.target_snr_b.max()),
                "detector_network": "H1L1",
                "train_duration_from_40hz_min_s": train.duration_from_40hz_s.min(),
                "train_duration_from_40hz_median_s": train.duration_from_40hz_s.median(),
                "train_duration_from_40hz_max_s": train.duration_from_40hz_s.max(),
                "max_abs_target_recovered_snr_error": max(
                    float(np.max(np.abs(train.target_snr_a - train.a_recovered_optimal_network_snr))),
                    float(np.max(np.abs(train.target_snr_b - train.b_recovered_optimal_network_snr))),
                ),
            })
    write_csv(root / "tables/G1_DATA_STRATA_SUMMARY.csv", pd.DataFrame(rows))
    write_json(marker, {
        "passed": True,
        "timestamp_utc": utc_stamp(),
        "audit_version": "global-source-noise-v3",
        "source_noise_parent_intersections": 0,
        "frozen_validation_test_gwlmc_origin_intersections": 0,
        "training_validation_waveform_parent_intersections": 0,
    })


def config_id(windows: tuple[int, ...], hard: bool, uncertainty: bool, embed: bool) -> str:
    window = "2s" if windows == (2,) else "2plus" + str(windows[-1]) + "s"
    suffix = ""
    if hard:
        suffix += "_H"
    if uncertainty:
        suffix += "_U"
    if embed:
        suffix += "_E"
    return f"D1-{window}{suffix or '_base'}"


def train_config(root: Path, deployment: str, windows: tuple[int, ...], seed: int, hard: bool, uncertainty: bool, embed: bool, stage: str) -> Path:
    cid = config_id(windows, hard, uncertainty, embed)
    output = root / "models" / stage / cid / deployment / f"seed_{seed}"
    if not (output / "summary.json").exists():
        command = [
            sys.executable, str(root / "scripts/waveform_multiscale_train.py"),
            "--root", str(root), "--deployment", deployment, "--windows", ",".join(map(str, windows)),
            "--seed", str(seed), "--epochs", "40", "--output", str(output),
        ]
        if hard:
            command.append("--hard-negatives")
        if uncertainty:
            command.append("--uncertainty-head")
        if embed:
            command.append("--embedding-regularization")
        run_checked(command, root / f"logs/{stage}_{cid}_{deployment}_seed{seed}.log")
    return output


def summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def metric_value(path: Path, key: str) -> float:
    return float(summary(path)["best_validation"][key])


def noninferior(candidate: Path, reference: Path) -> bool:
    return bool(
        metric_value(candidate, "composite_r_at_10") >= metric_value(reference, "composite_r_at_10") - R10_MARGIN
        and metric_value(candidate, "composite_pair_auprc") >= metric_value(reference, "composite_pair_auprc") - AUPRC_MARGIN
    )


def g2(root: Path) -> dict[str, Any]:
    marker = root / "contracts/G2_WINDOW_SELECTED.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    paths: dict[tuple[int, ...], dict[str, Path]] = {}
    rows = []
    for windows in WINDOWS:
        paths[windows] = {}
        for deployment in DEPLOYMENTS:
            path = train_config(root, deployment, windows, MODEL_SEEDS[0], False, False, False, "G2")
            paths[windows][deployment] = path
            rows.append({"stage": "G2", "config_id": config_id(windows, False, False, False), **summary(path)})
    write_csv(root / "tables/G2_WINDOW_PILOT.csv", pd.json_normalize(rows))
    baseline = paths[(2,)]
    feasible = []
    for windows in WINDOWS:
        if all(noninferior(paths[windows][dep], baseline[dep]) for dep in DEPLOYMENTS):
            low_mass = np.mean([
                0.5 * (metric_value(paths[windows][dep], "mass_bin_0_r_at_10") + metric_value(paths[windows][dep], "mass_bin_1_r_at_10"))
                for dep in DEPLOYMENTS
            ])
            feasible.append((
                windows,
                float(low_mass),
                float(np.mean([metric_value(paths[windows][dep], "composite_r_at_10") for dep in DEPLOYMENTS])),
                float(np.mean([metric_value(paths[windows][dep], "composite_pair_auprc") for dep in DEPLOYMENTS])),
            ))
    if not feasible:
        chosen = (2,)
        fallback = True
    else:
        best = max(feasible, key=lambda row: (row[1], row[2], row[3], -row[0][-1]))
        near = [row for row in feasible if best[1] - row[1] <= 0.02 and best[2] - row[2] <= 0.01 and best[3] - row[3] <= 0.005]
        chosen = min(near, key=lambda row: row[0][-1])[0]
        fallback = False
    payload = {
        "passed": True, "timestamp_utc": utc_stamp(), "selected_windows_seconds": chosen,
        "selected_config_id": config_id(chosen, False, False, False),
        "fallback_to_2s": fallback, "locked_test_opened": False, "real_PE_used": False,
    }
    write_json(marker, payload)
    return payload


def g3(root: Path, g2_config: dict[str, Any]) -> dict[str, Any]:
    marker = root / "contracts/G3_HARD_NEGATIVE_SELECTED.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    windows = tuple(g2_config["selected_windows_seconds"])
    base_paths = {dep: train_config(root, dep, windows, MODEL_SEEDS[0], False, False, False, "G2") for dep in DEPLOYMENTS}
    hard_paths = {dep: train_config(root, dep, windows, MODEL_SEEDS[0], True, False, False, "G3") for dep in DEPLOYMENTS}
    rows = []
    for dep in DEPLOYMENTS:
        base, hard = summary(base_paths[dep]), summary(hard_paths[dep])
        bfrac = float(base["best_validation"]["top100_false_catastrophic_mass_fraction"])
        hfrac = float(hard["best_validation"]["top100_false_catastrophic_mass_fraction"])
        rows.append({
            "deployment": dep, "base_config": config_id(windows, False, False, False),
            "hard_config": config_id(windows, True, False, False),
            "base_r10": base["best_validation"]["composite_r_at_10"], "hard_r10": hard["best_validation"]["composite_r_at_10"],
            "base_auprc": base["best_validation"]["composite_pair_auprc"], "hard_auprc": hard["best_validation"]["composite_pair_auprc"],
            "base_catastrophic_fraction": bfrac, "hard_catastrophic_fraction": hfrac,
            "relative_reduction": (bfrac - hfrac) / max(bfrac, 1e-12),
            "noninferior": noninferior(hard_paths[dep], base_paths[dep]),
        })
    audit = pd.DataFrame(rows)
    use_hard = bool(audit.noninferior.all() and audit.relative_reduction.mean() >= 0.10 and (audit.relative_reduction > 0).sum() >= 1)
    write_csv(root / "tables/G3_HARD_NEGATIVE_ABLATION.csv", audit)
    payload = {
        "passed": True, "timestamp_utc": utc_stamp(), "selected_windows_seconds": windows,
        "hard_negatives_selected": use_hard, "selected_config_id": config_id(windows, use_hard, False, False),
        "locked_test_opened": False, "real_PE_used": False,
    }
    write_json(marker, payload)
    return payload


def g4(root: Path, g3_config: dict[str, Any]) -> dict[str, Any]:
    marker = root / "contracts/G4_ARCHITECTURE_SELECTED.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    windows = tuple(g3_config["selected_windows_seconds"])
    hard = bool(g3_config["hard_negatives_selected"])
    base_paths = {dep: train_config(root, dep, windows, MODEL_SEEDS[0], hard, False, False, "G3" if hard else "G2") for dep in DEPLOYMENTS}
    u_paths = {dep: train_config(root, dep, windows, MODEL_SEEDS[0], hard, True, False, "G4") for dep in DEPLOYMENTS}
    e_paths = {dep: train_config(root, dep, windows, MODEL_SEEDS[0], hard, False, True, "G4") for dep in DEPLOYMENTS}
    base_mae = np.mean([metric_value(base_paths[d], "logmc_mae") for d in DEPLOYMENTS])
    u_mae = np.mean([metric_value(u_paths[d], "logmc_mae") for d in DEPLOYMENTS])
    coverage = [metric_value(u_paths[d], "logmc_90_coverage") for d in DEPLOYMENTS]
    u_pass = bool(
        all(noninferior(u_paths[d], base_paths[d]) for d in DEPLOYMENTS)
        and u_mae <= 0.95 * base_mae and all(0.75 <= value <= 0.98 for value in coverage)
    )
    base_rank = np.mean([metric_value(base_paths[d], "embedding_effective_rank") for d in DEPLOYMENTS])
    e_rank = np.mean([metric_value(e_paths[d], "embedding_effective_rank") for d in DEPLOYMENTS])
    e_pass = bool(all(noninferior(e_paths[d], base_paths[d]) for d in DEPLOYMENTS) and e_rank >= 1.10 * base_rank)
    candidates: list[tuple[bool, bool, dict[str, Path]]] = [(False, False, base_paths)]
    if u_pass:
        candidates.append((True, False, u_paths))
    if e_pass:
        candidates.append((False, True, e_paths))
    ue_pass = False
    if u_pass and e_pass:
        ue_paths = {dep: train_config(root, dep, windows, MODEL_SEEDS[0], hard, True, True, "G4") for dep in DEPLOYMENTS}
        ue_pass = bool(all(noninferior(ue_paths[d], base_paths[d]) for d in DEPLOYMENTS))
        if ue_pass:
            candidates.append((True, True, ue_paths))
    def candidate_key(item: tuple[bool, bool, dict[str, Path]]) -> tuple[float, ...]:
        u, e, paths = item
        return (
            -float(np.mean([metric_value(paths[d], "top100_false_catastrophic_mass_fraction") for d in DEPLOYMENTS])),
            float(np.mean([metric_value(paths[d], "composite_r_at_10") for d in DEPLOYMENTS])),
            float(np.mean([metric_value(paths[d], "composite_pair_auprc") for d in DEPLOYMENTS])),
            -float(np.mean([metric_value(paths[d], "logmc_mae") for d in DEPLOYMENTS])),
            float(u) + float(e),
        )
    uncertainty, embed, _ = max(candidates, key=candidate_key)
    rows = []
    reported_variants = [("base", base_paths), ("U", u_paths), ("E", e_paths)]
    if u_pass and e_pass:
        reported_variants.append(("U+E", ue_paths))
    for label, paths in reported_variants:
        for dep in DEPLOYMENTS:
            rows.append({"variant": label, "deployment": dep, **summary(paths[dep])["best_validation"]})
    write_csv(root / "tables/G4_ARCHITECTURE_ABLATION.csv", pd.DataFrame(rows))
    payload = {
        "passed": True, "timestamp_utc": utc_stamp(), "selected_windows_seconds": windows,
        "hard_negatives_selected": hard, "U_gate_passed": u_pass, "E_gate_passed": e_pass,
        "U_plus_E_gate_passed": ue_pass, "uncertainty_head_selected": uncertainty,
        "embedding_regularization_selected": embed,
        "selected_config_id": config_id(windows, hard, uncertainty, embed),
        "locked_test_opened": False, "real_PE_used": False,
    }
    write_json(marker, payload)
    return payload


def locate_pilot_model(root: Path, deployment: str, config: dict[str, Any]) -> Path:
    windows = tuple(config["selected_windows_seconds"])
    hard = bool(config["hard_negatives_selected"])
    uncertainty = bool(config["uncertainty_head_selected"])
    embed = bool(config["embedding_regularization_selected"])
    stage = "G4" if uncertainty or embed else ("G3" if hard else "G2")
    return train_config(root, deployment, windows, MODEL_SEEDS[0], hard, uncertainty, embed, stage)


def g5(root: Path, selected: dict[str, Any]) -> None:
    marker = root / "contracts/G5_THREE_SEED_COMPLETE.json"
    if marker.exists():
        return
    windows = tuple(selected["selected_windows_seconds"])
    hard = bool(selected["hard_negatives_selected"])
    uncertainty = bool(selected["uncertainty_head_selected"])
    embed = bool(selected["embedding_regularization_selected"])
    rows = []
    for deployment in DEPLOYMENTS:
        pilot = locate_pilot_model(root, deployment, selected)
        for seed in MODEL_SEEDS:
            target = root / "models/final" / deployment / f"seed_{seed}"
            if seed == MODEL_SEEDS[0] and not target.exists():
                shutil.copytree(pilot, target)
            elif seed != MODEL_SEEDS[0]:
                trained = train_config(root, deployment, windows, seed, hard, uncertainty, embed, "G5")
                if not target.exists():
                    shutil.copytree(trained, target)
            rows.append({"deployment": deployment, "model_seed": seed, "config_id": selected["selected_config_id"], **summary(target)["best_validation"]})
    metrics = pd.DataFrame(rows)
    write_csv(root / "tables/G5_VALIDATION_METRICS_PER_SEED.csv", metrics)
    stability = []
    for deployment, part in metrics.groupby("deployment"):
        stability.append({
            "deployment": deployment, "n_seeds": len(part),
            "r_at_10_mean": part.composite_r_at_10.mean(), "r_at_10_sd": part.composite_r_at_10.std(ddof=1),
            "auprc_mean": part.composite_pair_auprc.mean(), "auprc_sd": part.composite_pair_auprc.std(ddof=1),
            "logmc_mae_mean": part.logmc_mae.mean(), "effective_rank_mean": part.embedding_effective_rank.mean(),
            "catastrophic_top100_mean": part.top100_false_catastrophic_mass_fraction.mean(),
        })
    write_csv(root / "tables/G5_VALIDATION_METRICS_SUMMARY.csv", pd.DataFrame(stability))
    write_json(marker, {"passed": True, "timestamp_utc": utc_stamp(), "model_seeds": MODEL_SEEDS, "locked_test_opened": False})


def event_array_for_plan(deployment: str, evaluation_seed: int, split: str, retained: pd.DataFrame) -> np.ndarray:
    short = "val" if split == "validation" else "test"
    base = INPUT_ROOT / deployment / f"seed_{evaluation_seed}/data/real_noise_injections/matchroots/LIGO"
    stores = {
        ("SIS", "L1"): np.load(base / "SIS_data_0222/SIS_data_strain_1.npy", mmap_mode="r"),
        ("SIS", "L2"): np.load(base / "SIS_data_0222/SIS_data_strain_2.npy", mmap_mode="r"),
        ("PM", "L1"): np.load(base / "PM_data_0222/PM_data_strain_1.npy", mmap_mode="r"),
        ("PM", "L2"): np.load(base / "PM_data_0222/PM_data_strain_2.npy", mmap_mode="r"),
        ("U", "U"): np.load(base / "Unlensed_data_0222/unlensed_data_strain.npy", mmap_mode="r"),
    }
    output = np.empty((len(retained), 2, 49152), dtype=np.float32)
    for output_index, row in enumerate(retained.itertuples(index=False)):
        family = str(row.family); tag = str(row.tag)
        key = ("U", "U") if tag == "U" else (family, tag)
        output[output_index] = stores[key][int(row.source_index)]
    return output


def attach_frozen_time_sky(frame: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Attach frozen channels only after an exact local pair-index audit."""
    if len(frame) != len(frozen):
        raise RuntimeError(f"Pair table length mismatch: {len(frame)} != {len(frozen)}")
    for column in ("idx_i", "idx_j"):
        if not np.array_equal(frame[column].to_numpy(), frozen[column].to_numpy()):
            raise RuntimeError(f"Pair table ordering mismatch in {column}")
    out = frame.copy()
    out["time_score"] = frozen.time_score.to_numpy()
    out["sky_raw_log_bf"] = frozen.sky_raw_log_bf.to_numpy()
    return out


def component_values(frame: pd.DataFrame, embedding: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    ii = frame.idx_i.to_numpy(np.int64); jj = frame.idx_j.to_numpy(np.int64)
    cosine = np.sum(embedding[ii] * embedding[jj], axis=1)
    delta_mc = np.abs(mean[ii, 0] - mean[jj, 0]) / np.maximum(np.sqrt(sigma[ii, 0] ** 2 + sigma[jj, 0] ** 2), 0.15)
    delta_q = np.abs(mean[ii, 1] - mean[jj, 1]) / np.maximum(np.sqrt(sigma[ii, 1] ** 2 + sigma[jj, 1] ** 2), 0.25)
    return {"cosine": cosine, "mc_similarity": -delta_mc, "q_similarity": -delta_q, "delta_mc": delta_mc, "delta_q": delta_q}


def fit_waveform_calibration(frame: pd.DataFrame, components: dict[str, np.ndarray]) -> tuple[dict[str, Any], pd.DataFrame]:
    stats_values = {
        name: {"median": float(np.median(components[name])), "std": max(float(np.std(components[name])), 1e-6)}
        for name in ("cosine", "mc_similarity", "q_similarity")
    }
    z = {name: (components[name] - values["median"]) / values["std"] for name, values in stats_values.items()}
    rows = []
    best = None
    for mc_weight in (0.0, 0.25, 0.5, 1.0, 2.0):
        for q_weight in (0.0, 0.25, 0.5, 1.0):
            raw = z["cosine"] + mc_weight * z["mc_similarity"] + q_weight * z["q_similarity"]
            metrics = BASE.full_metrics(frame, raw)
            row = {"mc_weight": mc_weight, "q_weight": q_weight, **metrics}
            rows.append(row)
            key = (
                metrics["macro_r_at_10"], metrics["average_precision"], -metrics["false_at_recall_0p5"],
                metrics["macro_r_at_1"], -(mc_weight + q_weight), -mc_weight, -q_weight,
            )
            if best is None or key > best[0]:
                best = (key, mc_weight, q_weight, raw)
    assert best is not None
    labels = frame.is_true_pair.astype(bool).to_numpy()
    calibration_candidates = []
    for bandwidth in (0.75, 1.0, 1.5):
        lookup = PHYS.fit_score_likelihood_ratio(best[3][labels], best[3][~labels], bandwidth_scale=bandwidth)
        score = PHYS.apply_score_likelihood_ratio(best[3], lookup)
        metrics = BASE.full_metrics(frame, score)
        calibration_candidates.append((
            (metrics["average_precision"], metrics["macro_r_at_10"], -metrics["false_at_recall_0p5"], -abs(bandwidth - 1.0)),
            bandwidth, lookup, metrics,
        ))
    chosen = max(calibration_candidates, key=lambda item: item[0])
    config = {
        "feature_standardization": stats_values,
        "mc_weight": float(best[1]), "q_weight": float(best[2]),
        "bandwidth_scale": float(chosen[1]), "likelihood_ratio": chosen[2],
        "validation_metrics": chosen[3],
        "selection_order": "macro R@10, AUPRC, F50, R@1; then LR bandwidth by AUPRC/R@10/F50",
    }
    return config, pd.DataFrame(rows)


def apply_waveform_calibration(components: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    z = {}
    for name in ("cosine", "mc_similarity", "q_similarity"):
        values = config["feature_standardization"][name]
        z[name] = (components[name] - float(values["median"])) / float(values["std"])
    raw = z["cosine"] + float(config["mc_weight"]) * z["mc_similarity"] + float(config["q_weight"]) * z["q_similarity"]
    score = PHYS.apply_score_likelihood_ratio(raw, config["likelihood_ratio"])
    return raw, score


def enrich_waveform_frame(frame: pd.DataFrame, components: dict[str, np.ndarray], raw: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["waveform_embedding_cosine"] = components["cosine"]
    out["waveform_abs_delta_logmc_std"] = components["delta_mc"]
    out["waveform_abs_delta_logitq_std"] = components["delta_q"]
    out["waveform_composite_raw"] = raw
    out["waveform_score"] = score
    return out


def g6(root: Path, selected: dict[str, Any]) -> None:
    marker = root / "contracts/G6_CALIBRATION_FROZEN.json"
    if marker.exists():
        return
    configs: dict[str, Any] = {"schema": "run-specific-waveform-calibration-v1", "deployments": {}}
    metric_rows = []
    for deployment in DEPLOYMENTS:
        configs["deployments"][deployment] = {}
        for model_seed in MODEL_SEEDS:
            eval_seed = SEED_MAP[model_seed]
            retained = BASE.retained_event_plan(deployment, eval_seed, "validation")
            frame = BASE.subset_pair_table(deployment, eval_seed, "validation", retained, allow_test_scores=False)
            full24 = event_array_for_plan(deployment, eval_seed, "validation", retained)
            checkpoint = root / f"models/final/{deployment}/seed_{model_seed}/validation_selected_model.pt"
            embedding, mean, sigma = TRAIN.encode_full24(checkpoint, full24)
            components = component_values(frame, embedding, mean, sigma)
            calibration, grid = fit_waveform_calibration(frame, components)
            raw, score = apply_waveform_calibration(components, calibration)
            enriched = enrich_waveform_frame(frame, components, raw, score)
            out = root / f"results/validation/{deployment}/seed_{model_seed}"
            out.mkdir(parents=True, exist_ok=True)
            enriched.to_parquet(out / "pair_scores_new_waveform.parquet", index=False)
            write_csv(out / "waveform_component_grid.csv", grid)
            write_json(out / "waveform_calibration.json", calibration)
            np.savez_compressed(out / "event_embeddings_predictions.npz", embedding=embedding, mean=mean, sigma=sigma)
            metrics = BASE.full_metrics(enriched, score)
            metric_rows.append({"deployment": deployment, "model_seed": model_seed, "evaluation_seed": eval_seed, "split": "validation", "method": "new_waveform_only", **metrics})
            configs["deployments"][deployment][str(model_seed)] = calibration
    path = root / "configs/FROZEN_WAVEFORM_CALIBRATION.json"
    write_json(path, configs)
    config_hash = sha256_file(path)
    write_csv(root / "results/validation/WAVEFORM_CALIBRATION_METRICS.csv", pd.DataFrame(metric_rows))
    write_json(marker, {
        "passed": True, "timestamp_utc": utc_stamp(), "calibration_sha256": config_hash,
        "locked_test_opened": False, "real_catalog_opened": False,
    })


def metric_record(deployment: str, model_seed: int, eval_seed: int, split: str, method: str, frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    return {
        "deployment": deployment, "model_seed": model_seed, "evaluation_seed": eval_seed,
        "split": split, "method": method, "n_events": int(frame.event_count.iloc[0]),
        "n_pairs": len(frame), "n_true_pairs": int(frame.is_true_pair.sum()),
        **BASE.full_metrics(frame, scores),
        **BASE.system_bootstrap_ci(frame, scores, stable_seed(deployment, model_seed, split, method)),
    }


def old_baseline_rows(method: str) -> pd.DataFrame:
    frame = pd.read_csv(BASELINE / "results/locked_test_retrieval_metrics_per_seed.csv")
    return frame.loc[frame.method.eq(method)].copy()


def g7(root: Path) -> dict[str, Any]:
    marker = root / "contracts/G7_LOCKED_TEST_COMPLETE.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    freeze = json.loads((root / "contracts/G6_CALIBRATION_FROZEN.json").read_text(encoding="utf-8"))
    path = root / "configs/FROZEN_WAVEFORM_CALIBRATION.json"
    if sha256_file(path) != freeze["calibration_sha256"]:
        raise RuntimeError("Waveform calibration changed before locked test")
    configs = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for deployment in DEPLOYMENTS:
        for model_seed in MODEL_SEEDS:
            eval_seed = SEED_MAP[model_seed]
            retained = BASE.retained_event_plan(deployment, eval_seed, "test")
            frame = BASE.subset_pair_table(deployment, eval_seed, "test", retained, allow_test_scores=True)
            full24 = event_array_for_plan(deployment, eval_seed, "test", retained)
            checkpoint = root / f"models/final/{deployment}/seed_{model_seed}/validation_selected_model.pt"
            embedding, mean, sigma = TRAIN.encode_full24(checkpoint, full24)
            components = component_values(frame, embedding, mean, sigma)
            calibration = configs["deployments"][deployment][str(model_seed)]
            raw, score = apply_waveform_calibration(components, calibration)
            enriched = enrich_waveform_frame(frame, components, raw, score)
            out = root / f"results/locked_test/{deployment}/seed_{model_seed}"
            out.mkdir(parents=True, exist_ok=True)
            enriched.to_parquet(out / "pair_scores_new_waveform.parquet", index=False)
            np.savez_compressed(out / "event_embeddings_predictions.npz", embedding=embedding, mean=mean, sigma=sigma)
            rows.append(metric_record(deployment, model_seed, eval_seed, "test", "new_waveform_only", enriched, score))
    metrics = pd.DataFrame(rows)
    write_csv(root / "results/locked_test/WAVEFORM_METRICS_PER_SEED.csv", metrics)
    old = old_baseline_rows("waveform_only").rename(columns={"seed": "evaluation_seed"})
    comparison = metrics.merge(old, on=["deployment", "evaluation_seed"], suffixes=("_new", "_baseline"))
    for key in ("overall_r_at_1", "overall_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"):
        comparison[f"delta_{key}"] = comparison[f"{key}_new"] - comparison[f"{key}_baseline"]
    write_csv(root / "results/locked_test/NEW_VS_BASELINE_WAVEFORM_PAIRED.csv", comparison)
    deployment_gate = {}
    for deployment, part in comparison.groupby("deployment"):
        new_r10 = part.overall_r_at_10_new.mean(); old_r10 = part.overall_r_at_10_baseline.mean()
        new_ap = part.average_precision_new.mean(); old_ap = part.average_precision_baseline.mean()
        r10_seed_pass = int((part.delta_overall_r_at_10 >= -R10_MARGIN).sum())
        ap_seed_pass = int((part.delta_average_precision >= -AUPRC_MARGIN).sum())
        f50_ok = part.false_at_recall_0p5_new.mean() <= 1.10 * part.false_at_recall_0p5_baseline.mean() + 1
        f90_ok = part.false_at_recall_0p9_new.mean() <= 1.10 * part.false_at_recall_0p9_baseline.mean() + 1
        passed = bool(new_r10 >= old_r10 - R10_MARGIN and new_ap >= old_ap - AUPRC_MARGIN and r10_seed_pass >= 2 and ap_seed_pass >= 2 and f50_ok and f90_ok)
        deployment_gate[deployment] = {
            "passed": passed, "new_r10_mean": new_r10, "baseline_r10_mean": old_r10,
            "new_auprc_mean": new_ap, "baseline_auprc_mean": old_ap,
            "r10_noninferior_seed_count": r10_seed_pass, "auprc_noninferior_seed_count": ap_seed_pass,
            "f50_guardrail": bool(f50_ok), "f90_guardrail": bool(f90_ok),
        }
    passed = bool(all(value["passed"] for value in deployment_gate.values()))
    payload = {
        "passed": passed, "timestamp_utc": utc_stamp(), "opened_once": True,
        "deployment_gate": deployment_gate,
        "decision": "PASS_TO_G8" if passed else "FAIL_STOP_BEFORE_G8",
        "real_catalog_opened": False, "final_status": FINAL_STATUS,
    }
    write_json(marker, payload)
    return payload


def g8(root: Path) -> dict[str, Any]:
    marker = root / "contracts/G8_FUSION_COMPLETE.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    configs = json.loads((root / "configs/FROZEN_WAVEFORM_CALIBRATION.json").read_text(encoding="utf-8"))
    selected_weights: dict[str, Any] = {"deployments": {}}
    validation_rows = []
    for deployment in DEPLOYMENTS:
        selected_weights["deployments"][deployment] = {}
        for model_seed in MODEL_SEEDS:
            eval_seed = SEED_MAP[model_seed]
            path = root / f"results/validation/{deployment}/seed_{model_seed}/pair_scores_new_waveform.parquet"
            frame = pd.read_parquet(path)
            bayestar = pd.read_parquet(BASELINE / f"results/{deployment}/seed_{eval_seed}/validation_pair_scores_bayestar_sky.parquet")
            frame = attach_frozen_time_sky(frame, bayestar)
            frozen = BASE.FROZEN_V93_WEIGHTS[deployment][eval_seed]
            retuned, grid = BASE.select_retuned_weights(frame, frozen)
            selected_weights["deployments"][deployment][str(model_seed)] = {
                "evaluation_seed": eval_seed, "C_fixed_weights": frozen, "retuned": retuned,
            }
            write_csv(root / f"results/validation/{deployment}/seed_{model_seed}/fusion_weight_grid.csv", grid)
            for method, weights in (("new_fixed_fusion", frozen), ("new_retuned_fusion", retuned["weights"])):
                validation_rows.append(metric_record(deployment, model_seed, eval_seed, "validation", method, frame, BASE.score_vector(frame, weights)))
    weight_path = root / "configs/FROZEN_FUSION_WEIGHTS.json"
    write_json(weight_path, selected_weights)
    write_csv(root / "results/validation/FUSION_METRICS_PER_SEED.csv", pd.DataFrame(validation_rows))
    freeze_hash = sha256_file(weight_path)

    test_rows = []
    for deployment in DEPLOYMENTS:
        for model_seed in MODEL_SEEDS:
            eval_seed = SEED_MAP[model_seed]
            frame = pd.read_parquet(root / f"results/locked_test/{deployment}/seed_{model_seed}/pair_scores_new_waveform.parquet")
            bayestar = pd.read_parquet(BASELINE / f"results/{deployment}/seed_{eval_seed}/test_pair_scores_bayestar_sky.parquet")
            frame = attach_frozen_time_sky(frame, bayestar)
            conf = selected_weights["deployments"][deployment][str(model_seed)]
            for method, weights in (("new_fixed_fusion", conf["C_fixed_weights"]), ("new_retuned_fusion", conf["retuned"]["weights"])):
                scores = BASE.score_vector(frame, weights)
                test_rows.append(metric_record(deployment, model_seed, eval_seed, "test", method, frame, scores))
                scored = frame.copy(); scored["final_score"] = scores
                scored.to_parquet(root / f"results/locked_test/{deployment}/seed_{model_seed}/pair_scores_{method}.parquet", index=False)
    test_metrics = pd.DataFrame(test_rows)
    write_csv(root / "results/locked_test/FUSION_METRICS_PER_SEED.csv", test_metrics)
    baseline = old_baseline_rows("BAYESTAR_C_fixed").rename(columns={"seed": "evaluation_seed"})
    gate_rows = []
    scheme_pass = {}
    for scheme in ("new_fixed_fusion", "new_retuned_fusion"):
        scheme_pass[scheme] = {}
        for deployment in DEPLOYMENTS:
            new = test_metrics.loc[test_metrics.deployment.eq(deployment) & test_metrics.method.eq(scheme)]
            old = baseline.loc[baseline.deployment.eq(deployment)]
            merged = new.merge(old, on=["deployment", "evaluation_seed"], suffixes=("_new", "_baseline"))
            for _, row in merged.iterrows():
                gate_rows.append({
                    "scheme": scheme, "deployment": deployment, "model_seed": int(row.model_seed),
                    "delta_r10": row.overall_r_at_10_new - row.overall_r_at_10_baseline,
                    "delta_auprc": row.average_precision_new - row.average_precision_baseline,
                    "delta_f50": row.false_at_recall_0p5_new - row.false_at_recall_0p5_baseline,
                    "delta_f90": row.false_at_recall_0p9_new - row.false_at_recall_0p9_baseline,
                })
            passed = bool(
                merged.overall_r_at_10_new.mean() >= merged.overall_r_at_10_baseline.mean() - R10_MARGIN
                and merged.average_precision_new.mean() >= merged.average_precision_baseline.mean() - AUPRC_MARGIN
                and (merged.overall_r_at_10_new - merged.overall_r_at_10_baseline >= -R10_MARGIN).sum() >= 2
                and merged.false_at_recall_0p5_new.mean() <= 1.05 * merged.false_at_recall_0p5_baseline.mean() + 1
                and merged.false_at_recall_0p9_new.mean() <= 1.05 * merged.false_at_recall_0p9_baseline.mean() + 1
            )
            scheme_pass[scheme][deployment] = passed
    preferred = "new_retuned_fusion" if all(scheme_pass["new_retuned_fusion"].values()) else "new_fixed_fusion"
    eligible = bool(all(scheme_pass[preferred].values()))
    write_csv(root / "results/locked_test/FUSION_VS_BASELINE_GATE.csv", pd.DataFrame(gate_rows))
    payload = {
        "passed": eligible, "timestamp_utc": utc_stamp(), "fusion_weights_sha256": freeze_hash,
        "scheme_pass": scheme_pass, "preferred_frozen_scheme": preferred,
        "decision": "ELIGIBLE_FOR_AUTHOR_REVIEW" if eligible else "FUSION_GUARDRAIL_FAIL_RETAIN_BASELINE",
        "real_catalog_opened": False, "final_status": FINAL_STATUS,
    }
    write_json(marker, payload)
    return payload


def real_full24(deployment: str) -> np.ndarray:
    return np.load(SOURCE_ROOT / deployment / "shared/real_event_preprocessed_full24.npy", mmap_mode="r")


def g9(root: Path, fusion: dict[str, Any]) -> None:
    marker = root / "contracts/G9_REAL_READONLY_AUDIT_COMPLETE.json"
    if marker.exists():
        return
    calibrations = json.loads((root / "configs/FROZEN_WAVEFORM_CALIBRATION.json").read_text(encoding="utf-8"))
    weights = json.loads((root / "configs/FROZEN_FUSION_WEIGHTS.json").read_text(encoding="utf-8"))
    official_path = PROJECT / "results/sky_background_fast_followup_v104_20260824_20260824T080136Z/tables/frozen_lvk_historical_candidate_comparison.csv"
    official = pd.read_csv(official_path)
    all_budget = []
    failed_pairs = [
        "GW190924_021846--GW191105_143521", "GW190412--GW191204_171526",
        "GW190924_021846--GW190930_133541",
    ]
    failure_rows = []
    correlation_rows = []
    selected_scheme = fusion["preferred_frozen_scheme"]
    method_key = "retuned" if selected_scheme == "new_retuned_fusion" else "C_fixed_weights"
    for deployment in DEPLOYMENTS:
        seed_frames = []
        prediction_rows = []
        for model_seed in MODEL_SEEDS:
            eval_seed = SEED_MAP[model_seed]
            checkpoint = root / f"models/final/{deployment}/seed_{model_seed}/validation_selected_model.pt"
            embedding, mean, sigma = TRAIN.encode_full24(checkpoint, real_full24(deployment))
            baseline_real = pd.read_parquet(BASELINE / f"results/{deployment}/seed_{eval_seed}/real_strict_pair_scores_C_fixed.parquet")
            components = component_values(baseline_real, embedding, mean, sigma)
            calibration = calibrations["deployments"][deployment][str(model_seed)]
            raw, score = apply_waveform_calibration(components, calibration)
            frame = enrich_waveform_frame(baseline_real, components, raw, score)
            conf = weights["deployments"][deployment][str(model_seed)]
            chosen_weights = conf["retuned"]["weights"] if selected_scheme == "new_retuned_fusion" else conf["C_fixed_weights"]
            ranked = BASE.rank_real(frame, chosen_weights, selected_scheme, model_seed)
            out = root / f"results/real_{deployment}/seed_{model_seed}"
            out.mkdir(parents=True, exist_ok=True)
            ranked.to_parquet(out / "all_pair_ranking.parquet", index=False)
            write_csv(out / "top100.csv", ranked.head(100))
            seed_frames.append(ranked)
            audit = pd.read_csv(SOURCE_ROOT / deployment / "shared/real_event_preprocessing_audit.csv")
            for index, row in audit.iterrows():
                prediction_rows.append({
                    "deployment": deployment, "model_seed": model_seed, "event_index": index,
                    "event_name": row.event_name, "strict_h1l1": bool(row.strict_h1l1_preprocessing_pass),
                    "object_class": row.object_class, "predicted_chirp_mass_msun": float(np.exp(mean[index, 0])),
                    "predicted_logmc_sigma": float(sigma[index, 0]), "embedding_norm": float(np.linalg.norm(embedding[index])),
                })
        prediction_frame = pd.DataFrame(prediction_rows)
        write_csv(root / f"tables/{deployment}_REAL_EVENT_WAVEFORM_PREDICTIONS.csv", prediction_frame)
        consensus = BASE.consensus_real(seed_frames, selected_scheme)
        combined_seed = pd.concat(seed_frames, ignore_index=True)
        extra_columns = [
            column for column in (
                "waveform_embedding_cosine", "waveform_abs_delta_logmc_std",
                "waveform_abs_delta_logitq_std", "waveform_composite_raw",
            ) if column in combined_seed
        ]
        if extra_columns:
            extra = combined_seed.groupby("pair_key", as_index=False)[extra_columns].mean()
            extra = extra.rename(columns={column: f"{column}_mean" for column in extra_columns})
            consensus = consensus.merge(extra, on="pair_key", how="left", validate="one_to_one")
        waveform_order = consensus.sort_values("waveform_score_mean", ascending=False, kind="stable").pair_key
        waveform_rank = pd.Series(np.arange(1, len(waveform_order) + 1), index=waveform_order)
        consensus["waveform_only_consensus_rank"] = consensus.pair_key.map(waveform_rank).astype(int)
        pe_path = PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830" / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        pe = pd.read_parquet(pe_path)
        events = set(consensus.head(100).event_i.astype(str)) | set(consensus.head(100).event_j.astype(str))
        distances = BASE.morph.load_distance_posteriors(deployment, events)
        enriched = BASE.morph.attach_pe_official(deployment, consensus, pe, official, distances)
        enriched = BASE.add_official_columns(enriched)
        enriched.to_parquet(root / f"results/real_{deployment}/consensus_all_pairs_with_pe_official.parquet", index=False)
        write_csv(root / f"results/real_{deployment}/consensus_all_pairs_with_pe_official.csv", enriched)
        write_csv(root / f"tables/{deployment}_REAL_TOP100_WITH_PE_OFFICIAL.csv", enriched.head(100))
        valid_bc = enriched[["waveform_score_mean", "chirp_mass_bhattacharyya_coefficient"]].dropna()
        valid_d = enriched[["waveform_score_mean", "chirp_mass_standardized_posterior_distance"]].dropna()
        correlation_rows.append({
            "deployment": deployment,
            "n_pairs_with_BC": len(valid_bc),
            "spearman_waveform_vs_chirp_mass_BC": float(stats.spearmanr(valid_bc.iloc[:, 0], valid_bc.iloc[:, 1]).statistic) if len(valid_bc) > 2 else np.nan,
            "n_pairs_with_D": len(valid_d),
            "spearman_waveform_vs_negative_chirp_mass_D": float(stats.spearmanr(valid_d.iloc[:, 0], -valid_d.iloc[:, 1]).statistic) if len(valid_d) > 2 else np.nan,
            "interpretation": "descriptive only; unordered pairs share events",
        })
        for budget in (10, 20, 50, 100):
            part = enriched.head(budget)
            all_budget.append({
                "deployment": deployment, "scheme": selected_scheme, "budget": budget, "n_pairs": len(part),
                "chirp_mass_BC_ge_0p5": int((part.chirp_mass_bhattacharyya_coefficient >= 0.5).sum()),
                "chirp_mass_D_le_3": int((part.chirp_mass_standardized_posterior_distance <= 3).sum()),
                "Dmax_le_3": int((part.max_standardized_posterior_distance <= 3).sum()),
                "catastrophic_BC_lt_0p1": int((part.chirp_mass_bhattacharyya_coefficient < 0.1).sum()),
                "catastrophic_D_gt_5": int((part.chirp_mass_standardized_posterior_distance > 5).sum()),
                "official_frontend_overlap": int(part.official_frontend_overlap.sum()),
                "public_hanabi_table_overlap": int(part.public_hanabi_table_overlap.sum()),
            })
        catastrophic = enriched.head(100).loc[
            (enriched.head(100).chirp_mass_bhattacharyya_coefficient < 0.1)
            | (enriched.head(100).chirp_mass_standardized_posterior_distance > 5)
        ].pair_key.astype(str).tolist()
        tracked_pairs = list(dict.fromkeys(failed_pairs + catastrophic))
        old_consensus = pd.read_parquet(BASELINE / f"results/{deployment}/real_consensus_with_pe_official_C_fixed.parquet")
        for pair_key in tracked_pairs:
            row = enriched.loc[enriched.pair_key.eq(pair_key)]
            if not row.empty:
                item = row.iloc[0]
                old = old_consensus.loc[old_consensus.pair_key.eq(pair_key)]
                old_item = old.iloc[0] if not old.empty else None
                pi = prediction_frame.loc[prediction_frame.event_name.eq(str(item.event_i))]
                pj = prediction_frame.loc[prediction_frame.event_name.eq(str(item.event_j))]
                failure_rows.append({
                    "deployment": deployment, "pair_key": pair_key,
                    "tracking_reason": "pre_registered" if pair_key in failed_pairs else "new_top100_catastrophic_PE",
                    "old_C_fixed_consensus_rank": int(old_item.consensus_rank) if old_item is not None else np.nan,
                    "old_waveform_score_mean": float(old_item.waveform_score_mean) if old_item is not None else np.nan,
                    "new_consensus_rank": int(item.consensus_rank),
                    "new_waveform_only_rank": int(item.waveform_only_consensus_rank),
                    "new_waveform_score_mean": float(item.waveform_score_mean),
                    "new_embedding_cosine_mean": float(item.get("waveform_embedding_cosine_mean", np.nan)),
                    "predicted_chirp_mass_i_mean_msun": float(pi.predicted_chirp_mass_msun.mean()) if len(pi) else np.nan,
                    "predicted_chirp_mass_j_mean_msun": float(pj.predicted_chirp_mass_msun.mean()) if len(pj) else np.nan,
                    "predicted_logmc_sigma_i_mean": float(pi.predicted_logmc_sigma.mean()) if len(pi) else np.nan,
                    "predicted_logmc_sigma_j_mean": float(pj.predicted_logmc_sigma.mean()) if len(pj) else np.nan,
                    "chirp_mass_BC": float(item.chirp_mass_bhattacharyya_coefficient),
                    "chirp_mass_D": float(item.chirp_mass_standardized_posterior_distance),
                    "Dmax": float(item.max_standardized_posterior_distance),
                    "official_stage": item.official_stage_and_conclusion,
                })
    write_csv(root / "tables/REAL_PE_OFFICIAL_BUDGET_SUMMARY.csv", pd.DataFrame(all_budget))
    write_csv(root / "tables/REAL_WAVEFORM_PE_CORRELATION_AUDIT.csv", pd.DataFrame(correlation_rows))
    write_csv(root / "tables/KNOWN_FAILURE_PAIR_AUDIT.csv", pd.DataFrame(failure_rows))
    write_json(marker, {
        "passed": True, "timestamp_utc": utc_stamp(), "scheme": selected_scheme,
        "used_for_model_or_weight_selection": False, "candidate_claim": "ranked follow-up candidates, not detections",
        "final_status": FINAL_STATUS,
    })


def run_all(root: Path) -> None:
    initialize(root)
    g0(root)
    g1(root)
    selected2 = g2(root)
    selected3 = g3(root, selected2)
    selected4 = g4(root, selected3)
    g5(root, selected4)
    g6(root, selected4)
    locked = g7(root)
    if not locked["passed"]:
        write_json(root / "contracts/FINAL_STATUS.json", {"status": FINAL_STATUS, "reason": "G7 locked-test waveform gate failed", "timestamp_utc": utc_stamp()})
        return
    fusion = g8(root)
    g9(root, fusion)
    write_json(root / "contracts/FINAL_STATUS.json", {
        "status": FINAL_STATUS, "timestamp_utc": utc_stamp(),
        "G7_passed": locked["passed"], "G8_passed": fusion["passed"],
        "historical_outputs_overwritten": False, "paper_modified": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"), default="all")
    args = parser.parse_args()
    initialize(args.root)
    if args.stage == "all":
        run_all(args.root)
        return
    g0(args.root)
    if args.stage == "g0": return
    g1(args.root)
    if args.stage == "g1": return
    s2 = g2(args.root)
    if args.stage == "g2": return
    s3 = g3(args.root, s2)
    if args.stage == "g3": return
    s4 = g4(args.root, s3)
    if args.stage == "g4": return
    g5(args.root, s4)
    if args.stage == "g5": return
    g6(args.root, s4)
    if args.stage == "g6": return
    locked = g7(args.root)
    if args.stage == "g7" or not locked["passed"]: return
    fusion = g8(args.root)
    if args.stage == "g8": return
    g9(args.root, fusion)


if __name__ == "__main__":
    main()
