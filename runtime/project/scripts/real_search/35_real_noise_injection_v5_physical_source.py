#!/usr/bin/env python3
"""Formal O3/O4a real-noise experiment using physical H1/L1 source strain.

This is the authoritative replacement for the invalid v3/v4 waveform branch.
It reuses the independently audited time-likelihood-ratio, sky-Bayes-factor,
real-event preprocessing, scoring, and PE-follow-up components, while replacing
the source bank and every derived waveform artifact.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.physical_common import MODEL_SAMPLES, write_json  # noqa: E402
from scripts.real_search.physical_source_v5_common import (  # noqa: E402
    audit_physical_source_bank,
    draw_detected_pair_snrs,
    draw_unlensed_snr,
    empirical_detected_snrs,
)


OUT_ROOT = REPO / "results" / "real_noise_injection_v5_physical_source_20260721"
FORMAL_SEEDS = (202607221, 202607222, 202607223)
VARIANTS_PER_SOURCE = 8
PRETRAIN_EPOCHS = 12
ARCHITECTURE = "dual"
ADAPTATION = "teacher"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v3 = load_module("physical_v3_for_v5", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
formal = load_module("physical_v4_helpers_for_v5", REPO / "scripts" / "experiments" / "31_real_noise_injection_v4_formal.py")


def source_bank(shared_dir: Path) -> Path:
    return shared_dir / "physical_h1l1_source_bank"


def ensure_source_bank(deployment: str, shared_dir: Path, samples: int, seed: int) -> dict[str, Any]:
    root = source_bank(shared_dir)
    marker = root / "physical_source_bank_summary.json"
    if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")).get("status") != "complete":
        log = shared_dir.parent.parent / "logs" / f"generate_{deployment.lower()}_physical_source.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO / "scripts" / "real_search" / "34_generate_physical_h1l1_source_bank.py"),
            "--deployment",
            deployment,
            "--schedule",
            str(shared_dir / "h1l1_live_schedule.csv"),
            "--out-root",
            str(root),
            "--n-per-family",
            str(samples),
            "--n-unlensed",
            str(samples),
            "--seed",
            str(seed),
        ]
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    summary = json.loads(marker.read_text(encoding="utf-8"))
    audit = audit_physical_source_bank(root)
    write_json(shared_dir / "physical_source_units_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Physical source bank failed frozen units audit: {audit}")
    return summary


def materialize_dataset_v5(
    seed_dir: Path,
    shared_dir: Path,
    deployment: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    base = seed_dir / "data" / "real_noise_injections"
    marker = base / "compact_dataset_summary.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    bank = source_bank(shared_dir)
    units_audit = audit_physical_source_bank(bank)
    if not units_audit["passed"]:
        raise RuntimeError("Refusing to materialize from a nonphysical source bank")
    references = np.load(shared_dir / "noise_reference_bank.npy", mmap_mode="r")
    psds = np.load(shared_dir / "noise_psd_bank.npy", mmap_mode="r")
    psd_frequency = np.load(shared_dir / "noise_psd_frequency.npy")
    empirical = empirical_detected_snrs(seed_dir / "data" / "event_manifest.csv")
    rng = np.random.default_rng(seed)
    root = base / "matchroots" / "LIGO"
    rows: list[dict[str, Any]] = []

    def prepare_one(clean: np.ndarray, target_snr: float):
        noise, psd, bank_index, offset = v3.draw_noise_window(references, psds, rng)
        pure, mixed, audit = v3._preprocess_injection(clean, noise, psd_frequency, psd, target_snr)
        return pure, mixed, {**audit, "noise_bank_index": bank_index, "noise_offset_samples": offset}

    for family in v3.FAMILIES:
        source = bank / f"{family}_data_0222"
        clean1 = np.load(source / f"{family}_h_strain_1.npy", mmap_mode="r")[:samples]
        clean2 = np.load(source / f"{family}_h_strain_2.npy", mmap_mode="r")[:samples]
        pair_meta = pd.read_parquet(source / "physical_source_pair_metadata.parquet").sort_values("pair_id").reset_index(drop=True).iloc[:samples]
        out = root / f"{family}_data_0222"
        out.mkdir(parents=True, exist_ok=True)
        pure1 = np.empty((samples, 2, MODEL_SAMPLES), dtype=np.float32)
        pure2 = np.empty_like(pure1)
        noisy1 = np.empty_like(pure1)
        noisy2 = np.empty_like(pure1)
        for idx in range(samples):
            source_row = pair_meta.iloc[idx]
            snr1, snr2 = draw_detected_pair_snrs(source_row, empirical, rng)
            pure1[idx], noisy1[idx], audit1 = prepare_one(clean1[idx], snr1)
            pure2[idx], noisy2[idx], audit2 = prepare_one(clean2[idx], snr2)
            rows.append(
                {
                    "family": family,
                    "physical_lens_group": str(source_row.physical_lens_group),
                    "sample_index": idx,
                    "gwlmc_row": int(source_row.gwlmc_row),
                    "gps_image1": float(source_row.gps_image1),
                    "gps_image2": float(source_row.gps_image2),
                    "delay_days": float(source_row.delay_days),
                    "snr_ratio_prior": float(source_row.proposal_snr_ratio),
                    "target_snr_image1": snr1,
                    "target_snr_image2": snr2,
                    "ra_true": float(source_row.ra),
                    "dec_true": float(source_row.dec),
                    "morse_image1": float(source_row.morse_image1),
                    "morse_image2": float(source_row.morse_image2),
                    **{f"image1_{key}": value for key, value in audit1.items()},
                    **{f"image2_{key}": value for key, value in audit2.items()},
                }
            )
        np.save(out / f"{family}_h_strain_1.npy", pure1)
        np.save(out / f"{family}_h_strain_2.npy", pure2)
        np.save(out / f"{family}_data_strain_1.npy", noisy1)
        np.save(out / f"{family}_data_strain_2.npy", noisy2)
        np.save(out / f"{family}_optimal_SNR_network_1.npy", np.asarray([x["target_snr_image1"] for x in rows if x["family"] == family], dtype=np.float32))
        np.save(out / f"{family}_optimal_SNR_network_2.npy", np.asarray([x["target_snr_image2"] for x in rows if x["family"] == family], dtype=np.float32))
        del pure1, pure2, noisy1, noisy2
        gc.collect()

    source = bank / "Unlensed_data_0222"
    clean = np.load(source / "unlensed_h_strain.npy", mmap_mode="r")[:samples]
    source_meta = pd.read_parquet(source / "physical_unlensed_source_metadata.parquet").sort_values("sample_index").reset_index(drop=True).iloc[:samples]
    out = root / "Unlensed_data_0222"
    out.mkdir(parents=True, exist_ok=True)
    pure = np.empty((samples, 2, MODEL_SAMPLES), dtype=np.float32)
    noisy = np.empty_like(pure)
    for idx in range(samples):
        source_row = source_meta.iloc[idx]
        target = draw_unlensed_snr(empirical, rng)
        pure[idx], noisy[idx], audit = prepare_one(clean[idx], target)
        rows.append(
            {
                "family": "unlensed",
                "physical_lens_group": "unlensed_control",
                "sample_index": idx,
                "gwlmc_row": int(source_row.gwlmc_row),
                "gps_image1": float(source_row.gps),
                "gps_image2": np.nan,
                "delay_days": np.nan,
                "snr_ratio_prior": np.nan,
                "target_snr_image1": target,
                "target_snr_image2": np.nan,
                "ra_true": float(source_row.ra),
                "dec_true": float(source_row.dec),
                **{f"image1_{key}": value for key, value in audit.items()},
            }
        )
    np.save(out / "unlensed_h_strain.npy", pure)
    np.save(out / "unlensed_data_strain.npy", noisy)
    np.save(out / "unlensed_optimal_SNR_network.npy", np.asarray([x["target_snr_image1"] for x in rows if x["family"] == "unlensed"], dtype=np.float32))

    frame = pd.DataFrame(rows)
    base.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(base / "compact_injection_metadata.parquet", index=False)
    payload = {
        "protocol_version": "real_noise_injection_v5_physical_source_full24",
        "deployment": deployment,
        "seed": int(seed),
        "samples_per_family": int(samples),
        "prepared_shape": [2, MODEL_SAMPLES],
        "source_units": "dimensionless physical H1/L1 detector strain before preprocessing",
        "source_bank": str(bank),
        "source_bank_units_audit": units_audit,
        "family_slot_mapping": {
            "SIS": "GW-LMC smooth/non-subhalo compatibility slot",
            "PM": "GW-LMC subhalo-present compatibility slot",
        },
        "snr_definition": "PSD-weighted optimal H1-L1 network SNR",
        "snr_proposal": "Run-matched empirical primary-catalog network SNR with the same GW-LMC pair's image-SNR ratio; both images conditioned to 8 <= rho <= 60.",
        "timing": "Each pair retains its own GW-LMC delay and both times lie in actual H1-L1 joint public DATA exposure.",
        "preprocessing": "Off-source Welch PSD whitening; Tukey taper; physical 40-580 Hz Butterworth bandpass; anti-aliased 4096->2048 Hz resampling; complete 24 s crop; robust scale.",
        "noise": "Independent run-matched GWOSC H1/L1 off-source strain",
        "metadata_rows": int(len(frame)),
    }
    write_json(marker, payload)
    del pure, noisy
    gc.collect()
    return payload


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def train_family_v5(seed_dir: Path, family: str, seed: int, samples: int, epochs: int, batch_size: int) -> dict[str, Any]:
    target = formal.expected_dir(seed_dir, family, epochs)
    summary_path = target / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    bank = source_bank(seed_dir.parent / "shared")
    multiroot = seed_dir / "data" / "real_noise_injections" / f"multinoise_{family.lower()}_train_v{VARIANTS_PER_SOURCE}"
    if not (multiroot / "multinoise_summary.json").exists():
        run_checked(
            [
                sys.executable,
                str(REPO / "scripts" / "real_search" / "36_materialize_multinoise_v5.py"),
                "--seed-root", str(seed_dir),
                "--source-bank", str(bank),
                "--family", family,
                "--seed", str(seed),
                "--samples", str(samples),
                "--variants-per-source", str(VARIANTS_PER_SOURCE),
            ],
            seed_dir / "logs" / f"materialize_multinoise_{family.lower()}_v5.log",
        )
    selected_dir = formal.selected_dir(seed_dir, family)
    selected_summary = selected_dir / "multinoise_curriculum_summary.json"
    if not selected_summary.exists():
        run_checked(
            [
                sys.executable,
                str(REPO / "scripts" / "experiments" / "30_multinoise_curriculum_pilot.py"),
                "--seed-root", str(seed_dir),
                "--family", family,
                "--seed", str(seed),
                "--samples", str(samples),
                "--variants-per-source", str(VARIANTS_PER_SOURCE),
                "--pretrain-epochs", str(PRETRAIN_EPOCHS),
                "--adapt-epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--architecture", ARCHITECTURE,
                "--adaptation", ADAPTATION,
            ],
            seed_dir / "logs" / f"train_{family.lower()}_v5.log",
        )
    selected = json.loads(selected_summary.read_text(encoding="utf-8"))
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = REPO / checkpoint
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, target / "model.pt")
    payload = {
        **selected,
        "protocol": "v5_physical_source",
        "architecture_predeclared_before_formal_seed_results": True,
        "architecture_rationale": "Full 24-s spectral context plus a fixed trigger-aligned 1-s branch; no held-out test or real-catalog pair was used for architecture selection.",
        "source_units": "physical H1/L1 detector strain before the single frozen preprocessing pipeline",
        "formal_checkpoint": str(target / "model.pt"),
    }
    write_json(summary_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--deployments", nargs="+", choices=("GWTC3", "GWTC4"), default=["GWTC3", "GWTC4"])
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--adapt-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal waveform training")

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_config.json",
        {
            "protocol": "real_noise_injection_v5_physical_source",
            "status": "running",
            "formal_seeds": args.seeds,
            "deployments": args.deployments,
            "samples_per_compatibility_family": args.samples_per_family,
            "family_slot_mapping": {
                "SIS": "GW-LMC smooth/non-subhalo; compatibility name only",
                "PM": "GW-LMC subhalo-present; compatibility name only",
            },
            "training_variants_per_source": VARIANTS_PER_SOURCE,
            "clean_pretrain_epochs": PRETRAIN_EPOCHS,
            "real_noise_adapt_epochs": args.adapt_epochs,
            "architecture": "full 24-s log-STFT GroupNorm CNN plus fixed 1-s trigger-aligned log-STFT branch",
            "channels": ["calibrated waveform log evidence", "run-conditioned time-delay log likelihood ratio", "HEALPix common-source sky log Bayes factor"],
            "snr_amplitude_in_final_score": False,
            "real_catalog_or_pe_used_for_tuning": False,
            "strict_primary_catalog": "complete H1-L1, non-OOD BBH pairs",
            "pe_policy": "independent post-ranking physical-consistency audit; failure cannot trigger retuning on real pairs",
        },
    )

    v3.materialize_compact_dataset = materialize_dataset_v5
    v3.train_family = train_family_v5
    v3.load_checkpoint_model = formal.load_selected_model
    v3.embed_real_events_v3 = formal.embed_real_events_selected

    for deployment in args.deployments:
        shared = args.out_root / deployment.lower() / "shared"
        schedule = v3.build_live_schedule(deployment, v3.SOURCES[deployment], shared)
        ensure_source_bank(deployment, shared, args.samples_per_family, 20260731 + (0 if deployment == "GWTC3" else 1000))
        v3.build_noise_bank(v3.SOURCES[deployment], shared, 20260721 + (0 if deployment == "GWTC3" else 1000))
        v3.build_sky_template_library(v3.SOURCES[deployment], shared)
        delays, _, _ = v3.liao_delay_ratio_samples()
        v3.build_time_calibration(shared, schedule, delays, 20260722 + (0 if deployment == "GWTC3" else 1000))
        complete_seeds: list[int] = []
        for seed in args.seeds:
            seed_dir = args.out_root / deployment.lower() / f"seed_{seed}"
            v3.prepare_seed_layout(seed_dir, v3.SOURCES[deployment])
            v3.run_seed(deployment, seed, args.out_root, args.samples_per_family, args.adapt_epochs, args.batch_size, args.bootstrap_draws)
            formal.choose_primary_and_export(seed_dir)
            complete_seeds.append(seed)
            v3.aggregate_deployment(args.out_root, deployment, complete_seeds)
            formal.aggregate_primary(args.out_root, deployment, complete_seeds)
    config = json.loads((args.out_root / "run_config.json").read_text(encoding="utf-8"))
    config["status"] = "complete"
    write_json(args.out_root / "run_config.json", config)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
