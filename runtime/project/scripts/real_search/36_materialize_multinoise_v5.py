#!/usr/bin/env python3
"""Materialize training-only real-noise variants from physical H1/L1 strain."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import split_indices  # noqa: E402
from scripts.real_search.physical_common import MODEL_SAMPLES, write_json  # noqa: E402
from scripts.real_search.physical_source_v5_common import (  # noqa: E402
    audit_physical_source_bank,
    draw_detected_pair_snrs,
    empirical_detected_snrs,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_mixed_only(v3, clean, noise, psd_frequency, psd, target_snr):
    """Inject and preprocess one training variant without an unused clean pass."""
    signal = np.asarray(clean, dtype=np.float32)
    if signal.shape != (2, v3.RAW_MODEL_SAMPLES) or np.mean(np.isfinite(signal)) < 0.999:
        raise ValueError(f"Invalid physical H1/L1 source strain: {signal.shape}")
    scaled, factor, recovered = v3.scale_to_network_snr(signal, target_snr, psd_frequency, psd)
    padded = v3.embed_signal_in_padded_window(scaled)
    prepared = v3.preprocess_24s(np.asarray(noise, dtype=np.float32) + padded, psd_frequency, psd)
    return prepared, {
        "target_network_snr": float(target_snr),
        "recovered_optimal_network_snr": float(recovered),
        "physical_strain_scale_factor": float(factor),
        "prepared_mixed_std": float(np.std(prepared)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--family", choices=("SIS", "PM"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--variants-per-source", type=int, default=8)
    args = parser.parse_args()

    v3 = load_module("v3_for_v5_multinoise", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
    source_audit = audit_physical_source_bank(args.source_bank)
    if not source_audit["passed"]:
        raise RuntimeError(f"Physical source bank failed units audit: {source_audit}")

    cfg = v3.training_config(args.seed_root, args.family, args.seed, args.samples, 1, 8)
    parts = split_indices(args.samples, args.samples, cfg)
    train_sources = np.asarray(parts["lensed"]["train"], dtype=np.int64)
    output = args.seed_root / "data" / "real_noise_injections" / f"multinoise_{args.family.lower()}_train_v{args.variants_per_source}"
    marker = output / "multinoise_summary.json"
    if marker.exists():
        print(marker.read_text(encoding="utf-8"), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)

    shared = args.seed_root.parent / "shared"
    references = np.load(shared / "noise_reference_bank.npy", mmap_mode="r")
    psds = np.load(shared / "noise_psd_bank.npy", mmap_mode="r")
    psd_frequency = np.load(shared / "noise_psd_frequency.npy")
    source = args.source_bank / f"{args.family}_data_0222"
    clean_a = np.load(source / f"{args.family}_h_strain_1.npy", mmap_mode="r")
    clean_b = np.load(source / f"{args.family}_h_strain_2.npy", mmap_mode="r")
    pair_meta = pd.read_parquet(source / "physical_source_pair_metadata.parquet").sort_values("pair_id").reset_index(drop=True)
    empirical = empirical_detected_snrs(args.seed_root / "data" / "event_manifest.csv")

    rows = len(train_sources) * args.variants_per_source
    noisy_a = np.lib.format.open_memmap(output / "noisy_image_a.npy", mode="w+", dtype=np.float32, shape=(rows, 2, MODEL_SAMPLES))
    noisy_b = np.lib.format.open_memmap(output / "noisy_image_b.npy", mode="w+", dtype=np.float32, shape=(rows, 2, MODEL_SAMPLES))
    source_ids = np.empty(rows, dtype=np.int32)
    variants = np.empty(rows, dtype=np.int16)
    metadata = []
    rng = np.random.default_rng(args.seed + 3600 + (0 if args.family == "SIS" else 1000))
    started = time.perf_counter()
    row_index = 0
    for number, source_index in enumerate(train_sources, start=1):
        source_row = pair_meta.iloc[int(source_index)]
        for variant in range(args.variants_per_source):
            snr_a, snr_b = draw_detected_pair_snrs(source_row, empirical, rng)
            noise_a, psd_a, bank_a, offset_a = v3.draw_noise_window(references, psds, rng)
            noise_b, psd_b, bank_b, offset_b = v3.draw_noise_window(references, psds, rng)
            noisy_a[row_index], audit_a = prepare_mixed_only(v3, clean_a[source_index], noise_a, psd_frequency, psd_a, snr_a)
            noisy_b[row_index], audit_b = prepare_mixed_only(v3, clean_b[source_index], noise_b, psd_frequency, psd_b, snr_b)
            source_ids[row_index] = int(source_index)
            variants[row_index] = int(variant)
            metadata.append(
                {
                    "row_index": row_index,
                    "source_index": int(source_index),
                    "gwlmc_row": int(source_row.gwlmc_row),
                    "physical_lens_group": str(source_row.physical_lens_group),
                    "variant": int(variant),
                    "target_snr_a": snr_a,
                    "target_snr_b": snr_b,
                    "gwlmc_proposal_snr_ratio": float(source_row.proposal_snr_ratio),
                    "noise_bank_a": int(bank_a),
                    "noise_bank_b": int(bank_b),
                    "noise_offset_a": int(offset_a),
                    "noise_offset_b": int(offset_b),
                    "recovered_snr_a": float(audit_a["recovered_optimal_network_snr"]),
                    "recovered_snr_b": float(audit_b["recovered_optimal_network_snr"]),
                }
            )
            row_index += 1
        if number % 25 == 0 or number == len(train_sources):
            print(json.dumps({"sources_complete": number, "sources_total": len(train_sources), "rows_complete": row_index}), flush=True)
    noisy_a.flush()
    noisy_b.flush()
    np.save(output / "source_index.npy", source_ids)
    np.save(output / "variant.npy", variants)
    pd.DataFrame(metadata).to_parquet(output / "multinoise_metadata.parquet", index=False)
    payload = {
        "protocol": "v5_physical_source_multinoise",
        "family_slot": args.family,
        "physical_lens_group": "GW-LMC smooth/non-subhalo" if args.family == "SIS" else "GW-LMC subhalo-present",
        "seed": args.seed,
        "source_split": "training systems only",
        "n_unique_training_sources": int(len(train_sources)),
        "variants_per_source": int(args.variants_per_source),
        "n_training_pairs": int(rows),
        "validation_test_modified": False,
        "source_units": "dimensionless physical H1/L1 detector strain",
        "snr_definition": "PSD-weighted optimal H1-L1 network SNR",
        "snr_proposal": "Fainter image bootstrapped from run-matched real primary-catalog network SNR; brighter image preserves the same GW-LMC pair ratio; both conditioned to 8 <= rho <= 60.",
        "noise_source": "independent public GWOSC run-matched H1/L1 off-source segments",
        "preprocessing": "same physical_common.preprocess_24s used by train/validation/test/real deployment",
        "source_bank_audit": source_audit,
        "elapsed_s": float(time.perf_counter() - started),
    }
    write_json(marker, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
