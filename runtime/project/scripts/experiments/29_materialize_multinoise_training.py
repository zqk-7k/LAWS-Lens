#!/usr/bin/env python3
"""Create independent real-noise realizations for training sources only.

Validation and test arrays are not regenerated or inspected.  Source-system
splits are inherited from the physical deployment configuration, and every
noise realization of a source remains in the training partition.
"""

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

from matchgw.data import split_indices
from scripts.real_search.physical_common import MODEL_SAMPLES, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--family", choices=("SIS", "PM"), required=True)
    parser.add_argument("--seed", type=int, default=202607211)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--variants-per-source", type=int, default=8)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("v3", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
    v3 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(v3)
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
    source = v3.CLEAN_SOURCE / f"{args.family}_data_0222"
    clean_a = np.load(source / f"{args.family}_h_strain_1.npy", mmap_mode="r")
    clean_b = np.load(source / f"{args.family}_h_strain_2.npy", mmap_mode="r")
    _, ratios, ratio_label = v3.liao_delay_ratio_samples()

    rows = len(train_sources) * args.variants_per_source
    noisy_a = np.lib.format.open_memmap(output / "noisy_image_a.npy", mode="w+", dtype=np.float32, shape=(rows, 2, MODEL_SAMPLES))
    noisy_b = np.lib.format.open_memmap(output / "noisy_image_b.npy", mode="w+", dtype=np.float32, shape=(rows, 2, MODEL_SAMPLES))
    source_ids = np.empty(rows, dtype=np.int32)
    variants = np.empty(rows, dtype=np.int16)
    metadata = []
    rng = np.random.default_rng(args.seed + 2900 + (0 if args.family == "SIS" else 1000))
    started = time.perf_counter()
    row_index = 0
    for number, source_index in enumerate(train_sources, start=1):
        for variant in range(args.variants_per_source):
            ratio = float(ratios[int(rng.integers(0, len(ratios)))])
            snr_a, snr_b = v3.draw_snr_pair(ratio, rng)
            noise_a, psd_a, bank_a, offset_a = v3.draw_noise_window(references, psds, rng)
            noise_b, psd_b, bank_b, offset_b = v3.draw_noise_window(references, psds, rng)
            _, noisy_a[row_index], audit_a = v3._preprocess_injection(clean_a[source_index], noise_a, psd_frequency, psd_a, snr_a)
            _, noisy_b[row_index], audit_b = v3._preprocess_injection(clean_b[source_index], noise_b, psd_frequency, psd_b, snr_b)
            source_ids[row_index] = int(source_index)
            variants[row_index] = int(variant)
            metadata.append(
                {
                    "row_index": row_index,
                    "source_index": int(source_index),
                    "variant": int(variant),
                    "target_snr_a": snr_a,
                    "target_snr_b": snr_b,
                    "snr_ratio_prior": ratio,
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
    noisy_a.flush(); noisy_b.flush()
    np.save(output / "source_index.npy", source_ids)
    np.save(output / "variant.npy", variants)
    pd.DataFrame(metadata).to_parquet(output / "multinoise_metadata.parquet", index=False)
    payload = {
        "family": args.family,
        "seed": args.seed,
        "source_split": "training only",
        "n_unique_training_sources": int(len(train_sources)),
        "variants_per_source": int(args.variants_per_source),
        "n_training_pairs": int(rows),
        "validation_test_modified": False,
        "snr_definition": "PSD-weighted optimal H1-L1 network SNR",
        "snr_ratio_source": ratio_label,
        "noise_source": "independent public GWOSC run-matched H1/L1 off-source segments",
        "preprocessing": "same physical_common.preprocess_24s used by validation/test/real deployment",
        "elapsed_s": float(time.perf_counter() - started),
    }
    write_json(marker, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
