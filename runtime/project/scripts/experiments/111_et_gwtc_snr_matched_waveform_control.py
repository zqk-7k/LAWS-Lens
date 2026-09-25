#!/usr/bin/env python3
"""Matched-SNR ET control against the frozen O3/O4a v7 waveform results.

This is a waveform-domain control, not a replacement for any main catalog.
For each O3/O4a formal seed it reuses that seed's exact target pair SNRs,
rescales independent ET clean signals to those SNRs, and adds independent ET
noise realizations.  Catalog size and train/validation/test system counts then
match the real-noise v7 runs (600 systems/family, 450 test events).

The source populations and detector responses still differ, so equality is not
expected.  The control isolates the dominant SNR-distribution difference while
leaving a clearly documented residual source/noise-domain difference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import signal


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    across_seed_summary,
    bootstrap_system_ci,
    metric_rows,
    split_integrity_audit,
    write_json,
)


et = importlib.import_module("scripts.experiments.110_et3_v7_aligned_pipeline")

RAW_ROOT = et.RAW_ROOT
V7_ROOT = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
DEFAULT_ET_ROOT = REPO / "results/et3_v7_aligned_20260723"
DEFAULT_OUT = REPO / "results/et_gwtc_snr_matched_control_20260723"
SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENTS = ("gwtc3", "gwtc4")
RAW_FS = 4096
MODEL_FS = 2048
RAW_SAMPLES = 24 * RAW_FS
CONTEXT_SAMPLES = 6 * RAW_FS
GEOCENTRE_FROM_END = 1500
TARGET_START = RAW_SAMPLES - GEOCENTRE_FROM_END - int(1.75 * RAW_FS)
TARGET_END = RAW_SAMPLES - GEOCENTRE_FROM_END + int(0.25 * RAW_FS)
CONTEXT_START = RAW_SAMPLES - CONTEXT_SAMPLES
TARGET_SLICE = slice(
    (TARGET_START - CONTEXT_START) // 2,
    (TARGET_END - CONTEXT_START) // 2,
)


def preprocess_context(values: np.ndarray, method: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] != CONTEXT_SAMPLES:
        raise ValueError(x.shape)
    taper = signal.windows.tukey(CONTEXT_SAMPLES, alpha=0.02)
    sos = signal.butter(6, [40.0, 580.0], btype="bandpass", fs=RAW_FS, output="sos")
    filtered = signal.sosfiltfilt(sos, x * taper, axis=-1)
    if method == "polyphase":
        down = signal.resample_poly(
            filtered,
            up=1,
            down=2,
            axis=-1,
            window=("kaiser", 8.6),
        )
    elif method == "stride2":
        down = filtered[..., ::2]
    else:
        raise ValueError(method)
    cropped = down[..., TARGET_SLICE]
    if cropped.ndim == 2:
        return et.prepare(cropped, None, False)
    if cropped.ndim != 3:
        raise ValueError(f"Expected [batch,channel,time], got {cropped.shape}")
    # ``et.prepare`` intentionally accepts one event at a time so its
    # peak-polarity and robust channel scaling cannot mix batch members.
    return np.stack(
        [et.prepare(event, None, False) for event in cropped],
        axis=0,
    )


def v7_metadata(deployment: str, seed: int) -> pd.DataFrame:
    path = (
        V7_ROOT
        / deployment
        / f"seed_{seed}"
        / "data/real_noise_injections/compact_injection_metadata.parquet"
    )
    return pd.read_parquet(path)


def output_array(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def draw_noise_context(
    data: np.ndarray,
    clean: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(data[indices, :, CONTEXT_START:], dtype=np.float64)
        - np.asarray(clean[indices, :, CONTEXT_START:], dtype=np.float64)
    )


def materialize_control(
    deployment: str,
    seed: int,
    preprocessing: str,
    out_root: Path,
    samples: int,
) -> Path:
    root = out_root / "data" / deployment / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "control_data_summary.json"
    if marker.exists():
        return root
    metadata = v7_metadata(deployment, seed)
    rng = np.random.default_rng(seed + 55000)
    raw_u = RAW_ROOT / "Unlensed_data_0222"
    u_data = np.load(raw_u / "unlensed_data_strain.npy", mmap_mode="r")
    u_clean = np.load(raw_u / "unlensed_h_strain.npy", mmap_mode="r")
    u_snr = np.load(raw_u / "unlensed_optimal_SNR_network.npy", mmap_mode="r")
    rows = []

    for family in et.FAMILIES:
        raw = RAW_ROOT / f"{family}_data_0222"
        source_indices = rng.choice(10_000, size=samples, replace=False)
        target = (
            metadata[metadata.family == family]
            .sort_values("sample_index")
            .iloc[:samples]
            .reset_index(drop=True)
        )
        source = pd.read_csv(raw / "source_samples.csv").iloc[source_indices].reset_index(drop=True)
        timing = pd.read_csv(raw / f"{family}_trigger_time_features.csv").iloc[source_indices].reset_index(drop=True)
        source.to_csv(root / f"{family}_source_samples.csv", index=False)
        timing.to_csv(root / f"{family}_trigger_time_features.csv", index=False)
        np.save(root / f"{family}_source_indices.npy", source_indices.astype(np.int32))
        for image in (1, 2):
            clean = np.load(raw / f"{family}_h_strain_{image}.npy", mmap_mode="r")
            current_snr = np.load(raw / f"{family}_optimal_SNR_network_{image}.npy", mmap_mode="r")
            target_snr = target[f"target_snr_image{image}"].to_numpy(np.float64)
            pure_out = output_array(
                root / f"{family}_h_strain_{image}.npy",
                (samples, 3, et.INPUT_SAMPLES),
            )
            noisy_out = output_array(
                root / f"{family}_data_strain_{image}.npy",
                (samples, 3, et.INPUT_SAMPLES),
            )
            np.save(root / f"{family}_optimal_SNR_network_{image}.npy", target_snr.astype(np.float32))
            for start in range(0, samples, 24):
                stop = min(samples, start + 24)
                idx = source_indices[start:stop]
                scale = target_snr[start:stop] / np.asarray(current_snr[idx], dtype=np.float64)
                source_signal = np.asarray(
                    clean[idx, :, CONTEXT_START:],
                    dtype=np.float64,
                ) * scale[:, None, None]
                noise_idx = rng.integers(0, len(u_data), size=stop - start)
                noise = draw_noise_context(u_data, u_clean, noise_idx)
                pure_out[start:stop] = preprocess_context(source_signal, preprocessing)
                noisy_out[start:stop] = preprocess_context(source_signal + noise, preprocessing)
            pure_out.flush()
            noisy_out.flush()
        rows.append(
            {
                "deployment": deployment,
                "seed": seed,
                "family": family,
                "systems": samples,
                "target_snr_image1_median": float(target.target_snr_image1.median()),
                "target_snr_image2_median": float(target.target_snr_image2.median()),
            }
        )

    source_indices = rng.choice(10_000, size=samples, replace=False)
    target = (
        metadata[metadata.family == "unlensed"]
        .sort_values("sample_index")
        .iloc[:samples]
        .reset_index(drop=True)
    )
    source = pd.read_csv(raw_u / "source_samples.csv").iloc[source_indices].reset_index(drop=True)
    timing = pd.read_csv(raw_u / "unlensed_trigger_time_features.csv").iloc[source_indices].reset_index(drop=True)
    source.to_csv(root / "unlensed_source_samples.csv", index=False)
    timing.to_csv(root / "unlensed_trigger_time_features.csv", index=False)
    np.save(root / "unlensed_source_indices.npy", source_indices.astype(np.int32))
    target_snr = target["target_snr_image1"].to_numpy(np.float64)
    pure_out = output_array(root / "unlensed_h_strain.npy", (samples, 3, et.INPUT_SAMPLES))
    noisy_out = output_array(root / "unlensed_data_strain.npy", (samples, 3, et.INPUT_SAMPLES))
    np.save(root / "unlensed_optimal_SNR_network.npy", target_snr.astype(np.float32))
    for start in range(0, samples, 24):
        stop = min(samples, start + 24)
        idx = source_indices[start:stop]
        scale = target_snr[start:stop] / np.asarray(u_snr[idx], dtype=np.float64)
        source_signal = np.asarray(
            u_clean[idx, :, CONTEXT_START:],
            dtype=np.float64,
        ) * scale[:, None, None]
        noise_idx = rng.integers(0, len(u_data), size=stop - start)
        noise = draw_noise_context(u_data, u_clean, noise_idx)
        pure_out[start:stop] = preprocess_context(source_signal, preprocessing)
        noisy_out[start:stop] = preprocess_context(source_signal + noise, preprocessing)
    pure_out.flush()
    noisy_out.flush()
    rows.append(
        {
            "deployment": deployment,
            "seed": seed,
            "family": "unlensed",
            "systems": samples,
            "target_snr_image1_median": float(np.median(target_snr)),
            "target_snr_image2_median": np.nan,
        }
    )
    payload = {
        "status": "complete",
        "deployment_snr_source": deployment,
        "seed": int(seed),
        "systems_per_family": int(samples),
        "preprocessing": preprocessing,
        "target_snrs": "exact target SNR rows from the corresponding frozen GWTC v7 seed",
        "et_signal_scaling": "clean ET detector strain multiplied by target/current saved Bilby optimal network SNR",
        "noise": "independent ET whitened noise = unlensed_data_strain - unlensed_h_strain",
        "residual_difference": "ET legacy source population and ET detector response differ from the GW-LMC H1/L1 v7 source bank",
        "rows": rows,
    }
    write_json(marker, payload)
    return root


def load_control_bank(root: Path) -> Any:
    families = {}
    for family in et.FAMILIES:
        source = pd.read_csv(root / f"{family}_source_samples.csv")
        families[family] = et.FamilyBank(
            noisy1=np.load(root / f"{family}_data_strain_1.npy", mmap_mode="r"),
            noisy2=np.load(root / f"{family}_data_strain_2.npy", mmap_mode="r"),
            pure1=np.load(root / f"{family}_h_strain_1.npy", mmap_mode="r"),
            pure2=np.load(root / f"{family}_h_strain_2.npy", mmap_mode="r"),
            snr1=np.load(root / f"{family}_optimal_SNR_network_1.npy", mmap_mode="r"),
            snr2=np.load(root / f"{family}_optimal_SNR_network_2.npy", mmap_mode="r"),
            source=source,
            timing=pd.read_csv(root / f"{family}_trigger_time_features.csv"),
            targets=et.detector_frame_targets(source),
        )
    return et.ETBank(
        families=families,
        unlensed_noisy=np.load(root / "unlensed_data_strain.npy", mmap_mode="r"),
        unlensed_pure=np.load(root / "unlensed_h_strain.npy", mmap_mode="r"),
        unlensed_snr=np.load(root / "unlensed_optimal_SNR_network.npy", mmap_mode="r"),
        unlensed_source=pd.read_csv(root / "unlensed_source_samples.csv"),
        unlensed_timing=pd.read_csv(root / "unlensed_trigger_time_features.csv"),
    )


def control_splits(bank: Any, seed: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        "SIS": et.split_array(len(bank.families["SIS"].noisy1), seed + 101, (0.70, 0.15)),
        "PM": et.split_array(len(bank.families["PM"].noisy1), seed + 202, (0.70, 0.15)),
        "unlensed": et.split_array(len(bank.unlensed_noisy), seed + 909, (0.70, 0.15)),
    }


def run_control_seed(
    deployment: str,
    seed: int,
    preprocessing: str,
    out_root: Path,
    samples: int,
    pretrain_epochs: int,
    adapt_epochs: int,
    batch_size: int,
) -> None:
    seed_dir = out_root / deployment / f"seed_{seed}"
    complete = seed_dir / "seed_summary.json"
    if complete.exists():
        return
    started = time.perf_counter()
    data_root = materialize_control(deployment, seed, preprocessing, out_root, samples)
    bank = load_control_bank(data_root)
    splits = control_splits(bank, seed)
    audit = split_integrity_audit(splits)
    if not audit["all_disjoint"]:
        raise RuntimeError("Control split leakage")
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "split_integrity_audit.json", audit)
    checkpoint, training = et.train_seed(
        bank,
        splits,
        seed,
        seed_dir / "encoder",
        pretrain_epochs,
        adapt_epochs,
        batch_size,
    )
    model, payload = et.load_checkpoint(checkpoint)
    validation = et.CatalogDataset(bank, splits, "val")
    test = et.CatalogDataset(bank, splits, "test")
    val_z, val_p = et.embed_catalog(model, validation, max(96, batch_size * 2))
    test_z, test_p = et.embed_catalog(model, test, max(96, batch_size * 2))
    mean = np.asarray(payload["target_mean"], np.float32)
    std = np.asarray(payload["target_std"], np.float32)
    val_p = val_p * std[None, :] + mean[None, :]
    test_p = test_p * std[None, :] + mean[None, :]
    _, config, grid = et.calibrate_waveform(
        val_z,
        val_p,
        validation,
        seed + 3000,
        false_count=100_000,
    )
    score = et.apply_waveform_calibration(test_z, test_p, config)
    metrics, ranks = et.retrieval_from_matrix(score, test.partner, test.meta)
    ranks.insert(0, "method", "waveform_only")
    ranks.insert(0, "seed", int(seed))
    ranks.insert(0, "deployment", f"ET_{deployment.upper()}_SNR_MATCHED")
    ranks.to_parquet(seed_dir / "query_ranks.parquet", index=False)
    grid.to_csv(seed_dir / "waveform_intrinsic_weight_grid.csv", index=False)
    write_json(seed_dir / "waveform_calibration.json", config)
    bootstrap = bootstrap_system_ci(ranks, draws=10_000, seed=seed + 5000)
    bootstrap.to_csv(seed_dir / "bootstrap_95ci.csv", index=False)
    write_json(
        complete,
        {
            "status": "complete",
            "deployment_snr_source": deployment,
            "seed": int(seed),
            "preprocessing": preprocessing,
            "test_events": len(test),
            "test_queries": int(np.sum(test.partner >= 0)),
            "metrics": metrics,
            "training": training,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    del model, bank
    torch.cuda.empty_cache()


def aggregate(out_root: Path) -> None:
    frames = []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            path = out_root / deployment / f"seed_{seed}/query_ranks.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    query = pd.concat(frames, ignore_index=True)
    query.to_parquet(out_root / "et_snr_matched_query_ranks_all_seeds.parquet", index=False)
    per_seed = metric_rows(query)
    per_seed.to_csv(out_root / "et_snr_matched_retrieval_per_seed.csv", index=False)
    across_seed_summary(per_seed).to_csv(
        out_root / "et_snr_matched_retrieval_summary.csv",
        index=False,
    )
    comparison = [per_seed]
    for deployment in DEPLOYMENTS:
        source = pd.read_csv(
            V7_ROOT / deployment / "heldout_test_retrieval_metrics_per_seed_v7.csv"
        )
        source = source[
            (source.method == "waveform_only") & (source.subset.isin(["overall", "sis", "pm"]))
        ].copy()
        source["subset"] = source["subset"].str.upper().replace({"OVERALL": "overall"})
        source["deployment"] = deployment.upper()
        comparison.append(source)
    all_rows = pd.concat(comparison, ignore_index=True, sort=False)
    all_rows.to_csv(out_root / "et_o3_o4_snr_matched_waveform_comparison_per_seed.csv", index=False)
    metrics = ["r_at_1", "r_at_10", "median_rank"]
    rows = []
    for key, frame in all_rows.groupby(["deployment", "subset"], dropna=False):
        for metric in metrics:
            values = frame[metric].dropna().to_numpy(np.float64)
            rows.append(
                {
                    "deployment": key[0],
                    "subset": key[1],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low_normal": float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else float(values.mean()),
                    "ci95_high_normal": float(values.mean() + 1.96 * values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else float(values.mean()),
                }
            )
    pd.DataFrame(rows).to_csv(
        out_root / "et_o3_o4_snr_matched_waveform_comparison_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--et-root", type=Path, default=DEFAULT_ET_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--pretrain-epochs", type=int, default=16)
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    frozen = json.loads(
        (args.et_root / "development/frozen_preprocessing_selection.json").read_text(
            encoding="utf-8"
        )
    )
    preprocessing = str(frozen["selected_preprocessing"])
    write_json(
        args.out_root / "run_config.json",
        {
            "status": "running",
            "preprocessing": preprocessing,
            "deployments": DEPLOYMENTS,
            "seeds": SEEDS,
            "samples_per_family": args.samples,
            "scope": "waveform-only matched-SNR control",
            "not_full_three_channel_comparison": True,
        },
    )
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            run_control_seed(
                deployment,
                seed,
                preprocessing,
                args.out_root,
                args.samples,
                args.pretrain_epochs,
                args.adapt_epochs,
                args.batch_size,
            )
            aggregate(args.out_root)
    config = json.loads((args.out_root / "run_config.json").read_text(encoding="utf-8"))
    config["status"] = "complete"
    write_json(args.out_root / "run_config.json", config)


if __name__ == "__main__":
    main()
