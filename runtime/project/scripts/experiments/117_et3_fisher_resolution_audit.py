#!/usr/bin/env python3
"""Numerical-resolution audit for ET sky-v8 GWFAST Fisher areas."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/unified_sky_v8_20260724"
FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2


def load_fisher_module():
    path = REPO / "scripts/experiments/113_et3_sky_v8_gwfast_fisher.py"
    specification = importlib.util.spec_from_file_location("et3_fisher_v8", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def representative_sample(module, per_family: int, seed: int) -> pd.DataFrame:
    frame = module.source_columns(module.load_union(), module.load_source_tables())
    rng = np.random.default_rng(seed)
    selected = []
    for family, part in frame.groupby("family", sort=False):
        # Sample evenly over SNR quantiles so convergence is not assessed only
        # at the modal SNR.
        order = part.sort_values("snr").reset_index(drop=True)
        positions = np.linspace(0, len(order) - 1, per_family)
        positions += rng.uniform(-0.35, 0.35, len(positions))
        positions = np.clip(np.rint(positions), 0, len(order) - 1).astype(int)
        sample = order.iloc[np.unique(positions)].copy()
        sample["audit_family"] = family
        selected.append(sample)
    return pd.concat(selected, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--low-resolution", type=int, default=100)
    parser.add_argument("--high-resolution", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    module = load_fisher_module()
    events = representative_sample(module, args.per_family, args.seed)
    detector, waveform = module.build_signal()
    low = module.run_chunk(
        detector,
        waveform,
        events.copy(),
        args.low_resolution,
    )
    high = module.run_chunk(
        detector,
        waveform,
        events.copy(),
        args.high_resolution,
    )
    columns = [
        "event_key",
        "family",
        "snr",
        "fisher_local_area90_deg2_raw",
        "fisher_local_area90_deg2_capped",
        "fisher_finite",
    ]
    comparison = low[columns].merge(
        high[columns],
        on=["event_key", "family", "snr"],
        suffixes=("_low", "_high"),
        validate="one_to_one",
    )
    valid = comparison["fisher_finite_low"] & comparison["fisher_finite_high"]
    low_area = comparison.loc[
        valid, "fisher_local_area90_deg2_capped_low"
    ].to_numpy(dtype=np.float64)
    high_area = comparison.loc[
        valid, "fisher_local_area90_deg2_capped_high"
    ].to_numpy(dtype=np.float64)
    log_ratio = np.log10(np.maximum(low_area, 1e-12) / np.maximum(high_area, 1e-12))
    comparison["log10_area_ratio_low_over_high"] = np.nan
    comparison.loc[valid, "log10_area_ratio_low_over_high"] = log_ratio
    low_full = low_area >= (FULL_SKY_DEG2 - 1e-6)
    high_full = high_area >= (FULL_SKY_DEG2 - 1e-6)
    correlation = (
        float(spearmanr(np.log10(low_area), np.log10(high_area)).statistic)
        if len(low_area) > 2
        else np.nan
    )
    summary = {
        "status": "complete",
        "sample_definition": (
            "Fixed SNR-quantile sample with equal requested counts from SIS, "
            "PM, and unlensed event families."
        ),
        "n_events": int(len(comparison)),
        "n_finite_in_both": int(valid.sum()),
        "low_resolution": int(args.low_resolution),
        "high_resolution": int(args.high_resolution),
        "spearman_log_area": correlation,
        "median_abs_log10_area_ratio": float(np.median(np.abs(log_ratio))),
        "q90_abs_log10_area_ratio": float(np.quantile(np.abs(log_ratio), 0.9)),
        "full_sky_classification_agreement": float(np.mean(low_full == high_full)),
        "fraction_within_factor_2": float(np.mean(np.abs(log_ratio) <= np.log10(2))),
        "fraction_within_factor_1p25": float(
            np.mean(np.abs(log_ratio) <= np.log10(1.25))
        ),
        "acceptance_rule": {
            "spearman_log_area_min": 0.98,
            "median_abs_log10_ratio_max": float(np.log10(1.25)),
            "full_sky_classification_agreement_min": 0.95,
        },
    }
    summary["accepted"] = bool(
        summary["spearman_log_area"] >= 0.98
        and summary["median_abs_log10_area_ratio"] <= np.log10(1.25)
        and summary["full_sky_classification_agreement"] >= 0.95
    )
    output = args.output_root / "shared"
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output / "et3_fisher_frequency_resolution_audit.csv",
        index=False,
    )
    (output / "et3_fisher_frequency_resolution_audit.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
