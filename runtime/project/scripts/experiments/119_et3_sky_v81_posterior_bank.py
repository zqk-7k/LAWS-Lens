#!/usr/bin/env python3
"""Render the shared ET v8.1 posterior bank once for all experiment seeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.et_sky_v81 import generate_et_posterior_maps
from scripts.real_search.unified_sky_v8 import write_json


DEFAULT_OUTPUT = REPO / "results/unified_sky_v81_20260725"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--event-chunk-size", type=int, default=256)
    parser.add_argument("--render-chunk-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shared = args.output_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    fisher_path = shared / "et3_gwfast_sky_v81_events.parquet"
    if not fisher_path.exists():
        raise FileNotFoundError(
            "Run 118_et3_sky_v81_gwfast_covariance.py first"
        )
    fisher = (
        pd.read_parquet(fisher_path)
        .sort_values("event_key")
        .reset_index(drop=True)
    )
    if fisher["event_key"].duplicated().any():
        raise RuntimeError("Fisher table contains duplicate event keys")

    map_path = shared / f"et3_posterior_bank_nside{args.nside}_v81.npy"
    index_path = shared / "et3_posterior_bank_index_v81.parquet"
    diagnostics_path = shared / "et3_posterior_bank_diagnostics_v81.parquet"
    audit_path = shared / "et3_posterior_bank_audit_v81.json"
    progress_path = shared / "et3_posterior_bank_progress_v81.json"
    chunk_root = shared / "et3_posterior_bank_diagnostic_chunks_v81"
    chunk_root.mkdir(parents=True, exist_ok=True)

    if (
        map_path.exists()
        and index_path.exists()
        and diagnostics_path.exists()
        and audit_path.exists()
        and not args.force
    ):
        print(f"Reusing complete posterior bank: {map_path}")
        return
    if args.force:
        for path in (
            map_path,
            index_path,
            diagnostics_path,
            audit_path,
            progress_path,
        ):
            path.unlink(missing_ok=True)
        for path in chunk_root.glob("chunk_*.parquet"):
            path.unlink()

    shape = (len(fisher), hp.nside2npix(args.nside))
    if map_path.exists():
        bank = np.load(map_path, mmap_mode="r+")
        if bank.shape != shape:
            raise RuntimeError(
                f"Existing bank shape {bank.shape} does not match {shape}"
            )
    else:
        bank = np.lib.format.open_memmap(
            map_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
    progress = 0
    if progress_path.exists():
        progress = int(json.loads(progress_path.read_text())["next_index"])

    started = time.time()
    for start in range(progress, len(fisher), args.event_chunk_size):
        stop = min(start + args.event_chunk_size, len(fisher))
        frame = fisher.iloc[start:stop][
            [
                "family",
                "source_index",
                "system_id",
                "tag",
                "image",
                "snr",
                "gps",
                "ra",
                "dec",
            ]
        ].copy()
        maps, diagnostics, audit = generate_et_posterior_maps(
            frame,
            fisher,
            seed=20260725,
            nside=args.nside,
            mode_count=8,
            chunk_size=args.render_chunk_size,
            device=args.device,
        )
        bank[start:stop] = maps
        bank.flush()
        diagnostics.insert(0, "bank_index", np.arange(start, stop))
        diagnostics.to_parquet(
            chunk_root / f"chunk_{start:06d}_{stop:06d}.parquet",
            index=False,
        )
        write_json(
            progress_path,
            {
                "next_index": stop,
                "total_events": len(fisher),
                "last_chunk_audit": audit,
            },
        )
        print(
            f"ET posterior bank {stop}/{len(fisher)} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )

    diagnostics = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(chunk_root.glob("chunk_*.parquet"))
        ],
        ignore_index=True,
    ).sort_values("bank_index")
    if len(diagnostics) != len(fisher):
        raise RuntimeError(
            f"Diagnostics contain {len(diagnostics)} rows, expected {len(fisher)}"
        )
    diagnostics.to_parquet(diagnostics_path, index=False)
    index = fisher[["event_key"]].copy()
    index.insert(0, "bank_index", np.arange(len(index), dtype=np.int32))
    index.to_parquet(index_path, index=False)

    sampled_indices = np.linspace(
        0,
        len(fisher) - 1,
        min(len(fisher), 1024),
        dtype=np.int64,
    )
    sampled_sums = np.asarray(bank[sampled_indices], dtype=np.float32).sum(axis=1)
    coverage_count = int(diagnostics["true_sky_inside_90"].sum())
    coverage_total = int(len(diagnostics))
    coverage_fraction = coverage_count / coverage_total
    z_value = 1.959963984540054
    denominator = 1.0 + z_value**2 / coverage_total
    center = (
        coverage_fraction + z_value**2 / (2.0 * coverage_total)
    ) / denominator
    half_width = (
        z_value
        * np.sqrt(
            coverage_fraction
            * (1.0 - coverage_fraction)
            / coverage_total
            + z_value**2 / (4.0 * coverage_total**2)
        )
        / denominator
    )
    audit = {
        "version": "unified_sky_v81",
        "status": "complete",
        "n_events": int(len(fisher)),
        "nside": int(args.nside),
        "n_pixels": int(shape[1]),
        "stored_dtype": "float32",
        "map_path": str(map_path),
        "event_index_path": str(index_path),
        "diagnostics_path": str(diagnostics_path),
        "sampled_map_sum_min": float(sampled_sums.min()),
        "sampled_map_sum_max": float(sampled_sums.max()),
        "true_sky_90_coverage": float(coverage_fraction),
        "true_sky_90_coverage_wilson_95ci": [
            float(center - half_width),
            float(center + half_width),
        ],
        "area90_quantiles_deg2": {
            str(quantile): float(
                diagnostics["area90_deg2"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "mode_effective_count_quantiles": {
            str(quantile): float(
                diagnostics["mode_effective_count"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "fraction_mode_effective_count_below_7_5": float(
            np.mean(diagnostics["mode_effective_count"] < 7.5)
        ),
        "fraction_mode_effective_count_below_4": float(
            np.mean(diagnostics["mode_effective_count"] < 4.0)
        ),
        "elapsed_seconds": time.time() - started,
        "large_bank_excluded_from_deliverables": True,
    }
    write_json(audit_path, audit)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
