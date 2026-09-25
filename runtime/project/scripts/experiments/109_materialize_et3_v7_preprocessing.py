#!/usr/bin/env python3
"""Materialize the two frozen ET-3 peak-window preprocessing variants.

The source ET arrays are already detector-PSD-whitened by Bilby at generation
time.  This script therefore applies only the remaining deployment-aligned
steps: a physical-Hz bandpass, 4096->2048 Hz downsampling, a fixed
[geocentre-1.75 s, geocentre+0.25 s] crop, and per-channel robust scaling.

Two outputs are produced from the same filtered samples:

* ``stride2``: legacy every-other-sample downsampling.
* ``polyphase``: explicit Kaiser-windowed anti-aliased resampling.

No source strain is copied or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import signal


RAW_FS = 4096
MODEL_FS = 2048
RAW_DURATION_S = 24
MODEL_WINDOW_S = 2.0
GEOCENTRE_SAMPLES_BEFORE_END = 1500
WINDOW_BEFORE_GPS_S = 1.75
WINDOW_AFTER_GPS_S = 0.25
CONTEXT_S = 6
BAND_LOW_HZ = 40.0
BAND_HIGH_HZ = 580.0

ARRAYS = (
    ("SIS_data_0222", "SIS_h_strain_1.npy"),
    ("SIS_data_0222", "SIS_h_strain_2.npy"),
    ("SIS_data_0222", "SIS_data_strain_1.npy"),
    ("SIS_data_0222", "SIS_data_strain_2.npy"),
    ("PM_data_0222", "PM_h_strain_1.npy"),
    ("PM_data_0222", "PM_h_strain_2.npy"),
    ("PM_data_0222", "PM_data_strain_1.npy"),
    ("PM_data_0222", "PM_data_strain_2.npy"),
    ("Unlensed_data_0222", "unlensed_h_strain.npy"),
    ("Unlensed_data_0222", "unlensed_data_strain.npy"),
)


def robust_scale_channels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(values - median), axis=-1, keepdims=True)
    std = np.std(values, axis=-1, keepdims=True)
    scale = np.where(mad > 1e-12, mad, std)
    scale = np.maximum(scale, np.maximum(np.abs(median) * 1e-6, 1e-30))
    return ((values - median) / scale).astype(np.float32)


def target_slice(raw_samples: int, context_start: int) -> slice:
    geocentre = raw_samples - GEOCENTRE_SAMPLES_BEFORE_END
    start = geocentre - int(round(WINDOW_BEFORE_GPS_S * RAW_FS))
    end = geocentre + int(round(WINDOW_AFTER_GPS_S * RAW_FS))
    if end - start != int(MODEL_WINDOW_S * RAW_FS):
        raise RuntimeError("The fixed raw target window is not exactly two seconds")
    relative_start = start - context_start
    relative_end = end - context_start
    if relative_start % 2 or relative_end % 2:
        raise RuntimeError("The target window is not aligned to the 2:1 sample grid")
    return slice(relative_start // 2, relative_end // 2)


def link_metadata(source_root: Path, output_root: Path) -> None:
    for group in ("SIS_data_0222", "PM_data_0222", "Unlensed_data_0222"):
        source_dir = source_root / group
        target_dir = output_root / group
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.iterdir():
            name = source.name
            if name.endswith(("_strain_1.npy", "_strain_2.npy", "_strain.npy")):
                continue
            if "time_array" in name:
                continue
            target = target_dir / name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(source.resolve())


def materialize_array(
    source: Path,
    destinations: dict[str, Path],
    chunk_size: int,
) -> dict[str, object]:
    raw = np.load(source, mmap_mode="r")
    if raw.ndim != 3 or raw.shape[1] != 3 or raw.shape[2] != RAW_FS * RAW_DURATION_S:
        raise ValueError(f"Unexpected ET array shape for {source}: {raw.shape}")
    outputs = {
        method: np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=np.float32,
            shape=(raw.shape[0], raw.shape[1], int(MODEL_FS * MODEL_WINDOW_S)),
        )
        for method, destination in destinations.items()
    }
    context_samples = CONTEXT_S * RAW_FS
    context_start = raw.shape[-1] - context_samples
    keep = target_slice(raw.shape[-1], context_start)
    taper = signal.windows.tukey(context_samples, alpha=0.02).reshape(1, 1, -1)
    sos = signal.butter(
        6,
        [BAND_LOW_HZ, BAND_HIGH_HZ],
        btype="bandpass",
        fs=RAW_FS,
        output="sos",
    )
    summaries: dict[str, list[float]] = {method: [] for method in outputs}
    for start in range(0, len(raw), chunk_size):
        stop = min(len(raw), start + chunk_size)
        block = np.asarray(raw[start:stop, :, context_start:], dtype=np.float64)
        if np.mean(np.isfinite(block)) < 0.999999:
            raise ValueError(f"Non-finite ET samples in {source}, rows {start}:{stop}")
        filtered = signal.sosfiltfilt(sos, block * taper, axis=-1)
        variants = {
            "stride2": filtered[..., ::2],
            "polyphase": signal.resample_poly(
                filtered,
                up=1,
                down=2,
                axis=-1,
                window=("kaiser", 8.6),
            ),
        }
        for method, output in outputs.items():
            prepared = robust_scale_channels(variants[method][..., keep])
            if prepared.shape[-1] != int(MODEL_FS * MODEL_WINDOW_S):
                raise RuntimeError(f"Unexpected {method} output shape {prepared.shape}")
            output[start:stop] = prepared
            summaries[method].extend(np.std(prepared, axis=-1).reshape(-1).tolist())
        if start == 0 or stop == len(raw) or (start // chunk_size) % 20 == 0:
            print(
                json.dumps(
                    {
                        "source": str(source),
                        "rows_complete": stop,
                        "rows_total": len(raw),
                    }
                ),
                flush=True,
            )
    for output in outputs.values():
        output.flush()
    return {
        "source": str(source),
        "source_shape": list(raw.shape),
        "source_dtype": str(raw.dtype),
        "output_shape": [len(raw), 3, int(MODEL_FS * MODEL_WINDOW_S)],
        "channel_std_median": {
            method: float(np.median(values)) for method, values in summaries.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/root/autodl-tmp/gw-catalog/data_generation/et3_v7_preprocessed_20260723"),
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()

    methods = ("stride2", "polyphase")
    for method in methods:
        link_metadata(args.source_root, args.out_root / method)
    audit = []
    for group, filename in ARRAYS:
        source = args.source_root / group / filename
        destinations = {}
        for method in methods:
            destination = args.out_root / method / group / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destinations[method] = destination
        audit.append(materialize_array(source, destinations, args.chunk_size))

    payload = {
        "status": "complete",
        "source_root": str(args.source_root),
        "methods": {
            "stride2": "physical 40-580 Hz filter at 4096 Hz, then x[..., ::2]",
            "polyphase": "physical 40-580 Hz filter at 4096 Hz, then scipy.signal.resample_poly(1,2,Kaiser beta=8.6)",
        },
        "common_processing": {
            "source_is_already_detector_psd_whitened": True,
            "source_sample_rate_hz": RAW_FS,
            "model_sample_rate_hz": MODEL_FS,
            "gps_window": "[GPS-1.75 s, GPS+0.25 s]",
            "model_samples": int(MODEL_FS * MODEL_WINDOW_S),
            "bandpass_hz": [BAND_LOW_HZ, BAND_HIGH_HZ],
            "context_seconds": CONTEXT_S,
            "tukey_alpha": 0.02,
            "scale": "per-event per-channel median/MAD robust scaling",
        },
        "source_arrays_modified": False,
        "arrays": audit,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "preprocessing_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
