"""Shared helpers for the v5 physical-source real-noise experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SNR_THRESHOLD = 8.0
SNR_MAXIMUM = 60.0


def empirical_detected_snrs(event_manifest: Path) -> np.ndarray:
    """Return finite network SNRs from the run-matched primary catalog.

    These values define a transparent empirical proposal for detected-event
    SNR. They are not used as a retrieval feature or final-score channel.
    """

    frame = pd.read_csv(event_manifest)
    if "include_in_primary_search" in frame:
        frame = frame[frame["include_in_primary_search"].fillna(False).astype(bool)]
    values = pd.to_numeric(frame["network_snr"], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values) & (values >= SNR_THRESHOLD)]
    if len(values) < 10:
        raise RuntimeError(f"Only {len(values)} finite primary-catalog SNR values in {event_manifest}")
    return values


def draw_detected_pair_snrs(
    metadata: pd.Series | dict[str, Any],
    empirical_snrs: np.ndarray,
    rng: np.random.Generator,
    maximum: float = SNR_MAXIMUM,
) -> tuple[float, float]:
    """Draw pair SNRs while preserving the GW-LMC image-SNR ratio.

    The fainter image is bootstrapped from the run's observed detected-event
    SNR distribution. The brighter image is set by the same GW-LMC pair's SNR
    ratio. Conditioning both images to [8, 60] mirrors the stated experiment:
    recovery of cataloged doublets, not subthreshold-image discovery.
    """

    get = metadata.get if isinstance(metadata, dict) else metadata.get
    ratio = float(get("proposal_snr_ratio"))
    if not np.isfinite(ratio) or ratio < 1.0:
        raise ValueError(f"Invalid proposal SNR ratio {ratio}")
    eligible = empirical_snrs[empirical_snrs <= maximum / ratio]
    if len(eligible) == 0:
        raise RuntimeError(f"No empirical SNR supports ratio={ratio:g} with maximum={maximum:g}")
    faint = float(rng.choice(eligible))
    bright = float(faint * ratio)
    first_is_bright = float(get("proposal_snr_image1")) >= float(get("proposal_snr_image2"))
    return (bright, faint) if first_is_bright else (faint, bright)


def draw_unlensed_snr(empirical_snrs: np.ndarray, rng: np.random.Generator) -> float:
    return float(rng.choice(empirical_snrs))


def audit_physical_source_bank(root: Path, sample_count: int = 32) -> dict[str, Any]:
    checks = []
    for family in ("SIS", "PM"):
        directory = root / f"{family}_data_0222"
        for image in (1, 2):
            path = directory / f"{family}_h_strain_{image}.npy"
            values = np.load(path, mmap_mode="r")
            sample = np.asarray(values[: min(sample_count, len(values))], dtype=np.float64)
            checks.append(
                {
                    "path": str(path),
                    "shape": list(values.shape),
                    "finite_fraction": float(np.mean(np.isfinite(sample))),
                    "max_abs": float(np.max(np.abs(sample))),
                    "median_channel_std": float(np.median(np.std(sample, axis=-1))),
                    "identical_h1_l1": int(sum(np.array_equal(row[0], row[1]) for row in sample)),
                }
            )
    path = root / "Unlensed_data_0222" / "unlensed_h_strain.npy"
    values = np.load(path, mmap_mode="r")
    sample = np.asarray(values[: min(sample_count, len(values))], dtype=np.float64)
    checks.append(
        {
            "path": str(path),
            "shape": list(values.shape),
            "finite_fraction": float(np.mean(np.isfinite(sample))),
            "max_abs": float(np.max(np.abs(sample))),
            "median_channel_std": float(np.median(np.std(sample, axis=-1))),
            "identical_h1_l1": int(sum(np.array_equal(row[0], row[1]) for row in sample)),
        }
    )
    passed = all(
        item["shape"][1:] == [2, 98304]
        and item["finite_fraction"] == 1.0
        and 1e-27 < item["max_abs"] < 1e-18
        and item["median_channel_std"] > 1e-30
        and item["identical_h1_l1"] == 0
        for item in checks
    )
    return {
        "passed": bool(passed),
        "checks": checks,
        "interpretation": "Physical strain is finite, O(1e-23), nonzero, and H1/L1 are not identical. No whitening or z-scoring is permitted in the source bank.",
    }
