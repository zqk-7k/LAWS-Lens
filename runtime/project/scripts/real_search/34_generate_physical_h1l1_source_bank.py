#!/usr/bin/env python3
"""Generate a physical-unit H1/L1 lensed-BBH source bank.

This generator exists because the legacy match-style ``h_strain`` arrays were
already whitened and z-scored.  Those arrays cannot be added to public GWOSC
strain or assigned a PSD-defined SNR.  Here every saved waveform is Bilby's
detector response in dimensionless physical strain, before whitening,
normalization, or detector-noise injection.

The source/lens/image proposal is the published GW-LMC 2.5PLUS BBH
``Any_Detected_SNR1`` catalog.  The two compatibility directory names have the
following explicit meanings in this data product:

* ``SIS_data_0222``: GW-LMC smooth/non-subhalo lens realizations.
* ``PM_data_0222``: GW-LMC subhalo-present lens realizations.

They are not asserted to be analytic SIS and point-mass lens families.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import bilby
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.real_search.physical_common import (  # noqa: E402
    LiveSegment,
    SECONDS_PER_DAY,
    in_live_segments,
    sample_live_times,
    write_json,
)


GW_LMC_ROOT = Path("/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1")
FS = 4096
DURATION = 24
N_SAMPLES = FS * DURATION
END_AFTER_GEOCENTER_SECONDS = 0.25
FAMILY_SLOTS = {
    "SIS": (False, "gwlmc_smooth_non_subhalo"),
    "PM": (True, "gwlmc_subhalo_present"),
}


def parse_list(value: Any) -> list[float]:
    parsed = ast.literal_eval(str(value))
    return [float(x) for x in parsed]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tables(root: Path) -> pd.DataFrame:
    paths = {
        "source": next(root.glob("*_SourceParams.csv")),
        "lens": next(root.glob("*_LensParams.csv")),
        "image": next(root.glob("*_ImageParams.csv")),
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    lengths = {name: len(frame) for name, frame in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"GW-LMC row counts differ: {lengths}")
    lens = frames["lens"].drop(columns=["event_id"], errors="ignore").add_prefix("lens_")
    image = frames["image"].drop(columns=["event_id"], errors="ignore").add_prefix("image_")
    out = pd.concat(
        [frames["source"].reset_index(drop=True), lens.reset_index(drop=True), image.reset_index(drop=True)],
        axis=1,
    )
    out["gwlmc_row"] = np.arange(len(out), dtype=np.int64)
    out.attrs["input_paths"] = {name: str(path) for name, path in paths.items()}
    out.attrs["input_sha256"] = {name: sha256(path) for name, path in paths.items()}
    return out


def morse_index(kappa: float, gamma1: float, gamma2: float) -> float:
    """Classify minimum/saddle/maximum from the two Jacobian eigenvalues."""
    gamma = math.hypot(float(gamma1), float(gamma2))
    eig1 = 1.0 - float(kappa) - gamma
    eig2 = 1.0 - float(kappa) + gamma
    if eig1 > 0.0 and eig2 > 0.0:
        return 0.0
    if eig1 * eig2 < 0.0:
        return 0.5
    return 1.0


def choose_images(row: pd.Series, minimum_proposal_snr: float = 1.0) -> dict[str, Any] | None:
    snr = parse_list(row.image_img_snrs)
    mag = parse_list(row.image_img_mags)
    delay = parse_list(row.image_img_delays_days)
    kappa = parse_list(row.image_img_kappa)
    gamma1 = parse_list(row.image_img_gamma1)
    gamma2 = parse_list(row.image_img_gamma2)
    lengths = {len(x) for x in (snr, mag, delay, kappa, gamma1, gamma2)}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        return None
    detected = [idx for idx, value in enumerate(snr) if np.isfinite(value) and value >= minimum_proposal_snr]
    if len(detected) < 2:
        return None
    selected = sorted(detected, key=lambda idx: snr[idx], reverse=True)[:2]
    selected = sorted(selected, key=lambda idx: delay[idx])
    a, b = selected
    dt = abs(float(delay[b]) - float(delay[a]))
    if not np.isfinite(dt) or dt <= 0.0:
        return None
    return {
        "image1_index": int(a),
        "image2_index": int(b),
        "delay_days": dt,
        "mu_image1": float(mag[a]),
        "mu_image2": float(mag[b]),
        "proposal_snr_image1": float(snr[a]),
        "proposal_snr_image2": float(snr[b]),
        "proposal_snr_ratio": float(max(snr[a], snr[b]) / max(min(snr[a], snr[b]), 1e-12)),
        "morse_image1": morse_index(kappa[a], gamma1[a], gamma2[a]),
        "morse_image2": morse_index(kappa[b], gamma1[b], gamma2[b]),
    }


def load_schedule(path: Path) -> list[LiveSegment]:
    frame = pd.read_csv(path)
    return [
        LiveSegment(float(row.start_gps), float(row.end_gps), str(row.run), float(row.weight_per_second))
        for row in frame.itertuples(index=False)
    ]


def place_pair(delay_days: float, schedule: list[LiveSegment], rng: np.random.Generator) -> tuple[float, float] | None:
    """Sample uniformly from the exact live-time intersection at a fixed delay.

    For image-one time t and delay d, both detections are observable exactly
    when t belongs to L intersect (L-d). A two-pointer interval intersection
    computes that window without rejection-sampling failures.
    """
    delay_seconds = float(delay_days) * SECONDS_PER_DAY
    ordered = sorted(schedule, key=lambda item: item.start)
    i = 0
    j = 0
    intersections: list[tuple[float, float, float]] = []
    while i < len(ordered) and j < len(ordered):
        first_segment = ordered[i]
        shifted_start = ordered[j].start - delay_seconds
        shifted_end = ordered[j].end - delay_seconds
        lo = max(first_segment.start, shifted_start)
        hi = min(first_segment.end, shifted_end)
        if hi > lo:
            # The first-image event-intensity weight is the same convention as
            # sample_live_times() and the frozen time-LR null model.
            intersections.append((lo, hi, first_segment.weight_per_second))
        if first_segment.end <= shifted_end:
            i += 1
        else:
            j += 1
    if not intersections:
        return None
    weights = np.asarray([(hi - lo) * rate for lo, hi, rate in intersections], dtype=np.float64)
    selected = int(rng.choice(len(intersections), p=weights / weights.sum()))
    lo, hi, _ = intersections[selected]
    first = float(rng.uniform(lo, hi))
    return first, first + delay_seconds


def source_parameters(row: pd.Series, geocent_time: float) -> dict[str, float]:
    m1 = float(row.m1_det)
    m2 = float(row.m2_det)
    if m2 > m1:
        m1, m2 = m2, m1
    return {
        "mass_1": m1,
        "mass_2": m2,
        "a_1": float(row.a1),
        "a_2": float(row.a2),
        "tilt_1": float(row.tilt1),
        "tilt_2": float(row.tilt2),
        "phi_12": float(row.phi12),
        "phi_jl": float(row.phijl),
        "luminosity_distance": float(row.dl_source),
        "theta_jn": float(row.theta_jn),
        "psi": float(row.psi),
        "phase": float(row.phase),
        "geocent_time": float(geocent_time),
        "ra": float(row.ra),
        "dec": float(row.dec),
    }


def lens_factor(magnification: float, morse: float) -> complex:
    return math.sqrt(abs(float(magnification))) * np.exp(-1j * math.pi * float(morse))


def build_waveform_generator() -> bilby.gw.WaveformGenerator:
    return bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=FS,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomXPHM",
            "reference_frequency": 20.0,
            "minimum_frequency": 20.0,
        },
    )


def detector_response(
    waveform_generator: bilby.gw.WaveformGenerator,
    ifos: list[Any],
    parameters: dict[str, float],
    factor: complex,
) -> tuple[np.ndarray, dict[str, float]]:
    polarizations = waveform_generator.frequency_domain_strain(parameters)
    polarizations = {name: np.asarray(values) * factor for name, values in polarizations.items()}
    channels = []
    peaks: dict[str, float] = {}
    start = float(parameters["geocent_time"]) - (DURATION - END_AFTER_GEOCENTER_SECONDS)
    for ifo in ifos:
        ifo.set_strain_data_from_zero_noise(FS, DURATION, start_time=start)
        ifo.inject_signal(parameters=parameters, injection_polarizations=polarizations, raise_error=True)
        values = np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64)
        if values.shape != (N_SAMPLES,) or np.mean(np.isfinite(values)) < 1.0:
            raise ValueError(f"Invalid physical response for {ifo.name}: {values.shape}")
        channels.append(values.astype(np.float32))
        peak = int(np.argmax(np.abs(values)))
        peaks[f"{ifo.name}_peak_offset_s"] = float(ifo.strain_data.time_array[peak] - parameters["geocent_time"])
    return np.stack(channels), peaks


def select_lensed_rows(
    table: pd.DataFrame,
    subhalo: bool,
    count: int,
    schedule: list[LiveSegment],
    rng: np.random.Generator,
    maximum_proposal_snr_ratio: float,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(table.lens_is_subhalo.astype(bool).to_numpy() == bool(subhalo))
    indices = rng.permutation(indices)
    selected: list[dict[str, Any]] = []
    for index in indices:
        row = table.iloc[int(index)]
        images = choose_images(row)
        if images is None:
            continue
        # The deployment experiment is conditional on both images being above
        # the catalog threshold. Extremely unequal image pairs would require
        # an implausibly loud bright image after enforcing rho_faint >= 8.
        if images["proposal_snr_ratio"] > maximum_proposal_snr_ratio:
            continue
        times = place_pair(images["delay_days"], schedule, rng)
        if times is None:
            continue
        selected.append({"row_index": int(index), "gps_image1": times[0], "gps_image2": times[1], **images})
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} feasible systems for subhalo={subhalo}; requested {count}")


def open_array(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def generate_family(
    slot: str,
    label: str,
    selected: list[dict[str, Any]],
    table: pd.DataFrame,
    out_root: Path,
    generator: bilby.gw.WaveformGenerator,
    ifos: list[Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = out_root / f"{slot}_data_0222"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{slot}_h_strain_1.npy", out / f"{slot}_h_strain_2.npy"]
    arrays = [open_array(path, (len(selected), 2, N_SAMPLES)) for path in paths]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for local_index, item in enumerate(selected):
        if local_index % max(1, len(selected) // 50) == 0:
            print(f"{slot}/{label}: {local_index}/{len(selected)}", flush=True)
        row = table.iloc[item["row_index"]]
        record = {
            "family_slot": slot,
            "physical_lens_group": label,
            "pair_id": local_index,
            "gwlmc_row": int(row.gwlmc_row),
            "gwlmc_event_id": int(row.event_id),
            "mass_1_detector": float(row.m1_det),
            "mass_2_detector": float(row.m2_det),
            "redshift": float(row.z_source),
            "luminosity_distance_mpc": float(row.dl_source),
            "ra": float(row.ra),
            "dec": float(row.dec),
            **item,
        }
        try:
            for image_number in (1, 2):
                parameters = source_parameters(row, float(item[f"gps_image{image_number}"]))
                factor = lens_factor(float(item[f"mu_image{image_number}"]), float(item[f"morse_image{image_number}"]))
                values, peaks = detector_response(generator, ifos, parameters, factor)
                arrays[image_number - 1][local_index] = values
                record.update({f"image{image_number}_{key}": value for key, value in peaks.items()})
                record[f"image{image_number}_max_abs_strain"] = float(np.max(np.abs(values)))
                record[f"image{image_number}_h1_l1_equal"] = bool(np.array_equal(values[0], values[1]))
        except Exception as exc:
            arrays[0][local_index] = 0.0
            arrays[1][local_index] = 0.0
            failures.append({"pair_id": local_index, "gwlmc_row": int(row.gwlmc_row), "error": repr(exc)})
            record["generation_error"] = repr(exc)
        rows.append(record)
    for values in arrays:
        values.flush()
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "physical_source_pair_metadata.parquet", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "physical_source_generation_failures.csv", index=False)
    summary = {
        "family_slot": slot,
        "physical_lens_group": label,
        "n_requested": len(selected),
        "n_failures": len(failures),
        "elapsed_s": float(time.perf_counter() - started),
        "delay_days_median": float(frame.delay_days.median()),
        "delay_days_p90": float(frame.delay_days.quantile(0.9)),
        "strain_max_abs_min": float(min(frame.image1_max_abs_strain.min(), frame.image2_max_abs_strain.min())),
        "strain_max_abs_max": float(max(frame.image1_max_abs_strain.max(), frame.image2_max_abs_strain.max())),
        "identical_h1_l1_count": int(frame.image1_h1_l1_equal.sum() + frame.image2_h1_l1_equal.sum()),
    }
    return frame, summary


def generate_unlensed(
    table: pd.DataFrame,
    count: int,
    excluded_rows: set[int],
    schedule: list[LiveSegment],
    rng: np.random.Generator,
    out_root: Path,
    generator: bilby.gw.WaveformGenerator,
    ifos: list[Any],
) -> dict[str, Any]:
    candidates = np.asarray([idx for idx in range(len(table)) if idx not in excluded_rows], dtype=np.int64)
    chosen = rng.choice(candidates, size=count, replace=False)
    out = out_root / "Unlensed_data_0222"
    out.mkdir(parents=True, exist_ok=True)
    array = open_array(out / "unlensed_h_strain.npy", (count, 2, N_SAMPLES))
    rows = []
    failures = []
    started = time.perf_counter()
    for local_index, index in enumerate(chosen):
        if local_index % max(1, count // 50) == 0:
            print(f"unlensed: {local_index}/{count}", flush=True)
        row = table.iloc[int(index)]
        gps = float(sample_live_times(schedule, 1, rng)[0])
        record = {
            "sample_index": local_index,
            "gwlmc_row": int(row.gwlmc_row),
            "gwlmc_event_id": int(row.event_id),
            "gps": gps,
            "mass_1_detector": float(row.m1_det),
            "mass_2_detector": float(row.m2_det),
            "redshift": float(row.z_source),
            "luminosity_distance_mpc": float(row.dl_source),
            "ra": float(row.ra),
            "dec": float(row.dec),
        }
        try:
            values, peaks = detector_response(generator, ifos, source_parameters(row, gps), 1.0 + 0.0j)
            array[local_index] = values
            record.update(peaks)
            record["max_abs_strain"] = float(np.max(np.abs(values)))
            record["h1_l1_equal"] = bool(np.array_equal(values[0], values[1]))
        except Exception as exc:
            array[local_index] = 0.0
            failures.append({"sample_index": local_index, "gwlmc_row": int(row.gwlmc_row), "error": repr(exc)})
            record["generation_error"] = repr(exc)
        rows.append(record)
    array.flush()
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "physical_unlensed_source_metadata.parquet", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "physical_source_generation_failures.csv", index=False)
    return {
        "n_requested": count,
        "n_failures": len(failures),
        "elapsed_s": float(time.perf_counter() - started),
        "strain_max_abs_min": float(frame.max_abs_strain.min()),
        "strain_max_abs_max": float(frame.max_abs_strain.max()),
        "identical_h1_l1_count": int(frame.h1_l1_equal.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", choices=("GWTC3", "GWTC4"), required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gwlmc-root", type=Path, default=GW_LMC_ROOT)
    parser.add_argument("--n-per-family", type=int, default=600)
    parser.add_argument("--n-unlensed", type=int, default=600)
    parser.add_argument("--maximum-proposal-snr-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    logging.getLogger("bilby").setLevel(logging.ERROR)
    bilby.core.utils.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    table = load_tables(args.gwlmc_root)
    input_paths = table.attrs["input_paths"]
    input_sha256 = table.attrs["input_sha256"]
    schedule = load_schedule(args.schedule)
    generator = build_waveform_generator()
    ifos = list(bilby.gw.detector.InterferometerList(["H1", "L1"]))

    summary: dict[str, Any] = {
        "status": "in_progress",
        "protocol": "physical_h1l1_detector_response_v1",
        "deployment": args.deployment,
        "seed": args.seed,
        "gwlmc_root": str(args.gwlmc_root),
        "gwlmc_input_paths": input_paths,
        "gwlmc_input_sha256": input_sha256,
        "source_population": "GW-LMC 2.5PLUS BBH Any_Detected_SNR1 proposal",
        "family_slot_mapping": {
            "SIS_data_0222": "GW-LMC smooth/non-subhalo; compatibility name only",
            "PM_data_0222": "GW-LMC subhalo-present; compatibility name only",
        },
        "detectors": ["H1", "L1"],
        "units": "dimensionless physical detector strain",
        "whitened_at_source_generation": False,
        "zscored_at_source_generation": False,
        "waveform_approximant": "IMRPhenomXPHM",
        "sampling_frequency_hz": FS,
        "duration_s": DURATION,
        "window_relative_to_geocenter_s": [-(DURATION - END_AFTER_GEOCENTER_SECONDS), END_AFTER_GEOCENTER_SECONDS],
        "schedule": str(args.schedule),
        "time_placement": "Both images are placed in actual H1-L1 joint public DATA exposure; second time equals first plus its own GW-LMC delay.",
        "detectable_pair_conditioning": f"Both proposal images have SNR >= 1 and proposal SNR ratio <= {args.maximum_proposal_snr_ratio:g}; deployment SNR is assigned later from the run-matched empirical detected-event distribution.",
        "morse_classification": "Jacobian eigenvalues from GW-LMC kappa/gamma; minimum=0, saddle=1/2, maximum=1.",
        "reference_urls": {
            "gwlmc_zenodo": "https://doi.org/10.5281/zenodo.19212270",
            "bilby_detector_data": "https://lscsoft.docs.ligo.org/bilby/transient-gw-data.html",
        },
    }
    write_json(args.out_root / "physical_source_bank_summary.json", summary)

    selected_by_slot: dict[str, list[dict[str, Any]]] = {}
    excluded: set[int] = set()
    for slot, (subhalo, label) in FAMILY_SLOTS.items():
        selected = select_lensed_rows(
            table,
            subhalo,
            args.n_per_family,
            schedule,
            rng,
            args.maximum_proposal_snr_ratio,
        )
        selected_by_slot[slot] = selected
        excluded.update(item["row_index"] for item in selected)
        _, family_summary = generate_family(slot, label, selected, table, args.out_root, generator, ifos)
        summary[slot] = family_summary
        write_json(args.out_root / "physical_source_bank_summary.json", summary)

    summary["unlensed"] = generate_unlensed(
        table,
        args.n_unlensed,
        excluded,
        schedule,
        rng,
        args.out_root,
        generator,
        ifos,
    )
    failures = sum(summary[slot]["n_failures"] for slot in FAMILY_SLOTS) + summary["unlensed"]["n_failures"]
    identical = sum(summary[slot]["identical_h1_l1_count"] for slot in FAMILY_SLOTS) + summary["unlensed"]["identical_h1_l1_count"]
    summary["total_generation_failures"] = int(failures)
    summary["total_identical_h1_l1_waveforms"] = int(identical)
    summary["status"] = "complete" if failures == 0 and identical == 0 else "failed_audit"
    write_json(args.out_root / "physical_source_bank_summary.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError(f"Physical source bank failed audit: failures={failures}, identical_H1L1={identical}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
