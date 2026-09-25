#!/usr/bin/env python3
"""Build response-derived ET sky-localization widths for sky-v8.

Run this script with the isolated ``.venv_sky_v8`` interpreter.  It uses the
public GWFAST implementation, the ET-D ASD, the triangular ET response, Earth
rotation, and the same standard angular priors shown in the GWFAST tutorial.

The Fisher result is only the *local width of one sky mode*.  The downstream
sky-v8 map generator adds the eight exact short-signal ET-triangle modes
described by Santoliquido et al. (2025); therefore this table must not be
interpreted as a complete Bayesian sky posterior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from astropy.cosmology import Planck15


# GWFAST 1.1.2 still imports the pre-SciPy-1.14 aliases.
if not hasattr(integrate, "cumtrapz"):
    integrate.cumtrapz = integrate.cumulative_trapezoid
if not hasattr(integrate, "simps"):
    integrate.simps = integrate.simpson

from gwfast import fisherTools, gwfastGlobals, gwfastUtils, signal, waveforms


REPO = Path(__file__).resolve().parents[2]
V7_ET = REPO / "results/et3_v7_aligned_20260723/formal"
OUTPUT_ROOT = REPO / "results/unified_sky_v8_20260724"
ET_SOURCE_ROOT = Path(
    "/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root"
)
SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
FULL_SKY_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2
FAMILY_SOURCE_DIR = {
    "SIS": "SIS_data_0222",
    "PM": "PM_data_0222",
    "unlensed": "Unlensed_data_0222",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_union() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        seed_dir = V7_ET / f"seed_{seed}"
        for split in ("validation", "test"):
            frame = pd.read_parquet(seed_dir / f"{split}_event_catalog.parquet")
            frame["source_seed"] = seed
            frame["source_split"] = split
            rows.append(frame)
    union = pd.concat(rows, ignore_index=True)
    union["event_key"] = (
        union["family"].astype(str)
        + ":"
        + union["source_index"].astype(str)
        + ":"
        + union["image"].astype(str)
    )
    checks = (
        union.groupby("event_key", sort=False)
        .agg(
            gps_nunique=("gps", "nunique"),
            snr_nunique=("snr", "nunique"),
            ra_nunique=("ra", "nunique"),
            dec_nunique=("dec", "nunique"),
        )
        .reset_index()
    )
    if (checks.iloc[:, 1:] > 1).any().any():
        raise RuntimeError("An ET event_key maps to inconsistent physical values")
    return union.drop_duplicates("event_key", keep="first").reset_index(drop=True)


def load_source_tables() -> dict[str, pd.DataFrame]:
    return {
        family: pd.read_csv(ET_SOURCE_ROOT / directory / "source_samples.csv")
        for family, directory in FAMILY_SOURCE_DIR.items()
    }


def source_columns(
    frame: pd.DataFrame,
    source_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "luminosity_distance",
        "theta_jn",
        "psi",
        "phase",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
    ):
        output[column] = np.nan
    output["redshift_planck15_reconstructed"] = np.nan
    output["detector_logmc_reconstructed"] = np.nan
    output["logitq_reconstructed"] = np.nan
    redshift_grid = np.linspace(0.0, 5.0, 250_001)
    luminosity_distance_grid = Planck15.luminosity_distance(redshift_grid).value
    for family, table in source_tables.items():
        keep = output["family"].astype(str) == family
        indices = output.loc[keep, "source_index"].to_numpy(dtype=np.int64)
        selected = table.iloc[indices]
        for column in (
            "luminosity_distance",
            "theta_jn",
            "psi",
            "phase",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
        ):
            output.loc[keep, column] = selected[column].to_numpy(dtype=np.float64)
        distance = selected["luminosity_distance"].to_numpy(dtype=np.float64)
        redshift = np.interp(
            distance,
            luminosity_distance_grid,
            redshift_grid,
        )
        mass_1 = selected["mass_1_source"].to_numpy(dtype=np.float64)
        mass_2 = selected["mass_2_source"].to_numpy(dtype=np.float64)
        mass_high = np.maximum(mass_1, mass_2)
        mass_low = np.minimum(mass_1, mass_2)
        chirp_mass_source = (
            (mass_high * mass_low) ** (3.0 / 5.0)
            / (mass_high + mass_low) ** (1.0 / 5.0)
        )
        detector_logmc = np.log(chirp_mass_source * (1.0 + redshift))
        # Match detector_frame_targets() in the frozen ET-v7 data loader.
        q = np.clip(mass_low / mass_high, 1e-4, 1.0 - 1e-4)
        logit_q = np.log(q / (1.0 - q))
        output.loc[keep, "redshift_planck15_reconstructed"] = redshift
        output.loc[keep, "detector_logmc_reconstructed"] = detector_logmc
        output.loc[keep, "logitq_reconstructed"] = logit_q
    if output[
        [
            "luminosity_distance",
            "theta_jn",
            "psi",
            "phase",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
        ]
    ].isna().any().any():
        raise RuntimeError("Missing source parameters for ET Fisher calculation")
    existing = output["target_logmc"].notna()
    if existing.any():
        max_logmc_difference = float(
            np.max(
                np.abs(
                    output.loc[existing, "target_logmc"]
                    - output.loc[existing, "detector_logmc_reconstructed"]
                )
            )
        )
        max_logitq_difference = float(
            np.max(
                np.abs(
                    output.loc[existing, "target_logitq"]
                    - output.loc[existing, "logitq_reconstructed"]
                )
            )
        )
        if max_logmc_difference > 1e-4 or max_logitq_difference > 1e-4:
            raise RuntimeError(
                "Reconstructed source masses disagree with lensed event labels: "
                f"dlogMc={max_logmc_difference}, dlogitq={max_logitq_difference}"
            )
    missing = output["target_logmc"].isna() | output["target_logitq"].isna()
    output.loc[missing, "target_logmc"] = output.loc[
        missing, "detector_logmc_reconstructed"
    ]
    output.loc[missing, "target_logitq"] = output.loc[
        missing, "logitq_reconstructed"
    ]
    output["mass_labels_reconstructed_from_source_table"] = missing
    return output


def event_parameters(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    logit_q = frame["target_logitq"].to_numpy(dtype=np.float64)
    q = 1.0 / (1.0 + np.exp(-np.clip(logit_q, -40, 40)))
    return {
        "Mc": np.exp(frame["target_logmc"].to_numpy(dtype=np.float64)),
        "eta": q / (1.0 + q) ** 2,
        "dL": frame["luminosity_distance"].to_numpy(dtype=np.float64) / 1000.0,
        "theta": math.pi / 2.0 - frame["dec"].to_numpy(dtype=np.float64),
        "phi": frame["ra"].to_numpy(dtype=np.float64),
        "iota": frame["theta_jn"].to_numpy(dtype=np.float64),
        "psi": frame["psi"].to_numpy(dtype=np.float64),
        "tcoal": np.asarray(
            gwfastUtils.GPSt_to_LMST(
                frame["gps"].to_numpy(dtype=np.float64),
                lat=0.0,
                long=0.0,
            )
        ),
        "Phicoal": frame["phase"].to_numpy(dtype=np.float64),
        "chi1z": frame["a_1"].to_numpy(dtype=np.float64)
        * np.cos(frame["tilt_1"].to_numpy(dtype=np.float64)),
        "chi2z": frame["a_2"].to_numpy(dtype=np.float64)
        * np.cos(frame["tilt_2"].to_numpy(dtype=np.float64)),
    }


def inspiral_duration_from_5hz(mc_detector: np.ndarray) -> np.ndarray:
    """Leading-order time from 5 Hz to merger, used only as an audit column."""
    solar_mass_seconds = 4.925490947e-6
    mc_seconds = np.maximum(mc_detector * solar_mass_seconds, 1e-12)
    return (
        5.0
        / 256.0
        * mc_seconds ** (-5.0 / 3.0)
        * (math.pi * 5.0) ** (-8.0 / 3.0)
    )


def build_signal() -> tuple[signal.GWSignal, waveforms.IMRPhenomD]:
    waveform = waveforms.IMRPhenomD()
    detector = signal.GWSignal(
        waveform,
        psd_path=str(Path(gwfastGlobals.detPath) / "ET-0000A-18.txt"),
        detector_shape="T",
        det_lat=40.516666666666666,
        det_long=9.416666666666666,
        det_xax=0.0,
        verbose=False,
        is_ASD=True,
        useEarthMotion=True,
        fmin=5.0,
        compute2arms=False,
    )
    return detector, waveform


def run_chunk(
    detector: signal.GWSignal,
    waveform: waveforms.IMRPhenomD,
    frame: pd.DataFrame,
    resolution: int,
) -> pd.DataFrame:
    parameters = event_parameters(frame)
    model_snr = np.asarray(detector.SNRInteg(parameters, res=resolution), dtype=np.float64)
    fisher = np.asarray(
        detector.FisherMatr(parameters, res=resolution),
        dtype=np.longdouble,
    )
    target_snr = frame["snr"].to_numpy(dtype=np.float64)
    scale = (target_snr / np.maximum(model_snr, 1e-12)) ** 2
    fisher *= scale[None, None, :]

    fixed, parameter_numbers = fisherTools.fixParams(
        fisher,
        waveform.ParNums,
        ["chi1z", "chi2z"],
    )
    angles = ["theta", "phi", "iota", "psi", "Phicoal"]
    # This is the same weak bounded-angle Gaussian prior demonstrated in the
    # official GWFAST tutorial, not a fitted constant from the present data.
    prior_precision = np.repeat(1.0 / (2.0 * math.pi**2), len(angles))
    with_prior = fisherTools.addPrior(
        fixed,
        prior_precision,
        parameter_numbers,
        angles,
    )
    covariance, inversion_error = fisherTools.CovMatr(
        with_prior,
        invMethodIn="svd",
        truncate=True,
        verbose=False,
    )
    area = fisherTools.compute_localization_region(
        covariance,
        parameter_numbers,
        parameters["theta"],
        perc_level=90,
        units="SqDeg",
    )
    condition_values = []
    for index in range(len(frame)):
        matrix = np.asarray(with_prior[:, :, index], dtype=np.float64)
        if not np.isfinite(matrix).all():
            condition_values.append(np.inf)
            continue
        try:
            condition_values.append(float(np.linalg.cond(matrix)))
        except np.linalg.LinAlgError:
            condition_values.append(np.inf)
    condition = np.asarray(condition_values, dtype=np.float64)
    output = frame[
        [
            "event_key",
            "family",
            "source_index",
            "system_id",
            "tag",
            "image",
            "snr",
            "gps",
            "ra",
            "dec",
            "target_logmc",
            "target_logitq",
            "redshift_planck15_reconstructed",
            "mass_labels_reconstructed_from_source_table",
        ]
    ].copy()
    output["gwfast_model_snr_before_catalog_rescale"] = model_snr
    output["fisher_snr_rescale"] = np.sqrt(scale)
    output["gmst_rad"] = np.mod(
        2.0 * math.pi * np.asarray(parameters["tcoal"], dtype=np.float64),
        2.0 * math.pi,
    )
    output["fisher_local_area90_deg2_raw"] = np.asarray(area, dtype=np.float64)
    output["fisher_local_area90_deg2_capped"] = np.clip(
        np.asarray(area, dtype=np.float64),
        0.0,
        FULL_SKY_DEG2,
    )
    output["fisher_inversion_error"] = np.asarray(inversion_error, dtype=np.float64)
    output["fisher_log10_condition"] = np.log10(np.maximum(condition, 1.0))
    output["inspiral_duration_from_5hz_s_audit"] = inspiral_duration_from_5hz(
        parameters["Mc"]
    )
    output["fisher_finite"] = np.isfinite(
        output[
            [
                "fisher_local_area90_deg2_raw",
                "fisher_inversion_error",
                "fisher_log10_condition",
            ]
        ]
    ).all(axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--frequency-resolution", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shared = args.output_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    canonical_path = shared / "et3_gwfast_fisher_localization_events.parquet"
    if args.merge_shards:
        parts = [
            shared
            / f"et3_gwfast_fisher_localization_part_{index:02d}_of_{args.num_shards:02d}.parquet"
            for index in range(args.num_shards)
        ]
        missing = [str(path) for path in parts if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Fisher shards: {missing}")
        merged = (
            pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
            .drop_duplicates("event_key")
            .sort_values("event_key")
            .reset_index(drop=True)
        )
        expected = len(load_union())
        if len(merged) != expected:
            raise RuntimeError(f"Merged {len(merged)} Fisher rows, expected {expected}")
        merged.to_parquet(canonical_path, index=False)
        write_json(
            shared / "et3_gwfast_fisher_merge_audit.json",
            {
                "num_shards": args.num_shards,
                "expected_events": expected,
                "merged_events": len(merged),
                "unique_event_keys": int(merged["event_key"].nunique()),
                "complete": True,
            },
        )
        finite = merged[merged["fisher_finite"]]
        write_json(
            shared / "et3_gwfast_fisher_localization_audit.json",
            {
                "version": "unified_sky_v8",
                "event_count": int(len(merged)),
                "finite_event_count": int(len(finite)),
                "gwfast_version": "1.1.2",
                "waveform": "IMRPhenomD aligned-spin Fisher",
                "detector": "one triangular ET-D site with ET1/ET2/ET3 responses",
                "et_asd": "GWFAST psds/ET-0000A-18.txt",
                "fmin_hz": 5.0,
                "earth_motion": True,
                "fisher_resolution": int(args.frequency_resolution),
                "fixed_parameters": ["chi1z", "chi2z"],
                "angular_prior_precision": 1.0 / (2.0 * math.pi**2),
                "catalog_snr_rescaling": True,
                "mass_label_reconstruction": {
                    "cosmology": "Planck15",
                    "reason": (
                        "v7 event catalogs intentionally omit supervised "
                        "intrinsic targets for unlensed events"
                    ),
                    "reconstructed_event_count": int(
                        merged[
                            "mass_labels_reconstructed_from_source_table"
                        ].sum()
                    ),
                    "validated_against_existing_lensed_labels": True,
                },
                "area90_quantiles_deg2": {
                    str(quantile): float(
                        finite["fisher_local_area90_deg2_capped"].quantile(quantile)
                    )
                    for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                },
                "fraction_local_area_at_full_sky_cap": float(
                    np.mean(
                        finite["fisher_local_area90_deg2_raw"] >= FULL_SKY_DEG2
                    )
                ),
                "scope_warning": (
                    "These are local one-mode Fisher widths. The complete v8 "
                    "ET sky posterior adds the published eight-mode "
                    "ET-triangle degeneracy and remains a posterior surrogate."
                ),
            },
        )
        print(f"Merged {len(merged)} rows into {canonical_path}")
        return

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    output_path = (
        canonical_path
        if args.num_shards == 1
        else shared
        / (
            f"et3_gwfast_fisher_localization_part_{args.shard_index:02d}"
            f"_of_{args.num_shards:02d}.parquet"
        )
    )
    audit_name = (
        "et3_gwfast_fisher_localization_audit.json"
        if args.num_shards == 1
        else (
            f"et3_gwfast_fisher_localization_audit_part_{args.shard_index:02d}"
            f"_of_{args.num_shards:02d}.json"
        )
    )
    audit_path = shared / audit_name
    if (
        output_path.exists()
        and audit_path.exists()
        and not args.force
        and args.limit is None
    ):
        print(f"Reusing {output_path}")
        return

    union = source_columns(load_union(), load_source_tables())
    if args.limit is not None:
        union = union.iloc[: args.limit].copy()
    if args.num_shards > 1:
        shard_indices = np.array_split(np.arange(len(union)), args.num_shards)[
            args.shard_index
        ]
        union = union.iloc[shard_indices].reset_index(drop=True)
    detector, waveform = build_signal()
    chunks = []
    resume_from = 0
    if output_path.exists() and not args.force and args.limit is None:
        existing = pd.read_parquet(output_path)
        resume_from = len(existing)
        if resume_from > len(union):
            raise RuntimeError(
                f"Partial output has {resume_from} rows for a {len(union)}-row shard"
            )
        chunks.append(existing)
        print(f"Resuming {output_path} from row {resume_from}", flush=True)
    started = time.time()
    for start in range(resume_from, len(union), args.chunk_size):
        stop = min(start + args.chunk_size, len(union))
        chunk_started = time.time()
        chunk = run_chunk(
            detector,
            waveform,
            union.iloc[start:stop].copy(),
            args.frequency_resolution,
        )
        chunks.append(chunk)
        partial = pd.concat(chunks, ignore_index=True)
        partial.to_parquet(output_path, index=False)
        print(
            f"ET Fisher {stop}/{len(union)} "
            f"chunk={time.time()-chunk_started:.1f}s total={time.time()-started:.1f}s",
            flush=True,
        )

    result = pd.concat(chunks, ignore_index=True)
    finite = result[result["fisher_finite"]]
    audit = {
        "version": "unified_sky_v8",
        "event_count": int(len(result)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "finite_event_count": int(len(finite)),
        "gwfast_version": "1.1.2",
        "waveform": "IMRPhenomD aligned-spin Fisher",
        "detector": "one triangular ET-D site with ET1/ET2/ET3 responses",
        "et_asd": "GWFAST psds/ET-0000A-18.txt",
        "fmin_hz": 5.0,
        "earth_motion": True,
        "fisher_resolution": int(args.frequency_resolution),
        "fixed_parameters": ["chi1z", "chi2z"],
        "angular_prior_precision": 1.0 / (2.0 * math.pi**2),
        "catalog_snr_rescaling": True,
        "area90_quantiles_deg2": {
            str(quantile): float(
                finite["fisher_local_area90_deg2_capped"].quantile(quantile)
            )
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "fraction_local_area_at_full_sky_cap": float(
            np.mean(finite["fisher_local_area90_deg2_raw"] >= FULL_SKY_DEG2)
        ),
        "scope_warning": (
            "These are local one-mode Fisher widths. The complete v8 ET sky "
            "posterior is generated later with the published eight-mode "
            "ET-triangle degeneracy and is labeled a posterior surrogate."
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
