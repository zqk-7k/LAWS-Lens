#!/usr/bin/env python3
"""Independent BAYESTAR sky-PE replacement for the rotated-template injection maps.

The experiment freezes the v9.3/v9.4 waveform and one-dimensional time
channels.  It creates an event-specific BAYESTAR posterior for every O3/O4a
validation and test injection from the physical source parameters, event GPS,
H1/L1 response, the exact off-source PSD bank entry, and an independent
Gaussian matched-filter measurement realization.  Public real-event PE maps
remain the input for the real GWTC ranking.

This is rapid conditional sky PE, not full BBH parameter estimation.  Results
are written to a new directory and never overwrite historical products.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import time
import traceback
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import healpy as hp
import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path("/root/autodl-tmp/gw-catalog")
MODEL_SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENTS = ("gwtc3", "gwtc4")
SPLITS = ("validation", "test")
ANALYSIS_NSIDE = 512
COARSE_NSIDE = 256
REFERENCE_NSIDE = 1024
PRIMARY_WAVEFORM = "IMRPhenomPv2"
FALLBACK_WAVEFORM = "IMRPhenomD"
F_LOW_HZ = 20.0
F_HIGH_HZ = 1024.0
BAYESTAR_LOG_LIKELIHOOD_SCALE = 0.83
TEMPERATURE_GRID = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
BOOTSTRAP_DRAWS = 10_000

INPUT_ROOT = PROJECT / "results/real_noise_injection_v7_peak2s_formal_20260722"
SOURCE_ROOT = PROJECT / "results/real_noise_injection_v5_physical_source_20260721"
BASE_SCRIPT = PROJECT / "scripts/experiments/gwtc_c_scheme_ordering_confirmation_20260831.py"
ROTATED_TEMPLATE_C_ROOT = (
    PROJECT / "results/gwtc_c_scheme_ordering_confirmation_20260831_20260831T094500Z"
)
SOURCE_CSV = next(
    (PROJECT.parent / "GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1").glob("*_SourceParams.csv")
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SCRIPT, "bayestar_c_scheme_base")


def seed_dir(deployment: str, seed: int) -> Path:
    return INPUT_ROOT / deployment / f"seed_{seed}"


def shared_dir(deployment: str) -> Path:
    return SOURCE_ROOT / deployment / "shared"


def events_path(deployment: str, seed: int, split: str) -> Path:
    short = "val" if split == "validation" else "test"
    return seed_dir(deployment, seed) / f"results/mixed_{short}_synthetic_events_v7.parquet"


def event_map_dir(root: Path, deployment: str, seed: int, split: str) -> Path:
    return root / "event_maps" / deployment / f"seed_{seed}" / split


def dense_cache_dir(root: Path, deployment: str, seed: int, split: str) -> Path:
    return root / "work" / "dense_nside512" / deployment / f"seed_{seed}" / split


def map_paths(root: Path, deployment: str, seed: int, split: str, index: int) -> dict[str, Path]:
    stem = f"event_{int(index):04d}"
    return {
        "moc": event_map_dir(root, deployment, seed, split) / f"{stem}.fits.gz",
        "dense": dense_cache_dir(root, deployment, seed, split) / f"{stem}.npy",
        "metrics": event_map_dir(root, deployment, seed, split) / f"{stem}.json",
        "failure": event_map_dir(root, deployment, seed, split) / f"{stem}.failure.json",
    }


def initialize_root(root: Path) -> None:
    if root.exists() and not (root / "contracts/ANALYSIS_CONTRACT.json").exists():
        raise RuntimeError(f"Refusing ambiguous pre-existing output directory: {root}")
    for name in ("contracts", "results", "reports", "figures", "scripts", "logs", "manifest", "event_maps", "work"):
        (root / name).mkdir(parents=True, exist_ok=True)


def input_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        for seed in MODEL_SEEDS:
            metadata_path = seed_dir(deployment, seed) / "data/real_noise_injections/compact_injection_metadata.parquet"
            for split in SPLITS:
                events = pd.read_parquet(events_path(deployment, seed, split))
                rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "split": split,
                        "n_events": len(events),
                        "n_lensed_images": int(events.tag.isin(["L1", "L2"]).sum()),
                        "n_unlensed": int(events.tag.eq("U").sum()),
                        "n_unique_lensed_systems": int(events.loc[events.tag.isin(["L1", "L2"]), ["family", "source_index"]].drop_duplicates().shape[0]),
                        "events_path": str(events_path(deployment, seed, split)),
                        "events_sha256": sha256_file(events_path(deployment, seed, split)),
                        "metadata_path": str(metadata_path),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                )
    return pd.DataFrame(rows)


def prepare(root: Path) -> None:
    initialize_root(root)
    target_script = root / "scripts" / Path(__file__).name
    if Path(__file__).resolve() != target_script.resolve():
        shutil.copy2(Path(__file__), target_script)
    inventory = input_inventory()
    write_csv(root / "contracts/INPUT_INVENTORY.csv", inventory)
    frozen_paths = BASE.critical_historical_paths()
    frozen = BASE.snapshot_hashes(frozen_paths)
    write_csv(root / "contracts/HISTORICAL_INPUT_HASHES_BEFORE.csv", frozen)
    contract = {
        "schema": "bayestar-injection-sky-full-v1",
        "created_utc": utc_stamp(),
        "status": "CONFIG_FROZEN_BEFORE_VALIDATION",
        "scientific_role": "rapid conditional sky PE for injections; not full BBH PE and not lensing confirmation",
        "historical_outputs_overwritten": False,
        "frozen_channels": ["waveform encoder/checkpoint", "Z_wf", "one-dimensional Z_time"],
        "sky_method": {
            "algorithm": "ligo.skymap BAYESTAR",
            "trigger_realization": "official Gaussian matched-filter measurement realization",
            "event_specific_inputs": [
                "GW-LMC physical source parameters",
                "event GPS",
                "H1/L1 antenna response",
                "exact off-source PSD bank entry assigned to the injection",
                "v7 target network SNR",
            ],
            "does_not_use": "rotated public PE template",
            "injection_waveform": "IMRPhenomXPHM physical source bank (frozen upstream)",
            "rapid_localization_waveform": PRIMARY_WAVEFORM,
            "fallback_waveform": FALLBACK_WAVEFORM,
            "f_low_hz": F_LOW_HZ,
            "f_high_hz": F_HIGH_HZ,
            "bayestar_rescale_loglikelihood": BAYESTAR_LOG_LIKELIHOOD_SCALE,
            "analysis_nside": ANALYSIS_NSIDE,
            "moc_persisted": True,
            "dense_maps_persisted": False,
        },
        "measurement_noise": {
            "model": "Gaussian matched-filter noise implemented by ligo.skymap.tool.bayestar_realize_coincs.simulate_snr",
            "independent_per_event": True,
            "run_matched_nonstationarity_carried_by": "event-specific empirical off-source PSD",
            "limitation": "does not replay the exact non-Gaussian time-domain off-source realization",
        },
        "temperature_calibration": {
            "validation_only": True,
            "grid": list(TEMPERATURE_GRID),
            "unit_weighting": "each lensed system total weight 1; each unlensed event weight 1",
            "test_and_real_not_used": True,
        },
        "split_rules": {
            "evaluation_subset": "same parent-noise-bank-disjoint retained systems as 20260831 C-scheme confirmation",
            "maps_generated": "all 450 events in every validation/test catalog",
            "test_opened_only_after_selected_config_hash": True,
        },
        "weight_selection": "same validation-only C-fixed/C-retuned policy and 0.05 simplex as the ordering-corrected C experiment",
        "real_catalog": "corrected public PE maps at Nside=512; never rotated templates",
        "final_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    write_json(root / "contracts/ANALYSIS_CONTRACT.json", contract)
    print(root)


def event_records(deployment: str, seed: int, split: str) -> list[dict[str, Any]]:
    events = pd.read_parquet(events_path(deployment, seed, split)).sort_values("idx")
    metadata = pd.read_parquet(
        seed_dir(deployment, seed) / "data/real_noise_injections/compact_injection_metadata.parquet"
    )
    lookup = {
        (str(row.family), int(row.sample_index)): row
        for row in metadata.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        meta = lookup[(str(event.family), int(event.source_index))]
        image = 2 if str(event.tag).upper() == "L2" else 1
        bank = getattr(meta, f"image{image}_noise_bank_index")
        target = getattr(meta, f"image{image}_target_network_snr")
        if not np.isfinite(bank) or not np.isfinite(target):
            raise RuntimeError(f"Missing PSD/SNR provenance: {deployment}/{seed}/{split}/{event.idx}")
        morse = getattr(meta, f"morse_image{image}")
        rows.append(
            {
                **event._asdict(),
                "deployment": deployment,
                "model_seed": int(seed),
                "split": split,
                "image_number": image,
                "gwlmc_row": int(meta.gwlmc_row),
                "noise_bank_index": int(bank),
                "target_network_snr": float(target),
                "morse_index": float(morse) if np.isfinite(morse) else 0.0,
                "measurement_seed": stable_seed("bayestar", deployment, seed, split, int(event.idx)),
            }
        )
    return rows


_WORKER_ROOT: Path | None = None
_WORKER_DEPLOYMENT: str | None = None
_WORKER_SEED: int | None = None
_WORKER_SPLIT: str | None = None
_WORKER_SOURCES: pd.DataFrame | None = None
_WORKER_PSD_FREQ: np.ndarray | None = None
_WORKER_PSD_BANK: np.ndarray | None = None


def worker_init(root: str, deployment: str, seed: int, split: str) -> None:
    global _WORKER_ROOT, _WORKER_DEPLOYMENT, _WORKER_SEED, _WORKER_SPLIT
    global _WORKER_SOURCES, _WORKER_PSD_FREQ, _WORKER_PSD_BANK
    _WORKER_ROOT = Path(root)
    _WORKER_DEPLOYMENT = deployment
    _WORKER_SEED = int(seed)
    _WORKER_SPLIT = split
    _WORKER_SOURCES = pd.read_csv(SOURCE_CSV)
    shared = shared_dir(deployment)
    _WORKER_PSD_FREQ = np.load(shared / "noise_psd_frequency.npy")
    _WORKER_PSD_BANK = np.load(shared / "noise_psd_bank.npy", mmap_mode="r")
    logging.getLogger("ligo.skymap").setLevel(logging.ERROR)
    logging.getLogger("bilby").setLevel(logging.ERROR)


def lal_psd(values: np.ndarray, frequencies: np.ndarray) -> Any:
    import lal

    delta_f = float(frequencies[1] - frequencies[0])
    series = lal.CreateREAL8FrequencySeries(
        "run-matched off-source PSD",
        0,
        float(frequencies[0]),
        delta_f,
        lal.StrainUnit**2,
        len(values),
    )
    series.data.data[:] = np.asarray(values, dtype=np.float64)
    return series


def spin_components(source: pd.Series) -> tuple[float, tuple[float, ...]]:
    import bilby

    m1 = float(source.m1_det)
    m2 = float(source.m2_det)
    result = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        float(source.theta_jn),
        float(source.phijl),
        float(source.tilt1),
        float(source.tilt2),
        float(source.phi12),
        float(source.a1),
        float(source.a2),
        m1,
        m2,
        F_LOW_HZ,
        float(source.phase),
    )
    return float(result[0]), tuple(float(value) for value in result[1:])


def raster_probability(skymap: Any, nside: int) -> np.ndarray:
    from ligo.skymap import moc

    raster = moc.rasterize(skymap, order=int(round(math.log2(nside))))
    names = raster.dtype.names or ()
    if "PROB" in names:
        probability = np.asarray(raster["PROB"], dtype=np.float64)
    elif "PROBDENSITY" in names:
        probability = np.asarray(raster["PROBDENSITY"], dtype=np.float64) * hp.nside2pixarea(nside)
    else:
        raise KeyError(f"No probability column in rasterized map: {names}")
    probability = np.maximum(np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(probability.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid BAYESTAR probability normalization")
    return probability / total


def posterior_metrics(probability: np.ndarray, ra: float, dec: float) -> dict[str, Any]:
    order = np.argsort(probability)[::-1]
    sorted_probability = probability[order]
    cumulative = np.cumsum(sorted_probability, dtype=np.float64)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    truth_pixel = hp.ang2pix(
        ANALYSIS_NSIDE,
        0.5 * math.pi - float(dec),
        float(ra) % (2.0 * math.pi),
        nest=True,
    )
    result: dict[str, Any] = {
        "area50_deg2": float((np.searchsorted(cumulative, 0.5) + 1) * hp.nside2pixarea(ANALYSIS_NSIDE, degrees=True)),
        "area90_deg2": float((np.searchsorted(cumulative, 0.9) + 1) * hp.nside2pixarea(ANALYSIS_NSIDE, degrees=True)),
        "truth_credible_level_raw": float(cumulative[inverse[truth_pixel]]),
        "entropy_nats": float(-np.sum(probability * np.log(np.maximum(probability, 1e-300)), dtype=np.float64)),
        "kl_from_isotropic_nats": float(np.sum(probability * np.log(np.maximum(probability * len(probability), 1e-300)), dtype=np.float64)),
        "truth_pixel_nested": int(truth_pixel),
    }
    logp = np.log(np.maximum(sorted_probability, 1e-300))
    truth_rank = int(inverse[truth_pixel])
    for temperature in TEMPERATURE_GRID:
        scaled = np.exp((logp - logp[0]) / float(temperature))
        credible = float(np.sum(scaled[: truth_rank + 1], dtype=np.float64) / np.sum(scaled, dtype=np.float64))
        result[f"truth_credible_T{temperature:g}"] = credible
    return result


def _simulate_with_waveform(record: dict[str, Any], source: pd.Series, waveform: str) -> tuple[Any, dict[str, Any]]:
    import lal
    import lalsimulation
    from ligo.skymap.bayestar import filter, localize
    from ligo.skymap.io.events.base import Event, SingleEvent
    from ligo.skymap.tool.bayestar_realize_coincs import simulate_snr

    if _WORKER_PSD_BANK is None or _WORKER_PSD_FREQ is None:
        raise RuntimeError("Worker PSD context was not initialized")
    m1 = float(source.m1_det)
    m2 = float(source.m2_det)
    iota, spins = spin_components(source)
    s1x, s1y, s1z, s2x, s2y, s2z = spins
    if waveform == FALLBACK_WAVEFORM:
        s1x = s1y = s2x = s2y = 0.0
    template_args = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": s1x,
        "spin1y": s1y,
        "spin1z": s1z,
        "spin2x": s2x,
        "spin2y": s2y,
        "spin2z": s2z,
        "f_final": F_HIGH_HZ,
        "f_ref": F_LOW_HZ,
    }
    template = filter.sngl_inspiral_psd(waveform, f_min=F_LOW_HZ, **template_args)
    bank = int(record["noise_bank_index"])
    raw_psds = np.asarray(_WORKER_PSD_BANK[bank], dtype=np.float64)
    psd_series = [lal_psd(row, _WORKER_PSD_FREQ) for row in raw_psds]
    interpolated = [
        filter.InterpolatedPSD(filter.abscissa(item), item.data.data)
        for item in psd_series
    ]
    detectors = [
        lalsimulation.DetectorPrefixToLALDetector(name) for name in ("H1", "L1")
    ]
    epoch = lal.LIGOTimeGPS(float(record["gps_obs"]))
    gmst = lal.GreenwichMeanSiderealTime(epoch)
    common = {
        "ra": float(record["ra_true"]),
        "dec": float(record["dec_true"]),
        "psi": float(source.psi),
        "inc": iota,
        "epoch": epoch,
        "gmst": gmst,
        "H": template,
    }
    reference_distance = max(float(source.dl_source), 1e-3)
    zero = [
        simulate_snr(
            distance=reference_distance,
            S=psd,
            response=detector.response,
            location=detector.location,
            measurement_error="zero-noise",
            **common,
        )
        for psd, detector in zip(interpolated, detectors)
    ]
    zero_network = math.sqrt(sum(float(item[1]) ** 2 for item in zero))
    if not np.isfinite(zero_network) or zero_network <= 0:
        raise ValueError("Zero/invalid predicted network SNR")
    effective_distance = reference_distance * zero_network / float(record["target_network_snr"])
    np.random.seed(int(record["measurement_seed"]))
    noisy = [
        simulate_snr(
            distance=effective_distance,
            S=psd,
            response=detector.response,
            location=detector.location,
            measurement_error="gaussian-noise",
            **common,
        )
        for psd, detector in zip(interpolated, detectors)
    ]
    phase_factor = np.exp(-1j * math.pi * float(record["morse_index"]))
    adjusted = []
    for horizon, snr, phase, toa, series in noisy:
        series.data.data[:] = np.asarray(series.data.data) * phase_factor
        adjusted.append((horizon, snr, float(np.angle(np.exp(1j * phase) * phase_factor)), toa, series))

    _SingleTuple = namedtuple(
        "BayestarSingleTuple", "detector snr phase time zerolag_time psd snr_series"
    )

    class BayestarSingle(_SingleTuple, SingleEvent):
        pass

    _EventTuple = namedtuple("BayestarEventTuple", "singles template_args")

    class BayestarEvent(_EventTuple, Event):
        pass

    singles = [
        BayestarSingle(name, float(item[1]), float(item[2]), item[3], item[3], psd, item[4])
        for name, item, psd in zip(("H1", "L1"), adjusted, psd_series)
    ]
    started = time.perf_counter()
    skymap = localize(
        BayestarEvent(singles, template_args),
        waveform=waveform,
        f_low=F_LOW_HZ,
        enable_snr_series=True,
        f_high_truncate=1.0,
        rescale_loglikelihood=BAYESTAR_LOG_LIKELIHOOD_SCALE,
    )
    elapsed = time.perf_counter() - started
    audit = {
        "waveform": waveform,
        "mass1_detector": m1,
        "mass2_detector": m2,
        "effective_distance_mpc": effective_distance,
        "predicted_reference_network_snr": zero_network,
        "target_network_snr": float(record["target_network_snr"]),
        "realized_detector_snr_H1": float(adjusted[0][1]),
        "realized_detector_snr_L1": float(adjusted[1][1]),
        "realized_network_snr": float(math.sqrt(sum(float(item[1]) ** 2 for item in adjusted))),
        "realized_toa_H1": float(adjusted[0][3]),
        "realized_toa_L1": float(adjusted[1][3]),
        "realized_phase_H1": float(adjusted[0][2]),
        "realized_phase_L1": float(adjusted[1][2]),
        "bayestar_seconds": elapsed,
    }
    return skymap, audit


def localize_event(record: dict[str, Any]) -> dict[str, Any]:
    from ligo.skymap.io.fits import write_sky_map

    if _WORKER_ROOT is None or _WORKER_SOURCES is None:
        raise RuntimeError("Worker context is not initialized")
    paths = map_paths(
        _WORKER_ROOT,
        str(record["deployment"]),
        int(record["model_seed"]),
        str(record["split"]),
        int(record["idx"]),
    )
    if paths["metrics"].exists() and paths["moc"].exists() and paths["dense"].exists():
        return json.loads(paths["metrics"].read_text(encoding="utf-8"))
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        source = _WORKER_SOURCES.iloc[int(record["gwlmc_row"])]
        fallback_used = False
        try:
            skymap, audit = _simulate_with_waveform(record, source, PRIMARY_WAVEFORM)
        except Exception as primary_error:
            fallback_used = True
            skymap, audit = _simulate_with_waveform(record, source, FALLBACK_WAVEFORM)
            audit["primary_waveform_error"] = repr(primary_error)
        probability = raster_probability(skymap, ANALYSIS_NSIDE)
        uniq = np.asarray(skymap["UNIQ"], dtype=np.int64)
        from ligo.skymap import moc

        native_orders = np.asarray(moc.uniq2order(uniq), dtype=np.int64)
        metrics = {
            "deployment": str(record["deployment"]),
            "model_seed": int(record["model_seed"]),
            "split": str(record["split"]),
            "idx": int(record["idx"]),
            "family": str(record["family"]),
            "tag": str(record["tag"]),
            "pair_id": int(record["pair_id"]),
            "source_index": int(record["source_index"]),
            "gwlmc_row": int(record["gwlmc_row"]),
            "gps_obs": float(record["gps_obs"]),
            "ra_true": float(record["ra_true"]),
            "dec_true": float(record["dec_true"]),
            "noise_bank_index": int(record["noise_bank_index"]),
            "measurement_seed": int(record["measurement_seed"]),
            "morse_index": float(record["morse_index"]),
            "fallback_used": fallback_used,
            "analysis_nside": ANALYSIS_NSIDE,
            "ordering": "NESTED",
            "native_moc_order_min": int(native_orders.min()),
            "native_moc_order_max": int(native_orders.max()),
            "native_moc_nside_max": int(2 ** int(native_orders.max())),
            "probability_convention": "per-pixel probability mass",
            **audit,
            **posterior_metrics(probability, float(record["ra_true"]), float(record["dec_true"])),
            "total_seconds": time.perf_counter() - started,
        }
        temp_moc = paths["moc"].with_name(paths["moc"].stem + ".tmp.fits.gz")
        write_sky_map(temp_moc, skymap)
        temp_moc.replace(paths["moc"])
        temp_dense = paths["dense"].with_suffix(".tmp.npy")
        np.save(temp_dense, probability.astype(np.float32))
        temp_dense.replace(paths["dense"])
        metrics["moc_path"] = str(paths["moc"])
        metrics["moc_bytes"] = paths["moc"].stat().st_size
        metrics["moc_sha256"] = sha256_file(paths["moc"])
        write_json(paths["metrics"], metrics)
        paths["failure"].unlink(missing_ok=True)
        return metrics
    except Exception as exc:
        failure = {
            "record": record,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "timestamp_utc": utc_stamp(),
        }
        write_json(paths["failure"], failure)
        raise


def run_map_generation(
    root: Path,
    deployment: str,
    seed: int,
    split: str,
    workers: int,
    limit: int | None = None,
) -> pd.DataFrame:
    records = event_records(deployment, seed, split)
    if limit is not None:
        # A deterministic pilot covers both image slots of both lens-family
        # compatibility bins plus unlensed events, with exactly ``limit`` rows.
        frame = pd.DataFrame(records)
        frame["_pilot_group"] = np.where(
            frame.tag.eq("U"), "unlensed", frame.family.astype(str) + "_" + frame.tag.astype(str)
        )
        chosen: list[dict[str, Any]] = []
        groups = sorted(frame._pilot_group.unique())
        base_count, remainder = divmod(limit, len(groups))
        for group_number, group in enumerate(groups):
            part = frame.loc[frame._pilot_group.eq(group)].copy()
            part = part.assign(_h=part.idx.map(lambda value: stable_seed("pilot", deployment, seed, split, value)))
            take = base_count + int(group_number < remainder)
            chosen.extend(
                part.nsmallest(take, "_h").drop(columns=["_h", "_pilot_group"]).to_dict("records")
            )
        records = sorted(chosen, key=lambda row: int(row["idx"]))
        if len(records) != limit:
            raise RuntimeError(f"Pilot selection returned {len(records)} rows, expected {limit}")
    print(f"[maps] {deployment} seed={seed} split={split}: {len(records)} events, workers={workers}", flush=True)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_init,
        initargs=(str(root), deployment, seed, split),
    ) as pool:
        futures = {pool.submit(localize_event, record): int(record["idx"]) for record in records}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[maps] FAIL {deployment}/{seed}/{split}/idx={index}: {exc}", flush=True)
            if number % 25 == 0 or number == len(futures):
                print(f"[maps] {deployment}/{seed}/{split}: {number}/{len(futures)}", flush=True)
    frame = pd.DataFrame(results).sort_values("idx") if results else pd.DataFrame()
    output = root / "results" / deployment / f"seed_{seed}"
    write_csv(output / f"{split}_bayestar_event_metrics.csv", frame)
    return frame


def pilot(root: Path, workers: int, events_per_seed: int) -> None:
    if not (root / "contracts/ANALYSIS_CONTRACT.json").exists():
        prepare(root)
    rows = []
    for deployment in DEPLOYMENTS:
        for seed in MODEL_SEEDS:
            rows.append(run_map_generation(root, deployment, seed, "validation", workers, events_per_seed))
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    expected = len(DEPLOYMENTS) * len(MODEL_SEEDS) * events_per_seed
    ordering_rows: list[dict[str, Any]] = []
    from astropy.io import fits
    from ligo.skymap import moc
    from ligo.skymap.io.fits import read_sky_map

    for row in frame.itertuples(index=False):
        moc_path = Path(row.moc_path)
        dense_path = map_paths(
            root, str(row.deployment), int(row.model_seed), str(row.split), int(row.idx)
        )["dense"]
        with fits.open(moc_path) as hdus:
            header = hdus[1].header
            ordering = header.get("ORDERING")
            index_scheme = header.get("INDXSCHM")
            moc_order = header.get("MOCORDER")
        table = read_sky_map(moc_path, moc=True)
        raster = moc.rasterize(table, order=int(round(math.log2(ANALYSIS_NSIDE))))
        rebuilt = (
            np.asarray(raster["PROBDENSITY"], dtype=np.float64)
            * hp.nside2pixarea(ANALYSIS_NSIDE)
        )
        rebuilt /= rebuilt.sum(dtype=np.float64)
        cached = np.load(dense_path).astype(np.float64)
        cached /= cached.sum(dtype=np.float64)
        ordering_rows.append(
            {
                "deployment": row.deployment,
                "model_seed": row.model_seed,
                "idx": row.idx,
                "fits_ordering": ordering,
                "fits_index_scheme": index_scheme,
                "native_moc_order": moc_order,
                "rebuilt_probability_sum": rebuilt.sum(dtype=np.float64),
                "cached_probability_sum": cached.sum(dtype=np.float64),
                "max_abs_rebuild_delta": float(np.max(np.abs(rebuilt - cached))),
            }
        )
    ordering_audit = pd.DataFrame(ordering_rows)
    write_csv(root / "results/PILOT_ORDERING_AUDIT.csv", ordering_audit)
    ordering_passed = bool(
        len(ordering_audit) == expected
        and ordering_audit.fits_ordering.eq("NUNIQ").all()
        and ordering_audit.fits_index_scheme.eq("EXPLICIT").all()
        and np.allclose(ordering_audit.rebuilt_probability_sum, 1.0, atol=1e-12)
        and float(ordering_audit.max_abs_rebuild_delta.max()) <= 1e-10
    )
    summary = {
        "expected": expected,
        "completed": len(frame),
        "all_maps_valid": len(frame) == expected,
        "fallback_fraction": float(frame.fallback_used.mean()) if len(frame) else 1.0,
        "median_seconds": float(frame.total_seconds.median()) if len(frame) else math.nan,
        "p90_seconds": float(frame.total_seconds.quantile(0.9)) if len(frame) else math.nan,
        "median_area90_deg2": float(frame.area90_deg2.median()) if len(frame) else math.nan,
        "raw_hpd90_coverage": float((frame.truth_credible_level_raw <= 0.9).mean()) if len(frame) else math.nan,
        "ordering_and_rasterization_audit_passed": ordering_passed,
        "ordering_max_abs_rebuild_delta": float(ordering_audit.max_abs_rebuild_delta.max()) if len(ordering_audit) else math.nan,
        "passed_runtime_and_validity_gate": bool(
            len(frame) == expected
            and float(frame.fallback_used.mean()) <= 0.05
            and float(frame.total_seconds.quantile(0.9)) <= 90.0
            and ordering_passed
        ) if len(frame) else False,
        "coverage_role": "descriptive only at pilot size; full validation controls calibration",
    }
    write_csv(root / "results/PILOT_EVENT_METRICS.csv", frame)
    write_json(root / "contracts/PILOT_GATE.json", summary)
    if not summary["passed_runtime_and_validity_gate"]:
        raise RuntimeError(f"Pilot failed: {summary}")


def ensure_all_maps(root: Path, deployment: str, seed: int, split: str, workers: int) -> pd.DataFrame:
    frame = run_map_generation(root, deployment, seed, split, workers)
    expected = len(pd.read_parquet(events_path(deployment, seed, split)))
    if len(frame) != expected:
        raise RuntimeError(f"Incomplete maps for {deployment}/{seed}/{split}: {len(frame)}/{expected}")
    return frame


def event_weights(metrics: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(metrics), dtype=np.float64)
    weights[metrics.tag.isin(["L1", "L2"]).to_numpy()] = 0.5
    return weights


def weighted_cvm_uniform(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    x = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    w /= w.sum()
    empirical = np.cumsum(w) - 0.5 * w
    return float(np.sum(w * np.square(empirical - x)))


def select_temperature(metrics: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    weights = event_weights(metrics)
    rows = []
    for temperature in TEMPERATURE_GRID:
        column = f"truth_credible_T{temperature:g}"
        values = metrics[column].to_numpy(dtype=np.float64)
        cov50 = float(np.average(values <= 0.5, weights=weights))
        cov90 = float(np.average(values <= 0.9, weights=weights))
        cvm = weighted_cvm_uniform(values, weights)
        objective = cvm + (cov50 - 0.5) ** 2 + (cov90 - 0.9) ** 2
        rows.append(
            {
                "temperature": temperature,
                "weighted_coverage_50": cov50,
                "weighted_coverage_90": cov90,
                "weighted_cvm_uniform": cvm,
                "objective": objective,
            }
        )
    grid = pd.DataFrame(rows).sort_values(["objective", "temperature"], kind="stable")
    return float(grid.iloc[0].temperature), grid


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if math.isclose(float(temperature), 1.0):
        return p / p.sum(dtype=np.float64)
    logp = np.log(np.maximum(p, 1e-300)) / float(temperature)
    values = np.exp(logp - float(np.max(logp)))
    return values / values.sum(dtype=np.float64)


def load_retained_map_bank(
    root: Path,
    deployment: str,
    seed: int,
    split: str,
    retained: pd.DataFrame,
    temperature: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    maps = []
    metrics = pd.read_csv(root / "results" / deployment / f"seed_{seed}" / f"{split}_bayestar_event_metrics.csv")
    metric_lookup = metrics.set_index("idx")
    selected_rows = []
    for old_index in retained.old_idx.astype(int):
        path = map_paths(root, deployment, seed, split, old_index)["dense"]
        if not path.exists():
            raise FileNotFoundError(path)
        maps.append(apply_temperature(np.load(path, mmap_mode="r"), temperature).astype(np.float32))
        selected_rows.append(metric_lookup.loc[old_index])
    return np.stack(maps), pd.DataFrame(selected_rows).reset_index(drop=True)


def gpu_pair_features(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.from_numpy(np.asarray(bank, dtype=np.float32)).to(device)
    with torch.no_grad():
        overlap = tensor @ tensor.T
        bc = torch.sqrt(torch.clamp(tensor, min=0.0)) @ torch.sqrt(torch.clamp(tensor, min=0.0)).T
    overlap_np = overlap.cpu().numpy().astype(np.float64)
    bc_np = bc.cpu().numpy().astype(np.float64)
    ii, jj = np.triu_indices(len(bank), k=1)
    npix = bank.shape[1]
    log_bf = np.log(np.maximum(npix * overlap_np[ii, jj], 1e-300))
    bc_values = np.clip(bc_np[ii, jj], 0.0, 1.0)
    rng = np.random.default_rng(20260901 + len(bank))
    audit_indices = rng.choice(len(ii), size=min(128, len(ii)), replace=False)
    exact = np.asarray(
        [
            npix
            * np.sum(
                bank[int(ii[k])].astype(np.float64) * bank[int(jj[k])].astype(np.float64),
                dtype=np.float64,
            )
            for k in audit_indices
        ]
    )
    approx = npix * overlap_np[ii[audit_indices], jj[audit_indices]]
    audit = {
        "device": device,
        "n_events": len(bank),
        "n_pairs": len(ii),
        "seconds": time.perf_counter() - started,
        "float64_audit_pairs": len(audit_indices),
        "float64_max_abs_log_bf_delta": float(
            np.max(np.abs(np.log(np.maximum(exact, 1e-300)) - np.log(np.maximum(approx, 1e-300))))
        ),
    }
    del tensor, overlap, bc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log_bf, bc_values, audit


def pair_frame_for_split(
    root: Path,
    deployment: str,
    seed: int,
    split: str,
    temperature: float,
    allow_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retained = BASE.retained_event_plan(deployment, seed, split)
    frame = BASE.subset_pair_table(deployment, seed, split, retained, allow_test_scores=allow_test)
    bank, event_metrics = load_retained_map_bank(root, deployment, seed, split, retained, temperature)
    log_bf, bc, audit = gpu_pair_features(bank)
    if bank.shape[1] != hp.nside2npix(ANALYSIS_NSIDE):
        raise RuntimeError(f"Unexpected dense map size: {bank.shape}")
    # NESTED child pixels are contiguous. Summing each group of four gives the
    # exact Nside=256 probability mass represented by the Nside=512 raster.
    bank_coarse = bank.reshape(len(bank), hp.nside2npix(COARSE_NSIDE), 4).sum(axis=2)
    log_bf_coarse, _, coarse_audit = gpu_pair_features(bank_coarse)
    abs_delta = np.abs(log_bf - log_bf_coarse)
    true_mask = frame.is_true_pair.astype(bool).to_numpy()
    false_mask = ~true_mask
    false_threshold = np.quantile(log_bf[false_mask], 0.99) if false_mask.any() else np.inf
    tail_mask = false_mask & (log_bf >= false_threshold)
    native_moc_order_max = int(event_metrics.native_moc_order_max.max())
    reference_invariant = native_moc_order_max <= int(round(math.log2(ANALYSIS_NSIDE)))

    def convergence_record(mask: np.ndarray, label: str) -> dict[str, Any]:
        values = abs_delta[mask]
        sign_flips = (np.signbit(log_bf[mask]) != np.signbit(log_bf_coarse[mask]))
        if not len(values):
            return {"stratum": label, "n_pairs": 0}
        return {
            "stratum": label,
            "n_pairs": int(len(values)),
            "median_abs_delta_256_512": float(np.median(values)),
            "q90_abs_delta_256_512": float(np.quantile(values, 0.9)),
            "q99_abs_delta_256_512": float(np.quantile(values, 0.99)),
            "max_abs_delta_256_512": float(np.max(values)),
            "sign_flip_count_256_512": int(sign_flips.sum()),
            "sign_flip_fraction_256_512": float(sign_flips.mean()),
            "spearman_256_512": float(stats.spearmanr(log_bf_coarse[mask], log_bf[mask]).statistic),
            "native_moc_order_max": native_moc_order_max,
            "analytic_abs_delta_512_1024": 0.0 if reference_invariant else math.nan,
            "analytic_reason_512_1024": (
                "native MOC order does not exceed the Nside=512 analysis order; uniform NESTED "
                "subdivision to Nside=1024 preserves Npix*sum(P_i*P_j), including frozen temperature"
                if reference_invariant
                else "native MOC exceeds the analysis order; targeted numerical Nside=1024 audit required"
            ),
        }

    convergence = pd.DataFrame(
        [
            convergence_record(np.ones(len(frame), dtype=bool), "all_pairs"),
            convergence_record(true_mask, "true_companions"),
            convergence_record(false_mask, "non_companions"),
            convergence_record(tail_mask, "non_companion_positive_tail_1pct"),
        ]
    )
    frame["sky_raw_log_bf"] = log_bf
    frame["sky_bayes_factor"] = np.exp(np.clip(log_bf, -700, 700))
    frame["sky_bc"] = bc
    fallback = event_metrics.fallback_used.fillna(False).astype(bool).to_numpy()
    ii = frame.idx_i.to_numpy(dtype=np.int64)
    jj = frame.idx_j.to_numpy(dtype=np.int64)
    frame["sky_bayestar_ood_pair"] = fallback[ii] | fallback[jj]
    frame["sky_method"] = "BAYESTAR_gaussian_trigger_event_specific_PSD"
    frame["posterior_temperature"] = float(temperature)
    output = root / "results" / deployment / f"seed_{seed}"
    frame.to_parquet(output / f"{split}_pair_scores_bayestar_sky.parquet", index=False)
    write_csv(output / f"{split}_sky_resolution_convergence.csv", convergence)
    write_json(
        output / f"{split}_pair_overlap_precision_audit.json",
        {"nside512": audit, "nside256": coarse_audit},
    )
    del bank_coarse
    del bank
    gc.collect()
    return frame, retained, audit


def metric_record(
    deployment: str,
    seed: int,
    split: str,
    method: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    include_bootstrap: bool,
) -> dict[str, Any]:
    result = {
        "deployment": deployment,
        "seed": seed,
        "split": split,
        "method": method,
        "n_events": int(frame.event_count.iloc[0]),
        "n_pairs": len(frame),
        "n_true_pairs": int(frame.is_true_pair.sum()),
        "n_false_pairs": int((~frame.is_true_pair.astype(bool)).sum()),
        "pair_ood_fraction": float(frame.sky_bayestar_ood_pair.mean()),
        **BASE.full_metrics(frame, scores),
    }
    if include_bootstrap:
        result.update(BASE.system_bootstrap_ci(frame, scores, stable_seed("bootstrap", deployment, seed, method)))
    return result


def validation(root: Path, workers: int) -> None:
    if not (root / "contracts/PILOT_GATE.json").exists():
        pilot(root, workers, 10)
    metric_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {
        "schema": "bayestar-validation-selected-config-v1",
        "created_utc": utc_stamp(),
        "test_opened": False,
        "real_catalog_opened": False,
        "deployments": {},
    }
    for deployment in DEPLOYMENTS:
        selected["deployments"][deployment] = {}
        for seed in MODEL_SEEDS:
            event_metrics = ensure_all_maps(root, deployment, seed, "validation", workers)
            retained_validation = BASE.retained_event_plan(deployment, seed, "validation")
            calibration_indices = set(retained_validation.old_idx.astype(int))
            temperature_metrics = event_metrics.loc[
                event_metrics.idx.astype(int).isin(calibration_indices)
            ].copy()
            if len(temperature_metrics) != len(retained_validation):
                raise RuntimeError(
                    f"Temperature-calibration event mismatch for {deployment}/{seed}: "
                    f"{len(temperature_metrics)}/{len(retained_validation)}"
                )
            temperature, temp_grid = select_temperature(temperature_metrics)
            output = root / "results" / deployment / f"seed_{seed}"
            write_csv(output / "validation_temperature_grid.csv", temp_grid)
            write_csv(
                output / "validation_temperature_calibration_events.csv",
                retained_validation,
            )
            frame, _, _ = pair_frame_for_split(root, deployment, seed, "validation", temperature, False)
            frozen = BASE.FROZEN_V93_WEIGHTS[deployment][seed]
            selection, weight_grid = BASE.select_retuned_weights(frame, frozen)
            write_csv(output / "validation_weight_grid.csv", weight_grid)
            selected["deployments"][deployment][str(seed)] = {
                "posterior_temperature": temperature,
                "temperature_selection": temp_grid.iloc[0].to_dict(),
                "temperature_calibration_event_count": len(temperature_metrics),
                "temperature_calibration_scope": "source/noise-block-disjoint retained validation only",
                "C_fixed_raw_weights": frozen,
                "C_retuned": selection,
            }
            methods = {
                "waveform_only": frame.waveform_score.to_numpy(float),
                "time_only": frame.time_score.to_numpy(float),
                "sky_bayestar_only": frame.sky_raw_log_bf.to_numpy(float),
                "BAYESTAR_C_fixed": BASE.score_vector(frame, frozen),
                "BAYESTAR_C_retuned": BASE.score_vector(frame, selection["weights"]),
            }
            for method, scores in methods.items():
                metric_rows.append(metric_record(deployment, seed, "validation", method, frame, scores, False))
    selected_path = root / "contracts/selected_config.json"
    write_json(selected_path, selected)
    selected_hash = sha256_file(selected_path)
    (root / "contracts/selected_config.sha256").write_text(
        f"{selected_hash}  selected_config.json\n", encoding="utf-8"
    )
    write_csv(root / "results/validation_retrieval_metrics_per_seed.csv", pd.DataFrame(metric_rows))
    write_json(
        root / "contracts/VALIDATION_COMPLETE_TEST_STILL_BLINDED.json",
        {
            "completed": True,
            "timestamp_utc": utc_stamp(),
            "test_pair_scores_opened": False,
            "selected_config_sha256": selected_hash,
        },
    )


def t_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "overall_r_at_1",
        "overall_r_at_5",
        "overall_r_at_10",
        "median_rank",
        "roc_auc",
        "average_precision",
        "false_at_recall_0p5",
        "false_at_recall_0p9",
        "top_10_precision",
        "top_50_precision",
        "top_100_precision",
        "top_200_precision",
    ]
    rows = []
    for (deployment, method), part in frame.groupby(["deployment", "method"]):
        row = {"deployment": deployment, "method": method, "n_seeds": len(part)}
        for key in keys:
            if key not in part:
                continue
            values = part[key].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
            row[f"{key}_ci95_low"] = mean - half
            row[f"{key}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def test(root: Path, workers: int) -> None:
    selected_path = root / "contracts/selected_config.json"
    if not selected_path.exists():
        raise RuntimeError("Validation-selected config is missing")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    frozen_hash = (root / "contracts/selected_config.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(selected_path) != frozen_hash:
        raise RuntimeError("Validation-selected config changed before locked test")
    if (root / "contracts/LOCKED_TEST_COMPLETE.json").exists():
        raise RuntimeError("Locked test has already been opened")
    metric_rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        for seed in MODEL_SEEDS:
            ensure_all_maps(root, deployment, seed, "test", workers)
            config = selected["deployments"][deployment][str(seed)]
            frame, _, _ = pair_frame_for_split(
                root,
                deployment,
                seed,
                "test",
                float(config["posterior_temperature"]),
                True,
            )
            methods = {
                "waveform_only": frame.waveform_score.to_numpy(float),
                "time_only": frame.time_score.to_numpy(float),
                "sky_bayestar_only": frame.sky_raw_log_bf.to_numpy(float),
                "BAYESTAR_C_fixed": BASE.score_vector(frame, config["C_fixed_raw_weights"]),
                "BAYESTAR_C_retuned": BASE.score_vector(frame, config["C_retuned"]["weights"]),
            }
            for method, scores in methods.items():
                metric_rows.append(metric_record(deployment, seed, "test", method, frame, scores, True))
            scored = frame[[column for column in frame.columns]].copy()
            scored["score_C_fixed"] = methods["BAYESTAR_C_fixed"]
            scored["score_C_retuned"] = methods["BAYESTAR_C_retuned"]
            scored.to_parquet(
                root / "results" / deployment / f"seed_{seed}" / "locked_test_pair_scores_bayestar_C_fixed_C_retuned.parquet",
                index=False,
            )
    metrics = pd.DataFrame(metric_rows)
    write_csv(root / "results/locked_test_retrieval_metrics_per_seed.csv", metrics)
    summary = t_summary(metrics)
    write_csv(root / "results/locked_test_retrieval_metrics_summary.csv", summary)
    gate_input = metrics.copy()
    gate_input["method"] = gate_input.method.replace(
        {"BAYESTAR_C_fixed": "C_fixed", "BAYESTAR_C_retuned": "C_retuned"}
    )
    gate_per_seed, gate_summary = BASE.test_gate(gate_input)
    write_csv(root / "results/C_retuned_vs_C_fixed_paired_gate_per_seed.csv", gate_per_seed)
    write_json(root / "results/C_retuned_vs_C_fixed_gate_summary.json", gate_summary)
    write_json(
        root / "contracts/LOCKED_TEST_COMPLETE.json",
        {
            "completed": True,
            "timestamp_utc": utc_stamp(),
            "test_opened_once": True,
            "real_catalog_opened": False,
        },
    )


def real_catalog(root: Path) -> None:
    selected_path = root / "contracts/selected_config.json"
    if not (root / "contracts/LOCKED_TEST_COMPLETE.json").exists():
        raise RuntimeError("Real catalog remains blocked until locked test completes")
    if (root / "contracts/REAL_CATALOG_COMPLETE.json").exists():
        raise RuntimeError("Real catalog has already been opened")
    # The vetted C-scheme helper uses corrected public PE maps and the frozen
    # waveform/time pair tables.  Its selected-config schema is intentionally
    # preserved above, so this call cannot retune on real candidates.
    BASE.phase3_real_catalog(root)
    write_json(
        root / "contracts/REAL_CATALOG_COMPLETE.json",
        {
            "completed": True,
            "timestamp_utc": utc_stamp(),
            "selected_config_sha256": sha256_file(selected_path),
            "real_candidate_information_used_for_selection": False,
        },
    )


def map_manifest(root: Path) -> pd.DataFrame:
    selected_path = root / "contracts/selected_config.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else None
    rows = []
    for path in sorted((root / "event_maps").rglob("event_*.json")):
        if path.name.endswith("failure.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = {
                key: payload.get(key)
                for key in (
                    "deployment",
                    "model_seed",
                    "split",
                    "idx",
                    "family",
                    "tag",
                    "source_index",
                    "gps_obs",
                    "target_network_snr",
                    "realized_network_snr",
                    "area50_deg2",
                    "area90_deg2",
                    "truth_credible_level_raw",
                    "fallback_used",
                    "waveform",
                    "total_seconds",
                    "moc_path",
                    "moc_bytes",
                    "moc_sha256",
                    "native_moc_order_min",
                    "native_moc_order_max",
                    "native_moc_nside_max",
                )
            }
        if selected is not None:
            temperature = float(
                selected["deployments"][str(payload["deployment"])][str(payload["model_seed"])][
                    "posterior_temperature"
                ]
            )
            key = f"truth_credible_T{temperature:g}"
            record["posterior_temperature"] = temperature
            record["truth_credible_level_calibrated"] = payload.get(key)
        rows.append(record)
    return pd.DataFrame(rows)


def clean_dense_cache(root: Path) -> dict[str, Any]:
    dense_root = root / "work/dense_nside512"
    prior_path = root / "results/dense_cache_cleanup.json"
    if not dense_root.exists() and prior_path.exists():
        return json.loads(prior_path.read_text(encoding="utf-8"))
    bytes_before = sum(path.stat().st_size for path in dense_root.rglob("*.npy")) if dense_root.exists() else 0
    if dense_root.exists():
        shutil.rmtree(dense_root)
    return {"dense_cache_deleted": True, "bytes_removed": bytes_before}


def extended_audits(root: Path, manifest: pd.DataFrame) -> None:
    map_rows: list[dict[str, Any]] = []
    for (deployment, split), part in manifest.groupby(["deployment", "split"]):
        map_rows.append(
            {
                "deployment": deployment,
                "split": split,
                "n_events": len(part),
                "fallback_fraction": float(part.fallback_used.mean()),
                "raw_hpd50_coverage": float((part.truth_credible_level_raw <= 0.5).mean()),
                "raw_hpd90_coverage": float((part.truth_credible_level_raw <= 0.9).mean()),
                "calibrated_hpd50_coverage": float(
                    (part.truth_credible_level_calibrated <= 0.5).mean()
                ),
                "calibrated_hpd90_coverage": float(
                    (part.truth_credible_level_calibrated <= 0.9).mean()
                ),
                "area90_median_deg2": float(part.area90_deg2.median()),
                "area90_q90_deg2": float(part.area90_deg2.quantile(0.9)),
                "runtime_median_seconds": float(part.total_seconds.median()),
                "runtime_q90_seconds": float(part.total_seconds.quantile(0.9)),
                "realized_network_snr_median": float(part.realized_network_snr.median()),
                "native_moc_order_max": int(part.native_moc_order_max.max()),
            }
        )
    write_csv(root / "results/bayestar_map_coverage_runtime_summary.csv", pd.DataFrame(map_rows))

    snr_frame = manifest.copy()
    snr_frame["snr_bin"] = pd.cut(
        snr_frame.realized_network_snr,
        bins=[-np.inf, 10.0, 12.0, 20.0, np.inf],
        labels=["<10", "10-12", "12-20", ">20"],
        right=False,
    )
    snr_rows: list[dict[str, Any]] = []
    for (deployment, split, snr_bin), part in snr_frame.groupby(
        ["deployment", "split", "snr_bin"], observed=True
    ):
        snr_rows.append(
            {
                "deployment": deployment,
                "split": split,
                "snr_bin": str(snr_bin),
                "n_events": len(part),
                "calibrated_hpd90_coverage": float(
                    (part.truth_credible_level_calibrated <= 0.9).mean()
                ),
                "area90_median_deg2": float(part.area90_deg2.median()),
                "area90_q90_deg2": float(part.area90_deg2.quantile(0.9)),
            }
        )
    write_csv(root / "results/bayestar_map_audit_by_snr.csv", pd.DataFrame(snr_rows))

    score_rows: list[dict[str, Any]] = []
    convergence_frames: list[pd.DataFrame] = []
    for deployment in DEPLOYMENTS:
        for seed in MODEL_SEEDS:
            seed_root = root / "results" / deployment / f"seed_{seed}"
            pair_frame = pd.read_parquet(seed_root / "test_pair_scores_bayestar_sky.parquet")
            for label, part in pair_frame.groupby(pair_frame.is_true_pair.map({True: "companion", False: "non_companion"})):
                values = part.sky_raw_log_bf.to_numpy(dtype=np.float64)
                score_rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "class": label,
                        "n_pairs": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "sd": float(np.std(values, ddof=1)),
                        "q90": float(np.quantile(values, 0.9)),
                        "q95": float(np.quantile(values, 0.95)),
                        "q99": float(np.quantile(values, 0.99)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
            for split in SPLITS:
                convergence = pd.read_csv(seed_root / f"{split}_sky_resolution_convergence.csv")
                convergence.insert(0, "split", split)
                convergence.insert(0, "seed", seed)
                convergence.insert(0, "deployment", deployment)
                convergence_frames.append(convergence)
    write_csv(root / "results/locked_test_sky_score_distribution.csv", pd.DataFrame(score_rows))
    write_csv(
        root / "results/sky_resolution_convergence_all_seeds.csv",
        pd.concat(convergence_frames, ignore_index=True),
    )

    historical_path = ROTATED_TEMPLATE_C_ROOT / "results/retrieval_pair_metrics_per_seed.csv"
    if historical_path.exists():
        historical = pd.read_csv(historical_path)
        historical = historical.loc[historical.split.eq("test")].copy()
        current = pd.read_csv(root / "results/locked_test_retrieval_metrics_per_seed.csv").copy()
        historical.method = historical.method.replace(
            {"C_fixed": "C_fixed", "C_retuned": "C_retuned", "sky_only": "sky_only"}
        )
        current.method = current.method.replace(
            {
                "BAYESTAR_C_fixed": "C_fixed",
                "BAYESTAR_C_retuned": "C_retuned",
                "sky_bayestar_only": "sky_only",
            }
        )
        metrics = [
            "overall_r_at_1",
            "overall_r_at_10",
            "average_precision",
            "false_at_recall_0p5",
            "false_at_recall_0p9",
        ]
        old_summary = historical.groupby(["deployment", "method"])[metrics].agg(["mean", "std"])
        new_summary = current.groupby(["deployment", "method"])[metrics].agg(["mean", "std"])
        old_summary.columns = [f"rotated_template_{a}_{b}" for a, b in old_summary.columns]
        new_summary.columns = [f"bayestar_{a}_{b}" for a, b in new_summary.columns]
        comparison = old_summary.join(new_summary, how="inner").reset_index()
        for metric in metrics:
            comparison[f"delta_bayestar_minus_rotated_{metric}"] = (
                comparison[f"bayestar_{metric}_mean"]
                - comparison[f"rotated_template_{metric}_mean"]
            )
        write_csv(
            root / "results/BAYESTAR_vs_rotated_template_locked_test_comparison.csv",
            comparison,
        )


def make_figures(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
        }
    )
    metrics = pd.read_csv(root / "results/locked_test_retrieval_metrics_per_seed.csv")
    manifest = pd.read_csv(root / "results/bayestar_event_map_manifest.csv")
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    colors = {
        "waveform_only": "#35608D",
        "time_only": "#D67C37",
        "sky_bayestar_only": "#4A8F5E",
        "BAYESTAR_C_fixed": "#7A5195",
        "BAYESTAR_C_retuned": "#C43C39",
    }
    methods = list(colors)
    for col, deployment in enumerate(DEPLOYMENTS):
        ax = axes[0, col]
        part = metrics[metrics.deployment.eq(deployment)]
        x = np.arange(len(methods))
        r1 = [part[part.method.eq(method)].overall_r_at_1.mean() for method in methods]
        r10 = [part[part.method.eq(method)].overall_r_at_10.mean() for method in methods]
        ax.bar(x - 0.18, r1, 0.36, color=[colors[m] for m in methods], alpha=0.65, label="R@1")
        ax.bar(x + 0.18, r10, 0.36, color=[colors[m] for m in methods], hatch="//", alpha=0.85, label="R@10")
        for offset, metric in ((-0.18, "overall_r_at_1"), (0.18, "overall_r_at_10")):
            for seed_number, method in enumerate(methods):
                values = part[part.method.eq(method)][metric].to_numpy(float)
                ax.scatter(np.full(len(values), seed_number + offset), values, s=12, color="black", zorder=4)
        ax.set_xticks(x, ["wf", "time", "sky", "fixed", "retuned"], rotation=20)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Companion retrieval")
        ax.set_title(f"{chr(97 + col)}) {deployment.upper()} locked test", loc="left")
        if col == 1:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax = axes[1, 0]
    for deployment, color in zip(DEPLOYMENTS, ("#35608D", "#C43C39")):
        part = manifest[(manifest.deployment.eq(deployment)) & (manifest.split.eq("test"))]
        values = np.sort(part.truth_credible_level_raw.to_numpy(float))
        empirical = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, empirical, color=color, lw=1.5, label=deployment.upper())
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.9)
    ax.set_xlabel("True-sky credible level")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("c) BAYESTAR coverage", loc="left")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    bins = np.logspace(0, 4.7, 35)
    for deployment, color in zip(DEPLOYMENTS, ("#35608D", "#C43C39")):
        part = manifest[(manifest.deployment.eq(deployment)) & (manifest.split.eq("test"))]
        ax.hist(part.area90_deg2, bins=bins, histtype="step", density=True, lw=1.5, color=color, label=deployment.upper())
    ax.set_xscale("log")
    ax.set_xlabel(r"BAYESTAR $A_{90}$ (deg$^2$)")
    ax.set_ylabel("Density")
    ax.set_title("d) Injection sky-map width", loc="left")
    ax.legend(frameon=False)
    png = root / "figures/fig_bayestar_injection_sky_full_experiment.png"
    pdf = root / "figures/fig_bayestar_injection_sky_full_experiment.pdf"
    figure.savefig(png, dpi=300)
    figure.savefig(pdf)
    plt.close(figure)


def report(root: Path, cleanup: dict[str, Any]) -> None:
    metrics = pd.read_csv(root / "results/locked_test_retrieval_metrics_summary.csv")
    per_seed = pd.read_csv(root / "results/locked_test_retrieval_metrics_per_seed.csv")
    manifest = pd.read_csv(root / "results/bayestar_event_map_manifest.csv")
    selected = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "results/C_retuned_vs_C_fixed_gate_summary.json").read_text(encoding="utf-8"))
    budget_path = root / "results/real_PE_official_budget_summary.csv"
    budget = pd.read_csv(budget_path) if budget_path.exists() else pd.DataFrame()
    map_summary = pd.read_csv(root / "results/bayestar_map_coverage_runtime_summary.csv")
    comparison_path = root / "results/BAYESTAR_vs_rotated_template_locked_test_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    lines = [
        "# O3/O4a BAYESTAR 注入天空后验完整实验报告",
        "",
        "**最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`**",
        "",
        "> 本轮用事件级 BAYESTAR 快速天空 PE 替换了注入侧旋转公开 PE 模板；历史 v9.3/v9.4、encoder、波形分数、时间分数和论文均未覆盖。结果是快速条件天空 PE，不是完整 BBH PE，也不是透镜探测。",
        "",
        "## 方法",
        "",
        f"- 共生成并保存 {len(manifest)} 张独立 BAYESTAR MOC 天空图；正式重排分辨率为 Nside={ANALYSIS_NSIDE}。",
        "- 每张图使用该注入的 GW-LMC 源参数、GPS、H1/L1 响应、对应真实 off-source PSD 和目标 network SNR。",
        "- 匹配滤波测量噪声由 `bayestar-realize-coincs` 的官方 Gaussian realization 生成；没有读取或旋转任何真实 PE 模板。",
        f"- 快速定位模板为 {PRIMARY_WAVEFORM}；上游物理注入仍是 IMRPhenomXPHM。",
        "- validation-only 选择 posterior temperature 和三通道权重；locked test 与真实候选未参与调参。",
        "- 重要限制：快速触发层使用事件对应的真实 PSD，但没有回放该 26 s 窗口中的非高斯时域噪声。",
        "",
        "## 地图审计",
        "",
        f"- 地图成功率：{len(manifest)}/{int(input_inventory().n_events.sum())}。",
        f"- waveform fallback 比例：{manifest.fallback_used.mean():.4f}。",
        f"- 单图 wall time 中位数/P90：{manifest.total_seconds.median():.2f}/{manifest.total_seconds.quantile(0.9):.2f} s。",
        f"- test 真天空 raw 90% HPD 覆盖率：{(manifest[manifest.split.eq('test')].truth_credible_level_raw <= 0.9).mean():.3f}。",
        f"- test 真天空 validation-calibrated 90% HPD 覆盖率：{(manifest[manifest.split.eq('test')].truth_credible_level_calibrated <= 0.9).mean():.3f}。",
        "",
        map_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Validation 冻结配置",
        "",
    ]
    for deployment in DEPLOYMENTS:
        lines.append(f"### {deployment.upper()}")
        lines.append("")
        for seed in MODEL_SEEDS:
            cfg = selected["deployments"][deployment][str(seed)]
            w = cfg["C_retuned"]["weights"]
            lines.append(
                f"- seed {seed}: T={cfg['posterior_temperature']:.2f}; C-retuned wf/time/sky={w['waveform']:.2f}/{w['time']:.2f}/{w['sky']:.2f}."
            )
        lines.append("")
    lines.extend(["## Locked-test 指标", ""])
    view_columns = [
        "deployment",
        "method",
        "overall_r_at_1_mean",
        "overall_r_at_10_mean",
        "average_precision_mean",
        "false_at_recall_0p5_mean",
        "false_at_recall_0p9_mean",
    ]
    lines.append(metrics[[c for c in view_columns if c in metrics]].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "逐 seed 数值、bootstrap 95% CI、Top-B precision 和全部 pair 表见 `results/`。",
            "",
            f"C-retuned 相对 C-fixed 的预注册 paired gate：**{gate['decision']}**。",
            "",
        ]
    )
    if not comparison.empty:
        comparison_view = comparison.loc[comparison.method.isin(["C_fixed", "sky_only"])]
        columns = [
            "deployment",
            "method",
            "rotated_template_overall_r_at_1_mean",
            "bayestar_overall_r_at_1_mean",
            "rotated_template_overall_r_at_10_mean",
            "bayestar_overall_r_at_10_mean",
            "rotated_template_average_precision_mean",
            "bayestar_average_precision_mean",
            "delta_bayestar_minus_rotated_overall_r_at_10",
        ]
        lines.extend(
            [
                "## 与旋转公开 PE 模板方案的成对对照",
                "",
                comparison_view[columns].to_markdown(index=False, floatfmt=".4f"),
                "",
                "该表比较相同 frozen waveform/time、相同 retained test 的天空生成方案变化；正负变化都完整保留。",
                "",
            ]
        )
    if not budget.empty:
        lines.extend(["## 冻结后的真实目录 PE/公开阶段审计", "", budget.to_markdown(index=False, floatfmt=".3f"), ""])
    lines.extend(
        [
            "## 结论边界",
            "",
            "- 该实验回答的是：注入自身经过快速网络定位后，天空通道和融合排序如何变化。",
            "- 它消除了旋转模板带来的直接形态复用，但保留 BAYESTAR 的固定内禀参数、Gaussian trigger-noise 和近似 waveform 限制。",
            "- 是否优于 C-fixed 必须同时查看 O3/O4a 的 R@10、AUPRC 和固定召回假对负担；不能只挑一个较好的数字。",
            "- 真实目录仍使用公开完整 PE posterior，真实候选 PE/官方重合只作冻结后审计。",
            "- 本轮结果未经作者审核，不替代任何历史主结果。",
            "",
            f"Dense Nside=512 临时缓存已删除：{cleanup['dense_cache_deleted']}，释放 {cleanup['bytes_removed'] / 2**30:.2f} GiB；完整 MOC 图仍保留。",
            "",
            "`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`",
        ]
    )
    (root / "reports/BAYESTAR_INJECTION_SKY_FULL_EXPERIMENT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_outputs(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "/work/" in str(path) or path.name == "SHA256SUMS.txt":
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def finalize(root: Path) -> None:
    maps = map_manifest(root)
    write_csv(root / "results/bayestar_event_map_manifest.csv", maps)
    expected = int(input_inventory().n_events.sum())
    if len(maps) != expected:
        raise RuntimeError(f"Map manifest incomplete: {len(maps)}/{expected}")
    extended_audits(root, maps)
    make_figures(root)
    cleanup = clean_dense_cache(root)
    write_json(root / "results/dense_cache_cleanup.json", cleanup)
    report(root, cleanup)
    before = pd.read_csv(root / "contracts/HISTORICAL_INPUT_HASHES_BEFORE.csv")
    after = BASE.snapshot_hashes(BASE.critical_historical_paths())
    merged = before.merge(after, on="path", suffixes=("_before", "_after"), how="outer")
    merged["unchanged"] = (
        merged.sha256_before.eq(merged.sha256_after)
        & merged.size_bytes_before.eq(merged.size_bytes_after)
        & merged.sha256_before.notna()
        & merged.sha256_after.notna()
    )
    write_csv(root / "contracts/HISTORICAL_INPUT_HASHES_AFTER.csv", after)
    write_csv(root / "contracts/HISTORICAL_IMMUTABILITY_AUDIT.csv", merged)
    if not bool(merged.unchanged.fillna(False).all()):
        raise RuntimeError("Historical immutability audit failed")
    package = PROJECT / "packages" / f"{root.name}_deliverables.tar.gz"
    map_package = PROJECT / "packages" / f"{root.name}_event_moc_maps.tar"
    write_json(
        root / "contracts/FINAL_STATUS.json",
        {
            "status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
            "completed_utc": utc_stamp(),
            "map_count": len(maps),
            "package": str(package),
            "package_sha256_location": str(package.with_suffix(package.suffix + ".sha256")),
            "event_map_package": str(map_package),
            "event_map_package_sha256_location": str(map_package.with_suffix(map_package.suffix + ".sha256")),
            "historical_outputs_unchanged": True,
        },
    )
    output_manifest = manifest_outputs(root)
    write_csv(root / "manifest/OUTPUT_SHA256_MANIFEST.csv", output_manifest)
    sums = "\n".join(f"{row.sha256}  {row.path}" for row in output_manifest.itertuples(index=False)) + "\n"
    (root / "manifest/SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        for name in ("contracts", "results", "reports", "figures", "scripts", "manifest"):
            # Event MOC maps are intentionally excluded from the compact package;
            # their complete checksummed manifest and on-server paths are included.
            archive.add(root / name, arcname=f"{root.name}/{name}")
    (package.with_suffix(package.suffix + ".sha256")).write_text(
        f"{sha256_file(package)}  {package.name}\n", encoding="utf-8"
    )
    # FITS MOC files are already gzip-compressed.  An uncompressed tar avoids
    # spending hours recompressing ~5,400 independently compressed maps while
    # still yielding one checksummed transfer artifact.
    with tarfile.open(map_package, "w") as archive:
        archive.add(root / "event_maps", arcname=f"{root.name}/event_maps")
        archive.add(
            root / "results/bayestar_event_map_manifest.csv",
            arcname=f"{root.name}/results/bayestar_event_map_manifest.csv",
        )
    (map_package.with_suffix(map_package.suffix + ".sha256")).write_text(
        f"{sha256_file(map_package)}  {map_package.name}\n", encoding="utf-8"
    )
    print(json.dumps(json.loads((root / "contracts/FINAL_STATUS.json").read_text()), indent=2))


def run_all(root: Path, workers: int) -> None:
    prepare(root)
    pilot(root, workers, 10)
    validation(root, workers)
    test(root, workers)
    real_catalog(root)
    finalize(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "pilot", "validation", "test", "real", "finalize", "all"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(64, max(1, (os.cpu_count() or 8) - 16)))
    parser.add_argument("--pilot-events-per-seed", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    if args.command == "prepare":
        prepare(root)
    elif args.command == "pilot":
        pilot(root, args.workers, args.pilot_events_per_seed)
    elif args.command == "validation":
        validation(root, args.workers)
    elif args.command == "test":
        test(root, args.workers)
    elif args.command == "real":
        real_catalog(root)
    elif args.command == "finalize":
        finalize(root)
    elif args.command == "all":
        run_all(root, args.workers)


if __name__ == "__main__":
    main()
