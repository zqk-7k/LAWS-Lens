#!/usr/bin/env python3
"""Run the v10.1 homogeneous single-event sky-PE resource pilot.

This script deliberately does not touch historical v9.3/v10 results.  It
reads public GWOSC strain directly, estimates a same-file off-source PSD,
runs Bilby PE, builds a native multi-order sky map, and rasterizes that map
transiently at Nside 256/512/1024.

The pilot is a resource and basic-quality measurement.  It is not a
confirmation test and its outputs must not be used to tune real candidates.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bilby
import h5py
import healpy as hp
import numpy as np
import pandas as pd
import psutil
from scipy import signal, sparse
from scipy.sparse import csgraph


PROJECT = Path("/root/autodl-tmp/gw-catalog")
DEFAULT_CONFIG = PROJECT / "contracts" / "v10p1" / "V10P1_PE_PILOT_CONFIG.json"
DEFAULT_G05 = (
    PROJECT
    / "results"
    / "sky_background_exploratory_v10_nside512_20260823_20260822T171804Z"
)
PYTHON = Path(sys.executable)
SECONDS_PER_DAY = 86400.0
GWLMC_ROOT = Path("/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1")
OFFSOURCE_INVENTORY = DEFAULT_G05 / "tables/G0_OFFSOURCE_NOISE_BLOCK_INVENTORY.csv"
SOURCE_BANK_ROOT = PROJECT / "results/real_noise_injection_v5_physical_source_20260721"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_detector_string(value: Any) -> list[str]:
    text = str(value).replace(",", " ")
    return [item for item in (part.strip() for part in text.split()) if item in {"H1", "L1", "V1"}]


def snr_bin(value: float) -> str:
    if value < 12:
        return "low_8_to_12"
    if value <= 20:
        return "mid_12_to_20"
    return "high_gt_20"


def duration_for_chirp_mass(chirp_mass: float, config: dict[str, Any]) -> float:
    rule = config["pe_pipeline"]["duration_rule"]
    if chirp_mass >= 20:
        return float(rule["chirp_mass_ge_20"])
    if chirp_mass >= 10:
        return float(rule["chirp_mass_ge_10"])
    if chirp_mass >= 6:
        return float(rule["chirp_mass_ge_6"])
    return float(rule["otherwise"])


def existing_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT / value
    if not value.exists():
        raise FileNotFoundError(value)
    return value.resolve()


def existing_manifest_path(path: str | Path, manifest_path: Path) -> Path:
    """Resolve paths relative to either the project or their source manifest.

    The historical O3 download manifest stores paths relative to its run
    directory, whereas newer manifests use project-relative paths.
    """

    value = Path(path)
    candidates = [value] if value.is_absolute() else [PROJECT / value, manifest_path.parent.parent / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {path!s}; tried {candidates}")


def strain_rows(deployment: str) -> pd.DataFrame:
    if deployment == "O3":
        path = PROJECT / "runs/real_gwtc_lensing_search_20260625/data/strain_gwosc_download_manifest.csv"
    elif deployment == "O4a":
        path = PROJECT / "runs/gwtc4p1_data_completion_20260628/data/strain_manifest_gwtc4p1.csv"
    else:
        raise ValueError(deployment)
    frame = pd.read_csv(path)
    if deployment == "O3":
        frame = frame.rename(columns={"local_path": "strain_path"})
    else:
        frame = frame.rename(columns={"local_path": "strain_path"})
    frame = frame[frame.strain_path.notna()].copy()
    frame["strain_path"] = [
        str(existing_manifest_path(value, path)) for value in frame.strain_path
    ]
    if deployment == "O3":
        auxiliary = PROJECT / "data/v10p1_pilot_auxiliary_strain/manifest.csv"
        if auxiliary.exists():
            extra = pd.read_csv(auxiliary)
            extra = extra[extra.strain_path.notna()].copy()
            extra["strain_path"] = [str(existing_manifest_path(value, auxiliary)) for value in extra.strain_path]
            frame = pd.concat([frame, extra[frame.columns.intersection(extra.columns)]], ignore_index=True)
    return frame


def real_manifest(deployment: str) -> pd.DataFrame:
    if deployment == "O3":
        path = PROJECT / "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv"
        frame = pd.read_csv(path)
        frame = frame[frame.include_in_primary_search.fillna(False).astype(bool)].copy()
        frame["pe_path"] = frame.sky_map_path
        frame["pe_group"] = frame.sky_map_internal_group
        frame["object_class"] = "BBH"
    elif deployment == "O4a":
        path = PROJECT / "runs/real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest_gwtc4.csv"
        frame = pd.read_csv(path)
        frame = frame[frame.include_in_primary_search.fillna(False).astype(bool)].copy()
        frame["pe_path"] = frame.sky_map_path
        frame["pe_group"] = frame.sky_map_group
        frame = frame[frame.object_class.eq("BBH")].copy()
    else:
        raise ValueError(deployment)
    frame["pe_path"] = [str(existing_path(path)) for path in frame.pe_path]
    return frame


def map_audit_table() -> pd.DataFrame:
    path = DEFAULT_G05 / "tables/G05_NSIDE512_EVENT_PILOT.csv"
    frame = pd.read_csv(path)
    frame["deployment"] = frame.deployment.map({"GWTC3_O3": "O3", "GWTC4P1_O4A": "O4a"})
    return frame


def select_real_cases(config: dict[str, Any]) -> pd.DataFrame:
    """Select six deterministic real cases per deployment.

    Selection uses catalog SNR and pre-existing public-map morphology only for
    pilot coverage.  These cases are never used to tune ranking statistics.
    """

    audit = map_audit_table()
    rows: list[dict[str, Any]] = []
    targets = {
        "low_8_to_12": 10.0,
        "mid_12_to_20": 16.0,
        "high_gt_20": 26.0,
    }
    for deployment in ("O3", "O4a"):
        events = real_manifest(deployment)
        strain = strain_rows(deployment)
        available = (
            strain.groupby(["event_name", "detector"]).size().unstack(fill_value=0) > 0
        )
        events = events[events.event_name.isin(available.index)].copy()
        events["available_detectors"] = [
            ",".join([det for det in ("H1", "L1", "V1") if det in available.columns and bool(available.loc[name, det])])
            for name in events.event_name
        ]
        events = events[
            events.available_detectors.str.contains("H1")
            & events.available_detectors.str.contains("L1")
        ].copy()
        events = events.merge(
            audit[audit.deployment.eq(deployment)][["event_name", "a90_deg2", "kl_from_uniform_nats"]],
            on="event_name",
            how="left",
        )
        events["snr_bin"] = [snr_bin(float(x)) for x in events.network_snr]
        selected: list[tuple[str, pd.Series]] = []
        used: set[str] = set()

        for label, target in targets.items():
            subset = events[events.snr_bin.eq(label)].copy()
            if subset.empty:
                continue
            subset["distance"] = np.abs(subset.network_snr.astype(float) - target)
            row = subset.sort_values(["distance", "event_name"]).iloc[0]
            selected.append((f"snr_{label}", row))
            used.add(str(row.event_name))

        for label, ascending in (("narrow_public_map", True), ("wide_public_map", False)):
            subset = events[~events.event_name.astype(str).isin(used) & events.a90_deg2.notna()]
            if subset.empty:
                continue
            row = subset.sort_values(["a90_deg2", "event_name"], ascending=[ascending, True]).iloc[0]
            selected.append((label, row))
            used.add(str(row.event_name))

        # For O3, explicitly include a public HLV event when its V1 file has
        # been supplied.  Otherwise this slot is retained as H1L1 and marked.
        if deployment == "O3":
            preferred = events[events.event_name.eq("GW170814")]
            if not preferred.empty and "GW170814" not in used:
                selected.append(("network_h1l1v1_target", preferred.iloc[0]))
                used.add("GW170814")

        remainder = events[~events.event_name.astype(str).isin(used)].copy()
        remainder["hash"] = [
            hashlib.sha256(f"v10p1-real-pilot|{deployment}|{name}".encode()).hexdigest()
            for name in remainder.event_name
        ]
        for row in remainder.sort_values("hash").itertuples(index=False):
            if len(selected) >= 6:
                break
            selected.append(("deterministic_fill", pd.Series(row._asdict())))

        for purpose, row in selected[:6]:
            name = str(row.event_name)
            detector_paths = {
                det: str(existing_path(strain[(strain.event_name.eq(name)) & (strain.detector.eq(det))].iloc[0].strain_path))
                for det in ("H1", "L1", "V1")
                if not strain[(strain.event_name.eq(name)) & (strain.detector.eq(det))].empty
            }
            detectors = [det for det in ("H1", "L1", "V1") if det in detector_paths]
            rows.append(
                {
                    "case_id": f"{deployment.lower()}_real_{name.lower()}",
                    "stage": "smoke" if len([x for x in rows if x["deployment"] == deployment and x["stage"] == "smoke"]) == 0 else "expanded",
                    "deployment": deployment,
                    "event_role": "real",
                    "event_name": name,
                    "pilot_selection_purpose": purpose,
                    "gps_time": float(row.gps_time),
                    "network_snr": float(row.network_snr),
                    "snr_bin": snr_bin(float(row.network_snr)),
                    "detectors": ",".join(detectors),
                    "detector_paths_json": json.dumps(detector_paths, sort_keys=True),
                    "pe_path": str(existing_path(row.pe_path)),
                    "pe_group": str(row.pe_group),
                    "public_a90_deg2": float(row.a90_deg2) if pd.notna(row.a90_deg2) else np.nan,
                    "planned_map_morphology": purpose.replace("_public_map", "") if "public_map" in purpose else "to_be_measured",
                    "chirp_mass_hint": float(row.chirp_mass if "chirp_mass" in row and pd.notna(row.chirp_mass) else row.chirp_mass_source),
                    "source_row": np.nan,
                    "source_event_id": np.nan,
                    "lens_family": "none",
                    "image_index": np.nan,
                    "magnification": 1.0,
                    "morse_index": 0.0,
                    "target_snr": np.nan,
                }
            )
    # Ensure exactly one smoke real case per deployment; the scheduler adds
    # one injection smoke case per deployment after injection planning.
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def gwlmc_sources() -> pd.DataFrame:
    path = next(GWLMC_ROOT.glob("*_SourceParams.csv"))
    return pd.read_csv(path)


def source_bank_directory(deployment: str) -> Path:
    name = "gwtc3" if deployment == "O3" else "gwtc4"
    return SOURCE_BANK_ROOT / name / "shared/physical_h1l1_source_bank"


def offsource_inventory(deployment: str) -> pd.DataFrame:
    key = "GWTC3_O3" if deployment == "O3" else "GWTC4P1_O4A"
    frame = pd.read_csv(OFFSOURCE_INVENTORY)
    frame = frame[frame.deployment.eq(key) & frame.status.eq("usable") & frame.independent_blocks.gt(0)].copy()
    frame["hash"] = [
        hashlib.sha256(f"v10p1-noise|{deployment}|{int(start)}".encode()).hexdigest()
        for start in frame.gps_start
    ]
    return frame.sort_values("hash").reset_index(drop=True)


def known_event_times(deployment: str) -> np.ndarray:
    if deployment == "O3":
        path = PROJECT / "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv"
    else:
        path = PROJECT / "runs/real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest_gwtc4.csv"
    values = pd.to_numeric(pd.read_csv(path).gps_time, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return np.sort(values)


def joint_good_trigger(h1_path: Path, l1_path: Path, deployment: str, salt: str) -> float:
    h1_good, h1_start = dq_good_seconds(h1_path)
    l1_good, l1_start = dq_good_seconds(l1_path)
    common_start = int(math.ceil(max(h1_start, l1_start)))
    common_end = int(math.floor(min(h1_start + len(h1_good), l1_start + len(l1_good))))
    if common_end - common_start < 128:
        raise RuntimeError(f"Insufficient H1/L1 overlap in {h1_path} and {l1_path}")
    h_offset = common_start - int(round(h1_start))
    l_offset = common_start - int(round(l1_start))
    count = common_end - common_start
    good = h1_good[h_offset : h_offset + count] & l1_good[l_offset : l_offset + count]
    seconds = common_start + np.arange(count, dtype=np.float64)
    for event_time in known_event_times(deployment):
        if common_start - 128 <= event_time <= common_end + 128:
            good &= np.abs(seconds - event_time) >= 128.0
    # Require 64 contiguous good seconds and place the signal near the middle,
    # leaving room for the longest planned PE window and PSD exclusion.
    padded = np.r_[False, good, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    intervals = [
        (int(a), int(b)) for a, b in zip(changes[::2], changes[1::2]) if b - a >= 64
    ]
    if not intervals:
        raise RuntimeError(f"No 64 s event-vetoed joint-good interval in {h1_path}")
    keyed = sorted(
        intervals,
        key=lambda item: hashlib.sha256(f"{salt}|{common_start + item[0]}".encode()).hexdigest(),
    )
    first, last = keyed[0]
    return float(common_start + first + (last - first) // 2)


def select_noise_slots(deployment: str, count: int) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for row in offsource_inventory(deployment).itertuples(index=False):
        try:
            h1_path = existing_path(row.h1_path)
            l1_path = existing_path(row.l1_path)
            trigger = joint_good_trigger(h1_path, l1_path, deployment, f"slot-{len(slots)}")
        except Exception:
            continue
        slots.append(
            {
                "gps_time": trigger,
                "detectors": "H1,L1",
                "detector_paths_json": json.dumps(
                    {"H1": str(h1_path), "L1": str(l1_path)}, sort_keys=True
                ),
                "noise_block_uid": f"{deployment}:{int(row.gps_start)}",
            }
        )
        if len(slots) == count:
            return slots
    raise RuntimeError(f"Only found {len(slots)} independent pilot noise slots for {deployment}")


def deterministic_source_rows(frame: pd.DataFrame, count: int, salt: str) -> pd.DataFrame:
    selected = frame.copy()
    selected["_hash"] = [
        hashlib.sha256(f"{salt}|{int(value)}".encode()).hexdigest()
        for value in selected.gwlmc_event_id
    ]
    return selected.sort_values("_hash").head(count).drop(columns="_hash")


def select_injection_cases(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deployment in ("O3", "O4a"):
        bank = source_bank_directory(deployment)
        slots = select_noise_slots(deployment, 6)
        slot_index = 0
        system_targets = (("SIS", 10.0), ("PM", 15.0))
        for family, target_fainter_snr in system_targets:
            metadata_path = bank / f"{family}_data_0222/physical_source_pair_metadata.parquet"
            candidates = pd.read_parquet(metadata_path)
            candidates = candidates[
                candidates.proposal_snr_ratio.between(1.0, 2.5)
                & candidates.mass_1_detector.between(12.0, 150.0)
            ]
            selected = deterministic_source_rows(candidates, 1, f"v10p1-{deployment}-{family}").iloc[0]
            system_id = f"{deployment.lower()}_{family.lower()}_{int(selected.gwlmc_event_id)}"
            for image_number in (1, 2):
                slot = slots[slot_index]
                slot_index += 1
                rows.append(
                    {
                        "case_id": f"{system_id}_image{image_number}",
                        "stage": "smoke" if family == "SIS" and image_number == 1 else "expanded",
                        "deployment": deployment,
                        "event_role": "lensed_image",
                        "event_name": f"INJ_{system_id}_image{image_number}",
                        "pilot_selection_purpose": f"response_derived_{family.lower()}_image",
                        "gps_time": slot["gps_time"],
                        "network_snr": np.nan,
                        "snr_bin": "pending_calibration",
                        "detectors": slot["detectors"],
                        "detector_paths_json": slot["detector_paths_json"],
                        "pe_path": np.nan,
                        "pe_group": np.nan,
                        "public_a90_deg2": np.nan,
                        "planned_map_morphology": "to_be_measured",
                        "chirp_mass_hint": float(
                            (selected.mass_1_detector * selected.mass_2_detector) ** (3.0 / 5.0)
                            / (selected.mass_1_detector + selected.mass_2_detector) ** (1.0 / 5.0)
                        ),
                        "source_row": int(selected.gwlmc_row),
                        "source_event_id": int(selected.gwlmc_event_id),
                        "lens_family": str(selected.physical_lens_group),
                        "lens_system_id": system_id,
                        "image_index": int(selected[f"image{image_number}_index"]),
                        "magnification": float(selected[f"mu_image{image_number}"]),
                        "morse_index": float(selected[f"morse_image{image_number}"]),
                        "target_snr": float(target_fainter_snr),
                        "target_definition": "fainter_image_network_optimal_snr",
                        "common_amplitude_scale": np.nan,
                        "unscaled_network_snr": np.nan,
                        "noise_block_uid": slot["noise_block_uid"],
                        "time_placement_note": "independent_real_offsource_blocks; PE-resource pilot only",
                    }
                )
        unlensed_metadata = pd.read_parquet(
            bank / "Unlensed_data_0222/physical_unlensed_source_metadata.parquet"
        )
        unlensed_metadata = deterministic_source_rows(
            unlensed_metadata, 2, f"v10p1-{deployment}-unlensed"
        )
        for index, selected in enumerate(unlensed_metadata.itertuples(index=False)):
            slot = slots[slot_index]
            slot_index += 1
            target = (12.0, 25.0)[index]
            rows.append(
                {
                    "case_id": f"{deployment.lower()}_unlensed_{int(selected.gwlmc_event_id)}",
                    "stage": "expanded",
                    "deployment": deployment,
                    "event_role": "unlensed_injection",
                    "event_name": f"INJ_{deployment.lower()}_unlensed_{int(selected.gwlmc_event_id)}",
                    "pilot_selection_purpose": "response_derived_unlensed",
                    "gps_time": slot["gps_time"],
                    "network_snr": np.nan,
                    "snr_bin": "pending_calibration",
                    "detectors": slot["detectors"],
                    "detector_paths_json": slot["detector_paths_json"],
                    "pe_path": np.nan,
                    "pe_group": np.nan,
                    "public_a90_deg2": np.nan,
                    "planned_map_morphology": "to_be_measured",
                    "chirp_mass_hint": float(
                        (selected.mass_1_detector * selected.mass_2_detector) ** (3.0 / 5.0)
                        / (selected.mass_1_detector + selected.mass_2_detector) ** (1.0 / 5.0)
                    ),
                    "source_row": int(selected.gwlmc_row),
                    "source_event_id": int(selected.gwlmc_event_id),
                    "lens_family": "unlensed",
                    "lens_system_id": f"{deployment.lower()}_unlensed_{int(selected.gwlmc_event_id)}",
                    "image_index": -1,
                    "magnification": 1.0,
                    "morse_index": 0.0,
                    "target_snr": target,
                    "target_definition": "event_network_optimal_snr",
                    "common_amplitude_scale": np.nan,
                    "unscaled_network_snr": np.nan,
                    "noise_block_uid": slot["noise_block_uid"],
                    "time_placement_note": "independent_real_offsource_block",
                }
            )
    return pd.DataFrame(rows)


def environment_snapshot() -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT)
    packages = {}
    for module in (bilby, hp, np, pd):
        packages[module.__name__] = getattr(module, "__version__", "unknown")
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable:{type(exc).__name__}"
    return {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "python": sys.version,
        "packages": packages,
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": gpu,
    }


def prepare(output: Path, config_path: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for subdir in ("contracts", "manifests", "cases", "logs", "summaries", "figures"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "contracts/V10P1_PE_PILOT_CONFIG.json")
    addendum = config_path.parent / "V10P1_EXPLORATORY_PROTOCOL_ADDENDUM_CN.md"
    shutil.copy2(addendum, output / "contracts/V10P1_EXPLORATORY_PROTOCOL_ADDENDUM_CN.md")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    real_cases = select_real_cases(config)
    injection_cases = calibrate_injection_cases(select_injection_cases(config), config)
    real_cases.to_csv(
        output / "manifests/pe_pilot_cases_real.csv", index=False, encoding="utf-8-sig"
    )
    injection_cases.to_csv(
        output / "manifests/pe_pilot_cases_injections.csv", index=False, encoding="utf-8-sig"
    )
    cases = pd.concat([real_cases, injection_cases], ignore_index=True, sort=False)
    cases.to_csv(output / "manifests/pe_pilot_cases_all.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summaries/environment.json", environment_snapshot())
    write_json(
        output / "STATUS.json",
        {
            "status": "PREPARED_HOMOGENEOUS_PE_PILOT_CASES",
            "created_at_utc": utc_now(),
            "historical_results_modified": False,
            "real_candidate_rerank_performed": False,
            "cases": int(len(cases)),
            "real_cases": int(len(real_cases)),
            "injection_cases": int(len(injection_cases)),
            "smoke_cases": int(cases.stage.eq("smoke").sum()),
        },
    )


def read_public_fiducial(path: Path, group: str) -> tuple[dict[str, float], dict[str, Any]]:
    fields = (
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "theta_jn",
        "psi",
        "phase",
        "luminosity_distance",
        "geocent_time",
        "ra",
        "dec",
    )
    with h5py.File(path, "r") as handle:
        samples = handle[f"{group}/posterior_samples"]
        names = set(samples.dtype.names or ())
        missing = sorted(set(fields) - names)
        if missing:
            raise KeyError(f"Missing posterior fields in {path}:{group}: {missing}")
        if "log_likelihood" not in names:
            raise KeyError(f"Missing log_likelihood in {path}:{group}")
        log_likelihood = np.asarray(samples["log_likelihood"][()], dtype=np.float64)
        if not np.isfinite(log_likelihood).any():
            raise ValueError(f"No finite log_likelihood values in {path}:{group}")
        index = int(np.nanargmax(log_likelihood))
        result = {name: float(samples[name][index]) for name in fields}
    # Bilby's time-marginalized likelihood introduces this nuisance parameter
    # even though it is absent from public posterior products.
    result["time_jitter"] = 0.0
    audit = {
        "selection": "maximum_log_likelihood_public_posterior_sample",
        "posterior_path": str(path),
        "posterior_group": group,
        "sample_index": index,
        "public_log_likelihood": float(log_likelihood[index]),
        "posterior_sample_count": int(len(log_likelihood)),
    }
    return result, audit


def matching_fiducial_group(path: Path, requested_group: str, approximant: str) -> str:
    token = approximant.lower()
    with h5py.File(path, "r") as handle:
        candidates = [
            name
            for name in handle.keys()
            if token in name.lower() and "posterior_samples" in handle[name]
        ]
    if not candidates:
        return requested_group
    return sorted(candidates)[0]


@dataclass
class StrainFile:
    path: Path
    start: float
    sample_rate: float
    duration: float


def strain_file_metadata(path: Path) -> StrainFile:
    with h5py.File(path, "r") as handle:
        data = handle["strain/Strain"]
        start = float(data.attrs["Xstart"])
        delta_t = float(data.attrs["Xspacing"])
        duration = len(data) * delta_t
    return StrainFile(path=path, start=start, sample_rate=1.0 / delta_t, duration=duration)


def read_raw_slice(path: Path, start: float, duration: float) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as handle:
        data = handle["strain/Strain"]
        file_start = float(data.attrs["Xstart"])
        delta_t = float(data.attrs["Xspacing"])
        sampling_frequency = 1.0 / delta_t
        first = int(round((start - file_start) * sampling_frequency))
        count = int(round(duration * sampling_frequency))
        if first < 0 or first + count > len(data):
            raise ValueError(f"Requested [{start},{start + duration}) outside {path}")
        values = np.asarray(data[first : first + count], dtype=np.float64)
        actual_start = file_start + first / sampling_frequency
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite strain in {path}")
    return values, actual_start, sampling_frequency


def longest_boolean_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[::2]
    ends = changes[1::2]
    if len(starts) == 0:
        return 0, 0
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def dq_good_seconds(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        dataset = handle["quality/simple/DQmask"]
        values = np.asarray(dataset[()], dtype=np.uint32)
        start = float(dataset.attrs["Xstart"])
        shortnames = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["quality/simple/DQShortnames"][()]
        ]
    required = [name for name in ("DATA", "CBC_CAT2") if name in shortnames]
    if not required:
        return np.ones(len(values), dtype=bool), start
    good = np.ones(len(values), dtype=bool)
    for name in required:
        bit = shortnames.index(name)
        good &= ((values >> bit) & 1).astype(bool)
    return good, start


def psd_reference(path: Path, trigger: float, maximum_seconds: float, exclusion_seconds: float) -> tuple[np.ndarray, float, float]:
    metadata = strain_file_metadata(path)
    good, dq_start = dq_good_seconds(path)
    seconds = dq_start + np.arange(len(good), dtype=np.float64)
    good &= np.abs(seconds - trigger) >= exclusion_seconds
    first, last = longest_boolean_run(good)
    if last - first < 64:
        raise RuntimeError(f"No >=64 s contiguous DQ-good PSD interval in {path}")
    length = min(float(last - first), maximum_seconds)
    # Take the end of the longest interval when it lies before the event;
    # otherwise take its beginning.  This is deterministic and avoids the
    # excluded event neighborhood.
    if dq_start + last <= trigger - exclusion_seconds:
        start = dq_start + last - length
    else:
        start = dq_start + first
    values, actual_start, sampling_frequency = read_raw_slice(path, start, length)
    return values, actual_start, sampling_frequency


def median_welch(values: np.ndarray, sampling_frequency: float, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    seconds = float(config["pe_pipeline"]["psd"]["welch_segment_s"])
    overlap = float(config["pe_pipeline"]["psd"]["overlap_fraction"])
    nperseg = int(round(seconds * sampling_frequency))
    frequencies, psd = signal.welch(
        values,
        fs=sampling_frequency,
        window="hann",
        nperseg=nperseg,
        noverlap=int(round(overlap * nperseg)),
        detrend="constant",
        scaling="density",
        average="median",
    )
    floor = max(float(np.median(psd)) * 1e-12, np.finfo(np.float64).tiny)
    return frequencies.astype(np.float64), np.maximum(psd, floor).astype(np.float64)


def make_interferometers(case: pd.Series, config: dict[str, Any], duration: float) -> tuple[Any, dict[str, Any]]:
    pipeline = config["pe_pipeline"]
    target_fs = float(pipeline["sampling_frequency_hz"])
    post_trigger = float(pipeline["post_trigger_duration_s"])
    requested_start = float(case.gps_time) + post_trigger - duration
    paths = json.loads(case.detector_paths_json)
    ifos = bilby.gw.detector.InterferometerList([])
    audit = {"detectors": [], "psd": {}}
    actual_starts = []
    for detector in normalize_detector_string(case.detectors):
        path = existing_path(paths[detector])
        raw, actual_start, raw_fs = read_raw_slice(path, requested_start, duration)
        if raw_fs != target_fs:
            ratio = target_fs / raw_fs
            up = int(round(ratio * 1000))
            down = 1000
            divisor = math.gcd(up, down)
            values = signal.resample_poly(raw, up // divisor, down // divisor, window=("kaiser", 8.6))
        else:
            values = raw
        expected = int(round(duration * target_fs))
        values = np.asarray(values[:expected], dtype=np.float64)
        if len(values) != expected:
            raise RuntimeError(f"Unexpected resampled length for {detector}: {len(values)} != {expected}")
        reference, psd_start, psd_fs = psd_reference(
            path,
            float(case.gps_time),
            float(pipeline["psd"]["maximum_duration_s"]),
            float(pipeline["psd"]["event_exclusion_s"]),
        )
        frequency, psd = median_welch(reference, psd_fs, config)
        ifo = bilby.gw.detector.get_empty_interferometer(detector)
        ifo.minimum_frequency = float(pipeline["minimum_frequency_hz"])
        ifo.maximum_frequency = float(pipeline["maximum_frequency_hz"])
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequency,
            psd_array=psd,
        )
        ifo.strain_data.roll_off = 0.4
        ifo.strain_data.set_from_time_domain_strain(
            values,
            sampling_frequency=target_fs,
            duration=duration,
            start_time=actual_start,
        )
        ifos.append(ifo)
        actual_starts.append(actual_start)
        audit["detectors"].append(
            {
                "detector": detector,
                "strain_path": str(path),
                "strain_sha256": sha256(path),
                "raw_sampling_frequency_hz": raw_fs,
                "analysis_sampling_frequency_hz": target_fs,
                "analysis_start_gps": actual_start,
                "analysis_duration_s": duration,
            }
        )
        audit["psd"][detector] = {
            "reference_start_gps": psd_start,
            "reference_duration_s": len(reference) / psd_fs,
            "reference_sampling_frequency_hz": psd_fs,
            "frequency_bins": len(frequency),
        }
    if max(actual_starts) - min(actual_starts) > 0.5 / target_fs:
        raise RuntimeError(f"Detector analysis starts differ: {actual_starts}")
    return ifos, audit


def waveform_generator(duration: float, config: dict[str, Any], relative_binning: bool = True) -> Any:
    pipeline = config["pe_pipeline"]
    source_model = (
        bilby.gw.source.lal_binary_black_hole_relative_binning
        if relative_binning
        else bilby.gw.source.lal_binary_black_hole
    )
    return bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=float(pipeline["sampling_frequency_hz"]),
        frequency_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": pipeline["waveform_approximant"],
            "reference_frequency": float(pipeline["reference_frequency_hz"]),
            "minimum_frequency": float(pipeline["minimum_frequency_hz"]),
            "maximum_frequency": float(pipeline["maximum_frequency_hz"]),
            "catch_waveform_errors": True,
        },
    )


def source_row_parameters(case: pd.Series) -> tuple[dict[str, float], dict[str, float]]:
    source = gwlmc_sources().iloc[int(case.source_row)]
    m1 = float(source.m1_det)
    m2 = float(source.m2_det)
    if m2 > m1:
        m1, m2 = m2, m1
    base = {
        "mass_1": m1,
        "mass_2": m2,
        "a_1": float(source.a1),
        "a_2": float(source.a2),
        "tilt_1": float(source.tilt1),
        "tilt_2": float(source.tilt2),
        "phi_12": float(source.phi12),
        "phi_jl": float(source.phijl),
        "theta_jn": float(source.theta_jn),
        "psi": float(source.psi),
        "phase": float(source.phase),
        "luminosity_distance": float(source.dl_source),
        "geocent_time": float(case.gps_time),
        "ra": float(source.ra),
        "dec": float(source.dec),
    }
    magnification = abs(float(case.magnification))
    scale = float(case.common_amplitude_scale) if pd.notna(case.common_amplitude_scale) else 1.0
    apparent_distance = base["luminosity_distance"] / max(scale * math.sqrt(magnification), 1e-12)
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    # A common complex Morse factor is approximately absorbed by coalescence
    # phase in single-event PE.  The exact injected factor is applied below.
    apparent_phase = float((base["phase"] + 0.5 * math.pi * float(case.morse_index)) % (2 * math.pi))
    fiducial = {
        "chirp_mass": chirp_mass,
        "mass_ratio": m2 / m1,
        "a_1": base["a_1"],
        "a_2": base["a_2"],
        "tilt_1": base["tilt_1"],
        "tilt_2": base["tilt_2"],
        "phi_12": base["phi_12"],
        "phi_jl": base["phi_jl"],
        "theta_jn": base["theta_jn"],
        "psi": base["psi"],
        "phase": apparent_phase,
        "luminosity_distance": apparent_distance,
        "geocent_time": base["geocent_time"],
        "ra": base["ra"],
        "dec": base["dec"],
        "time_jitter": 0.0,
    }
    return base, fiducial


def injection_polarizations(
    case: pd.Series,
    duration: float,
    config: dict[str, Any],
    amplitude_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    base, fiducial = source_row_parameters(case)
    generator = waveform_generator(duration, config, relative_binning=False)
    polarizations = generator.frequency_domain_strain(base)
    factor = (
        float(amplitude_scale)
        * math.sqrt(abs(float(case.magnification)))
        * np.exp(-1j * math.pi * float(case.morse_index))
    )
    polarizations = {key: np.asarray(value, dtype=np.complex128) * factor for key, value in polarizations.items()}
    return polarizations, base, fiducial


def network_optimal_snr(
    ifos: Any,
    polarizations: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> tuple[float, dict[str, float]]:
    per_detector: dict[str, float] = {}
    total = 0.0
    for ifo in ifos:
        response = ifo.get_detector_response(polarizations, parameters)
        snr_squared = float(np.real(ifo.optimal_snr_squared(response)))
        per_detector[ifo.name] = math.sqrt(max(snr_squared, 0.0))
        total += max(snr_squared, 0.0)
    return math.sqrt(total), per_detector


def calibrate_injection_cases(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    calibrated = frame.copy()
    unscaled: dict[int, float] = {}
    for index, case in calibrated.iterrows():
        duration = duration_for_chirp_mass(float(case.chirp_mass_hint), config)
        ifos, _ = make_interferometers(case, config, duration)
        polarizations, base, _ = injection_polarizations(case, duration, config, amplitude_scale=1.0)
        snr, _ = network_optimal_snr(ifos, polarizations, base)
        if not np.isfinite(snr) or snr <= 0.0:
            raise RuntimeError(f"Invalid unscaled network SNR for {case.case_id}: {snr}")
        unscaled[int(index)] = snr
        calibrated.loc[index, "unscaled_network_snr"] = snr
    for system_id, group in calibrated.groupby("lens_system_id", sort=True):
        indices = [int(value) for value in group.index]
        if group.event_role.eq("lensed_image").all():
            reference = min(unscaled[index] for index in indices)
            scale = float(group.target_snr.iloc[0]) / reference
            maximum = max(unscaled[index] for index in indices)
            scale = min(
                scale,
                float(config["pilot_design"]["maximum_injection_network_snr"]) / maximum,
            )
        else:
            index = indices[0]
            scale = float(group.target_snr.iloc[0]) / unscaled[index]
        for index in indices:
            calibrated.loc[index, "common_amplitude_scale"] = scale
            calibrated.loc[index, "network_snr"] = unscaled[index] * scale
            calibrated.loc[index, "snr_bin"] = snr_bin(unscaled[index] * scale)
    return calibrated


def priors_from_fiducial(fiducial: dict[str, float], trigger: float) -> Any:
    priors = bilby.gw.prior.BBHPriorDict()
    priors.pop("mass_1", None)
    priors.pop("mass_2", None)
    chirp = float(fiducial["chirp_mass"])
    ratio = float(fiducial["mass_ratio"])
    distance = float(fiducial["luminosity_distance"])
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        max(2.0, 0.7 * chirp),
        min(250.0, 1.3 * chirp),
        name="chirp_mass",
    )
    priors["mass_ratio"] = bilby.core.prior.Uniform(
        max(0.1, 0.5 * ratio),
        1.0,
        name="mass_ratio",
    )
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger - 0.1,
        trigger + 0.1,
        name="geocent_time",
    )
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        max(10.0, 0.2 * distance),
        max(100.0, 5.0 * distance),
        name="luminosity_distance",
    )
    return priors


class ResourceMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.peak_gpu_bytes = 0
        self.gpu_active_seconds = 0.0
        self.peak_rss_bytes = 0
        self.samples = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(2.0):
            self.samples += 1
            try:
                parent = psutil.Process(os.getpid())
                processes = [parent, *parent.children(recursive=True)]
                rss = sum(process.memory_info().rss for process in processes if process.is_running())
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(rss))
            except (psutil.Error, OSError):
                pass
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
                self.peak_gpu_bytes = max(self.peak_gpu_bytes, sum(values) * 1024 * 1024)
                if any(value > 0 for value in values):
                    self.gpu_active_seconds += 2.0
            except Exception:
                pass

    def __enter__(self) -> "ResourceMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)


def usage_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)


def sample_interferometers(
    case: pd.Series,
    case_dir: Path,
    config: dict[str, Any],
    sampler_seed: int,
    fiducial: dict[str, float],
    duration: float,
    ifos: Any,
    strain_audit: dict[str, Any],
    setup_started: float,
    injection_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    complete = case_dir / "pe_metrics.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    generator = waveform_generator(duration, config)
    priors = priors_from_fiducial(fiducial, float(case.gps_time))
    likelihood_started = time.perf_counter()
    likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=generator,
        fiducial_parameters=fiducial,
        time_marginalization=bool(config["pe_pipeline"]["time_marginalization"]),
        distance_marginalization=bool(config["pe_pipeline"]["distance_marginalization"]),
        phase_marginalization=bool(config["pe_pipeline"]["phase_marginalization"]),
        priors=priors,
        epsilon=float(config["pe_pipeline"]["relative_binning_epsilon"]),
    )
    likelihood_setup_seconds = time.perf_counter() - likelihood_started
    setup_seconds = time.perf_counter() - setup_started
    sampler = config["pe_pipeline"]["sampler_settings"]
    sampler_started = time.perf_counter()
    cpu_before = usage_seconds()
    bilby_dir = case_dir / "bilby"
    restart_count = int(any(bilby_dir.glob("*resume*")))
    with ResourceMonitor() as monitor:
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=config["pe_pipeline"]["sampler"],
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            outdir=str(case_dir / "bilby"),
            label=str(case.case_id),
            resume=True,
            seed=int(sampler_seed),
            nlive=int(sampler["nlive"]),
            sample=str(sampler["sample"]),
            nact=int(sampler["nact"]),
            dlogz=float(sampler["dlogz"]),
            npool=int(sampler["npool"]),
            check_point_delta_t=int(sampler["check_point_delta_t"]),
            print_method=str(sampler["print_method"]),
            save="hdf5",
            plot=False,
        )
    sampler_wall_seconds = time.perf_counter() - sampler_started
    sampler_cpu_seconds = usage_seconds() - cpu_before
    posterior = result.posterior
    posterior_path = case_dir / "posterior_samples.hdf5"
    fields = ["ra", "dec", "luminosity_distance"]
    dtype = [(name, "f8") for name in fields]
    values = np.empty(len(posterior), dtype=dtype)
    for name in fields:
        values[name] = pd.to_numeric(posterior[name], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values["ra"]) & np.isfinite(values["dec"])
    values = values[valid]
    with h5py.File(posterior_path, "w") as handle:
        handle.create_dataset("posterior_samples", data=values, compression="gzip", shuffle=True)
    metrics = {
        "case_id": str(case.case_id),
        "sampler_seed": int(sampler_seed),
        "event_role": str(case.event_role),
        "deployment": str(case.deployment),
        "event_name": str(case.event_name),
        "detectors": str(case.detectors),
        "network_snr": float(case.network_snr),
        "snr_bin": str(case.snr_bin),
        "duration_s": duration,
        "setup_seconds": setup_seconds,
        "likelihood_setup_seconds": likelihood_setup_seconds,
        "sampler_wall_seconds": sampler_wall_seconds,
        "sampler_cpu_seconds": sampler_cpu_seconds,
        "peak_rss_bytes": int(
            max(monitor.peak_rss_bytes, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        ),
        "peak_gpu_memory_bytes": int(monitor.peak_gpu_bytes),
        "gpu_active_seconds": float(monitor.gpu_active_seconds),
        "posterior_samples": int(len(values)),
        "posterior_bytes": posterior_path.stat().st_size,
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_err),
        "sampler_converged": bool(np.isfinite(result.log_evidence) and len(values) >= 1000),
        "restart_count": restart_count,
        "strain_audit": strain_audit,
        "injection_audit": injection_audit,
        "fiducial": fiducial,
        "completed_at_utc": utc_now(),
    }
    write_json(case_dir / "pe_metrics.json", metrics)
    return metrics


def run_real_case(case: pd.Series, case_dir: Path, config: dict[str, Any], sampler_seed: int) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = existing_path(case.pe_path)
    requested_group = str(case.pe_group)
    fiducial_group = matching_fiducial_group(
        posterior_path,
        requested_group,
        str(config["pe_pipeline"]["waveform_approximant"]),
    )
    fiducial, fiducial_audit = read_public_fiducial(
        posterior_path, fiducial_group
    )
    fiducial_audit["manifest_posterior_group"] = requested_group
    fiducial_audit["waveform_matched_group_used"] = fiducial_group != requested_group
    duration = duration_for_chirp_mass(float(fiducial["chirp_mass"]), config)
    setup_started = time.perf_counter()
    ifos, strain_audit = make_interferometers(case, config, duration)
    strain_audit["relative_binning_fiducial"] = fiducial_audit
    return sample_interferometers(
        case,
        case_dir,
        config,
        sampler_seed,
        fiducial,
        duration,
        ifos,
        strain_audit,
        setup_started,
    )


def run_injection_case(case: pd.Series, case_dir: Path, config: dict[str, Any], sampler_seed: int) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    duration = duration_for_chirp_mass(float(case.chirp_mass_hint), config)
    setup_started = time.perf_counter()
    ifos, strain_audit = make_interferometers(case, config, duration)
    polarizations, base, fiducial = injection_polarizations(
        case,
        duration,
        config,
        amplitude_scale=float(case.common_amplitude_scale),
    )
    achieved_snr, detector_snrs = network_optimal_snr(ifos, polarizations, base)
    for ifo in ifos:
        ifo.inject_signal(
            parameters=base,
            injection_polarizations=polarizations,
            raise_error=True,
        )
    injection_audit = {
        "protocol": "response_derived_common_system_amplitude_scale_v10p1",
        "source_row": int(case.source_row),
        "source_event_id": int(case.source_event_id),
        "lens_system_id": str(case.lens_system_id),
        "lens_family": str(case.lens_family),
        "image_index": int(case.image_index),
        "magnification": float(case.magnification),
        "morse_index": float(case.morse_index),
        "common_amplitude_scale": float(case.common_amplitude_scale),
        "unscaled_network_snr": float(case.unscaled_network_snr),
        "planned_network_snr": float(case.network_snr),
        "achieved_network_snr": achieved_snr,
        "detector_optimal_snrs": detector_snrs,
        "truth": base,
        "apparent_fiducial": fiducial,
        "noise_block_uid": str(case.noise_block_uid),
    }
    if not math.isclose(achieved_snr, float(case.network_snr), rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(
            f"Injection SNR drift for {case.case_id}: planned={case.network_snr}, achieved={achieved_snr}"
        )
    return sample_interferometers(
        case,
        case_dir,
        config,
        sampler_seed,
        fiducial,
        duration,
        ifos,
        strain_audit,
        setup_started,
        injection_audit=injection_audit,
    )


def run_logged(command: list[str], log_path: Path) -> tuple[float, int]:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return time.perf_counter() - started, int(process.returncode)


def header_dict(header: list[tuple[str, Any]]) -> dict[str, Any]:
    return {str(key): value for key, value in header}


def hpd_components(probability: np.ndarray, nside: int, nest: bool, credible_level: float = 0.9) -> int:
    order = np.argsort(probability)[::-1]
    cumulative = np.cumsum(probability[order], dtype=np.float64)
    count = int(np.searchsorted(cumulative, credible_level, side="left") + 1)
    selected = np.asarray(order[:count], dtype=np.int64)
    if len(selected) == 0:
        return 0
    # Count connected components on the HEALPix neighbour graph.  The sparse
    # implementation is important for broad posteriors containing millions
    # of 90%-HPD pixels.
    lookup = np.full(len(probability), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected), dtype=np.int32)
    neighbours = hp.get_all_neighbours(nside, selected, nest=nest).T
    row = np.repeat(np.arange(len(selected), dtype=np.int32), neighbours.shape[1])
    neighbour_flat = neighbours.reshape(-1)
    valid = neighbour_flat >= 0
    row = row[valid]
    column = lookup[neighbour_flat[valid]]
    valid = column >= 0
    graph = sparse.coo_matrix(
        (np.ones(int(np.count_nonzero(valid)), dtype=np.uint8), (row[valid], column[valid])),
        shape=(len(selected), len(selected)),
    ).tocsr()
    components, _ = csgraph.connected_components(graph, directed=False, return_labels=True)
    return int(components)


def hpd_axis_ratio(
    probability: np.ndarray,
    nside: int,
    nest: bool,
    credible_level: float = 0.9,
) -> float:
    order = np.argsort(probability)[::-1]
    cumulative = np.cumsum(probability[order], dtype=np.float64)
    count = int(np.searchsorted(cumulative, credible_level, side="left") + 1)
    selected = np.asarray(order[:count], dtype=np.int64)
    if len(selected) < 3:
        return 1.0
    weights = np.asarray(probability[selected], dtype=np.float64)
    weights /= np.sum(weights, dtype=np.float64)
    center = np.asarray(hp.pix2vec(nside, int(order[0]), nest=nest), dtype=np.float64)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(center, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    basis_x = np.cross(reference, center)
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(center, basis_x)
    vectors = np.asarray(hp.pix2vec(nside, selected, nest=nest), dtype=np.float64).T
    x = vectors @ basis_x
    y = vectors @ basis_y
    x -= np.sum(weights * x)
    y -= np.sum(weights * y)
    covariance = np.array(
        [
            [np.sum(weights * x * x), np.sum(weights * x * y)],
            [np.sum(weights * x * y), np.sum(weights * y * y)],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    floor = max(float(eigenvalues[-1]) * 1e-12, np.finfo(np.float64).tiny)
    return float(math.sqrt(max(float(eigenvalues[-1]), floor) / max(float(eigenvalues[0]), floor)))


def probability_map_metrics(
    probability: np.ndarray,
    nside: int,
    nest: bool,
    compute_components: bool,
) -> dict[str, Any]:
    values = np.asarray(probability, dtype=np.float64)
    raw_sum = float(np.sum(values, dtype=np.float64))
    finite = bool(np.isfinite(values).all())
    nonnegative = bool(np.all(values >= 0.0))
    if not finite or not nonnegative or raw_sum <= 0.0:
        return {
            "map_valid": False,
            "normalization_before": raw_sum,
            "nside": int(nside),
            "nest": bool(nest),
        }
    values /= raw_sum
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    pixel_area = float(hp.nside2pixarea(nside, degrees=True))
    n50 = int(np.searchsorted(cumulative, 0.5, side="left") + 1)
    n90 = int(np.searchsorted(cumulative, 0.9, side="left") + 1)
    positive = values[values > 0]
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64))
    return {
        "map_valid": True,
        "normalization_before": raw_sum,
        "normalization_abs_error": abs(raw_sum - 1.0),
        "nside": int(nside),
        "nest": bool(nest),
        "pixels": int(len(values)),
        "a50_deg2": n50 * pixel_area,
        "a90_deg2": n90 * pixel_area,
        "entropy_nats": entropy,
        "kl_from_uniform_nats": float(math.log(len(values)) - entropy),
        "hpd90_component_count": (
            int(hpd_components(values, nside, nest)) if compute_components else None
        ),
        "hpd90_axis_ratio_proxy": (
            hpd_axis_ratio(values, nside, nest) if compute_components else None
        ),
    }


def build_and_audit_sky_map(case_dir: Path, config: dict[str, Any], case: pd.Series, sampler_seed: int) -> dict[str, Any]:
    posterior_path = case_dir / "posterior_samples.hdf5"
    map_dir = case_dir / "sky_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    moc_path = map_dir / "skymap.fits.gz"
    executable_dir = Path(sys.executable).parent
    from_samples = executable_dir / "ligo-skymap-from-samples"
    flatten = executable_dir / "ligo-skymap-flatten"
    if not from_samples.exists() or not flatten.exists():
        raise FileNotFoundError(
            f"Sky-map commands not found next to {sys.executable}; "
            "run from the frozen v10.1 environment"
        )
    map_config = config["posterior_map"]
    command = [
        str(from_samples),
        str(posterior_path),
        "--outdir",
        str(map_dir),
        "--fitsoutname",
        moc_path.name,
        "--maxpts",
        str(int(map_config["maximum_samples"])),
        "--trials",
        str(int(map_config["kde_trials"])),
        "--disable-distance-map",
        "--enable-multiresolution",
        "--jobs",
        str(min(16, os.cpu_count() or 1)),
        "--seed",
        str(int(sampler_seed)),
        "--objid",
        str(case.case_id),
    ]
    instruments = normalize_detector_string(case.detectors)
    if instruments:
        command.extend(["--instruments", *instruments])
    moc_seconds, returncode = run_logged(command, map_dir / "from_samples.log")
    if returncode != 0 or not moc_path.exists():
        raise RuntimeError(f"ligo-skymap-from-samples failed with code {returncode}")

    raster_metrics: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="v10p1-raster-") as temporary:
        temporary_dir = Path(temporary)
        for nside in (
            int(config["coarse_audit_nside"]),
            int(config["analysis_nside"]),
            int(config["convergence_reference_nside"]),
        ):
            flattened = temporary_dir / f"nside{nside}.fits"
            seconds, code = run_logged(
                [str(flatten), "--nside", str(nside), str(moc_path), str(flattened)],
                map_dir / f"flatten_nside{nside}.log",
            )
            if code != 0 or not flattened.exists():
                raise RuntimeError(f"ligo-skymap-flatten Nside={nside} failed with code {code}")
            probability, header = hp.read_map(flattened, field=0, h=True, verbose=False)
            metadata = header_dict(header)
            ordering = str(metadata.get("ORDERING", "RING")).upper()
            metrics = probability_map_metrics(
                probability,
                nside,
                nest=ordering == "NESTED",
                compute_components=nside == int(config["analysis_nside"]),
            )
            metrics["rasterize_seconds"] = seconds
            metrics["temporary_dense_bytes"] = int(np.asarray(probability).nbytes)
            metrics["ordering"] = ordering
            raster_metrics[str(nside)] = metrics
            del probability
    result = {
        "moc_path": str(moc_path),
        "moc_bytes": int(moc_path.stat().st_size),
        "moc_sha256": sha256(moc_path),
        "moc_generation_seconds": moc_seconds,
        "moc_generation_returncode": returncode,
        "raster": raster_metrics,
        "map_valid": bool(all(item.get("map_valid", False) for item in raster_metrics.values())),
    }
    write_json(case_dir / "sky_map_metrics.json", result)
    return result


def postprocess_case(output: Path, case_id: str, sampler_seed: int) -> None:
    config = json.loads((output / "contracts/V10P1_PE_PILOT_CONFIG.json").read_text(encoding="utf-8"))
    case = find_case(output, case_id)
    target = output / "cases" / f"{case_id}_seed{sampler_seed}"
    if not (target / "pe_metrics.json").exists():
        raise FileNotFoundError(target / "pe_metrics.json")
    metrics = build_and_audit_sky_map(target, config, case, sampler_seed)
    write_json(output / "logs" / f"{case_id}_seed{sampler_seed}.map_complete.json", metrics)


def summarize(output: Path) -> None:
    config = json.loads((output / "contracts/V10P1_PE_PILOT_CONFIG.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((output / "cases").glob("*/pe_metrics.json")):
        case_dir = metrics_path.parent
        pe = json.loads(metrics_path.read_text(encoding="utf-8"))
        sky_path = case_dir / "sky_map_metrics.json"
        sky = json.loads(sky_path.read_text(encoding="utf-8")) if sky_path.exists() else {}
        raster = sky.get("raster", {})
        row: dict[str, Any] = {
            key: pe.get(key)
            for key in (
                "case_id",
                "sampler_seed",
                "event_role",
                "deployment",
                "event_name",
                "detectors",
                "network_snr",
                "snr_bin",
                "duration_s",
                "setup_seconds",
                "likelihood_setup_seconds",
                "sampler_wall_seconds",
                "sampler_cpu_seconds",
                "peak_rss_bytes",
                "peak_gpu_memory_bytes",
                "gpu_active_seconds",
                "posterior_samples",
                "posterior_bytes",
                "log_evidence",
                "log_evidence_error",
                "sampler_converged",
            )
        }
        row.update(
            {
                "moc_generation_seconds": sky.get("moc_generation_seconds"),
                "moc_bytes": sky.get("moc_bytes"),
                "map_valid": sky.get("map_valid", False),
            }
        )
        for nside in (
            int(config["coarse_audit_nside"]),
            int(config["analysis_nside"]),
            int(config["convergence_reference_nside"]),
        ):
            values = raster.get(str(nside), {})
            for key in (
                "rasterize_seconds",
                "a50_deg2",
                "a90_deg2",
                "entropy_nats",
                "kl_from_uniform_nats",
                "hpd90_component_count",
                "hpd90_axis_ratio_proxy",
                "temporary_dense_bytes",
            ):
                row[f"{key}_nside{nside}"] = values.get(key)
        raster_seconds = sum(
            float(values.get("rasterize_seconds", 0.0) or 0.0) for values in raster.values()
        )
        row["total_case_wall_seconds"] = (
            float(pe.get("setup_seconds", 0.0) or 0.0)
            + float(pe.get("sampler_wall_seconds", 0.0) or 0.0)
            + float(sky.get("moc_generation_seconds", 0.0) or 0.0)
            + raster_seconds
        )
        row["failure_count"] = int(
            (output / "logs" / f"{pe['case_id']}_seed{pe['sampler_seed']}.failure.json").exists()
        )
        checkpoint_dir = case_dir / "bilby"
        row["checkpoint_files"] = len(list(checkpoint_dir.glob("*resume*"))) + len(
            list(checkpoint_dir.glob("*checkpoint*"))
        )
        row["restart_count"] = int(pe.get("restart_count", 0) or 0)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "summaries/pe_pilot_case_metrics.csv", index=False, encoding="utf-8-sig")
    if table.empty:
        write_json(
            output / "summaries/pe_pilot_summary.json",
            {"status": "NO_COMPLETED_CASES", "generated_at_utc": utc_now()},
        )
        return

    numeric = [
        "total_case_wall_seconds",
        "sampler_wall_seconds",
        "sampler_cpu_seconds",
        "peak_rss_bytes",
        "peak_gpu_memory_bytes",
        "gpu_active_seconds",
        "posterior_bytes",
        "moc_bytes",
    ]
    summaries: list[dict[str, Any]] = []
    grouping = [("all", table), *[(name, group) for name, group in table.groupby("deployment")]]
    for label, group in grouping:
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if not len(values):
                continue
            summaries.append(
                {
                    "group": label,
                    "metric": column,
                    "n": len(values),
                    "p50": float(np.quantile(values, 0.5)),
                    "p90": float(np.quantile(values, 0.9)),
                    "maximum": float(np.max(values)),
                }
            )
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "summaries/pe_pilot_resource_quantiles.csv", index=False)

    wall = summary_table[(summary_table.group.eq("all")) & summary_table.metric.eq("total_case_wall_seconds")]
    extrapolation: list[dict[str, Any]] = []
    if len(wall) == 1:
        source = wall.iloc[0]
        for scenario, count in config["resource_extrapolation_event_counts"].items():
            for quantile in ("p50", "p90", "maximum"):
                seconds = float(count) * float(source[quantile])
                extrapolation.append(
                    {
                        "scenario": scenario,
                        "n_events": int(count),
                        "per_event_statistic": quantile,
                        "serial_event_equivalent_seconds": seconds,
                        "serial_event_equivalent_days": seconds / 86400.0,
                    }
                )
    pd.DataFrame(extrapolation).to_csv(
        output / "summaries/pe_pilot_resource_extrapolation.csv", index=False
    )
    expected_smoke = int(config["pilot_design"]["stage_0_smoke_cases"])
    all_quality = bool(
        len(table) >= expected_smoke
        and table.sampler_converged.fillna(False).astype(bool).all()
        and table.map_valid.fillna(False).astype(bool).all()
    )
    status = "SMOKE_PASS_EXPANSION_ELIGIBLE" if all_quality else "SMOKE_INCOMPLETE_OR_FAILED"
    payload = {
        "status": status,
        "generated_at_utc": utc_now(),
        "completed_cases": int(len(table)),
        "expected_smoke_cases": expected_smoke,
        "sampler_converged_cases": int(table.sampler_converged.fillna(False).astype(bool).sum()),
        "map_valid_cases": int(table.map_valid.fillna(False).astype(bool).sum()),
        "historical_results_modified": False,
        "real_candidate_rerank_performed": False,
        "background_scope": "exploratory_calibration_only",
        "raw_nside_convergence_blocking": False,
        "terminal_boundary": config["terminal_status"],
    }
    write_json(output / "summaries/pe_pilot_summary.json", payload)
    write_json(output / "STATUS.json", payload)


def find_case(output: Path, case_id: str) -> pd.Series:
    combined = output / "manifests/pe_pilot_cases_all.csv"
    paths = [combined] if combined.exists() else sorted((output / "manifests").glob("pe_pilot_cases_*.csv"))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No pilot case manifests")
    frame = pd.concat(frames, ignore_index=True)
    selected = frame[frame.case_id.eq(case_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one case {case_id}, found {len(selected)}")
    return selected.iloc[0]


def run_case(output: Path, case_id: str, sampler_seed: int) -> None:
    config = json.loads((output / "contracts/V10P1_PE_PILOT_CONFIG.json").read_text(encoding="utf-8"))
    case = find_case(output, case_id)
    target = output / "cases" / f"{case_id}_seed{sampler_seed}"
    failure_path = output / "logs" / f"{case_id}_seed{sampler_seed}.failure.json"
    try:
        if str(case.event_role) == "real":
            metrics = run_real_case(case, target, config, sampler_seed)
        else:
            metrics = run_injection_case(case, target, config, sampler_seed)
        map_metrics = (
            json.loads((target / "sky_map_metrics.json").read_text(encoding="utf-8"))
            if (target / "sky_map_metrics.json").exists()
            else build_and_audit_sky_map(target, config, case, sampler_seed)
        )
        write_json(
            output / "logs" / f"{case_id}_seed{sampler_seed}.complete.json",
            {"pe": metrics, "sky_map": map_metrics},
        )
    except Exception as exc:
        write_json(
            failure_path,
            {
                "case_id": case_id,
                "sampler_seed": sampler_seed,
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser = sub.add_parser("run-case")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--case-id", required=True)
    run_parser.add_argument("--sampler-seed", type=int, default=202608231)
    map_parser = sub.add_parser("postprocess-case")
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.add_argument("--case-id", required=True)
    map_parser.add_argument("--sampler-seed", type=int, default=202608231)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve(), args.config.resolve())
    elif args.command == "run-case":
        run_case(args.output.resolve(), args.case_id, args.sampler_seed)
    elif args.command == "postprocess-case":
        postprocess_case(args.output.resolve(), args.case_id, args.sampler_seed)
    elif args.command == "summarize":
        summarize(args.output.resolve())
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
