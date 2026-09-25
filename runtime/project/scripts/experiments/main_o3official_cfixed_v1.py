#!/usr/bin/env python3
"""MAIN-O3OFFICIAL-CFIX-v1: official O3 scope expansion of C-fixed.

This independent analysis expands the real O3 catalog from the historical
63-event project scope to the 70-event LVK full-O3 PO/ML scope.  It freezes
the BAYESTAR C-fixed waveform encoder, waveform calibration, one-dimensional
time calibration, corrected NESTED-to-RING sky statistic, Nside=512, and the
three per-seed fusion weights.  No real PE/FPP/follow-up result is used for
selection.

The script is resumable and never writes into historical result directories.
Large strain and PE files remain in ``cache`` and are excluded from delivery.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Must be set before importing the frozen v7 model stack.
os.environ["GW_WAVEFORM_INPUT_SAMPLES"] = "4096"

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.experiments import gwtc_c_scheme_ordering_confirmation_20260831 as cbase
from scripts.real_search import common as real_common


EXPERIMENT_CODE = "MAIN-O3OFFICIAL-CFIX-v1"
FILESYSTEM_VERSION = "main_o3official_cfixed_v1_20260904"
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
MODEL_SEEDS = (202607241, 202607242, 202607243)
ANALYSIS_NSIDE = 512
PE_SAMPLE_LIMIT = 30_000
SKY_BF_FLOOR = 1e-30
BASELINE_ROOT = PROJECT / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
BASELINE_PACKAGE = PROJECT / "packages/bayestar_injection_sky_pe_20260901_20260901T102000Z_deliverables.tar.gz"
BASELINE_PACKAGE_SHA256 = "3ea3cf13992a5154e0dd757ab6087b73344cf28eb0ccf5e4a8a903b0e83850af"
V7_ROOT = PROJECT / "results/real_noise_injection_v7_peak2s_formal_20260722/gwtc3"
OLD_RUN = PROJECT / "runs/real_gwtc_lensing_search_20260625"
OFFICIAL_FPP_URL = (
    "https://zenodo.org/api/records/7693837/files/"
    "O3b_lensing_datafile_Figure1_ML_PO_FPPs.csv/content"
)
OFFICIAL_TABLE1_URL = (
    "https://zenodo.org/api/records/7693837/files/"
    "o3b_lensing_datafile_Table1.csv/content"
)
GWOSC_CATALOGS = ("GWTC-2.1-confident", "GWTC-3-confident")

FROZEN_WEIGHTS = {
    202607241: {"waveform": 0.50, "time": 0.25, "sky": 0.50},
    202607242: {"waveform": 1.00, "time": 0.25, "sky": 0.50},
    202607243: {"waveform": 1.00, "time": 0.25, "sky": 0.50},
}

# Pair labels visible in the official Hanabi Bayes-factor figure in the O3
# lensing-search paper source (arXiv:2304.08393).  The paper reports B_L/U < 1
# for every displayed pair.  This is an audit label, never a ranking input.
HANABI_FIGURE_PAIRS = (
    ("GW190413_134308", "GW191109_010717"),
    ("GW190413_052954", "GW200209_085452"),
    ("GW190413_052954", "GW200219_094415"),
    ("GW190421_213856", "GW191222_033537"),
    ("GW190527_092055", "GW190719_215514"),
    ("GW190602_175927", "GW191230_180458"),
    ("GW190620_030421", "GW200216_220804"),
    ("GW190701_203306", "GW200220_124850"),
    ("GW190803_022701", "GW200219_094415"),
    ("GW190805_211137", "GW190916_200658"),
    ("GW190929_012149", "GW200216_220804"),
    ("GW190930_133541", "GW191105_143521"),
    ("GW191103_012549", "GW191105_143521"),
    ("GW191222_033537", "GW200128_022011"),
)

# Pair-resolved Hanabi values published in the preceding O3a lensing paper.
# Only entries whose two events also belong to the present official-70 scope
# are retained.  They are external audit labels and never ranking inputs.
O3A_HANABI_LOG10_BF = {
    ("GW190412", "GW190708_232457"): -9.7,
    ("GW190421_213856", "GW190910_112807"): -3.8,
    ("GW190513_205428", "GW190630_185205"): -5.5,
    ("GW190706_222641", "GW190719_215514"): -3.4,
    ("GW190707_093326", "GW190930_133541"): -12.5,
    ("GW190719_215514", "GW190915_235702"): -3.8,
    ("GW190720_000836", "GW190728_064510"): -9.8,
    ("GW190720_000836", "GW190930_133541"): -12.3,
    ("GW190728_064510", "GW190930_133541"): -11.6,
    ("GW190421_213856", "GW190731_140936"): -3.3,
    ("GW190727_060333", "GW190910_112807"): -4.5,
    ("GW190731_140936", "GW190803_022701"): -4.0,
    ("GW190731_140936", "GW190910_112807"): -4.3,
    ("GW190803_022701", "GW190910_112807"): -3.2,
}


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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_pair(left: object, right: object) -> str:
    a, b = str(left), str(right)
    return f"{a}--{b}" if a <= b else f"{b}--{a}"


def ensure_dirs(root: Path) -> None:
    for name in (
        "contracts",
        "data",
        "results/full70",
        "results/strict_h1l1",
        "results/injection_control",
        "figures",
        "reports",
        "scripts",
        "logs",
        "manifests",
        "cache/catalog_json",
        "cache/event_json",
        "cache/pe_h5_cosmo",
        "cache/real_strain",
        "cache/source_run/data",
        "cache/preprocessed",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def curl_download(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return {"status": "already_exists", "bytes": destination.stat().st_size}
    partial = destination.with_suffix(destination.suffix + ".part")
    aria2 = shutil.which("aria2c")
    if aria2:
        command = [
            aria2,
            "--continue=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=1M",
            "--max-tries=0",
            "--retry-wait=5",
            "--summary-interval=0",
            "--console-log-level=warn",
            f"--dir={partial.parent}",
            f"--out={partial.name}",
            url,
        ]
    else:
        command = [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "10",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "-C",
            "-",
            "-o",
            str(partial),
            url,
        ]
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-500:],
            "elapsed_s": time.time() - started,
        }
    partial.replace(destination)
    return {
        "status": "downloaded",
        "bytes": destination.stat().st_size,
        "elapsed_s": time.time() - started,
    }


def download_many(tasks: list[tuple[str, Path, dict[str, Any]]], workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending = [(url, path, meta) for url, path, meta in tasks if not path.is_file()]
    for url, path, meta in tasks:
        if path.is_file():
            rows.append({**meta, "url": url, "path": str(path), "status": "reused", "bytes": path.stat().st_size})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(curl_download, url, path): (url, path, meta)
            for url, path, meta in pending
        }
        for future in as_completed(futures):
            url, path, meta = futures[future]
            result = future.result()
            row = {**meta, "url": url, "path": str(path), **result}
            rows.append(row)
            print(f"[download] {meta.get('kind')} {meta.get('event_name')} {meta.get('detector', '')}: {result['status']}", flush=True)
    return rows


def fetch_small(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size:
        return
    result = curl_download(url, path)
    if result["status"] == "failed":
        raise RuntimeError(f"Failed to download {url}: {result}")


def baseline_hashes() -> dict[str, str]:
    paths = {
        "baseline_package": BASELINE_PACKAGE,
        "selected_config": BASELINE_ROOT / "contracts/selected_config.json",
        "locked_test_metrics": BASELINE_ROOT / "results/locked_test_retrieval_metrics_per_seed.csv",
        "historical_real_consensus": BASELINE_ROOT / "results/gwtc3/real_consensus_with_pe_official_C_fixed.parquet",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def verify_baseline(root: Path) -> dict[str, Any]:
    if not BASELINE_ROOT.is_dir() or not BASELINE_PACKAGE.is_file():
        raise FileNotFoundError("Frozen C-fixed baseline inputs are missing")
    observed = baseline_hashes()
    if observed["baseline_package"] != BASELINE_PACKAGE_SHA256:
        raise RuntimeError("C-fixed package SHA-256 does not match the frozen contract")
    metrics = pd.read_csv(BASELINE_ROOT / "results/locked_test_retrieval_metrics_summary.csv")
    fingerprints: dict[str, float] = {}
    for deployment, expected_waveform, expected_cfixed in (
        ("gwtc3", 0.555, 0.865),
        ("gwtc4", 0.474, 0.777),
    ):
        part = metrics[metrics["deployment"].astype(str) == deployment]
        for method, expected in (("waveform_only", expected_waveform), ("BAYESTAR_C_fixed", expected_cfixed)):
            row = part[part["method"] == method]
            if row.empty:
                raise RuntimeError(f"Missing baseline fingerprint {deployment}/{method}")
            value = float(row.iloc[0]["overall_r_at_10_mean"])
            if abs(value - expected) > 0.0015:
                raise RuntimeError(f"Baseline fingerprint changed: {deployment}/{method}={value}")
            fingerprints[f"{deployment}_{method}_r10"] = value
    payload = {
        "passed": True,
        "experiment_code": EXPERIMENT_CODE,
        "parent_version": "BAYESTAR-CFIX-63-v1",
        "historical_strict_version": "HIST-CFIX-53-v1",
        "hashes": observed,
        "fingerprints": fingerprints,
        "checked_at_utc": utc_stamp(),
    }
    write_json(root / "contracts/BASELINE_IDENTITY.json", payload)
    write_json(root / "contracts/HISTORICAL_HASHES_BEFORE.json", observed)
    return payload


def load_catalog_records(root: Path, workers: int, requested_events: Iterable[str]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for catalog in GWOSC_CATALOGS:
        path = root / "cache/catalog_json" / f"{catalog}.json"
        fetch_small(f"https://gwosc.org/eventapi/json/{catalog}", path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload["events"].values():
            name = str(event["commonName"])
            current = summaries.get(name)
            if current is None or int(event.get("version", 0)) > int(current.get("version", 0)):
                summaries[name] = event
    requested = sorted(set(map(str, requested_events)))
    absent = sorted(set(requested) - set(summaries))
    if absent:
        raise RuntimeError(f"Official events absent from GWOSC catalog summaries: {absent}")
    tasks: list[tuple[str, Path, dict[str, Any]]] = []
    for name in requested:
        event = summaries[name]
        tasks.append(
            (
                str(event["jsonurl"]),
                root / "cache/event_json" / f"{name}.json",
                {"kind": "event_json", "event_name": name},
            )
        )
    downloaded = download_many(tasks, min(max(workers, 1), 12))
    failed = [row for row in downloaded if row["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Event-level GWOSC JSON downloads failed: {failed}")
    records: dict[str, dict[str, Any]] = {}
    for name in requested:
        payload = json.loads((root / "cache/event_json" / f"{name}.json").read_text(encoding="utf-8"))
        values = list(payload.get("events", {}).values())
        if len(values) != 1:
            raise RuntimeError(f"Unexpected event-level JSON for {name}: {len(values)} records")
        records[name] = values[0]
    return records


def choose_pe(record: dict[str, Any]) -> tuple[str, str]:
    candidates: list[tuple[tuple[int, int, str], str, str]] = []
    for name, values in record.get("parameters", {}).items():
        url = str(values.get("data_url", ""))
        if values.get("pipeline_type") != "pe" or ".h5" not in url:
            continue
        match = re.search(r"/([^/?]+\.h5)(?:/content|\?[^/]*)?$", url)
        if match is None:
            continue
        basename = match.group(1)
        score = (
            int(bool(values.get("is_preferred"))),
            int("mixed_cosmo" in basename),
            str(values.get("date_added", "")),
        )
        candidates.append((score, url, name))
    if not candidates:
        raise RuntimeError(f"No PE HDF5 for {record.get('commonName')}")
    _, url, parameter_name = max(candidates, key=lambda item: item[0])
    return url, parameter_name


def download_basename(url: str) -> str:
    match = re.search(r"/([^/?]+)(?:/content|\?[^/]*)?$", str(url))
    if match is None:
        raise ValueError(f"Cannot determine download filename from {url}")
    return match.group(1)


def choose_strain(record: dict[str, Any], detector: str) -> dict[str, Any] | None:
    choices = [
        row
        for row in record.get("strain", [])
        if str(row.get("detector")) == detector
        and str(row.get("format")) == "hdf5"
        and int(row.get("sampling_rate", 0)) == 4096
        and int(row.get("duration", 0)) == 4096
    ]
    if not choices:
        return None
    return sorted(choices, key=lambda row: str(row.get("url", "")))[-1]


def infer_run(gps: float) -> str:
    return "O3a" if gps < 1_256_655_618 else "O3b"


def existing_pe_map() -> dict[str, Path]:
    output: dict[str, Path] = {}
    old = pd.read_csv(OLD_RUN / "data/event_manifest.csv")
    for _, row in old.iterrows():
        raw = Path(str(row.get("sky_map_path", "")))
        candidates = (raw, PROJECT / raw, OLD_RUN / raw)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            output[str(row["event_name"])] = path.resolve()
    return output


def existing_strain_map() -> dict[tuple[str, str], Path]:
    output: dict[tuple[str, str], Path] = {}
    manifest = pd.read_csv(OLD_RUN / "data/strain_gwosc_download_manifest.csv")
    for _, row in manifest.iterrows():
        raw = Path(str(row.get("local_path", "")))
        if not str(raw) or str(raw) == ".":
            continue
        path = raw if raw.is_absolute() else OLD_RUN / raw
        if path.is_file():
            output[(str(row["event_name"]), str(row["detector"]))] = path.resolve()
    return output


def prepare(root: Path, workers: int) -> None:
    ensure_dirs(root)
    baseline = verify_baseline(root)
    fpp_path = root / "data/O3b_lensing_datafile_Figure1_ML_PO_FPPs.csv"
    table1_path = root / "data/o3b_lensing_datafile_Table1.csv"
    fetch_small(OFFICIAL_FPP_URL, fpp_path)
    fetch_small(OFFICIAL_TABLE1_URL, table1_path)
    fpp = pd.read_csv(fpp_path)
    official_events = sorted(set(fpp["event1"].astype(str)) | set(fpp["event2"].astype(str)))
    if len(fpp) != 2415 or len(official_events) != 70:
        raise RuntimeError(f"Official scope mismatch: pairs={len(fpp)}, events={len(official_events)}")
    records = load_catalog_records(root, workers, official_events)
    missing_records = sorted(set(official_events) - set(records))
    if missing_records:
        raise RuntimeError(f"GWOSC catalog JSON misses official events: {missing_records}")

    old_pe = existing_pe_map()
    old_strain = existing_strain_map()
    pe_tasks: list[tuple[str, Path, dict[str, Any]]] = []
    strain_tasks: list[tuple[str, Path, dict[str, Any]]] = []
    event_rows: list[dict[str, Any]] = []
    strain_plan: list[dict[str, Any]] = []
    for name in official_events:
        record = records[name]
        gps = float(record["GPS"])
        pe_url, pe_parameter = choose_pe(record)
        pe_basename = download_basename(pe_url)
        pe_path = old_pe.get(name)
        if pe_path is None or not pe_path.is_file():
            pe_path = root / "cache/pe_h5_cosmo" / pe_basename
            pe_tasks.append((pe_url, pe_path, {"kind": "pe", "event_name": name}))
        detectors = sorted({str(row.get("detector")) for row in record.get("strain", [])})
        nominal_h1l1 = {"H1", "L1"}.issubset(detectors)
        event_rows.append(
            {
                "event_name": name,
                "catalog": str(record.get("catalog.shortName", "")),
                "sample_role": "official_full_o3_po_ml_scope",
                "include_in_primary_search": True,
                "run": infer_run(gps),
                "gps_time": gps,
                "detectors_available": ",".join(detectors),
                "nominal_h1l1_network": nominal_h1l1,
                "network_snr": record.get("network_matched_filter_snr", np.nan),
                "far": record.get("far", np.nan),
                "catalog_confidence": str(record.get("catalog.shortName", "")),
                "chirp_mass": record.get("chirp_mass", np.nan),
                "mass_1": np.nan,
                "mass_2": np.nan,
                "mass_ratio": np.nan,
                "luminosity_distance": record.get("luminosity_distance", np.nan),
                "p_astro": record.get("p_astro", np.nan),
                "sky_map_path": str(pe_path.resolve()),
                "sky_map_format": "pe_hdf5_skymap_data",
                "sky_map_internal_group": "",
                "sky_map_available": True,
                "sky_map_source": "GWTC public PE HDF5 /skymap/data",
                "pe_release_url": pe_url,
                "pe_parameter_record": pe_parameter,
                "object_class": "BBH",
                "is_ood_for_bbh_encoder": False,
                "notes": "LVK full-O3 PO/ML 70-event scope; no extra SNR cut.",
            }
        )
        for detector in ("H1", "L1"):
            source = choose_strain(record, detector)
            existing = old_strain.get((name, detector))
            if source is None:
                strain_plan.append(
                    {
                        "event_name": name,
                        "detector": detector,
                        "gps_time": gps,
                        "url": "",
                        "gwosc_start": np.nan,
                        "gwosc_duration": np.nan,
                        "covers_requested_window": False,
                        "local_path": "",
                        "download_status": "not_in_public_event_network",
                    }
                )
                continue
            if existing is not None and existing.is_file():
                destination = existing
                status = "reused_historical_4096s"
            else:
                destination = root / "cache/real_strain" / name / Path(str(source["url"])).name
                status = "pending_download"
                strain_tasks.append(
                    (
                        str(source["url"]),
                        destination,
                        {"kind": "strain", "event_name": name, "detector": detector},
                    )
                )
            strain_plan.append(
                {
                    "event_name": name,
                    "detector": detector,
                    "gps_time": gps,
                    "url": str(source["url"]),
                    "gwosc_start": int(source["GPSstart"]),
                    "gwosc_duration": int(source["duration"]),
                    "covers_requested_window": bool(
                        float(source["GPSstart"]) <= gps - 128
                        and float(source["GPSstart"]) + float(source["duration"]) >= gps + 128
                    ),
                    "local_path": str(destination.resolve()),
                    "download_status": status,
                }
            )

    print(f"[prepare] PE downloads required: {len(pe_tasks)}", flush=True)
    pe_downloads = download_many(pe_tasks, min(workers, 4))
    print(f"[prepare] strain downloads required: {len(strain_tasks)}", flush=True)
    strain_downloads = download_many(strain_tasks, min(workers, 8))
    downloads = pd.DataFrame(pe_downloads + strain_downloads)
    write_csv(root / "manifests/DOWNLOAD_LOG.csv", downloads)
    failed = downloads[downloads["status"] == "failed"] if not downloads.empty else downloads
    if len(failed):
        raise RuntimeError(f"{len(failed)} downloads failed; rerun prepare to resume")

    events = pd.DataFrame(event_rows).sort_values("gps_time").reset_index(drop=True)
    groups: list[str] = []
    ordering_rows: list[dict[str, Any]] = []
    for index, row in events.iterrows():
        path = Path(row["sky_map_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        group = real_common.choose_h5_skymap_group(path)
        if group is None:
            raise RuntimeError(f"No skymap group: {path}")
        groups.append(group)
        probe = row.copy()
        probe["deployment"] = "gwtc3"
        probe["sky_map_internal_group"] = group
        _, metadata = cbase.corrected_read_probability_map(probe, target_nside=32)
        ordering_rows.append({"idx": index, "event_name": row["event_name"], **metadata})
    events["sky_map_internal_group"] = groups
    source_data = root / "cache/source_run/data"
    events.to_csv(source_data / "event_manifest.csv", index=False)
    write_csv(root / "data/official70_event_manifest.csv", events)

    strain = pd.DataFrame(strain_plan)
    for index, row in strain.iterrows():
        path = Path(str(row["local_path"]))
        if path.is_file():
            strain.loc[index, "download_status"] = (
                "reused_historical_4096s" if str(path).startswith(str(OLD_RUN)) else "downloaded_or_cached"
            )
    strain.to_csv(source_data / "strain_gwosc_download_manifest.csv", index=False)
    write_csv(root / "data/official70_strain_manifest.csv", strain)
    write_csv(root / "data/public_pe_ordering_audit.csv", pd.DataFrame(ordering_rows))

    input_rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        path = Path(str(row["sky_map_path"]))
        input_rows.append(
            {
                "input_role": "public_pe_hdf5",
                "event_name": row["event_name"],
                "detector": "",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": row["pe_release_url"],
            }
        )
    for _, row in strain.iterrows():
        path_text = str(row["local_path"])
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_file():
            continue
        input_rows.append(
            {
                "input_role": "public_gwosc_strain_4096s_4096hz",
                "event_name": row["event_name"],
                "detector": row["detector"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": row["url"],
            }
        )
    write_csv(root / "manifests/INPUT_SHA256_MANIFEST.csv", pd.DataFrame(input_rows))

    fpp_clean = fpp.rename(columns={"ML-FPP": "official_ml_fpp", "PO-FPP": "official_po_fpp"}).copy()
    fpp_clean["pair_key"] = [canonical_pair(a, b) for a, b in zip(fpp_clean["event1"], fpp_clean["event2"])]
    fpp_clean["official_po_or_ml_fpp_below_0p01"] = (
        (fpp_clean["official_ml_fpp"] < 0.01) | (fpp_clean["official_po_fpp"] < 0.01)
    )
    write_csv(root / "data/official_po_ml_fpp_all_2415_pairs.csv", fpp_clean)

    official_set = set(official_events)
    bad_hanabi = [pair for pair in HANABI_FIGURE_PAIRS if not set(pair).issubset(official_set)]
    if bad_hanabi:
        raise RuntimeError(f"Hanabi figure names not in official scope: {bad_hanabi}")
    followup = fpp_clean[["pair_key", "official_ml_fpp", "official_po_fpp", "official_po_or_ml_fpp_below_0p01"]].copy()
    hanabi_keys = {canonical_pair(a, b) for a, b in HANABI_FIGURE_PAIRS}
    followup["official_hanabi_figure_overlap"] = followup["pair_key"].isin(hanabi_keys)
    followup["official_hanabi_conclusion"] = np.where(
        followup["official_hanabi_figure_overlap"],
        "analyzed; all displayed pairs have B_L/U<1; lensing not supported",
        "not displayed in the pair-resolved Hanabi figure",
    )
    o3a_hanabi = {canonical_pair(a, b): value for (a, b), value in O3A_HANABI_LOG10_BF.items()}
    followup["official_o3a_hanabi_table_overlap"] = followup["pair_key"].isin(o3a_hanabi)
    followup["official_o3a_hanabi_log10_b_lu"] = followup["pair_key"].map(o3a_hanabi)
    followup["official_any_pair_resolved_hanabi_overlap"] = (
        followup["official_hanabi_figure_overlap"] | followup["official_o3a_hanabi_table_overlap"]
    )
    followup["official_golum_stage"] = np.where(
        followup["official_hanabi_figure_overlap"],
        "necessarily passed through GOLUM follow-up before Hanabi",
        "pair membership not resolved by the public machine-readable table; paper reports 75 GOLUM pairs in aggregate",
    )
    followup["official_screening_stage"] = np.where(
        followup["official_po_or_ml_fpp_below_0p01"],
        "machine-readable PO/ML screen below 1% in at least one method",
        "not below 1% in the machine-readable PO/ML table",
    )
    write_csv(root / "data/official_followup_stage_contract.csv", followup)
    write_json(
        root / "contracts/OFFICIAL_FOLLOWUP_PROVENANCE.json",
        {
            "full_o3_paper": "https://arxiv.org/abs/2304.08393",
            "full_o3_machine_data": "https://zenodo.org/records/7693837",
            "po_ml_pair_rows": len(fpp_clean),
            "golum_pairs_reported_in_aggregate": 75,
            "hanabi_pairs_reported_in_aggregate": 17,
            "full_o3_hanabi_pair_labels_resolvable_from_public_figure": len(hanabi_keys),
            "full_o3_hanabi_result": "all reported B_L/U < 1; no lensing support",
            "preceding_o3a_paper": "https://arxiv.org/abs/2105.06384",
            "o3a_pair_resolved_hanabi_rows_within_current_official70": len(o3a_hanabi),
            "important_limitation": (
                "The public 2415-pair machine table does not identify the complete 75 GOLUM "
                "or 17 Hanabi memberships. Aggregate counts are not expanded into invented pair labels."
            ),
            "used_for_ranking_or_weight_selection": False,
        },
    )

    contract = {
        "experiment_code": EXPERIMENT_CODE,
        "filesystem_version": FILESYSTEM_VERSION,
        "created_at_utc": utc_stamp(),
        "parent": baseline,
        "scope": {
            "official_events": 70,
            "official_unordered_pairs": 2415,
            "nominal_h1l1_events": int(events["nominal_h1l1_network"].sum()),
            "strict_h1l1_definition": "both 4096-s public H1/L1 files pass the frozen C-fixed finite-window, off-source PSD, and preprocessing audit",
            "extra_snr_cut": None,
        },
        "frozen": {
            "waveform_encoder": True,
            "waveform_calibration": True,
            "time_likelihood_ratio": True,
            "sky_statistic": "log[Npix sum_k P_i(k)P_j(k)]",
            "analysis_nside": ANALYSIS_NSIDE,
            "sky_ordering": "public PE NESTED maps explicitly converted to RING",
            "injection_sky": "existing event-level BAYESTAR locked test, not regenerated",
            "weights_by_seed": FROZEN_WEIGHTS,
            "real_candidate_information_used_for_selection": False,
        },
        "output_scopes": {
            "full70": "all 70 official events; unavailable waveform contributes exactly zero",
            "strict_h1l1": "only actual preprocessing-passing H1+L1 BBH events",
        },
        "claim_boundary": "candidate refinement/shortlisting only; not lensing detection",
        "final_status_required": FINAL_STATUS,
    }
    write_json(root / "contracts/ANALYSIS_CONTRACT.json", contract)
    write_json(
        root / "contracts/PREPARE_COMPLETE.json",
        {
            "passed": True,
            "timestamp_utc": utc_stamp(),
            "n_events": len(events),
            "n_pairs": len(fpp_clean),
            "nominal_h1l1_events": int(events["nominal_h1l1_network"].sum()),
            "nested_input_maps": int((pd.DataFrame(ordering_rows)["source_ordering"] == "NESTED").sum()),
        },
    )


def hpd_mask(probability: np.ndarray, mass: float) -> np.ndarray:
    order = np.argsort(probability)[::-1]
    count = int(np.searchsorted(np.cumsum(probability[order], dtype=np.float64), mass, side="left")) + 1
    mask = np.zeros(len(probability), dtype=bool)
    mask[order[:count]] = True
    return mask


def map_descriptor(probability: np.ndarray) -> dict[str, float]:
    values = np.asarray(probability, dtype=np.float64)
    area = hp.nside2pixarea(hp.npix2nside(len(values)), degrees=True)
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    n50 = int(np.searchsorted(cumulative, 0.5, side="left")) + 1
    n90 = int(np.searchsorted(cumulative, 0.9, side="left")) + 1
    entropy = -float(np.sum(values * np.log(np.maximum(values, 1e-300))))
    return {
        "area50_deg2": n50 * area,
        "area90_deg2": n90 * area,
        "entropy_nats": entropy,
        "kl_from_isotropic_nats": math.log(len(values)) - entropy,
    }


def build_sky_features(root: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_path = root / "results/real_pair_sky_features_corrected_nside512.parquet"
    event_path = root / "results/real_event_sky_features_corrected_nside512.csv"
    if pair_path.is_file() and event_path.is_file():
        return pd.read_parquet(pair_path), pd.read_csv(event_path)
    maps: list[np.ndarray] = []
    event_rows: list[dict[str, Any]] = []
    mask50: list[np.ndarray] = []
    mask90: list[np.ndarray] = []
    for index, row in manifest.iterrows():
        probe = row.copy()
        probe["deployment"] = "gwtc3"
        probability, metadata = cbase.corrected_read_probability_map(probe, ANALYSIS_NSIDE)
        maps.append(probability)
        mask50.append(hpd_mask(probability, 0.5))
        mask90.append(hpd_mask(probability, 0.9))
        event_rows.append(
            {
                "idx": index,
                "event_name": row["event_name"],
                **map_descriptor(probability),
                **metadata,
            }
        )
        print(f"[sky] {index + 1}/{len(manifest)} {row['event_name']}", flush=True)
    bank = np.stack(maps).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(bank.astype(np.float64, copy=False)).to(device)
    gram = (tensor @ tensor.T).cpu().numpy()
    sqrt_tensor = torch.sqrt(torch.clamp(tensor, min=0.0))
    bc = (sqrt_tensor @ sqrt_tensor.T).cpu().numpy()
    del tensor, sqrt_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ii, jj = np.triu_indices(len(manifest), k=1)
    m50 = np.stack(mask50)
    m90 = np.stack(mask90)
    j50 = np.asarray(
        [np.logical_and(m50[i], m50[j]).sum() / max(np.logical_or(m50[i], m50[j]).sum(), 1) for i, j in zip(ii, jj)],
        dtype=np.float64,
    )
    j90 = np.asarray(
        [np.logical_and(m90[i], m90[j]).sum() / max(np.logical_or(m90[i], m90[j]).sum(), 1) for i, j in zip(ii, jj)],
        dtype=np.float64,
    )
    npix = hp.nside2npix(ANALYSIS_NSIDE)
    sky_bf = npix * gram[ii, jj]
    event_names = manifest["event_name"].astype(str).to_numpy()
    pair = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": event_names[ii],
            "event_j": event_names[jj],
            "pair_key": [canonical_pair(a, b) for a, b in zip(event_names[ii], event_names[jj])],
            "sky_raw_overlap": gram[ii, jj],
            "sky_bayes_factor": sky_bf,
            "sky_raw_log_bf": np.log(np.maximum(sky_bf, SKY_BF_FLOOR)),
            "sky_score": np.log(np.maximum(sky_bf, SKY_BF_FLOOR)),
            "sky_bc": np.clip(bc[ii, jj], 0.0, 1.0),
            "sky_j50": j50,
            "sky_j90": j90,
            "common_nside": ANALYSIS_NSIDE,
        }
    )
    pair.to_parquet(pair_path, index=False)
    write_csv(event_path, pd.DataFrame(event_rows))
    del bank, maps, mask50, mask90, m50, m90
    gc.collect()
    return pair, pd.DataFrame(event_rows)


def checkpoint(seed: int) -> Path:
    return V7_ROOT / f"seed_{seed}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt"


def build_waveform_pairs(
    root: Path,
    manifest: pd.DataFrame,
    inputs: np.ndarray,
    preprocessing: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seed_root = root / f"results/seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    pair_path = seed_root / "waveform_pair_scores.parquet"
    emb_path = seed_root / "event_embeddings.parquet"
    audit_path = seed_root / "waveform_deployment_audit.json"
    if pair_path.is_file() and emb_path.is_file() and audit_path.is_file():
        return pd.read_parquet(pair_path), pd.read_parquet(emb_path), json.loads(audit_path.read_text())
    available = preprocessing["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    model, payload = cbase.v7.load_unified_model(checkpoint(seed))
    embedded, prediction_std = cbase.v7.embed_catalog(
        model,
        cbase.v7.ArrayCatalog(np.asarray(inputs[available])),
        batch_size=16,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dimension = embedded.shape[1]
    all_embedding = np.full((len(manifest), dimension), np.nan, dtype=np.float32)
    all_prediction = np.full((len(manifest), 2), np.nan, dtype=np.float32)
    all_embedding[available] = embedded
    all_prediction[available] = prediction_std
    target_mean = np.asarray(payload["target_mean"], dtype=np.float64)
    target_std = np.asarray(payload["target_std"], dtype=np.float64)
    physical = all_prediction * target_std + target_mean
    events = preprocessing.copy()
    events["unified_embedding"] = [row.tolist() if np.isfinite(row).all() else None for row in all_embedding]
    events["waveform_pred_chirp_mass_detector"] = np.exp(physical[:, 0])
    events["waveform_pred_mass_ratio"] = 1.0 / (1.0 + np.exp(-physical[:, 1]))
    events.to_parquet(emb_path, index=False)

    ii, jj = np.triu_indices(len(manifest), k=1)
    both = available[ii] & available[jj]
    cosine = np.full(len(ii), np.nan, dtype=np.float32)
    delta_mc = np.full(len(ii), np.nan, dtype=np.float32)
    delta_q = np.full(len(ii), np.nan, dtype=np.float32)
    cosine[both] = np.sum(all_embedding[ii[both]] * all_embedding[jj[both]], axis=1)
    delta_mc[both] = np.abs(all_prediction[ii[both], 0] - all_prediction[jj[both], 0])
    delta_q[both] = np.abs(all_prediction[ii[both], 1] - all_prediction[jj[both], 1])
    names = manifest["event_name"].astype(str).to_numpy()
    pairs = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": names[ii],
            "event_j": names[jj],
            "pair_key": [canonical_pair(a, b) for a, b in zip(names[ii], names[jj])],
            "waveform_available": both,
            "waveform_embedding_cosine": cosine,
            "waveform_abs_delta_logmc_std": delta_mc,
            "waveform_abs_delta_logitq_std": delta_q,
        }
    )
    config_path = V7_ROOT / f"seed_{seed}/results/waveform_channel_calibration_v7.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scored = cbase.v7.apply_waveform_channel(pairs.loc[both].copy(), config)
    pairs["waveform_composite_raw"] = np.nan
    pairs["waveform_score"] = 0.0
    pairs.loc[both, "waveform_composite_raw"] = scored["waveform_composite_raw"].to_numpy()
    pairs.loc[both, "waveform_score"] = scored["waveform_score"].to_numpy()
    pairs["waveform_neutral_missing"] = ~both
    pairs.to_parquet(pair_path, index=False)
    valid_scores = pairs.loc[both, "waveform_score"].to_numpy(dtype=float)
    audit = {
        "seed": seed,
        "checkpoint": str(checkpoint(seed)),
        "checkpoint_sha256": sha256_file(checkpoint(seed)),
        "waveform_config": str(config_path),
        "waveform_config_sha256": sha256_file(config_path),
        "strict_events": int(available.sum()),
        "strict_pairs": int(both.sum()),
        "embedding_dimension": dimension,
        "embedding_effective_rank": float(cbase.v7.effective_rank(embedded)),
        "waveform_score_std": float(np.std(valid_scores)),
        "silent_zero_fill": False,
        "missing_pair_contribution": 0.0,
    }
    write_json(audit_path, audit)
    return pairs, events, audit


def rank_pairs(frame: pd.DataFrame, seed: int, scope: str) -> pd.DataFrame:
    weights = FROZEN_WEIGHTS[seed]
    ranked = cbase.rank_real(frame, weights, f"{EXPERIMENT_CODE}:{scope}", seed)
    ranked["scope"] = scope
    ranked["weights_waveform"] = weights["waveform"]
    ranked["weights_time"] = weights["time"]
    ranked["weights_sky"] = weights["sky"]
    return ranked


def score(root: Path) -> None:
    if not (root / "contracts/PREPARE_COMPLETE.json").is_file():
        raise RuntimeError("prepare must complete first")
    manifest = pd.read_csv(root / "cache/source_run/data/event_manifest.csv").sort_values("gps_time").reset_index(drop=True)
    source_run = root / "cache/source_run"
    inputs, preprocessing = cbase.v7.v3.build_real_preprocessed_inputs(
        "GWTC3", source_run, root / "cache/preprocessed"
    )
    preprocessing = preprocessing.sort_values("idx").reset_index(drop=True)
    preprocessing["nominal_h1l1_network"] = manifest["nominal_h1l1_network"].to_numpy(dtype=bool)
    preprocessing["network_snr"] = manifest["network_snr"].to_numpy(dtype=float)
    write_csv(root / "results/strict_h1l1_event_audit.csv", preprocessing)
    actual_strict = preprocessing["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    sky, _ = build_sky_features(root, manifest)
    # Preserve the C-fixed v9.3 evidence floor exactly.  Earlier exploratory
    # code used 1e-30; changing only this numerical floor would alter highly
    # disjoint pairs by hundreds of nats without adding physical information.
    frozen_sky_log_bf = np.log(
        np.maximum(sky["sky_bayes_factor"].to_numpy(dtype=np.float64), SKY_BF_FLOOR)
    )
    sky["sky_raw_log_bf"] = frozen_sky_log_bf
    sky["sky_score"] = frozen_sky_log_bf
    sky.to_parquet(root / "results/real_pair_sky_features_corrected_nside512.parquet", index=False)
    time_calibration_path = V7_ROOT / "shared/time_delay_likelihood_ratio.json"
    time_calibration = json.loads(time_calibration_path.read_text(encoding="utf-8"))
    gps = manifest["gps_time"].to_numpy(dtype=float)
    ii = sky["idx_i"].to_numpy(dtype=int)
    jj = sky["idx_j"].to_numpy(dtype=int)
    sky["delta_t_days"] = np.abs(gps[ii] - gps[jj]) / 86400.0
    sky["time_score"] = cbase.v7.apply_time_likelihood_ratio(sky["delta_t_days"], time_calibration)
    sky["network_snr_i_audit_only"] = manifest["network_snr"].to_numpy(dtype=float)[ii]
    sky["network_snr_j_audit_only"] = manifest["network_snr"].to_numpy(dtype=float)[jj]
    sky["nominal_h1l1_pair"] = (
        manifest["nominal_h1l1_network"].to_numpy(dtype=bool)[ii]
        & manifest["nominal_h1l1_network"].to_numpy(dtype=bool)[jj]
    )
    sky["strict_h1l1_bbh_pair"] = actual_strict[ii] & actual_strict[jj]

    full_frames: list[pd.DataFrame] = []
    strict_frames: list[pd.DataFrame] = []
    for seed in MODEL_SEEDS:
        waveform, _, _ = build_waveform_pairs(root, manifest, inputs, preprocessing, seed)
        merge_columns = [
            "idx_i",
            "idx_j",
            "event_i",
            "event_j",
            "pair_key",
            "waveform_available",
            "waveform_embedding_cosine",
            "waveform_abs_delta_logmc_std",
            "waveform_abs_delta_logitq_std",
            "waveform_composite_raw",
            "waveform_score",
            "waveform_neutral_missing",
        ]
        frame = sky.merge(
            waveform[merge_columns],
            on=["idx_i", "idx_j", "event_i", "event_j", "pair_key"],
            how="left",
            validate="one_to_one",
        )
        frame["waveform_score"] = frame["waveform_score"].fillna(0.0)
        frame["waveform_neutral_missing"] = ~frame["waveform_available"].fillna(False)
        full = rank_pairs(frame, seed, "official70_neutral_waveform")
        strict = rank_pairs(frame.loc[frame["strict_h1l1_bbh_pair"]].copy(), seed, "strict_h1l1")
        seed_root = root / f"results/seed_{seed}"
        full.to_parquet(seed_root / "official70_pair_scores_C_fixed.parquet", index=False)
        strict.to_parquet(seed_root / "strict_h1l1_pair_scores_C_fixed.parquet", index=False)
        write_csv(seed_root / "official70_top50_C_fixed.csv", full.head(50))
        write_csv(seed_root / "strict_h1l1_top50_C_fixed.csv", strict.head(50))
        full_frames.append(full)
        strict_frames.append(strict)

    full_consensus = cbase.consensus_real(full_frames, f"{EXPERIMENT_CODE}:official70_neutral_waveform")
    strict_consensus = cbase.consensus_real(strict_frames, f"{EXPERIMENT_CODE}:strict_h1l1")
    pair_audit_columns = [
        "pair_key",
        "delta_t_days",
        "network_snr_i_audit_only",
        "network_snr_j_audit_only",
        "nominal_h1l1_pair",
        "strict_h1l1_bbh_pair",
        "waveform_available",
        "waveform_neutral_missing",
    ]
    pair_audit = full_frames[0][pair_audit_columns].drop_duplicates("pair_key")
    full_consensus = full_consensus.merge(pair_audit, on="pair_key", how="left", validate="one_to_one")
    strict_consensus = strict_consensus.merge(pair_audit, on="pair_key", how="left", validate="one_to_one")
    full_consensus.to_parquet(root / "results/full70/consensus_all_2415_pairs.parquet", index=False)
    strict_consensus.to_parquet(root / "results/strict_h1l1/consensus_all_pairs.parquet", index=False)
    write_csv(root / "results/full70/consensus_top50.csv", full_consensus.head(50))
    write_csv(root / "results/strict_h1l1/consensus_top50.csv", strict_consensus.head(50))

    comparison_rows: list[dict[str, Any]] = []
    for seed, new in zip(MODEL_SEEDS, full_frames):
        old_path = BASELINE_ROOT / f"results/gwtc3/seed_{seed}/real_strict_pair_scores_C_fixed.parquet"
        old = pd.read_parquet(old_path).copy()
        old["pair_key"] = [canonical_pair(a, b) for a, b in zip(old.event_i, old.event_j)]
        common = old.merge(new, on="pair_key", suffixes=("_old", "_new"), how="inner")
        for column in ("waveform_score", "time_score", "sky_raw_log_bf", "final_score"):
            difference = np.abs(common[f"{column}_old"].to_numpy(float) - common[f"{column}_new"].to_numpy(float))
            comparison_rows.append(
                {
                    "seed": seed,
                    "column": column,
                    "common_pairs": len(common),
                    "max_abs_difference": float(np.max(difference)) if len(difference) else np.nan,
                    "median_abs_difference": float(np.median(difference)) if len(difference) else np.nan,
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    write_csv(root / "results/historical_common_pair_reproduction.csv", comparison)
    historical_max_difference = float(comparison["max_abs_difference"].max())
    if historical_max_difference > 1e-4:
        raise RuntimeError(
            "Frozen historical pair scores were not reproduced within 1e-4: "
            f"max_abs_difference={historical_max_difference}"
        )
    injection_metrics = pd.read_csv(BASELINE_ROOT / "results/locked_test_retrieval_metrics_per_seed.csv")
    injection_metrics.to_csv(root / "results/injection_control/frozen_cfixed_locked_test_metrics.csv", index=False)
    injection_summary = pd.read_csv(BASELINE_ROOT / "results/locked_test_retrieval_metrics_summary.csv")
    injection_summary.to_csv(root / "results/injection_control/frozen_cfixed_locked_test_summary.csv", index=False)
    write_json(
        root / "contracts/SCORE_COMPLETE.json",
        {
            "passed": True,
            "timestamp_utc": utc_stamp(),
            "official_events": len(manifest),
            "official_pairs": len(full_consensus),
            "nominal_h1l1_events": int(manifest["nominal_h1l1_network"].sum()),
            "actual_strict_h1l1_events": int(actual_strict.sum()),
            "actual_strict_h1l1_pairs": int(len(strict_consensus)),
            "time_calibration_sha256": sha256_file(time_calibration_path),
            "historical_common_pair_max_abs_difference": historical_max_difference,
        },
    )


def dataset_values(dataset: h5py.Dataset | h5py.Group, name: str, index: np.ndarray) -> np.ndarray:
    if isinstance(dataset, h5py.Dataset):
        fields = set(dataset.dtype.names or ())
        if name in fields:
            return np.asarray(dataset[name][index], dtype=np.float64)
        getter = lambda key: np.asarray(dataset[key][index], dtype=np.float64)
        available = fields
    else:
        available = set(dataset.keys())
        getter = lambda key: np.asarray(dataset[key][index], dtype=np.float64)
    if name == "chirp_mass" and {"mass_1", "mass_2"}.issubset(available):
        m1, m2 = getter("mass_1"), getter("mass_2")
        return (m1 * m2) ** (3 / 5) / np.maximum(m1 + m2, 1e-30) ** (1 / 5)
    if name == "mass_ratio" and {"mass_1", "mass_2"}.issubset(available):
        m1, m2 = getter("mass_1"), getter("mass_2")
        return np.minimum(m1, m2) / np.maximum(m1, m2)
    return np.full(len(index), np.nan)


def load_pe_samples(row: pd.Series) -> dict[str, np.ndarray]:
    path = Path(str(row["sky_map_path"]))
    group = str(row["sky_map_internal_group"])
    with h5py.File(path, "r") as handle:
        posterior = handle[f"{group}/posterior_samples"]
        if isinstance(posterior, h5py.Dataset):
            count = len(posterior)
        else:
            lengths = [len(value) for value in posterior.values() if hasattr(value, "__len__")]
            if not lengths:
                raise RuntimeError(f"No posterior arrays in {path}:{group}")
            count = min(lengths)
        index = np.linspace(0, count - 1, min(count, PE_SAMPLE_LIMIT), dtype=np.int64)
        values = {
            name: dataset_values(posterior, name, index)
            for name in ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance")
        }
    return {name: value[np.isfinite(value)] for name, value in values.items()}


def posterior_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) < 10 or len(right) < 10:
        return {
            "median_i": np.nan,
            "median_j": np.nan,
            "sigma_i": np.nan,
            "sigma_j": np.nan,
            "standardized_distance": np.nan,
            "wasserstein_distance": np.nan,
            "bhattacharyya_coefficient": np.nan,
        }
    median_i, median_j = float(np.median(left)), float(np.median(right))
    q16_i, q84_i = np.quantile(left, [0.16, 0.84])
    q16_j, q84_j = np.quantile(right, [0.16, 0.84])
    sigma_i = float((q84_i - q16_i) / 2)
    sigma_j = float((q84_j - q16_j) / 2)
    scale = max(math.hypot(sigma_i, sigma_j), 1e-12)
    lo = min(float(np.quantile(left, 0.001)), float(np.quantile(right, 0.001)))
    hi = max(float(np.quantile(left, 0.999)), float(np.quantile(right, 0.999)))
    if hi > lo:
        edges = np.linspace(lo, hi, 257)
        px, _ = np.histogram(left, bins=edges)
        py, _ = np.histogram(right, bins=edges)
        px = px / max(px.sum(), 1)
        py = py / max(py.sum(), 1)
        bc = float(np.sum(np.sqrt(px * py)))
    else:
        bc = float(np.isclose(median_i, median_j))
    return {
        "median_i": median_i,
        "median_j": median_j,
        "sigma_i": sigma_i,
        "sigma_j": sigma_j,
        "standardized_distance": abs(median_i - median_j) / scale,
        "wasserstein_distance": float(stats.wasserstein_distance(left, right)),
        "bhattacharyya_coefficient": bc,
    }


def enrich_scope(root: Path, scope: str, consensus: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    top = consensus.head(50).copy()
    needed = sorted(set(top.event_i.astype(str)) | set(top.event_j.astype(str)))
    by_event = manifest.set_index("event_name")
    cache = {event: load_pe_samples(by_event.loc[event]) for event in needed}
    rows: list[dict[str, Any]] = []
    for _, pair in top.iterrows():
        output = pair.to_dict()
        distances: list[float] = []
        for parameter in ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"):
            values = posterior_metrics(cache[str(pair.event_i)][parameter], cache[str(pair.event_j)][parameter])
            prefix = {
                "chirp_mass": "mc",
                "mass_ratio": "q",
                "chi_eff": "chi_eff",
                "luminosity_distance": "dl_app",
            }[parameter]
            for key, value in values.items():
                output[f"pe_{prefix}_{key}"] = value
            if parameter != "luminosity_distance" and np.isfinite(values["standardized_distance"]):
                distances.append(values["standardized_distance"])
        output["pe_dmax_intrinsic"] = max(distances) if distances else np.nan
        output["pe_dmax_intrinsic_le_3"] = bool(distances and max(distances) <= 3)
        rows.append(output)
    enriched = pd.DataFrame(rows)
    fpp = pd.read_csv(root / "data/official_po_ml_fpp_all_2415_pairs.csv")
    followup = pd.read_csv(root / "data/official_followup_stage_contract.csv")
    enriched = enriched.merge(
        fpp[["pair_key", "official_ml_fpp", "official_po_fpp", "pair-type", "official_po_or_ml_fpp_below_0p01"]],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    enriched = enriched.merge(
        followup[
            [
                "pair_key",
                "official_hanabi_figure_overlap",
                "official_hanabi_conclusion",
                "official_o3a_hanabi_table_overlap",
                "official_o3a_hanabi_log10_b_lu",
                "official_any_pair_resolved_hanabi_overlap",
                "official_golum_stage",
                "official_screening_stage",
            ]
        ],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    output_root = root / "results" / scope
    enriched.to_parquet(output_root / "consensus_top50_with_pe_official.parquet", index=False)
    write_csv(output_root / "consensus_top50_with_pe_official.csv", enriched)
    for budget in (10, 20, 50):
        write_csv(output_root / f"consensus_top{budget}_with_pe_official.csv", enriched.head(budget))
    return enriched


def make_figures(root: Path, full: pd.DataFrame, strict: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    colors = {"waveform": "#2B6CB0", "time": "#D97706", "sky": "#138A72"}
    for axis, frame, title in (
        (axes[0, 0], full.head(15), "a  Official 70-event catalog"),
        (axes[0, 1], strict.head(15), "b  Strict H1+L1 subset"),
    ):
        y = np.arange(len(frame))
        for offset, channel, column in (
            (-0.20, "waveform", "waveform_contribution_mean"),
            (0.00, "time", "time_contribution_mean"),
            (0.20, "sky", "sky_contribution_mean"),
        ):
            axis.scatter(
                frame[column],
                y + offset,
                color=colors[channel],
                s=15,
                label=channel.capitalize(),
                zorder=3,
            )
        axis.set_yticks(y, labels=[str(value).replace("--", "\n") for value in frame["pair_key"]], fontsize=5.5)
        axis.invert_yaxis()
        axis.axvline(0, color="black", lw=0.6)
        axis.set_xlabel("Mean weighted contribution")
        axis.set_title(title, loc="left")
    axes[0, 1].legend(loc="lower right", ncol=3, fontsize=7)
    for axis, frame, title in (
        (axes[1, 0], full, "c  PE consistency: official 70"),
        (axes[1, 1], strict, "d  PE consistency: strict H1+L1"),
    ):
        x = frame["pe_mc_bhattacharyya_coefficient"].to_numpy(float)
        y = frame["pe_mc_standardized_distance"].to_numpy(float)
        color = -np.log10(np.maximum(frame["official_po_fpp"].to_numpy(float), 1e-6))
        scatter = axis.scatter(x, y, c=color, cmap="viridis", s=22, edgecolor="white", linewidth=0.3)
        axis.axvline(0.5, color="0.5", ls="--", lw=0.7)
        axis.axhline(3, color="0.5", ls="--", lw=0.7)
        axis.set_xlabel(r"$BC_{\mathcal{M}_c}$")
        axis.set_ylabel(r"$D_{\mathcal{M}_c}$")
        axis.set_title(title, loc="left")
        fig.colorbar(scatter, ax=axis, label=r"$-\log_{10}({\rm PO\ FPP})$")
    for suffix in ("pdf", "png"):
        fig.savefig(root / f"figures/fig_main_o3official_cfixed_v1.{suffix}", dpi=300)
    plt.close(fig)


def audit(root: Path) -> None:
    if not (root / "contracts/SCORE_COMPLETE.json").is_file():
        raise RuntimeError("score must complete first")
    manifest = pd.read_csv(root / "cache/source_run/data/event_manifest.csv")
    full = pd.read_parquet(root / "results/full70/consensus_all_2415_pairs.parquet")
    strict = pd.read_parquet(root / "results/strict_h1l1/consensus_all_pairs.parquet")
    full_enriched = enrich_scope(root, "full70", full, manifest)
    strict_enriched = enrich_scope(root, "strict_h1l1", strict, manifest)
    budget_rows: list[dict[str, Any]] = []
    for scope, frame in (("full70", full_enriched), ("strict_h1l1", strict_enriched)):
        for budget in (10, 20, 50):
            selected = frame.head(budget)
            budget_rows.append(
                {
                    "scope": scope,
                    "budget": budget,
                    "n_pairs": len(selected),
                    "mc_bc_ge_0p5": int((selected["pe_mc_bhattacharyya_coefficient"] >= 0.5).sum()),
                    "median_mc_bc": float(selected["pe_mc_bhattacharyya_coefficient"].median()),
                    "mc_d_le_3": int((selected["pe_mc_standardized_distance"] <= 3).sum()),
                    "intrinsic_dmax_le_3": int(selected["pe_dmax_intrinsic_le_3"].sum()),
                    "official_po_fpp_lt_0p01": int((selected["official_po_fpp"] < 0.01).sum()),
                    "official_ml_fpp_lt_0p01": int((selected["official_ml_fpp"] < 0.01).sum()),
                    "official_either_fpp_lt_0p01": int(selected["official_po_or_ml_fpp_below_0p01"].sum()),
                    "official_hanabi_figure_overlap": int(selected["official_hanabi_figure_overlap"].sum()),
                    "official_any_pair_resolved_hanabi_overlap": int(
                        selected["official_any_pair_resolved_hanabi_overlap"].sum()
                    ),
                }
            )
    budget = pd.DataFrame(budget_rows)
    write_csv(root / "results/pe_official_budget_summary.csv", budget)
    make_figures(root, full_enriched, strict_enriched)
    score_contract = json.loads((root / "contracts/SCORE_COMPLETE.json").read_text())
    top_columns = [
        "consensus_rank",
        "pair_key",
        "final_score_mean",
        "waveform_contribution_mean",
        "time_contribution_mean",
        "sky_contribution_mean",
        "waveform_available",
        "pe_mc_bhattacharyya_coefficient",
        "pe_mc_standardized_distance",
        "pe_dmax_intrinsic",
        "official_po_fpp",
        "official_ml_fpp",
        "official_hanabi_figure_overlap",
        "official_o3a_hanabi_table_overlap",
        "official_o3a_hanabi_log10_b_lu",
        "official_any_pair_resolved_hanabi_overlap",
    ]
    write_json(
        root / "results/MAIN_O3OFFICIAL_CFIX_V1_SUMMARY.json",
        {
            "experiment_code": EXPERIMENT_CODE,
            "status": FINAL_STATUS,
            "scope": score_contract,
            "pe_official_budget_summary": budget.to_dict(orient="records"),
            "full70_top10": full_enriched[top_columns].head(10).to_dict(orient="records"),
            "strict_h1l1_top10": strict_enriched[top_columns].head(10).to_dict(orient="records"),
            "real_catalog_has_known_lensing_labels": False,
            "real_catalog_recall_reported": False,
        },
    )
    top_display_columns = [
        "consensus_rank",
        "pair_key",
        "final_score_mean",
        "waveform_available",
        "pe_mc_bhattacharyya_coefficient",
        "pe_mc_standardized_distance",
        "pe_dmax_intrinsic",
        "official_po_fpp",
        "official_ml_fpp",
        "official_hanabi_figure_overlap",
        "official_o3a_hanabi_table_overlap",
        "official_o3a_hanabi_log10_b_lu",
    ]
    full_top10_text = full_enriched[top_display_columns].head(10).to_string(index=False)
    strict_top10_text = strict_enriched[top_display_columns].head(10).to_string(index=False)
    budget_text = budget.to_string(index=False)
    report = f"""# {EXPERIMENT_CODE} 完整报告

## 结果身份

本实验代号为 **{EXPERIMENT_CODE}**。父版本为 `BAYESTAR-CFIX-63-v1`，历史严格结果记为
`HIST-CFIX-53-v1`。本目录是独立输出，不覆盖 v9.3、v9.4、C-fixed 父结果或论文。

## 冻结方法

- 真实天空图从公开 PE HDF5 读取，严格解析 `[b'True']`，显式执行 NESTED 到 RING；
- 共同分辨率为 Nside=512；天空分数仍为 `log[Npix sum(P_i P_j)]`；
- 注入性能沿用已冻结的事件级 BAYESTAR locked test；
- waveform encoder、4096 点峰值 2 s 输入、waveform calibration 和一维 time LR 均未修改；
- C-fixed 三个 seed 权重分别为 0.5/0.25/0.5、1/0.25/0.5、1/0.25/0.5；
- 真实 PE、官方 FPP、GOLUM/Hanabi 阶段均只在排名冻结后联表，不参与调权。

## 样本范围

- LVK full-O3 PO/ML：70 个事件、2,415 个无序 pair；不增加 SNR 门槛；
- 名义公开 detector network 同时含 H1+L1：{score_contract['nominal_h1l1_events']} 个事件；
- 通过冻结 C-fixed 波形预处理的实际 strict H1+L1：{score_contract['actual_strict_h1l1_events']} 个事件、{score_contract['actual_strict_h1l1_pairs']} 个 pair；
- 全 70 事件表中，缺 waveform 的 pair 使用精确的中性贡献 0，未把缺失通道零填充送入 encoder。

“64”是按公开 detector network 得到的名义数量，不是预先保证的预处理通过数。严格结果以逐事件 finite-window、256 s off-source PSD 和相同预处理审计为准。

## 注入检索基线

本轮没有改变注入数据或任何检索通道，因此 C-fixed locked-test 指标按哈希复用：O3 waveform-only R@10 约 0.555，C-fixed 三通道 R@10 约 0.865；O4a 对应约 0.474 和 0.777。它们是冻结控制，不是对真实 70 事件的“召回率”。真实目录没有已知透镜标签，只能报告排名、PE 和官方 follow-up 对照。

## 官方阶段解释

`official_po_fpp` 与 `official_ml_fpp` 来自 LVK 2,415-pair 机器可读表。论文报告 75 对进入 GOLUM，但公开机器表未提供完整的逐 pair GOLUM 成员字段，因此本交付不会伪造该标签。完整 O3 论文的公开 Bayes-factor 图可逐对读取 14 个标签，并报告它们均有 B_L/U<1。本交付还单独联入先行 O3a 官方表中、同时属于当前 70 事件范围的 14 个逐 pair Hanabi 数值；两种 provenance 不混合成一个新的“官方候选集”。

## 冻结后的真实排名与审计

全 70 事件 Top-10：

```text
{full_top10_text}
```

严格 H1+L1 子集 Top-10：

```text
{strict_top10_text}
```

Top-10/20/50 PE 与官方阶段预算汇总：

```text
{budget_text}
```

## 结果边界

这些结果是 lens-candidate refinement / shortlist，不是 lensing detection。PE/FPP/Hanabi 的吻合程度是冻结后的外部审计，不允许反向优化三通道权重。

最终状态：`{FINAL_STATUS}`
"""
    (root / "reports/MAIN_O3OFFICIAL_CFIX_V1_REPORT_CN.md").write_text(report, encoding="utf-8")
    write_json(
        root / "contracts/AUDIT_COMPLETE.json",
        {
            "passed": True,
            "timestamp_utc": utc_stamp(),
            "pe_used_for_ranking": False,
            "official_followup_used_for_ranking": False,
            "full70_top50_rows": len(full_enriched),
            "strict_top50_rows": len(strict_enriched),
        },
    )


def finalize(root: Path) -> Path:
    if not (root / "contracts/AUDIT_COMPLETE.json").is_file():
        raise RuntimeError("audit must complete first")
    historical_after = baseline_hashes()
    historical_before = json.loads((root / "contracts/HISTORICAL_HASHES_BEFORE.json").read_text())
    unchanged = historical_after == historical_before
    write_json(
        root / "contracts/HISTORICAL_IMMUTABILITY_AFTER.json",
        {"passed": unchanged, "before": historical_before, "after": historical_after},
    )
    if not unchanged:
        raise RuntimeError("Historical baseline hashes changed")
    source = Path(__file__).resolve()
    shutil.copy2(source, root / "scripts" / source.name)
    dependency_paths = [
        PROJECT / "scripts/experiments/gwtc_c_scheme_ordering_confirmation_20260831.py",
        PROJECT / "scripts/experiments/sky_morphology_one_vs_two_stage_exploratory.py",
        PROJECT / "scripts/experiments/run_gwtc_sky_ordering_corrected_v94.py",
        PROJECT / "scripts/experiments/run_gwtc_sky_resolution_v93.py",
        PROJECT / "scripts/experiments/121_gwtc_sky_v81_pipeline.py",
        PROJECT / "scripts/experiments/20_real_noise_injection_v3_physical.py",
        PROJECT / "scripts/experiments/26_spectrogram_encoder_pilot.py",
        PROJECT / "scripts/real_search/unified_v7_common.py",
        PROJECT / "scripts/real_search/physical_common.py",
        PROJECT / "scripts/real_search/37_unified_intrinsic_multitask_pilot.py",
        PROJECT / "scripts/real_search/common.py",
    ]
    dependency_rows: list[dict[str, Any]] = []
    dependency_root = root / "scripts/dependencies"
    dependency_root.mkdir(parents=True, exist_ok=True)
    for dependency in dependency_paths:
        if not dependency.is_file():
            raise FileNotFoundError(f"Frozen code dependency missing: {dependency}")
        copied = dependency_root / dependency.name
        shutil.copy2(dependency, copied)
        dependency_rows.append(
            {
                "project_path": str(dependency.relative_to(PROJECT)),
                "packaged_copy": str(copied.relative_to(root)),
                "sha256": sha256_file(dependency),
            }
        )
    write_csv(root / "manifests/CODE_DEPENDENCY_SHA256.csv", pd.DataFrame(dependency_rows))
    reproduce = root / "scripts/reproduce.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/../../..\"\n"
        "OUT=${1:-results/main_o3official_cfixed_v1_reproduction_$(date -u +%Y%m%dT%H%M%SZ)}\n"
        "python scripts/experiments/main_o3official_cfixed_v1.py --root \"$OUT\" --phase all --download-workers 8\n",
        encoding="ascii",
    )
    reproduce.chmod(0o755)
    (root / "README_CN.md").write_text(
        f"# {EXPERIMENT_CODE}\n\n"
        "这是独立的 full-O3 70-event 与严格 H1+L1 子集 C-fixed 重排。"
        "原始 strain 和 PE HDF5 位于 cache，仅在输入 SHA-256 manifest 中登记，不进入交付包。\n\n"
        f"最终状态：`{FINAL_STATUS}`\n",
        encoding="utf-8",
    )
    version_registry = pd.DataFrame(
        [
            {
                "version_code": "SKY-V9.3-HIST",
                "scope": "historical sky-resolution deployment",
                "events": np.nan,
                "pairs": np.nan,
                "path": "results/gwtc_sky_resolution_v93_20260730",
                "status": "read-only historical result",
            },
            {
                "version_code": "SKY-V9.4-ORDERFIX",
                "scope": "historical NESTED/RING ordering correction",
                "events": np.nan,
                "pairs": np.nan,
                "path": "results/gwtc_sky_ordering_corrected_v94_20260830",
                "status": "read-only historical result",
            },
            {
                "version_code": "HIST-CFIX-53-v1",
                "scope": "historical project strict H1+L1 subset",
                "events": 53,
                "pairs": 1378,
                "path": str(BASELINE_ROOT),
                "status": "read-only historical parent",
            },
            {
                "version_code": "BAYESTAR-CFIX-63-v1",
                "scope": "historical project 63-event full scope plus strict53",
                "events": 63,
                "pairs": 1953,
                "path": str(BASELINE_ROOT),
                "status": "read-only frozen parent",
            },
            {
                "version_code": EXPERIMENT_CODE,
                "scope": "official full-O3 70 events plus audited strict H1+L1 subset",
                "events": 70,
                "pairs": 2415,
                "path": str(root),
                "status": FINAL_STATUS,
            },
        ]
    )
    write_csv(root / "contracts/VERSION_REGISTRY.csv", version_registry)
    write_json(
        root / "contracts/FINAL_STATUS.json",
        {
            "experiment_code": EXPERIMENT_CODE,
            "status": FINAL_STATUS,
            "historical_outputs_unchanged": True,
            "paper_modified": False,
            "timestamp_utc": utc_stamp(),
        },
    )

    included: list[Path] = []
    for directory in ("contracts", "data", "results", "figures", "reports", "scripts", "manifests", "logs"):
        included.extend(path for path in (root / directory).rglob("*") if path.is_file())
    included.append(root / "README_CN.md")
    included = list(dict.fromkeys(included))
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(included)
    ]
    write_csv(root / "manifests/OUTPUT_SHA256_MANIFEST.csv", pd.DataFrame(manifest_rows))
    included.append(root / "manifests/OUTPUT_SHA256_MANIFEST.csv")
    package = PROJECT / "packages/main_o3official_cfixed_v1_20260904_deliverables.tar.gz"
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite {package}")
    with tarfile.open(package, "w:gz") as archive:
        for path in sorted(included):
            archive.add(path, arcname=f"{root.name}/{path.relative_to(root)}", recursive=False)
    package_sha = sha256_file(package)
    sha_path = package.with_suffix(package.suffix + ".sha256")
    sha_path.write_text(f"{package_sha}  {package.name}\n", encoding="ascii")
    write_json(
        root / "contracts/PACKAGE_COMPLETE.json",
        {
            "package": str(package),
            "sha256": package_sha,
            "bytes": package.stat().st_size,
            "raw_strain_included": False,
            "raw_pe_hdf5_included": False,
        },
    )
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "score", "audit", "finalize", "all"), default="all")
    parser.add_argument("--download-workers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    ensure_dirs(root)
    phases = ("prepare", "score", "audit", "finalize") if args.phase == "all" else (args.phase,)
    for phase in phases:
        print(f"[{EXPERIMENT_CODE}] phase={phase} start", flush=True)
        if phase == "prepare":
            prepare(root, args.download_workers)
        elif phase == "score":
            score(root)
        elif phase == "audit":
            audit(root)
        elif phase == "finalize":
            package = finalize(root)
            print(f"[package] {package}", flush=True)
        print(f"[{EXPERIMENT_CODE}] phase={phase} complete", flush=True)


if __name__ == "__main__":
    main()
