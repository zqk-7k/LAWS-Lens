#!/usr/bin/env python3
"""Independent C-scheme confirmation after the GWTC sky-ordering audit.

This pipeline freezes the v9.3 waveform and one-dimensional time channels,
reconstructs public PE maps with strict NESTED-to-RING handling, builds a
map-matched injection sky background with source/noise/template-disjoint
validation and test subsets, and compares:

* C-fixed: corrected Nside=512 sky maps with frozen per-seed v9.3 weights;
* C-retuned: the same maps with weights selected on validation only.

The held-out test pair scores are not read until selected_config.json has
been written and hashed.  Real-catalog PE and historical-candidate metadata
are attached only after the test evaluation.  No historical output is
modified by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import sys
import tarfile
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEPLOYMENTS = ("gwtc3", "gwtc4")
MODEL_SEEDS = (202607241, 202607242, 202607243)
DESIGN_SEEDS = {
    202607241: 202608311,
    202607242: 202608312,
    202607243: 202608313,
}
ANALYSIS_NSIDE = 512
COARSE_NSIDE = 256
REFERENCE_NSIDE = 1024
ROTATION_WORKERS = 6
BOOTSTRAP_DRAWS = 10_000
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
VERSION = "gwtc_c_scheme_ordering_confirmation_20260831"
RAW_BF_FLOOR = 1e-30

# These parent-noise-bank partitions were selected before opening pair scores,
# using only compact injection provenance and retained-family balance.  The
# complement is the validation bank set.  A parent bank can never occur in
# both validation and test for the same design/model seed.
TEST_NOISE_BANKS: dict[int, tuple[int, ...]] = {
    202607241: (1, 2, 5, 7, 9, 10, 11, 13, 18, 22, 23, 24, 30, 31, 32, 33,
                34, 36, 37, 40, 41, 42, 44, 45, 48, 49, 54, 56, 58, 59, 60, 61),
    202607242: (1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 16, 19, 21, 23, 28, 29,
                33, 35, 37, 38, 42, 45, 46, 47, 48, 49, 50, 55, 60, 61, 62, 63),
    202607243: (0, 1, 5, 9, 12, 14, 15, 17, 19, 22, 25, 26, 27, 29, 31, 32,
                34, 35, 36, 37, 48, 49, 50, 51, 53, 54, 55, 56, 58, 60, 61, 63),
}

FROZEN_V93_WEIGHTS: dict[str, dict[int, dict[str, float]]] = {
    "gwtc3": {
        202607241: {"waveform": 0.50, "time": 0.25, "sky": 0.50},
        202607242: {"waveform": 1.00, "time": 0.25, "sky": 0.50},
        202607243: {"waveform": 1.00, "time": 0.25, "sky": 0.50},
    },
    "gwtc4": {
        202607241: {"waveform": 1.00, "time": 0.25, "sky": 0.50},
        202607242: {"waveform": 0.50, "time": 0.25, "sky": 0.50},
        202607243: {"waveform": 1.00, "time": 0.25, "sky": 0.25},
    },
}

NOISE_NONINFERIORITY = {"r_at_10": 0.02, "average_precision": 0.005}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MORPH_SCRIPT = PROJECT / "scripts/experiments/sky_morphology_one_vs_two_stage_exploratory.py"
V94_SCRIPT = PROJECT / "scripts/experiments/run_gwtc_sky_ordering_corrected_v94.py"
if not MORPH_SCRIPT.is_file() or not V94_SCRIPT.is_file():
    raise FileNotFoundError("Required v9.4/morphology helper scripts are missing")
morph = load_module(MORPH_SCRIPT, "morph_helpers_for_c_confirmation")
v94 = load_module(V94_SCRIPT, "v94_helpers_for_c_confirmation")
v7 = v94.v93.v7


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
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


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def canonical_pair(a: object, b: object) -> str:
    x, y = str(a), str(b)
    return f"{x}--{y}" if x <= y else f"{y}--{x}"


def source_run(deployment: str) -> Path:
    return morph.source_run(deployment)


def input_seed_dir(deployment: str, seed: int) -> Path:
    return (
        PROJECT
        / "results/real_noise_injection_v7_peak2s_formal_20260722"
        / deployment
        / f"seed_{seed}"
    )


def corrected_seed_dir(deployment: str, seed: int) -> Path:
    return (
        PROJECT
        / "results/gwtc_sky_ordering_corrected_v94_20260830"
        / deployment
        / f"seed_{seed}"
    )


def parse_boolish(raw: Any) -> bool:
    """Strict scalar bool decoder, including HDF5 arrays like [b'True']."""
    values = np.asarray(raw)
    if values.size != 1:
        raise ValueError(f"Expected one boolean value, got shape={values.shape}")
    value = values.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("ascii", errors="strict")
    else:
        text = str(value)
    normalized = text.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unrecognized boolean metadata: {raw!r}")


def corrected_read_probability_map(
    row: pd.Series,
    target_nside: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a public PE HDF5 map and always return normalized RING mass."""
    path = morph.resolve_pe_path(row["sky_map_path"], str(row.get("deployment", "gwtc3")))
    group = v94.choose_h5_skymap_group(path)
    if group is None:
        raise FileNotFoundError(f"No skymap group in {path}")
    expected_group = str(
        row.get("sky_map_internal_group", row.get("sky_map_group", ""))
    ).strip()
    if expected_group and expected_group.lower() != "nan" and expected_group != group:
        raise ValueError(f"PE group mismatch for {row.get('event_name')}: {expected_group} != {group}")
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle[f"{group}/skymap/data"][:], dtype=np.float64).reshape(-1)
        nest_key = f"{group}/skymap/meta_data/nest"
        if nest_key not in handle:
            raise KeyError(nest_key)
        raw_nest = handle[nest_key][()]
        nested = parse_boolish(raw_nest)
    probability = v94.sanitize_probability_map(data)
    source_nside = int(hp.npix2nside(len(probability)))
    if nested:
        probability = hp.reorder(probability, n2r=True)
    probability = v94.resize_probability_map(probability, source_nside, int(target_nside))
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.clip(probability, 0.0, None)
    probability /= probability.sum(dtype=np.float64)
    return probability.astype(np.float32), {
        "source_path": str(path),
        "source_group": group,
        "source_nside": source_nside,
        "source_ordering": "NESTED" if nested else "RING",
        "output_ordering": "RING",
        "target_nside": int(target_nside),
        "coordinate_frame": "celestial_equatorial_ra_dec",
        "ordering_conversion_applied": bool(nested),
        "ordering_metadata_raw": repr(raw_nest),
    }


def run_ordering_unit_tests(root: Path) -> dict[str, Any]:
    assertions = {
        "array_true": parse_boolish(np.asarray([b"True"])),
        "array_false": not parse_boolish(np.asarray([b"False"])),
        "scalar_true": parse_boolish(True),
        "scalar_zero": not parse_boolish(0),
    }
    nside = 8
    rng = np.random.default_rng(20260831)
    ring = rng.random(hp.nside2npix(nside))
    ring /= ring.sum()
    nested = hp.reorder(ring, r2n=True)
    with tempfile.TemporaryDirectory(dir=root / "logs") as tmp:
        path = Path(tmp) / "nested_test.h5"
        group = "C01:IMRPhenomXPHM"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(f"{group}/skymap/data", data=nested)
            handle.create_dataset(f"{group}/skymap/meta_data/nest", data=np.asarray([b"True"]))
        row = pd.Series(
            {
                "deployment": "gwtc3",
                "event_name": "unit-test",
                "sky_map_path": str(path),
                "sky_map_internal_group": group,
            }
        )
        recovered, metadata = corrected_read_probability_map(row, nside)
    max_error = float(np.max(np.abs(recovered.astype(np.float64) - ring)))
    passed = bool(all(assertions.values()) and max_error < 1e-7 and metadata["output_ordering"] == "RING")
    result = {
        "passed": passed,
        "assertions": assertions,
        "nested_to_ring_max_abs_error": max_error,
        "tolerance": 1e-7,
        "metadata": metadata,
    }
    write_json(root / "contracts/ordering_unit_test.json", result)
    if not passed:
        raise RuntimeError(f"Ordering unit tests failed: {result}")
    return result


def canonical_network(value: object) -> str:
    text = str(value).upper().replace(" ", "")
    detectors = [name for name in ("H1", "L1", "V1", "K1") if name in text]
    return "+".join(detectors) if detectors else "UNKNOWN"


def normalize_run(value: object, deployment: str) -> str:
    text = str(value)
    if deployment == "gwtc4" or text.startswith("O4a"):
        return "O4a"
    return text


def map_descriptor(probability: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(probability, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    values /= values.sum()
    nside = int(hp.npix2nside(len(values)))
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    n50 = int(np.searchsorted(cumulative, 0.50, side="left")) + 1
    n90 = int(np.searchsorted(cumulative, 0.90, side="left")) + 1
    entropy = float(-np.sum(values * np.log(np.maximum(values, 1e-300))))
    return {
        "area50_deg2": float(n50 * pixel_area),
        "area90_deg2": float(n90 * pixel_area),
        "entropy_nats": entropy,
        "kl_from_isotropic_nats": float(math.log(len(values)) - entropy),
        "hpd90_component_count": int(morph.component_count_hpd90(values, nside)),
    }


def primary_manifest(deployment: str) -> pd.DataFrame:
    frame = morph.primary_manifest(pd.read_csv(source_run(deployment) / "data/event_manifest.csv"))
    frame = frame.copy()
    frame["deployment"] = deployment
    frame["run_normalized"] = [
        normalize_run(value, deployment) for value in frame["run"]
    ]
    frame["network_normalized"] = frame["detectors_available"].map(canonical_network)
    return frame


def template_metadata(deployment: str, root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest = primary_manifest(deployment)
    for index, event in manifest.iterrows():
        probability, metadata = corrected_read_probability_map(event, target_nside=64)
        descriptor = map_descriptor(probability)
        rows.append(
            {
                "template_index": int(index),
                "deployment": deployment,
                "event_name": str(event["event_name"]),
                "run": str(event["run_normalized"]),
                "network": str(event["network_normalized"]),
                "network_snr": float(event.get("network_snr", np.nan)),
                **descriptor,
                **metadata,
            }
        )
    frame = pd.DataFrame(rows)
    write_csv(root / f"contracts/{deployment}_public_pe_template_metadata.csv", frame)
    return frame


def template_roles(metadata: pd.DataFrame, design_seed: int) -> pd.DataFrame:
    """Create disjoint reference/validation/test event-template pools."""
    output: list[pd.DataFrame] = []
    for (run, network), part in metadata.groupby(["run", "network"], sort=True):
        ordered = part.copy()
        ordered["split_hash"] = ordered["event_name"].map(
            lambda name: stable_seed("template-role", design_seed, run, network, name)
        )
        ordered = ordered.sort_values(["split_hash", "event_name"], kind="stable").reset_index(drop=True)
        n = len(ordered)
        if n == 1:
            roles = ["reference"]
        elif n == 2:
            roles = ["validation", "test"]
        else:
            # Twenty percent of each observing-run pool is used only to draw
            # target A90/morphology descriptors.  The remaining event maps
            # are split as evenly as possible between validation and test.
            n_reference = max(1, int(round(0.20 * n)))
            roles = ["reference"] * n_reference
            roles.extend(
                "validation" if k % 2 == 0 else "test"
                for k in range(n - n_reference)
            )
        ordered["template_role"] = roles
        output.append(ordered)
    result = pd.concat(output, ignore_index=True)
    overlap = set(result.loc[result.template_role == "validation", "event_name"]) & set(
        result.loc[result.template_role == "test", "event_name"]
    )
    if overlap:
        raise RuntimeError(f"Template split overlap: {sorted(overlap)}")
    return result


def system_id_frame(events: pd.DataFrame) -> pd.Series:
    return events["family"].astype(str) + ":" + events["source_index"].astype(str)


def attach_noise_provenance(deployment: str, seed: int, events: pd.DataFrame) -> pd.DataFrame:
    metadata_path = input_seed_dir(deployment, seed) / "data/real_noise_injections/compact_injection_metadata.parquet"
    metadata = pd.read_parquet(metadata_path)
    lookup: dict[tuple[str, int], pd.Series] = {}
    for _, row in metadata.iterrows():
        lookup[(str(row["family"]), int(row["sample_index"]))] = row
    out = events.copy()
    banks: list[int] = []
    offsets: list[int] = []
    for event in out.itertuples(index=False):
        row = lookup[(str(event.family), int(event.source_index))]
        image = 2 if str(event.tag).upper() in {"L2", "I2", "IMAGE2"} else 1
        bank = row[f"image{image}_noise_bank_index"]
        offset = row[f"image{image}_noise_offset_samples"]
        if not np.isfinite(bank) or not np.isfinite(offset):
            raise RuntimeError(f"Missing noise provenance for {deployment}/{seed}/{event}")
        banks.append(int(bank))
        offsets.append(int(offset))
    out["system_id"] = system_id_frame(out)
    out["parent_noise_bank"] = banks
    out["noise_offset_samples"] = offsets
    return out


def retained_event_plan(
    deployment: str,
    seed: int,
    split: str,
) -> pd.DataFrame:
    short = "val" if split == "validation" else "test"
    path = input_seed_dir(deployment, seed) / "results" / f"mixed_{short}_synthetic_events_v7.parquet"
    events = attach_noise_provenance(deployment, seed, pd.read_parquet(path))
    test_banks = set(TEST_NOISE_BANKS[seed])
    allowed = set(range(64)) - test_banks if split == "validation" else test_banks
    keep_systems = []
    for system_id, part in events.groupby("system_id", sort=False):
        if set(part["parent_noise_bank"].astype(int)).issubset(allowed):
            keep_systems.append(str(system_id))
    retained = events.loc[events["system_id"].isin(keep_systems)].copy()
    retained = retained.sort_values("idx", kind="stable").reset_index(drop=True)
    retained["old_idx"] = retained["idx"].astype(int)
    retained["idx"] = np.arange(len(retained), dtype=np.int32)
    retained["split"] = split
    retained["design_seed"] = DESIGN_SEEDS[seed]
    return retained


def subset_pair_table(
    deployment: str,
    seed: int,
    split: str,
    retained: pd.DataFrame,
    allow_test_scores: bool,
) -> pd.DataFrame:
    if split == "test" and not allow_test_scores:
        raise RuntimeError("Held-out test pair scores cannot be opened before configuration freeze")
    filename = "fusion_validation_pairs_v7.parquet" if split == "validation" else "fusion_heldout_test_pairs_v7.parquet"
    source = pd.read_parquet(input_seed_dir(deployment, seed) / "results" / filename)
    mapping = dict(zip(retained["old_idx"].astype(int), retained["idx"].astype(int)))
    mask = source["idx_i"].isin(mapping) & source["idx_j"].isin(mapping)
    frame = source.loc[mask].copy()
    frame["old_idx_i"] = frame["idx_i"].astype(int)
    frame["old_idx_j"] = frame["idx_j"].astype(int)
    frame["idx_i"] = frame["old_idx_i"].map(mapping).astype(np.int32)
    frame["idx_j"] = frame["old_idx_j"].map(mapping).astype(np.int32)
    frame["event_count"] = int(len(retained))
    expected = len(retained) * (len(retained) - 1) // 2
    if len(frame) != expected:
        raise RuntimeError(f"Incomplete subset pair table: {len(frame)} != {expected}")
    return frame.reset_index(drop=True)


def assignment_cost(
    event: pd.Series,
    candidate: pd.Series,
    reference: pd.Series,
    deployment: str,
) -> tuple[float, dict[str, Any]]:
    target_run = morph.infer_run(float(event["gps_obs"]), deployment)
    target_network = "H1+L1"
    run_match = str(candidate["run"]) == target_run
    network_match = str(candidate["network"]) == target_network
    snr_distance = abs(math.log(max(float(candidate["network_snr"]), 1e-6) / max(float(event["snr"]), 1e-6)))
    area_distance = abs(math.log(max(float(candidate["area90_deg2"]), 1e-6) / max(float(reference["area90_deg2"]), 1e-6)))
    kl_scale = max(float(reference["kl_from_isotropic_nats"]), 0.25)
    kl_distance = abs(float(candidate["kl_from_isotropic_nats"]) - float(reference["kl_from_isotropic_nats"])) / kl_scale
    component_distance = abs(int(candidate["hpd90_component_count"]) - int(reference["hpd90_component_count"]))
    cost = (
        25.0 * (not run_match)
        + 8.0 * (not network_match)
        + 2.0 * snr_distance
        + 1.0 * area_distance
        + 0.5 * kl_distance
        + 0.25 * min(component_distance, 8)
    )
    return cost, {
        "target_run": target_run,
        "target_network": target_network,
        "run_matched": bool(run_match),
        "network_matched": bool(network_match),
        "snr_log_distance": float(snr_distance),
        "area90_log_distance": float(area_distance),
        "kl_scaled_distance": float(kl_distance),
        "component_count_distance": int(component_distance),
    }


def build_template_assignment(
    deployment: str,
    seed: int,
    split: str,
    events: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    actual_pool = roles.loc[roles["template_role"] == split].copy()
    reference_pool = roles.loc[roles["template_role"] == "reference"].copy()
    if actual_pool.empty or reference_pool.empty:
        raise RuntimeError(f"Empty template pool for {deployment}/{seed}/{split}")
    rows: list[dict[str, Any]] = []
    use_count: dict[int, int] = {}
    used_in_system: dict[str, set[int]] = {}
    ordered_events = events.sort_values(["family", "source_index", "tag", "idx"], kind="stable")
    for _, event in ordered_events.iterrows():
        target_run = morph.infer_run(float(event["gps_obs"]), deployment)
        ref_cost = (
            25.0 * (reference_pool["run"].astype(str) != target_run).astype(float)
            + 8.0 * (reference_pool["network"].astype(str) != "H1+L1").astype(float)
            + 2.0 * np.abs(
                np.log(np.maximum(reference_pool["network_snr"].to_numpy(float), 1e-6) / max(float(event["snr"]), 1e-6))
            )
        )
        ref_jitter = reference_pool["event_name"].map(
            lambda name: stable_seed("reference", DESIGN_SEEDS[seed], split, int(event["old_idx"]), name) / 2**32
        ).to_numpy(float) * 1e-8
        reference = reference_pool.iloc[int(np.argmin(ref_cost.to_numpy(float) + ref_jitter))]
        system = str(event["system_id"])
        used = used_in_system.setdefault(system, set())
        candidates: list[tuple[float, int, dict[str, Any], int]] = []
        for candidate_index, candidate in actual_pool.iterrows():
            template_index = int(candidate["template_index"])
            if template_index in used:
                continue
            cost, diagnostics = assignment_cost(event, candidate, reference, deployment)
            cost += stable_seed("candidate", DESIGN_SEEDS[seed], split, int(event["old_idx"]), candidate["event_name"]) / 2**32 * 1e-8
            candidates.append(
                (
                    float(cost),
                    int(candidate_index),
                    diagnostics,
                    use_count.get(template_index, 0),
                )
            )
        if not candidates:
            raise RuntimeError(f"No distinct template for system {system}")
        # Preserve run/network matching whenever such templates exist.  Within
        # that scientifically matched pool, exhaust the least-used templates
        # before reuse, then minimize the SNR/A90/morphology cost.  This avoids
        # an artificial few-template sky background.
        exact = [
            item
            for item in candidates
            if item[2]["run_matched"] and item[2]["network_matched"]
        ]
        if exact:
            candidates = exact
        else:
            same_run = [item for item in candidates if item[2]["run_matched"]]
            if same_run:
                candidates = same_run
        cost, candidate_index, diagnostics, _ = min(
            candidates, key=lambda item: (item[3], item[0], item[1])
        )
        candidate = actual_pool.loc[candidate_index]
        template_index = int(candidate["template_index"])
        used.add(template_index)
        use_count[template_index] = use_count.get(template_index, 0) + 1
        ood = bool(
            not diagnostics["run_matched"]
            or not diagnostics["network_matched"]
            or diagnostics["snr_log_distance"] > math.log(2.5)
            or diagnostics["area90_log_distance"] > math.log(10.0)
        )
        rows.append(
            {
                "idx": int(event["idx"]),
                "old_idx": int(event["old_idx"]),
                "system_id": system,
                "family": str(event["family"]),
                "tag": str(event["tag"]),
                "source_index": int(event["source_index"]),
                "snr": float(event["snr"]),
                "ra_true": float(event["ra_true"]),
                "dec_true": float(event["dec_true"]),
                "template_index": template_index,
                "template_event": str(candidate["event_name"]),
                "template_role": split,
                "template_run": str(candidate["run"]),
                "template_network": str(candidate["network"]),
                "template_network_snr": float(candidate["network_snr"]),
                "template_area90_deg2": float(candidate["area90_deg2"]),
                "template_kl": float(candidate["kl_from_isotropic_nats"]),
                "template_components": int(candidate["hpd90_component_count"]),
                "reference_template_event": str(reference["event_name"]),
                "reference_area90_deg2": float(reference["area90_deg2"]),
                "reference_kl": float(reference["kl_from_isotropic_nats"]),
                "reference_components": int(reference["hpd90_component_count"]),
                "assignment_cost": float(cost),
                "template_ood": ood,
                **diagnostics,
            }
        )
    result = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    for _, part in result.groupby("system_id"):
        if len(part) > 1 and part["template_event"].nunique() != len(part):
            raise RuntimeError("Two lensed images share an actual PE template")
    return result


def target_vector(ra: float, dec: float) -> np.ndarray:
    return np.asarray(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)],
        dtype=np.float64,
    )


def frozen_rotation_parameters(
    assignment: pd.DataFrame,
    template_maps: dict[int, np.ndarray],
    deployment: str,
    seed: int,
    split: str,
) -> pd.DataFrame:
    out = assignment.copy()
    source_vectors = []
    roll_angles = []
    anchors = []
    for row in out.itertuples(index=False):
        probability = np.asarray(template_maps[int(row.template_index)], dtype=np.float64)
        probability /= probability.sum()
        rng = np.random.default_rng(
            stable_seed("rotation", deployment, seed, DESIGN_SEEDS[seed], split, int(row.old_idx))
        )
        anchor = int(rng.choice(len(probability), p=probability))
        source = np.asarray(hp.pix2vec(ANALYSIS_NSIDE, anchor, nest=False), dtype=np.float64)
        source_vectors.append(source)
        roll_angles.append(float(rng.uniform(0.0, 2.0 * math.pi)))
        anchors.append(anchor)
    vectors = np.stack(source_vectors)
    out["anchor_pixel_nside512"] = anchors
    out["source_vector_x"] = vectors[:, 0]
    out["source_vector_y"] = vectors[:, 1]
    out["source_vector_z"] = vectors[:, 2]
    out["roll_angle_rad"] = roll_angles
    return out


def rotate_with_frozen_geometry(
    probability: np.ndarray,
    source_vector: np.ndarray,
    roll_angle: float,
    ra_true: float,
    dec_true: float,
    output_vectors: np.ndarray,
) -> np.ndarray:
    source_map = np.asarray(probability, dtype=np.float64)
    source_map = np.clip(source_map, 0.0, None)
    source_map /= source_map.sum(dtype=np.float64)
    target = target_vector(ra_true, dec_true)
    rotation = morph.axis_angle_matrix(target, roll_angle) @ morph.rotation_between(source_vector, target)
    input_vectors = rotation.T @ output_vectors
    theta = np.arccos(np.clip(input_vectors[2], -1.0, 1.0))
    phi = np.mod(np.arctan2(input_vectors[1], input_vectors[0]), 2.0 * math.pi)
    rotated = hp.get_interp_val(source_map, theta, phi, nest=False)
    rotated = np.clip(rotated, 0.0, None)
    rotated /= rotated.sum(dtype=np.float64)
    return rotated.astype(np.float32)


def load_template_maps(
    deployment: str,
    metadata: pd.DataFrame,
    indices: Iterable[int],
    nside: int,
) -> dict[int, np.ndarray]:
    manifest = primary_manifest(deployment)
    result: dict[int, np.ndarray] = {}
    for index in sorted(set(map(int, indices))):
        probability, _ = corrected_read_probability_map(manifest.iloc[index], target_nside=nside)
        result[index] = probability
    return result


def generate_map_bank(
    deployment: str,
    seed: int,
    split: str,
    assignment: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    template_maps = load_template_maps(
        deployment, metadata, assignment["template_index"].astype(int), ANALYSIS_NSIDE
    )
    if "source_vector_x" not in assignment:
        assignment = frozen_rotation_parameters(assignment, template_maps, deployment, seed, split)
    npix = hp.nside2npix(ANALYSIS_NSIDE)
    output_vectors = np.asarray(hp.pix2vec(ANALYSIS_NSIDE, np.arange(npix), nest=False), dtype=np.float64)
    maps = np.empty((len(assignment), npix), dtype=np.float32)
    started = time.perf_counter()

    def task(row: Any) -> tuple[int, np.ndarray]:
        source = np.asarray([row.source_vector_x, row.source_vector_y, row.source_vector_z], dtype=np.float64)
        probability = rotate_with_frozen_geometry(
            template_maps[int(row.template_index)],
            source,
            float(row.roll_angle_rad),
            float(row.ra_true),
            float(row.dec_true),
            output_vectors,
        )
        return int(row.idx), probability

    with ThreadPoolExecutor(max_workers=ROTATION_WORKERS) as executor:
        futures = [executor.submit(task, row) for row in assignment.itertuples(index=False)]
        for future in as_completed(futures):
            index, probability = future.result()
            maps[index] = probability
    elapsed = time.perf_counter() - started
    del output_vectors, template_maps
    gc.collect()
    return maps, assignment, elapsed


def build_event_meta(
    features: Any,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    meta = features.event.copy()
    assignment_columns = assignment[
        [
            "idx",
            "snr",
            "template_network",
            "template_components",
            "template_ood",
            "template_event",
            "template_run",
            "run_matched",
            "network_matched",
        ]
    ].copy()
    assignment_columns["detector_count"] = assignment_columns["template_network"].map(
        lambda value: len(str(value).split("+")) if str(value) != "UNKNOWN" else 0
    )
    assignment_columns["hpd90_component_count"] = assignment_columns["template_components"].astype(int)
    return meta.merge(assignment_columns, on="idx", how="left", validate="one_to_one").sort_values("idx")


def attach_pair_ood(frame: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ood = assignment.sort_values("idx")["template_ood"].to_numpy(dtype=bool)
    ii = out["idx_i"].to_numpy(np.int32)
    jj = out["idx_j"].to_numpy(np.int32)
    out["sky_template_ood_i"] = ood[ii]
    out["sky_template_ood_j"] = ood[jj]
    out["sky_template_ood_pair"] = ood[ii] | ood[jj]
    return out


def normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError(weights)
    return {name: float(weights[name]) / total for name in ("waveform", "time", "sky")}


def score_vector(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        float(weights["waveform"]) * frame["waveform_score"].to_numpy(dtype=np.float64)
        + float(weights["time"]) * frame["time_score"].to_numpy(dtype=np.float64)
        + float(weights["sky"]) * frame["sky_raw_log_bf"].to_numpy(dtype=np.float64)
    )


def simplex_grid(step: float = 0.05) -> list[dict[str, float]]:
    denominator = int(round(1.0 / step))
    rows = []
    for waveform in range(denominator + 1):
        for time_weight in range(denominator - waveform + 1):
            sky = denominator - waveform - time_weight
            rows.append(
                {
                    "waveform": waveform / denominator,
                    "time": time_weight / denominator,
                    "sky": sky / denominator,
                }
            )
    return rows


def pair_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    result = dict(v7.pair_metrics(frame, scores))
    labels = frame["is_true_pair"].to_numpy(dtype=np.int8)
    result["roc_auc"] = float(roc_auc_score(labels, scores))
    order = np.argsort(-np.asarray(scores), kind="stable")
    for budget in (10, 20, 50, 100, 200):
        selected = labels[order[: min(budget, len(order))]]
        true_count = int(selected.sum())
        result[f"top_{budget}_true"] = true_count
        result[f"top_{budget}_false"] = int(len(selected) - true_count)
        result[f"top_{budget}_precision"] = float(true_count / max(len(selected), 1))
    return result


def full_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    return {**v7.retrieval_metrics(frame, scores), **pair_metrics(frame, scores)}


def select_retuned_weights(
    validation: pd.DataFrame,
    frozen: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    frozen_scores = score_vector(validation, frozen)
    baseline = full_metrics(validation, frozen_scores)
    baseline_normalized = normalized_weights(frozen)
    rows: list[dict[str, Any]] = []
    for weights in simplex_grid(0.05):
        scores = score_vector(validation, weights)
        metrics = full_metrics(validation, scores)
        distance = math.sqrt(
            sum((weights[name] - baseline_normalized[name]) ** 2 for name in weights)
        )
        feasible = bool(
            metrics["macro_r_at_10"] >= baseline["macro_r_at_10"] - NOISE_NONINFERIORITY["r_at_10"]
            and metrics["average_precision"] >= baseline["average_precision"] - NOISE_NONINFERIORITY["average_precision"]
            and metrics["false_at_recall_0p5"] <= baseline["false_at_recall_0p5"] * 1.05 + 1
            and metrics["false_at_recall_0p9"] <= baseline["false_at_recall_0p9"] * 1.05 + 1
        )
        rows.append(
            {
                **weights,
                **metrics,
                "distance_to_normalized_C_fixed": distance,
                "validation_guardrails_feasible": feasible,
            }
        )
    grid = pd.DataFrame(rows)
    candidates = grid.loc[grid["validation_guardrails_feasible"]].copy()
    fallback = False
    if candidates.empty:
        candidates = grid.copy()
        fallback = True
    candidates = candidates.sort_values(
        [
            "average_precision",
            "false_at_recall_0p9",
            "false_at_recall_0p5",
            "macro_r_at_10",
            "min_family_r_at_10",
            "macro_r_at_1",
            "distance_to_normalized_C_fixed",
            "waveform",
            "time",
            "sky",
        ],
        ascending=[False, True, True, False, False, False, True, False, False, False],
        kind="stable",
    )
    selected = candidates.iloc[0]
    weights = {name: float(selected[name]) for name in ("waveform", "time", "sky")}
    return {
        "weights": weights,
        "selection_metrics": {
            key: float(selected[key])
            for key in (
                "macro_r_at_1",
                "macro_r_at_10",
                "min_family_r_at_10",
                "average_precision",
                "false_at_recall_0p5",
                "false_at_recall_0p9",
            )
        },
        "C_fixed_validation_metrics": baseline,
        "validation_guardrail_fallback_used": fallback,
    }, grid


def retrieval_rows(frame: pd.DataFrame, scores: np.ndarray, method: str) -> pd.DataFrame:
    return v7.query_rank_rows(frame, scores, method)


def system_bootstrap_ci(
    frame: pd.DataFrame,
    scores: np.ndarray,
    seed: int,
) -> dict[str, float]:
    ranks = retrieval_rows(frame, scores, "bootstrap")
    systems = ranks.groupby(["family", "system_id"], as_index=False)["query_rank"].apply(list)
    grouped: dict[str, list[np.ndarray]] = {}
    for family, part in systems.groupby("family"):
        grouped[str(family)] = [np.asarray(values, dtype=np.int32) for values in part["query_rank"]]
    rng = np.random.default_rng(seed)
    r1 = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    r10 = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = []
        for values in grouped.values():
            indices = rng.integers(0, len(values), size=len(values))
            sampled.extend(values[index] for index in indices)
        query_ranks = np.concatenate(sampled)
        r1[draw] = np.mean(query_ranks <= 1)
        r10[draw] = np.mean(query_ranks <= 10)
    return {
        "system_bootstrap_r_at_1_ci_low": float(np.quantile(r1, 0.025)),
        "system_bootstrap_r_at_1_ci_high": float(np.quantile(r1, 0.975)),
        "system_bootstrap_r_at_10_ci_low": float(np.quantile(r10, 0.025)),
        "system_bootstrap_r_at_10_ci_high": float(np.quantile(r10, 0.975)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_unit": "lensed system, stratified by family; both directed queries retained",
    }


def metric_record(
    deployment: str,
    seed: int,
    split: str,
    method: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    assignment: pd.DataFrame,
    include_bootstrap: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "deployment": deployment,
        "seed": int(seed),
        "design_seed": DESIGN_SEEDS[seed],
        "split": split,
        "method": method,
        "n_events": int(frame["event_count"].iloc[0]),
        "n_pairs": int(len(frame)),
        "n_true_pairs": int(frame["is_true_pair"].sum()),
        "n_false_pairs": int((frame["is_true_pair"] == 0).sum()),
        "event_ood_fraction": float(assignment["template_ood"].mean()),
        "pair_ood_fraction": float(frame["sky_template_ood_pair"].mean()),
        **full_metrics(frame, scores),
    }
    if include_bootstrap:
        result.update(
            system_bootstrap_ci(
                frame, scores, stable_seed("bootstrap", deployment, seed, method)
            )
        )
    return result


def t_interval_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_r_at_1",
        "overall_r_at_10",
        "macro_r_at_1",
        "macro_r_at_10",
        "average_precision",
        "roc_auc",
        "false_at_recall_0p5",
        "false_at_recall_0p9",
        "event_ood_fraction",
        "pair_ood_fraction",
    ]
    rows = []
    for (deployment, split, method), part in frame.groupby(
        ["deployment", "split", "method"], sort=True
    ):
        row: dict[str, Any] = {
            "deployment": deployment,
            "split": split,
            "method": method,
            "n_seeds": int(len(part)),
            "ci_definition": "two-sided 95% Student-t interval across three frozen model/design seeds",
        }
        for metric in metrics:
            values = part[metric].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = 4.3026527299 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            low, high = mean - half, mean + half
            if metric not in {"false_at_recall_0p5", "false_at_recall_0p9"}:
                low, high = max(0.0, low), min(1.0, high)
            else:
                low = max(0.0, low)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def sky_distribution_records(
    deployment: str,
    seed: int,
    split: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for population, mask in (
        ("companion", frame["is_true_pair"].to_numpy(dtype=bool)),
        ("noncompanion", ~frame["is_true_pair"].to_numpy(dtype=bool)),
        ("noncompanion_in_domain", (~frame["is_true_pair"].to_numpy(dtype=bool)) & (~frame["sky_template_ood_pair"].to_numpy(dtype=bool))),
    ):
        for feature in ("sky_raw_log_bf", "sky_bc", "sky_j90"):
            values = frame.loc[mask, feature].to_numpy(dtype=np.float64)
            if len(values):
                rows.append(morph.distribution_summary(deployment, seed, split, population, feature, values))
    return rows


def critical_historical_paths() -> list[Path]:
    candidates = [
        PROJECT / "main.pdf",
        PROJECT / "results/gwtc_sky_resolution_v93_20260730/analysis_contract_v93.json",
        PROJECT / "results/gwtc_sky_resolution_v93_20260730/final_audit_summary_v93.json",
        PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830/final_audit_summary_v94.json",
        PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830/retrieval_metrics_summary_v94.csv",
    ]
    for deployment in DEPLOYMENTS:
        candidates.extend(
            [
                PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830" / deployment / "real_candidate_consensus_with_pe_v93.parquet",
                PROJECT / "results/gwtc_sky_resolution_v93_20260730" / deployment / "real_candidate_consensus_with_pe_v93.parquet",
            ]
        )
    return [path for path in candidates if path.is_file()]


def snapshot_hashes(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(path.resolve() for path in paths)):
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def initialize_root(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"Independent output already exists: {root}")
    for name in ("contracts", "results", "figures", "reports", "scripts", "logs", "manifest"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, root / "scripts" / SCRIPT_PATH.name)
    shutil.copy2(MORPH_SCRIPT, root / "scripts" / MORPH_SCRIPT.name)
    shutil.copy2(V94_SCRIPT, root / "scripts" / V94_SCRIPT.name)


def phase0(root: Path) -> None:
    initialize_root(root)
    run_ordering_unit_tests(root)
    historical_before = snapshot_hashes(critical_historical_paths())
    write_csv(root / "contracts/historical_authority_hashes_before.csv", historical_before)
    all_inventory: list[dict[str, Any]] = []
    all_ordering: list[pd.DataFrame] = []
    split_audits: list[dict[str, Any]] = []

    for deployment in DEPLOYMENTS:
        metadata = template_metadata(deployment, root)
        all_ordering.append(metadata)
        for seed in MODEL_SEEDS:
            design_seed = DESIGN_SEEDS[seed]
            roles = template_roles(metadata, design_seed)
            write_csv(root / f"contracts/{deployment}_seed_{seed}_template_roles.csv", roles)
            retained_by_split: dict[str, pd.DataFrame] = {}
            assignments: dict[str, pd.DataFrame] = {}
            for split in ("validation", "test"):
                retained = retained_event_plan(deployment, seed, split)
                retained_by_split[split] = retained
                write_csv(root / f"contracts/{deployment}_seed_{seed}_{split}_retained_events.csv", retained)
                assignment = build_template_assignment(deployment, seed, split, retained, roles)
                assignments[split] = assignment
                write_csv(root / f"contracts/{deployment}_seed_{seed}_{split}_template_assignment_prefreeze.csv", assignment)
                system_counts = retained.drop_duplicates("system_id")["family"].value_counts()
                all_inventory.append(
                    {
                        "deployment": deployment,
                        "model_seed": seed,
                        "design_seed": design_seed,
                        "split": split,
                        "n_events": len(retained),
                        "n_systems": retained["system_id"].nunique(),
                        "n_sis_systems": int(system_counts.get("SIS", 0)),
                        "n_pm_systems": int(system_counts.get("PM", 0)),
                        "n_unlensed_events": int(system_counts.get("unlensed", 0)),
                        "n_parent_noise_banks": retained["parent_noise_bank"].nunique(),
                        "n_actual_templates": assignment["template_event"].nunique(),
                        "template_ood_fraction": assignment["template_ood"].mean(),
                    }
                )
            source_overlap = set(retained_by_split["validation"]["system_id"]) & set(retained_by_split["test"]["system_id"])
            bank_overlap = set(retained_by_split["validation"]["parent_noise_bank"]) & set(retained_by_split["test"]["parent_noise_bank"])
            template_overlap = set(assignments["validation"]["template_event"]) & set(assignments["test"]["template_event"])
            split_audits.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "source_overlap_count": len(source_overlap),
                    "parent_noise_bank_overlap_count": len(bank_overlap),
                    "actual_template_overlap_count": len(template_overlap),
                    "passed": not source_overlap and not bank_overlap and not template_overlap,
                }
            )
    ordering = pd.concat(all_ordering, ignore_index=True)
    ordering_summary = {
        "n_public_pe_maps": int(len(ordering)),
        "n_nested": int((ordering["source_ordering"] == "NESTED").sum()),
        "n_ring": int((ordering["source_ordering"] == "RING").sum()),
        "n_converted_nested_to_ring": int(ordering["ordering_conversion_applied"].sum()),
        "all_output_ring": bool((ordering["output_ordering"] == "RING").all()),
        "all_normalized": True,
    }
    write_json(root / "contracts/public_pe_ordering_audit.json", ordering_summary)
    inventory = pd.DataFrame(all_inventory)
    split_audit = pd.DataFrame(split_audits)
    write_csv(root / "contracts/source_noise_template_split_inventory.csv", inventory)
    write_csv(root / "contracts/source_noise_template_disjointness_audit.csv", split_audit)
    if not split_audit["passed"].all() or not ordering_summary["all_output_ring"]:
        raise RuntimeError("Phase 0 ordering or disjointness audit failed")

    contract = {
        "experiment": VERSION,
        "created_utc": utc_stamp(),
        "status": FINAL_STATUS,
        "independent_output": str(root),
        "analysis_nside": ANALYSIS_NSIDE,
        "coarse_audit_nside": COARSE_NSIDE,
        "reference_audit_nside": REFERENCE_NSIDE,
        "formal_sky_score": "log(Npix * sum_pixel P_i P_j), maps are RING probability mass",
        "frozen": [
            "waveform encoder/checkpoints",
            "waveform_score (Z_wf)",
            "one-dimensional time_score (Z_time)",
            "v9.3 C-fixed per-seed weights",
            "original source-level validation/test split",
        ],
        "design_seeds": DESIGN_SEEDS,
        "model_seeds": list(MODEL_SEEDS),
        "parent_noise_bank_test_sets": {str(key): list(value) for key, value in TEST_NOISE_BANKS.items()},
        "map_matching": {
            "actual_template_events_disjoint_between_validation_and_test": True,
            "same_lens_images_use_distinct_templates": True,
            "dimensions": ["observing_run", "detector_network", "network_SNR", "A90", "KL", "HPD90_components"],
            "reference_descriptor_pool": "event-disjoint from actual validation/test map pools",
            "sparse_run_or_network_fallback": "allowed only with template_ood=true and neutral audit reporting",
        },
        "C_fixed_weights": FROZEN_V93_WEIGHTS,
        "C_retuned_grid": "non-negative normalized simplex, step 0.05, 231 points",
        "C_retuned_validation_objective": [
            "maximize pair AUPRC",
            "minimize F90",
            "minimize F50",
            "maximize macro R@10",
            "maximize min-family R@10",
            "maximize macro R@1",
            "prefer closest normalized C-fixed weights only for exact performance ties",
            "fixed lexicographic weight tie-break",
        ],
        "validation_guardrails": {
            "R10_noninferiority_margin_absolute": NOISE_NONINFERIORITY["r_at_10"],
            "AUPRC_noninferiority_margin_absolute": NOISE_NONINFERIORITY["average_precision"],
            "F50_and_F90_max_increase_during_selection": "5% plus one pair",
        },
        "test_adoption_gate": {
            "both_runs_required": True,
            "R10_noninferiority_margin_absolute": NOISE_NONINFERIORITY["r_at_10"],
            "AUPRC_noninferiority_margin_absolute": NOISE_NONINFERIORITY["average_precision"],
            "mean_F50_and_F90_must_not_increase": True,
            "all_three_seeds_must_meet_R10_and_AUPRC_noninferiority": True,
            "at_least_two_of_three_seeds_must_not_increase_each_false_burden": True,
            "at_least_one_mean_primary_metric_must_strictly_improve_to_prefer_C_retuned": True,
        },
        "selection_prohibitions": [
            "held-out test scores",
            "real-catalog PE consistency",
            "official FPP",
            "Hanabi overlap",
            "author preference",
        ],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "unit": "lensed system with both directed queries together",
            "stratified_by": ["SIS", "PM"],
        },
        "candidate_scope": "real strict H1-L1 BBH pairs; ranking only after config and test freeze",
        "claim_boundary": "candidate shortlist for Bayesian follow-up, not lensing detection",
    }
    contract_path = root / "contracts/ANALYSIS_CONTRACT.json"
    write_json(contract_path, contract)
    write_json(
        root / "contracts/PHASE0_COMPLETE.json",
        {
            "passed": True,
            "analysis_contract_sha256": sha256_file(contract_path),
            "test_pair_scores_opened": False,
            "real_catalog_used": False,
            "timestamp_utc": utc_stamp(),
        },
    )


def process_synthetic_split(
    root: Path,
    deployment: str,
    seed: int,
    split: str,
    allow_test_scores: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = pd.read_csv(root / f"contracts/{deployment}_public_pe_template_metadata.csv")
    events = pd.read_csv(root / f"contracts/{deployment}_seed_{seed}_{split}_retained_events.csv")
    assignment_path = root / f"contracts/{deployment}_seed_{seed}_{split}_template_assignment_prefreeze.csv"
    assignment = pd.read_csv(assignment_path)
    base = subset_pair_table(deployment, seed, split, events, allow_test_scores=allow_test_scores)
    maps, assignment, generation_seconds = generate_map_bank(deployment, seed, split, assignment, metadata)
    write_csv(root / f"results/{deployment}/seed_{seed}/{split}_template_assignment_with_rotation.csv", assignment)
    features = morph.compute_map_bank_features(maps, deployment=deployment, seed=seed, split=split)
    event_meta = build_event_meta(features, assignment)
    frame = morph.attach_pair_morphology(base, features, event_meta)
    frame = attach_pair_ood(frame, assignment)
    precision_audit = morph.float64_feature_audit(
        maps, features, frame, stable_seed("float64", deployment, seed, split)
    )
    if not precision_audit["passed"]:
        raise RuntimeError(f"Float64 sky audit failed: {precision_audit}")
    out_dir = root / "results" / deployment / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"{split}_pair_scores_map_matched_nside512.parquet", index=False)
    write_csv(out_dir / f"{split}_event_sky_metadata.csv", event_meta)
    write_json(out_dir / f"{split}_float64_sky_audit.json", precision_audit)
    resources = {
        **features.resources,
        "map_generation_seconds": generation_seconds,
        "n_retained_events": len(events),
        "n_retained_pairs": len(frame),
        "dense_maps_persisted": False,
    }
    del maps, features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame, assignment, resources


def phase1_validation(root: Path) -> None:
    selected: dict[str, Any] = {
        "experiment": VERSION,
        "created_utc": utc_stamp(),
        "selection_source": "new source/noise/template-disjoint validation only",
        "heldout_test_pair_scores_opened": False,
        "real_catalog_used": False,
        "deployments": {},
    }
    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        selected["deployments"][deployment] = {}
        for seed in MODEL_SEEDS:
            frame, assignment, resources = process_synthetic_split(
                root, deployment, seed, "validation", allow_test_scores=False
            )
            frozen = FROZEN_V93_WEIGHTS[deployment][seed]
            fixed_scores = score_vector(frame, frozen)
            retuned, grid = select_retuned_weights(frame, frozen)
            retuned_scores = score_vector(frame, retuned["weights"])
            seed_root = root / "results" / deployment / f"seed_{seed}"
            write_csv(seed_root / "C_retuned_validation_weight_grid.csv", grid)
            for method, scores in (
                ("C_fixed", fixed_scores),
                ("C_retuned", retuned_scores),
                ("waveform_only", frame["waveform_score"].to_numpy(float)),
                ("time_only", frame["time_score"].to_numpy(float)),
                ("sky_only", frame["sky_raw_log_bf"].to_numpy(float)),
            ):
                metric_rows.append(
                    metric_record(
                        deployment, seed, "validation", method, frame, scores, assignment, include_bootstrap=False
                    )
                )
            distribution_rows.extend(sky_distribution_records(deployment, seed, "validation", frame))
            resource_rows.append({"deployment": deployment, "seed": seed, "split": "validation", **resources})
            selected["deployments"][deployment][str(seed)] = {
                "model_seed": seed,
                "design_seed": DESIGN_SEEDS[seed],
                "C_fixed_raw_weights": frozen,
                "C_fixed_normalized_weights": normalized_weights(frozen),
                "C_retuned": retuned,
                "validation_event_count": int(frame["event_count"].iloc[0]),
                "validation_pair_count": int(len(frame)),
                "test_pair_scores_used": False,
                "real_candidate_information_used": False,
            }
    write_csv(root / "results/validation_metrics_per_seed.csv", pd.DataFrame(metric_rows))
    write_csv(root / "results/sky_distribution_statistics_validation.csv", pd.DataFrame(distribution_rows))
    write_csv(root / "results/resource_usage_validation.csv", pd.DataFrame(resource_rows))
    selected_path = root / "contracts/selected_config.json"
    write_json(selected_path, selected)
    selected_hash = sha256_file(selected_path)
    write_json(
        root / "contracts/SELECTED_CONFIG_FROZEN.json",
        {
            "selected_config_sha256": selected_hash,
            "timestamp_utc": utc_stamp(),
            "heldout_test_pair_scores_opened": False,
            "real_catalog_used": False,
        },
    )


def test_gate(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    deployment_pass: dict[str, bool] = {}
    for deployment in DEPLOYMENTS:
        part = metrics.loc[(metrics["deployment"] == deployment) & (metrics["split"] == "test")]
        fixed = part.loc[part["method"] == "C_fixed"].set_index("seed")
        retuned = part.loc[part["method"] == "C_retuned"].set_index("seed")
        per_seed = []
        for seed in MODEL_SEEDS:
            delta_r10 = float(retuned.loc[seed, "macro_r_at_10"] - fixed.loc[seed, "macro_r_at_10"])
            delta_ap = float(retuned.loc[seed, "average_precision"] - fixed.loc[seed, "average_precision"])
            delta_f50 = int(retuned.loc[seed, "false_at_recall_0p5"] - fixed.loc[seed, "false_at_recall_0p5"])
            delta_f90 = int(retuned.loc[seed, "false_at_recall_0p9"] - fixed.loc[seed, "false_at_recall_0p9"])
            row = {
                "deployment": deployment,
                "seed": seed,
                "delta_macro_r_at_10": delta_r10,
                "delta_average_precision": delta_ap,
                "delta_false_at_recall_0p5": delta_f50,
                "delta_false_at_recall_0p9": delta_f90,
                "r10_noninferior": delta_r10 >= -NOISE_NONINFERIORITY["r_at_10"],
                "auprc_noninferior": delta_ap >= -NOISE_NONINFERIORITY["average_precision"],
                "f50_not_increased": delta_f50 <= 0,
                "f90_not_increased": delta_f90 <= 0,
            }
            per_seed.append(row)
            rows.append(row)
        mean_delta_r10 = float(np.mean([row["delta_macro_r_at_10"] for row in per_seed]))
        mean_delta_ap = float(np.mean([row["delta_average_precision"] for row in per_seed]))
        mean_delta_f50 = float(np.mean([row["delta_false_at_recall_0p5"] for row in per_seed]))
        mean_delta_f90 = float(np.mean([row["delta_false_at_recall_0p9"] for row in per_seed]))
        passed = bool(
            all(row["r10_noninferior"] and row["auprc_noninferior"] for row in per_seed)
            and sum(row["f50_not_increased"] for row in per_seed) >= 2
            and sum(row["f90_not_increased"] for row in per_seed) >= 2
            and mean_delta_f50 <= 0
            and mean_delta_f90 <= 0
            and (mean_delta_r10 > 0 or mean_delta_ap > 0)
        )
        deployment_pass[deployment] = passed
    summary = {
        "deployment_pass": deployment_pass,
        "C_retuned_preferred_over_C_fixed": bool(all(deployment_pass.values())),
        "decision": (
            "C_retuned passes the preregistered paired guardrails"
            if all(deployment_pass.values())
            else "C_retuned does not pass all preregistered paired guardrails; retain C-fixed for this confirmation"
        ),
        "status": FINAL_STATUS,
    }
    return pd.DataFrame(rows), summary


def phase2_locked_test(root: Path) -> None:
    freeze_path = root / "contracts/SELECTED_CONFIG_FROZEN.json"
    selected_path = root / "contracts/selected_config.json"
    if not freeze_path.is_file() or not selected_path.is_file():
        raise RuntimeError("Selected configuration is not frozen")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha256_file(selected_path) != freeze["selected_config_sha256"]:
        raise RuntimeError("selected_config.json changed after freeze")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    marker = root / "contracts/LOCKED_TEST_OPENED_ONCE.json"
    if marker.exists():
        raise RuntimeError("Locked test has already been opened")
    write_json(
        marker,
        {
            "timestamp_utc": utc_stamp(),
            "selected_config_sha256": freeze["selected_config_sha256"],
            "reason": "single preregistered held-out evaluation",
        },
    )
    metric_rows = pd.read_csv(root / "results/validation_metrics_per_seed.csv").to_dict("records")
    distribution_rows = pd.read_csv(root / "results/sky_distribution_statistics_validation.csv").to_dict("records")
    resource_rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        for seed in MODEL_SEEDS:
            frame, assignment, resources = process_synthetic_split(
                root, deployment, seed, "test", allow_test_scores=True
            )
            config = selected["deployments"][deployment][str(seed)]
            methods = {
                "C_fixed": score_vector(frame, config["C_fixed_raw_weights"]),
                "C_retuned": score_vector(frame, config["C_retuned"]["weights"]),
                "waveform_only": frame["waveform_score"].to_numpy(float),
                "time_only": frame["time_score"].to_numpy(float),
                "sky_only": frame["sky_raw_log_bf"].to_numpy(float),
            }
            score_output = frame.copy()
            for method, scores in methods.items():
                score_output[f"score_{method}"] = scores
                score_output[f"rank_{method}"] = pd.Series(scores).rank(method="min", ascending=False).astype(int)
                metric_rows.append(
                    metric_record(
                        deployment,
                        seed,
                        "test",
                        method,
                        frame,
                        scores,
                        assignment,
                        include_bootstrap=method in {"C_fixed", "C_retuned"},
                    )
                )
            score_output.to_parquet(
                root / f"results/{deployment}/seed_{seed}/locked_test_pair_scores_C_fixed_C_retuned.parquet",
                index=False,
            )
            distribution_rows.extend(sky_distribution_records(deployment, seed, "test", frame))
            resource_rows.append({"deployment": deployment, "seed": seed, "split": "test", **resources})
    metrics = pd.DataFrame(metric_rows)
    write_csv(root / "results/retrieval_pair_metrics_per_seed.csv", metrics)
    write_csv(root / "results/retrieval_pair_metrics_summary.csv", t_interval_summary(metrics))
    write_csv(root / "results/sky_distribution_statistics.csv", pd.DataFrame(distribution_rows))
    write_csv(root / "results/resource_usage_test.csv", pd.DataFrame(resource_rows))
    gate_rows, gate_summary = test_gate(metrics)
    write_csv(root / "results/C_retuned_vs_C_fixed_paired_gate_per_seed.csv", gate_rows)
    write_json(root / "results/C_retuned_vs_C_fixed_gate_summary.json", gate_summary)
    write_json(
        root / "contracts/PHASE2_LOCKED_TEST_COMPLETE.json",
        {
            "passed": True,
            "timestamp_utc": utc_stamp(),
            "locked_test_evaluated_once": True,
            "selected_config_sha256": freeze["selected_config_sha256"],
            "real_catalog_used_for_selection": False,
        },
    )


def real_event_meta(features: Any, manifest: pd.DataFrame) -> pd.DataFrame:
    meta = features.event.copy()
    extra = pd.DataFrame(
        {
            "idx": np.arange(len(manifest), dtype=np.int32),
            "snr": pd.to_numeric(manifest["network_snr"], errors="coerce").to_numpy(float),
            "detector_count": manifest["detectors_available"].map(
                lambda value: len(canonical_network(value).split("+"))
                if canonical_network(value) != "UNKNOWN"
                else 0
            ).to_numpy(int),
            "hpd90_component_count": np.ones(len(manifest), dtype=np.int32),
            "event_name": manifest["event_name"].astype(str).to_numpy(),
        }
    )
    return meta.merge(extra, on="idx", how="left", validate="one_to_one").sort_values("idx")


def load_real_map_features(
    root: Path,
    deployment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = primary_manifest(deployment)
    maps: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for index, event in manifest.iterrows():
        probability, metadata = corrected_read_probability_map(event, ANALYSIS_NSIDE)
        maps.append(probability)
        audit_rows.append({"idx": index, "event_name": event["event_name"], **metadata})
    bank = np.stack(maps).astype(np.float32)
    features = morph.compute_map_bank_features(bank, deployment=deployment, seed=0, split="real")
    event_meta = real_event_meta(features, manifest)
    ii, jj = np.triu_indices(len(manifest), k=1)
    base = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": manifest.iloc[ii]["event_name"].astype(str).to_numpy(),
            "event_j": manifest.iloc[jj]["event_name"].astype(str).to_numpy(),
        }
    )
    morphology = morph.attach_pair_morphology(base, features, event_meta)
    morphology["pair_key"] = [
        canonical_pair(a, b) for a, b in zip(morphology["event_i"], morphology["event_j"])
    ]
    precision = morph.float64_feature_audit(
        bank,
        features,
        morphology.assign(is_true_pair=0),
        stable_seed("real-float64", deployment),
    )
    if not precision["passed"]:
        raise RuntimeError(f"Real-map float64 audit failed: {precision}")
    dep_root = root / "results" / deployment
    dep_root.mkdir(parents=True, exist_ok=True)
    morphology.to_parquet(dep_root / "real_pair_sky_features_corrected_nside512.parquet", index=False)
    write_csv(dep_root / "real_event_sky_features_corrected_nside512.csv", event_meta)
    write_csv(dep_root / "real_public_pe_ordering_audit.csv", pd.DataFrame(audit_rows))
    write_json(dep_root / "real_float64_sky_audit.json", precision)
    del bank, maps, features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return morphology, manifest


def rank_real(
    frame: pd.DataFrame,
    weights: dict[str, float],
    method: str,
    seed: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["final_score"] = score_vector(out, weights)
    out["waveform_contribution"] = float(weights["waveform"]) * out["waveform_score"]
    out["time_contribution"] = float(weights["time"]) * out["time_score"]
    out["sky_contribution"] = float(weights["sky"]) * out["sky_raw_log_bf"]
    out["method"] = method
    out["seed"] = int(seed)
    out = out.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def consensus_real(frames: list[pd.DataFrame], method: str) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    value_columns = [
        "final_score",
        "waveform_score",
        "time_score",
        "sky_raw_log_bf",
        "sky_bc",
        "sky_j50",
        "sky_j90",
        "waveform_contribution",
        "time_contribution",
        "sky_contribution",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "seed_count": ("seed", "nunique"),
        "rank_mean": ("rank", "mean"),
        "rank_sd": ("rank", "std"),
        "rank_min": ("rank", "min"),
        "rank_max": ("rank", "max"),
    }
    for column in value_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
    out = (
        combined.groupby(["pair_key", "event_i", "event_j"], as_index=False)
        .agg(**aggregations)
        .sort_values(["rank_mean", "rank_max", "final_score_mean"], ascending=[True, True, False], kind="stable")
        .reset_index(drop=True)
    )
    out.insert(0, "method", method)
    out.insert(0, "consensus_rank", np.arange(1, len(out) + 1, dtype=np.int32))
    return out


def add_official_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["official_po_ml_fpp"] = np.nan
    out["official_po_phazap_fpp"] = np.nan
    out["official_tier"] = "not_available_in_frozen_local_authoritative_table"
    out["official_fast_golum_stage"] = "not_available_in_frozen_local_authoritative_table"
    out["official_hanabi_stage"] = np.where(
        out.get("public_hanabi_table_overlap", False),
        "listed_as_LVK_literature_candidate_not_confirmed; numeric Hanabi result not available locally",
        "not_available_in_frozen_local_authoritative_table",
    )
    out["official_numeric_fpp_status"] = (
        "requested fields retained as NA; no machine-readable PO/ML or PO/Phazap FPP was found in the frozen project inputs"
    )
    return out


def phase3_real_catalog(root: Path) -> None:
    selected = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    official_path = (
        PROJECT
        / "results/sky_background_fast_followup_v104_20260824_20260824T080136Z/tables/frozen_lvk_historical_candidate_comparison.csv"
    )
    official = pd.read_csv(official_path)
    all_top50: list[pd.DataFrame] = []
    budget_rows: list[dict[str, Any]] = []
    rank_changes: list[pd.DataFrame] = []
    for deployment in DEPLOYMENTS:
        morphology, manifest = load_real_map_features(root, deployment)
        morphology_columns = [
            "idx_i",
            "idx_j",
            "event_i",
            "event_j",
            "pair_key",
            "sky_raw_log_bf",
            "sky_bc",
            "sky_j50",
            "sky_j90",
            "sky_area90_deg2_i",
            "sky_area90_deg2_j",
            "sky_area90_ratio",
        ]
        by_method: dict[str, list[pd.DataFrame]] = {"C_fixed": [], "C_retuned": []}
        for seed in MODEL_SEEDS:
            base_path = corrected_seed_dir(deployment, seed) / "real_pair_features_unified_sky_v81.parquet"
            base = pd.read_parquet(base_path)
            base = base.drop(columns=[column for column in base.columns if column.startswith("sky_")], errors="ignore")
            frame = base.merge(
                morphology[morphology_columns],
                on=["idx_i", "idx_j", "event_i", "event_j"],
                how="left",
                validate="one_to_one",
            )
            frame = frame.loc[frame["strict_h1l1_bbh_pair"].fillna(False).astype(bool)].copy().reset_index(drop=True)
            frame["pair_key"] = [canonical_pair(a, b) for a, b in zip(frame["event_i"], frame["event_j"])]
            config = selected["deployments"][deployment][str(seed)]
            for method, weights in (
                ("C_fixed", config["C_fixed_raw_weights"]),
                ("C_retuned", config["C_retuned"]["weights"]),
            ):
                ranked = rank_real(frame, weights, method, seed)
                seed_root = root / "results" / deployment / f"seed_{seed}"
                ranked.to_parquet(seed_root / f"real_strict_pair_scores_{method}.parquet", index=False)
                write_csv(seed_root / f"real_strict_top50_{method}.csv", ranked.head(50))
                by_method[method].append(ranked)

        pe_path = (
            PROJECT
            / "results/gwtc_sky_ordering_corrected_v94_20260830"
            / deployment
            / "real_candidate_consensus_with_pe_v93.parquet"
        )
        pe = pd.read_parquet(pe_path)
        enriched: dict[str, pd.DataFrame] = {}
        for method, frames in by_method.items():
            consensus = consensus_real(frames, method)
            events = set(consensus.head(50)["event_i"].astype(str)) | set(consensus.head(50)["event_j"].astype(str))
            distances = morph.load_distance_posteriors(deployment, events)
            result = morph.attach_pe_official(deployment, consensus, pe, official, distances)
            result = add_official_columns(result)
            dep_root = root / "results" / deployment
            result.to_parquet(dep_root / f"real_consensus_with_pe_official_{method}.parquet", index=False)
            write_csv(dep_root / f"real_top50_with_pe_official_{method}.csv", result.head(50))
            enriched[method] = result
            top50 = result.head(50).copy()
            top50.insert(0, "deployment", deployment)
            all_top50.append(top50)
            for budget in (10, 20, 50):
                chosen = result.head(budget)
                budget_rows.append(
                    {
                        "deployment": deployment,
                        "method": method,
                        "budget": budget,
                        "n_pairs": len(chosen),
                        "chirp_mass_BC_ge_0p5": int((chosen["chirp_mass_bhattacharyya_coefficient"] >= 0.5).sum()),
                        "median_chirp_mass_BC": float(chosen["chirp_mass_bhattacharyya_coefficient"].median()),
                        "Dmax_le_3": int((chosen["max_standardized_posterior_distance"] <= 3).sum()),
                        "official_frontend_overlap": int(chosen["official_frontend_overlap"].sum()),
                        "public_hanabi_table_overlap": int(chosen["public_hanabi_table_overlap"].sum()),
                    }
                )
        fixed = enriched["C_fixed"].set_index("pair_key")
        retuned = enriched["C_retuned"].set_index("pair_key")
        change = fixed[["consensus_rank", "event_i", "event_j"]].join(
            retuned[["consensus_rank"]], lsuffix="_C_fixed", rsuffix="_C_retuned", how="outer"
        )
        change["rank_change_retuned_minus_fixed"] = (
            change["consensus_rank_C_retuned"] - change["consensus_rank_C_fixed"]
        )
        change["deployment"] = deployment
        change["pair_key"] = change.index
        rank_changes.append(change.reset_index(drop=True))
    top50_frame = pd.concat(all_top50, ignore_index=True)
    write_csv(root / "results/real_top50_C_fixed_C_retuned.csv", top50_frame)
    top50_frame.to_parquet(root / "results/real_top50_C_fixed_C_retuned.parquet", index=False)
    write_csv(root / "results/real_PE_official_budget_summary.csv", pd.DataFrame(budget_rows))
    write_csv(root / "results/real_rank_change_C_retuned_vs_C_fixed.csv", pd.concat(rank_changes, ignore_index=True))
    write_json(
        root / "contracts/PHASE3_REAL_CATALOG_COMPLETE.json",
        {
            "passed": True,
            "timestamp_utc": utc_stamp(),
            "real_catalog_used_for_weight_selection": False,
            "Hanabi_run": False,
            "full_BBH_PE_run": False,
            "candidate_claim": "ranked catalog coincidences for follow-up, not lensing detections",
        },
    )


def z_sky(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    return float(np.log(max(len(left) * np.sum(left * right, dtype=np.float64), RAW_BF_FLOOR)))


def selected_audit_pairs(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    labels = frame["is_true_pair"].to_numpy(dtype=bool)
    selected_rows: list[pd.DataFrame] = []
    true = frame.loc[labels].copy()
    for family, part in true.groupby("true_pair_family", sort=True):
        part = part.copy()
        part["audit_hash"] = [
            stable_seed("resolution-true", seed, int(i), int(j))
            for i, j in zip(part["idx_i"], part["idx_j"])
        ]
        selected = part.sort_values("audit_hash", kind="stable").head(15).copy()
        selected["audit_population"] = f"true_{family}"
        selected_rows.append(selected)
    false = frame.loc[~labels].sort_values("score_C_fixed", ascending=False, kind="stable").head(30).copy()
    false["audit_population"] = "high_score_noncompanion"
    selected_rows.append(false)
    return pd.concat(selected_rows, ignore_index=True).drop_duplicates(["idx_i", "idx_j"])


def synthetic_maps_at_resolution(
    deployment: str,
    seed: int,
    split: str,
    assignment: pd.DataFrame,
    event_indices: Iterable[int],
    nside: int,
) -> dict[int, np.ndarray]:
    chosen = assignment.loc[assignment["idx"].isin(set(map(int, event_indices)))].copy()
    metadata = pd.read_csv(
        assignment.attrs.get(
            "metadata_path",
            PROJECT / "__unused__",
        )
    ) if assignment.attrs.get("metadata_path") else pd.DataFrame()
    template_maps = load_template_maps(
        deployment,
        metadata,
        chosen["template_index"].astype(int),
        nside,
    )
    npix = hp.nside2npix(nside)
    output_vectors = np.asarray(hp.pix2vec(nside, np.arange(npix), nest=False), dtype=np.float64)
    result: dict[int, np.ndarray] = {}

    def task(row: Any) -> tuple[int, np.ndarray]:
        source = np.asarray([row.source_vector_x, row.source_vector_y, row.source_vector_z], dtype=np.float64)
        return (
            int(row.idx),
            rotate_with_frozen_geometry(
                template_maps[int(row.template_index)],
                source,
                float(row.roll_angle_rad),
                float(row.ra_true),
                float(row.dec_true),
                output_vectors,
            ),
        )

    with ThreadPoolExecutor(max_workers=ROTATION_WORKERS) as executor:
        futures = [executor.submit(task, row) for row in chosen.itertuples(index=False)]
        for future in as_completed(futures):
            index, probability = future.result()
            result[index] = probability
    del output_vectors, template_maps
    gc.collect()
    return result


def convergence_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(group_columns, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        delta = np.abs(part["z_sky_nside512"] - part["z_sky_nside1024"])
        rho = spearmanr(part["z_sky_nside512"], part["z_sky_nside1024"]).statistic if len(part) > 2 else np.nan
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_pairs": len(part),
                "median_abs_delta_512_1024": float(np.median(delta)),
                "q90_abs_delta_512_1024": float(np.quantile(delta, 0.90)),
                "q99_abs_delta_512_1024": float(np.quantile(delta, 0.99)),
                "max_abs_delta_512_1024": float(np.max(delta)),
                "sign_flip_512_1024_count": int(
                    (np.sign(part["z_sky_nside512"]) != np.sign(part["z_sky_nside1024"])).sum()
                ),
                "spearman_512_1024": float(rho) if np.isfinite(rho) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def phase4_resolution_audit(root: Path) -> None:
    injection_path = root / "results/injection_sky_resolution_audit_pairs.csv"
    if injection_path.is_file():
        injection = pd.read_csv(injection_path)
    else:
        injection_rows: list[dict[str, Any]] = []
        for deployment in DEPLOYMENTS:
            metadata_path = root / f"contracts/{deployment}_public_pe_template_metadata.csv"
            for seed in MODEL_SEEDS:
                seed_root = root / "results" / deployment / f"seed_{seed}"
                frame = pd.read_parquet(seed_root / "locked_test_pair_scores_C_fixed_C_retuned.parquet")
                selected = selected_audit_pairs(frame, DESIGN_SEEDS[seed])
                assignment = pd.read_csv(seed_root / "test_template_assignment_with_rotation.csv")
                assignment.attrs["metadata_path"] = str(metadata_path)
                event_indices = set(selected["idx_i"].astype(int)) | set(selected["idx_j"].astype(int))
                maps_by_nside: dict[int, dict[int, np.ndarray]] = {}
                for nside in (COARSE_NSIDE, REFERENCE_NSIDE):
                    maps_by_nside[nside] = synthetic_maps_at_resolution(
                        deployment, seed, "test", assignment, event_indices, nside
                    )
                for row in selected.itertuples(index=False):
                    record = {
                        "deployment": deployment,
                        "seed": seed,
                        "design_seed": DESIGN_SEEDS[seed],
                        "audit_population": row.audit_population,
                        "idx_i": int(row.idx_i),
                        "idx_j": int(row.idx_j),
                        "is_true_pair": int(row.is_true_pair),
                        "z_sky_nside256": z_sky(maps_by_nside[COARSE_NSIDE][int(row.idx_i)], maps_by_nside[COARSE_NSIDE][int(row.idx_j)]),
                        "z_sky_nside512": float(row.sky_raw_log_bf),
                        "z_sky_nside1024": z_sky(maps_by_nside[REFERENCE_NSIDE][int(row.idx_i)], maps_by_nside[REFERENCE_NSIDE][int(row.idx_j)]),
                    }
                    record["sign_stable_256_512_1024"] = len(
                        {int(np.sign(record[f"z_sky_nside{nside}"])) for nside in (256, 512, 1024)}
                    ) == 1
                    injection_rows.append(record)
                del maps_by_nside
                gc.collect()
        injection = pd.DataFrame(injection_rows)
        write_csv(injection_path, injection)
        write_csv(
            root / "results/injection_sky_resolution_audit_summary.csv",
            convergence_summary(injection, ["deployment", "seed", "audit_population"]),
        )

    real_rows: list[dict[str, Any]] = []
    top50 = pd.read_csv(root / "results/real_top50_C_fixed_C_retuned.csv")
    for deployment in DEPLOYMENTS:
        # Keep each method's full Top-50.  Repeated physical pairs across the
        # two methods are intentionally evaluated twice in the method-stratified
        # summary; map values remain identical, while provenance stays clear.
        selected_pairs = top50.loc[top50["deployment"] == deployment].copy()
        manifest = primary_manifest(deployment).set_index("event_name")
        needed = set(selected_pairs["event_i"].astype(str)) | set(selected_pairs["event_j"].astype(str))
        maps: dict[int, dict[str, np.ndarray]] = {COARSE_NSIDE: {}, REFERENCE_NSIDE: {}}
        for event_name in sorted(needed):
            event = manifest.loc[event_name]
            for nside in (COARSE_NSIDE, REFERENCE_NSIDE):
                maps[nside][event_name], _ = corrected_read_probability_map(event, nside)
        for row in selected_pairs.itertuples(index=False):
            record = {
                "deployment": deployment,
                "method": row.method,
                "consensus_rank": int(row.consensus_rank),
                "pair_key": row.pair_key,
                "event_i": row.event_i,
                "event_j": row.event_j,
                "z_sky_nside256": z_sky(maps[COARSE_NSIDE][row.event_i], maps[COARSE_NSIDE][row.event_j]),
                "z_sky_nside512": float(row.sky_raw_log_bf_mean),
                "z_sky_nside1024": z_sky(maps[REFERENCE_NSIDE][row.event_i], maps[REFERENCE_NSIDE][row.event_j]),
            }
            record["sign_stable_256_512_1024"] = len(
                {int(np.sign(record[f"z_sky_nside{nside}"])) for nside in (256, 512, 1024)}
            ) == 1
            real_rows.append(record)
        del maps
        gc.collect()
    real = pd.DataFrame(real_rows)
    write_csv(root / "results/real_candidate_head_sky_resolution_audit.csv", real)
    write_csv(
        root / "results/real_candidate_head_sky_resolution_summary.csv",
        convergence_summary(real, ["deployment", "method"]),
    )
    write_json(
        root / "contracts/PHASE4_RESOLUTION_AUDIT_COMPLETE.json",
        {
            "passed": True,
            "timestamp_utc": utc_stamp(),
            "formal_ranking_nside": ANALYSIS_NSIDE,
            "resolution_audit_used_for_retuning": False,
            "audit_scope": "preregistered true companions, high-score noncompanions, and real Top-50 union",
        },
    )


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figures(root: Path) -> None:
    set_plot_style()
    metrics = pd.read_csv(root / "results/retrieval_pair_metrics_per_seed.csv")
    test = metrics.loc[
        (metrics["split"] == "test") & metrics["method"].isin(["C_fixed", "C_retuned"])
    ]
    methods = ["C_fixed", "C_retuned"]
    colors = {"C_fixed": "#4C78A8", "C_retuned": "#E45756"}
    measures = [
        ("overall_r_at_1", "R@1"),
        ("overall_r_at_10", "R@10"),
        ("average_precision", "Pair AUPRC"),
        ("false_at_recall_0p5", "False pairs at 50% recall"),
        ("false_at_recall_0p9", "False pairs at 90% recall"),
        ("pair_ood_fraction", "Pair OOD fraction"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 5.8), constrained_layout=True)
    for axis, (measure, label), letter in zip(axes.flat, measures, "abcdef"):
        positions = []
        tick_labels = []
        position = 0
        for deployment in DEPLOYMENTS:
            for method in methods:
                values = test.loc[
                    (test["deployment"] == deployment) & (test["method"] == method), measure
                ].to_numpy(float)
                jitter = np.linspace(-0.06, 0.06, len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=23,
                    color=colors[method],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                axis.plot(position, np.mean(values), marker="_", color="black", markersize=10)
                positions.append(position)
                tick_labels.append(f"{deployment.upper()}\n{method.replace('_', ' ')}")
                position += 1
            position += 0.35
        axis.set_xticks(positions, tick_labels, rotation=20, ha="right")
        axis.set_ylabel(label)
        axis.set_title(f"{letter}  Held-out test", loc="left")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(root / "figures/fig_C_fixed_vs_C_retuned_metrics.pdf", bbox_inches="tight")
    figure.savefig(root / "figures/fig_C_fixed_vs_C_retuned_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    for axis, deployment, letter in zip(axes, DEPLOYMENTS, "ab"):
        path = root / f"results/{deployment}/seed_{MODEL_SEEDS[0]}/locked_test_pair_scores_C_fixed_C_retuned.parquet"
        frame = pd.read_parquet(path, columns=["is_true_pair", "sky_raw_log_bf"])
        true = frame.loc[frame["is_true_pair"].astype(bool), "sky_raw_log_bf"].to_numpy(float)
        false = frame.loc[~frame["is_true_pair"].astype(bool), "sky_raw_log_bf"].to_numpy(float)
        if len(false) > 20_000:
            rng = np.random.default_rng(stable_seed("plot", deployment))
            false = rng.choice(false, 20_000, replace=False)
        low = min(np.quantile(false, 0.005), np.quantile(true, 0.005))
        high = max(np.quantile(false, 0.995), np.quantile(true, 0.995))
        bins = np.linspace(low, high, 55)
        axis.hist(false, bins=bins, density=True, histtype="step", linewidth=1.3, color="#777777", label="non-companion")
        axis.hist(true, bins=bins, density=True, histtype="step", linewidth=1.5, color="#D62728", label="companion")
        axis.axvline(0, color="black", linewidth=0.7, linestyle="--")
        axis.set_xlabel(r"Corrected $Z_{\rm sky}$ at Nside=512")
        axis.set_ylabel("Density")
        axis.set_title(f"{letter}  {deployment.upper()} map-matched test", loc="left")
        axis.legend(loc="upper left")
    figure.savefig(root / "figures/fig_corrected_sky_distributions.pdf", bbox_inches="tight")
    figure.savefig(root / "figures/fig_corrected_sky_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    top = pd.read_csv(root / "results/real_top50_C_fixed_C_retuned.csv")
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    for row_index, deployment in enumerate(DEPLOYMENTS):
        for column, method in enumerate(methods):
            axis = axes[row_index, column]
            part = top.loc[
                (top["deployment"] == deployment) & (top["method"] == method)
            ].sort_values("consensus_rank").head(10)
            y = np.arange(len(part))
            axis.barh(y, part["waveform_contribution_mean"], color="#4C78A8", label="waveform")
            axis.barh(
                y,
                part["time_contribution_mean"],
                left=part["waveform_contribution_mean"],
                color="#F2CF5B",
                label="time",
            )
            axis.barh(
                y,
                part["sky_contribution_mean"],
                left=part["waveform_contribution_mean"] + part["time_contribution_mean"],
                color="#54A24B",
                label="sky",
            )
            labels = [f"#{int(rank)} {a}--{b}" for rank, a, b in zip(part["consensus_rank"], part["event_i"], part["event_j"])]
            axis.set_yticks(y, labels, fontsize=6.5)
            axis.invert_yaxis()
            axis.set_xlabel("Mean weighted contribution")
            axis.set_title(f"{'abcd'[row_index * 2 + column]}  {deployment.upper()} {method}", loc="left")
            if row_index == 0 and column == 1:
                axis.legend(loc="lower right", ncol=3)
    figure.savefig(root / "figures/fig_real_top10_channel_contributions.pdf", bbox_inches="tight")
    figure.savefig(root / "figures/fig_real_top10_channel_contributions.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    convergence = pd.read_csv(root / "results/injection_sky_resolution_audit_pairs.csv")
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    for axis, deployment, letter in zip(axes, DEPLOYMENTS, "ab"):
        part = convergence.loc[convergence["deployment"] == deployment]
        for population, group in part.groupby("audit_population"):
            axis.scatter(
                group["z_sky_nside512"],
                group["z_sky_nside1024"],
                s=12,
                alpha=0.65,
                label=population,
            )
        low = min(part["z_sky_nside512"].min(), part["z_sky_nside1024"].min())
        high = max(part["z_sky_nside512"].max(), part["z_sky_nside1024"].max())
        axis.plot([low, high], [low, high], color="black", linewidth=0.7, linestyle="--")
        axis.set_xlabel(r"$Z_{\rm sky}$, Nside=512")
        axis.set_ylabel(r"$Z_{\rm sky}$, Nside=1024")
        axis.set_title(f"{letter}  {deployment.upper()} resolution audit", loc="left")
        axis.legend(loc="best", fontsize=6.5)
    figure.savefig(root / "figures/fig_sky_resolution_audit.pdf", bbox_inches="tight")
    figure.savefig(root / "figures/fig_sky_resolution_audit.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                cells.append("NA")
            elif column in formats:
                cells.append(formats[column].format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(root: Path, package_path: Path | None = None) -> None:
    contract = json.loads((root / "contracts/ANALYSIS_CONTRACT.json").read_text(encoding="utf-8"))
    selected = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "results/C_retuned_vs_C_fixed_gate_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(root / "results/retrieval_pair_metrics_summary.csv")
    test = metrics.loc[(metrics["split"] == "test") & metrics["method"].isin(["C_fixed", "C_retuned"])]
    inventory = pd.read_csv(root / "contracts/source_noise_template_split_inventory.csv")
    ordering = json.loads((root / "contracts/public_pe_ordering_audit.json").read_text(encoding="utf-8"))
    convergence = pd.read_csv(root / "results/injection_sky_resolution_audit_summary.csv")
    pe_budget = pd.read_csv(root / "results/real_PE_official_budget_summary.csv")
    top = pd.read_csv(root / "results/real_top50_C_fixed_C_retuned.csv")

    metric_view = test[
        [
            "deployment",
            "method",
            "overall_r_at_1_mean",
            "overall_r_at_10_mean",
            "average_precision_mean",
            "false_at_recall_0p5_mean",
            "false_at_recall_0p9_mean",
            "pair_ood_fraction_mean",
        ]
    ].copy()
    weight_rows = []
    for deployment in DEPLOYMENTS:
        for seed in MODEL_SEEDS:
            config = selected["deployments"][deployment][str(seed)]
            weight_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    "C_fixed": "/".join(f"{config['C_fixed_raw_weights'][name]:.2f}" for name in ("waveform", "time", "sky")),
                    "C_retuned": "/".join(f"{config['C_retuned']['weights'][name]:.2f}" for name in ("waveform", "time", "sky")),
                }
            )
    lines = [
        "# V93 天空图错误与 C 方案独立确认实验报告",
        "",
        f"**最终状态：`{FINAL_STATUS}`**",
        "",
        "> 本轮为独立确认实验。它修正公开 PE HDF5 的 NESTED/RING 读取，重建 Nside=512 map-matched 注入，并比较 C-fixed 与 validation-only C-retuned。结果没有覆盖 v9.3/v9.4、历史候选、论文或 SI，也不是透镜探测声明。",
        "",
        "## 1. 冻结范围",
        "",
        "- waveform encoder/checkpoint、`Z_wf` 和一维 `Z_time` 完全冻结；没有重训 encoder。",
        "- C-fixed 使用 v9.3 每个 seed 的原始权重比例；C-retuned 只在新的 validation 上从 0.05 步长 simplex 网格选择。",
        "- 真实 GWTC 的 PE、官方候选资料和 Hanabi 重合没有参与选权。",
        "- 本轮没有运行大规模 BBH PE，也没有运行 Hanabi。",
        "",
        "## 2. Ordering 修复",
        "",
        f"共审计 {ordering['n_public_pe_maps']} 张公开 PE 图，其中 {ordering['n_nested']} 张标记为 NESTED；全部显式转换为 RING 后再降/升到共同 Nside。输出全为 RING：{ordering['all_output_ring']}。",
        "`parse_boolish([b'True'])`、False 情形和合成 NESTED→RING round-trip 单元测试均通过。",
        "",
        "## 3. 独立 map-matched 注入",
        "",
        "每个模拟事件先按 run、H1/L1 network、SNR 匹配，再用独立 reference-template 描述符匹配 A90、KL 和 HPD 多峰数。透镜两像共享真实天空方向，但强制使用两个不同公开 PE 模板。validation/test 实际模板事件完全不交叉。",
        "旧 validation/test 共享父噪声 bank，因此本轮按预注册的 32/32 bank 划分保留完整系统。这个更严格子集改变了目录大小，所以本轮绝对 R@K 不应直接冒充历史 450-event v9.3 数字；C-fixed 与 C-retuned 的成对比较是公平的。",
        "",
        markdown_table(
            inventory,
            ["deployment", "model_seed", "split", "n_events", "n_sis_systems", "n_pm_systems", "n_unlensed_events", "n_actual_templates", "template_ood_fraction"],
            {"template_ood_fraction": "{:.3f}"},
        ),
        "",
        "## 4. 融合权重",
        "",
        "下表顺序均为 waveform/time/sky。C-retuned 的测试集和真实目录在这些权重冻结后才打开。",
        "",
        markdown_table(pd.DataFrame(weight_rows), ["deployment", "seed", "C_fixed", "C_retuned"]),
        "",
        "## 5. O3/O4a held-out 结果",
        "",
        "均值和 SD 来自三个冻结 model/design seeds；完整 95% Student-t CI 和每 seed 的 system-bootstrap R@1/R@10 CI 在 CSV 中。",
        "",
        markdown_table(
            metric_view,
            list(metric_view.columns),
            {
                "overall_r_at_1_mean": "{:.4f}",
                "overall_r_at_10_mean": "{:.4f}",
                "average_precision_mean": "{:.5f}",
                "false_at_recall_0p5_mean": "{:.1f}",
                "false_at_recall_0p9_mean": "{:.1f}",
                "pair_ood_fraction_mean": "{:.3f}",
            },
        ),
        "",
        f"预注册 gate 结论：**{gate['decision']}**。这不会自动更新论文方法。",
        "",
        "## 6. 256/512/1024 分辨率审计",
        "",
        "正式排名只使用 Nside=512。下表对子集中的真伴随、C-fixed 高分假对和真实候选头部比较 512 与 1024；任何差异都只报告，不用于事后改权重。",
        "",
        markdown_table(
            convergence,
            ["deployment", "seed", "audit_population", "n_pairs", "q99_abs_delta_512_1024", "max_abs_delta_512_1024", "sign_flip_512_1024_count", "spearman_512_1024"],
            {"q99_abs_delta_512_1024": "{:.4g}", "max_abs_delta_512_1024": "{:.4g}", "spearman_512_1024": "{:.5f}"},
        ),
        "",
        "## 7. 真实 GWTC 后验审计",
        "",
        "真实排名在配置和 held-out test 完成后一次性生成。PE 仅用于排名后审计。项目冻结输入中没有机器可读的 PO/ML 或 PO/Phazap 数值 FPP，也没有完整 Tier/Fast-GOLUM/Hanabi 阶段表，因此这些列保留为 NA 并明确标注，未伪造数值。",
        "",
        markdown_table(
            pe_budget,
            ["deployment", "method", "budget", "chirp_mass_BC_ge_0p5", "median_chirp_mass_BC", "Dmax_le_3", "official_frontend_overlap", "public_hanabi_table_overlap"],
            {"median_chirp_mass_BC": "{:.3f}"},
        ),
        "",
        "### Top-10",
        "",
    ]
    for deployment in DEPLOYMENTS:
        for method in ("C_fixed", "C_retuned"):
            part = top.loc[(top["deployment"] == deployment) & (top["method"] == method)].sort_values("consensus_rank").head(10)
            lines.extend(
                [
                    f"#### {deployment.upper()} {method}",
                    "",
                    markdown_table(
                        part,
                        [
                            "consensus_rank",
                            "event_i",
                            "event_j",
                            "final_score_mean",
                            "waveform_contribution_mean",
                            "time_contribution_mean",
                            "sky_contribution_mean",
                            "sky_bc_mean",
                            "max_standardized_posterior_distance",
                            "provenance_class",
                        ],
                        {
                            "final_score_mean": "{:.3f}",
                            "waveform_contribution_mean": "{:.3f}",
                            "time_contribution_mean": "{:.3f}",
                            "sky_contribution_mean": "{:.3f}",
                            "sky_bc_mean": "{:.3f}",
                            "max_standardized_posterior_distance": "{:.2f}",
                        },
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## 8. 科学边界",
            "",
            "- map-matched 注入复用公开 PE 后验形态，不等同于对每个注入从 strain 重新做相干参数估计。",
            "- 由于 O1 模板极少，部分事件会发生明确记录的 OOD fallback；结果表同时报告 OOD 比例。",
            "- 独立父噪声 bank 约束缩小了评估目录，因此本轮重点是同一 locked data 上 C-fixed/C-retuned 的成对差值。",
            "- 真实 Top-50 只是 catalog coincidence shortlist。即使 PE 边际一致，也不能称为 lensing detection。",
            "- C-retuned 即使通过本轮 gate，也仍保持 HOLD，必须由作者审核后另立确认性 locked experiment，才能考虑论文采纳。",
            "",
            "## 9. 交付",
            "",
            f"结果根目录：`{root}`",
            f"压缩包：`{package_path if package_path else '生成 manifest 后写入'}`",
            "",
            f"**{FINAL_STATUS}**",
        ]
    )
    (root / "reports/V93天空图错误与C方案独立确认实验报告_20260831_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def input_paths_for_manifest(root: Path) -> list[Path]:
    paths: list[Path] = [SCRIPT_PATH, MORPH_SCRIPT, V94_SCRIPT]
    for deployment in DEPLOYMENTS:
        metadata = pd.read_csv(root / f"contracts/{deployment}_public_pe_template_metadata.csv")
        paths.extend(Path(path) for path in metadata["source_path"].astype(str).unique())
        paths.append(source_run(deployment) / "data/event_manifest.csv")
        for seed in MODEL_SEEDS:
            seed_dir = input_seed_dir(deployment, seed)
            paths.extend(
                [
                    seed_dir / "data/real_noise_injections/compact_injection_metadata.parquet",
                    seed_dir / "results/mixed_val_synthetic_events_v7.parquet",
                    seed_dir / "results/mixed_test_synthetic_events_v7.parquet",
                    seed_dir / "results/fusion_validation_pairs_v7.parquet",
                    seed_dir / "results/fusion_heldout_test_pairs_v7.parquet",
                    corrected_seed_dir(deployment, seed) / "real_pair_features_unified_sky_v81.parquet",
                ]
            )
    paths.extend(critical_historical_paths())
    paths.append(
        PROJECT
        / "results/sky_background_fast_followup_v104_20260824_20260824T080136Z/tables/frozen_lvk_historical_candidate_comparison.csv"
    )
    return [path.resolve() for path in paths if path.is_file()]


def finalize(root: Path, package_path: Path) -> None:
    make_figures(root)
    reproduce = root / "scripts/reproduce.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "cd /root/autodl-tmp/gw-catalog\n"
        f"OUT=${{OUT:-results/{VERSION}_reproduction_$(date -u +%Y%m%dT%H%M%SZ)}}\n"
        f".venv_sky_v8/bin/python scripts/experiments/{SCRIPT_PATH.name} --output-root \"$OUT\" --phase all\n",
        encoding="utf-8",
    )
    reproduce.chmod(0o755)

    historical_after = snapshot_hashes(critical_historical_paths())
    write_csv(root / "contracts/historical_authority_hashes_after.csv", historical_after)
    before = pd.read_csv(root / "contracts/historical_authority_hashes_before.csv")
    comparison = before.merge(historical_after, on="path", suffixes=("_before", "_after"), how="outer")
    comparison["unchanged"] = comparison["sha256_before"] == comparison["sha256_after"]
    write_csv(root / "contracts/historical_authority_hash_comparison.csv", comparison)
    if not comparison["unchanged"].fillna(False).all():
        raise RuntimeError("A protected historical authority file changed")

    input_manifest = snapshot_hashes(input_paths_for_manifest(root))
    write_csv(root / "manifest/input_sha256_manifest.csv", input_manifest)
    write_report(root, package_path)
    write_json(
        root / "FINAL_STATUS.json",
        {
            "status": FINAL_STATUS,
            "completed_utc": utc_stamp(),
            "historical_authority_unchanged": True,
            "paper_modified": False,
            "v9_3_or_v9_4_overwritten": False,
            "Hanabi_run": False,
            "full_BBH_PE_run": False,
        },
    )
    write_json(
        root / "manifest/package_manifest.json",
        {
            "package_path": str(package_path),
            "package_sha256": "recorded in the external .sha256 file after archive creation",
            "status": FINAL_STATUS,
            "archive_contains_dense_HEALPix_maps": False,
        },
    )

    output_files = [
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    sum_lines = [
        f"{sha256_file(path)}  {path.relative_to(root)}" for path in sorted(output_files)
    ]
    (root / "manifest/SHA256SUMS.txt").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

    package_path.parent.mkdir(parents=True, exist_ok=True)
    if package_path.exists():
        raise FileExistsError(package_path)
    with tarfile.open(package_path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    package_sha = sha256_file(package_path)
    package_path.with_suffix(package_path.suffix + ".sha256").write_text(
        f"{package_sha}  {package_path.name}\n", encoding="utf-8"
    )
    write_json(
        package_path.with_suffix(package_path.suffix + ".manifest.json"),
        {
            "package_path": str(package_path),
            "package_sha256": package_sha,
            "package_size_bytes": package_path.stat().st_size,
            "status": FINAL_STATUS,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "phase0", "validation", "test", "real", "resolution", "finalize"),
        default="all",
    )
    parser.add_argument("--package-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    package = (
        args.package_path.resolve()
        if args.package_path
        else PROJECT / "packages" / f"{root.name}_final.tar.gz"
    )
    if args.phase in {"all", "phase0"}:
        phase0(root)
    if args.phase in {"all", "validation"}:
        phase1_validation(root)
    if args.phase in {"all", "test"}:
        phase2_locked_test(root)
    if args.phase in {"all", "real"}:
        phase3_real_catalog(root)
    if args.phase in {"all", "resolution"}:
        phase4_resolution_audit(root)
    if args.phase in {"all", "finalize"}:
        finalize(root, package)
    print(
        json.dumps(
            {
                "output_root": str(root),
                "package": str(package) if package.exists() else None,
                "status": FINAL_STATUS,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
