#!/usr/bin/env python3
"""Run the frozen-waveform/time sky-morphology exploratory comparison.

This pipeline is deliberately independent of historical v9.3 and corrected
v9.4 outputs.  It regenerates probability maps from the original public PE
HDF5 products with strict NESTED/RING handling, computes morphology features
without persisting dense maps, calibrates the new sky channel on a
source-disjoint partition of synthetic validation systems, and evaluates a
single locked held-out test.

The three paired arms are:

* C: frozen v9.3 waveform/time/sky weights on corrected common Nside=512 maps;
* A: one-stage waveform + time + morphology-calibrated sky;
* B: frozen waveform+time top-K front end, then sky reranking inside the pool.

Historical v9.3 is retained as provenance only because its PE HDF5 ordering
was decoded incorrectly.  Nothing in this script writes into any historical
result directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEPLOYMENTS = ("gwtc3", "gwtc4")
SEEDS = (202607241, 202607242, 202607243)
ANALYSIS_NSIDE = 512
COARSE_AUDIT_NSIDE = 256
BASE_SEED = 2026082701
ROTATION_WORKERS = 8
HISTOGRAM_BINS = 48
BANDWIDTH_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
CAP_GRID = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (2.0, 3.0))
ETA_GRID = (0.0, 0.25, 0.5, 1.0)
K_GRID = (20, 50, 100)
PRIMARY_K = 50
MIN_POSITIVE_CELL = 8
MIN_NEGATIVE_CELL = 100
MAX_NEGATIVE_FIT_PER_CELL = 20_000
CALIBRATION_FRACTION = 0.60
SNR_BOUNDARY = 12.0
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
RAW_BF_NUMERICAL_FLOOR = 1e-30

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


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V94_SCRIPT = PROJECT / "scripts/experiments/run_gwtc_sky_ordering_corrected_v94.py"
if not V94_SCRIPT.is_file():
    raise FileNotFoundError(V94_SCRIPT)
v94 = load_module(V94_SCRIPT, "ordering_corrected_v94_for_sky_morphology")
v7 = v94.v93.v7


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


def add_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for left, right in (("event_i", "event_j"), ("event_1", "event_2"), ("event_a", "event_b")):
        if left in out and right in out:
            out["pair_key"] = [canonical_pair(a, b) for a, b in zip(out[left], out[right])]
            return out
    raise KeyError("No event-pair columns found")


def source_run(deployment: str) -> Path:
    if deployment == "gwtc3":
        return PROJECT / "runs/real_gwtc_lensing_search_20260625"
    return PROJECT / "runs/real_gwtc34_lensing_search_20260629_full_o4"


def primary_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["include_in_primary_search"].fillna(False).astype(bool)
    mask &= frame["sky_map_available"].fillna(False).astype(bool)
    return frame.loc[mask].sort_values("gps_time").reset_index(drop=True)


def detector_count(value: object) -> int:
    return len([part for part in str(value).split(",") if part.strip()])


def infer_run(gps: float, deployment: str) -> str:
    if deployment == "gwtc4":
        return "O4a"
    for name, start, end in (
        ("O1", 1126051217.0, 1137254417.0),
        ("O2", 1164556817.0, 1187733618.0),
        ("O3a", 1238166018.0, 1253977218.0),
        ("O3b", 1256655618.0, 1269363618.0),
    ):
        if start <= gps <= end:
            return name
    return "outside_run"


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    cross = np.asarray(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(unit, unit)
        + math.sin(angle) * cross
    )


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross_vector = np.cross(a, b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    sine = float(np.linalg.norm(cross_vector))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = np.asarray([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.asarray([0.0, 1.0, 0.0])
        axis -= np.dot(axis, a) * a
        return axis_angle_matrix(axis, math.pi)
    cross_matrix = np.asarray(
        [
            [0.0, -cross_vector[2], cross_vector[1]],
            [cross_vector[2], 0.0, -cross_vector[0]],
            [-cross_vector[1], cross_vector[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + cross_matrix + cross_matrix @ cross_matrix * ((1.0 - cosine) / (sine * sine))


def rotate_map(
    probability: np.ndarray,
    true_ra: float,
    true_dec: float,
    seed: int,
    output_vectors: np.ndarray,
    nside: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    source_map = np.asarray(probability, dtype=np.float64)
    source_map = np.clip(source_map, 0.0, None)
    source_map /= source_map.sum(dtype=np.float64)
    anchor = int(rng.choice(len(source_map), p=source_map))
    source = np.asarray(hp.pix2vec(nside, anchor, nest=False), dtype=np.float64)
    target = np.asarray(
        [math.cos(true_dec) * math.cos(true_ra), math.cos(true_dec) * math.sin(true_ra), math.sin(true_dec)],
        dtype=np.float64,
    )
    rotation = axis_angle_matrix(target, float(rng.uniform(0.0, 2.0 * math.pi))) @ rotation_between(source, target)
    input_vectors = rotation.T @ output_vectors
    theta = np.arccos(np.clip(input_vectors[2], -1.0, 1.0))
    phi = np.mod(np.arctan2(input_vectors[1], input_vectors[0]), 2.0 * math.pi)
    rotated = hp.get_interp_val(source_map, theta, phi, nest=False)
    rotated = np.clip(rotated, 0.0, None)
    rotated /= rotated.sum(dtype=np.float64)
    return rotated.astype(np.float32), anchor


def component_count_hpd90(probability: np.ndarray, nside: int) -> int:
    """Count connected components in a coarse 90% HPD mask (diagnostic only)."""
    values = np.asarray(probability, dtype=np.float64)
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    count = min(int(np.searchsorted(cumulative, 0.9, side="left")) + 1, len(values))
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:count]] = True
    remaining = set(np.flatnonzero(mask).tolist())
    components = 0
    while remaining:
        components += 1
        start = remaining.pop()
        queue: deque[int] = deque([start])
        while queue:
            pixel = queue.popleft()
            for neighbour in hp.get_all_neighbours(nside, pixel, nest=False):
                neighbour = int(neighbour)
                if neighbour >= 0 and neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
    return components


@dataclass
class MapBankFeatures:
    raw_log_bf: np.ndarray
    bc: np.ndarray
    j50: np.ndarray
    j90: np.ndarray
    coverage50: np.ndarray
    coverage90: np.ndarray
    event: pd.DataFrame
    resources: dict[str, Any]


def _hpd_mask_gpu(tensor: torch.Tensor, target: float, iterations: int = 27) -> torch.Tensor:
    low = torch.zeros((tensor.shape[0], 1), dtype=tensor.dtype, device=tensor.device)
    high = tensor.max(dim=1, keepdim=True).values
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        mass = (tensor * (tensor >= middle)).sum(dim=1, keepdim=True)
        low = torch.where(mass >= target, middle, low)
        high = torch.where(mass < target, middle, high)
    return tensor >= low


def compute_map_bank_features(maps: np.ndarray, *, deployment: str, seed: int, split: str) -> MapBankFeatures:
    start = time.perf_counter()
    normalized = np.asarray(maps, dtype=np.float32)
    sums = normalized.sum(axis=1, dtype=np.float64)
    normalized /= sums[:, None].astype(np.float32)
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise RuntimeError("Invalid probability map bank")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    pixel_area_deg2 = 4.0 * math.pi * (180.0 / math.pi) ** 2 / normalized.shape[1]
    with torch.inference_mode():
        tensor = torch.from_numpy(normalized).to(device=device, dtype=torch.float32)
        raw_overlap = tensor @ tensor.T
        raw_log_bf = torch.log(
            torch.clamp(raw_overlap * normalized.shape[1], min=RAW_BF_NUMERICAL_FLOOR)
        ).cpu().numpy()
        root = torch.sqrt(torch.clamp(tensor, min=0.0))
        bc = torch.clamp(root @ root.T, min=0.0, max=1.0).cpu().numpy()
        del root

        areas: dict[int, np.ndarray] = {}
        jaccard: dict[int, np.ndarray] = {}
        coverage: dict[int, np.ndarray] = {}
        for level, target in ((50, 0.50), (90, 0.90)):
            mask_bool = _hpd_mask_gpu(tensor, target)
            mask = mask_bool.to(dtype=torch.float32)
            counts = mask.sum(dim=1)
            intersection = mask @ mask.T
            union = counts[:, None] + counts[None, :] - intersection
            jaccard[level] = torch.where(union > 0, intersection / union, torch.zeros_like(union)).cpu().numpy()
            coverage[level] = (tensor @ mask.T).cpu().numpy()
            areas[level] = (counts * pixel_area_deg2).cpu().numpy()
            del mask, intersection, union, mask_bool

        entropy = (-(tensor * torch.log(torch.clamp(tensor, min=1e-30))).sum(dim=1)).cpu().numpy()
        kl = math.log(normalized.shape[1]) - entropy
        del tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    event = pd.DataFrame(
        {
            "idx": np.arange(len(normalized), dtype=np.int32),
            "area50_deg2": areas[50],
            "area90_deg2": areas[90],
            "entropy_nats": entropy,
            "kl_from_isotropic_nats": kl,
            "map_sum": normalized.sum(axis=1, dtype=np.float64),
        }
    )
    for matrix in (raw_log_bf, bc, jaccard[50], jaccard[90]):
        np.fill_diagonal(matrix, np.nan)
    resources = {
        "deployment": deployment,
        "seed": int(seed),
        "split": split,
        "n_events": int(len(normalized)),
        "nside": ANALYSIS_NSIDE,
        "npix": int(normalized.shape[1]),
        "device": str(device),
        "dense_input_gib": float(normalized.nbytes / 2**30),
        "elapsed_seconds": float(time.perf_counter() - start),
        "dense_maps_persisted": False,
        "matrix_accumulation": "float32 GPU with deterministic TF32 disabled; selected pairs audited in float64",
        "raw_bayes_factor_numerical_floor": RAW_BF_NUMERICAL_FLOOR,
    }
    return MapBankFeatures(
        raw_log_bf=raw_log_bf.astype(np.float32),
        bc=bc.astype(np.float32),
        j50=jaccard[50].astype(np.float32),
        j90=jaccard[90].astype(np.float32),
        coverage50=coverage[50].astype(np.float32),
        coverage90=coverage[90].astype(np.float32),
        event=event,
        resources=resources,
    )


def float64_feature_audit(
    maps: np.ndarray,
    features: MapBankFeatures,
    pairs: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    labels = pairs["is_true_pair"].to_numpy(dtype=bool) if "is_true_pair" in pairs else np.zeros(len(pairs), dtype=bool)
    true_rows = np.flatnonzero(labels)
    null_rows = np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    # Audit every companion and a fixed random null subset.  Chunked float64
    # evaluation avoids thousands of repeated 24 MiB temporary allocations.
    selected_null = rng.choice(null_rows, size=min(100, len(null_rows)), replace=False)
    selected = np.concatenate([true_rows, selected_null])
    ii = pairs["idx_i"].to_numpy(np.int32)
    jj = pairs["idx_j"].to_numpy(np.int32)
    raw_errors: list[np.ndarray] = []
    bc_errors: list[np.ndarray] = []
    for start in range(0, len(selected), 8):
        rows = selected[start : start + 8]
        left = ii[rows]
        right = jj[rows]
        p = maps[left].astype(np.float64)
        q = maps[right].astype(np.float64)
        overlap = np.sum(p * q, axis=1, dtype=np.float64)
        raw = np.log(np.maximum(maps.shape[1] * overlap, RAW_BF_NUMERICAL_FLOOR))
        bc = np.sum(np.sqrt(p * q), axis=1, dtype=np.float64)
        raw_errors.append(np.abs(raw - features.raw_log_bf[left, right].astype(np.float64)))
        bc_errors.append(np.abs(bc - features.bc[left, right].astype(np.float64)))
    raw_array = np.concatenate(raw_errors) if raw_errors else np.asarray([], dtype=np.float64)
    bc_array = np.concatenate(bc_errors) if bc_errors else np.asarray([], dtype=np.float64)
    return {
        "n_pairs": int(len(selected)),
        "raw_log_bf_max_abs_error": float(raw_array.max(initial=0.0)),
        "raw_log_bf_q99_abs_error": float(np.quantile(raw_array, 0.99)) if len(raw_array) else 0.0,
        "bc_max_abs_error": float(bc_array.max(initial=0.0)),
        "bc_q99_abs_error": float(np.quantile(bc_array, 0.99)) if len(bc_array) else 0.0,
        "tolerance": 2e-3,
        "passed": bool(raw_array.max(initial=0.0) <= 2e-3 and bc_array.max(initial=0.0) <= 2e-3),
    }


def attach_pair_morphology(base: pd.DataFrame, features: MapBankFeatures, event_meta: pd.DataFrame) -> pd.DataFrame:
    out = base.drop(columns=[column for column in base.columns if column.startswith("sky_")], errors="ignore").copy()
    ii = out["idx_i"].to_numpy(np.int32)
    jj = out["idx_j"].to_numpy(np.int32)
    event = event_meta.set_index("idx")
    out["sky_raw_log_bf"] = features.raw_log_bf[ii, jj]
    out["sky_bc"] = features.bc[ii, jj]
    out["sky_j50"] = features.j50[ii, jj]
    out["sky_j90"] = features.j90[ii, jj]
    out["sky_coverage50_i_in_j"] = features.coverage50[ii, jj]
    out["sky_coverage50_j_in_i"] = features.coverage50[jj, ii]
    out["sky_coverage90_i_in_j"] = features.coverage90[ii, jj]
    out["sky_coverage90_j_in_i"] = features.coverage90[jj, ii]
    for column in (
        "area50_deg2",
        "area90_deg2",
        "entropy_nats",
        "kl_from_isotropic_nats",
        "detector_count",
        "snr",
        "hpd90_component_count",
    ):
        values = event[column].to_numpy()
        out[f"sky_{column}_i"] = values[ii]
        out[f"sky_{column}_j"] = values[jj]
    low = np.minimum(out["sky_area90_deg2_i"], out["sky_area90_deg2_j"])
    high = np.maximum(out["sky_area90_deg2_i"], out["sky_area90_deg2_j"])
    out["sky_area90_ratio"] = high / np.maximum(low, 1e-12)
    out["sky_min_snr"] = np.minimum(out["sky_snr_i"], out["sky_snr_j"])
    out["sky_min_detector_count"] = np.minimum(out["sky_detector_count_i"], out["sky_detector_count_j"])
    out["sky_max_area90_deg2"] = high
    out["sky_score"] = out["sky_raw_log_bf"]
    return out


def load_template_library(deployment: str) -> tuple[pd.DataFrame, list[np.ndarray], list[Path]]:
    manifest_path = source_run(deployment) / "data/event_manifest.csv"
    manifest = primary_manifest(pd.read_csv(manifest_path))
    coarse_path = (
        PROJECT
        / "results/gwtc_sky_ordering_corrected_v94_20260830/corrected_sky_inputs"
        / deployment
        / "shared/real_pe_sky_templates_nside64_ordering_corrected_v94.npy"
    )
    coarse_maps = np.load(coarse_path, mmap_mode="r")
    maps: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    used: list[Path] = [manifest_path, coarse_path]
    for index, row in manifest.iterrows():
        probability, metadata = v94.corrected_read_probability_map(row, target_nside=ANALYSIS_NSIDE)
        probability = np.asarray(probability, dtype=np.float32)
        probability /= probability.sum(dtype=np.float64)
        maps.append(probability)
        map_path = Path(str(row["sky_map_path"]))
        if not map_path.is_absolute():
            map_path = PROJECT / map_path
        used.append(map_path)
        rows.append(
            {
                "template_index": int(index),
                "event_name": str(row["event_name"]),
                "run": str(row["run"]),
                "network_snr": float(row.get("network_snr", np.nan)),
                "detectors_available": str(row.get("detectors_available", "")),
                "detector_count": detector_count(row.get("detectors_available", "")),
                "native_nside": int(metadata["source_nside"]),
                "source_ordering": str(metadata["source_ordering"]),
                "output_ordering": str(metadata["output_ordering"]),
                "coordinate_frame": str(metadata["coordinate_frame"]),
                "ordering_conversion_applied": bool(metadata["ordering_conversion_applied"]),
                "hpd90_component_count": component_count_hpd90(np.asarray(coarse_maps[index]), 64),
            }
        )
    return pd.DataFrame(rows), maps, used


def choose_templates(events: pd.DataFrame, templates: pd.DataFrame, deployment: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    used_by_lens_system: dict[str, set[int]] = {}
    for event in events.itertuples(index=False):
        run = infer_run(float(event.gps_obs), deployment)
        if deployment == "gwtc4":
            same_run = templates.index.to_numpy(dtype=np.int32)
        else:
            same_run = templates.index[templates["run"].astype(str) == run].to_numpy(dtype=np.int32)
        pool = same_run if same_run.size else templates.index.to_numpy(dtype=np.int32)
        template_snr = templates.loc[pool, "network_snr"].to_numpy(float)
        distances = np.abs(np.log(np.maximum(template_snr, 1e-6)) - math.log(max(float(event.snr), 1e-6)))
        nearest = pool[np.argsort(distances)[: min(4, len(pool))]]
        is_unlensed = str(event.family).lower() == "unlensed" or str(event.tag).upper() == "U"
        system_key = None if is_unlensed else f"{event.family}:{event.source_index}"
        used = set() if system_key is None else used_by_lens_system.setdefault(system_key, set())
        eligible = np.asarray([item for item in nearest if int(item) not in used], dtype=np.int32)
        if not len(eligible):
            eligible = np.asarray([item for item in pool if int(item) not in used], dtype=np.int32)
        if not len(eligible):
            raise RuntimeError(f"No distinct sky template remains for lensed system {system_key}")
        chosen = int(rng.choice(eligible))
        if system_key is not None:
            used.add(chosen)
        template = templates.loc[chosen]
        rows.append(
            {
                "idx": int(event.idx),
                "run": run,
                "snr": float(event.snr),
                "ra_true": float(event.ra_true),
                "dec_true": float(event.dec_true),
                "template_index": chosen,
                "template_event": str(template["event_name"]),
                "template_run": str(template["run"]),
                "template_run_matched": bool(
                    deployment == "gwtc4" or str(template["run"]) == run
                ),
                "template_network_snr": float(template["network_snr"]),
                "template_detectors": str(template["detectors_available"]),
                "detector_count": int(template["detector_count"]),
                "hpd90_component_count": int(template["hpd90_component_count"]),
                "lens_system_template_reuse_forbidden": bool(system_key is not None),
            }
        )
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def generate_synthetic_maps(
    events: pd.DataFrame,
    assignment: pd.DataFrame,
    template_maps: list[np.ndarray],
    deployment: str,
    seed: int,
    split: str,
    output_vectors: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    maps = np.empty((len(events), hp.nside2npix(ANALYSIS_NSIDE)), dtype=np.float32)
    anchors = np.full(len(events), -1, dtype=np.int64)
    start = time.perf_counter()

    def task(row: Any) -> tuple[int, np.ndarray, int]:
        event_index = int(row.idx)
        rotation_seed = stable_seed("morphology", deployment, seed, split, event_index, BASE_SEED)
        rotated, anchor = rotate_map(
            template_maps[int(row.template_index)],
            float(row.ra_true),
            float(row.dec_true),
            rotation_seed,
            output_vectors,
            ANALYSIS_NSIDE,
        )
        return event_index, rotated, anchor

    with ThreadPoolExecutor(max_workers=ROTATION_WORKERS) as executor:
        futures = [executor.submit(task, row) for row in assignment.itertuples(index=False)]
        for future in as_completed(futures):
            index, probability, anchor = future.result()
            maps[index] = probability
            anchors[index] = anchor
    assignment = assignment.copy()
    assignment["anchor_pixel"] = anchors[assignment["idx"].to_numpy(np.int32)]
    assignment["map_sum"] = maps.sum(axis=1, dtype=np.float64)
    assignment["finite_fraction"] = np.mean(np.isfinite(maps), axis=1)
    return maps, assignment, time.perf_counter() - start


def input_seed_dir(deployment: str, seed: int) -> Path:
    return PROJECT / "results/real_noise_injection_v7_peak2s_formal_20260722" / deployment / f"seed_{seed}"


def corrected_seed_dir(deployment: str, seed: int) -> Path:
    return PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830" / deployment / f"seed_{seed}"


def phase_marker(root: Path, phase: str, payload: dict[str, Any]) -> None:
    write_json(root / "contracts" / f"{phase}_STATUS.json", {"phase": phase, "timestamp_utc": utc_stamp(), **payload})


def phase0(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("contracts", "results", "figures", "reports", "scripts", "logs", "manifest"):
        (root / directory).mkdir(exist_ok=True)
    shutil.copy2(SCRIPT_PATH, root / "scripts" / SCRIPT_PATH.name)
    inventory_rows: list[dict[str, Any]] = []
    pe_source_rows: list[dict[str, Any]] = []
    input_paths: list[Path] = [SCRIPT_PATH, V94_SCRIPT]
    for deployment in DEPLOYMENTS:
        manifest_path = source_run(deployment) / "data/event_manifest.csv"
        manifest = primary_manifest(pd.read_csv(manifest_path))
        input_paths.append(manifest_path)
        for _, event in manifest.iterrows():
            map_path = resolve_pe_path(event["sky_map_path"], deployment)
            input_paths.append(map_path)
            pe_source_rows.append(
                {
                    "deployment": deployment,
                    "event_name": str(event["event_name"]),
                    "pe_path": str(map_path),
                    "pe_group_manifest": str(
                        event.get("sky_map_internal_group", event.get("sky_map_group", ""))
                    ),
                    "native_product": "public PE HDF5",
                }
            )
        inventory_rows.append(
            {
                "deployment": deployment,
                "kind": "real_primary_catalog",
                "seed": np.nan,
                "split": "real",
                "n_events": len(manifest),
                "n_pairs": len(manifest) * (len(manifest) - 1) // 2,
                "n_maps": len(manifest),
                "map_source": "public PE HDF5",
                "analysis_nside": ANALYSIS_NSIDE,
            }
        )
        for seed in SEEDS:
            seed_dir = input_seed_dir(deployment, seed) / "results"
            for split, short, pair_name in (
                ("validation", "val", "fusion_validation_pairs_v7.parquet"),
                ("test", "test", "fusion_heldout_test_pairs_v7.parquet"),
            ):
                events_path = seed_dir / f"mixed_{short}_synthetic_events_v7.parquet"
                pairs_path = seed_dir / pair_name
                events = pd.read_parquet(events_path, columns=["idx", "family", "tag", "pair_id", "source_index"])
                pairs = pd.read_parquet(pairs_path, columns=["is_true_pair"])
                input_paths.extend([events_path, pairs_path])
                inventory_rows.append(
                    {
                        "deployment": deployment,
                        "kind": "synthetic_map_matched_catalog",
                        "seed": seed,
                        "split": split,
                        "n_events": len(events),
                        "n_pairs": len(pairs),
                        "n_true_pairs": int(pairs["is_true_pair"].sum()),
                        "n_false_pairs": int((pairs["is_true_pair"] == 0).sum()),
                        "n_maps": len(events),
                        "map_source": "corrected public PE template rotated to simulated true direction",
                        "analysis_nside": ANALYSIS_NSIDE,
                    }
                )
    write_csv(root / "contracts/data_inventory.csv", pd.DataFrame(inventory_rows))
    write_csv(root / "contracts/public_pe_source_manifest.csv", pd.DataFrame(pe_source_rows))
    contract = {
        "experiment": "sky_morphology_one_vs_two_stage_exploratory",
        "status": FINAL_STATUS,
        "created_utc": utc_stamp(),
        "historical_v9_3": "read-only provenance; not a valid paired map baseline after the NESTED/RING audit",
        "paired_baseline_C": "frozen v9.3 weights on ordering-corrected common Nside=512 maps",
        "analysis_nside": ANALYSIS_NSIDE,
        "coarse_audit_nside": COARSE_AUDIT_NSIDE,
        "dense_maps_persisted": False,
        "deployments": list(DEPLOYMENTS),
        "seeds": list(SEEDS),
        "frozen_v93_weights": FROZEN_V93_WEIGHTS,
        "frozen": [
            "waveform encoder/checkpoint",
            "waveform embeddings and waveform score",
            "one-dimensional time score",
            "strict H1-L1 BBH scope",
            "validation/test source split",
        ],
        "morphology_features": ["raw_log_bf", "Bhattacharyya_coefficient"],
        "diagnostics": ["J50", "J90", "bidirectional_HPD_coverage", "A50", "A90", "entropy", "KL", "HPD90_components"],
        "calibration_partition": {
            "within_synthetic_validation": "deterministic source-disjoint 60/40 fit/tune",
            "fraction_fit": CALIBRATION_FRACTION,
            "heldout_test_used_for_selection": False,
        },
        "template_independence": "two images of the same simulated lens system must use distinct public PE templates",
        "template_run_matching": "use same observing-run templates whenever any exist; O4a_or_gap is normalized to O4a",
        "quality_strata": ["detector_count", f"min_network_snr_at_{SNR_BOUNDARY}", "max_A90_at_fit_median"],
        "density_estimator": {
            "type": "two-dimensional weighted histogram density with Gaussian smoothing",
            "bins": HISTOGRAM_BINS,
            "bandwidth_grid_bins": list(BANDWIDTH_GRID),
            "support": "fit-sample rectangular support; outside support receives neutral score",
            "minimum_positive_pairs_per_cell": MIN_POSITIVE_CELL,
            "minimum_negative_pairs_per_cell": MIN_NEGATIVE_CELL,
        },
        "raw_bayes_factor_numerical_floor": RAW_BF_NUMERICAL_FLOOR,
        "cap_grid": [list(item) for item in CAP_GRID],
        "one_stage_weight_grid": "non-negative simplex in 0.05 increments; primary requires all three weights > 0",
        "two_stage": {"K_grid": list(K_GRID), "primary_K": PRIMARY_K, "eta_grid": list(ETA_GRID)},
        "selection_priority": ["pair_AUPRC", "precision_at_50pct_recall", "macro_R10", "minimum_family_R10", "macro_R1", "lower_complexity"],
        "real_PE_and_LVK_used_for_selection": False,
        "claims": "candidate shortlist for follow-up only; no lensing detection claim",
    }
    write_json(root / "contracts/data_contract.json", contract)
    manifest_rows = []
    for path in sorted(set(input_paths)):
        if path.is_file():
            manifest_rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(root / "contracts/input_sha256_manifest.csv", pd.DataFrame(manifest_rows))
    protected_candidates = [
        PROJECT / "results/gwtc_sky_resolution_v93_20260730/analysis_contract_v93.json",
        PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830/analysis_contract_ordering_corrected_v94.json",
        PROJECT / "packages/gwtc_sky_resolution_v93_20260730.tar.gz",
        PROJECT / "packages/gwtc_sky_ordering_corrected_v94_20260830_deliverables.tar.gz",
    ]
    protected = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256_before": sha256_file(path)}
        for path in protected_candidates
        if path.is_file()
    ]
    write_json(root / "contracts/historical_protection_hashes.json", protected)
    phase_marker(root, "PHASE0", {"passed": True, "reason": "Complete public PE maps and all frozen pair tables are available."})


def phase1(root: Path) -> None:
    output_vectors = np.asarray(hp.pix2vec(ANALYSIS_NSIDE, np.arange(hp.nside2npix(ANALYSIS_NSIDE)), nest=False))
    resource_rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        dep_root = root / "results" / deployment
        dep_root.mkdir(parents=True, exist_ok=True)
        print(f"[phase1] {deployment}: load corrected public PE templates", flush=True)
        template_frame, template_maps, _ = load_template_library(deployment)
        write_csv(dep_root / "public_pe_template_library_nside512.csv", template_frame)

        # Real maps are common to all model seeds and are processed once.
        manifest = primary_manifest(pd.read_csv(source_run(deployment) / "data/event_manifest.csv"))
        real_maps = np.stack(template_maps).astype(np.float32)
        real_base = pd.read_parquet(corrected_seed_dir(deployment, SEEDS[0]) / "real_pair_features_unified_sky_v81.parquet")
        print(f"[phase1] {deployment}: real morphology", flush=True)
        real_features = compute_map_bank_features(real_maps, deployment=deployment, seed=0, split="real")
        real_event = real_features.event.copy()
        real_event["event_name"] = manifest["event_name"].astype(str).to_numpy()
        real_event["snr"] = pd.to_numeric(manifest["network_snr"], errors="coerce").to_numpy()
        real_event["detector_count"] = manifest["detectors_available"].map(detector_count).to_numpy()
        real_event["hpd90_component_count"] = template_frame["hpd90_component_count"].to_numpy()
        real_pairs = attach_pair_morphology(real_base, real_features, real_event)
        real_pairs.to_parquet(dep_root / "real_pair_sky_morphology_nside512.parquet", index=False)
        write_csv(dep_root / "real_event_sky_morphology_nside512.csv", real_event)
        resource_rows.append({**real_features.resources, "map_generation_seconds": 0.0})
        del real_features, real_maps, real_pairs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for seed in SEEDS:
            seed_root = dep_root / f"seed_{seed}"
            seed_root.mkdir(exist_ok=True)
            source_results = input_seed_dir(deployment, seed) / "results"
            for split, short, pair_name in (
                ("validation", "val", "fusion_validation_pairs_v7.parquet"),
                ("test", "test", "fusion_heldout_test_pairs_v7.parquet"),
            ):
                output_pairs = seed_root / f"synthetic_{split}_pair_morphology_nside512.parquet"
                if output_pairs.is_file():
                    print(f"[phase1] reuse {deployment} {seed} {split}", flush=True)
                    continue
                events = pd.read_parquet(source_results / f"mixed_{short}_synthetic_events_v7.parquet")
                base = pd.read_parquet(source_results / pair_name)
                assignment = choose_templates(events, template_frame, deployment, stable_seed(BASE_SEED, deployment, seed, split))
                print(f"[phase1] {deployment} {seed} {split}: generate 512 maps", flush=True)
                maps, assignment, generation_seconds = generate_synthetic_maps(
                    events, assignment, template_maps, deployment, seed, split, output_vectors
                )
                print(f"[phase1] {deployment} {seed} {split}: morphology matrices", flush=True)
                features = compute_map_bank_features(maps, deployment=deployment, seed=seed, split=split)
                event_meta = features.event.merge(
                    assignment[["idx", "snr", "detector_count", "hpd90_component_count", "template_index", "template_event", "run"]],
                    on="idx",
                    how="left",
                    validate="one_to_one",
                )
                pairs = attach_pair_morphology(base, features, event_meta)
                numerical = float64_feature_audit(maps, features, pairs, stable_seed(deployment, seed, split, "audit"))
                if not numerical["passed"]:
                    raise RuntimeError(f"Feature precision audit failed: {numerical}")
                pairs.to_parquet(output_pairs, index=False)
                write_csv(seed_root / f"synthetic_{split}_event_sky_morphology_nside512.csv", event_meta)
                write_csv(seed_root / f"synthetic_{split}_template_assignment.csv", assignment)
                write_json(seed_root / f"synthetic_{split}_float64_feature_audit.json", numerical)
                resource_rows.append({**features.resources, "map_generation_seconds": generation_seconds})
                del maps, features, pairs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del template_maps
        gc.collect()
    write_csv(root / "results/resource_usage_phase1.csv", pd.DataFrame(resource_rows))
    phase_marker(root, "PHASE1", {"passed": True, "dense_maps_persisted": False})


def validation_system_partition(events: pd.DataFrame, deployment: str, seed: int) -> pd.DataFrame:
    out = events[["idx", "family", "tag", "pair_id", "source_index"]].copy()
    is_unlensed = out["family"].astype(str).str.lower().eq("unlensed") | out["tag"].astype(str).str.upper().eq("U")
    out["system_id"] = np.where(
        is_unlensed,
        "unlensed:" + out["source_index"].astype(str),
        out["family"].astype(str) + ":" + out["source_index"].astype(str),
    )
    out["stratum"] = np.where(is_unlensed, "unlensed", out["family"].astype(str))
    group = out[["system_id", "stratum"]].drop_duplicates().copy()
    group["hash"] = group["system_id"].map(lambda value: stable_seed("fit-tune", deployment, seed, value))
    selected: dict[str, str] = {}
    for _, part in group.groupby("stratum", sort=True):
        ordered = part.sort_values(["hash", "system_id"], kind="stable")
        n_fit = int(math.floor(CALIBRATION_FRACTION * len(ordered)))
        for position, system_id in enumerate(ordered["system_id"]):
            selected[str(system_id)] = "fit" if position < n_fit else "tune"
    out["calibration_subset"] = out["system_id"].map(selected)
    counts = out.groupby("system_id")["calibration_subset"].nunique()
    if int(counts.max()) != 1:
        raise RuntimeError("A source system crossed the fit/tune partition")
    return out


def pair_partition(frame: pd.DataFrame, event_partition: pd.DataFrame) -> np.ndarray:
    subset = event_partition.sort_values("idx")["calibration_subset"].to_numpy(dtype=object)
    ii = frame["idx_i"].to_numpy(np.int32)
    jj = frame["idx_j"].to_numpy(np.int32)
    return np.where(subset[ii] == subset[jj], subset[ii], "cross")


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(array))
    return center, max(scale, 1e-6)


def morphology_coordinates(frame: pd.DataFrame, transform: dict[str, float]) -> np.ndarray:
    raw = frame["sky_raw_log_bf"].to_numpy(dtype=np.float64)
    bc = np.clip(frame["sky_bc"].to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit_bc = np.log(bc / (1.0 - bc))
    return np.column_stack(
        [
            (raw - float(transform["raw_center"])) / float(transform["raw_scale"]),
            (logit_bc - float(transform["bc_logit_center"])) / float(transform["bc_logit_scale"]),
        ]
    )


def quality_keys(frame: pd.DataFrame, area_boundary: float) -> np.ndarray:
    detector = np.where(frame["sky_min_detector_count"].to_numpy(float) >= 3, "network3plus", "network2")
    snr = np.where(frame["sky_min_snr"].to_numpy(float) >= SNR_BOUNDARY, "snr_high", "snr_low")
    area = np.where(frame["sky_max_area90_deg2"].to_numpy(float) <= area_boundary, "area_compact", "area_wide")
    return np.asarray([f"{a}|{b}|{c}" for a, b, c in zip(detector, snr, area)], dtype=object)


def _histogram_density(
    coordinates: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    histogram, _, _ = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=(edges_x, edges_y))
    smoothed = gaussian_filter(histogram.astype(np.float64), sigma=float(bandwidth), mode="nearest")
    smoothed = np.maximum(smoothed, 0.0)
    smoothed /= max(float(smoothed.sum()), 1e-300)
    floor = 1e-8 / smoothed.size
    smoothed = smoothed + floor
    smoothed /= smoothed.sum()
    return smoothed


def _lookup_density(coordinates: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray, density: np.ndarray) -> np.ndarray:
    x = np.searchsorted(edges_x, coordinates[:, 0], side="right") - 1
    y = np.searchsorted(edges_y, coordinates[:, 1], side="right") - 1
    valid = (x >= 0) & (x < density.shape[0]) & (y >= 0) & (y < density.shape[1])
    values = np.full(len(coordinates), np.nan, dtype=np.float64)
    values[valid] = density[x[valid], y[valid]]
    return values


def fit_morphology_calibrator(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    area_boundary: float,
    deployment: str,
    seed: int,
) -> dict[str, Any]:
    fit_bc = np.clip(fit["sky_bc"].to_numpy(float), 1e-6, 1.0 - 1e-6)
    raw_center, raw_scale = robust_location_scale(fit["sky_raw_log_bf"].to_numpy(float))
    bc_center, bc_scale = robust_location_scale(np.log(fit_bc / (1.0 - fit_bc)))
    transform = {
        "raw_center": raw_center,
        "raw_scale": raw_scale,
        "bc_logit_center": bc_center,
        "bc_logit_scale": bc_scale,
    }
    fit_coordinates = morphology_coordinates(fit, transform)
    tune_coordinates = morphology_coordinates(tune, transform)
    fit_keys = quality_keys(fit, area_boundary)
    tune_keys = quality_keys(tune, area_boundary)
    fit_labels = fit["is_true_pair"].to_numpy(dtype=bool)
    tune_labels = tune["is_true_pair"].to_numpy(dtype=bool)
    cells: dict[str, Any] = {}
    for key in sorted(set(fit_keys) | set(tune_keys)):
        fit_mask = fit_keys == key
        tune_mask = tune_keys == key
        positive = fit_coordinates[fit_mask & fit_labels]
        negative = fit_coordinates[fit_mask & ~fit_labels]
        tune_positive = tune_coordinates[tune_mask & tune_labels]
        tune_negative = tune_coordinates[tune_mask & ~tune_labels]
        rng = np.random.default_rng(stable_seed("negative-density", deployment, seed, key))
        if len(negative) > MAX_NEGATIVE_FIT_PER_CELL:
            negative = negative[rng.choice(len(negative), MAX_NEGATIVE_FIT_PER_CELL, replace=False)]
        if len(tune_negative) > MAX_NEGATIVE_FIT_PER_CELL:
            tune_negative = tune_negative[rng.choice(len(tune_negative), MAX_NEGATIVE_FIT_PER_CELL, replace=False)]
        active = bool(
            len(positive) >= MIN_POSITIVE_CELL
            and len(negative) >= MIN_NEGATIVE_CELL
            and len(tune_positive) >= max(3, MIN_POSITIVE_CELL // 2)
            and len(tune_negative) >= MIN_NEGATIVE_CELL // 2
        )
        cell: dict[str, Any] = {
            "active": active,
            "n_fit_positive": int(len(positive)),
            "n_fit_negative": int(len(negative)),
            "n_tune_positive": int(len(tune_positive)),
            "n_tune_negative": int(len(tune_negative)),
        }
        if not active:
            cell["reason"] = "insufficient local fit/tune support"
            cells[key] = cell
            continue
        combined = np.vstack([positive, negative])
        lower = np.min(combined, axis=0)
        upper = np.max(combined, axis=0)
        width = np.maximum(upper - lower, 1e-3)
        lower -= 0.01 * width
        upper += 0.01 * width
        edges_x = np.linspace(lower[0], upper[0], HISTOGRAM_BINS + 1)
        edges_y = np.linspace(lower[1], upper[1], HISTOGRAM_BINS + 1)
        bandwidth_rows = []
        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for bandwidth in BANDWIDTH_GRID:
            positive_density = _histogram_density(positive, edges_x, edges_y, bandwidth)
            negative_density = _histogram_density(negative, edges_x, edges_y, bandwidth)
            p_pos = _lookup_density(tune_positive, edges_x, edges_y, positive_density)
            p_neg = _lookup_density(tune_negative, edges_x, edges_y, negative_density)
            pos_score = float(np.mean(np.log(np.maximum(p_pos[np.isfinite(p_pos)], 1e-300)))) if np.isfinite(p_pos).any() else -np.inf
            neg_score = float(np.mean(np.log(np.maximum(p_neg[np.isfinite(p_neg)], 1e-300)))) if np.isfinite(p_neg).any() else -np.inf
            objective = pos_score + neg_score
            bandwidth_rows.append({"bandwidth": bandwidth, "positive_log_likelihood": pos_score, "negative_log_likelihood": neg_score, "objective": objective})
            candidate = (objective, -float(bandwidth), positive_density, negative_density)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        selected_bandwidth = -best[1]
        reliability = math.sqrt(min(1.0, len(positive) / 20.0) * min(1.0, len(negative) / 1000.0))
        cell.update(
            {
                "selected_bandwidth": selected_bandwidth,
                "reliability": reliability,
                "edges_x": edges_x.tolist(),
                "edges_y": edges_y.tolist(),
                "positive_density": best[2].tolist(),
                "negative_density": best[3].tolist(),
                "bandwidth_validation": bandwidth_rows,
            }
        )
        cells[key] = cell
    return {
        "deployment": deployment,
        "seed": int(seed),
        "fit_split": "source-disjoint 60% of synthetic validation systems",
        "tune_split": "source-disjoint 40% of synthetic validation systems",
        "features": ["sky_raw_log_bf", "logit(sky_bc)"],
        "transform": transform,
        "area90_boundary_deg2": float(area_boundary),
        "snr_boundary": SNR_BOUNDARY,
        "cells": cells,
        "ood_rule": "inactive/unknown cell or coordinates outside fit support receive neutral morphology score 0",
    }


def apply_morphology_calibrator(frame: pd.DataFrame, calibrator: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    coordinates = morphology_coordinates(out, calibrator["transform"])
    keys = quality_keys(out, float(calibrator["area90_boundary_deg2"]))
    raw_score = np.zeros(len(out), dtype=np.float64)
    reliability = np.zeros(len(out), dtype=np.float64)
    ood = np.ones(len(out), dtype=bool)
    active_cell = np.zeros(len(out), dtype=bool)
    for key in sorted(set(keys)):
        selected = np.flatnonzero(keys == key)
        cell = calibrator["cells"].get(str(key))
        if not cell or not cell.get("active", False):
            continue
        active_cell[selected] = True
        edges_x = np.asarray(cell["edges_x"], dtype=np.float64)
        edges_y = np.asarray(cell["edges_y"], dtype=np.float64)
        positive_density = np.asarray(cell["positive_density"], dtype=np.float64)
        negative_density = np.asarray(cell["negative_density"], dtype=np.float64)
        positive = _lookup_density(coordinates[selected], edges_x, edges_y, positive_density)
        negative = _lookup_density(coordinates[selected], edges_x, edges_y, negative_density)
        valid = np.isfinite(positive) & np.isfinite(negative)
        target = selected[valid]
        raw_score[target] = np.log(np.maximum(positive[valid], 1e-300) / np.maximum(negative[valid], 1e-300))
        reliability[target] = float(cell["reliability"])
        ood[target] = False
    out["sky_quality_key"] = keys
    out["sky_morph_llr_raw"] = raw_score
    out["sky_reliability"] = reliability
    out["sky_morph_ood"] = ood
    out["sky_calibration_cell_active"] = active_cell
    return out


def capped_morphology_score(frame: pd.DataFrame, positive_cap: float, negative_cap: float) -> np.ndarray:
    raw = frame["sky_morph_llr_raw"].to_numpy(dtype=np.float64)
    score = frame["sky_reliability"].to_numpy(dtype=np.float64) * np.clip(raw, -float(negative_cap), float(positive_cap))
    score[frame["sky_morph_ood"].to_numpy(dtype=bool)] = 0.0
    return score


def simplex_weights(step: float = 0.05, require_positive: bool = False) -> list[dict[str, float]]:
    denominator = int(round(1.0 / step))
    rows = []
    for waveform in range(denominator + 1):
        for time_weight in range(denominator - waveform + 1):
            sky = denominator - waveform - time_weight
            if require_positive and min(waveform, time_weight, sky) == 0:
                continue
            rows.append(
                {
                    "waveform": waveform / denominator,
                    "time": time_weight / denominator,
                    "sky": sky / denominator,
                }
            )
    return rows


def pair_metrics_extended(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    result = v7.pair_metrics(frame, scores)
    labels = frame["is_true_pair"].to_numpy(dtype=np.int8)
    finite_scores = np.asarray(scores, dtype=np.float64)
    result["roc_auc"] = float(roc_auc_score(labels, finite_scores))
    order = np.argsort(-finite_scores, kind="stable")
    for budget in (10, 50, 100, 200):
        chosen = labels[order[: min(budget, len(order))]]
        true_count = int(chosen.sum())
        result[f"top_{budget}_true"] = true_count
        result[f"top_{budget}_false"] = int(len(chosen) - true_count)
        result[f"top_{budget}_precision"] = float(true_count / max(len(chosen), 1))
    return result


def score_one_stage(frame: pd.DataFrame, weights: dict[str, float], morph_score: np.ndarray) -> np.ndarray:
    return (
        float(weights["waveform"]) * frame["waveform_score"].to_numpy(dtype=np.float64)
        + float(weights["time"]) * frame["time_score"].to_numpy(dtype=np.float64)
        + float(weights["sky"]) * morph_score
    )


def objective_record(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    return {**v7.retrieval_metrics(frame, scores), **pair_metrics_extended(frame, scores)}


def objective_sort_key(row: dict[str, Any], complexity: float) -> tuple[float, ...]:
    return (
        float(row["average_precision"]),
        float(row["precision_at_recall_0p5"]),
        float(row["macro_r_at_10"]),
        float(row["min_family_r_at_10"]),
        float(row["macro_r_at_1"]),
        -float(complexity),
    )


def topk_frontend_eligibility(frame: pd.DataFrame, frontend_scores: np.ndarray, k: int) -> np.ndarray:
    n = int(frame["event_count"].iloc[0])
    matrix = np.full((n, n), -np.inf, dtype=np.float64)
    ii = frame["idx_i"].to_numpy(np.int32)
    jj = frame["idx_j"].to_numpy(np.int32)
    matrix[ii, jj] = frontend_scores
    matrix[jj, ii] = frontend_scores
    directed = np.zeros((n, n), dtype=bool)
    present = np.unique(np.concatenate([ii, jj]))
    for query in present:
        candidates = np.flatnonzero(np.isfinite(matrix[query]))
        if not len(candidates):
            continue
        take = min(int(k), len(candidates))
        selected = candidates[np.argsort(-matrix[query, candidates], kind="stable")[:take]]
        directed[query, selected] = True
    return directed[ii, jj] | directed[jj, ii]


def score_two_stage(
    frame: pd.DataFrame,
    frozen_weights: dict[str, float],
    morph_score: np.ndarray,
    k: int,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    frontend = (
        float(frozen_weights["waveform"]) * frame["waveform_score"].to_numpy(dtype=np.float64)
        + float(frozen_weights["time"]) * frame["time_score"].to_numpy(dtype=np.float64)
    )
    eligible = topk_frontend_eligibility(frame, frontend, k)
    scores = frontend + float(eta) * morph_score
    floor = min(float(np.min(scores)) - 1e6, -1e12)
    scores = np.where(eligible, scores, floor)
    return scores, eligible


def select_one_stage(tune: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best: dict[str, Any] | None = None
    positive_grid = simplex_weights(0.05, require_positive=True)
    for positive_cap, negative_cap in CAP_GRID:
        morph = capped_morphology_score(tune, positive_cap, negative_cap)
        for weights in positive_grid:
            scores = score_one_stage(tune, weights, morph)
            metrics = objective_record(tune, scores)
            row = {
                "positive_cap": positive_cap,
                "negative_cap": negative_cap,
                **weights,
                **metrics,
            }
            rows.append(row)
            key = objective_sort_key(metrics, sum(value * value for value in weights.values()) + 0.01 * (positive_cap + negative_cap))
            if best_key is None or key > best_key:
                best_key = key
                best = row
    assert best is not None
    selected = {
        "positive_cap": float(best["positive_cap"]),
        "negative_cap": float(best["negative_cap"]),
        "weights": {name: float(best[name]) for name in ("waveform", "time", "sky")},
        "selection_metrics": {key: value for key, value in best.items() if key not in {"positive_cap", "negative_cap", "waveform", "time", "sky"}},
    }
    return selected, pd.DataFrame(rows)


def select_two_stage(
    tune: pd.DataFrame,
    frozen_weights: dict[str, float],
    positive_cap: float,
    negative_cap: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    morph = capped_morphology_score(tune, positive_cap, negative_cap)
    rows: list[dict[str, Any]] = []
    selected_by_k: dict[str, Any] = {}
    for k in K_GRID:
        best_key: tuple[float, ...] | None = None
        best: dict[str, Any] | None = None
        for eta in ETA_GRID:
            scores, eligible = score_two_stage(tune, frozen_weights, morph, k, eta)
            metrics = objective_record(tune, scores)
            row = {"k": k, "eta": eta, "eligible_fraction": float(np.mean(eligible)), **metrics}
            rows.append(row)
            key = objective_sort_key(metrics, abs(float(eta)))
            if best_key is None or key > best_key:
                best_key = key
                best = row
        assert best is not None
        selected_by_k[str(k)] = {"eta": float(best["eta"]), "selection_metrics": {key: value for key, value in best.items() if key not in {"k", "eta"}}}
    return {
        "frontend_weights": {"waveform": float(frozen_weights["waveform"]), "time": float(frozen_weights["time"])},
        "positive_cap": float(positive_cap),
        "negative_cap": float(negative_cap),
        "primary_k": PRIMARY_K,
        "selected_by_k": selected_by_k,
    }, pd.DataFrame(rows)


def phase2(root: Path) -> None:
    selected_all: dict[str, Any] = {
        "experiment": "sky_morphology_one_vs_two_stage_exploratory",
        "selection_data": "source-disjoint synthetic validation only",
        "heldout_test_used": False,
        "real_catalog_used": False,
        "deployments": {},
    }
    for deployment in DEPLOYMENTS:
        selected_all["deployments"][deployment] = {}
        for seed in SEEDS:
            seed_root = root / "results" / deployment / f"seed_{seed}"
            validation = pd.read_parquet(seed_root / "synthetic_validation_pair_morphology_nside512.parquet")
            source_events = pd.read_parquet(input_seed_dir(deployment, seed) / "results/mixed_val_synthetic_events_v7.parquet")
            event_partition = validation_system_partition(source_events, deployment, seed)
            validation["calibration_subset"] = pair_partition(validation, event_partition)
            event_morph = pd.read_csv(seed_root / "synthetic_validation_event_sky_morphology_nside512.csv")
            partition_lookup = event_partition.set_index("idx")["calibration_subset"]
            event_morph["calibration_subset"] = event_morph["idx"].map(partition_lookup)
            fit_event = event_morph.loc[event_morph["calibration_subset"] == "fit"]
            area_boundary = float(fit_event["area90_deg2"].median())
            fit = validation.loc[validation["calibration_subset"] == "fit"].reset_index(drop=True)
            tune = validation.loc[validation["calibration_subset"] == "tune"].reset_index(drop=True)
            calibrator = fit_morphology_calibrator(fit, tune, area_boundary, deployment, seed)
            tune_scored = apply_morphology_calibrator(tune, calibrator)
            one_stage, one_grid = select_one_stage(tune_scored)
            two_stage, two_grid = select_two_stage(
                tune_scored,
                FROZEN_V93_WEIGHTS[deployment][seed],
                one_stage["positive_cap"],
                one_stage["negative_cap"],
            )
            config = {
                "deployment": deployment,
                "seed": int(seed),
                "paired_baseline_C": {
                    "definition": "ordering-corrected common Nside=512 raw sky with frozen v9.3 weights",
                    "weights": FROZEN_V93_WEIGHTS[deployment][seed],
                },
                "calibrator": calibrator,
                "one_stage_A": one_stage,
                "two_stage_B": two_stage,
                "partition_counts": validation["calibration_subset"].value_counts().to_dict(),
                "test_used_for_selection": False,
                "real_PE_or_LVK_used_for_selection": False,
            }
            config_path = seed_root / "selected_config.json"
            write_json(config_path, config)
            config["sha256"] = sha256_file(config_path)
            write_json(config_path, config)
            write_csv(seed_root / "validation_system_partition.csv", event_partition)
            write_csv(seed_root / "one_stage_validation_grid.csv", one_grid)
            write_csv(seed_root / "two_stage_validation_grid.csv", two_grid)
            selected_all["deployments"][deployment][str(seed)] = config
    selected_path = root / "contracts/selected_config.json"
    write_json(selected_path, selected_all)
    selected_hash = sha256_file(selected_path)
    write_json(root / "contracts/selected_config_sha256.json", {"path": str(selected_path), "sha256": selected_hash})
    phase_marker(root, "PHASE2", {"passed": True, "selected_config_sha256": selected_hash})


def method_scores(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    calibrated = apply_morphology_calibrator(frame, config["calibrator"])
    one = config["one_stage_A"]
    morph = capped_morphology_score(calibrated, one["positive_cap"], one["negative_cap"])
    unbounded_reliable = calibrated["sky_reliability"].to_numpy(float) * calibrated["sky_morph_llr_raw"].to_numpy(float)
    frozen = config["paired_baseline_C"]["weights"]
    baseline = (
        float(frozen["waveform"]) * calibrated["waveform_score"].to_numpy(float)
        + float(frozen["time"]) * calibrated["time_score"].to_numpy(float)
        + float(frozen["sky"]) * calibrated["sky_raw_log_bf"].to_numpy(float)
    )
    waveform = calibrated["waveform_score"].to_numpy(float)
    time_score = calibrated["time_score"].to_numpy(float)
    raw_sky = calibrated["sky_raw_log_bf"].to_numpy(float)
    results: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
        "waveform_only": (waveform, None),
        "time_only": (time_score, None),
        "sky_raw_only": (raw_sky, None),
        "sky_morph_only": (morph, None),
        "waveform_time": (float(frozen["waveform"]) * waveform + float(frozen["time"]) * time_score, None),
        "C_v93_weights_corrected_common_maps": (baseline, None),
        "A_one_stage_morphology": (score_one_stage(calibrated, one["weights"], morph), None),
    }
    two = config["two_stage_B"]
    for k in K_GRID:
        eta = float(two["selected_by_k"][str(k)]["eta"])
        scores, eligible = score_two_stage(calibrated, frozen, morph, k, eta)
        name = "B_two_stage_primary" if k == PRIMARY_K else f"B_two_stage_K{k}_sensitivity"
        results[name] = (scores, eligible)
    calibrated["sky_score_new"] = morph
    frame["sky_morph_llr_raw"] = calibrated["sky_morph_llr_raw"].to_numpy()
    frame["sky_reliability"] = calibrated["sky_reliability"].to_numpy()
    frame["sky_morph_ood"] = calibrated["sky_morph_ood"].to_numpy()
    frame["sky_score_new"] = morph
    frame["sky_positive_clipped"] = (~frame["sky_morph_ood"].to_numpy(bool)) & (unbounded_reliable > morph + 1e-12)
    frame["sky_negative_clipped"] = (~frame["sky_morph_ood"].to_numpy(bool)) & (unbounded_reliable < morph - 1e-12)
    frame["sky_negative_veto"] = (~frame["sky_morph_ood"].to_numpy(bool)) & (morph < 0.0)
    return results


def phase3(root: Path) -> None:
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    selected = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            seed_root = root / "results" / deployment / f"seed_{seed}"
            config = selected["deployments"][deployment][str(seed)]
            for split in ("validation", "test"):
                frame = pd.read_parquet(seed_root / f"synthetic_{split}_pair_morphology_nside512.parquet")
                methods = method_scores(frame, config)
                for method, (scores, eligible) in methods.items():
                    retrieval = v7.retrieval_metrics(frame, scores)
                    pair = pair_metrics_extended(frame, scores)
                    for subset in ("overall", "sis", "pm"):
                        ranks = v7.query_rank_rows(frame, scores, method)
                        values = ranks["query_rank"].to_numpy(np.int32) if subset == "overall" else ranks.loc[ranks["family"].str.lower() == subset, "query_rank"].to_numpy(np.int32)
                        metric_rows.append(
                            {
                                "deployment": deployment,
                                "seed": seed,
                                "split": split,
                                "method": method,
                                "subset": subset,
                                "r_at_1": float(np.mean(values <= 1)),
                                "r_at_5": float(np.mean(values <= 5)),
                                "r_at_10": float(np.mean(values <= 10)),
                                "median_rank": float(np.median(values)),
                                "n_queries": len(values),
                            }
                        )
                    pair_rows.append(
                        {
                            "deployment": deployment,
                            "seed": seed,
                            "split": split,
                            "method": method,
                            "eligible_fraction": float(np.mean(eligible)) if eligible is not None else 1.0,
                            "sky_morph_ood_rate": float(frame["sky_morph_ood"].mean()),
                            "sky_positive_clip_rate": float(frame["sky_positive_clipped"].mean()),
                            "sky_negative_clip_rate": float(frame["sky_negative_clipped"].mean()),
                            "sky_negative_veto_rate": float(frame["sky_negative_veto"].mean()),
                            **pair,
                        }
                    )
                labels = frame["is_true_pair"].to_numpy(dtype=bool)
                for population, mask in (("companion", labels), ("non_companion", ~labels)):
                    for feature in ("sky_raw_log_bf", "sky_bc", "sky_score_new"):
                        values = frame.loc[mask, feature].to_numpy(float)
                        distribution_rows.append(distribution_summary(deployment, seed, split, population, feature, values))
                for feature in ("sky_raw_log_bf", "sky_score_new"):
                    concentration_rows.extend(
                        positive_concentration_rows(deployment, seed, split, feature, frame[feature].to_numpy(float))
                    )
                frame[[
                    "idx_i", "idx_j", "is_true_pair", "true_pair_family", "sky_raw_log_bf", "sky_bc", "sky_j50", "sky_j90",
                    "sky_morph_llr_raw", "sky_reliability", "sky_morph_ood", "sky_score_new",
                    "sky_positive_clipped", "sky_negative_clipped", "sky_negative_veto",
                ]].to_parquet(seed_root / f"synthetic_{split}_sky_scored_audit.parquet", index=False)
    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    write_csv(root / "results/retrieval_metrics_per_seed.csv", metrics)
    write_csv(root / "results/pair_metrics_per_seed.csv", pairs)
    write_csv(root / "results/sky_distribution_statistics.csv", pd.DataFrame(distribution_rows))
    write_csv(root / "results/sky_positive_reward_concentration.csv", pd.DataFrame(concentration_rows))
    test_metrics = metrics.loc[metrics["split"] == "test"]
    summary = numeric_summary(test_metrics, ["deployment", "method", "subset"])
    write_csv(root / "results/retrieval_metrics_summary.csv", summary)
    pair_summary = numeric_summary(pairs.loc[pairs["split"] == "test"], ["deployment", "method"])
    write_csv(root / "results/pair_metrics_summary.csv", pair_summary)
    write_csv(root / "results/paired_delta_vs_C.csv", paired_deltas(metrics, pairs))
    phase_marker(root, "PHASE3", {"passed": True, "locked_test_evaluated_once": True})


def distribution_summary(
    deployment: str,
    seed: int,
    split: str,
    population: str,
    feature: str,
    values: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "deployment": deployment,
        "seed": seed,
        "split": split,
        "population": population,
        "feature": feature,
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "q995": float(np.quantile(values, 0.995)),
        "q999": float(np.quantile(values, 0.999)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def positive_concentration_rows(
    deployment: str,
    seed: int,
    split: str,
    feature: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    positive = np.sort(np.asarray(values, dtype=np.float64)[np.asarray(values) > 0])[::-1]
    total = float(positive.sum())
    rows = []
    for label, count in (
        ("top_1", 1),
        ("top_5", 5),
        ("top_10", 10),
        ("top_1_percent", max(1, int(math.ceil(0.01 * len(values))))),
        ("top_5_percent", max(1, int(math.ceil(0.05 * len(values))))),
    ):
        rows.append(
            {
                "deployment": deployment,
                "seed": seed,
                "split": split,
                "feature": feature,
                "budget": label,
                "n_positive_pairs": int(len(positive)),
                "positive_reward_total": total,
                "concentration_fraction": float(positive[:count].sum() / total) if total > 0 else 0.0,
            }
        )
    return rows


def paired_deltas(metrics: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    baseline = "C_v93_weights_corrected_common_maps"
    metric_columns = ["r_at_1", "r_at_5", "r_at_10", "median_rank"]
    pair_columns = [
        "average_precision",
        "false_at_recall_0p5",
        "false_at_recall_0p9",
        "top_10_precision",
        "top_50_precision",
        "top_100_precision",
        "top_200_precision",
    ]
    rows: list[dict[str, Any]] = []
    test_metrics = metrics.loc[(metrics["split"] == "test") & (metrics["subset"] == "overall")]
    test_pairs = pairs.loc[pairs["split"] == "test"]
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            base_m = test_metrics.loc[
                (test_metrics["deployment"] == deployment) & (test_metrics["seed"] == seed) & (test_metrics["method"] == baseline)
            ].iloc[0]
            base_p = test_pairs.loc[
                (test_pairs["deployment"] == deployment) & (test_pairs["seed"] == seed) & (test_pairs["method"] == baseline)
            ].iloc[0]
            for method in ("A_one_stage_morphology", "B_two_stage_primary"):
                new_m = test_metrics.loc[
                    (test_metrics["deployment"] == deployment) & (test_metrics["seed"] == seed) & (test_metrics["method"] == method)
                ].iloc[0]
                new_p = test_pairs.loc[
                    (test_pairs["deployment"] == deployment) & (test_pairs["seed"] == seed) & (test_pairs["method"] == method)
                ].iloc[0]
                row: dict[str, Any] = {"deployment": deployment, "seed": seed, "method": method, "baseline": baseline}
                for column in metric_columns:
                    row[f"delta_{column}"] = float(new_m[column] - base_m[column])
                for column in pair_columns:
                    row[f"delta_{column}"] = float(new_p[column] - base_p[column])
                rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in groups and column != "seed"]
    rows: list[dict[str, Any]] = []
    for key, part in frame.groupby(groups, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, keys))
        row["n_seeds"] = int(part["seed"].nunique()) if "seed" in part else len(part)
        for column in numeric:
            values = part[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_q25"] = float(np.quantile(values, 0.25))
            row[f"{column}_q75"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_real_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    method: str,
    config: dict[str, Any],
    eligible: np.ndarray | None,
) -> pd.DataFrame:
    out = frame.copy()
    out["final_score"] = np.asarray(scores, dtype=np.float64)
    out["method"] = method
    out["frontend_eligible"] = eligible if eligible is not None else True
    frozen = config["paired_baseline_C"]["weights"]
    if method == "C_v93_weights_corrected_common_maps":
        out["waveform_contribution"] = float(frozen["waveform"]) * out["waveform_score"]
        out["time_contribution"] = float(frozen["time"]) * out["time_score"]
        out["sky_contribution"] = float(frozen["sky"]) * out["sky_raw_log_bf"]
    elif method == "A_one_stage_morphology":
        weights = config["one_stage_A"]["weights"]
        out["waveform_contribution"] = float(weights["waveform"]) * out["waveform_score"]
        out["time_contribution"] = float(weights["time"]) * out["time_score"]
        out["sky_contribution"] = float(weights["sky"]) * out["sky_score_new"]
    elif method.startswith("B_two_stage"):
        k = PRIMARY_K if method == "B_two_stage_primary" else int(method.split("K")[1].split("_")[0])
        eta = float(config["two_stage_B"]["selected_by_k"][str(k)]["eta"])
        out["waveform_contribution"] = float(frozen["waveform"]) * out["waveform_score"]
        out["time_contribution"] = float(frozen["time"]) * out["time_score"]
        out["sky_contribution"] = eta * out["sky_score_new"]
        out.loc[~out["frontend_eligible"].astype(bool), "sky_contribution"] = 0.0
    else:
        out["waveform_contribution"] = np.nan
        out["time_contribution"] = np.nan
        out["sky_contribution"] = np.nan
    out = out.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def consensus_table(frames: list[pd.DataFrame], scheme: str) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    value_columns = [
        "final_score",
        "waveform_score",
        "time_score",
        "sky_raw_log_bf",
        "sky_bc",
        "sky_j50",
        "sky_j90",
        "sky_morph_llr_raw",
        "sky_reliability",
        "sky_score_new",
        "waveform_contribution",
        "time_contribution",
        "sky_contribution",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "seeds": ("seed", "nunique"),
        "rank_mean": ("rank", "mean"),
        "rank_std": ("rank", "std"),
        "rank_min": ("rank", "min"),
        "rank_max": ("rank", "max"),
        "ood_seed_count": ("sky_morph_ood", "sum"),
        "positive_clip_seed_count": ("sky_positive_clipped", "sum"),
        "negative_clip_seed_count": ("sky_negative_clipped", "sum"),
        "negative_veto_seed_count": ("sky_negative_veto", "sum"),
        "frontend_eligible_seed_count": ("frontend_eligible", "sum"),
    }
    for column in value_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
    consensus = (
        combined.groupby(["pair_key", "event_i", "event_j"], as_index=False)
        .agg(**aggregations)
        .sort_values(["rank_mean", "rank_max", "final_score_mean"], ascending=[True, True, False], kind="stable")
        .reset_index(drop=True)
    )
    consensus.insert(0, "scheme", scheme)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1, dtype=np.int32))
    return consensus


def resolve_pe_path(value: Any, deployment: str) -> Path:
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [PROJECT / raw, source_run(deployment) / raw]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(candidates[0])


def load_distance_posteriors(deployment: str, event_names: set[str]) -> dict[str, np.ndarray]:
    import h5py

    manifest = primary_manifest(pd.read_csv(source_run(deployment) / "data/event_manifest.csv"))
    group_column = "sky_map_internal_group" if deployment == "gwtc3" else "sky_map_group"
    output: dict[str, np.ndarray] = {}
    for _, row in manifest.loc[manifest["event_name"].astype(str).isin(event_names)].iterrows():
        path = resolve_pe_path(row["sky_map_path"], deployment)
        group = str(row[group_column])
        with h5py.File(path, "r") as handle:
            posterior = handle[f"{group}/posterior_samples"]
            names = set(posterior.dtype.names or ())
            key = next((item for item in ("luminosity_distance", "distance", "d_L") if item in names), None)
            if key is None:
                continue
            count = len(posterior)
            index = np.linspace(0, count - 1, min(count, 30_000), dtype=np.int64)
            values = np.asarray(posterior[key][index], dtype=np.float64)
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                output[str(row["event_name"])] = values
    return output


def bhattacharyya_coefficient_1d(x: np.ndarray, y: np.ndarray, bins: int = 256) -> float:
    lo = min(float(np.quantile(x, 0.001)), float(np.quantile(y, 0.001)))
    hi = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
    if hi <= lo:
        return float(np.isclose(np.median(x), np.median(y)))
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges)
    py, _ = np.histogram(y, bins=edges)
    px = px / max(int(px.sum()), 1)
    py = py / max(int(py.sum()), 1)
    return float(np.sum(np.sqrt(px * py)))


def attach_pe_official(
    deployment: str,
    consensus: pd.DataFrame,
    pe: pd.DataFrame,
    official: pd.DataFrame,
    distance_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = consensus.copy()
    pe = pe.copy()
    pe["canonical_key"] = [canonical_pair(a, b) for a, b in zip(pe["event_i"], pe["event_j"])]
    keep = [
        "canonical_key",
        "pe_available",
        "chirp_mass_bhattacharyya_coefficient",
        "mass_ratio_bhattacharyya_coefficient",
        "chi_eff_bhattacharyya_coefficient",
        "chirp_mass_standardized_posterior_distance",
        "mass_ratio_standardized_posterior_distance",
        "chi_eff_standardized_posterior_distance",
        "max_standardized_posterior_distance",
        "intrinsic_3sigma_consistent",
    ]
    out = out.merge(pe[keep], left_on="pair_key", right_on="canonical_key", how="left", validate="one_to_one").drop(columns="canonical_key")
    official = official.loc[official["deployment"].astype(str) == deployment].copy()
    official["pair_key"] = [canonical_pair(a, b) for a, b in zip(official["event_i"], official["event_j"])]
    official = official[["pair_key", "audit_label", "provenance_class"]].drop_duplicates("pair_key")
    out = out.merge(official, on="pair_key", how="left", validate="one_to_one")
    out["official_fpp"] = np.nan
    out["official_stage_and_conclusion"] = out["provenance_class"].fillna("not_in_frozen_local_historical_audit_table")
    out["official_frontend_overlap"] = out["provenance_class"].notna()
    out["public_hanabi_table_overlap"] = out["provenance_class"].fillna("").str.contains("LVK_literature", regex=False)
    distance_bc = []
    for row in out.itertuples(index=False):
        left = distance_cache.get(str(row.event_i))
        right = distance_cache.get(str(row.event_j))
        distance_bc.append(bhattacharyya_coefficient_1d(left, right) if left is not None and right is not None else np.nan)
    out["apparent_luminosity_distance_bhattacharyya_coefficient"] = distance_bc
    return out


def phase4(root: Path) -> None:
    selected = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    official_path = PROJECT / "results/sky_background_fast_followup_v104_20260824_20260824T080136Z/tables/frozen_lvk_historical_candidate_comparison.csv"
    official = pd.read_csv(official_path)
    all_top_rows: list[pd.DataFrame] = []
    budget_rows: list[dict[str, Any]] = []
    rank_change_rows: list[dict[str, Any]] = []
    real_distribution_rows: list[dict[str, Any]] = []
    primary_schemes = (
        "C_v93_weights_corrected_common_maps",
        "A_one_stage_morphology",
        "B_two_stage_primary",
    )
    for deployment in DEPLOYMENTS:
        dep_root = root / "results" / deployment
        morphology = pd.read_parquet(dep_root / "real_pair_sky_morphology_nside512.parquet")
        morphology_columns = [
            "idx_i", "idx_j", "event_i", "event_j", "sky_raw_log_bf", "sky_bc", "sky_j50", "sky_j90",
            "sky_coverage50_i_in_j", "sky_coverage50_j_in_i", "sky_coverage90_i_in_j", "sky_coverage90_j_in_i",
            "sky_area50_deg2_i", "sky_area50_deg2_j", "sky_area90_deg2_i", "sky_area90_deg2_j", "sky_area90_ratio",
            "sky_min_snr", "sky_min_detector_count", "sky_max_area90_deg2",
        ]
        by_scheme: dict[str, list[pd.DataFrame]] = {name: [] for name in primary_schemes}
        for seed in SEEDS:
            config = selected["deployments"][deployment][str(seed)]
            base = pd.read_parquet(corrected_seed_dir(deployment, seed) / "real_pair_features_unified_sky_v81.parquet")
            base = base.drop(columns=[column for column in base.columns if column.startswith("sky_")], errors="ignore")
            frame = base.merge(morphology[morphology_columns], on=["idx_i", "idx_j", "event_i", "event_j"], how="left", validate="one_to_one")
            frame = frame.loc[frame["strict_h1l1_bbh_pair"].astype(bool)].copy().reset_index(drop=True)
            frame["event_count"] = int(primary_manifest(pd.read_csv(source_run(deployment) / "data/event_manifest.csv")).shape[0])
            frame["pair_key"] = [canonical_pair(a, b) for a, b in zip(frame["event_i"], frame["event_j"])]
            methods = method_scores(frame, config)
            for scheme in primary_schemes:
                scores, eligible = methods[scheme]
                ranked = rank_real_frame(frame, scores, scheme, config, eligible)
                ranked["seed"] = int(seed)
                ranked.to_parquet(dep_root / f"seed_{seed}/real_strict_pair_scores_{scheme}.parquet", index=False)
                write_csv(dep_root / f"seed_{seed}/real_strict_top50_{scheme}.csv", ranked.head(50))
                by_scheme[scheme].append(ranked)
            calibrated = apply_morphology_calibrator(frame, config["calibrator"])
            morph_score = capped_morphology_score(
                calibrated,
                config["one_stage_A"]["positive_cap"],
                config["one_stage_A"]["negative_cap"],
            )
            for feature, values in (
                ("sky_raw_log_bf", frame["sky_raw_log_bf"].to_numpy(float)),
                ("sky_bc", frame["sky_bc"].to_numpy(float)),
                ("sky_score_new", morph_score),
            ):
                real_distribution_rows.append(distribution_summary(deployment, seed, "real", "all_pairs", feature, values))

        consensus_tables: dict[str, pd.DataFrame] = {}
        for scheme, frames in by_scheme.items():
            stability = pd.concat(frames, ignore_index=True)
            consensus = consensus_table(frames, scheme)
            stability.to_parquet(dep_root / f"real_rank_stability_{scheme}.parquet", index=False)
            consensus.to_parquet(dep_root / f"real_consensus_{scheme}.parquet", index=False)
            write_csv(dep_root / f"real_consensus_top50_{scheme}.csv", consensus.head(50))
            consensus_tables[scheme] = consensus

        union_events: set[str] = set()
        for table in consensus_tables.values():
            union_events.update(table.head(50)["event_i"].astype(str))
            union_events.update(table.head(50)["event_j"].astype(str))
        distance_cache = load_distance_posteriors(deployment, union_events)
        pe_path = PROJECT / "results/gwtc_sky_ordering_corrected_v94_20260830" / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        pe = pd.read_parquet(pe_path)
        enriched: dict[str, pd.DataFrame] = {}
        for scheme, table in consensus_tables.items():
            result = attach_pe_official(deployment, table, pe, official, distance_cache)
            result.to_parquet(dep_root / f"real_consensus_with_pe_official_{scheme}.parquet", index=False)
            write_csv(dep_root / f"real_top50_with_pe_official_{scheme}.csv", result.head(50))
            enriched[scheme] = result
            top = result.head(10).copy()
            top.insert(0, "deployment", deployment)
            all_top_rows.append(top)
            for budget in (10, 20, 50):
                chosen = result.head(budget)
                budget_rows.append(
                    {
                        "deployment": deployment,
                        "scheme": scheme,
                        "budget": budget,
                        "n_pairs": len(chosen),
                        "chirp_mass_bc_ge_0p5": int((chosen["chirp_mass_bhattacharyya_coefficient"] >= 0.5).sum()),
                        "chirp_mass_bc_median": float(chosen["chirp_mass_bhattacharyya_coefficient"].median()),
                        "dmax_le_3": int((chosen["max_standardized_posterior_distance"] <= 3).sum()),
                        "official_frontend_overlap": int(chosen["official_frontend_overlap"].sum()),
                        "public_hanabi_table_overlap": int(chosen["public_hanabi_table_overlap"].sum()),
                    }
                )
        baseline = enriched["C_v93_weights_corrected_common_maps"].set_index("pair_key")
        for scheme in ("A_one_stage_morphology", "B_two_stage_primary"):
            comparison = enriched[scheme].set_index("pair_key")
            joined = baseline[["consensus_rank", "event_i", "event_j"]].join(
                comparison[["consensus_rank"]], lsuffix="_baseline", rsuffix="_new", how="outer"
            )
            joined["rank_change_new_minus_baseline"] = joined["consensus_rank_new"] - joined["consensus_rank_baseline"]
            joined["deployment"] = deployment
            joined["scheme"] = scheme
            joined["pair_key"] = joined.index
            rank_change_rows.extend(joined.reset_index(drop=True).to_dict("records"))
    write_csv(root / "results/real_top10_all_schemes.csv", pd.concat(all_top_rows, ignore_index=True))
    write_csv(root / "results/pe_official_budget_summary.csv", pd.DataFrame(budget_rows))
    write_csv(root / "results/rank_change_audit.csv", pd.DataFrame(rank_change_rows))
    real_distribution = pd.DataFrame(real_distribution_rows)
    existing_distribution = pd.read_csv(root / "results/sky_distribution_statistics.csv")
    write_csv(root / "results/sky_distribution_statistics.csv", pd.concat([existing_distribution, real_distribution], ignore_index=True))
    phase_marker(root, "PHASE4", {"passed": True, "real_catalog_used_for_selection": False, "candidate_claim": "shortlist for Bayesian follow-up only"})


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


def make_top10_map_figures(root: Path) -> None:
    top = pd.read_csv(root / "results/real_top10_all_schemes.csv")
    short_names = {
        "C_v93_weights_corrected_common_maps": "C_baseline",
        "A_one_stage_morphology": "A_one_stage",
        "B_two_stage_primary": "B_two_stage",
    }
    for deployment in DEPLOYMENTS:
        manifest = primary_manifest(pd.read_csv(source_run(deployment) / "data/event_manifest.csv")).set_index("event_name")
        needed = set(top.loc[top["deployment"] == deployment, "event_i"].astype(str))
        needed.update(top.loc[top["deployment"] == deployment, "event_j"].astype(str))
        maps: dict[str, np.ndarray] = {}
        for event_name in sorted(needed):
            probability, _ = v94.corrected_read_probability_map(manifest.loc[event_name], target_nside=128)
            probability = np.asarray(probability, dtype=np.float64)
            maps[event_name] = probability / max(float(probability.max()), 1e-30)
        for scheme, short in short_names.items():
            part = top.loc[(top["deployment"] == deployment) & (top["scheme"] == scheme)].sort_values("consensus_rank").head(10)
            figure = plt.figure(figsize=(10.5, 12.5))
            for row_number, row in enumerate(part.itertuples(index=False)):
                for side, event_name in enumerate((str(row.event_i), str(row.event_j))):
                    panel = row_number * 2 + side + 1
                    suffix = "i" if side == 0 else "j"
                    detail = "" if side == 0 else f"\nBC={row.sky_bc_mean:.2f}, J90={row.sky_j90_mean:.2f}"
                    hp.mollview(
                        maps[event_name],
                        fig=figure.number,
                        sub=(5, 4, panel),
                        title=f"#{int(row.consensus_rank)}{suffix} {event_name}{detail}",
                        min=0.0,
                        max=1.0,
                        cbar=False,
                        notext=True,
                        cmap="viridis",
                    )
            figure.suptitle(
                f"{deployment.upper()} {scheme}: public PE sky posteriors (visualized at Nside=128)",
                fontsize=11,
                fontweight="bold",
                y=0.995,
            )
            figure.savefig(root / f"figures/fig_real_top10_skymaps_{deployment}_{short}.pdf", bbox_inches="tight")
            figure.savefig(root / f"figures/fig_real_top10_skymaps_{deployment}_{short}.png", dpi=200, bbox_inches="tight")
            plt.close(figure)


def make_figures(root: Path) -> None:
    set_plot_style()
    metrics = pd.read_csv(root / "results/retrieval_metrics_per_seed.csv")
    metrics = metrics.loc[(metrics["split"] == "test") & (metrics["subset"] == "overall")]
    methods = ["C_v93_weights_corrected_common_maps", "A_one_stage_morphology", "B_two_stage_primary"]
    labels = ["C: frozen weights\ncorrected maps", "A: one-stage\nmorphology", "B: WT then sky\nrerank"]
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    for column, deployment in enumerate(DEPLOYMENTS):
        part = metrics.loc[metrics["deployment"] == deployment]
        for row_index, measure in enumerate(("r_at_1", "r_at_10")):
            ax = axes[row_index, column]
            for position, (method, color) in enumerate(zip(methods, colors)):
                values = part.loc[part["method"] == method, measure].to_numpy(float)
                ax.scatter(np.full(len(values), position) + np.linspace(-0.08, 0.08, len(values)), values, color=color, s=25, zorder=3)
                ax.errorbar(position, np.mean(values), yerr=np.std(values, ddof=1), color="black", marker="_", capsize=3, linewidth=0.8)
            ax.set_xticks(range(3), labels, rotation=12, ha="right")
            ax.set_ylabel(measure.replace("r_at_", "R@").replace("_", ""))
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"{'abcd'[row_index * 2 + column]}  {deployment.upper()} held-out test", loc="left")
    fig.savefig(root / "figures/fig_retrieval_comparison.pdf", bbox_inches="tight")
    fig.savefig(root / "figures/fig_retrieval_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    distributions = pd.read_csv(root / "results/sky_distribution_statistics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), constrained_layout=True)
    features = ("sky_raw_log_bf", "sky_bc", "sky_score_new")
    for row_index, deployment in enumerate(DEPLOYMENTS):
        for column, feature in enumerate(features):
            ax = axes[row_index, column]
            selected = distributions.loc[
                (distributions["deployment"] == deployment)
                & (distributions["feature"] == feature)
                & (distributions["seed"] == SEEDS[0])
            ]
            for population, color in (("companion", "#009E73"), ("non_companion", "#777777"), ("all_pairs", "#D55E00")):
                expected_split = "real" if population == "all_pairs" else "test"
                row = selected.loc[(selected["population"] == population) & (selected["split"] == expected_split)]
                if len(row):
                    median = float(row["median"].iloc[0])
                    ax.errorbar(
                        [population], [median],
                        yerr=[[median - float(row["q25"].iloc[0])], [float(row["q75"].iloc[0]) - median]],
                        fmt="o", color=color, capsize=2,
                    )
            ax.tick_params(axis="x", rotation=25)
            ax.set_title(f"{'abcdef'[row_index * 3 + column]}  {deployment.upper()} {feature}", loc="left", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "figures/fig_sky_morphology_distribution.pdf", bbox_inches="tight")
    fig.savefig(root / "figures/fig_sky_morphology_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top = pd.read_csv(root / "results/real_top10_all_schemes.csv")
    budget = pd.read_csv(root / "results/pe_official_budget_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    for ax, deployment, letter in zip(axes, DEPLOYMENTS, "ab"):
        part = budget.loc[(budget["deployment"] == deployment) & (budget["budget"] == 10)].set_index("scheme")
        x = np.arange(len(methods))
        pe_pass = [part.loc[method, "dmax_le_3"] for method in methods]
        mc_pass = [part.loc[method, "chirp_mass_bc_ge_0p5"] for method in methods]
        ax.bar(x - 0.18, pe_pass, 0.36, label=r"$D_{max}\leq3$", color="#4C78A8")
        ax.bar(x + 0.18, mc_pass, 0.36, label=r"$BC_{\mathcal{M}_c}\geq0.5$", color="#F2CF5B")
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylim(0, 10.5)
        ax.set_ylabel("Pairs in Top-10")
        ax.set_title(f"{letter}  {deployment.upper()} post-ranking audit", loc="left")
        ax.legend(loc="lower left", fontsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "figures/fig_real_candidate_comparison.pdf", bbox_inches="tight")
    fig.savefig(root / "figures/fig_real_candidate_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    make_top10_map_figures(root)


def format_metric_table(summary: pd.DataFrame, method: str) -> list[str]:
    rows = []
    for deployment in DEPLOYMENTS:
        row = summary.loc[(summary["deployment"] == deployment) & (summary["method"] == method) & (summary["subset"] == "overall")].iloc[0]
        rows.append(
            f"| {deployment.upper()} | {row['r_at_1_mean']:.4f} ± {row['r_at_1_sd']:.4f} | {row['r_at_10_mean']:.4f} ± {row['r_at_10_sd']:.4f} | {row['median_rank_mean']:.2f} ± {row['median_rank_sd']:.2f} |"
        )
    return rows


def make_report(root: Path) -> None:
    retrieval = pd.read_csv(root / "results/retrieval_metrics_summary.csv")
    pair = pd.read_csv(root / "results/pair_metrics_summary.csv")
    budget = pd.read_csv(root / "results/pe_official_budget_summary.csv")
    top = pd.read_csv(root / "results/real_top10_all_schemes.csv")
    selected = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    methods = (
        "C_v93_weights_corrected_common_maps",
        "A_one_stage_morphology",
        "B_two_stage_primary",
    )
    lines = [
        "# 天空形态校准的一阶段与两阶段探索对照完整报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "> 本轮为独立探索实验。historical v9.3、ordering-corrected v9.4、ET-3、论文和 Overleaf 均未覆盖。结果只用于方法审计，不授权替换论文结果。",
        "",
        "## 1. 方法边界",
        "",
        "波形 encoder、waveform score、一维 time score、strict H1-L1 BBH 事件范围和原始 validation/test source split 全部冻结。本轮唯一新增的是由完整 HEALPix posterior 计算的天空形态特征与其 validation-only 校准。",
        "",
        "历史 v9.3 的公开 PE HDF5 ordering 读取存在已确认的 NESTED/RING 错误，因此历史数字仅作为 provenance。公平成对对照 C 使用原 v9.3 每个 seed 的冻结权重，但与 A/B 共用严格修正 ordering 后重新生成的 Nside=512 地图。这样比较只改变天空评分规则，不把已知读取错误当成基线优势或劣势。",
        "",
        "每个 synthetic event 从同一 observing run、network SNR 邻近的公开 PE posterior 中选模板，并把一次 posterior draw 旋转到模拟真方向。伴随像共享真方向，但模板、posterior draw 和真实噪声条件独立。地图不落盘为 dense bank，只保存 pair 特征和可重建 provenance。",
        "",
        "主形态输入为 `(raw log B_sky, BC_sky)`。J50、J90、双向 HPD 覆盖、A50/A90、entropy、KL 和多峰数只作诊断。校准按 detector network、SNR 和 A90 分层；局部支持不足或超出 fit 支持域时，新天空正奖励回退为 0。",
        "",
        "具体地，对已经归一化为像素概率质量的两张图，首先计算",
        "",
        "```text",
        "B_sky = Npix * sum_k P_i(k) P_j(k)",
        "Z_raw = log(max(B_sky, 1e-30))",
        "BC_sky = sum_k sqrt(P_i(k) P_j(k))",
        "J_a = |HPD_a(i) intersect HPD_a(j)| / |HPD_a(i) union HPD_a(j)|, a=50%,90%",
        "```",
        "",
        "`B_sky` 是相对于两个独立各向同性方向的共同方向证据；`BC_sky` 衡量整张后验概率质量的形态相似度。二者物理含义不同，因此只把 `(Z_raw, logit(BC_sky))` 作为二维校准输入，不把 BC 冒充第二个独立 Bayes factor。raw Bayes factor 的 `1e-30` 只是明确冻结的数值 floor；它只影响已经极强的负证据尾部，后续负分还会再被 validation cap 限制。",
        "",
        "每个质量层 Q 分别建立 companion 与 non-companion 的二维平滑直方图密度，定义 `Z_morph=log[p(x|L,Q,O-run)/p(x|N,Q,O-run)]`。层内正样本少于 8、负样本少于 100、tune 支持不足，或 pair 落在 fit 矩形支持域外时，`Z_new=0`；有效层使用 `sqrt(min(1,nL/20)*min(1,nN/1000))` 可靠度缩放，并在 tune 上选择正/负 cap。",
        "",
        "三臂分数定义为：C 使用冻结 v9.3 权重对 `W+T+raw sky` 加权；A 使用 validation 选择的严格正 simplex 权重对 `W+T+new sky` 一阶段相加；B 先按冻结的 `W+T` 为每个事件保留 Top-K，再在这个池内加 `eta*new sky`，池外 pair 永远不能被天空救回。",
        "",
        "## 2. 数据隔离",
        "",
        "每个 seed 的原 validation systems 再按 global source ID 确定性切成 60% density-fit 与 40% tune。fit 只拟合正/负二维密度；tune 只选择 bandwidth、分数 cap、一阶段权重和两阶段 eta。held-out test 只评估一次。真实 PE、候选排名和公开候选表均在配置冻结后附加。",
        "",
        "## 3. Held-out companion retrieval",
        "",
    ]
    for method in methods:
        lines.extend(
            [
                f"### {method}",
                "",
                "| deployment | R@1 | R@10 | median rank |",
                "|---|---:|---:|---:|",
                *format_metric_table(retrieval, method),
                "",
            ]
        )
    lines.extend(["## 4. Pair-level false burden", "", "| deployment | method | AUPRC | false pairs at 50% recall | false pairs at 90% recall |", "|---|---|---:|---:|---:|"])
    for _, row in pair.loc[pair["method"].isin(methods)].iterrows():
        lines.append(
            f"| {row['deployment'].upper()} | {row['method']} | {row['average_precision_mean']:.5f} ± {row['average_precision_sd']:.5f} | {row['false_at_recall_0p5_mean']:.1f} ± {row['false_at_recall_0p5_sd']:.1f} | {row['false_at_recall_0p9_mean']:.1f} ± {row['false_at_recall_0p9_sd']:.1f} |"
        )
    lines.extend(["", "## 5. Validation-selected configurations", ""])
    for deployment in DEPLOYMENTS:
        lines.append(f"### {deployment.upper()}")
        lines.append("")
        for seed in SEEDS:
            config = selected["deployments"][deployment][str(seed)]
            weights = config["one_stage_A"]["weights"]
            eta = config["two_stage_B"]["selected_by_k"][str(PRIMARY_K)]["eta"]
            lines.append(
                f"- `{seed}`: A weights W/T/S={weights['waveform']:.2f}/{weights['time']:.2f}/{weights['sky']:.2f}, caps +{config['one_stage_A']['positive_cap']:.1f}/-{config['one_stage_A']['negative_cap']:.1f}; B K={PRIMARY_K}, eta={eta:.2f}."
            )
        lines.append("")
    lines.extend(["## 6. 冻结后的真实目录审计", "", "真实目录没有透镜真值。下表中的 pair 是 candidate shortlist for Bayesian follow-up，不是 detections。PE 和公开阶段只用于后验审计，不参与任何配置选择。", ""])
    for deployment in DEPLOYMENTS:
        for scheme in methods:
            lines.extend([f"### {deployment.upper()} — {scheme}", "", "| rank | event_i | event_j | score | raw sky | BC sky | J90 | Mc BC | Dmax | official/LVK provenance |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"])
            part = top.loc[(top["deployment"] == deployment) & (top["scheme"] == scheme)].sort_values("consensus_rank")
            for _, row in part.iterrows():
                lines.append(
                    f"| {int(row['consensus_rank'])} | {row['event_i']} | {row['event_j']} | {row['final_score_mean']:.4f} | {row['sky_raw_log_bf_mean']:.4f} | {row['sky_bc_mean']:.4f} | {row['sky_j90_mean']:.4f} | {row['chirp_mass_bhattacharyya_coefficient']:.3f} | {row['max_standardized_posterior_distance']:.2f} | {row['official_stage_and_conclusion']} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 7. 科学解释限制",
            "",
            "1. synthetic sky 仍是公开 PE posterior template 经旋转后的 map-matched surrogate，不是为每个注入重新运行完整 BBH PE。",
            "2. fit/tune/test 在 source level 隔离，但同一 catalog 内大量 non-companion pairs 共享事件，因此 pair-level uncertainty 不能按独立 Bernoulli pair 解释。",
            "3. 两阶段 K=50 在规模较小的真实 strict catalog 中是较宽的前端池；K=20/100 敏感性结果均保留，不能只挑最好的一项。",
            "4. PE 一致性和公开候选重合不是透镜真值，不能用于反向调权。",
            "5. 官方 FPP 在本地冻结输入中并非对全部候选可用，缺失值保持 NA，未作推测或补造。",
            "",
            f"最终状态：`{FINAL_STATUS}`。",
        ]
    )
    (root / "reports/EXPLORATORY_SKY_MORPHOLOGY_ONE_VS_TWO_STAGE_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "experiment": "sky_morphology_one_vs_two_stage_exploratory",
        "status": FINAL_STATUS,
        "analysis_nside": ANALYSIS_NSIDE,
        "historical_v9_3_overwritten": False,
        "paired_baseline": "frozen v9.3 weights on ordering-corrected common Nside=512 maps",
        "retrieval_summary": retrieval.to_dict("records"),
        "pair_summary": pair.to_dict("records"),
        "pe_official_budget": budget.to_dict("records"),
    }
    write_json(root / "results/sequence_summary.json", summary)


def finalize(root: Path) -> tuple[Path, str]:
    make_figures(root)
    make_report(root)
    for phase in range(5):
        log_path = PROJECT / "logs" / f"sky_morphology_phase{phase}_final_20260830.log"
        if log_path.is_file():
            shutil.copy2(log_path, root / "logs" / log_path.name)
    required = [
        "contracts/data_contract.json",
        "contracts/selected_config.json",
        "contracts/input_sha256_manifest.csv",
        "results/retrieval_metrics_per_seed.csv",
        "results/retrieval_metrics_summary.csv",
        "results/pair_metrics_summary.csv",
        "results/sky_distribution_statistics.csv",
        "results/real_top10_all_schemes.csv",
        "results/pe_official_budget_summary.csv",
        "results/rank_change_audit.csv",
        "figures/fig_sky_morphology_distribution.pdf",
        "figures/fig_retrieval_comparison.pdf",
        "figures/fig_real_candidate_comparison.pdf",
        "reports/EXPLORATORY_SKY_MORPHOLOGY_ONE_VS_TWO_STAGE_CN.md",
        f"scripts/{SCRIPT_PATH.name}",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required deliverables: {missing}")
    protected = json.loads((root / "contracts/historical_protection_hashes.json").read_text(encoding="utf-8"))
    protection_rows = []
    for record in protected:
        path = Path(record["path"])
        current = sha256_file(path) if path.is_file() else None
        protection_rows.append(
            {
                **record,
                "sha256_after": current,
                "unchanged": bool(current == record["sha256_before"]),
            }
        )
    if not all(row["unchanged"] for row in protection_rows):
        raise RuntimeError("A protected historical artifact changed during the experiment")
    write_json(root / "manifest/historical_nonoverwrite_audit.json", protection_rows)
    shutil.copy2(SCRIPT_PATH, root / "scripts" / SCRIPT_PATH.name)
    manifest_path = root / "manifest/SHA256SUMS.txt"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest_path):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    phase_marker(root, "FINAL", {"passed": True, "status": FINAL_STATUS, "required_deliverables": len(required)})
    # Refresh manifest after writing the final marker.
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest_path):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    package_dir = PROJECT / "packages"
    package_dir.mkdir(exist_ok=True)
    package = package_dir / f"{root.name}_final.tar.gz"
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {package}")
    with tarfile.open(package, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    package_hash = sha256_file(package)
    (package.with_suffix(package.suffix + ".sha256")).write_text(f"{package_hash}  {package.name}\n", encoding="utf-8")
    return package, package_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT / f"results/sky_morphology_one_vs_two_stage_exploratory_20260827_{utc_stamp()}",
    )
    parser.add_argument("--phase", choices=("0", "1", "2", "3", "4", "final", "all"), default="all")
    args = parser.parse_args()
    if args.phase in {"0", "all"}:
        phase0(args.output_root)
    if args.phase in {"1", "all"}:
        phase1(args.output_root)
    if args.phase in {"2", "all"}:
        phase2(args.output_root)
    if args.phase in {"3", "all"}:
        phase3(args.output_root)
    if args.phase in {"4", "all"}:
        phase4(args.output_root)
    if args.phase in {"final", "all"}:
        package, digest = finalize(args.output_root)
        print(f"[complete] output={args.output_root}", flush=True)
        print(f"[complete] package={package}", flush=True)
        print(f"[complete] package_sha256={digest}", flush=True)
        print(FINAL_STATUS, flush=True)


if __name__ == "__main__":
    main()
