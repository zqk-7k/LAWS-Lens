#!/usr/bin/env python3
"""Nside=512 map-matched synthetic-sky and frozen-ranking pilot.

This is an independent exploratory pilot.  It regenerates synthetic event sky
posteriors directly from run-matched public PE maps at Nside=512, without
upsampling the historical Nside=64 synthetic maps.  Fusion weights are selected
on synthetic validation pairs only, then frozen for held-out synthetic test and
one descriptive real-catalog reranking.  The historical v9.3 products are read
only and never overwritten.

The pilot uses one pre-registered model seed per deployment to provide a fast
map-level answer before any multi-seed expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEPLOYMENTS = ("gwtc3", "gwtc4")
PILOT_SEED = 202607241
ANALYSIS_NSIDE = 512
WORKERS = 8
BASE_ROTATION_SEED = 2026082308


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_pair(a: object, b: object) -> str:
    x, y = str(a), str(b)
    return f"{x}--{y}" if x <= y else f"{y}--{x}"


def pair_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for left, right in (("event_i", "event_j"), ("event_1", "event_2"), ("event_a", "event_b")):
        if left in frame and right in frame:
            return left, right
    raise KeyError(frame.columns.tolist())


def add_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    left, right = pair_columns(frame)
    out = frame.copy()
    out["pair_key"] = [canonical_pair(a, b) for a, b in zip(out[left], out[right])]
    return out


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    cross = np.asarray(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) * math.cos(angle) + (1.0 - math.cos(angle)) * np.outer(unit, unit) + math.sin(angle) * cross


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
        ]
    )
    return np.eye(3) + cross_matrix + cross_matrix @ cross_matrix * ((1.0 - cosine) / (sine * sine))


def rotate_map(
    probability: np.ndarray,
    true_ra: float,
    true_dec: float,
    seed: int,
    output_vectors: np.ndarray,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    source_map = np.asarray(probability, dtype=np.float64)
    source_map = np.clip(source_map, 0.0, None)
    source_map /= source_map.sum(dtype=np.float64)
    anchor = int(rng.choice(len(source_map), p=source_map))
    source = np.asarray(hp.pix2vec(ANALYSIS_NSIDE, anchor, nest=False), dtype=np.float64)
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


def primary_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["include_in_primary_search"].fillna(False).astype(bool)
    mask &= frame["sky_map_available"].fillna(False).astype(bool)
    return frame.loc[mask].sort_values("gps_time").reset_index(drop=True)


def source_run(project: Path, deployment: str) -> Path:
    if deployment == "gwtc3":
        return project / "runs" / "real_gwtc_lensing_search_20260625"
    return project / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"


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


def detector_count(value: object) -> int:
    return len([part for part in str(value).split(",") if part.strip()])


def load_template_library(
    project: Path,
    deployment: str,
    read_probability_map,
) -> tuple[pd.DataFrame, list[np.ndarray], list[dict[str, Any]], list[Path]]:
    manifest_path = source_run(project, deployment) / "data" / "event_manifest.csv"
    manifest = primary_manifest(pd.read_csv(manifest_path))
    maps: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    used_paths = [manifest_path]
    for index, row in manifest.iterrows():
        probability, metadata = read_probability_map(row, target_nside=ANALYSIS_NSIDE)
        probability = np.asarray(probability, dtype=np.float32)
        probability = np.clip(probability, 0.0, None)
        probability /= probability.sum(dtype=np.float64)
        maps.append(probability)
        map_path = Path(str(row["sky_map_path"]))
        if not map_path.is_absolute():
            map_path = project / map_path
        if map_path.exists():
            used_paths.append(map_path)
        rows.append(
            {
                "template_index": index,
                "event_name": str(row["event_name"]),
                "run": str(row["run"]),
                "network_snr": float(row["network_snr"]),
                "detectors_available": str(row["detectors_available"]),
                "detector_count": detector_count(row["detectors_available"]),
                "source_nside": metadata.get("source_nside"),
                "source_ordering": metadata.get("source_ordering"),
                "source_format": metadata.get("source_format"),
            }
        )
    return pd.DataFrame(rows), maps, rows, used_paths


def choose_templates(events: pd.DataFrame, templates: pd.DataFrame, deployment: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        run = infer_run(float(event.gps_obs), deployment)
        same_run = templates.index[templates["run"].astype(str) == run].to_numpy(dtype=np.int32)
        pool = same_run if same_run.size >= 4 else templates.index.to_numpy(dtype=np.int32)
        template_snr = templates.loc[pool, "network_snr"].to_numpy(float)
        distances = np.abs(np.log(np.maximum(template_snr, 1e-6)) - math.log(max(float(event.snr), 1e-6)))
        nearest = pool[np.argsort(distances)[: min(4, len(pool))]]
        chosen = int(rng.choice(nearest))
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
                "template_network_snr": float(template["network_snr"]),
                "template_detectors": str(template["detectors_available"]),
                "snr_log_distance": float(abs(math.log(max(float(event.snr), 1e-6)) - math.log(max(float(template["network_snr"]), 1e-6)))),
            }
        )
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def generate_maps(
    events: pd.DataFrame,
    assignment: pd.DataFrame,
    template_maps: list[np.ndarray],
    deployment: str,
    split: str,
    output_vectors: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    npix = hp.nside2npix(ANALYSIS_NSIDE)
    maps = np.empty((len(events), npix), dtype=np.float32)
    anchors = np.full(len(events), -1, dtype=np.int64)
    start = time.perf_counter()

    def task(row: Any) -> tuple[int, np.ndarray, int]:
        event_index = int(row.idx)
        stable = hashlib.sha256(f"{deployment}:{split}:{event_index}:{BASE_ROTATION_SEED}".encode()).digest()
        rotation_seed = int.from_bytes(stable[:8], "little") % (2**32 - 1)
        rotated, anchor = rotate_map(
            template_maps[int(row.template_index)],
            float(row.ra_true),
            float(row.dec_true),
            rotation_seed,
            output_vectors,
        )
        return event_index, rotated, anchor

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
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


def log_bf_matrix_float32(maps: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    normalized = np.asarray(maps, dtype=np.float32)
    normalized /= normalized.sum(axis=1, dtype=np.float64)[:, None].astype(np.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    tensor = torch.from_numpy(normalized).to(device=device, dtype=torch.float32)
    overlap = tensor @ tensor.T
    log_bf = torch.log(torch.clamp(overlap * normalized.shape[1], min=1e-30)).cpu().numpy().astype(np.float64)
    elapsed = time.perf_counter() - start
    del tensor, overlap
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.fill_diagonal(log_bf, -np.inf)
    return log_bf, {"device": str(device), "seconds": elapsed, "accumulation": "float32 GPU pilot"}


def float64_pair_audit(
    maps: np.ndarray,
    approximate: np.ndarray,
    base: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    true_mask = base["is_true_pair"].fillna(False).astype(bool).to_numpy()
    ii = base["idx_i"].to_numpy(np.int32)
    jj = base["idx_j"].to_numpy(np.int32)
    true_rows = np.flatnonzero(true_mask)
    null_rows = np.flatnonzero(~true_mask)
    rng = np.random.default_rng(seed)
    selected_null = rng.choice(null_rows, size=min(1000, len(null_rows)), replace=False)
    selected = np.concatenate([true_rows, selected_null])
    errors = []
    for row_index in selected:
        left, right = int(ii[row_index]), int(jj[row_index])
        overlap = np.sum(maps[left].astype(np.float64) * maps[right].astype(np.float64), dtype=np.float64)
        exact = math.log(max(maps.shape[1] * overlap, 1e-300))
        errors.append(abs(exact - float(approximate[left, right])))
    error = np.asarray(errors)
    return {
        "n_pairs": int(len(error)),
        "median_abs_error_nats": float(np.median(error)),
        "q99_abs_error_nats": float(np.quantile(error, 0.99)),
        "max_abs_error_nats": float(np.max(error)),
        "pilot_tolerance_nats": 1e-3,
        "pass": bool(np.max(error) <= 1e-3),
    }


def attach_sky(base: pd.DataFrame, log_bf: np.ndarray) -> pd.DataFrame:
    output = base.drop(
        columns=[c for c in base.columns if c.startswith("sky_") or c in {"sky_score", "sky_bayes_factor"}],
        errors="ignore",
    ).copy()
    ii = output["idx_i"].to_numpy(np.int32)
    jj = output["idx_j"].to_numpy(np.int32)
    values = log_bf[ii, jj].astype(np.float32)
    output["sky_log_bayes_factor_raw"] = values
    output["sky_score"] = values
    output["sky_bayes_factor"] = np.exp(np.clip(values, -80.0, 80.0))
    return output


def method_weights(v7, validation: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    methods: dict[str, dict[str, float]] = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
    }
    grids: dict[str, pd.DataFrame] = {}
    for objective in ("retrieval", "candidate"):
        unconstrained, grid = v7.select_fusion_weights(validation, require_all_positive=False, objective=objective)
        methods[f"{objective}_three_channel_unconstrained"] = unconstrained
        grids[f"{objective}_three_channel_unconstrained"] = grid
        positive, grid = v7.select_fusion_weights(validation, require_all_positive=True, objective=objective)
        methods[f"{objective}_three_channel_strict_positive"] = positive
        grids[f"{objective}_three_channel_strict_positive"] = grid
    return methods, grids


def rerank_real(
    v7,
    seed_dir: Path,
    resolution: pd.DataFrame,
    methods: dict[str, dict[str, float]],
    output_seed: Path,
) -> dict[str, pd.DataFrame]:
    real = pd.read_parquet(seed_dir / "real_pair_features_unified_sky_v81.parquet")
    real = add_pair_key(real)
    merged = real.merge(
        resolution[["pair_key", "sky_log_bf_nside512"]],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    if merged["sky_log_bf_nside512"].isna().any():
        raise RuntimeError("Missing real Nside=512 sky scores")
    merged["sky_score"] = merged["sky_log_bf_nside512"].astype(np.float32)
    merged["sky_log_bayes_factor_raw"] = merged["sky_score"]
    merged["sky_bayes_factor"] = np.exp(np.clip(merged["sky_score"], -80.0, 80.0))
    merged.to_parquet(output_seed / "real_pair_features_nside512_map_matched_pilot.parquet", index=False)
    outputs: dict[str, pd.DataFrame] = {}
    for method, weights in methods.items():
        strict = v7._rank_real_pairs(merged.loc[merged["strict_h1l1_bbh_pair"]].copy(), weights, method)
        strict.to_parquet(output_seed / f"real_strict_pair_scores_{method}.parquet", index=False)
        write_csv(strict.head(100), output_seed / f"real_strict_candidate_top100_{method}.csv")
        outputs[method] = strict
    return outputs


def run_deployment(
    project: Path,
    deployment: str,
    output: Path,
    v7,
    read_probability_map,
    output_vectors: np.ndarray,
) -> tuple[dict[str, Any], list[Path]]:
    source_seed = project / "results" / "real_noise_injection_v7_peak2s_formal_20260722" / deployment / f"seed_{PILOT_SEED}"
    v93_seed = project / "results" / "gwtc_sky_resolution_v93_20260730" / deployment / f"seed_{PILOT_SEED}"
    output_seed = output / deployment / f"seed_{PILOT_SEED}"
    output_seed.mkdir(parents=True)
    template_frame, template_maps, _, template_paths = load_template_library(project, deployment, read_probability_map)
    write_csv(template_frame, output_seed / "public_pe_template_library_nside512.csv")
    resource_rows: list[dict[str, Any]] = []
    split_frames: dict[str, pd.DataFrame] = {}
    used_paths: list[Path] = [*template_paths]
    for split, short in (("validation", "val"), ("test", "test")):
        event_path = source_seed / "results" / f"mixed_{short}_synthetic_events_v7.parquet"
        base_path = source_seed / "results" / ("fusion_validation_pairs_v7.parquet" if split == "validation" else "fusion_heldout_test_pairs_v7.parquet")
        events = pd.read_parquet(event_path)
        base = pd.read_parquet(base_path)
        assignment = choose_templates(
            events,
            template_frame,
            deployment,
            BASE_ROTATION_SEED + (0 if split == "validation" else 1000),
        )
        maps, assignment, generation_seconds = generate_maps(
            events, assignment, template_maps, deployment, split, output_vectors
        )
        log_bf, matrix_resource = log_bf_matrix_float32(maps)
        numerical_audit = float64_pair_audit(
            maps, log_bf, base, BASE_ROTATION_SEED + (1 if split == "validation" else 2)
        )
        if not numerical_audit["pass"]:
            raise RuntimeError(f"Float32 overlap pilot failed float64 audit: {numerical_audit}")
        split_frame = attach_sky(base, log_bf)
        split_frames[split] = split_frame
        split_frame.to_parquet(output_seed / f"synthetic_{split}_pairs_nside512_map_matched.parquet", index=False)
        write_csv(assignment, output_seed / f"synthetic_{split}_event_template_assignment.csv")
        write_json(numerical_audit, output_seed / f"synthetic_{split}_float64_overlap_audit.json")
        resource_rows.append(
            {
                "deployment": deployment,
                "split": split,
                "n_events": len(events),
                "n_pairs": len(base),
                "map_generation_seconds": generation_seconds,
                "matrix_seconds": matrix_resource["seconds"],
                "matrix_device": matrix_resource["device"],
                "dense_peak_array_gib": maps.nbytes / 2**30,
                "dense_maps_persisted": False,
            }
        )
        used_paths.extend([event_path, base_path])
        del maps, log_bf

    methods, grids = method_weights(v7, split_frames["validation"])
    write_json(
        {
            "selection_split": "synthetic validation only",
            "analysis_nside": ANALYSIS_NSIDE,
            "template_selection": "same observing run, then random among four nearest network-SNR templates",
            "methods": methods,
        },
        output_seed / "selected_weights_nside512_map_matched.json",
    )
    for name, grid in grids.items():
        write_csv(grid, output_seed / f"weight_grid_{name}.csv")

    metric_frames = []
    pair_metric_frames = []
    for split in ("validation", "test"):
        query, metrics, pair_metrics = v7.evaluation_tables(
            split_frames[split], methods, deployment, PILOT_SEED
        )
        query.insert(2, "split", split)
        metrics.insert(2, "split", split)
        pair_metrics.insert(2, "split", split)
        query.to_parquet(output_seed / f"{split}_query_ranks.parquet", index=False)
        metric_frames.append(metrics)
        pair_metric_frames.append(pair_metrics)
    metrics = pd.concat(metric_frames, ignore_index=True)
    pair_metrics = pd.concat(pair_metric_frames, ignore_index=True)
    write_csv(metrics, output_seed / "retrieval_metrics.csv")
    write_csv(pair_metrics, output_seed / "pair_metrics.csv")
    write_csv(pd.DataFrame(resource_rows), output_seed / "resource_usage.csv")

    resolution_path = project / "results" / "gwtc_sky_resolution_v93_20260730" / deployment / "real_sky_resolution_convergence_all_pairs_v93.parquet"
    resolution = add_pair_key(pd.read_parquet(resolution_path))
    real_outputs = rerank_real(v7, v93_seed, resolution, methods, output_seed)
    used_paths.extend([resolution_path, v93_seed / "real_pair_features_unified_sky_v81.parquet"])

    primary_method = "candidate_three_channel_strict_positive"
    top = real_outputs[primary_method].head(10)
    left, right = pair_columns(top)
    top_rows = [
        {
            "rank": int(row.rank),
            "event_i": str(getattr(row, left)),
            "event_j": str(getattr(row, right)),
            "final_score": float(row.final_score),
            "waveform_score": float(row.waveform_score),
            "time_score": float(row.time_score),
            "sky_score": float(row.sky_score),
        }
        for row in top.itertuples(index=False)
    ]
    test_overall = metrics.loc[
        (metrics["split"] == "test") & (metrics["subset"] == "overall")
    ]
    summary = {
        "deployment": deployment,
        "pilot_seed": PILOT_SEED,
        "nside": ANALYSIS_NSIDE,
        "selected_weights": methods,
        "test_overall_metrics": test_overall.to_dict("records"),
        "real_top10_primary_method": top_rows,
        "resource_usage": resource_rows,
    }
    write_json(summary, output_seed / "deployment_summary.json")
    return summary, used_paths


def make_report(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    lines = [
        "# Nside=512 真实 PE 模板匹配天空 pilot",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 实验边界",
        "",
        "本轮是独立 v10 快速 map-level pilot，不覆盖 v9.3。它不把历史 Nside=64 注入图插值到 512，而是从同一公开 PE posterior 源直接 rasterize 到 Nside=512；每个 synthetic event 按 observing run 和 network SNR 选择模板，再将一次 posterior draw 对齐到模拟真实方向。",
        "",
        f"为优先获得快速结果，本轮只使用预注册 model seed `{PILOT_SEED}`。waveform/time、事件划分和标签保持冻结；融合权重仅在 synthetic validation 选择。真实候选只在规则冻结后描述性重排一次。",
        "",
        "GPU 使用 float32 计算完整 overlap matrix，并对全部真对加 1,000 个固定随机假对用 float64 逐对复算；最大误差必须小于 1e-3 nats。正式多种子版本仍应按协议采用 float64 累积或同等精度实现。",
        "",
        "## Held-out test 指标",
        "",
        "| catalog | method | R@1 | R@5 | R@10 | median rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for deployment, summary in summaries.items():
        for row in summary["test_overall_metrics"]:
            lines.append(
                f"| {deployment.upper()} | {row['method']} | {row['r_at_1']:.4f} | {row['r_at_5']:.4f} | {row['r_at_10']:.4f} | {row['median_rank']:.1f} |"
            )
    lines.extend(["", "## Validation-selected weights", ""])
    for deployment, summary in summaries.items():
        lines.append(f"### {deployment.upper()}")
        lines.append("")
        for method, weights in summary["selected_weights"].items():
            if "three_channel" in method:
                lines.append(
                    f"- `{method}`: waveform={weights['waveform']}, time={weights['time']}, sky={weights['sky']}"
                )
        lines.append("")
    lines.extend(["## 真实目录 Top-10（描述性，不是 detection）", ""])
    for deployment, summary in summaries.items():
        lines.append(f"### {deployment.upper()}")
        lines.append("")
        lines.append("| rank | event_i | event_j | final | waveform | time | sky |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for row in summary["real_top10_primary_method"]:
            lines.append(
                f"| {row['rank']} | {row['event_i']} | {row['event_j']} | {row['final_score']:.4f} | {row['waveform_score']:.4f} | {row['time_score']:.4f} | {row['sky_score']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释限制",
            "",
            "该 pilot 仍是公开 PE 模板 surrogate，而不是对每个注入从 strain 重新运行完整天空 PE。它检验了 Nside=512、run/SNR 条件化和真实后验形态能否修正主要错配；快速相干 PE pilot 用于随后验证该 surrogate。只有多 seed 和快速/严格 PE 对照通过后，才可考虑建立新的 locked test。",
            "",
            "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。",
        ]
    )
    (output / "nside512_map_matched_pilot_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    methods = ["waveform_only", "time_only", "sky_only", "retrieval_three_channel_strict_positive"]
    labels = ["Waveform", "Time", "Sky", "Three-channel"]
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    for ax, deployment, letter in zip(axes, DEPLOYMENTS, "ab"):
        frame = pd.DataFrame(summaries[deployment]["test_overall_metrics"]).set_index("method")
        x = np.arange(len(methods))
        r1 = [frame.loc[method, "r_at_1"] for method in methods]
        r10 = [frame.loc[method, "r_at_10"] for method in methods]
        width = 0.34
        ax.bar(x - width / 2, r1, width, color=colors, alpha=0.65, label="R@1")
        ax.bar(x + width / 2, r10, width, color=colors, hatch="//", alpha=0.85, label="R@10")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Companion retrieval")
        ax.set_title(f"{letter}  {deployment.upper()} Nside=512 pilot", loc="left")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.legend(fontsize=7, loc="upper left")
    figure_dir = output / "figures"
    figure_dir.mkdir()
    fig.savefig(figure_dir / "fig_nside512_map_matched_pilot.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_nside512_map_matched_pilot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/gw-catalog"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    os.chdir(project)
    output = (args.output or project / "results" / f"sky_map_matched_nside512_pilot_v103_20260823_{stamp()}").resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = {
        "experiment": "sky_map_matched_nside512_pilot_v103",
        "status": "EXPLORATORY_PILOT",
        "analysis_nside": ANALYSIS_NSIDE,
        "pilot_seed": PILOT_SEED,
        "synthetic_maps_regenerated_from_public_pe": True,
        "historical_nside64_maps_upsampled": False,
        "weight_selection": "synthetic validation only",
        "heldout_test_used_for_selection": False,
        "real_candidate_rank_used_for_selection": False,
        "dense_maps_persisted": False,
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    write_json(contract, output / "analysis_contract.json")

    v7 = load_module("v7_common_pilot", project / "scripts" / "real_search" / "unified_v7_common.py")
    common = load_module("real_search_common_pilot", project / "scripts" / "real_search" / "common.py")
    npix = hp.nside2npix(ANALYSIS_NSIDE)
    output_vectors = np.asarray(hp.pix2vec(ANALYSIS_NSIDE, np.arange(npix), nest=False), dtype=np.float64)
    summaries: dict[str, dict[str, Any]] = {}
    used_paths: list[Path] = []
    for deployment in DEPLOYMENTS:
        summary, paths = run_deployment(
            project,
            deployment,
            output,
            v7,
            common.read_probability_map,
            output_vectors,
        )
        summaries[deployment] = summary
        used_paths.extend(paths)
    del output_vectors
    write_json(
        {
            "experiment": "sky_map_matched_nside512_pilot_v103",
            "deployments": summaries,
            "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
        },
        output / "pilot_summary.json",
    )
    make_report(summaries, output)
    make_figure(summaries, output)
    scripts = output / "scripts"
    scripts.mkdir()
    shutil.copy2(Path(__file__).resolve(), scripts / Path(__file__).name)
    write_csv(
        pd.DataFrame(
            [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(set(path for path in used_paths if path.exists()))
            ]
        ),
        output / "input_manifest.csv",
    )
    manifest = output / "manifest"
    manifest.mkdir()
    files = [path for path in output.rglob("*") if path.is_file()]
    write_csv(
        pd.DataFrame(
            [
                {"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(files)
            ]
        ),
        manifest / "files_sha256.csv",
    )
    latest = project / "results" / "sky_map_matched_nside512_pilot_v103_LATEST.txt"
    latest.write_text(str(output) + "\n", encoding="utf-8")
    package = project / "packages" / f"{output.name}.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    package_sha = sha256_file(package)
    package.with_suffix(package.suffix + ".sha256").write_text(f"{package_sha}  {package.name}\n", encoding="ascii")
    print(json.dumps({"output": str(output), "package": str(package), "sha256": package_sha}, indent=2))


if __name__ == "__main__":
    main()
