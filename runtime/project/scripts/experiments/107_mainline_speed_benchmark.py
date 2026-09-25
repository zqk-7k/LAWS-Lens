from __future__ import annotations

import argparse
import copy
import gc
import importlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import hnswlib
import numpy as np
import pandas as pd
import psutil
import torch
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sky.sky_posterior_overlap import gaussian_log_cosine_overlap_pairs  # noqa: E402


DEFAULT_SEED_DIR = REPO_ROOT / "results" / "mainline_uncertainty_20260713" / "et3" / "seed_202607101"
DEFAULT_OUT = REPO_ROOT / "results" / "mainline_speed_benchmark_20260714"
ET3_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
O4_RUN = REPO_ROOT / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"
SECONDS_PER_DAY = 86400.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def sync_device(device: str | torch.device | None) -> None:
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_repetitions(
    fn: Callable[[], Any],
    warmups: int,
    repeats: int,
    device: str | torch.device | None = None,
) -> tuple[list[float], Any]:
    last = None
    for _ in range(max(0, warmups)):
        last = fn()
        sync_device(device)
    values: list[float] = []
    for _ in range(max(1, repeats)):
        sync_device(device)
        start = time.perf_counter()
        last = fn()
        sync_device(device)
        values.append(time.perf_counter() - start)
    return values, last


def timing_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median_s": float(np.median(data)),
        "q25_s": float(np.quantile(data, 0.25)),
        "q75_s": float(np.quantile(data, 0.75)),
        "min_s": float(np.min(data)),
        "max_s": float(np.max(data)),
    }


def normalize_embeddings(embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32).copy()
    array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
    return array


def synthesize_embeddings(base: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, str]:
    if size <= len(base):
        return np.ascontiguousarray(base[:size]), "native_et3_subset"
    rng = np.random.default_rng(seed)
    out = np.empty((size, base.shape[1]), dtype=np.float32)
    chunk = 100_000
    for start in range(0, size, chunk):
        stop = min(size, start + chunk)
        indices = rng.integers(0, len(base), size=stop - start)
        block = base[indices] + rng.normal(0.0, 0.02, size=(stop - start, base.shape[1])).astype(np.float32)
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        out[start:stop] = block
    return out, "tiled_perturbed_et3_embeddings_scaling_only"


def remove_self(labels: np.ndarray, query_ids: np.ndarray, k: int) -> np.ndarray:
    out = np.full((len(query_ids), k), -1, dtype=np.int32)
    for row, query in enumerate(query_ids):
        keep = labels[row][labels[row] != query]
        out[row, : min(k, len(keep))] = keep[:k]
    return out


def remove_self_with_scores(
    labels: np.ndarray,
    distances: np.ndarray,
    query_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_labels = np.full((len(query_ids), k), -1, dtype=np.int32)
    out_scores = np.full((len(query_ids), k), -np.inf, dtype=np.float32)
    for row, query in enumerate(query_ids):
        keep = labels[row] != query
        kept_labels = labels[row][keep][:k]
        kept_scores = (1.0 - distances[row][keep][:k]).astype(np.float32)
        out_labels[row, : len(kept_labels)] = kept_labels
        out_scores[row, : len(kept_scores)] = kept_scores
    return out_labels, out_scores


def build_hnsw(
    embedding: np.ndarray,
    threads: int,
    seed: int,
    m: int = 32,
    ef_construction: int = 256,
) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=embedding.shape[1])
    index.init_index(
        max_elements=len(embedding),
        ef_construction=int(ef_construction),
        M=int(m),
        random_seed=int(seed % 100_000),
    )
    index.set_num_threads(int(threads))
    index.add_items(embedding, np.arange(len(embedding), dtype=np.int32))
    return index


def exact_topk(embedding: np.ndarray, k: int) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    score = embedding @ embedding.T
    np.fill_diagonal(score, -np.inf)
    matrix_s = time.perf_counter() - start
    start = time.perf_counter()
    part = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(score, part, axis=1)
    order = np.argsort(-selected, axis=1)
    result = np.take_along_axis(part, order, axis=1).astype(np.int32)
    topk_s = time.perf_counter() - start
    del score, part, selected, order
    gc.collect()
    return result, matrix_s, topk_s


def score_matrix_topk(score: np.ndarray, k: int) -> np.ndarray:
    """Return row-wise top-k indices without sorting every catalog row in full."""
    part = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(score, part, axis=1)
    order = np.argsort(-selected, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def load_et_context(seed_dir: Path, with_raw: bool = False) -> dict[str, Any]:
    base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
    liao = load_module(
        "liao_speed_benchmark",
        REPO_ROOT / "scripts" / "experiments" / "88_liao_realistic_p1_p2_rerank.py",
    )
    unified = load_module(
        "unified_speed_benchmark",
        REPO_ROOT / "scripts" / "experiments" / "102_unified_sky_formal_pipeline.py",
    )
    base.ROOTS[("SIS", "ET3")] = ET3_ROOT
    base.ROOTS[("PM", "ET3")] = ET3_ROOT
    liao.base.ROOTS[("SIS", "ET3")] = ET3_ROOT
    liao.base.ROOTS[("PM", "ET3")] = ET3_ROOT
    cfg = base.make_cfg("ET3", "noisy", seed_dir / "encoder")
    checkpoint = torch.load(seed_dir / "encoder" / "model.pt", map_location="cpu", weights_only=False)
    for key, value in checkpoint.get("config", {}).items():
        if hasattr(cfg, key) and key not in {"data_root", "out_dir"}:
            setattr(cfg, key, value)
    split_file = np.load(seed_dir / "split_indices.npz")
    splits: dict[str, dict[str, np.ndarray]] = {}
    for family in ("SIS", "PM", "SIS_U", "PM_U"):
        splits[family] = {split: split_file[f"{family}_{split}"] for split in ("train", "val", "test")}
    arrays = {family: base.FamilyArrays(family, ET3_ROOT, "noisy") for family in ("SIS", "PM")}
    test_ds = base.MixedEvaluationSet(arrays, splits, "test", cfg)
    test_gt = base.ground_truth(test_ds.meta)
    test_time = base.mixed_obs_frame("ET3", "test", splits, "time")
    test_raw = base.mixed_obs_frame("ET3", "test", splits, "raw")
    embedding = normalize_embeddings(np.load(seed_dir / "encoder" / "test_embeddings.npy"))
    ranks = pd.read_parquet(seed_dir / "query_ranks.parquet")
    weights = json.loads((seed_dir / "selected_weights.json").read_text(encoding="utf-8"))[
        "waveform_time_sky"
    ]["selected_weights"]
    model = None
    if with_raw:
        in_channels = base.data_mod.prepared_channel_count(arrays["SIS"].l1[0], cfg)
        model = base.build_model(cfg, in_channels=in_channels)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
    return {
        "base": base,
        "liao": liao,
        "unified": unified,
        "cfg": cfg,
        "arrays": arrays,
        "splits": splits,
        "test_ds": test_ds,
        "gt": test_gt,
        "time": test_time,
        "raw": test_raw,
        "embedding": embedding,
        "ranks": ranks,
        "weights": {key: float(value) for key, value in weights.items()},
        "model": model,
        "sky": pd.read_parquet(seed_dir / "test_observed_sky.parquet"),
    }


def benchmark_waveform(args: argparse.Namespace, out_root: Path) -> None:
    output = out_root / "waveform_encoding_summary.csv"
    if output.exists() and not args.force:
        print(f"waveform benchmark exists: {output}", flush=True)
        return
    context = load_et_context(args.seed_dir, with_raw=True)
    base, cfg, ds, model = context["base"], context["cfg"], context["test_ds"], context["model"]
    torch.set_num_threads(args.cpu_threads)
    rng = np.random.default_rng(20260714)
    sample_indices = np.sort(rng.choice(len(ds), size=min(args.encoding_events, len(ds)), replace=False))

    def preprocess_indices(indices: np.ndarray) -> np.ndarray:
        first = base.prepare_waveform(ds.waveforms[int(indices[0])], cfg, False)
        out = np.empty((len(indices), *first.shape), dtype=np.float32)
        out[0] = first
        for row, index in enumerate(indices[1:], start=1):
            out[row] = base.prepare_waveform(ds.waveforms[int(index)], cfg, False)
        return out

    warm_indices = sample_indices[: min(32, len(sample_indices))]
    for _ in range(args.warmups):
        preprocess_indices(warm_indices)
    preprocessing_times, processed = timed_repetitions(
        lambda: preprocess_indices(sample_indices), 0, args.repeats
    )
    rows: list[dict[str, Any]] = []
    summary = timing_summary(preprocessing_times)
    rows.append(
        {
            "stage": "preprocessing",
            "device": "CPU",
            "batch_size": 1,
            "n_events_bench": len(sample_indices),
            "warmups": args.warmups,
            "repeats": args.repeats,
            **summary,
            "ms_per_event": summary["median_s"] * 1000.0 / len(sample_indices),
            "events_per_s": len(sample_indices) / summary["median_s"],
            "peak_gpu_memory_mb": 0.0,
        }
    )

    checkpoint_state = copy.deepcopy(model.state_dict())
    for device_name in ("cpu", "cuda"):
        if device_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(device_name)
        local_model = copy.deepcopy(model).to(device).eval()
        local_model.load_state_dict(checkpoint_state, strict=True)
        for batch_size in args.batch_sizes:
            n_target = min(len(processed), max(batch_size, args.batch1_events if batch_size == 1 else args.encoding_events))
            data = torch.from_numpy(np.ascontiguousarray(processed[:n_target]))

            @torch.no_grad()
            def infer() -> np.ndarray:
                chunks = []
                for start in range(0, len(data), batch_size):
                    x = data[start : start + batch_size].to(device)
                    amp = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with amp:
                        chunks.append(local_model(x).float().cpu())
                return torch.cat(chunks).numpy()

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            values, embedded = timed_repetitions(infer, args.warmups, args.repeats, device)
            peak = (
                torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
            )
            summary = timing_summary(values)
            rows.append(
                {
                    "stage": "model_inference",
                    "device": device.type.upper(),
                    "batch_size": int(batch_size),
                    "n_events_bench": int(n_target),
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    **summary,
                    "ms_per_event": summary["median_s"] * 1000.0 / n_target,
                    "events_per_s": n_target / summary["median_s"],
                    "peak_gpu_memory_mb": float(peak),
                }
            )
            norm_values, _ = timed_repetitions(
                lambda: embedded
                / np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12),
                args.warmups,
                args.repeats,
            )
            norm_summary = timing_summary(norm_values)
            rows.append(
                {
                    "stage": "l2_normalization",
                    "device": "CPU",
                    "batch_size": int(batch_size),
                    "n_events_bench": int(n_target),
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    **norm_summary,
                    "ms_per_event": norm_summary["median_s"] * 1000.0 / n_target,
                    "events_per_s": n_target / norm_summary["median_s"],
                    "peak_gpu_memory_mb": 0.0,
                }
            )
        del local_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(output, index=False)
    del processed, model, context
    gc.collect()


def benchmark_ann_scaling(args: argparse.Namespace, out_root: Path) -> None:
    output = out_root / "retrieval_scaling_raw.csv"
    if output.exists() and not args.force:
        print(f"retrieval scaling exists: {output}", flush=True)
        return
    base = normalize_embeddings(np.load(args.seed_dir / "encoder" / "test_embeddings.npy"))
    process = psutil.Process()
    rows: list[dict[str, Any]] = []
    for n in args.sizes:
        embedding, source = synthesize_embeddings(base, n, 20260714 + n)
        if n >= 1_000_000:
            warmups, repeats = args.million_warmups, args.million_repeats
        elif n >= 100_000:
            warmups, repeats = args.large_warmups, args.large_repeats
        else:
            warmups, repeats = args.warmups, args.repeats
        if n <= args.exact_max_n:
            for warm in range(warmups):
                exact_topk(embedding, max(args.ks))
            for repeat in range(repeats):
                rss_before = process.memory_info().rss
                _, matrix_s, topk_s = exact_topk(embedding, max(args.ks))
                start = time.perf_counter()
                single_score = embedding @ embedding[0]
                single_score[0] = -np.inf
                np.argpartition(-single_score, kth=max(args.ks) - 1)[: max(args.ks)]
                incremental_query_ms = (time.perf_counter() - start) * 1000.0
                rows.append(
                    {
                        "method": "exact_cosine",
                        "embedding_source": source,
                        "n_events": n,
                        "k": max(args.ks),
                        "ef_search": np.nan,
                        "repeat": repeat,
                        "build_s": 0.0,
                        "query_s": matrix_s,
                        "postprocess_s": topk_s,
                        "total_s": matrix_s + topk_s,
                        "single_query_ms": (matrix_s + topk_s) * 1000.0 / n,
                        "incremental_query_ms": incremental_query_ms,
                        "rss_delta_mb": max(0, process.memory_info().rss - rss_before) / (1024**2),
                        "index_size_mb": n * n * 4 / (1024**2),
                        "threads": args.cpu_threads,
                        "warmups": warmups,
                        "repeats": repeats,
                        "full_catalog_query": True,
                    }
                )
        for warm in range(warmups):
            index = build_hnsw(embedding, args.cpu_threads, 9100 + warm)
            active_ef = [args.scaling_ef] if n >= 100_000 else args.ef_search
            index.set_ef(max(max(active_ef), max(args.ks) + 1))
            index.knn_query(embedding[: min(256, n)], k=max(args.ks) + 1, num_threads=args.cpu_threads)
            del index
        for repeat in range(repeats):
            rss_before = process.memory_info().rss
            start = time.perf_counter()
            index = build_hnsw(embedding, args.cpu_threads, 9200 + repeat)
            build_s = time.perf_counter() - start
            rss_after_build = process.memory_info().rss
            index_size_mb = np.nan
            if repeat == 0:
                with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as handle:
                    temp_path = Path(handle.name)
                index.save_index(str(temp_path))
                index_size_mb = temp_path.stat().st_size / (1024**2)
                temp_path.unlink(missing_ok=True)
            query_ids = np.arange(n, dtype=np.int32)
            active_ef = [args.scaling_ef] if n >= 100_000 else args.ef_search
            for ef in active_ef:
                for k in args.ks:
                    index.set_ef(max(int(ef), int(k) + 1))
                    start = time.perf_counter()
                    labels, _ = index.knn_query(embedding, k=int(k) + 1, num_threads=args.cpu_threads)
                    query_s = time.perf_counter() - start
                    start = time.perf_counter()
                    index.knn_query(embedding[:1], k=int(k) + 1, num_threads=1)
                    incremental_query_ms = (time.perf_counter() - start) * 1000.0
                    start = time.perf_counter()
                    remove_self(labels, query_ids, int(k))
                    post_s = time.perf_counter() - start
                    rows.append(
                        {
                            "method": "HNSW",
                            "embedding_source": source,
                            "n_events": n,
                            "k": int(k),
                            "ef_search": int(ef),
                            "repeat": repeat,
                            "build_s": build_s,
                            "query_s": query_s,
                            "postprocess_s": post_s,
                            "total_s": build_s + query_s + post_s,
                            "single_query_ms": query_s * 1000.0 / n,
                            "incremental_query_ms": incremental_query_ms,
                            "rss_delta_mb": max(0, rss_after_build - rss_before) / (1024**2),
                            "index_size_mb": index_size_mb,
                            "threads": args.cpu_threads,
                            "warmups": warmups,
                            "repeats": repeats,
                            "full_catalog_query": True,
                        }
                    )
                    del labels
            del index
            gc.collect()
        pd.DataFrame(rows).to_csv(output, index=False)
        del embedding
        gc.collect()


def build_mainline_components(context: dict[str, Any], seed_dir: Path) -> dict[str, Any]:
    liao, unified = context["liao"], context["unified"]
    seed_summary = json.loads((seed_dir / "seed_summary.json").read_text(encoding="utf-8"))
    prior = liao.fit_time_lr_from_liao(
        "ET3", context["time"], context["gt"], seed=int(seed_summary["time_prior_seed"])
    )
    time_raw = liao.time_lr_score_matrix(context["time"], prior)
    sky_raw = importlib.import_module("scripts.sky.sky_posterior_overlap").gaussian_log_cosine_overlap_matrix(
        context["sky"], chunk_rows=64, diagonal=0.0
    )
    waveform_raw = np.load(seed_dir / "encoder" / "test_scores.npy", mmap_mode="r")
    components = {
        "waveform": unified.row_z_zero_diag(waveform_raw),
        "time": unified.row_z_zero_diag(time_raw),
        "sky": unified.row_z_zero_diag(sky_raw),
    }
    final = unified.score_matrix_from_components(components, context["weights"])
    return {"prior": prior, "raw_time": time_raw, "raw_sky": sky_raw, "components": components, "final": final}


def retrieval_metrics_from_candidates(candidates: np.ndarray, final_score: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    values = final_score[np.arange(len(candidates))[:, None], candidates]
    order = np.argsort(-values, axis=1)
    ranked = np.take_along_axis(candidates, order, axis=1)
    valid = np.where(gt >= 0)[0]
    partner = gt[valid]
    hits = ranked[valid] == partner[:, None]
    return {
        "r_at_1": float(np.mean(np.any(hits[:, :1], axis=1))),
        "r_at_10": float(np.mean(np.any(hits[:, : min(10, ranked.shape[1])], axis=1))),
    }


def benchmark_et3_accuracy(args: argparse.Namespace, out_root: Path) -> None:
    output = out_root / "et3_ann_accuracy_pareto.csv"
    if output.exists() and not args.force:
        print(f"ET3 accuracy benchmark exists: {output}", flush=True)
        return
    context = load_et_context(args.seed_dir, with_raw=False)
    embedding, gt = context["embedding"], context["gt"]
    exact_waveform, exact_matrix_s, exact_topk_s = exact_topk(embedding, max(args.ks))
    component = build_mainline_components(context, args.seed_dir)
    exact_final_candidates = score_matrix_topk(component["final"], max(args.ks))
    exact_final_metrics = retrieval_metrics_from_candidates(
        exact_final_candidates, component["final"], gt
    )
    index = build_hnsw(embedding, args.cpu_threads, 20260714)
    rows: list[dict[str, Any]] = []
    query_ids = np.arange(len(embedding), dtype=np.int32)
    for ef in args.ef_search:
        for k in args.ks:
            index.set_ef(max(int(ef), int(k) + 1))
            for _ in range(args.warmups):
                index.knn_query(embedding, k=int(k) + 1, num_threads=args.cpu_threads)
            for repeat in range(args.repeats):
                start = time.perf_counter()
                labels, _ = index.knn_query(embedding, k=int(k) + 1, num_threads=args.cpu_threads)
                query_s = time.perf_counter() - start
                candidates = remove_self(labels, query_ids, int(k))
                start = time.perf_counter()
                final_metrics = retrieval_metrics_from_candidates(candidates, component["final"], gt)
                rerank_s = time.perf_counter() - start
                fidelity = float(
                    np.mean(
                        [
                            len(set(a[:10]) & set(b[:10])) / 10.0
                            for a, b in zip(candidates, exact_waveform)
                        ]
                    )
                )
                valid = np.where(gt >= 0)[0]
                companion = float(np.mean([gt[q] in candidates[q] for q in valid]))
                rows.append(
                    {
                        "n_events": len(embedding),
                        "k": int(k),
                        "ef_search": int(ef),
                        "repeat": repeat,
                        "hnsw_query_s": query_s,
                        "physical_rerank_s": rerank_s,
                        "ann_top10_fidelity": fidelity,
                        "candidate_companion_recall": companion,
                        "final_r_at_1": final_metrics["r_at_1"],
                        "final_r_at_10": final_metrics["r_at_10"],
                        "exact_final_r_at_1": exact_final_metrics["r_at_1"],
                        "exact_final_r_at_10": exact_final_metrics["r_at_10"],
                        "exact_waveform_matrix_s": exact_matrix_s,
                        "exact_waveform_topk_s": exact_topk_s,
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False)
    del component, context, index
    gc.collect()


def benchmark_native_pipeline(args: argparse.Namespace, out_root: Path) -> None:
    """Measure a native 9,000-event sparse deployment from strain to top candidates."""
    output = out_root / "native_et3_end_to_end_stage_times.csv"
    summary_output = out_root / "native_et3_end_to_end_summary.json"
    if output.exists() and not args.force:
        print(f"native ET3 pipeline benchmark exists: {output}", flush=True)
        return
    context = load_et_context(args.seed_dir, with_raw=True)
    base, cfg, ds = context["base"], context["cfg"], context["test_ds"]
    model = context["model"].cuda().eval()
    n = len(ds)
    k = max(args.ks)
    rows: list[dict[str, Any]] = []

    def measure(stage: str, fn: Callable[[], Any], device: str | None = None) -> Any:
        values, last = timed_repetitions(
            fn, args.pipeline_warmups, args.pipeline_repeats, device
        )
        for repeat, value in enumerate(values):
            rows.append(
                {
                    "stage": stage,
                    "repeat": repeat,
                    "seconds": value,
                    "n_events": n,
                    "k": k,
                    "ef_search": args.scaling_ef,
                    "pipeline_scope": "native_ET3_sparse_mainline",
                }
            )
        return last

    def preprocess_catalog() -> np.ndarray:
        first = base.prepare_waveform(ds.waveforms[0], cfg, False)
        prepared = np.empty((n, *first.shape), dtype=np.float32)
        prepared[0] = first
        for row in range(1, n):
            prepared[row] = base.prepare_waveform(ds.waveforms[row], cfg, False)
        return prepared

    prepared = measure("waveform_preprocessing", preprocess_catalog)

    @torch.no_grad()
    def infer_catalog() -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, n, args.pipeline_batch_size):
            x = torch.from_numpy(prepared[start : start + args.pipeline_batch_size]).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunks.append(model(x).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    raw_embedding = measure("waveform_gpu_inference", infer_catalog, "cuda")
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
    embedding = measure("embedding_l2_normalization", lambda: normalize_embeddings(raw_embedding))
    index = measure(
        "hnsw_index_build",
        lambda: build_hnsw(embedding, args.cpu_threads, 20260714),
    )
    index.set_ef(max(args.scaling_ef, k + 1))
    query_ids = np.arange(n, dtype=np.int32)

    def query_catalog() -> tuple[np.ndarray, np.ndarray]:
        labels, distances = index.knn_query(
            embedding, k=k + 1, num_threads=args.cpu_threads
        )
        return remove_self_with_scores(labels, distances, query_ids, k)

    candidates, waveform_score_matrix = measure("hnsw_full_catalog_query", query_catalog)
    ii = np.repeat(query_ids, k)
    jj = candidates.reshape(-1)
    time_values = context["time"]["trigger_time_obs"].to_numpy(dtype=np.float64)
    sky = context["sky"]
    ra = sky["ra_obs"].to_numpy(dtype=np.float64)
    dec = sky["dec_obs"].to_numpy(dtype=np.float64)
    sigma = sky["sky_sigma_rad"].to_numpy(dtype=np.float64)
    seed_summary = json.loads((args.seed_dir / "seed_summary.json").read_text(encoding="utf-8"))
    prior = context["liao"].fit_time_lr_from_liao(
        "ET3", context["time"], context["gt"], seed=int(seed_summary["time_prior_seed"])
    )

    waveform_score = waveform_score_matrix.reshape(-1)
    rows.append(
        {
            "stage": "candidate_waveform_score_reused_from_hnsw",
            "repeat": 0,
            "seconds": 0.0,
            "n_events": n,
            "k": k,
            "ef_search": args.scaling_ef,
            "pipeline_scope": "native_ET3_sparse_mainline",
        }
    )

    def score_time() -> np.ndarray:
        delta = np.abs(time_values[ii] - time_values[jj]) / SECONDS_PER_DAY
        bins = np.searchsorted(
            prior["edges"], np.log10(np.maximum(delta, 1e-6)), side="right"
        ) - 1
        return prior["lr"][np.clip(bins, 0, len(prior["lr"]) - 1)].astype(np.float32)

    time_score = measure("candidate_time_delay_lr", score_time)
    sky_score = measure(
        "candidate_gaussian_posterior_overlap",
        lambda: gaussian_log_cosine_overlap_pairs(
            ra[ii], dec[ii], sigma[ii], ra[jj], dec[jj], sigma[jj]
        )[0].astype(np.float32),
    )

    def standardize() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = []
        for value in (waveform_score, time_score, sky_score):
            matrix = value.reshape(n, k)
            result.append(
                (
                    (matrix - matrix.mean(axis=1, keepdims=True))
                    / np.maximum(matrix.std(axis=1, keepdims=True), 1e-8)
                ).astype(np.float32)
            )
        return tuple(result)  # type: ignore[return-value]

    standardized = measure("candidate_row_standardization", standardize)

    def fuse_sort() -> np.ndarray:
        final = (
            context["weights"]["waveform"] * standardized[0]
            + context["weights"]["time"] * standardized[1]
            + context["weights"]["sky"] * standardized[2]
        )
        return np.argsort(-final, axis=1)[:, :10]

    local_order = measure("weighted_fusion_and_top10_sort", fuse_sort)
    ranked = np.take_along_axis(candidates, local_order, axis=1)
    valid = np.where(context["gt"] >= 0)[0]
    hits = ranked[valid] == context["gt"][valid, None]
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    stage_medians = frame.groupby("stage")["seconds"].median().to_dict()
    write_json(
        summary_output,
        {
            "n_events": n,
            "n_lensed_queries": len(valid),
            "k": k,
            "ef_search": args.scaling_ef,
            "batch_size": args.pipeline_batch_size,
            "warmups": args.pipeline_warmups,
            "repeats": args.pipeline_repeats,
            "stage_median_seconds": stage_medians,
            "median_total_seconds_sum_of_stage_medians": float(sum(stage_medians.values())),
            "sparse_pipeline_r_at_1": float(np.mean(np.any(hits[:, :1], axis=1))),
            "sparse_pipeline_r_at_10": float(np.mean(np.any(hits[:, :10], axis=1))),
            "candidate_companion_recall": float(
                np.mean([context["gt"][q] in candidates[q] for q in valid])
            ),
            "peak_gpu_memory_mb": float(peak_gpu_mb),
            "note": "Native 9,000-event ET3 strain-level sparse deployment; stage medians are measured, not projected.",
        },
    )
    del context, model, prepared, raw_embedding, embedding, index
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_physics(args: argparse.Namespace, out_root: Path) -> None:
    output = out_root / "physical_scoring_raw.csv"
    if output.exists() and not args.force:
        print(f"physical benchmark exists: {output}", flush=True)
        return
    context = load_et_context(args.seed_dir, with_raw=False)
    component = build_mainline_components(context, args.seed_dir)
    embedding = context["embedding"]
    n, k = len(embedding), max(args.ks)
    index = build_hnsw(embedding, args.cpu_threads, 20260714)
    index.set_ef(max(args.ef_search))
    labels, _ = index.knn_query(embedding, k=k + 1, num_threads=args.cpu_threads)
    candidates = remove_self(labels, np.arange(n, dtype=np.int32), k)
    ii = np.repeat(np.arange(n, dtype=np.int32), k)
    jj = candidates.reshape(-1)
    time_values = context["time"]["trigger_time_obs"].to_numpy(dtype=np.float64)
    sky = context["sky"]
    prior = component["prior"]
    rows: list[dict[str, Any]] = []

    def measure(name: str, fn: Callable[[], Any], edges: int) -> Any:
        values, last = timed_repetitions(fn, args.warmups, args.repeats)
        for repeat, value in enumerate(values):
            rows.append(
                {
                    "kernel": name,
                    "repeat": repeat,
                    "n_edges": edges,
                    "seconds": value,
                    "microseconds_per_edge": value * 1e6 / edges,
                    "catalog": "ET3",
                }
            )
        return last

    def time_kernel() -> np.ndarray:
        delta = np.abs(time_values[ii] - time_values[jj]) / SECONDS_PER_DAY
        x = np.log10(np.maximum(delta, 1e-6))
        bins = np.searchsorted(prior["edges"], x, side="right") - 1
        return prior["lr"][np.clip(bins, 0, len(prior["lr"]) - 1)].astype(np.float32)

    time_score = measure("time_delay_lr", time_kernel, len(ii))
    sky_score = measure(
        "gaussian_posterior_overlap",
        lambda: gaussian_log_cosine_overlap_pairs(
            sky["ra_obs"].to_numpy()[ii],
            sky["dec_obs"].to_numpy()[ii],
            sky["sky_sigma_rad"].to_numpy()[ii],
            sky["ra_obs"].to_numpy()[jj],
            sky["dec_obs"].to_numpy()[jj],
            sky["sky_sigma_rad"].to_numpy()[jj],
        )[0],
        len(ii),
    )
    waveform_score = np.sum(embedding[ii] * embedding[jj], axis=1).astype(np.float32)

    def standardize() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blocks = []
        for value in (waveform_score, time_score, sky_score):
            matrix = value.reshape(n, k)
            blocks.append(((matrix - matrix.mean(axis=1, keepdims=True)) / np.maximum(matrix.std(axis=1, keepdims=True), 1e-8)).astype(np.float32))
        return tuple(blocks)  # type: ignore[return-value]

    standardized = measure("candidate_row_standardization", standardize, len(ii))

    def fuse_sort() -> np.ndarray:
        final = (
            context["weights"]["waveform"] * standardized[0]
            + context["weights"]["time"] * standardized[1]
            + context["weights"]["sky"] * standardized[2]
        )
        return np.argsort(-final, axis=1)[:, :10]

    measure("weighted_fusion_and_top10_sort", fuse_sort, len(ii))
    pd.DataFrame(rows).to_csv(output, index=False)
    del component, context, index
    gc.collect()


def benchmark_dense_physics(args: argparse.Namespace, out_root: Path) -> None:
    """Time the native N^2 physical-scoring reference at N=9,000."""
    output = out_root / "physical_scoring_dense_reference_raw.csv"
    if output.exists() and not args.force:
        print(f"dense physical benchmark exists: {output}", flush=True)
        return
    context = load_et_context(args.seed_dir, with_raw=False)
    liao, unified = context["liao"], context["unified"]
    seed_summary = json.loads((args.seed_dir / "seed_summary.json").read_text(encoding="utf-8"))
    prior = liao.fit_time_lr_from_liao(
        "ET3", context["time"], context["gt"], seed=int(seed_summary["time_prior_seed"])
    )
    n = len(context["embedding"])
    directed_edges = n * (n - 1)
    rows: list[dict[str, Any]] = []

    def measure(stage: str, fn: Callable[[], Any]) -> Any:
        values, last = timed_repetitions(
            fn, args.dense_warmups, args.dense_repeats
        )
        for repeat, value in enumerate(values):
            rows.append(
                {
                    "kernel": stage,
                    "repeat": repeat,
                    "n_edges": directed_edges,
                    "seconds": value,
                    "microseconds_per_edge": value * 1e6 / directed_edges,
                    "catalog": "ET3_dense_reference",
                }
            )
        return last

    time_raw = measure(
        "time_delay_lr_full_dense",
        lambda: liao.time_lr_score_matrix(context["time"], prior),
    )
    sky_module = importlib.import_module("scripts.sky.sky_posterior_overlap")
    sky_raw = measure(
        "gaussian_posterior_overlap_full_dense",
        lambda: sky_module.gaussian_log_cosine_overlap_matrix(
            context["sky"], chunk_rows=64, diagonal=0.0
        ),
    )
    waveform_raw = np.load(args.seed_dir / "encoder" / "test_scores.npy", mmap_mode="r")

    def standardize_full() -> dict[str, np.ndarray]:
        return {
            "waveform": unified.row_z_zero_diag(waveform_raw),
            "time": unified.row_z_zero_diag(time_raw),
            "sky": unified.row_z_zero_diag(sky_raw),
        }

    standardized = measure("three_channel_row_standardization_full_dense", standardize_full)
    final = measure(
        "weighted_fusion_full_dense",
        lambda: unified.score_matrix_from_components(standardized, context["weights"]),
    )
    measure("top10_selection_full_dense", lambda: score_matrix_topk(final, 10))
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(
        out_root / "physical_scoring_dense_reference_memory.json",
        {
            "n_events": n,
            "directed_nonself_edges": directed_edges,
            "single_float32_matrix_mb": n * n * 4 / (1024**2),
            "three_float32_matrices_mb": 3 * n * n * 4 / (1024**2),
            "warmups": args.dense_warmups,
            "repeats": args.dense_repeats,
        },
    )
    del context, time_raw, sky_raw, standardized, final
    gc.collect()


def benchmark_healpix(args: argparse.Namespace, out_root: Path) -> None:
    output = out_root / "healpix_benchmark_raw.csv"
    if output.exists() and not args.force:
        print(f"HEALPix benchmark exists: {output}", flush=True)
        return
    common = importlib.import_module("scripts.real_search.common")
    manifest = pd.read_csv(O4_RUN / "data" / "event_manifest.csv")
    events = manifest[
        manifest["include_in_primary_search"].astype(bool) & manifest["sky_map_available"].astype(bool)
    ].reset_index(drop=True)
    maps: list[np.ndarray] = []
    ingest_rows: list[dict[str, Any]] = []
    for index, row in events.iterrows():
        start = time.perf_counter()
        probability, meta = common.read_probability_map(row, target_nside=512)
        elapsed = time.perf_counter() - start
        maps.append(np.asarray(probability, dtype=np.float32))
        ingest_rows.append(
            {
                "kernel": "healpix_ingest_resample_normalize",
                "repeat": int(index),
                "n_edges": 1,
                "seconds": elapsed,
                "microseconds_per_edge": elapsed * 1e6,
                "catalog": "O4a_real_PE",
                "event_name": row.event_name,
                "source_nside": meta.get("source_nside"),
            }
        )
    stack = np.ascontiguousarray(np.vstack(maps), dtype=np.float32)
    norms = np.maximum(np.linalg.norm(stack, axis=1), 1e-30)
    normalized = stack / norms[:, None]
    rows = ingest_rows

    full_values, _ = timed_repetitions(
        lambda: normalized @ normalized.T,
        min(args.warmups, 2),
        min(args.repeats, args.healpix_repeats),
    )
    n_pairs = len(events) * (len(events) - 1) // 2
    for repeat, value in enumerate(full_values):
        rows.append(
            {
                "kernel": "healpix_full_cosine_matrix",
                "repeat": repeat,
                "n_edges": n_pairs,
                "seconds": value,
                "microseconds_per_edge": value * 1e6 / n_pairs,
                "catalog": "O4a_real_PE",
                "event_name": "",
                "source_nside": 512,
            }
        )
    candidate_count = min(50, len(events) - 1)
    query = normalized[0]
    candidates = normalized[1 : candidate_count + 1]
    sparse_values, _ = timed_repetitions(
        lambda: candidates @ query,
        args.warmups,
        args.repeats,
    )
    for repeat, value in enumerate(sparse_values):
        rows.append(
            {
                "kernel": "healpix_sparse_candidate_dot",
                "repeat": repeat,
                "n_edges": candidate_count,
                "seconds": value,
                "microseconds_per_edge": value * 1e6 / candidate_count,
                "catalog": "O4a_real_PE",
                "event_name": str(events.iloc[0].event_name),
                "source_nside": 512,
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(
        out_root / "healpix_memory_audit.json",
        {
            "n_events": len(events),
            "n_pixels": int(stack.shape[1]),
            "stack_memory_mb": stack.nbytes / (1024**2),
            "normalized_stack_memory_mb": normalized.nbytes / (1024**2),
            "primary_pairs": n_pairs,
            "source": "real O4a PE HDF5 skymap/data",
        },
    )


def benchmark_online_latency(args: argparse.Namespace, out_root: Path) -> None:
    output = out_root / "online_latency_raw.csv"
    if output.exists() and not args.force:
        print(f"online latency exists: {output}", flush=True)
        return
    context = load_et_context(args.seed_dir, with_raw=True)
    base, cfg, model = context["base"], context["cfg"], context["model"]
    embedding = context["embedding"]
    model = model.cuda().eval()
    index = build_hnsw(embedding, args.cpu_threads, 20260714)
    index.set_ef(max(args.ef_search))
    raw = context["test_ds"].waveforms[0]
    rows: list[dict[str, Any]] = []

    def record(stage: str, fn: Callable[[], Any], device: str | None = None) -> Any:
        values, last = timed_repetitions(fn, args.warmups, args.online_repeats, device)
        for repeat, value in enumerate(values):
            rows.append({"stage": stage, "repeat": repeat, "seconds": value, "milliseconds": value * 1000.0})
        return last

    prepared = record("waveform_preprocess", lambda: base.prepare_waveform(raw, cfg, False))

    @torch.no_grad()
    def infer_one() -> np.ndarray:
        x = torch.from_numpy(prepared[None]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x).float().cpu().numpy()

    query_embedding = normalize_embeddings(record("waveform_inference", infer_one, "cuda"))
    candidates = record(
        "hnsw_query_k200",
        lambda: index.knn_query(query_embedding, k=max(args.ks), num_threads=args.cpu_threads)[0],
    )
    rng = np.random.default_rng(20260714)
    waveform = rng.normal(size=max(args.ks)).astype(np.float32)
    timing = rng.normal(size=max(args.ks)).astype(np.float32)
    sky = rng.normal(size=max(args.ks)).astype(np.float32)

    def physical() -> np.ndarray:
        parts = []
        for value in (waveform, timing, sky):
            parts.append((value - value.mean()) / max(value.std(), 1e-8))
        score = (
            context["weights"]["waveform"] * parts[0]
            + context["weights"]["time"] * parts[1]
            + context["weights"]["sky"] * parts[2]
        )
        return np.argsort(-score)[:10]

    record("physical_score_fusion_sort_k200", physical)
    pd.DataFrame(rows).to_csv(output, index=False)
    del candidates, context, index, model
    gc.collect()
    torch.cuda.empty_cache()


def write_environment(args: argparse.Namespace, out_root: Path) -> None:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    cpu_model = subprocess.run(
        ["lscpu", "-J"], capture_output=True, text=True, check=False
    ).stdout
    try:
        lscpu_rows = json.loads(cpu_model).get("lscpu", [])
        cpu_model = next(
            row["data"].strip()
            for row in lscpu_rows
            if row.get("field", "").strip(":") == "Model name"
        )
    except (json.JSONDecodeError, StopIteration, KeyError):
        cpu_model = platform.processor()
    write_json(
        out_root / "benchmark_environment.json",
        {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": cpu_model,
            "logical_cpu_count": os.cpu_count(),
            "fixed_cpu_threads": args.cpu_threads,
            "gpu": gpu,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "hnswlib": "0.8.0",
            "warmups_default": args.warmups,
            "repeats_default": args.repeats,
            "large_scale_policy": {
                "n_ge_100k": [args.large_warmups, args.large_repeats],
                "n_ge_1m": [args.million_warmups, args.million_repeats],
                "fixed_ef_search": args.scaling_ef,
                "reason": "Full efSearch Pareto scan is measured on native ET3; large-N runs isolate scaling at the prespecified operating point.",
            },
            "seed_dir": args.seed_dir,
            "scope": "native strain-level ET3 at N=9000; N>9000 uses tiled/perturbed embeddings for scaling only",
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["waveform", "retrieval", "accuracy", "pipeline", "physics", "dense_physics", "healpix", "online"],
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000, 9000, 10000, 100000, 1000000])
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 200])
    parser.add_argument("--ef-search", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--scaling-ef", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--large-warmups", type=int, default=2)
    parser.add_argument("--large-repeats", type=int, default=10)
    parser.add_argument("--million-warmups", type=int, default=1)
    parser.add_argument("--million-repeats", type=int, default=3)
    parser.add_argument("--online-repeats", type=int, default=100)
    parser.add_argument("--healpix-repeats", type=int, default=5)
    parser.add_argument("--pipeline-warmups", type=int, default=1)
    parser.add_argument("--pipeline-repeats", type=int, default=5)
    parser.add_argument("--pipeline-batch-size", type=int, default=512)
    parser.add_argument("--dense-warmups", type=int, default=1)
    parser.add_argument("--dense-repeats", type=int, default=5)
    parser.add_argument("--encoding-events", type=int, default=1024)
    parser.add_argument("--batch1-events", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--exact-max-n", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        if "waveform" in args.sections:
            benchmark_waveform(args, args.out_root)
        if "retrieval" in args.sections:
            benchmark_ann_scaling(args, args.out_root)
        if "accuracy" in args.sections:
            benchmark_et3_accuracy(args, args.out_root)
        if "pipeline" in args.sections:
            benchmark_native_pipeline(args, args.out_root)
        if "physics" in args.sections:
            benchmark_physics(args, args.out_root)
        if "dense_physics" in args.sections:
            benchmark_dense_physics(args, args.out_root)
        if "healpix" in args.sections:
            benchmark_healpix(args, args.out_root)
        if "online" in args.sections:
            benchmark_online_latency(args, args.out_root)
    write_environment(args, args.out_root)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
