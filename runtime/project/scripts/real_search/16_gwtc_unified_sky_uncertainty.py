from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matchgw.config import MatchRunConfig  # noqa: E402
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices  # noqa: E402
from matchgw.matching import similarity_matrix  # noqa: E402
from matchgw.pipeline import embed_eval, run_train_eval  # noqa: E402
from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    across_seed_summary,
    bootstrap_system_ci,
    metric_rows,
    seed_everything,
    split_integrity_audit,
    write_json,
)
from scripts.sky.sky_posterior_overlap import gaussian_log_cosine_overlap_pairs  # noqa: E402


O3_SOURCE = REPO_ROOT / "runs" / "real_gwtc_lensing_search_20260625"
O4_SOURCE = REPO_ROOT / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"
CLEAN_SOURCE = O3_SOURCE / "data" / "real_noise_injections" / "matchroots" / "LIGO"
DEFAULT_OUT = REPO_ROOT / "results" / "mainline_uncertainty_20260713" / "gwtc"
DEFAULT_SEEDS = (202607111, 202607112, 202607113)
FAMILIES = ("SIS", "PM")
FULL_WINDOW = 98_304
MODEL_TAIL = 8_192
SAMPLE_RATE = 4096.0
SECONDS_PER_DAY = 86400.0
STRAIN_CACHES: dict[str, "LruStrainCache"] = {}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def prepare_seed_layout(seed_dir: Path, source_run: Path) -> None:
    for name in ("event_manifest.csv", "strain_gwosc_download_manifest.csv"):
        ensure_link(seed_dir / "data" / name, source_run / "data" / name)
    ensure_link(seed_dir / "data" / "real_strain", source_run / "data" / "real_strain")
    ensure_link(
        seed_dir / "features" / "real_pair_observable_features.parquet",
        source_run / "features" / "real_pair_observable_features.parquet",
    )
    inj = seed_dir / "data" / "real_noise_injections"
    inj.mkdir(parents=True, exist_ok=True)
    ensure_link(
        inj / "offsource_noise_segments.parquet",
        source_run / "data" / "real_noise_injections" / "offsource_noise_segments.parquet",
    )
    (seed_dir / "results").mkdir(parents=True, exist_ok=True)
    (seed_dir / "logs").mkdir(parents=True, exist_ok=True)
    (seed_dir / "waveform_gate").mkdir(parents=True, exist_ok=True)


def zscore_channelwise(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mean) * 1e-6, 1e-30)
    return ((x - mean) / np.maximum(std, floor)).astype(np.float32, copy=False)


class LruStrainCache:
    def __init__(self, source_run: Path, max_files: int = 512):
        self.source_run = source_run
        self.max_files = max_files
        self.cache: OrderedDict[str, tuple[np.ndarray, float, float]] = OrderedDict()

    def get(self, rel_path: str) -> tuple[np.ndarray, float, float]:
        if rel_path in self.cache:
            value = self.cache.pop(rel_path)
            self.cache[rel_path] = value
            return value
        path = self.source_run / rel_path
        with h5py.File(path, "r") as h5:
            strain = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
            start = float(np.asarray(h5["meta/GPSstart"][()]).item())
            duration = float(np.asarray(h5["meta/Duration"][()]).item())
        value = (np.nan_to_num(strain), start, duration)
        self.cache[rel_path] = value
        while len(self.cache) > self.max_files:
            self.cache.popitem(last=False)
        return value


def draw_full_noise(
    cache: LruStrainCache,
    segments: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    segment = segments.iloc[int(rng.integers(0, len(segments)))]
    latest = float(segment.segment_end) - FULL_WINDOW / SAMPLE_RATE
    start = float(rng.uniform(float(segment.segment_start), latest))
    channels = []
    for detector in ("H1", "L1"):
        data, gps_start, duration = cache.get(str(segment[f"{detector}_path"]))
        rate = len(data) / duration
        if abs(rate - SAMPLE_RATE) > 1e-3:
            raise ValueError(f"Unexpected {detector} sample rate {rate}")
        i0 = int(round((start - gps_start) * rate))
        i0 = max(0, min(i0, len(data) - FULL_WINDOW))
        channels.append(data[i0 : i0 + FULL_WINDOW])
    return np.stack(channels).astype(np.float32), {
        "noise_event": str(segment.event_name),
        "noise_segment_kind": str(segment.segment_kind),
        "noise_start_gps": start,
    }


def inject_and_crop(clean: np.ndarray, noise: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    clean_full = zscore_channelwise(clean)
    noise_full = zscore_channelwise(noise)
    # The model next applies a linear bandpass that removes DC, followed by a
    # channelwise z-score. Therefore the full-window affine normalization of the
    # mixture cancels exactly; retaining the unnormalized mixed tail is equivalent
    # at model input while avoiding an unnecessary second 24 s pass.
    mixed_tail = (noise_full[..., -MODEL_TAIL:] + float(scale) * clean_full[..., -MODEL_TAIL:]).astype(np.float32)
    return clean_full[..., -MODEL_TAIL:], mixed_tail


def materialize_compact_dataset(
    seed_dir: Path,
    source_run: Path,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    marker = seed_dir / "data" / "real_noise_injections" / "compact_dataset_summary.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    segments = pd.read_parquet(source_run / "data" / "real_noise_injections" / "offsource_noise_segments.parquet")
    cache_key = str(source_run.resolve())
    cache = STRAIN_CACHES.setdefault(cache_key, LruStrainCache(source_run))
    root = seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    metadata: list[dict[str, Any]] = []
    for family in FAMILIES:
        source = CLEAN_SOURCE / f"{family}_data_0222"
        clean1 = np.load(source / f"{family}_h_strain_1.npy", mmap_mode="r")[:samples]
        clean2 = np.load(source / f"{family}_h_strain_2.npy", mmap_mode="r")[:samples]
        out = root / f"{family}_data_0222"
        out.mkdir(parents=True, exist_ok=True)
        h1 = np.empty((samples, 2, MODEL_TAIL), dtype=np.float32)
        h2 = np.empty_like(h1)
        d1 = np.empty_like(h1)
        d2 = np.empty_like(h1)
        for idx in range(samples):
            noise1, info1 = draw_full_noise(cache, segments, rng)
            noise2, info2 = draw_full_noise(cache, segments, rng)
            scale1 = float(rng.lognormal(mean=-0.05, sigma=0.35))
            scale2 = float(rng.lognormal(mean=-0.05, sigma=0.35))
            h1[idx], d1[idx] = inject_and_crop(clean1[idx], noise1, scale1)
            h2[idx], d2[idx] = inject_and_crop(clean2[idx], noise2, scale2)
            metadata.append(
                {
                    "family": family,
                    "sample_index": idx,
                    "noise_event_a": info1["noise_event"],
                    "noise_event_b": info2["noise_event"],
                    "noise_start_gps_a": info1["noise_start_gps"],
                    "noise_start_gps_b": info2["noise_start_gps"],
                    "signal_scale_a": scale1,
                    "signal_scale_b": scale2,
                }
            )
        np.save(out / f"{family}_h_strain_1.npy", h1)
        np.save(out / f"{family}_h_strain_2.npy", h2)
        np.save(out / f"{family}_data_strain_1.npy", d1)
        np.save(out / f"{family}_data_strain_2.npy", d2)
        np.save(out / f"{family}_optimal_SNR_network_1.npy", np.ones(samples, dtype=np.float32))
        np.save(out / f"{family}_optimal_SNR_network_2.npy", np.ones(samples, dtype=np.float32))
        del h1, h2, d1, d2
        gc.collect()

    source = CLEAN_SOURCE / "Unlensed_data_0222" / "unlensed_h_strain.npy"
    clean = np.load(source, mmap_mode="r")[:samples]
    out = root / "Unlensed_data_0222"
    out.mkdir(parents=True, exist_ok=True)
    h = np.empty((samples, 2, MODEL_TAIL), dtype=np.float32)
    d = np.empty_like(h)
    for idx in range(samples):
        noise, info = draw_full_noise(cache, segments, rng)
        scale = float(rng.lognormal(mean=-0.05, sigma=0.35))
        h[idx], d[idx] = inject_and_crop(clean[idx], noise, scale)
        metadata.append(
            {
                "family": "unlensed",
                "sample_index": idx,
                "noise_event_a": info["noise_event"],
                "noise_event_b": "",
                "noise_start_gps_a": info["noise_start_gps"],
                "noise_start_gps_b": np.nan,
                "signal_scale_a": scale,
                "signal_scale_b": np.nan,
            }
        )
    np.save(out / "unlensed_h_strain.npy", h)
    np.save(out / "unlensed_data_strain.npy", d)
    np.save(out / "unlensed_optimal_SNR_network.npy", np.ones(samples, dtype=np.float32))
    pd.DataFrame(metadata).to_parquet(
        seed_dir / "data" / "real_noise_injections" / "compact_injection_metadata.parquet", index=False
    )
    payload = {
        "seed": int(seed),
        "samples_per_family": int(samples),
        "source_noise_run": str(source_run),
        "clean_signal_source": str(CLEAN_SOURCE),
        "full_injection_window_samples": FULL_WINDOW,
        "cached_model_tail_samples": MODEL_TAIL,
        "equivalence_note": "Normalization and injection are computed on the full 24 s window; only the exact tail consumed by pad_or_trim(target_len=8192) is retained.",
        "paired_noise_segments": int(len(segments)),
        "metadata_rows": int(len(metadata)),
    }
    write_json(marker, payload)
    return payload


def train_family(seed_dir: Path, family: str, seed: int, samples: int, epochs: int) -> dict[str, Any]:
    out = seed_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean"
    summary = out / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    cfg = MatchRunConfig(
        data_root=seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        model_type=family,
        data_mode="noisy",
        out_dir=out,
        backbone="inceptiontime",
        lensed_limit=samples,
        unlensed_limit=samples,
        epochs=epochs,
        batch_size=min(64, max(8, int(round(1.4 * samples)))),
        eval_batch_size=256,
        target_len=8192,
        stride=2,
        preprocess="bandpass",
        bandpass_low=40,
        bandpass_high=580,
        candidate_topk=10,
        seed=seed + (0 if family == "SIS" else 1000),
        num_workers=0,
        pin_memory=False,
        amp=False,
        export_candidates=False,
    )
    seed_everything(cfg.seed)
    return run_train_eval(cfg, cpu=False)


def build_split_pair_table(
    deploy,
    seed_dir: Path,
    family: str,
    epochs: int,
    samples: int,
    split: str,
    delay_prior: np.ndarray,
    observable_seed: int,
) -> pd.DataFrame:
    checkpoint = seed_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "model.pt"
    model, cfg = deploy.load_model_from_checkpoint(
        checkpoint,
        seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        samples,
        False,
    )
    cfg.model_type = family
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits["lensed"][split], splits["unlensed"][split], cfg)
    embeddings = embed_eval(model, ds, cfg, cpu=False)
    waveform = similarity_matrix(embeddings)
    gt = ground_truth_partner(ds.meta)
    events = deploy.synthetic_validation_observables(ds.meta, family, delay_prior, seed=observable_seed)
    ii, jj = np.triu_indices(len(events), k=1)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    delta_days = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    bins = deploy.log_bins_for(delta_days, delay_prior, min_floor=1e-4, n_bins=80)
    time_score = deploy.empirical_hist_lr(delta_days, delay_prior, delta_days, bins)
    sky_score, theta, norm_sep = gaussian_log_cosine_overlap_pairs(
        events["ra_obs"].to_numpy(dtype=np.float64)[ii],
        events["dec_obs"].to_numpy(dtype=np.float64)[ii],
        events["sky_sigma_rad"].to_numpy(dtype=np.float64)[ii],
        events["ra_obs"].to_numpy(dtype=np.float64)[jj],
        events["dec_obs"].to_numpy(dtype=np.float64)[jj],
        events["sky_sigma_rad"].to_numpy(dtype=np.float64)[jj],
    )
    return pd.DataFrame(
        {
            "family": family,
            "split": split,
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "is_true_pair": (gt[ii] == jj).astype(np.int8),
            "waveform_score": waveform[ii, jj].astype(np.float32),
            "time_score": time_score.astype(np.float32),
            "sky_score": sky_score.astype(np.float32),
            "delta_t_days": delta_days,
            "sky_ang_sep_rad": theta.astype(np.float32),
            "sky_norm_sep": norm_sep.astype(np.float32),
            "event_count": int(len(events)),
        }
    )


def query_ranks_from_pairs(deploy, pairs: pd.DataFrame, weights: dict[str, float], deployment: str, seed: int, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, frame in pairs.groupby("family", sort=True):
        n = int(frame["event_count"].iloc[0])
        label = np.zeros((n, n), dtype=bool)
        ii = frame["idx_i"].to_numpy(dtype=np.int32)
        jj = frame["idx_j"].to_numpy(dtype=np.int32)
        truth = frame["is_true_pair"].to_numpy(dtype=bool)
        label[ii, jj] = truth
        label[jj, ii] = truth
        score = np.zeros((n, n), dtype=np.float32)
        for channel in ("waveform", "time", "sky"):
            score += float(weights.get(channel, 0.0)) * deploy.row_z_neutral(
                deploy.matrix_from_pairs(frame, n, f"{channel}_score")
            )
        np.fill_diagonal(score, -np.inf)
        for query in np.where(label.any(axis=1))[0]:
            partner = int(np.flatnonzero(label[query])[0])
            rank = int(1 + np.sum(score[query] > score[query, partner]))
            system = f"{family}:{min(query, partner)}-{max(query, partner)}"
            rows.append(
                {
                    "deployment": deployment,
                    "seed": int(seed),
                    "method": method,
                    "query_index": int(query),
                    "partner_index": partner,
                    "family": str(family).upper(),
                    "system_id": system,
                    "query_tag": "query",
                    "query_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def embed_real_events(seed_dir: Path, epochs: int) -> None:
    summary = seed_dir / "features" / "real_waveform_embedding_summary.json"
    if summary.exists():
        return
    embed = load_module(f"embed_real_{seed_dir.name}", REPO_ROOT / "scripts" / "real_search" / "06_embed_real_gwtc_events.py")
    old_argv = sys.argv
    try:
        sys.argv = ["06_embed_real_gwtc_events.py", "--run-dir", str(seed_dir), "--epochs", str(epochs)]
        embed.main()
    finally:
        sys.argv = old_argv


def real_catalog_scores(
    deploy,
    unified,
    seed_dir: Path,
    source_run: Path,
    weights: dict[str, float],
    baseline_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observable = pd.read_parquet(source_run / "features" / "real_pair_observable_features.parquet")
    waveform = pd.read_parquet(seed_dir / "features" / "real_waveform_similarity.parquet")
    primary = unified.primary_manifest(seed_dir)
    real = observable.merge(
        waveform[["idx_i", "idx_j", "waveform_available", "waveform_score", "waveform_score_sis", "waveform_score_pm"]],
        on=["idx_i", "idx_j"],
        how="left",
    )
    object_class = (
        primary["object_class"].fillna("unknown").astype(str).to_numpy()
        if "object_class" in primary.columns
        else np.asarray(["BBH"] * len(primary), dtype=object)
    )
    event_ood = (
        primary["is_ood_for_bbh_encoder"].fillna(False).astype(bool).to_numpy()
        if "is_ood_for_bbh_encoder" in primary.columns
        else np.zeros(len(primary), dtype=bool)
    )
    ii = real["idx_i"].to_numpy(dtype=np.int32)
    jj = real["idx_j"].to_numpy(dtype=np.int32)
    real["object_class_i"] = object_class[ii]
    real["object_class_j"] = object_class[jj]
    real["pair_has_ood"] = event_ood[ii] | event_ood[jj]
    real["waveform_available"] = real["waveform_available"].fillna(False).astype(bool)
    real["full_waveform_scored"] = real["waveform_available"] & (~real["pair_has_ood"])
    real["raw_waveform_score"] = real["waveform_score"]
    real.loc[~real["full_waveform_scored"], "waveform_score"] = np.nan
    selected = deploy.add_real_scores(real, weights, "unified_waveform_time_sky")
    baseline = deploy.add_real_scores(real, baseline_weights, "unified_time_sky")
    selected.to_parquet(seed_dir / "results" / "real_pair_scores_unified_waveform_time_sky.parquet", index=False)
    baseline.to_parquet(seed_dir / "results" / "real_pair_scores_unified_time_sky.parquet", index=False)
    deploy.shortlist(selected, 100).to_csv(seed_dir / "results" / "candidate_shortlist_unified_waveform_time_sky.csv", index=False)
    deploy.shortlist(baseline, 100).to_csv(seed_dir / "results" / "candidate_shortlist_unified_time_sky.csv", index=False)
    wf = real.loc[real["full_waveform_scored"], "raw_waveform_score"].dropna()
    audit = {
        "n_events": int(len(primary)),
        "n_pairs": int(len(selected)),
        "n_full_waveform_scored_pairs": int(selected["full_waveform_scored"].sum()),
        "waveform_score_min": float(wf.min()) if len(wf) else None,
        "waveform_score_max": float(wf.max()) if len(wf) else None,
        "waveform_score_std": float(wf.std()) if len(wf) else None,
        "waveform_score_unique": int(wf.nunique()) if len(wf) else 0,
    }
    return selected, baseline, audit


def run_seed(
    deployment: str,
    source_run: Path,
    seed: int,
    out_root: Path,
    samples: int,
    epochs: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    seed_dir = out_root / deployment.lower() / f"seed_{seed}"
    complete = seed_dir / "seed_summary.json"
    if complete.exists():
        print(f"{deployment} seed {seed}: complete, reusing", flush=True)
        return json.loads(complete.read_text(encoding="utf-8"))
    started = time.perf_counter()
    prepare_seed_layout(seed_dir, source_run)
    dataset = materialize_compact_dataset(seed_dir, source_run, seed + 100, samples)
    split_sets: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        split_cfg = MatchRunConfig(
            data_root=seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
            model_type=family,
            data_mode="noisy",
            lensed_limit=samples,
            unlensed_limit=samples,
            seed=seed + (0 if family == "SIS" else 1000),
        )
        parts = split_indices(samples, samples, split_cfg)
        split_sets[f"{family}_lensed"] = parts["lensed"]
        split_sets[f"{family}_unlensed"] = parts["unlensed"]
    split_audit = split_integrity_audit(split_sets)
    if not split_audit["all_disjoint"]:
        raise RuntimeError(f"{deployment} train/validation/test system split leakage detected")
    write_json(seed_dir / "results" / "split_integrity_audit.json", split_audit)
    for family in FAMILIES:
        print(f"{deployment} seed {seed}: train {family}", flush=True)
        train_family(seed_dir, family, seed, samples, epochs)
    deploy = load_module(
        f"deploy_uncertainty_{deployment}_{seed}",
        REPO_ROOT / "scripts" / "real_search" / "15_gwtc34_real_deployment.py",
    )
    unified = load_module(
        f"unified_uncertainty_{deployment}_{seed}",
        REPO_ROOT / "scripts" / "experiments" / "102_unified_sky_formal_pipeline.py",
    )
    delay_prior = deploy.liao_delay_samples()
    val = pd.concat(
        [
            build_split_pair_table(deploy, seed_dir, family, epochs, samples, "val", delay_prior, seed + 2000)
            for family in FAMILIES
        ],
        ignore_index=True,
    )
    test = pd.concat(
        [
            build_split_pair_table(deploy, seed_dir, family, epochs, samples, "test", delay_prior, seed + 3000)
            for family in FAMILIES
        ],
        ignore_index=True,
    )
    val.to_parquet(seed_dir / "results" / "fusion_validation_pairs.parquet", index=False)
    test.to_parquet(seed_dir / "results" / "fusion_heldout_test_pairs.parquet", index=False)
    weights, grid = deploy.select_weights(val, force_waveform_zero=False, require_all_positive=False)
    baseline_weights, baseline_grid = deploy.select_weights(val, force_waveform_zero=True, require_time_sky_positive=False)
    grid.to_csv(seed_dir / "results" / "fusion_weight_grid.csv", index=False)
    baseline_grid.to_csv(seed_dir / "results" / "fusion_weight_grid_time_sky.csv", index=False)
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_sky": baseline_weights,
        "waveform_time_sky": weights,
    }
    query = pd.concat(
        [query_ranks_from_pairs(deploy, test, method_weights, deployment, seed, method) for method, method_weights in methods.items()],
        ignore_index=True,
    )
    query.to_parquet(seed_dir / "results" / "heldout_test_query_ranks.parquet", index=False)
    point = metric_rows(query)
    point.to_csv(seed_dir / "results" / "heldout_test_retrieval_metrics.csv", index=False)
    bootstrap = bootstrap_system_ci(query, draws=bootstrap_draws, seed=seed + 4000)
    bootstrap.to_csv(seed_dir / "results" / "heldout_test_bootstrap_95ci.csv", index=False)
    write_json(
        seed_dir / "results" / "selected_weights.json",
        {
            "selection_split": "validation",
            "evaluation_split": "held-out test",
            "waveform_time_sky": weights,
            "time_sky": baseline_weights,
            "validation_best": grid.iloc[0].to_dict(),
            "heldout_test_metrics": point.to_dict(orient="records"),
        },
    )

    embed_real_events(seed_dir, epochs)
    selected, baseline, waveform_audit = real_catalog_scores(
        deploy, unified, seed_dir, source_run, weights, baseline_weights
    )
    top = selected.iloc[0]
    historical = unified.safe_rank(selected, "GW170104", "GW170814") if deployment == "GWTC3" else {"found": False}
    summary = {
        "deployment": deployment,
        "seed": int(seed),
        "status": "complete",
        "samples_per_family": int(samples),
        "epochs": int(epochs),
        "noise_realization_seed": int(seed + 100),
        "validation_observable_seed": int(seed + 2000),
        "test_observable_seed": int(seed + 3000),
        "selected_weights": weights,
        "time_sky_weights": baseline_weights,
        "validation_metrics": grid.iloc[0].to_dict(),
        "heldout_test_metrics": point.to_dict(orient="records"),
        "dataset": dataset,
        "real_waveform_audit": waveform_audit,
        "top_pair": {
            "event_i": str(top.event_i),
            "event_j": str(top.event_j),
            "rank": int(top["rank"]),
            "final_score": float(top.final_score),
            "waveform_available": bool(top.waveform_available),
            "full_waveform_scored": bool(top.full_waveform_scored),
            "pair_has_ood": bool(top.pair_has_ood),
        },
        "gw170104_gw170814": historical,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(complete, summary)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_deployment(out_root: Path, deployment: str, seeds: list[int]) -> None:
    root = out_root / deployment.lower()
    query = pd.concat(
        [pd.read_parquet(root / f"seed_{seed}" / "results" / "heldout_test_query_ranks.parquet") for seed in seeds],
        ignore_index=True,
    )
    query.to_parquet(root / "heldout_test_query_ranks_all_seeds.parquet", index=False)
    per_seed = metric_rows(query)
    per_seed.to_csv(root / "heldout_test_retrieval_metrics_per_seed.csv", index=False)
    across_seed_summary(per_seed).to_csv(root / "heldout_test_retrieval_metrics_across_seed_summary.csv", index=False)
    boot = pd.concat(
        [pd.read_csv(root / f"seed_{seed}" / "results" / "heldout_test_bootstrap_95ci.csv") for seed in seeds],
        ignore_index=True,
    )
    boot.to_csv(root / "heldout_test_bootstrap_95ci_per_seed.csv", index=False)

    ranking = []
    for seed in seeds:
        scores = pd.read_parquet(root / f"seed_{seed}" / "results" / "real_pair_scores_unified_waveform_time_sky.parquet")
        scores = scores[["event_i", "event_j", "rank", "final_score", "waveform_available", "full_waveform_scored", "pair_has_ood"]].copy()
        scores["seed"] = int(seed)
        scores["pair_key"] = scores.apply(lambda r: "--".join(sorted((str(r.event_i), str(r.event_j)))), axis=1)
        ranking.append(scores)
    all_ranks = pd.concat(ranking, ignore_index=True)
    all_ranks.to_parquet(root / "real_candidate_ranks_all_seeds.parquet", index=False)
    stability = (
        all_ranks.groupby("pair_key", as_index=False)
        .agg(
            event_i=("event_i", "first"),
            event_j=("event_j", "first"),
            median_rank=("rank", "median"),
            q25_rank=("rank", lambda x: float(np.quantile(x, 0.25))),
            q75_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            top10_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 10))),
            top20_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 20))),
            full_waveform_scored=("full_waveform_scored", "all"),
            pair_has_ood=("pair_has_ood", "any"),
        )
        .sort_values(["median_rank", "q75_rank", "min_rank"])
        .reset_index(drop=True)
    )
    stability.to_csv(root / "real_candidate_rank_stability.csv", index=False)
    write_json(
        root / "uncertainty_summary.json",
        {
            "deployment": deployment,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "fusion_weight_selection": "validation-only for each seed",
            "split_policy": "independent source-system split per seed; both images of each injected doublet remain in the same split",
            "reported_retrieval_split": "held-out real-noise synthetic injection test",
            "real_catalog_interpretation": "rank stability only; no confirmed real positive pairs",
            "most_stable_candidates": stability.head(20).to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--deployments", nargs="+", choices=["GWTC3", "GWTC4"], default=["GWTC3", "GWTC4"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    if not args.materialize_only and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; refusing to launch the formal O3/O4 multi-seed retraining on CPU")
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_config.json",
        {
            "deployments": args.deployments,
            "seeds": args.seeds,
            "samples_per_family": args.samples_per_family,
            "epochs": args.epochs,
            "bootstrap_draws": args.bootstrap_draws,
            "gwtc3_source": O3_SOURCE,
            "gwtc4_source": O4_SOURCE,
            "main_channels": ["waveform", "time-delay", "unified posterior-overlap sky"],
            "snr_policy": "audit-only",
        },
    )
    sources = {"GWTC3": O3_SOURCE, "GWTC4": O4_SOURCE}
    if args.materialize_only:
        for deployment in args.deployments:
            for seed in args.seeds:
                seed_dir = args.out_root / deployment.lower() / f"seed_{int(seed)}"
                prepare_seed_layout(seed_dir, sources[deployment])
                payload = materialize_compact_dataset(
                    seed_dir, sources[deployment], int(seed) + 100, args.samples_per_family
                )
                print(json.dumps({"deployment": deployment, **payload}, indent=2), flush=True)
        return
    for deployment in args.deployments:
        completed: list[int] = []
        for seed in args.seeds:
            run_seed(
                deployment,
                sources[deployment],
                int(seed),
                args.out_root,
                args.samples_per_family,
                args.epochs,
                args.bootstrap_draws,
            )
            completed.append(int(seed))
            aggregate_deployment(args.out_root, deployment, completed)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
