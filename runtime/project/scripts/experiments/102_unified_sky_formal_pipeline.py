from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import itertools
import json
import math
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ET3_MATCH_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
ET3_ENCODER_ROOT = REPO_ROOT / "runs" / "et3_fresh50_full_catalog_20260616" / "fresh_mixed_encoders"
GWTC3_RUN = REPO_ROOT / "runs" / "real_gwtc_lensing_search_20260625"
GWTC4_RUN = REPO_ROOT / "runs" / "real_gwtc34_lensing_search_20260629"
DEFAULT_OUT = REPO_ROOT / "runs" / "unified_sky_posterior_overlap_20260708"
DEFAULT_PACKAGE = REPO_ROOT / "packages" / "unified_sky_posterior_overlap_20260708.tar.gz"

WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SECONDS_PER_DAY = 86400.0

from scripts.sky.sky_posterior_overlap import (  # noqa: E402
    angular_separation_rad,
    gaussian_log_cosine_overlap_matrix,
    gaussian_log_cosine_overlap_pairs,
)


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return val if math.isfinite(val) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n", encoding="utf-8")


def row_z_zero_diag(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float32).copy()
    np.fill_diagonal(arr, np.nan)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    out = (arr - mu) / np.maximum(sd, 1e-8)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32, copy=False)


def valid_queries(gt: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(gt) >= 0)[0].astype(np.int32)


def family_by_index(meta: list[dict[str, Any]], rows: np.ndarray) -> np.ndarray:
    return np.asarray([meta[int(row)]["family"] for row in rows])


def metrics_from_components(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
    gt: np.ndarray,
    meta: list[dict[str, Any]],
    fresh_module,
) -> dict[str, dict]:
    rows = valid_queries(gt)
    if len(rows) == 0:
        raise RuntimeError("No valid lensed-image queries")
    score_rows = None
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        part = np.asarray(components[name][rows], dtype=np.float32) * float(weight)
        score_rows = part if score_rows is None else score_rows + part
    if score_rows is None:
        raise ValueError("all weights are zero")
    score_rows = np.asarray(score_rows, dtype=np.float32)
    score_rows[np.arange(len(rows)), rows] = -np.inf
    partners = gt[rows].astype(np.int32)
    true_scores = score_rows[np.arange(len(rows)), partners]
    ranks = 1 + np.sum(score_rows > true_scores[:, None], axis=1)
    query_family = family_by_index(meta, rows)
    return fresh_module.metrics_from_ranks(ranks.astype(np.int32), query_family, len(gt))


def score_matrix_from_components(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names = [name for name, weight in weights.items() if weight != 0.0]
    if not names:
        raise ValueError("all weights are zero")
    out = np.zeros_like(components[names[0]], dtype=np.float32)
    for name in names:
        out += float(weights[name]) * np.asarray(components[name], dtype=np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def metric_key(metrics: dict[str, dict]) -> tuple[float, float, float, float]:
    overall = metrics["overall"]
    return (
        float(overall.get("r@10", 0.0)),
        float(overall.get("r@5", 0.0)),
        float(overall.get("r@1", 0.0)),
        float(overall.get("top_1pct", 0.0)),
    )


def select_et3_weights(
    val_components: dict[str, np.ndarray],
    channels: tuple[str, ...],
    val_gt: np.ndarray,
    val_meta: list[dict[str, Any]],
    fresh_module,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, dict] | None = None
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for values in itertools.product(WEIGHT_GRID, repeat=len(channels)):
        if all(v == 0.0 for v in values):
            continue
        weights = {ch: float(v) for ch, v in zip(channels, values)}
        metrics = metrics_from_components(val_components, weights, val_gt, val_meta, fresh_module)
        key = metric_key(metrics)
        if key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics
        rows.append({
            **weights,
            "overall_r@1": metrics["overall"]["r@1"],
            "overall_r@5": metrics["overall"]["r@5"],
            "overall_r@10": metrics["overall"]["r@10"],
            "overall_median_true_rank": metrics["overall"]["median_true_rank"],
            "sis_r@10": metrics["SIS"]["r@10"],
            "pm_r@10": metrics["PM"]["r@10"],
        })
    if best_weights is None or best_metrics is None:
        raise RuntimeError(f"No valid weights for {channels}")
    grid = pd.DataFrame(rows).sort_values(
        ["overall_r@10", "overall_r@5", "overall_r@1"],
        ascending=[False, False, False],
    )
    return best_weights, grid.reset_index(drop=True)


def flatten_metric_rows(
    metrics: dict[str, dict],
    variant: str,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extra = extra or {}
    rows = []
    for subset, vals in metrics.items():
        rows.append({"stage": stage, "variant": variant, "subset": subset, **vals, **extra})
    return rows


def run_et3_unified(out_root: Path) -> dict[str, Any]:
    out_dir = out_root / "et3"
    out_dir.mkdir(parents=True, exist_ok=True)
    liao = load_module_from_path(
        "liao_realistic_p1_p2_rerank",
        REPO_ROOT / "scripts" / "experiments" / "88_liao_realistic_p1_p2_rerank.py",
    )
    fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")

    liao.base.ROOTS[("SIS", "ET3")] = ET3_MATCH_ROOT
    liao.base.ROOTS[("PM", "ET3")] = ET3_MATCH_ROOT
    liao.ENCODER_ROOT = ET3_ENCODER_ROOT
    liao.JOBS = [("ET3", "noisy")]

    loaded = liao.load_job("ET3", "noisy")
    cfg = loaded["cfg"]
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt, seed=20260708)
    val_sky = liao.make_observed_sky("ET3", val_raw, val_time, seed=801000)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=802000)
    val_sky.to_csv(out_dir / "et3_val_unified_observed_sky_audit.csv", index=False)
    test_sky.to_csv(out_dir / "et3_test_unified_observed_sky_audit.csv", index=False)

    print("ET3: building unified sky matrices", flush=True)
    val_sky_score = gaussian_log_cosine_overlap_matrix(val_sky, chunk_rows=64, diagonal=0.0)
    test_sky_score = gaussian_log_cosine_overlap_matrix(test_sky, chunk_rows=64, diagonal=0.0)

    print("ET3: building time LR matrices", flush=True)
    val_time_score = liao.time_lr_score_matrix(val_time, prior)
    test_time_score = liao.time_lr_score_matrix(test_time, prior)

    val_components = {
        "waveform": row_z_zero_diag(val_scores),
        "time": row_z_zero_diag(val_time_score),
        "sky": row_z_zero_diag(val_sky_score),
    }
    test_components = {
        "waveform": row_z_zero_diag(test_scores),
        "time": row_z_zero_diag(test_time_score),
        "sky": row_z_zero_diag(test_sky_score),
    }
    del val_sky_score, test_sky_score, val_time_score, test_time_score
    gc.collect()

    variants: dict[str, dict[str, float]] = {
        "waveform_only": {"waveform": 1.0},
        "liao_time_delay_lr_only": {"time": 1.0},
        "unified_observed_sky_logcos_only": {"sky": 1.0},
    }
    selected_specs = {
        "waveform_plus_time_val_selected": ("waveform", "time"),
        "waveform_plus_unified_sky_val_selected": ("waveform", "sky"),
        "time_plus_unified_sky_val_selected": ("time", "sky"),
        "waveform_plus_time_plus_unified_sky_val_selected": ("waveform", "time", "sky"),
    }

    rows: list[dict[str, Any]] = []
    grid_paths: dict[str, str] = {}
    weights_payload: dict[str, Any] = {}
    for variant, weights in variants.items():
        metrics = metrics_from_components(test_components, weights, test_gt, test_ds.meta, fresh)
        rows.extend(flatten_metric_rows(metrics, variant, "single_channel", {"weights_json": json.dumps(weights)}))

    for variant, channels in selected_specs.items():
        print(f"ET3: selecting {variant}", flush=True)
        weights, grid = select_et3_weights(val_components, channels, val_gt, val_ds.meta, fresh)
        grid_path = out_dir / f"et3_weight_grid_{variant}.csv"
        grid.to_csv(grid_path, index=False)
        metrics = metrics_from_components(test_components, weights, test_gt, test_ds.meta, fresh)
        rows.extend(flatten_metric_rows(metrics, variant, "validation_selected", {"weights_json": json.dumps(weights)}))
        grid_paths[variant] = str(grid_path)
        weights_payload[variant] = {
            "channels": list(channels),
            "selected_weights": weights,
            "validation_best": grid.iloc[0].to_dict(),
        }

    summary = pd.DataFrame(rows)
    summary.insert(0, "detector", "ET3")
    summary.insert(1, "data_mode", "noisy")
    summary.to_csv(out_dir / "et3_unified_sky_modality_summary.csv", index=False)
    prior_public = {k: v for k, v in prior.items() if not isinstance(v, np.ndarray)}
    sky_diag = {
        "scenario": str(test_sky["scenario"].iloc[0]),
        "sky_model": str(test_sky["sky_model"].iloc[0]),
        "sky_sampling": str(test_sky["sky_sampling"].iloc[0]),
        "a90_median_deg2": float(test_sky["sky_area90_deg2"].median()),
        "a90_p90_deg2": float(test_sky["sky_area90_deg2"].quantile(0.9)),
        "sigma_median_rad": float(test_sky["sky_sigma_rad"].median()),
    }
    write_json(out_dir / "et3_unified_sky_summary.json", {
        "method": "ET-3 existing waveform scores + Liao time-delay LR + unified Gaussian posterior log-cosine sky overlap",
        "n_test_events": int(len(test_gt)),
        "n_valid_lensed_queries": int(len(valid_queries(test_gt))),
        "catalog_composition": fresh.full_catalog_composition(test_ds.meta),
        "time_prior": prior_public,
        "sky_diagnostics": sky_diag,
        "selected_weights": weights_payload,
        "grid_paths": grid_paths,
        "summary_csv": str(out_dir / "et3_unified_sky_modality_summary.csv"),
    })

    make_et3_figure(summary, out_root / "figures")
    return {
        "summary": summary,
        "weights": weights_payload,
        "sky_diag": sky_diag,
        "n_events": int(len(test_gt)),
        "n_queries": int(len(valid_queries(test_gt))),
    }


def make_et3_figure(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    overall = summary[summary["subset"] == "overall"].copy()
    order = [
        "waveform_only",
        "liao_time_delay_lr_only",
        "unified_observed_sky_logcos_only",
        "time_plus_unified_sky_val_selected",
        "waveform_plus_time_plus_unified_sky_val_selected",
    ]
    labels = ["Waveform", "Time", "Sky", "Time+sky", "Waveform+time+sky"]
    overall = overall.set_index("variant").reindex(order).reset_index()
    x = np.arange(len(order))
    plt.rcParams.update({"font.family": "Times New Roman", "axes.labelweight": "bold", "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - 0.18, overall["r@1"].to_numpy(), width=0.34, label="R@1", color="#4C78A8")
    ax.bar(x + 0.18, overall["r@10"].to_numpy(), width=0.34, label="R@10", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Companion retrieval recall", fontweight="bold")
    ax.set_title("ET-3 unified observed-sky posterior-overlap reranking")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_et3_unified_sky_summary.pdf", dpi=300)
    fig.savefig(fig_dir / "fig_et3_unified_sky_summary.png", dpi=300)
    plt.close(fig)


def deploy_module():
    return load_module_from_path(
        "gwtc34_real_deployment",
        REPO_ROOT / "scripts" / "real_search" / "15_gwtc34_real_deployment.py",
    )


def unified_validation_pair_table(
    deploy,
    run_dir: Path,
    family: str,
    epochs: int,
    samples: int,
    cpu: bool,
    delay_prior: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    ckpt = run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "model.pt"
    model, cfg = deploy.load_model_from_checkpoint(
        ckpt,
        run_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        samples,
        cpu,
    )
    cfg.model_type = family
    arrays = deploy.load_match_arrays(cfg)
    splits = deploy.split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = deploy.EvaluationSet(arrays, splits["lensed"]["val"], splits["unlensed"]["val"], cfg)
    emb = deploy.embed_eval(model, ds, cfg, cpu=cpu)
    wf = deploy.similarity_matrix(emb)
    gt = deploy.ground_truth_partner(ds.meta)
    events = deploy.synthetic_validation_observables(ds.meta, family, delay_prior, seed=seed)
    ii, jj = np.triu_indices(len(events), k=1)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    dt_days = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    bins = deploy.log_bins_for(dt_days, delay_prior, min_floor=1e-4, n_bins=80)
    time_score = deploy.empirical_hist_lr(dt_days, delay_prior, dt_days, bins)
    sky_score, theta, norm_sep = gaussian_log_cosine_overlap_pairs(
        events["ra_obs"].to_numpy(dtype=np.float64)[ii],
        events["dec_obs"].to_numpy(dtype=np.float64)[ii],
        events["sky_sigma_rad"].to_numpy(dtype=np.float64)[ii],
        events["ra_obs"].to_numpy(dtype=np.float64)[jj],
        events["dec_obs"].to_numpy(dtype=np.float64)[jj],
        events["sky_sigma_rad"].to_numpy(dtype=np.float64)[jj],
    )
    return pd.DataFrame({
        "family": family,
        "idx_i": ii.astype(np.int32),
        "idx_j": jj.astype(np.int32),
        "is_true_pair": (gt[ii] == jj).astype(np.int8),
        "waveform_score": wf[ii, jj].astype(np.float32),
        "time_score": time_score.astype(np.float32),
        "sky_score": sky_score.astype(np.float32),
        "sky_log_cosine_overlap": sky_score.astype(np.float32),
        "delta_t_days": dt_days.astype(np.float64),
        "sky_ang_sep_rad": theta.astype(np.float32),
        "sky_norm_sep": norm_sep.astype(np.float32),
        "event_count": int(len(events)),
    })


def primary_manifest(run_dir: Path) -> pd.DataFrame:
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[events["include_in_primary_search"] == True].copy()
    return primary.sort_values("gps_time").reset_index(drop=True)


def safe_rank(scores: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    ai = scores["event_i"].astype(str).str.contains(a, regex=False)
    aj = scores["event_j"].astype(str).str.contains(b, regex=False)
    bi = scores["event_i"].astype(str).str.contains(b, regex=False)
    bj = scores["event_j"].astype(str).str.contains(a, regex=False)
    row = scores[(ai & aj) | (bi & bj)]
    if row.empty:
        return {"found": False}
    r = row.iloc[0]
    return {
        "found": True,
        "rank": int(r["rank"]),
        "event_i": str(r["event_i"]),
        "event_j": str(r["event_j"]),
        "final_score": float(r["final_score"]),
        "waveform_available": bool(r.get("waveform_available", False)),
        "full_waveform_scored": bool(r.get("full_waveform_scored", False)),
    }


def run_gwtc_unified(
    out_root: Path,
    source_run: Path,
    label: str,
    prefix: str,
    epochs: int,
    samples: int,
    cpu: bool,
) -> dict[str, Any]:
    deploy = deploy_module()
    out_dir = out_root / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    delay_prior = deploy.liao_delay_samples()
    validation_frames = []
    for fam in deploy.FAMILIES:
        print(f"{label}: building unified validation table for {fam}", flush=True)
        validation_frames.append(unified_validation_pair_table(deploy, source_run, fam, epochs, samples, cpu, delay_prior, seed=20260708))
    validation = pd.concat(validation_frames, ignore_index=True)
    validation_path = out_dir / f"{prefix}_unified_fusion_validation.parquet"
    validation.to_parquet(validation_path, index=False)
    weights, grid = deploy.select_weights(validation, force_waveform_zero=False, require_all_positive=False)
    baseline_weights, baseline_grid = deploy.select_weights(validation, force_waveform_zero=True, require_time_sky_positive=False)
    grid.to_csv(out_dir / f"{prefix}_unified_weight_grid_waveform_time_sky.csv", index=False)
    baseline_grid.to_csv(out_dir / f"{prefix}_unified_weight_grid_time_sky_baseline.csv", index=False)

    obs = pd.read_parquet(source_run / "features" / "real_pair_observable_features.parquet")
    sim = pd.read_parquet(source_run / "features" / "real_waveform_similarity.parquet")
    primary = primary_manifest(source_run)
    real = obs.merge(
        sim[["idx_i", "idx_j", "waveform_available", "waveform_score", "waveform_score_sis", "waveform_score_pm"]],
        on=["idx_i", "idx_j"],
        how="left",
    )
    if "object_class" in primary.columns:
        object_class = primary["object_class"].fillna("unknown").astype(str).to_numpy()
    else:
        object_class = np.asarray(["BBH"] * len(primary), dtype=object)
    if "is_ood_for_bbh_encoder" in primary.columns:
        event_ood = primary["is_ood_for_bbh_encoder"].fillna(False).astype(bool).to_numpy()
    else:
        event_ood = np.zeros(len(primary), dtype=bool)
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
    forced = deploy.add_real_scores(real, {"waveform": 1.0, "time": 1.0, "sky": 1.0}, "unified_forced_equal_waveform_time_sky")
    selected_path = out_dir / f"{prefix}_pair_scores_unified_waveform_time_sky.parquet"
    baseline_path = out_dir / f"{prefix}_pair_scores_unified_time_sky_baseline.parquet"
    forced_path = out_dir / f"{prefix}_pair_scores_unified_forced_equal_waveform_time_sky.parquet"
    selected.to_parquet(selected_path, index=False)
    baseline.to_parquet(baseline_path, index=False)
    forced.to_parquet(forced_path, index=False)
    deploy.shortlist(selected, n=100).to_csv(out_dir / f"{prefix}_candidate_shortlist_unified_waveform_time_sky.csv", index=False)
    deploy.shortlist(baseline, n=100).to_csv(out_dir / f"{prefix}_candidate_shortlist_unified_time_sky_baseline.csv", index=False)
    deploy.shortlist(forced, n=100).to_csv(out_dir / f"{prefix}_candidate_shortlist_unified_forced_equal_waveform_time_sky.csv", index=False)

    top = selected.iloc[0]
    full_subset = selected[selected["full_waveform_scored"] == True]
    full_top = full_subset.iloc[0] if len(full_subset) else None
    wf = real.loc[real["full_waveform_scored"], "raw_waveform_score"].dropna()
    gate_path = source_run / "results" / ("waveform_gate1_metrics.csv" if prefix == "gwtc3" else "gwtc4_waveform_gate1_metrics.csv")
    gate = pd.read_csv(gate_path) if gate_path.exists() else pd.DataFrame()
    validation_best = grid.iloc[0].to_dict()
    baseline_best = baseline_grid.iloc[0].to_dict()
    summary = {
        "catalog": label,
        "source_run": str(source_run),
        "method": "waveform + time-delay + real HEALPix sky log-cosine overlap",
        "snr_policy": "SNR/amplitude columns are retained for audit only and do not enter final_score.",
        "validation_sky_policy": "Synthetic validation uses Gaussian posterior log-cosine overlap, matching real HEALPix cosine-overlap semantics.",
        "n_events": int(len(primary)),
        "n_pairs": int(len(selected)),
        "n_full_waveform_scored_pairs": int(selected["full_waveform_scored"].sum()),
        "n_waveform_available_pairs_raw": int(selected["waveform_available"].sum()),
        "n_ood_events": int(event_ood.sum()),
        "selected_weights": weights,
        "time_sky_baseline_weights": baseline_weights,
        "validation_best": validation_best,
        "time_sky_baseline_validation_best": baseline_best,
        "gate1_metrics_path": str(gate_path),
        "gate1_macro_r_at_10": float(gate[gate["split"] == "test"]["r_at_10"].mean()) if "split" in gate.columns else None,
        "gate1_min_family_r_at_10": float(gate[gate["split"] == "test"]["r_at_10"].min()) if "split" in gate.columns else None,
        "waveform_score_audit": {
            "n": int(len(wf)),
            "unique": int(wf.nunique()) if len(wf) else 0,
            "min": float(wf.min()) if len(wf) else None,
            "max": float(wf.max()) if len(wf) else None,
            "std": float(wf.std()) if len(wf) else None,
        },
        "top_pair": {
            "rank": int(top["rank"]),
            "event_i": str(top["event_i"]),
            "event_j": str(top["event_j"]),
            "final_score": float(top["final_score"]),
            "waveform_score": None if pd.isna(top.get("waveform_score")) else float(top["waveform_score"]),
            "time_score": float(top["time_score"]),
            "sky_score": float(top["sky_score"]),
            "waveform_available": bool(top["waveform_available"]),
            "full_waveform_scored": bool(top["full_waveform_scored"]),
            "pair_has_ood": bool(top["pair_has_ood"]),
        },
        "full_waveform_subset_top_pair": None if full_top is None else {
            "catalog_rank": int(full_top["rank"]),
            "event_i": str(full_top["event_i"]),
            "event_j": str(full_top["event_j"]),
            "final_score": float(full_top["final_score"]),
            "waveform_score": float(full_top["waveform_score"]),
        },
        "gw170104_gw170814_rank": safe_rank(selected, "GW170104", "GW170814") if prefix == "gwtc3" else {"found": False},
        "gw170104_gw170814_time_sky_rank": safe_rank(baseline, "GW170104", "GW170814") if prefix == "gwtc3" else {"found": False},
        "outputs": {
            "validation": str(validation_path),
            "pair_scores": str(selected_path),
            "candidate_shortlist": str(out_dir / f"{prefix}_candidate_shortlist_unified_waveform_time_sky.csv"),
        },
    }
    write_json(out_dir / f"{prefix}_unified_fusion_summary.json", summary)
    return summary


def make_gwtc_figure(out_root: Path, g3: dict[str, Any], g4: dict[str, Any]) -> None:
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "axes.labelweight": "bold", "axes.titleweight": "bold"})
    rows = []
    for summary in (g3, g4):
        rows.append({
            "catalog": summary["catalog"],
            "WTS R@10": summary["validation_best"]["macro_r_at_10"],
            "TS R@10": summary["time_sky_baseline_validation_best"]["macro_r_at_10"],
            "WTS R@1": summary["validation_best"]["macro_r_at_1"],
            "TS R@1": summary["time_sky_baseline_validation_best"]["macro_r_at_1"],
        })
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].bar(x - 0.18, df["TS R@10"], width=0.34, label="Time+sky", color="#72B7B2")
    axes[0].bar(x + 0.18, df["WTS R@10"], width=0.34, label="Waveform+time+sky", color="#4C78A8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["catalog"])
    axes[0].set_ylabel("Validation R@10", fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)

    pair_df = pd.DataFrame([
        {"catalog": g3["catalog"], "events": g3["n_events"], "pairs": g3["n_pairs"], "waveform pairs": g3["n_full_waveform_scored_pairs"]},
        {"catalog": g4["catalog"], "events": g4["n_events"], "pairs": g4["n_pairs"], "waveform pairs": g4["n_full_waveform_scored_pairs"]},
    ])
    axes[1].bar(x - 0.18, pair_df["pairs"], width=0.34, label="All pairs", color="#B279A2")
    axes[1].bar(x + 0.18, pair_df["waveform pairs"], width=0.34, label="Full waveform-scored", color="#54A24B")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_df["catalog"])
    axes[1].set_ylabel("Real-catalog unordered pairs", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Unified sky-overlap real-catalog deployment audit", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_unified_sky_gwtc34_summary.pdf", dpi=300)
    fig.savefig(fig_dir / "fig_unified_sky_gwtc34_summary.png", dpi=300)
    plt.close(fig)


def write_report(out_root: Path, et3: dict[str, Any], g3: dict[str, Any], g4: dict[str, Any]) -> None:
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    et3_summary = et3["summary"]

    def metric_line(variant: str) -> str:
        row = et3_summary[(et3_summary["subset"] == "overall") & (et3_summary["variant"] == variant)].iloc[0]
        return f"{variant}: R@1={row['r@1']:.4f}, R@10={row['r@10']:.4f}, median rank={row['median_true_rank']:.1f}"

    lines = [
        "# 统一天空后验重叠正式实验报告",
        "",
        "## 方法口径",
        "本实验把模拟目录和真实 GWTC 目录的 sky 通道统一为 posterior-overlap score。真实 GWTC 使用 PE release 的 HEALPix posterior map，并计算 cosine overlap；ET-3 和真实噪声注入验证集没有 PE map，因此使用由 observed sky center 和 A90/sigma 构造的局部二维高斯 posterior，其 log-cosine overlap 为",
        "",
        "`log C_ij = log(2 sigma_i sigma_j / (sigma_i^2 + sigma_j^2)) - theta_ij^2 / [2 (sigma_i^2 + sigma_j^2)]`。",
        "",
        "该实验不重新训练 waveform encoder；它只统一 sky score 定义，并重新选择 validation weights、重新排序真实目录候选。SNR/amplitude 不进入 final score。",
        "",
        "## ET-3 结果",
        f"- test events: {et3['n_events']}",
        f"- valid lensed-image queries: {et3['n_queries']}",
        f"- observed-sky scenario: {et3['sky_diag']['scenario']}",
        f"- median A90: {et3['sky_diag']['a90_median_deg2']:.3f} deg2",
        "",
        metric_line("waveform_only"),
        metric_line("liao_time_delay_lr_only"),
        metric_line("unified_observed_sky_logcos_only"),
        metric_line("time_plus_unified_sky_val_selected"),
        metric_line("waveform_plus_time_plus_unified_sky_val_selected"),
        "",
        "## GWTC-3 / GWTC-4.1 真实目录重排",
        f"- GWTC-3: events={g3['n_events']}, pairs={g3['n_pairs']}, full waveform-scored pairs={g3['n_full_waveform_scored_pairs']}, selected weights={g3['selected_weights']}",
        f"- GWTC-3 top pair: {g3['top_pair']['event_i']} -- {g3['top_pair']['event_j']}, full waveform-scored={g3['top_pair']['full_waveform_scored']}",
        f"- GW170104--GW170814 rank: {g3['gw170104_gw170814_rank']}",
        f"- GWTC-4.1: events={g4['n_events']}, pairs={g4['n_pairs']}, full waveform-scored pairs={g4['n_full_waveform_scored_pairs']}, selected weights={g4['selected_weights']}",
        f"- GWTC-4.1 top pair: {g4['top_pair']['event_i']} -- {g4['top_pair']['event_j']}, full waveform-scored={g4['top_pair']['full_waveform_scored']}, OOD={g4['top_pair']['pair_has_ood']}",
        "",
        "## 结论边界",
        "这些真实目录结果只能解释为 candidate shortlist for Bayesian follow-up，不是透镜探测声明。GWTC-4.1 的 encoder 仍是小预算 run-matched 部署，若作为正文重点，需要进一步训练或放入补充材料说明。",
        "",
        "## 输出文件",
        "- ET-3 summary: `et3/et3_unified_sky_modality_summary.csv`",
        "- GWTC-3 shortlist: `gwtc3/gwtc3_candidate_shortlist_unified_waveform_time_sky.csv`",
        "- GWTC-4.1 shortlist: `gwtc4/gwtc4_candidate_shortlist_unified_waveform_time_sky.csv`",
        "- Figures: `figures/fig_et3_unified_sky_summary.pdf`, `figures/fig_unified_sky_gwtc34_summary.pdf`",
    ]
    text = "\n".join(lines) + "\n"
    (report_dir / "unified_sky_posterior_overlap_report_cn.md").write_text(text, encoding="utf-8")
    (out_root / "README_CURRENT_RESULTS.md").write_text(text, encoding="utf-8")


def package_outputs(out_root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    include = [
        REPO_ROOT / "scripts" / "sky" / "sky_posterior_overlap.py",
        REPO_ROOT / "scripts" / "experiments" / "102_unified_sky_formal_pipeline.py",
        out_root / "et3",
        out_root / "gwtc3",
        out_root / "gwtc4",
        out_root / "figures",
        out_root / "reports",
        out_root / "README_CURRENT_RESULTS.md",
        out_root / "unified_sky_formal_summary.json",
    ]
    with tarfile.open(package_path, "w:gz") as tar:
        for path in include:
            path = path.resolve()
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(REPO_ROOT.resolve())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package-path", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--skip-et3", action="store_true")
    parser.add_argument("--skip-gwtc", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    out_root = args.out_root
    for sub in ["et3", "gwtc3", "gwtc4", "figures", "reports"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    et3 = None
    g3 = None
    g4 = None
    if not args.skip_et3:
        et3 = run_et3_unified(out_root)
    if not args.skip_gwtc:
        g3 = run_gwtc_unified(out_root, GWTC3_RUN, "GWTC-3", "gwtc3", epochs=20, samples=600, cpu=args.cpu)
        g4 = run_gwtc_unified(out_root, GWTC4_RUN, "GWTC-4.1", "gwtc4", epochs=3, samples=200, cpu=args.cpu)
        make_gwtc_figure(out_root, g3, g4)

    if et3 is not None and g3 is not None and g4 is not None:
        write_report(out_root, et3, g3, g4)
        write_json(out_root / "unified_sky_formal_summary.json", {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "method": "Unified sky posterior-overlap rerun for ET-3 and GWTC real deployments",
            "out_root": str(out_root),
            "et3": {
                "n_events": et3["n_events"],
                "n_queries": et3["n_queries"],
                "sky_diag": et3["sky_diag"],
                "selected_weights": et3["weights"],
            },
            "gwtc3": g3,
            "gwtc4": g4,
            "wall_time_s": float(time.perf_counter() - started),
        })
        package_outputs(out_root, args.package_path)
    print(json.dumps({
        "out_root": str(out_root),
        "package_path": str(args.package_path),
        "wall_time_s": float(time.perf_counter() - started),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
