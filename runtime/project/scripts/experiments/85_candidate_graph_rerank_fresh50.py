from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
hard = importlib.import_module("scripts.experiments.81_time_matched_hard_negative_mixed_catalog")
fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")

OUT_ROOT = Path("runs/candidate_graph_rerank_fresh50_20260612")
ENCODER_ROOT = Path("runs/fresh50_full_catalog_ranking_20260611/fresh_mixed_encoders")
JOBS = [("ET", "noisy"), ("LIGO", "noisy")]
FAMILIES = ["SIS", "PM"]
RERANK_VARIANTS = [
    "waveform_plus_time",
    "waveform_plus_time_plus_predicted_sky_overlap",
]
CHUNK_ROWS = 64
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def row_rank_matrix(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(scores.shape, dtype=np.uint16)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1, dtype=np.uint16)
    return ranks


def zscore_matrix(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    vals = x[finite]
    if len(vals) == 0:
        return np.zeros_like(x, dtype=np.float32)
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    return ((x - mu) / max(sd, EPS)).astype(np.float32)


def family_by_index(meta: list[dict], rows: np.ndarray) -> np.ndarray:
    return np.asarray([meta[int(row)]["family"] for row in rows])


def metrics_from_ranks(ranks: np.ndarray, query_family: np.ndarray, catalog_total: int) -> dict[str, dict]:
    return fresh.metrics_from_ranks(ranks, query_family, catalog_total)


def full_catalog_composition(meta: list[dict]) -> dict:
    return fresh.full_catalog_composition(meta)


def add_group_rows(rows: list[dict], detector: str, mode: str, variant: str, stage: str, metrics: dict[str, dict], diag: dict, extra: dict) -> None:
    for subset, values in metrics.items():
        rows.append({
            "detector": detector,
            "data_mode": mode,
            "catalog": "fresh50_mixed_SIS_PM_unlensed_full_catalog",
            "candidate_kind": "full_catalog",
            "subset": subset,
            "variant": variant,
            "stage": stage,
            **diag,
            **values,
            **extra,
        })


def split_all(detector: str, mode: str):
    encoder_dir = ENCODER_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    cfg = base.make_cfg(detector, mode, encoder_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {family: base.FamilyArrays(family, base.ROOTS[(family, detector)], mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    model_path = cfg.out_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing fresh50 model: {model_path}")
    model = base.build_model(cfg)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    history_path = cfg.out_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        train_info = {
            "train_s": float(history["epoch_s"].sum()),
            "mean_epoch_s": float(history["epoch_s"].mean()),
        }
    else:
        train_info = {"train_s": np.nan, "mean_epoch_s": np.nan}
    train_ds, train_raw, _, _, train_emb, _ = base.split_pack(detector, "train", cfg, arrays, splits, model)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = base.split_pack(detector, "val", cfg, arrays, splits, model)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = base.split_pack(detector, "test", cfg, arrays, splits, model)
    sky_model, sky_sigma, sky_mean_err, sky_med_err = base.fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = base.normalize_vectors(sky_model.predict(val_emb))
    test_sky_mu = base.normalize_vectors(sky_model.predict(test_emb))
    return {
        "cfg": cfg,
        "train_info": train_info,
        "val": (val_ds, val_raw, val_time, val_gt, val_scores, val_sky_mu),
        "test": (test_ds, test_raw, test_time, test_gt, test_scores, test_sky_mu),
        "sky_sigma": sky_sigma,
        "sky_mean_err": sky_mean_err,
        "sky_med_err": sky_med_err,
    }


def rerank_score_matrix(variant: str, clf, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, sky_mu: np.ndarray, sky_sigma: float, scores: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    n = len(scores)
    cols = np.arange(n, dtype=np.int32)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = np.arange(start, min(start + CHUNK_ROWS, n), dtype=np.int32)
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(cols, len(rows)).astype(np.int32)
        x = hard.rerank_features(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
        pred = clf.predict_proba(x)[:, 1].reshape(len(rows), n).astype(np.float32)
        pred[np.arange(len(rows)), rows - start] = 0.0
        out[start:start + len(rows)] = pred
    np.fill_diagonal(out, 0.0)
    return out


def graph_components(score_prob: np.ndarray) -> dict[str, np.ndarray]:
    ranks = row_rank_matrix(score_prob)
    reciprocal = (1.0 / np.maximum(ranks.astype(np.float32), 1.0)) + (1.0 / np.maximum(ranks.T.astype(np.float32), 1.0))
    mutual_50 = ((ranks <= 50) & (ranks.T <= 50)).astype(np.float32)
    mutual_100 = ((ranks <= 100) & (ranks.T <= 100)).astype(np.float32)
    indegree_50 = np.sum(ranks <= 50, axis=0).astype(np.float32)
    indegree_100 = np.sum(ranks <= 100, axis=0).astype(np.float32)
    hub = np.log1p(indegree_50)[None, :] + 0.5 * np.log1p(indegree_100)[None, :]
    return {
        "base_logit": logit(score_prob),
        "ranks": ranks,
        "reciprocal_z": zscore_matrix(reciprocal),
        "mutual_50": mutual_50,
        "mutual_100": mutual_100,
        "hub_z": zscore_matrix(hub),
    }


def apply_graph_score(parts: dict[str, np.ndarray], alpha: float, gamma50: float, gamma100: float, beta: float) -> np.ndarray:
    out = (
        parts["base_logit"]
        + alpha * parts["reciprocal_z"]
        + gamma50 * parts["mutual_50"]
        + gamma100 * parts["mutual_100"]
        - beta * parts["hub_z"]
    ).astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def ranks_for_valid(score_matrix: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = hard.valid_queries(gt)
    true_scores = score_matrix[rows, gt[rows].astype(int)]
    ranks = 1 + np.sum(score_matrix[rows] > true_scores[:, None], axis=1)
    return rows, ranks.astype(np.int32)


def eval_parts(parts: dict[str, np.ndarray], gt: np.ndarray, meta: list[dict], params: tuple[float, float, float, float]) -> dict:
    score = apply_graph_score(parts, *params)
    rows, metric_ranks = ranks_for_valid(score, gt)
    query_family = family_by_index(meta, rows)
    return metrics_from_ranks(metric_ranks, query_family, len(gt))


def choose_graph_params(parts: dict[str, np.ndarray], gt: np.ndarray, meta: list[dict]) -> tuple[float, float, float, float, dict]:
    grid = []
    for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
        for gamma50 in (0.0, 0.25, 0.5, 1.0):
            for gamma100 in (0.0, 0.15, 0.3):
                for beta in (0.0, 0.25, 0.5, 1.0):
                    grid.append((alpha, gamma50, gamma100, beta))
    best_params = grid[0]
    best_metrics = eval_parts(parts, gt, meta, best_params)
    best_key = (
        best_metrics["overall"]["r@10"],
        best_metrics["overall"]["r@5"],
        best_metrics["overall"]["r@1"],
        best_metrics["overall"]["top_1pct"],
    )
    for params in grid[1:]:
        metrics = eval_parts(parts, gt, meta, params)
        key = (
            metrics["overall"]["r@10"],
            metrics["overall"]["r@5"],
            metrics["overall"]["r@1"],
            metrics["overall"]["top_1pct"],
        )
        if key > best_key:
            best_key = key
            best_params = params
            best_metrics = metrics
    return (*best_params, best_metrics)


def run_one(detector: str, mode: str) -> list[dict]:
    out_dir = OUT_ROOT / f"{detector.lower()}_{mode}_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    packed = split_all(detector, mode)
    cfg = packed["cfg"]
    train_info = packed["train_info"]
    val_ds, val_raw, val_time, val_gt, val_scores, val_sky_mu = packed["val"]
    test_ds, test_raw, test_time, test_gt, test_scores, test_sky_mu = packed["test"]
    sky_sigma = packed["sky_sigma"]
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    val_hard, val_hard_diag = hard.make_candidate_lists(val_time, val_ds.meta, val_gt, "hard", hard.N_NEG, seed=91001)
    diag = {
        "query_total": int(len(hard.valid_queries(test_gt))),
        **full_catalog_composition(test_ds.meta),
        "epochs": int(cfg.epochs),
        "backbone": cfg.backbone,
        "preprocess": cfg.preprocess,
        "train_s": train_info.get("train_s", np.nan),
        "mean_epoch_s": train_info.get("mean_epoch_s", np.nan),
        "sky_sigma_rad": sky_sigma,
        "sky_val_mean_angular_error_rad": packed["sky_mean_err"],
        "sky_val_median_angular_error_rad": packed["sky_med_err"],
        **{f"val_hard_{key}": value for key, value in val_hard_diag.items()},
    }
    rows = []
    for index, variant in enumerate(RERANK_VARIANTS):
        print("GRAPH_RERANK_TRAIN", detector, mode, variant, flush=True)
        clf, auc, n_train, n_pos = hard.train_reranker(
            variant, val_raw, val_time, val_sky_mu, sky_sigma, val_scores, val_ranks, val_hard, seed=95000 + index
        )
        print("GRAPH_RERANK_VAL_MATRIX", detector, mode, variant, flush=True)
        val_prob = rerank_score_matrix(variant, clf, val_raw, val_time, val_sky_mu, sky_sigma, val_scores, val_ranks)
        val_parts = graph_components(val_prob)
        base_val_score = apply_graph_score(val_parts, 0.0, 0.0, 0.0, 0.0)
        val_q, val_base_ranks = ranks_for_valid(base_val_score, val_gt)
        val_base_metrics = metrics_from_ranks(val_base_ranks, family_by_index(val_ds.meta, val_q), len(val_gt))
        alpha, gamma50, gamma100, beta, best_val_metrics = choose_graph_params(val_parts, val_gt, val_ds.meta)
        print("GRAPH_RERANK_TEST_MATRIX", detector, mode, variant, alpha, gamma50, gamma100, beta, flush=True)
        test_prob = rerank_score_matrix(variant, clf, test_raw, test_time, test_sky_mu, sky_sigma, test_scores, test_ranks)
        test_parts = graph_components(test_prob)
        base_test_score = apply_graph_score(test_parts, 0.0, 0.0, 0.0, 0.0)
        test_q, test_base_ranks = ranks_for_valid(base_test_score, test_gt)
        test_base_metrics = metrics_from_ranks(test_base_ranks, family_by_index(test_ds.meta, test_q), len(test_gt))
        graph_test_metrics = eval_parts(test_parts, test_gt, test_ds.meta, (alpha, gamma50, gamma100, beta))
        common = {"val_auc_hard_sampled": auc, "train_examples": n_train, "train_positive": n_pos}
        add_group_rows(rows, detector, mode, variant + "_baseline_recomputed", "fresh50_full_catalog_rerank_recomputed", test_base_metrics, diag, {
            **common,
            "graph_alpha": 0.0,
            "graph_gamma50": 0.0,
            "graph_gamma100": 0.0,
            "graph_beta": 0.0,
            "val_selected_r@10": val_base_metrics["overall"]["r@10"],
        })
        add_group_rows(rows, detector, mode, variant + "_candidate_graph", "validation_selected_candidate_graph", graph_test_metrics, diag, {
            **common,
            "graph_alpha": alpha,
            "graph_gamma50": gamma50,
            "graph_gamma100": gamma100,
            "graph_beta": beta,
            "val_selected_r@10": best_val_metrics["overall"]["r@10"],
            "val_baseline_r@10": val_base_metrics["overall"]["r@10"],
        })
        pd.DataFrame(rows).to_csv(out_dir / "candidate_graph_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "candidate_graph_summary.csv", index=False)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    all_rows = []
    for detector, mode in JOBS:
        print("RUN_CANDIDATE_GRAPH", detector, mode, flush=True)
        all_rows.extend(run_one(detector, mode))
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "candidate_graph_summary_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "candidate_graph_summary.csv", index=False)
    (OUT_ROOT / "protocol_summary.json").write_text(json.dumps({
        "elapsed_s": float(time.perf_counter() - start),
        "jobs": JOBS,
        "variants": RERANK_VARIANTS,
        "note": "Candidate-graph reranking sweeps reciprocal-rank, mutual-topK, and hub-penalty weights on validation full catalogs, then applies selected weights to test full catalogs.",
    }, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
