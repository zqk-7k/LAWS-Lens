from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matchgw.data import ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import retrieval_metrics
from matchgw.pipeline import build_model
from matchgw.trigger_time import log1p_delta_time_obs

aux = importlib.import_module("scripts.experiments.21_observable_aux_reranker")

EXP_MODULES = {
    "PM": importlib.import_module("scripts.experiments.75_pm_mass_1e4_1e10_td_min24s_ep50_aux_compare"),
    "SIS": importlib.import_module("scripts.experiments.76_sis_ep50_aux_compare"),
}

OUT_ROOT = Path("runs/full_modality_percent_compare_20260609")
JOBS = [("ET", "pure"), ("ET", "noisy"), ("LIGO", "pure"), ("LIGO", "noisy")]
DIRECT_VARIANTS = [
    "waveform_only",
    "time_only",
    "true_sky_sep_only",
    "true_sky_overlap_only",
    "predicted_sky_overlap_only",
]
RERANK_VARIANTS = [
    "waveform_plus_time",
    "waveform_plus_true_sky_sep",
    "waveform_plus_true_sky_overlap",
    "waveform_plus_predicted_sky_overlap",
    "time_plus_true_sky_sep",
    "time_plus_true_sky_overlap",
    "time_plus_predicted_sky_overlap",
    "waveform_plus_time_plus_true_sky_sep",
    "waveform_plus_time_plus_true_sky_overlap",
    "waveform_plus_time_plus_predicted_sky_overlap",
]
NEG_PER_POS = 500
CHUNK_ROWS = 64
EPS = 1e-8


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    return (x / norm).astype(np.float32)


def unit_vectors(obs: pd.DataFrame) -> np.ndarray:
    ra = obs["ra"].to_numpy(dtype=np.float64)
    dec = obs["dec"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    ).astype(np.float32)


def angular_sep_from_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def log_gaussian_overlap_from_unit(mu_i: np.ndarray, mu_j: np.ndarray, sigma_i: float, sigma_j: float) -> np.ndarray:
    sep = angular_sep_from_unit(mu_i, mu_j)
    var = sigma_i * sigma_i + sigma_j * sigma_j
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def fit_sky_predictor(train_obs: pd.DataFrame, train_emb: np.ndarray, val_obs: pd.DataFrame, val_emb: np.ndarray):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 12)))
    model.fit(train_emb, unit_vectors(train_obs))
    val_pred = normalize_vectors(model.predict(val_emb))
    err = angular_sep_from_unit(val_pred, unit_vectors(val_obs))
    sigma = float(max(0.03, np.median(err)))
    return model, sigma, float(np.mean(err)), float(np.median(err))


def row_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int32)
    return ranks


def true_sky_sep(raw_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    ra = raw_obs["ra"].to_numpy(dtype=np.float64)
    dec = raw_obs["dec"].to_numpy(dtype=np.float64)
    return aux.angular_sep(ra[a], dec[a], ra[c], dec[c]).astype(np.float32)


def true_log_sky_overlap(raw_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    sep = true_sky_sep(raw_obs, a, c)
    var = 0.08 * 0.08 * 2.0
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def add_percent_metrics(metrics: dict, ranks: np.ndarray, n_candidates: int) -> dict:
    usable = max(n_candidates - 1, 1)
    for pct in (1, 5, 10):
        k = max(1, int(math.ceil(usable * pct / 100.0)))
        metrics[f"top_{pct}pct_k"] = k
        metrics[f"top_{pct}pct"] = float(np.mean(ranks <= k))
    metrics["median_rank_pct"] = float(np.median(ranks / usable))
    metrics["mean_rank_pct"] = float(np.mean(ranks / usable))
    return metrics


def metrics_from_scores(scores: np.ndarray, gt: np.ndarray) -> dict:
    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    true = scores[valid, gt[valid].astype(int)]
    ranks = 1 + np.sum(scores[valid] > true[:, None], axis=1)
    metrics = retrieval_metrics(scores, gt, ks=(1, 5, 10, 50, 100, 500))
    metrics["valid"] = int(len(valid))
    return add_percent_metrics(metrics, ranks.astype(np.int32), scores.shape[1])


def direct_scores(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, scores: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None) -> np.ndarray:
    n = len(time_obs)
    if variant == "waveform_only":
        return scores.astype(np.float32)
    out = np.empty((n, n), dtype=np.float32)
    cols = np.arange(n, dtype=np.int32)
    for start in range(0, n, CHUNK_ROWS):
        rows = np.arange(start, min(start + CHUNK_ROWS, n), dtype=np.int32)
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(cols, len(rows)).astype(np.int32)
        if variant == "time_only":
            values = -log1p_delta_time_obs(time_obs, a, c)
        elif variant == "true_sky_sep_only":
            values = -true_sky_sep(raw_obs, a, c)
        elif variant == "true_sky_overlap_only":
            values = true_log_sky_overlap(raw_obs, a, c)
        elif variant == "predicted_sky_overlap_only":
            if sky_mu is None or sky_sigma is None:
                raise ValueError("predicted sky overlap requires sky_mu and sky_sigma")
            values = log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma)
        else:
            raise ValueError(variant)
        out[start : start + len(rows)] = values.reshape(len(rows), n)
    np.fill_diagonal(out, -np.inf)
    return out


def feature_matrix(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    cols = []
    if "time" in variant:
        cols.append(log1p_delta_time_obs(time_obs, a, c))
    if "true_sky_sep" in variant:
        cols.append(true_sky_sep(raw_obs, a, c))
    if "true_sky_overlap" in variant:
        cols.append(true_log_sky_overlap(raw_obs, a, c))
    if "predicted_sky_overlap" in variant:
        if sky_mu is None or sky_sigma is None:
            raise ValueError("predicted sky overlap requires sky_mu and sky_sigma")
        cols.append(log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma))
    if "waveform" in variant:
        cols.append(scores[a, c].astype(np.float32))
        cols.append((1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32))
    return np.column_stack(cols).astype(np.float32)


def train_examples(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    pos_a = valid
    pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    a = np.concatenate([pos_a, neg_a])
    c = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    x = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return x[order], y[order]


def eval_rerank(variant: str, clf, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    true_ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start : start + CHUNK_ROWS]
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        x = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
        pred = clf.predict_proba(x)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        true_ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(true_ranks, dtype=np.int32)
    metrics = {
        "r@1": float(np.mean(r <= 1)),
        "r@5": float(np.mean(r <= 5)),
        "r@10": float(np.mean(r <= 10)),
        "r@50": float(np.mean(r <= 50)),
        "r@100": float(np.mean(r <= 100)),
        "r@500": float(np.mean(r <= 500)),
        "median_true_rank": float(np.median(r)),
        "valid": int(len(valid)),
    }
    return add_percent_metrics(metrics, r, n)


def load_pack(family: str, detector: str, mode: str):
    exp = EXP_MODULES[family]
    cfg = exp.make_cfg(detector, mode)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    model_path = cfg.out_dir / "model.pt"
    model = build_model(cfg)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    train_ds, train_raw, train_time, train_gt, train_emb, train_scores = exp.split_pack(arrays, splits, cfg, detector, "train", model, cfg.out_dir)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = exp.split_pack(arrays, splits, cfg, detector, "val", model, cfg.out_dir)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = exp.split_pack(arrays, splits, cfg, detector, "test", model, cfg.out_dir)
    return cfg, (train_raw, train_time, train_gt, train_emb, train_scores), (val_raw, val_time, val_gt, val_emb, val_scores), (test_raw, test_time, test_gt, test_emb, test_scores)


def run_one(family: str, detector: str, mode: str) -> list[dict]:
    t0 = time.perf_counter()
    cfg, train, val, test = load_pack(family, detector, mode)
    train_raw, train_time, train_gt, train_emb, train_scores = train
    val_raw, val_time, val_gt, val_emb, val_scores = val
    test_raw, test_time, test_gt, test_emb, test_scores = test

    val_ranks = row_ranks(val_scores)
    test_ranks = row_ranks(test_scores)
    sky_model, sky_sigma, sky_mean_err, sky_med_err = fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = normalize_vectors(sky_model.predict(val_emb))
    test_sky_mu = normalize_vectors(sky_model.predict(test_emb))

    rows = []
    for variant in DIRECT_VARIANTS:
        print("DIRECT", family, detector, mode, variant, flush=True)
        score = direct_scores(variant, test_raw, test_time, test_gt, test_scores, test_sky_mu, sky_sigma)
        met = metrics_from_scores(score, test_gt)
        rows.append({
            "family": family,
            "detector": detector,
            "data_mode": mode,
            "variant": variant,
            "stage": "direct_score",
            "features": variant.replace("_only", ""),
            **met,
            "sky_sigma_rad": sky_sigma,
            "sky_val_mean_angular_error_rad": sky_mean_err,
            "sky_val_median_angular_error_rad": sky_med_err,
            "elapsed_s": float(time.perf_counter() - t0),
        })

    for idx, variant in enumerate(RERANK_VARIANTS):
        print("RERANK", family, detector, mode, variant, flush=True)
        x_val, y_val = train_examples(
            variant, val_raw, val_time, val_gt, val_sky_mu, sky_sigma, val_scores, val_ranks, seed=90609 + idx
        )
        clf = HistGradientBoostingClassifier(
            max_iter=320,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1e-4,
            class_weight="balanced",
            random_state=90609 + idx,
        )
        clf.fit(x_val, y_val)
        pred_val = clf.predict_proba(x_val)[:, 1]
        met = eval_rerank(variant, clf, test_raw, test_time, test_gt, test_sky_mu, sky_sigma, test_scores, test_ranks)
        rows.append({
            "family": family,
            "detector": detector,
            "data_mode": mode,
            "variant": variant,
            "stage": "catalog_hgb_rerank",
            "features": variant,
            "val_auc_sampled": float(roc_auc_score(y_val, pred_val)),
            "train_examples": int(len(y_val)),
            "train_positive": int(y_val.sum()),
            **met,
            "sky_sigma_rad": sky_sigma,
            "sky_val_mean_angular_error_rad": sky_mean_err,
            "sky_val_median_angular_error_rad": sky_med_err,
            "elapsed_s": float(time.perf_counter() - t0),
        })

    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for family in ("PM", "SIS"):
        for detector, mode in JOBS:
            all_rows.extend(run_one(family, detector, mode))
            pd.DataFrame(all_rows).to_csv(OUT_ROOT / "full_modality_percent_summary_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "full_modality_percent_summary.csv", index=False)
    for metric in ("r@1", "r@5", "r@10", "top_1pct", "top_5pct", "top_10pct"):
        pivot = df.pivot_table(index=["family", "detector", "data_mode"], columns="variant", values=metric, aggfunc="first")
        pivot.to_csv(OUT_ROOT / f"{metric.replace('@', '')}_pivot.csv")
    meta = {
        "note": "Direct variants are pure scores. Rerank variants train a catalog-level HGB classifier on validation pairs and evaluate full test-catalog ranking.",
        "top_percent": "top_1pct/top_5pct/top_10pct use ceil((catalog_size-1)*percent) as cutoff.",
        "config_sources": {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in asdict(EXP_MODULES[k].make_cfg("ET", "pure")).items()} for k in EXP_MODULES},
    }
    (OUT_ROOT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
