from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from matchgw.pipeline import build_model, embed_eval
from matchgw.trigger_time import catalog_trigger_time_frame, log1p_delta_time_obs

aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

JOBS = [
    {'detector': 'ET', 'mode': 'pure', 'family': 'SIS', 'run': Path('runs/et10000_full_20260527_111510/SIS_pure_ep20_inceptiontime')},
    {'detector': 'ET', 'mode': 'pure', 'family': 'PM', 'run': Path('runs/et10000_full_20260527_111510/PM_pure_ep20_inceptiontime')},
    {'detector': 'ET', 'mode': 'noisy', 'family': 'SIS', 'run': Path('runs/et10000_bandpass_full_ep50_20260528_101013/SIS_noisy_bandpass_n10000_ep50')},
    {'detector': 'ET', 'mode': 'noisy', 'family': 'PM', 'run': Path('runs/et10000_bandpass_full_ep50_20260528_101013/PM_noisy_bandpass_n10000_ep50')},
    {'detector': 'LIGO', 'mode': 'pure', 'family': 'SIS', 'run': Path('runs/ligo_pure_inceptiontime_bandpass_full_ep50_20260601_103901/SIS_pure_inceptiontime_bandpass_n10000_ep50')},
    {'detector': 'LIGO', 'mode': 'pure', 'family': 'PM', 'run': Path('runs/ligo_pure_inceptiontime_bandpass_full_ep50_20260601_103901/PM_pure_inceptiontime_bandpass_n10000_ep50')},
    {'detector': 'LIGO', 'mode': 'noisy', 'family': 'SIS', 'run': Path('runs/ligo_inceptiontime_bandpass_full_ep50_20260601_095620/SIS_noisy_inceptiontime_bandpass_n10000_ep50')},
    {'detector': 'LIGO', 'mode': 'noisy', 'family': 'PM', 'run': Path('runs/ligo_inceptiontime_bandpass_full_ep50_20260601_095620/PM_noisy_inceptiontime_bandpass_n10000_ep50')},
]

MODE = 'realistic'
NEG_PER_POS = 500
CHUNK_ROWS = 64
MIN_SKY_SIGMA = 0.03
OUT_ROOT = Path('runs/waveform_predicted_skymap_rerank_20260602')


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for k in ['data_root', 'out_dir']:
        if k in kwargs:
            kwargs[k] = Path(kwargs[k])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def unit_vectors(obs: pd.DataFrame) -> np.ndarray:
    ra = obs['ra'].to_numpy(dtype=np.float64)
    dec = obs['dec'].to_numpy(dtype=np.float64)
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float32)


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return (x / norm).astype(np.float32)


def angular_sep_from_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def log_gaussian_skymap_overlap_from_unit(mu_i: np.ndarray, mu_j: np.ndarray, sigma_i: float, sigma_j: float) -> np.ndarray:
    sep = angular_sep_from_unit(mu_i, mu_j)
    var = sigma_i * sigma_i + sigma_j * sigma_j
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)



def baseline_metrics(run_dir: Path) -> dict:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    t = data.get('test', {})
    return {
        'waveform_r@1': t.get('r@1'),
        'waveform_r@5': t.get('r@5'),
        'waveform_r@10': t.get('r@10'),
        'waveform_median_true_rank': t.get('median_true_rank'),
    }


def load_split(job: dict, split: str, out_dir: Path):
    run_dir = job['run']
    cfg = cfg_from_run(run_dir, out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits['lensed'][split], splits['unlensed'][split], cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs = aux.catalog_observable_frame(cfg.data_root, job['family'], splits['lensed'][split], splits['unlensed'][split]).reset_index(drop=True)
    time_obs = catalog_trigger_time_frame(cfg.data_root, job['family'], splits['lensed'][split], splits['unlensed'][split], detector=job['detector']).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = out_dir / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_embeddings.npy"
    score_cache = out_dir / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_scores.npy"
    if emb_cache.exists() and score_cache.exists():
        emb = np.load(emb_cache)
        scores = np.load(score_cache)
    else:
        model = build_model(cfg)
        ckpt = torch.load(run_dir / 'model.pt', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        emb = embed_eval(model, ds, cfg, cpu=False).astype(np.float32)
        scores = similarity_matrix(emb).astype(np.float32)
        np.fill_diagonal(scores, -np.inf)
        np.save(emb_cache, emb)
        np.save(score_cache, scores)
    return raw_obs, time_obs, gt, emb, scores


def row_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int32)
    return ranks


def fit_sky_predictor(train_obs: pd.DataFrame, train_emb: np.ndarray, val_obs: pd.DataFrame, val_emb: np.ndarray) -> tuple[object, float, float]:
    y_train = unit_vectors(train_obs)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 3, 12)),
    )
    model.fit(train_emb, y_train)
    val_pred = normalize_vectors(model.predict(val_emb))
    val_true = unit_vectors(val_obs)
    err = angular_sep_from_unit(val_pred, val_true)
    sigma = float(max(MIN_SKY_SIGMA, np.median(err)))
    return model, sigma, float(np.mean(err))


def predict_sky_mu(model, emb: np.ndarray) -> np.ndarray:
    return normalize_vectors(model.predict(emb))


def feature_matrix(time_obs: pd.DataFrame, sky_mu: np.ndarray, sky_sigma: float, scores: np.ndarray, ranks: np.ndarray, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    return np.column_stack([
        log1p_delta_time_obs(time_obs, anchors, cands),
        log_gaussian_skymap_overlap_from_unit(sky_mu[anchors], sky_mu[cands], sky_sigma, sky_sigma),
        scores[anchors, cands],
        1.0 / np.maximum(ranks[anchors, cands], 1),
    ]).astype(np.float32)


def train_examples(time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray, sky_sigma: float, scores: np.ndarray, ranks: np.ndarray, job: dict):
    rng = np.random.default_rng(92000 + abs(hash((job['detector'], job['mode'], job['family']))) % 10000)
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    pos_a = valid.astype(np.int32)
    pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    anchors = np.concatenate([pos_a, neg_a])
    cands = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    X = feature_matrix(time_obs, sky_mu, sky_sigma, scores, ranks, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray, sky_sigma: float, scores: np.ndarray, ranks: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out_ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(time_obs, sky_mu, sky_sigma, scores, ranks, anchors, cands)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out_ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out_ranks)
    return {
        'rerank_r@1': float(np.mean(r <= 1)),
        'rerank_r@5': float(np.mean(r <= 5)),
        'rerank_r@10': float(np.mean(r <= 10)),
        'rerank_r@50': float(np.mean(r <= 50)),
        'rerank_r@100': float(np.mean(r <= 100)),
        'rerank_r@500': float(np.mean(r <= 500)),
        'rerank_median_true_rank': float(np.median(r)),
        'valid': int(len(valid)),
    }


def run_job(job: dict) -> dict:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('RUN', job, flush=True)
    train_raw, train_time, train_gt, train_emb, train_scores = load_split(job, 'train', out_dir / 'train')
    val_raw, val_time, val_gt, val_emb, val_scores = load_split(job, 'val', out_dir / 'val')
    test_raw, test_time, test_gt, test_emb, test_scores = load_split(job, 'test', out_dir / 'test')

    sky_model, sky_sigma, sky_val_mean_error = fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = predict_sky_mu(sky_model, val_emb)
    test_sky_mu = predict_sky_mu(sky_model, test_emb)

    val_ranks = row_ranks(val_scores)
    test_ranks = row_ranks(test_scores)
    Xv, yv = train_examples(val_time, val_gt, val_sky_mu, sky_sigma, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'method': 'waveform_predicted_skymap_overlap_catalog_rerank',
        'features': 'log1p_delta_time_obs,predicted_log_sky_map_overlap,waveform_score,waveform_reciprocal_rank',
        'sky_predictor': 'RidgeCV waveform_embedding_to_sky_unit_vector',
        'sky_sigma_rad': sky_sigma,
        'sky_val_mean_angular_error_rad': sky_val_mean_error,
        'note': 'ra/dec are used only as supervised labels for training/evaluating the sky-map predictor. Time input uses trigger_time_obs-derived delta_time_obs, not true geocent_time or lens t_d; sky input uses waveform-predicted sky maps, not direct ra/dec or sky_sep.',
        'run_dir': str(job['run']),
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **baseline_metrics(job['run']),
        **eval_full(clf, test_time, test_gt, test_sky_mu, sky_sigma, test_scores, test_ranks),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(row, flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in JOBS:
        rows.append(run_job(job))
        pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    (OUT_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
