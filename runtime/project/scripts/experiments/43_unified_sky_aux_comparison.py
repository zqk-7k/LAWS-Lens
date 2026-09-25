from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

OUT_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
NEG_PER_POS = 500
CHUNK_ROWS = 64
REAL_SKY_SIGMA_RAD = 0.08


def stable_seed(*parts: object) -> int:
    text = '|'.join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)


def real_log_sky_overlap(raw_obs: pd.DataFrame, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    ra = raw_obs['ra'].to_numpy(dtype=np.float64)
    dec = raw_obs['dec'].to_numpy(dtype=np.float64)
    sep = aux.angular_sep(ra[anchors], dec[anchors], ra[cands], dec[cands])
    var = REAL_SKY_SIGMA_RAD * REAL_SKY_SIGMA_RAD * 2.0
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def real_sky_sep(raw_obs: pd.DataFrame, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    ra = raw_obs['ra'].to_numpy(dtype=np.float64)
    dec = raw_obs['dec'].to_numpy(dtype=np.float64)
    return aux.angular_sep(ra[anchors], dec[anchors], ra[cands], dec[cands]).astype(np.float32)


def feature_matrix(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    cols = [
        scores[anchors, cands].astype(np.float32),
        (1.0 / np.maximum(ranks[anchors, cands], 1)).astype(np.float32),
    ]
    if variant != 'waveform_only':
        cols.insert(0, base.log1p_delta_time_obs(time_obs, anchors, cands).astype(np.float32))
    if variant == 'trigger_time_plus_real_sky_sep':
        cols.append(real_sky_sep(raw_obs, anchors, cands))
    elif variant == 'trigger_time_plus_real_sky_overlap':
        cols.append(real_log_sky_overlap(raw_obs, anchors, cands))
    elif variant == 'trigger_time_plus_predicted_sky_overlap':
        if sky_mu is None or sky_sigma is None:
            raise ValueError('predicted sky overlap requires sky_mu and sky_sigma')
        cols.append(base.log_gaussian_skymap_overlap_from_unit(sky_mu[anchors], sky_mu[cands], sky_sigma, sky_sigma))
    return np.column_stack(cols).astype(np.float32)


def feature_names(variant: str) -> str:
    names = ['waveform_score', 'waveform_reciprocal_rank']
    if variant != 'waveform_only':
        names.insert(0, 'log1p_delta_time_obs')
    if variant == 'trigger_time_plus_real_sky_sep':
        names.append('oracle_sky_sep_from_ra_dec')
    elif variant == 'trigger_time_plus_real_sky_overlap':
        names.append('oracle_log_sky_map_overlap_from_true_ra_dec')
    elif variant == 'trigger_time_plus_predicted_sky_overlap':
        names.append('predicted_log_sky_map_overlap')
    return ','.join(names)


def train_examples(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, job: dict):
    rng = np.random.default_rng(stable_seed('train', variant, job['detector'], job['mode'], job['family']))
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
    X = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(variant: str, clf, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out_ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, anchors, cands)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out_ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out_ranks)
    return {
        'r@1': float(np.mean(r <= 1)),
        'r@5': float(np.mean(r <= 5)),
        'r@10': float(np.mean(r <= 10)),
        'r@50': float(np.mean(r <= 50)),
        'r@100': float(np.mean(r <= 100)),
        'r@500': float(np.mean(r <= 500)),
        'median_rank': float(np.median(r)),
        'valid': int(len(valid)),
    }


def run_variant(job: dict, variant: str, val_pack: tuple, test_pack: tuple, sky_sigma: float | None, sky_val_mean_error: float | None) -> dict:
    val_raw, val_time, val_gt, val_sky_mu, val_scores, val_ranks = val_pack
    test_raw, test_time, test_gt, test_sky_mu, test_scores, test_ranks = test_pack
    Xv, yv = train_examples(variant, val_raw, val_time, val_gt, val_sky_mu, sky_sigma, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'variant': variant,
        'method': 'unified_catalog_rerank_hgb',
        'features': feature_names(variant),
        'oracle_feature': bool('real_sky' in variant),
        'sky_sigma_rad': sky_sigma if variant == 'trigger_time_plus_predicted_sky_overlap' else (REAL_SKY_SIGMA_RAD if variant == 'trigger_time_plus_real_sky_overlap' else np.nan),
        'sky_val_mean_angular_error_rad': sky_val_mean_error if variant == 'trigger_time_plus_predicted_sky_overlap' else np.nan,
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **eval_full(variant, clf, test_raw, test_time, test_gt, test_sky_mu, sky_sigma, test_scores, test_ranks),
    }
    return row


def run_job(job: dict) -> list[dict]:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD', name, flush=True)
    train_raw, train_time, train_gt, train_emb, train_scores = base.load_split(job, 'train', out_dir / 'train')
    val_raw, val_time, val_gt, val_emb, val_scores = base.load_split(job, 'val', out_dir / 'val')
    test_raw, test_time, test_gt, test_emb, test_scores = base.load_split(job, 'test', out_dir / 'test')
    sky_model, sky_sigma, sky_val_mean_error = base.fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = base.predict_sky_mu(sky_model, val_emb)
    test_sky_mu = base.predict_sky_mu(sky_model, test_emb)
    val_pack = (val_raw, val_time, val_gt, val_sky_mu, val_scores, base.row_ranks(val_scores))
    test_pack = (test_raw, test_time, test_gt, test_sky_mu, test_scores, base.row_ranks(test_scores))
    variants = [
        'waveform_only',
        'trigger_time_only',
        'trigger_time_plus_real_sky_sep',
        'trigger_time_plus_real_sky_overlap',
        'trigger_time_plus_predicted_sky_overlap',
    ]
    rows = []
    for variant in variants:
        print('RUN', name, variant, flush=True)
        row = run_variant(job, variant, val_pack, test_pack, sky_sigma, sky_val_mean_error)
        rows.append(row)
        print(row, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'summary.csv', index=False)
    (out_dir / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in base.JOBS:
        rows.extend(run_job(job))
        pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    (OUT_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    pivot = df.pivot_table(index=['detector', 'data_mode', 'family'], columns='variant', values='r@1', aggfunc='first')
    pivot.to_csv(OUT_ROOT / 'r1_pivot.csv')
    print(df.to_string(index=False), flush=True)
    print('\nR@1 pivot\n', pivot.to_string(), flush=True)


if __name__ == '__main__':
    main()
