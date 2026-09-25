from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.trigger_time import catalog_trigger_time_frame, log1p_delta_time_obs

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')
skyfeat = importlib.import_module('scripts.experiments.44_waveform_feature_skymap_predictor')

OUT_ROOT = Path('runs/ligo_pair_sky_overlap_predictor_20260604')
SOURCE_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
FEATURE_ROOT = Path('runs/ligo_waveform_feature_skymap_predictor_light_20260604')
JOBS = [j for j in base.JOBS if j['detector'] == 'LIGO' and j['mode'] == 'noisy']
NEG_PER_POS_OVERLAP = 200
NEG_PER_POS_RERANK = 500
PCA_DIM = 64
CHUNK_ROWS = 32
REAL_SKY_SIGMA_RAD = 0.08


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode('utf-8')).hexdigest()[:8], 16)


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for key in ['data_root', 'out_dir']:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def cache_name(job: dict, split: str, suffix: str) -> Path:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    return SOURCE_ROOT / name / split / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_{suffix}.npy"


def pack(job: dict, split: str, out_dir: Path, need_scores: bool):
    cfg = cfg_from_run(job['run'], out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    lidx = splits['lensed'][split]
    uidx = splits['unlensed'][split]
    ds = EvaluationSet(arrays, lidx, uidx, cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs = aux.catalog_observable_frame(cfg.data_root, job['family'], lidx, uidx).reset_index(drop=True)
    time_obs = catalog_trigger_time_frame(cfg.data_root, job['family'], lidx, uidx, detector=job['detector']).reset_index(drop=True)
    emb = np.load(cache_name(job, split, 'embeddings'))
    scores = np.load(cache_name(job, split, 'scores')) if need_scores else None
    return cfg, ds, raw_obs, time_obs, gt, emb, scores


def stats(job: dict, split: str, ds: EvaluationSet, out_dir: Path) -> np.ndarray:
    return skyfeat.extract_waveform_features(job, split, ds, out_dir / 'features')


def unit_vectors(obs: pd.DataFrame) -> np.ndarray:
    return base.unit_vectors(obs).astype(np.float32)


def real_log_overlap_from_unit(unit: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    sep = base.angular_sep_from_unit(unit[a], unit[c])
    var = 2.0 * REAL_SKY_SIGMA_RAD * REAL_SKY_SIGMA_RAD
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def fit_event_feature_model(train_stats: np.ndarray, train_emb: np.ndarray):
    X = np.concatenate([train_stats.astype(np.float32), train_emb.astype(np.float32)], axis=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=PCA_DIM, svd_solver='randomized', random_state=42)
    Z = pca.fit_transform(Xs).astype(np.float32)
    return scaler, pca, Z


def transform_event_features(scaler, pca, stats_x: np.ndarray, emb: np.ndarray) -> np.ndarray:
    X = np.concatenate([stats_x.astype(np.float32), emb.astype(np.float32)], axis=1)
    return pca.transform(scaler.transform(X)).astype(np.float32)


def pair_features(Z: np.ndarray, scores: np.ndarray | None, ranks: np.ndarray | None, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    za = Z[a]
    zc = Z[c]
    cols = [np.abs(za - zc), za * zc]
    if scores is not None and ranks is not None:
        cols.append(scores[a, c][:, None].astype(np.float32))
        cols.append((1.0 / np.maximum(ranks[a, c], 1))[:, None].astype(np.float32))
    return np.concatenate(cols, axis=1).astype(np.float32)


def sample_overlap_pairs(gt: np.ndarray, n: int, job: dict, split: str):
    rng = np.random.default_rng(stable_seed('overlap_pairs', job['family'], split))
    valid = np.flatnonzero(gt >= 0)
    pos_a = valid.astype(np.int32)
    pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS_OVERLAP)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    a = np.concatenate([pos_a, neg_a])
    c = np.concatenate([pos_c, neg_c])
    order = rng.permutation(len(a))
    return a[order], c[order]


def fit_overlap_model(Z: np.ndarray, raw_obs: pd.DataFrame, gt: np.ndarray, job: dict):
    a, c = sample_overlap_pairs(gt, len(raw_obs), job, 'train')
    unit = unit_vectors(raw_obs)
    y = real_log_overlap_from_unit(unit, a, c)
    # Clip extreme negative tails so the regressor focuses on ranking useful near-sky candidates.
    y = np.clip(y, -80.0, 4.0)
    X = pair_features(Z, None, None, a, c)
    reg = HistGradientBoostingRegressor(max_iter=260, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1e-4, random_state=42)
    reg.fit(X, y)
    return reg


def predicted_overlap(reg, Z: np.ndarray, scores: np.ndarray, ranks: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return reg.predict(pair_features(Z, None, None, a, c)).astype(np.float32)


def rerank_features(time_obs, pred_ov: np.ndarray, scores, ranks, a, c) -> np.ndarray:
    return np.column_stack([
        log1p_delta_time_obs(time_obs, a, c),
        pred_ov,
        scores[a, c].astype(np.float32),
        (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32),
    ]).astype(np.float32)


def train_reranker(time_obs, gt, Z, scores, ranks, reg, job):
    rng = np.random.default_rng(stable_seed('rerank', job['family']))
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    pos_a = valid.astype(np.int32)
    pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS_RERANK)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    a = np.concatenate([pos_a, neg_a])
    c = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    pov = predicted_overlap(reg, Z, scores, ranks, a, c)
    X = rerank_features(time_obs, pov, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs, gt, Z, scores, ranks, reg):
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        a = np.repeat(rows.astype(np.int32), n)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        pov = predicted_overlap(reg, Z, scores, ranks, a, c)
        X = rerank_features(time_obs, pov, scores, ranks, a, c)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out)
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


def run_job(job: dict):
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD', name, flush=True)
    _, train_ds, train_raw, train_time, train_gt, train_emb, _ = pack(job, 'train', out_dir / 'train', False)
    _, val_ds, val_raw, val_time, val_gt, val_emb, val_scores = pack(job, 'val', out_dir / 'val', True)
    _, test_ds, test_raw, test_time, test_gt, test_emb, test_scores = pack(job, 'test', out_dir / 'test', True)
    train_stats = stats(job, 'train', train_ds, FEATURE_ROOT / name)
    val_stats = stats(job, 'val', val_ds, FEATURE_ROOT / name)
    test_stats = stats(job, 'test', test_ds, FEATURE_ROOT / name)
    scaler, pca, train_Z = fit_event_feature_model(train_stats, train_emb)
    val_Z = transform_event_features(scaler, pca, val_stats, val_emb)
    test_Z = transform_event_features(scaler, pca, test_stats, test_emb)
    print('FIT_OVERLAP', name, flush=True)
    reg = fit_overlap_model(train_Z, train_raw, train_gt, job)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    print('FIT_RERANK', name, flush=True)
    Xv, yv = train_reranker(val_time, val_gt, val_Z, val_scores, val_ranks, reg, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'method': 'pair_level_predicted_sky_overlap_catalog_rerank',
        'features': 'log1p_delta_time_obs,pair_predicted_log_sky_overlap,waveform_score,waveform_reciprocal_rank',
        'event_feature': f'waveform_stats_plus_embedding_pca{PCA_DIM}',
        'overlap_model': 'HistGradientBoostingRegressor',
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **eval_full(clf, test_time, test_gt, test_Z, test_scores, test_ranks, reg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    pd.DataFrame([row]).to_csv(out_dir / 'summary.csv', index=False)
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
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
