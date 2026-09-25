from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from matchgw.pipeline import build_model, embed_eval

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
OUT_ROOT = Path('runs/all_waveform_time_sky_rerank_20260601')


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for k in ['data_root', 'out_dir']:
        if k in kwargs:
            kwargs[k] = Path(kwargs[k])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


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
    obs = aux.catalog_observable_frame(cfg.data_root, job['family'], splits['lensed'][split], splits['unlensed'][split])
    obs = aux.perturb_observables(obs, MODE, seed=81000 + abs(hash((job['detector'], job['mode'], job['family'], split))) % 10000).reset_index(drop=True)
    cache = out_dir / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_scores.npy"
    if cache.exists():
        scores = np.load(cache)
    else:
        model = build_model(cfg)
        ckpt = torch.load(run_dir / 'model.pt', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        scores = similarity_matrix(embed_eval(model, ds, cfg, cpu=False)).astype(np.float32)
        np.fill_diagonal(scores, -np.inf)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, scores)
    return obs, gt, scores


def row_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int32)
    return ranks


def feature_matrix(obs: pd.DataFrame, scores: np.ndarray, ranks: np.ndarray, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    ra = obs['ra'].to_numpy(); dec = obs['dec'].to_numpy(); t = obs['geocent_time'].to_numpy()
    return np.column_stack([
        np.log1p(np.abs(t[anchors] - t[cands])),
        aux.angular_sep(ra[anchors], dec[anchors], ra[cands], dec[cands]),
        scores[anchors, cands],
        1.0 / np.maximum(ranks[anchors, cands], 1),
    ]).astype(np.float32)


def train_examples(obs: pd.DataFrame, gt: np.ndarray, scores: np.ndarray, ranks: np.ndarray, job: dict):
    rng = np.random.default_rng(82000 + abs(hash((job['detector'], job['mode'], job['family']))) % 10000)
    valid = np.flatnonzero(gt >= 0)
    n = len(obs)
    pos_a = valid.astype(np.int32); pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    anchors = np.concatenate([pos_a, neg_a])
    cands = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    X = feature_matrix(obs, scores, ranks, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, obs: pd.DataFrame, gt: np.ndarray, scores: np.ndarray, ranks: np.ndarray):
    valid = np.flatnonzero(gt >= 0)
    n = len(obs)
    out_ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(obs, scores, ranks, anchors, cands)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    print('RUN', job, flush=True)
    val_obs, val_gt, val_scores = load_split(job, 'val', out_dir / 'val')
    test_obs, test_gt, test_scores = load_split(job, 'test', out_dir / 'test')
    val_ranks = row_ranks(val_scores); test_ranks = row_ranks(test_scores)
    Xv, yv = train_examples(val_obs, val_gt, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'method': 'waveform_model_plus_delta_time_sky_sep_catalog_rerank',
        'features': 'log1p_delta_time,sky_sep,waveform_score,waveform_reciprocal_rank',
        'run_dir': str(job['run']),
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **baseline_metrics(job['run']),
        **eval_full(clf, test_obs, test_gt, test_scores, test_ranks),
    }
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
