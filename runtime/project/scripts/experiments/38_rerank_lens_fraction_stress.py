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
OUT_ROOT = Path('runs/rerank_lens_fraction_10pct_20260601')
MODE = 'realistic'
TARGET_LENSED_EVENT_FRAC = 0.10
SEEDS = list(range(10))
NEG_PER_POS = 500
CHUNK_ROWS = 64


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
    return {'full_waveform_r@1': t.get('r@1'), 'full_waveform_r@5': t.get('r@5'), 'full_waveform_r@10': t.get('r@10')}


def load_full_split(job: dict, split: str, out_dir: Path):
    cfg = cfg_from_run(job['run'], out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    lidx = splits['lensed'][split]
    uidx = splits['unlensed'][split]
    ds = EvaluationSet(arrays, lidx, uidx, cfg)
    gt = ground_truth_partner(ds.meta)
    obs = aux.catalog_observable_frame(cfg.data_root, job['family'], lidx, uidx)
    obs = aux.perturb_observables(obs, MODE, seed=91000 + abs(hash((job['detector'], job['mode'], job['family'], split))) % 10000).reset_index(drop=True)
    cache = out_dir / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_scores.npy"
    if cache.exists():
        scores = np.load(cache)
    else:
        model = build_model(cfg)
        ckpt = torch.load(job['run'] / 'model.pt', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        scores = similarity_matrix(embed_eval(model, ds, cfg, cpu=False)).astype(np.float32)
        np.fill_diagonal(scores, -np.inf)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, scores)
    return obs, gt, scores, len(lidx), len(uidx)


def sample_catalog(obs: pd.DataFrame, gt: np.ndarray, scores: np.ndarray, n_pairs_full: int, n_unlensed_full: int, seed: int):
    rng = np.random.default_rng(seed)
    # With all available unlensed events, choose pair count to make 2*pairs/(2*pairs+unlensed) ~= target fraction.
    pair_count = int(round(TARGET_LENSED_EVENT_FRAC * n_unlensed_full / (2.0 * (1.0 - TARGET_LENSED_EVENT_FRAC))))
    pair_count = max(1, min(pair_count, n_pairs_full))
    pair_ids = np.sort(rng.choice(n_pairs_full, size=pair_count, replace=False))
    unlensed_ids = np.arange(n_unlensed_full, dtype=np.int64)
    full_idx = np.concatenate([pair_ids, n_pairs_full + pair_ids, 2 * n_pairs_full + unlensed_ids])
    sub_obs = obs.iloc[full_idx].reset_index(drop=True)
    sub_scores = scores[np.ix_(full_idx, full_idx)].copy()
    np.fill_diagonal(sub_scores, -np.inf)
    index_map = {int(old): i for i, old in enumerate(full_idx)}
    sub_gt = np.full(len(full_idx), -1, dtype=np.int64)
    for old, new in index_map.items():
        partner = int(gt[old])
        if partner in index_map:
            sub_gt[new] = index_map[partner]
    lensed_events = int(np.sum(sub_gt >= 0))
    return sub_obs, sub_gt, sub_scores, {'pair_count': pair_count, 'unlensed_count': int(n_unlensed_full), 'catalog_size': int(len(full_idx)), 'lensed_event_fraction': float(lensed_events / len(full_idx))}


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


def eval_scores(scores: np.ndarray, gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    order = np.argsort(-scores, axis=1)
    ranks = []
    for i in valid:
        pos = np.where(order[i] == gt[i])[0]
        ranks.append(int(pos[0]) + 1 if len(pos) else 10**9)
    r = np.asarray(ranks)
    return {'sub_waveform_r@1': float(np.mean(r <= 1)), 'sub_waveform_r@5': float(np.mean(r <= 5)), 'sub_waveform_r@10': float(np.mean(r <= 10)), 'sub_waveform_r@50': float(np.mean(r <= 50)), 'sub_waveform_median_rank': float(np.median(r)), 'valid': int(len(valid))}


def train_examples(obs: pd.DataFrame, gt: np.ndarray, scores: np.ndarray, ranks: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
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


def eval_rerank(clf, obs: pd.DataFrame, gt: np.ndarray, scores: np.ndarray, ranks: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    n = len(obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        pred = clf.predict_proba(feature_matrix(obs, scores, ranks, anchors, cands))[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out)
    return {'rerank_r@1': float(np.mean(r <= 1)), 'rerank_r@5': float(np.mean(r <= 5)), 'rerank_r@10': float(np.mean(r <= 10)), 'rerank_r@50': float(np.mean(r <= 50)), 'rerank_median_rank': float(np.median(r))}


def run_job(job: dict) -> list[dict]:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print('LOAD', job, flush=True)
    vobs, vgt, vscores, vp, vu = load_full_split(job, 'val', out_dir / 'val')
    tobs, tgt, tscores, tp, tu = load_full_split(job, 'test', out_dir / 'test')
    rows = []
    for seed in SEEDS:
        svobs, svgt, svscores, vmeta = sample_catalog(vobs, vgt, vscores, vp, vu, seed=1000 + seed)
        stobs, stgt, stscores, tmeta = sample_catalog(tobs, tgt, tscores, tp, tu, seed=2000 + seed)
        vranks = row_ranks(svscores); tranks = row_ranks(stscores)
        Xv, yv = train_examples(svobs, svgt, svscores, vranks, seed=3000 + seed)
        clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
        clf.fit(Xv, yv)
        pv = clf.predict_proba(Xv)[:, 1]
        row = {'detector': job['detector'], 'data_mode': job['mode'], 'family': job['family'], 'seed': seed, 'target_lensed_event_fraction': TARGET_LENSED_EVENT_FRAC, 'val_auc_sampled': float(roc_auc_score(yv, pv)), **baseline_metrics(job['run']), **{f'test_{k}': v for k, v in tmeta.items()}, **eval_scores(stscores, stgt), **eval_rerank(clf, stobs, stgt, stscores, tranks)}
        print(row, flush=True)
        rows.append(row)
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in JOBS:
        rows.extend(run_job(job))
        pd.DataFrame(rows).to_csv(OUT_ROOT / 'trials_partial.csv', index=False)
    trials = pd.DataFrame(rows)
    trials.to_csv(OUT_ROOT / 'trials.csv', index=False)
    metric_cols = ['sub_waveform_r@1','sub_waveform_r@5','sub_waveform_r@10','sub_waveform_r@50','rerank_r@1','rerank_r@5','rerank_r@10','rerank_r@50','rerank_median_rank','test_lensed_event_fraction','test_catalog_size','test_pair_count','test_unlensed_count']
    summary = trials.groupby(['detector','data_mode','family'])[metric_cols].agg(['mean','std']).reset_index()
    summary.columns = ['_'.join([c for c in col if c]) for col in summary.columns.to_flat_index()]
    summary.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
