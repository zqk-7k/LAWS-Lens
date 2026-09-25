from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')

OUT_ROOT = Path('runs/trigger_time_uncertainty_sweep_20260603')
SOURCE_ROOT = Path('runs/waveform_predicted_skymap_rerank_20260602')
NEG_PER_POS = 500
CHUNK_ROWS = 128
SCALES: list[float | str] = [1, 10, 100, 1000, 10000, 'randomized_time']


def load_cached(job: dict, split: str):
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    _, time_obs, gt, _, _ = base.load_split(job, split, SOURCE_ROOT / name / split)
    return time_obs.reset_index(drop=True), gt


def observed_time(time_obs: pd.DataFrame, scale: float | str, seed: int) -> np.ndarray:
    if scale == 1:
        return time_obs['trigger_time_obs'].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    base_t = time_obs['geocent_time_true'].to_numpy(dtype=float)
    sigma = time_obs['trigger_time_sigma'].to_numpy(dtype=float)
    if scale == 'randomized_time':
        return rng.permutation(time_obs['trigger_time_obs'].to_numpy(dtype=float))
    return base_t + rng.normal(0.0, sigma * float(scale), size=len(time_obs))


def feature_matrix(t: np.ndarray, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(t[anchors] - t[cands])).reshape(-1, 1).astype(np.float32)


def train_examples(t: np.ndarray, gt: np.ndarray, job: dict, scale: float | str):
    rng = np.random.default_rng(42000 + abs(hash((job['detector'], job['mode'], job['family'], str(scale)))) % 100000)
    valid = np.flatnonzero(gt >= 0)
    n = len(t)
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
    X = feature_matrix(t, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, t: np.ndarray, gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    n = len(t)
    ranks = []
    all_idx = np.arange(n, dtype=np.int32)
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(all_idx, len(rows))
        X = feature_matrix(t, anchors, cands)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(ranks)
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


def run_job(job: dict) -> list[dict]:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    print('LOAD', name, flush=True)
    val_time, val_gt = load_cached(job, 'val')
    test_time, test_gt = load_cached(job, 'test')
    rows = []
    for scale in SCALES:
        seed_val = 20260603 + abs(hash((name, 'val', str(scale)))) % 100000
        seed_test = 20260603 + abs(hash((name, 'test', str(scale)))) % 100000
        val_t = observed_time(val_time, scale, seed_val)
        test_t = observed_time(test_time, scale, seed_test)
        Xv, yv = train_examples(val_t, val_gt, job, scale)
        clf = HistGradientBoostingClassifier(
            max_iter=320,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1e-4,
            class_weight='balanced',
            random_state=42,
        )
        print('RUN', name, 'scale', scale, flush=True)
        clf.fit(Xv, yv)
        pv = clf.predict_proba(Xv)[:, 1]
        row = {
            'detector': job['detector'],
            'data_mode': job['mode'],
            'family': job['family'],
            'method': 'time_only_trigger_time_uncertainty',
            'sigma_scale': scale,
            'features': 'log1p_delta_time_obs',
            'val_auc': float(roc_auc_score(yv, pv)),
            **base.baseline_metrics(job['run']),
            **eval_full(clf, test_t, test_gt),
        }
        print(row, flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_ROOT / f'{name}_partial.csv', index=False)
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in base.JOBS:
        rows.extend(run_job(job))
        pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    (OUT_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    noisy = df[df['data_mode'].eq('noisy')]
    noisy.to_csv(OUT_ROOT / 'noisy_summary.csv', index=False)
    print('\nNOISY')
    print(noisy.to_string(index=False))


if __name__ == '__main__':
    main()
