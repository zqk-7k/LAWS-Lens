from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

m = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base = m.base
aux = m.aux
OUT_ROOT = Path('runs/ligo_sis_skymap_temperature_sharpening_20260605')
PROB_ROOT = Path('runs/ligo_sis_grid18_rank_fusion_20260604')
SRC_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
EPS = 1e-8
ALPHAS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
NEG_PER_POS = 500
CHUNK_ROWS = 64


def load_pack(job, split):
    _, ds, raw, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    prob_path = PROB_ROOT / f'{split}_prob.npy'
    if not prob_path.exists():
        raise FileNotFoundError(prob_path)
    prob = np.load(prob_path).astype(np.float32)
    ranks = base.row_ranks(scores)
    return raw, time_obs.reset_index(drop=True), gt, scores.astype(np.float32), ranks.astype(np.int32), prob


def sharpen(prob, alpha):
    if alpha == 1.0:
        return prob.astype(np.float32)
    p = np.power(np.maximum(prob, EPS), alpha).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p


def overlap(prob, a, c):
    return np.log(np.sum(prob[a] * prob[c], axis=1) + EPS).astype(np.float32)


def feature_matrix(time_obs, prob, scores, ranks, a, c):
    return np.column_stack([
        m.log1p_delta_time_obs(time_obs, a, c),
        overlap(prob, a, c),
        scores[a, c].astype(np.float32),
        (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32),
    ]).astype(np.float32)


def train_reranker(time_obs, gt, prob, scores, ranks):
    rng = np.random.default_rng(63001)
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
    X = feature_matrix(time_obs, prob, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs, gt, prob, scores, ranks):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start+CHUNK_ROWS]
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(time_obs, prob, scores, ranks, a, c)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out)
    return {'r@1': float(np.mean(r <= 1)), 'r@5': float(np.mean(r <= 5)), 'r@10': float(np.mean(r <= 10)), 'r@50': float(np.mean(r <= 50)), 'r@100': float(np.mean(r <= 100)), 'r@500': float(np.mean(r <= 500)), 'median_rank': float(np.median(r)), 'valid': int(len(valid))}


def pair_overlap_quality(prob, gt):
    rng = np.random.default_rng(63002)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    pos = np.sum(prob[valid] * prob[gt[valid].astype(int)], axis=1)
    n = len(prob)
    a = rng.choice(valid, size=200000, replace=True)
    c = rng.integers(0, n, size=len(a), dtype=np.int32)
    bad = (c == a) | (c == gt[a])
    while bad.any():
        c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (c == a) | (c == gt[a])
    neg = np.sum(prob[a] * prob[c], axis=1)
    pp = rng.choice(pos, size=len(neg), replace=True)
    auc = float(np.mean(pp > neg) + 0.5 * np.mean(pp == neg))
    entropy = -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=1) / np.log(prob.shape[1])
    return {
        'entropy_norm_mean': float(np.mean(entropy)),
        'pos_overlap_mean': float(np.mean(pos)),
        'neg_overlap_mean': float(np.mean(neg)),
        'overlap_ratio': float(np.mean(pos) / max(float(np.mean(neg)), EPS)),
        'overlap_auc_sampled': auc,
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    _, val_time, val_gt, val_scores, val_ranks, val_prob0 = load_pack(job, 'val')
    _, test_time, test_gt, test_scores, test_ranks, test_prob0 = load_pack(job, 'test')
    rows = []
    for alpha in ALPHAS:
        val_prob = sharpen(val_prob0, alpha)
        test_prob = sharpen(test_prob0, alpha)
        X, y = train_reranker(val_time, val_gt, val_prob, val_scores, val_ranks)
        clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=63)
        clf.fit(X, y)
        pv = clf.predict_proba(X[:200000])[:, 1]
        auc = roc_auc_score(y[:len(pv)], pv)
        met = eval_full(clf, test_time, test_gt, test_prob, test_scores, test_ranks)
        q = pair_overlap_quality(test_prob, test_gt)
        row = {'alpha': alpha, 'sample_auc': float(auc), **q, **met}
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
