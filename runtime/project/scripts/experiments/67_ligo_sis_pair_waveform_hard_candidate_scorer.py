from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

m = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base = m.base
OUT_ROOT = Path('runs/ligo_sis_pair_waveform_hard_candidate_scorer_20260605')
PROB_ROOT = Path('runs/ligo_sis_grid18_rank_fusion_20260604')
EPS = 1e-8
ALPHA = 2.0
K_EACH = 100
CHUNK_ROWS = 32
DOWNSAMPLE = 64
MAX_LAG_BINS = 4


def sharpen(prob, alpha):
    p = np.power(np.maximum(prob, EPS), alpha).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p


def load_split(job, split):
    _, ds, raw, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    prob = sharpen(np.load(PROB_ROOT / f'{split}_prob.npy').astype(np.float32), ALPHA)
    ranks = base.row_ranks(scores)
    return ds, time_obs.reset_index(drop=True), gt, scores.astype(np.float32), ranks.astype(np.int32), prob


def dt_mat(time_obs, rows):
    n = len(time_obs)
    allc = np.arange(n, dtype=np.int32)
    out = np.empty((len(rows), n), np.float32)
    for i, a in enumerate(rows):
        out[i] = -m.log1p_delta_time_obs(time_obs, np.full(n, int(a), dtype=np.int32), allc)
    return out


def candidate_sets(time_obs, scores, prob, rows):
    sky = np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)
    dt = dt_mat(time_obs, rows)
    cand_list = []
    for rp, a in enumerate(rows):
        cand = set()
        for arr in [scores[a].astype(np.float32), sky[rp], dt[rp]]:
            s = arr.copy(); s[int(a)] = -np.inf
            top = np.argpartition(-s, min(K_EACH, len(s)-1))[:K_EACH]
            cand.update(map(int, top))
        cand.discard(int(a))
        cand_list.append(np.array(sorted(cand), dtype=np.int32))
    return cand_list, sky, dt


def get_wave(ds, idx):
    x = ds[int(idx)]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.float32)


def compress_wave(x):
    c, t = x.shape
    trim = (t // DOWNSAMPLE) * DOWNSAMPLE
    y = x[:, :trim].reshape(c, -1, DOWNSAMPLE).mean(axis=2)
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (y.std(axis=1, keepdims=True) + 1e-6)
    return y.astype(np.float32)


def wave_pair_features(a, b):
    feats = []
    for ch in range(a.shape[0]):
        aa = a[ch]; bb = b[ch]
        vals = []
        for lag in range(-MAX_LAG_BINS, MAX_LAG_BINS + 1):
            if lag < 0:
                x = aa[:lag]; y = bb[-lag:]
            elif lag > 0:
                x = aa[lag:]; y = bb[:-lag]
            else:
                x = aa; y = bb
            vals.append(float(np.mean(x * y)))
        vals = np.asarray(vals, dtype=np.float32)
        imax = int(np.argmax(vals))
        feats.extend([float(vals[imax]), float(imax - MAX_LAG_BINS), float(vals[MAX_LAG_BINS]), float(vals.max() - vals.mean())])
        feats.extend([
            float(np.mean(np.abs(aa - bb))),
            float(np.mean((aa - bb) ** 2)),
            float(np.max(np.abs(aa)) / (np.max(np.abs(bb)) + 1e-6)),
            float(np.sqrt(np.mean(aa**2)) / (np.sqrt(np.mean(bb**2)) + 1e-6)),
        ])
    feats.extend([
        float(np.mean((a[0] - a[1]) * (b[0] - b[1]))),
        float(np.mean(np.abs((a[0] - a[1]) - (b[0] - b[1])))),
        float(np.mean(np.abs(a - b))),
        float(np.mean((a - b) ** 2)),
    ])
    return feats


def pair_features_for(ds, time_obs, scores, ranks, prob, a, cols, sky_row=None, dt_row=None):
    wa = compress_wave(get_wave(ds, a))
    if sky_row is None:
        sky_row = np.log(np.maximum(prob[int(a):int(a)+1] @ prob.T, EPS)).reshape(-1)
    if dt_row is None:
        dt_row = dt_mat(time_obs, np.array([a], dtype=np.int32))[0]
    out = []
    for c in cols:
        wc = compress_wave(get_wave(ds, int(c)))
        f = wave_pair_features(wa, wc)
        f.extend([
            float(scores[a, c]),
            float(1.0 / max(int(ranks[a, c]), 1)),
            float(-np.log1p(int(ranks[a, c]))),
            float(sky_row[c]),
            float(dt_row[c]),
            float(sky_row[c] * dt_row[c]),
        ])
        out.append(f)
    return np.asarray(out, dtype=np.float32)


def build_train(ds, time_obs, gt, scores, ranks, prob):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    cand_list, sky, dt = candidate_sets(time_obs, scores, prob, valid)
    Xs, ys = [], []
    for rp, a in enumerate(valid):
        p = int(gt[a])
        cand = cand_list[rp]
        if p not in set(map(int, cand)):
            cand = np.concatenate([cand, np.array([p], dtype=np.int32)])
        Xs.append(pair_features_for(ds, time_obs, scores, ranks, prob, int(a), cand, sky[rp], dt[rp]))
        ys.append((cand == p).astype(np.int8))
        if (rp + 1) % 250 == 0:
            print('TRAIN_FEATS', rp + 1, flush=True)
    X = np.vstack(Xs); y = np.concatenate(ys)
    rng = np.random.default_rng(67001); order = rng.permutation(len(y))
    return X[order], y[order]


def eval_candidate(clf, ds, time_obs, gt, scores, ranks, prob):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    all_ranks, in_cand = [], []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start+CHUNK_ROWS]
        cand_list, sky, dt = candidate_sets(time_obs, scores, prob, rows)
        for rp, a in enumerate(rows):
            p = int(gt[a]); cand = cand_list[rp]
            hit = p in set(map(int, cand)); in_cand.append(hit)
            if not hit:
                all_ranks.append(len(cand) + 1); continue
            X = pair_features_for(ds, time_obs, scores, ranks, prob, int(a), cand, sky[rp], dt[rp])
            pred = clf.predict_proba(X)[:, 1]
            true = pred[np.where(cand == p)[0][0]]
            all_ranks.append(int(1 + np.sum(pred > true)))
        print('EVAL_ROWS', start + len(rows), flush=True)
    r = np.asarray(all_ranks)
    return {'candidate_recall': float(np.mean(in_cand)), 'r@1': float(np.mean(r <= 1)), 'r@5': float(np.mean(r <= 5)), 'r@10': float(np.mean(r <= 10)), 'r@50': float(np.mean(r <= 50)), 'median_rank': float(np.median(r)), 'valid': int(len(valid))}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    train_ds, train_time, train_gt, train_scores, train_ranks, train_prob = load_split(job, 'val')
    test_ds, test_time, test_gt, test_scores, test_ranks, test_prob = load_split(job, 'test')
    X, y = build_train(train_ds, train_time, train_gt, train_scores, train_ranks, train_prob)
    print('TRAIN_SHAPE', X.shape, 'pos', int(y.sum()), 'neg', int((y == 0).sum()), flush=True)
    models = {
        'hgb_pair_waveform_top100': HistGradientBoostingClassifier(max_iter=240, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=1e-3, class_weight='balanced', random_state=67),
        'logreg_pair_waveform_top100': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)),
    }
    rows = []
    for name, clf in models.items():
        print('FIT', name, flush=True)
        clf.fit(X, y)
        met = eval_candidate(clf, test_ds, test_time, test_gt, test_scores, test_ranks, test_prob)
        row = {'method': name, 'feature_dim': int(X.shape[1]), **met}
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
