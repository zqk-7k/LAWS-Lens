from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

m = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base = m.base

OUT_ROOT = Path('runs/ligo_sis_hard_negative_calibrated_rerank_20260605')
SRC_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
CACHE_ROOT = Path('runs/ligo_sis_grid18_rank_fusion_20260604')
EPS = 1e-8
TOP_WAVEFORM = 300
TOP_SKY = 300
TOP_TIME = 300
RANDOM_NEG = 200
CHUNK_ROWS = 48


def load_model(job):
    _, train_ds, _, _, _, _ = m.load_pack(job, 'train', SRC_ROOT / 'train', False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = m.SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    ckpt = torch.load(SRC_ROOT / 'ligo_noisy_sis' / 'grid_skymap_cnn.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, device


def load_split(job, split, model, device):
    cfg, ds, raw, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    cache = CACHE_ROOT / f'{split}_prob.npy'
    if cache.exists():
        prob = np.load(cache)
    else:
        prob = m.predict_maps(model, ds, device)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        np.save(OUT_ROOT / f'{split}_prob.npy', prob)
    ranks = base.row_ranks(scores)
    return raw, time_obs, gt, scores.astype(np.float32), ranks.astype(np.int32), prob.astype(np.float32)


def log_overlap_matrix(prob, rows=None):
    if rows is None:
        return np.log(np.maximum(prob @ prob.T, EPS)).astype(np.float32)
    return np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)


def dt_score_matrix(time_obs, rows):
    n = len(time_obs)
    all_c = np.arange(n, dtype=np.int32)
    out = np.empty((len(rows), n), dtype=np.float32)
    for i, a in enumerate(rows):
        aa = np.full(n, int(a), dtype=np.int32)
        out[i] = -m.log1p_delta_time_obs(time_obs, aa, all_c)
    return out


def row_rank_desc(mat):
    order = np.argsort(-mat, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    rr = np.arange(1, mat.shape[1] + 1, dtype=np.int32)
    ranks[np.arange(mat.shape[0])[:, None], order] = rr
    return ranks


def zrow_values(x):
    x = x.astype(np.float32, copy=True)
    finite = np.isfinite(x)
    if not finite.all():
        for i in range(x.shape[0]):
            ok = finite[i]
            fill = float(np.min(x[i, ok])) if ok.any() else 0.0
            x[i, ~ok] = fill
    return (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)


def build_full_feature_cube(time_obs, scores, wf_ranks, prob, rows):
    sky = log_overlap_matrix(prob, rows)
    dt = dt_score_matrix(time_obs, rows)
    wf = scores[rows].astype(np.float32)
    rr = (1.0 / np.maximum(wf_ranks[rows], 1)).astype(np.float32)
    sky_rank = row_rank_desc(sky)
    time_rank = row_rank_desc(dt)
    feats = {
        'wf': wf,
        'wf_z': zrow_values(wf),
        'recip_rank': rr,
        'sky': sky,
        'sky_z': zrow_values(sky),
        'dt': dt,
        'dt_z': zrow_values(dt),
        'sky_recip_rank': 1.0 / np.maximum(sky_rank, 1),
        'time_recip_rank': 1.0 / np.maximum(time_rank, 1),
        'wf_rank_log': -np.log1p(wf_ranks[rows].astype(np.float32)),
        'sky_rank_log': -np.log1p(sky_rank.astype(np.float32)),
        'time_rank_log': -np.log1p(time_rank.astype(np.float32)),
    }
    feats['sky_time_z'] = feats['sky_z'] * feats['dt_z']
    feats['wf_sky_z'] = feats['wf_z'] * feats['sky_z']
    feats['wf_time_z'] = feats['wf_z'] * feats['dt_z']
    return feats


FEATURE_KEYS = ['wf','wf_z','recip_rank','sky','sky_z','dt','dt_z','sky_recip_rank','time_recip_rank','wf_rank_log','sky_rank_log','time_rank_log','sky_time_z','wf_sky_z','wf_time_z']


def take_features(feats, row_pos, cols):
    return np.column_stack([feats[k][row_pos, cols] for k in FEATURE_KEYS]).astype(np.float32)


def train_data(time_obs, gt, scores, wf_ranks, prob):
    rng = np.random.default_rng(55001)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    feats = build_full_feature_cube(time_obs, scores, wf_ranks, prob, valid)
    Xs, ys = [], []
    n = len(time_obs)
    for rp, a in enumerate(valid):
        pos = int(gt[a])
        cols = {pos}
        # hard negatives from waveform, sky, and time rankings.
        for key, topn in [('wf', TOP_WAVEFORM), ('sky', TOP_SKY), ('dt', TOP_TIME)]:
            arr = feats[key][rp].copy()
            arr[a] = -np.inf
            arr[pos] = -np.inf
            top = np.argpartition(-arr, min(topn, n - 1))[:min(topn, n - 1)]
            cols.update(map(int, top))
        rand = rng.integers(0, n, size=RANDOM_NEG)
        cols.update(map(int, rand))
        cols.discard(int(a)); cols.discard(pos)
        neg = np.fromiter(cols, dtype=np.int32)
        pos_x = take_features(feats, rp, np.array([pos], dtype=np.int32))
        neg_x = take_features(feats, rp, neg)
        Xs.append(pos_x); ys.append(np.ones(1, dtype=np.int8))
        Xs.append(neg_x); ys.append(np.zeros(len(neg), dtype=np.int8))
    X = np.vstack(Xs); y = np.concatenate(ys)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs, gt, scores, wf_ranks, prob):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    all_ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        feats = build_full_feature_cube(time_obs, scores, wf_ranks, prob, rows)
        pred = np.empty((len(rows), n), dtype=np.float32)
        cols = np.arange(n, dtype=np.int32)
        for rp, a in enumerate(rows):
            X = take_features(feats, rp, cols)
            pred[rp] = clf.predict_proba(X)[:, 1]
            pred[rp, a] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        all_ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(all_ranks)
    return {'r@1': float(np.mean(r <= 1)), 'r@5': float(np.mean(r <= 5)), 'r@10': float(np.mean(r <= 10)), 'r@50': float(np.mean(r <= 50)), 'r@100': float(np.mean(r <= 100)), 'r@500': float(np.mean(r <= 500)), 'median_rank': float(np.median(r)), 'valid': int(len(valid))}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    model, device = load_model(job)
    val_raw, val_time, val_gt, val_scores, val_ranks, val_prob = load_split(job, 'val', model, device)
    test_raw, test_time, test_gt, test_scores, test_ranks, test_prob = load_split(job, 'test', model, device)
    X, y = train_data(val_time, val_gt, val_scores, val_ranks, val_prob)
    print('TRAIN_SHAPE', X.shape, 'pos', int(y.sum()), 'neg', int((y == 0).sum()), flush=True)
    models = {
        'hgb_hardneg': HistGradientBoostingClassifier(max_iter=420, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=1e-4, class_weight='balanced', random_state=55),
        'extratrees_hardneg': ExtraTreesClassifier(n_estimators=360, max_depth=18, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=4, random_state=56),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X, y)
        sample = clf.predict_proba(X[:min(len(X), 200000)])[:, 1]
        auc = roc_auc_score(y[:len(sample)], sample)
        met = eval_full(clf, test_time, test_gt, test_scores, test_ranks, test_prob)
        row = {'detector': 'LIGO', 'data_mode': 'noisy', 'family': 'SIS', 'method': name, 'features': ','.join(FEATURE_KEYS), 'train_pairs': int(len(y)), 'train_pos': int(y.sum()), 'sample_auc': float(auc), **met}
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
