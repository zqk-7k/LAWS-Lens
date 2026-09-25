from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

m = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base = m.base

OUT_ROOT = Path('runs/ligo_sis_grid18_rank_fusion_20260604')
SRC_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
EPS = 1e-8


def load_or_predict(job, split, out_dir, model, device):
    cache = out_dir / f'{split}_prob.npy'
    cfg, ds, raw, time_obs, gt, scores = m.load_pack(job, split, out_dir / split, True)
    if cache.exists():
        prob = np.load(cache)
    else:
        prob = m.predict_maps(model, ds, device)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, prob)
    return ds, raw, time_obs, gt, scores, prob


def zrow(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    finite = np.isfinite(x)
    for i in range(x.shape[0]):
        row_finite = finite[i]
        if not row_finite.any():
            x[i] = 0.0
            continue
        vals = x[i, row_finite]
        fill = float(vals.min())
        x[i, ~row_finite] = fill
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def feature_cube(time_obs, prob, scores, ranks, gt):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    rows = valid
    wf = scores[rows].astype(np.float32)
    rr = (1.0 / np.maximum(ranks[rows], 1)).astype(np.float32)
    ov = np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)
    dt = np.empty((len(rows), n), dtype=np.float32)
    all_c = np.arange(n, dtype=np.int32)
    for i, a in enumerate(rows):
        aa = np.full(n, a, dtype=np.int32)
        dt[i] = -m.log1p_delta_time_obs(time_obs, aa, all_c)
    feats = {
        'waveform_score_z': zrow(wf),
        'reciprocal_rank_z': zrow(rr),
        'neg_log1p_delta_time_obs_z': zrow(dt),
        'grid18_skymap_overlap_z': zrow(ov),
    }
    return valid, feats


def metrics_from_score(score: np.ndarray, valid: np.ndarray, gt: np.ndarray):
    score = score.copy()
    score[~np.isfinite(score)] = -np.inf
    pos = np.arange(len(valid))
    score[pos, valid] = -np.inf
    true = score[pos, gt[valid].astype(int)]
    rank = 1 + np.sum(score > true[:, None], axis=1)
    return {
        'r@1': float(np.mean(rank <= 1)),
        'r@5': float(np.mean(rank <= 5)),
        'r@10': float(np.mean(rank <= 10)),
        'r@50': float(np.mean(rank <= 50)),
        'r@100': float(np.mean(rank <= 100)),
        'r@500': float(np.mean(rank <= 500)),
        'median_rank': float(np.median(rank)),
    }


def score_from_weights(feats, w):
    return (
        w['waveform'] * feats['waveform_score_z']
        + w['rank'] * feats['reciprocal_rank_z']
        + w['time'] * feats['neg_log1p_delta_time_obs_z']
        + w['sky'] * feats['grid18_skymap_overlap_z']
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    _, train_ds, train_raw, _, _, _ = m.load_pack(job, 'train', SRC_ROOT / 'train', False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = m.SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    ckpt = torch.load(SRC_ROOT / 'ligo_noisy_sis' / 'grid_skymap_cnn.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    _, val_raw, val_time, val_gt, val_scores, val_prob = load_or_predict(job, 'val', OUT_ROOT, model, device)
    _, test_raw, test_time, test_gt, test_scores, test_prob = load_or_predict(job, 'test', OUT_ROOT, model, device)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)

    val_valid, val_feats = feature_cube(val_time, val_prob, val_scores, val_ranks, val_gt)
    test_valid, test_feats = feature_cube(test_time, test_prob, test_scores, test_ranks, test_gt)

    weights = []
    for waveform in [0.0, 0.25, 0.5, 1.0]:
        for rank in [0.0, 0.25, 0.5, 1.0]:
            for time in [0.5, 1.0, 1.5, 2.0, 3.0]:
                for sky in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]:
                    weights.append({'waveform': waveform, 'rank': rank, 'time': time, 'sky': sky})
    rows = []
    best = None
    for w in weights:
        met = metrics_from_score(score_from_weights(val_feats, w), val_valid, val_gt)
        row = {**w, **{f'val_{k}': v for k, v in met.items()}}
        rows.append(row)
        key = (met['r@1'], met['r@5'], met['r@10'], -met['median_rank'])
        if best is None or key > best[0]:
            best = (key, w, met)
    pd.DataFrame(rows).sort_values(['val_r@1', 'val_r@5', 'val_r@10', 'val_median_rank'], ascending=[False, False, False, True]).to_csv(OUT_ROOT / 'weight_search.csv', index=False)
    best_w = best[1]
    test_met = metrics_from_score(score_from_weights(test_feats, best_w), test_valid, test_gt)
    val_met = best[2]
    result = {
        'detector': 'LIGO', 'data_mode': 'noisy', 'family': 'SIS',
        'method': 'grid18_rank_fusion_weight_search',
        'selection': 'best validation R@1 then R@5/R@10/median_rank',
        **{f'weight_{k}': v for k, v in best_w.items()},
        **{f'val_{k}': v for k, v in val_met.items()},
        **test_met,
        'valid': int(len(test_valid)),
    }
    pd.DataFrame([result]).to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(result, flush=True)


if __name__ == '__main__':
    main()
