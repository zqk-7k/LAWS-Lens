from __future__ import annotations

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

import importlib
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

RUNS = {
    'SIS': Path('runs/ligo_inceptiontime_bandpass_full_ep50_20260601_095620/SIS_noisy_inceptiontime_bandpass_n10000_ep50'),
    'PM': Path('runs/ligo_inceptiontime_bandpass_full_ep50_20260601_095620/PM_noisy_inceptiontime_bandpass_n10000_ep50'),
}
TOPKS = [50, 100, 500, 1000, 2000]
MODE = 'realistic'


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for k in ['data_root', 'out_dir']:
        if k in kwargs:
            kwargs[k] = Path(kwargs[k])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def topk_order(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[1]
    k = max(1, min(k, n - 1))
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    local = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, local, axis=1)


def eval_order(order: np.ndarray, gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    ranks = []
    for i in valid:
        rank = 10**9
        for r, j in enumerate(order[int(i)], start=1):
            if int(j) == int(gt[int(i)]):
                rank = r
                break
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {'r@1': float(np.mean(ranks <= 1)), 'r@5': float(np.mean(ranks <= 5)), 'r@10': float(np.mean(ranks <= 10)), 'r@50': float(np.mean(ranks <= 50)), 'r@500': float(np.mean(ranks <= 500)), 'r@1000': float(np.mean(ranks <= 1000)), 'median_true_rank': float(np.median(ranks)), 'valid': int(len(valid))}


def load_scores(family: str, split: str, out_dir: Path):
    run_dir = RUNS[family]
    cfg = cfg_from_run(run_dir, out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits['lensed'][split], splits['unlensed'][split], cfg)
    gt = ground_truth_partner(ds.meta)
    model = build_model(cfg)
    ckpt = torch.load(run_dir / 'model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    scores = similarity_matrix(embed_eval(model, ds, cfg, cpu=False))
    np.fill_diagonal(scores, -np.inf)
    print(f'loaded {family} {split} scores {scores.shape}', flush=True)
    return splits, scores, gt, cfg


def make_examples(obs: pd.DataFrame, order: np.ndarray, gt: np.ndarray):
    vals = obs.reset_index(drop=True)
    ra = vals['ra'].to_numpy(); dec = vals['dec'].to_numpy(); t = vals['geocent_time'].to_numpy()
    n = order.shape[0] * order.shape[1]
    X = np.empty((n, 2), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    anchors = np.empty(n, dtype=np.int32)
    cands = np.empty(n, dtype=np.int32)
    pos = 0
    for i, row in enumerate(order):
        for j in row:
            X[pos, 0] = np.log1p(abs(t[i] - t[j]))
            X[pos, 1] = aux.angular_sep(ra[i], dec[i], ra[j], dec[j])
            y[pos] = 1 if int(gt[i]) == int(j) else 0
            anchors[pos] = i; cands[pos] = j
            pos += 1
    return X, y, anchors, cands


def eval_ranker(proba: np.ndarray, anchors: np.ndarray, cands: np.ndarray, gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    by: dict[int, list[tuple[float, int]]] = {}
    for p, i, j in zip(proba, anchors, cands):
        by.setdefault(int(i), []).append((float(p), int(j)))
    ranks = []
    for i in valid:
        rank = 10**9
        for r, (_, j) in enumerate(sorted(by.get(int(i), []), reverse=True), start=1):
            if j == int(gt[int(i)]):
                rank = r; break
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {'r@1': float(np.mean(ranks <= 1)), 'r@5': float(np.mean(ranks <= 5)), 'r@10': float(np.mean(ranks <= 10)), 'r@50': float(np.mean(ranks <= 50)), 'r@500': float(np.mean(ranks <= 500)), 'r@1000': float(np.mean(ranks <= 1000)), 'median_true_rank': float(np.median(ranks)), 'valid': int(len(valid))}


def run_family(family: str, out_root: Path):
    val_splits, val_scores, val_gt, cfg = load_scores(family, 'val', out_root / family / 'val')
    test_splits, test_scores, test_gt, _ = load_scores(family, 'test', out_root / family / 'test')
    vobs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, val_splits['lensed']['val'], val_splits['unlensed']['val']), MODE, seed=11100 + abs(hash((family, 'val'))) % 10000)
    tobs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, test_splits['lensed']['test'], test_splits['unlensed']['test']), MODE, seed=11200 + abs(hash((family, 'test'))) % 10000)
    rows = []
    for topk in TOPKS:
        print('TOPK', family, topk, flush=True)
        vorder = topk_order(val_scores, topk); torder = topk_order(test_scores, topk)
        w = eval_order(torder, test_gt)
        Xv, yv, av, cv = make_examples(vobs, vorder, val_gt)
        Xt, yt, at, ct = make_examples(tobs, torder, test_gt)
        clf = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
        clf.fit(Xv, yv)
        pt = clf.predict_proba(Xt)[:, 1]
        pv = clf.predict_proba(Xv)[:, 1]
        r = eval_ranker(pt, at, ct, test_gt)
        row = {'family': family, 'mode': MODE, 'topk': topk, 'val_auc': float(roc_auc_score(yv, pv)), 'train_examples': int(len(yv)), 'train_positive': int(yv.sum()), **{f'waveform_{k}': v for k, v in w.items()}, **{f'aux_{k}': v for k, v in r.items()}}
        print(row, flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_root / f'{family}_partial.csv', index=False)
    return rows


def main():
    out_root = Path('runs/ligo_noisy_time_sky_large_topk_20260601')
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in ['SIS', 'PM']:
        rows.extend(run_family(family, out_root))
    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'summary.csv', index=False)
    (out_root / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
