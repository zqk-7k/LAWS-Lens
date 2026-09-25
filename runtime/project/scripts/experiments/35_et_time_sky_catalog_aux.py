from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import ground_truth_partner, load_match_arrays, split_indices, EvaluationSet
import importlib
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

RUNS = {
    'SIS': Path('runs/et10000_bandpass_full_ep50_20260528_101013/SIS_noisy_bandpass_n10000_ep50'),
    'PM': Path('runs/et10000_bandpass_full_ep50_20260528_101013/PM_noisy_bandpass_n10000_ep50'),
}
MODE = 'realistic'
NEG_PER_POS = 300
CHUNK_ROWS = 96


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for k in ['data_root', 'out_dir']:
        if k in kwargs:
            kwargs[k] = Path(kwargs[k])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def split_obs(family: str, split: str, out_dir: Path):
    cfg = cfg_from_run(RUNS[family], out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits['lensed'][split], splits['unlensed'][split], cfg)
    gt = ground_truth_partner(ds.meta)
    obs = aux.catalog_observable_frame(cfg.data_root, family, splits['lensed'][split], splits['unlensed'][split])
    obs = aux.perturb_observables(obs, MODE, seed=31000 + abs(hash((family, split, MODE))) % 10000).reset_index(drop=True)
    return obs, gt


def feature_matrix(obs: pd.DataFrame, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    ra = obs['ra'].to_numpy(); dec = obs['dec'].to_numpy(); t = obs['geocent_time'].to_numpy()
    return np.column_stack([
        np.log1p(np.abs(t[anchors] - t[cands])),
        aux.angular_sep(ra[anchors], dec[anchors], ra[cands], dec[cands]),
    ]).astype(np.float32)


def train_examples(obs: pd.DataFrame, gt: np.ndarray, family: str):
    rng = np.random.default_rng(42000 + abs(hash(family)) % 10000)
    valid = np.flatnonzero(gt >= 0)
    n = len(obs)
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
    X = feature_matrix(obs, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full_catalog(clf, obs: pd.DataFrame, gt: np.ndarray):
    valid = np.flatnonzero(gt >= 0)
    n = len(obs)
    ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(obs, anchors, cands)
        score = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        score[np.arange(len(rows)), rows] = -np.inf
        true_score = score[np.arange(len(rows)), gt[rows].astype(int)]
        rank = 1 + np.sum(score > true_score[:, None], axis=1)
        ranks.extend(rank.tolist())
    ranks = np.asarray(ranks)
    return {
        'r@1': float(np.mean(ranks <= 1)),
        'r@5': float(np.mean(ranks <= 5)),
        'r@10': float(np.mean(ranks <= 10)),
        'r@50': float(np.mean(ranks <= 50)),
        'r@100': float(np.mean(ranks <= 100)),
        'r@500': float(np.mean(ranks <= 500)),
        'median_true_rank': float(np.median(ranks)),
        'valid': int(len(valid)),
    }


def run_family(family: str, out_root: Path):
    val_obs, val_gt = split_obs(family, 'val', out_root / family / 'val')
    test_obs, test_gt = split_obs(family, 'test', out_root / family / 'test')
    Xv, yv = train_examples(val_obs, val_gt, family)
    clf = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    result = {
        'family': family,
        'mode': MODE,
        'method': 'et_catalog_level_delta_time_sky_sep_hgb',
        'features': ['log1p_delta_time', 'sky_sep'],
        'neg_per_pos': NEG_PER_POS,
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **eval_full_catalog(clf, test_obs, test_gt),
    }
    print(result, flush=True)
    return result


def main():
    out_root = Path('runs/et_noisy_time_sky_catalog_aux_20260601')
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [run_family(f, out_root) for f in ['SIS', 'PM']]
    pd.DataFrame(rows).to_csv(out_root / 'summary.csv', index=False)
    (out_root / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
