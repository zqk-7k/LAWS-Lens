from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import ground_truth_partner, load_match_arrays, split_indices, EvaluationSet
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

RUNS = {
    'SIS': Path('runs/ligo_inceptiontime_bandpass_full_ep50_20260601_095620/SIS_noisy_inceptiontime_bandpass_n10000_ep50'),
    'PM': Path('runs/ligo_inceptiontime_bandpass_full_ep50_20260601_095620/PM_noisy_inceptiontime_bandpass_n10000_ep50'),
}
MODE = 'realistic'
NEG_PER_POS = 500
CHUNK_ROWS = 64
FEATURES = ['log1p_delta_time', 'sky_sep', 'chirp_mass_logdiff', 'mass_ratio_absdiff', 'mass_1_logdiff', 'mass_2_logdiff', 'chi_eff_absdiff', 'spin_a1_absdiff', 'spin_a2_absdiff', 'luminosity_distance_logdiff']


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
    obs = aux.perturb_observables(obs, MODE, seed=71000 + abs(hash((family, split, MODE))) % 10000).reset_index(drop=True)
    return obs, gt


def feature_matrix(obs: pd.DataFrame, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    ra = obs['ra'].to_numpy(); dec = obs['dec'].to_numpy(); t = obs['geocent_time'].to_numpy()
    cm = obs['chirp_mass'].to_numpy(); q = obs['mass_ratio'].to_numpy(); chi = obs['chi_eff_proxy'].to_numpy(); dl = obs['luminosity_distance'].to_numpy(); m1 = obs['mass_1_source'].to_numpy(); m2 = obs['mass_2_source'].to_numpy(); a1 = obs['a_1'].to_numpy(); a2 = obs['a_2'].to_numpy()
    return np.column_stack([
        np.log1p(np.abs(t[anchors] - t[cands])),
        aux.angular_sep(ra[anchors], dec[anchors], ra[cands], dec[cands]),
        np.abs(np.log(cm[anchors] / cm[cands])),
        np.abs(q[anchors] - q[cands]),
        np.abs(np.log(m1[anchors] / m1[cands])),
        np.abs(np.log(m2[anchors] / m2[cands])),
        np.abs(chi[anchors] - chi[cands]),
        np.abs(a1[anchors] - a1[cands]),
        np.abs(a2[anchors] - a2[cands]),
        np.abs(np.log(dl[anchors] / dl[cands])),
    ]).astype(np.float32)


def train_examples(obs: pd.DataFrame, gt: np.ndarray, family: str):
    rng = np.random.default_rng(72000 + abs(hash(family)) % 10000)
    valid = np.flatnonzero(gt >= 0); n = len(obs)
    pos_a = valid.astype(np.int32); pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    anchors = np.concatenate([pos_a, neg_a]); cands = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    X = feature_matrix(obs, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, obs: pd.DataFrame, gt: np.ndarray):
    valid = np.flatnonzero(gt >= 0); n = len(obs); ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        pred = clf.predict_proba(feature_matrix(obs, anchors, cands))[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(ranks)
    return {'r@1': float(np.mean(r <= 1)), 'r@5': float(np.mean(r <= 5)), 'r@10': float(np.mean(r <= 10)), 'r@50': float(np.mean(r <= 50)), 'r@100': float(np.mean(r <= 100)), 'r@500': float(np.mean(r <= 500)), 'median_true_rank': float(np.median(r)), 'valid': int(len(valid))}


def run_family(family: str, out_root: Path):
    val_obs, val_gt = split_obs(family, 'val', out_root / family / 'val')
    test_obs, test_gt = split_obs(family, 'test', out_root / family / 'test')
    Xv, yv = train_examples(val_obs, val_gt, family)
    clf = HistGradientBoostingClassifier(max_iter=360, learning_rate=0.05, max_leaf_nodes=21, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    result = {'family': family, 'mode': MODE, 'method': 'catalog_level_extended_observable_v2_hgb', 'features': FEATURES, 'neg_per_pos': NEG_PER_POS, 'train_examples': int(len(yv)), 'train_positive': int(yv.sum()), 'val_auc_sampled': float(roc_auc_score(yv, pv)), **eval_full(clf, test_obs, test_gt)}
    print(result, flush=True)
    return result


def main():
    out_root = Path('runs/ligo_noisy_catalog_extended_observable_v2_20260601')
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [run_family(f, out_root) for f in ['SIS', 'PM']]
    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'summary.csv', index=False)
    (out_root / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
