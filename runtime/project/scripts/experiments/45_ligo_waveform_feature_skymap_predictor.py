from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')
unified = importlib.import_module('scripts.experiments.43_unified_sky_aux_comparison')

OUT_ROOT = Path('runs/ligo_waveform_feature_skymap_predictor_20260604')
SOURCE_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
JOBS = [j for j in base.JOBS if j['mode'] == 'noisy' and j['detector'] == 'LIGO']
PREDICTORS = ['waveform_stats_ridge', 'fusion_stats_embedding_ridge']
N_BINS = 256
NEG_PER_POS = 500
CHUNK_ROWS = 64


def stable_seed(*parts: object) -> int:
    text = '|'.join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for key in ['data_root', 'out_dir']:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def split_pack(job: dict, split: str, cache_dir: Path):
    cfg = cfg_from_run(job['run'], cache_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits['lensed'][split], splits['unlensed'][split], cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs, time_obs, _, emb, scores = base.load_split(job, split, SOURCE_ROOT / f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}" / split)
    return cfg, ds, raw_obs, time_obs, gt, emb, scores


def sample_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    c, t = x.shape[0], x.shape[-1]
    bins = min(N_BINS, t)
    width = max(1, t // bins)
    y = x[..., -width * bins:].reshape(c, bins, width)
    mean = y.mean(axis=-1)
    std = y.std(axis=-1)
    maxabs = np.max(np.abs(y), axis=-1)
    rms = np.sqrt(np.mean(y * y, axis=-1) + 1e-8)
    feats = [mean, std, maxabs, rms]
    # Low-cost frequency summaries. They help if sky information is encoded through detector response amplitude/frequency envelopes.
    sp = np.abs(np.fft.rfft(x, axis=-1)).astype(np.float32)
    fbins = min(64, sp.shape[-1])
    fwidth = max(1, sp.shape[-1] // fbins)
    f = sp[..., :fwidth * fbins].reshape(c, fbins, fwidth)
    feats.extend([np.log1p(f.mean(axis=-1)), np.log1p(f.max(axis=-1))])
    return np.concatenate([z.reshape(-1) for z in feats]).astype(np.float32)


def extract_waveform_features(job: dict, split: str, ds: EvaluationSet, out_dir: Path) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_waveform_stats_b{N_BINS}.npy"
    if path.exists():
        return np.load(path)
    rows = []
    for i in range(len(ds)):
        rows.append(sample_features(ds[i].numpy()))
        if (i + 1) % 5000 == 0:
            print('  features', split, i + 1, '/', len(ds), flush=True)
    X = np.stack(rows).astype(np.float32)
    np.save(path, X)
    return X


def predictor_inputs(name: str, stats: np.ndarray, emb: np.ndarray) -> np.ndarray:
    if name == 'waveform_stats_ridge':
        return stats
    if name in {'fusion_stats_embedding_ridge', 'fusion_stats_embedding_mlp'}:
        return np.concatenate([stats, emb.astype(np.float32)], axis=1)
    raise ValueError(name)


def make_predictor(name: str):
    if name.endswith('_ridge'):
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 14)))
    if name.endswith('_mlp'):
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(512, 256), activation='relu', alpha=1e-4, learning_rate_init=8e-4, batch_size=512, max_iter=120, early_stopping=True, n_iter_no_change=10, random_state=42),
        )
    raise ValueError(name)


def fit_predictor(name: str, train_raw: pd.DataFrame, train_stats: np.ndarray, train_emb: np.ndarray, val_raw: pd.DataFrame, val_stats: np.ndarray, val_emb: np.ndarray):
    model = make_predictor(name)
    X_train = predictor_inputs(name, train_stats, train_emb)
    X_val = predictor_inputs(name, val_stats, val_emb)
    y_train = base.unit_vectors(train_raw)
    model.fit(X_train, y_train)
    val_mu = base.normalize_vectors(model.predict(X_val))
    val_true = base.unit_vectors(val_raw)
    err = base.angular_sep_from_unit(val_mu, val_true)
    sigma = float(max(base.MIN_SKY_SIGMA, np.median(err)))
    return model, sigma, float(np.mean(err)), float(np.median(err)), float(np.mean(err < 0.5)), float(np.mean(err < 1.0))


def predict_mu(name: str, model, stats: np.ndarray, emb: np.ndarray) -> np.ndarray:
    return base.normalize_vectors(model.predict(predictor_inputs(name, stats, emb)))


def train_examples(time_obs, gt, sky_mu, sky_sigma, scores, ranks, job, predictor):
    rng = np.random.default_rng(stable_seed('skyfeat', predictor, job['detector'], job['mode'], job['family']))
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
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
    X = base.feature_matrix(time_obs, sky_mu, sky_sigma, scores, ranks, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def run_one(job: dict, predictor: str) -> dict:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD', name, predictor, flush=True)
    train_cfg, train_ds, train_raw, train_time, train_gt, train_emb, train_scores = split_pack(job, 'train', out_dir / 'train')
    val_cfg, val_ds, val_raw, val_time, val_gt, val_emb, val_scores = split_pack(job, 'val', out_dir / 'val')
    test_cfg, test_ds, test_raw, test_time, test_gt, test_emb, test_scores = split_pack(job, 'test', out_dir / 'test')
    train_stats = extract_waveform_features(job, 'train', train_ds, out_dir / 'features')
    val_stats = extract_waveform_features(job, 'val', val_ds, out_dir / 'features')
    test_stats = extract_waveform_features(job, 'test', test_ds, out_dir / 'features')

    print('FIT_SKY', name, predictor, train_stats.shape, flush=True)
    model, sigma, mean_err, med_err, acc05, acc10 = fit_predictor(predictor, train_raw, train_stats, train_emb, val_raw, val_stats, val_emb)
    val_mu = predict_mu(predictor, model, val_stats, val_emb)
    test_mu = predict_mu(predictor, model, test_stats, test_emb)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)

    Xv, yv = train_examples(val_time, val_gt, val_mu, sigma, val_scores, val_ranks, job, predictor)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'predictor': predictor,
        'sky_predictor_input': 'waveform_stats' if predictor == 'waveform_stats_ridge' else 'waveform_stats_plus_matching_embedding',
        'sky_sigma_rad': sigma,
        'sky_val_mean_error_rad': mean_err,
        'sky_val_median_error_rad': med_err,
        'sky_val_frac_err_lt_0p5': acc05,
        'sky_val_frac_err_lt_1p0': acc10,
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **base.eval_full(clf, test_time, test_gt, test_mu, sigma, test_scores, test_ranks),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{predictor}_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(row, flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in JOBS:
        for predictor in PREDICTORS:
            rows.append(run_one(job, predictor))
            pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    (OUT_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    best = df.sort_values(['detector', 'family', 'rerank_r@1'], ascending=[True, True, False]).groupby(['detector', 'data_mode', 'family']).head(1)
    best.to_csv(OUT_ROOT / 'best_by_group.csv', index=False)
    print('BEST_BY_GROUP')
    print(best[['detector', 'data_mode', 'family', 'predictor', 'sky_val_mean_error_rad', 'rerank_r@1', 'rerank_r@5', 'rerank_r@10', 'rerank_r@50']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
