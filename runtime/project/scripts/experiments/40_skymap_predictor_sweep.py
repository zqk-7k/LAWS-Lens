from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')

OUT_ROOT = Path('runs/skymap_predictor_sweep_20260602')
SOURCE_ROOT = Path('runs/waveform_predicted_skymap_rerank_20260602')
JOBS = [
    {'detector': 'ET', 'mode': 'noisy', 'family': 'SIS'},
    {'detector': 'ET', 'mode': 'noisy', 'family': 'PM'},
    {'detector': 'LIGO', 'mode': 'noisy', 'family': 'SIS'},
    {'detector': 'LIGO', 'mode': 'noisy', 'family': 'PM'},
]
PREDICTORS = ['ridge', 'knn32', 'extratrees', 'randomforest', 'mlp']
NEG_PER_POS = 500


def predictor_factory(name: str):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 12)))
    if name == 'knn32':
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=32, weights='distance', metric='cosine'))
    if name == 'extratrees':
        return ExtraTreesRegressor(n_estimators=240, max_features=0.6, min_samples_leaf=3, n_jobs=-1, random_state=42)
    if name == 'randomforest':
        return RandomForestRegressor(n_estimators=180, max_features=0.6, min_samples_leaf=3, n_jobs=-1, random_state=42)
    if name == 'mlp':
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', alpha=1e-4, learning_rate_init=1e-3, batch_size=512, max_iter=80, early_stopping=True, random_state=42),
        )
    raise ValueError(name)


def load_cached(job: dict):
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    source = SOURCE_ROOT / name
    out = OUT_ROOT / name
    real_job = next(j for j in base.JOBS if j['detector'] == job['detector'] and j['mode'] == job['mode'] and j['family'] == job['family'])
    train_raw, train_time, train_gt, train_emb, train_scores = base.load_split(real_job, 'train', source / 'train')
    val_raw, val_time, val_gt, val_emb, val_scores = base.load_split(real_job, 'val', source / 'val')
    test_raw, test_time, test_gt, test_emb, test_scores = base.load_split(real_job, 'test', source / 'test')
    return real_job, out, train_raw, train_emb, val_raw, val_time, val_gt, val_emb, val_scores, test_time, test_gt, test_emb, test_scores


def fit_predictor(name: str, train_obs: pd.DataFrame, train_emb: np.ndarray, val_obs: pd.DataFrame, val_emb: np.ndarray):
    model = predictor_factory(name)
    y = base.unit_vectors(train_obs)
    model.fit(train_emb, y)
    val_mu = base.normalize_vectors(model.predict(val_emb))
    val_true = base.unit_vectors(val_obs)
    err = base.angular_sep_from_unit(val_mu, val_true)
    sigma = float(max(base.MIN_SKY_SIGMA, np.median(err)))
    return model, sigma, float(np.mean(err)), float(np.median(err)), float(np.mean(err < 0.5)), float(np.mean(err < 1.0))


def train_examples(time_obs, gt, sky_mu, sky_sigma, scores, ranks, job):
    rng = np.random.default_rng(93000 + abs(hash((job['detector'], job['mode'], job['family'], job.get('predictor','')))) % 10000)
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
    real_job, out_dir, train_raw, train_emb, val_raw, val_time, val_gt, val_emb, val_scores, test_time, test_gt, test_emb, test_scores = load_cached(job)
    out_dir.mkdir(parents=True, exist_ok=True)
    print('RUN', job, predictor, flush=True)
    model, sigma, mean_err, med_err, acc05, acc10 = fit_predictor(predictor, train_raw, train_emb, val_raw, val_emb)
    val_mu = base.predict_sky_mu(model, val_emb)
    test_mu = base.predict_sky_mu(model, test_emb)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    Xv, yv = train_examples(val_time, val_gt, val_mu, sigma, val_scores, val_ranks, {**job, 'predictor': predictor})
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'], 'data_mode': job['mode'], 'family': job['family'], 'predictor': predictor,
        'sky_sigma_rad': sigma, 'sky_val_mean_error_rad': mean_err, 'sky_val_median_error_rad': med_err,
        'sky_val_frac_err_lt_0p5': acc05, 'sky_val_frac_err_lt_1p0': acc10,
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(real_job['run']),
        **base.eval_full(clf, test_time, test_gt, test_mu, sigma, test_scores, test_ranks),
    }
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
    best = df.sort_values(['detector', 'family', 'rerank_r@1'], ascending=[True, True, False]).groupby(['detector','data_mode','family']).head(1)
    best.to_csv(OUT_ROOT / 'best_by_group.csv', index=False)
    print('BEST_BY_GROUP')
    print(best[['detector','data_mode','family','predictor','sky_val_mean_error_rad','rerank_r@1','rerank_r@5','rerank_r@10','rerank_r@50']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
