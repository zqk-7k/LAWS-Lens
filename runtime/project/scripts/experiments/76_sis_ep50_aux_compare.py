from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import retrieval_metrics, similarity_matrix
from matchgw.pipeline import build_model, embed_eval, train_encoder
from matchgw.trigger_time import catalog_trigger_time_frame, log1p_delta_time_obs

# Reuse the observable-frame and angular separation definitions used in previous aux experiments.
import importlib
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

OUT_ROOT = Path('runs/sis_ep50_aux_compare_20260608')
DATA_ROOTS = {
    'ET': Path('/root/autodl-tmp/gw_et_10000_matchstyle_20260527_091859'),
    'LIGO': Path('/root/autodl-tmp/gw_ligo_10000_matchstyle_20260527_091859'),
}
JOBS = [('ET', 'pure'), ('ET', 'noisy'), ('LIGO', 'pure'), ('LIGO', 'noisy')]
VARIANTS = [
    'waveform_only',
    'delta_time_only',
    'delta_time_plus_true_sky_sep',
    'delta_time_plus_true_sky_overlap',
    'delta_time_plus_predicted_sky_overlap',
]
REAL_SKY_SIGMA_RAD = 0.08
MIN_SKY_SIGMA = 0.03
NEG_PER_POS = 500
CHUNK_ROWS = 64
EPS = 1e-8


def make_cfg(detector: str, mode: str) -> MatchRunConfig:
    is_ligo = detector == 'LIGO'
    return MatchRunConfig(
        data_root=DATA_ROOTS[detector],
        model_type='SIS',
        data_mode=mode,
        out_dir=OUT_ROOT / f'{detector.lower()}_{mode}_sis_ep50',
        backbone='inceptiontime',
        preprocess='bandpass',
        bandpass_low=40,
        bandpass_high=580,
        target_len=8192,
        stride=2,
        lensed_limit=10000,
        unlensed_limit=10000,
        epochs=50,
        batch_size=128 if not is_ligo else 96,
        eval_batch_size=512 if not is_ligo else 256,
        lr=1e-3,
        weight_decay=1e-4,
        tau=0.07,
        emb_dim=128,
        d_model=256,
        width_scale=2.0,
        aug_roll=128,
        aug_scale=0.10,
        aug_noise=0.01,
        aug_flip=True,
        amp=True,
        amp_dtype='bf16',
        num_workers=2,
        pin_memory=True,
        export_candidates=False,
    )


def unit_vectors(obs: pd.DataFrame) -> np.ndarray:
    ra = obs['ra'].to_numpy(dtype=np.float64)
    dec = obs['dec'].to_numpy(dtype=np.float64)
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float32)


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    return (x / norm).astype(np.float32)


def angular_sep_from_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def log_gaussian_overlap_from_unit(mu_i: np.ndarray, mu_j: np.ndarray, sigma_i: float, sigma_j: float) -> np.ndarray:
    sep = angular_sep_from_unit(mu_i, mu_j)
    var = sigma_i * sigma_i + sigma_j * sigma_j
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def fit_sky_predictor(train_obs: pd.DataFrame, train_emb: np.ndarray, val_obs: pd.DataFrame, val_emb: np.ndarray):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 12)))
    model.fit(train_emb, unit_vectors(train_obs))
    val_pred = normalize_vectors(model.predict(val_emb))
    err = angular_sep_from_unit(val_pred, unit_vectors(val_obs))
    sigma = float(max(MIN_SKY_SIGMA, np.median(err)))
    return model, sigma, float(np.mean(err)), float(np.median(err))


def row_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int32)
    return ranks


def split_pack(arrays, splits, cfg: MatchRunConfig, detector: str, split: str, model=None, out_dir: Path | None = None):
    ds = EvaluationSet(arrays, splits['lensed'][split], splits['unlensed'][split], cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs = aux.catalog_observable_frame(cfg.data_root, 'SIS', splits['lensed'][split], splits['unlensed'][split]).reset_index(drop=True)
    time_obs = catalog_trigger_time_frame(cfg.data_root, 'SIS', splits['lensed'][split], splits['unlensed'][split], detector=detector).reset_index(drop=True)
    emb = None
    scores = None
    if model is not None:
        emb_path = out_dir / f'{split}_embeddings.npy'
        scores_path = out_dir / f'{split}_scores.npy'
        if emb_path.exists() and scores_path.exists():
            emb = np.load(emb_path)
            scores = np.load(scores_path)
        else:
            emb = embed_eval(model, ds, cfg, cpu=False).astype(np.float32)
            scores = similarity_matrix(emb).astype(np.float32)
            np.fill_diagonal(scores, -np.inf)
            np.save(emb_path, emb)
            np.save(scores_path, scores)
    return ds, raw_obs, time_obs, gt, emb, scores


def true_sky_sep(raw_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    ra = raw_obs['ra'].to_numpy(dtype=np.float64)
    dec = raw_obs['dec'].to_numpy(dtype=np.float64)
    return aux.angular_sep(ra[a], dec[a], ra[c], dec[c]).astype(np.float32)


def true_log_sky_overlap(raw_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    sep = true_sky_sep(raw_obs, a, c)
    var = REAL_SKY_SIGMA_RAD * REAL_SKY_SIGMA_RAD * 2.0
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def feature_matrix(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    cols = []
    if variant != 'waveform_only':
        cols.append(log1p_delta_time_obs(time_obs, a, c))
    if variant == 'delta_time_plus_true_sky_sep':
        cols.append(true_sky_sep(raw_obs, a, c))
    elif variant == 'delta_time_plus_true_sky_overlap':
        cols.append(true_log_sky_overlap(raw_obs, a, c))
    elif variant == 'delta_time_plus_predicted_sky_overlap':
        if sky_mu is None or sky_sigma is None:
            raise ValueError('predicted sky overlap requires sky_mu and sky_sigma')
        cols.append(log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma))
    cols.extend([
        scores[a, c].astype(np.float32),
        (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32),
    ])
    return np.column_stack(cols).astype(np.float32)


def feature_names(variant: str) -> str:
    names = []
    if variant != 'waveform_only':
        names.append('log1p_delta_time_obs')
    if variant == 'delta_time_plus_true_sky_sep':
        names.append('oracle_sky_sep_from_ra_dec')
    elif variant == 'delta_time_plus_true_sky_overlap':
        names.append('oracle_log_sky_overlap_from_ra_dec')
    elif variant == 'delta_time_plus_predicted_sky_overlap':
        names.append('predicted_log_sky_overlap_from_waveform_sky_model')
    names.extend(['waveform_score', 'waveform_reciprocal_rank'])
    return ','.join(names)


def train_examples(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
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
    X = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(variant: str, clf, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out)
    return {
        'r@1': float(np.mean(r <= 1)),
        'r@5': float(np.mean(r <= 5)),
        'r@10': float(np.mean(r <= 10)),
        'r@50': float(np.mean(r <= 50)),
        'r@100': float(np.mean(r <= 100)),
        'r@500': float(np.mean(r <= 500)),
        'median_true_rank': float(np.median(r)),
        'valid': int(len(valid)),
    }


def run_one(detector: str, mode: str) -> list[dict]:
    cfg = make_cfg(detector, mode)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print('TRAIN_OR_LOAD', detector, mode, cfg.out_dir, flush=True)
    total_t0 = time.perf_counter()
    summary_path = cfg.out_dir / 'waveform_summary.json'
    model_path = cfg.out_dir / 'model.pt'
    if model_path.exists() and summary_path.exists():
        arrays = load_match_arrays(cfg)
        splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
        model = build_model(cfg)
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        train_info = json.loads(summary_path.read_text(encoding='utf-8')).get('timing', {})
    else:
        model, state, train_info = train_encoder(cfg, cpu=False)
        arrays = state['arrays']
        splits = state['splits']
        torch.save({'model': model.state_dict(), 'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}}, model_path)
        pd.DataFrame(train_info['history']).to_csv(cfg.out_dir / 'history.csv', index=False)

    train_ds, train_raw, train_time, train_gt, train_emb, train_scores = split_pack(arrays, splits, cfg, detector, 'train', model, cfg.out_dir)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = split_pack(arrays, splits, cfg, detector, 'val', model, cfg.out_dir)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = split_pack(arrays, splits, cfg, detector, 'test', model, cfg.out_dir)

    train_ranks = row_ranks(train_scores)
    val_ranks = row_ranks(val_scores)
    test_ranks = row_ranks(test_scores)

    sky_model, sky_sigma, sky_val_mean_err, sky_val_med_err = fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = normalize_vectors(sky_model.predict(val_emb))
    test_sky_mu = normalize_vectors(sky_model.predict(test_emb))

    waveform_test = retrieval_metrics(test_scores, test_gt, ks=(1, 5, 10, 50, 100, 500))
    waveform_val = retrieval_metrics(val_scores, val_gt, ks=(1, 5, 10, 50, 100, 500))
    rows = []
    for variant in VARIANTS:
        print('VARIANT', detector, mode, variant, flush=True)
        if variant == 'waveform_only':
            met = {k: waveform_test[k] for k in ['r@1', 'r@5', 'r@10', 'r@50', 'r@100', 'r@500', 'median_true_rank', 'mrr']}
            row = {
                'detector': detector,
                'data_mode': mode,
                'family': 'SIS',
                'variant': variant,
                'method': 'ep50_inceptiontime_bandpass_direct_retrieval',
                'features': 'waveform_embedding_cosine_similarity',
                'val_auc_sampled': np.nan,
                'val_r@10': waveform_val['r@10'],
                **met,
                'sky_sigma_rad': sky_sigma,
                'sky_val_mean_angular_error_rad': sky_val_mean_err,
                'sky_val_median_angular_error_rad': sky_val_med_err,
                'train_s': train_info.get('train_s', np.nan),
                'mean_epoch_s': train_info.get('mean_epoch_s', np.nan),
                'total_s_so_far': float(time.perf_counter() - total_t0),
            }
        else:
            Xv, yv = train_examples(variant, val_raw, val_time, val_gt, val_sky_mu, sky_sigma, val_scores, val_ranks, seed=74000 + len(rows))
            clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=74)
            clf.fit(Xv, yv)
            pv = clf.predict_proba(Xv)[:, 1]
            met = eval_full(variant, clf, test_raw, test_time, test_gt, test_sky_mu, sky_sigma, test_scores, test_ranks)
            row = {
                'detector': detector,
                'data_mode': mode,
                'family': 'SIS',
                'variant': variant,
                'method': 'ep50_inceptiontime_bandpass_catalog_hgb_rerank',
                'features': feature_names(variant),
                'train_examples': int(len(yv)),
                'train_positive': int(yv.sum()),
                'val_auc_sampled': float(roc_auc_score(yv, pv)),
                'val_r@10': np.nan,
                **met,
                'sky_sigma_rad': sky_sigma,
                'sky_val_mean_angular_error_rad': sky_val_mean_err,
                'sky_val_median_angular_error_rad': sky_val_med_err,
                'train_s': train_info.get('train_s', np.nan),
                'mean_epoch_s': train_info.get('mean_epoch_s', np.nan),
                'total_s_so_far': float(time.perf_counter() - total_t0),
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(cfg.out_dir / 'aux_compare_partial.csv', index=False)
        print(json.dumps(row, indent=2), flush=True)

    wave_summary = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        'sizes': {
            'lensed_total': int(len(arrays.l1)),
            'unlensed_total': int(len(arrays.unlensed)),
            'train_lensed': int(len(splits['lensed']['train'])),
            'val_lensed': int(len(splits['lensed']['val'])),
            'test_lensed': int(len(splits['lensed']['test'])),
        },
        'timing': {k: v for k, v in train_info.items() if k != 'history'},
        'test_waveform': waveform_test,
        'val_waveform': waveform_val,
        'sky_predictor': {
            'model': 'RidgeCV waveform_embedding_to_sky_unit_vector',
            'sigma_rad': sky_sigma,
            'val_mean_angular_error_rad': sky_val_mean_err,
            'val_median_angular_error_rad': sky_val_med_err,
        },
        'note': 'delta_time uses trigger_time_obs. true_sky_* uses ra/dec only as oracle comparison. predicted_sky_overlap uses waveform-predicted sky unit vectors.',
    }
    summary_path.write_text(json.dumps(wave_summary, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(cfg.out_dir / 'aux_compare.csv', index=False)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for detector, mode in JOBS:
        all_rows.extend(run_one(detector, mode))
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    pivot = df.pivot_table(index=['detector', 'data_mode'], columns='variant', values='r@10', aggfunc='first')
    pivot.to_csv(OUT_ROOT / 'r10_pivot.csv')
    print(df.to_string(index=False), flush=True)
    print('\nR@10 pivot\n', pivot.to_string(), flush=True)


if __name__ == '__main__':
    main()
