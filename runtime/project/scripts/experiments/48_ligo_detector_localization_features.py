from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, pad_or_trim, split_indices, zscore_channels
from matchgw.trigger_time import catalog_trigger_time_frame, log1p_delta_time_obs

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

OUT_ROOT = Path('runs/ligo_detector_localization_features_20260604')
SOURCE_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
JOBS = [j for j in base.JOBS if j['detector'] == 'LIGO' and j['mode'] == 'noisy']
NEG_PER_POS = 500
CHUNK_ROWS = 64
BANDS = [(40, 160), (160, 320), (320, 580), (580, 1200), (1200, 2400)]


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode('utf-8')).hexdigest()[:8], 16)


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for key in ['data_root', 'out_dir']:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def cache_name(job: dict, split: str, suffix: str) -> Path:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    return SOURCE_ROOT / name / split / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_{suffix}.npy"


def pack(job: dict, split: str, out_dir: Path):
    cfg = cfg_from_run(job['run'], out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    lidx = splits['lensed'][split]
    uidx = splits['unlensed'][split]
    ds = EvaluationSet(arrays, lidx, uidx, cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs = aux.catalog_observable_frame(cfg.data_root, job['family'], lidx, uidx).reset_index(drop=True)
    time_obs = catalog_trigger_time_frame(cfg.data_root, job['family'], lidx, uidx, detector=job['detector']).reset_index(drop=True)
    scores = np.load(cache_name(job, split, 'scores'))
    return cfg, ds, raw_obs, time_obs, gt, scores


def _safe_log_ratio(a: float, b: float, eps: float = 1e-8) -> float:
    return float(np.log((float(a) + eps) / (float(b) + eps)))


def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 2048) -> float:
    # Use FFT-free local cross-correlation around zero lag to avoid large allocations.
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = np.empty(len(lags), dtype=np.float32)
    for k, lag in enumerate(lags):
        if lag < 0:
            vals[k] = float(np.dot(a[-lag:], b[:n + lag]))
        elif lag > 0:
            vals[k] = float(np.dot(a[:n - lag], b[lag:]))
        else:
            vals[k] = float(np.dot(a, b))
    return float(lags[int(np.argmax(vals))] / max_lag)


def event_detector_features(x: np.ndarray, cfg: MatchRunConfig) -> np.ndarray:
    y = pad_or_trim(x, cfg.target_len, cfg.stride)
    if y.ndim == 1:
        y = y.reshape(2, -1) if y.size % 2 == 0 else y[None, :]
    y = y.reshape(-1, y.shape[-1]).astype(np.float32)
    if y.shape[0] < 2:
        y = np.vstack([y[0], y[0]])
    y = zscore_channels(y[:2])
    h, l = y[0], y[1]
    eps = 1e-8
    peak_h = float(np.max(np.abs(h)))
    peak_l = float(np.max(np.abs(l)))
    i_h = int(np.argmax(np.abs(h)))
    i_l = int(np.argmax(np.abs(l)))
    rms_h = float(np.sqrt(np.mean(h * h) + eps))
    rms_l = float(np.sqrt(np.mean(l * l) + eps))
    eng_h = float(np.sum(h * h))
    eng_l = float(np.sum(l * l))
    std_h = float(np.std(h) + eps)
    std_l = float(np.std(l) + eps)
    sp_h = np.fft.rfft(h)
    sp_l = np.fft.rfft(l)
    amp_h = np.abs(sp_h).astype(np.float32)
    amp_l = np.abs(sp_l).astype(np.float32)
    feats = [
        float((i_h - i_l) / max(len(h), 1)),
        _xcorr_lag(h, l),
        _safe_log_ratio(peak_h, peak_l),
        _safe_log_ratio(rms_h, rms_l),
        _safe_log_ratio(eng_h, eng_l),
        _safe_log_ratio(peak_h / std_h, peak_l / std_l),
        float(np.corrcoef(h, l)[0, 1]) if np.std(h) > 0 and np.std(l) > 0 else 0.0,
    ]
    phase = np.angle(sp_h * np.conj(sp_l))
    for lo, hi in BANDS:
        lo = max(0, min(lo, len(amp_h) - 1))
        hi = max(lo + 1, min(hi, len(amp_h)))
        eh = float(np.sum(amp_h[lo:hi] ** 2))
        el = float(np.sum(amp_l[lo:hi] ** 2))
        feats.append(_safe_log_ratio(eh, el))
        feats.append(float(np.angle(np.mean(np.exp(1j * phase[lo:hi])))))
    return np.asarray(feats, dtype=np.float32)


def extract_detector_features(job: dict, split: str, ds: EvaluationSet, cfg: MatchRunConfig, out_dir: Path) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_detloc.npy"
    if path.exists():
        return np.load(path)
    rows = []
    for i in range(len(ds)):
        rows.append(event_detector_features(ds.waveforms[i], cfg))
        if (i + 1) % 5000 == 0:
            print('  det_features', split, i + 1, '/', len(ds), flush=True)
    X = np.stack(rows).astype(np.float32)
    np.save(path, X)
    return X


def pair_feature_matrix(variant: str, det: np.ndarray, time_obs, scores, ranks, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    cols = [
        log1p_delta_time_obs(time_obs, a, c).astype(np.float32)[:, None],
        scores[a, c].astype(np.float32)[:, None],
        (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32)[:, None],
    ]
    if variant == 'detloc':
        da = det[a]
        dc = det[c]
        cols.extend([np.abs(da - dc).astype(np.float32), (da * dc).astype(np.float32)])
    return np.concatenate(cols, axis=1).astype(np.float32)


def train_examples(variant: str, det, time_obs, gt, scores, ranks, job):
    rng = np.random.default_rng(stable_seed('detloc_train', variant, job['family']))
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
    a = np.concatenate([pos_a, neg_a])
    c = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    X = pair_feature_matrix(variant, det, time_obs, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(variant: str, clf, det, time_obs, gt, scores, ranks):
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        a = np.repeat(rows.astype(np.int32), n)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = pair_feature_matrix(variant, det, time_obs, scores, ranks, a, c)
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
        'median_rank': float(np.median(r)),
        'valid': int(len(valid)),
    }


def run_variant(job, variant, val_pack, test_pack):
    val_det, val_time, val_gt, val_scores, val_ranks = val_pack
    test_det, test_time, test_gt, test_scores, test_ranks = test_pack
    Xv, yv = train_examples(variant, val_det, val_time, val_gt, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'variant': variant,
        'method': 'ligo_detector_localization_feature_rerank',
        'features': 'log1p_delta_time_obs,waveform_score,waveform_reciprocal_rank' + (',detector_localization_pair_features' if variant == 'detloc' else ''),
        'detloc_feature_count': int(val_det.shape[1]),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **eval_full(variant, clf, test_det, test_time, test_gt, test_scores, test_ranks),
    }
    print(row, flush=True)
    return row


def run_job(job):
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD', name, flush=True)
    val_cfg, val_ds, val_raw, val_time, val_gt, val_scores = pack(job, 'val', out_dir / 'val')
    test_cfg, test_ds, test_raw, test_time, test_gt, test_scores = pack(job, 'test', out_dir / 'test')
    val_det = extract_detector_features(job, 'val', val_ds, val_cfg, out_dir / 'features')
    test_det = extract_detector_features(job, 'test', test_ds, test_cfg, out_dir / 'features')
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    val_pack = (val_det, val_time, val_gt, val_scores, val_ranks)
    test_pack = (test_det, test_time, test_gt, test_scores, test_ranks)
    rows = [run_variant(job, 'baseline', val_pack, test_pack), run_variant(job, 'detloc', val_pack, test_pack)]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'summary.csv', index=False)
    (out_dir / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in JOBS:
        rows.extend(run_job(job))
        pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    pivot = df.pivot_table(index=['detector','data_mode','family'], columns='variant', values='r@1', aggfunc='first')
    pivot.to_csv(OUT_ROOT / 'r1_pivot.csv')
    print(df.to_string(index=False), flush=True)
    print(pivot.to_string(), flush=True)


if __name__ == '__main__':
    main()
