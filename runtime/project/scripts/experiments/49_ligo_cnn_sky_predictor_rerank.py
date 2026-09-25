from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.trigger_time import catalog_trigger_time_frame, log1p_delta_time_obs

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

OUT_ROOT = Path('runs/ligo_cnn_sky_predictor_rerank_20260604')
SOURCE_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
JOBS = [j for j in base.JOBS if j['detector'] == 'LIGO' and j['mode'] == 'noisy']
EPOCHS = 18
BATCH_SIZE = 192
LR = 2e-3
WEIGHT_DECAY = 1e-4
NEG_PER_POS = 500
CHUNK_ROWS = 64
MIN_SKY_SIGMA = 0.03


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


def load_pack(job: dict, split: str, out_dir: Path, need_scores: bool):
    cfg = cfg_from_run(job['run'], out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    lidx = splits['lensed'][split]
    uidx = splits['unlensed'][split]
    ds = EvaluationSet(arrays, lidx, uidx, cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs = aux.catalog_observable_frame(cfg.data_root, job['family'], lidx, uidx).reset_index(drop=True)
    time_obs = catalog_trigger_time_frame(cfg.data_root, job['family'], lidx, uidx, detector=job['detector']).reset_index(drop=True)
    scores = np.load(cache_name(job, split, 'scores')) if need_scores else None
    return cfg, ds, raw_obs, time_obs, gt, scores


class SkyDataset(Dataset):
    def __init__(self, ds: EvaluationSet, raw_obs: pd.DataFrame) -> None:
        self.ds = ds
        self.y = base.unit_vectors(raw_obs).astype(np.float32)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        return self.ds[idx], torch.from_numpy(self.y[idx])


class SkyCNN(nn.Module):
    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 21, stride=4, padding=10, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 192, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
        )
        self.attn = nn.Sequential(nn.Conv1d(192, 48, 1), nn.GELU(), nn.Conv1d(48, 1, 1))
        self.head = nn.Sequential(nn.Linear(192 * 3, 256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.float())
        w = torch.softmax(self.attn(y), dim=-1)
        attn = (y * w).sum(dim=-1)
        avg = y.mean(dim=-1)
        mx = y.amax(dim=-1)
        return F.normalize(self.head(torch.cat([attn, avg, mx], dim=1)), dim=-1)


def angular_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(pred * true, axis=1), -1.0, 1.0))


def predict_mu(model: nn.Module, ds: EvaluationSet, device: str) -> np.ndarray:
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    outs = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            outs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def train_sky_model(job: dict, train_ds: EvaluationSet, train_raw: pd.DataFrame, val_ds: EvaluationSet, val_raw: pd.DataFrame, out_dir: Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(SkyDataset(train_ds, train_raw), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_true = base.unit_vectors(val_raw).astype(np.float32)
    model = SkyCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = {'mean_err': 999.0, 'state': None, 'epoch': 0}
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = F.normalize(y.to(device), dim=-1)
            pred = model(x)
            cos = torch.sum(pred * y, dim=1).clamp(-1.0, 1.0)
            loss = (1.0 - cos).mean() + 0.05 * F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        val_mu = predict_mu(model, val_ds, device)
        err = angular_error(val_mu, val_true)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'val_mean_err': float(err.mean()), 'val_median_err': float(np.median(err)), 'val_lt1': float(np.mean(err < 1.0))}
        history.append(row)
        print('SKY_EPOCH', job['family'], row, flush=True)
        if row['val_mean_err'] < best['mean_err']:
            best = {'mean_err': row['val_mean_err'], 'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'epoch': epoch}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'best': best, 'history': history}, out_dir / 'sky_cnn.pt')
    pd.DataFrame(history).to_csv(out_dir / 'sky_cnn_history.csv', index=False)
    return model, device, history, best


def log_gaussian_overlap(mu: np.ndarray, sigma: float, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    sep = base.angular_sep_from_unit(mu[a], mu[c])
    var = 2.0 * sigma * sigma
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def feature_matrix(time_obs, sky_mu, sky_sigma, scores, ranks, a, c):
    return np.column_stack([
        log1p_delta_time_obs(time_obs, a, c),
        log_gaussian_overlap(sky_mu, sky_sigma, a, c),
        scores[a, c].astype(np.float32),
        (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32),
    ]).astype(np.float32)


def train_reranker(time_obs, gt, sky_mu, sky_sigma, scores, ranks, job):
    rng = np.random.default_rng(91000 + (0 if job['family'] == 'SIS' else 1))
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
    X = feature_matrix(time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs, gt, sky_mu, sky_sigma, scores, ranks):
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        a = np.repeat(rows.astype(np.int32), n)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out)
    return {'r@1': float(np.mean(r <= 1)), 'r@5': float(np.mean(r <= 5)), 'r@10': float(np.mean(r <= 10)), 'r@50': float(np.mean(r <= 50)), 'r@100': float(np.mean(r <= 100)), 'r@500': float(np.mean(r <= 500)), 'median_rank': float(np.median(r)), 'valid': int(len(valid))}


def run_job(job: dict):
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD', name, flush=True)
    _, train_ds, train_raw, train_time, train_gt, _ = load_pack(job, 'train', out_dir / 'train', False)
    _, val_ds, val_raw, val_time, val_gt, val_scores = load_pack(job, 'val', out_dir / 'val', True)
    _, test_ds, test_raw, test_time, test_gt, test_scores = load_pack(job, 'test', out_dir / 'test', True)
    model, device, history, best = train_sky_model(job, train_ds, train_raw, val_ds, val_raw, out_dir)
    val_mu = predict_mu(model, val_ds, device)
    test_mu = predict_mu(model, test_ds, device)
    val_true = base.unit_vectors(val_raw).astype(np.float32)
    val_err = angular_error(val_mu, val_true)
    sky_sigma = float(max(MIN_SKY_SIGMA, np.median(val_err)))
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    Xv, yv = train_reranker(val_time, val_gt, val_mu, sky_sigma, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'method': 'ligo_cnn_sky_predictor_rerank',
        'features': 'log1p_delta_time_obs,cnn_predicted_log_sky_overlap,waveform_score,waveform_reciprocal_rank',
        'sky_model': 'SkyCNN_dual_detector_waveform',
        'sky_best_epoch': int(best['epoch']),
        'sky_sigma_rad': sky_sigma,
        'sky_val_mean_error_rad': float(val_err.mean()),
        'sky_val_median_error_rad': float(np.median(val_err)),
        'sky_val_frac_err_lt_1p0': float(np.mean(val_err < 1.0)),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **eval_full(clf, test_time, test_gt, test_mu, sky_sigma, test_scores, test_ranks),
    }
    pd.DataFrame([row]).to_csv(out_dir / 'summary.csv', index=False)
    (out_dir / 'summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(row, flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in JOBS:
        rows.append(run_job(job))
        pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
