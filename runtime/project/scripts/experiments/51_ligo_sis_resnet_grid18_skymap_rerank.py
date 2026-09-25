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

OUT_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
SOURCE_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
JOBS = [j for j in base.JOBS if j['detector'] == 'LIGO' and j['mode'] == 'noisy' and j['family'] == 'SIS']
RA_BINS = 36
DEC_BINS = 18
N_PIX = RA_BINS * DEC_BINS
SOFT_SIGMA = 0.28
EPOCHS = 14
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
NEG_PER_POS = 500
CHUNK_ROWS = 48
EPS = 1e-8


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


def grid_centers() -> np.ndarray:
    ra_edges = np.linspace(0.0, 2.0 * np.pi, RA_BINS + 1)
    dec_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, DEC_BINS + 1)
    ra = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    rr, dd = np.meshgrid(ra, dec)
    return np.column_stack([rr.reshape(-1), dd.reshape(-1)]).astype(np.float32)


CENTERS = grid_centers()
CENTER_UNIT = np.column_stack([
    np.cos(CENTERS[:, 1]) * np.cos(CENTERS[:, 0]),
    np.cos(CENTERS[:, 1]) * np.sin(CENTERS[:, 0]),
    np.sin(CENTERS[:, 1]),
]).astype(np.float32)


def soft_skymap(obs: pd.DataFrame, sigma: float = SOFT_SIGMA) -> np.ndarray:
    true = base.unit_vectors(obs).astype(np.float32)
    sep = np.arccos(np.clip(true @ CENTER_UNIT.T, -1.0, 1.0))
    logits = -(sep * sep) / (2.0 * sigma * sigma)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p


def map_center_error(prob: np.ndarray, obs: pd.DataFrame) -> np.ndarray:
    mu = prob @ CENTER_UNIT
    mu /= np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), EPS)
    true = base.unit_vectors(obs).astype(np.float32)
    return np.arccos(np.clip(np.sum(mu * true, axis=1), -1.0, 1.0))


class SkyMapDataset(Dataset):
    def __init__(self, ds: EvaluationSet, raw_obs: pd.DataFrame) -> None:
        self.ds = ds
        self.y = soft_skymap(raw_obs)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        return self.ds[idx], torch.from_numpy(self.y[idx])


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, dilation: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        pad = dilation * (kernel // 2)
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.main(x) + self.skip(x))


class SkyMapCNN(nn.Module):
    def __init__(self, in_channels: int = 2, n_pix: int = N_PIX) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 48, 31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            ResidualBlock1D(48, 64, 25, stride=4, dropout=0.05),
            ResidualBlock1D(64, 96, 19, stride=4, dropout=0.05),
            ResidualBlock1D(96, 128, 15, stride=2, dropout=0.08),
            ResidualBlock1D(128, 192, 11, stride=2, dropout=0.08),
            ResidualBlock1D(192, 256, 9, stride=2, dropout=0.10),
            ResidualBlock1D(256, 256, 7, stride=1, dilation=2, dropout=0.10),
            ResidualBlock1D(256, 256, 5, stride=1, dilation=4, dropout=0.10),
        )
        self.attn = nn.Sequential(nn.Conv1d(256, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, n_pix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(self.stem(x.float()))
        w = torch.softmax(self.attn(y), dim=-1)
        pooled = torch.cat([(y * w).sum(dim=-1), y.mean(dim=-1), y.amax(dim=-1)], dim=1)
        return self.head(pooled)

def predict_maps(model: nn.Module, ds: EvaluationSet, device: str) -> np.ndarray:
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    outs = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            outs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def train_model(job, train_ds, train_raw, val_ds, val_raw, out_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(SkyMapDataset(train_ds, train_raw), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_true = soft_skymap(val_raw)
    model = SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = {'val_kl': 999.0, 'val_mean_err': 999.0, 'state': None, 'epoch': 0}
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            logp = F.log_softmax(logits, dim=1)
            loss = F.kl_div(logp, y, reduction='batchmean')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        val_prob = predict_maps(model, val_ds, device)
        kl = float(np.mean(np.sum(val_true * (np.log(np.maximum(val_true, EPS)) - np.log(np.maximum(val_prob, EPS))), axis=1)))
        err = map_center_error(val_prob, val_raw)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'val_kl': kl, 'val_mean_err': float(err.mean()), 'val_median_err': float(np.median(err)), 'val_lt1': float(np.mean(err < 1.0))}
        history.append(row)
        print('GRID_EPOCH', job['family'], row, flush=True)
        if row['val_mean_err'] < best['val_mean_err']:
            best = {'val_kl': kl, 'val_mean_err': row['val_mean_err'], 'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'epoch': epoch}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'best': best, 'history': history}, out_dir / 'grid_skymap_cnn.pt')
    pd.DataFrame(history).to_csv(out_dir / 'grid_skymap_cnn_history.csv', index=False)
    return model, device, history, best


def map_overlap(prob: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.log(np.sum(prob[a] * prob[c], axis=1) + EPS).astype(np.float32)


def feature_matrix(time_obs, prob, scores, ranks, a, c):
    return np.column_stack([
        log1p_delta_time_obs(time_obs, a, c),
        map_overlap(prob, a, c),
        scores[a, c].astype(np.float32),
        (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32),
    ]).astype(np.float32)


def train_reranker(time_obs, gt, prob, scores, ranks, job):
    rng = np.random.default_rng(93000 + (0 if job['family'] == 'SIS' else 1))
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    pos_a = valid.astype(np.int32); pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    a = np.concatenate([pos_a, neg_a]); c = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    X = feature_matrix(time_obs, prob, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs, gt, prob, scores, ranks):
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        a = np.repeat(rows.astype(np.int32), n)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(time_obs, prob, scores, ranks, a, c)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out)
    return {'r@1': float(np.mean(r <= 1)), 'r@5': float(np.mean(r <= 5)), 'r@10': float(np.mean(r <= 10)), 'r@50': float(np.mean(r <= 50)), 'r@100': float(np.mean(r <= 100)), 'r@500': float(np.mean(r <= 500)), 'median_rank': float(np.median(r)), 'valid': int(len(valid))}


def run_job(job):
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD', name, flush=True)
    _, train_ds, train_raw, _, _, _ = load_pack(job, 'train', out_dir / 'train', False)
    _, val_ds, val_raw, val_time, val_gt, val_scores = load_pack(job, 'val', out_dir / 'val', True)
    _, test_ds, test_raw, test_time, test_gt, test_scores = load_pack(job, 'test', out_dir / 'test', True)
    model, device, history, best = train_model(job, train_ds, train_raw, val_ds, val_raw, out_dir)
    val_prob = predict_maps(model, val_ds, device)
    test_prob = predict_maps(model, test_ds, device)
    val_err = map_center_error(val_prob, val_raw)
    val_ranks = base.row_ranks(val_scores); test_ranks = base.row_ranks(test_scores)
    Xv, yv = train_reranker(val_time, val_gt, val_prob, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'], 'data_mode': job['mode'], 'family': job['family'],
        'method': 'ligo_sis_resnet_grid18_skymap_rerank',
        'features': 'log1p_delta_time_obs,predicted_grid18_skymap_overlap,waveform_score,waveform_reciprocal_rank',
        'sky_model': f'SkyMapCNN_grid_{DEC_BINS}x{RA_BINS}',
        'sky_best_epoch': int(best['epoch']),
        'sky_val_kl': float(best['val_kl']),
        'sky_val_mean_error_rad': float(val_err.mean()),
        'sky_val_median_error_rad': float(np.median(val_err)),
        'sky_val_frac_err_lt_1p0': float(np.mean(val_err < 1.0)),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **eval_full(clf, test_time, test_gt, test_prob, test_scores, test_ranks),
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
