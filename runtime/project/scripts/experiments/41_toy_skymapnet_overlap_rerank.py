from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import healpy as hp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')

OUT_ROOT = Path('runs/toy_skymapnet_overlap_rerank_20260602')
SOURCE_ROOT = Path('runs/waveform_predicted_skymap_rerank_20260602')
JOBS = [
    {'detector': 'ET', 'mode': 'noisy', 'family': 'SIS'},
    {'detector': 'ET', 'mode': 'noisy', 'family': 'PM'},
    {'detector': 'LIGO', 'mode': 'noisy', 'family': 'SIS'},
    {'detector': 'LIGO', 'mode': 'noisy', 'family': 'PM'},
]
NSIDE = 8
N_PIX = hp.nside2npix(NSIDE)
TOY_SIGMA_RAD = 0.35
EPOCHS = 12
BATCH_SIZE = 512
NEG_PER_POS = 500
CHUNK_ROWS = 16


class SkyMapNet(nn.Module):
    def __init__(self, in_dim: int, n_pix: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 384), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(384, n_pix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def healpix_unit_vectors(nside: int) -> np.ndarray:
    pix = np.arange(hp.nside2npix(nside))
    vec = hp.pix2vec(nside, pix, nest=False)
    return np.column_stack(vec).astype(np.float32)


PIXELS = healpix_unit_vectors(NSIDE)


def toy_skymap_labels(obs: pd.DataFrame, sigma: float = TOY_SIGMA_RAD) -> np.ndarray:
    u = base.unit_vectors(obs).astype(np.float32)
    cosang = np.clip(u @ PIXELS.T, -1.0, 1.0)
    ang2 = np.arccos(cosang) ** 2
    logits = -ang2 / (2.0 * sigma * sigma)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p


def map_centroid_error(probs: np.ndarray, obs: pd.DataFrame) -> tuple[float, float, float, float]:
    pred = probs @ PIXELS
    pred = base.normalize_vectors(pred)
    true = base.unit_vectors(obs)
    err = base.angular_sep_from_unit(pred, true)
    return float(np.mean(err)), float(np.median(err)), float(np.mean(err < 0.5)), float(np.mean(err < 1.0))


def train_skymapnet(train_emb: np.ndarray, train_obs: pd.DataFrame, val_emb: np.ndarray, val_obs: pd.DataFrame, out_dir: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_train = toy_skymap_labels(train_obs)
    y_val = toy_skymap_labels(val_obs)
    model = SkyMapNet(train_emb.shape[1], N_PIX).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(train_emb.astype(np.float32)), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    best = None
    history = []
    val_x = torch.from_numpy(val_emb.astype(np.float32)).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logp = torch.log_softmax(model(xb), dim=1)
            loss = -(yb * logp).sum(dim=1).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logp = torch.log_softmax(model(val_x), dim=1)
            val_loss = float((-(val_y * val_logp).sum(dim=1).mean()).detach().cpu())
        row = {'epoch': epoch, 'train_kl': float(np.mean(losses)), 'val_kl': val_loss}
        history.append(row)
        if best is None or val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(row, flush=True)
    if best is not None:
        model.load_state_dict(best[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'skymapnet_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return model


def predict_maps(model: nn.Module, emb: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(emb), batch_size):
            xb = torch.from_numpy(emb[start:start + batch_size].astype(np.float32)).to(device)
            p = torch.softmax(model(xb), dim=1).detach().cpu().numpy().astype(np.float32)
            out.append(p)
    return np.concatenate(out, axis=0)


def overlap_features(maps: np.ndarray, anchors: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omin = np.empty(len(anchors), dtype=np.float32)
    obc = np.empty(len(anchors), dtype=np.float32)
    sqrt_maps = np.sqrt(np.maximum(maps, 0.0)).astype(np.float32)
    step = 65536
    for start in range(0, len(anchors), step):
        sl = slice(start, min(start + step, len(anchors)))
        pi = maps[anchors[sl]]
        pj = maps[cands[sl]]
        omin[sl] = np.minimum(pi, pj).sum(axis=1)
        obc[sl] = (sqrt_maps[anchors[sl]] * sqrt_maps[cands[sl]]).sum(axis=1)
    return omin, obc


def feature_matrix(time_obs: pd.DataFrame, maps: np.ndarray, scores: np.ndarray, ranks: np.ndarray, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    omin, obc = overlap_features(maps, anchors, cands)
    return np.column_stack([
        base.log1p_delta_time_obs(time_obs, anchors, cands),
        omin,
        obc,
        scores[anchors, cands],
        1.0 / np.maximum(ranks[anchors, cands], 1),
    ]).astype(np.float32)


def train_examples(time_obs: pd.DataFrame, gt: np.ndarray, maps: np.ndarray, scores: np.ndarray, ranks: np.ndarray, job: dict):
    rng = np.random.default_rng(95000 + abs(hash((job['detector'], job['mode'], job['family']))) % 10000)
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
    X = feature_matrix(time_obs, maps, scores, ranks, anchors, cands)
    order = rng.permutation(len(y))
    return X[order], y[order]


def eval_full(clf, time_obs: pd.DataFrame, gt: np.ndarray, maps: np.ndarray, scores: np.ndarray, ranks: np.ndarray):
    valid = np.flatnonzero(gt >= 0)
    n = len(time_obs)
    out_ranks = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        anchors = np.repeat(rows.astype(np.int32), n)
        cands = np.tile(np.arange(n, dtype=np.int32), len(rows))
        X = feature_matrix(time_obs, maps, scores, ranks, anchors, cands)
        pred = clf.predict_proba(X)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out_ranks.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out_ranks)
    return {
        'rerank_r@1': float(np.mean(r <= 1)),
        'rerank_r@5': float(np.mean(r <= 5)),
        'rerank_r@10': float(np.mean(r <= 10)),
        'rerank_r@50': float(np.mean(r <= 50)),
        'rerank_r@100': float(np.mean(r <= 100)),
        'rerank_r@500': float(np.mean(r <= 500)),
        'rerank_median_true_rank': float(np.median(r)),
        'valid': int(len(valid)),
    }


def load_cached(job: dict, split: str):
    real = next(j for j in base.JOBS if j['detector'] == job['detector'] and j['mode'] == job['mode'] and j['family'] == job['family'])
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    raw, time_obs, gt, emb, scores = base.load_split(real, split, SOURCE_ROOT / name / split)
    return real, raw, time_obs, gt, emb, scores


def run_one(job: dict) -> dict:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('RUN', job, flush=True)
    real, train_raw, train_time, train_gt, train_emb, train_scores = load_cached(job, 'train')
    _, val_raw, val_time, val_gt, val_emb, val_scores = load_cached(job, 'val')
    _, test_raw, test_time, test_gt, test_emb, test_scores = load_cached(job, 'test')

    model = train_skymapnet(train_emb, train_raw, val_emb, val_raw, out_dir)
    val_maps = predict_maps(model, val_emb)
    test_maps = predict_maps(model, test_emb)
    sky_mean, sky_median, sky_acc05, sky_acc10 = map_centroid_error(val_maps, val_raw)

    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    Xv, yv = train_examples(val_time, val_gt, val_maps, val_scores, val_ranks, job)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'method': 'toy_skymapnet_overlap_catalog_rerank',
        'features': 'log1p_delta_time_obs,predicted_O_min,predicted_O_BC,waveform_score,waveform_reciprocal_rank',
        'nside': NSIDE,
        'n_pix': N_PIX,
        'toy_sigma_rad': TOY_SIGMA_RAD,
        'sky_val_mean_centroid_error_rad': sky_mean,
        'sky_val_median_centroid_error_rad': sky_median,
        'sky_val_frac_err_lt_0p5': sky_acc05,
        'sky_val_frac_err_lt_1p0': sky_acc10,
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(real['run']),
        **eval_full(clf, test_time, test_gt, test_maps, test_scores, test_ranks),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(row, flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in JOBS:
        rows.append(run_one(job))
        pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    (OUT_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
