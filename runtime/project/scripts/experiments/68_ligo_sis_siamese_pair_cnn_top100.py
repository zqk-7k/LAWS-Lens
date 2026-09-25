from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 复用当前最佳 LIGO SIS sky-map 与 catalog 检索代码，保证数据划分、ground truth、
# trigger_time_obs、waveform score 与前面实验完全一致。
m = importlib.import_module("scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank")
base = m.base

OUT_ROOT = Path("runs/ligo_sis_siamese_pair_cnn_top100_20260605")
PROB_ROOT = Path("runs/ligo_sis_grid18_rank_fusion_20260604")

EPS = 1e-8
ALPHA = 2.0
K_EACH = 100
NEG_PER_ANCHOR = 48
TARGET_LEN = 512
BATCH_SIZE = 256
EPOCHS = 10
LR = 8e-4
WEIGHT_DECAY = 1e-4
CHUNK_ROWS = 32


def sharpen(prob: np.ndarray, alpha: float) -> np.ndarray:
    p = np.power(np.maximum(prob, EPS), alpha).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p


def load_split(job: dict, split: str):
    _, ds, _, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    prob = sharpen(np.load(PROB_ROOT / f"{split}_prob.npy").astype(np.float32), ALPHA)
    ranks = base.row_ranks(scores)
    return ds, time_obs.reset_index(drop=True), gt, scores.astype(np.float32), ranks.astype(np.int32), prob


def compress_waveform(x: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """把原始波形压缩到固定长度，保留每个探测器通道的整体形态。"""
    x = x.astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected waveform shape [channels,time], got {x.shape}")
    c, t = x.shape
    if t >= target_len:
        step = t // target_len
        trim = step * target_len
        y = x[:, :trim].reshape(c, target_len, step).mean(axis=2)
    else:
        y = np.zeros((c, target_len), dtype=np.float32)
        y[:, :t] = x
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (y.std(axis=1, keepdims=True) + 1e-6)
    return y.astype(np.float32)


def materialize_waves(ds) -> np.ndarray:
    waves = []
    for i in range(len(ds)):
        x = ds[i]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        waves.append(compress_waveform(x))
        if (i + 1) % 1000 == 0:
            print("MATERIALIZE_WAVES", i + 1, flush=True)
    return np.stack(waves).astype(np.float32)


def dt_mat(time_obs: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    n = len(time_obs)
    allc = np.arange(n, dtype=np.int32)
    out = np.empty((len(rows), n), np.float32)
    for i, a in enumerate(rows):
        out[i] = -m.log1p_delta_time_obs(time_obs, np.full(n, int(a), dtype=np.int32), allc)
    return out


def candidate_sets(time_obs: pd.DataFrame, scores: np.ndarray, ranks: np.ndarray, prob: np.ndarray, rows: np.ndarray):
    # 候选池沿用前面实验：waveform score、trigger_time_obs、predicted sky overlap 各取 Top100 并集。
    sky = np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)
    dt = dt_mat(time_obs, rows)
    cand_list = []
    for rp, a in enumerate(rows):
        cand = set()
        for arr in (scores[int(a)].astype(np.float32), sky[rp], dt[rp]):
            s = arr.copy()
            s[int(a)] = -np.inf
            top = np.argpartition(-s, min(K_EACH, len(s) - 1))[:K_EACH]
            cand.update(map(int, top))
        cand.discard(int(a))
        cand_list.append(np.array(sorted(cand), dtype=np.int32))
    return cand_list, sky, dt


def pair_scalar_features(scores: np.ndarray, ranks: np.ndarray, sky_row: np.ndarray, dt_row: np.ndarray, a: int, cols: np.ndarray) -> np.ndarray:
    rr = ranks[a, cols].astype(np.float32)
    out = np.column_stack(
        [
            scores[a, cols].astype(np.float32),
            1.0 / np.maximum(rr, 1.0),
            -np.log1p(np.maximum(rr, 1.0)),
            sky_row[cols].astype(np.float32),
            dt_row[cols].astype(np.float32),
            (sky_row[cols] * dt_row[cols]).astype(np.float32),
        ]
    )
    return out.astype(np.float32)


def build_train_pairs(time_obs, gt, scores, ranks, prob):
    rng = np.random.default_rng(68001)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    cand_list, sky, dt = candidate_sets(time_obs, scores, ranks, prob, valid)
    anchors, partners, scalar_rows, labels = [], [], [], []
    for rp, a in enumerate(valid):
        p = int(gt[int(a)])
        cand = cand_list[rp]
        if p not in set(map(int, cand)):
            cand = np.concatenate([cand, np.array([p], dtype=np.int32)])
        neg = cand[cand != p]
        # 优先使用候选池里更靠前的 hard negatives，同时保留少量随机性，避免只学单一排序信号。
        if len(neg) > NEG_PER_ANCHOR:
            hard_score = scores[int(a), neg] + 0.15 * sky[rp, neg] + 0.15 * dt[rp, neg]
            hard = neg[np.argpartition(-hard_score, min(NEG_PER_ANCHOR, len(neg)) - 1)[:NEG_PER_ANCHOR]]
            if len(hard) > NEG_PER_ANCHOR:
                hard = rng.choice(hard, size=NEG_PER_ANCHOR, replace=False)
            neg = hard.astype(np.int32)
        cols = np.concatenate([np.array([p], dtype=np.int32), neg.astype(np.int32)])
        scalars = pair_scalar_features(scores, ranks, sky[rp], dt[rp], int(a), cols)
        anchors.extend([int(a)] * len(cols))
        partners.extend(map(int, cols))
        scalar_rows.append(scalars)
        labels.extend([1] + [0] * (len(cols) - 1))
        if (rp + 1) % 500 == 0:
            print("BUILD_TRAIN_PAIRS", rp + 1, flush=True)
    Xs = np.vstack(scalar_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    order = rng.permutation(len(y))
    return np.asarray(anchors, dtype=np.int32)[order], np.asarray(partners, dtype=np.int32)[order], Xs[order], y[order]


class PairDataset(Dataset):
    def __init__(self, waves: np.ndarray, anchors: np.ndarray, partners: np.ndarray, scalars: np.ndarray, labels: np.ndarray) -> None:
        self.waves = waves
        self.anchors = anchors
        self.partners = partners
        self.scalars = scalars
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        a = self.waves[int(self.anchors[idx])]
        b = self.waves[int(self.partners[idx])]
        # 输入包含两个事件的双探测器波形、差分和绝对差分，帮助模型直接学习 pair-level 相似性。
        pair = np.concatenate([a, b, a - b, np.abs(a - b)], axis=0)
        return torch.from_numpy(pair), torch.from_numpy(self.scalars[idx]), torch.tensor(self.labels[idx])


class PairCNN(nn.Module):
    def __init__(self, in_channels: int = 8, scalar_dim: int = 6) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 17, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, 13, stride=2, padding=6, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 128, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 160, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
        )
        self.scalar = nn.Sequential(nn.LayerNorm(scalar_dim), nn.Linear(scalar_dim, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(160 * 2 + 32, 192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, pair: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        y = self.conv(pair.float())
        pooled = torch.cat([y.mean(dim=-1), y.amax(dim=-1)], dim=1)
        return self.head(torch.cat([pooled, self.scalar(scalar.float())], dim=1)).squeeze(1)


def train_model(waves, anchors, partners, scalars, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDataset(waves, anchors, partners, scalars, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    model = PairCNN().to(device)
    pos = float(labels.sum())
    neg = float(len(labels) - labels.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for pair, scalar, y in loader:
            pair = pair.to(device, non_blocking=True)
            scalar = scalar.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = criterion(model(pair, scalar), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print("PAIR_CNN_EPOCH", row, flush=True)
    return model, device, history


def predict_pairs(model, device, waves, a: int, cols: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(cols), 512):
            cc = cols[start : start + 512]
            aa = np.repeat(waves[int(a) : int(a) + 1], len(cc), axis=0)
            bb = waves[cc]
            pair = np.concatenate([aa, bb, aa - bb, np.abs(aa - bb)], axis=1)
            scalar = scalars[start : start + 512]
            logit = model(torch.from_numpy(pair).to(device), torch.from_numpy(scalar).to(device))
            outs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def eval_candidate(model, device, waves, time_obs, gt, scores, ranks, prob):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    all_ranks, in_cand = [], []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start : start + CHUNK_ROWS]
        cand_list, sky, dt = candidate_sets(time_obs, scores, ranks, prob, rows)
        for rp, a in enumerate(rows):
            p = int(gt[int(a)])
            cand = cand_list[rp]
            hit = p in set(map(int, cand))
            in_cand.append(hit)
            if not hit:
                all_ranks.append(len(cand) + 1)
                continue
            scalars = pair_scalar_features(scores, ranks, sky[rp], dt[rp], int(a), cand)
            pred = predict_pairs(model, device, waves, int(a), cand, scalars)
            true = pred[np.where(cand == p)[0][0]]
            all_ranks.append(int(1 + np.sum(pred > true)))
        print("EVAL_ROWS", start + len(rows), flush=True)
    r = np.asarray(all_ranks)
    return {
        "candidate_recall": float(np.mean(in_cand)),
        "r@1": float(np.mean(r <= 1)),
        "r@5": float(np.mean(r <= 5)),
        "r@10": float(np.mean(r <= 10)),
        "r@50": float(np.mean(r <= 50)),
        "r@100": float(np.mean(r <= 100)),
        "median_rank": float(np.median(r)),
        "valid": int(len(valid)),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    train_ds, train_time, train_gt, train_scores, train_ranks, train_prob = load_split(job, "val")
    test_ds, test_time, test_gt, test_scores, test_ranks, test_prob = load_split(job, "test")
    print("MATERIALIZE train waves", flush=True)
    train_waves = materialize_waves(train_ds)
    print("MATERIALIZE test waves", flush=True)
    test_waves = materialize_waves(test_ds)
    anchors, partners, scalars, labels = build_train_pairs(train_time, train_gt, train_scores, train_ranks, train_prob)
    print("TRAIN_PAIR_SHAPE", len(labels), "pos", int(labels.sum()), "neg", int((labels == 0).sum()), flush=True)
    model, device, history = train_model(train_waves, anchors, partners, scalars, labels)
    torch.save({"model": model.state_dict(), "history": history}, OUT_ROOT / "pair_cnn.pt")
    pd.DataFrame(history).to_csv(OUT_ROOT / "history.csv", index=False)
    metrics = eval_candidate(model, device, test_waves, test_time, test_gt, test_scores, test_ranks, test_prob)
    row = {"method": "siamese_pair_cnn_top100_union", "alpha": ALPHA, "k_each": K_EACH, "neg_per_anchor": NEG_PER_ANCHOR, **metrics}
    pd.DataFrame([row]).to_csv(OUT_ROOT / "summary.csv", index=False)
    print(row, flush=True)


if __name__ == "__main__":
    main()
