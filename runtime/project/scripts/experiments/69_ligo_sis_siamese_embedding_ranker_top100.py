from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 第 68 个实验说明：直接拼接两条波形做 pair CNN 对 LIGO noisy SIS 效果很差。
# 本脚本改用更适合透镜像的 Siamese embedding：
#   1. 两条事件分别经过同一个 waveform encoder；
#   2. 用 |za-zb| 与 za*zb 表示“同源相似性”，避免强制同一时刻逐点对齐；
#   3. 每个 anchor 的候选列表用 softmax ranking loss，直接优化候选内排序。
m = importlib.import_module("scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank")
base = m.base

OUT_ROOT = Path("runs/ligo_sis_siamese_embedding_ranker_top100_20260608")
PROB_ROOT = Path("runs/ligo_sis_grid18_rank_fusion_20260604")

EPS = 1e-8
ALPHA = 2.0
K_EACH = 100
NEG_PER_ANCHOR = 64
TARGET_LEN = 512
BATCH_SIZE = 96
EPOCHS = 18
LR = 7e-4
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


def compress_waveform(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    c, t = x.shape
    if t >= TARGET_LEN:
        step = t // TARGET_LEN
        y = x[:, : step * TARGET_LEN].reshape(c, TARGET_LEN, step).mean(axis=2)
    else:
        y = np.zeros((c, TARGET_LEN), dtype=np.float32)
        y[:, :t] = x
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (y.std(axis=1, keepdims=True) + 1e-6)
    return y.astype(np.float32)


def materialize_waves(ds) -> np.ndarray:
    out = []
    for i in range(len(ds)):
        x = ds[i]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        out.append(compress_waveform(x))
        if (i + 1) % 1000 == 0:
            print("MATERIALIZE_WAVES", i + 1, flush=True)
    return np.stack(out).astype(np.float32)


def dt_mat(time_obs: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    n = len(time_obs)
    allc = np.arange(n, dtype=np.int32)
    out = np.empty((len(rows), n), np.float32)
    for i, a in enumerate(rows):
        out[i] = -m.log1p_delta_time_obs(time_obs, np.full(n, int(a), dtype=np.int32), allc)
    return out


def candidate_sets(time_obs, scores, prob, rows):
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


def scalar_features(scores, ranks, sky_row, dt_row, a: int, cols: np.ndarray) -> np.ndarray:
    rr = ranks[a, cols].astype(np.float32)
    x = np.column_stack(
        [
            scores[a, cols].astype(np.float32),
            1.0 / np.maximum(rr, 1.0),
            -np.log1p(np.maximum(rr, 1.0)),
            sky_row[cols].astype(np.float32),
            dt_row[cols].astype(np.float32),
            (sky_row[cols] * dt_row[cols]).astype(np.float32),
        ]
    ).astype(np.float32)
    return x


def build_anchor_lists(time_obs, gt, scores, ranks, prob):
    rng = np.random.default_rng(69001)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    cand_list, sky, dt = candidate_sets(time_obs, scores, prob, valid)
    anchors, partners, scalars = [], [], []
    for rp, a in enumerate(valid):
        p = int(gt[int(a)])
        cand = cand_list[rp]
        if p not in set(map(int, cand)):
            cand = np.concatenate([cand, np.array([p], dtype=np.int32)])
        neg = cand[cand != p]
        if len(neg) > NEG_PER_ANCHOR:
            hard_score = scores[int(a), neg] + 0.20 * sky[rp, neg] + 0.25 * dt[rp, neg]
            hard = neg[np.argpartition(-hard_score, NEG_PER_ANCHOR - 1)[:NEG_PER_ANCHOR]]
            neg = rng.choice(hard, size=NEG_PER_ANCHOR, replace=False).astype(np.int32)
        cols = np.concatenate([np.array([p], dtype=np.int32), neg.astype(np.int32)])
        anchors.append(int(a))
        partners.append(cols)
        scalars.append(scalar_features(scores, ranks, sky[rp], dt[rp], int(a), cols))
        if (rp + 1) % 500 == 0:
            print("BUILD_ANCHOR_LISTS", rp + 1, flush=True)
    return np.asarray(anchors, dtype=np.int32), partners, scalars


class AnchorListDataset(Dataset):
    def __init__(self, waves, anchors, partners, scalars) -> None:
        self.waves = waves
        self.anchors = anchors
        self.partners = partners
        self.scalars = scalars

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int):
        a = int(self.anchors[idx])
        cols = self.partners[idx]
        return (
            torch.from_numpy(self.waves[a]),
            torch.from_numpy(self.waves[cols]),
            torch.from_numpy(self.scalars[idx]),
            torch.zeros((), dtype=torch.long),
        )


def collate_anchor_lists(batch):
    a, b, s, y = zip(*batch)
    return torch.stack(a), torch.stack(b), torch.stack(s), torch.stack(y)


class WaveEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, emb_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 48, 17, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 80, 13, stride=2, padding=6, bias=False),
            nn.BatchNorm1d(80),
            nn.GELU(),
            nn.Conv1d(80, 128, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 160, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.Conv1d(160, 192, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(192 * 2, 192), nn.GELU(), nn.Dropout(0.15), nn.Linear(192, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.float())
        z = self.head(torch.cat([y.mean(dim=-1), y.amax(dim=-1)], dim=1))
        return F.normalize(z, dim=1)


class SiameseRanker(nn.Module):
    def __init__(self, scalar_dim: int = 6, emb_dim: int = 128) -> None:
        super().__init__()
        self.encoder = WaveEncoder(2, emb_dim)
        self.scalar = nn.Sequential(nn.LayerNorm(scalar_dim), nn.Linear(scalar_dim, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 3 + 32, 192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def score(self, anchor_wave, partner_waves, scalar):
        bsz, k, ch, t = partner_waves.shape
        za = self.encoder(anchor_wave)
        zb = self.encoder(partner_waves.reshape(bsz * k, ch, t)).reshape(bsz, k, -1)
        zae = za[:, None, :].expand_as(zb)
        feat = torch.cat([torch.abs(zae - zb), zae * zb, zb, self.scalar(scalar.float())], dim=2)
        return self.head(feat).squeeze(2)

    def forward(self, anchor_wave, partner_waves, scalar):
        return self.score(anchor_wave, partner_waves, scalar)


def train_model(waves, anchors, partners, scalars):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = AnchorListDataset(waves, anchors, partners, scalars)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_anchor_lists, pin_memory=(device == "cuda"))
    model = SiameseRanker().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses, acc1, acc10 = [], [], []
        for aw, pw, sf, target in loader:
            aw = aw.to(device, non_blocking=True)
            pw = pw.to(device, non_blocking=True)
            sf = sf.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(aw, pw, sf)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rank = 1 + torch.sum(logits > logits[:, :1], dim=1)
            losses.append(float(loss.detach().cpu()))
            acc1.append(float((rank <= 1).float().mean().detach().cpu()))
            acc10.append(float((rank <= 10).float().mean().detach().cpu()))
        sched.step()
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "train_r@1": float(np.mean(acc1)), "train_r@10": float(np.mean(acc10))}
        history.append(row)
        print("SIAMESE_RANK_EPOCH", row, flush=True)
    return model, device, history


def predict_anchor(model, device, waves, a: int, cols: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        aw = torch.from_numpy(waves[int(a)][None]).to(device)
        for start in range(0, len(cols), 512):
            cc = cols[start : start + 512]
            pw = torch.from_numpy(waves[cc][None]).to(device)
            sf = torch.from_numpy(scalars[start : start + 512][None]).to(device)
            out.append(model(aw, pw, sf).squeeze(0).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def eval_candidate(model, device, waves, time_obs, gt, scores, ranks, prob):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    all_ranks, in_cand = [], []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start : start + CHUNK_ROWS]
        cand_list, sky, dt = candidate_sets(time_obs, scores, prob, rows)
        for rp, a in enumerate(rows):
            p = int(gt[int(a)])
            cand = cand_list[rp]
            hit = p in set(map(int, cand))
            in_cand.append(hit)
            if not hit:
                all_ranks.append(len(cand) + 1)
                continue
            sf = scalar_features(scores, ranks, sky[rp], dt[rp], int(a), cand)
            pred = predict_anchor(model, device, waves, int(a), cand, sf)
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
    anchors, partners, scalars = build_anchor_lists(train_time, train_gt, train_scores, train_ranks, train_prob)
    print("TRAIN_ANCHOR_LISTS", len(anchors), "list_len", len(partners[0]), flush=True)
    model, device, history = train_model(train_waves, anchors, partners, scalars)
    torch.save({"model": model.state_dict(), "history": history}, OUT_ROOT / "siamese_embedding_ranker.pt")
    pd.DataFrame(history).to_csv(OUT_ROOT / "history.csv", index=False)
    row = {
        "method": "siamese_embedding_softmax_ranker_top100_union",
        "alpha": ALPHA,
        "k_each": K_EACH,
        "neg_per_anchor": NEG_PER_ANCHOR,
        **eval_candidate(model, device, test_waves, test_time, test_gt, test_scores, test_ranks, test_prob),
    }
    pd.DataFrame([row]).to_csv(OUT_ROOT / "summary.csv", index=False)
    print(row, flush=True)


if __name__ == "__main__":
    main()
