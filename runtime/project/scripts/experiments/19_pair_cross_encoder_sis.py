from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

base = importlib.import_module("scripts.experiments.17_waveform_reranker_existing")


class PairWaveDataset(Dataset):
    def __init__(self, waves: np.ndarray, pairs: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.waves = waves
        self.pairs = pairs.astype(np.int32)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        a = self.waves[i]
        b = self.waves[j]
        x = np.concatenate([a, b, np.abs(a - b), a * b], axis=0).astype(np.float32, copy=False)
        if self.labels is None:
            return torch.from_numpy(x), int(i), int(j)
        return torch.from_numpy(x), torch.tensor(float(self.labels[idx]), dtype=torch.float32)


class PairCrossEncoder(nn.Module):
    def __init__(self, in_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 192, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(192, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.float())).squeeze(-1)


def union_candidates(scores: list[np.ndarray], ens: np.ndarray, topk: int) -> list[list[int]]:
    orders = [base.topk_order(s, topk) for s in scores + [ens]]
    out = []
    for i in range(ens.shape[0]):
        cand = set()
        for o in orders:
            cand.update(map(int, o[i, :topk]))
        cand.discard(i)
        out.append(list(cand))
    return out


def make_train_pairs(gt: np.ndarray, candidates: list[list[int]], hard_per_anchor: int = 6):
    rng = np.random.default_rng(42)
    pairs = []
    labels = []
    for i, true_j in enumerate(gt):
        if true_j >= 0:
            pairs.append((i, int(true_j)))
            labels.append(1)
        neg = [j for j in candidates[i] if j != int(true_j)]
        if neg:
            take = min(hard_per_anchor, len(neg))
            for j in rng.choice(np.asarray(neg, dtype=np.int32), size=take, replace=False):
                pairs.append((i, int(j)))
                labels.append(0)
    return np.asarray(pairs, dtype=np.int32), np.asarray(labels, dtype=np.float32)


def eval_ranker(scores: list[tuple[int, int, float]], gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    by: dict[int, list[tuple[float, int]]] = {}
    for i, j, s in scores:
        by.setdefault(int(i), []).append((float(s), int(j)))
    ranks = []
    for i in valid:
        rank = 10**9
        for r, (_, j) in enumerate(sorted(by.get(int(i), []), reverse=True), start=1):
            if j == int(gt[i]):
                rank = r
                break
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {"r@1": float(np.mean(ranks <= 1)), "r@5": float(np.mean(ranks <= 5)), "r@10": float(np.mean(ranks <= 10)), "r@50": float(np.mean(ranks <= 50)), "median_true_rank": float(np.median(ranks)), "valid": int(len(valid))}


def main():
    family = "SIS"
    out_dir = Path("runs/et10000_pair_cross_encoder_sis")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("load train scores", flush=True)
    _, _, train_ds, train_scores, train_ens, train_gt, base_cfg = base.load_score_sets(family, "train", out_dir / "train")
    print("load test scores", flush=True)
    _, _, test_ds, test_scores, test_ens, test_gt, _ = base.load_score_sets(family, "test", out_dir / "test")
    print("precompute waves", flush=True)
    train_waves = base.precompute_waveforms(train_ds, base_cfg)
    test_waves = base.precompute_waveforms(test_ds, base_cfg)

    print("make candidates", flush=True)
    train_cands = union_candidates(train_scores, train_ens, topk=30)
    test_cands = union_candidates(test_scores, test_ens, topk=50)
    train_pairs, train_labels = make_train_pairs(train_gt, train_cands, hard_per_anchor=6)
    test_pairs = np.asarray([(i, j) for i, row in enumerate(test_cands) for j in row], dtype=np.int32)
    print("pairs", len(train_pairs), "pos", int(train_labels.sum()), "test", len(test_pairs), flush=True)

    model = PairCrossEncoder().to(device)
    pos = float(train_labels.sum())
    neg = float(len(train_labels) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    loader = DataLoader(PairWaveDataset(train_waves, train_pairs, train_labels), batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, 9):
        model.train()
        total = 0.0
        seen = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu()) * len(y)
            seen += len(y)
        item = {"epoch": epoch, "loss": total / max(seen, 1)}
        history.append(item)
        print(json.dumps(item), flush=True)

    print("score test", flush=True)
    model.eval()
    scored = []
    test_loader = DataLoader(PairWaveDataset(test_waves, test_pairs, None), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for x, ii, jj in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p = torch.sigmoid(model(x)).float().cpu().numpy()
            scored.extend((int(i), int(j), float(s)) for i, j, s in zip(ii, jj, p))
    test = eval_ranker(scored, test_gt)
    pd.DataFrame(scored, columns=["anchor", "candidate", "p_hat"]).to_csv(out_dir / "test_cross_encoder_candidates.csv", index=False)
    result = {"family": family, "method": "waveform_pair_cross_encoder", "train_pairs": int(len(train_pairs)), "train_positive": int(pos), "train_negative": int(neg), "test_pairs": int(len(test_pairs)), "history": history, "test": test}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": model.state_dict(), "result": result}, out_dir / "model.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
