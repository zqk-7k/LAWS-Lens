#!/usr/bin/env python3
"""Validation-only multi-positive waveform representation repair.

Each lensed source contributes four views: noisy image A, noisy image B,
clean image A, and clean image B.  Every other view of the same source is a
positive for an anchor.  This removes the false-negative error created when
pair-wise noisy/clean auxiliary examples from one source co-occur in NT-Xent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, peak_flip_channels, split_indices, zscore_channels
from matchgw.matching import similarity_matrix
from scripts.experiments.mainline_uncertainty_common import seed_everything
from scripts.real_search.physical_common import effective_rank, write_json


def load_spectrogram_encoder():
    spec = importlib.util.spec_from_file_location("spectrogram_pilot", REPO / "scripts" / "experiments" / "26_spectrogram_encoder_pilot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Full24SpectrogramEncoder


class LensedFourViewDataset(Dataset):
    def __init__(self, arrays, indices: np.ndarray, seed: int) -> None:
        if arrays.l1_pure is None or arrays.l2_pure is None:
            raise ValueError("Clean auxiliary arrays are required")
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def prepare(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32).copy()
        y = peak_flip_channels(y)
        y = np.roll(y, int(self.rng.integers(-128, 129)), axis=-1)
        y *= float(1.0 + self.rng.uniform(-0.1, 0.1))
        y += self.rng.normal(0.0, 0.005 * (y.std(axis=-1, keepdims=True) + 1e-8), size=y.shape)
        return zscore_channels(y)

    def __getitem__(self, item: int) -> torch.Tensor:
        idx = int(self.indices[item])
        views = (
            self.arrays.l1[idx],
            self.arrays.l2[idx],
            self.arrays.l1_pure[idx],
            self.arrays.l2_pure[idx],
        )
        return torch.from_numpy(np.stack([self.prepare(view) for view in views]))


def multi_positive_supcon(embedding: torch.Tensor, source_id: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    z = nn.functional.normalize(embedding.float(), dim=-1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = source_id[:, None].eq(source_id[None, :]) & ~self_mask
    if not torch.all(positive.sum(dim=1) > 0):
        raise RuntimeError("Each anchor must have another view of the same source")
    logits = logits.masked_fill(self_mask, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -((log_probability * positive).sum(dim=1) / positive.sum(dim=1)).mean()


@torch.no_grad()
def evaluate(model: nn.Module, dataset: EvaluationSet, batch_size: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    rows = []
    model.eval()
    for x in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rows.append(model(x.cuda(non_blocking=True)).float().cpu().numpy())
    z = np.concatenate(rows)
    score = similarity_matrix(z)
    truth = ground_truth_partner(dataset.meta)
    ranks = []
    for query, partner in enumerate(truth):
        if partner >= 0:
            ranks.append(1 + int(np.sum(score[query] > score[query, partner])))
    ranks = np.asarray(ranks, dtype=np.int32)
    upper = score[np.triu_indices(len(score), k=1)]
    return {
        "n_events": int(len(dataset)),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "embedding_effective_rank": effective_rank(z),
        "cosine_median": float(np.median(upper)),
        "cosine_std": float(np.std(upper)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--family", choices=("SIS", "PM"), required=True)
    parser.add_argument("--seed", type=int, default=202607211)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source-batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=24)
    args = parser.parse_args()

    v3_spec = importlib.util.spec_from_file_location("v3", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
    v3 = importlib.util.module_from_spec(v3_spec)
    assert v3_spec.loader is not None
    v3_spec.loader.exec_module(v3)
    cfg = v3.training_config(args.seed_root, args.family, args.seed, args.samples, args.epochs, args.source_batch_size)
    cfg.use_pure_aux = True
    arrays = load_match_arrays(cfg)
    parts = split_indices(args.samples, args.samples, cfg)
    train = LensedFourViewDataset(arrays, parts["lensed"]["train"], cfg.seed + 2700)
    val = EvaluationSet(arrays, parts["lensed"]["val"], parts["unlensed"]["val"], cfg)
    loader = DataLoader(train, batch_size=args.source_batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    Encoder = load_spectrogram_encoder()
    seed_everything(cfg.seed + 2700)
    model = Encoder(base_channels=args.base_channels).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    history = []
    best_key = None
    best_validation = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for views in loader:
            batch, n_views, channels, samples = views.shape
            x = views.reshape(batch * n_views, channels, samples).cuda(non_blocking=True)
            source = torch.arange(batch, device=x.device).repeat_interleave(n_views)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = multi_positive_supcon(model(x), source)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "epoch_s": float(time.perf_counter() - started)}
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            current = evaluate(model, val, args.source_batch_size * 2)
            row.update({f"val_{key}": value for key, value in current.items()})
            key = (current["r_at_10"], current["r_at_1"], -current["median_rank"], current["embedding_effective_rank"])
            if best_key is None or key > best_key:
                best_key = key
                best_validation = current
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row), flush=True)

    output = args.seed_root / "waveform_gate" / f"{args.family.lower()}_full24_multiview_supcon_spectrogram_pilot"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "validation_selected_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "family": args.family,
            "architecture": "Full24SpectrogramEncoder",
            "objective": "four-view multi-positive supervised contrastive",
            "base_channels": args.base_channels,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    summary = {
        "family": args.family,
        "architecture": "Full24SpectrogramEncoder",
        "objective": "four-view multi-positive supervised contrastive",
        "views_per_source": ["noisy_A", "noisy_B", "clean_A", "clean_B"],
        "false_negative_policy": "All views from the same source are positives; only other source systems are negatives.",
        "selection_split": "validation only",
        "heldout_test_was_evaluated": False,
        "best_validation": best_validation,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "multiview_supcon_pilot_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
