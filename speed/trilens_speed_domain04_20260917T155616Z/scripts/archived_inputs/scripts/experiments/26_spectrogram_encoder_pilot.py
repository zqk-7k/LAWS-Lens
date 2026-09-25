#!/usr/bin/env python3
"""Validation-only full-24-s H1/L1 spectrogram encoder pilot.

The waveform window, physical injection SNR, real off-source noise, PSD
whitening, and train/validation source split are inherited unchanged from the
physical v3 dataset.  This pilot only replaces global time-domain averaging
with a transient-aware time-frequency readout.  Architecture selection is
performed on validation retrieval; the held-out test is optional and disabled
by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import EvaluationSet, PairDataset, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from matchgw.models import NTXentLoss
from scripts.experiments.mainline_uncertainty_common import seed_everything
from scripts.real_search.physical_common import effective_rank, write_json


def groups(channels: int) -> int:
    for value in range(min(8, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Full24SpectrogramEncoder(nn.Module):
    """Log-magnitude STFT encoder with transient-aware pooling.

    Magnitude removes the arbitrary coalescence/Morse phase from the primary
    representation.  H1 and L1 remain separate input channels.  The model sees
    the complete 24-s window; no test-set or PE quantity enters the encoder.
    """

    def __init__(
        self,
        sample_rate: float = 2048.0,
        n_fft: int = 512,
        hop_length: int = 128,
        f_low: float = 40.0,
        f_high: float = 580.0,
        base_channels: int = 24,
        d_model: int = 192,
        emb_dim: int = 96,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        frequency = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        keep = (frequency >= f_low) & (frequency <= f_high)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer("frequency_keep", keep, persistent=False)
        c = int(base_channels)
        self.feature_channels = c * 4
        self.features = nn.Sequential(
            ConvBlock(2, c, (2, 2)),
            ConvBlock(c, c * 2, (2, 2)),
            ConvBlock(c * 2, c * 4, (2, 2)),
            ConvBlock(c * 4, c * 4, (1, 2)),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(c * 4, c, 1),
            nn.GELU(),
            nn.Conv2d(c, 1, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(c * 4 * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, emb_dim),
        )

    def pool_features(self, y: torch.Tensor) -> torch.Tensor:
        batch = y.shape[0]
        weights = torch.softmax(self.attention(y).flatten(2), dim=-1).reshape(batch, 1, y.shape[-2], y.shape[-1])
        attentive = (y * weights).sum(dim=(-2, -1))
        average = y.mean(dim=(-2, -1))
        maximum = y.amax(dim=(-2, -1))
        return torch.cat([attentive, average, maximum], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, detector, samples = x.shape
        spec = torch.stft(
            x.float().reshape(batch * detector, samples),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = torch.log1p(spec.abs()).reshape(batch, detector, -1, spec.shape[-1])
        magnitude = magnitude[:, :, self.frequency_keep, :]
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        y = self.features((magnitude - mean) / std)
        return F.normalize(self.head(self.pool_features(y)), dim=-1)


class TriggerAlignedDualScaleSpectrogramEncoder(Full24SpectrogramEncoder):
    """Full-window context plus a predeclared one-second trigger crop.

    GWOSC event GPS times are known before lens-pair ranking.  The local branch
    therefore uses no pair label or PE information; it prevents the short BBH
    transient from being diluted by the remaining 24-s noise context.
    """

    def __init__(self, *args, trigger_seconds: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trigger_samples = int(round(trigger_seconds * self.sample_rate))
        self.crop_n_fft = 256
        self.crop_hop = 64
        crop_frequency = torch.fft.rfftfreq(self.crop_n_fft, d=1.0 / self.sample_rate)
        self.register_buffer("crop_window", torch.hann_window(self.crop_n_fft), persistent=False)
        self.register_buffer("crop_frequency_keep", (crop_frequency >= 40.0) & (crop_frequency <= 580.0), persistent=False)
        old_head = self.head
        d_model = old_head[0].out_features
        emb_dim = old_head[-1].out_features
        self.head = nn.Sequential(
            nn.Linear(self.feature_channels * 6, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, emb_dim),
        )

    def _magnitude(self, x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        batch, detector, samples = x.shape
        spec = torch.stft(
            x.float().reshape(batch * detector, samples),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=False,
            return_complex=True,
        )
        magnitude = torch.log1p(spec.abs()).reshape(batch, detector, -1, spec.shape[-1])[:, :, keep, :]
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (magnitude - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = self._magnitude(x, self.n_fft, self.hop_length, self.window, self.frequency_keep)
        local_x = x[..., -self.trigger_samples :]
        local = self._magnitude(local_x, self.crop_n_fft, self.crop_hop, self.crop_window, self.crop_frequency_keep)
        full_features = self.pool_features(self.features(full))
        local_features = self.pool_features(self.features(local))
        return F.normalize(self.head(torch.cat([full_features, local_features], dim=1)), dim=-1)


@torch.no_grad()
def embed(model: nn.Module, dataset: EvaluationSet, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    rows = []
    for x in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rows.append(model(x.cuda(non_blocking=True)).float().cpu().numpy())
    return np.concatenate(rows)


def retrieval_metrics(model: nn.Module, dataset: EvaluationSet, batch_size: int, split: str) -> dict:
    z = embed(model, dataset, batch_size)
    score = similarity_matrix(z)
    truth = ground_truth_partner(dataset.meta)
    ranks = []
    for query, partner in enumerate(truth):
        if partner >= 0:
            ranks.append(1 + int(np.sum(score[query] > score[query, partner])))
    ranks = np.asarray(ranks, dtype=np.int32)
    upper = score[np.triu_indices(len(score), k=1)]
    return {
        "split": split,
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--use-pure-aux", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location("v3", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cfg = module.training_config(args.seed_root, args.family, args.seed, args.samples, args.epochs, args.batch_size)
    cfg = replace(cfg, use_pure_aux=args.use_pure_aux, batch_size=args.batch_size, epochs=args.epochs)
    arrays = load_match_arrays(cfg)
    parts = split_indices(args.samples, args.samples, cfg)
    train = PairDataset(arrays, parts["lensed"]["train"], np.asarray([], dtype=np.int64), cfg)
    val = EvaluationSet(arrays, parts["lensed"]["val"], parts["unlensed"]["val"], cfg)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    seed_everything(cfg.seed + 1700)
    model = Full24SpectrogramEncoder(base_channels=args.base_channels).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    loss_fn = NTXentLoss(tau=0.10)
    history = []
    best_key = None
    best_state = None
    best_validation = None
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for xa, xb in loader:
            optimizer.zero_grad(set_to_none=True)
            xa = xa.cuda(non_blocking=True)
            xb = xb.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xa), model(xb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "epoch_s": float(time.perf_counter() - started)}
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            current = retrieval_metrics(model, val, max(4, args.batch_size), "validation")
            row.update({f"val_{key}": value for key, value in current.items() if key != "split"})
            key = (current["r_at_10"], current["r_at_1"], -current["median_rank"], current["embedding_effective_rank"])
            if best_key is None or key > best_key:
                best_key = key
                best_validation = current
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row), flush=True)

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    auxiliary_tag = "pure_aux" if args.use_pure_aux else "no_pure_aux"
    output = args.seed_root / "waveform_gate" / f"{args.family.lower()}_full24_spectrogram_{auxiliary_tag}_pilot"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "validation_selected_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "family": args.family,
            "architecture": "Full24SpectrogramEncoder",
            "base_channels": args.base_channels,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    test_metrics = None
    if args.evaluate_test:
        model.load_state_dict(best_state)
        test = EvaluationSet(arrays, parts["lensed"]["test"], parts["unlensed"]["test"], cfg)
        test_metrics = retrieval_metrics(model, test, max(4, args.batch_size), "held-out test")
    summary = {
        "family": args.family,
        "architecture": "Full24SpectrogramEncoder",
        "input": "complete 24-s, 2048-Hz, PSD-whitened H1/L1 strain",
        "time_frequency_transform": "STFT n_fft=512, hop=128, Hann, 40-580 Hz, log magnitude",
        "normalization": "GroupNorm; no batch-statistic coupling",
        "positive_pairs": "independent-noise lensed A/B only",
        "unlensed_self_positive_pairs": False,
        "batch_size": args.batch_size,
        "use_pure_aux": args.use_pure_aux,
        "selection_split": "validation only",
        "heldout_test_was_evaluated": bool(args.evaluate_test),
        "best_validation": best_validation,
        "heldout_test": test_metrics,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "spectrogram_pilot_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
