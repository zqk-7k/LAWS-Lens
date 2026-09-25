#!/usr/bin/env python3
"""Train and validate one waveform-domain ablation configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal, stats
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT = Path("/root/autodl-tmp/gw-catalog")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from matchgw.models import InceptionAttentionEncoder1D, NTXentLoss  # noqa: E402
from scripts.experiments.mainline_uncertainty_common import seed_everything  # noqa: E402
from scripts.real_search.physical_common import effective_rank  # noqa: E402


MODEL_SAMPLE_RATE = 2048
MODEL_INPUT_SAMPLES = 4096
EMBEDDING_DIM = 96


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def zscore_channels(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    center = x.mean(axis=-1, keepdims=True)
    scale = np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    return ((x - center) / scale).astype(np.float32)


def make_window_view(values: np.ndarray, seconds: int) -> np.ndarray:
    """Create a 4096-point view with anti-aliased long-window resampling."""
    x = np.asarray(values, dtype=np.float32)
    samples = int(seconds * MODEL_SAMPLE_RATE)
    if x.shape[-1] < samples:
        raise ValueError(f"Need {samples} samples for {seconds}s; got {x.shape[-1]}")
    cropped = x[..., -samples:].copy()
    short = cropped[..., -MODEL_INPUT_SAMPLES:]
    peak = np.argmax(np.abs(short), axis=-1)
    signs = np.take_along_axis(short, peak[..., None], axis=-1)
    cropped *= np.where(signs < 0.0, -1.0, 1.0)
    if seconds > 2:
        down = int(seconds // 2)
        cropped = signal.resample_poly(cropped, up=1, down=down, axis=-1, window=("kaiser", 8.6))
    if cropped.shape[-1] != MODEL_INPUT_SAMPLES:
        raise RuntimeError(f"Unexpected {seconds}s view shape {cropped.shape}")
    return zscore_channels(cropped)


def cache_view(source: Path, target: Path, seconds: int, batch: int = 32) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    source_values = np.load(source, mmap_mode="r")
    output = np.lib.format.open_memmap(
        target, mode="w+", dtype=np.float16, shape=(*source_values.shape[:-1], MODEL_INPUT_SAMPLES)
    )
    for start in range(0, len(source_values), batch):
        stop = min(start + batch, len(source_values))
        output[start:stop] = make_window_view(np.asarray(source_values[start:stop]), seconds).astype(np.float16)
    output.flush()
    return target


def cache_development_views(root: Path, deployment: str, seconds: Iterable[int]) -> dict[int, dict[str, Path]]:
    data = root / "data/development" / deployment
    result: dict[int, dict[str, Path]] = {}
    inputs = {
        "sis_train_a": data / "sis_train_a.npy",
        "sis_train_b": data / "sis_train_b.npy",
        "pm_train_a": data / "pm_train_a.npy",
        "pm_train_b": data / "pm_train_b.npy",
        "sis_validation_a": data / "sis_validation_a.npy",
        "sis_validation_b": data / "sis_validation_b.npy",
        "pm_validation_a": data / "pm_validation_a.npy",
        "pm_validation_b": data / "pm_validation_b.npy",
        "unlensed_validation": data / "unlensed_validation.npy",
    }
    for sec in sorted(set(int(x) for x in seconds)):
        result[sec] = {}
        for label, source in inputs.items():
            target = root / "cache/window_views" / deployment / f"{label}_{sec}s.npy"
            result[sec][label] = cache_view(source, target, sec)
    return result


@dataclass
class Targets:
    mean: np.ndarray
    std: np.ndarray


def target_array(frame: pd.DataFrame) -> np.ndarray:
    m1 = frame.mass_1_detector.to_numpy(float)
    m2 = frame.mass_2_detector.to_numpy(float)
    high, low = np.maximum(m1, m2), np.minimum(m1, m2)
    mc = (high * low) ** (3.0 / 5.0) / (high + low) ** (1.0 / 5.0)
    q = np.clip(low / high, 1e-4, 1.0 - 1e-4)
    return np.column_stack([np.log(mc), np.log(q / (1.0 - q))]).astype(np.float32)


class TrainingPairs(Dataset):
    def __init__(self, root: Path, deployment: str, windows: tuple[int, ...], seed: int) -> None:
        self.root = root
        self.deployment = deployment
        self.windows = windows
        self.seed = int(seed)
        self.epoch = 0
        self.views = cache_development_views(root, deployment, windows)
        self.storage: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = {}
        self.row_metadata: dict[str, np.ndarray] = {}
        items = []
        targets = []
        metadata = []
        for family in ("sis", "pm"):
            data = root / "data/development" / deployment
            frame = pd.read_parquet(data / f"{family}_train_metadata.parquet")
            grouped = {int(k): part.index.to_numpy(dtype=np.int64) for k, part in frame.groupby("source_index")}
            a = {sec: np.load(self.views[sec][f"{family}_train_a"], mmap_mode="r") for sec in windows}
            b = {sec: np.load(self.views[sec][f"{family}_train_b"], mmap_mode="r") for sec in windows}
            self.storage[family] = (a, b)
            self.row_metadata[family] = np.column_stack([
                frame.chirp_mass_detector.to_numpy(np.float32),
                frame.duration_from_40hz_s.to_numpy(np.float32),
                0.5 * (
                    frame.target_snr_a.to_numpy(np.float32)
                    + frame.target_snr_b.to_numpy(np.float32)
                ),
            ]).astype(np.float32)
            family_targets = target_array(frame)
            for source_index, rows in grouped.items():
                first = int(rows[0])
                items.append((family, source_index, rows))
                targets.append(family_targets[first])
                metadata.append(
                    [
                        float(frame.iloc[first].chirp_mass_detector),
                        float(frame.iloc[first].duration_from_40hz_s),
                        float(0.5 * (frame.iloc[first].target_snr_a + frame.iloc[first].target_snr_b)),
                    ]
                )
        self.items = items
        self.raw_targets = np.asarray(targets, dtype=np.float32)
        self.meta = np.asarray(metadata, dtype=np.float32)
        self.target_stats = Targets(self.raw_targets.mean(axis=0), np.maximum(self.raw_targets.std(axis=0), 1e-6))
        self.targets = ((self.raw_targets - self.target_stats.mean) / self.target_stats.std).astype(np.float32)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        family, _, rows = self.items[index]
        row = int(rows[(self.epoch + stable_seed(self.seed, index)) % len(rows)])
        a_store, b_store = self.storage[family]
        a = np.stack([np.asarray(a_store[sec][row], dtype=np.float32) for sec in self.windows])
        b = np.stack([np.asarray(b_store[sec][row], dtype=np.float32) for sec in self.windows])
        return (
            torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(self.targets[index]),
            torch.from_numpy(self.row_metadata[family][row]),
        )


class ValidationEvents:
    def __init__(self, root: Path, deployment: str, windows: tuple[int, ...], target_stats: Targets) -> None:
        self.windows = windows
        views = cache_development_views(root, deployment, windows)
        event_views: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        partner: list[int] = []
        data = root / "data/development" / deployment
        for family in ("sis", "pm"):
            frame = pd.read_parquet(data / f"{family}_validation_metadata.parquet").reset_index(drop=True)
            first_start = len(records)
            for image in ("a", "b"):
                storage = {sec: np.load(views[sec][f"{family}_validation_{image}"], mmap_mode="r") for sec in windows}
                block_start = len(records)
                for index, row in frame.iterrows():
                    event_views.append(np.stack([np.asarray(storage[sec][index], dtype=np.float32) for sec in windows]))
                    records.append({
                        "idx": len(records), "family": family.upper(), "tag": image.upper(),
                        "source_index": int(row.source_index), "chirp_mass_detector": float(row.chirp_mass_detector),
                        "mass_1_detector": float(row.mass_1_detector), "mass_2_detector": float(row.mass_2_detector),
                        "duration_from_40hz_s": float(row.duration_from_40hz_s),
                        "target_snr": float(row[f"target_snr_{image}"]), "mass_stratum": int(row.mass_stratum),
                    })
                if image == "a":
                    partner.extend(range(block_start + len(frame), block_start + 2 * len(frame)))
                else:
                    partner.extend(range(first_start, first_start + len(frame)))
        uframe = pd.read_parquet(data / "unlensed_validation_metadata.parquet").reset_index(drop=True)
        ustorage = {sec: np.load(views[sec]["unlensed_validation"], mmap_mode="r") for sec in windows}
        for index, row in uframe.iterrows():
            event_views.append(np.stack([np.asarray(ustorage[sec][index], dtype=np.float32) for sec in windows]))
            records.append({
                "idx": len(records), "family": "U", "tag": "U", "source_index": int(row.source_index),
                "chirp_mass_detector": float(row.chirp_mass_detector), "mass_1_detector": float(row.mass_1_detector),
                "mass_2_detector": float(row.mass_2_detector), "duration_from_40hz_s": float(row.duration_from_40hz_s),
                "target_snr": float(row.target_snr), "mass_stratum": int(row.mass_stratum),
            })
            partner.append(-1)
        self.views = np.asarray(event_views, dtype=np.float32)
        self.frame = pd.DataFrame(records)
        self.partner = np.asarray(partner, dtype=np.int64)
        raw = target_array(self.frame)
        self.targets = ((raw - target_stats.mean) / target_stats.std).astype(np.float32)


class MultiScalePhysicsEncoder(nn.Module):
    def __init__(self, n_views: int, uncertainty_head: bool = False) -> None:
        super().__init__()
        self.n_views = int(n_views)
        self.uncertainty_head = bool(uncertainty_head)
        self.base = InceptionAttentionEncoder1D(
            in_channels=2, d_model=192, emb_dim=EMBEDDING_DIM, width_scale=1.0, depth=6
        )
        output_dim = 4 if uncertainty_head else 2
        self.parameter_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 96), nn.GELU(), nn.Dropout(0.05), nn.Linear(96, output_dim)
        )

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if views.ndim != 4 or views.shape[1] != self.n_views:
            raise ValueError(f"Expected [batch,{self.n_views},2,time], got {tuple(views.shape)}")
        encoded = [self.base(views[:, index]) for index in range(self.n_views)]
        embedding = F.normalize(torch.stack(encoded, dim=0).mean(dim=0), dim=-1)
        output = self.parameter_head(embedding)
        if self.uncertainty_head:
            mean, logvar = output[:, :2], torch.clamp(output[:, 2:], -5.0, 3.0)
            return embedding, mean, logvar
        return embedding, output, None


def regression_loss(mean: torch.Tensor, logvar: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if logvar is None:
        mc = F.smooth_l1_loss(mean[:, 0], target[:, 0], beta=0.5)
        q = F.smooth_l1_loss(mean[:, 1], target[:, 1], beta=0.5)
        return (mc + 0.5 * q) / 1.5
    error2 = (mean - target) ** 2
    nll = 0.5 * (torch.exp(-logvar) * error2 + logvar)
    return (nll[:, 0].mean() + 0.5 * nll[:, 1].mean()) / 1.5


def embedding_regularizer(embedding: torch.Tensor) -> torch.Tensor:
    x = embedding.float() - embedding.float().mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(0.08 - std).mean()
    cov = (x.T @ x) / max(len(x) - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    covariance = off.square().sum() / embedding.shape[1]
    return variance + covariance


def hard_negative_loss(embedding: torch.Tensor, metadata: torch.Tensor) -> tuple[torch.Tensor, int]:
    x = F.normalize(embedding.float(), dim=-1)
    similarity = x @ x.T
    logmc = torch.log(torch.clamp(metadata[:, 0], min=1e-6))
    logduration = torch.log(torch.clamp(metadata[:, 1], min=1e-6))
    logsnr = torch.log(torch.clamp(metadata[:, 2], min=1e-6))
    mask = (
        (torch.abs(logmc[:, None] - logmc[None, :]) > 0.20)
        & (torch.abs(logduration[:, None] - logduration[None, :]) < 0.45)
        & (torch.abs(logsnr[:, None] - logsnr[None, :]) < 0.30)
    )
    mask &= ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    if not bool(mask.any()):
        return similarity.sum() * 0.0, 0
    return F.relu(similarity[mask] - 0.25).mean(), int(mask.sum().item())


@torch.no_grad()
def encode_views(model: nn.Module, views: np.ndarray, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings, means, sigmas = [], [], []
    for start in range(0, len(views), batch_size):
        x = torch.from_numpy(np.asarray(views[start : start + batch_size], dtype=np.float32)).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z, mean, logvar = model(x)
        embeddings.append(z.float().cpu().numpy())
        means.append(mean.float().cpu().numpy())
        sigmas.append(
            np.exp(0.5 * logvar.float().cpu().numpy()) if logvar is not None else np.ones_like(mean.float().cpu().numpy())
        )
    return np.concatenate(embeddings), np.concatenate(means), np.concatenate(sigmas)


def pair_arrays(frame: pd.DataFrame, partner: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frame)
    ii, jj = np.triu_indices(n, 1)
    true = np.zeros(len(ii), dtype=bool)
    companion = {(min(i, int(p)), max(i, int(p))) for i, p in enumerate(partner) if p >= 0}
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ii, jj))}
    for pair in companion:
        true[lookup[pair]] = True
    return ii, jj, true


def rank_metrics(scores: np.ndarray, partner: np.ndarray, families: np.ndarray) -> dict[str, float]:
    ranks = []
    rank_family = []
    for query, target in enumerate(partner):
        if target < 0:
            continue
        row = scores[query].copy()
        row[query] = -np.inf
        rank = 1 + int(np.sum(row > row[target]))
        ranks.append(rank)
        rank_family.append(str(families[query]))
    ranks_array = np.asarray(ranks)
    out = {
        "r_at_1": float(np.mean(ranks_array <= 1)),
        "r_at_5": float(np.mean(ranks_array <= 5)),
        "r_at_10": float(np.mean(ranks_array <= 10)),
        "median_rank": float(np.median(ranks_array)),
        "n_queries": int(len(ranks_array)),
    }
    for family in ("SIS", "PM"):
        values = ranks_array[np.asarray(rank_family) == family]
        out[f"{family.lower()}_r_at_1"] = float(np.mean(values <= 1))
        out[f"{family.lower()}_r_at_10"] = float(np.mean(values <= 10))
    return out


def false_at_recall(labels: np.ndarray, scores: np.ndarray, recall: float) -> int:
    order = np.argsort(scores)[::-1]
    y = labels[order]
    target = int(math.ceil(recall * int(labels.sum())))
    position = int(np.flatnonzero(np.cumsum(y) >= target)[0])
    return int((~y[: position + 1]).sum())


def evaluate_validation(
    model: nn.Module,
    validation: ValidationEvents,
    target_stats: Targets,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    z, mean_std, sigma_std = encode_views(model, validation.views)
    mean_raw = mean_std * target_stats.std + target_stats.mean
    sigma_raw = sigma_std * target_stats.std
    cosine = np.clip(z @ z.T, -1.0, 1.0)
    mc_distance = np.abs(mean_raw[:, None, 0] - mean_raw[None, :, 0]) / np.maximum(
        np.sqrt(sigma_raw[:, None, 0] ** 2 + sigma_raw[None, :, 0] ** 2), 0.15
    )
    q_distance = np.abs(mean_raw[:, None, 1] - mean_raw[None, :, 1]) / np.maximum(
        np.sqrt(sigma_raw[:, None, 1] ** 2 + sigma_raw[None, :, 1] ** 2), 0.25
    )
    ii, jj, labels = pair_arrays(validation.frame, validation.partner)

    def standardized(matrix: np.ndarray) -> np.ndarray:
        values = matrix[ii, jj]
        return (matrix - np.median(values)) / max(float(np.std(values)), 1e-6)

    composite = standardized(cosine) + 0.5 * standardized(-mc_distance) + 0.25 * standardized(-q_distance)
    np.fill_diagonal(cosine, -np.inf)
    np.fill_diagonal(composite, -np.inf)
    raw_target = target_array(validation.frame)
    labelled = validation.partner >= 0
    prediction_error = mean_raw - raw_target
    pair_scores = composite[ii, jj]
    pair_cosine = cosine[ii, jj]
    metrics: dict[str, Any] = {
        **{f"embedding_{k}": v for k, v in rank_metrics(cosine, validation.partner, validation.frame.family.to_numpy()).items()},
        **{f"composite_{k}": v for k, v in rank_metrics(composite, validation.partner, validation.frame.family.to_numpy()).items()},
        "embedding_pair_auprc": float(average_precision_score(labels, pair_cosine)),
        "composite_pair_auprc": float(average_precision_score(labels, pair_scores)),
        "embedding_roc_auc": float(roc_auc_score(labels, pair_cosine)),
        "composite_roc_auc": float(roc_auc_score(labels, pair_scores)),
        "composite_false_at_recall_0p5": false_at_recall(labels, pair_scores, 0.5),
        "composite_false_at_recall_0p9": false_at_recall(labels, pair_scores, 0.9),
        "embedding_effective_rank": float(effective_rank(z)),
        "logmc_mae": float(np.mean(np.abs(prediction_error[labelled, 0]))),
        "logitq_mae": float(np.mean(np.abs(prediction_error[labelled, 1]))),
        "logmc_bias": float(np.mean(prediction_error[labelled, 0])),
        "logmc_50_coverage": float(np.mean(np.abs(prediction_error[labelled, 0]) <= 0.67449 * sigma_raw[labelled, 0])),
        "logmc_90_coverage": float(np.mean(np.abs(prediction_error[labelled, 0]) <= 1.64485 * sigma_raw[labelled, 0])),
    }
    false_indices = np.flatnonzero(~labels)
    top_false = false_indices[np.argsort(pair_scores[false_indices])[::-1][:100]]
    true_logmc = raw_target[:, 0]
    catastrophic = np.abs(true_logmc[ii[top_false]] - true_logmc[jj[top_false]]) > 0.35
    metrics["top100_false_catastrophic_mass_count"] = int(catastrophic.sum())
    metrics["top100_false_catastrophic_mass_fraction"] = float(catastrophic.mean())
    for mass_bin in sorted(validation.frame.mass_stratum.unique()):
        keep_queries = (validation.partner >= 0) & validation.frame.mass_stratum.eq(mass_bin).to_numpy()
        ranks = []
        for query in np.flatnonzero(keep_queries):
            target = validation.partner[query]
            row = composite[query].copy(); row[query] = -np.inf
            ranks.append(1 + int(np.sum(row > row[target])))
        metrics[f"mass_bin_{mass_bin}_r_at_10"] = float(np.mean(np.asarray(ranks) <= 10))
    event_output = validation.frame.copy()
    event_output["pred_log_chirp_mass"] = mean_raw[:, 0]
    event_output["pred_logit_q"] = mean_raw[:, 1]
    event_output["pred_log_chirp_mass_sigma"] = sigma_raw[:, 0]
    event_output["pred_logit_q_sigma"] = sigma_raw[:, 1]
    arrays = {
        "embedding": z, "pred_mean": mean_raw, "pred_sigma": sigma_raw,
        "pair_i": ii, "pair_j": jj, "pair_label": labels,
        "pair_cosine": pair_cosine, "pair_composite": pair_scores,
    }
    return metrics, event_output, arrays


def train_one(
    root: Path,
    deployment: str,
    windows: tuple[int, ...],
    seed: int,
    uncertainty_head: bool,
    embedding_reg: bool,
    hard_negatives: bool,
    epochs: int,
    output: Path,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    dataset = TrainingPairs(root, deployment, windows, seed)
    validation = ValidationEvents(root, deployment, windows, dataset.target_stats)
    seed_everything(seed)
    model = MultiScalePhysicsEncoder(len(windows), uncertainty_head=uncertainty_head).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    contrastive_loss = NTXentLoss(0.10)
    best_key = None
    best_state = None
    best_metrics = None
    history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(stable_seed(seed, epoch, "loader"))
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, generator=generator)
        model.train()
        started = time.perf_counter()
        losses = []
        hard_count = 0
        for a, b, target, metadata in loader:
            a = a.cuda(non_blocking=True); b = b.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True); metadata = metadata.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                za, mean_a, logvar_a = model(a)
                zb, mean_b, logvar_b = model(b)
                contrastive = contrastive_loss(za, zb)
                auxiliary = 0.5 * (
                    regression_loss(mean_a, logvar_a, target) + regression_loss(mean_b, logvar_b, target)
                )
                hard_loss, count = hard_negative_loss(F.normalize(za + zb, dim=-1), metadata)
                embed_loss = embedding_regularizer(torch.cat([za, zb]))
                loss = contrastive + 0.5 * auxiliary
                if hard_negatives:
                    loss = loss + 0.20 * hard_loss
                if embedding_reg:
                    loss = loss + 0.10 * embed_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            hard_count += count
            losses.append([float(loss.detach()), float(contrastive.detach()), float(auxiliary.detach()), float(hard_loss.detach()), float(embed_loss.detach())])
        row: dict[str, Any] = {
            "epoch": epoch, "loss": float(np.mean(np.asarray(losses)[:, 0])),
            "contrastive_loss": float(np.mean(np.asarray(losses)[:, 1])),
            "auxiliary_loss": float(np.mean(np.asarray(losses)[:, 2])),
            "hard_negative_loss": float(np.mean(np.asarray(losses)[:, 3])),
            "embedding_regularizer": float(np.mean(np.asarray(losses)[:, 4])),
            "hard_negative_pairs_seen": hard_count, "epoch_seconds": time.perf_counter() - started,
        }
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            metrics, _, _ = evaluate_validation(model, validation, dataset.target_stats)
            row.update({f"val_{key}": value for key, value in metrics.items()})
            key = (
                metrics["composite_r_at_10"], metrics["composite_pair_auprc"],
                metrics["composite_r_at_1"], -metrics["top100_false_catastrophic_mass_fraction"],
                metrics["embedding_effective_rank"], -metrics["logmc_mae"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = metrics
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        print(json.dumps({"deployment": deployment, "seed": seed, "windows": windows, **row}), flush=True)
    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state)
    final_metrics, event_output, arrays = evaluate_validation(model, validation, dataset.target_stats)
    torch.save(
        {
            "model_state": best_state, "windows_seconds": windows, "seed": seed,
            "uncertainty_head": uncertainty_head, "embedding_regularization": embedding_reg,
            "hard_negatives": hard_negatives, "target_mean": dataset.target_stats.mean,
            "target_std": dataset.target_stats.std, "embedding_dim": EMBEDDING_DIM,
        },
        output / "validation_selected_model.pt",
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    event_output.to_parquet(output / "validation_event_predictions.parquet", index=False)
    np.savez_compressed(output / "validation_arrays.npz", **arrays)
    summary = {
        "status": "validation_selected", "deployment": deployment, "seed": seed,
        "windows_seconds": windows, "long_window_resampling": "scipy.signal.resample_poly with Kaiser beta=8.6 to 4096 points",
        "hard_negatives": hard_negatives, "uncertainty_head": uncertainty_head,
        "embedding_regularization": embedding_reg, "epochs": epochs,
        "training_systems": len(dataset), "validation_events": len(validation.frame),
        "target_mean": dataset.target_stats.mean.tolist(), "target_std": dataset.target_stats.std.tolist(),
        "best_validation": final_metrics,
    }
    write_json(summary_path, summary)
    return summary


def model_from_checkpoint(path: Path) -> tuple[MultiScalePhysicsEncoder, dict[str, Any], Targets]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MultiScalePhysicsEncoder(len(payload["windows_seconds"]), payload["uncertainty_head"])
    model.load_state_dict(payload["model_state"])
    model.cuda().eval()
    stats_value = Targets(np.asarray(payload["target_mean"], dtype=np.float32), np.asarray(payload["target_std"], dtype=np.float32))
    return model, payload, stats_value


def encode_full24(
    checkpoint: Path,
    full24: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, payload, target_stats = model_from_checkpoint(checkpoint)
    windows = tuple(int(x) for x in payload["windows_seconds"])
    outputs_z, outputs_mean, outputs_sigma = [], [], []
    for start in range(0, len(full24), batch_size):
        block = np.asarray(full24[start : start + batch_size], dtype=np.float32)
        views = np.stack([make_window_view(block, seconds) for seconds in windows], axis=1)
        z, mean_std, sigma_std = encode_views(model, views, batch_size=batch_size)
        outputs_z.append(z)
        outputs_mean.append(mean_std * target_stats.std + target_stats.mean)
        outputs_sigma.append(sigma_std * target_stats.std)
    return np.concatenate(outputs_z), np.concatenate(outputs_mean), np.concatenate(outputs_sigma)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deployment", choices=("gwtc3", "gwtc4"), required=True)
    parser.add_argument("--windows", default="2")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hard-negatives", action="store_true")
    parser.add_argument("--uncertainty-head", action="store_true")
    parser.add_argument("--embedding-regularization", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = tuple(int(x) for x in args.windows.split(","))
    if windows not in ((2,), (2, 8), (2, 16)):
        raise ValueError(f"Unsupported windows: {windows}")
    summary = train_one(
        args.root, args.deployment, windows, args.seed, args.uncertainty_head,
        args.embedding_regularization, args.hard_negatives, args.epochs, args.output,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
