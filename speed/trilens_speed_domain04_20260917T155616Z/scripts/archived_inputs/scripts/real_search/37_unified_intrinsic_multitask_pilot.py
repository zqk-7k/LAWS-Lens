#!/usr/bin/env python3
"""Validation-only pilot for a unified, physics-regularized waveform encoder.

This script is deliberately separated from the formal v5 run.  It combines
the two GW-LMC lens-environment compatibility slots in one encoder and adds
auxiliary supervision for detector-frame chirp mass and mass ratio.  The
labels are injection truth; real-event PE samples are never read here.

Model and hyperparameter selection is performed on validation systems only.
The held-out test split is not instantiated or evaluated by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import load_match_arrays, peak_flip_channels, split_indices, zscore_channels
from matchgw.matching import similarity_matrix
from matchgw.models import InceptionAttentionEncoder1D, InceptionTimeEncoder1D, NTXentLoss
from scripts.experiments.mainline_uncertainty_common import seed_everything
from scripts.real_search.physical_common import effective_rank, write_json


FAMILIES = ("SIS", "PM")
INTRINSIC_FUSION_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
INPUT_SAMPLES = int(os.environ.get("GW_WAVEFORM_INPUT_SAMPLES", "0"))


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare(x: np.ndarray, rng: np.random.Generator | None, train: bool) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if INPUT_SAMPLES > 0:
        if values.shape[-1] < INPUT_SAMPLES:
            raise ValueError(
                f"Waveform has {values.shape[-1]} samples; fixed crop requires {INPUT_SAMPLES}"
            )
        values = values[..., -INPUT_SAMPLES:]
    y = peak_flip_channels(values.copy())
    if train:
        assert rng is not None
        y = np.roll(y, int(rng.integers(-128, 129)), axis=-1)
        y *= float(1.0 + rng.uniform(-0.1, 0.1))
        # The input is already PSD-whitened.  This small perturbation is an
        # augmentation, not a substitute for detector noise injection.
        channel_scale = y.std(axis=-1, keepdims=True)
        y += rng.normal(0.0, 0.003 * np.maximum(channel_scale, 1e-6), size=y.shape)
    return zscore_channels(y)


def intrinsic_target(frame: pd.DataFrame) -> np.ndarray:
    m1 = frame["mass_1_detector"].to_numpy(np.float64)
    m2 = frame["mass_2_detector"].to_numpy(np.float64)
    high = np.maximum(m1, m2)
    low = np.minimum(m1, m2)
    chirp = (high * low) ** (3.0 / 5.0) / (high + low) ** (1.0 / 5.0)
    q = np.clip(low / high, 1e-4, 1.0 - 1e-4)
    return np.column_stack([np.log(chirp), np.log(q / (1.0 - q))]).astype(np.float32)


@dataclass
class FamilyData:
    arrays: object
    train_ids: np.ndarray
    val_ids: np.ndarray
    targets: np.ndarray
    multinoise_root: Path


class UnifiedCleanPairs(Dataset):
    def __init__(self, data: dict[str, FamilyData], seed: int) -> None:
        self.data = data
        self.items = [(family, int(idx)) for family in FAMILIES for idx in data[family].train_ids]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        family, idx = self.items[item]
        current = self.data[family]
        return (
            torch.from_numpy(prepare(current.arrays.l1_pure[idx], self.rng, True)),
            torch.from_numpy(prepare(current.arrays.l2_pure[idx], self.rng, True)),
            torch.from_numpy(current.targets[idx]),
        )


class UnifiedMultiNoise(Dataset):
    def __init__(self, data: dict[str, FamilyData], seed: int) -> None:
        self.data = data
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.storage = {}
        self.items: list[tuple[str, int]] = []
        for family in FAMILIES:
            root = data[family].multinoise_root
            a = np.load(root / "noisy_image_a.npy", mmap_mode="r")
            b = np.load(root / "noisy_image_b.npy", mmap_mode="r")
            source = np.load(root / "source_index.npy")
            variant = np.load(root / "variant.npy")
            rows = {
                int(idx): np.flatnonzero(source == idx)[np.argsort(variant[source == idx])]
                for idx in np.unique(source)
            }
            self.storage[family] = (a, b, rows)
            self.items.extend((family, int(idx)) for idx in data[family].train_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        family, idx = self.items[item]
        a, b, rows = self.storage[family]
        choices = rows[idx]
        row = int(choices[(self.epoch + item) % len(choices)])
        clean = self.data[family].arrays
        return (
            torch.from_numpy(prepare(a[row], self.rng, True)),
            torch.from_numpy(prepare(b[row], self.rng, True)),
            torch.from_numpy(prepare(clean.l1_pure[idx], self.rng, True)),
            torch.from_numpy(prepare(clean.l2_pure[idx], self.rng, True)),
            torch.from_numpy(self.data[family].targets[idx]),
        )


class UnifiedValidation(Dataset):
    def __init__(self, data: dict[str, FamilyData], pure: bool = False) -> None:
        self.rows: list[tuple[str, int, int]] = []
        self.partner: list[int] = []
        self.group: list[str] = []
        self.targets: list[np.ndarray] = []
        for family in FAMILIES:
            first = len(self.rows)
            ids = data[family].val_ids
            for idx in ids:
                self.rows.append((family, int(idx), 1))
                self.group.append(family)
                self.targets.append(data[family].targets[int(idx)])
            second = len(self.rows)
            for idx in ids:
                self.rows.append((family, int(idx), 2))
                self.group.append(family)
                self.targets.append(data[family].targets[int(idx)])
            n = len(ids)
            self.partner.extend(list(range(second, second + n)))
            self.partner.extend(list(range(first, first + n)))
        # Add the unlensed validation controls exactly once.  They contribute
        # realistic distractors but have no retrieval partner.
        reference = data[FAMILIES[0]]
        for idx in reference.val_ids:
            self.rows.append(("U", int(idx), 0))
            self.group.append("unlensed")
            self.partner.append(-1)
            self.targets.append(np.full(2, np.nan, dtype=np.float32))
        self.data = data
        self.pure = bool(pure)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        family, idx, image = self.rows[item]
        if family == "U":
            arrays = self.data[FAMILIES[0]].arrays
            x = arrays.unlensed_pure[idx] if self.pure else arrays.unlensed[idx]
        else:
            arrays = self.data[family].arrays
            if self.pure:
                x = arrays.l1_pure[idx] if image == 1 else arrays.l2_pure[idx]
            else:
                x = arrays.l1[idx] if image == 1 else arrays.l2[idx]
        return torch.from_numpy(prepare(x, None, False))


class PhysicsRegularizedEncoder(nn.Module):
    def __init__(self, base: nn.Module, target_dim: int = 2) -> None:
        super().__init__()
        self.base = base
        # The reused base architecture returns a 96-dimensional normalized
        # embedding.  The auxiliary head regularizes, but does not replace,
        # the retrieval embedding.
        self.parameter_head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, target_dim))

    def forward(self, x: torch.Tensor, return_parameters: bool = False):
        embedding = self.base(x)
        if return_parameters:
            return embedding, self.parameter_head(embedding)
        return embedding


def build_base_encoder(name: str, specmod) -> nn.Module:
    if name == "trigger_spectrogram":
        return specmod.TriggerAlignedDualScaleSpectrogramEncoder(base_channels=24)
    if name == "inceptiontime":
        return InceptionTimeEncoder1D(
            in_channels=2, d_model=192, emb_dim=96, width_scale=1.0, depth=6
        )
    if name == "inception_attention":
        return InceptionAttentionEncoder1D(
            in_channels=2, d_model=192, emb_dim=96, width_scale=1.0, depth=6
        )
    raise ValueError(f"Unknown backbone: {name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: UnifiedValidation,
    batch_size: int,
    *,
    allow_q_feature: bool = True,
) -> dict[str, float]:
    model.eval()
    embeddings = []
    predictions = []
    for x in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embedding, prediction = model(x.cuda(non_blocking=True), return_parameters=True)
            embeddings.append(embedding.float().cpu().numpy())
            predictions.append(prediction.float().cpu().numpy())
    z = np.concatenate(embeddings)
    predicted = np.concatenate(predictions)
    score = similarity_matrix(z)

    # Lensing preserves detector-frame intrinsic parameters. The auxiliary
    # head is therefore audited as a waveform-derived compatibility feature;
    # no real-event PE sample enters this calculation. For this validation
    # pilot only, choose the relative coefficient on a fixed grid using the
    # same validation catalog. Formal test evaluation will freeze this choice.
    intrinsic_mc = -np.abs(predicted[:, None, 0] - predicted[None, :, 0])
    intrinsic_q = -np.abs(predicted[:, None, 1] - predicted[None, :, 1])
    intrinsic = -np.sqrt(np.sum((predicted[:, None, :] - predicted[None, :, :]) ** 2, axis=-1))
    for values in (intrinsic_mc, intrinsic_q, intrinsic):
        np.fill_diagonal(values, -np.inf)

    def finite_standardize(values: np.ndarray) -> np.ndarray:
        upper = values[np.triu_indices(len(values), 1)]
        center = float(np.median(upper))
        scale = max(float(np.std(upper)), 1e-8)
        result = (values - center) / scale
        np.fill_diagonal(result, -np.inf)
        return result

    def ranks_for(values: np.ndarray) -> np.ndarray:
        result = np.full(len(dataset), np.nan)
        for query, partner in enumerate(dataset.partner):
            if partner >= 0:
                result[query] = 1 + np.sum(values[query] > values[query, partner])
        return result

    ranks = ranks_for(score)
    intrinsic_ranks = ranks_for(intrinsic)
    intrinsic_mc_ranks = ranks_for(intrinsic_mc)
    intrinsic_q_ranks = ranks_for(intrinsic_q)
    z_score = finite_standardize(score)
    z_mc = finite_standardize(intrinsic_mc)
    z_q = finite_standardize(intrinsic_q)
    best_hybrid = None
    q_grid = INTRINSIC_FUSION_GRID if allow_q_feature else (0.0,)
    for coefficient_mc, coefficient_q in itertools.product(INTRINSIC_FUSION_GRID, q_grid):
        # Add only active terms. Multiplying a zero coefficient by the
        # deliberately -inf self-match diagonal produces NaNs even though the
        # diagonal is excluded from every rank calculation.
        hybrid_score = z_score.copy()
        if coefficient_mc > 0:
            hybrid_score += float(coefficient_mc) * z_mc
        if coefficient_q > 0:
            hybrid_score += float(coefficient_q) * z_q
        hybrid_ranks = ranks_for(hybrid_score)
        keep = np.isfinite(hybrid_ranks)
        key = (
            float(np.mean(hybrid_ranks[keep] <= 10)),
            float(np.mean(hybrid_ranks[keep] <= 1)),
            -float(np.median(hybrid_ranks[keep])),
            -float(coefficient_mc + coefficient_q),
        )
        if best_hybrid is None or key > best_hybrid[0]:
            best_hybrid = (key, float(coefficient_mc), float(coefficient_q), hybrid_ranks)
    assert best_hybrid is not None
    hybrid_coefficient_mc = best_hybrid[1]
    hybrid_coefficient_q = best_hybrid[2]
    hybrid_ranks = best_hybrid[3]
    valid = np.isfinite(ranks)
    out: dict[str, float] = {
        "overall_r_at_1": float(np.mean(ranks[valid] <= 1)),
        "overall_r_at_10": float(np.mean(ranks[valid] <= 10)),
        "overall_median_rank": float(np.median(ranks[valid])),
        "embedding_effective_rank": effective_rank(z),
        "intrinsic_only_r_at_1": float(np.mean(intrinsic_ranks[valid] <= 1)),
        "intrinsic_only_r_at_10": float(np.mean(intrinsic_ranks[valid] <= 10)),
        "intrinsic_only_median_rank": float(np.median(intrinsic_ranks[valid])),
        "intrinsic_mc_only_r_at_1": float(np.mean(intrinsic_mc_ranks[valid] <= 1)),
        "intrinsic_mc_only_r_at_10": float(np.mean(intrinsic_mc_ranks[valid] <= 10)),
        "intrinsic_q_only_r_at_1": float(np.mean(intrinsic_q_ranks[valid] <= 1)),
        "intrinsic_q_only_r_at_10": float(np.mean(intrinsic_q_ranks[valid] <= 10)),
        "hybrid_r_at_1": float(np.mean(hybrid_ranks[valid] <= 1)),
        "hybrid_r_at_10": float(np.mean(hybrid_ranks[valid] <= 10)),
        "hybrid_median_rank": float(np.median(hybrid_ranks[valid])),
        "hybrid_intrinsic_mc_coefficient": hybrid_coefficient_mc,
        "hybrid_intrinsic_q_coefficient": hybrid_coefficient_q,
    }
    target = np.asarray(dataset.targets, dtype=np.float64)
    labelled = np.all(np.isfinite(target), axis=1)
    out["intrinsic_standardized_mae"] = float(np.mean(np.abs(predicted[labelled] - target[labelled])))
    for parameter, column in (("log_chirp_mass", 0), ("logit_q", 1)):
        out[f"{parameter}_prediction_correlation"] = float(
            np.corrcoef(predicted[labelled, column], target[labelled, column])[0, 1]
        )
    for family in FAMILIES:
        keep = np.asarray(dataset.group) == family
        out[f"{family.lower()}_r_at_1"] = float(np.mean(ranks[keep] <= 1))
        out[f"{family.lower()}_r_at_10"] = float(np.mean(ranks[keep] <= 10))
        out[f"{family.lower()}_median_rank"] = float(np.median(ranks[keep]))
    upper = score[np.triu_indices(len(score), 1)]
    out["cosine_median"] = float(np.median(upper))
    out["cosine_std"] = float(np.std(upper))
    return out


def standardized_targets(data: dict[str, FamilyData]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([data[family].targets[data[family].train_ids] for family in FAMILIES])
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-6)
    for family in FAMILIES:
        data[family].targets = ((data[family].targets - mean) / std).astype(np.float32)
    return mean, std


def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, beta=0.5)


def weighted_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    q_loss_weight: float,
) -> torch.Tensor:
    mc = F.smooth_l1_loss(prediction[:, 0], target[:, 0], beta=0.5)
    q = F.smooth_l1_loss(prediction[:, 1], target[:, 1], beta=0.5)
    return (mc + float(q_loss_weight) * q) / (1.0 + float(q_loss_weight))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202607221)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--pretrain-epochs", type=int, default=16)
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--q-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--backbone",
        choices=("trigger_spectrogram", "inceptiontime", "inception_attention"),
        default="trigger_spectrogram",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    v3 = module_from(REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py", "v3_for_unified")
    specmod = module_from(REPO / "scripts/experiments/26_spectrogram_encoder_pilot.py", "spec_for_unified")
    data: dict[str, FamilyData] = {}
    for offset, family in enumerate(FAMILIES):
        cfg = v3.training_config(args.seed_root, family, args.seed, args.samples, args.adapt_epochs, args.batch_size)
        arrays = load_match_arrays(cfg)
        parts = split_indices(args.samples, args.samples, cfg)
        metadata = pd.read_parquet(args.source_bank / f"{family}_data_0222" / "physical_source_pair_metadata.parquet")
        targets = intrinsic_target(metadata)
        data[family] = FamilyData(
            arrays=arrays,
            train_ids=parts["lensed"]["train"],
            val_ids=parts["lensed"]["val"],
            targets=targets,
            multinoise_root=args.seed_root / "data/real_noise_injections" / f"multinoise_{family.lower()}_train_v8",
        )
    target_mean, target_std = standardized_targets(data)
    clean = UnifiedCleanPairs(data, args.seed + 7000)
    multinoise = UnifiedMultiNoise(data, args.seed + 7001)
    clean_validation = UnifiedValidation(data, pure=True)
    noisy_validation = UnifiedValidation(data, pure=False)

    seed_everything(args.seed + 7000)
    base = build_base_encoder(args.backbone, specmod)
    model = PhysicsRegularizedEncoder(base).cuda()
    ntxent = NTXentLoss(0.10)
    history: list[dict] = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    best_clean_state = None
    best_clean_key = None
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        loader = DataLoader(clean, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
        for a, b, target in loader:
            a = a.cuda(non_blocking=True)
            b = b.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                za, pa = model(a, return_parameters=True)
                zb, pb = model(b, return_parameters=True)
                contrastive = ntxent(za, zb)
                auxiliary = 0.5 * (
                    weighted_regression_loss(pa, target, args.q_loss_weight)
                    + weighted_regression_loss(pb, target, args.q_loss_weight)
                )
                loss = contrastive + args.aux_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append((float(loss.detach()), float(contrastive.detach()), float(auxiliary.detach())))
        row = {
            "phase": "clean_pretrain",
            "epoch": epoch,
            "loss": float(np.mean(np.asarray(losses)[:, 0])),
            "contrastive_loss": float(np.mean(np.asarray(losses)[:, 1])),
            "intrinsic_loss": float(np.mean(np.asarray(losses)[:, 2])),
            "epoch_s": float(time.perf_counter() - started),
        }
        if epoch == 1 or epoch % 2 == 0 or epoch == args.pretrain_epochs:
            metrics = evaluate(
                model,
                clean_validation,
                args.batch_size,
                allow_q_feature=args.q_loss_weight > 0,
            )
            row.update({f"clean_val_{key}": value for key, value in metrics.items()})
            key = (metrics["overall_r_at_10"], metrics["overall_r_at_1"], -metrics["overall_median_rank"], metrics["embedding_effective_rank"])
            if best_clean_key is None or key > best_clean_key:
                best_clean_key = key
                best_clean_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row), flush=True)

    assert best_clean_state is not None
    model.load_state_dict(best_clean_state)
    teacher = PhysicsRegularizedEncoder(build_base_encoder(args.backbone, specmod)).cuda()
    teacher.load_state_dict(best_clean_state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    best_state = None
    best_metrics = None
    best_key = None
    for epoch in range(1, args.adapt_epochs + 1):
        multinoise.set_epoch(epoch)
        model.train()
        started = time.perf_counter()
        losses = []
        loader = DataLoader(multinoise, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
        for a, b, clean_a, clean_b, target in loader:
            a = a.cuda(non_blocking=True)
            b = b.cuda(non_blocking=True)
            clean_a = clean_a.cuda(non_blocking=True)
            clean_b = clean_b.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                za, pa = model(a, return_parameters=True)
                zb, pb = model(b, return_parameters=True)
                contrastive = ntxent(za, zb)
                auxiliary = 0.5 * (
                    weighted_regression_loss(pa, target, args.q_loss_weight)
                    + weighted_regression_loss(pb, target, args.q_loss_weight)
                )
                with torch.no_grad():
                    # Use each clean source as a stable identity prototype.
                    clean_za = teacher(clean_a)
                    clean_zb = teacher(clean_b)
                    prototype = F.normalize(clean_za + clean_zb, dim=-1)
                identity = torch.arange(len(prototype), device=prototype.device)
                prototype_loss = 0.5 * (
                    F.cross_entropy(za @ prototype.T / 0.10, identity)
                    + F.cross_entropy(zb @ prototype.T / 0.10, identity)
                )
                loss = 0.5 * contrastive + 0.5 * prototype_loss + args.aux_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append((float(loss.detach()), float(contrastive.detach()), float(prototype_loss.detach()), float(auxiliary.detach())))
        values = np.asarray(losses)
        row = {
            "phase": "real_noise_adaptation",
            "epoch": epoch,
            "loss": float(values[:, 0].mean()),
            "contrastive_loss": float(values[:, 1].mean()),
            "prototype_loss": float(values[:, 2].mean()),
            "intrinsic_loss": float(values[:, 3].mean()),
            "epoch_s": float(time.perf_counter() - started),
        }
        if epoch == 1 or epoch % 2 == 0 or epoch == args.adapt_epochs:
            metrics = evaluate(
                model,
                noisy_validation,
                args.batch_size,
                allow_q_feature=args.q_loss_weight > 0,
            )
            row.update({f"noisy_val_{key}": value for key, value in metrics.items()})
            key = (
                metrics["hybrid_r_at_10"],
                metrics["hybrid_r_at_1"],
                metrics["overall_r_at_10"],
                metrics["overall_r_at_1"],
                -metrics["hybrid_median_rank"],
                metrics["embedding_effective_rank"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = metrics
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "validation_selected_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "target_mean": target_mean,
            "target_std": target_std,
            "aux_weight": args.aux_weight,
            "q_loss_weight": args.q_loss_weight,
            "architecture": "unified waveform encoder with injection-truth intrinsic auxiliary head",
            "backbone": args.backbone,
            "waveform_input_samples": INPUT_SAMPLES or None,
            "waveform_input_seconds_at_2048_hz": (INPUT_SAMPLES / 2048.0) if INPUT_SAMPLES else 24.0,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    summary = {
        "status": "validation_only_pilot",
        "seed": args.seed,
        "families_combined": list(FAMILIES),
        "physical_group_interpretation": {
            "SIS": "GW-LMC smooth/non-subhalo compatibility slot",
            "PM": "GW-LMC subhalo-present compatibility slot",
        },
        "n_training_systems": len(clean),
        "n_validation_lensed_systems": int(sum(len(data[family].val_ids) for family in FAMILIES)),
        "heldout_test_instantiated": False,
        "real_catalog_or_pe_used": False,
        "target_parameters": ["log_detector_frame_chirp_mass", "logit_mass_ratio"],
        "aux_weight": args.aux_weight,
        "q_loss_weight": args.q_loss_weight,
        "backbone": args.backbone,
        "waveform_input_samples": INPUT_SAMPLES or None,
        "waveform_input_seconds_at_2048_hz": (INPUT_SAMPLES / 2048.0) if INPUT_SAMPLES else 24.0,
        "best_noisy_validation": best_metrics,
        "checkpoint": str(checkpoint),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
