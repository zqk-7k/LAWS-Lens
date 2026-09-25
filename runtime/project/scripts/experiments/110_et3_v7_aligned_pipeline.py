#!/usr/bin/env python3
"""ET-3 rerun aligned with the frozen GWTC v7 evidence framework.

The detector domains remain different (ET1/ET2/ET3 versus H1/L1), but the
methodological contract is shared:

* two-second, 2048 Hz whitened detector-strain input;
* one unified lens-family encoder with waveform-derived intrinsic heads;
* waveform evidence calibrated only on validation systems;
* a population time-delay likelihood ratio;
* an absolute common-source sky Bayes factor;
* validation-only fusion weights; and
* held-out test retrieval and unordered-pair diagnostics.

The script has two stages. ``development`` compares the two predeclared ET
downsampling variants on one validation-only seed. ``formal`` freezes that
choice and evaluates five new training/split seeds.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bilby.gw.conversion import luminosity_distance_to_redshift
from scipy import stats
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import peak_flip_channels, zscore_channels  # noqa: E402
from matchgw.matching import similarity_matrix  # noqa: E402
from matchgw.models import InceptionAttentionEncoder1D, NTXentLoss  # noqa: E402
from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    across_seed_summary,
    bootstrap_system_ci,
    full_unordered_pair_metrics,
    metric_rows,
    seed_everything,
    split_integrity_audit,
    write_json,
)
from scripts.real_search.physical_common import (  # noqa: E402
    apply_score_likelihood_ratio,
    apply_time_likelihood_ratio,
    effective_rank,
    fit_score_likelihood_ratio,
    fit_time_likelihood_ratio,
)
from matchgw.aux_priors.observed_sky import (  # noqa: E402
    build_observed_sky_table,
    scenario_for_detector,
)


RAW_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
PREPROCESSED_ROOT = REPO / "data_generation/et3_v7_preprocessed_20260723"
DEFAULT_OUT = REPO / "results/et3_v7_aligned_20260723"
DEFAULT_DEVELOPMENT_SEED = 202607230
DEFAULT_FORMAL_SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
FAMILIES = ("SIS", "PM")
INPUT_SAMPLES = 4096
MODEL_SAMPLE_RATE = 2048
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
INTRINSIC_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
ET_BASE_START_GPS = 1126259462.39
ET_BASE_END_GPS = 1449422343.39
SECONDS_PER_DAY = 86400.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FamilyBank:
    noisy1: np.ndarray
    noisy2: np.ndarray
    pure1: np.ndarray
    pure2: np.ndarray
    snr1: np.ndarray
    snr2: np.ndarray
    source: pd.DataFrame
    timing: pd.DataFrame
    targets: np.ndarray


@dataclass
class ETBank:
    families: dict[str, FamilyBank]
    unlensed_noisy: np.ndarray
    unlensed_pure: np.ndarray
    unlensed_snr: np.ndarray
    unlensed_source: pd.DataFrame
    unlensed_timing: pd.DataFrame


def detector_frame_targets(source: pd.DataFrame) -> np.ndarray:
    z = np.asarray(
        luminosity_distance_to_redshift(source["luminosity_distance"].to_numpy(np.float64)),
        dtype=np.float64,
    )
    m1 = source["mass_1_source"].to_numpy(np.float64) * (1.0 + z)
    m2 = source["mass_2_source"].to_numpy(np.float64) * (1.0 + z)
    high = np.maximum(m1, m2)
    low = np.minimum(m1, m2)
    chirp = (high * low) ** (3.0 / 5.0) / (high + low) ** (1.0 / 5.0)
    q = np.clip(low / high, 1e-4, 1.0 - 1e-4)
    return np.column_stack([np.log(chirp), np.log(q / (1.0 - q))]).astype(np.float32)


def load_bank(preprocessing: str) -> ETBank:
    root = PREPROCESSED_ROOT / preprocessing
    families: dict[str, FamilyBank] = {}
    for family in FAMILIES:
        path = root / f"{family}_data_0222"
        source = pd.read_csv(RAW_ROOT / f"{family}_data_0222/source_samples.csv")
        families[family] = FamilyBank(
            noisy1=np.load(path / f"{family}_data_strain_1.npy", mmap_mode="r"),
            noisy2=np.load(path / f"{family}_data_strain_2.npy", mmap_mode="r"),
            pure1=np.load(path / f"{family}_h_strain_1.npy", mmap_mode="r"),
            pure2=np.load(path / f"{family}_h_strain_2.npy", mmap_mode="r"),
            snr1=np.load(path / f"{family}_optimal_SNR_network_1.npy", mmap_mode="r"),
            snr2=np.load(path / f"{family}_optimal_SNR_network_2.npy", mmap_mode="r"),
            source=source,
            timing=pd.read_csv(path / f"{family}_trigger_time_features.csv"),
            targets=detector_frame_targets(source),
        )
    uroot = root / "Unlensed_data_0222"
    return ETBank(
        families=families,
        unlensed_noisy=np.load(uroot / "unlensed_data_strain.npy", mmap_mode="r"),
        unlensed_pure=np.load(uroot / "unlensed_h_strain.npy", mmap_mode="r"),
        unlensed_snr=np.load(uroot / "unlensed_optimal_SNR_network.npy", mmap_mode="r"),
        unlensed_source=pd.read_csv(uroot / "source_samples.csv"),
        unlensed_timing=pd.read_csv(uroot / "unlensed_trigger_time_features.csv"),
    )


def split_array(n: int, seed: int, fractions: tuple[float, float]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def make_splits(bank: ETBank, seed: int) -> dict[str, dict[str, np.ndarray]]:
    splits = {
        family: split_array(len(bank.families[family].noisy1), seed + 101 * pos, (0.70, 0.15))
        for pos, family in enumerate(FAMILIES, start=1)
    }
    # Preserve the paper's 9,000-event validation and test catalogs:
    # 1,500 systems/family -> 6,000 lensed events, plus 3,000 unlensed.
    splits["unlensed"] = split_array(len(bank.unlensed_noisy), seed + 909, (0.40, 0.30))
    return splits


def prepare(values: np.ndarray, rng: np.random.Generator | None, train: bool) -> np.ndarray:
    y = peak_flip_channels(np.asarray(values, dtype=np.float32).copy())
    if train:
        assert rng is not None
        y = np.roll(y, int(rng.integers(-128, 129)), axis=-1)
        y *= float(1.0 + rng.uniform(-0.1, 0.1))
        scale = np.std(y, axis=-1, keepdims=True)
        y += rng.normal(0.0, 0.003 * np.maximum(scale, 1e-6), size=y.shape)
    return zscore_channels(y)


class PairDataset(Dataset):
    def __init__(self, bank: ETBank, splits: dict[str, dict[str, np.ndarray]], seed: int, pure: bool):
        self.bank = bank
        self.pure = bool(pure)
        self.base_seed = int(seed)
        self.rng: np.random.Generator | None = None
        self.rng_worker_id: int | None = None
        self.items = [
            (family, int(idx))
            for family in FAMILIES
            for idx in splits[family]["train"]
        ]

    def __len__(self) -> int:
        return len(self.items)

    def worker_rng(self) -> np.random.Generator:
        info = get_worker_info()
        worker_id = -1 if info is None else int(info.id)
        if self.rng is None or self.rng_worker_id != worker_id:
            seed = self.base_seed if info is None else self.base_seed + int(info.seed)
            self.rng = np.random.default_rng(seed % (2**63 - 1))
            self.rng_worker_id = worker_id
        return self.rng

    def __getitem__(self, item: int):
        family, idx = self.items[item]
        current = self.bank.families[family]
        a = current.pure1[idx] if self.pure else current.noisy1[idx]
        b = current.pure2[idx] if self.pure else current.noisy2[idx]
        rng = self.worker_rng()
        return (
            torch.from_numpy(prepare(a, rng, True)),
            torch.from_numpy(prepare(b, rng, True)),
            torch.from_numpy(current.targets[idx]),
        )


class AdaptDataset(Dataset):
    def __init__(self, bank: ETBank, splits: dict[str, dict[str, np.ndarray]], seed: int):
        self.bank = bank
        self.base_seed = int(seed)
        self.rng: np.random.Generator | None = None
        self.rng_worker_id: int | None = None
        self.items = [
            (family, int(idx))
            for family in FAMILIES
            for idx in splits[family]["train"]
        ]

    def __len__(self) -> int:
        return len(self.items)

    def worker_rng(self) -> np.random.Generator:
        info = get_worker_info()
        worker_id = -1 if info is None else int(info.id)
        if self.rng is None or self.rng_worker_id != worker_id:
            seed = self.base_seed if info is None else self.base_seed + int(info.seed)
            self.rng = np.random.default_rng(seed % (2**63 - 1))
            self.rng_worker_id = worker_id
        return self.rng

    def __getitem__(self, item: int):
        family, idx = self.items[item]
        current = self.bank.families[family]
        rng = self.worker_rng()
        return (
            torch.from_numpy(prepare(current.noisy1[idx], rng, True)),
            torch.from_numpy(prepare(current.noisy2[idx], rng, True)),
            torch.from_numpy(prepare(current.pure1[idx], rng, True)),
            torch.from_numpy(prepare(current.pure2[idx], rng, True)),
            torch.from_numpy(current.targets[idx]),
        )


class CatalogDataset(Dataset):
    def __init__(
        self,
        bank: ETBank,
        splits: dict[str, dict[str, np.ndarray]],
        split: str,
        *,
        pure: bool = False,
        limit_systems_per_family: int | None = None,
        limit_unlensed: int | None = None,
    ):
        self.bank = bank
        self.pure = bool(pure)
        self.rows: list[tuple[str, int, int]] = []
        self.meta: list[dict[str, Any]] = []
        self.partner: list[int] = []
        for family in FAMILIES:
            ids = np.asarray(splits[family][split], dtype=np.int64)
            if limit_systems_per_family is not None:
                ids = ids[: int(limit_systems_per_family)]
            first = len(self.rows)
            for idx in ids:
                self.rows.append((family, int(idx), 1))
                self.meta.append(self._lensed_meta(family, int(idx), 1))
            second = len(self.rows)
            for idx in ids:
                self.rows.append((family, int(idx), 2))
                self.meta.append(self._lensed_meta(family, int(idx), 2))
            count = len(ids)
            self.partner.extend(range(second, second + count))
            self.partner.extend(range(first, first + count))
        ids = np.asarray(splits["unlensed"][split], dtype=np.int64)
        if limit_unlensed is not None:
            ids = ids[: int(limit_unlensed)]
        for idx in ids:
            self.rows.append(("unlensed", int(idx), 0))
            self.meta.append(self._unlensed_meta(int(idx)))
            self.partner.append(-1)
        self.partner = np.asarray(self.partner, dtype=np.int32)

    def _lensed_meta(self, family: str, idx: int, image: int) -> dict[str, Any]:
        current = self.bank.families[family]
        source = current.source.iloc[idx]
        timing = current.timing.iloc[idx]
        return {
            "family": family,
            "source_index": idx,
            "system_id": f"{family}:{idx}",
            "tag": f"L{image}",
            "image": image,
            "snr": float(current.snr1[idx] if image == 1 else current.snr2[idx]),
            "gps": float(timing[f"trigger_time_obs_{image}"]),
            "ra": float(source.ra),
            "dec": float(source.dec),
            "target_logmc": float(current.targets[idx, 0]),
            "target_logitq": float(current.targets[idx, 1]),
        }

    def _unlensed_meta(self, idx: int) -> dict[str, Any]:
        source = self.bank.unlensed_source.iloc[idx]
        timing = self.bank.unlensed_timing.iloc[idx]
        trigger_column = "trigger_time_obs" if "trigger_time_obs" in timing.index else "geocent_time_true"
        return {
            "family": "unlensed",
            "source_index": idx,
            "system_id": f"unlensed:{idx}",
            "tag": "U",
            "image": 0,
            "snr": float(self.bank.unlensed_snr[idx]),
            "gps": float(timing[trigger_column]),
            "ra": float(source.ra),
            "dec": float(source.dec),
            "target_logmc": np.nan,
            "target_logitq": np.nan,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        family, idx, image = self.rows[item]
        if family == "unlensed":
            x = self.bank.unlensed_pure[idx] if self.pure else self.bank.unlensed_noisy[idx]
        else:
            current = self.bank.families[family]
            if self.pure:
                x = current.pure1[idx] if image == 1 else current.pure2[idx]
            else:
                x = current.noisy1[idx] if image == 1 else current.noisy2[idx]
        return torch.from_numpy(prepare(x, None, False))


class PhysicsRegularizedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = InceptionAttentionEncoder1D(
            in_channels=3,
            d_model=192,
            emb_dim=96,
            width_scale=1.0,
            depth=6,
        )
        self.parameter_head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, return_parameters: bool = False):
        embedding = self.base(x)
        if return_parameters:
            return embedding, self.parameter_head(embedding)
        return embedding


def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, beta=0.5)


@torch.no_grad()
def embed_catalog(
    model: PhysicsRegularizedEncoder,
    dataset: CatalogDataset,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings, predictions = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    for values in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z, prediction = model(values.cuda(non_blocking=True), return_parameters=True)
        embeddings.append(z.float().cpu().numpy())
        predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


def retrieval_from_matrix(
    score: np.ndarray,
    partner: np.ndarray,
    meta: list[dict[str, Any]],
) -> tuple[dict[str, float], pd.DataFrame]:
    query = np.flatnonzero(partner >= 0)
    values = np.asarray(score[query], dtype=np.float32).copy()
    values[np.arange(len(query)), query] = -np.inf
    truth = partner[query]
    true_score = values[np.arange(len(query)), truth]
    ranks = 1 + np.sum(values > true_score[:, None], axis=1)
    families = np.asarray([meta[int(idx)]["family"] for idx in query])
    rows = []
    for subset in ("overall", *FAMILIES):
        keep = np.ones(len(query), dtype=bool) if subset == "overall" else families == subset
        current = ranks[keep]
        rows.append(
            {
                "subset": subset,
                "n_queries": int(len(current)),
                "r_at_1": float(np.mean(current <= 1)),
                "r_at_5": float(np.mean(current <= 5)),
                "r_at_10": float(np.mean(current <= 10)),
                "r_at_50": float(np.mean(current <= 50)),
                "median_rank": float(np.median(current)),
            }
        )
    rank_frame = pd.DataFrame(
        {
            "query_index": query.astype(np.int32),
            "partner_index": truth.astype(np.int32),
            "family": families,
            "system_id": [meta[int(idx)]["system_id"] for idx in query],
            "query_snr": [meta[int(idx)]["snr"] for idx in query],
            "partner_snr": [meta[int(idx)]["snr"] for idx in truth],
            "query_rank": ranks.astype(np.int32),
        }
    )
    overall = rows[0]
    metrics = {
        "overall_r_at_1": overall["r_at_1"],
        "overall_r_at_10": overall["r_at_10"],
        "macro_r_at_1": float(np.mean([row["r_at_1"] for row in rows[1:]])),
        "macro_r_at_10": float(np.mean([row["r_at_10"] for row in rows[1:]])),
        "min_family_r_at_10": float(min(row["r_at_10"] for row in rows[1:])),
        "median_rank": overall["median_rank"],
    }
    return metrics, rank_frame


def sample_false_pairs(partner: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left, right = [], []
    remaining = int(count)
    while remaining > 0:
        batch = max(100_000, remaining * 2)
        a = rng.integers(0, len(partner), size=batch, dtype=np.int32)
        b = rng.integers(0, len(partner), size=batch, dtype=np.int32)
        i, j = np.minimum(a, b), np.maximum(a, b)
        keep = (i != j) & (partner[i] != j)
        take = min(remaining, int(keep.sum()))
        left.append(i[keep][:take])
        right.append(j[keep][:take])
        remaining -= take
    return np.concatenate(left), np.concatenate(right)


def true_pairs(partner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = np.flatnonzero(partner > np.arange(len(partner))).astype(np.int32)
    return i, partner[i].astype(np.int32)


def sampled_pair_ap(
    score: np.ndarray,
    partner: np.ndarray,
    false_i: np.ndarray,
    false_j: np.ndarray,
) -> float:
    true_i, true_j = true_pairs(partner)
    values = np.concatenate([score[true_i, true_j], score[false_i, false_j]])
    labels = np.concatenate(
        [np.ones(len(true_i), dtype=np.int8), np.zeros(len(false_i), dtype=np.int8)]
    )
    return float(average_precision_score(labels, values))


def sampled_pair_candidate_metrics(
    score: np.ndarray,
    partner: np.ndarray,
    false_i: np.ndarray,
    false_j: np.ndarray,
) -> dict[str, float]:
    true_i, true_j = true_pairs(partner)
    true_score = score[true_i, true_j]
    false_score = score[false_i, false_j]
    values = np.concatenate([true_score, false_score])
    labels = np.concatenate(
        [np.ones(len(true_score), dtype=np.int8), np.zeros(len(false_score), dtype=np.int8)]
    )
    threshold = float(
        np.sort(true_score)[max(0, len(true_score) - int(math.ceil(0.5 * len(true_score))))]
    )
    true_accepted = int(np.sum(true_score >= threshold))
    false_accepted = int(np.sum(false_score >= threshold))
    return {
        "sampled_average_precision": float(average_precision_score(labels, values)),
        "sampled_precision_at_recall_0p5": float(
            true_accepted / max(true_accepted + false_accepted, 1)
        ),
    }


def feature_standardization(
    values: np.ndarray,
    false_i: np.ndarray,
    false_j: np.ndarray,
    partner: np.ndarray,
) -> dict[str, float]:
    true_i, true_j = true_pairs(partner)
    sample = np.concatenate([values[false_i, false_j], values[true_i, true_j]])
    return {
        "center": float(np.median(sample)),
        "scale": max(float(np.std(sample)), 1e-8),
        "sample_count": int(len(sample)),
    }


def apply_z(values: np.ndarray, config: dict[str, float]) -> np.ndarray:
    return ((values - config["center"]) / config["scale"]).astype(np.float32)


def calibrate_waveform(
    embedding: np.ndarray,
    prediction: np.ndarray,
    dataset: CatalogDataset,
    seed: int,
    false_count: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    cosine = similarity_matrix(embedding).astype(np.float32)
    negative_mc = -np.abs(prediction[:, None, 0] - prediction[None, :, 0]).astype(np.float32)
    negative_q = -np.abs(prediction[:, None, 1] - prediction[None, :, 1]).astype(np.float32)
    false_i, false_j = sample_false_pairs(dataset.partner, false_count, seed)
    configs = {
        "embedding_cosine": feature_standardization(cosine, false_i, false_j, dataset.partner),
        "negative_delta_logmc": feature_standardization(negative_mc, false_i, false_j, dataset.partner),
        "negative_delta_logitq": feature_standardization(negative_q, false_i, false_j, dataset.partner),
    }
    z_cosine = apply_z(cosine, configs["embedding_cosine"])
    z_mc = apply_z(negative_mc, configs["negative_delta_logmc"])
    z_q = apply_z(negative_q, configs["negative_delta_logitq"])
    rows = []
    best = None
    for lambda_mc, lambda_q in itertools.product(INTRINSIC_GRID, repeat=2):
        composite = z_cosine + lambda_mc * z_mc + lambda_q * z_q
        metrics, _ = retrieval_from_matrix(composite, dataset.partner, dataset.meta)
        ap = sampled_pair_ap(composite, dataset.partner, false_i, false_j)
        row = {
            "lambda_mc": lambda_mc,
            "lambda_q": lambda_q,
            **metrics,
            "sampled_average_precision": ap,
            "regularization_l1": lambda_mc + lambda_q,
        }
        rows.append(row)
        key = (
            row["macro_r_at_10"],
            row["min_family_r_at_10"],
            row["sampled_average_precision"],
            row["macro_r_at_1"],
            -row["regularization_l1"],
        )
        if best is None or key > best[0]:
            best = (key, row, composite)
    assert best is not None
    selected = best[1]
    composite = best[2]
    true_i, true_j = true_pairs(dataset.partner)
    likelihood = fit_score_likelihood_ratio(
        composite[true_i, true_j],
        composite[false_i, false_j],
        grid_size=2048,
    )
    score = apply_score_likelihood_ratio(composite, likelihood)
    np.fill_diagonal(score, -np.inf)
    config = {
        "definition": "Validation-frozen log likelihood ratio of embedding cosine plus waveform-derived detector-frame log-Mc/logit-q compatibility.",
        "feature_standardization": configs,
        "z_scope": "global deterministic sample of validation unordered pairs; never row/query standardized",
        "false_pair_calibration_samples": int(false_count),
        "lambda_mc": float(selected["lambda_mc"]),
        "lambda_q": float(selected["lambda_q"]),
        "selection_split": "validation only",
        "selection_metrics": selected,
        "likelihood_ratio": likelihood,
    }
    grid = pd.DataFrame(rows).sort_values(
        [
            "macro_r_at_10",
            "min_family_r_at_10",
            "sampled_average_precision",
            "macro_r_at_1",
            "regularization_l1",
        ],
        ascending=[False, False, False, False, True],
    )
    return score, config, grid.reset_index(drop=True)


def apply_waveform_calibration(
    embedding: np.ndarray,
    prediction: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    cosine = similarity_matrix(embedding).astype(np.float32)
    negative_mc = -np.abs(prediction[:, None, 0] - prediction[None, :, 0]).astype(np.float32)
    negative_q = -np.abs(prediction[:, None, 1] - prediction[None, :, 1]).astype(np.float32)
    feature = config["feature_standardization"]
    composite = (
        apply_z(cosine, feature["embedding_cosine"])
        + float(config["lambda_mc"]) * apply_z(negative_mc, feature["negative_delta_logmc"])
        + float(config["lambda_q"]) * apply_z(negative_q, feature["negative_delta_logitq"])
    )
    score = apply_score_likelihood_ratio(composite, config["likelihood_ratio"])
    np.fill_diagonal(score, -np.inf)
    return score


def build_et_time_calibration(output: Path, seed: int) -> dict[str, Any]:
    path = output / "shared/et_time_delay_likelihood_ratio.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    v3 = load_module(
        REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py",
        "v3_for_et3_v7_time",
    )
    delays, _, label = v3.liao_delay_ratio_samples()
    delays = np.asarray(delays, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lens = rng.choice(delays, size=100_000, replace=True)
    # Match the actual legacy ET catalog calendar without using validation/test
    # event values: image-1 and unlensed times are uniform in the fixed base
    # interval; image-2 times are a base draw plus an independent GW-LMC delay.
    def draw_event_times(size: int) -> np.ndarray:
        base = rng.uniform(ET_BASE_START_GPS, ET_BASE_END_GPS, size=size)
        category = rng.integers(0, 3, size=size)
        delayed = category == 1
        if np.any(delayed):
            base[delayed] += rng.choice(delays, size=int(delayed.sum()), replace=True) * SECONDS_PER_DAY
        return base

    a = draw_event_times(250_000)
    b = draw_event_times(250_000)
    null = np.abs(a - b) / SECONDS_PER_DAY
    null = null[null > 0]
    calibration = fit_time_likelihood_ratio(lens, null, grid_size=2048)
    calibration.update(
        {
            "definition": "log p(delta_t|lensed,E_ET) - log p(delta_t|null,E_ET)",
            "lensed_delay_source": label,
            "calendar": {
                "base_start_gps": ET_BASE_START_GPS,
                "base_end_gps": ET_BASE_END_GPS,
                "base_duration_days": (ET_BASE_END_GPS - ET_BASE_START_GPS) / SECONDS_PER_DAY,
                "duty_cycle": 1.0,
                "image2_policy": "base time plus GW-LMC delay; delayed images may extend beyond base end, matching the existing ET generator",
            },
            "null_definition": "two independent draws from the frozen 1/3 image1, 1/3 image2, 1/3 unlensed calendar mixture",
            "fit_policy": "frozen before validation, held-out test, and weight selection",
            "uses_test_catalog_times": False,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, calibration)
    np.save(path.parent / "et_time_lensed_samples_days.npy", lens.astype(np.float32))
    np.save(path.parent / "et_time_null_samples_days.npy", null.astype(np.float32))
    return calibration


def time_score_matrix(dataset: CatalogDataset, calibration: dict[str, Any]) -> np.ndarray:
    gps = np.asarray([item["gps"] for item in dataset.meta], dtype=np.float64)
    delay = np.abs(gps[:, None] - gps[None, :]) / SECONDS_PER_DAY
    score = apply_time_likelihood_ratio(delay, calibration)
    np.fill_diagonal(score, -np.inf)
    return score


def sky_score_matrix(
    dataset: CatalogDataset,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    raw = pd.DataFrame(
        {
            "ra": [item["ra"] for item in dataset.meta],
            "dec": [item["dec"] for item in dataset.meta],
        }
    )
    observed = pd.DataFrame({"snr": [item["snr"] for item in dataset.meta]})
    scenario = scenario_for_detector("ET3")
    sky = build_observed_sky_table(
        raw,
        observed,
        scenario,
        np.random.default_rng(seed),
        sampling="tangent_2d_gaussian",
    )
    ra = sky["ra_obs"].to_numpy(np.float64)
    dec = sky["dec_obs"].to_numpy(np.float64)
    sigma = np.maximum(sky["sky_sigma_rad"].to_numpy(np.float64), 1e-12)
    unit = np.column_stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    theta = np.arccos(cosine)
    variance = sigma[:, None] ** 2 + sigma[None, :] ** 2
    score = math.log(2.0) - np.log(variance) - theta * theta / (2.0 * variance)
    np.fill_diagonal(score, -np.inf)
    sky["score_definition"] = "analytic tangent-Gaussian log B_sky; same common-source BF as GWTC HEALPix implementation"
    return score.astype(np.float32), sky


def unique_weight_rows(require_positive: bool = False, waveform_zero: bool = False):
    seen = set()
    for waveform, time_weight, sky in itertools.product(WEIGHT_GRID, repeat=3):
        if waveform == time_weight == sky == 0:
            continue
        if require_positive and min(waveform, time_weight, sky) <= 0:
            continue
        if waveform_zero and (waveform != 0 or min(time_weight, sky) <= 0):
            continue
        scale = max(waveform, time_weight, sky)
        normalized = tuple(round(value / scale, 8) for value in (waveform, time_weight, sky))
        if normalized in seen:
            continue
        seen.add(normalized)
        yield {"waveform": waveform / scale, "time": time_weight / scale, "sky": sky / scale}


def combine(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    result = np.zeros_like(components["waveform"], dtype=np.float32)
    for name, weight in weights.items():
        if weight:
            result += float(weight) * components[name]
    np.fill_diagonal(result, -np.inf)
    return result


def select_fusion_modes(
    components: dict[str, np.ndarray],
    dataset: CatalogDataset,
    seed: int,
    false_count: int = 500_000,
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    false_i, false_j = sample_false_pairs(dataset.partner, false_count, seed)
    weight_rows = list(unique_weight_rows())
    query = np.flatnonzero(dataset.partner >= 0).astype(np.int64)
    truth = dataset.partner[query].astype(np.int64)
    families = np.asarray([dataset.meta[int(idx)]["family"] for idx in query])
    component_tensor = np.stack(
        [
            np.nan_to_num(
                np.asarray(components[name][query], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for name in ("waveform", "time", "sky")
        ],
        axis=0,
    )
    tensor = torch.from_numpy(component_tensor).cuda()
    query_gpu = torch.from_numpy(query).cuda()
    truth_gpu = torch.from_numpy(truth).cuda()
    local_gpu = torch.arange(len(query), device="cuda")
    retrieval_by_index: dict[int, dict[str, float]] = {}
    # Batch the exact validation retrieval grid on the GPU.  This changes only
    # execution speed; scores and rank comparisons remain float32 and exact for
    # the frozen grid.
    for start in range(0, len(weight_rows), 12):
        stop = min(len(weight_rows), start + 12)
        weights_gpu = torch.tensor(
            [
                [weight_rows[idx][name] for name in ("waveform", "time", "sky")]
                for idx in range(start, stop)
            ],
            dtype=torch.float32,
            device="cuda",
        )
        score = torch.einsum("bc,cqn->bqn", weights_gpu, tensor)
        true_score = score[:, local_gpu, truth_gpu]
        counts = torch.sum(score > true_score[:, :, None], dim=-1)
        # The diagonal was replaced by zero before transfer to avoid 0*-inf
        # NaNs. Remove a self-match if zero happened to exceed the true score.
        self_score = score[:, local_gpu, query_gpu]
        counts -= (self_score > true_score).to(counts.dtype)
        ranks_batch = (1 + counts).cpu().numpy()
        del score, true_score, counts, self_score
        for offset, ranks in enumerate(ranks_batch):
            family_rows = []
            for family in FAMILIES:
                keep = families == family
                family_rows.append(
                    {
                        "r1": float(np.mean(ranks[keep] <= 1)),
                        "r10": float(np.mean(ranks[keep] <= 10)),
                    }
                )
            retrieval_by_index[start + offset] = {
                "overall_r_at_1": float(np.mean(ranks <= 1)),
                "overall_r_at_10": float(np.mean(ranks <= 10)),
                "macro_r_at_1": float(np.mean([row["r1"] for row in family_rows])),
                "macro_r_at_10": float(np.mean([row["r10"] for row in family_rows])),
                "min_family_r_at_10": float(min(row["r10"] for row in family_rows)),
                "median_rank": float(np.median(ranks)),
            }
    del tensor, component_tensor
    torch.cuda.empty_cache()

    rows = []
    for index, weights in enumerate(weight_rows):
        metrics = retrieval_by_index[index]
        rows.append(
            {
            **weights,
            **metrics,
            "sampled_average_precision": np.nan,
            "sampled_precision_at_recall_0p5": np.nan,
            "weight_l2": math.sqrt(sum(value * value for value in weights.values())),
            }
        )
    frame = pd.DataFrame(rows)
    mode_masks = {
        "time_sky": (frame.waveform == 0) & (frame.time > 0) & (frame.sky > 0),
        "unconstrained": np.ones(len(frame), dtype=bool),
        "strict_positive": (frame.waveform > 0) & (frame.time > 0) & (frame.sky > 0),
    }
    ap_indices: set[int] = set()
    for mask in mode_masks.values():
        subset = frame.loc[mask]
        best_macro = float(subset["macro_r_at_10"].max())
        subset = subset[np.isclose(subset["macro_r_at_10"], best_macro)]
        best_min_family = float(subset["min_family_r_at_10"].max())
        subset = subset[np.isclose(subset["min_family_r_at_10"], best_min_family)]
        ap_indices.update(int(idx) for idx in subset.index)
    for idx in sorted(ap_indices):
        row = frame.loc[idx]
        weights = {name: float(row[name]) for name in ("waveform", "time", "sky")}
        score = combine(components, weights)
        candidate_metrics = sampled_pair_candidate_metrics(
            score,
            dataset.partner,
            false_i,
            false_j,
        )
        for metric, value in candidate_metrics.items():
            frame.loc[idx, metric] = value

    selected: dict[str, dict[str, float]] = {}
    grids: dict[str, pd.DataFrame] = {}
    for mode, mask in mode_masks.items():
        grid = frame.loc[mask].copy()
        grid["_ap_sort"] = grid["sampled_average_precision"].fillna(-np.inf)
        grid = grid.sort_values(
        [
            "macro_r_at_10",
            "min_family_r_at_10",
            "_ap_sort",
            "sampled_precision_at_recall_0p5",
            "macro_r_at_1",
            "weight_l2",
        ],
        ascending=[False, False, False, False, False, True],
        ).drop(columns="_ap_sort")
        grid = grid.reset_index(drop=True)
        selected[mode] = {
            name: float(grid.iloc[0][name]) for name in ("waveform", "time", "sky")
        }
        grids[mode] = grid
    return selected, grids


def training_target_scale(bank: ETBank, splits: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        [bank.families[family].targets[splits[family]["train"]] for family in FAMILIES]
    )
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)


def scaled_target(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (target - mean) / std


def checkpoint_validation(
    model: PhysicsRegularizedEncoder,
    dataset: CatalogDataset,
    batch_size: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, float]:
    z, prediction = embed_catalog(model, dataset, batch_size)
    prediction = prediction * target_std[None, :] + target_mean[None, :]
    cosine = similarity_matrix(z).astype(np.float32)
    mc = -np.abs(prediction[:, None, 0] - prediction[None, :, 0]).astype(np.float32)
    q = -np.abs(prediction[:, None, 1] - prediction[None, :, 1]).astype(np.float32)
    tri = np.triu_indices(len(dataset), 1)
    components = []
    for values in (cosine, mc, q):
        sample = values[tri]
        components.append((values - np.median(sample)) / max(np.std(sample), 1e-8))
    best = None
    for lambda_mc, lambda_q in itertools.product(INTRINSIC_GRID, repeat=2):
        score = components[0] + lambda_mc * components[1] + lambda_q * components[2]
        metrics, _ = retrieval_from_matrix(score, dataset.partner, dataset.meta)
        key = (
            metrics["macro_r_at_10"],
            metrics["min_family_r_at_10"],
            metrics["macro_r_at_1"],
            -(lambda_mc + lambda_q),
        )
        if best is None or key > best[0]:
            best = (key, metrics, lambda_mc, lambda_q)
    assert best is not None
    return {
        **best[1],
        "lambda_mc": float(best[2]),
        "lambda_q": float(best[3]),
        "embedding_effective_rank": effective_rank(z),
    }


def train_seed(
    bank: ETBank,
    splits: dict[str, dict[str, np.ndarray]],
    seed: int,
    output: Path,
    pretrain_epochs: int,
    adapt_epochs: int,
    batch_size: int,
) -> tuple[Path, dict[str, Any]]:
    checkpoint = output / "validation_selected_model.pt"
    summary_path = output / "training_summary.json"
    if checkpoint.exists() and summary_path.exists():
        return checkpoint, json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    target_mean, target_std = training_target_scale(bank, splits)
    mean_gpu = torch.from_numpy(target_mean).cuda()
    std_gpu = torch.from_numpy(target_std).cuda()
    clean = PairDataset(bank, splits, seed + 1, pure=True)
    adapt = AdaptDataset(bank, splits, seed + 2)
    checkpoint_val = CatalogDataset(
        bank,
        splits,
        "val",
        limit_systems_per_family=300,
        limit_unlensed=600,
    )
    seed_everything(seed)
    model = PhysicsRegularizedEncoder().cuda()
    loss_fn = NTXentLoss(0.10)
    history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    clean_loader = DataLoader(
        clean,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        losses = []
        started = time.perf_counter()
        for a, b, target in clean_loader:
            a, b, target = a.cuda(non_blocking=True), b.cuda(non_blocking=True), target.cuda(non_blocking=True)
            target = scaled_target(target, mean_gpu, std_gpu)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                za, pa = model(a, return_parameters=True)
                zb, pb = model(b, return_parameters=True)
                contrastive = loss_fn(za, zb)
                auxiliary = 0.5 * (regression_loss(pa, target) + regression_loss(pb, target))
                loss = contrastive + 0.25 * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {
            "phase": "clean_pretrain",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "epoch_s": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    del clean_loader

    teacher = PhysicsRegularizedEncoder().cuda()
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    best_state = None
    best_metrics = None
    best_key = None
    adapt_loader = DataLoader(
        adapt,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    for epoch in range(1, adapt_epochs + 1):
        model.train()
        losses = []
        started = time.perf_counter()
        for a, b, clean_a, clean_b, target in adapt_loader:
            tensors = [item.cuda(non_blocking=True) for item in (a, b, clean_a, clean_b, target)]
            a, b, clean_a, clean_b, target = tensors
            target = scaled_target(target, mean_gpu, std_gpu)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                za, pa = model(a, return_parameters=True)
                zb, pb = model(b, return_parameters=True)
                contrastive = loss_fn(za, zb)
                auxiliary = 0.5 * (regression_loss(pa, target) + regression_loss(pb, target))
                with torch.no_grad():
                    prototype = F.normalize(teacher(clean_a) + teacher(clean_b), dim=-1)
                identity = torch.arange(len(prototype), device=prototype.device)
                prototype_loss = 0.5 * (
                    F.cross_entropy(za @ prototype.T / 0.10, identity)
                    + F.cross_entropy(zb @ prototype.T / 0.10, identity)
                )
                loss = 0.5 * contrastive + 0.5 * prototype_loss + 0.25 * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {
            "phase": "noisy_adaptation",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "epoch_s": time.perf_counter() - started,
        }
        if epoch == 1 or epoch % 2 == 0 or epoch == adapt_epochs:
            metrics = checkpoint_validation(
                model,
                checkpoint_val,
                max(batch_size * 2, 256),
                target_mean,
                target_std,
            )
            row.update({f"val_{key}": value for key, value in metrics.items()})
            key = (
                metrics["macro_r_at_10"],
                metrics["min_family_r_at_10"],
                metrics["macro_r_at_1"],
                metrics["embedding_effective_rank"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = metrics
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
        history.append(row)
        print(json.dumps(row), flush=True)
    del adapt_loader
    assert best_state is not None
    torch.save(
        {
            "model_state": best_state,
            "target_mean": target_mean,
            "target_std": target_std,
            "backbone": "inception_attention",
            "in_channels": 3,
            "input_samples": INPUT_SAMPLES,
            "sample_rate_hz": MODEL_SAMPLE_RATE,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    summary = {
        "seed": int(seed),
        "clean_pretrain_epochs": int(pretrain_epochs),
        "noisy_adaptation_epochs": int(adapt_epochs),
        "data_loader_workers": 4,
        "augmentation_rng": "independent numpy Generator per persistent PyTorch worker",
        "best_validation": best_metrics,
        "checkpoint_validation_catalog_events": len(checkpoint_val),
        "checkpoint_validation_uses_test": False,
    }
    write_json(summary_path, summary)
    del model, teacher
    torch.cuda.empty_cache()
    return checkpoint, summary


def load_checkpoint(path: Path) -> tuple[PhysicsRegularizedEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PhysicsRegularizedEncoder()
    model.load_state_dict(payload["model_state"])
    return model.cuda().eval(), payload


def evaluate_seed(
    preprocessing: str,
    seed: int,
    out_root: Path,
    pretrain_epochs: int,
    adapt_epochs: int,
    batch_size: int,
    bootstrap_draws: int,
    pair_metrics: bool,
) -> dict[str, Any]:
    seed_dir = out_root / f"seed_{seed}"
    complete = seed_dir / "seed_summary.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    started = time.perf_counter()
    bank = load_bank(preprocessing)
    splits = make_splits(bank, seed)
    audit = split_integrity_audit(splits)
    if not audit["all_disjoint"]:
        raise RuntimeError("Source-system split leakage")
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "split_integrity_audit.json", audit)
    np.savez_compressed(
        seed_dir / "split_indices.npz",
        **{f"{group}_{part}": ids for group, sections in splits.items() for part, ids in sections.items()},
    )
    checkpoint, training = train_seed(
        bank,
        splits,
        seed,
        seed_dir / "encoder",
        pretrain_epochs,
        adapt_epochs,
        batch_size,
    )
    model, payload = load_checkpoint(checkpoint)
    validation = CatalogDataset(bank, splits, "val")
    test = CatalogDataset(bank, splits, "test")
    val_z, val_pred_std = embed_catalog(model, validation, max(batch_size * 2, 256))
    test_z, test_pred_std = embed_catalog(model, test, max(batch_size * 2, 256))
    target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
    target_std = np.asarray(payload["target_std"], dtype=np.float32)
    val_prediction = val_pred_std * target_std[None, :] + target_mean[None, :]
    test_prediction = test_pred_std * target_std[None, :] + target_mean[None, :]
    np.save(seed_dir / "validation_embeddings.npy", val_z.astype(np.float32))
    np.save(seed_dir / "test_embeddings.npy", test_z.astype(np.float32))
    np.save(seed_dir / "validation_waveform_intrinsic_predictions.npy", val_prediction.astype(np.float32))
    np.save(seed_dir / "test_waveform_intrinsic_predictions.npy", test_prediction.astype(np.float32))
    pd.DataFrame(validation.meta).to_parquet(seed_dir / "validation_event_catalog.parquet", index=False)
    pd.DataFrame(test.meta).to_parquet(seed_dir / "test_event_catalog.parquet", index=False)

    val_waveform, waveform_config, waveform_grid = calibrate_waveform(
        val_z,
        val_prediction,
        validation,
        seed + 3000,
        false_count=1_000_000,
    )
    test_waveform = apply_waveform_calibration(test_z, test_prediction, waveform_config)
    write_json(seed_dir / "waveform_evidence_calibration.json", waveform_config)
    waveform_grid.to_csv(seed_dir / "waveform_intrinsic_weight_grid.csv", index=False)

    time_calibration = build_et_time_calibration(out_root, 20260723)
    val_time = time_score_matrix(validation, time_calibration)
    test_time = time_score_matrix(test, time_calibration)
    val_sky, val_sky_events = sky_score_matrix(validation, seed + 4000)
    test_sky, test_sky_events = sky_score_matrix(test, seed + 5000)
    val_sky_events.to_parquet(seed_dir / "validation_observed_sky_events.parquet", index=False)
    test_sky_events.to_parquet(seed_dir / "test_observed_sky_events.parquet", index=False)
    val_components = {"waveform": val_waveform, "time": val_time, "sky": val_sky}
    test_components = {"waveform": test_waveform, "time": test_time, "sky": test_sky}

    selected_fusion, fusion_grids = select_fusion_modes(
        val_components,
        validation,
        seed + 6000,
    )
    weights_time_sky = selected_fusion["time_sky"]
    weights_unconstrained = selected_fusion["unconstrained"]
    weights_positive = selected_fusion["strict_positive"]
    grid_time_sky = fusion_grids["time_sky"]
    grid_unconstrained = fusion_grids["unconstrained"]
    grid_positive = fusion_grids["strict_positive"]
    grid_time_sky.to_csv(seed_dir / "fusion_grid_time_sky.csv", index=False)
    grid_unconstrained.to_csv(seed_dir / "fusion_grid_three_channel_unconstrained.csv", index=False)
    grid_positive.to_csv(seed_dir / "fusion_grid_three_channel_strict_positive.csv", index=False)
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "time_sky_validation_selected": weights_time_sky,
        "three_channel_unconstrained": weights_unconstrained,
        "three_channel_strict_positive": weights_positive,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
    }
    write_json(
        seed_dir / "selected_weights.json",
        {
            "methods": methods,
            "selection_split": "validation only",
            "test_or_real_catalog_used": False,
            "fusion_channel_standardization": "none; each channel is already an absolute validation-frozen log evidence",
        },
    )

    validation_metric_rows = []
    for method, weights in methods.items():
        score = combine(val_components, weights)
        _, ranks = retrieval_from_matrix(score, validation.partner, validation.meta)
        for subset in ("overall", *FAMILIES):
            part = ranks if subset == "overall" else ranks[ranks.family == subset]
            values = part["query_rank"].to_numpy(np.int32)
            validation_metric_rows.append(
                {
                    "method": method,
                    "subset": subset,
                    "n_queries": len(values),
                    "r_at_1": float(np.mean(values <= 1)),
                    "r_at_5": float(np.mean(values <= 5)),
                    "r_at_10": float(np.mean(values <= 10)),
                    "r_at_50": float(np.mean(values <= 50)),
                    "median_rank": float(np.median(values)),
                }
            )
    pd.DataFrame(validation_metric_rows).to_csv(
        seed_dir / "validation_retrieval_metrics.csv",
        index=False,
    )

    query_frames, metric_frames = [], []
    score_cache = {}
    for method, weights in methods.items():
        score = combine(test_components, weights)
        metrics, ranks = retrieval_from_matrix(score, test.partner, test.meta)
        ranks.insert(0, "method", method)
        ranks.insert(0, "seed", int(seed))
        ranks.insert(0, "deployment", "ET3")
        query_frames.append(ranks)
        rows = []
        families = ranks["family"].to_numpy()
        for subset in ("overall", *FAMILIES):
            part = ranks if subset == "overall" else ranks[families == subset]
            values = part["query_rank"].to_numpy(np.int32)
            rows.append(
                {
                    "deployment": "ET3",
                    "seed": int(seed),
                    "method": method,
                    "subset": subset,
                    "n_queries": len(values),
                    "n_systems": int(part["system_id"].nunique()),
                    "r_at_1": float(np.mean(values <= 1)),
                    "r_at_5": float(np.mean(values <= 5)),
                    "r_at_10": float(np.mean(values <= 10)),
                    "r_at_50": float(np.mean(values <= 50)),
                    "median_rank": float(np.median(values)),
                }
            )
        metric_frames.append(pd.DataFrame(rows))
        if method in {"three_channel_strict_positive", "three_channel_unconstrained"}:
            score_cache[method] = score
    query = pd.concat(query_frames, ignore_index=True)
    query.to_parquet(seed_dir / "query_ranks.parquet", index=False)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(seed_dir / "retrieval_metrics.csv", index=False)
    bootstrap = bootstrap_system_ci(query, draws=bootstrap_draws, seed=seed + 9000)
    bootstrap.to_csv(seed_dir / "bootstrap_95ci.csv", index=False)

    snr_rows = []
    bins = [
        ("min_snr_lt_8", -np.inf, 8.0),
        ("min_snr_8_12", 8.0, 12.0),
        ("min_snr_12_20", 12.0, 20.0),
        ("min_snr_ge_20", 20.0, np.inf),
    ]
    for (method, family), frame in query.groupby(["method", "family"]):
        pair_min = np.minimum(frame["query_snr"], frame["partner_snr"])
        for label, low, high in bins:
            keep = (pair_min >= low) & (pair_min < high)
            values = frame.loc[keep, "query_rank"].to_numpy(np.int32)
            snr_rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "family": family,
                    "snr_bin": label,
                    "n_queries": len(values),
                    "r_at_1": float(np.mean(values <= 1)) if len(values) else np.nan,
                    "r_at_10": float(np.mean(values <= 10)) if len(values) else np.nan,
                }
            )
    pd.DataFrame(snr_rows).to_csv(seed_dir / "snr_stratified_retrieval.csv", index=False)

    pair_payload = {}
    if pair_metrics:
        for method, score in score_cache.items():
            summary, operating = full_unordered_pair_metrics(
                score,
                test.partner,
                test.meta,
                aggregation="max",
            )
            summary.update(
                {
                    "deployment": "ET3",
                    "seed": int(seed),
                    "method": method,
                    "unordered_score_rule": (
                        "max(S_ij,S_ji); scores are symmetric absolute "
                        "validation-frozen evidence, so max equals mean"
                    ),
                    "row_standardization": False,
                }
            )
            pair_payload[method] = summary
            write_json(seed_dir / f"pair_level_metrics_{method}.json", summary)
            operating.insert(0, "method", method)
            operating.to_csv(seed_dir / f"pair_level_operating_points_{method}.csv", index=False)

    # Compact pair table for plotting: all true pairs plus a frozen false sample.
    false_i, false_j = sample_false_pairs(test.partner, 200_000, seed + 10000)
    true_i, true_j = true_pairs(test.partner)
    pair_i = np.concatenate([true_i, false_i])
    pair_j = np.concatenate([true_j, false_j])
    diagnostics = pd.DataFrame(
        {
            "idx_i": pair_i,
            "idx_j": pair_j,
            "is_true_pair": np.concatenate(
                [np.ones(len(true_i), np.int8), np.zeros(len(false_i), np.int8)]
            ),
            "waveform_score": test_waveform[pair_i, pair_j],
            "time_score": test_time[pair_i, pair_j],
            "sky_score": test_sky[pair_i, pair_j],
            "strict_positive_score": score_cache["three_channel_strict_positive"][pair_i, pair_j],
            "sampling_strategy": np.concatenate(
                [
                    np.full(len(true_i), "all_true_pairs", object),
                    np.full(len(false_i), "random_false_pairs", object),
                ]
            ),
        }
    )
    diagnostics.to_parquet(seed_dir / "pair_diagnostics_sample.parquet", index=False)

    summary = {
        "status": "complete",
        "seed": int(seed),
        "preprocessing": preprocessing,
        "n_validation_events": len(validation),
        "n_test_events": len(test),
        "n_test_queries": int(np.sum(test.partner >= 0)),
        "n_test_true_pairs": int(len(true_i)),
        "n_test_unordered_pairs": int(len(test) * (len(test) - 1) // 2),
        "selected_weights": methods,
        "training": training,
        "pair_level": pair_payload,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(complete, summary)
    del val_components, test_components, score_cache, model, bank
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def development_compare(args: argparse.Namespace) -> None:
    rows = []
    for preprocessing in ("stride2", "polyphase"):
        root = args.out_root / "development" / preprocessing
        summary = evaluate_seed(
            preprocessing,
            args.development_seed,
            root,
            args.pretrain_epochs,
            args.adapt_epochs,
            args.batch_size,
            min(args.bootstrap_draws, 1000),
            pair_metrics=False,
        )
        metrics = pd.read_csv(
            root / f"seed_{args.development_seed}/validation_retrieval_metrics.csv"
        )
        selected = metrics[
            (metrics.method == "three_channel_strict_positive") & (metrics.subset == "overall")
        ].iloc[0]
        waveform = metrics[(metrics.method == "waveform_only") & (metrics.subset == "overall")].iloc[0]
        rows.append(
            {
                "preprocessing": preprocessing,
                "validation_seed": args.development_seed,
                "heldout_test_used_for_selection": False,
                "waveform_r_at_1": waveform.r_at_1,
                "waveform_r_at_10": waveform.r_at_10,
                "three_channel_r_at_1": selected.r_at_1,
                "three_channel_r_at_10": selected.r_at_10,
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["waveform_r_at_10", "waveform_r_at_1", "three_channel_r_at_10"],
        ascending=False,
    )
    path = args.out_root / "development/preprocessing_comparison_validation_only.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    selected = str(frame.iloc[0].preprocessing)
    write_json(
        args.out_root / "development/frozen_preprocessing_selection.json",
        {
            "status": "frozen_before_formal_seeds",
            "selected_preprocessing": selected,
            "selection_rule": "validation waveform R@10, then waveform R@1, then strict-positive three-channel R@10",
            "development_seed": args.development_seed,
            "formal_seeds": args.formal_seeds,
            "heldout_formal_test_used": False,
            "comparison": rows,
        },
    )


def aggregate(out_root: Path, seeds: list[int]) -> None:
    query = pd.concat(
        [pd.read_parquet(out_root / f"seed_{seed}/query_ranks.parquet") for seed in seeds],
        ignore_index=True,
    )
    query.to_parquet(out_root / "et3_query_ranks_all_seeds.parquet", index=False)
    per_seed = metric_rows(query)
    per_seed.to_csv(out_root / "et3_retrieval_metrics_per_seed.csv", index=False)
    across_seed_summary(per_seed).to_csv(
        out_root / "et3_retrieval_metrics_across_seed_summary.csv",
        index=False,
    )
    bootstrap = pd.concat(
        [pd.read_csv(out_root / f"seed_{seed}/bootstrap_95ci.csv") for seed in seeds],
        ignore_index=True,
    )
    bootstrap.to_csv(out_root / "et3_bootstrap_95ci_per_seed.csv", index=False)
    snr = pd.concat(
        [pd.read_csv(out_root / f"seed_{seed}/snr_stratified_retrieval.csv") for seed in seeds],
        ignore_index=True,
    )
    snr.to_csv(out_root / "et3_snr_stratified_retrieval_per_seed.csv", index=False)
    summary = (
        snr.groupby(["method", "family", "snr_bin"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_queries_mean=("n_queries", "mean"),
            r_at_1_mean=("r_at_1", "mean"),
            r_at_1_std=("r_at_1", "std"),
            r_at_10_mean=("r_at_10", "mean"),
            r_at_10_std=("r_at_10", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_root / "et3_snr_stratified_retrieval_summary.csv", index=False)

    weight_rows = []
    for seed in seeds:
        path = out_root / f"seed_{seed}/selected_weights.json"
        if not path.exists():
            continue
        methods = json.loads(path.read_text(encoding="utf-8"))["methods"]
        for method, weights in methods.items():
            weight_rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "waveform_weight": float(weights["waveform"]),
                    "time_weight": float(weights["time"]),
                    "sky_weight": float(weights["sky"]),
                }
            )
    if weight_rows:
        weights = pd.DataFrame(weight_rows)
        weights.to_csv(out_root / "et3_selected_weights_per_seed.csv", index=False)
        (
            weights.groupby("method", sort=False)
            .agg(
                n_seeds=("seed", "nunique"),
                waveform_weight_mean=("waveform_weight", "mean"),
                waveform_weight_std=("waveform_weight", "std"),
                time_weight_mean=("time_weight", "mean"),
                time_weight_std=("time_weight", "std"),
                sky_weight_mean=("sky_weight", "mean"),
                sky_weight_std=("sky_weight", "std"),
            )
            .reset_index()
            .to_csv(out_root / "et3_selected_weights_summary.csv", index=False)
        )

    pair_rows = []
    operating_frames = []
    pair_methods = (
        "three_channel_strict_positive",
        "three_channel_unconstrained",
    )
    for seed in seeds:
        for method in pair_methods:
            metric_path = out_root / f"seed_{seed}/pair_level_metrics_{method}.json"
            if metric_path.exists():
                payload = json.loads(metric_path.read_text(encoding="utf-8"))
                payload.update({"seed": int(seed), "method": method})
                pair_rows.append(payload)
            operating_path = (
                out_root / f"seed_{seed}/pair_level_operating_points_{method}.csv"
            )
            if operating_path.exists():
                operating = pd.read_csv(operating_path)
                operating.insert(0, "seed", int(seed))
                operating_frames.append(operating)
    if pair_rows:
        pair_per_seed = pd.DataFrame(pair_rows)
        pair_per_seed.to_csv(
            out_root / "et3_pair_level_metrics_per_seed.csv",
            index=False,
        )
        identity = {
            "deployment",
            "seed",
            "method",
            "n_events",
            "n_pairs",
            "n_true_pairs",
            "n_false_pairs",
            "aggregation",
            "row_standardization",
        }
        metrics = [
            column
            for column in pair_per_seed.columns
            if column not in identity
            and pd.api.types.is_numeric_dtype(pair_per_seed[column])
        ]
        rows = []
        for method, frame in pair_per_seed.groupby("method", sort=False):
            for metric in metrics:
                values = frame[metric].dropna().to_numpy(np.float64)
                if not len(values):
                    continue
                rows.append(
                    {
                        "deployment": "ET3",
                        "method": method,
                        "metric": metric,
                        "n_seeds": int(len(values)),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    }
                )
        pd.DataFrame(rows).to_csv(
            out_root / "et3_pair_level_metrics_across_seed_summary.csv",
            index=False,
        )
    if operating_frames:
        pd.concat(operating_frames, ignore_index=True).to_csv(
            out_root / "et3_pair_level_operating_points_per_seed.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "formal"), required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-seed", type=int, default=DEFAULT_DEVELOPMENT_SEED)
    parser.add_argument("--formal-seeds", type=int, nargs="+", default=list(DEFAULT_FORMAL_SEEDS))
    parser.add_argument("--pretrain-epochs", type=int, default=16)
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--skip-pair-metrics", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.stage == "development":
        development_compare(args)
        return
    frozen_path = args.out_root / "development/frozen_preprocessing_selection.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run --stage development before formal evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    preprocessing = str(frozen["selected_preprocessing"])
    formal_root = args.out_root / "formal"
    write_json(
        formal_root / "run_config.json",
        {
            "status": "running",
            "preprocessing": preprocessing,
            "formal_seeds": args.formal_seeds,
            "pretrain_epochs": args.pretrain_epochs,
            "adapt_epochs": args.adapt_epochs,
            "batch_size": args.batch_size,
            "waveform_input": "ET1/ET2/ET3, [GPS-1.75,GPS+0.25] s, 2048 Hz, 4096 samples",
            "sky_score": "absolute common-source log Bayes factor",
            "time_score": "GW-LMC population delay LR under frozen ET synthetic calendar",
            "row_standardization": False,
            "fusion_selection": "validation only",
            "data_loader_workers": 4,
            "augmentation_rng": "independent numpy Generator per persistent PyTorch worker",
        },
    )
    completed = []
    for seed in args.formal_seeds:
        evaluate_seed(
            preprocessing,
            int(seed),
            formal_root,
            args.pretrain_epochs,
            args.adapt_epochs,
            args.batch_size,
            args.bootstrap_draws,
            pair_metrics=not args.skip_pair_metrics,
        )
        completed.append(int(seed))
        aggregate(formal_root, completed)
    config = json.loads((formal_root / "run_config.json").read_text(encoding="utf-8"))
    config["status"] = "complete"
    write_json(formal_root / "run_config.json", config)


if __name__ == "__main__":
    main()
