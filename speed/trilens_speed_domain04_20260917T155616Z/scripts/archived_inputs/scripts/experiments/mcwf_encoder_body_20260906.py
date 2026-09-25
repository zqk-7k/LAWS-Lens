#!/usr/bin/env python3
"""Train source-aware waveform representations without real PE inputs.

All science outputs live in a new run directory. The cached physical filter
bank is a strain feature extractor, not a PE likelihood or a sky/time input.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["GW_WAVEFORM_INPUT_SAMPLES"] = "4096"

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf
import mcwf_phasebank_20260905 as phase

PREVIOUS = dev.PROJECT / "results/main_o3_mcwf_dev_v1_20260905T135000Z"
MODEL_SEEDS = (202609061, 202609062, 202609063)
CONFIGS = ("PHASE-SOURCE", "RAW-PHASE-SOURCE", "PSD-PHASE-SOURCE", "O3-RUNMATCHED-RAW")
EPOCHS = 50


def initialize(root: Path) -> None:
    path = root / "contracts/ENCODER_BODY_V2.json"
    if path.exists():
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Refusing a nonempty directory without this contract")
    for directory in ("contracts", "scripts", "models", "cache", "tables", "results", "audit", "logs", "reports", "figures", "manifest"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    protected = pd.read_csv(PREVIOUS / "manifest/PROTECTED_INPUT_SHA256.csv")
    for item in protected.itertuples(index=False):
        if dev.sha(Path(item.path)) != item.sha256:
            raise RuntimeError(f"Protected input changed: {item.path}")
    dev.csv_write(root / "manifest/PROTECTED_INPUT_SHA256.csv", protected)
    for name in ("gwtc3_all_pair_pe_official.parquet", "gwtc4_all_pair_pe_official_verified.parquet", "gwtc3_event_PE_reference.csv"):
        source = PREVIOUS / "audit" / name
        if source.exists():
            shutil.copy2(source, root / "audit" / name)
    contract = {
        "code": "MAIN-O3-MCWF-ENCODER-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(dev.MAIN), "prior_exploration": str(PREVIOUS),
        "goal_not_completed_by_packaging": True,
        "input": {"detectors": ["H1", "L1"], "seconds": 2, "samples": 4096, "sample_rate": 2048,
                  "raw_preprocessing": "unchanged last4096, per-detector peak sign convention and channel standardization; reject nonfinite input"},
        "frozen": ["time_score", "sky_raw_log_bf", "Nside512 ordering-corrected sky", "real scope", "C-fixed per-seed weights"],
        "changed": "learned waveform representation body and mass prediction jointly optimized; not an auxiliary penalty on unchanged old embeddings",
        "development_data": str(dev.OLD / "data/development"),
        "data_role": "reused development population; historical test and observed real O3/O4a are development comparisons, NOT blind confirmation",
        "network_inputs": "only 2s detector strain and fixed strain-derived quadrature-filter features; never real PE, event identity, sky, time delay or official FPP",
        "training_targets": "independent simulation source parent UID and detector-frame chirp mass",
        "source_balance": "four passes/epoch; one sample group per batch entry; two independent waveform/noise views per source, every parent equal expected weight",
        "models": {
            "PHASE-SOURCE": "fixed physical bank -> jointly trained 256/128 phase body, source embedding and mass distribution; phase body initialized from simulation-only head",
            "RAW-PHASE-SOURCE": "same phase body plus trainable InceptionAttention raw2s branch; only this branch is added relative to PHASE-SOURCE",
        },
        "loss": "source-supervised contrastive loss(T=0.1) + categorical soft-label logMc cross entropy; both propagate through representation body",
        "epochs": EPOCHS, "seeds": list(MODEL_SEEDS), "batch_sources": 64,
        "optimizer": "AdamW lr2e-4 weight_decay1e-4, cosine to2e-5, gradient norm<=5",
        "checkpoint_rule": "minimum independent development validation CE +0.2*source contrastive loss; earliest tie",
        "temperature_grid": [.5, .75, 1., 1.25, 1.5, 2., 3., 4.],
        "temperature_rule": "minimum development validation CE; tie closest1",
        "calibration_rule": "simulation validation only, frozen before historical-test or real audit; no change to time/sky weights",
        "catastrophic_PE_definition": "BC_Mc<0.1 OR D_Mc>5; missing PE reported separately",
        "primary_development_target": "zero catastrophic Mc pairs in O3 consensus waveform Top10 AND C-fixed Top10; report all individual seeds and budgets20/50/100",
        "O4a_transfer_target": "no catastrophic Mc pairs in either consensus Top10; BC_Mc>=0.5 and Dmax<=3 counts not below frozen baseline",
        "injection_guardrails": {"R10_drop_max": .02, "AUPRC_drop_max": .005, "F50_F90_ratio_max": 1.10},
        "final_confirmation": "requires separate newly generated source/noise-disjoint injection data after model freeze; reused test is insufficient",
        "official_audit": "descriptive PO/ML/Phazap and GOLUM/Hanabi cross-match, not labels or a success guarantee",
        "promotion": "no automatic adoption or paper modification; archive failed variants; do not declare goal met on a single seed or PE alone",
        "references": ["https://arxiv.org/abs/2004.11362", "https://arxiv.org/abs/2002.05709", "https://pycbc.org/pycbc/latest/html/pycbc.filter.html"],
    }
    dev.json_write(path, contract)
    dev.json_write(root / "contracts/STATUS.json", {"status": "ENCODER_DEVELOPMENT_ACTIVE", "goal_achieved": False})
    shutil.copy2(__file__, root / "scripts" / Path(__file__).name)
    print(json.dumps({"initialized": str(root), "protected_files_verified": len(protected)}), flush=True)


def source_metadata(dep: str, split: str) -> pd.DataFrame:
    parts = []
    for family in ("sis", "pm"):
        frame = pd.read_parquet(dev.OLD / f"data/development/{dep}/{family}_{split}_metadata.parquet")
        for image in ("a", "b"):
            item = frame.copy()
            item["family_slot"] = family
            item["image"] = image
            parts.append(item)
    return pd.concat(parts, ignore_index=True)


def prepare(root: Path, dep: str) -> Path:
    out = root / "cache" / dep
    if (out / "DATA_COMPLETE.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    group_sets = {}
    for split in ("train", "validation"):
        meta = source_metadata(dep, split)
        groups = meta.waveform_parent_uid.astype(str).to_numpy()
        group_sets[split] = set(groups)
        fpath = PREVIOUS / f"cache/phasebank/{dep}/{split}_features.npy"
        ypath = PREVIOUS / f"cache/masstf/{dep}/{split}_targets.npy"
        x = np.load(fpath, mmap_mode="r")
        y = np.load(ypath)
        if len(meta) != len(x) or not np.allclose(y, tf.target_prob(np.log(meta.chirp_mass_detector.to_numpy())), atol=1e-6):
            raise RuntimeError("Cached phase feature / mass label order mismatch")
        oldgroups = pd.read_csv(PREVIOUS / f"cache/masstf/{dep}/{split}_groups.csv").parent_uid.astype(str).to_numpy()
        if not np.array_equal(oldgroups, groups):
            raise RuntimeError("Cached source order mismatch")
        rawpath = out / f"{split}_raw2s.npy"
        if not rawpath.exists():
            raw = np.lib.format.open_memmap(rawpath, mode="w+", dtype=np.float16, shape=(len(meta), 2, 4096))
            offset = 0
            for family in ("sis", "pm"):
                for image in ("a", "b"):
                    path = dev.OLD / f"cache/window_views/{dep}/{family}_{split}_{image}_2s.npy"
                    block = np.load(path, mmap_mode="r")
                    if block.shape[1:] != (2, 4096) or not np.isfinite(block).all():
                        raise ValueError(f"Invalid preprocessed training strain: {path}")
                    raw[offset:offset + len(block)] = block
                    offset += len(block)
            raw.flush()
        meta.to_parquet(out / f"{split}_metadata.parquet", index=False)
        summaries.append({"split": split, "rows": len(meta), "independent_source_groups": len(set(groups)),
                          "phase_features_sha256": dev.sha(fpath), "targets_sha256": dev.sha(ypath),
                          "raw2s_sha256": dev.sha(rawpath)})
    overlap = group_sets["train"] & group_sets["validation"]
    if overlap:
        raise RuntimeError(f"Train/validation parent leakage: {len(overlap)}")
    dev.json_write(out / "DATA_COMPLETE.json", {"source_group_overlap": 0, "inputs": summaries, "new_strain_generated": False})
    return out


class Encoder(nn.Module):
    def __init__(self, config: str):
        super().__init__()
        self.config = config
        self.phase = phase.MassPhase()
        if config in ("RAW-PHASE-SOURCE", "O3-RUNMATCHED-RAW"):
            self.raw = dev.TRAIN.InceptionAttentionEncoder1D(in_channels=2, d_model=192, emb_dim=96, width_scale=1., depth=6)
            self.merge = nn.Sequential(nn.Linear(128 + 96, 128), nn.LayerNorm(128), nn.SiLU())
        else:
            self.raw = None
            self.merge = nn.Identity()
        self.embedding = nn.Linear(128, 128)

    def forward(self, features: torch.Tensor, strain: torch.Tensor | None = None):
        hidden = self.phase.layers(features)
        if self.raw is not None:
            hidden = self.merge(torch.cat([hidden, self.raw(strain)], dim=-1))
        logits = self.phase.head(hidden)
        z = F.normalize(self.embedding(hidden), dim=-1)
        return logits, z


def source_contrastive(z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """SupCon with source identity, allowing more than two views per parent."""
    z = F.normalize(z.float(), dim=-1)
    logits = z @ z.T / .1
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = groups[:, None].eq(groups[None, :]) & ~diagonal
    valid = positive.any(-1)
    logden = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=-1)
    logprob = logits - logden[:, None]
    terms = torch.where(positive, logprob, torch.zeros_like(logprob))
    if not valid.any():
        return z.sum() * 0
    return -(terms.sum(-1)[valid] / positive.sum(-1)[valid]).mean()


@torch.no_grad()
def infer(model: Encoder, x: np.ndarray, raw: np.ndarray | None, batch: int = 128):
    model.eval()
    logits, embeddings = [], []
    for start in range(0, len(x), batch):
        features = torch.as_tensor(np.asarray(x[start:start + batch]), dtype=torch.float32, device="cuda")
        strain = None if model.raw is None else torch.as_tensor(np.asarray(raw[start:start + batch]), dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, z = model(features, strain)
        logits.append(l.float().cpu().numpy())
        embeddings.append(z.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def train(root: Path, dep: str, config: str, seed: int) -> None:
    initialize(root)
    out = root / f"models/{config}/{dep}/seed_{seed}"
    if (out / "COMPLETE.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    cache = prepare(root, dep)
    if config == "O3-RUNMATCHED-RAW":
        if dep != "gwtc3":
            raise ValueError("This is an O3-only data correction; O4a is unchanged")
        cache = root / "cache/runmatched/gwtc3"
        feature_root = cache
    else:
        feature_root = (root / f"cache/adaptive_psd/{dep}" if config == "PSD-PHASE-SOURCE"
                        else PREVIOUS / f"cache/phasebank/{dep}")
    x = np.load(feature_root / "train_features.npy")
    vx = np.load(feature_root / "validation_features.npy")
    y = np.load(PREVIOUS / f"cache/masstf/{dep}/train_targets.npy")
    vy = np.load(PREVIOUS / f"cache/masstf/{dep}/validation_targets.npy")
    meta = pd.read_parquet(cache / "train_metadata.parquet")
    vmeta = pd.read_parquet(cache / "validation_metadata.parquet")
    groups, labels = pd.factorize(meta.waveform_parent_uid, sort=True)
    vg, _ = pd.factorize(vmeta.waveform_parent_uid, sort=True)
    members = [np.flatnonzero(groups == j) for j in range(len(labels))]
    tx = np.load(cache / "train_raw2s.npy", mmap_mode="r")
    tv = np.load(cache / "validation_raw2s.npy", mmap_mode="r")
    prior_seed = (202609051, 202609052, 202609053)[MODEL_SEEDS.index(seed)]
    warm = PREVIOUS / f"models/PHASEBANK/{dep}/seed_{prior_seed}/validation_selected_model.pt"
    ck = torch.load(warm, weights_only=False, map_location="cpu")
    mu, sd = ((x.mean(0), x.std(0).clip(.05)) if config in ("PSD-PHASE-SOURCE", "O3-RUNMATCHED-RAW")
              else (ck["mu"], ck["sd"]))
    if config == "O3-RUNMATCHED-RAW":
        ck["bankpath"] = str(feature_root / "quadrature_bank.npy")
    x = (x - mu) / sd
    vx = (vx - mu) / sd
    dev.TRAIN.seed_everything(seed)
    model = Encoder(config).cuda()
    model.phase.load_state_dict(ck["state"])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=2e-5)
    history, best, start_epoch = [], float("inf"), 1
    resume = out / "resume.pt"
    if resume.exists():
        r = torch.load(resume, weights_only=False, map_location="cpu")
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["optimizer"])
        schedule.load_state_dict(r["scheduler"])
        torch.set_rng_state(r["rng"])
        torch.cuda.set_rng_state_all(r["cuda_rng"])
        history, best, start_epoch = r["history"], r["best"], r["epoch"] + 1
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS + 1):
        started = time.perf_counter()
        rng = np.random.default_rng(seed + epoch)
        # Independent noise views, balanced by physical parent instead of row.
        orders = [rng.permutation(len(labels)) for _ in range(4)]
        model.train()
        losses = []
        for order in orders:
            for start in range(0, len(order), 64):
                sources = order[start:start + 64]
                paired = np.stack([rng.choice(members[j], 2, replace=False) for j in sources])
                ids = paired.T.reshape(-1)
                sg = torch.as_tensor(np.tile(sources, 2), device="cuda")
                features = torch.as_tensor(x[ids], dtype=torch.float32, device="cuda")
                raw = None if model.raw is None else torch.as_tensor(np.asarray(tx[ids]), dtype=torch.float32, device="cuda")
                target = torch.as_tensor(y[ids], device="cuda")
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, z = model(features, raw)
                    ce = -(F.log_softmax(l.float(), -1) * target).sum(-1).mean()
                    sc = source_contrastive(z, sg)
                    loss = ce + sc
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                losses.append([loss.item(), ce.item(), sc.item()])
        schedule.step()
        l, z = infer(model, vx, tv)
        lp = l.astype(float) - np.logaddexp.reduce(l.astype(float), axis=-1, keepdims=True)
        ce = float(-(vy * lp).sum(-1).mean())
        vsc = float(source_contrastive(torch.as_tensor(z, device="cuda"), torch.as_tensor(vg, device="cuda")))
        criterion = ce + .2 * vsc
        cosine = z.astype(float) @ z.astype(float).T
        truth = vg[:, None] == vg[None, :]
        np.fill_diagonal(truth, False)
        np.fill_diagonal(cosine, -np.inf)
        ranks = []
        for i in range(len(vg)):
            target = np.flatnonzero(truth[i])
            if len(target):
                ranks.append(1 + np.sum(cosine[i] > cosine[i, target].max()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses, axis=0)[0]),
               "train_ce": float(np.mean(losses, axis=0)[1]), "train_source_contrastive": float(np.mean(losses, axis=0)[2]),
               "validation_ce": ce, "validation_source_contrastive": vsc, "selection_loss": criterion,
               "validation_embedding_R1": float(np.mean(np.asarray(ranks) <= 1)),
               "validation_embedding_R10": float(np.mean(np.asarray(ranks) <= 10)),
               "validation_logmc_mae": float(np.mean(np.abs(np.exp(lp) @ tf.LOG_CENTERS - vy @ tf.LOG_CENTERS))),
               "seconds": time.perf_counter() - started, "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
        history.append(row)
        if criterion < best:
            best = criterion
            torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "config": config, "seed": seed, "deployment": dep, "epoch": epoch,
                        "mu": mu, "sd": sd, "prior": ck["prior"], "bankpath": ck["bankpath"],
                        "warm_start": str(warm), "warm_sha256": dev.sha(warm), "criterion": criterion,
                        "training_feature_path": str(feature_root / "train_features.npy"),
                        "training_feature_sha256": dev.sha(feature_root / "train_features.npy")},
                       out / "validation_selected_model.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": schedule.state_dict(),
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "history": history, "best": best, "epoch": epoch}, resume)
        dev.csv_write(out / "training_history.csv", pd.DataFrame(history))
        print(json.dumps({"config": config, "deployment": dep, "seed": seed, **row}), flush=True)
    payload = torch.load(out / "validation_selected_model.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(payload["model"])
    l, z = infer(model, vx, tv)
    grid = []
    for temperature in (.5, .75, 1., 1.25, 1.5, 2., 3., 4.):
        lp = l.astype(float) / temperature
        lp -= np.logaddexp.reduce(lp, axis=-1, keepdims=True)
        grid.append({"temperature": temperature, "ce": float(-(vy * lp).sum(-1).mean())})
    payload["temperature"] = min(grid, key=lambda r: (r["ce"], abs(r["temperature"] - 1)))["temperature"]
    torch.save(payload, out / "validation_selected_model.pt")
    dev.csv_write(out / "temperature_grid.csv", pd.DataFrame(grid))
    np.savez_compressed(out / "development_validation.npz", logits=l, embedding=z, truth=vy, group=vg)
    dev.json_write(out / "COMPLETE.json", {"selected_epoch": payload["epoch"], "temperature": payload["temperature"],
        "checkpoint_sha256": dev.sha(out / "validation_selected_model.pt"), "encoder_body_trained": True,
        "n_train_sources": len(labels), "n_train_rows": len(x), "real_PE_used_for_training": False})


@torch.no_grad()
def encode(checkpoint: Path, full24: np.ndarray):
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if ck["config"] == "PSD-PHASE-SOURCE":
        raise RuntimeError("Event-PSD encoder requires explicit event PSD context")
    model = Encoder(ck["config"]).cuda().eval()
    model.load_state_dict(ck["model"])
    short = dev.TRAIN.make_window_view(np.asarray(full24[..., -4096:], dtype=np.float32), 2)
    if not np.isfinite(short).all():
        raise ValueError("Nonfinite real or injection input is not eligible")
    bank = Path(ck["bankpath"])
    if not bank.is_absolute():
        bank = dev.PROJECT / bank
    features = phase.features(short, np.load(bank))
    l, z = infer(model, (features - ck["mu"]) / ck["sd"], short)
    lp = l.astype(float) / ck["temperature"]
    lp -= np.logaddexp.reduce(lp, axis=-1, keepdims=True)
    return np.exp(lp), z, ck


def tests() -> None:
    torch.manual_seed(42)
    z = F.normalize(torch.randn(8, 16), dim=-1)
    groups = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    baseline = source_contrastive(z, groups)
    permutation = torch.randperm(8)
    assert torch.allclose(baseline, source_contrastive(z[permutation], groups[permutation]), atol=1e-6)
    duplicate = F.normalize(torch.randn(4, 16), dim=-1).repeat(2, 1)
    assert source_contrastive(duplicate, groups) < baseline
    z.requires_grad_()
    source_contrastive(z, groups).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert tf.target_prob(np.array([np.log(5.), np.log(200.)])).sum(1).tolist() == [1., 1.]
    print("PASS: source contrastive permutation, pair identity, finite gradient, mass-label normalization", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--deployment", choices=("gwtc3", "gwtc4"))
    parser.add_argument("--config", choices=CONFIGS, default=CONFIGS[0])
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS, default=MODEL_SEEDS[0])
    parser.add_argument("--action", choices=("init", "train", "test"), required=True)
    args = parser.parse_args()
    if args.action == "test":
        tests()
    elif args.action == "init":
        initialize(args.root)
    else:
        train(args.root, args.deployment, args.config, args.seed)
