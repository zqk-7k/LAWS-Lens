#!/usr/bin/env python3
"""Waveform-only, multiresolution-STFT chirp-mass estimator.

The learned categorical distribution is a calibrated predictive distribution
under the simulation training population, not a replacement for LVK full PE.
No real event parameter, event ID, sky score, or time-delay score is an input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

import mcwf_development_20260905 as dev

LOG_EDGES = np.linspace(np.log(5.), np.log(200.), 65)
LOG_CENTERS = (LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2
SEEDS = (202609051, 202609052, 202609053)


class MassTF(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        cin = 6
        for cout in (24, 48, 96):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                       nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU()]
            cin = cout
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 8 * 12, 128), nn.SiLU(), nn.Dropout(.15))
        self.mass = nn.Linear(128, 64)

    def forward(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)[None, None, :, None].expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)[None, None, None, :].expand(b, 1, h, w)
        z = self.head(self.conv(torch.cat([x, yy, xx], 1)))
        return self.mass(z), F.normalize(z, dim=-1)


@torch.no_grad()
def features(values, batch=128):
    """PSD-whitened two-second strain to 4 STFT-power channels."""
    chunks = []
    for start in range(0, len(values), batch):
        x = torch.as_tensor(np.asarray(values[start:start+batch], dtype=np.float32), device="cuda")
        if x.shape[1:] != (2, 4096) or not torch.isfinite(x).all():
            raise ValueError("Need finite 2-detector 4096-sample input")
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        b = len(x)
        views = []
        for nfft in (256, 512):
            stft = torch.stft(x.reshape(b * 2, 4096), n_fft=nfft, hop_length=32,
                             window=torch.hann_window(nfft, device=x.device), return_complex=True,
                             normalized=True, center=True)
            power = stft.abs().square()
            f = torch.fft.rfftfreq(nfft, 1 / 2048).to(x.device)
            power = power[:, (f >= 40) & (f <= 580)]
            noise = power.median(-1, keepdim=True).values.clamp_min(1e-4)
            contrast = torch.log1p(power / noise)
            view = F.interpolate(contrast[:, None], size=(64, 96), mode="bilinear", align_corners=False)
            views.append(view.reshape(b, 2, 64, 96))
        chunks.append(torch.cat(views, dim=1).cpu().numpy().astype(np.float16))
    return np.concatenate(chunks)


def target_prob(logmc):
    c = np.asarray(logmc)
    position = np.clip((c - LOG_CENTERS[0]) / (LOG_CENTERS[1] - LOG_CENTERS[0]), 0, 63)
    left = np.floor(position).astype(int)
    right = np.minimum(left + 1, 63)
    prob = np.zeros((len(c), 64), dtype=np.float32)
    prob[np.arange(len(c)), left] += 1 - (position - left)
    prob[np.arange(len(c)), right] += position - left
    return prob


def initialize_round(root):
    path = root / "contracts/ROUND1_MASSTF.json"
    if path.exists(): return
    dev.json_write(path, {
        "round": "R1-MASSTF",
        "purpose": "test whether frequency-evolution features recover Mc more reliably than the old normalized time-domain auxiliary head",
        "input": "same preprocessed peak 2 s H1/L1, 4096 samples at 2048 Hz; STFT nfft256/512 hop32, 40-580 Hz",
        "time_sky_and_initial_fusion_frozen": True,
        "supervision": "synthetic detector-frame log-Mc only; no real PE/official labels",
        "estimator": "CNN 64-bin categorical predictive mass distribution, empirical training class prior",
        "interpretation": "simulation-conditional predictive distribution; not full PE or astrophysical evidence",
        "source": str(dev.OLD / "data/development"),
        "data_role": "existing independent development bank; reused historical injection tests are development comparators",
        "model_seeds": list(SEEDS), "epochs": 50, "batch_size": 96,
        "optimizer": "AdamW lr3e-4 weight_decay1e-4 cosine schedule", "checkpoint": "minimum source-disjoint simulation validation cross-entropy",
        "mass_range_detector_Msun": [5, 200],
        "temperature_grid": [.5, .75, 1., 1.25, 1.5, 2., 3., 4.],
        "temperature_choice": "minimum validation soft-label cross-entropy; tied closest to1",
        "pair_feature": "log(sum_b p_i[b]*p_j[b]/training_prior[b]); numerical log floor1e-12",
        "waveform_variants": ["mass-TF-only diagnostic", "old waveform plus negative mass-consistency penalty", "noncompensating mass-supported waveform cap"],
        "gamma_grid": [0., .25, .5, 1., 2., 4.],
        "guard_formula": "old_Zwf + gamma * min(log_mass_overlap,0)",
        "cap_formula": "min(old_Zwf, gamma*log_mass_overlap), gamma>0",
        "selection": "validation R10/AP noninferiority guardrails then F50,F90,AP,R10 and smallest gamma; never real PE or official FPP",
        "scope_change": False, "no_guaranteed_improvement": True,
        "references": ["https://arxiv.org/abs/2106.12594", "https://arxiv.org/abs/2505.18311", "https://arxiv.org/abs/2305.19003"],
    })


def prepare_data(root, dep):
    out = root / f"cache/masstf/{dep}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DATA_COMPLETE.json").exists(): return out
    paths = dev.TRAIN.cache_development_views(dev.OLD, dep, (2,))[2]
    all_train_x, all_train_y, all_train_group = [], [], []
    all_val_x, all_val_y, all_val_group = [], [], []
    audit = []
    for split in ("train", "validation"):
        xs, ys, groups = [], [], []
        for family in ("sis", "pm"):
            m = pd.read_parquet(dev.OLD / f"data/development/{dep}/{family}_{split}_metadata.parquet")
            for image in ("a", "b"):
                values = np.load(paths[f"{family}_{split}_{image}"], mmap_mode="r")
                xs.append(features(values))
                ys.append(target_prob(np.log(m.chirp_mass_detector.to_numpy())))
                groups.extend(m.waveform_parent_uid.astype(str).tolist())
            audit.extend({"family": family, "split": split, "parent_uid": k} for k in m.waveform_parent_uid.unique())
        x, y = np.concatenate(xs), np.concatenate(ys)
        np.save(out / f"{split}_features.npy", x)
        np.save(out / f"{split}_targets.npy", y)
        dev.csv_write(out / f"{split}_groups.csv", pd.DataFrame({"parent_uid": groups}))
        print(json.dumps({"features": split, "deployment": dep, "shape": x.shape}), flush=True)
    af = pd.DataFrame(audit)
    overlap = set(af.loc[af.split.eq("train"), "parent_uid"]) & set(af.loc[af.split.eq("validation"), "parent_uid"])
    if overlap: raise RuntimeError(f"Parent split overlap: {len(overlap)}")
    dev.csv_write(out / "parent_split_audit.csv", af)
    dev.json_write(out / "DATA_COMPLETE.json", {"parent_overlap": 0, "source_event_count": af.groupby('split').size().to_dict(), "prior": "empirical class frequencies from training labels, add 1e-4 then normalize"})
    return out


@torch.no_grad()
def infer_logits(model, x, batch=128):
    model.eval()
    ls, zs = [], []
    for i in range(0, len(x), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, z = model(torch.as_tensor(x[i:i+batch], dtype=torch.float32, device="cuda"))
        ls.append(logits.float().cpu().numpy()); zs.append(z.float().cpu().numpy())
    return np.concatenate(ls), np.concatenate(zs)


def train(root, dep, seed):
    initialize_round(root)
    out = root / f"models/MASSTF/{dep}/seed_{seed}"
    if (out / "COMPLETE.json").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    data = prepare_data(root, dep)
    train_x = np.load(data / "train_features.npy", mmap_mode="r")
    train_y = np.load(data / "train_targets.npy")
    val_x = np.load(data / "validation_features.npy", mmap_mode="r")
    val_y = np.load(data / "validation_targets.npy")
    dev.ORCH.TRAIN.seed_everything(seed)
    model = MassTF().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=3e-5)
    best = np.inf
    history = []
    for epoch in range(1, 51):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_x))
        losses = []
        start = time.perf_counter()
        for i in range(0, len(order), 96):
            ids = order[i:i+96]
            x = torch.as_tensor(np.asarray(train_x[ids], dtype=np.float32), device="cuda")
            y = torch.as_tensor(train_y[ids], device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = -(F.log_softmax(logits.float(), dim=-1) * y).sum(-1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        logits, _ = infer_logits(model, val_x)
        logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        ce = float(-(logp * val_y).sum(-1).mean())
        mean = np.exp(logp) @ LOG_CENTERS
        true = val_y @ LOG_CENTERS
        row = {"epoch": epoch, "train_ce": float(np.mean(losses)), "validation_ce": ce,
               "validation_logmc_mae": float(np.mean(np.abs(mean-true))), "seconds": time.perf_counter()-start}
        history.append(row)
        if ce < best:
            best = ce
            prior = train_y.mean(0) + 1e-4
            prior /= prior.sum()
            torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "prior": prior, "centers": LOG_CENTERS, "seed": seed,
                        "deployment": dep}, out / "validation_selected_model.pt")
        dev.csv_write(out / "training_history.csv", pd.DataFrame(history))
        print(json.dumps({"model": "MASSTF", "deployment": dep, "seed": seed, **row}), flush=True)
    payload = torch.load(out / "validation_selected_model.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(payload["state"])
    logits, emb = infer_logits(model, val_x)
    temp_rows = []
    for t in (.5, .75, 1., 1.25, 1.5, 2., 3., 4.):
        lp = logits/t - np.logaddexp.reduce(logits/t, axis=-1, keepdims=True)
        temp_rows.append({"temperature": t, "ce": float(-(val_y*lp).sum(-1).mean())})
    selected = min(temp_rows, key=lambda r: (r["ce"], abs(r["temperature"]-1)))
    dev.csv_write(out / "temperature_grid.csv", pd.DataFrame(temp_rows))
    payload["temperature"] = selected["temperature"]
    torch.save(payload, out / "validation_selected_model.pt")
    np.savez_compressed(out / "development_validation_predictions.npz", logits=logits, embedding=emb, truth=val_y)
    dev.json_write(out / "COMPLETE.json", {"selected_epoch": payload["epoch"], "temperature": payload["temperature"], "history_minimum_ce": best, "real_inputs_used": False, "model_sha256": dev.sha(out / "validation_selected_model.pt")})


@torch.no_grad()
def encode(checkpoint, full24, batch=128):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MassTF().cuda().eval()
    model.load_state_dict(payload["state"])
    ls, zs = [], []
    for i in range(0, len(full24), batch):
        # Same peak two seconds used by the frozen C-fixed waveform network.
        f = features(np.asarray(full24[i:i+batch, :, -4096:]))
        l, z = infer_logits(model, f, batch)
        ls.append(l); zs.append(z)
    l = np.concatenate(ls).astype(np.float64) / payload["temperature"]
    lp = l - np.logaddexp.reduce(l, axis=-1, keepdims=True)
    return np.exp(lp), np.concatenate(zs), payload


def mass_overlap(frame, probability, prior):
    i = frame.idx_i.to_numpy(int); j = frame.idx_j.to_numpy(int)
    val = np.sum(probability[i] * probability[j] / prior[None, :], axis=1)
    return np.log(np.maximum(val, 1e-12))


def evaluate(root, dep, model_seed, eval_seed, config="MASSTF", encoder=encode,
             model_config=None, mass_feature_builder=None, context_encoder=None):
    out = root / f"results/{config}/{dep}/model_{model_seed}_eval_{eval_seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists(): return
    checkpoint = root / f"models/{model_config or config}/{dep}/seed_{model_seed}/validation_selected_model.pt"
    evaluations = {}
    grids = []
    metrics = []
    for split in ("validation", "test"):
        ref = pd.read_parquet(dev.BAY / f"results/{dep}/seed_{eval_seed}/{split}_pair_scores_bayestar_sky.parquet")
        plan = dev.BASE.retained_event_plan(dep, eval_seed, split)
        full = dev.ORCH.event_array_for_plan(dep, eval_seed, split, plan)
        p, z, payload = (encoder(checkpoint, full) if context_encoder is None
                         else context_encoder(checkpoint, full, split))
        np.savez_compressed(out / f"{split}_mass_predictions.npz", probability=p, embedding=z, prior=payload["prior"])
        plan.to_parquet(out / f"{split}_event_plan.parquet", index=False)
        m = (mass_overlap(ref, p, payload["prior"]) if mass_feature_builder is None
             else mass_feature_builder(ref, p, payload["prior"], split, out))
        ref = ref.copy(); ref["mass_predictive_overlap"] = m
        old = ref.waveform_score.to_numpy()
        evaluations[split] = ref
        if split == "validation":
            reference = dev.BASE.full_metrics(ref, old)
            selected = {}
            for method in ("negative_guard", "mass_cap"):
                options = []
                for gamma in (0., .25, .5, 1., 2., 4.):
                    score = old + gamma * np.minimum(m, 0) if method == "negative_guard" else (np.minimum(old, gamma*m) if gamma else old)
                    met = dev.BASE.full_metrics(ref, score)
                    noninferior = met["macro_r_at_10"] >= reference["macro_r_at_10"]-.02 and met["average_precision"] >= reference["average_precision"]-.005
                    row = {"method": method, "gamma": gamma, "noninferior": noninferior, **met}
                    grids.append(row)
                    if noninferior:
                        key = (met["false_at_recall_0p5"], met["false_at_recall_0p9"], -met["average_precision"], -met["macro_r_at_10"], gamma)
                        options.append((key, gamma))
                selected[method] = min(options)[1]
            dev.json_write(out / "selected_validation_waveform_rules.json", selected)
            dev.csv_write(out / "validation_guard_grid.csv", pd.DataFrame(grids))
        for method in ("baseline", "mass_only", "negative_guard", "mass_cap"):
            gamma = selected.get(method, 0.)
            if method == "baseline": score = old
            elif method == "mass_only": score = m
            elif method == "negative_guard": score = old + gamma*np.minimum(m, 0)
            else: score = np.minimum(old, gamma*m) if gamma else old
            frame = ref.copy(); frame["waveform_score"] = score
            frame.to_parquet(out / f"{split}_{method}_pairs.parquet", index=False)
            for rankmethod in ("waveform_only", "C_fixed"):
                s = score if rankmethod == "waveform_only" else dev.BASE.score_vector(frame, dev.BASE.FROZEN_V93_WEIGHTS[dep][eval_seed])
                metrics.append({"config": config, "deployment": dep, "model_seed": model_seed, "eval_seed": eval_seed,
                                "split": split, "waveform_rule": method, "gamma": gamma, "method": rankmethod, **dev.BASE.full_metrics(frame, s)})
    real = dev.real_frame(dep, eval_seed)
    full, events = dev.real_inputs(dep)
    # Invalid channels never enter the network. Maps and time scores stay fixed.
    valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
    p, z, payload = (encoder(checkpoint, np.asarray(full[valid])) if context_encoder is None
                     else context_encoder(checkpoint, np.asarray(full[valid]), "real"))
    allp = np.broadcast_to(payload["prior"], (len(full), 64)).copy()
    allp[valid] = p
    m = (mass_overlap(real, allp, payload["prior"]) if mass_feature_builder is None
         else mass_feature_builder(real, allp, payload["prior"], "real", out))
    np.savez_compressed(out / "real_mass_predictions.npz", probability=allp, valid=valid, prior=payload["prior"], centers=LOG_CENTERS)
    ev = events[["idx", "event_name", "strict_h1l1_preprocessing_pass"]].copy()
    ev["pred_logmc"] = allp @ LOG_CENTERS
    ev["pred_logmc_std"] = np.sqrt(np.maximum(allp @ LOG_CENTERS**2 - ev.pred_logmc.to_numpy()**2, 0))
    ev["pred_mc"] = np.exp(ev.pred_logmc)
    dev.csv_write(out / "real_event_mass_predictions.csv", ev)
    pe = dev.pe_audit(root, dep)
    budgets = []
    for method in ("mass_only", "negative_guard", "mass_cap"):
        frame = real.copy()
        old = real.waveform_score.to_numpy()
        gamma = selected.get(method, 0.)
        if method == "mass_only": score = m
        elif method == "negative_guard": score = old + gamma*np.minimum(m, 0)
        else: score = np.minimum(old, gamma*m) if gamma else old
        frame["waveform_score"] = score
        frame["mass_predictive_overlap"] = m
        assert np.array_equal(frame.time_score, real.time_score) and np.array_equal(frame.sky_raw_log_bf, real.sky_raw_log_bf)
        key = f"{config}-{method}-model{model_seed}"
        budgets.extend(dev.save_evaluation(root, key, dep, {eval_seed: frame}, pe))
    dev.csv_write(out / "retrieval_pair_metrics.csv", pd.DataFrame(metrics))
    dev.csv_write(out / "pe_official_budget.csv", pd.DataFrame(budgets))
    dev.json_write(out / "COMPLETE.json", {"selected": selected, "time_sky_unchanged": True, "real_inputs_in_training": False})
    print(json.dumps({"mass_tf_eval": dep, "seed": model_seed, "selected": selected, "top10": [b for b in budgets if b['budget']==10 and b['seed']=='consensus']}), flush=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", type=Path, required=True)
    pa.add_argument("--deployment", choices=("gwtc3", "gwtc4"), required=True)
    pa.add_argument("--seed", type=int, default=SEEDS[0])
    pa.add_argument("--eval-seed", type=int, default=dev.SEEDS[0])
    pa.add_argument("--phase", choices=("train", "evaluate"), required=True)
    a = pa.parse_args()
    if a.phase == "train": train(a.root, a.deployment, a.seed)
    else: evaluate(a.root, a.deployment, a.seed, a.eval_seed)


if __name__ == "__main__": main()
