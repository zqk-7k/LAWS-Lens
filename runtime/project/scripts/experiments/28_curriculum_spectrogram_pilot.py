#!/usr/bin/env python3
"""Clean-pretrain then real-noise adaptation pilot for full 24-s retrieval."""

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
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import EvaluationSet, MatchArrays, ground_truth_partner, load_match_arrays, peak_flip_channels, split_indices, zscore_channels
from matchgw.matching import similarity_matrix
from matchgw.models import NTXentLoss
from scripts.experiments.mainline_uncertainty_common import seed_everything
from scripts.real_search.physical_common import effective_rank, write_json


def load_encoder():
    spec = importlib.util.spec_from_file_location("spectrogram_pilot", REPO / "scripts" / "experiments" / "26_spectrogram_encoder_pilot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Full24SpectrogramEncoder


def prepare(x: np.ndarray, rng: np.random.Generator, train: bool) -> np.ndarray:
    y = peak_flip_channels(np.asarray(x, dtype=np.float32).copy())
    if train:
        y = np.roll(y, int(rng.integers(-128, 129)), axis=-1)
        y *= float(1.0 + rng.uniform(-0.1, 0.1))
        y += rng.normal(0.0, 0.003 * (y.std(axis=-1, keepdims=True) + 1e-8), size=y.shape)
    return zscore_channels(y)


class CleanPairDataset(Dataset):
    def __init__(self, arrays, indices: np.ndarray, seed: int) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return (
            torch.from_numpy(prepare(self.arrays.l1_pure[idx], self.rng, True)),
            torch.from_numpy(prepare(self.arrays.l2_pure[idx], self.rng, True)),
        )


class AdaptationDataset(Dataset):
    def __init__(self, arrays, indices: np.ndarray, seed: int) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        views = (self.arrays.l1[idx], self.arrays.l2[idx], self.arrays.l1_pure[idx], self.arrays.l2_pure[idx])
        return tuple(torch.from_numpy(prepare(view, self.rng, True)) for view in views)


@torch.no_grad()
def evaluate(model, dataset: EvaluationSet, batch_size: int) -> dict:
    model.eval()
    rows = []
    for x in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True):
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
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--adapt-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    v3_spec = importlib.util.spec_from_file_location("v3", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
    v3 = importlib.util.module_from_spec(v3_spec)
    assert v3_spec.loader is not None
    v3_spec.loader.exec_module(v3)
    cfg = v3.training_config(args.seed_root, args.family, args.seed, args.samples, args.adapt_epochs, args.batch_size)
    cfg.use_pure_aux = True
    arrays = load_match_arrays(cfg)
    parts = split_indices(args.samples, args.samples, cfg)
    if arrays.l1_pure is None or arrays.l2_pure is None or arrays.unlensed_pure is None:
        raise RuntimeError("Physical clean arrays are missing")
    pure_arrays = MatchArrays(arrays.l1_pure, arrays.l2_pure, arrays.unlensed_pure)
    clean_train = CleanPairDataset(arrays, parts["lensed"]["train"], cfg.seed + 2800)
    adapt_train = AdaptationDataset(arrays, parts["lensed"]["train"], cfg.seed + 2801)
    pure_val = EvaluationSet(pure_arrays, parts["lensed"]["val"], parts["unlensed"]["val"], cfg)
    noisy_val = EvaluationSet(arrays, parts["lensed"]["val"], parts["unlensed"]["val"], cfg)

    Encoder = load_encoder()
    seed_everything(cfg.seed + 2800)
    model = Encoder(base_channels=24).cuda()
    contrastive = NTXentLoss(tau=0.10)
    history = []

    clean_loader = DataLoader(clean_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    best_clean_key = None
    best_clean_state = None
    for epoch in range(1, args.pretrain_epochs + 1):
        start = time.perf_counter(); model.train(); losses = []
        for xa, xb in clean_loader:
            xa, xb = xa.cuda(non_blocking=True), xb.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = contrastive(model(xa), model(xb))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"phase": "clean_pretrain", "epoch": epoch, "loss": float(np.mean(losses)), "epoch_s": float(time.perf_counter()-start)}
        if epoch == 1 or epoch % 2 == 0 or epoch == args.pretrain_epochs:
            clean_metrics = evaluate(model, pure_val, args.batch_size)
            noisy_metrics = evaluate(model, noisy_val, args.batch_size)
            row.update({f"pure_val_{k}": v for k,v in clean_metrics.items()})
            row.update({f"noisy_val_{k}": v for k,v in noisy_metrics.items()})
            key = (clean_metrics["r_at_10"], clean_metrics["r_at_1"], -clean_metrics["median_rank"])
            if best_clean_key is None or key > best_clean_key:
                best_clean_key = key
                best_clean_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row); print(json.dumps(row), flush=True)

    model.load_state_dict(best_clean_state)
    adapt_loader = DataLoader(adapt_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    best_noisy_key = None
    best_noisy_state = None
    best_noisy_metrics = None
    for epoch in range(1, args.adapt_epochs + 1):
        start = time.perf_counter(); model.train(); losses=[]
        for noisy_a, noisy_b, clean_a, clean_b in adapt_loader:
            noisy_a=noisy_a.cuda(non_blocking=True); noisy_b=noisy_b.cuda(non_blocking=True)
            clean_a=clean_a.cuda(non_blocking=True); clean_b=clean_b.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_na=model(noisy_a); z_nb=model(noisy_b); z_ca=model(clean_a); z_cb=model(clean_b)
                loss_noise=contrastive(z_na,z_nb)
                loss_clean=contrastive(z_ca,z_cb)
                loss_align=0.5*((1-(z_na*z_ca).sum(-1)).mean()+(1-(z_nb*z_cb).sum(-1)).mean())
                loss=loss_noise+0.25*loss_clean+0.75*loss_align
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            losses.append((float(loss.detach().cpu()),float(loss_noise.detach().cpu()),float(loss_clean.detach().cpu()),float(loss_align.detach().cpu())))
        values=np.asarray(losses)
        row={"phase":"real_noise_adaptation","epoch":epoch,"loss":float(values[:,0].mean()),"loss_noise":float(values[:,1].mean()),"loss_clean":float(values[:,2].mean()),"loss_align":float(values[:,3].mean()),"epoch_s":float(time.perf_counter()-start)}
        if epoch == 1 or epoch % 2 == 0 or epoch == args.adapt_epochs:
            current=evaluate(model,noisy_val,args.batch_size)
            row.update({f"noisy_val_{k}":v for k,v in current.items()})
            key=(current["r_at_10"],current["r_at_1"],-current["median_rank"],current["embedding_effective_rank"])
            if best_noisy_key is None or key>best_noisy_key:
                best_noisy_key=key;best_noisy_metrics=current;best_noisy_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row);print(json.dumps(row),flush=True)

    output=args.seed_root/'waveform_gate'/f'{args.family.lower()}_full24_curriculum_spectrogram_pilot';output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'validation_selected_model.pt';torch.save({"model_state":best_noisy_state,"family":args.family,"architecture":"Full24SpectrogramEncoder","objective":"clean pretrain then real-noise adaptation"},checkpoint)
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    summary={"family":args.family,"selection_split":"validation only","heldout_test_was_evaluated":False,"pretrain":"clean A/B NT-Xent","adaptation":"noisy A/B NT-Xent + clean preservation + paired noisy-clean alignment","best_noisy_validation":best_noisy_metrics,"checkpoint":str(checkpoint)}
    write_json(output/'curriculum_pilot_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
