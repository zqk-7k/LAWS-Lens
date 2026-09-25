#!/usr/bin/env python3
"""Validation-selected GroupNorm repair pilot for the full-24-s encoder.

BatchNorm can exploit paired mini-batch statistics in Siamese contrastive
training. GroupNorm makes each event's representation independent of the other
events in its batch. Variants are selected on validation retrieval only; the
held-out test split is evaluated once for the selected variant.
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
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import EvaluationSet, PairDataset, ground_truth_partner, load_match_arrays, prepared_channel_count, split_indices
from matchgw.matching import similarity_matrix
from matchgw.models import InceptionAttentionEncoder1D, InceptionTimeEncoder1D, NTXentLoss
from matchgw.pipeline import embed_eval
from scripts.experiments.mainline_uncertainty_common import seed_everything
from scripts.real_search.physical_common import effective_rank, write_json


def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def replace_batchnorm(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm1d):
            setattr(module, name, torch.nn.GroupNorm(group_count(child.num_features), child.num_features))
            replaced += 1
        else:
            replaced += replace_batchnorm(child)
    return replaced


def metrics(model: torch.nn.Module, dataset: EvaluationSet, cfg, split: str) -> dict:
    z = embed_eval(model, dataset, cfg, cpu=False)
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
        "n_events": len(dataset),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "embedding_effective_rank": effective_rank(z),
        "cosine_median": float(np.median(upper)),
        "cosine_std": float(np.std(upper)),
    }


def make_model(cfg, arrays, backbone: str) -> torch.nn.Module:
    model_class = InceptionAttentionEncoder1D if backbone == "inceptionattention" else InceptionTimeEncoder1D
    return model_class(
        in_channels=prepared_channel_count(arrays.l1[0], cfg),
        d_model=cfg.d_model,
        emb_dim=cfg.emb_dim,
        width_scale=cfg.width_scale,
    )


def train_variant(
    cfg,
    arrays,
    parts,
    out: Path,
    use_pure_aux: bool,
    epochs: int,
    backbone: str,
    width_scale: float,
    d_model: int,
    emb_dim: int,
) -> dict:
    seed_everything(cfg.seed + (100 if use_pure_aux else 0))
    cfg = replace(
        cfg,
        use_pure_aux=use_pure_aux,
        epochs=epochs,
        width_scale=width_scale,
        d_model=d_model,
        emb_dim=emb_dim,
    )
    # Only independent-noise lensed A/B views define positives. Unlensed events
    # remain in validation/test catalogs, but pairing an unlensed noisy event
    # with itself here creates a noise-fingerprint shortcut.
    train = PairDataset(arrays, parts["lensed"]["train"], np.asarray([], dtype=np.int64), cfg)
    val = EvaluationSet(arrays, parts["lensed"]["val"], parts["unlensed"]["val"], cfg)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    model = make_model(cfg, arrays, backbone)
    replaced_layers = replace_batchnorm(model)
    model = model.cuda()
    loss_fn = NTXentLoss(cfg.tau)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    best_key = None
    best_state = None
    best_metrics = None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for xa, xb in loader:
            optimizer.zero_grad(set_to_none=True)
            xa, xb = xa.cuda(non_blocking=True), xb.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xa), model(xb))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "epoch_s": float(time.perf_counter() - started),
        }
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            val_metrics = metrics(model, val, cfg, "validation")
            row.update({f"val_{key}": value for key, value in val_metrics.items() if key != "split"})
            key = (
                val_metrics["r_at_10"],
                val_metrics["r_at_1"],
                -val_metrics["median_rank"],
                val_metrics["embedding_effective_rank"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = val_metrics
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        print(json.dumps({"variant": "lensed_only_pure_aux" if use_pure_aux else "lensed_only_no_pure_aux", **row}), flush=True)
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state}, out / "validation_selected_model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    return {
        "variant": "lensed_only_pure_aux" if use_pure_aux else "lensed_only_no_pure_aux",
        "use_pure_aux": use_pure_aux,
        "unlensed_self_positive_pairs": False,
        "negative_definition": "other lensed source systems in the same contrastive batch",
        "normalization": "GroupNorm",
        "backbone": backbone,
        "batchnorm_layers_replaced": replaced_layers,
        "best_validation": best_metrics,
        "checkpoint": str(out / "validation_selected_model.pt"),
        "state": best_state,
        "cfg": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--family", choices=("SIS", "PM"), required=True)
    parser.add_argument("--seed", type=int, default=202607211)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone", choices=("inceptiontime", "inceptionattention"), default="inceptiontime")
    parser.add_argument("--width-scale", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument(
        "--variants",
        choices=("no_aux", "aux", "both"),
        default="both",
        help="Architecture pilots can avoid unnecessary auxiliary runs.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not inspect held-out test while selecting an architecture.",
    )
    args = parser.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location("v3", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cfg = module.training_config(args.seed_root, args.family, args.seed, args.samples, args.epochs, args.batch_size)
    arrays = load_match_arrays(cfg)
    parts = split_indices(args.samples, args.samples, cfg)
    pilot_root = args.seed_root / "waveform_gate" / f"{args.family.lower()}_groupnorm_{args.backbone}_pilot"
    runs = []
    use_aux_values = {"no_aux": (False,), "aux": (True,), "both": (False, True)}[args.variants]
    for use_aux in use_aux_values:
        runs.append(
            train_variant(
                cfg,
                arrays,
                parts,
                pilot_root / ("lensed_only_pure_aux" if use_aux else "lensed_only_no_pure_aux"),
                use_aux,
                args.epochs,
                args.backbone,
                args.width_scale,
                args.d_model,
                args.emb_dim,
            )
        )
    selected = max(
        runs,
        key=lambda item: (
            item["best_validation"]["r_at_10"],
            item["best_validation"]["r_at_1"],
            -item["best_validation"]["median_rank"],
            item["best_validation"]["embedding_effective_rank"],
        ),
    )
    test_metrics = None
    if not args.skip_test:
        selected_model = make_model(selected["cfg"], arrays, args.backbone)
        replace_batchnorm(selected_model)
        selected_model = selected_model.cuda()
        selected_model.load_state_dict(selected["state"])
        test = EvaluationSet(arrays, parts["lensed"]["test"], parts["unlensed"]["test"], selected["cfg"])
        test_metrics = metrics(selected_model, test, selected["cfg"], "held-out test")
    final_path = pilot_root / "selected_groupnorm_model.pt"
    torch.save(
        {
            "model_state": selected["state"],
            "family": args.family,
            "variant": selected["variant"],
            "normalization": "GroupNorm",
            "backbone": args.backbone,
        },
        final_path,
    )
    summary = {
        "family": args.family,
        "backbone": args.backbone,
        "selection_split": "validation only",
        "test_evaluations_before_variant_selection": 0,
        "heldout_test_was_evaluated": not args.skip_test,
        "variants": [
            {key: value for key, value in item.items() if key not in {"state", "cfg"}} for item in runs
        ],
        "selected_variant": selected["variant"],
        "selected_validation": selected["best_validation"],
        "heldout_test": test_metrics,
        "selected_checkpoint": str(final_path),
    }
    write_json(pilot_root / "groupnorm_pilot_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
