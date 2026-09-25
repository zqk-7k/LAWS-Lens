#!/usr/bin/env python3
"""Train-only noisy-domain BatchNorm recalibration diagnostic.

This does not optimize model weights and does not inspect validation/test labels
while recalibrating. It tests whether train/eval BatchNorm state mismatch causes
the v3 full-24-s encoder collapse.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from matchgw.pipeline import embed_eval
from scripts.real_search.physical_common import effective_rank, write_json


def load_v3_module():
    path = REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py"
    spec = importlib.util.spec_from_file_location("real_noise_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recalibrate_batchnorm(model: torch.nn.Module, dataset: EvaluationSet, batch_size: int) -> int:
    modules = [item for item in model.modules() if isinstance(item, torch.nn.BatchNorm1d)]
    for item in modules:
        item.reset_running_stats()
        item.momentum = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.train()
    device = next(model.parameters()).device
    with torch.no_grad():
        for values in loader:
            model(values.to(device, non_blocking=True))
    model.eval()
    return len(modules)


def retrieval(model: torch.nn.Module, dataset: EvaluationSet, cfg, split: str) -> dict:
    embeddings = embed_eval(model, dataset, cfg, cpu=False)
    scores = similarity_matrix(embeddings)
    truth = ground_truth_partner(dataset.meta)
    ranks = []
    for query, partner in enumerate(truth):
        if partner >= 0:
            ranks.append(1 + int(np.sum(scores[query] > scores[query, partner])))
    ranks = np.asarray(ranks, dtype=np.int32)
    upper = scores[np.triu_indices(len(scores), k=1)]
    return {
        "split": split,
        "n_events": len(dataset),
        "n_directed_queries": len(ranks),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "embedding_effective_rank": effective_rank(embeddings),
        "cosine_median": float(np.median(upper)),
        "cosine_std": float(np.std(upper)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("SIS", "PM"), default="SIS")
    parser.add_argument("--seed", type=int, default=202607211)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    v3 = load_v3_module()
    cfg = v3.training_config(args.root, args.family, args.seed, args.samples, args.epochs, args.batch_size)
    arrays = load_match_arrays(cfg)
    parts = split_indices(args.samples, args.samples, cfg)
    checkpoint = args.root / "waveform_gate" / f"{args.family.lower()}_physical_full24_ep{args.epochs}" / "model.pt"
    model, eval_cfg = v3.load_checkpoint_model(checkpoint, cfg.data_root, args.samples, args.family)
    train_noisy = EvaluationSet(arrays, parts["lensed"]["train"], parts["unlensed"]["train"], eval_cfg)
    bn_count = recalibrate_batchnorm(model, train_noisy, args.batch_size)

    rows = []
    for split in ("train", "val", "test"):
        dataset = EvaluationSet(arrays, parts["lensed"][split], parts["unlensed"][split], eval_cfg)
        rows.append(retrieval(model, dataset, eval_cfg, split))
    out = args.root / "results" / "batchnorm_recalibration_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.family.lower()}_retrieval_after_train_noisy_bn_recalibration.csv", index=False)
    summary = {
        "family": args.family,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "batchnorm_layers_recalibrated": bn_count,
        "calibration_data": "training split noisy lensed-image and unlensed events only",
        "validation_or_test_labels_used_for_recalibration": False,
        "model_weights_updated": False,
        "metrics": rows,
    }
    write_json(out / f"{args.family.lower()}_bn_recalibration_summary.json", summary)
    torch.save({"model_state": model.state_dict()}, out / f"{args.family.lower()}_model_bn_recalibrated.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
