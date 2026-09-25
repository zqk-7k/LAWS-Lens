#!/usr/bin/env python3
"""Executable final smoke checks and independent metric-definition audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_fresh_confirmation_20260906 as fresh


def run(root):
    torch.manual_seed(21931)
    z = torch.randn(12, 128, requires_grad=True)
    group = torch.arange(6).repeat(2)
    value = body.source_contrastive(z, group)
    order = torch.randperm(12)
    shuffled = body.source_contrastive(z[order], group[order])
    assert torch.allclose(value, shuffled, atol=1e-6)
    value.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    no_positive = body.source_contrastive(z, torch.arange(12))
    assert no_positive == 0
    rng = np.random.default_rng(917)
    p = rng.dirichlet(np.ones(64), size=3)
    z = rng.normal(size=(3, 128))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    f = pd.DataFrame({"idx_i": [0, 1], "idx_j": [1, 0]})
    feat = ev.features(f, p, z)
    for col in ("cosine", "mass", "predictive_BC"):
        assert np.isclose(feat[col][0], feat[col][1], atol=1e-12)
    checks, tieaudit = [], []
    directory = fresh.data_root(root)
    for path in sorted(directory.glob("*/catalog_*/model_*/*_pairs.parquet")):
        frame = pd.read_parquet(path)
        y = frame.is_true_pair.to_numpy(bool)
        dep = path.parents[2].name
        modelseed = int(path.parent.name.split("_")[1])
        es = dev.SEEDS[body.MODEL_SEEDS.index(modelseed)]
        for method in ("waveform_only", "C_fixed"):
            weights = {"waveform": 1., "time": 0., "sky": 0.} if method == "waveform_only" else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
            s = dev.BASE.score_vector(frame, weights)
            m = dev.BASE.full_metrics(frame, s)
            assert abs(m["average_precision"]-average_precision_score(y, s)) < 1e-12
            # This is a separate inclusive-threshold audit; historical metric
            # definitions and ranking remain unchanged.
            positive = np.sort(s[y])[::-1]
            row = {"path": str(path.relative_to(root)), "method": method, "n_false": int((~y).sum()),
                   "minimum_nonzero_empirical_FPR": 1/int((~y).sum())}
            for r in (.5, .9):
                threshold = positive[int(np.ceil(r*len(positive)))-1]
                row[f"F{int(r*100)}_inclusive_ties"] = int(((s >= threshold) & ~y).sum())
                row[f"F{int(r*100)}_frozen_stable_order"] = m[f"false_at_recall_{str(r).replace('.', 'p')}"]
            threshold = np.nextafter(s[~y].max(), np.inf)
            row["threshold_for_empirical_FPR_at_most_1e_minus5"] = threshold
            row["FP_at_that_threshold"] = int(((s >= threshold) & ~y).sum())
            row["TP_at_that_threshold"] = int(((s >= threshold) & y).sum())
            tieaudit.append(row)
        checks.append({"path": str(path.relative_to(root)), "pairs": len(frame), "true_pairs": int(y.sum()),
                       "all_finite_scores": bool(np.isfinite(frame[["waveform_score", "time_score", "sky_raw_log_bf"]]).all().all())})
    assert len(checks) == 36
    assert all(x["all_finite_scores"] and x["true_pairs"] == 70 and x["pairs"] == 17955 for x in checks)
    trainings = list((root / "models").glob("*/*/seed_*/training_history.csv"))
    assert len(trainings) == 21 and all(len(pd.read_csv(p)) == 50 for p in trainings)
    assert len(list((root / "evaluation").glob("*/*/model_*/COMPLETE.json"))) == 21
    fresh.verify_freeze(root)
    dev.csv_write(directory / "TIES_AND_EMPIRICAL_FPR_AUDIT.csv", pd.DataFrame(tieaudit))
    dev.json_write(root / "audit/FINAL_EXECUTABLE_CHECKS.json", {
        "source_contrastive_permutation_invariant": True, "finite_nonzero_gradient": True,
        "no_positive_batch_zero_loss": True, "waveform_features_symmetric": True,
        "training_models": 21, "epochs_each": 50, "complete_evaluations": 21,
        "fresh_pair_tables": len(checks), "AP_independent_sklearn_recompute": "exact",
        "candidate_freeze_hashes_unchanged": True, "tables": checks})
    print("PASS: contrastive, symmetry, finite data, 21 models, 36 fresh pair tables, AP and freeze checks", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
