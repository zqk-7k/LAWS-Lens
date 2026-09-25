#!/usr/bin/env python3
"""Pair-aligned comparisons, uncertainty, and exact frozen-channel checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mcwf_development_20260905 as dev
import mcwf_summarize_20260905 as previous
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as evaluate


def audit(root, pair_bootstrap=False):
    checks, rows, coverage = [], [], []
    for config in body.CONFIGS:
        for dep in ("gwtc3", "gwtc4"):
            for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
                directory = root / f"evaluation/{config}/{dep}/model_{ms}_eval_{es}"
                if not (directory / "COMPLETE.json").exists():
                    continue
                base = pd.read_parquet(dev.BAY / f"results/{dep}/seed_{es}/test_pair_scores_bayestar_sky.parquet")
                plan = dev.BASE.retained_event_plan(dep, es, "test")
                for rule in evaluate.RULES:
                    frame = pd.read_parquet(directory / f"test_{rule}_pairs.parquet")
                    for column in ("idx_i", "idx_j", "is_true_pair", "time_score", "sky_raw_log_bf"):
                        if not np.array_equal(frame[column].to_numpy(), base[column].to_numpy()):
                            raise AssertionError(f"Frozen input changed: {config}/{dep}/{ms}/{rule}/{column}")
                    checks.append({"config": config, "deployment": dep, "seed": ms, "rule": rule,
                                   "time_sky_truth_order_exact": True, "n_pairs": len(frame)})
                    for method in ("waveform_only", "C_fixed"):
                        weights = {"waveform": 1., "time": 0., "sky": 0.} if method == "waveform_only" else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                        s = dev.BASE.score_vector(frame, weights)
                        b = dev.BASE.score_vector(base, weights)
                        m = dev.BASE.full_metrics(frame, s)
                        bm = dev.BASE.full_metrics(base, b)
                        ci, ranks = previous.ranks_bootstrap(frame, s, b, es)
                        row = {"config": config, "deployment": dep, "model_seed": ms, "eval_seed": es,
                               "rule": rule, "method": method, **m, **ci}
                        for key in ("macro_r_at_1", "macro_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"):
                            row["baseline_" + key] = bm[key]
                            row["delta_" + key] = m[key] - bm[key]
                        row["point_guardrail_pass"] = (m["macro_r_at_10"] >= bm["macro_r_at_10"] - .02
                            and m["average_precision"] >= bm["average_precision"] - .005
                            and m["false_at_recall_0p5"] <= bm["false_at_recall_0p5"] * 1.10
                            and m["false_at_recall_0p9"] <= bm["false_at_recall_0p9"] * 1.10)
                        if pair_bootstrap and rule == "VAL-ENSEMBLE":
                            row.update(previous.pair_bootstrap(frame, s, b, plan, es, repeats=2000))
                        rows.append(row)
                        out = root / f"audit/query_ranks/{config}/{dep}/{ms}"
                        out.mkdir(parents=True, exist_ok=True)
                        ranks.to_parquet(out / f"{rule}_{method}.parquet", index=False)
                model = root / f"models/{config}/{dep}/seed_{ms}"
                stored = np.load(model / "development_validation.npz")
                ck = torch.load(model / "validation_selected_model.pt", map_location="cpu", weights_only=False)
                lp = stored["logits"].astype(float) / ck["temperature"]
                lp -= np.logaddexp.reduce(lp, -1, keepdims=True)
                p = np.exp(lp)
                truth = stored["truth"] @ body.tf.LOG_CENTERS
                pos = np.clip((truth - body.tf.LOG_EDGES[0]) / np.diff(body.tf.LOG_EDGES)[0], 0, 64 - 1e-9)
                bi = pos.astype(int)
                cdf = np.c_[np.zeros(len(p)), np.cumsum(p, -1)]
                pit = cdf[np.arange(len(p)), bi] + (pos - bi) * p[np.arange(len(p)), bi]
                singular = np.maximum(np.linalg.eigvalsh(np.cov(stored["embedding"], rowvar=False)), 0)
                fractions = singular[singular > 0] / singular.sum()
                coverage.append({"config": config, "deployment": dep, "seed": ms,
                    "selected_epoch": ck["epoch"], "temperature": ck["temperature"],
                    "validation_logMc_MAE": float(np.mean(np.abs(p @ body.tf.LOG_CENTERS - truth))),
                    "central50_coverage": float(np.mean((pit >= .25) & (pit <= .75))),
                    "central90_coverage": float(np.mean((pit >= .05) & (pit <= .95))),
                    "effective_rank": float(np.exp(-np.sum(fractions * np.log(fractions)))),
                    "interpretation": "development calibration diagnostic, not independent PE coverage"})
    if rows:
        frame = pd.DataFrame(rows)
        dev.csv_write(root / "tables/PAIRED_GUARDRAILS_AND_CI.csv", frame)
        summary = frame.groupby(["config", "deployment", "rule", "method"]).agg(
            tested_seeds=("model_seed", "count"), passing_seeds=("point_guardrail_pass", "sum"),
            delta_R10=("delta_macro_r_at_10", "mean"), delta_AP=("delta_average_precision", "mean"),
            delta_F50=("delta_false_at_recall_0p5", "mean"), delta_F90=("delta_false_at_recall_0p9", "mean")).reset_index()
        dev.csv_write(root / "tables/GUARDRAIL_SUMMARY.csv", summary)
        print(summary.to_json(orient="records"), flush=True)
    dev.csv_write(root / "audit/FROZEN_CHANNELS_EXACT.csv", pd.DataFrame(checks))
    dev.csv_write(root / "tables/ENCODER_COVERAGE_AND_EFFECTIVE_RANK.csv", pd.DataFrame(coverage))
    manifest = pd.read_csv(root / "manifest/PROTECTED_INPUT_SHA256.csv")
    mismatches = [row.path for row in manifest.itertuples() if dev.sha(Path(row.path)) != row.sha256]
    dev.json_write(root / "audit/PROTECTED_INPUTS_AFTER.json", {"checked": len(manifest), "mismatches": mismatches})
    if mismatches:
        raise AssertionError("Historical input hashes changed")
    dev.json_write(root / "contracts/UNCERTAINTY_INTERPRETATION.json", {
        "retrieval": "10000 paired lens-system bootstrap replicates, two directed queries kept together, family strata retained",
        "pairs": "2000 weighted source-block replicates for ensemble; null edge weight m_i*m_j, within-system true edge m_i",
        "limitations": "fixed noise/sky realization uncertainty, reused development catalogs; no iid-pair or independent population significance claim",
        "seed_SD": "combines new initialization, original encoder deployment and catalog realization; not pure initialization variance",
        "goal_achieved": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pair-bootstrap", action="store_true")
    a = p.parse_args()
    audit(a.root, a.pair_bootstrap)
