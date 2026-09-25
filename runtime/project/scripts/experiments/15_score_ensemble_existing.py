from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import evaluate_scores, retrieval_metrics, similarity_matrix, tune_matching
from matchgw.pipeline import build_model, default_tuning_grid, embed_eval

RUNS = {
    "SIS": [
        ("inceptiontime", Path("runs/et10000_bandpass_full_ep50_20260528_101013/SIS_noisy_bandpass_n10000_ep50")),
        ("inceptionattn_lr5e4", Path("runs/et10000_inceptionattn_lr5e4_full_ep50_20260528_162132/SIS_noisy_inceptionattn_lr5e4_bandpass_n10000_ep50")),
        ("gatedtcn", Path("runs/et10000_gatedtcn_bandpass_full_ep50_20260528_140806/SIS_noisy_gatedtcn_bandpass_n10000_ep50")),
    ],
    "PM": [
        ("inceptiontime", Path("runs/et10000_bandpass_full_ep50_20260528_101013/PM_noisy_bandpass_n10000_ep50")),
        ("inceptionattn_lr5e4", Path("runs/et10000_inceptionattn_lr5e4_full_ep50_20260528_162132/PM_noisy_inceptionattn_lr5e4_bandpass_n10000_ep50")),
        ("gatedtcn", Path("runs/et10000_gatedtcn_bandpass_full_ep50_20260528_140806/PM_noisy_gatedtcn_bandpass_n10000_ep50")),
    ],
}


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / "summary.json").read_text())["config"]
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for k in ["data_root", "out_dir"]:
        if k in kwargs:
            kwargs[k] = Path(kwargs[k])
    kwargs["out_dir"] = out_dir
    return MatchRunConfig(**kwargs)


def combine(scores: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    out = np.zeros_like(scores[0], dtype=np.float32)
    for s, w in zip(scores, weights):
        ss = s.astype(np.float32, copy=True)
        np.fill_diagonal(ss, 0.0)
        out += float(w) * ss
    np.fill_diagonal(out, -np.inf)
    return out


def weight_grid(n: int, step: float = 0.05):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for a in vals:
        for b in vals:
            c = 1.0 - a - b
            if c >= -1e-9:
                yield (round(float(a), 4), round(float(b), 4), round(float(max(c, 0.0)), 4))


def run_family(family: str, out_root: Path) -> dict:
    out_dir = out_root / family
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [x[0] for x in RUNS[family]]
    base_cfg = cfg_from_run(RUNS[family][0][1], out_dir)
    arrays = load_match_arrays(base_cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), base_cfg)
    val_ds = EvaluationSet(arrays, splits["lensed"]["val"], splits["unlensed"]["val"], base_cfg)
    test_ds = EvaluationSet(arrays, splits["lensed"]["test"], splits["unlensed"]["test"], base_cfg)
    val_gt = ground_truth_partner(val_ds.meta)
    test_gt = ground_truth_partner(test_ds.meta)
    val_scores=[]; test_scores=[]
    for label, run_dir in RUNS[family]:
        cfg = cfg_from_run(run_dir, out_dir / label)
        model = build_model(cfg)
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        val_scores.append(similarity_matrix(embed_eval(model, val_ds, cfg, cpu=False)))
        test_scores.append(similarity_matrix(embed_eval(model, test_ds, cfg, cpu=False)))
        print(f"loaded {family} {label}", flush=True)
    rows=[]; best=None
    for weights in weight_grid(len(labels), step=0.05):
        vs = combine(val_scores, weights)
        r = retrieval_metrics(vs, val_gt)
        row={"family":family,"weights":json.dumps(dict(zip(labels, weights))), **r}
        rows.append(row)
        key=(r["r@1"], r["r@10"], r["mrr"])
        if best is None or key > best[0]:
            best=(key, weights, r)
    weights=best[1]
    val_comb=combine(val_scores, weights)
    test_comb=combine(test_scores, weights)
    best_params, val_pair = tune_matching(val_comb, val_gt, default_tuning_grid(base_cfg), metric="f1")
    test_stats = evaluate_scores(test_comb, test_gt, **best_params)
    pd.DataFrame(rows).to_csv(out_dir / "weight_grid_val.csv", index=False)
    result={"family":family,"labels":labels,"weights":dict(zip(labels, weights)),"val_retrieval":best[2],"best_params":best_params,"val_pair":val_pair,"test":test_stats}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    out_root=Path("runs/et10000_score_ensemble_existing")
    out_root.mkdir(parents=True, exist_ok=True)
    results=[run_family(f, out_root) for f in ["SIS","PM"]]
    (out_root / "summary.json").write_text(json.dumps(results, indent=2))
    for r in results:
        t=r["test"]
        print(r["family"], r["weights"], "r1", round(t.get("r@1",0),4), "r10", round(t.get("r@10",0),4), "pairf1", round(t.get("f1",0),4), "pairs", t.get("pairs"))

if __name__ == "__main__":
    main()
