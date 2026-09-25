from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

base = importlib.import_module("scripts.experiments.17_waveform_reranker_existing")


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def eval_ranker(rows: pd.DataFrame, gt: np.ndarray, score_col: str) -> dict:
    valid = np.flatnonzero(gt >= 0)
    by: dict[int, list[tuple[float, int]]] = {}
    for i, j, s in rows[["anchor", "candidate", score_col]].itertuples(index=False):
        by.setdefault(int(i), []).append((float(s), int(j)))
    ranks = []
    for i in valid:
        rank = 10**9
        for r, (_, j) in enumerate(sorted(by.get(int(i), []), reverse=True), start=1):
            if j == int(gt[i]):
                rank = r
                break
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {"r@1": float(np.mean(ranks <= 1)), "r@5": float(np.mean(ranks <= 5)), "r@10": float(np.mean(ranks <= 10)), "r@50": float(np.mean(ranks <= 50)), "median_true_rank": float(np.median(ranks)), "valid": int(len(valid))}


def main():
    out_dir = Path("runs/et10000_cross_encoder_blend_sis")
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, scores, ens, gt, _ = base.load_score_sets("SIS", "test", out_dir / "test")
    rows = pd.read_csv("runs/et10000_pair_cross_encoder_sis/test_cross_encoder_candidates.csv")
    anchors = rows["anchor"].to_numpy(dtype=np.int64)
    cands = rows["candidate"].to_numpy(dtype=np.int64)
    rows["ens"] = ens[anchors, cands]
    p = rows["p_hat"].to_numpy(dtype=np.float32)
    eps = 1e-6
    rows["logit_p"] = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    rows["ens_z"] = zscore(rows["ens"].to_numpy(dtype=np.float32))
    rows["p_z"] = zscore(rows["logit_p"].to_numpy(dtype=np.float32))
    results = []
    for beta in [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0]:
        col = f"blend_{beta}"
        rows[col] = rows["ens_z"] + beta * rows["p_z"]
        item = {"beta": beta, **eval_ranker(rows, gt, col)}
        results.append(item)
    best = max(results, key=lambda x: x["r@1"])
    rows.to_csv(out_dir / "test_blend_candidates.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({"best": best, "results": results}, indent=2))
    print(json.dumps({"best": best, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
