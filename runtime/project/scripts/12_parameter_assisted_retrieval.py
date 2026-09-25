from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matchgw import MatchRunConfig
from matchgw.catalog import catalog_system_report
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import retrieval_metrics, topk_edges, greedy_pairs, pair_metrics

# 参数辅助检索使用模拟/参数估计可给出的源本征参数。
# geocent_time 会被透镜时延改变，luminosity_distance 可能受放大率退化影响，默认不参与匹配。
DEFAULT_COLUMNS = [
    "mass_1_source", "mass_2_source", "a_1", "a_2", "tilt_1", "tilt_2",
    "phi_12", "phi_jl", "ra", "dec", "theta_jn", "psi", "phase",
]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def parameter_catalog_features(data_root: Path, family: str, lensed_idx: np.ndarray, unlensed_idx: np.ndarray, columns: list[str]) -> np.ndarray:
    family = family.upper()
    lensed = pd.read_csv(data_root / f"{family}_data_0222" / "lensed_source_samples.csv")
    unlensed = pd.read_csv(data_root / "Unlensed_data_0222" / "source_samples.csv")
    n_lensed_total = len(lensed) // 2
    l1 = lensed.iloc[lensed_idx][columns].to_numpy(dtype=np.float32)
    l2 = lensed.iloc[n_lensed_total + lensed_idx][columns].to_numpy(dtype=np.float32)
    u = unlensed.iloc[unlensed_idx][columns].to_numpy(dtype=np.float32)
    return _zscore(np.vstack([l1, l2, u]).astype(np.float32, copy=False))


def score_from_features(x: np.ndarray) -> np.ndarray:
    # 用负欧氏距离做相似度，再映射到 (0,1]，完全相同的源参数得到 1。
    dist = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    scores = np.exp(-0.5 * dist).astype(np.float32)
    np.fill_diagonal(scores, -np.inf)
    return scores


def run_one(data_root: Path, family: str, out_dir: Path, columns: list[str], exact_threshold: float) -> dict:
    cfg = MatchRunConfig(data_root=data_root, model_type=family, data_mode="noisy", lensed_limit=10000, unlensed_limit=10000)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    test_l = splits["lensed"]["test"]
    test_u = splits["unlensed"]["test"]
    ds = EvaluationSet(arrays, test_l, test_u, cfg)
    gt = ground_truth_partner(ds.meta)
    scores = score_from_features(parameter_catalog_features(data_root, family, test_l, test_u, columns))

    retrieval = retrieval_metrics(scores, gt)
    edges = topk_edges(scores, topk=1, min_score=exact_threshold, mutual=False, reciprocal_rank_max=None)
    pairs = greedy_pairs(edges)
    pair_stats = pair_metrics(pairs, gt)
    candidate_rows = []
    true_partner = {tuple(sorted((i, int(j)))) for i, j in enumerate(gt) if j >= 0 and i < j}
    for i, j, score in edges:
        e = tuple(sorted((int(i), int(j))))
        candidate_rows.append({"i": int(i), "j": int(j), "score": float(score), "p_hat": float(score), "tier": "tier1", "is_true": int(e in true_partner)})
    candidates = pd.DataFrame(candidate_rows)
    systems, catalog_stats = catalog_system_report(candidates, ds.meta, exact_threshold, threshold_name="tier1")

    result = {
        "family": family.upper(),
        "data_mode": "noisy",
        "method": "parameter_assisted_oracle",
        "columns": columns,
        "exact_threshold": exact_threshold,
        "test": {**retrieval, **pair_stats, "candidate_edges": int(len(edges))},
        "test_catalog_tier1": catalog_stats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(out_dir / f"{family.upper()}_test_parameter_candidates.csv", index=False)
    systems.to_csv(out_dir / f"{family.upper()}_test_parameter_catalog_systems_tier1.csv", index=False)
    with open(out_dir / f"{family.upper()}_parameter_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Parameter-assisted catalog retrieval upper-bound experiment.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--families", nargs="+", choices=["SIS", "PM"], default=["SIS", "PM"])
    ap.add_argument("--exact-threshold", type=float, default=0.999999)
    args = ap.parse_args()
    results = [run_one(args.data_root, fam, args.out_dir, DEFAULT_COLUMNS, args.exact_threshold) for fam in args.families]
    rows = []
    for r in results:
        t = r["test"]
        cat = r["test_catalog_tier1"]
        rows.append({
            "family": r["family"],
            "r@1": t["r@1"],
            "r@10": t["r@10"],
            "pair_f1": t["f1"],
            "catalog_f1": cat["catalog_system_f1"],
            "catalog_precision": cat["catalog_system_precision"],
            "catalog_recall": cat["catalog_system_recall"],
        })
    pd.DataFrame(rows).to_csv(args.out_dir / "parameter_assisted_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
