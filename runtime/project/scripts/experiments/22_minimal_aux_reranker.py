from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module("scripts.experiments.17_waveform_reranker_existing")
aux = importlib.import_module("scripts.experiments.21_observable_aux_reranker")

GROUPS = {
    "time": ["delta_time"],
    "sky": ["sky_sep"],
    "mass": ["chirp_diff", "q_diff"],
    "mass_time": ["chirp_diff", "q_diff", "delta_time"],
    "mass_sky": ["chirp_diff", "q_diff", "sky_sep"],
    "time_sky": ["delta_time", "sky_sep"],
    "mass_time_sky": ["chirp_diff", "q_diff", "delta_time", "sky_sep"],
}


def make_feature_table(obs: pd.DataFrame, scores: list[np.ndarray], ens: np.ndarray, gt: np.ndarray, topk: int, feature_names: list[str]):
    candidates = aux.topk_union(scores, ens, topk)
    vals = obs.reset_index(drop=True)
    ra = vals["ra"].to_numpy(); dec = vals["dec"].to_numpy()
    cm = vals["chirp_mass"].to_numpy(); q = vals["mass_ratio"].to_numpy()
    t = vals["geocent_time"].to_numpy()
    rows, y, anchors, cands = [], [], [], []
    for i, row in enumerate(candidates):
        for j in row:
            feat = []
            for name in feature_names:
                if name == "delta_time":
                    feat.append(float(np.log1p(abs(t[i] - t[j]))))
                elif name == "sky_sep":
                    feat.append(float(aux.angular_sep(ra[i], dec[i], ra[j], dec[j])))
                elif name == "chirp_diff":
                    feat.append(float(abs(np.log(cm[i] / cm[j]))))
                elif name == "q_diff":
                    feat.append(float(abs(q[i] - q[j])))
                else:
                    raise ValueError(name)
            rows.append(feat)
            y.append(1 if int(gt[i]) == int(j) else 0)
            anchors.append(i); cands.append(j)
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int8), np.asarray(anchors, dtype=np.int32), np.asarray(cands, dtype=np.int32)


def run_family_group(family: str, mode: str, group: str, out_root: Path) -> dict:
    out_dir = out_root / family / mode / group
    out_dir.mkdir(parents=True, exist_ok=True)
    _, val_splits, _, val_scores, val_ens, val_gt, cfg = base.load_score_sets(family, "val", out_dir / "val")
    _, test_splits, _, test_scores, test_ens, test_gt, _ = base.load_score_sets(family, "test", out_dir / "test")
    val_obs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, val_splits["lensed"]["val"], val_splits["unlensed"]["val"]), mode, seed=1000 + abs(hash((family, mode, group))) % 10000)
    test_obs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, test_splits["lensed"]["test"], test_splits["unlensed"]["test"]), mode, seed=2000 + abs(hash((family, mode, group))) % 10000)
    names = GROUPS[group]
    Xv, yv, av, cv = make_feature_table(val_obs, val_scores, val_ens, val_gt, 50, names)
    Xt, yt, at, ct = make_feature_table(test_obs, test_scores, test_ens, test_gt, 50, names)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1e-4, class_weight="balanced", random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xt)[:, 1]
    result = {
        "family": family,
        "mode": mode,
        "group": group,
        "features": names,
        "feature_count": len(names),
        "val_auc": float(roc_auc_score(yv, pv)),
        "test": aux.eval_ranker(pt, at, ct, test_gt),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    out_root = Path("runs/et10000_minimal_aux_reranker")
    results = []
    # exact 是上界，rough 是保守扰动；优先看 rough 下最少参数能否过 0.7。
    for mode in ["exact", "realistic", "rough"]:
        for family in ["SIS", "PM"]:
            for group in GROUPS:
                print("RUN", family, mode, group, flush=True)
                r = run_family_group(family, mode, group, out_root)
                print(family, mode, group, r["test"], flush=True)
                results.append(r)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for r in results:
        t = r["test"]
        rows.append({"family": r["family"], "mode": r["mode"], "group": r["group"], "feature_count": r["feature_count"], "features": "+".join(r["features"]), "r@1": t["r@1"], "r@5": t["r@5"], "r@10": t["r@10"], "r@50": t["r@50"], "median_rank": t["median_true_rank"], "val_auc": r["val_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "minimal_aux_summary.csv", index=False)
    print(df.sort_values(["family", "mode", "r@1"], ascending=[True, True, False]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
