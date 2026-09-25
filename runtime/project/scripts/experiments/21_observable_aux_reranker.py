from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module("scripts.experiments.17_waveform_reranker_existing")


def chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return ((m1 * m2) ** (3.0 / 5.0)) / ((m1 + m2) ** (1.0 / 5.0))


def angle_diff(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def angular_sep(ra1, dec1, ra2, dec2):
    sin1, cos1 = np.sin(dec1), np.cos(dec1)
    sin2, cos2 = np.sin(dec2), np.cos(dec2)
    cosd = sin1 * sin2 + cos1 * cos2 * np.cos(ra1 - ra2)
    return np.arccos(np.clip(cosd, -1.0, 1.0))


def catalog_observable_frame(data_root: Path, family: str, lensed_idx: np.ndarray, unlensed_idx: np.ndarray) -> pd.DataFrame:
    fam = family.upper()
    lensed = pd.read_csv(data_root / f"{fam}_data_0222" / "lensed_source_samples.csv")
    unlensed = pd.read_csv(data_root / "Unlensed_data_0222" / "source_samples.csv")
    n = len(lensed) // 2
    l1 = lensed.iloc[lensed_idx].copy()
    l2 = lensed.iloc[n + lensed_idx].copy()
    u = unlensed.iloc[unlensed_idx].copy()
    out = pd.concat([l1, l2, u], ignore_index=True)
    out["chirp_mass"] = chirp_mass(out["mass_1_source"].to_numpy(), out["mass_2_source"].to_numpy())
    out["mass_ratio"] = np.minimum(out["mass_1_source"], out["mass_2_source"]) / np.maximum(out["mass_1_source"], out["mass_2_source"])
    # chi_eff-like rough spin summary: observable proxy, not a lens truth label.
    m1 = out["mass_1_source"].to_numpy()
    m2 = out["mass_2_source"].to_numpy()
    out["chi_eff_proxy"] = (m1 * out["a_1"].to_numpy() * np.cos(out["tilt_1"].to_numpy()) + m2 * out["a_2"].to_numpy() * np.cos(out["tilt_2"].to_numpy())) / (m1 + m2)
    return out


def perturb_observables(df: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    if mode == "exact":
        return df.copy()
    rng = np.random.default_rng(seed)
    x = df.copy()
    if mode == "mild":
        mass_frac, sky_sigma, dist_frac, spin_sigma, time_sigma = 0.05, 0.03, 0.20, 0.10, 0.01
    elif mode == "realistic":
        mass_frac, sky_sigma, dist_frac, spin_sigma, time_sigma = 0.10, 0.08, 0.35, 0.20, 0.05
    elif mode == "rough":
        mass_frac, sky_sigma, dist_frac, spin_sigma, time_sigma = 0.20, 0.20, 0.60, 0.35, 0.10
    else:
        raise ValueError(mode)
    for col in ["mass_1_source", "mass_2_source", "chirp_mass"]:
        x[col] = x[col] * np.exp(rng.normal(0.0, mass_frac, size=len(x)))
    x["mass_ratio"] = np.clip(x["mass_ratio"] + rng.normal(0.0, mass_frac, size=len(x)), 0.02, 1.0)
    x["ra"] = np.mod(x["ra"] + rng.normal(0.0, sky_sigma, size=len(x)), 2 * np.pi)
    x["dec"] = np.clip(x["dec"] + rng.normal(0.0, sky_sigma, size=len(x)), -np.pi / 2, np.pi / 2)
    x["luminosity_distance"] = x["luminosity_distance"] * np.exp(rng.normal(0.0, dist_frac, size=len(x)))
    for col in ["a_1", "a_2", "chi_eff_proxy"]:
        x[col] = np.clip(x[col] + rng.normal(0.0, spin_sigma, size=len(x)), -1.0, 1.0)
    # GW trigger-time uncertainty is tiny compared with lensing delays here; keep as a very small perturbation.
    x["geocent_time"] = x["geocent_time"] + rng.normal(0.0, time_sigma, size=len(x))
    return x


def topk_union(scores: list[np.ndarray], ens: np.ndarray, topk: int) -> list[list[int]]:
    orders = [base.topk_order(s, topk) for s in scores + [ens]]
    rows = []
    for i in range(ens.shape[0]):
        cand = set()
        for o in orders:
            cand.update(map(int, o[i, :topk]))
        cand.discard(i)
        rows.append(sorted(cand))
    return rows


def rank_lookup(scores: np.ndarray, topk: int) -> np.ndarray:
    order = base.topk_order(scores, topk)
    return base.rank_matrix(scores, order)


def make_pair_features(obs: pd.DataFrame, scores: list[np.ndarray], ens: np.ndarray, gt: np.ndarray, topk: int):
    candidates = topk_union(scores, ens, topk)
    all_scores = scores + [ens]
    ranks = [rank_lookup(s, topk) for s in all_scores]
    rows, y, anchors, cands = [], [], [], []
    vals = obs.reset_index(drop=True)
    ra = vals["ra"].to_numpy(); dec = vals["dec"].to_numpy()
    cm = vals["chirp_mass"].to_numpy(); q = vals["mass_ratio"].to_numpy()
    m1 = vals["mass_1_source"].to_numpy(); m2 = vals["mass_2_source"].to_numpy()
    chi = vals["chi_eff_proxy"].to_numpy(); a1 = vals["a_1"].to_numpy(); a2 = vals["a_2"].to_numpy()
    dl = vals["luminosity_distance"].to_numpy(); t = vals["geocent_time"].to_numpy()
    for i, row in enumerate(candidates):
        for j in row:
            feat = []
            for s, r in zip(all_scores, ranks):
                feat.extend([float(s[i, j]), float(1.0 / r[i, j])])
            feat.extend([
                float(np.log1p(abs(t[i] - t[j]))),
                float(angular_sep(ra[i], dec[i], ra[j], dec[j])),
                float(abs(np.log(cm[i] / cm[j]))),
                float(abs(q[i] - q[j])),
                float(abs(np.log(m1[i] / m1[j]))),
                float(abs(np.log(m2[i] / m2[j]))),
                float(abs(chi[i] - chi[j])),
                float(abs(a1[i] - a1[j])),
                float(abs(a2[i] - a2[j])),
                float(abs(np.log(dl[i] / dl[j]))),
            ])
            rows.append(feat)
            y.append(1 if int(gt[i]) == int(j) else 0)
            anchors.append(i); cands.append(j)
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int8), np.asarray(anchors, dtype=np.int32), np.asarray(cands, dtype=np.int32)


def eval_ranker(proba: np.ndarray, anchors: np.ndarray, cands: np.ndarray, gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    by: dict[int, list[tuple[float, int]]] = {}
    for p, i, j in zip(proba, anchors, cands):
        by.setdefault(int(i), []).append((float(p), int(j)))
    ranks = []
    for i in valid:
        rank = 10**9
        for r, (_, j) in enumerate(sorted(by.get(int(i), []), reverse=True, ), start=1):
            if j == int(gt[i]):
                rank = r; break
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {"r@1": float(np.mean(ranks <= 1)), "r@5": float(np.mean(ranks <= 5)), "r@10": float(np.mean(ranks <= 10)), "r@50": float(np.mean(ranks <= 50)), "median_true_rank": float(np.median(ranks)), "valid": int(len(valid))}


def run_family(family: str, mode: str, out_root: Path, topk: int = 50) -> dict:
    out_dir = out_root / family / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    val_arrays, val_splits, val_ds, val_scores, val_ens, val_gt, cfg = base.load_score_sets(family, "val", out_dir / "val")
    test_arrays, test_splits, test_ds, test_scores, test_ens, test_gt, _ = base.load_score_sets(family, "test", out_dir / "test")
    val_l = val_splits["lensed"]["val"]; val_u = val_splits["unlensed"]["val"]
    test_l = test_splits["lensed"]["test"]; test_u = test_splits["unlensed"]["test"]
    val_obs = perturb_observables(catalog_observable_frame(cfg.data_root, family, val_l, val_u), mode, seed=100 + hash((family, mode)) % 10000)
    test_obs = perturb_observables(catalog_observable_frame(cfg.data_root, family, test_l, test_u), mode, seed=200 + hash((family, mode)) % 10000)
    Xv, yv, av, cv = make_pair_features(val_obs, val_scores, val_ens, val_gt, topk=topk)
    Xt, yt, at, ct = make_pair_features(test_obs, test_scores, test_ens, test_gt, topk=topk)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1e-4, class_weight="balanced", random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xt)[:, 1]
    result = {
        "family": family,
        "mode": mode,
        "method": "observable_aux_top50_reranker",
        "topk": topk,
        "features": ["model_scores_ranks", "delta_trigger_time", "sky_angular_sep", "chirp_mass_diff", "mass_ratio_diff", "component_mass_diff", "spin_proxy_diff", "luminosity_distance_ratio"],
        "train_examples": int(len(yv)),
        "train_positive": int(yv.sum()),
        "val_auc": float(roc_auc_score(yv, pv)),
        "test": eval_ranker(pt, at, ct, test_gt),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"anchor": at, "candidate": ct, "p_hat": pt, "is_true": yt}).to_csv(out_dir / "test_aux_reranked_candidates.csv", index=False)
    return result


def main():
    out_root = Path("runs/et10000_observable_aux_reranker")
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ["exact", "mild", "realistic", "rough"]:
        for family in ["SIS", "PM"]:
            print("RUN", family, mode, flush=True)
            r = run_family(family, mode, out_root)
            print(family, mode, r["test"], "auc", r["val_auc"], flush=True)
            results.append(r)
    (out_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for r in results:
        t = r["test"]
        rows.append({"family": r["family"], "mode": r["mode"], "r@1": t["r@1"], "r@5": t["r@5"], "r@10": t["r@10"], "r@50": t["r@50"], "median_rank": t["median_true_rank"], "val_auc": r["val_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "observable_aux_summary.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
