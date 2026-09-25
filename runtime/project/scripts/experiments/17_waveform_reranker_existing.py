from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from matchgw.pipeline import build_model, embed_eval
from matchgw.data import pad_or_trim, multiband_preprocess, zscore_channels, peak_flip_channels

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
ENSEMBLE_WEIGHTS = {
    "SIS": (0.50, 0.35, 0.15),
    "PM": (0.35, 0.30, 0.35),
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


def topk_order(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[1]
    k = max(1, min(k, n - 1))
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    local = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, local, axis=1)


def rank_matrix(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    ranks = np.full(scores.shape, order.shape[1] + 1, dtype=np.float32)
    for i, row in enumerate(order):
        ranks[i, row] = np.arange(1, len(row) + 1, dtype=np.float32)
    return ranks


def combine(scores: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    out = np.zeros_like(scores[0], dtype=np.float32)
    for s, w in zip(scores, weights):
        ss = s.astype(np.float32, copy=True)
        np.fill_diagonal(ss, 0.0)
        out += float(w) * ss
    np.fill_diagonal(out, -np.inf)
    return out


def load_score_sets(family: str, split: str, out_dir: Path):
    base_cfg = cfg_from_run(RUNS[family][0][1], out_dir)
    arrays = load_match_arrays(base_cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), base_cfg)
    ds = EvaluationSet(arrays, splits["lensed"][split], splits["unlensed"][split], base_cfg)
    gt = ground_truth_partner(ds.meta)
    scores = []
    for label, run_dir in RUNS[family]:
        cfg = cfg_from_run(run_dir, out_dir / label)
        model = build_model(cfg)
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        scores.append(similarity_matrix(embed_eval(model, ds, cfg, cpu=False)))
        print(f"loaded {family} {split} {label}", flush=True)
    ens = combine(scores, ENSEMBLE_WEIGHTS[family])
    return arrays, splits, ds, scores, ens, gt, base_cfg


def precompute_waveforms(ds: EvaluationSet, cfg: MatchRunConfig) -> np.ndarray:
    # Waveform-only representation for reranking: four frequency bands, no physical parameters.
    out = np.empty((len(ds), 4, cfg.target_len // cfg.stride), dtype=np.float32)
    mb_cfg = MatchRunConfig(**{**asdict(cfg), "preprocess": "multiband"})
    for i, wf in enumerate(ds.waveforms):
        x = pad_or_trim(wf, cfg.target_len, cfg.stride)
        x = multiband_preprocess(x, mb_cfg)
        if cfg.aug_flip:
            x = peak_flip_channels(x)
        out[i] = zscore_channels(x)
    return out


def lag_features(anchor: np.ndarray, cand: np.ndarray, lags: tuple[int, ...] = (-16, -8, 0, 8, 16)) -> list[float]:
    full_a = anchor[-1]
    full_c = cand[-1]
    vals = []
    for lag in lags:
        vals.append(float(np.mean(full_a * np.roll(full_c, lag))))
    vals_arr = np.asarray(vals, dtype=np.float32)
    best_idx = int(np.argmax(vals_arr))
    return [float(vals_arr[best_idx]), float(lags[best_idx]), float(vals_arr.max() - vals_arr.mean())]


def make_examples(scores: list[np.ndarray], ens: np.ndarray, gt: np.ndarray, waves: np.ndarray, topk: int = 50):
    all_scores = scores + [ens]
    orders = [topk_order(s, topk) for s in all_scores]
    ranks = [rank_matrix(s, o) for s, o in zip(all_scores, orders)]
    top1 = []
    margins = []
    for s, o in zip(all_scores, orders):
        vals = np.take_along_axis(s, o, axis=1)
        top1.append(vals[:, 0])
        margins.append(vals[:, 0] - vals[:, 1])
    X = []
    y = []
    anchors = []
    cands = []
    for i in range(ens.shape[0]):
        cand = set()
        for o in orders:
            cand.update(map(int, o[i, :topk]))
        cand.discard(i)
        ai = waves[i]
        for j in cand:
            cj = waves[j]
            # 每个频带的归一化点积近似相关系数；MSE 可由 z-score 后的相关性补充表达。
            corr = np.mean(ai * cj, axis=1)
            abs_corr = np.abs(corr)
            mse = np.mean((ai - cj) ** 2, axis=1)
            feat = []
            for mi, s in enumerate(all_scores):
                feat.extend([float(s[i, j]), float(1.0 / ranks[mi][i, j]), float(top1[mi][i]), float(margins[mi][i])])
            feat.extend(corr.astype(float).tolist())
            feat.extend(abs_corr.astype(float).tolist())
            feat.extend(mse.astype(float).tolist())
            feat.extend([float(corr.mean()), float(corr.max()), float(corr[-1] - corr[:3].mean())])
            feat.extend(lag_features(ai, cj))
            feat.extend([float(ens[i, j]), float(ens[j, i]), float(abs(ens[i, j] - ens[j, i]))])
            X.append(feat)
            y.append(1 if int(gt[i]) == int(j) else 0)
            anchors.append(i)
            cands.append(j)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), np.asarray(anchors, dtype=np.int32), np.asarray(cands, dtype=np.int32)


def eval_ranker(proba: np.ndarray, anchors: np.ndarray, cands: np.ndarray, gt: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0)
    by: dict[int, list[tuple[float, int]]] = {}
    for p, i, j in zip(proba, anchors, cands):
        by.setdefault(int(i), []).append((float(p), int(j)))
    ranks = []
    for i in valid:
        items = sorted(by.get(int(i), []), reverse=True)
        rank = 10**9
        for r, (_, j) in enumerate(items, start=1):
            if j == int(gt[i]):
                rank = r
                break
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {"r@1": float(np.mean(ranks <= 1)), "r@5": float(np.mean(ranks <= 5)), "r@10": float(np.mean(ranks <= 10)), "r@50": float(np.mean(ranks <= 50)), "median_true_rank": float(np.median(ranks)), "valid": int(len(valid))}


def run_family(family: str, out_root: Path):
    out_dir = out_root / family
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, val_ds, val_scores, val_ens, val_gt, base_cfg = load_score_sets(family, "val", out_dir)
    _, _, test_ds, test_scores, test_ens, test_gt, _ = load_score_sets(family, "test", out_dir)
    print(f"precompute waves {family} val", flush=True)
    val_waves = precompute_waveforms(val_ds, base_cfg)
    print(f"precompute waves {family} test", flush=True)
    test_waves = precompute_waveforms(test_ds, base_cfg)
    print(f"examples {family} val", flush=True)
    Xv, yv, av, cv = make_examples(val_scores, val_ens, val_gt, val_waves, topk=50)
    print(f"examples {family} test", flush=True)
    Xt, yt, at, ct = make_examples(test_scores, test_ens, test_gt, test_waves, topk=50)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1e-4, class_weight="balanced", random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xt)[:, 1]
    val = eval_ranker(pv, av, cv, val_gt)
    test = eval_ranker(pt, at, ct, test_gt)
    try:
        auc = float(roc_auc_score(yv, pv))
    except Exception:
        auc = float("nan")
    pd.DataFrame({"anchor": at, "candidate": ct, "p_hat": pt, "is_true": yt}).to_csv(out_dir / "test_waveform_reranked_candidates.csv", index=False)
    result = {"family": family, "train_examples": int(len(yv)), "train_positive": int(yv.sum()), "train_negative": int(len(yv) - yv.sum()), "val_auc": auc, "val": val, "test": test, "features": int(Xv.shape[1]), "topk": 50}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    out_root = Path("runs/et10000_waveform_reranker_existing")
    out_root.mkdir(parents=True, exist_ok=True)
    results = [run_family(f, out_root) for f in ["SIS", "PM"]]
    (out_root / "summary.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(r["family"], "val_auc", round(r["val_auc"], 4), "test", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["test"].items()})


if __name__ == "__main__":
    main()
