from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# 回到当前最稳定的 4 个特征：
#   trigger_time_obs, predicted sky_map_overlap, waveform score, waveform reciprocal rank。
# 与 66 不同的是，本实验使用 train+val 两个 split 训练 reranker，再在 test 上做 full-catalog ranking。
m = importlib.import_module("scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank")
base = m.base

OUT_ROOT = Path("runs/ligo_sis_trainval_full_catalog_rerank_20260608")
PROB_ROOT = Path("runs/ligo_sis_grid18_rank_fusion_20260604")

EPS = 1e-8
ALPHA = 2.0
NEG_PER_POS = 220
CHUNK_ROWS = 64


def sharpen(prob: np.ndarray) -> np.ndarray:
    p = np.power(np.maximum(prob, EPS), ALPHA).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p


def load_split(job: dict, split: str):
    _, _, _, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    prob = np.load(PROB_ROOT / f"{split}_prob.npy").astype(np.float32)
    ranks = base.row_ranks(scores)
    return time_obs.reset_index(drop=True), gt, scores.astype(np.float32), ranks.astype(np.int32), sharpen(prob)


def overlap(prob: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.log(np.sum(prob[a] * prob[c], axis=1) + EPS).astype(np.float32)


def feature_matrix(time_obs, gt, scores, ranks, prob, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            m.log1p_delta_time_obs(time_obs, a, c),
            overlap(prob, a, c),
            scores[a, c].astype(np.float32),
            (1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32),
        ]
    ).astype(np.float32)


def build_random_train(split_pack, seed: int):
    time_obs, gt, scores, ranks, prob = split_pack
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    pos_a = valid
    pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    a = np.concatenate([pos_a, neg_a])
    c = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    x = feature_matrix(time_obs, gt, scores, ranks, prob, a, c)
    return x, y


def build_hard_train(split_pack, seed: int, k_each: int = 80):
    time_obs, gt, scores, ranks, prob = split_pack
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    xs, ys = [], []
    allc = np.arange(n, dtype=np.int32)
    for i, a0 in enumerate(valid):
        p = int(gt[int(a0)])
        aa = np.full(n, int(a0), dtype=np.int32)
        dt = -m.log1p_delta_time_obs(time_obs, aa, allc)
        sky = np.log(np.maximum(prob[int(a0) : int(a0) + 1] @ prob.T, EPS)).reshape(-1)
        cand = set()
        for arr in (scores[int(a0)].astype(np.float32), dt.astype(np.float32), sky.astype(np.float32)):
            s = arr.copy()
            s[int(a0)] = -np.inf
            s[p] = -np.inf
            top = np.argpartition(-s, min(k_each, n - 2))[:k_each]
            cand.update(map(int, top))
        cand.discard(int(a0))
        cand.discard(p)
        cand = np.array(sorted(cand), dtype=np.int32)
        if len(cand) > NEG_PER_POS:
            cand = rng.choice(cand, size=NEG_PER_POS, replace=False).astype(np.int32)
        cols = np.concatenate([np.array([p], dtype=np.int32), cand])
        a = np.full(len(cols), int(a0), dtype=np.int32)
        xs.append(feature_matrix(time_obs, gt, scores, ranks, prob, a, cols))
        ys.append(np.concatenate([np.ones(1, dtype=np.int8), np.zeros(len(cols) - 1, dtype=np.int8)]))
        if (i + 1) % 1000 == 0:
            print("BUILD_HARD_TRAIN", seed, i + 1, flush=True)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.int8)


def eval_full(clf, split_pack):
    time_obs, gt, scores, ranks, prob = split_pack
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start : start + CHUNK_ROWS]
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        pred = clf.predict_proba(feature_matrix(time_obs, gt, scores, ranks, prob, a, c))[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
        print("EVAL_ROWS", start + len(rows), flush=True)
    r = np.asarray(out)
    return {
        "r@1": float(np.mean(r <= 1)),
        "r@5": float(np.mean(r <= 5)),
        "r@10": float(np.mean(r <= 10)),
        "r@50": float(np.mean(r <= 50)),
        "r@100": float(np.mean(r <= 100)),
        "r@500": float(np.mean(r <= 500)),
        "median_rank": float(np.median(r)),
        "valid": int(len(valid)),
    }


def train_and_eval(name: str, X: np.ndarray, y: np.ndarray, test_pack, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]
    clf = HistGradientBoostingClassifier(
        max_iter=360,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1e-4,
        class_weight="balanced",
        random_state=seed,
    )
    print("FIT", name, X.shape, "pos", int(y.sum()), "neg", int((y == 0).sum()), flush=True)
    clf.fit(X, y)
    sample = clf.predict_proba(X[: min(300000, len(X))])[:, 1]
    return {"method": name, "train_rows": int(len(y)), "train_auc_sample": float(roc_auc_score(y[: len(sample)], sample)), **eval_full(clf, test_pack)}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    train = load_split(job, "train")
    val = load_split(job, "val")
    test = load_split(job, "test")
    rows = []

    x_train, y_train = build_random_train(train, 71001)
    x_val, y_val = build_random_train(val, 71002)
    rows.append(train_and_eval("hgb_random_neg_trainval", np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]), test, 71003))
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary_partial.csv", index=False)
    print(rows[-1], flush=True)

    xh_train, yh_train = build_hard_train(train, 71004)
    xh_val, yh_val = build_hard_train(val, 71005)
    rows.append(train_and_eval("hgb_hard_neg_trainval", np.vstack([xh_train, xh_val]), np.concatenate([yh_train, yh_val]), test, 71006))
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
