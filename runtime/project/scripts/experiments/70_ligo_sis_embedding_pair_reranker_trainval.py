from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 第 68/69 个深度 pair 模型在 val->test 上明显过拟合。
# 本实验改用已经训练好的 match waveform embedding，扩大到 train+val 训练：
#   特征 = embedding pair 统计特征 + waveform score/rank + trigger_time_obs + predicted sky overlap。
# 这样模型不重新学习底层波形，只学习候选对层面的稳定组合方式。
m = importlib.import_module("scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank")
base = m.base

OUT_ROOT = Path("runs/ligo_sis_embedding_pair_reranker_trainval_20260608")
PROB_ROOT = Path("runs/ligo_sis_grid18_rank_fusion_20260604")
EMB_ROOT = Path("runs/unified_sky_aux_comparison_20260603/ligo_noisy_sis")
SKY_CKPT = Path("runs/ligo_sis_resnet_grid18_skymap_rerank_20260604/ligo_noisy_sis/grid_skymap_cnn.pt")

EPS = 1e-8
ALPHA = 2.0
K_EACH = 100
NEG_PER_POS = 80
CHUNK_ROWS = 48


def sharpen(prob: np.ndarray) -> np.ndarray:
    p = np.power(np.maximum(prob, EPS), ALPHA).astype(np.float32)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p


def ensure_prob(job: dict, split: str, ds) -> np.ndarray:
    # val/test 的预测 sky-map 已有；train 如果缺失，则用 51 checkpoint 补一个缓存。
    path = PROB_ROOT / f"{split}_prob.npy"
    if path.exists():
        return sharpen(np.load(path).astype(np.float32))
    if split != "train":
        raise FileNotFoundError(path)
    print("PREDICT_TRAIN_SKY_PROB", path, flush=True)
    ckpt = torch.load(SKY_CKPT, map_location="cpu")
    model = m.SkyMapCNN(in_channels=int(ds[0].shape[0]))
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prob = m.predict_maps(model, ds, device)
    PROB_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(path, prob.astype(np.float32))
    return sharpen(prob)


def load_split(job: dict, split: str):
    _, ds, _, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    prob = ensure_prob(job, split, ds)
    ranks = base.row_ranks(scores)
    emb = np.load(EMB_ROOT / split / f"LIGO_noisy_SIS_{split}_embeddings.npy").astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), EPS)
    return time_obs.reset_index(drop=True), gt, scores.astype(np.float32), ranks.astype(np.int32), prob, emb


def dt_mat(time_obs: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    n = len(time_obs)
    allc = np.arange(n, dtype=np.int32)
    out = np.empty((len(rows), n), np.float32)
    for i, a in enumerate(rows):
        out[i] = -m.log1p_delta_time_obs(time_obs, np.full(n, int(a), dtype=np.int32), allc)
    return out


def candidate_sets(time_obs, scores, prob, rows):
    sky = np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)
    dt = dt_mat(time_obs, rows)
    cand_list = []
    for rp, a in enumerate(rows):
        cand = set()
        for arr in (scores[int(a)].astype(np.float32), sky[rp], dt[rp]):
            s = arr.copy()
            s[int(a)] = -np.inf
            top = np.argpartition(-s, min(K_EACH, len(s) - 1))[:K_EACH]
            cand.update(map(int, top))
        cand.discard(int(a))
        cand_list.append(np.array(sorted(cand), dtype=np.int32))
    return cand_list, sky, dt


def compact_pair_features(emb: np.ndarray, scores, ranks, sky_row, dt_row, a: int, cols: np.ndarray) -> np.ndarray:
    za = emb[int(a)][None, :]
    zb = emb[cols]
    diff = np.abs(za - zb)
    prod = za * zb
    dot = np.sum(za * zb, axis=1)
    l2 = np.sqrt(np.sum((za - zb) ** 2, axis=1))
    l1 = np.mean(diff, axis=1)
    rr = ranks[a, cols].astype(np.float32)
    x = np.column_stack(
        [
            scores[a, cols].astype(np.float32),
            1.0 / np.maximum(rr, 1.0),
            -np.log1p(np.maximum(rr, 1.0)),
            sky_row[cols].astype(np.float32),
            dt_row[cols].astype(np.float32),
            (sky_row[cols] * dt_row[cols]).astype(np.float32),
            dot.astype(np.float32),
            l2.astype(np.float32),
            l1.astype(np.float32),
            diff.mean(axis=1).astype(np.float32),
            diff.std(axis=1).astype(np.float32),
            diff.max(axis=1).astype(np.float32),
            prod.mean(axis=1).astype(np.float32),
            prod.std(axis=1).astype(np.float32),
            prod.min(axis=1).astype(np.float32),
            prod.max(axis=1).astype(np.float32),
        ]
    )
    return x.astype(np.float32)


def build_train_split(time_obs, gt, scores, ranks, prob, emb, seed: int):
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    cand_list, sky, dt = candidate_sets(time_obs, scores, prob, valid)
    xs, ys = [], []
    for rp, a in enumerate(valid):
        p = int(gt[int(a)])
        cand = cand_list[rp]
        if p not in set(map(int, cand)):
            cand = np.concatenate([cand, np.array([p], dtype=np.int32)])
        neg = cand[cand != p]
        if len(neg) > NEG_PER_POS:
            hard_score = scores[int(a), neg] + 0.20 * sky[rp, neg] + 0.25 * dt[rp, neg]
            hard = neg[np.argpartition(-hard_score, NEG_PER_POS - 1)[:NEG_PER_POS]]
            neg = rng.choice(hard, size=NEG_PER_POS, replace=False).astype(np.int32)
        cols = np.concatenate([np.array([p], dtype=np.int32), neg])
        xs.append(compact_pair_features(emb, scores, ranks, sky[rp], dt[rp], int(a), cols))
        ys.append(np.concatenate([np.ones(1, dtype=np.int8), np.zeros(len(cols) - 1, dtype=np.int8)]))
        if (rp + 1) % 1000 == 0:
            print("BUILD_TRAIN_SPLIT", seed, rp + 1, flush=True)
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.int8)


def eval_candidate(clf, time_obs, gt, scores, ranks, prob, emb):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    all_ranks, in_cand = [], []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start : start + CHUNK_ROWS]
        cand_list, sky, dt = candidate_sets(time_obs, scores, prob, rows)
        for rp, a in enumerate(rows):
            p = int(gt[int(a)])
            cand = cand_list[rp]
            hit = p in set(map(int, cand))
            in_cand.append(hit)
            if not hit:
                all_ranks.append(len(cand) + 1)
                continue
            x = compact_pair_features(emb, scores, ranks, sky[rp], dt[rp], int(a), cand)
            pred = clf.predict_proba(x)[:, 1]
            true = pred[np.where(cand == p)[0][0]]
            all_ranks.append(int(1 + np.sum(pred > true)))
        print("EVAL_ROWS", start + len(rows), flush=True)
    r = np.asarray(all_ranks)
    return {
        "candidate_recall": float(np.mean(in_cand)),
        "r@1": float(np.mean(r <= 1)),
        "r@5": float(np.mean(r <= 5)),
        "r@10": float(np.mean(r <= 10)),
        "r@50": float(np.mean(r <= 50)),
        "r@100": float(np.mean(r <= 100)),
        "median_rank": float(np.median(r)),
        "valid": int(len(valid)),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    job = m.JOBS[0]
    train = load_split(job, "train")
    val = load_split(job, "val")
    test = load_split(job, "test")
    x1, y1 = build_train_split(*train, seed=70001)
    x2, y2 = build_train_split(*val, seed=70002)
    X = np.vstack([x1, x2]).astype(np.float32)
    y = np.concatenate([y1, y2]).astype(np.int8)
    rng = np.random.default_rng(70003)
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]
    print("TRAIN_SHAPE", X.shape, "pos", int(y.sum()), "neg", int((y == 0).sum()), flush=True)
    models = {
        "hgb_embedding_pair_trainval": HistGradientBoostingClassifier(
            max_iter=360,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            class_weight="balanced",
            random_state=70,
        ),
        "sgd_embedding_pair_trainval": make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, class_weight="balanced", random_state=70),
        ),
        "extratrees_embedding_pair_trainval": ExtraTreesClassifier(
            n_estimators=320,
            max_depth=18,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=70,
        ),
    }
    rows = []
    for name, clf in models.items():
        print("FIT", name, flush=True)
        clf.fit(X, y)
        met = eval_candidate(clf, *test)
        row = {"method": name, "feature_dim": int(X.shape[1]), "train_rows": int(len(y)), **met}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_ROOT / "summary_partial.csv", index=False)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
