from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module("scripts.experiments.17_waveform_reranker_existing")


def subsample_for_classifier(X: np.ndarray, y: np.ndarray, max_neg: int = 700_000):
    rng = np.random.default_rng(42)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(neg) > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return X[idx], y[idx], {"positive": int(len(pos)), "negative_total": int(np.sum(y == 0)), "negative_used": int(len(neg))}


def main():
    family = "SIS"
    out_dir = Path("runs/et10000_waveform_reranker_train_sis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("load train scores", flush=True)
    _, _, train_ds, train_scores, train_ens, train_gt, base_cfg = base.load_score_sets(family, "train", out_dir / "train")
    print("load test scores", flush=True)
    _, _, test_ds, test_scores, test_ens, test_gt, _ = base.load_score_sets(family, "test", out_dir / "test")

    print("precompute waves train", flush=True)
    train_waves = base.precompute_waveforms(train_ds, base_cfg)
    print("precompute waves test", flush=True)
    test_waves = base.precompute_waveforms(test_ds, base_cfg)

    print("make train examples top30", flush=True)
    Xtr, ytr, atr, ctr = base.make_examples(train_scores, train_ens, train_gt, train_waves, topk=30)
    print("make test examples top50", flush=True)
    Xte, yte, ate, cte = base.make_examples(test_scores, test_ens, test_gt, test_waves, topk=50)

    print("subsample train", flush=True)
    Xfit, yfit, sample_info = subsample_for_classifier(Xtr, ytr)
    print("fit classifier", Xfit.shape, sample_info, flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=700,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=3e-4,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(Xfit, yfit)
    ptr = clf.predict_proba(Xfit)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    test = base.eval_ranker(pte, ate, cte, test_gt)
    try:
        auc = float(roc_auc_score(yfit, ptr))
    except Exception:
        auc = float("nan")
    pd.DataFrame({"anchor": ate, "candidate": cte, "p_hat": pte, "is_true": yte}).to_csv(out_dir / "test_waveform_reranked_candidates.csv", index=False)
    result = {
        "family": family,
        "train_split": "train",
        "train_topk": 30,
        "test_topk": 50,
        "raw_train_examples": int(len(ytr)),
        "fit_examples": int(len(yfit)),
        "sample_info": sample_info,
        "fit_auc": auc,
        "test": test,
        "features": int(Xfit.shape[1]),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
