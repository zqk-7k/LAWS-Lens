from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")

OUT_ROOT = Path("runs/time_matched_hard_negative_mixed_catalog_20260609")
BASE_OUT_ROOT = Path("runs/mixed_sis_pm_catalog_modality_compare_20260609")
JOBS = [("ET", "pure"), ("ET", "noisy"), ("LIGO", "pure"), ("LIGO", "noisy")]
FAMILIES = ["SIS", "PM"]
N_NEG = 200
HARD_EPS_SEQUENCE = (0.25, 0.5, 1.0)
RERANK_VARIANTS = [
    "waveform_plus_time",
    "waveform_plus_predicted_sky_overlap",
    "waveform_plus_time_plus_predicted_sky_overlap",
    "waveform_plus_true_sky_overlap",
    "waveform_plus_time_plus_true_sky_overlap",
]
DIRECT_VARIANTS = [
    "waveform_only",
    "time_only",
    "true_sky_overlap_only",
    "predicted_sky_overlap_only",
]
EPS = 1e-8


def log10_delta_time_obs(time_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    t = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    return np.log10(np.abs(t[a] - t[c]) + 1.0).astype(np.float32)


def valid_queries(gt: np.ndarray) -> np.ndarray:
    return np.flatnonzero(gt >= 0).astype(np.int32)


def same_lensed_system(meta: list[dict], q: int, k: int) -> bool:
    pq = int(meta[q].get("pair_id", -1))
    pk = int(meta[k].get("pair_id", -2))
    return pq >= 0 and pq == pk


def make_candidate_lists(
    time_obs: pd.DataFrame,
    meta: list[dict],
    gt: np.ndarray,
    kind: str,
    n_neg: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    n = len(gt)
    all_idx = np.arange(n, dtype=np.int32)
    t = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    candidates: dict[int, np.ndarray] = {}
    shortage = 0
    used_eps: list[float] = []
    pos_less = []
    for q in valid_queries(gt):
        mate = int(gt[q])
        base_bad = (all_idx == q) | (all_idx == mate)
        if int(meta[q].get("pair_id", -1)) >= 0:
            same = np.array([same_lensed_system(meta, int(q), int(k)) for k in all_idx], dtype=bool)
            base_bad |= same
        pool = all_idx[~base_bad]
        dt_pos = abs(float(t[q]) - float(t[mate]))
        log_pos = math.log10(dt_pos + 1.0)
        if kind == "uniform":
            chosen = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
            if len(chosen) < n_neg:
                shortage += 1
        elif kind == "hard":
            chosen = None
            chosen_eps = None
            log_neg_all = np.log10(np.abs(t[q] - t[pool]) + 1.0)
            for eps in HARD_EPS_SEQUENCE:
                ok = np.flatnonzero(np.abs(log_neg_all - log_pos) <= eps)
                if len(ok) >= n_neg:
                    chosen = rng.choice(pool[ok], size=n_neg, replace=False)
                    chosen_eps = eps
                    break
            if chosen is None:
                eps = HARD_EPS_SEQUENCE[-1]
                ok = np.flatnonzero(np.abs(log_neg_all - log_pos) <= eps)
                source = pool[ok] if len(ok) else pool
                replace = len(source) < n_neg
                chosen = rng.choice(source, size=n_neg, replace=replace)
                chosen_eps = eps
                shortage += 1
            used_eps.append(float(chosen_eps))
        else:
            raise ValueError(kind)
        dt_neg = np.abs(t[q] - t[chosen])
        pos_less.extend((dt_pos < dt_neg).astype(np.float32).tolist())
        candidates[int(q)] = np.concatenate([[mate], chosen.astype(np.int32)])
    diag = {
        "candidate_kind": kind,
        "n_queries": int(len(candidates)),
        "target_negatives_per_query": int(n_neg),
        "shortage_queries": int(shortage),
        "mean_used_eps": float(np.mean(used_eps)) if used_eps else np.nan,
        "p_dt_pos_lt_dt_neg": float(np.mean(pos_less)) if pos_less else np.nan,
    }
    return candidates, diag


def row_z(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, EPS)


def pair_values(
    variant: str,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    scores: np.ndarray,
    sky_mu: np.ndarray | None,
    sky_sigma: float | None,
    a: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    if variant == "waveform_only":
        return scores[a, c].astype(np.float32)
    if variant == "time_only":
        return -log10_delta_time_obs(time_obs, a, c)
    if variant == "true_sky_overlap_only":
        return base.true_log_sky_overlap(raw_obs, a, c)
    if variant == "predicted_sky_overlap_only":
        return base.log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma)
    raise ValueError(variant)


def direct_candidate_scores(
    variant: str,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    scores: np.ndarray,
    sky_mu: np.ndarray | None,
    sky_sigma: float | None,
    candidates: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    qs = np.asarray(list(candidates.keys()), dtype=np.int32)
    cand = np.stack([candidates[int(q)] for q in qs]).astype(np.int32)
    a = np.repeat(qs, cand.shape[1])
    c = cand.reshape(-1)
    vals = pair_values(variant, raw_obs, time_obs, scores, sky_mu, sky_sigma, a, c)
    return qs, vals.reshape(cand.shape)


def rerank_features(
    variant: str,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    sky_mu: np.ndarray | None,
    sky_sigma: float | None,
    scores: np.ndarray,
    ranks: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    cols = []
    if "time" in variant:
        cols.append(log10_delta_time_obs(time_obs, a, c))
    if "predicted_sky_overlap" in variant:
        cols.append(base.log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma))
    if "true_sky_overlap" in variant:
        cols.append(base.true_log_sky_overlap(raw_obs, a, c))
    if "waveform" in variant:
        cols.append(scores[a, c].astype(np.float32))
        cols.append((1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32))
    return np.column_stack(cols).astype(np.float32)


def train_reranker(
    variant: str,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    sky_mu: np.ndarray,
    sky_sigma: float,
    scores: np.ndarray,
    ranks: np.ndarray,
    candidates: dict[int, np.ndarray],
    seed: int,
):
    rows_a, rows_c, labels = [], [], []
    for q, cand in candidates.items():
        rows_a.extend([q] * len(cand))
        rows_c.extend(cand.tolist())
        labels.extend([1] + [0] * (len(cand) - 1))
    a = np.asarray(rows_a, dtype=np.int32)
    c = np.asarray(rows_c, dtype=np.int32)
    y = np.asarray(labels, dtype=np.int8)
    x = rerank_features(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
    clf = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1e-4,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x, y)
    return clf, float(roc_auc_score(y, clf.predict_proba(x)[:, 1])), int(len(y)), int(y.sum())


def eval_rerank_candidates(
    variant: str,
    clf,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    sky_mu: np.ndarray,
    sky_sigma: float,
    scores: np.ndarray,
    ranks: np.ndarray,
    candidates: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    qs = np.asarray(list(candidates.keys()), dtype=np.int32)
    cand = np.stack([candidates[int(q)] for q in qs]).astype(np.int32)
    a = np.repeat(qs, cand.shape[1])
    c = cand.reshape(-1)
    x = rerank_features(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
    pred = clf.predict_proba(x)[:, 1].reshape(cand.shape)
    return qs, pred


def metrics_from_candidate_scores(score_rows: np.ndarray, query_families: np.ndarray) -> dict[str, dict]:
    ranks = 1 + np.sum(score_rows[:, 1:] > score_rows[:, [0]], axis=1)

    def one(mask: np.ndarray) -> dict:
        r = ranks[mask]
        if len(r) == 0:
            return {}
        out = {
            "valid": int(len(r)),
            "r@1": float(np.mean(r <= 1)),
            "r@5": float(np.mean(r <= 5)),
            "r@10": float(np.mean(r <= 10)),
            "r@50": float(np.mean(r <= 50)),
            "r@100": float(np.mean(r <= 100)),
            "median_true_rank": float(np.median(r)),
        }
        usable = max(score_rows.shape[1] - 1, 1)
        for pct in (1, 5, 10):
            k = max(1, int(math.ceil(usable * pct / 100.0)))
            out[f"top_{pct}pct_k"] = k
            out[f"top_{pct}pct"] = float(np.mean(r <= k))
        return out

    overall = one(np.ones(len(ranks), dtype=bool))
    sis = one(query_families == "SIS")
    pm = one(query_families == "PM")
    macro = {"valid": int(overall.get("valid", 0))}
    for key in ["r@1", "r@5", "r@10", "r@50", "r@100", "top_1pct", "top_5pct", "top_10pct", "median_true_rank"]:
        vals = [d[key] for d in (sis, pm) if key in d]
        if vals:
            macro[key] = float(np.mean(vals))
    for pct in (1, 5, 10):
        macro[f"top_{pct}pct_k"] = overall.get(f"top_{pct}pct_k", np.nan)
    return {"overall": overall, "SIS": sis, "PM": pm, "macro": macro}


def add_rows(
    rows: list[dict],
    detector: str,
    mode: str,
    candidate_kind: str,
    variant: str,
    stage: str,
    metrics_by_group: dict[str, dict],
    diag: dict,
    extra: dict | None = None,
) -> None:
    extra = extra or {}
    for group, met in metrics_by_group.items():
        row = {
            "detector": detector,
            "data_mode": mode,
            "catalog": "mixed_SIS_PM",
            "candidate_kind": candidate_kind,
            "subset": group,
            "variant": variant,
            "stage": stage,
            **diag,
            **met,
            **extra,
        }
        rows.append(row)


def run_one(detector: str, mode: str) -> list[dict]:
    base_dir = BASE_OUT_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    out_dir = OUT_ROOT / f"{detector.lower()}_{mode}_hardneg"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = base.make_cfg(detector, mode, base_dir)
    arrays = {fam: base.FamilyArrays(fam, base.ROOTS[(fam, detector)], mode) for fam in FAMILIES}
    splits = {}
    for i, fam in enumerate(FAMILIES):
        splits[fam] = base.split_indices(len(arrays[fam].l1), cfg.seed + i)
        splits[f"{fam}_U"] = base.split_indices(len(arrays[fam].unlensed), cfg.seed + 100 + i)
    model, _ = base.train_or_load_encoder(cfg, arrays, splits)
    train_ds, train_raw, _, _, train_emb, _ = base.split_pack(detector, "train", cfg, arrays, splits, model)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = base.split_pack(detector, "val", cfg, arrays, splits, model)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = base.split_pack(detector, "test", cfg, arrays, splits, model)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    sky_model, sky_sigma, sky_mean_err, sky_med_err = base.fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = base.normalize_vectors(sky_model.predict(val_emb))
    test_sky_mu = base.normalize_vectors(sky_model.predict(test_emb))
    query_family = np.asarray([test_ds.meta[int(q)]["family"] for q in valid_queries(test_gt)])

    val_hard, val_hard_diag = make_candidate_lists(val_time, val_ds.meta, val_gt, "hard", N_NEG, seed=91001)
    rows = []
    for candidate_kind in ("uniform", "hard"):
        test_candidates, diag = make_candidate_lists(test_time, test_ds.meta, test_gt, candidate_kind, N_NEG, seed=92000 + (candidate_kind == "hard"))
        diag = {**diag, "sky_sigma_rad": sky_sigma, "sky_val_mean_angular_error_rad": sky_mean_err, "sky_val_median_angular_error_rad": sky_med_err}
        for variant in DIRECT_VARIANTS:
            qs, score_rows = direct_candidate_scores(variant, test_raw, test_time, test_scores, test_sky_mu, sky_sigma, test_candidates)
            fam = np.asarray([test_ds.meta[int(q)]["family"] for q in qs])
            mets = metrics_from_candidate_scores(score_rows, fam)
            add_rows(rows, detector, mode, candidate_kind, variant, "direct_score", mets, diag)
        for idx, variant in enumerate(RERANK_VARIANTS):
            clf, auc, n_train, n_pos = train_reranker(
                variant, val_raw, val_time, val_sky_mu, sky_sigma, val_scores, val_ranks, val_hard, seed=93000 + idx
            )
            qs, score_rows = eval_rerank_candidates(
                variant, clf, test_raw, test_time, test_sky_mu, sky_sigma, test_scores, test_ranks, test_candidates
            )
            fam = np.asarray([test_ds.meta[int(q)]["family"] for q in qs])
            mets = metrics_from_candidate_scores(score_rows, fam)
            add_rows(
                rows,
                detector,
                mode,
                candidate_kind,
                variant,
                "hard_trained_hgb_rerank",
                mets,
                diag,
                {"val_auc_hard_sampled": auc, "train_examples": n_train, "train_positive": n_pos, **{f"val_{k}": v for k, v in val_hard_diag.items()}},
            )
        pd.DataFrame(rows).to_csv(out_dir / "hard_negative_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "hard_negative_summary.csv", index=False)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    t0 = time.perf_counter()
    for detector, mode in JOBS:
        print("RUN", detector, mode, flush=True)
        all_rows.extend(run_one(detector, mode))
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "time_matched_hard_negative_summary_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "time_matched_hard_negative_summary.csv", index=False)
    for metric in ["r@1", "r@5", "r@10", "top_1pct", "top_5pct", "top_10pct"]:
        df.pivot_table(
            index=["detector", "data_mode", "candidate_kind", "subset"],
            columns="variant",
            values=metric,
            aggfunc="first",
        ).to_csv(OUT_ROOT / f"{metric.replace('@', '')}_pivot.csv")
    summary = {
        "elapsed_s": float(time.perf_counter() - t0),
        "n_negatives_per_query": N_NEG,
        "hard_eps_sequence": HARD_EPS_SEQUENCE,
        "note": "Time feature uses observed trigger_time_obs only: log10(abs(t_i-t_j)+1). Rerankers are trained on validation time-matched hard negatives.",
    }
    (OUT_ROOT / "protocol_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
