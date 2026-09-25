#!/usr/bin/env python3
"""P3-A: end-to-end sparse triage plus fast confirmation closure.

This script addresses NC gating checklist item 1.  It runs a labelled,
observable-level ET3 catalog at realistic rarity (default f_lens=1e-3) through:

  synthetic waveform-like embedding -> HNSW candidate generation -> cheap
  shortlist -> posterior-summary fast-confirmation surrogate.

The confirmation stage is intentionally labelled as a fast posterior-summary
surrogate, not a full lensing Bayes factor.  Its role is to quantify whether
cheap triage can reduce the all-pairs problem to a tractable follow-up set while
retaining true lensed doublets.

Important: the plain posterior shortlist uses exact simulator intrinsic
parameters unless PE scatter is added externally.  For realistic PE-uncertainty
audits, use p3a_pe_scatter_sweep.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.server_experiments.observable_simulator import simulate_catalog, T_START, T_END
from scripts.server_experiments.rerank_engine import TimeDelayPrior, a90_to_sigma_rad, true_partner_map

WINDOW_S = T_END - T_START
WEIGHTS_PHYSICAL = {"time_lr": 1.0, "sky_step": 4.0, "sky_logoverlap": 1.0, "snr_ratio": 0.25}
FOLLOWUP_CURVE_BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000]
SHORTLIST_DIAGNOSTIC_BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def standardize_cols(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-9)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def make_embedding(df: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """Synthetic waveform-like embeddings from intrinsic parameters only.

    Lensed images share intrinsic mass/distance summaries but differ in SNR and
    observed sky.  We deliberately exclude observed sky and trigger time so this
    stage resembles waveform candidate generation, leaving physical consistency
    to the reranker.
    """
    rng = np.random.default_rng(seed + 1729)
    base = np.column_stack([
        np.log(df["chirp_mass"].to_numpy(float)),
        df["mass_ratio"].to_numpy(float),
        np.log(df["luminosity_distance"].to_numpy(float)),
        np.log(np.abs(df["mu"].to_numpy(float)) + 1e-6),
        np.log(df["network_snr"].to_numpy(float) + 1e-6),
    ]).astype("float32")
    base = standardize_cols(base)
    proj = rng.normal(0, 1, size=(base.shape[1], dim)).astype("float32")
    emb = base @ proj
    emb += rng.normal(0, 0.20, size=emb.shape).astype("float32")
    return l2_normalize(emb.astype("float32"))


def hnsw_edges(emb: np.ndarray, topk: int, hnsw_m: int, ef_construction: int, ef_search: int):
    if faiss is None:
        raise RuntimeError("faiss is required for P3-A")
    d = emb.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(emb)
    build_s = time.perf_counter() - t0
    index.hnsw.efSearch = max(ef_search, topk + 1)
    t0 = time.perf_counter()
    _, raw = index.search(emb, topk + 1)
    query_s = time.perf_counter() - t0
    q = np.repeat(np.arange(len(emb), dtype=np.int32), topk)
    neigh = np.empty((len(emb), topk), dtype=np.int32)
    for i in range(len(emb)):
        keep = raw[i][raw[i] != i]
        if len(keep) < topk:
            pad = np.full(topk - len(keep), -1, dtype=np.int64)
            keep = np.concatenate([keep, pad])
        neigh[i] = keep[:topk]
    j = neigh.reshape(-1).astype(np.int32)
    ok = j >= 0
    q, j = q[ok], j[ok]
    a = np.minimum(q, j)
    b = np.maximum(q, j)
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    return pairs.astype(np.int32), {"hnsw_build_s": build_s, "hnsw_query_s": query_s}


def pair_channel_values(df: pd.DataFrame, pairs: np.ndarray, prior: TimeDelayPrior) -> dict[str, np.ndarray]:
    i = pairs[:, 0]
    j = pairs[:, 1]
    t = df["geocent_time"].to_numpy(float)
    ra = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    sig = a90_to_sigma_rad(df["sky_area_90_deg2"].to_numpy(float))
    snr = df["network_snr"].to_numpy(float)
    dt = np.abs(t[i] - t[j])
    time_lr = prior.lr_score(dt)
    sd = np.sin(dec[i]) * np.sin(dec[j]) + np.cos(dec[i]) * np.cos(dec[j]) * np.cos(ra[i] - ra[j])
    sep = np.arccos(np.clip(sd, -1.0, 1.0))
    sig2 = sig[i] ** 2 + sig[j] ** 2
    norm_sep = sep / np.sqrt(sig2)
    sky_step = np.where(norm_sep < 1.0, 1.0, np.where(norm_sep < 2.0, 0.5, np.where(norm_sep < 3.0, 0.2, 0.0)))
    sky_logoverlap = -0.5 * sep ** 2 / sig2
    snr_ratio = -np.abs(np.log(snr[i] / snr[j]))
    return {"time_lr": time_lr, "sky_step": sky_step, "sky_logoverlap": sky_logoverlap, "snr_ratio": snr_ratio,
            "sky_norm_sep": norm_sep, "dt_days": dt / 86400.0}


def zscore(v: np.ndarray) -> np.ndarray:
    finite = np.isfinite(v)
    out = np.zeros_like(v, dtype=float)
    out[finite] = (v[finite] - v[finite].mean()) / (v[finite].std() + 1e-9)
    out[~finite] = -10.0
    return out


def physical_score(channels: dict[str, np.ndarray]) -> np.ndarray:
    score = np.zeros(len(next(iter(channels.values()))), dtype=float)
    for name, weight in WEIGHTS_PHYSICAL.items():
        score += weight * zscore(channels[name])
    return score


def posterior_param_terms(df: pd.DataFrame, pairs: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap posterior-summary consistency terms available before confirmation."""
    i = pairs[:, 0]
    j = pairs[:, 1]
    log_mc = np.log(df["chirp_mass"].to_numpy(float))
    q = df["mass_ratio"].to_numpy(float)
    log_dl = np.log(df["luminosity_distance"].to_numpy(float))
    return {
        "post_log_mc": -0.5 * ((log_mc[i] - log_mc[j]) / 0.05) ** 2,
        "post_q": -0.5 * ((q[i] - q[j]) / 0.15) ** 2,
        "post_log_dl": -0.5 * ((log_dl[i] - log_dl[j]) / 0.35) ** 2,
    }


def posterior_param_score(terms: dict[str, np.ndarray]) -> np.ndarray:
    return zscore(terms["post_log_mc"]) + 0.5 * zscore(terms["post_q"]) + 0.5 * zscore(terms["post_log_dl"])


def score_modes(phys: np.ndarray, post: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "physical": phys,
        "posterior": post,
        "physical_posterior": phys + post,
    }


def shortlist_diagnostics(scores: dict[str, np.ndarray], pairs: np.ndarray, truth: set[tuple[int, int]]) -> pd.DataFrame:
    pair_to_index = {(int(a), int(b)): idx for idx, (a, b) in enumerate(pairs)}
    rows = []
    for mode, score in scores.items():
        order = np.argsort(-score)
        ranks = []
        for pair in truth:
            idx = pair_to_index.get(pair)
            if idx is None:
                continue
            ranks.append(1 + int(np.sum(score > score[idx])))
        finite_ranks = np.asarray(ranks, dtype=float)
        for budget in SHORTLIST_DIAGNOSTIC_BUDGETS:
            b = min(budget, len(order))
            selected = pairs[order[:b]]
            tp = count_true(selected, truth)
            rows.append({
                "score_mode": mode,
                "budget": int(b),
                "true_pairs": int(tp),
                "recall": float(tp / max(len(truth), 1)),
                "precision": float(tp / max(b, 1)),
                "hnsw_truth_pairs_available": int(len(finite_ranks)),
                "true_rank_median": float(np.median(finite_ranks)) if len(finite_ranks) else float("nan"),
                "true_rank_p90": float(np.percentile(finite_ranks, 90)) if len(finite_ranks) else float("nan"),
                "true_rank_max": float(np.max(finite_ranks)) if len(finite_ranks) else float("nan"),
            })
    return pd.DataFrame(rows)


def confirmation_score(df: pd.DataFrame, pairs: np.ndarray, channels: dict[str, np.ndarray]) -> np.ndarray:
    """Fast posterior-summary confirmation surrogate.

    Uses source-parameter consistency (chirp mass, mass ratio, luminosity
    distance), observed-sky overlap and time prior.  This is a deterministic,
    cheap stand-in for an expensive PE/joint-Bayes confirmation stage.
    """
    i = pairs[:, 0]
    j = pairs[:, 1]
    log_mc = np.log(df["chirp_mass"].to_numpy(float))
    q = df["mass_ratio"].to_numpy(float)
    log_dl = np.log(df["luminosity_distance"].to_numpy(float))
    # Conservative posterior-summary widths used only for ranking.
    mc_term = -0.5 * ((log_mc[i] - log_mc[j]) / 0.05) ** 2
    q_term = -0.5 * ((q[i] - q[j]) / 0.15) ** 2
    dl_term = -0.5 * ((log_dl[i] - log_dl[j]) / 0.35) ** 2
    return zscore(mc_term) + 0.5 * zscore(q_term) + 0.5 * zscore(dl_term) + zscore(channels["sky_logoverlap"]) + 0.5 * zscore(channels["time_lr"])


def true_pair_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    out = set()
    for members in true_partner_map(df).values():
        a, b = members
        out.add((min(a, b), max(a, b)))
    return out


def count_true(pairs: np.ndarray, truth: set[tuple[int, int]]) -> int:
    return sum((int(a), int(b)) in truth for a, b in pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-true-pairs", type=int, default=60)
    ap.add_argument("--lens-fraction", type=float, default=1e-3)
    ap.add_argument("--n-background", type=int, default=None,
                    help="override generated unlensed background count; useful because SNR cuts change realized f_lens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--shortlist-budget", type=int, default=5000)
    ap.add_argument("--followup-budget", type=int, default=200)
    ap.add_argument("--shortlist-mode", choices=["physical", "posterior", "physical_posterior"],
                    default="physical",
                    help="score used to select the cheap shortlist before confirmation")
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=256)
    ap.add_argument("--ef-search", type=int, default=256)
    ap.add_argument("--out", default="runs/p3a_end_to_end_confirmation_20260619")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    n_sis = args.n_true_pairs // 2
    n_pm = args.n_true_pairs - n_sis
    n_bg = int(args.n_background) if args.n_background is not None else int(round(args.n_true_pairs / args.lens_fraction))
    print(f"simulate n_true_pairs={args.n_true_pairs} n_bg={n_bg} seed={args.seed}", flush=True)
    df = simulate_catalog(n_sis, n_pm, n_bg, detector="ET3", seed=args.seed, snr_threshold=8.0)
    truth = true_pair_set(df)
    delays = [abs(df.loc[a, "geocent_time"] - df.loc[b, "geocent_time"]) for a, b in truth]
    prior = TimeDelayPrior(delays, window_s=WINDOW_S)

    emb = make_embedding(df, args.embedding_dim, args.seed)
    pairs, timing = hnsw_edges(emb, args.topk, args.hnsw_m, args.ef_construction, args.ef_search)
    hnsw_tp = count_true(pairs, truth)

    t0 = time.perf_counter()
    channels = pair_channel_values(df, pairs, prior)
    phys = physical_score(channels)
    post_terms = posterior_param_terms(df, pairs)
    post = posterior_param_score(post_terms)
    scores = score_modes(phys, post)
    diagnostics = shortlist_diagnostics(scores, pairs, truth)
    diagnostics.to_csv(out / "p3a_shortlist_diagnostics.csv", index=False)
    physical_s = time.perf_counter() - t0
    shortlist_scores = scores[args.shortlist_mode]
    order = np.argsort(-shortlist_scores)
    shortlist_n = min(args.shortlist_budget, len(order))
    shortlist_idx = order[:shortlist_n]
    shortlist_pairs = pairs[shortlist_idx]
    shortlist_tp = count_true(shortlist_pairs, truth)

    t0 = time.perf_counter()
    confirm = confirmation_score(df, shortlist_pairs, {k: v[shortlist_idx] for k, v in channels.items()})
    confirm_s = time.perf_counter() - t0
    sorted_confirm = np.argsort(-confirm)
    final_n = min(args.followup_budget, len(confirm))
    final_order = sorted_confirm[:final_n]
    final_pairs = shortlist_pairs[final_order]
    final_scores = confirm[final_order]
    final_tp = count_true(final_pairs, truth)
    curve_rows = []
    for budget in FOLLOWUP_CURVE_BUDGETS:
        b = min(budget, len(confirm))
        bpairs = shortlist_pairs[sorted_confirm[:b]]
        tp_b = count_true(bpairs, truth)
        curve_rows.append({
            "followup_budget": int(b),
            "true_pairs": int(tp_b),
            "recall": float(tp_b / max(len(truth), 1)),
            "precision": float(tp_b / max(b, 1)),
            "fdr": float(1.0 - tp_b / max(b, 1)),
        })
    wall_s = time.perf_counter() - wall0

    summary = {
        "N_events": int(len(df)),
        "n_true_pairs": int(len(truth)),
        "n_unlensed": int((df["kind"] == "unlensed").sum()),
        "target_lens_fraction": args.lens_fraction,
        "realized_lens_pair_fraction_vs_events": float(len(truth) / len(df)),
        "hnsw_topk": args.topk,
        "hnsw_candidate_pairs": int(len(pairs)),
        "hnsw_true_pairs_recovered": int(hnsw_tp),
        "hnsw_recall": float(hnsw_tp / max(len(truth), 1)),
        "shortlist_mode": args.shortlist_mode,
        "shortlist_budget": int(shortlist_n),
        "shortlist_true_pairs": int(shortlist_tp),
        "shortlist_recall": float(shortlist_tp / max(len(truth), 1)),
        # Backward-compatible aliases for the original P3-A CSV schema.
        "physical_shortlist_budget": int(shortlist_n),
        "physical_shortlist_true_pairs": int(shortlist_tp),
        "physical_shortlist_recall": float(shortlist_tp / max(len(truth), 1)),
        "followup_budget": int(final_n),
        "final_true_pairs": int(final_tp),
        "final_recall": float(final_tp / max(len(truth), 1)),
        "final_precision": float(final_tp / max(final_n, 1)),
        "final_fdr": float(1.0 - final_tp / max(final_n, 1)),
        "all_pairs_exhaustive": int(len(df) * (len(df) - 1) // 2),
        "candidate_reduction_vs_all_pairs": float((len(df) * (len(df) - 1) / 2) / max(len(pairs), 1)),
        "followup_reduction_vs_all_pairs": float((len(df) * (len(df) - 1) / 2) / max(final_n, 1)),
        "hnsw_build_s": timing["hnsw_build_s"],
        "hnsw_query_s": timing["hnsw_query_s"],
        "physical_score_s": physical_s,
        "confirmation_score_s": confirm_s,
        "total_wall_s": wall_s,
        "confirmation_label": "posterior-summary fast-confirmation surrogate; not full Bayes factor",
    }
    pd.DataFrame([summary]).to_csv(out / "p3a_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "p3a_followup_curve.csv", index=False)
    pd.DataFrame({"pair_i": final_pairs[:, 0], "pair_j": final_pairs[:, 1], "confirm_score": final_scores,
                  "is_true_pair": [(int(a), int(b)) in truth for a, b in final_pairs]}).to_csv(out / "p3a_final_candidates.csv", index=False)
    df.head(2000).to_csv(out / "p3a_catalog_head.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
