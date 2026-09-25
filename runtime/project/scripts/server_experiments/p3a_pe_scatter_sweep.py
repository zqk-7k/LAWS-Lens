#!/usr/bin/env python3
"""P3-A PE-scatter closure sweep.

This script audits the corrected P3-A posterior shortlist under realistic
per-event PE measurement scatter.  It intentionally leaves
observable_simulator.py unchanged, then creates observed PE-summary columns by
perturbing each event independently before scoring posterior-summary
consistency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.server_experiments.observable_simulator import simulate_catalog, T_START, T_END
from scripts.server_experiments.rerank_engine import TimeDelayPrior
from scripts.server_experiments.p3a_end_to_end_confirmation import (
    FOLLOWUP_CURVE_BUDGETS,
    count_true,
    hnsw_edges,
    make_embedding,
    pair_channel_values,
    physical_score,
    posterior_param_score,
    posterior_param_terms,
    true_pair_set,
    zscore,
)

WINDOW_S = T_END - T_START
DEFAULT_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def apply_pe_scatter(
    df: pd.DataFrame,
    seed: int,
    log_mc_sigma: float,
    q_sigma: float,
    log_dl_sigma: float,
) -> pd.DataFrame:
    """Return a copy whose PE-summary columns are independently scattered."""
    rng = np.random.default_rng(seed + 91357)
    out = df.copy()
    out["true_chirp_mass"] = out["chirp_mass"].to_numpy(float)
    out["true_mass_ratio"] = out["mass_ratio"].to_numpy(float)
    out["true_luminosity_distance"] = out["luminosity_distance"].to_numpy(float)

    log_mc = np.log(out["chirp_mass"].to_numpy(float))
    q = out["mass_ratio"].to_numpy(float)
    log_dl = np.log(out["luminosity_distance"].to_numpy(float))
    out["chirp_mass"] = np.exp(log_mc + rng.normal(0.0, log_mc_sigma, len(out)))
    out["mass_ratio"] = np.clip(q + rng.normal(0.0, q_sigma, len(out)), 0.05, 1.0)
    out["luminosity_distance"] = np.exp(log_dl + rng.normal(0.0, log_dl_sigma, len(out)))
    out["pe_log_mc_sigma"] = log_mc_sigma
    out["pe_q_sigma"] = q_sigma
    out["pe_log_dl_sigma"] = log_dl_sigma
    return out


def posterior_terms_with_widths(
    df: pd.DataFrame,
    pairs: np.ndarray,
    log_mc_width: float,
    q_width: float,
    log_dl_width: float,
) -> dict[str, np.ndarray]:
    i = pairs[:, 0]
    j = pairs[:, 1]
    log_mc = np.log(df["chirp_mass"].to_numpy(float))
    q = df["mass_ratio"].to_numpy(float)
    log_dl = np.log(df["luminosity_distance"].to_numpy(float))
    return {
        "post_log_mc": -0.5 * ((log_mc[i] - log_mc[j]) / log_mc_width) ** 2,
        "post_q": -0.5 * ((q[i] - q[j]) / q_width) ** 2,
        "post_log_dl": -0.5 * ((log_dl[i] - log_dl[j]) / log_dl_width) ** 2,
    }


def confirmation_score_from_components(
    channels: dict[str, np.ndarray],
    post_terms: dict[str, np.ndarray],
    time_weight: float,
    sky_weight: float,
    posterior_weight: float,
) -> np.ndarray:
    post = posterior_param_score(post_terms)
    return (
        posterior_weight * post
        + sky_weight * zscore(channels["sky_logoverlap"])
        + time_weight * zscore(channels["time_lr"])
    )


def score_rank_stats(score: np.ndarray, pairs: np.ndarray, truth: set[tuple[int, int]]) -> dict[str, float]:
    ranks = []
    pair_to_index = {(int(a), int(b)): idx for idx, (a, b) in enumerate(pairs)}
    for pair in truth:
        idx = pair_to_index.get(pair)
        if idx is not None:
            ranks.append(1 + int(np.sum(score > score[idx])))
    arr = np.asarray(ranks, dtype=float)
    return {
        "hnsw_truth_pairs_available": int(len(arr)),
        "true_rank_median": float(np.median(arr)) if len(arr) else float("nan"),
        "true_rank_p90": float(np.percentile(arr, 90)) if len(arr) else float("nan"),
        "true_rank_max": float(np.max(arr)) if len(arr) else float("nan"),
    }


def evaluate_mode(
    name: str,
    shortlist_score: np.ndarray,
    rank_score: np.ndarray,
    pairs: np.ndarray,
    truth: set[tuple[int, int]],
    shortlist_budget: int,
) -> tuple[list[dict], dict]:
    order = np.argsort(-shortlist_score)
    shortlist_n = min(shortlist_budget, len(order))
    shortlist_idx = order[:shortlist_n]
    shortlist_pairs = pairs[shortlist_idx]
    shortlist_tp = count_true(shortlist_pairs, truth)
    rank_order = np.argsort(-rank_score[shortlist_idx])
    rows = []
    for budget in FOLLOWUP_CURVE_BUDGETS:
        b = min(budget, len(rank_order))
        selected = shortlist_pairs[rank_order[:b]]
        tp = count_true(selected, truth)
        rows.append(
            {
                "mode": name,
                "followup_budget": int(b),
                "true_pairs": int(tp),
                "recall": float(tp / max(len(truth), 1)),
                "precision": float(tp / max(b, 1)),
                "fdr": float(1.0 - tp / max(b, 1)),
            }
        )
    summary = {
        "mode": name,
        "shortlist_budget": int(shortlist_n),
        "shortlist_true_pairs": int(shortlist_tp),
        "shortlist_recall": float(shortlist_tp / max(len(truth), 1)),
        "shortlist_precision": float(shortlist_tp / max(shortlist_n, 1)),
    }
    return rows, summary


def harmonic(recall: float, precision: float) -> float:
    return 0.0 if recall + precision <= 0 else 2.0 * recall * precision / (recall + precision)


def run_one_seed(args: argparse.Namespace, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    n_sis = args.n_true_pairs // 2
    n_pm = args.n_true_pairs - n_sis
    n_bg = int(args.n_background) if args.n_background is not None else int(round(args.n_true_pairs / args.lens_fraction))
    df_true = simulate_catalog(n_sis, n_pm, n_bg, detector="ET3", seed=seed, snr_threshold=8.0)
    truth = true_pair_set(df_true)
    delays = [abs(df_true.loc[a, "geocent_time"] - df_true.loc[b, "geocent_time"]) for a, b in truth]
    prior = TimeDelayPrior(delays, window_s=WINDOW_S)

    emb = make_embedding(df_true, args.embedding_dim, seed)
    pairs, timing = hnsw_edges(emb, args.topk, args.hnsw_m, args.ef_construction, args.ef_search)
    hnsw_tp = count_true(pairs, truth)

    df_pe = apply_pe_scatter(df_true, seed, args.pe_log_mc_sigma, args.pe_q_sigma, args.pe_log_dl_sigma)
    channels = pair_channel_values(df_true, pairs, prior)
    phys = physical_score(channels)
    post_terms = posterior_terms_with_widths(
        df_pe,
        pairs,
        args.posterior_log_mc_width,
        args.posterior_q_width,
        args.posterior_log_dl_width,
    )
    post = posterior_param_score(post_terms)

    score_bank: dict[str, np.ndarray] = {
        "physical": phys,
        "posterior_pe": post,
    }
    for w in args.fusion_weights:
        score_bank[f"physical_plus_{w:g}post"] = phys + float(w) * post
    score_bank["confirm_pe"] = confirmation_score_from_components(
        channels,
        post_terms,
        time_weight=args.confirm_time_weight,
        sky_weight=args.confirm_sky_weight,
        posterior_weight=args.confirm_posterior_weight,
    )

    rows = []
    summaries = []
    diag_rows = []
    for shortlist_name, shortlist_score in score_bank.items():
        stats = score_rank_stats(shortlist_score, pairs, truth)
        stats.update({"seed": seed, "mode": f"shortlist={shortlist_name}"})
        diag_rows.append(stats)
        for rank_name, rank_score in score_bank.items():
            name = f"shortlist={shortlist_name}|rank={rank_name}"
            curve, summary = evaluate_mode(name, shortlist_score, rank_score, pairs, truth, args.shortlist_budget)
            for row in curve:
                row.update(
                    {
                        "seed": seed,
                        "N_events": int(len(df_true)),
                        "n_true_pairs": int(len(truth)),
                        "hnsw_true_pairs": int(hnsw_tp),
                        "hnsw_recall": float(hnsw_tp / max(len(truth), 1)),
                        "pe_log_mc_sigma": args.pe_log_mc_sigma,
                        "pe_q_sigma": args.pe_q_sigma,
                        "pe_log_dl_sigma": args.pe_log_dl_sigma,
                        "shortlist_score": shortlist_name,
                        "rank_score": rank_name,
                    }
                )
            summary.update(
                {
                    "seed": seed,
                    "N_events": int(len(df_true)),
                    "n_true_pairs": int(len(truth)),
                    "hnsw_true_pairs": int(hnsw_tp),
                    "hnsw_recall": float(hnsw_tp / max(len(truth), 1)),
                    "total_seed_wall_s": float(time.perf_counter() - t0),
                    "shortlist_score": shortlist_name,
                    "rank_score": rank_name,
                }
            )
            rows.extend(curve)
            summaries.append(summary)

    return pd.DataFrame(rows), pd.DataFrame(summaries), pd.DataFrame(diag_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-true-pairs", type=int, default=60)
    ap.add_argument("--lens-fraction", type=float, default=1e-3)
    ap.add_argument("--n-background", type=int, default=110000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--shortlist-budget", type=int, default=5000)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=256)
    ap.add_argument("--ef-search", type=int, default=256)
    ap.add_argument("--pe-log-mc-sigma", type=float, default=0.03)
    ap.add_argument("--pe-q-sigma", type=float, default=0.15)
    ap.add_argument("--pe-log-dl-sigma", type=float, default=0.40)
    ap.add_argument("--posterior-log-mc-width", type=float, default=0.05)
    ap.add_argument("--posterior-q-width", type=float, default=0.15)
    ap.add_argument("--posterior-log-dl-width", type=float, default=0.35)
    ap.add_argument("--fusion-weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    ap.add_argument("--confirm-time-weight", type=float, default=0.5)
    ap.add_argument("--confirm-sky-weight", type=float, default=1.0)
    ap.add_argument("--confirm-posterior-weight", type=float, default=1.0)
    ap.add_argument("--select-budget", type=int, default=50)
    ap.add_argument("--out", default="runs/p3a_pe_scatter_sweep_20260620")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    curves = []
    summaries = []
    diagnostics = []
    for seed in args.seeds:
        print(f"run seed={seed}", flush=True)
        curve, summary, diag = run_one_seed(args, seed)
        curve.to_csv(out / f"curve_seed{seed}.csv", index=False)
        summary.to_csv(out / f"summary_seed{seed}.csv", index=False)
        diag.to_csv(out / f"diagnostics_seed{seed}.csv", index=False)
        curves.append(curve)
        summaries.append(summary)
        diagnostics.append(diag)

    curve_all = pd.concat(curves, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)
    diag_all = pd.concat(diagnostics, ignore_index=True)
    curve_all.to_csv(out / "p3a_pe_scatter_curve.csv", index=False)
    summary_all.to_csv(out / "p3a_pe_scatter_summary.csv", index=False)
    diag_all.to_csv(out / "p3a_pe_scatter_diagnostics.csv", index=False)

    grouped = (
        curve_all.groupby(["mode", "followup_budget"])[["recall", "precision", "fdr", "true_pairs"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.to_csv(out / "p3a_pe_scatter_curve_mean.csv", index=False)
    shortlist_grouped = (
        summary_all.groupby("mode")[["shortlist_recall", "shortlist_precision", "hnsw_recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    shortlist_grouped.to_csv(out / "p3a_pe_scatter_shortlist_mean.csv", index=False)
    diag_grouped = (
        diag_all.groupby("mode")[["true_rank_median", "true_rank_p90", "true_rank_max"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    diag_grouped.to_csv(out / "p3a_pe_scatter_diagnostics_mean.csv", index=False)

    sel = curve_all[curve_all["followup_budget"] == args.select_budget].copy()
    score = pd.Series(
        {
            mode: harmonic(group["recall"].mean(), group["precision"].mean())
            for mode, group in sel.groupby("mode")
        }
    )
    best_mode = str(score.sort_values(ascending=False).index[0])
    best = sel[sel["mode"] == best_mode]
    best_summary = {
        "selected_budget": args.select_budget,
        "best_mode": best_mode,
        "best_mean_recall": float(best["recall"].mean()),
        "best_mean_precision": float(best["precision"].mean()),
        "best_mean_fdr": float(best["fdr"].mean()),
        "best_f1_like": float(score.loc[best_mode]),
    }
    (out / "best_mode.json").write_text(json.dumps(best_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(best_summary, indent=2, sort_keys=True), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
