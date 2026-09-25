#!/usr/bin/env python3
"""P3-B: domain baseline comparison for NC gating checklist item 2.

Adds a posterior-summary kNN baseline and compares it with a synthetic
waveform-proxy ANN, physical time/sky/SNR ranking, and a simple fusion.  The
synthetic proxy is explicitly labelled because it is a lossy random projection
of clean parameters, not a trained strain encoder.  The script also reports the
native ET3 trained-embedding retrieval baseline when cached embeddings are
available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.server_experiments.observable_simulator import simulate_catalog, T_START, T_END
from scripts.server_experiments.rerank_engine import TimeDelayPrior, a90_to_sigma_rad, true_partner_map
from scripts.server_experiments.p3a_end_to_end_confirmation import make_embedding

WINDOW_S = T_END - T_START
DEFAULT_NATIVE_ET3_EMBEDDINGS = (
    "runs/et3_fresh50_full_catalog_20260616/fresh_mixed_encoders/"
    "et3_noisy_mixed_sis_pm_ep50/test_embeddings.npy"
)
DEFAULT_NATIVE_ET3_META = "runs/p2a_ann_et3_noisy_meta_20260618/catalog_meta.csv"


def zscore(v: np.ndarray) -> np.ndarray:
    finite = np.isfinite(v)
    out = np.zeros_like(v, dtype=float)
    out[finite] = (v[finite] - v[finite].mean()) / (v[finite].std() + 1e-9)
    out[~finite] = -10.0
    return out


def posterior_features(df: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([
        np.log(df["chirp_mass"].to_numpy(float)),
        df["mass_ratio"].to_numpy(float),
        np.log(df["luminosity_distance"].to_numpy(float)),
        np.cos(df["ra"].to_numpy(float)),
        np.sin(df["ra"].to_numpy(float)),
        np.sin(df["dec"].to_numpy(float)),
        np.log(df["network_snr"].to_numpy(float) + 1e-6),
    ]).astype("float32")
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-9)


def partner_queries(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    kinds = df["kind"].to_numpy()
    out = []
    for a, b in true_partner_map(df).values():
        out.append((a, b, kinds[a]))
        out.append((b, a, kinds[a]))
    return out


def ranks_from_neighbor_indices(neigh: dict[int, np.ndarray], queries: list[tuple[int, int, str]]) -> list[int]:
    ranks = []
    for q, partner, _ in queries:
        arr = neigh[q]
        hit = np.where(arr == partner)[0]
        ranks.append(int(hit[0] + 1) if len(hit) else 10**9)
    return ranks


def summarize_ranks(method: str, ranks: list[int], wall_s: float, extra: dict | None = None) -> dict:
    ranks_arr = np.asarray(ranks)
    row = {
        "method": method,
        "n_queries": int(len(ranks_arr)),
        "R@1": float(np.mean(ranks_arr <= 1)),
        "R@5": float(np.mean(ranks_arr <= 5)),
        "R@10": float(np.mean(ranks_arr <= 10)),
        "R@50": float(np.mean(ranks_arr <= 50)),
        "median_rank": float(np.median(ranks_arr[ranks_arr < 10**8])) if np.any(ranks_arr < 10**8) else float("inf"),
        "wall_s": wall_s,
    }
    if extra:
        row.update(extra)
    return row


def add_context(row: dict, dataset: str, n_events: int, n_true_pairs: int, lens_fraction: float | None) -> dict:
    out = {
        "dataset": dataset,
        "N_events": int(n_events),
        "n_true_pairs": int(n_true_pairs),
        "target_lens_fraction": lens_fraction,
    }
    out.update(row)
    return out


def waveform_ann_ranks(df: pd.DataFrame, queries: list[tuple[int, int, str]], topk: int, seed: int):
    if faiss is None:
        raise RuntimeError("faiss is required")
    emb = make_embedding(df, 64, seed)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 256
    index.add(emb)
    index.hnsw.efSearch = max(256, topk + 1)
    qidx = np.asarray([q for q, _, _ in queries], dtype=np.int64)
    _, raw = index.search(emb[qidx], topk + 1)
    neigh = {}
    for row, q in zip(raw, qidx):
        neigh[int(q)] = row[row != q][:topk]
    ranks = ranks_from_neighbor_indices(neigh, queries)
    return ranks, time.perf_counter() - t0


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def native_partner_queries(meta: pd.DataFrame) -> tuple[list[tuple[int, int, str]], int]:
    groups = {}
    for idx, row in meta.iterrows():
        kind = row["kind"]
        if kind in {"SIS", "PM"}:
            groups.setdefault(int(row["source_id"]), []).append((int(idx), kind))
    pairs = [members for members in groups.values() if len(members) == 2]
    queries = []
    for (a, kind_a), (b, _) in pairs:
        queries.append((a, b, kind_a))
        queries.append((b, a, kind_a))
    return queries, len(pairs)


def native_embedding_ann_ranks(emb_path: str, meta_path: str, topk: int):
    if faiss is None:
        raise RuntimeError("faiss is required")
    emb = np.load(emb_path).astype("float32")
    emb = l2_normalize(emb)
    meta = pd.read_csv(meta_path)
    if len(emb) != len(meta):
        raise RuntimeError(f"native embedding/meta length mismatch: {len(emb)} != {len(meta)}")
    queries, n_pairs = native_partner_queries(meta)
    qidx = np.asarray([q for q, _, _ in queries], dtype=np.int64)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 256
    index.add(emb)
    index.hnsw.efSearch = max(512, topk + 1)
    _, raw = index.search(emb[qidx], topk + 1)
    neigh = {}
    for row, q in zip(raw, qidx):
        neigh[int(q)] = row[row != q][:topk]
    ranks = ranks_from_neighbor_indices(neigh, queries)
    return ranks, time.perf_counter() - t0, len(meta), n_pairs


def posterior_knn_ranks(df: pd.DataFrame, queries: list[tuple[int, int, str]], topk: int):
    x = posterior_features(df)
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=topk + 1, algorithm="auto", metric="euclidean")
    nn.fit(x)
    qidx = np.asarray([q for q, _, _ in queries], dtype=np.int64)
    _, ind = nn.kneighbors(x[qidx], return_distance=True)
    neigh = {}
    for row, q in zip(ind, qidx):
        neigh[int(q)] = row[row != q][:topk]
    ranks = ranks_from_neighbor_indices(neigh, queries)
    return ranks, time.perf_counter() - t0


def score_query(q: int, df: pd.DataFrame, prior: TimeDelayPrior, mode: str) -> np.ndarray:
    t = df["geocent_time"].to_numpy(float)
    ra = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    sig = a90_to_sigma_rad(df["sky_area_90_deg2"].to_numpy(float))
    snr = df["network_snr"].to_numpy(float)
    dt = np.abs(t - t[q])
    time_lr = prior.lr_score(dt)
    sd = np.sin(dec) * np.sin(dec[q]) + np.cos(dec) * np.cos(dec[q]) * np.cos(ra - ra[q])
    sep = np.arccos(np.clip(sd, -1.0, 1.0))
    sig2 = sig ** 2 + sig[q] ** 2
    norm_sep = sep / np.sqrt(sig2)
    sky_step = np.where(norm_sep < 1.0, 1.0, np.where(norm_sep < 2.0, 0.5, np.where(norm_sep < 3.0, 0.2, 0.0)))
    sky_log = -0.5 * sep ** 2 / sig2
    snr_ratio = -np.abs(np.log(snr / snr[q]))
    physical = zscore(time_lr) + 4.0 * zscore(sky_step) + zscore(sky_log) + 0.25 * zscore(snr_ratio)
    if mode == "physical_time_sky_snr":
        score = physical
    elif mode == "posterior_summary_plus_physical":
        feats = posterior_features(df)
        dist = np.linalg.norm(feats - feats[q], axis=1)
        score = zscore(-dist) + physical
    else:
        raise ValueError(mode)
    score[q] = -np.inf
    return score


def full_catalog_ranks(df: pd.DataFrame, queries: list[tuple[int, int, str]], prior: TimeDelayPrior, mode: str):
    t0 = time.perf_counter()
    ranks = []
    for q, partner, _ in queries:
        score = score_query(q, df, prior, mode)
        ranks.append(1 + int(np.sum(score > score[partner])))
    return ranks, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-true-pairs", type=int, default=60)
    ap.add_argument("--lens-fraction", type=float, default=1e-3)
    ap.add_argument("--n-background", type=int, default=None,
                    help="override generated unlensed background count; useful because SNR cuts change realized f_lens")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--native-embeddings", default=DEFAULT_NATIVE_ET3_EMBEDDINGS)
    ap.add_argument("--native-meta", default=DEFAULT_NATIVE_ET3_META)
    ap.add_argument("--skip-native", action="store_true")
    ap.add_argument("--out", default="runs/p3b_domain_baselines_20260619")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_sis = args.n_true_pairs // 2
    n_pm = args.n_true_pairs - n_sis
    n_bg = int(args.n_background) if args.n_background is not None else int(round(args.n_true_pairs / args.lens_fraction))
    df = simulate_catalog(n_sis, n_pm, n_bg, detector="ET3", seed=args.seed, snr_threshold=8.0)
    queries = partner_queries(df)
    truth = true_partner_map(df)
    delays = [abs(df.loc[a, "geocent_time"] - df.loc[b, "geocent_time"]) for a, b in truth.values()]
    prior = TimeDelayPrior(delays, window_s=WINDOW_S)

    rows = []
    ranks, wall = waveform_ann_ranks(df, queries, args.topk, args.seed)
    rows.append(add_context(
        summarize_ranks("synthetic_waveform_proxy_HNSW_lossy_projection", ranks, wall, {"topk": args.topk}),
        "synthetic_realistic_rarity_et3",
        len(df),
        len(truth),
        args.lens_fraction,
    ))
    ranks, wall = posterior_knn_ranks(df, queries, args.topk)
    rows.append(add_context(
        summarize_ranks("posterior_summary_kNN", ranks, wall, {"topk": args.topk}),
        "synthetic_realistic_rarity_et3",
        len(df),
        len(truth),
        args.lens_fraction,
    ))
    for mode in ["physical_time_sky_snr", "posterior_summary_plus_physical"]:
        ranks, wall = full_catalog_ranks(df, queries, prior, mode)
        rows.append(add_context(
            summarize_ranks(mode, ranks, wall, {"topk": len(df) - 1}),
            "synthetic_realistic_rarity_et3",
            len(df),
            len(truth),
            args.lens_fraction,
        ))

    if not args.skip_native:
        ranks, wall, native_n, native_pairs = native_embedding_ann_ranks(args.native_embeddings, args.native_meta, args.topk)
        rows.append(add_context(
            summarize_ranks(
                "native_et3_trained_embedding_HNSW",
                ranks,
                wall,
                {"topk": args.topk, "embeddings": args.native_embeddings, "meta": args.native_meta},
            ),
            "native_et3_noisy_test_embeddings",
            native_n,
            native_pairs,
            None,
        ))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "p3b_baseline_summary.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
