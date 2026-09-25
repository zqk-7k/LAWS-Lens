from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.aux_priors import a90_to_sigma_rad, observed_sky_pair_features
from scripts.gwtc import config

inj = importlib.import_module("scripts.gwtc.04_injection_recovery")


DEFAULT_SEEDS = tuple(range(20260617, 20260627))
SECONDS_PER_DAY = 86400.0


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def rank_matrix(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(score.shape[0])[:, None], order] = np.arange(1, score.shape[1] + 1, dtype=np.int32)
    return ranks


def supplemental_pair_matrices(obs: pd.DataFrame) -> dict[str, np.ndarray]:
    t = obs["gps_trigger_time"].to_numpy(dtype=np.float64)
    snr = np.maximum(obs["network_snr"].to_numpy(dtype=np.float64), 1e-8)
    delta_t_days = np.abs(t[:, None] - t[None, :]) / SECONDS_PER_DAY
    snr_ratio = np.maximum(snr[:, None], snr[None, :]) / np.maximum(np.minimum(snr[:, None], snr[None, :]), 1e-8)

    sky_obs = pd.DataFrame({
        "ra_obs": obs["ra_median"].to_numpy(dtype=np.float64),
        "dec_obs": obs["dec_median"].to_numpy(dtype=np.float64),
        "sky_area90_deg2": obs["sky_area_90_deg2"].to_numpy(dtype=np.float64),
    })
    sky_obs["sky_sigma_rad"] = a90_to_sigma_rad(sky_obs["sky_area90_deg2"].to_numpy(dtype=np.float64))
    sky = observed_sky_pair_features(sky_obs)
    return {
        "delta_t_days": delta_t_days.astype(np.float32),
        "snr_ratio": snr_ratio.astype(np.float32),
        "sky_norm_sep": sky["sky_norm_sep"].astype(np.float32),
        "ang_sep_deg": np.degrees(sky["sky_sep_obs"]).astype(np.float32),
    }


def pair_class(left_id, right_id, is_true_pair: bool) -> str:
    left_inj = pd.notna(left_id)
    right_inj = pd.notna(right_id)
    if is_true_pair:
        return "true_injected_pair"
    if not left_inj and not right_inj:
        return "background_pair"
    if left_inj and right_inj:
        return "injection_injection_false"
    return "injection_background_false"


def export_seed(catalog: str, background: pd.DataFrame, k: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    mixed, true_pairs = inj.make_injected_catalog(background, k, seed)
    scores = inj.build_score_matrices(mixed)
    aux = supplemental_pair_matrices(mixed)
    combined_ranks = rank_matrix(scores["combined_time_sky"])

    true_pair_set = {tuple(sorted(pair)) for pair in true_pairs}
    n = len(mixed)
    left, right = np.triu_indices(n, k=1)
    ids = mixed["injected_pair_id"].to_numpy()
    event_names = mixed["event_name"].astype(str).to_numpy()
    is_true = np.fromiter((tuple(sorted((int(i), int(j)))) in true_pair_set for i, j in zip(left, right)), dtype=bool, count=len(left))
    classes = [pair_class(ids[i], ids[j], bool(flag)) for i, j, flag in zip(left, right, is_true)]

    df = pd.DataFrame({
        "catalog": catalog,
        "seed": seed,
        "k_injected_pairs": k,
        "event_i": event_names[left],
        "event_j": event_names[right],
        "idx_i": left.astype(np.int32),
        "idx_j": right.astype(np.int32),
        "is_true_pair": is_true,
        "pair_class": classes,
        "injected_pair_id_i": ids[left],
        "injected_pair_id_j": ids[right],
        "delta_t_days": aux["delta_t_days"][left, right],
        "time_score": scores["time_lr"][left, right],
        "sky_norm_sep": aux["sky_norm_sep"][left, right],
        "sky_step_weight": scores["sky_step"][left, right],
        "sky_log_overlap": scores["sky_log_overlap"][left, right],
        "ang_sep_deg": aux["ang_sep_deg"][left, right],
        "snr_ratio": aux["snr_ratio"][left, right],
        "combined_time_sky": scores["combined_time_sky"][left, right],
        "rank_i_to_j": combined_ranks[left, right].astype(np.int32),
        "rank_j_to_i": combined_ranks[right, left].astype(np.int32),
    })

    records: list[dict] = []
    for score_name in inj.SCORE_NAMES:
        score = scores[score_name]
        ranks = []
        for lidx, ridx in true_pairs:
            ranks.append(inj.target_rank(score, lidx, ridx))
            ranks.append(inj.target_rank(score, ridx, lidx))
        ranks_arr = np.asarray(ranks, dtype=np.int32)
        records.append({
            "catalog": catalog,
            "k_injected_pairs": k,
            "seed": seed,
            "score": score_name,
            "queries": int(len(ranks_arr)),
            "median_rank": float(np.median(ranks_arr)),
            "mean_rank": float(np.mean(ranks_arr)),
            "mrr": float(np.mean(1.0 / ranks_arr)),
            "recall_at_1": float(np.mean(ranks_arr <= 1)),
            "recall_at_5": float(np.mean(ranks_arr <= 5)),
            "recall_at_10": float(np.mean(ranks_arr <= 10)),
            "recall_at_50": float(np.mean(ranks_arr <= 50)),
        })
    return df, records


def validate_against_existing(records: pd.DataFrame, existing_path: Path, mismatch_path: Path) -> None:
    if not existing_path.exists():
        print(f"WARNING missing existing recovery records: {existing_path}")
        return
    existing = pd.read_csv(existing_path)
    keys = ["catalog", "k_injected_pairs", "seed", "score"]
    cols = ["queries", "median_rank", "mean_rank", "mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50"]
    merged = records.merge(existing[keys + cols], on=keys, suffixes=("_new", "_old"), how="left")
    bad_rows = []
    for _, row in merged.iterrows():
        for col in cols:
            old = row[f"{col}_old"]
            new = row[f"{col}_new"]
            if pd.isna(old) or abs(float(new) - float(old)) > 1e-9:
                bad_rows.append({**{key: row[key] for key in keys}, "metric": col, "new": new, "old": old})
    if bad_rows:
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bad_rows).to_csv(mismatch_path, index=False)
        raise RuntimeError(f"Pair diagnostic validation mismatched existing recovery records; wrote {mismatch_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GWTC injection pair-level diagnostics.")
    parser.add_argument("--catalog", default="gwtc3", choices=["gwtc3", "gwtc5"])
    parser.add_argument("--k-injected-pairs", type=int, default=20)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--out", default=str(config.DATA_DIR / "gwtc_injection_pair_diagnostics.csv"))
    parser.add_argument("--records-out", default=str(config.DATA_DIR / "gwtc_injection_pair_diagnostics_recovery_records.csv"))
    parser.add_argument("--mismatch-out", default=str(config.DATA_DIR / "gwtc_injection_pair_diagnostics_mismatch_report.csv"))
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    background = inj.load_observables(args.catalog)
    all_frames = []
    all_records = []
    for seed in seeds:
        print(f"EXPORT_GWTC_PAIR_DIAGNOSTICS catalog={args.catalog} k={args.k_injected_pairs} seed={seed}", flush=True)
        frame, records = export_seed(args.catalog, background, args.k_injected_pairs, seed)
        all_frames.append(frame)
        all_records.extend(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.concat(all_frames, ignore_index=True)
    diagnostics.to_csv(out, index=False)

    records = pd.DataFrame(all_records)
    records_path = Path(args.records_out)
    records.to_csv(records_path, index=False)
    validate_against_existing(records, config.DATA_DIR / "gwtc_injection_recovery_records.csv", Path(args.mismatch_out))

    counts = {
        "path": str(out),
        "rows": int(len(diagnostics)),
        "true_pairs": int(diagnostics["is_true_pair"].sum()),
        "false_pairs": int((~diagnostics["is_true_pair"]).sum()),
        "pair_class_counts": {str(k): int(v) for k, v in diagnostics["pair_class"].value_counts().sort_index().items()},
        "records_path": str(records_path),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
