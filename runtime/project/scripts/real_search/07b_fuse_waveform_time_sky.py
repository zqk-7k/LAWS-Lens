from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from matchgw.pipeline import build_model, embed_eval
from scripts.real_search.common import (
    SECONDS_PER_DAY,
    angular_sep_rad,
    default_run_dir,
    empirical_hist_lr,
    ensure_run_dirs,
    row_z_matrix,
    utc_now,
    write_json,
)


FAMILIES = ("SIS", "PM")
CHANNELS = ("waveform", "time", "sky")
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SOURCE_ROOT = Path("data_generation/ligo_et_snr_like_2000_outputs")


def cfg_from_checkpoint(raw: dict[str, Any]) -> MatchRunConfig:
    allowed = {f.name for f in fields(MatchRunConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        kwargs[key] = Path(value) if key in {"data_root", "out_dir"} else value
    return MatchRunConfig(**kwargs)


def load_gate_model(run_dir: Path, family: str, epochs: int, cpu: bool) -> tuple[torch.nn.Module, MatchRunConfig]:
    ckpt_path = run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = cfg_from_checkpoint(ckpt["config"])
    model = build_model(cfg, in_channels=2)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model.eval().to(device)
    return model, cfg


def liao_delay_samples() -> np.ndarray:
    liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")
    cfg = liao.LIAO_PRIOR_CONFIG["LIGO"]
    delays, _ = liao.extract_liao_delay_snr_pairs(cfg["image_csv"], cfg["snr_threshold"])
    delays = np.asarray(delays, dtype=np.float64)
    delays = delays[np.isfinite(delays) & (delays > 0)]
    if len(delays) < 10:
        raise RuntimeError("Liao delay prior yielded too few samples")
    return delays


def log_bins_for(values: np.ndarray, signal: np.ndarray, min_floor: float, n_bins: int = 80) -> np.ndarray:
    merged = np.concatenate([np.asarray(values, dtype=np.float64), np.asarray(signal, dtype=np.float64)])
    merged = merged[np.isfinite(merged) & (merged > 0)]
    lo = max(float(np.nanmin(merged)), min_floor)
    hi = max(float(np.nanmax(merged)), lo * 1.01)
    return np.geomspace(lo, hi * 1.001, n_bins + 1)


def validation_source_paths(family: str) -> tuple[Path, Path, Path, Path]:
    fam_dir = SOURCE_ROOT / f"{family}_GW_events_LIGO_et_snr_like_2000"
    unl_dir = SOURCE_ROOT / "unlensed_GW_events_LIGO_et_snr_like_2000"
    return (
        fam_dir / "source_samples.csv",
        fam_dir / f"{family}_trigger_time_features.csv",
        unl_dir / "source_samples.csv",
        unl_dir / "unlensed_trigger_time_features.csv",
    )


def sigma_sky_rad_from_snr(snr: float) -> float:
    # H1/L1 observed-sky proxy used only for validation weighting. It is not
    # used for real-data sky scoring, where real HEALPix maps are used.
    snr = max(float(snr), 1.0)
    area90_deg2 = 650.0 * (20.0 / snr) ** 2
    area90_deg2 = float(np.clip(area90_deg2, 20.0, 5000.0))
    r90_deg = math.sqrt(area90_deg2 / math.pi)
    sigma_deg = r90_deg / 2.146
    return math.radians(sigma_deg)


def perturb_sky(ra: float, dec: float, sigma: float, rng: np.random.Generator) -> tuple[float, float]:
    dra = float(rng.normal(0.0, sigma / max(math.cos(dec), 0.2)))
    ddec = float(rng.normal(0.0, sigma))
    ra_obs = (float(ra) + dra) % (2.0 * math.pi)
    dec_obs = float(np.clip(float(dec) + ddec, -0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4))
    return ra_obs, dec_obs


def validation_event_table(family: str, ds: EvaluationSet, seed: int = 20260625) -> pd.DataFrame:
    src_path, trig_path, unl_src_path, unl_trig_path = validation_source_paths(family)
    src = pd.read_csv(src_path)
    trig = pd.read_csv(trig_path)
    unl_src = pd.read_csv(unl_src_path)
    unl_trig = pd.read_csv(unl_trig_path)
    rng = np.random.default_rng(seed + (0 if family == "SIS" else 1000))
    rows = []
    for idx, meta in enumerate(ds.meta):
        source_index = int(meta["source_index"])
        tag = str(meta["tag"])
        if tag == "L1":
            base = src.iloc[source_index]
            obs = trig.iloc[source_index]
            gps = float(obs["trigger_time_obs_1"])
            snr = float(obs["snr_1"])
            true_pair_id = int(meta["pair_id"])
        elif tag == "L2":
            base = src.iloc[source_index]
            obs = trig.iloc[source_index]
            gps = float(obs["trigger_time_obs_2"])
            snr = float(obs["snr_2"])
            true_pair_id = int(meta["pair_id"])
        else:
            base = unl_src.iloc[source_index]
            obs = unl_trig.iloc[source_index]
            gps = float(obs["trigger_time_obs"])
            snr = float(obs["snr"])
            true_pair_id = -1
        sigma = sigma_sky_rad_from_snr(snr)
        ra_obs, dec_obs = perturb_sky(float(base["ra"]), float(base["dec"]), sigma, rng)
        rows.append({
            "event_index": int(idx),
            "family": family,
            "tag": tag,
            "source_index": source_index,
            "true_pair_id": true_pair_id,
            "gps_obs": gps,
            "snr": snr,
            "ra_true": float(base["ra"]),
            "dec_true": float(base["dec"]),
            "ra_obs": ra_obs,
            "dec_obs": dec_obs,
            "sky_sigma_rad": sigma,
        })
    return pd.DataFrame(rows)


def build_validation_pair_table(run_dir: Path, family: str, epochs: int, cpu: bool, delay_prior: np.ndarray) -> pd.DataFrame:
    model, cfg = load_gate_model(run_dir, family, epochs, cpu=cpu)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits["lensed"]["val"], splits["unlensed"]["val"], cfg)
    emb = embed_eval(model, ds, cfg, cpu=cpu)
    wf = similarity_matrix(emb)
    gt = ground_truth_partner(ds.meta)
    events = validation_event_table(family, ds)
    ii, jj = np.triu_indices(len(events), k=1)
    dt_days = np.abs(events["gps_obs"].to_numpy()[ii] - events["gps_obs"].to_numpy()[jj]) / SECONDS_PER_DAY
    bins = log_bins_for(dt_days, delay_prior, min_floor=1e-4, n_bins=80)
    time_score = empirical_hist_lr(dt_days, delay_prior, dt_days, bins)
    theta = angular_sep_rad(
        events["ra_obs"].to_numpy()[ii],
        events["dec_obs"].to_numpy()[ii],
        events["ra_obs"].to_numpy()[jj],
        events["dec_obs"].to_numpy()[jj],
    )
    sigma = np.sqrt(events["sky_sigma_rad"].to_numpy()[ii] ** 2 + events["sky_sigma_rad"].to_numpy()[jj] ** 2)
    norm_sep = theta / np.maximum(sigma, 1e-8)
    sky_score = (-0.5 * norm_sep ** 2).astype(np.float32)
    labels = (gt[ii] == jj).astype(np.int8)
    out = pd.DataFrame({
        "family": family,
        "idx_i": ii.astype(np.int32),
        "idx_j": jj.astype(np.int32),
        "is_true_pair": labels,
        "waveform_score": wf[ii, jj].astype(np.float32),
        "time_score": time_score.astype(np.float32),
        "sky_score": sky_score.astype(np.float32),
        "delta_t_days": dt_days.astype(np.float64),
        "sky_norm_sep": norm_sep.astype(np.float32),
    })
    out["event_count"] = len(events)
    return out


def matrix_from_pairs(df: pd.DataFrame, n: int, column: str) -> np.ndarray:
    mat = np.full((n, n), np.nan, dtype=np.float64)
    ii = df["idx_i"].to_numpy(dtype=np.int32)
    jj = df["idx_j"].to_numpy(dtype=np.int32)
    vv = df[column].to_numpy(dtype=np.float64)
    mat[ii, jj] = vv
    mat[jj, ii] = vv
    return mat


def row_z_neutral(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64).copy()
    np.fill_diagonal(arr, np.nan)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    out = (arr - mu) / np.maximum(sd, 1e-8)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def validation_metrics_for_weights(validation: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    rows = []
    for family, df in validation.groupby("family"):
        n = int(df["event_count"].iloc[0])
        label_mat = np.full((n, n), False)
        ii = df["idx_i"].to_numpy(dtype=np.int32)
        jj = df["idx_j"].to_numpy(dtype=np.int32)
        label_mat[ii, jj] = df["is_true_pair"].to_numpy(dtype=bool)
        label_mat[jj, ii] = label_mat[ii, jj]
        score = np.zeros((n, n), dtype=np.float32)
        for ch in CHANNELS:
            mat = matrix_from_pairs(df, n, f"{ch}_score")
            score += float(weights[ch]) * row_z_neutral(mat)
        np.fill_diagonal(score, -np.inf)
        query_indices = np.where(label_mat.any(axis=1))[0]
        ranks = []
        for q in query_indices:
            partner = int(np.where(label_mat[q])[0][0])
            order = np.argsort(score[q])[::-1]
            rank = int(np.where(order == partner)[0][0]) + 1
            ranks.append(rank)
        ranks_arr = np.asarray(ranks, dtype=np.int32)
        rows.append({
            "family": family,
            "r_at_1": float(np.mean(ranks_arr <= 1)),
            "r_at_5": float(np.mean(ranks_arr <= 5)),
            "r_at_10": float(np.mean(ranks_arr <= 10)),
            "median_rank": float(np.median(ranks_arr)),
        })
    out: dict[str, float] = {}
    for k in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
        out[f"macro_{k}"] = float(np.mean([r[k] for r in rows]))
    for r in rows:
        for k, v in r.items():
            if k != "family":
                out[f"{r['family'].lower()}_{k}"] = float(v)
    return out


def select_weights(validation: pd.DataFrame, force_waveform_zero: bool = False) -> tuple[dict[str, float], pd.DataFrame]:
    grid_rows = []
    for ww in WEIGHT_GRID:
        if force_waveform_zero and ww != 0.0:
            continue
        for wt in WEIGHT_GRID:
            for ws in WEIGHT_GRID:
                if ww == 0.0 and wt == 0.0 and ws == 0.0:
                    continue
                weights = {"waveform": ww, "time": wt, "sky": ws}
                metrics = validation_metrics_for_weights(validation, weights)
                grid_rows.append({**weights, **metrics})
    grid = pd.DataFrame(grid_rows)
    grid = grid.sort_values(
        ["macro_r_at_10", "macro_r_at_5", "macro_r_at_1", "waveform", "time", "sky"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    best = grid.iloc[0]
    weights = {"waveform": float(best["waveform"]), "time": float(best["time"]), "sky": float(best["sky"])}
    return weights, grid


def add_score_columns(real: pd.DataFrame, weights: dict[str, float], output_name: str) -> pd.DataFrame:
    n = int(max(real["idx_i"].max(), real["idx_j"].max()) + 1)
    out = real.copy()
    mapping = {"waveform": "waveform_score", "time": "time_score", "sky": "sky_score"}
    channel_z = {}
    for ch, col in mapping.items():
        mat = matrix_from_pairs(out, n, col)
        z = row_z_neutral(mat)
        channel_z[ch] = z
        ii = out["idx_i"].to_numpy(dtype=np.int32)
        jj = out["idx_j"].to_numpy(dtype=np.int32)
        out[f"{ch}_rowz_i_to_j"] = z[ii, jj]
        out[f"{ch}_rowz_j_to_i"] = z[jj, ii]
        out[f"{ch}_rowz_mean"] = 0.5 * (z[ii, jj] + z[jj, ii])
    score = np.zeros((n, n), dtype=np.float32)
    for ch in CHANNELS:
        score += float(weights[ch]) * channel_z[ch]
    ii = out["idx_i"].to_numpy(dtype=np.int32)
    jj = out["idx_j"].to_numpy(dtype=np.int32)
    out[f"{output_name}_score_i_to_j"] = score[ii, jj]
    out[f"{output_name}_score_j_to_i"] = score[jj, ii]
    out["final_score"] = np.maximum(score[ii, jj], score[jj, ii])
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["empirical_catalog_tail_p"] = out["rank"] / float(len(out))
    return out


def shortlist(scores: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    cols = [
        "rank", "event_i", "event_j", "final_score",
        "waveform_score", "time_score", "healpix_sky_score",
        "delta_t_days", "healpix_overlap", "waveform_available",
        "detector_coverage", "snr_ratio", "snr_score",
        "empirical_catalog_tail_p",
    ]
    out = scores.head(n).copy()
    out["healpix_sky_score"] = out["sky_score"]
    out["healpix_overlap"] = out["sky_cosine_overlap"]
    out["detector_coverage"] = out["detectors_i"].astype(str) + " | " + out["detectors_j"].astype(str)
    return out[cols]


def pair_rank(scores: pd.DataFrame, a: str, b: str) -> int:
    mask = ((scores["event_i"] == a) & (scores["event_j"] == b)) | ((scores["event_i"] == b) & (scores["event_j"] == a))
    if not mask.any():
        return -1
    return int(scores.loc[mask].iloc[0]["rank"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    delay_prior = liao_delay_samples()
    validation_parts = [build_validation_pair_table(run_dir, fam, args.epochs, args.cpu, delay_prior) for fam in FAMILIES]
    validation = pd.concat(validation_parts, ignore_index=True)
    validation.to_parquet(run_dir / "results" / "fusion_validation_waveform_time_sky.parquet", index=False)

    weights, grid = select_weights(validation, force_waveform_zero=False)
    baseline_weights, baseline_grid = select_weights(validation, force_waveform_zero=True)
    grid.to_csv(run_dir / "results" / "fusion_weight_grid_waveform_time_sky.csv", index=False)
    baseline_grid.to_csv(run_dir / "results" / "fusion_weight_grid_time_sky_baseline.csv", index=False)

    obs = pd.read_parquet(run_dir / "features" / "real_pair_observable_features.parquet")
    sim = pd.read_parquet(run_dir / "features" / "real_waveform_similarity.parquet")
    real = obs.merge(sim[["idx_i", "idx_j", "waveform_available", "waveform_score", "waveform_score_sis", "waveform_score_pm"]], on=["idx_i", "idx_j"], how="left")
    real["waveform_score"] = real["waveform_score"].where(real["waveform_available"].fillna(False), np.nan)

    selected_scores = add_score_columns(real, weights, "waveform_time_sky")
    baseline_scores = add_score_columns(real, baseline_weights, "time_sky")
    forced_scores = add_score_columns(real, {"waveform": 1.0, "time": 1.0, "sky": 1.0}, "forced_equal_waveform_time_sky")

    selected_scores.to_parquet(run_dir / "results" / "real_pair_scores_waveform_time_sky.parquet", index=False)
    selected_scores.to_parquet(run_dir / "real_pair_scores_waveform_time_sky.parquet", index=False)
    baseline_scores.to_parquet(run_dir / "results" / "real_pair_scores_time_sky_baseline.parquet", index=False)
    forced_scores.to_parquet(run_dir / "results" / "real_pair_scores_forced_equal_weights_waveform_time_sky.parquet", index=False)

    main_short = shortlist(selected_scores)
    base_short = shortlist(baseline_scores)
    forced_short = shortlist(forced_scores)
    main_short.to_csv(run_dir / "results" / "candidate_shortlist_waveform_time_sky.csv", index=False)
    main_short.to_csv(run_dir / "candidate_shortlist_waveform_time_sky.csv", index=False)
    base_short.to_csv(run_dir / "results" / "candidate_shortlist_time_sky_baseline.csv", index=False)
    forced_short.to_csv(run_dir / "results" / "candidate_shortlist_forced_equal_weights_waveform_time_sky.csv", index=False)

    wf_values = real["waveform_score"].dropna()
    payload = {
        "generated_at_utc": utc_now(),
        "method": "validation-selected fusion for real-data search using waveform + time-delay + real HEALPix sky-overlap only",
        "snr_policy": "SNR/amplitude is retained only as audit/supplementary diagnostic columns and is not used in final_score or shortlist rank.",
        "validation_set": "Held-out real-noise injection validation catalogs from SIS and PM Gate-1 encoders; time/sky validation channels use the same simulated source metadata with observed-time and H1/L1 observed-sky proxy. Real-data sky scores use real HEALPix posterior maps.",
        "weight_grid": list(WEIGHT_GRID),
        "selected_weights": weights,
        "time_sky_baseline_weights": baseline_weights,
        "forced_equal_weights": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
        "selected_validation_metrics": grid.iloc[0].to_dict(),
        "time_sky_baseline_validation_metrics": baseline_grid.iloc[0].to_dict(),
        "validation_waveform_weight_selected": float(weights["waveform"]),
        "real_waveform_score_audit": {
            "available_pairs": int(real["waveform_available"].fillna(False).sum()),
            "non_missing_scores": int(wf_values.shape[0]),
            "n_unique_scores": int(wf_values.nunique()) if not wf_values.empty else 0,
            "min": float(wf_values.min()) if not wf_values.empty else None,
            "max": float(wf_values.max()) if not wf_values.empty else None,
            "std": float(wf_values.std()) if not wf_values.empty else None,
            "note": "If real waveform scores are nearly constant, waveform contributes little after row-standardization even when validation selects a positive weight.",
        },
        "gw170104_gw170814_rank": {
            "waveform_time_sky": pair_rank(selected_scores, "GW170104", "GW170814"),
            "time_sky_baseline": pair_rank(baseline_scores, "GW170104", "GW170814"),
            "forced_equal_waveform_time_sky": pair_rank(forced_scores, "GW170104", "GW170814"),
        },
        "outputs": {
            "candidate_shortlist_waveform_time_sky": str(run_dir / "results" / "candidate_shortlist_waveform_time_sky.csv"),
            "real_pair_scores_waveform_time_sky": str(run_dir / "results" / "real_pair_scores_waveform_time_sky.parquet"),
            "channel_weights_waveform_time_sky": str(run_dir / "results" / "channel_weights_waveform_time_sky.json"),
            "candidate_shortlist_time_sky_baseline": str(run_dir / "results" / "candidate_shortlist_time_sky_baseline.csv"),
            "candidate_shortlist_forced_equal_weights_waveform_time_sky": str(run_dir / "results" / "candidate_shortlist_forced_equal_weights_waveform_time_sky.csv"),
        },
    }
    write_json(run_dir / "results" / "channel_weights_waveform_time_sky.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
