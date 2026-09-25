from __future__ import annotations

import argparse
import itertools
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal, stats


REPO = Path("/root/autodl-tmp/gw-catalog")
LATEST = REPO / "results" / "real_noise_injection_v2_20260719"
DEFAULT_OUT = REPO / "results" / "real_candidate_forensic_audit_20260720"
DEFAULT_PACKAGE = REPO / "packages" / "real_candidate_forensic_audit_20260720.tar.gz"
REFERENCE_SEED = 202607191
SEEDS = (202607191, 202607192, 202607193)
DETECTORS = ("H1", "L1")
FS = 4096.0
WINDOW_SECONDS = 24.0
WINDOW_END_OFFSET = 0.25
MODEL_WINDOW_START = -1.75
MODEL_WINDOW_END = 0.25
DISPLAY_WINDOW_START = -4.0
DISPLAY_WINDOW_END = 1.0
POSTERIOR_LIMIT = 30_000
WEIGHT_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
COLORS = {"waveform": "#277DA1", "time": "#F8961E", "sky": "#43AA8B"}


@dataclass(frozen=True)
class Deployment:
    key: str
    label: str
    source_run: Path
    event_manifest: Path
    group_column: str


DEPLOYMENTS = {
    "gwtc3": Deployment(
        key="gwtc3",
        label="GWTC-3 (O1-O3)",
        source_run=REPO / "runs" / "real_gwtc_lensing_search_20260625",
        event_manifest=REPO / "runs" / "real_gwtc_lensing_search_20260625" / "data" / "event_manifest.csv",
        group_column="sky_map_internal_group",
    ),
    "gwtc4": Deployment(
        key="gwtc4",
        label="GWTC-4.1 O4a",
        source_run=REPO / "runs" / "real_gwtc34_lensing_search_20260629_full_o4",
        event_manifest=REPO / "runs" / "real_gwtc34_lensing_search_20260629_full_o4" / "data" / "event_manifest.csv",
        group_column="sky_map_group",
    ),
}


PARAMETERS = {
    "chirp_mass": (r"Detector-frame $\mathcal{M}$", r"$M_\odot$"),
    "chirp_mass_source": (r"Source-frame $\mathcal{M}$", r"$M_\odot$"),
    "mass_ratio": (r"Mass ratio $q$", ""),
    "chi_eff": (r"Effective spin $\chi_{\rm eff}$", ""),
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def primary_manifest(deploy: Deployment) -> pd.DataFrame:
    frame = pd.read_csv(deploy.event_manifest)
    mask = frame["include_in_primary_search"].fillna(False).astype(bool)
    if "sky_map_available" in frame:
        mask &= frame["sky_map_available"].fillna(False).astype(bool)
    return frame.loc[mask].sort_values("gps_time").reset_index(drop=True)


def seed_dir(deployment: str, seed: int) -> Path:
    return LATEST / deployment / f"seed_{seed}"


def score_path(deployment: str, seed: int) -> Path:
    return seed_dir(deployment, seed) / "results" / "real_pair_scores_waveform_time_sky_positive.parquet"


def current_scores(deployment: str, seed: int) -> pd.DataFrame:
    return pd.read_parquet(score_path(deployment, seed)).sort_values("rank").reset_index(drop=True)


def selected_weights(deployment: str, seed: int) -> dict[str, float]:
    path = seed_dir(deployment, seed) / "results" / "selected_weights.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in raw["strictly_positive_waveform_time_sky"].items()}


def score_contributions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            weights = selected_weights(deployment, seed)
            scores = current_scores(deployment, seed).head(5)
            prefix = "waveform_time_sky_positive_score"
            for pair in scores.itertuples(index=False):
                i_score = float(getattr(pair, f"{prefix}_i_to_j"))
                j_score = float(getattr(pair, f"{prefix}_j_to_i"))
                direction = "i_to_j" if i_score >= j_score else "j_to_i"
                record: dict[str, Any] = {
                    "deployment": deployment,
                    "seed": seed,
                    "rank": int(pair.rank),
                    "event_i": pair.event_i,
                    "event_j": pair.event_j,
                    "selected_direction": direction,
                    "directed_score_i_to_j": i_score,
                    "directed_score_j_to_i": j_score,
                    "final_score": float(pair.final_score),
                    "waveform_available": bool(pair.waveform_available),
                    "full_waveform_scored": bool(pair.full_waveform_scored),
                }
                positive_sum = 0.0
                for channel in ("waveform", "time", "sky"):
                    z = float(getattr(pair, f"{channel}_rowz_{direction}"))
                    contribution = weights[channel] * z
                    record[f"{channel}_raw_score"] = float(getattr(pair, f"{channel}_score"))
                    record[f"{channel}_weight"] = weights[channel]
                    record[f"{channel}_rowz_selected_direction"] = z
                    record[f"{channel}_weighted_contribution"] = contribution
                    positive_sum += max(contribution, 0.0)
                for channel in ("waveform", "time", "sky"):
                    value = record[f"{channel}_weighted_contribution"]
                    record[f"{channel}_positive_share"] = max(value, 0.0) / positive_sum if positive_sum else np.nan
                rows.append(record)
    return pd.DataFrame(rows)


def resolve_path(value: Any, roots: list[Path]) -> Path | None:
    if value is None or pd.isna(value):
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [root / raw for root in roots]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0] if candidates else None


def scalar(h5: h5py.File, key: str) -> float:
    return float(np.asarray(h5[key][()]).item())


def detector_window(path: Path, gps: float, start_offset: float, end_offset: float) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as h5:
        dataset = h5["strain/Strain"]
        gps_start = scalar(h5, "meta/GPSstart")
        duration = scalar(h5, "meta/Duration")
        sample_rate = len(dataset) / duration
        i0 = int(round((gps + start_offset - gps_start) * sample_rate))
        i1 = int(round((gps + end_offset - gps_start) * sample_rate))
        if i0 < 0 or i1 > len(dataset) or i1 <= i0:
            raise ValueError(f"window outside file: {path}")
        values = np.asarray(dataset[i0:i1], dtype=np.float64)
    return values, float(sample_rate), float(gps_start)


def strain_rows(deployment: str) -> pd.DataFrame:
    path = seed_dir(deployment, REFERENCE_SEED) / "data" / "strain_gwosc_download_manifest.csv"
    return pd.read_csv(path)


def strain_path_for(deployment: str, event: str, detector: str, manifest: pd.DataFrame) -> Path | None:
    rows = manifest[(manifest["event_name"].astype(str) == event) & (manifest["detector"].astype(str) == detector)]
    if rows.empty:
        return None
    value = rows.iloc[0].get("local_path")
    roots = [seed_dir(deployment, REFERENCE_SEED), REPO, DEPLOYMENTS[deployment].source_run]
    return resolve_path(value, roots)


def audit_exact_windows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deployment, deploy in DEPLOYMENTS.items():
        events = primary_manifest(deploy)
        strain_manifest = strain_rows(deployment)
        embedding_audit = pd.read_parquet(seed_dir(deployment, REFERENCE_SEED) / "features" / "real_waveform_embeddings.parquet")
        old_status = embedding_audit.set_index("event_name")["embedding_available"].to_dict()
        for idx, event_row in events.iterrows():
            event = str(event_row.event_name)
            gps = float(event_row.gps_time)
            fractions: dict[str, float] = {}
            paths: dict[str, str] = {}
            notes: dict[str, str] = {}
            for detector in DETECTORS:
                path = strain_path_for(deployment, event, detector, strain_manifest)
                paths[detector] = str(path) if path else ""
                if path is None or not path.exists():
                    fractions[detector] = 0.0
                    notes[detector] = "missing_file"
                    continue
                try:
                    x, fs, _ = detector_window(path, gps, -23.75, 0.25)
                    fractions[detector] = float(np.mean(np.isfinite(x)))
                    notes[detector] = "ok" if abs(fs - FS) < 1e-3 else f"sample_rate_{fs:g}"
                except Exception as exc:
                    fractions[detector] = 0.0
                    notes[detector] = f"error:{type(exc).__name__}"
            h1 = fractions["H1"] >= 0.99
            l1 = fractions["L1"] >= 0.99
            if h1 and l1:
                pattern = "H1L1"
            elif h1:
                pattern = "H1_only"
            elif l1:
                pattern = "L1_only"
            else:
                pattern = "neither"
            rows.append(
                {
                    "deployment": deployment,
                    "idx": int(idx),
                    "event_name": event,
                    "gps_time": gps,
                    "h1_path": paths["H1"],
                    "l1_path": paths["L1"],
                    "h1_finite_fraction_24s": fractions["H1"],
                    "l1_finite_fraction_24s": fractions["L1"],
                    "h1_status": notes["H1"],
                    "l1_status": notes["L1"],
                    "detector_pattern_exact_window": pattern,
                    "old_embedding_available": bool(old_status.get(event, False)),
                    "corrected_embedding_available": bool(h1 and l1),
                    "old_false_positive_availability": bool(old_status.get(event, False) and not (h1 and l1)),
                }
            )
    return pd.DataFrame(rows)


def effective_rank(vectors: np.ndarray) -> float:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular * singular
    denom = float(np.sum(variance * variance))
    return float(np.sum(variance) ** 2 / denom) if denom > 0.0 else 0.0


def embedding_geometry(window_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    pattern_map = window_audit.set_index(["deployment", "event_name"])["detector_pattern_exact_window"].to_dict()
    valid_map = window_audit.set_index(["deployment", "event_name"])["corrected_embedding_available"].to_dict()
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            emb = pd.read_parquet(seed_dir(deployment, seed) / "features" / "real_waveform_embeddings.parquet")
            sim = pd.read_parquet(seed_dir(deployment, seed) / "features" / "real_waveform_similarity.parquet")
            sim["pattern_i"] = sim["event_i"].map(lambda event: pattern_map.get((deployment, event), "missing"))
            sim["pattern_j"] = sim["event_j"].map(lambda event: pattern_map.get((deployment, event), "missing"))
            sim["corrected_waveform_available"] = [
                bool(valid_map.get((deployment, a), False) and valid_map.get((deployment, b), False))
                for a, b in zip(sim.event_i, sim.event_j)
            ]
            for family in ("sis", "pm"):
                columns = sorted(c for c in emb.columns if c.startswith(f"{family}_emb_"))
                available = emb.loc[emb["embedding_available"].fillna(False), ["event_name", *columns]].copy()
                vectors = available[columns].to_numpy(dtype=np.float64)
                norms = np.linalg.norm(vectors, axis=1)
                mean_norm = float(np.linalg.norm(vectors.mean(axis=0))) if len(vectors) else np.nan
                values = sim.loc[sim["waveform_available"].fillna(False), f"waveform_score_{family}"].dropna().to_numpy(dtype=float)
                summaries.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "family_encoder": family.upper(),
                        "n_old_available_events": int(len(vectors)),
                        "n_corrected_available_events": int(sum(valid_map.get((deployment, e), False) for e in emb.event_name)),
                        "embedding_norm_min": float(np.min(norms)) if len(norms) else np.nan,
                        "embedding_norm_median": float(np.median(norms)) if len(norms) else np.nan,
                        "embedding_norm_max": float(np.max(norms)) if len(norms) else np.nan,
                        "mean_embedding_vector_norm": mean_norm,
                        "centered_effective_rank": effective_rank(vectors) if len(vectors) > 1 else 0.0,
                        "n_embedding_dimensions": int(len(columns)),
                        "pair_cosine_min": float(np.min(values)) if len(values) else np.nan,
                        "pair_cosine_q50": float(np.quantile(values, 0.5)) if len(values) else np.nan,
                        "pair_cosine_q90": float(np.quantile(values, 0.9)) if len(values) else np.nan,
                        "pair_cosine_q99": float(np.quantile(values, 0.99)) if len(values) else np.nan,
                        "pair_cosine_max": float(np.max(values)) if len(values) else np.nan,
                        "pair_cosine_std": float(np.std(values)) if len(values) else np.nan,
                    }
                )
                valid_sim = sim[sim["waveform_available"].fillna(False)].copy()
                valid_sim["same_detector_pattern"] = valid_sim["pattern_i"] == valid_sim["pattern_j"]
                for same, group in valid_sim.groupby("same_detector_pattern"):
                    vals = group[f"waveform_score_{family}"].dropna().to_numpy(dtype=float)
                    leakage.append(
                        {
                            "deployment": deployment,
                            "seed": seed,
                            "family_encoder": family.upper(),
                            "pair_group": "same_detector_pattern" if same else "different_detector_pattern",
                            "n_pairs": int(len(vals)),
                            "cosine_mean": float(np.mean(vals)) if len(vals) else np.nan,
                            "cosine_median": float(np.median(vals)) if len(vals) else np.nan,
                            "cosine_q90": float(np.quantile(vals, 0.9)) if len(vals) else np.nan,
                        }
                    )
                for pattern, group in valid_sim[valid_sim.pattern_i == valid_sim.pattern_j].groupby("pattern_i"):
                    vals = group[f"waveform_score_{family}"].dropna().to_numpy(dtype=float)
                    leakage.append(
                        {
                            "deployment": deployment,
                            "seed": seed,
                            "family_encoder": family.upper(),
                            "pair_group": f"both_{pattern}",
                            "n_pairs": int(len(vals)),
                            "cosine_mean": float(np.mean(vals)) if len(vals) else np.nan,
                            "cosine_median": float(np.median(vals)) if len(vals) else np.nan,
                            "cosine_q90": float(np.quantile(vals, 0.9)) if len(vals) else np.nan,
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(leakage)


def matrix_from_pairs(frame: pd.DataFrame, n: int, column: str) -> np.ndarray:
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    ii = frame["idx_i"].to_numpy(dtype=np.int32)
    jj = frame["idx_j"].to_numpy(dtype=np.int32)
    values = frame[column].to_numpy(dtype=np.float64)
    matrix[ii, jj] = values
    matrix[jj, ii] = values
    return matrix


def row_z_neutral(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(values, np.nan)
    finite = np.isfinite(values)
    count = finite.sum(axis=1, keepdims=True)
    mean = np.divide(
        np.where(finite, values, 0.0).sum(axis=1, keepdims=True),
        count,
        out=np.zeros((len(values), 1), dtype=np.float64),
        where=count > 0,
    )
    squared = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=1, keepdims=True)
    variance = np.divide(
        squared,
        count,
        out=np.zeros((len(values), 1), dtype=np.float64),
        where=count > 0,
    )
    std = np.sqrt(variance)
    output = (values - mean) / np.maximum(std, 1e-8)
    output[~np.isfinite(output)] = 0.0
    return output.astype(np.float32)


def add_pair_scores(frame: pd.DataFrame, weights: dict[str, float], label: str) -> pd.DataFrame:
    out = frame.copy()
    n = int(max(out.idx_i.max(), out.idx_j.max()) + 1)
    ii = out.idx_i.to_numpy(dtype=np.int32)
    jj = out.idx_j.to_numpy(dtype=np.int32)
    total = np.zeros((n, n), dtype=np.float32)
    for channel in ("waveform", "time", "sky"):
        z = row_z_neutral(matrix_from_pairs(out, n, f"{channel}_score"))
        out[f"{channel}_rowz_i_to_j"] = z[ii, jj]
        out[f"{channel}_rowz_j_to_i"] = z[jj, ii]
        total += float(weights[channel]) * z
    out[f"{label}_score_i_to_j"] = total[ii, jj]
    out[f"{label}_score_j_to_i"] = total[jj, ii]
    out["final_score"] = np.maximum(total[ii, jj], total[jj, ii])
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def fit_hist_lr(signal_days: np.ndarray, background_days: np.ndarray, bins: int = 80) -> dict[str, np.ndarray]:
    signal_values = np.asarray(signal_days, dtype=np.float64)
    background_values = np.asarray(background_days, dtype=np.float64)
    signal_values = signal_values[np.isfinite(signal_values) & (signal_values > 0)]
    background_values = background_values[np.isfinite(background_values) & (background_values > 0)]
    merged = np.concatenate([signal_values, background_values])
    lo = max(1e-6, float(np.min(merged)))
    hi = max(float(np.max(merged)), lo * 1.01)
    edges = np.linspace(np.log10(lo), np.log10(hi) + 1e-9, bins + 1)
    hs, _ = np.histogram(np.log10(signal_values), bins=edges)
    hb, _ = np.histogram(np.log10(background_values), bins=edges)
    ps = (hs + 1.0) / (hs.sum() + len(hs))
    pb = (hb + 1.0) / (hb.sum() + len(hb))
    return {"edges": edges, "log_lr": np.log(ps) - np.log(pb)}


def apply_hist_lr(days: np.ndarray, calibration: dict[str, np.ndarray]) -> np.ndarray:
    values = np.log10(np.maximum(np.asarray(days, dtype=np.float64), 1e-12))
    edges = calibration["edges"]
    indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return calibration["log_lr"][indices].astype(np.float32)


def gaussian_log_bayes_factor(pair_frame: pd.DataFrame, event_frame: pd.DataFrame) -> np.ndarray:
    ii = pair_frame.idx_i.to_numpy(dtype=np.int32)
    jj = pair_frame.idx_j.to_numpy(dtype=np.int32)
    sigma = event_frame.sky_sigma_rad.to_numpy(dtype=np.float64)
    variance = np.maximum(sigma[ii] ** 2 + sigma[jj] ** 2, 1e-24)
    theta = pair_frame.sky_ang_sep_rad.to_numpy(dtype=np.float64)
    return (math.log(2.0) - np.log(variance) - theta * theta / (2.0 * variance)).astype(np.float32)


def retrieval_metrics_for_weights(frame: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    family_rows: list[dict[str, float]] = []
    for family, group in frame.groupby("family", sort=True):
        n = int(group.event_count.iloc[0])
        ii = group.idx_i.to_numpy(dtype=np.int32)
        jj = group.idx_j.to_numpy(dtype=np.int32)
        label = np.zeros((n, n), dtype=bool)
        truth = group.is_true_pair.to_numpy(dtype=bool)
        label[ii, jj] = truth
        label[jj, ii] = truth
        total = np.zeros((n, n), dtype=np.float32)
        for channel in ("waveform", "time", "sky"):
            total += weights[channel] * row_z_neutral(matrix_from_pairs(group, n, f"{channel}_score"))
        np.fill_diagonal(total, -np.inf)
        ranks = []
        for query in np.flatnonzero(label.any(axis=1)):
            partner = int(np.flatnonzero(label[query])[0])
            ranks.append(1 + int(np.sum(total[query] > total[query, partner])))
        ranks_array = np.asarray(ranks)
        family_rows.append(
            {
                "family": str(family),
                "r_at_1": float(np.mean(ranks_array <= 1)),
                "r_at_5": float(np.mean(ranks_array <= 5)),
                "r_at_10": float(np.mean(ranks_array <= 10)),
                "median_rank": float(np.median(ranks_array)),
            }
        )
    result: dict[str, float] = {}
    for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
        result[f"macro_{metric}"] = float(np.mean([row[metric] for row in family_rows]))
    for row in family_rows:
        for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
            result[f"{row['family'].lower()}_{metric}"] = float(row[metric])
    return result


def select_positive_weights(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for waveform, time, sky in itertools.product(WEIGHT_GRID, repeat=3):
        if min(waveform, time, sky) <= 0.0:
            continue
        weights = {"waveform": waveform, "time": time, "sky": sky}
        metrics = retrieval_metrics_for_weights(frame, weights)
        rows.append({**weights, **metrics, "weight_l2": math.sqrt(waveform**2 + time**2 + sky**2)})
    grid = pd.DataFrame(rows).sort_values(
        ["macro_r_at_10", "macro_r_at_5", "macro_r_at_1", "weight_l2"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return {channel: float(best[channel]) for channel in ("waveform", "time", "sky")}, grid


def sky_bf_validation(deployment: str, seed: int) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    root = seed_dir(deployment, seed) / "results"
    val = pd.read_parquet(root / "fusion_validation_pairs.parquet")
    test = pd.read_parquet(root / "fusion_heldout_test_pairs.parquet")
    for family in ("SIS", "PM"):
        val_events = pd.read_parquet(root / f"{family.lower()}_val_synthetic_event_observables.parquet")
        test_events = pd.read_parquet(root / f"{family.lower()}_test_synthetic_event_observables.parquet")
        val_mask = val.family.astype(str).str.upper() == family
        test_mask = test.family.astype(str).str.upper() == family
        val.loc[val_mask, "sky_score"] = gaussian_log_bayes_factor(val.loc[val_mask], val_events)
        test.loc[test_mask, "sky_score"] = gaussian_log_bayes_factor(test.loc[test_mask], test_events)
    weights, _ = select_positive_weights(val)
    return weights, retrieval_metrics_for_weights(val, weights), retrieval_metrics_for_weights(test, weights)


def corrected_rankings(window_audit: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_rows: list[dict[str, Any]] = []
    sensitivity_frames: list[pd.DataFrame] = []
    top5_channel_rows: list[dict[str, Any]] = []
    availability = window_audit.set_index(["deployment", "event_name"])["corrected_embedding_available"].to_dict()
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            current = current_scores(deployment, seed)
            weights_current = selected_weights(deployment, seed)
            weights_bf, val_metrics, test_metrics = sky_bf_validation(deployment, seed)
            validation_rows.append(
                {
                    "deployment": deployment,
                    "seed": seed,
                    **{f"weight_{k}": v for k, v in weights_bf.items()},
                    **{f"validation_{k}": v for k, v in val_metrics.items()},
                    **{f"heldout_test_{k}": v for k, v in test_metrics.items()},
                }
            )
            base = current.copy()
            corrected_available = np.asarray(
                [availability.get((deployment, a), False) and availability.get((deployment, b), False) for a, b in zip(base.event_i, base.event_j)],
                dtype=bool,
            )
            if "pair_has_ood" in base:
                corrected_available &= ~base.pair_has_ood.fillna(False).to_numpy(dtype=bool)
            base["old_waveform_available"] = base.waveform_available
            base["waveform_available"] = corrected_available
            base["full_waveform_scored"] = corrected_available
            base.loc[~corrected_available, "waveform_score"] = np.nan
            base["sky_score_cosine"] = base.sky_score
            nside = 512
            npix = 12 * nside * nside
            base["sky_score_log_bayes_factor"] = np.log(
                np.clip(npix * base.sky_overlap.to_numpy(dtype=np.float64), 1e-300, None)
            )
            prior_path = seed_dir(deployment, seed) / "data" / "real_noise_injections" / "exposure_conditioned_delay_prior_days.npy"
            prior = np.load(prior_path)
            time_cal = fit_hist_lr(prior, base.delta_t_days.to_numpy(dtype=np.float64))
            base["time_score_old"] = base.time_score
            base["time_score_exposure_conditioned_real_background"] = apply_hist_lr(base.delta_t_days, time_cal)

            finite_current = base.copy()
            finite_current["time_score"] = finite_current.time_score_old
            finite_current["sky_score"] = finite_current.sky_score_cosine
            finite_current = add_pair_scores(finite_current, weights_current, "finite_window_current_channels")
            finite_current["ranking_variant"] = "finite_window_current_time_cosine_sky"

            corrected = base.copy()
            corrected["time_score"] = corrected.time_score_exposure_conditioned_real_background
            corrected["sky_score"] = corrected.sky_score_log_bayes_factor
            corrected = add_pair_scores(corrected, weights_bf, "corrected_diagnostic")
            corrected["ranking_variant"] = "finite_window_exposure_time_area_sky_bf"
            corrected["selected_waveform_weight"] = weights_bf["waveform"]
            corrected["selected_time_weight"] = weights_bf["time"]
            corrected["selected_sky_weight"] = weights_bf["sky"]

            map_audit_path = DEPLOYMENTS[deployment].source_run / "features" / "real_sky_map_audit.csv"
            map_audit = pd.read_csv(map_audit_path).set_index("event_name")
            corrected_by_pair = corrected.set_index(["event_i", "event_j"])
            for old_pair in current.head(5).itertuples(index=False):
                new_pair = corrected_by_pair.loc[(old_pair.event_i, old_pair.event_j)]
                top5_channel_rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "current_rank": int(old_pair.rank),
                        "corrected_diagnostic_rank": int(new_pair["rank"]),
                        "event_i": old_pair.event_i,
                        "event_j": old_pair.event_j,
                        "delta_t_days": float(old_pair.delta_t_days),
                        "time_score_old": float(old_pair.time_score),
                        "time_score_exposure_conditioned_real_background": float(
                            new_pair.time_score_exposure_conditioned_real_background
                        ),
                        "map_area90_i_deg2": float(map_audit.loc[old_pair.event_i, "map_area90_deg2"]),
                        "map_area90_j_deg2": float(map_audit.loc[old_pair.event_j, "map_area90_deg2"]),
                        "map_angular_separation_deg": float(old_pair.angular_sep_map_deg),
                        "sky_cosine_overlap": float(old_pair.sky_cosine_overlap),
                        "sky_log_cosine_overlap": float(old_pair.sky_score),
                        "sky_log_bayes_factor": float(new_pair.sky_score_log_bayes_factor),
                        "waveform_available_old": bool(old_pair.waveform_available),
                        "waveform_available_corrected": bool(new_pair.waveform_available),
                    }
                )

            destination = out / "corrected_diagnostic" / deployment / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            finite_current.to_parquet(destination / "pair_scores_finite_window_current_channels.parquet", index=False)
            corrected.to_parquet(destination / "pair_scores_exposure_time_area_sky_bf.parquet", index=False)
            finite_current.head(100).to_csv(destination / "shortlist_finite_window_current_channels.csv", index=False)
            corrected.head(100).to_csv(destination / "shortlist_exposure_time_area_sky_bf.csv", index=False)

            old_rank = current.set_index(["event_i", "event_j"])["rank"]
            finite_rank = finite_current.set_index(["event_i", "event_j"])["rank"]
            corrected_rank = corrected.set_index(["event_i", "event_j"])["rank"]
            keys = old_rank.index.union(corrected_rank.index)
            comparison = pd.DataFrame(index=keys).reset_index()
            comparison["deployment"] = deployment
            comparison["seed"] = seed
            comparison["current_rank"] = [int(old_rank.loc[key]) for key in keys]
            comparison["finite_window_current_channels_rank"] = [int(finite_rank.loc[key]) for key in keys]
            comparison["corrected_diagnostic_rank"] = [int(corrected_rank.loc[key]) for key in keys]
            comparison["current_top20"] = comparison.current_rank <= 20
            comparison["corrected_top20"] = comparison.corrected_diagnostic_rank <= 20
            sensitivity_frames.append(comparison[comparison.current_top20 | comparison.corrected_top20])
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / "sky_bayes_factor_validation_selected_weights.csv", index=False)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity.to_csv(out / "candidate_rank_sensitivity.csv", index=False)
    pd.DataFrame(top5_channel_rows).to_csv(out / "time_sky_top5_audit.csv", index=False)
    return validation, sensitivity


def posterior_dataset(deploy: Deployment, event_row: pd.Series) -> tuple[Path, str, h5py.Dataset]:
    path = resolve_path(event_row.sky_map_path, [REPO, deploy.source_run])
    if path is None or not path.exists():
        raise FileNotFoundError(str(path))
    group = str(event_row[deploy.group_column])
    handle = h5py.File(path, "r")
    return path, group, handle[f"{group}/posterior_samples"]


def chirp_mass(mass_1: np.ndarray, mass_2: np.ndarray) -> np.ndarray:
    return (mass_1 * mass_2) ** (3.0 / 5.0) / np.maximum(mass_1 + mass_2, 1e-30) ** (1.0 / 5.0)


def posterior_values(dataset: h5py.Dataset, parameter: str, indices: np.ndarray) -> np.ndarray:
    names = set(dataset.dtype.names or ())
    if parameter in names:
        return np.asarray(dataset[parameter][indices], dtype=np.float64)
    if parameter == "chirp_mass" and {"mass_1", "mass_2"}.issubset(names):
        return chirp_mass(np.asarray(dataset["mass_1"][indices]), np.asarray(dataset["mass_2"][indices]))
    if parameter == "chirp_mass_source" and {"mass_1_source", "mass_2_source"}.issubset(names):
        return chirp_mass(np.asarray(dataset["mass_1_source"][indices]), np.asarray(dataset["mass_2_source"][indices]))
    if parameter == "mass_ratio" and {"mass_1", "mass_2"}.issubset(names):
        m1 = np.asarray(dataset["mass_1"][indices], dtype=np.float64)
        m2 = np.asarray(dataset["mass_2"][indices], dtype=np.float64)
        return np.minimum(m1, m2) / np.maximum(m1, m2)
    return np.full(len(indices), np.nan, dtype=np.float64)


def overlap_histogram(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    lo = min(float(np.quantile(x, 0.001)), float(np.quantile(y, 0.001)))
    hi = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
    if not np.isfinite(lo + hi) or hi <= lo:
        return float(np.isclose(np.median(x), np.median(y)))
    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    return float(np.sum(np.minimum(hx, hy) * np.diff(edges)))


def overlap_kde(x: np.ndarray, y: np.ndarray, max_samples: int = 5000) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) > max_samples:
        x = x[np.linspace(0, len(x) - 1, max_samples, dtype=int)]
    if len(y) > max_samples:
        y = y[np.linspace(0, len(y) - 1, max_samples, dtype=int)]
    lo = min(float(np.quantile(x, 0.0005)), float(np.quantile(y, 0.0005)))
    hi = max(float(np.quantile(x, 0.9995)), float(np.quantile(y, 0.9995)))
    if hi <= lo:
        return float(np.isclose(np.median(x), np.median(y)))
    try:
        kx = stats.gaussian_kde(x)
        ky = stats.gaussian_kde(y)
        grid = np.linspace(lo, hi, 4096)
        return float(np.trapezoid(np.minimum(kx(grid), ky(grid)), grid))
    except np.linalg.LinAlgError:
        return overlap_histogram(x, y, 300)


def load_candidate_posteriors(
    variants: dict[tuple[str, str], pd.DataFrame], out: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    needed: dict[str, set[str]] = {key: set() for key in DEPLOYMENTS}
    for (deployment, _), pairs in variants.items():
        needed[deployment].update(pd.unique(pairs[["event_i", "event_j"]].to_numpy().ravel()).tolist())
    for deployment, events in needed.items():
        deploy = DEPLOYMENTS[deployment]
        manifest = primary_manifest(deploy).set_index("event_name")
        for event in sorted(events):
            row = manifest.loc[event]
            path, group, dataset = posterior_dataset(deploy, row)
            handle = dataset.file
            samples_total = int(len(dataset))
            try:
                indices = np.linspace(0, samples_total - 1, min(samples_total, POSTERIOR_LIMIT), dtype=np.int64)
                frame = pd.DataFrame({parameter: posterior_values(dataset, parameter, indices) for parameter in PARAMETERS})
            finally:
                handle.close()
            frame.insert(0, "event", event)
            frame.insert(0, "deployment", deployment)
            long_frames.append(frame)
            metadata.append(
                {
                    "deployment": deployment,
                    "event": event,
                    "pe_path": str(path),
                    "pe_group": group,
                    "samples_total": samples_total,
                    "samples_exported": int(len(frame)),
                }
            )
    posterior = pd.concat(long_frames, ignore_index=True)
    posterior.to_parquet(out / "top_candidate_pe_posterior_samples.parquet", index=False)
    pd.DataFrame(metadata).to_csv(out / "top_candidate_pe_files.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for (deployment, variant), pairs in variants.items():
        for pair in pairs.itertuples(index=False):
            for parameter in PARAMETERS:
                x = posterior.loc[(posterior.deployment == deployment) & (posterior.event == pair.event_i), parameter].dropna().to_numpy()
                y = posterior.loc[(posterior.deployment == deployment) & (posterior.event == pair.event_j), parameter].dropna().to_numpy()
                if not len(x) or not len(y):
                    continue
                histogram_values = {bins: overlap_histogram(x, y, bins) for bins in (64, 128, 300, 600, 1200)}
                row = {
                    "deployment": deployment,
                    "ranking_variant": variant,
                    "rank": int(pair.rank),
                    "event_i": pair.event_i,
                    "event_j": pair.event_j,
                    "parameter": parameter,
                    "event_i_q05": float(np.quantile(x, 0.05)),
                    "event_i_median": float(np.median(x)),
                    "event_i_q95": float(np.quantile(x, 0.95)),
                    "event_j_q05": float(np.quantile(y, 0.05)),
                    "event_j_median": float(np.median(y)),
                    "event_j_q95": float(np.quantile(y, 0.95)),
                    "overlap_kde": overlap_kde(x, y),
                    "wasserstein_distance": float(stats.wasserstein_distance(x, y)),
                }
                row.update({f"overlap_hist_{bins}": value for bins, value in histogram_values.items()})
                row["overlap_hist_median_across_bins"] = float(np.median(list(histogram_values.values())))
                row["overlap_hist_min_across_bins"] = float(np.min(list(histogram_values.values())))
                row["overlap_hist_max_across_bins"] = float(np.max(list(histogram_values.values())))
                summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "pe_overlap_robustness_top5.csv", index=False)
    return posterior, summary


def checkpoint_config(deployment: str) -> dict[str, Any]:
    checkpoint = seed_dir(deployment, REFERENCE_SEED) / "waveform_gate" / "sis_real_noise_inceptiontime_ep20_clean" / "model.pt"
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return dict(raw["config"])


def legacy_model_input(raw: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    target_len = int(config.get("target_len", 8192))
    stride = int(config.get("stride", 2))
    values = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[-1] >= target_len:
        values = values[..., -target_len:]
    else:
        padded = np.zeros((values.shape[0], target_len), dtype=np.float32)
        padded[..., -values.shape[-1] :] = values
        values = padded
    values = values[..., ::stride]
    mode = str(config.get("preprocess", "none")).lower()
    lo = int(config.get("bandpass_low", 0))
    hi = int(config.get("bandpass_high", values.shape[-1] // 2))
    if "bandpass" in mode:
        spectrum = np.fft.rfft(values, axis=-1)
        mask = np.zeros(spectrum.shape[-1], dtype=bool)
        lo_index = max(0, lo)
        hi_index = min(spectrum.shape[-1] - 1, hi)
        mask[lo_index : hi_index + 1] = True
        spectrum = np.where(mask[None, :], spectrum, 0.0)
        values = np.fft.irfft(spectrum, n=values.shape[-1], axis=-1).astype(np.float32)
    if bool(config.get("aug_flip", False)):
        reference = values.sum(axis=0)
        if reference[np.argmax(np.abs(reference))] < 0:
            values = -values
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mean) * 1e-6, 1e-30)
    values = ((values - mean) / np.maximum(std, floor)).astype(np.float32)
    effective_fs = FS / stride
    bin_width = effective_fs / values.shape[-1]
    return values, {
        "target_len_before_stride": target_len,
        "stride": stride,
        "model_input_samples": int(values.shape[-1]),
        "effective_sample_rate_hz": effective_fs,
        "configured_bandpass_low": lo,
        "configured_bandpass_high": hi,
        "actual_fft_band_low_hz": lo * bin_width,
        "actual_fft_band_high_hz": hi * bin_width,
    }


def whiten_display_window(path: Path, gps: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
    segment, fs, _ = detector_window(path, gps, DISPLAY_WINDOW_START, DISPLAY_WINDOW_END)
    finite_fraction = float(np.mean(np.isfinite(segment)))
    time = np.arange(len(segment), dtype=np.float64) / fs + DISPLAY_WINDOW_START
    if finite_fraction < 0.95:
        return time.astype(np.float32), np.zeros(len(segment), dtype=np.float32), finite_fraction, False
    with h5py.File(path, "r") as h5:
        dataset = h5["strain/Strain"]
        gps_start = scalar(h5, "meta/GPSstart")
        duration = scalar(h5, "meta/Duration")
        ref_fs = len(dataset) / duration
        ref0 = max(0, int(round((gps - 128.0 - gps_start) * ref_fs)))
        ref1 = min(len(dataset), int(round((gps + 128.0 - gps_start) * ref_fs)))
        reference = np.asarray(dataset[ref0:ref1], dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    if len(reference) < int(32 * fs):
        return time.astype(np.float32), np.zeros(len(segment), dtype=np.float32), finite_fraction, False
    frequencies, psd = signal.welch(
        reference,
        fs=fs,
        window="hann",
        nperseg=int(8 * fs),
        noverlap=int(4 * fs),
        detrend="constant",
        scaling="density",
    )
    floor = max(float(np.nanmedian(psd)) * 1e-12, 1e-60)
    psd = np.maximum(psd, floor)
    clean = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    tapered = clean * signal.windows.tukey(len(clean), alpha=0.08)
    spectrum = np.fft.rfft(tapered)
    f_segment = np.fft.rfftfreq(len(clean), d=1.0 / fs)
    white = np.fft.irfft(spectrum / np.sqrt(np.interp(f_segment, frequencies, psd)), n=len(clean))
    sos = signal.butter(4, [20.0, 500.0], btype="bandpass", fs=fs, output="sos")
    white = signal.sosfiltfilt(sos, white)
    off = np.r_[white[: int(1.5 * fs)], white[-int(0.5 * fs) :]]
    scale = float(np.median(np.abs(off - np.median(off))) * 1.4826)
    white = (white - np.median(off)) / max(scale, 1e-12)
    return time.astype(np.float32), white.astype(np.float32), finite_fraction, True


def normalized_correlation(x: np.ndarray, y: np.ndarray, max_lag_samples: int) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, 0
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    zero = float(np.mean(x * y))
    corr = signal.correlate(x, y, mode="full", method="fft") / len(x)
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    mask = np.abs(lags) <= max_lag_samples
    index = np.flatnonzero(mask)[np.argmax(np.abs(corr[mask]))]
    return zero, float(corr[index]), int(lags[index])


def waveform_audit(
    variants: dict[tuple[str, str], pd.DataFrame], window_audit: pd.DataFrame, out: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    event_audit = window_audit.set_index(["deployment", "event_name"])
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    needed: dict[str, set[str]] = {key: set() for key in DEPLOYMENTS}
    for (deployment, _), frame in variants.items():
        needed[deployment].update(pd.unique(frame[["event_i", "event_j"]].to_numpy().ravel()).tolist())
    event_inputs: dict[tuple[str, str, str], np.ndarray] = {}
    event_white: dict[tuple[str, str, str], np.ndarray] = {}
    configs: dict[str, dict[str, Any]] = {}
    for deployment, events in needed.items():
        configs[deployment] = checkpoint_config(deployment)
        for event in sorted(events):
            event_row = event_audit.loc[(deployment, event)]
            raw_channels: list[np.ndarray] = []
            event_row_indices: list[int] = []
            for detector in DETECTORS:
                path_value = event_row[f"{detector.lower()}_path"]
                path = Path(str(path_value)) if path_value else None
                if path is None or not path.exists():
                    raw = np.full(int(WINDOW_SECONDS * FS), np.nan, dtype=np.float64)
                else:
                    raw, _, _ = detector_window(path, float(event_row.gps_time), -23.75, 0.25)
                raw_channels.append(raw)
                if path is not None and path.exists():
                    time, white, finite_fraction, usable = whiten_display_window(path, float(event_row.gps_time))
                else:
                    time = np.arange(int((DISPLAY_WINDOW_END - DISPLAY_WINDOW_START) * FS)) / FS + DISPLAY_WINDOW_START
                    white = np.zeros_like(time)
                    finite_fraction = 0.0
                    usable = False
                event_white[(deployment, event, detector)] = white
                arrays[f"{deployment}__{event}__{detector}__white_time"] = np.asarray(time, dtype=np.float32)
                arrays[f"{deployment}__{event}__{detector}__white"] = np.asarray(white, dtype=np.float32)
                event_row_indices.append(len(rows))
                rows.append(
                    {
                        "deployment": deployment,
                        "event": event,
                        "detector": detector,
                        "strain_path": str(path) if path else "",
                        "finite_fraction_24s": float(event_row[f"{detector.lower()}_finite_fraction_24s"]),
                        "finite_fraction_display_5s": finite_fraction,
                        "whitened_quicklook_usable": usable,
                    }
                )
            model_input, config_audit = legacy_model_input(np.stack(raw_channels), configs[deployment])
            time_model = np.arange(model_input.shape[-1]) / config_audit["effective_sample_rate_hz"] + MODEL_WINDOW_START
            for detector_index, detector in enumerate(DETECTORS):
                event_inputs[(deployment, event, detector)] = model_input[detector_index]
                arrays[f"{deployment}__{event}__{detector}__legacy_model_time"] = time_model.astype(np.float32)
                arrays[f"{deployment}__{event}__{detector}__legacy_model_input"] = model_input[detector_index]
                peak_index = int(np.argmax(np.abs(model_input[detector_index])))
                rows[event_row_indices[detector_index]].update(
                    {
                        **config_audit,
                        "legacy_input_peak_index": peak_index,
                        "legacy_input_peak_time_s": float(time_model[peak_index]),
                        "legacy_input_boundary_peak": bool(peak_index <= 8 or peak_index >= model_input.shape[-1] - 9),
                        "legacy_input_std": float(np.std(model_input[detector_index])),
                    }
                )
    event_frame = pd.DataFrame(rows)
    event_frame.to_csv(out / "top_candidate_waveform_input_audit.csv", index=False)
    np.savez_compressed(out / "top_candidate_waveform_arrays.npz", **arrays)

    for (deployment, variant), pairs in variants.items():
        config = configs[deployment]
        effective_fs = FS / int(config.get("stride", 2))
        for pair in pairs.itertuples(index=False):
            record: dict[str, Any] = {
                "deployment": deployment,
                "ranking_variant": variant,
                "rank": int(pair.rank),
                "event_i": pair.event_i,
                "event_j": pair.event_j,
                "waveform_score_from_ranking": float(pair.waveform_score) if np.isfinite(pair.waveform_score) else np.nan,
                "waveform_available_from_ranking": bool(pair.waveform_available),
            }
            for detector in DETECTORS:
                x = event_inputs[(deployment, pair.event_i, detector)]
                y = event_inputs[(deployment, pair.event_j, detector)]
                zero, best, lag = normalized_correlation(x, y, int(round(0.1 * effective_fs)))
                wx = event_white[(deployment, pair.event_i, detector)]
                wy = event_white[(deployment, pair.event_j, detector)]
                wzero, wbest, wlag = normalized_correlation(wx, wy, int(round(0.1 * FS)))
                record[f"{detector.lower()}_legacy_zero_lag_corr"] = zero
                record[f"{detector.lower()}_legacy_best_abs_corr_pm100ms"] = best
                record[f"{detector.lower()}_legacy_best_lag_ms"] = 1000.0 * lag / effective_fs
                record[f"{detector.lower()}_whitened_zero_lag_corr"] = wzero
                record[f"{detector.lower()}_whitened_best_abs_corr_pm100ms"] = wbest
                record[f"{detector.lower()}_whitened_best_lag_ms"] = 1000.0 * wlag / FS
            pair_rows.append(record)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(out / "top5_waveform_pair_diagnostics.csv", index=False)
    return pair_frame, arrays


def density_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label_x: str, label_y: str, parameter: str) -> None:
    values_x = x[np.isfinite(x)]
    values_y = y[np.isfinite(y)]
    lo = min(np.quantile(values_x, 0.001), np.quantile(values_y, 0.001))
    hi = max(np.quantile(values_x, 0.999), np.quantile(values_y, 0.999))
    edges = np.linspace(lo, hi, 80)
    ax.hist(values_x, bins=edges, density=True, histtype="step", linewidth=1.3, color="#277DA1", label=label_x)
    ax.hist(values_y, bins=edges, density=True, histtype="step", linewidth=1.3, color="#F94144", label=label_y)
    ax.set_xlabel(f"{PARAMETERS[parameter][0]} {PARAMETERS[parameter][1]}")
    ax.set_yticks([])
    ax.grid(alpha=0.18)


def plot_pe(
    deployment: str,
    variant: str,
    pairs: pd.DataFrame,
    posterior: pd.DataFrame,
    summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(14.5, 12.0), constrained_layout=True)
    for row_index, pair in enumerate(pairs.itertuples(index=False)):
        for column_index, parameter in enumerate(PARAMETERS):
            ax = axes[row_index, column_index]
            x = posterior.loc[(posterior.deployment == deployment) & (posterior.event == pair.event_i), parameter].to_numpy(dtype=float)
            y = posterior.loc[(posterior.deployment == deployment) & (posterior.event == pair.event_j), parameter].to_numpy(dtype=float)
            density_plot(ax, x, y, pair.event_i, pair.event_j, parameter)
            match = summary[
                (summary.deployment == deployment)
                & (summary.ranking_variant == variant)
                & (summary["rank"] == int(pair.rank))
                & (summary.parameter == parameter)
            ]
            if not match.empty:
                ovl = float(match.iloc[0].overlap_hist_300)
                kde = float(match.iloc[0].overlap_kde)
                ax.text(0.02, 0.95, f"OVL={ovl:.3g}\nKDE={kde:.3g}", transform=ax.transAxes, va="top", fontsize=8)
            if column_index == 0:
                ax.set_ylabel(f"Rank {int(pair.rank)}\nDensity", fontweight="bold")
            if row_index == 0 and column_index == 3:
                ax.legend(loc="upper right", fontsize=7)
    title_variant = "current ranking" if variant == "current" else "corrected diagnostic ranking"
    fig.suptitle(f"{DEPLOYMENTS[deployment].label}: Rank 1-5 PE posterior consistency ({title_variant})", fontsize=14, fontweight="bold")
    stem = f"fig_top5_pe_posteriors_{deployment}_{variant}"
    fig.savefig(figure_dir / f"{stem}.pdf", dpi=300)
    fig.savefig(figure_dir / f"{stem}.png", dpi=220)
    plt.close(fig)


def plot_waveforms(
    deployment: str,
    variant: str,
    pairs: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    figure_dir: Path,
    kind: str,
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(13.5, 11.5), sharex=True, constrained_layout=True)
    for row_index, pair in enumerate(pairs.itertuples(index=False)):
        for detector_index, detector in enumerate(DETECTORS):
            ax = axes[row_index, detector_index]
            if kind == "legacy":
                time = arrays[f"{deployment}__{pair.event_i}__{detector}__legacy_model_time"]
                x = arrays[f"{deployment}__{pair.event_i}__{detector}__legacy_model_input"]
                y = arrays[f"{deployment}__{pair.event_j}__{detector}__legacy_model_input"]
                ylabel = "Legacy model input"
            else:
                time = arrays[f"{deployment}__{pair.event_i}__{detector}__white_time"]
                x = arrays[f"{deployment}__{pair.event_i}__{detector}__white"]
                y = arrays[f"{deployment}__{pair.event_j}__{detector}__white"]
                ylabel = "Whitened strain"
            ax.plot(time, x, color="#277DA1", linewidth=0.65, alpha=0.9, label=pair.event_i)
            ax.plot(time, y, color="#F94144", linewidth=0.65, alpha=0.8, label=pair.event_j)
            ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.grid(alpha=0.18)
            ax.set_title(f"Rank {int(pair.rank)}, {detector}", fontsize=9)
            if detector_index == 0:
                ax.set_ylabel(ylabel)
            if row_index == 4:
                ax.set_xlabel("Time from event GPS (s)")
            if row_index == 0 and detector_index == 1:
                ax.legend(loc="upper right", fontsize=7)
    descriptor = "actual legacy encoder input" if kind == "legacy" else "independent PSD-whitened quicklook"
    fig.suptitle(
        f"{DEPLOYMENTS[deployment].label}: Rank 1-5 waveform audit ({descriptor})",
        fontsize=14,
        fontweight="bold",
    )
    stem = f"fig_top5_waveforms_{deployment}_{variant}_{kind}"
    fig.savefig(figure_dir / f"{stem}.pdf", dpi=300)
    fig.savefig(figure_dir / f"{stem}.png", dpi=220)
    plt.close(fig)


def plot_contributions(contributions: pd.DataFrame, figure_dir: Path) -> None:
    for deployment in DEPLOYMENTS:
        frame = contributions[(contributions.deployment == deployment) & (contributions.seed == REFERENCE_SEED)].sort_values("rank")
        fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
        y = np.arange(len(frame))
        left_pos = np.zeros(len(frame))
        left_neg = np.zeros(len(frame))
        for channel in ("waveform", "time", "sky"):
            values = frame[f"{channel}_weighted_contribution"].to_numpy(dtype=float)
            positive = values >= 0
            ax.barh(y[positive], values[positive], left=left_pos[positive], color=COLORS[channel], label=channel)
            ax.barh(y[~positive], values[~positive], left=left_neg[~positive], color=COLORS[channel])
            left_pos[positive] += values[positive]
            left_neg[~positive] += values[~positive]
        labels = [f"R{rank}: {a}\n{b}" for rank, a, b in zip(frame["rank"], frame.event_i, frame.event_j)]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Weighted contribution in max-scoring direction")
        ax.set_title(f"{DEPLOYMENTS[deployment].label}: current Rank 1-5 score decomposition")
        ax.legend(loc="lower right", ncol=3)
        ax.grid(axis="x", alpha=0.2)
        stem = f"fig_top5_score_contributions_{deployment}"
        fig.savefig(figure_dir / f"{stem}.pdf", dpi=300)
        fig.savefig(figure_dir / f"{stem}.png", dpi=220)
        plt.close(fig)


def plot_rank_sensitivity(sensitivity: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    for ax, deployment in zip(axes, DEPLOYMENTS):
        frame = sensitivity[(sensitivity.deployment == deployment) & (sensitivity.seed == REFERENCE_SEED)]
        ax.scatter(frame.current_rank, frame.corrected_diagnostic_rank, s=25, alpha=0.75, color="#577590")
        limit = max(20, int(max(frame.current_rank.max(), frame.corrected_diagnostic_rank.max())))
        ax.plot([1, limit], [1, limit], linestyle="--", linewidth=0.8, color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Current rank (cosine sky, old availability)")
        ax.set_ylabel("Corrected diagnostic rank")
        ax.set_title(DEPLOYMENTS[deployment].label)
        ax.grid(alpha=0.2, which="both")
    fig.suptitle("Candidate-rank sensitivity to waveform availability, time prior, and area-sensitive sky score", fontweight="bold")
    fig.savefig(figure_dir / "fig_time_sky_rank_sensitivity.pdf", dpi=300)
    fig.savefig(figure_dir / "fig_time_sky_rank_sensitivity.png", dpi=220)
    plt.close(fig)


def build_report(
    out: Path,
    contributions: pd.DataFrame,
    windows: pd.DataFrame,
    geometry: pd.DataFrame,
    pe_summary: pd.DataFrame,
    validation: pd.DataFrame,
    variants: dict[tuple[str, str], pd.DataFrame],
) -> None:
    lines = [
        "# 真实 GWTC 候选排序全流程 forensic audit",
        "",
        "## 结论先行",
        "",
        "1. 一维后验重叠系数 $O_x=\\int\\min[p_i(x),p_j(x)]\\,dx$ 在两个密度都归一化时是合理的 overlap coefficient，等价于 $1-\\mathrm{TV}(p_i,p_j)$。它的范围为 $[0,1]$。当前直方图实现是合理数值近似，但它不是 lensing Bayes factor，也忽略参数间相关性和先验。",
        "2. 当前 Rank 1-4 的 detector-frame chirp-mass overlap 接近或等于 0 不是小数显示或 bin 选择造成的；多 bin 与 KDE 复算均确认这些 PE posterior 基本不相交。因此这些 pair 不能因高 catalog score 被解释为物理上相似的透镜候选。",
        "3. 当前 final score 没有使用 PE 参数。高排名只说明 waveform embedding、time 和 sky 的加权 row-z score 高；它不保证 chirp mass 一致。",
        "4. O4a 部署存在关键可用性 bug：原代码在检查前将 strain NaN 变成 0，使单探测器缺测事件被标成有效 H1/L1 embedding；高排名中多个 pair 实际按共同的零通道模式聚类。",
        "5. 当前 waveform preprocessing 先抽取尾部 2 s、无抗混叠地 stride=2，再把 40/580 当 FFT bin 而不是 Hz。实际带通约为 20-290 Hz，并出现明显窗口边界峰。真实 embedding 还呈现窄锥/低有效秩，现有 deployment audit 未捕获。",
        "6. HEALPix cosine overlap 会把两个很宽的 posterior 判得过于相容。应同时审计面积敏感的 common-source sky Bayes factor $B_\\Omega=N_{\\rm pix}\\sum_k P_{ik}P_{jk}$。候选排名对这一替换高度敏感。",
        "7. 最新 synthetic validation 的 time LR 使用 exposure-conditioned delay prior，但 real catalog 排序复用了旧 observable 表中的未条件化 time score，训练/部署口径不一致。",
        "",
        "因此，现有 Rank 1-5 应视为被审计的旧 shortlist，不能直接作为物理 follow-up 结论。包内 corrected ranking 只是诊断结果；正式结果需要修复双探测器有效性、按 Hz 正确滤波并重训 encoder，再在 validation 上统一选择 time/sky/waveform 权重。",
        "",
        "## PE overlap 的数学含义",
        "",
        "对归一化一维 posterior $p_i(x)$ 与 $p_j(x)$：",
        "",
        "$$O_x=\\int\\min[p_i(x),p_j(x)]\\,dx=1-\\frac12\\int|p_i(x)-p_j(x)|\\,dx.$$",
        "",
        "它适合做易解释的 follow-up diagnostic。detector-frame masses 在透镜变换下应保持一致，所以 detector-frame $\\mathcal{M}$ 的近零重叠是强烈的不一致证据。source-frame masses、distance 和 redshift 会受独立无透镜 PE 中的放大率偏置影响，不应作为同样严格的 veto。正式确认应使用联合 posterior overlap 或 lensing Bayes factor。",
        "",
        "## 当前权重和 Rank 1-5",
        "",
    ]
    for deployment in DEPLOYMENTS:
        frame = contributions[(contributions.deployment == deployment) & (contributions.seed == REFERENCE_SEED)].sort_values("rank")
        weights = {channel: float(frame.iloc[0][f"{channel}_weight"]) for channel in ("waveform", "time", "sky")}
        lines.append(f"### {DEPLOYMENTS[deployment].label}")
        lines.append("")
        lines.append(f"Validation-selected strictly-positive weights: waveform={weights['waveform']}, time={weights['time']}, sky={weights['sky']}.")
        lines.append("")
        lines.append("| Rank | Pair | final | waveform contribution | time contribution | sky contribution | waveform available |")
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        for row in frame.itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.event_i} -- {row.event_j} | {row.final_score:.3f} | "
                f"{row.waveform_weighted_contribution:.3f} | {row.time_weighted_contribution:.3f} | "
                f"{row.sky_weighted_contribution:.3f} | {row.waveform_available} |"
            )
        lines.append("")
    lines.extend(["## Detector-frame chirp-mass overlap", ""])
    mc = pe_summary[(pe_summary.ranking_variant == "current") & (pe_summary.parameter == "chirp_mass")]
    lines.append("| Catalog | Rank | Pair | OVL (300 bins) | KDE OVL | q05-q95 event i | q05-q95 event j |")
    lines.append("|---|---:|---|---:|---:|---|---|")
    for row in mc.sort_values(["deployment", "rank"]).itertuples(index=False):
        lines.append(
            f"| {row.deployment} | {row.rank} | {row.event_i} -- {row.event_j} | {row.overlap_hist_300:.4g} | "
            f"{row.overlap_kde:.4g} | {row.event_i_q05:.3f}-{row.event_i_q95:.3f} | {row.event_j_q05:.3f}-{row.event_j_q95:.3f} |"
        )
    lines.extend(["", "## Strain availability audit", ""])
    for deployment in DEPLOYMENTS:
        frame = windows[windows.deployment == deployment]
        lines.append(
            f"- {DEPLOYMENTS[deployment].label}: {len(frame)} primary events; old embedding_available={int(frame.old_embedding_available.sum())}; "
            f"exact-window valid H1/L1={int(frame.corrected_embedding_available.sum())}; falsely accepted after NaN->0={int(frame.old_false_positive_availability.sum())}."
        )
    input_audit_path = out / "top_candidate_waveform_input_audit.csv"
    if input_audit_path.exists():
        input_audit = pd.read_csv(input_audit_path)
        lines.append("")
        for deployment in DEPLOYMENTS:
            frame = input_audit[input_audit.deployment == deployment]
            lines.append(
                f"- {DEPLOYMENTS[deployment].label}: plotted detector inputs={len(frame)}; "
                f"legacy-input boundary peaks={int(frame.legacy_input_boundary_peak.sum())}/{len(frame)}; "
                f"configured 40-580 is actually {frame.actual_fft_band_low_hz.iloc[0]:.1f}-{frame.actual_fft_band_high_hz.iloc[0]:.1f} Hz."
            )
    lines.extend(["", "## Embedding geometry audit", ""])
    reference_geometry = geometry[geometry.seed == REFERENCE_SEED]
    lines.append("| Catalog | encoder | mean-vector norm | centered effective rank / 128 | pair cosine median | q90 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in reference_geometry.itertuples(index=False):
        lines.append(
            f"| {row.deployment} | {row.family_encoder} | {row.mean_embedding_vector_norm:.3f} | "
            f"{row.centered_effective_rank:.2f} | {row.pair_cosine_q50:.3f} | {row.pair_cosine_q90:.3f} |"
        )
    lines.extend(["", "## Area-sensitive sky diagnostic", ""])
    lines.append("HEALPix maps are probability masses $P_{ik}$ on a common equal-area grid. Under an isotropic sky prior, the common-source Bayes factor is:")
    lines.append("")
    lines.append("$$B_\\Omega=N_{\\rm pix}\\sum_k P_{ik}P_{jk}.$$")
    lines.append("")
    lines.append("Cosine overlap divides by both map L2 norms and therefore removes much of the localization-area information. Very broad maps can have cosine overlap near one. The corrected diagnostic recalibrates the analogous Gaussian sky BF on validation, then applies the HEALPix BF to real pairs.")
    lines.append("")
    top5_channel_path = out / "time_sky_top5_audit.csv"
    if top5_channel_path.exists():
        channel_audit = pd.read_csv(top5_channel_path)
        for deployment in DEPLOYMENTS:
            frame = channel_audit[
                (channel_audit.deployment == deployment)
                & (channel_audit.seed == REFERENCE_SEED)
                & (channel_audit.current_rank == 1)
            ]
            if not frame.empty:
                row = frame.iloc[0]
                lines.append(
                    f"- {DEPLOYMENTS[deployment].label} current Rank 1: A90={row.map_area90_i_deg2:.1f}/{row.map_area90_j_deg2:.1f} deg2, "
                    f"MAP separation={row.map_angular_separation_deg:.1f} deg, cosine={row.sky_cosine_overlap:.3f}, "
                    f"log sky BF={row.sky_log_bayes_factor:.3f}, corrected diagnostic rank={int(row.corrected_diagnostic_rank)}."
                )
        lines.append("")
    lines.append("| Catalog | seed | BF-selected weights (w,t,s) | validation R@1/R@10 | held-out test R@1/R@10 |")
    lines.append("|---|---:|---|---|---|")
    for row in validation.itertuples(index=False):
        lines.append(
            f"| {row.deployment} | {row.seed} | ({row.weight_waveform:g}, {row.weight_time:g}, {row.weight_sky:g}) | "
            f"{row.validation_macro_r_at_1:.3f}/{row.validation_macro_r_at_10:.3f} | "
            f"{row.heldout_test_macro_r_at_1:.3f}/{row.heldout_test_macro_r_at_10:.3f} |"
        )
    lines.extend(["", "## Corrected diagnostic Rank 1-5（不是正式重跑结果）", ""])
    for deployment in DEPLOYMENTS:
        frame = variants[(deployment, "corrected_diagnostic")].sort_values("rank")
        lines.append(f"### {DEPLOYMENTS[deployment].label}")
        lines.append("")
        lines.append("| Rank | Pair | final | waveform | exposure-time | area-sensitive sky BF | waveform valid |")
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        for row in frame.itertuples(index=False):
            waveform_value = f"{row.waveform_score:.4f}" if np.isfinite(row.waveform_score) else "NA"
            lines.append(
                f"| {row.rank} | {row.event_i} -- {row.event_j} | {row.final_score:.3f} | {waveform_value} | "
                f"{row.time_score:.3f} | {row.sky_score:.3f} | {row.waveform_available} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## 必须修复后再形成正式 shortlist",
            "",
            "1. 在 `load_strain()` 中保留 NaN；对 H1、L1 各自检查 exact 24 s window 的 finite fraction，任一探测器低于阈值则 waveform unavailable，不能先清零再判定。",
            "2. 把 bandpass 参数明确改成 Hz，并在降采样前使用抗混叠滤波；训练、validation 和真实部署必须共用同一个 preprocessing artifact/version。",
            "3. 对真实 embedding 增加 mean-vector norm、centered effective rank、pair cosine quantiles、detector-pattern leakage gate，不能只检查 std 是否非零。",
            "4. 天空主分数改为或至少并列报告面积敏感 posterior-overlap BF；在 held-out validation 上重新选权重。",
            "5. time score 的 signal prior、background construction 和 bin calibration 在 validation/test/real deployment 中必须有明确且一致的版本。",
            "6. PE detector-frame mass overlap作为 shortlist 后的物理 confirmation gate；不要用真实 top pairs 调融合权重。",
            "",
            "## 文件说明",
            "",
            "- `pe_overlap_robustness_top5.csv`: 多 bin 与 KDE 的 PE overlap 复算。",
            "- `top5_score_contributions_all_seeds.csv`: 三通道的 raw score、row-z、权重和最终贡献。",
            "- `real_event_exact_window_finite_audit.csv`: 所有 primary events 的 H1/L1 exact-window 有效性。",
            "- `embedding_geometry_audit.csv`: embedding 窄锥与有效秩检查。",
            "- `corrected_diagnostic/`: 不覆盖原结果的敏感性排序，只能作为诊断。",
            "- `figures/`: 当前和 corrected Rank 1-5 的 PE、legacy model input、独立 whitened quicklook、权重贡献和排名敏感性图。",
        ]
    )
    (out / "real_candidate_forensic_audit_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_results(out: Path, package: Path, script_path: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        archive.add(out, arcname=out.name)
        archive.add(script_path, arcname=f"{out.name}/scripts/{script_path.name}")
        deployment_script = REPO / "scripts" / "real_search" / "06_embed_real_gwtc_events.py"
        archive.add(deployment_script, arcname=f"{out.name}/scripts/{deployment_script.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic audit of real GWTC top-ranked pairs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    out = args.output
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    contributions = score_contributions()
    contributions.to_csv(out / "top5_score_contributions_all_seeds.csv", index=False)
    print("score contributions complete", flush=True)

    windows = audit_exact_windows()
    windows.to_csv(out / "real_event_exact_window_finite_audit.csv", index=False)
    geometry, leakage = embedding_geometry(windows)
    geometry.to_csv(out / "embedding_geometry_audit.csv", index=False)
    leakage.to_csv(out / "detector_pattern_leakage.csv", index=False)
    print("strain availability and embedding geometry complete", flush=True)

    validation, sensitivity = corrected_rankings(windows, out)
    print("time/sky score sensitivity complete", flush=True)

    variants: dict[tuple[str, str], pd.DataFrame] = {}
    for deployment in DEPLOYMENTS:
        variants[(deployment, "current")] = current_scores(deployment, REFERENCE_SEED).head(5).copy()
        corrected_path = out / "corrected_diagnostic" / deployment / f"seed_{REFERENCE_SEED}" / "pair_scores_exposure_time_area_sky_bf.parquet"
        variants[(deployment, "corrected_diagnostic")] = pd.read_parquet(corrected_path).head(5).copy()
    posterior, pe_summary = load_candidate_posteriors(variants, out)
    print("PE posterior extraction and overlap robustness complete", flush=True)

    waveform_pairs, waveform_arrays = waveform_audit(variants, windows, out)
    print("waveform quicklook and input audit complete", flush=True)

    plot_contributions(contributions, figure_dir)
    plot_rank_sensitivity(sensitivity, figure_dir)
    for (deployment, variant), pairs in variants.items():
        plot_pe(deployment, variant, pairs, posterior, pe_summary, figure_dir)
        plot_waveforms(deployment, variant, pairs, waveform_arrays, figure_dir, "legacy")
        plot_waveforms(deployment, variant, pairs, waveform_arrays, figure_dir, "whitened")

    build_report(out, contributions, windows, geometry, pe_summary, validation, variants)

    summary = {
        "status": "complete",
        "reference_seed": REFERENCE_SEED,
        "existing_results_modified": False,
        "pe_overlap_formula": "integral min(p_i, p_j) dx = 1 - total variation distance for normalized 1D densities",
        "critical_findings": {
            "pe_zero_overlap_is_numerically_robust": True,
            "pe_used_in_current_final_score": False,
            "nan_zero_fill_before_availability_check": True,
            "legacy_bandpass_values_are_fft_bins_not_hz": True,
            "sky_cosine_is_area_insensitive": True,
            "real_time_score_reuses_pre_v2_observable_table": True,
        },
        "exact_window_counts": windows.groupby("deployment").agg(
            n_events=("event_name", "size"),
            old_available=("old_embedding_available", "sum"),
            corrected_h1l1_available=("corrected_embedding_available", "sum"),
            false_old_availability=("old_false_positive_availability", "sum"),
        ).reset_index().to_dict(orient="records"),
        "outputs": {
            "report": str(out / "real_candidate_forensic_audit_report_cn.md"),
            "figures": str(figure_dir),
            "package": str(args.package),
        },
    }
    write_json(out / "forensic_audit_summary.json", summary)
    package_results(out, args.package, Path(__file__).resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"package={args.package} bytes={args.package.stat().st_size:,}", flush=True)


if __name__ == "__main__":
    main()
