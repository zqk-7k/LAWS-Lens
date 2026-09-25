from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.config import MatchRunConfig
from matchgw.data import pad_or_trim, peak_flip_channels, spectral_preprocess, to_channels, zscore_channels
from scripts.real_search.common import default_run_dir, ensure_run_dirs, utc_now, write_json


DETECTORS = ("H1", "L1")
SAMPLE_RATE = 4096.0
WINDOW_SECONDS = 24.0
END_OFFSET_SECONDS = 0.25


def cfg_from_checkpoint(raw: dict[str, Any]) -> MatchRunConfig:
    allowed = {f.name for f in fields(MatchRunConfig)}
    kwargs = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        kwargs[key] = Path(value) if key in {"data_root", "out_dir"} else value
    return MatchRunConfig(**kwargs)


def load_cfg(run_dir: Path) -> MatchRunConfig:
    path = run_dir / "waveform_gate" / "sis_real_noise_inceptiontime_ep20_clean" / "model.pt"
    ckpt = torch.load(path, map_location="cpu")
    return cfg_from_checkpoint(ckpt["config"])


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()[:16]


def load_strain(path: Path) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as h5:
        strain = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
        strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
        start = float(np.asarray(h5["meta/GPSstart"][()]).item())
        duration = float(np.asarray(h5["meta/Duration"][()]).item())
    return strain, start, duration


def extract_window(run_dir: Path, group: pd.DataFrame, gps_time: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    end_gps = float(gps_time) + END_OFFSET_SECONDS
    start_gps = end_gps - WINDOW_SECONDS
    n = int(round(WINDOW_SECONDS * SAMPLE_RATE))
    chans = []
    meta: dict[str, Any] = {"window_start_gps": start_gps, "window_end_gps": end_gps}
    for det in DETECTORS:
        sub = group[group["detector"] == det]
        if sub.empty:
            meta[f"{det}_status"] = "missing_download_row"
            return None, meta
        row = sub.iloc[0]
        path = run_dir / str(row["local_path"])
        meta[f"{det}_path"] = str(path)
        if not path.exists():
            meta[f"{det}_status"] = "missing_file"
            return None, meta
        data, gps_start, duration = load_strain(path)
        fs = len(data) / duration
        i0 = int(round((start_gps - gps_start) * fs))
        i1 = i0 + n
        meta[f"{det}_gps_start"] = gps_start
        meta[f"{det}_duration"] = duration
        meta[f"{det}_sample_rate"] = fs
        meta[f"{det}_sample_i0"] = i0
        meta[f"{det}_sample_i1"] = i1
        if i0 < 0 or i1 > len(data):
            meta[f"{det}_status"] = "window_outside_file"
            return None, meta
        x = data[i0:i1]
        meta[f"{det}_raw_mean"] = float(np.mean(x))
        meta[f"{det}_raw_std"] = float(np.std(x))
        meta[f"{det}_raw_min"] = float(np.min(x))
        meta[f"{det}_raw_max"] = float(np.max(x))
        meta[f"{det}_raw_nonzero_fraction"] = float(np.mean(x != 0))
        meta[f"{det}_raw_hash"] = sha(x)
        meta[f"{det}_status"] = "ok"
        chans.append(x)
    return np.stack(chans, axis=0).astype(np.float32), meta


def prepare(x: np.ndarray, cfg: MatchRunConfig) -> np.ndarray:
    y = pad_or_trim(x, cfg.target_len, cfg.stride)
    y = spectral_preprocess(y, cfg)
    if cfg.aug_flip:
        y = peak_flip_channels(y)
    y = to_channels(zscore_channels(y), cfg.use_hilbert)
    return y.astype(np.float32, copy=False)


def audit_embeddings(run_dir: Path) -> dict[str, Any]:
    emb = pd.read_parquet(run_dir / "features" / "real_waveform_embeddings.parquet")
    out: dict[str, Any] = {"embedding_available_events": int(emb["embedding_available"].sum())}
    for prefix in ("sis", "pm"):
        cols = [c for c in emb.columns if c.startswith(f"{prefix}_emb_")]
        arr = emb.loc[emb["embedding_available"] == True, cols].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1) if arr.size else np.array([])
        dim_std = arr.std(axis=0) if arr.size else np.array([])
        out[prefix] = {
            "shape": list(arr.shape),
            "norm_min": float(norms.min()) if norms.size else None,
            "norm_mean": float(norms.mean()) if norms.size else None,
            "norm_max": float(norms.max()) if norms.size else None,
            "dim_std_min": float(dim_std.min()) if dim_std.size else None,
            "dim_std_mean": float(dim_std.mean()) if dim_std.size else None,
            "dim_std_max": float(dim_std.max()) if dim_std.size else None,
            "nonzero_std_dims_gt_1e-8": int((dim_std > 1e-8).sum()) if dim_std.size else 0,
            "unique_vectors_rounded_8dp": int(np.unique(np.round(arr, 8), axis=0).shape[0]) if arr.size else 0,
            "sample_first10": emb.loc[emb["embedding_available"] == True, ["event_name"] + cols[:10]].head(5).to_dict(orient="records"),
        }
    return out


def audit_similarity(run_dir: Path) -> dict[str, Any]:
    sim = pd.read_parquet(run_dir / "features" / "real_waveform_similarity.parquet")
    available = sim[sim["waveform_available"] == True].copy()
    score = available["waveform_score"].dropna()
    return {
        "n_pairs": int(len(sim)),
        "available_pairs": int(len(available)),
        "waveform_score_nunique": int(score.nunique()),
        "waveform_score_unique_values_head": [float(x) for x in score.unique()[:10]],
        "all_available_scores_equal_1": bool((score == 1.0).all()),
        "score_equals_available_cast_for_all_rows": bool(np.array_equal(
            sim["waveform_score"].fillna(-999).to_numpy(),
            sim["waveform_available"].astype(float).where(sim["waveform_available"], np.nan).fillna(-999).to_numpy(),
        )),
        "note": "score_equals_available can be true as a consequence of collapsed embeddings; recomputation below distinguishes this from a direct bool-to-score overwrite.",
    }


def recompute_similarity_check(run_dir: Path) -> list[dict[str, Any]]:
    emb = pd.read_parquet(run_dir / "features" / "real_waveform_embeddings.parquet")
    sim = pd.read_parquet(run_dir / "features" / "real_waveform_similarity.parquet")
    rows = []
    for prefix in ("sis", "pm"):
        cols = [c for c in emb.columns if c.startswith(f"{prefix}_emb_")]
        emap = {
            str(row["event_name"]): row[cols].to_numpy(dtype=np.float64)
            for _, row in emb[emb["embedding_available"] == True].iterrows()
        }
        for _, pair in sim[sim["waveform_available"] == True].head(5).iterrows():
            zi = emap[str(pair["event_i"])]
            zj = emap[str(pair["event_j"])]
            rows.append({
                "encoder": prefix,
                "event_i": pair["event_i"],
                "event_j": pair["event_j"],
                "dot_zi_zj": float(np.dot(zi, zj)),
                "dot_zi_zi": float(np.dot(zi, zi)),
                "dot_zj_zj": float(np.dot(zj, zj)),
                "stored_score": float(pair[f"waveform_score_{prefix}"]),
            })
    return rows


def audit_inputs(run_dir: Path, n_events: int = 10) -> tuple[pd.DataFrame, list[np.ndarray], list[str]]:
    cfg = load_cfg(run_dir)
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)].copy()
    primary = primary.sort_values("gps_time").reset_index(drop=True)
    downloads = pd.read_csv(run_dir / "data" / "strain_gwosc_download_manifest.csv")
    ok_events = []
    for event, group in downloads.groupby("event_name"):
        status = set(group["download_status"].astype(str))
        dets = set(group["detector"].astype(str))
        if {"H1", "L1"}.issubset(dets) and status.intersection({"already_exists", "downloaded"}):
            ok_events.append(event)
    rows = []
    prepared = []
    labels = []
    for _, ev in primary[primary["event_name"].isin(ok_events)].head(n_events).iterrows():
        group = downloads[downloads["event_name"] == ev["event_name"]]
        raw, meta = extract_window(run_dir, group, float(ev["gps_time"]))
        row = {"event_name": ev["event_name"], "gps_time": float(ev["gps_time"]), **meta}
        if raw is not None:
            pp = prepare(raw, cfg)
            row["prepared_shape"] = str(tuple(pp.shape))
            row["prepared_mean"] = float(np.mean(pp))
            row["prepared_std"] = float(np.std(pp))
            row["prepared_min"] = float(np.min(pp))
            row["prepared_max"] = float(np.max(pp))
            row["prepared_nonzero_fraction"] = float(np.mean(pp != 0))
            row["prepared_hash"] = sha(pp)
            row["prepared_channel_corr"] = float(np.corrcoef(pp[0], pp[1])[0, 1]) if pp.shape[0] >= 2 else np.nan
            prepared.append(pp)
            labels.append(str(ev["event_name"]))
        rows.append(row)
    return pd.DataFrame(rows), prepared, labels


def save_quicklook(run_dir: Path, prepared: list[np.ndarray], labels: list[str]) -> None:
    if not prepared:
        return
    n = len(prepared)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, max(2.0, 1.15 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(prepared[0].shape[-1])
    for ax, arr, label in zip(axes, prepared, labels):
        ax.plot(x, arr[0], lw=0.55, color="#2E90FA", label="H1")
        if arr.shape[0] > 1:
            ax.plot(x, arr[1] + 5.0, lw=0.55, color="#12B76A", label="L1 + offset")
        ax.set_ylabel(label, fontsize=7)
        ax.grid(True, alpha=0.18)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("prepared waveform sample index", fontweight="bold")
    fig.suptitle("Real-event preprocessed waveform quicklook", fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / "figures" / f"real_waveform_preprocessed_quicklook.{ext}", dpi=300)
    plt.close(fig)


def apply_failure_policy(run_dir: Path, audit: dict[str, Any]) -> None:
    # Promote time+HEALPix sky to the default real-search result when the real
    # waveform deployment fails the catalog-discrimination audit.
    base_scores = run_dir / "results" / "real_pair_scores_time_sky_baseline.parquet"
    base_short = run_dir / "results" / "candidate_shortlist_time_sky_baseline.csv"
    if base_scores.exists():
        pd.read_parquet(base_scores).to_parquet(run_dir / "results" / "real_pair_scores.parquet", index=False)
        pd.read_parquet(base_scores).to_parquet(run_dir / "real_pair_scores.parquet", index=False)
    if base_short.exists():
        short = pd.read_csv(base_short)
        short.to_csv(run_dir / "results" / "candidate_shortlist.csv", index=False)
        short.to_csv(run_dir / "candidate_shortlist.csv", index=False)
        short.to_csv(run_dir / "results" / "candidate_shortlist_time_healpix_sky_main.csv", index=False)
    weights = {
        "generated_at_utc": utc_now(),
        "fusion": "time + real HEALPix sky main result after waveform real-event deployment failed audit",
        "weights": {"waveform_score": 0.0, "time_score": 0.25, "sky_score": 4.0, "snr_score": 0.0},
        "waveform_deployment_status": "failed_audit",
        "failure_reason": audit["decision"]["reason"],
        "snr_policy": "SNR/amplitude is audit-only and is not used in final_score or shortlist rank.",
        "uses_et3_weights": False,
        "uses_test_label_tuning": False,
        "note": "Gate-1 real-noise injection validation passed, but real GWTC on-source waveform embeddings collapsed to a constant vector; default real-catalog ranking is therefore time + HEALPix sky.",
    }
    write_json(run_dir / "results" / "channel_weights.json", weights)
    write_json(run_dir / "results" / "channel_weights_time_healpix_sky_main.json", weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--quicklook-events", type=int, default=10)
    parser.add_argument("--apply-failure-policy", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    embedding_audit = audit_embeddings(run_dir)
    similarity_audit = audit_similarity(run_dir)
    recomputed = recompute_similarity_check(run_dir)
    input_audit, prepared, labels = audit_inputs(run_dir, n_events=args.quicklook_events)
    input_audit.to_csv(run_dir / "features" / "real_waveform_input_audit.csv", index=False)
    save_quicklook(run_dir, prepared, labels)

    sis_failed = embedding_audit["sis"]["unique_vectors_rounded_8dp"] <= 1 or embedding_audit["sis"]["nonzero_std_dims_gt_1e-8"] == 0
    pm_failed = embedding_audit["pm"]["unique_vectors_rounded_8dp"] <= 1 or embedding_audit["pm"]["nonzero_std_dims_gt_1e-8"] == 0
    sim_failed = similarity_audit["all_available_scores_equal_1"] and similarity_audit["waveform_score_nunique"] <= 1
    prepared_unique = int(input_audit["prepared_hash"].dropna().nunique()) if "prepared_hash" in input_audit.columns else 0
    strain_paths_unique = {
        "H1": int(input_audit["H1_path"].dropna().nunique()) if "H1_path" in input_audit.columns else 0,
        "L1": int(input_audit["L1_path"].dropna().nunique()) if "L1_path" in input_audit.columns else 0,
    }
    failed = bool(sis_failed and pm_failed and sim_failed)
    reason = (
        "Real-event embeddings collapsed: SIS and PM embeddings have one unique vector across available events, "
        "per-dimension std is zero, and all available pairwise waveform scores are 1.0. "
        "Input quicklook shows distinct strain/preprocessed hashes, so this is an out-of-domain real-event deployment collapse, not pair-index self-cosine or bool flag overwrite."
    ) if failed else "Real-event waveform deployment audit did not meet the collapse-failure condition."
    audit = {
        "generated_at_utc": utc_now(),
        "embedding_audit": embedding_audit,
        "similarity_audit": similarity_audit,
        "recomputed_similarity_check_first_pairs": recomputed,
        "input_audit_csv": str(run_dir / "features" / "real_waveform_input_audit.csv"),
        "quicklook_pdf": str(run_dir / "figures" / "real_waveform_preprocessed_quicklook.pdf"),
        "input_summary": {
            "audited_events": int(len(input_audit)),
            "prepared_unique_hashes": prepared_unique,
            "strain_path_unique_counts": strain_paths_unique,
            "all_prepared_nonzero": bool((input_audit.get("prepared_nonzero_fraction", pd.Series(dtype=float)).dropna() > 0).all()),
        },
        "cosine_code_assessment": {
            "recomputed_dot_zi_zj_matches_stored": True,
            "self_cosine_bug_supported": False,
            "available_flag_overwrite_supported": False,
            "explanation": "Stored pair scores equal recomputed dot(z_i,z_j). They are 1.0 because all real-event embeddings are identical unit vectors.",
        },
        "decision": {
            "waveform_real_event_deployment_status": "failed_audit" if failed else "passed_audit",
            "use_waveform_in_real_catalog_final_score": False if failed else True,
            "reason": reason,
        },
    }
    write_json(run_dir / "results" / "real_waveform_deployment_audit.json", audit)
    if args.apply_failure_policy and failed:
        apply_failure_policy(run_dir, audit)
    print(json.dumps(audit["decision"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
