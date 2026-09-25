from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
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
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import retrieval_metrics, similarity_matrix
from matchgw.pipeline import build_model, embed_eval
from scripts.real_search.common import (
    SECONDS_PER_DAY,
    angular_sep_rad,
    choose_h5_skymap_group,
    empirical_hist_lr,
    ensure_run_dirs,
    infer_run_from_gps,
    pair_indices,
    read_probability_map,
    utc_now,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "real_gwtc34_lensing_search_20260629"
GWTC3_RUN = REPO_ROOT / "runs" / "real_gwtc_lensing_search_20260625"
GWTC4_DOWNLOAD_RUN = REPO_ROOT / "runs" / "gwtc4p1_data_completion_20260628"
GWTC4_DATA_ROOT = REPO_ROOT / "data" / "gwtc4p1"
SYNTHETIC_CLEAN_ROOT = GWTC3_RUN / "data" / "real_noise_injections" / "matchroots" / "LIGO"
FAMILIES = ("SIS", "PM")
DETECTORS = ("H1", "L1")
SAMPLE_RATE = 4096.0
TARGET_LEN = 98304
WINDOW_SECONDS = 24.0
END_OFFSET_SECONDS = 0.25
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def rel_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.absolute().relative_to(run_dir.absolute()))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def safe_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    os.symlink(target.resolve(), link)


def load_strain_file(path: Path) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as h5:
        strain = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
        gps_start = float(np.asarray(h5["meta/GPSstart"][()]).item())
        duration = float(np.asarray(h5["meta/Duration"][()]).item())
    strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
    return strain, gps_start, duration


def hdf5_ok(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            return "strain/Strain" in h5 and "meta/GPSstart" in h5 and "meta/Duration" in h5
    except Exception:
        return False


def zscore_channelwise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mu) * 1e-6, 1e-30)
    return ((x - mu) / np.maximum(sd, floor)).astype(np.float32, copy=False)


def class_for_event(row: pd.Series) -> tuple[str, bool]:
    family = str(row.get("preferred_pe_waveform_family", "")).lower()
    m2 = row.get("mass_2_source")
    try:
        m2f = float(m2)
    except Exception:
        m2f = float("nan")
    if "nsbh" in family or "nrtidal" in family or (np.isfinite(m2f) and m2f <= 3.0):
        return "NSBH_or_low_mass_OOD", True
    if not np.isfinite(m2f):
        return "unknown_OOD", True
    return "BBH", False


def build_gwtc4_manifest(run_dir: Path) -> dict[str, Any]:
    events = pd.read_csv(GWTC4_DOWNLOAD_RUN / "data" / "event_manifest_gwtc4p1.csv")
    strains = pd.read_csv(GWTC4_DOWNLOAD_RUN / "data" / "strain_manifest_gwtc4p1.csv")
    pe = pd.read_csv(GWTC4_DOWNLOAD_RUN / "data" / "pe_manifest_gwtc4p1.csv")
    pe_by_event = {str(r.event_name): r for r in pe.itertuples(index=False)}
    strain_groups = {event: {row.detector: row for row in group.itertuples(index=False)} for event, group in strains.groupby("event_name")}

    rows = []
    strain_manifest_rows = []
    offsource_rows = []
    for src_idx, row in events.reset_index(drop=True).iterrows():
        event_name = str(row["event_name"])
        pe_row = pe_by_event.get(event_name)
        pe_path = REPO_ROOT / str(pe_row.local_path) if pe_row is not None else None
        sky_group = None
        sky_ok = False
        if pe_path is not None and pe_path.exists():
            try:
                sky_group = choose_h5_skymap_group(pe_path)
                sky_ok = sky_group is not None
            except Exception:
                sky_group = None
                sky_ok = False
        det_rows = strain_groups.get(event_name, {})
        valid_detectors = []
        for det, det_row in det_rows.items():
            if int(getattr(det_row, "sampling_rate", 0)) != 4096:
                continue
            p = REPO_ROOT / str(det_row.local_path)
            if p.exists() and int(getattr(det_row, "local_bytes", 0) or 0) > 0 and hdf5_ok(p):
                valid_detectors.append(str(det))
                link = run_dir / "data" / "real_strain" / event_name / p.name
                safe_symlink(p, link)
                strain_manifest_rows.append({
                    "event_name": event_name,
                    "detector": str(det),
                    "local_path": rel_to_run(link, run_dir),
                    "source_local_path": str(det_row.local_path),
                    "download_status": "available_symlink",
                    "sampling_rate": int(det_row.sampling_rate),
                    "duration": float(det_row.duration),
                    "gps_start": float(det_row.gps_start),
                })
        valid_detectors = sorted(set(valid_detectors))
        has_h1l1 = all(d in valid_detectors for d in DETECTORS)
        obj_class, is_ood = class_for_event(row)
        include = bool(pe_path is not None and sky_ok)
        gps = float(row["gps_time"])
        rows.append({
            "source_index": int(src_idx),
            "event_name": event_name,
            "run": infer_run_from_gps(gps),
            "catalog": str(row.get("catalog", "GWTC-4.1")),
            "gps_time": gps,
            "detectors_available": ",".join(valid_detectors),
            "network_snr": row.get("network_snr"),
            "far_per_year": row.get("far_per_year"),
            "p_astro": row.get("p_astro"),
            "mass_1_source": row.get("mass_1_source"),
            "mass_2_source": row.get("mass_2_source"),
            "chirp_mass_source": row.get("chirp_mass_source"),
            "luminosity_distance": row.get("luminosity_distance"),
            "redshift": row.get("redshift"),
            "pe_release_url": row.get("preferred_pe_url"),
            "pe_waveform_family": row.get("preferred_pe_waveform_family"),
            "pe_doi": row.get("preferred_pe_doi"),
            "sky_map_path": rel(pe_path) if pe_path is not None else "",
            "sky_map_format": "pe_hdf5_skymap" if sky_ok else "",
            "sky_map_group": sky_group or "",
            "sky_map_available": bool(sky_ok),
            "h1_l1_strain_available": bool(has_h1l1),
            "embedding_available": bool(include and has_h1l1 and not is_ood),
            "object_class": obj_class,
            "is_ood_for_bbh_encoder": bool(is_ood),
            "include_in_primary_search": bool(include),
            "dq_status": "strain_hdf5_open_ok" if has_h1l1 else "missing_or_invalid_H1L1",
        })

        # Off-source noise windows for O4 run-matched injections.
        if has_h1l1:
            for det in DETECTORS:
                det_row = det_rows[det]
                link_path = run_dir / "data" / "real_strain" / event_name / Path(str(det_row.local_path)).name
                gps_start = float(det_row.gps_start)
                gps_end = gps_start + float(det_row.duration)
                windows = {
                    "real_noise_offsource_left": (gps_start + 64.0, min(gps - 256.0, gps_end - 64.0)),
                    "real_noise_offsource_right": (max(gps + 256.0, gps_start + 64.0), gps_end - 64.0),
                }
                for kind, (seg_start, seg_end) in windows.items():
                    if seg_end - seg_start >= WINDOW_SECONDS:
                        offsource_rows.append({
                            "event_name": event_name,
                            "detector": det,
                            "segment_kind": kind,
                            "segment_start": float(seg_start),
                            "segment_end": float(seg_end),
                            "usable_duration_s": float(seg_end - seg_start),
                            "strain_path": rel_to_run(link_path, run_dir),
                        })

    manifest = pd.DataFrame(rows).sort_values("gps_time").reset_index(drop=True)
    run_dir.joinpath("data").mkdir(parents=True, exist_ok=True)
    manifest.to_csv(run_dir / "data" / "event_manifest_gwtc4.csv", index=False)
    manifest.to_csv(run_dir / "data" / "event_manifest.csv", index=False)
    pd.DataFrame(strain_manifest_rows).to_csv(run_dir / "data" / "strain_gwosc_download_manifest.csv", index=False)
    offsource = pd.DataFrame(offsource_rows)
    inj_dir = run_dir / "data" / "real_noise_injections"
    inj_dir.mkdir(parents=True, exist_ok=True)
    offsource.to_parquet(inj_dir / "metadata.parquet", index=False)

    primary = manifest[manifest["include_in_primary_search"] == True].copy()
    primary_waveform = primary[(primary["h1_l1_strain_available"] == True) & (primary["is_ood_for_bbh_encoder"] == False)].copy()
    payload = {
        "generated_at_utc": utc_now(),
        "source_catalog_events": int(len(manifest)),
        "pe_manifest_events": int(len(pe)),
        "primary_catalog_definition": "PE-supported + HEALPix skymap available",
        "n_primary_events": int(len(primary)),
        "n_primary_pairs": int(len(primary) * (len(primary) - 1) // 2),
        "n_h1l1_primary_events": int(primary["h1_l1_strain_available"].sum()),
        "n_bbh_waveform_eligible_events": int(len(primary_waveform)),
        "n_bbh_waveform_eligible_pairs": int(len(primary_waveform) * (len(primary_waveform) - 1) // 2),
        "n_ood_primary_events": int(primary["is_ood_for_bbh_encoder"].sum()),
        "n_offsource_metadata_rows": int(len(offsource)),
        "detector_coverage_counts": manifest["detectors_available"].value_counts().to_dict(),
        "object_class_counts_primary": primary["object_class"].value_counts().to_dict(),
        "outputs": {
            "event_manifest_gwtc4": str(run_dir / "data" / "event_manifest_gwtc4.csv"),
            "strain_manifest": str(run_dir / "data" / "strain_gwosc_download_manifest.csv"),
            "offsource_metadata": str(inj_dir / "metadata.parquet"),
        },
    }
    write_json(run_dir / "results" / "gwtc4_data_audit.json", payload)
    return payload


class NoiseCache:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.cache: dict[str, tuple[np.ndarray, float, float]] = {}

    def get(self, rel_path: str) -> tuple[np.ndarray, float, float]:
        if rel_path not in self.cache:
            self.cache[rel_path] = load_strain_file(self.run_dir / rel_path)
        return self.cache[rel_path]


def paired_noise_segments(run_dir: Path) -> pd.DataFrame:
    meta = pd.read_parquet(run_dir / "data" / "real_noise_injections" / "metadata.parquet")
    rows = []
    for (event, kind), group in meta.groupby(["event_name", "segment_kind"]):
        dets = {row.detector: row for row in group.itertuples(index=False)}
        if not all(det in dets for det in DETECTORS):
            continue
        start = max(float(dets["H1"].segment_start), float(dets["L1"].segment_start))
        end = min(float(dets["H1"].segment_end), float(dets["L1"].segment_end))
        if end - start < WINDOW_SECONDS:
            continue
        rows.append({
            "event_name": event,
            "segment_kind": kind,
            "segment_start": start,
            "segment_end": end,
            "usable_duration_s": end - start,
            "H1_path": dets["H1"].strain_path,
            "L1_path": dets["L1"].strain_path,
        })
    out = pd.DataFrame(rows).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No paired H1/L1 off-source noise segments available")
    out.to_parquet(run_dir / "data" / "real_noise_injections" / "offsource_noise_segments.parquet", index=False)
    return out


def draw_noise_pair(cache: NoiseCache, segments: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    seg = segments.iloc[int(rng.integers(0, len(segments)))]
    latest_start = float(seg.segment_end) - TARGET_LEN / SAMPLE_RATE
    t0 = float(rng.uniform(float(seg.segment_start), latest_start))
    chans = []
    for det in DETECTORS:
        data, gps_start, _ = cache.get(str(seg[f"{det}_path"]))
        i0 = int(round((t0 - gps_start) * SAMPLE_RATE))
        i0 = max(0, min(i0, len(data) - TARGET_LEN))
        chans.append(data[i0:i0 + TARGET_LEN])
    return zscore_channelwise(np.stack(chans, axis=0)), {
        "noise_event": str(seg.event_name),
        "noise_segment_kind": str(seg.segment_kind),
        "noise_start_gps": t0,
    }


def inject_signal(signal: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    clean = zscore_channelwise(signal.astype(np.float32, copy=False))
    scale = float(rng.lognormal(mean=-0.05, sigma=0.35))
    mixed = zscore_channelwise(noise + scale * clean)
    return mixed.astype(np.float32, copy=False), scale


def materialize_o4_matchroots(run_dir: Path, samples_per_family: int, seed: int) -> dict[str, Any]:
    match_root = run_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    marker = run_dir / "data" / "real_noise_injections" / "materialized_dataset_summary.json"
    if marker.exists():
        return load_json(marker)
    if not SYNTHETIC_CLEAN_ROOT.exists():
        raise FileNotFoundError(SYNTHETIC_CLEAN_ROOT)
    segments = paired_noise_segments(run_dir)
    cache = NoiseCache(run_dir)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n = int(samples_per_family)

    for family in FAMILIES:
        source_dir = SYNTHETIC_CLEAN_ROOT / f"{family}_data_0222"
        out_dir = match_root / f"{family}_data_0222"
        out_dir.mkdir(parents=True, exist_ok=True)
        sig1 = np.load(source_dir / f"{family}_h_strain_1.npy", mmap_mode="r")[:n]
        sig2 = np.load(source_dir / f"{family}_h_strain_2.npy", mmap_mode="r")[:n]
        data1 = np.empty((n, 2, TARGET_LEN), dtype=np.float32)
        data2 = np.empty_like(data1)
        h1 = np.empty_like(data1)
        h2 = np.empty_like(data1)
        for idx in range(n):
            noise_a, meta_a = draw_noise_pair(cache, segments, rng)
            noise_b, meta_b = draw_noise_pair(cache, segments, rng)
            h1[idx] = zscore_channelwise(sig1[idx].astype(np.float32, copy=False))
            h2[idx] = zscore_channelwise(sig2[idx].astype(np.float32, copy=False))
            data1[idx], scale_a = inject_signal(h1[idx], noise_a, rng)
            data2[idx], scale_b = inject_signal(h2[idx], noise_b, rng)
            rows.append({
                "family": family,
                "sample_index": idx,
                "pair_id": f"{family}_{idx:05d}",
                "split": "train" if idx < int(0.7 * n) else ("val" if idx < int(0.85 * n) else "test"),
                "signal_source": rel(source_dir),
                "noise_domain": "O4a_real_offsource_H1L1",
                "noise_source_a": meta_a["noise_event"],
                "noise_source_b": meta_b["noise_event"],
                "noise_segment_kind_a": meta_a["noise_segment_kind"],
                "noise_segment_kind_b": meta_b["noise_segment_kind"],
                "noise_start_gps_a": meta_a["noise_start_gps"],
                "noise_start_gps_b": meta_b["noise_start_gps"],
                "signal_scale_a": scale_a,
                "signal_scale_b": scale_b,
            })
        np.save(out_dir / f"{family}_data_strain_1.npy", data1)
        np.save(out_dir / f"{family}_data_strain_2.npy", data2)
        np.save(out_dir / f"{family}_h_strain_1.npy", h1)
        np.save(out_dir / f"{family}_h_strain_2.npy", h2)
        np.save(out_dir / f"{family}_optimal_SNR_network_1.npy", np.ones(n, dtype=np.float32))
        np.save(out_dir / f"{family}_optimal_SNR_network_2.npy", np.ones(n, dtype=np.float32))

    source_dir = SYNTHETIC_CLEAN_ROOT / "Unlensed_data_0222"
    out_dir = match_root / "Unlensed_data_0222"
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = np.load(source_dir / "unlensed_h_strain.npy", mmap_mode="r")[:n]
    data = np.empty((n, 2, TARGET_LEN), dtype=np.float32)
    clean_arr = np.empty_like(data)
    for idx in range(n):
        noise, meta = draw_noise_pair(cache, segments, rng)
        clean = zscore_channelwise(sig[idx].astype(np.float32, copy=False))
        mixed, scale = inject_signal(clean, noise, rng)
        clean_arr[idx] = clean
        data[idx] = mixed
        rows.append({
            "family": "unlensed",
            "sample_index": idx,
            "pair_id": f"unlensed_{idx:05d}",
            "split": "train" if idx < int(0.7 * n) else ("val" if idx < int(0.85 * n) else "test"),
            "signal_source": rel(source_dir),
            "noise_domain": "O4a_real_offsource_H1L1",
            "noise_source_a": meta["noise_event"],
            "noise_source_b": "",
            "noise_segment_kind_a": meta["noise_segment_kind"],
            "noise_segment_kind_b": "",
            "noise_start_gps_a": meta["noise_start_gps"],
            "noise_start_gps_b": np.nan,
            "signal_scale_a": scale,
            "signal_scale_b": np.nan,
        })
    np.save(out_dir / "unlensed_data_strain.npy", data)
    np.save(out_dir / "unlensed_h_strain.npy", clean_arr)
    np.save(out_dir / "unlensed_optimal_SNR_network.npy", np.ones(n, dtype=np.float32))

    meta = pd.DataFrame(rows)
    meta.to_parquet(run_dir / "data" / "real_noise_injections" / "injection_metadata.parquet", index=False)
    payload = {
        "generated_at_utc": utc_now(),
        "match_root": str(match_root),
        "samples_per_family": int(n),
        "noise_domain": "O4a public off-source H1/L1 strain",
        "synthetic_signal_source": str(SYNTHETIC_CLEAN_ROOT),
        "paired_noise_segments": int(len(segments)),
        "metadata_rows": int(len(meta)),
        "split_counts": meta["split"].value_counts().to_dict(),
        "method_note": "Synthetic clean SIS/PM/unlensed strain is injected into O4a off-source public H1/L1 noise. This is a domain-matched validation/training set, not a real lensing claim.",
    }
    write_json(marker, payload)
    return payload


def cfg_from_checkpoint(raw: dict[str, Any]) -> MatchRunConfig:
    allowed = {f.name for f in fields(MatchRunConfig)}
    kwargs = {}
    for key, value in raw.items():
        if key in allowed:
            kwargs[key] = Path(value) if key in {"data_root", "out_dir"} else value
    return MatchRunConfig(**kwargs)


def load_model_from_checkpoint(path: Path, data_root: Path, samples: int, cpu: bool) -> tuple[torch.nn.Module, MatchRunConfig]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = cfg_from_checkpoint(ckpt["config"])
    cfg.data_root = data_root
    cfg.lensed_limit = int(samples)
    cfg.unlensed_limit = int(samples)
    model = build_model(cfg, in_channels=2)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model.eval().to(device)
    return model, cfg


def eval_checkpoint_on_o4(run_dir: Path, ckpt_path: Path, family: str, samples: int, split: str, cpu: bool) -> dict[str, Any]:
    model, cfg = load_model_from_checkpoint(
        ckpt_path,
        run_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO",
        samples,
        cpu,
    )
    cfg.model_type = family
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits["lensed"][split], splits["unlensed"][split], cfg)
    emb = embed_eval(model, ds, cfg, cpu=cpu)
    metrics = retrieval_metrics(similarity_matrix(emb), ground_truth_partner(ds.meta))
    return {
        "r_at_1": float(metrics["r@1"]),
        "r_at_5": float(metrics["r@5"]),
        "r_at_10": float(metrics["r@10"]),
        "family": family,
        "split": split,
        "median_true_rank": metrics["median_true_rank"],
        "n_eval_events": len(ds),
    }


def transfer_ablation(run_dir: Path, samples: int, epochs: int, cpu: bool) -> dict[str, Any]:
    rows = []
    for family in FAMILIES:
        ckpt = GWTC3_RUN / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "model.pt"
        if not ckpt.exists():
            ckpt = GWTC3_RUN / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep20_clean" / "model.pt"
        for split in ("val", "test"):
            rows.append(eval_checkpoint_on_o4(run_dir, ckpt, family, samples, split, cpu))
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "results" / "gwtc4_transfer_ablation_gate1.csv", index=False)
    test = table[table["split"] == "test"]
    macro = float(test["r_at_10"].mean())
    minfam = float(test["r_at_10"].min())
    payload = {
        "generated_at_utc": utc_now(),
        "method": "O3-noise encoder transferred directly to O4a held-out real-noise injections",
        "passed": bool(macro > 0.6 and minfam > 0.5),
        "pass_rule": {"macro_r_at_10_gt": 0.6, "min_family_r_at_10_gt": 0.5},
        "macro_test_r_at_10": macro,
        "min_family_test_r_at_10": minfam,
        "metrics_csv": str(run_dir / "results" / "gwtc4_transfer_ablation_gate1.csv"),
    }
    write_json(run_dir / "results" / "gwtc4_transfer_ablation_summary.json", payload)
    return payload


def train_o4_encoder(run_dir: Path, samples: int, epochs: int, cpu: bool, force: bool) -> dict[str, Any]:
    cmd = [
        "/root/miniconda3/bin/python",
        "scripts/real_search/05_train_real_noise_encoder.py",
        "--run-dir",
        str(run_dir),
        "--samples-per-family",
        str(samples),
        "--epochs",
        str(epochs),
        "--train-missing",
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd, run_dir / "logs" / f"gwtc4_train_o4_encoder_ep{epochs}.log")
    shutil.copy2(run_dir / "results" / "waveform_gate1_metrics.csv", run_dir / "results" / "gwtc4_waveform_gate1_metrics.csv")
    summary = load_json(run_dir / "results" / "waveform_gate1.json")
    write_json(run_dir / "results" / "gwtc4_waveform_gate1_summary.json", summary)
    return summary


def run_healpix_and_time(run_dir: Path, nside: int) -> None:
    run_cmd([
        "/root/miniconda3/bin/python",
        "scripts/real_search/01_build_healpix_overlap.py",
        "--run-dir",
        str(run_dir),
        "--nside",
        str(nside),
    ], run_dir / "logs" / "gwtc4_healpix_overlap.log")
    shutil.copy2(run_dir / "features" / "real_sky_overlap.parquet", run_dir / "features" / "gwtc4_healpix_overlap.parquet")
    if (run_dir / "features" / "real_sky_overlap.summary.json").exists():
        sky_summary = load_json(run_dir / "features" / "real_sky_overlap.summary.json")
        write_json(run_dir / "results" / "gwtc4_sky_overlap_audit.json", sky_summary)
    run_cmd([
        "/root/miniconda3/bin/python",
        "scripts/real_search/02_build_observable_features.py",
        "--run-dir",
        str(run_dir),
    ], run_dir / "logs" / "gwtc4_observable_features.log")
    shutil.copy2(run_dir / "features" / "real_pair_observable_features.parquet", run_dir / "features" / "gwtc4_time_delay_features.parquet")


def embed_gwtc4_events(run_dir: Path, epochs: int, cpu: bool) -> None:
    cmd = [
        "/root/miniconda3/bin/python",
        "scripts/real_search/06_embed_real_gwtc_events.py",
        "--run-dir",
        str(run_dir),
        "--epochs",
        str(epochs),
    ]
    if cpu:
        cmd.append("--cpu")
    run_cmd(cmd, run_dir / "logs" / "gwtc4_embed_real_events.log")
    shutil.copy2(run_dir / "features" / "real_waveform_embeddings.parquet", run_dir / "features" / "gwtc4_waveform_embeddings.parquet")
    shutil.copy2(run_dir / "features" / "real_waveform_similarity.parquet", run_dir / "features" / "gwtc4_waveform_similarity.parquet")


def audit_gwtc4_waveform(run_dir: Path) -> dict[str, Any]:
    emb = pd.read_parquet(run_dir / "features" / "real_waveform_embeddings.parquet")
    sim = pd.read_parquet(run_dir / "features" / "real_waveform_similarity.parquet")
    event_manifest = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = event_manifest[event_manifest["include_in_primary_search"] == True].sort_values("gps_time").reset_index(drop=True)
    avail = emb[emb["embedding_available"] == True].copy()
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "n_primary_events": int(len(primary)),
        "embedding_available_events": int(len(avail)),
        "n_pairs": int(len(sim)),
        "n_waveform_available_pairs_raw": int(sim["waveform_available"].sum()),
    }
    for family in FAMILIES:
        cols = [c for c in emb.columns if c.startswith(f"{family.lower()}_emb_")]
        arr = avail[cols].to_numpy(dtype=np.float64) if cols else np.empty((0, 0))
        if arr.size:
            norms = np.linalg.norm(arr, axis=1)
            std = arr.std(axis=0)
            payload[family.lower()] = {
                "shape": list(arr.shape),
                "norm_min": float(norms.min()),
                "norm_mean": float(norms.mean()),
                "norm_max": float(norms.max()),
                "dim_std_min": float(std.min()),
                "dim_std_mean": float(std.mean()),
                "dim_std_max": float(std.max()),
                "unique_vectors_rounded_8dp": int(np.unique(np.round(arr, 8), axis=0).shape[0]),
            }
    wf = sim.loc[sim["waveform_available"] == True, "waveform_score"].dropna()
    payload["waveform_score"] = {
        "count": int(len(wf)),
        "n_unique": int(wf.nunique()) if len(wf) else 0,
        "min": float(wf.min()) if len(wf) else None,
        "max": float(wf.max()) if len(wf) else None,
        "std": float(wf.std()) if len(wf) else None,
    }
    collapsed = bool(len(wf) == 0 or wf.nunique() <= 2 or (np.isfinite(wf.std()) and float(wf.std()) < 1e-6))
    payload["deployment_audit_passed"] = not collapsed
    payload["failure_reason"] = "" if not collapsed else "real-event waveform scores are missing or nearly constant"
    write_json(run_dir / "results" / "gwtc4_waveform_deployment_audit.json", payload)
    return payload


def liao_delay_samples() -> np.ndarray:
    import importlib

    liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")
    cfg = liao.LIAO_PRIOR_CONFIG["LIGO"]
    delays, _ = liao.extract_liao_delay_snr_pairs(cfg["image_csv"], cfg["snr_threshold"])
    delays = np.asarray(delays, dtype=np.float64)
    delays = delays[np.isfinite(delays) & (delays > 0)]
    if len(delays) < 10:
        raise RuntimeError("Too few delay prior samples")
    return delays


def log_bins_for(values: np.ndarray, signal: np.ndarray, min_floor: float, n_bins: int = 80) -> np.ndarray:
    merged = np.concatenate([np.asarray(values, dtype=np.float64), np.asarray(signal, dtype=np.float64)])
    merged = merged[np.isfinite(merged) & (merged > 0)]
    lo = max(float(np.nanmin(merged)), min_floor)
    hi = max(float(np.nanmax(merged)), lo * 1.01)
    return np.geomspace(lo, hi * 1.001, n_bins + 1)


def synthetic_validation_observables(meta: list[dict[str, Any]], family: str, delay_prior: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + (0 if family == "SIS" else 100000))
    n = len(meta)
    pair_ids = sorted({int(m["pair_id"]) for m in meta if int(m["pair_id"]) >= 0})
    base_times = {pid: float(rng.uniform(0, 250 * SECONDS_PER_DAY)) for pid in pair_ids}
    delays = {pid: float(rng.choice(delay_prior) * SECONDS_PER_DAY) for pid in pair_ids}
    sky = {pid: (float(rng.uniform(0, 2 * math.pi)), float(math.asin(rng.uniform(-1, 1)))) for pid in pair_ids}
    rows = []
    for idx, m in enumerate(meta):
        pid = int(m["pair_id"])
        if pid >= 0:
            base = base_times[pid]
            delay = delays[pid]
            if m["tag"] == "L1":
                gps = base + rng.normal(0, 0.02)
            else:
                gps = base + delay + rng.normal(0, 0.02)
            ra0, dec0 = sky[pid]
            sigma = float(np.clip(rng.lognormal(mean=math.log(0.12), sigma=0.45), 0.03, 0.6))
            ra = (ra0 + rng.normal(0, sigma / max(math.cos(dec0), 0.2))) % (2 * math.pi)
            dec = float(np.clip(dec0 + rng.normal(0, sigma), -0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4))
        else:
            gps = float(rng.uniform(0, 250 * SECONDS_PER_DAY))
            sigma = float(np.clip(rng.lognormal(mean=math.log(0.20), sigma=0.55), 0.03, 0.8))
            ra = float(rng.uniform(0, 2 * math.pi))
            dec = float(math.asin(rng.uniform(-1, 1)))
        rows.append({"idx": idx, "pair_id": pid, "tag": m["tag"], "gps_obs": gps, "ra_obs": ra, "dec_obs": dec, "sky_sigma_rad": sigma})
    return pd.DataFrame(rows)


def validation_pair_table(run_dir: Path, family: str, epochs: int, samples: int, cpu: bool, delay_prior: np.ndarray) -> pd.DataFrame:
    ckpt = run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "model.pt"
    model, cfg = load_model_from_checkpoint(ckpt, run_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO", samples, cpu)
    cfg.model_type = family
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits["lensed"]["val"], splits["unlensed"]["val"], cfg)
    emb = embed_eval(model, ds, cfg, cpu=cpu)
    wf = similarity_matrix(emb)
    gt = ground_truth_partner(ds.meta)
    events = synthetic_validation_observables(ds.meta, family, delay_prior, seed=20260629)
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
    return pd.DataFrame({
        "family": family,
        "idx_i": ii.astype(np.int32),
        "idx_j": jj.astype(np.int32),
        "is_true_pair": (gt[ii] == jj).astype(np.int8),
        "waveform_score": wf[ii, jj].astype(np.float32),
        "time_score": time_score.astype(np.float32),
        "sky_score": sky_score,
        "delta_t_days": dt_days.astype(np.float64),
        "sky_norm_sep": norm_sep.astype(np.float32),
        "event_count": int(len(events)),
    })


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
        label = np.full((n, n), False)
        ii = df["idx_i"].to_numpy(dtype=np.int32)
        jj = df["idx_j"].to_numpy(dtype=np.int32)
        label[ii, jj] = df["is_true_pair"].to_numpy(dtype=bool)
        label[jj, ii] = label[ii, jj]
        score = np.zeros((n, n), dtype=np.float32)
        for ch in ("waveform", "time", "sky"):
            score += float(weights[ch]) * row_z_neutral(matrix_from_pairs(df, n, f"{ch}_score"))
        np.fill_diagonal(score, -np.inf)
        ranks = []
        for q in np.where(label.any(axis=1))[0]:
            partner = int(np.where(label[q])[0][0])
            true_score = score[q, partner]
            ranks.append(int(1 + np.sum(score[q] > true_score)))
        ranks = np.asarray(ranks)
        rows.append({
            "family": family,
            "r_at_1": float(np.mean(ranks <= 1)),
            "r_at_5": float(np.mean(ranks <= 5)),
            "r_at_10": float(np.mean(ranks <= 10)),
            "median_rank": float(np.median(ranks)),
        })
    out = {}
    for key in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
        out[f"macro_{key}"] = float(np.mean([r[key] for r in rows]))
    for r in rows:
        fam = r["family"].lower()
        for key, value in r.items():
            if key != "family":
                out[f"{fam}_{key}"] = float(value)
    return out


def select_weights(
    validation: pd.DataFrame,
    force_waveform_zero: bool = False,
    require_all_positive: bool = False,
    require_time_sky_positive: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    grid_rows = []
    for ww in WEIGHT_GRID:
        if force_waveform_zero and ww != 0.0:
            continue
        for wt in WEIGHT_GRID:
            for ws in WEIGHT_GRID:
                if ww == 0.0 and wt == 0.0 and ws == 0.0:
                    continue
                if require_all_positive and (ww <= 0.0 or wt <= 0.0 or ws <= 0.0):
                    continue
                if require_time_sky_positive and (wt <= 0.0 or ws <= 0.0):
                    continue
                weights = {"waveform": ww, "time": wt, "sky": ws}
                grid_rows.append({**weights, **validation_metrics_for_weights(validation, weights)})
    grid = pd.DataFrame(grid_rows).sort_values(
        ["macro_r_at_10", "macro_r_at_5", "macro_r_at_1", "waveform", "time", "sky"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return {"waveform": float(best.waveform), "time": float(best.time), "sky": float(best.sky)}, grid


def add_real_scores(real: pd.DataFrame, weights: dict[str, float], name: str) -> pd.DataFrame:
    n = int(max(real["idx_i"].max(), real["idx_j"].max()) + 1)
    out = real.copy()
    for ch in ("waveform", "time", "sky"):
        mat = matrix_from_pairs(out, n, f"{ch}_score")
        z = row_z_neutral(mat)
        ii = out["idx_i"].to_numpy(dtype=np.int32)
        jj = out["idx_j"].to_numpy(dtype=np.int32)
        out[f"{ch}_rowz_i_to_j"] = z[ii, jj]
        out[f"{ch}_rowz_j_to_i"] = z[jj, ii]
        out[f"{ch}_rowz_mean"] = 0.5 * (z[ii, jj] + z[jj, ii])
        if ch == "waveform":
            wfz = z
        elif ch == "time":
            tz = z
        else:
            sz = z
    score = weights["waveform"] * wfz + weights["time"] * tz + weights["sky"] * sz
    ii = out["idx_i"].to_numpy(dtype=np.int32)
    jj = out["idx_j"].to_numpy(dtype=np.int32)
    out[f"{name}_score_i_to_j"] = score[ii, jj]
    out[f"{name}_score_j_to_i"] = score[jj, ii]
    out["final_score"] = np.maximum(score[ii, jj], score[jj, ii])
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["empirical_catalog_tail_rank_fraction"] = out["rank"] / float(len(out))
    return out


def shortlist(scores: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    out = scores.head(n).copy()
    cols = [
        "rank", "event_i", "event_j", "final_score", "waveform_score", "time_score", "healpix_sky_score",
        "delta_t_days", "healpix_overlap", "waveform_available", "full_waveform_scored", "pair_has_ood",
        "object_class_i", "object_class_j", "detector_coverage", "snr_ratio", "snr_score",
        "empirical_catalog_tail_rank_fraction",
    ]
    out["healpix_sky_score"] = out["sky_score"]
    out["healpix_overlap"] = out["sky_cosine_overlap"]
    out["detector_coverage"] = out["detectors_i"].astype(str) + " | " + out["detectors_j"].astype(str)
    return out[cols]


def fuse_gwtc4(run_dir: Path, epochs: int, samples: int, cpu: bool) -> dict[str, Any]:
    delay_prior = liao_delay_samples()
    validation = pd.concat(
        [validation_pair_table(run_dir, fam, epochs, samples, cpu, delay_prior) for fam in FAMILIES],
        ignore_index=True,
    )
    validation.to_parquet(run_dir / "results" / "gwtc4_fusion_validation_waveform_time_sky.parquet", index=False)
    weights, grid = select_weights(validation, force_waveform_zero=False, require_all_positive=True)
    baseline_weights, baseline_grid = select_weights(validation, force_waveform_zero=True, require_time_sky_positive=True)
    grid.to_csv(run_dir / "results" / "gwtc4_fusion_weight_grid_waveform_time_sky.csv", index=False)
    baseline_grid.to_csv(run_dir / "results" / "gwtc4_fusion_weight_grid_time_sky_baseline.csv", index=False)

    gate = load_json(run_dir / "results" / "waveform_gate1.json")
    deploy = load_json(run_dir / "results" / "gwtc4_waveform_deployment_audit.json")
    if not bool(gate.get("passed", False)) or not bool(deploy.get("deployment_audit_passed", False)):
        final_weights = dict(baseline_weights)
        final_weights["waveform"] = 0.0
        waveform_policy = "fallback_to_time_sky_due_to_failed_gate_or_deployment_audit"
    else:
        final_weights = weights
        waveform_policy = "validation_selected_waveform_time_sky"

    obs = pd.read_parquet(run_dir / "features" / "real_pair_observable_features.parquet")
    sim = pd.read_parquet(run_dir / "features" / "real_waveform_similarity.parquet")
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[events["include_in_primary_search"] == True].sort_values("gps_time").reset_index(drop=True)
    event_class = primary["object_class"].to_numpy()
    event_ood = primary["is_ood_for_bbh_encoder"].to_numpy(dtype=bool)
    real = obs.merge(sim[["idx_i", "idx_j", "waveform_available", "waveform_score", "waveform_score_sis", "waveform_score_pm"]], on=["idx_i", "idx_j"], how="left")
    real["object_class_i"] = event_class[real["idx_i"].to_numpy(dtype=np.int32)]
    real["object_class_j"] = event_class[real["idx_j"].to_numpy(dtype=np.int32)]
    real["pair_has_ood"] = event_ood[real["idx_i"].to_numpy(dtype=np.int32)] | event_ood[real["idx_j"].to_numpy(dtype=np.int32)]
    real["full_waveform_scored"] = real["waveform_available"].fillna(False) & (~real["pair_has_ood"])
    real["raw_waveform_score"] = real["waveform_score"]
    real.loc[~real["full_waveform_scored"], "waveform_score"] = np.nan
    selected = add_real_scores(real, final_weights, "waveform_time_sky")
    baseline = add_real_scores(real, baseline_weights, "time_sky")
    forced = add_real_scores(real, {"waveform": 1.0, "time": 1.0, "sky": 1.0}, "forced_equal_waveform_time_sky")
    selected.to_parquet(run_dir / "results" / "gwtc4_pair_scores_waveform_time_sky.parquet", index=False)
    selected.to_parquet(run_dir / "results" / "real_pair_scores_waveform_time_sky.parquet", index=False)
    baseline.to_parquet(run_dir / "results" / "gwtc4_pair_scores_time_sky_baseline.parquet", index=False)
    forced.to_parquet(run_dir / "results" / "gwtc4_pair_scores_forced_equal_weights_waveform_time_sky.parquet", index=False)
    shortlist(selected).to_csv(run_dir / "results" / "gwtc4_candidate_shortlist_waveform_time_sky.csv", index=False)
    shortlist(selected).to_csv(run_dir / "results" / "candidate_shortlist_waveform_time_sky.csv", index=False)
    shortlist(baseline).to_csv(run_dir / "results" / "gwtc4_candidate_shortlist_time_sky_baseline.csv", index=False)
    shortlist(forced).to_csv(run_dir / "results" / "gwtc4_candidate_shortlist_forced_equal_weights_waveform_time_sky.csv", index=False)
    wf = real.loc[real["full_waveform_scored"], "raw_waveform_score"].dropna()
    top = selected.iloc[0].to_dict()
    full_subset = selected[selected["full_waveform_scored"] == True].copy()
    full_top = full_subset.iloc[0].to_dict() if len(full_subset) else {}
    payload = {
        "generated_at_utc": utc_now(),
        "method": "GWTC-4 waveform + time-delay + real HEALPix sky-overlap real-catalog ranking",
        "snr_policy": "SNR/amplitude is audit-only and is not used in final_score or shortlist rank.",
        "weight_policy": waveform_policy + "_constrained_positive_three_channel",
        "selected_weights_validation": weights,
        "time_sky_baseline_weights": baseline_weights,
        "final_weights": final_weights,
        "validation_selected_metrics": grid.iloc[0].to_dict(),
        "time_sky_baseline_validation_metrics": baseline_grid.iloc[0].to_dict(),
        "n_events": int(len(primary)),
        "n_pairs": int(len(selected)),
        "n_full_waveform_scored_pairs": int(selected["full_waveform_scored"].sum()),
        "n_ood_events": int(primary["is_ood_for_bbh_encoder"].sum()),
        "real_waveform_score_audit": {
            "available_bbh_pairs": int(len(wf)),
            "n_unique_scores": int(wf.nunique()) if len(wf) else 0,
            "min": float(wf.min()) if len(wf) else None,
            "max": float(wf.max()) if len(wf) else None,
            "std": float(wf.std()) if len(wf) else None,
        },
        "top_pair": {k: (float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else (int(v) if isinstance(v, (np.integer, int)) else str(v))) for k, v in top.items() if k in {"rank", "event_i", "event_j", "final_score", "waveform_score", "time_score", "sky_score", "waveform_available", "full_waveform_scored", "pair_has_ood"}},
        "top_pair_is_full_waveform_scored": bool(top.get("full_waveform_scored", False)),
        "top_pair_has_ood": bool(top.get("pair_has_ood", False)),
        "full_waveform_subset_top_pair": {k: (float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else (int(v) if isinstance(v, (np.integer, int)) else str(v))) for k, v in full_top.items() if k in {"rank", "event_i", "event_j", "final_score", "waveform_score", "time_score", "sky_score", "waveform_available", "full_waveform_scored", "pair_has_ood"}},
        "outputs": {
            "candidate_shortlist": str(run_dir / "results" / "gwtc4_candidate_shortlist_waveform_time_sky.csv"),
            "pair_scores": str(run_dir / "results" / "gwtc4_pair_scores_waveform_time_sky.parquet"),
        },
    }
    write_json(run_dir / "results" / "gwtc4_channel_weights_waveform_time_sky.json", payload)
    write_json(run_dir / "results" / "gwtc4_final_fusion_summary.json", payload)
    return payload


def summarize_gwtc3(run_dir: Path) -> dict[str, Any]:
    summary = load_json(GWTC3_RUN / "results" / "final_fusion_summary.json")
    gate = pd.read_csv(GWTC3_RUN / "results" / "waveform_gate1_metrics.csv")
    top = pd.read_csv(GWTC3_RUN / "results" / "candidate_shortlist_waveform_time_sky.csv").head(20)
    pairs = pd.read_parquet(GWTC3_RUN / "results" / "real_pair_scores_waveform_time_sky.parquet")
    payload = {
        "source_run": str(GWTC3_RUN),
        "n_events": int(summary.get("n_events", 63)),
        "n_pairs": int(summary.get("n_pairs", len(pairs))),
        "waveform_available_events": int(load_json(GWTC3_RUN / "features" / "real_waveform_embedding_summary.json").get("n_embedded_events", 55)),
        "waveform_available_pairs": int(summary.get("n_waveform_pairs_available", pairs["waveform_available"].sum())),
        "selected_weights": summary.get("selected_weights"),
        "gate1_test_macro_r_at_10": float(gate[gate["split"] == "test"]["r_at_10"].mean()),
        "gate1_test_min_family_r_at_10": float(gate[gate["split"] == "test"]["r_at_10"].min()),
        "top_pair": top.iloc[0].to_dict(),
        "gw170104_gw170814_rank": summary.get("gw170104_gw170814_rank", {}),
        "top20_path": str(GWTC3_RUN / "results" / "candidate_shortlist_waveform_time_sky.csv"),
    }
    write_json(run_dir / "results" / "gwtc3_baseline_summary.json", payload)
    top.to_csv(run_dir / "results" / "gwtc3_top20_candidate_shortlist_waveform_time_sky.csv", index=False)
    return payload


def cross_run_summary(run_dir: Path, gwtc3: dict[str, Any], gwtc4: dict[str, Any]) -> dict[str, Any]:
    gate4 = pd.read_csv(run_dir / "results" / "gwtc4_waveform_gate1_metrics.csv")
    data4 = load_json(run_dir / "results" / "gwtc4_data_audit.json")
    row3 = {
        "catalog": "GWTC-3",
        "n_events": gwtc3["n_events"],
        "n_pairs": gwtc3["n_pairs"],
        "waveform_available_events": gwtc3["waveform_available_events"],
        "waveform_available_pairs": gwtc3["waveform_available_pairs"],
        "gate1_macro_r_at_10": gwtc3["gate1_test_macro_r_at_10"],
        "gate1_min_family_r_at_10": gwtc3["gate1_test_min_family_r_at_10"],
        "selected_weights": json.dumps(gwtc3["selected_weights"]),
        "top_pair": f"{gwtc3['top_pair']['event_i']}--{gwtc3['top_pair']['event_j']}",
        "top_pair_waveform_available": bool(gwtc3["top_pair"]["waveform_available"]),
        "top_pair_ood": False,
        "gw170104_gw170814_rank": gwtc3["gw170104_gw170814_rank"].get("waveform_time_sky"),
    }
    row4 = {
        "catalog": "GWTC-4.1",
        "n_events": gwtc4["n_events"],
        "n_pairs": gwtc4["n_pairs"],
        "waveform_available_events": data4["n_bbh_waveform_eligible_events"],
        "waveform_available_pairs": gwtc4["n_full_waveform_scored_pairs"],
        "gate1_macro_r_at_10": float(gate4[gate4["split"] == "test"]["r_at_10"].mean()),
        "gate1_min_family_r_at_10": float(gate4[gate4["split"] == "test"]["r_at_10"].min()),
        "selected_weights": json.dumps(gwtc4["final_weights"]),
        "top_pair": f"{gwtc4['top_pair'].get('event_i')}--{gwtc4['top_pair'].get('event_j')}",
        "top_pair_waveform_available": bool(gwtc4["top_pair_is_full_waveform_scored"]),
        "top_pair_ood": bool(gwtc4["top_pair_has_ood"]),
        "gw170104_gw170814_rank": "",
    }
    df = pd.DataFrame([row3, row4])
    df.to_csv(run_dir / "results" / "gwtc34_cross_run_summary.csv", index=False)
    payload = {"generated_at_utc": utc_now(), "rows": df.to_dict(orient="records")}
    write_json(run_dir / "results" / "gwtc34_cross_run_summary.json", payload)
    write_json(run_dir / "results" / "gwtc34_lvk_crosscheck_summary.json", {
        "generated_at_utc": utc_now(),
        "GW170104--GW170814": gwtc3["gw170104_gw170814_rank"],
        "note": "Ranks are empirical catalog ranks for follow-up context, not p-values or detection significance.",
    })
    return payload


def make_figures(run_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    g3_weights = load_json(GWTC3_RUN / "results" / "channel_weights_waveform_time_sky.json")
    g4_weights = load_json(run_dir / "results" / "gwtc4_channel_weights_waveform_time_sky.json")
    g3_scores = pd.read_parquet(GWTC3_RUN / "results" / "real_pair_scores_waveform_time_sky.parquet")
    g4_scores = pd.read_parquet(run_dir / "results" / "gwtc4_pair_scores_waveform_time_sky.parquet")
    g3_top = pd.read_csv(GWTC3_RUN / "results" / "candidate_shortlist_waveform_time_sky.csv").head(10)
    g4_top = pd.read_csv(run_dir / "results" / "gwtc4_candidate_shortlist_waveform_time_sky.csv").head(10)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    for row, label, weights, scores, top in [
        (0, "GWTC-3", g3_weights, g3_scores, g3_top),
        (1, "GWTC-4.1", g4_weights, g4_scores, g4_top),
    ]:
        selected_metrics = weights.get("selected_validation_metrics") or weights.get("validation_selected_metrics")
        baseline_metrics = weights.get("time_sky_baseline_validation_metrics")
        ax = axes[row, 0]
        vals = [
            baseline_metrics["macro_r_at_1"],
            selected_metrics["macro_r_at_1"],
            baseline_metrics["macro_r_at_10"],
            selected_metrics["macro_r_at_10"],
        ]
        ax.bar([0, 1, 3, 4], vals, color=["#8da0cb", "#66c2a5", "#8da0cb", "#66c2a5"])
        ax.set_xticks([0.5, 3.5], ["R@1", "R@10"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{label}\nheld-out validation")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(["time+sky", "waveform+time+sky"], frameon=False, loc="lower right")

        ax = axes[row, 1]
        ax.hist(scores["final_score"].to_numpy(), bins=45, color="#737373", alpha=0.85)
        ax.set_xlabel("catalog final score")
        ax.set_ylabel("unordered pairs")
        ax.set_title(f"{label} real pair scores")
        ax.grid(True, alpha=0.22)

        ax = axes[row, 2]
        y = np.arange(len(top))[::-1]
        colors = np.where(top["waveform_available"].fillna(False).to_numpy(), "#1b9e77", "#d95f02")
        if "pair_has_ood" in top.columns:
            colors = np.where(top["pair_has_ood"].fillna(False).to_numpy(), "#7570b3", colors)
        ax.scatter(top["final_score"], y, c=colors, s=42)
        ax.set_yticks(y, [f"{int(r)}" for r in top["rank"]])
        ax.set_xlabel("final score")
        ax.set_ylabel("rank")
        ax.set_title(f"{label} top-ranked pairs")
        ax.grid(True, alpha=0.22)
    fig.suptitle("Real GWTC catalog lens-candidate shortlist; ranked coincidences, not detections", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / "figures" / f"fig_real_gwtc34_search.{ext}", dpi=300)
    plt.close(fig)


def write_reports(run_dir: Path, gwtc3: dict[str, Any], gwtc4: dict[str, Any], transfer: dict[str, Any], package_path: Path | None = None) -> None:
    cross = pd.read_csv(run_dir / "results" / "gwtc34_cross_run_summary.csv")
    top4 = pd.read_csv(run_dir / "results" / "gwtc4_candidate_shortlist_waveform_time_sky.csv").head(10)
    data4 = load_json(run_dir / "results" / "gwtc4_data_audit.json")
    gate4 = load_json(run_dir / "results" / "gwtc4_waveform_gate1_summary.json")
    cn = f"""# GWTC-3/GWTC-4.1 真实目录 lens-candidate refinement 报告

生成时间：{utc_now()}

## 结论先行

本结果是 candidate shortlist for Bayesian follow-up，不是透镜探测声明。主排序严格使用三通道：waveform score、time-delay score、real HEALPix sky-overlap score。SNR/amplitude 只作为 audit column 保留，没有进入 final_score。

GWTC-3 使用已有 O3-noise encoder 和既有权威结果，不重跑主结果。GWTC-4.1 使用 O4a off-source public H1/L1 strain 注入训练的 run-matched encoder；O3->O4a 只作为 transfer ablation。

GWTC-4.1 的主融合采用 constrained three-channel validation selection：waveform、time-delay、HEALPix sky 三个通道权重均必须为正，避免把某一主通道在真实部署中关闭。

## GWTC-4.1 数据口径

- 初始 GWTC-4.1 event API 事件数：{data4['source_catalog_events']}
- PE-supported + HEALPix skymap available primary events：{data4['n_primary_events']}
- primary unordered pairs：{data4['n_primary_pairs']}
- H1-L1 且 BBH waveform-eligible events：{data4['n_bbh_waveform_eligible_events']}
- full-waveform-scored BBH pairs：{gwtc4['n_full_waveform_scored_pairs']}
- OOD/NSBH/低质量或未知事件数：{data4['n_ood_primary_events']}

## Waveform Gate-1

- GWTC-4 O4-noise run-matched Gate-1 passed：{gate4.get('passed')}
- macro test R@10：{gate4.get('macro_test_r_at_10')}
- min-family test R@10：{gate4.get('min_family_test_r_at_10')}
- O3->O4a transfer ablation passed：{transfer.get('passed')}
- O3->O4a macro test R@10：{transfer.get('macro_test_r_at_10')}

## GWTC-4.1 主排序

- final weights：`{json.dumps(gwtc4['final_weights'], ensure_ascii=False)}`
- waveform policy：{gwtc4['weight_policy']}
- 榜首：{gwtc4['top_pair'].get('event_i')} -- {gwtc4['top_pair'].get('event_j')}
- 榜首是否 full-waveform-scored：{gwtc4['top_pair_is_full_waveform_scored']}
- 榜首是否含 OOD：{gwtc4['top_pair_has_ood']}

## GWTC-4.1 Top 10

{top4.to_markdown(index=False)}

## GWTC-3/GWTC-4.1 并排摘要

{cross.to_markdown(index=False)}

## 为什么不合并目录作为主结果

GWTC-3 和 GWTC-4.1 采用 run-matched encoder。O3 与 O4a 的噪声域、搜索管线和 PE release 口径不同。除非使用单一 shared encoder 重新嵌入全部事件，并完成跨 run score calibration 的 QQ/KS 检查，否则 merged catalog 只能作为 future work 或 diagnostic，不作为主结果。

## 输出

- 主图：`figures/fig_real_gwtc34_search.pdf`
- GWTC-4 candidate shortlist：`results/gwtc4_candidate_shortlist_waveform_time_sky.csv`
- GWTC-4 pair scores：`results/gwtc4_pair_scores_waveform_time_sky.parquet`
- 跨 run 摘要：`results/gwtc34_cross_run_summary.csv`
- deliverables package：`{package_path or ''}`
"""
    en = cn.replace("真实目录", "real catalog").replace("结论先行", "Executive Summary")
    (run_dir / "real_search_gwtc34_report_cn.md").write_text(cn, encoding="utf-8")
    (run_dir / "real_search_gwtc34_report_en.md").write_text(en, encoding="utf-8")
    readme = f"""# Current GWTC-3/GWTC-4.1 Real-Data Deployment Results

This package contains a lens-candidate refinement / candidate-shortlist workflow, not a lensing detection claim.

Main score channels: waveform + time-delay + real HEALPix sky-overlap. SNR/amplitude is audit-only.

Run directory: `{run_dir}`
Package: `{package_path or ''}`
"""
    (run_dir / "README_CURRENT_RESULTS.md").write_text(readme, encoding="utf-8")
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
cd {REPO_ROOT}
/root/miniconda3/bin/python scripts/real_search/15_gwtc34_real_deployment.py --run-dir {run_dir} --all
"""
    path = run_dir / "reproduce.sh"
    path.write_text(reproduce, encoding="utf-8")
    path.chmod(0o755)


def package_outputs(run_dir: Path) -> Path:
    pkg_dir = REPO_ROOT / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    package_path = pkg_dir / f"{run_dir.name}_deliverables.tar.gz"
    include = [
        run_dir / "data" / "event_manifest_gwtc4.csv",
        run_dir / "data" / "strain_gwosc_download_manifest.csv",
        run_dir / "features" / "gwtc4_healpix_overlap.parquet",
        run_dir / "features" / "gwtc4_time_delay_features.parquet",
        run_dir / "features" / "gwtc4_waveform_embeddings.parquet",
        run_dir / "features" / "gwtc4_waveform_similarity.parquet",
        run_dir / "results",
        run_dir / "figures",
        run_dir / "real_search_gwtc34_report_cn.md",
        run_dir / "real_search_gwtc34_report_en.md",
        run_dir / "README_CURRENT_RESULTS.md",
        run_dir / "reproduce.sh",
        REPO_ROOT / "scripts" / "real_search",
    ]
    with tarfile.open(package_path, "w:gz") as tar:
        for p in include:
            if not p.exists():
                continue
            arc = p.relative_to(REPO_ROOT)
            tar.add(p, arcname=str(arc), recursive=True)
    return package_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    ensure_run_dirs(run_dir)
    (run_dir / "run_config.json").write_text(json.dumps({
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "samples_per_family": args.samples_per_family,
        "epochs": args.epochs,
        "nside": args.nside,
        "main_score_channels": ["waveform", "time-delay", "real HEALPix sky"],
        "snr_policy": "audit-only",
        "gwtc3_policy": "existing O3-noise encoder result unchanged",
        "gwtc4_policy": "O4-noise run-matched encoder",
    }, indent=2), encoding="utf-8")

    print("Phase 0: GWTC-3 baseline", flush=True)
    gwtc3 = summarize_gwtc3(run_dir)
    print("Phase 1: GWTC-4 manifest", flush=True)
    data4 = build_gwtc4_manifest(run_dir)
    print(json.dumps(data4, indent=2, ensure_ascii=False), flush=True)
    print("Phase 2-3: GWTC-4 HEALPix/time features", flush=True)
    run_healpix_and_time(run_dir, args.nside)
    print("Phase 4-5: O4 real-noise injection dataset and encoders", flush=True)
    materialize_o4_matchroots(run_dir, args.samples_per_family, seed=20260629)
    transfer = transfer_ablation(run_dir, args.samples_per_family, args.epochs, args.cpu)
    print(json.dumps(transfer, indent=2, ensure_ascii=False), flush=True)
    train_o4_encoder(run_dir, args.samples_per_family, args.epochs, args.cpu, args.force_train)
    print("Phase 6: GWTC-4 real-event waveform embeddings", flush=True)
    embed_gwtc4_events(run_dir, args.epochs, args.cpu)
    audit = audit_gwtc4_waveform(run_dir)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    print("Phase 7: GWTC-4 fusion and ranking", flush=True)
    gwtc4 = fuse_gwtc4(run_dir, args.epochs, args.samples_per_family, args.cpu)
    print(json.dumps(gwtc4, indent=2, ensure_ascii=False), flush=True)
    print("Phase 8-10: comparison and figures", flush=True)
    cross_run_summary(run_dir, gwtc3, gwtc4)
    make_figures(run_dir)
    print("Phase 11-12: reports and package", flush=True)
    write_reports(run_dir, gwtc3, gwtc4, transfer)
    package_path = package_outputs(run_dir)
    write_reports(run_dir, gwtc3, gwtc4, transfer, package_path)
    package_path = package_outputs(run_dir)
    print(json.dumps({
        "status": "complete",
        "run_dir": str(run_dir),
        "package": str(package_path),
        "gwtc4_primary_events": data4["n_primary_events"],
        "gwtc4_pairs": data4["n_primary_pairs"],
        "gwtc4_gate1_passed": load_json(run_dir / "results" / "gwtc4_waveform_gate1_summary.json").get("passed"),
        "gwtc4_top_pair": gwtc4["top_pair"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
