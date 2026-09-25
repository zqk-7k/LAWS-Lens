from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.config import MatchRunConfig
from matchgw.data import pad_or_trim, peak_flip_channels, spectral_preprocess, to_channels, zscore_channels
from matchgw.pipeline import build_model
from scripts.real_search.common import default_run_dir, ensure_run_dirs, pair_indices, utc_now, write_json


FAMILIES = ("SIS", "PM")
DETECTORS = ("H1", "L1")
SAMPLE_RATE = 4096.0
WINDOW_SECONDS = 24.0
END_OFFSET_SECONDS = 0.25
MIN_FINITE_FRACTION_PER_DETECTOR = 0.99


def cfg_from_checkpoint_config(raw: dict) -> MatchRunConfig:
    allowed = {f.name for f in fields(MatchRunConfig)}
    kwargs = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        if key in {"data_root", "out_dir"}:
            kwargs[key] = Path(value)
        else:
            kwargs[key] = value
    return MatchRunConfig(**kwargs)


def load_model(path: Path, cpu: bool = False) -> tuple[torch.nn.Module, MatchRunConfig]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = cfg_from_checkpoint_config(ckpt["config"])
    model = build_model(cfg, in_channels=2)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model.eval().to(device)
    return model, cfg


def load_strain(path: Path) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as h5:
        strain = np.asarray(h5["strain/Strain"][:], dtype=np.float32)
        start = float(np.asarray(h5["meta/GPSstart"][()]).item())
        duration = float(np.asarray(h5["meta/Duration"][()]).item())
    return strain, start, duration


def extract_window(
    run_dir: Path,
    rows_by_det: dict[str, pd.Series],
    gps_time: float,
) -> tuple[np.ndarray | None, str, dict[str, float]]:
    chans = []
    finite_fractions: dict[str, float] = {det: 0.0 for det in DETECTORS}
    end_gps = float(gps_time) + END_OFFSET_SECONDS
    start_gps = end_gps - WINDOW_SECONDS
    n = int(round(WINDOW_SECONDS * SAMPLE_RATE))
    for det in DETECTORS:
        row = rows_by_det.get(det)
        if row is None:
            return None, f"missing_{det}_download", finite_fractions
        rel = str(getattr(row, "local_path"))
        path = run_dir / rel
        if not path.exists() or str(getattr(row, "download_status", "")) == "no_gwosc_url":
            return None, f"missing_{det}_file", finite_fractions
        data, gps_start, duration = load_strain(path)
        sample_rate = len(data) / duration
        i0 = int(round((start_gps - gps_start) * sample_rate))
        i1 = i0 + n
        if i0 < 0 or i1 > len(data):
            return None, f"{det}_window_outside_file", finite_fractions
        if abs(sample_rate - SAMPLE_RATE) > 1e-3:
            return None, f"{det}_unexpected_sample_rate_{sample_rate:.3f}", finite_fractions
        window = data[i0:i1]
        finite_fraction = float(np.mean(np.isfinite(window)))
        finite_fractions[det] = finite_fraction
        if finite_fraction < MIN_FINITE_FRACTION_PER_DETECTOR:
            return None, f"{det}_nonfinite_window_{finite_fraction:.6f}", finite_fractions
        chans.append(window)
    return np.stack(chans, axis=0).astype(np.float32), "ok", finite_fractions


def prepare_waveform(x: np.ndarray, cfg: MatchRunConfig) -> torch.Tensor:
    y = pad_or_trim(x, cfg.target_len, cfg.stride)
    y = spectral_preprocess(y, cfg)
    if cfg.aug_flip:
        y = peak_flip_channels(y)
    y = to_channels(zscore_channels(y), cfg.use_hilbert)
    return torch.from_numpy(y[None, ...].astype(np.float32, copy=False))


@torch.no_grad()
def embed_one(model: torch.nn.Module, cfg: MatchRunConfig, x: np.ndarray, cpu: bool) -> np.ndarray:
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    batch = prepare_waveform(x, cfg).to(device)
    return model(batch).float().cpu().numpy()[0].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)].copy()
    primary = primary.sort_values("gps_time").reset_index(drop=True)
    downloads = pd.read_csv(run_dir / "data" / "strain_gwosc_download_manifest.csv")
    download_groups = {event: {row.detector: row for row in group.itertuples(index=False)} for event, group in downloads.groupby("event_name")}

    models = {}
    cfgs = {}
    for family in FAMILIES:
        path = run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{args.epochs}_clean" / "model.pt"
        models[family], cfgs[family] = load_model(path, cpu=args.cpu)

    rows = []
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for idx, ev in primary.iterrows():
        event_name = str(ev["event_name"])
        x, status, finite_fractions = extract_window(
            run_dir, download_groups.get(event_name, {}), float(ev["gps_time"])
        )
        row = {
            "idx": int(idx),
            "event_name": event_name,
            "gps_time": float(ev["gps_time"]),
            "waveform_status": status,
            "embedding_available": status == "ok",
            "h1_finite_fraction_24s": finite_fractions["H1"],
            "l1_finite_fraction_24s": finite_fractions["L1"],
            "embedding_note": (
                "H1/L1 24 s window ending at GPS+0.25 s; each detector must have "
                f">={MIN_FINITE_FRACTION_PER_DETECTOR:.2f} finite samples before any zero filling"
            ),
        }
        if x is not None:
            embeddings[event_name] = {}
            for family in FAMILIES:
                emb = embed_one(models[family], cfgs[family], x, cpu=args.cpu)
                embeddings[event_name][family] = emb
                for k, value in enumerate(emb):
                    row[f"{family.lower()}_emb_{k:03d}"] = float(value)
        rows.append(row)

    emb_df = pd.DataFrame(rows)
    emb_df.to_parquet(run_dir / "features" / "real_waveform_embeddings.parquet", index=False)
    emb_df.to_csv(run_dir / "features" / "real_waveform_embedding_audit.csv", index=False)

    ii, jj = pair_indices(len(primary))
    sim_rows = []
    names = primary["event_name"].astype(str).to_numpy()
    for i, j in zip(ii, jj):
        a = names[int(i)]
        b = names[int(j)]
        row = {
            "idx_i": int(i),
            "idx_j": int(j),
            "event_i": a,
            "event_j": b,
            "waveform_available": bool(a in embeddings and b in embeddings),
        }
        vals = []
        for family in FAMILIES:
            if a in embeddings and b in embeddings:
                score = float(np.dot(embeddings[a][family], embeddings[b][family]))
                vals.append(score)
            else:
                score = np.nan
            row[f"waveform_score_{family.lower()}"] = score
        row["waveform_score"] = float(np.nanmean(vals)) if vals else np.nan
        sim_rows.append(row)
    sim = pd.DataFrame(sim_rows)
    sim.to_parquet(run_dir / "features" / "real_waveform_similarity.parquet", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "n_primary_events": int(len(primary)),
        "n_embedded_events": int(emb_df["embedding_available"].sum()),
        "n_pairs": int(len(sim)),
        "n_pairs_with_waveform": int(sim["waveform_available"].sum()),
        "window_seconds": WINDOW_SECONDS,
        "window_end_offset_seconds": END_OFFSET_SECONDS,
        "minimum_finite_fraction_per_detector": MIN_FINITE_FRACTION_PER_DETECTOR,
        "output_embeddings": str(run_dir / "features" / "real_waveform_embeddings.parquet"),
        "output_similarity": str(run_dir / "features" / "real_waveform_similarity.parquet"),
        "note": "Waveform similarity is exploratory real-event embedding. Final real-catalog fusion may set waveform weight to zero unless a validation-selected weight is available.",
    }
    write_json(run_dir / "features" / "real_waveform_embedding_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
