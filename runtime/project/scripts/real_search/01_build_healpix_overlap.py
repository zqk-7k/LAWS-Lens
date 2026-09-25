from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import (
    angular_sep_rad,
    default_run_dir,
    ensure_run_dirs,
    map_point_estimates,
    pair_indices,
    read_probability_map,
    utc_now,
    write_json,
)


def load_primary_manifest(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "data" / "event_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[(df["include_in_primary_search"] == True) & (df["sky_map_available"] == True)].copy()
    if df.empty:
        raise RuntimeError("No primary events with sky maps")
    return df.sort_values("gps_time").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--nside", type=int, default=512)
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    start = time.perf_counter()
    events = load_primary_manifest(run_dir)
    n = len(events)
    maps = []
    map_rows = []
    for idx, row in events.iterrows():
        prob, meta = read_probability_map(row, target_nside=args.nside)
        point = map_point_estimates(prob, args.nside)
        maps.append(prob.astype(np.float32, copy=False))
        map_rows.append({
            "idx": int(idx),
            "event_name": row["event_name"],
            "sky_map_path": row["sky_map_path"],
            "source_format": meta["source_format"],
            "source_group": meta.get("source_group", ""),
            "source_nside": int(meta["source_nside"]),
            "source_ordering": meta["source_ordering"],
            "common_nside": int(args.nside),
            "map_ra": point["map_ra"],
            "map_dec": point["map_dec"],
            "map_max_prob": point["map_max_prob"],
            "map_area90_deg2": point["map_area90_deg2"],
        })
        print(f"loaded {idx + 1}/{n} {row['event_name']} nside={meta['source_nside']} group={meta.get('source_group','')}", flush=True)

    map_table = pd.DataFrame(map_rows)
    map_table.to_csv(run_dir / "features" / "real_sky_map_audit.csv", index=False)

    stack = np.vstack(maps).astype(np.float32, copy=False)
    raw = stack @ stack.T
    norms = np.sqrt(np.clip(np.diag(raw), 0.0, None))
    denom = np.maximum(norms[:, None] * norms[None, :], 1e-30)
    cosine = raw / denom

    rows, cols = pair_indices(n)
    ra = map_table["map_ra"].to_numpy(dtype=np.float64)
    dec = map_table["map_dec"].to_numpy(dtype=np.float64)
    ang = angular_sep_rad(ra[rows], dec[rows], ra[cols], dec[cols])
    out = pd.DataFrame({
        "idx_i": rows.astype(np.int32),
        "idx_j": cols.astype(np.int32),
        "event_i": events["event_name"].to_numpy()[rows],
        "event_j": events["event_name"].to_numpy()[cols],
        "raw_posterior_overlap": raw[rows, cols].astype(np.float64),
        "cosine_overlap": cosine[rows, cols].astype(np.float64),
        "angular_sep_map_deg": np.degrees(ang).astype(np.float64),
        "common_nside": int(args.nside),
    })
    out["sky_log_cosine_overlap"] = np.log(np.clip(out["cosine_overlap"].to_numpy(dtype=np.float64), 1e-300, None))
    out_path = run_dir / "features" / "real_sky_overlap.parquet"
    out.to_parquet(out_path, index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    vals = out["sky_log_cosine_overlap"].to_numpy(dtype=np.float64)
    ax.hist(vals[np.isfinite(vals)], bins=50, color="#667085", alpha=0.85)
    ax.set_xlabel("log cosine sky-posterior overlap", fontweight="bold")
    ax.set_ylabel("unordered event pairs", fontweight="bold")
    ax.set_title("GWTC primary null sky-overlap distribution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(run_dir / "figures" / f"sky_overlap_background_hist.{ext}", dpi=300)
    plt.close(fig)

    summary = {
        "generated_at_utc": utc_now(),
        "n_events": int(n),
        "n_pairs": int(len(out)),
        "common_nside": int(args.nside),
        "output": str(out_path),
        "wall_time_s": float(time.perf_counter() - start),
        "cosine_overlap_quantiles": {str(q): float(out["cosine_overlap"].quantile(q)) for q in [0.0, 0.5, 0.9, 0.99, 1.0]},
        "angular_sep_deg_quantiles": {str(q): float(out["angular_sep_map_deg"].quantile(q)) for q in [0.0, 0.5, 0.9, 0.99, 1.0]},
    }
    write_json(run_dir / "features" / "real_sky_overlap.summary.json", summary)
    (run_dir / "logs" / "healpix_overlap.log").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

