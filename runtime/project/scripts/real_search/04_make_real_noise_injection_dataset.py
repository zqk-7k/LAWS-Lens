from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gwosc.locate import get_event_urls

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import REPO_ROOT, default_run_dir, ensure_run_dirs, utc_now, write_json


FILENAME_RE = re.compile(r"-(\d+)-(\d+)\.hdf5(?:\.gz)?$")


def parse_gwosc_url(url: str) -> tuple[int | None, int | None]:
    match = FILENAME_RE.search(url)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def choose_url(event_name: str, detector: str, gps: float, window: float) -> tuple[str, int | None, int | None, bool] | tuple[None, None, None, bool]:
    try:
        urls = get_event_urls(event_name, detector=detector)
    except Exception:
        urls = []
    candidates = []
    for url in urls:
        if not url.endswith(".hdf5"):
            continue
        start, duration = parse_gwosc_url(url)
        covers = bool(start is not None and duration is not None and start <= gps - window and start + duration >= gps + window)
        candidates.append((url, start, duration, covers))
    if not candidates:
        return None, None, None, False
    covering = [item for item in candidates if item[3]]
    pool = covering or candidates
    # Prefer 4096-second public strain where possible; otherwise choose the longest available file.
    pool.sort(key=lambda x: ((x[2] or 0), "4096" in x[0], "4KHZ" in x[0] or "LOSC_4" in x[0]), reverse=True)
    return pool[0]


def curl_download(url: str, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "already_exists"
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable GWOSC downloads")
    cmd = [
        curl, "-L", "--fail", "--silent", "--show-error", "--retry", "5", "--retry-delay", "5", "-C", "-",
        "-o", str(tmp), url,
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(out_path)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--detectors", default="H1,L1")
    parser.add_argument("--window-s", type=float, default=128.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[events["include_in_primary_search"] == True].sort_values("gps_time").reset_index(drop=True)
    if args.max_events is not None:
        primary = primary.head(args.max_events).copy()

    rows = []
    for _, row in primary.iterrows():
        event = str(row["event_name"])
        gps = float(row["gps_time"])
        for det in detectors:
            url, start, duration, covers = choose_url(event, det, gps, args.window_s)
            if url is None:
                rows.append({
                    "event_name": event, "detector": det, "gps_time": gps,
                    "url": "", "gwosc_start": np.nan, "gwosc_duration": np.nan,
                    "covers_requested_window": False, "local_path": "",
                    "download_status": "no_gwosc_url",
                })
                print(f"{event} {det}: no URL", flush=True)
                continue
            filename = Path(url).name
            out_path = run_dir / "data" / "real_strain" / event / filename
            status = "located_not_downloaded"
            if args.download:
                try:
                    status = curl_download(url, out_path)
                except Exception as exc:
                    status = f"download_failed:{type(exc).__name__}:{str(exc)[:160]}"
            rows.append({
                "event_name": event,
                "detector": det,
                "gps_time": gps,
                "url": url,
                "gwosc_start": start if start is not None else np.nan,
                "gwosc_duration": duration if duration is not None else np.nan,
                "covers_requested_window": covers,
                "local_path": str(out_path.relative_to(run_dir)) if out_path.exists() else "",
                "download_status": status,
            })
            print(f"{event} {det}: {status} covers={covers} file={filename}", flush=True)

    strain_url_manifest = pd.DataFrame(rows)
    out_path = run_dir / "data" / "strain_gwosc_download_manifest.csv"
    strain_url_manifest.to_csv(out_path, index=False)

    noise_rows = []
    for _, row in strain_url_manifest.iterrows():
        if not row.get("local_path"):
            continue
        gps = float(row["gps_time"])
        start = float(row["gwosc_start"])
        duration = float(row["gwosc_duration"])
        # Two off-source intervals per detector, avoiding the central +/-128 s on-source window.
        left_end = gps - args.window_s
        right_start = gps + args.window_s
        if start < left_end:
            noise_rows.append({
                "event_name": row["event_name"], "detector": row["detector"], "strain_path": row["local_path"],
                "segment_kind": "real_noise_offsource_left", "segment_start": start, "segment_end": left_end,
                "usable_duration_s": max(0.0, left_end - start),
            })
        file_end = start + duration
        if right_start < file_end:
            noise_rows.append({
                "event_name": row["event_name"], "detector": row["detector"], "strain_path": row["local_path"],
                "segment_kind": "real_noise_offsource_right", "segment_start": right_start, "segment_end": file_end,
                "usable_duration_s": max(0.0, file_end - right_start),
            })
    noise_index = pd.DataFrame(noise_rows)
    inj_dir = run_dir / "data" / "real_noise_injections"
    for split in ["train", "val", "test"]:
        (inj_dir / split).mkdir(parents=True, exist_ok=True)
    if not noise_index.empty:
        noise_index.to_parquet(inj_dir / "metadata.parquet", index=False)
    else:
        pd.DataFrame(columns=["event_name", "detector", "strain_path", "segment_kind", "segment_start", "segment_end", "usable_duration_s"]).to_parquet(inj_dir / "metadata.parquet", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "download_requested": bool(args.download),
        "detectors": detectors,
        "events_processed": int(len(primary)),
        "rows": int(len(strain_url_manifest)),
        "downloaded_or_existing": int(strain_url_manifest["download_status"].isin(["downloaded", "already_exists"]).sum()) if not strain_url_manifest.empty else 0,
        "covers_requested_window": int(strain_url_manifest["covers_requested_window"].sum()) if not strain_url_manifest.empty else 0,
        "noise_segments": int(len(noise_index)),
        "note": "This phase currently builds the real-strain and off-source-noise index. Synthetic lensed waveform injection materialization is the next sub-step before domain-matched encoder training.",
    }
    write_json(run_dir / "data" / "real_noise_injections" / "phase_e_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
