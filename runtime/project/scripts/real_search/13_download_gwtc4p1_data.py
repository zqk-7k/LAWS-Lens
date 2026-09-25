from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = "GWTC-4.1"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "gwtc4p1_data_completion_20260628"
DATA_ROOT = REPO_ROOT / "data" / "gwtc4p1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs(run_dir: Path) -> None:
    for rel in ["data", "logs"]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["api_json/events", "strain_4khz_hdf5", "pe_hdf5"]:
        (DATA_ROOT / rel).mkdir(parents=True, exist_ok=True)


def fetch_json(url: str, timeout: int = 120) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "gw-catalog-gwtc4p1-completion/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def fetch_event_details(summary_events: dict[str, Any], max_workers: int) -> dict[str, Any]:
    def one(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        event_id, event = item
        url = event.get("jsonurl") or ""
        if not url:
            return event_id, event
        try:
            detail = fetch_json(str(url))
            detail_events = detail.get("events", {})
            if detail_events:
                detail_id, detail_event = next(iter(detail_events.items()))
                return detail_id, detail_event
        except Exception as exc:
            event = dict(event)
            event["detail_fetch_error"] = f"{type(exc).__name__}:{exc}"
        return event_id, event

    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(one, sorted(summary_events.items())))


def head_size(url: str, timeout: int = 60) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gw-catalog-gwtc4p1-completion/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        return None


def filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "download.bin"
    if parts[-1] == "content" and len(parts) >= 2:
        return urllib.parse.unquote(parts[-2])
    return urllib.parse.unquote(parts[-1])


def choose_preferred_pe(event: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for name, item in (event.get("parameters") or {}).items():
        if item.get("pipeline_type") != "pe":
            continue
        data_url = item.get("data_url") or ""
        if not data_url:
            continue
        row = {
            "pe_key": name,
            "pe_url": data_url,
            "pe_waveform_family": item.get("waveform_family", ""),
            "pe_is_preferred": bool(item.get("is_preferred", False)),
            "pe_doi": ";".join((item.get("links") or {}).values()),
        }
        if row["pe_is_preferred"]:
            return row
        if not best:
            best = row
    return best


def selected_strain_rows(event: dict[str, Any], sampling_rate: int, fmt: str, detectors: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in event.get("strain") or []:
        det = str(item.get("detector", ""))
        if det not in detectors:
            continue
        if int(item.get("sampling_rate", -1)) != int(sampling_rate):
            continue
        if str(item.get("format", "")).lower() != fmt.lower():
            continue
        url = item.get("url") or ""
        if not url:
            continue
        out.append({
            "detector": det,
            "sampling_rate": int(item.get("sampling_rate")),
            "duration": int(item.get("duration")),
            "format": item.get("format"),
            "gps_start": float(item.get("GPSstart")),
            "url": url,
        })
    return sorted(out, key=lambda r: r["detector"])


def event_manifest_rows(catalog: str, events: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    pe_rows: list[dict[str, Any]] = []
    api_rows: list[dict[str, Any]] = []
    for event_id, event in sorted(events.items()):
        common = str(event.get("commonName") or event_id.split("-")[0])
        event_json_path = DATA_ROOT / "api_json" / "events" / f"{common}_v{event.get('version', '')}.json"
        event_json_path.parent.mkdir(parents=True, exist_ok=True)
        event_json_path.write_text(json.dumps({"events": {event_id: event}}, indent=2, ensure_ascii=False), encoding="utf-8")
        pe = choose_preferred_pe(event)
        event_rows.append({
            "event_id": event_id,
            "event_name": common,
            "catalog": catalog,
            "version": event.get("version"),
            "gracedb_id": event.get("gracedb_id", ""),
            "gps_time": event.get("GPS"),
            "network_snr": event.get("network_matched_filter_snr"),
            "far_per_year": event.get("far"),
            "p_astro": event.get("p_astro"),
            "mass_1_source": event.get("mass_1_source"),
            "mass_2_source": event.get("mass_2_source"),
            "chirp_mass_source": event.get("chirp_mass_source"),
            "luminosity_distance": event.get("luminosity_distance"),
            "redshift": event.get("redshift"),
            "jsonurl": event.get("jsonurl", ""),
            "reference": event.get("reference", ""),
            "local_event_json": str(event_json_path.relative_to(REPO_ROOT)),
            "preferred_pe_url": pe.get("pe_url", ""),
            "preferred_pe_key": pe.get("pe_key", ""),
            "preferred_pe_waveform_family": pe.get("pe_waveform_family", ""),
            "preferred_pe_doi": pe.get("pe_doi", ""),
            "n_strain_entries": len(event.get("strain") or []),
        })
        if pe:
            filename = filename_from_url(pe["pe_url"])
            local = DATA_ROOT / "pe_hdf5" / common / filename
            pe_rows.append({
                "event_name": common,
                **pe,
                "local_path": str(local.relative_to(REPO_ROOT)),
                "expected_bytes": "",
                "download_status": "pending",
            })
        api_rows.append({
            "event_name": common,
            "event_id": event_id,
            "jsonurl": event.get("jsonurl", ""),
            "local_event_json": str(event_json_path.relative_to(REPO_ROOT)),
        })
    return event_rows, pe_rows, api_rows


def build_strain_manifest(events: dict[str, Any], sampling_rate: int, fmt: str, detectors: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_id, event in sorted(events.items()):
        common = str(event.get("commonName") or event_id.split("-")[0])
        for item in selected_strain_rows(event, sampling_rate=sampling_rate, fmt=fmt, detectors=detectors):
            filename = filename_from_url(item["url"])
            local = DATA_ROOT / "strain_4khz_hdf5" / common / filename
            rows.append({
                "event_id": event_id,
                "event_name": common,
                "gps_time": event.get("GPS"),
                **item,
                "local_path": str(local.relative_to(REPO_ROOT)),
                "expected_bytes": "",
                "download_status": "pending",
            })
    return rows


def download_one(row: dict[str, Any], repo_root: Path = REPO_ROOT, timeout: int = 180) -> dict[str, Any]:
    url = str(row["url"] if "url" in row else row["pe_url"])
    local = repo_root / str(row["local_path"])
    local.parent.mkdir(parents=True, exist_ok=True)
    expected = row.get("expected_bytes")
    expected_int = int(expected) if str(expected).strip().isdigit() else None
    if local.exists() and local.stat().st_size > 0:
        if expected_int is None or local.stat().st_size == expected_int:
            row["download_status"] = "already_exists"
            row["local_bytes"] = local.stat().st_size
            return row
    tmp = local.with_suffix(local.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gw-catalog-gwtc4p1-completion/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh, length=1024 * 1024)
        os.replace(tmp, local)
        row["download_status"] = "downloaded"
        row["local_bytes"] = local.stat().st_size
    except Exception as exc:
        row["download_status"] = f"failed:{type(exc).__name__}:{exc}"
        row["local_bytes"] = local.stat().st_size if local.exists() else 0
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    return row


def fill_expected_sizes(rows: list[dict[str, Any]], url_key: str, max_workers: int) -> list[dict[str, Any]]:
    def one(row: dict[str, Any]) -> dict[str, Any]:
        size = head_size(str(row[url_key]))
        row["expected_bytes"] = size if size is not None else ""
        return row
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, rows))


def download_rows(rows: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(download_one, rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sampling-rate", type=int, default=4096)
    parser.add_argument("--format", default="hdf5")
    parser.add_argument("--detectors", default="H1,L1,V1")
    parser.add_argument("--download-strain", action="store_true")
    parser.add_argument("--download-pe", action="store_true")
    parser.add_argument("--head-sizes", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="Limit downloads for smoke tests; 0 means no limit.")
    args = parser.parse_args()

    ensure_dirs(args.run_dir)
    catalog_url = f"https://gwosc.org/eventapi/json/{args.catalog}"
    catalog = fetch_json(catalog_url)
    summary_events = catalog.get("events", {})
    events = fetch_event_details(summary_events, max_workers=args.max_workers)
    (DATA_ROOT / "api_json" / f"{args.catalog}.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    detectors = {d.strip() for d in args.detectors.split(",") if d.strip()}
    event_rows, pe_rows, api_rows = event_manifest_rows(args.catalog, events)
    strain_rows = build_strain_manifest(events, sampling_rate=args.sampling_rate, fmt=args.format, detectors=detectors)

    if args.head_sizes:
        strain_rows = fill_expected_sizes(strain_rows, "url", max_workers=args.max_workers)
        pe_rows = fill_expected_sizes(pe_rows, "pe_url", max_workers=args.max_workers)

    if args.max_files > 0:
        strain_download_rows = strain_rows[: args.max_files]
        pe_download_rows = pe_rows[: args.max_files]
    else:
        strain_download_rows = strain_rows
        pe_download_rows = pe_rows

    if args.download_strain:
        updated = {id(row): row for row in download_rows(strain_download_rows, max_workers=args.max_workers)}
        for idx, row in enumerate(strain_rows):
            for changed in updated.values():
                if row.get("local_path") == changed.get("local_path"):
                    strain_rows[idx] = changed
                    break

    if args.download_pe:
        updated = {id(row): row for row in download_rows(pe_download_rows, max_workers=max(1, min(args.max_workers, 3)))}
        for idx, row in enumerate(pe_rows):
            for changed in updated.values():
                if row.get("local_path") == changed.get("local_path"):
                    pe_rows[idx] = changed
                    break

    event_df = pd.DataFrame(event_rows)
    strain_df = pd.DataFrame(strain_rows)
    pe_df = pd.DataFrame(pe_rows)
    api_df = pd.DataFrame(api_rows)
    event_df.to_csv(args.run_dir / "data" / "event_manifest_gwtc4p1.csv", index=False)
    strain_df.to_csv(args.run_dir / "data" / "strain_manifest_gwtc4p1.csv", index=False)
    pe_df.to_csv(args.run_dir / "data" / "pe_manifest_gwtc4p1.csv", index=False)
    api_df.to_csv(args.run_dir / "data" / "api_manifest_gwtc4p1.csv", index=False)

    summary = {
        "generated_at_utc": utc_now(),
        "catalog": args.catalog,
        "catalog_url": catalog_url,
        "n_events": int(len(event_df)),
        "n_strain_files_selected": int(len(strain_df)),
        "n_pe_files_selected": int(len(pe_df)),
        "strain_sampling_rate": args.sampling_rate,
        "strain_format": args.format,
        "detectors": sorted(detectors),
        "strain_download_status": strain_df["download_status"].value_counts(dropna=False).to_dict() if len(strain_df) else {},
        "pe_download_status": pe_df["download_status"].value_counts(dropna=False).to_dict() if len(pe_df) else {},
        "strain_expected_bytes_total": int(pd.to_numeric(strain_df.get("expected_bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(strain_df) else 0,
        "pe_expected_bytes_total": int(pd.to_numeric(pe_df.get("expected_bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(pe_df) else 0,
        "data_root": str(DATA_ROOT.relative_to(REPO_ROOT)),
        "note": "4 kHz HDF5 strain is the waveform input used for the local deployment pipeline. 16 kHz/GWF duplicates are intentionally not mirrored by default.",
    }
    (args.run_dir / "data" / "gwtc4p1_download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.run_dir / "logs" / "gwtc4p1_download.log").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
