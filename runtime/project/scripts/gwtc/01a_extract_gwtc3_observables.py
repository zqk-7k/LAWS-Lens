from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import requests
from astropy.time import Time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.gwtc import config


GWTC3_ERA_CATALOGS = {
    "GWTC-1-confident",
    "GWTC-1-marginal",
    "GWTC-2.1-confident",
    "GWTC-2.1-marginal",
    "GWTC-3-confident",
    "GWTC-3-marginal",
}
EXCLUDE_EVENTS = {"GW170817", "GW190425", "GW190814", "GW200105_162426", "GW200115_042309"}
CATALOG_PRIORITY = {
    "GWTC-1-marginal": 1,
    "GWTC-1-confident": 2,
    "GWTC-2.1-marginal": 3,
    "GWTC-2.1-confident": 4,
    "GWTC-3-marginal": 5,
    "GWTC-3-confident": 6,
}
DEG2_PER_SR = (180.0 / math.pi) ** 2
CHI2_2_90 = 4.605170185988092


def load_status() -> dict[str, Any]:
    return json.loads(config.SOURCE_STATUS_JSON.read_text(encoding="utf-8"))


def get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def zenodo_files(record_url: str) -> dict[str, dict[str, Any]]:
    record = get_json(record_url)
    out = {}
    for item in record.get("files", []):
        key = item.get("key") or item.get("filename")
        if not key:
            continue
        out[key] = {
            "key": key,
            "size": int(item.get("size") or 0),
            "url": item.get("links", {}).get("self") or item.get("links", {}).get("download"),
        }
    return out


def download_file(url: str, path: Path, expected_size: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return
    tmp = path.with_suffix(path.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-C",
            "-",
            "-o",
            str(tmp),
            url,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise RuntimeError(f"incomplete download for {path}: {tmp.stat().st_size} != {expected_size}")
            tmp.replace(path)
            return
        except Exception:
            if tmp.exists():
                tmp.unlink()
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)


def download_many(tasks: list[tuple[str, str, Path, int | None]], workers: int) -> None:
    if workers <= 1:
        for label, url, path, expected_size in tasks:
            print(f"DOWNLOAD {label}", flush=True)
            download_file(url, path, expected_size)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, url, path, expected_size): label
            for label, url, path, expected_size in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            future.result()
            print(f"DOWNLOADED {label}", flush=True)


def read_gwosc_allevents() -> pd.DataFrame:
    config.GWOSC_ALLEVENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not config.GWOSC_ALLEVENTS_CACHE.exists():
        last_error = None
        for attempt in range(1, 4):
            try:
                response = requests.get(config.GWOSC_ALLEVENTS_CSV, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()
                config.GWOSC_ALLEVENTS_CACHE.write_text(response.text, encoding="utf-8")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2.0 * attempt)
        if not config.GWOSC_ALLEVENTS_CACHE.exists():
            raise RuntimeError(f"Could not download GWOSC allevents CSV: {last_error}")
    return pd.read_csv(config.GWOSC_ALLEVENTS_CACHE)


def event_key(name: str) -> str:
    return str(name).strip()


def select_gwtc3_bbh_events() -> pd.DataFrame:
    df = read_gwosc_allevents()
    df = df[df["catalog.shortName"].isin(GWTC3_ERA_CATALOGS)].copy()
    df["catalog_priority"] = df["catalog.shortName"].map(CATALOG_PRIORITY).fillna(0)
    df = df.sort_values(["commonName", "catalog_priority", "version"]).drop_duplicates("commonName", keep="last")

    p_astro = pd.to_numeric(df["p_astro"], errors="coerce")
    confident_default = pd.Series(
        np.select(
            [
                df["catalog.shortName"].str.contains("confident", case=False),
                df["catalog.shortName"].str.contains("marginal", case=False),
            ],
            [1.0, 0.1],
            default=0.0,
        ),
        index=df.index,
    )
    p_astro = p_astro.fillna(confident_default)
    snr = pd.to_numeric(df["network_matched_filter_snr"], errors="coerce")
    is_not_known_bns_nsbh = ~df["commonName"].isin(EXCLUDE_EVENTS)
    selected = df[(p_astro > 0.0) & (snr > 9.0) & is_not_known_bns_nsbh].copy()
    selected["p_astro_effective"] = p_astro.loc[selected.index].astype(float)
    selected["network_snr_effective"] = snr.loc[selected.index].astype(float)
    return selected.sort_values("GPS").reset_index(drop=True)


def gps_from_event_token(token: str) -> float | None:
    match = re.search(r"GW(\d{6})_(\d{6})", token)
    if not match:
        return None
    date_token, time_token = match.groups()
    dt = datetime.strptime(date_token + time_token, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return float(Time(dt).gps)


def match_cosmo_h5(
    event: str,
    event_gps: float,
    file_indexes: list[dict[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    prefix = event
    candidates = []
    for index in file_indexes:
        for key, meta in index.items():
            if key.endswith("_PEDataRelease_mixed_cosmo.h5") and re.search(rf"{re.escape(prefix)}(?:_|-)", key):
                candidates.append((key, meta))
    if not candidates and re.fullmatch(r"GW\d{6}", prefix):
        for index in file_indexes:
            for key, meta in index.items():
                if key.endswith("_PEDataRelease_mixed_cosmo.h5") and prefix in key:
                    candidates.append((key, meta))
    if not candidates:
        return None, None
    if re.fullmatch(r"GW\d{6}", prefix) and len(candidates) > 1 and np.isfinite(event_gps):
        candidates.sort(key=lambda item: abs((gps_from_event_token(item[0]) or float("inf")) - float(event_gps)))
    else:
        candidates.sort(key=lambda item: item[1].get("size", 0))
    return candidates[0]


def skymap_stats_files(file_indexes: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out = {}
    for index in file_indexes:
        for key, meta in index.items():
            if "skymap_stats" in key.lower():
                out[key] = meta
    return out


def parse_area90_from_stats(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(text), sep=r"\s+|\t", engine="python", comment="#")
        for col in df.columns:
            if col.lower() in {"area(90)", "area90", "area_90"}:
                val = float(df[col].iloc[0])
                return val if np.isfinite(val) and val > 0 else None
    except Exception:
        return None
    return None


def circular_mean(angles: np.ndarray) -> float:
    return float(np.mod(np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles))), 2.0 * np.pi))


def posterior_a90_gaussian(ra: np.ndarray, dec: np.ndarray) -> float:
    ra0 = circular_mean(ra)
    dec0 = float(np.nanmedian(dec))
    dra = np.angle(np.exp(1j * (ra - ra0))) * max(math.cos(dec0), 1e-3)
    ddec = dec - dec0
    xy = np.column_stack([dra, ddec])
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < 10:
        return float("nan")
    cov = np.cov(xy.T)
    det = max(float(np.linalg.det(cov)), 0.0)
    return float(math.pi * CHI2_2_90 * math.sqrt(det) * DEG2_PER_SR)


def scalar_dataset(group: h5py.Group, suffix: str) -> float | None:
    for path, obj in group.file.items():
        pass
    return None


def read_h5_observables(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as h5:
        sample_path = "C01:Mixed/posterior_samples"
        if sample_path not in h5:
            sample_path = next((name for name in h5 if f"{name}/posterior_samples" in h5), None)
            if sample_path is None:
                raise KeyError(f"No posterior_samples in {path}")
            sample_path = f"{sample_path}/posterior_samples"
        samples = h5[sample_path][:]
        names = samples.dtype.names or ()
        def med(name: str) -> float:
            if name not in names:
                return float("nan")
            values = np.asarray(samples[name], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if len(values) else float("nan")

        ra = np.asarray(samples["ra"], dtype=np.float64) if "ra" in names else np.asarray([], dtype=np.float64)
        dec = np.asarray(samples["dec"], dtype=np.float64) if "dec" in names else np.asarray([], dtype=np.float64)
        area90 = float("nan")
        for group_name in ["C01:Mixed", sample_path.rsplit("/", 1)[0]]:
            for rel in ["meta_data/other/area90", "meta_data/other/area(90)"]:
                full = f"{group_name}/{rel}"
                if full in h5:
                    raw = h5[full][()]
                    try:
                        area90 = float(np.asarray(raw).ravel()[0])
                    except Exception:
                        try:
                            area90 = float(np.asarray(raw).ravel()[0].decode())
                        except Exception:
                            area90 = float("nan")
                    if np.isfinite(area90) and area90 > 0:
                        break
            if np.isfinite(area90) and area90 > 0:
                break
        if not (np.isfinite(area90) and area90 > 0) and len(ra) and len(dec):
            area90 = posterior_a90_gaussian(ra, dec)

        return {
            "ra_median": circular_mean(ra) if len(ra) else float("nan"),
            "dec_median": med("dec"),
            "sky_area_90_deg2": area90,
            "chirp_mass_median": med("chirp_mass_source") if "chirp_mass_source" in names else med("chirp_mass"),
            "mass_ratio_median": med("mass_ratio"),
            "luminosity_distance_median": med("luminosity_distance"),
            "network_snr_h5": med("network_matched_filter_snr"),
        }


def maybe_download_skymap_tar(gwtc3_files: dict[str, dict[str, Any]], raw_dir: Path, dry_run: bool) -> Path | None:
    matches = [meta for key, meta in gwtc3_files.items() if key.endswith("PESkyLocalizations.tar.gz")]
    if not matches:
        return None
    meta = matches[0]
    path = raw_dir / meta["key"]
    if not dry_run:
        download_file(meta["url"], path, meta["size"])
        extract_dir = raw_dir / "PESkyLocalizations"
        marker = extract_dir / ".extracted"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(extract_dir)
            marker.write_text(str(time.time()), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="debug limit after event selection")
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()

    status = load_status()
    if not status.get("capabilities", {}).get("FULL_GWTC3"):
        raise RuntimeError("source_status.json does not advertise FULL_GWTC3")

    raw_dir = config.GWTC3_RAW_DIR
    h5_dir = raw_dir / "pe_h5_cosmo"
    stats_dir = raw_dir / "skymap_stats"
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    gwtc3_files = zenodo_files(config.ZENODO_GWTC3_RECORD)
    gwtc21_files = zenodo_files(config.ZENODO_GWTC21_RECORD)
    file_indexes = [gwtc3_files, gwtc21_files]
    stats_index = skymap_stats_files(file_indexes)

    selected = select_gwtc3_bbh_events()
    if args.limit:
        selected = selected.head(args.limit).copy()

    plan = []
    missing = []
    total_bytes = 0
    for _, row in selected.iterrows():
        event = event_key(row["commonName"])
        key, meta = match_cosmo_h5(event, float(row["GPS"]), file_indexes)
        if key is None:
            missing.append(event)
            continue
        total_bytes += int(meta["size"])
        plan.append((event, key, meta, row))

    sky_tar = maybe_download_skymap_tar(gwtc3_files, raw_dir, args.dry_run)
    print(f"Selected {len(selected)} GWTC-3-era BBH rows; matched PE H5 for {len(plan)}; missing={missing[:20]}")
    print(f"Planned cosmo H5 download size: {total_bytes / 1e9:.2f} GB")
    if sky_tar is not None:
        print(f"GWTC-3 skymap tar: {sky_tar}")
    if args.dry_run:
        print("Dry run only; no PE H5 downloaded.")
        return

    stat_paths: dict[str, Path] = {}
    download_tasks = []
    for event, key, meta, _event_row in plan:
        h5_path = h5_dir / key
        stat_key = next((k for k in stats_index if event in k), None)
        download_tasks.append((f"{event} PE", meta["url"], h5_path, meta["size"]))
        if stat_key:
            stat_meta = stats_index[stat_key]
            stat_path = stats_dir / stat_key
            stat_paths[event] = stat_path
            download_tasks.append((f"{event} skymap_stats", stat_meta["url"], stat_path, stat_meta["size"]))

    print(f"Starting downloads with workers={args.download_workers}", flush=True)
    download_many(download_tasks, workers=max(1, int(args.download_workers)))

    rows = []
    for idx, (event, key, meta, event_row) in enumerate(plan, 1):
        print(f"[{idx}/{len(plan)}] EXTRACT {event} {key}", flush=True)
        h5_path = h5_dir / key
        stats_area = None
        stat_path = stat_paths.get(event)
        if stat_path is not None:
            stats_area = parse_area90_from_stats(stat_path)
        obs = read_h5_observables(h5_path)
        if stats_area is not None and np.isfinite(stats_area) and stats_area > 0:
            obs["sky_area_90_deg2"] = float(stats_area)
            obs["sky_area_source"] = "skymap_stats"
        else:
            obs["sky_area_source"] = "posterior_ra_dec_gaussian"

        network_snr = float(event_row["network_snr_effective"])
        if not np.isfinite(network_snr) and np.isfinite(obs.get("network_snr_h5", np.nan)):
            network_snr = float(obs["network_snr_h5"])
        rows.append({
            "event_name": event,
            "catalog": event_row["catalog.shortName"],
            "gps_trigger_time": float(event_row["GPS"]),
            "ra_median": obs["ra_median"],
            "dec_median": obs["dec_median"],
            "sky_area_90_deg2": obs["sky_area_90_deg2"],
            "network_snr": network_snr,
            "chirp_mass_median": obs["chirp_mass_median"],
            "mass_ratio_median": obs["mass_ratio_median"],
            "luminosity_distance_median": obs["luminosity_distance_median"],
            "p_astro": float(event_row["p_astro_effective"]),
            "mass_1_source": float(event_row["mass_1_source"]),
            "mass_2_source": float(event_row["mass_2_source"]),
            "pe_h5_file": str(h5_path.relative_to(config.REPO_ROOT)),
            "sky_area_source": obs["sky_area_source"],
            "skymap_stats_file": str(stat_path.relative_to(config.REPO_ROOT)) if stat_path else "",
        })

    out = pd.DataFrame(rows).sort_values("gps_trigger_time")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.GWTC3_OBSERVABLES_CSV, index=False)
    print(f"Wrote {config.GWTC3_OBSERVABLES_CSV} rows={len(out)}")
    print(f"SNR range: {out['network_snr'].min():.2f} - {out['network_snr'].max():.2f}")
    print(f"A90 median/p90 deg2: {out['sky_area_90_deg2'].median():.2f} / {out['sky_area_90_deg2'].quantile(0.9):.2f}")
    present = set(out["event_name"])
    print(f"GW170104 present: {'GW170104' in present}; GW170814 present: {'GW170814' in present}")


if __name__ == "__main__":
    main()
