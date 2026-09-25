from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy_healpix import level_to_nside, uniq_to_level_ipix

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.gwtc import config


SECONDS_PER_YEAR = 365.25 * 86400.0
FAR_1_PER_YEAR_HZ = 1.0 / SECONDS_PER_YEAR


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
        cmd = [curl, "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", str(tmp), url]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if expected_size is not None and tmp.stat().st_size != expected_size:
            raise RuntimeError(f"incomplete download for {path}: {tmp.stat().st_size} != {expected_size}")
        tmp.replace(path)
        return
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)


def decode_value(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if hasattr(value, "item"):
        return value.item()
    return value


def load_search_summary(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as h5:
        arr = h5["search_summary"][:]
        rows = []
        for row in arr:
            rows.append({name: decode_value(row[name]) for name in arr.dtype.names})
    df = pd.DataFrame(rows)
    df["event_name"] = df["gw_name"].astype(str)
    return df


def load_pipeline_details(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as h5:
        for dataset_name in ["gstlal", "pycbc", "MBTA", "CWB"]:
            if dataset_name not in h5:
                continue
            arr = h5[dataset_name][:]
            for row in arr:
                item = {name: decode_value(row[name]) for name in arr.dtype.names}
                item["pipeline_detail"] = dataset_name
                rows.append(item)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["event_name"] = df["gw_name"].astype(str)
    return df


def choose_pipeline_detail(details: pd.DataFrame, event_name: str, pipeline: str) -> dict[str, Any]:
    if details.empty:
        return {}
    subset = details[details["event_name"] == event_name]
    if subset.empty:
        return {}
    exact = subset[subset["pipeline_detail"].str.lower() == str(pipeline).lower()]
    row = exact.iloc[0] if not exact.empty else subset.iloc[0]
    return row.to_dict()


def extract_selected_skymaps(tar_path: Path, selected_paths: set[str], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    remaining = set(selected_paths)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = member.name
            normalized = name.lstrip("./")
            wanted = normalized in remaining or any(normalized.endswith(path) for path in remaining)
            if not wanted:
                continue
            target_name = next((path for path in remaining if normalized == path or normalized.endswith(path)), normalized)
            out_path = out_dir / Path(target_name).name
            if not out_path.exists():
                src = tar.extractfile(member)
                if src is None:
                    continue
                with out_path.open("wb") as handle:
                    shutil.copyfileobj(src, handle)
            extracted[target_name] = out_path
            remaining.discard(target_name)
            if not remaining:
                break
    if remaining:
        print(f"WARNING missing {len(remaining)} skymaps in tar; sample={list(sorted(remaining))[:10]}", flush=True)
    return extracted


def skymap_summary(path: Path) -> dict[str, float]:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[1]
        header = hdu.header
        data = hdu.data
        columns = set(hdu.columns.names)
        ordering = str(header.get("ORDERING", "")).upper()
        index_scheme = str(header.get("INDXSCHM", "")).upper()

        if "UNIQ" in columns and "PROBDENSITY" in columns:
            uniq = np.asarray(data["UNIQ"], dtype=np.int64)
            probdensity = np.asarray(data["PROBDENSITY"], dtype=np.float64)
            level, ipix = uniq_to_level_ipix(uniq)
            nside = level_to_nside(level)
            area_sr = hp.nside2pixarea(nside)
            prob = np.where(np.isfinite(probdensity) & (probdensity > 0), probdensity * area_sr, 0.0)
            total = prob.sum()
            if total <= 0:
                return {"ra_median": float("nan"), "dec_median": float("nan"), "sky_area_90_deg2": float("nan")}
            prob = prob / total
            order = np.argsort(probdensity)[::-1]
            cdf = np.cumsum(prob[order])
            stop = int(np.searchsorted(cdf, 0.9, side="left")) + 1
            keep = order[:stop]
            max_row = int(order[0])
            theta, phi = hp.pix2ang(int(nside[max_row]), int(ipix[max_row]), nest=True)
            area90 = float(np.sum(area_sr[keep]) * (180.0 / math.pi) ** 2)
            return {"ra_median": float(phi), "dec_median": float(0.5 * math.pi - theta), "sky_area_90_deg2": area90}

        if "PROB" not in columns:
            return {"ra_median": float("nan"), "dec_median": float("nan"), "sky_area_90_deg2": float("nan")}
        raw = np.asarray(data["PROB"], dtype=np.float64)
        prob = raw.reshape(-1)
        prob = np.where(np.isfinite(prob) & (prob > 0), prob, 0.0)
        total = prob.sum()
        if total <= 0:
            return {"ra_median": float("nan"), "dec_median": float("nan"), "sky_area_90_deg2": float("nan")}
        prob = prob / total
        nside = int(header.get("NSIDE") or hp.npix2nside(len(prob)))
        nest = ordering == "NESTED"
        order = np.argsort(prob)[::-1]
        cdf = np.cumsum(prob[order])
        stop = int(np.searchsorted(cdf, 0.9, side="left")) + 1
        keep = order[:stop]
        max_pix = int(order[0])
        theta, phi = hp.pix2ang(nside, max_pix, nest=nest)
        area90 = float(len(keep) * hp.nside2pixarea(nside, degrees=True))
    return {
        "ra_median": float(phi),
        "dec_median": float(0.5 * math.pi - theta),
        "sky_area_90_deg2": area90,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--p-bbh-min", type=float, default=0.5)
    parser.add_argument("--far-per-year-max", type=float, default=1.0)
    args = parser.parse_args()

    raw_dir = config.GWTC5_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = zenodo_files(config.ZENODO_GWTC5_CANDIDATE_RECORD)
    summary_meta = next(meta for key, meta in files.items() if "SearchSummaryTable" in key)
    tar_meta = next(meta for key, meta in files.items() if key.endswith("Archived_SearchResults.tar.gz"))
    summary_path = raw_dir / summary_meta["key"]
    tar_path = raw_dir / tar_meta["key"]

    download_file(summary_meta["url"], summary_path, summary_meta["size"])
    summary = load_search_summary(summary_path)
    details = load_pipeline_details(summary_path)

    far_threshold_hz = float(args.far_per_year_max) / SECONDS_PER_YEAR
    selected = summary[(summary["p_BBH"] > args.p_bbh_min) & (summary["far"] < far_threshold_hz)].copy()
    selected = selected.sort_values("gps_time").reset_index(drop=True)
    print(
        f"GWTC-5 search rows={len(summary)} selected_BBH={len(selected)} "
        f"pairs={len(selected) * (len(selected) - 1) // 2} far_threshold_hz={far_threshold_hz:.3e}",
        flush=True,
    )
    if args.dry_run:
        print(f"Would download skymap tar {tar_path} size={tar_meta['size'] / 1e9:.2f} GB")
        print(selected[["event_name", "snr", "far", "p_BBH", "skymap_file"]].head(10).to_string(index=False))
        return

    download_file(tar_meta["url"], tar_path, tar_meta["size"])
    skymap_paths = set(selected["skymap_file"].astype(str))
    extracted = extract_selected_skymaps(tar_path, skymap_paths, raw_dir / "selected_skymaps")

    rows = []
    for _, row in selected.iterrows():
        detail = choose_pipeline_detail(details, row["event_name"], row["pipeline"])
        mass1 = float(detail.get("mass1", np.nan))
        mass2 = float(detail.get("mass2", np.nan))
        mass_ratio = float(min(mass1, mass2) / max(mass1, mass2)) if np.isfinite(mass1) and np.isfinite(mass2) and max(mass1, mass2) > 0 else float("nan")
        sky_path = extracted.get(str(row["skymap_file"]))
        sky = skymap_summary(sky_path) if sky_path else {"ra_median": float("nan"), "dec_median": float("nan"), "sky_area_90_deg2": float("nan")}
        rows.append({
            "event_name": row["event_name"],
            "catalog": "GWTC-5.0-search",
            "gps_trigger_time": float(row["gps_time"]),
            "ra_median": sky["ra_median"],
            "dec_median": sky["dec_median"],
            "sky_area_90_deg2": sky["sky_area_90_deg2"],
            "network_snr": float(row["snr"]),
            "chirp_mass_median": float(row["mchirp"]),
            "mass_ratio_median": mass_ratio,
            "luminosity_distance_median": float("nan"),
            "p_astro": float(row["p_BBH"]),
            "far_hz": float(row["far"]),
            "pipeline": row["pipeline"],
            "source_path": "SearchSummaryTable + Archived_SearchResults skymap",
            "skymap_file": str(sky_path.relative_to(config.REPO_ROOT)) if sky_path else "",
        })

    out = pd.DataFrame(rows).sort_values("gps_trigger_time")
    out.to_csv(config.GWTC5_OBSERVABLES_CSV, index=False)
    pair_count = len(out) * (len(out) - 1) // 2
    print(f"Wrote {config.GWTC5_OBSERVABLES_CSV} rows={len(out)} pairs={pair_count}")
    print(f"SNR range: {out['network_snr'].min():.2f} - {out['network_snr'].max():.2f}")
    print(f"A90 median/p90 deg2: {out['sky_area_90_deg2'].median():.2f} / {out['sky_area_90_deg2'].quantile(0.9):.2f}")
    for event in ["GW240615", "GW250114"]:
        hits = out[out["event_name"].str.contains(event, regex=False)]
        print(f"{event} present rows={len(hits)}")
        if not hits.empty:
            print(hits[["event_name", "network_snr", "sky_area_90_deg2"]].to_string(index=False))


if __name__ == "__main__":
    main()
