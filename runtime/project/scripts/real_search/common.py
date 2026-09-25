from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[2]
SECONDS_PER_DAY = 86400.0
DEG2_PER_SR = (180.0 / math.pi) ** 2
DEFAULT_DATE = "20260625"


def default_run_dir() -> Path:
    return REPO_ROOT / "runs" / f"real_gwtc_lensing_search_{DEFAULT_DATE}"


def ensure_run_dirs(run_dir: Path) -> None:
    for name in ["data", "features", "results", "figures", "logs"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_run_from_gps(gps: float) -> str:
    # Broad public observing-run boundaries. This is only for manifest grouping.
    if gps < 1137254417:
        return "O1"
    if gps < 1187733618:
        return "O2"
    if gps < 1253977218:
        return "O3a"
    if gps < 1269363618:
        return "O3b"
    if gps < 1390000000:
        return "O4a_or_gap"
    return "O4_search_extension"


def event_short_name(name: str) -> str:
    text = str(name).strip()
    match = re.search(r"(GW\d{6}(?:_\d{6})?)", text)
    return match.group(1) if match else text


def parse_boolish(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode(errors="ignore")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def decode_bytes(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"S", "O", "U"}:
            return ",".join(decode_bytes(v) for v in value.reshape(-1))
        return ",".join(str(v) for v in value.reshape(-1))
    if hasattr(value, "item"):
        try:
            return decode_bytes(value.item())
        except Exception:
            return str(value)
    return str(value)


def choose_h5_skymap_group(path: Path) -> str | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        preferred = ["C01:Mixed", "C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]
        for group in preferred:
            if f"{group}/skymap/data" in h5:
                return group
        for group in h5.keys():
            if f"{group}/skymap/data" in h5:
                return str(group)
    return None


def h5_detectors(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with h5py.File(path, "r") as h5:
            group = choose_h5_skymap_group(path)
            candidates = []
            if group:
                candidates.extend([
                    f"{group}/meta_data/other/command_line_args/detectors",
                    f"{group}/meta_data/meta_data/IFOs",
                    f"{group}/config_file/config/detectors",
                ])
            for key in candidates:
                if key in h5:
                    text = decode_bytes(h5[key][()])
                    if text:
                        return text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            psd_groups = []
            if group and f"{group}/psds" in h5:
                psd_groups = list(h5[f"{group}/psds"].keys())
            return ",".join(psd_groups)
    except Exception:
        return ""
    return ""


def read_hdf5_healpix(path: Path, target_nside: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    group = choose_h5_skymap_group(path)
    if group is None:
        raise FileNotFoundError(f"No /skymap/data group in {path}")
    with h5py.File(path, "r") as h5:
        prob = np.asarray(h5[f"{group}/skymap/data"][:], dtype=np.float64).reshape(-1)
        nest = False
        nest_key = f"{group}/skymap/meta_data/nest"
        if nest_key in h5:
            nest = parse_boolish(h5[nest_key][()])
    prob = sanitize_probability_map(prob)
    nside = hp.npix2nside(len(prob))
    if nest:
        prob = hp.reorder(prob, n2r=True)
    prob = resize_probability_map(prob, nside, target_nside)
    meta = {"source_format": "pe_hdf5_skymap", "source_group": group, "source_nside": nside, "source_ordering": "NESTED" if nest else "RING"}
    return prob.astype(np.float32), meta


def read_fits_healpix(path: Path, target_nside: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[1]
        header = hdu.header
        data = hdu.data
        cols = set(hdu.columns.names or [])
        ordering = str(header.get("ORDERING", "RING")).upper()
        if "PROB" not in cols:
            raise ValueError(f"{path} does not contain regular PROB HEALPix map")
        prob = np.asarray(data["PROB"], dtype=np.float64).reshape(-1)
        nside = int(header.get("NSIDE") or hp.npix2nside(len(prob)))
    prob = sanitize_probability_map(prob)
    if ordering == "NESTED":
        prob = hp.reorder(prob, n2r=True)
    prob = resize_probability_map(prob, nside, target_nside)
    meta = {"source_format": "fits_prob", "source_group": "", "source_nside": nside, "source_ordering": ordering}
    return prob.astype(np.float32), meta


def sanitize_probability_map(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.where(np.isfinite(prob) & (prob > 0), prob, 0.0)
    total = float(prob.sum())
    if total <= 0.0:
        raise ValueError("HEALPix map has non-positive probability sum")
    return prob / total


def resize_probability_map(prob: np.ndarray, source_nside: int, target_nside: int) -> np.ndarray:
    if int(source_nside) == int(target_nside):
        return sanitize_probability_map(prob)
    resized = hp.ud_grade(prob, nside_out=int(target_nside), order_in="RING", order_out="RING", power=-2)
    return sanitize_probability_map(resized)


def read_probability_map(row: pd.Series, target_nside: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(row.get("sky_map_format", ""))
    path = REPO_ROOT / str(row["sky_map_path"])
    if source.startswith("pe_hdf5"):
        return read_hdf5_healpix(path, target_nside=target_nside)
    if source.startswith("fits"):
        return read_fits_healpix(path, target_nside=target_nside)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return read_hdf5_healpix(path, target_nside=target_nside)
    return read_fits_healpix(path, target_nside=target_nside)


def map_point_estimates(prob: np.ndarray, nside: int) -> dict[str, float]:
    prob = sanitize_probability_map(prob)
    max_pix = int(np.argmax(prob))
    theta, phi = hp.pix2ang(int(nside), max_pix, nest=False)
    order = np.argsort(prob)[::-1]
    cdf = np.cumsum(prob[order])
    n90 = int(np.searchsorted(cdf, 0.9, side="left")) + 1
    area90 = float(n90 * hp.nside2pixarea(int(nside), degrees=True))
    return {
        "map_ra": float(phi),
        "map_dec": float(0.5 * math.pi - theta),
        "map_max_prob": float(prob[max_pix]),
        "map_area90_deg2": area90,
    }


def angular_sep_rad(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    sin_d1 = np.sin(dec1)
    sin_d2 = np.sin(dec2)
    cos_d1 = np.cos(dec1)
    cos_d2 = np.cos(dec2)
    dra = ra1 - ra2
    cosang = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(dra)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def row_z_matrix(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64).copy()
    np.fill_diagonal(arr, np.nan)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    out = (arr - mu) / np.maximum(sd, 1e-8)
    out[~np.isfinite(out)] = -np.inf
    return out.astype(np.float32)


def empirical_hist_lr(values: np.ndarray, signal: np.ndarray, background: np.ndarray, bins: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    background = background[np.isfinite(background)]
    sig_counts, _ = np.histogram(signal, bins=bins)
    bg_counts, _ = np.histogram(background, bins=bins)
    sig_prob = (sig_counts.astype(np.float64) + alpha) / (sig_counts.sum() + alpha * len(sig_counts))
    bg_prob = (bg_counts.astype(np.float64) + alpha) / (bg_counts.sum() + alpha * len(bg_counts))
    score_by_bin = np.log(sig_prob / bg_prob)
    idx = np.searchsorted(bins, values, side="right") - 1
    idx = np.clip(idx, 0, len(score_by_bin) - 1)
    out = score_by_bin[idx]
    out[~np.isfinite(values)] = float("nan")
    return out.astype(np.float32)


def pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(n), k=1)

