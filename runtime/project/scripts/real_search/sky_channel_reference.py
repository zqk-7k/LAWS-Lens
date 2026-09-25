#!/usr/bin/env python3
"""Reference implementation for the sky-overlap channel.

This file is intentionally self-contained. It extracts the sky-channel logic
used in the catalog experiments into a smaller reference module:

1. simulated observed-sky proxy from true ra/dec + SNR-derived A90;
2. pair-level observed-sky features;
3. real HEALPix posterior-map overlap;
4. row-wise z-score and weighted fusion helpers.

It is not meant to replace the production scripts. The production entry points
remain:
  - scripts/real_search/01_build_healpix_overlap.py
  - scripts/real_search/02_build_observable_features.py
  - scripts/real_search/15_gwtc34_real_deployment.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-8
DEG2_TO_SR = (math.pi / 180.0) ** 2


@dataclass(frozen=True)
class SkyScenario:
    """Approximate localization scenario for simulated catalogs."""

    name: str
    label: str
    a90_ref_deg2: float
    rho_ref: float = 12.0
    clip_min_deg2: float = 10.0
    clip_max_deg2: float = 1000.0
    lognormal_sigma: float = 0.35
    snr_column: str = "snr"


SCENARIOS: dict[str, SkyScenario] = {
    "ET3": SkyScenario(
        name="ET3",
        label="ET three-arm network-SNR A90 approximation",
        a90_ref_deg2=100.0,
        rho_ref=12.0,
        clip_min_deg2=20.0,
        clip_max_deg2=1000.0,
    ),
    "LIGO_HL": SkyScenario(
        name="LIGO_HL",
        label="LIGO H1+L1 network-SNR A90 approximation",
        a90_ref_deg2=100.0,
        rho_ref=12.0,
        clip_min_deg2=10.0,
        clip_max_deg2=500.0,
    ),
    "FIXED_A90_100": SkyScenario(
        name="FIXED_A90_100",
        label="Fixed A90=100 deg2 ablation",
        a90_ref_deg2=100.0,
        rho_ref=12.0,
        clip_min_deg2=10.0,
        clip_max_deg2=2000.0,
        lognormal_sigma=0.0,
    ),
}


def with_a90_ref(scenario: SkyScenario, a90_ref_deg2: float | None) -> SkyScenario:
    if a90_ref_deg2 is None:
        return scenario
    return replace(
        scenario,
        a90_ref_deg2=float(a90_ref_deg2),
        label=f"{scenario.label} sweep A90={float(a90_ref_deg2):g} deg2",
    )


def unit_from_radec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    ).astype(np.float64)


def radec_from_unit(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), EPS)
    ra = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2.0 * np.pi)
    dec = np.arcsin(np.clip(vec[:, 2], -1.0, 1.0))
    return ra.astype(np.float64), dec.astype(np.float64)


def angular_sep_rad(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    sin_d1 = np.sin(dec1)
    sin_d2 = np.sin(dec2)
    cos_d1 = np.cos(dec1)
    cos_d2 = np.cos(dec2)
    dra = ra1 - ra2
    cosang = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(dra)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def tangent_basis(true_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref = np.tile(z_axis, (len(true_vec), 1))
    near_pole = np.abs(true_vec @ z_axis) > 0.95
    ref[near_pole] = x_axis
    e1 = np.cross(ref, true_vec)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), EPS)
    e2 = np.cross(true_vec, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), EPS)
    return e1, e2


def a90_to_sigma_rad(a90_deg2: np.ndarray) -> np.ndarray:
    a90_rad2 = np.asarray(a90_deg2, dtype=np.float64) * DEG2_TO_SR
    return np.sqrt(a90_rad2 / (2.0 * np.pi * np.log(10.0))).astype(np.float64)


def compute_a90_from_snr(snr: np.ndarray, scenario: SkyScenario, rng: np.random.Generator) -> np.ndarray:
    if scenario.lognormal_sigma > 0:
        jitter = rng.lognormal(mean=0.0, sigma=scenario.lognormal_sigma, size=len(snr))
    else:
        jitter = np.ones(len(snr), dtype=np.float64)
    a90 = scenario.a90_ref_deg2 * (scenario.rho_ref / np.maximum(snr, 1.0)) ** 2 * jitter
    return np.clip(a90, scenario.clip_min_deg2, scenario.clip_max_deg2).astype(np.float64)


def sample_observed_sky_center(
    ra_true: np.ndarray,
    dec_true: np.ndarray,
    sigma_rad: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample observed sky centers around true coordinates on the tangent plane."""

    true_vec = unit_from_radec(ra_true, dec_true)
    e1, e2 = tangent_basis(true_vec)
    dx = rng.normal(0.0, sigma_rad)
    dy = rng.normal(0.0, sigma_rad)
    obs_vec = true_vec + dx[:, None] * e1 + dy[:, None] * e2
    obs_vec /= np.maximum(np.linalg.norm(obs_vec, axis=1, keepdims=True), EPS)
    return radec_from_unit(obs_vec)


def build_observed_sky_proxy(
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    scenario: SkyScenario,
    seed: int = 20260629,
) -> pd.DataFrame:
    """Build simulated observed sky information.

    Required columns:
      raw_obs: ra, dec in radians
      time_obs: scenario.snr_column, usually network SNR
    """

    rng = np.random.default_rng(seed)
    snr = time_obs[scenario.snr_column].to_numpy(dtype=np.float64)
    a90 = compute_a90_from_snr(snr, scenario, rng)
    sigma = a90_to_sigma_rad(a90)
    ra_true = raw_obs["ra"].to_numpy(dtype=np.float64)
    dec_true = raw_obs["dec"].to_numpy(dtype=np.float64)
    ra_obs, dec_obs = sample_observed_sky_center(ra_true, dec_true, sigma, rng)
    return pd.DataFrame(
        {
            "event_id": np.arange(len(raw_obs), dtype=np.int64),
            "scenario": scenario.name,
            "scenario_label": scenario.label,
            "snr_for_sky": snr,
            "ra_obs": ra_obs,
            "dec_obs": dec_obs,
            "sky_area90_deg2": a90,
            "sky_sigma_rad": sigma,
        }
    )


def observed_sky_pair_table(sky_obs: pd.DataFrame) -> pd.DataFrame:
    """Build unordered pair-level observed-sky features."""

    n = len(sky_obs)
    ii, jj = np.triu_indices(n, k=1)
    theta = angular_sep_rad(
        sky_obs["ra_obs"].to_numpy(dtype=np.float64)[ii],
        sky_obs["dec_obs"].to_numpy(dtype=np.float64)[ii],
        sky_obs["ra_obs"].to_numpy(dtype=np.float64)[jj],
        sky_obs["dec_obs"].to_numpy(dtype=np.float64)[jj],
    )
    sigma = np.sqrt(
        sky_obs["sky_sigma_rad"].to_numpy(dtype=np.float64)[ii] ** 2
        + sky_obs["sky_sigma_rad"].to_numpy(dtype=np.float64)[jj] ** 2
    )
    d_sky = theta / np.maximum(sigma, EPS)
    step = np.full(len(d_sky), -0.5, dtype=np.float32)
    step[d_sky <= 3.03] = 0.1
    step[d_sky <= 2.15] = 0.5
    step[d_sky <= 1.18] = 1.0
    var = np.maximum(sigma**2, EPS)
    return pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "sky_sep_obs_rad": theta.astype(np.float64),
            "sky_norm_sep": d_sky.astype(np.float32),
            "sky_step_weight": step,
            "sky_gaussian_weight": np.exp(-0.5 * d_sky * d_sky).astype(np.float32),
            "sky_log_overlap": (-np.log(2.0 * np.pi * var) - theta * theta / (2.0 * var)).astype(np.float32),
        }
    )


def sanitize_probability_map(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.where(np.isfinite(prob) & (prob > 0), prob, 0.0)
    total = float(prob.sum())
    if total <= 0.0:
        raise ValueError("HEALPix map has non-positive probability sum")
    return prob / total


def read_healpix_probability(path: Path, target_nside: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a HDF5/FITS HEALPix probability map and normalize it.

    Requires optional dependencies h5py/healpy/astropy when used.
    """

    import healpy as hp

    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5:
            group = None
            for preferred in ["C01:Mixed", "C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM", "C00:Mixed"]:
                if f"{preferred}/skymap/data" in h5:
                    group = preferred
                    break
            if group is None:
                for key in h5.keys():
                    if f"{key}/skymap/data" in h5:
                        group = str(key)
                        break
            if group is None:
                raise FileNotFoundError(f"No /skymap/data group in {path}")
            prob = np.asarray(h5[f"{group}/skymap/data"][:], dtype=np.float64).reshape(-1)
            nest = False
            nest_key = f"{group}/skymap/meta_data/nest"
            if nest_key in h5:
                nest = str(h5[nest_key][()]).strip().lower() in {"1", "true", "b'true'"}
        source_nside = hp.npix2nside(len(prob))
        prob = sanitize_probability_map(prob)
        if nest:
            prob = hp.reorder(prob, n2r=True)
        if source_nside != target_nside:
            prob = hp.ud_grade(prob, nside_out=target_nside, order_in="RING", order_out="RING", power=-2)
            prob = sanitize_probability_map(prob)
        return prob.astype(np.float32), {"source_format": "pe_hdf5_skymap", "source_nside": int(source_nside)}

    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[1]
        ordering = str(hdu.header.get("ORDERING", "RING")).upper()
        prob = np.asarray(hdu.data["PROB"], dtype=np.float64).reshape(-1)
        source_nside = int(hdu.header.get("NSIDE") or hp.npix2nside(len(prob)))
    prob = sanitize_probability_map(prob)
    if ordering == "NESTED":
        prob = hp.reorder(prob, n2r=True)
    if source_nside != target_nside:
        prob = hp.ud_grade(prob, nside_out=target_nside, order_in="RING", order_out="RING", power=-2)
        prob = sanitize_probability_map(prob)
    return prob.astype(np.float32), {"source_format": "fits_prob", "source_nside": int(source_nside)}


def healpix_overlap_table(events: pd.DataFrame, target_nside: int = 512, repo_root: Path | None = None) -> pd.DataFrame:
    """Compute unordered real-event HEALPix posterior overlaps.

    Required event columns:
      event_name, sky_map_path
    """

    repo_root = repo_root or Path(".")
    maps: list[np.ndarray] = []
    for _, row in events.reset_index(drop=True).iterrows():
        path = Path(str(row["sky_map_path"]))
        if not path.is_absolute():
            path = repo_root / path
        prob, _ = read_healpix_probability(path, target_nside=target_nside)
        maps.append(prob.astype(np.float32, copy=False))
    stack = np.vstack(maps).astype(np.float32, copy=False)
    raw = stack @ stack.T
    norms = np.sqrt(np.clip(np.diag(raw), 0.0, None))
    cosine = raw / np.maximum(norms[:, None] * norms[None, :], 1e-30)
    ii, jj = np.triu_indices(len(events), k=1)
    return pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": events["event_name"].to_numpy()[ii],
            "event_j": events["event_name"].to_numpy()[jj],
            "raw_posterior_overlap": raw[ii, jj].astype(np.float64),
            "cosine_overlap": cosine[ii, jj].astype(np.float64),
            "sky_log_cosine_overlap": np.log(np.clip(cosine[ii, jj], 1e-300, None)).astype(np.float64),
        }
    )


def row_z_neutral(mat: np.ndarray) -> np.ndarray:
    """Row-wise z-score. NaN entries receive neutral contribution 0."""

    arr = np.asarray(mat, dtype=np.float64).copy()
    np.fill_diagonal(arr, np.nan)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    z = (arr - mu) / np.maximum(sd, 1e-8)
    z[~np.isfinite(z)] = 0.0
    np.fill_diagonal(z, -np.inf)
    return z.astype(np.float32)


def matrix_from_pair_table(df: pd.DataFrame, n: int, column: str) -> np.ndarray:
    mat = np.full((n, n), np.nan, dtype=np.float64)
    ii = df["idx_i"].to_numpy(dtype=np.int32)
    jj = df["idx_j"].to_numpy(dtype=np.int32)
    vv = df[column].to_numpy(dtype=np.float64)
    mat[ii, jj] = vv
    mat[jj, ii] = vv
    return mat


def fuse_waveform_time_sky(
    pairs: pd.DataFrame,
    n_events: int,
    weights: dict[str, float],
    waveform_col: str = "waveform_score",
    time_col: str = "time_score",
    sky_col: str = "sky_score",
) -> pd.DataFrame:
    """Minimal waveform/time/sky row-z fusion for unordered pair tables."""

    out = pairs.copy()
    wf = row_z_neutral(matrix_from_pair_table(out, n_events, waveform_col))
    tm = row_z_neutral(matrix_from_pair_table(out, n_events, time_col))
    sk = row_z_neutral(matrix_from_pair_table(out, n_events, sky_col))
    score = weights["waveform"] * wf + weights["time"] * tm + weights["sky"] * sk
    ii = out["idx_i"].to_numpy(dtype=np.int32)
    jj = out["idx_j"].to_numpy(dtype=np.int32)
    out["final_score_i_to_j"] = score[ii, jj]
    out["final_score_j_to_i"] = score[jj, ii]
    out["final_score"] = np.maximum(score[ii, jj], score[jj, ii])
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def _cmd_observed_proxy(args: argparse.Namespace) -> None:
    raw = pd.read_csv(args.raw_observations)
    time = pd.read_csv(args.time_observations)
    scenario = with_a90_ref(SCENARIOS[args.scenario], args.a90_ref_deg2)
    sky = build_observed_sky_proxy(raw, time, scenario=scenario, seed=args.seed)
    pairs = observed_sky_pair_table(sky)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sky.to_csv(args.out_dir / "observed_sky_proxy.csv", index=False)
    pairs.to_parquet(args.out_dir / "observed_sky_pair_features.parquet", index=False)
    print(json.dumps({"n_events": len(sky), "n_pairs": len(pairs), "out_dir": str(args.out_dir)}, indent=2))


def _cmd_healpix_overlap(args: argparse.Namespace) -> None:
    events = pd.read_csv(args.event_manifest)
    if args.primary_only and "include_in_primary_search" in events.columns:
        events = events[events["include_in_primary_search"] == True].copy()
    if args.sky_available_only and "sky_map_available" in events.columns:
        events = events[events["sky_map_available"] == True].copy()
    events = events.reset_index(drop=True)
    out = healpix_overlap_table(events, target_nside=args.nside, repo_root=args.repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(json.dumps({"n_events": len(events), "n_pairs": len(out), "output": str(args.out)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("observed-proxy", help="Build simulated observed-sky proxy and pair features")
    p.add_argument("--raw-observations", type=Path, required=True)
    p.add_argument("--time-observations", type=Path, required=True)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="ET3")
    p.add_argument("--a90-ref-deg2", type=float, default=None)
    p.add_argument("--seed", type=int, default=20260629)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=_cmd_observed_proxy)

    p = sub.add_parser("healpix-overlap", help="Build real HEALPix posterior overlap table")
    p.add_argument("--event-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--primary-only", action="store_true")
    p.add_argument("--sky-available-only", action="store_true")
    p.set_defaults(func=_cmd_healpix_overlap)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
