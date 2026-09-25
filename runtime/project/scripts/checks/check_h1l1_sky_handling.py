from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape), arr.dtype


def check_ligo_waveform_root(root: Path, family: str) -> list[str]:
    issues: list[str] = []
    prefix = family.upper()
    data_path = root / f"{prefix}_data_strain_1.npy"
    snr_single_path = root / f"{prefix}_optimal_SNR_single_1.npy"
    snr_network_path = root / f"{prefix}_optimal_SNR_network_1.npy"
    if not data_path.exists():
        issues.append(f"missing waveform file: {data_path}")
        return issues
    shape, dtype = load_header(data_path)
    if len(shape) != 3 or shape[1] != 2:
        issues.append(f"LIGO waveform should be [n_events,2,N], got {shape} in {data_path}")
    if not snr_single_path.exists() or not snr_network_path.exists():
        issues.append("missing LIGO single/network SNR files")
        return issues
    snr_single = np.load(snr_single_path, mmap_mode="r")
    snr_network = np.load(snr_network_path, mmap_mode="r")
    if snr_single.ndim != 2 or snr_single.shape[1] != 2:
        issues.append(f"single SNR should be [n_events,2], got {snr_single.shape}")
    if snr_network.ndim != 1:
        issues.append(f"network SNR should be [n_events], got {snr_network.shape}")
    n = min(len(snr_network), snr_single.shape[0], 2048)
    expected = np.sqrt(np.sum(np.asarray(snr_single[:n], dtype=np.float64) ** 2, axis=1))
    actual = np.asarray(snr_network[:n], dtype=np.float64)
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-8):
        issues.append("network SNR != sqrt(sum single SNR^2) for sampled rows")
    meta_path = root / "generation_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("scenario") not in {"LIGO_HL", "H1L1", "LIGO"}:
            issues.append(f"unexpected metadata scenario: {meta.get('scenario')}")
        if list(meta.get("ifos", [])) != ["H1", "L1"]:
            issues.append(f"metadata ifos should be ['H1','L1'], got {meta.get('ifos')}")
    else:
        issues.append(f"metadata missing: {meta_path}")
    print(f"LIGO waveform root checked: {root} data_shape={shape} dtype={dtype}")
    return issues


def check_observed_sky_audit(path: Path, snr_network_path: Path | None = None) -> list[str]:
    issues: list[str] = []
    sky = pd.read_csv(path)
    required = {
        "scenario",
        "sky_model",
        "snr_for_sky_mode",
        "snr_for_sky",
        "a90_ref_deg2",
        "ra_true",
        "dec_true",
        "ra_obs",
        "dec_obs",
        "sky_area90_deg2",
        "sky_sigma_rad",
        "uses_h1l1_timing",
        "uses_antenna_pattern_localization",
        "uses_healpix_skymap",
    }
    missing = sorted(required - set(sky.columns))
    if missing:
        issues.append(f"observed sky audit missing columns: {missing}")
        return issues
    scenario = set(sky["scenario"].astype(str))
    if "LIGO_HL" not in scenario:
        issues.append(f"LIGO observed sky should use scenario LIGO_HL, got {sorted(scenario)}")
    if set(sky["snr_for_sky_mode"].astype(str)) != {"network"}:
        issues.append(f"snr_for_sky_mode should be network, got {sorted(set(sky['snr_for_sky_mode'].astype(str)))}")
    if bool(sky["uses_h1l1_timing"].any()):
        issues.append("current A90 approximation should mark uses_h1l1_timing=false")
    if bool(sky["uses_antenna_pattern_localization"].any()):
        issues.append("current A90 approximation should mark uses_antenna_pattern_localization=false")
    if bool(sky["uses_healpix_skymap"].any()):
        issues.append("current A90 approximation should mark uses_healpix_skymap=false")
    if snr_network_path is not None and snr_network_path.exists():
        snr = np.load(snr_network_path, mmap_mode="r")
        n = min(len(snr), len(sky))
        if not np.allclose(sky["snr_for_sky"].to_numpy(dtype=np.float64)[:n], np.asarray(snr[:n], dtype=np.float64), rtol=1e-5, atol=1e-8):
            issues.append("observed sky snr_for_sky does not match provided network SNR")
    print(f"observed sky audit checked: {path} rows={len(sky)} scenarios={sorted(scenario)}")
    return issues


def check_no_true_sky_leakage(feature_csv: Path) -> list[str]:
    issues: list[str] = []
    cols = set(pd.read_csv(feature_csv, nrows=0).columns)
    forbidden = {"ra_true", "dec_true", "true_sky_sep", "true_sky_overlap"}
    leaked = sorted(cols & forbidden)
    if leaked:
        issues.append(f"feature file leaks true sky columns: {leaked}")
    print(f"feature columns checked: {feature_csv}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Check H1+L1 waveform/SNR and observed-sky handling.")
    parser.add_argument("--ligo-root", type=Path, help="Directory containing LIGO family waveform npy files.")
    parser.add_argument("--family", choices=["SIS", "PM"], default="SIS")
    parser.add_argument("--sky-audit", type=Path, help="Observed sky audit CSV generated by stage2.")
    parser.add_argument("--snr-network", type=Path, help="Optional network SNR npy used to verify sky snr_for_sky.")
    parser.add_argument("--feature-csv", type=Path, help="Optional pair feature CSV to check for true-sky leakage.")
    args = parser.parse_args()

    issues: list[str] = []
    if args.ligo_root:
        issues.extend(check_ligo_waveform_root(args.ligo_root, args.family))
    if args.sky_audit:
        issues.extend(check_observed_sky_audit(args.sky_audit, args.snr_network))
    if args.feature_csv:
        issues.extend(check_no_true_sky_leakage(args.feature_csv))

    if issues:
        print("FAILED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
