from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import (
    REPO_ROOT,
    choose_h5_skymap_group,
    default_run_dir,
    ensure_run_dirs,
    event_short_name,
    h5_detectors,
    infer_run_from_gps,
    utc_now,
    write_json,
)


DETECTORS = ("H1", "L1", "V1")


def rel(path: Path | str | float | None) -> str:
    if path is None:
        return ""
    text = str(path)
    if not text or text == "nan":
        return ""
    p = Path(text)
    if p.is_absolute():
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    return text


def local_strain_paths(event_name: str) -> dict[str, str]:
    short = event_short_name(event_name)
    event_dir = REPO_ROOT / "data" / "gwtc_real" / short
    out = {}
    for det in DETECTORS:
        matches = sorted(event_dir.glob(f"*{det}*.hdf5")) + sorted(event_dir.glob(f"*{det}*.h5"))
        out[det] = rel(matches[0]) if matches else ""
    return out


def read_gwosc_far_map() -> dict[str, dict[str, float | str]]:
    path = REPO_ROOT / "data" / "gwtc_raw" / "allevents.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, float | str]] = {}
    for _, row in df.iterrows():
        name = event_short_name(str(row.get("commonName", "")))
        if not name:
            continue
        # Keep the latest/highest-confidence row already used by the existing GWTC extractor.
        out[name] = {
            "far": row.get("far", np.nan),
            "reference": row.get("reference", ""),
            "jsonurl": row.get("jsonurl", ""),
            "catalog": row.get("catalog.shortName", ""),
        }
    return out


def h5_has_skymap(path: Path) -> tuple[bool, str]:
    group = choose_h5_skymap_group(path)
    return (group is not None), (group or "")


def load_gwtc5_instruments() -> dict[str, str]:
    path = REPO_ROOT / "data" / "gwtc5_raw" / "IGWN-GWTC5p0-59d160a18_25-SearchSummaryTable.hdf5"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with h5py.File(path, "r") as h5:
        if "search_summary" not in h5:
            return out
        arr = h5["search_summary"][:]
        for row in arr:
            name = event_short_name(row["gw_name"].decode() if isinstance(row["gw_name"], bytes) else row["gw_name"])
            inst = row["instruments"].decode() if isinstance(row["instruments"], bytes) else str(row["instruments"])
            out[name] = inst
    return out


def build_gwtc3_rows(far_map: dict[str, dict[str, float | str]]) -> list[dict]:
    path = REPO_ROOT / "data" / "gwtc3_observables.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    obs = pd.read_csv(path)
    rows = []
    for _, row in obs.iterrows():
        event = event_short_name(row["event_name"])
        h5_path = REPO_ROOT / rel(row.get("pe_h5_file", ""))
        sky_ok, sky_group = h5_has_skymap(h5_path)
        strain = local_strain_paths(event)
        far_info = far_map.get(event, {})
        detectors = h5_detectors(h5_path)
        rows.append({
            "event_name": event,
            "catalog": row.get("catalog", "GWTC-3"),
            "sample_role": "primary_o1_o3_confident_bbh_pe",
            "include_in_primary_search": bool(sky_ok),
            "run": infer_run_from_gps(float(row["gps_trigger_time"])),
            "gps_time": float(row["gps_trigger_time"]),
            "detectors_available": detectors,
            "network_snr": float(row.get("network_snr", np.nan)),
            "far": far_info.get("far", np.nan),
            "catalog_confidence": row.get("catalog", ""),
            "chirp_mass": float(row.get("chirp_mass_median", np.nan)),
            "mass_1": float(row.get("mass_1_source", np.nan)),
            "mass_2": float(row.get("mass_2_source", np.nan)),
            "mass_ratio": float(row.get("mass_ratio_median", np.nan)),
            "luminosity_distance": float(row.get("luminosity_distance_median", np.nan)),
            "p_astro": float(row.get("p_astro", np.nan)),
            "sky_map_path": rel(h5_path),
            "sky_map_format": "pe_hdf5_skymap_data",
            "sky_map_internal_group": sky_group,
            "sky_map_available": bool(sky_ok),
            "sky_map_source": "GWTC PE release HDF5 /skymap/data",
            "pe_release_url": far_info.get("jsonurl", ""),
            "strain_H1_path": strain["H1"],
            "strain_L1_path": strain["L1"],
            "strain_V1_path": strain["V1"],
            "strain_H1_available": bool(strain["H1"]),
            "strain_L1_available": bool(strain["L1"]),
            "strain_V1_available": bool(strain["V1"]),
            "dq_status": "not_checked",
            "notes": "Primary observable search sample. Known NS/NSBH events were excluded by existing GWTC extractor.",
        })
    return rows


def build_gwtc5_extension_rows() -> list[dict]:
    path = REPO_ROOT / "data" / "gwtc5_observables.csv"
    if not path.exists():
        return []
    obs = pd.read_csv(path)
    inst_map = load_gwtc5_instruments()
    rows = []
    for _, row in obs.iterrows():
        event = event_short_name(row["event_name"])
        sky_path = REPO_ROOT / rel(row.get("skymap_file", ""))
        strain = local_strain_paths(event)
        rows.append({
            "event_name": event,
            "catalog": row.get("catalog", "GWTC-5.0-search"),
            "sample_role": "extension_gwtc5_search_bbh_not_primary",
            "include_in_primary_search": False,
            "run": infer_run_from_gps(float(row["gps_trigger_time"])),
            "gps_time": float(row["gps_trigger_time"]),
            "detectors_available": inst_map.get(event, ""),
            "network_snr": float(row.get("network_snr", np.nan)),
            "far": float(row.get("far_hz", np.nan)),
            "catalog_confidence": "GWTC-5 search p_astro/p_BBH filtered; search-map extension only",
            "chirp_mass": float(row.get("chirp_mass_median", np.nan)),
            "mass_1": float("nan"),
            "mass_2": float("nan"),
            "mass_ratio": float(row.get("mass_ratio_median", np.nan)),
            "luminosity_distance": float(row.get("luminosity_distance_median", np.nan)),
            "p_astro": float(row.get("p_astro", np.nan)),
            "sky_map_path": rel(sky_path),
            "sky_map_format": "fits_prob_search_skymap",
            "sky_map_internal_group": "",
            "sky_map_available": bool(sky_path.exists()),
            "sky_map_source": "GWTC-5 search skymap, not full PE posterior",
            "pe_release_url": "",
            "strain_H1_path": strain["H1"],
            "strain_L1_path": strain["L1"],
            "strain_V1_path": strain["V1"],
            "strain_H1_available": bool(strain["H1"]),
            "strain_L1_available": bool(strain["L1"]),
            "strain_V1_available": bool(strain["V1"]),
            "dq_status": "not_checked",
            "notes": "Listed for audit only. Not included in primary Phase A-D scores unless explicitly enabled later.",
        })
    return rows


def write_secondary_manifests(run_dir: Path, event_manifest: pd.DataFrame) -> None:
    strain_rows = []
    for _, row in event_manifest.iterrows():
        for det in DETECTORS:
            path = str(row.get(f"strain_{det}_path", ""))
            strain_rows.append({
                "event_name": row["event_name"],
                "run": row["run"],
                "detector": det,
                "strain_path": path,
                "exists": bool(path),
                "planned_window_s": 256,
                "window_definition": "event center +/-128 s",
                "status": "local_found" if path else "missing_not_downloaded_phase_ad",
                "dq_status": row.get("dq_status", "not_checked"),
                "include_in_primary_search": bool(row["include_in_primary_search"]),
            })
    pd.DataFrame(strain_rows).to_csv(run_dir / "data" / "strain_manifest.csv", index=False)

    sky_cols = [
        "event_name", "run", "catalog", "sample_role", "include_in_primary_search",
        "sky_map_path", "sky_map_format", "sky_map_internal_group",
        "sky_map_available", "sky_map_source",
    ]
    event_manifest[sky_cols].to_csv(run_dir / "data" / "skymap_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--include-gwtc5-extension", action="store_true", default=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    far_map = read_gwosc_far_map()
    rows = build_gwtc3_rows(far_map)
    if args.include_gwtc5_extension:
        rows.extend(build_gwtc5_extension_rows())

    event_manifest = pd.DataFrame(rows).sort_values(["include_in_primary_search", "gps_time"], ascending=[False, True])
    event_manifest.to_csv(run_dir / "data" / "event_manifest.csv", index=False)
    write_secondary_manifests(run_dir, event_manifest)

    primary = event_manifest[event_manifest["include_in_primary_search"] == True]
    summary = {
        "generated_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "total_manifest_events": int(len(event_manifest)),
        "primary_events": int(len(primary)),
        "extension_events": int(len(event_manifest) - len(primary)),
        "primary_with_skymap": int(primary["sky_map_available"].sum()),
        "events_with_any_local_strain": int(event_manifest[["strain_H1_available", "strain_L1_available", "strain_V1_available"]].any(axis=1).sum()),
        "events_with_H1L1_local_strain": int((event_manifest["strain_H1_available"] & event_manifest["strain_L1_available"]).sum()),
        "scope_note": "Phase A-D primary search uses O1-O3 confident BBH events with GWTC PE HDF5 HEALPix skymaps. GWTC-5/O4 search events are manifest-only extension rows unless explicitly enabled later.",
    }
    write_json(run_dir / "run_config.json", {
        "created_at_utc": utc_now(),
        "phase": "A-D observable-only baseline",
        "primary_scope": "O1-O3 confident BBH from data/gwtc3_observables.csv",
        "extension_scope": "GWTC-5 search BBH rows listed for audit, not primary",
        "nside": 512,
        "candidate_output_policy": "candidate shortlist for Bayesian follow-up; no real lensing claim",
    })
    write_json(run_dir / "data" / "data_scope_summary.json", summary)
    (run_dir / "logs" / "data_audit.log").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

