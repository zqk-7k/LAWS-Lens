from __future__ import annotations

import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal


REPO = Path("/root/autodl-tmp/gw-catalog")
OUT = REPO / "results" / "top5_candidate_waveform_pe_audit_20260714"
FS = 4096.0
WINDOW_START = -4.0
WINDOW_END = 1.0
PSD_HALF_WIDTH = 128.0
POSTERIOR_LIMIT = 20_000

CATALOGS = {
    "GWTC-3 O3": {
        "event_manifest": REPO
        / "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv",
        "strain_manifest": REPO
        / "runs/real_gwtc_lensing_search_20260625/data/strain_gwosc_download_manifest.csv",
        "strain_base": REPO / "runs/real_gwtc_lensing_search_20260625",
        "group_column": "sky_map_internal_group",
        "shortlist": REPO
        / "runs/unified_sky_posterior_overlap_20260708/gwtc3/gwtc3_candidate_shortlist_unified_waveform_time_sky.csv",
    },
    "GWTC-4.0 O4a": {
        "event_manifest": REPO
        / "runs/real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest.csv",
        "strain_manifest": REPO
        / "runs/gwtc4p1_data_completion_20260628/data/strain_manifest_gwtc4p1.csv",
        "strain_base": REPO,
        "group_column": "sky_map_group",
        "shortlist": REPO
        / "runs/unified_sky_posterior_overlap_20260708/gwtc4/gwtc4_candidate_shortlist_unified_waveform_time_sky.csv",
    },
}

POSTERIOR_FIELDS = [
    "chirp_mass",
    "chirp_mass_source",
    "mass_1_source",
    "mass_2_source",
    "mass_ratio",
    "chi_eff",
    "luminosity_distance",
    "redshift",
]


def scalar(h5: h5py.File, path: str) -> float:
    return float(np.asarray(h5[path][()]).item())


def read_strain(path: Path) -> tuple[np.ndarray, float, float]:
    with h5py.File(path, "r") as h5:
        x = np.asarray(h5["strain/Strain"][:], dtype=np.float64)
        start = scalar(h5, "meta/GPSstart")
        duration = scalar(h5, "meta/Duration")
    return x, start, duration


def whiten_event_window(
    path: Path, gps: float
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    x, gps_start, duration = read_strain(path)
    fs = len(x) / duration
    if abs(fs - FS) > 1e-3:
        raise ValueError(f"Unexpected sample rate {fs} in {path}")

    i0 = int(round((gps + WINDOW_START - gps_start) * fs))
    i1 = int(round((gps + WINDOW_END - gps_start) * fs))
    segment_raw = x[i0:i1]
    expected = int(round((WINDOW_END - WINDOW_START) * fs))
    if len(segment_raw) != expected:
        raise ValueError(f"Event window outside strain file: {path}")
    finite_fraction = float(np.mean(np.isfinite(segment_raw)))
    time = np.arange(expected, dtype=np.float64) / fs + WINDOW_START
    if finite_fraction < 0.95:
        return time.astype(np.float32), np.zeros(expected, dtype=np.float32), finite_fraction, False

    ref0 = max(0, int(round((gps - PSD_HALF_WIDTH - gps_start) * fs)))
    ref1 = min(len(x), int(round((gps + PSD_HALF_WIDTH - gps_start) * fs)))
    reference = np.nan_to_num(x[ref0:ref1], nan=0.0, posinf=0.0, neginf=0.0)
    frequencies, psd = signal.welch(
        reference,
        fs=fs,
        window="hann",
        nperseg=int(8 * fs),
        noverlap=int(4 * fs),
        detrend="constant",
        scaling="density",
    )
    psd = np.maximum(psd, np.nanmedian(psd) * 1e-12)

    segment = np.nan_to_num(segment_raw, nan=0.0, posinf=0.0, neginf=0.0)
    taper = signal.windows.tukey(len(segment), alpha=0.08)
    spectrum = np.fft.rfft(segment * taper)
    f_segment = np.fft.rfftfreq(len(segment), d=1.0 / fs)
    psd_interp = np.interp(f_segment, frequencies, psd)
    white_spectrum = spectrum / np.sqrt(np.maximum(psd_interp, 1e-60))
    white = np.fft.irfft(white_spectrum, n=len(segment))

    sos = signal.butter(4, [20.0, 500.0], btype="bandpass", fs=fs, output="sos")
    white = signal.sosfiltfilt(sos, white)
    off = np.r_[white[: int(1.5 * fs)], white[-int(0.5 * fs) :]]
    scale = np.median(np.abs(off - np.median(off))) * 1.4826
    white = (white - np.median(off)) / max(scale, 1e-12)
    return time.astype(np.float32), white.astype(np.float32), finite_fraction, True


def overlap_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    lo = min(np.quantile(finite_x, 0.001), np.quantile(finite_y, 0.001))
    hi = max(np.quantile(finite_x, 0.999), np.quantile(finite_y, 0.999))
    bins = np.linspace(lo, hi, 301)
    hx, _ = np.histogram(finite_x, bins=bins, density=True)
    hy, _ = np.histogram(finite_y, bins=bins, density=True)
    return float(np.sum(np.minimum(hx, hy) * np.diff(bins)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    waveform_arrays: dict[str, np.ndarray] = {}
    posterior_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    detector_rows: list[dict] = []
    candidate_frames: list[pd.DataFrame] = []

    for catalog, cfg in CATALOGS.items():
        event_manifest = pd.read_csv(cfg["event_manifest"])
        strain_manifest = pd.read_csv(cfg["strain_manifest"])
        shortlist = pd.read_csv(cfg["shortlist"]).sort_values("rank").head(5).copy()
        shortlist.insert(0, "catalog", catalog)
        candidate_frames.append(shortlist)
        events = pd.unique(shortlist[["event_i", "event_j"]].to_numpy().ravel())

        for event in events:
            event_row = event_manifest[event_manifest["event_name"] == event].iloc[0]
            gps = float(event_row["gps_time"])
            pe_path = Path(str(event_row["sky_map_path"]))
            if not pe_path.is_absolute():
                pe_path = REPO / pe_path
            group = str(event_row[cfg["group_column"]])

            strain_rows = strain_manifest[strain_manifest["event_name"] == event]
            available_detectors: list[str] = []
            for detector in ("H1", "L1"):
                detector_match = strain_rows[strain_rows["detector"] == detector]
                if detector_match.empty or pd.isna(detector_match.iloc[0].get("local_path")):
                    time = np.arange(int((WINDOW_END - WINDOW_START) * FS), dtype=np.float32) / FS + WINDOW_START
                    white = np.zeros_like(time)
                    finite_fraction = 0.0
                    available = False
                    strain_path = ""
                else:
                    strain_row = detector_match.iloc[0]
                    path = Path(str(strain_row["local_path"]))
                    if not path.is_absolute():
                        path = cfg["strain_base"] / path
                    time, white, finite_fraction, available = whiten_event_window(path, gps)
                    strain_path = str(path)
                waveform_arrays[f"{event}_{detector}_time"] = np.asarray(time, dtype=np.float32)
                waveform_arrays[f"{event}_{detector}_white"] = np.asarray(white, dtype=np.float32)
                if available:
                    available_detectors.append(detector)
                detector_rows.append(
                    {
                        "catalog": catalog,
                        "event": event,
                        "detector": detector,
                        "strain_path": strain_path,
                        "event_window_finite_fraction": finite_fraction,
                        "usable_for_display": available,
                    }
                )

            with h5py.File(pe_path, "r") as h5:
                ds = h5[f"{group}/posterior_samples"]
                names = set(ds.dtype.names or [])
                index = np.linspace(0, len(ds) - 1, min(len(ds), POSTERIOR_LIMIT), dtype=int)
                frame = pd.DataFrame(
                    {
                        field: np.asarray(ds[field][index], dtype=float)
                        for field in POSTERIOR_FIELDS
                        if field in names
                    }
                )
            frame.insert(0, "event", event)
            frame.insert(0, "catalog", catalog)
            posterior_frames.append(frame)
            metadata_rows.append(
                {
                    "catalog": catalog,
                    "event": event,
                    "gps_time": gps,
                    "network_snr": event_row.get("network_snr", np.nan),
                    "pe_group": group,
                    "pe_file": str(pe_path),
                    "posterior_samples_exported": len(frame),
                    "finite_detectors_at_trigger": "; ".join(available_detectors) or "none",
                }
            )

    posterior = pd.concat(posterior_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    detector_audit = pd.DataFrame(detector_rows)

    summary_rows: list[dict] = []
    for _, pair in candidates.iterrows():
        for parameter in POSTERIOR_FIELDS:
            if parameter not in posterior.columns:
                continue
            x = posterior.loc[posterior.event == pair.event_i, parameter].dropna().to_numpy()
            y = posterior.loc[posterior.event == pair.event_j, parameter].dropna().to_numpy()
            if not len(x) or not len(y):
                continue
            overlap = overlap_coefficient(x, y)
            for event, values in ((pair.event_i, x), (pair.event_j, y)):
                q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
                summary_rows.append(
                    {
                        "catalog": pair.catalog,
                        "rank": int(pair["rank"]),
                        "event_i": pair.event_i,
                        "event_j": pair.event_j,
                        "event": event,
                        "parameter": parameter,
                        "q05": q05,
                        "median": q50,
                        "q95": q95,
                        "pair_overlap_coefficient": overlap,
                    }
                )

    np.savez_compressed(OUT / "top5_whitened_strain_windows.npz", **waveform_arrays)
    posterior.to_csv(OUT / "top5_pe_posterior_samples.csv.gz", index=False, compression="gzip")
    candidates.to_csv(OUT / "top5_candidate_score_records.csv", index=False)
    metadata.to_csv(OUT / "top5_event_metadata.csv", index=False)
    detector_audit.to_csv(OUT / "top5_detector_window_audit.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "top5_pe_summary.csv", index=False)

    provenance = {
        "purpose": "Independent waveform and PE audit of the top five pairs in each fixed-reference unified-sky deployment.",
        "interpretation": "Descriptive follow-up context only; not a lensing Bayes factor or detection claim.",
        "catalogs": CATALOGS,
        "posterior_fields": POSTERIOR_FIELDS,
        "posterior_limit_per_event": POSTERIOR_LIMIT,
    }
    with (OUT / "README.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, default=str)

    package = REPO / "packages" / "top5_candidate_waveform_pe_audit_20260714.tar.gz"
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        archive.add(OUT, arcname=OUT.name)
    print(package)
    print(f"events={len(metadata)}; pairs={len(candidates)}; posterior rows={len(posterior):,}")
    print(f"package bytes={package.stat().st_size:,}")


if __name__ == "__main__":
    main()
