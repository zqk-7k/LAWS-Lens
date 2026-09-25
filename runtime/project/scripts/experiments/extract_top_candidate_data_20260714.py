from __future__ import annotations

import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal


REPO = Path("/root/autodl-tmp/gw-catalog")
OUT = REPO / "results" / "top_candidate_waveform_pe_audit_20260714"
FS = 4096.0
WINDOW_START = -4.0
WINDOW_END = 1.0
PSD_HALF_WIDTH = 128.0
POSTERIOR_LIMIT = 20_000

CATALOGS = {
    "GWTC-3 O3": {
        "events": ["GW191219_163120", "GW191230_180458"],
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
        "events": ["GW230726_002940", "GW230814_230901"],
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
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, start, duration


def whiten_event_window(path: Path, gps: float) -> tuple[np.ndarray, np.ndarray]:
    x, gps_start, duration = read_strain(path)
    fs = len(x) / duration
    if abs(fs - FS) > 1e-3:
        raise ValueError(f"Unexpected sample rate {fs} in {path}")

    ref0 = max(0, int(round((gps - PSD_HALF_WIDTH - gps_start) * fs)))
    ref1 = min(len(x), int(round((gps + PSD_HALF_WIDTH - gps_start) * fs)))
    reference = x[ref0:ref1]
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

    i0 = int(round((gps + WINDOW_START - gps_start) * fs))
    i1 = int(round((gps + WINDOW_END - gps_start) * fs))
    segment = x[i0:i1]
    if len(segment) != int(round((WINDOW_END - WINDOW_START) * fs)):
        raise ValueError(f"Event window outside strain file: {path}")

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
    time = np.arange(len(white), dtype=np.float64) / fs + WINDOW_START
    return time.astype(np.float32), white.astype(np.float32)


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
    candidate_rows: list[pd.DataFrame] = []

    for catalog, cfg in CATALOGS.items():
        event_manifest = pd.read_csv(cfg["event_manifest"])
        strain_manifest = pd.read_csv(cfg["strain_manifest"])
        shortlist = pd.read_csv(cfg["shortlist"])
        a, b = cfg["events"]
        pair_row = shortlist[
            ((shortlist["event_i"] == a) & (shortlist["event_j"] == b))
            | ((shortlist["event_i"] == b) & (shortlist["event_j"] == a))
        ].copy()
        pair_row.insert(0, "catalog", catalog)
        candidate_rows.append(pair_row)

        for event in cfg["events"]:
            event_row = event_manifest[event_manifest["event_name"] == event].iloc[0]
            gps = float(event_row["gps_time"])
            pe_path = Path(str(event_row["sky_map_path"]))
            if not pe_path.is_absolute():
                pe_path = REPO / pe_path
            group = str(event_row[cfg["group_column"]])

            strain_rows = strain_manifest[strain_manifest["event_name"] == event]
            for detector in ("H1", "L1"):
                strain_row = strain_rows[strain_rows["detector"] == detector].iloc[0]
                strain_path = Path(str(strain_row["local_path"]))
                if not strain_path.is_absolute():
                    strain_path = cfg["strain_base"] / strain_path
                time, white = whiten_event_window(strain_path, gps)
                waveform_arrays[f"{event}_{detector}_time"] = time
                waveform_arrays[f"{event}_{detector}_white"] = white

            with h5py.File(pe_path, "r") as h5:
                ds = h5[f"{group}/posterior_samples"]
                names = set(ds.dtype.names or [])
                index = np.linspace(0, len(ds) - 1, min(len(ds), POSTERIOR_LIMIT), dtype=int)
                frame = pd.DataFrame(
                    {field: np.asarray(ds[field][index], dtype=float) for field in POSTERIOR_FIELDS if field in names}
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
                    "waveform_window_seconds": f"{WINDOW_START} to {WINDOW_END}",
                    "waveform_processing": "Welch PSD whitening; 20-500 Hz fourth-order zero-phase Butterworth bandpass; robust off-source scaling",
                }
            )

    posterior = pd.concat(posterior_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    candidates = pd.concat(candidate_rows, ignore_index=True)
    np.savez_compressed(OUT / "top_candidate_whitened_strain_windows.npz", **waveform_arrays)
    posterior.to_csv(OUT / "top_candidate_pe_posterior_samples.csv.gz", index=False, compression="gzip")
    metadata.to_csv(OUT / "top_candidate_event_metadata.csv", index=False)
    candidates.to_csv(OUT / "top_candidate_score_records.csv", index=False)

    summary_rows = []
    for catalog, cfg in CATALOGS.items():
        a, b = cfg["events"]
        for parameter in POSTERIOR_FIELDS:
            if parameter not in posterior.columns:
                continue
            x = posterior.loc[posterior.event == a, parameter].dropna().to_numpy()
            y = posterior.loc[posterior.event == b, parameter].dropna().to_numpy()
            if not len(x) or not len(y):
                continue
            for event, values in [(a, x), (b, y)]:
                q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
                summary_rows.append(
                    {
                        "catalog": catalog,
                        "event": event,
                        "parameter": parameter,
                        "q05": q05,
                        "median": q50,
                        "q95": q95,
                        "pair_overlap_coefficient": overlap_coefficient(x, y),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(OUT / "top_candidate_pe_summary.csv", index=False)

    provenance = {
        "purpose": "Independent waveform and PE audit of the rank-1 pair in each fixed-reference unified-sky deployment.",
        "interpretation": "Descriptive follow-up context only; not a lensing Bayes factor or detection claim.",
        "catalogs": CATALOGS,
        "posterior_fields": POSTERIOR_FIELDS,
        "strain_processing": metadata["waveform_processing"].iloc[0],
    }
    with (OUT / "README.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, default=str)

    package = REPO / "packages" / "top_candidate_waveform_pe_audit_20260714.tar.gz"
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        archive.add(OUT, arcname=OUT.name)
    print(package)
    print(f"posterior rows={len(posterior):,}; package bytes={package.stat().st_size:,}")


if __name__ == "__main__":
    main()
