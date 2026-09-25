#!/usr/bin/env python3
"""Build a multipage morphology atlas for the displayed v9.3 Top-10 pairs."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import jax
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import IMRPhenomXPHM
from scipy.signal import hilbert


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path(__file__).resolve().parent
CURRENT = ROOT / "_ovl_sync_6a0019996cb0d99d490519e9/source_data/current_v93"
O3_MANIFEST = (
    ROOT
    / "overleaf-project/real_gwtc_lensing_search_20260625_scaleaware_waveform_passed_clean_current"
    / "runs/real_gwtc_lensing_search_20260625/data/event_manifest.csv"
)
O4_MANIFEST = (
    ROOT
    / "server_pull/real_noise_injection_v6_20260721/extracted/source_manifests"
    / "real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest.csv"
)
PE_ROOT = AUDIT / "pe_files"
PE_AUDIT = AUDIT / "top10_expanded_pe_audit.csv"
OUTPUT_PDF = AUDIT / "top10_noise_free_waveform_atlas.pdf"
OUTPUT_SAMPLES = AUDIT / "top10_noise_free_waveform_samples.csv.gz"
OUTPUT_METADATA = AUDIT / "top10_noise_free_waveform_metadata.csv"

PARAMETER_FIELDS = (
    "chirp_mass",
    "symmetric_mass_ratio",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "luminosity_distance",
    "phase",
    "iota",
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_pe_file(deployment: str, event: str) -> Path:
    matches = sorted((PE_ROOT / deployment).glob(f"*{event}*PEDataRelease*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one PE file for {deployment} {event}: {matches}")
    return matches[0]


def maximum_likelihood_sample(path: Path, group: str) -> tuple[int, dict[str, float]]:
    with h5py.File(path, "r") as handle:
        samples = handle[f"{group}/posterior_samples"]
        names = set(samples.dtype.names or ())
        missing = (set(PARAMETER_FIELDS) | {"log_likelihood"}) - names
        if missing:
            raise KeyError(f"{path.name}:{group} missing {sorted(missing)}")
        log_likelihood = np.asarray(samples["log_likelihood"][:], dtype=float)
        index = int(np.nanargmax(log_likelihood))
        row = samples[index]
        parameters = {name: float(row[name]) for name in PARAMETER_FIELDS}
        parameters["log_likelihood"] = float(row["log_likelihood"])
    return index, parameters


def taper(frequency: np.ndarray) -> np.ndarray:
    weight = np.zeros_like(frequency)
    rising = (frequency >= 20.0) & (frequency < 25.0)
    weight[rising] = 0.5 * (1.0 - np.cos(np.pi * (frequency[rising] - 20.0) / 5.0))
    weight[(frequency >= 25.0) & (frequency <= 900.0)] = 1.0
    falling = (frequency > 900.0) & (frequency <= 1024.0)
    weight[falling] = 0.5 * (1.0 + np.cos(np.pi * (frequency[falling] - 900.0) / 124.0))
    return weight


def reconstruct(
    sample: dict[str, float], model: IMRPhenomXPHM
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sample_rate = 2048.0
    duration = 8.0
    size = int(sample_rate * duration)
    frequency = np.fft.rfftfreq(size, d=1.0 / sample_rate)
    active = (frequency >= 20.0) & (frequency <= 1024.0)
    parameters = {
        "M_c": sample["chirp_mass"],
        "eta": sample["symmetric_mass_ratio"],
        "s1_x": sample["spin_1x"],
        "s1_y": sample["spin_1y"],
        "s1_z": sample["spin_1z"],
        "s2_x": sample["spin_2x"],
        "s2_y": sample["spin_2y"],
        "s2_z": sample["spin_2z"],
        "d_L": sample["luminosity_distance"],
        "phase_c": sample["phase"],
        "iota": sample["iota"],
    }
    spectrum = np.zeros(frequency.size, dtype=np.complex128)
    spectrum[active] = np.asarray(model(frequency[active], parameters)["p"])
    spectrum *= taper(frequency)
    strain = np.fft.irfft(spectrum, n=size) * sample_rate
    envelope = np.abs(hilbert(strain))
    peak = int(np.argmax(envelope))
    centre = size // 2
    strain = np.roll(strain, centre - peak)
    envelope = np.roll(envelope, centre - peak)
    time = (np.arange(size) - centre) / sample_rate
    keep = (time >= -0.80) & (time <= 0.08)
    normalization = float(np.max(envelope[keep]))
    return (
        time[keep],
        strain[keep] / max(normalization, 1e-30),
        strain[keep],
        normalization,
    )


def main() -> None:
    style()
    candidates = pd.read_csv(CURRENT / "fig4_selected_seed_candidates.csv")
    audit = pd.read_csv(PE_AUDIT)
    manifests = {
        "gwtc3": pd.read_csv(O3_MANIFEST).set_index("event_name", drop=False),
        "gwtc4": pd.read_csv(O4_MANIFEST).set_index("event_name", drop=False),
    }
    colors = {"i": "#2474A6", "j": "#E9782D"}
    waveforms: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ] = {}
    metadata: list[dict[str, object]] = []

    if OUTPUT_SAMPLES.exists() and OUTPUT_METADATA.exists():
        sample_frame = pd.read_csv(OUTPUT_SAMPLES)
        metadata_frame = pd.read_csv(OUTPUT_METADATA)
        time = sample_frame["time_from_peak_s"].to_numpy(float)
        for record in metadata_frame.itertuples(index=False):
            deployment, event = str(record.deployment), str(record.event)
            normalized = sample_frame[f"{deployment}:{event}:normalized"].to_numpy(float)
            raw = sample_frame[f"{deployment}:{event}:raw_hplus"].to_numpy(float)
            peak = float(record.reconstructed_peak_hplus)
            waveforms[(deployment, event)] = (time, normalized, raw, peak)
    else:
        model = IMRPhenomXPHM(f_ref=20.0)
        for deployment in ("gwtc3", "gwtc4"):
            manifest = manifests[deployment]
            group_column = "sky_map_internal_group" if deployment == "gwtc3" else "sky_map_group"
            events = sorted(
                set(candidates.loc[candidates.deployment == deployment, "event_i"])
                | set(candidates.loc[candidates.deployment == deployment, "event_j"])
            )
            for event in events:
                group = str(manifest.loc[event, group_column])
                path = find_pe_file(deployment, event)
                sample_index, sample = maximum_likelihood_sample(path, group)
                time, normalized, raw, peak = reconstruct(sample, model)
                waveforms[(deployment, event)] = (time, normalized, raw, peak)
                metadata.append(
                    {
                        "deployment": deployment,
                        "event": event,
                        "source_file": path.name,
                        "posterior_group": group,
                        "maximum_likelihood_sample_index": sample_index,
                        "reconstructed_peak_hplus": peak,
                        **sample,
                    }
                )

        sample_frame = pd.DataFrame({"time_from_peak_s": next(iter(waveforms.values()))[0]})
        for (deployment, event), (_, normalized, raw, _) in waveforms.items():
            sample_frame[f"{deployment}:{event}:normalized"] = normalized
            sample_frame[f"{deployment}:{event}:raw_hplus"] = raw
        sample_frame.to_csv(OUTPUT_SAMPLES, index=False, compression="gzip", float_format="%.9e")
        pd.DataFrame(metadata).to_csv(OUTPUT_METADATA, index=False)

    with PdfPages(OUTPUT_PDF) as pdf:
        for deployment, label in (("gwtc3", "GWTC-3 / O3"), ("gwtc4", "GWTC-4.1 / O4a")):
            selected = candidates[candidates.deployment == deployment].sort_values("rank")
            for start in (0, 4, 8):
                page = selected.iloc[start : start + 4]
                nrows = len(page)
                short_page = nrows <= 2
                figure_height = 8.3 if short_page else 2.45 * nrows + 1.8
                fig, axes = plt.subplots(
                    nrows, 2, figsize=(10.0, figure_height), sharex=True, squeeze=False
                )
                fig.subplots_adjust(
                    left=0.08,
                    right=0.985,
                    bottom=0.15 if short_page else 0.10,
                    top=0.81 if short_page else 0.88,
                    hspace=0.88 if short_page else 0.66,
                    wspace=0.16,
                )
                fig.suptitle(
                    f"{label}: Top-10 noise-free PE waveform audit (ranks {start + 1}–{start + nrows})",
                    fontsize=13,
                    fontweight="bold",
                    y=0.972 if short_page else 0.965,
                )
                fig.text(
                    0.5,
                    0.925 if short_page else 0.935,
                    "IMRPhenomXPHM maximum-likelihood reconstructions; shape and relative-amplitude views",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                column_header_y = 0.855 if short_page else 0.902
                fig.text(0.285, column_header_y, "Morphology", ha="center", fontweight="bold", fontsize=10)
                fig.text(0.755, column_header_y, "Relative reconstructed amplitude", ha="center", fontweight="bold", fontsize=10)
                for row_axes, pair in zip(axes, page.itertuples(index=False)):
                    shape_axis, amplitude_axis = row_axes
                    ti, hi, raw_i, peak_i = waveforms[(deployment, pair.event_i)]
                    tj, hj, raw_j, peak_j = waveforms[(deployment, pair.event_j)]
                    pair_peak = max(peak_i, peak_j)
                    shape_axis.plot(ti, hi, color=colors["i"], lw=0.9)
                    shape_axis.plot(tj, hj, color=colors["j"], lw=0.9, alpha=0.92)
                    amplitude_axis.plot(ti, raw_i / pair_peak, color=colors["i"], lw=0.9)
                    amplitude_axis.plot(tj, raw_j / pair_peak, color=colors["j"], lw=0.9, alpha=0.92)
                    pair_audit = audit[
                        (audit.deployment == deployment)
                        & (audit["rank"] == int(pair.rank))
                    ].iloc[0]
                    mu = float(pair_audit.apparent_mu_ratio_i_over_j_median)
                    q05 = float(pair_audit.apparent_mu_ratio_i_over_j_q05)
                    q95 = float(pair_audit.apparent_mu_ratio_i_over_j_q95)
                    shape_axis.set_title(f"Rank {int(pair.rank)}", loc="left", pad=3, fontweight="bold")
                    amplitude_axis.set_title(
                        rf"$Z_{{\rm wf}}={pair.waveform_score:.2f}$; $\Delta t={pair_audit.delay_days:.2f}$ d; "
                        rf"$|\mu_i|/|\mu_j|={mu:.2f}\ [{q05:.2f},{q95:.2f}]$",
                        loc="left", pad=3, fontsize=8.5
                    )
                    shape_axis.text(
                        0.012, 0.88, pair.event_i, color=colors["i"], transform=shape_axis.transAxes,
                        ha="left", va="top", fontsize=8.3,
                    )
                    shape_axis.text(
                        0.012, 0.69, pair.event_j, color=colors["j"], transform=shape_axis.transAxes,
                        ha="left", va="top", fontsize=8.3,
                    )
                    for axis in row_axes:
                        axis.axhline(0.0, color="#BFC7CE", lw=0.55, zorder=0)
                        axis.axvline(0.0, color="#7E8790", lw=0.65, ls="--", zorder=0)
                        axis.set_xlim(-0.80, 0.08)
                        axis.set_ylim(-1.12, 1.12)
                        axis.spines[["top", "right"]].set_visible(False)
                    shape_axis.set_ylabel(r"$h_+/\max|h_+|$", fontweight="bold")
                    amplitude_axis.set_ylabel("Pair-scaled $h_+$", fontweight="bold")
                for axis in axes[-1]:
                    axis.set_xlabel("Time from reconstructed envelope peak (s)", fontweight="bold")
                fig.text(
                    0.5,
                    0.035,
                    "The right column preserves the relative peak amplitude of the two PE reconstructions but omits detector response. "
                    "The distance-derived magnification ratio is descriptive, not a lensing Bayes factor.",
                    ha="center",
                    fontsize=8.5,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(OUTPUT_PDF)
    print(OUTPUT_SAMPLES)
    print(OUTPUT_METADATA)


if __name__ == "__main__":
    main()
