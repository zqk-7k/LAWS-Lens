from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import jax
import numpy as np
import pandas as pd
from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import IMRPhenomXPHM
from scipy.signal import hilbert


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "source_data" / "current_v93"

EVENTS = {
    "GW170809_082821": {
        "filename": "GW170809_082821.h5",
        "group": "C01:IMRPhenomXPHM",
        "record_url": "https://zenodo.org/records/6513631",
    },
    "GW170814_103043": {
        "filename": "GW170814_103043.h5",
        "group": "C01:IMRPhenomXPHM",
        "record_url": "https://zenodo.org/records/6513631",
    },
    "GW230707_124047": {
        "filename": "GW230707_124047.hdf5",
        "group": "C00:IMRPhenomXPHM-SpinTaylor",
        "record_url": "https://zenodo.org/records/20275769",
    },
    "GW230709_122727": {
        "filename": "GW230709_122727.hdf5",
        "group": "C00:IMRPhenomXPHM-SpinTaylor",
        "record_url": "https://zenodo.org/records/20275769",
    },
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build peak-aligned, noise-free waveform reconstructions from "
            "public GWTC parameter-estimation samples."
        )
    )
    parser.add_argument(
        "--pe-root",
        type=Path,
        required=True,
        help="Directory containing the four public PE HDF5 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the compact CSV and JSON provenance files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_likelihood_sample(path: Path, group: str) -> tuple[int, dict[str, float]]:
    with h5py.File(path, "r") as handle:
        samples = handle[f"{group}/posterior_samples"]
        fields = set(samples.dtype.names or ())
        required = set(PARAMETER_FIELDS) | {"log_likelihood"}
        missing = sorted(required - fields)
        if missing:
            raise KeyError(f"{path.name}:{group} lacks fields {missing}")
        log_likelihood = samples["log_likelihood"][:]
        finite = np.isfinite(log_likelihood)
        if not finite.any():
            raise ValueError(f"{path.name}:{group} has no finite log likelihood")
        finite_indices = np.flatnonzero(finite)
        sample_index = int(finite_indices[np.argmax(log_likelihood[finite])])
        row = samples[sample_index]
        values = {name: float(row[name]) for name in PARAMETER_FIELDS}
        values["log_likelihood"] = float(row["log_likelihood"])
    return sample_index, values


def cosine_taper(frequency: np.ndarray) -> np.ndarray:
    taper = np.zeros_like(frequency)
    rising = (frequency >= 20.0) & (frequency < 25.0)
    taper[rising] = 0.5 * (1.0 - np.cos(np.pi * (frequency[rising] - 20.0) / 5.0))
    taper[(frequency >= 25.0) & (frequency <= 900.0)] = 1.0
    falling = (frequency > 900.0) & (frequency <= 1024.0)
    taper[falling] = 0.5 * (1.0 + np.cos(np.pi * (frequency[falling] - 900.0) / 124.0))
    return taper


def reconstruct(sample: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = 2048.0
    duration = 8.0
    n_samples = int(sample_rate * duration)
    frequency = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    active = (frequency >= 20.0) & (frequency <= 1024.0)
    model = IMRPhenomXPHM(f_ref=20.0)
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
    polarizations = model(frequency[active], parameters)
    spectrum[active] = np.asarray(polarizations["p"], dtype=np.complex128)
    spectrum *= cosine_taper(frequency)

    strain = np.fft.irfft(spectrum, n=n_samples) * sample_rate
    envelope = np.abs(hilbert(strain))
    peak_index = int(np.argmax(envelope))
    centre_index = n_samples // 2
    strain = np.roll(strain, centre_index - peak_index)
    envelope = np.roll(envelope, centre_index - peak_index)

    relative_time = (np.arange(n_samples) - centre_index) / sample_rate
    display = (relative_time >= -0.90) & (relative_time <= 0.10)
    normalization = float(np.max(envelope[display]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Waveform normalization is not finite and positive")
    return relative_time[display], strain[display] / normalization


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    waveforms: dict[str, np.ndarray] = {}
    time: np.ndarray | None = None
    metadata: dict[str, object] = {
        "description": (
            "Noise-free plus-polarization waveform reconstructions generated "
            "from the maximum-likelihood sample in each public PE posterior."
        ),
        "waveform_model": "rippleGW 0.3.0 IMRPhenomXPHM",
        "sample_selection": "maximum finite log_likelihood within the named PE group",
        "frequency_range_hz": [20.0, 1024.0],
        "low_frequency_taper_hz": [20.0, 25.0],
        "high_frequency_taper_hz": [900.0, 1024.0],
        "sample_rate_hz": 2048.0,
        "generation_duration_s": 8.0,
        "display_window_s": [-0.90, 0.10],
        "alignment": "maximum analytic-signal envelope placed at t=0",
        "normalization": "each event divided by its own peak analytic-signal envelope",
        "interpretation_limit": (
            "Model-dependent PE reconstruction; not a direct, uniquely denoised "
            "detector observation and not used by ranking or PE screening."
        ),
        "events": {},
    }

    for event, specification in EVENTS.items():
        path = args.pe_root / specification["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        sample_index, sample = maximum_likelihood_sample(path, specification["group"])
        event_time, strain = reconstruct(sample)
        if time is None:
            time = event_time
        elif not np.array_equal(time, event_time):
            raise RuntimeError("Waveform time grids are inconsistent")
        waveforms[event] = strain
        metadata["events"][event] = {
            "source_file": path.name,
            "source_sha256": sha256(path),
            "source_record": specification["record_url"],
            "posterior_group": specification["group"],
            "sample_index": sample_index,
            "parameters": sample,
        }

    if time is None:
        raise RuntimeError("No waveforms were generated")
    frame = pd.DataFrame({"time_from_peak_s": time, **waveforms})
    frame.to_csv(
        args.output_root / "reconstructed_waveform_inputs.csv",
        index=False,
        float_format="%.10e",
    )
    with (args.output_root / "reconstructed_waveform_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
