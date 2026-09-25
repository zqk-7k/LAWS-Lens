from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def infer_shape(path: Path) -> tuple[int, ...]:
    return tuple(np.load(path, mmap_mode="r").shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write generation_metadata.json for generated waveform directories.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--ifos", required=True, help="Comma-separated interferometer names, e.g. H1,L1")
    parser.add_argument("--family", default=None, help="SIS, PM, or unlensed. If omitted, inferred from files.")
    parser.add_argument("--waveform-approximant", default="IMRPhenomXPHM")
    parser.add_argument("--sampling-frequency", type=int, default=4096)
    parser.add_argument("--duration", type=int, default=24)
    parser.add_argument("--minimum-frequency", type=float, default=20.0)
    args = parser.parse_args()

    ifos = [x.strip() for x in args.ifos.split(",") if x.strip()]
    family = args.family
    candidates = []
    if family:
        prefix = family.upper()
        candidates.append(args.root / f"{prefix}_data_strain_1.npy")
    candidates.append(args.root / "unlensed_data_strain.npy")
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        matches = sorted(args.root.glob("*_data_strain_1.npy"))
        data_path = matches[0] if matches else None
    if data_path is None:
        raise FileNotFoundError(f"no data strain npy found under {args.root}")

    shape = infer_shape(data_path)
    metadata = {
        "scenario": args.scenario,
        "ifos": ifos,
        "n_ifos": len(ifos),
        "data_shape": list(shape),
        "snr_single_file": None,
        "snr_network_file": None,
        "waveform_approximant": args.waveform_approximant,
        "sampling_frequency": args.sampling_frequency,
        "duration": args.duration,
        "minimum_frequency": args.minimum_frequency,
    }
    if family and family.lower() != "unlensed":
        prefix = family.upper()
        metadata["snr_single_file"] = f"{prefix}_optimal_SNR_single_1.npy"
        metadata["snr_network_file"] = f"{prefix}_optimal_SNR_network_1.npy"
    elif (args.root / "unlensed_optimal_SNR_single.npy").exists():
        metadata["snr_single_file"] = "unlensed_optimal_SNR_single.npy"
        metadata["snr_network_file"] = "unlensed_optimal_SNR_network.npy"

    out = args.root / "generation_metadata.json"
    out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
