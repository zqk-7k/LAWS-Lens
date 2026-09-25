from __future__ import annotations

import argparse
from pathlib import Path


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def populate(src_dir: Path, dst_dir: Path, files: list[str], copy: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        link_or_copy(src_dir / name, dst_dir / name, copy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrange ET three-arm generated outputs as a match-style data root.")
    parser.add_argument("--generated-root", type=Path, required=True, help="Directory containing SIS_GW_events_ET3, PM_GW_events_ET3 and unlensed_GW_events_ET3 outputs.")
    parser.add_argument("--out-root", type=Path, required=True, help="Destination data root consumed by scripts/08_match_first_train.py.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    args = parser.parse_args()

    lensed_files = {
        "SIS": [
            "SIS_data_strain_1.npy",
            "SIS_data_strain_2.npy",
            "SIS_h_strain_1.npy",
            "SIS_h_strain_2.npy",
            "SIS_time_array_1.npy",
            "SIS_time_array_2.npy",
            "SIS_optimal_SNR_1.npy",
            "SIS_optimal_SNR_2.npy",
            "SIS_optimal_SNR_single_1.npy",
            "SIS_optimal_SNR_single_2.npy",
            "SIS_optimal_SNR_network_1.npy",
            "SIS_optimal_SNR_network_2.npy",
            "source_samples.csv",
            "lensed_source_samples.csv",
            "lens.csv",
            "lens_params.csv",
            "lensed_index.csv",
        ],
        "PM": [
            "PM_data_strain_1.npy",
            "PM_data_strain_2.npy",
            "PM_h_strain_1.npy",
            "PM_h_strain_2.npy",
            "PM_time_array_1.npy",
            "PM_time_array_2.npy",
            "PM_optimal_SNR_1.npy",
            "PM_optimal_SNR_2.npy",
            "PM_optimal_SNR_single_1.npy",
            "PM_optimal_SNR_single_2.npy",
            "PM_optimal_SNR_network_1.npy",
            "PM_optimal_SNR_network_2.npy",
            "source_samples.csv",
            "lensed_source_samples.csv",
            "lens.csv",
            "lens_params.csv",
            "lensed_index.csv",
        ],
    }
    unlensed_files = [
        "unlensed_data_strain.npy",
        "unlensed_h_strain.npy",
        "unlensed_time_array.npy",
        "unlensed_optimal_SNR.npy",
        "unlensed_optimal_SNR_single.npy",
        "unlensed_optimal_SNR_network.npy",
        "source_samples.csv",
    ]

    populate(args.generated_root / "SIS_GW_events_ET3", args.out_root / "SIS_data_0222", lensed_files["SIS"], args.copy)
    populate(args.generated_root / "PM_GW_events_ET3", args.out_root / "PM_data_0222", lensed_files["PM"], args.copy)
    populate(args.generated_root / "unlensed_GW_events_ET3", args.out_root / "Unlensed_data_0222", unlensed_files, args.copy)
    print(f"ET3 match-style root ready: {args.out_root}")


if __name__ == "__main__":
    main()
