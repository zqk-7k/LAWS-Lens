#!/usr/bin/env python3
"""Create a compact, verified deliverable for the completed rapid-sky pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/gw-catalog")
SCRIPT_NAMES = (
    "03_v10p1_homogeneous_pe_pilot.py",
    "09_v10p2_rapid_coherent_sky_pe.py",
    "11_run_v10p2_rapid_coherent_sky_pe.sh",
    "12_package_v10p2_rapid_sky_pe.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(output: Path) -> list[Path]:
    files: set[Path] = set()
    for name in ("contracts", "manifests", "summaries", "figures", "logs"):
        root = output / name
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file())
    for name in ("RAPID_COHERENT_SKY_PE_REPORT_CN.md", "STATUS.json"):
        path = output / name
        if path.exists():
            files.add(path)
    for case_dir in (output / "cases").glob("*"):
        for relative in (
            "pe_metrics.json",
            "sky_map_metrics.json",
            "posterior_samples.hdf5",
            "sky_map/skymap.fits.gz",
            "sky_map/from_samples.log",
            "sky_map/flatten_nside256.log",
            "sky_map/flatten_nside512.log",
            "sky_map/flatten_nside1024.log",
        ):
            path = case_dir / relative
            if path.exists():
                files.add(path)
    script_root = PROJECT / "scripts/experiments"
    files.update(script_root / name for name in SCRIPT_NAMES if (script_root / name).exists())
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=PROJECT / "packages")
    args = parser.parse_args()
    output = args.output.resolve()
    status_path = output / "STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status.get("completed_cases", 0)) != 12:
        raise RuntimeError(f"Refusing partial package: {status}")
    files = selected_files(output)
    manifest_path = output / "DELIVERABLES_MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_path", "bytes", "sha256"))
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "source_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    files.append(manifest_path)
    args.package_dir.mkdir(parents=True, exist_ok=True)
    package = args.package_dir / f"{output.name}_deliverables.tar.gz"
    with tarfile.open(package, "w:gz", compresslevel=6) as archive:
        for path in sorted(files):
            if path.is_relative_to(output):
                relative = path.relative_to(output)
                archive_name = Path(output.name) / relative
            else:
                archive_name = Path(output.name) / "scripts" / path.name
            archive.add(path, arcname=str(archive_name), recursive=False)
    digest = sha256(package)
    checksum = package.with_suffix(package.suffix + ".sha256")
    checksum.write_text(f"{digest}  {package.name}\n", encoding="ascii")
    with tarfile.open(package, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError("Empty package")
    print(json.dumps({"package": str(package), "sha256": digest, "members": len(members)}, indent=2))


if __name__ == "__main__":
    main()
