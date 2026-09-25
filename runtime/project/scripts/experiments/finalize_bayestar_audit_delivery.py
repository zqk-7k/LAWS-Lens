#!/usr/bin/env python3
"""Refresh manifests and the compact delivery after the Nside=1024 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


INCLUDE_DIRS = ("contracts", "results", "reports", "figures", "scripts")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--map-package", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    package = args.package.resolve()
    map_package = args.map_package.resolve()

    detail = pd.read_csv(root / "results/bayestar_injection_targeted_nside1024_audit_pairs.csv")
    real = pd.read_csv(root / "results/real_candidate_head_sky_resolution_summary.csv")
    global_summary = {
        "injection_targeted_pair_count": int(len(detail)),
        "selection": "all true companions + top-30 sky non-companions + 30 hash-selected non-companions per deployment/seed/split",
        "sign_flip_count_512_1024": int(detail.sign_flip_512_1024.sum()),
        "median_abs_delta_512_1024": float(detail.abs_delta_512_1024.median()),
        "q90_abs_delta_512_1024": float(detail.abs_delta_512_1024.quantile(0.90)),
        "q99_abs_delta_512_1024": float(detail.abs_delta_512_1024.quantile(0.99)),
        "q999_abs_delta_512_1024": float(detail.abs_delta_512_1024.quantile(0.999)),
        "max_abs_delta_512_1024": float(detail.abs_delta_512_1024.max()),
        "max_abs_stored_float32_minus_float64_512": float(
            detail.abs_delta_stored_float32_vs_float64_512.max()
        ),
        "real_top50_sign_flip_count_512_1024": int(real.sign_flip_512_1024_count.sum()),
        "real_top50_max_abs_delta_512_1024_by_deployment": {
            row.deployment: float(row.max_abs_delta_512_1024) for row in real.itertuples()
        },
        "decision_relevant_convergence_passed": bool(
            not detail.sign_flip_512_1024.any()
            and float(detail.abs_delta_512_1024.max()) < 0.001
            and int(real.sign_flip_512_1024_count.sum()) == 0
        ),
    }
    (root / "results/targeted_nside1024_global_summary.json").write_text(
        json.dumps(global_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    status_path = root / "contracts/FINAL_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
            "targeted_nside1024_audit_complete": True,
            "targeted_nside1024_audit_pairs": int(len(detail)),
            "targeted_nside1024_sign_flips": int(detail.sign_flip_512_1024.sum()),
            "decision_relevant_convergence_passed": global_summary[
                "decision_relevant_convergence_passed"
            ],
            "final_verification_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
    )
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = []
    for dirname in INCLUDE_DIRS:
        files.extend(path for path in (root / dirname).rglob("*") if path.is_file())
    files = sorted(set(files))
    rows = [(str(path.relative_to(root)), path.stat().st_size, sha256(path)) for path in files]
    manifest_dir = root / "manifest"
    manifest_dir.mkdir(exist_ok=True)
    with (manifest_dir / "OUTPUT_SHA256_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(rows)
    (manifest_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {path}\n" for path, _, digest in rows), encoding="utf-8"
    )

    package.parent.mkdir(parents=True, exist_ok=True)
    temp = package.with_suffix(package.suffix + ".tmp")
    with tarfile.open(temp, "w:gz") as archive:
        for dirname in (*INCLUDE_DIRS, "manifest"):
            archive.add(root / dirname, arcname=f"{root.name}/{dirname}")
    temp.replace(package)
    package_hash = sha256(package)
    package.with_suffix(package.suffix + ".sha256").write_text(
        f"{package_hash}  {package.name}\n", encoding="utf-8"
    )
    if not map_package.exists():
        raise FileNotFoundError(map_package)
    map_hash = sha256(map_package)
    map_package.with_suffix(map_package.suffix + ".sha256").write_text(
        f"{map_hash}  {map_package.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "compact_package": str(package),
        "compact_sha256": package_hash,
        "compact_bytes": package.stat().st_size,
        "map_package": str(map_package),
        "map_sha256": map_hash,
        "map_bytes": map_package.stat().st_size,
        **global_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
