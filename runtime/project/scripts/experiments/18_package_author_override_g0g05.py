#!/usr/bin/env python3
"""Verify and package a G0/G0.5 author-override feasibility run."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--packages-root", type=Path, required=True)
    parser.add_argument("--gminus1-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    packages = args.packages_root.resolve()
    gminus1 = args.gminus1_root.resolve()

    status = json.loads((run / "STATUS.json").read_text(encoding="utf-8"))
    g0 = json.loads((run / "G0_SOURCE_NOISE_INVENTORY.json").read_text(encoding="utf-8"))
    g05 = json.loads((run / "G05_POWER_SUPPORT_RECOMMENDATION.json").read_text(encoding="utf-8"))
    old_verify = json.loads((gminus1 / "provenance/FINAL_GMINUS1_VERIFICATION.json").read_text(encoding="utf-8"))

    checks = {
        "status_is_hold_no_go": status.get("status") == "HOLD_NO_GO_OR_REDESIGN",
        "author_override_recorded": (run / "contracts/AUTHOR_DIRECT_EXECUTION_OVERRIDE.json").exists(),
        "g0_inventory_complete": g0.get("status") == "G0_INVENTORY_COMPLETE",
        "g05_is_no_go": g05.get("status") == "NO_GO_OR_REDESIGN",
        "scientific_gate_not_waived": g05.get("scientific_gate_waived") is False,
        "no_real_reranking": status.get("real_catalog_reranked") is False,
        "no_v93_modification_claimed": status.get("v93_modified") is False,
        "gminus1_v93_inputs_were_unchanged": all(
            old_verify.get("checks", {}).get(key) is True
            for key in ("v93_contract_unchanged", "v93_checksum_manifest_unchanged")
        ),
        "required_tables_exist": all(
            (run / rel).exists()
            for rel in (
                "tables/G0_OFFSOURCE_NOISE_BLOCK_INVENTORY.csv",
                "tables/G05_FINITE_SAMPLE_SUPPORT.csv",
                "tables/V93_HISTORICAL_BASELINE_CONTEXT.csv",
            )
        ),
        "required_figures_exist": all(
            (run / rel).exists()
            for rel in ("figures/fig_g05_feasibility.pdf", "figures/fig_g05_feasibility.png")
        ),
        "no_new_pair_score_files": not any((run / "scores").rglob("*.parquet")),
        "no_new_injection_arrays": not any((run / "injections").rglob("*.npy")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Final verification failed: {checks}")

    verification = {
        "schema": "sky-background-v10-g0g05-final-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run.name,
        "status": status["status"],
        "checks": checks,
    }
    write_json(run / "provenance/FINAL_VERIFICATION.json", verification)

    summary = f"""# 作者审阅摘要

本 run 根据作者 2026-08-22 的直接指令跳过行政签署步骤，完成了协议 G0 与
G0.5。科学 Gate 未被跳过，最终状态为 **{status['status']}**。

- G0 source/noise inventory：完成；
- G0.5 prospective power/support：`{g05['status']}`；
- G1--G5：未启动；
- 真实目录：未重排；
- v9.3 与论文：未修改。

主要硬约束为：缺少真实/注入同质 PE 管线；O3/O4a 的独立 null audit pairs
分别少于每层 3,242 的预注册需求；dense Nside=1024 最低存储投影超过当前
空闲空间。详细数字见 `G05_POWER_SUPPORT_RECOMMENDATION.json`。
"""
    (run / "AUTHOR_REVIEW_SUMMARY_CN.md").write_text(summary, encoding="utf-8")

    excluded = {"provenance/SHA256SUMS"}
    files = sorted(p for p in run.rglob("*") if p.is_file() and str(p.relative_to(run)) not in excluded)
    checksum_lines = [f"{sha256(path)}  {path.relative_to(run).as_posix()}" for path in files]
    # Paths may contain the Chinese protocol filename, so the checksum
    # manifest itself must be UTF-8 even though every digest is ASCII.
    (run / "provenance/SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    packages.mkdir(parents=True, exist_ok=True)
    package = packages / f"{run.name}_G0_G05_NO_GO_author_review.tar.gz"
    if package.exists() or package.with_suffix(package.suffix + ".sha256").exists():
        raise FileExistsError(f"Refusing to overwrite package: {package}")
    with tarfile.open(package, "w:gz") as archive:
        archive.add(run, arcname=run.name)
    package_hash = sha256(package)
    hash_path = Path(str(package) + ".sha256")
    hash_path.write_text(f"{package_hash}  {package.name}\n", encoding="ascii")

    failures = []
    with tarfile.open(package, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for line in checksum_lines:
            expected, rel = line.split("  ", 1)
            name = f"{run.name}/{rel}"
            member = members.get(name)
            if member is None:
                failures.append(f"missing:{rel}")
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            actual = hashlib.sha256(handle.read()).hexdigest()
            if actual != expected:
                failures.append(f"hash:{rel}")
    if failures:
        raise RuntimeError(f"Package verification failed: {failures[:10]}")

    receipt = {
        "schema": "sky-background-v10-g0g05-package-receipt-v1",
        "run_id": run.name,
        "package": str(package),
        "package_bytes": package.stat().st_size,
        "package_sha256": package_hash,
        "checksum_entries_verified_inside_package": len(checksum_lines),
        "internal_verification_failures": 0,
        "status": status["status"],
    }
    write_json(run / "package/G0_G05_PACKAGE_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
