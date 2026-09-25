#!/usr/bin/env python3
"""Verify and package a G-1 HOLD run without touching historical results."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_JCS = "ead281ad1bafe9430cab45ec834338d58b867cb471ab42f9402390815b080900"
EXPECTED_STATUS = "HOLD_FOR_AUTHOR_SIGNATURE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--packages-root", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_root.resolve()
    packages = args.packages_root.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    packages.mkdir(parents=True, exist_ok=True)
    package = packages / f"{run.name}_G_MINUS1_author_review.tar.gz"
    package_hash_path = package.with_suffix(package.suffix + ".sha256")
    if package.exists() or package_hash_path.exists():
        raise FileExistsError("Refusing to overwrite an existing package")

    status = json.loads((run / "STATUS.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "provenance/G_MINUS1_CONFIG_AUDIT.json").read_text(encoding="utf-8"))
    signed_audit_path = run / "provenance/G_MINUS1_SIGNED_CONFIG_VALIDATION.json"
    signed_audit = json.loads(signed_audit_path.read_text(encoding="utf-8"))
    current_v93_contract = Path(audit["read_only_baseline_presence"]["analysis_contract_path"])
    current_v93_manifest = Path(audit["read_only_baseline_presence"]["checksum_manifest_path"])

    template_dir = run / "contracts/unsigned_templates"
    payloads = sorted(template_dir.glob("*.json"))
    payloads = [path for path in payloads if not path.name.endswith(".approval.json")]
    approvals = sorted(template_dir.glob("*.approval.json"))
    schemas = sorted((run / "contracts/schemas").glob("*.schema.json"))
    prohibited_outputs = [
        run / "G0_SOURCE_NOISE_INVENTORY.json",
        run / "G05_POWER_SUPPORT_RECOMMENDATION.json",
        run / "injections/o3/sky_event_manifest_v10.parquet",
        run / "injections/o4a/sky_event_manifest_v10.parquet",
        run / "real_catalog/descriptive_only/real_pair_scores.parquet",
    ]
    checks = {
        "status_is_hold": status.get("status") == EXPECTED_STATUS,
        "hold_sentinel_exists": (run / EXPECTED_STATUS).is_file(),
        "g_minus1_not_passed": audit.get("g_minus1_pass") is False,
        "four_initial_signed_configs_found_is_zero": len(audit.get("signed_initial_author_configs_found", [])) == 0,
        "unsigned_validator_holds": signed_audit.get("status") == EXPECTED_STATUS,
        "five_unsigned_payloads": len(payloads) == 5,
        "five_unsigned_sidecars": len(approvals) == 5 and all(json.loads(p.read_text())["approval_status"] == "UNSIGNED" for p in approvals),
        "six_schemas": len(schemas) == 6,
        "jcs_vector_matches": (run / "contracts/canonicalization/JCS_TEST_VECTOR.sha256").read_text().strip() == EXPECTED_JCS,
        "v93_contract_unchanged": current_v93_contract.is_file() and sha256_file(current_v93_contract) == audit["read_only_baseline_presence"]["analysis_contract_sha256"],
        "v93_checksum_manifest_unchanged": current_v93_manifest.is_file() and sha256_file(current_v93_manifest) == audit["read_only_baseline_presence"]["checksum_manifest_sha256"],
        "no_g0_or_science_outputs": not any(path.exists() for path in prohibited_outputs),
        "no_figures_generated": not any((run / "figures").iterdir()),
        "no_score_files_generated": not any(path.is_file() for path in (run / "scores").rglob("*")),
    }
    passed = all(checks.values())
    verification = {
        "schema": "sky-background-v10-gminus1-final-verification-v1",
        "run_id": run.name,
        "verified_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": EXPECTED_STATUS if passed else "FAILED_OR_INCOMPLETE",
        "checks": checks,
        "next_author_action": "SIGN_INITIAL_FOUR_CONFIGS_FOR_G0_AND_G05_ONLY" if passed else "REVIEW_FAILED_CHECKS",
        "science_computation_performed": False,
    }
    write_json(run / "provenance/FINAL_GMINUS1_VERIFICATION.json", verification)
    if not passed:
        raise RuntimeError(f"G-1 final verification failed: {checks}")

    (run / "reports/AUTHOR_REVIEW_SUMMARY_CN.md").write_text(
        "# 作者审核摘要\n\n"
        f"- Run：`{run.name}`\n"
        f"- 状态：`{EXPECTED_STATUS}`\n"
        "- 本轮完成：协议归档、Schema、未签名模板、detached sidecar、JCS 测试、签名缺失审计。\n"
        "- 本轮未完成且未授权：G0 inventory、G0.5 power/support、注入、PE、校准、holdout 和真实目录重排。\n"
        "- v9.3 权威合同与 checksum manifest 在执行前后哈希一致。\n"
        "- 该包没有新的召回率、假对负担或候选排名，不能称为方法实验结果。\n\n"
        "作者需先填写并签署四份初始配置，且 `approved_for` 只能为 `G0_AND_G05_ONLY`。\n",
        encoding="utf-8",
    )

    # Generate a complete manifest after every report and validation artifact exists.
    checksum_path = run / "provenance/SHA256SUMS"
    files_before_manifest = sorted(
        path for path in run.rglob("*")
        if path.is_file() and path != checksum_path and "package" not in path.parts
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(run)}\n" for path in files_before_manifest),
        encoding="utf-8",
    )

    with tarfile.open(package, "w:gz", compresslevel=6) as archive:
        archive.add(run, arcname=run.name, recursive=True)

    package_hash = sha256_file(package)
    package_hash_path.write_text(f"{package_hash}  {package.name}\n", encoding="ascii")

    # Verify every checksum-listed file from inside the compressed artifact.
    checksum_rows = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_rows.append((digest, relative))
    with tarfile.open(package, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        failures = []
        for digest, relative in checksum_rows:
            name = f"{run.name}/{relative}"
            member = members.get(name)
            if member is None:
                failures.append(f"missing:{relative}")
                continue
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                failures.append(f"hash:{relative}")
        if failures:
            raise RuntimeError(f"Package internal verification failed: {failures[:10]}")

    receipt = {
        "schema": "sky-background-v10-gminus1-package-receipt-v1",
        "run_id": run.name,
        "package": str(package),
        "package_bytes": package.stat().st_size,
        "package_sha256": package_hash,
        "checksum_entries_verified_inside_package": len(checksum_rows),
        "internal_verification_failures": 0,
        "status": EXPECTED_STATUS,
    }
    write_json(run / "package/G_MINUS1_PACKAGE_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
