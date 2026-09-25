#!/usr/bin/env python3
"""Final integrity audit for the independent GWTC sky-ordering v9.4 rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_HISTORICAL_HASH = (
    "9569f91889170d8ce0bc477d789ea2b1f4dac7311a6497094ac877b3dcde1763"
)
DEPLOYMENTS = ("gwtc3", "gwtc4")
EXPECTED_EVENTS = {"gwtc3": 63, "gwtc4": 84}
EXPECTED_FULL_PAIRS = {"gwtc3": 1953, "gwtc4": 3486}
EXPECTED_STRICT_EVENTS = {"gwtc3": 53, "gwtc4": 74}
EXPECTED_STRICT_PAIRS = {"gwtc3": 1378, "gwtc4": 2701}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--historical-package", type=Path, required=True)
    args = parser.parse_args()

    root = args.result_root.resolve()
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    contract = json.loads(
        (root / "analysis_contract_ordering_corrected_v94.json").read_text(
            encoding="utf-8"
        )
    )
    historical_hash = sha256_file(args.historical_package)
    require(
        historical_hash == EXPECTED_HISTORICAL_HASH,
        "historical v9.3 package remains byte-identical",
        checks,
    )
    require(
        contract["historical_v93_package_sha256_before_rerun"]
        == contract["historical_v93_package_sha256_after_rerun"]
        == historical_hash,
        "contract before/after historical hashes match current package",
        checks,
    )

    frozen = pd.read_csv(root / "frozen_pair_feature_audit_v94.csv")
    require(len(frozen) == 12, "all 12 deployment/seed/split audits exist", checks)
    require(bool(frozen["exact_match"].all()), "all frozen pair features match exactly", checks)
    require(
        float(frozen["max_abs_numeric_delta"].max()) == 0.0,
        "frozen pair-feature maximum numeric delta is zero",
        checks,
    )

    source_total = 0
    injection_audit_rows = 0
    deployment_details: dict[str, Any] = {}
    for deployment in DEPLOYMENTS:
        source = pd.read_csv(root / deployment / "real_map_source_audit_v93.csv")
        source_total += len(source)
        require(
            len(source) == EXPECTED_EVENTS[deployment],
            f"{deployment}: expected number of real PE maps",
            checks,
        )
        require(
            set(source["source_ordering"].astype(str)) == {"NESTED"},
            f"{deployment}: every source map is recorded as NESTED",
            checks,
        )
        require(
            set(source["output_ordering"].astype(str)) == {"RING"},
            f"{deployment}: every scoring map is RING",
            checks,
        )
        require(
            bool(source["ordering_conversion_applied"].all()),
            f"{deployment}: NESTED-to-RING conversion applied to every map",
            checks,
        )
        require(
            bool(source["ordering_metadata_raw"].str.contains("True", regex=False).all()),
            f"{deployment}: raw HDF5 ordering metadata is True",
            checks,
        )

        map_audit = pd.read_csv(
            root / "corrected_sky_inputs" / deployment / "injection_map_rebuild_audit_v94.csv"
        )
        injection_audit_rows += len(map_audit)
        require(len(map_audit) == 6, f"{deployment}: six rebuilt map banks exist", checks)
        require(
            bool(np.isclose(map_audit["used_template_selection_match_fraction"], 1.0).all()),
            f"{deployment}: frozen template identities match exactly",
            checks,
        )
        require(
            float(map_audit["max_abs_normalization_error"].max()) <= 2e-6,
            f"{deployment}: rebuilt map normalization is valid",
            checks,
        )
        hashes_ok = True
        for row in map_audit.itertuples(index=False):
            hashes_ok &= sha256_file(Path(row.corrected_map_path)) == row.map_sha256
            hashes_ok &= sha256_file(Path(row.corrected_event_path)) == row.event_sha256
        require(hashes_ok, f"{deployment}: rebuilt map/event hashes verify", checks)

        reproduction = pd.read_csv(
            root / deployment / "frozen_waveform_time_reproduction_v93.csv"
        )
        require(
            bool(reproduction["exact_match"].all())
            and float(reproduction["delta"].abs().max()) == 0.0,
            f"{deployment}: waveform-only and time-only reproduce v9.3 exactly",
            checks,
        )

        gate = pd.read_csv(root / deployment / "waveform_gate_per_seed_v81.csv")
        require(
            len(gate) == 3 and bool(gate["passed_for_primary_deployment"].all()),
            f"{deployment}: all three waveform gates pass",
            checks,
        )

        convergence = json.loads(
            (root / deployment / "real_sky_resolution_convergence_summary_v93.json").read_text()
        )
        require(
            convergence["n_events"] == EXPECTED_EVENTS[deployment]
            and convergence["n_unordered_pairs"] == EXPECTED_FULL_PAIRS[deployment],
            f"{deployment}: real catalog event/pair counts are complete",
            checks,
        )
        require(
            convergence["high_resolution_sign_unstable"] == 0,
            f"{deployment}: no 256/512/1024 sign-unstable real pair",
            checks,
        )

        consensus = pd.read_parquet(
            root / deployment / "real_candidate_consensus_with_pe_v93.parquet"
        )
        require(
            len(consensus) == EXPECTED_STRICT_PAIRS[deployment]
            and consensus["pair_key"].nunique() == EXPECTED_STRICT_PAIRS[deployment]
            and len(set(consensus["event_i"]) | set(consensus["event_j"]))
            == EXPECTED_STRICT_EVENTS[deployment],
            f"{deployment}: complete strict H1-L1 BBH real-pair consensus table",
            checks,
        )
        require(
            np.isfinite(consensus["consensus_rank"].to_numpy(dtype=float)).all()
            and consensus["consensus_rank"].nunique() == EXPECTED_STRICT_PAIRS[deployment],
            f"{deployment}: consensus ranks are finite and unique",
            checks,
        )
        require(
            np.isfinite(consensus["final_score_mean"].to_numpy(dtype=float)).all(),
            f"{deployment}: consensus scores are finite",
            checks,
        )

        deployment_details[deployment] = {
            "n_events": len(source),
            "n_full_catalog_pairs": EXPECTED_FULL_PAIRS[deployment],
            "n_strict_h1l1_bbh_events": EXPECTED_STRICT_EVENTS[deployment],
            "n_strict_h1l1_bbh_pairs": len(consensus),
            "n_waveform_gates_passed": int(gate["passed_for_primary_deployment"].sum()),
            "max_abs_zsky_512_1024": convergence["abs_delta_nside512_1024"]["max"],
        }

    require(source_total == 147, "all 147 real PE maps audited", checks)
    require(injection_audit_rows == 12, "all 12 injection map banks audited", checks)

    key_metrics = pd.read_csv(root / "ordering_corrected_key_metrics_v94.csv")
    numeric = key_metrics.select_dtypes(include=[np.number])
    require(np.isfinite(numeric.to_numpy()).all(), "key metric table contains no NaN/Inf", checks)

    required = [
        "gwtc_sky_ordering_corrected_v94_report_cn.md",
        "gwtc_sky_ordering_corrected_v94_report_en.md",
        "ordering_corrected_summary_v94.json",
        "ordering_corrected_real_top10_with_pe_v94.csv",
        "v93_vs_v94_candidate_rank_comparison.parquet",
        "v93_vs_v94_real_sky_all_pairs.parquet",
        "figures/fig_gwtc_sky_ordering_corrected_v94.pdf",
        "figures/fig_gwtc_sky_ordering_corrected_v94.png",
    ]
    require(all((root / item).is_file() for item in required), "all required reports/tables/figures exist", checks)

    details["historical_package_sha256"] = historical_hash
    details["source_maps_audited"] = source_total
    details["injection_map_banks_audited"] = injection_audit_rows
    details["deployments"] = deployment_details
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    payload = {
        "status": status,
        "result_root": str(root),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "details": details,
    }
    output = root / "final_integrity_audit_v94.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_ready) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=json_ready))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
