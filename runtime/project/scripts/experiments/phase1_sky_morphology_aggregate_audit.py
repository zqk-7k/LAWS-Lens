#!/usr/bin/env python3
"""Aggregate Phase-1 map and template audits for the sky morphology experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEPLOYMENTS = ("gwtc3", "gwtc4")
SEEDS = (202607241, 202607242, 202607243)
SPLITS = (("validation", "val"), ("test", "test"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            source = (
                args.project_root
                / "results/real_noise_injection_v7_peak2s_formal_20260722"
                / deployment
                / f"seed_{seed}"
                / "results"
            )
            destination = args.output_root / "results" / deployment / f"seed_{seed}"
            for split, short in SPLITS:
                assignment_path = destination / f"synthetic_{split}_template_assignment.csv"
                event_path = source / f"mixed_{short}_synthetic_events_v7.parquet"
                pair_path = destination / f"synthetic_{split}_pair_morphology_nside512.parquet"
                numerical_path = destination / f"synthetic_{split}_float64_feature_audit.json"

                assignment = pd.read_csv(assignment_path).sort_values("idx")
                events = pd.read_parquet(
                    event_path,
                    columns=["idx", "family", "tag", "source_index"],
                ).sort_values("idx")
                pairs = pd.read_parquet(pair_path)
                numerical = json.loads(numerical_path.read_text())

                merged = events.merge(
                    assignment[["idx", "template_index"]], on="idx", validate="one_to_one"
                )
                is_unlensed = (
                    merged["family"].astype(str).str.lower().eq("unlensed")
                    | merged["tag"].astype(str).str.upper().eq("U")
                )
                lensed = merged.loc[~is_unlensed].copy()
                lensed["system_id"] = (
                    lensed["family"].astype(str) + ":" + lensed["source_index"].astype(str)
                )
                reuse = lensed.groupby("system_id")["template_index"].agg(
                    n_events="size", n_unique="nunique"
                )
                reused_systems = int((reuse["n_events"] != reuse["n_unique"]).sum())
                run_mismatch = int((~assignment["template_run_matched"].astype(bool)).sum())
                map_sum_max_error = float(np.max(np.abs(assignment["map_sum"].to_numpy(float) - 1.0)))
                nonfinite_maps = int((assignment["finite_fraction"].to_numpy(float) != 1.0).sum())
                expected_pairs = len(events) * (len(events) - 1) // 2
                passed = bool(
                    numerical.get("passed")
                    and reused_systems == 0
                    and run_mismatch == 0
                    and map_sum_max_error <= 2e-7
                    and nonfinite_maps == 0
                    and len(pairs) == expected_pairs
                )
                row = {
                    "deployment": deployment,
                    "seed": seed,
                    "split": split,
                    "n_events": len(events),
                    "n_pairs": len(pairs),
                    "expected_pairs": expected_pairs,
                    "n_lensed_systems": len(reuse),
                    "reused_template_systems": reused_systems,
                    "run_mismatch_events": run_mismatch,
                    "map_sum_max_abs_error": map_sum_max_error,
                    "nonfinite_maps": nonfinite_maps,
                    "raw_log_bf_max_abs_error": numerical["raw_log_bf_max_abs_error"],
                    "bc_max_abs_error": numerical["bc_max_abs_error"],
                    "float64_audit_tolerance": numerical["tolerance"],
                    "passed": passed,
                }
                rows.append(row)
                if not passed:
                    failures.append(f"{deployment}/{seed}/{split}")

    frame = pd.DataFrame(rows)
    csv_path = args.output_root / "results/phase1_aggregate_audit.csv"
    json_path = args.output_root / "results/phase1_aggregate_audit.json"
    frame.to_csv(csv_path, index=False)
    summary = {
        "passed": not failures,
        "n_banks": len(frame),
        "failed_banks": failures,
        "max_map_sum_abs_error": float(frame["map_sum_max_abs_error"].max()),
        "max_raw_log_bf_abs_error": float(frame["raw_log_bf_max_abs_error"].max()),
        "max_bc_abs_error": float(frame["bc_max_abs_error"].max()),
        "total_template_reuse_systems": int(frame["reused_template_systems"].sum()),
        "total_run_mismatch_events": int(frame["run_mismatch_events"].sum()),
        "total_nonfinite_maps": int(frame["nonfinite_maps"].sum()),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
