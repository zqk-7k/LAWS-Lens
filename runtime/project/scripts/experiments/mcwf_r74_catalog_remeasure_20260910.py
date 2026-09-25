#!/usr/bin/env python3
"""R74: remeasure the frozen catalog with the R71 shared-Mc operator.

This driver deliberately has no scoring or real-catalog stage.  It first creates
an independent physical-measurement root and only permits downstream scoring
after every injection pair is complete and the numerical gate passes.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

P = Path("/root/autodl-tmp/gw-catalog")
sys.path.insert(0, str(P / "scripts/experiments"))
import mcwf_shared_profile_catalog_20260909 as old_catalog
import mcwf_shared_mass_nuisance_pilot_20260910 as r71

R51 = P / "results/mcwf_nodup_shared_profile_catalog_51b_20260909T223530Z"
R73 = P / "results/mcwf_shared_mass_conditional_73_20260910T110300Z"
ROOT = None


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_signature(arrays):
    """Hash the same compressed representation used for the R72 cache."""
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, full20=arrays[0], frequency=arrays[1], psd=arrays[2])
    return sha_bytes(buf.getvalue())


def pair_path(root, row):
    return root / "pairs" / str(row["deployment"]) / str(row["tag"]) / f"{int(row['idx_i'])}_{int(row['idx_j'])}.json"


def freeze(root):
    if root.exists():
        raise RuntimeError("Independent R74 root required")
    gate = json.loads((R73 / "contracts/PILOT_GATE.json").read_text())
    if gate.get("gate") != "PASS":
        raise RuntimeError("R73 gate is not PASS")
    r73_models = json.loads((R73 / "configs/ALL_CONDITIONAL_MODELS.json").read_text())
    chosen = gate["selected_common_arm"]
    if chosen not in ("FULL-SHARED-NUMERICAL-CONTROL", "SHARED-MASS-CONDITIONAL"):
        raise RuntimeError("Unexpected R73 arm")
    plan = pd.read_parquet(R51 / "contracts/PAIR_PLAN.parquet")
    injection = plan[plan.scope.eq("injection")].copy()
    if injection.pair_id.duplicated().any() or injection.empty:
        raise RuntimeError("Invalid frozen injection plan")
    for folder in ("contracts", "pairs", "tables", "audit", "reports", "scripts", "logs", "manifest"):
        (root / folder).mkdir(parents=True)
    injection.to_parquet(root / "contracts/PAIR_PLAN.parquet", index=False)
    contract = {
        "UTC": r71.io.utc(),
        "id": "MCWF-R74-CATALOG-PHYSICAL-REMEASUREMENT",
        "status": r71.n.STATUS,
        "goal_achieved": False,
        "parent_catalog_plan": str(R51 / "contracts/PAIR_PLAN.parquet"),
        "r73_root": str(R73),
        "selected_r73_arm": chosen,
        "pairs": int(len(injection)),
        "scope": "All frozen injection catalog panels from R51, excluding real events. No PE/official fields and no real ranking.",
        "operator": "Exactly R71 shared-Mc five-parameter and full-shared three-parameter numerical operator; same starts, budgets and canonical waveform-content ordering.",
        "provenance": "R51 physical measurements are not reused as new D values. Only frozen event cache, profile and pair plan are reused.",
        "fixed": ["encoder checkpoints", "waveform score", "time", "sky", "outer weights", "scope", "splits", "all historical results"],
        "measurement_gate": "All planned injection pairs COMPLETE; finite D_mass/D_full; nested inequalities and frozen-power replay pass. Best-mass nonconvergence is reported, not silently dropped.",
        "downstream": "Only after the injection gate may the R73 selected model be applied to injection panels. Real measurement/ranking remains a separate later stage.",
        "no_real_or_test_selection": True,
    }
    r71.io.write(root / "contracts/ANALYSIS_CONTRACT.json", contract)
    files = [Path(__file__), Path(old_catalog.__file__), Path(r71.__file__), Path(r71.physical.__file__),
             R51 / "contracts/PAIR_PLAN.parquet", R51 / "contracts/START_FREEZE.json",
             R51 / "manifest/INPUT_SHA256.csv", R73 / "contracts/PILOT_GATE.json",
             R73 / "configs/ALL_CONDITIONAL_MODELS.json"]
    files.extend(Path(x) for x in pd.read_csv(R51 / "manifest/INPUT_SHA256.csv").path.tolist())
    unique = []
    seen = set()
    for path in files:
        path = Path(path)
        if path.exists() and str(path) not in seen:
            unique.append({"path": str(path), "sha256": r71.io.sha(path), "bytes": path.stat().st_size})
            seen.add(str(path))
    r71.io.csv(root / "manifest/INPUT_SHA256.csv", unique)
    shutil.copy2(Path(__file__), root / "scripts/mcwf_r74_catalog_remeasure_20260910.py")
    r71.io.write(root / "contracts/START_FREEZE.json", {
        "UTC": r71.io.utc(), "runtime_sha256": r71.io.sha(Path(__file__)),
        "contract_sha256": r71.io.sha(root / "contracts/ANALYSIS_CONTRACT.json"),
        "plan_sha256": r71.io.sha(root / "contracts/PAIR_PLAN.parquet"),
        "manifest_sha256": r71.io.sha(root / "manifest/INPUT_SHA256.csv"),
        "selected_r73_arm": chosen, "r73_model_hash": sha_bytes(json.dumps(r73_models, sort_keys=True).encode()),
    })
    print("R74_CATALOG_REMEASUREMENT_FROZEN", len(injection), flush=True)


def check(root):
    stamp = json.loads((root / "contracts/START_FREEZE.json").read_text())
    for key, path in (("runtime_sha256", Path(__file__)), ("contract_sha256", root / "contracts/ANALYSIS_CONTRACT.json"),
                      ("plan_sha256", root / "contracts/PAIR_PLAN.parquet"), ("manifest_sha256", root / "manifest/INPUT_SHA256.csv")):
        if r71.io.sha(path) != stamp[key]:
            raise RuntimeError("Frozen R74 input changed: " + str(path))
    for row in pd.read_csv(root / "manifest/INPUT_SHA256.csv").itertuples():
        if r71.io.sha(row.path) != row.sha256:
            raise RuntimeError("Protected input changed: " + row.path)


def work(task):
    root, row = task
    row = dict(row)
    start = time.monotonic()
    ctx = old_catalog.context(row["deployment"], row["tag"])
    models, profiles, signatures = [], [], []
    for index in (int(row["idx_i"]), int(row["idx_j"])):
        pos = ctx["indices"][index]
        arrays = (np.asarray(ctx["full20"][pos]), np.asarray(ctx["frequency"]), np.asarray(ctx["psd"][pos]))
        models.append(r71.physical.physical.LowBand(*arrays, 16))
        profiles.append(json.loads((old_catalog.PROFILE / f"profile_events/{row['deployment']}/{row['tag']}/{index}.json").read_text()))
        signatures.append(canonical_signature(arrays))
    reference_path = R51 / "pairs" / str(row["deployment"]) / str(row["tag"]) / f"{int(row['idx_i'])}_{int(row['idx_j'])}.json"
    reference = json.loads(reference_path.read_text())
    result = r71.fit_pair(models, profiles, reference, signatures)
    result.update({"pair_id": row["pair_id"], "deployment": row["deployment"], "tag": row["tag"],
                   "fold": int(row.get("fold", -1)), "kind": row.get("kind", "catalog"),
                   "idx_i": int(row["idx_i"]), "idx_j": int(row["idx_j"]), "status": "COMPLETE",
                   "reference_sha256": r71.io.sha(reference_path), "seconds": time.monotonic() - start,
                   "peak_worker_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    # Compatibility aliases are for the existing panel loader only; D_mass remains explicit.
    result["deficit"] = result["D_mass"]
    result["relative_deficit"] = result["D_mass"] / max(result["independent_power_common_denominator"][0] + result["independent_power_common_denominator"][1], 1e-12)
    result["minimum_independent_power"] = min(result["independent_power_common_denominator"])
    result["any_shared_converged"] = result["any_mass_converged"]
    return result


def measure(root, workers):
    check(root)
    if (root / "contracts/INJECTION_MEASUREMENT_COMPLETE.json").exists():
        raise RuntimeError("R74 injection measurement is immutable")
    plan = pd.read_parquet(root / "contracts/PAIR_PLAN.parquet")
    jobs = [(str(root), row) for row in plan.to_dict("records") if not pair_path(root, row).exists()]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=r71.physical.init_compute) as pool:
        futures = {pool.submit(work, job): job[1] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception:
                result = {"pair_id": row["pair_id"], "deployment": row["deployment"], "tag": row["tag"],
                          "idx_i": int(row["idx_i"]), "idx_j": int(row["idx_j"]), "status": "FAIL",
                          "traceback": traceback.format_exc()}
            path = pair_path(root, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            r71.io.write(path, result)
            print("R74_CATALOG_PAIR", number, len(jobs), row["pair_id"], result["status"], result.get("seconds"), result.get("D_mass"), flush=True)
    records = [json.loads(pair_path(root, row).read_text()) for row in plan.to_dict("records") if pair_path(root, row).exists()]
    failures = sum(r.get("status") != "COMPLETE" for r in records)
    r71.io.write(root / "contracts/INJECTION_MEASUREMENT_COMPLETE.json", {
        "UTC": r71.io.utc(), "status": r71.n.STATUS, "goal_achieved": False,
        "expected": len(plan), "pairs": len(records), "failures": failures,
        "new_pairs": len(jobs), "workers": workers, "wall_seconds": time.monotonic() - start})
    if failures or len(records) != len(plan):
        raise RuntimeError("R74 injection physical measurement incomplete; failures retained")
    summarize(root)


def summarize(root):
    check(root)
    if (root / "contracts/MEASUREMENT_GATE.json").exists():
        raise RuntimeError("R74 measurement gate immutable")
    plan = pd.read_parquet(root / "contracts/PAIR_PLAN.parquet")
    rows = [json.loads(pair_path(root, row).read_text()) for row in plan.to_dict("records")]
    frame = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
    if len(frame) != len(plan) or not frame.status.eq("COMPLETE").all():
        raise RuntimeError("R74 summary incomplete")
    frame.to_parquet(root / "tables/CATALOG_PHYSICAL_MEASUREMENTS.parquet", index=False)
    r71.io.csv(root / "tables/CATALOG_PHYSICAL_MEASUREMENTS.csv", frame.to_dict("records"))
    summary = []
    for (dep, tag), g in frame.groupby(["deployment", "tag"]):
        row = {"deployment": dep, "tag": tag, "pairs": len(g), "best_mass_not_converged": int((~g.best_mass_has_converged_run).sum()),
               "D_mass_P50": g.D_mass.quantile(.5), "D_mass_P90": g.D_mass.quantile(.9), "D_mass_P99": g.D_mass.quantile(.99),
               "D_full_P50": g.D_full_common_denominator.quantile(.5), "D_full_P90": g.D_full_common_denominator.quantile(.9),
               "seconds_P50": g.seconds.quantile(.5), "seconds_P90": g.seconds.quantile(.9)}
        summary.append(row)
    r71.io.csv(root / "tables/CATALOG_PHYSICAL_STRATUM_SUMMARY.csv", summary)
    gate = {"UTC": r71.io.utc(), "gate": "PASS", "status": r71.n.STATUS, "goal_achieved": False,
            "complete_pairs": len(frame), "failures": 0, "nested_inequalities_pass": bool(frame.nested_inequalities_pass.all()),
            "max_frozen_power_replay_error": float(frame.shared_power_replay_error.abs().max()),
            "best_mass_not_converged": int((~frame.best_mass_has_converged_run).sum()),
            "scoring_or_real_ranking": False, "interpretation": "Numerical catalog remeasurement only; no performance or goal acceptance."}
    if not gate["nested_inequalities_pass"] or gate["max_frozen_power_replay_error"] > 1e-6:
        gate["gate"] = "FAIL"
    r71.io.write(root / "contracts/MEASUREMENT_GATE.json", gate)
    report = "# R74 完整注入 catalog 物理复测\n\n本轮使用 R71 冻结算子重新测量 R51 的全部注入 pair，未使用 R51 旧 deficit 作为新分数。尚未应用 R73 校准器，也未读取真实候选。\n\n```json\n" + json.dumps(gate, indent=2) + "\n```\n\n"
    report += pd.DataFrame(summary).to_markdown(index=False, floatfmt=".6g") + "\n\nHOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n"
    (root / "reports/R74_CATALOG_PHYSICAL_REMEASUREMENT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--stage", required=True, choices=("freeze", "measure", "summarize"))
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    ROOT = args.root
    if args.stage == "freeze":
        freeze(ROOT)
    elif args.stage == "measure":
        measure(ROOT, args.workers)
    else:
        summarize(ROOT)
