#!/usr/bin/env python3
"""Run registered seeds sequentially, retaining each job's stdout and failure."""
import argparse
from pathlib import Path
import subprocess
import sys
import json
import time
import mcwf_development_20260905 as dev

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--deployment", required=True, choices=("gwtc3", "gwtc4"))
a = p.parse_args()
script = Path(__file__).with_name("mcwf_mass_tf_20260905.py")
for model_seed, eval_seed in zip((202609051, 202609052, 202609053), dev.SEEDS):
    for phase in ("train", "evaluate"):
        command = [sys.executable, str(script), "--root", str(a.root), "--deployment", a.deployment,
                   "--seed", str(model_seed), "--eval-seed", str(eval_seed), "--phase", phase]
        log = a.root / f"logs/replication_{a.deployment}_{model_seed}_{phase}.log"
        if log.exists():
            log = log.with_name(log.stem + f"_resume{int(time.time())}.log")
        start = time.time()
        with log.open("w") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        row = {"command": command, "returncode": result.returncode, "seconds": time.time()-start, "log": str(log)}
        print(json.dumps(row), flush=True)
        if result.returncode:
            dev.json_write(a.root / f"contracts/REPLICATION_FAILURE_{a.deployment}_{model_seed}_{phase}.json", row)
            raise SystemExit(result.returncode)
pe = dev.pe_audit(a.root, a.deployment)
metrics = []
for method in ("mass_only", "negative_guard", "mass_cap"):
    frames = {}
    for seed, ev in zip((202609051, 202609052, 202609053), dev.SEEDS):
        path = a.root / f"results/MASSTF-{method}-model{seed}/{a.deployment}/seed_{ev}_C_fixed.parquet"
        frames[ev] = dev.pd.read_parquet(path)
    rows = dev.save_evaluation(a.root, f"MASSTF-{method}-three-seed", a.deployment, frames, pe)
    print(json.dumps({"consensus": method, "top10": [r for r in rows if r["budget"]==10 and r["seed"]=="consensus"]}), flush=True)
for model_seed, ev in zip((202609051, 202609052, 202609053), dev.SEEDS):
    metrics.append(dev.pd.read_csv(a.root / f"results/MASSTF/{a.deployment}/model_{model_seed}_eval_{ev}/retrieval_pair_metrics.csv"))
table = dev.pd.concat(metrics, ignore_index=True)
dev.csv_write(a.root / f"tables/MASSTF_{a.deployment}_REPLICATION_METRICS.csv", table)
cols = ["macro_r_at_1", "macro_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"]
summary = table.groupby(["split", "waveform_rule", "method"])[cols].agg(["mean", "std"]).reset_index()
summary.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in summary.columns]
dev.csv_write(a.root / f"tables/MASSTF_{a.deployment}_REPLICATION_SUMMARY.csv", summary)
dev.json_write(a.root / f"contracts/MASSTF_{a.deployment}_REPLICATION_COMPLETE.json", {"seeds": [202609051,202609052,202609053], "historical_time_sky_frozen": True, "status": dev.FINAL_STATUS})
