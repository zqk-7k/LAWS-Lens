#!/usr/bin/env python3
"""Registered three-seed replication, preserving every log and comparison."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import json
import pandas as pd
import mcwf_development_20260905 as dev

p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--deployment",required=True);a=p.parse_args()
models=(202609051,202609052,202609053)
tasks=(("mcwf_phasebank_20260905.py",["--phase","train"]),("mcwf_phasebank_20260905.py",["--phase","evaluate"]),
       ("mcwf_replace_mass_20260905.py",[]),("mcwf_mass_lr_20260905.py",[]))
for m,e in zip(models,dev.SEEDS):
    for script,extra in tasks:
        command=[sys.executable,str(Path(__file__).with_name(script)),"--root",str(a.root),"--deployment",a.deployment,
                 "--seed",str(m),"--eval-seed",str(e)]+extra
        log=a.root/f"logs/PHASE_REPLICATION_{a.deployment}_{m}_{Path(script).stem}_{extra[-1] if extra else 'evaluate'}_{int(time.time())}.log"
        start=time.monotonic()
        with log.open("w") as f: r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        state={"command":command,"exit_code":r.returncode,"seconds":time.monotonic()-start,"log":str(log)}
        print(json.dumps(state),flush=True)
        if r.returncode:
            dev.json_write(log.with_suffix(".FAIL.json"),state);raise SystemExit(r.returncode)
pe=dev.pe_audit(a.root,a.deployment)
configs=[f"{model}-{rule}" for model in ("PHASEBANK","PHASEBANK-MASSLR") for rule in ("mass_only","negative_guard","mass_cap")]
configs += [f"PHASEBANK-MCREPLACED-{v}" for v in ("point","uncertainty")]
for config in configs:
    frames={e:pd.read_parquet(a.root/f"results/{config}-model{m}/{a.deployment}/seed_{e}_C_fixed.parquet") for m,e in zip(models,dev.SEEDS)}
    rows=dev.save_evaluation(a.root,f"{config}-three-seed",a.deployment,frames,pe)
    print(json.dumps({"config":config,"Top10":[r for r in rows if r["budget"]==10 and r["seed"]=="consensus"]}),flush=True)
tables=[]
for config in ("PHASEBANK","PHASEBANK-MASSLR","PHASEBANK-MCREPLACED-point","PHASEBANK-MCREPLACED-uncertainty"):
    for m,e in zip(models,dev.SEEDS):
        f=a.root/f"results/{config}/{a.deployment}/model_{m}_eval_{e}/retrieval_pair_metrics.csv"
        tables.append(pd.read_csv(f))
dev.csv_write(a.root/f"tables/PHASEBANK_{a.deployment}_ALL_METRICS.csv",pd.concat(tables,ignore_index=True))
dev.json_write(a.root/f"contracts/PHASEBANK_{a.deployment}_REPLICATION_COMPLETE.json",{"model_seeds":models,"frozen_baseline_seeds":dev.SEEDS,"no_adoption":True})
