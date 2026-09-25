#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys
import time
import json
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_bounded_20260905 as bounded

p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--deployment",required=True);a=p.parse_args()
for m,e in zip((202609051,202609052,202609053),dev.SEEDS):
    for phase in ("train","evaluate"):
        cmd=[sys.executable,str(Path(__file__).with_name("mcwf_psd_conditioned_20260905.py")),"--root",str(a.root),"--deployment",a.deployment,"--seed",str(m),"--eval-seed",str(e),"--phase",phase]
        log=a.root/f"logs/PSD_REPLICATION_{a.deployment}_{m}_{phase}_{int(time.time())}.log"
        t=time.time()
        with log.open("w") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.time()-t,"log":str(log)}),flush=True)
        if r.returncode:raise SystemExit(r.returncode)
for rule in ("mass_only","negative_guard","mass_cap"):
    frames={e:pd.read_parquet(a.root/f"results/PHASEPSD-MASSLR-{rule}-model{m}/{a.deployment}/seed_{e}_C_fixed.parquet") for m,e in zip((202609051,202609052,202609053),dev.SEEDS)}
    dev.save_evaluation(a.root,f"PHASEPSD-MASSLR-{rule}-three-seed",a.deployment,frames,dev.pe_audit(a.root,a.deployment))
bounded.run(a.root,a.deployment,source_config="PHASEPSD-MASSLR",output_prefix="PHASEPSD")
dev.json_write(a.root/f"contracts/PHASEPSD_{a.deployment}_REPLICATION_COMPLETE.json",{"seeds":[202609051,202609052,202609053],"status":"DEVELOPMENT_REPLICATION_COMPLETE_NO_ADOPTION"})
