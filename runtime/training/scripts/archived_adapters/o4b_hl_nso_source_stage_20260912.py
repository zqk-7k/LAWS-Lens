#!/usr/bin/env python3
"""Generate a new physical O4b H1/L1 source bank, without old result writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def main(root):
    ws=root/'workspace'
    source=ws/'scripts/real_search/34_generate_physical_h1l1_source_bank.py'
    output=ws/'results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared/physical_h1l1_source_bank'
    shared=output.parent
    shared.mkdir(parents=True,exist_ok=True)
    schedule=ws/'runs/real_gwtc5_o4b_v93_extension_20260830/shared/h1l1_live_schedule.csv'
    target_schedule=shared/'h1l1_live_schedule.csv'
    if not target_schedule.exists():shutil.copy2(schedule,target_schedule)
    if output.exists():raise RuntimeError('Source output already exists; no automatic overwrite')
    command=[sys.executable,'-u',str(source),'--deployment','GWTC5','--schedule',str(schedule),
             '--out-root',str(output),'--n-per-family','600','--n-unlensed','600','--seed','2026091220']
    contract={'stage':'PHYSICAL_SOURCE_BANK','created_utc':datetime.now(timezone.utc).isoformat(),
        'generator':str(source),'generator_sha256':sha(source),'command':command,
        'units':'physical dimensionless H1/L1 detector strain; no whitening at generation',
        'waveform':'IMRPhenomXPHM','sample_rate':4096,'duration':24,
        'source_bank_seed':2026091220,'systems_per_legacy_family':600,'unlensed_systems':600,
        'scope':'input generation only; no training/test or real ranking',
        'following_gate':'global source IDs and noise isolation must pass before training',
        'legacy_families':{'SIS':'GW-LMC smooth/non-subhalo','PM':'GW-LMC subhalo-present'},
        'new_score_only':True,'no_outer_old_new_mixture':True}
    write(root/'contracts/SOURCE_BANK_CONTRACT.json',contract)
    write(root/'contracts/SOURCE_BANK_FREEZE.json',{'sha256':sha(root/'contracts/SOURCE_BANK_CONTRACT.json')})
    state=root/'contracts/SOURCE_BANK_STATUS.json'
    write(state,{'state':'GENERATING','pid':os.getpid(),'output':str(output)})
    with (root/'logs/physical_source_generation.log').open('x') as log:
        r=subprocess.run(command,cwd=ws,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        write(state,{'state':'HOLD_GENERATION_FAILED','exit_code':r.returncode});raise RuntimeError('Generation failed')
    summary=json.loads((output/'physical_source_bank_summary.json').read_text())
    records=[]
    for family in ('SIS','PM'):
        d=pd.read_parquet(output/f'{family}_data_0222/physical_source_pair_metadata.parquet')
        d['family']=family;records.append(d)
    d=pd.read_parquet(output/'Unlensed_data_0222/physical_unlensed_source_metadata.parquet')
    d['family']='unlensed';records.append(d)
    all_rows=pd.concat(records,ignore_index=True)
    all_rows.to_parquet(root/'tables/PHYSICAL_SOURCE_METADATA.parquet',index=False)
    write(state,{'state':'GENERATED_PENDING_GLOBAL_SPLIT_AUDIT','output':str(output),
        'summary_status':summary['status'],'generation_failures':summary['total_generation_failures'],
        'identical_H1_L1':summary['total_identical_h1_l1_waveforms'],
        'metadata_rows':len(all_rows),'training_started':False})
    print(state.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();main(a.root)
