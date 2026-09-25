#!/usr/bin/env python3
"""Run the frozen R56 simulation stages, preserving failures and checkpoints."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

SCRIPT=Path('/root/autodl-tmp/gw-catalog/scripts/experiments/mcwf_shared_profile_population_expansion_20260909.py')


def write(path,data):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2)


def main(args):
    root=args.root
    reference=json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())
    frozen=json.loads((root/'contracts/START_FREEZE.json').read_text())
    def check():
        if hashlib.sha256(SCRIPT.read_bytes()).hexdigest()!=frozen['runtime_sha256']:
            raise RuntimeError('Frozen R56 runtime changed')
    write(root/'contracts/DRIVER_START.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'workers':24,'stages':['prepare','unit','measure','fit'],'no_catalog_or_real_stage':True})
    shutil.copy2(__file__,root/'scripts/shared_expansion_driver.py')
    for stage in ('prepare','unit','measure','fit'):
        check()
        command=['/root/miniconda3/bin/python','-B',str(SCRIPT),'--root',str(root),
            '--data-root',reference['data'],'--pilot-root',reference['reuse_physical_pilot'],
            '--reference-root',reference['reference'],'--stage',stage,'--workers','24']
        start=time.monotonic()
        print('BEGIN_EXPANSION_STAGE',stage,datetime.now(timezone.utc).isoformat(),flush=True)
        with (root/f'logs/{stage}.log').open('x') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            with process:
                for line in process.stdout:
                    log.write(line);log.flush()
                    print(stage,line.rstrip(),flush=True)
        write(root/f'logs/{stage}_RECEIPT.json',{'UTC':datetime.now(timezone.utc).isoformat(),
            'returncode':process.returncode,'wall_seconds':time.monotonic()-start,'command':command})
        check()
        if process.returncode:
            write(root/'contracts/DRIVER_FAILURE.json',{'stage':stage,'returncode':process.returncode,
                'goal_achieved':False,'later_stages_not_launched':True})
            raise RuntimeError('R56 stage failed: '+stage)
    write(root/'contracts/BOUNDED_RUN_COMPLETE.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'goal_achieved':False,'no_real_or_catalog_scoring':True,
        'status':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
