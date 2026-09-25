#!/usr/bin/env python3
"""Durable O4b data/learned-component controller; never scores held-out or real data.

Explicitly stops after trained components, because final scoring/calibration gates
must be implemented and audited before opening the locked test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import o4b_hl_nso_data_training_20260912 as s

SEEDS=(2026091221,2026091222,2026091223)


def call(root,label,script,args):
    logs=root/'logs/training_controller';logs.mkdir(exist_ok=True)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    log=logs/f'{label}_{stamp}.log';command=[sys.executable,'-u',str(root/'scripts'/script),'--root',str(root),*args]
    started=time.monotonic()
    with log.open('x') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=root/'workspace')
        state={'stage':label,'state':'RUNNING','utc':s.now(),'pid':p.pid,'log':str(log),'command':command}
        s.write(root/'contracts/task_status'/f'{label}.json',state)
        code=p.wait()
    state.update(state='PASS' if code==0 else 'FAIL',exit_code=code,finished_utc=s.now(),seconds=time.monotonic()-started)
    s.write(root/'contracts/task_status'/f'{label}.json',state)
    if code:raise RuntimeError(f'{label} failed with exit{code};see{log}')
    return state


def preparation(root,seed):
    for stage in ('catalog','multinoise'):
        call(root,f'{stage}_{seed}','o4b_hl_nso_data_training_20260912.py',['--stage',stage,'--seed',str(seed)])
    return seed


def auxiliary(root):
    for split in ('validation','train'):
        call(root,f'auxiliary_{split}','o4b_hl_nso_auxiliary_data_v2_20260912.py',
             ['--stage','generate','--split',split,'--workers','8'])


def run(root):
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    status=root/'RUN_STATUS.json';start=time.monotonic()
    s.write(status,{'state':'WAIT_VERIFIED_NOISE_BANK','utc':s.now(),'training_started':False,'complete_results':False})
    while not (shared/'noise_bank_complete.json').exists():
        s.disk_guard(root)
        if (root/'contracts/NOISE_INPUT_FAILURE.json').exists():raise RuntimeError('Frozen noise input failed')
        if time.monotonic()-start>12*3600:raise RuntimeError('Input wait exceeded12h; inspect downloads before resuming')
        time.sleep(30)
    s.get_split(root)
    s.write(status,{'state':'MATERIALIZING_TRAINING_DATA','utc':s.now(),'training_started':False,'complete_results':False})
    with ThreadPoolExecutor(max_workers=4) as pool:
        aux=pool.submit(auxiliary,root)
        futures={pool.submit(preparation,root,seed):seed for seed in SEEDS}
        ready=set()
        # Start GPU training as soon as the first complete per-seed bank is ready.
        for future in as_completed(futures):
            seed=future.result();ready.add(seed)
            s.write(status,{'state':'TRAINING_SHORT_ENCODER','seed':seed,'utc':s.now(),'training_started':True,'complete_results':False})
            call(root,f'short_encoder_{seed}','o4b_hl_nso_data_training_20260912.py',['--stage','train','--seed',str(seed)])
        aux.result()
    script='o4b_hl_nso_multiscale_training_20260912.py'
    s.write(status,{'state':'MULTISCALE_MATCH_FEATURES','utc':s.now(),'training_started':True,'complete_results':False})
    for split in ('validation','train'):
        call(root,f'matching_features_{split}',script,['--stage','features','--split',split])
    for seed in SEEDS:
        for stage in ('ordered','multirate','conditional'):
            s.write(status,{'state':'TRAINING_MULTISCALE_INTRINSICS','seed':seed,'stage':stage,'utc':s.now(),'training_started':True,'complete_results':False})
            call(root,f'{stage}_{seed}',script,['--stage',stage,'--seed',str(seed)])
    s.write(status,{'state':'TRAINING_COMPONENTS_COMPLETE_SCORE_INTEGRATION_PENDING','utc':s.now(),
        'training_started':True,'complete_results':False,'locked_test_opened':False,'real_ranked':False,
        'pending':['complete RAW-PHASE/FRT/OMC integration','one-dimensional time provenance','event BAYESTAR localization',
                   'validation-only score calibration/freeze','locked-test evaluation','real PE/official audit','final package'],
        'no_claim_of_final_experiment_completion':True,'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'contracts/TRAINING_CONTROLLER.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a.root)
        except Exception:
            s.write(a.root/'RUN_STATUS.json',{'state':'HOLD_COMPUTATION_FAILURE','utc':s.now(),'complete_results':False,
                'traceback':traceback.format_exc(),'test_or_real_selection':False})
            raise
