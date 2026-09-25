#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import argparse
import mcwf_distilled_encoder_20260906 as training
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body

ROOT=dev.PROJECT/'results/mcwf_unified_v4_distilled_20260906'
SCRIPT=Path(__file__).with_name('mcwf_distilled_encoder_20260906.py')
MASS_LOCAL=False
HARD_NEGATIVES=False
WAIT_FIRST=True


def run(dep):
    for seed in body.MODEL_SEEDS:
        marker=ROOT/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/COMPLETE.json'
        if WAIT_FIRST and dep=='gwtc3' and seed==202609061:
            start=time.monotonic()
            while not marker.exists():
                if time.monotonic()-start>1200:
                    raise RuntimeError('First externally launched model did not finish; inspect its checkpoint')
                time.sleep(2)
        if marker.exists():
            continue
        log=ROOT/f'logs/train_{dep}_{seed}.log'
        with log.open('x') as stream:
            cmd=[sys.executable,'-B',str(SCRIPT),'--root',str(ROOT),'--deployment',dep,'--seed',str(seed)]
            if MASS_LOCAL:
                cmd.append('--mass-local')
            if HARD_NEGATIVES:
                cmd.append('--hard-negatives')
            p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        print(json.dumps({'training_finished':dep,'seed':seed,'returncode':p.returncode}),flush=True)
        if p.returncode:
            raise RuntimeError(f'Training failed; see {log}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--mass-local',action='store_true')
    parser.add_argument('--hard-negatives',action='store_true')
    a=parser.parse_args()
    if a.root:
        ROOT=a.root
        WAIT_FIRST=False
    MASS_LOCAL=a.mass_local
    HARD_NEGATIVES=a.hard_negatives
    if HARD_NEGATIVES and not MASS_LOCAL:
        raise ValueError('Hard-negative control requires mass-local distillation')
    training.initialize(ROOT,mass_local=MASS_LOCAL,hard_negatives=HARD_NEGATIVES)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run,('gwtc3','gwtc4')):
            pass
    dev.json_write(ROOT/'contracts/ALL_SIX_TRAINED.json',{'models':6,'epochs_each':50,'architecture_loss_protocol_identical':True,'old_O4_fallback':False,'mass_local':MASS_LOCAL,'hard_negatives':HARD_NEGATIVES})
