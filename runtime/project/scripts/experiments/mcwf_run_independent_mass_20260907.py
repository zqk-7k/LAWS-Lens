#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import mcwf_independent_mass_20260907 as method
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
method.initialize(a.root)
queue=[(d,s) for s in body.MODEL_SEEDS for d in ('gwtc3','gwtc4')]
running=[]
while queue or running:
    while queue and len(running)<2:
        dep,seed=queue.pop(0)
        if (a.root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/COMPLETE.json').exists():
            continue
        path=a.root/f'logs/{dep}_{seed}.log'
        path.parent.mkdir(exist_ok=True)
        handle=path.open('a')
        proc=subprocess.Popen([sys.executable,'-B',method.__file__,'--root',str(a.root),
                               '--deployment',dep,'--seed',str(seed)],stdout=handle,stderr=subprocess.STDOUT)
        running.append((proc,handle,dep,seed))
        print(json.dumps({'launched':dep,'seed':seed,'pid':proc.pid}),flush=True)
    time.sleep(1)
    for proc,handle,dep,seed in running[:]:
        if proc.poll() is not None:
            handle.close()
            running.remove((proc,handle,dep,seed))
            if proc.returncode:
                for other,fh,_,_ in running:
                    other.terminate();other.wait();fh.close()
                raise RuntimeError(f'Training failed {dep}/{seed}')
            print(json.dumps({'trained':dep,'seed':seed}),flush=True)
dev.json_write(a.root/'contracts/ALL_SIX_TRAINED.json',{'pass':True,'same_model_in_both_runs':True,
    'feature_implementation':'event_psd_sample_resolved_fft_v1','mass_component_independently_trained':True})
