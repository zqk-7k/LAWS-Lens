#!/usr/bin/env python3
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import mcwf_finelag_encoder_20260907 as e
import mcwf_finelag_features_20260907 as f
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as b

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--event-psd',action='store_true')
p.add_argument('--prepared',action='store_true')
a=p.parse_args()
if a.prepared:
    if not (a.root/'contracts/FINE_LAG_TRAINING.json').exists():
        raise RuntimeError('Prepared input contract missing')
    preparation=lambda root,dep: None
    train_script=e.__file__
elif a.event_psd:
    import mcwf_finelag_eventpsd_20260907 as psd
    psd.initialize(a.root)
    preparation=psd.prepare
    train_script=psd.__file__
else:
    e.initialize(a.root)
    preparation=f.prepare
    train_script=e.__file__
for dep in ('gwtc3','gwtc4'):
    preparation(a.root,dep)
queue=[(d,s) for s in b.MODEL_SEEDS for d in ('gwtc3','gwtc4')]
running=[]
while queue or running:
    while queue and len(running)<2:
        dep,seed=queue.pop(0)
        if (a.root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/COMPLETE.json').exists():
            continue
        log=a.root/f'logs/{dep}_{seed}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        handle=log.open('a')
        proc=subprocess.Popen([sys.executable,'-B',train_script,'--root',str(a.root),'--deployment',dep,'--seed',str(seed)],stdout=handle,stderr=subprocess.STDOUT)
        running.append((proc,handle,dep,seed))
        print(json.dumps({'launched':dep,'seed':seed,'pid':proc.pid}),flush=True)
    time.sleep(1)
    for proc,handle,dep,seed in running[:]:
        if proc.poll() is not None:
            handle.close()
            running.remove((proc,handle,dep,seed))
            if proc.returncode:
                for other,fh,_,_ in running:
                    other.terminate()
                    other.wait()
                    fh.close()
                raise RuntimeError(f'Training failed {dep}/{seed}; retain checkpoints')
            print(json.dumps({'trained':dep,'seed':seed}),flush=True)
contract=json.loads((a.root/'contracts/FINE_LAG_TRAINING.json').read_text())
dev.json_write(a.root/'contracts/ALL_SIX_TRAINED.json',{'pass':True,'feature_implementation':contract.get('feature_implementation','sample_resolved_zero_padded_fft_v1'),'trained_six':True})
