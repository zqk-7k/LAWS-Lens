#!/usr/bin/env python3
"""Durable SSH-independent execution; no scientific-configuration changes."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(args):
    root=args.root.resolve()
    if not (root/'contracts/UNIFIED_FRESH_FREEZE.json').is_file():
        raise RuntimeError('A frozen confirmation contract is required')
    if not args.foreground:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        work=root/f'logs/runtime/{args.deployment}_{args.phase}_{stamp}_{os.getpid()}'
        work.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--phase',args.phase,
            '--deployment',args.deployment,'--workers',str(args.workers),'--foreground','--work',str(work)]
        with (work/'supervisor.log').open('w') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(work/'LAUNCHED.json',{'pid':child.pid,'command':command,'UTC':stamp,
            'role':'durable execution wrapper only','scientific_settings_changed':False})
        print(json.dumps({'pid':child.pid,'work':str(work)}),flush=True)
        return
    work=args.work
    lock=(root/f'logs/runtime/{args.deployment}_{args.phase}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).resolve().parent/'mcwf_new_confirmation_20260906.py'
    command=[sys.executable,'-B',str(script),'--root',str(root),'--phase',args.phase,
        '--deployment',args.deployment,'--workers',str(args.workers)]
    env=os.environ.copy()
    project=Path('/root/autodl-tmp/gw-catalog')
    env['PYTHONPATH']=str(project/'results/main_o3_mcwf_encoder_v2_20260906T153600Z/environment/crypto')+':'+str(script.parent)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        env[k]='1'
    started=time.time()
    write(work/'RUNNING.json',{'started_unix':started,'command':command,'workers':args.workers,
        'restart_policy':'existing per-system checkpoints reused; never change seeds or data/model/calibration definitions'})
    with (work/'stdout_stderr.log').open('w') as log:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(work/'EXIT.json',{'returncode':result.returncode,'wall_seconds':time.time()-started,
        'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,
        'ru_maxrss_KiB':usage.ru_maxrss,'RAM_interpretation':'maximum process RSS reported by wait4, not aggregate simultaneous worker RAM',
        'successful':result.returncode==0})
    sys.exit(result.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('generate','score'),required=True)
    p.add_argument('--deployment',choices=('gwtc3','gwtc4'),required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--foreground',action='store_true');p.add_argument('--work',type=Path)
    run(p.parse_args())
