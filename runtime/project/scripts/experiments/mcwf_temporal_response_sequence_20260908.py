#!/usr/bin/env python3
"""Resume only complete stages; preserve the original experiment inputs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

P=Path('/root/autodl-tmp/gw-catalog')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def run(root,stage,script,args):
    complete=root/f'contracts/STAGE_{stage}_COMPLETE.json'
    if complete.exists():return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log=root/f'logs/{stage}_{stamp}'
    cmd=['/root/miniconda3/bin/python','-B',str(script),'--root',str(root),*args]
    started=time.perf_counter()
    with log.with_suffix('.stdout.log').open('xb') as out,log.with_suffix('.stderr.log').open('xb') as err:
        process=subprocess.run(cmd,stdout=out,stderr=err)
    report={'command':cmd,'returncode':process.returncode,'wall_seconds':time.perf_counter()-started,
            'utc':datetime.now(timezone.utc).isoformat(),'stdout':str(log.with_suffix('.stdout.log')),
            'stderr':str(log.with_suffix('.stderr.log'))}
    target=complete if process.returncode==0 else root/f'contracts/STAGE_{stage}_FAIL_{stamp}.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if process.returncode:
        print(log.with_suffix('.stderr.log').read_text()[-6000:],flush=True)
        raise SystemExit(process.returncode)


def main(root):
    scripts=P/'scripts/experiments'
    hashes=[]
    for name in ('mcwf_temporal_response_20260908.py','mcwf_temporal_response_evaluate_20260908.py','mcwf_temporal_response_sequence_20260908.py'):
        src=scripts/name;dst=root/'scripts'/name
        if dst.exists() and digest(src)!=digest(dst):raise RuntimeError('Frozen script changed')
        if not dst.exists():shutil.copy2(src,dst)
        hashes.append({'source':str(src),'snapshot':str(dst),'sha256':digest(src)})
    freeze=root/'contracts/EXECUTION_CODE_HASHES.json'
    if not freeze.exists():freeze.write_text(json.dumps(hashes,indent=2)+'\n')
    begin=time.monotonic()
    while not all((root/f'features/{dep}/{part}.COMPLETE.json').exists()
                  for dep in ('gwtc3','gwtc4') for part in ('validation','train','additional')):
        if time.monotonic()-begin>7200:raise RuntimeError('Feature completion timeout,inspect logs')
        time.sleep(15)
    run(root,'TRAIN',scripts/'mcwf_temporal_response_20260908.py',['--stage','train'])
    run(root,'SELECT',scripts/'mcwf_temporal_response_evaluate_20260908.py',['--stage','select'])
    run(root,'EVALUATE',scripts/'mcwf_temporal_response_evaluate_20260908.py',['--stage','evaluate'])
    run(root,'REAL_AUDIT',scripts/'mcwf_temporal_response_evaluate_20260908.py',['--stage','real'])
    run(root,'ASSESS',scripts/'mcwf_temporal_response_evaluate_20260908.py',['--stage','assess'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
