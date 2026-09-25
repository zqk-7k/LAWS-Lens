#!/usr/bin/env python3
"""Sequential gates; preserve each exit status and stop on any failure."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.root.exists():raise RuntimeError('Do not overwrite or resume an unreviewed experiment')
    here=Path(__file__).parent
    stages=[('tests',[sys.executable,'-B',str(here/'test_calfuse.py')])]
    stages += [(phase,[sys.executable,'-B',str(here/'calfuse.py'),'--root',str(a.root),'--phase',phase])
               for phase in ('init','calibrate','evaluate','real')]
    initial=[]
    for phase,command in stages:
        t=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
        proc=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        state={'phase':phase,'UTC_start':utc,'wall_seconds':time.monotonic()-t,'exit_code':proc.returncode,'command':command}
        if a.root.exists():
            for name,output,record in initial+[(phase,proc.stdout,state)]:
                log=a.root/'logs'/f'{name}.log'
                with log.open('x') as f:f.write(output)
                with log.with_suffix('.runtime.json').open('x') as f:json.dump(record,f,indent=2)
            initial=[]
            for source in ('run_calfuse.py','test_calfuse.py'):
                target=a.root/'scripts'/source
                if not target.exists():shutil.copy2(here/source,target)
        else:initial.append((phase,proc.stdout,state))
        print(json.dumps(state),flush=True)
        print(proc.stdout[-3000:],flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
