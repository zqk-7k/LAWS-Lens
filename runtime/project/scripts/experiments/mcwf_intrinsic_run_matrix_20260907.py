#!/usr/bin/env python3
"""Run each registered comparison once, keeping separate durable logs."""
import argparse
from pathlib import Path
import subprocess
import sys


p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--group',choices=('conditional','joint','dense'),required=True);a=p.parse_args()
project=Path('/root/autodl-tmp/gw-catalog')
scripts=project/'scripts/experiments'
root=a.root if a.group=='conditional' else a.root/a.group
script=scripts/('mcwf_intrinsic_dense_20260907.py' if a.group=='dense' else 'mcwf_intrinsic_grid_20260907.py')
arms=('JOINT-PRIOR','JOINT-BC','MASS-Q-PRIOR','MASS-PRIOR')
for arm in arms:
    if (root/f'trials/{arm}/contracts/SEARCH_COMPLETE.json').exists():
        print('Already complete: '+a.group+' '+arm,flush=True)
        continue
    command=[sys.executable,'-B',str(scripts/'mcwf_omc_extension_run_logged_20260907.py'),
             '--log',str(a.root/f'logs/{a.group}_{arm.lower().replace("-","_")}.log'),
             sys.executable,'-B',str(script),'--root',str(root),'--phase','evaluate','--arm',arm]
    result=subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)
