#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--group',choices=('conditional','joint','dense'),required=True);a=p.parse_args()
scripts=Path('/root/autodl-tmp/gw-catalog/scripts/experiments');w=a.root if a.group=='conditional' else a.root/a.group
w=w/'replacement';s=scripts/'mcwf_intrinsic_replacement_control_20260907.py'
if not w.exists():subprocess.run([sys.executable,'-B',str(s),'--root',str(w),'--phase','init'],check=True)
for arm in ('JOINT-PRIOR','JOINT-BC'):
    if (w/f'trials/{arm}/contracts/SEARCH_COMPLETE.json').exists():continue
    subprocess.run([sys.executable,'-B',str(scripts/'mcwf_omc_extension_run_logged_20260907.py'),
                    '--log',str(a.root/f'logs/{a.group}_replacement_{arm.lower()}.log'),
                    sys.executable,'-B',str(s),'--root',str(w),'--phase','run','--arm',arm],check=True)
