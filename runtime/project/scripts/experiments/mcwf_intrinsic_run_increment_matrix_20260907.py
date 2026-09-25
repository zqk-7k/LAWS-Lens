#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--group',choices=('conditional','joint','dense'),required=True);a=p.parse_args()
scripts=Path('/root/autodl-tmp/gw-catalog/scripts/experiments')
work=a.root if a.group=='conditional' else a.root/a.group
work=work/'increment';script=scripts/'mcwf_intrinsic_conditional_increment_20260907.py'
if not work.exists():subprocess.run([sys.executable,'-B',str(script),'--root',str(work),'--phase','init'],check=True)
for arm in ('JOINT-GIVEN-MASS','Q-GIVEN-MASS'):
    if (work/f'trials/{arm}/contracts/SEARCH_COMPLETE.json').exists():continue
    subprocess.run([sys.executable,'-B',str(scripts/'mcwf_omc_extension_run_logged_20260907.py'),
                    '--log',str(a.root/f'logs/{a.group}_{arm.lower()}.log'),sys.executable,'-B',str(script),
                    '--root',str(work),'--phase','run','--arm',arm],check=True)
