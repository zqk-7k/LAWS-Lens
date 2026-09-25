#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
s=Path('/root/autodl-tmp/gw-catalog/scripts/experiments')
subprocess.run([sys.executable,'-B',str(s/'mcwf_intrinsic_quality_audit_20260907.py'),'--root',str(a.root/'dense')],check=True)
for name in ('run_matrix','run_increment_matrix','run_replacement_matrix'):
    subprocess.run([sys.executable,'-B',str(s/f'mcwf_intrinsic_{name}_20260907.py'),'--root',str(a.root),'--group','dense'],check=True)
