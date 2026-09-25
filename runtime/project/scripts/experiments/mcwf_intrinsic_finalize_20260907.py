#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
s=Path('/root/autodl-tmp/gw-catalog/scripts/experiments/mcwf_intrinsic_results_export_20260907.py')
for phase in ('materialize','counterexamples','verify','report'):
    subprocess.run([sys.executable,'-B',str(s),'--root',str(a.root),'--phase',phase],check=True)
