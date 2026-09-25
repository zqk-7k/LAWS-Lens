#!/usr/bin/env python3
"""Generate only validation BAYESTAR maps; locked test remains blocked."""
import argparse
import fcntl
from pathlib import Path
import time
import traceback
import o4b_hl_nso_data_training_20260912 as s
from o4b_hl_nso_training_controller_20260912 import call,SEEDS


def run(root):
    start=time.monotonic();status=root/'SKY_RUN_STATUS.json'
    for seed in SEEDS:
        marker=root/f'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/seed_{seed}/data/real_noise_injections/catalog_complete.json'
        while not marker.exists():
            s.disk_guard(root)
            if time.monotonic()-start>12*3600:raise RuntimeError('Catalog wait exceeded12h')
            s.write(status,{'state':'WAIT_INJECTION_CATALOG','seed':seed,'utc':s.now(),'locked_test_opened':False})
            time.sleep(30)
        s.write(status,{'state':'BAYESTAR_VALIDATION_RUNNING','seed':seed,'utc':s.now(),'locked_test_opened':False})
        call(root,f'BAYESTAR_validation_{seed}','o4b_hl_nso_bayestar_20260912.py',['--seed',str(seed),'--split','validation','--workers','4'])
    s.write(status,{'state':'VALIDATION_MAPS_AND_TEMPERATURE_READY','utc':s.now(),'locked_test_opened':False,
                    'pending':'final score calibration and final frozen config before any test sky evaluation'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'contracts/BAYESTAR_CONTROLLER.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a.root)
        except Exception:
            s.write(a.root/'SKY_RUN_STATUS.json',{'state':'HOLD_BAYESTAR_FAILURE','utc':s.now(),'traceback':traceback.format_exc()});raise
