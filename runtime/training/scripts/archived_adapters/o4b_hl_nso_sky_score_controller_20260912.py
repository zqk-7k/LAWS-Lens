#!/usr/bin/env python3
"""Wait for frozen validation maps and compute raw sky evidence only."""
import argparse
from pathlib import Path
import time
import traceback
import o4b_hl_nso_training_controller_20260912 as c


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    status=a.root/'SKY_SCORE_STATUS.json';start=time.monotonic()
    try:
        for seed in c.SEEDS:
            folder=a.root/f'event_maps/seed_{seed}/validation'
            c.s.write(status,{'state':'WAIT_VALIDATION_MAPS','seed':seed,'utc':c.s.now(),'test_scored':False})
            while not (folder/'TEMPERATURE_FREEZE.json').exists():
                c.s.disk_guard(a.root)
                if list(folder.glob('event_*.failure.*.json')):raise RuntimeError('BAYESTAR event failure needs audit')
                if time.monotonic()-start>24*3600:raise RuntimeError('Map wait exceeded24h')
                time.sleep(30)
            c.s.write(status,{'state':'COMPUTING_VALIDATION_SKY_MATRICES','seed':seed,'utc':c.s.now(),'test_scored':False})
            c.call(a.root,f'validation_sky_score_{seed}','o4b_hl_nso_sky_scores_20260912.py',
                   ['--seed',str(seed),'--split','validation'])
        c.s.write(status,{'state':'VALIDATION_SKY_MATRICES_READY','utc':c.s.now(),'test_scored':False,
                          'full_NEW_SCORE_ONLY_evaluation_complete':False})
    except Exception:
        c.s.write(status,{'state':'HOLD_SKY_SCORE_FAILURE','utc':c.s.now(),'traceback':traceback.format_exc()})
        raise
