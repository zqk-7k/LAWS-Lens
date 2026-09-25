#!/usr/bin/env python3
"""Run only the missing RAW-PHASE/RNC development component; no test access."""
import argparse
import json
from pathlib import Path
import traceback
import o4b_hl_nso_training_controller_20260912 as c


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    status=a.root/'RNC_V2_RUN_STATUS.json';script='o4b_hl_nso_rankncontrast_v2_20260912.py'
    try:
        c.s.write(status,{'state':'WAIT_GPU_FOR_RNC_FEATURES','utc':c.s.now(),'locked_test_opened':False})
        c.call(a.root,'rnc_v2_features',script,['--stage','prepare'])
        for seed in c.SEEDS:
            c.s.write(status,{'state':'O4B_RNC_TRAINING','seed':seed,'utc':c.s.now(),'locked_test_opened':False})
            c.call(a.root,f'rnc_v2_train_{seed}',script,['--stage','train','--seed',str(seed)])
        c.s.write(status,{'state':'RNC_COMPONENT_COMPLETE_CALIBRATION_PENDING','utc':c.s.now(),'locked_test_opened':False})
    except Exception:
        c.s.write(status,{'state':'HOLD_RNC_FAILURE','utc':c.s.now(),'traceback':traceback.format_exc()})
        raise
