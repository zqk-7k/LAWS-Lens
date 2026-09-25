#!/usr/bin/env python3
"""Do not turn a competing profile mode into a narrow predictive density."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_trusted_local_20260909 as local
c, app, n, r, d = local.c, local.application, local.n, local.r, local.d
BASE_QUALITY = local.quality
ROOT = None
METHODS = ('NODUP-DIRECT-REPLAY', 'MODE-PROFILE-FIXED', 'MODE-PROFILE-VAL')
GAP = 4.
SEPARATION = .02


def competing_mode(row):
    return any(abs(a['parameters'][0] - row['logmc']) > SEPARATION and
               row['projection_statistic'] - a['value'] <= GAP
               for a in row['optimizer_runs'] if np.isfinite(a['value']))


def quality(parent, rows, coverage):
    active, centers = BASE_QUALITY(parent, rows, coverage)
    for row in rows:
        i = int(row['row_index'])
        if active[i] and competing_mode(row):
            active[i] = False
    return active, centers


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independentdirectoryrequired')
    for folder in ('contracts', 'calibration', 'predictions', 'configs', 'tables', 'audit', 'reports',
                   'scripts', 'logs', 'manifest', 'results', 'figures', 'cache', 'profile_events'):
        (ROOT / folder).mkdir(parents=True)
    previous = P / 'results/mcwf_nodup_trusted_local_09_20260909T084714Z'
    contract = json.loads((previous / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract.update(id='MCWF-NODUP-MODE-AWARE-PROFILE-10', UTC=n.utc(), previous_predictive_failure=str(previous),
        previous_not_rejudged=True,
        adaptive_simulation_development='09O3passesbutO4addsonecatastrophicpointprediction.InthatSIMULATIONmultipledifferentmassmodeshavesimilarprofilepower.Point-to-densitycollapseisnotjustified.',
        additional_quality={'profile_power_gap': GAP, 'logmass_mode_separation': SEPARATION,
            'rule': 'IfanotherstoredoptimumdiffersinlogMc>.02andiswithin4profilepowerunits,retainentireparentdensity.',
            'scope': 'Allfourstoredoptimizedmodesincludingnonconvergedonesarechecked;no eventIDorPEfeature.',
            'interpretation': 'Numericalmode-ambiguitysafeguardwhosepredictiveperformanceisvalidatedonsimulations;powerisnotanormalizedloglikelihood,4isNOTaBayesfactor/significancethreshold.',
            'limitations': 'Finitefour-startsearchcannotguaranteeallmodesfound;failureiskeptasfailure,notconfirmedPE.'},
        same_both_runs=True, goal_achieved=False)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.read_csv(previous / 'manifest/INPUT_SHA256.csv'))
    example = {'logmc': 2.5, 'projection_statistic': 50., 'optimizer_runs': [{'parameters': [2.55, .5, 0.], 'value': 49.}]}
    assert competing_mode(example)
    example['optimizer_runs'][0]['value'] = 40.
    assert not competing_mode(example)
    n.write_json(ROOT / 'audit/MODE_RULE_UNIT.json', {'passed': True, 'near_degenerate_separate_mode_rejected': True,
        'weak_competing_mode_not_rejected': True, 'no_PE_fields_used': True})
    shutil.copy2(__file__, ROOT / 'scripts/profile_mode_aware.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate_predictive', 'freeze_score', 'calibrate', 'evaluate', 'real'), required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    ROOT = local.ROOT = c.ROOT = d.ROOT = r.ROOT = app.ROOT = app.score.ROOT = args.root
    app.WORKERS = args.workers
    local.quality = c.quality_mask = quality
    app.score.METHODS = METHODS
    r.install()
    d.predict = app.predict
    n.METHODS, n.load_panel, n.infer = METHODS, app.score.load_panel, app.score.infer
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'calibrate_predictive':
        local.calibrate_predictive()
    elif args.stage == 'freeze_score':
        app.freeze_score()
    elif args.stage == 'calibrate':
        app.score.calibrate()
    else:
        n.run(ROOT, args.stage)
