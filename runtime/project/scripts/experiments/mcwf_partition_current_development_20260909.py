#!/usr/bin/env python3
"""Controlled partition-normalization test on archived development only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_waveform_calibration_20260909 as c
n, r, h, s, co = c.n, c.r, c.h, c.s, c.co
ROOT = None


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for folder in ('contracts', 'configs', 'calibration', 'cache', 'tables', 'audit', 'scripts', 'logs',
                   'manifest', 'reports', 'results', 'figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {'UTC': n.utc(),
        'id': 'MCWF-NODUP-PARTITION-NORMALIZATION-CONTROL-24', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'same_both_runs': True,
        'purpose': 'Isolateconditional-scorepartitionnormalization whileindependentR22Bdataarebeingacquired.',
        'data': 'Existing512developmentparents/32noiseblocks,originalnoisehashfolds;crossfoldsourcesdropped. NOTnewindependentcalibration.',
        'predictive_distribution': 'UnmodifiedR10 profilemass andconditionaleta/chi;oldbase96Dembedding only.',
        'features': list(s.FEATURES), 'methods_internal_ids': list(c.METHODS),
        'internal_id_note': 'NEWDEV in inheritedmethodIDs meansnewcalibrationcontrol,NOTnewindependentsourcepopulation inthisround.',
        'conditional_formula': 'logp(x|L,A)-logp(x|N,A)+logP(A|L)-logP(A|N);Afromwaveformqualityonly,allprobabilitiesfitondevelopmentfold0.',
        'difference_fromR14': 'TwoexpertsA=0and1 formONEgloballyreferencedwaveformLR;nohybridNODUPfallback,sourcepairweightednullswithsameexactnoiseblockexcluded.',
        'global_controls': 'SingleclassifierglobalLINEAR/TREE withsamefeatures/population/nullrule;no separate Zcos.',
        'hyperparameters': 'LINEARorTREE3/7leaves100iterations,lr.05,ridge.001,.01,.1,1;monotonicfirst4features;minimum20fit/tunecompaniongroupsperpartition.',
        'selection': 'Properbalancedloglossonfold1only,tiesfewerleaves/strongerridge. No realPE,officialmatch,examplepairidentityorheldoutmetrics.',
        'cap_OOD': 'Commonlog(totalfitcompanions+1)cap;featureboxperexpert;positiveOOD0AFTERpartitionoffset.',
        'frozen': ['allencoders', 'time', 'sky', 'outerweights', 'scope', 'historicalrankings', 'paper'],
        'forbidden': ['oldMc_q_head', 'total-scoreblend', 'event-specificrule', 'PE_official_scoreinputs'],
        'interpretation': 'Smallreuseddevelopmentcontrol;notconfirmatory. Allpositiveandnegativearmsreported. Awaitnewindependentpopulationbeforeanyupgrade.'})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', pd.read_csv(r.PRIOR/'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__, ROOT/'scripts/partition_current_development.py')
    shutil.copy2(c.__file__, ROOT/'scripts/independent_waveform_calibration.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
        'dependency_sha256': n.sha(Path(c.__file__)), 'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def panels(dep, seed):
    meta = pd.read_parquet(n.t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
    meta['deployment'] = dep
    fold = r.source_fold(meta)
    group, names = pd.factorize(meta.source_uid, sort=True)
    noise = meta.noise_bank_index.to_numpy(int)
    if set(group[fold == 0]) & set(group[fold == 1]) or set(noise[fold == 0]) & set(noise[fold == 1]):
        raise RuntimeError('Source/noise leakage')
    a = h.arrays(dep, seed, 'development')
    z = np.load(s.EMB/f'{dep}/{seed}_development.npy').astype(float)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    result = {}
    for side in (0, 1):
        ids = np.flatnonzero(fold == side)
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        mask = noise[i] != noise[j]
        i, j = i[mask], j[mask]
        y = group[i] == group[j]
        x = np.column_stack([np.sum(z[i]*z[j], 1), h.features(a, i, j)])
        active = a['active'][i] | a['active'][j]
        keys = np.minimum(group[i], group[j])*len(names)+np.maximum(group[i], group[j])
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        w = 1/counts[inverse]
        w[y] *= .5/w[y].sum(); w[~y] *= .5/w[~y].sum()
        result[side] = x, y, w, active, i, j
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = c.ROOT = s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
    c.panels = panels
    r.install()
    co.score.METHODS = c.METHODS
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer = c.METHODS, h.load_panel, c.infer
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'calibrate':
        c.calibrate()
    else:
        n.run(ROOT, args.stage)
