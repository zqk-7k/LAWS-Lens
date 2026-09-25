#!/usr/bin/env python3
"""Freeze fallback coefficients independently of active-profile retuning."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_conditional_20260909 as co
n, r, d, score = co.n, co.r, co.d, co.score
ROOT = None
METHODS = ('NODUP-DIRECT-REPLAY', 'MODE-ISOLATED-FIXED', 'MODE-ISOLATED-VAL')
ORIGINALS = {(v['deployment'], v['seed']): v for v in json.loads(
    (r.PRIOR / 'configs/SELECTED_CONFIGURATIONS.json').read_text()) if v['method'] == 'NODUP-DIRECT'}


def infer(frame, config):
    original = ORIGINALS[config['deployment'], config['seed']]
    base = co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    if config['method'] == METHODS[0]:
        return base
    candidate = co.BASE_INFER(frame, config)
    active = frame.profile_pair_active.to_numpy(bool)
    result = tuple(np.where(active, a, b) for a, b in zip(candidate, base))
    # Compare with independently loaded immutable coefficients, not the
    # candidate's coefficients passed into a nominally baseline branch.
    if not np.array_equal(result[0][~active], base[0][~active]):
        raise RuntimeError('Inactive evidence differs from frozen original')
    return result


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    previous = P / 'results/mcwf_nodup_conditional_profile_11_20260909T090510Z'
    for folder in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    contract = json.loads((previous / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract.update(id='MCWF-NODUP-ISOLATED-PROFILE-12', UTC=n.utc(), previous_implementation=str(previous),
        confirmed_bug='11 usedcandidate gamma/beta in its fallback,so its self-comparison didnot prove equality to immutable NODUP.12 loads original coefficients separately.',
        changed_scientific_hypothesis=False, previous_NOT_overwritten=True,
        parent_real_complete_at_fix=(co.PARENT / 'contracts/REAL_COMPLETE.json').exists(),
        selection_not_based_on_PE=True, goal_achieved=False)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.read_csv(previous / 'manifest/INPUT_SHA256.csv'))
    tests = []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            f = co.load_panel(dep, seed, 'validation')
            archived = pd.read_parquet(r.PRIOR / f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            original = ORIGINALS[dep, seed]
            reference = infer(f, {**original, 'method': METHODS[0]})[0]
            if not np.array_equal(f[['idx_i', 'idx_j']].to_numpy(), archived[['idx_i', 'idx_j']].to_numpy()):
                raise RuntimeError('Archived pair order differs')
            error = float(abs(reference - archived.waveform_score.to_numpy()).max())
            if error > 1e-12:
                raise RuntimeError('Immutable NODUP replay failed')
            spec = json.loads((previous / f'calibration/{dep}_{seed}_JOINT.json').read_text())
            inactive = ~f.profile_pair_active.to_numpy(bool)
            for gamma, beta in [(0., 0.), (8., 4.), (original['gamma'], original['beta'])]:
                c = {**original, 'method': METHODS[2], 'joint_calibration_subgrid': spec, 'gamma': gamma, 'beta': beta}
                z = infer(f, c)[0]
                diff = float(abs(z[inactive] - archived.waveform_score.to_numpy()[inactive]).max())
                if diff > 1e-12:
                    raise RuntimeError('Retuning changed inactive archived score')
                tests.append({'deployment': dep, 'seed': seed, 'gamma': gamma, 'beta': beta,
                              'inactive_pairs': int(inactive.sum()), 'max_diff_vs_archive': diff,
                              'reference_replay_error': error})
    n.write_csv(ROOT / 'audit/INDEPENDENT_ARCHIVE_EQUALITY_UNIT.csv', tests)
    shutil.copy2(__file__, ROOT / 'scripts/profile_conditional_corrected.py')
    shutil.copy2(previous / 'reports/METHOD_AND_LIMITATIONS_CN.md', ROOT / 'reports/METHOD_AND_LIMITATIONS_CN.md')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__)),
        'unit_tests': len(tests), 'all_archive_equality_tests_pass': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = co.ROOT = d.ROOT = r.ROOT = score.ROOT = args.root
    co.METHODS = score.METHODS = METHODS
    co.infer = infer
    score.matrices = co.matrices
    r.install()
    n.METHODS, n.load_panel, n.infer = METHODS, co.load_panel, infer
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'calibrate':
        co.calibrate()
    else:
        n.run(ROOT, args.stage)
