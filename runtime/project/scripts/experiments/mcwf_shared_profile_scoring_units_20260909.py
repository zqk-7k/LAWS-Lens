#!/usr/bin/env python3
"""Synthetic boundary/taint units; these rows are not experimental events."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_catalog_20260909 as app
n = app.n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output = root/'audit/SYNTHETIC_SCORING_BOUNDARY_UNITS.json'
    if output.exists():
        raise RuntimeError('Synthetic unit receipt already exists')
    records = []
    for config in n.selections(root):
        lo, hi = config['minimum_power_support']
        dl, dh = config['deficit_support']
        mid = (lo+hi)/2
        dmid = (dl+dh)/2
        frame = pd.DataFrame({
            'NODUP_frozen_waveform': [-8., -2., 4., 20., 4., 4., 4., 4., 4., 4., 4., 4.],
            'NODUP_frozen_ood': [False]*12, 'NODUP_frozen_clip': [False]*12,
            'shared_profile_available': [True]*10+[False, True],
            'shared_profile_any_optimizer_converged': [True]*11+[False],
            'shared_profile_minimum_power': [mid, mid, lo, hi, lo/2, hi*2, mid, mid, mid, mid, np.nan, mid],
            'shared_profile_deficit': [dmid, dl, dl, dh, dl, dl, dl/2, dh*2, 0., dh*100, np.nan, dl],
            'embedding_only': [.7]*12})
        old = frame.NODUP_frozen_waveform.to_numpy()
        eligible = np.array([True, True, True, True, False, False, True, True, True, True, False, False])
        expected_candidate = old.copy()
        spec = config['shared_classifier']
        for i in np.flatnonzero(eligible):
            d = frame.shared_profile_deficit.iloc[i]
            x = [.7, np.log(frame.shared_profile_minimum_power.iloc[i]), -np.log1p(d)]
            z = float(spec['coefficients'][0])
            for j in range(3):
                z += (x[j]-spec['mean'][j])/spec['scale'][j]*spec['coefficients'][j+1]
            if config['shared_cap'] is not None:
                z = max(-config['shared_cap'], min(config['shared_cap'], z))
            if d < dl or d > dh:
                z = min(z, 0.)
            expected_candidate[i] = z
        if config['method'] == app.METHODS[0]:
            expected = old.copy()
        elif config['method'] == app.METHODS[1]:
            expected = expected_candidate
        else:
            expected = np.minimum(old, expected_candidate)
        actual, _, _ = app.infer(frame, config)
        delta = float(np.max(np.abs(actual-expected)))
        if delta > 1e-12 or not np.array_equal(frame.shared_profile_eligible, eligible):
            raise RuntimeError('Independent scalar boundary replay failed')
        poisoned = frame.copy()
        for col in ('joint_BC', 'joint_logbc', 'pe_mc_bhattacharyya_coefficient', 'official_po_fpp', 'old_mc_gap', 'old_q_gap'):
            poisoned[col] = np.nan
        other = app.infer(poisoned, {**config, 'alpha': 98765.})[0]
        if not np.array_equal(actual, other):
            raise RuntimeError('External/duplicate fields affect active scores')
        permutation = np.array([11, 2, 4, 9, 0, 8, 1, 7, 10, 6, 3, 5])
        reordered = app.infer(frame.iloc[permutation].copy(), config)[0]
        if not np.array_equal(reordered, actual[permutation]):
            raise RuntimeError('Batch order changed pair scores')
        records.append({'deployment': config['deployment'], 'seed': config['seed'], 'method': config['method'],
            'synthetic_rows': len(frame), 'eligible_rows': int(eligible.sum()),
            'scalar_replay_max_difference': delta, 'PE_old_mass_totalblend_poison_difference': 0.,
            'permutation_difference': 0., 'inactive_and_optimizer_failure_exact_fallback': True,
            'power_boundary_inclusive': True, 'outside_deficit_positive_reward_zero': True})
    n.write_csv(root/'audit/SYNTHETIC_SCORING_BOUNDARY_UNITS.csv', records)
    n.write_json(output, {'UTC': n.utc(), 'passed': True, 'configurations': len(records),
        'unit_rows_are_not_experimental_samples': True, 'production_code_unchanged': True})
    shutil.copy2(__file__, root/'scripts/shared_profile_scoring_units.py')
    print('SYNTHETIC_SCORING_BOUNDARY_UNITS_PASS', len(records), flush=True)


if __name__ == '__main__':
    main()
