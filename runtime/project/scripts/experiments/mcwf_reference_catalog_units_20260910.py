#!/usr/bin/env python3
"""Pre-score input and dependency-invariance tests for R62."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_reference_regularized_catalog_20260910 as app
n = app.n


def main(root):
    if (root / 'contracts/PIPELINE_UNIT_PASS.json').exists():
        raise RuntimeError('Units already frozen')
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    app.ROOT, app.PILOT, app.REFERENCE, app.PHYSICAL = root, *(Path(contract[k]) for k in ('pilot', 'reference', 'physical'))
    inputs = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in inputs.itertuples():
        if n.sha(Path(r.path)) != r.sha256:
            raise RuntimeError('Protected input changed before evaluation')
    configs = {(c['deployment'], c['seed'], c['method']): c for c in n.selections(root)}
    rows, bcs = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            frame = app.load_panel(dep, seed, 'validation')
            active = frame.shared_profile_eligible.to_numpy(bool)
            if active.any() and np.isfinite(frame.shared_normalized_joint_BC[active]).all():
                new = frame.shared_normalized_joint_BC[active].to_numpy(float)
                original = frame.joint_BC[active].to_numpy(float)
                error = float(abs(new - original).max())
                rounding = float(frame.shared_joint_probability_mass_error[active].max())
                if error > 2 * rounding + 1e-12:
                    raise RuntimeError('Normalized event BC disagrees beyond probability roundoff')
                bcs.append({'deployment': dep, 'seed': seed, 'eligible_pairs': int(active.sum()),
                    'legacy_vs_normalized_BC_max_difference': error, 'event_mass_max_error': rounding})
            for method in app.METHODS:
                config = configs[dep, seed, method]
                reference = frame.copy()
                z, _, _ = app.infer(reference, config)
                poisoned = frame.copy()
                cols = [c for c in poisoned if c.startswith(('pe_', 'official_')) or c in
                        ('time_score', 'sky_raw_log_bf', 'PATH875_waveform', 'PATH875_final_score',
                         'waveform_score', 'final_score', 'old_predicted_mc', 'old_predicted_q')]
                for col in cols:
                    poisoned[col] = -987654321.
                poisoned['pe_mc_bhattacharyya_coefficient'] = 1.
                poisoned['official_hanabi_fake'] = True
                poisoned['old_predicted_mc'] = 123456789.
                zz, _, _ = app.infer(poisoned, config)
                if not np.array_equal(z, zz):
                    raise RuntimeError('Forbidden external or duplicated scores affect waveform')
                if method == app.METHODS[0] and not np.array_equal(z, frame.NODUP_frozen_waveform):
                    raise RuntimeError('NODUP replay changed')
                if method == app.METHODS[1] or (method == app.NEW and config['no_update']):
                    if not np.array_equal(z, frame.R55_frozen_waveform):
                        raise RuntimeError('No-update R55 replay changed')
                if not np.array_equal(z[~active], frame.NODUP_frozen_waveform.to_numpy()[~active]):
                    raise RuntimeError('Inactive fallback changed')
                rows.append({'deployment': dep, 'seed': seed, 'method': method,
                    'waveform_invariance_max_difference': 0., 'forbidden_columns_poisoned': len(cols) + 3,
                    'eligible_pairs': int(active.sum()), 'no_total_blend': True})
    n.write_csv(root / 'audit/PRE_EVALUATION_FORBIDDEN_INPUT_UNITS.csv', rows)
    n.write_csv(root / 'audit/LEARNED_JOINT_BC_ROUNDING_AUDIT.csv', bcs)
    shutil.copy2(__file__, root / 'scripts/reference_catalog_units.py')
    app.pilot.common.write_once(root / 'contracts/PIPELINE_UNIT_PASS.json', {'UTC': n.utc(),
        'passed': True, 'tests': len(rows), 'protected_inputs_checked': len(inputs),
        'all_catalog_or_real_scores_evaluated': False, 'scope': 'validation preflight only',
        'runtime_sha256': n.sha(Path(app.__file__)), 'test_script_sha256': n.sha(Path(__file__))})
    print('PIPELINE_UNITS_PASS', len(rows), len(inputs), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
