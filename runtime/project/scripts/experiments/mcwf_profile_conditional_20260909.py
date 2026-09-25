#!/usr/bin/env python3
"""Quality-conditional profile update with exact inactive NODUP replay."""
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
from sklearn.isotonic import IsotonicRegression

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_mode_aware_20260909 as mode
app, d, r, n = mode.app, mode.d, mode.r, mode.n
score = app.score
PARENT = P / 'results/mcwf_nodup_mode_profile_10_20260909T085830Z'
METHODS = ('NODUP-DIRECT-REPLAY', 'MODE-CONDITIONAL-FIXED', 'MODE-CONDITIONAL-VAL')
ROOT = None
BASE_PANEL = score.load_panel
BASE_INFER = score.infer


def mass(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    file = PARENT / f'predictions/{dep}/{slot}_{app.tag_for(seed, split, catalog)}.npz'
    if not file.exists():
        raise RuntimeError('Parent profile panel not complete: ' + str(file))
    return dict(np.load(file))


def matrices(dep, seed, split, catalog=None):
    tag = split if catalog is None else f'{split}_{catalog}'
    file = PARENT / f'cache/{dep}/{seed}_{tag}.npz'
    if not file.exists():
        raise RuntimeError('Parent matrix panel not complete: ' + str(file))
    return dict(np.load(file))


def load_panel(dep, seed, split, catalog=None):
    f = BASE_PANEL(dep, seed, split, catalog)
    active = mass(dep, seed, split, catalog)['active']
    f['profile_active_i'] = active[f.idx_i.to_numpy(int)]
    f['profile_active_j'] = active[f.idx_j.to_numpy(int)]
    f['profile_pair_active'] = f.profile_active_i | f.profile_active_j
    return f


def infer(frame, config):
    base = BASE_INFER(frame, {**config, 'method': METHODS[0]})
    if config['method'] == METHODS[0]:
        return base
    candidate = BASE_INFER(frame, config)
    active = frame.profile_pair_active.to_numpy(bool)
    result = tuple(np.where(active, a, b) for a, b in zip(candidate, base))
    if not np.array_equal(result[0][~active], base[0][~active]):
        raise RuntimeError('Inactive waveform evidence changed')
    return result


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    parent_freeze = json.loads((PARENT / 'contracts/PREDICTIVE_FROZEN.json').read_text())
    if not parent_freeze['both_runs_pass']:
        raise RuntimeError('Parent predictive gate must pass in both runs')
    for folder in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-QUALITY-CONDITIONAL-PROFILE-11', 'UTC': n.utc(),
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'motivation': 'Changing low-mass prediction should not silently recalibrate unchanged high-mass pairs. Round10 validation shows global calibration can shift scores even when no event activates.',
        'parent': str(PARENT), 'parent_predictive_sha256': parent_freeze['sha256'],
        'fixed_before_parent_real_ranking': not (PARENT / 'contracts/REAL_COMPLETE.json').exists(),
        'event_profile_and_quality': 'Exactly frozen round10;no new fit,threshold or PE input',
        'pair_activation': 'At least one endpoint passed the same simulation-frozen event quality rule. No names,truth,PE,official fields.',
        'inactive': 'Bit-exact NODUP-DIRECT waveform replay. No old encoder Mc/q terms;no old/new total blend.',
        'active': 'ONE updated joint(Mc,eta,chi) compatibility. Conditional isotonic LR and true-source tail fitted only among profile-active fitfold0 pairs. Source-pair multiplicity weighting,balanced classes.',
        'fit_minimum_true_sources': 20, 'calibration_clip': [-4., 4.],
        'OOD': 'No positive extrapolation;endpoint mass-boundary OOD neutral',
        'outer_weights': 'Frozen pure NODUP w1;time and sky scores unchanged',
        'fixed': 'Original NODUP gamma,beta',
        'validation_grid': {'gamma': [0., .25, .5, 1., 2., 4., 8.], 'beta': [0., .0625, .125, .25, .5, 1., 2., 4.]},
        'validation_priority': ['current NODUP waveform/fusion guard', 'F50', 'F90', '-AUPRC', '-R10', '-R1', 'squared distance to frozen gamma,beta', 'gamma', 'beta'],
        'identical_validation_performance': 'Retain coefficients closest to frozen parent,never choose arbitrarily small or large coefficients',
        'same_both_runs': True, 'PE_official_only_posthoc': True,
        'test_reuse': 'Existing catalogs have been viewed in earlier development;not independent confirmation',
        'frozen': ['oldencoder', 'time', 'sky', 'scope', 'oldresults', 'paper']})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.read_csv(PARENT / 'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__, ROOT / 'scripts/profile_conditional.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json')})


def calibrate_one(dep, seed):
    meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
    meta['deployment'] = dep
    fold = r.source_fold(meta)
    a, p = matrices(dep, seed, 'development'), mass(dep, seed, 'development')
    ids = np.flatnonzero(fold == 0)
    i, j = np.triu_indices(len(ids), 1)
    i, j = ids[i], ids[j]
    active = p['active'][i] | p['active'][j]
    i, j = i[active], j[active]
    groups = meta.source_uid.to_numpy(str)
    y = groups[i] == groups[j]
    if y.sum() < 20:
        raise RuntimeError('Insufficient active independent companion systems')
    keys = np.array([':'.join(sorted((x, z))) for x, z in zip(groups[i], groups[j])])
    weight = pd.Series(keys).map(1 / pd.Series(keys).value_counts()).to_numpy()
    weight[y] *= .5 / weight[y].sum()
    weight[~y] *= .5 / weight[~y].sum()
    value = a['logbc'][i, j]
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(value, y, sample_weight=weight)
    prob = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
    spec = {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(prob) - np.log1p(-prob)).tolist(),
        'minimum': float(value.min()), 'maximum': float(value.max()), 'reference': np.sort(-value[y]).tolist(),
        'fit_source_systems': int(y.sum()), 'active_null_pairs': int((~y).sum()),
        'fit_tune_source_noise_disjoint': True, 'conditional_fallback_mass_max': float(a['conditional_fallback_mass_max'])}
    n.write_json(ROOT / f'calibration/{dep}_{seed}_JOINT.json', spec)
    return spec


def calibrate():
    configs = json.loads((r.PRIOR / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    originals = {(c['deployment'], c['seed']): c for c in configs if c['method'] == 'NODUP-DIRECT'}
    selected, grid, replay = [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old = originals[dep, seed]
            common = {**old, 'joint_calibration_subgrid': calibrate_one(dep, seed)}
            selected.extend([{**old, 'method': METHODS[0]}, {**common, 'method': METHODS[1]}])
            f = load_panel(dep, seed, 'validation')
            z, _, _ = infer(f, {**old, 'method': METHODS[0]})
            bm, bwm = n.cf.fast_metrics(f, n.cf.channels(f, z) @ np.asarray(old['weights'])), n.cf.fast_metrics(f, z)
            options = []
            for gamma in (0., .25, .5, 1., 2., 4., 8.):
                for beta in (0., .0625, .125, .25, .5, 1., 2., 4.):
                    config = {**common, 'method': METHODS[2], 'gamma': gamma, 'beta': beta}
                    candidate, _, _ = infer(f, config)
                    m = n.cf.fast_metrics(f, n.cf.channels(f, candidate) @ np.asarray(old['weights']))
                    wm = n.cf.fast_metrics(f, candidate)
                    row = {'gamma': gamma, 'beta': beta, 'guard': n.cf.guard(m, bm) and n.cf.guard(wm, bwm), **m}
                    options.append(row)
                    grid.append({'deployment': dep, 'seed': seed, **row, **{'waveform_' + k: v for k, v in wm.items()}})
            passing = [x for x in options if x['guard']]
            win = min(passing or options, key=lambda x: (x['false_at_recall_0p5'], x['false_at_recall_0p9'],
                -x['average_precision'], -x['macro_r_at_10'], -x['macro_r_at_1'],
                (x['gamma'] - old['gamma'])**2 + (x['beta'] - old['beta'])**2, x['gamma'], x['beta']))
            selected.append({**common, 'method': METHODS[2], 'gamma': win['gamma'], 'beta': win['beta'],
                             'tune_guard': bool(passing), 'tune_metrics': win})
            replay.append({'deployment': dep, 'seed': seed, 'validation_pairs': len(f),
                'active_pairs': int(f.profile_pair_active.sum()), 'inactive_max_waveform_change': 0.})
            print('CONDITIONAL_SELECTION', dep, seed, win, flush=True)
    n.write_csv(ROOT / 'tables/CONDITIONAL_VALIDATION_GRID.csv', grid)
    n.write_csv(ROOT / 'audit/INACTIVE_REPLAY.csv', replay)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', selected)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection': False, 'no_total_alpha': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = d.ROOT = r.ROOT = score.ROOT = args.root
    r.install()
    score.METHODS = METHODS
    score.matrices = matrices
    n.METHODS, n.load_panel, n.infer = METHODS, load_panel, infer
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
