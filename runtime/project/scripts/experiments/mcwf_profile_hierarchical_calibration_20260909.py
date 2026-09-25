#!/usr/bin/env python3
"""One intrinsic calibrator retaining mass and conditional overlap information."""
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
import mcwf_profile_conditional_corrected_20260909 as isolated
co, r, n = isolated.co, isolated.r, isolated.n
ROOT = None
ACTIVE_FIT = False
FEATURES = ('profile_log_mass_BC', 'profile_log_conditional_BC', 'profile_negative_mass_distance',
            'profile_log_pooled_width', 'profile_mean_logmass', 'profile_both_active')
METHODS = ('NODUP-DIRECT-REPLAY', 'PROFILE-HIER-LINEAR', 'PROFILE-HIER-TREE',
           'PROFILE-HIER-ACTIVE-LINEAR', 'PROFILE-HIER-ACTIVE-TREE')
CACHE = {}


def arrays(dep, seed, split, catalog=None):
    key = dep, seed, split, catalog
    if key in CACHE:
        return CACHE[key]
    mass, joint = co.mass(dep, seed, split, catalog), co.matrices(dep, seed, split, catalog)
    p = mass['p']
    sm = r.mass_summary(p)
    valid = np.isfinite(p).all(1)
    b = np.full((len(p), len(p)), np.nan)
    x = np.sqrt(p[valid])
    ix = np.flatnonzero(valid)
    b[np.ix_(ix, ix)] = (x @ x.T).clip(1e-300, 1.)
    if np.nanmax(joint['bc'] - b) > 1e-8:
        raise RuntimeError('Joint BC cannot exceed its mass-marginal BC')
    CACHE[key] = dict(massbc=b, jointbc=joint['bc'], sm=sm, active=mass['active'])
    return CACHE[key]


def features(a, i, j):
    mb = a['massbc'][i, j]
    cb = (a['jointbc'][i, j] / mb).clip(1e-300, 1.)
    old = r.pair_features(a['sm'], i, j, np.log(a['jointbc'][i, j]))
    return np.column_stack([np.log(mb), np.log(cb), old[:, 2], old[:, 3], old[:, 4],
                            (a['active'][i] & a['active'][j]).astype(float)])


def load_panel(dep, seed, split, catalog=None):
    f = co.load_panel(dep, seed, split, catalog)
    x = features(arrays(dep, seed, split, catalog), f.idx_i.to_numpy(int), f.idx_j.to_numpy(int))
    for k, name in enumerate(FEATURES):
        f[name] = x[:, k]
    return f


def development(dep, seed):
    meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
    meta['deployment'] = dep
    fold = r.source_fold(meta)
    groups = meta.source_uid.to_numpy(str)
    a = arrays(dep, seed, 'development')
    result = {}
    for side in (0, 1):
        ids = np.flatnonzero(fold == side)
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        if ACTIVE_FIT:
            mask = a['active'][i] | a['active'][j]
            i, j = i[mask], j[mask]
        y = groups[i] == groups[j]
        if y.sum() < 20:
            raise RuntimeError('Insufficient independent companion calibration support')
        keys = np.array([':'.join(sorted((x, z))) for x, z in zip(groups[i], groups[j])])
        w = pd.Series(keys).map(1 / pd.Series(keys).value_counts()).to_numpy()
        w[y] *= .5 / w[y].sum()
        w[~y] *= .5 / w[~y].sum()
        result[side] = (features(a, i, j), y, w, i, j)
    assert not set(groups[fold == 0]) & set(groups[fold == 1])
    assert not set(meta.noise_bank_index[fold == 0]) & set(meta.noise_bank_index[fold == 1])
    return result


def infer(f, config):
    original = isolated.ORIGINALS[config['deployment'], config['seed']]
    base = co.BASE_INFER(f, {**original, 'method': METHODS[0]})
    if config['method'] == METHODS[0]:
        return base
    z, oo, cl = r.apply_classifier(f[list(FEATURES)].to_numpy(float), config['intrinsic_calibrator'])
    cos, cood, cclip = n.apply_model(f, original['cosine_calibration'])
    output = cos + z
    if config['active_only']:
        active = f.profile_pair_active.to_numpy(bool)
        output, oo, cl = [np.where(active, new, old) for new, old in
                          zip((output, oo | cood, cl | cclip), base)]
        if not np.array_equal(output[~active], base[0][~active]):
            raise RuntimeError('Inactive score differs from immutable NODUP')
    else:
        oo, cl = oo | cood, cl | cclip
    return output, oo, cl


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for folder in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-PROFILE-HIERARCHICAL-CALIBRATION-13', 'UTC': n.utc(), 'status': n.STATUS,
        'parent_predictions': str(co.PARENT), 'goal_achieved': False, 'adaptive_development': True,
        'motivation': 'One scalar joint BC conflates well-resolved mass disagreement and weakly measured eta/chi agreement. Preserve marginal and conditional components as features of ONE classifier.',
        'identity': 'BC_joint=BC_mass * conditional_overlap;the conditional term is weighted over common mass,not an independent second mass evidence.',
        'features': list(FEATURES), 'methods': list(METHODS),
        'prediction_changed': False, 'parent_profile_quality_frozen': True,
        'fit': 'Source/noise-disjoint development fold0;per-source-pair weighting,balanced class prior. At least20 distinct positive sources per fold. NoPE or official input.',
        'selection': 'Proper balanced logloss on development fold1 for ridge .001,.01,.1,1 and monotone logistic or 3/7-leaf HistGB100iters. No ranking-based beta search.',
        'score': 'Zcos + ONE calibrated joint-reliability LR;beta=1 for all new models,bothruns. Replaces old new-joint tail/increment,not adds to it.',
        'active_controls': 'Only actual profile-active pairs use newcalibrator;inactive use independent immutable NODUP config. Global controls apply to all pairs.',
        'OOD': 'Fit feature box;positive reward0 outside;score cap log(nfittrue+1). Not physical Bayes evidence.',
        'frozen': ['encoder', 'profile probability', 'eta/chi conditional prediction', 'time', 'sky', 'outer weights', 'scope', 'old results', 'paper'],
        'forbidden': ['old encoder Mc/q scores', 'old/new total blending', 'event-specific veto', 'PE/official weight tuning'],
        'real_audit': 'All prespecified arms reported;both NODUP and historical PATH875 comparisons;no claim of blind confirmation.'})
    inputs = pd.read_csv(co.PARENT / 'manifest/INPUT_SHA256.csv').to_dict('records')
    files = set((co.PARENT / 'predictions').rglob('*.npz'))
    files.update((co.PARENT / 'cache').glob('gwtc*/*.npz'))
    files.add(co.PARENT / 'calibration/PROFILE_PREDICTIVE.json')
    for path in sorted(files):
        inputs.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', inputs)
    shutil.copy2(__file__, ROOT / 'scripts/profile_hierarchical_calibration.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


def calibrate():
    global ACTIVE_FIT
    configs, audits = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            original = isolated.ORIGINALS[dep, seed]
            configs.append({**original, 'method': METHODS[0]})
            f = load_panel(dep, seed, 'validation')
            zbase = infer(f, configs[-1])[0]
            bm = n.cf.fast_metrics(f, n.cf.channels(f, zbase) @ np.asarray(original['weights']))
            bwm = n.cf.fast_metrics(f, zbase)
            for active in (False, True):
                ACTIVE_FIT = active
                # Separate output namespace prevents pooling the active/global fit.
                r.ROOT = ROOT / ('active_fit' if active else 'global_fit')
                for kind in ('LINEAR', 'TREE'):
                    spec = r.fit_classifier(dep, seed, kind)
                    spec['file'] = str(Path('active_fit' if active else 'global_fit') / spec['file'])
                    r.ROOT = ROOT
                    method = ('PROFILE-HIER-ACTIVE-' if active else 'PROFILE-HIER-') + kind
                    config = {**original, 'method': method, 'intrinsic_calibrator': spec, 'active_only': active, 'beta': 1.}
                    z, oo, cl = infer(f, config)
                    metrics = n.cf.fast_metrics(f, n.cf.channels(f, z) @ np.asarray(original['weights']))
                    wm = n.cf.fast_metrics(f, z)
                    config.update(tune_guard=n.cf.guard(metrics, bm) and n.cf.guard(wm, bwm), tune_metrics=metrics)
                    poisoned = f.copy()
                    poisoned['pe_mc_bhattacharyya_coefficient'] = -1234.
                    poisoned['official_po_fpp'] = 1.
                    poisoned['pair_key'] = 'ignored'
                    if not np.array_equal(z, infer(poisoned, config)[0]):
                        raise RuntimeError('Forbidden metadata influenced waveform score')
                    configs.append(config)
                    audits.append({'deployment': dep, 'seed': seed, 'method': method, 'tune_guard': config['tune_guard'],
                        'fit_sources': spec['fit_sources'], 'tune_sources': spec['tune_sources'], 'calibration_logloss': spec['selected']['logloss'],
                        'forbidden_input_delta': 0., **metrics})
                    print('HIERARCHICAL_SELECTION', dep, seed, method, spec['selected'], config['tune_guard'], flush=True)
                    r.ROOT = ROOT / ('active_fit' if active else 'global_fit')
    r.ROOT = ROOT
    n.write_csv(ROOT / 'tables/HIERARCHICAL_CALIBRATION.csv', audits)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_or_test_selection': True, 'no_total_alpha': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
    co.score.METHODS = METHODS
    co.score.matrices = co.matrices
    r.install()
    original_export = n.public_frame
    original_consensus = n.dev.BASE.consensus_real
    def export(frame, z, weights, method):
        out = original_export(frame, z, weights, method)
        for name in FEATURES:
            out[name] = frame[name].to_numpy()
        return out
    def consensus(items, method):
        out = original_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(FEATURES)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus
    r.development = development
    n.METHODS, n.load_panel, n.infer = METHODS, load_panel, infer
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
