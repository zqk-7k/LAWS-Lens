#!/usr/bin/env python3
"""Put an active-only waveform replacement on a frozen null-score scale."""
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
import mcwf_profile_single_waveform_20260909 as s
h, co, r, n = s.h, s.co, s.r, s.n
ROOT = None
PARENT = P / 'results/mcwf_nodup_single_waveform_14_20260909T095000Z'
METHODS = ('NODUP-DIRECT-REPLAY', 'NULL-ALIGNED-LINEAR', 'NULL-ALIGNED-TREE')
PARENT_CONFIGS = {(c['deployment'], c['seed'], c['method']): c for c in json.loads(
    (PARENT / 'configs/SELECTED_CONFIGURATIONS.json').read_text())}


def baseline_scores(dep, seed, split, x, i, j):
    slot = n.recipes()[dep, seed]['slot']
    input_seed = 0 if split == 'development' else seed
    path = n.JOINT / f'cache/CONDITIONAL-ETA-CHI/{dep}/{slot}_{input_seed}_{split}.npz'
    a = dict(np.load(path))
    spec = n.recipes()[dep, seed]['joint_config']['calibration']
    value, bc = a['joint_logbc'][i, j], a['joint_bc'][i, j]
    endpoint = (a['outside'][i] > .25) | (a['outside'][j] > .25)
    tail = n.ev.tail.tail_probability(-np.log(bc), spec['joint_reference'])
    penalty = np.minimum(np.log(tail / .05), 0.)
    outside = endpoint | (value < spec['minimum']) | (value > spec['maximum'])
    increment = np.interp(value, spec['knots'], spec['loglr'])
    increment = np.where(outside | (tail < .05), np.minimum(increment, 0.), increment).clip(-4., 4.)
    penalty[endpoint], increment[endpoint] = 0., 0.
    f = pd.DataFrame({'embedding_only': x[:, 0]})
    original = h.isolated.ORIGINALS[dep, seed]
    cos = n.apply_model(f, original['cosine_calibration'])[0]
    return cos + original['gamma'] * penalty + original['beta'] * increment


def weighted_cdf(values, weights):
    unique, inverse = np.unique(values, return_inverse=True)
    count = np.bincount(inverse, weights=weights)
    q = (np.cumsum(count) - .5 * count) / count.sum()
    return unique, q


def mapping(new, old, weights):
    x, q = weighted_cdf(new, weights)
    z, c = weighted_cdf(old, weights)
    y = np.interp(q, c, z)
    if np.any(np.diff(y) < -1e-12):
        raise RuntimeError('Null-scale mapping must be monotonic')
    return {'x': x.tolist(), 'y': y.tolist(), 'q': q.tolist(),
            'reference_cdf_x': z.tolist(), 'reference_cdf_q': c.tolist(),
            'minimum': float(x[0]), 'maximum': float(x[-1]),
            'definition': 'weighted mid-CDF then inverse reference null CDF; endpoints clipped, no extrapolation'}


def infer(frame, c):
    original = h.isolated.ORIGINALS[c['deployment'], c['seed']]
    base = co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    if c['method'] == METHODS[0]:
        return base
    new, oo, cl = r.apply_classifier(frame[list(s.FEATURES)].to_numpy(float), c['waveform_calibrator'])
    m = c['null_scale']
    output = np.interp(new, m['x'], m['y'])
    output[oo] = np.minimum(output[oo], 0.)
    active = frame.profile_pair_active.to_numpy(bool)
    out = tuple(np.where(active, a, b) for a, b in zip((output, oo, cl), base))
    if not np.array_equal(out[0][~active], base[0][~active]):
        raise RuntimeError('Inactive baseline score changed')
    return out


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports', 'scripts',
                 'logs', 'manifest', 'results', 'figures'):
        (ROOT / name).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-NULL-SCALE-15', 'UTC': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'parent': str(PARENT),
        'motivation': 'An active-only single-classifier LR and the immutable inactive ranking score have different null scales. This ablation aligns units without mixing pair-specific old and new scores.',
        'same_both_runs': True, 'methods': list(METHODS),
        'formula': 'Znew=Q_old_null(F_new_null(single_classifier_score));weighted midrank CDF and inverse CDF. Monotonic, deterministic, endpoints clipped.',
        'fit': 'Only active noncompanion pairs of source/noise-disjoint development fold0. One weight per unordered sourcepair. Both distributions use SAME pairs.',
        'reference': 'NODUP-DIRECT waveform null distribution only. No PATH875 alpha, old encoder mass/q heads, individual old pair score at inference or total-score mixture.',
        'selection': 'No grid or real-driven coefficient; source fold1 is a diagnostic. Both frozen LINEAR/TREE classifiers reported, not re-trained.',
        'score_meaning': 'Empirically unit-aligned waveform ranking score, NOT a log likelihood ratio after quantile transform. Time/sky unchanged.',
        'OOD': 'Inherited feature support; positive OOD reward0 after mapping. No upper-tail extrapolation.',
        'frozen': ['encoder', 'predictive densities', 'classifier files', 'outer weights', 'time', 'sky', 'scope', 'old results', 'paper'],
        'forbidden': ['PE/official labels in fit', 'event-specific penalty', 'old Mc/q score', 'total blend'],
        'real_audit': 'Adaptive external development only; all arms and injection losses reported. No automatic adoption.'})
    files = pd.read_csv(PARENT / 'manifest/INPUT_SHA256.csv').to_dict('records')
    files.extend({'path': str(p), 'sha256': n.sha(p), 'bytes': p.stat().st_size}
                 for p in sorted((PARENT / 'calibration').rglob('*.pkl')))
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', files)
    units = []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            f = h.load_panel(dep, seed, 'validation')
            actual = co.BASE_INFER(f, {**h.isolated.ORIGINALS[dep, seed], 'method': METHODS[0]})[0]
            replay = baseline_scores(dep, seed, 'validation', f[list(s.FEATURES)].to_numpy(float),
                                     f.idx_i.to_numpy(int), f.idx_j.to_numpy(int))
            diff = float(abs(actual - replay).max())
            if diff > 1e-12:
                raise RuntimeError('Reference null score reconstruction differs: ' + str(diff))
            units.append({'deployment': dep, 'seed': seed, 'baseline_reconstruction_max_error': diff})
    n.write_csv(ROOT / 'audit/NULL_REFERENCE_REPLAY_UNITS.csv', units)
    shutil.copy2(__file__, ROOT / 'scripts/profile_null_scale.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


def calibrate():
    configs, diagnostics = [], []
    s.ACTIVE_FIT = True
    for dep in n.DEPS:
        for seed in n.SEEDS:
            original = h.isolated.ORIGINALS[dep, seed]
            configs.append({**original, 'method': METHODS[0]})
            parts = s.development(dep, seed)
            x, y, w, i, j = parts[0]
            reference = baseline_scores(dep, seed, 'development', x, i, j)
            frame = h.load_panel(dep, seed, 'validation')
            zbase = infer(frame, configs[-1])[0]
            bm = n.cf.fast_metrics(frame, n.cf.channels(frame, zbase) @ np.asarray(original['weights']))
            bwm = n.cf.fast_metrics(frame, zbase)
            for kind in ('LINEAR', 'TREE'):
                parent = PARENT_CONFIGS[dep, seed, 'SINGLE-WF-ACTIVE-' + kind]
                spec = dict(parent['waveform_calibrator'])
                spec['file'] = str(PARENT / spec['file'])
                pred = r.apply_classifier(x, spec)[0]
                transform = mapping(pred[~y], reference[~y], w[~y])
                c = {**original, 'method': 'NULL-ALIGNED-' + kind, 'waveform_calibrator': spec,
                     'null_scale': transform, 'active_only': True}
                z, oo, cl = infer(frame, c)
                fm = n.cf.fast_metrics(frame, n.cf.channels(frame, z) @ np.asarray(original['weights']))
                wm = n.cf.fast_metrics(frame, z)
                c.update(tune_guard=n.cf.guard(fm, bm) and n.cf.guard(wm, bwm), tune_metrics=fm)
                bad = frame.copy()
                bad['pair_key'], bad['pe_mc_bhattacharyya_coefficient'], bad['official_po_fpp'] = 'ignored', -1., 1.
                if not np.array_equal(z, infer(bad, c)[0]):
                    raise RuntimeError('Forbidden metadata affected score')
                configs.append(c)
                diagnostics.append({'deployment': dep, 'seed': seed, 'method': c['method'],
                    'null_fit_rows': int((~y).sum()), 'monotonic_mapping': True, 'tune_guard': c['tune_guard'],
                    'classifier_retrained': False, 'forbidden_input_delta': 0., **fm})
                print('NULL_SCALE_FROZEN', dep, seed, kind, c['tune_guard'], flush=True)
    n.write_csv(ROOT / 'tables/NULL_SCALE_DIAGNOSTICS.csv', diagnostics)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json',
        'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'), 'real_test_input_used': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    a = p.parse_args()
    ROOT = s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = a.root
    r.install()
    co.score.METHODS, co.score.matrices = METHODS, co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, h.load_panel, infer
    old_export, old_consensus = n.public_frame, n.dev.BASE.consensus_real
    def export(frame, z, weights, method):
        out = old_export(frame, z, weights, method)
        for name in h.FEATURES:
            out[name] = frame[name].to_numpy()
        return out
    def consensus(items, method):
        out = old_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(h.FEATURES)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus
    if a.stage in ('freeze', 'calibrate'):
        globals()[a.stage]()
    else:
        n.run(ROOT, a.stage)
