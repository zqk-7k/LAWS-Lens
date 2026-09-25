#!/usr/bin/env python3
"""Fixed NODUP coefficients with one newly calibrated joint consistency."""
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
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_predictive_scoring_20260909 as engine
base, n, r, co, h = engine.base, engine.n, engine.r, engine.co, engine.h
ROOT = DATA = PREDICTIVE = SELECTION = None
METHODS = ('NODUP-DIRECT-REPLAY', 'NEWPROFILE-JOINT-GLOBAL', 'NEWPROFILE-PAIRED-MIN')


def freeze():
    base.freeze()
    n.write_json(ROOT/'contracts/FIXED_JOINT_CONTROL.json', {'UTC': n.utc(),
        'id': 'MCWF-NODUP-SELECTED-PREDICTIVE-JOINT-34',
        'overrides_base': 'No seven-feature classifier. Refit only the original jointBC isotonic calibration and finite true-source tail on newfit512parents;retain all original gamma/beta and outerweights.',
        'predictive_selection_from': str(SELECTION), 'same_model_selection_both_runs': True,
        'requirement': 'R29 both-run predictiveGate andR33predeclaredcommonkindselection mustcompletefirst;copyselectedkind withhash,neverchooseusingrealresults.',
        'arms': list(METHODS), 'GLOBAL': 'OneupdatedjointBC instead of parent jointBC,plus unchanged cosine calibration;not oldMc/q heads.',
        'PAIRED_MIN': 'Only when both endpoints pass frozenR10profilequality,take lower of parent and updated waveform scores;otherwise exactparent. Correlated conservative ranking heuristic,not a proper Bayes factor.',
        'tail': 'Sameoriginal.05true-tail,finite ranktail,incrementclip[-4,4],positiveOOD0,endpointOODneutral;no newlysearchedthreshold.',
        'fit': 'Newsourcefold0 only,source-pairtotalweight1,balancedclasses,excludeidentical-noisebanknulls;no tune/test/real fit.',
        'why_control': 'Isolate predictive/error calibration changes from changes in the seven-feature classifier;results cannot be attributed to encoderretraining.',
        'frozen': ['allmodels', 'time', 'sky', 'outerweights', 'gamma', 'beta', 'scope', 'oldranks'],
        'no_old_Mc_q': True, 'no_total_blend': True, 'no_real_selection': True,
        'integration_limit': 'Same native512massbins asR33;continuous quadrature contrast remainsR31.',
        'adaptive_development': True, 'status': n.STATUS, 'script_sha256': n.sha(Path(__file__))})
    shutil.copy2(__file__, ROOT/'scripts/selected_predictive_joint_control.py')


def prepare():
    receipt = json.loads((SELECTION/'contracts/PREDICTIVE_KIND_FROZEN.json').read_text())
    src = SELECTION/receipt['file']
    if n.sha(src) != receipt['sha256']:
        raise RuntimeError('Predictive selection changed')
    dest = ROOT/'configs/SELECTED_PREDICTIVE_KIND.json'
    if dest.exists():
        if n.sha(dest) != receipt['sha256']:
            raise RuntimeError('Copied predictive selection mismatch')
        return
    shutil.copy2(src, dest)
    n.write_json(ROOT/'contracts/PREDICTIVE_KIND_FROZEN.json', {'UTC': n.utc(),
        'file': str(dest.relative_to(ROOT)), 'sha256': n.sha(dest), 'source': str(src)})


def panel(dep, seed, split, catalog=None):
    frame = engine.panel(dep, seed, split, catalog)
    frame['joint_logbc'] = frame.profile_log_mass_BC + frame.profile_log_conditional_BC
    frame['joint_BC'] = np.exp(frame.joint_logbc)
    return frame


def infer(frame, config):
    original = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    reference = co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    if config['method'] == METHODS[0]:
        return reference
    value = co.BASE_INFER(frame, config)
    if config['method'] == METHODS[1]:
        return value
    active = frame.profile_active_i.to_numpy(bool) & frame.profile_active_j.to_numpy(bool)
    mask = active & (value[0] < reference[0])
    result = tuple(np.where(mask, a, b) for a, b in zip(value, reference))
    if np.any(result[0] > reference[0]) or not np.array_equal(result[0][~active], reference[0][~active]):
        raise RuntimeError('Conservative waveform invariant failed')
    return result


def calibrate():
    configs, audits, units = [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            original = h.isolated.ORIGINALS[dep, seed]
            fit = base.panels(dep, seed)[0]
            x, y, weight = fit[:3]
            logbc = x[:, 1]+x[:, 2]
            isotonic = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(logbc, y, sample_weight=weight)
            probability = isotonic.y_thresholds_.clip(1/(y.sum()+2), 1-1/(y.sum()+2))
            spec = {'knots': isotonic.X_thresholds_.tolist(),
                'loglr': (np.log(probability)-np.log1p(-probability)).tolist(),
                'minimum': float(logbc.min()), 'maximum': float(logbc.max()),
                'reference': np.sort(-logbc[y]).tolist(), 'fit_source_systems': int(y.sum()),
                'source_fold': 0, 'real_or_test_used': False}
            n.write_json(ROOT/f'calibration/{dep}_{seed}_JOINT.json', spec)
            frame = panel(dep, seed, 'validation')
            cbase = {**original, 'method': METHODS[0]}
            reference = infer(frame, cbase)[0]
            archived = pd.read_parquet(r.PRIOR/f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(reference, archived.waveform_score.to_numpy()):
                raise RuntimeError('NODUP waveform replay changed')
            configs.append(cbase)
            fm = n.cf.fast_metrics(frame, n.cf.channels(frame, reference)@np.asarray(original['weights']))
            wm = n.cf.fast_metrics(frame, reference)
            for method in METHODS[1:]:
                config = {**original, 'method': method, 'joint_calibration_subgrid': spec}
                z = infer(frame, config)[0]
                metric = n.cf.fast_metrics(frame, n.cf.channels(frame, z)@np.asarray(original['weights']))
                waveform = n.cf.fast_metrics(frame, z)
                config.update(tune_guard=n.cf.guard(metric, fm) and n.cf.guard(waveform, wm), tune_metrics=metric)
                poisoned = frame.copy()
                poisoned['pair_key'], poisoned['pe_mc_bhattacharyya_coefficient'], poisoned['official_po_fpp'] = 'unused', -999., 1.
                if not np.array_equal(z, infer(poisoned, {**config, 'alpha': -999., 'old_Mc_weight': 999.})[0]):
                    raise RuntimeError('Forbidden input effect')
                configs.append(config)
                audits.append({'deployment': dep, 'seed': seed, 'method': method,
                    'tune_guard': config['tune_guard'], **metric})
                units.append({'deployment': dep, 'seed': seed, 'method': method,
                    'baseline_difference': 0., 'forbidden_input_delta': 0., 'gamma_beta_weights_unchanged': True})
            base.MEMO.clear()
    n.write_csv(ROOT/'tables/JOINT_VALIDATION_AUDIT.csv', audits)
    n.write_csv(ROOT/'audit/INVARIANCE.csv', units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_test_selection': True, 'fixed_gamma_beta_weights': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--predictive-root', type=Path, required=True)
    parser.add_argument('--selection-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'prepare', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT, DATA, PREDICTIVE, SELECTION = args.root, args.data_root, args.predictive_root, args.selection_root
    engine.ROOT, engine.DATA, engine.PREDICTIVE = ROOT, DATA, PREDICTIVE
    base.ROOT, base.DATA = ROOT, DATA
    base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = ROOT
    r.install()
    co.score.matrices = co.matrices
    base.population = engine.population
    n.METHODS, n.load_panel, n.infer = METHODS, panel, infer
    if args.stage in ('freeze', 'prepare', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
