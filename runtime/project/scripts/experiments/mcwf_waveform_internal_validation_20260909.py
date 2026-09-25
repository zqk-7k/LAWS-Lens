#!/usr/bin/env python3
"""Validation-only internal waveform calibration; frozen physical channels."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_confidence_level_validation_20260909 as cv
st, ctrl, engine, base, n, r, co, h = (getattr(cv, k) for k in
    ('st', 'ctrl', 'engine', 'base', 'n', 'r', 'co', 'h'))
ROOT = PARENT = None
METHODS = ('NODUP-DIRECT-REPLAY', 'JOINTSTATE-FROZEN95',
           'WAVEFORM-INTERNAL-VALIDATED95', 'WAVEFORM-INTERNAL-VALIDATED')
GAMMAS = (0., .5, 1., 2., 4., 8.)
BETAS = (.0625, .125, .25, .5, 1., 2., 4.)
FIELDS = st.FIELDS + ('waveform_confidence_alpha', 'waveform_internal_gamma',
                     'waveform_internal_beta')


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    cv.frozen_parent()
    for folder in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports',
                   'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {
        'UTC': n.utc(), 'id': 'MCWF-NODUP-WAVEFORM-INTERNAL-VALIDATION-39',
        'status': n.STATUS, 'parent': str(PARENT),
        'change': 'Only gamma, beta and empirical confidence alpha inside the waveform channel. No outer-fusion change.',
        'formula': 'Zwf=Zcos+gamma*min(log(p_true/alpha),0)+beta*bounded_joint_state_LR',
        'not_independent_evidence': 'LR and true-tail are two functions of ONE full joint mass/eta/spin overlap. This is a constrained ranking rule, NOT their product as independent Bayes factors.',
        'gamma_grid': list(GAMMAS), 'beta_grid': list(BETAS),
        'confidence_levels': list(cv.LEVELS),
        'methods': list(METHODS), 'fixed95_arm': 'alpha=.05; tune only gamma,beta',
        'free_level_arm': 'Choose alpha,gamma,beta jointly on archived validation only.',
        'same_framework_both_runs': True, 'per_seed_selection': True,
        'guard': 'Original .02R10/.005AP/10percentF50/F90 guard for BOTH waveform and fusion, against BOTH NODUP and PATH875.',
        'selection_priority': ['both-baseline guards', 'F50', 'F90', '-AP', '-R10', '-R1',
                               'distance to original gamma,beta,alpha=.05', 'lexicographic alpha,gamma,beta'],
        'no_passing_point': 'Retain failed best point and full grid as exploratory failure; never call upgrade.',
        'grid_deduplication': 'Exact duplicate vectors for waveform and fusion metrics merged before selection; deterministic closest-old tie rule.',
        'zero_gamma': 'Means no true-tail attenuation; the joint likelihood-ratio term remains nonzero. Report transparently.',
        'frozen': ['encoder', 'joint predictive density', 'cosine calibration', 'joint state calibration',
                   'outer waveform/time/sky weights', 'Ztime', 'Zsky', 'scope', 'historical results'],
        'no_old_Mc_q_heads': True, 'no_total_score_blend': True,
        'adaptive_real_feedback': True, 'blind_confirmation': False,
        'real_PE_official_not_inputs_or_selection': True,
        'references': ['https://arxiv.org/abs/1506.02169',
                       'https://www.jmlr.org/papers/v11/cawley10a.html'],
        'reference_limit': 'Calibration and evaluation discipline only; these papers do not prove this grid is optimal.'})
    paths = [PARENT/'configs/SELECTED_CONFIGURATIONS.json', Path(__file__), Path(cv.__file__), Path(st.__file__)]
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', [
        {'path': str(p), 'sha256': n.sha(p), 'bytes': p.stat().st_size} for p in paths])
    shutil.copy2(__file__, ROOT/'scripts/waveform_internal_validation.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def infer(frame, config):
    old = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    if config['method'] == METHODS[0]:
        return st.infer(frame, {**old, 'method': METHODS[0]})
    state = st.states(frame)
    cv.ALPHA = float(config['confidence_alpha'])
    tail, inc, penalty, outside = cv.terms(frame.joint_logbc.to_numpy(float), state,
        frame.joint_ood.to_numpy(bool), config['state_specs'])
    cosine, cos_ood, cos_clip = n.apply_model(frame, old['cosine_calibration'])
    gamma, beta = float(config['gamma']), float(config['beta'])
    z = cosine + gamma*penalty + beta*inc
    for key, value in {
        'actual_waveform_joint_BC': frame.joint_BC.to_numpy(),
        'new_prediction_used_in_score': True, 'profile_state': state,
        'profile_state_true_tail': tail, 'profile_state_loglr': inc,
        'profile_state_OOD': outside, 'waveform_confidence_alpha': cv.ALPHA,
        'waveform_internal_gamma': gamma, 'waveform_internal_beta': beta}.items():
        frame[key] = value
    return z, outside | cos_ood, cos_clip | (abs(inc) >= 4.)


def key(row):
    return (row['false_at_recall_0p5'], row['false_at_recall_0p9'],
            -row['average_precision'], -row['macro_r_at_10'], -row['macro_r_at_1'],
            row['distance'], row['confidence_alpha'], row['gamma'], row['beta'])


def calibrate():
    import hashlib
    parents = {(c['deployment'], c['seed']): c for c in cv.frozen_parent()
               if c['method'] == 'JOINTSTATE-GLOBAL'}
    configs, grid, selections, units = [], [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old = h.isolated.ORIGINALS[dep, seed]
            frame = cv.panel(dep, seed, 'validation')
            cref = {**old, 'method': METHODS[0]}
            ref = infer(frame, cref)[0]
            refs = ((n.cf.fast_metrics(frame, n.cf.channels(frame, ref)@np.asarray(old['weights'])),
                     n.cf.fast_metrics(frame, ref)),
                    (n.cf.fast_metrics(frame, frame.PATH875_final_score.to_numpy()),
                     n.cf.fast_metrics(frame, frame.PATH875_waveform.to_numpy())))
            fixed = {**deepcopy(parents[dep, seed]), 'method': METHODS[1], 'confidence_alpha': .05}
            archived = pd.read_parquet(PARENT/f'results/JOINTSTATE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(infer(frame, fixed)[0], archived.waveform_score.to_numpy()):
                raise RuntimeError('Frozen95 control replay failed')
            configs.extend([cref, fixed])
            unique, options = {}, []
            for alpha in cv.LEVELS:
                for gamma in GAMMAS:
                    for beta in BETAS:
                        config = {**deepcopy(fixed), 'confidence_alpha': alpha, 'gamma': gamma, 'beta': beta}
                        z = infer(frame, config)[0]
                        digest = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
                        if digest not in unique:
                            fm = n.cf.fast_metrics(frame, n.cf.channels(frame, z)@np.asarray(old['weights']))
                            wm = n.cf.fast_metrics(frame, z)
                            guard = all(n.cf.guard(fm, fref) and n.cf.guard(wm, wref) for fref, wref in refs)
                            unique[digest] = (fm, wm, guard)
                        fm, wm, guard = unique[digest]
                        distance = abs(gamma-old['gamma'])/4 + abs(np.log2(beta/old['beta'])) + abs(alpha-.05)/.05
                        row = {'deployment': dep, 'seed': seed, 'confidence_alpha': alpha,
                               'gamma': gamma, 'beta': beta, 'both_guard': guard, 'distance': distance,
                               'score_sha256': digest, **fm, **{'waveform_'+k:v for k,v in wm.items()}}
                        grid.append(row); options.append((row, config))
            for method, subset in ((METHODS[2], [v for v in options if v[0]['confidence_alpha'] == .05]),
                                    (METHODS[3], options)):
                passing = [v for v in subset if v[0]['both_guard']]
                row, config = min(passing or subset, key=lambda item: key(item[0]))
                config.update(method=method, tune_guard=bool(passing), tune_metrics=row)
                configs.append(deepcopy(config)); selections.append({'method': method, **row})
                z = infer(frame, config)[0]
                poison = frame.copy()
                poison['pair_key'], poison['pe_mc_bhattacharyya_coefficient'], poison['official_po_fpp'] = 'unused', -999., 1.
                if not np.array_equal(z, infer(poison, {**config, 'alpha': -999., 'old_Mc_weight': 999.})[0]):
                    raise RuntimeError('Forbidden score input effect')
                units.append({'deployment': dep, 'seed': seed, 'method': method,
                    'parent_replay_delta': 0., 'forbidden_input_delta': 0.,
                    'unique_grid_scores': len(unique), 'outer_weights_unchanged': True})
                print('INTERNAL_SELECTED', dep, seed, method, row['confidence_alpha'],
                      row['gamma'], row['beta'], bool(passing), flush=True)
    n.write_csv(ROOT/'tables/INTERNAL_VALIDATION_GRID.csv', grid)
    n.write_csv(ROOT/'tables/INTERNAL_VALIDATION_SELECTION.csv', selections)
    n.write_csv(ROOT/'audit/INVARIANCE.csv', units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json',
        'sha256': n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'), 'no_real_or_test_selection': True})


def main():
    global ROOT, PARENT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args(); ROOT, PARENT = args.root, args.parent
    cv.ROOT, cv.PARENT = ROOT, PARENT
    base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = ROOT
    engine.ROOT = PARENT
    r.install(); st.FIELDS = FIELDS; st.install_export()
    st.METHODS = (METHODS[0], 'GLOBAL-PARENT', 'UNUSED-REJECT')
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, cv.panel, infer
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)


if __name__ == '__main__':
    main()
