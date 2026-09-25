#!/usr/bin/env python3
"""Simulation-only joint calibration of correlated waveform measurements."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_classifier_20260909 as base
n = base.n
DATA = P/'results/mcwf_nodup_independent_bulk_development_22b_20260909T114300Z'
ARMS = ('SHARED-PROFILE-DEFICIT', 'JOINT-NN-PROFILE-FEATURES')


def fit(x, y, w, ridge):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    if not np.isfinite(x).all() or np.any(scale <= 0):
        raise RuntimeError('Invalid fit feature')
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    def objective(coef):
        z = design @ coef
        prob = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        gradient = design.T @ (w * (prob-y))
        gradient[1:] += ridge * coef[1:]
        return base.loss(y, z, w) + .5 * ridge * np.dot(coef[1:], coef[1:]), gradient
    bounds = [(None, None), (0., None), (None, None), (0., None)]
    if x.shape[1] == 4:
        bounds.append((0., None))
    result = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success:
        raise RuntimeError('Classifier did not converge: ' + str(result.message))
    return {'mean': mean.tolist(), 'scale': scale.tolist(),
        'coefficients': result.x.tolist(), 'ridge': ridge,
        'optimizer_message': str(result.message)}


def main(root, pilot):
    if root.exists():
        raise RuntimeError('Independent pilot output required')
    for name in ('contracts', 'tables', 'scripts', 'audit', 'manifest', 'reports'):
        (root/name).mkdir(parents=True)
    contract = {'UTC': n.utc(), 'id': 'MCWF-NODUP-JOINT-WAVEFORM-FEATURES-52-PILOT',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'motivation': 'R51 discards the existing learned joint-density information on active pairs; its averaged AP improved but some catalog tail guards failed. Test complementary measurements jointly, not an additive pair of LRs.',
        'data': str(DATA), 'pilot': str(pilot), 'arms': list(ARMS),
        'features': ['embedding cosine', 'log minimum independent projection power',
                     '-log1p shared deficit', 'optional log learned joint Mc/eta/chi BC'],
        'learned_joint_definition': 'Unmodified parent.npz joint probability, normalized once per event in float64. No public PE.',
        'one_waveform_score': 'One class-balanced conditional logistic logit. Correlated waveform summaries are fitted together; no independence or physical Bayes-factor claim.',
        'coefficients': 'cosine and both consistency slopes >=0; strength and intercept free',
        'ridges': list(base.RIDGES), 'fit_fold': 0, 'selection_fold': 1,
        'fit_and_select': 'True + random null only; same source endpoint weights as R50; ridge by tune balanced logloss, exact ties stronger ridge.',
        'hard_null': 'Diagnostic only, does not select ridge.',
        'gate': 'New mean tune proper logloss nonworse than exact R50 replay in random AND hard null separately, both runs; at least one strict gain. Otherwise no catalog scoring.',
        'old_Mc_q_heads_or_scores': False, 'total_blend': False,
        'real_PE_or_official_selection': False,
        'unchanged': ['encoder checkpoints', 'time', 'sky', 'outer weights', 'all old results'],
        'not_new_blind_confirmation': True}
    n.write_json(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    shutil.copy2(__file__, root/'scripts/shared_joint_feature_pilot.py')
    n.write_json(root/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(root/'contracts/ANALYSIS_CONTRACT.json'),
        'runtime_sha256': n.sha(Path(__file__))})
    frame = pd.read_parquet(pilot/'tables/PAIR_RESULTS.parquet')
    old_specs = json.loads((pilot/'contracts/SELECTED_PILOT_CLASSIFIERS.json').read_text())
    old_predictions = pd.read_parquet(pilot/'tables/CLASSIFIER_PREDICTIONS.parquet')
    if len(frame) != 480 or frame.status.ne('COMPLETE').any():
        raise RuntimeError('Complete frozen R50 pair set required')
    inputs = [pilot/'tables/PAIR_RESULTS.parquet', pilot/'contracts/SELECTED_PILOT_CLASSIFIERS.json',
              pilot/'tables/CLASSIFIER_PREDICTIONS.parquet', Path(base.__file__)]
    metrics, grids, predictions, normalizations = [], [], [], []
    configs = {}
    for dep in n.DEPS:
        f = frame[frame.deployment.eq(dep)].reset_index(drop=True)
        train = f.fold.eq(0) & f.kind.ne('hard_null')
        tune = f.fold.eq(1) & f.kind.ne('hard_null')
        y, raw = base.base_weights(f[train]); w = base.balanced(y, raw)
        vy, vr = base.base_weights(f[tune]); vw = base.balanced(vy, vr)
        i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
        for seed in n.SEEDS:
            slot = n.recipes()[dep, seed]['slot']
            epath = DATA/f'predictions/{dep}/{seed}_embedding.npy'
            path = DATA/f'predictions/{dep}/{slot}_parent.npz'
            inputs.extend([epath, path])
            receipt = json.loads(path.with_suffix('.json').read_text())
            if n.sha(path) != receipt['sha256']:
                raise RuntimeError('Learned joint data hash mismatch')
            embedding = np.load(epath).astype(float)
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            cosine = np.einsum('ij,ij->i', embedding[i], embedding[j])
            with np.load(path) as bank:
                joint = bank['joint'].astype(float).reshape(len(embedding), -1)
            sums = joint.sum(1)
            if not np.isfinite(joint).all() or joint.min() < 0 or np.any(sums <= 0):
                raise RuntimeError('Invalid learned joint probability')
            normalizations.append({'deployment': dep, 'seed': seed,
                'raw_normalization_max_error': float(abs(sums-1).max()),
                'probability_cells': joint.shape[1]})
            joint = np.sqrt(joint / sums[:, None])
            bc = np.asarray([np.dot(joint[a], joint[b]) for a, b in zip(i, j)]).clip(1e-300, 1.)
            x0 = np.column_stack([cosine, np.log(f.minimum_independent_power), -np.log1p(f.deficit)])
            for arm in ARMS:
                x = x0 if arm == ARMS[0] else np.column_stack([x0, np.log(bc)])
                if arm == ARMS[0]:
                    selected = old_specs[f'{dep}/{seed}/{arm}']
                else:
                    options = []
                    for ridge in base.RIDGES:
                        spec = fit(x[train], y, w, ridge)
                        value = base.loss(vy, base.predict(x[tune], spec), vw)
                        options.append((value, -ridge, spec))
                        grids.append({'deployment': dep, 'seed': seed, 'arm': arm,
                            'ridge': ridge, 'tune_random_logloss': value})
                    selected = min(options, key=lambda a: (a[0], a[1]))[2]
                configs[f'{dep}/{seed}/{arm}'] = selected
                z = base.predict(x, selected)
                if arm == ARMS[0]:
                    old = old_predictions[(old_predictions.deployment == dep) &
                        (old_predictions.seed == seed) & (old_predictions.arm == arm)]
                    expected = old.set_index('pair_id').loc[f.pair_id, 'logit'].to_numpy()
                    if not np.allclose(z, expected, rtol=0, atol=1e-10):
                        raise RuntimeError('R50 classifier replay changed')
                for kind in ('random_null', 'hard_null'):
                    take = f.fold.eq(1) & f.kind.isin(['true', kind])
                    yy, rr = base.base_weights(f[take]); ww = base.balanced(yy, rr)
                    metrics.append({'deployment': dep, 'seed': seed, 'arm': arm,
                        'null_kind': kind, 'balanced_logloss': base.loss(yy, z[take], ww)})
                out = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j']].copy()
                out['seed'] = seed; out['arm'] = arm; out['logit'] = z
                out['learned_joint_BC'] = bc
                predictions.extend(out.to_dict('records'))
            print('JOINT_FEATURE_PILOT', dep, seed, flush=True)
    metric = pd.DataFrame(metrics)
    summary = metric.groupby(['deployment', 'arm', 'null_kind']).balanced_logloss.agg(['mean', 'std']).reset_index()
    comparison = summary.pivot(index=['deployment', 'null_kind'], columns='arm', values='mean')
    comparison['new_minus_shared'] = comparison[ARMS[1]] - comparison[ARMS[0]]
    passed = bool((comparison.new_minus_shared <= 1e-12).all() and (comparison.new_minus_shared < -1e-12).any())
    n.write_csv(root/'tables/PILOT_METRICS_PER_SEED.csv', metrics)
    n.write_csv(root/'tables/PILOT_SUMMARY.csv', summary)
    n.write_csv(root/'tables/PILOT_GATE_COMPARISON.csv', comparison.reset_index())
    n.write_csv(root/'tables/RIDGE_GRID.csv', grids)
    n.write_csv(root/'audit/LEARNED_JOINT_NORMALIZATION.csv', normalizations)
    pd.DataFrame(predictions).to_parquet(root/'tables/CLASSIFIER_PREDICTIONS.parquet', index=False)
    n.write_json(root/'contracts/SELECTED_CLASSIFIERS.json', configs)
    n.write_csv(root/'manifest/INPUT_SHA256.csv', [{'path': str(path), 'sha256': n.sha(path)}
        for path in sorted(set(inputs))])
    n.write_json(root/'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'PASS' if passed else 'FAIL',
        'goal_achieved': False, 'catalog_scoring_allowed': passed,
        'real_or_catalog_outcomes_read': False, 'status': n.STATUS})
    print(comparison.to_string(), flush=True)
    print('JOINT_FEATURE_PILOT_GATE', 'PASS' if passed else 'FAIL', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.pilot_root)
