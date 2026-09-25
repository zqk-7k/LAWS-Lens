#!/usr/bin/env python3
"""R61: a single waveform classifier regularized toward a simulation reference."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_deficit_coordinate_20260909 as common
n = common.n
ARMS = ('REFERENCE-REGULARIZED-3D', 'REFERENCE-REGULARIZED-4D')
RIDGES = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1, .3, 1., 3., 10., 30., 100.)


def transported_reference(mean, scale, reference):
    slopes = np.zeros(len(mean))
    old = np.asarray(reference['coefficients'])
    slopes[:3] = old[1:] / np.asarray(reference['scale'])
    intercept = old[0] - np.dot(slopes[:3], reference['mean'])
    return np.r_[intercept + np.dot(slopes, mean), slopes * scale]


def objective(beta, design, y, weights, center, ridge):
    z = design @ beta
    probability = 1. / (1. + np.exp(-np.clip(z, -700., 700.)))
    difference = beta - center
    value = common.classifier.base.loss(y, z, weights) + .5 * ridge * np.dot(difference, difference)
    gradient = design.T @ (weights * (probability - y)) + ridge * difference
    return value, gradient


def fit_model(x, y, weights, reference, ridge):
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean)**2, axis=0, weights=weights))
    if not np.isfinite(x).all() or np.any(scale <= 0):
        raise RuntimeError('Invalid training features')
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    center = transported_reference(mean, scale, reference)
    bounds = [(None, None), (0., None), (None, None), (0., None)]
    if x.shape[1] == 4:
        bounds.append((0., None))
    result = minimize(objective, center.copy(), args=(design, y, weights, center, ridge),
        jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Reference-regularized optimizer failed: ' + str(result.message))
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficients': result.x.tolist(),
        'ridge': ridge, 'regularization_center': center.tolist(),
        'penalizes_intercept_and_slopes': True, 'optimizer_message': str(result.message),
        'reference_deviation_norm': float(np.linalg.norm(result.x - center)),
        'no_total_score_blend': True}


def unit(reference):
    rng = np.random.default_rng(2026090961)
    checks = []
    for dimension in (3, 4):
        x = rng.normal(size=(50, dimension))
        mean, scale = rng.normal(size=dimension), np.exp(rng.normal(size=dimension))
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        center = transported_reference(mean, scale, reference)
        original = common.classifier.base.predict(x[:, :3], reference)
        error = float(abs(design @ center - original).max())
        if error > 1e-12:
            raise RuntimeError('Reference coordinate transport changed prediction')
        beta = center + rng.normal(0., .1, len(center))
        y = rng.integers(0, 2, size=len(x)).astype(float)
        weights = common.classifier.base.balanced(y, np.ones(len(y)))
        value, analytic = objective(beta, design, y, weights, center, .03)
        numerical = []
        for k in range(len(beta)):
            dx = np.eye(len(beta))[k] * 1e-5
            numerical.append((objective(beta + dx, design, y, weights, center, .03)[0] -
                              objective(beta - dx, design, y, weights, center, .03)[0]) / 2e-5)
        gradient_error = float(abs(np.asarray(numerical) - analytic).max())
        if gradient_error > 1e-7 or not np.isfinite(value):
            raise RuntimeError('Regularization gradient finite difference failed')
        checks.append({'dimension': dimension, 'reference_transport_max_error': error,
            'gradient_max_error': gradient_error, 'fourth_prior_coefficient_zero': dimension == 3 or center[-1] == 0})
    return checks


def freeze(root, expansion, reference):
    if root.exists():
        raise RuntimeError('Independent R61 output required')
    for name in ('contracts', 'configs', 'tables', 'audit', 'scripts', 'manifest', 'reports', 'logs'):
        (root / name).mkdir(parents=True)
    source = json.loads((expansion / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract = {'UTC': n.utc(), 'id': 'MCWF-REFERENCE-REGULARIZED-WAVEFORM-61',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'expansion': str(expansion), 'reference': str(reference), 'data': source['data'],
        'arms': ARMS, 'ridge_grid': RIDGES, 'exact_reference_fallback': True,
        'motivation': 'Expanded zero-centered ridge fits traded off ordinary and neighbor-null calibration in O3. Fit one classifier near the existing simulation-only classifier instead of pulling its coefficients toward zero.',
        'objective': 'HT/source-pair class-balanced fold0 cross entropy + ridge/2 * ||beta-beta_R55||^2, in fold0 standardized coordinates; penalize intercept and slopes. Fourth-feature reference coefficient0.',
        'not_a_score_mixture': 'Regularization is used during fitting only. Deployment evaluates ONE selected logistic classifier, not alpha*old_total+(1-alpha)*new_total and not a sum of old Mc/q scores.',
        'features': 'cosine, log minimum independent projection power, -log1p shared-fit deficit; optional fourth logBC of frozen learned Mc/eta/chi joint probability.',
        'reference_scope': 'R55 shared-profile classifier only; does not resurrect original encoder Mc/q regressors or PATH875 total score.',
        'unchanged': source['unchanged'] + ['R55 power/D supports and capped monotone policy', 'physical measurements'],
        'validation_selection': 'For each run/model/arm: keep finite-ridge candidates whose capped-policy full HT, ordinary-null and neighbor-null fold1 losses are each <= exact R55+1e-12. Include exact R55 as no-update option. Minimize full HT loss among these; ties prefer no update, then larger ridge.',
        'important_change_from_R59': 'Three fold1 diagnostics are now explicit validation selection constraints, not independent test evidence. This is a separately frozen algorithm, not a reclassification of prior failed arms.',
        'pilot_gate': 'Same six run/diagnostic means must be nonworse than R55 and at leastone strictgain. No-update fallback alone is not improvement.',
        'common_arm_selection': 'Among passing3D/4D arms: largest worst-run relative full HT gain, then mean gain, then3D. Same selected dimension and algorithm O3/O4a; coefficients and ridge remain per-run/per-model.',
        'future_catalog_and_external_gates': 'Exactly inherited R55 per-model/per-catalog waveform and fusion injection guards; consensus and each model Top10/20 PE/official budgets. No changes to their thresholds.',
        'statistical_limits': 'Reference and expanded fits share development population; this is not an independent Bayesian prior or posterior. Repeatedly used validation and catalogs imply adaptive development, not new locked confirmation.',
        'on_fail': 'Retain results; do not score real/catalog data for failed pilot. Finite predeclared grid only.',
        'real_PE_official_or_oracle_rank_used_for_selection': False,
        'no_encoder_training_or_Hanabi': True,
        'references': [
            {'url': 'https://proceedings.mlr.press/v80/li18a.html',
             'relation': 'L2-SP motivates penalizing departures from a reference model instead of zero.',
             'limitation': 'That work studies CNN transfer. This is an explicitly tested analogue for a small waveform classifier, not a reproduction or guarantee of GW performance.'},
            {'url': 'https://www.jmlr.org/papers/v11/cawley10a.html',
             'relation': 'Validation model-selection bias; no independent confirmation claim.'}]}
    common.write_once(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    paths = [Path(__file__), Path(common.__file__), Path(common.classifier.__file__),
        Path(common.classifier.base.__file__), Path(common.boundary.__file__),
        expansion / 'contracts/ANALYSIS_CONTRACT.json', expansion / 'contracts/PAIR_PLAN.parquet',
        expansion / 'tables/PAIR_RESULTS.parquet', expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json',
        reference / 'configs/SELECTED_CONFIGURATIONS.json', reference / 'audit/REFERENCE_AUDIT_FROZEN.json']
    data = Path(source['data'])
    for dep in n.DEPS:
        paths.extend([data / f'data/{dep}/event_metadata.parquet', data / f'data/{dep}/noise/noise_manifest.csv'])
        for seed in n.SEEDS:
            slot = n.recipes()[dep, seed]['slot']
            paths.extend([data / f'predictions/{dep}/{seed}_embedding.npy', data / f'predictions/{dep}/{slot}_parent.npz'])
    n.write_csv(root / 'manifest/INPUT_SHA256.csv', [{'path': str(p), 'sha256': n.sha(p)} for p in sorted(set(paths))])
    refs = json.loads((expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())
    common.write_once(root / 'audit/OBJECTIVE_AND_TRANSPORT_UNITS.json', unit(refs[0]['shared_classifier']))
    shutil.copy2(__file__, root / 'scripts/shared_reference_regularization.py')
    common.write_once(root / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'runtime_sha256': n.sha(Path(__file__)), 'contract_sha256': n.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'input_manifest_sha256': n.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('REFERENCE_REGULARIZATION_FROZEN', root, flush=True)


def check(root):
    record = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for name, path in [('runtime_sha256', Path(__file__)), ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                       ('input_manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if record[name] != n.sha(path):
            raise RuntimeError('Frozen R61 protocol changed')
    for r in pd.read_csv(root / 'manifest/INPUT_SHA256.csv').itertuples():
        if n.sha(Path(r.path)) != r.sha256:
            raise RuntimeError('Frozen input changed: ' + r.path)


def fit(root):
    check(root)
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('R61 pilot already complete')
    start = time.monotonic()
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data, expansion = (Path(contract[k]) for k in ('data', 'expansion'))
    pairs = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    refs = {(r['deployment'], r['seed']): r for r in json.loads((expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    rows, grids, predictions, selections, units, isolation = [], [], [], {}, [], []
    for dep in n.DEPS:
        meta = pd.read_parquet(data / f'data/{dep}/event_metadata.parquet')
        noise = pd.read_csv(data / f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        parents = meta.noise_bank_index.map(noise.parent_file_gps)
        if (set(meta.source_uid[meta.fold == 0]) & set(meta.source_uid[meta.fold == 1]) or
                set(parents[meta.fold == 0]) & set(parents[meta.fold == 1])):
            raise RuntimeError('Source/noise-parent overlap')
        isolation.append({'deployment': dep, 'source_overlap': 0, 'noise_parent_overlap': 0})
        f = pairs[pairs.deployment == dep].reset_index(drop=True)
        ref = refs[dep, n.SEEDS[0]]
        f = f[f.minimum_independent_power.between(*ref['minimum_power_support']) & f.any_shared_converged].reset_index(drop=True)
        train, tune = f.fold.eq(0).to_numpy(), f.fold.eq(1).to_numpy()
        y = f.kind.eq('true').to_numpy(float)
        fitw = common.classifier.base.balanced(y[train], f.HT_weight.to_numpy()[train])
        power, deficit = f.minimum_independent_power.to_numpy(float), f.deficit.to_numpy(float)
        masks = {'full_population': tune, 'random_draw': tune & f.kind.ne('hard_null').to_numpy(),
                 'neighbor_population': tune & (f.neighbor_population.to_numpy() | (y == 1))}
        weights = {name: common.classifier.base.balanced(y[mask],
            (f.HT_weight if name == 'full_population' else f.source_pair_weight).to_numpy()[mask])
            for name, mask in masks.items()}
        def losses(z):
            return {name: common.classifier.base.loss(y[mask], z[mask], weights[name]) for name, mask in masks.items()}
        for seed in n.SEEDS:
            ref = {**refs[dep, seed], 'coordinate': 'log1p', 'features': 3}
            e = np.load(data / f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            cosine = np.einsum('ij,ij->i', e[f.idx_i], e[f.idx_j])
            slot = n.recipes()[dep, seed]['slot']
            with np.load(data / f'predictions/{dep}/{slot}_parent.npz') as bank:
                joint = bank['joint'].astype(float).reshape(len(e), -1)
            mass = joint.sum(1)
            if not np.isfinite(joint).all() or np.any(joint < 0) or np.any(mass <= 0):
                raise RuntimeError('Invalid learned joint distribution')
            joint = np.sqrt(joint / mass[:, None])
            bc = np.asarray([np.dot(joint[int(i)], joint[int(j)]) for i, j in zip(f.idx_i, f.idx_j)]).clip(1e-300, 1.)
            base_z = common.policy(cosine, power, deficit, ref)
            base_loss = losses(base_z)
            controls = {'R55-FROZEN': base_z}
            for dimension, arm in zip((3, 4), ARMS):
                bj = bc if dimension == 4 else None
                x = common.features(cosine, power, deficit, 'log1p', bj)
                fallback = {**ref, 'features': 3, 'arm': arm, 'no_update': True,
                            'reference_regularization': None, 'selection_reason': 'exact reference fallback'}
                options = [(base_loss['full_population'], -float('inf'), fallback, base_z)]
                grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': None,
                    'no_update': True, 'validation_feasible': True, **base_loss})
                for ridge in RIDGES:
                    spec = fit_model(x[train], y[train], fitw, ref['shared_classifier'], ridge)
                    config = {**ref, 'shared_classifier': spec, 'features': dimension, 'arm': arm,
                              'no_update': False, 'reference_regularization': ridge}
                    z = common.policy(cosine, power, deficit, config, bj)
                    score = losses(z)
                    feasible = all(score[name] <= base_loss[name] + 1e-12 for name in masks)
                    grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': ridge,
                        'no_update': False, 'validation_feasible': feasible, **score})
                    if feasible:
                        options.append((score['full_population'], -ridge, config, z))
                _, _, winner, z = min(options, key=lambda a: (a[0], a[1]))
                selections[f'{dep}/{seed}/{arm}'] = winner
                controls[arm] = z
                lo, hi = ref['deficit_support']
                d = np.unique(np.r_[0., lo / 2, lo, np.nextafter(lo, np.inf), np.geomspace(lo, hi, 100), hi * 1.1])
                unit_z = common.policy(np.full(len(d), .8), np.full(len(d), np.sqrt(np.prod(ref['minimum_power_support']))),
                    d, winner, np.full(len(d), .5) if winner['features'] == 4 else None)
                if np.any(np.diff(unit_z) > 1e-10) or np.ptp(unit_z[d <= lo]) > 1e-10:
                    raise RuntimeError('Monotone D boundary policy changed')
                units.append({'deployment': dep, 'seed': seed, 'arm': arm, 'monotone': True,
                    'low_D_plateau': True, 'no_update_selected': winner['no_update']})
            for arm, z in controls.items():
                for diagnostic, value in losses(z).items():
                    rows.append({'deployment': dep, 'seed': seed, 'arm': arm, 'diagnostic': diagnostic,
                                 'logloss': value, 'pairs': int(masks[diagnostic].sum())})
                out = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j', 'HT_weight',
                    'source_pair_weight', 'neighbor_population', 'deficit', 'minimum_independent_power']].copy()
                out['seed'], out['arm'], out['waveform_score'], out['learned_joint_BC'] = seed, arm, z, bc
                predictions.extend(out.to_dict('records'))
            print('REFERENCE_REGULARIZATION_FIT', dep, seed,
                {arm: selections[f'{dep}/{seed}/{arm}']['reference_regularization'] for arm in ARMS}, flush=True)
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(['deployment', 'arm', 'diagnostic']).logloss.agg(['mean', 'std']).reset_index()
    compare = summary.pivot(index=['deployment', 'diagnostic'], columns='arm', values='mean')
    passing = []
    for arm in ARMS:
        delta = compare[arm] - compare['R55-FROZEN']
        compare[arm + '_delta'] = delta
        if (delta <= 1e-12).all() and (delta < -1e-12).any():
            gain = (-delta / compare['R55-FROZEN']).xs('full_population', level='diagnostic')
            passing.append({'arm': arm, 'worst_run_relative_gain': float(gain.min()), 'mean_relative_gain': float(gain.mean())})
    chosen = min(passing, key=lambda a: (-a['worst_run_relative_gain'], -a['mean_relative_gain'], ARMS.index(a['arm']))) if passing else None
    n.write_csv(root / 'tables/CALIBRATION_METRICS_PER_SEED.csv', metrics)
    n.write_csv(root / 'tables/CALIBRATION_SUMMARY.csv', summary)
    n.write_csv(root / 'tables/PILOT_GATE_COMPARISON.csv', compare.reset_index())
    n.write_csv(root / 'tables/REGULARIZATION_VALIDATION_GRID.csv', grids)
    n.write_csv(root / 'audit/MONOTONICITY_UNITS.csv', units)
    n.write_csv(root / 'audit/FOLD_ISOLATION.csv', isolation)
    pd.DataFrame(predictions).to_parquet(root / 'tables/PREDICTIONS.parquet', index=False)
    common.write_once(root / 'configs/ALL_SELECTED_CLASSIFIERS.json', selections)
    common.write_once(root / 'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'PASS' if chosen else 'FAIL',
        'passing_arms': passing, 'selected_common_arm': chosen, 'goal_achieved': False, 'status': n.STATUS,
        'no_real_or_catalog_inputs_used': True, 'full_goal_not_implied': True, 'seconds': time.monotonic() - start})
    print(compare.to_string(), flush=True)
    print('REFERENCE_REGULARIZATION_GATE', 'PASS' if chosen else 'FAIL', chosen, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expansion-root', type=Path)
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--stage', choices=('freeze', 'fit'), required=True)
    a = p.parse_args()
    if a.stage == 'freeze':
        freeze(a.root, a.expansion_root, a.reference_root)
    else:
        fit(a.root)
