#!/usr/bin/env python3
"""R73: simulation-selected shared-Mc versus full-control conditional score."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_mass_population_20260910 as measurement
import mcwf_nodup_conditional_deficit_20260910 as base

n, io = base.n, base.population
ARMS = ('SHARED-MASS-CONDITIONAL', 'FULL-SHARED-NUMERICAL-CONTROL')
STATISTICS = {'SHARED-MASS-CONDITIONAL': 'D_mass',
              'FULL-SHARED-NUMERICAL-CONTROL': 'D_full_common_denominator'}


def apply(z, power, deficit, eligible, spec):
    value = np.asarray(z, float).copy()
    eligible = np.asarray(eligible, bool)
    outside = ~eligible
    clipped = np.zeros(len(z), bool)
    if eligible.any():
        score, ood, clip = base.apply(value[eligible], np.asarray(power)[eligible], np.asarray(deficit)[eligible], spec)
        value[eligible], outside[eligible], clipped[eligible] = score, ood, clip
    return value, outside, clipped


def freeze(root, measured, expansion, global_root, catalog):
    if root.exists():
        raise RuntimeError('Independent R73 root required')
    receipt = json.loads((measured / 'contracts/MEASUREMENT_GATE.json').read_text())
    if receipt['gate'] != 'PASS':
        raise RuntimeError('Complete population measurements must pass first')
    measurement.check(measured)
    data = Path(json.loads((measured / 'contracts/ANALYSIS_CONTRACT.json').read_text())['data'])
    for directory in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest'):
        (root / directory).mkdir(parents=True)
    contract = {'UTC': io.utc(), 'id': 'MCWF-SHARED-MASS-CONDITIONAL-CALIBRATION-73',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'measured': str(measured), 'expansion': str(expansion), 'global_population': str(global_root),
        'catalog_reference': str(catalog), 'data': str(data), 'arms': ARMS, 'statistic_columns': STATISTICS,
        'ridge_grid': base.RIDGES,
        'question': 'Does isolating shared-Mc compatibility improve simulation calibration over a full-shared control with the same numerical independent denominator?',
        'model': 'ONE logistic waveform classifier of current NODUP waveform, log original minimum power, minus log1p measured deficit. Ridge toward exact NODUP. Nonnegative NODUP and negative-deficit slopes. No added independent mass evidence and no total-score blend.',
        'control': 'Both arms use identical pairs, numeric common independent denominator, original power covariate and eligibility. Full-control includes numerical-search gains; it is not relabeled as mass-only physics.',
        'eligibility': 'Original R56 quality plus frozen original power support and any-shared-converged, AND R72 best-mass-has-converged-run. Identical for both arms and independent of truth/PE. Ineligible: EXACT NODUP, never drop from risk accounting.',
        'support': 'Per-model NODUP fit-active min/max; per-arm deficit[0,fit-active maximum]; unchanged original power support. Outside supports exact NODUP; cap16 on supported score. No post-hoc positive ceiling.',
        'fit': 'Source/noise-parent-disjoint fold0 only; exact R64 full source-pair multiplicity and HT weights, no rebalancing of eligible branch.',
        'selection': 'Fold1 full-source-weighted proper-risk contribution, fixed ridge grid plus exact identity. Ties1e-12: identity, then stronger ridge. Common arm among passing arms: smallest worst-run relative risk delta, then mean, then ARMS order.',
        'pilot_gate': 'Every model tune risk nonworse than NODUP, strictly improved model mean and source-bootstrap one-sided95% risk-delta upper<=0 in BOTH runs. A pilot PASS is not goal completion.',
        'bootstrap': {'draws': 2000, 'seed': 2026091073, 'unit': 'full source doublet',
            'limitations': 'Conditional on repeated development data and selected models; not independent locked-test or noise-design uncertainty.'},
        'future_goal': 'Unchanged every injection-panel waveform/fusion guards, consensus AND eachmodel Top10/20 PE/official nonloss, key outside every Top10. No real result used to select calibrator or thresholds.',
        'frozen': ['all encoders and learned joint Mc/eta/chi distributions', 'time', 'sky', 'outer weights',
                   'source/noise folds', 'strict scopes', 'all historical outputs'],
        'no_legacy_Mc_q_no_total_blend': True, 'no_catalog_real_or_PE_for_model_selection': True,
        'references': [{'url': 'https://arxiv.org/abs/gr-qc/9402014', 'use': 'Mass-spin correlation motivates the weaker necessary mass test; not proof of this approximation.'},
            {'url': 'https://arxiv.org/abs/1506.02169', 'use': 'Joint discriminative calibration of correlated features; not independent-evidence multiplication.'},
            {'url': 'https://www.jmlr.org/papers/v11/cawley10a.html', 'use': 'Repeated validation remains adaptive development, not confirmation.'}],
        'limitations': 'Both projection statistics are approximate, not PE or Bayes factors. Per-image SNR scaling and lens-environment reuse remain.'}
    io.write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    paths = [Path(__file__), Path(base.__file__), Path(io.__file__), Path(measurement.__file__),
        measured / 'contracts/MEASUREMENT_GATE.json', measured / 'tables/PAIR_RESULTS.parquet',
        measured / 'contracts/ANALYSIS_CONTRACT.json', expansion / 'tables/PAIR_RESULTS.parquet',
        expansion / 'tables/PLANNING_INVENTORY.csv', expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json',
        global_root / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet',
        base.BASE / 'configs/SELECTED_CONFIGURATIONS.json']
    paths += [Path(r.path) for r in pd.read_csv(global_root / 'manifest/INPUT_SHA256.csv').itertuples()]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            slot = n.recipes()[dep, seed]['slot']
            paths += [data / f'predictions/{dep}/{seed}_embedding.npy', data / f'predictions/{dep}/{slot}_parent.npz',
                base.replay.joint_path(dep, seed, 'validation'),
                base.replay.archive_path(catalog, 'NODUP-DIRECT-REPLAY', dep, seed, 'validation')]
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and str(P / 'scripts/experiments') in file and Path(file).suffix == '.py':
            paths.append(Path(file))
    io.snapshot(root, paths)
    io.write(root / 'audit/BASE_CALIBRATION_UNITS.json', base.units())
    shutil.copy2(__file__, root / 'scripts/shared_mass_conditional_calibration.py')
    io.write(root / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'runtime_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'manifest_sha256': io.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('SHARED_MASS_CALIBRATION_FROZEN', root, flush=True)


def check(root):
    freeze_record = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)), ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if io.sha(file) != freeze_record[key]:
            raise RuntimeError('Frozen file changed ' + str(file))
    inputs = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for row in inputs.itertuples():
        if io.sha(row.path) != row.sha256:
            raise RuntimeError('Protected input changed ' + row.path)
    return len(inputs)


def fit(root):
    count = check(root)
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Completed calibration immutable')
    start = time.monotonic()
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    measured, exp, glob, data, cat = [Path(contract[k]) for k in
        ('measured', 'expansion', 'global_population', 'data', 'catalog_reference')]
    old = pd.read_parquet(exp / 'tables/PAIR_RESULTS.parquet')
    new = pd.read_parquet(measured / 'tables/PAIR_RESULTS.parquet')
    inv = pd.read_csv(exp / 'tables/PLANNING_INVENTORY.csv')
    sealed = pd.read_parquet(glob / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet')
    references = {(x['deployment'], x['seed']): x for x in json.loads((exp / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    original = base.original_configs()
    configs, predictions, metrics, grids, bootstrap, bdraws, checks, eligibility_rows = {}, [], [], [], [], {}, [], []
    for dep_index, dep in enumerate(n.DEPS):
        full, _, structures = io.population(data, dep, old, inv)
        expected = sealed[sealed.deployment == dep].set_index('pair_id').loc[full.pair_id]
        for key in ('global_balanced_weight', 'global_HT_weight', 'full_sourcepair_multiplicity'):
            if not np.array_equal(expected[key].to_numpy(), full[key].to_numpy()):
                raise RuntimeError('Original source-population weights changed')
        f = full.merge(new[['pair_id', 'D_mass', 'D_full_common_denominator', 'best_mass_has_converged_run']],
                       on='pair_id', validate='one_to_one')
        if len(f) != len(full):
            raise RuntimeError('Measurement missing from population')
        ref = references[dep, n.SEEDS[0]]
        power = f.minimum_independent_power.to_numpy(float)
        active = (f.minimum_independent_power.between(*ref['minimum_power_support']) &
                  f.any_shared_converged & f.best_mass_has_converged_run).to_numpy(bool)
        y, w = f.kind.eq('true').to_numpy(float), f.global_balanced_weight.to_numpy(float)
        train, tune = f.fold.eq(0).to_numpy(), f.fold.eq(1).to_numpy()
        fit_mask = train & active
        if min(np.sum(fit_mask & (y == label)) for label in (0., 1.)) < 20:
            raise RuntimeError('Insufficient active fit support, no calibration')
        for fold in (0, 1):
            for kind in ('true', 'random_null', 'hard_null'):
                mask = f.fold.eq(fold).to_numpy() & f.kind.eq(kind).to_numpy()
                eligibility_rows.append({'deployment': dep, 'fold': fold, 'kind': kind, 'pairs': int(mask.sum()),
                    'active': int((mask & active).sum()), 'best_mass_not_converged': int((mask & ~f.best_mass_has_converged_run.to_numpy(bool)).sum())})
        bw = io.bootstrap_weights(f.loc[tune].reset_index(drop=True), *structures[1], np.random.default_rng(2026091073 + dep_index))
        for seed in n.SEEDS:
            recipe = n.recipes()[dep, seed]
            embedding = np.load(data / f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
            cosine = np.sum(embedding[i] * embedding[j], axis=1)
            bc, endpoint, masserror = base.raw_joint_bc(data / f'predictions/{dep}/{recipe["slot"]}_parent.npz', i, j)
            z, _, _ = base.nodup_terms(cosine, bc, endpoint, original[dep, seed], recipe)
            archive = pd.read_parquet(base.replay.archive_path(cat, 'NODUP-DIRECT-REPLAY', dep, seed, 'validation'))
            ii, jj = archive.idx_i.to_numpy(int), archive.idx_j.to_numpy(int)
            with np.load(base.replay.joint_path(dep, seed, 'validation')) as bank:
                ood = bank['outside']
            zz, _, _ = base.nodup_terms(archive.embedding_only.to_numpy(float), archive.joint_BC.to_numpy(float),
                (ood[ii] > .25) | (ood[jj] > .25), original[dep, seed], recipe)
            err = float(abs(zz - archive.waveform_score.to_numpy(float)).max())
            if err > 1e-10:
                raise RuntimeError('Original NODUP replay failed')
            checks.append({'deployment': dep, 'seed': seed, 'NODUP_replay_max_error': err, 'NN_joint_mass_error': masserror})
            reference_loss = float(np.dot(w[tune], io.binary_loss(y[tune], z[tune])))
            for arm in ARMS:
                statistic = STATISTICS[arm]
                deficit = f[statistic].to_numpy(float)
                spec0 = {'arm': base.ARMS[0], 'experiment_arm': arm, 'statistic': statistic,
                    'minimum_power_support': ref['minimum_power_support'],
                    'deficit_support': [0., float(deficit[fit_mask].max())], 'hinge_knots': [],
                    'nodup_support': [float(z[fit_mask].min()), float(z[fit_mask].max())],
                    'weights': original[dep, seed]['weights'], 'deployment': dep, 'seed': seed}
                identity = {**spec0, 'no_update': True, 'ridge': None}
                options = [(reference_loss, True, float('inf'), identity, z)]
                grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': None,
                    'no_update': True, 'risk_delta': 0., 'risk': reference_loss})
                x = base.feature_matrix(z, power, deficit, spec0)
                for ridge in base.RIDGES:
                    model = {**spec0, **base.train_model(x[fit_mask], y[fit_mask], w[fit_mask], ridge)}
                    value, outside, clip = apply(z, power, deficit, active, model)
                    loss = float(np.dot(w[tune], io.binary_loss(y[tune], value[tune])))
                    grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': ridge, 'no_update': False,
                        'risk_delta': loss - reference_loss, 'risk': loss, 'fallback_rate': float(outside[tune].mean()),
                        'clip_rate': float(clip[tune].mean())})
                    options.append((loss, False, ridge, model, value))
                best = min(r[0] for r in options)
                selected = min([r for r in options if r[0] <= best + 1e-12], key=lambda r: (not r[1], -r[2]))
                spec, value = selected[3:]
                configs[f'{dep}/{seed}/{arm}'] = spec
                if not np.array_equal(value[~active], z[~active]):
                    raise RuntimeError('Ineligible baseline changed')
                reverse = apply(z[::-1], power[::-1], deficit[::-1], active[::-1], spec)[0][::-1]
                if not np.array_equal(value, reverse):
                    raise RuntimeError('Order-dependent classifier')
                if not spec['no_update']:
                    grid = np.linspace(*spec['deficit_support'], 501)
                    curve = base.apply(np.full(501, np.mean(spec['nodup_support'])),
                        np.full(501, np.sqrt(np.prod(spec['minimum_power_support']))), grid, spec)[0]
                    if np.max(np.diff(curve)) > 1e-10:
                        raise RuntimeError('Deficit monotonicity failed')
                for fold in (0, 1):
                    use = f.fold.eq(fold).to_numpy()
                    current = float(np.dot(w[use], io.binary_loss(y[use], value[use])))
                    oldrisk = float(np.dot(w[use], io.binary_loss(y[use], z[use])))
                    metrics.append({'deployment': dep, 'seed': seed, 'arm': arm, 'fold': fold,
                        'global_risk_contribution': current, 'nodup_risk_contribution': oldrisk,
                        'global_risk_delta': current - oldrisk, 'no_update': spec['no_update']})
                delta = io.binary_loss(y[tune], value[tune]) - io.binary_loss(y[tune], z[tune])
                draws = bw @ delta
                bdraws[dep, seed, arm] = draws
                bootstrap.append({'deployment': dep, 'seed': str(seed), 'arm': arm,
                    'risk_delta': float(np.dot(w[tune], delta)), 'lower95': float(np.quantile(draws, .025)),
                    'upper95_two_sided': float(np.quantile(draws, .975)), 'upper95_one_sided': float(np.quantile(draws, .95))})
                out = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j', 'group_i', 'group_j',
                    'global_balanced_weight', 'global_HT_weight', 'minimum_independent_power',
                    'D_mass', 'D_full_common_denominator', 'best_mass_has_converged_run']].copy()
                out['seed'], out['arm'], out['waveform_score'], out['NODUP_waveform'] = seed, arm, value, z
                out['eligible'], out['joint_BC'], out['embedding_only'] = active, bc, cosine
                predictions.append(out)
                print('MASS_CONDITIONAL_FIT', dep, seed, arm, spec['ridge'], selected[0] - reference_loss, flush=True)
    metrics_df = pd.DataFrame(metrics)
    gates = []
    for arm in ARMS:
        for dep in n.DEPS:
            f = metrics_df[(metrics_df.arm == arm) & (metrics_df.deployment == dep) & (metrics_df.fold == 1)]
            draws = np.mean([bdraws[dep, seed, arm] for seed in n.SEEDS], axis=0)
            upper = float(np.quantile(draws, .95))
            good = bool((f.global_risk_delta <= 1e-12).all() and f.global_risk_delta.mean() < -1e-12 and upper <= 0.)
            gates.append({'deployment': dep, 'arm': arm, 'run_pass': good,
                'mean_delta': float(f.global_risk_delta.mean()),
                'relative_delta': float(f.global_risk_delta.mean() / f.nodup_risk_contribution.mean()),
                'bootstrap_upper95': upper})
            bootstrap.append({'deployment': dep, 'seed': 'MODEL_MEAN', 'arm': arm,
                'risk_delta': float(f.global_risk_delta.mean()), 'lower95': float(np.quantile(draws, .025)),
                'upper95_two_sided': float(np.quantile(draws, .975)), 'upper95_one_sided': upper})
    passing = [arm for arm in ARMS if all(r['run_pass'] for r in gates if r['arm'] == arm)]
    selected = min(passing, key=lambda arm: (max(r['relative_delta'] for r in gates if r['arm'] == arm),
        np.mean([r['relative_delta'] for r in gates if r['arm'] == arm]), ARMS.index(arm))) if passing else None
    for name, rows in [('CALIBRATION_GRID', grids), ('CALIBRATION_METRICS', metrics), ('SOURCE_BOOTSTRAP', bootstrap),
                       ('PILOT_GATES', gates), ('ELIGIBILITY_AUDIT', eligibility_rows), ('NODUP_REPLAY_AUDIT', checks)]:
        io.csv(root / f'tables/{name}.csv', rows)
    pd.concat(predictions).to_parquet(root / 'tables/PREDICTIONS.parquet', index=False)
    io.write(root / 'configs/ALL_CONDITIONAL_MODELS.json', configs)
    gate = {'UTC': io.utc(), 'gate': 'PASS' if selected else 'FAIL', 'status': n.STATUS, 'goal_achieved': False,
        'selected_common_arm': selected, 'passing_arms': passing, 'catalog_or_real_scored': False,
        'verified_inputs': check(root), 'initial_verified_inputs': count, 'seconds': time.monotonic() - start}
    io.write(root / 'contracts/PILOT_GATE.json', gate)
    report = '# R73 共享Mc与全共享数值对照的条件校准\n\n本轮仅模拟development/validation，不读取真实PE或候选结果。\n\n'
    report += '两个模型使用同一批对、同一独立功率参照和同一eligibility。单一分类器联合处理NODUP波形与投影特征，不恢复旧Mc/q头，也不混合新旧总分。\n\n'
    report += '```json\n' + json.dumps(gate, indent=2) + '\n```\n\n'
    report += pd.DataFrame(gates).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += pd.DataFrame(eligibility_rows).to_markdown(index=False) + '\n\n'
    report += '即使校准PASS也不是目标达成，仍须全注入面板、所有模型与共识的PE/官方nonloss及关键pair检查。Bootstrap条件于已反复使用的development，不是新的确认性区间。\n\n'
    report += 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n'
    (root / 'reports/R73_SHARED_MASS_CONDITIONAL_CN.md').write_text(report, encoding='utf-8')
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', required=True, choices=('freeze', 'fit'))
    for name in ('measured', 'expansion', 'global-root', 'catalog'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if args.stage == 'freeze':
        freeze(args.root, args.measured, args.expansion, args.global_root, args.catalog)
    else:
        fit(args.root)
