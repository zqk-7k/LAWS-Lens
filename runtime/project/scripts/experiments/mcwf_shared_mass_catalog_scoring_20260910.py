#!/usr/bin/env python3
"""R75: apply frozen R73 models using original power and new R74 deficits."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_mass_conditional_calibration_20260910 as cal
import mcwf_shared_profile_catalog_20260909 as parent
import mcwf_shared_profile_catalog_audit_20260909 as audit
import mcwf_r74_catalog_remeasure_20260910 as measurement

n, io = cal.n, cal.io
ROOT = None
PILOT = P / 'results/mcwf_shared_mass_conditional_73_20260910T110300Z'
PHYSICAL = P / 'results/mcwf_r74_catalog_remeasure_20260910T111000Z'
OLD = measurement.R51
METHODS = ('NODUP-DIRECT-REPLAY', 'R73-FROZEN-CONDITIONAL-WF')
NEW = METHODS[-1]
BASE_EXPORT, BASE_CSV = n.public_frame, n.write_csv
BASE_RANK = n.dev.BASE.rank_real


def result_path(method, dep, seed, split, catalog=None):
    panel = split if catalog is None else f'{split}_{catalog}'
    filename = 'fusion_all_pairs.parquet' if split == 'real' else 'pairs.parquet'
    return ROOT / f'results/{method}/{dep}/seed_{seed}/{panel}/{filename}'


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent R75 directory required')
    cal.check(PILOT)
    measurement.check(PHYSICAL)
    gate = json.loads((PILOT / 'contracts/PILOT_GATE.json').read_text())
    if gate['gate'] != 'PASS':
        raise RuntimeError('Both-run R73 calibration gate required')
    for folder in ('contracts', 'configs', 'results', 'pairs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'logs', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    chosen = gate['selected_common_arm']
    options = json.loads((PILOT / 'configs/ALL_CONDITIONAL_MODELS.json').read_text())
    original = cal.base.original_configs()
    configs = []
    for (dep, seed), old in original.items():
        model = options[f'{dep}/{seed}/{chosen}']
        if model['weights'] != old['weights']:
            raise RuntimeError('Outer weights changed')
        configs += [{**old, 'method': method, 'conditional_model': model} for method in METHODS]
    contract = {'UTC': io.utc(), 'id': 'MCWF-SHARED-MASS-CATALOG-SCORING-75', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'pilot': str(PILOT),
        'physical': str(PHYSICAL), 'original_physical': str(OLD), 'selected_common_arm': chosen,
        'methods': METHODS, 'primary': NEW,
        'transport': 'R73 original minimum_independent_power and original any_shared_converged from R51; new R74 D_mass/D_full and best_mass_has_converged_run only. Never consume the R74 compatibility aliases as calibration covariates.',
        'eligibility': 'Both original quality-valid endpoints AND original shared convergence AND original power support AND new best-mass convergence. Same rule as R73, independent of labels/ranks/PE.',
        'score': 'Exactly frozen R73 selected model. Outside support or ineligible exact original NODUP. No tail ceiling, old Mc/q head, additive physical LR, or total-score mixture.',
        'injection': 'All30 original validation/test and three reused catalog panels across six run/model combinations; waveform and fusion. Existing reused panels are adaptive development, not a new locked test.',
        'guards': {'R10_drop_max': .02, 'AUPRC_drop_max': .005, 'F50_F90_ratio_max': 1.1,
            'require_every_panel_and_mode': True, 'reference': METHODS[0]},
        'real': 'Only after full injection calculation AND every injection guard PASS: remeasure all90 original quality-valid real pairs with the same R71/R74 operator, then rank full strict scopes. No real PE/official selection.',
        'external_goal': 'Key outside O3 consensus and all3models Top10; both runs consensus AND all models Top10/20 PE/official nonloss versus NODUP. Official overlap is not lensing truth.',
        'diagnostic_fix': 'Explicit waveform-only sorting uses weights[1,0,0]; frozen historical total/consensus fusion score definition is unchanged.',
        'frozen': ['all encoder checkpoints', 'learned intrinsic distributions', 'time', 'sky', 'outer weights', 'scopes/splits', 'all historical results'],
        'no_automatic_adoption': True}
    io.write(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    io.write(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    io.write(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': io.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': io.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json')})
    fullplan = pd.read_parquet(OLD / 'contracts/PAIR_PLAN.parquet')
    fullplan.to_parquet(ROOT / 'contracts/FULL_PHYSICAL_PLAN.parquet', index=False)
    files = [Path(__file__), PILOT / 'contracts/PILOT_GATE.json', PILOT / 'configs/ALL_CONDITIONAL_MODELS.json',
        PILOT / 'tables/PREDICTIONS.parquet', PHYSICAL / 'contracts/ANALYSIS_CONTRACT.json',
        PHYSICAL / 'contracts/START_FREEZE.json', PHYSICAL / 'contracts/PAIR_PLAN.parquet',
        OLD / 'contracts/PAIR_PLAN.parquet', parent.BASE / 'configs/SELECTED_CONFIGURATIONS.json']
    files += [Path(r.path) for r in pd.read_csv(PHYSICAL / 'manifest/INPUT_SHA256.csv').itertuples()]
    # Reference optimizer solutions are immutable inputs, not new measurements.
    files += [OLD / f'pairs/{r.pair_id}.json' for r in fullplan.itertuples()]
    for dep in n.DEPS:
        files.append(n.t.EXTERNAL / f'{dep}_external_reference.parquet')
        for seed in n.SEEDS:
            for split, catalog in [('validation', None), ('test', None), ('real', None)] + [('sept8_reused', c) for c in n.CATALOGS]:
                files += [parent.archive_file(method, dep, seed, split, catalog) for method in ('NODUP-DIRECT', n.BASELINE)]
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and str(P / 'scripts/experiments') in file and Path(file).suffix == '.py':
            files.append(Path(file))
    io.snapshot(ROOT, files)
    shutil.copy2(__file__, ROOT / 'scripts/shared_mass_catalog_scoring.py')
    io.write(ROOT / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'runtime_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
        'config_sha256': io.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'manifest_sha256': io.sha(ROOT / 'manifest/INPUT_SHA256.csv')})
    print('R75_FROZEN', chosen, flush=True)


def check():
    stamp = json.loads((ROOT / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)), ('contract_sha256', ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('config_sha256', ROOT / 'configs/SELECTED_CONFIGURATIONS.json'), ('manifest_sha256', ROOT / 'manifest/INPUT_SHA256.csv')]:
        if io.sha(file) != stamp[key]:
            raise RuntimeError('Frozen R75 changed: ' + str(file))
    rows = pd.read_csv(ROOT / 'manifest/INPUT_SHA256.csv')
    for r in rows.itertuples():
        if io.sha(r.path) != r.sha256:
            raise RuntimeError('Protected input changed: ' + r.path)
    return len(rows)


@lru_cache(maxsize=4)
def measured(dep, tag):
    folder = ROOT if tag == 'real' else PHYSICAL
    rows = [json.loads(p.read_text()) for p in (folder / f'pairs/{dep}/{tag}').glob('*.json')]
    if any(r['status'] != 'COMPLETE' for r in rows):
        raise RuntimeError('Failed physical measurement')
    return {(int(r['idx_i']), int(r['idx_j'])): r for r in rows}


def load_panel(dep, seed, split, catalog=None):
    parent.ROOT = OLD
    f = parent.load_panel(dep, seed, split, catalog)
    for key in ('r75_D_mass', 'r75_D_full_common_denominator', 'r75_new_minimum_power', 'r75_power_replay_error'):
        f[key] = np.nan
    f['r75_best_mass_converged'] = False
    f['r75_original_power'] = f.shared_profile_minimum_power.to_numpy(copy=True)
    f['r75_original_shared_converged'] = f.shared_profile_any_optimizer_converged.to_numpy(copy=True)
    rows = measured(dep, parent.panel_tag(seed, split, catalog))
    for pos in np.flatnonzero(f.shared_profile_available.to_numpy(bool)):
        a, b = int(f.idx_i.iloc[pos]), int(f.idx_j.iloc[pos])
        row = rows.get(tuple(sorted((a, b))))
        if row is None:
            raise RuntimeError('Missing planned measurement; no silent fallback')
        ref = OLD / f'pairs/{row["pair_id"]}.json'
        if io.sha(ref) != row['reference_sha256']:
            raise RuntimeError('Optimizer reference changed')
        for col, value in [('r75_D_mass', row['D_mass']), ('r75_D_full_common_denominator', row['D_full_common_denominator']),
            ('r75_new_minimum_power', min(row['independent_power_common_denominator'])),
            ('r75_best_mass_converged', row['best_mass_has_converged_run']), ('r75_power_replay_error', row['shared_power_replay_error'])]:
            f.loc[f.index[pos], col] = value
    return f


def infer(frame, config):
    z, ood, clipped = [frame['NODUP_frozen_' + key].to_numpy(copy=True) for key in ('waveform', 'ood', 'clip')]
    c = config['conditional_model']
    power = frame.r75_original_power.to_numpy(float)
    active = (frame.shared_profile_available.to_numpy(bool) & frame.r75_original_shared_converged.to_numpy(bool) &
        frame.r75_best_mass_converged.to_numpy(bool) & (power >= c['minimum_power_support'][0]) & (power <= c['minimum_power_support'][1]))
    used = np.zeros(len(frame), bool)
    outside = np.zeros(len(frame), bool)
    if config['method'] == NEW:
        deficit = frame['r75_' + c['statistic']].to_numpy(float)
        value, oo, cc = cal.base.apply(z[active], power[active], deficit[active], c)
        z[active], outside[active] = value, oo
        if not c['no_update']:
            ood[active], clipped[active], used[active] = oo, cc, ~oo
    if not np.isfinite(z).all() or not np.array_equal(z[~active], frame.NODUP_frozen_waveform.to_numpy()[~active]):
        raise RuntimeError('Nonfinite score or changed inactive fallback')
    frame['r75_eligible'], frame['r75_used'], frame['r75_support_ood'] = active, used, outside
    frame['r75_NODUP_waveform'] = frame.NODUP_frozen_waveform.to_numpy(copy=True)
    return z, ood, clipped


def export(frame, z, weights, method):
    out = BASE_EXPORT(frame, z, weights, method)
    for col in frame:
        if col.startswith('r75_') or col in ('sky_j50', 'sky_j90'):
            out[col] = frame[col].to_numpy(copy=True)
    return out


def save_csv(file, rows):
    if Path(file).name == 'PER_SEED_PE_OFFICIAL_BUDGETS.csv':
        rows = []
        for dep in n.DEPS:
            for seed in n.SEEDS:
                for method in (n.BASELINE, *METHODS):
                    f = pd.read_parquet(result_path(method, dep, seed, 'real'))
                    rows += [{**n.dev.budget_row(f, method, dep, 'fusion', b), 'seed': seed} for b in (10, 20, 50, 100)]
    BASE_CSV(file, rows)


def rank_real(frame, weights, mode, seed):
    if mode == 'waveform':
        weights = {'waveform': 1., 'time': 0., 'sky': 0.}
    return BASE_RANK(frame, weights, mode, seed)


def units():
    check()
    prediction = pd.read_parquet(PILOT / 'tables/PREDICTIONS.parquet')
    configs = n.selections(ROOT)
    rows = []
    for config in configs:
        dep, seed = config['deployment'], config['seed']
        c = config['conditional_model']
        g = prediction[(prediction.deployment == dep) & (prediction.seed == seed) & (prediction.arm == c['experiment_arm'])].copy()
        f = pd.DataFrame({'NODUP_frozen_waveform': g.NODUP_waveform.to_numpy(), 'NODUP_frozen_ood': False,
            'NODUP_frozen_clip': False, 'shared_profile_available': g.eligible.to_numpy(bool),
            'r75_original_shared_converged': g.eligible.to_numpy(bool), 'r75_best_mass_converged': g.best_mass_has_converged_run.to_numpy(bool),
            'r75_original_power': g.minimum_independent_power.to_numpy(),
            'r75_D_mass': g.D_mass.to_numpy(), 'r75_D_full_common_denominator': g.D_full_common_denominator.to_numpy()})
        value = infer(f.copy(), config)[0]
        target = g.NODUP_waveform.to_numpy() if config['method'] == METHODS[0] else g.waveform_score.to_numpy()
        if not np.array_equal(value, target):
            raise RuntimeError('R73 serialization replay failed')
        poison = f.copy()
        for col in ('minimum_independent_power', 'any_shared_converged', 'r75_new_minimum_power', 'time_score', 'sky_raw_log_bf',
            'pe_mc_bhattacharyya_coefficient', 'official_frontend', 'old_Mc_score', 'old_q_score', 'final_score'):
            poison[col] = 99999.
        if not np.array_equal(value, infer(poison, {**config, 'alpha': .875})[0]):
            raise RuntimeError('Forbidden or compatibility alias input consumed')
        if not np.array_equal(value, infer(f.iloc[::-1].copy(), config)[0][::-1]):
            raise RuntimeError('Row order changes score')
        rows.append({'deployment': dep, 'seed': seed, 'method': config['method'], 'rows': len(f),
            'pilot_serialization_max_error': 0., 'poison_max_error': 0., 'reverse_max_error': 0.})
    io.csv(ROOT / 'audit/PIPELINE_UNITS.csv', rows)
    io.write(ROOT / 'contracts/PIPELINE_UNIT_PASS.json', {'UTC': io.utc(), 'tests': len(rows),
        'original_power_transport': True, 'real_or_catalog_scores_not_read': True})


def injection_guard():
    a = pd.read_csv(ROOT / 'tables/RETRIEVAL_PER_SEED.csv')
    rows = []
    keys = ['deployment', 'seed', 'split', 'catalog', 'mode']
    old = a[a.method == METHODS[0]].set_index(keys)
    for row in a[a.method == NEW].to_dict('records'):
        ref = old.loc[tuple(row[k] for k in keys)]
        tests = {'R10': row['macro_r_at_10'] >= ref.macro_r_at_10 - .02 - 1e-12,
            'AUPRC': row['average_precision'] >= ref.average_precision - .005 - 1e-12,
            'F50': row['false_at_recall_0p5'] <= ref.false_at_recall_0p5 * 1.1 + 1e-12,
            'F90': row['false_at_recall_0p9'] <= ref.false_at_recall_0p9 * 1.1 + 1e-12}
        rows.append({**{k: row[k] for k in keys}, 'pass': all(tests.values()), **{k + '_pass': v for k, v in tests.items()},
            **{k + '_delta': float(row[k] - ref[k]) for k in ('macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9')}})
    if len(rows) != 60:
        raise RuntimeError('Expected all60 panel/mode guards')
    io.csv(ROOT / 'tables/NODUP_INJECTION_GUARDS.csv', rows)
    gate = {'UTC': io.utc(), 'gate': 'PASS' if all(r['pass'] for r in rows) else 'FAIL', 'status': n.STATUS,
        'goal_achieved': False, 'rows': len(rows), 'failures': sum(not r['pass'] for r in rows),
        'real_scoring_authorized_by_gate': all(r['pass'] for r in rows)}
    io.write(ROOT / 'contracts/INJECTION_GUARD.json', gate)
    print('R75_INJECTION_GUARD', json.dumps(gate), flush=True)


def evaluate():
    check()
    if not (ROOT / 'contracts/PIPELINE_UNIT_PASS.json').exists():
        raise RuntimeError('Units first')
    gate = json.loads((PHYSICAL / 'contracts/MEASUREMENT_GATE.json').read_text())
    if gate['gate'] != 'PASS' or gate['complete_pairs'] != 5241:
        raise RuntimeError('Complete R74 injection measurement gate required')
    n.METHODS = METHODS
    n.load_panel, n.infer, n.public_frame, n.write_csv = load_panel, infer, export, save_csv
    n.run(ROOT, 'evaluate')
    injection_guard()


def measure_real(workers):
    check()
    if json.loads((ROOT / 'contracts/INJECTION_GUARD.json').read_text())['gate'] != 'PASS':
        raise RuntimeError('Real blocked by injection guard')
    if (ROOT / 'contracts/REAL_MEASUREMENT_COMPLETE.json').exists():
        raise RuntimeError('Completed real measurement immutable')
    plan = pd.read_parquet(ROOT / 'contracts/FULL_PHYSICAL_PLAN.parquet')
    plan = plan[plan.scope == 'real']
    jobs = []
    for row in plan.to_dict('records'):
        dest = measurement.pair_path(ROOT, row)
        if dest.exists():
            if json.loads(dest.read_text())['status'] != 'COMPLETE':
                raise RuntimeError('Retained failure requires explicit recovery')
        else:
            jobs.append((str(ROOT), row))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=measurement.r71.physical.init_compute) as pool:
        futures = {pool.submit(measurement.work, job): job[1] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception:
                result = {**row, 'status': 'FAIL', 'traceback': traceback.format_exc()}
            destination = measurement.pair_path(ROOT, row)
            destination.parent.mkdir(parents=True, exist_ok=True)
            io.write(destination, result)
            print('R75_REAL_MEASUREMENT', number, len(jobs), result['status'], flush=True)
    results = [json.loads(measurement.pair_path(ROOT, row).read_text()) for row in plan.to_dict('records')]
    if len(results) != 90 or any(r['status'] != 'COMPLETE' or not r['nested_inequalities_pass'] or abs(r['shared_power_replay_error']) > 1e-6 for r in results):
        raise RuntimeError('Real measurement numerical gate failed')
    io.write(ROOT / 'contracts/REAL_MEASUREMENT_COMPLETE.json', {'UTC': io.utc(), 'pairs': 90, 'failed': 0})


def real():
    check()
    if json.loads((ROOT / 'contracts/INJECTION_GUARD.json').read_text())['gate'] != 'PASS':
        raise RuntimeError('Injection guard blocks real ranking')
    if not (ROOT / 'contracts/REAL_MEASUREMENT_COMPLETE.json').exists():
        raise RuntimeError('Real numerical measurements required')
    n.METHODS = METHODS
    n.load_panel, n.infer, n.public_frame, n.write_csv = load_panel, infer, export, save_csv
    n.dev.BASE.rank_real = rank_real
    n.run(ROOT, 'real')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('freeze', 'units', 'evaluate', 'guard', 'measure-real', 'real'))
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    ROOT = args.root
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if args.stage == 'measure-real':
        measure_real(args.workers)
    elif args.stage == 'guard':
        injection_guard()
    else:
        globals()[args.stage]()
