#!/usr/bin/env python3
"""R72: frozen shared-Mc projection on the complete R56 sampling design."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
import mcwf_shared_mass_nuisance_pilot_20260910 as pilot

io, n = pilot.io, pilot.n


def freeze(root, expansion, pilot_root):
    if root.exists():
        raise RuntimeError('Independent R72 directory required')
    prior = json.loads((pilot_root / 'contracts/PILOT_NUMERICAL_GATE.json').read_text())
    if prior['numerical_gate'] != 'PASS':
        raise RuntimeError('R71 numerical gate has not passed')
    pilot.check(pilot_root)
    data = Path(json.loads((expansion / 'contracts/ANALYSIS_CONTRACT.json').read_text())['data'])
    for folder in ('contracts', 'pairs', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest'):
        (root / folder).mkdir(parents=True)
    plan = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    if plan.pair_id.duplicated().any() or not plan.status.eq('COMPLETE').all():
        raise RuntimeError('Original measurement plan invalid')
    plan.to_parquet(root / 'contracts/PAIR_PLAN.parquet', index=False)
    contract = {'UTC': io.utc(), 'id': 'MCWF-SHARED-MASS-POPULATION-72', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'expansion': str(expansion),
        'data': str(data), 'pilot': str(pilot_root), 'pairs': len(plan),
        'scope': 'Complete R56 simulation sampling design, not the R71 enriched pilot. No calibrator fitting, real scoring, test access or automatic adoption.',
        'sampling': 'Exactly all measured R56 source doublets and hash-selected random/hard null pairs. Preserve inclusion probabilities, full source-pair multiplicity and source/noise-parent-disjoint folds. Do not recensor by new D_mass.',
        'operator': 'Exactly frozen R71 shared-Mc five-parameter and full-shared three-parameter optimizations on the same union-search independent denominator, unchanged budgets and starts.',
        'interpretation': 'Independent nuisance q/spin fits test necessary Mc compatibility under an approximate waveform, not the complete lensing hypothesis. Not normalized likelihood, PE or Bayes factor.',
        'reuse': 'Existing R71 measurements are reused only after identical plan identity, inputs and runtime hashes are checked. All other R56 pairs are measured regardless of outcome.',
        'eligibility': 'No new downstream eligibility rule chosen here. Retain original power and convergence fields alongside new numerical diagnostics. Nonconvergence must be visible and handled by a separately frozen calibration policy.',
        'measurement_gate': 'All planned pairs COMPLETE, old full-shared power replay<=1e-6, finite scores and exact nesting inequalities. Best-run convergence is reported separately; a numerical PASS is not a performance PASS.',
        'frozen': ['all encoder checkpoints and learned distributions', 'time', 'sky', 'outer weights',
                   'scope/splits', 'all old results', 'no legacy Mc/q or total blend restoration'],
        'next': 'Only a separately frozen simulation-only conditional calibration may follow. No actual real or test ranks are used to decide calibration settings.',
        'limitations': 'Repeated development data, per-image target-SNR scaling, reused lens environments and approximate equal-aligned-spin waveform remain. No independent confirmation claim.',
        'pilot_sha256': io.sha(pilot_root / 'contracts/ANALYSIS_CONTRACT.json'),
        'operator_sha256': io.sha(Path(pilot.__file__))}
    io.write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    files = [Path(__file__), Path(pilot.__file__), Path(pilot.physical.__file__),
        Path(pilot.physical.physical.__file__), Path(pilot.physical.physical.base.__file__),
        expansion / 'tables/PAIR_RESULTS.parquet', expansion / 'tables/PLANNING_INVENTORY.csv',
        expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json', expansion / 'contracts/ANALYSIS_CONTRACT.json',
        pilot_root / 'contracts/START_FREEZE.json', pilot_root / 'contracts/ANALYSIS_CONTRACT.json',
        pilot_root / 'contracts/PILOT_NUMERICAL_GATE.json']
    reused = []
    for row in plan.to_dict('records'):
        old = pilot_root / f'pairs/{row["pair_id"]}.json'
        if not old.exists():
            continue
        result = json.loads(old.read_text())
        if result['status'] != 'COMPLETE' or any(result[k] != row[k] for k in ('pair_id', 'deployment', 'fold', 'kind', 'idx_i', 'idx_j')):
            raise RuntimeError('R71 reuse mismatch')
        dest = root / 'pairs' / old.name
        shutil.copy2(old, dest)
        if io.sha(old) != io.sha(dest):
            raise RuntimeError('Reuse hash mismatch')
        reused.append({'source': str(old), 'target': str(dest), 'sha256': io.sha(old)})
        files.append(old)
    for dep in n.DEPS:
        files += [data / f'data/{dep}/event_metadata.parquet', data / f'data/{dep}/noise/noise_manifest.csv']
        f = plan[plan.deployment == dep]
        for idx in np.unique(np.r_[f.idx_i, f.idx_j]):
            files += [data / f'profile_events/{dep}/{int(idx)}.json', expansion / f'cache/{dep}/{int(idx)}.npz']
    io.csv(root / 'audit/EXACT_MEASUREMENT_REUSE.csv', reused)
    io.snapshot(root, files)
    shutil.copy2(__file__, root / 'scripts/shared_mass_population.py')
    shutil.copy2(pilot.__file__, root / 'scripts/shared_mass_nuisance_pilot.py')
    io.write(root / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'runtime_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'plan_sha256': io.sha(root / 'contracts/PAIR_PLAN.parquet'),
        'manifest_sha256': io.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('SHARED_MASS_POPULATION_FROZEN', len(plan), 'reused', len(reused), 'new', len(plan) - len(reused), flush=True)


def check(root):
    stamp = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, path in [('runtime_sha256', Path(__file__)), ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('plan_sha256', root / 'contracts/PAIR_PLAN.parquet'), ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if io.sha(path) != stamp[key]:
            raise RuntimeError('Frozen file changed ' + str(path))
    files = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in files.itertuples():
        if io.sha(r.path) != r.sha256:
            raise RuntimeError('Protected input changed ' + r.path)
    return len(files)


def measure(root, workers):
    check(root)
    if (root / 'contracts/MEASUREMENTS_COMPLETE.json').exists():
        raise RuntimeError('Completed population measurement immutable')
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    plan = pd.read_parquet(root / 'contracts/PAIR_PLAN.parquet')
    tasks = []
    for row in plan.to_dict('records'):
        dest = root / f'pairs/{row["pair_id"]}.json'
        if dest.exists():
            if json.loads(dest.read_text())['status'] != 'COMPLETE':
                raise RuntimeError('Failure retained: use a new explicit recovery version')
            continue
        tasks.append((str(root), contract['data'], contract['expansion'], row))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=pilot.physical.init_compute) as pool:
        pending = {pool.submit(pilot.work, task): task[-1] for task in tasks}
        for number, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                result = future.result()
            except Exception:
                result = {'pair_id': row['pair_id'], 'status': 'FAIL', 'traceback': traceback.format_exc()}
            io.write(root / f'pairs/{row["pair_id"]}.json', result)
            print('MASS_POPULATION_PAIR', number, len(tasks), row['pair_id'], result['status'],
                  result.get('seconds'), result.get('D_mass'), flush=True)
    results = [json.loads(p.read_text()) for p in (root / 'pairs').glob('*.json')]
    failures = sum(r['status'] != 'COMPLETE' for r in results)
    verified = check(root)
    io.write(root / 'contracts/MEASUREMENTS_COMPLETE.json', {'UTC': io.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'expected': len(plan), 'pairs': len(results), 'failures': failures,
        'new_pairs': len(tasks), 'workers': workers, 'wall_seconds': time.monotonic() - started,
        'verified_inputs': verified})
    if failures or len(results) != len(plan):
        raise RuntimeError('Population measurement incomplete; preserved, no calibration')
    summarize(root)


def summarize(root):
    check(root)
    if (root / 'contracts/MEASUREMENT_GATE.json').exists():
        raise RuntimeError('Completed summary immutable')
    status = json.loads((root / 'contracts/MEASUREMENTS_COMPLETE.json').read_text())
    if status['failures']:
        raise RuntimeError('Cannot summarize failed measurements as complete')
    plan = pd.read_parquet(root / 'contracts/PAIR_PLAN.parquet')
    results = [json.loads(p.read_text()) for p in sorted((root / 'pairs').glob('*.json'))]
    scalar = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results])
    keys = ['pair_id', 'deployment', 'fold', 'kind', 'idx_i', 'idx_j']
    provenance = plan[[c for c in plan.columns if c not in scalar.columns or c in keys]]
    f = scalar.merge(provenance, on=keys, validate='one_to_one')
    if len(f) != len(plan) or not np.isfinite(f[['D_mass', 'D_full_common_denominator', 'shared_mass_power']].to_numpy()).all():
        raise RuntimeError('Incomplete or nonfinite population')
    f.to_parquet(root / 'tables/PAIR_RESULTS.parquet', index=False)
    io.csv(root / 'tables/PAIR_RESULTS.csv', f)
    statistics = []
    for (dep, fold, kind), g in f.groupby(['deployment', 'fold', 'kind']):
        r = {'deployment': dep, 'fold': fold, 'kind': kind, 'pairs': len(g),
             'best_mass_converged': int(g.best_mass_has_converged_run.sum()),
             'any_mass_converged': int(g.any_mass_converged.sum())}
        for col in ('D_mass', 'D_full_historical', 'D_full_common_denominator', 'nuisance_release_gain',
                    'independent_search_gain', 'full_search_gain', 'seconds'):
            for label, quantile in (('P50', .5), ('P90', .9), ('P99', .99)):
                r[col + '_' + label] = float(g[col].quantile(quantile))
            r[col + '_max'] = float(g[col].max())
        statistics.append(r)
    io.csv(root / 'tables/POPULATION_STRATUM_SUMMARY.csv', statistics)
    gate = {'UTC': io.utc(), 'gate': 'PASS', 'status': n.STATUS, 'goal_achieved': False,
        'measured_pairs': len(f), 'true_pairs': int(f.kind.eq('true').sum()),
        'hash_selected_null_pairs': int(f.kind.ne('true').sum()),
        'nesting_pass': bool(f.nested_inequalities_pass.all()),
        'max_frozen_power_replay_error': float(f.shared_power_replay_error.abs().max()),
        'best_mass_not_converged': int((~f.best_mass_has_converged_run).sum()),
        'calibration_not_run': True, 'catalog_or_real_not_run': True,
        'interpretation': 'Only complete numerical measurements, not population performance or goal acceptance.'}
    if not gate['nesting_pass'] or gate['max_frozen_power_replay_error'] > 1e-6:
        gate['gate'] = 'FAIL'
    io.write(root / 'contracts/MEASUREMENT_GATE.json', gate)
    text = '# R72 完整模拟抽样的共享Mc数值复算\n\n'
    text += '本轮不训练网络、不拟合校准器、不重排真实目录。完整保留R56抽样与两折source/noise-parent隔离，未按新分数删选样本。\n\n'
    text += '独立模型、仅共享Mc模型、全共享模型使用同一数值独立功率分母。q/自旋释放仅是近似波形下的Mc兼容性诊断，不代表透镜可改变源内禀参数。\n\n'
    text += '```json\n' + json.dumps(gate, indent=2) + '\n```\n\n'
    text += pd.DataFrame(statistics).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    text += '校准必须另立合同，并使用原全source-pair采样权重。采样器未收敛项保留，不能静默删除。尚无新的recall、PE或官方重合结果。\n\n'
    text += 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n'
    (root / 'reports/R72_SHARED_MASS_POPULATION_CN.md').write_text(text, encoding='utf-8')
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--expansion', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--stage', required=True, choices=('freeze', 'measure', 'summarize'))
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.root, args.expansion, args.pilot)
    elif args.stage == 'measure':
        measure(args.root, args.workers)
    else:
        summarize(args.root)
