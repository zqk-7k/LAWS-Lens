#!/usr/bin/env python3
"""R71: nested shared-Mc profile diagnostic with per-image q/spin nuisances."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as physical
import mcwf_nodup_conditional_deficit_20260910 as calibration

n, io = physical.n, calibration.population
BOUNDS3 = physical.BOUNDS
BOUNDS5 = (BOUNDS3[0], BOUNDS3[1], BOUNDS3[2], BOUNDS3[1], BOUNDS3[2])
BUDGETS = {'shared_mass_per_start': 900, 'independent_per_image': 450,
           'shared_full_reference': 900, 'shared_mass_final': 900}
PREFIX = 4


def optimize(fn, start, bounds, budget):
    start = np.asarray(start, float)
    simplex = np.tile(start, (len(start) + 1, 1))
    for k in range(len(start)):
        step = .004 if k == 0 else .035
        simplex[k + 1, k] += step if start[k] + step <= bounds[k][1] else -step
    fit = minimize(fn, start, method='Nelder-Mead', bounds=bounds,
        options={'maxfev': budget, 'xatol': 1e-4, 'fatol': 1e-4, 'initial_simplex': simplex})
    return {'x': fit.x.tolist(), 'value': float(-fit.fun), 'success': bool(fit.success),
            'nfev': int(fit.nfev), 'message': str(fit.message)}


class SharedMassObjective:
    def __init__(self, models):
        self.models = models
        self.single_cache = [{}, {}]
        self.mass_cache = {}

    def single(self, image, x):
        key = tuple(map(float, x))
        if key not in self.single_cache[image]:
            value = float(self.models[image].evaluate(np.asarray(key)))
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite projection')
            self.single_cache[image][key] = value
        return self.single_cache[image][key]

    def values(self, x):
        key = tuple(map(float, x))
        if key not in self.mass_cache:
            self.mass_cache[key] = (self.single(0, (key[0], key[1], key[2])),
                                    self.single(1, (key[0], key[3], key[4])))
        return self.mass_cache[key]

    def __call__(self, x):
        return -sum(self.values(x))

    def full(self, x):
        return self((x[0], x[1], x[2], x[1], x[2]))

    def best_mass(self):
        return min(self.mass_cache, key=lambda x: (-sum(self.mass_cache[x]), x))

    def best_full(self):
        keys = [x for x in self.mass_cache if x[1] == x[3] and x[2] == x[4]]
        return min(keys, key=lambda x: (-sum(self.mass_cache[x]), x))


def fit_pair(models, profiles, reference, signatures):
    # Canonicalize by waveform content, not event name, truth, PE or pair rank.
    order = sorted(range(2), key=lambda k: signatures[k])
    models = [models[k] for k in order]
    profiles = [profiles[k] for k in order]
    old_ind = np.asarray(reference['independent_power'], float)[order]
    refiner = [reference['independent_refinements'][k] for k in order]
    obj = SharedMassObjective(models)
    full = np.asarray(reference['shared_parameters'], float)
    replay = -obj.full(full)
    if abs(replay - reference['shared_power']) > 1e-6:
        raise RuntimeError('Frozen shared-power replay failed')
    separate = []
    for k, (profile, refined) in enumerate(zip(profiles, refiner)):
        options = [physical.point(profile), np.asarray(refined['x'], float)]
        for x in options:
            obj.single(k, x)
            obj.full(x)
        separate.append(max(options, key=lambda x: obj.single(k, x)))
    x0 = np.asarray(obj.best_full())[:3]
    full_run = optimize(obj.full, x0, BOUNDS3, BUDGETS['shared_full_reference'])
    separate = np.asarray(separate)
    candidates = {obj.best_full()}
    for mc in (separate[0, 0], separate[1, 0], separate[:, 0].mean(), full[0], full_run['x'][0]):
        candidates.add((float(mc), *separate[0, 1:], *separate[1, 1:]))
    for x in sorted(candidates):
        obj.values(x)
    starts = sorted(candidates, key=lambda x: (-sum(obj.values(x)), x))[:3]
    runs = [optimize(obj, x, BOUNDS5, BUDGETS['shared_mass_per_start']) for x in starts]
    initial = [max(cache, key=lambda x: (cache[x], tuple(-a for a in x))) for cache in obj.single_cache]
    ind = [optimize(lambda x, image=k: -obj.single(image, x), initial[k], BOUNDS3,
                    BUDGETS['independent_per_image']) for k in (0, 1)]
    separate = np.asarray([x['x'] for x in ind])
    for mc in (separate[0, 0], separate[1, 0], separate[:, 0].mean(), full[0]):
        obj.values((float(mc), *separate[0, 1:], *separate[1, 1:]))
    for x in separate:
        obj.full(x)
    runs.append(optimize(obj, obj.best_mass(), BOUNDS5, BUDGETS['shared_mass_final']))
    for fit in runs:
        x = fit['x']
        obj.full(x[:3])
        obj.full((x[0], x[3], x[4]))
    best = obj.best_mass()
    fullbest = obj.best_full()
    shared_mc = float(sum(obj.mass_cache[best]))
    shared_full = float(sum(obj.mass_cache[fullbest]))
    independent = np.maximum(old_ind, [max(cache.values()) for cache in obj.single_cache])
    dmc = float(independent.sum() - shared_mc)
    dfull = float(independent.sum() - shared_full)
    if dmc < -1e-8 or dfull < dmc - 1e-8 or shared_full < reference['shared_power'] - 1e-6:
        raise RuntimeError('Nested hypothesis inequality failed')
    converge = [r['value'] for r in runs if r['success']]
    x = list(best)
    if order != [0, 1]:
        x = [x[0], x[3], x[4], x[1], x[2]]
    return {'shared_mass_parameters': x, 'shared_mass_power': shared_mc, 'shared_full_control_power': shared_full,
        'shared_full_control_parameters': list(fullbest[:3]),
        'independent_power_common_denominator': independent[np.argsort(order)].tolist(),
        'D_mass': max(dmc, 0.), 'D_full_common_denominator': max(dfull, 0.),
        'D_full_historical': reference['deficit'], 'nuisance_release_gain': shared_mc - shared_full,
        'independent_search_gain': float(independent.sum() - old_ind.sum()),
        'full_search_gain': shared_full - reference['shared_power'],
        'shared_power_replay_error': replay - reference['shared_power'],
        'shared_mass_runs': runs, 'independent_refinements': ind, 'full_control_run': full_run,
        'any_mass_converged': any(r['success'] for r in runs),
        'best_mass_has_converged_run': bool(converge and shared_mc - max(converge) <= 1e-3),
        'mass_points': len(obj.mass_cache), 'single_evaluations': [len(x) for x in obj.single_cache],
        'nested_inequalities_pass': True, 'input_canonical_order': order}


def units():
    class Quadratic:
        def __init__(self, center):
            self.center = np.asarray(center)
        def evaluate(self, x):
            return 100. - np.dot([1000., 10., 5.], (np.asarray(x) - self.center)**2)
    output = []
    for gap in (0., .05):
        centers = [np.asarray([2.3, .4, -.3]), np.asarray([2.3 + gap, .8, .3])]
        models = [Quadratic(x) for x in centers]
        full = np.mean(centers, axis=0)
        power = sum(m.evaluate(full) for m in models)
        profiles = [{'logmc': x[0], 'q': x[1], 'chieff_equal': x[2]} for x in centers]
        ref = {'shared_parameters': full.tolist(), 'shared_power': power, 'independent_power': [100., 100.],
               'deficit': 200. - power, 'independent_refinements': [{'x': x.tolist()} for x in centers]}
        a = fit_pair(models, profiles, ref, ['a', 'b'])
        reverse = {**ref, 'independent_refinements': ref['independent_refinements'][::-1]}
        b = fit_pair(models[::-1], profiles[::-1], reverse, ['b', 'a'])
        expected = 500. * gap**2
        if abs(a['D_mass'] - expected) > 1e-5 or a['D_mass'] != b['D_mass']:
            raise RuntimeError('Analytic mass profile/swap unit failed')
        output.append({'logmc_gap': gap, 'analytic_D_mass': expected, 'computed_D_mass': a['D_mass'],
                       'swap_exact': True, 'nuisance_mismatch_not_mass_mismatch': gap != 0. or a['D_mass'] < 1e-5})
    return output


def freeze(root, expansion, forensic):
    if root.exists():
        raise RuntimeError('Independent R71 directory required')
    data = Path(json.loads((expansion / 'contracts/ANALYSIS_CONTRACT.json').read_text())['data'])
    for folder in ('contracts', 'pairs', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest', 'figures'):
        (root / folder).mkdir(parents=True)
    pairs = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    selected = set()
    for (_, _, _), g in pairs.groupby(['deployment', 'fold', 'kind']):
        selected.update(g.sort_values('selection_hash').head(PREFIX).pair_id)
    diag = pd.read_csv(forensic / 'tables/TRUE_SYSTEM_FORENSIC.csv')
    tails = set(diag[diag.eligible.astype(bool) & diag['tail'].astype(bool)].pair_id)
    keep = pairs[pairs.pair_id.isin(selected | tails)].copy()
    keep['hash_control'] = keep.pair_id.isin(selected)
    keep['enriched_tail_diagnostic'] = keep.pair_id.isin(tails)
    keep.to_parquet(root / 'contracts/PAIR_PLAN.parquet', index=False)
    contract = {'UTC': io.utc(), 'id': 'MCWF-SHARED-MASS-NUISANCE-PILOT-71', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'expansion': str(expansion), 'data': str(data),
        'forensic': str(forensic), 'same_method_both_runs': True,
        'scope': 'Waveform-only numerical/mechanistic pilot, no classifier fitting or real/catalog ranking.',
        'pair_plan': f'First{PREFIX} existing hash-ordered R56 pairs per run/fold/true-random-hard stratum, plus ALL R70 eligible high-D true systems as explicitly enriched diagnostics. Not a density-development population.',
        'hypotheses': 'Independent(logMc_i,q_i,chi_i;logMc_j,q_j,chi_j), shared-Mc(logMc,q_i,chi_i,q_j,chi_j), shared-full(logMc,q,chi). The intermediate hypothesis is a weaker necessary-compatibility test, NOT a complete lensed-source model.',
        'physics_reason': 'Inspect whether approximate q/spin nuisance fitting creates false Mc incompatibility. Real lensing does not change intrinsic q/spins; allowing independent nuisance fits here does NOT assert otherwise.',
        'statistic': 'D_mass=P_independent-P_sharedMc. Compare to D_full on the SAME union-search independent denominator. Also report the historical D_full and separate numerical-search gains.',
        'operator': 'Exactly frozen20-580Hz16sIMRPhenomD equal-aligned-spin projection; each image retains H1/L1 lag/quadrature amplitude nuisance maximization. No full PE or normalized likelihood.',
        'numerics': {'bounds3': BOUNDS3, 'bounds5': BOUNDS5, 'budgets': BUDGETS,
            'starts': 'Best3 of reference shared solution plus endpoint/midpoint Mc with separate q/spin from frozen waveform-only fits. One full3D control refinement, independent endpoint refinements and final5D refinement.',
            'ordering': 'Canonicalize by waveform cacheSHA only; names/truth/PE never determine search or score.', 'xatol': 1e-4, 'fatol': 1e-4},
        'gate': 'All planned measurements complete/finite, original shared power replay<=1e-6, analytic and event-swap tests pass, 0<=D_mass<=D_full_same_denominator and old full power retained. This is NUMERICAL gate only, not performance/adoption.',
        'next': 'Only after this pilot may a separately frozen full simulation calibration be considered. No direct use of enriched sample to train a density. No real/test scoring in R71.',
        'frozen': ['all encoders', 'learned distributions', 'time', 'sky', 'outerweights', 'scope/splits', 'all historical outputs',
                   'legacy Mc/q remains removed', 'old/new total blend remains removed'],
        'references': [{'url': 'https://arxiv.org/abs/gr-qc/9402014', 'use': 'Inspiral mass information and mass-spin correlations; not a proof that this surrogate is optimal.'},
            {'url': 'https://pycbc.org/pycbc/latest/html/inference/examples/hierarchical.html', 'use': 'Distinguish common and event-specific parameters; this optimization is not the documented marginalized Bayesian analysis.'},
            {'url': 'https://arxiv.org/abs/2104.09339', 'use': 'Complete lensing confirmation requires much more than one shared mass or a profile ratio.'}]}
    io.write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    paths = [Path(__file__), Path(physical.__file__), Path(physical.physical.__file__), Path(physical.physical.base.__file__),
             expansion / 'tables/PAIR_RESULTS.parquet', expansion / 'contracts/ANALYSIS_CONTRACT.json',
             forensic / 'tables/TRUE_SYSTEM_FORENSIC.csv', forensic / 'contracts/AUDIT_COMPLETE.json']
    for dep in n.DEPS:
        paths += [data / f'data/{dep}/event_metadata.parquet', data / f'data/{dep}/noise/noise_manifest.csv']
        f = keep[keep.deployment == dep]
        for idx in np.unique(np.r_[f.idx_i, f.idx_j]):
            paths += [data / f'profile_events/{dep}/{int(idx)}.json', expansion / f'cache/{dep}/{int(idx)}.npz']
    io.snapshot(root, paths)
    io.csv(root / 'audit/ANALYTIC_UNITS.csv', units())
    shutil.copy2(__file__, root / 'scripts/shared_mass_nuisance_pilot.py')
    io.write(root / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'runtime_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'plan_sha256': io.sha(root / 'contracts/PAIR_PLAN.parquet'),
        'manifest_sha256': io.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('SHARED_MASS_PILOT_FROZEN', len(keep), keep.groupby(['deployment', 'fold', 'kind']).size().to_dict(), flush=True)


def check(root):
    frozen = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, path in [('runtime_sha256', Path(__file__)), ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('plan_sha256', root / 'contracts/PAIR_PLAN.parquet'), ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if io.sha(path) != frozen[key]:
            raise RuntimeError('Frozen input changed ' + str(path))
    manifest = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in manifest.itertuples():
        if io.sha(r.path) != r.sha256:
            raise RuntimeError('Frozen source changed ' + r.path)
    return len(manifest)


def work(task):
    root, data, expansion, row = task
    tick = time.monotonic()
    models, profiles, signatures = [], [], []
    for idx in (row['idx_i'], row['idx_j']):
        path = Path(expansion) / f'cache/{row["deployment"]}/{int(idx)}.npz'
        with np.load(path) as bank:
            models.append(physical.physical.LowBand(bank['full20'], bank['frequency'], bank['psd'], 16))
        profiles.append(json.loads((Path(data) / f'profile_events/{row["deployment"]}/{int(idx)}.json').read_text()))
        signatures.append(io.sha(path))
    result = fit_pair(models, profiles, row, signatures)
    if row.get('physical_swap_unit', False):
        reverse = {**row, 'independent_power': row['independent_power'][::-1],
                   'independent_refinements': row['independent_refinements'][::-1]}
        swapped = fit_pair(models[::-1], profiles[::-1], reverse, signatures[::-1])
        if any(result[k] != swapped[k] for k in ('D_mass', 'D_full_common_denominator', 'shared_mass_power')):
            raise RuntimeError('Physical pair swap failed')
        result['physical_swap_exact'] = True
    result.update(pair_id=row['pair_id'], deployment=row['deployment'], fold=int(row['fold']), kind=row['kind'],
                  idx_i=int(row['idx_i']), idx_j=int(row['idx_j']), status='COMPLETE',
                  seconds=time.monotonic() - tick, peak_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return result


def measure(root, workers):
    check(root)
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data, expansion = Path(contract['data']), Path(contract['expansion'])
    plan = pd.read_parquet(root / 'contracts/PAIR_PLAN.parquet')
    if (root / 'contracts/MEASUREMENTS_COMPLETE.json').exists():
        raise RuntimeError('Completed pilot immutable')
    first = set(plan.sort_values('selection_hash').groupby('deployment').head(1).pair_id)
    tasks = []
    for row in plan.to_dict('records'):
        dest = root / f'pairs/{row["pair_id"]}.json'
        if dest.exists():
            if json.loads(dest.read_text())['status'] != 'COMPLETE':
                raise RuntimeError('Failed measurement needs explicit new recovery version')
            continue
        row['physical_swap_unit'] = row['pair_id'] in first
        tasks.append((str(root), str(data), str(expansion), row))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=physical.init_compute) as pool:
        pending = {pool.submit(work, task): task[-1] for task in tasks}
        for number, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                out = future.result()
            except Exception:
                out = {'pair_id': row['pair_id'], 'status': 'FAIL', 'traceback': traceback.format_exc()}
            io.write(root / f'pairs/{row["pair_id"]}.json', out)
            print('MASS_PROFILE_PAIR', number, len(tasks), row['pair_id'], out['status'],
                  out.get('seconds'), out.get('D_mass'), out.get('D_full_common_denominator'), flush=True)
    results = [json.loads(p.read_text()) for p in (root / 'pairs').glob('*.json')]
    failures = [r for r in results if r['status'] != 'COMPLETE']
    count = check(root)
    io.write(root / 'contracts/MEASUREMENTS_COMPLETE.json', {'UTC': io.utc(), 'pairs': len(results),
        'expected': len(plan), 'failures': len(failures), 'wall_seconds': time.monotonic() - started,
        'workers': workers, 'verified_inputs': count, 'goal_achieved': False, 'status': n.STATUS})
    if failures or len(results) != len(plan):
        raise RuntimeError('Pilot measurements failed; retained, no scoring allowed')
    summarize(root)


def summarize(root):
    check(root)
    if (root / 'contracts/PILOT_NUMERICAL_GATE.json').exists():
        raise RuntimeError('Completed summary immutable')
    status = json.loads((root / 'contracts/MEASUREMENTS_COMPLETE.json').read_text())
    if status['failures']:
        raise RuntimeError('Cannot summarize as passed with failed measurements')
    plan = pd.read_parquet(root / 'contracts/PAIR_PLAN.parquet')
    results = [json.loads(p.read_text()) for p in sorted((root / 'pairs').glob('*.json'))]
    scalar = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results])
    f = scalar.merge(plan[['pair_id', 'hash_control', 'enriched_tail_diagnostic', 'source_i', 'source_j']], on='pair_id', validate='one_to_one')
    f.to_parquet(root / 'tables/PAIR_RESULTS.parquet', index=False)
    io.csv(root / 'tables/PAIR_RESULTS.csv', f)
    summary = []
    for (dep, fold, kind), g in f.groupby(['deployment', 'fold', 'kind']):
        summary.append({'deployment': dep, 'fold': fold, 'kind': kind, 'pairs': len(g),
            'median_D_mass': g.D_mass.median(), 'median_D_full_control': g.D_full_common_denominator.median(),
            'median_D_full_old': g.D_full_historical.median(), 'median_nuisance_gain': g.nuisance_release_gain.median(),
            'median_independent_gain': g.independent_search_gain.median(), 'median_full_search_gain': g.full_search_gain.median(),
            'best_mass_converged': int(g.best_mass_has_converged_run.sum()), 'seconds_P50': g.seconds.median(),
            'seconds_P90': g.seconds.quantile(.9), 'seconds_max': g.seconds.max()})
    aucs = []
    for (dep, fold), g in f[f.hash_control].groupby(['deployment', 'fold']):
        for key in ('D_mass', 'D_full_common_denominator', 'D_full_historical'):
            aucs.append({'deployment': dep, 'fold': fold, 'statistic': key, 'pairs': len(g),
                         'descriptive_AUC': roc_auc_score(g.kind.eq('true'), -g[key]),
                         'small_selected_pilot_not_population_metric': True})
    io.csv(root / 'tables/STRATUM_SUMMARY.csv', summary)
    io.csv(root / 'tables/DESCRIPTIVE_HASH_CONTROL_AUC.csv', aucs)
    enriched = f[f.enriched_tail_diagnostic]
    gate = {'UTC': io.utc(), 'numerical_gate': 'PASS', 'goal_achieved': False, 'status': n.STATUS,
        'complete_pairs': len(f), 'physical_swap_checks': sum(r.get('physical_swap_exact', False) for r in results),
        'all_nested_inequalities_pass': bool(f.nested_inequalities_pass.all()),
        'all_best_mass_converged': bool(f.best_mass_has_converged_run.all()),
        'performance_gate': 'NOT_EVALUATED_NO_CALIBRATOR_NO_REAL_RANKING',
        'next_requires_separate_full_population_calibration': True}
    if gate['physical_swap_checks'] != 2 or not gate['all_nested_inequalities_pass']:
        gate['numerical_gate'] = 'FAIL'
    io.write(root / 'contracts/PILOT_NUMERICAL_GATE.json', gate)
    report = '# R71 共享Mc、释放q/自旋干扰参数的数值对照\n\n本轮为探索，未修改任何真实排名或训练校准器。\n\n'
    report += '只比较同一波形是否能共享Mc；q与等效自旋允许各次拟合独立，用于隔离近似模型误差。真实透镜仍共享内禀参数，这只是比完整共同源更弱的必要条件检查，不是透镜模型确认。\n\n'
    report += '三个嵌套模型：独立6参数、共享Mc的5参数、全部共享的3参数。使用同一个已探索独立功率分母，保证0<=D_Mc<=D_full；另外报告旧D和数值搜索增益，不能将更多搜索带来的变化归功于物理假设。\n\n'
    report += '## 数值门槛\n\n```json\n' + json.dumps(gate, indent=2) + '\n```\n\n'
    report += '## 所有富集高D真对\n\n' + enriched[['deployment', 'fold', 'pair_id', 'D_full_historical', 'D_full_common_denominator', 'D_mass', 'nuisance_release_gain', 'best_mass_has_converged_run']].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 分层与耗时\n\n' + pd.DataFrame(summary).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 小样本描述性AUC\n\n' + pd.DataFrame(aucs).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '这个样本包含专门富集的已知模拟尾部，不能直接拿它拟合目录密度或解释为总体性能。没有PE/官方特征，没有真实候选结果，没有新的recall。数值PASS不等于目标达成。\n\n'
    report += '## 依据\n\n'
    report += '- [Cutler与Flanagan](https://arxiv.org/abs/gr-qc/9402014)：质量信息与质量/自旋相关性提供动机，不证明当前近似最优。\n'
    report += '- [PyCBC共同参数模型](https://pycbc.org/pycbc/latest/html/inference/examples/hierarchical.html)：明确区分共同和事件特有参数；本轮不是该文档的完整贝叶斯边缘化。\n'
    report += '- [Lo与Magana Hernandez](https://arxiv.org/abs/2104.09339)：确认透镜需要完整模型、人口和选择效应，不能由Mc兼容直接确认。\n\n'
    report += '状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。\n'
    (root / 'reports/R71_SHARED_MASS_NUISANCE_CN.md').write_text(report, encoding='utf-8')
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--expansion', type=Path)
    parser.add_argument('--forensic', type=Path)
    parser.add_argument('--stage', required=True, choices=('freeze', 'measure', 'summarize'))
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.root, args.expansion, args.forensic)
    elif args.stage == 'measure':
        measure(args.root, args.workers)
    else:
        summarize(args.root)
