#!/usr/bin/env python3
"""R68: a fixed empirical compatibility ceiling on the R66 waveform score."""
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
import mcwf_nodup_conditional_deficit_20260910 as conditional

io, n = conditional.population, conditional.n
METHOD = 'NODUP-CONDITIONAL-TAIL95-CEILING'


def apply(z, power, deficit, spec):
    value, outside, clipped = conditional.apply(z, power, deficit, spec['conditional_model'])
    tail = (~outside) & (np.asarray(deficit) > spec['tail_calibration']['cutoff_D'])
    changed = tail & (value > 0.)
    value[tail] = np.minimum(value[tail], 0.)
    return value, outside, clipped | changed, tail, changed


def units(specs):
    rows = []
    for key, spec in specs.items():
        model = spec['conditional_model']
        cutoff = spec['tail_calibration']['cutoff_D']
        d = np.array([cutoff - 1e-8, cutoff, cutoff + 1e-8, model['deficit_support'][1] + 1.])
        z = np.full(4, .5 * sum(model['nodup_support']))
        power = np.full(4, np.sqrt(np.prod(model['minimum_power_support'])))
        before, outside, _ = conditional.apply(z, power, d, model)
        value, _, _, tail, changed = apply(z, power, d, spec)
        assert not tail[0] and not tail[1] and tail[2] and not tail[3]
        assert value[2] <= 0. and value[3] == z[3]
        assert np.all(value <= before) and np.array_equal(value[~tail], before[~tail])
        poison = {**spec, 'pe_mc': 0., 'official_fpp': 0., 'alpha': .875, 'legacy_mc_weight': 999.}
        assert np.array_equal(value, apply(z, power, d, poison)[0])
        rows.append({'model': key, 'cutoff_boundary_pass': True, 'OOD_exact_fallback': True,
                     'cannot_increase_waveform': True, 'forbidden_fields_unused': True})
    return rows


def freeze(root, parent, tail):
    if root.exists():
        raise RuntimeError('Independent R68 output required')
    conditional.check(parent)
    gate = json.loads((parent / 'contracts/PILOT_GATE.json').read_text())
    if gate['gate'] != 'PASS':
        raise RuntimeError('R66 pilot must pass')
    tgate = json.loads((tail / 'contracts/PILOT_GATE.json').read_text())
    if tgate['gate'] != 'PASS':
        raise RuntimeError('Frozen empirical-tail pilot must pass')
    for folder in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest'):
        (root / folder).mkdir(parents=True)
    choices = json.loads((parent / 'configs/ALL_CONDITIONAL_MODELS.json').read_text())
    thresholds = {}
    for c in json.loads((tail / 'configs/SELECTED_CONFIGURATIONS.json').read_text()):
        key = c['deployment'], c['seed']
        previous = thresholds.setdefault(key, c['tail_calibration'])
        if previous != c['tail_calibration']:
            raise RuntimeError('Historical tail rule differs across arms')
    specs = {}
    for dep in n.DEPS:
        for seed in n.SEEDS:
            model = choices[f'{dep}/{seed}/{gate["selected_common_arm"]}']
            specs[f'{dep}/{seed}'] = {'deployment': dep, 'seed': seed, 'method': METHOD,
                'conditional_model': model, 'weights': model['weights'],
                'tail_calibration': thresholds[dep, seed], 'ceiling': 0.}
    contract = {'UTC': io.utc(), 'id': 'MCWF-CONDITIONAL-TAIL95-CEILING-68',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'parent': str(parent), 'tail_reference': str(tail), 'method': METHOD,
        'rule': 'One R66 waveform score. Only inside its existing supported eligible branch, if shared-profile deficit exceeds the already frozen R65 run-specific fit-true95% order statistic, replace a positive waveform contribution with zero. Keep negative values. No separate added score or old/new total blend.',
        'threshold_source': 'Unchanged R65 fold0 source quantiles: ceil((n+1)*.95), shared across all3models within eachrun. No new threshold grid, fitting, or real-pair-specific exception.',
        'why': 'Test whether a learned positive waveform reward should be withheld when the approximate common-source fit fails a separate empirical compatibility screen. This is a conservative ranking constraint, not a Bayesian identity or calibrated posterior probability.',
        'fit_and_test_boundary': 'Only R66 saved source/noise-parent-disjoint fold0/fold1 simulation predictions are read. Original catalog test and real tables are not evaluated in this pilot.',
        'pilot_gate': 'Relative to NODUP, everymodel tune global-risk delta<=1e-12, strict negative runmean and source-bootstrap one-sided95% upper<=0 in BOTH runs. In addition supported-tail null positive-score mass must strictly decrease relative to R66 runmean. No requirement that all R66 calibration gains survive; report its exact loss delta and true-pair attenuation.',
        'coverage': 'Inherit R65 empirical-tail pilot, report fit/tune exceedance and cap rates. These are conditional reused-simulation diagnostics, NOT guaranteed95% coverage for real events or independent noise.',
        'bootstrap': {'draws': 2000, 'seed': 2026091068, 'unit': 'full source doublet with model-shared draws',
                      'limits': 'Conditional on fixed sampling weights, reused simulations and noise. No adaptive-selection or independent-noise variance.'},
        'frozen': ['all encoder and predictor checkpoints', 'NODUP legacy-head removal', 'R66 classifier coefficients',
                   'R65 true-tail threshold', 'physical measurements', 'time', 'sky', 'outer weights', 'scope/splits', 'all previous outputs'],
        'later_goal': 'Existing full per-model/panel injection guards and everymodel+consensus Top10/20 PE/official nonloss; critical pair outside all Top10. No tolerance relaxation or real-label tuning.',
        'references': [{'url': 'https://arxiv.org/abs/2107.07511',
            'use': 'Finite-sample empirical quantile convention; no exchangeability guarantee in this adaptive real-data development.'},
            {'url': 'https://arxiv.org/abs/2104.09339', 'use': 'Motivates shared-source compatibility; the projection deficit is not Hanabi evidence.'}]}
    io.write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    io.write(root / 'configs/SELECTED_CONFIGURATIONS.json', specs)
    inputs = [Path(__file__), Path(conditional.__file__), parent / 'contracts/PILOT_GATE.json',
              parent / 'tables/PREDICTIONS.parquet', parent / 'configs/ALL_CONDITIONAL_MODELS.json',
              parent / 'contracts/ANALYSIS_CONTRACT.json', tail / 'contracts/PILOT_GATE.json',
              tail / 'configs/SELECTED_CONFIGURATIONS.json']
    inputs += [Path(row.path) for row in pd.read_csv(parent / 'manifest/INPUT_SHA256.csv').itertuples()]
    io.snapshot(root, inputs)
    io.csv(root / 'audit/NUMERICAL_UNITS.csv', units(specs))
    shutil.copy2(__file__, root / 'scripts/conditional_tail_ceiling.py')
    io.write(root / 'contracts/START_FREEZE.json', {'UTC': io.utc(),
        'runtime_sha256': io.sha(Path(__file__)), 'contract_sha256': io.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'config_sha256': io.sha(root / 'configs/SELECTED_CONFIGURATIONS.json'),
        'manifest_sha256': io.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('CONDITIONAL_TAIL_CEILING_FROZEN', root, flush=True)


def check(root):
    freeze = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)), ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('config_sha256', root / 'configs/SELECTED_CONFIGURATIONS.json'), ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if io.sha(file) != freeze[key]:
            raise RuntimeError('Frozen file changed: ' + str(file))
    manifest = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for row in manifest.itertuples():
        if io.sha(row.path) != row.sha256:
            raise RuntimeError('Protected input changed: ' + row.path)
    return len(manifest)


def run(root):
    tick = time.monotonic()
    check(root)
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Completed pilot immutable')
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    parent = Path(contract['parent'])
    configs = json.loads((root / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    prior = pd.read_parquet(parent / 'tables/PREDICTIONS.parquet')
    pc = json.loads((parent / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data, expansion = Path(pc['data']), Path(pc['expansion'])
    pairs = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    inventory = pd.read_csv(expansion / 'tables/PLANNING_INVENTORY.csv')
    metrics, predictions, boot, gates = [], [], [], []
    for didx, dep in enumerate(n.DEPS):
        _, _, structures = io.population(data, dep, pairs, inventory)
        draws = []
        for seed in n.SEEDS:
            spec = configs[f'{dep}/{seed}']
            arm = spec['conditional_model']['arm']
            f = prior[(prior.deployment == dep) & (prior.seed == seed) & (prior.arm == arm)].copy().reset_index(drop=True)
            z = f.NODUP_waveform.to_numpy(float)
            value, outside, clipped, tail, changed = apply(z, f.minimum_independent_power.to_numpy(), f.deficit.to_numpy(), spec)
            before = conditional.apply(z, f.minimum_independent_power.to_numpy(), f.deficit.to_numpy(), spec['conditional_model'])[0]
            if not np.array_equal(before, f.waveform_score.to_numpy()):
                raise RuntimeError('R66 prediction replay changed')
            y, w = f.kind.eq('true').to_numpy(float), f.global_balanced_weight.to_numpy(float)
            loss_delta = io.binary_loss(y, value) - io.binary_loss(y, z)
            change_delta = io.binary_loss(y, value) - io.binary_loss(y, before)
            for fold in (0, 1):
                use = f.fold.eq(fold).to_numpy()
                true, null = use & (y == 1.), use & (y == 0.)
                null_d = float(np.dot(w[null], (value[null] > 0.).astype(float) - (before[null] > 0.).astype(float)))
                metrics.append({'deployment': dep, 'seed': seed, 'fold': fold,
                    'nodup_global_risk': float(np.dot(w[use], io.binary_loss(y[use], z[use]))),
                    'R66_global_risk': float(np.dot(w[use], io.binary_loss(y[use], before[use]))),
                    'new_global_risk': float(np.dot(w[use], io.binary_loss(y[use], value[use]))),
                    'delta_vs_NODUP': float(np.dot(w[use], loss_delta[use])),
                    'delta_vs_R66': float(np.dot(w[use], change_delta[use])),
                    'null_positive_mass_delta': null_d,
                    'true_sources': int(true.sum()), 'true_tail': int(tail[true].sum()),
                    'true_positive_capped': int(changed[true].sum()),
                    'null_pairs': int(null.sum()), 'null_tail': int(tail[null].sum()),
                    'null_positive_capped': int(changed[null].sum()),
                    'outside_count': int(outside[use].sum())})
            tune = f.fold.eq(1).to_numpy()
            weights = io.bootstrap_weights(f.loc[tune].reset_index(drop=True), *structures[1],
                                           np.random.default_rng(2026091068 + didx))
            draw = weights @ loss_delta[tune]
            draws.append(draw)
            boot.append({'deployment': dep, 'seed': str(seed), 'ci_lower': float(np.quantile(draw, .025)),
                         'ci_upper': float(np.quantile(draw, .975)), 'upper95': float(np.quantile(draw, .95))})
            f['R66_waveform_score'], f['waveform_score'] = before, value
            f['tail_screen'], f['positive_reward_withheld'], f['OOD'] = tail, changed, outside
            f['method'] = METHOD
            predictions.append(f)
            print('CEILING_PILOT', dep, seed, metrics[-1], flush=True)
        m = pd.DataFrame(metrics)
        m = m[(m.deployment == dep) & (m.fold == 1)]
        draw = np.mean(draws, axis=0)
        upper = float(np.quantile(draw, .95))
        passed = bool((m.delta_vs_NODUP <= 1e-12).all() and m.delta_vs_NODUP.mean() < 0.
                      and upper <= 0. and m.null_positive_mass_delta.mean() < 0.)
        gates.append({'deployment': dep, 'run_pass': passed, 'mean_delta_vs_NODUP': float(m.delta_vs_NODUP.mean()),
                      'mean_delta_vs_R66': float(m.delta_vs_R66.mean()), 'upper95_vs_NODUP': upper,
                      'null_positive_mass_delta': float(m.null_positive_mass_delta.mean())})
        boot.append({'deployment': dep, 'seed': 'MODEL_MEAN', 'ci_lower': float(np.quantile(draw, .025)),
                     'ci_upper': float(np.quantile(draw, .975)), 'upper95': upper})
    io.csv(root / 'tables/PILOT_METRICS.csv', metrics)
    io.csv(root / 'tables/SOURCE_BOOTSTRAP.csv', boot)
    io.csv(root / 'tables/PILOT_GATES.csv', gates)
    pd.concat(predictions, ignore_index=True).to_parquet(root / 'tables/PREDICTIONS.parquet', index=False)
    count = check(root)
    result = {'UTC': io.utc(), 'gate': 'PASS' if all(x['run_pass'] for x in gates) else 'FAIL',
              'status': n.STATUS, 'goal_achieved': False, 'catalog_or_real_scored': False,
              'verified_inputs': count, 'seconds': time.monotonic() - tick}
    io.write(root / 'contracts/PILOT_GATE.json', result)
    lines = ['# R68: Conditional Waveform Compatibility Ceiling', '',
             '本轮为自适应探索，不覆盖历史结果，也不构成新的独立确认。', '',
             '继承 R66 的单一波形分数，并使用此前已冻结的模拟真伴随对 95% 共享拟合差异阈值。',
             '仅在既有有效支持域内，超过该阈值时把正波形分数降为零，保留原负分；不额外相加另一份证据。',
             '时间、天空、外层权重、encoder、旧 Mc/q 删除与总分混合删除均保持不变。', '',
             '该限制是待检验的保守排序规则，不是 Bayesian identity、透镜概率或真实事件的 95% 保证。',
             '阈值来自历史模拟 fold0；本轮不重新拟合，不按关键真实 pair 调阈值。', '',
             '## Simulation Gate', '', '```json', json.dumps({'gate': result, 'runs': gates}, ensure_ascii=False, indent=2), '```', '',
             'PILOT_METRICS.csv 同时报告相对 NODUP/R66 的损失变化、真对误伤、非伴随正分减少和 OOD。',
             'bootstrap 以源系统为单位，三个模型使用同一抽样。它不包含独立噪声、反复试验或方法选择的不确定性。', '',
             '## Boundary', '', '本 pilot 不读取真实 PE、官方重合或候选排名，也不计算真实目录。',
             '只有两运行期都通过才可在新目录进行完整 catalog 验证。失败时保留原始结果，不调整本轮阈值。',
             '最终状态: HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。', '',
             'References: https://arxiv.org/abs/2107.07511 ; https://arxiv.org/abs/2104.09339', '']
    (root / 'reports/R68_CONDITIONAL_CEILING_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--tail', type=Path)
    parser.add_argument('--stage', choices=('freeze', 'pilot'), required=True)
    a = parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if a.stage == 'freeze':
        freeze(a.root, a.parent, a.tail)
    else:
        run(a.root)
