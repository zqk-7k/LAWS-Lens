#!/usr/bin/env python3
"""R63 simulation-only accounting of branch selection and false burden."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_reference_regularized_catalog_20260910 as app
n = app.n
METHODS = app.METHODS


def threshold(scores, y, fraction):
    order = np.argsort(-scores, kind='stable')
    count = np.cumsum(y[order])
    pos = min(int(np.searchsorted(count, fraction * count[-1])), len(order) - 1)
    boundary = int(order[pos])
    return float(scores[boundary]), boundary


def selected(scores, point):
    value, index = point
    return (scores > value) | ((scores == value) & (np.arange(len(scores)) <= index))


def decompose(a, b, y, active, fraction):
    ta, tb = threshold(a, y, fraction), threshold(b, y, fraction)
    aa, ab = selected(a, ta) & ~y, selected(a, tb) & ~y
    ba, bb = selected(b, ta) & ~y, selected(b, tb) & ~y
    score_effect = .5 * ((ba.sum() - aa.sum()) + (bb.sum() - ab.sum()))
    threshold_effect = .5 * ((ab.sum() - aa.sum()) + (bb.sum() - ba.sum()))
    entering, exiting = bb & ~aa, aa & ~bb
    inactive_score_effect = .5 * (((ba & ~active).sum() - (aa & ~active).sum()) +
                                 ((bb & ~active).sum() - (ab & ~active).sum()))
    result = {'false_before': int(aa.sum()), 'false_after': int(bb.sum()),
        'false_delta': int(bb.sum() - aa.sum()), 'score_effect': float(score_effect),
        'threshold_effect': float(threshold_effect),
        'inactive_false_score_effect': float(inactive_score_effect),
        'entered_active_false': int((entering & active).sum()),
        'entered_inactive_false': int((entering & ~active).sum()),
        'left_active_false': int((exiting & active).sum()),
        'left_inactive_false': int((exiting & ~active).sum()),
        'threshold_before': ta[0], 'threshold_after': tb[0],
        'threshold_true_pair_index_before': ta[1], 'threshold_true_pair_index_after': tb[1],
        'threshold_true_active_before': bool(active[ta[1]]),
        'threshold_true_active_after': bool(active[tb[1]]),
        'whole_tie_false_before': int(((a >= ta[0]) & ~y).sum()),
        'whole_tie_false_after': int(((b >= tb[0]) & ~y).sum())}
    if abs(score_effect + threshold_effect - result['false_delta']) > 1e-12:
        raise RuntimeError('False burden decomposition did not close')
    if int(entering.sum() - exiting.sum()) != result['false_delta']:
        raise RuntimeError('Entered/exited false count did not close')
    return result


def units():
    rng = np.random.default_rng(2026090963)
    for k in range(1000):
        y = np.r_[np.ones(10, bool), np.zeros(90, bool)]
        rng.shuffle(y)
        active = rng.random(len(y)) < .3
        a = rng.integers(-3, 4, len(y)).astype(float)
        b = np.where(active, rng.integers(-3, 4, len(y)), a)
        for fraction in (.5, .9):
            row = decompose(a, b, y, active, fraction)
            if row['inactive_false_score_effect'] != 0:
                raise RuntimeError('Inactive score effect must be zero')
            for scores, key in ((a, 'false_before'), (b, 'false_after')):
                order = np.argsort(-scores, kind='stable')
                loc = int(np.searchsorted(np.cumsum(y[order]), fraction * y.sum()))
                if int((~y[order[:loc + 1]]).sum()) != row[key]:
                    raise RuntimeError('Historical stable-tie cutoff mismatch')
    return {'randomized_cases': 1000, 'recall_cutoff_checks': 2000,
            'exact_additive_decomposition': True, 'unchanged_branch_score_effect_zero': True,
            'ties_tested': True}


def freeze(root, source):
    if root.exists():
        raise RuntimeError('Independent read-only audit directory required')
    for name in ('contracts', 'tables', 'audit', 'scripts', 'reports', 'manifest', 'logs'):
        (root / name).mkdir(parents=True)
    if not (source / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json').exists():
        raise RuntimeError('R62 must finish first')
    files = []
    for method in METHODS:
        paths = sorted((source / f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        if len(paths) != 30 or any(p.parent.name == 'real' for p in paths):
            raise RuntimeError('Expected injection-only inputs')
        files.extend(paths)
    files.extend([Path(__file__), source / 'configs/SELECTED_CONFIGURATIONS.json',
                  source / 'contracts/ANALYSIS_CONTRACT.json'])
    n.write_csv(root / 'manifest/INPUT_SHA256.csv', [{'path': str(p), 'sha256': n.sha(p)} for p in files])
    n.write_json(root / 'contracts/ANALYSIS_CONTRACT.json', {'UTC': n.utc(),
        'id': 'MCWF-REFERENCE-BRANCH-ACCOUNTING-63', 'status': n.STATUS, 'goal_achieved': False,
        'source': str(source), 'simulation_only': True, 'new_ranking_or_fit': False,
        'prespecified_scope': 'All30 archived injection panels, bothruns/all3models, waveform/fusion, F50/F90; no real panels or PE/official columns.',
        'comparison': 'R55 and R62 against NODUP, R62 against R55. Identical inactive waveform/time/sky, no counterfactual score publication.',
        'accounting': 'Use archived stable-index tie convention. Decompose F_new(t_new)-F_old(t_old) into symmetric score and threshold effects; split entered/exited nulls by frozen eligibility. This is descriptive algebra, not causal identification.',
        'selection_rates': 'Report P(E|L), P(E|N) and their log ratio only. They are empirical diagnostic quantities, not fitted offsets and never applied to ranks.',
        'interpretation': 'Class balancing after eligibility learns a conditional contrast. Branch-selection rates can expose distribution/scale questions, but these alone do not establish a correction for the existing NODUP score, which is not a calibrated complete log likelihood ratio.',
        'history': 'Catalogs reused in development. This does not create independent validation or authorize adjusting a method to failed test panels.',
        'frozen_everything': True, 'no_old_Mc_q_or_total_blend': True,
        'unit_tests': units()})
    shutil.copy2(__file__, root / 'scripts/reference_branch_audit.py')
    n.write_json(root / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'runtime_sha256': n.sha(Path(__file__)), 'manifest_sha256': n.sha(root / 'manifest/INPUT_SHA256.csv'),
        'contract_sha256': n.sha(root / 'contracts/ANALYSIS_CONTRACT.json')})
    print('BRANCH_AUDIT_FROZEN', len(files), flush=True)


def run(root):
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Read-only audit already complete')
    frozen = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)), ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv'),
                      ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json')]:
        if n.sha(file) != frozen[key]:
            raise RuntimeError('Audit contract changed')
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    source = Path(contract['source'])
    accounting, rates, ranges, replay = [], [], [], []
    for path in sorted((source / f'results/{app.NEW}').glob('gwtc*/seed_*/*/pairs.parquet')):
        rel = path.relative_to(source / f'results/{app.NEW}')
        dep, seedname, panel = rel.parts[:3]
        seed = int(seedname.split('_')[1])
        frames = {m: pd.read_parquet(source / 'results' / m / rel) for m in METHODS}
        base = frames[METHODS[0]]
        columns = ['idx_i', 'idx_j', 'is_true_pair', 'true_pair_family', 'time_score', 'sky_raw_log_bf']
        for f in frames.values():
            if not f[columns].equals(base[columns]):
                raise RuntimeError('Pair order, truth or frozen channel changed')
        y, active = base.is_true_pair.to_numpy(bool), frames[app.NEW].shared_profile_eligible.to_numpy(bool)
        common = {'deployment': dep, 'seed': seed, 'panel': panel}
        pl, pn = float(active[y].mean()), float(active[~y].mean())
        rates.append({**common, 'true_pairs': int(y.sum()), 'null_pairs': int((~y).sum()),
            'eligible_true': int((active & y).sum()), 'eligible_null': int((active & ~y).sum()),
            'P_eligible_given_true': pl, 'P_eligible_given_null': pn,
            'log_selection_rate_ratio': float(np.log(pl / pn)) if pl > 0 and pn > 0 else None,
            'no_offset_applied': True})
        metric_cache = {}
        for method, f in frames.items():
            for mode, col in [('waveform', 'waveform_score'), ('fusion', 'final_score')]:
                z = f[col].to_numpy(float)
                metric_cache[method, mode] = n.cf.fast_metrics(f, z)
                for label, truth in [('true', y), ('null', ~y)]:
                    for branch, elig in [('active', active), ('fallback', ~active)]:
                        vals = z[truth & elig]
                        ranges.append({**common, 'method': method, 'mode': mode, 'label': label,
                            'branch': branch, 'count': len(vals), 'mean': float(vals.mean()) if len(vals) else None,
                            **{f'q{int(q * 100):02d}': float(np.quantile(vals, q)) if len(vals) else None
                               for q in (0., .1, .5, .9, 1.)}})
        for ref, method in [(METHODS[0], METHODS[1]), (METHODS[0], app.NEW), (METHODS[1], app.NEW)]:
            for mode, col in [('waveform', 'waveform_score'), ('fusion', 'final_score')]:
                a, b = frames[ref][col].to_numpy(float), frames[method][col].to_numpy(float)
                if not np.array_equal(a[~active], b[~active]):
                    raise RuntimeError('Inactive scores changed')
                for fraction, label in [(.5, '0p5'), (.9, '0p9')]:
                    row = decompose(a, b, y, active, fraction)
                    if row['inactive_false_score_effect'] != 0:
                        raise RuntimeError('Unexpected inactive score effect')
                    for method_name, key in [(ref, 'false_before'), (method, 'false_after')]:
                        if row[key] != metric_cache[method_name, mode]['false_at_recall_' + label]:
                            raise RuntimeError('Archived false-burden replay failed')
                    accounting.append({**common, 'method': method, 'reference': ref, 'mode': mode,
                                       'recall': fraction, **row})
                replay.append({**common, 'method': method, 'reference': ref, 'mode': mode,
                               'F50_F90_replayed': True, 'inactive_scores_unchanged': True})
    checked = []
    for row in pd.read_csv(root / 'manifest/INPUT_SHA256.csv').itertuples():
        value = n.sha(Path(row.path))
        if value != row.sha256:
            raise RuntimeError('Read-only source changed')
        checked.append({'path': row.path, 'before': row.sha256, 'after': value})
    for name, rows in [('FALSE_BURDEN_DECOMPOSITION', accounting), ('ELIGIBILITY_SELECTION_RATES', rates),
                       ('SCORE_DISTRIBUTIONS_BY_BRANCH', ranges)]:
        n.write_csv(root / f'tables/{name}.csv', rows)
    n.write_csv(root / 'audit/REPLAY_UNITS.csv', replay)
    n.write_csv(root / 'audit/INPUT_HASHES_UNCHANGED.csv', checked)
    values = pd.DataFrame(accounting)
    failed = values[(values.deployment == 'gwtc4') & (values.seed == 202607242) &
        values.panel.isin(['sept8_reused_202609941', 'sept8_reused_202609943']) &
        (values.method == app.NEW) & (values.reference == METHODS[0]) & (values.recall == .5)]
    # This subset only explains already reported failures; it does not select a model.
    rates_df = pd.DataFrame(rates)
    aggregated = rates_df.groupby(['deployment', 'panel'])[['P_eligible_given_true', 'P_eligible_given_null',
                                                         'log_selection_rate_ratio']].mean().reset_index()
    text = '# R63 波形分支与固定召回率假对负担只读审计\n\n'
    text += f'状态：`{n.STATUS}`。未训练、未改变分数或排名，完整目标仍未达成。\n\n'
    text += '## 核心统计\n\nF50/F90不是一个固定阈值下的假对数。波形分数改变后，为达到同样的真对召回率，阈值也会移动。'
    text += '因此即使某个假对的分数没有改变，它也可能进入新的F50集合。以下采用原稳定索引tie规则，分解恒等式为对称的分数效应与阈值效应；它不是因果实验。\n\n'
    text += r'$$\Delta F=\tfrac12\{F_b(t_a)-F_a(t_a)+F_b(t_b)-F_a(t_b)\}+\tfrac12\{F_a(t_b)-F_a(t_a)+F_b(t_b)-F_b(t_a)\}.$$' + '\n\n'
    cols = ['panel', 'mode', 'false_before', 'false_after', 'false_delta', 'score_effect', 'threshold_effect',
            'entered_active_false', 'entered_inactive_false', 'left_active_false', 'left_inactive_false']
    text += failed[cols].to_markdown(index=False, floatfmt='.6f') + '\n\n'
    text += '完整所有run、模型、目录、50%/90%阈值及三组比较都保存在CSV；这里单列已报告O4a失败项，而不是只选择支持某个解释的数据。\n\n'
    text += '## 适用分支的选择分布\n\n下表为每panel的三个模型平均描述性比例。样本来自同一目录，不能把共享事件的pair当独立重复。\n\n'
    text += aggregated.to_markdown(index=False, floatfmt='.6f') + '\n\n'
    text += 'R61/R55在合格pair内做类别平衡分类，学习的是合格条件下的对照。在严格生成式LR中，选择事件E的完整证据还涉及 '
    text += r'$\log[P(E\mid L)/P(E\mid N)]$。' + '\n\n'
    text += '但当前NODUP回退分数本身不是完整校准生成式LR；因此不能把表中的经验log比例直接加到新分支，或据此断言缺了某个确定的Bayes factor。'
    text += '该审计仅暴露分支间可比较性和数据分布问题。下一版若研究共同校准，必须在模拟development/validation重新冻结完整方法，并仍检查所有外部门槛。\n\n'
    text += f'## 完整性\n\n{len(replay)}个面板/方法/模式的F50/F90精确复算，1000个含ties随机单元测试通过，{len(checked)}个输入hash未变。'
    text += '没有读取真实目录、公开PE或官方候选表，没有生成候选排序、拟合密度或应用任何选择率修正。\n'
    with (root / 'reports/R63_BRANCH_ACCOUNTING_CN.md').open('x', encoding='utf-8') as f:
        f.write(text)
    n.write_json(root / 'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'AUDIT_COMPLETE_NOT_METHOD_PASS',
        'goal_achieved': False, 'status': n.STATUS, 'panels': len(rates), 'comparisons': len(accounting),
        'new_score_or_rank': False, 'input_hashes_unchanged': len(checked), 'unit_tests': units()})
    print(failed[cols].to_string(index=False), flush=True)
    print('BRANCH_AUDIT_COMPLETE', len(accounting), len(checked), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--stage', choices=('freeze', 'run'), required=True)
    a = p.parse_args()
    if a.stage == 'freeze':
        freeze(a.root, a.source_root)
    else:
        run(a.root)
