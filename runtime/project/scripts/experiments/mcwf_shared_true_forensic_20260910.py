#!/usr/bin/env python3
"""R70: simulation-only read-only shared-profile truth and solver audit."""
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
from scipy.stats import spearmanr

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_nodup_conditional_deficit_20260910 as prior
import mcwf_shared_profile_coherence_pilot_20260909 as physical

n, io = prior.n, prior.population


def freeze(root, expansion, tail):
    if root.exists():
        raise RuntimeError('Independent audit directory required')
    data = Path(json.loads((expansion / 'contracts/ANALYSIS_CONTRACT.json').read_text())['data'])
    for directory in ('contracts', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'logs'):
        (root / directory).mkdir(parents=True)
    contract = {'UTC': io.utc(), 'id': 'MCWF-SHARED-TRUE-FORENSIC-70', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'expansion': str(expansion),
        'data': str(data), 'tail_reference': str(tail),
        'question': 'Why can genuinely shared simulated sources have a large common-versus-independent projection deficit?',
        'scope': 'All R56 measured true companions, both runs and both source/noise-disjoint folds; no sample selected by real PE or candidate identity.',
        'diagnostics': ['true parameters versus frozen fit bounds', 'best shared optimizer convergence rather than merely any start',
                        'shared optimum power versus converged starts', 'per-image independent refinement convergence',
                        'chirp-mass error', 'unequal aligned spins and transverse-spin magnitude',
                        'source inclination, target SNR and image SNR asymmetry', 'shared/independent gain and empirical tail'],
        'truth': 'Simulation truth for diagnosis ONLY; never passed to a ranking/calibration function. No real/test catalogs read.',
        'threshold': 'Inherit R65 empirical fit-true95 quantile unchanged, used only to label diagnostic strata.',
        'statistics': 'All true systems once, descriptive per-run/fold and tail/core summaries. Correlations are descriptive; no pair-independent significance claim.',
        'no_scoring_changes': True, 'no_new_optimization': True,
        'frozen': ['all encoders and predictive checkpoints', 'time', 'sky', 'outer weights', 'scope/splits',
                   'all archived scores and physical fits', 'no legacy Mc/q restoration', 'no total blend'],
        'possible_next_step': 'A separately frozen numerical or waveform-model pilot only after this audit. Do not adjust real-candidate thresholds.',
        'limitations': 'Approximate projection is not full PE, normalized likelihood, chi-square test or Bayes factor. Simulations and lens environments have been reused.'}
    io.write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    files = [Path(__file__), Path(physical.__file__), Path(physical.physical.__file__),
             Path(physical.physical.base.__file__), expansion / 'contracts/ANALYSIS_CONTRACT.json',
             expansion / 'tables/PAIR_RESULTS.parquet', expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json',
             tail / 'configs/SELECTED_CONFIGURATIONS.json']
    pairs = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    for dep in n.DEPS:
        files += [data / f'data/{dep}/event_metadata.parquet', data / f'data/{dep}/noise/noise_manifest.csv']
        f = pairs[(pairs.deployment == dep) & (pairs.kind == 'true')]
        for i in np.unique(np.r_[f.idx_i, f.idx_j]):
            files.append(data / f'profile_events/{dep}/{int(i)}.json')
    io.snapshot(root, files)
    shutil.copy2(__file__, root / 'scripts/shared_true_forensic.py')
    io.write(root / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'script_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'manifest_sha256': io.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('TRUE_PROFILE_FORENSIC_FROZEN', root, flush=True)


def check(root):
    f = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, p in [('script_sha256', Path(__file__)), ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                   ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if io.sha(p) != f[key]:
            raise RuntimeError('Frozen file changed ' + str(p))
    rows = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in rows.itertuples():
        if io.sha(r.path) != r.sha256:
            raise RuntimeError('Protected source changed ' + r.path)
    return len(rows)


def run(root):
    check(root)
    if (root / 'contracts/AUDIT_COMPLETE.json').exists():
        raise RuntimeError('Audit immutable')
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data, exp, tail = [Path(contract[k]) for k in ('data', 'expansion', 'tail_reference')]
    frame = pd.read_parquet(exp / 'tables/PAIR_RESULTS.parquet')
    refs = {(x['deployment'], x['seed']): x for x in json.loads((exp / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    thresholds = {x['deployment']: x['tail_calibration']['cutoff_D'] for x in json.loads((tail / 'configs/SELECTED_CONFIGURATIONS.json').read_text())}
    rows, isolated = [], []
    for dep in n.DEPS:
        meta = pd.read_parquet(data / f'data/{dep}/event_metadata.parquet').set_index('row_index')
        noise = pd.read_csv(data / f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        block = meta.noise_bank_index.map(noise.parent_file_gps)
        sources = len(set(meta.source_uid[meta.fold == 0]) & set(meta.source_uid[meta.fold == 1]))
        noises = len(set(block[meta.fold == 0]) & set(block[meta.fold == 1]))
        if sources or noises:
            raise RuntimeError('Source or parent-noise overlap')
        isolated.append({'deployment': dep, 'source_overlap': sources, 'noise_parent_overlap': noises})
        for r in frame[(frame.deployment == dep) & (frame.kind == 'true')].to_dict('records'):
            a, b = meta.loc[r['idx_i']], meta.loc[r['idx_j']]
            shared = ['source_uid', 'fold', 'm1_det', 'm2_det', 'mc_det', 'a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi12', 'phijl']
            if any(a[k] != b[k] for k in shared):
                raise RuntimeError('True pair source parameters not actually shared')
            runs, indep = list(r['shared_optimizer_runs']), list(r['independent_refinements'])
            best = max(runs, key=lambda x: x['value'])
            converged = [x['value'] for x in runs if x['success']]
            par = np.asarray(r['shared_parameters'])
            trueq = a.m2_det / a.m1_det
            s1, s2 = a.a1 * np.cos(a.tilt1), a.a2 * np.cos(a.tilt2)
            eff = (s1 + trueq * s2) / (1. + trueq)
            perp = max(a.a1 * np.sin(a.tilt1), a.a2 * np.sin(a.tilt2))
            ref = refs[dep, n.SEEDS[0]]
            eligible = ref['minimum_power_support'][0] <= r['minimum_independent_power'] <= ref['minimum_power_support'][1] and bool(r['any_shared_converged'])
            originals = [json.loads((data / f'profile_events/{dep}/{int(i)}.json').read_text()) for i in (r['idx_i'], r['idx_j'])]
            rows.append({'deployment': dep, 'fold': int(r['fold']), 'pair_id': r['pair_id'], 'source_uid': a.source_uid,
                'idx_i': int(r['idx_i']), 'idx_j': int(r['idx_j']), 'eligible': eligible,
                'tail': bool(r['deficit'] > thresholds[dep]), 'threshold_D': thresholds[dep],
                'deficit': r['deficit'], 'relative_deficit': r['relative_deficit'],
                'minimum_power': r['minimum_independent_power'], 'power_total': sum(r['independent_power']),
                'true_Mc': a.mc_det, 'true_q': trueq, 'true_chieff': eff,
                'unequal_aligned_spin': abs(s1 - s2), 'max_transverse_spin_NOT_chip': perp,
                'inclination_sine': float(np.sin(a.theta_jn)), 'SNR_min': min(a.target_network_snr, b.target_network_snr),
                'SNR_max': max(a.target_network_snr, b.target_network_snr),
                'SNR_ratio': max(a.target_network_snr, b.target_network_snr) / min(a.target_network_snr, b.target_network_snr),
                'truth_outside_q_bound': not .25 <= trueq <= 1.,
                'truth_outside_effspin_bound': not -.8 <= eff <= .8,
                'truth_outside_Mc_bound': not 5. <= a.mc_det <= 200.,
                'shared_q_boundary': min(abs(par[1] - .25), abs(par[1] - 1.)) < 1e-3,
                'shared_spin_boundary': min(abs(par[2] - -.8), abs(par[2] - .8)) < 1e-3,
                'shared_logMc_abs_error': abs(par[0] - np.log(a.mc_det)),
                'best_shared_start_converged': bool(best['success']),
                'best_shared_power_has_converged_solution': bool(converged and r['shared_power'] - max(converged) <= 1e-3),
                'power_beyond_best_shared_start': r['shared_power'] - best['value'],
                'any_shared_converged': bool(r['any_shared_converged']),
                'shared_starts_at_budget': sum(x['nfev'] >= 450 for x in runs),
                'independent_refinements_converged': sum(bool(x['success']) for x in indep),
                'independent_refinements_at_budget': sum(x['nfev'] >= 300 for x in indep),
                'original_profile_logMc_gap': abs(originals[0]['logmc'] - originals[1]['logmc']),
                'optimized_independent_logMc_gap': abs(indep[0]['x'][0] - indep[1]['x'][0]),
                'seconds': r['seconds']})
    f = pd.DataFrame(rows)
    if f.duplicated(['deployment', 'source_uid']).any():
        raise RuntimeError('A source appears more than once')
    io.csv(root / 'tables/TRUE_SYSTEM_FORENSIC.csv', f)
    io.csv(root / 'audit/SOURCE_NOISE_ISOLATION.csv', isolated)
    flags = ['truth_outside_q_bound', 'truth_outside_effspin_bound', 'truth_outside_Mc_bound', 'shared_q_boundary',
             'shared_spin_boundary', 'best_shared_start_converged', 'best_shared_power_has_converged_solution']
    quantities = ['deficit', 'relative_deficit', 'minimum_power', 'true_Mc', 'true_q', 'unequal_aligned_spin',
                  'max_transverse_spin_NOT_chip', 'inclination_sine', 'SNR_min', 'SNR_ratio', 'shared_logMc_abs_error']
    summary, correlation = [], []
    for (dep, fold, eligible, tail), g in f.groupby(['deployment', 'fold', 'eligible', 'tail']):
        s = {'deployment': dep, 'fold': fold, 'eligible': bool(eligible), 'tail': bool(tail), 'sources': len(g)}
        s.update({key + '_count': int(g[key].sum()) for key in flags})
        s.update({key + '_median': float(g[key].median()) for key in quantities})
        summary.append(s)
    for (dep, fold), g in f[f.eligible].groupby(['deployment', 'fold']):
        for key in quantities[2:]:
            correlation.append({'deployment': dep, 'fold': fold, 'sources': len(g), 'feature': key,
                                'Spearman_D': float(spearmanr(g.deficit, g[key]).statistic)})
    io.csv(root / 'tables/FORENSIC_STRATUM_SUMMARY.csv', summary)
    io.csv(root / 'tables/DESCRIPTIVE_CORRELATIONS.csv', correlation)
    ef = f[f.eligible]
    report = '# R70 模拟真伴随共享拟合审计\n\n状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。只读审计，不改变评分。\n\n'
    report += f'全部测量真系统{len(f)}个，原有效分支{len(ef)}个。没有读取真实候选PE或重新采样波形。\n\n'
    cols = ['deployment', 'fold', 'pair_id', 'deficit', 'true_Mc', 'true_q', 'true_chieff', 'max_transverse_spin_NOT_chip',
            'best_shared_power_has_converged_solution', 'shared_starts_at_budget', 'independent_refinements_at_budget', 'shared_logMc_abs_error']
    report += '## 原有效分支的所有高D真对\n\n' + ef[ef['tail']][cols].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 分层概览\n\n' + pd.DataFrame(summary).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 解释限制\n\n等自旋、非进动、主模的IMRPhenomD不是一般进动注入的完整模型。边界、有限多起点优化、噪声波动及模型误差都可能抬高D。相关性本身不能区分因果。\n\n'
    report += 'any_shared_converged只说明至少一个起点停止，并不自动证明达到全局最大值。另列最佳起点/最佳共享功率是否有收敛解，以及预算触顶；不能把这些诊断静默变成排名规则。\n\n'
    report += '真实参数用于本次模拟诊断，不是部署特征。横向自旋量仅为max(a1 sin tilt1,a2 sin tilt2)，不是chi_p。之后任何新优化器或波形模型都须另立合同、先模拟验证。\n'
    (root / 'reports/R70_SHARED_TRUE_FORENSIC_CN.md').write_text(report, encoding='utf-8')
    verified = check(root)
    result = {'UTC': io.utc(), 'status': n.STATUS, 'goal_achieved': False, 'true_sources': len(f),
              'eligible_sources': len(ef), 'eligible_tail_sources': int(ef['tail'].sum()),
              'tail_best_shared_not_converged': int((~ef.loc[ef['tail'], 'best_shared_power_has_converged_solution']).sum()),
              'verified_inputs': verified, 'new_optimization': False, 'real_or_test_read': False, 'scores_changed': False}
    io.write(root / 'contracts/AUDIT_COMPLETE.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print(ef[ef['tail']][cols].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--expansion', type=Path)
    parser.add_argument('--tail', type=Path)
    parser.add_argument('--stage', choices=('freeze', 'audit'), required=True)
    a = parser.parse_args()
    if a.stage == 'freeze':
        freeze(a.root, a.expansion, a.tail)
    else:
        run(a.root)
