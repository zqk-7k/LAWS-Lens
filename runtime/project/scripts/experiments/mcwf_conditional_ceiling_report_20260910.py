#!/usr/bin/env python3
"""R68/R69 reports, paired uncertainty and immutable delivery metadata."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_ceiling_catalog_20260910 as app
n, io = app.n, app.io


def text(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(value)


def paired_reference(root):
    source = root / 'uncertainty_complete'
    table = pd.read_csv(source / 'tables/CONDITIONAL_SYSTEM_CI_PER_PANEL.csv')
    selected = table[table.method.eq(app.NEW)]
    rows = []
    for dep in n.DEPS:
        batches = [(str(seed), split, selected[selected.deployment.eq(dep) &
            selected.seed.eq(f'seed_{seed}') & (selected.panel.eq('test') if split == 'test' else
            selected.panel.str.startswith('sept8_reused_'))]) for seed in n.SEEDS for split in ('test', 'sept8_reused')]
        batches += [('three_fixed_models', 'sept8_reused', selected[selected.deployment.eq(dep) &
                    selected.panel.str.startswith('sept8_reused_')])]
        for seed, split, batch in batches:
            for (mode, quantity), f in batch.groupby(['mode', 'quantity']):
                differences, nominal = [], []
                name = f'{mode}_query_draws.parquet' if quantity.startswith('R') else f'{mode}_pair_draws.parquet'
                for r in f.itertuples():
                    folder = source / f'draws/{dep}/{r.seed}/{r.panel}'
                    a = pd.read_parquet(folder / app.NEW / name)[quantity].to_numpy()
                    b = pd.read_parquet(folder / app.METHODS[1] / name)[quantity].to_numpy()
                    reference = table[(table.deployment == dep) & (table.seed == r.seed) & (table.panel == r.panel) &
                        (table.method == app.METHODS[1]) & (table['mode'] == mode) & (table.quantity == quantity)]
                    if len(reference) != 1 or len(a) != len(b):
                        raise RuntimeError('Paired interval identity failed')
                    differences.append(a - b)
                    nominal.append(r.nominal - reference.iloc[0].nominal)
                delta = np.mean(differences, axis=0)
                rows.append({'deployment': dep, 'seed': seed, 'split': split, 'mode': mode,
                    'quantity': quantity, 'method': app.NEW, 'baseline': app.METHODS[1],
                    'nominal_delta': float(np.mean(nominal)), 'delta_q025': float(np.quantile(delta, .025)),
                    'delta_q975': float(np.quantile(delta, .975)), 'repetitions': len(delta)})
    io.csv(root / 'tables/R67_PAIRED_DELTA_CI.csv', rows)
    return pd.DataFrame(rows)


def figures(root, per, budgets):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    labels = ['NODUP', 'R67', 'R69']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, key, label in zip(axes.flat, ['macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'],
                             ['Companion R@10', 'Pair AUPRC', 'False pairs at 50% recall', 'False pairs at 90% recall']):
        for dep, offset, color, run in [('gwtc3', -.13, '#007c91', 'O3'), ('gwtc4', .13, '#bd4551', 'O4a')]:
            for j, method in enumerate(app.METHODS):
                a = per[(per.deployment == dep) & (per.method == method) & (per['mode'] == 'fusion')][key].to_numpy()
                if len(a) != 3:
                    raise RuntimeError('Missing three-model figure points')
                x = j + offset
                ax.scatter(x + np.linspace(-.035, .035, 3), a, facecolors='none', edgecolors=color,
                           s=25, label=run if j == 0 else None)
                ax.plot([x - .065, x + .065], [a.mean()] * 2, lw=2, color=color)
        ax.set_xticks(range(3), labels)
        ax.set_title(label, loc='left', fontsize=11)
        ax.grid(axis='y', alpha=.15)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0., 0., 1., .94))
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'figures/R69_INJECTION_COMPARISON.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    keys = ['BC_mc_ge_0p5', 'Dmax_le_3', 'official_frontend', 'official_hanabi']
    ticks = ['Mc BC >= 0.5', 'Dmax <= 3', 'Official 1%', 'Hanabi table']
    for row, dep in enumerate(n.DEPS):
        for column, budget in enumerate((10, 20)):
            ax = axes[row, column]
            for k, (method, color) in enumerate(zip(app.METHODS, ['#626262', '#007c91', '#b8464b'])):
                f = budgets[(budgets.config == method) & (budgets.deployment == dep) & (budgets.budget == budget)]
                if len(f) != 1:
                    raise RuntimeError('Missing consensus figure budget')
                ax.bar(np.arange(4) + (k - 1) * .23, f.iloc[0][keys].to_numpy(float),
                       width=.21, color=color, label=labels[k])
            ax.set_xticks(range(4), ticks, rotation=12)
            ax.set_ylim(0, budget + 1)
            ax.set_title(('O3' if dep == 'gwtc3' else 'O4a') + f' consensus Top-{budget}', loc='left')
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0., 0., 1., .94))
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'figures/R69_PE_OFFICIAL_COMPARISON.{ext}', dpi=180)
    plt.close(fig)


def snapshot(root, pilot):
    modules = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(P / 'scripts'):
            modules.setdefault(Path(path).resolve(), []).append(name)
    for name in ('mcwf_conditional_ceiling_audit_20260910.py', 'mcwf_conditional_ceiling_threshold_audit_20260910.py',
                 'mcwf_shared_profile_system_bootstrap_20260909.py', 'mcwf_shared_profile_bootstrap_summary_20260909.py',
                 'mcwf_shared_profile_package_20260909.py'):
        modules.setdefault(P / 'scripts/experiments' / name, []).append('execution_entry')
    for destination_root in (root, pilot):
        rows = []
        for path, names in sorted(modules.items()):
            if re.search(rb'-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}', path.read_bytes()):
                raise RuntimeError('Credential in runtime code')
            dest = destination_root / 'scripts/runtime_dependencies' / path.relative_to(P / 'scripts')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise RuntimeError('Existing runtime snapshot')
            shutil.copy2(path, dest)
            rows.append({'source': str(path), 'destination': str(dest.relative_to(destination_root)),
                         'sha256': io.sha(path), 'aliases': '|'.join(names)})
        io.csv(destination_root / 'manifest/RUNTIME_DEPENDENCIES.csv', rows)



def main(root, pilot):
    if not (root / 'uncertainty_complete/contracts/COMPLETE.json').exists():
        raise RuntimeError('Complete system bootstrap required')
    result = json.loads((root / 'audit/FINAL_GOAL_READOUT.json').read_text())
    metric = pd.read_csv(root / 'tables/RECOMPUTED_METRICS_ALL_PANELS.csv')
    keys = ['macro_r_at_1', 'macro_r_at_5', 'macro_r_at_10', 'average_precision',
            'false_at_recall_0p5', 'false_at_recall_0p9']
    reused = metric[metric.panel.str.startswith('sept8_reused_')]
    per = reused.groupby(['deployment', 'method', 'mode', 'seed'])[keys].mean().reset_index()
    aggregate = per.groupby(['deployment', 'method', 'mode'])[keys].agg(['mean', 'std'])
    aggregate.columns = ['_'.join(x) for x in aggregate.columns]
    aggregate = aggregate.reset_index()
    io.csv(root / 'tables/R69_REUSED_PER_MODEL.csv', per)
    io.csv(root / 'tables/R69_REUSED_MEAN_SD.csv', aggregate)
    critical = pd.read_csv(root / 'tables/CRITICAL_PAIR_ALL_RANKS.csv')
    critical['reported_rank'] = critical['rank'].where(critical.unit.eq('model'), critical.consensus_rank)
    ccols = [c for c in ('method', 'seed', 'reported_rank', 'waveform_score', 'time_score', 'sky_raw_log_bf',
                        'pe_mc_bhattacharyya_coefficient', 'pe_mc_standardized_median_distance',
                        'official_po_fpp', 'official_ml_fpp') if c in critical]
    concise = critical[ccols]
    io.csv(root / 'tables/CRITICAL_PAIR_CONCISE.csv', concise)
    budgets = pd.read_csv(root / 'tables/PE_OFFICIAL_BUDGETS.csv')
    figures(root, per, budgets)
    paired_reference(root)
    top, movements = [], []
    for dep in n.DEPS:
        for unit, folder in [('consensus', 'consensus')] + [('model', f'seed_{s}/real') for s in n.SEEDS]:
            rank = 'consensus_rank' if unit == 'consensus' else 'rank'
            nd = pd.read_parquet(root / f'results/{app.METHODS[0]}/{dep}/{folder}/fusion_all_pairs.parquet')
            for method in app.METHODS:
                f = pd.read_parquet(root / f'results/{method}/{dep}/{folder}/fusion_all_pairs.parquet').sort_values(rank)
                if unit == 'consensus':
                    top += [{**r, 'method': method, 'deployment': dep} for r in f.head(50).to_dict('records')]
                if method != app.NEW:
                    continue
                for b in (10, 20):
                    before, after = set(nd.nsmallest(b, rank).pair_key), set(f.head(b).pair_key)
                    for label, names, table in [('entered', after - before, f), ('left', before - after, nd)]:
                        movements += [{**r, 'deployment': dep, 'unit': unit, 'model_folder': folder,
                                       'budget': b, 'movement': label} for r in table[table.pair_key.isin(names)].to_dict('records')]
    io.csv(root / 'tables/REAL_CONSENSUS_TOP50_ALL_METHODS.csv', top)
    io.csv(root / 'tables/REAL_TOP10_20_ENTERED_LEFT.csv', movements)
    specs = json.loads((pilot / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    configrows = []
    for c in specs.values():
        model = c['conditional_model']
        slopes = model['raw_slopes']
        configrows.append({'deployment': c['deployment'], 'seed': c['seed'], 'threshold_D': c['tail_calibration']['cutoff_D'],
                           'fit_true_sources': c['tail_calibration']['fit_sources'], 'ceiling': c['ceiling'],
                           'intercept': model['raw_intercept'], 'ND_slope': slopes[0], 'log_power_slope': slopes[1],
                           'negative_log_deficit_slope': slopes[2], 'wf_weight': c['weights'][0],
                           'time_weight': c['weights'][1], 'sky_weight': c['weights'][2]})
    io.csv(root / 'tables/FROZEN_RULES_AND_WEIGHTS.csv', configrows)
    attenuation = []
    for file in sorted((root / f'results/{app.NEW}').glob('gwtc*/seed_*/*/pairs.parquet')):
        f = pd.read_parquet(file)
        dep, seed, panel = file.relative_to(root / f'results/{app.NEW}').parts[:3]
        t, used = f.is_true_pair.to_numpy(bool), f.ceiling_positive_withheld.to_numpy(bool)
        attenuation.append({'deployment': dep, 'seed': seed, 'panel': panel, 'true_pairs': int(t.sum()),
                            'null_pairs': int((~t).sum()), 'true_withheld': int((t & used).sum()),
                            'null_withheld': int((~t & used).sum()), 'tail_pairs': int(f.ceiling_tail_screen.sum())})
    io.csv(root / 'tables/REWARD_WITHHOLDING_AUDIT.csv', attenuation)
    allci = pd.read_csv(root / 'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv')
    ci = allci[(allci.method == app.NEW) & (allci.baseline == app.METHODS[0]) &
               (allci.seed == 'three_fixed_models') & (allci['mode'] == 'fusion')]
    bcols = ['deployment', 'config', 'budget', 'BC_mc_ge_0p5', 'median_BC_mc', 'catastrophic_mc',
             'Dmax_le_3', 'official_frontend', 'official_hanabi']
    report = '# R68/R69 条件波形兼容性上限完整报告\n\n'
    report += f'状态：{n.STATUS}。完整目标：{result["goal_achieved"]}。本轮为自适应探索，不覆盖任何历史结果。\n\n'
    report += '## 改动\n\n只改波形通道：R66条件分类器仍为唯一打分器，在其有效支持域内，D超过历史模拟真伴随对95%阈值时不再给正奖励：\n\n'
    report += r'$$Z_W^{69}=\begin{cases}\min(Z_W^{67},0),& E_{\rm supported}\ \text{and}\ D>q_{.95},\\ Z_W^{67},&\text{otherwise}.\end{cases}$$' + '\n\n'
    report += '阈值直接继承R65模拟fold0次序统计量，不从真实关键pair、PE或官方名单选择。O3/O4a执行相同规则，各自用运行期模拟标定阈值。\n\n'
    report += 'D是近似共享/独立波形拟合的投影差，不是规范化likelihood。95%是受选模拟样本的经验尾部，不是对真实事件的覆盖保证。非正上限是待检验的保守排序约束，不是贝叶斯恒等式。它可能误伤真对，必须报告。\n\n'
    report += 'encoder和预测器没有重训，旧Mc/q评分与0.875总分混合没有恢复；时间、天空、外层权重精确冻结。公开PE只用于评分后的审计。\n\n'
    report += pd.DataFrame(configrows).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 初筛与完整目标\n\n' + pd.read_csv(pilot / 'tables/PILOT_GATES.csv').to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '两运行期初筛均通过相对NODUP风险检查和null正分减少检查，但相对R66分类损失上升。每模型tune集合都有两对真伴随被撤销正奖励。不改变阈值补救。\n\n'
    report += f'完整30注入面板、波形和三通道两种模式，既有非劣失败{len(result["injection_failed_panels"])}项；真实模型/共识预算失败{len(result["external_failed_budgets"])}项。\n\n'
    report += concise[['method', 'seed', 'reported_rank']].to_markdown(index=False) + '\n\n'
    report += '关键pair GW191103_012549--GW191105_143521 的公开Mc BC仍约0.091665。排序改变不会改变PE；它本身也是官方前端/公开Hanabi表成员，移除它可能降低官方计数。官方重合不是透镜真值。\n\n'
    report += '## 注入结果\n\n先平均每模型三个复用目录，再计算三模型mean/SD；不是新独立locked test。完整原validation/test和逐目录值均保留。\n\n'
    for mode in ('fusion', 'waveform'):
        report += '### ' + mode + '\n\n' + aggregate[aggregate['mode'] == mode].drop(columns='mode').to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '系统bootstrap查询10000次、pair2000次，同源两像共同抽取。相对NODUP三通道差值：\n\n'
    report += ci[['deployment', 'quantity', 'nominal_delta', 'delta_q025', 'delta_q975']].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '相对R67区间见tables/R67_PAIRED_DELTA_CI.csv。区间是固定数据/噪声/模型的条件波动，不包含反复选择、训练或独立噪声总体不确定度。\n\n'
    for title, records in [('未通过注入检查', result['injection_failed_panels']), ('未通过PE/官方检查', result['external_failed_budgets'])]:
        failures = pd.DataFrame(records)
        report += '### ' + title + '\n\n' + (failures.to_markdown(index=False, floatfmt='.6g') if len(failures) else '无。') + '\n\n'
    threshold = pd.read_csv(root / 'tables/F90_THRESHOLD_AND_TIE_AUDIT.csv')
    report += '### F90失败机制\n\n少量真伴随被压低，使达到90%召回需要降低阈值；新增假对主要来自更低的阈值，不是并列分数。下面同时保留旧顺序F90和纳入全部同分对的结果，不能靠改变tie规则掩盖退化。\n\n'
    report += threshold[['deployment', 'seed', 'panel', 'method', 'threshold_at_90pct',
                         'stable_index_F90', 'full_tie_FP', 'tie_true', 'tie_false']].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 真实PE和官方阶段\n\nO3严格62事件/1891对，O4a严格74事件/2701对；没有改变范围。共识摘要不能替代逐模型验收：\n\n'
    report += budgets[budgets.budget.isin([10, 20])][bcols].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += 'Top10/20/50/100及全部pair在results；三臂共识Top50、PE/官方预算、换入换出、各通道贡献、公开FPP与阶段在tables。本轮未运行Hanabi；公开表成员不等于支持透镜。\n\n'
    report += '## 完整性与限制\n\n时间、天空、权重、embedding、PE/官方字段逐pair核对；NODUP/R67历史分数和排名精确重放。旧Mc/q、PE、官方和total-blend字段投毒不影响新waveform score。完整脚本、合同、日志及失败项保留。\n\n'
    report += '数据重复用于开发；原逐像target-SNR缩放、复用透镜环境、有限合格源与近似波形profile的限制仍在。不能声称新独立人口验证或完整PE。\n\n'
    report += '- [Angelopoulos与Bates](https://arxiv.org/abs/2107.07511)：经验分位数次序规则的背景；不授予本实验真实覆盖保证。\n'
    report += '- [Lo与Magana Hernandez](https://arxiv.org/abs/2104.09339)：共享源物理动机；当前投影与上限不是Hanabi evidence。\n\n'
    report += '## 最终边界\n\n关键pair在全部模型及共识离开Top10、两个运行期PE/官方非劣、全部注入Guard须共同满足。未达到不得升级或放宽门槛，不按真实结果调整阈值。\n'
    text(root / 'reports/R68_R69_FULL_REPORT_CN.md', report)
    text(root / 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md', report)
    io.write(root / 'audit/REPORT_FILENAME_COMPATIBILITY.json', {'content_version': 'R68/R69',
        'primary': 'reports/R68_R69_FULL_REPORT_CN.md', 'alias': 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md', 'identical_content': True})
    text(root / 'README_CN.md', '# R68/R69 独立交付\n\n' + f'状态：{n.STATUS}。完整目标：{result["goal_achieved"]}。\n\n'
         + f'服务器 root@connect.westd.seetacloud.com:32328。\n\nPilot: {pilot}\n\n目录评估: {root}\n\n'
         + '完整报告reports/R68_R69_FULL_REPORT_CN.md。全部失败保留。时间、天空、checkpoint、外层权重不变。\n')
    snapshot(root, pilot)
    shutil.copy2(__file__, root / 'scripts/conditional_ceiling_report.py')
    expected = {}
    for folder in (root, pilot, root / 'uncertainty_complete', root / 'uncertainty_summary'):
        for r in pd.read_csv(folder / 'manifest/INPUT_SHA256.csv').itertuples():
            if r.path in expected and expected[r.path] != r.sha256:
                raise RuntimeError('Conflicting input hash')
            expected[r.path] = r.sha256
    checks = [{'path': p, 'before': h, 'after': io.sha(p)} for p, h in expected.items()]
    if any(r['before'] != r['after'] for r in checks):
        raise RuntimeError('Protected input changed before delivery')
    io.csv(root / 'audit/DELIVERY_INPUT_HASHES_UNCHANGED.csv', checks)
    cg = Path('/sys/fs/cgroup')
    io.write(root / 'audit/EXECUTION_RESOURCES.json', {'UTC': io.utc(), 'logical_host_CPUs': psutil.cpu_count(),
        'host_RAM_GiB': psutil.virtual_memory().total / 2**30, 'GPU_used': False,
        'cgroup': {k: (cg / k).read_text().strip() for k in ('cpu.max', 'memory.max') if (cg / k).exists()},
        'disk_free_GiB': shutil.disk_usage(P).free / 2**30, 'bootstrap_workers': 6,
        'pilot_wall_seconds': json.loads((pilot / 'contracts/PILOT_GATE.json').read_text())['seconds'],
        'whole_workflow_peak_RAM': None, 'reason_peak_unavailable': 'No process-tree RSS sampler; do not infer peak from host RAM.'})
    io.write(root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {'UTC': io.utc(), 'status': n.STATUS,
        'goal_achieved': result['goal_achieved'], 'input_hashes_verified': len(checks), 'changed_inputs': 0,
        'report': 'reports/R68_R69_FULL_REPORT_CN.md', 'report_sha256': io.sha(root / 'reports/R68_R69_FULL_REPORT_CN.md'),
        'bootstrap_panels': 24, 'no_automatic_adoption': True})
    print('CEILING_FULL_REPORT_COMPLETE', len(checks), result['goal_achieved'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.pilot)
