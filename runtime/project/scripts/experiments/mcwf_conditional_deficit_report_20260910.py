#!/usr/bin/env python3
"""R66/R67 reports, paired uncertainty and immutable delivery metadata."""
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
import mcwf_nodup_conditional_catalog_20260910 as app
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
    io.csv(root / 'tables/R62_PAIRED_DELTA_CI.csv', rows)
    return pd.DataFrame(rows)


def figures(root, per, budgets):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    labels = ['NODUP', 'R62', 'R67']
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
        fig.savefig(root / f'figures/R67_INJECTION_COMPARISON.{ext}', dpi=180)
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
        fig.savefig(root / f'figures/R67_PE_OFFICIAL_COMPARISON.{ext}', dpi=180)
    plt.close(fig)


def snapshot(root, pilot):
    modules = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(P / 'scripts'):
            modules.setdefault(Path(path).resolve(), []).append(name)
    for name in ('mcwf_nodup_conditional_audit_20260910.py', 'mcwf_timed_frozen_stage_20260910.py',
                 'mcwf_shared_profile_system_bootstrap_20260909.py', 'mcwf_shared_profile_bootstrap_summary_20260909.py',
                 'mcwf_system_bootstrap_audit_20260909.py', 'mcwf_shared_profile_package_20260909.py'):
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
    result = json.loads((root / 'audit/FINAL_GOAL_READOUT.json').read_text())
    if not (root / 'uncertainty_complete/contracts/COMPLETE.json').exists():
        raise RuntimeError('Complete bootstrap first')
    metrics = pd.read_csv(root / 'tables/RECOMPUTED_METRICS_ALL_PANELS.csv')
    keys = ['macro_r_at_1', 'macro_r_at_5', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9']
    reused = metrics[metrics.panel.str.startswith('sept8_reused_')]
    per = reused.groupby(['deployment', 'method', 'mode', 'seed'])[keys].mean().reset_index()
    aggregate = per.groupby(['deployment', 'method', 'mode'])[keys].agg(['mean', 'std'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    io.csv(root / 'tables/R67_REUSED_PER_MODEL.csv', per)
    io.csv(root / 'tables/R67_REUSED_MEAN_SD.csv', aggregate)
    critical = pd.read_csv(root / 'tables/CRITICAL_PAIR_ALL_RANKS.csv')
    critical['reported_rank'] = critical['rank'].where(critical.unit.eq('model'), critical.consensus_rank)
    fields = [c for c in ('method', 'seed', 'reported_rank', 'waveform_score', 'time_score', 'sky_raw_log_bf',
        'pe_mc_bhattacharyya_coefficient', 'pe_mc_standardized_median_distance', 'official_po_fpp', 'official_ml_fpp') if c in critical]
    concise = critical[fields].copy()
    io.csv(root / 'tables/CRITICAL_PAIR_CONCISE.csv', concise)
    budgets = pd.read_csv(root / 'tables/PE_OFFICIAL_BUDGETS.csv')
    perbudgets = pd.read_csv(root / 'tables/PER_SEED_PE_OFFICIAL_BUDGETS.csv')
    top, movements = [], []
    for dep in n.DEPS:
        scopes = [('consensus', 'consensus')] + [('model', f'seed_{s}/real') for s in n.SEEDS]
        for unit, folder in scopes:
            nd = pd.read_parquet(root / f'results/{app.METHODS[0]}/{dep}/{folder}/fusion_all_pairs.parquet')
            rank = 'consensus_rank' if unit == 'consensus' else 'rank'
            for method in app.METHODS:
                f = pd.read_parquet(root / f'results/{method}/{dep}/{folder}/fusion_all_pairs.parquet')
                f = f.sort_values(rank, kind='stable')
                if unit == 'consensus':
                    top.extend({**r, 'method': method, 'deployment': dep} for r in f.head(50).to_dict('records'))
                if method != app.NEW:
                    continue
                for budget in (10, 20):
                    before = set(nd.nsmallest(budget, rank).pair_key)
                    after = set(f.head(budget).pair_key)
                    for label, names, table in [('entered', after - before, f), ('left', before - after, nd)]:
                        for record in table[table.pair_key.isin(names)].to_dict('records'):
                            movements.append({**record, 'deployment': dep, 'unit': unit, 'model_folder': folder,
                                              'budget': budget, 'movement': label})
    io.csv(root / 'tables/REAL_CONSENSUS_TOP50_ALL_METHODS.csv', top)
    io.csv(root / 'tables/REAL_TOP10_20_ENTERED_LEFT.csv', movements)
    ci = paired_reference(root)
    figures(root, per, budgets)
    frozen = json.loads((pilot / 'configs/ALL_CONDITIONAL_MODELS.json').read_text())
    gate = json.loads((pilot / 'contracts/PILOT_GATE.json').read_text())
    coefficients = []
    for c in frozen.values():
        if c['arm'] != gate['selected_common_arm']:
            continue
        slopes = c.get('raw_slopes', [1., 0., 0.])
        coefficients.append({'deployment': c['deployment'], 'seed': c['seed'], 'ridge': c['ridge'],
            'intercept': c.get('raw_intercept', 0.), 'NODUP_slope': slopes[0], 'log_power_slope': slopes[1],
            'negative_log_deficit_slope': slopes[2], 'waveform_weight': c['weights'][0],
            'time_weight': c['weights'][1], 'sky_weight': c['weights'][2]})
    io.csv(root / 'tables/SELECTED_COEFFICIENTS_AND_OUTER_WEIGHTS.csv', coefficients)
    before, after = root / 'uncertainty/draws', root / 'uncertainty_complete/draws'
    recovery = []
    for file in sorted(before.rglob('*.parquet')):
        target = after / file.relative_to(before)
        if not target.exists() or not pd.read_parquet(file).equals(pd.read_parquet(target)):
            raise RuntimeError('Completed old bootstrap draws changed under boolean fix')
        recovery.append({'old': str(file), 'new': str(target), 'arrays_equal': True})
    io.csv(root / 'audit/BOOTSTRAP_RECOVERY_MATCH.csv', recovery)
    io.write(root / 'audit/EXECUTION_RECOVERIES.json', {'UTC': io.utc(),
        'timer_failure': 'GNU /usr/bin/time absent; no evaluation started. Python runpy/rusage wrapper ran unchanged frozen evaluation.',
        'bootstrap_failure': 'First launcher used old helper: int8 label interpreted as column names. Existing verified shared-profile helper explicitly casts boolean masks.',
        'original_failures_preserved': True, 'completed_draw_files_rechecked': len(recovery),
        'partial_output_arrays_unchanged': True, 'scientific_rule_changed': False,
        'complete_bootstrap_path': 'uncertainty_complete'})
    allci = pd.read_csv(root / 'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv')
    primaryci = allci[(allci.method == app.NEW) & (allci.baseline == app.METHODS[0]) &
        (allci.seed == 'three_fixed_models') & (allci['mode'] == 'fusion')]
    bcols = ['deployment', 'config', 'budget', 'BC_mc_ge_0p5', 'median_BC_mc', 'catastrophic_mc',
             'Dmax_le_3', 'official_frontend', 'official_hanabi']
    report = '# R66/R67 NODUP 条件共享波形证据完整报告\n\n'
    report += f'状态：`{n.STATUS}`。整体目标达成：`{result["goal_achieved"]}`。本轮是适应性开发，不是独立确认，不覆盖任何历史结果。\n\n'
    report += '## 结论与未通过项\n\n'
    report += f'R66只用模拟fit/tune，选择共同算法 `{gate["selected_common_arm"]}`。R67全部30注入目录，waveform-only与三通道融合两种评估的既有非劣门槛，失败数为{len(result["injection_failed_panels"])}；真实单模型Top10/20预算失败{len(result["external_failed_budgets"])}项。\n\n'
    report += '关键pair为GW191103_012549--GW191105_143521。公开Mc BC仍约0.091665，不修改PE。共识和三个模型必须都离开Top10，不能只看共识。\n\n'
    report += concise[['method', 'seed', 'reported_rank']].to_markdown(index=False) + '\n\n'
    report += '本轮不把官方表重合当透镜真值：这个Mc不相容pair本身就是官方前端/公开Hanabi表成员，因此把它移出可以降低官方计数。仍按作者要求原样报告，不改变验收口径。\n\n'
    report += '## 单一波形分数\n\n'
    report += '在冻结且受支持的物理拟合分支上，所选线性模型为：\n\n'
    report += '$$Z_W=b+aZ_W^{ND}+c\log P_{\min}-d\log(1+\max(D,D_{lo})),\qquad a,d\ge0.$$\n\n'
    report += '`D=P_independent-P_shared`是允许独立相位/振幅/时移、要求共享质量与等效对齐自旋时的投影损失。P不是严格PSD optimal SNR，D不是规范化对数likelihood；该模型不是完整PE或Hanabi。低D按既有支持边界饱和，支持内最终截断[-16,16]；NODUP/P超出fit支持或D超过既有上界时精确回退NODUP，不外推扣分。\n\n'
    report += '这是一次联合分类校准，不再加一份独立Mc证据，也不是0.875旧新总分混合。它读取删除旧Mc/q项后的NODUP分数作为特征。岭回归中心在fit标准化坐标中精确对应NODUP，最终仅执行一个选定函数。\n\n'
    report += 'O3/O4a使用相同形式、相同网格与选择规则；每个运行期/原冻结模型在模拟数据上各自校准系数。外层三通道权重完全冻结：\n\n'
    report += pd.DataFrame(coefficients).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '## 数据与选择\n\n'
    report += 'R22B每运行期2048个事件、1024个双像源，fit/tune各512个源；source与4096s噪声父块隔离。D测量来自R56预冻结抽样。每个完整source-pair总权重为1，再使用确切抽样包含概率的HT权重；不在合格分支内重新平衡正负类。只有活跃分支风险贡献被报告，未改变分支的损失在差值中抵消，不能把该数称作全体绝对logloss。\n\n'
    report += '候选为三特征线性与带fit真对50%/90%结点的单调分段模型。fit只用fold0；fold1选岭参数，精确平局选择NODUP，再选较强正则。两个运行期都要求逐模型风险不升、模型平均严格下降且系统bootstrap单侧95%上界不大于0。只有通过者进入完整目录评估。\n\n'
    report += pd.read_csv(pilot / 'tables/PILOT_GATES.csv').to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '数据反复用于开发，不能称新的locked test；bootstrap不包含模型选择、训练初始化或独立噪声总体的不确定度。原来的三个encoder没有重训。注入仍保留逐像target-SNR缩放、复用透镜环境的限制，不升级为response-derived或独立透镜人口证据。\n\n'
    report += '## 注入结果\n\n每个模型先平均三个复用目录，再汇总三个模型的mean/SD；全部原validation/test以及逐目录值另存，不与复用目录混为独立样本。\n\n'
    fusion = aggregate[aggregate['mode'] == 'fusion']
    report += fusion.drop(columns='mode').to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '波形单通道：\n\n' + aggregate[aggregate['mode'] == 'waveform'].drop(columns='mode').to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '相对NODUP的三通道系统bootstrap差值，查询10000次、pair2000次，双像共同抽取：\n\n'
    report += primaryci[['deployment', 'quantity', 'nominal_delta', 'delta_q025', 'delta_q975']].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '相对上一轮R62的差值另存`tables/R62_PAIRED_DELTA_CI.csv`。相对NODUP通过非劣门槛并不意味着所有指标逐点提高，也不等于超过所有历史版本。\n\n'
    report += '## 真实目录PE与官方对照\n\n严格范围为O3 62事件/1891对、O4a 74事件/2701对。公开PE与官方字段仅在评分冻结后联入。下面是共识；不能用它替代单模型验收。\n\n'
    report += budgets[budgets.budget.isin([10, 20])][bcols].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '单模型失败项完整如下；Top10/20/50/100所有值另见两个预算CSV。\n\n'
    failures = pd.DataFrame(result['external_failed_budgets'])
    report += (failures.to_markdown(index=False, floatfmt='.6g') if len(failures) else '无失败项。') + '\n\n'
    report += '全部三臂共识Top50在`tables/REAL_CONSENSUS_TOP50_ALL_METHODS.csv`；全pair与Top10/20/50/100波形/融合排名位于`results/<method>/<run>/`。每个pair保存分数贡献、PE、官方FPP与阶段。官方Hanabi表重合不等于透镜被确认，本轮没有运行Hanabi。\n\n'
    report += '## 修复、资源与完整性\n\n'
    report += f'计时器路径错误没有启动评分；改用runpy记录进程资源。bootstrap旧脚本整数标签错误的日志与部分输出保留在`uncertainty/`；正式区间在`uncertainty_complete/`，已核对{len(recovery)}个已有抽样文件与修正后数组完全一致。没有修改标签、点估计或抽样规则。\n\n'
    report += '时间、天空、外层权重与历史NODUP/R62重放共108面板逐pair核对；公开PE/官方值不变。输入哈希、运行代码快照、错误日志、合同与各层结果全部保留。\n\n'
    report += '## 依据与边界\n\n'
    report += '- [Cranmer、Pavez、Louppe](https://arxiv.org/abs/1506.02169)：分类器估计相关摘要的密度比提供方法动机，不证明本校准器输出为严格GW Bayes factor。\n'
    report += '- [Cawley与Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)：重复验证选择会带来乐观偏差，本轮维持适应性开发标签。\n'
    report += '- 共享源与独立源的物理问题遵循此前冻结的R50/R55说明；当前近似投影不是Hanabi边缘化证据。\n\n'
    report += '## 最终边界\n\n保留完整目标：关键pair须在全部模型与共识离开Top10；O3/O4a逐模型和共识的PE、官方预算及全部注入非劣门槛须同时通过。未满足则不升级，不覆盖，不自动采纳。\n'
    text(root / 'reports/R66_R67_FULL_REPORT_CN.md', report)
    text(root / 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md', report)
    io.write(root / 'audit/REPORT_FILENAME_COMPATIBILITY.json', {
        'primary': 'reports/R66_R67_FULL_REPORT_CN.md',
        'identical_package_helper_alias': 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md',
        'content_version': 'R66/R67', 'older_results_not_relabelled': True})
    short = '# R66 模拟条件校准\n\n' + report.split('## 数据与选择')[1].split('## 注入结果')[0]
    text(pilot / 'reports/R66_CONDITIONAL_CALIBRATION_CN.md', short)
    text(root / 'README_CN.md', '# R66/R67 独立交付\n\n' +
        f'状态：`{n.STATUS}`。目标完成：{result["goal_achieved"]}。\n\n' +
        '服务器：root@connect.westd.seetacloud.com，SSH端口32328。无认证信息。\n\n' +
        f'校准：`{pilot}`\n\n目录评估：`{root}`\n\n' +
        '完整报告：reports/R66_R67_FULL_REPORT_CN.md。逐模型指标/PE/官方/失败项在tables与audit；完整排名在results；正式bootstrap在uncertainty_complete。\n\n' +
        '时间、天空、外层权重不变；不恢复旧Mc/q打分或总分混合。不重训encoder，不运行PE/Hanabi。\n')
    snapshot(root, pilot)
    shutil.copy2(__file__, root / 'scripts/conditional_deficit_report.py')
    expected = {}
    for folder in (root, pilot, root / 'uncertainty_complete', root / 'uncertainty_summary'):
        for r in pd.read_csv(folder / 'manifest/INPUT_SHA256.csv').itertuples():
            if r.path in expected and expected[r.path] != r.sha256:
                raise RuntimeError('Conflicting input hash')
            expected[r.path] = r.sha256
    checks = [{'path': p, 'before': h, 'after': io.sha(p)} for p, h in expected.items()]
    if any(r['before'] != r['after'] for r in checks):
        raise RuntimeError('Changed input before delivery')
    io.csv(root / 'audit/DELIVERY_INPUT_HASHES_UNCHANGED.csv', checks)
    cg = Path('/sys/fs/cgroup')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True)
    io.write(root / 'audit/EXECUTION_RESOURCES.json', {'UTC': io.utc(), 'logical_host_CPUs': psutil.cpu_count(),
        'host_RAM_GiB': psutil.virtual_memory().total / 2**30,
        'cgroup': {k: (cg / k).read_text().strip() for k in ('cpu.max', 'memory.max') if (cg / k).exists()},
        'GPU': gpu.stdout.strip(), 'GPU_used': False, 'disk_free_GiB': shutil.disk_usage(P).free / 2**30,
        'evaluate': json.loads((root / 'audit/EVALUATE_RESOURCES.json').read_text()),
        'real': json.loads((root / 'audit/REAL_RESOURCES.json').read_text()),
        'bootstrap_workers': 6, 'not_whole_workflow_peak_RAM': True})
    io.write(root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {'UTC': io.utc(), 'status': n.STATUS,
        'goal_achieved': result['goal_achieved'], 'input_hashes_verified': len(checks), 'changed_inputs': 0,
        'report': 'reports/R66_R67_FULL_REPORT_CN.md', 'report_sha256': io.sha(root / 'reports/R66_R67_FULL_REPORT_CN.md'),
        'bootstrap_panels': 24, 'no_automatic_adoption': True})
    print('CONDITIONAL_FULL_REPORT_COMPLETE', len(checks), len(recovery), result['goal_achieved'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    a = parser.parse_args()
    main(a.root, a.pilot)
