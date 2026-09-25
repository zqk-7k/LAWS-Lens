#!/usr/bin/env python3
"""Complete R61/R62 reporting without altering scores or model selection."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_reference_regularized_catalog_20260910 as app
n = app.n


def text_once(path, value):
    if any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise RuntimeError('Unexpected control character in report')
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def reference_intervals(root):
    source = root / 'uncertainty'
    nominal = pd.read_csv(source / 'tables/CONDITIONAL_SYSTEM_CI_PER_PANEL.csv')
    target = nominal[nominal.method.eq(app.NEW)]
    rows = []
    for dep in n.DEPS:
        groups = [(str(seed), split, target[target.deployment.eq(dep) &
                    target.seed.eq(f'seed_{seed}') & (target.panel.eq('test') if split == 'test' else
                    target.panel.str.startswith('sept8_reused_'))])
                  for seed in n.SEEDS for split in ('test', 'sept8_reused')]
        groups.append(('three_fixed_models', 'sept8_reused', target[target.deployment.eq(dep) &
                       target.panel.str.startswith('sept8_reused_')]))
        for seed, split, group in groups:
            if group.empty:
                raise RuntimeError('Missing paired bootstrap group')
            for (mode, quantity), panel_rows in group.groupby(['mode', 'quantity']):
                differences, nominal_deltas = [], []
                filename = f'{mode}_query_draws.parquet' if quantity.startswith('R') else f'{mode}_pair_draws.parquet'
                for row in panel_rows.itertuples():
                    base = source / f'draws/{dep}/{row.seed}/{row.panel}'
                    a = pd.read_parquet(base / app.NEW / filename)[quantity].to_numpy()
                    b = pd.read_parquet(base / app.METHODS[1] / filename)[quantity].to_numpy()
                    if len(a) != len(b):
                        raise RuntimeError('Bootstrap draws not aligned')
                    differences.append(a - b)
                    ref = nominal[(nominal.deployment == dep) & (nominal.seed == row.seed) &
                        (nominal.panel == row.panel) & (nominal.method == app.METHODS[1]) &
                        (nominal['mode'] == mode) & (nominal.quantity == quantity)]
                    if len(ref) != 1:
                        raise RuntimeError('Ambiguous nominal bootstrap reference')
                    nominal_deltas.append(row.nominal - ref.iloc[0].nominal)
                delta = np.mean(differences, axis=0)
                rows.append({'deployment': dep, 'seed': seed, 'split': split, 'mode': mode,
                    'quantity': quantity, 'method': app.NEW, 'baseline': app.METHODS[1],
                    'nominal_delta': np.mean(nominal_deltas), 'delta_q025': np.quantile(delta, .025),
                    'delta_q975': np.quantile(delta, .975), 'repetitions': len(delta),
                    'shared_draws_across_methods_and_models': True,
                    'conditional_fixed_models_not_selection_uncertainty': True})
    n.write_csv(root / 'tables/R55_REFERENCE_PAIRED_DELTA_CI.csv', rows)
    return pd.DataFrame(rows)


def snapshot(root, pilot):
    aliases = {}
    for name, module in list(sys.modules.items()):
        value = getattr(module, '__file__', None)
        if not value:
            continue
        path = Path(value).resolve()
        if path.suffix == '.py' and path.is_relative_to(P / 'scripts'):
            aliases.setdefault(path, []).append(name)
    for filename in ('mcwf_reference_catalog_audit_20260910.py', 'mcwf_reference_catalog_units_20260910.py',
        'mcwf_shared_reference_serialization_20260910.py', 'mcwf_shared_profile_system_bootstrap_20260909.py',
        'mcwf_shared_profile_bootstrap_summary_20260909.py', 'mcwf_shared_profile_package_20260909.py'):
        aliases.setdefault(P / 'scripts/experiments' / filename, []).append('execution_entry')
    rows = []
    for path, names in sorted(aliases.items()):
        if re.search(rb'-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}', path.read_bytes()):
            raise RuntimeError('Credential in runtime source')
        dest = root / 'scripts/runtime_dependencies' / path.relative_to(P / 'scripts')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise RuntimeError('Runtime snapshot already exists')
        shutil.copy2(path, dest)
        if n.sha(path) != n.sha(dest):
            raise RuntimeError('Runtime snapshot mismatch')
        rows.append({'source': str(path), 'aliases': '|'.join(names), 'destination': str(dest.relative_to(root)),
                     'sha256': n.sha(path)})
    n.write_csv(root / 'manifest/RUNTIME_DEPENDENCIES.csv', rows)
    compatibility = json.loads((pilot / 'contracts/SERIALIZATION_COMPATIBILITY.json').read_text())
    failed = Path(compatibility['failed_preflight_root'])
    saved = []
    for row in compatibility['preserved_failed_files']:
        src = Path(row['path'])
        if n.sha(src) != row['sha256']:
            raise RuntimeError('Initial preflight failure changed')
        dest = root / 'audit/initial_preflight_failure_snapshot' / src.relative_to(failed)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        saved.append({**row, 'snapshot': str(dest.relative_to(root))})
    n.write_json(root / 'audit/PRESERVED_PREFLIGHT_FAILURE.json', {**compatibility, 'snapshots': saved})
    return len(rows)


def figure(root, per, budgets):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    names = ['NODUP', 'R55', 'R62']
    for ax, key, label in zip(axes.flat,
        ['macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'],
        ['Companion R@10', 'Pair AUPRC', 'False pairs at 50% recall', 'False pairs at 90% recall']):
        for dep, offset, color, run in [('gwtc3', -.13, '#007c91', 'O3'), ('gwtc4', .13, '#bd4551', 'O4a')]:
            for j, method in enumerate(app.METHODS):
                vals = per[(per.deployment == dep) & (per.method == method)][key].to_numpy()
                if len(vals) != 3:
                    raise RuntimeError('Missing individual model points')
                x = j + offset
                ax.scatter(x + np.linspace(-.035, .035, 3), vals, s=24, facecolors='none',
                           edgecolors=color, label=run if j == 0 else None)
                ax.plot([x - .06, x + .06], [vals.mean()] * 2, color=color, lw=2)
        ax.set_xticks(range(3), names)
        ax.set_title(label, loc='left', fontsize=11)
        ax.grid(axis='y', alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R62_INJECTION_COMPARISON.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, dep in enumerate(n.DEPS):
        for col, budget in enumerate((10, 20)):
            ax = axes[row, col]
            cols, labels = ['BC_mc_ge_0p5', 'Dmax_le_3', 'official_frontend', 'official_hanabi'], ['Mc BC >= 0.5', 'Dmax <= 3', 'Official 1%', 'Public Hanabi table']
            for j, (method, color) in enumerate(zip(app.METHODS, ['#606d76', '#007c91', '#bd4551'])):
                f = budgets[(budgets.deployment == dep) & (budgets.config == method) & (budgets.budget == budget)]
                if len(f) != 1:
                    raise RuntimeError('Missing consensus budget')
                ax.bar(np.arange(4) + (j - 1) * .23, f.iloc[0][cols].to_numpy(float), width=.21,
                       color=color, label=names[j])
            ax.set_xticks(range(4), labels, rotation=15, ha='right')
            ax.set_ylim(0, budget + 2)
            ax.set_title(('O3' if dep == 'gwtc3' else 'O4a') + f' consensus Top-{budget}', loc='left')
            ax.set_ylabel('Pair count')
            ax.grid(axis='y', alpha=.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R62_PE_OFFICIAL_COMPARISON.{ext}', dpi=180)
    plt.close(fig)


def main(root):
    report = root / 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md'
    if report.exists():
        raise RuntimeError('Independent final report required')
    readout = json.loads((root / 'audit/FINAL_GOAL_READOUT.json').read_text())
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    pilot = Path(contract['pilot'])
    gate = json.loads((pilot / 'contracts/PILOT_GATE.json').read_text())
    configs = [c for c in n.selections(root) if c['method'] == app.NEW]
    config_rows = [{'deployment': c['deployment'], 'seed': c['seed'], 'selected_dimension_arm': c['selected_common_arm'],
        'active_features': c['features'], 'no_update': c['no_update'], 'regularization': c['reference_regularization'],
        'wf_weight': c['weights'][0], 'time_weight': c['weights'][1], 'sky_weight': c['weights'][2],
        'classifier_coefficients': json.dumps(c['shared_classifier']['coefficients'])} for c in configs]
    n.write_csv(root / 'tables/SELECTED_COEFFICIENTS_AND_WEIGHTS.csv', config_rows)
    metric = pd.read_csv(root / 'tables/RETRIEVAL_SUMMARY.csv')
    display = []
    for row in metric[(metric.split == 'sept8_reused') & metric.method.isin(app.METHODS)].itertuples():
        r = row._asdict()
        display.append({'run': r['deployment'], 'method': r['method'], 'mode': r['mode'],
            **{name: f"{r[key + '_mean']:.6f} +/- {r[key + '_std']:.6f}" for name, key in
               [('R@1', 'macro_r_at_1'), ('R@10', 'macro_r_at_10'), ('AUPRC', 'average_precision'),
                ('F50', 'false_at_recall_0p5'), ('F90', 'false_at_recall_0p9')]}})
    display = pd.DataFrame(display)
    n.write_csv(root / 'tables/MAIN_RETRIEVAL_DISPLAY.csv', display)
    budgets = pd.read_csv(root / 'tables/PE_OFFICIAL_BUDGETS.csv')
    bf = budgets[budgets.config.isin(app.METHODS) & budgets.budget.isin([10, 20])]
    bc = ['deployment', 'config', 'budget', 'BC_mc_ge_0p5', 'median_BC_mc', 'catastrophic_mc',
          'Dmax_le_3', 'official_frontend', 'official_hanabi']
    key = pd.read_csv(root / 'tables/CRITICAL_PAIR_ALL_RANKS.csv')
    keycols = ['method', 'seed', 'unit', 'rank', 'consensus_rank', 'waveform_score', 'final_score']
    gf = pd.DataFrame(readout['injection_failed_panels'])
    pf = pd.DataFrame(readout['external_failed_budgets'])
    n.write_csv(root / 'tables/PRIMARY_REMAINING_INJECTION_FAILURES.csv', gf)
    n.write_csv(root / 'tables/PRIMARY_REMAINING_EXTERNAL_FAILURES.csv', pf)
    cis = pd.read_csv(root / 'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv')
    ci = cis[(cis.method == app.NEW) & (cis.baseline == app.METHODS[0]) & (cis['mode'] == 'fusion') &
             cis.quantity.isin(['R1', 'R10', 'average_precision', 'F50', 'F90'])]
    n.write_csv(root / 'tables/PRIMARY_BOOTSTRAP_DELTA_CI.csv', ci)
    r55_ci = reference_intervals(root)
    per = pd.read_csv(root / 'tables/RETRIEVAL_PER_MODEL.csv')
    per = per[(per.split == 'sept8_reused') & (per['mode'] == 'fusion') & per.method.isin(app.METHODS)]
    figure(root, per, budgets)
    grid = pd.read_csv(pilot / 'tables/REGULARIZATION_VALIDATION_GRID.csv')
    platform = grid.groupby(['deployment', 'seed', 'arm'], dropna=False).agg(
        evaluated_options=('validation_feasible', 'size'), feasible_options=('validation_feasible', 'sum'),
        minimum_full_loss=('full_population', 'min')).reset_index()
    n.write_csv(root / 'tables/VALIDATION_FEASIBILITY_PLATFORM.csv', platform)
    dependencies = snapshot(root, pilot)
    resources = []
    for stage in ('evaluate', 'real'):
        value = json.loads((root / f'logs/{stage}_COMPLETE.json').read_text())
        resources.append({'stage': stage, **value})
    resources.append({'stage': 'R61_calibration', 'wall_seconds': gate['seconds']})
    n.write_csv(root / 'tables/RESOURCE_SUMMARY.csv', resources)
    failures = gf[['deployment', 'seed', 'panel', 'mode', 'macro_r_at_10_delta', 'average_precision_delta',
                   'false_at_recall_0p5_delta', 'false_at_recall_0p9_delta']]
    ext = pf[['deployment', 'seed', 'budget', 'BC_mc_ge_0p5_delta', 'Dmax_le_3_delta',
              'official_frontend_delta', 'official_hanabi_delta']]
    combined_ci = ci[ci.aggregation.eq('same_catalog_system_draws_shared_across_models')]
    selected_grid = pd.DataFrame(config_rows)
    calibration = pd.read_csv(pilot / 'tables/PILOT_GATE_COMPARISON.csv')
    sections = [
        '# R61/R62 参考约束单一波形分类器：完整结果与失败项\n',
        f"状态：`{n.STATUS}`。**完整目标未达成，不能宣布升级。** 本文件生成于 {n.utc()}。\n",
        '## 1. 本轮结论\n',
        '本轮没有修改 encoder、一维时间、天空、外层融合权重或旧结果。没有恢复旧 encoder 的显式 Mc/q 差异分数，也没有恢复 0.875 新旧总分混合。\n',
        'R61 模拟校准通过：O3 三个模型均选择 R55 不更新；O4a 三个模型选择四特征更新。R62 将这个预先选定配置接回完整目录。关键 pair 的共识 rank 从 NODUP 的 6 降为 28，三个模型为 34/16/65。两运行期共识 Top-10/20 PE 和官方重合未下降，但 O3 三项逐模型官方预算和 O4a 三项注入非劣检查仍失败。因此只解决了关键灾难性候选，不满足全部目标。\n',
        '## 2. 改动的统计依据与公式\n',
        '此前扩充样本后的零中心正则化在 O3 普通 null 与近邻 null 之间发生权衡。本轮测试的是把新分类器参数约束在模拟数据训练的 R55 附近，而不是把所有参数拉向零。这是训练目标的变化，不是两个旧/新总分相加。\n',
        r'$$L(\beta)=\sum_i w_i[\log(1+e^{X_i\beta})-y_iX_i\beta]+\frac{\lambda}{2}\|\beta-\beta_{R55}\|^2.$$' + '\n',
        'w 为既有按 source-pair 和抽样概率校正的 class-balanced 权重。中心系数被严格变换到 fold0 新标准化坐标，包含截距；新增第四维的参考系数为0。有限差分梯度、坐标变换和单调性单元测试通过。\n',
        r'$$x=(\mathrm{cosine},\log P_{\min},-\log[1+\max(D,D_{\min})],\log BC_{\rm learned,joint}),\quad D=P_{\rm independent}-P_{\rm shared}\ge0.$$' + '\n',
        'P 是冻结波形的拟合投影量，D 是两段波形分别拟合与共享 Mc/q/对齐自旋拟合的差异。它不是已经规范化的似然或完整 PE。第四维来自冻结神经网络预测的 Mc/eta/chi 联合概率，不是公开 PE 后验。相关特征由同一个分类器联合校准，不宣称它们是独立 Bayes factor。\n',
        '在原 R55 有效性、功率与支持域内，更新分支只求值一个分类器。无更新配置逐项回放 R55；不合格 pair 严格回退到 NODUP 波形分数。后者保留既有 embedding 和新学习联合分布信息，不含被删除的旧显式 Mc/q 回归差异。必须说明这是有明确适用范围的分支方案，而不是声称所有 pair 都用了新物理拟合。\n',
        r'$$S=w_{\rm wf}Z_{\rm wf}+w_{\rm time}Z_{\rm time}+w_{\rm sky}Z_{\rm sky}.$$' + '\n',
        '支持域、低 D 饱和、高 D 正奖励回退及 cap=16 沿用 R55。总分仍只是检索排序量，不是透镜后验概率。原始 2 s encoder 输入未改；共享拟合复用已生成的 16 s 辅助波形，没有重新生成 strain。\n',
        '研究依据：[Li、Grandvalet 与 Davoine (2018)](https://proceedings.mlr.press/v80/li18a.html) 研究将参数正则化到预训练参考而非零。其论文对象为 CNN 迁移学习；本项目将该动机用于小型波形分类器，不能据此声称引力波结果必然改善或本分类器是物理后验。\n',
        '## 3. 数据隔离、选参与数值细节\n',
        '复用 R56 的1053个物理 pair测量，含285个真伴随和768个null。fold0拟合，fold1选择；source/noise-parent 隔离审核保留。扩充数据不等于全新独立透镜人口，既有 lens realization 与噪声生成限制继续适用。普通null和近邻null通过显式抽样概率及source-pair权重处理，不能把多条同源记录当独立系统。\n',
        'lambda 网格为 1e-5、3e-5、1e-4、3e-4、0.001、0.003、0.01、0.03、0.1、0.3、1、3、10、30、100，另有精确 R55 不更新选项。每个run/model只保留全人口、普通null、近邻null三个fold1损失均不差于R55的配置，再最小化全人口损失；完全相同则选不更新，其次更大lambda。3D/4D共同方案通过同一预写规则选择，真实 PE 和官方名单不参与。\n',
        '这三个fold1诊断已经参与选择，不能再作为三个独立验证证据。它是新冻结算法，不是把 R59 失败事后改成通过。共同算法选中4D，但O3合法选择不更新，实际仍是原3D分类器；不是人为单独给O3换一种方法。\n',
        selected_grid.to_markdown(index=False) + '\n',
        calibration.to_markdown(index=False, floatfmt='.6f') + '\n',
        '更新的4D分支直接读取冻结事件联合概率数组，float64逐事件归一化后计算BC。历史 float32 概率质量有约1e-8舍入误差；旧导出BC列保留，新列另名 `shared_normalized_joint_BC`。归一化差异已有误差界单元测试，不改变公开PE或时间、天空分数。\n',
        '重复使用validation和历史目录导致选择偏差，因此所有结果均为自适应开发，不能重新称为一次性locked确认。[Cawley与Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) 说明有限样本模型选择本身也会过拟合。本轮不通过挑选真实候选回调系数。\n',
        '## 4. 注入结果\n',
        '完整覆盖原validation、原test及三个已经使用过的目录，每run三个冻结模型。下表对每模型先平均三个复用目录，再报告三模型均值与样本SD；不能把九个组合说成九次独立训练。两个分数模式均列出。其他面板完整保存在CSV。\n',
        display.to_markdown(index=False) + '\n',
        '既有非劣门槛逐模型/目录和waveform/fusion分别检查，没有通过均值掩盖失败。主要参照为 NODUP；R55和PATH875另外分别列出。下列O4a三项仍失败：\n',
        failures.to_markdown(index=False, floatfmt='.6f') + '\n',
        'R@10下降不超过0.02、AUPRC下降不超过0.005、F50/F90不超过对照1.1倍是沿用的容差；同时另列严格逐项不下降，二者不可混用。O4a整体平均有所改善不代表每个面板通过。\n',
        '## 5. 关键 pair 与真实 PE/官方阶段\n',
        '`GW191103_012549--GW191105_143521` 公开 Mc BC 仍为约0.091665，Dmc约3.969372；本轮没有改变后验估计器来提高BC。共享波形D约15.208，符合适用域，不是通过缺失或OOD强行删除。其波形支持降到约-0.0795，时间和天空贡献完全保留。\n',
        key[keycols].to_markdown(index=False, floatfmt='.6f') + '\n',
        'O3真实scope为62事件、1891无序pair；O4a为74事件、2701pair。下表为跨模型共识，不是选最好seed。\n',
        bf[bc].to_markdown(index=False, floatfmt='.6f') + '\n',
        '逐seed尚有以下官方重合下降，完整目标仍失败：\n',
        ext.to_markdown(index=False, floatfmt='.6f') + '\n',
        '关键pair本身在官方前端和公开Hanabi表中，因此排除这个质量不相容pair与“所有seed官方数量不降”存在实际权衡。官方重合不等于真透镜，公开Hanabi表重合也不等于Hanabi支持透镜。本轮没有运行Hanabi或重新做公开PE。Top10/20/50/100及全部pair均保存，不能只展示共识里好看的数量。\n',
        '## 6. 不确定度\n',
        'query指标10000次、pair指标2000次配对system-bootstrap，同系统两幅像一起抽取，按family分层；复用目录的相同抽样在模型和方法间共享。CI条件于固定模型、噪声和候选集，不覆盖完整模型搜索、人口、地图或噪声变化。旧模型专属test不强行合并跨模型CI，因为没有在该审计中证明跨模型source映射。\n',
        combined_ci[['deployment', 'quantity', 'nominal_delta', 'delta_q025', 'delta_q975']].to_markdown(index=False, floatfmt='.6f') + '\n',
        '以上为相对NODUP的三模型/目录配对差值；相对R55另存 `tables/R55_REFERENCE_PAIRED_DELTA_CI.csv`。不以CI重写既有逐面板点估计门槛。\n',
        r55_ci[(r55_ci.seed == 'three_fixed_models') & (r55_ci['mode'] == 'fusion') &
               r55_ci.quantity.isin(['R10', 'average_precision', 'F50', 'F90'])][
                   ['deployment', 'quantity', 'nominal_delta', 'delta_q025', 'delta_q975']].to_markdown(index=False, floatfmt='.6f') + '\n',
        '## 7. 完整性、失败保留和资源\n',
        f"{readout['hash_checks']}项冻结输入hash核对无变化；108个method/run/model/panel通过逐pair时间、天空、权重、适用域回退和总分重构检查。18个预运行测试包括旧Mc/q、总分、PE/官方字段扰动不影响波形输出。运行依赖按真实路径去重，共{dependencies}个文件快照。\n",
        'R61首次预检遇到NumPy布尔值JSON序列化错误，没有拟合或测试结果。原失败目录及3个文件hash保留；R61B只加JSON标量转换包装器，不改数学规则，失败文件副本随包提供。R62计算和只读审计均正常结束。\n',
        pd.DataFrame(resources).to_markdown(index=False) + '\n',
        '这些耗时只包括复用既有物理拟合后的校准/目录计算，不包括早先训练、数据生成或共享拟合，不能称为完整端到端耗时。\n',
        '## 8. 文件与复现\n',
        '- `contracts/ANALYSIS_CONTRACT.json`、`configs/SELECTED_CONFIGURATIONS.json`：冻结的算法和系数。\n- `tables/RETRIEVAL_PER_SEED.csv`、`RETRIEVAL_PER_MODEL.csv`、`RETRIEVAL_SUMMARY.csv`：全部注入面板。\n- `tables/REFERENCE_SPECIFIC_INJECTION_GUARDS.csv`、`REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS.csv`：全部参考对照，不只正结果。\n- `results/<method>/<run>/seed_<seed>/real/`：逐模型全pair、贡献及PE/官方字段。\n- `results/<method>/<run>/consensus/`：共识全pair和Top10/20/50/100。\n- `tables/PE_OFFICIAL_BUDGETS.csv`、`PER_SEED_PE_OFFICIAL_BUDGETS.csv`：预算表。\n- `uncertainty/`、`uncertainty_summary/`：system-bootstrap抽样与区间。\n- `scripts/runtime_dependencies/`：运行脚本和依赖快照；大型输入通过输入manifest定位，不随紧凑包重复分发。\n- `audit/FINAL_GOAL_READOUT.json`：机器可读完整目标判定，当前为false。\n',
        f'服务器：`connect.westd.seetacloud.com:32328`。R61目录：`{pilot}`。R62目录：`{root}`。历史R55不覆盖。\n',
        '## 9. 结论边界\n',
        '共同目标尚未实现。不能把“关键pair退出Top10、共识PE改善”写成“所有PE/官方/注入指标均不下降”；也不能因O4a校准更好就宣布新主线。现有证据只支持把R61/R62保留为完整探索对照。下一步需研究真实泛化失败和分支间评分尺度，不应拿官方名单补排名或继续无界搜参。\n',
        f'最终状态：`{n.STATUS}`。\n']
    text_once(report, '\n'.join(sections))
    text_once(root / 'README_CN.md', '# R61/R62 交付入口\n\n完整目标尚未达成。请先阅读 `reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md` 和 `audit/FINAL_GOAL_READOUT.json`。\n\n本包不覆盖历史结果，不含原始strain、PE HDF5、私钥或大型可重建缓存。两个运行期方法统一，时间、天空和外层权重冻结。\n')
    text_once(pilot / 'reports/R61_PILOT_REPORT_CN.md', '# R61 模拟校准报告\n\n校准Gate通过不代表最终目标通过。\n\n' + calibration.to_markdown(index=False, floatfmt='.6f') + '\n\n' + selected_grid.to_markdown(index=False) + '\n\n完整后续结果位于 `' + str(report) + '`，其中O3逐模型官方预算和O4a注入仍有失败。原序列化失败与恢复合同均保留。\n')
    n.write_json(root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {'UTC': n.utc(),
        'computations_complete': True, 'goal_achieved': bool(readout['goal_achieved']), 'status': n.STATUS,
        'stages': ['simulation calibration', 'preflight units', 'all injection panels', 'postfreeze real audit',
                   'system bootstrap', 'reference-specific invariance/guard audit', 'report'],
        'report': str(report.relative_to(root)), 'historical_overwrite': False,
        'future_actions_not_authorized_by_this_receipt': ['adoption', 'paper changes', 'real-outcome tuning']})
    print('R61_R62_REPORT_COMPLETE', dependencies, len(r55_ci), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
