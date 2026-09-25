#!/usr/bin/env python3
"""Read-only final R64/R65 reporting, including every failed comparison."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_deficit_tail_catalog_20260910 as app
import mcwf_branch_accounting_20260910 as accounting
import mcwf_reference_catalog_report_20260910 as report_helpers
cal, n = app.cal, app.n


def write_text(path, value):
    if any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise RuntimeError('Report contains control characters')
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(value)


def table(f):
    return f.to_markdown(index=False, floatfmt='.6f') + '\n\n'


def snapshot(root, failures=()):
    sources = {}
    for name, module in list(sys.modules.items()):
        value = getattr(module, '__file__', None)
        if value and str(P / 'scripts') in value and Path(value).suffix == '.py':
            sources.setdefault(Path(value).resolve(), []).append(name)
    for filename in ('mcwf_shared_tail_catalog_audit_20260910.py',
                     'mcwf_shared_profile_system_bootstrap_20260909.py',
                     'mcwf_shared_profile_bootstrap_summary_20260909.py',
                     'mcwf_shared_profile_package_20260909.py'):
        sources.setdefault(P / 'scripts/experiments' / filename, []).append('execution_entry')
    rows = []
    for source, aliases in sorted(sources.items()):
        target = root / 'scripts/runtime_dependencies' / source.relative_to(P / 'scripts')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError('Immutable dependency snapshot already exists')
        shutil.copy2(source, target)
        if cal.sha(source) != cal.sha(target):
            raise RuntimeError('Dependency copy mismatch')
        rows.append({'source': str(source), 'destination': str(target.relative_to(root)),
                     'sha256': cal.sha(source), 'aliases': '|'.join(aliases)})
    cal.csv(root / 'manifest/RUNTIME_DEPENDENCIES.csv', rows)
    saved = []
    for failed in failures:
        for source in sorted(failed.rglob('*')):
            if not source.is_file():
                continue
            target = root / 'audit/implementation_recoveries' / failed.name / source.relative_to(failed)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if cal.sha(source) != cal.sha(target):
                raise RuntimeError('Failure snapshot mismatch')
            saved.append({'source': str(source), 'destination': str(target.relative_to(root)), 'sha256': cal.sha(source)})
    cal.write(root / 'audit/IMPLEMENTATION_RECOVERIES.json', {'UTC': cal.utc(), 'snapshots': saved,
        'failed_before_fit': True, 'original_roots_preserved': True,
        'scope': 'R64 initializer error; R64B all-events profile assumption; R64C per-event versus source-expanded profile-plan assumption. All recovered by exact original plan, not changing physical eligibility.',
        'bootstrap_launcher': 'An initial shell-delivered Python command had a missing closing list bracket and failed parsing before launching or writing outputs; corrected launcher executed unchanged bootstrap program.'})
    return len(rows)


def r64_report(root, failures):
    gate = json.loads((root / 'contracts/PILOT_GATE.json').read_text())
    data = {name: pd.read_csv(root / f'tables/{name}.csv') for name in
            ('POPULATION_AUDIT', 'ELIGIBILITY_RATES', 'PILOT_GATES', 'GLOBAL_RISK_METRICS', 'GLOBAL_RISK_BOOTSTRAP')}
    c = json.loads((root / 'configs/ALL_CALIBRATIONS.json').read_text())
    configs = pd.DataFrame([{'deployment': key.split('/')[0], 'seed': key.split('/')[1],
        'arm': key.split('/')[2], **{k: value[k] for k in ('intercept', 'slope', 'ridge', 'no_update')}}
        for key, value in c.items()])
    cal.csv(root / 'tables/SELECTED_AFFINE_PARAMETERS.csv', configs)
    text = '# R64 总体计权与波形分支校准完整报告\n\n'
    text += f'状态：`{cal.STATUS}`。模拟 Gate：**{gate["gate"]}**；没有运行新真实排名，没有替换 R62 或任何历史结果。\n\n'
    text += '## 问题与统计单位\n\n'
    text += 'R62 在合格事件对内平衡正负样本。该分支随后与不合格分支的 NODUP 分数竞争，不能仅凭条件 logit 宣称二者同尺度。R64 检验总体计权，而不是把目录内分数重新 z-score。\n\n'
    text += '每个 run 有 1024 个模拟源，fit/tune 各512个，每源两幅像。每个不同源组合是一个 null source-pair；只有噪声父区块不同的 image pairs 进入允许总体。每个允许 source-pair 的全部像对权重之和严格为1。\n\n'
    text += r'$$w_{ij}^{N}=\frac{1}{m_{ij}\pi_{ij}},\qquad w_{ij}^{\rm global}=\begin{cases}(2N_L)^{-1},&L,\\(2N_Nm_{ij}\pi_{ij})^{-1},&N.\end{cases}$$' + '\n\n'
    text += '这里 m 是整个允许总体的像对数，不是质量门控后的像对数；pi 是 R56 冻结的不等概率抽样纳入率。真实源使用完整合格 census，不重复计算两幅像。\n\n'
    text += table(data['POPULATION_AUDIT'])
    text += '## 两个固定检验\n\n'
    text += '1. ELIGIBILITY-OFFSET：仅在 fit 上估计 log[P(E|L)/P(E|N)]，加到合格分支。因为原分数不是精确条件 LR，而且此前条件权重不同，这只是近似控制，不是被证明缺失的 Bayes factor。\n'
    text += '2. GLOBAL-AFFINE：对合格 R62 波形分数拟合正斜率仿射变换，向恒等变换正则化。仍是一个 waveform score，不加旧 Mc/q，不混合旧新总分。\n\n'
    text += '未触发分支的分数完全不变，因此全体 logloss 差值等于合格分支贡献的差值；合格贡献本身不是完整总体的绝对 logloss。fit/tune 类别不在 E 内重新归一化。\n\n'
    text += table(data['ELIGIBILITY_RATES']) + table(data['PILOT_GATES'])
    text += table(configs[configs.arm == 'GLOBAL-AFFINE'])
    text += '## 判断\n\n'
    text += 'O3 三个仿射模型全部选择不更新；O4a 选到的小修正没有通过 source-bootstrap 支持门槛。近似 offset 也没有在两个运行期同时通过。因此没有把失败校准带入真实目录，也没有为了匹配已知 PE 调整截距。\n\n'
    text += '这不是证明分支尺度绝无问题，只说明当前有限开发总体、这些固定变换和验证条件不足以支持采用。source bootstrap 条件于固定样本和抽样权重，不包括共享噪声或抽样设计全部不确定性。\n\n'
    text += '## 来源与限制\n\n'
    text += '旧 R23/R24 已研究过一般的分区概率校准；本次不把这一思想当作首次发现。不同点是当前使用实际 pair 共享拟合 deficit D、双端质量门控和功率支持域。\n\n'
    text += '分类器密度比只提供方法动机，并不保证这里是物理 Bayes factor。[Cranmer et al.](https://arxiv.org/abs/1506.02169)。反复使用验证集的结果不是独立确认。[Cawley & Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)。\n\n'
    text += 'R22B 有 source/noise-parent 跨 fold 隔离，但仍复用 GW-LMC lens environments、逐像 target-SNR 缩放；不是新的独立透镜人口。\n\n'
    count = snapshot(root, failures)
    checked = cal.check(root)
    text += f'## 完整性\n\n{checked} 个输入 hash 再次通过；保存 {count} 个去重运行依赖和全部3次拟合前失败日志。目录名只是独立标识，实际 UTC 以冻结合同和日志为准。所有科学拟合与门槛在成功执行前已冻结。\n'
    write_text(root / 'reports/R64_GLOBAL_BRANCH_CALIBRATION_CN.md', text)


def diagnostics(root):
    rows, flags, moves = [], [], []
    for p in sorted((root / f'results/{app.NEW}').glob('gwtc*/seed_*/*/pairs.parquet')):
        rel = p.relative_to(root / f'results/{app.NEW}')
        dep, seedname, panel = rel.parts[:3]
        f = pd.read_parquet(p)
        nd = pd.read_parquet(root / 'results' / app.METHODS[0] / rel)
        prev = pd.read_parquet(root / 'results' / app.METHODS[1] / rel)
        y = f.is_true_pair.to_numpy(bool)
        active = f.shared_profile_eligible.to_numpy(bool)
        flag = f.deficit_tail_flag.to_numpy(bool)
        changed = f.waveform_score.to_numpy() != nd.waveform_score.to_numpy()
        common = {'deployment': dep, 'seed': int(seedname.split('_')[1]), 'panel': panel}
        for label, mask in [('true', y), ('null', ~y)]:
            vals = f.shared_profile_deficit.to_numpy()[mask & active]
            flags.append({**common, 'label': label, 'all_pairs': int(mask.sum()),
                'eligible': int((mask & active).sum()), 'tail_flag': int((mask & flag).sum()),
                'attenuated': int((mask & changed).sum()),
                'tail_rate_within_eligible': float(flag[mask & active].mean()) if (mask & active).any() else None,
                'median_D': float(np.median(vals)) if len(vals) else None,
                'q95_D': float(np.quantile(vals, .95)) if len(vals) else None})
        for method, ref in [(app.METHODS[0], nd), (app.METHODS[1], prev)]:
            for mode, col in [('waveform', 'waveform_score'), ('fusion', 'final_score')]:
                for recall in (.5, .9):
                    result = accounting.decompose(ref[col].to_numpy(), f[col].to_numpy(), y, active, recall)
                    rows.append({**common, 'reference': method, 'method': app.NEW, 'mode': mode,
                                 'recall': recall, **result})
    for dep in n.DEPS:
        for unit in ('consensus', *[f'seed_{s}' for s in n.SEEDS]):
            folder = 'consensus' if unit == 'consensus' else unit + '/real'
            f = pd.read_parquet(root / f'results/{app.NEW}/{dep}/{folder}/fusion_all_pairs.parquet')
            for reference in app.METHODS[:2]:
                old = pd.read_parquet(root / f'results/{reference}/{dep}/{folder}/fusion_all_pairs.parquet')
                for budget in (10, 20):
                    left = set(old.head(budget).pair_key) - set(f.head(budget).pair_key)
                    entered = set(f.head(budget).pair_key) - set(old.head(budget).pair_key)
                    for state, keys, df in [('left', left, old), ('entered', entered, f)]:
                        for row in df[df.pair_key.isin(keys)].to_dict('records'):
                            fields = {k: v for k, v in row.items() if k.startswith(('pe_', 'official_')) or
                                      k in ('pair_key', 'rank', 'consensus_rank', 'final_score', 'final_score_mean')}
                            moves.append({**fields, 'deployment': dep, 'unit': unit, 'budget': budget,
                                'reference': reference, 'method': app.NEW, 'change': state})
    cal.csv(root / 'tables/TAIL_FLAG_RATES_ALL_PANELS.csv', flags)
    cal.csv(root / 'tables/TAIL_FALSE_BURDEN_DECOMPOSITION.csv', rows)
    cal.csv(root / 'tables/REAL_BUDGET_ENTERED_LEFT.csv', moves)
    cal.write(root / 'audit/TAIL_DECOMPOSITION_UNITS.json', accounting.units())
    return pd.DataFrame(flags), pd.DataFrame(rows)


def figures(root, per, budgets):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    names = ['NODUP', 'R62', 'Tail95']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, key, label in zip(axes.flat, ['macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'],
         ['Companion R@10', 'Pair AUPRC', 'False pairs at 50% recall', 'False pairs at 90% recall']):
        for dep, offset, color, name in [('gwtc3', -.14, '#007c91', 'O3'), ('gwtc4', .14, '#bd4551', 'O4a')]:
            for j, method in enumerate(app.METHODS):
                vals = per[(per.deployment == dep) & (per.method == method)][key].to_numpy()
                if len(vals) != 3:
                    raise RuntimeError('Three model points required')
                ax.scatter(j + offset + np.linspace(-.025, .025, 3), vals, s=24,
                           facecolors='none', edgecolors=color, label=name if j == 0 else None)
                ax.plot([j + offset - .06, j + offset + .06], [vals.mean()] * 2, color=color)
        ax.set_xticks(range(3), names)
        ax.set_title(label, loc='left', fontsize=10)
        ax.grid(axis='y', alpha=.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', frameon=False, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, .94])
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R65_INJECTION_COMPARISON.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    fields = ['BC_mc_ge_0p5', 'Dmax_le_3', 'official_frontend', 'official_hanabi']
    for i, dep in enumerate(n.DEPS):
        for j, budget in enumerate((10, 20)):
            ax = axes[i, j]
            for k, (method, color) in enumerate(zip(app.METHODS, ['#606d76', '#007c91', '#bd4551'])):
                f = budgets[(budgets.deployment == dep) & (budgets.config == method) & (budgets.budget == budget)]
                if len(f) != 1:
                    raise RuntimeError('Ambiguous consensus budget')
                ax.bar(np.arange(4) + (k - 1) * .23, f.iloc[0][fields].to_numpy(float), width=.21, color=color, label=names[k])
            ax.set_xticks(range(4), ['Mc BC >= 0.5', 'Dmax <= 3', 'Official 1%', 'Hanabi table'], rotation=15, ha='right')
            ax.set_ylim(0, budget + 2)
            ax.set_ylabel('Pair count')
            ax.set_title(('O3' if dep == 'gwtc3' else 'O4a') + f' consensus Top-{budget}', loc='left', fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, .94])
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R65_PE_OFFICIAL_COMPARISON.{ext}', dpi=180)
    plt.close(fig)


def r65_report(root, global_root):
    for required in ('uncertainty/contracts/COMPLETE.json',
                     'uncertainty_summary/contracts/SUMMARY_CONTRACT.json',
                     'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv'):
        if not (root / required).is_file():
            raise RuntimeError('Finish system bootstrap and summary first: ' + required)
    result = json.loads((root / 'audit/FINAL_GOAL_READOUT.json').read_text())
    thresholds = pd.read_csv(root / 'tables/TAIL_VALIDATION.csv')
    rates, decomposition = diagnostics(root)
    metrics = pd.read_csv(root / 'tables/RETRIEVAL_SUMMARY.csv')
    per = pd.read_csv(root / 'tables/RETRIEVAL_PER_MODEL.csv')
    consensus = pd.read_csv(root / 'tables/PE_OFFICIAL_BUDGETS.csv')
    modelbudgets = pd.read_csv(root / 'tables/PER_SEED_PE_OFFICIAL_BUDGETS.csv')
    critical = pd.read_csv(root / 'tables/CRITICAL_PAIR_ALL_RANKS.csv')
    critical['reported_rank'] = critical['rank'].where(critical.unit == 'model', critical.consensus_rank)
    cal.csv(root / 'tables/CRITICAL_PAIR_CONCISE.csv', critical[[c for c in ('method', 'seed', 'unit', 'reported_rank',
        'waveform_score', 'shared_profile_deficit', 'deficit_tail_cutoff', 'deficit_tail_flag',
        'pe_mc_bhattacharyya_coefficient', 'pe_mc_standardized_distance') if c in critical]])
    report_helpers.app = app
    reference_ci = report_helpers.reference_intervals(root)
    # Keep the helper's historical filename, but identify the actual reference in a manifest.
    cal.write(root / 'audit/PAIRED_CI_FILENAME_COMPATIBILITY.json', {'UTC': cal.utc(),
        'file': 'tables/R55_REFERENCE_PAIRED_DELTA_CI.csv', 'actual_reference': app.METHODS[1],
        'reason': 'Unmodified report helper filename; CSV baseline column is authoritative.'})
    figures(root, per[(per['split'] == 'sept8_reused') & (per['mode'] == 'fusion')], consensus)
    text = '# R64/R65 完整实验报告：分支校准与共享拟合尾部限制\n\n'
    text += f'状态：`{cal.STATUS}`。本轮完整目标是否达到：**{result["goal_achieved"]}**。不能将关键 pair 改善或共识表通过，替代全部逐模型与注入门槛。\n\n'
    text += '## 本轮实际做了什么\n\n'
    text += f'R64 独立目录：`{global_root}`。总体计权的 offset/affine 校准未通过两个运行期共同检验，因此没有生成真实重排。详见该目录中文报告。\n\n'
    text += f'R65 独立目录：`{root}`。只在共享源拟合异常的尾部降低波形分数。没有训练 encoder，没有修改一维时间、天空图、外层权重；没有恢复旧 encoder Mc/q 打分，也没有恢复0.875总分混合。\n\n'
    text += '## R65 的完整计算\n\n'
    text += r'$$D=P_{\rm independent}-P_{\rm shared}\ge0,\quad k=\lceil(n_L+1)0.95\rceil,\quad d_{95}=D_{(k)}.$$' + '\n\n'
    text += 'D 来自冻结的16秒辅助波形共同质量/质量比/等效自旋拟合；它是 projection deficit，不是 full-PE 对数 Bayes factor。每个运行期只用 fit 的合格真源 doublets 决定一个 d95，三个 encoder seeds 共用该运行期阈值，alpha 固定0.05，没有参数网格。\n\n'
    text += r'$$Z_{\rm wf}^{65}=\begin{cases}\min(Z_{\rm wf}^{\rm NODUP},Z_{\rm wf}^{62}),&E\land D>d_{95},\\Z_{\rm wf}^{\rm NODUP},&\text{otherwise}.\end{cases}$$' + '\n\n'
    text += r'$$S=w_WZ_{\rm wf}^{65}+w_TZ_{\rm time}+w_SZ_{\rm sky}.$$' + '\n\n'
    text += '只取一个波形分数，不把两个总分相加。min 是保守排序约束，不是相互独立证据的乘法；旧 Mc/q 项仍未参与。外层 w 完全沿用 NODUP/R62。\n\n'
    text += table(thresholds)
    text += '验证 true-tail 的区间上界约11%–14%，不能声称已经证明误拒绝率小于5%。本轮预冻结 Gate 只检查没有显著过度拒绝，以及 null 尾部大于 true 尾部；真正采用仍需全部目录指标。\n\n'
    text += '## 关键 pair\n\n'
    text += table(critical[['method', 'seed', 'unit', 'reported_rank', 'pe_mc_bhattacharyya_coefficient']])
    text += '公开 Mc 后验及 BC 估计方法没有修改。BC 约0.091665，不因降序而变好；修正的是检索不应把这类不相容 pair 放在头部。官方 Hanabi 表列出该 pair 也不是透镜确认。\n\n'
    text += '## 注入结果\n\n'
    text += '先在同一模型的三个复用目录内平均，再对三个固定模型计算 mean/SD，不把九个模型-目录组合称为九次独立训练。全部旧 validation/test 与复用目录明细保存在逐seed表。以下是三目录平均的融合结果。\n\n'
    cols = ['deployment', 'method', 'macro_r_at_1_mean', 'macro_r_at_10_mean', 'macro_r_at_10_std',
        'average_precision_mean', 'false_at_recall_0p5_mean', 'false_at_recall_0p9_mean']
    text += table(metrics[(metrics['split'] == 'sept8_reused') & (metrics['mode'] == 'fusion')][cols])
    text += '波形单独结果：\n\n' + table(metrics[(metrics['split'] == 'sept8_reused') & (metrics['mode'] == 'waveform')][cols])
    text += '## 真实 PE 与官方阶段\n\n'
    fields = ['deployment', 'config', 'budget', 'BC_mc_ge_0p5', 'median_BC_mc', 'catastrophic_mc',
        'Dmax_le_3', 'official_frontend', 'official_hanabi']
    text += table(consensus[consensus.budget.isin([10, 20])][fields])
    text += 'Top50/100 及所有单模型均完整保存，所有 pair 的 PE 和官方列是在排序之后联表。官方1%为公开前端 FPP 的既有阈值；公开 Hanabi 表重合是成员数，不是支持透镜数。\n\n'
    text += '## 未通过项必须保留\n\n'
    text += f'逐模型/目录注入门槛失败 {len(result["injection_failed_panels"])} 项；逐模型 PE/官方预算失败 {len(result["external_failed_budgets"])} 项。共识 PE/官方不退化：{result["consensus_PE_official_no_loss"]}；全部单模型不退化：{result["per_model_PE_official_no_loss"]}。\n\n'
    text += table(pd.DataFrame(result['injection_failed_panels']))
    text += table(pd.DataFrame(result['external_failed_budgets']))
    text += '本轮未把旧失败门槛放宽为只看平均值，也未从三个 seed 中挑选真实 PE 好看的一个。R65 未达到用户的完整共同目标，不能替代 R62、NODUP、C-fixed 或论文结果。\n\n'
    text += '## 固定召回率假对的解释\n\n'
    text += '即使只降低一些分数，F50/F90也不一定降低：部分真对下降后，达到同样召回率所需的阈值也下降，原本未改变的假对会进入集合。完整对称分数/阈值效应在 TAIL_FALSE_BURDEN_DECOMPOSITION.csv；这是恒等式分解，不是因果结论。\n\n'
    failed = pd.DataFrame(result['injection_failed_panels'])
    explain = decomposition[decomposition.reference == app.METHODS[0]].merge(
        failed[['deployment', 'seed', 'panel', 'mode']].drop_duplicates(), on=['deployment', 'seed', 'panel', 'mode'])
    text += table(explain[['deployment', 'seed', 'panel', 'mode', 'recall', 'false_before', 'false_after',
                          'score_effect', 'threshold_effect', 'entered_inactive_false', 'left_active_false']])
    text += '## 不确定度\n\n'
    text += '每panel query 10000次、pair指标2000次 source-system bootstrap；两幅像一起抽取，方法间复用相同draws。复用目录跨模型也对齐draws。它仅表示固定模型、固定样本条件下的抽样波动，不含新训练、采样设计或噪声共享的全部不确定性。\n\n'
    text += '与 R62 的成对差值区间：\n\n' + table(reference_ci[(reference_ci.seed == 'three_fixed_models') &
        (reference_ci['mode'] == 'fusion') & reference_ci.quantity.isin(['R10', 'average_precision', 'F50', 'F90'])])
    text += '与 NODUP/PATH875 的区间在 uncertainty_summary；R62对照表文件名为兼容旧报告保留 R55 字样，baseline列实际明确为 R62-FROZEN-REPLAY。\n\n'
    text += '## 方法依据与边界\n\n'
    text += '有限样本分位点借鉴 split-conformal 的次序统计量。由于已有适应性开发、选择域和真实域偏移，本项目不能据此宣称严格的 real-GWTC coverage。[Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)。\n\n'
    text += '共享源假设的物理检验应最终使用联合似然及 population/selection。本轮低维 projection 和一维 PE 审计都不是 Hanabi 确认。[Lo & Magaña Hernandez](https://arxiv.org/abs/2104.09339)。\n\n'
    text += '反复看真实候选指导研究方向是适应性开发，不能因本次系数来自模拟验证集就抹去之前的真实反馈。需要新的未触碰数据才可做独立确认。[Cawley & Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)。\n\n'
    count = snapshot(root)
    report_path = root / 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md'
    text += f'## 交付与保护\n\n{result["hash_checks"]}个冻结输入hash不变；{result["invariance_panels"]}个方法/面板的时间、天空、权重、fallback和总分重构通过。{count}个运行依赖已去重保存。完整Top10/20/50/100、所有pair分数、失败项、配置、日志、bootstrap draws均保留。最终包不含strain、PE HDF5、私钥或大型缓存。\n'
    write_text(report_path, text)
    shutil.copy2(__file__, root / 'scripts/global_and_tail_report.py')
    cal.write(root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {'UTC': cal.utc(),
        'status': cal.STATUS, 'goal_achieved': result['goal_achieved'], 'not_adopted': True,
        'report': str(report_path.relative_to(root)), 'report_sha256': cal.sha(report_path),
        'input_hash_checks': result['hash_checks'], 'runtime_dependencies': count,
        'all_injection_and_real_arms_reported': True, 'bootstrap_completed': True})
    print('FINAL_REPORT_COMPLETE', root, 'goal', result['goal_achieved'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--global-root', required=True, type=Path)
    p.add_argument('--tail-root', required=True, type=Path)
    p.add_argument('--failed-root', nargs='*', default=[], type=Path)
    a = p.parse_args()
    r64_report(a.global_root, a.failed_root)
    r65_report(a.tail_root, a.global_root)
