#!/usr/bin/env python3
"""Read frozen R51 outputs; report every arm without selecting on real data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY = 'SHARED-PROFILE-SINGLE-WF'
REPLAY = 'NODUP-DIRECT-REPLAY'
CONTROL = 'SHARED-PROFILE-REJECT-ONLY'
HISTORY = 'PATH875-ARCHIVED'
METHODS = [REPLAY, PRIMARY, CONTROL]
KEY = 'GW191103_012549--GW191105_143521'


def sha(path):
    out = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            out.update(block)
    return out.hexdigest()


def md(frame):
    if frame.empty:
        return '(no rows)'
    def cell(value):
        if pd.isna(value):
            return 'NA'
        if isinstance(value, (float, np.floating)):
            return f'{value:.6g}'
        return str(value).replace('|', '/').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(map(cell, row)) + ' |'
                for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def checked_bool(series):
    if series.isna().any():
        raise RuntimeError('Missing guard outcome')
    converted = series.astype(str).str.lower().map({'true': True, 'false': False})
    if converted.isna().any():
        raise RuntimeError('Unrecognized guard outcome')
    return converted.astype(bool)


def frame_csv(path):
    return pd.read_csv(path, encoding='utf-8-sig')


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)


def main(root):
    for name in ['contracts/FROZEN_COMPUTATIONS_COMPLETE.json',
                 'audit/REFERENCE_COMPARISON_COMPLETE.json',
                 'audit/TRANSITIVE_RUNTIME_FINAL_CHECK.json',
                 'audit/FINAL_FROZEN_INPUT_INTEGRITY.json']:
        if not (root / name).exists():
            raise RuntimeError('Missing completion receipt: ' + name)
    integrity = json.loads((root / 'audit/FINAL_FROZEN_INPUT_INTEGRITY.json').read_text())
    if not integrity['passed']:
        raise RuntimeError('Frozen scientific input integrity failed')
    output = root / 'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md'
    if output.exists():
        raise RuntimeError('Final report exists; do not overwrite')
    frozen = json.loads((root / 'audit/REFERENCE_AUDIT_FROZEN.json').read_text())
    if frozen['primary_arm'] != PRIMARY:
        raise RuntimeError('Unexpected primary arm')
    tables = root / 'tables'
    pe = frame_csv(tables / 'PE_OFFICIAL_BUDGETS.csv')
    recall = frame_csv(tables / 'RETRIEVAL_SUMMARY.csv')
    per_model = frame_csv(tables / 'RETRIEVAL_PER_MODEL.csv')
    ig = frame_csv(tables / 'REFERENCE_SPECIFIC_INJECTION_GUARDS.csv')
    pg = frame_csv(tables / 'REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS.csv')
    key = frame_csv(tables / 'CRITICAL_PAIR_ALL_RANKS.csv')
    historical_key_rows = []
    history_root = root / 'results' / HISTORY / 'gwtc3'
    for path in sorted(history_root.glob('seed_*/real/fusion_all_pairs.parquet')):
        history = pd.read_parquet(path)
        for row in history[history.pair_key == KEY].to_dict('records'):
            historical_key_rows.append({**row, 'method': HISTORY,
                'seed': int(path.parts[-3].split('_')[1]), 'unit': 'model'})
    history = pd.read_parquet(history_root / 'consensus/fusion_all_pairs.parquet')
    for row in history[history.pair_key == KEY].to_dict('records'):
        historical_key_rows.append({**row, 'method': HISTORY,
            'seed': 'consensus', 'unit': 'consensus'})
    if len(historical_key_rows) != 4:
        raise RuntimeError('Historical key pair requires all four ranks')
    key = pd.concat([pd.DataFrame(historical_key_rows), key], ignore_index=True)
    invariant = frame_csv(root / 'audit/PAIR_ALIGNED_INVARIANCE.csv')
    comparison = json.loads((root / 'audit/REFERENCE_COMPARISON_COMPLETE.json').read_text())
    checks = []
    for method in METHODS:
        for dep in ('gwtc3', 'gwtc4'):
            gi = ig[(ig.method == method) & (ig.deployment == dep) & (ig.reference == REPLAY)]
            gp = pg[(pg.method == method) & (pg.deployment == dep) & (pg.reference == REPLAY)]
            if len(gi) != 30 or len(gp) != 8:
                raise RuntimeError(f'Incomplete guard panels: {method}/{dep}: {len(gi)}/{len(gp)}')
            kk = key[(key.method == method) & (key.deployment == dep)]
            key_ok = None
            if dep == 'gwtc3':
                if len(kk) != 4:
                    raise RuntimeError('Critical pair requires consensus and all three model ranks')
                ranks = np.where(kk.unit == 'consensus', kk.consensus_rank, kk['rank'])
                key_ok = bool(np.all(np.isfinite(ranks)) and np.all(ranks > 10))
            checks.append({'method': method, 'deployment': dep,
                'injection_guard_all_panels': bool(checked_bool(gi.existing_guard).all()),
                'strict_pointwise_injection_no_loss': bool(checked_bool(gi.strict_pointwise_no_loss).all()),
                'PE_official_no_loss_all_seeds_and_consensus': bool(checked_bool(gp.all_no_loss).all()),
                'critical_pair_leaves_all_top10': key_ok,
                'injection_guard_failed_panels': int((~checked_bool(gi.existing_guard)).sum()),
                'PE_official_failed_budgets': int((~checked_bool(gp.all_no_loss)).sum())})
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(tables / 'FINAL_PREDECLARED_GUARD_READOUT.csv', index=False, encoding='utf-8-sig')
    prime = check_frame[check_frame.method == PRIMARY]
    met = bool(prime.injection_guard_all_panels.all() and
        prime.PE_official_no_loss_all_seeds_and_consensus.all() and
        prime.loc[prime.deployment == 'gwtc3', 'critical_pair_leaves_all_top10'].iloc[0])
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'primary_arm': PRIMARY,
        'predeclared_primary_numerical_guards_met': met,
        'goal_achieved': False, 'author_adoption': False,
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'interpretation': 'Numerical guard summary only; no automatic method adoption or goal completion.',
        'guards': checks, 'historical_input_files_unchanged': comparison['historical_files_unchanged']}
    write_new(root / 'audit/FINAL_REPORT_READOUT.json', json.dumps(result, ensure_ascii=False, indent=2))

    selected_pe = pe[pe.config.isin([HISTORY, *METHODS]) & pe.budget.isin([10, 20])]
    selected_pe = selected_pe[['deployment', 'config', 'budget', 'BC_mc_ge_0p5',
        'median_BC_mc', 'catastrophic_mc', 'Dmax_le_3', 'official_frontend', 'official_hanabi']]
    selected_recall = recall[(recall.split == 'sept8_reused') & (recall['mode'] == 'fusion')]
    selected_recall = selected_recall[['deployment', 'method', 'macro_r_at_1_mean',
        'macro_r_at_1_std', 'macro_r_at_10_mean', 'macro_r_at_10_std',
        'average_precision_mean', 'average_precision_std',
        'false_at_recall_0p5_mean', 'false_at_recall_0p9_mean']]
    key_columns = ['method', 'seed', 'unit', 'rank', 'consensus_rank',
        'pe_mc_bhattacharyya_coefficient', 'pe_mc_standardized_distance',
        'shared_profile_deficit', 'waveform_score', 'waveform_score_mean']
    key_table = key[[c for c in key_columns if c in key]]
    key_table.to_csv(tables / 'CRITICAL_PAIR_WITH_HISTORICAL_RANKS.csv', index=False, encoding='utf-8-sig')
    failed_pe = pg[(pg.method == PRIMARY) & (pg.reference == REPLAY) & ~checked_bool(pg.all_no_loss)]
    failed_inj = ig[(ig.method == PRIMARY) & (ig.reference == REPLAY) & ~checked_bool(ig.existing_guard)]
    ci_path = root / 'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv'
    ci = frame_csv(ci_path) if ci_path.exists() else pd.DataFrame()
    if ci.empty:
        raise RuntimeError('Required paired bootstrap summary missing')

    records = pd.read_parquet(tables / 'injection_shared_profile_pairs.parquet')
    resources = records.groupby('deployment').seconds.agg(['count', 'median', 'mean', 'max'])
    resources['q90'] = records.groupby('deployment').seconds.quantile(.9)
    resources['q99'] = records.groupby('deployment').seconds.quantile(.99)
    resources.reset_index().to_csv(tables / 'PHYSICAL_PAIR_RUNTIME_SUMMARY.csv', index=False, encoding='utf-8-sig')
    group_cols = ['method', 'deployment']
    activity = invariant.groupby(group_cols)[['pairs', 'eligible_pairs', 'changed_waveform_pairs']].sum().reset_index()
    activity.to_csv(tables / 'PAIR_ACTIVITY_SUMMARY.csv', index=False, encoding='utf-8-sig')

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.1), layout='constrained')
    colors = ['#555555', '#187b66', '#b05a40']
    labels = ['NODUP', 'Shared profile', 'Reject-only']
    for row, dep in enumerate(('gwtc3', 'gwtc4')):
        for col, (metric, title) in enumerate([('macro_r_at_10', 'Injection R@10'),
                                             ('average_precision', 'Pair average precision')]):
            data = selected_recall[selected_recall.deployment == dep].set_index('method').loc[METHODS]
            axes[row, col].bar(np.arange(3), data[metric + '_mean'],
                yerr=data[metric + '_std'], color=colors, capsize=3)
            for number, method in enumerate(METHODS):
                points = per_model[(per_model.deployment == dep) &
                    (per_model.method == method) & (per_model.split == 'sept8_reused') &
                    (per_model['mode'] == 'fusion')].sort_values('seed')
                if len(points) != 3:
                    raise RuntimeError('Figure requires all three frozen model points')
                axes[row, col].scatter(number + np.linspace(-.09, .09, 3),
                    points[metric], s=18, color='white', edgecolor='#222222',
                    linewidth=.6, zorder=4)
            axes[row, col].set_xticks(np.arange(3), labels, rotation=15)
            axes[row, col].set_ylim(0, 1.05)
            axes[row, col].set_title(('O3' if row == 0 else 'O4a') + ': ' + title)
            axes[row, col].set_ylabel('Mean across frozen models')
        ax = axes[row, 2]
        x = np.arange(4)
        for m, (method, color) in enumerate(zip(METHODS, colors)):
            data = selected_pe[(selected_pe.deployment == dep) & (selected_pe.config == method)].set_index('budget')
            values = [data.loc[10, 'BC_mc_ge_0p5'], data.loc[20, 'BC_mc_ge_0p5'],
                      data.loc[10, 'official_frontend'], data.loc[20, 'official_frontend']]
            ax.bar(x + (m - 1) * .25, values, width=.24, color=color, label=labels[m])
        ax.set_xticks(x, ['Mc / 10', 'Mc / 20', 'Official / 10', 'Official / 20'], rotation=20)
        ax.set_ylim(0, 21)
        ax.set_ylabel('Consensus pair count')
        ax.set_title(('O3' if row == 0 else 'O4a') + ': External descriptive audit')
        if row == 0:
            ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, .5), ncol=1)
    (root / 'figures').mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'figures/SHARED_PROFILE_FULL_COMPARISON.{ext}', dpi=220)
    plt.close(fig)

    text = [
        '# MCWF-NODUP 共享参数波形检验：完整目录结果',
        '\n本轮为自适应开发后的独立输出版本，不是独立盲确认。历史结果、时间、天空、外层权重和 encoder checkpoint 未覆盖。最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。',
        '\n## 1. 完整验收结果',
        f'\n预先指定的主方案 `{PRIMARY}` 是否同时达到本轮数值门槛：**{met}**。这不是自动采纳、透镜确认或保证未来泛化。敏感性方案不得因真实候选更好看而替代主方案。',
        '\n' + md(check_frame),
        '\n## 2. 具体修改',
        '\n旧 encoder 及其 embedding 保持冻结。对两端均通过既有质量检查、数值拟合收敛且位于模拟支持域内的 pair，比较“分别拟合两段数据”与“强制共享质量、质量比和有效自旋后拟合”的投影拟合强度。定义 `D = P_independent - P_shared >= 0`。共享参数明显破坏拟合时，D 增大。该量与原 embedding cosine、最小单事件拟合强度一同进入一个仅用模拟 fit/tune 数据标定的波形分类器。',
        '\n主方案以这个单一波形分数替换原 waveform 分数，不叠加旧 Mc/q 回归差，不使用 0.875 或其他新旧总分混合。质量、优化或支持域不满足时，精确回退冻结的 NODUP 波形分数。拒绝型对照只允许降低原波形分数。两个运行期采用同一算法，分别标定其噪声分布。',
        '\n这不是完整 PE、规范化应变 likelihood、Hanabi 或物理 Bayes factor。当前有限搜索使用 IMRPhenomD、16 s 辅助输入及自由探测器投影振幅；原 encoder 的峰值 2 s 输入未改变。模板近似、有限优化和噪声归一化均是限制。',
        '\n## 3. 关键失败 pair',
        '\n`' + KEY + '` 的公开 Mc BC 约 0.092，保持原估计器和值，不通过改变 PE 度量美化结果。应同时看全部三个 model seed 和 consensus 排名。',
        '\n' + md(key_table),
        '\n## 4. 两运行期 PE 和官方重合',
        '\n以下是 consensus。逐 seed 和 Top-50/100 见原始预算表。官方前端通过或公开 Hanabi 表重合不是透镜真值；相关公开 Hanabi 结论仍不偏好透镜，不能改写为确认。',
        '\n' + md(selected_pe),
        '\n主方案相对当前 NODUP 未通过的 PE/官方预算（空表表示没有）：\n' + md(failed_pe),
        '\n## 5. 注入检索与假对负担',
        '\n下面使用三个已反复分析的 9 月 8 日 catalog；先在每个冻结 encoder 内平均 catalog，再计算三个模型的均值及 SD。不是本轮重新训练的方差。全部 validation、旧 test、逐 catalog、waveform-only 和三通道表均保留。',
        '\n' + md(selected_recall),
        '\n主方案相对 NODUP 未通过的逐 panel 注入非劣门槛（不以总均值掩盖单项）：\n' + md(failed_inj),
        '\n冻结门槛允许 R@10 最多下降 0.02、AP 最多下降 0.005、F50/F90 最多增加 10%；这不等于每个点值完全不降。`strict_pointwise_injection_no_loss` 单列更严格的零退化检查。',
        '\n## 6. 不确定性与数据隔离',
        '\npaired system bootstrap 保持同一源的两幅像一起抽取并按 lens family 分层。query 使用 10,000 次、pair 使用 2,000 次重采样。CI 是固定模型、固定噪声和候选集合条件下的波动，不能代替新训练或新观测目录。完整差值区间见 `uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv`。',
        '\nR50 校准 fit/tune 与本轮 catalog 的源身份及物理参数指纹没有交集；噪声 parent 分隔另有清单。有限 GW-LMC 透镜环境会重复，不能把新 source 个数等同于独立透镜人口样本。此前逐像 target-SNR 缩放及 BAYESTAR 近似仍继承，不是新的 response-derived 全 PE 数据。',
        '\n## 7. 完整性和资源',
        f'\n逐 pair 对齐审计验证 {len(invariant)} 个方法/部署/seed/panel；归档对照 {comparison["historical_files_unchanged"]} 个文件的 SHA-256 不变。源码依赖由运行时 hash guard 保护。完整输入、输出和归档校验分别保存在 manifest/audit 中。',
        '\n单 pair 运行时间（秒）：\n' + md(resources.reset_index()),
        '\n本机受 25 CPU core 配额限制，使用 24 workers。不能把宿主机可见的 208 个逻辑核称为本任务可用核。RSS 求和可能重复计入共享页，CPU 秒与 wall time 分开记录。本方案新增物理搜索成本，不能继续沿用原神经网络快速推理耗时作为完整流程速度。',
        '\n## 8. 文献依据与不能据此声称的内容',
        '\n[Cutler & Flanagan (1994)](https://arxiv.org/abs/gr-qc/9402014) 支持 inspiral 质量信息及质量/自旋相关性的物理动机，不证明本项目窗口或模型最优。',
        '\n[PyCBC hierarchical inference 示例](https://pycbc.org/pycbc/latest/html/inference/examples/hierarchical.html) 说明共享参数与多数据集联合模型的实现思路，不使本项目最大化投影成为完整 Bayesian evidence。',
        '\n[Lo & Magaña Hernandez (2023)](https://arxiv.org/abs/2104.09339) 为透镜共同源、population/selection 和确认边界提供依据；本轮没有执行该完整 Hanabi 分析。',
        '\n[Vallisneri (2008)](https://arxiv.org/abs/gr-qc/0703086) 说明局部 Fisher 近似的适用限制；本轮不将 D 当作通用卡方显著性或校准的后验标准差。',
        '\n## 9. 结果文件',
        '\n`tables/FINAL_PREDECLARED_GUARD_READOUT.csv` 为共同门槛；`CRITICAL_PAIR_ALL_RANKS.csv` 为关键 pair；`PE_OFFICIAL_BUDGETS.csv`、`PER_SEED_PE_OFFICIAL_BUDGETS.csv` 为外部预算；`results/<method>/<deployment>/consensus/fusion_all_pairs.parquet` 为完整候选；`configs/SELECTED_CONFIGURATIONS.json` 为冻结分数和权重。成功、失败和敏感性结果全部保存。',
    ]
    write_new(output, '\n'.join(text) + '\n')
    shutil.copy2(__file__, root / 'scripts/shared_profile_final_report.py')
    write_new(root / 'audit/FINAL_REPORT_SCRIPT_SHA256.json', json.dumps({
        'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__)),
        'report_sha256': sha(output)}, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
