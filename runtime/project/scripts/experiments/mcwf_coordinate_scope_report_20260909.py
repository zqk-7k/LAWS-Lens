#!/usr/bin/env python3
"""Finalize R59/R60 diagnostics without inventing new catalog results."""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
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
import mcwf_shared_deficit_coordinate_20260909 as c
n = c.n


def bootstrap(root):
    table = pd.read_parquet(root / 'tables/PREDICTIONS.parquet')
    rng = np.random.default_rng(2026090959)
    results = []
    for dep in n.DEPS:
        f = table[(table.deployment == dep) & (table.fold == 1)]
        unique = f.drop_duplicates('pair_id').set_index('pair_id').sort_index()
        source = sorted(set(unique.source_i) | set(unique.source_j))
        index = {s: i for i, s in enumerate(source)}
        counts = rng.multinomial(len(source), np.full(len(source), 1 / len(source)), size=2000)
        y = unique.kind.eq('true').to_numpy(float)
        a, b = unique.source_i.map(index).to_numpy(), unique.source_j.map(index).to_numpy()
        multiplicity = np.where(y[None, :] == 1, counts[:, a], counts[:, a] * counts[:, b])
        losses = {}
        for arm in ('R55-FROZEN', *c.ARMS):
            values = []
            for seed in n.SEEDS:
                z = f[(f.arm == arm) & (f.seed == seed)].set_index('pair_id').loc[unique.index, 'waveform_score'].to_numpy()
                values.append(np.logaddexp(0., z) - y * z)
            losses[arm] = np.mean(values, axis=0)
        diagnostics = {'full_population': np.ones(len(unique), bool),
            'random_draw': unique.kind.ne('hard_null').to_numpy(),
            'neighbor_population': unique.neighbor_population.to_numpy() | (y == 1)}
        for diagnostic, take in diagnostics.items():
            raw = unique.HT_weight.to_numpy() if diagnostic == 'full_population' else unique.source_pair_weight.to_numpy()
            w = multiplicity[:, take] * raw[take][None, :]
            labels = y[take]
            for klass in (0, 1):
                group = labels == klass
                mass = w[:, group].sum(1)
                if np.any(mass <= 0):
                    raise RuntimeError('Zero bootstrap class mass')
                w[:, group] *= .5 / mass[:, None]
            fixed = c.classifier.base.balanced(labels, raw[take])
            for arm in c.ARMS:
                difference = (losses[arm] - losses['R55-FROZEN'])[take]
                sample = w @ difference
                results.append({'deployment': dep, 'arm': arm, 'diagnostic': diagnostic,
                    'point_delta_logloss': float(fixed @ difference),
                    'percentile95_low': float(np.quantile(sample, .025)),
                    'percentile95_high': float(np.quantile(sample, .975)),
                    'draws': len(sample), 'source_groups': len(source),
                    'unit': 'source system, shared across models and arms; null endpoint product',
                    'conditioned_on_fixed_noise_and_fitted_models': True,
                    'selection_uncertainty_included': False, 'used_for_selection': False})
    n.write_csv(root / 'tables/CONDITIONAL_SOURCE_BOOTSTRAP_CI.csv', results)


def main(root, scope, reference):
    c.check(root)
    gate = json.loads((root / 'contracts/PILOT_GATE.json').read_text())
    if gate['gate'] != 'FAIL':
        raise RuntimeError('This report is for the completed failed R59 pilot')
    if (root / 'reports/R59_R60_COMPLETE_REPORT_CN.md').exists():
        raise RuntimeError('Do not overwrite completed report')
    scope_gate = json.loads((scope / 'contracts/PILOT_GATE.json').read_text())
    if scope_gate['gate'] != 'AUDIT_COMPLETE':
        raise RuntimeError('Scope audit incomplete')
    checks = []
    for r in (root, scope):
        for row in pd.read_csv(r / 'manifest/INPUT_SHA256.csv').itertuples():
            after = n.sha(Path(row.path))
            checks.append({'experiment': r.name, 'path': row.path,
                           'before': row.sha256, 'after': after, 'exact': after == row.sha256})
        if (r / 'results').exists() and list((r / 'results').rglob('*.parquet')):
            raise RuntimeError('No catalog scores are authorized for these diagnostics')
    if not all(r['exact'] for r in checks):
        raise RuntimeError('Historical input changed')
    n.write_csv(root / 'audit/FINAL_INPUT_HASH_CHECKS.csv', checks)
    bootstrap(root)
    metrics = pd.read_csv(root / 'tables/CALIBRATION_METRICS_PER_SEED.csv')
    compare = pd.read_csv(root / 'tables/PILOT_GATE_COMPARISON.csv')
    feasible = pd.read_csv(scope / 'tables/SCOPE_FEASIBILITY.csv')
    configs = json.loads((root / 'configs/ALL_CLASSIFIERS.json').read_text())
    selected = []
    for key, config in configs.items():
        dep, seed, arm = key.split('/')
        spec = config['shared_classifier']
        slopes = np.asarray(spec['coefficients'][1:]) / np.asarray(spec['scale'])
        selected.append({'deployment': dep, 'seed': int(seed), 'arm': arm, 'ridge': spec['ridge'],
            'intercept_raw_features': spec['coefficients'][0] - float(np.dot(slopes, spec['mean'])),
            'cosine_slope': float(slopes[0]), 'log_power_slope': float(slopes[1]),
            'negative_deficit_coordinate_slope': float(slopes[2]),
            'learned_joint_logBC_slope': float(slopes[3]) if len(slopes) == 4 else 0.,
            'shared_cap': config['shared_cap'],
            'physical_likelihood_interpretation': False})
    n.write_csv(root / 'tables/SELECTED_COEFFICIENTS_PER_SEED.csv', selected)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = ['#247d86', '#b34651', '#76652c', '#48649e', '#8b497d', '#4d8152']
    diagnostics = ['full_population', 'random_draw', 'neighbor_population']
    for ax, dep, title in zip(axes, n.DEPS, ['O3', 'O4a']):
        for ai, (arm, color) in enumerate(zip(c.ARMS, colors)):
            offsets = (ai - 2.5) * .105
            values = []
            for di, diagnostic in enumerate(diagnostics):
                part = metrics[(metrics.deployment == dep) & (metrics.diagnostic == diagnostic)]
                old = part[part.arm == 'R55-FROZEN'].set_index('seed').logloss
                new = part[part.arm == arm].set_index('seed').logloss.loc[old.index]
                delta = (new - old).to_numpy()
                ax.scatter(di + offsets + np.array([-.014, 0., .014]), delta, s=12, color=color, alpha=.5)
                values.append(delta.mean())
            ax.plot(np.arange(3) + offsets, values, 'o', color=color, markersize=4, label=arm)
        ax.axhline(0., color='.25', lw=.8)
        ax.set_xticks(range(3), ['Full HT', 'Random null', 'Neighbor null'])
        ax.set_title(title, loc='left')
        ax.set_ylabel('Validation log-loss change vs R55')
        ax.grid(axis='y', alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc='upper center', frameon=False)
    fig.tight_layout(rect=(0., 0., 1., .90))
    (root / 'figures').mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'figures/R59_COORDINATE_COMPARISON.{ext}', dpi=180)
    plt.close(fig)
    reference_readout = json.loads((reference / 'audit/FINAL_GOAL_READOUT.json').read_text())
    archive = json.loads((P / 'packages/mcwf_shared_profile_boundary_55_20260909T225508Z_ARCHIVE_VERIFICATION.json').read_text())
    text = r'''# R59 坐标对照与 R60 可行性约束审计

状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。

**完整目标尚未达成。R59 的六组模拟校准均未通过共同门槛；没有将失败配置用于注入目录或真实候选重排。R60 是只读可行性界限审计，不是一个新评分方法或性能结果。**

## 1. 本轮具体做了什么

R59 复用 R56 的 1,053 个已完成物理 pair 拟合，比较三种坐标：

\[
D=P_{\mathrm{independent}}-P_{\mathrm{shared}},\qquad
f(D)\in\{\log(1+D),\sqrt D,D\}.
\]

这里的投影功率是当前快速波形比较的代理统计量，不是严格归一化的对数 likelihood、optimal SNR 或 Hanabi evidence。平方根不可以解释成多少 sigma。

三个基础输入为 embedding cosine、最弱像独立投影功率的对数、\(-f(D)\)。另设四维对照，把既有神经网络联合 Mc/eta/chi 概率分布的 log BC 加入同一个分类器。相关特征一起校准，不把两个独立 Bayes factor 相乘。

没有重新训练 encoder，没有恢复旧 encoder 的显式 Mc/q 差值评分，没有恢复 0.875 新旧总分混合。时间、天空、融合权重、数据范围和物理拟合结果全部冻结。

## 2. 数据与选型

- 使用现有开发数据：285 个真伴随 pair、768 个抽样 null pair，总计 1,053 对；不是新生成的独立 lens population。
- source/noise-parent 的 fit/tune 交集为零；每个事件的嵌入、学习参数分布和投影拟合不变。
- null 采用已验证的两阶段抽样包含概率和 source-pair 权重；类别各归一化为 0.5。
- fold0 拟合，fold1 的完整 HT 平衡 log-loss 选择每个 run/seed 的正则化参数；字典序/较强正则化的平局规则已冻结。
- 六种表示均必须在 O3、O4a 的完整 HT、普通 null 和邻近困难 null 上不劣于 R55，且至少一项严格改善。没有因结果改容差。
- 原 log 坐标四维方案及 R55 的复算与 R58 相符；全部 36 项单调性和低 D 饱和单元测试通过。
- D 的上下界一直按原物理单位处理，随后才变换坐标，避免把 log 坐标的阈值误用于 raw/sqrt。

这是反复使用 development/validation 的自适应研究，不是新的盲确认。bootstrap 也不消除这种选择偏差。

## 3. 完整校准结果

下表是三模型的平均校准 log-loss，越低越好；不是 R@10，也不是 AUPRC。

'''
    text += compare[['deployment', 'diagnostic', 'R55-FROZEN', *c.ARMS]].to_markdown(index=False, floatfmt='.6f') + '\n'
    text += r'''

六个方案都没有同时通过全部要求。原始 D 或平方根 D 未解决 O3 的 ordinary-null/hard-null 权衡，不能把“对数压缩”认定为此次失败的唯一原因，也不能因此直接增加惩罚。

`tables/CONDITIONAL_SOURCE_BOOTSTRAP_CI.csv` 提供 2,000 次 source-system 成对重采样的差值区间。两个 directed query 所属的源作为共同单元，null pair 使用两端源 multiplicity 的乘积；三个模型和所有方案共享抽样。区间条件于现有噪声和已拟合模型，不含模型选择不确定度，不用于改判 Gate。

## 4. R60：严格目标是否在数量上就无法实现

仅使用已归档的 NODUP/R55 真实表做只读约束检查。R55 当前只有每个运行期 10 个波形质量合格事件，即 45 个 eligible pair；其它 pair 的相对顺序被冻结。

为得到乐观的可行性上界，审计暂时允许这 45 对拥有任意分数，检查能否同时构造嵌套 Top10/20：PE 的 BC≥0.5 数量、中位 BC、Dmax≤3 数量、官方前端数量和 Hanabi 表重合数量均不下降，灾难性 Mc 对数不增加；O3 关键 pair 必须离开每个 seed 的 Top10。

使用二元线性约束求解。中位数以两个中间顺序统计量精确编码，包含相等值测试。未合格 pair 只能保持原顺序的前缀，因此最多保留前 20 对即可作等价缩减；其余 inactive pair 不可能进入 Top20。求解超时只能标记不确定，不能标记不可能。

'''
    cols = ['deployment', 'seed', 'eligible_pairs', 'solver_status', 'feasible_witness_verified', 'seconds', 'conclusion']
    text += feasible[cols].to_markdown(index=False) + '\n'
    text += r'''

六个 seed/run 都找到满足这些**单 seed 外部预算约束**的抽象集合，说明不能把失败归咎于候选数量上的必然矛盾。

但这没有满足共同波形函数、模拟注入泛化或跨 seed consensus 等约束，所以**不是实际可实现性能，更不是本实验的成功结果**。程序不保存抽象集合里的 pair 名单、排序或分数，不用它们训练、调参或重排。

## 5. 当前真正完成的目录结果

当前完整目录结果仍是 R55，不是 R59/R60：

- 关键 pair `GW191103_012549--GW191105_143521`：NODUP 共识 rank6 → R55 rank28；三个模型 rank 为 34/16/65。
- 公开 Mc BC 仍约 0.091665，未改变后验算法或结果。移动 rank 并不改变物理不相容本身。
- O3 共识 Top10 的 BCmc≥0.5 为 9→10，Top20 为 16→18；共识 Top10/20 的官方计数未下降。
- O4a 的共识 PE/官方预算未下降。
- 仍有 3 个 O3 单模型官方预算和 3 个 O4a 注入面板未通过既有门槛。因此完整目标仍未满足，不能升级。

本轮没有新的 R@1/R@10、真实 Top10 PE 或官方重合提升可报告；不能将 R55 的数字冒充成 R59/R60 的结果。

## 6. 文献依据及边界

| 文献 | 本轮用途 | 不能据此声称 |
|---|---|---|
| [Cutler & Flanagan 1994](https://arxiv.org/abs/gr-qc/9402014) | inspiral 包含质量/自旋信息且存在相关性，支持研究共同参数能否解释波形 | 本项目窗口、CNN 或某种 D 坐标最优 |
| [Lo & Magaña Hernandez 2023](https://arxiv.org/abs/2104.09339) | 共享源与独立源的物理假设比较 | 快速投影分数等于 Hanabi Bayes factor；本轮已运行 Hanabi |
| [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html) | 验证集反复选型的偏差边界 | 本次 bootstrap 已消除自适应选择偏差 |

官方表中的候选不是透镜真值。已公开候选的存在不能替代透镜假设的联合贝叶斯验证。本轮公开 PE/官方字段仅用于 R60 的描述性可行性检查，没有参与 R59 的任何模型选择。

## 7. 文件与复现

'''
    text += f'- R59 目录：`{root}`\n- R60 目录：`{scope}`\n- R55 完整结果：`{reference}`\n'
    text += f'- R55 已验证包：`{archive["archive"]}`\n- R55 包 SHA-256：`{archive["sha256"]}`\n'
    text += '- 本轮原始配置、逐 seed 结果、全部候选正则化、bootstrap、日志、单元测试和历史文件 hash 在对应目录。\n'
    text += '- 脚本快照与依赖在 `scripts/`；紧凑包不含原始 strain、PE HDF5、模型大文件或凭据。复跑依赖 manifest 中的服务器原始数据，不能宣称是完全自包含包。\n'
    text += f'- R59 校准耗时：{gate["seconds"]:.3f} s；R60 六次求解合计 {feasible.seconds.sum():.3f} s。这不是重新进行 BBH PE 的时间。\n'
    text += f'- 历史输入 hash 检查 {len(checks)} 项，变化数为 0。\n'
    text += '\n最终保持 `HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。\n'
    report = root / 'reports/R59_R60_COMPLETE_REPORT_CN.md'
    with report.open('x', encoding='utf-8') as stream:
        stream.write(text)
    with (scope / 'reports/R60_REPORT_INDEX_CN.md').open('x', encoding='utf-8') as stream:
        stream.write('# R60 只读约束审计\n\n完整联合报告：`' + str(report) + '`。\n\n没有生成可部署评分或新排名，未完成整体目标。\n')
    dependencies = []
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix != '.py' or not path.is_relative_to(P / 'scripts'):
            continue
        content = path.read_bytes()
        if re.search(rb'-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}', content):
            raise RuntimeError('Credential content cannot be packaged')
        destination = root / 'scripts/runtime_dependencies' / path.relative_to(P / 'scripts')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError('Dependency snapshot already exists')
        shutil.copy2(path, destination)
        dependencies.append({'path': str(path), 'sha256': n.sha(path), 'copy': str(destination.relative_to(root))})
    shutil.copy2(__file__, root / 'scripts/coordinate_scope_report.py')
    n.write_csv(root / 'manifest/RUNTIME_DEPENDENCIES.csv', dependencies)
    c.write_once(root / 'audit/FINAL_DIAGNOSTIC_AUDIT.json', {'UTC': n.utc(),
        'R59': gate, 'R60': scope_gate, 'latest_full_catalog': reference_readout,
        'goal_achieved': False, 'new_catalog_metrics_generated': False,
        'historical_input_hash_checks': len(checks), 'historical_input_changes': 0,
        'runtime_dependencies': len(dependencies), 'status': n.STATUS})
    for r in (root, scope):
        c.write_once(r / 'contracts/FINAL_EXPORT_COMPLETE.json', {'UTC': n.utc(), 'goal_achieved': False,
            'status': n.STATUS, 'report': str(report), 'no_historical_overwrite': True})
    print('COORDINATE_SCOPE_REPORT_COMPLETE', report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scope-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.scope_root, a.reference_root)
