"""Audit, summarize and package UAB without altering historical products."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd

import uab_completion as c
from uab_evaluate import csv

FINAL = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'


def table(frame):
    def cell(v):
        if pd.isna(v): return 'NA'
        if isinstance(v, (float, np.floating)): return f'{v:.6g}'
        return str(v).replace('|', '/')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


def validate():
    c.gate('test'); c.U.verify(c.ROOT)
    if not c.check_complete(c.OUT/'contracts/REAL_INPUT_MANIFEST_FREEZE.json'):
        raise RuntimeError('Real external-input manifest missing')
    files = pd.read_csv(c.ROOT/'manifests/PROTECTED_INPUTS.csv')
    rows = []
    for row in files.itertuples():
        actual = c.U.sha(row.path)
        rows.append(dict(path=row.path, before=row.sha256, after=actual, unchanged=actual == row.sha256))
    if not all(r['unchanged'] for r in rows):
        csv(c.OUT/'tables/HISTORICAL_HASH_AUDIT.csv', pd.DataFrame(rows))
        raise RuntimeError('Historical input hash changed')
    csv(c.OUT/'tables/HISTORICAL_HASH_AUDIT.csv', pd.DataFrame(rows))
    for run in c.U.RUNS:
        if not c.check_complete(c.OUT/'real_PE'/run/'COMPLETE.json'):
            raise RuntimeError('Real PE/official audit missing')
        for arm in c.U.ARMS:
            for split in ('validation', 'test'):
                p = c.deployment(run, arm)
                if not c.check_complete(p/f'sky_pair_scores/{split}/REFERENCE1024_COMPLETE.json'):
                    raise RuntimeError('Resolution audit missing')
                for seed in c.U.SEEDS:
                    for name in ('COMPLETE.json', 'CATALOG190_COMPLETE.json'):
                        if not c.check_complete(p/f'evaluation/{split}/seed_{seed}/{name}'):
                            raise RuntimeError('Evaluation incomplete')
    c.write(c.OUT/'contracts/COMPLETENESS_AUDIT.json', dict(state='PASS', deployments=6,
        trained_models=90, model_seeds=3, full_and_size_control=True, both_arms_all_runs=True,
        historical_files_checked=len(rows), historical_changed=0, final_state=FINAL))


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = c.OUT/'figures'; dest.mkdir(exist_ok=True)
    d = pd.read_csv(c.OUT/'tables/retrieval_summary_190.csv')
    d = d[(d.split == 'test') & (d.method == 'three-channel')]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, metric, label in zip(axs, ['macro_r_at_10', 'average_precision', 'false_at_recall_0p9'],
                                ['Recall@10 (190 events)', 'Pair average precision', 'False pairs at 90% recall']):
        for offset, arm, color in [(-.18, 'A_NEUTRAL', '#277da1'), (.18, 'B_CUE', '#d1495b')]:
            z = d[d.arm == arm].set_index('run').loc[list(c.U.RUNS)]
            ax.bar(np.arange(3)+offset, z[metric+'_mean'], .34, yerr=z[metric+'_std'], capsize=3,
                   label=arm, color=color)
        ax.set_xticks(range(3), c.U.RUNS); ax.set_ylabel(label); ax.spines[['top', 'right']].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(dest/'fig_unified_arm_comparison.pdf'); fig.savefig(dest/'fig_unified_arm_comparison.png', dpi=180); plt.close(fig)
    b = pd.read_csv(c.OUT/'tables/real_PE_official_budget_summary.csv')
    b = b[(b.method == 'three-channel') & (b.budget == 10)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, field, label in zip(axs, ['Mc_BC_ge05', 'Dmax_le3', 'official_frontend_count'],
                               ['Top10 Mc BC >= 0.5', 'Top10 Dmax <= 3', 'Top10 official 1% front-end']):
        for offset, arm, color in [(-.18, 'A_NEUTRAL', '#277da1'), (.18, 'B_CUE', '#d1495b')]:
            z = b[b.arm == arm].set_index('run').loc[list(c.U.RUNS)]
            ax.bar(np.arange(3)+offset, z[field], .34, color=color, label=arm)
        ax.set_xticks(range(3), c.U.RUNS); ax.set_ylim(0, 11); ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
    axs[2].text(2, .5, 'O4b: NA', ha='center', fontsize=9)
    fig.savefig(dest/'fig_real_PE_official_audit.pdf'); fig.savefig(dest/'fig_real_PE_official_audit.png', dpi=180); plt.close(fig)


def report():
    budgets = pd.concat([pd.read_csv(c.OUT/'real_PE'/run/'budget_summary.csv') for run in c.U.RUNS], ignore_index=True)
    csv(c.OUT/'tables/real_PE_official_budget_summary.csv', budgets)
    ranks = []
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            source = c.OUT/'real_PE'/run/arm/'three-channel'
            ranks.append(pd.read_csv(source/'top50_with_PE_official.csv'))
    csv(c.OUT/'tables/real_top50_both_arms_all_runs.csv', pd.concat(ranks, ignore_index=True))
    weights = pd.read_csv(c.OUT/'tables/selected_weights.csv')
    summary = pd.read_csv(c.OUT/'tables/retrieval_summary_190.csv')
    full = pd.read_csv(c.OUT/'tables/retrieval_summary_450.csv')
    snr = pd.read_csv(c.OUT/'tables/SNR_only_diagnostic.csv')
    population = pd.read_csv(c.OUT/'tables/population_calendar_noise_audit.csv')
    resolution = pd.read_csv(c.OUT/'tables/sky_resolution_summary.csv')
    sky_quality = pd.read_csv(c.OUT/'tables/sky_truth_coverage_and_runtime.csv')
    tasks = []
    for path in (c.OUT/'contracts/tasks').glob('*.json'):
        if path.name.endswith('_RUNNING.json'): continue
        d = json.loads(path.read_text())
        tasks.append(dict(task=path.stem, exit_code=d.get('exit_code'), seconds=d.get('elapsed_seconds'), log=d.get('log')))
    csv(c.OUT/'tables/COMPLETION_TASK_LEDGER.csv', pd.DataFrame(tasks))
    display = []
    for label, data in [('190', summary), ('450', full)]:
        for row in data[(data.split == 'test') & data.method.isin(['waveform-new', 'time-only', 'sky-only', 'three-channel'])].to_dict('records'):
            out = dict(run=row['run'], arm=row['arm'], events=label, method=row['method'])
            for metric in ('macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'):
                out[metric] = f"{row[metric+'_mean']:.4f} +/- {row[metric+'_std']:.4f}"
            display.append(out)
    presentation = pd.DataFrame(display)
    csv(c.OUT/'tables/PRIMARY_DISPLAY_TABLE.csv', presentation)
    chosen_budget = budgets[(budgets.method == 'three-channel') & budgets.budget.isin([10, 20, 50])]
    report_text = f'''# 三运行期统一 SNR 对照实验：完整报告

实验代号：GWLR-UAB-01。最终状态：`{FINAL}`。

本轮为独立探索实验，不覆盖任何历史版本、论文或候选表。A_NEUTRAL 移除人为 SNR 类别配对；B_CUE 故意保留该线索，其他设置一致。B 不得作为无偏主方法采纳。真实 PE 和官方候选仅用于冻结后的审计，未参与本轮选参。

## 1. 本轮完成什么

三个运行期分别训练相同架构和规则的模型，每个运行期两组 SNR 方案、三个模型种子，总计 90 个模型组件。短窗、RNC、ordered-Mc、16 秒低频 multirate 和条件 eta/chi 分支均使用本轮训练结果。不是旧模型在新数据上的一次简单重排。

正式推理使用每个运行期自己的噪声、PSD 和曝光日历。O3 限于 O3a/O3b，不再沿用累计 O1--O3 日历。共同源总体、质量范围、类别数量、数据划分和评价规模采用同一规则。A/B 共用源、噪声安排与 SNR 多重集合，仅 SNR 的事件分配不同。

主目录按全局 GW-LMC 环境 ID 分组：每类 600 个源，训练/验证/测试为 420/90/90；两类双像及一类孤立源合计 1,800 个不同父源。表中兼容字段 `SIS` 指 GW-LMC 无子晕环境，`PM` 指有子晕环境，不是解析 SIS 或点质量注入。时延、放大率和 Morse 类型来自所选成像记录，但源质量、自旋、方向和目标 SNR 由本项目的受控方案重新抽取，不声称完整复现 GW-LMC 总体。

辅助分布网络使用 4,096 个训练波形父源和 512 个验证波形父源；它们可能在各自 split 内复用透镜环境，不能把这些波形父源数量当成同等数量独立透镜环境。全局环境 ID 在 train/validation/test 之间无重叠。每运行期 32 个真实 HL 噪声父块按 20/6/6 分配，训练的多视图和三个模型 seed 不增加独立噪声块数量。

共同目标 SNR 档位为 8--10、10--12、12--20、20--40。B 保留指定的强弱档位配对；A 在相同事件集合内置换 B 的 SNR 多重集合。详细事件/源/噪声清单与随机种子均在配置和 manifests 中，不能仅用一条 Recall 概括数据生成差异。

## 2. 分数流程

1. H1/L1 的 2 秒窗口进入 attention-Inception 编码器，产生 embedding 与 Mc/q 均值预测。
2. 16 秒、20--80 Hz 的低频支路转成模板匹配特征，与短窗特征一起预测质量分布。
3. 条件网络给出 eta/chi_eff 分布，与 Mc 分布组成联合参数预测；它不是公开 PE，也不是经过证明的完整贝叶斯后验。
4. 短窗 cosine 和 Mc/q 差异在 validation 上校准。RNC、ordered 和联合分布的相容性分别以冻结的尾部惩罚、有限增量进入同一波形通道。无支持域的正奖励被禁止。
5. 时间为训练源系统与本运行期曝光背景的一维对数密度比。天空为 `log(Npix * sum(P_i * P_j))`，正式 Nside=512，显式 NESTED 转 RING、float64 累积。
6. 三通道只加权一次，没有旧总分与新总分的外层混合。每组/运行期/seed 只在 validation 的 0.05 simplex 上选权，规则相同但数值可不同。

同一源环境的校准重复样本合计权重为 1；子集划分之后重新核对该归一化。联合校准的 fit/audit 波形父源和噪声折不同，但可能复用同一透镜环境，故不声称独立总体覆盖率验证。所有组配置冻结完成后才生成本轮测试波形。

## 3. 注入结果

完整目录为 450 事件：180 个双像系统和 90 个孤立事件，101,025 个无序对。规模控制为预先固定的 500 个 190 事件子目录，各含 70 个真对和 50 个孤立事件，17,955 对。三个运行期、A/B 使用同一源抽样索引。不是按结果挑选子目录。

下表为三个模型的均值和样本标准差。190 事件口径先在每个模型内平均 500 个子目录，再计算模型间 SD；500 个子目录不独立。

{table(presentation)}

R@10 是每个伴随查询在候选库中找回伙伴的比例，不是真实 Top10 的透镜比例。Pair AUPRC 使用 average precision。F50/F90 包括达到阈值的全部同分 pair；检索和 Top-B 使用同分随机排序的期望值。主结果、单通道和中间波形阶段全部保存在逐 seed CSV。

## 4. SNR 人工线索审计

这个诊断只读目标 SNR 档位，不读波形，也不用于评分器训练或调参。它衡量类别线索是否存在，不能证明网络使用了该线索或量化历史 Recall 虚高多少。

A/B 差值包含 SNR 分配改变后重新训练、校准和选权的总体影响，不能解释成旧结果中可直接扣除的“泄露比例”。跨运行期统一了源总体和目标 SNR，也就控制掉了部分仪器升级带来的可探测性优势；因此新版 Recall 仍不要求按 O3、O4a、O4b 单调增加。实测噪声形态、PSD、曝光日历和竞争事件时间密度仍会影响结果。

{table(snr[snr.split.eq('test')][['run','arm','roc_auc','macro_r_at_10']])}

## 5. 真实 PE 与官方阶段

真实范围冻结为 O3 62 事件/1,891 对、O4a 74 事件/2,701 对、O4b 86 事件/3,655 对。A/B 均对完整相同范围排序，再连接 PE 和官方表，没有按 PE 淘汰后重排。

三个运行期统一使用公开单事件 posterior 的 Gaussian KDE Bhattacharyya 重合与样本标准差 D；Mc 为探测器系。Dmax 仅取 Mc、q、chi_eff，不包含表观距离。公开 posterior group 按事件契约固定。

{table(chosen_budget[['run','arm','budget','Mc_BC_ge05','Dmax_le3','median_Mc_BC','catastrophic_Mc','official_frontend_count','public_Hanabi_count']])}

O3 的官方前端为 PO/ML 任一 FPP<1%；O4a 为 PO/Phazap 任一<1%。公开 Hanabi 表重合不是透镜真值，也不代表本轮运行 Hanabi。O4b 冻结输入没有可核验逐对透镜表，官方数量记为 NA，而非零。

各组 Top10/20/50/100 和全部 pair 机器表含分数、wave/time/sky 贡献、四项 PE BC、Dmax、官方数值与公开阶段。汇总入口为 `tables/real_top50_both_arms_all_runs.csv`，完整表在 `real_PE/<run>/<arm>/`。

## 6. 日历、噪声和总体

{table(population[population.split.eq('test')])}

每个运行期仅有 6 个独立测试噪声父块；100,000 量级的 pair 不能当作同等数量独立实验。源系统与噪声父块分别进行 1,000 次聚类敏感性重采样，报告条件区间，不声称已处理所有层次的总体不确定性。A/B 使用相同重采样，差值区间在 `paired_arm_difference_intervals.csv`。

## 7. 天空数值审计

修复 SI 质量单位后的 BAYESTAR 入口用于三个运行期。注入采用事件级 native MOC，不旋转旧模板，不把旧 Nside64 注入图上采样。公共 HDF5 的 `[b'True']` 严格解析并转换 ordering，单元测试单独交付。

256/512/1024 使用相同 validation 温度；本轮额外计算全部 pair 的 1024 参考值，覆盖真对、假对正尾、哈希随机样本与符号变化对象。真实全部 strict pair 也审计。审计值不参与重新选权，不因测试表现临时更改分辨率。

天空真值覆盖按每个源系统总权重为 1 计算。下表同时保留原始温度和冻结温度下的覆盖、原始 A90 与定位回退记录；测试覆盖仅作诊断，不再校准温度。并行任务 wall time 不应直接相加作为总运行时间。

{table(sky_quality)}

{table(resolution[resolution.population.isin(['true_companion','null_high_tail_1pct','all_real_pairs'])])}

## 8. 仍然存在的限制

- 注入仍按每幅像独立设置目标 SNR，是受控检索实验，不是完整 response-derived 透镜总体。A 移除的是指定的人为档位线索，不等于消除所有选择效应。
- BAYESTAR 输入包含已知内禀参数和模拟高斯触发误差，不是从同一非高斯带噪应变恢复完整触发量，也不是完整 BBH PE。
- 波形与注入定位使用 H1/L1；真实公共天空图可能包含 Virgo 等网络信息。这一历史观测产品差异明确保留，未宣称真实与注入 PE 管线完全等价。
- 三个模型 seed 共用本轮固定源/噪声数据，模型间 SD 主要是模型初始化差异，不是三套独立宇宙/噪声总体。
- 官方重合、PE 相容性和网络分数都不是透镜确认。无新的 Hanabi、联合 evidence 或目录级假阳性声明。
- 历史开发已看过相关数据和真实目录，本轮不能追认成全项目首次盲测。测试只在本轮最终配置冻结后计算，且结果没有反馈选参。

## 9. 权重与交付

{table(weights[weights['mode'].eq('POSITIVE')])}

全部正负结果、零权重补充、校准平台、模型诊断、分辨率差异与历史 hash 审计均保留。详细输入在 `contracts/` 和 `manifests/`；脚本在 `scripts/`；完整模型留在原独立实验目录 `arms/`，紧凑包附模型与可复现脚本；native MOC 单独打包。原始 strain、大型可重建缓存和任何登录凭据不进入交付包。

服务器：connect.westd.seetacloud.com:32328。

结果：`{c.OUT}`。

最终只提交作者审核，不自动升级版本，不修改论文或 Overleaf。
'''
    dest = c.OUT/'reports/GWLR_UAB_01_COMPLETE_METHOD_AND_RESULTS_CN.md'
    dest.write_text(report_text, encoding='utf-8')
    (c.OUT/'README_CN.md').write_text('# GWLR-UAB-01\n\n'+FINAL+'\n\n完整报告：reports/GWLR_UAB_01_COMPLETE_METHOD_AND_RESULTS_CN.md\n\n450/190规模结果：tables/retrieval_summary_450.csv、tables/retrieval_summary_190.csv\n\n真实PE和官方阶段：tables/real_PE_official_budget_summary.csv；全部Top50：tables/real_top50_both_arms_all_runs.csv\n', encoding='utf-8')
    plots()


def package():
    target = c.ROOT/'package'; target.mkdir(exist_ok=True)
    files = {}
    def add(path, name):
        if path.is_file(): files[name] = path
    for folder in ('contracts','scripts','tables','reports','figures'):
        for path in (c.OUT/folder).rglob('*'):
            if path.is_file() and not path.name.endswith('.lock') and '__pycache__' not in str(path):
                add(path, 'completion/'+str(path.relative_to(c.OUT)))
    add(c.OUT/'README_CN.md', 'README_CN.md')
    for folder in ('real_PE','real_sky'):
        for path in (c.OUT/folder).rglob('*'):
            add(path, 'completion/'+str(path.relative_to(c.OUT)))
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            root = c.deployment(run, arm)
            for folder in ('calibration', 'evaluation', 'sky_pair_scores', 'real_ranking'):
                for path in (root/folder).rglob('*'):
                    add(path, 'completion/'+str(path.relative_to(c.OUT)))
    for folder in ('contracts', 'manifests', 'reports', 'scripts', 'plans'):
        for path in (c.ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix not in ('.npy', '.pyc') and not path.name.endswith('.lock'):
                add(path, 'training/'+str(path.relative_to(c.ROOT)))
    models = pd.read_csv(c.OUT/'contracts/TRAINED_MODEL_MANIFEST.csv')
    for row in models.itertuples():
        path = Path(row.path)
        add(path, 'training/'+str(path.relative_to(c.ROOT)))
    # Complete logs are small compared with posterior tensors and preserve failures.
    for path in (c.OUT/'logs').glob('*.log'):
        add(path, 'completion/logs/'+path.name)
    forbidden = [b'-----BEGIN RSA PRIVATE KEY-----', b'-----BEGIN OPENSSH PRIVATE KEY-----', b'sshpass -p ']
    for name, path in files.items():
        if path.suffix in ('.json','.csv','.md','.py','.log','.txt') and path.stat().st_size < 30*2**20:
            content = path.read_bytes()
            if any(token in content for token in forbidden):
                raise RuntimeError('Credential-like content in package: '+name)
    records = [dict(path=name, sha256=c.U.sha(path), bytes=path.stat().st_size) for name, path in sorted(files.items())]
    manifest = c.OUT/'manifests/DELIVERABLE_SHA256.csv'; csv(manifest, pd.DataFrame(records))
    files['manifest/DELIVERABLE_SHA256.csv'] = manifest
    archive = target/'GWLR_UAB_01_20260918_deliverables.tar.gz'
    if archive.exists():
        raise RuntimeError('Do not overwrite an existing delivery archive')
    c.U.guard(c.ROOT)
    with tarfile.open(archive, 'w:gz', compresslevel=5, dereference=True) as tar:
        for name, path in sorted(files.items()): tar.add(path, arcname=name, recursive=False)
    verified = 0
    expected = {row['path']: row['sha256'] for row in records}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected: continue
            h = hashlib.sha256(); stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(2**20), b''): h.update(block)
            if h.hexdigest() != expected[member.name]: raise RuntimeError('Archive internal hash failure')
            verified += 1
    if verified != len(expected): raise RuntimeError('Archive member count mismatch')
    sha = c.U.sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha+'  '+archive.name+'\n')
    maps = target/'GWLR_UAB_01_20260918_native_BAYESTAR_maps.tar.gz'
    if maps.exists(): raise RuntimeError('Do not overwrite native-map archive')
    mapfiles = [path for path in (c.OUT/'maps').rglob('*') if path.is_file() and '.tmp.' not in path.name]
    map_records = [dict(path=str(path.relative_to(c.OUT)), sha256=c.U.sha(path), bytes=path.stat().st_size) for path in mapfiles]
    c.write(c.OUT/'manifests/NATIVE_MAP_SHA256.json', map_records)
    with tarfile.open(maps, 'w:gz', compresslevel=2) as tar:
        for path in mapfiles: tar.add(path, arcname=str(path.relative_to(c.OUT)), recursive=False)
        tar.add(c.OUT/'manifests/NATIVE_MAP_SHA256.json', arcname='manifest/NATIVE_MAP_SHA256.json')
    mapsha = c.U.sha(maps)
    maps.with_suffix(maps.suffix+'.sha256').write_text(mapsha+'  '+maps.name+'\n')
    map_expected = {row['path']: row['sha256'] for row in map_records}; map_verified = 0
    with tarfile.open(maps, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in map_expected: continue
            h = hashlib.sha256(); stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(2**20), b''): h.update(block)
            if h.hexdigest() != map_expected[member.name]: raise RuntimeError('Native map archive hash failure')
            map_verified += 1
    if map_verified != len(map_expected): raise RuntimeError('Native map member count mismatch')
    return dict(deliverables=str(archive), deliverables_sha256=sha, internal_files_verified=verified,
                native_maps=str(maps), native_maps_sha256=mapsha, map_files_verified=map_verified)


def deliver():
    validate(); report()
    archived = package()
    record = dict(state=FINAL, complete_results=True, utc=c.U.now(), results=str(c.OUT),
        report=str(c.OUT/'reports/GWLR_UAB_01_COMPLETE_METHOD_AND_RESULTS_CN.md'),
        historical_results_overwritten=False, paper_changed=False, automatically_adopted=False,
        A_NEUTRAL_complete=True, B_CUE_complete=True, all_three_runs_complete=True, **archived)
    c.write(c.OUT/'contracts/FINAL_DELIVERY.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['deliver'], required=True)
    a = p.parse_args(); c.initialize(a.root, a.out); deliver()
