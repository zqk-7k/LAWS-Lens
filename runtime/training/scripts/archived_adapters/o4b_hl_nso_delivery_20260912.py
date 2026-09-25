#!/usr/bin/env python3
"""Audited O4b result report and compact, credential-free delivery archives."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_evaluate_20260912 as ev

STATUS = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
PRIMARY = 'NEW-SCORE-ONLY-POSITIVE-CANDIDATE'


def text_table(frame):
    return frame.to_markdown(index=False, floatfmt='.4f')


def figures(root):
    out = root/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    data = pd.read_csv(root/'tables/retrieval_metrics_per_seed.csv')
    methods = ['waveform-short', 'waveform-OMC', 'waveform-new', 'time-only', 'sky-only',
               'SHORT-HL-CANDIDATE', 'OMC-HL-CANDIDATE', PRIMARY]
    labels = ['Short WF', 'OMC WF', 'New WF', 'Time', 'Sky', 'Short+T+S', 'OMC+T+S', 'New+T+S']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, key, title in zip(axes, ['macro_r_at_10', 'average_precision', 'false_at_recall_0p5'],
                              ['Companion R@10', 'Pair average precision', 'False pairs at 50% recall']):
        for k, name in enumerate(methods):
            v = data[data.method == name][key].to_numpy()
            ax.scatter(k+np.linspace(-.12, .12, len(v)), v, s=20, color='#256d85')
            ax.plot([k-.22, k+.22], [v.mean()]*2, color='#9c3f48', lw=2)
        ax.set_xticks(range(len(methods)), labels, rotation=50, ha='right')
        ax.set_ylabel(title); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out/'fig_o4b_retrieval_and_false_burden.pdf'); fig.savefig(out/'fig_o4b_retrieval_and_false_burden.png', dpi=180); plt.close(fig)
    if (root/'tables/real_PE_and_official_budget_summary.csv').exists():
        data = pd.read_csv(root/'tables/real_PE_and_official_budget_summary.csv')
        methods = ['SHORT-HL-CANDIDATE', 'OMC-HL-CANDIDATE', PRIMARY]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, key, title in zip(axes, ['BC_Mc_ge05', 'Dmax_le3', 'catastrophic_Mc'],
                                  ['Mc BC >= 0.5', 'Intrinsic Dmax <= 3', 'Severe Mc inconsistency']):
            for name, color in zip(methods, ['#777777', '#256d85', '#9c3f48']):
                f = data[data.method == name]
                ax.plot(f.budget, f[key], marker='o', label=name.replace('-HL-CANDIDATE', '').replace(PRIMARY, 'New score'), color=color)
            ax.set_xlabel('Candidate budget'); ax.set_ylabel(title); ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False, fontsize=7); fig.tight_layout()
        fig.savefig(out/'fig_o4b_real_PE_budget.pdf'); fig.savefig(out/'fig_o4b_real_PE_budget.png', dpi=180); plt.close(fig)


def report(root):
    if not (root/'contracts/INJECTION_EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('No evaluated held-out data; cannot manufacture a results report')
    import o4b_hl_nso_diagnostics_20260912 as diagnostics
    diagnostics.injection(root)
    diagnostics.real_figures(root)
    figures(root)
    summary = pd.read_csv(root/'tables/retrieval_metrics_summary.csv')
    keep = ['waveform-short', 'waveform-OMC', 'waveform-new', 'time-only', 'sky-only',
            'waveform-time-fixed-ratio', 'time-sky-fixed-ratio', 'SHORT-HL-CANDIDATE', 'OMC-HL-CANDIDATE',
            PRIMARY, 'NEW-SCORE-ONLY-NONNEGATIVE-CANDIDATE']
    summary = summary[summary.method.isin(keep)]
    display = pd.DataFrame({'method': summary.method})
    for old, new in [('macro_r_at_1', 'R@1'), ('macro_r_at_10', 'R@10'), ('average_precision', 'Pair AUPRC'),
                     ('false_at_recall_0p5', 'F50'), ('false_at_recall_0p9', 'F90')]:
        display[new] = [f'{m:.4f} +/- {sd:.4f}' for m, sd in zip(summary[old+'_mean'], summary[old+'_std'])]
    pe_done = (root/'contracts/PE_AUDIT_COMPLETE.json').exists()
    state = STATUS if pe_done else 'RUNNING_PE_INPUTS_AND_AUDIT_PENDING_NOT_FINAL'
    lines = [
        '# O4B-HL-NSO-01: BAYESTAR / NEW-SCORE-ONLY 完整实验说明', '',
        f'状态: `{state}`。本轮为独立 O4b 扩展，不修改 O3/O4a、ET-3、v9.3/v9.4、历史排名、论文或 Overleaf。', '',
        '## 1. 实验与目录范围',
        '- 服务器: connect.westd.seetacloud.com:32328；实验目录: `'+str(root)+'`。',
        '- 真实目录为冻结的 86 个 H1/L1 双通道可用事件、3,655 个无序 pair。75 个公开天空后验使用 HLV，11 个使用 HL；波形网络始终只输入 H1/L1。',
        '- 三个模型 seed: 2026091221/222/223；对应模拟目录 seed: 2026091231/232/233。每个 held-out 目录450事件、180透镜系统、360 directed queries、90背景事件、101025无序pair。',
        '- 不能把本轮450事件的 R@K 与历史经过筛选、候选数不同的 O3/O4a R@K 直接解释为运行期优劣。随机 R@10 为10/449。', '',
        '## 2. 数据生成和隔离',
        '- 真实 GWOSC O4b off-source H1/L1 噪声；192个256s块，128/32/32分配train/validation/test。原始4096s父文件也跨split隔离。已知事件中心正负128s排除，不保证噪声绝对没有未知弱信号。',
        '- IMRPhenomXPHM生成24s、4096Hz物理双探测器strain，包含各像放大率、Morse相位和对应到达时刻的天线响应。主源库600 smooth、600 subhalo、600非透镜源；SIS/PM是兼容目录名，不是本轮解析SIS/PM人口标签。',
        '- 所有主源按全局ID切为420/90/90，每一像对整体留在同一split。辅助网络用4096训练/512开发波形父源，主库和辅助库的GW-LMC环境ID隔离；辅助训练/开发环境也隔离。环境重复次数单独记录，不能将视图数或波形父源数冒充独立透镜环境数。',
        '- 当前继承BAYESTAR基线的逐像 target-SNR/proposal-ratio 缩放，不是共同距离单一缩放的response-derived检验。透镜像8--60；非透镜沿用经验SNR边缘，最大可能超过60。SNR来源包含公开网络定义差异，应按网络/SNR分层解释。',
        '- O4b曝光来自已有公开live窗口的重建，不是已验证的完整O4b duty-cycle日历。训练噪声在split内复用，pair并非独立。', '',
        '## 3. 波形分数的完整路径',
        '每个seed有五个内部学习组件，不是五个检索通道：短窗encoder+Mc/q回归、RAW-PHASE/RNC、ordered-Mc、2s+16s multirate Mc、条件eta/chi head。',
        '1. 2s窗口为合并前1.75s至后0.25s，2048Hz×2s=4096点。40--580Hz、off-source PSD白化、Tukey、抗混叠、稳健缩放，再按继承函数定向/标准化。',
        '2. 短窗cosine与预测Mc/q差异只用模拟validation的全局统计标准化并做KDE密度比；没有真实目录row-wise z-score。',
        '3. RNC预测Mc分布的BC经有限伴随系统尾概率形成FRT惩罚；ordered-Mc再加有界的尾惩罚和先验校正重合增量，得到Z_wf,OMC。旧的内部质量信息保留，不是NODUP-DIRECT。',
        '4. 从同一原始信号和同一噪声重建20--80Hz的16s窗口，合并前15.75s至后0.25s，256Hz×16s=4096点。不是拉长旧2s数组。短窗物理重放必须逐点相同。',
        '5. 2277个模板形成27×253短窗及27×253长窗响应；质量轴CNN预测p(Mc)，条件head预测p(eta,chi_eff|Mc)，联合分布边缘化后必须恢复同一p(Mc)。',
        '6. Z_wf,new = Z_wf,OMC + gamma*T + beta*I。T=min[log(p_tail/0.05),0]；I为单调校准的联合BC增量，截断到[-4,4]，无支持域正奖励回退0。T、I不是独立证据。',
        '这些网络分布不是完整贝叶斯PE；短窗回归也不是后验不确定度。各中间分数和系数均在Parquet中可追溯。', '',
        '## 4. 时间与天空',
        '- 时间: Z_time=log[p(delay|lensed,O4b exposure)/p(delay|null,same exposure)]。仅840个独立训练透镜系统拟合正分布；250000个null Monte Carlo draws不是250000个实测独立pair。lookup在validation/test/real固定不变，超界按继承端点规则并应审计。',
        '- 注入天空: 事件级conditional BAYESTAR，已知注入内禀模板、经验PSD和独立Gaussian matched-filter触发误差。不是旋转公开PE模板；也不是在完全相同非平稳注入strain上重新做完整BBH PE。必须保留这一条件化近似限制。',
        '- 真实天空: 公开PE原生MOC/FITS；正式512，统一NESTED概率质量表示。原生MOC栅格顺序明确，不把NESTED当RING。统一置换不改变像素点积；单位测试覆盖hot pixel和均匀图。',
        '- Z_sky=log[Npix sum(P_i P_j)]，BC_sky仅审计。验证集选择天空temperature后冻结。HL注入天空与多数HLV真实PE不是完全同质网络，不能宣称已消除这一域差异。',
        '- 256/512/1024只检查同一原生MOC的数值离散稳定性，不创造新的定位信息；完整分辨率审计见tables/sky_resolution_convergence.csv。', '',
        '## 5. 选参、融合与排名',
        '- 各seed独立、相同规则：只用模拟开发/validation拟合和选参；真实PE、官方重合不进入选择。优先F50、F90、AUPRC、R10、R1，完全并列时按基线距离和固定字典序。',
        '- 每阶段保留不改变分数的零修正对照；固定融合guard为R10下降不超过0.02、AP下降不超过0.005、F50/F90不超过1.1倍。失败保留，不回看test调参。',
        '- 权重0.05 simplex；主对照继承PATH的strict-positive，另完整输出nonnegative允许零权重版本。两者都只有一次三通道加和：S=wW*Z_wf,new+wT*Z_time+wS*Z_sky，alpha=1，没有0.125旧总分+0.875新总分。',
        '- 三通道使用冻结全局校准，pair分数对称，因此本轮unordered max与mean完全等价。共识按三个seed的平均rank、最坏rank、平均score、pair key确定。',
        '- test只在全部配置和模型hash冻结后打开；不存在反复根据真实候选修改模型来达到好看结果。', '',
        '## 6. Held-out 注入结果', text_table(display), '',
        'mean +/- SD跨三个seed；每个seed的10000次system-level bootstrap 95% CI另见逐seed CSV。两幅像的queries一起抽样。pair bootstrap按source multiplicity加权；不能消除共享noise或人口近似。', '',
        '## 7. 真实候选、PE与官方阶段',
    ]
    if pe_done:
        budgets = pd.read_csv(root/'tables/real_PE_and_official_budget_summary.csv')
        lines += [text_table(budgets[budgets.method.isin(['SHORT-HL-CANDIDATE', 'OMC-HL-CANDIDATE', PRIMARY])]),
                  '完整候选含总分、wf/time/sky贡献、BC_Mc/q/chi_eff/表观距离、Dmax，见results/PE_audit/<method>/。',
                  'Dmax只包括Mc、q、chi_eff；表观距离不作为共同源必须相等的条件。BC和D是描述性边际审计，不是Hanabi或共享源Bayes factor。']
    else:
        lines += ['公开PE文件仍在下载或尚未完成原始哈希核验。本报告的PE结果暂缺，不填0，不声称完整交付。真实冻结排名已在results/real/consensus/保存。']
    lines += [
        '当前输入中没有经过核验的O4b官方lensing pair PO/ML/Phazap FPP或公开Hanabi配对表，因此对应数量标记不可用。不能将事件detection FAR当成pair FPP，也不能借用O4a官方表。未运行Hanabi；无已确认透镜标签。', '',
        '## 8. 文献与适用边界',
        '| 文献/软件 | 与本实验的关系 | 不能据此声称 |',
        '|---|---|---|',
        '| [Cutler & Flanagan 1994](https://arxiv.org/abs/gr-qc/9402014) | inspiral质量/自旋信息和相关性的物理动机 | 证明本CNN或16s、20--80Hz最优 |',
        '| [Singer & Price 2016](https://arxiv.org/abs/1508.03634) | BAYESTAR快速相干天空定位 | 本轮完成所有注入的非高斯strain full PE |',
        '| [ligo.skymap injection workflow](https://lscsoft.docs.ligo.org/ligo.skymap/quickstart/bayestar-injections.html) | conditional trigger simulation与快速定位实现 | 与真实事件所有搜索测量条件完全一致 |',
        '| [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) | 神经预测temperature校准 | 预测分布就是GW物理PE后验 |',
        '| [Lo & Magana Hernandez 2023](https://arxiv.org/abs/2104.09339) | 检索与联合透镜贝叶斯确认的边界 | 已完成Hanabi或候选被确认 |',
        '| [O4b public catalog](https://ligo.org/detections/o4b-catalog/) | 事件数据与公开PE来源 | event FAR等于lensing pair FPP |', '',
        '## 9. 交付与复现',
        '- contracts/: 初始合同、数据隔离、模型训练、时间lookup、sky温度和最终选参冻结。',
        '- models/、rankncontrast_component_v2/、ordered_mass_predictor/: 新O4b选中checkpoint；旧停止的RNC v1保留但不用于结果。',
        '- tables/: 逐seed与汇总Recall、pair指标、bootstrap、权重、PE预算和分辨率审计。',
        '- results/: test所有pair分数、directed query ranks、真实全部3655pair及Top10/20/50、PE完整联表。',
        '- event_maps/: 每个注入事件的原生BAYESTAR MOC仍在服务器；紧凑包不包含大型strain/PE或dense sky缓存，另提供native-maps包。',
        '- scripts/、logs/、manifests/: 完整可复现代码、失败与运行日志、输入与交付文件SHA-256。复现需要manifest列出的原始公共数据和运行环境。',
        '- 完成后保持HOLD，禁止自动写回论文或替代已有O3/O4a结果。',
    ]
    path = root/'reports/O4B_HL_NSO_01_COMPLETE_METHOD_AND_RESULTS_CN.md'
    path.write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    s.write(root/'contracts/REPORT_STATE.json', {'utc': s.now(), 'PE_complete': pe_done, 'state': state,
                                               'report': str(path), 'report_sha256': s.sha(path)})


def protected_check(root):
    expected = json.loads((root/'manifests/PROTECTED_UPSTREAM.json').read_text())
    rows = []
    for row in expected:
        p = Path(row['path']); digest = s.sha(p)
        rows.append({'path': str(p), 'expected': row['sha256'], 'actual': digest, 'unchanged': digest == row['sha256']})
    ev.csv(root/'manifests/HISTORICAL_HASH_FINAL_CHECK.csv', pd.DataFrame(rows))
    if not all(r['unchanged'] for r in rows):
        raise RuntimeError('Historical input changed; investigate rather than overwrite or claim unchanged')


def archive(root, path, files):
    if path.exists():
        raise RuntimeError('Delivery archive already exists; do not overwrite')
    s.disk_guard(root)
    pattern = re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|sshpass\s+-p\s+[\x27\x22]|Bearer\s+[A-Za-z0-9_-]{20,}')
    for p in files:
        if p.suffix in ('.py', '.json', '.md', '.txt', '.csv') and p.stat().st_size < 10*1024**2:
            if pattern.search(p.read_text(errors='replace')):
                raise RuntimeError('Potential credential in proposed delivery: '+str(p.relative_to(root)))
    manifest = root/'manifests'/f'{path.stem}_FILES.csv'
    ev.csv(manifest, pd.DataFrame([{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': s.sha(p)} for p in files]))
    files = [*files, manifest]
    with tarfile.open(path, 'w:gz', compresslevel=3) as tar:
        for p in sorted(set(files)):
            tar.add(p, arcname=str(Path(root.name)/p.relative_to(root)), recursive=False)
    expected = pd.read_csv(manifest).set_index('path').sha256.to_dict()
    expected[str(manifest.relative_to(root))] = s.sha(manifest)
    with tarfile.open(path, 'r:gz') as tar:
        names = tar.getnames()
        if len(names) != len(set(names)) or len(names) != len(set(files)):
            raise RuntimeError('Archive file membership mismatch')
        for member in tar:
            relative = str(Path(member.name).relative_to(root.name))
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError('Unexpected non-file archive member')
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024**2), b''):
                digest.update(block)
            if digest.hexdigest() != expected[relative]:
                raise RuntimeError('Internal archive checksum mismatch:'+relative)
    digest = s.sha(path)
    path.with_suffix(path.suffix+'.sha256').write_text(digest+'  '+path.name+'\n')
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size, 'members': len(names)}


def package(root):
    for name in ('PE_AUDIT_COMPLETE.json', 'SKY_CONVERGENCE_AUDIT_COMPLETE.json', 'REAL_RANKINGS_COMPLETE.json',
                 'INJECTION_EVALUATION_COMPLETE.json', 'FINAL_SCORE_CONFIG_FREEZE.json'):
        if not (root/'contracts'/name).exists():
            raise RuntimeError('Required final audit missing:'+name)
    protected_check(root); report(root)
    folder = root/'package'; folder.mkdir(exist_ok=True)
    files = []
    for name in ('contracts', 'tables', 'reports', 'figures', 'scripts', 'logs', 'results', 'calibration', 'manifests'):
        for p in (root/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink():
                if p.suffix == '.npy' or 'null_delay_montecarlo' in p.name:
                    continue
                files.append(p)
    for glob in ('models/**/selected.pt', 'models/**/validation_selected_model.pt',
                 'ordered_mass_predictor/**/validation_selected_model.pt',
                 'rankncontrast_component_v2/models/**/validation_selected_model.pt'):
        files += list(root.glob(glob))
    for p in (root/'event_maps').rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.parquet'):
            files.append(p)
    files = sorted(set(files))
    core = archive(root, folder/f'{root.name}_deliverables.tar.gz', files)
    maps = [p for p in (root/'event_maps').rglob('*.fits.gz') if p.is_file()]
    map_archive = archive(root, folder/f'{root.name}_native_BAYESTAR_maps.tar.gz', maps)
    s.write(root/'DELIVERY_VERIFICATION.json', {'utc': s.now(), 'state': STATUS, 'core': core, 'native_maps': map_archive,
                                             'historical_hashes_unchanged': True, 'no_Hanabi_or_paper_adoption': True,
                                             'PE_official_unknown_values_not_filled': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('report', 'package'), required=True); a = p.parse_args()
    globals()[a.stage](a.root)
