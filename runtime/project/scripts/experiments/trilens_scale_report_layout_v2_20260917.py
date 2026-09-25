"""Summarize completed or failed scaling runs without replacing source outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main(root):
    final=json.loads((root/'contracts/FINAL_STATUS.json').read_text())
    manifest=root/'manifest/DELIVERY_SHA256SUMS.txt'
    package=root.parent/(root.name+'_deliverables.tar.gz')
    if manifest.exists() or package.exists():raise RuntimeError('No report/package overwrite')
    timings=root/'tables/scaling_timings.csv'
    data=pd.read_csv(timings) if timings.exists() else pd.DataFrame()
    reference=root/'reference_runs/r2_scaling_timings.csv'
    if reference.exists():
        prior=pd.read_csv(reference)
        prior=prior[prior.method.eq('Phazap') & prior.n_events.eq(5)].copy()
        prior['timing_source']='r2_valid_5_event_Phazap_measurement'
        data['timing_source']='this_run_TriLens_completion'
        data=pd.concat([data,prior],ignore_index=True)
        data.to_csv(root/'tables/AGGREGATED_TIMING_TABLE.csv',index=False,encoding='utf-8-sig')
    passed=final['compute_complete'] and final['protected_inputs_unchanged']
    complete_count=0 if data.empty else len(data[data.boundary.eq('PRODUCTS_READY_TOTAL')])
    passed=passed and complete_count==8 and final['checks_total']==final['checks_passed']
    contract=json.loads((root/'contracts/CONTRACT.json').read_text())
    components=root/'tables/timings.csv'
    component_summary=None
    if components.exists():
        comp=pd.read_csv(components)
        comp=comp[comp.method.eq('TriLens')]
        component_summary=comp.groupby(['n_events','stage']).wall_seconds.sum().reset_index()
        component_summary.to_csv(root/'tables/TRILENS_PREPARATION_COMPONENTS.csv',index=False,encoding='utf-8-sig')
    warm_summary=pd.DataFrame()
    if not data.empty:
        warm=data[data.boundary.eq('EVENT_CACHE_PAIR_RANK')]
        warm_summary=warm.groupby(['method','n_events','n_pairs']).wall_s.agg(['median','min','max','size']).reset_index()
        warm_summary.to_csv(root/'tables/WARM_PAIR_SUMMARY.csv',index=False,encoding='utf-8-sig')
        first=data[data.boundary.eq('PRODUCTS_READY_TOTAL')].copy()
        first.to_csv(root/'tables/FIRST_PASS_SUMMARY.csv',index=False,encoding='utf-8-sig')
        fig,axes=plt.subplots(1,2,figsize=(11,4.3))
        colors={'TriLens':'#267b8c','Phazap':'#ab4967'}
        for name,color in colors.items():
            f=first[first.method.eq(name)].sort_values('n_pairs')
            axes[0].plot(f.n_pairs,f.wall_s,'o-',color=color,label=name)
            w=warm_summary[warm_summary.method.eq(name)].sort_values('n_pairs')
            axes[1].errorbar(w.n_pairs,w['median'],
                yerr=np.array([w['median']-w['min'],w['max']-w['median']]),fmt='o-',capsize=3,color=color,label=name)
        for ax in axes:
            ax.set(xscale='log',yscale='log',xlabel='Distinct pairs per catalog',ylabel='Measured wall time [s]')
            ax.set_xticks([10,105,1035,1891],['10','105','1,035','1,891'])
            ax.legend(frameon=False,loc='lower right')
            ax.margins(y=.2)
        axes[0].set_title('Event preparation + all pairs (one pass)',fontsize=11)
        axes[1].set_title('Prepared events: recompute all pair scores',fontsize=11)
        if final.get('trilens_all_scales_complete'):
            fig.text(.5,.91,'Phazap: 105+ pairs on HOLD (undefined phase in one shared event)',
                ha='center',va='top',fontsize=9,color='#8a2947')
        fig.suptitle('O3 frozen scope: 5 / 15 / 46 / 62 events',fontsize=12)
        fig.text(.5,.015,'Existing public PE products; shared load; GPU vs six CPU workers. No equal-recall speedup claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.06,1,.94])
        fig.savefig(root/'figures/SCALING.png',dpi=180);fig.savefig(root/'figures/SCALING.pdf');plt.close(fig)
    lines=['# TriLens / Phazap 四量级评分计时', '',
        '**状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。**', '',
        ('本轮四个规模均完成并通过评分复现检查。' if passed else
         'TriLens 四个规模已全部完成并通过归档复核；Phazap 仅10对规模完成，105对因非有限相位停止，1,035和1,891对不再启动。不能作为完整成功比较。' if final.get('trilens_all_scales_complete') else
         '本轮存在失败或未完成规模，不能作为完整成功比较。'),
        '本报告是在线评分成本测量，不是等透镜召回率的质量-成本比较，也没有运行官方 PO/FPP、Fast-GOLUM 或 Hanabi。', '',
        '## 范围与边界', '',
        '冻结 O3 strict H1/L1 的 62 个事件，沿用前一轮事件名称哈希排序；取 5、15、46、62 事件的嵌套前缀，对应全部 10、105、1,035、1,891 个无序 pair。未按 PE、官方 FPP、排名或运行速度替换事件。子目录共享事件，不是四个独立天体物理实验。', '',
        '首次处理每个规模实测一次：从本地原始 strain、公开 posterior/map 开始，计入事件准备、全部配对、排序和结果写入。双方均不计原始 PE 生产、下载和离线训练/校准；Python 主进程冷启动不在该总数内。不清空 OS 缓存，不称磁盘冷启动。', '',
        '事件准备后，对同一规模从事件级中间量重算全部 pair 两次。TriLens 缓存单事件网络预测、规范化天空图和固定校准配置；Phazap 缓存单事件相位后验。双方均未用缓存 pair 分数代替计算。TriLens 仍重新计算三通道、三模型排序和共识；Phazap 仍调用作者 phazap() 重新计算每对 DJ 等统计量。', '',
        'TriLens 包含原始应变读取、PSD、2 s/16 s 预处理、模板特征、三个冻结 seed 的全部网络与校准、NESTED→RING、Nside=512 天空和时间。没有外层旧/新总分混合。天空 BC 仅用于诊断、不是 PATH1 排名输入，因此本轮不将它计入必要评分成本；前一轮六事件 wrapper 额外计算了它，不能把两轮数字直接相减解释为模型加速。', '',
        'Phazap 使用未修改的作者 0.3.3 实现及每事件全部已有 posterior 样本；样本数、原 group 和配置见 PE_INPUT_INVENTORY.csv。前一轮小样本收敛不足，所以本轮没有以抽样削减来优化计时。这不表示 Phazap 理论上必须使用每一个抽样点。', '',
        '输入审计先发现 GW200220_124850 的 C01:Mixed 组没有单一波形配置，第一版在计时前停止并保留失败包。修订版在计时前冻结统一 Phazap XPHM 后验输入：其余61事件本来就是 XPHM，此事件使用同一公开文件中的 C01:IMRPhenomXPHM 完整样本及原配置。TriLens 仍使用其冻结的 Mixed 天图。这一个事件的 PE 产品组不同已在 manifest 标记，不能声称所有输入产品完全相同，也没有按其评分结果选择后验。', '',
        '随后 Phazap 的15事件准备出现 GW200322_091133 的非有限相位：4,140个后验样本中268个（约6.47%）的100 Hz相位/20–100 Hz相位演化为非有限值。原程序停止，不把这些样本删除或置零。support_audit 为独立零信号支撑审计：模板在100 Hz处两种偏振为零时，此处相位没有定义。完整审计若存在，见 support_audit_full；抽样审计见 support_audit。', '',
        '该事件位于固定的前15事件内，两个更大的嵌套目录同样包含它，故完整作者基准不能继续。此次没有调低 fhigh、替换事件或删去后验尾部来强行得到对照曲线。Phazap 10对点只读引用前一版已通过的计时，来源列单独记录。两种方法的时间不是同一瞬间同时测量，不能宣称等资源/等质量的速度倍数。', '',
        'TriLens 扩展测试也发现并保留了一次复现失败：计时适配器给 fine-mass 分支传入了已经标准化的短窗，而归档实现直接截取 full[:,:,-4096:]、在 fine.features 内部标准化。重复标准化使 fine 特征最大变化9.54e-7；coarse 和16秒 low 特征逐值不变，但低精度网络与尾部校准将部分差异放大。46事件时2个 pair 超差；独立62事件诊断中两个 seed 分别有2个、4个 pair 超差，最大0.132。诊断输出未替换任何排名。', '',
        '新目录只修复计时适配器，使其严格调用原 fine 分支输入路径；模型、训练、校准、权重、时间、天空和复现容差均未修改。先前失败的 FINAL_STATUS、checks、日志和特征比较表保留在 reference_runs/replay_failure。只有新目录逐 seed 评分和共识复现通过的规模才是有效计时，不能用放宽容差掩盖失败。', '',
        '## 实测耗时', '',
        '| 方法 | 事件/对 | 事件准备(s) | 首次准备+配对(s) | 复用事件后的配对中位数(s) | 两次范围(s) |',
        '|---|---:|---:|---:|---:|---:|']
    if not data.empty:
        for n in [5,15,46,62]:
            for method in ['TriLens','Phazap']:
                d=data[(data.n_events==n)&data.method.eq(method)]
                prep=d[d.boundary.eq('PREPARE_EVENTS')]
                total=d[d.boundary.eq('PRODUCTS_READY_TOTAL')]
                w=d[d.boundary.eq('EVENT_CACHE_PAIR_RANK')]
                val=lambda s:'未完成' if len(s)==0 else f'{s.iloc[0]:.4f}'
                med='未完成' if w.empty else f'{w.wall_s.median():.4f}'
                span='未完成' if w.empty else f'{w.wall_s.min():.4f}–{w.wall_s.max():.4f}'
                lines.append(f'| {method} | {n}/{n*(n-1)//2} | {val(prep.wall_s)} | {val(total.wall_s)} | {med} | {span} |')
    lines += ['', '表中首次总耗时是直接墙钟测量，不是各任务耗时相加。Phazap 六个单线程进程并行；各 worker 的 CPU 秒和 job wall 秒在逐事件/逐对表中另记。job 时长之和不能当作并行完工时间。首次总计中含 worker 启动及相位载入。只有一次首次测量，暖缓存两次范围也不是总体置信区间。', '',
        '## 数值与资源审计', '',
        f"评分与归档、重复计算的检查：{final['checks_passed']}/{final['checks_total']} 通过。追踪输入的前后哈希一致：{final['protected_inputs_unchanged']}。失败记录：{final.get('error')}。", '',
        '所有 TriLens 规模均逐 seed 对照归档 PATH1 的 waveform/time/sky 数值和子目录共识 rank；容差在运行前冻结，不能因为大规模不通过而放宽。Phazap 比较前一轮已验证的重叠 pair，并检查全部 pair 的重复确定性。这些检查是软件复现，不是证明其物理检出性能。', '',
        '为保持归档 BF16 内核行为，TriLens 神经网络仍按原 62 个槽位处理，小目录空槽为零值；额外计算全部计入。62 事件点没有额外占位事件。未宣称小目录推理已优化，也不据四点拟合超出测量范围的性能。', '',
        'TriLens 使用 RTX 5090 和至多两条 CPU 数值库线程；Phazap 使用六个单线程 CPU workers。UAB 两组统一实验未停止，故这些是共享服务器负载下、各自实现的计算成本，不是严格等资源比较。CPU/GPU 时间和能耗尚未完整积分，不公布虚构的总核时节省。', '',
        '输入、软件版本、batch 策略、日志、全部 pair 统计量均保留。没有重新训练模型、重新选权、修改原真实排名或论文。后台其他实验的进度不构成本轮的检出效果。', '',
        '## PE 与科学解释', '',
        'TriLens 波形评分不读取公开质量/自旋联合后验；当前真实天空评分读取公开 PE 天图。不能把这称为整条真实流程完全不依赖 PE。Phazap 读取参数后验并生成相位分布；已有相位时可直接配对，两种使用边界都在本轮报告。', '',
        'Phazap 的 DJ / upstream p_value 不等于官方目录背景 FPP，DJ 排序只作诊断；本轮未完成 PO/Phazap 联合官方门槛复现。真实事件没有本实验的透镜真值，不报告真实 Recall 或把官方重合当真阳性。', '',
        '要声称在相同检出能力下减少后续贝叶斯工作量，仍需 source/noise-disjoint 注入的合格后验、validation 冻结阈值、持出实际 pair recall 与假对数，以及相同后续算法的计时。只有这四个规模耗时，不足以证明该主目标。', '',
        '## 交付', '',
        f'服务器：connect.westd.seetacloud.com:32328。目录：`{root}`。',
        '紧凑包排除原始 strain/PE、模型大文件和可重建 phase HDF5；phase HDF5 在服务器本轮目录中仍保留。包内含合同、事件清单、脚本、计时、逐对统计、成功/失败日志、图和逐文件哈希。',
        '[Phazap 作者软件](https://github.com/ezquiaga/phazap)。', '', 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE']
    if component_summary is not None and not component_summary.empty:
        text=['## TriLens 事件准备耗时分解', '',
            '这是准备阶段子步骤的实测墙钟和，完整总耗时仍以前表直接测量为准。所有三 seed 网络都计入；并非一次神经网络推理需要这么久。', '',
            '| 事件数 | 应变读取+PSD(s) | 特征生成(s) | 网络/校准文件读取与推理(s) | 天图读取/ordering/512(s) |',
            '|---:|---:|---:|---:|---:|']
        for n,g in component_summary.groupby('n_events'):
            def cost(names):return g[g.stage.isin(names)].wall_seconds.sum()
            io=cost(['strain_reference_IO','PSD_estimation'])
            feature=cost(['short_preprocessing','low16s_preprocessing','spectral_bank_IO','coarse_phase_feature','fine_mass_feature','low16s_feature'])
            network=cost(['short_checkpoint_inference','RNC_checkpoint_inference','ordered_mass_checkpoint_inference','multirate_joint_checkpoint_inference'])
            sky=cost(['sky_IO_ordering_Nside512'])
            text.append(f'| {n} | {io:.3f} | {feature:.3f} | {network:.3f} | {sky:.3f} |')
        index=lines.index('## 数值与资源审计')
        lines[index:index]=text+['']
    (root/'reports/SCALING_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    (root/'README_CN.md').write_text('# TRILENS-SCALE-03\n\n先读 reports/SCALING_REPORT_CN.md。\n\n仅评分计算成本；无等召回率或最终贝叶斯确认声明。\n')
    files=[]
    for p in root.rglob('*'):
        if not p.is_file() or 'phases' in p.relative_to(root).parts or '__pycache__' in p.parts:continue
        if p.suffix in ('.npy','.npz','.pt','.hdf5'):continue
        if p.stat().st_size>64*2**20:raise RuntimeError('Unexpected large compact artifact: '+str(p))
        content=p.read_bytes()
        if any(marker in content for marker in (b'BEGIN '+b'RSA PRIVATE KEY',b'BEGIN '+b'OPENSSH PRIVATE KEY',b'sshpass '+b'-p')):
            raise RuntimeError('Credential marker')
        files.append(p)
    manifest.write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(files)))
    files.append(manifest)
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(files):archive.add(p,arcname=root.name+'/'+str(p.relative_to(root)),recursive=False)
    failures=[]
    with tarfile.open(package,'r:gz') as archive:
        for line in manifest.read_text().splitlines():
            expected,name=line.split('  ',1)
            if hashlib.sha256(archive.extractfile(root.name+'/'+name).read()).hexdigest()!=expected:failures.append(name)
    if failures:raise RuntimeError('Package integrity failed')
    value=sha(package)
    package.with_suffix('.gz.sha256').write_text(value+'  '+package.name+'\n')
    verification=dict(package=str(package),sha256=value,bytes=package.stat().st_size,members=len(files),
        internal_hash_failures=failures,all_scales_complete=passed,method_superiority_established=False)
    package.with_suffix('.verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
