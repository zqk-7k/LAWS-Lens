#!/usr/bin/env python3
"""Assemble a guarded development-plus-independent-injection delivery."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import numpy as np
import pandas as pd
P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_path_release_v3_20260908 as release
import mcwf_path_fresh_confirmation_v2_20260908 as fresh
import mcwf_path_fresh_audit_20260908 as audit
import mcwf_document_mechanisms_v9_20260908 as plotting
dev,t,cf=release.dev,release.t,release.cf
CODE=release.CODE


def copy_verified(source,target,inventory):
    if source.is_symlink():raise RuntimeError('Do not bundle symlink:'+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    digest=dev.sha(source)
    if target.exists():
        if dev.sha(target)!=digest:raise RuntimeError('Refuse to overwrite different artifact:'+str(target))
    else:shutil.copy2(source,target)
    if dev.sha(target)!=digest:raise RuntimeError('Copied artifact failed hash')
    inventory.append({'original_path':str(source),'delivery_path':str(target),'bytes':source.stat().st_size,'sha256':digest})


def frozen_checks(development,confirmation):
    fresh.verify(confirmation)
    checked=[]
    for origin in (development,confirmation):
        for r in pd.read_csv(origin/'manifest/INPUT_SHA256.csv').to_dict('records'):
            digest=dev.sha(Path(r['path']))
            checked.append({'path':r['path'],'before':r['sha256'],'after':digest,'unchanged':digest==r['sha256']})
    if not all(x['unchanged'] for x in checked):raise RuntimeError('Protected input changed')
    return pd.DataFrame(checked).drop_duplicates('path')


def prepare(root,development,confirmation):
    if root.exists():raise RuntimeError('Require new final delivery directory')
    assessment=json.loads((confirmation/'contracts/FRESH_ASSESSMENT.json').read_text())
    independence=json.loads((confirmation/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json').read_text())
    if not assessment['fresh_guard_pass'] or not independence['source_noise_independence_pass']:
        raise RuntimeError('Independent confirmation failed; cannot issue successful delivery')
    if not (confirmation/'contracts/BOOTSTRAP_COMPLETE.json').exists():raise RuntimeError('Uncertainty incomplete')
    if not (confirmation/'contracts/NONOVERLAP_SENSITIVITY_COMPLETE.json').exists():raise RuntimeError('Noise-overlap diagnostic incomplete')
    if not json.loads((development/'contracts/SELECTED_EXPORT_COMPLETE.json').read_text())['search_replay_exact']:
        raise RuntimeError('Selected ranking replay failed')
    for name in ('contracts','tables','reports','figures','scripts','manifest','models','development','independent_confirmation'):
        (root/name).mkdir(parents=True)
    rows=[]
    for origin,dest,folders in (
        (development,root/'development',('contracts','tables','evaluation','mechanism_audit','scripts','logs')),
        (confirmation,root/'independent_confirmation',('contracts','tables','scripts','logs'))):
        for name in folders:
            for source in sorted((origin/name).rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts:
                    copy_verified(source,dest/source.relative_to(origin),rows)
    for source in sorted((confirmation/'confirmation').rglob('*')):
        if not source.is_file():continue
        rel=source.relative_to(confirmation)
        if 'events' in rel.parts:continue
        if source.suffix in ('.csv','.json','.parquet') and source.stat().st_size<100*2**20:
            copy_verified(source,root/'independent_confirmation'/rel,rows)
    failed=P/'results/mcwf_path25_fresh_confirmation_20260908T172752Z'
    if failed.exists():
        for name in ('contracts','scripts','logs','tables'):
            for source in sorted((failed/name).rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts:
                    copy_verified(source,root/'implementation_attempts/unit_v1'/source.relative_to(failed),rows)
    modelpaths=set()
    for r in pd.read_csv(confirmation/'manifest/INPUT_SHA256.csv').to_dict('records'):
        p=Path(r['path'])
        if p.suffix=='.pt' and p.name!='resume.pt':modelpaths.add(p)
    for dep in t.DEPS:
        for slot,seed in zip(t.MODEL_SLOTS,t.SEEDS):
            modelpaths.add(fresh.e.TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{slot}/validation_selected_model.pt')
            modelpaths.add(dev.V7/dep/f'seed_{seed}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt')
    for source in sorted(modelpaths):copy_verified(source,root/'models'/source.relative_to(P),rows)
    calibrations={dev.BAY/'contracts/selected_config.json',
        t.PREVIOUS/'fine_mass_context/contracts/FEATURES.json'}
    for dep in t.DEPS:
        calibrations.add(dev.V7/dep/'shared/time_delay_likelihood_ratio.json')
        for seed in t.SEEDS:
            calibrations.add(dev.V7/dep/f'seed_{seed}/results/waveform_channel_calibration_v7.json')
            calibrations.add(fresh.e.BASE/f'calibration/{dep}/seed_{seed}/SELECTED_CONFIG.json')
    for source in sorted(calibrations):
        copy_verified(source,root/'frozen_calibrations'/source.relative_to(P),rows)
    fresh.old.modules()
    sources=set()
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).is_relative_to(P):sources.add(Path(file))
    for source in sorted(sources):copy_verified(source,root/'dependency_source'/source.relative_to(P),rows)
    copy_verified(Path(__file__),root/'scripts/path_finalize.py',rows)
    for source in (development/'manifest/INPUT_SHA256.csv',confirmation/'manifest/INPUT_SHA256.csv'):
        copy_verified(source,root/'manifest'/('DEVELOPMENT_INPUT_SHA256.csv' if source.parent.parent==development else 'CONFIRMATION_INPUT_SHA256.csv'),rows)
    dev.csv_write(root/'manifest/COPIED_ARTIFACTS.csv',pd.DataFrame(rows))
    dev.csv_write(root/'manifest/PROTECTED_BEFORE_AFTER.csv',frozen_checks(development,confirmation))
    dev.json_write(root/'contracts/DELIVERY_CONTRACT.json',{
        'UTC':datetime.now(timezone.utc).isoformat(),'code':CODE,'development_root':str(development),'confirmation_root':str(confirmation),
        'scope':{'gwtc3':{'events':62,'unordered_pairs':1891},'gwtc4':{'events':74,'unordered_pairs':2701}},
        'same_algorithm_both_runs':True,'global_alpha':.875,'raw_time_sky_unchanged':True,'outer_weights_changed':True,
        'real_selection':'ADAPTIVE_DEVELOPMENT_NOT_BLIND_VALIDATION','independent_check':'NEW_SOURCE_NOISE_SIMULATION_GUARDRAILS',
        'no_Hanabi_run':True,'official_overlap_is_not_lensing_truth':True,'no_paper_or_historical_adoption':True,
        'status':t.STATUS,'goal_achieved':False,'delivery_validation_pending':True,
        'omitted':['GWOSC strain','PE HDF5','dense HEALPix maps','large joint probability arrays','raw and feature caches','installed runtime libraries'],
        'not_standalone_without_manifest_inputs':True})


def md_table(frame,columns):
    out=['|'+'|'.join(columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    for _,row in frame.iterrows():
        cells=[]
        for col in columns:
            v=row[col]
            if pd.isna(v):s='NA'
            elif isinstance(v,(float,np.floating)):s=f'{v:.4f}'
            else:s=str(v)
            cells.append(s.replace('|','/'))
        out.append('|'+'|'.join(cells)+'|')
    return out


def report(root):
    contract=json.loads((root/'contracts/DELIVERY_CONTRACT.json').read_text())
    development=Path(contract['development_root']);confirmation=Path(contract['confirmation_root'])
    allbudgets=pd.read_csv(development/'tables/SELECTED_PE_OFFICIAL_BUDGETS.csv')
    b=allbudgets[(allbudgets.seed.astype(str)=='consensus')&(allbudgets.method=='fusion')].copy()
    perseed=allbudgets[(allbudgets.seed.astype(str)!='consensus')&(allbudgets.method=='fusion')&allbudgets.budget.isin([10,20])].copy()
    dev.csv_write(root/'tables/PER_SEED_REAL_TOP10_TOP20.csv',perseed)
    search=pd.read_csv(development/'tables/ALL_GLOBAL_CONFIGURATIONS.csv')
    selected=json.loads((development/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json').read_text())
    platform=search[search.upstream_method==selected['upstream_method']].copy()
    dev.csv_write(root/'tables/GLOBAL_ALPHA_SENSITIVITY.csv',platform)
    freshmetrics=pd.read_csv(confirmation/'tables/FRESH_METRICS_PER_CATALOG_MODEL.csv')
    means=pd.read_csv(confirmation/'tables/FRESH_MODEL_MEAN_METRICS.csv')
    metrics=['macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9']
    fs=means.groupby(['configuration','deployment','mode'])[metrics].agg(['mean','std']).reset_index()
    fs.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in fs.columns]
    dev.csv_write(root/'tables/FRESH_SUMMARY_BY_TRAINING_SEED.csv',fs)
    dev.csv_write(root/'tables/TOP_B_PE_OFFICIAL_COMPARISON.csv',b)
    hubs=[]
    for dep in t.DEPS:
        for name in ('OMC',CODE):
            pairs=pd.read_parquet(development/f'evaluation/{name}/{dep}/consensus_fusion_all_pairs.parquet')
            for budget in (10,20,50,100):
                top=pairs.head(budget)
                degrees=Counter(top.event_i.tolist()+top.event_j.tolist())
                hubs.append({'deployment':dep,'configuration':name,'budget':budget,'unique_events':len(degrees),
                    'maximum_event_degree':max(degrees.values()),'largest_hub_edge_fraction':max(degrees.values())/len(top),
                    'interpretation':'Descriptive shared-event dependence; not independent pairs.'})
    dev.csv_write(root/'tables/REAL_SHARED_EVENT_HUB_AUDIT.csv',pd.DataFrame(hubs))
    weights=pd.read_csv(development/'tables/SELECTED_WEIGHTS.csv')
    dev.csv_write(root/'tables/FINAL_EFFECTIVE_WEIGHTS.csv',weights)
    independence=json.loads((confirmation/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json').read_text())
    guards=pd.read_csv(confirmation/'tables/FRESH_GUARDRAILS.csv')
    selections=json.loads((confirmation/'contracts/FRESH_FREEZE.json').read_text())
    aliases=[]
    for name in ('mcwf_end_to_end_intrinsics_exploratory_20260908T170714Z','mcwf_large_multirate_exploratory_20260908T171624Z'):
        r=P/'results'/name
        assessment=json.loads((r/'contracts/ROUND_ASSESSMENT.json').read_text())
        aliases.append({'root':str(r),'qualifiers':assessment['development_candidates_requiring_fresh_confirmation'],'status':assessment['status']})
    archive_rows=[]
    for p in sorted((P/'packages').glob('mcwf*20260908T*_deliverables.tar.gz')):
        receipt=p.with_suffix(p.suffix+'.sha256')
        if receipt.exists():archive_rows.append({'package':str(p),'bytes':p.stat().st_size,'declared_sha256':receipt.read_text().split()[0]})
    dev.csv_write(root/'tables/RELATED_EXPLORATION_ARCHIVES.csv',pd.DataFrame(archive_rows))
    dev.json_write(root/'contracts/ADDITIONAL_NEGATIVE_ROUNDS.json',aliases)
    lines=[f'# {CODE}：统一波形改进与独立注入保护检验','',
        '## 结论与边界','',
        '本版在已经反复查看的 O3/O4a 真实开发目录上，达到本轮预先列明的共同改善目标；冻结后另用新源、新噪声完成注入保护检验。它不是未见真实目录上的盲测成功，也不是透镜探测或 Hanabi 确认。',
        '版本名 DEVCONF 表示 development 对照加独立注入保护，不表示真实候选得到贝叶斯确认。阶段快照中的 pending/goal_achieved=false 是当时状态；最终任务状态以本包 FINAL_OUTCOME.json 和包外 DELIVERY_CHECK.json 为准。',
        'O3 与 O4a 使用同一评分流程、同一全局替换系数 0.875、同一保护规则。各运行期的模拟训练模型、校准和逐 seed 外层权重不同，这不等于两套方法。',
        '本轮没有修改历史 v9.3/v9.4/C-fixed/OMC、ET-3、论文或 Overleaf。最终状态：`'+t.STATUS+'`。','',
        '## 最关心的 Top-10/20','',
        'Mc 相容计数采用公开探测器系 chirp-mass 后验 BC >= 0.5；灾难性不相容定义 BC < 0.1 或 D_Mc > 5。Dmax 同时审计 Mc、q、chi_eff。下表中的官方列是 1% 前端表重合，不是已确认透镜数。',
        '旧版 OMC 与新版均采用固定跨 seed 共识规则：平均 rank、最大 rank、平均分依次打破平局。Top-10 包含于 Top-20，两个预算不能作为独立样本相加做显著性检验。','']
    lines+=md_table(b[b.budget.isin([10,20])],[ 'deployment','config','budget','BC_mc_ge_0p5','median_BC_mc','catastrophic_mc','Dmax_le_3','official_frontend','official_hanabi'])
    correlations=pd.read_csv(development/'tables/SELECTED_REAL_SCORE_PE_CORRELATIONS.csv')
    correlations=correlations[(correlations['mode']=='fusion')&(correlations.score=='waveform_score_mean')]
    lines+=['','完整真实 pair 集合中，波形分数与公开 Mc 一致性的相关性如下。它也是适应性开发后的描述性统计；共享事件的 pair 不独立，因此不报告普通 pair-level p 值。','']
    lines+=md_table(correlations,['configuration','deployment','PE','spearman'])
    lines+=['','O4a 的 Top-20 官方重合维持不变；改善来自 Top-10，不能写成每一个预算的官方计数都增加。Top-50/100 的全部指标也保留，并未要求这些扩展预算全部单调改善。','',
        '## 到底改了什么','',
        '旧 InceptionAttention 编码器保留，没有覆盖或重训旧 checkpoint。旧波形证据仍包含 embedding cosine、模拟质量/质量比辅助预测及原校准。保留版随后已有 RAW-PHASE 质量预测器的有限参考尾部约束，以及 ordered-mass 预测器的质量一致性校准。',
        '本轮选中的新增组件为：一个同时读旧短时特征与 20--80 Hz、16 s 低频特征的质量 CNN，以及一个条件 eta/chi_eff 预测 head。它们只用模拟真值训练，不读真实 PE。预测 head 在冻结新质量 backbone 后训练；这次胜出者不是后面试过的端到端联合重训模型。',
        '每运行期、每 seed 的有效模块链为：旧统一 encoder -> 已保留 RAW-PHASE 预测器 -> 已保留 ordered-mass 预测器 -> 新多时间尺度质量预测器 -> 新条件 eta/chi_eff head。后三类质量预测不是完整 PE，不能称为公开参数后验的替代。',
        '新增低频分支保留原 2 s、4096 点主输入，同时从同一个更长的原始信号提取低频 16 s、4096 点。使用相同 PSD whitening、真实频率滤波和抗混叠重采样；不是把已经归一化的 2 s 振幅当作物理 SNR。',
        '原始 Z_time、Z_sky、公开 PE 地图、目录范围均未改变。但是外层 waveform/time/sky 权重确实变了，因此“只有波形分数变化、时间天空贡献完全不变”是不准确的；变的是证据组合，不是时间或天空物理模型。','',
        '## 新波形分数的计算','',
        r'$\eta=q/(1+q)^2$，$\chi_{\rm eff}=(m_1 a_1\cos t_1+m_2 a_2\cos t_2)/(m_1+m_2)$。',
        r'模型给出离散预测质量分布 $p_i(M)$ 和条件分布 $p_i(\eta,\chi_{\rm eff}\mid M)$；联合预测为 $p_i(M,\eta,\chi_{\rm eff})=p_i(M)p_i(\eta,\chi_{\rm eff}\mid M)$。代码检验其质量边际不变。',
        r'波形预测的一致性 $BC^{\rm pred}_{ij}=\sum_{m,e,c}\sqrt{p_i(m,e,c)p_j(m,e,c)}$。它与后续真实 PE 审计的 $BC_{\mathcal M_c}$ 是两个不同的量。',
        r'用模拟 validation 参考真对构造有限样本尾部概率 $p_{\rm tail}$，惩罚 $T=\min[\log(p_{\rm tail}/0.05),0]$；再用模拟真/假对的 isotonic 密度比映射得到有界增量 $I\in[-4,4]$。',
        '参考集按独立源处理。超出支持域或真对尾部约束不足时禁止正奖励；预测质量端点概率大于 0.25 时新增项中性回退。不是根据真实候选是否出现在官方表里加分。',
        r'$Z_{\rm up}=Z_{\rm OMC}+\gamma T+\beta I$。所有内部 gamma/beta、温度、支持域和映射均来自之前的模拟选择，逐 seed 完整保存。','',
        '## 统一的保守替换路径','',
        r'先把旧、新外层权重分别归一化为 $\sum w=1$。设 $a=0.875$，$w^*=(1-a)w^0+a w^1$。',
        r'$Z^*_{\rm wf}=\{(1-a)w_W^0Z_{\rm OMC}+a w_W^1Z_{\rm up}\}/w_W^*$。',
        r'$S^*=w_W^*Z^*_{\rm wf}+w_T^*Z_{\rm time}+w_S^*Z_{\rm sky}=(1-a)S^0_{\rm norm}+aS^1_{\rm norm}$。',
        '这仍是三个矩阵的线性组合，不是第四个 PE 检索通道，不是后验概率。未增加逐候选权重、事件名称规则或手工置顶。共识排名沿用原规则。',
        r'$Z_{\rm time}=\log[p(\Delta t\mid L)/p(\Delta t\mid N)]$ 读取同一冻结的一维 lookup；$Z_{\rm sky}=\log[N_{\rm pix}\sum P_iP_j]$ 读取同一冻结天空分数。没有为迎合名单改变其先验或地图。','']
    lines+=md_table(weights,['deployment','seed','effective_weight_waveform','effective_weight_time','effective_weight_sky'])
    attribution=pd.read_csv(development/'tables/ATTRIBUTION_PE_OFFICIAL_BUDGETS.csv')
    dev.csv_write(root/'tables/FROZEN_COMPONENT_ATTRIBUTION.csv',attribution)
    lines+=['','权重大小本身不能等同通道重要性，因为各证据的数值尺度不同。应结合每个候选的实际加权贡献、单通道和消融结果判断。',
        '逐 pair 的推理输入只有两端波形预测、两事件时间差和两张天空图。训练标准化、预测先验、密度映射和权重来自全局开发数据；时间 lookup 还使用固定透镜/背景分布。最后排名当然取决于整个候选集合。没有在真实目录内重新做 row-wise z-score，也没有以真实候选 PE 逐行修正分数。',
        '下表为配置冻结之后补做的归因对照：OLD-WAVEFORM-NEW-WEIGHTS 只换外层权重；NEW-WAVEFORM-OLD-WEIGHTS 只换波形。它不参与再次选择。本轮完整收益不能全部归给新网络。','']
    lines+=md_table(attribution[attribution.budget.isin([10,20])],['deployment','config','budget','BC_mc_ge_0p5','median_BC_mc','official_frontend','official_hanabi'])
    lines+=['','## 选择过程，必须披露','',
        '最终这一步在 34 个已经冻结的正权重模型与 10 个全局替换系数上比较，共 340 个预列配置。先检查两个运行期全部 seed 的模拟 validation/复用 test 保护；再按真实开发目录 Top-10/20 的共同目标选择一个全球一致配置。仅一个点同时满足全部开发条件。',
        '因此最终全局系数与方法选择使用了真实 PE/官方预算反馈，不能称 validation-only 或完全没有利用真实标签。标签没有进入网络输入和逐 pair 公式，但用于选择全局方案仍是适应性开发，会产生选择乐观偏差。',
        '内部网络和密度校准使用模拟数据；最终全局选择与真实目录开发这一层必须在论文方法与局限性中单独说明。旧 test 在本轮已反复使用，只称复用保护集。',
        '重要限制：同一上游方法的 10 个替换系数中，只有 0.875 同时满足全部真实共识预算保护。这不是宽阔的稳定平台。逐 seed 也不是所有 PE/Hanabi 指标都改善，例如 O4a seed 202607243 的 Top-10 median BC 和 Hanabi 表重合下降。不能把共识改善改写成三个 seed 每一项都一致改善。所有逐 seed 结果和邻近系数结果分别列在 PER_SEED_REAL_TOP10_TOP20.csv 与 GLOBAL_ALPHA_SENSITIVITY.csv。',
        '没有依据新独立注入结果重新选择系数。即使独立注入检验通过，也只支持该模拟域的保护，不会把已经适应性选择的真实重合自动变成独立证据。','',
        '## 训练与负对照','',
        '胜出新增质量网络：每运行期 4096 training sources、96 noise blocks、8 views/source；512 development sources、32 blocks；从原质量模型 warm start，15 epochs、AdamW lr=1e-4、weight_decay=1e-4、gradient clip=5；按 development cross entropy 选 checkpoint，包含 epoch 0。',
        '条件 eta/chi_eff head：质量 backbone 固定，25 epochs、AdamW lr=1e-3、weight_decay=1e-4、clip=5、cosine schedule；每源等权。温度候选 0.5/0.75/1/1.25/1.5/2，只在模拟 development 选择。',
        '后续的端到端联合 head/backbone 重训和扩大低频训练人口对照没有满足两个运行期共同目标，均保留，不把它们拼接成不同运行期的“最佳模型”。全部更早负结果见探索归档索引。','',
        '## 全新注入确认','',
        '冻结后采用 202609941/942/943 三个新 catalog seeds。每运行期、每 catalog 为 35 smooth + 35 subhalo 透镜源及 50 背景源，共 190 events/70 真对；总计 720 新源、1140 events、420 真对。每 catalog 使用 16 个新 256 s 噪声块，总计 96 blocks。',
        '这是透镜富集的小目录检验：每 catalog 共 17,955 个无序 pair，其中 70 个真伴随、17,885 个非伴随。它不代表真实低透镜率人口，不能把其 precision 直接解释为真实目录的透镜候选可信度。',
        '使用实际 GPS 区间和 guard 检查噪声独立，而非仅比较 bank index；新源 UID 和内禀质量与旧训练/开发/确认源交叉审计。不同 model seed 评估的是同一批事件，不能把 3 个模型当作 3 倍独立事件。',
        '生成使用原物理 H1/L1 响应及真实 off-source 噪声。新低频输入逐事件重放原 full24 数组，必须 bit-exact；再提取低频分支。',
        '注入天空仍为冻结的条件 BAYESTAR：使用已知内禀参数、局部 PSD 与独立高斯 matched-filter 测量误差，并不是对这段完整非高斯注入 strain 做完整 PE。Nside=512；同一集合的图保持统一 NESTED 概率质量约定，统一像素置换不改变 overlap。真实历史图的 NESTED->RING 修正版保持不动。',
        '保留限制：这是覆盖型 Mc=5--200/SNR=8--40 proposal，不是天体物理事件率样本；逐像目标 SNR 缩放仍保留，不是 response-derived SNR-ratio 新实验。GW-LMC lens environments 可以复用，不能称独立透镜人口检验。O3 噪声为 O3，但旧累计 O1--O3 response calendar 仍保留。',
        '预冻结保护：每 run、每模型、waveform/fusion 两种模式，各自对三个新 catalog 取均值；R@10 下降不超过 0.02、AUPRC 下降不超过 0.005、F50/F90 不超过旧版 1.10 倍。全部 12 项必须通过，不能用 O3 弥补 O4a。','']
    lines+=md_table(fs,['configuration','deployment','mode','macro_r_at_1_mean','macro_r_at_1_std','macro_r_at_10_mean','macro_r_at_10_std','average_precision_mean','false_at_recall_0p5_mean','false_at_recall_0p9_mean'])
    windows=pd.read_csv(confirmation/'tables/WITHIN_CATALOG_NOISE_WINDOW_AUDIT.csv')
    lines+=['',f"目录内真伴随系统中，两次噪声窗口发生重叠的数量为 {int(windows.same_system_noise_windows_overlap.sum())}/{int(windows.true_systems.sum())}。新噪声块与旧训练/开发数据隔离，不代表目录内每对窗口都互不相关；逐 catalog 审计与 noise-block bootstrap 均保留。"]
    noisecheck=json.loads((confirmation/'contracts/NONOVERLAP_SENSITIVITY_COMPLETE.json').read_text())
    lines += [f"另按实际噪声区间重叠这一数据质量规则，移除上述 {noisecheck['excluded_systems']} 个系统的双像，构建完全相同的旧/新诱导子目录。重新检查的跨 catalog 模型均值保护结果为 {noisecheck['model_run_mode_mean_guards_pass']}。该诊断不用于调参，不替换原主测试，也不是另一套独立确认数据。"]
    lines+=['','上表 SD 是三个模型 seed 的跨 catalog 均值的标准差，不是九次独立训练。逐 catalog、逐模型的全部 72 行指标另存，不隐藏个别 catalog 的起伏。',
        '全部 12 项预设保护均通过，但不代表每一项数值都严格变好：O4a model 202609062 的 fusion F90 均值增加 35，仍在预先规定的 10% 范围内。两个运行期整体均值的 F50/F90 下降，R@1/R@10 和 AUPRC 上升。',
        'R@1/R@10 以透镜源为抽样单位，两个 directed queries 一起进行 10,000 次 bootstrap；pair 指标做 2,000 次 source bootstrap；另做 2,000 次 noise-block bootstrap。二者分别报告，不假装一个简单区间同时涵盖源和噪声的所有依赖。',
        '小目录无法解析 1e-5 的经验 false rate。旧 helper 的“至少 1 个 FP”列另外给出实际 false rate，不能直接当作 1e-5 FPR 性能。','',
        '## 全部真实 Top-10','']
    for dep in t.DEPS:
        f=pd.read_csv(development/f'evaluation/{CODE}/{dep}/consensus_fusion_top10.csv')
        f['pair']=f.event_i+'--'+f.event_j
        official='official_po_or_ml_fpp_below_0p01' if dep=='gwtc3' else 'official_po_or_phazap_fpp_below_0p01'
        lines+=['',f'### {dep}','']+md_table(f,['consensus_rank','pair','final_score_mean','waveform_contribution_mean','time_contribution_mean','sky_contribution_mean','pe_mc_bhattacharyya_coefficient','pe_dmax_intrinsic',official,'official_any_pair_resolved_hanabi_overlap'])
    lines+=['','完整 Top-20/50/100、全部 pair Parquet/UTF-8 BOM CSV、PO/ML 或 PO/Phazap FPP、GOLUM/Fast-GOLUM/Hanabi 公开阶段与结论均在 development/evaluation 中。保留原始公开结论，不把“公开 Hanabi 表重合”写成 Hanabi 支持透镜。',
        '公开 PE 的 BC、Dmax 只在排名完成后联表，但其预算统计曾参与全局开发选择。没有运行新的 Hanabi，也没有得到新的 joint evidence、posterior odds 或 catalog FPP。','',
        '## 复算与文件','',
        '`development/contracts`：340 配置合同、完整选择台账、冻结最终配置与导出检查。',
        '`development/evaluation`：旧 OMC 与新版本的 validation/复用 test/真实排名及 PE、官方表。',
        '`independent_confirmation`：新源和噪声 manifest、原始生成协议、逐 catalog/model pair scores、bootstrap、独立性审计和运行日志。',
        '`models`：所选旧/新增 checkpoint 的路径保持快照；`dependency_source`：项目内调用源码快照。',
        '`manifest`：输入与输出 SHA-256、历史输入前后复核；`tables/RELATED_EXPLORATION_ARCHIVES.csv`：负结果归档索引。',
        '为控制体积，不打包 GWOSC strain、PE HDF5、dense maps、原始/feature 缓存及大型联合预测数组；它们保留在源目录，路径和哈希可追溯。复算需要 manifest 中的原项目输入，本包不是脱离数据的独立安装镜像。',
        '新确认运行时发现系统 cryptography Python/binary 不一致，使用相同版本 46.0.6 的隔离 wheel 修复 import；没有修改全局环境、物理配置或门槛。first adapter 小批量与原批量 GEMM 的数值差异用完整原 validation 批量核对，六项完全一致；原失败日志保留。','',
        '## 科学依据与限制','',
        '低频 inspiral 的相位演化提供 chirp mass 信息，但质量比/自旋存在退化，因此比较联合预测而不是只要求一个点估计接近：[Cutler & Flanagan](https://arxiv.org/abs/gr-qc/9402014)。',
        '温度校准、概率预测和经验密度比是机器学习 ranking 组件，不是完整引力波 PE；本轮是否有效由保留的模拟及真实开发审计决定，不能借理论引用保证性能：[Guo et al.](https://arxiv.org/abs/1706.04599)。',
        'BAYESTAR 是快速天空定位方法；本项目使用其条件测量模拟接口，不能把本项目的注入模式自动等同于原论文完整观测处理：[Singer & Price](https://arxiv.org/abs/1508.03634)。',
        'BC 衡量概率质量相似；完整共享源透镜假设仍需带先验、selection 的联合分析。本轮不以 BC 或 retrieval score 替代该分析：[Lo & Magana Hernandez](https://arxiv.org/abs/2104.09339)。',
        '不能保证继续调整总能得到未见真实目录上的同样提升。当前交付的“达到目标”仅指明确列出的开发预算改善及独立模拟保护均通过；仍须作者审核并在新的确认性协议下决定是否建立正式版本。','',t.STATUS]
    (root/'reports/FINAL_UNIFIED_METHOD_AND_RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'README_CN.md').write_text(f'# {CODE}\n\n详见 reports/FINAL_UNIFIED_METHOD_AND_RESULTS_CN.md。\n\n服务器：connect.westd.seetacloud.com，SSH 端口 32328，用户 root。登录命令：`ssh -p 32328 root@connect.westd.seetacloud.com`。包内不含登录凭据。\n\n结果目录：`{root}`。\n\n真实目录结果为适应性开发；新注入只用于独立保护检验。官方重合不是透镜真值；未经作者审核，不替换历史结果或论文。\n\n{t.STATUS}\n',encoding='utf-8')
    make_plots(root,b,means)
    dev.json_write(root/'contracts/REPORT_COMPLETE.json',{'report': 'reports/FINAL_UNIFIED_METHOD_AND_RESULTS_CN.md','fresh_guard_pass':bool(guards['pass'].all()),'source_noise_independence_pass':independence['source_noise_independence_pass'],'goal_achieved':False,'archive_checks_pending':True})


def make_plots(root,b,means):
    font=plotting.configure_plot();plt=plotting.plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for r,dep in enumerate(t.DEPS):
        for c,(metric,title) in enumerate((('BC_mc_ge_0p5','Mc posterior BC >= 0.5'),('official_frontend','Official 1% frontend overlap'),('median_BC_mc','Median Mc posterior BC'))):
            ax=axes[r,c]
            for k,(name,label,color) in enumerate((('OMC','Retained OMC','#66727B'),(CODE,'Common PATH875','#17827B'))):
                f=b[(b.deployment==dep)&(b.config==name)&b.budget.isin([10,20])].sort_values('budget')
                ax.bar(np.arange(2)+(k-.5)*.32,f[metric],width=.30,label=label,color=color)
                for x,y in zip(np.arange(2)+(k-.5)*.32,f[metric]):ax.text(x,y+.008 if c==2 else y+.15,f'{y:.3f}' if c==2 else str(int(y)),ha='center',fontsize=8)
            ax.set_xticks([0,1],['Top-10','Top-20']);ax.set_title(dep+' | '+title,fontsize=10)
            ax.set_ylim(0,1.07 if c==2 else 22);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper left')
    plotting.savefig(fig,root,'fig_real_development_top10_top20')
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for r,dep in enumerate(t.DEPS):
        for c,(metric,label) in enumerate((('macro_r_at_10','R@10'),('average_precision','Pair AUPRC'),('false_at_recall_0p9','False pairs at 90% recall'))):
            ax=axes[r,c]
            for k,mode in enumerate(('waveform','fusion')):
                f=means[(means.deployment==dep)&(means['mode']==mode)]
                a=f[f.configuration=='OMC'].set_index('model_seed').sort_index();z=f[f.configuration=='PATH25'].set_index('model_seed').loc[a.index]
                for j,(old,new) in enumerate(zip(a[metric],z[metric])):
                    xx=k*3+np.array([0,1])+((j-1)*.09)
                    ax.plot(xx,[old,new],'-o',lw=.8,ms=4,color=plt.get_cmap('tab10')(j),alpha=.8)
            ax.set_xticks([0,1,3,4],['Old WF','New WF','Old fusion','New fusion'],rotation=20,fontsize=8)
            ax.set_title(dep+' | '+label,fontsize=10);ax.spines[['top','right']].set_visible(False)
    plotting.savefig(fig,root,'fig_fresh_independent_guardrails')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for r,dep in enumerate(t.DEPS):
        for c,(name,label) in enumerate((('OMC','Retained OMC'),(CODE,'Common PATH875'))):
            f=pd.read_parquet(root/f'development/evaluation/{name}/{dep}/consensus_fusion_all_pairs.parquet')
            ax=axes[r,c]
            ax.scatter(f.waveform_score_mean,f.pe_mc_bhattacharyya_coefficient,s=5,alpha=.17,color='#687680',rasterized=True)
            top=f.head(10)
            ax.scatter(top.waveform_score_mean,top.pe_mc_bhattacharyya_coefficient,s=30,facecolors='none',edgecolors='#B54F60',linewidths=1,label='Fusion Top-10')
            ax.axhline(.5,color='black',lw=.7,ls='--');ax.axhline(.1,color='#B54F60',lw=.7,ls=':')
            ax.set_xlabel('Mean waveform ranking score');ax.set_ylabel('Public PE Mc posterior BC')
            ax.set_title(dep+' | '+label);ax.set_ylim(-.02,1.04);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=8,loc='lower left')
    plotting.savefig(fig,root,'fig_waveform_vs_public_PE')
    dev.json_write(root/'figures/PLOT_PROVENANCE.json',{'font':font,'real_plot':'adaptive development descriptive counts,not independent significance','fresh_points':'one trainingseed averaged over three newcatalogs;paired same events','SD':'not nine independent trainingruns'})


def seal(root):
    contract=json.loads((root/'contracts/DELIVERY_CONTRACT.json').read_text())
    development,confirmation=Path(contract['development_root']),Path(contract['confirmation_root'])
    frozen_checks(development,confirmation)
    if not (root/'contracts/REPORT_COMPLETE.json').exists():raise RuntimeError('Report incomplete')
    archive=P/'packages'/f'{root.name}_deliverables.tar.gz'
    if archive.exists():raise RuntimeError('Do not overwrite package')
    required=['reports/FINAL_UNIFIED_METHOD_AND_RESULTS_CN.md','tables/TOP_B_PE_OFFICIAL_COMPARISON.csv',
        'tables/FRESH_SUMMARY_BY_TRAINING_SEED.csv','tables/FINAL_EFFECTIVE_WEIGHTS.csv',
        'figures/fig_real_development_top10_top20.pdf','figures/fig_fresh_independent_guardrails.pdf',
        'independent_confirmation/tables/FRESH_PAIRED_METRICS_AND_CI.csv',
        'independent_confirmation/confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json']
    if any(not (root/p).exists() for p in required):raise RuntimeError('Missing required artifact')
    dev.json_write(root/'contracts/FINAL_OUTCOME.json',{'code':CODE,'real_development_goal_met':True,'independent_injection_guardrails_pass':True,
        'scientific_work_status':'DEVELOPMENT_OBJECTIVE_MET_WITH_LIMITATIONS','adoption_authorized':False,
        'real_blind_confirmation':False,'no_claim_of_lensing_detection':True,'status':t.STATUS,
        'historical_protected_hashes_unchanged':True,'package_verification_is_required_for_delivery':True})
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
        if p.name in ('OUTPUT_SHA256.csv','SHA256SUMS.txt','DELIVERY_CHECK.json'):continue
        if p.suffix in ('.py','.md','.json','.csv','.log','.txt') and p.stat().st_size<20*2**20:
            if re.search(r'(?m)^-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----\s*$',p.read_text(errors='replace')):raise RuntimeError('Secret material detected')
        files.append(p)
    manifest=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':dev.sha(p)} for p in files]
    dev.csv_write(root/'manifest/OUTPUT_SHA256.csv',pd.DataFrame(manifest))
    (root/'manifest/SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in manifest),encoding='utf-8')
    files += [root/'manifest/OUTPUT_SHA256.csv',root/'manifest/SHA256SUMS.txt']
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for p in files:tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    expected={str(Path(root.name)/r['path']):r['sha256'] for r in manifest};checked=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.name not in expected:continue
            digest=hashlib.sha256();stream=tar.extractfile(member)
            for chunk in iter(lambda:stream.read(4*2**20),b''):digest.update(chunk)
            if digest.hexdigest()!=expected[member.name]:raise RuntimeError('Archive hash mismatch')
            checked+=1
    if checked!=len(expected):raise RuntimeError('Incomplete archive')
    digest=dev.sha(archive);archive.with_suffix(archive.suffix+'.sha256').write_text(digest+'  '+archive.name+'\n')
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,
        'members':len(files),'internal_verified_files':checked,'required_artifacts':len(required),'all_checks_pass':True,'status':t.STATUS,
        'bounded_development_task_complete':True,'adoption_authorized':False,
        'scope':'Real development improvement plus independent simulation guardrails,not blind real confirmation.'}
    dev.json_write(root/'manifest/DELIVERY_CHECK.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--development',type=Path);parser.add_argument('--confirmation',type=Path)
    parser.add_argument('--stage',choices=['prepare','report','seal'],required=True)
    a=parser.parse_args()
    if a.stage=='prepare':prepare(a.root,a.development,a.confirmation)
    else:globals()[a.stage](a.root)
