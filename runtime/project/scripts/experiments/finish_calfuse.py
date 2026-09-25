#!/usr/bin/env python3
"""Uncertainty, input-domain audit, all-outcome reporting and verified packaging."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import calfuse as c
import mcwf_summarize_20260905 as boot


def uncertainty(root):
    destination=root/'tables/SYSTEM_BOOTSTRAP.csv'
    if destination.exists():raise RuntimeError('Bootstrap already complete')
    rows=[]
    for conf in c.selections(root):
        dep,seed,method=conf['deployment'],conf['seed'],conf['method']
        f=pd.read_parquet(root/f'results/{method}/{dep}/seed_{seed}/test_pairs.parquet')
        plan=pd.read_csv(root/f'audit/{dep}_{seed}_test_event_plan.csv').sort_values('idx')
        base=c.read(dep,seed,'test')
        for mode,score,reference in (
            ('fusion',f.final_score.to_numpy(),c.channels(base,base.waveform_score)@c.frozen_weights(dep,seed)),
            ('waveform_only',f.waveform_score.to_numpy(),base.waveform_score.to_numpy())):
            rr,rank=boot.ranks_bootstrap(f,score,reference,seed,10000)
            if mode=='fusion':rr.update(boot.pair_bootstrap(f,score,reference,plan,seed,1000))
            rows.append({'deployment':dep,'seed':seed,'method':method,'mode':mode,
                         'rank_repeats':10000,'pair_repeats':1000 if mode=='fusion' else 0,**rr})
            path=root/f'audit/query_ranks/{method}_{dep}_{seed}_{mode}.parquet'
            path.parent.mkdir(exist_ok=True,parents=True);rank.to_parquet(path,index=False)
        print('BOOTSTRAP',dep,seed,method,flush=True)
    c.csv(destination,rows)


def weight_stability(root):
    rows=[]
    choices=c.selections(root)
    for dep in c.DEPS:
        for seed in c.SEEDS:
            full=c.read(dep,seed,'validation')
            p=pd.read_csv(root/f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv')
            for family,cohort in (('OMC','full-validation'),('AFFINE','tune'),('JOINT','tune')):
                frame=full if cohort=='full-validation' else c.subset(full,p[p.calibration_fold==1].idx)
                spec=json.loads((root/f'configs/{dep}_{seed}_{family}_CALIBRATION.json').read_text())
                z,_,_=c.apply(frame,spec)
                design=c.channels(frame,z);grid=c.weight_grid(dep,seed)
                old=c.frozen_weights(dep,seed)
                bscore=c.channels(frame,frame.waveform_score)@old
                used=sorted(set(frame.idx_i)|set(frame.idx_j))
                plan=p.set_index('idx').loc[used]
                groups=plan.system_id.unique()
                rng=np.random.default_rng(202609082+seed)
                for rep in range(100):
                    count=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))
                    lookup=dict(zip(groups,count))
                    mul=np.zeros(int(frame.event_count.iloc[0]))
                    mul[plan.index]=plan.system_id.map(lookup)
                    baseline=c.fast_metrics(frame,bscore,mul)
                    table=[]
                    for k,w in enumerate(grid):
                        m=c.fast_metrics(frame,design@w,mul)
                        table.append({'weights':w.tolist(),'guard_pass':c.guard(m,baseline),**m})
                    for policy in ('VF50','VR10'):
                        selected=c.select_grid(table,policy,old)
                        weight=old/old.sum() if selected is None else selected['weights']
                        method=f'OMC-FULL-{policy}' if family=='OMC' else f'{family}-{policy}'
                        ref=next(x for x in choices if x['deployment']==dep and x['seed']==seed and x['method']==method)
                        rw=np.array(ref['weights']);rw/=rw.sum()
                        rows.append({'deployment':dep,'seed':seed,'family':family,'method':method,
                            'bootstrap':rep,'fallback':selected is None,
                            **dict(zip(('w_waveform','w_time','w_sky'),weight)),
                            'L1_distance_from_selected':float(np.sum(abs(np.asarray(weight)-rw))),
                            'identical_selected_weight':bool(np.max(abs(np.asarray(weight)-rw))<1e-10)})
                print('WEIGHT_BOOTSTRAP',dep,seed,family,flush=True)
    c.csv(root/'tables/WEIGHT_BOOTSTRAP_PER_DRAW.csv',rows)
    f=pd.DataFrame(rows)
    summary=f.groupby(['deployment','seed','method']).agg(
        selection_frequency=('identical_selected_weight','mean'),
        mean_L1_distance=('L1_distance_from_selected','mean'),fallback_rate=('fallback','mean'),
        w_waveform_median=('w_waveform','median'),w_time_median=('w_time','median'),w_sky_median=('w_sky','median')).reset_index()
    c.csv(root/'tables/WEIGHT_STABILITY.csv',summary)


def gates(root):
    b=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    b=b[(b.seed.astype(str)=='consensus')&(b.method=='fusion')&b.budget.isin([10,20])]
    m=pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv')
    rows=[]
    for dep in c.DEPS:
        base=b[(b.deployment==dep)&(b.config=='OMC-FIXED')].set_index('budget')
        for method in b.config.unique():
            cur=b[(b.deployment==dep)&(b.config==method)].set_index('budget')
            pe_ok=bool((cur.catastrophic_mc<=base.catastrophic_mc).all())
            for col in ('BC_mc_ge_0p5','Dmax_le_3','median_BC_mc'):
                pe_ok &= bool((cur[col]>=base[col]-1e-12).all())
            pe_improve=pe_ok and (cur.BC_mc_ge_0p5.sum()>base.BC_mc_ge_0p5.sum() or cur.median_BC_mc.sum()>base.median_BC_mc.sum()+1e-10)
            public_ok=all((cur[col]>=base[col]).all() for col in ('official_frontend','official_hanabi'))
            public_improve=public_ok and all(cur[col].sum()>base[col].sum() for col in ('official_frontend','official_hanabi'))
            selected=m[(m.deployment==dep)&(m.method==method)&(m.split=='test')]
            sim=bool(selected.guard_pass.all())
            reasons=[]
            if not pe_improve:reasons.append('PE_target_not_met')
            if not public_improve:reasons.append('official_target_not_met')
            if not sim:reasons.append('reused_test_guard_failed')
            rows.append({'deployment':dep,'method':method,'PE_noninferior':pe_ok,
                'PE_improved':pe_improve,'official_noninferior':bool(public_ok),
                'official_frontend_and_Hanabi_improved':bool(public_improve),
                'all_seed_waveform_and_fusion_test_guard':sim,
                'target_pass':bool(pe_improve and public_improve and sim),'failure_reasons':';'.join(reasons)})
    table=pd.DataFrame(rows)
    c.csv(root/'tables/TARGET_GATES.csv',table)
    both=table.groupby('method').target_pass.all()
    c.dump(root/'contracts/FINAL_DECISION.json',{'status':c.STATUS,
        'joint_target_achieved':bool(both.any()),'joint_passing_methods':both[both].index.tolist(),
        'no_adoption':True,'no_overwrite':True,'real_reranking_selected_after_configs_frozen':True,
        'independent_confirmation_performed':False})


def domain(root):
    sky_contract=c.dev.BAY/'contracts/ANALYSIS_CONTRACT.json'
    sky_script=c.dev.BAY/'scripts/bayestar_injection_sky_full_experiment.py'
    dest=root/'audit/input_provenance';dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(sky_contract,dest/'FROZEN_BAYESTAR_ANALYSIS_CONTRACT.json')
    shutil.copy2(sky_script,dest/sky_script.name)
    manifest=pd.read_csv(c.dev.BAY/'results/bayestar_event_map_manifest.csv')
    clock_rows,map_rows,pair_rows=[] ,[],[]
    for dep in c.DEPS:
        for seed in c.SEEDS:
            for split in ('validation','test'):
                p=pd.read_csv(root/f'audit/{dep}_{seed}_{split}_event_plan.csv')
                if dep=='gwtc3':
                    in_o3=((p.gps_obs>=1238166018)&(p.gps_obs<1253977218))|((p.gps_obs>=1256655618)&(p.gps_obs<1269363618))
                    clock_rows.append({'deployment':dep,'seed':seed,'split':split,
                        'n_events':len(p),'outside_O3_calendar':int((~in_o3).sum()),
                        'fraction_outside_O3':float((~in_o3).mean()),'min_gps':p.gps_obs.min(),'max_gps':p.gps_obs.max(),
                        'reference':'https://gwosc.org/O3/O3a/ ; https://gwosc.org/O3/O3b/',
                        'meaning':'calendar audit only,not H1L1 live-time test; synthetic trigger times,no claim on noise-file observing dates'})
                selected=manifest[(manifest.deployment==dep)&(manifest.model_seed==seed)&(manifest.split==split)&manifest.idx.isin(p.old_idx)]
                assert len(selected)==len(p)
                map_rows.append({'deployment':dep,'seed':seed,'split':split,'n_maps':len(selected),
                    'calibrated_HPD90_fraction':float((selected.truth_credible_level_calibrated<=.9).mean()),
                    'raw_A90_median':float(selected.area90_deg2.median()),
                    'A90_note':'manifest area90 is raw-map area; not recomputed tempered area',
                    'target_SNR_median':float(selected.target_network_snr.median()),
                    'realized_SNR_median':float(selected.realized_network_snr.median()),
                    'input':'Gaussian matched-filter measurement realization with assigned empirical PSD and known source parameters',
                    'actual_injection_strain_matched_filtered':False})
                if split=='test':
                    f=c.read(dep,seed,split);real=c.read(dep,seed,'real')
                    for kind,mask in (('companion',f.is_true_pair.to_numpy(bool)),('null',~f.is_true_pair.to_numpy(bool))):
                        reference=f.loc[mask,'sky_raw_log_bf'].to_numpy(float)
                        for q in (90,95,99,100):
                            threshold=float(np.percentile(reference,q))
                            pair_rows.append({'deployment':dep,'seed':seed,'reference_population':kind,'quantile':q,
                                'threshold':threshold,'real_all_pairs':len(real),
                                'real_pairs_above':int((real.sky_raw_log_bf>threshold).sum()),
                                'real_fraction_above':float((real.sky_raw_log_bf>threshold).mean()),
                                'real_companion_labels_known':False})
    c.csv(root/'audit/O3_TIME_CALENDAR_SCOPE.csv',clock_rows)
    c.csv(root/'audit/RETAINED_BAYESTAR_MAP_AUDIT.csv',map_rows)
    c.csv(root/'audit/REAL_SKY_TAIL_VS_INJECTION.csv',pair_rows)
    sources=[sky_contract,sky_script,c.dev.BAY/'results/bayestar_event_map_manifest.csv']
    c.csv(root/'manifest/PROVENANCE_INPUTS.csv',[{'path':str(p),'sha256':c.dev.sha(p)} for p in sources])
    c.dump(root/'audit/SKY_TIME_AUDIT_FINDINGS.json',{
        'known_intrinsics_injection':True,'exact_nonGaussian_injection_strain_replayed_for_sky':False,
        'public_real_PE_not_same_measurement_pipeline':True,
        'temperature_calibration_not_proof_of_fullPE_equivalence':True,
        'O3_calendar_scope_mismatch_detected':any(r['outside_O3_calendar']>0 for r in clock_rows),
        'time_lookup_or_sky_scores_changed':False,
        'new_map_PE_performed':False,
        'required_separate_next_audit':'recovered matched-filter trigger pilot from identical injected strain; O3-only exposure lookup with system-disjoint prior; preregister new version before any reranking',
        'no_claim':'does not demonstrate that fixing either domain gap will increase official overlaps'})


def figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    b=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    b=b[(b.seed.astype(str)=='consensus')&(b.method=='fusion')]
    methods=['OMC-FIXED','OMC-FULL-VF50','OMC-FULL-VR10','AFFINE-FIXED','AFFINE-VF50','AFFINE-VR10','JOINT-FIXED','JOINT-VF50','JOINT-VR10','OMC-VF50','OMC-VR10']
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,dep in enumerate(c.DEPS):
        for ax,key,title in zip(axes[row],('BC_mc_ge_0p5','official_frontend','official_hanabi'),('Mc BC >= 0.5','Official frontend < 1%','Published Hanabi overlap')):
            for budget,color in ((10,'#27748e'),(20,'#b74b69')):
                vals=b[(b.deployment==dep)&(b.budget==budget)].set_index('config').loc[methods,key]
                ax.plot(np.arange(len(methods)),vals,'o-',lw=.8,ms=4,color=color,label=f'Top-{budget}')
            ax.set_xticks(range(len(methods)),methods,rotation=65,ha='right',fontsize=7)
            ax.set_title(f'{dep}: {title}');ax.set_ylabel('Number of pairs');ax.grid(axis='y',alpha=.2)
            ax.legend(frameon=False,fontsize=8)
    fig.savefig(root/'figures/fig_PE_official_comparison.png',dpi=180)
    fig.savefig(root/'figures/fig_PE_official_comparison.pdf');plt.close(fig)
    m=pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv')
    m=m[(m.split=='test')&(m['mode']=='fusion')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,dep in enumerate(c.DEPS):
        for ax,key in zip(axes[row],('macro_r_at_10','average_precision')):
            for k,seed in enumerate(c.SEEDS):
                v=m[(m.deployment==dep)&(m.seed==seed)].set_index('method').loc[methods,key]
                ax.scatter(np.arange(len(methods))+(k-1)*.15,v,s=22,label=str(seed))
            ax.set_xticks(range(len(methods)),methods,rotation=65,ha='right',fontsize=7)
            ax.set_title(dep);ax.set_ylabel(key);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=7)
    fig.savefig(root/'figures/fig_retrieval_individual_seeds.png',dpi=180)
    fig.savefig(root/'figures/fig_retrieval_individual_seeds.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,dep in zip(axes,c.DEPS):
        f=c.read(dep,c.SEEDS[0],'test');r=c.read(dep,c.SEEDS[0],'real')
        for name,x,color in (('Injection companion',f.loc[f.is_true_pair==1,'sky_raw_log_bf'],'#318063'),
            ('Injection null',f.loc[f.is_true_pair==0,'sky_raw_log_bf'],'#707070'),
            ('Real all pairs (unknown labels)',r.sky_raw_log_bf,'#b34c69')):
            x=np.sort(x.to_numpy(float));ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=name,color=color)
        ax.set_xscale('symlog',linthresh=2);ax.set_xlabel('Raw log sky Bayes factor');ax.set_ylabel('Empirical CDF');ax.set_title(dep);ax.legend(frameon=False,fontsize=7)
    fig.savefig(root/'figures/fig_sky_domain_CDF.png',dpi=180)
    fig.savefig(root/'figures/fig_sky_domain_CDF.pdf');plt.close(fig)


def table_markdown(frame):
    return frame.to_markdown(index=False,floatfmt='.4f')


def report(root):
    decision=json.loads((root/'contracts/FINAL_DECISION.json').read_text())
    b=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    b=b[(b.seed.astype(str)=='consensus')&(b.method=='fusion')&b.budget.isin([10,20])]
    m=pd.read_csv(root/'tables/RETRIEVAL_SUMMARY.csv');m=m[(m.split=='test')&(m['mode']=='fusion')]
    rows=[]
    for r in m.itertuples():
        q=b[(b.deployment==r.deployment)&(b.config==r.method)].set_index('budget')
        rows.append({'run':r.deployment,'method':r.method,'R10':f'{r.macro_r_at_10_mean:.4f} +/- {r.macro_r_at_10_std:.4f}',
            'AP':round(r.average_precision_mean,4),'F50':round(r.false_at_recall_0p5_mean,1),'F90':round(r.false_at_recall_0p9_mean,1),
            'Mc Top10/20':f'{q.loc[10,"BC_mc_ge_0p5"]}/{q.loc[20,"BC_mc_ge_0p5"]}',
            'catastrophic Top10/20':f'{q.loc[10,"catastrophic_mc"]}/{q.loc[20,"catastrophic_mc"]}',
            'official Top10/20':f'{q.loc[10,"official_frontend"]}/{q.loc[20,"official_frontend"]}',
            'Hanabi Top10/20':f'{q.loc[10,"official_hanabi"]}/{q.loc[20,"official_hanabi"]}'})
    summary=pd.DataFrame(rows);c.csv(root/'tables/COMPACT_RESULT_SUMMARY.csv',summary)
    cal=pd.read_csv(root/'audit/O3_TIME_CALENDAR_SCOPE.csv')
    groups=pd.read_csv(root/'audit/DATA_SPLIT_AUDIT.csv')
    lines=['# MCWF-CALFUSE-01：完整波形分数与融合校准探索','',f'状态：{c.STATUS}','',
        '## 结论','',
        ('存在满足本轮探索性目标的配置；仍需新的独立确认。' if decision['joint_target_achieved'] else
         '11种预定义对照、两个运行期和三个冻结部署seed均完成。未找到同时改善O3/O4a的PE/Mc、官方前端与Hanabi重合并通过注入保护条件的方案。本轮不升级、不替换历史结果。'),
        '这次没有新训练encoder，没有修改时间或天空的原始分数；修改的是波形分数的整体校准和外层融合权重。不能把某个Top10改善而Top20退化的配置宣布为成功。',
        '', '## 为什么做本轮','',
        '旧OMC波形分数叠加了原waveform、FRT增量和OMC质量增量，外层仍使用历史C-fixed权重。本轮检验旧权重是否仍合适、独立相加的波形子证据能否用一次联合校准替代。并非已经认定时间与天空最优。',
        '', '## 方法与物理边界','',
        '- OMC：当前波形分数完全不变。',
        '- AFFINE：对完整OMC分数拟合非负斜率的balanced logistic校准。',
        '- JOINT：以原波形分数、FRT增量、OMC增量为三个输入，拟合非负系数的正则化logistic。该输出整体替换波形通道，不再把三个预测logit当作独立Bayes factor相乘。',
        '- FIXED沿用逐seed原C-fixed权重；VF50优先F50/F90；VR10优先R10/R1。两个运行期完全使用同一规则，但各自在模拟validation中选参数。',
        '- 网格为非负0.05 simplex，加上原权重归一化点并去重；允许零权重。未添加query-row标准化。',
        '- 与原始波形分数同尺度的资格不是自动成立的：新输出是平衡模拟标签下的预测logit，不是实际透镜后验赔率，也不是proper lensing BF。',
        '',r'公式：$Z_W^{new}=b+\sum_k a_k(x_k-\mu_k)/\sigma_k,\ a_k\ge0$；支持域内截断到 $\pm\log(n_{L,fit}+1)$。特征超出fit范围时整条更新回退到旧OMC。',
        r'总分仍为 $S=w_W Z_W+w_T Z_T+w_S Z_S$。$Z_T$、$Z_S$逐元素不变；权重改变会改变它们的贡献，不应误写为“所有通道贡献都没变化”。',
        '', '## 数据隔离和选择','',
        '父噪声bank按预定义哈希划为fit/tune。跨两侧的整个源系统删除，不允许一个源或父噪声bank同时出现在两侧。每seed只有14--22个独立真系统用于拟合，不能用数千相关null pairs声称大样本校准；因此只做低维正则化探索。OMC-FULL两个额外对照不拟合校准器，使用完整validation选权。',
        table_markdown(groups[['deployment','seed','fit_events','tune_events','fit_positive_systems','tune_positive_systems']]),
        '', '所有本轮配置只用模拟fit/tune标签选择，在读取新比较test分数和本轮真实PE联表前写入带hash的冻结文件。旧模型、旧test和真实目录曾经被适应性使用，本轮不能称为新的盲测或确认。真实PE/官方表未参与本轮选参，但作为目标完成与否的外部开发性审计。',
        '', '## 全部结果','',table_markdown(summary),'',
        'R10为既有SIS/PM legacy family宏平均；O3/O4a family名字不应据此重新解释为解析SIS/PM人口。SD来自三个冻结部署/目录seed，不是本轮独立重训模型方差。',
        '', '## 天空输入审计发现','',
        '现有BAYESTAR注入地图不再使用旋转公开模板，但也没有从完全相同的非高斯注入应变重新恢复触发量。原始合同和代码明确：使用已知源参数、事件GPS、分配的真实off-source PSD及target SNR，通过ligo.skymap的高斯匹配滤波测量模拟生成触发量。真实目录则使用公开PE后验图。二者共享Nside512和公式，不等于共享数据误差模型。',
        'BAYESTAR本身是合法快速定位算法；问题是当前实现的输入条件和真实数据不同。已有HPD90温度校准不能单独证明非高斯噪声、模板误差和真实PE多维边缘化均已匹配。本轮只审计，不更换地图，不把真实高分强行压回注入范围。',
        '', '## 时间观测期审计发现','',
        '当前名为gwtc3的注入计划中存在O3正式观测期之外的合成事件GPS，而真实比较范围已变成官方O3 strict62。下表仅核对合成时间，不把该结果误解为实际off-source噪声文件也必然不属于O3。当前一维时间lookup的全部独立系统provenance尚未在本轮重新证明。',
        table_markdown(cal[['seed','split','n_events','outside_O3_calendar','fraction_outside_O3']]),
        '这意味着“时间方案无需再审计”的判断不成立。应另立O3-only exposure的成对敏感性版本，而不是在已经看到真实排名后偷偷改当前lookup。即便纠正范围，也不能承诺官方重合必然增加。',
        '', '## 不确定度和验证','',
        'SYSTEM_BOOTSTRAP.csv使用10000次系统级、legacy-family分层的R1/R10 bootstrap，双向query共同抽样；融合pair指标另做1000次source-block加权bootstrap。WEIGHT_STABILITY.csv用100次validation source bootstrap检查选权波动，校准器固定不重拟合。这些区间不校正历史适应性选择。真实pair共享事件，只提供描述性PE相关系数，不给iid-pair显著性。',
        '保留原optimistic tie检索定义，同时输出pessimistic R10，防止校准截断引起大量并列时只看乐观召回。原有FPR=1e-5统计有最少1假对的离散约定，本轮不将其作为准确的1e-5测量报告。',
        '', '## PE和官方阶段交付','',
        '每个方案和运行期均提供逐seed与consensus完整pair Parquet、Top100 UTF-8-BOM CSV，包含Mc/q/chi_eff/表观距离PE、官方PO/ML或PO/Phazap FPP与可解析的GOLUM/Hanabi阶段。未解析字段不伪造结论，Hanabi表重合不等于偏好透镜。本轮不运行Hanabi、不重算公开PE。',
        '', '## 下一步判断','',
        '没有证据支持仅通过重校准就已完成双运行期目标。优先补齐实际注入应变的recovered-trigger天空pilot和O3-only时间背景匹配，再决定是否重做独立源/噪声校准数据。不要继续把越来越多质量项相加，或按真实官方名单直接调排名。',
        '', '## 参考依据','',
        '- [scikit-learn calibration文档](https://scikit-learn.org/stable/modules/calibration.html)：校准数据独立和小样本校准限制。',
        '- [Cranmer等似然比学习](https://arxiv.org/abs/1506.02169)：分类器校准估计密度比的原则，不保证本项目近似正确。',
        '- [Singer与Price BAYESTAR](https://arxiv.org/abs/1508.03634)：从搜索测量构造快速条件天空推断。',
        '- [GWOSC O3a](https://gwosc.org/O3/O3a/)和[O3b](https://gwosc.org/O3/O3b/)：观测期边界。',
        '正则化、score cap、网格和非劣容差为本轮明确冻结的工程/探索选择，不冒充文献提供的普适阈值。',
        '', '## 复现与路径','',
        '`scripts/run_calfuse.py --root <new_directory>`依次执行tests/init/calibrate/evaluate/real；随后`finish_calfuse.py`各phase执行uncertainty/weight_stability/gates/domain/report/verify/package。旧目录必须只读；原始strain和模型依赖需仍可访问。',
        '完整方案、冻结配置、失败Gate、全部seed指标、PE官方表、图和逐文件hash均保留。包不包含原始strain、私钥或密码。']
    (root/'reports/MCWF_CALFUSE_01_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # All top tables remain separate; no selection of only visually attractive candidates.
    top=[]
    for path in sorted((root/'results').glob('*/*/consensus_fusion_top100.csv')):
        f=pd.read_csv(path);f.insert(0,'configuration',path.parent.parent.name);f.insert(0,'deployment',path.parent.name)
        top.append(f)
    c.csv(root/'tables/ALL_REAL_CONSENSUS_TOP100.csv',pd.concat(top,ignore_index=True))
    content=['# 逐方案真实Top10（只读审计）','', '原始全字段见ALL_REAL_CONSENSUS_TOP100.csv；所有方案均保留，不按PE择优省略。','']
    for f in top:
        content += [f'## {f.deployment.iloc[0]} {f.configuration.iloc[0]}','',table_markdown(f.head(10)[[
            'consensus_rank','event_i','event_j','final_score_mean','waveform_contribution_mean','time_contribution_mean','sky_contribution_mean',
            'pe_mc_bhattacharyya_coefficient','pe_dmax_intrinsic']]),'']
    (root/'reports/ALL_REAL_TOP10_CN.md').write_text('\n'.join(content),encoding='utf-8')
    figures(root)


def verify(root):
    c.selections(root)
    hist=[]
    for name in ('PROTECTED_INPUTS.csv','PROVENANCE_INPUTS.csv'):
        for r in pd.read_csv(root/'manifest'/name).itertuples():
            actual=c.dev.sha(Path(r.path));assert actual==r.sha256,r.path
            hist.append({'path':r.path,'sha256':actual,'unchanged':True})
    c.csv(root/'manifest/HISTORICAL_HASH_RECHECK.csv',pd.DataFrame(hist).drop_duplicates('path'))
    scores=[]
    for conf in c.selections(root):
        dep,seed,method=conf['deployment'],conf['seed'],conf['method']
        for split in ('validation','test'):
            src=c.read(dep,seed,split)
            out=pd.read_parquet(root/f'results/{method}/{dep}/seed_{seed}/{split}_pairs.parquet')
            for col in ('idx_i','idx_j','is_true_pair','time_score','sky_raw_log_bf'):
                assert np.array_equal(src[col],out[col])
            spec=conf['calibration']
            if spec['family']=='OMC':expected=src.waveform_score.to_numpy(float)
            else:
                x=c.features(src,spec['family']);active=np.asarray(spec['coef'])>1e-8
                val=spec['intercept']+np.sum(((x-spec['mu'])/spec['sd'])*spec['coef'],axis=1)
                outside=(((x<np.asarray(spec['minimum'])-1e-7)|(x>np.asarray(spec['maximum'])+1e-7))&active).any(1)
                expected=np.clip(val,-spec['score_cap'],spec['score_cap'])
                expected[outside]=src.waveform_score.to_numpy(float)[outside]
            error=float(np.max(abs(expected-out.waveform_score)))
            total=conf['weights'][0]*expected+conf['weights'][1]*src.time_score.to_numpy(float)+conf['weights'][2]*src.sky_raw_log_bf.to_numpy(float)
            total_error=float(np.max(abs(total-out.final_score.to_numpy())))
            assert error<1e-10 and total_error<1e-10
            reference=c.dev.BASE.full_metrics(src,total)
            own=c.fast_metrics(out,out.final_score.to_numpy(float))
            for k in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                assert abs(reference[k]-own[k])<1e-10
            scores.append({'deployment':dep,'seed':seed,'method':method,'split':split,
                'waveform_max_error':error,'fusion_max_error':total_error,'time_sky_labels_exact':True,'historical_metric_replay':True})
    c.csv(root/'audit/INDEPENDENT_ALL_SCORE_REPLAY.csv',scores)
    # Compare baseline budgets with the preceding delivered report, not just this script's own baseline.
    old=pd.read_csv(c.NOISE/'contracts/BASELINE_BUDGETS.csv')
    new=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    new=new[(new.config=='OMC-FIXED')&(new.seed.astype(str)=='consensus')]
    old=old[old.seed.astype(str)=='consensus']
    for row in new.itertuples():
        name='C_fixed' if row.method=='fusion' else 'waveform_only'
        ref=old[(old.deployment==row.deployment)&(old.method==name)&(old.budget==row.budget)].iloc[0]
        for col in ('catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3','median_BC_mc','official_frontend','official_hanabi'):
            assert abs(getattr(row,col)-ref[col])<1e-10,(row.deployment,row.budget,col)
    c.dump(root/'audit/FINAL_VERIFICATION.json',{'all_pass':True,'score_replays':len(scores),
        'historical_files_unchanged':len(pd.DataFrame(hist).drop_duplicates('path')),
        'baseline_PE_official_budgets_reproduced':True,'configs_hash_unchanged':True,
        'time_sky_unchanged':True,'no_new_encoder_training':True,
        'official_labels_only_downstream_audit':True})


def package(root):
    assert json.loads((root/'audit/FINAL_VERIFICATION.json').read_text())['all_pass']
    for p in Path(__file__).parent.glob('*calfuse*.py'):
        dest=root/'scripts'/p.name
        if not dest.exists():shutil.copy2(p,dest)
    for module in (boot,):
        shutil.copy2(module.__file__,root/'scripts'/Path(module.__file__).name)
    # Only allow known text/data outputs; check for common private-key and credential artifacts.
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in files:
        assert p.suffix not in ('.pem','.key'),p
        if p.suffix in ('.py','.md','.json','.log','.txt','.csv'):
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',text):raise RuntimeError('Secret artifact: '+str(p))
    manifest=[{'path':str(p.relative_to(root)),'sha256':c.dev.sha(p),'bytes':p.stat().st_size}
              for p in files if p.name not in ('OUTPUT_SHA256.csv','SHA256SUMS.txt')]
    c.csv(root/'manifest/OUTPUT_SHA256.csv',manifest)
    (root/'manifest/SHA256SUMS.txt').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in manifest))
    output=c.PROJECT/'packages'/f'{root.name}_deliverables.tar.gz'
    if output.exists():raise RuntimeError('Package already exists')
    with tarfile.open(output,'w:gz') as tar:
        tar.add(root,arcname=root.name,filter=lambda info:None if '__pycache__' in info.name else info)
    sha=c.dev.sha(output)
    output.with_suffix(output.suffix+'.sha256').write_text(f'{sha}  {output.name}\n')
    checked=0
    with tarfile.open(output,'r:gz') as tar:
        for r in manifest:
            f=tar.extractfile(root.name+'/'+r['path']);h=hashlib.sha256()
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            assert h.hexdigest()==r['sha256'],r['path'];checked+=1
    record={'archive':str(output),'sha256':sha,'bytes':output.stat().st_size,
            'member_files_sha256_checked':checked,'all_pass':True,'status':c.STATUS}
    c.dump(output.with_suffix(output.suffix+'.verification.json'),record)
    print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['uncertainty','weight_stability','gates','domain','report','verify','package'],required=True)
    args=p.parse_args();t=time.monotonic()
    globals()[args.phase](args.root)
    if args.phase!='package':
        c.dump(args.root/f'logs/{args.phase}.runtime.json',{'wall_seconds':time.monotonic()-t,'exit_code':0,'UTC_finish':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
