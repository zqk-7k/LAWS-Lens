#!/usr/bin/env python3
"""Preserve completed mechanism comparisons, including invalid/negative results."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_20260908 as t
dev=t.dev

DESCRIPTIONS={
 'MCWF-TEMPORAL-RESPONSE-01':'在原质量响应上增加匹配滤波峰附近的时间形状残差。与相同数据、相同训练步数的 CONTINUE 对照；残差不是经过校准的卡方概率，不作为硬否决。',
 'MCWF-DETECTOR-EXCHANGE-04':'同一个网络分别读 H1/L1 与 L1/H1 的匹配响应，平均预测概率。交换的是证据的输入排列，不假设两个探测器有相同响应或 PSD。',
 'MCWF-MULTIRATE-LOWBAND-06':'保留原峰值2秒4096点的40-580Hz主输入，额外使用20-80Hz、16秒、256Hz采样的4096点匹配响应。与仅在相同4096训练源上续训的 SMALL-CONTROL 成对比较。',
 'MCWF-MULTIRATE-INTEGRATION-07':'冻结 MULTIRATE 模型，比较单模型或三模型平均预测，以及保留旧质量项叠加或替换旧质量项。无新网络训练；模型目录中的 selected.pt 仅存先验和上游模型引用，不是可执行网络权重。',
 'MCWF-CONDITIONAL-ETA-08':'冻结 MULTIRATE 的质量边缘分布，只训练给定质量后的对称质量比 eta 条件头。联合预测除以匹配的模拟先验，再扣除已包含的质量重叠；不把质量比与质量重复当作独立证据。',
 'MCWF-GENUINE-NOISE-SCALE-09':'仅在原峰值2秒质量响应上加入额外标准化之前的 H1/L1 真实尺度。输入由26秒原始信号与噪声精确重放验证，不是已经 z-score 后的近常数。此前02轮被标记无效；本轮独立保留，两个数字不能混用。',
 'MCWF-CONDITIONAL-INTRINSICS-10':'固定质量边缘分布，分别训练条件 chi_eff 和条件 eta+chi_eff；联合头保留二者相关性，不把几个一维重叠直接相乘。所有标签来自模拟源，真实 PE 未进入训练。',
 'MCWF-JOINT-INTRINSIC-EVIDENCE-12':'冻结条件质量、自旋和质量比预测，比较完整联合概率的先验校正重叠或 Bhattacharyya 重叠。新分数经模拟校准后，以叠加或替换旧质量项的独立对照接入；不是几个相关边缘证据的乘积。',
 'MCWF-NAMESPACED-SOURCE-TIME-13':'只修改一维时间先验的独立系统加权、有限样本带宽和不确定度处理。63条可探测像对延迟来自34个2.5PLUS系统，不把重采样条数当作独立样本数。ET与2.5PLUS的ID必须分别命名；跨网络物理身份及重复记录红移冲突另表审计。保留历史观测日历，未声称完成O3-only时间修正。',
 'MCWF-JOINT-FUSION-RETUNE-14':'冻结12轮的波形分数以及原时间、天空原始矩阵，只在模拟validation上重新选择归一化三通道权重。加入旧OMC单独重选权重对照；严格正权重与允许零权重分别报告。不能把权重为零的对照称为完整三通道。',
 'MCWF-JOINT-PREDICTIVE-ENSEMBLE-15':'先平均三个冻结模型的联合预测概率，再计算质量、自旋、质量比一致性；不是挑最优真实seed。比较原权重与validation重选权重，时间/天空原始矩阵不变。三个catalog/calibration seed共享同一个三模型集成，其SD不能称为三个独立集成模型重训方差。',
 'MCWF-JOINT-2D-CALIBRATION-17':'将联合预测的先验校正重叠与Bhattacharyya重叠一起输入两维单调logistic校准器，不把二者当成独立证据。正则强度、二维支持域只由source/noise分离的development拟合与调参子集确定，之后冻结，比较原权重与validation重选权重。',
 'MCWF-VALIDATION-FUSION-STABILITY-18':'保持每个对照的波形分数完全不变，依据模拟validation的source-system bootstrap，判断重新选择融合权重的F50收益是否超过抽样波动；否则保留原权重。ONE-SE与CI95是预先固定的开发稳定性规则，不是经过择优后仍有效的确认性显著性检验。',
 'MCWF-LOWBAND-TIMEFREQUENCY-19':'保留原2秒主输入及06轮匹配响应，在20-80Hz低频16秒4096点上增加时频演化残差。与相同数据、相同步数的LOW-CONTINUE对照，均包含epoch0且仅按模拟development交叉熵选checkpoint。未改变时间、天空或融合权重。',
 'MCWF-CONDITIONAL-MASS-CALIBRATION-20':'冻结全部神经网络，只根据波形自身预测的质量、宽度和熵调整质量密度的温度及指数倾斜。八个系数用模拟真值拟合、独立noise半组调正则，超支持域保留原分布。没有使用真实PE校准质量，没有重新训练encoder。',
}


def configure_plot():
    names={f.name for f in font_manager.fontManager.ttflist}
    face='Times New Roman' if 'Times New Roman' in names else 'DejaVu Serif'
    plt.rcParams.update({'font.family':face,'font.size':9,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    return face


def savefig(fig,root,name):
    (root/'figures').mkdir(exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(root/f'figures/{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def comparison(root):
    contract=json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())
    ident=contract.get('id',root.name)
    contract_files=' '.join(p.name for p in (root/'contracts').iterdir())
    if 'SCALE_INPUT_INVALIDITY' in contract_files or 'SOURCE_ID_NAMESPACE_INVALIDITY' in contract_files:
        raise RuntimeError('Invalid input/identity experiment must not receive an ordinary performance report')
    assessment=json.loads((root/'contracts/ROUND_ASSESSMENT.json').read_text())
    summary=pd.read_csv(root/'tables/RETRIEVAL_SUMMARY.csv')
    seeds=pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv')
    budgets=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    real=budgets[(budgets.seed.astype(str)=='consensus')&(budgets.method=='fusion')&(budgets.budget.isin([10,20]))]
    target=pd.read_csv(root/'tables/DEVELOPMENT_TARGET_CHECK.csv')
    dev.csv_write(root/'tables/CONSENSUS_TOP10_TOP20.csv',real)
    training=[]
    for path in sorted((root/'models').glob('*/*/*/COMPLETE.json')):
        row=json.loads(path.read_text());row.update(kind=path.parts[-4],deployment=path.parts[-3],slot=path.parts[-2]);training.append(row)
    dev.csv_write(root/'tables/TRAINING_SUMMARY.csv',pd.DataFrame(training))
    changes=contract.get('changed_channels',[])
    physical_note=('本轮时间分数已改变；波形和天空原始分数及外层权重保持冻结。时间变化、旧lookup重放和source来源见专门审计表。'
        if 'time' in changes else '时间和天空原始分数保持冻结；逐pair重放见FROZEN_CHANNEL_REPLAY等审计表。')
    if contract.get('outer_weights_changed',False):
        physical_note+=' 本轮重新选择外层权重，不可声称权重保持不变；允许零权重的方案须按active_channels解释。'
    lines=[f'# {ident} 独立机制对照报告','',
      '本轮计算完成不代表共同改善目标已经完成。历史保留版 OMC、旧矩阵、旧权重和旧候选范围没有覆盖。',
      '真实目录在本项目中已经反复用于开发反馈，因此下面不是盲测或确认性发现。官方前端/公开 Hanabi 表重合不是透镜真值，本轮没有运行 Hanabi。','',
      '## 改动与冻结','',DESCRIPTIONS.get(ident,'具体方法见冻结合同。'),
      'O3/O4a 使用相同算法、候选网格、优化及评估规则，模型与校准分别使用对应运行期模拟数据。不能把两个运行期各自胜出的不同方法合并为统一新结果。',
      '所有网络参数、温度和增量系数先由模拟 development/validation 固定；之后才计算 test，并在配置冻结后附加真实 PE/官方审计。',
      physical_note,
      '预测质量密度不是公开 PE 后验；加权波形分数不是完整透镜 Bayes factor。','',
      '## 注入三通道结果','',
      '| Run | 方法 | R@1 mean +/- SD | R@10 mean +/- SD | AUPRC | F50 | F90 |',
      '|---|---|---:|---:|---:|---:|---:|']
    z=summary[(summary.split=='test')&(summary['mode']=='fusion')]
    for r in z.itertuples():
        lines.append(f'|{r.deployment}|{r.method}|{r.macro_r_at_1_mean:.5f} +/- {r.macro_r_at_1_std:.5f}|{r.macro_r_at_10_mean:.5f} +/- {r.macro_r_at_10_std:.5f}|{r.average_precision_mean:.5f}|{r.false_at_recall_0p5_mean:.2f}|{r.false_at_recall_0p9_mean:.2f}|')
    lines+=['','F50/F90 是达到相应真对 recall 时的假对数，不是误配概率。完整 waveform-only 与逐 seed 数字另存 CSV。','',
      '## 真实 Top-10 与 Top-20','',
      '| Run | 方法 | Top-B | Mc BC>=0.5 | Mc BC中位数 | 灾难性Mc | Dmax<=3 | 官方1%前端 | 公开Hanabi重合 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in real.itertuples():
        lines.append(f'|{r.deployment}|{r.config}|{r.budget}|{r.BC_mc_ge_0p5}|{r.median_BC_mc:.5f}|{r.catastrophic_mc}|{r.Dmax_le_3}|{r.official_frontend}|{r.official_hanabi}|')
    lines+=['','完整 Top-100 和所有事件对的 wf/time/sky、PE、官方 FPP/阶段表见 evaluation/；Top-50/100 预算见 PE_OFFICIAL_BUDGETS.csv。灾难性 Mc 定义沿用 BC<0.1 或 D>5。','',
      '## 逐运行期判定','',
      '| 配置 | Run | Top10/20保护项 | PE增加 | 官方增加 | 每seed注入保护 | 通过 |',
      '|---|---|---|---|---|---|---|']
    for r in target.itertuples():
        lines.append(f'|{r.configuration}|{r.deployment}|{r.no_loss_top10_top20}|{r.PE_gain}|{r.official_gain}|{r.all_seed_injection_guard}|{r.run_target_pass}|')
    winners=assessment['development_candidates_requiring_fresh_confirmation']
    lines+=['','共同通过且仍需独立确认的开发配置：'+(', '.join(winners) if winners else '无。本轮不能替换保留版。'),'',
      '## 不确定度与限制','',
      'PAIRED_SOURCE_BOOTSTRAP.csv 给出固定模型条件下的成对差值区间：R@1/R@10 用10000次按family分层的透镜系统抽样，两个方向一起抽取；AUPRC/F50/F90 用1000次source-system加权bootstrap。后者没有另外重采样共享噪声块，不能被称为完全独立catalog uncertainty。',
      '三seed SD 与上述条件bootstrap回答不同问题。共享事件的真实pair相关系数及候选比例仅作描述，不报告普通独立pair的显著性。',
      '旧的BAYESTAR模拟匹配滤波观测、有限源/噪声数据、运行期时间约定和实际strain域差异仍存在；新波形分支不会自动消除它们。若某方法满足开发目标，仍需新source/noise注入确认。',
      '历史保护文件SHA-256复查见manifest/FINAL_PROTECTED_RECHECK.csv。新目录的变化项必须按本轮合同和逐pair审计解释。','',t.STATUS]
    (root/'reports/ROUND_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    font=configure_plot()
    methods=sorted(seeds.method.unique(),key=lambda s:(s!='OMC',s))
    labels=['\n'.join(textwrap.wrap(x.replace('-',' '),width=23)) for x in methods]
    fig,axes=plt.subplots(2,3,figsize=(15,max(8,len(methods)*1.3)),layout='constrained')
    for i,dep in enumerate(t.DEPS):
        a=seeds[(seeds.deployment==dep)&(seeds.split=='test')&(seeds['mode']=='fusion')]
        for j,(column,label) in enumerate([('macro_r_at_10','Companion R@10'),('average_precision','Pair AUPRC'),('false_at_recall_0p9','False pairs at 90% recall')]):
            ax=axes[i,j]
            for k,name in enumerate(methods):
                values=a[a.method==name][column].to_numpy(float)
                ax.scatter(values,k+np.linspace(-.12,.12,len(values)),s=22,color=plt.get_cmap('tab10')(k),alpha=.85)
                if len(values):ax.plot([np.median(values)]*2,[k-.2,k+.2],color='black',lw=1)
            ax.set_yticks(range(len(methods)),labels,fontsize=7);ax.invert_yaxis()
            ax.set_xlabel(label);ax.set_title(dep.replace('gwtc3','O3').replace('gwtc4','O4a'));ax.grid(axis='x',alpha=.18)
    savefig(fig,root,'fig_injection_individual_seeds')
    fig,axes=plt.subplots(2,3,figsize=(15,max(8,len(methods)*1.3)),layout='constrained')
    for i,dep in enumerate(t.DEPS):
        a=real[real.deployment==dep]
        for j,(column,label) in enumerate([('BC_mc_ge_0p5','Mc BC >= 0.5'),('official_frontend','Official 1% frontend overlap'),('official_hanabi','Published Hanabi-table overlap')]):
            ax=axes[i,j]
            for b,offset,color in [(10,-.17,'#14857B'),(20,.17,'#B75975')]:
                vals=[a[(a.config==name)&(a.budget==b)][column].iloc[0] for name in methods]
                ax.barh(np.arange(len(methods))+offset,vals,height=.3,color=color,label=f'Top-{b}')
            ax.set_yticks(range(len(methods)),labels,fontsize=7);ax.invert_yaxis()
            ax.set_xlabel(label);ax.set_title(dep.replace('gwtc3','O3').replace('gwtc4','O4a'));ax.set_xlim(0,21)
    axes[0,0].legend(frameon=False,loc='upper right');savefig(fig,root,'fig_real_PE_official_top10_top20')
    dev.json_write(root/'figures/PLOT_PROVENANCE.json',{'font':font,'injection_points':'individual seeds,median black line',
        'real':'descriptive counts,not significance or lens true labels'})


def consensus(root):
    contract=json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())
    assessment=json.loads((root/'contracts/ROUND_ASSESSMENT.json').read_text())
    budgets=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    sub=budgets[budgets.budget.isin([10,20])]
    dev.csv_write(root/'tables/CONSENSUS_TOP10_TOP20.csv',sub)
    display=[]
    for path in (root/'evaluation').glob('*/*/consensus_fusion_pairs.parquet'):
        name,dep=path.parts[-3:-1];rule=name.rsplit('__',1)[-1];f=pd.read_parquet(path)
        key={'MEAN-RANK':'rank_mean','MEDIAN-RANK':'rank_median','MEAN-SCORE':'normalized_score_mean','MEDIAN-SCORE':'normalized_score_median'}[rule]
        frame=f[['pair_key','consensus_rank',key]].copy();frame['configuration']=name;frame['deployment']=dep
        frame['aggregation_score']=frame[key]*(-1 if rule.endswith('RANK') else 1)
        frame['aggregation_units']='negative rank' if rule.endswith('RANK') else 'normalized weighted score'
        display.append(frame.drop(columns=[key]))
    dev.csv_write(root/'tables/CONSENSUS_DISPLAY_SCORE.csv',pd.concat(display,ignore_index=True))
    metrics=[]
    for item in contract['upstreams']:
        f=pd.read_csv(Path(item['root'])/'tables/RETRIEVAL_PER_SEED.csv')
        f=f[(f.method==item['method'])&(f.split=='test')].copy();f['upstream_code']=item['code']
        f['interpretation']='Original single-model test;NOT new common-catalog ensemble performance';metrics.append(f)
    dev.csv_write(root/'tables/ORIGINAL_SINGLE_MODEL_METRICS_NOT_ENSEMBLE.csv',pd.concat(metrics,ignore_index=True))
    lines=['# MCWF-CONSENSUS-DIAGNOSTIC-16：共识汇总方式对照','',
        '本轮只改变三个部署的最终汇总，未改动任何逐seed波形、时间、天空分数或权重，未重训模型。它不能称为新的波形编码器结果。',
        '原平均名次准确重放。其余规则为归一化分数平均、名次中位数、归一化分数中位数。归一化仅除以该seed冻结权重之和，不使用真实目录z-score。',
        '原始 final_score_mean 只是旧分数的均值，并非所有汇总规则的排序依据。实际汇总数值和单位见CONSENSUS_DISPLAY_SCORE.csv。',
        '各seed原注入指标仍保留，但三个seed使用不同模拟目录，不能直接拼成新的集成recall。故本轮不伪造集成R@K；若候选满足开发目标，必须让全部模型对同一批新source/noise目录评分。',
        '真实PE和官方重合只在汇总后附加，仍属于适应性开发审计。官方1%/公开Hanabi表并不是透镜真值，也未执行新Hanabi。','',
        '| Run | 配置 | Top-B | Mc BC>=0.5 | Mc BC中位数 | 灾难性Mc | Dmax<=3 | 官方1% | 公开Hanabi |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sub.itertuples():
        lines.append(f'|{r.deployment}|{r.config}|{r.budget}|{r.BC_mc_ge_0p5}|{r.median_BC_mc:.6f}|{r.catastrophic_mc}|{r.Dmax_le_3}|{r.official_frontend}|{r.official_hanabi}|')
    winners=assessment['development_candidates_requiring_fresh_confirmation']
    lines+=['','需要全模型共同新目录验证的开发候选：'+(', '.join(winners) if winners else '无。'),
        '没有完成整体目标；不能把不同运行期各自最优的不同汇总方案拼接成统一方法。','',t.STATUS]
    (root/'reports/CONSENSUS_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    font=configure_plot();names=sorted(sub.config.unique())
    fig,axes=plt.subplots(1,2,figsize=(13,max(8,.36*len(names))),layout='constrained')
    for ax,dep in zip(axes,t.DEPS):
        a=sub[sub.deployment==dep].set_index(['config','budget'])
        values=np.asarray([[a.loc[(name,b),column] for b,column in [(10,'BC_mc_ge_0p5'),(20,'BC_mc_ge_0p5'),(10,'official_frontend'),(20,'official_frontend')]] for name in names])
        ax.imshow(values,aspect='auto',cmap='YlGnBu',vmin=0,vmax=20)
        for i in range(len(names)):
            for j in range(4):ax.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=7,color='white' if values[i,j]>12 else 'black')
        ax.set_xticks(range(4),['Mc Top10','Mc Top20','Official Top10','Official Top20'],rotation=20,ha='right')
        ax.set_yticks(range(len(names)),names,fontsize=6);ax.set_title(dep)
    savefig(fig,root,'fig_consensus_descriptive_counts')
    dev.json_write(root/'figures/PLOT_PROVENANCE.json',{'font':font,'counts':'descriptive,not lensed truths or statistical significance'})


def pilot(root):
    path=root/'tables/PAIRED_PILOT_EVENT_RESULTS.csv'
    if not path.exists():path=root/'tables/PILOT_EVENT_RESULTS.csv'
    a=pd.read_csv(path);rows=[]
    baseline='OMC-mean-predicted-logMc'
    for dep in t.DEPS:
        sub=a[a.deployment==dep]
        reference=sub[sub.method==baseline].set_index('row_index')
        for name in sorted(set(sub.method)-{baseline}):
            z=sub[sub.method==name].set_index('row_index').loc[reference.index].copy()
            z['delta']=z.absolute_logmc_error-reference.absolute_logmc_error
            group=z.groupby(['mass_bin','source_uid']).delta.mean().reset_index()
            rng=np.random.default_rng(202609870);draw=np.zeros(10000)
            for _,part in group.groupby('mass_bin'):
                values=part.delta.to_numpy(float)
                draw+=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)*len(values)/len(group)
            rows.append({'deployment':dep,'method':name,'source_systems':len(group),'delta_MAE_logMc':group.delta.mean(),
                'CI_low':np.quantile(draw,.025),'CI_high':np.quantile(draw,.975),'replicates':10000,
                'unit':'source-system stratified by five preselected mass bins;both images together;development pilot'})
    table=pd.DataFrame(rows);dev.csv_write(root/'tables/PILOT_PAIRED_SOURCE_BOOTSTRAP.csv',table)
    methods=sorted(a.method.unique());font=configure_plot()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,dep in zip(axes,t.DEPS):
        for k,name in enumerate(methods):
            z=a[(a.deployment==dep)&(a.method==name)].groupby('source_uid').absolute_logmc_error.mean().to_numpy()
            ax.scatter(k+np.linspace(-.15,.15,len(z)),z,s=10,alpha=.7,color=plt.get_cmap('tab10')(k))
            ax.plot([k-.2,k+.2],[np.median(z)]*2,color='black')
        ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right',fontsize=7)
        ax.set_ylabel('Mean absolute log Mc error per source');ax.set_title(dep)
    savefig(fig,root,'fig_profile_mass_error_source_points')
    lines=['# 连续质量拟合pilot：独立记录','',
        '这是按预定哈希与五个质量层选择的20个源/运行期，每源两幅像的development诊断。它不是真实候选PE，也没有运行候选重排、recall或官方重合优化。',
        '投影统计量只最大化有限对齐自旋模板的相位/到达时刻响应；它不是完整非高斯噪声似然、evidence或公开PE后验。',
        '以原神经网络预测均值为基线；正的delta表示质量误差变大。','',
        '| Run | 方法 | 源数 | MAE差值 | 95%区间 |','|---|---|---:|---:|---|']
    for r in table.itertuples():lines.append(f'|{r.deployment}|{r.method}|{r.source_systems}|{r.delta_MAE_logMc:.5f}|[{r.CI_low:.5f}, {r.CI_high:.5f}]|')
    lines+=['','每个源两幅像一起bootstrap，并按预选质量层分层。仅20个独立源，不能称为大样本覆盖率证明。',
        '连续20Hz长窗口比同类40Hz点优化减少误差，但仍未超过保留网络；不能因此替换现有波形分数。',
        '已有的原始采样重放错误及修复记录都保留；不得把格式适配失败解读成物理模型失败。','',t.STATUS]
    (root/'reports/PILOT_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dev.json_write(root/'figures/PLOT_PROVENANCE.json',{'font':font,'points':'one source,mean over two images'})


def seal(root):
    # Only completed outputs are packaged; regenerable raw/features stay remote.
    destination=P/'packages'/f'{root.name}_deliverables.tar.gz'
    if destination.exists():raise RuntimeError('Never overwrite a delivery')
    dependency_rows=[]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        source=Path(filename)
        if source.suffix!='.py' or not source.is_relative_to(P) or not source.is_file():continue
        relative=source.relative_to(P)
        output=root/'dependency_source'/relative
        output.parent.mkdir(parents=True,exist_ok=True)
        if not output.exists():shutil.copy2(source,output)
        if dev.sha(output)!=dev.sha(source):raise RuntimeError('Dependency changed during freeze')
        dependency_rows.append({'original_path':str(source),'snapshot':str(output.relative_to(root)),'sha256':dev.sha(source)})
    dev.csv_write(root/'manifest/DEPENDENCY_SHA256.csv',pd.DataFrame(dependency_rows).drop_duplicates())
    (root/'README_DELIVERY_CN.md').write_text(
        '# 独立开发对照交付\n\n'
        '这是完整保留正负结果的紧凑记录，不是主方法替换。正式状态见contracts和报告。\n'
        '包内包括新模型选定checkpoint、配置、计算脚本、已加载项目依赖快照、逐seed评估、完整真实pair及官方PE联表、日志、图和逐文件哈希。\n'
        '未打包原始GWOSC strain、大型可重建feature/raw缓存及历史模型bank。复算需要原项目中的manifest所列输入；本包不是脱离这些输入的独立安装包。\n'
        '所有输入路径和哈希保留在manifest；dependency_source按原项目相对路径归档，不修改服务器中的原文件。\n'
        '当前真实目录已反复用于自适应开发；官方候选重合不是真值，未运行新的Hanabi。\n',encoding='utf-8')
    checks=[]
    if (root/'manifest/INPUT_SHA256.csv').exists():
        for r in pd.read_csv(root/'manifest/INPUT_SHA256.csv').itertuples():
            now=dev.sha(Path(r.path));checks.append({'path':r.path,'before':r.sha256,'after':now,'unchanged':now==r.sha256})
        if not all(r['unchanged'] for r in checks):raise RuntimeError('Historical hash changed')
        dev.csv_write(root/'manifest/FINAL_PROTECTED_RECHECK.csv',pd.DataFrame(checks))
    target=root/'scripts/document_rounds.py';shutil.copy2(__file__,target)
    chosen=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        relative=p.relative_to(root)
        if any(x in ('cache','features','__pycache__') for x in relative.parts):continue
        if p.name in ('resume.pt','OUTPUT_SHA256.csv','SHA256SUMS.txt','DELIVERY_CHECK.json'):continue
        if p.suffix in ('.py','.md','.log','.json','.csv','.txt') and p.stat().st_size<10*2**20:
            text=p.read_text(errors='replace')
            if re.search(r'(?m)^-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----\s*$',text):raise RuntimeError('Secret found in delivery')
        chosen.append(p)
    manifest=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':dev.sha(p)} for p in chosen]
    (root/'manifest').mkdir(exist_ok=True)
    dev.csv_write(root/'manifest/OUTPUT_SHA256.csv',pd.DataFrame(manifest))
    (root/'manifest/SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in manifest),encoding='utf-8')
    for p in (root/'manifest/OUTPUT_SHA256.csv',root/'manifest/SHA256SUMS.txt'):chosen.append(p)
    with tarfile.open(destination,'w:gz',compresslevel=6) as archive:
        for p in chosen:archive.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    expected={str(Path(root.name)/r['path']):r['sha256'] for r in manifest};count=0
    with tarfile.open(destination,'r:gz') as archive:
        for member in archive:
            if member.name in expected:
                f=archive.extractfile(member);digest=hashlib.sha256()
                for chunk in iter(lambda:f.read(4*2**20),b''):digest.update(chunk)
                if digest.hexdigest()!=expected[member.name]:raise RuntimeError('Internal archive hash mismatch')
                count+=1
    if count!=len(expected):raise RuntimeError('Missing archived artifacts')
    sha=dev.sha(destination);destination.with_suffix(destination.suffix+'.sha256').write_text(sha+'  '+destination.name+'\n')
    dev.json_write(root/'manifest/DELIVERY_CHECK.json',{'utc':datetime.now(timezone.utc).isoformat(),'archive':str(destination),
        'sha256':sha,'members':len(chosen),'internally_verified_files':count,'raw_feature_caches_excluded':True,
        'protected_unchanged':all(r['unchanged'] for r in checks),'is_goal_completion':False})
    print(json.dumps({'archive':str(destination),'sha256':sha,'verified':count}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['comparison','consensus','pilot','seal'],required=True);a=p.parse_args()
    globals()[a.stage](a.root)
