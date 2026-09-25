#!/usr/bin/env python3
"""Archive R56-R58 calibration outcomes and failures without claiming adoption."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_monotone_boundary_20260909 as common
n=common.n


def main(expansion,joint,path,boundary):
    roots=[expansion,joint,path]
    gates=[json.loads((r/'contracts/PILOT_GATE.json').read_text()) for r in roots]
    if any(g['gate']!='FAIL' for g in gates):
        raise RuntimeError('This report applies only to the observed three failed controls')
    report=path/'reports/EXPANDED_CALIBRATION_COMPLETE_CN.md'
    if report.exists(): raise RuntimeError('Do not overwrite completed report')
    hashes=[]
    for root in roots:
        for name in ('INPUT_SHA256.csv','SEALED_EXPANDED_INPUTS.csv'):
            file=root/'manifest'/name
            if file.exists():
                for row in pd.read_csv(file).itertuples():
                    after=n.sha(Path(row.path))
                    hashes.append({'experiment':root.name,'path':row.path,'before':row.sha256,'after':after,'exact':after==row.sha256})
        if (root/'results').exists() and list((root/'results').rglob('*.parquet')):
            raise RuntimeError('Failed pilot must not have catalog results')
    if not all(r['exact'] for r in hashes):
        raise RuntimeError('Frozen input changed')
    n.write_csv(path/'audit/ALL_EXPANSION_INPUT_HASH_CHECKS.csv',hashes)
    units=pd.read_csv(expansion/'tables/PLANNING_INVENTORY.csv')
    plan=pd.read_parquet(expansion/'contracts/PAIR_PLAN.parquet')
    result=pd.read_parquet(expansion/'tables/PAIR_RESULTS.parquet')
    if len(plan)!=1053 or len(result)!=1053 or result.status.ne('COMPLETE').any():
        raise RuntimeError('Incomplete physical measurement set')
    if not np.isfinite(result[['deficit','minimum_independent_power']]).all().all():
        raise RuntimeError('Nonfinite physical statistic')
    if result.profile_replay_max_difference.max()!=0:
        raise RuntimeError('Frozen profile replay changed')
    checks=[]
    for dep,g in plan.groupby('deployment'):
        a,b=g[g.fold==0],g[g.fold==1]
        sa=set(a.source_i)|set(a.source_j);sb=set(b.source_i)|set(b.source_j)
        na=set(a.noise_parent_i)|set(a.noise_parent_j);nb=set(b.noise_parent_i)|set(b.noise_parent_j)
        if sa&sb or na&nb: raise RuntimeError('Source/noise fold leak')
        checks.append({'deployment':dep,'fit_source_groups':len(sa),'tune_source_groups':len(sb),
            'fit_noise_parents':len(na),'tune_noise_parents':len(nb),'source_overlap':0,'noise_overlap':0})
    n.write_csv(path/'audit/EXPANDED_FOLD_ISOLATION.csv',checks)
    comparisons=[]
    for label,root in [('R56',expansion),('R57',joint),('R58',path)]:
        frame=pd.read_csv(root/'tables/PILOT_GATE_COMPARISON.csv')
        frame.insert(0,'experiment',label)
        comparisons.append(frame)
    compared=pd.concat(comparisons,ignore_index=True)
    n.write_csv(path/'tables/ALL_EXPANDED_CONTROLS_COMPARISON.csv',compared)
    selected=[]
    for label,root,name in [('R56',expansion,'EXPANDED_CLASSIFIERS.json'),
                            ('R57',joint,'EXPANDED_JOINT_CLASSIFIERS.json'),
                            ('R58',path,'EXPANDED_JOINT_CLASSIFIERS.json')]:
        for key,c in json.loads((root/'configs'/name).read_text()).items():
            dep,seed=key.split('/')
            selected.append({'experiment':label,'deployment':dep,'seed':seed,
                'ridge':c['shared_classifier']['ridge'],'features':len(c['shared_classifier']['mean'])})
    n.write_csv(path/'tables/ALL_SELECTED_REGULARIZERS.csv',selected)
    resource=[]
    for name in ('prepare','unit','measure','fit'):
        receipt=json.loads((expansion/f'logs/{name}_RECEIPT.json').read_text())
        resource.append({'stage':name,'wall_seconds':receipt['wall_seconds'],'returncode':receipt['returncode']})
    n.write_csv(path/'tables/EXPANSION_RUNTIME.csv',resource)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    labels=['Full population','Random null','Neighbor null']
    diag=['full_population','random_draw','neighbor_population']
    for ax,dep,title in zip(axes,['gwtc3','gwtc4'],['O3','O4a']):
        for root,method,label,color,offset in [(expansion,'EXPANDED-HT-PROFILE','R56 profile','#00888d',-.15),
            (joint,'EXPANDED-HT-JOINT','R57 joint','#bb455d',0.),
            (path,'EXPANDED-HT-JOINT','R58 ridge path','#77642a',.15)]:
            f=pd.read_csv(root/'tables/PILOT_GATE_COMPARISON.csv').set_index(['deployment','diagnostic'])
            values=[f.loc[(dep,d),method]-f.loc[(dep,d),'R55-FROZEN'] for d in diag]
            ax.plot(np.arange(3)+offset,values,'o',label=label,color=color,markersize=5)
        ax.axhline(0,color='.2',lw=.8)
        ax.set_xticks(range(3),labels,rotation=12)
        ax.set_title(title,loc='left')
        ax.set_ylabel('Validation log-loss change vs R55')
        ax.grid(axis='y',alpha=.15)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,ncol=3,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.90))
    (path/'figures').mkdir(exist_ok=True)
    for ext in ('pdf','png'): fig.savefig(path/f'figures/EXPANDED_CALIBRATION_GUARDS.{ext}',dpi=180)
    plt.close(fig)
    sentences=[]
    for label,root in [('R56',expansion),('R57',joint),('R58',path)]:
        frame=pd.read_csv(root/'tables/PILOT_GATE_COMPARISON.csv')
        cols=['deployment','diagnostic','R55-FROZEN','EXPANDED-HT-PROFILE']
        if 'EXPANDED-HT-JOINT' in frame: cols.append('EXPANDED-HT-JOINT')
        sentences.append('### '+label+'\n\n'+frame[cols].to_markdown(index=False)+'\n')
    text='''# R56-R58 扩展模拟校准完整报告

状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。**三个对照均未通过全部预定验证检查，未进入真实目录重排。完整目标仍未达成。**

## 1. 为什么继续做这批计算

R55 已把关键 pair 从 NODUP 共识rank6降至rank28，共识Top10/20的PE和官方重合未下降，但还有3个O4a注入面板和3个O3单seed官方预算未通过严格门槛。
原物理校准每个run/fold只使用24个真伴随系统。为减少小样本标定问题，本轮扩大模拟校准，而不是按真实PE或官方名单调分。

## 2. 新增与冻结内容

- R56：使用现有开发集中全部质量合格真伴随系统；每个run/fold固定抽取128个普通null与64个邻近困难null，仍用三个波形特征。
- R57：在同一批扩展样本上，增加已有神经网络联合Mc/eta/chi分布的BC作为相关特征，四个特征合拟一个分类器；不把独立证据相加。
- R58：R57六个模型的最优ridge均位于0.001网格下界，因此独立预注册更细的有限正则化路径；没有修改R57结果或验收容差。
- waveform encoder、神经网络参数预测器、时间、天空、外层权重、原source/noise切分、真实范围均未修改。
- 旧encoder显式Mc/q差值项与新旧总分混合均未恢复。

物理统计仍为两个独立投影最优值之和减去共享Mc/q/等对齐自旋的投影最优值。使用IMRPhenomD近似、16s辅助波形和冻结优化预算；各像独立处理相位、振幅与到达窗口。
这不是完整BBH PE，不是完整联合likelihood，也没有运行Hanabi。

## 3. 样本与隔离

'''+units.to_markdown(index=False)+'''

合计1,053个物理pair，其中285个真伴随系统、768个null pair。复用480个已校验计算，只新增573个。新增计算全部完成，无失败；单事件投影回放最大差严格为0。
285个真伴随系统不是1,053个独立系统；null共享事件。它们来自现有R22B开发数据，不是重新生成的独立锁定测试。GW-LMC透镜环境有复用、逐像target-SNR缩放等限制仍保留。

'''+pd.DataFrame(checks).to_markdown(index=False)+'''

困难null被额外抽样，所以按精确两阶段纳入概率作逆概率加权。每个源对在完整、质量合格的背景总体中的各像组合总权重为1；再限制功率支持域后，目标是该总体中可评分的像组合质量，不声称每个条件子群仍严格等权。
类内归一化损失是比值估计量，不宣称有限样本精确无偏。方法依据为 [Horvitz–Thompson (1952)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446) 的不等概率抽样加权；本项目额外进行了可穷举小总体单元测试。

## 4. 选择规则与完整结果

拟合只用fold0，ridge只用fold1总体HT加权的实际部署策略log-loss选择，完全相同则选更强正则。普通null和困难邻近null另作诊断。
接受要求为：两个run的总体、普通和困难三类诊断，模型seed平均损失均不高于冻结R55，且至少一项严格改善。未根据真实PE、官方重合或新目录recall选择参数。
以下数值越低越好。R56-R58使用相同扩展评价样本；不能与旧R50仅24个tune真对的损失直接当成相同测试集比较。

'''+ '\n'.join(sentences)+'''

R56在O3总体和普通背景上略差；R57、R58均改善六项诊断中的五项，但O3普通背景仍略差，因此按原门槛保留FAIL。差异小不等于程序报错，也不足以自行放宽门槛或宣称方法整体成功。
全部ridge、逐seed损失和预测值均保留，没有从真实候选结果里选择一个看起来更好的模型。

## 5. Recall、PE、官方结果在哪里

这三个对照没有通过进入目录评分的门槛，因此没有产生R56/R57/R58的新recall或真实PE/官方排名；不能把R55的这些指标贴成它们的新结果。
最后一份完整目录评分是R55，路径为：

`'''+str(boundary)+'''`

其中文报告为 `reports/R55_FINAL_COMPLETE_REPORT_CN.md`，包内包括全部注入recall/AUPRC/F50/F90、O3/O4a各方法完整排名、Top10/20/50/100的PE和官方表，以及逐seed失败项。
关键pair的公开Mc BC仍为约0.091665，未改后验。该pair自己就在公开候选表里，移出某个seed的头部可能降低官方重合；官方入表不是透镜标签，不能为恢复数量而按名单补分。

## 6. 不确定性与文献边界

使用共同源约束的物理动机来自 [Lo与Magaña Hernandez](https://arxiv.org/abs/2104.09339)，但此处没有其Bayesian证据积分、人口与选择效应，不能称为复现Hanabi。
这些开发数据已多次查看。即使某个验证门槛随后通过，也需要独立数据验证，不能把模型选择后的结果称作盲测；相关选择偏差风险见 [Cawley与Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html)。
R55的system-bootstrap只衡量固定模型、噪声与目录下的条件不确定性，并未消除本项目自适应研发历史。

## 7. 资源与完整性

'''+pd.DataFrame(resource).to_markdown(index=False)+'''

实际物理计算使用24个CPU进程；573个新增pair的计算阶段约35.1分钟。运行中进程树RSS约18.8GiB为采样快照，不冒充严格峰值。
没有新encoder训练、GPU采样任务或大规模PE任务。全部输入SHA核对通过；旧结果未覆盖。原R55配置回执/报告排版问题及其修复记录均保留，不隐藏失败日志。

本包应包含R56-R58合同、全部预测与loss表、脚本、测量JSON、哈希和日志；原始strain、模型大文件和可重建的dense缓存不打包，按输入manifest在原项目中定位。
'''
    with report.open('x',encoding='utf-8') as f: f.write(text)
    for root,label in [(expansion,'R56'),(joint,'R57')]:
        with (root/f'reports/{label}_RESULT_AND_REPORT_INDEX_CN.md').open('x',encoding='utf-8') as f:
            f.write(f'# {label}结果\n\n校准Gate为FAIL，未重排真实候选，未产生新的recall或PE排名。\n\n完整联合报告：`{report}`。\n所有失败记录保留；不覆盖历史结果。\n')
    shutil.copy2(__file__,path/'scripts/shared_expansion_report.py')
    for root in roots:
        n.write_json(root/'contracts/FINAL_EXPORT_COMPLETE.json',{'UTC':n.utc(),
            'goal_achieved':False,'gate':'FAIL','status':n.STATUS,'no_real_or_catalog_scoring':True,
            'report':str(report),'input_hash_failures':0})
    n.write_json(path/'audit/FINAL_EXPANSION_AUDIT.json',{'UTC':n.utc(),'input_hash_checks':len(hashes),
        'input_hash_failures':0,'all_measurements_finite':True,'source_noise_overlap':0,
        'new_physical_pairs':573,'total_physical_pairs':1053,'goal_achieved':False})
    print('EXPANSION_REPORT_COMPLETE',len(hashes),report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('expansion-root','joint-root','path-root','boundary-root'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    main(a.expansion_root,a.joint_root,a.path_root,a.boundary_root)
