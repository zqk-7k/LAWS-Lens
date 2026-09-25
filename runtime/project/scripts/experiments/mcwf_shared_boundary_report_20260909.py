#!/usr/bin/env python3
"""Report all R55 outcomes, including unchanged real ranks and failing gates."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_monotone_boundary_20260909 as app
n=app.n
METHODS=app.METHODS


def main(root):
    output=root/'reports/R55_BOUNDARY_CORRECTION_COMPLETE_CN.md'
    if output.exists():
        raise RuntimeError('Do not overwrite final report')
    audit=json.loads((root/'audit/REFERENCE_COMPARISON_COMPLETE.json').read_text())
    if not audit['real_ranks_exact_R51']:
        raise RuntimeError('Real exact-replay audit required')
    metric=pd.read_csv(root/'tables/RETRIEVAL_SUMMARY.csv')
    display=metric[(metric.split=='sept8_reused')&(metric['mode']=='fusion')&metric.method.isin(METHODS)].copy()
    def pm(row,key):
        return f"{row[key+'_mean']:.6f} +/- {row[key+'_std']:.6f}"
    table=[]
    for _,row in display.iterrows():
        table.append({'run':row.deployment,'method':row.method,'R@1':pm(row,'macro_r_at_1'),
            'R@10':pm(row,'macro_r_at_10'),'AUPRC':pm(row,'average_precision'),
            'F50':pm(row,'false_at_recall_0p5'),'F90':pm(row,'false_at_recall_0p9')})
    display=pd.DataFrame(table)
    budgets=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    budgets=budgets[budgets.config.isin(METHODS)&budgets.budget.isin([10,20])]
    budgets=budgets[['deployment','config','budget','BC_mc_ge_0p5','median_BC_mc',
                     'catastrophic_mc','Dmax_le_3','official_frontend','official_hanabi']]
    g=pd.read_csv(root/'tables/REFERENCE_SPECIFIC_INJECTION_GUARDS.csv')
    g=g[(g.method==app.NEW)&(g.reference==METHODS[0])]
    failed=g[~g.existing_guard]
    pe=pd.read_csv(root/'tables/REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS.csv')
    pe=pe[(pe.method==app.NEW)&(pe.reference==METHODS[0])]
    pefail=pe[~pe.all_no_loss]
    key=pd.read_csv(root/'tables/CRITICAL_PAIR_ALL_RANKS.csv')
    key=key[['method','seed','unit','rank','consensus_rank','waveform_score','final_score']]
    changed=pd.read_csv(root/'tables/BOUNDARY_CHANGED_PAIRS.csv')
    distinct=changed.drop_duplicates(['deployment','panel','idx_i','idx_j'])
    if len(distinct)!=6 or not distinct.is_true_pair.astype(bool).all():
        raise RuntimeError('Unexpected boundary-change population')
    ci=pd.read_csv(root/'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv')
    ci=ci[(ci.method==app.NEW)&(ci.baseline==METHODS[0])&(ci['mode']=='fusion')]
    ci=ci[ci['quantity'].isin(['R1','R10','AUPRC','F50','F90'])]
    n.write_csv(root/'tables/PRIMARY_FUSION_DISPLAY.csv',display)
    n.write_csv(root/'tables/PRIMARY_REMAINING_INJECTION_FAILURES.csv',failed)
    n.write_csv(root/'tables/PRIMARY_REMAINING_EXTERNAL_FAILURES.csv',pefail)
    n.write_csv(root/'tables/BOUNDARY_DISTINCT_PHYSICAL_PAIRS.csv',distinct)
    n.write_csv(root/'tables/PRIMARY_BOOTSTRAP_DELTA_CI.csv',ci)
    evidence={}
    for dep in n.DEPS:
        evidence[dep]={'injection_failed_panels':int((failed.deployment==dep).sum()),
            'external_failed_budgets':int((pefail.deployment==dep).sum()),
            'consensus_external_all_pass':bool(pe[(pe.deployment==dep)&(pe.unit=='consensus')].all_no_loss.all())}
    n.write_json(root/'audit/FINAL_GOAL_READOUT.json',{'UTC':n.utc(),'goal_achieved':False,
        'status':n.STATUS,'runs':evidence,'real_exact_R51':True,
        'critical_consensus_rank':28,'distinct_injection_pairs_affected':6,
        'all_model_ranks_critical':[34,16,65],
        'reason_not_complete':'O4a three individual injection panels and O3 three individual-model official budgets remain below the inherited guards.'})
    per=pd.read_csv(root/'tables/RETRIEVAL_PER_MODEL.csv')
    per=per[(per.split=='sept8_reused')&(per['mode']=='fusion')&per.method.isin(METHODS)]
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,6.2))
    names=['NODUP','Shared profile','Boundary correction']
    for ax,quantity,title in zip(axes.flat,['macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'],
                              ['Companion R@10','Pair AUPRC','False pairs at 50% recall','False pairs at 90% recall']):
        for dep,offset,color,label in [('gwtc3',-.11,'#007c91','O3'),('gwtc4',.11,'#bd4551','O4a')]:
            for i,method in enumerate(METHODS):
                values=per[(per.deployment==dep)&(per.method==method)][quantity].to_numpy()
                if len(values)!=3:
                    raise RuntimeError('Three model points expected')
                x=i+offset
                ax.scatter(x+np.linspace(-.035,.035,3),values,s=20,facecolors='none',edgecolors=color,
                    label=label if i==0 else None)
                ax.plot([x-.065,x+.065],[values.mean()]*2,color=color,lw=2)
        ax.set_xticks(range(3),names)
        ax.set_title(title,loc='left',fontsize=11)
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    (root/'figures').mkdir(exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(root/f'figures/R55_COMPLETE_COMPARISON.{ext}',dpi=180)
    plt.close(fig)
    configs=json.loads((root/'configs/SELECTED_CONFIGURATIONS.json').read_text())
    weights=pd.DataFrame([{'deployment':r['deployment'],'seed':r['seed'],
        'waveform':r['weights'][0],'time':r['weights'][1],'sky':r['weights'][2]}
        for r in configs if r['method']==app.NEW])
    text=f'''# R55 共享波形一致性边界修正完整报告

状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。**完整目标仍未达成。**
这是独立探索结果，未修改旧 NODUP、PATH875、R51、v9.3、论文或既有候选表。

## 1. 本轮解决什么

R51 对低于拟合样本最小值的共享拟合差异 D，将正波形分清零。
这符合原先写下的 OOD 规则，但产生不合理的不连续：两个信号更容易由同一参数解释时，分数反而骤降。
本轮只修订这条边界策略，不把它描述为算术错误。

$$D=P_{{\rm independent}}-P_{{\rm shared}}\ge0,\qquad D_*=\max(D,D_{{\rm fit,min}}).$$

只在原质量合格且功率落在冻结支持域的 pair 中，用 D_* 查询原校准器。小于边界时只保持边界分数，不继续外推正奖励。
高 D 侧回退、功率支持域、分数 cap、全部分类器系数、encoder、时间、天空、外层权重不变。
六个分类器的单调性单元测试通过。模拟 tune 的普通和困难 null 两类损失在两个运行期均下降。

## 2. 波形评分流程和边界

质量合格 pair：独立拟合两段波形，再拟合共享 Mc/q/等对齐自旋，计算 D；使用一个三维条件分类器
`cosine, log(min independent projection power), -log1p(D_*)` 输出单一波形分数。
不合格 pair：严格沿用 NODUP 波形分数。两条分支不相加。
没有恢复旧 encoder 的显式 Mc/q 差异评分，也没有 .875 新旧总分混合。

$$S=w_{{\rm wf}}Z_{{\rm wf}}+w_{{\rm time}}Z_{{\rm time}}+w_{{\rm sky}}Z_{{\rm sky}}.$$

两个运行期算法相同，只沿用原先各自验证集拟合的系数。冻结权重：

{weights.to_markdown(index=False)}

这不是完整联合 PE；P 是经过处理波形的投影统计量，并非归一化 likelihood。没有运行 Hanabi，分数不是透镜 Bayes factor。
共享源检验的物理动机来自 [Lo 与 Magaña Hernandez 的贝叶斯框架](https://arxiv.org/abs/2104.09339)，但本轮没有执行其人口、选择效应和证据积分。
低侧饱和是本项目明确记录的单调外推限制，不声称由该论文推导。

## 3. 完整注入结果

以下对每个模型先平均三个已经使用过的目录，再报告三个模型的均值及样本 SD。
这些不是新锁定测试，不能把九个模型/目录组合当作九次独立训练。

{display.to_markdown(index=False)}

边界实际改变 6 个不同物理真伴随 pair；三个模型重复评分形成 18 行，不是18个独立系统。
O3 的所有逐模型/目录预定非劣检查已通过；O4a 仍有下列3个失败面板，不能用平均 AUPRC 改善掩盖：

{failed[['deployment','seed','panel','mode','macro_r_at_10_delta','average_precision_delta','false_at_recall_0p5_delta','false_at_recall_0p9_delta']].to_markdown(index=False)}

## 4. 关键 pair 与真实目录

`GW191103_012549--GW191105_143521` 的公开 Mc BC 保持约0.091665，未修改公开后验或其计算。
相对 NODUP 的共识 rank6，本轮为 rank28；三个模型分别34/16/65，全部退出Top10。
原 PATH875 为rank34，不能把本轮描述为比所有历史版本都更低。
真实 pair 的 D 均不触及此次低端修正，因此本轮全部真实分数、单seed排名和共识排名与R51逐项完全一致。

{budgets.to_markdown(index=False)}

共识Top10/20的PE和官方重合对NODUP均未变差，但严格逐seed要求尚未满足：

{pefail[['deployment','seed','budget','BC_mc_ge_0p5_delta','Dmax_le_3_delta','official_frontend_delta','official_hanabi_delta']].to_markdown(index=False)}

关键 pair 本身在官方表中，因此剔除这个低Mc重叠pair可能直接降低某些单seed官方重合。
公开候选重合不是透镜真值，也不是公开Hanabi支持透镜的数量；本轮不得按这些名单反向补排名。

## 5. 不确定性和未通过项

系统级bootstrap：query指标10,000次，pair指标2,000次，按透镜源系统/家族联合抽样，两幅像不拆开。
结果见 `uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv`，同时保存各面板及差值CI。
这些CI条件于现有模型、噪声和候选集合，不包含完整自适应模型搜索不确定性。
严格逐seed/逐目录门槛没有放宽，均值改善不等于达成完整目标。

## 6. 一致性、失败日志和交付

108个模型/方法/面板通过对齐核对；time、sky、外层权重与范围未变。原输入哈希核对完成。
初次评分入口因缺少旧评估器要求的配置哈希回执而在评分前退出；原失败日志保留。
兼容性补充只从早先冻结配置生成回执，配置哈希完全一致，没有重新选参。

- `tables/RETRIEVAL_PER_SEED.csv`：所有原验证、原测试及重复目录，波形/融合各自指标。
- `tables/PE_OFFICIAL_BUDGETS.csv`、`PER_SEED_PE_OFFICIAL_BUDGETS.csv`：共识与单seed预算。
- `results/<method>/<run>/consensus/`：完整排名及Top10/20/50/100，保留PE与官方字段。
- `tables/REFERENCE_SPECIFIC_*_GUARDS.csv`：与NODUP、R51、PATH875逐面板比较。
- `tables/BOUNDARY_DISTINCT_PHYSICAL_PAIRS.csv`：6个实际改变pair。
- `audit/FINAL_GOAL_READOUT.json`：明确 `goal_achieved=false`。
- `figures/R55_COMPLETE_COMPARISON.pdf/png`：逐模型点，不只均值误差棒。

后续只允许先在模拟development/validation改进统计可靠性；真实PE与官方结果已经用于发现问题，后续一律标为自适应开发，不能再宣称盲测。
'''
    with output.open('x',encoding='utf-8') as stream:
        stream.write(text)
    shutil.copy2(__file__,root/'scripts/shared_boundary_report.py')
    print('BOUNDARY_REPORT_COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
