#!/usr/bin/env python3
"""Report finite adaptive grid results without implying blind confirmation."""
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_document_mechanisms_v8_20260908 as d
dev=d.dev


def report(root):
    contract=json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())
    result=json.loads((root/'contracts/SEARCH_COMPLETE.json').read_text())
    a=pd.read_csv(root/'tables/ALL_GLOBAL_CONFIGURATIONS.csv')
    budgets=pd.read_csv(root/'tables/ALL_GLOBAL_PE_OFFICIAL_BUDGETS.csv')
    if result['qualifying']:
        raise RuntimeError('Qualifiedconfiguration requires fullevaluation andfreshconfirmation;do not produce negative-onlyreport')
    dev.json_write(root/'contracts/ROUND_ASSESSMENT.json',{'development_candidates_requiring_fresh_confirmation':[],
        'fresh_confirmation_completed':False,'goal_achieved':False,'status':d.t.STATUS,'historical_unchanged':'see final protected SHA recheck',
        'adaptive_selection_used_real_development_outcomes':True})
    grouping=[k for k in ('kind','statistic','integration','alpha') if k in a]
    summary=a.groupby(grouping,dropna=False).agg(configurations=('configuration','size'),
        validation_pass=('validation_guard','sum'),reused_test_pass=('reused_test_guard','sum'),both_real_goal=('both_real_target','sum')).reset_index()
    dev.csv_write(root/'tables/GLOBAL_SEARCH_SUMMARY.csv',summary)
    lines=[f'# {contract["id"]}：全局波形组合开发对照','',
      '本轮明确使用反复查看过的真实PE/官方预算结果作为全局超参数的开发选择指标。这不是validation-only选系数，不是盲测。网络和密度校准仍只从模拟数据学习。',
      'PE、官方FPP、Hanabi和事件名称不作为逐pair分数输入；没有候选专属加分、手工置顶或重排例外。所有全局系数同时用于O3/O4a与三个模型seed。',
      '原始时间、天空、外层C-fixed权重、目录范围与历史输出均保持冻结。旧simulation test已反复使用，本轮只称复用开发保护集，不称独立测试。','',
      f'预先冻结的配置数：{result["configurations"]}；同时通过两个运行期全部保护条件的配置数：{result["qualifying"]}。',
      '因此没有选出可替换保留版的配置，也没有为本轮启动独立确认注入。目标尚未完成，不能把单运行期改善包装成共同改善。','',
      '## 评分定义','',contract['formula'],'',
      '新联合项来自质量、自旋或质量比的相关预测，包含有限参考真对尾部惩罚和有界模拟校准增量。它不是完整PE或物理透镜Bayesfactor。',
      'alpha=0（若该轮包含）是移除旧波形信息的消融，不是说没有使用新波形。gamma/beta为0表示相应附加分量关闭，所有分量状态保留。','',
      '## 台账','',
      '`ALL_GLOBAL_CONFIGURATIONS.csv` 保存全部成功和失败网格点。未通过simulation validation的点不进入真实开发筛选，不能把未评估记作真实失败。',
      '`ALL_REUSED_SIMULATION_METRICS.csv` 保存逐配置、run、seed、split的waveform/fusion R@1/R@10、AUPRC、F50、F90及保护结果。',
      '`ALL_GLOBAL_PE_OFFICIAL_BUDGETS.csv` 保存通过validation点的两个运行期Top10/20/50/100 PE、灾难性Mc、官方1%和Hanabi表重合。未选择最终配置，因此没有新权威Top100名单。',
      '官方前端及公开Hanabi表重合不是透镜真值；本轮未运行Hanabi，不可称确认或探测。','',
      '## 选择偏差','',
      '真实目录的重复选择会高估其上面的改善。任何以后满足开发目标的配置，仍需冻结后用全新source/noise检查simulation保护；即使该检查通过，也不能把真实开发重合提升称为独立复现。',
      '不同运行期的最优配置不能拼接成一个统一方法；本轮不修改保护阈值来制造通过。','',d.t.STATUS]
    (root/'reports/GLOBAL_DEVELOPMENT_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    font=d.configure_plot();fig,axes=d.plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,dep in zip(axes,d.t.DEPS):
        sub=a[a.validation_guard.fillna(False)]
        good=sub[dep+'_no_loss'].fillna(False).astype(bool)
        for take,label,color in ((~good,'At least one protection fails','#B75975'),(good,'All Top10/20 protections retained','#14857B')):
            ax.scatter(sub.loc[take,dep+'_Mc_count_gain'],sub.loc[take,dep+'_frontend_gain'],s=14,alpha=.45,c=color,label=label)
        ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6)
        ax.set_xlabel('Summed Top10/20 Mc-pass count change');ax.set_ylabel('Summed Top10/20 official overlap change');ax.set_title(dep)
    axes[0].legend(frameon=False,fontsize=7,loc='lower left');d.savefig(fig,root,'fig_global_development_tradeoff')
    dev.json_write(root/'figures/PLOT_PROVENANCE.json',{'font':font,'points':'prelistedglobalconfigurations,adaptive realdevelopment,notindependent experiments','budgets_overlap':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();report(a.root)
