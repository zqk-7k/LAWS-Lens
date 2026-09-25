#!/usr/bin/env python3
"""Focused human-readable summary, figures, and runtime dependency snapshot."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import campaign_deliverables as io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bootstrap-summary', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    marker = root/'contracts/FOCUSED_READOUT_COMPLETE.json'
    if marker.exists():
        raise RuntimeError('Do not overwrite readout')
    primary = 'PAIRED-QUALITY-GLOBAL-BACKOFF'
    current = 'NODUP-DIRECT-REPLAY'
    earlier = 'PATH875-ARCHIVED'
    pe = pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS_CORRECTED_SEEDS.csv')
    focus = pe[(pe['round']==49)&pe.method.eq('fusion')&pe.config.isin([earlier,current,primary])]
    pe_summary = focus[focus.seed.astype(str).eq('consensus')&focus.budget.isin([10,20])]
    ret = pd.read_csv(root/'tables/RETRIEVAL_SUMMARY_ALL_ROUNDS.csv')
    ret = ret[(ret['round']==49)&ret['mode'].eq('fusion')&ret.method.isin([current,primary])]
    ci = pd.read_csv(args.bootstrap_summary/'tables/PAIRED_METHOD_DELTA_CI.csv')
    ci = ci[ci.seed.eq('three_fixed_models')&ci['mode'].eq('fusion')&ci.baseline.eq(current)&ci.method.eq(primary)
        &ci.quantity.isin(['R1','R10','average_precision','F50','F90'])]
    io.csv(root/'focused_final/PAIRED_DELTA_CI.csv',ci)
    paragraphs = [
        '# MCWF-PROFILE49：NODUP 基础上的物理波形回退对照',
        '', '状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。不覆盖或自动替换任何历史版本。',
        '', '## 结论边界',
        '', '**共识 Top-10/20 相对当前 NODUP 的目标达到；逐 seed、所有 Top-N、全部注入 guard 和相对早期 PATH875 的全面非劣尚未达到。**',
        '', '关键 pair `GW191103_012549--GW191105_143521` 从 NODUP 共识第6名降到第28名。公开 Mc BC 仍是0.091665；PE没有被修改。三个模型分别为第26、15、53名，仍有一个模型将其放在Top-20。',
        '', 'O3 共识 Top-10 的 Mc BC>=0.5 和 Dmax<=3 均从9/10升到10/10；官方1%重合仍7/10，公开Hanabi表重合仍6/10。Top-20 Mc合格数16->18，Dmax合格数19->20，官方重合10和Hanabi重合8不变。O4a共识Top-10/20上述指标均保持当前NODUP水平。',
        '', '**不能把共识结果说成全部seed都不变差：** O3模型202607241的Top-20官方/Hanabi重合各少1，202607243的Top-20官方重合少1。完整逐seed预算表都在包内。官方候选重合不是真实透镜标签，公开Hanabi重合不表示Hanabi支持透镜。',
        '', '## 修改了什么',
        '', '最终R49保留当前encoder和网络预测，增加波形派生的物理模板profile可靠性检查。只有两端profile都合格时，才使用模拟标定的质量/内禀一致性专家；经验真伴随尾部<0.05时，选择更保守的波形一致性分数。其他pair逐项回到NODUP。',
        '', '没有恢复旧encoder的显式Mc/q回归差及其校准，没有恢复0.875旧/新总分混合。time、sky、外层权重不变。R49在有效子集可升可降，不能误称为对所有NODUP分数只扣分。该回退和minimum是探索性排序规则，不是规范化Bayes factor。',
        '', 'R48使用同一物理检查，但无profile时仍重新标定波形。其真实共识PE更好，但旧O4a注入AUPRC下降，故作为独立完整对照保留，不用真实候选优势掩盖其代价。',
        '', '## 共识PE与官方预算', '',
        io.markdown(pe_summary,['config','deployment','budget','BC_mc_ge_0p5','median_BC_mc','Dmax_le_3','official_frontend','official_hanabi']),
        '', '## 注入结果', '',
        io.markdown(ret,['deployment','method','split','macro_r_at_1_mean','macro_r_at_10_mean','macro_r_at_10_std','average_precision_mean','false_at_recall_0p5_mean','false_at_recall_0p9_mean']),
        '', '旧test的分数和指标精确保持NODUP。reused injection平均AUPRC小幅提高，R@10小幅下降。R49在O3/O4a的validation三通道guard分别为3/3、2/3；reused的9个模型-目录面板分别为6/9、8/9。不能据此宣布完整validation Gate通过。',
        '', '以下是相对NODUP的源级bootstrap差值区间，未包含方法搜索偏差或独立噪声总体不确定度：', '',
        io.markdown(ci,['deployment','quantity','nominal_delta','delta_q025','delta_q975']),
        '', '## 科学限制',
        '', '真实反馈影响了研究方向，属于自适应开发，不是新的blind confirmation。没有用PE或官方标签训练模型，并不消除反复选择方案产生的评估偏差。若要正式升级，仍需预冻结后的独立确认。没有运行Hanabi或full BBH PE。',
        '', 'R49物理误差参考包含仅约38--39个fit源的旧小样本标定，新扩充误差校准也有近似与有限支持问题。它不会保证未来每个高分pair都通过PE，也不能保证所有预算下官方重合增加。',
        '', '## 文件入口',
        '', '- `reports/FINAL_EXPLORATION_REPORT_CN.md`：全部53次尝试/重试的完整审计。',
        '- `reports/INDEPENDENT_CALIBRATION_METHOD_CN.md`：逐步公式、数据、参考文献与不能据此声称的内容。',
        '- `focused_final/real_pairs/`：NODUP、PATH875和R49完整真实pair表及Top-10/20/50/100。',
        '- `focused_final/ALL_BUDGETS_ALL_SEEDS.csv`：每个训练seed与共识的PE、官方预算。',
        '- `focused_final/pair_scores/`：紧凑注入pair分数，可复算AUPRC和假对负担。',
        '- `additional_audits/`：R48/R49 bootstrap、72个面板的严格通道不变性审计。',
        '- `critical_public_pe/`：关键pair的公开PE样本摘要、图和独立复算。',
        '- `scripts/runtime_dependencies/`：本机项目内运行依赖快照；不包含原始strain和大型模型。',
        '', '服务器：`connect.westd.seetacloud.com:32328`。项目：`/root/autodl-tmp/gw-catalog`。此包不含凭据。',
    ]
    with (root/'README_CN.md').open('x',encoding='utf-8')as file:
        file.write('\n'.join(paragraphs)+'\n')
    topdocs=[]
    for dep in ('gwtc3','gwtc4'):
        path=root/f'focused_final/real_pairs/{primary}/{dep}/Top10.csv'
        table=pd.read_csv(path)
        cols=['consensus_rank','event_i','event_j','final_score_mean',
              'waveform_contribution_mean','time_contribution_mean','sky_contribution_mean',
              'pe_mc_bhattacharyya_coefficient','pe_dmax_intrinsic']
        cols += [c for c in ('official_po_fpp','official_ml_fpp','official_phazap_fpp',
            'official_any_pair_resolved_hanabi_overlap','official_hanabi_conclusion','official_screening_stage')if c in table]
        cols=[c for c in cols if c in table]
        topdocs.extend(['## '+dep,'',io.markdown(table,cols),''])
    with (root/'focused_final/TOP10_CN.md').open('x',encoding='utf-8')as file:
        file.write('# R49完整Top-10审计\n\n公开阶段仅作对照，不表示透镜确认。\n\n'+'\n'.join(topdocs))
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    colors={'gwtc3':'#2274a5','gwtc4':'#b83a55'}
    for dep,color in colors.items():
        for method,style in [(current,'--'),(primary,'-')]:
            sub=pe_summary[pe_summary.deployment.eq(dep)&pe_summary.config.eq(method)].sort_values('budget')
            axes[0,0].plot(sub.budget,sub.BC_mc_ge_0p5,style+'o',color=color,label=dep+' '+('NODUP'if method==current else'R49'))
            axes[0,1].plot(sub.budget,sub.official_frontend,style+'o',color=color)
    axes[0,0].set(xlabel='Consensus shortlist budget',ylabel='Public Mc BC >= 0.5 count',xticks=[10,20])
    axes[0,1].set(xlabel='Consensus shortlist budget',ylabel='Official 1% overlap count',xticks=[10,20])
    axes[0,0].legend(frameon=False,fontsize=8)
    for ax,quantity,title in [(axes[1,0],'R10','Paired change in R@10'),(axes[1,1],'average_precision','Paired change in AUPRC')]:
        for j,(dep,color)in enumerate(colors.items()):
            row=ci[ci.deployment.eq(dep)&ci.quantity.eq(quantity)].iloc[0]
            ax.plot([row.delta_q025,row.delta_q975],[j,j],color=color,linewidth=2)
            ax.scatter([row.nominal_delta],[j],color=color,s=28)
        ax.axvline(0,color='#555555',linestyle=':',linewidth=.8)
        ax.set(xlabel=title,ylabel='Reused injection',yticks=[0,1],yticklabels=['O3','O4a'],ylim=(-.5,1.5))
    for ax,letter in zip(axes.flat,['a','b','c','d']):
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.08,1.04,letter,transform=ax.transAxes,fontweight='bold')
    for suffix in ('pdf','png'):
        fig.savefig(root/f'figures/PROFILE49_FOCUSED_COMPARISON.{suffix}',dpi=180)
    plt.close(fig)
    importlib.import_module('mcwf_paired_quality_backoff_20260909')
    dependencies=[]
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:
            continue
        path=Path(path).resolve()
        if path.suffix=='.py'and path.is_relative_to(P/'scripts'):
            dest=root/'scripts/runtime_dependencies'/path.relative_to(P/'scripts')
            io.safe_copy(path,dest)
            dependencies.append({'source':str(path),'destination':str(dest.relative_to(root)),'sha256':io.sha(path)})
    io.csv(root/'manifest/RUNTIME_DEPENDENCIES.csv',pd.DataFrame(dependencies).drop_duplicates('source'))
    versions=[]
    for name in ('numpy','scipy','pandas','pyarrow','torch','scikit-learn','pycbc','lalsuite','matplotlib','bilby'):
        try:
            version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version='not installed'
        versions.append({'package':name,'version':version})
    io.csv(root/'manifest/PYTHON_VERSIONS.csv',pd.DataFrame(versions))
    io.safe_copy(Path(__file__),root/'scripts/final_profile_readout.py')
    io.write_json(marker,{'primary_comparison':'R49 versus currentNODUP,not automatic adoption',
        'consensus_top10_top20_PE_official_no_loss':True,'all_seeds_all_budgets_no_loss':False,
        'all_injection_guards_pass':False,'new_independent_confirmation':False,
        'status':io.STATUS,'project_runtime_modules':len(dependencies),
        'source_bootstrap_queries':10000,'source_bootstrap_pair_draws':2000})
    print('FOCUSED_READOUT_COMPLETE',len(dependencies),flush=True)


if __name__=='__main__':
    main()
