#!/usr/bin/env python3
"""Generate an honest audit report/package without selecting by real labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path('/root/autodl-tmp/gw-catalog')
STATUS='HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
ROOT=None


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)


def csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def markdown(frame,columns=None):
    if columns is not None:
        frame=frame[columns]
    def cell(value):
        if pd.isna(value):
            return 'NA'
        if isinstance(value,(float,np.floating)):
            return f'{value:.6g}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(x)for x in row)+' |'for row in frame.itertuples(index=False,name=None)])


def collect_status():
    pe=pd.read_csv(ROOT/'tables/PE_OFFICIAL_BOTH_BASELINE_GUARDS.csv')
    inj=pd.read_csv(ROOT/'tables/INJECTION_BOTH_BASELINE_GUARDS.csv')
    key=pd.read_csv(ROOT/'tables/CRITICAL_PAIR_RANK_TRACE.csv')
    pg=pe[pe.seed.astype(str)=='consensus'].groupby(['round','config']).agg(
        external_pass=('all_PE_official_nonworse','all'),external_checks=('all_PE_official_nonworse','size'))
    kr=key[(key.seed.astype(str)=='consensus')&(key['mode']=='fusion')][['round','config','rank']]
    ig=inj[inj.split.isin(['test','sept8_reused'])].groupby(['round','config']).agg(
        existing_injection_guards_pass=('existing_guard','all'),injection_checks=('existing_guard','size'),
        strict_injection_no_loss=('strict_no_loss','all'))
    joined=pg.reset_index().merge(kr,on=['round','config'],how='left',validate='one_to_one').merge(
        ig.reset_index(),on=['round','config'],how='left',validate='one_to_one')
    joined['key_outside_Top20']=joined['rank']>20
    joined['external_goal_pass']=joined.external_pass&(joined.external_checks==8)&joined.key_outside_Top20
    joined['all_summary_guards_pass']=joined.external_goal_pass&joined.existing_injection_guards_pass.fillna(False)
    current=pe[(pe.seed.astype(str)=='consensus')&pe.baseline.eq('NODUP-DIRECT-REPLAY')].groupby(['round','config']).agg(
        current_NODUP_external_pass=('all_PE_official_nonworse','all'),
        current_NODUP_external_checks=('all_PE_official_nonworse','size')).reset_index()
    joined=joined.merge(current,on=['round','config'],how='left',validate='one_to_one')
    joined['current_NODUP_external_goal_pass']=joined.current_NODUP_external_pass&(
        joined.current_NODUP_external_checks==4)&joined.key_outside_Top20
    csv(ROOT/'tables/COMBINED_GOAL_AUDIT.csv',joined)
    return joined


def figures(status):
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family='Times New Roman'if 'Times New Roman'in fonts else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family,'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    budgets=pd.read_csv(ROOT/'tables/PE_OFFICIAL_BUDGETS_CORRECTED_SEEDS.csv')
    b=budgets[(budgets.seed.astype(str)=='consensus')&(budgets.method=='fusion')&(budgets.budget==20)]
    b=b[~b.config.isin(['NODUP-DIRECT-REPLAY','PATH875-ARCHIVED'])]
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for dep,color in [('gwtc3','#2274a5'),('gwtc4','#b83a55')]:
        d=b[b.deployment==dep]
        axes[0].scatter(d.official_frontend,d.median_BC_mc,s=22,c=color,alpha=.55,label=dep)
        axes[1].scatter(d.official_hanabi,d.BC_mc_ge_0p5,s=22,c=color,alpha=.55)
    axes[0].set(xlabel='Top-20 official 1% overlap count',ylabel='Top-20 median public Mc BC')
    axes[0].legend(frameon=False)
    axes[1].set(xlabel='Top-20 published Hanabi table overlap count',ylabel='Top-20 Mc BC >= 0.5 count')
    axes[2].scatter(status['round'],status['rank'],s=20,c=np.where(status.external_goal_pass,'#2d8659','#555555'),alpha=.6)
    axes[2].axhline(20,color='#b83a55',linestyle='--',linewidth=.8)
    axes[2].set(xlabel='Development round',ylabel='Critical pair consensus rank',yscale='log')
    for ax,label in zip(axes,['a','b','c']):
        ax.text(-.08,1.04,label,transform=ax.transAxes,fontweight='bold')
    for suffix in ('pdf','png'):
        fig.savefig(ROOT/f'figures/CAMPAIGN_TRADEOFFS.{suffix}')
    plt.close(fig)
    write_json(ROOT/'contracts/PLOT_PROVENANCE.json',{'font':family,
        'not_independent_trials':True,'real_labels_not_lensing_truth':True,
        'meaning':'All exploratory arms shown;green indicates external summary criteria only,not confirmation.'})


def safe_copy(src,dest):
    if dest.exists():
        if sha(src)!=sha(dest):
            raise RuntimeError('Conflicting output:'+str(dest))
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)


def build(method_document,public_root,extra_audits):
    status=collect_status()
    ledger=pd.read_csv(ROOT/'tables/EXPERIMENT_LEDGER.csv')
    pe=pd.read_csv(ROOT/'tables/PE_OFFICIAL_BUDGETS_CORRECTED_SEEDS.csv')
    ret=pd.read_csv(ROOT/'tables/RETRIEVAL_SUMMARY_ALL_ROUNDS.csv')
    key=pd.read_csv(ROOT/'tables/CRITICAL_PAIR_RANK_TRACE.csv')
    real=pe[(pe.seed.astype(str)=='consensus')&(pe.method=='fusion')&pe.budget.isin([10,20])]
    displayed=sorted(set([26,35,44,46,47,48,49])&set(real['round'].unique()))
    show=real[real['round'].isin(displayed)].copy()
    csv(ROOT/'tables/FOCUSED_TOP10_TOP20_PE_OFFICIAL.csv',show)
    recalls=ret[ret['round'].isin(displayed)&(ret['mode']=='fusion')&ret.split.eq('sept8_reused')]
    csv(ROOT/'tables/FOCUSED_REUSED_INJECTION.csv',recalls)
    trace=key[(key.seed.astype(str)=='consensus')&(key['mode']=='fusion')&key['round'].isin(displayed)]
    csv(ROOT/'tables/FOCUSED_CRITICAL_PAIR.csv',trace)
    protected=pd.read_csv(ROOT/'tables/HISTORICAL_INPUT_HASH_VERIFICATION.csv')
    if not protected.unchanged.all():
        raise RuntimeError('Historical input changed')
    for row in protected.itertuples():
        if sha(Path(row.path))!=row.before_sha256:
            raise RuntimeError('Historical input changed after audit')
    external=status[status.external_goal_pass][['round','config']].to_dict('records')
    current=status[status.current_NODUP_external_goal_pass][['round','config']].to_dict('records')
    full=status[status.all_summary_guards_pass][['round','config']].to_dict('records')
    complete=[];artifact_sources=[]
    for row in ledger[ledger.exists].itertuples():
        source=Path(row.path);alias=source.name
        completed={}
        for marker in ('EVALUATION_COMPLETE','REAL_COMPLETE','PREDICTIVE_FROZEN','PILOT_COMPLETE','COVARIATES_COMPLETE'):
            path=source/f'contracts/{marker}.json'
            if path.exists():
                completed[marker]=json.loads(path.read_text())
        complete.append({'round':int(row.round),'path':row.path,'receipts':completed})
        for folder in ('contracts','configs','calibration','tables','audit','scripts','logs'):
            for src in sorted((source/folder).rglob('*'))if(source/folder).exists()else[]:
                if not src.is_file()or src.suffix in ('.npy','.npz','.pt','.h5','.hdf5')or src.stat().st_size>50*1024*1024:
                    continue
                dest=ROOT/'round_artifacts'/alias/src.relative_to(source)
                safe_copy(src,dest)
                artifact_sources.append({'source':str(src),'destination':str(dest.relative_to(ROOT)),
                    'sha256':sha(src),'bytes':src.stat().st_size})
        for src in sorted(source.glob('results/*/gwtc*/consensus/*_all_pairs.parquet')):
            dest=ROOT/'real_consensus'/alias/src.relative_to(source/'results')
            safe_copy(src,dest)
            artifact_sources.append({'source':str(src),'destination':str(dest.relative_to(ROOT)),
                'sha256':sha(src),'bytes':src.stat().st_size})
        for src in sorted(source.glob('results/*/gwtc*/seed_*/real/fusion_all_pairs.parquet')):
            a=pd.read_parquet(src).sort_values('rank',kind='mergesort').head(100)
            rel=src.relative_to(source/'results')
            csv(ROOT/'real_seed_top100'/alias/rel.with_suffix('.csv'),a)
        for src in sorted(source.glob('results/*/gwtc*/seed_*/*/*query_ranks.parquet')):
            safe_copy(src,ROOT/'injection_query_ranks'/alias/src.relative_to(source/'results'))
    csv(ROOT/'manifest/PACKAGED_SOURCE_ARTIFACTS.csv',pd.DataFrame(artifact_sources))
    write_json(ROOT/'contracts/ALL_ROUND_COMPLETION_RECEIPTS.json',complete)
    safe_copy(method_document,ROOT/'reports/INDEPENDENT_CALIBRATION_METHOD_CN.md')
    for folder in ('tables','contracts','manifest','figures','data'):
        for src in sorted((public_root/folder).rglob('*'))if(public_root/folder).exists()else[]:
            if src.is_file()and src.stat().st_size<10*1024*1024:
                safe_copy(src,ROOT/'critical_public_pe'/src.relative_to(public_root))
    for source in extra_audits:
        for folder in ('contracts','tables','scripts','manifest'):
            for src in sorted((source/folder).rglob('*'))if(source/folder).exists()else[]:
                if src.is_file()and src.stat().st_size<50*1024*1024:
                    safe_copy(src,ROOT/'additional_audits'/source.name/src.relative_to(source))
    # Preserve the compact pair scores needed for paired uncertainty calculations.
    finalroot=P/'results/mcwf_nodup_paired_quality_backoff_49_20260909T160024Z'
    primary='PAIRED-QUALITY-GLOBAL-BACKOFF'
    final_methods=('NODUP-DIRECT-REPLAY','PATH875-ARCHIVED',primary)
    for method in final_methods:
        for src in sorted((finalroot/'results'/method).glob('gwtc*/seed_*/*/pairs.parquet')):
            frame=pd.read_parquet(src)
            columns=[c for c in ('idx_i','idx_j','is_true_pair','true_pair_family','event_count',
                'waveform_score','final_score','time_score','sky_raw_log_bf')if c in frame]
            dest=ROOT/'focused_final/pair_scores'/method/src.relative_to(finalroot/'results'/method)
            dest.parent.mkdir(parents=True,exist_ok=True)
            frame[columns].to_parquet(dest,index=False)
        for dep in ('gwtc3','gwtc4'):
            src=finalroot/f'results/{method}/{dep}/consensus/fusion_all_pairs.parquet'
            frame=pd.read_parquet(src).sort_values('consensus_rank',kind='stable')
            safe_copy(src,ROOT/f'focused_final/real_pairs/{method}/{dep}/all_pairs.parquet')
            for budget in (10,20,50,100):
                csv(ROOT/f'focused_final/real_pairs/{method}/{dep}/Top{budget}.csv',frame.head(budget))
    focused=real[(real['round']==49)&real.config.isin(final_methods)]
    csv(ROOT/'focused_final/TOP10_TOP20_COMPARISON.csv',focused)
    all_budgets=pe[(pe['round']==49)&pe.config.isin(final_methods)&pe.method.eq('fusion')]
    csv(ROOT/'focused_final/ALL_BUDGETS_ALL_SEEDS.csv',all_budgets)
    csv(ROOT/'focused_final/ALL_INJECTION_SUMMARIES.csv',ret[(ret['round']==49)&ret.method.isin(final_methods)])
    focused_trace=key[(key['round']==49)&key.config.isin(final_methods)]
    csv(ROOT/'focused_final/CRITICAL_PAIR_ALL_RANKS.csv',focused_trace)
    safe_copy(Path(__file__),ROOT/'scripts/campaign_deliverables.py')
    figures(status)
    document=f'''# NODUP 关键质量不一致：完整探索审计

状态：`{STATUS}`。本轮不覆盖历史结果，不自动采纳，也不修改论文。

## 1. 实际结论

本包记录了 {len(ledger)} 个独立尝试或重试目录。相对当前 NODUP 满足两个运行期共识 Top-10/20 的全部既定 PE/官方摘要约束、且关键 pair 退出 Top-20 的配置：{json.dumps(current,ensure_ascii=False)}。

进一步要求相对 NODUP 与更早 PATH875 两个基线都满足上述外部摘要约束的配置：{json.dumps(external,ensure_ascii=False)}。

在上述条件之外还通过本报告全部既有注入摘要 guard 的配置：{json.dumps(full,ensure_ascii=False)}。

空列表表示没有达到，不是遗漏结果。摘要 guard 通过仍不等于新的独立确认；逐 seed、validation 和 bootstrap 还需分别阅读。不能仅凭关键 pair 排名下降宣布完成，更不能把不同方法的 O3/O4a 拼在一起。

### 本次 NODUP 基础上的最小回退对照

`R49 / PAIRED-QUALITY-GLOBAL-BACKOFF` 是这里重点交付的对照，不是自动升级：关键 pair 共识 rank 6 -> 28；三训练种子实际排名为 26、15、53，所以只能说它退出共识 Top-20，不能说每个 seed 都退出 Top-20。公开 BC 仍为原值。

相对当前 NODUP，O3 Top-10 的 Mc BC>=0.5 从 9/10 到 10/10，Dmax<=3 从 9/10 到 10/10，官方1%重合保持7，公开Hanabi表重合保持6。O3 Top-20 的 Mc BC>=0.5 从16到18，Dmax<=3从19到20，官方重合10及Hanabi重合8保持不变。O4a 共识Top-10/20的这些PE/官方摘要均与NODUP相同。

旧注入 test 的所有已报告指标与 NODUP 完全一致；reused injection 的三通道平均 R@10 从 O3 .964286 到 .959524、O4a .914286 到 .911111，而平均 AUPRC 从 .527731 到 .532671、.404479 到 .407496。少量 recall 损失没有被隐去，逐模型/目录 guard 也并非全部通过，故不能声称严格逐项不下降或稳定正式升级。

R48 GLOBAL 的共识 Top-10/20 PE 更好，关键 rank=22，但旧 O4a test AUPRC 从 .303669 降到 .295531；因此保留为另一项完整对照，不用其真实名单优势掩盖注入退化。R49 没有恢复 NODUP 相对 PATH875 的全部既有差距，两个历史参考都保留。

{markdown(focused,['config','deployment','budget','BC_mc_ge_0p5','median_BC_mc','catastrophic_mc','Dmax_le_3','official_frontend','official_hanabi'])}

## 2. 关键 pair 到底哪里不一致

对 GW191103_012549--GW191105_143521，冻结公开 `C01:IMRPhenomXPHM` posterior group 独立复算：探测器系 Mc 中位数分别为 9.975781 和 9.573048，标准化间隔 D_Mc=3.969372。原 256-bin BC=0.091665，最小密度重叠约 0.018697。原 NODUP rank=6，PATH875 rank=34。

公开 BC 不会因检索改动而变大；需要改变的是对该波形对的一致性评价。100/256/512/1024 bins 的 BC 约为 .1023/.0917/.0810/.0650，KDE BC 约 .1185。BC=.1 的单个标签有估计器敏感性，因此同时报告 D 和完整后验图；本轮没有换估计器来人为消除失败。

该 pair 的官方 PO FPP 约 .003802752，ML FPP 约 .1653010041，且列入公开 Hanabi 表，但公开分析未偏好透镜。官方前端重合不是透镜真值。

## 3. 主要对照与关键排名

这些是提前存在的对照组及最后几项探索，不是只列最佳结果；全体见 `tables/`。

{markdown(trace,['round','config','rank','waveform_score_mean','final_score_mean'])}

{markdown(show,['round','config','deployment','budget','BC_mc_ge_0p5','median_BC_mc','catastrophic_mc','Dmax_le_3','official_frontend','official_hanabi'])}

## 4. 注入检索

已有评估目录被反复查看，统一标记 reused evaluation，不再称为未开封 locked test。所有方法均保存 R@1/R@10、AUPRC、F50/F90、逐训练 seed 和目录 seed 数据。这里列三通道均值，完整均值、SD 和 waveform-only 见 CSV。

{markdown(recalls,['round','method','deployment','macro_r_at_1_mean','macro_r_at_10_mean','average_precision_mean','false_at_recall_0p5_mean','false_at_recall_0p9_mean'])}

既有 guard 是 R@10 下降不超过 .02、AUPRC 下降不超过 .005、F50/F90 不超过基线的 1.10 倍；严格逐项不下降也另列。这些门槛没有按真实结果放宽。不能把某个单指标改善写成全面改善。

## 5. 方法边界

- 不恢复旧 encoder 显式 Mc/q 回归差评分，不恢复 .875 旧/新总分混合。
- 时间和天空的原始输入、分数、历史 scope 均冻结。
- R32 是明确独立的 validation-only 外层权重对照；其余重点后续对照冻结外层权重。R39/R42 的部分臂改变波形内部 gamma/beta，不等于外层 time/sky 改动。
- 物理 profile 和模拟标定的预测分布，不是 full PE。R43 的分频统计在已处理序列上计算，不是正式搜索 chi-square p-value。
- R47 的下包络不相加两个质量证据，也不混合总分；它仍是相关分数的保守排序规则，不是规范化 Bayes factor。其旧物理误差参考只有约 38--39 个 fit 源，限制不能隐藏。
- R48 只在旧经验真伴随尾部<.05时削弱分数。R49 又规定双端profile不合格时逐pair精确回退NODUP；它没有给某个真实event ID设置例外。外部历史合同中 inherited lower-envelope 的诊断应结合新 backoff addendum 阅读，最终规则由独立公式复算审计验证。
- 真实反馈已经影响研究方向和方案比较，因此本轮属于自适应开发。事件 ID、公开 PE 和官方标签没有进入预测输入或模拟拟合损失，不足以把本轮称为盲测确认。
- 没有重新运行 Hanabi，没有确认透镜，没有伪造新的真实标签。

## 6. 数据与统计单位

新校准每运行期 1,024 个源、2,048 幅像；生成前按源分成 512 fit 和 512 tune。64 个 256 s 噪声块来自 16 个父时间块，每个分区占 8 个父块。源数、噪声块数、独立透镜环境数不能混为一谈。新源样本不等于新的独立 GW-LMC lens population。每源两幅像不跨分区，source/noise-parent 交叉审计为零。

现有 system bootstrap 以源为单位共同抽两幅像；只反映相应固定模型和噪声条件下的波动。R26 的 O4a AUPRC 差值区间仍为负，故不能只引用它的 O3 PE 改善。每 seed 完整 PE/官方预算表的历史标签问题已在本包重新按实际 seed 计算，旧表未覆盖。

R48、R49 分别完成10,000次query源级重抽样和2,000次pair指标重抽样；摘要及逐模型区间在additional_audits。固定同源两像成组，shared catalog在所有方法/训练模型中使用同一抽样权重。历史test跨模型的global-source映射尚不完备，所以不伪造合并置信区间，只报逐模型区间。该CI不包含整轮方法搜索的不确定度。

本轮逐pair对齐审计覆盖72个评分面板：time、sky、embedding输入、time/sky贡献差值严格为0；PE与官方字段不变；最终分数和R48/R49规则复算误差不超过1e-12。全部这些属于代码/数据完整性检查，不等于统计验收通过。

## 7. 文件与复现

服务器：`connect.westd.seetacloud.com:32328`，项目：`/root/autodl-tmp/gw-catalog`。包内不包含登录口令或密钥。

- `tables/EXPERIMENT_LEDGER.csv`：全部尝试，包括失败和重试。
- `tables/COMBINED_GOAL_AUDIT.csv`：相对两个基线逐项检查。
- `tables/PE_OFFICIAL_BUDGETS_CORRECTED_SEEDS.csv`：全部 seed 及 consensus 的 Top-10/20/50/100。
- `real_consensus/`：所有已完成探索臂的完整真实 pair 表。
- `real_seed_top100/`：逐 seed Top-100，含公开 PE 与官方阶段字段。
- `injection_query_ranks/`：用于源级 retrieval bootstrap 的 query ranks。
- `round_artifacts/`：各轮合同、配置、校准、表格、脚本和日志。
- `critical_public_pe/`：关键 pair 的冻结组复算、敏感性、样本摘要和图。
- `focused_final/`：NODUP、PATH875与R49的完整共识pair表、Top-10/20/50/100、三seed预算及紧凑注入分数。
- `additional_audits/`：R48/R49源级CI、成对差值CI及独立规则/通道不变性检查。
- `reports/INDEPENDENT_CALIBRATION_METHOD_CN.md`：公式、依据、适用范围及不能据此声称的内容。

历史保护清单 {len(protected)} 个文件在本次打包前全部保持原 SHA-256。未删除原 strain、PE、checkpoint 或失败输出。大型波形、模型与缓存未重复打包，原路径和哈希保留用于复现。

## 8. 科学解释

本轮的正、负结果都应保留。只有在严格区分开发和确认、重新进行独立验证后，才可讨论正式方法更新。即使某个探索臂满足当前真实名单的摘要目标，也不能保证未来目录 PE 一致性，不能将官方候选重合率解释为真实检出率。
'''
    report=ROOT/'reports/FINAL_EXPLORATION_REPORT_CN.md'
    report.write_text(document,encoding='utf-8')
    write_json(ROOT/'contracts/DELIVERABLE_STATUS.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'status':STATUS,'external_summary_goal_candidates':external,'all_summary_guard_candidates':full,
        'current_NODUP_external_summary_goal_candidates':current,
        'automatic_adoption':False,'blind_confirmation':False,'historical_hash_failures':0,
        'not_a_goal_success_declaration':True,'raw_strain_PE_checkpoints_not_packaged':True})


def package():
    if not (ROOT/'contracts/DELIVERABLE_STATUS.json').exists():
        raise RuntimeError('Build report first')
    if shutil.disk_usage(ROOT).free<20*1024**3:
        raise RuntimeError('HOLD_DISK_LIMIT')
    # Detect actual PEM headers, not references to credentials in documentation.
    private=re.compile(rb'-{5}BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-{5}')
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file()or p.is_symlink():
            continue
        if p.name in ('SHA256SUMS.txt','OUTPUT_SHA256.csv','ARCHIVE_VERIFICATION.json'):
            continue
        with p.open('rb')as stream:
            if private.search(stream.read(50*1024*1024)):
                raise RuntimeError('Private-key material detected in '+str(p))
        if any(part in ('.ssh','.ssh_mux')for part in p.relative_to(ROOT).parts):
            raise RuntimeError('Credential directory included')
        files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
    csv(ROOT/'manifest/OUTPUT_SHA256.csv',pd.DataFrame(files))
    with (ROOT/'manifest/SHA256SUMS.txt').open('x',encoding='utf-8')as f:
        f.write(''.join(row['sha256']+'  '+row['path']+'\n'for row in files))
    dest=P/'packages'/f'{ROOT.name}_deliverables.tar.gz'
    if dest.exists():
        raise RuntimeError('Do not overwrite package')
    with tarfile.open(dest,'w:gz',compresslevel=6)as archive:
        archive.add(ROOT,arcname=ROOT.name,recursive=True)
    digest=sha(dest)
    with dest.with_suffix(dest.suffix+'.sha256').open('x',encoding='utf-8')as f:
        f.write(digest+'  '+dest.name+'\n')
    expected={ROOT.name+'/'+row['path']:row['sha256']for row in files}
    seen=set();count=0
    with tarfile.open(dest,'r:gz')as archive:
        for member in archive:
            if not member.isfile():
                continue
            count+=1
            if member.name in expected:
                h=hashlib.sha256();stream=archive.extractfile(member)
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):
                    h.update(chunk)
                if h.hexdigest()!=expected[member.name]:
                    raise RuntimeError('Archive internal hash failure')
                seen.add(member.name)
    if seen!=set(expected):
        raise RuntimeError('Archive missing manifest members')
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'archive':str(dest),'sha256':digest,
        'bytes':dest.stat().st_size,'archive_files':count,'hashed_files':len(expected),
        'internal_hash_failures':0,'missing_files':0,'status':STATUS}
    write_json(ROOT/'manifest/ARCHIVE_VERIFICATION.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('build','package'),required=True)
    parser.add_argument('--method-document',type=Path);parser.add_argument('--public-root',type=Path)
    parser.add_argument('--extra-audit',type=Path,action='append',default=[])
    args=parser.parse_args();ROOT=args.root
    if args.stage=='build':
        build(args.method_document,args.public_root,args.extra_audit)
    else:
        package()
