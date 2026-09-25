#!/usr/bin/env python3
"""Paired source-bootstrap and descriptive real-catalog comparisons."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
t,dev,cf=ev.t,ev.dev,ev.cf


def pair_metrics(scores,truth,weights):
    order=np.argsort(-scores,kind='stable')
    yy=truth[order];ww=np.asarray(weights[order],dtype=float)
    tp=np.cumsum(yy*ww);fp=np.cumsum((~yy)*ww)
    ends=np.r_[np.flatnonzero(np.diff(scores[order])!=0),len(order)-1]
    t=tp[ends];f=fp[ends];delta=np.diff(np.r_[0.,t])
    ap=float(np.sum(delta*np.divide(t,t+f,out=np.zeros_like(t),where=(t+f)>0))/t[-1])
    return {'average_precision':ap,
            'false_at_recall_0p5':float(fp[min(np.searchsorted(tp,.5*tp[-1]),len(tp)-1)]),
            'false_at_recall_0p9':float(fp[min(np.searchsorted(tp,.9*tp[-1]),len(tp)-1)])}


def boot_task(args):
    root,dep,seed,method=args;root=Path(root)
    f=pd.read_parquet(root/f'evaluation/{method}/{dep}/seed_{seed}/test_pairs.parquet')
    baseline=cf.read(dep,seed,'test');plan=dev.BASE.retained_event_plan(dep,seed,'test')
    ids=plan.set_index('idx').system_id
    names=sorted(ids.unique());mapping={g:i for i,g in enumerate(names)}
    a=ids.loc[f.idx_i].map(mapping).to_numpy(int);b=ids.loc[f.idx_j].map(mapping).to_numpy(int)
    y=f.is_true_pair.to_numpy(bool)
    strata={g:'background' for g in range(len(names))}
    for g,family in zip(a[y],f.loc[y,'true_pair_family']):strata[g]=str(family)
    groups=[np.array([g for g in strata if strata[g]==kind]) for kind in sorted(set(strata.values()))]
    rng=np.random.default_rng(202609859+seed%10000)
    draws=np.zeros((1000,len(names)),int)
    for row in draws:
        for gs in groups:row+=np.bincount(rng.choice(gs,len(gs),replace=True),minlength=len(names))
    rows=[]
    for mode in ('fusion','waveform'):
        score=f.final_score.to_numpy(float) if mode=='fusion' else f.waveform_score.to_numpy(float)
        old=cf.channels(baseline,baseline.waveform_score)@cf.frozen_weights(dep,seed) if mode=='fusion' else baseline.waveform_score.to_numpy(float)
        positive=np.flatnonzero(y)
        family=f.true_pair_family.to_numpy(str)[positive]
        query_hits=[]
        ii,jj=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
        n=int(f.event_count.iloc[0])
        for values in (score,old):
            matrix=np.full((n,n),-np.inf);matrix[ii,jj]=values;matrix[jj,ii]=values
            ri=1+(matrix[ii[positive]]>values[positive,None]).sum(1)
            rj=1+(matrix[jj[positive]]>values[positive,None]).sum(1)
            query_hits.append({k:((ri<=k).astype(float)+(rj<=k).astype(float))/2 for k in (1,10)})
        for k in (1,10):
            delta=query_hits[0][k]-query_hits[1][k]
            samples=np.zeros(10000);point=0.
            kinds=sorted(set(family))
            qrng=np.random.default_rng(202609870+seed%10000+k)
            for kind in kinds:
                use=family==kind;v=delta[use]
                samples+=v[qrng.integers(0,len(v),size=(10000,len(v)))].mean(1)/len(kinds)
                point+=v.mean()/len(kinds)
            rows.append({'method':method,'deployment':dep,'seed':seed,'mode':mode,'metric':f'macro_r_at_{k}',
                         'delta':point,'CI_low':float(np.quantile(samples,.025)),'CI_high':float(np.quantile(samples,.975)),
                         'replicates':10000,'unit':'lensed system with both directed queries;family stratified;fixed candidate catalog',
                         'interpretation':'conditional development-sample uncertainty,not correction for adaptive real selection'})
        for values in (score,old):
            fast=pair_metrics(values,y,np.ones(len(y)))
            ref=cf.fast_metrics(f,values)
            if any(abs(fast[k]-ref[k])>1e-10 for k in fast):raise RuntimeError('Weighted AP/F50/F90 implementation does not replay original')
        samples={k:[] for k in ('average_precision','false_at_recall_0p5','false_at_recall_0p9')}
        for draw in draws:
            weights=draw[a]*draw[b];weights[y]=draw[a[y]]
            mm=pair_metrics(score,y,weights);bb=pair_metrics(old,y,weights)
            for k in samples:samples[k].append(mm[k]-bb[k])
        for k,v in samples.items():
            rows.append({'method':method,'deployment':dep,'seed':seed,'mode':mode,'metric':k,
                         'delta':pair_metrics(score,y,np.ones(len(y)))[k]-pair_metrics(old,y,np.ones(len(y)))[k],
                         'CI_low':float(np.quantile(v,.025)),'CI_high':float(np.quantile(v,.975)),
                         'replicates':1000,'unit':'source-system stratified;shared noise not independently resampled',
                         'interpretation':'conditional development-sample uncertainty,not correction for adaptive real selection'})
    return rows


def bootstrap(root):
    path=root/'tables/PAIRED_SOURCE_BOOTSTRAP.csv'
    if path.exists():return
    methods=sorted(set(pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv').method)-{'OMC'})
    tasks=[(str(root),dep,seed,m) for dep in t.DEPS for seed in t.SEEDS for m in methods]
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for task,result in zip(tasks,pool.map(boot_task,tasks)):
            rows.extend(result);print('BOOTSTRAPPED',task,flush=True)
    dev.csv_write(path,pd.DataFrame(rows))


def report(root):
    metrics=pd.read_csv(root/'tables/RETRIEVAL_SUMMARY.csv')
    a=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    consensus=a[(a.seed.astype(str)=='consensus')&(a.method=='fusion')]
    dev.csv_write(root/'tables/CONSENSUS_TOP10_TOP20.csv',consensus[consensus.budget.isin([10,20])])
    target=pd.read_csv(root/'tables/DEVELOPMENT_TARGET_CHECK.csv')
    training=[]
    for p in sorted((root/'models').glob('*/*/*/COMPLETE.json')):
        value=json.loads(p.read_text());value.update(kind=p.parts[-4],deployment=p.parts[-3],slot=p.parts[-2])
        training.append(value)
    dev.csv_write(root/'tables/TRAINING_SUMMARY.csv',pd.DataFrame(training))
    lines=['# MCWF-TEMPORAL-RESPONSE-01 开发比较记录','',
           '本文件记录一个机制对照，不将计算完成等同于用户目标达成。历史 OMC、时间、天空和外层权重保持不变。',
           '真实目录曾多次参与开发反馈，因此本轮真实 PE/官方对照不是盲测。官方候选不是透镜真值，未运行新的 Hanabi。','',
           '## 方法','',
           'CONTINUE 仅续训 OMC；TEMPORAL 加入模板匹配峰附近的时间形状残差。两运行期使用相同模型、优化规则、模拟校准与外层权重。',
           '质量预测分布不是 PE 后验，新波形加权和不是透镜 Bayes factor。时间形状残差只作为特征，没有硬否决。',
           '具体公式、引用、数据、参数及统计边界见 PROTOCOL_CN.md 和 contracts/ANALYSIS_CONTRACT.json。','',
           '## 注入比较','',
           '| Run | 方法 | R@1 | R@10 | AUPRC | F50 | F90 |',
           '|---|---|---:|---:|---:|---:|---:|']
    mm=metrics[(metrics.split=='test')&(metrics['mode']=='fusion')]
    for r in mm.itertuples():
        lines.append(f'|{r.deployment}|{r.method}|{r.macro_r_at_1_mean:.4f} +/- {r.macro_r_at_1_std:.4f}|{r.macro_r_at_10_mean:.4f} +/- {r.macro_r_at_10_std:.4f}|{r.average_precision_mean:.4f}|{r.false_at_recall_0p5_mean:.1f}|{r.false_at_recall_0p9_mean:.1f}|')
    lines+=['','## 真实 Top-10 和 Top-20','',
            '| Run | 方法 | Top-B | Mc BC>=0.5 | Mc BC中位数 | 灾难性Mc | Dmax<=3 | 官方1%前端 | 公开Hanabi重合 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in consensus[consensus.budget.isin([10,20])].itertuples():
        lines.append(f'|{r.deployment}|{r.config}|{r.budget}|{r.BC_mc_ge_0p5}|{r.median_BC_mc:.4f}|{r.catastrophic_mc}|{r.Dmax_le_3}|{r.official_frontend}|{r.official_hanabi}|')
    lines+=['','## 目标逐运行期检查','',
            '| 方法 | Run | Top10/20全部保护项 | PE增加 | 官方前端增加 | 注入逐seed保护 | 该run达标 |',
            '|---|---|---|---|---|---|---|']
    for r in target.itertuples():
        lines.append(f'|{r.configuration}|{r.deployment}|{r.no_loss_top10_top20}|{r.PE_gain}|{r.official_gain}|{r.all_seed_injection_guard}|{r.run_target_pass}|')
    winners=target.groupby('configuration').run_target_pass.all()
    lines+=['','## 后续判定','',
            '开发检查通过、需要独立注入确认的配置：'+(', '.join(winners[winners].index) or '无。不能将本轮称为共同改善。'),
            '如果没有共同通过配置，保留全部负结果并继续不同机制；不得把两个运行期各自最好的不同方法拼成统一主结果。',
            '如果存在共同通过配置，在新的源和噪声上冻结验证前，也不能标记总目标完成。','',
            '## 不确定度与限制','',
            '逐seed标准差衡量这批初始化和运行的离散程度；1000次分层source-system bootstrap保留同系统两幅像的相关性，但不独立重采样共享噪声块。它不能修正真实候选反复开发的选择偏差。',
            '历史注入为均衡模拟提议分布，不是天体物理检出率；BAYESTAR使用模拟匹配滤波测量，未对每个精确非高斯注入做full PE。上述冻结限制不会因新波形特征而消失。',
            '旧的有限源/噪声数据、对齐自旋模板失配、每窗口标准化以及已查看测试集限制均保留。若进入正式新版本，还需要独立验证。','',
            t.STATUS]
    (root/'reports/TEMPORAL_RESPONSE_ROUND_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();bootstrap(args.root);report(args.root)
