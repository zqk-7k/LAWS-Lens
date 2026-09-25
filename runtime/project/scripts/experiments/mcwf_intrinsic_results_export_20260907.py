#!/usr/bin/env python3
"""Replay and export the entire intrinsic-grid exploration, including failures."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np
import pandas as pd
import mcwf_intrinsic_grid_20260907 as g
import mcwf_intrinsic_score_calibration_20260907 as recal
import mcwf_omc_extension_results_20260907 as export

e,dev=g.e,g.dev


def applied_score(f,x,s,a):
    if s.get('replace_OMC'):
        f=f.copy();f['waveform_score']=f.FRT_baseline_waveform_score
    return (recal.score(f,x,s,a) if 'arm' in s else g.score(f,x,s,a))[0]


def cached(work,dep,es,split,arm):
    f=pd.read_parquet(e.TRIAL/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    if work.name=='recalibration':return f,{}
    cachework=work.parent if work.name=='replacement' else work
    path=cachework/f'cache/pair_features/{arm}/{dep}_{es}_{split}.npz'
    if not path.exists():raise RuntimeError('Missing completed feature cache: '+str(path))
    return f,dict(np.load(path))


def materialize(root):
    def setup(work):
        # The existing writer supplies the arm later; cache lookup is scoped to each trial.
        def values(r,d,s,sp):return cached(r,d,s,sp,active['arm'])
        export.e.values=values
        return ('SCORE_RECALIBRATION' if work.name=='recalibration' else
                'CONDITIONAL_INCREMENT' if work.name=='increment' else 'INTRINSIC_GRID'),work
    active={}
    original_read=pd.read_csv
    def read(path,*args,**kwargs):
        if Path(path).name=='ALL_COMBINATIONS.csv':active['arm']=Path(path).parents[1].name
        return original_read(path,*args,**kwargs)
    export.setup=setup
    export.scored=applied_score
    pd.read_csv=read
    try:export.materialize(root)
    finally:pd.read_csv=original_read
    rows=pd.read_csv(root/'tables/ALL_DIAGNOSTIC_RETRIEVAL.csv')
    keys=['experiment','deployment','split','method','config']
    metrics=[c for c in rows if c not in keys+['seed'] and pd.api.types.is_numeric_dtype(rows[c])]
    summary=rows.groupby(keys)[metrics].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') for x in summary.columns]
    dev.csv_write(root/'tables/RETRIEVAL_MEAN_SD.csv',summary)


def independent_score(f,x,s):
    old=f.waveform_score.to_numpy(float)
    if s.get('unchanged_baseline'):return old.copy()
    if s.get('replace_OMC'):old=f.FRT_baseline_waveform_score.to_numpy(float)
    if 'arm' in s:
        if s['arm']=='ISOTONIC':a=np.interp(old,s['knots'],s['loglr'])
        else:
            a=(old.clip(s['minimum'],s['maximum'])-s['mu'])/s['sd']*s['slope']+s['intercept']
            c=np.log((1-s['probability_floor'])/s['probability_floor']);a=a.clip(-c,c)
        return (1-s['gamma'])*old+s['gamma']*s['beta']*a
    # Independent NumPy calculation mirrors the frozen finite-reference mass rule.
    ref=np.asarray(s['mass_reference']);distance=-np.log(np.asarray(x['bc']).clip(1e-300))
    p=(1+len(ref)-np.searchsorted(ref,distance,side='left'))/(len(ref)+1)
    penalty=np.minimum(np.log(p/.05),0.)
    feature=np.asarray(x['prior_overlap']);cal=s['prior_overlap']
    inc=np.interp(feature,cal['knots'],cal['loglr']).clip(-4,4)
    positive_ood=(feature<cal['minimum'])|(feature>cal['maximum'])|(p<.05)
    inc=np.where(positive_ood & (inc>0),0,inc)
    penalty=np.where(x['ood'],0,penalty);inc=np.where(x['ood'],0,inc)
    return old+s['gamma']*penalty+s['beta']*inc


def verify(root):
    checks=[]
    for choice in json.loads((root/'tables/ALL_DIAGNOSTIC_CHOICES.json').read_text()):
        trial=next(p.parent.parent for p in root.rglob('SEARCH_COMPLETE.json')
                   if str(p.parent.parent.relative_to(root)).replace('/trials/',':').removeprefix('trials/')==choice['experiment'])
        work=trial.parent.parent
        for es,spec in choice['configs'].items():
            for split in ('validation','test','real'):
                f,x=cached(work,choice['deployment'],int(es),split,trial.name)
                expected=independent_score(f,x,spec)
                actual=pd.read_parquet(trial/f'diagnostic_export/evaluation/{choice["deployment"]}/seed_{es}/{split}_pairs.parquet')
                error=float(np.max(abs(expected-actual.waveform_score.to_numpy(float))))
                frozen=all(np.array_equal(f[k],actual[k]) for k in ('idx_i','idx_j','time_score','sky_raw_log_bf'))
                checks.append({'experiment':choice['experiment'],'deployment':choice['deployment'],'seed':es,'split':split,
                               'max_absolute_score_error':error,'time_sky_indices_exact':frozen})
                assert error<1e-10 and frozen,(choice['experiment'],es,split,error)
    dev.csv_write(root/'audit/INDEPENDENT_SCORE_REPLAY.csv',pd.DataFrame(checks))
    protected=[];seen=set()
    for manifest in root.rglob('*PROTECTED*.csv'):
        if manifest.name not in ('PROTECTED_INPUT_SHA256.csv','GLOBAL_PRIOR_PROTECTED.csv'):continue
        for row in pd.read_csv(manifest).itertuples():
            if row.path in seen:continue
            seen.add(row.path);actual=dev.sha(Path(row.path));assert actual==row.sha256
            protected.append({'path':row.path,'sha256':actual,'unchanged':True})
    dev.csv_write(root/'manifest/ALL_HISTORICAL_RECHECK.csv',pd.DataFrame(protected))
    dev.json_write(root/'audit/FINAL_REPLAY_AND_PROTECTION.json',{
        'replayed_pair_tables':len(checks),'score_replay_tolerance':1e-10,'all_pass':True,
        'historical_files_unchanged':len(protected),'GLOBAL_PRIOR_preserved':True,
        'time_sky_unchanged':True,'no_independent_real_validation_claim':True})
    print(json.dumps({'replay_pass':len(checks),'protected_unchanged':len(protected)}),flush=True)


def counterexamples(root):
    all_budgets=[];all_metrics=[];all_choices=[]
    for marker in sorted(root.rglob('SEARCH_COMPLETE.json')):
        trial=marker.parent.parent;work=trial.parent.parent
        code=str(trial.relative_to(root)).replace('/trials/',':').removeprefix('trials/')
        dest=trial/'pe_leading_counterexample'
        if (dest/'COMPLETE.json').exists():
            all_budgets.append(pd.read_csv(dest/'BUDGETS.csv'));all_metrics.append(pd.read_csv(dest/'RETRIEVAL.csv'))
            all_choices.extend(json.loads((dest/'CHOICES.json').read_text()));continue
        dest.mkdir(exist_ok=False);combos=pd.read_csv(trial/'tables/ALL_COMBINATIONS.csv')
        budgets=[];metrics=[];choices=[]
        for dep in e.old.DEPS:
            a=combos[combos.deployment==dep]
            row=a.sort_values(['BC_gain','median_gain','front_gain','hanabi_gain','combination'],ascending=[False,False,False,False,True]).iloc[0]
            real={};selected={}
            for k,es in enumerate(dev.SEEDS):
                pool=json.loads((trial/f'configs/{dep}_{es}_ELIGIBLE.json').read_text())['eligible']
                spec=next(c['spec'] for c in pool if c['id']==row[f'seed{k+1}']);selected[str(es)]=spec
                for split in ('validation','test','real'):
                    f,x=cached(work,dep,es,split,trial.name);z=applied_score(f,x,spec,trial.name)
                    assert np.max(abs(z-independent_score(f,x,spec)))<1e-10
                    if split=='real':
                        n=f.drop(columns=[c for c in ('rank','final_score','method','waveform_contribution','time_contribution','sky_contribution') if c in f]).copy()
                        n['OMC_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z;real[es]=n
                    else:
                        bm=e.ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);cm=e.ev.metrics(f,z,dep,es)
                        assert e.ev.guard(cm,bm)
                        for method in cm:
                            metrics.append({'experiment':code,'deployment':dep,'seed':es,'split':split,'method':method,
                                            'configuration':'PE_LEADING_NOT_ADOPTED',**cm[method]})
            dev.save_evaluation(dest,'PE_LEADING_NOT_ADOPTED',dep,real,pd.read_parquet(root/f'audit/{dep}_external_reference.parquet'))
            b=pd.read_csv(dest/f'results/PE_LEADING_NOT_ADOPTED/{dep}/pe_official_budget.csv');b.insert(0,'experiment',code)
            for budget in (10,20):
                material=b[(b.seed.astype(str)=='consensus')&(b.method=='C_fixed')&(b.budget==budget)].iloc[0]
                for key in ('catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3','median_BC_mc','official_frontend','official_hanabi'):
                    assert abs(float(material[key])-float(row[f'Top{budget}_{key}']))<1e-10
            b['target_pass']=bool(row.target_pass);budgets.append(b)
            choices.append({'experiment':code,'deployment':dep,'configs':selected,'target_pass':bool(row.target_pass),
                            'selected_for':'maximum PE-count improvement within simulation-admissible grid;not adopted,not blind',
                            'independent_score_replay_pass':True})
        b=pd.concat(budgets,ignore_index=True);m=pd.DataFrame(metrics)
        dev.csv_write(dest/'BUDGETS.csv',b);dev.csv_write(dest/'RETRIEVAL.csv',m);dev.json_write(dest/'CHOICES.json',choices)
        dev.json_write(dest/'COMPLETE.json',{'all_outputs_present':True,'not_adopted':True,'replay_pass':True})
        all_budgets.append(b);all_metrics.append(m);all_choices.extend(choices)
        print(json.dumps({'counterexample_exported':code}),flush=True)
    dev.csv_write(root/'tables/PE_LEADING_COUNTEREXAMPLE_BUDGETS.csv',pd.concat(all_budgets,ignore_index=True))
    dev.csv_write(root/'tables/PE_LEADING_COUNTEREXAMPLE_RETRIEVAL.csv',pd.concat(all_metrics,ignore_index=True))
    dev.json_write(root/'tables/PE_LEADING_COUNTEREXAMPLE_CHOICES.json',all_choices)


def failure_breakdown(root):
    base=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv')
    base=base[(base.seed.astype(str)=='consensus')&(base.method=='C_fixed')]
    rows=[];examples=[]
    for marker in sorted(root.rglob('SEARCH_COMPLETE.json')):
        trial=marker.parent.parent
        code=str(trial.relative_to(root)).replace('/trials/',':').removeprefix('trials/')
        combos=pd.read_csv(trial/'tables/ALL_COMBINATIONS.csv')
        for dep in e.old.DEPS:
            a=combos[combos.deployment==dep].copy();pe=np.ones(len(a),bool);official=np.ones(len(a),bool)
            for b in (10,20):
                old=base[(base.deployment==dep)&(base.budget==b)].iloc[0]
                for key in ('BC_mc_ge_0p5','Dmax_le_3','median_BC_mc'):
                    pe&=a[f'Top{b}_{key}'].to_numpy()>=old[key]-1e-12
                pe&=a[f'Top{b}_catastrophic_mc'].to_numpy()<=old.catastrophic_mc
                for key in ('official_frontend','official_hanabi'):
                    official&=a[f'Top{b}_{key}'].to_numpy()>=old[key]
            gain=(a.front_gain.to_numpy()>0)&(a.hanabi_gain.to_numpy()>0)
            rows.append({'experiment':code,'deployment':dep,'eligible_combinations':len(a),
                         'Cfixed_PE_noninferior':int(pe.sum()),'official_both_types_gain_and_no_budget_loss':int((official&gain).sum()),
                         'both_Cfixed_PE_and_official':int((pe&official&gain).sum()),
                         'full_target_pass':int(a.target_pass.sum()),'max_front_sum_gain':float(a.front_gain.max()),
                         'max_hanabi_sum_gain':float(a.hanabi_gain.max())})
            actual=(a.BC_gain!=0)|(a.median_gain.abs()>1e-12)|(a.front_gain!=0)|(a.hanabi_gain!=0)
            a=a[actual]
            if len(a):
                row=a.sort_values(['noninferior','front_gain','hanabi_gain','BC_gain','median_gain'],ascending=False).iloc[0]
                examples.append({'experiment':code,**row.to_dict(),'interpretation':'failed or local diagnostic,not adopted;not a new test'})
    dev.csv_write(root/'tables/TARGET_FAILURE_DECOMPOSITION.csv',pd.DataFrame(rows))
    dev.csv_write(root/'tables/NONBASELINE_DIAGNOSTIC_EXAMPLES.csv',pd.DataFrame(examples))


def collect_scripts(root):
    folder=dev.PROJECT/'scripts/experiments';target=root/'scripts/reproduction_dependencies'
    target.mkdir(exist_ok=True)
    queue=list(folder.glob('mcwf_intrinsic*_20260907.py'))+[folder/'mcwf_omc_extension_results_20260907.py']
    seen=set()
    while queue:
        p=queue.pop()
        if p in seen or not p.exists():continue
        seen.add(p);shutil.copy2(p,target/p.name)
        for node in ast.walk(ast.parse(p.read_text())):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            queue.extend(folder/(name+'.py') for name in names if name.startswith('mcwf_'))
    dev.csv_write(root/'manifest/REPRODUCTION_SCRIPT_SOURCES.csv',pd.DataFrame([
        {'path':str(p),'sha256':dev.sha(p)} for p in sorted(seen)]))


def report(root):
    ledger=pd.read_csv(root/'tables/EXPERIMENT_LEDGER.csv');assert not ledger.both_runs_target_pass.any(), 'Fresh confirmation required before final release'
    failure_breakdown(root)
    budgets=pd.read_csv(root/'tables/ALL_DIAGNOSTIC_BUDGETS.csv')
    compare=budgets[(budgets.seed.astype(str)=='consensus')&(budgets.method=='C_fixed')&(budgets.budget.isin([10,20]))]
    base=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv')
    base=base[(base.seed.astype(str)=='consensus')&(base.method=='C_fixed')&(base.budget.isin([10,20]))]
    selected=[]
    for r in compare.to_dict('records'):
        b=base[(base.deployment==r['deployment'])&(base.budget==r['budget'])].iloc[0]
        selected.append({**r,**{'baseline_'+k:b[k] for k in ('median_BC_mc','BC_mc_ge_0p5','Dmax_le_3','catastrophic_mc','official_frontend','official_hanabi')}})
    dev.csv_write(root/'tables/BASELINE_VS_DIAGNOSTICS.csv',pd.DataFrame(selected))
    training=[]
    for work in (root,root/'joint',root/'dense'):
        for p in work.glob('models/*/*/COMPLETE.json'):
            training.append({'model_family':work.name if work!=root else 'CONDITIONAL',**json.loads(p.read_text())})
    dev.csv_write(root/'tables/TRAINING_SUMMARY.csv',pd.DataFrame(training))
    runtimes=[{'log':str(p.relative_to(root)),**json.loads(p.read_text())} for p in root.rglob('*.runtime.json')]
    dev.csv_write(root/'tables/RUNTIME_SUMMARY.csv',pd.DataFrame(runtimes))
    recall=pd.read_csv(root/'tables/RETRIEVAL_MEAN_SD.csv')
    table=[]
    for dep in e.old.DEPS:
        for code in ledger.experiment.unique():
            m=recall[(recall.experiment==code)&(recall.deployment==dep)&(recall.split=='test')&(recall.method=='C_fixed')&(recall.config=='DIAGNOSTIC')].iloc[0]
            b=compare[(compare.experiment==code)&(compare.deployment==dep)].set_index('budget')
            table.append(f'| {dep} | {code} | {m.macro_r_at_10_mean:.4f} +/- {m.macro_r_at_10_std:.4f} | {m.average_precision_mean:.4f} | {int(b.loc[10].BC_mc_ge_0p5)}/{int(b.loc[20].BC_mc_ge_0p5)} | {int(b.loc[10].official_frontend)}/{int(b.loc[20].official_frontend)} | {int(b.loc[10].official_hanabi)}/{int(b.loc[20].official_hanabi)} |')
    totals=ledger.groupby('deployment')[['combinations','target_pass_combinations']].sum()
    text=f'''# MCWF-INTRINSIC-GRID 波形后续探索完整报告

状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE

## 最终结论

本轮完成 {len(training)} 个新模型训练和 {ledger.experiment.nunique()} 个完整评分对照，没有找到 O3、O4a 同时满足既定 PE 与官方重合改善条件的统一版本。不能宣布目标达成，不能把 O4a 单运行期改善包装成两个运行期共同升级。

O3 检查 {int(totals.loc['gwtc3','combinations'])} 个通过模拟开发护栏的跨 seed 组合，完整目标通过 {int(totals.loc['gwtc3','target_pass_combinations'])} 个；O4a 对应 {int(totals.loc['gwtc4','combinations'])} 和 {int(totals.loc['gwtc4','target_pass_combinations'])}。组合共享数据和模型，不是独立试验，不以这些数量计算显著性。

上一版 MCWF-UNIFIED-OMC-DEVCONF 不变。上一轮 O4a GLOBAL-PRIOR 的排名、逐 seed 表和共识表保存在 comparators/O4a_GLOBAL_PRIOR，并对历史模型和输出进行了 SHA-256 复核；它仍是下一轮对照，不是两个运行期统一升级。

## 实际改动

仅探索波形通道。原始峰值 2 s、4096 点 H1/L1 输入、旧 encoder、时间 lookup、天空 Nside=512/ordering、strict scope 和逐 seed C-fixed 外层权重均冻结。真实 O3 仍为 62 事件/1891 对，O4a 为 74 事件/2701 对；没有为了提高公开重合而更换目录。

1. CONDITIONAL：冻结已有 GLOBAL 的 Mc 主干，训练质量比 q 与有效自旋 chi_eff 的条件分类头。预测 p(logMc,q,chi_eff|waveform features)。
2. JOINT：相同结构，但允许 Mc 主干一起小学习率训练。
3. DENSE3D：利用已有更密 q/spin 模板网格的三维结构训练条件头，Mc 主干冻结；没有扩大时间窗口，没有新建完整 PE。既往平面网络曾用过密网格，本次新增的是结构化卷积，不声称这些信息从未用过。
4. 每种模型分别检验完整联合 prior-overlap、联合 BC、Mc+q overlap 和 Mc-only overlap。
5. 条件增量对照只加入联合 overlap 与 Mc overlap 的对数差，检查重复奖励 Mc 的影响。该量经过模拟类平衡校准，仍是经验预测特征，不是严格的物理条件 Bayes factor。
6. ISOTONIC/LOGISTIC 不训练新网络，仅重标定现有复合波形分数。重标定会改变与固定 time/sky 的有效相对尺度，不能解释成新增物理信息。
7. REPLACEMENT 从同一个冻结 Z_FRT 起步，用新联合预测证据替换原 OMC 辅助项，而不是把两套质量项叠加。仍与较新的 OMC 基线比较，不把较弱旧基线当作晋级门槛。

## 数据和训练

每个运行期 12,288 个训练 source systems，512 个独立 development systems；两种噪声集合隔离。每个训练 batch 选择同源两幅视图；development 1024 事件对应 512 个源，不把相关 null pair 当作独立系统。三个模型初始化各自训练，所有训练标签来自模拟源参数，而不是公开 PE、官方表或异常真实 pair。

模型预测是离散条件分类分布，不是 PE posterior。q、chi_eff 存在退化，预测质量和中央区间覆盖在各工作目录 audit/PREDICTIVE_QUALITY.csv 中列出；其覆盖率是同分布开发诊断，不是独立物理 PE 校准证明。超出网格的真值另计，不隐瞒。三模型等权集成在三个评估 seed 间共享，因此表中 SD 并非三份独立集成系统重训的方差。

本轮未生成新 strain。继承数据仍有明确限制：O3 噪声是 O3-only，但 detector response 采用原累积时间日历；注入强度继承逐像 target-SNR 缩放；GW-LMC lens environments 可以重用，独立生成的 source draws 不能冒充同数量的独立透镜环境。source/noise/GPS 隔离和这些限制分别在 audit/SOURCE_NOISE_DISJOINT_RECHECK.csv 与 audit/DATA_PROVENANCE_LIMITS.json 中报告。

## 分数和校准

主要对照使用 Z_new = Z_OMC + gamma*mass_penalty + beta*bounded_increment。三个模型的预测分布等权平均；经验密度/overlap 在模拟 development 上做类平衡 isotonic 校准。增量限制在 [-4,4]，超支持域正奖励回退为 0，质量不相容 p<0.05 时不增加正分；边界 OOD 按冻结规则回退。mass_penalty 是有限 development 真对参考分布的尾概率对数，不是单候选误报概率。

具体计算为：

```text
BC_M(i,j) = sum_m sqrt(p_i(m)*p_j(m))
D_M = -log BC_M
p_M = (1 + count(D_M,development,true >= D_M,pair))/(n_true + 1)
mass_penalty = min(log(p_M/0.05), 0)
joint_feature = log sum_(m,q,chi) p_i(m,q,chi)*p_j(m,q,chi)/training_prior(m,q,chi)
joint_BC_feature = sum_(m,q,chi) sqrt(p_i(m,q,chi)*p_j(m,q,chi))
conditional_feature = joint_feature - mass_only_feature
increment = clip(logit(isotonic_balanced_probability(feature)), -4, 4)
```

所有 p 均为离散预测概率质量；training_prior 是模拟训练源的平滑直方图，不是真实天体人口先验。isotonic 的伴随/非伴随总权重各为 0.5，因此 logit 是类平衡下的经验似然比分数，而不是原稀有率下的透镜后验赔率。0.05、cap=4 和边界占比 0.25 是继承且预登记的工程控制，不是文献推导的普适物理常数。

训练超参数、网格和各增补协议按启动顺序保存在 contracts。子目录由共同初始化器生成的训练字段，对无训练的 calibration/increment 对照不适用；以这些子目录明确的增补合同为准。

## 评价口径与适用边界

本轮按照作者授权，真实 PE 和官方重合结果参与有限开发配置选择。它们没有进入波形网络输入或训练标签，但已经是自适应方法选择依据，因此本轮不能称为盲测或 validation-only 的真实泛化证据。

之前的注入 validation/test 也用于本轮开发护栏，不再称为新的 locked test。护栏逐 seed 检查 waveform-only 与 C-fixed：R@10 最大下降 0.02，AUPRC 最大下降 0.005，F50/F90 不超过基线 1.1 倍。真实 Top-10/20 的 Mc BC 数量、中位数、Dmax 和灾难性不一致不能退化，前端与公开 Hanabi 重合均需增加，并保留上一轮 O4a GLOBAL-PRIOR 改善。

真实目标没有在两个运行期一起通过，因此没有启动新注入确认，也没有对旧测试 bootstrap 后声称达成独立确认。真实 pair 共享事件，PE/官方预算与相关系数均为描述性统计。公开前端或 Hanabi 表重合不等于透镜真值、阳性标签或 detection；本轮没有运行新 Hanabi。

## 全部对照汇总

下表是各对照在非劣组合中按冻结规则保留的诊断输出，不一定为新增模型。若全部新增设置不通过，则显示原基线；tables/ALL_DIAGNOSTIC_CHOICES.json 的 all_baseline 明确标识该情况。不能把这些回退行写成新模型成功。完整失败组合及真实预算见各 trials/*/tables/ALL_COMBINATIONS.csv。

| run | 对照 | reused-test R@10 mean +/- SD | Pair AUPRC mean | Top10/20 Mc BC>=0.5 | Top10/20 官方1% | Top10/20 Hanabi重合 |
|---|---|---|---|---|---|---|
{chr(10).join(table)}

## 交付和复核

- tables：全部对照、逐 seed recall/AUPRC/F50/F90、共识 PE/官方预算、模型误差和失败组合。
- tables/PE_LEADING_COUNTEREXAMPLE_* 与各 trials/*/pe_leading_counterexample：每个对照 PE 数量提升最多的模拟护栏内配置，包含全部真实排名与逐 seed 注入指标；用于显示 PE 与官方重合的取舍，不是选出的统一版本。
- trials/*/diagnostic_export/results：各方法 waveform-only/C-fixed 的逐 seed 和共识全部真实 pair、Top-100、BC/Dmax/官方 FPP/阶段列；这些表不参与重新训练。
- audit/INDEPENDENT_SCORE_REPLAY.csv：独立 NumPy 复算分数，检查 time/sky/事件索引严格不变。
- manifest/ALL_HISTORICAL_RECHECK.csv：旧版模型/结果、GLOBAL-PRIOR 前后哈希一致。
- scripts/reproduction_dependencies：本轮脚本和项目内依赖；不打包原始 strain、私钥、密码或大型可重建 feature caches。

## 文献与局限

联合内禀参数的一致性用于透镜 follow-up 有物理依据，但本实验学习的预测 overlap 并未替代完整联合似然。参见 [Haris et al.](https://arxiv.org/abs/1807.07062)。神经条件后验估计的相关工作参见 [Green et al.](https://arxiv.org/abs/2008.03312)；本实现不是该论文复现、不是 DINGO，也没有其完整 PE 精度保证。

本轮负结果不证明波形通道已无改进空间，只说明在冻结的数据、2 s 输入、模型/评分家族和验收边界下没有得到作者要求的共同改善。不得放宽已看过的验收条件、手动提升公开候选、或用未通过的局部改善宣布升级。后续若改变数据分布、窗口或统计目标，需要另立版本；本轮历史文件和论文均不修改。
'''
    (root/'reports/FINAL_INTRINSIC_EXPLORATION_CN.md').write_text(text,encoding='utf-8')
    (root/'README_CN.md').write_text('本轮完整报告：reports/FINAL_INTRINSIC_EXPLORATION_CN.md\n旧 O4a GLOBAL-PRIOR：comparators/O4a_GLOBAL_PRIOR/\n状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n',encoding='utf-8')
    dev.json_write(root/'contracts/FINAL_STATUS.json',{'status':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'user_target_achieved':False,'both_run_upgrade':False,'new_models':len(training),'completed_contrasts':int(ledger.experiment.nunique()),
        'new_independent_confirmation_started':False,'real_selection':'adaptive development, not blind',
        'GLOBAL_PRIOR_preserved':True,'historical_results_overwritten':False,'utc':datetime.now(timezone.utc).isoformat()})
    plot(root,compare,base)
    collect_scripts(root)


def plot(root,compare,base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for ax,dep in zip(axes,e.old.DEPS):
        x=compare[(compare.deployment==dep)&(compare.budget==20)]
        b=base[(base.deployment==dep)&(base.budget==20)].iloc[0]
        ax.scatter(x.median_BC_mc,x.official_hanabi,c='#227d81',s=38,alpha=.5,label='Admissible diagnostics')
        ax.scatter([b.median_BC_mc],[b.official_hanabi],c='#ae3941',marker='*',s=130,label='OMC baseline')
        if dep=='gwtc4':
            c=pd.read_csv(root/'comparators/O4a_GLOBAL_PRIOR/pe_official_budget.csv')
            c=c[(c.seed.astype(str)=='consensus')&(c.method=='C_fixed')&(c.budget==20)].iloc[0]
            ax.scatter([c.median_BC_mc],[c.official_hanabi],c='#a47716',marker='D',s=50,label='Previous GLOBAL-PRIOR')
        ax.set(xlabel='Top-20 median Mc BC',ylabel='Top-20 published Hanabi overlap',title='O3' if dep=='gwtc3' else 'O4a')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.set_ylim((6,8) if dep=='gwtc3' else (10,13))
        ax.spines[['top','right']].set_visible(False)
    h,l=axes[1].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ('pdf','png'):fig.savefig(root/f'figures/fig_intrinsic_PE_official_comparison.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    counter=pd.read_csv(root/'tables/PE_LEADING_COUNTEREXAMPLE_BUDGETS.csv')
    counter=counter[(counter.seed.astype(str)=='consensus')&(counter.method=='C_fixed')&(counter.budget.isin([10,20]))]
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for ax,dep in zip(axes,e.old.DEPS):
        c=counter[counter.deployment==dep].groupby('experiment')[['BC_mc_ge_0p5','official_frontend']].sum()
        b=base[base.deployment==dep][['BC_mc_ge_0p5','official_frontend']].sum()
        ax.scatter(c.BC_mc_ge_0p5,c.official_frontend,c='#457daa',s=36,alpha=.5,label='PE-leading, not adopted')
        ax.scatter([b.BC_mc_ge_0p5],[b.official_frontend],c='#ae3941',marker='*',s=130,label='OMC baseline')
        ax.axhline(b.official_frontend,ls=':',color='gray',lw=.8)
        ax.set(xlabel='Mc BC >= 0.5: Top-10 + Top-20 counts',ylabel='Official 1%: Top-10 + Top-20 counts',title='O3' if dep=='gwtc3' else 'O4a')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[['top','right']].set_visible(False)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ('pdf','png'):fig.savefig(root/f'figures/fig_PE_leading_tradeoff.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def package(root):
    assert (root/'audit/FINAL_REPLAY_AND_PROTECTION.json').exists()
    assert json.loads((root/'contracts/FINAL_STATUS.json').read_text())['user_target_achieved'] is False
    archive=dev.PROJECT/f'packages/{root.name}_deliverables.tar.gz'
    if archive.exists():raise RuntimeError('Refusing existing package')
    # All compact results/configs/models are retained; cached inputs and prediction arrays are reconstructible.
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if 'cache' in rel.parts or '__pycache__' in rel.parts or p.name=='development_predictions.npz':continue
        if p.suffix in ('.npy','.npz'):continue
        if 'PRIVATE KEY' in p.name.upper() or p.name.endswith(('.pem','.key')):raise RuntimeError('Forbidden credential filename')
        if p.suffix in ('.py','.json','.md','.txt','.log','.csv'):
            data=p.read_bytes()
            if re.search(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',data):
                raise RuntimeError('Private-key content found in deliverable')
            if re.search(rb'(?i)(?:password|passwd|api_key)\s*[:=]\s*[\"\x27][^\"\x27\r\n]{8,}[\"\x27]',data):
                raise RuntimeError('Possible hard-coded credential requires inspection: '+str(rel))
        files.append(p)
    manifest=root/'manifest/DELIVERABLE_SHA256.csv'
    dev.csv_write(manifest,pd.DataFrame([{'path':str(p.relative_to(root)),'sha256':dev.sha(p),'bytes':p.stat().st_size} for p in files]))
    files.append(manifest)
    with tarfile.open(archive,'w:gz',compresslevel=5) as tar:
        for p in files:tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    expected=pd.read_csv(manifest).set_index('path');checked=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            rel=str(Path(member.name).relative_to(root.name))
            if rel in expected.index:
                h=hashlib.sha256()
                with tar.extractfile(member) as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
                assert h.hexdigest()==expected.loc[rel,'sha256'],rel;checked+=1
    assert checked==len(expected)
    sha=dev.sha(archive);archive.with_suffix(archive.suffix+'.sha256').write_text(f'{sha}  {archive.name}\n')
    verification={'archive':str(archive),'sha256':sha,'bytes':archive.stat().st_size,'verified_internal_files':checked,
                  'all_required_outputs_present':True,'historical_unchanged':True,'no_new_confirmation_claim':True}
    dev.json_write(archive.with_suffix(archive.suffix+'.verification.json'),verification)
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('materialize','counterexamples','verify','report','package'),required=True)
    a=p.parse_args();globals()[a.phase](a.root)
