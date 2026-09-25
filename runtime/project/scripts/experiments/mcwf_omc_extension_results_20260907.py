#!/usr/bin/env python3
"""Export every completed contrast, including unchanged and failed outcomes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import mcwf_omc_ensemble_extension_20260907 as e
import mcwf_omc_architecture_extension_20260907 as arch
import mcwf_omc_single_extension_20260907 as single
import mcwf_omc_source_consistency_20260907 as source

dev = e.dev
ORIGINAL = {k: getattr(e, k) for k in ('values', 'calibrate', 'ensemble', 'score', 'raw_prediction')}


def setup(work):
    for k, value in ORIGINAL.items():
        setattr(e, k, value)
    if (work / 'contracts/SOURCE_CONSISTENCY_ADDENDUM.json').exists():
        e.values = source.values
        return 'SOURCE', work
    if (work / 'contracts/ARCHITECTURE_ADDENDUM.json').exists():
        e.raw_prediction = arch.prediction
        return 'ENSEMBLE', work
    for filename in ('SINGLE_MODEL_ADDENDUM.json', 'SCORE_STRUCTURE_ADDENDUM.json'):
        path = work / 'contracts' / filename
        if path.exists():
            model_root = Path(json.loads(path.read_text())['model_root'])
            single.configure(work, model_root)
            return 'SINGLE', model_root
    return 'OLD_OMC_ENSEMBLE', None


def scored(f, x, spec, arm):
    if arm == 'SOURCE-COSINE' and not spec.get('unchanged_baseline'):
        spec = {**spec, 'prior_overlap': spec['cosine']}
        x = {**x, 'prior_overlap': x['cosine']}
    return ORIGINAL['score'](f, x, spec, 'MIXTURE-PRIOR' if arm.startswith('SOURCE') else arm)[0]


def materialize(root):
    overview, budgets, retrieval, choices, correlations, checks = [], [], [], [], [], []
    baseline = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    bcons = baseline[baseline.seed.astype(str) == 'consensus']
    for marker in sorted(root.rglob('SEARCH_COMPLETE.json')):
        trial, work = marker.parent.parent, marker.parent.parent.parent.parent
        if 'diagnostic_export' in marker.parts:
            continue
        state = json.loads(marker.read_text())
        mode, modelroot = setup(work)
        code = str(trial.relative_to(root)).replace('/trials/', ':')
        if code.startswith('trials/'):
            code = code[7:]
        combos = pd.read_csv(trial / 'tables/ALL_COMBINATIONS.csv')
        export = trial / 'diagnostic_export'
        if (export / 'COMPLETE.json').exists():
            budgets.append(pd.read_csv(export / 'BUDGETS.csv'))
            retrieval.append(pd.read_csv(export / 'RETRIEVAL.csv'))
            choices += json.loads((export / 'CHOICES.json').read_text())
            correlations += json.loads((export / 'CORRELATIONS.json').read_text())
        else:
            export.mkdir(exist_ok=False)
            local_budgets, local_metrics, local_choices, local_corr = [], [], [], []
            for dep in e.old.DEPS:
                candidates = combos[(combos.deployment == dep) & combos.noninferior]
                assert len(candidates), 'Unchanged baseline must remain admissible'
                winner = candidates.sort_values(['target_pass', 'BC_gain', 'median_gain', 'front_gain', 'hanabi_gain', 'combination'],
                                               ascending=[False, False, False, False, False, True]).iloc[0]
                real, configs = {}, {}
                for k, es in enumerate(dev.SEEDS):
                    pool = json.loads((trial / f'configs/{dep}_{es}_ELIGIBLE.json').read_text())['eligible']
                    spec = next(p['spec'] for p in pool if p['id'] == winner[f'seed{k+1}'])
                    configs[str(es)] = spec
                    for split in ('validation', 'test', 'real'):
                        f, x = e.values(work, dep, es, split)
                        z = scored(f, x, spec, trial.name)
                        n = f.drop(columns=[c for c in ('final_score','rank','method','waveform_contribution','time_contribution','sky_contribution') if c in f]).copy()
                        n['OMC_baseline_waveform_score'], n['waveform_score'] = f.waveform_score, z
                        for key, v in x.items():
                            n['new_waveform_' + key] = v
                        for c in ('idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf'):
                            assert np.array_equal(n[c], f[c])
                        out = export / f'evaluation/{dep}/seed_{es}'
                        out.mkdir(parents=True, exist_ok=True)
                        n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                        if split == 'real':
                            real[es] = n
                        else:
                            bm = e.ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
                            cm = e.ev.metrics(n, z, dep, es)
                            assert e.ev.guard(cm, bm)
                            for method in cm:
                                for config, m in [('BASELINE', bm[method]), ('DIAGNOSTIC', cm[method])]:
                                    local_metrics.append({'experiment': code, 'deployment': dep, 'seed': es, 'split': split,
                                                          'method': method, 'config': config, **m})
                pe = pd.read_parquet(root / f'audit/{dep}_external_reference.parquet')
                dev.save_evaluation(export, 'DIAGNOSTIC', dep, real, pe)
                b = pd.read_csv(export / f'results/DIAGNOSTIC/{dep}/pe_official_budget.csv')
                b.insert(0, 'experiment', code)
                b['development_target_pass'] = bool(winner.target_pass)
                b['both_runs_target_pass'] = bool(state['both_runs_qualify'])
                local_budgets.append(b)
                selected_consensus = b[(b.seed.astype(str)=='consensus') & (b.method=='C_fixed')]
                for budget in (10, 20):
                    row = selected_consensus[selected_consensus.budget==budget].iloc[0]
                    for key in ('catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3','official_frontend','official_hanabi','median_BC_mc'):
                        assert abs(float(row[key])-float(winner[f'Top{budget}_{key}'])) < 1e-10, (code,dep,budget,key)
                fr = pd.read_parquet(export / f'results/DIAGNOSTIC/{dep}/consensus_waveform_only_all_pairs_pe_official.parquet')
                scorecol = 'mean_score' if 'mean_score' in fr else 'final_score_mean' if 'final_score_mean' in fr else 'final_score'
                if scorecol in fr:
                    for metric, sign in [('pe_mc_bhattacharyya_coefficient',1),('pe_mc_standardized_distance',-1)]:
                        rho = spearmanr(fr[scorecol], sign*fr[metric], nan_policy='omit').statistic
                        local_corr.append({'experiment':code,'deployment':dep,'PE_metric':metric,'spearman':float(rho),
                                           'interpretation':'descriptive dependent-pair correlation,no ordinary pair-level pvalue'})
                local_choices.append({'experiment': code, 'deployment': dep, 'mode': mode,
                                      'model_root': str(modelroot), 'configs': configs,
                                      'development_target_pass': bool(winner.target_pass),
                                      'both_runs_target_pass': bool(state['both_runs_qualify']),
                                      'all_baseline': all(c.get('unchanged_baseline',False) for c in configs.values()),
                                      'selection': 'Highest target-pass then BC-count,median,official improvements among noninferior grid outcomes;adaptive descriptive diagnostic'})
            bb, rr = pd.concat(local_budgets,ignore_index=True), pd.DataFrame(local_metrics)
            dev.csv_write(export/'BUDGETS.csv',bb)
            dev.csv_write(export/'RETRIEVAL.csv',rr)
            dev.json_write(export/'CHOICES.json',local_choices)
            dev.json_write(export/'CORRELATIONS.json',local_corr)
            dev.json_write(export/'COMPLETE.json',{'exact_vectorized_to_materialized_agreement':True,'time_sky_exact':True})
            budgets.append(bb); retrieval.append(rr); choices+=local_choices; correlations+=local_corr
        for dep in e.old.DEPS:
            f = combos[combos.deployment==dep]
            overview.append({'experiment':code,'deployment':dep,'combinations':len(f),
                             'noninferior_combinations':int(f.noninferior.sum()),'target_pass_combinations':int(f.target_pass.sum()),
                             'both_runs_target_pass':bool(state['both_runs_qualify'])})
        print(json.dumps({'materialized':code,'both_runs_pass':state['both_runs_qualify']}),flush=True)
    dev.csv_write(root/'tables/EXPERIMENT_LEDGER.csv',pd.DataFrame(overview))
    dev.csv_write(root/'tables/ALL_DIAGNOSTIC_BUDGETS.csv',pd.concat(budgets,ignore_index=True))
    dev.csv_write(root/'tables/ALL_DIAGNOSTIC_RETRIEVAL.csv',pd.concat(retrieval,ignore_index=True))
    dev.json_write(root/'tables/ALL_DIAGNOSTIC_CHOICES.json',choices)
    dev.csv_write(root/'tables/WAVEFORM_PE_CORRELATIONS.csv',pd.DataFrame(correlations))
    checks = []
    for r in pd.read_csv(root/'manifest/PROTECTED_INPUT_SHA256.csv').itertuples():
        current=dev.sha(Path(r.path))
        checks.append({'path':r.path,'before':r.sha256,'after':current,'unchanged':current==r.sha256})
    dev.csv_write(root/'manifest/PROTECTED_INPUT_RECHECK.csv',pd.DataFrame(checks))
    assert all(c['unchanged'] for c in checks)
    dev.json_write(root/'audit/MATERIALIZATION_AUDIT.json',{'old_files_unchanged':len(checks),'all_unchanged':True,
                    'all_selected_real_budgets_reproduced':True,'time_sky_unchanged':True})


def package(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in Path('/usr/share/fonts').rglob('*.ttf'):
        if 'times' in font.name.lower():
            font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42})
    ledger=pd.read_csv(root/'tables/EXPERIMENT_LEDGER.csv')
    budgets=pd.read_csv(root/'tables/ALL_DIAGNOSTIC_BUDGETS.csv')
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv')
    selected=budgets[(budgets.seed.astype(str)=='consensus')&(budgets.method=='C_fixed')&(budgets.budget.isin([10,20]))]
    training=[]
    for p in root.glob('architectures/*/models/*/*/COMPLETE.json'):
        training.append({'family':p.parents[3].name,**json.loads(p.read_text())})
    for p in root.glob('source_consistency/models/*/*/COMPLETE.json'):
        training.append({'family':'SOURCE',**json.loads(p.read_text())})
    dev.csv_write(root/'tables/MODEL_TRAINING_SUMMARY.csv',pd.DataFrame(training))
    paired=[]
    for r in selected.to_dict('records'):
        b=baseline[(baseline.deployment==r['deployment'])&(baseline.seed.astype(str)=='consensus')&(baseline.method=='C_fixed')&(baseline.budget==r['budget'])].iloc[0]
        paired.append({**r,**{'baseline_'+k:b[k] for k in ('BC_mc_ge_0p5','median_BC_mc','Dmax_le_3','official_frontend','official_hanabi','catastrophic_mc')}})
    compare=pd.DataFrame(paired)
    dev.csv_write(root/'tables/BASELINE_VS_DIAGNOSTICS.csv',compare)
    recall=pd.read_csv(root/'tables/REUSED_RETRIEVAL_MEAN_SD.csv')
    detail_rows=[]
    def alias(code):
        return (code.replace('architectures/','').replace('single_controls/','SINGLE-')
                .replace('score_controls/','').replace('source_consistency:','')
                .replace(':MIXTURE-','-').replace('MIXTURE-','OMC-ENSEMBLE-'))
    for dep in e.old.DEPS:
        for code in sorted(ledger.experiment.unique()):
            r=recall[(recall.deployment==dep)&(recall.experiment==code)&(recall.split=='test')&
                     (recall.method=='C_fixed')&(recall.config=='DIAGNOSTIC')].iloc[0]
            bb=compare[(compare.deployment==dep)&(compare.experiment==code)].set_index('budget')
            b10,b20=bb.loc[10],bb.loc[20]
            detail_rows.append(f"| {dep} | {alias(code)} | {r.macro_r_at_1_mean:.4f} +/- {r.macro_r_at_1_std:.4f} | {r.macro_r_at_10_mean:.4f} +/- {r.macro_r_at_10_std:.4f} | {r.average_precision_mean:.4f} | {r.false_at_recall_0p5_mean:.1f} / {r.false_at_recall_0p9_mean:.1f} | {int(b10.BC_mc_ge_0p5)} / {int(b20.BC_mc_ge_0p5)} | {int(b10.official_frontend)} / {int(b20.official_frontend)} | {int(b10.official_hanabi)} / {int(b20.official_hanabi)} |")
    strict=bool(ledger.both_runs_target_pass.any())
    if strict:
        raise RuntimeError('A qualifying candidate exists: complete its fresh confirmation before final packaging')
    per=ledger.groupby('deployment')[['combinations','target_pass_combinations']].sum()
    chart=selected[selected.budget==20].copy()
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for ax,dep in zip(axes,e.old.DEPS):
        q=chart[chart.deployment==dep]
        ax.scatter(q.BC_mc_ge_0p5,q.official_hanabi,c='#257a77',s=30,label='Admissible diagnostics',alpha=.65)
        b=baseline[(baseline.deployment==dep)&(baseline.seed.astype(str)=='consensus')&(baseline.method=='C_fixed')&(baseline.budget==20)].iloc[0]
        ax.scatter([b.BC_mc_ge_0p5],[b.official_hanabi],marker='*',s=125,c='#ad3b37',label='OMC baseline')
        ax.set(xlabel='Top-20: PE Mc BC >= 0.5',ylabel='Top-20: published Hanabi overlap',title='O3' if dep=='gwtc3' else 'O4a')
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.02),frameon=False)
    fig.tight_layout(rect=(0,0,1,.88))
    fig.savefig(root/'figures/fig_PE_official_tradeoff.pdf',bbox_inches='tight')
    fig.savefig(root/'figures/fig_PE_official_tradeoff.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    rows=[]
    for dep in e.old.DEPS:
        d=compare[compare.deployment==dep]
        diff=d.median_BC_mc-d.baseline_median_BC_mc
        if len(d):
            r=d.iloc[int(np.argmax(diff.to_numpy()))]
            rows.append(f"| {dep} | {r['experiment']} | {int(r.budget)} | {int(r.baseline_BC_mc_ge_0p5)} -> {int(r.BC_mc_ge_0p5)} | {r.baseline_median_BC_mc:.4f} -> {r.median_BC_mc:.4f} | {int(r.baseline_official_frontend)} -> {int(r.official_frontend)} | {int(r.baseline_official_hanabi)} -> {int(r.official_hanabi)} |")
    text=f'''# OMC 后续波形优化探索报告

状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE

## 结论

本轮已完成 {ledger.experiment.nunique()} 个评分对照、{len(training)} 个新模型训练，但没有找到在 O3 和 O4a 同时满足“PE 更一致、官方前端和公开 Hanabi 重合更多、注入检索不明显退化”的统一配置。目标未达成；上一版 MCWF-UNIFIED-OMC-DEVCONF 保留，不宣布替代，不覆盖历史结果。

O3 评估了 {int(per.loc['gwtc3','combinations'])} 个种子配置组合，严格目标通过数 {int(per.loc['gwtc3','target_pass_combinations'])}。O4a 对应 {int(per.loc['gwtc4','combinations'])} 和 {int(per.loc['gwtc4','target_pass_combinations'])}。这些网格点和共享模型组合不是独立统计样本，不能用组合数量计算显著性。

## 与上一版的比较

下面分别展示各运行期在已满足非劣条件的诊断中，Mc BC 中位数改善最大的预算单元。它们只是局部改善示例，不代表同一个统一新版本，也不表示全部目标通过。

| 运行期 | 诊断 | Top-B | Mc BC>=.5 | Mc BC中位数 | 官方1%前端 | 公开Hanabi重合 |
|---|---|---:|---|---|---|---|
{chr(10).join(rows)}

每个完整对照的逐 seed recall、AUPRC、F50/F90、真实 Top-10/20/50/100 和全部 pair 均在对应 `diagnostic_export/` 中。`tables/BASELINE_VS_DIAGNOSTICS.csv` 是统一 PE/官方比较表。若某一运行期最优可接受诊断仍为三个 UNCHANGED，其结果确实等于旧版，不是遗漏新分数。

## 全部对照数值

以下是各对照中选出的开发性诊断配置，排序选择依据见 `ALL_DIAGNOSTIC_CHOICES.json`，不是只展示全部目标通过者。R@K 为原项目定义的 family-macro 指标，显示三个部署种子的均值与 SD；AUPRC、F50/F90 为三 seed 均值。它们来自复用 test 数据，不能称作新增独立测试结果。BC 与官方列分别是共识 Top10 / Top20 的计数。单 seed、waveform-only、Top50/100 及其他指标均在 CSV 中。

| 运行期 | 对照 | 三通道 R@1 | 三通道 R@10 | Pair AUPRC | F50 / F90 | Mc BC>=.5 (10/20) | 官方1% (10/20) | Hanabi表重合 (10/20) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(detail_rows)}

## 做了什么

1. 以刚交付的 OMC-DEVCONF 而非更早的 C-fixed/FRT 作为比较基线，校验其压缩包 SHA-256。
2. 在不修改原模型的情况下测试三个原 OMC 质量预测分布的等权混合，并分别校准 BC 与训练先验校正的重合特征。
3. 复用原 12,288 个训练 source parents/160 noise blocks，以及独立的 512 个 development parents/32 noise blocks。分别训练更宽的质量轴卷积、加入全局 attention 的网络、加入有序概率评分损失的网络。每种 O3/O4a 各三个种子。
4. 在相同输入上测试单模型与等权集成，避免把集成收益误认为结构收益。WIDE/GLOBAL/ORDINAL 为 50 epochs，旧 OMC 为 30 epochs，因此不是严格只改变一个网络结构的消融。
5. 在 WIDE 基础上测试移除旧神经质量惩罚，以及保留 OMC 后只添加有上限的相容性正分。
6. 另训练一个同时预测 Mc 和学习同源波形一致性的网络。从 WIDE checkpoint 出发训练 15 epochs，损失为质量 CE 加 0.25 倍同源双视图 InfoNCE；未使用真实 PE 或官方标签作为训练标签。

## 完全未改动的部分

H1/L1 峰值 2秒/4096点、40--580Hz 输入、原 waveform encoder、RNC-FRT/原 OMC checkpoint、一维时间查分表、天空图/分辨率/天空分数、C-fixed 外层权重、62事件 O3/74事件 O4a strict scope、历史论文和排名均不改动。新增网络和候选评分只存在独立目录。保护清单中的旧文件在运行前后逐项核验。

## 分数与选择

通常对照为 `Zwf = Z_FRT + gamma * P_new + beta * I_new`；独立的替换对照用原 C-fixed waveform LR 作起点；正分对照在当前 OMC 分数上仅增加 `beta * max(I_new,0)`。最终仍为 `S=wW*Zwf+wT*Ztime+wS*Zsky`。没有第四个检索通道，也没有真实 PE、官方标签或事件名评分特征。

质量分布为神经预测分布，不是完整 PE posterior。有限参照尾概率不是透镜 FPP。isotonic 输出是平衡类别下的经验判别特征，不是完整物理 Bayes factor；相关的波形特征不能被解释为独立证据相乘。

模型 checkpoint 与温度依据模拟 development 选择。随后保留满足原注入 validation/reused-test guard 的网格点，再使用真实 PE/官方预算结果选择开发配置。本轮明确是自适应开发，不能称为真实目录盲测，也不能把复用测试集误差当作独立泛化证据。真实 PE 不直接输入评分不等于没有真实数据选择偏差。

guard：每个种子和两种方法 R@10 下降不超过 .02、AUPRC 下降不超过 .005、F50/F90 不超过基线的1.1倍。真实预算要求 Top10/20 的灾难性 Mc 数不增、BC>=.5和 Dmax<=3数量不降、BC中位数不降、官方前端/Hanabi计数不降，且每个运行期 PE 有改善、前端与Hanabi计数总和分别增加。没有在看到失败后放宽这些门槛。

## 解释边界

更低的模拟质量误差不保证官方候选重合上升。官方候选不是透镜真值；公开 Hanabi 表重合不是本轮做了 Hanabi，也不等于该候选被确认。只改善 PE 或只改善 O4a 不能写成两个运行期都成功。

由于没有统一候选通过开发门槛，未启动新 source/noise 确认，不把旧确认重新包装成新确认。包内 `scripts/NOT_EXECUTED_fresh_confirmation_draft.py` 仅是未执行、未获本轮运行验证的后续草案；没有新的独立检索置信区间或真实候选独立验证可报告。

集成对照共享三个已训练模型的混合预测，其三个旧部署种子差异不是三次独立集成重训的不确定性。所有 recall 为复用模拟数据上的开发结果；PE/官方指标为相关事件对的描述性结果，不报告普通 pair 独立 p 值。

## 文献依据

- [Deep Ensembles](https://arxiv.org/abs/1612.01474) 提供预测分布集成的动机，不保证在本 GW 数据上有效。
- [Attention 的结构依据](https://arxiv.org/abs/1706.03762)。
- [同源对比表示学习动机](https://arxiv.org/abs/2004.11362)。
- [透镜像内禀波形一致性的物理动机](https://arxiv.org/abs/1807.07062)。

这些文献不替本项目的数据选择、阈值或实际效果背书。
'''
    (root/'reports/FINAL_EXPLORATION_REPORT_CN.md').write_text(text,encoding='utf-8')
    status={'utc':datetime.now(timezone.utc).isoformat(),'objective_achieved':False,'both_runs_qualifying':False,
            'baseline_retained':'MCWF-UNIFIED-OMC-DEVCONF','experiments':int(ledger.experiment.nunique()),
            'new_models':len(training),'fresh_confirmation_executed':False,
            'status':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'}
    dev.json_write(root/'contracts/FINAL_STATUS.json',status)
    package_path=dev.PROJECT/'packages'/f'{root.name}_deliverables.tar.gz'
    if package_path.exists():
        raise RuntimeError('Refusing existing package')
    allowed={'.csv','.json','.md','.py','.pt','.parquet','.npz','.png','.pdf','.log'}
    payload=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts
             and p != root/'manifest/OUTPUT_SHA256.csv']
    private_key = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s*[A-Za-z0-9+/=]{32,}')
    for p in payload:
        if p.suffix in {'.json','.md','.py','.log'}:
            assert not private_key.search(p.read_text(errors='replace')), str(p)
    manifest=pd.DataFrame([{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':dev.sha(p)} for p in payload])
    dev.csv_write(root/'manifest/OUTPUT_SHA256.csv',manifest)
    payload.append(root/'manifest/OUTPUT_SHA256.csv')
    with tarfile.open(package_path,'w:gz',compresslevel=6) as tar:
        for p in payload:
            tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    digest=dev.sha(package_path)
    package_path.with_suffix(package_path.suffix+'.sha256').write_text(f'{digest}  {package_path.name}\n',encoding='ascii')
    failures=[]
    with tarfile.open(package_path,'r:gz') as tar:
        for r in manifest.itertuples():
            handle=tar.extractfile(str(Path(root.name)/r.path)); h=hashlib.sha256()
            while True:
                b=handle.read(1024*1024)
                if not b: break
                h.update(b)
            if h.hexdigest()!=r.sha256: failures.append(r.path)
    assert not failures
    verification={'path':str(package_path),'sha256':digest,'bytes':package_path.stat().st_size,
                  'verified_files':len(manifest),'internal_hash_failures':failures,**status}
    dev.json_write(root/'FINAL_DELIVERY_VERIFICATION.json',verification)
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('materialize','package'),required=True)
    a=p.parse_args()
    if a.phase=='materialize': materialize(a.root)
    else: package(a.root)
