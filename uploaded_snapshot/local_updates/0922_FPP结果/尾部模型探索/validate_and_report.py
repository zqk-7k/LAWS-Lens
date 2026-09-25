from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
from scipy.stats import genpareto
BASE=Path(__file__).resolve().parent
ROOT=next((BASE.parent/'extracted').iterdir());T=BASE/'tables'
fits=pd.read_csv(T/'tail_fits.csv');pred=pd.read_csv(T/'candidate_predictions_all.csv');de=pd.read_csv(T/'deletion_sensitivity.csv');summary=pd.read_csv(T/'candidate_Top20_summary.csv');held=pd.read_csv(T/'heldout_checks.csv')
tests=[]
def passed(label,condition):
    assert condition,label
    tests.append({'test':label,'pass':True})
passed('30 complete fits',len(fits)==30)
passed('all 600 model-threshold-candidate predictions retained',len(pred)==600)
passed('360 leaveout fits plus 3 maximum-pair deletions',len(de)==7260)
passed('300 held-out checks',len(held)==300)
for _,f in fits.iterrows():
    # Independently evaluate GPD survival algebra and normalization.
    sel=pred[(pred.run==f.run)&pred.q.eq(f.q)&pred.model.eq(f.model)]
    y=(sel.S.to_numpy()-f.u)/f.scale
    if f.xi==0:p=f.tail_weight*np.exp(-y)
    else:
        base=1+f.xi*y;p=np.zeros_like(y);ok=base>0
        p[ok]=f.tail_weight*np.exp(-np.log(base[ok])/f.xi)
    np.testing.assert_allclose(p,sel.estimated_tail_probability,atol=1e-14,rtol=1e-8)
    ordered=sel.sort_values('S');assert np.all(np.diff(ordered.estimated_tail_probability)<=1e-14)
    probs=np.array([.01,.001,.00025]);scores=f.u+genpareto.isf(probs/f.tail_weight,f.xi,scale=f.scale)
    np.testing.assert_allclose(f.tail_weight*genpareto.sf(scores-f.u,f.xi,scale=f.scale),probs,rtol=1e-10)
passed('GPD survival algebra, score monotonicity and inverse-tail checks',True)
aud=json.loads((BASE/'AUDIT.json').read_text(encoding='utf-8'))
passed('all 12 inputs unchanged',all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in aud['inputs']))
passed('O4b rank1 outside all five fitted GPD supports',pred[pred.run.eq('O4b')&pred['rank'].eq(1)&pred.model.eq('GPD')].outside_fitted_support.all())
support=[];candidate_test=[]
for run in ['O3','O4a','O4b']:
    test=pd.read_parquet(ROOT/f'tables/{run}/injection_mean_S_FPP.parquet')
    single=test[test.event_i.str.contains('unlensed')&test.event_j.str.contains('unlensed')]
    for _,f in fits[fits.run.eq(run)&fits.model.eq('GPD')].iterrows():
        outside=single[single.final_score_POSITIVE>=f.endpoint]
        support.append({'run':run,'q':f.q,'fitted_upper_endpoint':f.endpoint,'test_singleton_pairs_outside_support':len(outside),'test_singleton_max_score':single.final_score_POSITIVE.max()})
    for _,r in summary[summary.run.eq(run)].iterrows():
        k=int((single.final_score_POSITIVE>=r.S).sum())
        candidate_test.append({'run':run,'rank':int(r['rank']),'S':r.S,'test_singleton_count':k,'n_test_singleton_pairs':len(single)})
pd.DataFrame(support).to_csv(T/'test_support_violations.csv',index=False)
pd.DataFrame(candidate_test).to_csv(T/'candidate_test_counts_diagnostic_only.csv',index=False)
passed('test support and candidate thresholds checked without refitting',len(support)==15 and len(candidate_test)==60)
(BASE/'TESTS.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')

def percent(p):return f'{100*p:.5f}%'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
lines=['# 现有背景高分尾部建模结果（审阅版）\n',
'## 结论\n',
'**已完成建模，未补模拟。连续模型可以区分O4b中原本计数相同的候选，但当前不足以替换论文的经验FPP，也不能可靠地给Rank 1填入一个小概率。** 当前建议保持探索性结果，历史论文和数据不变。\n',
'## 做了什么\n',
'每运行期用90个validation孤立源的4005个配对平均分，分别拟合90%、92.5%、95%、97.5%、98%分位点以上的高分尾部。95%分位点作为拟合前固定的展示基准，共201个尾部分数；另外4个起点检查阈值敏感性。每个起点同时拟合GPD和指数尾部，共30个基本拟合。GPD用3个初值检查数值解。\n',
'对每运行期逐一删除90个源（95%基准），逐一删除6个噪声父块（所有5个起点），共360个删除后拟合；再检查各运行期删除最高分背景对的影响。测试集仅用于检查，不参与拟合或选择。无重训、无新增信号或噪声。\n',
'这些是模型尾概率，不是透镜概率或年度FAR。它们也不是官方FPP。原4005对共享源和噪声，不能视为4005个独立观测。敏感性范围不是置信区间。\n',
'## O4b Top-10：原计数与模型估计\n',
'下列数值固定使用95%分位点GPD，**不是通过检验后采用的主结果**。原背景计数保留。\n']
rows=[]
for _,r in summary[summary.run.eq('O4b')&summary['rank'].le(10)].iterrows():
    out='超出拟合支持范围，不能报零风险' if r.GPD_q95_probability==0 else percent(r.GPD_q95_probability)
    rows.append([int(r['rank']),r.pair,f'{r.S:.6f}',f'{r.empirical_count}/4005',out])
lines.append(table(['Rank','候选对','S','原背景计数','GPD尾概率估计（探索）'],rows))
lines += ['这说明不同分数可以映射到不同的连续尾概率，但这种差别来自分布假设，不是增加了新的背景观测。模型概率不必随共识Rank单调，因为共识名次与平均分不是同一个排序量。\n',
'## 为什么暂时不能进入论文主表\n',
'### 1. Rank 1 超出模型支持范围\n',
'O4b Rank 1的分数为2.984595。五个GPD拟合都得到负形状参数，即有限分数上端点。其上端点如下：\n']
lines.append(table(['背景阈值分位','尾部配对数','拟合上端点'],[[f'{100*r.q:g}%',int(r.m),f'{r.endpoint:.6f}'] for _,r in fits[fits.run.eq('O4b')&fits.model.eq('GPD')].iterrows()]))
lines += ['最高起点的上端点2.983330与Rank 1只差约0.0013，进一步说明该外推对端点估计很敏感。**超出拟合上端点只是该模型不支持该分数，不能解释为自然界中假对不可能达到该分数。**\n',
'### 2. 对最高分背景对及其源敏感\n',
'O4b最大背景分数为2.752655，来自UAB-main-unlensed-0455-1与UAB-main-unlensed-0504-1。删除这一对后，95%基准GPD的端点下降，原Rank 2–9全部落到模型支持范围之外。删除有关源或噪声父块也会改变这一结论。因此当前高分尾部受到极少数观测的影响。删除检查仅是诊断，正式原始背景没有删点。\n',
'### 3. 更换尾部模型会改变数值\n']
ref=fits[fits.run.eq('O4b')&fits.q.eq(.95)]
lines.append(table(['模型','尾部CDF最大差异（描述性）','QQ均方根误差','Rank 1估计'],[[r.model,f'{r.descriptive_KS:.4f}',f'{r.QQ_RMSE:.4f}','超出支持范围' if r.model=='GPD' else percent(summary[summary.run.eq('O4b')&summary['rank'].eq(1)].iloc[0].EXP_q95_probability)] for _,r in ref.iterrows()]))
lines += ['指数模型可以给Rank 1约0.169%的正值，但这依赖无限延伸的指数尾假设，且其拟合内误差更大。不能因为它能填满表格就选择它。上述CDF差异不是具有独立样本假设的KS检验p值。\n',
'### 4. 测试背景检查不能验证极端小概率\n',
'在既有测试目录的90个孤立源、4005对上，O4b 95%基准GPD的检查为：\n']
h=held[held.run.eq('O4b')&held.q.eq(.95)&held.model.eq('GPD')&held.check_set.eq('test_singletons')]
lines.append(table(['模型目标尾概率','模型阈值S','期望超越数','实测超越数'],[[percent(r.target_tail_probability),f'{r.threshold_S:.5f}',f'{r.expected_exceedances:.3f}',int(r.observed_exceedances)] for _,r in h.iterrows()]))
lines += ['在1%和0.5%附近，实测39和21对，与模型期望40.05和20.025接近；更深尾部只有0–2对，尚不能检验几万分之一量级的精度。零实测计数不是验证模型极小概率的证据。测试目录以前已查看过，不能称为本轮新盲测。\n',
'同流程在其他运行期还发现测试背景超过模型上端点：\n']
sp=pd.DataFrame(support);lines.append(table(['运行期','95%基准上端点','测试背景超出端点对数','测试背景最大分数'],[[r.run,f'{r.fitted_upper_endpoint:.6f}',int(r.test_singleton_pairs_outside_support),f'{r.test_singleton_max_score:.6f}'] for _,r in sp[sp.q.eq(.95)].iterrows()]))
lines += ['O3/O4a的超端点测试分数表明，不能将拟合上端点当作确定的物理上限。由于共享噪声、配对依赖和小样本，该分析没有给出正式显著性检验或置信区间。\n',
'## 模型怎么算\n',
'令u为背景分数分位点，m为超过u的背景对数，N=4005。对超出量y=S−u拟合GPD，形状参数为ξ，尺度为σ。\n',
'在拟合支持范围内：P模型(S≥s) = (m/N) × [1+ξ(s−u)/σ]<sup>−1/ξ</sup>。当ξ=0时，使用(m/N) × exp[−(s−u)/σ]。\n',
'这把尾部条件分布重新乘回全体背景中的尾部占比。负ξ的拟合上端点为u−σ/ξ。所有候选使用同一运行期的同一背景函数，不按候选单独拟合，不把经验计数直接插值成概率。\n',
'## 建议与边界\n',
'保留本轮为统计模型敏感性分析，不推送论文，不改变表1。若不增加有效背景，目前可继续展示经验计数与S，而不能要求每个候选都有可靠且不同的FPP。进一步更换模型也应由拟合外预测和稳健性决定，不能以是否产生非空、较小的Rank 1概率为目标。\n',
'## 文件与复现\n',
'- [固定分析计划](ANALYSIS_PLAN_CN.md)\n- [完整拟合参数](tables/tail_fits.csv)\n- [三个运行期Top-20汇总](tables/candidate_Top20_summary.csv)\n- [全部阈值与模型预测](tables/candidate_predictions_all.csv)\n- [拟合外检查](tables/heldout_checks.csv)\n- [删除敏感性](tables/deletion_sensitivity.csv)\n- [测试分数超出模型支持的检查](tables/test_support_violations.csv)\n- [原始数据哈希与软件版本](AUDIT.json)\n- [数值检查](TESTS.json)\n\n先运行 `python fit_tail.py`，再运行 `python validate_and_report.py` 生成检查及报告补丁。CSV中的0是分布函数在拟合支持外的数值返回，不是经验证的零风险；`outside_fitted_support`需同时读取。\n',
'方法参考：[Knijnenburg等，2009，Fewer permutations, more accurate P-values](https://doi.org/10.1093/bioinformatics/btp211)；[SciPy genpareto说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html)。本项目借鉴尾部近似，不将置换检验的有效性自动移植到共享源配对背景。\n']
report='\n'.join(lines)
print(json.dumps('*** Begin Patch\n*** Add File: '+(BASE/'尾部模型探索结果_CN.md').as_posix()+'\n'+''.join('+'+l+'\n' for l in report.splitlines())+'*** End Patch'))
