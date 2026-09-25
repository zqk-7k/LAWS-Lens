import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.stats import genpareto

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'0922_GWLR_TAIL02诊断/extracted/gwlr_tail02_20260922T113500Z_r2'
OUT=ROOT/'tables'
OUT.mkdir(exist_ok=True)
RUNS=['O3','O4a','O4b']; BOUNDS=[4.,4.5,5.]
tracked={}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
    tracked[str(p)]=digest(p)
    return pd.read_csv(p)
def fit(x,B,verify=False):
    x=np.asarray(x,float)
    assert np.isfinite(x).all()
    u=float(np.quantile(x,.95));y=x[x>u]-u;E=B-u
    if E<=0 or len(y)<30 or np.max(y)>=E:return None
    logs=np.log1p(-y/E)
    a=float(-len(y)/logs.sum());xi=-1/a;scale=E/a
    loglik=float(np.sum(np.log(a/E)+(a-1)*logs))
    if verify:
        res=minimize_scalar(lambda z: -np.sum(z-np.log(E)+(np.exp(z)-1)*logs),bounds=(np.log(a)-3,np.log(a)+3),method='bounded',options={'xatol':1e-12})
        assert res.success and np.isclose(np.exp(res.x),a,rtol=1e-6)
        assert np.isclose(loglik,genpareto.logpdf(y,xi,scale=scale).sum(),rtol=1e-10,atol=1e-9)
    return dict(B=B,u=u,a=a,xi=xi,scale=scale,n=len(x),tail_n=len(y),tail_mass=len(y)/len(x),loglik=loglik)
def prob(f,s):
    if s>=f['B']:return np.nan
    assert s>=f['u']
    p=float(f['tail_mass']*np.exp(f['a']*np.log((f['B']-s)/(f['B']-f['u']))))
    assert np.isclose(p,f['tail_mass']*genpareto.sf(s-f['u'],f['xi'],scale=f['scale']),rtol=1e-8,atol=1e-15)
    return p

fits=[];candidates=[];checks=[];deletions=[];allreal={}
for run in RUNS:
    bg=read(DATA/f'inputs/{run}_fit_scores.csv')
    test=read(DATA/f'inputs/{run}_check_scores.csv')
    events=read(DATA/f'inputs/{run}_fit_events.csv')
    real=read(DATA/f'results/{run}_real_Top50_empirical_preserved.csv').sort_values('consensus_rank')
    allreal[run]=real
    x=bg.mean_S.to_numpy();h=test.mean_S.to_numpy()
    assert len(x)==len(h)==4005 and events.event_uid.nunique()==90
    assert np.allclose(bg.mean_S,bg[['2026091721','2026091722','2026091723']].mean(axis=1))
    labels=events.set_index('event_uid').noise_parent_uid
    ni=bg.event_i.map(labels);nj=bg.event_j.map(labels)
    assert ni.notna().all() and nj.notna().all() and labels.nunique()==6
    masks=[('source',e,~((bg.event_i==e)|(bg.event_j==e)).to_numpy()) for e in events.event_uid]
    masks += [('noise',n,~((ni==n)|(nj==n)).to_numpy()) for n in labels.unique()]
    mask=np.ones(len(x),bool);mask[int(np.argmax(x))]=False
    masks += [('maximum_pair','maximum_pair',mask)]
    for B in BOUNDS:
        f=fit(x,B,True);assert f
        tail=h[h>f['u']]-f['u']
        above=int((h>=B).sum())
        held_ll=float(genpareto.logpdf(tail,f['xi'],scale=f['scale']).mean()) if above==0 else np.nan
        fits.append(dict(run=run,**f,heldout_above_endpoint=above,heldout_tail_n=len(tail),heldout_mean_tail_logpdf=held_ll))
        for target in [.01,.005,.001,.0005,.00025]:
            threshold=B-(B-f['u'])*(target/f['tail_mass'])**(1/f['a'])
            assert np.isclose(prob(f,threshold),target)
            k=int((h>=threshold).sum())
            checks.append(dict(run=run,B=B,target_probability=target,score_threshold=threshold,heldout_pairs=len(h),expected=len(h)*target,observed=k,observed_expected_ratio=k/(len(h)*target)))
        for _,r in real.iterrows():
            s=float(r.score_mean);k=int((x>=s).sum())
            assert k==int(r.background_exceedances)
            p=prob(f,s)
            candidates.append(dict(run=run,rank=int(r.consensus_rank),pair=r.pair_key,S=s,B=B,empirical_count=k,background_n=len(x),estimated_tail_probability=p,estimated_tail_percent=100*p,outside_support=s>=B,validated_real_FPP=False))
        for kind,key,mask in masks:
            ff=fit(x[mask],B)
            s=float(real.iloc[0].score_mean)
            p=prob(ff,s) if ff and s>=ff['u'] else np.nan
            deletions.append(dict(run=run,B=B,unit=kind,deleted=str(key),rank1_tail_probability=p,rank1_tail_percent=p*100,remaining_pairs=int(mask.sum()),fit_valid=ff is not None))

df=pd.DataFrame(candidates);ft=pd.DataFrame(fits);ck=pd.DataFrame(checks);dd=pd.DataFrame(deletions)
assert len(ft)==9 and len(df)==450 and len(dd)==873
assert df.estimated_tail_probability.between(0,1,inclusive='neither').all()
for _,g in df.groupby(['run','B']):
    assert np.all(np.diff(g.sort_values('S').estimated_tail_probability)<=1e-12)
ds=dd.groupby(['run','B','unit']).agg(fits=('fit_valid','size'),valid=('fit_valid','sum'),rank1_min_pct=('rank1_tail_percent','min'),rank1_max_pct=('rank1_tail_percent','max')).reset_index()
for name,frame in [('fits.csv',ft),('candidate_Top50_all_bounds.csv',df),('heldout_checks.csv',ck),('deletions.csv',dd),('deletion_summary.csv',ds)]:
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:frame.to_csv(f,index=False)
for run in RUNS:
    table=allreal[run].copy()
    for B in BOUNDS:
        d=df[(df.run==run)&(df.B==B)].set_index('pair')
        table[f'fixed_endpoint_{B:g}_tail_percent']=table.pair_key.map(d.estimated_tail_percent)
    with (OUT/f'{run}_Top50_PE_official_fixed_endpoints.csv').open('x',encoding='utf-8-sig',newline='') as f:table.to_csv(f,index=False)
assert tracked=={p:digest(Path(p)) for p in tracked}
audit={'scipy':scipy.__version__,'numpy':np.__version__,'fit_threshold_quantile':.95,'fixed_endpoints':BOUNDS,'mean_score_fits':9,'candidate_evaluations':450,'deletion_fits':873,'analytical_numerical_MLE_agree':True,'GPD_formula_check':True,'all_Top50_inside_endpoints':True,'input_hashes_unchanged':tracked,'no_new_simulation':True,'paper_modified':False,'validated_real_FPP':False}
with (ROOT/'AUDIT.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2,ensure_ascii=False)

def number(v):return f'{v:.8f}' if v>=.000001 else f'{v:.4e}'
lines=['# 固定上界4、4.5、5：GPD尾概率对照', '',
    '日期：2026-09-23。保持95%尾部起点，按用户指定的三个有限上界分别重新拟合三个运行期的平均分背景。此处是固定上界假设下的探索性 FPP 估计，不是已验证真实目录 FPP、候选为假的后验概率或年度 FAR。', '',
    '9组拟合、三运行期各Top-50共450项候选计算完成，另有873次删除敏感性拟合。没有补模拟、重训、改变分数或排名，也没有修改或推送论文。', '',
    '## 1. Rank 1结果', '',
    '| 运行期 | 候选对 | S | 背景计数 | 上界4（%） | 上界4.5（%） | 上界5（%） |',
    '| --- | --- | ---: | --- | ---: | ---: | ---: |']
for run in RUNS:
    rr=df[(df.run==run)&(df['rank']==1)].sort_values('B');r=rr.iloc[0]
    lines.append(f"| {run} | {r['pair']} | {r.S:.6f} | {r.empirical_count}/4005 | "+' | '.join(number(p) for p in rr.estimated_tail_percent)+' |')
lines += ['', '三种上界全部覆盖交付中的三个运行期Top-50，但本次没有检查或承诺整个真实目录所有分数均小于4。表中范围是人为上界敏感性，不是置信区间。', '',
    '## 2. 怎样重新拟合', '',
    '每运行期有90个validation孤立事件、4005个相关配对、6个噪声父块；95%阈值上方为201对。B表示固定上界，u表示95%阈值，y=S−u，E=B−u。', '',
    'a = −n / Σ log(1−y/E)，ξ = −1/a，σ = E/a。', '',
    'FPP<sub>模型</sub>(S) = (n/N) × [(B−S)/(B−u)]<sup>a</sup>，适用于 u≤S<B。', '',
    '这里不是简单改变截断位置，而是在固定端点约束下重估形状和尺度。参数用配对乘积似然求得，解析解已与数值优化及SciPy GPD公式交叉验证。共享事件造成依赖，所以不据此计算独立配对置信区间。三个上界是在看过候选以后指定，不称为预注册，也不称为物理上限。', '',
    '## 3. 背景拟合与留出检查', '',
    '| 运行期 | 固定上界 | 起点u | 形状ξ | 尺度σ | 拟合对数似然 | 旧test尾部平均对数密度 | test超过上界数 |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in ft.itertuples():
    lines.append(f'| {r.run} | {r.B:g} | {r.u:.6f} | {r.xi:.6f} | {r.scale:.6f} | {r.loglik:.3f} | {r.heldout_mean_tail_logpdf:.3f} | {r.heldout_above_endpoint} |')
lines += ['', '同一运行期各设置在相同阈值与相同背景上计算；对数似然、平均对数密度越大表示该批数据上的拟合/预测密度越高，不是显著性或正式选型结论。旧test已在历史分析中使用，不是新增盲测。完整多阈值留出计数见 tables/heldout_checks.csv。', '',
    '| 运行期 | 上界 | 模型名义尾概率 | 4005对期望超越数 | 旧test实测超越数 |',
    '| --- | ---: | ---: | ---: | ---: |']
for r in ck.itertuples():
    if r.target_probability not in [.01,.001]:continue
    lines.append(f'| {r.run} | {r.B:g} | {r.target_probability*100:g}% | {r.expected:.3f} | {r.observed} |')
lines += ['', '## 4. Rank 1删除敏感性（%）', '',
    '每次删去一个源及其全部配对、一个噪声父块相关配对，或最高分背景配对，重算95%阈值和参数，但保持指定上界。源与噪声范围不是置信区间。固定上界使删除后不再发生端点下降越过候选，这来自设定，不代表问题由数据验证解决。', '',
    '| 运行期 | 上界 | 删除单位 | 重拟合数 | 最小尾概率（%） | 最大尾概率（%） |',
    '| --- | ---: | --- | ---: | ---: | ---: |']
for r in ds.itertuples():
    label={'source':'单源','noise':'单噪声块','maximum_pair':'最高分配对'}[r.unit]
    lines.append(f'| {r.run} | {r.B:g} | {label} | {r.fits} | {number(r.rank1_min_pct)} | {number(r.rank1_max_pct)} |')
lines += ['', '## 5. 三运行期完整Top-20', '', '所有尾概率列单位均为百分数；保留原共识排名，而尾概率由平均分计算，故不要求随名次单调。']
for run in RUNS:
    lines += ['',f'### {run}', '', '| Rank | 候选对 | S | 背景计数 | 上界4（%） | 上界4.5（%） | 上界5（%） |', '| ---: | --- | ---: | --- | ---: | ---: | ---: |']
    for rank in range(1,21):
        rr=df[(df.run==run)&(df['rank']==rank)].sort_values('B');r=rr.iloc[0]
        lines.append(f"| {rank} | {r['pair']} | {r.S:.6f} | {r.empirical_count}/4005 | "+' | '.join(number(p) for p in rr.estimated_tail_percent)+' |')
lines += ['', '## 6. 解释边界', '',
    '这些计算满足固定有限上界并给交付Top-50填上数值的要求。数值能否用于真实显著性仍取决于背景可比性、上界依据与预测验证，不能仅因为不再留空而称为更准确。未按最小候选FPP挑选上界，未用候选分数拟合形状/尺度；但上界选择本身已受候选信息影响。', '',
    '[分析设置](分析设置_CN.md) · [计算审计](AUDIT.json) · [全部数值表](tables/candidate_Top50_all_bounds.csv)', '']
print(json.dumps({'markdown':'\n'.join(lines),'rank1':df[df['rank']==1].to_dict('records'),'fits':ft.to_dict('records')},ensure_ascii=False,allow_nan=False))
