"""Reproduce frozen mean-score MLE fits and tabulate diagnostic tails only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import genpareto

root = Path(__file__).resolve().parent
data = root/'extracted/gwlr_tail02_20260922T113500Z_r2'
out = root/'候选GPD复算'
out.mkdir(exist_ok=True)
saved = pd.read_csv(data/'results/gpd_fits.csv')
mpsfile = root.parent/'0922_FPP结果/有限端点GPD复核/tables/finite_mps_Top20_all_thresholds.csv'
mps = pd.read_csv(mpsfile)
inputs = [mpsfile, data/'results/gpd_fits.csv']
for run in ['O3','O4a','O4b']:
    inputs += [data/f'inputs/{run}_fit_scores.csv', data/f'inputs/{run}_check_scores.csv', data/f'results/{run}_real_Top50_empirical_preserved.csv']
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
before = {str(p):sha(p) for p in inputs}
rows, fitrows, joined = [], [], []
def fmt(p):
    return '超出拟合上界' if p is None or pd.isna(p) else f'{100*p:.8f}'
lines = ['# TAIL-02 诊断与候选 GPD 尾概率复算', '',
    '本次复算三个运行期平均分 S 的 GPD 最大似然拟合（MLE），沿用 TAIL-02 的 90%、95%、97.5% 起点，不依据真实候选选阈值。95% 仍是主要起点。另列此前 MPS 有限端点探索的 97.5%、98% 情景，明确不是 TAIL-02 发布的新结果。', '',
    '**所有 GPD 列均为未获验证的模型尾概率，不是正式真实目录 FPP、年度 FAR 或候选为假的后验概率。超出拟合上界时不报告零概率。**', '',
    '经验比例按 k/4005 计算；零超越显示“—（0/4005）”。数值列单位均为百分数（%），例如 0.00033590 表示 0.00033590%，不是概率 0.00033590。排名保持原共识排名；FPP 按平均分映射，所以不要求随共识名次单调。', '',
    '## 1. TAIL-02 检验结果', '',
    '本包完成 12 种统计量、36 次拟合和 4,800 次节点重采样敏感性计算。预设筛查只在背景 98%、99%、99.5% 分位数处进行；30 个真实 Top-10 分数全部高于最高检查位置。两侧每运行期各 90 个事件、4,005 个相关配对、6 个噪声父块，不能把配对当成 4,005 个独立样本。新增已评分背景事件为 0。', '',
    '下表是此次本地复算的平均分拟合与旧 test 检查，非新增盲测。', '',
    '| 运行期 | 尾部起点分位数 | 阈值 u | 尾部对数 | 形状 ξ | 有限端点 | 留出背景超过端点数 |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for run in ['O3','O4a','O4b']:
    x = pd.read_csv(data/f'inputs/{run}_fit_scores.csv')['mean_S'].to_numpy()
    h = pd.read_csv(data/f'inputs/{run}_check_scores.csv')['mean_S'].to_numpy()
    real = pd.read_csv(data/f'results/{run}_real_Top50_empirical_preserved.csv').sort_values('consensus_rank').head(20)
    assert len(x)==len(h)==4005 and len(real)==20
    fits = {}
    for q in [.9,.95,.975]:
        u=float(np.quantile(x,q)); y=x[x>u]-u
        xi, loc, scale=genpareto.fit(y,floc=0.)
        assert xi<0 and loc==0 and scale>0
        endpoint=u-scale/xi
        old=saved[(saved.run==run)&(saved.model=='mean_S')&np.isclose(saved.q,q)].iloc[0]
        assert np.allclose([xi,scale,endpoint],[old['shape'],old['scale'],old['endpoint']],rtol=1e-5,atol=1e-6)
        fits[q]=(u,xi,scale,endpoint,len(y)/len(x))
        fitrows.append(dict(run=run,q=q,u=u,xi=xi,scale=scale,endpoint=endpoint,tail_n=len(y),heldout_above=int((h>=endpoint).sum())))
        lines.append(f'| {run} | {100*q:g}% | {u:.6f} | {len(y)} | {xi:.6f} | {endpoint:.6f} | {int((h>=endpoint).sum())} |')
    for _,r in real.iterrows():
        s=float(r.score_mean); k=int((x>=s).sum())
        assert k==int(r.background_exceedances)
        display=dict(run=run,rank=int(r.consensus_rank),pair=r.pair_key,S=s,k=k,empirical_pct=k/4005*100)
        for q,(u,xi,scale,endpoint,mass) in fits.items():
            outside=s>=endpoint
            assert s>u
            p=None if outside else float(mass*genpareto.sf(s-u,xi,scale=scale))
            if p is not None:
                assert np.isclose(p, mass*(1+xi*(s-u)/scale)**(-1/xi)) and 0<p<1
            rows.append(dict(run=run,rank=int(r.consensus_rank),pair=r.pair_key,S=s,q=q,endpoint=endpoint,tail_probability=p,tail_percent=None if p is None else p*100,outside_support=outside,validated_real_FPP=False))
            display[f'MLE_{q}']=p
        for q in [.975,.98]:
            v=mps[(mps.run==run)&(mps['rank']==int(r.consensus_rank))&np.isclose(mps.q,q)].iloc[0]
            assert v['pair']==r.pair_key and np.isclose(v.S,s)
            display[f'MPS_{q}']=None if v.outside_fitted_support else float(v.estimated_tail_probability)
        joined.append(display)
lines += ['', 'O3/O4a 的旧 test 出现超出拟合端点的配对。O4b 旧 test 未出现此现象，但 Rank 1 高于三个拟合端点。全部候选的正式 GPD-FPP 发布门槛仍未通过。', '',
    '## 2. 三运行期 Top-20：直接计数与 GPD 数值', '',
    'MLE 为本次复算；MPS 为此前已计算结果的逐对核对展示。MPS 两个情景不能因为能给 Rank 1 非零值而挑选为正式结果。较低起点（90%、92.5%、95%）的 MPS 在 O4b Rank 1 均超出上界，完整历史表仍保留。']
for run in ['O3','O4a','O4b']:
    lines += ['', f'### {run}', '', '| Rank | 候选对 | S | 经验比例%（计数） | MLE 90%（%） | MLE 95%（%） | MLE 97.5%（%） | MPS 97.5%（%） | MPS 98%（%） |', '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in joined:
        if r['run']!=run: continue
        emp=f"{r['empirical_pct']:.4f}（{r['k']}/4005）" if r['k'] else '—（0/4005）'
        pp=' | '.join(fmt(r[key]) for key in ['MLE_0.9','MLE_0.95','MLE_0.975','MPS_0.975','MPS_0.98'])
        lines.append(f"| {r['rank']} | {r['pair']} | {r['S']:.6f} | {emp} | {pp} |")
lines += ['', '## 3. 这些数值应怎样使用', '',
    '- 经验列是有限背景中直接数到的超越比例；GPD 列是假设尾部分布成立时的平滑估计。二者差异本身也是拟合诊断信息。',
    '- 这次只复算固定模型的平均分尾概率，没有补模拟、重训、改变排名或推送论文。逐 seed 的原始诊断仍在 TAIL-02 包内。',
    '- O4b Rank 1 的两个 MPS 正值对阈值和删除背景很敏感。删除最高分背景配对后，两情景端点均不再覆盖 Rank 1；不能把这些数值报告成稳定测量。',
    '- TAIL-02 的正式结论仍是保留经验计数、不发布候选 GPD-FPP。此处应用户要求列出数学复算值，不改变该结论。', '',
    '[TAIL-02 原报告](../extracted/gwlr_tail02_20260922T113500Z_r2/reports/READ_FIRST_TAIL02_CN.md) · [此前 MPS 完整敏感性报告](../../0922_FPP结果/有限端点GPD复核/有限端点GPD结果_CN.md)', '']
assert before=={str(p):sha(p) for p in inputs}
for name,frame in [('MLE_recalculated_fits.csv',pd.DataFrame(fitrows)),('MLE_Top20_all_thresholds.csv',pd.DataFrame(rows)),('COMPARISON_Top20.csv',pd.DataFrame(joined))]:
    if (out/name).exists():
        prior=pd.read_csv(out/name)
        pd.testing.assert_frame_equal(prior,frame,check_dtype=False,rtol=1e-12,atol=1e-14)
    else:
        with (out/name).open('x',encoding='utf-8-sig',newline='') as f: frame.to_csv(f,index=False)
audit=dict(scipy=scipy.__version__,fits_recomputed=9,fit_matches_server=True,candidate_threshold_evaluations=len(rows),input_hashes=before,unchanged_inputs=True,paper_modified=False,validated_real_FPP=False)
if (out/'AUDIT.json').exists():
    assert json.loads((out/'AUDIT.json').read_text(encoding='utf-8'))==audit
else:
    with (out/'AUDIT.json').open('x',encoding='utf-8') as f: json.dump(audit,f,indent=2,ensure_ascii=False)
print(json.dumps({'markdown':'\n'.join(lines),'rank1':[r for r in joined if r['rank']==1],'O4b_top10':[r for r in joined if r['run']=='O4b' and r['rank']<=10]},ensure_ascii=False,allow_nan=False))
