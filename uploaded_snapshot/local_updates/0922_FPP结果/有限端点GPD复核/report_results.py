from pathlib import Path
import json
import pandas as pd
BASE=Path(__file__).resolve().parent
T=BASE/'tables'
f=pd.read_csv(T/'finite_mps_fits.csv')
p=pd.read_csv(T/'finite_mps_Top20_all_thresholds.csv')
d=pd.read_csv(T/'finite_mps_deletions.csv')
h=pd.read_csv(T/'finite_mps_test_checks.csv')
aud=json.loads((BASE/'AUDIT.json').read_text(encoding='utf-8'))
assert (f.xi<0).all() and (~f.boundary_hit).all() and f.optimizer_success.all()
lines=['# 有限端点 GPD 复核结果','',
'结论：使用有方法学依据的最大乘积间距估计（MPS），背景数据可以拟合出高于 O4b Rank 1 的有限端点，并计算非零尾概率。但覆盖候选的两个阈值结果仍不稳定，不能选择其中一个作为已验证的正式 FPP。没有使用候选分数作为端点约束，也没有替换论文。','',
'## 1. 方法与依据','',
'本轮分布仍为负形状参数的 GPD，仅将参数估计准则由最大似然换为最大乘积间距。将排序后的超阈值样本映射到拟合 CDF，最大化相邻概率间距（含首尾间距）的对数和。由于最后一段包含最大背景分数到概率1的距离，端点由数据拟合，而不是人为等于样本最大值。该准则不保证在本项目共享源背景上无偏或可靠。','',
'SciPy 官方支持这一准则，`scipy.stats.fit(..., method="mse")` 中的 mse 指负对数乘积间距目标，不是均方误差。','',
'- [SciPy 官方 fit 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html)','- [MPS 原始方法学研究，Shao 与 Hahn，1999](https://www.ism.ac.jp/editsec/aism/51/3.html)','- [SciPy GPD 定义及有限支持](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html)','',
'不能将上述理论对独立样本的性质直接当作本次相互依赖配对的保证。这里只计算模型尾概率，不估计年度 FAR，也不是“候选为假的概率”。','',
'## 2. O4b Rank 1 的具体结果','',
'候选：GW240511_031507--GW241007_082943；平均分 S = 2.984594999759801。','',
'| 拟合阈值 | 拟合尾部样本数 | GPD 形状 ξ | 拟合上限 | Rank 1 模型尾概率（%） |','| --- | ---: | ---: | ---: | ---: |']
for _,r in p[(p.run=='O4b')&(p['rank']==1)].iterrows():
    ff=f[(f.run=='O4b')&(f.q==r.q)].iloc[0]
    value='超出拟合支持，不报告概率' if r.outside_fitted_support else f'{100*r.estimated_tail_probability:.8f}%'
    lines.append(f'| {r.q*100:g}% | {ff.m:d} | {r.xi:.6f} | {r.endpoint:.6f} | {value} |')
lines += ['',
'95%仍是分析计划中的中心阈值；不得因为97.5%或98%阈值使 Rank 1 获得非零数值，就事后将其选为唯一正式结果。两者是同一预设阈值网格中的敏感性结果。','',
'## 3. 删除数据后的稳定性','',
'初算后增加对97.5%和98%阈值的删除诊断，三个运行期同规则执行。本轮共15个完整背景拟合和873个删除拟合，每次重新估计阈值与端点。没有改变旧结果文件。','',
'| O4b 阈值 | 删除单位 | 重拟合次数 | Rank 1 超出新拟合支持次数 | 有效模型尾概率范围（%） | 端点范围 |','| --- | --- | ---: | ---: | --- | --- |']
for q in [.95,.975,.98]:
    for kind,label in [('source','单个源及其配对'),('noise_parent','单个噪声父块'),('maximum_pair','最高分背景配对')]:
        sub=d[(d.run=='O4b')&(d['rank']==1)&(d.q==q)&(d.kind==kind)]
        assert len(sub)=={'source':90,'noise_parent':6,'maximum_pair':1}[kind]
        valid=sub.loc[~sub.outside_fitted_support,'p']
        val='无有效估计' if len(valid)==0 else f'{100*valid.min():.8f}–{100*valid.max():.8f}'
        lines.append(f'| {q*100:g}% | {label} | {len(sub)} | {int(sub.outside_fitted_support.sum())} | {val} | {sub.endpoint.min():.6f}–{sub.endpoint.max():.6f} |')
lines += ['', '表中范围只是删除敏感性范围，不是置信区间；“超出支持”不是零风险。表中只列有效正值范围，不能忽略旁边列出的支持失败次数。','',
'## 4. 已有测试背景的检查','',
'| O4b 拟合阈值 | 名义模型尾概率 | 期望超越数（4005对） | 实际超越数 | 测试背景超出端点数 |','| --- | --- | ---: | ---: | ---: |']
for _,r in h[(h.run=='O4b')&(h.q>=.95)].iterrows():
    lines.append(f'| {r.q*100:g}% | {r.target_probability*100:g}% | {r.expected_count:.3f} | {int(r.observed_count)} | {int(r.test_pairs_outside_endpoint)} |')
lines += ['',
'这批测试数据已在此前实验中使用，不是新增盲测。在中等尾概率处的计数接近，不足以验证万分之一乃至百万分之一的概率。单个小概率对应的期望计数可能远小于1，没有观察到超越不是验证模型正确。','',
'## 5. O4b Top-20 两个有限端点情景','',
'| rank | 候选对 | S | 背景计数 | MPS-GPD，97.5%阈值（%） | MPS-GPD，98%阈值（%） |','| ---: | --- | ---: | --- | ---: | ---: |']
for _,r in p[(p.run=='O4b')&(p.q==.975)].sort_values('rank').iterrows():
    z=p[(p.run=='O4b')&(p.q==.98)&(p['rank']==r['rank'])].iloc[0]
    def fmt(v):return '超出支持' if v.outside_fitted_support else f'{100*v.estimated_tail_probability:.8f}'
    lines.append(f'| {int(r["rank"])} | {r.pair} | {r.S:.6f} | {int(r.empirical_count)}/4005 | {fmt(r)} | {fmt(z)} |')
lines += ['',
'## 6. 其他检查与边界','',
'- 三运行期均计算同一阈值网格；完整 Top-20、测试检查和删除结果见 tables 文件夹。','- 比较的矩估计、概率加权矩估计在若干设置下甚至把已有背景最大值置于支持外，未作为可用替代。','- 评分公式的宽松上界不等于无关总体实际端点，不能据此随意固定GPD的端点。此次没有采用一个人为固定的总分上限。','- 数值目标逐拟合与 SciPy 的间距目标交叉核对，允许误差1e-5；完整拟合没有命中数值搜索边界。','- 配对共享90个源和6个噪声父块，没有将4005对视作4005个独立实验。','- 原始输入哈希复核不变，没有补模拟、重训、改排名或推送论文。','',
'最终状态：可以报告两个有依据的有限端点模型情景；尚未找到经过稳定性验证、可唯一采用的 Rank 1 FPP。']
body='\n'.join(lines)+'\n'
dest=BASE/'有限端点GPD结果_CN.md'
print(json.dumps('*** Begin Patch\n*** Add File: '+dest.as_posix()+'\n'+''.join('+'+s+'\n' for s in body.splitlines())+'*** End Patch',ensure_ascii=False))
