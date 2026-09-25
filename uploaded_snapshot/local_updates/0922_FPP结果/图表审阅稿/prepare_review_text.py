"""Emit an apply_patch for review documents; never modify the manuscript."""
from pathlib import Path
import pandas as pd,json
B=Path(__file__).resolve().parent
R=next((B.parent/'extracted').iterdir())
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def flag(v):return str(v).lower() in ['true','1','1.0']
def pct(v):return f'{v*100:.4f}%'
parts=['''# GWLR-FPP-01 图4与表格审阅稿

本次只生成审阅稿，**未覆盖旧图、未改模型与排名、未推送 Overleaf**。主方法仍为 C_PHYSICAL，本轮增加背景评价，不增加透镜确认结论。

## 1. 文件入口

- [新版图4：PNG预览](figures/Fig4_FPP_review.png) ／ [矢量PDF](figures/Fig4_FPP_review.pdf) ／ [可编辑SVG](figures/Fig4_FPP_review.svg)
- [三运行期Top-20候选表PDF：6页，每页10对](tables/Tables_candidates_Top20_review.pdf)
- [1%阈值性能表PDF](tables/Table_efficiency_FPP1_review.pdf) ／ [预览](tables/Table_efficiency_FPP1_review.png)
- 三运行期完整中文候选表见下文；[绘图数据](source_data/candidate_top20_all_runs.csv)含分组删除敏感性范围。

## 2. 图4怎么看？

![图4审阅稿](figures/Fig4_FPP_review.png)

### 上排 a–c：一个分数对应多少背景超越？

从左到右是 O3、O4a、O4b。横轴是三通道分数阈值 S，纵轴是**条件FPP，单位为百分数**。

每个运行期有90个模拟孤立源，形成4005个背景对。对于某个分数，统计这4005对中有多少对得分不低于它：FPP = k/4005。紫色阶梯曲线使用全部背景分数，不拟合或外推极端尾部。横轴使用反双曲正弦尺度，刻度仍是原始分数，负分没有取普通对数。

- 紫色曲线：分数越高，能够超过它的背景对越少。
- 灰色虚线：1%背景阈值。它不是“候选有99%透镜概率”。
- 紫色圆点／方点：分别保留至少50%／90%注入真透镜对的分数阈值，包含并列分数。
- 浅灰点线：1/4005，约0.02497%，表示一个超越计数的网格间距，不是置信上限。

三个运行期的90%恢复标记都在1%虚线上方。这说明：想保留90%的真透镜对，需要容许超过1%的背景尾部比例；若坚持1%阈值，保留率约为74%／74%／81%。这是召回能力与无关候选负担的权衡。

### 下排 d–f：实际候选落在背景的哪个位置？

横轴是各运行期原有的Top-20共识名次，纵轴是该候选平均分对应的条件FPP，不重新排序。

- 绿色空心圆：有1–19个背景对超过或达到候选分数，尾部计数较少。
- 绿色实心圆：至少20个背景对达到候选分数。20只是支持量警示的界线，不是真实性通过门槛。
- 下方灰色带中的橙色叉号：零背景超越，**尾部未分辨**。这条带没有数值概率坐标，叉号不代表FPP等于0或小于某个统计上限。

O3／O4a／O4b的Top-20都在本轮经验1%阈值以内，但 O4a Rank 1–6和O4b Rank 1没有背景超越。不能仅凭这些零计数宣称它们最可靠。原排名依据三个模型的名次汇总，而本图使用平均分尾部，所以FPP不必随名次严格增加。

### 图4能够与不能够表达什么

能够表达：候选分数相对当前无关配对背景的位置，以及提高保留率要付出的背景代价。

不能表达：候选透镜概率、每年误报次数、整个真实目录的发现显著性。背景只有90个源和6个噪声父块，配对相关，而且该validation已用于选模型／系数。图中的函数不是独立概率校准的保证。

曲线和点均由三模型平均分重新查询背景得到，**不是三个模型FPP的均值**。因此没有给它们套用旧模型SD误差棒。逐源／逐噪声父块删除敏感性范围随数据表保留，不能称为置信区间。

## 3. 表A：1%阈值下的注入表现

这里固定条件FPP < 1%，而不是每查询只保留前十候选。
''']
eff=pd.read_csv(B/'source_data/efficiency_at_one_percent.csv')
parts.append(table(['运行期','保留真透镜对','Pair recall','保留无关对','所有无关对实际超阈值率','纯孤立源配对实际超阈值率'],[[r.run,f'{int(r.true_pairs_retained)}/180',f'{r.true_pair_recovery_fraction:.2%}',int(r.false_pairs_retained),pct(r.noncompanion_tail_fraction),pct(r.test_singleton_tail_fraction)] for _,r in eff.iterrows()]))
parts.append('''每期test有450个事件，共180个真透镜对、100845个无关对；纯孤立源子集另外形成4005对。最后两列的分母不同，不能混用。O4a纯孤立源检查为1.4232%，所以不能声称名义1%已保证测试或真实目录误报率小于1%。

这一表的意义是：说明筛选能留下多少真对，同时留下多少假对。留下数百个无关对意味着小FPP不等于候选列表高纯度。

### 原C的R@10仅作对照，没有变化

| 运行期 | 190事件、189候选／查询的三通道R@10 |
| --- | --- |
| O3 | 0.9410 ± 0.0161 |
| O4a | 0.9345 ± 0.0089 |
| O4b | 0.9665 ± 0.0043 |

这是原C的三个模型均值±样本SD，每模型先平均500个固定子目录。本次Pair recall使用完整450事件的平均分阈值，两种指标和汇总方式不同，不能拿74%与94%直接判定性能退化。

## 4. 表B：候选分数、PE与背景评价

沿用八列候选表并新增两列：**背景超越数k/4005**和**本文条件FPP**。官方FPP保留小数比例，例如0.00545约为0.545%；本文FPP明确带百分号。两列不能混为同一种校准。

- S及wf／time／sky：原三模型平均总分及平均加权贡献。
- BC：公开Mc、q、χeff和表观距离的边际后验重合；不进入本轮FPP计算。
- Dmax：项目的描述性参数距离，不是统一的透镜确证标准。
- 背景超越数：让读者看到小FPP背后的计数支持。
- “未分辨”：经验计数为0，但不能当成0风险。
- 官方阶段：只记录公开前端和Hanabi表的匹配；未运行新的Hanabi。
''')
for run in ['O3','O4a','O4b']:
    df=pd.read_csv(R/f'tables/{run}/real_GWTC_Top20_FPP.csv').sort_values('consensus_rank');rows=[]
    for _,r in df.iterrows():
        if run=='O4b':fpp='不可核验';stage='公开逐对结果不可核验；未运行新Hanabi'
        else:
            fpp=f'{r.official_po_fpp:.5f} / {r["official_ml_fpp" if run=="O3" else "official_phazap_fpp"]:.5f}'
            stage=('通过' if flag(r.official_frontend) else '未通过')+'官方1%前端；'+('公开Hanabi匹配，既有分析未支持透镜' if flag(r.public_Hanabi_overlap) else '未匹配公开Hanabi逐对表')
        if r.Dmax>3:stage+='；Dmax > 3'
        rows.append([int(r.consensus_rank),r.pair_key,f'{r.score_mean:.4f}',' / '.join(f'{r[c]:+.4f}' for c in ['wf_contribution_mean','time_contribution_mean','sky_contribution_mean']),' / '.join(f'{r[c]:.4f}' for c in ['BC_Mc','BC_q','BC_chi_eff','BC_apparent_distance']),f'{r.Dmax:.4f}',fpp,stage,f'{int(r.background_exceedances)}/4005','未分辨' if r.zero_exceedance_unresolved else pct(r.conditional_FPP)])
    parts.extend([f'### {run} Top-20',table(['rank','候选对','S','wf / time / sky','BC：M<sub>c</sub> / q / χ<sub>eff</sub> / d<sub>L</sub><sup>app</sup>','Dmax','官方 FPP：'+('PO / ML' if run=='O3' else 'PO / Phazap' if run=='O4a' else '不可核验'),'官方阶段和结论','背景超越数','本文条件FPP'],rows)])
parts.append('''## 5. 审核后建议怎样入文

图4上排改为“分数—背景FPP”，下排显示真实Top-20；替换旧年化图，不变图2／3的分数与检索结果。O4b主候选表用本文FPP及超越计数替换旧年化列；O3／O4a／O4b补充Top-20表同步。表A适合加入补充性能表，也可以在正文简要报告其保留率与假对数。

本次PDF候选表为便于审核采用横向A4和每页10对，不是直接压缩到论文栏宽的最终排版。正式正文表可保留关键列，完整十列表放补充材料。

## 6. 图注审阅稿

**图4｜三运行期的模拟背景尾部与候选分数评价。** a–c，O3、O4a和O4b三通道平均分阈值与条件FPP的关系。每运行期背景由90个模拟孤立源组成，共4005个无序对，涉及6个噪声父块。FPP为分数不低于阈值的背景对占比。圆点和方点分别标记至少保留50%和90%注入真透镜对的阈值，评价使用180个真对并计入并列分数。d–f，冻结Top-20共识候选的平均分条件FPP。空心圆表示1–19个背景超越，实心圆表示至少20个；橙色叉号在独立非数值标记带中表示零超越、尾部未分辨。虚线为1%，点线为1/4005的计数间距而非置信上限。曲线未作尾部外推，不提供置信区间。背景validation曾用于模型／系数选择，且配对共享源和噪声。本图不是年度FAR、真实目录校准显著性或透镜后验概率。

**Figure 4 | Simulated background tails and candidate scores in O3, O4a and O4b.** a–c, Conditional FPP as a function of the three-channel score averaged across three models. Each run uses 4,005 unordered pairs of 90 simulated isolated sources associated with six noise blocks. FPP is the fraction of background pairs with scores at least as high as the threshold. Circles and squares mark thresholds retaining at least 50% and 90% of the 180 injected lensed pairs, including ties. d–f, Conditional FPP at the mean scores of the frozen Top-20 consensus candidates. Open circles denote 1–19 background exceedances and filled circles denote at least 20. Orange crosses in a separate nonnumeric strip indicate zero exceedances and unresolved tails. Dashed lines mark 1%. Dotted lines show the one-count spacing, 1/4,005, not a confidence bound. No tail extrapolation or confidence intervals are shown. The validation data previously informed model and coefficient selection, and pairs share sources and noise. These background comparisons are not annual false-alarm rates, calibrated real-catalog significances or lensing posterior probabilities.

## 7. 检查记录

- 原数据、原图和论文未覆盖；无推送。
- 每运行期全部4005个背景对进入计数，真实Top-20共60对完整显示；保留原排名。
- Python后端，Arial字体。主图183×155 mm，PDF／SVG可编辑，PNG／TIFF 600 dpi。
- 静态检查无FAIL。LOG-GUARD警示为保守扫描结果：代码已断言背景曲线严格正值，候选概率只绘制k>0行，k=0放独立标记带，不加伪计数。
- PDF页面边界和字号检查通过；表格逐单元格检查内容不超出边界。候选PDF的上下标字形亦满足5pt底线。
- 统计技能检查落实在零计数展示、平均顺序、配对依赖、指标分母和置信区间边界；没有把验证背景尾部比例改写成真实概率。
''')
text='\n\n'.join(parts)+'\n'
patch='*** Begin Patch\n*** Add File: '+str(B/'图表审阅说明_CN.md')+'\n'+''.join('+'+s+'\n' for s in text.splitlines())
# Optional LaTeX candidate-table fragment for later integration, no manuscript write.
df=pd.read_csv(R/'tables/O4b/real_GWTC_Top10_FPP.csv')
tex=['% Review fragment only. Requires booktabs. Not inserted into the manuscript.',r'\begin{tabular}{rlrrrr}',r'\toprule',r'Rank & Candidate pair & $S$ & $BC_{M_c}$ & $k/4005$ & Conditional FPP \\',r'\midrule']
for _,r in df.iterrows():
    key=r.pair_key.replace('_',r'\_')
    f='Unresolved' if r.zero_exceedance_unresolved else pct(r.conditional_FPP).replace('%',r'\%')
    tex.append(f'{int(r.consensus_rank)} & {key} & {r.score_mean:.4f} & {r.BC_Mc:.4f} & {int(r.background_exceedances)}/4005 & {f} '+r'\\')
tex += [r'\bottomrule',r'\end{tabular}','% Conditional FPP uses the finite simulation background; unresolved means zero exceedances, not zero risk.']
patch+='*** Add File: '+str(B/'tables/O4b_Top10_FPP_review.tex')+'\n'+''.join('+'+s+'\n' for s in tex)+'*** End Patch'
print(json.dumps({'patch':patch},ensure_ascii=False))
