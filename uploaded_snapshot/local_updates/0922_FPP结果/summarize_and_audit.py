"""Read immutable deliveries, independently check arithmetic, emit an apply_patch."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parent
R=next((B/'extracted').iterdir())
C=B.parent/'0919结果/extracted/completion'
runs=['O3','O4a','O4b']
def rd(p):return pd.read_csv(p)
def tab(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x,n=4):return '不可核验' if pd.isna(x) else f'{float(x):.{n}f}'
def truth(v):return str(v).lower() in ['true','1','1.0']
def stage(r):
    if r['run']=='O4b':return '公开逐对官方结论不可核验；未运行新 Hanabi'
    out=('通过' if truth(r['official_frontend']) else '未通过')+'官方 1% 前端；'
    if truth(r['public_Hanabi_overlap']):out+='公开 Hanabi 逐对表匹配，既有分析未支持透镜'
    else:out+='未匹配公开 Hanabi 表；不能据此确定是否分析过'
    if r['BC_Mc']<.1 or r['D_Mc']>5:out+='；严重 Mc 不相容'
    elif r['Dmax']>3:out+='；Dmax 超过描述性阈值 3'
    elif r['BC_Mc']<.5:out+='；Mc BC < 0.5'
    return out
audits=[];reals={};eff=[];heads=[];budget=[];tailrows=[]
sourcefiles=[]
for run in runs:
    bg=rd(R/f'background/{run}/mean_S_calibration_pairs.csv')
    events=rd(R/f'background/{run}/calibration_events.csv')
    assert len(bg)==4005 and len(events)==90 and events.source_uid.nunique()==90 and events.noise_parent_uid.nunique()==6
    inj=pd.read_parquet(R/f'tables/{run}/injection_mean_S_FPP.parquet')
    real=rd(R/f'tables/{run}/real_GWTC_mean_S_FPP.csv').sort_values('consensus_rank')
    reals[run]=real
    for model in ['2026091721','2026091722','2026091723','mean_S']:
        bb=rd(R/f'background/{run}/{model}_calibration_pairs.csv')
        null=np.sort(bb.final_score_POSITIVE.to_numpy())
        tag='mean_S' if model=='mean_S' else 'seed_'+model
        for domain,prefix in [('injection','injection'),('real','real_GWTC')]:
            f=pd.read_parquet(R/f'tables/{run}/{prefix}_{tag}_FPP.parquet')
            k=len(null)-np.searchsorted(null,f.final_score_POSITIVE.to_numpy(),side='left')
            np.testing.assert_array_equal(k,f.background_exceedances)
            np.testing.assert_allclose(k/len(null),f.conditional_FPP,rtol=1e-12,atol=1e-15)
            np.testing.assert_allclose(f.iloc[:,0:0].shape[0],len(f))
            assert f.FAR_per_year.isna().all() and f.validated_GWTC_FPP.isna().all()
            np.testing.assert_array_equal(k==0,f.zero_exceedance_unresolved)
            audits.append({'run':run,'model':model,'domain':domain,'all_rows_recounted':len(f),'passed':True})
    old=rd(C/f'real_PE/{run}/C_PHYSICAL/three-channel/all_pairs_with_PE_official.csv').set_index('pair_key')
    rr=real.set_index('pair_key').loc[old.index]
    np.testing.assert_array_equal(old.consensus_rank,rr.consensus_rank)
    for col in ['score_mean','wf_contribution_mean','time_contribution_mean','sky_contribution_mean','BC_Mc','BC_q','BC_chi_eff','BC_apparent_distance','Dmax']:
        np.testing.assert_allclose(old[col],rr[col],rtol=1e-11,atol=1e-12,equal_nan=True)
    assert len(inj)==101025 and inj.is_true_pair.sum()==180
    choose=inj.conditional_FPP<.01
    tp=int((choose & inj.is_true_pair).sum());fp=int((choose & ~inj.is_true_pair).sum())
    e=rd(R/'summaries/injection_efficiency_at_FPP.csv').query('run==@run and model=="mean_S" and FPP_threshold==0.01').iloc[0]
    assert tp==e.true_pairs_retained and fp==e.false_pairs_retained
    eff.append([run,f'{tp}/180',f'{tp/180:.2%}',fp,f'{fp/100845:.4%}',f'{e.test_singleton_tail_fraction:.4%}'])
    top=real.head(10);first=top.iloc[0]
    heads.append([run,int((top.conditional_FPP<.01).sum()),int(top.zero_exceedance_unresolved.sum()),int(top.low_tail_count.sum())])
    tailrows.append([run,first.pair_key,int(first.background_exceedances), '未分辨（0/4005）' if first.zero_exceedance_unresolved else f'{first.conditional_FPP:.4%}',f'{first.conditional_expected_false_pairs:.3f}',f'{first.conditional_catalog_FPP_mc:.1%}'+('（零超越未分辨）' if first.catalog_zero_exceedance_unresolved else '')])
    for n in [10,20,50,100]:
        t=real.head(n)
        budget.append([run,f'Top-{n}',int((t.BC_Mc>=.5).sum()),int((t.Dmax<=3).sum()),int(((t.BC_Mc<.1)|(t.D_Mc>5)).sum()),int(t.official_frontend.map(truth).sum()) if run!='O4b' else '不可核验',int(t.public_Hanabi_overlap.map(truth).sum()) if run!='O4b' else '不可核验',int((t.conditional_FPP<.01).sum()),int(t.zero_exceedance_unresolved.sum())])
    # Independent isolation check against archived C test event grouping.
    test=pd.read_parquet(C/f'deployments/{run}/C_PHYSICAL/evaluation/test/seed_2026091721/events.parquet')
    for field in ['source_uid','global_source_id','noise_parent_uid']:
        assert not set(events[field])&set(test[field]),field
    sourcefiles.extend([R/f'background/{run}/mean_S_calibration_pairs.csv',R/f'tables/{run}/real_GWTC_mean_S_FPP.csv',C/f'real_PE/{run}/C_PHYSICAL/three-channel/all_pairs_with_PE_official.csv'])

parts=['''# 0922 精炼版：GWLR-FPP-01 背景评价、Top-20 与 Recall

## 1. 先看结论：能否放入论文？

**可以作为当前 C 主结果的新增背景评价，但不是新模型升级，也不是透镜确认或年度 FAR。** 主方法仍为 TriLens，主结果版本仍为 **GWLR-UC-01 / C_PHYSICAL**；本轮后处理与审计版本为 **GWLR-FPP-01**，交付修正版 `20260922T061000Z_r2`。

它为原分数添加一个可复算的背景尾部位置，给出阈值下能保留多少透镜对、同时留下多少无关对。真实候选表可以附加此指标，名称必须与官方 PO／ML／Phazap FPP 分开。本轮未修改模型、权重、PE 或共识排名，没有补模拟，没有推送或覆盖论文。

## 2. 这次具体算什么？

每个运行期从 C 的 validation 中取 90 个孤立源，排除所有透镜像，形成 4,005 个无序配对。三运行期各有自己的背景，不混用。背景涉及 6 个噪声父块。

FPP<sub>cond</sub>(s) = k(s) / 4,005，其中 k(s) 为背景中分数 S ≥ s 的无关配对数量。

例如 O3 Rank 1 有 1 个背景对得分不低于它，经验条件 FPP = 1/4,005 ≈ 0.02497%。这不是“该候选有 99.975% 的概率为透镜”。FPP<sub>cond</sub>描述指定背景中的尾部比例，不是候选所属类别的后验概率。

先对同一配对的三个模型分数取均值得到 mean S，再用背景的 mean S 建立尾部。**平均分的 FPP 不等于三个模型 FPP 的平均值。** 原候选依旧按各模型平均名次排序，mean S 的 FPP 不是共识排名选择流程的整体显著性。

### 与此前图4的区别

此前图4使用完整 test 的 100,845 个非伴随对，并按目标目录对数及日历做年化。本轮使用独立于 test 源／噪声的 validation 孤立源背景 4,005 对，不再除以日历。背景组成、样本量和模型汇总方式都有变化，数值不能直接当作旧年化指标的单位转换。

validation 曾用于模型／系数选择，它不是全新的独立概率校准集。test 也是历史已查看数据，不能把本轮描述成新增独立盲测。

## 3. 注入：条件 FPP < 1% 时保留什么？

本表以平均分查询条件 FPP，使用完整 450 事件 test：180 个透镜双像系统＋90 个孤立事件，共 180 个真透镜对、100,845 个非伴随对。非伴随对包括不同透镜系统的像之间的配对，并不全是孤立源配对。''',tab(['运行期','保留真透镜对','Pair recall','保留非伴随对','全部非伴随对实际超阈值率','纯孤立 test 配对超阈值率'],eff),'''
这里是 **pair recall**：满足固定阈值的真透镜对数除以 180，不能替代每查询前十召回率 R@10。筛选中仍有数百个非伴随对，不应将 FPP < 1% 解释为筛选后列表有 99% 纯度。

纯孤立 test 配对也是 4,005 对，超阈值率 O3/O4a/O4b 分别约 0.849%／1.423%／0.949%。尤其 O4a 的经验值高于名义 1%，不能写成“保证实际误报率低于 1%”。共享源与噪声使配对相关，尚不能据此用简单二项检验判断偏差显著性。

## 4. 原 C 的检索结果：未重训，未改变

本节回列冻结 C 结果，不是本次 FPP 分析产生的新 Recall。190 事件的每查询候选数为 189；每模型先平均 500 个固定子目录，再汇总三个模型的均值与样本 SD。子目录重用源和噪声，并非 500 次独立实验。''']
methods={'waveform-new':'完整波形','time-only':'时间','sky-only':'天空','three-channel':'三通道'}
for n in [190,450]:
    df=rd(C/f'tables/retrieval_summary_{n}.csv');out=[]
    for run in runs:
        for m,name in methods.items():
            r=df.query('run==@run and split=="test" and method==@m').iloc[0]
            def ms(k,d=4):return f'{r[k+"_mean"]:.{d}f} ± {r[k+"_std"]:.{d}f}'
            out.append([run,name,ms('macro_r_at_1'),ms('macro_r_at_10'),ms('average_precision'),ms('false_at_recall_0p5',1),ms('false_at_recall_0p9',1)])
    parts.extend([f'### {n} 事件，{n-1} 候选／查询',tab(['运行期','方法','R@1','R@10','Pair AUPRC','F50 假对数','F90 假对数'],out)])
parts.extend(['时间、天空在同一运行期由各模型共享，模型 SD 为零不代表抽样不确定性为零。F50／F90 是达到 50%／90% pair recall 的假对数量，不是次／年的 FAR。','## 5. 真实候选与 PE／官方重合',tab(['运行期','Top-10 本文条件 FPP < 1%','其中零超越','尾部计数 < 20'],heads),tab(['运行期','预算','Mc BC ≥ 0.5','Dmax ≤ 3','严重 Mc 冲突','官方 1% 前端','公开 Hanabi 匹配','本文条件 FPP < 1%','本文零超越'],budget),'''
“本文条件 FPP”和“官方前端 FPP”是不同统计量及背景的结果。官方前端按 PO 或 ML（O3）、PO 或 Phazap（O4a）任一 FPP < 1% 计数。公开 Hanabi 匹配不等于本轮运行过 Hanabi。O4b 没有可核验官方逐对对照，不能记为 0 对。

### Rank 1：单对尾部小，不代表整个目录很罕见
''',tab(['运行期','Rank 1 候选','背景超越数 k','条件 FPP','有限池匹配目录的期望假对数','有限池目录最大分超越比例'],tailrows),'''
这里的目录指标来自同一 90 源池内无放回抽取 1,000 个 62／74／86 事件目录。它只说明这个有限池下的目录效应，不是 1,000 个独立总体背景，也不是正式真实目录 FPP。期望假对数用目标对数 × 条件 FPP，由期望线性性得到，不要求配对独立。零超越导致的零值仍属尾部未分辨。

### Top-20 阅读规则

以下保留原八列格式。S 为三模型平均分，wf／time／sky 为平均加权贡献，数值显示舍入可能导致末位加和差异。排名仍为冻结共识排名。BC 为公开边际后验重合，Dmax 是项目描述性距离指标，不是统一的透镜确证标准。每张候选表后单列本轮条件 FPP，避免与官方 FPP 混淆。
'''])
for run in runs:
    data=reals[run].head(20);out=[];frows=[]
    for _,r in data.iterrows():
        official='不可核验' if run=='O4b' else fmt(r.official_po_fpp,5)+' / '+fmt(r['official_ml_fpp' if run=='O3' else 'official_phazap_fpp'],5)
        out.append([int(r.consensus_rank),r.pair_key,fmt(r.score_mean),' / '.join(f'{r[c]:+.4f}' for c in ['wf_contribution_mean','time_contribution_mean','sky_contribution_mean']),' / '.join(fmt(r[c]) for c in ['BC_Mc','BC_q','BC_chi_eff','BC_apparent_distance']),fmt(r.Dmax),official,stage(r)])
        frows.append([int(r.consensus_rank),f'{int(r.background_exceedances)}/4005','未分辨（经验计数为零）' if r.zero_exceedance_unresolved else f'{r.conditional_FPP:.4%}',f'{r.source_leave_one_out_min:.4%}–{r.source_leave_one_out_max:.4%}',f'{r.noise_leave_one_out_min:.4%}–{r.noise_leave_one_out_max:.4%}','零超越，未分辨' if r.zero_exceedance_unresolved else '尾部少于 20 对' if r.low_tail_count else '计数可读，不代表已校准'])
    parts.extend([f'### {run} 冻结 Top-20',tab(['rank','候选对','S','wf / time / sky','BC：M<sub>c</sub> / q / χ<sub>eff</sub> / d<sub>L</sub><sup>app</sup>','Dmax','官方 FPP：'+('PO / ML' if run=='O3' else 'PO / Phazap' if run=='O4a' else '不可核验'),'官方阶段和结论'],out),f'### {run} Top-20 本轮新增条件 FPP',tab(['rank','背景超越计数','条件 FPP','删一个源的敏感性范围','删一个噪声父块的敏感性范围','尾部标记'],frows)])
parts.append('''## 6. 论文放在哪里，哪些不能直接替换？

| 位置 | 建议修改 | 必须保留的口径 |
| --- | --- | --- |
| 图4 a–c | 改为平均分阈值 S 与条件 FPP 的关系，标注 1% 阈值及对应 pair recall；不再显示年化 FAR | 使用 4,005 对背景；零超越尾部停止曲线或独立标记，不向下外推 |
| 图4 d–f | O3／O4a／O4b Top-20 的条件 FPP，配合尾部计数和源／噪声删除敏感性展示 | 零超越写“未分辨”，不能画成零风险或画成 1/4005 的统计上限；敏感性范围不是置信区间 |
| O4b 正文候选表（当前主表） | 用本文条件 FPP、超越数 k／4005、未分辨标记替换年化列 | 不改变排名、PE、加权贡献；不把此列称为官方 FPP |
| O3／O4a 候选展示及三运行期补充 Top-20 表 | 添加或替换本文背景列；保留独立的官方 PO／ML／Phazap 列 | O3／O4a 官方重合数保持原 C 结果，不能改成本文 10/10 |
| 表1或新增补充性能表 | 增加“条件 FPP < 1% 时 pair recall／假对数量”的独立栏或行 | 不能覆盖 R@10；mean-S 单次阈值结果不能套用原三模型 SD |
| Methods 与补充材料 | 写明孤立源背景、4005 对、mean-S 查询、validation 复用、分组敏感性与尾部限制 | 年度 FAR 未建立，不再把日历跨度当有效背景时间 |
| 图2／图3／图5、完整450事件检索扩展图 | 原始分数、Recall、排名及 PE 未变，无须因本轮全部重绘 | 仅涉及旧 FAR 的文字／标注需要同步删除或替换 |

现有论文工作树 `0909结果/论文Methods更新_20260911/overleaf_worktree/main.tex` 的图4、O4b结果段和背景方法段仍有旧年化描述。它们应整体撤换，而不是仅将轴标签 FAR 改成 FPP。本次只拉取、核验并提出方案，未修改或推送论文。

服务端概览图可用于审阅，但不建议原样当正式图4：标题包含实验代号和长说明，正文沿用的图4需要按现有版式重绘；上排概览是“FPP阈值—保留率”，并不是此前要求的“分数—背景比例”。图中零计数点画在 1/4005 处只是显示约定，应在图注明确不是测量值或上限。应统一图／表的严格小于或小于等于阈值约定。

## 7. 入文前需要说清的限制

1. 4,005 对共享 90 个源和 6 个噪声块，不是 4,005 次独立试验。删除一个源／噪声块的范围是敏感性分析，不是 95% 置信区间。
2. Top-10 尾部计数全部少于 20；O4a 六对、O4b 一对为零超越。1/4005 ≈ 0.02497% 只是计数网格间距，不是可信上界。两个未分辨候选不能仅凭零值比较谁更可信。
3. background 与 test 在源／噪声上隔离，但背景来自用于选模型与系数的 validation，不能保证条件 FPP 的名义覆盖率，也不是新的严格独立校准。
4. 真实候选使用公开 PE 天图，注入使用恢复触发量的 HL BAYESTAR；总体、事件选择及探测器网络差异仍可能影响尾部映射。因此真实表是探索性背景对照，不是已验证的真实误报概率。
5. 无有效背景搜索时间，本轮年度 FAR 明确 NO-GO。FAR 按单位搜索时间的背景超越事件期望定义，而非将配对比例直接除以观测日历；见 [IGWN 官方说明](https://emfollow.docs.ligo.org/userguide/analysis/searches.html#false-alarm-rate-and-significance)。这里的透镜配对任务还需定义自身的搜索与背景曝光。

本轮可以补强“候选筛选能力及背景负担”的证据，不能升级为“识别出真实透镜”或“已排除所有候选”。

## 8. 交付与校验

服务器压缩包：`/root/autodl-tmp/gw-catalog/results/gwlr_fpp_injection_real_01_20260922T061000Z_r2_deliverables.tar.gz`。

本地下载目录：`0922_FPP结果`。原始包约 146.84 MB（140.04 MiB），远端实际哈希、随包哈希与本地 SHA-256 一致：

`6e57bda810b8744aca1b0c694365e1549cc89cec53e65e77f20073600819c907`

179 个清单文件逐一校验通过；压缩包共 180 个文件，包含 SHA 清单本身。安全检查未发现路径穿越或链接成员。16 项单元测试本地重跑通过；独立重算 24 组运行期／模型／域的全部条件 FPP，与表中超越计数一致。真实分数、贡献、PE、排名与本地 C 归档逐项核对通过。

服务器报告称 180 个受保护历史输入哈希未变；本地没有重新读取全部远端历史输入，只验证该报告的包内完整性并核对本地可用 C 结果，不将其描述成重新执行了全部远端输入审计。

本次按照 Nature Statistics 的检查口径，区分了 pair recall 与 R@10、经验尾部比例与后验概率、敏感性范围与置信区间，并明确模型平均顺序和有效样本限制。上述区分未改变原数据。

注意：本目录生成于既有 Windows 迁移包之后，不包含在此前迁移压缩包中，迁移时需要另带本目录。
''')
rel=R.relative_to(B).as_posix()
parts.extend(['### 阅读与复算入口',f'- [服务端完整中文报告]({rel}/reports/FINAL_FPP_AND_FAR_AUDIT_CN.md)',f'- [服务端概览图]({rel}/figures/conditional_FPP_overview.png)',f'- [全部阈值效率]({rel}/summaries/injection_efficiency_at_FPP.csv)',f'- [背景冻结定义]({rel}/contracts/BACKGROUND_FREEZE.json)',f'- [本地独立核验记录](audit/LOCAL_NUMERICAL_AUDIT.json)'])
for run in runs:parts.append(f'- [{run} 完整真实候选 FPP 表]({rel}/tables/{run}/real_GWTC_mean_S_FPP.csv)')
content='\n\n'.join(parts)+'\n'
assert '$' not in content and '\\frac' not in content
audit={'all_checks_passed':True,'groups':audits,'all_rows_recounted_total':sum(a['all_rows_recounted'] for a in audits),'real_ranks_scores_contributions_PE_match_local_C':True,'mean_score_operating_point_recounted':eff,'background_test_source_noise_disjoint':True,'unit_tests_rerun':16,'sources':[],'paper_modified':False}
for path in sourcefiles:
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    audit['sources'].append({'path':str(path),'sha256':h})
patch='*** Begin Patch\n'
for name,text in [('0922精炼版_FPP_Top20_Recall与入文建议.md',content),('audit/LOCAL_NUMERICAL_AUDIT.json',json.dumps(audit,ensure_ascii=False,indent=2)+'\n')]:
    assert not (B/name).exists()
    patch+='*** Add File: '+str(B/name)+'\n'+''.join('+'+line+'\n' for line in text.splitlines())
patch+='*** End Patch'
print(json.dumps({'patch':patch,'checks':len(audits),'recounted_rows':audit['all_rows_recounted_total']},ensure_ascii=False))
