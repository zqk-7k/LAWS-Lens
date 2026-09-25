#!/usr/bin/env python3
"""Build an auditable development report; never promote a scientific version."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mcwf_development_20260905 as dev


def fmt(x):return f"{x:.4f}" if pd.notna(x) else "NA"


def summarize_mass(root):
    rows=[];events=[]
    old=pd.read_csv(root/"audit/O3_BASELINE_EVENT_MASS_INPUT_AUDIT.csv")
    for seed,p in old.groupby("model_seed"):
        err=p.abs_logmc_error_vs_PE.to_numpy()
        rows.append({"model":"old_aux_head","deployment":"gwtc3","seed":seed,"n":len(p),"median_abs_logMc_error":np.median(err),"q90_abs_logMc_error":np.quantile(err,.9),"factor2_error_count":int((err>np.log(2)).sum()),"factor5_error_count":int((err>np.log(5)).sum())})
    pe=pd.read_csv(root/"audit/gwtc3_event_PE_reference.csv")
    for name in ("MASSTF","PHASEBANK","PHASEPSD-MASSLR"):
        for path in (root/f"results/{name}/gwtc3").glob("model_*_eval_*/real_event_mass_predictions.csv"):
            seed=int(path.parent.name.split("_")[1]);e=pd.read_csv(path).merge(pe,on="event_name")
            e=e[e.strict_h1l1_preprocessing_pass]
            err=abs(np.log(e.pred_mc/e.pe_chirp_mass_median)).to_numpy()
            rows.append({"model":name,"deployment":"gwtc3","seed":seed,"n":len(e),"median_abs_logMc_error":np.median(err),"q90_abs_logMc_error":np.quantile(err,.9),"factor2_error_count":int((err>np.log(2)).sum()),"factor5_error_count":int((err>np.log(5)).sum())})
            e["model"]=name;e["model_seed"]=seed;events.append(e)
    dev.csv_write(root/"tables/MASS_PREDICTION_DEVELOPMENT_AUDIT.csv",pd.DataFrame(rows))
    if events:dev.csv_write(root/"tables/MASS_PREDICTION_EVENT_AUDIT.csv",pd.concat(events,ignore_index=True))
    return pd.DataFrame(rows)


def checks(root):
    protected=pd.read_csv(root/"manifest/PROTECTED_INPUT_SHA256.csv")
    check=[]
    for r in protected.itertuples():
        now=dev.sha(Path(r.path));check.append({"path":r.path,"before":r.sha256,"after":now,"unchanged":now==r.sha256})
    dev.csv_write(root/"manifest/PROTECTED_INPUT_HASH_BEFORE_AFTER.csv",pd.DataFrame(check))
    if not all(r["unchanged"] for r in check):raise RuntimeError("Historical input changed; do not deliver success")
    real=[]
    for f in (root/"results").glob("*/gwtc*/seed_*_C_fixed.parquet"):
        dep=f.parent.name;seed=int(f.stem.split("_")[1]);current=pd.read_parquet(f).sort_values("pair_key").reset_index(drop=True)
        base=dev.real_frame(dep,seed).sort_values("pair_key").reset_index(drop=True)
        equal=all(np.array_equal(current[c],base[c]) for c in ("pair_key","time_score","sky_raw_log_bf"))
        real.append({"path":str(f),"same_scope_time_sky":equal})
        if not equal:raise RuntimeError(f"Real frozen channel mismatch: {f}")
    dev.csv_write(root/"audit/REAL_TIME_SKY_SCOPE_INVARIANCE.csv",pd.DataFrame(real))
    # Freeze a read-only diagnostic of the failed hard-mask comparison.
    dev.json_write(root/"audit/NUMERICAL_AND_OPERATIONAL_NOTES.json",{
        "hard_mask_matched_filter":"invalid as an SNR comparison: truncating raw colored strain created spectral leakage; full32 PE-conditioned reference comparison is audit only; neither is a ranking feature",
        "phase_bank_build_retry":"PyCBC TD convenience wrapper failed at an extreme high-mass template with zero taper length; used direct IMRPhenomD FD generation before any model training; original failure log retained",
        "resume_reporting_retry":"completed mass-replacement task skip produced empty temporary aggregate and pandas query error; per-seed pair outputs unchanged; skip reporter fixed and consolidated tables regenerated",
        "frozen_test_ties":"historical retrieval assigns rank1+strictly-higher count; ties are optimistic. Evaluation definition preserved, not evidence of extra distinct retrieval power",
        "R2_vs_R3_guardrails":"R2 selected using R10/AP guards then F50/F90 tie priorities; R3 additionally requires validation F50/F90 limits for both waveform and frozen fusion; R2 failures retained",
        "official_O4":"prior local audit placeholders were not reliable coverage statements. New tables use public PO/Phazap,Fast-GOLUM,Hanabi source data; no new Hanabi run"})
    files=[str(p) for p in (root/"scripts").glob("*.py")]
    subprocess.run([sys.executable,"-m","py_compile"]+files,check=True)
    return {"historical_files_checked":len(check),"historical_changed":0,"real_pair_tables_checked":len(real),"all_real_time_sky_scope_identical":True}


def make_figures(root,metrics,budgets):
    fonts={x.name for x in font_manager.fontManager.ttflist}
    font="Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    plt.rcParams.update({"font.family":font,"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":.7})
    selected=["CFIX-baseline","PHASEBANK-BOUNDED-PHYS"]
    colors={selected[0]:"#526677",selected[1]:"#AA3B52"}
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.0))
    for ax,metric,label in ((axes[0,0],"macro_r_at_10","Companion R@10"),(axes[0,1],"average_precision","Pair AUPRC")):
        for di,dep in enumerate(("gwtc3","gwtc4")):
            for ci,config in enumerate(selected):
                d=metrics[(metrics.deployment==dep)&(metrics.config==config)&(metrics.method=="C_fixed")]
                x=di+(ci-.5)*.2;y=d[metric].to_numpy()
                ax.scatter(x+np.linspace(-.025,.025,len(y)),y,s=17,color=colors[config],alpha=.8,label=config if di==0 else None)
                ax.errorbar(x,y.mean(),yerr=y.std(ddof=1),fmt="_",color=colors[config],capsize=3)
        ax.set_xticks([0,1],["O3","O4a"]);ax.set_ylabel(label)
    for ax,metric,label in ((axes[1,0],"catastrophic_mc","Top-10 catastrophic Mc pairs"),(axes[1,1],"official_frontend","Top-10 official frontend overlap")):
        for di,dep in enumerate(("gwtc3","gwtc4")):
            for ci,config in enumerate(selected):
                name=config if config=="CFIX-baseline" else config+"-three-seed"
                d=budgets[(budgets.deployment==dep)&(budgets.config==name)&(budgets.method=="C_fixed")&(budgets.budget==10)]
                ax.bar(di+(ci-.5)*.26,float(d.iloc[0][metric]),width=.24,color=colors[config])
        ax.set_xticks([0,1],["O3","O4a"]);ax.set_ylabel(label);ax.set_ylim(0,10.5);ax.set_yticks([0,2,4,6,8,10])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,["C-fixed baseline","Bounded mass guard (development)"],loc="upper center",bbox_to_anchor=(.5,1.0),ncol=2,frameon=False)
    for ax,tag in zip(axes.flat,"abcd"):ax.text(-.17,1.04,tag,transform=ax.transAxes,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/"figures/fig_mcwf_development_overview.pdf",bbox_inches="tight");fig.savefig(root/"figures/fig_mcwf_development_overview.png",dpi=220,bbox_inches="tight");plt.close(fig)
    old=pd.read_csv(root/"audit/O3_BASELINE_EVENT_MASS_INPUT_AUDIT.csv")
    new=pd.read_csv(root/"tables/MASS_PREDICTION_EVENT_AUDIT.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    for ax,name in zip(axes,("Old auxiliary head","Phase-sensitive mass head")):
        if name.startswith("Old"):
            x=old.pe_chirp_mass_median;y=old.waveform_pred_chirp_mass_detector
        else:
            d=new[new.model.eq("PHASEBANK")];x=d.pe_chirp_mass_median;y=d.pred_mc
        ax.scatter(x,y,s=9,alpha=.55,color="#4B7185");ax.plot([5,200],[5,200],"--",color="black",lw=.8)
        ax.set(xscale="log",yscale="log",xlim=(5,200),ylim=(5,200),xlabel="Public PE median detector-frame Mc",ylabel="Waveform-derived Mc",title=name)
    fig.tight_layout();fig.savefig(root/"figures/fig_mc_prediction_audit.pdf",bbox_inches="tight");fig.savefig(root/"figures/fig_mc_prediction_audit.png",dpi=220,bbox_inches="tight");plt.close(fig)
    dev.json_write(root/"figures/FIGURE_METADATA.json",{"font":font,"seed_points":True,"mass_plot_role":"real O3 development audit, not unbiased model validation"})


def write_reports(root):
    metrics=pd.read_csv(root/"tables/ALL_INJECTION_METRICS_PER_SEED.csv")
    budgets=pd.read_csv(root/"tables/ALL_REAL_PE_OFFICIAL_BUDGETS.csv")
    mass=summarize_mass(root)
    selected=["CFIX-baseline","PHASEBANK-BOUNDED-SAFE","PHASEBANK-BOUNDED-PHYS"]
    table=[]
    for dep in ("gwtc3","gwtc4"):
        for config in selected:
            for method in ("waveform_only","C_fixed"):
                f=metrics[(metrics.deployment==dep)&(metrics.config==config)&(metrics.method==method)]
                rec={"run":dep,"config":config,"method":method}
                for col in ("macro_r_at_1","macro_r_at_10","average_precision","false_at_recall_0p5","false_at_recall_0p9"):
                    rec[col]=f"{f[col].mean():.4f} +/- {f[col].std(ddof=1):.4f}"
                table.append(rec)
    table=pd.DataFrame(table);dev.csv_write(root/"tables/PRIMARY_COMPARISON_MEAN_SD.csv",table)
    ledger=metrics.groupby(["config","deployment","method"]).agg(
        n_seeds=("model_seed","nunique"),R1=("macro_r_at_1","mean"),R10=("macro_r_at_10","mean"),
        AUPRC=("average_precision","mean"),F50=("false_at_recall_0p5","mean"),F90=("false_at_recall_0p9","mean"),
        guardrail_pass_seeds=("development_guardrail_pass","sum")).reset_index()
    b=budgets[budgets.budget.eq(10)].copy();b["config"]=b.config.str.replace("-three-seed$","",regex=True)
    ledger=ledger.merge(b[["config","deployment","method","catastrophic_mc","BC_mc_ge_0p5","Dmax_le_3","official_frontend","official_hanabi"]],on=["config","deployment","method"],how="left")
    ledger["stage_status"]=np.where(ledger.guardrail_pass_seeds.eq(ledger.n_seeds),"DEVELOPMENT_GUARDRAILS_ALL_SEEDS_PASS_NOT_CONFIRMATION","DEVELOPMENT_GUARDRAIL_FAILURE_RETAINED")
    dev.csv_write(root/"tables/EXPERIMENT_LEDGER_RECALL_PE_OFFICIAL.csv",ledger)
    bd=budgets[budgets.config.isin(["CFIX-baseline","PHASEBANK-BOUNDED-SAFE-three-seed","PHASEBANK-BOUNDED-PHYS-three-seed"]) & budgets.budget.eq(10)]
    basepath=root/"final_tables/gwtc3/CFIX-baseline/C_fixed/all_pairs.parquet"
    newpath=root/"final_tables/gwtc3/PHASEBANK-BOUNDED-PHYS-three-seed/C_fixed/all_pairs.parquet"
    base=pd.read_parquet(basepath);new=pd.read_parquet(newpath)
    change=base[["pair_key","consensus_rank","waveform_score_mean","final_score_mean","pe_mc_bhattacharyya_coefficient","pe_mc_standardized_distance"]].merge(
        new[["pair_key","consensus_rank","waveform_score_mean","final_score_mean"]],on="pair_key",suffixes=("_old","_new"))
    dev.csv_write(root/"tables/O3_CANDIDATE_RANK_CHANGE_ALL_PAIRS.csv",change)
    old_bad=base.head(10).loc[(base.head(10).pe_mc_bhattacharyya_coefficient<.1)|(base.head(10).pe_mc_standardized_distance>5),"pair_key"]
    dev.csv_write(root/"tables/O3_BASELINE_TOP10_CATASTROPHES_TRACKED.csv",change[change.pair_key.isin(old_bad)])
    topcols=["consensus_rank","event_i","event_j","final_score_mean","waveform_contribution_mean","time_contribution_mean","sky_contribution_mean","pe_mc_bhattacharyya_coefficient","pe_mc_standardized_distance","pe_dmax_intrinsic","official_po_fpp","official_ml_fpp","official_screening_stage"]
    topcols=[c for c in topcols if c in new]
    report=fr'''# MAIN-O3-MCWF-DEV-v1：波形与 Mc 一致性开发实验

状态：**{dev.FINAL_STATUS}**。本轮没有覆盖旧结果，没有修改论文、时间、天空或 C-fixed 融合权重。这里报告开发研究，不声称发现或确认透镜。

## 1. 结论与边界

旧 head 存在可复现的质量估计灾难性偏差，不是把 PE overlap 换成另一个公式就能解决的问题。采用峰值 2 s/4096 点的物理相位滤波特征后，多个原来被估成数十至百余太阳质量的低 Mc 事件得到更合理估计。仅改善质量预测并不自动改善整个候选排序。

本轮保留所有失败消融。有的无上限校准虽然使 O3 Top-10 的 PE 更好，却增加注入 F90；不能只凭好看的 PE 表升级。相对稳妥的开发候选为 **PHASEBANK-BOUNDED-PHYS**：通过模拟 validation 的全部 guardrails 后，以模拟真值质量一致性选取有上限的负向波形修正。它不是新正式主方法。

O3 Top-10 仍可能保留 Mc 冲突；O4a 的个别 seed 仍有假对负担波动。不能声称所有高分 pair 均通过 PE，不能声称三个 seed 一致改进，更不能把官方重合当成真透镜标签。需结合下列逐 seed 和 bootstrap 表评价。保留全部失败结果，未自动替换任何基线。

具体地，PHASEBANK-BOUNDED-PHYS 的 O3 波形单通道 Top-10 灾难性 Mc 对从 1 降到 0；三通道从 3 降到 1，Dmax<=3 从 8/10 到 9/10，但 BC_Mc>=0.5 仍为 4/10。官方前端重合从 4 到 5，公开 Hanabi 表重合从 4 到 3，并非所有官方指标同时改善。残留冲突是 GW191113_071753--GW200316_215756：BC_Mc 约 0.0072，不能称为 PE 已全部吻合。

O3 三通道 R@10 从 0.8655 到 0.8630，AUPRC 从 0.3196 到 0.3274；O4a R@10 从 0.7765 到 0.7645，AUPRC 从 0.1919 到 0.1885，F90 均值从 1435 到 1608。O4a 虽仍是 Top-10 全部 Dmax<=3，但注入 guardrail 并非全部 seed 通过。因此按开发合同的停止条件，本轮交付部分改善和明确失败，**不升级为统一正式方法**。不是保证消除所有真实 Mc 冲突后才展示结果。

## 2. 冻结基线与范围

- MAIN-O3OFFICIAL-CFIX-v1.2（科学计算 v1），不是旧累积 O1–O3 目录。
- 官方 O3：70 个事件；严格双探测器有效输入为 **62 个、1891 对**，不是最初预计的 64。
- O4a：当前冻结的 **74 个、2701 对**。官方 PO/Phazap 对这 2701 对全部可联表。
- 天空仍为 ordering 修复、Nside=512、注入 BAYESTAR；一维时间查找表不变。
- baseline 注入的源、噪声、天空、pair 顺序全部不变。本轮不重新做天空 PE，不运行 Hanabi。
- 202607241/242/243 为原部署；202609051/052/053 为新质量 head 初始化。二者对应比较，不假装重新训练了原 embedding。

## 3. 做了哪些实验

1. 审计原始应变、数据门控、质量 head、校准、事件级预测和全部 PE。PE 条件化 matched-filter 只作数据诊断，不进入网络或排名。原始 colored strain 硬截窗的诊断出现边缘伪影，已明确作废，不引用其中异常大数值。
2. 在新的官方 O3 62 事件范围上复用旧的 2 s、2+8 s、2+16 s、hard-negative、不确定度 head、embedding 处理模型。旧 G7 失败没有改判。单纯这些修改没有可靠解决问题。
3. MASS-TF：用峰值 2 s 的两个 STFT 分辨率预测 64 个 logMc bins。三个 seed 的汇总没有稳定减少三通道 Top-10 冲突，因此未采纳。
4. PHASEBANK：使用 PyCBC IMRPhenomD 建立 64 Mc ×3 q ×3 自旋的双相位滤波特征，再用模拟真实质量监督小型 head；只用 validation 校准温度。它是物理辅助的 waveform-derived 预测分布，不是完整 PE。
5. 比较 mass-only、负向 guard、非补偿 cap、旧 Mc head 替换、按模拟 companion/null 拟合的质量差 LR。只输出单个预测值或未经限制地处罚存在明显局限。
6. BOUNDED-SAFE 与 BOUNDED-PHYS：同一新质量特征，用模拟 validation 限制修正幅度；后者的优先目标使用**模拟注入真实质量**，不是公开 PE。
7. 补充 PSD-conditioned 三 seed 对照：把每个事件实际 whitening 使用的噪声 PSD 形状提供给质量 head，用于检查噪声条件是否解释残余偏差。所有结果单独列入总台账，不与前述结果混名，不因官方重合较好就自动挑选它。

## 4. 公式与选参

原三通道：\(S=w_W Z_W+w_T Z_T+w_S Z_S\)。始终保持 \(Z_T,Z_S,w_W,w_T,w_S\) 不变。

新 head 给出 \(p_b\)（64-bin 模拟条件预测分布）。质量预测为 \(\widehat{{\log M_c}}=\sum_b p_b\log M_{{c,b}}\)。

质量差特征 \(u=-|\widehat{{\log M_{{c,i}}}}-\widehat{{\log M_{{c,j}}}}|\)，在模拟 validation companion 与 null 上拟合 \(L_M=\log p(u|L)-\log p(u|N)\)。这不是公开 PE 证据。

有界修正：\(Z_W^{{new}}=Z_W^{{old}}+\gamma\,\mathrm{{clip}}(L_M,-c,0)\)。只限制原波形分数，不增加第四个天空或 PE 通道，也不再给弱质量信息无限正奖励。

网格 \(\gamma\in\{{0,.25,.5,1,2,4\}}\)，\(c\in\{{.25,.5,1,2,4\}}\)。每 seed 独立选择，允许 \(\gamma=0\) 回退。

validation 中，波形单独和冻结三通道都要求 R@10 下降不超过 .02、AUPRC 不超过 .005、F50/F90 不超过原值的 1.1 倍。这些是预先写入本轮合同的**开发容差**，不是物理定律。SAFE 再按 F50/F90 等选择；PHYS 先最小化模拟 Top-10/50 的平均真实 logMc 差。全部网格、可行配置和退回零修正均保留。

## 5. 注入结果：三 seed mean +/- SD

{table.to_markdown(index=False)}

全部配置：`tables/ALL_INJECTION_METRICS_PER_SEED.csv`、`ALL_INJECTION_METRICS_SUMMARY.csv`。包含 waveform-only、固定三通道、AP、F50/F90 和各预算真假对。它们是反复使用过的历史测试目录，正式命名为 reused development comparator，不包装成新 locked test。

**每个实验的 recall、PE 和官方阶段在同一张表中：** `tables/EXPERIMENT_LEDGER_RECALL_PE_OFFICIAL.csv`。`n_seeds=1` 的旧模型复用不冒充三 seed 稳定性；缺少运行期的实验不填假数据。

## 6. 真实 Top-10 PE 与官方阶段

{bd[['config','deployment','method','catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3','official_frontend','official_hanabi']].to_markdown(index=False)}

Mc 灾难性标准固定为 BC<0.1 **或** D>5；Dmax 只包含 Mc、q、chi_eff，不把表观距离误当成透镜不变量。BC、D 是描述性边际一致性，不是 lensing probability。

O3 开发候选完整 Top-10：

{new.head(10)[topcols].round(4).to_markdown(index=False)}

`final_tables/<run>/<config>/<method>/all_pairs.parquet` 与 `top100.csv` 保存全部排序和贡献。`ALL_REAL_PE_OFFICIAL_BUDGETS.csv` 同时保存 Top-10/20/50/100，不能只看前十。

原例 GW190924_021846 不在当前官方 O3 70 事件范围，因此它与 GW191105 的旧 pair 不能混回新的 62 事件榜单。当前基线头部冲突另由 `O3_BASELINE_TOP10_CATASTROPHES_TRACKED.csv` 逐对追踪。

## 7. 数据隔离和不确定度

网络监督只来自模拟质量；train/validation source-parent 不重合，旧 development noise partition 报告 train/validation 与历史噪声父事件不重合。源、noise manifests 和配置保留。ET-3 未触碰。

最后从原始 compact injection metadata 逐事件核对 source_index、GW-LMC row 与各像 noise parent：`audit/SOURCE_TRUTH_AND_NOISE_MAPPING_SUMMARY.csv`，覆盖两运行期、三个 seed、validation/test。新 head 的模拟 validation 中央 50%/90% 预测覆盖及 effective rank 保存在 `tables/PREDICTIVE_VALIDATION_COVERAGE_AND_RANK.csv`。同一 validation 已参与温度选择，因此这不是独立校准检验，更不是真实 GWTC full-PE 覆盖证明。

`tables/MASS_LR_SUPPORT_AUDIT.csv` 报告所有 lookup 支持域外 pair。原一维 helper 在边界使用常数插值，本轮没有将它伪称为物理外推；bounded 方案只允许有限负向影响。PHASEBANK 真实 O3 最大约 0.69%、O4a 约 0.22% 的 pair 在 lookup 范围外；PSD-conditioned O3 最大约 1.27%。这些是分布转移风险，不能忽略。

训练 seed SD 与系统抽样 CI 分开：每个源的两条 directed queries 一起抽样，SIS/PM legacy slots 分层，10,000 次 bootstrap；pair 指标采用 2,000 次 source-block 加权成对 bootstrap。此类 CI 反映固定模型/观测目录条件下的不确定度，不是多人口确认，也不是 iid pair 的显著性。三个原部署对应的注入 realization/目录也有差异，因此表中 SD 包含 head 初始化、原 encoder 与评估目录变化，不能称为纯初始化方差。

公开 O3 PE 的反馈已用于确定研究方向，所以 O3 明确是 development audit。真实 PE、官方标签不输入模型，但不能因此声称全过程没有选择偏差。新鲜独立注入和未消费的真实盲样本验证仍是未来正式采纳的条件。

## 8. 官方文件不是空值臆测

O3 使用完整 2415 对的官方 PO/ML 文件及已有 full-O3/O3a follow-up 映射。O4a 新下载 LVK release 的 PO、Phazap、Fast-GOLUM 及三种人口模型 Hanabi 表，并以 GWOSC 官方 gracedb_id 对照事件名。O4a 2701 个 scope pairs 的 PO/Phazap 均有数值；80 对列入公开 Fast-GOLUM 表，35 对列入公开 Hanabi 表。这些不是本项目新跑的 Hanabi。

官方前端阈值按原 Figure2 的 FPP<.01；不是透镜确认阈值。官方文章未发现稳健强透镜证据，不能把重合数称为真阳性。

## 9. 参考与局限

- PyCBC matched filtering/波形库：[官方文档](https://pycbc.org/pycbc/latest/html/pycbc.filter.html)。本项目的稀疏特征 bank 没有完整搜索 bank 的 minimal-match 保证，不宣称等价 PE。
- 噪声 PSD 条件化神经推断依据：[DINGO](https://arxiv.org/abs/2106.12594)。本实验不是原文 DINGO 复现。
- O3 官方 lensing search：[LVK](https://arxiv.org/abs/2304.08393)。
- O4a 官方 lensing search：[LVK](https://arxiv.org/abs/2512.16347)，[机器可读发布](https://zenodo.org/records/18163632)。

当前能够支持的是：旧质量 head 的若干明显偏差可通过相位敏感波形表征减轻；限制质量修正的影响比无上限处罚更稳妥。不能支持“所有高分候选都必然 PE 相容”“官方重合证明透镜”“已经得到最终无偏确认”。

最终保持 `{dev.FINAL_STATUS}`。所有失败消融、旧结果和未解决的问题均保留。
'''
    (root/"reports/FINAL_WAVEFORM_EXPERIMENT_REPORT_CN.md").write_text(report,encoding="utf-8")
    (root/"README_CN.md").write_text("# MAIN-O3-MCWF-DEV-v1\n\n独立开发研究，不替换 C-fixed。先读 reports/FINAL_WAVEFORM_EXPERIMENT_REPORT_CN.md。\n\n- 全配置注入指标：tables/ALL_INJECTION_METRICS_PER_SEED.csv\n- PE/官方预算：tables/ALL_REAL_PE_OFFICIAL_BUDGETS.csv\n- 全排名/Top100：final_tables/\n- 冻结规则：contracts/\n- 可复现脚本：scripts/；在原服务器项目环境运行，原始 strain 和 development bank 不随紧凑包分发。\n\n"+dev.FINAL_STATUS+"\n",encoding="utf-8")
    make_figures(root,metrics,budgets)


def snapshot(root):
    scripts=dev.PROJECT/"scripts/experiments"
    for p in scripts.glob("mcwf*_20260905.py"):
        shutil.copy2(p,root/"scripts"/p.name)
    dependencies={"old_orchestrator":dev.ORCH.__file__,"old_training":dev.TRAIN.__file__,"frozen_baseline":dev.BASE.__file__,
        "main_official_scope":dev.MAINCODE.__file__,"v7_common":dev.BASE.v7.__file__,"v3_physical":dev.BASE.v7.v3.__file__,"physical_helpers":dev.ORCH.PHYS.__file__}
    dest=root/"scripts/reference_dependencies";dest.mkdir(exist_ok=True)
    for label,path in dependencies.items():shutil.copy2(path,dest/f"{label}.py")
    dev.json_write(dest/"PATH_MAPPING.json",dependencies)
    for dep in ("gwtc3","gwtc4"):
        dest=root/f"source_data/development_provenance/{dep}";dest.mkdir(parents=True,exist_ok=True)
        for p in (dev.OLD/f"data/noise_banks/{dep}").glob("*.csv"):shutil.copy2(p,dest/p.name)
        for p in (dev.OLD/f"data/noise_banks/{dep}").glob("*.json"):shutil.copy2(p,dest/p.name)
        for p in (dev.OLD/f"data/development/{dep}").glob("*metadata.parquet"):shutil.copy2(p,dest/p.name)
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True)
    (root/"contracts/ENVIRONMENT_PIP_FREEZE.txt").write_text(freeze.stdout)
    dev.json_write(root/"contracts/ENVIRONMENT_RUNTIME.json",{"python":sys.version,"platform":sys.platform,"torch":dev.torch.__version__,"cuda":dev.torch.version.cuda,"device":dev.torch.cuda.get_device_name(0),"threads":dev.torch.get_num_threads(),"disk":shutil.disk_usage(root)._asdict()})


def package(root):
    excluded={"cache","__pycache__"}
    paths=[p for p in root.rglob("*") if p.is_file() and not any(x in excluded for x in p.relative_to(root).parts)
           and p.name not in ("OUTPUT_SHA256.csv","SHA256SUMS.txt")]
    secrets=(b"-----BEGIN RSA "+b"PRIVATE KEY-----",b"-----BEGIN OPENSSH "+b"PRIVATE KEY-----",b"sshpass "+b"-p")
    for p in paths:
        if p.suffix in (".py",".md",".json",".txt",".log",".csv"):
            content=p.read_bytes()
            if any(s in content for s in secrets):raise RuntimeError(f"Credential marker in package input {p}")
    manifest=pd.DataFrame([{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":dev.sha(p)} for p in sorted(paths)])
    dev.csv_write(root/"manifest/OUTPUT_SHA256.csv",manifest)
    (root/"manifest/SHA256SUMS.txt").write_text("".join(f"{r.sha256}  {r.path}\n" for r in manifest.itertuples()))
    target=dev.PROJECT/"packages"/f"{root.name}_deliverables.tar.gz"
    if target.exists():raise RuntimeError("Package exists; do not overwrite")
    with tarfile.open(target,"w:gz",compresslevel=6) as tfp:
        for p in sorted(paths+[root/"manifest/OUTPUT_SHA256.csv",root/"manifest/SHA256SUMS.txt"]):
            tfp.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    digest=dev.sha(target);target.with_suffix(target.suffix+".sha256").write_text(f"{digest}  {target.name}\n")
    with tarfile.open(target,"r:gz") as tfp:
        members={m.name:m for m in tfp.getmembers()}
        for r in manifest.itertuples():
            obj=tfp.extractfile(members[str(Path(root.name)/r.path)])
            h=__import__("hashlib").sha256()
            for chunk in iter(lambda:obj.read(8<<20),b""):h.update(chunk)
            if h.hexdigest()!=r.sha256:raise RuntimeError(f"Archive verification failed {r.path}")
    print(json.dumps({"package":str(target),"sha256":digest,"files_verified":len(manifest),"bytes":target.stat().st_size}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--package",action="store_true");a=p.parse_args()
    snapshot(a.root);write_reports(a.root);check=checks(a.root)
    dev.json_write(a.root/"contracts/DELIVERY_AUDIT.json",{**check,"status":dev.FINAL_STATUS,"timestamp":datetime.now(timezone.utc).isoformat()})
    if a.package:package(a.root)
