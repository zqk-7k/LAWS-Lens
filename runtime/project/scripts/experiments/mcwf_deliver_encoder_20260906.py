#!/usr/bin/env python3
"""Collect all experiments and the frozen run-specific candidate, transparently."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_fresh_confirmation_20260906 as fresh

FINAL_CODE = "MAIN-O3-MCWF-ENCODER-v2-RUNSPEC"


def tab(frame, columns=None):
    if columns:
        frame = frame[columns]
    return frame.to_markdown(index=False, floatfmt=".4f")


def compact_metric_table(frame, fresh_sample=False):
    rows = []
    for r in frame.to_dict("records"):
        roles = (("baseline", "baseline_"), ("candidate", "")) if fresh_sample else ((r["final_version"], ""),)
        for role, prefix in roles:
            out = {"run": r["deployment"], "version": role, "method": r["method"]}
            for name, col in (("R1", "macro_r_at_1"), ("R10", "macro_r_at_10"), ("AUPRC", "average_precision"),
                              ("F50", "false_at_recall_0p5"), ("F90", "false_at_recall_0p9")):
                out[name] = f"{r[prefix+col+'_mean']:.4f} +/- {r[prefix+col+'_std']:.4f}"
            rows.append(out)
    return tab(pd.DataFrame(rows))


def collect(root):
    chosen = fresh.verify_freeze(root)
    config = chosen["O3_config"]
    budgets, real, correlations, known = [], [], [], []
    for dep in ("gwtc3", "gwtc4"):
        pe = dev.pe_audit(root, dep)
        baseline = {es: dev.real_frame(dep, es) for es in dev.SEEDS}
        budgets.extend(dev.save_evaluation(root, "CFIX-baseline", dep, baseline, pe))
        candidate = baseline if dep == "gwtc4" else {es: pd.read_parquet(root / f"evaluation/{config}/{dep}/model_{ms}_eval_{es}/real_VAL-ENSEMBLE_pairs.parquet") for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS)}
        for es in dev.SEEDS:
            for column in ("pair_key", "time_score", "sky_raw_log_bf"):
                if not np.array_equal(candidate[es][column].to_numpy(), baseline[es][column].to_numpy()):
                    raise RuntimeError("Final real non-waveform channel changed")
        budgets.extend(dev.save_evaluation(root, FINAL_CODE, dep, candidate, pe))
        for code in ("CFIX-baseline", FINAL_CODE):
            for method in ("waveform_only", "C_fixed"):
                frame = pd.read_parquet(root / f"results/{code}/{dep}/consensus_{method}_all_pairs_pe_official.parquet")
                distance_supplement = root / "audit/gwtc4_additional_distance_pe.parquet"
                if dep == "gwtc4" and distance_supplement.exists():
                    frame = frame.merge(pd.read_parquet(distance_supplement), on="pair_key", validate="one_to_one")
                out = root / f"final_results/{dep}/{code}/{method}"
                out.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(out / "all_pairs.parquet", index=False)
                dev.csv_write(out / "all_pairs.csv", frame)
                dev.csv_write(out / "top100.csv", frame.head(100))
                real.append(frame.head(10).assign(deployment=dep, config=code, ranking=method))
                for col, sign in (("pe_mc_bhattacharyya_coefficient", 1), ("pe_mc_standardized_distance", -1)):
                    correlations.append({"deployment": dep, "config": code, "method": method, "PE_metric": col,
                        "spearman_descriptive_only": spearmanr(frame.waveform_score_mean, sign*frame[col], nan_policy="omit").statistic})
                if dep == "gwtc3":
                    for a, b in (("GW190924_021846", "GW191105_143521"), ("GW190412", "GW191204_171526"),
                                 ("GW190924_021846", "GW190930_133541")):
                        hit = frame.loc[((frame.event_i == a) & (frame.event_j == b)) | ((frame.event_i == b) & (frame.event_j == a))]
                        if hit.empty:
                            known.append({"config": code, "method": method, "event_i": a, "event_j": b,
                                          "audit_status": "outside_frozen_official_O3_scope_not_scored"})
                        for record in hit.to_dict("records"):
                            known.append({"config": code, "method": method, "audit_status": "in_scope", **record})
    budget = pd.DataFrame(budgets)
    dev.csv_write(root / "tables/FINAL_BASELINE_PE_OFFICIAL_BUDGETS.csv", budget)
    dev.csv_write(root / "tables/FINAL_TOP10_ALL.csv", pd.concat(real, ignore_index=True))
    dev.csv_write(root / "tables/FINAL_KNOWN_FAILURE_PAIR_AUDIT.csv", pd.DataFrame(known))
    dev.csv_write(root / "tables/FINAL_DESCRIPTIVE_WAVEFORM_PE_CORRELATIONS.csv", pd.DataFrame(correlations))
    allmetrics = pd.read_csv(root / "tables/RETRIEVAL_PAIR_PER_SEED.csv")
    picked = []
    for dep in ("gwtc3", "gwtc4"):
        source = allmetrics.query("deployment == @dep and config == @config and split == 'test'")
        for role, rule in (("CFIX-baseline", "BASELINE"), (FINAL_CODE, "VAL-ENSEMBLE" if dep == "gwtc3" else "BASELINE")):
            f = source.loc[source.rule.eq(rule)].copy()
            f["final_version"] = role
            picked.append(f)
    finalmetrics = pd.concat(picked, ignore_index=True)
    dev.csv_write(root / "tables/FINAL_REUSED_INJECTION_PER_SEED.csv", finalmetrics)
    fields = ["macro_r_at_1", "macro_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"]
    summary = finalmetrics.groupby(["deployment", "final_version", "method"])[fields].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in summary.columns]
    dev.csv_write(root / "tables/FINAL_REUSED_INJECTION_SUMMARY.csv", summary)
    weights = []
    for dep in ("gwtc3", "gwtc4"):
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            cal = json.loads((root / f"evaluation/{config}/gwtc3/model_{ms}_eval_{es}/SELECTED_CONFIG.json").read_text())
            weights.append({"deployment": dep, "new_model_seed": ms if dep == "gwtc3" else "not_deployed",
                "baseline_deployment_seed": es, **dev.BASE.FROZEN_V93_WEIGHTS[dep][es],
                "waveform_ensemble_mode": cal["VAL-ENSEMBLE"]["mode"] if dep == "gwtc3" else "original",
                "waveform_gamma": cal["VAL-ENSEMBLE"]["gamma"] if dep == "gwtc3" else 0.,
                "new_mass_feature_weight": cal["NEW-PHYSICAL"]["mass_weight"] if dep == "gwtc3" else "not_applicable"})
    dev.csv_write(root / "tables/FINAL_FROZEN_WEIGHTS.csv", pd.DataFrame(weights))
    allbudget = pd.read_csv(root / "tables/PE_OFFICIAL_PER_SEED_AND_CONSENSUS.csv")
    allbudget = allbudget.loc[allbudget.seed.astype(str).eq("consensus") & allbudget.budget.eq(10)]
    ledger = pd.read_csv(root / "tables/RETRIEVAL_PAIR_SUMMARY.csv").query("split == 'test' and rule != 'BASELINE'")
    ledger["joined_config"] = ledger.config+"--"+ledger.rule
    ledger = ledger.merge(allbudget.drop(columns=["seed", "budget"]).rename(columns={"config": "joined_config"}),
        on=["joined_config", "deployment", "method"], validate="one_to_one")
    dev.csv_write(root / "tables/ALL_ABLATIONS_RECALL_PE_OFFICIAL_LEDGER.csv", ledger)
    return config, budget, summary, finalmetrics


def figures(root, budget, metrics):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.labelsize": 10})
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    colors = {"CFIX-baseline": "#6C757D", FINAL_CODE: "#237A69"}
    for col, dep in enumerate(("gwtc3", "gwtc4")):
        for n, code in enumerate(colors):
            b = budget.loc[budget.deployment.eq(dep) & budget.config.eq(code) & budget.method.eq("C_fixed") & budget.seed.astype(str).eq("consensus") & budget.budget.eq(10)].iloc[0]
            vals = [b.catastrophic_mc, b.BC_mc_ge_0p5, b.Dmax_le_3, b.official_frontend]
            axs[0, col].bar(np.arange(4)+(n-.5)*.34, vals, .34, color=colors[code], label="Baseline" if n == 0 else "Run-specific candidate")
            f = metrics.loc[metrics.deployment.eq(dep) & metrics.final_version.eq(code) & metrics.method.eq("C_fixed")]
            vals = f.macro_r_at_10.to_numpy()
            axs[1, col].scatter(np.full(len(vals), n)+np.linspace(-.07, .07, len(vals)), vals, s=22, color=colors[code])
            axs[1, col].errorbar(n, vals.mean(), vals.std(ddof=1), fmt="_", markersize=15, capsize=4, color=colors[code])
        axs[0, col].set_xticks(np.arange(4), ["Severe Mc\nconflict", "Mc BC\n>= 0.5", "Dmax\n<= 3", "Official\nfrontend"])
        axs[0, col].set_ylabel("Consensus Top-10 count")
        axs[0, col].set_ylim(0, 11)
        axs[0, col].set_title("O3" if dep == "gwtc3" else "O4a: retained original model")
        axs[1, col].set_xticks([0, 1], ["Baseline", "Candidate"])
        axs[1, col].set_ylabel("Reused injection R@10")
        axs[1, col].set_ylim(.6, 1)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    for ax, label in zip(axs.flat, "abcd"):
        ax.text(-.16, 1.04, label, transform=ax.transAxes, weight="bold", fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"figures/fig_final_encoder_PE_retrieval.{ext}", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, dep in zip(axes, ("gwtc3", "gwtc4")):
        for code in colors:
            frame = pd.read_parquet(root / f"final_results/{dep}/{code}/C_fixed/all_pairs.parquet").head(20)
            ax.plot(frame.consensus_rank, frame.pe_mc_bhattacharyya_coefficient, "o-", ms=3, lw=1, color=colors[code], label="Baseline" if code == "CFIX-baseline" else "Candidate")
        ax.axhline(.5, ls=":", color="black", lw=.8)
        ax.axhline(.1, ls="--", color="#AF423D", lw=.8)
        ax.set(xlabel="Consensus rank", ylabel="Chirp-mass posterior BC", title=dep, ylim=(-.03, 1.03))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"figures/fig_final_top20_Mc_PE.{ext}", dpi=300)
    plt.close(fig)


def report(root, config, budget, summary):
    auditfile = fresh.data_root(root) / "FINAL_GUARDRAIL_AUDIT.json"
    confirmation = json.loads(auditfile.read_text()) if auditfile.exists() else {"status": "PENDING", "per_model_mean_across3catalog_guardrails_pass": False}
    pe = budget.loc[budget.seed.astype(str).eq("consensus") & budget.budget.eq(10)]
    independence_path = fresh.data_root(root) / "GLOBAL_SOURCE_NOISE_INDEPENDENCE.json"
    independence = json.loads(independence_path.read_text()) if independence_path.exists() else {}
    protected_path = root / "audit/PROTECTED_INPUTS_FINAL.json"
    protected = json.loads(protected_path.read_text()) if protected_path.exists() else {"mismatches": ["not_checked"]}
    candidate_pe = pe.loc[pe.config.eq(FINAL_CODE)]
    pe_pass = (len(candidate_pe) == 4 and candidate_pe.catastrophic_mc.eq(0).all()
               and candidate_pe.Dmax_le_3.eq(10).all())
    complete = bool(confirmation["per_model_mean_across3catalog_guardrails_pass"] and pe_pass
                    and independence.get("source_noise_independence_pass", False)
                    and not protected["mismatches"])
    status = "TARGET_MET_ON_DECLARED_CHECKS_HOLD_FOR_AUTHOR_REVIEW" if complete else "TARGET_NOT_YET_MET_DO_NOT_DECLARE_SUCCESS"
    lines = ["# O3 波形编码器改进与 O4a 保留验证报告", "",
        f"版本代号：`{FINAL_CODE}`。计算服务器：`connect.westd.seetacloud.com:32328`。", "",
        f"最终检查状态：`{status}`。作者审核前不覆盖历史结果或论文。", "",
        "## 1. 本轮做了什么", "",
        "本轮实际训练了21个编码器：PHASE-SOURCE、RAW-PHASE-SOURCE、PSD-PHASE-SOURCE分别在O3/O4a各运行三个初始化，另做三个严格O3噪声开发对照。每个训练50epochs。不是只改旧模型后面的阈值或重排权重。所有失败消融均保存。",
        f"最终冻结O3配置为 `{config}--VAL-ENSEMBLE`；O4a保留原始C-fixed模型。新O4a模型未保持全部约束，不能声称O4a也获得新模型升级。", "",
        "O3仍保留旧waveform证据作为集成的一部分。新编码器单独替代旧模型时，注入检索与假对负担不达标；集成利用两种表示的互补性。这个限制不能省略。", "",
        "## 2. 未改变的内容", "",
        "- 输入仍为峰值2s、4096点、2048Hz，H1/L1双通道。24s只作为原物理预处理缓冲；不改为长窗口网络。",
        "- 一维时间查分表、天空Bayes-factor、Nside512、ordering修正、真实目录范围、C-fixed逐seed外层权重均不变。",
        "- 官方O3严格H1/L1为62事件、1891无序对；O4a严格范围74事件、2701无序对。不是早期混合O1-O3的53事件目录，也不是预计的64事件。",
        "- 不运行Hanabi，不把官方候选作为透镜标签，不修改论文或历史排名。", "",
        "## 3. 新编码器与分数计算", "",
        "仅从2s strain提取576个固定IMRPhenomD正交相位模板的匹配特征，与可训练的InceptionAttention原始波形分支合并。输出128维单位embedding和64-bin探测器系logMc预测分布。网络同时优化同源监督对比损失和logMc软标签交叉熵，两项梯度均进入编码器主体。",
        "模板网格覆盖64个logMc位置、3个质量比、3个有效自旋。这是有限的物理形态基，不是用真实事件PE选择模板，也不是完整模板搜索的Bayes factor。模型训练标签来自模拟源；真实PE、事件名称、时间差、天空或官方FPP不进入模型。", "",
        "每个运行期训练使用960个独立源父模板、7680个视图；模型validation使用240个独立源、480个视图，源父模板交叉数为0。每epoch对源做4轮均衡遍历，每batch选64个源、各2个不同视图。训练50epochs，AdamW学习率2e-4、weight decay1e-4、cosine衰减至2e-5、梯度范数上限5。",
        "相位特征主体从前轮仅模拟训练的PHASEBANK warm-start，再联合训练主体与原始波形分支。checkpoint按开发validation CE+0.2*SupCon最低值选取；温度从[0.5,0.75,1,1.25,1.5,2,3,4]按validation CE冻结。这是学习分布的温度，不是天空温度。",
        r"同源对比损失为 $L_{\rm SupCon}=-\langle |P(i)|^{-1}\sum_{p\in P(i)}\log[\exp(e_i^Te_p/0.1)/\sum_{a\ne i}\exp(e_i^Te_a/0.1)]\rangle_i$。训练总损失为 $L_{\rm SupCon}+L_{\rm CE}(\log M_c)$。同一源的两幅像不跨模型train/validation。",
        "模型validation与后续用于waveform密度/集成选参的BAYESTAR validation是两个任务不同的表；所有来源路径、实际参数与哈希均保留。重复使用的旧test及真实O3明确视作开发反馈，最终新确认才在候选冻结后生成。", "",
        r"新波形原始量为 $u_{ij}=z_{\rm val}(\cos(e_i,e_j))+k z_{\rm val}(-|\widehat{\log M}_{c,i}-\widehat{\log M}_{c,j}|)$。标准化的中位数与SD仅由模拟validation冻结，不按真实目录或每query重新拟合。",
        r"通过模拟validation的一维KDE密度比得到 $Z_{\rm new}=\log \widehat p(u|L)-\log \widehat p(u|N)$。这个学习校准量是ranking evidence，不是完整的物理透镜Bayes factor。",
        r"最终波形采用validation选择的 $(1-\gamma)Z_{\rm old}+\gamma Z_{\rm new}$，或 $Z_{\rm old}+\gamma\min(Z_{\rm new},0)$。O4a直接使用原 $Z_{\rm old}$。",
        r"最终三通道仍为 $S=w_WZ_W+w_TZ_T+w_SZ_S$，外层权重未重选。", "",
        "共识排名仍按三个部署rank的平均值从小到大，再按最坏rank从小到大、平均总分从大到小打破平局。因此共识表的平均分数不必逐行递减；没有另挑单seed或按PE重排。",
        tab(pd.read_csv(root / "tables/FINAL_FROZEN_WEIGHTS.csv")), "",
        "## 4. 开发过程与独立性的边界", "",
        "本轮已经看过真实O3及历史注入结果，并按作者要求把Mc不一致作为开发反馈。因此真实O3 PE改善是开发审计，不是盲检验，也不能用作透镜发现的显著性证据。官方重合只报告，不参与本轮最终候选配置选择。",
        "新确认开始前，对模型checkpoint、各seed校准器、旧模型、时间表和天空配置写入不可变哈希合同。新的信号/噪声结果不用于再次调参。",
        "GW-LMC可用且未使用过的系统不足：当前库存仅21个O3可用smooth系统、7个O4a可用smooth系统，未使用subhalo系统为0。因此新确认采用新BBH内禀参数/噪声块，但可复用透镜环境。它检验的是条件于这些环境的源/噪声推广，不是独立新透镜人口验证。未把重采样当作新增GW-LMC系统。", "",
        "## 5. 噪声来源核查", "",
        "旧G1所谓O3训练的48段噪声中16段来自O2；validation的16段中10段来自O1。这个来源混合在本报告中明确披露，不能继续叫全流程严格O3-run-matched。最终RAW-PHASE-SOURCE仍使用该混合开发噪声，可视为跨运行期噪声增强，但不能改写其来源。",
        "为检验影响，新建了O3-RUNMATCHED-RAW：保留同一源、SNR、网络与初始化方式，只将开发噪声/PSD换为O3-only，训练48个、验证16个独立256s参考块。该对照没有消除全部真实Top10冲突，故未选择。它仍从此前混合噪声模型warm-start，准确名称是O3-only噪声微调对照，不是从零开始纯O3训练。",
        "新的最终确认使用O3-only/O4a-only噪声，且与所有旧/新开发参考块排除重叠。", "",
        "## 6. 历史注入复用对照", "",
        "以下为已经看过的BAYESTAR注入目录，只能作为开发复用对照。各seed事件数177/187/195，真伴随对68/70/70；均值和SD跨三个部署seed计算。", "",
        compact_metric_table(summary), "",
        "## 7. 真实Top10：PE与官方阶段", "",
        r"严重Mc冲突的预先定义为 $BC_{M_c}<0.1$ 或 $D_{M_c}>5$。零严重冲突不等于所有后验相同，也不等于所有 $BC\ge0.5$。例如较小但非零重合仍须明确展示，不能叫物理确认。", "",
        tab(pe, ["deployment", "config", "method", "catastrophic_mc", "BC_mc_ge_0p5", "D_mc_le_3", "Dmax_le_3", "official_frontend", "official_hanabi"]), "",
        "上述零严重冲突结论只适用于三seed共识Top10，不适用于每个单模型Top10。O3新三通道三个单部署仍分别有1、3、0个严重Mc冲突；O4a保留基线的单部署为2、0、1个。不能写成每个seed都消除了异常。共识规则仍沿用原先规则，没有挑选最漂亮的单seed。",
        "O3全pair波形分数与公开Mc BC的描述性Spearman相关由0.4109变为0.7387；与负D_Mc的相关由0.3694变为0.6144。pair共享事件且已经用于开发审计，不提供普通独立pair显著性p值。",
        "事件级质量诊断也改善：相对于公开PE中位数，预测Mc误差达到2倍的事件数在三个seed中由15/16/15降到4/2/2；中位绝对log误差由0.186/0.214/0.148降到0.058/0.064/0.049。PE中位数不是注入真值，因此这是外部一致性审计，仍不能叫真实质量预测误差。",
        "Top10/20/50/100的全部PE预算、官方PO/ML或PO/Phazap、GOLUM/Hanabi阶段均在机器可读表中。公开Hanabi重合不代表被确认透镜；这里没有重新运行Hanabi。",
        "O4a旧汇总未保存dL的BC，本轮仅从相同冻结公开PE group补充距离描述量，未改内禀PE指标或分数。dL是受放大率影响的表观距离，不加入Dmax，也不要求两幅像的dL必须一致。",
        "当前官方O3基线三通道Top10的三个严重冲突：GW191126_115259--GW191129_134029从rank1到193；GW191105_143521--GW191126_115259从rank3到12；GW200316_215756--GW200322_091133从rank7到48。其中rank12说明只是移出Top10，并非全目录不再有假相似。旧GW190924相关pair不在当前官方O3 scope，不能把未入选算作模型修复。",
        "完整新旧排名：`final_results/<run>/<version>/<method>/all_pairs.parquet`及BOM CSV。每个pair保留总分、三个贡献、Mc/q/chi_eff/dL的BC和D、官方FPP与阶段。", "",
        "## 8. 新信号/噪声确认", "",
        f"状态：`{confirmation['status']}`。",
        "每个运行期3个新目录，每目录35个smooth伴随系统、35个subhalo伴随系统、50个无透镜新源，共190事件、70真对；两个运行期合计720个新BBH源、1140事件、96个独立256s噪声块。三模型逐一应用于三目录，不能把9行当作9个独立训练seed。",
        "质量与SNR沿用预先定义的均衡coverage proposal，而非天体物理发生率先验。每幅像独立PSD-optimal target-SNR缩放；不是本项目曾讨论的response-derived共同振幅实验。",
        "BAYESTAR保持基线条件定位协议：已知模拟内禀参数、对应PSD、独立Gaussian matched-filter测量噪声，再生成Nside512图。它不是完整BBH PE，也不是对同一份非高斯注入strain重新测出的定位。不能将本轮称为完全一致的strain-to-PE端到端确认。",
        f"主要表：`{fresh.data_root(root).name}/PAIRED_METRICS_AND_CI.csv`、`MODEL_MEAN_OVER_CATALOGS.csv`、`SUMMARY.csv`。每模型先对3个新目录取均值，再检查R10下降不超过0.02、AP下降不超过0.005、F50/F90增加不超过10%的冻结工程容差；所有逐目录失败也必须报告。",
        "R1/R10使用10000次分层system bootstrap，双向query不拆开；pair量给2000次source-block和noise-block两套条件区间。模型SD、目录随机性和bootstrap条件不确定性不能混为一谈。这些容差是开发验收规则，不是文献推导的自然常数。", ""]
    lines += ["新waveform校准的OOD也单列：前两个模型的新目录OOD约0.19%-0.86%，第三个约6.97%-9.68%。这里OOD指超出validation原始组合分数min/max；lookup网格另有2%边距，超网格时沿用冻结端点常数，而非线性外推。不能把尾部分数当作精确校准的物理Bayes factor，也没有因新测试结果重拟合它。完整表见`tables/FINAL_WAVEFORM_OOD_AUDIT.csv`。", ""]
    lines += [f"实际有效的新确认目录：`{fresh.data_root(root).name}/`。旧`confirmation/`只保留全局GPS噪声冲突修复前的生成记录，不是验收结果。模型、源清单、种子和科学阈值未因这一修复改变。", ""]
    if (fresh.data_root(root) / "SUMMARY.csv").exists():
        lines += [compact_metric_table(pd.read_csv(fresh.data_root(root) / "SUMMARY.csv"), True), "",
            f"逐目录/模型/方法检查：{confirmation['per_catalog_model_method_point_checks']}/{confirmation['per_catalog_model_method_checks_total']}通过。O3新目录三通道R10约0.8865，不能与旧目录0.8655直接当作同数据增益；同一新目录的正确baseline是0.8063。O4a新目录0.7000也不能与旧目录0.7765解释为退化，因为新旧模型在同一新目录上完全相同。",
            "O3九个模型-目录组合的AUPRC source-block差值CI均为正，但部分R10系统CI与F50/F90噪声block差值CI跨过0。因此不声称每一项改善都在所有不确定性口径下显著。三目录样本量有限。", ""]
    physical_path = fresh.data_root(root) / "NEW_ENCODER_PHYSICAL_DIAGNOSTICS.csv"
    if physical_path.exists():
        lines += ["### 新注入的Mc预测诊断", "", tab(pd.read_csv(physical_path)), "",
            "这是模拟条件下的预测校准，不是对真实GWTC获得的新PE后验。个别真实事件预测仍偏离公开PE，详见波形/Mc对照图；本轮改善的是排序一致性，不能声称已准确恢复所有真实事件质量。", ""]
    lines += ["## 9. 数值、环境与文件保护", "",
        "- 对比学习损失的置换不变性、有限梯度及目标归一化已检查。PSD模板CPU/GPU同条件最大差1.34e-7。",
        "- 历史waveform重新推理的抽样回归，最大分数差4.58e-5，R10一致；这是浮点路径差异，不是重新拟合。真实最终O4a直接复用冻结分数，严格逐项相同。",
        "- 新服务器cryptography的Python与Rust扩展不一致，导致首次BAYESTAR导入失败。只在本实验`environment/crypto`安装相同版本46.0.6的完整wheel，使用PYTHONPATH隔离；未改全局环境。原失败日志保留，原source plan和种子继续使用。",
        "- 受保护历史输入共108项，最终前后SHA核对见`audit/PROTECTED_INPUTS_FINAL.json`。无原始strain、PE HDF5、旧checkpoint或历史结果被删除。",
        "- 新确认720组BBH质量参数与7200条历史/开发源元数据比较，无相同质量对；96个256s参考块与320条历史/开发参考区间做全局GPS检查，无重叠。透镜环境仍允许重复，不等于独立新透镜系统。",
        "- 新目录只有17885个非伴随pair，经验FPR最小非零步长约5.59e-5。任何沿用旧schema的1e-5列均不能视为可靠低误配率证据；本轮主要比较AUPRC、F50/F90和Top-B，不作1e-5显著性结论。", "",
        "## 10. 依据与解释", "",
        "1. [Khosla等，Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)：提供同类正样本对比损失的依据。本轮把类定义为模拟源身份；该论文并不保证GW效果，效果由本轮实验给出。",
        "2. [PyCBC matched-filter文档](https://pycbc.org/pycbc/latest/html/filter.html)：PSD加权与模板匹配提供固定物理特征的依据。新特征不等于完整搜索pipeline或PE。",
        "3. [Singer与Price，BAYESTAR](https://arxiv.org/abs/1508.03634)及[官方实现](https://lscsoft.docs.ligo.org/ligo.skymap/_modules/ligo/skymap/bayestar.html)：快速天空定位算法依据；本轮继承的条件模拟与真实单事件PE之间的差异仍需明确。", "",
        "## 11. 可复现文件", "",
        "- `scripts/`：本轮脚本及读取的历史辅助脚本快照。",
        "- `contracts/`：每轮协议、逐seed校准、候选冻结；`models/`保存全部成功/失败模型。",
        "- `evaluation/`：每个配置和seed的validation、复用test、真实pair表及选参网格。",
        "- `tables/`：全部消融recall/FP、PE/官方预算、新旧比较和权重。",
        f"- `{fresh.data_root(root).name}/`：有效新源清单、noise block清单、MOC、每模型分数及验收。旧`confirmation/`仅供修复追溯。",
        "- `final_results/`：最终候选与旧基线的全pair、Top100。",
        "- `manifest/`：逐文件SHA；紧凑包不含原始strain、全部dense map、私钥或密码。", "",
        "未经作者审核，不写回论文、不替换历史结果。即使本轮工程检查通过，也不能宣称发现真实透镜，或宣称已经完成独立透镜人口/full-PE确认。", ""]
    path = root / "reports/FINAL_ENCODER_EXPERIMENT_REPORT_CN.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    dev.json_write(root / "contracts/DELIVERY_STATUS.json", {"code": FINAL_CODE, "status": status,
        "fresh_confirmation_status": confirmation["status"], "goal_met_under_declared_conditional_checks": complete,
        "consensus_Top10_PE_check": bool(pe_pass), "single_seed_zero_conflicts_claimed": False,
        "global_source_noise_check": independence.get("source_noise_independence_pass", False),
        "protected_inputs_unchanged": not protected["mismatches"],
        "paper_adoption": False, "independent_lens_population_confirmed": False, "full_strain_PE_confirmed": False,
        "old_O4_encoder_retained": True})
    return complete


def package(root):
    pack = dev.PROJECT / "packages" / f"{root.name}_deliverables.tar.gz"
    if pack.exists():
        raise RuntimeError("Package already exists; refuse overwrite")
    required = ["README_CN.md", "contracts/DELIVERY_STATUS.json", "reports/FINAL_ENCODER_EXPERIMENT_REPORT_CN.md",
        "reports/FINAL_TOP10_PE_OFFICIAL_CN.md", "reports/ALL_ABLATIONS_RECALL_PE_OFFICIAL_CN.md",
        "tables/FINAL_BASELINE_PE_OFFICIAL_BUDGETS.csv", "tables/FINAL_REUSED_INJECTION_PER_SEED.csv",
        "audit/FINAL_EXECUTABLE_CHECKS.json", "audit/PROTECTED_INPUTS_FINAL.json",
        "confirmation_global_noise_fixed/FINAL_GUARDRAIL_AUDIT.json", "confirmation_global_noise_fixed/GLOBAL_SOURCE_NOISE_INDEPENDENCE.json",
        "confirmation_global_noise_fixed/PAIRED_METRICS_AND_CI.csv", "figures/fig_fresh_confirmation.pdf",
        "figures/fig_known_pairs_waveform_Mc.pdf", "manifest/SOURCE_CODE_PROVENANCE.csv"]
    absent = [name for name in required if not (root / name).is_file()]
    if absent:
        raise RuntimeError(f"Missing required deliverables: {absent}")
    include = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        rel = file.relative_to(root)
        if "__pycache__" in rel.parts:
            continue
        if rel.parts[0] in ("cache", "data") and file.suffix not in (".json", ".csv", ".parquet"):
            continue
        if rel.parts[0] == "confirmation" and file.suffix == ".fits":
            # Preserve old generation manifests, not duplicate its bulky MOCs.
            continue
        if rel.parts[:2] == ("environment", "crypto"):
            continue
        if file.name == "resume.pt" or file.name.endswith("_sky512.npy") or file.name.endswith("_full24.npy") or file.name == "reference.npy":
            continue
        if rel.parts[0] == "manifest" and file.name in ("DELIVERY_SHA256.csv", "PACKAGE_VALIDATION.json"):
            continue
        include.append(file)
    dev.csv_write(root / "manifest/DELIVERY_SHA256.csv", pd.DataFrame([
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": dev.sha(p)} for p in include]))
    include.append(root / "manifest/DELIVERY_SHA256.csv")
    with tarfile.open(pack, "w:gz", compresslevel=6) as tar:
        for file in include:
            tar.add(file, arcname=f"{root.name}/{file.relative_to(root)}", recursive=False)
    digest = dev.sha(pack)
    pack.with_suffix(pack.suffix + ".sha256").write_text(f"{digest}  {pack.name}\n")
    failed = []
    manifest = pd.read_csv(root / "manifest/DELIVERY_SHA256.csv")
    with tarfile.open(pack, "r:gz") as tar:
        for row in manifest.itertuples():
            data = tar.extractfile(f"{root.name}/{row.path}")
            h = hashlib.sha256()
            for block in iter(lambda: data.read(8 << 20), b""):
                h.update(block)
            if h.hexdigest() != row.sha256:
                failed.append(row.path)
    if failed:
        raise RuntimeError(f"Package internal hash mismatch: {failed}")
    dev.json_write(root / "manifest/PACKAGE_VALIDATION.json", {"package": str(pack), "sha256": digest,
        "members": len(include), "internal_hash_failures": failed, "bytes": pack.stat().st_size,
        "required_deliverables_present": required})
    print(json.dumps({"package": str(pack), "sha256": digest, "files": len(include)}), flush=True)


def run(root, make_package=False):
    config, budget, summary, metrics = collect(root)
    figures(root, budget, metrics)
    complete = report(root, config, budget, summary)
    if make_package:
        package(root)
    print(json.dumps({"goal_check": complete, "report": str(root / "reports/FINAL_ENCODER_EXPERIMENT_REPORT_CN.md")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--package", action="store_true")
    args = p.parse_args()
    run(args.root, args.package)
