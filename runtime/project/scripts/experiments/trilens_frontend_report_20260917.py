"""Report and package measured quick frontend runs without adoption claims."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def main(root):
    final=json.loads((root/"contracts/FINAL_STATUS.json").read_text())
    if final["state"]!="QUICK_TIMING_COMPLETE_OFFICIAL_PO_AND_MATCHED_RECALL_PENDING":
        raise RuntimeError("Incomplete pilot, no success report")
    if not final["historical_inputs_unchanged"]:
        raise RuntimeError("Protected inputs changed")
    if (root/"manifest/SHA256SUMS.txt").exists():
        raise RuntimeError("No overwrite")
    table=pd.read_csv(root/"tables/timings.csv")
    total=table[table.stage.eq("TOTAL_PRODUCTS_READY_TO_SCORES_AND_RANKS")]
    if len(total)!=12:
        raise RuntimeError("Expected 2 methods x 3 sizes x 2 repeats")
    summary=total.groupby(["method","n_events"]).agg(
        wall_median_s=("wall_seconds","median"),wall_min_s=("wall_seconds","min"),wall_max_s=("wall_seconds","max"),
        process_cpu_median_s=("process_cpu_seconds","median"),technical_repeats=("wall_seconds","size")).reset_index()
    summary["n_pairs"]=summary.n_events*(summary.n_events-1)//2
    summary.to_csv(root/"tables/FRONTEND_TIMING_SUMMARY.csv",index=False,encoding="utf-8-sig")
    stage=table[(table.n_events==6)&(~table.stage.isin(["TOTAL_PRODUCTS_READY_TO_SCORES_AND_RANKS"]))]
    stage=stage.groupby(["method","repetition","stage"])["wall_seconds"].sum().reset_index()
    stage.groupby(["method","stage"]).wall_seconds.agg(["median","min","max"]).to_csv(root/"tables/STAGES_6_EVENTS.csv",encoding="utf-8-sig")
    checks=json.loads((root/"results/checks.json").read_text())
    errors=[c["max_abs_difference"] for c in checks if "max_abs_difference" in c]
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.1))
    colors={"TriLens":"#267b8c","Phazap":"#ab4967"}
    for method,g in summary.groupby("method"):
        axes[0].errorbar(g.n_events,g.wall_median_s,
            yerr=np.array([g.wall_median_s-g.wall_min_s,g.wall_max_s-g.wall_median_s]),
            marker="o",capsize=3,color=colors[method],label=method)
    axes[0].set(xlabel="Real events (all pairs)",ylabel="Measured wall time [s]",yscale="log",xticks=[2,4,6])
    axes[0].legend(frameon=False)
    six=summary[summary.n_events==6]
    axes[1].bar(six.method,six.wall_median_s,color=[colors[m] for m in six.method])
    axes[1].set(ylabel="Measured wall time [s]",title="Six events, 15 pairs")
    for _,r in six.iterrows():
        axes[1].text(r.method,r.wall_median_s,f"{r.wall_median_s:.2f} s",ha="center",va="bottom")
    fig.suptitle("Products-ready scoring pilot: native implementations, shared load",fontsize=11)
    fig.text(.5,.015,"Existing PE products; GPU vs CPU; neural padding charged. No matched-recall or overall speedup claim.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.055,1,.94])
    fig.savefig(root/"figures/FRONTEND_PILOT.pdf")
    fig.savefig(root/"figures/FRONTEND_PILOT.png",dpi=180)
    plt.close(fig)
    lines=["# TriLens 完整评分计时：快速验证第二轮", "",
        "**状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。**",
        "本轮为小规模计算验证，不是方法优越性确认。没有训练新模型，没有修改历史分数、权重、排名或论文；没有运行 PE sampler、Fast-GOLUM 或 Hanabi。另两组统一 SNR 实验没有停止。", "",
        "## 实测范围", "",
        "沿用前一轮按事件名称哈希固定的六个 O3 事件，取嵌套 2/4/6 事件子目录，每个子目录比较全部 1/6/15 对。各方法两次技术重复，第二次反转执行顺序。没有按结果选择事件。两次重复不是独立天体物理实验，也不是充分的性能置信区间。", "",
        "TriLens 从本地原始应变开始：读取 26 s 信号段和 256 s 噪声参考，重估 PSD，白化与生成 2 s/16 s 输入，重算所有模板匹配特征及三个 seed 的短窗、RNC、ordered-mass、多尺度、条件参数预测和校准。另从公开 PE 原生天空图显式 NESTED→RING 并生成 Nside=512 分数，读取时间查找表，重新做三通道融合和共识排序。没有用缓存 channel scores 代替推理。", "",
        "Phazap 使用未修改的作者 0.3.3 软件，读取全部公开后验样本，重新生成相位，再比较全部 pair、输出 DJ 等统计量。没有使用此前失败的 512/1024 样本截断来加速。DJ 排序只作诊断，未复现官方背景校准 FPP、联合前端门槛或最终候选名单。", "",
        "两者共同边界为 public-products-ready：原始 strain、公开 posterior/map 已经在磁盘上。双方均不计 PE 生产、网络下载、模型训练、离线校准。TriLens 的模板谱生成属于离线准备，模板读取和按事件 PSD 适配计入。软件启动/导入单列；输入哈希与正确性审计在计时外。没有清空 OS 页缓存，首次调用不称 disk-cold。", "",
        "## 计时结果", "",
        "| 方法 | 事件/对 | 中位耗时(s) | 两次范围(s) | 进程 CPU(s) |", "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | {r.n_events}/{r.n_pairs} | {r.wall_median_s:.4f} | {r.wall_min_s:.4f}–{r.wall_max_s:.4f} | {r.process_cpu_median_s:.4f} |")
    lines += ["", "这些是完整在线评分部分的实际耗时，不是几毫秒缓存排序。输入边界已对齐，但两者物理信息、设备、数值实现并不相同：TriLens 用 GPU，Phazap 用 CPU；真实 PE/map 可以含 Virgo，短窗仍为 H1/L1。因此不把耗时相除称为等资源、等检出能力的整体加速倍数。", "",
        "服务器同时在运行 UAB 训练，这些数字是共享负载下的可行性测量，不是独占节点 benchmark。Python/CPU 库限制 2 线程；记录 CPU 配额、RAM 上限和 GPU 显存。进程 CPU 时间不是 GPU 时间，也不是能耗。正式实验要安排独占测量并补充资源积分。", "",
        "## 数值失败与修正", "",
        "第一轮仅在 wrapper 导入路径上失败；第二轮在小 batch 的混合精度输出复现上失败。均原样保存，不改判 PASS。只读诊断表明：BF16 网络随 batch 形状变化，短窗 embedding 最大差约 1.4e-3，后续 KDE 会放大差异；原始输入并未改变。", "",
        "第三轮保持原模型和精度，使用零值占位恢复归档 62 事件的网络 batch 形状及事件槽位。未选事件不是新的真实输入；占位计算全部计入耗时。所有实际特征、天空图和 pair 仍只计算 2/4/6 个真实事件。这个实现优先可复现，不能拿三点斜率宣称优化后的大目录扩展性能。正式部署应另冻固定 batch/精度规则并审计，不能悄悄改变模型数值语义。", "",
        f"第三轮 {len(checks)} 项核对全部通过；三通道分数对归档对应子集最大绝对误差 {max(errors):.8g}。子目录共识名次完全一致；原始 24 s 输入逐值完全一致；Phazap 与前一轮全部 posterior 的输出也通过核对。", "",
        "注意子目录共识必须在子目录内重新聚合每个 seed 的 rank，不能要求等于原完整目录共识列表简单截取。这里核对的是同一子目录、同一冻结方法的重放，不是产生新正式真实候选。", "",
        "## 尚未完成，不能提前下结论", "",
        "1. 官方 PO：已核对论文要求联合质量、自旋、天空重合及时间项。现有工作区没有本轮已验证的官方 PO 端到端入口，GitLab 获取尝试超时。本包不把简化 Mc BC 当成官方 PO，也不把读取官方分数表的时间当成 PO 计算。", "2. 官方 PO/Phazap 联合前端：尚无统一验证过的背景 FPP 校准和选取流程，不报告其耗时或胜负。", "3. 相同召回率的成本：新独立注入的模型训练仍在运行，且尚未为 PO/Phazap 生产对应合格的联合后验。BAYESTAR sky 和神经网络预测质量分布不能替代这些后验。没有这些输入，就没有本轮有效的两方法 90% pair-recall 比较。", "4. 后续贝叶斯工作量：未对同一预选代表样本完成 Fast-GOLUM/Hanabi 实测，故不编造每对时长或已节省核时。", "",
        "## 后续最小实验", "",
        "完成正在运行的 A_NEUTRAL/B_CUE，两者不可混为一个独立成功结果；正式速度-质量主比较优先使用去人为 SNR 线索的 A。另冻结少量 source/noise-disjoint PE pilot，先测单事件合格后验的 P50/P90/失败率，再决定是否扩大量级。旧 rapid sky-only PE 不够用于联合 PO/Phazap。", "",
        "每个方法仅在 validation 选择达到目标 companion-pair recall 的阈值。test 冻结应用同一阈值，保留所有 ties，报告实际 pair recall 及区间，而不是强制 test 恰好 90%。若实际召回明显不同，不宣称 equal-recall 优越性；曲线可作独立诊断，不能反馈选参。R@10 单列。", "",
        "使用相同后续配置，按实际通过前端及 Fast-GOLUM 的候选计成本。总核时/GPU时与并行墙钟完成时间分开，不能把 job 时长求和当成并行完工时间。若仅抽样后续任务，需冻结分层抽样及权重，估算结果与实测分列。system/noise/catalog 分块不确定度，禁止把相关 pairs 当独立样本。", "",
        "包内 trilens_retention_cost.py 提供 validation 门槛和实际候选成本核算单元测试；测试 PASS 仅表示代码逻辑检查，不代表实测召回或成本结果。", "",
        "## 依据与交付", "",
        "官方 O4a 流程为 3,486→105→50，前端采用 PO/Phazap FPP 任一小于 1%。不能用所有 pair 都跑 Hanabi 的假想路线作对手。[LVK 原文 Figure 1、Section 4.1、Appendix A](https://dcc.ligo.org/public/0201/P2500419/010/O4aLensingPaperMTapproved.pdf)。",
        "[Phazap 作者软件](https://github.com/ezquiaga/phazap)；[LensingFlow 方法文献](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzag003/8424226)。", "",
        f"服务器：connect.westd.seetacloud.com:32328。完整目录：`{root}`。原始 strain/PE/模型仍按 manifest 路径保留，紧凑包不重复打包这些大文件或 phases，不包含凭据。", "",
        "本轮可以确认完整评分重放与小规模实测可行；不能确认保持召回率后减少联合分析成本的科学目标已经达成。"]
    (root/"reports/TRILENS_FRONTEND_QUICK_REPORT_CN.md").write_text("\n".join(lines)+"\n")
    (root/"README_CN.md").write_text("# TRILENS-FRONTEND-PILOT-02\n\n先读 reports/TRILENS_FRONTEND_QUICK_REPORT_CN.md。\n\n实测完整评分，不是官方全流程或同召回率成本比较成功。历史结果未覆盖。\n\nHOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n")
    files=[]
    for path in root.rglob("*"):
        if not path.is_file() or "phases" in path.relative_to(root).parts or "__pycache__" in path.parts:
            continue
        if path.stat().st_size>10*2**20:
            raise RuntimeError("Unexpected large artifact")
        data=path.read_bytes()
        if any(token in data for token in (b"BEGIN RSA PRIVATE KEY",b"BEGIN OPENSSH PRIVATE KEY",b"sshpass -p")):
            raise RuntimeError("Credential marker found")
        files.append(path)
    manifest=root/"manifest/SHA256SUMS.txt"
    manifest.write_text("".join(sha(path)+"  "+str(path.relative_to(root))+"\n" for path in sorted(files)))
    files.append(manifest)
    package=root.parent/(root.name+"_deliverables.tar.gz")
    if package.exists():
        raise RuntimeError("No package overwrite")
    with tarfile.open(package,"w:gz") as handle:
        for path in sorted(files):handle.add(path,arcname=root.name+"/"+str(path.relative_to(root)),recursive=False)
    failures=[]
    with tarfile.open(package,"r:gz") as handle:
        for line in manifest.read_text().splitlines():
            expected,name=line.split("  ",1)
            actual=hashlib.sha256(handle.extractfile(root.name+"/"+name).read()).hexdigest()
            if expected!=actual:failures.append(name)
    digest=sha(package)
    package.with_suffix(".gz.sha256").write_text(digest+"  "+package.name+"\n")
    result=dict(package=str(package),sha256=digest,bytes=package.stat().st_size,members=len(files),
                internal_hash_failures=failures,method_comparison_complete=False,quick_scoring_timing_complete=True)
    package.with_suffix(".verification.json").write_text(json.dumps(result,indent=2))
    if failures:raise RuntimeError("Archive validation failed")
    print(summary.to_string(index=False));print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
