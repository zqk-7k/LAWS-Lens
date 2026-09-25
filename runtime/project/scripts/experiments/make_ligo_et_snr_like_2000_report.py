from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("runs/ligo_et_snr_like_2000_full_catalog_20260623")
DATA_ROOT = Path("data_generation/ligo_et_snr_like_2000_outputs")
SUMMARY_PATH = ROOT / "fresh50_full_catalog_summary.csv"
SNR_PATH = DATA_ROOT / "snr_summary_from_npy.csv"


def fmt(x, nd: int = 4) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def md_table(frame: pd.DataFrame, cols: list[str], names: list[str] | None = None, nd: int = 4) -> str:
    names = names or cols
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join(["---" for _ in cols]) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join(["" for _ in cols]) + " |")
        return "\n".join(lines)
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col], nd) if col in frame.columns else "" for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    snr = pd.read_csv(SNR_PATH) if SNR_PATH.exists() else pd.DataFrame()

    variant_cn = {
        "waveform_only": "只用波形相似度",
        "time_only": "只用观测触发时间差",
        "true_sky_overlap_only": "只用真实 sky overlap（理想上限）",
        "predicted_sky_overlap_only": "只用机器学习预测 sky overlap",
        "waveform_plus_time": "波形 + 时间重排",
        "waveform_plus_predicted_sky_overlap": "波形 + 预测 sky overlap 重排",
        "waveform_plus_time_plus_predicted_sky_overlap": "波形 + 时间 + 预测 sky overlap 重排",
        "waveform_plus_true_sky_overlap": "波形 + 真实 sky overlap 重排（理想上限）",
        "waveform_plus_time_plus_true_sky_overlap": "波形 + 时间 + 真实 sky overlap 重排（理想上限）",
    }
    stage_cn = {
        "direct_full_catalog": "直接全 catalog 排序",
        "hard_trained_full_catalog_rerank": "hard-negative 训练后全 catalog 重排",
    }
    metric_cols = ["r@1", "r@5", "r@10", "top_1pct", "top_5pct", "top_10pct", "median_true_rank"]

    overall = df[df["subset"].eq("overall")].copy()
    overall["method"] = overall["variant"].map(variant_cn).fillna(overall["variant"])
    overall["stage_label"] = overall["stage"].map(stage_cn).fillna(overall["stage"])
    main_show = overall[["data_mode", "method", "stage_label"] + metric_cols].rename(
        columns={"method": "variant", "stage_label": "stage"}
    )
    noisy_overall = main_show[main_show["data_mode"].eq("noisy")]
    pure_overall = main_show[main_show["data_mode"].eq("pure")]

    major_variants = [
        "waveform_only",
        "time_only",
        "predicted_sky_overlap_only",
        "true_sky_overlap_only",
        "waveform_plus_time",
        "waveform_plus_time_plus_predicted_sky_overlap",
        "waveform_plus_true_sky_overlap",
    ]
    split = df[df["subset"].isin(["SIS", "PM"]) & df["variant"].isin(major_variants)].copy()
    split["method"] = split["variant"].map(variant_cn).fillna(split["variant"])
    split["stage_label"] = split["stage"].map(stage_cn).fillna(split["stage"])
    split_show = split[["data_mode", "subset", "method", "stage_label"] + metric_cols].rename(
        columns={"method": "variant", "stage_label": "stage"}
    )

    best = (
        df.sort_values(["data_mode", "subset", "r@10", "r@1"], ascending=[True, True, False, False])
        .groupby(["data_mode", "subset"], as_index=False)
        .head(1)
        .copy()
    )
    best["method"] = best["variant"].map(variant_cn).fillna(best["variant"])
    best["stage_label"] = best["stage"].map(stage_cn).fillna(best["stage"])
    best_show = best[
        ["data_mode", "subset", "method", "stage_label", "r@1", "r@5", "r@10", "top_1pct", "median_true_rank"]
    ].rename(columns={"method": "variant", "stage_label": "stage"})

    comp_cols = [
        "data_mode",
        "query_total",
        "catalog_total",
        "sis_lensed_images",
        "sis_unlensed",
        "pm_lensed_images",
        "pm_unlensed",
        "total_lensed_images",
        "total_unlensed",
    ]
    comp = overall[comp_cols].drop_duplicates().copy()

    train_rows = []
    for mode in ["noisy", "pure"]:
        history_path = ROOT / "fresh_mixed_encoders" / f"ligo_{mode}_mixed_sis_pm_ep50" / "history.csv"
        if history_path.exists():
            hist = pd.read_csv(history_path)
            train_rows.append(
                {
                    "data_mode": mode,
                    "epochs": int(len(hist)),
                    "first_loss": float(hist["loss"].iloc[0]),
                    "last_loss": float(hist["loss"].iloc[-1]),
                    "mean_epoch_s": float(hist["epoch_s"].mean()),
                    "total_epoch_s": float(hist["epoch_s"].sum()),
                    "batches_per_epoch": int(hist["batches"].median()),
                    "batch_pairs_per_epoch": int(hist["batch_pairs"].median()),
                }
            )
    train = pd.DataFrame(train_rows)

    snr_show = snr.copy()
    if not snr_show.empty:
        snr_show = snr_show[["dataset", "image", "n", "median", "q10", "q90", "q99", "min", "max"]]

    lines: list[str] = []
    lines += [
        "# LIGO ET-like SNR 2000 样本小测试实验报告",
        "",
        "生成时间：2026-06-23",
        "",
        "## 1. 实验目的",
        "",
        "此前 LIGO 数据的 noisy 结果较差，一个主要怀疑是 LIGO 注入噪声后整体 SNR 偏低，导致波形相似度阶段难以稳定识别强透镜双像。为验证这一点，本实验新生成一份小规模 LIGO 测试数据，并把数据生成条件调到更接近 ET 数据的 SNR 分布，再使用当前 catalog-level ranking 流程重新训练和评估。",
        "",
        "本实验关注三个问题：",
        "",
        "1. 当 LIGO 数据的 SNR 提高到 ET-like 水平后，纯波形检索是否明显改善。",
        "2. 在 mixed catalog 场景下，SIS、PM、未透镜信号同时存在时，catalog-level ranking 的表现如何。",
        "3. waveform、time、sky overlap、预测 sky overlap 及其组合各自贡献如何。",
        "",
        "## 2. 数据设置",
        "",
        f"数据输出目录：`{DATA_ROOT}`",
        "",
        "本次数据不是从旧数据裁剪，也不是简单重标定旧波形，而是重新生成的一组 LIGO 小测试数据。生成时保留原有 LIGO 双探测器 H1/L1 形式，并通过降低红移范围提升可观测 SNR，使其更接近 ET 数据的强信号分布。",
        "",
        "主要设置：",
        "",
        "- detector: LIGO H1/L1",
        "- lensed families: SIS、PM",
        "- unlensed: 单独生成",
        "- 每类源样本数：2000",
        "- SIS：2000 对透镜双像，A/B 两像各 2000 条",
        "- PM：2000 对透镜双像，A/B 两像各 2000 条",
        "- unlensed：2000 条",
        "- PM 透镜质量范围沿用扩展版：10^4 到 10^10",
        "- 数据不覆盖原有 10000 样本数据",
        "",
        "数据目录结构：",
        "",
        "```text",
        "data_generation/ligo_et_snr_like_2000_outputs/",
        "  unlensed_GW_events_LIGO_et_snr_like_2000/",
        "  SIS_GW_events_LIGO_et_snr_like_2000/",
        "  PM_GW_events_LIGO_et_snr_like_2000/",
        "```",
        "",
        "用于训练/评估的 match-style 软链接目录：",
        "",
        "```text",
        "data_generation/ligo_et_snr_like_2000_matchroots/LIGO/",
        "  SIS_data_0222",
        "  PM_data_0222",
        "  Unlensed_data_0222",
        "```",
        "",
        "## 3. SNR 分布检查",
        "",
        md_table(snr_show, ["dataset", "image", "n", "median", "q10", "q90", "q99", "min", "max"], nd=3),
        "",
        "SNR 分布说明：",
        "",
        "- unlensed median SNR 约 16.5，明显高于之前普通 LIGO 低 SNR 数据。",
        "- SIS 双像合并 median SNR 约 43.8，PM 双像合并 median SNR 约 31.6。",
        "- 该设置能够作为“提高 LIGO 信号强度后，波形检索是否改善”的小规模验证集。",
        "- 这不是最终真实天体分布，只是一个诊断性小测试数据集，用来隔离 SNR 对模型表现的影响。",
        "",
        "SNR 图目录：",
        "",
        "```text",
        "data_generation/ligo_et_snr_like_2000_outputs/snr_distribution_plot/",
        "```",
        "",
        "## 4. 实验协议",
        "",
        f"结果目录：`{ROOT}`",
        "",
        "本实验采用当前项目里的 mixed full-catalog ranking 流程：SIS、PM、未透镜事件放入同一个 LIGO catalog 中，分别评估 pure 和 noisy。该设置比单独 SIS 或 PM 更接近真实检索场景，因为候选池中同时存在不同透镜模型和未透镜干扰事件。",
        "",
        "### 4.1 Catalog 构成",
        "",
        md_table(
            comp,
            comp_cols,
            [
                "mode",
                "query_total",
                "catalog_total",
                "SIS lensed images",
                "SIS unlensed",
                "PM lensed images",
                "PM unlensed",
                "total lensed",
                "total unlensed",
            ],
            nd=0,
        ),
        "",
        "解释：",
        "",
        "- test catalog 总事件数为 1800。",
        "- 其中 SIS lensed images 为 600，PM lensed images 为 600，未透镜共 600。",
        "- 有效 query 是 1200 个透镜像；每个透镜像的真值目标是同一透镜系统的另一幅像。",
        "- 未透镜事件进入候选池，但不作为正样本 query。",
        "",
        "### 4.2 模型和训练配置",
        "",
        "波形模型沿用当前主流程：",
        "",
        "- backbone: InceptionTime",
        "- preprocess: bandpass",
        "- bandpass: 40-580 Hz",
        "- target_len: 8192",
        "- stride: 2",
        "- epochs: 50",
        "- loss: NT-Xent contrastive loss",
        "- mixed training: SIS lensed pair、PM lensed pair、unlensed self-pair 共同训练 embedding encoder",
        "- catalog evaluation: 对测试 catalog 全量候选排序，不只是在小候选池内排序",
        "",
        "训练耗时和 loss：",
        "",
        md_table(
            train,
            [
                "data_mode",
                "epochs",
                "first_loss",
                "last_loss",
                "mean_epoch_s",
                "total_epoch_s",
                "batches_per_epoch",
                "batch_pairs_per_epoch",
            ],
            ["mode", "epochs", "first loss", "last loss", "mean epoch s", "total epoch s", "batches", "batch pairs"],
            nd=4,
        ),
        "",
        "### 4.3 评价指标",
        "",
        "- R@1：真实另一幅像排在第 1 位的比例。",
        "- R@5/R@10：真实另一幅像排进前 5/10 的比例。",
        "- Top1%/Top5%/Top10%：真实目标进入全 catalog 前 1%、5%、10% 的比例。当前 test catalog 为 1800 个事件，因此 Top1% 约对应前 18 名。",
        "- median_true_rank：真实目标的中位排名，越小越好。",
        "",
        "## 5. 方法设置",
        "",
        "本实验比较以下方法：",
        "",
        "1. waveform_only：只用波形 encoder embedding 的相似度排序。",
        "2. time_only：只用观测触发时间差进行排序。",
        "3. true_sky_overlap_only：使用真实 ra/dec 计算 sky overlap，是理想上限实验，不代表真实可直接使用。",
        "4. predicted_sky_overlap_only：由波形 embedding 预测天空方向，再计算 predicted sky overlap。",
        "5. waveform_plus_time：用 hard-negative 样本训练二阶段 reranker，输入波形相似度和时间特征。",
        "6. waveform_plus_predicted_sky_overlap：二阶段 reranker 输入波形和预测 sky overlap。",
        "7. waveform_plus_time_plus_predicted_sky_overlap：二阶段 reranker 输入波形、时间和预测 sky overlap。",
        "8. waveform_plus_true_sky_overlap / waveform_plus_time_plus_true_sky_overlap：包含真实天空信息的上限参考。",
        "",
        "需要特别注意：真实 sky overlap 在当前代码中来自生成参数中的真实天空位置，因此只用于上限分析。真实场景下 ra/dec 不能直接作为可用辅助参数；真实可行方向应是观测 sky map 或机器学习估计 sky map。",
        "",
        "## 6. Overall 结果",
        "",
        "### 6.1 Noisy",
        "",
        md_table(
            noisy_overall,
            ["data_mode", "variant", "stage"] + metric_cols,
            ["mode", "method", "stage", "R@1", "R@5", "R@10", "Top1%", "Top5%", "Top10%", "median rank"],
            nd=4,
        ),
        "",
        "### 6.2 Pure",
        "",
        md_table(
            pure_overall,
            ["data_mode", "variant", "stage"] + metric_cols,
            ["mode", "method", "stage", "R@1", "R@5", "R@10", "Top1%", "Top5%", "Top10%", "median rank"],
            nd=4,
        ),
        "",
        "## 7. SIS / PM 分解结果",
        "",
        "下表只列主要方法，便于比较 SIS 与 PM 的差异。",
        "",
        md_table(
            split_show,
            ["data_mode", "subset", "variant", "stage"] + metric_cols,
            ["mode", "subset", "method", "stage", "R@1", "R@5", "R@10", "Top1%", "Top5%", "Top10%", "median rank"],
            nd=4,
        ),
        "",
        "## 8. 最优方法摘要",
        "",
        md_table(
            best_show,
            ["data_mode", "subset", "variant", "stage", "r@1", "r@5", "r@10", "top_1pct", "median_true_rank"],
            ["mode", "subset", "best method", "stage", "R@1", "R@5", "R@10", "Top1%", "median rank"],
            nd=4,
        ),
        "",
        "## 9. 结果分析",
        "",
        "### 9.1 SNR 提高后，pure 波形检索明显改善",
        "",
        "pure 条件下，waveform_only 已经达到 R@1 = 0.9100、R@10 = 0.9742，说明当没有噪声干扰且 SNR 足够高时，当前 InceptionTime embedding 能够很好地捕捉双像波形相似性。相比之前低 SNR LIGO 数据，这说明模型本身并非完全无法处理 LIGO；主要困难来自 noisy 条件下的噪声扰动和双探测器波形质量。",
        "",
        "### 9.2 Noisy 波形仍然是瓶颈",
        "",
        "noisy 条件下，即便 SNR 已提升，waveform_only 只有 R@1 = 0.1575、R@10 = 0.3417。这比极低 SNR 数据有改善，但仍远低于 pure。这说明 LIGO noisy 的主要难点不只是总 SNR 低，还包括：",
        "",
        "- H1/L1 双探测器下噪声会改变局部波形形态。",
        "- A/B 两像之间放大率、相位/Morse index、时间平移等因素叠加后，embedding 学到的相似性仍不稳定。",
        "- full catalog 排序比 pair-level 或小候选池难得多，候选中有大量非配对事件。",
        "",
        "### 9.3 时间信息对 noisy 提升最大",
        "",
        "noisy 中 time_only 达到 R@10 = 0.7367，waveform_plus_time rerank 达到 R@10 = 0.7542、Top1% = 0.8117。说明在这批数据中，时间延迟是非常强的区分信息。PM 的 time_only 尤其强，R@10 = 1.0000；SIS time_only 相对较弱，R@10 = 0.4733。",
        "",
        "这也提示：如果论文实验使用时间信息，需要明确说明 trigger_time_obs 的生成方式和误差模型，否则时间特征可能显得过强。后续如果要贴近真实场景，应系统增加 trigger_time_obs 误差并做敏感性实验。",
        "",
        "### 9.4 predicted sky overlap 当前不可用",
        "",
        "predicted_sky_overlap_only 在 noisy 中 R@10 = 0.0083，pure 中 R@10 = 0.1650。它不但没有接近真实 sky overlap，上限也远低于 waveform 和 time。加入 predicted sky overlap 后，waveform_plus_predicted_sky_overlap 在 noisy 中 R@10 = 0.3250，低于 waveform_only 的 0.3417；waveform_plus_time_plus_predicted_sky_overlap 为 0.7483，也略低于 waveform_plus_time 的 0.7542。",
        "",
        "这说明当前 predicted sky overlap 质量不足，作为辅助特征会引入噪声。要让 sky-map 方向有效，需要继续提升单事件 sky map/sky overlap 估计质量，或改为 pair-level overlap 直接监督。",
        "",
        "### 9.5 true sky overlap 是理想上限，不能作为真实方案结果",
        "",
        "true_sky_overlap_only 在 pure/noisy 中均为满分，这是因为它使用生成时的真实天空位置，两个透镜像共享同一源天空位置，因此在当前 synthetic 数据里几乎直接给出答案。这个结果不能作为真实可用方案，只能说明：如果未来能获得足够准确的 sky localization/sky map overlap，天空信息理论上可以极大提升 catalog-level 检索。",
        "",
        "### 9.6 SIS 和 PM 差异",
        "",
        "在 noisy waveform_only 下，SIS R@10 = 0.3800，PM R@10 = 0.3033；SIS 略好于 PM。time_only 下，PM R@10 = 1.0000，而 SIS R@10 = 0.4733，说明当前 PM 数据的时间延迟分布与非配对候选更容易区分。",
        "",
        "这点需要进一步诊断：PM 在时间上更容易，可能与 PM 质量范围和时间延迟筛选有关；SIS 对时间更难，说明 SIS 的真实配对时间延迟和随机候选时间差重叠更大。",
        "",
        "## 10. 与研究问题的关系",
        "",
        "本实验支持以下判断：",
        "",
        "1. LIGO noisy 差的问题，低 SNR 是重要因素，但不是唯一因素。提高 SNR 后 pure 几乎解决，noisy 仍明显受限。",
        "2. 当前波形模型在高质量 pure 数据上足够强，但 noisy 下需要更强的抗噪训练、数据增强或模型结构。",
        "3. 时间信息在当前 synthetic catalog 中非常有用，是提升 noisy ranking 的主要因素。",
        "4. 当前 predicted sky overlap 方向仍未成熟，不应在论文中作为最终有效提升点；更适合作为待改进模块或消融实验。",
        "5. true sky overlap 结果只能作为 oracle upper bound，用来说明 sky localization 的潜在价值。",
        "",
        "## 11. 后续建议",
        "",
        "### 11.1 对 noisy waveform 的优化",
        "",
        "建议继续尝试：",
        "",
        "- 更强的噪声增强：训练中加入与 LIGO 噪声一致的扰动，而不是只做简单 aug_noise。",
        "- 双分支输入：分别编码 H1/L1，再做 detector-aware fusion。",
        "- 相位/时间对齐增强：对 A/B 两像进行 peak 对齐或 learned alignment。",
        "- hard negative waveform training：专门采样波形相似但非配对的负样本。",
        "- pair-level cross encoder：先用 embedding 召回 Top-K，再用更重的 pair 模型比较两条波形。",
        "",
        "### 11.2 对时间辅助参数的真实性检查",
        "",
        "当前 time_only 很强，后续应补做：",
        "",
        "- trigger_time_obs 误差从小到大的 sweep。",
        "- 不同 lens 模型的时间延迟分布对比。",
        "- 随机非透镜事件密度增加时，time-only 是否仍稳定。",
        "- 真实观测触发时间估计是否能达到当前误差水平。",
        "",
        "### 11.3 对 sky 方向的改进",
        "",
        "当前 predicted sky overlap 不可用，建议：",
        "",
        "- 先单独评估 sky 预测误差：角距离误差、overlap 误差、正负 pair score 分布。",
        "- 从单事件 sky map 预测改成 pair-level overlap 直接预测。",
        "- 使用 detector time delay、SNR ratio、相位差等真实可观测定位特征，但要注意真实场景可获得性。",
        "- 将 true sky overlap 作为 upper bound，predicted sky overlap 作为当前弱基线，不混淆二者。",
        "",
        "## 12. 文件清单",
        "",
        "关键文件：",
        "",
        f"- 总结果表：`{ROOT / 'fresh50_full_catalog_summary.csv'}`",
        f"- noisy 结果：`{ROOT / 'ligo_noisy_full_catalog/fresh50_full_catalog_summary.csv'}`",
        f"- pure 结果：`{ROOT / 'ligo_pure_full_catalog/fresh50_full_catalog_summary.csv'}`",
        f"- noisy encoder：`{ROOT / 'fresh_mixed_encoders/ligo_noisy_mixed_sis_pm_ep50/model.pt'}`",
        f"- pure encoder：`{ROOT / 'fresh_mixed_encoders/ligo_pure_mixed_sis_pm_ep50/model.pt'}`",
        f"- SNR 汇总：`{DATA_ROOT / 'snr_summary_from_npy.csv'}`",
        f"- SNR 图目录：`{DATA_ROOT / 'snr_distribution_plot'}`",
        "",
        "## 13. 总结",
        "",
        "这次 2000 样本 LIGO ET-like SNR 小测试表明：提高 SNR 后，pure 波形检索已经非常好，但 noisy 波形检索仍然明显受限。时间信息能把 noisy R@10 从 0.3417 提升到约 0.7542，是当前最有效的真实可观测辅助方向之一。预测 sky overlap 目前质量不足，暂时不能作为有效提升来源；真实 sky overlap 的满分结果只能作为理论上限，说明准确 sky localization 对 catalog-level 检索有潜在价值。",
    ]

    report = ROOT / "ligo_et_snr_like_2000_full_catalog_experiment_report_cn.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
