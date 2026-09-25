#!/usr/bin/env python3
"""Human-readable final/failed-configuration audits without changing scores."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_fresh_confirmation_20260906 as fresh


def run(root):
    code = "MAIN-O3-MCWF-ENCODER-v2-RUNSPEC"
    lines = ["# 最终共识Top10的PE与官方阶段", "",
        "本表只是冻结后的检索/PE审计，不是真实透镜标签。公开Hanabi表重合表示曾被分析，不表示证实透镜；本轮没有运行Hanabi。",
        "O3真实PE已用于开发审计，不能称为盲测。所有单seed与Top20/50/100结果保存在CSV/Parquet中。", ""]
    for dep in ("gwtc3", "gwtc4"):
        for version in ("CFIX-baseline", code):
            for method in ("waveform_only", "C_fixed"):
                frame = pd.read_parquet(root / f"final_results/{dep}/{version}/{method}/all_pairs.parquet").head(10)
                columns = ["consensus_rank", "event_i", "event_j", "final_score_mean", "waveform_contribution_mean",
                    "time_contribution_mean", "sky_contribution_mean", "pe_mc_bhattacharyya_coefficient",
                    "pe_q_bhattacharyya_coefficient", "pe_chi_eff_bhattacharyya_coefficient",
                    "pe_dl_app_bhattacharyya_coefficient", "pe_dmax_intrinsic"]
                columns += [c for c in frame if c.startswith("official_") and ("fpp" in c or "stage" in c or "conclusion" in c or "resolved_hanabi" in c)]
                sub = frame[columns].rename(columns={"consensus_rank": "rank", "final_score_mean": "S_mean",
                    "waveform_contribution_mean": "wf", "time_contribution_mean": "time", "sky_contribution_mean": "sky",
                    "pe_mc_bhattacharyya_coefficient": "BC_Mc", "pe_q_bhattacharyya_coefficient": "BC_q",
                    "pe_chi_eff_bhattacharyya_coefficient": "BC_chi_eff", "pe_dl_app_bhattacharyya_coefficient": "BC_dL_app",
                    "pe_dmax_intrinsic": "Dmax"})
                lines += [f"## {dep} / {version} / {method}", "", sub.to_markdown(index=False, floatfmt=".4f"), ""]
    (root / "reports/FINAL_TOP10_PE_OFFICIAL_CN.md").write_text("\n".join(lines), encoding="utf-8")
    table = pd.read_csv(root / "tables/ALL_ABLATIONS_RECALL_PE_OFFICIAL_LEDGER.csv")
    lines = ["# 全部编码器消融的Recall、PE和官方重合", "",
        "21个实际训练模型；每种表示另报告NEW-EMBED、NEW-PHYSICAL、VAL-ENSEMBLE三种分数对照。以下旧注入指标属于重复使用的开发对照，不是新独立确认。PE/官方统计为三seed共识Top10。失败配置未删除。", ""]
    for (config, dep), group in table.groupby(["config", "deployment"]):
        fields = ["rule", "method", "macro_r_at_1_mean", "macro_r_at_10_mean", "macro_r_at_10_std",
            "average_precision_mean", "false_at_recall_0p5_mean", "false_at_recall_0p9_mean", "catastrophic_mc",
            "BC_mc_ge_0p5", "Dmax_le_3", "official_frontend", "official_hanabi"]
        lines += [f"## {config} / {dep}", "", group[fields].to_markdown(index=False, floatfmt=".4f"), ""]
    (root / "reports/ALL_ABLATIONS_RECALL_PE_OFFICIAL_CN.md").write_text("\n".join(lines), encoding="utf-8")
    chosen = fresh.verify_freeze(root)
    pe = pd.read_csv(root / "audit/gwtc3_event_PE_reference.csv")
    new, old, drift = [], [], []
    for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
        f = pd.read_csv(root / f"cache/predictions/{chosen['O3_config']}/gwtc3/model_{ms}_eval_{es}/real_event_mass_predictions.csv")
        f = f.loc[f.strict_h1l1_preprocessing_pass].merge(pe, on="event_name", validate="one_to_one")
        f["model_seed"] = ms
        f["predicted_over_PE_median"] = f.pred_mc/f.pe_chirp_mass_median
        f["absolute_log_error_vs_PE_median"] = np.abs(np.log(f.predicted_over_PE_median))
        new.append(f)
        b = pd.read_parquet(dev.MAIN / f"results/seed_{es}/event_embeddings.parquet").merge(pe, on="event_name", validate="one_to_one")
        b = b.loc[b.event_name.isin(f.event_name)]
        b["absolute_log_error_vs_PE_median"] = np.abs(np.log(b.waveform_pred_chirp_mass_detector/b.pe_chirp_mass_median))
        for label, values in (("old_head", b.absolute_log_error_vs_PE_median), ("new_head", f.absolute_log_error_vs_PE_median)):
            old.append({"model_seed": ms, "old_eval_seed": es, "variant": label, "events": len(values),
                "median_abs_log_error_vs_PE": values.median(), "q90_abs_log_error_vs_PE": values.quantile(.9),
                "factor2_or_more_errors": int((values >= np.log(2)).sum()), "not_true_parameter_error": True})
        for split in ("validation", "test", "real"):
            g = pd.read_parquet(root / f"evaluation/{chosen['O3_config']}/gwtc3/model_{ms}_eval_{es}/{split}_VAL-ENSEMBLE_pairs.parquet")
            drift.append({"split": split, "model_seed": ms, "pairs": len(g),
                "waveform_calibration_OOD_fraction": g.waveform_calibration_ood.mean(),
                "waveform_score_q99": g.waveform_score.quantile(.99), "waveform_score_max": g.waveform_score.max()})
        for cs in fresh.CATALOG_SEEDS:
            g = pd.read_parquet(fresh.data_root(root) / f"gwtc3/catalog_{cs}/model_{ms}/CANDIDATE_pairs.parquet")
            drift.append({"split": f"fresh_{cs}", "model_seed": ms, "pairs": len(g),
                "waveform_calibration_OOD_fraction": g.new_waveform_OOD.mean(),
                "waveform_score_q99": g.waveform_score.quantile(.99), "waveform_score_max": g.waveform_score.max()})
    dev.csv_write(root / "tables/FINAL_REAL_EVENT_Mc_AUDIT.csv", pd.concat(new, ignore_index=True))
    dev.csv_write(root / "tables/FINAL_REAL_EVENT_Mc_SUMMARY.csv", pd.DataFrame(old))
    dev.csv_write(root / "tables/FINAL_WAVEFORM_OOD_AUDIT.csv", pd.DataFrame(drift))
    text = "\n".join(["# 交付索引", "", f"代号：`{code}`。", "",
        "- 总报告：`reports/FINAL_ENCODER_EXPERIMENT_REPORT_CN.md`",
        "- 新旧Top10的PE、贡献和官方阶段：`reports/FINAL_TOP10_PE_OFFICIAL_CN.md`",
        "- 全部消融：`reports/ALL_ABLATIONS_RECALL_PE_OFFICIAL_CN.md`",
        "- 新噪声确认：`confirmation_global_noise_fixed/`，不是修复前的`confirmation/`。",
        "- 全pair、Top100、全部逐seed排名：`final_results/`、`results/`。",
        "- 模型与校准：`models/`、`evaluation/`。",
        "- 机器可读结论：`contracts/DELIVERY_STATUS.json`。",
        "- 脚本/依赖/原始路径映射：`scripts/`、`environment/`、`reproduction_inputs/PATH_MAPPING.csv`。",
        "- 原始GWOSC strain、PE HDF5、旧训练库和全部dense天空图不在紧凑包；仍在服务器原路径，未删除。",
        "- 包内文件哈希：`manifest/DELIVERY_SHA256.csv`；压缩包哈希另附。",
        "", "本轮只提交作者审核，不写论文、不覆盖历史结果，不把官方重合当作透镜确认。", ""])
    (root / "README_CN.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
