#!/usr/bin/env python3
"""Create figures, bilingual reports, and a compact v6 deliverable archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/real_noise_injection_v6_unified_physics_20260721"
DEFAULT_PE = DEFAULT_ROOT / "pe_followup"
DEFAULT_PACKAGE = REPO / "packages/real_noise_injection_v6_unified_physics_20260721_deliverables.tar.gz"
DEPLOYMENTS = {"gwtc3": "GWTC-3 (O1--O3)", "gwtc4": "GWTC-4.1 (O4a)"}
DISPLAY_METHODS = {
    "waveform_only": "Waveform",
    "time_sky_candidate_selected": "Time + sky",
    "candidate_three_channel_unconstrained": "Waveform + time + sky",
    "candidate_three_channel_strictly_positive": "Three-channel, positive weights",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results(root: Path, pe_root: Path, key: str) -> dict[str, Any]:
    retrieval = pd.read_csv(root / key / "heldout_test_retrieval_metrics_per_seed_v6.csv")
    pair = pd.read_csv(root / key / "heldout_test_pair_level_metrics_per_seed_v6.csv")
    deployment = json.loads((root / key / "deployment_summary_v6.json").read_text(encoding="utf-8"))
    enrichment = pd.read_csv(pe_root / f"{key}_pe_consistency_rank_enrichment.csv")
    top = pd.read_csv(pe_root / f"{key}_consensus_top50_pe_followup.csv")
    followup = pd.read_csv(pe_root / f"{key}_pe_consistent_followup_shortlist.csv")
    seed_summaries = [
        json.loads((root / key / f"seed_{seed}/seed_summary_v6.json").read_text(encoding="utf-8"))
        for seed in deployment["seeds"]
    ]
    return {
        "retrieval": retrieval,
        "pair": pair,
        "deployment": deployment,
        "enrichment": enrichment,
        "top": top,
        "followup": followup,
        "seed_summaries": seed_summaries,
    }


def metric_summary(frame: pd.DataFrame, method: str, column: str, subset: str = "overall") -> dict[str, float]:
    keep = frame[(frame["method"] == method) & (frame["subset"] == subset)]
    values = keep[column].to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def pair_summary(frame: pd.DataFrame, method: str, column: str) -> dict[str, float]:
    values = frame.loc[frame["method"] == method, column].to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def load_et3_refusion(root: Path) -> dict[str, Any]:
    et_root = root / "et3_unified_sky_bf_refusion"
    summary_path = et_root / "et3_sky_bf_refusion_summary.csv"
    if not summary_path.exists():
        return {"status": "not_run", "path": str(et_root)}
    frame = pd.read_csv(summary_path)
    overall = frame[frame["subset"] == "overall"].copy()
    methods: dict[str, Any] = {}
    for row in overall.itertuples(index=False):
        methods[str(row.method)] = {
            "r_at_1_mean": float(row.r_at_1_mean),
            "r_at_1_std": float(row.r_at_1_std),
            "r_at_10_mean": float(row.r_at_10_mean),
            "r_at_10_std": float(row.r_at_10_std),
            "median_rank_mean": float(row.median_rank_mean),
            "n_sky_realizations": int(row.n_sky_realizations),
        }
    config = json.loads(
        (et_root / "et3_sky_bf_refusion_run_config.json").read_text(encoding="utf-8")
    )
    weights = json.loads(
        (et_root / "et3_sky_bf_refusion_weights.json").read_text(encoding="utf-8")
    )
    return {
        "status": "complete",
        "path": str(et_root),
        "methods": methods,
        "weights_by_sky_seed": weights,
        "config": config,
    }


def make_summary(root: Path, pe_root: Path) -> dict[str, Any]:
    pe_summary = json.loads((pe_root / "pe_followup_summary_v6.json").read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "status": "complete",
        "detection_claim": False,
        "real_pe_used_for_triage_tuning": False,
        "deployments": {},
    }
    for key, label in DEPLOYMENTS.items():
        data = load_results(root, pe_root, key)
        item: dict[str, Any] = {
            "label": label,
            "waveform_primary_seed_count": data["deployment"]["waveform_primary_seed_count"],
            "primary_methods_by_seed": data["deployment"]["primary_methods"],
            "consensus_primary_policy": pe_summary[key]["primary_consensus_policy"],
            "strict_pairs": pe_summary[key]["strict_pairs"],
            "top10_pe_consistent": pe_summary[key]["top10_3sigma_pass"],
            "top10_pe_enrichment": pe_summary[key]["top10_pe_enrichment"],
            "pe_followup_count": pe_summary[key]["consensus_top50_pe_followup_count"],
            "top_pe_followup_candidates": pe_summary[key]["top_pe_followup_candidates"],
            "top10_triage_pairs": pe_summary[key]["consensus_top10_triage_pairs"],
            "gw170104_gw170814": pe_summary[key]["gw170104_gw170814"],
            "external_candidate_crosscheck": pe_summary[key]["external_candidate_crosscheck"],
            "event_waveform_prediction_pe_audit": pe_summary[key]["event_waveform_prediction_pe_audit"],
            "forced_three_channel_sensitivity": pe_summary[key][
                "forced_three_channel_sensitivity"
            ],
            "candidate_weights_by_seed": {
                str(seed_summary["seed"]): seed_summary["weights"]["candidate_three_channel_unconstrained"]
                for seed_summary in data["seed_summaries"]
            },
            "candidate_positive_weights_by_seed": {
                str(seed_summary["seed"]): seed_summary["weights"]["candidate_three_channel_strictly_positive"]
                for seed_summary in data["seed_summaries"]
            },
            "time_sky_candidate_weights_by_seed": {
                str(seed_summary["seed"]): seed_summary["weights"]["time_sky_candidate_selected"]
                for seed_summary in data["seed_summaries"]
            },
            "methods": {},
        }
        for method in DISPLAY_METHODS:
            if method not in set(data["retrieval"]["method"]):
                continue
            item["methods"][method] = {
                "r_at_1": metric_summary(data["retrieval"], method, "r_at_1"),
                "r_at_10": metric_summary(data["retrieval"], method, "r_at_10"),
                "average_precision": pair_summary(data["pair"], method, "average_precision"),
                "false_at_recall_0p5": pair_summary(data["pair"], method, "false_at_recall_0p5"),
                "precision_at_recall_0p5": pair_summary(data["pair"], method, "precision_at_recall_0p5"),
            }
        output["deployments"][key] = item
    output["et3_sky_bayes_factor_refusion"] = load_et3_refusion(root)
    return output


def plot_summary(root: Path, pe_root: Path, figure_dir: Path) -> None:
    set_style()
    colors = ["#277DA1", "#F8961E", "#43AA8B", "#6A4C93"]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.1), constrained_layout=True)
    for row, (key, label) in enumerate(DEPLOYMENTS.items()):
        data = load_results(root, pe_root, key)
        methods = [m for m in DISPLAY_METHODS if m in set(data["retrieval"]["method"])]
        x = np.arange(len(methods))
        for col, metric in ((0, "r_at_10"), (1, "average_precision")):
            ax = axes[row, col]
            source = data["retrieval"] if col == 0 else data["pair"]
            for pos, (method, color) in enumerate(zip(methods, colors)):
                if col == 0:
                    values = source[(source.method == method) & (source.subset == "overall")][metric].to_numpy(float)
                else:
                    values = source[source.method == method][metric].to_numpy(float)
                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(np.full(len(values), pos) + jitter, values, color=color, s=24, alpha=0.8, zorder=3)
                ax.errorbar(pos, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0, fmt="_", color="black", ms=13, capsize=3, lw=0.9)
            ax.set_xticks(x, [DISPLAY_METHODS[m] for m in methods], rotation=25, ha="right")
            ax.set_ylabel("Companion R@10" if col == 0 else "Unordered-pair AUPRC")
            ax.set_title(f"{label}: {'retrieval' if col == 0 else 'pair discrimination'}")
            ax.grid(axis="y", alpha=0.18)
            if col == 1:
                ax.set_yscale("log")

        ax = axes[row, 2]
        top = data["top"].sort_values("rank").head(10)
        y = np.arange(len(top))
        ax.barh(y, top["final_score"], color="#A8DADC", edgecolor="#457B9D", lw=0.5)
        passed = top["pe_followup_screen_pass"].fillna(False).to_numpy(dtype=bool)
        ax.scatter(top.loc[passed, "final_score"], y[passed], marker="*", s=65, color="#D62828", label="Passes PE screen", zorder=4)
        ax.set_yticks(y, [f"#{int(rank)}" for rank in top["rank"]])
        ax.invert_yaxis()
        ax.set_xlabel("Frozen consensus triage score")
        ax.set_ylabel("Retrieval rank")
        ax.set_title(f"{label}: real-catalog top pairs")
        ax.axvline(0, color="black", lw=0.6)
        if np.any(passed):
            ax.legend(loc="best", fontsize=7)
    for label, axis in zip("abcdef", axes.flat):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    fig.suptitle(
        "Corrected run-matched validation and real-catalog candidate audit",
        fontweight="bold",
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "fig_real_noise_v6_validation_and_search"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def report_cn(summary: dict[str, Any]) -> str:
    lines = [
        "# GWTC-3 / GWTC-4.1 真实噪声三通道部署 v6",
        "",
        "## 结论边界",
        "",
        "本实验输出的是供 coherent Bayesian analysis 使用的 candidate shortlist，不是透镜探测。候选通道限定为 strain-derived waveform evidence、按 observing-run exposure 条件化的 time-delay likelihood ratio 和真实 HEALPix sky Bayes factor；预定 Gate 可在 waveform 部署失败时把主结果回退到 time+sky，并始终另报严格正权重三通道敏感性结果。SNR 不进入最终分数，真实 PE 不参与训练、校准、融合权重或主方法选择。",
        "",
        "旧 v3/v4 waveform 分支把已 whiten/z-score 的数组当作物理 strain，并存在单探测器静默零填充等问题，已经标记为 superseded；本报告只汇报 v6。",
        "结果表为兼容旧 pipeline 保留了 `sis`/`pm` 字段名；在 v6 GW-LMC source bank 中它们分别表示 smooth/non-subhalo 与 subhalo-present proposal group，不等同于解析 SIS 与 point-mass 透镜模型。",
        "",
    ]
    for key, item in summary["deployments"].items():
        lines.extend([f"## {item['label']}", ""])
        for method, metrics in item["methods"].items():
            lines.append(
                f"- {DISPLAY_METHODS[method]}：R@1={metrics['r_at_1']['mean']:.3f}±{metrics['r_at_1']['std']:.3f}，"
                f"R@10={metrics['r_at_10']['mean']:.3f}±{metrics['r_at_10']['std']:.3f}，"
                f"AUPRC={metrics['average_precision']['mean']:.4g}±{metrics['average_precision']['std']:.3g}。"
            )
        lines.extend(
            [
                f"- 三个 seed 中 waveform 可进入主部署的 seed 数：{item['waveform_primary_seed_count']} / 3。",
                f"- 冻结 consensus 主方法：{item['consensus_primary_policy']['method']}；原因：{item['consensus_primary_policy']['reason']}",
                f"- 严格 H1/L1、非 OOD BBH pair 数：{item['strict_pairs']}。",
                f"- consensus top-10 中通过一维 3-sigma PE screen：{item['top10_pe_consistent']} / 10；相对全目录 enrichment={item['top10_pe_enrichment']:.3f}。",
                f"- consensus top-50 中通过预定 PE follow-up screen：{item['pe_followup_count']} 对。",
                f"- 严格正权重三通道补充排名：top-10 PE enrichment={item['forced_three_channel_sensitivity']['top10_pe_enrichment']:.3f}，top-50 通过 PE screen={item['forced_three_channel_sensitivity']['consensus_top50_pe_followup_count']} 对。",
                "",
            ]
        )
        lines.append("冻结 consensus triage top-5：")
        lines.append("- 共识顺序按三个 seed 的 pair rank 中位数及预定 tie-break 排定；下列 score 为 seed 间中位数，因此不要求随共识 rank 单调下降。")
        lines.append(
            "- 实际主 time+sky 权重（waveform/time/sky）："
            + "；".join(
                f"{seed}={weights['waveform']:g}/{weights['time']:g}/{weights['sky']:g}"
                for seed, weights in item["time_sky_candidate_weights_by_seed"].items()
            )
            + "。"
        )
        lines.append(
            "- validation-selected unconstrained 三通道敏感性权重（waveform/time/sky）："
            + "；".join(
                f"{seed}={weights['waveform']:g}/{weights['time']:g}/{weights['sky']:g}"
                for seed, weights in item["candidate_weights_by_seed"].items()
            )
            + "。"
        )
        lines.append(
            "- validation-selected strictly-positive 三通道敏感性权重（waveform/time/sky）："
            + "；".join(
                f"{seed}={weights['waveform']:g}/{weights['time']:g}/{weights['sky']:g}"
                for seed, weights in item["candidate_positive_weights_by_seed"].items()
            )
            + "。"
        )
        for pair in item["top10_triage_pairs"][:5]:
            lines.append(
                f"- #{int(pair['rank'])} {pair['event_i']} -- {pair['event_j']}；"
                f"score={pair['final_score']:.3f}，waveform/time/sky="
                f"{pair['waveform_score']:.3f}/{pair['time_score']:.3f}/{pair['sky_score']:.3f}。"
            )
        if key == "gwtc3":
            historical = item["gw170104_gw170814"]
            lines.append(
                f"- GW170104--GW170814 consensus rank：{int(historical['rank']) if historical.get('found') else '不在严格主目录'}。"
            )
        lines.append("")
    et3 = summary.get("et3_sky_bayes_factor_refusion", {})
    if et3.get("status") == "complete":
        lines.extend(
            [
                "## ET-3 统一 sky Bayes-factor 重融合",
                "",
                "该部分复用既有 ET-3 encoder 和固定 strain test catalog，不重训波形模型；waveform cosine、GW-LMC time-delay LR 和 observed-sky Gaussian posterior 的绝对 common-source Bayes factor 均在 validation 上校准/选权重，test catalog 不做 row-wise z-score。天空仍是模拟 posterior proxy，不是完整 ET 参数估计。",
                "",
            ]
        )
        for method in (
            "waveform_only",
            "time_sky_validation_selected",
            "three_channel_validation_selected",
            "three_channel_strictly_positive",
        ):
            if method not in et3["methods"]:
                continue
            metric = et3["methods"][method]
            lines.append(
                f"- {method}：R@1={metric['r_at_1_mean']:.3f}±{metric['r_at_1_std']:.3f}，"
                f"R@10={metric['r_at_10_mean']:.3f}±{metric['r_at_10_std']:.3f}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释",
            "",
            "三通道 retrieval rank 与 PE-confirmed follow-up rank 是两个不同产物。前者用于廉价 triage，后者要求 detector-frame chirp mass、mass ratio 和 effective spin 的后验相容，并通过近似 prior-corrected 共享源 evidence。若 top triage 没有 PE enrichment，结论是没有可信 follow-up candidate，不能用真实 PE 反向调权。",
            "",
            "详细公式、数据 split、sky/time 构造、Gate、文献依据与实现限制见 `docs/methods/real_noise_v6_unified_physics_protocol_cn.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def report_en(summary: dict[str, Any]) -> str:
    lines = [
        "# GWTC-3 / GWTC-4.1 real-noise three-channel deployment v6",
        "",
        "This experiment produces a candidate shortlist for coherent Bayesian follow-up, not a lensing detection. Candidate evidence is restricted to strain-derived waveform evidence, a run-exposure-conditioned time-delay likelihood ratio, and the real HEALPix sky Bayes factor. A predeclared Gate falls back to time+sky if real waveform deployment fails, while a strictly-positive three-channel sensitivity ranking is always reported. SNR is audit-only, and real-event PE is not used for training, tuning, or primary-method selection.",
        "For compatibility with the legacy pipeline, some tables retain the internal labels `sis` and `pm`; in the v6 GW-LMC source bank these denote the smooth/non-subhalo and subhalo-present proposal groups, not analytic SIS and point-mass lens models.",
        "",
    ]
    for item in summary["deployments"].values():
        lines.extend([f"## {item['label']}", ""])
        for method, metrics in item["methods"].items():
            lines.append(
                f"- {DISPLAY_METHODS[method]}: R@1={metrics['r_at_1']['mean']:.3f}+/-{metrics['r_at_1']['std']:.3f}, "
                f"R@10={metrics['r_at_10']['mean']:.3f}+/-{metrics['r_at_10']['std']:.3f}, "
                f"AUPRC={metrics['average_precision']['mean']:.4g}+/-{metrics['average_precision']['std']:.3g}."
            )
        lines.extend(
            [
                f"- Seeds eligible for waveform-inclusive primary deployment: {item['waveform_primary_seed_count']} / 3.",
                f"- Frozen consensus primary method: {item['consensus_primary_policy']['method']}; reason: {item['consensus_primary_policy']['reason']}",
                f"- Strict complete-H1/L1 non-OOD BBH pairs: {item['strict_pairs']}.",
                f"- Consensus top-10 passing the one-dimensional 3-sigma PE screen: {item['top10_pe_consistent']} / 10; enrichment={item['top10_pe_enrichment']:.3f}.",
                f"- Consensus top-50 pairs passing the predeclared PE follow-up screen: {item['pe_followup_count']}.",
                f"- Strictly-positive three-channel sensitivity: top-10 PE enrichment={item['forced_three_channel_sensitivity']['top10_pe_enrichment']:.3f}; PE-screened pairs in top 50={item['forced_three_channel_sensitivity']['consensus_top50_pe_followup_count']}.",
                "",
            ]
        )
        lines.append("Frozen consensus triage top five:")
        lines.append("- Consensus order is defined by the median pair rank across the three seeds and predeclared tie-breaks; the displayed score is the across-seed median and therefore need not decrease monotonically with consensus rank.")
        lines.append(
            "- Primary time+sky weights (waveform/time/sky): "
            + "; ".join(
                f"{seed}={weights['waveform']:g}/{weights['time']:g}/{weights['sky']:g}"
                for seed, weights in item["time_sky_candidate_weights_by_seed"].items()
            )
            + "."
        )
        lines.append(
            "- Validation-selected unconstrained three-channel sensitivity weights (waveform/time/sky): "
            + "; ".join(
                f"{seed}={weights['waveform']:g}/{weights['time']:g}/{weights['sky']:g}"
                for seed, weights in item["candidate_weights_by_seed"].items()
            )
            + "."
        )
        lines.append(
            "- Validation-selected strictly-positive three-channel sensitivity weights (waveform/time/sky): "
            + "; ".join(
                f"{seed}={weights['waveform']:g}/{weights['time']:g}/{weights['sky']:g}"
                for seed, weights in item["candidate_positive_weights_by_seed"].items()
            )
            + "."
        )
        for pair in item["top10_triage_pairs"][:5]:
            lines.append(
                f"- #{int(pair['rank'])} {pair['event_i']} -- {pair['event_j']}; "
                f"score={pair['final_score']:.3f}, waveform/time/sky="
                f"{pair['waveform_score']:.3f}/{pair['time_score']:.3f}/{pair['sky_score']:.3f}."
            )
        lines.append("")
    et3 = summary.get("et3_sky_bayes_factor_refusion", {})
    if et3.get("status") == "complete":
        lines.extend(
            [
                "## ET-3 sky-Bayes-factor refusion",
                "",
                "This analysis reuses the existing ET-3 encoder and fixed strain test catalog. Waveform similarity, the GW-LMC time-delay likelihood ratio, and the absolute common-source Bayes factor of the observed-sky Gaussian posterior proxy are calibrated or weighted on validation data only; no test-catalog row standardization is used. The sky input remains a simulated posterior proxy rather than full ET parameter estimation.",
                "",
            ]
        )
        for method in (
            "waveform_only",
            "time_sky_validation_selected",
            "three_channel_validation_selected",
            "three_channel_strictly_positive",
        ):
            if method not in et3["methods"]:
                continue
            metric = et3["methods"][method]
            lines.append(
                f"- {method}: R@1={metric['r_at_1_mean']:.3f}+/-{metric['r_at_1_std']:.3f}, "
                f"R@10={metric['r_at_10_mean']:.3f}+/-{metric['r_at_10_std']:.3f}."
            )
        lines.append("")
    lines.extend(
        [
            "Retrieval rank and PE-screened follow-up rank are separate outputs. A lack of PE enrichment means that no physically credible follow-up candidate was produced; real PE is never used to retune the frozen triage score.",
            "",
        ]
    )
    return "\n".join(lines)


def make_reproduce(root: Path) -> Path:
    path = root / "reproduce_v6.sh"
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=results/real_noise_injection_v6_unified_physics_20260721
SELECTION=$ROOT/development/pilot_selection_final/pilot_selection_validation_only.json

echo "Prerequisites: GWOSC strain/PE files, audited v5 physical H1/L1 source banks, and the frozen validation-only development-selection JSON. These large inputs are external to the compact archive; their manifests and checksums are packaged."

read -r AUX QLOSS < <(python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("results/real_noise_injection_v6_unified_physics_20260721/development/pilot_selection_final/pilot_selection_validation_only.json").read_text())
print(p["selected_aux_weight"], p["selected_q_loss_weight"])
PY
)

python scripts/real_search/42_prebuild_v6_training_data.py \
  --deployments GWTC3 GWTC4 --seeds 202607231 202607232 202607233 \
  --samples 600 --variants 8

python scripts/real_search/38_real_noise_injection_v6_unified_physics.py \
  --aux-weight "$AUX" --q-loss-weight "$QLOSS" \
  --deployments GWTC3 GWTC4 --seeds 202607231 202607232 202607233 \
  --samples-per-group 600 --batch-size 48 --bootstrap-draws 10000 \
  --development-selection "$SELECTION"

python scripts/experiments/19_validate_physical_scoring_invariants.py --result-root "$ROOT"
python scripts/real_search/39_physical_deployment_v6_pe_audit.py \
  --input-root "$ROOT" --out "$ROOT/pe_followup" \
  --seeds 202607231 202607232 202607233
python scripts/real_search/44_et3_sky_bayes_factor_refusion.py \
  --out "$ROOT/et3_unified_sky_bf_refusion" \
  --sky-seeds 202607081 202607082 202607083 --bootstrap-draws 10000
python scripts/real_search/43_make_v6_deliverables.py \
  --root "$ROOT" --pe-root "$ROOT/pe_followup" \
  --package packages/real_noise_injection_v6_unified_physics_20260721_deliverables.tar.gz
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def write_environment_manifest(root: Path) -> Path:
    """Record the software/hardware context without making it a requirement file."""
    commands = {
        "python": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "nvidia_smi": ["nvidia-smi"],
    }
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "commands": {},
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            payload["commands"][name] = {
                "returncode": int(result.returncode),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except OSError as exc:
            payload["commands"][name] = {"error": str(exc)}
    path = root / "environment_manifest_v6.json"
    write_json(path, payload)
    return path


def write_readme(root: Path) -> Path:
    path = root / "README_CURRENT_RESULTS.md"
    path.write_text(
        """# Authoritative v6 real-noise deployment results

Use only the files under `results/real_noise_injection_v6_unified_physics_20260721` for the corrected GWTC-3/O3 and GWTC-4.1/O4a deployment. The earlier v3/v4 waveform deployments are superseded because their source-unit and missing-detector handling did not pass the physical audit.

Primary interpretation:

- triage channels: calibrated waveform evidence, run-conditioned time-delay likelihood ratio, and HEALPix common-source sky Bayes factor;
- strict main catalog: complete H1/L1, in-domain BBH events;
- real PE is an independent post-ranking consistency screen, never a tuning input;
- outputs are candidates for coherent Bayesian follow-up, not lensing detections.

Entry points:

- `run_config_v6.json`: frozen run configuration and development-selection record;
- `gwtc3/` and `gwtc4/`: per-seed held-out metrics, rankings, weights, and deployment audits;
- `pe_followup/`: catalog-wide PE diagnostics and PE-consistent follow-up shortlists;
- `et3_unified_sky_bf_refusion/`: ET-3 validation-frozen refusion with absolute observed-sky Bayes-factor evidence;
- `final_summary_v6.json`: compact cross-run result summary;
- `real_noise_v6_report_cn.md` and `real_noise_v6_report_en.md`: bilingual reports;
- `docs/methods/real_noise_v6_unified_physics_protocol_cn.md`: formulas, references, and experimental boundaries;
- `artifact_manifest_v6.csv`: checksums for compact deliverables.
""",
        encoding="utf-8",
    )
    return path


def write_artifact_manifest(root: Path) -> Path:
    path = root / "artifact_manifest_v6.csv"
    rows = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item == path:
            continue
        if any(part in {"matchroots", "waveform_gate"} for part in item.parts):
            continue
        if item.suffix in {".npy", ".pt", ".h5", ".hdf5"}:
            continue
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "relative_path": str(item.relative_to(root)),
                "size_bytes": int(item.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def package(root: Path, pe_root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    include_scripts = list((REPO / "scripts/real_search").glob("3[4-9]_*.py")) + list(
        (REPO / "scripts/real_search").glob("4[0-4]_*.py")
    ) + [
        REPO / "scripts/real_search/unified_v6_common.py",
        REPO / "scripts/real_search/physical_common.py",
        REPO / "scripts/real_search/physical_source_v5_common.py",
        REPO / "scripts/real_search/common.py",
        REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py",
        REPO / "scripts/experiments/19_validate_physical_scoring_invariants.py",
        REPO / "scripts/experiments/26_spectrogram_encoder_pilot.py",
        REPO / "scripts/experiments/31_real_noise_injection_v4_formal.py",
        REPO / "scripts/experiments/32_physical_deployment_v4_pe_audit.py",
        REPO / "scripts/experiments/88_liao_realistic_p1_p2_rerank.py",
        REPO / "scripts/experiments/102_unified_sky_formal_pipeline.py",
        REPO / "scripts/experiments/105_et3_sky_localization_sensitivity.py",
        REPO / "scripts/experiments/mainline_uncertainty_common.py",
        REPO / "scripts/sky/sky_posterior_overlap.py",
        REPO / "docs/methods/real_noise_v6_unified_physics_protocol_cn.md",
        REPO / "docs/methods/real_noise_v6_design_reference_matrix_cn.md",
    ]
    include_scripts.extend((REPO / "matchgw").glob("*.py"))
    with tarfile.open(package_path, "w:gz") as archive:
        for path in include_scripts:
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(REPO)))
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in {"matchroots", "waveform_gate"} for part in path.parts):
                continue
            if path.suffix in {".npy", ".pt", ".h5", ".hdf5"}:
                continue
            archive.add(path, arcname=str(path.relative_to(REPO)))
        for source_run in (
            REPO / "runs/real_gwtc_lensing_search_20260625",
            REPO / "runs/real_gwtc34_lensing_search_20260629_full_o4",
        ):
            for relative in ("data/event_manifest.csv", "data/strain_gwosc_download_manifest.csv"):
                path = source_run / relative
                if path.exists():
                    archive.add(path, arcname=f"source_manifests/{source_run.name}/{relative}")
        v5_root = REPO / "results/real_noise_injection_v5_physical_source_20260721"
        shared_names = {
            "physical_source_bank_summary.json",
            "physical_source_units_audit.json",
            "h1l1_live_schedule.csv",
            "noise_bank_manifest.csv",
            "time_delay_likelihood_ratio.json",
            "time_delay_prior_source_audit_v6.json",
            "time_delay_bandwidth_sensitivity_v6.json",
            "real_pe_sky_template_manifest.csv",
            "real_pe_sky_template_summary.json",
        }
        for deployment in ("gwtc3", "gwtc4"):
            shared = v5_root / deployment / "shared"
            for path in shared.rglob("*"):
                if not path.is_file():
                    continue
                if path.name in shared_names or path.name in {
                    "physical_source_pair_metadata.parquet",
                    "physical_unlensed_source_metadata.parquet",
                }:
                    archive.add(path, arcname=f"source_audits/{deployment}/{path.relative_to(shared)}")
    digest = hashlib.sha256()
    with package_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    package_path.with_suffix(package_path.suffix + ".sha256").write_text(
        f"{digest.hexdigest()}  {package_path.name}\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pe-root", type=Path, default=DEFAULT_PE)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.pe_root = args.pe_root.resolve()
    args.package = args.package.resolve()
    summary = make_summary(args.root, args.pe_root)
    write_json(args.root / "final_summary_v6.json", summary)
    plot_summary(args.root, args.pe_root, args.root / "figures")
    (args.root / "real_noise_v6_report_cn.md").write_text(report_cn(summary), encoding="utf-8")
    (args.root / "real_noise_v6_report_en.md").write_text(report_en(summary), encoding="utf-8")
    make_reproduce(args.root)
    write_environment_manifest(args.root)
    write_readme(args.root)
    write_artifact_manifest(args.root)
    package(args.root, args.pe_root, args.package)
    print(json.dumps({"status": "complete", "package": str(args.package)}, indent=2))


if __name__ == "__main__":
    main()
