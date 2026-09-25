from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results" / "real_noise_injection_v4_physical_20260721"
DEFAULT_PACKAGE = REPO / "packages" / "real_noise_injection_v4_physical_20260721_deliverables.tar.gz"
SEEDS = (202607221, 202607222, 202607223)
DEPLOYMENTS = {"gwtc3": "GWTC-3 (O1-O3)", "gwtc4": "GWTC-4.1 O4a"}
METHODS = {
    "waveform_only": "Waveform only",
    "time_sky_validation_selected": "Time + sky",
    "three_channel_unconstrained": "Three-channel (free)",
    "three_channel_strictly_positive": "Three-channel (>0)",
    "three_channel_equal_evidence": "Three-channel (1:1:1)",
}
COLORS = {
    "waveform_only": "#277DA1",
    "time_sky_validation_selected": "#F8961E",
    "three_channel_unconstrained": "#43AA8B",
    "three_channel_strictly_positive": "#6A4C93",
    "three_channel_equal_evidence": "#577590",
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def seed_summary(root: Path, deployment: str, seed: int) -> dict[str, Any]:
    return json.loads((root / deployment / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))


def metric_value(frame: pd.DataFrame, method: str, subset: str, metric: str) -> tuple[float, float]:
    values = frame[(frame.method == method) & (frame.subset == subset)][metric].to_numpy(dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def make_main_figure(root: Path, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    selected_methods = list(METHODS)
    for row, (deployment, label) in enumerate(DEPLOYMENTS.items()):
        retrieval = pd.read_csv(root / deployment / "heldout_test_retrieval_metrics_per_seed.csv")
        pair = pd.read_csv(root / deployment / "heldout_test_pair_level_metrics_per_seed.csv")
        weights = pd.read_csv(root / deployment / "fusion_weights_per_seed.csv")
        ax = axes[row, 0]
        x = np.arange(len(selected_methods))
        width = 0.36
        for offset, metric in ((-width / 2, "r_at_1"), (width / 2, "r_at_10")):
            means, stds = zip(*(metric_value(retrieval, method, "overall", metric) for method in selected_methods))
            ax.bar(x + offset, means, width, yerr=stds, capsize=2, alpha=0.82, label=metric.replace("r_at_", "R@"))
            for method_index, method in enumerate(selected_methods):
                points = retrieval[(retrieval.method == method) & (retrieval.subset == "overall")]
                ax.scatter(np.full(len(points), x[method_index] + offset), points[metric], s=14, color="black", zorder=3)
        ax.set_ylim(0, 1.06)
        ax.set_ylabel("Held-out companion recall")
        ax.set_xticks(x, [METHODS[m].replace("Three-channel", "3-ch.") for m in selected_methods], rotation=25, ha="right")
        ax.set_title(f"{label}: retrieval")
        if row == 0:
            ax.legend(ncol=2, loc="lower right")

        ax = axes[row, 1]
        ap = pair[pair.metric == "average_precision"]
        for method_index, method in enumerate(selected_methods):
            values = ap[ap.method == method]["value"].to_numpy(dtype=float)
            ax.scatter(np.full(len(values), method_index), values, color=COLORS[method], s=22)
            ax.plot([method_index - 0.18, method_index + 0.18], [np.mean(values), np.mean(values)], color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("Pair-level average precision")
        ax.set_xticks(x, [METHODS[m].replace("Three-channel", "3-ch.") for m in selected_methods], rotation=25, ha="right")
        ax.set_title(f"{label}: false-pair burden")

        ax = axes[row, 2]
        positive = weights[weights.method == "three_channel_strictly_positive"]
        channels = ["waveform", "time", "sky"]
        channel_colors = ["#277DA1", "#F8961E", "#43AA8B"]
        for channel_index, (channel, color) in enumerate(zip(channels, channel_colors)):
            values = positive[channel].to_numpy(dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(np.full(len(values), channel_index) + jitter, values, color=color, s=28)
            ax.plot([channel_index - 0.18, channel_index + 0.18], [np.median(values), np.median(values)], color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("Validation-selected weight")
        ax.set_xticks(range(3), ["Waveform", "Time", "Sky"])
        ax.set_title(f"{label}: positive weights")
    for label, ax in zip("abcdef", axes.ravel()):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.grid(alpha=0.15)
    fig.suptitle("Run-matched real-noise injection validation and frozen real-catalog deployment", fontweight="bold", fontsize=13)
    stem = figure_dir / "fig_physical_gwtc34_deployment"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def make_channel_diagnostic_figure(root: Path, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    channels = [("waveform_score", "Calibrated waveform log evidence"), ("time_score", "Run-conditioned time log LR"), ("sky_score", "Sky log Bayes factor")]
    for row, (deployment, label) in enumerate(DEPLOYMENTS.items()):
        frame = pd.read_parquet(root / deployment / f"seed_{SEEDS[0]}" / "results" / "fusion_heldout_test_pairs_physical.parquet")
        for col, (column, axis_label) in enumerate(channels):
            ax = axes[row, col]
            null = frame.loc[frame.is_true_pair == 0, column].to_numpy(dtype=float)
            true = frame.loc[frame.is_true_pair == 1, column].to_numpy(dtype=float)
            limits = np.quantile(np.concatenate([null, true]), [0.005, 0.995])
            bins = np.linspace(limits[0], limits[1], 45)
            ax.hist(null, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#9E9E9E", label="Non-companion")
            ax.hist(true, bins=bins, density=True, histtype="step", lw=1.4, color="#D62828", label="True injected")
            ax.axvline(0, color="black", ls="--", lw=0.6)
            ax.set_xlabel(axis_label)
            ax.set_ylabel("Density")
            ax.set_title(label)
            ax.grid(alpha=0.15)
            if row == 0 and col == 2:
                ax.legend(fontsize=8)
    for label, ax in zip("abcdef", axes.ravel()):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.suptitle("Held-out injected companion and non-companion evidence distributions", fontweight="bold", fontsize=13)
    stem = figure_dir / "fig_physical_channel_evidence_diagnostics"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def make_candidate_figure(root: Path, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    for row, (deployment, label) in enumerate(DEPLOYMENTS.items()):
        path = root / deployment / f"seed_{SEEDS[0]}" / "results" / "real_pair_scores_strict_h1l1_bbh_primary.parquet"
        frame = pd.read_parquet(path).head(10).sort_values("rank")
        ax = axes[row, 0]
        y = np.arange(len(frame))
        left = np.zeros(len(frame))
        for column, name, color in (("waveform_contribution", "Waveform", "#277DA1"), ("time_contribution", "Time", "#F8961E"), ("sky_contribution", "Sky", "#43AA8B")):
            values = frame[column].to_numpy(dtype=float)
            ax.barh(y, values, left=left, color=color, alpha=0.85, label=name)
            left += values
        ax.set_yticks(y, [f"{r.event_i}\n{r.event_j}" for r in frame.itertuples(index=False)], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Weighted log-evidence contribution")
        ax.set_title(f"{label}: primary strict H1L1 top 10")
        ax.axvline(0, color="black", lw=0.6)
        if row == 0:
            ax.legend(ncol=3, fontsize=8)
        ax = axes[row, 1]
        stability = pd.read_csv(root / deployment / "real_candidate_primary_rank_stability.csv")
        subset = stability[stability.scope == "strict_h1l1_bbh"].head(20)
        x = np.arange(len(subset))
        ax.errorbar(x, subset.median_rank, yerr=[subset.median_rank - subset.q25_rank, subset.q75_rank - subset.median_rank], fmt="o", ms=3.5, capsize=2, color="#6A4C93")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("Consensus candidate order")
        ax.set_ylabel("Rank across seeds")
        ax.set_title(f"{label}: seed stability")
        ax.grid(alpha=0.15)
    fig.suptitle("Real-catalog coincidences are triage ranks, not detections", fontweight="bold", fontsize=13)
    stem = figure_dir / "fig_physical_real_candidate_context"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def deployment_result(root: Path, deployment: str) -> dict[str, Any]:
    summaries = [seed_summary(root, deployment, seed) for seed in SEEDS]
    retrieval = pd.read_csv(root / deployment / "heldout_test_retrieval_metrics_per_seed.csv")
    pair = pd.read_csv(root / deployment / "heldout_test_pair_level_metrics_per_seed.csv")
    weights = pd.read_csv(root / deployment / "fusion_weights_per_seed.csv")
    stability = pd.read_csv(root / deployment / "real_candidate_primary_rank_stability.csv")
    strict = stability[stability.scope == "strict_h1l1_bbh"].head(10)
    metrics = {}
    for method in METHODS:
        metrics[method] = {}
        for metric in ("r_at_1", "r_at_10"):
            mean, std = metric_value(retrieval, method, "overall", metric)
            metrics[method][metric] = {"mean": mean, "std": std}
        values = pair[(pair.method == method) & (pair.metric == "average_precision")]["value"].to_numpy(dtype=float)
        metrics[method]["average_precision"] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    pre = pd.read_csv(root / deployment / "shared" / "real_event_preprocessing_audit.csv")
    return {
        "label": DEPLOYMENTS[deployment],
        "n_primary_events": int(len(pre)),
        "n_strict_h1l1_events": int(pre.strict_h1l1_preprocessing_pass.sum()),
        "n_strict_h1l1_bbh_events": int((pre.strict_h1l1_preprocessing_pass & ~pre.is_ood_for_bbh_encoder).sum()),
        "n_strict_h1l1_bbh_pairs": int(math.comb(int((pre.strict_h1l1_preprocessing_pass & ~pre.is_ood_for_bbh_encoder).sum()), 2)),
        "gate1_all_seeds_passed": bool(all(item["waveform_gate1"]["passed"] for item in summaries)),
        "deployment_audit_all_seeds_passed": bool(all(item["deployment_passed"] for item in summaries)),
        "primary_methods": [item["primary_method_policy"] for item in summaries],
        "primary_method_reasons": [item["primary_method_reason"] for item in summaries],
        "gate1": [item["waveform_gate1"] for item in summaries],
        "embedding_audits": [item["real_deployment_audit"] for item in summaries],
        "metrics": metrics,
        "weights": weights.to_dict(orient="records"),
        "top10_consensus_primary": strict.to_dict(orient="records"),
        "historical_pair": [item.get("gw170104_gw170814", {}) for item in summaries],
    }


def fmt_metric(item: dict[str, float]) -> str:
    return f"{item['mean']:.3f} +/- {item['std']:.3f}"


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(summary: dict[str, Any], chinese: bool) -> str:
    title = "# GWTC-3 / GWTC-4.1 修正后的真实噪声部署报告" if chinese else "# Corrected GWTC-3 / GWTC-4.1 real-noise deployment report"
    lines = [title, ""]
    if chinese:
        lines += [
            "## 结论边界",
            "",
            "本实验输出的是供后续 Bayesian 分析的候选短名单，不是透镜探测。三通道权重只由 synthetic lensed signals injected into run-matched real off-source noise 的 validation split 选择；真实 GWTC 排名和 PE 后验均未用于调权。",
            "",
        ]
    else:
        lines += [
            "## Scope",
            "",
            "This experiment produces a candidate shortlist for Bayesian follow-up, not a lensing detection. Fusion is selected only on synthetic lensed signals injected into run-matched real off-source noise; real GWTC ranks and PE posteriors are never used for tuning.",
            "",
        ]
    rows = []
    for key in DEPLOYMENTS:
        item = summary[key]
        rows.append([item["label"], item["n_primary_events"], item["n_strict_h1l1_bbh_events"], item["n_strict_h1l1_bbh_pairs"], item["gate1_all_seeds_passed"], item["deployment_audit_all_seeds_passed"]])
    lines += ["## " + ("数据与审计" if chinese else "Data and audits"), "", markdown_table(rows, ["Catalog", "Primary events", "Strict H1L1 BBH", "Strict pairs", "Gate-1 all seeds", "Deployment audit all seeds"]), ""]
    for key in DEPLOYMENTS:
        item = summary[key]
        metric_rows = []
        for method, label in METHODS.items():
            metric_rows.append([label, fmt_metric(item["metrics"][method]["r_at_1"]), fmt_metric(item["metrics"][method]["r_at_10"]), fmt_metric(item["metrics"][method]["average_precision"])])
        lines += [f"## {item['label']}", "", markdown_table(metric_rows, ["Method", "R@1 mean +/- SD", "R@10 mean +/- SD", "AUPRC mean +/- SD"]), ""]
        lines += [
            ("Primary method by seed: " if not chinese else "各 seed 主方法：")
            + ", ".join(item["primary_methods"]),
            "",
        ]
        top_rows = [[int(row["median_rank"]), row["event_i"], row["event_j"], f"{row['q25_rank']:.1f}-{row['q75_rank']:.1f}", f"{row['top10_frequency']:.2f}"] for row in item["top10_consensus_primary"]]
        lines += [markdown_table(top_rows, ["Median rank", "Event i", "Event j", "IQR", "Top-10 frequency"]), ""]
    if chinese:
        lines += [
            "## 方法解释",
            "",
            "波形通道使用 PSD 物理标定的 24 s H1/L1 输入；时间通道是 GW-LMC delay prior 对同一 run exposure null 的对数似然比；天空通道是 HEALPix common-source Bayes factor。真实目录不再逐行 z-score，因此宽天空图的接近零证据不会被强行放大。",
            "",
            "如果 waveform 的 held-out Gate-1 或真实 embedding audit 失败，主结果按预注册规则回退为 time+sky；严格正权重三通道只保留为补充敏感性结果。PE 内禀参数一致性是冻结排名后的独立筛查。若榜首不通过 PE，结论是当前检索没有给出可信 follow-up candidate，而不是在真实榜首上继续调权。",
            "",
        ]
    else:
        lines += [
            "## Interpretation",
            "",
            "The waveform channel uses PSD-calibrated full-24-s H1/L1 inputs; time is a GW-LMC delay likelihood ratio against a run-exposure null; sky is a HEALPix common-source Bayes factor. Real-catalog row standardization is removed, so weak evidence from broad maps is not artificially magnified.",
            "",
            "If waveform fails held-out Gate-1 or the real-embedding audit, the preregistered primary result falls back to time+sky; the strictly positive three-channel model is retained only as a supplementary sensitivity result. PE intrinsic-parameter consistency is an independent screen after ranking is frozen. A failed PE screen means that the current retrieval produced no credible follow-up candidate; it is not a license to retune on the real catalog.",
            "",
        ]
    pe = summary.get("pe_followup")
    if pe:
        lines += ["## " + ("独立 PE 后验一致性审计" if chinese else "Independent PE posterior-consistency audit"), ""]
        pe_rows = []
        for key in DEPLOYMENTS:
            item = pe[key]
            pe_rows.append(
                [
                    item["label"],
                    f"{item['top10_3sigma_pass']}/10",
                    f"{item['top10_pe_enrichment']:.3f}",
                    f"{item['top10_pe_enrichment_p']:.3g}",
                    f"{item['spearman_score_vs_negative_dmc']:.3f}",
                    item["consensus_top50_pe_followup_count"],
                ]
            )
        lines += [
            markdown_table(
                pe_rows,
                ["Catalog", "Top-10 D<=3", "Enrichment", "Hypergeom. p", "Spearman(score,-Dmc)", "Top-50 PE follow-up"],
            ),
            "",
        ]
        if chinese:
            lines += [
                "这里的 3-sigma 条件和 KDE posterior-overlap 只是进入完整 GOLUM/hanabi 分析前的物理一致性筛查，不是探测显著性。若 Top-50 没有通过者，正式结论就是没有得到 PE-consistent follow-up candidate。",
                "",
            ]
        else:
            lines += [
                "The 3-sigma condition and KDE posterior-overlap are physical screens before a full GOLUM/hanabi analysis, not detection significances. If no Top-50 pair passes, the formal conclusion is that no PE-consistent follow-up candidate was obtained.",
                "",
            ]
    lines += [
        "## References",
        "",
        "- Haris et al. 2018: https://arxiv.org/abs/1807.07062",
        "- LVK O3a lensing search: https://dcc.ligo.org/LIGO-P2000400-v10/public",
        "- LVK full O3 lensing search: https://dcc.ligo.org/LIGO-P2200031-v8/public",
        "- GW-LMC: https://arxiv.org/abs/2603.09289",
        "- FINDCHIRP: https://arxiv.org/abs/gr-qc/0509116",
        "- GOLUM: https://arxiv.org/abs/2203.06444",
        "- Posterior overlap 2.0: https://arxiv.org/abs/2412.01278",
        "- O1--O4a PO2.0 search: https://arxiv.org/abs/2607.08466",
        "",
    ]
    return "\n".join(lines)


def write_reproduce(root: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gw-catalog
/root/miniconda3/bin/python scripts/experiments/31_real_noise_injection_v4_formal.py \\
  --out-root results/real_noise_injection_v4_physical_20260721 \\
  --deployments GWTC3 GWTC4 \\
  --seeds 202607221 202607222 202607223 \\
  --samples-per-family 600 --adapt-epochs 40 --batch-size 32 --bootstrap-draws 10000
/root/miniconda3/bin/python scripts/experiments/19_validate_physical_scoring_invariants.py \\
  --result-root results/real_noise_injection_v4_physical_20260721
/root/miniconda3/bin/python scripts/experiments/32_physical_deployment_v4_pe_audit.py
/root/miniconda3/bin/python scripts/experiments/33_make_physical_deployment_v4_report.py
"""
    path = root / "reproduce.sh"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def package(root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    explicit = [
        REPO / "scripts" / "real_search" / "physical_common.py",
        REPO / "scripts" / "experiments" / "19_validate_physical_scoring_invariants.py",
        REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py",
        REPO / "scripts" / "experiments" / "26_spectrogram_encoder_pilot.py",
        REPO / "scripts" / "experiments" / "29_materialize_multinoise_training.py",
        REPO / "scripts" / "experiments" / "30_multinoise_curriculum_pilot.py",
        REPO / "scripts" / "experiments" / "31_real_noise_injection_v4_formal.py",
        REPO / "scripts" / "experiments" / "32_physical_deployment_v4_pe_audit.py",
        REPO / "scripts" / "experiments" / "33_make_physical_deployment_v4_report.py",
        REPO / "docs" / "methods" / "real_noise_physical_deployment_protocol_20260721_cn.md",
    ]
    with tarfile.open(package_path, "w:gz") as archive:
        for path in explicit:
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(REPO)))
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in {"matchroots", "waveform_gate"} for part in path.parts):
                continue
            if path.name in {"noise_reference_bank.npy", "noise_psd_bank.npy", "real_pe_sky_templates_nside64.npy", "real_event_preprocessed_full24.npy"}:
                continue
            if path.suffix in {".h5", ".hdf5", ".pt"}:
                continue
            archive.add(path, arcname=str(path.relative_to(REPO)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    figure_dir = args.root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    make_main_figure(args.root, figure_dir)
    make_channel_diagnostic_figure(args.root, figure_dir)
    make_candidate_figure(args.root, figure_dir)
    summary = {key: deployment_result(args.root, key) for key in DEPLOYMENTS}
    pe_summary_path = args.root / "pe_followup" / "pe_followup_summary.json"
    if pe_summary_path.exists():
        summary["pe_followup"] = json.loads(pe_summary_path.read_text(encoding="utf-8"))
    write_json(args.root / "physical_deployment_summary.json", summary)
    (args.root / "real_search_physical_report_cn.md").write_text(build_report(summary, True), encoding="utf-8")
    (args.root / "real_search_physical_report_en.md").write_text(build_report(summary, False), encoding="utf-8")
    (args.root / "README_CURRENT_RESULTS.md").write_text(build_report(summary, False), encoding="utf-8")
    write_reproduce(args.root)
    package(args.root, args.package)
    print(json.dumps({"status": "complete", "root": str(args.root), "package": str(args.package)}, indent=2))


if __name__ == "__main__":
    main()
