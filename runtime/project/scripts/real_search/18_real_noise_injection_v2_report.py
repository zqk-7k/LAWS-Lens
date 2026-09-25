from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "results" / "real_noise_injection_v2_20260719"
DEFAULT_PACKAGE = REPO_ROOT / "packages" / "real_noise_injection_v2_20260719.tar.gz"
DEPLOYMENTS = ("GWTC3", "GWTC4")
PRIMARY = "waveform_time_sky_positive"
METHOD_LABELS = {
    "waveform_only": "Waveform only",
    "time_only": "Time only",
    "sky_only": "Sky only",
    "time_sky_positive": "Time + sky",
    "waveform_time_sky_unconstrained": "Three-channel unconstrained",
    "waveform_time_sky_positive": "Three-channel positive",
    "waveform_time_sky_equal": "Three-channel equal",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def discover_seeds(root: Path, deployment: str) -> list[int]:
    return sorted(int(path.name.split("_", 1)[1]) for path in (root / deployment.lower()).glob("seed_*") if (path / "seed_summary.json").exists())


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    retrieval = []
    pair = []
    weights = []
    gates = []
    candidates: dict[str, Any] = {}
    for deployment in DEPLOYMENTS:
        seeds = discover_seeds(root, deployment)
        candidates[deployment] = {"seeds": seeds}
        for seed in seeds:
            base = root / deployment.lower() / f"seed_{seed}"
            retrieval.append(pd.read_csv(base / "results" / "heldout_test_retrieval_metrics.csv"))
            pair.append(pd.read_csv(base / "results" / "heldout_test_pair_level_metrics.csv"))
            selected = json.loads((base / "results" / "selected_weights.json").read_text(encoding="utf-8"))
            for key, method in (
                ("unconstrained_waveform_time_sky", "waveform_time_sky_unconstrained"),
                ("strictly_positive_waveform_time_sky", "waveform_time_sky_positive"),
                ("equal_waveform_time_sky", "waveform_time_sky_equal"),
                ("strictly_positive_time_sky", "time_sky_positive"),
            ):
                weights.append({"deployment": deployment, "seed": seed, "method": method, **selected[key]})
            gate = json.loads((base / "results" / "waveform_gate1_metrics.json").read_text(encoding="utf-8"))
            gates.append({"deployment": deployment, "seed": seed, **gate})
        stability_path = root / deployment.lower() / "real_candidate_rank_stability.csv"
        if stability_path.exists():
            stability = pd.read_csv(stability_path)
            candidates[deployment]["primary_top20"] = stability[stability["method"] == PRIMARY].head(20).to_dict(orient="records")
    return (
        pd.concat(retrieval, ignore_index=True),
        pd.concat(pair, ignore_index=True),
        pd.DataFrame(weights),
        pd.DataFrame(gates),
        candidates,
    )


def summarize(retrieval: pd.DataFrame, pair: pd.DataFrame, weights: pd.DataFrame, gates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["r_at_1", "r_at_5", "r_at_10", "r_at_50", "median_rank"]
    agg = {f"{name}_mean": (name, "mean") for name in metrics}
    agg.update({f"{name}_std": (name, "std") for name in metrics})
    retrieval_summary = retrieval.groupby(["deployment", "method", "subset"], as_index=False).agg(**agg, seeds=("seed", "nunique"))
    pair_summary = pair.groupby(["deployment", "method", "metric"], as_index=False).agg(
        value_mean=("value", "mean"), value_std=("value", "std"), seeds=("seed", "nunique")
    )
    weight_summary = weights.groupby(["deployment", "method"], as_index=False).agg(
        waveform_mean=("waveform", "mean"), waveform_std=("waveform", "std"),
        time_mean=("time", "mean"), time_std=("time", "std"),
        sky_mean=("sky", "mean"), sky_std=("sky", "std"), seeds=("seed", "nunique"),
    )
    return retrieval_summary, pair_summary, weight_summary


def injection_parameter_summary(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for deployment in DEPLOYMENTS:
        for seed in discover_seeds(root, deployment):
            base = root / deployment.lower() / f"seed_{seed}" / "data" / "real_noise_injections"
            meta = pd.read_parquet(base / "compact_injection_metadata.parquet")
            sky = json.loads((base / "sky_localization_calibration.json").read_text(encoding="utf-8"))
            for family, frame in meta.groupby("family"):
                snr = pd.concat([frame["target_snr_image1"], frame["target_snr_image2"]], ignore_index=True).dropna()
                delay = frame["delay_days"].dropna()
                tail = pd.concat([frame["tail_snr2_fraction_a"], frame["tail_snr2_fraction_b"]], ignore_index=True).dropna()
                rows.append({
                    "deployment": deployment,
                    "seed": seed,
                    "family": family,
                    "systems": int(len(frame)),
                    "snr_median": float(snr.median()),
                    "snr_p10": float(snr.quantile(0.1)),
                    "snr_p90": float(snr.quantile(0.9)),
                    "delay_days_median": float(delay.median()) if len(delay) else np.nan,
                    "delay_days_p90": float(delay.quantile(0.9)) if len(delay) else np.nan,
                    "tail_snr2_fraction_median": float(tail.median()),
                    "a90_ref_deg2": float(sky["a90_ref_deg2"]),
                    "a90_scatter_log_sigma": float(sky["lognormal_sigma"]),
                    "a90_clip_min_deg2": float(sky["clip_min_deg2"]),
                    "a90_clip_max_deg2": float(sky["clip_max_deg2"]),
                })
    return pd.DataFrame(rows)


def nature_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def make_figure(root: Path, retrieval: pd.DataFrame, weights: pd.DataFrame, candidates: dict[str, Any]) -> None:
    nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    methods = ["waveform_only", "time_sky_positive", "waveform_time_sky_unconstrained", PRIMARY]
    colors = {"GWTC3": "#2C7FB8", "GWTC4": "#D95F0E"}
    for col, deployment in enumerate(DEPLOYMENTS):
        ax = axes[0, col]
        frame = retrieval[(retrieval.deployment == deployment) & (retrieval.subset == "overall") & retrieval.method.isin(methods)]
        x = np.arange(len(methods), dtype=float)
        for seed_index, (_, seed_frame) in enumerate(frame.groupby("seed")):
            vals = [float(seed_frame.loc[seed_frame.method == method, "r_at_10"].iloc[0]) for method in methods]
            jitter = (seed_index - (frame.seed.nunique() - 1) / 2) * 0.035
            ax.scatter(x + jitter, vals, s=24, facecolor="white", edgecolor=colors[deployment], linewidth=1.0, zorder=3)
        means = frame.groupby("method")["r_at_10"].mean()
        ax.plot(x, [means.get(method, np.nan) for method in methods], color=colors[deployment], marker="o", lw=1.2, ms=3)
        ax.set_xticks(x, ["WF", "T+S", "3-ch\nunconstr.", "3-ch\npositive"])
        ax.set_ylim(0, 1.04)
        ax.set_ylabel("Held-out R@10")
        ax.set_title("GWTC-3 / O3 noise" if deployment == "GWTC3" else "GWTC-4.1 / O4a noise")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.text(-0.13, 1.04, chr(ord("a") + col), transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1, 0]
    primary_weights = weights[weights.method == PRIMARY]
    channel_colors = {"waveform": "#2166AC", "time": "#4D4D4D", "sky": "#B2182B"}
    xpos = 0
    ticks = []
    labels = []
    for deployment in DEPLOYMENTS:
        frame = primary_weights[primary_weights.deployment == deployment]
        for channel in ("waveform", "time", "sky"):
            vals = frame[channel].to_numpy(dtype=float)
            ax.scatter(np.full(len(vals), xpos) + np.linspace(-0.08, 0.08, len(vals)), vals, s=22, color=channel_colors[channel], alpha=0.85)
            ax.hlines(np.mean(vals), xpos - 0.23, xpos + 0.23, color="black", lw=1.0)
            ticks.append(xpos)
            labels.append(channel[0].upper())
            xpos += 1
        xpos += 0.7
    ax.set_xticks(ticks, labels)
    ax.set_ylabel("Validation-selected weight")
    ax.set_title("Strictly positive three-channel weights")
    ax.text(1, -0.18, "GWTC-3", transform=ax.get_xaxis_transform(), ha="center")
    ax.text(4.7, -0.18, "GWTC-4.1", transform=ax.get_xaxis_transform(), ha="center")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.text(-0.13, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1, 1]
    y = 0
    yticks = []
    ylabels = []
    for deployment in DEPLOYMENTS:
        rows = candidates.get(deployment, {}).get("primary_top20", [])[:8]
        for row in reversed(rows):
            marker = "o" if bool(row.get("full_waveform_scored", False)) else "o"
            face = colors[deployment] if bool(row.get("full_waveform_scored", False)) else "white"
            ax.scatter(float(row["median_rank"]), y, marker=marker, s=28, facecolor=face, edgecolor=colors[deployment])
            ax.hlines(y, float(row["min_rank"]), float(row["max_rank"]), color=colors[deployment], lw=0.8)
            yticks.append(y)
            ylabels.append(str(row["pair_key"]).replace("--", " / "))
            y += 1
        y += 1
    ax.set_yticks(yticks, ylabels, fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("Real-catalog rank across seeds")
    ax.set_title("Primary shortlist stability")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.text(-0.13, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.text(0.5, 0.002, "Filled markers: full waveform-scored; open markers: neutral waveform contribution. Ranked coincidences, not detections.", ha="center", fontsize=6.5)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "fig_real_noise_injection_v2.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig_real_noise_injection_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def report(root: Path, retrieval_summary: pd.DataFrame, pair_summary: pd.DataFrame, weight_summary: pd.DataFrame, gates: pd.DataFrame, candidates: dict[str, Any]) -> None:
    lines_cn = [
        "# Run-matched real-noise injection v2 结果报告",
        "",
        "## 结论边界",
        "",
        "本实验是 O3/O4a 真实 off-source 噪声中的合成透镜双像检索与真实目录候选排序，不是透镜探测。真实目录输出只能解释为后续 Bayesian follow-up 的候选清单。",
        "",
        "## 方案",
        "",
        "- 固定复用主线 SIS/PM clean waveform bank；重新抽取每个 seed 的真实 H1/L1 off-source 噪声。",
        "- 透镜时延来自 GW-LMC 2.5PLUS BBH 样本，并乘以 O3 或 O4a 实际 H1/L1 live-time autocorrelation 完成观测暴露条件化。",
        "- 单事件天空定位面积模型由相应真实目录 HEALPix A90 与 network SNR 拟合；透镜和非透镜事件使用完全相同的观测模型，不使用类别条件误差。",
        "- 网络 SNR 从 8 到 60 的截断 rho^-4 分布采样；SNR 仅用于注入强度和天空定位误差，不进入最终分数。",
        "- 每个 seed 按 source system 做 70/15/15 train/validation/test 划分；两个像不会跨 split。融合权重只在 validation 上选择。",
        "- 主结果预先规定三个权重都严格大于 0；同时报告无约束网格、等权三通道和 time+sky baseline。",
        "",
        "## Held-out test",
        "",
        "| Deployment | Method | R@1 mean+/-SD | R@10 mean+/-SD | median rank |",
        "|---|---|---:|---:|---:|",
    ]
    for deployment in DEPLOYMENTS:
        for method in ("waveform_only", "time_sky_positive", "waveform_time_sky_unconstrained", PRIMARY, "waveform_time_sky_equal"):
            row = retrieval_summary[(retrieval_summary.deployment == deployment) & (retrieval_summary.method == method) & (retrieval_summary.subset == "overall")]
            if row.empty:
                continue
            r = row.iloc[0]
            lines_cn.append(f"| {deployment} | {METHOD_LABELS[method]} | {fmt(r.r_at_1_mean)} +/- {fmt(r.r_at_1_std)} | {fmt(r.r_at_10_mean)} +/- {fmt(r.r_at_10_std)} | {fmt(r.median_rank_mean)} |")
    lines_cn.extend(["", "## 验证集权重", "", "| Deployment | Policy | waveform | time | sky |", "|---|---|---:|---:|---:|"])
    for _, r in weight_summary.iterrows():
        lines_cn.append(f"| {r.deployment} | {METHOD_LABELS.get(r.method, r.method)} | {fmt(r.waveform_mean)} | {fmt(r.time_mean)} | {fmt(r.sky_mean)} |")
    lines_cn.extend(["", "## Gate-1", "", "| Deployment | passed seeds | total seeds | waveform R@10 mean |", "|---|---:|---:|---:|"])
    for deployment in DEPLOYMENTS:
        frame = gates[gates.deployment == deployment]
        lines_cn.append(f"| {deployment} | {int(frame.passed.sum())} | {len(frame)} | {fmt(frame.waveform_only_r_at_10.mean())} |")
    lines_cn.extend([
        "",
        "## 解释限制",
        "",
        "1. synthetic sky 是按真实 HEALPix A90 校准的 Gaussian posterior surrogate，不是为每条注入运行 BAYESTAR/PE。",
        "2. 注入 SNR 是 full-window clean waveform L2 的 empirical whitened-SNR 标定，不等同于搜索管线 recovered matched-filter SNR。",
        "3. off-source 段避开已知事件 +/-128 s，但没有逐段 BayesWave 清理。",
        "4. 正权重结果是预先定义的 validation constraint；若无约束结果把某通道选为 0，应如实报告，不能把正权重结果描述为无约束最优。",
        "5. 真实 GWTC 排名没有真实正标签，因此 rank 不是 p-value，也不是 detection significance。",
        "",
        "## 参数依据",
        "",
        "- GWOSC O3 数据质量与公开 strain：https://gwosc.org/O3/o3_details/",
        "- GWOSC O4a 公开数据：https://gwosc.org/O4/O4a/",
        "- GW-LMC population delay prior：https://arxiv.org/abs/2603.09289",
        "- LVK O4a lensing search 的 real-noise injection 设计：https://dcc.ligo.org/public/0201/P2500419/010/O4aLensingPaperMTapproved.pdf",
        "- GWTC-3 population data release：https://zenodo.org/records/11254021",
        "",
    ])
    (root / "real_noise_injection_v2_report_cn.md").write_text("\n".join(lines_cn), encoding="utf-8")

    lines_en = [
        "# Run-matched real-noise injection v2 report",
        "",
        "This experiment evaluates synthetic lensed doublets injected into run-matched O3/O4a off-source noise and produces real-catalog candidate rankings. It is not a lensing detection analysis.",
        "",
        "The primary fusion policy is fixed before held-out evaluation: waveform, time-delay and unified posterior-overlap sky weights must all be positive. Unconstrained validation selection, equal weights and a positive time+sky baseline are reported alongside it.",
        "",
        "## Important limitations",
        "",
        "- Synthetic localization uses a Gaussian posterior surrogate calibrated to the real run's HEALPix A90 distribution, not per-injection BAYESTAR/PE.",
        "- Target SNR is an empirical full-window waveform-norm calibration, not recovered search-pipeline SNR.",
        "- Off-source windows exclude known events by +/-128 s but are not BayesWave-cleaned.",
        "- A zero channel in unconstrained validation selection must be reported; the positive result is a separately pre-specified constrained comparison.",
        "- Real-catalog ranks are candidate priorities, not p-values or detection significances.",
        "",
        "References: GWOSC O3/O4a public data, the GW-LMC delay population (arXiv:2603.09289), the LVK O4a lensing search injection protocol (LIGO-P2500419), and the GWTC-3 population data release (Zenodo 11254021).",
    ]
    (root / "real_noise_injection_v2_report_en.md").write_text("\n".join(lines_en) + "\n", encoding="utf-8")


def make_reproduce(root: Path, config: dict[str, Any]) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gw-catalog
/root/miniconda3/bin/python scripts/real_search/17_real_noise_injection_v2.py \\
  --out-root {root} \\
  --deployments {' '.join(config['deployments'])} \\
  --seeds {' '.join(map(str, config['seeds']))} \\
  --samples-per-family {config['samples_per_family']} \\
  --epochs {config['epochs']} \\
  --bootstrap-draws {config['bootstrap_draws']}
/root/miniconda3/bin/python scripts/real_search/18_real_noise_injection_v2_report.py --root {root}
"""
    path = root / "reproduce.sh"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def make_protocol_doc(root: Path) -> None:
    text = r"""# Real-noise injection v2 方法协议

## 1. 目标与边界

该协议验证 catalog-level `waveform + time-delay + sky posterior overlap` 排序在 O3/O4a 真实 off-source 噪声中的表现。注入是真值已知的合成双像；真实 GWTC catalog 没有被标注为透镜，因此真实目录只输出 follow-up shortlist。

## 2. 数据隔离

每个 SIS/PM source system 作为不可拆分单元，按 70%/15%/15% 划分 train/validation/test。双像的两个 directed query 始终位于同一 split。encoder 只见 train，权重只由 validation 决定，test 只做一次最终评价。

## 3. Run-matched 真实噪声

O3 使用 O3a+O3b，O4 使用 O4a。通过 GWOSC `H1_DATA` 与 `L1_DATA` 质量段求交集，并从本地 H1/L1 4096 Hz HDF5 的 off-source 区间抽取 24 s 噪声。所有区间避开已知事件的 +/-128 s on-source window。

## 4. SNR 条件化

检出事件近似采用截断分布

\[
p(\rho) \propto \rho^{-4},\qquad 8\leq\rho\leq60.
\]

两幅像的 SNR ratio 从 GW-LMC image population 抽取。clean waveform 先在完整 24 s 内去均值，再按全通道 L2 norm 缩放到目标 network-SNR surrogate，最后与真实噪声相加。该量是 empirical whitened-SNR calibration，不等同于搜索管线 recovered matched-filter SNR。

## 5. 时间通道

令 H1/L1 联合 live-time 指示函数为 \(W_r(t)\)。GW-LMC 时延先验经过观测暴露条件化：

\[
p_{\rm obs}(\Delta t\mid L,r)\propto p_{\rm GW-LMC}(\Delta t)A_r(\Delta t),
\]

\[
A_r(\Delta t)=\int W_r(t)W_r(t+\Delta t)\,dt.
\]

背景只使用 validation catalog 中的 non-companion pairs：

\[
s_t(\Delta t)=\log p_{\rm obs}(\Delta t\mid L,r)-
\log p(\Delta t\mid B,\mathrm{validation}).
\]

两个密度均在 \(\log_{10}\Delta t\) 直方图上估计并加 Laplace pseudocount。test labels 不参与拟合。

## 6. 天空通道

每个 run 从真实 PE HEALPix map 的 \(A_{90}\) 与 catalog network SNR 拟合：

\[
A_{90}=A_{90,\mathrm{ref}}\left(\frac{12}{\rho}\right)^2\epsilon,
\qquad \log\epsilon\sim\mathcal N(0,\sigma_A^2).
\]

透镜与非透镜事件共用同一模型。将面积换成 tangent-plane Gaussian 宽度：

\[
\sigma_{\rm sky}=\sqrt{\frac{A_{90,\rm rad^2}}{2\pi\ln 10}}.
\]

对两个 posterior surrogate，主分数为 analytic log cosine overlap：

\[
s_{\rm sky}=\log\frac{2\sigma_i\sigma_j}{\sigma_i^2+\sigma_j^2}
-\frac{\theta_{ij}^2}{2(\sigma_i^2+\sigma_j^2)}.
\]

真实目录不使用 surrogate，而继续使用真实 PE HEALPix posterior cosine overlap 的对数，二者使用同一 posterior-overlap 语义。

## 7. 融合与正权重约束

三个通道按 query row 标准化：

\[
z_c(i,j)=\frac{s_c(i,j)-\mu_{c,i}}{\sigma_{c,i}}.
\]

方向分数为

\[
S_{i\to j}=w_w z_w(i,j)+w_t z_t(i,j)+w_s z_s(i,j).
\]

主策略在 validation grid 上预先要求

\[
w_w>0,\quad w_t>0,\quad w_s>0.
\]

同时保存 unconstrained、equal-weight 和 time+sky baseline。无序 pair-level 分数沿用正文口径：

\[
S_{\{i,j\}}=\max(S_{i\to j},S_{j\to i}).
\]

## 8. 不应过度解释的部分

1. synthetic sky 不是逐注入 BAYESTAR/PE。
2. off-source 噪声未逐段执行 BayesWave cleaning。
3. clean SIS/PM waveform bank 固定复用主线数据，当前 v2 隔离检验的是 real-noise/SNR/time/sky deployment 条件。
4. constrained-positive 不是 unconstrained optimum 的同义词；二者必须并列报告。
"""
    (root / "REAL_NOISE_INJECTION_V2_PROTOCOL_CN.md").write_text(text + "\n", encoding="utf-8")


def package(root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    include_suffixes = {".csv", ".json", ".md", ".pdf", ".png", ".parquet", ".sh", ".pt"}
    manifest_path = root / "package_manifest_sha256.csv"
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in include_suffixes and p != manifest_path
    ]
    scripts = [
        REPO_ROOT / "scripts" / "real_search" / "17_real_noise_injection_v2.py",
        REPO_ROOT / "scripts" / "real_search" / "18_real_noise_injection_v2_report.py",
    ]
    manifest = []
    for path in sorted(paths + scripts):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest})
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    paths.append(manifest_path)
    with tarfile.open(package_path, "w:gz") as tar:
        # Result trees intentionally reuse authoritative manifests/features via
        # absolute symlinks. Store their contents so the handoff is portable.
        tar.dereference = True
        for path in sorted(set(paths + scripts)):
            if path in scripts:
                arcname = Path("scripts/real_search") / path.name
            else:
                arcname = Path(root.name) / path.relative_to(root)
            tar.add(path, arcname=str(arcname), recursive=False)
    package_digest = hashlib.sha256(package_path.read_bytes()).hexdigest()
    checksum_path = package_path.with_name(package_path.name + ".sha256")
    checksum_path.write_text(f"{package_digest}  {package_path.name}\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    retrieval, pair, weights, gates, candidates = collect(args.root)
    retrieval_summary, pair_summary, weight_summary = summarize(retrieval, pair, weights, gates)
    retrieval.to_csv(args.root / "heldout_test_retrieval_metrics_all_seeds.csv", index=False)
    retrieval_summary.to_csv(args.root / "heldout_test_retrieval_summary.csv", index=False)
    pair.to_csv(args.root / "heldout_test_pair_level_metrics_all_seeds.csv", index=False)
    pair_summary.to_csv(args.root / "heldout_test_pair_level_summary.csv", index=False)
    weights.to_csv(args.root / "fusion_weights_all_seeds.csv", index=False)
    weight_summary.to_csv(args.root / "fusion_weights_summary.csv", index=False)
    gates.to_csv(args.root / "waveform_gate1_all_seeds.csv", index=False)
    injection_parameter_summary(args.root).to_csv(args.root / "injection_parameter_summary.csv", index=False)
    write_json(args.root / "real_candidate_summary.json", candidates)
    make_figure(args.root, retrieval, weights, candidates)
    report(args.root, retrieval_summary, pair_summary, weight_summary, gates, candidates)
    config = json.loads((args.root / "run_config.json").read_text(encoding="utf-8"))
    make_reproduce(args.root, config)
    make_protocol_doc(args.root)
    package(args.root, args.package)
    print(json.dumps({"status": "complete", "root": str(args.root), "package": str(args.package)}, indent=2))


if __name__ == "__main__":
    main()
