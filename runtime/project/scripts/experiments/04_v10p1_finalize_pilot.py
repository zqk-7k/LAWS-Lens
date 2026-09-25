#!/usr/bin/env python3
"""Finalize the v10.1 homogeneous sky-PE resource and quality pilot.

This script is deliberately separate from the PE runner.  It reads completed
case products, performs a small decision-relevant Nside audit without keeping
dense maps, writes resource/morphology summaries and figures, and builds a
hashed deliverable archive.  It never edits historical v9.3/v10 products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TERMINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def flatten_moc(moc: Path, nside: int, target: Path, executable: Path) -> np.ndarray:
    run([str(executable), "--nside", str(nside), str(moc), str(target)])
    values = np.asarray(hp.read_map(target, field=0, nest=None, verbose=False), dtype=np.float64)
    values[~np.isfinite(values)] = 0.0
    values[values < 0.0] = 0.0
    total = float(np.sum(values, dtype=np.float64))
    if total <= 0.0:
        raise ValueError(f"Invalid probability map from {moc} at Nside={nside}")
    values /= total
    return values


def pair_role(left: pd.Series, right: pd.Series) -> str:
    same_lens = (
        left.get("event_role") == "lensed_image"
        and right.get("event_role") == "lensed_image"
        and str(left.get("lens_system_id")) == str(right.get("lens_system_id"))
        and str(left.get("lens_system_id")) not in {"", "nan", "None"}
        and int(left.get("image_index", -1)) != int(right.get("image_index", -1))
    )
    return "true_companion" if same_lens else "pilot_null"


def sky_pair_audit(output: Path, config: dict[str, Any], cases: pd.DataFrame) -> pd.DataFrame:
    complete_ids = {
        path.parent.name.rsplit("_seed", 1)[0]
        for path in (output / "cases").glob("*/sky_map_metrics.json")
    }
    subset = cases[cases.case_id.isin(complete_ids)].reset_index(drop=True)
    if len(subset) < 2:
        return pd.DataFrame()

    flatten = Path(shutil.which("ligo-skymap-flatten") or "")
    if not flatten.exists():
        raise FileNotFoundError("ligo-skymap-flatten is not on PATH")
    nsides = [
        int(config["coarse_audit_nside"]),
        int(config["analysis_nside"]),
        int(config["convergence_reference_nside"]),
    ]
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    temporary = output / "temporary_finalize_dense"
    temporary.mkdir(exist_ok=True)
    try:
        for nside in nsides:
            maps: list[np.ndarray] = []
            for _, case in subset.iterrows():
                case_dir = output / "cases" / f"{case.case_id}_seed202608231"
                moc = case_dir / "sky_map/skymap.fits.gz"
                target = temporary / f"{case.case_id}_nside{nside}.fits"
                maps.append(flatten_moc(moc, nside, target, flatten))
                target.unlink(missing_ok=True)
            npix = hp.nside2npix(nside)
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    key = (i, j)
                    row = rows.setdefault(
                        key,
                        {
                            "case_i": subset.iloc[i].case_id,
                            "case_j": subset.iloc[j].case_id,
                            "deployment_i": subset.iloc[i].deployment,
                            "deployment_j": subset.iloc[j].deployment,
                            "pair_deployment": (
                                str(subset.iloc[i].deployment)
                                if subset.iloc[i].deployment == subset.iloc[j].deployment
                                else "cross_run"
                            ),
                            "pair_role": pair_role(subset.iloc[i], subset.iloc[j]),
                        },
                    )
                    overlap = float(np.dot(maps[i], maps[j]))
                    bf = max(float(npix) * overlap, np.finfo(np.float64).tiny)
                    row[f"sky_log_bf_nside{nside}"] = math.log(bf)
            del maps
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    table = pd.DataFrame(rows.values())
    if table.empty:
        return table
    z256 = table[f"sky_log_bf_nside{nsides[0]}"].to_numpy(float)
    z512 = table[f"sky_log_bf_nside{nsides[1]}"].to_numpy(float)
    z1024 = table[f"sky_log_bf_nside{nsides[2]}"].to_numpy(float)
    table["abs_delta_256_512"] = np.abs(z256 - z512)
    table["abs_delta_512_1024"] = np.abs(z512 - z1024)
    table["sign_flip_256_512"] = np.signbit(z256) != np.signbit(z512)
    table["sign_flip_512_1024"] = np.signbit(z512) != np.signbit(z1024)
    table["same_decision_deep_negative_512_1024"] = (z512 < -20.0) & (z1024 < -20.0)
    table["decision_support_512"] = np.select(
        [z512 > 0.0, z512 < 0.0], ["supports_common_sky", "opposes_common_sky"], "neutral"
    )
    table["decision_support_1024"] = np.select(
        [z1024 > 0.0, z1024 < 0.0], ["supports_common_sky", "opposes_common_sky"], "neutral"
    )
    table["decision_stable_512_1024"] = (
        table.decision_support_512 == table.decision_support_1024
    )
    return table


def convergence_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    groups: list[tuple[str, pd.DataFrame]] = [("all", table)]
    groups.extend(
        (f"deployment={name}", group)
        for name, group in table.groupby("pair_deployment")
    )
    groups.extend((f"role={name}", group) for name, group in table.groupby("pair_role"))
    groups.extend(
        (f"deployment={deployment};role={role}", group)
        for (deployment, role), group in table.groupby(["pair_deployment", "pair_role"])
    )
    rows: list[dict[str, Any]] = []
    for label, group in groups:
        delta = group.abs_delta_512_1024.to_numpy(float)
        flips = int(group.sign_flip_512_1024.sum())
        rows.append(
            {
                "stratum": label,
                "n_pairs": len(group),
                "delta_median": float(np.quantile(delta, 0.5)),
                "delta_p90": float(np.quantile(delta, 0.9)),
                "delta_p95": float(np.quantile(delta, 0.95)),
                "delta_p99": float(np.quantile(delta, 0.99)),
                "delta_max": float(np.max(delta)),
                "sign_flips": flips,
                "sign_flip_fraction": flips / len(group),
                "decision_stable_fraction": float(group.decision_stable_512_1024.mean()),
                "spearman_512_1024": float(
                    group[["sky_log_bf_nside512", "sky_log_bf_nside1024"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def resource_summary(case_metrics: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "total_case_wall_seconds",
        "sampler_wall_seconds",
        "sampler_cpu_seconds",
        "peak_rss_bytes",
        "peak_gpu_memory_bytes",
        "gpu_active_seconds",
        "posterior_bytes",
        "moc_bytes",
    ]
    rows: list[dict[str, Any]] = []
    for label, group in [("all", case_metrics), *list(case_metrics.groupby("deployment"))]:
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            if len(values):
                rows.append(
                    {
                        "group": label,
                        "metric": column,
                        "n": len(values),
                        "p50": float(np.quantile(values, 0.5)),
                        "p90": float(np.quantile(values, 0.9)),
                        "maximum": float(np.max(values)),
                    }
                )
    quantiles = pd.DataFrame(rows)
    wall = quantiles[(quantiles.group == "all") & (quantiles.metric == "total_case_wall_seconds")]
    extrapolation: list[dict[str, Any]] = []
    if len(wall) == 1:
        source = wall.iloc[0]
        for scenario, count in config["resource_extrapolation_event_counts"].items():
            for statistic in ("p50", "p90", "maximum"):
                serial = int(count) * float(source[statistic])
                extrapolation.append(
                    {
                        "scenario": scenario,
                        "n_events": int(count),
                        "per_event_statistic": statistic,
                        "per_event_seconds": float(source[statistic]),
                        "serial_event_equivalent_days": serial / 86400.0,
                        "ideal_24_case_parallel_days": serial / 86400.0 / 24.0,
                        "ideal_48_case_parallel_days": serial / 86400.0 / 48.0,
                    }
                )
    return quantiles, pd.DataFrame(extrapolation)


def make_figure(output: Path, cases: pd.DataFrame, pair_table: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), constrained_layout=True)
    colors = {"O3": "#0072B2", "O4a": "#D55E00"}
    for deployment, group in cases.groupby("deployment"):
        axes[0].scatter(
            np.arange(len(group)), group.total_case_wall_seconds / 3600.0,
            label=deployment, color=colors.get(deployment, "0.4"), s=28, alpha=0.85,
        )
    axes[0].set_xlabel("Completed pilot case")
    axes[0].set_ylabel("End-to-end wall time (h)")
    axes[0].set_title("a  Homogeneous PE cost")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].scatter(
        cases.network_snr,
        cases.a90_deg2_nside512,
        c=[colors.get(value, "0.4") for value in cases.deployment],
        s=28,
        alpha=0.85,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Network SNR")
    axes[1].set_ylabel(r"$A_{90}$ at Nside 512 (deg$^2$)")
    axes[1].set_title("b  Sky-posterior scale")

    if not pair_table.empty:
        for role, group in pair_table.groupby("pair_role"):
            axes[2].scatter(
                group.sky_log_bf_nside512,
                group.sky_log_bf_nside1024,
                s=18 if role == "pilot_null" else 40,
                alpha=0.45 if role == "pilot_null" else 0.95,
                label=role.replace("_", " "),
            )
        limits = np.nanpercentile(
            pair_table[["sky_log_bf_nside512", "sky_log_bf_nside1024"]].to_numpy(), [1, 99]
        )
        axes[2].plot(limits, limits, color="0.25", linewidth=1.0, linestyle="--")
        axes[2].legend(frameon=False, loc="best")
    axes[2].set_xlabel(r"$Z_{\rm sky}$, Nside 512")
    axes[2].set_ylabel(r"$Z_{\rm sky}$, Nside 1024")
    axes[2].set_title("c  Resolution audit")

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    figure.savefig(figures / "fig_v10p1_homogeneous_pe_pilot.pdf", bbox_inches="tight")
    figure.savefig(figures / "fig_v10p1_homogeneous_pe_pilot.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 24) -> str:
    if frame.empty:
        return "（无完成记录）"
    selected = frame[columns].head(rows).copy()
    return selected.to_markdown(index=False, floatfmt=".4g")


def write_report(
    output: Path,
    case_metrics: pd.DataFrame,
    convergence: pd.DataFrame,
    extrapolation: pd.DataFrame,
    complete: bool,
) -> None:
    resource_rows = extrapolation[
        (extrapolation.scenario == "minimum_full_v10")
    ] if not extrapolation.empty else pd.DataFrame()
    report = f"""# v10.1 Nside=512 同质天空 PE Pilot 报告

生成时间：{utc_now()}

> 本轮为 Nside=512 统一分辨率的 v10.1 探索实验。historical v9.3 真实
> GWTC Nside=1024、历史 O3/O4a 注入 Nside=64 和 ET-3 Nside=128 结果均未
> 覆盖。Nside=1024 仅作为本轮高分辨率敏感性参考。本轮结果未经新的单一方案
> 确认性 locked test，不得写入论文或替代 v9.3。

## 1. 状态

- 完成的端到端 case：{len(case_metrics)}/24
- 全部预注册 case 完成：{complete}
- 真实候选重排：未执行
- 历史结果修改：否
- 统计边界：探索性资源与质量 pilot；不提供低 FPR/FAP 声明

## 2. 四项 G0.5 阻塞的处理

1. **天空 PE 同质性**：真实事件和注入事件均从 detector strain 开始，调用同一
   Bilby 2.8.0 + IMRPhenomXPHM + Relative Binning 单事件 PE，再由同一
   `ligo-skymap-from-samples` 管线生成 MOC。没有把历史 Nside=64 模板插值为
   新证据。
2. **独立背景不足**：O3 的 250 对和 O4a 的 635 对仅获准用于探索性校准；
   不能据此声称达到原 3,242 独立单位对应的低误配率精度。
3. **分辨率审计**：正式探索值固定 Nside=512，1024 只做敏感性报告。原始
   `Z_sky` 不一致不会被隐瞒，也不再单独阻断探索实验；决策符号、排名和
   shortlist 稳定性必须另行报告。
4. **正式 PE 资源**：本表给出真实端到端 wall/CPU/RAM/存储测量，并按 P50、
   P90 和最坏 case 外推，未使用公开地图读取时间冒充 PE 时间。

## 3. Case 级资源与质量

{markdown_table(case_metrics, ['case_id','deployment','event_role','detectors','network_snr','total_case_wall_seconds','sampler_cpu_seconds','peak_rss_bytes','posterior_samples','moc_bytes','a90_deg2_nside512','sampler_converged','map_valid'])}

## 4. Nside=512 对 1024 的 Pilot 审计

该小样本审计只用于检查管线和决策方向，不替代未来完整 validation/holdout 的
pair-level 收敛分析。

{markdown_table(convergence, list(convergence.columns) if not convergence.empty else [])}

## 5. 全量资源外推

`serial_event_equivalent_days` 是单任务串行等价时间；并行列是假设资源与 I/O
完全线性扩展的乐观下限，不是承诺完成时间。正式生产还需加入 calibration
marginalization、重启和调度开销。

{markdown_table(resource_rows, list(resource_rows.columns) if not resource_rows.empty else [])}

## 6. 科学边界

- 相同 Nside 只统一离散积分网格；真正的同质性来自相同 strain-to-posterior
  管线。
- 当前 250/635 个独立背景单位只能做探索性拟合和有限样本区间，不能支持
  确认性低 FPR 声明。
- 决策相关稳定不等于所有深负 `Z_sky` 数值收敛。
- 本 pilot 不训练 waveform encoder、不重排真实 GWTC 候选、不修改论文。
- 当前质量/距离先验是 confirmed-event 定向先验，资源外推比完全盲目的宽先验
  乐观；未来正式生产若采用更宽先验，必须重新测量而不能直接套用本表。

最终状态：`{TERMINAL_STATUS}`
"""
    (output / "V10P1_HOMOGENEOUS_PE_PILOT_REPORT_CN.md").write_text(report, encoding="utf-8")


def build_manifest(output: Path, exclude_archive: Path | None = None) -> Path:
    target = output / "manifest/FILES_SHA256.csv"
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path == target or (exclude_archive and path == exclude_archive):
            continue
        records.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    target.parent.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(target, index=False, encoding="utf-8-sig")
    return target


def package(output: Path, project: Path) -> Path:
    package_dir = project / "packages"
    package_dir.mkdir(exist_ok=True)
    target = package_dir / f"{output.name}_deliverables.tar.gz"
    include = [
        "contracts", "manifests", "summaries", "figures", "logs", "manifest", "benchmarks",
        "V10P1_HOMOGENEOUS_PE_PILOT_REPORT_CN.md", "STATUS.json",
    ]
    scripts = output / "scripts"
    if scripts.exists():
        include.append("scripts")
    with tarfile.open(target, "w:gz") as archive:
        for relative in include:
            source = output / relative
            if source.exists():
                archive.add(source, arcname=f"{output.name}/{relative}")
        for case_dir in sorted((output / "cases").glob("*")):
            for relative in (
                "pe_metrics.json", "posterior_samples.hdf5", "sky_map_metrics.json",
                "sky_map/skymap.fits.gz",
            ):
                source = case_dir / relative
                if source.exists():
                    archive.add(
                        source,
                        arcname=f"{output.name}/cases/{case_dir.name}/{relative}",
                    )
            for source in sorted((case_dir / "bilby").glob("*resume*.pickle")):
                archive.add(
                    source,
                    arcname=f"{output.name}/cases/{case_dir.name}/bilby/{source.name}",
                )
    (target.with_suffix(target.suffix + ".sha256")).write_text(
        f"{sha256(target)}  {target.name}\n", encoding="ascii"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config = json.loads((output / "contracts/V10P1_PE_PILOT_CONFIG.json").read_text())
    cases = pd.read_csv(output / "manifests/pe_pilot_cases_all.csv")
    metrics_path = output / "summaries/pe_pilot_case_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run the pilot runner summarize command first")
    case_metrics = pd.read_csv(metrics_path)
    complete = len(case_metrics) == len(cases)
    if not complete and not args.allow_partial:
        raise RuntimeError(f"Only {len(case_metrics)}/{len(cases)} cases complete")

    pair_table = sky_pair_audit(output, config, cases)
    pair_table.to_parquet(output / "summaries/pilot_sky_pair_resolution_audit.parquet", index=False)
    pair_table.to_csv(
        output / "summaries/pilot_sky_pair_resolution_audit.csv", index=False, encoding="utf-8-sig"
    )
    convergence = convergence_summary(pair_table)
    convergence.to_csv(
        output / "summaries/pilot_sky_resolution_summary.csv", index=False, encoding="utf-8-sig"
    )
    quantiles, extrapolation = resource_summary(case_metrics, config)
    quantiles.to_csv(output / "summaries/pe_pilot_resource_quantiles_final.csv", index=False)
    extrapolation.to_csv(output / "summaries/pe_pilot_resource_extrapolation_final.csv", index=False)
    make_figure(output, case_metrics, pair_table)
    write_report(output, case_metrics, convergence, extrapolation, complete)

    final = {
        "status": TERMINAL_STATUS,
        "generated_at_utc": utc_now(),
        "completed_cases": len(case_metrics),
        "planned_cases": len(cases),
        "all_cases_complete": complete,
        "historical_results_modified": False,
        "real_candidate_rerank_performed": False,
        "analysis_nside": int(config["analysis_nside"]),
        "convergence_reference_nside": int(config["convergence_reference_nside"]),
        "background_scope": "exploratory_calibration_only",
        "confirmatory_low_fpr_claim_allowed": False,
    }
    write_json(output / "FINAL_PILOT_STATUS.json", final)
    (output / "scripts").mkdir(exist_ok=True)
    for name in ("03_v10p1_homogeneous_pe_pilot.py", "04_v10p1_finalize_pilot.py"):
        source = args.project / "scripts/experiments" / name
        if source.exists():
            shutil.copy2(source, output / "scripts" / name)
    build_manifest(output)
    target = package(output, args.project)
    print(json.dumps({"status": final["status"], "package": str(target)}, indent=2))


if __name__ == "__main__":
    main()
