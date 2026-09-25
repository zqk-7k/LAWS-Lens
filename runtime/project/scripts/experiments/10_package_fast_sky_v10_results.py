#!/usr/bin/env python3
"""Bundle the three fast v10 sky-domain diagnostics into one deliverable."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target)


def strict_pilot_snapshot(project: Path) -> dict:
    pointer = Path("/tmp/v10p1_current_output")
    if not pointer.exists():
        return {"available": False}
    output_text = pointer.read_text(encoding="utf-8").strip()
    output = Path(output_text)
    if not output.is_absolute():
        output = project / output
    complete = sorted((output / "logs").glob("*.complete.json")) if output.exists() else []
    failures = sorted((output / "logs").glob("*.failure.json")) if output.exists() else []
    case_dirs = sorted((output / "cases").glob("*_seed*")) if output.exists() else []
    return {
        "available": output.exists(),
        "path": str(output),
        "case_directories": len(case_dirs),
        "complete_markers": len(complete),
        "failure_markers": len(failures),
        "role": "strict full-PE reference pilot; independent of fast map-surrogate results",
        "included_in_package": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/gw-catalog"))
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output or project / "results" / f"sky_background_fast_results_v10_20260823_{stamp()}").resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    stages = {
        "stage1_domain_shift": args.stage1.resolve(),
        "stage2_covariate_match": args.stage2.resolve(),
        "stage3_nside512_map_pilot": args.stage3.resolve(),
    }
    for name, source in stages.items():
        copy_tree(source, output / name)

    scripts = output / "scripts"
    scripts.mkdir()
    for filename in (
        "06_sky_domain_shift_fast_audit.py",
        "07_sky_covariate_matched_null_audit.py",
        "08_nside512_map_matched_pilot.py",
        "10_package_fast_sky_v10_results.py",
    ):
        source = project / "scripts" / "experiments" / filename
        if source.exists():
            shutil.copy2(source, scripts / filename)
        elif filename == Path(__file__).name:
            shutil.copy2(Path(__file__).resolve(), scripts / filename)
        else:
            raise FileNotFoundError(source)

    stage1 = read_json(stages["stage1_domain_shift"] / "sky_domain_shift_summary.json")
    stage2 = read_json(stages["stage2_covariate_match"] / "sky_covariate_matched_null_summary.json")
    stage3 = read_json(stages["stage3_nside512_map_pilot"] / "pilot_summary.json")
    strict_status = strict_pilot_snapshot(project)
    contract = {
        "experiment": "sky_background_fast_results_v10",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_v93_modified": False,
        "paper_modified": False,
        "real_candidate_used_for_calibration": False,
        "stage1": "same-resolution domain-shift and resolution decomposition",
        "stage2": "event-level run/A90/KL/SNR covariate matching",
        "stage3": "one-seed Nside512 public-PE-template map-level validation/test and frozen rerank",
        "strict_full_pe_reference": strict_status,
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (output / "strict_full_pe_reference_status.json").write_text(json.dumps(strict_status, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# v10 天空背景快速实验总报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "本报告合并三个彼此独立、只读的快速阶段。它们不覆盖 historical v9.3，不修改论文，不使用真实候选或 PE 一致性选择模板、阈值或融合权重。结果用于定位天空通道的部署域差异，不能称为透镜探测。",
        "",
        "## 1. 直接答案",
        "",
        "历史 O3/O4a 注入天空背景与真实 GWTC 公开 PE 天空背景不匹配。该结论在双方都按 Nside=64 评分时已经成立，因此不是单纯的 64/512/1024 分辨率问题。",
        "",
        "| catalog | real q99-tail inflation vs injection null | q99.9 inflation | domain shift |",
        "|---|---:|---:|:---:|",
    ]
    for deployment in ("gwtc3", "gwtc4"):
        item = stage1["headline"][deployment]
        lines.append(
            f"| {deployment.upper()} | {item['real_tail_inflation_at_injection_q99_mean']:.3f} | "
            f"{item['real_tail_inflation_at_injection_q999_mean']:.3f} | {'YES' if item['domain_shift_flag'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## 2. 可观测协变量解释了多少",
            "",
            "仅按 run 随机抽样时 synthetic null 尾部约为 validation 预期的 1 倍。加入 A90/KL 匹配后尾部明显变重，说明旧注入平均定位更宽是主要原因之一；但仍不能完全达到真实目录尾部。",
            "",
            "| catalog | run only | matched A90+KL | matched A90+KL+SNR | observed real tail from stage 1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for deployment in ("gwtc3", "gwtc4"):
        item = stage2["headline"][deployment]
        observed = stage1["headline"][deployment]["real_tail_inflation_at_injection_q99_mean"]
        lines.append(
            f"| {deployment.upper()} | {item['run_only_random']['q99_tail_inflation_mean_of_seed_medians']:.3f} | "
            f"{item['localization']['q99_tail_inflation_mean_of_seed_medians']:.3f} | "
            f"{item['localization_and_snr']['q99_tail_inflation_mean_of_seed_medians']:.3f} | {observed:.3f} |"
        )
    lines.extend(
        [
            "",
            "剩余差异指向地图形态、多峰/弧形结构、Nside=64 旋转插值、模板重复和 PE pipeline 差异。",
            "",
            "## 3. Nside=512 map-level pilot",
            "",
            "该阶段从公开 PE posterior 直接生成 Nside=512 synthetic maps，不上采样旧 Nside=64 图；按 observing run 与 SNR 选模板，并仅在 synthetic validation 上选权重。",
            "",
            "| catalog | sky-only R@10 | three-channel retrieval R@1 | R@10 | candidate R@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for deployment in ("gwtc3", "gwtc4"):
        records = pd.DataFrame(stage3["deployments"][deployment]["test_overall_metrics"]).set_index("method")
        lines.append(
            f"| {deployment.upper()} | {records.loc['sky_only','r_at_10']:.4f} | "
            f"{records.loc['retrieval_three_channel_strict_positive','r_at_1']:.4f} | "
            f"{records.loc['retrieval_three_channel_strict_positive','r_at_10']:.4f} | "
            f"{records.loc['candidate_three_channel_strict_positive','r_at_10']:.4f} |"
        )
    lines.extend(["", "Validation-selected weights:", ""])
    for deployment in ("gwtc3", "gwtc4"):
        weights = stage3["deployments"][deployment]["selected_weights"]
        retrieval = weights["retrieval_three_channel_strict_positive"]
        candidate = weights["candidate_three_channel_strict_positive"]
        lines.append(
            f"- {deployment.upper()}: retrieval W/T/S={retrieval['waveform']}/{retrieval['time']}/{retrieval['sky']}; "
            f"candidate W/T/S={candidate['waveform']}/{candidate['time']}/{candidate['sky']}."
        )
    lines.extend(
        [
            "",
            "## 4. 科学结论",
            "",
            "1. 旧注入 null 明显低估真实 GWTC 天空正尾，因而 validation-selected sky 权重部署到真实目录时存在外推风险。",
            "2. 定位面积和 KL 分布不匹配解释了部分问题，但不是全部；不能仅靠重调权掩盖。",
            "3. Nside=512 真实后验模板 pilot 没有制造虚假的召回跃升，O3/O4a 三通道 R@10 分别约 0.519 和 0.336。",
            "4. 本轮仅一个 model seed，且 synthetic sky 仍是公开模板 surrogate；它足以支持域差异诊断，但不足以替代 v9.3 或写入论文正式数字。",
            "5. 下一项是 8--12 个快速相干天空 PE 与当前严格 full-PE smoke 的 A90、覆盖率、Z_sky 和形态对照；通过后再决定是否扩展多 seed。",
            "",
            "## 5. 目录",
            "",
            "- `stage1_domain_shift/`：原始分布、尾部、分辨率审计和图。",
            "- `stage2_covariate_match/`：500 次/seed 的事件级匹配结果。",
            "- `stage3_nside512_map_pilot/`：validation/test pair 表、权重、指标、真实 Top-100 和资源表。",
            "- `scripts/`：全部可复现脚本。",
            "",
            "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。",
        ]
    )
    (output / "SKY_BACKGROUND_FAST_RESULTS_V10_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "experiment": "sky_background_fast_results_v10",
        "stage1_headline": stage1["headline"],
        "stage2_headline": stage2["headline"],
        "stage3_deployments": stage3["deployments"],
        "strict_full_pe_reference": strict_status,
        "conclusion": "Strong synthetic-vs-real sky-background domain shift; event-level localization covariates explain part but not all; Nside512 map-matched one-seed pilot completed.",
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "sky_background_fast_results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_dir = output / "manifest"
    manifest_dir.mkdir()
    files = [path for path in output.rglob("*") if path.is_file()]
    write_csv(
        pd.DataFrame(
            [
                {"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(files)
            ]
        ),
        manifest_dir / "files_sha256.csv",
    )
    latest = project / "results" / "sky_background_fast_results_v10_LATEST.txt"
    latest.write_text(str(output) + "\n", encoding="utf-8")
    package = project / "packages" / f"{output.name}.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    package_sha = sha256_file(package)
    package.with_suffix(package.suffix + ".sha256").write_text(f"{package_sha}  {package.name}\n", encoding="ascii")
    print(json.dumps({"output": str(output), "package": str(package), "sha256": package_sha}, indent=2))


if __name__ == "__main__":
    main()
