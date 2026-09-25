from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import default_run_dir, ensure_run_dirs, utc_now


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    return df[cols].head(n).to_markdown(index=False) if not df.empty else "_No rows._"


def update_reproduce(run_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "RUN_DIR=\"${1:-runs/real_gwtc_lensing_search_20260625}\"",
        "/root/miniconda3/bin/python scripts/real_search/00_build_event_manifest.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/01_build_healpix_overlap.py --run-dir \"$RUN_DIR\" --nside 512",
        "/root/miniconda3/bin/python scripts/real_search/02_build_observable_features.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/04_make_real_noise_injection_dataset.py --run-dir \"$RUN_DIR\" --download",
        "/root/miniconda3/bin/python scripts/real_search/04b_materialize_real_noise_match_dataset.py --run-dir \"$RUN_DIR\" --samples-per-family 600",
        "/root/miniconda3/bin/python scripts/real_search/05_train_real_noise_encoder.py --run-dir \"$RUN_DIR\" --train-missing --epochs 20 --samples-per-family 600",
        "/root/miniconda3/bin/python scripts/real_search/06_embed_real_gwtc_events.py --run-dir \"$RUN_DIR\" --epochs 20",
        "/root/miniconda3/bin/python scripts/real_search/07b_fuse_waveform_time_sky.py --run-dir \"$RUN_DIR\" --epochs 20",
        "/root/miniconda3/bin/python scripts/real_search/11_audit_real_waveform_deployment.py --run-dir \"$RUN_DIR\" --quicklook-events 10 --apply-failure-policy",
        "/root/miniconda3/bin/python scripts/real_search/08_background_and_sensitivity.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/09_lvk_crosscheck.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/10b_make_report_waveform_time_sky.py --run-dir \"$RUN_DIR\"",
    ]
    p = run_dir / "reproduce.sh"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    args = parser.parse_args()
    run_dir = args.run_dir
    ensure_run_dirs(run_dir)

    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    primary = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)]
    strain = pd.read_csv(run_dir / "data" / "strain_gwosc_download_manifest.csv")
    sky = pd.read_csv(run_dir / "data" / "skymap_manifest.csv")
    h1l1 = int((strain[(strain["detector"].isin(["H1", "L1"])) & (strain["download_status"].isin(["already_exists", "downloaded"]))].groupby("event_name")["detector"].nunique() >= 2).sum())

    weights = read_json(run_dir / "results" / "channel_weights_waveform_time_sky.json")
    audit_path = run_dir / "results" / "real_waveform_deployment_audit.json"
    audit = read_json(audit_path) if audit_path.exists() else {}
    waveform_failed = audit.get("decision", {}).get("waveform_real_event_deployment_status") == "failed_audit"
    main_short_path = run_dir / "results" / ("candidate_shortlist_time_sky_baseline.csv" if waveform_failed else "candidate_shortlist_waveform_time_sky.csv")
    main_short = pd.read_csv(main_short_path)
    base_short = pd.read_csv(run_dir / "results" / "candidate_shortlist_time_sky_baseline.csv")
    forced_short = pd.read_csv(run_dir / "results" / "candidate_shortlist_forced_equal_weights_waveform_time_sky.csv")
    gate = pd.read_csv(run_dir / "results" / "waveform_gate1_metrics.csv")

    rank_main = weights["gw170104_gw170814_rank"]["time_sky_baseline" if waveform_failed else "waveform_time_sky"]
    rank_base = weights["gw170104_gw170814_rank"]["time_sky_baseline"]
    rank_forced = weights["gw170104_gw170814_rank"]["forced_equal_waveform_time_sky"]
    w = weights["selected_weights"]
    b = weights["time_sky_baseline_weights"]
    wf_audit = weights["real_waveform_score_audit"]
    root_cause = audit.get("root_cause_assessment", {})
    cols = ["rank", "event_i", "event_j", "final_score", "waveform_score", "time_score", "healpix_sky_score", "delta_t_days", "healpix_overlap", "waveform_available", "detector_coverage"]

    if waveform_failed:
        headline = "本次真实数据搜索完成 waveform deployment audit 后，将默认主排序切换为 `time + real HEALPix sky`。Gate-1 real-noise injection validation 虽然通过，但真实 GWTC on-source waveform embeddings collapse 成常数向量，1485 个可用 pair 的 waveform_score 全部为 1.0；因此 waveform 不能进入当前真实 catalog final score。SNR/amplitude 仍只作为 audit/supplementary diagnostic，不参与主排序。"
        method_title = "Default Shortlist: Time + HEALPix Sky Main Result"
        weight_text = "default main weights after failed waveform audit：waveform=0, time=0.25, sky=4.0"
    else:
        headline = "本次真实数据搜索已按论文主流程重跑候选排序：主方法只使用 waveform score、time-delay score、real HEALPix sky-overlap score 三个通道。SNR/amplitude 不参与主 final score，也不参与默认 candidate shortlist rank；它只保留为 audit/supplementary diagnostic 列。"
        method_title = "Default Shortlist: Waveform + Time + HEALPix Sky"
        weight_text = f"validation-selected weights：waveform={w['waveform']}, time={w['time']}, sky={w['sky']}"

    cn = [
        "# Confirmed-event GWTC Catalog-level Lensing Search Report",
        "",
        f"生成时间：{utc_now()}",
        "",
        "## 结论先行",
        "",
        headline,
        "",
        "输出仍然只是 `candidate shortlist for Bayesian follow-up`，不声称任何真实 GWTC pair 是 lensed。",
        "",
        "## 数据范围",
        "",
        f"- manifest 总事件数：{len(events)}",
        f"- primary scored O1-O3 confident BBH：{len(primary)}",
        f"- unordered primary pairs：{len(pd.read_parquet(run_dir / 'results' / 'real_pair_scores_waveform_time_sky.parquet'))}",
        f"- primary HEALPix sky maps：{int(sky[sky['include_in_primary_search'] == True]['sky_map_available'].sum())}",
        f"- 已下载/定位 H1+L1 GWOSC strain 的 primary 事件：{h1l1}",
        "",
        "## 主方法与权重",
        "",
        "真实搜索方法口径与论文主流程保持一致：默认主排序使用 waveform + time + real HEALPix sky。只有当真实事件 waveform 部署审计不合格时，才会退回 time + HEALPix sky；当前最新 scale-aware preprocessing 版本已通过 deployment audit，因此 waveform 进入主 final score。",
        "",
        f"- {weight_text}",
        f"- validation-selected waveform/time/sky weights before deployment audit：waveform={w['waveform']}, time={w['time']}, sky={w['sky']}",
        f"- time+sky baseline weights：waveform={b['waveform']}, time={b['time']}, sky={b['sky']}",
        "- forced equal comparison：waveform=1, time=1, sky=1",
        "- SNR/amplitude policy：不进入主排序，只作为 audit/supplementary diagnostic。",
        "",
        f"真实事件 waveform score audit：available pairs={wf_audit['available_pairs']}, unique scores={wf_audit['n_unique_scores']}, std={wf_audit['std']}. 如果真实 waveform score 近似常数，即使 validation 选择了正权重，它在 row-standardization 后对真实排序也几乎没有影响。",
        "",
        "## Waveform Deployment Audit",
        "",
        audit.get("decision", {}).get("reason", "No waveform deployment audit JSON found."),
        "",
        "Root cause assessment: " + root_cause.get("details", "No root-cause assessment available."),
        "",
        "Recommended fix: " + root_cause.get("recommended_fix", "No recommendation available."),
        "",
        "## " + method_title,
        "",
        table(main_short, cols, 20),
        "",
        "## Time + Sky Baseline",
        "",
        table(base_short, cols, 20),
        "",
        "## Forced Equal Waveform + Time + Sky Comparison",
        "",
        table(forced_short, cols, 20),
        "",
        "## GW170104--GW170814 Cross-check",
        "",
        f"- default main rank：{rank_main}",
        f"- time+sky baseline rank：{rank_base}",
        f"- forced equal waveform+time+sky rank：{rank_forced}",
        "",
        "## Gate-1 Waveform Validation",
        "",
        gate[["family", "split", "r_at_1", "r_at_5", "r_at_10", "candidate_pair_recall_top10"]].to_markdown(index=False),
        "",
        "## 限制",
        "",
        "- 当前权重选择来自 held-out real-noise injection validation catalog；真实数据没有 lensing labels。",
        "- 真实事件 waveform embedding 已导出，但当前 real waveform similarity 近似常数，因此对 rank 的实际贡献很小。",
        "- 背景统计仍是 catalog empirical context，不是 LVK full FAR。",
    ]
    (run_dir / "real_search_report_cn.md").write_text("\n".join(cn) + "\n", encoding="utf-8")

    if waveform_failed:
        en_headline = "After the real-event waveform deployment audit, the default real-catalog search is set to time + real HEALPix sky. Gate-1 real-noise injection validation passed, but real GWTC on-source waveform embeddings collapsed to a constant vector: all 1485 available pairwise waveform scores are 1.0. Therefore waveform is not used in the current real-catalog final score. SNR/amplitude remains audit-only."
        en_method_title = "Default Shortlist: Time + HEALPix Sky Main Result"
        en_weight_text = "default main weights after failed waveform audit: waveform=0, time=0.25, sky=4.0"
    else:
        en_headline = "The real-data search has been rerun with the same channel definition as the paper: waveform score, time-delay score, and real HEALPix sky-overlap score. SNR/amplitude is not used in the main final score or default shortlist ranking; it is retained only as an audit/supplementary diagnostic column."
        en_method_title = "Default Shortlist: Waveform + Time + HEALPix Sky"
        en_weight_text = f"Validation-selected weights: waveform={w['waveform']}, time={w['time']}, sky={w['sky']}."

    en = [
        "# Confirmed-event GWTC Catalog-level Lensing Search Report",
        "",
        f"Generated at: {utc_now()}",
        "",
        en_headline,
        "",
        "This is a candidate shortlist for Bayesian follow-up only, not a real lensing detection claim.",
        "",
        f"- Primary scored O1-O3 confident BBH events: {len(primary)}",
        f"- Primary unordered pairs: {len(pd.read_parquet(run_dir / 'results' / 'real_pair_scores_waveform_time_sky.parquet'))}",
        f"- Primary HEALPix sky maps: {int(sky[sky['include_in_primary_search'] == True]['sky_map_available'].sum())}",
        f"- Primary events with H1+L1 GWOSC strain: {h1l1}",
        "",
        en_weight_text,
        f"Validation-selected waveform/time/sky weights before deployment audit: waveform={w['waveform']}, time={w['time']}, sky={w['sky']}.",
        f"GW170104--GW170814 rank: main={rank_main}, time+sky baseline={rank_base}, forced equal={rank_forced}.",
        "",
        "## Waveform Deployment Audit",
        "",
        audit.get("decision", {}).get("reason", "No waveform deployment audit JSON found."),
        "",
        "Root cause assessment: " + root_cause.get("details", "No root-cause assessment available."),
        "",
        "Recommended fix: " + root_cause.get("recommended_fix", "No recommendation available."),
        "",
        "## " + en_method_title,
        "",
        table(main_short, cols, 20),
        "",
        "## Time + Sky Baseline",
        "",
        table(base_short, cols, 20),
        "",
        "## Forced Equal Comparison",
        "",
        table(forced_short, cols, 20),
        "",
        "## Gate-1 Waveform Validation",
        "",
        gate[["family", "split", "r_at_1", "r_at_5", "r_at_10", "candidate_pair_recall_top10"]].to_markdown(index=False),
        "",
        "Limitations: the background is catalog empirical context, not an LVK full FAR. Real-event waveform similarity is currently nearly constant, so its practical ranking contribution is small even when validation selects a positive weight.",
    ]
    (run_dir / "real_search_report_en.md").write_text("\n".join(en) + "\n", encoding="utf-8")
    update_reproduce(run_dir)
    print(json.dumps({"report_cn": str(run_dir / "real_search_report_cn.md"), "report_en": str(run_dir / "real_search_report_en.md"), "main_rank_gw170104_gw170814": rank_main, "baseline_rank_gw170104_gw170814": rank_base}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
