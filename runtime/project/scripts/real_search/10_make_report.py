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
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    data = df[cols].head(n).copy() if n is not None else df[cols].copy()
    return data.to_markdown(index=False)


def update_reproduce(run_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "RUN_DIR=\"${1:-runs/real_gwtc_lensing_search_20260625}\"",
        "/root/miniconda3/bin/python scripts/real_search/00_build_event_manifest.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/01_build_healpix_overlap.py --run-dir \"$RUN_DIR\" --nside 512",
        "/root/miniconda3/bin/python scripts/real_search/02_build_observable_features.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/03_score_observable_baseline.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/04_make_real_noise_injection_dataset.py --run-dir \"$RUN_DIR\" --download",
        "/root/miniconda3/bin/python scripts/real_search/04b_materialize_real_noise_match_dataset.py --run-dir \"$RUN_DIR\" --samples-per-family 600",
        "/root/miniconda3/bin/python scripts/real_search/05_train_real_noise_encoder.py --run-dir \"$RUN_DIR\" --train-missing --epochs 20 --samples-per-family 600",
        "/root/miniconda3/bin/python scripts/real_search/06_embed_real_gwtc_events.py --run-dir \"$RUN_DIR\" --epochs 20",
        "/root/miniconda3/bin/python scripts/real_search/07_fuse_real_pair_scores.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/08_background_and_sensitivity.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/09_lvk_crosscheck.py --run-dir \"$RUN_DIR\"",
        "/root/miniconda3/bin/python scripts/real_search/10_make_report.py --run-dir \"$RUN_DIR\"",
    ]
    path = run_dir / "reproduce.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    args = parser.parse_args()
    run_dir = args.run_dir
    ensure_run_dirs(run_dir)

    events = pd.read_csv(run_dir / "data" / "event_manifest.csv")
    strain = pd.read_csv(run_dir / "data" / "strain_gwosc_download_manifest.csv") if (run_dir / "data" / "strain_gwosc_download_manifest.csv").exists() else pd.DataFrame()
    sky = pd.read_csv(run_dir / "data" / "skymap_manifest.csv")
    scores = pd.read_parquet(run_dir / "results" / "real_pair_scores.parquet")
    shortlist = pd.read_csv(run_dir / "results" / "candidate_shortlist.csv")
    cross = pd.read_csv(run_dir / "results" / "candidate_crosscheck_lvk.csv")
    gate = read_json(run_dir / "results" / "waveform_gate1.json")
    embed = read_json(run_dir / "features" / "real_waveform_embedding_summary.json")
    bg = pd.read_csv(run_dir / "results" / "background_null_summary.csv")
    inj = pd.read_csv(run_dir / "results" / "injection_recovery_summary.csv")
    weights = read_json(run_dir / "results" / "channel_weights.json")
    materialized = read_json(run_dir / "data" / "real_noise_injections" / "materialized_dataset_summary.json")
    phase_e = read_json(run_dir / "data" / "real_noise_injections" / "phase_e_summary.json")

    primary = events[(events["include_in_primary_search"] == True) & (events["sky_map_available"] == True)]
    h1l1_events = 0
    if not strain.empty:
        ok = strain[(strain["detector"].isin(["H1", "L1"])) & (strain["download_status"].isin(["already_exists", "downloaded"]))]
        h1l1_events = int((ok.groupby("event_name")["detector"].nunique() >= 2).sum())
    gw_rank = int(cross.iloc[0]["rank"]) if not cross.empty and "rank" in cross else -1

    top_cols = ["rank", "event_i", "event_j", "final_score", "delta_t_days", "sky_cosine_overlap", "snr_ratio", "empirical_catalog_tail_p"]
    gate_table_path = run_dir / "results" / "waveform_gate1_metrics.csv"
    gate_table = pd.read_csv(gate_table_path) if gate_table_path.exists() else pd.DataFrame()
    gate_cols = ["family", "split", "r_at_1", "r_at_5", "r_at_10", "candidate_pair_recall_top10", "first_loss", "last_loss"]

    cn = [
        "# Confirmed-event GWTC Catalog-level Lensing Search Report",
        "",
        f"生成时间：{utc_now()}",
        "",
        "## 结论先行",
        "",
        "本次运行完成了 O1-O3 confident BBH 的真实目录候选筛选流程，输出对象是 `candidate shortlist for Bayesian follow-up`。本报告不声称发现真实强透镜引力波事件。",
        "",
        "主候选排序采用真实可观测通道：真实事件时间差、真实 HEALPix sky posterior overlap、以及 network SNR ratio。real-noise waveform encoder 已完成 Gate-1 注入验证；真实事件 embedding 已导出为可审计通道，但默认不进入最终排序权重，因为尚未完成真实事件 waveform 通道的 validation-selected 融合权重。",
        "",
        "## 数据范围",
        "",
        f"- manifest 总事件数：{len(events)}",
        f"- primary scored O1-O3 confident BBH 事件数：{len(primary)}",
        f"- primary unordered pairs：{len(scores)}",
        f"- primary HEALPix sky maps 可用数：{int(sky[sky['include_in_primary_search'] == True]['sky_map_available'].sum())}",
        f"- 已下载/定位 H1+L1 4096s GWOSC strain 的 primary 事件数：{h1l1_events}",
        f"- off-source real-noise segments：{phase_e.get('noise_segments', 'NA')}",
        "",
        "O4/GWTC-5 事件目前只作为 manifest extension，不进入 primary scoring；原因是该阶段优先使用 PE release 的真实 HEALPix posterior maps，而不是 search skymap surrogate。",
        "",
        "## Scoring 与融合",
        "",
        "- `time_score`：Liao/GW-LMC lensed delay prior 对真实 catalog unordered pair delay background 的 empirical log-likelihood ratio。",
        "- `sky_score`：真实 HEALPix posterior maps 的 log cosine overlap；没有使用 A90 surrogate。",
        "- `snr_score`：Liao lensed SNR-ratio prior 对真实 catalog SNR-ratio background 的 empirical log-likelihood ratio。",
        "- `final_score`：各通道 row-standardized 后等权求和，再取 directed score 的 max。",
        f"- waveform weight：{weights.get('weights', {}).get('waveform_score', 0.0)}；策略：{weights.get('waveform_policy', '')}",
        "",
        "## Top 20 Candidate Shortlist",
        "",
        md_table(shortlist, top_cols, 20),
        "",
        "这些 pair 只能作为后续 Bayesian lensing follow-up 的候选，不是检测声明。",
        "",
        "## LVK Cross-check",
        "",
        md_table(cross, [c for c in ["event_i", "event_j", "rank", "final_score", "delta_t_days", "sky_cosine_overlap", "snr_ratio", "empirical_catalog_tail_p", "lvk_context"] if c in cross.columns]),
        "",
        f"GW170104--GW170814 rank：{gw_rank}",
        "",
        "## Waveform Gate-1",
        "",
        f"- Gate passed：{gate.get('passed', 'NA')}",
        f"- macro test R@10：{gate.get('macro_test_r_at_10', 'NA')}",
        f"- min-family test R@10：{gate.get('min_family_test_r_at_10', 'NA')}",
        f"- real-noise injection metadata rows：{materialized.get('metadata_rows', 'NA')}",
        f"- real-event embedded events：{embed.get('n_embedded_events', 'NA')} / {embed.get('n_primary_events', 'NA')}",
        "",
        md_table(gate_table, gate_cols) if not gate_table.empty else "_No Gate table._",
        "",
        "## 背景统计与注入灵敏度",
        "",
        "背景统计使用真实 catalog 全部 unordered non-self pairs 作为 empirical null context；这不是 LVK full FAR。",
        "",
        md_table(bg, [c for c in ["summary_type", "metric", "quantile", "value", "n_pairs"] if c in bg.columns], 12),
        "",
        "注入表当前是 domain-matched waveform Gate-1 held-out retrieval，不是完整 population-rate sensitivity campaign。",
        "",
        md_table(inj, [c for c in ["experiment", "family", "split", "r_at_1", "r_at_5", "r_at_10", "candidate_pair_recall_top10"] if c in inj.columns]),
        "",
        "## 主要限制",
        "",
        "- 没有 claim 任何真实 GWTC pair 是 lensed。",
        "- 当前 final shortlist 是 observable-first ranking；waveform real-event score 已导出但默认权重为 0。",
        "- 背景 p-value 是目录内部 empirical tail context，不等价于 LVK false-alarm rate。",
        "- 注入灵敏度还不是完整端到端 Bayesian follow-up 灵敏度。",
    ]
    (run_dir / "real_search_report_cn.md").write_text("\n".join(cn) + "\n", encoding="utf-8")

    en = [
        "# Confirmed-event GWTC Catalog-level Lensing Search Report",
        "",
        f"Generated at: {utc_now()}",
        "",
        "This run produces a candidate shortlist for Bayesian follow-up only. It does not claim any real GWTC pair is lensed.",
        "",
        f"- Manifest events: {len(events)}",
        f"- Primary scored O1-O3 confident BBH events: {len(primary)}",
        f"- Unordered primary pairs: {len(scores)}",
        f"- Primary PE HEALPix maps: {int(sky[sky['include_in_primary_search'] == True]['sky_map_available'].sum())}",
        f"- Primary events with downloaded/local H1+L1 4096 s GWOSC strain: {h1l1_events}",
        f"- GW170104--GW170814 rank: {gw_rank}",
        "",
        "The final score uses observable channels: event-time delay, real HEALPix sky-posterior overlap, and network-SNR ratio. The real-noise waveform Gate-1 test passed on held-out injections, and real-event waveform embeddings were exported for audit, but waveform has zero default weight in the final real-catalog shortlist until a validation-selected real-event fusion weight is established.",
        "",
        "## Top 20",
        "",
        md_table(shortlist, top_cols, 20),
        "",
        "## Waveform Gate-1",
        "",
        md_table(gate_table, gate_cols) if not gate_table.empty else "_No Gate table._",
        "",
        "## Limitations",
        "",
        "The empirical background is catalog-context only and is not an LVK FAR. The injection summary is a domain-matched waveform Gate-1 retrieval test, not a full population sensitivity campaign.",
    ]
    (run_dir / "real_search_report_en.md").write_text("\n".join(en) + "\n", encoding="utf-8")
    update_reproduce(run_dir)
    print(json.dumps({"report_cn": str(run_dir / "real_search_report_cn.md"), "report_en": str(run_dir / "real_search_report_en.md"), "gw170104_gw170814_rank": gw_rank}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
