from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import default_run_dir, ensure_run_dirs, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    args = parser.parse_args()
    run_dir = args.run_dir
    ensure_run_dirs(run_dir)

    weights_path = run_dir / "results" / "channel_weights_waveform_time_sky.json"
    audit_path = run_dir / "results" / "real_waveform_deployment_audit.json"
    scores_path = run_dir / "results" / "real_pair_scores_waveform_time_sky.parquet"
    shortlist_path = run_dir / "results" / "candidate_shortlist_waveform_time_sky.csv"
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scores = pd.read_parquet(scores_path)
    shortlist = pd.read_csv(shortlist_path)

    summary = {
        "generated_at_utc": utc_now(),
        "status": "current_authoritative_scaleaware_waveform_time_sky",
        "method": "waveform + time-delay + real HEALPix sky-overlap",
        "snr_policy": "SNR/amplitude is audit-only and is not used in final_score or shortlist rank.",
        "n_events": int(max(scores["idx_i"].max(), scores["idx_j"].max()) + 1),
        "n_pairs": int(len(scores)),
        "n_waveform_pairs_available": int(scores["waveform_available"].fillna(False).sum()),
        "selected_weights": weights["selected_weights"],
        "time_sky_baseline_weights": weights["time_sky_baseline_weights"],
        "waveform_deployment_status": audit["decision"]["waveform_real_event_deployment_status"],
        "real_waveform_score_audit": weights["real_waveform_score_audit"],
        "gw170104_gw170814_rank": weights["gw170104_gw170814_rank"],
        "top20": shortlist.head(20).to_dict(orient="records"),
        "authoritative_files": [
            "results/real_waveform_deployment_audit.json",
            "results/channel_weights_waveform_time_sky.json",
            "results/candidate_shortlist_waveform_time_sky.csv",
            "results/real_pair_scores_waveform_time_sky.parquet",
            "results/waveform_gate1_metrics.csv",
            "real_search_report_cn.md",
            "real_search_report_en.md",
        ],
    }
    write_json(run_dir / "results" / "final_fusion_summary.json", summary)

    readme = f"""# Current Authoritative Results

Generated at: {utc_now()}

This run directory contains historical intermediate files from earlier attempts. For the current scale-aware preprocessing run, cite the files listed below.

## Current Method

- Main ranking channels: waveform score, time-delay score, real HEALPix sky-overlap score.
- SNR/amplitude: audit/supplementary diagnostic only; not used in `final_score` or shortlist rank.
- Waveform deployment audit: `{audit['decision']['waveform_real_event_deployment_status']}`.
- Validation-selected weights: `waveform={weights['selected_weights']['waveform']}`, `time={weights['selected_weights']['time']}`, `sky={weights['selected_weights']['sky']}`.

## Authoritative Files

- `results/real_waveform_deployment_audit.json`
- `results/channel_weights_waveform_time_sky.json`
- `results/candidate_shortlist_waveform_time_sky.csv`
- `results/real_pair_scores_waveform_time_sky.parquet`
- `results/final_fusion_summary.json`
- `results/waveform_gate1_metrics.csv`
- `real_search_report_cn.md`
- `real_search_report_en.md`

## Compatibility Copies

- `candidate_shortlist.csv`
- `results/candidate_shortlist.csv`
- `real_pair_scores.parquet`
- `results/real_pair_scores.parquet`
- `results/channel_weights.json`

These have been refreshed to match the current waveform+time+sky result, but the explicit `*_waveform_time_sky.*` names are preferred for citation.

## Historical Files

Files with names from older pipeline stages should not be cited unless they match the current summary above. The `final_fusion_summary.json` in this package has been regenerated from the current scale-aware waveform+time+sky result.
"""
    (run_dir / "README_CURRENT_RESULTS.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"summary": str(run_dir / "results" / "final_fusion_summary.json"), "readme": str(run_dir / "README_CURRENT_RESULTS.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
