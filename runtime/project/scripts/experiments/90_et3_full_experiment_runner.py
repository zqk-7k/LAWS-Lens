from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ET3_MATCH_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
FRESH_ROOT = Path("runs/et3_fresh50_full_catalog_20260616")
RERANK_ROOT = Path("runs/et3_liao_realistic_p1_p2_rerank_20260616")


def configure_modules():
    base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
    fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")
    liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")
    pdf = importlib.import_module("scripts.experiments.89_pdf_rule_time_sky_baseline")

    base.ROOTS[("SIS", "ET3")] = ET3_MATCH_ROOT
    base.ROOTS[("PM", "ET3")] = ET3_MATCH_ROOT

    fresh.OUT_ROOT = FRESH_ROOT
    fresh.ENCODER_ROOT = FRESH_ROOT / "fresh_mixed_encoders"
    fresh.JOBS = [("ET3", "pure"), ("ET3", "noisy")]

    liao.OUT_ROOT = RERANK_ROOT
    liao.ENCODER_ROOT = fresh.ENCODER_ROOT
    liao.JOBS = [("ET3", "noisy")]

    pdf.OUT_DIR = RERANK_ROOT / "stage2b_pdf_rule_time_sky_baseline"
    pdf.DOC_PATH = Path("docs/stage2b_pdf_rule_time_sky_baseline_et3_20260616_cn.md")

    return fresh, liao, pdf


def summarize_outputs() -> None:
    rows = []
    for csv in sorted([*FRESH_ROOT.glob("**/*summary.csv"), *RERANK_ROOT.glob("**/*summary.csv")]):
        try:
            df = pd.read_csv(csv)
            rows.append({"path": str(csv), "rows": int(len(df)), "columns": ",".join(df.columns[:8])})
        except Exception as exc:
            rows.append({"path": str(csv), "rows": -1, "error": str(exc)})
    out = RERANK_ROOT / "et3_run_outputs_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ET3 fresh50 and realistic rerank experiments.")
    parser.add_argument(
        "--phase",
        choices=["fresh", "p0", "pdf", "p1p2", "true_sky", "all"],
        default="all",
        help="fresh trains/evaluates ET3 pure+noisy encoders; p0 runs stages 0-3; pdf runs PDF hard-mask baseline; p1p2 runs stages 4-6; true_sky runs oracle true-ra/dec ablation.",
    )
    args = parser.parse_args()
    fresh, liao, pdf = configure_modules()
    t0 = time.perf_counter()

    if args.phase in {"fresh", "all"}:
        fresh.main()
    if args.phase in {"p0", "all"}:
        liao.stage0_baseline()
        liao.stage1_liao_time()
        liao.stage2_observed_sky()
        liao.stage3_liao_time_plus_observed_sky()
    if args.phase in {"pdf", "all"}:
        pdf.run()
    if args.phase in {"p1p2", "all"}:
        liao.stage4_snr_amplitude_prior()
        liao.stage5_reranker_model_compare()
        liao.stage6_catalog_graph_discovery()
    if args.phase in {"true_sky", "all"}:
        true_sky = importlib.import_module("scripts.experiments.95_et3_true_sky_combinations")
        true_sky.run()

    summarize_outputs()
    RERANK_ROOT.mkdir(parents=True, exist_ok=True)
    (RERANK_ROOT / f"et3_{args.phase}_protocol_summary.json").write_text(json.dumps({
        "phase": args.phase,
        "elapsed_s": float(time.perf_counter() - t0),
        "match_root": str(ET3_MATCH_ROOT),
        "fresh_root": str(FRESH_ROOT),
        "rerank_root": str(RERANK_ROOT),
        "jobs": {"fresh": fresh.JOBS, "rerank": liao.JOBS},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
