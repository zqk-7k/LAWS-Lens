#!/usr/bin/env python3
"""Choose a disclosed development candidate once, before fresh confirmation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev


def run(root):
    destination = root / "contracts/FINAL_CANDIDATE_BEFORE_FRESH_TEST.json"
    if destination.exists():
        raise RuntimeError("Final candidate already frozen; no reselection")
    if (root / "confirmation").exists():
        raise RuntimeError("Cannot select after fresh data generation")
    ev.summarize(root)
    records, eligible = [], []
    for config in body.CONFIGS:
        passing = 0
        metrics = []
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            path = root / f"evaluation/{config}/gwtc3/model_{ms}_eval_{es}/retrieval_pair_metrics.csv"
            if not path.exists():
                raise RuntimeError(f"Incomplete O3 comparison: {path}")
            frame = pd.read_csv(path).query("split == 'test'")
            base = {r.method: r._asdict() for r in frame.query("rule == 'BASELINE'").itertuples()}
            cand = {r.method: r._asdict() for r in frame.query("rule == 'VAL-ENSEMBLE'").itertuples()}
            passing += int(ev.guard(cand, base))
            metrics.append(cand["C_fixed"])
        budget = pd.read_csv(root / f"results/{config}--VAL-ENSEMBLE/gwtc3/pe_official_budget.csv")
        budget = budget.loc[budget.seed.astype(str).eq("consensus") & budget.budget.eq(10)]
        indexed = budget.set_index("method")
        # Column names come from the shared frozen PE audit, not recomputed labels.
        records.append({"config": config, "passing_development_seeds": passing,
                        "budget": indexed.to_dict("index"),
                        "mean_Cfixed_F50": sum(m["false_at_recall_0p5"] for m in metrics)/3})
        if passing == 3 and indexed.catastrophic_mc.eq(0).all() and indexed.n_pe_valid.eq(10).all():
            key = (-int(indexed.BC_mc_ge_0p5.min()), -int(indexed.BC_mc_ge_0p5.sum()),
                   -int(indexed.Dmax_le_3.sum()), records[-1]["mean_Cfixed_F50"], config)
            eligible.append((key, config))
    dev.json_write(root / "audit/CANDIDATE_SELECTION_INPUTS.json", records)
    print(json.dumps(records, ensure_ascii=False), flush=True)
    if not eligible:
        raise RuntimeError("No candidate meets the disclosed development gates; do not generate confirmation")
    chosen = min(eligible)[1]
    files = []
    for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
        files += [root / f"models/{chosen}/gwtc3/seed_{ms}/validation_selected_model.pt",
                  root / f"evaluation/{chosen}/gwtc3/model_{ms}_eval_{es}/SELECTED_CONFIG.json"]
        for dep in ("gwtc3", "gwtc4"):
            files.extend((dev.V7 / dep / f"seed_{es}/waveform_gate").glob("*/validation_selected_model.pt"))
            files.append(dev.V7 / dep / f"seed_{es}/results/waveform_channel_calibration_v7.json")
    files += [dev.BAY / "contracts/selected_config.json"]
    files += [dev.V7 / dep / "shared/time_delay_likelihood_ratio.json" for dep in ("gwtc3", "gwtc4")]
    dev.json_write(destination, {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "O3_config": chosen, "O3_rule": "VAL-ENSEMBLE",
        "O4_config": "CFIX-baseline", "O4_rule": "original frozen encoder; new O4 models did not preserve all guardrails",
        "selection_basis": "explicit reused-development and observed-real-O3 audit; NOT a blind selection",
        "selection_rule": "all3 reused-development injection guards; zero catastrophic consensus waveform/Cfixed Top10 with completePE; maximize minimum and total BC_Mc>=0.5 counts, then Dmax counts, then minimize Cfixed F50, fixed lexical tie; official overlap not used",
        "no_fresh_data_examined": True,
        "new_population_confirmation": False,
        "fresh_confirmation_next": "fresh BBH sources and disjoint actual noise blocks conditional on reused lens environments",
        "time_sky_weights_unchanged": True,
        "frozen_files": [{"path": str(p), "sha256": dev.sha(p)} for p in sorted(set(files))],
        "status": "CANDIDATE_FROZEN_PENDING_FRESH_CONFIRMATION_NOT_GOAL_COMPLETE"})
    print(json.dumps({"frozen_candidate": chosen, "O4": "original-C-fixed", "goal_achieved": False}), flush=True)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    run(args.root)
