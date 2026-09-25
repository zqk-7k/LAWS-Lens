#!/usr/bin/env python3
"""Independent waveform/Mc development audit. Historical inputs are read-only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

os.environ["GW_WAVEFORM_INPUT_SAMPLES"] = "4096"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy import stats
import torch

PROJECT = Path("/root/autodl-tmp/gw-catalog")
MAIN = PROJECT / "results/main_o3official_cfixed_v1_20260904_20260904T072435Z"
BAY = PROJECT / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
OLD = PROJECT / "results/waveform_domain_multiscale_exploratory_20260903_20260903T063500Z"
V7 = PROJECT / "results/real_noise_injection_v7_peak2s_formal_20260722"
SEEDS = (202607241, 202607242, 202607243)
FINAL_STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


ORCH = module(OLD / "scripts/waveform_domain_multiscale_exploratory.py", "mcwf_old_orch")
TRAIN = ORCH.TRAIN
BASE = ORCH.BASE
MAINCODE = module(PROJECT / "scripts/experiments/main_o3official_cfixed_v1.py", "mcwf_maincode")
torch.set_num_threads(4)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    temp.replace(path)


def csv_write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def initialize(root):
    if (root / "contracts/DEVELOPMENT_CONTRACT.json").exists():
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("New output directory required")
    for p in ("contracts", "scripts", "tables", "results", "audit", "cache", "logs", "reports", "figures", "models", "configs", "manifest"):
        (root / p).mkdir(parents=True, exist_ok=True)
    packages = {
        "main_o3official_cfixed_v1p2_20260904_deliverables.tar.gz": "a27ec8bbfec18b481c391b8f2a7c35ca872989a2a70f5456760bfe5ea6324545",
        "bayestar_injection_sky_pe_20260901_20260901T102000Z_deliverables.tar.gz": "3ea3cf13992a5154e0dd757ab6087b73344cf28eb0ccf5e4a8a903b0e83850af",
    }
    protected = list(MAIN.glob("results/**/*.parquet")) + list(MAIN.glob("results/**/*.csv"))
    protected += list(BAY.glob("results/**/*.parquet")) + list(BAY.glob("results/*.csv"))
    for dep in ("gwtc3", "gwtc4"):
        for seed in SEEDS:
            protected.extend((V7 / dep / f"seed_{seed}/results").glob("*calibration*.json"))
            protected.extend((V7 / dep / f"seed_{seed}/waveform_gate").glob("*/validation_selected_model.pt"))
    for filename, digest in packages.items():
        p = PROJECT / "packages" / filename
        if sha(p) != digest:
            raise RuntimeError(f"Baseline hash mismatch: {p}")
        protected.append(p)
    csv_write(root / "manifest/PROTECTED_INPUT_SHA256.csv", pd.DataFrame([
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(set(protected))
    ]))
    contract = {
        "experiment_code": "MAIN-O3-MCWF-DEV-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "MAIN-O3OFFICIAL-CFIX-v1.2 (science v1)",
        "status": "DEVELOPMENT_ACTIVE",
        "final_status": FINAL_STATUS,
        "author_priority": ["reduce catastrophic high-waveform/Mc discordance", "describe official follow-up overlap", "preserve injection retrieval and false burden"],
        "real_O3_role": "explicit development audit; feedback permitted but not an unbiased held-out scientific test",
        "real_O4_role": "run-matched transfer audit; historical outcomes already observed, not pristine blind confirmation",
        "historical_test_role": "reused development comparator; never described as new locked confirmation",
        "network_scoring_inputs": "strain-derived quantities only; no real PE, event ID, official FPP or official candidate labels",
        "frozen": ["one-dimensional time_score", "BAYESTAR/public-PE sky_raw_log_bf", "Nside512 NESTED-to-RING", "real scope", "initial per-seed C-fixed fusion weights"],
        "initial_model_seeds": [202609051, 202609052, 202609053],
        "audit_metrics": {
            "catastrophic": "BC_Mc<0.1 OR D_Mc>5; missing PE separately counted",
            "budgets": [10, 20, 50, 100],
            "also_report": ["BC_Mc>=0.5", "D_Mc<=3", "Dmax<=3", "all-pair descriptive Spearman", "event-block uncertainty", "official PO/ML and follow-up stages"],
        },
        "injection_guardrails": {"r10_noninferiority_margin": 0.02, "auprc_noninferiority_margin": 0.005, "false_burden_relative_limit": 1.10},
        "guardrail_interpretation": "exploratory engineering tolerances; report exact deltas and confidence intervals, not a scientific theorem",
        "rounds": ["input/target/calibration audit", "reuse existing checkpoints on official62", "targeted waveform-only changes with fresh protocol before each round", "replicate promising variant over three seeds and both runs", "fresh independent injection confirmation if a candidate survives"],
        "promotion": "no adoption; failed configurations retained; no guarantee of higher recall or PE overlap",
        "stop": "if repeated mechanistically distinct changes do not improve without violating guardrails, report unresolved and HOLD, never tune weights to real candidate preferences",
        "no_new_hanabi": True,
        "paper_and_historical_outputs_readonly": True,
        "references": [
            "https://arxiv.org/abs/2106.12594",
            "https://arxiv.org/abs/2505.18311",
            "https://arxiv.org/abs/2304.08393",
            "https://pycbc.org/pycbc/latest/html/pycbc.filter.html",
        ],
    }
    json_write(root / "contracts/DEVELOPMENT_CONTRACT.json", contract)
    shutil.copy2(__file__, root / "scripts" / Path(__file__).name)
    print(json.dumps({"initialized": str(root), "protected_files": len(protected)}), flush=True)


def official_o3():
    fpp = pd.read_csv(MAIN / "data/official_po_ml_fpp_all_2415_pairs.csv")
    followup = pd.read_csv(MAIN / "data/official_followup_stage_contract.csv")
    common = [c for c in followup if c in fpp and c != "pair_key"]
    return fpp.merge(followup.drop(columns=common), on="pair_key", validate="one_to_one")


def real_frame(dep, seed):
    if dep == "gwtc3":
        return pd.read_parquet(MAIN / f"results/seed_{seed}/strict_h1l1_pair_scores_C_fixed.parquet")
    return pd.read_parquet(BAY / f"results/{dep}/seed_{seed}/real_strict_pair_scores_C_fixed.parquet")


def real_inputs(dep):
    if dep == "gwtc3":
        x = np.load(MAIN / "cache/preprocessed/real_event_preprocessed_full24.npy", mmap_mode="r")
        events = pd.read_csv(MAIN / "results/strict_h1l1_event_audit.csv")
    else:
        x = ORCH.real_full24(dep)
        events = pd.read_csv(ORCH.SOURCE_ROOT / dep / "shared/real_event_preprocessing_audit.csv")
    return x, events


def pe_audit(root, dep):
    verified = root / f"audit/{dep}_all_pair_pe_official_verified.parquet"
    if verified.exists():
        return pd.read_parquet(verified)
    path = root / f"audit/{dep}_all_pair_pe_official.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = real_frame(dep, SEEDS[0])[["pair_key", "event_i", "event_j"]].copy()
    if dep == "gwtc3":
        manifest = pd.read_csv(MAIN / "cache/source_run/data/event_manifest.csv").set_index("event_name")
        names = sorted(set(frame.event_i) | set(frame.event_j))
        cache = {n: MAINCODE.load_pe_samples(manifest.loc[n]) for n in names}
        event_rows = []
        for n in names:
            row = {"event_name": n}
            for p, values in cache[n].items():
                q = np.quantile(values, [.05, .16, .5, .84, .95])
                row.update({f"pe_{p}_{key}": float(value) for key, value in zip(("q05", "q16", "median", "q84", "q95"), q)})
            event_rows.append(row)
        csv_write(root / "audit/gwtc3_event_PE_reference.csv", pd.DataFrame(event_rows))
        rows = []
        for k, pair in enumerate(frame.itertuples(index=False)):
            row = {"pair_key": pair.pair_key, "event_i": pair.event_i, "event_j": pair.event_j}
            ds = []
            for p, prefix in (("chirp_mass", "mc"), ("mass_ratio", "q"), ("chi_eff", "chi_eff"), ("luminosity_distance", "dl_app")):
                vals = MAINCODE.posterior_metrics(cache[pair.event_i][p], cache[pair.event_j][p])
                row.update({f"pe_{prefix}_{key}": value for key, value in vals.items()})
                if p != "luminosity_distance":
                    ds.append(vals["standardized_distance"])
            row["pe_dmax_intrinsic"] = max(ds)
            rows.append(row)
            if k % 300 == 0:
                print(json.dumps({"pe_pairs_done": k, "total": len(frame)}), flush=True)
        frame = pd.DataFrame(rows)
        official = official_o3()
        columns = [c for c in official if c not in frame or c == "pair_key"]
        frame = frame.merge(official[columns], on="pair_key", validate="one_to_one")
    else:
        old = pd.read_parquet(BAY / f"results/{dep}/real_consensus_with_pe_official_C_fixed.parquet")
        renames = {
            "chirp_mass_bhattacharyya_coefficient": "pe_mc_bhattacharyya_coefficient",
            "chirp_mass_standardized_posterior_distance": "pe_mc_standardized_distance",
            "max_standardized_posterior_distance": "pe_dmax_intrinsic",
            "mass_ratio_bhattacharyya_coefficient": "pe_q_bhattacharyya_coefficient",
            "chi_eff_bhattacharyya_coefficient": "pe_chi_eff_bhattacharyya_coefficient",
            "luminosity_distance_bhattacharyya_coefficient": "pe_dl_app_bhattacharyya_coefficient",
        }
        cols = ["pair_key"] + [c for c in old if c in renames or c.startswith("official_") or c == "public_hanabi_table_overlap"]
        frame = frame.merge(old[cols].rename(columns=renames), on="pair_key", how="left", validate="one_to_one")
    frame.to_parquet(path, index=False)
    return frame


def budget_row(df, config, dep, method, budget, seed="consensus"):
    part = df.head(budget)
    bc = pd.to_numeric(part.pe_mc_bhattacharyya_coefficient, errors="coerce")
    d = pd.to_numeric(part.pe_mc_standardized_distance, errors="coerce")
    valid = bc.notna() & d.notna()
    def count(col):
        if col not in part:
            return None
        return int(part[col].fillna(False).astype(bool).sum())
    return {
        "config": config, "deployment": dep, "method": method, "seed": seed, "budget": budget,
        "n_pairs": len(part), "n_pe_valid": int(valid.sum()),
        "catastrophic_mc": int((((bc < .1) | (d > 5)) & valid).sum()),
        "BC_mc_ge_0p5": int((bc >= .5).sum()), "median_BC_mc": float(bc.median()),
        "D_mc_le_3": int((d <= 3).sum()), "Dmax_le_3": int((part.pe_dmax_intrinsic <= 3).sum()),
        "official_frontend": count("official_po_or_ml_fpp_below_0p01") if dep == "gwtc3" else count("official_po_or_phazap_fpp_below_0p01"),
        "official_hanabi": count("official_any_pair_resolved_hanabi_overlap"),
        "scope": "official O3 strict62" if dep == "gwtc3" else "frozen O4a strict",
        "interpretation": "development descriptive audit, official candidates are not lensing labels",
    }


def save_evaluation(root, config, dep, frames, pe):
    out = root / f"results/{config}/{dep}"
    out.mkdir(parents=True, exist_ok=True)
    budget = []
    for method in ("waveform_only", "C_fixed"):
        ranks = []
        for seed, frame in frames.items():
            weights = {"waveform": 1., "time": 0., "sky": 0.} if method == "waveform_only" else BASE.FROZEN_V93_WEIGHTS[dep][seed]
            ranked = BASE.rank_real(frame, weights, method, seed)
            ranked.to_parquet(out / f"seed_{seed}_{method}.parquet", index=False)
            auditcols = [c for c in pe if c not in ranked or c == "pair_key"]
            aud = ranked.merge(pe[auditcols], on="pair_key", validate="one_to_one")
            budget.extend(budget_row(aud, config, dep, method, b, seed) for b in (10, 20, 50, 100))
            ranks.append(ranked)
        consensus = BASE.consensus_real(ranks, method)
        auditcols = [c for c in pe if c not in consensus or c == "pair_key"]
        consensus = consensus.merge(pe[auditcols], on="pair_key", validate="one_to_one")
        consensus.to_parquet(out / f"consensus_{method}_all_pairs_pe_official.parquet", index=False)
        csv_write(out / f"consensus_{method}_top100_pe_official.csv", consensus.head(100))
        budget.extend(budget_row(consensus, config, dep, method, b) for b in (10, 20, 50, 100))
    csv_write(out / "pe_official_budget.csv", pd.DataFrame(budget))
    return budget


def baseline_audit(root):
    for dep in ("gwtc3", "gwtc4"):
        pe = pe_audit(root, dep)
        frames = {seed: real_frame(dep, seed) for seed in SEEDS}
        budgets = save_evaluation(root, "CFIX-baseline", dep, frames, pe)
        print(pd.DataFrame(budgets).query("budget==10 and seed=='consensus'").to_json(orient="records"), flush=True)
    rows = []
    pe = pd.read_csv(root / "audit/gwtc3_event_PE_reference.csv")
    x, events = real_inputs("gwtc3")
    for seed in SEEDS:
        ev = pd.read_parquet(MAIN / f"results/seed_{seed}/event_embeddings.parquet")
        ev = ev.drop(columns=["unified_embedding"]).merge(pe, on="event_name", how="left")
        ev["model_seed"] = seed
        ev["prediction_over_PE_median"] = ev.waveform_pred_chirp_mass_detector / ev.pe_chirp_mass_median
        ev["abs_logmc_error_vs_PE"] = np.abs(np.log(ev.prediction_over_PE_median))
        short = np.asarray(x[..., -4096:])
        ev["short_input_rms"] = np.sqrt(np.mean(short**2, axis=(1, 2)))
        ev["short_input_abspeak"] = np.max(np.abs(short), axis=(1, 2))
        ev["peak_time_to_trigger_h1_s"] = np.argmax(np.abs(short[:, 0]), axis=-1) / 2048 - 1.75
        ev["peak_time_to_trigger_l1_s"] = np.argmax(np.abs(short[:, 1]), axis=-1) / 2048 - 1.75
        rows.append(ev)
    all_events = pd.concat(rows, ignore_index=True)
    csv_write(root / "audit/O3_BASELINE_EVENT_MASS_INPUT_AUDIT.csv", all_events)
    summary = all_events.groupby("model_seed").agg(n=("abs_logmc_error_vs_PE", "count"), median_absolute_log_error=("abs_logmc_error_vs_PE", "median"), median_mass_ratio=("prediction_over_PE_median", "median"))
    csv_write(root / "audit/O3_BASELINE_EVENT_MASS_SUMMARY.csv", summary.reset_index())
    print(summary.to_json(), flush=True)


def reused_registry():
    return {
        "OLD-D1-2s": OLD / "models/G2/D1-2s_base/{dep}/seed_202609031/validation_selected_model.pt",
        "OLD-D1-2plus8s": OLD / "models/G2/D1-2plus8s_base/{dep}/seed_202609031/validation_selected_model.pt",
        "OLD-D1-2plus16s": OLD / "models/G2/D1-2plus16s_base/{dep}/seed_202609031/validation_selected_model.pt",
        "OLD-D1-hard": OLD / "models/G3/D1-2s_H/{dep}/seed_202609031/validation_selected_model.pt",
        "OLD-D1-uncertainty": OLD / "models/G4/D1-2s_U/{dep}/seed_202609031/validation_selected_model.pt",
        "OLD-D1-embedding": OLD / "models/G4/D1-2s_E/{dep}/seed_202609031/validation_selected_model.pt",
    }


def evaluate_reused(root, dep):
    pe = pe_audit(root, dep)
    seed = SEEDS[0]
    real_reference = real_frame(dep, seed)
    xreal, event_meta = real_inputs(dep)
    for config, template in reused_registry().items():
        out = root / f"results/{config}/{dep}"
        if (out / "COMPLETE.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(str(template).format(dep=dep))
        metrics = []
        for split in ("validation", "test"):
            ref = pd.read_parquet(BAY / f"results/{dep}/seed_{seed}/{split}_pair_scores_bayestar_sky.parquet")
            retained = BASE.retained_event_plan(dep, seed, split)
            full = ORCH.event_array_for_plan(dep, seed, split, retained)
            emb, mean, sigma = TRAIN.encode_full24(checkpoint, full)
            np.savez_compressed(out / f"{split}_event_predictions.npz", embedding=emb, mean=mean, sigma=sigma)
            retained.to_parquet(out / f"{split}_event_plan.parquet", index=False)
            components = ORCH.component_values(ref, emb, mean, sigma)
            if split == "validation":
                calibration, grid = ORCH.fit_waveform_calibration(ref, components)
                json_write(out / "waveform_calibration.json", calibration)
                csv_write(out / "component_grid.csv", grid)
            raw, score = ORCH.apply_waveform_calibration(components, calibration)
            frame = ORCH.enrich_waveform_frame(ref, components, raw, score)
            frame.to_parquet(out / f"{split}_pair_scores.parquet", index=False)
            for method, scores in (("waveform_only", score), ("C_fixed", BASE.score_vector(frame, BASE.FROZEN_V93_WEIGHTS[dep][seed]))):
                metrics.append({"config": config, "deployment": dep, "seed": seed, "split": split, "test_role": "reused development comparator", "method": method, **BASE.full_metrics(frame, scores)})
        emb, mean, sigma = TRAIN.encode_full24(checkpoint, xreal)
        np.savez_compressed(out / "real_event_predictions.npz", embedding=emb, mean=mean, sigma=sigma)
        components = ORCH.component_values(real_reference, emb, mean, sigma)
        raw, score = ORCH.apply_waveform_calibration(components, calibration)
        frame = ORCH.enrich_waveform_frame(real_reference, components, raw, score)
        for c in ("time_score", "sky_raw_log_bf"):
            assert np.array_equal(frame[c], real_reference[c])
        budgets = save_evaluation(root, config, dep, {seed: frame}, pe)
        pred = event_meta[["idx", "event_name", "strict_h1l1_preprocessing_pass"]].copy()
        pred["pred_mc"] = np.exp(mean[:, 0]); pred["pred_logmc_sigma"] = sigma[:, 0]
        csv_write(out / "real_event_predictions.csv", pred)
        csv_write(out / "retrieval_pair_metrics.csv", pd.DataFrame(metrics))
        json_write(out / "COMPLETE.json", {"checkpoint": str(checkpoint), "sha256": sha(checkpoint), "time_sky_identical": True, "interpretation": "reused model new official-scope development audit"})
        print(json.dumps({"completed": config, "deployment": dep, "test": [m for m in metrics if m["split"] == "test"], "top10": [b for b in budgets if b["budget"] == 10 and b["seed"] == "consensus"]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("init", "audit", "reuse"), required=True)
    p.add_argument("--deployment", default="gwtc3", choices=("gwtc3", "gwtc4"))
    args = p.parse_args()
    if args.phase == "init": initialize(args.root)
    elif args.phase == "audit": baseline_audit(args.root)
    else: evaluate_reused(args.root, args.deployment)


if __name__ == "__main__":
    main()
