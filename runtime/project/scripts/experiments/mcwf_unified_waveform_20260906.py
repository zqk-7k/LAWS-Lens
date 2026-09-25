#!/usr/bin/env python3
"""Shared O3/O4 waveform recipe; real PE is development audit only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev

PREV = dev.PROJECT / "results/main_o3_mcwf_encoder_v2_20260906T153600Z"
DEPS = ("gwtc3", "gwtc4")
MODEL = "RAW-PHASE-SOURCE"
MASS_GRID = (.25, .5, 1., 2., 4.)
GAMMA_GRID = (.125, .25, .5, .75)


def initialize(root, negative_only=False, conservative_strength=False):
    contract = root / "contracts/UNIFIED_METHOD.json"
    if contract.exists():
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("New independent output directory required")
    for p in ("contracts", "scripts", "calibration", "evaluation", "results", "tables", "audit", "logs", "manifest", "reports", "figures"):
        (root / p).mkdir(parents=True, exist_ok=True)
    definition = {
        "code": "MCWF-UNIFIED-v3", "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_run_specific_result_not_unified_success": str(PREV),
        "shared_model": MODEL,
        "same_across_runs": ["peak2s4096 H1L1 input", "InceptionAttention+quadrature phase bank", "source SupCon plus simulation logMc soft-label CE", "new waveform features", "KDE calibration and selection algorithm", "convex old/new waveform ensemble", "nonzero new-encoder participation"],
        "different_allowed": ["trained checkpoint", "run-specific validation calibration", "validation-selected numeric parameters", "historically frozen outer C-fixed weights"],
        "new_training_this_round": False,
        "trained_models": "reuse six independently trained RAW-PHASE-SOURCE checkpoints with identical architecture/loss/training procedure; no O3-only model and no old-only O4 fallback",
        "data_caveat": "legacy O3 development noise includes O1/O2; disclosed and held fixed for this scoring ablation; fresh confirmation uses O3-only/O4a-only noise",
        "new_score": "u=z_validation(cosine)+k*z_validation(-abs(pred_logMc_i-pred_logMc_j)); Znew=frozen KDE log density ratio",
        "waveform_recipe": "Zwf=(1-gamma)*Zold+gamma*Znew; same formula for EVERY run and seed",
        "mass_weight_grid": MASS_GRID, "gamma_grid": GAMMA_GRID,
        "no_original_fallback": True,
        "selection": "joint k,gamma selection on simulation validation only; require waveform and C-fixed guardrails; order C-fixed F50,F90,waveform F50,F90,negative C-fixed AP,negative C-fixed R10,negative waveform AP,negative waveform R10,gamma,k",
        "guardrails": {"R10_drop_max": .02, "AP_drop_max": .005, "F50_F90_ratio_max": 1.1},
        "same_model_seeds": body.MODEL_SEEDS, "deployment_seeds": dev.SEEDS,
        "old_test_and_previous_confirmation": "development comparators, not pristine heldout after repeated evaluation",
        "real_PE": "explicit development feedback, never scoring input, not blind validation or confirmed lens labels",
        "official_stages": "post-score descriptive checks only, not optimization labels",
        "development_success": "all six reused injection guardrails; O3 consensus waveform and C-fixed Top10 catastrophic Mc count0; O4 consensus both count0 and BC>=.5/Dmax<=3 counts not below C-fixed baseline; no old-only O4",
        "per_seed_PE": "report all, including failures; consensus success never claims every seed has zero conflicts",
        "fresh_confirmation_required": True,
        "frozen": ["original time lookup", "sky maps and sky scores", "outer C-fixed weights", "real scope", "historical outputs and paper"],
        "final_status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
        "references": ["https://arxiv.org/abs/2004.11362", "https://arxiv.org/abs/1706.04599", "https://pycbc.org/pycbc/latest/html/pycbc.filter.html"],
        "reference_limits": "references justify SupCon, temperature calibration and matched-filter features, not claimed performance or our engineering tolerances",
    }
    if negative_only:
        definition.update(code="MCWF-UNIFIED-v3-R5", previous_failures="R1 positive rewards; R2-R4 predictive-mass-only corrections did not preserve all injection guardrails",
            waveform_recipe="Zwf=Zold+gamma*min(Znew,0); same new cosine+predicted-logMc likelihood ratio in both deployments; no original fallback",
            gamma_grid=[.25,.5,1.,2.], negative_only=True,
            rationale="Preserve the jointly learned shape AND mass information for contradictions; pure mass-distribution tests were insufficient. Reject positive new-model extrapolation without removing the shape feature.")
    if conservative_strength:
        definition.update(code="MCWF-UNIFIED-v3-R6",gamma_grid=[.25],
            previous_failure="R5: high validation-selected correction strengths fail some reused injection guardrails",
            rationale="A fixed common quarter-strength correction removes the strength-selection degree of freedom in the small validation catalogs. This is an exploratory shrinkage ablation, not a new physical constant.",
            shared_strength=.25)
    dev.json_write(contract,definition)
    protected = pd.read_csv(PREV / "manifest/PROTECTED_INPUT_SHA256.csv")
    additions = []
    for dep in DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            folder = PREV / f"evaluation/{MODEL}/{dep}/model_{ms}_eval_{es}"
            for p in [PREV / f"models/{MODEL}/{dep}/seed_{ms}/validation_selected_model.pt", *folder.glob("*_NEW-PHYSICAL_pairs.parquet"), *folder.glob("*_BASELINE_pairs.parquet")]:
                additions.append({"path": str(p), "sha256": dev.sha(p), "bytes": p.stat().st_size})
    protected = pd.concat([protected, pd.DataFrame(additions)], ignore_index=True).drop_duplicates("path")
    dev.csv_write(root / "manifest/PROTECTED_INPUT_SHA256.csv", protected)
    for p in (PREV / "audit").glob("*pe_official*.parquet"):
        shutil.copy2(p, root / "audit" / p.name)
    shutil.copy2(__file__, root / "scripts" / Path(__file__).name)
    print(json.dumps({"initialized": str(root), "model_recipe": MODEL}), flush=True)


def frame(dep, ms, es, split):
    return pd.read_parquet(PREV / f"evaluation/{MODEL}/{dep}/model_{ms}_eval_{es}/{split}_NEW-PHYSICAL_pairs.parquet")


def pair_features(f):
    return {"cosine": f.new_encoder_cosine.to_numpy(float), "mass": f.new_mass_similarity.to_numpy(float)}


def frozen_score(f, spec):
    new, raw, ood = ev.apply(pair_features(f), spec)
    old = f.previous_waveform_score.to_numpy(float)
    combined = old + spec["gamma"] * np.minimum(new,0.) if spec.get("negative_only",False) else (1 - spec["gamma"]) * old + spec["gamma"] * new
    return combined, new, raw, ood


def objective(m, gamma, k):
    w, c = m["waveform_only"], m["C_fixed"]
    return (c["false_at_recall_0p5"], c["false_at_recall_0p9"], w["false_at_recall_0p5"], w["false_at_recall_0p9"], -c["average_precision"], -c["macro_r_at_10"], -w["average_precision"], -w["macro_r_at_10"], gamma, k)


def fit(root, negative_only=False, conservative_strength=False):
    initialize(root, negative_only=negative_only, conservative_strength=conservative_strength)
    status = []
    for dep in DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            out = root / f"calibration/{dep}/seed_{es}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "FIT_STATUS.json").exists():
                status.append(json.loads((out / "FIT_STATUS.json").read_text()))
                continue
            f = frame(dep, ms, es, "validation")
            x = pair_features(f)
            stats = {k: [float(np.median(x[k])), max(float(np.std(x[k])), 1e-6)] for k in x}
            old = f.previous_waveform_score.to_numpy(float)
            base = ev.metrics(f, old, dep, es)
            y = f.is_true_pair.to_numpy(bool)
            candidates, rows = [], []
            for k in MASS_GRID:
                spec = {"normalization": stats, "mass_weight": k}
                raw = ev.transformed(x, spec)
                spec.update(lookup=dev.ORCH.PHYS.fit_score_likelihood_ratio(raw[y], raw[~y], bandwidth_scale=1.), validation_min=float(raw.min()), validation_max=float(raw.max()))
                for gamma in ((.25,) if conservative_strength else ((.25,.5,1.,2.) if negative_only else GAMMA_GRID)):
                    option = {**spec, "gamma": gamma, "recipe": "joint_shape_mass_negative_only" if negative_only else "convex_old_new", "negative_only": negative_only, "model_seed": ms, "eval_seed": es, "deployment": dep}
                    s, _, _, _ = frozen_score(f, option)
                    met = ev.metrics(f, s, dep, es)
                    good = ev.guard(met, base)
                    row = {"mass_weight": k, "gamma": gamma, "pass": good}
                    row.update({method + "_" + key: value for method, mm in met.items() for key, value in mm.items()})
                    rows.append(row)
                    if good:
                        candidates.append((objective(met, gamma, k), option))
            dev.csv_write(out / "validation_joint_grid.csv", pd.DataFrame(rows))
            st = {"deployment": dep, "model_seed": ms, "eval_seed": es, "status": "PASS" if candidates else "FAIL_NO_NONZERO_SHARED_RECIPE", "eligible": len(candidates)}
            if candidates:
                selected = min(candidates, key=lambda v: v[0])[1]
                dev.json_write(out / "SELECTED_CONFIG.json", selected)
                st["selected_sha256"] = dev.sha(out / "SELECTED_CONFIG.json")
                st.update(gamma=selected["gamma"], mass_weight=selected["mass_weight"])
            dev.json_write(out / "FIT_STATUS.json", st)
            status.append(st)
            print(json.dumps(st), flush=True)
    dev.csv_write(root / "tables/VALIDATION_SELECTION.csv", pd.DataFrame(status))
    dev.json_write(root / "contracts/VALIDATION_GATE.json", {"pass": all(s["status"] == "PASS" for s in status), "seeds": status})


def evaluate(root, score_function=None):
    if score_function is None:
        score_function = frozen_score
    gate = json.loads((root / "contracts/VALIDATION_GATE.json").read_text())
    if not gate["pass"]:
        raise RuntimeError("Shared method validation failed; do not evaluate candidate")
    rows, specs, guard_rows = [], [], []
    for dep in DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            out = root / f"evaluation/{dep}/seed_{es}"
            out.mkdir(parents=True, exist_ok=True)
            path = root / f"calibration/{dep}/seed_{es}/SELECTED_CONFIG.json"
            spec = json.loads(path.read_text())
            expected = next(s["selected_sha256"] for s in gate["seeds"] if s["deployment"] == dep and s["eval_seed"] == es)
            assert dev.sha(path) == expected
            specs.append({"deployment": dep, "seed": es, "model_seed": ms, "recipe": spec["recipe"], "gamma": spec["gamma"], "mass_weight": spec["mass_weight"], **dev.BASE.FROZEN_V93_WEIGHTS[dep][es]})
            for split in ("validation", "test", "real"):
                f = frame(dep, ms, es, split)
                score, new, raw, ood = score_function(f, spec)
                old = f.previous_waveform_score.to_numpy(float)
                changed = f.copy()
                changed["waveform_score"] = score
                changed["new_calibrated_waveform_score"] = new
                changed["waveform_calibration_ood"] = ood
                changed["raw_new_waveform_statistic"] = raw
                if split == "real":
                    for c in ("final_score", "waveform_contribution", "time_contribution", "sky_contribution", "rank", "method"):
                        if c in changed:
                            changed["previous_" + c] = changed[c]
                    weights = dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                    changed["waveform_contribution"] = weights["waveform"] * score
                    changed["time_contribution"] = weights["time"] * changed.time_score
                    changed["sky_contribution"] = weights["sky"] * changed.sky_raw_log_bf
                    changed["final_score"] = changed[["waveform_contribution", "time_contribution", "sky_contribution"]].sum(axis=1)
                    changed["rank"] = changed.final_score.rank(ascending=False, method="first").astype(int)
                    changed["method"] = "UNIFIED_C_FIXED_DEVELOPMENT"
                for c in ("time_score", "sky_raw_log_bf", "idx_i", "idx_j"):
                    assert np.array_equal(changed[c], f[c])
                if "pair_key" in f:
                    assert np.array_equal(changed.pair_key, f.pair_key)
                changed.to_parquet(out / f"{split}_pairs.parquet", index=False)
                if split == "real":
                    real[es] = changed
                else:
                    base = ev.metrics(f, old, dep, es)
                    met = ev.metrics(f, score, dep, es)
                    guard_rows.append({"deployment": dep, "seed": es, "split": split, "pass": ev.guard(met, base)})
                    for method in met:
                        for name, values in (("BASELINE", base[method]), ("UNIFIED", met[method])):
                            rows.append({"deployment": dep, "seed": es, "split": split, "method": method, "config": name, "ood_rate": float(ood.mean()), **values})
            print(json.dumps({"scored": dep, "seed": es}), flush=True)
        pe = dev.pe_audit(root, dep)
        dev.save_evaluation(root, "UNIFIED", dep, real, pe)
        dev.save_evaluation(root, "BASELINE", dep, {s: dev.real_frame(dep, s) for s in dev.SEEDS}, pe)
    dev.csv_write(root / "tables/RETRIEVAL_PER_SEED.csv", pd.DataFrame(rows))
    dev.csv_write(root / "tables/SHARED_METHOD_PARAMETERS.csv", pd.DataFrame(specs))
    dev.csv_write(root / "tables/REUSED_GUARDRAILS.csv", pd.DataFrame(guard_rows))
    budgets = pd.concat([pd.read_csv(p) for p in (root / "results").glob("*/*/pe_official_budget.csv")], ignore_index=True)
    dev.csv_write(root / "tables/PE_OFFICIAL_ALL.csv", budgets)
    cons = budgets.loc[(budgets.seed.astype(str) == "consensus") & (budgets.budget == 10)]
    checks = []
    for dep in DEPS:
        for method in ("waveform_only", "C_fixed"):
            n = cons.loc[(cons.deployment == dep) & (cons.method == method) & (cons.config == "UNIFIED")].iloc[0]
            b = cons.loc[(cons.deployment == dep) & (cons.method == method) & (cons.config == "BASELINE")].iloc[0]
            passed = n.catastrophic_mc == 0 and (dep == "gwtc3" or (n.BC_mc_ge_0p5 >= b.BC_mc_ge_0p5 and n.Dmax_le_3 >= b.Dmax_le_3))
            checks.append({"deployment": dep, "method": method, "pass": bool(passed), "catastrophic": int(n.catastrophic_mc), "BC_mc_ge_0p5": int(n.BC_mc_ge_0p5), "Dmax_le_3": int(n.Dmax_le_3)})
    result = {"development_pass": all(r["pass"] for r in guard_rows) and all(c["pass"] for c in checks), "PE": checks, "real_is_development": True, "all_six_same_recipe": True, "fresh_confirmation_complete": False}
    dev.json_write(root / "contracts/DEVELOPMENT_GATE.json", result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    p.add_argument("--negative-only", action="store_true")
    p.add_argument("--conservative-strength", action="store_true")
    args = p.parse_args()
    if args.phase=="fit":
        if args.conservative_strength and not args.negative_only:
            raise ValueError("The conservative-strength ablation applies only to the negative joint-shape/mass recipe")
        fit(args.root,negative_only=args.negative_only,conservative_strength=args.conservative_strength)
    else:
        evaluate(args.root)


if __name__ == "__main__":
    main()
