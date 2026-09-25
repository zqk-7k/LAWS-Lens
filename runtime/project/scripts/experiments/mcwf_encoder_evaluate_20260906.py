#!/usr/bin/env python3
"""Frozen simulation calibration, then explicit development audits.

The old held-out catalogs have already been inspected in prior rounds. They
are reused comparisons here, never newly blinded confirmation data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf
import mcwf_encoder_body_20260906 as body

RULES = ("NEW-EMBED", "NEW-PHYSICAL", "VAL-ENSEMBLE")


def contract(root: Path) -> None:
    path = root / "contracts/WAVEFORM_CALIBRATION_V2.json"
    if path.exists():
        return
    dev.json_write(path, {
        "created_before_real_evaluation": True,
        "features": ["new encoder cosine", "negative absolute difference of predicted detector-frame logMc"],
        "raw_composite": "global validation standardized cosine + k*global validation standardized mass similarity",
        "mass_weight_grid": [0., .25, .5, 1., 2., 4.],
        "feature_standardization": "frozen global validation median and SD; no query-row or real-catalog refitting",
        "density": "existing 1D KDE log-density ratio, bandwidth_scale=1; frozen on simulation validation",
        "calibration_select": "within0.02 R10/0.005 AP of new-embedding validation: minimum F50, F90, largest AP, R10, smallest mass weight",
        "ensemble": "retain old waveform evidence plus new learned representation, candidates convex blends and negative-only increment; time and sky are unchanged",
        "ensemble_grid": {"blend": [.25, .5, .75, 1.], "negative_increment": [.25, .5, 1., 2.], "baseline": [0.]},
        "ensemble_select": "both waveform-only and C-fixed pass old validation R10/AP/F50/F90 guardrails; minimum C-fixed F50 then F90, waveform F50 then F90, largest AP/R10, closest to baseline; fixed lexicographic final tie",
        "OOD": "existing frozen lookup boundary behavior audited explicitly; not extrapolated evidence or physical Bayes factor",
        "PE_and_official_usage": "only after all configs frozen; real scores do not change per-seed validation rules",
        "real_role": "O3 development diagnostic and previously inspected O4a transfer diagnostic",
        "all_variants_reported": list(RULES),
    })
    shutil.copy2(__file__, root / "scripts" / Path(__file__).name)


def features(frame, p, z):
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    means = p @ tf.LOG_CENTERS
    cosine = np.sum(z[i].astype(float) * z[j].astype(float), axis=1)
    mass = -np.abs(means[i] - means[j])
    return {"cosine": cosine, "mass": mass,
            "predictive_BC": np.sqrt(np.maximum(p[i] * p[j], 0)).sum(-1),
            "mean_i": means[i], "mean_j": means[j]}


def metrics(frame, waveform, dep, seed):
    frozen = dev.BASE.FROZEN_V93_WEIGHTS[dep][seed]
    combined = (frozen["waveform"] * waveform + frozen["time"] * frame.time_score.to_numpy()
                + frozen["sky"] * frame.sky_raw_log_bf.to_numpy())
    return {"waveform_only": dev.BASE.full_metrics(frame, waveform), "C_fixed": dev.BASE.full_metrics(frame, combined)}


def guard(candidate, base):
    return all(candidate[key]["macro_r_at_10"] >= base[key]["macro_r_at_10"] - .02
               and candidate[key]["average_precision"] >= base[key]["average_precision"] - .005
               and candidate[key]["false_at_recall_0p5"] <= base[key]["false_at_recall_0p5"] * 1.1
               and candidate[key]["false_at_recall_0p9"] <= base[key]["false_at_recall_0p9"] * 1.1
               for key in ("waveform_only", "C_fixed"))


def transformed(f, spec):
    stats = spec["normalization"]
    return ((f["cosine"] - stats["cosine"][0]) / stats["cosine"][1]
            + spec["mass_weight"] * (f["mass"] - stats["mass"][0]) / stats["mass"][1])


def apply(f, spec):
    raw = transformed(f, spec)
    score = dev.ORCH.PHYS.apply_score_likelihood_ratio(raw, spec["lookup"])
    ood = (raw < spec["validation_min"]) | (raw > spec["validation_max"])
    return score, raw, ood


def combine(old, new, config):
    gamma = config["gamma"]
    if config["mode"] == "baseline":
        return old.copy()
    if config["mode"] == "blend":
        return (1 - gamma) * old + gamma * new
    return old + gamma * np.minimum(new, 0)


def fit(frame, f, dep, seed, out):
    y = frame.is_true_pair.to_numpy(bool)
    stats = {k: [float(np.median(f[k])), max(float(np.std(f[k])), 1e-6)] for k in ("cosine", "mass")}
    options, table = [], []
    for mass_weight in (0., .25, .5, 1., 2., 4.):
        spec = {"normalization": stats, "mass_weight": mass_weight}
        raw = transformed(f, spec)
        spec["lookup"] = dev.ORCH.PHYS.fit_score_likelihood_ratio(raw[y], raw[~y], bandwidth_scale=1.)
        spec["validation_min"], spec["validation_max"] = float(raw.min()), float(raw.max())
        score, _, _ = apply(f, spec)
        m = dev.BASE.full_metrics(frame, score)
        table.append({"mass_weight": mass_weight, **m})
        options.append((spec, score, m))
    reference = options[0][2]
    eligible = [o for o in options if o[2]["macro_r_at_10"] >= reference["macro_r_at_10"] - .02
                and o[2]["average_precision"] >= reference["average_precision"] - .005]
    selected = min(eligible, key=lambda o: (o[2]["false_at_recall_0p5"], o[2]["false_at_recall_0p9"],
                   -o[2]["average_precision"], -o[2]["macro_r_at_10"], o[0]["mass_weight"]))
    old = frame.waveform_score.to_numpy()
    reference_full = metrics(frame, old, dep, seed)
    grid, good = [], []
    for mode, gammas in (("baseline", (0.,)), ("blend", (.25, .5, .75, 1.)), ("negative_increment", (.25, .5, 1., 2.))):
        for gamma in gammas:
            conf = {"mode": mode, "gamma": gamma}
            s = combine(old, selected[1], conf)
            met = metrics(frame, s, dep, seed)
            passes = guard(met, reference_full)
            row = {**conf, "passes_validation_guardrails": passes}
            row.update({method + "_" + k: v for method, values in met.items() for k, v in values.items()})
            grid.append(row)
            if passes:
                w, c = met["waveform_only"], met["C_fixed"]
                key = (c["false_at_recall_0p5"], c["false_at_recall_0p9"], w["false_at_recall_0p5"], w["false_at_recall_0p9"],
                       -c["average_precision"], -c["macro_r_at_10"], gamma, mode)
                good.append((key, conf))
    result = {"NEW-EMBED": options[0][0], "NEW-PHYSICAL": selected[0],
              "VAL-ENSEMBLE": min(good)[1], "baseline_validation_metrics": reference_full}
    dev.csv_write(out / "validation_mass_grid.csv", pd.DataFrame(table))
    dev.csv_write(out / "validation_ensemble_grid.csv", pd.DataFrame(grid))
    dev.json_write(out / "SELECTED_CONFIG.json", result)
    dev.json_write(out / "SELECTED_CONFIG_SHA256.json", {"sha256": dev.sha(out / "SELECTED_CONFIG.json"),
                   "frozen_before_reused_test_and_real_audit": True})
    return result


def input_prediction(root, dep, model_seed, eval_seed, config, split):
    out = root / f"cache/predictions/{config}/{dep}/model_{model_seed}_eval_{eval_seed}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{split}.npz"
    checkpoint = root / f"models/{config}/{dep}/seed_{model_seed}/validation_selected_model.pt"
    if cache.exists():
        saved = np.load(cache)
        if str(saved["checkpoint_sha256"]) != dev.sha(checkpoint):
            raise RuntimeError("Prediction cache belongs to a different checkpoint")
        return saved["p"], saved["z"]
    if split != "real":
        plan = dev.BASE.retained_event_plan(dep, eval_seed, split)
        full = dev.ORCH.event_array_for_plan(dep, eval_seed, split, plan)
        if config == "PSD-PHASE-SOURCE":
            import mcwf_adaptive_psd_encoder_20260906 as adaptive
            p, z, ck = adaptive.encode(root, checkpoint, full, dep, eval_seed, split)
        else:
            p, z, ck = body.encode(checkpoint, full)
        plan.to_parquet(out / f"{split}_plan.parquet", index=False)
    else:
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        if config == "PSD-PHASE-SOURCE":
            import mcwf_adaptive_psd_encoder_20260906 as adaptive
            pvalid, zvalid, ck = adaptive.encode(root, checkpoint, np.asarray(full[valid]), dep, eval_seed, split)
        else:
            pvalid, zvalid, ck = body.encode(checkpoint, np.asarray(full[valid]))
        p = np.full((len(full), 64), np.nan)
        z = np.full((len(full), 128), np.nan)
        p[valid], z[valid] = pvalid, zvalid
        ev = events[["idx", "event_name", "strict_h1l1_preprocessing_pass"]].copy()
        ev["pred_logmc"] = p @ tf.LOG_CENTERS
        ev["pred_mc"] = np.exp(ev.pred_logmc)
        ev["pred_logmc_std"] = np.sqrt(np.maximum(p @ tf.LOG_CENTERS**2 - ev.pred_logmc.to_numpy()**2, 0))
        dev.csv_write(out / "real_event_mass_predictions.csv", ev)
    np.savez_compressed(cache, p=p, z=z, checkpoint_sha256=dev.sha(checkpoint))
    return p, z


def evaluate(root, dep, config, model_seed, eval_seed):
    contract(root)
    out = root / f"evaluation/{config}/{dep}/model_{model_seed}_eval_{eval_seed}"
    if (out / "COMPLETE.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    rows, distributions = [], []
    for split in ("validation", "test", "real"):
        frame = (dev.real_frame(dep, eval_seed) if split == "real" else
                 pd.read_parquet(dev.BAY / f"results/{dep}/seed_{eval_seed}/{split}_pair_scores_bayestar_sky.parquet"))
        p, z = input_prediction(root, dep, model_seed, eval_seed, config, split)
        f = features(frame, p, z)
        if not all(np.isfinite(v).all() for v in f.values()):
            raise RuntimeError("Invalid event included in strict pair table")
        if split == "validation":
            chosen = fit(frame, f, dep, eval_seed, out)
        new, raw, ood = apply(f, chosen["NEW-PHYSICAL"])
        embedding_only, eraw, eood = apply(f, chosen["NEW-EMBED"])
        values = {"BASELINE": frame.waveform_score.to_numpy(), "NEW-EMBED": embedding_only,
                  "NEW-PHYSICAL": new, "VAL-ENSEMBLE": combine(frame.waveform_score.to_numpy(), new, chosen["VAL-ENSEMBLE"])}
        for rule, score in values.items():
            changed = frame.copy()
            changed["previous_waveform_score"] = frame.waveform_score
            changed["waveform_score"] = score
            changed["new_encoder_cosine"] = f["cosine"]
            changed["new_mass_similarity"] = f["mass"]
            changed["new_mass_predictive_BC"] = f["predictive_BC"]
            changed["new_mass_pred_logmc_i"] = f["mean_i"]
            changed["new_mass_pred_logmc_j"] = f["mean_j"]
            changed["waveform_calibration_ood"] = eood if rule == "NEW-EMBED" else ood
            for column in ("time_score", "sky_raw_log_bf"):
                if not np.array_equal(changed[column].to_numpy(), frame[column].to_numpy()):
                    raise RuntimeError(f"Frozen {column} changed")
            changed.to_parquet(out / f"{split}_{rule}_pairs.parquet", index=False)
            distributions.append({"config": config, "rule": rule, "deployment": dep, "model_seed": model_seed,
                                  "eval_seed": eval_seed, "split": split, "ood_fraction": float(ood.mean()),
                                  "score_min": float(np.min(score)), "score_median": float(np.median(score)),
                                  "score_max": float(np.max(score))})
            if split == "real":
                if rule != "BASELINE":
                    pe = dev.pe_audit(root, dep)
                    dev.save_evaluation(root, f"{config}--{rule}--pilot{model_seed}", dep, {eval_seed: changed}, pe)
            else:
                for method, m in metrics(changed, score, dep, eval_seed).items():
                    rows.append({"config": config, "rule": rule, "deployment": dep, "model_seed": model_seed,
                                 "eval_seed": eval_seed, "split": split, "method": method, **m})
        print(json.dumps({"evaluated": config, "dep": dep, "model_seed": model_seed, "split": split,
                          "mass_weight": chosen["NEW-PHYSICAL"]["mass_weight"], "ensemble": chosen["VAL-ENSEMBLE"]}), flush=True)
    dev.csv_write(out / "retrieval_pair_metrics.csv", pd.DataFrame(rows))
    dev.csv_write(out / "waveform_score_distribution.csv", pd.DataFrame(distributions))
    dev.json_write(out / "COMPLETE.json", {"time_sky_identical": True, "calibration_sha256": dev.sha(out / "SELECTED_CONFIG.json"),
        "real_role": "development audit; not blind", "independent_confirmation_completed": False})


def summarize(root):
    tables, budgets, drifts = [], [], []
    for config in body.CONFIGS:
        for dep in ("gwtc3", "gwtc4"):
            paths = sorted((root / f"evaluation/{config}/{dep}").glob("model_*/retrieval_pair_metrics.csv"))
            if not paths:
                continue
            tables.extend(pd.read_csv(p) for p in paths)
            for rule in RULES:
                frames = {}
                for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
                    path = root / f"evaluation/{config}/{dep}/model_{ms}_eval_{es}/real_{rule}_pairs.parquet"
                    if path.exists():
                        frames[es] = pd.read_parquet(path)
                if not frames:
                    continue
                pe = dev.pe_audit(root, dep)
                budgets.extend(dev.save_evaluation(root, f"{config}--{rule}", dep, frames, pe))
                for es, frame in frames.items():
                    audit = frame[["pair_key", "waveform_score"]].merge(pe, on="pair_key", validate="one_to_one")
                    drifts.append({"config": config, "rule": rule, "deployment": dep, "seed": es,
                                   "Spearman_waveform_BC_Mc": spearmanr(audit.waveform_score, audit.pe_mc_bhattacharyya_coefficient).statistic,
                                   "Spearman_waveform_neg_D_Mc": spearmanr(audit.waveform_score, -audit.pe_mc_standardized_distance).statistic})
    if tables:
        allm = pd.concat(tables, ignore_index=True)
        dev.csv_write(root / "tables/RETRIEVAL_PAIR_PER_SEED.csv", allm)
        means = allm.groupby(["config", "rule", "deployment", "split", "method"])[[
            "macro_r_at_1", "macro_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"]].agg(["mean", "std", "count"])
        means.columns = ["_".join(c) for c in means.columns]
        dev.csv_write(root / "tables/RETRIEVAL_PAIR_SUMMARY.csv", means.reset_index())
    if budgets:
        dev.csv_write(root / "tables/PE_OFFICIAL_PER_SEED_AND_CONSENSUS.csv", pd.DataFrame(budgets))
        print(pd.DataFrame(budgets).query("seed=='consensus' and budget==10").to_json(orient="records"), flush=True)
    if drifts:
        dev.csv_write(root / "tables/WAVEFORM_PE_DESCRIPTIVE_CORRELATIONS.csv", pd.DataFrame(drifts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deployment", choices=("gwtc3", "gwtc4"))
    parser.add_argument("--config", choices=body.CONFIGS)
    parser.add_argument("--seed", type=int, default=body.MODEL_SEEDS[0])
    parser.add_argument("--eval-seed", type=int, default=dev.SEEDS[0])
    parser.add_argument("--action", choices=("freeze", "evaluate", "summarize"), required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        contract(args.root)
    elif args.action == "summarize":
        summarize(args.root)
    else:
        evaluate(args.root, args.deployment, args.config, args.seed, args.eval_seed)
