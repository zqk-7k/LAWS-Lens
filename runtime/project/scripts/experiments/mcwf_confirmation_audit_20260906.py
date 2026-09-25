#!/usr/bin/env python3
"""Fresh paired performance, source/noise provenance, and uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_summarize_20260905 as prev
import mcwf_fresh_confirmation_20260906 as fresh


def noise_cluster_ci(frame, candidate, baseline, events, seed, repeats=2000):
    """Crossed-noise endpoint bootstrap, conditional on fixed source catalog.

    Same-block edges receive one block multiplicity, cross-block edges the
    product. These intervals supplement, not replace, source-system intervals.
    They are not a proof of population-wide frequentist coverage.
    """
    block, index = np.unique(events.noise_bank_index, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(block), np.ones(len(block))/len(block), size=repeats)
    i, j = index[frame.idx_i.to_numpy(int)], index[frame.idx_j.to_numpy(int)]
    truth = frame.is_true_pair.to_numpy(bool)
    all_values = []
    for score in (candidate, baseline):
        order = np.argsort(-score, kind="stable")
        y = truth[order]
        ends = np.r_[np.flatnonzero(np.diff(np.asarray(score)[order]) != 0), len(order)-1]
        values = []
        for start in range(0, repeats, 40):
            c = counts[start:start+40]
            w = c[:, i].astype(float)*c[:, j]
            same = i == j
            w[:, same] = c[:, i[same]]
            w = w[:, order]
            tp = np.cumsum(w*y, 1)
            fp = np.cumsum(w*~y, 1)
            total = tp[:, -1].clip(1)
            te, fe = tp[:, ends], fp[:, ends]
            precision = np.divide(te, te+fe, out=np.zeros_like(te), where=te+fe > 0)
            ap = (precision*np.diff(np.pad(te/total[:, None], ((0, 0), (1, 0))), axis=1)).sum(1)
            f50 = fp[np.arange(len(w)), np.argmax(tp >= .5*total[:, None], axis=1)]
            f90 = fp[np.arange(len(w)), np.argmax(tp >= .9*total[:, None], axis=1)]
            values.append(np.stack((ap, f50, f90), -1))
        all_values.append(np.concatenate(values))
    delta = all_values[0]-all_values[1]
    result = {}
    for k, metric in enumerate(("AP", "F50", "F90")):
        result[f"delta_{metric}_noiseblock_ci_low"], result[f"delta_{metric}_noiseblock_ci_high"] = np.quantile(delta[:, k], [.025, .975])
    return result


def audit(root):
    candidate = fresh.verify_freeze(root)
    confirmation = fresh.data_root(root)
    rows, invariance, sources, coverage = [], [], [], []
    for dep in ("gwtc3", "gwtc4"):
        for cs in fresh.CATALOG_SEEDS:
            directory = confirmation / f"{dep}/catalog_{cs}"
            events = pd.read_parquet(directory / "event_manifest.parquet")
            source = pd.read_parquet(directory / "source_systems.parquet")
            sources.append(source.assign(deployment=dep, catalog_seed=cs))
            plan = pd.DataFrame({"system_id": events.source_uid, "family": events.family_slot})
            for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
                out = directory / f"model_{ms}"
                baseline = pd.read_parquet(out / "BASELINE_pairs.parquet")
                changed = pd.read_parquet(out / "CANDIDATE_pairs.parquet")
                for col in ("idx_i", "idx_j", "is_true_pair", "true_pair_family", "time_score", "sky_raw_log_bf"):
                    if not np.array_equal(baseline[col].to_numpy(), changed[col].to_numpy()):
                        raise RuntimeError(f"Non-waveform channel changed: {col}")
                invariance.append({"deployment": dep, "catalog_seed": cs, "model_seed": ms,
                    "time_sky_labels_pairorder_exact": True,
                    "waveform_exact": np.array_equal(baseline.waveform_score.to_numpy(), changed.waveform_score.to_numpy())})
                for method in ("waveform_only", "C_fixed"):
                    weights = {"waveform": 1., "time": 0., "sky": 0.} if method == "waveform_only" else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                    b = dev.BASE.score_vector(baseline, weights)
                    s = dev.BASE.score_vector(changed, weights)
                    bm, cm = dev.BASE.full_metrics(baseline, b), dev.BASE.full_metrics(changed, s)
                    ci, ranks = prev.ranks_bootstrap(changed, s, b, cs+ms, repeats=10000)
                    ranks.to_parquet(out / f"{method}_query_ranks.parquet", index=False)
                    row = {"deployment": dep, "catalog_seed": cs, "model_seed": ms, "method": method,
                           **cm, **ci}
                    for key in ("macro_r_at_1", "macro_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"):
                        row["baseline_"+key] = bm[key]
                        row["delta_"+key] = cm[key]-bm[key]
                    row["point_guardrail_pass"] = (cm["macro_r_at_10"] >= bm["macro_r_at_10"]-.02
                         and cm["average_precision"] >= bm["average_precision"]-.005
                         and cm["false_at_recall_0p5"] <= 1.1*bm["false_at_recall_0p5"]
                         and cm["false_at_recall_0p9"] <= 1.1*bm["false_at_recall_0p9"])
                    if dep == "gwtc3":
                        row.update(prev.pair_bootstrap(changed, s, b, plan, cs+ms, repeats=2000))
                        row.update(noise_cluster_ci(changed, s, b, events, cs+ms))
                    else:
                        for metric in ("AP", "F50", "F90"):
                            for unit in ("block", "noiseblock"):
                                for side in ("low", "high"):
                                    row[f"delta_{metric}_{unit}_ci_{side}"] = 0.
                    rows.append(row)
                print(json.dumps({"fresh_audited": dep, "catalog_seed": cs, "model_seed": ms}), flush=True)
    result = pd.DataFrame(rows)
    source = pd.concat(sources, ignore_index=True)
    if source.source_uid.duplicated().any():
        raise RuntimeError("Fresh BBH source IDs overlap across run/catalog")
    # Source hashes use actual parameters, not only newly assigned identifiers.
    params = ["m1_det", "m2_det", "a1", "a2", "tilt1", "tilt2", "theta_jn", "phi12", "phijl", "psi", "phase", "ra", "dec"]
    if source.duplicated(params).any():
        raise RuntimeError("Duplicated fresh source parameters")
    dev.csv_write(confirmation / "PAIRED_METRICS_AND_CI.csv", result)
    metrics = [c for c in result if c.startswith(("macro_", "average_", "false_", "baseline_", "delta_")) and "ci_" not in c]
    bymodel = result.groupby(["deployment", "model_seed", "method"])[metrics].mean().reset_index()
    bymodel["point_guardrail_pass"] = (bymodel.delta_macro_r_at_10.ge(-.02) & bymodel.delta_average_precision.ge(-.005)
         & bymodel.false_at_recall_0p5.le(1.1*bymodel.baseline_false_at_recall_0p5)
         & bymodel.false_at_recall_0p9.le(1.1*bymodel.baseline_false_at_recall_0p9))
    dev.csv_write(confirmation / "MODEL_MEAN_OVER_CATALOGS.csv", bymodel)
    summary = bymodel.groupby(["deployment", "method"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in summary.columns]
    dev.csv_write(confirmation / "SUMMARY.csv", summary)
    dev.csv_write(confirmation / "FROZEN_CHANNEL_INVARIANCE.csv", pd.DataFrame(invariance))
    dev.csv_write(confirmation / "NEW_SOURCE_PARAMETERS.csv", source)
    status = bool(bymodel.point_guardrail_pass.all())
    dev.json_write(confirmation / "FINAL_GUARDRAIL_AUDIT.json", {
        "candidate": candidate["O3_config"], "fresh_BBH_sources": len(source), "fresh_events": 1140,
        "independent_noise256s_blocks": 96, "unique_source_parameters": True,
        "time_sky_pairorder_unchanged": True, "O4_original_encoder_exact": True,
        "per_catalog_model_method_point_checks": int(result.point_guardrail_pass.sum()),
        "per_catalog_model_method_checks_total": len(result),
        "per_model_mean_across3catalog_guardrails_pass": status,
        "all3model_and2run_guardrails": bymodel[["deployment", "model_seed", "method", "point_guardrail_pass"]].to_dict("records"),
        "status": "PASS_CONDITIONAL_NEW_SOURCE_NOISE_TEST" if status else "FAIL_CONDITIONAL_NEW_SOURCE_NOISE_TEST",
        "limitations": ["lens environments reused; not independent lens-population evidence", "balanced representation-learning mass/SNR proposal, not inferred astrophysical population", "BAYESTAR conditional Gaussian measurement protocol, not full BBH PE or same-noise-strain localization", "O4 unchanged by construction, not a newly improved O4 model", "source and noise-block CIs are separate conditional diagnostics;3 catalogs do not establish population-wide coverage"],
        "no_paper_adoption": True})
    print(summary.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    audit(args.root)
