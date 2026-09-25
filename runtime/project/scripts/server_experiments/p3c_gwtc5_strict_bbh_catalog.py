#!/usr/bin/env python3
"""P3-C: GWTC-5 strict-BBH observable-only catalog expansion.

Addresses NC gating checklist item 3.  This is not a PE posterior analysis:
GWTC-5 complete PE is not available here.  The script explicitly analyses the
strict-BBH candidate catalog using search summaries plus candidate skymaps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

CAMPAILLA_REFERENCE_KEPT = 185
CAMPAILLA_REFERENCE_TOTAL = 3655


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    out[finite] = (x[finite] - x[finite].mean()) / (x[finite].std() + 1e-9)
    return out


def load_inputs(data_dir: Path):
    obs = pd.read_csv(data_dir / "gwtc5_observables.csv")
    pairs = pd.read_csv(data_dir / "gwtc5_pair_features.csv")
    inj = pd.read_csv(data_dir / "gwtc_injection_recovery_summary.csv")
    return obs, pairs, inj


def catalog_summary(obs: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    gps = obs["gps_trigger_time"].to_numpy(float)
    a90 = obs["sky_area_90_deg2"].to_numpy(float)
    far = obs["far_hz"].to_numpy(float)
    return {
        "catalog": "GWTC-5 strict-BBH observable-only",
        "n_events": int(len(obs)),
        "n_pairs": int(len(pairs)),
        "time_span_days": float((gps.max() - gps.min()) / 86400.0),
        "snr_min": float(obs["network_snr"].min()),
        "snr_median": float(obs["network_snr"].median()),
        "snr_max": float(obs["network_snr"].max()),
        "a90_median_deg2": float(np.nanmedian(a90)),
        "a90_p90_deg2": float(np.nanpercentile(a90, 90)),
        "a90_max_deg2": float(np.nanmax(a90)),
        "pastro_min": float(obs["p_astro"].min()),
        "pastro_median": float(obs["p_astro"].median()),
        "far_per_year_max": float(np.nanmax(far) * 365.25 * 24 * 3600),
        "n_candidate_skymaps": int(obs["skymap_file"].notna().sum()),
        "chirp_mass_missing": int(obs["chirp_mass_median"].isna().sum()),
        "mass_ratio_missing": int(obs["mass_ratio_median"].isna().sum()),
    }


def shortlist_summary(pairs: pd.DataFrame) -> dict:
    scores = pairs["combined_time_sky_z"].to_numpy(float)
    total = len(scores)
    ref_frac = CAMPAILLA_REFERENCE_KEPT / CAMPAILLA_REFERENCE_TOTAL
    equiv = int(round(ref_frac * total))
    sorted_scores = np.sort(scores)[::-1]
    return {
        "pairs": total,
        "campailla_fraction": ref_frac,
        "equivalent_top_count": equiv,
        "equivalent_threshold": float(sorted_scores[max(0, min(total - 1, equiv - 1))]),
        "top185_fraction": float(CAMPAILLA_REFERENCE_KEPT / total),
        "top185_threshold": float(sorted_scores[max(0, min(total - 1, CAMPAILLA_REFERENCE_KEPT - 1))]),
        "q90": float(np.quantile(scores, 0.90)),
        "q99": float(np.quantile(scores, 0.99)),
        "q999": float(np.quantile(scores, 0.999)),
    }


def posterior_summary_pairs(obs: pd.DataFrame, topn: int) -> pd.DataFrame:
    raw = np.column_stack([
        np.log(obs["chirp_mass_median"].to_numpy(float)),
        obs["mass_ratio_median"].to_numpy(float),
        zscore(obs["network_snr"].to_numpy(float)),
        np.cos(obs["ra_median"].to_numpy(float)),
        np.sin(obs["ra_median"].to_numpy(float)),
        np.sin(obs["dec_median"].to_numpy(float)),
    ]).astype(float)
    # Some GWTC-5 observable-only rows lack mass summaries.  For a null triage
    # baseline we keep all events and median-impute missing summary columns, then
    # report missing counts in the catalog summary.
    feats = raw.copy()
    for col in range(feats.shape[1]):
        finite = np.isfinite(feats[:, col])
        fill = float(np.nanmedian(feats[finite, col])) if finite.any() else 0.0
        feats[~finite, col] = fill
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-9)
    D = squareform(pdist(feats, metric="euclidean"))
    rows, cols = np.triu_indices(len(obs), k=1)
    df = pd.DataFrame({
        "event_i": obs["event_name"].to_numpy()[rows],
        "event_j": obs["event_name"].to_numpy()[cols],
        "posterior_summary_distance": D[rows, cols],
        "posterior_summary_score": -D[rows, cols],
    })
    return df.sort_values("posterior_summary_distance").head(topn).reset_index(drop=True)


def best_injection_rows(inj: pd.DataFrame) -> pd.DataFrame:
    sub = inj[inj["catalog"] == "gwtc5"].copy()
    cols = ["catalog", "k_injected_pairs", "score", "median_rank_mean", "recall_at_1_mean", "recall_at_5_mean", "recall_at_10_mean", "recall_at_50_mean"]
    return sub[cols].sort_values(["k_injected_pairs", "recall_at_10_mean", "recall_at_1_mean"], ascending=[True, False, False])


def write_report(out: Path, cat: dict, short: dict, top_combined: pd.DataFrame, top_sky: pd.DataFrame,
                 top_param: pd.DataFrame, inj_best: pd.DataFrame) -> None:
    lines = []
    lines.append("# P3-C GWTC-5 strict-BBH observable-only catalog expansion")
    lines.append("")
    lines.append("This analysis uses GWTC-5 search-summary observables plus candidate skymaps. It is not a full PE-posterior catalog analysis.")
    lines.append("")
    lines.append("## Catalog")
    lines.append("")
    lines.append(f"- events: {cat["n_events"]}")
    lines.append(f"- pairs: {cat["n_pairs"]}")
    lines.append(f"- time span: {cat["time_span_days"]:.1f} days")
    lines.append(f"- network SNR: median {cat["snr_median"]:.2f}, range {cat["snr_min"]:.2f}-{cat["snr_max"]:.2f}")
    lines.append(f"- A90: median {cat["a90_median_deg2"]:.2f} deg2, p90 {cat["a90_p90_deg2"]:.2f} deg2")
    lines.append(f"- candidate skymaps parsed: {cat["n_candidate_skymaps"]}")
    lines.append(f"- missing chirp mass / mass ratio summaries: {cat["chirp_mass_missing"]} / {cat["mass_ratio_missing"]}")
    lines.append("")
    lines.append("## Null shortlist")
    lines.append("")
    lines.append(f"Campailla 185/3655 equivalent keeps {short["equivalent_top_count"]} / {short["pairs"]} pairs at threshold {short["equivalent_threshold"]:.4f}.")
    lines.append(f"Top-185 fraction in this catalog is {short["top185_fraction"]:.4f}; score quantiles q90/q99/q999 = {short["q90"]:.4f}/{short["q99"]:.4f}/{short["q999"]:.4f}.")
    lines.append("")
    lines.append("## Top combined time+sky pairs")
    lines.append("")
    lines.append(top_combined.to_markdown(index=False))
    lines.append("")
    lines.append("## Top candidate-skymap sky pairs")
    lines.append("")
    lines.append(top_sky.to_markdown(index=False))
    lines.append("")
    lines.append("## Posterior-summary kNN baseline top pairs")
    lines.append("")
    lines.append(top_param.to_markdown(index=False))
    lines.append("")
    lines.append("## GWTC-5 injection recovery summary")
    lines.append("")
    lines.append(inj_best.to_markdown(index=False))
    lines.append("")
    lines.append("Interpretation: the real GWTC-5 extension is a null/triage and injection-recovery demonstration. It expands beyond the 63-event GWTC-3 PE-supported case, but without complete PE posterior products it should be written as candidate-skymap observable-only analysis.")
    (out / "p3c_gwtc5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--out", default="runs/p3c_gwtc5_strict_bbh_20260619")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs, pairs, inj = load_inputs(Path(args.data_dir))
    cat = catalog_summary(obs, pairs)
    short = shortlist_summary(pairs)
    top_combined = pairs.sort_values("combined_time_sky_z", ascending=False).head(args.topn)[
        ["event_i", "event_j", "delta_t_days", "time_score", "sky_norm_sep", "ang_sep_deg", "combined_time_sky_z"]
    ]
    top_sky = pairs.sort_values(["sky_norm_sep", "ang_sep_deg"], ascending=[True, True]).head(args.topn)[
        ["event_i", "event_j", "sky_norm_sep", "ang_sep_deg", "sky_log_overlap", "sky_step_weight"]
    ]
    top_param = posterior_summary_pairs(obs, args.topn)
    inj_best = best_injection_rows(inj)
    pd.DataFrame([cat]).to_csv(out / "p3c_catalog_summary.csv", index=False)
    pd.DataFrame([short]).to_csv(out / "p3c_shortlist_summary.csv", index=False)
    top_combined.to_csv(out / "p3c_top_combined_pairs.csv", index=False)
    top_sky.to_csv(out / "p3c_top_sky_pairs.csv", index=False)
    top_param.to_csv(out / "p3c_posterior_summary_knn_pairs.csv", index=False)
    inj_best.to_csv(out / "p3c_gwtc5_injection_recovery_summary.csv", index=False)
    write_report(out, cat, short, top_combined, top_sky, top_param, inj_best)
    print(json.dumps({"catalog": cat, "shortlist": short}, indent=2, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
