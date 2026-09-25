#!/usr/bin/env python3
"""Validate physical invariants of the v3 sky and time scoring channels.

This script contains no trainable quantities.  It checks the algebraic sky
Bayes-factor identities and audits the frozen, run-conditioned time-delay
lookup tables produced before validation/test/catalog scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "real_search"))

from physical_common import apply_time_likelihood_ratio, sky_log_bayes_factor_from_maps, write_json


def sky_invariant_rows() -> list[dict[str, float | str | bool]]:
    npix = 12 * 8**2
    uniform = np.full(npix, 1.0 / npix)

    localized_a = np.full(npix, 1e-12)
    localized_a[17] = 1.0
    localized_a /= localized_a.sum()
    localized_b = np.full(npix, 1e-12)
    localized_b[17] = 1.0
    localized_b /= localized_b.sum()
    disjoint = np.full(npix, 1e-12)
    disjoint[311] = 1.0
    disjoint /= disjoint.sum()

    # A deliberately broad posterior: cosine rewards identical shape, whereas
    # B_sky correctly reports only weak evidence relative to an isotropic prior.
    broad = uniform.copy()
    broad[: npix // 2] *= 1.05
    broad[npix // 2 :] *= 0.95
    broad /= broad.sum()

    maps = np.stack([uniform, uniform, localized_a, localized_b, disjoint, broad, broad])
    log_bf, raw, cosine = sky_log_bayes_factor_from_maps(maps)
    checks = [
        ("uniform_vs_uniform", 0, 1, np.isclose(np.exp(log_bf[0, 1]), 1.0, rtol=1e-10)),
        ("localized_same_pixel", 2, 3, log_bf[2, 3] > np.log(0.5 * npix)),
        ("localized_disjoint", 2, 4, log_bf[2, 4] < 0.0),
        ("broad_identical", 5, 6, cosine[5, 6] > 0.999),
    ]
    rows = []
    for label, i, j, passed in checks:
        rows.append(
            {
                "case": label,
                "raw_overlap": float(raw[i, j]),
                "cosine_overlap": float(cosine[i, j]),
                "sky_bayes_factor": float(np.exp(log_bf[i, j])),
                "log_sky_bayes_factor": float(log_bf[i, j]),
                "passed": bool(passed),
            }
        )
    return rows


def audit_time_calibration(path: Path, deployment: str) -> tuple[dict, pd.DataFrame]:
    raw = path.read_bytes()
    calibration = json.loads(raw)
    grid = np.asarray(calibration["log10_delay_grid"], dtype=np.float64)
    p_lens = np.asarray(calibration["p_lens_log10"], dtype=np.float64)
    p_null = np.asarray(calibration["p_null_log10"], dtype=np.float64)
    stored = np.asarray(calibration["log_likelihood_ratio"], dtype=np.float64)
    recomputed = np.log(p_lens) - np.log(p_null)
    probe_days = np.geomspace(1e-3, 3e3, 500)
    score_a = apply_time_likelihood_ratio(probe_days, calibration)
    score_b = apply_time_likelihood_ratio(probe_days.copy(), calibration)
    frame = pd.DataFrame(
        {
            "deployment": deployment,
            "delta_t_days": probe_days,
            "time_log_likelihood_ratio": score_a,
        }
    )
    audit = {
        "deployment": deployment,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "lens_samples": int(calibration["lens_samples"]),
        "null_samples": int(calibration["null_samples"]),
        "grid_points": int(len(grid)),
        "grid_strictly_increasing": bool(np.all(np.diff(grid) > 0)),
        "densities_positive_finite": bool(
            np.all(np.isfinite(p_lens))
            and np.all(np.isfinite(p_null))
            and np.all(p_lens > 0)
            and np.all(p_null > 0)
        ),
        "density_integral_lens_log10_space": float(np.trapz(p_lens, grid)),
        "density_integral_null_log10_space": float(np.trapz(p_null, grid)),
        "stored_lr_matches_density_ratio": bool(np.allclose(stored, recomputed, rtol=1e-10, atol=1e-10)),
        "lookup_is_deterministic": bool(np.array_equal(score_a, score_b)),
        "score_min": float(np.min(score_a)),
        "score_max": float(np.max(score_a)),
        "fit_policy": calibration.get("fit_policy"),
        "exposure_conditioning": calibration.get("exposure_conditioning"),
        "null_definition": calibration.get("null_definition"),
    }
    audit["passed"] = bool(
        audit["grid_strictly_increasing"]
        and audit["densities_positive_finite"]
        and abs(audit["density_integral_lens_log10_space"] - 1.0) < 1e-6
        and abs(audit["density_integral_null_log10_space"] - 1.0) < 1e-6
        and audit["stored_lr_matches_density_ratio"]
        and audit["lookup_is_deterministic"]
    )
    return audit, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "real_noise_injection_v3_physical_20260721",
    )
    args = parser.parse_args()
    out = args.result_root / "physical_scoring_invariants"
    out.mkdir(parents=True, exist_ok=True)

    sky = pd.DataFrame(sky_invariant_rows())
    sky.to_csv(out / "sky_bayes_factor_invariant_tests.csv", index=False)

    time_audits = []
    curves = []
    for deployment in ("gwtc3", "gwtc4"):
        path = args.result_root / deployment / "shared" / "time_delay_likelihood_ratio.json"
        audit, curve = audit_time_calibration(path, deployment.upper())
        time_audits.append(audit)
        curves.append(curve)
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(out / "time_delay_frozen_lookup_curves.csv", index=False)

    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for deployment, group in curve.groupby("deployment"):
            ax.plot(group["delta_t_days"], group["time_log_likelihood_ratio"], label=deployment, lw=1.8)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Time separation (days)", fontweight="bold")
        ax.set_ylabel("Time-delay log likelihood ratio", fontweight="bold")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig_time_delay_frozen_lookup.pdf")
        fig.savefig(out / "fig_time_delay_frozen_lookup.png", dpi=300)
        plt.close(fig)

    summary = {
        "sky_tests": sky.to_dict(orient="records"),
        "time_calibration_audits": time_audits,
        "all_sky_tests_passed": bool(sky["passed"].all()),
        "all_time_tests_passed": bool(all(x["passed"] for x in time_audits)),
        "interpretation": (
            "B_sky is evidence relative to an isotropic sky prior; the run-specific time table is a "
            "frozen numerical representation of a likelihood ratio, not a hand-assigned score."
        ),
    }
    write_json(out / "physical_scoring_invariant_summary.json", summary)
    if not (summary["all_sky_tests_passed"] and summary["all_time_tests_passed"]):
        raise SystemExit("Physical scoring invariant audit failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
