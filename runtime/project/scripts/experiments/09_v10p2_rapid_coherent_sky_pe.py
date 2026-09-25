#!/usr/bin/env python3
"""Run a bounded rapid coherent-sky PE pilot without touching v9.3/v10.1.

The strict v10.1 pilot samples the full BBH parameter set with IMRPhenomXPHM.
This rapid pilot fixes intrinsic parameters to a pre-existing fiducial point,
samples six sky/extrinsic parameters, and uses IMRPhenomXP for inference.  For
synthetic cases the injected signal remains IMRPhenomXPHM, so the test does not
benefit from an inference/injection waveform identity shortcut.

The output is a resource and sky-domain diagnostic.  It is not full BBH PE and
must not be used to update v9.3, tune real candidates, or claim detection.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import bilby
import healpy as hp
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/gw-catalog")
BASE_SCRIPT = PROJECT / "scripts/experiments/03_v10p1_homogeneous_pe_pilot.py"
DEFAULT_CONFIG = PROJECT / "contracts/v10p2/V10P2_RAPID_COHERENT_SKY_PE_CONFIG.json"
DEFAULT_SEED = 202608232

SELECTED_CASES = (
    "o3_real_gw190828_065509",
    "o3_real_gw191219_163120",
    "o3_real_gw170814",
    "o3_sis_2303_image1",
    "o3_sis_2303_image2",
    "o3_unlensed_1247",
    "o4a_real_gw230708_230935",
    "o4a_real_gw230627_015337",
    "o4a_real_gw230704_212616",
    "o4a_sis_3754_image1",
    "o4a_sis_3754_image2",
    "o4a_unlensed_3365",
)


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("v10p1_homogeneous_pe_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()
ORIGINAL_INJECTION_POLARIZATIONS = BASE.injection_polarizations


def rapid_priors(fiducial: dict[str, float], trigger: float) -> Any:
    """Return the frozen six-dimensional rapid-sky sampling contract."""

    priors = bilby.gw.prior.BBHPriorDict()
    priors.pop("mass_1", None)
    priors.pop("mass_2", None)
    for name in (
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
    ):
        priors[name] = float(fiducial[name])
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger - 0.1,
        trigger + 0.1,
        name="geocent_time",
    )
    distance = float(fiducial["luminosity_distance"])
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        max(10.0, 0.2 * distance),
        max(100.0, 5.0 * distance),
        name="luminosity_distance",
    )
    return priors


def strict_waveform_injection(
    case: pd.Series,
    duration: float,
    config: dict[str, Any],
    amplitude_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    """Keep the strict XPHM injection while XP is used for rapid inference."""

    injection_config = copy.deepcopy(config)
    injection_config["pe_pipeline"]["waveform_approximant"] = config["pe_pipeline"][
        "injection_waveform_approximant"
    ]
    return ORIGINAL_INJECTION_POLARIZATIONS(
        case,
        duration,
        injection_config,
        amplitude_scale,
    )


def write_json(path: Path, payload: Any) -> None:
    BASE.write_json(path, payload)


def prepare(output: Path, strict_output: Path, config_path: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_manifest = strict_output / "manifests/pe_pilot_cases_all.csv"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    for name in ("contracts", "manifests", "cases", "logs", "summaries", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "contracts/V10P1_PE_PILOT_CONFIG.json")
    shutil.copy2(config_path, output / "contracts/V10P2_RAPID_COHERENT_SKY_PE_CONFIG.json")
    compatibility_addendum = config_path.parent / "V10P2_XAS_COMPATIBILITY_ADDENDUM.md"
    if compatibility_addendum.exists():
        shutil.copy2(
            compatibility_addendum,
            output / "contracts/V10P2_XAS_COMPATIBILITY_ADDENDUM.md",
        )
    method_document = config_path.parent / "V10P2_RAPID_COHERENT_SKY_PE_METHOD_CN.md"
    if method_document.exists():
        shutil.copy2(
            method_document,
            output / "contracts/V10P2_RAPID_COHERENT_SKY_PE_METHOD_CN.md",
        )
    strict_contract = strict_output / "contracts/V10P1_PE_PILOT_CONFIG.json"
    if strict_contract.exists():
        shutil.copy2(strict_contract, output / "contracts/STRICT_V10P1_PE_PILOT_CONFIG.json")
    for name in (
        "V10P1_EXPLORATORY_PROTOCOL_ADDENDUM_CN.md",
        "V10P1_HOMOGENEOUS_SKY_PE_METHOD_CN.md",
    ):
        source = strict_output / "contracts" / name
        if source.exists():
            shutil.copy2(source, output / "contracts" / name)

    source = pd.read_csv(source_manifest)
    selected = source[source.case_id.isin(SELECTED_CASES)].copy()
    missing = sorted(set(SELECTED_CASES) - set(selected.case_id))
    if missing or len(selected) != len(SELECTED_CASES):
        raise RuntimeError(f"Rapid case selection mismatch; missing={missing}, rows={len(selected)}")
    order = {case_id: index for index, case_id in enumerate(SELECTED_CASES)}
    selected["rapid_selection_order"] = selected.case_id.map(order)
    selected["stage"] = "rapid_coherent_sky_pilot"
    selected["rapid_intrinsic_treatment"] = "fixed_to_fiducial"
    selected = selected.sort_values("rapid_selection_order").reset_index(drop=True)
    selected.to_csv(
        output / "manifests/pe_pilot_cases_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected[selected.event_role.eq("real")].to_csv(
        output / "manifests/pe_pilot_cases_real.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected[~selected.event_role.eq("real")].to_csv(
        output / "manifests/pe_pilot_cases_injections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection = {
        "schema": "v10p2.1-rapid-case-selection-v1",
        "created_at_utc": BASE.utc_now(),
        "strict_source_output": str(strict_output),
        "strict_source_manifest_sha256": BASE.sha256(source_manifest),
        "case_ids": list(SELECTED_CASES),
        "case_count": len(selected),
        "selection_frozen_before_rapid_results": True,
        "real_candidate_rank_used": False,
        "historical_results_modified": False,
    }
    write_json(output / "contracts/RAPID_CASE_SELECTION_CONTRACT.json", selection)
    write_json(output / "summaries/environment.json", BASE.environment_snapshot())
    write_json(
        output / "STATUS.json",
        {
            "status": "PREPARED_RAPID_COHERENT_SKY_PE",
            "created_at_utc": BASE.utc_now(),
            "case_count": len(selected),
            "historical_results_modified": False,
            "real_candidate_rerank_performed": False,
            "terminal_boundary": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
        },
    )


def install_rapid_contract() -> None:
    BASE.priors_from_fiducial = rapid_priors
    BASE.injection_polarizations = strict_waveform_injection


def run_case(output: Path, case_id: str, sampler_seed: int) -> None:
    install_rapid_contract()
    BASE.run_case(output, case_id, sampler_seed)


def flatten_moc(moc_path: Path, nside: int) -> tuple[np.ndarray, bool]:
    executable = Path(sys.executable).parent / "ligo-skymap-flatten"
    with tempfile.TemporaryDirectory(prefix="v10p2-rapid-flatten-") as directory:
        target = Path(directory) / f"nside{nside}.fits"
        result = BASE.subprocess.run(
            [str(executable), "--nside", str(nside), str(moc_path), str(target)],
            stdout=BASE.subprocess.DEVNULL,
            stderr=BASE.subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0 or not target.exists():
            raise RuntimeError(f"Could not flatten {moc_path} to Nside={nside}")
        probability, header = hp.read_map(target, field=0, h=True, verbose=False)
    metadata = BASE.header_dict(header)
    nest = str(metadata.get("ORDERING", "RING")).upper() == "NESTED"
    probability = np.asarray(probability, dtype=np.float64)
    probability /= np.sum(probability, dtype=np.float64)
    return probability, nest


def truth_hpd_metrics(probability: np.ndarray, nside: int, nest: bool, ra: float, dec: float) -> dict[str, Any]:
    theta = 0.5 * math.pi - float(dec)
    phi = float(ra) % (2.0 * math.pi)
    true_pixel = int(hp.ang2pix(nside, theta, phi, nest=nest))
    level = float(np.sum(probability[probability >= probability[true_pixel]], dtype=np.float64))
    map_pixel = int(np.argmax(probability))
    map_theta, map_phi = hp.pix2ang(nside, map_pixel, nest=nest)
    map_ra = float(map_phi)
    map_dec = float(0.5 * math.pi - map_theta)
    vector_true = np.asarray(hp.ang2vec(theta, phi), dtype=np.float64)
    vector_map = np.asarray(hp.ang2vec(map_theta, map_phi), dtype=np.float64)
    separation = math.degrees(math.acos(float(np.clip(np.dot(vector_true, vector_map), -1.0, 1.0))))
    return {
        "truth_hpd_credible_level": level,
        "truth_in_hpd90": level <= 0.9,
        "truth_to_map_separation_deg": separation,
        "map_ra_rad": map_ra,
        "map_dec_rad": map_dec,
    }


def pair_scores(case_maps: dict[str, np.ndarray], manifest: pd.DataFrame, nside: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_case = manifest.set_index("case_id")
    for deployment in sorted(manifest.deployment.unique()):
        cases = manifest[manifest.deployment.eq(deployment)].case_id.tolist()
        for left_index, left in enumerate(cases):
            for right in cases[left_index + 1 :]:
                p_left = case_maps[left]
                p_right = case_maps[right]
                raw = float(np.sum(p_left * p_right, dtype=np.float64))
                log_bf = float(math.log(max(hp.nside2npix(nside) * raw, np.finfo(float).tiny)))
                row_left = by_case.loc[left]
                row_right = by_case.loc[right]
                same_lens = bool(
                    row_left.event_role == "lensed_image"
                    and row_right.event_role == "lensed_image"
                    and str(row_left.lens_system_id) == str(row_right.lens_system_id)
                )
                records.append(
                    {
                        "deployment": deployment,
                        "case_i": left,
                        "case_j": right,
                        "role_i": row_left.event_role,
                        "role_j": row_right.event_role,
                        "true_lensed_companion": same_lens,
                        "nside": nside,
                        "raw_overlap": raw,
                        "sky_log_bayes_factor": log_bf,
                    }
                )
    result = pd.DataFrame(records)
    result["rank_within_deployment"] = result.groupby("deployment")["sky_log_bayes_factor"].rank(
        method="min", ascending=False
    ).astype(int)
    return result


def summarize(output: Path, strict_output: Path, sampler_seed: int) -> None:
    install_rapid_contract()
    BASE.summarize(output)
    config = json.loads((output / "contracts/V10P1_PE_PILOT_CONFIG.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(output / "manifests/pe_pilot_cases_all.csv")
    base_metrics_path = output / "summaries/pe_pilot_case_metrics.csv"
    metrics = pd.read_csv(base_metrics_path) if base_metrics_path.exists() else pd.DataFrame()
    if metrics.empty:
        raise RuntimeError("No completed rapid PE cases to summarize")
    metrics = metrics.merge(
        manifest[
            [
                "case_id",
                "public_a90_deg2",
                "planned_map_morphology",
                "lens_system_id",
                "source_row",
            ]
        ],
        on="case_id",
        how="left",
    )
    # The v10.1 base pilot used ``len(posterior) >= 1000`` as its sole
    # ``sampler_converged`` flag.  Equal-weight posterior size is a useful
    # precision audit, but it is not Dynesty's stopping condition.  A normal
    # run-case completion means Dynesty exited with the frozen dlogz target
    # (there is no maxiter/maxcall escape in this contract).  Keep the legacy
    # flag and report the two checks separately.
    metrics["legacy_sampler_converged_flag"] = metrics.sampler_converged
    metrics["posterior_samples_ge1000"] = (
        pd.to_numeric(metrics.posterior_samples, errors="coerce") >= 1000
    )
    metrics["sampler_stop_completed"] = metrics.case_id.map(
        lambda case_id: (
            output / "logs" / f"{case_id}_seed{sampler_seed}.complete.json"
        ).exists()
    )
    metrics["evidence_finite"] = np.isfinite(
        pd.to_numeric(metrics.log_evidence, errors="coerce")
    ) & np.isfinite(pd.to_numeric(metrics.log_evidence_error, errors="coerce"))
    metrics["rapid_to_public_a90_ratio"] = (
        pd.to_numeric(metrics.a90_deg2_nside512, errors="coerce")
        / pd.to_numeric(metrics.public_a90_deg2, errors="coerce")
    )

    maps_by_nside: dict[int, dict[str, np.ndarray]] = {256: {}, 512: {}, 1024: {}}
    map_nest: dict[tuple[str, int], bool] = {}
    truth_rows: list[dict[str, Any]] = []
    manifest_index = manifest.set_index("case_id")
    for case_id in metrics.case_id:
        case_dir = output / "cases" / f"{case_id}_seed{sampler_seed}"
        moc = case_dir / "sky_map/skymap.fits.gz"
        if not moc.exists():
            continue
        for nside in maps_by_nside:
            probability, nest = flatten_moc(moc, nside)
            maps_by_nside[nside][case_id] = probability
            map_nest[(case_id, nside)] = nest
        row = manifest_index.loc[case_id]
        if row.event_role != "real":
            pe = json.loads((case_dir / "pe_metrics.json").read_text(encoding="utf-8"))
            truth = pe["injection_audit"]["truth"]
            truth_rows.append(
                {
                    "case_id": case_id,
                    "deployment": row.deployment,
                    "event_role": row.event_role,
                    "network_snr": row.network_snr,
                    **truth_hpd_metrics(
                        maps_by_nside[512][case_id],
                        512,
                        map_nest[(case_id, 512)],
                        truth["ra"],
                        truth["dec"],
                    ),
                }
            )

    truth_table = pd.DataFrame(truth_rows)
    truth_table.to_csv(
        output / "summaries/injection_truth_sky_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pair_tables = []
    for nside, maps in maps_by_nside.items():
        pair_tables.append(pair_scores(maps, manifest[manifest.case_id.isin(maps)], nside))
    pair_table = pd.concat(pair_tables, ignore_index=True)
    pair_table.to_csv(
        output / "summaries/rapid_sky_pair_scores_256_512_1024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    convergence = (
        pair_table.pivot_table(
            index=["deployment", "case_i", "case_j", "true_lensed_companion"],
            columns="nside",
            values="sky_log_bayes_factor",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={256: "sky_log_bf_nside256", 512: "sky_log_bf_nside512", 1024: "sky_log_bf_nside1024"})
    )
    convergence["abs_delta_256_512"] = np.abs(
        convergence.sky_log_bf_nside256 - convergence.sky_log_bf_nside512
    )
    convergence["abs_delta_512_1024"] = np.abs(
        convergence.sky_log_bf_nside512 - convergence.sky_log_bf_nside1024
    )
    convergence["sign_flip_256_512"] = (
        np.sign(convergence.sky_log_bf_nside256)
        != np.sign(convergence.sky_log_bf_nside512)
    )
    convergence["sign_flip_512_1024"] = (
        np.sign(convergence.sky_log_bf_nside512)
        != np.sign(convergence.sky_log_bf_nside1024)
    )
    convergence.to_csv(
        output / "summaries/rapid_pair_resolution_convergence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strict_rows: list[dict[str, Any]] = []
    for case_id in metrics.case_id:
        strict_case = strict_output / "cases" / f"{case_id}_seed202608231"
        strict_pe = strict_case / "pe_metrics.json"
        strict_sky = strict_case / "sky_map_metrics.json"
        if not strict_pe.exists() or not strict_sky.exists():
            continue
        pe = json.loads(strict_pe.read_text(encoding="utf-8"))
        sky = json.loads(strict_sky.read_text(encoding="utf-8"))
        rapid = metrics[metrics.case_id.eq(case_id)].iloc[0]
        strict_a90 = float(sky["raster"]["512"]["a90_deg2"])
        strict_rows.append(
            {
                "case_id": case_id,
                "deployment": rapid.deployment,
                "event_role": rapid.event_role,
                "rapid_a90_deg2": rapid.a90_deg2_nside512,
                "strict_a90_deg2": strict_a90,
                "rapid_to_strict_a90_ratio": float(rapid.a90_deg2_nside512) / strict_a90,
                "rapid_wall_seconds": rapid.total_case_wall_seconds,
                "strict_wall_seconds": float(pe.get("sampler_wall_seconds", np.nan)),
                "rapid_log_evidence_error": rapid.log_evidence_error,
                "strict_log_evidence_error": pe.get("log_evidence_error"),
            }
        )
    strict_table = pd.DataFrame(strict_rows)
    strict_table.to_csv(
        output / "summaries/rapid_vs_strict_completed_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(base_metrics_path, index=False, encoding="utf-8-sig")

    real = metrics[metrics.event_role.eq("real")].copy()
    log_ratios = np.abs(np.log10(pd.to_numeric(real.rapid_to_public_a90_ratio, errors="coerce")))
    injection_covered = int(truth_table.truth_in_hpd90.sum()) if len(truth_table) else 0
    thresholds = config["validation_thresholds"]
    a90_pass = bool(
        len(log_ratios.dropna())
        and float(np.nanmedian(log_ratios))
        <= float(thresholds["real_a90_median_absolute_log10_ratio_max"])
    )
    coverage_pass = bool(
        len(truth_table) == 6
        and injection_covered >= int(thresholds["injection_hpd90_min_covered_of_6"])
    )
    true_pairs_512 = pair_table[(pair_table.nside.eq(512)) & pair_table.true_lensed_companion]
    payload = {
        "schema": "v10p2.1-rapid-coherent-sky-pe-summary-v1",
        "generated_at_utc": BASE.utc_now(),
        "completed_cases": int(len(metrics)),
        "expected_cases": len(SELECTED_CASES),
        "all_sampler_stops_completed": bool(metrics.sampler_stop_completed.all()),
        "all_evidence_finite": bool(metrics.evidence_finite.all()),
        "all_posterior_samples_ge1000": bool(metrics.posterior_samples_ge1000.all()),
        "legacy_all_sampler_converged_flag": bool(
            metrics.legacy_sampler_converged_flag.fillna(False).astype(bool).all()
        ),
        "all_maps_valid": bool(metrics.map_valid.fillna(False).astype(bool).all()),
        "real_a90_median_absolute_log10_ratio": float(np.nanmedian(log_ratios)),
        "real_a90_factor_at_median_absolute_log_ratio": float(10 ** np.nanmedian(log_ratios)),
        "real_a90_validation_pass": a90_pass,
        "injection_hpd90_covered": injection_covered,
        "injection_hpd90_total": int(len(truth_table)),
        "injection_coverage_descriptive_pass": coverage_pass,
        "true_lensed_pair_sky_log_bf_nside512": true_pairs_512[
            ["deployment", "case_i", "case_j", "sky_log_bayes_factor", "rank_within_deployment"]
        ].to_dict(orient="records"),
        "strict_reference_cases_available": int(len(strict_table)),
        "pair_resolution_512_to_1024_q99_abs_delta": float(
            np.quantile(convergence.abs_delta_512_1024, 0.99)
        ),
        "pair_resolution_512_to_1024_max_abs_delta": float(
            convergence.abs_delta_512_1024.max()
        ),
        "pair_resolution_512_to_1024_sign_flips": int(
            convergence.sign_flip_512_1024.sum()
        ),
        "scientific_status": (
            "RAPID_SKY_PE_FEASIBLE_PENDING_STRICT_COMPARISON"
            if a90_pass and coverage_pass
            else "RAPID_SKY_PE_NOT_YET_VALIDATED"
        ),
        "full_bbh_pe_claim_allowed": False,
        "real_candidate_rerank_performed": False,
        "historical_results_modified": False,
        "terminal_status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    write_json(output / "summaries/rapid_coherent_sky_pe_summary.json", payload)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    colors = {"O3": "#1f77b4", "O4a": "#d95f02"}
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    for deployment, group in real.groupby("deployment"):
        axes[0].scatter(
            group.public_a90_deg2,
            group.a90_deg2_nside512,
            s=30,
            alpha=0.85,
            label=deployment,
            color=colors.get(deployment),
        )
    finite_a90 = pd.concat(
        [
            pd.to_numeric(real.public_a90_deg2, errors="coerce"),
            pd.to_numeric(real.a90_deg2_nside512, errors="coerce"),
        ]
    ).dropna()
    lower = max(float(finite_a90.min()) * 0.7, 1.0)
    upper = float(finite_a90.max()) * 1.4
    axes[0].plot([lower, upper], [lower, upper], color="0.25", linewidth=1, linestyle="--")
    axes[0].set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper))
    axes[0].set_xlabel(r"Public PE $A_{90}$ (deg$^2$)")
    axes[0].set_ylabel(r"Rapid PE $A_{90}$ (deg$^2$)")
    axes[0].legend(frameon=False, loc="best")

    x_positions = np.arange(len(truth_table))
    for deployment, group in truth_table.groupby("deployment"):
        indices = group.index.to_numpy()
        axes[1].scatter(
            indices,
            group.truth_hpd_credible_level,
            s=30,
            color=colors.get(deployment),
            label=deployment,
        )
    axes[1].axhline(0.9, color="0.25", linewidth=1, linestyle="--")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([str(index + 1) for index in x_positions])
    axes[1].set_xlabel("Injection case")
    axes[1].set_ylabel("Truth HPD credible level")

    for deployment, group in convergence.groupby("deployment"):
        true_group = group[group.true_lensed_companion]
        if len(true_group):
            values = true_group[
                ["sky_log_bf_nside256", "sky_log_bf_nside512", "sky_log_bf_nside1024"]
            ].iloc[0].to_numpy(dtype=float)
            axes[2].plot(
                [256, 512, 1024],
                values,
                marker="o",
                linewidth=1.4,
                color=colors.get(deployment),
                label=f"{deployment} true pair",
            )
        null_group = group[~group.true_lensed_companion]
        median = [
            float(null_group[f"sky_log_bf_nside{nside}"].median())
            for nside in (256, 512, 1024)
        ]
        axes[2].plot(
            [256, 512, 1024],
            median,
            marker="x",
            linewidth=1,
            linestyle=":",
            color=colors.get(deployment),
            alpha=0.8,
            label=f"{deployment} null median",
        )
    axes[2].axhline(0, color="0.65", linewidth=0.8)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([256, 512, 1024])
    axes[2].set_xticklabels(["256", "512", "1024"])
    axes[2].set_xlabel("HEALPix Nside")
    axes[2].set_ylabel(r"$Z_{\rm sky}=\log B_{\rm sky}$")
    axes[2].legend(frameon=False, fontsize=6.7, loc="best")
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    figure.savefig(output / "figures/fig_rapid_coherent_sky_pe_pilot.pdf")
    figure.savefig(output / "figures/fig_rapid_coherent_sky_pe_pilot.png", dpi=300)
    plt.close(figure)

    lines = [
        "# v10.2 快速相干天空 PE pilot 报告",
        "",
        "> 本实验固定内禀参数，只采样六个天空/外参；它不是完整 BBH PE，不修改 v9.3，也不用于候选调参。",
        "",
        "## 设计",
        "",
        "- O3 与 O4a 各 6 个代表事件，共 12 个；覆盖真实事件、透镜像和无透镜注入。",
        "- `O3` 是历史部署标签：真实侧对应 GWTC-3 scope（含 O1/O2/O3 事件），注入侧使用 O3 off-source noise；二者在结果表中继续分列 provenance。",
        "- 注入信号保持 IMRPhenomXPHM；快速推断使用支持进动但省略高阶模的 IMRPhenomXP。",
        "- Dynesty: nlive=400, nact=8, dlogz=0.5；正式地图 Nside=512。",
        "- 真实事件与公开 PE 的 A90 比较；注入检查二维 90% HPD 真值覆盖。",
        "",
        "## 快速结果",
        "",
        f"- 完成：{len(metrics)}/{len(SELECTED_CASES)}；Dynesty 正常停止={payload['all_sampler_stops_completed']}；evidence 全部有限={payload['all_evidence_finite']}；地图全有效={payload['all_maps_valid']}。",
        f"- posterior 均不少于 1000 条={payload['all_posterior_samples_ge1000']}；该项单独作为样本量审计，不再冒充 Dynesty 的 dlogz 停止条件。",
        f"- 真实事件 A90 的中位绝对偏差因子：{payload['real_a90_factor_at_median_absolute_log_ratio']:.3f}。",
        f"- 注入真值位于 90% HPD：{injection_covered}/{len(truth_table)}。",
        f"- 已完成严格 full-PE 对照：{len(strict_table)} 个；其余严格任务仍可稍后刷新。",
        f"- 512 到 1024 的 pair 分数 q99/max 绝对差：{payload['pair_resolution_512_to_1024_q99_abs_delta']:.4g}/{payload['pair_resolution_512_to_1024_max_abs_delta']:.4g} nats；符号翻转 {payload['pair_resolution_512_to_1024_sign_flips']}。",
        "",
        "## 边界",
        "",
        "该 pilot 只能判断快速天空 PE 是否适合作为统一注入天空管线。它不能替代完整参数估计，不能给真实候选提供 lensing evidence，也不能据此更新论文。",
        "",
        f"最终状态：`{payload['terminal_status']}`",
    ]
    (output / "RAPID_COHERENT_SKY_PE_REPORT_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(output / "STATUS.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--strict-output", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser = sub.add_parser("run-case")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--case-id", required=True)
    run_parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SEED)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--strict-output", type=Path, required=True)
    summary_parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve(), args.strict_output.resolve(), args.config.resolve())
    elif args.command == "run-case":
        run_case(args.output.resolve(), args.case_id, args.sampler_seed)
    elif args.command == "summarize":
        summarize(args.output.resolve(), args.strict_output.resolve(), args.sampler_seed)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
