from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "results" / "real_noise_injection_v4_physical_20260721"
DEFAULT_OUT = REPO / "results" / "real_noise_injection_v4_physical_20260721" / "pe_followup"
DEFAULT_PACKAGE = REPO / "packages" / "real_noise_injection_v4_physical_20260721_deliverables.tar.gz"
SEEDS = (202607221, 202607222, 202607223)
PARAMETERS = ("chirp_mass", "mass_ratio", "chi_eff")
PE_LIMIT = 30_000
KDE_LIMIT = 2_000


@dataclass(frozen=True)
class Deployment:
    key: str
    label: str
    source_run: Path
    group_column: str


DEPLOYMENTS = {
    "gwtc3": Deployment(
        "gwtc3",
        "GWTC-3 (O1-O3)",
        REPO / "runs" / "real_gwtc_lensing_search_20260625",
        "sky_map_internal_group",
    ),
    "gwtc4": Deployment(
        "gwtc4",
        "GWTC-4.1 O4a",
        REPO / "runs" / "real_gwtc34_lensing_search_20260629_full_o4",
        "sky_map_group",
    ),
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def primary_manifest(deployment: Deployment) -> pd.DataFrame:
    frame = pd.read_csv(deployment.source_run / "data" / "event_manifest.csv")
    mask = frame["include_in_primary_search"].fillna(False).astype(bool)
    mask &= frame["sky_map_available"].fillna(False).astype(bool)
    return frame.loc[mask].sort_values("gps_time").reset_index(drop=True)


def resolve_pe_path(value: Any, deployment: Deployment) -> Path:
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [REPO / raw, deployment.source_run / raw]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(str(candidates[0]))


def chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return (m1 * m2) ** (3.0 / 5.0) / np.maximum(m1 + m2, 1e-30) ** (1.0 / 5.0)


def structured_parameter(dataset: h5py.Dataset, parameter: str, index: np.ndarray) -> np.ndarray:
    names = set(dataset.dtype.names or ())
    if parameter in names:
        return np.asarray(dataset[parameter][index], dtype=np.float64)
    if parameter == "chirp_mass" and {"mass_1", "mass_2"}.issubset(names):
        return chirp_mass(np.asarray(dataset["mass_1"][index]), np.asarray(dataset["mass_2"][index]))
    if parameter == "mass_ratio" and {"mass_1", "mass_2"}.issubset(names):
        m1 = np.asarray(dataset["mass_1"][index], dtype=np.float64)
        m2 = np.asarray(dataset["mass_2"][index], dtype=np.float64)
        return np.minimum(m1, m2) / np.maximum(m1, m2)
    if parameter == "chi_eff":
        if {"mass_1", "mass_2", "spin_1z", "spin_2z"}.issubset(names):
            m1 = np.asarray(dataset["mass_1"][index], dtype=np.float64)
            m2 = np.asarray(dataset["mass_2"][index], dtype=np.float64)
            return (m1 * np.asarray(dataset["spin_1z"][index]) + m2 * np.asarray(dataset["spin_2z"][index])) / np.maximum(m1 + m2, 1e-30)
        if {"mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2"}.issubset(names):
            m1 = np.asarray(dataset["mass_1"][index], dtype=np.float64)
            m2 = np.asarray(dataset["mass_2"][index], dtype=np.float64)
            s1 = np.asarray(dataset["a_1"][index]) * np.cos(np.asarray(dataset["tilt_1"][index]))
            s2 = np.asarray(dataset["a_2"][index]) * np.cos(np.asarray(dataset["tilt_2"][index]))
            return (m1 * s1 + m2 * s2) / np.maximum(m1 + m2, 1e-30)
    return np.full(len(index), np.nan)


def prior_parameter(group: h5py.Group, parameter: str, index: np.ndarray) -> np.ndarray:
    if parameter in group:
        return np.asarray(group[parameter][index], dtype=np.float64)
    if parameter == "chirp_mass" and "mass_1" in group and "mass_2" in group:
        return chirp_mass(np.asarray(group["mass_1"][index]), np.asarray(group["mass_2"][index]))
    if parameter == "mass_ratio" and "mass_1" in group and "mass_2" in group:
        m1 = np.asarray(group["mass_1"][index], dtype=np.float64)
        m2 = np.asarray(group["mass_2"][index], dtype=np.float64)
        return np.minimum(m1, m2) / np.maximum(m1, m2)
    if parameter == "chi_eff":
        if all(name in group for name in ("mass_1", "mass_2", "spin_1z", "spin_2z")):
            m1 = np.asarray(group["mass_1"][index], dtype=np.float64)
            m2 = np.asarray(group["mass_2"][index], dtype=np.float64)
            return (m1 * np.asarray(group["spin_1z"][index]) + m2 * np.asarray(group["spin_2z"][index])) / np.maximum(m1 + m2, 1e-30)
        if all(name in group for name in ("mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2")):
            m1 = np.asarray(group["mass_1"][index], dtype=np.float64)
            m2 = np.asarray(group["mass_2"][index], dtype=np.float64)
            s1 = np.asarray(group["a_1"][index]) * np.cos(np.asarray(group["tilt_1"][index]))
            s2 = np.asarray(group["a_2"][index]) * np.cos(np.asarray(group["tilt_2"][index]))
            return (m1 * s1 + m2 * s2) / np.maximum(m1 + m2, 1e-30)
        if all(name in group for name in ("mass_1", "mass_2", "chi_1", "chi_2", "cos_tilt_1", "cos_tilt_2")):
            m1 = np.asarray(group["mass_1"][index], dtype=np.float64)
            m2 = np.asarray(group["mass_2"][index], dtype=np.float64)
            s1 = np.asarray(group["chi_1"][index]) * np.asarray(group["cos_tilt_1"][index])
            s2 = np.asarray(group["chi_2"][index]) * np.asarray(group["cos_tilt_2"][index])
            return (m1 * s1 + m2 * s2) / np.maximum(m1 + m2, 1e-30)
    return np.full(len(index), np.nan)


def load_event_pe(deployment: Deployment, row: pd.Series) -> dict[str, Any]:
    path = resolve_pe_path(row.sky_map_path, deployment)
    group_name = str(row[deployment.group_column])
    with h5py.File(path, "r") as handle:
        posterior = handle[f"{group_name}/posterior_samples"]
        count = len(posterior)
        index = np.linspace(0, count - 1, min(count, PE_LIMIT), dtype=np.int64)
        posterior_values = {name: structured_parameter(posterior, name, index) for name in PARAMETERS}
        prior_values: dict[str, np.ndarray] = {}
        prior_group_name = f"{group_name}/priors/samples"
        if prior_group_name in handle and isinstance(handle[prior_group_name], h5py.Group):
            prior = handle[prior_group_name]
            lengths = [len(prior[name]) for name in PARAMETERS if name in prior]
            if "mass_1" in prior:
                lengths.append(len(prior["mass_1"]))
            prior_count = min(lengths) if lengths else 0
            prior_index = np.linspace(0, prior_count - 1, min(prior_count, PE_LIMIT), dtype=np.int64) if prior_count else np.asarray([], dtype=np.int64)
            prior_values = {name: prior_parameter(prior, name, prior_index) for name in PARAMETERS}
    for collection in (posterior_values, prior_values):
        for name in list(collection):
            values = np.asarray(collection[name], dtype=np.float64)
            collection[name] = values[np.isfinite(values)]
    return {
        "event": str(row.event_name),
        "path": str(path),
        "group": group_name,
        "posterior": posterior_values,
        "prior": prior_values,
        "posterior_samples_total": int(count),
    }


def load_catalog_pe(deployment: Deployment) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    cache: dict[str, dict[str, Any]] = {}
    rows = []
    for _, event in primary_manifest(deployment).iterrows():
        try:
            record = load_event_pe(deployment, event)
            cache[str(event.event_name)] = record
            rows.append(
                {
                    "event_name": str(event.event_name),
                    "pe_available": True,
                    "pe_path": record["path"],
                    "pe_group": record["group"],
                    "posterior_samples_total": record["posterior_samples_total"],
                    **{f"posterior_{name}_samples": len(record["posterior"].get(name, [])) for name in PARAMETERS},
                    **{f"prior_{name}_samples": len(record["prior"].get(name, [])) for name in PARAMETERS},
                }
            )
        except Exception as exc:
            rows.append({"event_name": str(event.event_name), "pe_available": False, "error": str(exc)})
    return cache, pd.DataFrame(rows)


def posterior_sigma(values: np.ndarray) -> float:
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def bhattacharyya_coefficient(x: np.ndarray, y: np.ndarray, bins: int = 256) -> float:
    lo = min(float(np.quantile(x, 0.001)), float(np.quantile(y, 0.001)))
    hi = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
    if hi <= lo:
        return float(np.isclose(np.median(x), np.median(y)))
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges, density=False)
    py, _ = np.histogram(y, bins=edges, density=False)
    px = px / max(px.sum(), 1)
    py = py / max(py.sum(), 1)
    return float(np.sum(np.sqrt(px * py)))


def one_dimensional_pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    median_x, median_y = float(np.median(x)), float(np.median(y))
    sigma_x, sigma_y = posterior_sigma(x), posterior_sigma(y)
    denominator = math.sqrt(sigma_x * sigma_x + sigma_y * sigma_y)
    w1 = float(stats.wasserstein_distance(x, y))
    return {
        "median_i": median_x,
        "median_j": median_y,
        "sigma_i_q16_q84": sigma_x,
        "sigma_j_q16_q84": sigma_y,
        "standardized_posterior_distance": abs(median_x - median_y) / max(denominator, 1e-12),
        "wasserstein_distance": w1,
        "normalized_wasserstein_distance": w1 / max(denominator, 1e-12),
        "bhattacharyya_coefficient": bhattacharyya_coefficient(x, y),
    }


def transform_theta(values: dict[str, np.ndarray], count: int | None = None) -> np.ndarray:
    n = min(len(values[name]) for name in PARAMETERS)
    if count is not None:
        n = min(n, count)
    mc = np.asarray(values["chirp_mass"][:n], dtype=np.float64)
    q = np.clip(np.asarray(values["mass_ratio"][:n], dtype=np.float64), 1e-5, 1 - 1e-5)
    chi = np.clip(np.asarray(values["chi_eff"][:n], dtype=np.float64), -1 + 1e-5, 1 - 1e-5)
    return np.column_stack([np.log(mc), np.log(q / (1 - q)), np.arctanh(chi)])


def deterministic_subsample(values: np.ndarray, count: int, key: str) -> np.ndarray:
    if len(values) <= count:
        return values
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(len(values), size=count, replace=False))]


def _directional_overlap_log_evidence(query: np.ndarray, reference_posterior: np.ndarray, reference_prior: np.ndarray, bandwidth: float) -> float:
    posterior_kde = stats.gaussian_kde(reference_posterior.T, bw_method=lambda kde: kde.scotts_factor() * bandwidth)
    prior_kde = stats.gaussian_kde(reference_prior.T, bw_method=lambda kde: kde.scotts_factor() * bandwidth)
    log_ratio = np.log(np.maximum(posterior_kde(query.T), 1e-300)) - np.log(np.maximum(prior_kde(query.T), 1e-300))
    return float(logsumexp(log_ratio) - math.log(len(log_ratio)))


def shared_source_overlap_evidence(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if any(len(a["posterior"].get(name, [])) < 50 or len(b["posterior"].get(name, [])) < 50 for name in PARAMETERS):
        return {"available": False, "reason": "missing_posterior_parameter"}
    if any(len(a["prior"].get(name, [])) < 50 or len(b["prior"].get(name, [])) < 50 for name in PARAMETERS):
        return {"available": False, "reason": "missing_native_prior_samples"}
    pa = deterministic_subsample(transform_theta(a["posterior"]), KDE_LIMIT, a["event"] + ":post")
    pb = deterministic_subsample(transform_theta(b["posterior"]), KDE_LIMIT, b["event"] + ":post")
    pia = deterministic_subsample(transform_theta(a["prior"]), KDE_LIMIT, a["event"] + ":prior")
    pib = deterministic_subsample(transform_theta(b["prior"]), KDE_LIMIT, b["event"] + ":prior")
    pooled = np.vstack([pa, pb, pia, pib])
    center = np.median(pooled, axis=0)
    scale = np.std(pooled, axis=0)
    scale = np.maximum(scale, 1e-8)
    pa, pb, pia, pib = [(x - center) / scale for x in (pa, pb, pia, pib)]
    values = []
    for bandwidth in (0.7, 1.0, 1.4):
        try:
            a_to_b = _directional_overlap_log_evidence(pa, pb, pib, bandwidth)
            b_to_a = _directional_overlap_log_evidence(pb, pa, pia, bandwidth)
            values.append(
                {
                    "bandwidth_scale": bandwidth,
                    "log_evidence_a_to_b": a_to_b,
                    "log_evidence_b_to_a": b_to_a,
                    "symmetric_log_evidence": 0.5 * (a_to_b + b_to_a),
                }
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            return {"available": False, "reason": f"kde_failure:{exc}"}
    central = values[1]
    return {
        "available": True,
        "definition": "Symmetrized native-prior-corrected 3D posterior-overlap evidence in (log Mc_det, logit q, atanh chi_eff); approximate KDE diagnostic, not a full coherent lensing Bayes factor.",
        **central,
        "bandwidth_sensitivity": values,
        "symmetric_log_evidence_min_bandwidth_scan": float(min(x["symmetric_log_evidence"] for x in values)),
        "symmetric_log_evidence_max_bandwidth_scan": float(max(x["symmetric_log_evidence"] for x in values)),
        "min_directional_log_evidence_all_bandwidths": float(
            min(
                min(x["log_evidence_a_to_b"], x["log_evidence_b_to_a"])
                for x in values
            )
        ),
        "max_directional_log_evidence_disagreement": float(
            max(abs(x["log_evidence_a_to_b"] - x["log_evidence_b_to_a"]) for x in values)
        ),
    }


def all_pair_pe_metrics(scores: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pair in scores.itertuples(index=False):
        a, b = cache.get(str(pair.event_i)), cache.get(str(pair.event_j))
        row: dict[str, Any] = {
            "event_i": str(pair.event_i),
            "event_j": str(pair.event_j),
            "rank": int(pair.rank),
            "final_score": float(pair.final_score),
            "waveform_score": float(pair.waveform_score),
            "time_score": float(pair.time_score),
            "sky_score": float(pair.sky_score),
            "pe_available": bool(a is not None and b is not None),
        }
        for name in (
            "sky_bayes_factor",
            "cosine_overlap",
            "angular_sep_map_deg",
            "sky_area90_i_deg2",
            "sky_area90_j_deg2",
            "waveform_available",
        ):
            if hasattr(pair, name):
                row[name] = getattr(pair, name)
        if a is not None and b is not None:
            distances = []
            for parameter in PARAMETERS:
                x = a["posterior"].get(parameter, np.asarray([]))
                y = b["posterior"].get(parameter, np.asarray([]))
                if len(x) and len(y):
                    metric = one_dimensional_pair_metrics(x, y)
                    for key, value in metric.items():
                        row[f"{parameter}_{key}"] = value
                    distances.append(metric["standardized_posterior_distance"])
            row["max_standardized_posterior_distance"] = float(max(distances)) if distances else np.nan
            row["intrinsic_3sigma_consistent"] = bool(distances and max(distances) <= 3.0)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_union(input_root: Path, deployment: str, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = input_root / deployment / f"seed_{seed}" / "results" / "real_pair_scores_strict_h1l1_bbh_primary.parquet"
        frame = pd.read_parquet(path).head(top_n).copy()
        frame["seed"] = seed
        rows.append(frame)
    merged = pd.concat(rows, ignore_index=True)
    merged["pair_key"] = merged.apply(lambda r: "--".join(sorted((str(r.event_i), str(r.event_j)))), axis=1)
    return merged.groupby("pair_key", as_index=False).agg(
        event_i=("event_i", "first"),
        event_j=("event_j", "first"),
        median_rank=("rank", "median"),
        q75_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
        best_rank=("rank", "min"),
        top10_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 10))),
    ).sort_values(["median_rank", "q75_rank", "best_rank"])


def pe_rank_enrichment(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compare frozen triage top-N PE consistency with its catalog prevalence."""
    valid = diagnostics[diagnostics["pe_available"]].copy()
    valid["intrinsic_3sigma_consistent"] = valid["intrinsic_3sigma_consistent"].fillna(False).astype(bool)
    population = int(len(valid))
    successes = int(valid["intrinsic_3sigma_consistent"].sum())
    rows = []
    for top_n in (5, 10, 20, 50, 100, population):
        n = min(int(top_n), population)
        subset = valid.sort_values("rank").head(n)
        observed = int(subset["intrinsic_3sigma_consistent"].sum())
        expected = n * successes / max(population, 1)
        p_value = float(stats.hypergeom.sf(observed - 1, population, successes, n)) if population else np.nan
        rows.append(
            {
                "top_n": n,
                "pe_available_pairs": population,
                "catalog_consistent_pairs": successes,
                "observed_consistent": observed,
                "expected_under_random_ranking": expected,
                "consistent_fraction": observed / max(n, 1),
                "catalog_consistent_fraction": successes / max(population, 1),
                "enrichment_over_catalog": (observed / max(n, 1)) / max(successes / max(population, 1), 1e-300),
                "hypergeometric_one_sided_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def add_shared_evidence_to_candidates(candidates: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pair in candidates.itertuples(index=False):
        result = shared_source_overlap_evidence(cache[str(pair.event_i)], cache[str(pair.event_j)])
        rows.append({"pair_key": pair.pair_key, "event_i": pair.event_i, "event_j": pair.event_j, **result})
    return pd.DataFrame(rows)


def plot_top5_posteriors(deployment: Deployment, top: pd.DataFrame, cache: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    labels = {"chirp_mass": r"$\mathcal{M}_{c,\rm det}\ (M_\odot)$", "mass_ratio": r"$q$", "chi_eff": r"$\chi_{\rm eff}$"}
    fig, axes = plt.subplots(5, 3, figsize=(11.0, 11.5), constrained_layout=True)
    for row_index, pair in enumerate(top.head(5).itertuples(index=False)):
        for col_index, parameter in enumerate(PARAMETERS):
            ax = axes[row_index, col_index]
            x = cache[str(pair.event_i)]["posterior"][parameter]
            y = cache[str(pair.event_j)]["posterior"][parameter]
            lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
            hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
            grid = np.linspace(lo, hi, 500)
            ax.plot(grid, stats.gaussian_kde(x)(grid), color="#277DA1", lw=1.0, label=str(pair.event_i))
            ax.plot(grid, stats.gaussian_kde(y)(grid), color="#F94144", lw=1.0, label=str(pair.event_j))
            metric = one_dimensional_pair_metrics(x, y)
            ax.text(0.02, 0.96, f"D={metric['standardized_posterior_distance']:.2f}\nBC={metric['bhattacharyya_coefficient']:.2g}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlabel(labels[parameter])
            if col_index == 0:
                ax.set_ylabel(f"Median seed rank {int(pair.median_rank)}\nDensity")
            if row_index == 0 and col_index == 2:
                ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"{deployment.label}: intrinsic-parameter consistency of the top five triage pairs", fontweight="bold")
    stem = figure_dir / f"fig_top5_pe_consistency_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_waveform_inputs(input_root: Path, deployment: Deployment, top: pd.DataFrame, figure_dir: Path) -> None:
    values = np.load(input_root / deployment.key / "shared" / "real_event_preprocessed_full24.npy", mmap_mode="r")
    audit = pd.read_csv(input_root / deployment.key / "shared" / "real_event_preprocessing_audit.csv")
    index = audit.set_index("event_name")["idx"].to_dict()
    time_axis = np.arange(values.shape[-1]) / 2048.0 - 23.75
    fig, axes = plt.subplots(5, 2, figsize=(12.0, 10.5), sharex=True, constrained_layout=True)
    for row_index, pair in enumerate(top.head(5).itertuples(index=False)):
        for detector, ax in enumerate(axes[row_index]):
            x = np.asarray(values[int(index[str(pair.event_i)]), detector])
            y = np.asarray(values[int(index[str(pair.event_j)]), detector])
            ax.plot(time_axis, x, color="#277DA1", lw=0.35, alpha=0.8, label=str(pair.event_i))
            ax.plot(time_axis, y, color="#F94144", lw=0.35, alpha=0.7, label=str(pair.event_j))
            ax.axvline(0, color="black", ls="--", lw=0.6)
            ax.set_title(f"Median seed rank {int(pair.median_rank)}, {'H1' if detector == 0 else 'L1'}")
            if detector == 0:
                ax.set_ylabel("Corrected model input")
            if row_index == 4:
                ax.set_xlabel("Time from event GPS (s)")
            if row_index == 0 and detector == 1:
                ax.legend(fontsize=6)
    fig.suptitle(f"{deployment.label}: corrected full-24-s waveform inputs for the top five pairs", fontweight="bold")
    stem = figure_dir / f"fig_top5_corrected_waveform_inputs_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_score_vs_pe(deployment: Deployment, diagnostics: pd.DataFrame, figure_dir: Path) -> None:
    valid = diagnostics[np.isfinite(diagnostics["chirp_mass_standardized_posterior_distance"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    axes[0].scatter(valid["final_score"], valid["chirp_mass_standardized_posterior_distance"], s=9, alpha=0.45, color="#277DA1")
    axes[0].axhline(3, color="#D62828", ls="--", lw=0.8)
    axes[0].set_xlabel("Frozen three-channel triage score")
    axes[0].set_ylabel(r"$D_{\mathcal{M}_c}$")
    axes[1].scatter(valid["waveform_score"], valid["chirp_mass_standardized_posterior_distance"], s=9, alpha=0.45, color="#6A4C93")
    axes[1].axhline(3, color="#D62828", ls="--", lw=0.8)
    axes[1].set_xlabel("Calibrated waveform evidence")
    axes[1].set_ylabel(r"$D_{\mathcal{M}_c}$")
    ordered = valid.sort_values("rank")
    for top_n in (5, 10, 20, 50, len(ordered)):
        subset = ordered.head(top_n)
        axes[2].scatter(top_n, np.mean(subset["intrinsic_3sigma_consistent"]), color="#43AA8B", s=28)
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Top-N triage pairs")
    axes[2].set_ylabel("Fraction passing 3-sigma PE screen")
    fig.suptitle(f"{deployment.label}: independent PE consistency audit", fontweight="bold")
    stem = figure_dir / f"fig_triage_vs_pe_consistency_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_top5_sky_audit(deployment: Deployment, diagnostics: pd.DataFrame, figure_dir: Path) -> None:
    top = diagnostics.sort_values("rank").head(5).copy()
    labels = [f"{row.event_i}\n{row.event_j}" for row in top.itertuples(index=False)]
    x = np.arange(len(top))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    axes[0].bar(x - 0.18, top["sky_area90_i_deg2"], width=0.36, color="#277DA1", label="Event i")
    axes[0].bar(x + 0.18, top["sky_area90_j_deg2"], width=0.36, color="#F94144", label="Event j")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$A_{90}$ (deg$^2$)")
    axes[0].legend(fontsize=7)
    axes[1].scatter(x, top["cosine_overlap"], color="#6A4C93", s=32)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Sky cosine overlap")
    axes[2].scatter(x, top["sky_bayes_factor"], color="#43AA8B", s=32)
    axes[2].axhline(1.0, color="black", ls="--", lw=0.7)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$B_{\rm sky}$")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=6)
        ax.grid(alpha=0.15)
    fig.suptitle(f"{deployment.label}: top-five sky evidence audit", fontweight="bold")
    stem = figure_dir / f"fig_top5_sky_evidence_{deployment.key}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# Corrected real-catalog deployment and PE follow-up audit",
        "",
        "## Scope",
        "",
        "This report keeps waveform, run-conditioned time delay, and HEALPix common-source sky evidence as the frozen triage channels. PE intrinsic-parameter information is not used to tune those channels; it is an independent post-ranking physical-consistency screen.",
        "",
        "A high triage rank is not a lensing detection. A pair is called a follow-up candidate only when it also remains compatible in detector-frame intrinsic parameters and is suitable for a coherent lensing analysis.",
        "",
        "## PE metrics",
        "",
        r"For each parameter, $D_x=|\mathrm{median}(x_i)-\mathrm{median}(x_j)|/\sqrt{\sigma_i^2+\sigma_j^2}$, with $\sigma=(q_{84}-q_{16})/2$. We also report the 1-Wasserstein distance and Bhattacharyya coefficient.",
        "",
        r"For selected candidates we estimate a symmetrized, native-prior-corrected 3D posterior-overlap evidence in $(\log \mathcal{M}_{c,\rm det},\mathrm{logit}\,q,\mathrm{atanh}\,\chi_{\rm eff})$. This KDE quantity is an auditable approximation to posterior-overlap evidence, not a replacement for GOLUM/hanabi or a full coherent lensing Bayes factor.",
        "",
        "## Results",
        "",
    ]
    for key in ("gwtc3", "gwtc4"):
        item = summary[key]
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Strict pair count: {item['strict_pairs']}",
                f"- Top-10 pairs passing the predeclared 3-sigma intrinsic screen: {item['top10_3sigma_pass']} / 10",
                f"- Top-10 PE-consistency enrichment over the full strict catalog: {item['top10_pe_enrichment']:.3f}",
                f"- One-sided hypergeometric p-value for that enrichment: {item['top10_pe_enrichment_p']:.3g}",
                f"- Spearman correlation between triage score and negative chirp-mass distance: {item['spearman_score_vs_negative_dmc']:.3f}",
                f"- PE-consistent follow-up pairs among the consensus top 50: {item['consensus_top50_pe_followup_count']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation rule",
            "",
            "If the top triage pairs fail the independent PE screen, the correct conclusion is that the current real-catalog deployment has not produced a physically credible follow-up candidate. The real catalog is never used to retune the retrieval weights until agreement appears.",
            "",
        ]
    )
    return "\n".join(lines)


def package_outputs(input_root: Path, out: Path, package: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        for path in [
            REPO / "scripts" / "real_search" / "physical_common.py",
            REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py",
            REPO / "scripts" / "experiments" / "29_materialize_multinoise_training.py",
            REPO / "scripts" / "experiments" / "30_multinoise_curriculum_pilot.py",
            REPO / "scripts" / "experiments" / "31_real_noise_injection_v4_formal.py",
            REPO / "scripts" / "experiments" / "32_physical_deployment_v4_pe_audit.py",
            input_root / "run_config.json",
        ]:
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(REPO)))
        for path in input_root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in {"waveform_gate", "matchroots"} for part in path.parts):
                continue
            if path.suffix in {".npy", ".h5", ".hdf5"} and "pe_followup" not in path.parts:
                continue
            archive.add(path, arcname=str(path.relative_to(REPO)))
        for path in out.rglob("*"):
            if path.is_file() and not str(path).startswith(str(input_root)):
                archive.add(path, arcname=str(path.relative_to(REPO)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out / "figures"
    figure_dir.mkdir(exist_ok=True)
    set_style()
    overall: dict[str, Any] = {}
    for key, deployment in DEPLOYMENTS.items():
        cache, pe_manifest = load_catalog_pe(deployment)
        pe_manifest.to_csv(args.out / f"{key}_pe_manifest.csv", index=False)
        reference_path = args.input_root / key / f"seed_{SEEDS[0]}" / "results" / "real_pair_scores_strict_h1l1_bbh_primary.parquet"
        scores = pd.read_parquet(reference_path).sort_values("rank")
        sky_templates = pd.read_csv(args.input_root / key / "shared" / "real_pe_sky_template_manifest.csv")
        sky_templates = sky_templates[sky_templates["usable"] == True].set_index("event_name")  # noqa: E712
        area = sky_templates["area90_deg2"].to_dict()
        scores["sky_area90_i_deg2"] = scores["event_i"].map(area)
        scores["sky_area90_j_deg2"] = scores["event_j"].map(area)
        diagnostics = all_pair_pe_metrics(scores, cache)
        diagnostics.to_parquet(args.out / f"{key}_all_strict_pair_pe_diagnostics.parquet", index=False)
        enrichment = pe_rank_enrichment(diagnostics)
        enrichment.to_csv(args.out / f"{key}_pe_consistency_rank_enrichment.csv", index=False)
        consensus = candidate_union(args.input_root, key, top_n=50)
        consensus.to_csv(args.out / f"{key}_consensus_top50.csv", index=False)
        shared = add_shared_evidence_to_candidates(consensus, cache)
        shared.to_json(args.out / f"{key}_consensus_top50_shared_source_evidence.jsonl", orient="records", lines=True)
        one_d = diagnostics.assign(pair_key=diagnostics.apply(lambda r: "--".join(sorted((str(r.event_i), str(r.event_j)))), axis=1))
        followup = consensus.merge(one_d, on=["pair_key", "event_i", "event_j"], how="left").merge(shared, on=["pair_key", "event_i", "event_j"], how="left", suffixes=("", "_shared"))
        followup["pe_followup_screen_pass"] = (
            followup["intrinsic_3sigma_consistent"].fillna(False)
            & followup["available"].fillna(False)
            & (followup["symmetric_log_evidence_min_bandwidth_scan"].fillna(-np.inf) > 0)
        )
        followup.to_csv(args.out / f"{key}_consensus_top50_pe_followup.csv", index=False)
        plot_top5_posteriors(deployment, consensus, cache, figure_dir)
        plot_waveform_inputs(args.input_root, deployment, consensus, figure_dir)
        plot_score_vs_pe(deployment, diagnostics, figure_dir)
        plot_top5_sky_audit(deployment, diagnostics, figure_dir)
        valid = diagnostics[np.isfinite(diagnostics["chirp_mass_standardized_posterior_distance"])]
        rho = stats.spearmanr(valid["final_score"], -valid["chirp_mass_standardized_posterior_distance"]).statistic
        top10_enrichment = enrichment.loc[enrichment["top_n"] == min(10, len(valid))].iloc[0]
        overall[key] = {
            "label": deployment.label,
            "strict_pairs": int(len(scores)),
            "pe_available_events": int(pe_manifest["pe_available"].sum()),
            "top10_3sigma_pass": int(diagnostics.head(10)["intrinsic_3sigma_consistent"].sum()),
            "top10_pe_enrichment": float(top10_enrichment["enrichment_over_catalog"]),
            "top10_pe_enrichment_p": float(top10_enrichment["hypergeometric_one_sided_p"]),
            "spearman_score_vs_negative_dmc": float(rho),
            "consensus_top50_pe_followup_count": int(followup["pe_followup_screen_pass"].sum()),
            "top_pe_followup_candidates": followup[followup["pe_followup_screen_pass"]].head(20).to_dict(orient="records"),
        }
    overall["methodology"] = {
        "triage_channels": ["waveform", "run-conditioned time delay", "HEALPix sky Bayes factor"],
        "pe_used_for_retrieval_tuning": False,
        "screen_definition": "D <= 3 for Mc_det, q and chi_eff, plus the minimum approximate symmetric 3D posterior-overlap log evidence over the predeclared KDE bandwidth scan > 0",
        "detection_claim": False,
    }
    write_json(args.out / "pe_followup_summary.json", overall)
    text = report_text(overall)
    (args.out / "physical_deployment_pe_audit_report_en.md").write_text(text, encoding="utf-8")
    (args.out / "physical_deployment_pe_audit_report_cn.md").write_text(text, encoding="utf-8")
    package_outputs(args.input_root, args.out, args.package)
    print(json.dumps({"status": "complete", "out": str(args.out), "package": str(args.package)}, indent=2))


if __name__ == "__main__":
    main()
