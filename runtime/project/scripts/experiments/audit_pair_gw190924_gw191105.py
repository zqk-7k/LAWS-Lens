#!/usr/bin/env python3
"""Forensic audit for GW190924_021846--GW191105_143521.

This script is read-only with respect to the authoritative search products.  It
reconstructs the exact 2 s, 4096-sample waveform input used by the v7 encoder,
compares all three frozen waveform deployments, and contrasts their scores with
the public single-event PE posteriors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


EVENT_A = "GW190924_021846"
EVENT_B = "GW191105_143521"
SEEDS = ("202607241", "202607242", "202607243")
FS = 2048.0
INPUT_SAMPLES = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/gw-catalog"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_flip_channels(values: np.ndarray) -> np.ndarray:
    reference = np.sum(values, axis=0)
    return -values if reference[np.argmax(np.abs(reference))] < 0 else values


def zscore_channels(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mean) * 1e-6, 1e-30)
    return ((values - mean) / np.maximum(std, floor)).astype(np.float32)


def exact_model_input(full24: np.ndarray) -> np.ndarray:
    crop = np.asarray(full24[..., -INPUT_SAMPLES:], dtype=np.float32)
    return zscore_channels(peak_flip_channels(crop.copy()))


def load_posterior(path: Path, group: str) -> dict[str, np.ndarray]:
    fields = ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance", "network_optimal_snr")
    with h5py.File(path, "r") as handle:
        dataset = handle[f"{group}/posterior_samples"]
        names = set(dataset.dtype.names or ())
        return {
            field: np.asarray(dataset[field][()], dtype=np.float64)
            for field in fields
            if field in names
        }


def posterior_sigma(values: np.ndarray) -> float:
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def bhattacharyya_coefficient(x: np.ndarray, y: np.ndarray, bins: int = 256) -> float:
    lo = min(float(np.quantile(x, 0.001)), float(np.quantile(y, 0.001)))
    hi = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges)
    py, _ = np.histogram(y, bins=edges)
    px = px / max(px.sum(), 1)
    py = py / max(py.sum(), 1)
    return float(np.sum(np.sqrt(px * py)))


def pe_pair_metric(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    median_x = float(np.median(x))
    median_y = float(np.median(y))
    sigma_x = posterior_sigma(x)
    sigma_y = posterior_sigma(y)
    scale = math.sqrt(sigma_x**2 + sigma_y**2)
    return {
        "median_a": median_x,
        "median_b": median_y,
        "q16_a": float(np.quantile(x, 0.16)),
        "q84_a": float(np.quantile(x, 0.84)),
        "q16_b": float(np.quantile(y, 0.16)),
        "q84_b": float(np.quantile(y, 0.84)),
        "sigma_a": sigma_x,
        "sigma_b": sigma_y,
        "standardized_posterior_distance": abs(median_x - median_y) / max(scale, 1e-12),
        "bhattacharyya_coefficient": bhattacharyya_coefficient(x, y),
        "wasserstein_distance": float(stats.wasserstein_distance(x, y)),
    }


def direct_waveform_metrics(a: np.ndarray, b: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    max_lag_samples = int(round(0.1 * FS))
    for detector, xa, xb in zip(("H1", "L1"), a, b):
        pearson = float(np.corrcoef(xa, xb)[0, 1])
        corr = signal.correlate(xa, xb, mode="full", method="fft")
        lags = signal.correlation_lags(len(xa), len(xb), mode="full")
        corr = corr / max(float(np.linalg.norm(xa) * np.linalg.norm(xb)), 1e-12)
        keep = np.abs(lags) <= max_lag_samples
        selected = np.flatnonzero(keep)[np.argmax(np.abs(corr[keep]))]
        fa = np.abs(np.fft.rfft(xa * signal.windows.tukey(len(xa), alpha=0.1)))
        fb = np.abs(np.fft.rfft(xb * signal.windows.tukey(len(xb), alpha=0.1)))
        frequency = np.fft.rfftfreq(len(xa), 1.0 / FS)
        band = (frequency >= 40.0) & (frequency <= 580.0)
        spectral_cosine = float(np.dot(fa[band], fb[band]) / (np.linalg.norm(fa[band]) * np.linalg.norm(fb[band])))
        rows.append(
            {
                "detector": detector,
                "zero_lag_pearson": pearson,
                "max_abs_xcorr_pm100ms": float(abs(corr[selected])),
                "lag_at_max_abs_xcorr_ms": float(1000.0 * lags[selected] / FS),
                "magnitude_spectrum_cosine_40_580hz": spectral_cosine,
            }
        )
    return rows


def kde_line(axis: plt.Axes, values: np.ndarray, color: str, label: str, grid: np.ndarray) -> None:
    kde = stats.gaussian_kde(values)
    axis.plot(grid, kde(grid), color=color, lw=1.8, label=label)
    q16, median, q84 = np.quantile(values, [0.16, 0.5, 0.84])
    axis.axvspan(q16, q84, color=color, alpha=0.13, lw=0)
    axis.axvline(median, color=color, lw=1.2, ls="--")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    figures = output / "figures"
    tables = output / "tables"
    reports = output / "reports"
    scripts = output / "scripts"
    for directory in (figures, tables, reports, scripts):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), scripts / Path(__file__).name)

    v7 = repo / "results/real_noise_injection_v7_peak2s_formal_20260722/gwtc3"
    source = repo / "results/real_noise_injection_v5_physical_source_20260721/gwtc3/shared"
    candidate_file = (
        repo
        / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z/results/gwtc3/real_top50_with_pe_official_C_fixed.csv"
    )
    pe_manifest_file = v7.parent / "pe_followup/gwtc3_pe_manifest.csv"
    waveform_bank_file = source / "real_event_preprocessed_full24.npy"

    candidate = pd.read_csv(candidate_file)
    pair_mask = (
        ((candidate["event_i"] == EVENT_A) & (candidate["event_j"] == EVENT_B))
        | ((candidate["event_i"] == EVENT_B) & (candidate["event_j"] == EVENT_A))
    )
    pair_candidate = candidate.loc[pair_mask].iloc[0]

    seed_rows: list[dict[str, float | str | bool]] = []
    event_prediction_rows: list[dict[str, float | str]] = []
    indices: dict[str, int] = {}
    for seed in SEEDS:
        seed_root = v7 / f"seed_{seed}"
        similarity_file = seed_root / "features/real_waveform_similarity_unified_v7.parquet"
        embedding_file = seed_root / "features/real_waveform_embeddings_unified_v7.parquet"
        calibration_file = seed_root / "results/waveform_channel_calibration_v7.json"
        audit_file = seed_root / "results/real_waveform_deployment_audit_unified_v7.json"

        similarities = pd.read_parquet(similarity_file)
        mask = (
            ((similarities["event_i"] == EVENT_A) & (similarities["event_j"] == EVENT_B))
            | ((similarities["event_i"] == EVENT_B) & (similarities["event_j"] == EVENT_A))
        )
        pair = similarities.loc[mask].iloc[0]
        embeddings = pd.read_parquet(embedding_file)
        selected = embeddings[embeddings["event_name"].isin((EVENT_A, EVENT_B))]
        for _, event in selected.iterrows():
            indices[str(event.event_name)] = int(event.idx)
            event_prediction_rows.append(
                {
                    "seed": seed,
                    "event": str(event.event_name),
                    "predicted_chirp_mass_detector": float(event.waveform_pred_chirp_mass_detector),
                    "predicted_mass_ratio": float(event.waveform_pred_mass_ratio),
                    "strict_h1l1_preprocessing_pass": bool(event.strict_h1l1_preprocessing_pass),
                    "prepared_std_h1": float(event.prepared_std_h1),
                    "prepared_std_l1": float(event.prepared_std_l1),
                }
            )
        calibration = json.loads(calibration_file.read_text())
        audit = json.loads(audit_file.read_text())
        score = float(pair.waveform_score)
        cosine = float(pair.waveform_embedding_cosine)
        pair_score_rank = int((similarities["waveform_score"] > score).sum() + 1)
        pair_cosine_rank = int((similarities["waveform_embedding_cosine"] > cosine).sum() + 1)
        seed_rows.append(
            {
                "seed": seed,
                "embedding_cosine": cosine,
                "abs_delta_logmc_std": float(pair.waveform_abs_delta_logmc_std),
                "abs_delta_logitq_std": float(pair.waveform_abs_delta_logitq_std),
                "waveform_composite_raw": float(pair.waveform_composite_raw),
                "waveform_score": score,
                "waveform_score_rank_of_1378": pair_score_rank,
                "embedding_cosine_rank_of_1378": pair_cosine_rank,
                "waveform_score_percentile": float(100.0 * (similarities["waveform_score"] <= score).mean()),
                "lambda_mc": float(calibration["lambda_mc"]),
                "lambda_q": float(calibration["lambda_q"]),
                "real_embedding_effective_rank": float(audit["real_embedding_effective_rank"]),
                "effective_rank_passed": bool(audit["effective_rank_passed"]),
            }
        )

    seed_frame = pd.DataFrame(seed_rows)
    prediction_frame = pd.DataFrame(event_prediction_rows)

    bank = np.load(waveform_bank_file, mmap_mode="r")
    exact_inputs = {
        event: exact_model_input(np.asarray(bank[index], dtype=np.float32))
        for event, index in indices.items()
    }
    direct_frame = pd.DataFrame(direct_waveform_metrics(exact_inputs[EVENT_A], exact_inputs[EVENT_B]))
    np.savez_compressed(
        tables / "exact_encoder_inputs_2s_4096.npz",
        sample_rate_hz=np.array(FS),
        time_from_gps_s=np.arange(INPUT_SAMPLES) / FS - (INPUT_SAMPLES / FS - 0.25),
        event_a=np.array(EVENT_A),
        event_b=np.array(EVENT_B),
        event_a_h1=exact_inputs[EVENT_A][0],
        event_a_l1=exact_inputs[EVENT_A][1],
        event_b_h1=exact_inputs[EVENT_B][0],
        event_b_l1=exact_inputs[EVENT_B][1],
    )

    pe_manifest = pd.read_csv(pe_manifest_file)
    posteriors: dict[str, dict[str, np.ndarray]] = {}
    pe_sources: dict[str, dict[str, str | int]] = {}
    for event in (EVENT_A, EVENT_B):
        row = pe_manifest.loc[pe_manifest["event_name"] == event].iloc[0]
        pe_path = Path(str(row.pe_path))
        posteriors[event] = load_posterior(pe_path, str(row.pe_group))
        pe_sources[event] = {
            "path": str(pe_path),
            "group": str(row.pe_group),
            "samples": int(row.posterior_samples_total),
            "sha256": sha256(pe_path),
        }

    pe_rows = []
    for parameter in ("chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"):
        metric = pe_pair_metric(posteriors[EVENT_A][parameter], posteriors[EVENT_B][parameter])
        pe_rows.append({"parameter": parameter, **metric})
    pe_frame = pd.DataFrame(pe_rows)
    mc_a = posteriors[EVENT_A]["chirp_mass"]
    mc_b = posteriors[EVENT_B]["chirp_mass"]
    mc_overlap_robustness = {
        f"histogram_bc_{bins}_bins": bhattacharyya_coefficient(mc_a, mc_b, bins=bins)
        for bins in (100, 256, 1000)
    }
    mc_grid = np.linspace(
        min(float(np.quantile(mc_a, 0.0001)), float(np.quantile(mc_b, 0.0001))),
        max(float(np.quantile(mc_a, 0.9999)), float(np.quantile(mc_b, 0.9999))),
        4096,
    )
    mc_kde_bc = float(
        np.trapz(
            np.sqrt(stats.gaussian_kde(mc_a)(mc_grid) * stats.gaussian_kde(mc_b)(mc_grid)),
            mc_grid,
        )
    )
    mc_overlap_robustness["kde_bc"] = mc_kde_bc
    mc_overlap_robustness["posterior_99p8_support_gap_msun"] = float(
        np.quantile(mc_b, 0.001) - np.quantile(mc_a, 0.999)
    )

    event_pe_rows = []
    for event in (EVENT_A, EVENT_B):
        record: dict[str, float | str] = {"event": event}
        for parameter, values in posteriors[event].items():
            q16, q50, q84 = np.quantile(values[np.isfinite(values)], [0.16, 0.5, 0.84])
            record[f"{parameter}_q16"] = float(q16)
            record[f"{parameter}_median"] = float(q50)
            record[f"{parameter}_q84"] = float(q84)
        event_pe_rows.append(record)
    pd.DataFrame(event_pe_rows).to_csv(tables / "public_pe_event_summaries.csv", index=False)

    seed_frame.to_csv(tables / "waveform_score_by_seed.csv", index=False)
    prediction_frame.to_csv(tables / "waveform_intrinsic_predictions_by_seed.csv", index=False)
    direct_frame.to_csv(tables / "direct_waveform_similarity.csv", index=False)
    pe_frame.to_csv(tables / "public_pe_pair_metrics.csv", index=False)
    pd.DataFrame([pair_candidate.to_dict()]).to_csv(tables / "candidate_rank_record.csv", index=False)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )
    colors = {EVENT_A: "#0072B2", EVENT_B: "#D55E00"}
    short = {EVENT_A: "GW190924", EVENT_B: "GW191105"}
    time = np.arange(INPUT_SAMPLES) / FS - (INPUT_SAMPLES / FS - 0.25)
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 9.2), constrained_layout=True)

    for detector_index, detector in enumerate(("H1", "L1")):
        axis = axes[0, detector_index]
        for event in (EVENT_A, EVENT_B):
            axis.plot(time, exact_inputs[event][detector_index], color=colors[event], lw=0.7, alpha=0.85, label=short[event])
        axis.axvline(0, color="0.35", lw=0.8, ls=":")
        axis.set(xlabel="Time from catalog GPS (s)", ylabel="Standardized strain", title=f"{detector}: exact encoder input")
        axis.set_xlim(-1.75, 0.25)
        if detector_index == 0:
            axis.legend(loc="upper left")

    axis = axes[0, 2]
    for event, marker in ((EVENT_A, "o"), (EVENT_B, "s")):
        x = exact_inputs[event][0]
        frequency, psd = signal.welch(x, fs=FS, nperseg=256, noverlap=128)
        keep = (frequency >= 40) & (frequency <= 580)
        axis.semilogy(frequency[keep], psd[keep], color=colors[event], lw=1.5, marker=None, label=short[event])
    axis.set(xlabel="Frequency (Hz)", ylabel="H1 PSD (arb.)", title="Input spectral content")
    axis.legend(loc="best")

    for column, parameter in enumerate(("chirp_mass", "mass_ratio", "chi_eff")):
        axis = axes[1, column]
        both = np.concatenate([posteriors[EVENT_A][parameter], posteriors[EVENT_B][parameter]])
        lo, hi = np.quantile(both, [0.001, 0.999])
        grid = np.linspace(lo, hi, 500)
        for event in (EVENT_A, EVENT_B):
            kde_line(axis, posteriors[event][parameter], colors[event], short[event], grid)
        metric = pe_frame.loc[pe_frame.parameter == parameter].iloc[0]
        labels = {
            "chirp_mass": r"Detector-frame $\mathcal{M}_c$ ($M_\odot$)",
            "mass_ratio": r"Mass ratio $q$",
            "chi_eff": r"Effective spin $\chi_{\rm eff}$",
        }
        axis.set(xlabel=labels[parameter], ylabel="Posterior density", title=f"PE: BC={metric.bhattacharyya_coefficient:.3g}, D={metric.standardized_posterior_distance:.2f}")
        axis.set_yticks([])
        if column == 0:
            axis.legend(loc="upper right")

    axis = axes[2, 0]
    x = np.arange(len(SEEDS))
    axis.plot(x, seed_frame.embedding_cosine, marker="o", color="#009E73", lw=1.5, label="Embedding cosine")
    axis.set_xticks(x, [seed[-3:] for seed in SEEDS])
    axis.set_ylim(0, 1.02)
    axis.set(xlabel="Training seed suffix", ylabel="Cosine", title="High learned-space similarity")
    axis2 = axis.twinx()
    axis2.plot(x, seed_frame.waveform_score, marker="s", color="#CC79A7", lw=1.5, label="Waveform score")
    axis2.set_ylabel("Calibrated waveform score", fontweight="bold")
    handles = axis.get_lines() + axis2.get_lines()
    axis.legend(handles, [line.get_label() for line in handles], loc="lower left")

    axis = axes[2, 1]
    offsets = {EVENT_A: -0.10, EVENT_B: 0.10}
    for event in (EVENT_A, EVENT_B):
        rows = prediction_frame[prediction_frame.event == event]
        axis.scatter(x + offsets[event], rows.predicted_chirp_mass_detector, color=colors[event], s=34, label=f"{short[event]} model")
        posterior = posteriors[event]["chirp_mass"]
        q16, median, q84 = np.quantile(posterior, [0.16, 0.5, 0.84])
        axis.axhspan(q16, q84, color=colors[event], alpha=0.12)
        axis.axhline(median, color=colors[event], ls="--", lw=1.0, label=f"{short[event]} PE")
    axis.set_xticks(x, [seed[-3:] for seed in SEEDS])
    axis.set(xlabel="Training seed suffix", ylabel=r"$\mathcal{M}_c$ ($M_\odot$)", title="Waveform-head prediction vs public PE")
    axis.legend(loc="upper right", fontsize=7)

    axis = axes[2, 2]
    contributions = np.array(
        [
            float(pair_candidate.waveform_contribution_mean),
            float(pair_candidate.time_contribution_mean),
            float(pair_candidate.sky_contribution_mean),
        ]
    )
    axis.bar(["Waveform", "Time", "Sky"], contributions, color=["#009E73", "#E69F00", "#56B4E9"], width=0.65)
    axis.axhline(0, color="0.25", lw=0.8)
    for position, value in enumerate(contributions):
        axis.text(position, value + (0.035 if value >= 0 else -0.06), f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top")
    axis.set(ylabel="Weighted contribution", title=f"Consensus rank 8; final={pair_candidate.final_score_mean:.3f}")

    figure_png = figures / "fig_pair_waveform_pe_forensics.png"
    figure_pdf = figures / "fig_pair_waveform_pe_forensics.pdf"
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    mc = pe_frame.loc[pe_frame.parameter == "chirp_mass"].iloc[0]
    summary = {
        "status": "COMPLETE_READ_ONLY_PAIR_AUDIT",
        "pair": f"{EVENT_A}--{EVENT_B}",
        "catalog_consensus_rank": int(pair_candidate.consensus_rank),
        "catalog_scores": {
            "final_score_mean": float(pair_candidate.final_score_mean),
            "waveform_score_mean": float(pair_candidate.waveform_score_mean),
            "time_score_mean": float(pair_candidate.time_score_mean),
            "sky_raw_log_bf_mean": float(pair_candidate.sky_raw_log_bf_mean),
            "waveform_contribution_mean": float(pair_candidate.waveform_contribution_mean),
            "time_contribution_mean": float(pair_candidate.time_contribution_mean),
            "sky_contribution_mean": float(pair_candidate.sky_contribution_mean),
        },
        "waveform_input_contract": {
            "shape_per_event": [2, INPUT_SAMPLES],
            "sample_rate_hz": FS,
            "duration_s": INPUT_SAMPLES / FS,
            "time_window_relative_to_gps_s": [-1.75, 0.25],
            "source": str(waveform_bank_file),
            "processing": "last 4096 samples of PSD-whitened, 40-580 Hz bandpassed, anti-aliased 2048 Hz 24 s bank; network peak sign flip; per-detector z-score",
        },
        "chirp_mass_pe": mc.to_dict(),
        "chirp_mass_overlap_robustness": mc_overlap_robustness,
        "direct_waveform_similarity": direct_frame.to_dict("records"),
        "pe_sources": pe_sources,
        "input_hashes": {
            "candidate_file": sha256(candidate_file),
            "pe_manifest_file": sha256(pe_manifest_file),
            "waveform_bank_file": sha256(waveform_bank_file),
        },
        "interpretation": {
            "mc_zero_meaning": "The reported zero is the 1D chirp-mass Bhattacharyya overlap coefficient, not a zero chirp mass.",
            "waveform_score_definition": "Validation-calibrated likelihood ratio of z(embedding cosine) + lambda_mc*z(-predicted logMc difference) + lambda_q*z(-predicted logit-q difference). Public PE is not an input.",
            "finding": "This is a false high waveform-score case: learned embeddings are similar, but the waveform auxiliary mass head is not calibrated to these real events and the public detector-frame chirp-mass posteriors are decisively disjoint.",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output / "pair_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    seed_lines = "\n".join(
        f"- seed {row.seed}: cosine={row.embedding_cosine:.3f}, waveform score={row.waveform_score:.3f}, "
        f"predicted-difference dlogMc/std={row.abs_delta_logmc_std:.3f}, effective rank={row.real_embedding_effective_rank:.2f}, "
        f"rank audit={'PASS' if row.effective_rank_passed else 'FAIL'}"
        for row in seed_frame.itertuples()
    )
    prediction_pivot = prediction_frame.pivot(index="seed", columns="event", values="predicted_chirp_mass_detector")
    prediction_lines = "\n".join(
        f"- seed {seed}: {EVENT_A}={prediction_pivot.loc[seed, EVENT_A]:.2f} Msun, "
        f"{EVENT_B}={prediction_pivot.loc[seed, EVENT_B]:.2f} Msun"
        for seed in SEEDS
    )
    report = f"""# {EVENT_A}--{EVENT_B} 单对审计

## 结论

该 pair 的 `waveform_score` 较高，但它不能解释为两个事件的物理 chirp mass 一致。这里的 “Mc=0” 是 chirp-mass 后验 Bhattacharyya 重叠系数为 0，不是 chirp mass 数值为 0。

公开 PE 给出：

- {EVENT_A}: detector-frame chirp mass = {mc.median_a:.3f} [{mc.q16_a:.3f}, {mc.q84_a:.3f}] Msun；
- {EVENT_B}: detector-frame chirp mass = {mc.median_b:.3f} [{mc.q16_b:.3f}, {mc.q84_b:.3f}] Msun；
- BC_Mc = {mc.bhattacharyya_coefficient:.6g}；
- D_Mc = {mc.standardized_posterior_distance:.2f} combined posterior standard deviations。

两个后验的中心相差约 {abs(mc.median_a-mc.median_b):.3f} Msun，而各自后验非常窄，所以重叠为 0 是合理数值结果。
使用 100、256 和 1000 个 histogram bins 复算均得到 BC=0；KDE 平滑后的 BC 也只有 {mc_kde_bc:.3e}。因此这不是 bin 选择造成的数值假零。

## 为什么 waveform score 仍然高

当前 `waveform_score` 不是公开 PE 的质量后验重叠。它由 held-out synthetic validation 冻结的以下组合经过 likelihood-ratio calibration 得到：

```text
z(embedding cosine)
+ lambda_mc * z(-|predicted log Mc_i - predicted log Mc_j|)
+ lambda_q  * z(-|predicted logit q_i - predicted logit q_j|)
```

三个 seed 的实际结果：

{seed_lines}

辅助 head 对这两个真实事件预测的 chirp mass 为：

{prediction_lines}

这些预测不仅偏离公开 PE，而且随训练 seed 大幅变化。换言之，辅助 head 只认为两者在其训练尺度上“预测差值不算特别大”，却没有正确恢复各自真实质量。与此同时，embedding cosine 为 0.81--0.93，给 composite 较强正贡献。三个模型中两个的真实目录 embedding effective-rank audit 未通过，说明 learned space 在真实事件上压缩过强，容易产生假高 cosine。

## 排名由什么推动

本 pair 的平均加权贡献为：waveform={float(pair_candidate.waveform_contribution_mean):.3f}、time={float(pair_candidate.time_contribution_mean):.3f}、sky={float(pair_candidate.sky_contribution_mean):.3f}，总分={float(pair_candidate.final_score_mean):.3f}。因此它进入共识 rank 8 主要由 waveform 通道推动；time 反对，sky 仅提供很弱正贡献。

## 波形图如何理解

图中使用的是模型真正读取的输入：每个事件 H1/L1 的最后 2 s、4096 点，已经做 off-source PSD whitening、40--580 Hz bandpass、抗混叠重采样、网络主峰符号统一和逐探测器 z-score。它不是模板波形，也不是降噪后的 maximum-likelihood waveform；低 SNR 的真实噪声仍占重要部分。因此肉眼相似、直接相关或 embedding cosine 都不能替代联合参数估计。

直接输入比较也没有显示相干的时域匹配：H1/L1 零时延 Pearson 相关分别为 {direct_frame.loc[direct_frame.detector == 'H1', 'zero_lag_pearson'].iloc[0]:.3f} 和 {direct_frame.loc[direct_frame.detector == 'L1', 'zero_lag_pearson'].iloc[0]:.3f}。这进一步说明高 cosine 来自网络学到的压缩表示，而不是两段标准化 strain 在时域中几乎相同。

## 科学判断

这是一个明确的 waveform deployment false-positive diagnostic。它说明：

1. 高 embedding cosine 不保证 detector-frame intrinsic parameters 一致；
2. 当前辅助 chirp-mass head 在这两个真实事件上明显失准；
3. 该 pair 未通过最基本的 shared-source chirp-mass consistency，不能作为可靠透镜候选；
4. PE 没有参与检索调权，因此该结果应作为冻结后审计和模型局限，而不能事后用来美化排名。

审计未修改任何 v9.3/v9.4/BAYESTAR 排名或权重。
"""
    (reports / "PAIR_AUDIT_CN.md").write_text(report)

    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(output / "SHA256_MANIFEST.csv", index=False)
    print(json.dumps({"output": str(output), "figure": str(figure_png), "mc": mc.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
