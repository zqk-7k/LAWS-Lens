from __future__ import annotations

import importlib
import struct
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")

OUT = Path("runs/current_et_qc_word_report_intuitive_20260623_times_bold_labels_no_snr_thresholds")
DOC_ROOT = Path("docs")
OUT.mkdir(parents=True, exist_ok=True)
DOC_ROOT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["PM", "SIS"]
DETECTORS = ["ET"]
FS = 4096.0
EPS = 1e-12

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Tinos", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.dpi": 120,
    }
)


READABLE = {
    "mu_0": "A image magnification",
    "|mu_1|": "B image magnification",
    "mu_total": "Total magnification",
    "m_l": "PM lens mass",
    "sigma_v": "SIS velocity dispersion",
    "y": "Source position y",
    "z_l": "Lens redshift",
    "z_s": "Source redshift",
    "mass_1_source": "Primary black-hole mass",
    "mass_2_source": "Secondary black-hole mass",
    "luminosity_distance": "Luminosity distance",
    "a_1": "Primary spin magnitude",
    "a_2": "Secondary spin magnitude",
    "theta_jn": "Viewing angle",
    "ra": "Right ascension",
    "dec": "Declination",
    "psi": "Polarization angle",
    "phase": "Coalescence phase",
    "tilt_1": "Primary spin tilt",
    "tilt_2": "Secondary spin tilt",
    "phi_12": "Spin azimuth phi_12",
    "phi_jl": "Spin azimuth phi_jl",
    "geocent_time": "Event time",
}


def nice(name: str) -> str:
    return READABLE.get(name, name)


def finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def clipped(vals, lo, hi) -> tuple[np.ndarray, int, int]:
    vals = finite(vals)
    return vals[(vals >= lo) & (vals <= hi)], int(np.sum(vals > hi)), int(np.sum(vals < lo))


def note(ax, text, loc="upper right") -> None:
    pos = {
        "upper right": (0.98, 0.96, "right", "top"),
        "upper left": (0.02, 0.96, "left", "top"),
        "lower right": (0.98, 0.04, "right", "bottom"),
        "lower left": (0.02, 0.04, "left", "bottom"),
    }
    x, y, ha, va = pos[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
    )


def style_axes(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=8.5)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")


def savefig(fig: plt.Figure, stem: str) -> None:
    style_axes(fig)
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fdir(root: Path, family: str) -> Path:
    return root / f"{family}_data_0222"


def udir(root: Path) -> Path:
    return root / "Unlensed_data_0222"


def load_group(family: str, detector: str) -> dict:
    root = Path(base.ROOTS[(family, detector)])
    fd = fdir(root, family)
    ud = udir(root)
    lens = pd.read_csv(fd / "lens.csv")
    lens_params = pd.read_csv(fd / "lens_params.csv")
    source = pd.read_csv(fd / "lensed_source_samples.csv")
    trigger = pd.read_csv(fd / f"{family}_trigger_time_features.csv")
    unlensed_source = pd.read_csv(ud / "source_samples.csv")
    unlensed_trigger = pd.read_csv(ud / "unlensed_trigger_time_features.csv")
    lens = lens.copy()
    lens["abs_mu_0"] = np.abs(lens["mu_0"].to_numpy(dtype=float))
    lens["abs_mu_1"] = np.abs(lens["mu_1"].to_numpy(dtype=float))
    lens["mu_total"] = lens["abs_mu_0"] + lens["abs_mu_1"]
    trigger = trigger.copy()
    trigger["delta_t_days"] = trigger["delta_time_obs"].to_numpy(dtype=float) / 86400.0
    return {
        "family": family,
        "detector": detector,
        "root": root,
        "fdir": fd,
        "udir": ud,
        "lens": lens,
        "lens_params": lens_params,
        "source": source,
        "trigger": trigger,
        "unlensed_source": unlensed_source,
        "unlensed_trigger": unlensed_trigger,
    }


def group_key(detector: str, family: str) -> str:
    return f"{detector}_{family}"


def representative_index(group: dict) -> int:
    tr = group["trigger"]
    lens = group["lens"]
    ok = (
        np.isfinite(tr["delta_time_obs"])
        & (tr["delta_time_obs"] > 24.0)
        & np.isfinite(lens["mu_total"])
        & (lens["mu_total"] > 2.0)
    )
    idx = np.where(ok.to_numpy())[0]
    if len(idx) == 0:
        return int(len(tr) // 2)
    snr_sum = tr["snr_1"].to_numpy(dtype=float) + tr["snr_2"].to_numpy(dtype=float)
    target = np.nanmedian(snr_sum[idx])
    return int(idx[np.nanargmin(np.abs(snr_sum[idx] - target))])


def load_wave_pair(group: dict, idx: int) -> tuple[np.ndarray, np.ndarray]:
    family = group["family"]
    w1 = np.load(group["fdir"] / f"{family}_data_strain_1.npy", mmap_mode="r")[idx]
    w2 = np.load(group["fdir"] / f"{family}_data_strain_2.npy", mmap_mode="r")[idx]
    w1 = np.asarray(w1[0] if w1.ndim == 2 else w1, dtype=float).copy()
    w2 = np.asarray(w2[0] if w2.ndim == 2 else w2, dtype=float).copy()
    for arr in [w1, w2]:
        arr -= np.nanmedian(arr)
        sd = np.nanstd(arr)
        if np.isfinite(sd) and sd > 0:
            arr /= sd
    return w1, w2


def make_fig1(groups: dict[tuple[str, str], dict]) -> pd.DataFrame:
    order = [("ET", "SIS"), ("ET", "PM")]
    fig, axes = plt.subplots(len(order), 2, figsize=(12.2, 5.8), constrained_layout=True)
    rows = []
    for row, (detector, family) in enumerate(order):
        g = groups[(family, detector)]
        idx = representative_index(g)
        w1, w2 = load_wave_pair(g, idx)
        n = len(w1)
        t = np.arange(n) / FS
        peak = int(np.nanargmax(np.abs(w1)))
        lo = max(0, peak - int(0.15 * FS))
        hi = min(n, peak + int(0.15 * FS))
        axes[row, 0].plot(t, w1, lw=0.55, label="A image")
        axes[row, 0].plot(t, w2, lw=0.55, alpha=0.78, label="B image")
        axes[row, 0].set_title(f"{detector} {family}: full noisy strain, sample {idx}")
        axes[row, 0].set_xlabel("time from segment start [s]")
        axes[row, 0].set_ylabel("standardized strain")
        axes[row, 0].legend(loc="upper right")
        zt = (np.arange(lo, hi) - peak) / FS
        axes[row, 1].plot(zt, w1[lo:hi], lw=0.9, label="A image")
        axes[row, 1].plot(zt, w2[lo:hi], lw=0.9, alpha=0.78, label="B image")
        axes[row, 1].set_title(f"{detector} {family}: 0.3 s peak zoom")
        axes[row, 1].set_xlabel("time around A-image peak [s]")
        axes[row, 1].set_ylabel("standardized strain")
        axes[row, 1].legend(loc="upper right")

        src = g["source"].iloc[idx]
        lens = g["lens"].iloc[idx]
        lp = g["lens_params"].iloc[idx]
        tr = g["trigger"].iloc[idx]
        record = {
            "detector": detector,
            "family": family,
            "sample_index": idx,
            "m1_source": src.get("mass_1_source", np.nan),
            "m2_source": src.get("mass_2_source", np.nan),
            "z_s": lp.get("z_s", np.nan),
            "z_l": lp.get("z_l", np.nan),
            "y": lp.get("y", np.nan),
            "mu_0": lens.get("mu_0", np.nan),
            "mu_1": lens.get("mu_1", np.nan),
            "mu_total": lens.get("mu_total", np.nan),
            "delta_t_seconds": tr.get("delta_time_obs", np.nan),
            "delta_t_days": tr.get("delta_t_days", np.nan),
            "SNR_image1": tr.get("snr_1", np.nan),
            "SNR_image2": tr.get("snr_2", np.nan),
            "morse_index_image1": 0,
            "morse_index_image2": 1,
        }
        record["M_L"] = lp.get("m_l", np.nan)
        record["sigma_v"] = lp.get("sigma_v", np.nan)
        rows.append(record)
    fig.suptitle("Representative lensed pairs: waveform sanity check", fontsize=13)
    style_axes(fig)
    fig.savefig(OUT / "Fig1_lensed_pair_example_adaptive_peak.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table1_representative_event_parameters.csv", index=False)
    return df


def snr_series(groups: dict[tuple[str, str], dict], detector: str):
    series = []
    for family in FAMILIES:
        tr = groups[(family, detector)]["trigger"]
        series.append((f"{family} A image", finite(tr["snr_1"])))
        series.append((f"{family} B image", finite(tr["snr_2"])))
    # Current SIS and PM roots use equivalent unlensed trigger tables for a detector; use SIS root as representative.
    un = groups[("SIS", detector)]["unlensed_trigger"]
    series.append(("Unlensed signal", finite(un["snr"])))
    return series


def make_fig2(groups: dict[tuple[str, str], dict]) -> None:
    fig = plt.figure(figsize=(7.6, 7.2), constrained_layout=True)
    gs = GridSpec(2, len(DETECTORS), figure=fig, height_ratios=[2.2, 1.0])
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:olive", "tab:orange"]
    for col, detector in enumerate(DETECTORS):
        series = snr_series(groups, detector)
        allv = np.concatenate([v for _, v in series])
        lo, hi = 0.0, max(np.percentile(allv, 99), 12.0)
        ax = fig.add_subplot(gs[0, col])
        bins = np.linspace(lo, hi, 72)
        for (label, vals), color in zip(series, colors):
            vals_c, _, _ = clipped(vals, lo, hi)
            ax.hist(vals_c, bins=bins, histtype="step", lw=1.4, label=label, color=color)
        ax.set_xlim(lo, hi)
        ax.set_yscale("log")
        ax.set_title(f"{detector}: SNR distribution, p99 display range")
        ax.set_xlabel("network/available SNR")
        ax.set_ylabel("count, log scale")
        note(ax, "x-axis clipped at p99")
        ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, borderaxespad=0.0)
        axb = fig.add_subplot(gs[1, col])
        data = [np.clip(vals, lo, hi) for _, vals in series]
        axb.boxplot(
            data,
            vert=False,
            tick_labels=[s[0] for s in series],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "lightgray", "alpha": 0.75},
            medianprops={"color": "black"},
        )
        axb.set_xlim(lo, hi)
        axb.set_xlabel("SNR, clipped at p99")
    fig.suptitle("Signal strength (SNR): distribution and percentile summary", fontsize=13)
    savefig(fig, "Fig2_SNR_distribution_intuitive")


def make_fig3(groups: dict[tuple[str, str], dict]) -> None:
    order = [("ET", "PM"), ("ET", "SIS")]
    fig, axes = plt.subplots(len(order), 3, figsize=(13.4, 6.0), constrained_layout=True)
    for row, (detector, family) in enumerate(order):
        lens = groups[(family, detector)]["lens"]
        vals_list = [lens["abs_mu_0"], lens["abs_mu_1"], lens["mu_total"]]
        titles = ["A image magnification", "B image magnification", "Total magnification"]
        for col, (ax, vals, title) in enumerate(zip(axes[row], vals_list, titles)):
            vals = finite(vals)
            lo = max(np.percentile(vals, 0.2), 1e-3)
            hi = np.percentile(vals, 99.0)
            vals_c, n_hi, _ = clipped(vals, lo, hi)
            bins = np.logspace(np.log10(lo), np.log10(hi), 58)
            ax.hist(vals_c, bins=bins, color=("tab:blue" if family == "PM" else "tab:green"), alpha=0.78)
            ax.set_xscale("log")
            med = np.median(vals)
            ax.axvline(med, color="black", ls="--", lw=1.1)
            threshold = 2 if title == "Total magnification" else 1
            ax.axvline(threshold, color="tab:red", ls=":", lw=1.2)
            ax.set_xlim(lo, hi)
            ax.set_title(f"{detector} {family} {title}", pad=8)
            ax.set_xlabel(title)
            ax.set_ylabel("count")
            note(ax, f"median={med:.2g}\nreference={threshold}", "upper left")
            note(ax, f"xmax=p99={hi:.2g}\n>{hi:.2g}: {n_hi}/{len(vals)}", "lower right")
    fig.suptitle("Lensing magnification: compact log scale for A image, B image, and total", fontsize=13)
    savefig(fig, "Fig3_magnification_distribution_intuitive")


def lens_main_param(group: dict) -> tuple[str, np.ndarray]:
    lp = group["lens_params"]
    if group["family"] == "PM":
        return "m_l", lp["m_l"].to_numpy(dtype=float)
    return "sigma_v", lp["sigma_v"].to_numpy(dtype=float)


def make_fig4(groups: dict[tuple[str, str], dict]) -> None:
    order = [("ET", "PM"), ("ET", "SIS")]
    fig, axes = plt.subplots(len(order), 3, figsize=(13.4, 6.4), constrained_layout=True)
    for row, (detector, family) in enumerate(order):
        g = groups[(family, detector)]
        tr = g["trigger"]
        lp = g["lens_params"]
        dt = finite(tr["delta_time_obs"])
        param_col, param = lens_main_param(g)
        y = lp["y"].to_numpy(dtype=float)
        ax = axes[row, 0]
        bins = np.logspace(np.log10(dt.min()), np.log10(dt.max()), 72)
        ax.hist(dt, bins=bins, color="tab:purple", alpha=0.78)
        ax.set_xscale("log")
        ax.axvline(24.0, color="tab:red", ls=":", lw=1.2)
        ax.axvline(np.median(dt), color="black", ls="--", lw=1.1)
        ax.set_title(f"{detector} {family} time-delay distribution")
        ax.set_xlabel("A/B time delay [s], log scale")
        ax.set_ylabel("count")
        note(ax, f"median={np.median(dt):.2g}s\nmin={dt.min():.2g}s", "upper left")
        ax = axes[row, 1]
        ax.scatter(param, dt / 86400.0, s=3, alpha=0.22, color="tab:blue")
        ax.set_yscale("log")
        if family == "PM":
            ax.set_xscale("log")
        ax.set_title(f"Time delay vs {nice(param_col)}")
        ax.set_xlabel(nice(param_col))
        ax.set_ylabel("time delay [days], log")
        ax = axes[row, 2]
        ax.scatter(y, dt / 86400.0, s=3, alpha=0.22, color="tab:green")
        ax.set_yscale("log")
        ax.set_title("Time delay vs source position")
        ax.set_xlabel("Source position y")
        ax.set_ylabel("time delay [days], log")
    fig.suptitle("Time delay between A and B images", fontsize=13)
    savefig(fig, "Fig4_time_delay_distribution_intuitive")


def make_fig5(groups: dict[tuple[str, str], dict]) -> None:
    order = [("ET", "PM"), ("ET", "SIS")]
    fig, axes = plt.subplots(len(order), 4, figsize=(13.4, 6.0), constrained_layout=True)
    for row, (detector, family) in enumerate(order):
        g = groups[(family, detector)]
        lp = g["lens_params"]
        main_col, _ = lens_main_param(g)
        cols = [main_col, "y", "z_l", "z_s"]
        for ax, col in zip(axes[row], cols):
            vals = finite(lp[col])
            lo, hi = np.percentile(vals, [0.5, 99.5])
            vals_c, _, _ = clipped(vals, lo, hi)
            ax.hist(vals_c, bins=58, color="tab:cyan", alpha=0.82)
            ax.axvline(np.median(vals), color="black", ls="--", lw=1.1)
            if col == "m_l":
                ax.set_xscale("log")
            else:
                ax.set_xlim(lo, hi)
            ax.set_title(f"{detector} {family} {nice(col)}")
            note(ax, f"median={np.median(vals):.2g}", "upper left")
    fig.suptitle("Lens-parameter distributions", fontsize=13)
    savefig(fig, "Fig5_lens_parameter_distribution_intuitive")


def source_sets(groups: dict[tuple[str, str], dict], detector: str) -> dict[str, pd.DataFrame]:
    return {
        "PM lensed": groups[("PM", detector)]["source"],
        "SIS lensed": groups[("SIS", detector)]["source"],
        "unlensed": groups[("SIS", detector)]["unlensed_source"],
    }


def make_fig6(groups: dict[tuple[str, str], dict]) -> None:
    colors = {"PM lensed": "tab:blue", "SIS lensed": "tab:green", "unlensed": "tab:orange"}
    fig_specs = [
        ("mass_distance", "GW source masses and distance", ["mass_1_source", "mass_2_source", "luminosity_distance"], (12.0, 3.3)),
        ("spin_orientation", "GW source spins and viewing angle", ["a_1", "a_2", "theta_jn"], (12.0, 3.3)),
        ("sky_phase", "GW source sky position and phase", ["ra", "dec", "psi", "phase"], (13.4, 3.3)),
        ("spin_angles_time", "GW source spin angles and event time", ["tilt_1", "tilt_2", "phi_12", "phi_jl", "geocent_time"], (14.2, 3.3)),
    ]
    for suffix, title, cols, size in fig_specs:
        fig, axes = plt.subplots(len(DETECTORS), len(cols), figsize=size, constrained_layout=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[None, :]
        for row, detector in enumerate(DETECTORS):
            sets = source_sets(groups, detector)
            for col_idx, col in enumerate(cols):
                ax = axes[row, col_idx]
                allv = np.concatenate([finite(df[col]) for df in sets.values()])
                lo, hi = np.percentile(allv, [0.5, 99.5])
                bins = np.linspace(lo, hi, 56)
                for name, df in sets.items():
                    vals = finite(df[col])
                    vals = vals[(vals >= lo) & (vals <= hi)]
                    ax.hist(vals, bins=bins, histtype="step", lw=1.25, density=True, color=colors[name], label=name)
                ax.set_title(f"{detector} {nice(col)}")
                ax.set_yticks([])
        axes[0, -1].legend(loc="upper right", fontsize=7, frameon=True)
        fig.suptitle(title, fontsize=12)
        savefig(fig, f"Fig6_GW_source_parameter_distribution_{suffix}_intuitive")


def make_fig7_snr_block_hist(groups: dict[tuple[str, str], dict]) -> None:
    fig, axes = plt.subplots(3, len(DETECTORS), figsize=(6.2, 8.6), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, None]
    for col, detector in enumerate(DETECTORS):
        blocks = []
        un = finite(groups[("SIS", detector)]["unlensed_trigger"]["snr"])
        blocks.append(("Unlensed GW signals", un))
        for family in ["PM", "SIS"]:
            tr = groups[(family, detector)]["trigger"]
            vals = np.concatenate([finite(tr["snr_1"]), finite(tr["snr_2"])])
            blocks.append((f"{family} lensed GW signals", vals))
        all_vals = np.concatenate([vals for _, vals in blocks])
        all_vals = all_vals[all_vals > 0]
        lo = max(np.percentile(all_vals, 0.2), 1e-3)
        hi = np.percentile(all_vals, 99.6)
        bins = np.logspace(np.log10(lo), np.log10(hi), 28)
        for row, (title, vals) in enumerate(blocks):
            ax = axes[row, col]
            vals = finite(vals)
            vals = vals[(vals > 0) & (vals >= lo) & (vals <= hi)]
            ax.hist(vals, bins=bins, color="tab:blue", alpha=0.72, edgecolor="white", linewidth=0.35)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{detector}: {title}")
            ax.set_ylabel("Count (log scale)")
            if row == 2:
                ax.set_xlabel("Optimal SNR")
            else:
                ax.set_xticklabels([])
            ax.set_xlim(lo, hi)
    fig.suptitle("Optimal SNR distributions for unlensed and lensed signals", fontsize=13)
    savefig(fig, "Fig7_SNR_block_histogram_lens_models")


def summarize_variable(label: dict, table: str, df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = np.asarray(df[col], dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        row = {
            **label,
            "table": table,
            "variable": col,
            "count": int(len(vals)),
            "finite_count": int(len(finite_vals)),
            "nan_count": int(np.isnan(vals).sum()),
            "inf_count": int(np.isinf(vals).sum()),
        }
        if len(finite_vals):
            row.update(
                {
                    "mean": float(np.mean(finite_vals)),
                    "std": float(np.std(finite_vals)),
                    "min": float(np.min(finite_vals)),
                    "q01": float(np.quantile(finite_vals, 0.01)),
                    "q05": float(np.quantile(finite_vals, 0.05)),
                    "median": float(np.median(finite_vals)),
                    "q95": float(np.quantile(finite_vals, 0.95)),
                    "q99": float(np.quantile(finite_vals, 0.99)),
                    "max": float(np.max(finite_vals)),
                }
            )
        rows.append(row)
    return rows


def make_tables(groups: dict[tuple[str, str], dict], representative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, detector), g in groups.items():
        label = {"detector": detector, "family": family, "root": str(g["root"])}
        trig = g["trigger"].copy()
        trig["snr_ratio_abs"] = np.maximum(trig["snr_1"], trig["snr_2"]) / np.maximum(np.minimum(trig["snr_1"], trig["snr_2"]), EPS)
        rows += summarize_variable(label, "trigger_time_features", trig, ["delta_time_obs", "delta_t_days", "sigma_delta_time", "snr_1", "snr_2", "snr_ratio_abs"])
        rows += summarize_variable(label, "lens", g["lens"], ["mu_0", "mu_1", "abs_mu_0", "abs_mu_1", "mu_total", "t_d"])
        rows += summarize_variable(label, "lens_params", g["lens_params"], ["m_l", "sigma_v", "theta_E(arcsec)", "y", "z_l", "z_s", "beta_x", "beta_y"])
        rows += summarize_variable(label, "lensed_source_samples", g["source"], ["mass_1_source", "mass_2_source", "luminosity_distance", "a_1", "a_2", "theta_jn", "ra", "dec", "psi", "phase", "geocent_time"])
        rows += summarize_variable(label, "unlensed_trigger_time_features", g["unlensed_trigger"], ["snr", "trigger_time_sigma"])
        rows.append({**label, "table": "quality_flags", "variable": "fraction_mu_total_gt_2", "count": len(g["lens"]), "finite_count": len(g["lens"]), "mean": float(np.mean(g["lens"]["mu_total"] > 2))})
        rows.append({**label, "table": "quality_flags", "variable": "fraction_delta_t_obs_gt_24s", "count": len(g["trigger"]), "finite_count": len(g["trigger"]), "mean": float(np.mean(g["trigger"]["delta_time_obs"] > 24))})
        rows.append({**label, "table": "quality_flags", "variable": "fraction_abs_mu0_gt_1", "count": len(g["lens"]), "finite_count": len(g["lens"]), "mean": float(np.mean(g["lens"]["abs_mu_0"] > 1))})
        rows.append({**label, "table": "quality_flags", "variable": "fraction_abs_mu1_gt_1", "count": len(g["lens"]), "finite_count": len(g["lens"]), "mean": float(np.mean(g["lens"]["abs_mu_1"] > 1))})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "Table2_population_summary.csv", index=False)
    return summary


def p_xml(text="", style=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    return struct.unpack(">II", data[16:24]) if data[:8] == b"\x89PNG\r\n\x1a\n" else (1200, 800)


def image_xml(rid, cx, cy, name, docpr_id):
    wp_ns = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    a_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    pic_ns = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    return f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="{wp_ns}"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{docpr_id}" name="{escape(name)}"/><a:graphic xmlns:a="{a_ns}"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="{pic_ns}"><pic:nvPicPr><pic:cNvPr id="0" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'


def build_docx(groups: dict[tuple[str, str], dict]) -> None:
    sections = [
        ("代表性双像波形", OUT / "Fig1_lensed_pair_example_adaptive_peak.png", "用来看同一个源的 A 像（先到达）和 B 像（后到达）在峰值附近是否形态相似，峰值和幅度是否正常。本版只保留 ET 数据，并对 SIS/PM 各展示一个代表样本。"),
        ("信号强度 SNR 分布", OUT / "Fig2_SNR_distribution_intuitive.png", "用来看 ET 数据中透镜像和未透镜信号强弱的整体分布，并比较各组 SNR 的主体范围和长尾。"),
        ("透镜放大率分布", OUT / "Fig3_magnification_distribution_intuitive.png", "用来看 A 像和 B 像的放大率以及总放大率是否合理，确认样本是否处在强放大双像区域。"),
        ("A/B 像时间延迟分布", OUT / "Fig4_time_delay_distribution_intuitive.png", "用来看 A 像和 B 像之间的时间延迟是否为正、数量级是否合理，以及它和透镜参数的关系。"),
        ("透镜模型参数分布", OUT / "Fig5_lens_parameter_distribution_intuitive.png", "用来看 PM 和 SIS 透镜模型的参数采样范围是否符合设定。"),
        ("引力波源质量和距离分布", OUT / "Fig6_GW_source_parameter_distribution_mass_distance_intuitive.png", "用来看生成引力波源时的质量和距离分布是否正常。"),
        ("引力波源自旋和观测角分布", OUT / "Fig6_GW_source_parameter_distribution_spin_orientation_intuitive.png", "用来看黑洞自旋大小和观测倾角的采样分布。"),
        ("引力波源天空位置和相位分布", OUT / "Fig6_GW_source_parameter_distribution_sky_phase_intuitive.png", "用来看天空位置、偏振角和相位角是否覆盖完整范围。"),
        ("引力波源自旋角和事件时间分布", OUT / "Fig6_GW_source_parameter_distribution_spin_angles_time_intuitive.png", "用来看自旋方向角和事件时间的采样分布是否正常。"),
        ("未透镜与不同透镜模型的 SNR 柱状分布", OUT / "Fig7_SNR_block_histogram_lens_models.png", "用类似论文示意图的方式展示 ET 的 SNR 分布。A/B 两像不再分开，而是把同一透镜模型的两像合并；展示 Unlensed、PM lensed 和 SIS lensed 三块。"),
    ]
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    body = [
        p_xml("当前引力波生成结果 QC 图表说明（优化版）", "Title"),
        p_xml("本版参考 createdata/make_qc_word_report_v2_intuitive.py 的展示方式：减少空白、压缩长尾显示，并在每张图下面保留直观说明。"),
        p_xml("数据范围：当前 gw-catalog 代码 ROOTS 指向的 ET、SIS/PM 最新数据；本版不包含 LIGO。", None),
    ]
    rels, media = [], []
    ridn, docid = 1, 1
    for title, img, desc in sections:
        body.append(p_xml(title, "Heading1"))
        if img.exists():
            rid = f"rId{ridn}"
            ridn += 1
            media_name = f"image{ridn - 1}.png"
            rels.append((rid, media_name))
            media.append((media_name, img.read_bytes()))
            w, h = png_size(img)
            cx = int(6.35 * 914400)
            cy = int(cx * h / w)
            body.append(image_xml(rid, cx, cy, img.name, docid))
            docid += 1
        body.append(p_xml(desc))
    body.append(p_xml("附表", "Heading1"))
    body.append(p_xml("Table1_representative_event_parameters.csv：每组代表性双像事件参数。"))
    body.append(p_xml("Table2_population_summary.csv：源参数、透镜参数、放大率、时间延迟和 SNR 的群体统计量。"))
    sect = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="800" w:right="800" w:bottom="800" w:left="800" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    document_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{w_ns}" xmlns:r="{r_ns}"><w:body>{"".join(body)}{sect}</w:body></w:document>'
    styles_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{w_ns}"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="SimSun" w:hAnsi="Arial"/><w:sz w:val="21"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="160"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Arial" w:eastAsia="SimHei" w:hAnsi="Arial"/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="180" w:after="90"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Arial" w:eastAsia="SimHei" w:hAnsi="Arial"/><w:sz w:val="24"/></w:rPr></w:style></w:styles>'
    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    doc_rels_items = ['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'] + [
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{m}"/>'
        for rid, m in rels
    ]
    doc_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(doc_rels_items) + "</Relationships>"
    docx = OUT / "当前引力波生成结果_QC图表说明_优化版.docx"
    with ZipFile(docx, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml)
        for m, data in media:
            z.writestr(f"word/media/{m}", data)
    md = ["# 当前引力波生成结果 QC 图表说明（优化版）", ""]
    for title, img, desc in sections:
        md += [f"## {title}", "", f"图片：{img.name}", "", desc, ""]
    md += ["## 附表", "", "Table1_representative_event_parameters.csv", "", "Table2_population_summary.csv", ""]
    (OUT / "当前引力波生成结果_QC图表说明_优化版.md").write_text("\n".join(md), encoding="utf-8")
    (DOC_ROOT / "当前引力波生成结果_QC图表说明_优化版.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    groups = {(family, detector): load_group(family, detector) for detector in DETECTORS for family in FAMILIES}
    rep = make_fig1(groups)
    make_fig2(groups)
    make_fig3(groups)
    make_fig4(groups)
    make_fig5(groups)
    make_fig6(groups)
    make_fig7_snr_block_hist(groups)
    summary = make_tables(groups, rep)
    build_docx(groups)
    print(OUT)


if __name__ == "__main__":
    main()
