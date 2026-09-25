from __future__ import annotations

import importlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")

OUT_ROOT = Path("runs/current_data_distribution_figures_20260622")
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
DOC_ROOT = Path("docs")

FAMILIES = ["SIS", "PM"]
DETECTORS = ["ET", "LIGO"]
FS = 4096.0
EPS = 1e-12


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)


def family_dir(root: Path, family: str) -> Path:
    return root / f"{family}_data_0222"


def unlensed_dir(root: Path) -> Path:
    return root / "Unlensed_data_0222"


def load_group(family: str, detector: str) -> dict:
    root = Path(base.ROOTS[(family, detector)])
    fdir = family_dir(root, family)
    udir = unlensed_dir(root)
    lens = pd.read_csv(fdir / "lens.csv")
    lens_params = pd.read_csv(fdir / "lens_params.csv")
    source = pd.read_csv(fdir / "lensed_source_samples.csv")
    trigger = pd.read_csv(fdir / f"{family}_trigger_time_features.csv")
    unlensed_trigger = pd.read_csv(udir / "unlensed_trigger_time_features.csv")
    unlensed_source = pd.read_csv(udir / "source_samples.csv")
    lens = lens.copy()
    lens_params = lens_params.copy()
    trigger = trigger.copy()
    lens["abs_mu_0"] = np.abs(lens["mu_0"].to_numpy(dtype=float))
    lens["abs_mu_1"] = np.abs(lens["mu_1"].to_numpy(dtype=float))
    lens["mu_total"] = lens["abs_mu_0"] + lens["abs_mu_1"]
    trigger["delta_t_days"] = trigger["delta_time_obs"].to_numpy(dtype=float) / 86400.0
    trigger["snr_ratio_abs"] = np.maximum(trigger["snr_1"], trigger["snr_2"]) / np.maximum(
        np.minimum(trigger["snr_1"], trigger["snr_2"]), EPS
    )
    return {
        "family": family,
        "detector": detector,
        "root": root,
        "fdir": fdir,
        "udir": udir,
        "lens": lens,
        "lens_params": lens_params,
        "source": source,
        "trigger": trigger,
        "unlensed_trigger": unlensed_trigger,
        "unlensed_source": unlensed_source,
    }


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def finite_series(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def plot_hist(ax, values, bins=60, label=None, logx=False, density=False, alpha=0.55):
    arr = finite_series(values)
    arr = arr[arr > 0] if logx else arr
    if len(arr) == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        return
    if logx:
        bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), bins)
        ax.set_xscale("log")
    ax.hist(arr, bins=bins, alpha=alpha, label=label, density=density)


def plot_cdf(ax, values, label=None, logx=False):
    arr = finite_series(values)
    arr = arr[arr > 0] if logx else arr
    if len(arr) == 0:
        return
    arr = np.sort(arr)
    y = np.arange(1, len(arr) + 1) / len(arr)
    ax.plot(arr, y, label=label, lw=1.8)
    if logx:
        ax.set_xscale("log")


def load_waveform_pair(group: dict, idx: int, mode: str = "noisy") -> tuple[np.ndarray, np.ndarray]:
    family = group["family"]
    tag = "data_strain" if mode == "noisy" else "h_strain"
    w1 = np.load(group["fdir"] / f"{family}_{tag}_1.npy", mmap_mode="r")[idx]
    w2 = np.load(group["fdir"] / f"{family}_{tag}_2.npy", mmap_mode="r")[idx]
    return np.asarray(w1), np.asarray(w2)


def to_1d(w: np.ndarray) -> np.ndarray:
    arr = np.asarray(w, dtype=float)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr - np.nanmedian(arr)
    sd = np.nanstd(arr)
    return arr / sd if np.isfinite(sd) and sd > 0 else arr


def representative_index(group: dict) -> int:
    tr = group["trigger"]
    lens = group["lens"]
    ok = np.isfinite(tr["delta_time_obs"]) & (tr["delta_time_obs"] > 24.0) & np.isfinite(lens["mu_total"]) & (lens["mu_total"] > 2.0)
    candidates = np.where(ok.to_numpy())[0]
    if len(candidates) == 0:
        return int(len(tr) // 2)
    snr_sum = tr["snr_1"].to_numpy(dtype=float) + tr["snr_2"].to_numpy(dtype=float)
    target = np.nanmedian(snr_sum[candidates])
    return int(candidates[np.nanargmin(np.abs(snr_sum[candidates] - target))])


def plot_representative_examples(groups: list[dict]) -> pd.DataFrame:
    rows = []
    for group in groups:
        family = group["family"]
        detector = group["detector"]
        idx = representative_index(group)
        w1, w2 = load_waveform_pair(group, idx, "noisy")
        y1 = to_1d(w1)
        y2 = to_1d(w2)
        n = len(y1)
        x = np.arange(n) / FS
        peak = int(np.nanargmax(np.abs(y1)))
        lo = max(0, peak - int(0.15 * FS))
        hi = min(n, peak + int(0.15 * FS))
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=False)
        axes[0].plot(x, y1, lw=0.7, label="image 1")
        axes[0].plot(x, y2, lw=0.7, label="image 2", alpha=0.8)
        axes[0].set_title(f"{detector} {family} representative lensed pair, sample {idx}")
        axes[0].set_xlabel("time from segment start [s]")
        axes[0].set_ylabel("standardized strain")
        axes[0].legend(loc="upper right")
        zx = (np.arange(lo, hi) - peak) / FS
        axes[1].plot(zx, y1[lo:hi], lw=1.0, label="image 1")
        axes[1].plot(zx, y2[lo:hi], lw=1.0, label="image 2", alpha=0.8)
        axes[1].set_xlabel("time around image-1 peak [s]")
        axes[1].set_ylabel("standardized strain")
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        savefig(fig, f"Fig1_lensed_pair_example_{detector}_{family}")

        source = group["source"].iloc[idx]
        lens = group["lens"].iloc[idx]
        lens_params = group["lens_params"].iloc[idx]
        trigger = group["trigger"].iloc[idx]
        row = {
            "detector": detector,
            "family": family,
            "sample_index": idx,
            "m1_source": source.get("mass_1_source", np.nan),
            "m2_source": source.get("mass_2_source", np.nan),
            "z_s": lens_params.get("z_s", np.nan),
            "z_l": lens_params.get("z_l", np.nan),
            "y": lens_params.get("y", np.nan),
            "mu_0": lens.get("mu_0", np.nan),
            "mu_1": lens.get("mu_1", np.nan),
            "mu_total": lens.get("mu_total", np.nan),
            "delta_t_seconds": trigger.get("delta_time_obs", np.nan),
            "delta_t_days": trigger.get("delta_t_days", np.nan),
            "SNR_image1": trigger.get("snr_1", np.nan),
            "SNR_image2": trigger.get("snr_2", np.nan),
            "morse_index_image1": 0,
            "morse_index_image2": 1,
        }
        if family == "PM":
            row["M_L"] = lens_params.get("m_l", np.nan)
        else:
            row["sigma_v"] = lens_params.get("sigma_v", np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "Table1_representative_event_parameters.csv", index=False)
    return df


def plot_snr_distribution(groups: list[dict]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for row, group in enumerate(groups):
        detector = group["detector"]
        family = group["family"]
        tr = group["trigger"]
        un = group["unlensed_trigger"]
        axh, axc = axes[row]
        plot_hist(axh, tr["snr_1"], bins=60, label="image 1")
        plot_hist(axh, tr["snr_2"], bins=60, label="image 2")
        plot_hist(axh, un["snr"], bins=60, label="unlensed")
        for thr in [8, 10]:
            axh.axvline(thr, color="k", ls="--", lw=0.8)
        axh.set_title(f"{detector} {family} SNR histogram")
        axh.set_xlabel("network SNR")
        axh.set_ylabel("count")
        axh.legend(fontsize=8)
        plot_cdf(axc, tr["snr_1"], label="image 1")
        plot_cdf(axc, tr["snr_2"], label="image 2")
        plot_cdf(axc, un["snr"], label="unlensed")
        for thr in [8, 10]:
            axc.axvline(thr, color="k", ls="--", lw=0.8)
        axc.set_title(f"{detector} {family} SNR CDF")
        axc.set_xlabel("network SNR")
        axc.set_ylabel("CDF")
        axc.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "Fig2_SNR_distribution")


def plot_magnification_distribution(groups: list[dict]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    for ax, group in zip(axes, groups):
        detector = group["detector"]
        family = group["family"]
        lens = group["lens"]
        plot_hist(ax, lens["abs_mu_0"], bins=60, label="abs(mu_0)", logx=True)
        plot_hist(ax, lens["abs_mu_1"], bins=60, label="abs(mu_1)", logx=True)
        plot_hist(ax, lens["mu_total"], bins=60, label="mu_total", logx=True)
        frac_total = float(np.mean(lens["mu_total"] > 2))
        frac_0 = float(np.mean(lens["abs_mu_0"] > 1))
        frac_1 = float(np.mean(lens["abs_mu_1"] > 1))
        ax.set_title(f"{detector} {family} magnification distribution")
        ax.set_xlabel("magnification")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.text(
            0.98,
            0.78,
            f"frac(mu_total>2)={frac_total:.3f}\nfrac(abs(mu0)>1)={frac_0:.3f}\nfrac(abs(mu1)>1)={frac_1:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.tight_layout()
    savefig(fig, "Fig3_magnification_distribution")


def lens_mass_or_sigma(group: dict) -> tuple[str, np.ndarray]:
    lp = group["lens_params"]
    if group["family"] == "PM":
        return "M_L [Msun]", lp["m_l"].to_numpy(dtype=float)
    return "sigma_v [km/s]", lp["sigma_v"].to_numpy(dtype=float)


def plot_time_delay_distribution(groups: list[dict]) -> None:
    with plt.rc_context({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }):
        fig, axes = plt.subplots(4, 3, figsize=(18, 17))
    for row, group in enumerate(groups):
        detector = group["detector"]
        family = group["family"]
        tr = group["trigger"]
        lp = group["lens_params"]
        label, xparam = lens_mass_or_sigma(group)
        dt = tr["delta_time_obs"].to_numpy(dtype=float)
        dtd = tr["delta_t_days"].to_numpy(dtype=float)
        y = lp["y"].to_numpy(dtype=float)
        ax = axes[row, 0]
        plot_hist(ax, dt, bins=70, logx=True)
        ax.axvline(24.0, color="k", ls="--", lw=1.2, label="24 s")
        ax.set_title(f"{detector} {family} delta_t histogram")
        ax.set_xlabel("delta_t_obs [s]")
        ax.set_ylabel("count")
        ax.legend()
        ax = axes[row, 1]
        ax.scatter(xparam, dtd, s=5, alpha=0.22)
        if group["family"] == "PM":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{detector} {family} delta_t vs {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("delta_t_obs [days]")
        ax = axes[row, 2]
        ax.scatter(y, dtd, s=5, alpha=0.22)
        ax.set_yscale("log")
        ax.set_title(f"{detector} {family} delta_t vs y")
        ax.set_xlabel("y")
        ax.set_ylabel("delta_t_obs [days]")
    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=12)
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(12)
    fig.tight_layout()
    for name in ["Fig4_time_delay_distribution", "fig_methods_time_delay_distribution"]:
        for ext in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)


def plot_lens_parameter_distribution(groups: list[dict]) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    for row, group in enumerate(groups):
        detector = group["detector"]
        family = group["family"]
        lp = group["lens_params"]
        main_label, main_values = lens_mass_or_sigma(group)
        params = [
            (main_label, main_values, family == "PM"),
            ("y", lp["y"].to_numpy(dtype=float), False),
            ("z_l", lp["z_l"].to_numpy(dtype=float), False),
            ("z_s", lp["z_s"].to_numpy(dtype=float), False),
        ]
        for col, (label, values, logx) in enumerate(params):
            ax = axes[row, col]
            plot_hist(ax, values, bins=60, logx=logx)
            ax.set_title(f"{detector} {family} {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("count")
    fig.tight_layout()
    savefig(fig, "Fig5_lens_parameter_distribution")


def summary_rows_for_array(group_label: dict, table_name: str, df: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        vals = np.asarray(df[col], dtype=float)
        finite = vals[np.isfinite(vals)]
        row = {
            **group_label,
            "table": table_name,
            "variable": col,
            "count": int(len(vals)),
            "finite_count": int(len(finite)),
            "nan_count": int(np.isnan(vals).sum()),
            "inf_count": int(np.isinf(vals).sum()),
        }
        if len(finite):
            row.update(
                {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "min": float(np.min(finite)),
                    "q01": float(np.quantile(finite, 0.01)),
                    "q05": float(np.quantile(finite, 0.05)),
                    "median": float(np.median(finite)),
                    "q95": float(np.quantile(finite, 0.95)),
                    "q99": float(np.quantile(finite, 0.99)),
                    "max": float(np.max(finite)),
                }
            )
        rows.append(row)
    return rows


def write_population_summary(groups: list[dict]) -> pd.DataFrame:
    rows = []
    for group in groups:
        label = {"detector": group["detector"], "family": group["family"], "root": str(group["root"])}
        rows += summary_rows_for_array(
            label,
            "trigger_time_features",
            group["trigger"],
            ["delta_time_obs", "delta_t_days", "sigma_delta_time", "snr_1", "snr_2", "snr_ratio_abs"],
        )
        rows += summary_rows_for_array(label, "lens", group["lens"], ["mu_0", "mu_1", "abs_mu_0", "abs_mu_1", "mu_total", "t_d"])
        rows += summary_rows_for_array(
            label,
            "lens_params",
            group["lens_params"],
            ["m_l", "sigma_v", "theta_E(arcsec)", "y", "z_l", "z_s", "beta_x", "beta_y"],
        )
        rows += summary_rows_for_array(
            label,
            "lensed_source_samples",
            group["source"],
            ["mass_1_source", "mass_2_source", "luminosity_distance", "ra", "dec", "theta_jn", "geocent_time"],
        )
        rows += summary_rows_for_array(label, "unlensed_trigger_time_features", group["unlensed_trigger"], ["snr", "trigger_time_sigma"])
        rows.append(
            {
                **label,
                "table": "quality_flags",
                "variable": "fraction_mu_total_gt_2",
                "count": len(group["lens"]),
                "finite_count": len(group["lens"]),
                "mean": float(np.mean(group["lens"]["mu_total"] > 2)),
            }
        )
        rows.append(
            {
                **label,
                "table": "quality_flags",
                "variable": "fraction_delta_t_obs_gt_24s",
                "count": len(group["trigger"]),
                "finite_count": len(group["trigger"]),
                "mean": float(np.mean(group["trigger"]["delta_time_obs"] > 24.0)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "Table2_population_summary.csv", index=False)
    return df


def write_report(groups: list[dict], rep: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# 当前数据分布展示包",
        "",
        "生成日期：2026-06-22",
        "",
        "## 数据范围",
        "",
        "本展示包按 `/root/work/create data/展示内容与展示原因.docx` 的最小展示要求生成，覆盖当前代码实际使用的 ET/LIGO、SIS/PM 数据根。",
        "",
        "| detector | family | root | lensed pairs | unlensed events |",
        "|---|---|---|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['detector']} | {g['family']} | `{g['root']}` | {len(g['trigger'])} | {len(g['unlensed_trigger'])} |"
        )
    lines += [
        "",
        "## 输出文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `figures/Fig1_lensed_pair_example_<detector>_<family>.pdf/png` | 每组一个代表性双像 waveform 与 0.3 s zoom |",
        "| `figures/Fig2_SNR_distribution.pdf/png` | image1、image2、unlensed 的 SNR histogram 与 CDF |",
        "| `figures/Fig3_magnification_distribution.pdf/png` | abs(mu_0)、abs(mu_1)、mu_total 分布和关键 fraction |",
        "| `figures/Fig4_time_delay_distribution.pdf/png` | delta_t 分布、delta_t vs M_L/sigma_v、delta_t vs y |",
        "| `figures/Fig5_lens_parameter_distribution.pdf/png` | PM: M_L/y/z_l/z_s；SIS: sigma_v/y/z_l/z_s |",
        "| `tables/Table1_representative_event_parameters.csv` | 每组代表性双像事件参数 |",
        "| `tables/Table2_population_summary.csv` | 群体统计量和质量检查 fraction |",
        "",
        "## 注意",
        "",
        "- LIGO 当前 trigger 表中只有每个 image 的 network/available SNR 字段，没有单独 H1/L1 SNR 列；因此本展示包按现有 `snr_1/snr_2` 和 unlensed `snr` 展示。",
        "- waveform 图使用当前 noisy `data_strain`，并做标准化显示，用于形态 sanity check；物理统计以 CSV 元数据为准。",
    ]
    (OUT_ROOT / "current_data_distribution_report_20260622_cn.md").write_text("\n".join(lines), encoding="utf-8")
    (DOC_ROOT / "current_data_distribution_report_20260622_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    groups = [load_group(fam, det) for det in DETECTORS for fam in FAMILIES]
    rep = plot_representative_examples(groups)
    plot_snr_distribution(groups)
    plot_magnification_distribution(groups)
    plot_time_delay_distribution(groups)
    plot_lens_parameter_distribution(groups)
    summary = write_population_summary(groups)
    write_report(groups, rep, summary)
    print(f"wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
