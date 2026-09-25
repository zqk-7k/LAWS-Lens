from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices

exp_sis = importlib.import_module("scripts.experiments.76_sis_ep50_aux_compare")

OUT = Path("runs/ligo_noisy_sis_diagnostic_20260609")
SIS_ROOT = Path("/root/autodl-tmp/gw_ligo_10000_matchstyle_20260527_091859")
PM_ROOT = Path("data_generation/pm_mass_1e4_1e10_td_min24s_matchroots/LIGO")


def finite_sample_stats(path: Path, n_sample: int = 1024) -> dict:
    x = np.load(path, mmap_mode="r")
    n = x.shape[0]
    idx = np.linspace(0, n - 1, min(n, n_sample), dtype=int)
    vals = np.asarray(x[idx], dtype=np.float32)
    flat = vals.reshape(len(idx), -1)
    abs_flat = np.abs(flat)
    row_std = flat.std(axis=1)
    row_max = abs_flat.max(axis=1)
    return {
        "path": str(path),
        "shape": str(tuple(x.shape)),
        "dtype": str(x.dtype),
        "sample_n": int(len(idx)),
        "finite_fraction_sample": float(np.isfinite(flat).mean()),
        "zero_fraction_sample": float((flat == 0).mean()),
        "all_zero_rows_sample": int(np.sum(row_max == 0)),
        "abs_max_sample": float(row_max.max()),
        "abs_p50_sample": float(np.median(abs_flat)),
        "abs_p99_sample": float(np.percentile(abs_flat, 99)),
        "row_std_p50_sample": float(np.median(row_std)),
        "row_std_p01_sample": float(np.percentile(row_std, 1)),
        "row_std_p99_sample": float(np.percentile(row_std, 99)),
    }


def noisy_pure_alignment(family: str, root: Path, n_sample: int = 512) -> list[dict]:
    rows = []
    for image in (1, 2):
        pure = np.load(root / f"{family}_data_0222" / f"{family}_h_strain_{image}.npy", mmap_mode="r")
        noisy = np.load(root / f"{family}_data_0222" / f"{family}_data_strain_{image}.npy", mmap_mode="r")
        idx = np.linspace(0, pure.shape[0] - 1, min(pure.shape[0], n_sample), dtype=int)
        corrs, noise_to_signal, noisy_to_pure_std = [], [], []
        for i in idx:
            p = np.asarray(pure[i], dtype=np.float32).reshape(-1)
            q = np.asarray(noisy[i], dtype=np.float32).reshape(-1)
            ps = float(p.std()) + 1e-12
            qs = float(q.std()) + 1e-12
            corrs.append(float(np.corrcoef(p, q)[0, 1]))
            noise_to_signal.append(float((q - p).std() / ps))
            noisy_to_pure_std.append(float(qs / ps))
        rows.append({
            "family": family,
            "image": image,
            "sample_n": len(idx),
            "pure_noisy_corr_median": float(np.median(corrs)),
            "pure_noisy_corr_q05": float(np.percentile(corrs, 5)),
            "pure_noisy_corr_q95": float(np.percentile(corrs, 95)),
            "noise_residual_std_over_pure_std_median": float(np.median(noise_to_signal)),
            "noisy_std_over_pure_std_median": float(np.median(noisy_to_pure_std)),
        })
    return rows


def peak_crop_stats(family: str, root: Path, n_sample: int = 1000, target_len: int = 8192) -> list[dict]:
    rows = []
    for image in (1, 2):
        arr = np.load(root / f"{family}_data_0222" / f"{family}_h_strain_{image}.npy", mmap_mode="r")
        idx = np.linspace(0, arr.shape[0] - 1, min(arr.shape[0], n_sample), dtype=int)
        starts = arr.shape[-1] - target_len
        peak_idx = []
        in_crop = []
        for i in idx:
            x = np.asarray(arr[i], dtype=np.float32)
            ch = x.reshape(-1, x.shape[-1])
            per_ch_peak = np.argmax(np.abs(ch), axis=1)
            global_peak = int(per_ch_peak[np.argmax(np.max(np.abs(ch), axis=1))])
            peak_idx.append(global_peak)
            in_crop.append(global_peak >= starts)
        rows.append({
            "family": family,
            "image": image,
            "sample_n": len(idx),
            "raw_len": int(arr.shape[-1]),
            "crop_start_index": int(starts),
            "peak_in_tail_crop_fraction": float(np.mean(in_crop)),
            "peak_index_q01": float(np.percentile(peak_idx, 1)),
            "peak_index_q50": float(np.median(peak_idx)),
            "peak_index_q99": float(np.percentile(peak_idx, 99)),
        })
    return rows


def snr_and_lens_stats(family: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = root / f"{family}_data_0222"
    snr_rows = []
    for image in (1, 2):
        net = np.load(data_dir / f"{family}_optimal_SNR_network_{image}.npy")
        single = np.load(data_dir / f"{family}_optimal_SNR_single_{image}.npy")
        snr_rows.append({
            "family": family,
            "image": image,
            "network_snr_mean": float(np.mean(net)),
            "network_snr_q01": float(np.percentile(net, 1)),
            "network_snr_q05": float(np.percentile(net, 5)),
            "network_snr_q50": float(np.median(net)),
            "network_snr_q95": float(np.percentile(net, 95)),
            "network_snr_q99": float(np.percentile(net, 99)),
            "single_shape": str(single.shape),
            "single_ch0_q50": float(np.median(single[:, 0])),
            "single_ch1_q50": float(np.median(single[:, 1])) if single.ndim == 2 and single.shape[1] > 1 else np.nan,
            "single_ch0_gt_ch1_fraction": float(np.mean(single[:, 0] > single[:, 1])) if single.ndim == 2 and single.shape[1] > 1 else np.nan,
        })
    lens = pd.read_csv(data_dir / "lens.csv")
    lens_df = pd.DataFrame([{
        "family": family,
        "t_d_min": float(lens["t_d"].min()),
        "t_d_q01": float(lens["t_d"].quantile(0.01)),
        "t_d_q05": float(lens["t_d"].quantile(0.05)),
        "t_d_q50": float(lens["t_d"].median()),
        "t_d_q95": float(lens["t_d"].quantile(0.95)),
        "t_d_q99": float(lens["t_d"].quantile(0.99)),
        "t_d_max": float(lens["t_d"].max()),
        "abs_mu0_q50": float(np.abs(lens["mu_0"]).median()),
        "abs_mu1_q50": float(np.abs(lens["mu_1"]).median()),
        "abs_mu1_over_mu0_q01": float((np.abs(lens["mu_1"]) / np.abs(lens["mu_0"])).quantile(0.01)),
        "abs_mu1_over_mu0_q50": float((np.abs(lens["mu_1"]) / np.abs(lens["mu_0"])).median()),
        "abs_mu1_over_mu0_q99": float((np.abs(lens["mu_1"]) / np.abs(lens["mu_0"])).quantile(0.99)),
        "mu1_negative_fraction_morse_phase_pi": float(np.mean(lens["mu_1"] < 0)),
    }])
    return pd.DataFrame(snr_rows), lens_df


def source_stats(family: str, root: Path) -> pd.DataFrame:
    src = pd.read_csv(root / f"{family}_data_0222" / "lensed_source_samples.csv")
    cols = ["luminosity_distance", "mass_1_source", "mass_2_source", "ra", "dec", "theta_jn"]
    rows = []
    for c in cols:
        rows.append({
            "family": family,
            "name": c,
            "mean": float(src[c].mean()),
            "q05": float(src[c].quantile(0.05)),
            "q50": float(src[c].median()),
            "q95": float(src[c].quantile(0.95)),
        })
    return pd.DataFrame(rows)


def waveform_rank_bucket_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = exp_sis.make_cfg("LIGO", "noisy")
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    ds = EvaluationSet(arrays, splits["lensed"]["test"], splits["unlensed"]["test"], cfg)
    gt = ground_truth_partner(ds.meta)
    scores = np.load(cfg.out_dir / "test_scores.npy")
    np.fill_diagonal(scores, -np.inf)
    valid = np.flatnonzero(gt >= 0)
    true_scores = scores[valid, gt[valid]]
    ranks = 1 + np.sum(scores[valid] > true_scores[:, None], axis=1)

    n_lensed = len(splits["lensed"]["test"])
    rank_by_source = []
    for local, src_idx in enumerate(splits["lensed"]["test"]):
        r1 = int(ranks[np.where(valid == local)[0][0]])
        r2 = int(ranks[np.where(valid == local + n_lensed)[0][0]])
        rank_by_source.append({
            "source_index": int(src_idx),
            "rank_l1": r1,
            "rank_l2": r2,
            "rank_min": min(r1, r2),
            "rank_max": max(r1, r2),
            "hit10_both": int(r1 <= 10 and r2 <= 10),
            "hit10_any": int(r1 <= 10 or r2 <= 10),
        })
    rank_df = pd.DataFrame(rank_by_source)

    data_dir = SIS_ROOT / "SIS_data_0222"
    lens = pd.read_csv(data_dir / "lens.csv").reset_index().rename(columns={"index": "source_index"})
    snr1 = np.load(data_dir / "SIS_optimal_SNR_network_1.npy")
    snr2 = np.load(data_dir / "SIS_optimal_SNR_network_2.npy")
    feat = pd.DataFrame({
        "source_index": np.arange(len(snr1)),
        "snr1": snr1,
        "snr2": snr2,
        "snr_min": np.minimum(snr1, snr2),
        "snr_ratio_2_over_1": snr2 / np.maximum(snr1, 1e-12),
    })
    tab = rank_df.merge(lens, on="source_index").merge(feat, on="source_index")
    tab["abs_mu1_over_mu0"] = np.abs(tab["mu_1"]) / np.maximum(np.abs(tab["mu_0"]), 1e-12)

    bucket_rows = []
    for col in ["snr_min", "snr_ratio_2_over_1", "t_d", "abs_mu1_over_mu0"]:
        tab[f"{col}_bucket"] = pd.qcut(tab[col], 5, duplicates="drop")
        g = tab.groupby(f"{col}_bucket", observed=True)
        for bucket, sub in g:
            bucket_rows.append({
                "bucket_feature": col,
                "bucket": str(bucket),
                "n": int(len(sub)),
                "hit10_any": float(sub["hit10_any"].mean()),
                "hit10_both": float(sub["hit10_both"].mean()),
                "rank_min_median": float(sub["rank_min"].median()),
                "rank_max_median": float(sub["rank_max"].median()),
                "snr_min_median": float(sub["snr_min"].median()),
                "t_d_median": float(sub["t_d"].median()),
                "abs_mu_ratio_median": float(sub["abs_mu1_over_mu0"].median()),
            })
    return tab, pd.DataFrame(bucket_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    waveform_rows = []
    for family, root in [("SIS", SIS_ROOT), ("PM", PM_ROOT)]:
        data_dir = root / f"{family}_data_0222"
        for name in [
            f"{family}_h_strain_1.npy",
            f"{family}_h_strain_2.npy",
            f"{family}_data_strain_1.npy",
            f"{family}_data_strain_2.npy",
        ]:
            waveform_rows.append({"family": family, **finite_sample_stats(data_dir / name)})
    pd.DataFrame(waveform_rows).to_csv(OUT / "waveform_integrity_sample_stats.csv", index=False)

    align_rows = noisy_pure_alignment("SIS", SIS_ROOT) + noisy_pure_alignment("PM", PM_ROOT)
    pd.DataFrame(align_rows).to_csv(OUT / "noisy_pure_alignment_sample.csv", index=False)

    peak_rows = peak_crop_stats("SIS", SIS_ROOT) + peak_crop_stats("PM", PM_ROOT)
    pd.DataFrame(peak_rows).to_csv(OUT / "peak_tail_crop_sample.csv", index=False)

    snrs, lenses, sources = [], [], []
    for family, root in [("SIS", SIS_ROOT), ("PM", PM_ROOT)]:
        snr_df, lens_df = snr_and_lens_stats(family, root)
        snrs.append(snr_df)
        lenses.append(lens_df)
        sources.append(source_stats(family, root))
    pd.concat(snrs, ignore_index=True).to_csv(OUT / "snr_stats.csv", index=False)
    pd.concat(lenses, ignore_index=True).to_csv(OUT / "lens_stats.csv", index=False)
    pd.concat(sources, ignore_index=True).to_csv(OUT / "source_stats.csv", index=False)

    rank_detail, rank_buckets = waveform_rank_bucket_stats()
    rank_detail.to_csv(OUT / "sis_ligo_noisy_waveform_rank_detail.csv", index=False)
    rank_buckets.to_csv(OUT / "sis_ligo_noisy_waveform_rank_buckets.csv", index=False)

    meta = {
        "sis_root": str(SIS_ROOT),
        "pm_root": str(PM_ROOT),
        "note": "Waveform integrity and alignment use sampled mmap reads. Rank buckets use cached SIS LIGO noisy test_scores.npy from ep50 run.",
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
