#!/usr/bin/env python3
"""Frozen O3 problem-pair waveform and public-PE audit; never scoring inputs."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as mass


def run(root):
    config = json.loads((root / "contracts/FINAL_CANDIDATE_BEFORE_FRESH_TEST.json").read_text())["O3_config"]
    old_combined = pd.read_parquet(root / "final_results/gwtc3/CFIX-baseline/C_fixed/all_pairs.parquet")
    new_combined = pd.read_parquet(root / "final_results/gwtc3/MAIN-O3-MCWF-ENCODER-v2-RUNSPEC/C_fixed/all_pairs.parquet")
    top = old_combined.head(10)
    bad = top.loc[(top.pe_mc_bhattacharyya_coefficient < .1) | (top.pe_mc_standardized_distance > 5)]
    pairs = list(zip(bad.event_i, bad.event_j))
    full, events = dev.real_inputs("gwtc3")
    index = events.set_index("event_name").idx.to_dict()
    manifest = pd.read_csv(dev.MAIN / "cache/source_run/data/event_manifest.csv").set_index("event_name")
    names = sorted(set(name for pair in pairs for name in pair))
    posterior = {name: np.asarray(dev.MAINCODE.load_pe_samples(manifest.loc[name])["chirp_mass"]) for name in names}
    predicted = []
    for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
        path = root / f"cache/predictions/{config}/gwtc3/model_{ms}_eval_{es}/real.npz"
        predicted.append(np.load(path)["p"])
    old = pd.read_parquet(root / "final_results/gwtc3/CFIX-baseline/waveform_only/all_pairs.parquet")
    new = pd.read_parquet(root / "final_results/gwtc3/MAIN-O3-MCWF-ENCODER-v2-RUNSPEC/waveform_only/all_pairs.parquet")
    out = root / "audit/cases"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "public_detector_frame_Mc_samples.npz", **posterior)
    np.savez_compressed(out / "normalized_model_inputs_peak2s.npz", **{
        n: np.asarray(full[index[n], :, -4096:]) for n in names})
    colors = ("#376F95", "#AF4157")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(pairs), 3, figsize=(10.8, 2.6*len(pairs)), constrained_layout=True, squeeze=False)
    records = []
    t = np.arange(4096)/2048-1.75
    for k, (a, b) in enumerate(pairs):
        sel = lambda f: f.loc[((f.event_i == a) & (f.event_j == b)) | ((f.event_i == b) & (f.event_j == a))].iloc[0]
        before, after = sel(old), sel(new)
        records.append({"event_i": a, "event_j": b, "old_waveform_rank": int(before.consensus_rank),
            "new_waveform_rank": int(after.consensus_rank), "old_waveform_score": before.waveform_score_mean,
            "new_waveform_score": after.waveform_score_mean,
            "old_Cfixed_rank": int(sel(old_combined).consensus_rank), "new_Cfixed_rank": int(sel(new_combined).consensus_rank),
            "BC_Mc": after.pe_mc_bhattacharyya_coefficient, "D_Mc": after.pe_mc_standardized_distance})
        for detector in range(2):
            ax = axes[k, detector]
            for name, color in zip((a, b), colors):
                values = np.asarray(full[index[name], detector, -4096:], float)
                values = (values-values.mean())/values.std()
                ax.plot(t, values, lw=.35, alpha=.75, color=color, label=name)
            ax.axvline(0, ls=":", color="black", lw=.6)
            ax.set(xlim=(-1.75, .25), xlabel="Time from event GPS (s)", ylabel="Whitened, normalized strain",
                   title=f"{('H1', 'L1')[detector]}: {a}\n{b}")
        ax = axes[k, 2]
        lo = min(np.quantile(posterior[n], .001) for n in (a, b))*.9
        hi = max(np.quantile(posterior[n], .999) for n in (a, b))*1.1
        pp = {n: np.mean([p[index[n]] for p in predicted], axis=0) for n in (a, b)}
        for n in (a, b):
            hi = max(hi, np.exp(mass.LOG_EDGES[min(np.searchsorted(np.cumsum(pp[n]), .995)+1, 64)]))
            lo = min(lo, np.exp(mass.LOG_EDGES[max(np.searchsorted(np.cumsum(pp[n]), .005), 0)]))
        grid = np.geomspace(lo, hi, 1200)
        for name, color in zip((a, b), colors):
            samples = posterior[name]
            samples = samples[np.isfinite(samples) & (samples > 0)]
            ax.plot(grid, gaussian_kde(np.log(samples))(np.log(grid))/grid, color=color, lw=1.2)
            ax.step(np.exp(mass.LOG_CENTERS), pp[name]/np.diff(np.exp(mass.LOG_EDGES)), where="mid", color=color, ls="--", lw=1)
        ax.set(xscale="log", xlim=(lo, hi), xlabel="Detector-frame Mc (solar masses)", ylabel="Probability density",
            title=f"Waveform rank {int(before.consensus_rank)} to {int(after.consensus_rank)}\nMc BC={after.pe_mc_bhattacharyya_coefficient:.3g}; D={after.pe_mc_standardized_distance:.2f}")
    fig.suptitle("Solid: public PE; dashed: learned predictive density, not full PE", fontsize=11)
    fig.legend([Line2D([0], [0], color=c, lw=1.5) for c in colors],
               ["First named event", "Second named event"],
               loc="outside lower center", ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"figures/fig_known_pairs_waveform_Mc.{ext}", dpi=250)
    plt.close(fig)
    dev.csv_write(out / "known_pair_rank_score_Mc.csv", pd.DataFrame(records))
    dev.json_write(out / "VISUAL_AUDIT_CONTRACT.json", {"public_PE": "exact frozen manifest posterior group; detector-frame chirp mass",
        "waveforms": "same peak2s model input, independently centered at event GPS; not physical absolute amplitude",
        "learned_density": "mean of three simulation-supervised categorical heads; not joint PE and not scored using public PE",
        "feedback_role": "descriptive development audit only", "pair_list": pairs,
        "selection": "all catastrophic pairs in frozen official-O3 baseline Cfixed Top10; not a selected subset",
        "historical_GW190924_pairs": "outside current official-O3 scope; not counted as corrected by encoder",
        "source_manifest_sha256": dev.sha(dev.MAIN / "cache/source_run/data/event_manifest.csv")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    run(a.root)
