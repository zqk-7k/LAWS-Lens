#!/usr/bin/env python3
"""
SERVER EXPERIMENT P2-C: real/recolored-noise robustness + real GWTC null test.

Answers the reviewer's #7 ("noise is stationary Gaussian; no real-data null
test"). Two parts:

PART 1 - recolored real-noise injection:
    Take real O3/O4 noise segments from your strain store, recolor to the target
    PSD, inject the SAME simulated lensed/unlensed signals you already use, run
    the trained encoder, and re-measure catalog retrieval. Compares Gaussian-noise
    vs recolored-real-noise R@k to show the encoder's degradation (or robustness).

PART 2 - real GWTC null test:
    Run the full physical reranker on real GWTC events (using the observables you
    already extracted in scripts/gwtc/). The goal is NOT to find lenses but to
    show the pipeline does not manufacture high-confidence false associations:
    report the score distribution and the top candidates, and confirm the known
    historical pair GW170104-GW170814 behaves as expected (parameter/sky
    consistency high, time-delay down-weighted).

This file is a SCAFFOLD: the strain I/O and encoder-forward calls must be wired
to your actual modules (marked TODO). The reranking + null-analysis logic is
complete and reuses your matchgw/aux_priors scorers.

Run:
    cd /root/autodl-tmp/gw-catalog
    # Part 2 (no GPU needed, uses your extracted observables):
    python scripts/server_experiments/p2c_noise_and_null.py null \
        --observables data/gwtc3_observables.csv --out runs/p2c_null
    # Part 1 (needs strain + trained encoder):
    python scripts/server_experiments/p2c_noise_and_null.py noise \
        --signal_bank runs/<run>/signal_catalog.npy \
        --noise_dir   /path/to/real_O3O4_noise \
        --encoder     runs/<run>/encoder.pt \
        --out runs/p2c_noise
"""
import argparse, importlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.aux_priors import a90_to_sigma_rad, observed_sky_pair_features

liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")

DEG2RAD = np.pi / 180.0


# ======================================================================
# PART 2 : real GWTC null test (complete, observable-only)
# ======================================================================
def row_z(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() < 2:
        return out
    out[finite] = (values[finite] - values[finite].mean()) / max(values[finite].std(), 1e-8)
    return out


def gwtc_null_test(observables_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(observables_csv)
    need = ['event_name', 'gps_trigger_time', 'ra_median', 'dec_median',
            'sky_area_90_deg2', 'network_snr']
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"observables csv missing column: {c}")
    n = len(df)
    t = df.gps_trigger_time.values.astype(float)
    ra = np.radians(df.ra_median.values) if df.ra_median.max() > 7 else df.ra_median.values
    dec = np.radians(df.dec_median.values) if abs(df.dec_median).max() > 1.6 else df.dec_median.values
    sig = a90_to_sigma_rad(df.sky_area_90_deg2.values)

    time_obs = pd.DataFrame({
        "trigger_time_obs": t,
        "snr": df.network_snr.values.astype(float),
    })
    sky_obs = pd.DataFrame({
        "ra_obs": ra,
        "dec_obs": dec,
        "sky_area90_deg2": df.sky_area_90_deg2.values.astype(float),
        "sky_sigma_rad": sig,
    })
    gt_none = np.full(n, -1, dtype=np.int32)
    prior = liao.fit_time_lr_from_liao("LIGO", time_obs, gt_none)
    time_score_matrix = liao.time_lr_score_matrix(time_obs, prior)
    sky_features = observed_sky_pair_features(sky_obs)

    # pairwise time+sky scores (same form as the paper's reranker)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            dt_days = abs(t[i] - t[j]) / 86400.0
            sep = float(sky_features["sky_sep_obs"][i, j])
            nsep = float(sky_features["sky_norm_sep"][i, j])
            sky_log = float(sky_features["sky_log_overlap"][i, j])
            sky_step = float(sky_features["sky_step_weight"][i, j])
            time_score = float(time_score_matrix[i, j])
            rows.append(dict(event_i=df.event_name.iloc[i],
                             event_j=df.event_name.iloc[j],
                             delta_t_days=dt_days, ang_sep_deg=np.degrees(sep),
                             sky_norm_sep=nsep, sky_log_overlap=sky_log,
                             sky_step_weight=sky_step,
                             time_score=time_score,
                             combined=0.0))
    pairs = pd.DataFrame(rows)
    pairs["combined"] = row_z(pairs.time_score.values) + row_z(pairs.sky_log_overlap.values)
    pairs.to_csv(os.path.join(out_dir, 'gwtc_null_pairs.csv'), index=False)

    # null-distribution summary
    print(f"GWTC null test: {n} events, {len(pairs)} pairs")
    print("combined-score quantiles:",
          {q: round(float(pairs.combined.quantile(q)), 3)
           for q in [0.5, 0.9, 0.99, 0.999]})
    print("Liao prior:",
          {k: prior[k] for k in ["liao_label", "liao_delay_count", "liao_delay_median_days", "liao_delay_p90_days"]})
    # top candidates (triage shortlist)
    top = pairs.sort_values('combined', ascending=False).head(20)
    top.to_csv(os.path.join(out_dir, 'gwtc_null_top20.csv'), index=False)
    print("top-5 triage candidates:")
    print(top[['event_i', 'event_j', 'delta_t_days', 'ang_sep_deg', 'combined']].head().to_string(index=False))

    # historical candidate check
    def find_pair(a, b):
        m = pairs[((pairs.event_i.str.contains(a)) & (pairs.event_j.str.contains(b))) |
                  ((pairs.event_i.str.contains(b)) & (pairs.event_j.str.contains(a)))]
        return m
    hist = find_pair('170104', '170814')
    if len(hist):
        h = hist.iloc[0]
        sky_rank = int((pairs.sky_log_overlap > h.sky_log_overlap).sum()) + 1
        time_rank = int((pairs.time_score > h.time_score).sum()) + 1
        print(f"\nGW170104-GW170814: dt={h.delta_t_days:.1f}d, ang_sep={h.ang_sep_deg:.1f}deg")
        print(f"  sky rank {sky_rank}/{len(pairs)}, time rank {time_rank}/{len(pairs)}")
        print("  EXPECTED: long time delay -> time down-weighted (high time_rank).")
        print("  NOTE: median-sky+A90 Gaussian cannot capture banana-shaped posterior")
        print("        overlap; use p2b_real_skymap_overlap.py for true sky consistency.")
    else:
        print("\nGW170104-GW170814 not both present in this observables file.")
    print(f"\nsaved {out_dir}/gwtc_null_pairs.csv, gwtc_null_top20.csv")


# ======================================================================
# PART 1 : recolored real-noise injection (scaffold - wire to your modules)
# ======================================================================
def recolor_noise_experiment(args):
    """
    Measures encoder retrieval under recolored REAL noise vs Gaussian noise.
    You must wire three TODOs to your code; the experiment logic is given.
    """
    import numpy as np
    print("=== Recolored real-noise robustness experiment ===")

    # --- TODO 1: load your simulated signal bank (the clean projected strains
    #             you already generate), shape (n_events, n_det, n_samples),
    #             plus their meta (source_id, kind) ---
    # signals = np.load(args.signal_bank)            # (N, C, L)
    # meta = pd.read_csv(args.signal_bank.replace('.npy', '_meta.csv'))
    raise NotImplementedError(
        "Wire TODO 1-3 to your modules. Steps:\n"
        "  1. load clean signal bank + meta (as in your data_generation).\n"
        "  2. for each event: draw a REAL noise segment from --noise_dir, estimate\n"
        "     its PSD, recolor to your target PSD (e.g. with gwpy/pycbc\n"
        "     `interpolate` + `fir_from_transfer`, or scipy), add to the signal.\n"
        "  3. run your trained encoder (--encoder) forward to get embeddings;\n"
        "     then call the SAME retrieval+rerank you use elsewhere and record\n"
        "     R@1/R@10 for (a) Gaussian noise and (b) recolored real noise.\n"
        "Compare the two: the delta is the real-noise robustness result the\n"
        "reviewer asked for. Reuse rerank_engine.py / matchgw.aux_priors so the\n"
        "physical-prior path is identical to the paper.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_null = sub.add_parser('null')
    p_null.add_argument('--observables', required=True)
    p_null.add_argument('--out', default='runs/p2c_null')
    p_noise = sub.add_parser('noise')
    p_noise.add_argument('--signal_bank', required=True)
    p_noise.add_argument('--noise_dir', required=True)
    p_noise.add_argument('--encoder', required=True)
    p_noise.add_argument('--out', default='runs/p2c_noise')
    args = ap.parse_args()
    if args.cmd == 'null':
        gwtc_null_test(args.observables, args.out)
    else:
        recolor_noise_experiment(args)


if __name__ == '__main__':
    main()
