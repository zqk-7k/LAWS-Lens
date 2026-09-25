#!/usr/bin/env python3
"""EXPERIMENT 2 (Monte-Carlo estimator): realistic rarity & false-alarm burden.

To reach realistic lens fractions (1e-2 .. 1e-4) with enough true pairs for clean
statistics WITHOUT 1e12 pair operations, we:
  - fix a moderate number of TRUE lensed pairs (good recall statistics),
  - represent the unlensed background of size N_bg implicitly,
  - estimate the per-pair FALSE-positive rate above any threshold by Monte-Carlo
    sampling random unlensed pairs, then scale by the true number of unlensed
    pairs ~ N_bg^2 / 2.

This yields catalog-level false candidates (per catalog and per year) and FDR as
a function of threshold and lens fraction, plus recall at a fixed follow-up
budget, all statistically valid at N_bg up to ~1e6.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
from observable_simulator import simulate_catalog, T_START, T_END
from rerank_engine import a90_to_sigma_rad, TimeDelayPrior, true_partner_map

WINDOW_S = T_END - T_START
WINDOW_YR = WINDOW_S / (365.25 * 24 * 3600)


def pair_score(idx_i, idx_j, arrays, tpp, weights, ref_stats):
    """Vectorised fused score for arrays of pair indices (i,j).
    ref_stats: per-channel (mean,std) used for standardisation (precomputed from
    a reference sample so true and false pairs share the same scale)."""
    t, ra, dec, sig, snr = arrays
    dt = np.abs(t[idx_i] - t[idx_j])
    time_lr = tpp.lr_score(dt)
    sd = (np.sin(dec[idx_i]) * np.sin(dec[idx_j]) +
          np.cos(dec[idx_i]) * np.cos(dec[idx_j]) * np.cos(ra[idx_i] - ra[idx_j]))
    sep = np.arccos(np.clip(sd, -1, 1))
    sig2 = sig[idx_i]**2 + sig[idx_j]**2
    nsep = sep / np.sqrt(sig2)
    sky_step = np.where(nsep < 1.0, 1.0, np.where(nsep < 2.0, 0.5,
               np.where(nsep < 3.0, 0.2, 0.0)))
    sky_log = -0.5 * sep**2 / sig2
    snr_ratio = -np.abs(np.log(snr[idx_i] / snr[idx_j]))
    chans = dict(time_lr=time_lr, sky_step=sky_step, sky_logoverlap=sky_log, snr_ratio=snr_ratio)
    total = np.zeros(len(idx_i))
    for k, w in weights.items():
        if w == 0:
            continue
        m, s = ref_stats[k]
        total += w * (chans[k] - m) / s
    return total


def main(n_true_pairs=80, N_bg_list=(5000, 50000, 500000),
         frac_labels=(1e-2, 1e-3, 1e-4), n_false_samples=3_000_000,
         seeds=(0, 1, 2), budget=50):
    W = {'time_lr': 1.0, 'sky_step': 4.0, 'sky_logoverlap': 1.0, 'snr_ratio': 0.25}
    curve = []; budget_rows = []

    for N_bg, frac in zip(N_bg_list, frac_labels):
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            # build a catalog: n_true_pairs lensed systems + N_bg unlensed
            n_sis = n_true_pairs // 2; n_pm = n_true_pairs - n_sis
            df = simulate_catalog(n_sis, n_pm, N_bg, detector='ET3',
                                  seed=seed, snr_threshold=8.0)
            t = df.geocent_time.values.astype(float); ra = df.ra.values; dec = df.dec.values
            sig = a90_to_sigma_rad(df.sky_area_90_deg2.values); snr = df.network_snr.values
            arrays = (t, ra, dec, sig, snr)
            n = len(df)
            pmap = true_partner_map(df)
            n_true = len(pmap)
            delays = [abs(t[a] - t[b]) for (a, b) in pmap.values()]
            tpp = TimeDelayPrior(delays, window_s=WINDOW_S)

            # reference standardisation stats from a random sample of pairs
            si = rng.integers(0, n, 200000); sj = rng.integers(0, n, 200000)
            ok = si != sj
            si, sj = si[ok], sj[ok]
            ref_stats = {}
            # compute raw channel values on the sample
            dt = np.abs(t[si] - t[sj]); tl = tpp.lr_score(dt)
            sd = (np.sin(dec[si]) * np.sin(dec[sj]) + np.cos(dec[si]) * np.cos(dec[sj]) * np.cos(ra[si] - ra[sj]))
            sep = np.arccos(np.clip(sd, -1, 1)); s2 = sig[si]**2 + sig[sj]**2
            nsep = sep / np.sqrt(s2)
            ss = np.where(nsep < 1, 1.0, np.where(nsep < 2, 0.5, np.where(nsep < 3, 0.2, 0.0)))
            slog = -0.5 * sep**2 / s2
            sr = -np.abs(np.log(snr[si] / snr[sj]))
            for k, v in [('time_lr', tl), ('sky_step', ss), ('sky_logoverlap', slog), ('snr_ratio', sr)]:
                vv = v[np.isfinite(v)]
                ref_stats[k] = (vv.mean(), vv.std() + 1e-9)

            # TRUE pair scores
            ti = np.array([a for a, b in pmap.values()])
            tj = np.array([b for a, b in pmap.values()])
            true_scores = pair_score(ti, tj, arrays, tpp, W, ref_stats)

            # FALSE pair scores via Monte-Carlo sampling of unlensed-involving pairs
            unl = np.where(df.kind.values == 'unlensed')[0]
            fi = rng.choice(unl, n_false_samples); fj = rng.choice(unl, n_false_samples)
            ok = fi != fj; fi, fj = fi[ok], fj[ok]
            false_scores = pair_score(fi, fj, arrays, tpp, W, ref_stats)

            # total number of unlensed-involving pairs (the real denominator)
            n_unl = len(unl)
            total_false_pairs = n_unl * (n_unl - 1) / 2.0
            yr = 1.0 / WINDOW_YR

            # threshold sweep
            ths = np.quantile(true_scores, np.linspace(0.0, 1.0, 40))
            for th in ths:
                tp = int(np.sum(true_scores >= th))
                # estimated false-positive rate above threshold
                fp_rate = np.mean(false_scores >= th)
                fp_est = fp_rate * total_false_pairs
                npred = tp + fp_est
                prec = tp / max(npred, 1e-9)
                rec = tp / max(n_true, 1)
                fdr = fp_est / max(npred, 1e-9)
                curve.append(dict(lens_fraction=frac, seed=seed, N_bg=n_unl, N=n,
                                  n_true=n_true, threshold=th, tp=tp,
                                  fp_est=fp_est, n_pred=npred, precision=prec,
                                  recall=rec, fdr=fdr, false_per_year=fp_est * yr,
                                  total_false_pairs=total_false_pairs))
            # recall at fixed follow-up budget:
            # take the global top-`budget` pairs. Among the budget slots, the
            # expected number of TRUE pairs = number of true scores that rank in
            # the global top-budget. Approx: a true pair makes the cut if its
            # score exceeds the (budget)-th highest score overall. The (budget)-th
            # highest overall score ~ quantile of false dist (since false pairs
            # vastly dominate): threshold s.t. fp_rate*total_false_pairs ~ budget.
            target_rate = budget / total_false_pairs
            if target_rate < 1.0:
                th_budget = np.quantile(false_scores, 1 - target_rate)
            else:
                th_budget = -np.inf
            tp_b = int(np.sum(true_scores >= th_budget))
            budget_rows.append(dict(lens_fraction=frac, seed=seed, N_bg=n_unl,
                                    n_true=n_true, recall_at_budget=tp_b / max(n_true, 1),
                                    tp_at_budget=tp_b, budget=budget))
            print(f"frac={frac:.0e} seed={seed} N_bg={n_unl} n_true={n_true} "
                  f"recall@{budget}={tp_b/max(n_true,1):.3f} "
                  f"(total false pairs={total_false_pairs:.2e})", flush=True)

    pd.DataFrame(curve).to_csv('results_rarity_curves.csv', index=False)
    bd = pd.DataFrame(budget_rows); bd.to_csv('results_rarity_budget.csv', index=False)
    agg = bd.groupby('lens_fraction').agg(
        N_bg=('N_bg', 'mean'), n_true=('n_true', 'mean'),
        rab_m=('recall_at_budget', 'mean'), rab_s=('recall_at_budget', 'std')).reset_index()
    print("\n=== RECALL @ FIXED FOLLOW-UP BUDGET (50 candidates/catalog) ===")
    for _, r in agg.iterrows():
        s = r.rab_s if pd.notna(r.rab_s) else 0
        print(f"  frac={r.lens_fraction:.0e} (N_bg={int(r.N_bg)}, {int(r.n_true)} true pairs): "
              f"recall@50={r.rab_m:.3f}+/-{s:.3f}")
    cv = pd.DataFrame(curve)
    print("\n=== FALSE CANDIDATES/YEAR & FDR at threshold achieving ~50% recall ===")
    for frac in frac_labels:
        sub = cv[cv.lens_fraction == frac]
        g = sub.groupby('threshold').agg(recall=('recall', 'mean'),
                                         fpy=('false_per_year', 'mean'),
                                         fdr=('fdr', 'mean'),
                                         prec=('precision', 'mean')).reset_index()
        near = g.iloc[(g.recall - 0.5).abs().argmin()]
        print(f"  frac={frac:.0e}: at recall~{near.recall:.2f}: "
              f"false/yr={near.fpy:.3g}, FDR={near.fdr:.4f}, precision={near.prec:.2e}")
    print("\nsaved results_rarity_curves.csv, results_rarity_budget.csv")


if __name__ == '__main__':
    main()
