#!/usr/bin/env python3
"""
EXPERIMENT 1 (reviewer's #1 concern): event-density stress test.

Question: is the high performance of the time/sky priors an artefact of the
sparse ~10-year observation window (9000 events => ~900/yr), which makes
unrelated events far apart in time and sky? At 3G densities (1e4-1e6/yr) many
unrelated events fall within the time-delay window and sky-overlap region of
any given lensed image, so the true partner should be harder to rank first.

Design: FIX the lensed population (same SIS/PM physics, fixed number of injected
lensed pairs used as queries). VARY the background density by scaling the number
of unlensed events in the same ~10-year window. Report R@1, R@10 and a
false-alarm metric vs density.

Memory-efficient: never materialise NxN. For each query event, compute its score
vector against the whole catalog on the fly.
"""
import numpy as np
import pandas as pd
import sys, time
sys.path.insert(0, '.')
from observable_simulator import simulate_catalog, T_START, T_END
from rerank_engine import a90_to_sigma_rad, TimeDelayPrior, true_partner_map

WINDOW_S = T_END - T_START
WINDOW_YR = WINDOW_S / (365.25 * 24 * 3600)


def _rowstd_vec(v):
    finite = np.isfinite(v)
    if finite.sum() > 1:
        mu = v[finite].mean(); sd = v[finite].std() + 1e-9
        out = np.full_like(v, -10.0)
        out[finite] = (v[finite] - mu) / sd
        return out
    return np.zeros_like(v)


def score_query_vs_catalog(qi, df_arrays, time_prior, weights):
    """Return fused score vector of query qi against all catalog events."""
    t, ra, dec, sig, snr = df_arrays
    dt = np.abs(t - t[qi])
    time_lr = time_prior.lr_score(dt)
    # sky
    sd = (np.sin(dec) * np.sin(dec[qi]) +
          np.cos(dec) * np.cos(dec[qi]) * np.cos(ra - ra[qi]))
    sep = np.arccos(np.clip(sd, -1, 1))
    sig2 = sig**2 + sig[qi]**2
    nsep = sep / np.sqrt(sig2)
    sky_step = np.where(nsep < 1.0, 1.0, np.where(nsep < 2.0, 0.5,
               np.where(nsep < 3.0, 0.2, 0.0)))
    sky_log = -0.5 * sep**2 / sig2
    snr_ratio = -np.abs(np.log(snr / snr[qi]))
    chans = dict(time_lr=time_lr, sky_step=sky_step,
                 sky_logoverlap=sky_log, snr_ratio=snr_ratio)
    total = None
    for k, w in weights.items():
        if w == 0:
            continue
        z = _rowstd_vec(chans[k])
        total = z * w if total is None else total + z * w
    total[qi] = -np.inf
    return total


def run_density_point(density_per_yr, n_lensed_each=150, seed=0,
                      weights_combined=None):
    """Build a catalog at the given background density and measure recall."""
    n_bg = int(density_per_yr * WINDOW_YR)
    # generate lensed queries + background
    df = simulate_catalog(n_lensed_each, n_lensed_each, n_bg,
                          detector='ET3', seed=seed, snr_threshold=8.0)
    n = len(df)
    t = df.geocent_time.values.astype(float)
    ra = df.ra.values; dec = df.dec.values
    sig = a90_to_sigma_rad(df.sky_area_90_deg2.values)
    snr = df.network_snr.values
    arrays = (t, ra, dec, sig, snr)

    pmap = true_partner_map(df)
    delays = [abs(t[a] - t[b]) for (a, b) in pmap.values()]
    tp = TimeDelayPrior(delays, window_s=WINDOW_S)

    configs = {
        'time_only': {'time_lr': 1.0},
        'sky_only': {'sky_step': 4.0, 'sky_logoverlap': 1.0},
        'time+sky': weights_combined or {'time_lr': 1.0, 'sky_step': 4.0,
                                         'sky_logoverlap': 1.0, 'snr_ratio': 0.25},
    }
    # queries
    queries = []
    kinds = df.kind.values
    for sid, (a, b) in pmap.items():
        queries.append((a, b, kinds[a])); queries.append((b, a, kinds[a]))

    res = {}
    for cfg_name, w in configs.items():
        hits = {1: 0, 5: 0, 10: 0}
        ranks = []
        for q, partner, fam in queries:
            sv = score_query_vs_catalog(q, arrays, tp, w)
            # rank of partner = 1 + number of events scoring strictly higher
            r = 1 + int(np.sum(sv > sv[partner]))
            ranks.append(r)
            for k in hits:
                if r <= k:
                    hits[k] += 1
        nq = len(queries)
        res[cfg_name] = dict(
            R1=hits[1] / nq, R5=hits[5] / nq, R10=hits[10] / nq,
            median_rank=float(np.median(ranks)))
    return n, res


def main():
    densities = [1e2, 1e3, 1e4, 1e5]   # events/yr; N ~ 1e3..1e6
    n_lensed_each = 120
    seeds = [0, 1, 2]
    rows = []
    for dens in densities:
        for seed in seeds:
            t0 = time.time()
            n, res = run_density_point(dens, n_lensed_each, seed)
            dt = time.time() - t0
            for cfg, m in res.items():
                rows.append(dict(density_per_yr=dens, N_events=n, seed=seed,
                                 config=cfg, **m, runtime_s=dt))
            print(f"density={dens:.0e}/yr N={n:6d} seed={seed} "
                  f"time+sky R@1={res['time+sky']['R1']:.3f} "
                  f"R@10={res['time+sky']['R10']:.3f} "
                  f"sky R@10={res['sky_only']['R10']:.3f} "
                  f"time R@10={res['time_only']['R10']:.3f} ({dt:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv('/tmp/paper/exp/results_density_stress.csv', index=False)
    # aggregate mean+/-std
    agg = df.groupby(['density_per_yr', 'N_events', 'config']).agg(
        R1_m=('R1', 'mean'), R1_s=('R1', 'std'),
        R10_m=('R10', 'mean'), R10_s=('R10', 'std'),
        medrank_m=('median_rank', 'mean')).reset_index()
    agg.to_csv('/tmp/paper/exp/results_density_stress_agg.csv', index=False)
    print("\n=== AGGREGATE (mean over seeds) ===")
    for cfg in ['time_only', 'sky_only', 'time+sky']:
        print(f"\n{cfg}:")
        sub = agg[agg.config == cfg]
        for _, r in sub.iterrows():
            print(f"  {r.density_per_yr:.0e}/yr (N={int(r.N_events):>7d}): "
                  f"R@1={r.R1_m:.3f}±{r.R1_s:.3f}  R@10={r.R10_m:.3f}±{r.R10_s:.3f}  "
                  f"medrank={r.medrank_m:.1f}")
    print("\nsaved results_density_stress[_agg].csv")


if __name__ == '__main__':
    main()
