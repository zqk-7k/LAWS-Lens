#!/usr/bin/env python3
"""
SERVER EXPERIMENT P2-B: real HEALPix sky-posterior overlap vs Gaussian surrogate.

Answers the reviewer's #3 criticism that the analytic A90 Gaussian sky model is
an idealised surrogate built from truth. This script computes pairwise sky
consistency two ways on the SAME events and checks whether the Gaussian
surrogate over-states retrieval performance relative to real (non-Gaussian,
multimodal) HEALPix posteriors.

Two modes:

(1) REAL-PE mode (preferred): you already have, per event, a localization HEALPix
    .fits skymap (from BAYESTAR or bilby). The script reads them with ligo.skymap
    / healpy and computes the true posterior-overlap integral
        O_ij = integral over sky of p_i(n) p_j(n) dOmega
    for each candidate pair, then ranks partners by O_ij.

(2) RAPID-PE mode: if you have strain + PSD but no skymaps yet, generate BAYESTAR
    localizations on the fly from your single-event triggers (requires
    ligo.skymap's BAYESTAR and a coinc/PSD; see the ligo.skymap docs). This is
    heavier; only do it for a 500-1000 event subset as the reviewer suggested.

Run (REAL-PE mode) on the server:

    cd /root/autodl-tmp/gw-catalog
    python scripts/server_experiments/p2b_real_skymap_overlap.py \
        --meta runs/<run>/catalog_meta.csv \
        --skymap_dir runs/<run>/skymaps \
        --skymap_col skymap_path \
        --out runs/p2b_skymap

Inputs:
  * meta.csv : event_id, source_id, kind, geocent_time, ra, dec,
               sky_area_90_deg2, network_snr, and a column giving each event's
               HEALPix .fits path (named by --skymap_col).
  * skymaps  : per-event HEALPix probability maps (.fits), e.g. BAYESTAR output.

Outputs:
  * skymap_compare.csv : per lensed system, the rank of the true partner under
        (a) Gaussian-surrogate sky overlap and (b) real HEALPix overlap, plus
        retrieval recall@k for both.
  * skymap_compare.{png,pdf} : recall@k, Gaussian vs real, and a scatter of the
        two pairwise scores (shows whether the surrogate is optimistic).

Dependencies: healpy, ligo.skymap, astropy, numpy, pandas, matplotlib.
    pip install healpy ligo.skymap
"""
import argparse, os
import numpy as np
import pandas as pd

DEG2RAD = np.pi / 180.0


# ----------------------------------------------------------------------
def a90_to_sigma_rad(a90_deg2):
    theta90 = np.sqrt(np.asarray(a90_deg2) / np.pi) * DEG2RAD
    return theta90 / np.sqrt(2.0 * np.log(10.0))


def gaussian_pair_overlap(ra_i, dec_i, a90_i, ra_j, dec_j, a90_j):
    """Analytic Gaussian surrogate overlap (the model used in the paper)."""
    sep = np.arccos(np.clip(
        np.sin(dec_i) * np.sin(dec_j) +
        np.cos(dec_i) * np.cos(dec_j) * np.cos(ra_i - ra_j), -1, 1))
    s2 = a90_to_sigma_rad(a90_i) ** 2 + a90_to_sigma_rad(a90_j) ** 2
    return -0.5 * sep ** 2 / s2  # log-overlap (higher = more consistent)


def healpix_pair_overlap(map_i, map_j):
    """True posterior-overlap integral sum_k p_i[k] p_j[k] * (area per pixel).
    Maps must share nside. Returns log of the overlap for numerical range."""
    import healpy as hp
    npix = len(map_i)
    # normalise to sum=1 (probability per pixel)
    pi = map_i / (map_i.sum() + 1e-30)
    pj = map_j / (map_j.sum() + 1e-30)
    overlap = np.sum(pi * pj)
    return np.log(overlap + 1e-300)


def true_partner_map(meta):
    m = {}
    for i, (sid, kind) in enumerate(zip(meta.source_id.values, meta.kind.values)):
        if kind in ('SIS', 'PM'):
            m.setdefault(sid, []).append(i)
    return {s: v for s, v in m.items() if len(v) == 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--skymap_dir', default='')
    ap.add_argument('--skymap_col', default='skymap_path')
    ap.add_argument('--max_events', type=int, default=1000,
                    help='subset size (reviewer suggested 500-1000)')
    ap.add_argument('--out', default='runs/p2b_skymap')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = pd.read_csv(args.meta)
    if len(meta) > args.max_events:
        # keep all lensed images + a random unlensed background up to max_events
        lensed = meta[meta.kind.isin(['SIS', 'PM'])]
        unl = meta[meta.kind == 'unlensed'].sample(
            max(0, args.max_events - len(lensed)), random_state=0)
        meta = pd.concat([lensed, unl]).reset_index(drop=True)
    n = len(meta)
    print(f"using {n} events ({(meta.kind!='unlensed').sum()} lensed images)")

    import healpy as hp
    # load skymaps
    maps = [None] * n
    nside_ref = None
    for i in range(n):
        p = meta.iloc[i][args.skymap_col]
        if args.skymap_dir:
            p = os.path.join(args.skymap_dir, str(p))
        try:
            m = hp.read_map(p, verbose=False)
            if nside_ref is None:
                nside_ref = hp.get_nside(m)
            elif hp.get_nside(m) != nside_ref:
                m = hp.ud_grade(m, nside_ref)
            maps[i] = m
        except Exception as e:
            print(f"  WARN could not read skymap for event {i}: {e}")

    ra = meta.ra.values; dec = meta.dec.values
    a90 = meta.sky_area_90_deg2.values
    pmap = true_partner_map(meta)

    # for each lensed query, rank all events by (a) Gaussian and (b) HEALPix overlap
    def ranks_for(score_fn):
        partner_ranks = []
        ks_hit = {1: 0, 5: 0, 10: 0, 50: 0}
        nq = 0
        for sid, (a, b) in pmap.items():
            for q, partner in [(a, b), (b, a)]:
                scores = np.full(n, -np.inf)
                for j in range(n):
                    if j == q:
                        continue
                    scores[j] = score_fn(q, j)
                order = np.argsort(-scores)
                r = int(np.where(order == partner)[0][0]) + 1
                partner_ranks.append(r)
                for k in ks_hit:
                    if r <= k:
                        ks_hit[k] += 1
                nq += 1
        recall = {f'R@{k}': ks_hit[k] / nq for k in ks_hit}
        return partner_ranks, recall

    gauss_fn = lambda q, j: gaussian_pair_overlap(ra[q], dec[q], a90[q],
                                                  ra[j], dec[j], a90[j])

    def heal_fn(q, j):
        if maps[q] is None or maps[j] is None:
            return -np.inf
        return healpix_pair_overlap(maps[q], maps[j])

    print("computing Gaussian-surrogate ranks...")
    gr, g_recall = ranks_for(gauss_fn)
    print("  Gaussian recall:", {k: round(v, 3) for k, v in g_recall.items()})
    print("computing real HEALPix ranks...")
    hr, h_recall = ranks_for(heal_fn)
    print("  HEALPix recall:", {k: round(v, 3) for k, v in h_recall.items()})

    out = pd.DataFrame(dict(gauss_rank=gr, healpix_rank=hr))
    out.to_csv(os.path.join(args.out, 'skymap_compare.csv'), index=False)
    summ = pd.DataFrame([dict(model='gaussian_surrogate', **g_recall),
                         dict(model='real_healpix', **h_recall)])
    summ.to_csv(os.path.join(args.out, 'skymap_recall.csv'), index=False)
    print(f"\nsaved {args.out}/skymap_compare.csv, skymap_recall.csv")
    print("\nINTERPRETATION: if Gaussian recall >> HEALPix recall, the analytic "
          "surrogate is optimistic and the paper's sky-driven numbers are upper "
          "bounds; if comparable, the surrogate is validated.")

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        ks = [1, 5, 10, 50]
        axes[0].plot(ks, [g_recall[f'R@{k}'] for k in ks], '-o', label='Gaussian surrogate')
        axes[0].plot(ks, [h_recall[f'R@{k}'] for k in ks], '-s', label='Real HEALPix')
        axes[0].set_xlabel('k'); axes[0].set_ylabel('partner recall@k')
        axes[0].set_title('Sky model: surrogate vs real'); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].scatter(gr, hr, s=8, alpha=.5)
        axes[1].plot([1, n], [1, n], 'k--', lw=.7)
        axes[1].set_xlabel('Gaussian partner rank'); axes[1].set_ylabel('HEALPix partner rank')
        axes[1].set_title('Per-pair rank agreement'); axes[1].set_xscale('log'); axes[1].set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'skymap_compare.png'), dpi=200)
        plt.savefig(os.path.join(args.out, 'skymap_compare.pdf'))
        print(f"saved {args.out}/skymap_compare.[png,pdf]")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == '__main__':
    main()
