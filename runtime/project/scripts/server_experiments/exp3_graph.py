#!/usr/bin/env python3
"""EXPERIMENT 3: exact graph-reconstruction metrics + maximum-weight matching.

Fixes the reviewer's two graph criticisms:
  (a) the loose 'contains a complete system' precision lets a giant component
      score 1.0 -> recompute EXACT-match precision/recall, B-cubed, ARI,
      over-merging and fragmentation rates, and singleton precision/recall.
  (b) naive top-k connected components over-merge -> compare against
      maximum-weight matching (the natural doublet-optimal method).
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
from observable_simulator import simulate_catalog, T_START, T_END
from rerank_engine import (build_pair_scores, fuse, TimeDelayPrior,
                           true_partner_map, build_graph_components,
                           graph_metrics_exact, maximum_weight_matching)

WINDOW_S = T_END - T_START


def run(n_sis=250, n_pm=250, n_unlensed=2000, seed=0, detector='ET3'):
    df = simulate_catalog(n_sis, n_pm, n_unlensed, detector=detector,
                          seed=seed, snr_threshold=8.0)
    t = df.geocent_time.values.astype(float)
    pmap = true_partner_map(df)
    delays = [abs(t[a] - t[b]) for (a, b) in pmap.values()]
    tpp = TimeDelayPrior(delays, window_s=WINDOW_S)
    scores = build_pair_scores(df, tpp)
    W = {'time_lr': 1.0, 'sky_step': 4.0, 'sky_logoverlap': 1.0, 'snr_ratio': 0.25}
    fused = fuse(scores, W)

    # choose tau as a high quantile of off-diagonal scores (abstention threshold)
    offdiag = fused[np.isfinite(fused)]
    results = []

    # 1) connected components at k=1,2,5 (no tau) -- reproduces giant-component story
    for topk in [1, 2, 5]:
        comps, _ = build_graph_components(fused, df, topk=topk, tau=None)
        m = graph_metrics_exact(comps, df)
        results.append(dict(method=f'CC_topk{topk}_no_tau', **m))

    # 2) connected components with abstention tau (validation-style high quantile)
    for q in [0.99, 0.995, 0.999]:
        tau = np.quantile(offdiag, q)
        comps, _ = build_graph_components(fused, df, topk=1, tau=tau)
        m = graph_metrics_exact(comps, df)
        results.append(dict(method=f'CC_topk1_tau_q{q}', tau=tau, **m))

    # 3) maximum-weight matching (doublet-optimal) with abstention tau
    for q in [0.99, 0.995, 0.999]:
        tau = np.quantile(offdiag, q)
        comps = maximum_weight_matching(fused, df, tau=tau)
        m = graph_metrics_exact(comps, df)
        results.append(dict(method=f'MWM_tau_q{q}', tau=(tau if tau else np.nan), **m))

    return results, len(pmap), len(df)


def main(seeds=(0, 1, 2)):
    all_rows = []
    for seed in seeds:
        res, n_true, n = run(seed=seed)
        for r in res:
            all_rows.append(dict(seed=seed, n_true_systems=n_true, N=n, **r))
        print(f"seed={seed} done (n_true_systems={n_true}, N={n})", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv('results_graph_metrics.csv', index=False)
    # aggregate across seeds
    metr = ['exact_precision', 'exact_recall', 'bcubed_precision', 'bcubed_recall',
            'overmerge_rate', 'fragmentation_rate', 'singleton_recall',
            'singleton_precision', 'mean_component_size', 'max_component_size']
    agg = df.groupby('method')[metr].mean().reset_index()
    agg.to_csv('results_graph_metrics_agg.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)
    print("\n=== GRAPH RECONSTRUCTION (mean over seeds) ===")
    show = agg[['method', 'exact_precision', 'exact_recall', 'bcubed_precision',
                'bcubed_recall', 'overmerge_rate', 'fragmentation_rate',
                'singleton_precision', 'singleton_recall',
                'max_component_size']].copy()
    for c in show.columns[1:]:
        if show[c].dtype.kind == 'f':
            show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\nKEY CONTRASTS:")
    cc5 = agg[agg.method == 'CC_topk5_no_tau'].iloc[0]
    print(f"  CC top-5 (giant): exact_precision={cc5.exact_precision:.3f} "
          f"exact_recall={cc5.exact_recall:.3f} "
          f"bcubed_recall={cc5.bcubed_recall:.3f} "
          f"fragmentation={cc5.fragmentation_rate:.3f} "
          f"max_comp={int(cc5.max_component_size)} "
          f"(loose precision would be ~1.0 -- this is the artifact)")
    best_mwm = agg[agg.method.str.startswith('MWM')].sort_values('exact_recall').iloc[-1]
    print(f"  Best MWM: {best_mwm.method} exact_precision={best_mwm.exact_precision:.3f} "
          f"exact_recall={best_mwm.exact_recall:.3f} "
          f"bcubed_precision={best_mwm.bcubed_precision:.3f} "
          f"bcubed_recall={best_mwm.bcubed_recall:.3f} "
          f"singleton_precision={best_mwm.singleton_precision:.3f} "
          f"singleton_recall={best_mwm.singleton_recall:.3f}")
    best_cc = agg[agg.method.str.startswith('CC_topk1_tau')].sort_values('exact_recall').iloc[-1]
    print(f"  Best CC+tau: {best_cc.method} exact_precision={best_cc.exact_precision:.3f} "
          f"exact_recall={best_cc.exact_recall:.3f} "
          f"bcubed_precision={best_cc.bcubed_precision:.3f} "
          f"bcubed_recall={best_cc.bcubed_recall:.3f} "
          f"singleton_precision={best_cc.singleton_precision:.3f} "
          f"singleton_recall={best_cc.singleton_recall:.3f}")
    print("\nsaved results_graph_metrics[_agg].csv")


if __name__ == '__main__':
    main()
