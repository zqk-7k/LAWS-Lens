#!/usr/bin/env python3
"""Development-only ranking stability; never selects a model on these results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main(root, qa, qb):
    events = json.loads((root / 'contracts/EVENTS.json').read_text())
    suffix = f'Q{qa}_Q{qb}'
    pairs = pd.read_csv(root / 'tables' / f'PAIR_CONVERGENCE_{suffix}.csv')
    n = len(events)
    if len(pairs) != n * (n - 1) // 2:
        raise RuntimeError('Incomplete maps/pairs cannot pass a ranking audit')
    matrices = []
    for column in ('Z_from', 'Z_to'):
        matrix = np.full((n, n), -np.inf)
        i, j = pairs.i.to_numpy(int), pairs.j.to_numpy(int)
        matrix[i, j] = matrix[j, i] = pairs[column].to_numpy(float)
        if not np.isfinite(matrix[~np.eye(n, dtype=bool)]).all():
            raise RuntimeError('Nonfinite sky scores')
        matrices.append(matrix)
    rows = []
    for i, event in enumerate(events):
        ranks = [np.argsort(-m[i], kind='stable')[:n-1] for m in matrices]
        companions = [j for j, other in enumerate(events)
                      if i != j and event['source_id'] == other['source_id']]
        if len(companions) > 1:
            raise RuntimeError('This pilot contract expects doublets only')
        rank = [int(np.flatnonzero(order == companions[0])[0]) + 1
                if companions else None for order in ranks]
        rows.append({'query': event['event_uid'], 'source_id': event['source_id'],
                     'companion_query': bool(companions),
                     'rank_from': rank[0], 'rank_to': rank[1],
                     'top10_overlap': len(set(ranks[0][:10]) & set(ranks[1][:10])) / 10,
                     'spearman': float(spearmanr(matrices[0][i, np.arange(n) != i],
                                                matrices[1][i, np.arange(n) != i]).statistic)})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'tables' / f'DEVELOPMENT_RANK_STABILITY_{suffix}.csv', index=False)
    companion = frame[frame.companion_query]
    result = {'development_only': True, 'not_formal_retrieval_result': True,
              'not_used_to_select_physical_assumptions': True,
              'events': n, 'independent_sources': len({e['source_id'] for e in events}),
              'directed_companion_queries': len(companion), 'random_R10': 10 / (n - 1),
              'top10_overlap_mean': float(frame.top10_overlap.mean()),
              'top10_overlap_min': float(frame.top10_overlap.min()),
              'min_query_spearman': float(frame.spearman.min()),
              'max_companion_rank_change': int(abs(companion.rank_to - companion.rank_from).max()),
              'from': {}, 'to': {}}
    for stage in ('from', 'to'):
        result[stage] = {f'R@{k}': float((companion[f'rank_{stage}'] <= k).mean())
                         for k in (1, 5, 10)}
    for label, subset in [('true_companions', pairs[pairs.companion]),
                          ('all_noncompanions', pairs[~pairs.companion]),
                          ('noncompanion_upper1pct_at_lower_q',
                           pairs[~pairs.companion & (pairs.Z_from >= pairs.loc[~pairs.companion, 'Z_from'].quantile(.99))])]:
        result[label] = {'pairs': len(subset), 'sign_flips': int(subset.sign_flip.sum()),
                         'median_abs_delta': float(subset.abs_delta.median()),
                         'q99_abs_delta': float(subset.abs_delta.quantile(.99)),
                         'max_abs_delta': float(subset.abs_delta.max())}
    path = root / 'contracts' / f'DEVELOPMENT_RANK_STABILITY_{suffix}.json'
    if path.exists():
        raise RuntimeError('Audit output already exists')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--qa', type=int, required=True)
    parser.add_argument('--qb', type=int, required=True)
    args = parser.parse_args()
    main(args.root, args.qa, args.qb)
