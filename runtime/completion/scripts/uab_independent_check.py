"""Read-only independent recheck of saved UAB test scores, not model selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def recompute(f, scores, n):
    truth = f.is_true_pair.to_numpy(bool)
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    assert np.isfinite(scores).all()
    matrix = np.full((n, n), -np.inf)
    matrix[i, j] = matrix[j, i] = scores
    qi, qj = np.r_[i[truth], j[truth]], np.r_[j[truth], i[truth]]
    assert len(set(qi)) == len(qi)
    correct = matrix[qi, qj]
    above = (matrix[qi] > correct[:, None]).sum(1)
    tied = (matrix[qi] == correct[:, None]).sum(1)
    result = {'average_precision': float(average_precision_score(truth, scores)),
              'roc_auc': float(roc_auc_score(truth, scores))}
    for k in (1, 10):
        result[f'macro_r_at_{k}'] = float(np.clip((k-above)/tied, 0, 1).mean())
    positive = np.sort(scores[truth])[::-1]
    for target, label in ((.5, '0p5'), (.9, '0p9')):
        threshold = positive[int(np.ceil(target*len(positive)))-1]
        result['false_at_recall_'+label] = int(((~truth) & (scores >= threshold)).sum())
    return result


def main(root, out):
    if not (out/'contracts/INJECTION_EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Run only after injection evaluation is complete')
    columns = {'waveform-new': 'waveform_score', 'time-only': 'time_score',
               'sky-only': 'sky_raw_log_bf', 'three-channel': 'final_score_POSITIVE'}
    subsets = pd.read_parquet(root/'plans/catalog190_subsets.parquet')
    records = []
    for run in ('O3', 'O4a', 'O4b'):
        for arm in ('A_NEUTRAL', 'B_CUE'):
            for seed in (2026091721, 2026091722, 2026091723):
                p = out/f'deployments/{run}/{arm}/evaluation/test/seed_{seed}'
                f = pd.read_parquet(p/'all_pair_scores.parquet')
                events = pd.read_parquet(p/'events.parquet')
                table = pd.read_csv(p/'metrics.csv').set_index('method')
                assert len(events) == 450 and len(f) == 101025 and f.is_true_pair.sum() == 180
                subgroup = pd.read_csv(p/'catalog190_metrics.csv')
                assert len(subgroup) == 2000 and subgroup.draw.nunique() == 500
                assert subgroup.n_events.eq(190).all() and subgroup.n_true_pairs.eq(70).all()
                for draw in (None, 0, 499):
                    if draw is None:
                        sample, n, expected = f, 450, table
                    else:
                        sources = subsets[(subsets.split == 'test') & (subsets.draw == draw)].source_uid
                        keep = events.source_uid.isin(sources).to_numpy()
                        mask = keep[f.idx_i] & keep[f.idx_j]
                        sample = f.loc[mask].copy()
                        remap = np.full(450, -1); remap[keep] = np.arange(190)
                        sample['idx_i'] = remap[sample.idx_i]
                        sample['idx_j'] = remap[sample.idx_j]
                        assert keep.sum() == 190 and len(sample) == 17955 and sample.is_true_pair.sum() == 70
                        n, expected = 190, subgroup[subgroup.draw == draw].set_index('method')
                    for method, column in columns.items():
                        values = recompute(sample, sample[column].to_numpy(float), n)
                        for metric, observed in values.items():
                            saved = float(expected.loc[method, metric])
                            error = abs(observed-saved)
                            if error > 1e-11:
                                raise RuntimeError(f'Metric mismatch: {run}/{arm}/{seed}/{draw}/{method}/{metric}: {error}')
                            records.append(dict(run=run, arm=arm, seed=seed, draw=draw,
                                events=n, method=method, metric=metric, absolute_error=error))
    destination = out/'tables/independent_metric_recheck.csv'
    pd.DataFrame(records).to_csv(destination, index=False, encoding='utf-8-sig')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = dict(state='PASS', read_only=True, selection_changed=False, datasets=18,
                  full_catalog_and_fixed_draws=[450, '190:draw0', '190:draw499'],
                  scalar_checks=len(records), maximum_error=max(r['absolute_error'] for r in records),
                  independent_AP_ROC_implementation='sklearn', R_at_K='direct dense comparison with exact expected ties',
                  F50_F90='positive order statistic, includes all threshold ties',
                  output=str(destination), sha256=digest)
    (out/'contracts/INDEPENDENT_METRIC_RECHECK.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out)
