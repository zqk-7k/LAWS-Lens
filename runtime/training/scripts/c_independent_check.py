"""Independent, post-freeze verification of C metrics and frozen inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def main(root):
    out = root/'completion'
    if not (out/'contracts/INJECTION_EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Complete injection evaluation before independent checks')
    destination = out/'tables/independent_metric_recheck_C.csv'
    report_path = out/'contracts/INDEPENDENT_METRIC_RECHECK_C.json'
    if destination.exists() or report_path.exists():
        raise RuntimeError('Independent audit outputs already exist; no overwrite')
    sys.path.insert(0, str(out/'scripts'))
    from uab_independent_check import recompute

    frozen = json.loads((root/'contracts/FINAL_SCORE_FREEZE.json').read_text())
    changed = [row['path'] for row in frozen['files']
               if digest(row['path']) != row['sha256']]
    if changed:
        raise RuntimeError('Frozen inputs changed: '+repr(changed))

    columns = {'waveform-new': 'waveform_score', 'time-only': 'time_score',
               'sky-only': 'sky_raw_log_bf', 'three-channel': 'final_score_POSITIVE'}
    subsets = pd.read_parquet(root/'plans/catalog190_subsets.parquet')
    records = []
    for run in ('O3', 'O4a', 'O4b'):
        for seed in (2026091721, 2026091722, 2026091723):
            path = out/f'deployments/{run}/C_PHYSICAL/evaluation/test/seed_{seed}'
            pairs = pd.read_parquet(path/'all_pair_scores.parquet')
            events = pd.read_parquet(path/'events.parquet')
            table = pd.read_csv(path/'metrics.csv').set_index('method')
            subgroup = pd.read_csv(path/'catalog190_metrics.csv')
            assert len(events) == 450 and len(pairs) == 101025
            assert pairs.is_true_pair.sum() == 180
            assert len(subgroup) == 2000 and subgroup.draw.nunique() == 500
            assert subgroup.n_events.eq(190).all() and subgroup.n_true_pairs.eq(70).all()
            for draw in (None, 0, 499):
                if draw is None:
                    sample, n, expected = pairs, 450, table
                else:
                    sources = subsets[(subsets.split == 'test') & (subsets.draw == draw)].source_uid
                    keep = events.source_uid.isin(sources).to_numpy()
                    mask = keep[pairs.idx_i] & keep[pairs.idx_j]
                    sample = pairs.loc[mask].copy()
                    remap = np.full(450, -1)
                    remap[keep] = np.arange(190)
                    sample['idx_i'] = remap[sample.idx_i]
                    sample['idx_j'] = remap[sample.idx_j]
                    assert keep.sum() == 190 and len(sample) == 17955
                    assert sample.is_true_pair.sum() == 70
                    n = 190
                    expected = subgroup[subgroup.draw == draw].set_index('method')
                for method, column in columns.items():
                    values = recompute(sample, sample[column].to_numpy(float), n)
                    for metric, value in values.items():
                        error = abs(value-float(expected.loc[method, metric]))
                        if error > 1e-11:
                            raise RuntimeError(f'Metric mismatch: {run}/{seed}/{draw}/{method}/{metric}: {error}')
                        records.append(dict(run=run, arm='C_PHYSICAL', seed=seed,
                            draw=draw, events=n, method=method, metric=metric,
                            absolute_error=error))
    pd.DataFrame(records).to_csv(destination, index=False, encoding='utf-8-sig', mode='x')
    report = dict(state='PASS', post_freeze_read_only=True, selection_changed=False,
        datasets=9, checked_metrics=len(records), maximum_error=max(r['absolute_error'] for r in records),
        full_catalog_and_fixed_draws=[450, '190:draw0', '190:draw499'],
        AP_ROC_implementation='sklearn', retrieval='direct ranks with expected ties',
        F50_F90='positive order statistic, includes threshold ties',
        frozen_files_checked=len(frozen['files']), changed_frozen_files=changed,
        script_sha256=digest(__file__), output=str(destination), output_sha256=digest(destination))
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
