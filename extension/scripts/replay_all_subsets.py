"""Independently recompute every archived 190-event draw from frozen pair scores."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

def work(task):
    root, out, run = Path(task[0]), Path(task[1]), task[2]
    historical = Path('/root/autodl-tmp/gw-catalog')
    environments = tuple({Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()})
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0]))
            if path.is_absolute() and path.is_relative_to(historical) and not path.is_relative_to(root):
                if not any(path.is_relative_to(env) for env in environments):
                    raise PermissionError('Subset replay attempted historical-project read')
    sys.addaudithook(guard)
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(root/'runtime/completion/scripts'))
    from uab_independent_check import recompute
    subsets = pd.read_parquet(root/'runtime/training/plans/catalog190_subsets.parquet')
    columns = {'waveform-new': 'waveform_score', 'time-only': 'time_score',
               'sky-only': 'sky_raw_log_bf', 'three-channel': 'final_score_POSITIVE'}
    rows = []
    for split in ('validation', 'test'):
        indices = subsets[subsets.split == split].groupby('draw').source_uid.apply(set).to_dict()
        if set(indices) != set(range(500)):
            raise RuntimeError('Unexpected subset draw IDs')
        for seed in (2026091721, 2026091722, 2026091723):
            folder = root/f'runtime/completion/deployments/{run}/C_PHYSICAL/evaluation/{split}/seed_{seed}'
            pairs = pd.read_parquet(folder/'all_pair_scores.parquet')
            events = pd.read_parquet(folder/'events.parquet')
            expected = pd.read_csv(folder/'catalog190_metrics.csv').set_index(['draw','method'])
            for draw in range(500):
                keep = events.source_uid.isin(indices[draw]).to_numpy()
                sample = pairs.loc[keep[pairs.idx_i] & keep[pairs.idx_j]].copy()
                remap = np.full(len(events), -1)
                remap[keep] = np.arange(int(keep.sum()))
                sample['idx_i'], sample['idx_j'] = remap[sample.idx_i], remap[sample.idx_j]
                if keep.sum() != 190 or len(sample) != 17955 or sample.is_true_pair.sum() != 70:
                    raise RuntimeError('Subset population changed')
                i, j, truth = sample.idx_i.to_numpy(int), sample.idx_j.to_numpy(int), sample.is_true_pair.to_numpy(bool)
                qi, qj = np.r_[i[truth],j[truth]], np.r_[j[truth],i[truth]]
                for method, col in columns.items():
                    score = sample[col].to_numpy(float)
                    values = recompute(sample, score, 190)
                    matrix = np.full((190,190), -np.inf)
                    matrix[i,j] = matrix[j,i] = score
                    correct = matrix[qi,qj]
                    above = (matrix[qi] > correct[:,None]).sum(1)
                    tied = (matrix[qi] == correct[:,None]).sum(1)
                    for k in (5,50):
                        values[f'macro_r_at_{k}'] = float(np.clip((k-above)/tied,0,1).mean())
                    for metric, value in values.items():
                        archived = float(expected.loc[(draw,method), metric])
                        delta = abs(value-archived)
                        if not np.isfinite(delta) or delta >= 1e-11:
                            raise RuntimeError(f'{run}/{split}/{seed}/{draw}/{method}/{metric}: {delta}')
                        rows.append({'run':run,'split':split,'seed':seed,'draw':draw,'method':method,
                                     'metric':metric,'actual':value,'archived':archived,'abs_delta':delta})
            print(run,split,seed,'ALL_500_DRAWS_PASS',flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/(run+'_checks.csv.gz'), index=False)
    result = {'run':run,'checks':len(rows),'max_delta':float(frame.abs_delta.max()),'status':'PASS'}
    (out/(run+'_REPORT.json')).write_text(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--output-tag',default='all_500_subsets')
    a=p.parse_args()
    out=a.release/'verification'/a.output_tag
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(work,[(str(a.release),str(out),r) for r in ('O3','O4a','O4b')]))
    report={'status':'PASS','checks':sum(x['checks'] for x in results),'results':results,
            'wall_seconds':time.monotonic()-start,'draws_per_catalog':500,'splits':['validation','test'],
            'seeds_per_run':3,'methods':4,'overlapping_subsets_are_not_independent_experiments':True,
            'training_or_selection_performed':False}
    (out/'REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
