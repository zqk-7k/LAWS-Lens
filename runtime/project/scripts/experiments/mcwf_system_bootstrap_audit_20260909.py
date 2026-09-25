#!/usr/bin/env python3
"""Paired source-system uncertainty for fixed ranks and weighted pair metrics."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_nodup_reliability_20260909 as r
n = r.n


def groups(frame):
    count = int(frame.event_count.iloc[0])
    mapping = np.full(count, -1, int)
    labels = []
    pos = frame[frame.is_true_pair]
    for row in pos.itertuples():
        if mapping[row.idx_i] != -1 or mapping[row.idx_j] != -1:
            raise RuntimeError('Bootstrap requires disjoint two-image systems')
        mapping[row.idx_i] = mapping[row.idx_j] = len(labels)
        labels.append(str(row.true_pair_family))
    for i in np.flatnonzero(mapping < 0):
        mapping[i] = len(labels)
        labels.append('background_singleton')
    return mapping, np.asarray(labels)


def pair_draws(frame, values, event_weights):
    ii, jj = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    y = frame.is_true_pair.to_numpy(bool)
    order = np.argsort(-values, kind='stable')
    yy = y[order]
    ends = np.r_[np.flatnonzero(np.diff(values[order]) != 0), len(y)-1]
    answer = []
    for first in range(0, len(event_weights), 32):
        ew = event_weights[first:first+32]
        weight = ew[:, ii]*ew[:, jj]
        weight[:, y] = (ew[:, ii[y]]+ew[:, jj[y]])/2
        weight = weight[:, order]
        tp = np.cumsum(weight*yy, axis=1)
        fp = np.cumsum(weight*(~yy), axis=1)
        tt, ff = tp[:, ends], fp[:, ends]
        precision = np.divide(tt, tt+ff, out=np.zeros_like(tt), where=(tt+ff)>0)
        increments = np.diff(np.c_[np.zeros(len(tt)), tt], axis=1)
        ap = (precision*increments).sum(axis=1)/tt[:, -1]
        output = {'average_precision': ap}
        for target, label in ((.5, 'F50'), (.9, 'F90')):
            location = (tp >= target*tp[:, -1, None]).argmax(axis=1)
            output[label] = fp[np.arange(len(fp)), location]
            complete = np.searchsorted(ends, location)
            output[label+'_whole_tie'] = ff[np.arange(len(ff)), complete]
        answer.append(pd.DataFrame(output))
    return pd.concat(answer, ignore_index=True)


def run(job):
    root, panel, methods, repetitions, query_repetitions, output = job
    root, output = Path(root), Path(output)
    dep, seed, split = panel
    key = f'{dep}/{seed}/{split}'
    draw_key = f'{dep}/{split}' if split.startswith('sept8_reused_') else key
    seed_value = 2026090930+int(hashlib.sha256(draw_key.encode()).hexdigest()[:8], 16)
    frames = {m: pd.read_parquet(root/f'results/{m}/{dep}/{seed}/{split}/pairs.parquet') for m in methods}
    base = frames[methods[0]]
    identity = ['idx_i', 'idx_j', 'is_true_pair', 'true_pair_family']
    for f in frames.values():
        if not f[identity].equals(base[identity]):
            raise RuntimeError('Paired bootstrap tables differ')
    mapping, labels = groups(base)
    rng = np.random.default_rng(seed_value)
    mult = np.zeros((query_repetitions, len(labels)), np.float64)
    for family in np.unique(labels):
        use = np.flatnonzero(labels == family)
        mult[:, use] = rng.multinomial(len(use), np.full(len(use), 1/len(use)), size=query_repetitions)
    ew = mult[:repetitions, mapping]
    results, checks = [], []
    for method, frame in frames.items():
        for mode, column in (('waveform','waveform_score'), ('fusion','final_score')):
            score = frame[column].to_numpy(float)
            draws = pair_draws(frame, score, ew)
            truth = n.cf.fast_metrics(frame, score)
            nominal = pair_draws(frame, score, np.ones((1, len(mapping)))).iloc[0]
            for a, b in (('average_precision','average_precision'), ('F50','false_at_recall_0p5'), ('F90','false_at_recall_0p9')):
                if abs(nominal[a]-truth[b]) > 1e-10:
                    raise RuntimeError('Nominal metric replay failed')
            ranks = pd.read_parquet(root/f'results/{method}/{dep}/{seed}/{split}/{mode}_query_ranks.parquet')
            system = ranks.assign(group=mapping[ranks.query_idx.to_numpy(int)])
            qdraw = {}
            for k in (1, 5, 10):
                temp = system.assign(hit=(system['rank'] <= k).astype(float)).groupby(['family','group']).hit.mean()
                family_draws = []
                for family in temp.index.get_level_values(0).unique():
                    x = temp.loc[family]
                    weights = mult[:, x.index.to_numpy(int)]
                    family_draws.append(weights@x.to_numpy()/weights.sum(axis=1))
                qdraw[f'R{k}'] = np.mean(family_draws, axis=0)
                if abs(temp.groupby('family').mean().mean()-truth[f'macro_r_at_{k}']) > 1e-12:
                    raise RuntimeError('Query-system metric replay failed')
            dest = output/f'draws/{dep}/{seed}/{split}/{method}'
            dest.mkdir(parents=True, exist_ok=True)
            draws.to_parquet(dest/f'{mode}_pair_draws.parquet', index=False)
            pd.DataFrame(qdraw).to_parquet(dest/f'{mode}_query_draws.parquet', index=False)
            values = {**{c: draws[c].to_numpy() for c in draws}, **qdraw}
            nom = {**nominal.to_dict(), **{f'R{k}':truth[f'macro_r_at_{k}'] for k in (1,5,10)}}
            for quantity, x in values.items():
                results.append({'deployment':dep,'seed':seed,'panel':split,'method':method,'mode':mode,
                    'quantity':quantity,'nominal':nom[quantity],'bootstrap_repetitions':len(x),
                    'q025':np.quantile(x,.025),'q975':np.quantile(x,.975),
                    'bootstrap_median':np.median(x),'system_groups':len(labels),
                    'true_systems':int(base.is_true_pair.sum()),'events':len(mapping)})
            checks.append({'deployment':dep,'seed':seed,'panel':split,'method':method,'mode':mode,
                'nominal_replay_pass':True,'same_draws_across_methods':True,'both_queries_sampled_together':True})
    print('BOOTSTRAP_PANEL', key, len(methods), flush=True)
    return results, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',required=True)
    parser.add_argument('--pair-draws',type=int,default=2000)
    parser.add_argument('--query-draws',type=int,default=10000)
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    if args.root.exists():
        raise RuntimeError('Independent audit directory required')
    for folder in ('contracts','tables','draws','scripts','manifest','logs'):
        (args.root/folder).mkdir(parents=True)
    if args.pair_draws > args.query_draws:
        raise ValueError('Pair draws must be a subset of query draws')
    paths = list((args.experiment_root/f'results/{args.methods[0]}').glob('gwtc*/seed_*/**/pairs.parquet'))
    panels = sorted({tuple(p.relative_to(args.experiment_root/f'results/{args.methods[0]}').parts[:3])
                     for p in paths if p.parent.name=='test' or p.parent.name.startswith('sept8_reused_')})
    inputs = []
    for panel in panels:
        for method in args.methods:
            directory=args.experiment_root/'results'/method/Path(*panel)
            for name in ('pairs.parquet','waveform_query_ranks.parquet','fusion_query_ranks.parquet'):
                p=directory/name
                inputs.append({'path':str(p),'sha256':n.sha(p)})
    n.write_csv(args.root/'manifest/INPUT_SHA256.csv',inputs)
    n.write_json(args.root/'contracts/BOOTSTRAP_CONTRACT.json',{'UTC':n.utc(),'adaptive_development':True,
        'experimental_root':str(args.experiment_root),'methods':args.methods,'query_draws':args.query_draws,
        'pair_draws':args.pair_draws,'bootstrap':'Family-stratified source systems; companion images always together; background singleton systems. Fixed-model, fixed-noise, fixed-candidate conditional bootstrap.',
        'pair_weights':'False edges m_i*m_j; true companion edge (m_i+m_j)/2, preserving archived definition. Does not create new physical observations or independent noise blocks.',
        'ties':'Historical optimistic retrieval; AP groups equal scores. Historical stable-index F50/F90 plus conservative full-tie threshold sensitivity.',
        'reused_data_not_locked_test':True,'not_run_to_run_or_noise_population_uncertainty':True,
        'aggregate':'Shared catalog system draws reused across methods and training models; average catalog replicates within each model before across-model summaries.',
        'no_PE_official_input':True,'code_sha256':n.sha(Path(__file__))})
    shutil.copy2(__file__,args.root/'scripts/system_bootstrap_audit.py')
    records,checks=[],[]
    jobs=[(str(args.experiment_root),p,args.methods,args.pair_draws,args.query_draws,str(args.root)) for p in panels]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            a,b=future.result(); records.extend(a);checks.extend(b)
    n.write_csv(args.root/'tables/CONDITIONAL_SYSTEM_CI_PER_PANEL.csv',records)
    n.write_csv(args.root/'tables/BOOTSTRAP_REPLAY_TESTS.csv',checks)
    n.write_json(args.root/'contracts/COMPLETE.json',{'UTC':n.utc(),'panels':len(panels),'checks':len(checks)})


if __name__=='__main__':
    main()
