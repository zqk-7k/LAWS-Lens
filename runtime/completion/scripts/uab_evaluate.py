"""Frozen UAB evaluation: same catalogs, explicit ties, clustered sensitivity."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import uab_completion as c
import uab_scoring as s

BUDGETS = (10, 20, 50, 100, 200, 500)
PRIMARY = 'three-channel'
BOOTSTRAPS = 1000


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def configurations(f, spec):
    w0 = np.asarray(spec['baseline_fusion']['POSITIVE']['weights'])
    wn = np.asarray(spec['final_fusion']['POSITIVE']['weights'])
    matrix = s.inherited().channels(f, f.waveform_score)
    wt = np.array([wn[0], wn[1], 0.]); wt /= wt.sum()
    return {
        'waveform-short': f.Z_wf_short.to_numpy(),
        'waveform-FRT': f.Z_wf_FRT.to_numpy(),
        'waveform-OMC': f.Z_wf_OMC.to_numpy(),
        'waveform-new': f.waveform_score.to_numpy(),
        'time-only': f.time_score.to_numpy(), 'sky-only': f.sky_raw_log_bf.to_numpy(),
        'waveform-time': matrix@wt,
        'time-sky': matrix@np.array([0., wn[1], wn[2]]),
        'short-three-channel': s.inherited().channels(f, f.Z_wf_short)@w0,
        'OMC-three-channel': s.inherited().channels(f, f.Z_wf_OMC)@w0,
        PRIMARY: f.final_score_POSITIVE.to_numpy(),
        'three-channel-zero-allowed': f.final_score_NONNEGATIVE.to_numpy()}


def budget_metrics(values, truth):
    order = np.argsort(-values, kind='stable')
    v, y = values[order], np.asarray(truth, bool)[order]
    result = {}
    for b in BUDGETS:
        k = min(b, len(v)); threshold = v[k-1]
        before, tied = v > threshold, v == threshold
        expected = y[before].sum()+(k-before.sum())*y[tied].mean()
        result.update({f'top{b}_precision': float(expected/k), f'top{b}_false_pairs': float(k-expected),
                       f'top{b}_pair_recall': float(expected/y.sum())})
    return result


def weighted_evaluator(frame, events, values):
    order = np.argsort(-values, kind='stable')
    x = values[order]; truth = frame.is_true_pair.to_numpy(bool)[order]
    ends = np.r_[np.flatnonzero(x[:-1] != x[1:]), len(x)-1]
    _, q = s.query_metrics(frame, values, True)
    def evaluate(pair_weights, event_weights):
        w = pair_weights[order]
        t, f = np.cumsum(w*truth)[ends], np.cumsum(w*~truth)[ends]
        if t[-1] <= 0:
            return np.full(5, np.nan)
        ap = np.sum(np.diff(np.r_[0., t])*t/np.maximum(t+f, 1e-300))/t[-1]
        fp = [f[np.searchsorted(t, recall*t[-1])] for recall in (.5, .9)]
        rw = event_weights[q.query_idx]
        recalls = []
        for k in (1, 10):
            family = []
            for name in sorted(q.family.unique()):
                m = q.family.eq(name).to_numpy()
                family.append(np.average(q.loc[m, f'r_at_{k}'], weights=rw[m]) if rw[m].sum() else np.nan)
            recalls.append(np.mean(family))
        return np.array([ap, *fp, *recalls])
    return evaluate


def cluster_bootstrap(frame, events, values, run, seed, unit):
    field = 'source_uid' if unit == 'source' else 'noise_parent_uid'
    groups, names = pd.factorize(events[field].astype(str), sort=True)
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    truth = frame.is_true_pair.to_numpy(bool)
    if unit == 'source':
        family = events.assign(_g=groups).groupby('_g').family.first()
        strata = [family.index[family.eq(fam)].to_numpy(int) for fam in sorted(family.unique())]
    else:
        strata = [np.arange(len(names))]
    methods = ('waveform-new', PRIMARY)
    functions = {k: weighted_evaluator(frame, events, values[k]) for k in methods}
    rng = np.random.default_rng(c.U.stable('UAB-paired-bootstrap-v1', run, seed, unit))
    draws = {k: [] for k in methods}
    for _ in range(BOOTSTRAPS):
        counts = np.zeros(len(names), int)
        for pool in strata:
            counts += np.bincount(rng.choice(pool, size=len(pool), replace=True), minlength=len(names))
        ew = counts[groups]
        weights = ew[i]*ew[j]
        if unit == 'source':
            weights[truth] = ew[i[truth]]
        for method in methods:
            draws[method].append(functions[method](weights, ew))
    return {k: np.asarray(v) for k, v in draws.items()}, len(names)


def interval_rows(draws, run, arm, seed, unit, clusters):
    rows = []
    for method, array in draws.items():
        valid = np.isfinite(array).all(1)
        if valid.sum() < .9*len(array):
            raise RuntimeError('Too many invalid clustered replicates')
        lo, hi = np.quantile(array[valid], [.025, .975], axis=0)
        for metric, low, high in zip(('average_precision', 'F50', 'F90', 'R1', 'R10'), lo, hi):
            rows.append(dict(run=run, arm=arm, seed=seed, method=method, metric=metric,
                cluster_unit=unit, clusters=clusters, low=float(low), high=float(high),
                replicates=BOOTSTRAPS, valid_replicates=int(valid.sum()),
                interpretation='conditional cluster sensitivity, not independent pair CI',
                warning='Only six test noise parents' if unit == 'noise' else 'Fixed noise and model'))
    return rows


def score_catalog(run, arm, seed, split):
    c.gate(split)
    root = c.deployment(run, arm)
    out = root/f'evaluation/{split}/seed_{seed}'
    if c.check_complete(out/'COMPLETE.json'):
        return
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((root/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
    f, events = s.frame(run, arm, seed, split)
    f = s.apply(root, f, spec)
    f.to_parquet(out/'all_pair_scores.parquet', index=False)
    events.to_parquet(out/'events.parquet', index=False)
    values = configurations(f, spec)
    rows = []
    for method, v in values.items():
        metric, queries = s.query_metrics(f, v, True)
        queries.to_parquet(out/f'{method}_query_ranks.parquet', index=False)
        rows.append(dict(run=run, arm=arm, seed=seed, split=split, method=method,
            n_events=len(events), n_pairs=len(f), n_true_pairs=int(f.is_true_pair.sum()),
            random_R10=10/(len(events)-1), **s.pair_metrics(v, f.is_true_pair), **metric,
            **budget_metrics(v, f.is_true_pair)))
    csv(out/'metrics.csv', pd.DataFrame(rows))
    if split == 'test':
        intervals = []
        for unit in ('source', 'noise'):
            draws, n = cluster_bootstrap(f, events, values, run, seed, unit)
            np.savez_compressed(out/f'bootstrap_{unit}.npz', **draws)
            intervals += interval_rows(draws, run, arm, seed, unit, n)
        csv(out/'cluster_intervals.csv', pd.DataFrame(intervals))
    distributions = []
    for label, use in [('companion', f.is_true_pair), ('noncompanion', ~f.is_true_pair)]:
        for name in ('Z_wf_short', 'Z_wf_FRT', 'Z_wf_OMC', 'waveform_score', 'time_score', 'sky_raw_log_bf', 'joint_BC'):
            x = f.loc[use, name].to_numpy(float)
            distributions.append(dict(run=run, arm=arm, seed=seed, split=split, population=label, channel=name,
                n=len(x), mean=float(x.mean()), SD=float(x.std(ddof=1)),
                **{f'q{q:g}': float(np.quantile(x, q)) for q in (0, .01, .1, .25, .5, .75, .9, .95, .99, .999, 1)}))
    csv(out/'distributions.csv', pd.DataFrame(distributions))
    ood = []
    for name in f:
        if 'ood' in name.lower():
            ood.append(dict(column=name, fraction=float(f[name].mean())))
    csv(out/'ood.csv', pd.DataFrame(ood))
    c.seal(out/'COMPLETE.json', sorted(path for path in out.iterdir() if path.suffix in ('.csv', '.parquet', '.npz')),
           test_retuning=False, calibration_sha256=c.U.sha(root/f'calibration/score/seed_{seed}/SELECTED.json'))
    print(json.dumps(dict(stage='CATALOG_EVALUATED', run=run, arm=arm, seed=seed, split=split)), flush=True)


def subcatalogs(run, arm, seed, split):
    root = c.deployment(run, arm)
    src = root/f'evaluation/{split}/seed_{seed}'
    if c.check_complete(src/'CATALOG190_COMPLETE.json'):
        return
    if not c.check_complete(src/'COMPLETE.json'):
        raise RuntimeError('Full catalog not evaluated')
    f, events = pd.read_parquet(src/'all_pair_scores.parquet'), pd.read_parquet(src/'events.parquet')
    spec = json.loads((root/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
    all_values = configurations(f, spec)
    indices = pd.read_parquet(c.ROOT/'plans/catalog190_subsets.parquet')
    indices = indices[indices.split.eq(split)]
    rows = []
    for draw, subset in indices.groupby('draw', sort=True):
        keep = events.source_uid.isin(subset.source_uid).to_numpy()
        if keep.sum() != 190:
            raise RuntimeError('Wrong subcatalog size')
        pairmask = keep[f.idx_i] & keep[f.idx_j]
        g = f.loc[pairmask].copy()
        remap = np.full(len(events), -1); remap[keep] = np.arange(190)
        g['idx_i'], g['idx_j'], g['event_count'] = remap[g.idx_i], remap[g.idx_j], 190
        if g.is_true_pair.sum() != 70 or len(g) != 17955:
            raise RuntimeError('Incomplete doublets in 190-event control')
        for method in ('waveform-new', 'time-only', 'sky-only', PRIMARY):
            v = all_values[method][pairmask]
            rows.append(dict(run=run, arm=arm, seed=seed, split=split, draw=int(draw), method=method,
                n_events=190, n_pairs=17955, n_true_pairs=70, **s.metrics(g, v), **budget_metrics(v, g.is_true_pair)))
    csv(src/'catalog190_metrics.csv', pd.DataFrame(rows))
    c.seal(src/'CATALOG190_COMPLETE.json', [src/'catalog190_metrics.csv'], subcatalogs=500,
           independent_experiments=False, identical_draws_across_runs_and_arms=True)


def snr_diagnostic(run, arm, split):
    p = c.prepare_inputs(run, arm, split)
    e = pd.read_parquet(p/'events.parquet')
    f = pd.read_parquet(c.deployment(run, arm)/f'sky_pair_scores/{split}/pairs.parquet')
    f['event_count'] = len(e)
    f['true_pair_family'] = np.where(f.is_true_pair, e.family.to_numpy()[f.idx_i], 'unlensed')
    bins = np.digitize(e.target_network_snr.to_numpy(), [10., 12., 20.])
    a, b = np.minimum(bins[f.idx_i], bins[f.idx_j]), np.maximum(bins[f.idx_i], bins[f.idx_j])
    cue = ((a == 0) & (b == 1)) | ((a == 2) & (b == 3))
    return dict(run=run, arm=arm, split=split, **s.metrics(f, cue.astype(float)),
                diagnostic_only=True, not_encoder_input=True)


def summarize():
    metrics, subs, intervals, distributions, weights, snr = [], [], [], [], [], []
    delta_rows = []
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            p = c.deployment(run, arm)
            for split in ('validation', 'test'):
                snr.append(snr_diagnostic(run, arm, split))
                for seed in c.U.SEEDS:
                    src = p/f'evaluation/{split}/seed_{seed}'
                    if not c.check_complete(src/'CATALOG190_COMPLETE.json'):
                        raise RuntimeError('Missing size-control evaluation')
                    metrics.append(pd.read_csv(src/'metrics.csv'))
                    subs.append(pd.read_csv(src/'catalog190_metrics.csv'))
                    distributions.append(pd.read_csv(src/'distributions.csv'))
                    if split == 'test':
                        intervals.append(pd.read_csv(src/'cluster_intervals.csv'))
            for seed in c.U.SEEDS:
                spec = json.loads((p/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
                for mode in ('POSITIVE', 'NONNEGATIVE'):
                    chosen = spec['final_fusion'][mode]
                    weights.append(dict(run=run, arm=arm, seed=seed, mode=mode,
                        **dict(zip(('waveform', 'time', 'sky'), chosen['weights'])),
                        **{branch+'_'+k: spec[branch][k] for branch in ('FRT', 'OMC', 'joint') for k in ('gamma', 'beta')},
                        equal_metric_plateau=json.dumps(chosen['equal_metric_plateau'])))
        for seed in c.U.SEEDS:
            for unit in ('source', 'noise'):
                paths = [c.deployment(run, arm)/f'evaluation/test/seed_{seed}/bootstrap_{unit}.npz' for arm in c.U.ARMS]
                a, b = [dict(np.load(path)) for path in paths]
                for method in a:
                    v = b[method]-a[method]
                    valid = np.isfinite(v).all(1)
                    lo, hi = np.quantile(v[valid], [.025, .975], axis=0)
                    for metric, low, high in zip(('average_precision', 'F50', 'F90', 'R1', 'R10'), lo, hi):
                        delta_rows.append(dict(run=run, seed=seed, unit=unit, method=method, metric=metric,
                            contrast='B_CUE minus A_NEUTRAL', low=float(low), high=float(high), replicates=int(valid.sum()),
                            paired_source_catalog_and_bootstrap=True))
    allm = pd.concat(metrics, ignore_index=True); allsub = pd.concat(subs, ignore_index=True)
    csv(c.OUT/'tables/retrieval_metrics_per_seed.csv', allm)
    csv(c.OUT/'tables/catalog190_all_draws.csv', allsub)
    columns = ['macro_r_at_1', 'macro_r_at_5', 'macro_r_at_10', 'macro_r_at_50', 'average_precision',
               'false_at_recall_0p5', 'false_at_recall_0p9', 'roc_auc']
    for name, data in [('450', allm), ('190', allsub.groupby(['run', 'arm', 'split', 'seed', 'method'])[columns].mean().reset_index())]:
        grouped = data.groupby(['run', 'arm', 'split', 'method'])[columns].agg(['mean', 'std']).reset_index()
        grouped.columns = ['_'.join(filter(None, x)) if isinstance(x, tuple) else x for x in grouped.columns]
        csv(c.OUT/f'tables/retrieval_summary_{name}.csv', grouped)
    csv(c.OUT/'tables/cluster_intervals.csv', pd.concat(intervals, ignore_index=True))
    csv(c.OUT/'tables/paired_arm_difference_intervals.csv', pd.DataFrame(delta_rows))
    csv(c.OUT/'tables/channel_distributions.csv', pd.concat(distributions, ignore_index=True))
    csv(c.OUT/'tables/selected_weights.csv', pd.DataFrame(weights))
    csv(c.OUT/'tables/SNR_only_diagnostic.csv', pd.DataFrame(snr))
    c.seal(c.OUT/'contracts/INJECTION_EVALUATION_COMPLETE.json', [p for p in (c.OUT/'tables').glob('*.csv') if not p.name.startswith('RESOURCE_')],
           all_three_runs_both_arms=True, model_seeds=3, native_events=450, controlled_events=190,
           controls_not_independent=True, real_used=False, no_test_retuning=True)


def real_ranking(run, arm):
    c.gate('real')
    if not (c.OUT/'contracts/INJECTION_EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('All injection results must complete before real ranking')
    root = c.deployment(run, arm); dest = root/'real_ranking'
    if c.check_complete(dest/'COMPLETE.json'):
        return
    dest.mkdir(parents=True, exist_ok=True)
    stack = []
    for seed in c.U.SEEDS:
        spec = json.loads((root/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
        f, events = s.frame(run, arm, seed, 'real')
        f = s.apply(root, f, spec)
        f['pair_key'] = ['--'.join(sorted((a, b))) for a, b in zip(f.event_i, f.event_j)]
        f.to_parquet(dest/f'seed_{seed}_all_scores.parquet', index=False)
        for method in ('waveform-new', PRIMARY, 'three-channel-zero-allowed'):
            values = configurations(f, spec)[method]
            d = f[['pair_key', 'event_i', 'event_j', 'waveform_score', 'time_score', 'sky_raw_log_bf', 'sky_BC', 'joint_BC']].copy()
            w = np.array([1., 0., 0.]) if method == 'waveform-new' else np.asarray(spec['final_fusion']['POSITIVE' if method == PRIMARY else 'NONNEGATIVE']['weights'])
            parts = s.inherited().channels(f, f.waveform_score)*w
            assert np.allclose(parts.sum(1), values, rtol=1e-12, atol=1e-12)
            for k, channel in enumerate(('wf', 'time', 'sky')):
                d[channel+'_contribution'] = parts[:, k]
            d['score'] = values; d['seed'] = seed; d['method'] = method
            d = d.sort_values(['score', 'pair_key'], ascending=[False, True])
            d['rank'] = np.arange(1, len(d)+1)
            stack.append(d)
    stacked = pd.concat(stack, ignore_index=True)
    stacked.to_parquet(dest/'all_seed_rankings.parquet', index=False)
    result = []
    for method, data in stacked.groupby('method'):
        keys = ['score', 'waveform_score', 'time_score', 'sky_raw_log_bf', 'sky_BC', 'joint_BC', 'wf_contribution', 'time_contribution', 'sky_contribution']
        table = data.groupby('pair_key')[keys].mean().add_suffix('_mean')
        table['rank_mean'] = data.groupby('pair_key')['rank'].mean()
        table['rank_max'] = data.groupby('pair_key')['rank'].max()
        table['rank_SD'] = data.groupby('pair_key')['rank'].std()
        table = table.join(data.groupby('pair_key')[['event_i', 'event_j']].first()).reset_index()
        table = table.sort_values(['rank_mean', 'rank_max', 'score_mean', 'pair_key'], ascending=[True, True, False, True])
        table['consensus_rank'] = np.arange(1, len(table)+1)
        table['run'], table['arm'], table['method'] = run, arm, method
        result.append(table)
    pd.concat(result, ignore_index=True).to_parquet(dest/'consensus_all_pairs.parquet', index=False)
    c.seal(dest/'COMPLETE.json', [dest/'consensus_all_pairs.parquet', dest/'all_seed_rankings.parquet'],
           no_PE_or_official_input=True, seed_count=3, events=len(events), pair_count=len(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['catalog', 'subcatalogs', 'summarize', 'real'], required=True)
    p.add_argument('--run', default='O3'); p.add_argument('--arm', default='A_NEUTRAL')
    p.add_argument('--seed', type=int, default=2026091721)
    p.add_argument('--split', default='test', choices=['validation', 'test'])
    a = p.parse_args(); c.initialize(a.root, a.out)
    if a.stage == 'catalog': score_catalog(a.run, a.arm, a.seed, a.split)
    elif a.stage == 'subcatalogs': subcatalogs(a.run, a.arm, a.seed, a.split)
    elif a.stage == 'real': real_ranking(a.run, a.arm)
    else: summarize()


if __name__ == '__main__':
    main()
