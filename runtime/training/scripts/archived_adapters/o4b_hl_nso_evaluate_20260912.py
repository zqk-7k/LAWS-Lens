#!/usr/bin/env python3
"""One-way held-out/real evaluation after the immutable O4b score freeze."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_inference_20260912 as inference
import o4b_hl_nso_score_20260912 as score

BUDGETS = (10, 20, 50, 100, 200, 500)


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def configurations(f, spec):
    w0 = np.asarray(spec['baseline_fusion']['POSITIVE']['weights'])
    wn = np.asarray(spec['final_fusion']['POSITIVE']['weights'])
    wt = np.array([wn[0], wn[1], 0.]); wt /= wt.sum() if wt.sum() else 1
    return {
        'waveform-short': f.Z_wf_short.to_numpy(), 'waveform-OMC': f.Z_wf_OMC.to_numpy(),
        'waveform-new': f.waveform_score.to_numpy(), 'time-only': f.time_score.to_numpy(),
        'sky-only': f.sky_raw_log_bf.to_numpy(),
        'time-sky-fixed-ratio': score.channels(f, f.waveform_score)@np.array([0., wn[1], wn[2]]),
        'waveform-time-fixed-ratio': score.channels(f, f.waveform_score)@wt,
        'SHORT-HL-CANDIDATE': score.channels(f, f.Z_wf_short)@w0,
        'OMC-HL-CANDIDATE': score.channels(f, f.Z_wf_OMC)@w0,
        'NEW-SCORE-ONLY-POSITIVE-CANDIDATE': f.final_score_POSITIVE.to_numpy(),
        'NEW-SCORE-ONLY-NONNEGATIVE-CANDIDATE': f.final_score_NONNEGATIVE.to_numpy(),
    }


def query_ranks(f, events, values):
    n = len(events); ii, jj = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    mat = np.full((n, n), -np.inf); mat[ii, jj] = values; mat[jj, ii] = values
    positive = np.flatnonzero(f.is_true_pair.to_numpy(bool))
    query = np.r_[ii[positive], jj[positive]]; partner = np.r_[jj[positive], ii[positive]]
    true = mat[query, partner]
    ranks = 1+(mat[query] > true[:, None]).sum(1)
    worst = (mat[query] >= true[:, None]).sum(1)
    if len(np.unique(query)) != len(query):
        raise RuntimeError('More than one true companion per directed query')
    return pd.DataFrame({'event_uid': events.event_uid.to_numpy()[query],
                         'companion_uid': events.event_uid.to_numpy()[partner],
                         'source_id': events.global_source_id.to_numpy()[query],
                         'family': events.family.to_numpy()[query],
                         'rank': ranks, 'rank_pessimistic': worst, 'true_score': true})


def query_ci(ranks, seed):
    grouped = ranks.groupby(['family', 'source_id']).agg(
        R1=('rank', lambda x: (x <= 1).mean()), R5=('rank', lambda x: (x <= 5).mean()),
        R10=('rank', lambda x: (x <= 10).mean()), R50=('rank', lambda x: (x <= 50).mean()))
    rng = np.random.default_rng(seed+17001)
    draws = []
    for family in sorted(ranks.family.unique()):
        x = grouped.loc[family].to_numpy()
        take = rng.integers(len(x), size=(10000, len(x)))
        draws.append(x[take].mean(1))
    lower, upper = np.quantile(np.mean(draws, axis=0), [.025, .975], axis=0)
    return {**{f'{k}_ci_low': float(v) for k, v in zip(('R1', 'R5', 'R10', 'R50'), lower)},
            **{f'{k}_ci_high': float(v) for k, v in zip(('R1', 'R5', 'R10', 'R50'), upper)},
            'bootstrap_replicates': 10000, 'bootstrap_unit': 'lens system, both directed queries, family stratified'}


def weighted_pair_evaluator(values, truth):
    order = np.argsort(-values, kind='stable'); y = truth[order]
    ends = np.r_[np.flatnonzero(np.diff(values[order]) != 0), len(order)-1]
    def calculate(weights):
        w = weights[order]
        tp, fp = np.cumsum(w*y), np.cumsum(w*(~y))
        if tp[-1] == 0:
            raise RuntimeError('Bootstrap lost all positive systems')
        t, f = tp[ends], fp[ends]
        ap = np.sum(np.diff(np.r_[0., t])*t/(t+f).clip(1e-30))/t[-1]
        out = [float(ap)]
        for fraction in (.5, .9):
            k = min(np.searchsorted(tp, fraction*tp[-1]), len(tp)-1)
            out.append(float(fp[k]))
        return np.array(out)
    return calculate


def pair_bootstrap(f, events, values, seed, root):
    groups, names = pd.factorize(events.global_source_id.astype(str), sort=True)
    ii, jj = groups[f.idx_i], groups[f.idx_j]
    truth = f.is_true_pair.to_numpy(bool)
    family = {g: str(events.iloc[np.flatnonzero(groups == g)[0]].family) for g in range(len(names))}
    strata = [np.array([k for k, v in family.items() if v == fam]) for fam in sorted(set(family.values()))]
    selected = ['SHORT-HL-CANDIDATE', 'OMC-HL-CANDIDATE', 'NEW-SCORE-ONLY-POSITIVE-CANDIDATE',
                'NEW-SCORE-ONLY-NONNEGATIVE-CANDIDATE']
    funcs = {key: weighted_pair_evaluator(values[key], truth) for key in selected}
    for key, fn in funcs.items():
        check = fn(np.ones(len(truth)))[0]
        if abs(check-average_precision_score(truth, values[key])) > 1e-10:
            raise RuntimeError('Weighted pair AP unit check failed')
    rng = np.random.default_rng(seed+18001); draws = {key: [] for key in selected}
    for rep in range(1000):
        counts = np.zeros(len(names), int)
        for subset in strata:
            counts += np.bincount(rng.choice(subset, len(subset), replace=True), minlength=len(names))
        weights = counts[ii]*counts[jj]; weights[truth] = counts[ii[truth]]
        for key, fn in funcs.items():
            draws[key].append(fn(weights))
    rows = []
    for name, data in draws.items():
        data = np.array(data)
        for comparison, array in [('absolute', data), ('delta_vs_SHORT', data-np.array(draws['SHORT-HL-CANDIDATE']))]:
            lo, hi = np.quantile(array, [.025, .975], axis=0)
            rows.append({'seed': seed, 'method': name, 'comparison': comparison,
                         **{k+'_ci_low': v for k, v in zip(('AUPRC', 'F50', 'F90'), lo)},
                         **{k+'_ci_high': v for k, v in zip(('AUPRC', 'F50', 'F90'), hi)},
                         'bootstrap_replicates': 1000, 'unit': 'source-system multiplicity; shared noise not separately resampled'})
    np.savez_compressed(root/f'results/injection/seed_{seed}/pair_bootstrap_draws.npz', **draws)
    return rows


def injection(root):
    inference.guard(root, 'test')
    marker = root/'contracts/INJECTION_EVALUATION_COMPLETE.json'
    if marker.exists():
        return
    cf, _ = score.api(root)
    metrics, pair_ci, distributions, gates, counts = [], [], [], [], []
    for seed in score.SEEDS:
        out = root/f'results/injection/seed_{seed}'; out.mkdir(parents=True, exist_ok=True)
        spec = json.loads((root/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
        f, events = score.frame(root, seed, 'test')
        f = score.apply(root, f, spec)
        f.to_parquet(out/'all_pair_scores.parquet', index=False)
        events.to_parquet(out/'events.parquet', index=False)
        values = configurations(f, spec)
        counts.append({'seed': seed, 'n_events': len(events), 'n_true_pairs': int(f.is_true_pair.sum()),
                       'n_unordered_pairs': len(f), 'n_queries': int(f.is_true_pair.sum())*2,
                       'positive_pair_rate': float(f.is_true_pair.mean()), 'n_candidates_per_query': len(events)-1,
                       'random_R10': 10/(len(events)-1)})
        for method, val in values.items():
            m = cf.fast_metrics(f, val); ranks = query_ranks(f, events, val)
            ranks['seed'], ranks['method'] = seed, method
            ranks.to_parquet(out/f'{method}_query_ranks.parquet', index=False)
            r50 = ranks.groupby('family')['rank'].apply(lambda x: (x <= 50).mean()).mean()
            row = {'seed': seed, 'method': method, **m, 'macro_r_at_50': float(r50),
                   'median_rank': float(ranks['rank'].median()), **query_ci(ranks, seed)}
            truth = f.is_true_pair.to_numpy(bool); order = np.argsort(-val, kind='stable')
            tp = np.cumsum(truth[order])
            for fraction in (.5, .9):
                ix = min(np.searchsorted(tp, fraction*tp[-1]), len(tp)-1); threshold = val[order[ix]]
                row[f'threshold_inclusive_FP_at_recall_{fraction}'] = int(((val >= threshold) & ~truth).sum())
            for b in BUDGETS:
                row[f'top_{b}_recall'] = float(truth[order[:b]].sum()/truth.sum())
            metrics.append(row)
        for method in ('NEW-SCORE-ONLY-POSITIVE-CANDIDATE', 'NEW-SCORE-ONLY-NONNEGATIVE-CANDIDATE'):
            gates.append({'seed': seed, 'method': method,
                          'versus_SHORT_guard': cf.guard(cf.fast_metrics(f, values[method]), cf.fast_metrics(f, values['SHORT-HL-CANDIDATE'])),
                          'versus_OMC_guard': cf.guard(cf.fast_metrics(f, values[method]), cf.fast_metrics(f, values['OMC-HL-CANDIDATE'])),
                          'failures_do_not_trigger_test_retuning': True})
        pair_ci += pair_bootstrap(f, events, values, seed, root)
        for label, use in [('companion', f.is_true_pair), ('noncompanion', ~f.is_true_pair)]:
            for col in ('Z_wf_short', 'Z_wf_FRT', 'Z_wf_OMC', 'waveform_score', 'time_score', 'sky_raw_log_bf', 'joint_BC'):
                x = f.loc[use, col].to_numpy(float)
                distributions.append({'seed': seed, 'population': label, 'channel': col, 'n': len(x),
                                      'mean': x.mean(), 'std': x.std(ddof=1),
                                      **{f'q{q:g}': float(np.quantile(x, q)) for q in (0, .01, .1, .25, .5, .75, .9, .99, .999, 1)}})
        csv(root/'tables/retrieval_metrics_per_seed.csv', pd.DataFrame(metrics))
        csv(root/'tables/pair_bootstrap_intervals.csv', pd.DataFrame(pair_ci))
    df = pd.DataFrame(metrics)
    columns = ['macro_r_at_1', 'macro_r_at_5', 'macro_r_at_10', 'macro_r_at_50', 'average_precision',
               'false_at_recall_0p5', 'false_at_recall_0p9', 'roc_auc']
    summary = df.groupby('method')[columns].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in summary.columns]
    csv(root/'tables/retrieval_metrics_summary.csv', summary)
    csv(root/'tables/test_catalog_denominators.csv', pd.DataFrame(counts))
    csv(root/'tables/locked_test_guards.csv', pd.DataFrame(gates))
    csv(root/'tables/channel_distributions.csv', pd.DataFrame(distributions))
    s.write(marker, {'utc': s.now(), 'all3_seeds': True, 'test_opened_after_freeze': True,
                     'no_test_or_real_retuning': True, 'all_guards_pass': bool(pd.DataFrame(gates).versus_OMC_guard.all()),
                     'uncertainty_limit': 'Query CI conditions on fixed model/catalog; seed SD includes model initialization and new noise realizations; pair bootstrap does not remove shared-noise dependence'})


def real(root):
    inference.guard(root, 'real')
    if not (root/'contracts/INJECTION_EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Complete held-out evaluation before real ranking')
    marker = root/'contracts/REAL_RANKINGS_COMPLETE.json'
    if marker.exists():
        return
    results, weight_rows = {}, []
    for seed in score.SEEDS:
        folder = root/f'results/real/seed_{seed}'; folder.mkdir(parents=True, exist_ok=True)
        spec = json.loads((root/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
        f, events = score.frame(root, seed, 'real'); f = score.apply(root, f, spec)
        f['pair_key'] = ['--'.join(sorted((a, b))) for a, b in zip(f.event_i, f.event_j)]
        f.to_parquet(folder/'all_pair_scores.parquet', index=False)
        for method, value in configurations(f, spec).items():
            d = f.copy(); d['final_score'] = value
            w0 = np.asarray(spec['baseline_fusion']['POSITIVE']['weights'])
            wn = np.asarray(spec['final_fusion']['POSITIVE']['weights'])
            if method.startswith('waveform-') and not method.startswith('waveform-time'):
                parts = np.column_stack([value, np.zeros(len(d)), np.zeros(len(d))])
            elif method == 'time-only':
                parts = np.column_stack([np.zeros(len(d)), value, np.zeros(len(d))])
            elif method == 'sky-only':
                parts = np.column_stack([np.zeros(len(d)), np.zeros(len(d)), value])
            elif method == 'time-sky-fixed-ratio':
                parts = score.channels(f, f.waveform_score)*np.array([0., wn[1], wn[2]])
            elif method == 'waveform-time-fixed-ratio':
                w = np.array([wn[0], wn[1], 0.]); w /= w.sum()
                parts = score.channels(f, f.waveform_score)*w
            elif method in ('SHORT-HL-CANDIDATE', 'OMC-HL-CANDIDATE'):
                z = f.Z_wf_short if method.startswith('SHORT') else f.Z_wf_OMC
                parts = score.channels(f, z)*w0
            else:
                constraint = 'NONNEGATIVE' if 'NONNEGATIVE' in method else 'POSITIVE'
                parts = score.channels(f, f.waveform_score)*np.asarray(spec['final_fusion'][constraint]['weights'])
            if not np.allclose(parts.sum(1), value, atol=1e-12, rtol=1e-12):
                raise RuntimeError('Per-method contribution reconstruction failed')
            for k, name in enumerate(('wf', 'time', 'sky')):
                d['method_'+name+'_contribution'] = parts[:, k]
            d = d.sort_values(['final_score', 'pair_key'], ascending=[False, True]).reset_index(drop=True)
            d['rank'], d['seed'], d['method'] = np.arange(1, len(d)+1), seed, method
            d.to_parquet(folder/f'{method}_ranked.parquet', index=False)
            csv(folder/f'{method}_top50.csv', d.head(50))
            results.setdefault(method, []).append(d)
        for constraint, chosen in spec['final_fusion'].items():
            weight_rows.append({'seed': seed, 'constraint': constraint,
                                **dict(zip(('waveform', 'time', 'sky'), chosen['weights'])),
                                **{k+'_'+x: spec[k][x] for k in ('FRT', 'OMC', 'joint') for x in ('gamma', 'beta')}})
    out = root/'results/real/consensus'; out.mkdir(parents=True, exist_ok=True)
    for method, frames in results.items():
        stacked = pd.concat(frames)
        columns = ['final_score', 'waveform_score', 'Z_wf_OMC', 'Z_wf_short', 'time_score', 'sky_raw_log_bf', 'sky_BC', 'joint_BC',
                   'method_wf_contribution', 'method_time_contribution', 'method_sky_contribution',
                   'waveform_contribution_POSITIVE', 'time_contribution_POSITIVE', 'sky_contribution_POSITIVE',
                   'waveform_contribution_NONNEGATIVE', 'time_contribution_NONNEGATIVE', 'sky_contribution_NONNEGATIVE']
        d = stacked.groupby('pair_key')[columns].mean().add_suffix('_mean')
        d['rank_mean'] = stacked.groupby('pair_key')['rank'].mean()
        d['rank_max'] = stacked.groupby('pair_key')['rank'].max()
        d['rank_std'] = stacked.groupby('pair_key')['rank'].std()
        d = d.join(stacked.groupby('pair_key')[['event_i', 'event_j']].first()).reset_index()
        d = d.sort_values(['rank_mean', 'rank_max', 'final_score_mean', 'pair_key'], ascending=[True, True, False, True]).reset_index(drop=True)
        d['consensus_rank'] = np.arange(1, len(d)+1); d['method'] = method
        d['PE_audit_status'] = 'PENDING_VERIFIED_PUBLIC_POSTERIOR'
        d['official_lensing_pair_status'] = 'NO_VERIFIED_O4B_PAIR_FPP_OR_HANABI_RELEASE_IN_INPUTS'
        d.to_parquet(out/f'{method}_all_pairs.parquet', index=False)
        csv(out/f'{method}_all_pairs.csv', d)
        for b in (10, 20, 50):
            csv(out/f'{method}_top{b}.csv', d.head(b))
    csv(root/'tables/selected_waveform_and_fusion_weights.csv', pd.DataFrame(weight_rows))
    s.write(marker, {'utc': s.now(), 'events': 86, 'pairs': 3655, 'three_seed_consensus': True,
                     'PE_used_in_ranking': False, 'official_event_FAR_not_pair_FPP': True,
                     'PE_audit_pending': True, 'unordered_max_equals_mean': 'All three channels use globally frozen symmetric pair features; no row-wise standardization'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('injection', 'real'), required=True); a = p.parse_args()
    globals()[a.stage](a.root)
