"""Validation-only NSO calibration with explicit ties and group weights."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import uab_completion as c


def pair_metrics(score, truth, weight=None):
    score, truth = np.asarray(score, float), np.asarray(truth, bool)
    if not np.isfinite(score).all():
        raise RuntimeError('Nonfinite score')
    w = np.ones(len(score)) if weight is None else np.asarray(weight, float)
    order = np.argsort(-score, kind='stable')
    x, y, w = score[order], truth[order], w[order]
    end = np.r_[np.flatnonzero(x[:-1] != x[1:]), len(x)-1]
    tp, fp = np.cumsum(w*y)[end], np.cumsum(w*~y)[end]
    if tp[-1] <= 0 or fp[-1] <= 0:
        raise RuntimeError('Both classes required')
    inc = np.diff(np.r_[0., tp])
    ap = float(np.sum(inc*tp/np.maximum(tp+fp, 1e-300))/tp[-1])
    tpr, fpr = tp/tp[-1], fp/fp[-1]
    roc = float(np.trapz(np.r_[0., tpr], np.r_[0., fpr]))
    result = dict(average_precision=ap, roc_auc=roc,
        false_at_recall_0p5=float(fp[np.searchsorted(tp, .5*tp[-1])]),
        false_at_recall_0p9=float(fp[np.searchsorted(tp, .9*tp[-1])]),
        positive_pairs=float(tp[-1]), negative_pairs=float(fp[-1]),
        tied_pair_fraction=float(1-len(end)/len(score)))
    return result


def query_metrics(frame, values, keep_rows=False):
    n = int(frame.event_count.iloc[0])
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    matrix = np.full((n, n), -np.inf)
    matrix[i, j] = matrix[j, i] = values
    true = frame.is_true_pair.to_numpy(bool)
    a, b = i[true], j[true]
    query, companion = np.r_[a, b], np.r_[b, a]
    target = matrix[query, companion]
    greater = (matrix[query] > target[:, None]).sum(1)
    equal = (matrix[query] == target[:, None]).sum(1)
    if (equal < 1).any():
        raise RuntimeError('Companion absent from candidate table')
    family = np.r_[frame.true_pair_family.to_numpy()[true], frame.true_pair_family.to_numpy()[true]]
    out = pd.DataFrame(dict(query_idx=query, companion_idx=companion, family=family,
        greater=greater, tied=equal, expected_rank=greater+(equal+1)/2))
    result = {}
    for k in (1, 5, 10, 50):
        out[f'r_at_{k}'] = np.clip((k-greater)/equal, 0, 1)
        result[f'macro_r_at_{k}'] = float(out.groupby('family')[f'r_at_{k}'].mean().mean())
    result['median_rank'] = float(out.expected_rank.median())
    result['n_queries'] = len(out)
    result['query_tie_fraction'] = float((out.tied > 1).mean())
    return (result, out) if keep_rows else result


def metrics(frame, values):
    return {**pair_metrics(values, frame.is_true_pair), **query_metrics(frame, np.asarray(values))}


def guard(candidate, baseline):
    return (candidate['macro_r_at_10'] >= baseline['macro_r_at_10']-.02-1e-12 and
            candidate['average_precision'] >= baseline['average_precision']-.005-1e-12 and
            candidate['false_at_recall_0p5'] <= 1.1*baseline['false_at_recall_0p5']+1e-12 and
            candidate['false_at_recall_0p9'] <= 1.1*baseline['false_at_recall_0p9']+1e-12)


def api(root):
    from scripts.real_search import physical_common as phys
    return SimpleNamespace(fast_metrics=metrics, guard=guard), phys


def inherited():
    import o4b_hl_nso_score_20260912 as s
    s.api = api
    s.SEEDS = c.U.SEEDS
    return s


def frame(run, arm, seed, split):
    c.gate(split)
    p = c.deployment(run, arm)
    src = p/f'predictions_o4b/seed_{seed}/{split}'
    if not (src/'COMPLETE.json').exists():
        raise RuntimeError('Waveform inference incomplete')
    receipt = json.loads((src/'COMPLETE.json').read_text())
    expected = {Path(item['path']).name: item['sha256'] for item in receipt['outputs']}
    for name in ('waveform_features.parquet', 'events.parquet'):
        if c.U.sha(src/name) != expected[name]:
            raise RuntimeError('Frozen waveform pair or event table changed')
    f = pd.read_parquet(src/'waveform_features.parquet')
    events = pd.read_parquet(src/'events.parquet')
    f['event_count'] = len(events)
    if split != 'real':
        fam = events.family.to_numpy(str)
        f['true_pair_family'] = np.where(f.is_true_pair, fam[f.idx_i], 'unlensed')
        exact = (events.source_uid.to_numpy()[f.idx_i] == events.source_uid.to_numpy()[f.idx_j])
        if not np.array_equal(exact, f.is_true_pair):
            raise RuntimeError('Pair labels disagree with waveform parents')
    if split != 'development':
        skyroot = c.OUT/'real_sky'/run if split == 'real' else p/'sky_pair_scores'/split
        if not c.check_complete(skyroot/'COMPLETE.json'):
            raise RuntimeError('Sky scoring incomplete')
        sk = pd.read_parquet(skyroot/'pairs.parquet')
        keys = ['event_i', 'event_j']
        if sk.duplicated(keys).any() or f.duplicated(keys).any():
            raise RuntimeError('Duplicate pair')
        aligned = sk.set_index(keys).reindex(f.set_index(keys).index)
        if aligned.sky_raw_log_bf.isna().any():
            raise RuntimeError('Sky/waveform UID alignment failed')
        f['sky_raw_log_bf'] = aligned.sky_raw_log_bf.to_numpy()
        f['sky_BC'] = aligned.sky_BC.to_numpy()
        spec = json.loads((c.ROOT/'shared_time'/run/'lookup.json').read_text())
        gps = events['gps_time' if split == 'real' else 'gps_obs'].to_numpy(float)
        f['time_delay_days'] = abs(gps[f.idx_i]-gps[f.idx_j])/86400
        _, phys = api(p)
        f['time_score'] = phys.apply_time_likelihood_ratio(f.time_delay_days.to_numpy(), spec)
    return f, events


def priority(m):
    return (m['false_at_recall_0p5'], m['false_at_recall_0p9'], -m['average_precision'],
            -m['macro_r_at_10'], -m['macro_r_at_1'])


def fusion(root, seed, f, z, name, baseline=None):
    s = inherited()
    grid = np.array([(i/20, j/20, (20-i-j)/20) for i in range(21) for j in range(21-i)])
    anchor = np.full(3, 1/3)
    rows = []
    values = s.channels(f, z)
    for w in grid:
        m = metrics(f, values@w)
        rows.append(dict(weights=w.tolist(), positive=bool((w >= .05-1e-12).all()),
                         guard_pass=baseline is None or guard(m, baseline), **m))
    selected = {}
    for mode in ('POSITIVE', 'NONNEGATIVE'):
        options = [r for r in rows if r['guard_pass'] and (mode == 'NONNEGATIVE' or r['positive'])]
        if not options:
            raise RuntimeError('No eligible fusion candidate: '+name+' '+mode)
        win = min(options, key=lambda v: (*priority(v), float(((np.asarray(v['weights'])-anchor)**2).sum()), *v['weights']))
        plateau = [r['weights'] for r in options if priority(r) == priority(win)]
        selected[mode] = dict(weights=win['weights'], validation_metrics={k: v for k, v in win.items() if k != 'weights'},
                              equal_metric_plateau=plateau, tie_anchor=anchor.tolist())
    c.write(root/f'calibration/score/seed_{seed}/{name}_fusion_grid.json', rows)
    return selected


def source_weights(f, events, mask=None):
    parent = events.source_uid.astype(str)
    group = events['lens_system_group_id'].astype(str) if 'lens_system_group_id' in events else events.global_source_id.astype(str)
    active = np.ones(len(events), bool)
    if mask is not None:
        active[:] = False
        active[np.r_[f.loc[mask, 'idx_i'].to_numpy(int), f.loc[mask, 'idx_j'].to_numpy(int)]] = True
    e = pd.DataFrame(dict(parent=parent[active], group=group[active])).drop_duplicates('parent')
    count = e.groupby('group').parent.size()
    w = 1/group.map(count).fillna(1).to_numpy(float)
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    true = f.is_true_pair.to_numpy(bool)
    weight = np.where(true, w[i], w[i]*w[j])
    return weight, group.to_numpy()[i]


def weighted_iso(f, events, feature, bc, mask=None):
    mask = np.ones(len(f), bool) if mask is None else np.asarray(mask, bool)
    weight, group = source_weights(f, events, mask)
    truth = f.is_true_pair.to_numpy(bool)[mask]
    value, overlap = f[feature].to_numpy()[mask], f[bc].to_numpy()[mask]
    weight, group = weight[mask], group[mask]
    n_groups = len(np.unique(group[truth]))
    group_total = pd.Series(weight[truth]).groupby(group[truth]).sum().to_numpy()
    if not np.allclose(group_total, 1., rtol=0, atol=1e-12):
        raise RuntimeError('Each represented lens group must have total positive weight one')
    if n_groups < 30:
        raise RuntimeError(f'Calibration has only {n_groups} independent lens groups')
    weights = np.where(truth, .5*weight/weight[truth].sum(), .5*weight/weight[~truth].sum())
    fit = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(value, truth, sample_weight=weights)
    p = fit.y_thresholds_.clip(1/(n_groups+2), 1-1/(n_groups+2))
    ref = -np.log(overlap[truth].clip(1e-15, 1))
    ix = np.argsort(ref)
    refw = weight[truth][ix]
    refw = refw/refw.sum()
    return dict(knots=fit.X_thresholds_.tolist(), loglr=(np.log(p)-np.log1p(-p)).tolist(),
        minimum=float(value.min()), maximum=float(value.max()), reference=ref[ix].tolist(),
        reference_weights=refw.tolist(), independent_groups=n_groups, positive_waveform_parents=int(truth.sum()),
        one_lens_group_total_weight=True, maximum_group_weight_error=float(abs(group_total-1).max()))


def weighted_tail(disagreement, spec):
    ref = np.asarray(spec['reference'])
    w = np.asarray(spec.get('reference_weights', np.full(len(ref), 1/len(ref))))
    n = spec.get('independent_groups', len(ref))
    rank = np.searchsorted(ref, disagreement, side='left')
    left = np.r_[0., np.cumsum(w)]
    return (1+n*np.maximum(1-left[rank], 0))/(n+1)


def correction(value, bc, endpoint_ood, spec):
    pp = weighted_tail(-np.log(np.asarray(bc).clip(1e-15, 1)), spec)
    penalty = np.minimum(np.log(pp/.05), 0.)
    endpoint_ood = np.asarray(endpoint_ood, bool)
    ood = endpoint_ood | (value < spec['minimum']) | (value > spec['maximum'])
    increment = np.interp(value, spec['knots'], spec['loglr']).clip(-4, 4)
    increment = np.where(ood | (pp < .05), np.minimum(increment, 0.), increment)
    return np.where(endpoint_ood, 0., penalty), np.where(endpoint_ood, 0., increment), ood


def calibrate(run, arm, seed):
    root = c.deployment(run, arm)
    out = root/f'calibration/score/seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if c.check_complete(out/'FREEZE.json'):
        return
    s = inherited()
    f, events = frame(run, arm, seed, 'validation')
    d, de = frame(run, arm, seed, 'development')
    spec = dict(seed=seed, run=run, arm=arm, short=s.fit_short(root, seed, f),
                source_group_weighting=True, posterior_or_official_selection=False,
                correction_guards=dict(R10_absolute=.02, AP_absolute=.005, false_burden_ratio=1.1))
    old = s.apply_short(root, f, spec['short'])
    spec['baseline_fusion'] = fusion(root, seed, f, old, 'short')
    w0 = np.asarray(spec['baseline_fusion']['POSITIVE']['weights'])
    ref = np.sort(-np.log(f.loc[f.is_true_pair, 'rnc_BC'].to_numpy().clip(1e-12)))
    pen = np.minimum(np.log(s.tail(-np.log(f.rnc_BC.to_numpy().clip(1e-12)), ref)/.05), 0)
    spec['FRT'] = dict(reference=ref.tolist(), **s.choose_correction(root, seed, f, old, w0, pen,
        np.zeros(len(f)), 'FRT', gamma=(0.,)+tuple(2.**k for k in range(-6, 2)), beta=(0.,)))
    frt = old+spec['FRT']['gamma']*pen
    omc = weighted_iso(d, de, 'ordered_log_prior_overlap', 'ordered_BC')
    pen, inc, _ = correction(f.ordered_log_prior_overlap.to_numpy(), f.ordered_BC.to_numpy(), f.ordered_ood.to_numpy(), omc)
    omc.update(s.choose_correction(root, seed, f, frt, w0, pen, inc, 'OMC'))
    spec['OMC'] = omc
    omcz = frt+omc['gamma']*pen+omc['beta']*inc
    fit, audit = s.joint_fit_partition(root, seed, d, de)
    joint = weighted_iso(d, de, 'joint_log_BC', 'joint_BC', fit)
    pen, inc, _ = correction(f.joint_log_BC.to_numpy(), f.joint_BC.to_numpy(), f.joint_ood.to_numpy(), joint)
    joint.update(s.choose_correction(root, seed, f, omcz, w0, pen, inc, 'JOINT'))
    spec['joint'] = joint
    z = omcz+joint['gamma']*pen+joint['beta']*inc
    spec['final_fusion'] = fusion(root, seed, f, z, 'new_score_only', metrics(f, s.channels(f, omcz)@w0))
    w, groups = source_weights(d, de)
    truth = d.is_true_pair.to_numpy(bool)
    spec['joint_audit'] = dict(fit_waveform_parents=int((fit & truth).sum()),
        audit_waveform_parents=int((audit & truth).sum()), fit_lens_groups=len(set(groups[fit & truth])),
        audit_lens_groups=len(set(groups[audit & truth])),
        shared_lens_environments=len(set(groups[fit & truth]) & set(groups[audit & truth])),
        distinct_waveform_parents_and_noise_folds=True, not_independent_population_coverage=True,
        audit_true_tail_below05=float(np.average(weighted_tail(-d.loc[audit & truth, 'joint_log_BC'].to_numpy(), joint)<.05,
                                                weights=w[audit & truth])))
    c.write(out/'SELECTED.json', spec)
    result = apply(root, f, spec)
    result.to_parquet(out/'validation_scored.parquet', index=False)
    c.seal(out/'FREEZE.json', [out/'SELECTED.json', out/'validation_scored.parquet'],
           test_used=False, real_used=False, inherited_topology_no_outer_mixture=True)
    print(json.dumps(dict(stage='CALIBRATION_COMPLETE', run=run, arm=arm, seed=seed,
                          weights=spec['final_fusion']['POSITIVE']['weights'])), flush=True)


def apply(root, f, spec):
    s = inherited()
    # Replace only the calibration-tail evaluator; all channel algebra is archived NSO.
    original = s.correction
    s.correction = correction
    try:
        return s.apply(root, f, spec)
    finally:
        s.correction = original


def freeze():
    marker = c.ROOT/'contracts/FINAL_SCORE_FREEZE.json'
    if c.check_complete(marker):
        return
    files = [c.ROOT/'contracts/SCORING_SPEC.json', c.OUT/'contracts/COMPLETION_ADAPTER_CONTRACT.json']
    for run in c.U.RUNS:
        files.append(c.ROOT/'shared_time'/run/'lookup.json')
        for arm in c.U.ARMS:
            p = c.deployment(run, arm)
            temp = c.OUT/'maps'/run/arm/'validation'
            if not c.check_complete(temp/'TEMPERATURE_FREEZE.json'):
                raise RuntimeError('Sky temperature not frozen')
            files.append(temp/'TEMPERATURE_SELECTED.json')
            for seed in c.U.SEEDS:
                score = p/f'calibration/score/seed_{seed}'
                if not c.check_complete(score/'FREEZE.json'):
                    raise RuntimeError('Not all 18 calibrations frozen')
                files.append(score/'SELECTED.json')
                files += [score/'FREEZE.json', score/'validation_scored.parquet']
    model_inputs = json.loads((c.OUT/'contracts/INPUT_AUDIT.json').read_text())['files']
    for row in model_inputs:
        if c.U.sha(row['path']) != row['sha256']:
            raise RuntimeError('Trained model or contract changed')
        files.append(Path(row['path']))
    files += list((c.OUT/'scripts').glob('*.py'))
    files += [c.OUT/'contracts/REAL_INPUT_MANIFEST.json', c.OUT/'contracts/REAL_INPUT_MANIFEST_FREEZE.json',
              c.OUT/'contracts/EVALUATION_COMPLETION_CONTRACT.json']
    c.seal(marker, sorted(set(files)), state='ALL_RUNS_BOTH_ARMS_FROZEN_BEFORE_TEST',
           continuation_directory=str(c.OUT), test_previously_opened=False, real_results_used=False,
           model_training_complete=True, seed_count=3, deployments=6)


def tests():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(2026091801)
    records = []
    for ties in (False, True):
        x = rng.normal(size=200)
        if ties:
            x = np.round(x)
        y = np.arange(200) % 7 == 0
        actual = pair_metrics(x, y)
        assert abs(actual['average_precision']-average_precision_score(y, x)) < 1e-12
        assert abs(actual['roc_auc']-roc_auc_score(y, x)) < 1e-12
        for recall, key in ((.5, 'false_at_recall_0p5'), (.9, 'false_at_recall_0p9')):
            cutoff = np.sort(x[y])[::-1][int(np.ceil(recall*y.sum()))-1]
            assert actual[key] == ((x >= cutoff) & ~y).sum()
        records.append(dict(test='AP_ROC_F50_F90', ties=ties, pass_test=True))
    i, j = np.triu_indices(5, 1)
    f = pd.DataFrame(dict(idx_i=i, idx_j=j, event_count=5, is_true_pair=(i == 0) & (j == 1),
                          true_pair_family=np.where((i == 0) & (j == 1), 'SIS', 'unlensed')))
    v = metrics(f, np.ones(len(f)))
    assert v['macro_r_at_1'] == .25 and v['macro_r_at_10'] == 1
    assert v['false_at_recall_0p5'] == 9
    spec = dict(reference=[0., 1., 2.], reference_weights=[1/3]*3, independent_groups=3)
    x = np.array([-.1, 0., .1, 1., 1.5, 2., 3.])
    s = inherited()
    assert np.allclose(weighted_tail(x, spec), s.tail(x, spec['reference']), atol=1e-15, rtol=0)
    records += [dict(test='uniform_retrieval_ties', pass_test=True), dict(test='weighted_tail_reduces_to_original', pass_test=True)]
    from uab_evaluate import weighted_evaluator
    callback = weighted_evaluator(f, pd.DataFrame(index=range(5)), np.ones(len(f)))
    b = callback(np.ones(len(f)), np.ones(5))
    assert np.allclose(b, [v['average_precision'], 9., 9., .25, 1.], rtol=0, atol=1e-12)
    events = pd.DataFrame(dict(source_uid=['a', 'a', 'b', 'b', 'c', 'c'],
                               lens_system_group_id=['G','G','G','G','H','H']))
    i, j = np.triu_indices(6, 1)
    pairs = pd.DataFrame(dict(idx_i=i, idx_j=j, is_true_pair=np.array(['a','a','b','b','c','c'])[i] == np.array(['a','a','b','b','c','c'])[j]))
    mask = (i >= 2) & (j >= 2)
    w, groups = source_weights(pairs, events, mask)
    totals = pd.Series(w[mask & pairs.is_true_pair]).groupby(groups[mask & pairs.is_true_pair]).sum()
    assert np.allclose(totals, 1., atol=1e-12, rtol=0)
    records += [dict(test='weighted_metric_uniform_replay', pass_test=True),
                dict(test='source_group_total_one_after_subset', pass_test=True)]
    c.write(c.OUT/'contracts/SCORING_UNIT_TESTS.json', dict(state='PASS', tests=records))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['tests', 'calibrate', 'freeze'], required=True)
    p.add_argument('--run', choices=['O3', 'O4a', 'O4b'], default='O3')
    p.add_argument('--arm', choices=['A_NEUTRAL', 'B_CUE'], default='A_NEUTRAL')
    p.add_argument('--seed', type=int, default=2026091721)
    a = p.parse_args()
    c.initialize(a.root, a.out)
    if a.stage == 'tests':
        tests()
    elif a.stage == 'calibrate':
        calibrate(a.run, a.arm, a.seed)
    else:
        freeze()


if __name__ == '__main__':
    main()
