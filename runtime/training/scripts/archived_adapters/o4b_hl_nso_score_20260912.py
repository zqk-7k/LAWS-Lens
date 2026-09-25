#!/usr/bin/env python3
"""Frozen O4b NEW-SCORE-ONLY calibration; never selects with real PE.

The inherited scoring topology is retained. New O4b calibration distributions,
correction strengths and fusion weights are learned only from development and
validation. Every intermediate score is exported to make the ancestry explicit.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import o4b_hl_nso_data_training_20260912 as s

P = Path('/root/autodl-tmp/gw-catalog')
SEEDS = (2026091221, 2026091222, 2026091223)
GAMMAS = (0., .125, .25, .5, 1., 2., 4.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)


def api(root):
    s.environment(root)
    sys.path.insert(0, str(P/'scripts/experiments'))
    import mcwf_temporal_response_evaluate_20260908 as ev
    from scripts.real_search import physical_common as phys
    return ev.cf, phys


def register(root):
    path = root/'contracts/SCORE_SELECTION_PROTOCOL.json'
    if path.exists():
        check = json.loads((root/'contracts/SCORE_SELECTION_PROTOCOL_FREEZE.json').read_text())
        if s.sha(path) != check['sha256']:
            raise RuntimeError('Score protocol changed after registration')
        return
    recipe = {
        'utc': s.now(), 'code': 'O4B-HL-NSO-01', 'before_locked_test': True,
        'exact_inherited_method_family': 'JOINT-ETA-CHI-BC-ADD-CANDIDATE with alpha=1; no outer old/new score mixture',
        'ancestry': ['short embedding cosine and old Mc/q regression -> validation-global standardization and KDE LR',
                     'RNC predictive-Mc finite companion tail -> FRT',
                     'ordered-Mc tail plus bounded prior-overlap increment -> OMC',
                     '16s+2s conditional joint Mc/eta/chi tail plus bounded BC increment -> new waveform',
                     'one linear waveform/time/sky score'],
        'not_NODUP_DIRECT': True, 'no_fourth_channel': True,
        'calibration': 'Main validation fits short global scaling/KDE and FRT reference. Auxiliary development fits ordered overlap. Joint BC isotonic fit uses source/noise-disjoint half of auxiliary development; other half audits only.',
        'development_reuse': 'Auxiliary development also selected checkpoint/temperature; calibration audit is not independent population coverage.',
        'gamma_grid': GAMMAS, 'beta_grid': BETAS,
        'FRT_gamma_grid': [0.]+[2.**k for k in range(-6, 2)],
        'zero_correction': 'Retained as declared null control; no forced nonzero penalty if validation rejects it.',
        'tail': 'p=(1+number reference disagreements>=x)/(n+1); T=min(log(p/.05),0); ties conservative',
        'increment': 'Class-balanced monotone isotonic log odds, clip[-4,4]; no positive OOD or companion-tail<.05; endpoint occupancy>.25 neutralizes new terms',
        'short_mass_weights': [0., .25, .5, 1., 2.], 'short_q_weights': [0., .25, .5, 1.],
        'short_KDE_bandwidths': [.75, 1., 1.5],
        'short_parameter_uncertainty': 'Original short head predicts standardized means, not posterior sigma; differences are standardized by frozen validation global scales.',
        'initial_reference_weights': 'O4b SHORT-HL-CANDIDATE selected on new validation positive simplex; normalized O4a inherited slot weights only break exact ties, never numerical O4b calibration reuse.',
        'reference_tie_vectors': [[1., .25, .5], [.5, .25, .5], [1., .25, .25]],
        'fusion': '0.05 simplex231points; positive and nonnegative arms both exported; closest normalized stage baseline then fixed lexicographic ties',
        'primary_arm': 'NEW-SCORE-ONLY-POSITIVE-CANDIDATE, matching inherited PATH method constraint',
        'supplementary_arm': 'NEW-SCORE-ONLY-NONNEGATIVE-CANDIDATE; zero weights allowed, never hidden',
        'priority': 'F50,F90,-AUPRC,-R10,-R1; correction ties smallest squared coefficient norm then gamma,beta',
        'correction_guards': 'waveform and fixed-fusion R10>=preceding-.02; AUPRC>=preceding-.005; F50,F90<=1.1preceding',
        'selection_unit': 'per seed, same algorithm all3; no seed chosen by real outcomes',
        'ranking_ties': 'Inherited optimistic directed retrieval ties, pessimistic retrieval also reported. Stable pair ordering retained; threshold-inclusive FP additionally reported.',
        'time': 'O4b training-only840source lookup, frozen once for all seeds',
        'sky': 'Frozen validation-temperature conditional BAYESTAR probability mass at Nside512',
        'real_PE_official_used': False, 'test_used': False,
        'bootstrap': '10000 system-stratified directed-query resamples; paired source-multiplicity pair bootstrap1000; dependent noise caveat explicit',
        'difference_from_historical_training': 'Fresh O4b source/noise/checkpoints; ordered-Mc training4096parents rather than historical12288; no claim of bitwise O3/O4a replication',
    }
    s.write(path, recipe)
    cf, _ = api(root)
    s.write(root/'contracts/SCORE_SELECTION_PROTOCOL_FREEZE.json', {
        'sha256': s.sha(path), 'code_sha256': s.sha(Path(__file__)),
        'historical_metric_code': str(Path(cf.__file__)), 'metric_sha256': s.sha(Path(cf.__file__))})


def frame(root, seed, split):
    if split in ('test', 'real'):
        from o4b_hl_nso_inference_20260912 import guard
        guard(root, split)
    folder = root/f'predictions_o4b/seed_{seed}/{split}'
    if not (folder/'COMPLETE.json').exists():
        raise RuntimeError('Inference incomplete')
    f = pd.read_parquet(folder/'waveform_features.parquet')
    e = pd.read_parquet(folder/'events.parquet')
    f['event_count'] = len(e)
    if split != 'real':
        fam = e.family.to_numpy(str)
        f['true_pair_family'] = np.where(f.is_true_pair, fam[f.idx_i], 'unlensed')
    if split != 'development':
        sky_seed = SEEDS[0] if split == 'real' else seed
        skyroot = root/f'sky_pair_scores/seed_{sky_seed}/{split}'
        if not (skyroot/'COMPLETE.json').exists():
            raise RuntimeError('Sky pair scoring incomplete')
        sk = pd.read_parquet(skyroot/'pairs.parquet')
        keys = ['event_i', 'event_j']
        if f.duplicated(keys).any() or sk.duplicated(keys).any():
            raise RuntimeError('Duplicate pair UID')
        sk = sk.set_index(keys).reindex(f.set_index(keys).index)
        if sk.sky_log_bf_nside512.isna().any():
            raise RuntimeError('Waveform/sky pair UID mismatch')
        f['sky_raw_log_bf'] = sk.sky_log_bf_nside512.to_numpy()
        f['sky_BC'] = sk.sky_BC_nside512.to_numpy()
        _, phys = api(root)
        troot = root/'calibration/time'
        freeze = json.loads((troot/'FREEZE.json').read_text())
        for item in freeze.get('files', []):
            if s.sha(item['path']) != item['sha256']:
                raise RuntimeError('Time calibration changed')
        spec = json.loads((troot/'time_delay_likelihood_ratio.json').read_text())
        gps = e['gps_time' if split == 'real' else 'gps_obs'].to_numpy(float)
        dt = abs(gps[f.idx_i]-gps[f.idx_j])/86400.
        f['time_delay_days'] = dt
        f['time_score'] = phys.apply_time_likelihood_ratio(dt, spec)
    return f, e


def tail(x, reference):
    r = np.sort(np.asarray(reference, float))
    if len(r) == 0 or not np.isfinite(r).all():
        raise RuntimeError('Invalid companion reference')
    return (1.+len(r)-np.searchsorted(r, x, side='left'))/(len(r)+1.)


def priority(m):
    return (m['false_at_recall_0p5'], m['false_at_recall_0p9'], -m['average_precision'],
            -m['macro_r_at_10'], -m['macro_r_at_1'])


def channels(f, z):
    return np.column_stack([z, f.time_score, f.sky_raw_log_bf])


def short_features(f):
    return np.column_stack([f.waveform_embedding_cosine, -f.waveform_abs_delta_logmc_std,
                            -f.waveform_abs_delta_logitq_std])


def fit_short(root, seed, f):
    cf, phys = api(root)
    x = short_features(f)
    center = np.median(x, axis=0); scale = np.maximum(x.std(axis=0), 1e-6)
    x = (x-center)/scale
    rows = []
    for mc in (0., .25, .5, 1., 2.):
        for q in (0., .25, .5, 1.):
            raw = x@np.array([1., mc, q]); m = cf.fast_metrics(f, raw)
            rows.append({'mc': mc, 'q': q, **m})
    selected = min(rows, key=lambda m: (-m['macro_r_at_10'], -m['average_precision'], m['false_at_recall_0p5'],
                                        -m['macro_r_at_1'], m['mc']+m['q'], m['mc'], m['q']))
    raw = x@np.array([1., selected['mc'], selected['q']])
    y = f.is_true_pair.to_numpy(bool)
    choices, kde_rows = [], []
    for bw in (.75, 1., 1.5):
        cal = phys.fit_score_likelihood_ratio(raw[y], raw[~y], bandwidth_scale=bw)
        cal = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cal.items()}
        z = phys.apply_score_likelihood_ratio(raw, cal); m = cf.fast_metrics(f, z)
        choices.append(((-m['average_precision'], -m['macro_r_at_10'], m['false_at_recall_0p5'], abs(bw-1)), cal))
        kde_rows.append({'bandwidth_scale': bw, **m})
    spec = {'center': center.tolist(), 'scale': scale.tolist(), 'mc_weight': selected['mc'], 'q_weight': selected['q'],
            'calibration': min(choices, key=lambda a: a[0])[1]}
    out = root/f'calibration/score/seed_{seed}'; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'short_parameter_grid.csv', index=False)
    pd.DataFrame(kde_rows).to_csv(out/'short_KDE_grid.csv', index=False)
    return spec


def apply_short(root, f, spec):
    _, phys = api(root)
    x = (short_features(f)-np.asarray(spec['center']))/np.asarray(spec['scale'])
    raw = x@np.array([1., spec['mc_weight'], spec['q_weight']])
    return phys.apply_score_likelihood_ratio(raw, spec['calibration'])


def fit_iso(value, truth, bc):
    truth = np.asarray(truth, bool)
    count = int(truth.sum())
    if count < 30:
        raise RuntimeError('Insufficient source-level calibration positives')
    weight = np.where(truth, .5/count, .5/(~truth).sum())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(value, truth, sample_weight=weight)
    p = iso.y_thresholds_.clip(1/(count+2), 1-1/(count+2))
    return {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(p)-np.log1p(-p)).tolist(),
            'minimum': float(np.min(value)), 'maximum': float(np.max(value)),
            'reference': np.sort(-np.log(np.asarray(bc)[truth].clip(1e-15, 1))).tolist(), 'positive_sources': count}


def correction(value, bc, endpoint_ood, spec):
    pp = tail(-np.log(np.asarray(bc).clip(1e-15, 1)), spec['reference'])
    penalty = np.minimum(np.log(pp/.05), 0.)
    ood = np.asarray(endpoint_ood, bool) | (value < spec['minimum']) | (value > spec['maximum'])
    increment = np.interp(value, spec['knots'], spec['loglr']).clip(-4, 4)
    increment = np.where(ood | (pp < .05), np.minimum(increment, 0.), increment)
    return np.where(endpoint_ood, 0., penalty), np.where(endpoint_ood, 0., increment), ood


def joint_fit_partition(root, seed, f, events):
    groups = events.global_source_id.to_numpy(str)
    banks = sorted(events.noise_bank_index.unique(), key=lambda n: hashlib.sha256(f'202609850:gwtc5:{n}'.encode()).hexdigest())
    fold = events.noise_bank_index.map({n: k % 2 for k, n in enumerate(banks)}).to_numpy(int)
    cross = {g for g in np.unique(groups) if len(np.unique(fold[groups == g])) > 1}
    fold[np.isin(groups, list(cross))] = -1
    events = events.assign(calibration_fold=fold)
    events.to_parquet(root/f'calibration/score/seed_{seed}/joint_calibration_folds.parquet', index=False)
    masks = [(fold[f.idx_i] == k) & (fold[f.idx_j] == k) for k in (0, 1)]
    for mask in masks:
        if f.loc[mask, 'is_true_pair'].sum() < 30:
            raise RuntimeError('Joint calibration fit/audit has fewer than30 source systems')
    return masks


def fusion_search(root, seed, f, z, reference, name, baseline=None):
    cf, _ = api(root)
    reference = np.asarray(reference, float); reference /= reference.sum()
    grid = np.array([(i/20, j/20, (20-i-j)/20) for i in range(21) for j in range(21-i)])
    grid = np.unique(np.round(np.vstack([grid, reference]), 12), axis=0)
    rows = []
    for w in grid:
        m = cf.fast_metrics(f, channels(f, z)@w)
        rows.append({'weights': w.tolist(), 'positive': bool((w >= .05-1e-12).all()),
                     'guard_pass': baseline is None or cf.guard(m, baseline), **m})
    selected = {}
    for constraint in ('POSITIVE', 'NONNEGATIVE'):
        options = [r for r in rows if r['guard_pass'] and (constraint == 'NONNEGATIVE' or r['positive'])]
        if not options:
            raise RuntimeError(f'No eligible {name} {constraint} fusion')
        win = min(options, key=lambda r: (*priority(r), float(np.square(np.asarray(r['weights'])-reference).sum()), *r['weights']))
        plateau = [r['weights'] for r in options if priority(r) == priority(win)]
        selected[constraint] = {'weights': win['weights'], 'validation_metrics': {k: v for k, v in win.items() if k != 'weights'},
                                'equal_metric_plateau': plateau}
    pd.DataFrame(rows).to_json(root/f'calibration/score/seed_{seed}/{name}_fusion_grid.json', orient='records', indent=2)
    return selected


def choose_correction(root, seed, f, baseline_z, weights, penalty, increment, name, gamma=GAMMAS, beta=BETAS):
    cf, _ = api(root)
    bm, wm = cf.fast_metrics(f, channels(f, baseline_z)@weights), cf.fast_metrics(f, baseline_z)
    rows = []
    for g in gamma:
        for b in beta:
            z = baseline_z+g*penalty+b*increment
            m, w = cf.fast_metrics(f, channels(f, z)@weights), cf.fast_metrics(f, z)
            rows.append({'gamma': g, 'beta': b, 'pass': cf.guard(m, bm) and cf.guard(w, wm), **m,
                         **{'waveform_'+k: v for k, v in w.items()}})
    selected = min((r for r in rows if r['pass']), key=lambda r: (*priority(r), r['gamma']**2+r['beta']**2, r['gamma'], r['beta']))
    pd.DataFrame(rows).to_csv(root/f'calibration/score/seed_{seed}/{name}_coefficient_grid.csv', index=False)
    return {k: selected[k] for k in ('gamma', 'beta')}


def calibrate(root, seed):
    register(root)
    out = root/f'calibration/score/seed_{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out/'FREEZE.json').exists():
        if s.sha(out/'SELECTED.json') != json.loads((out/'FREEZE.json').read_text())['sha256']:
            raise RuntimeError('Selected waveform/fusion configuration changed')
        return
    f, events = frame(root, seed, 'validation')
    d, de = frame(root, seed, 'development')
    spec = {'seed': seed, 'short': fit_short(root, seed, f)}
    old = apply_short(root, f, spec['short'])
    tie_vectors = ([1., .25, .5], [.5, .25, .5], [1., .25, .25])
    spec['baseline_fusion'] = fusion_search(root, seed, f, old, tie_vectors[SEEDS.index(seed)], 'short')
    w0 = np.asarray(spec['baseline_fusion']['POSITIVE']['weights'])
    ref = np.sort(-np.log(f.loc[f.is_true_pair, 'rnc_BC'].to_numpy().clip(1e-12)))
    t = np.minimum(np.log(tail(-np.log(f.rnc_BC.to_numpy().clip(1e-12)), ref)/.05), 0)
    spec['FRT'] = {'reference': ref.tolist(), **choose_correction(root, seed, f, old, w0, t, np.zeros(len(f)), 'FRT',
                                                               gamma=(0.,)+tuple(2.**k for k in range(-6, 2)), beta=(0.,))}
    frt = old+spec['FRT']['gamma']*t
    omc = fit_iso(d.ordered_log_prior_overlap.to_numpy(), d.is_true_pair.to_numpy(), d.ordered_BC.to_numpy())
    t, inc, _ = correction(f.ordered_log_prior_overlap.to_numpy(), f.ordered_BC.to_numpy(), f.ordered_ood.to_numpy(), omc)
    omc.update(choose_correction(root, seed, f, frt, w0, t, inc, 'OMC'))
    spec['OMC'] = omc
    omcz = frt+omc['gamma']*t+omc['beta']*inc
    fit, audit = joint_fit_partition(root, seed, d, de)
    joint = fit_iso(d.loc[fit, 'joint_log_BC'].to_numpy(), d.loc[fit, 'is_true_pair'].to_numpy(), d.loc[fit, 'joint_BC'].to_numpy())
    t, inc, _ = correction(f.joint_log_BC.to_numpy(), f.joint_BC.to_numpy(), f.joint_ood.to_numpy(), joint)
    joint.update(choose_correction(root, seed, f, omcz, w0, t, inc, 'JOINT'))
    spec['joint'] = joint
    jointz = omcz+joint['gamma']*t+joint['beta']*inc
    cf, _ = api(root)
    baseline = cf.fast_metrics(f, channels(f, omcz)@w0)
    spec['final_fusion'] = fusion_search(root, seed, f, jointz, w0, 'new_score_only', baseline)
    spec['joint_audit'] = {'fit_sources': int(d.loc[fit, 'is_true_pair'].sum()), 'audit_sources': int(d.loc[audit, 'is_true_pair'].sum()),
                           'audit_true_tail_below05': float((tail(-d.loc[audit & d.is_true_pair, 'joint_log_BC'].to_numpy(), joint['reference']) < .05).mean()),
                           'checkpoint_development_reuse': True, 'test_used': False}
    s.write(out/'SELECTED.json', spec)
    scored = apply(root, f, spec)
    scored.to_parquet(out/'validation_scored.parquet', index=False)
    s.write(out/'FREEZE.json', {'utc': s.now(), 'sha256': s.sha(out/'SELECTED.json'),
                               'validation_pairs_sha256': s.sha(out/'validation_scored.parquet'),
                               'test_used': False, 'real_PE_used': False})
    print(json.dumps({'score_calibration_complete': seed, 'weights': spec['final_fusion'],
                      'coefficients': {k: {x: spec[k][x] for x in ('gamma', 'beta')} for k in ('FRT', 'OMC', 'joint')}}), flush=True)


def apply(root, f, spec):
    result = f.copy()
    result['Z_wf_short'] = apply_short(root, f, spec['short'])
    p = tail(-np.log(f.rnc_BC.to_numpy().clip(1e-12)), spec['FRT']['reference'])
    result['FRT_tail_probability'] = p
    result['FRT_penalty'] = np.minimum(np.log(p/.05), 0)
    result['Z_wf_FRT'] = result.Z_wf_short+spec['FRT']['gamma']*result.FRT_penalty
    t, inc, ood = correction(f.ordered_log_prior_overlap.to_numpy(), f.ordered_BC.to_numpy(), f.ordered_ood.to_numpy(), spec['OMC'])
    result['OMC_penalty'], result['OMC_increment'], result['OMC_calibration_ood'] = t, inc, ood
    result['Z_wf_OMC'] = result.Z_wf_FRT+spec['OMC']['gamma']*t+spec['OMC']['beta']*inc
    t, inc, ood = correction(f.joint_log_BC.to_numpy(), f.joint_BC.to_numpy(), f.joint_ood.to_numpy(), spec['joint'])
    result['joint_penalty'], result['joint_increment'], result['joint_calibration_ood'] = t, inc, ood
    result['waveform_score'] = result.Z_wf_OMC+spec['joint']['gamma']*t+spec['joint']['beta']*inc
    for constraint, selection in spec['final_fusion'].items():
        w = np.asarray(selection['weights'])
        result[f'final_score_{constraint}'] = channels(f, result.waveform_score)@w
        for name, value in zip(('waveform', 'time', 'sky'), channels(f, result.waveform_score).T*w[:, None]):
            result[f'{name}_contribution_{constraint}'] = value
    return result


def freeze(root):
    register(root)
    target = root/'contracts/FINAL_SCORE_CONFIG_FREEZE.json'
    if target.exists():
        from o4b_hl_nso_inference_20260912 import guard
        guard(root, 'test'); return
    files = [root/'contracts/SCORE_SELECTION_PROTOCOL.json', root/'calibration/time/time_delay_likelihood_ratio.json',
             root/'contracts/DATA_SPLIT_CONTRACT_v2.json', root/'contracts/CONVERGENCE_AUDIT_PROTOCOL.json',
             root/'scripts/o4b_hl_nso_score_20260912.py',
             root/'scripts/o4b_hl_nso_inference_20260912.py',
             root/'scripts/o4b_hl_nso_multiscale_training_20260912.py',
             root/'scripts/o4b_hl_nso_rankncontrast_v2_20260912.py',
             root/'scripts/o4b_hl_nso_real_inputs_20260912.py',
             root/'scripts/o4b_hl_nso_bayestar_20260912.py',
             root/'scripts/o4b_hl_nso_sky_scores_20260912.py']
    for seed in SEEDS:
        out = root/f'calibration/score/seed_{seed}'
        if not (out/'FREEZE.json').exists():
            raise RuntimeError('All3 seed calibrations required before opening test')
        files += [out/'SELECTED.json', root/f'event_maps/seed_{seed}/validation/TEMPERATURE_SELECTED.json']
        inputs = json.loads((root/f'predictions_o4b/seed_{seed}/validation/COMPLETE.json').read_text())['inputs']
        for item in inputs:
            if s.sha(item['path']) != item['sha256']:
                raise RuntimeError('Model hash changed after calibration')
            files.append(Path(item['path']))
    s.write(target, {'utc': s.now(), 'state': 'FROZEN_BEFORE_TEST_AND_REAL', 'alpha': 1.,
                     'files': [{'path': str(p), 'sha256': s.sha(p)} for p in sorted(set(files))],
                     'real_PE_or_official_selection': False, 'test_opened_before_this_freeze': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('register', 'calibrate', 'freeze'), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0]); a = p.parse_args()
    if a.stage == 'calibrate': calibrate(a.root, a.seed)
    else: globals()[a.stage](a.root)
