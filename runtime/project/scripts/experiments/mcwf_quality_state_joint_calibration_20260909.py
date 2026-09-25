#!/usr/bin/env python3
"""Source-weighted joint-BC calibration conditional on profile quality state."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_selected_predictive_joint_control_20260909 as ctrl
engine, base, n, r, co, h = ctrl.engine, ctrl.base, ctrl.n, ctrl.r, ctrl.co, ctrl.h
ROOT = DATA = PREDICTIVE = SELECTION = None
METHODS = ('NODUP-DIRECT-REPLAY', 'JOINTSTATE-GLOBAL', 'JOINTSTATE-REJECT-ONLY')
FIELDS = ('actual_waveform_joint_BC', 'new_prediction_used_in_score', 'profile_state',
          'profile_state_true_tail', 'profile_state_loglr', 'profile_state_OOD')


def freeze():
    base.freeze()
    n.write_json(ROOT/'contracts/QUALITY_STATE_ADDENDUM.json', {
        'UTC': n.utc(), 'id': 'MCWF-NODUP-QUALITY-STATE-JOINT-35',
        'overrides_base': 'No new classifier grid.Three fixed quality states0/1/2validprofileendpoints;one jointBC LR and true-source tail perstate.',
        'predictive': 'Same R29validated,R33selectedBOUNDED error model.OriginalNNconditional eta/chi givenMc.',
        'state_is_label_free': True, 'state_probability': 'Add logP(state|L)-logP(state|N) fitted with original source-pair weights before class balancing withinstate.',
        'per_state_minimum_true_sources': 20,
        'fit': '512 independent development sources in each run;fold0 only,exclude identicalnoisebank nulls. No oldreal or test fit.',
        'methods': list(METHODS),
        'REJECT_ONLY': 'Only BOTH valid endpoints and conditional true-tail<.05:take minimum of unchangedNODUP and updatedwaveform;allotherpairs exact NODUP.',
        'not_probability_claim': 'Conditional empirical ranktail is NOT catalog FPP or PE. Shared noise limits exchangeability;report tune tail incidence and sources.',
        'fixed': ['allencodercheckpoints','cosinecalibration','gamma','beta','outerweights','time','sky','scope','oldresults'],
        'cap': [-4., 4.], 'tail_threshold': .05, 'positive_OOD': 0,
        'no_total_mix': True, 'no_old_Mc_q_score': True,
        'adaptive_development': True, 'new_blind_confirmation': False,
        'no_real_PE_or_official_model_input': True,
        'status': n.STATUS, 'script_sha256': n.sha(Path(__file__))})
    shutil.copy2(__file__, ROOT/'scripts/quality_state_joint_calibration.py')


def states(frame):
    return frame.profile_active_i.to_numpy(np.int8)+frame.profile_active_j.to_numpy(np.int8)


def terms(value, state, endpoint, specs):
    tail, loglr, outside = np.empty(len(value)), np.empty(len(value)), np.empty(len(value), bool)
    for k in range(3):
        mask = state == k
        if not mask.any():
            continue
        spec = specs[str(k)]
        tail[mask] = n.ev.tail.tail_probability(-value[mask], spec['reference'])
        raw = np.interp(value[mask], spec['knots'], spec['loglr'])+spec['state_log_ratio']
        loglr[mask] = raw.clip(-4., 4.)
        outside[mask] = (value[mask] < spec['minimum']) | (value[mask] > spec['maximum']) | endpoint[mask]
    if not np.isin(state, [0, 1, 2]).all():
        raise RuntimeError('Invalid quality state')
    loglr[outside | (tail < .05)] = np.minimum(loglr[outside | (tail < .05)], 0.)
    penalty = np.minimum(np.log(tail/.05), 0.)
    penalty[endpoint], loglr[endpoint] = 0., 0.
    return tail, loglr, penalty, outside


def infer(frame, config):
    original = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    reference = co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    state = states(frame)
    frame['profile_state'] = state
    if config['method'] == METHODS[0]:
        frame['actual_waveform_joint_BC'] = frame.parent_joint_BC
        frame['new_prediction_used_in_score'] = False
        frame['profile_state_true_tail'] = np.nan
        frame['profile_state_loglr'] = np.nan
        frame['profile_state_OOD'] = False
        return reference
    value = frame.joint_logbc.to_numpy(float)
    tail, inc, penalty, outside = terms(value, state, frame.joint_ood.to_numpy(bool), config['state_specs'])
    cosine, cos_ood, cos_clip = n.apply_model(frame, original['cosine_calibration'])
    candidate = (cosine+original['gamma']*penalty+original['beta']*inc,
                 outside | cos_ood, cos_clip | (abs(inc) >= 4.))
    use = np.ones(len(value), bool)
    if config['method'] == METHODS[2]:
        use = (state == 2) & (tail < .05) & (candidate[0] < reference[0])
        result = tuple(np.where(use, a, b) for a, b in zip(candidate, reference))
        if np.any(result[0] > reference[0]) or not np.array_equal(result[0][~use], reference[0][~use]):
            raise RuntimeError('Conditional exact fallback invariant failed')
    else:
        result = candidate
    frame['actual_waveform_joint_BC'] = np.where(use, frame.joint_BC, frame.parent_joint_BC)
    frame['new_prediction_used_in_score'] = use
    frame['profile_state_true_tail'] = tail
    frame['profile_state_loglr'] = inc
    frame['profile_state_OOD'] = outside
    return result


def calibrate():
    configurations, audits, units, tails = [], [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            pop = base.population(dep, seed)
            panels = base.panels(dep, seed)
            x, y, weights, _, i, j = panels[0]
            state = pop['active'][i].astype(int)+pop['active'][j].astype(int)
            value = x[:, 1]+x[:, 2]
            specs = {}
            for k in range(3):
                mask = state == k
                xx, yy, ww = value[mask], y[mask], weights[mask].copy()
                if min(yy.sum(), (~yy).sum()) < 20:
                    raise RuntimeError('HOLD_INSUFFICIENT_STATE_SOURCE_SUPPORT:'+str((dep,seed,k)))
                offset = float(np.log(weights[mask & y].sum()/weights[y].sum())-
                    np.log(weights[mask & ~y].sum()/weights[~y].sum()))
                ww[yy] *= .5/ww[yy].sum()
                ww[~yy] *= .5/ww[~yy].sum()
                iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(xx, yy, sample_weight=ww)
                probability = iso.y_thresholds_.clip(1/(yy.sum()+2), 1-1/(yy.sum()+2))
                specs[str(k)] = {'knots': iso.X_thresholds_.tolist(),
                    'loglr': (np.log(probability)-np.log1p(-probability)).tolist(),
                    'minimum': float(xx.min()), 'maximum': float(xx.max()),
                    'reference': np.sort(-xx[yy]).tolist(), 'state_log_ratio': offset,
                    'true_fit_sources': int(yy.sum()), 'null_rows': int((~yy).sum())}
            vx, vy, _, _, vi, vj = panels[1]
            vs = pop['active'][vi].astype(int)+pop['active'][vj].astype(int)
            vt = terms(vx[:, 1]+vx[:, 2], vs, np.zeros(len(vy), bool), specs)[0]
            for k in range(3):
                mask = vy & (vs == k)
                tails.append({'deployment': dep, 'seed': seed, 'state': k,
                    'true_fit_sources': specs[str(k)]['true_fit_sources'],
                    'true_tune_sources': int(mask.sum()), 'tune_tail_below_0p05': float((vt[mask] < .05).mean()),
                    'state_log_ratio': specs[str(k)]['state_log_ratio']})
            n.write_json(ROOT/f'calibration/{dep}_{seed}_STATE.json', specs)
            frame = ctrl.panel(dep, seed, 'validation')
            old = h.isolated.ORIGINALS[dep, seed]
            cbase = {**old, 'method': METHODS[0]}
            reference = infer(frame, cbase)[0]
            archived = pd.read_parquet(r.PRIOR/f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(reference, archived.waveform_score.to_numpy()):
                raise RuntimeError('NODUP replay changed')
            configurations.append(cbase)
            fm = n.cf.fast_metrics(frame, n.cf.channels(frame, reference)@np.asarray(old['weights']))
            wm = n.cf.fast_metrics(frame, reference)
            for method in METHODS[1:]:
                config = {**old, 'method': method, 'state_specs': specs}
                z = infer(frame, config)[0]
                metric = n.cf.fast_metrics(frame, n.cf.channels(frame, z)@np.asarray(old['weights']))
                waveform = n.cf.fast_metrics(frame, z)
                config['tune_guard'] = n.cf.guard(metric, fm) and n.cf.guard(waveform, wm)
                config['tune_metrics'] = metric
                poisoned = frame.copy()
                poisoned['pair_key'], poisoned['pe_mc_bhattacharyya_coefficient'], poisoned['official_po_fpp'] = 'unused', -999., 1.
                if not np.array_equal(z, infer(poisoned, {**config, 'alpha': -999., 'old_Mc_weight': 999.})[0]):
                    raise RuntimeError('Forbidden field changed waveform')
                reversed_frame = frame.iloc[::-1].copy()
                if not np.array_equal(z, infer(reversed_frame, config)[0][::-1]):
                    raise RuntimeError('Row ordering changed waveform')
                configurations.append(config)
                audits.append({'deployment': dep, 'seed': seed, 'method': method,
                    'tune_guard': config['tune_guard'], **metric})
                units.append({'deployment': dep, 'seed': seed, 'method': method,
                    'baseline_delta': 0., 'forbidden_delta': 0., 'row_order_delta': 0.,
                    'weights_gamma_beta_unchanged': True})
            base.MEMO.clear()
            print('QUALITY_STATE_CALIBRATED', dep, seed, flush=True)
    n.write_csv(ROOT/'tables/QUALITY_STATE_TUNE_TAIL.csv', tails)
    n.write_csv(ROOT/'tables/VALIDATION_GUARDS.csv', audits)
    n.write_csv(ROOT/'audit/INVARIANCE.csv', units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json', configurations)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_or_test_selection': True})


def install_export():
    old_export, old_consensus = n.public_frame, n.dev.BASE.consensus_real
    def export(frame, z, weights, method):
        out = old_export(frame, z, weights, method)
        for field in FIELDS:
            out[field] = frame[field].to_numpy() if field in frame else np.nan
        if method == n.BASELINE:
            out['actual_waveform_joint_BC'] = np.nan
            out['new_prediction_used_in_score'] = False
        return out
    def consensus(items, method):
        out = old_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(FIELDS)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus


def main():
    global ROOT, DATA, PREDICTIVE, SELECTION
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--predictive-root', type=Path, required=True)
    parser.add_argument('--selection-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'prepare', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT, DATA, PREDICTIVE, SELECTION = args.root, args.data_root, args.predictive_root, args.selection_root
    ctrl.ROOT, ctrl.SELECTION = ROOT, SELECTION
    engine.ROOT, engine.DATA, engine.PREDICTIVE = ROOT, DATA, PREDICTIVE
    base.ROOT, base.DATA = ROOT, DATA
    base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = ROOT
    r.install()
    install_export()
    co.score.matrices = co.matrices
    base.population = engine.population
    n.METHODS, n.load_panel, n.infer = METHODS, ctrl.panel, infer
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'prepare':
        ctrl.prepare()
    elif args.stage == 'calibrate':
        calibrate()
    else:
        n.run(ROOT, args.stage)


if __name__ == '__main__':
    main()
