#!/usr/bin/env python3
"""Finite, disclosed real-development exploration after OMC-DEVCONF."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

import mcwf_pe_frontend_extension_v2_20260907 as old
import mcwf_ordered_mass_predictor_20260907 as omc
import mcwf_ordered_mass_evaluate_20260907 as evaluate
import mcwf_seed_plateau_20260907 as consensus

dev, body, ev = old.dev, old.body, old.ev
PREVIOUS = dev.PROJECT / 'results/mcwf_pe_frontend_extension_20260907T041300Z'
RELEASE = PREVIOUS / 'release_MCWF_UNIFIED_OMC_DEVCONF'
TRIAL = PREVIOUS / 'trials/ORDERED-MASS-COMPLETE-GRID-PRIOR'
GAMMAS = (0., .125, .25, .5, 1., 2., 4.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)
ARMS = ('MIXTURE-PRIOR', 'MIXTURE-BC')


def write_json(path, value):
    dev.json_write(path, value)


def initialize(root):
    if root.exists():
        raise RuntimeError('Refusing existing output root')
    for name in ('contracts', 'scripts', 'logs', 'models', 'cache', 'trials',
                 'reports', 'tables', 'audit', 'manifest', 'figures'):
        (root / name).mkdir(parents=True)
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)
    package = dev.PROJECT / 'packages/MCWF_UNIFIED_OMC_DEVCONF_20260907.tar.gz'
    expected = 'a2d4a540b8291ac343684909ce7994a937d325f896cc0e956bef51e7a9f47bd3'
    assert dev.sha(package) == expected, 'Baseline archive identity mismatch'
    contract = {
        'code': 'MCWF-OMC-ENSEMBLE-EXPLORATORY',
        'utc': datetime.now(timezone.utc).isoformat(), 'baseline': str(RELEASE),
        'baseline_archive_sha256': expected,
        'frozen': ['old encoders', 'RNC-FRT encoders', 'OMC encoders',
                   'time', 'sky', 'outer C-fixed weights', 'strict scope',
                   '2s4096H1L1', 'all historical outputs'],
        'hypothesis': 'Uniform arithmetic mixture of three predictive mass distributions may reduce initialization errors; it is not a PE posterior.',
        'same_method_O3_O4a': True, 'no_training_this_initial_family': True,
        'ensemble_weight': '1/3 for each frozen model; not tuned with PE',
        'arms': ARMS, 'gamma_grid': GAMMAS, 'beta_grid': BETAS,
        'score': 'Z_FRT + gamma*finite_reference_mass_penalty + beta*bounded_simulation_calibrated_ensemble_evidence; unchanged OMC baseline is an explicit option',
        'calibration': '512 development source systems; class-balanced isotonic; overlapping null pairs not independent; same floor/cap/boundary handling as OMC baseline',
        'real_selection_disclosure': 'PE and official outcomes DO select the finite development operating point. NOT a blinded real test and NOT simulation-validation-only selection.',
        'forbidden_inputs': ['real PE', 'official labels', 'event IDs', 'time changes', 'sky changes'],
        'reuse_disclosure': 'Earlier validation/test and earlier independent confirmations are now development data if used. New confirmation after selection required for a retrieval claim.',
        'simulation_guard': {'R10_drop_max': .02, 'AP_drop_max': .005,
                             'F50_F90_ratio_max': 1.1},
        'development_acceptance': {
            'both_runs': True, 'budgets': [10, 20],
            'PE': 'Catastrophic count no higher, BC>=.5 and Dmax<=3 counts no lower, median BC no lower at either budget; at least one BC count or median increase per run',
            'official': 'Frontend and published Hanabi overlaps no lower at either budget; sum of each of these two counts must increase per run',
            'waveform_only': 'Top10 catastrophic count, BC count and Dmax count no worse',
            'tie_break': 'PE count gains, median gains, total frontend+Hanabi gains, coefficient sum, lexical IDs',
            'no_success_required': 'If no admissible method qualifies, retain baseline; never alter labels, scopes or thresholds to force success'},
        'references': ['https://arxiv.org/abs/1612.01474',
                       'https://arxiv.org/abs/1807.07062'],
        'references_scope': 'Motivation for predictive ensembles and mass consistency, not proof of GW calibration or numerical thresholds.',
        'final_status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'}
    write_json(root / 'contracts/EXPERIMENT_CONTRACT.json', contract)
    rows = []
    for folder in (RELEASE, TRIAL / 'evaluation',
                   PREVIOUS / 'ordered_mass_predictor/models',
                   PREVIOUS / 'ordered_mass_predictor/calibration'):
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                rows.append({'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size})
    dev.csv_write(root / 'manifest/PROTECTED_INPUT_SHA256.csv', pd.DataFrame(rows))
    budgets = pd.read_csv(RELEASE / 'tables/PE_OFFICIAL_ALL.csv')
    dev.csv_write(root / 'contracts/BASELINE_BUDGETS.csv', budgets)
    for dep in old.DEPS:
        pe = pd.read_parquet(PREVIOUS / f'audit/{dep}_frozen_external_reference.parquet')
        pe.to_parquet(root / f'audit/{dep}_external_reference.parquet', index=False)
    print(json.dumps({'initialized': str(root), 'protected_files': len(rows)}), flush=True)


@lru_cache(maxsize=6)
def checkpoint(dep, ms):
    path = PREVIOUS / f'ordered_mass_predictor/models/{dep}/seed_{ms}/validation_selected_model.pt'
    return torch.load(path, weights_only=False, map_location='cpu')


def raw_prediction(root, dep, ms, es, split):
    path = root / f'cache/predictions/{dep}/{es}_{split}_{ms}.npz'
    if path.exists():
        return dict(np.load(path))
    if split == 'development':
        original = PREVIOUS / f'ordered_mass_predictor/models/{dep}/seed_{ms}/validation_predictions.npz'
        a = dict(np.load(original))
    else:
        original = PREVIOUS / f'ordered_mass_predictor/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
        if original.exists():
            a = dict(np.load(original))
        else:
            coarse = np.load(old.TRAINED / f'cache/deployment_event_psd/{dep}' /
                             ('real_features.npy' if split == 'real' else f'{es}_{split}_features.npy'))
            finepath = PREVIOUS / f'fine_mass_context/features/{dep}' / ('real.npy' if split == 'real' else f'{es}_{split}.npy')
            if not finepath.exists():
                raise RuntimeError(f'Missing read-only fine feature cache: {finepath}')
            x = omc.arrange(coarse, np.load(finepath))
            ck = checkpoint(dep, ms)
            model = omc.Predictor().cuda().eval()
            model.load_state_dict(ck['model'])
            p, outside = omc.probability(omc.infer(model, (x - ck['mu']) / ck['sd']), ck['temperature'])
            if split == 'real':
                full, events = dev.real_inputs(dep)
                valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
                pp, bb = np.full((len(full), 512), np.nan), np.full(len(full), np.nan)
                pp[valid], bb[valid] = p, outside
                p, outside = pp, bb
            a = {'p': p, 'outside': outside}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **a)
    return a


def ensemble(root, dep, es, split):
    all_a = [raw_prediction(root, dep, ms, es, split) for ms in body.MODEL_SEEDS]
    p = np.stack([a['p'] for a in all_a]).mean(0)
    outside = np.stack([a['outside'] for a in all_a]).mean(0)
    if split == 'development':
        for a in all_a[1:]:
            assert np.array_equal(a['group'], all_a[0]['group'])
            assert np.array_equal(a['truth'], all_a[0]['truth'])
    return {'p': p, 'outside': outside, **({k: all_a[0][k] for k in ('group', 'truth')}
                                        if split == 'development' else {})}


def prior(dep):
    ps = [checkpoint(dep, ms)['prior'] for ms in body.MODEL_SEEDS]
    assert all(np.array_equal(ps[0], p) for p in ps[1:])
    return ps[0]


def calibrate(root, dep, es=None):
    path = root / f'contracts/{dep}_ENSEMBLE_CALIBRATION.json'
    if path.exists():
        return json.loads(path.read_text())
    a = ensemble(root, dep, 0, 'development')
    i, j = np.triu_indices(len(a['p']), 1)
    y = a['group'][i] == a['group'][j]
    x = evaluate.pair_features(a, i, j, prior(dep))
    weights = np.where(y, .5 / y.sum(), .5 / (~y).sum())
    spec = {'mass_reference': np.sort(-np.log(x['bc'][y])).tolist(),
            'independent_source_systems': len(np.unique(a['group'])),
            'source_pairs': int(y.sum()), 'dependent_null_pairs': int((~y).sum()),
            'noise_blocks': 32}
    for feature in ('bc', 'prior_overlap'):
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x[feature], y, sample_weight=weights)
        q = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
        spec[feature] = {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(q) - np.log1p(-q)).tolist(),
                         'minimum': float(x[feature].min()), 'maximum': float(x[feature].max())}
    write_json(path, spec)
    return spec


def values(root, dep, es, split):
    f = pd.read_parquet(TRIAL / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = ensemble(root, dep, es, split)
    return f, evaluate.pair_features(a, f.idx_i.to_numpy(int), f.idx_j.to_numpy(int), prior(dep))


def score(f, x, spec, arm):
    if spec.get('unchanged_baseline'):
        return f.waveform_score.to_numpy(float), np.zeros(len(f)), np.zeros(len(f))
    frame = f.copy()
    mode = spec.get('score_mode', 'FRT_BASE')
    frame['waveform_score'] = (f.previous_waveform_score if mode == 'REPLACE_FRT'
                               else f.waveform_score if mode == 'ADD_POSITIVE'
                               else f.FRT_baseline_waveform_score)
    z, penalty, increment = evaluate.score(frame, x, spec, arm.split('-')[-1])
    if mode == 'ADD_POSITIVE':
        increment = np.maximum(increment, 0.)
        z = frame.waveform_score.to_numpy(float) + spec['beta'] * increment
    return z, penalty, increment


def batch_target(combos, pools, pe, baseline, dep):
    bc = pe.pe_mc_bhattacharyya_coefficient.to_numpy(float)
    vals = {'catastrophic_mc': ((bc < .1) | (pe.pe_mc_standardized_distance > 5)).to_numpy(),
            'BC_mc_ge_0p5': bc >= .5, 'Dmax_le_3': (pe.pe_dmax_intrinsic <= 3).to_numpy(),
            'official_frontend': pe['official_po_or_ml_fpp_below_0p01' if dep == 'gwtc3' else 'official_po_or_phazap_fpp_below_0p01'].fillna(False).to_numpy(bool),
            'official_hanabi': pe.official_any_pair_resolved_hanabi_overlap.fillna(False).to_numpy(bool)}
    out = {k: np.zeros(len(combos), float) for k in ('BC_gain', 'median_gain', 'front_gain', 'hanabi_gain')}
    out['noninferior'] = np.ones(len(combos), bool)
    for method in ('C_fixed', 'waveform_only'):
        ranks = [np.stack([c['rank_arrays'][method][0] for c in pool])[combos[:, k]] for k, pool in enumerate(pools)]
        scores = [np.stack([c['rank_arrays'][method][1] for c in pool])[combos[:, k]] for k, pool in enumerate(pools)]
        mean = np.mean(ranks, 0)
        order = np.lexsort((np.broadcast_to(np.arange(len(pe)), mean.shape), -np.mean(scores, 0),
                            np.max(ranks, 0), mean), axis=1)
        for b in ((10, 20) if method == 'C_fixed' else (10,)):
            oldrow = baseline[(baseline.deployment == dep) & (baseline.method == method) & (baseline.budget == b)].iloc[0]
            ix = order[:, :b]
            counts = {k: v[ix].sum(1) for k, v in vals.items()}
            med = np.median(bc[ix], 1)
            okay = ((counts['catastrophic_mc'] <= oldrow.catastrophic_mc) &
                    (counts['BC_mc_ge_0p5'] >= oldrow.BC_mc_ge_0p5) &
                    (counts['Dmax_le_3'] >= oldrow.Dmax_le_3))
            if method == 'C_fixed':
                okay &= (med >= oldrow.median_BC_mc - 1e-12)
                okay &= (counts['official_frontend'] >= oldrow.official_frontend)
                okay &= (counts['official_hanabi'] >= oldrow.official_hanabi)
                for src, dst in (('BC_mc_ge_0p5', 'BC_gain'), ('official_frontend', 'front_gain'), ('official_hanabi', 'hanabi_gain')):
                    out[dst] += counts[src] - oldrow[src]
                out['median_gain'] += med - oldrow.median_BC_mc
                out[f'Top{b}_median_BC_mc'] = med
                for key, value in counts.items():
                    out[f'Top{b}_{key}'] = value
            out['noninferior'] &= okay
    out['target_pass'] = (out['noninferior'] & ((out['BC_gain'] > 0) | (out['median_gain'] > 1e-10)) &
                          (out['front_gain'] > 0) & (out['hanabi_gain'] > 0))
    return out


def run(root, arm):
    trial = root / 'trials' / arm
    if trial.exists():
        raise RuntimeError('Refusing existing trial')
    for name in ('contracts', 'tables', 'evaluation', 'results', 'configs'):
        (trial / name).mkdir(parents=True)
    base = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    base = base[base.seed.astype(str) == 'consensus']
    data, selected, rows, all_combos = {}, {}, [], []
    for dep in old.DEPS:
        pe = pd.read_parquet(root / f'audit/{dep}_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        pools = []
        for es in dev.SEEDS:
            cal = calibrate(root, dep, es)
            fs, xs, bm = {}, {}, {}
            for split in ('validation', 'test', 'real'):
                fs[split], xs[split] = values(root, dep, es, split)
                if split != 'real':
                    bm[split] = ev.metrics(fs[split], fs[split].waveform_score.to_numpy(float), dep, es)
            data[dep, es] = fs, xs, bm
            pool = [{'id': 'UNCHANGED', 'spec': {'unchanged_baseline': True, 'gamma': 0., 'beta': 0.}}]
            for gamma in GAMMAS:
                for beta in BETAS:
                    spec = {**cal, 'gamma': gamma, 'beta': beta}
                    z, _, _ = score(fs['validation'], xs['validation'], spec, arm)
                    vm = ev.metrics(fs['validation'], z, dep, es)
                    vp, tp = ev.guard(vm, bm['validation']), False
                    if vp:
                        z, _, _ = score(fs['test'], xs['test'], spec, arm)
                        tp = ev.guard(ev.metrics(fs['test'], z, dep, es), bm['test'])
                    ident = f'G{gamma:g}_B{beta:g}'
                    rows.append({'deployment': dep, 'seed': es, 'id': ident, 'gamma': gamma, 'beta': beta,
                                 'validation_pass': vp, 'reused_test_pass': tp,
                                 **{m + '_' + k: v for m, obj in vm.items() for k, v in obj.items()}})
                    if vp and tp:
                        pool.append({'id': ident, 'spec': spec})
            for c in pool:
                z, _, _ = score(fs['real'], xs['real'], c['spec'], arm)
                c['rank_arrays'] = consensus.rank_arrays(fs['real'], z, dep, es, pe.pair_key.tolist())
            pools.append(pool)
            write_json(trial / f'configs/{dep}_{es}_ELIGIBLE.json', {'eligible': [{k: v for k, v in c.items() if k != 'rank_arrays'} for c in pool]})
            print(json.dumps({'arm': arm, 'dep': dep, 'seed': es, 'eligible': len(pool)}), flush=True)
        dev.csv_write(trial / 'tables/VALIDATION_GRID.csv', pd.DataFrame(rows))
        combos = np.array(list(itertools.product(*(range(len(p)) for p in pools))), int)
        qualified = []
        for start in range(0, len(combos), 128):
            block = combos[start:start + 128]
            results = batch_target(block, pools, pe, base, dep)
            for k, choice in enumerate(block):
                records = [pool[c] for pool, c in zip(pools, choice)]
                ids = [r['id'] for r in records]
                row = {'deployment': dep, 'combination': start + k,
                       **{f'seed{j + 1}': ids[j] for j in range(3)},
                       **{a: v[k].item() for a, v in results.items()}}
                all_combos.append(row)
                if row['target_pass']:
                    key = (-row['BC_gain'], -row['median_gain'], -row['front_gain'] - row['hanabi_gain'],
                           sum(r['spec']['gamma'] + r['spec']['beta'] for r in records), ids)
                    qualified.append((key, records, row))
        dev.csv_write(trial / 'tables/ALL_COMBINATIONS.csv', pd.DataFrame(all_combos))
        if qualified:
            _, records, row = min(qualified, key=lambda z: z[0])
            selected[dep] = {es: r['spec'] for es, r in zip(dev.SEEDS, records)}
            write_json(trial / f'contracts/{dep}_SELECTED.json', {'configs': selected[dep], 'development_outcome': row})
        print(json.dumps({'completed_arm': arm, 'deployment': dep, 'combinations': len(combos),
                          'qualifying': len(qualified)}), flush=True)
    write_json(trial / 'contracts/SEARCH_COMPLETE.json', {'both_runs_qualify': len(selected) == 2,
                                                        'selected': selected, 'fresh_confirmation_done': False})
    if len(selected) != 2:
        return
    metrics, guards = [], []
    for dep in old.DEPS:
        real = {}
        for es in dev.SEEDS:
            fs, xs, bm = data[dep, es]
            for split, f in fs.items():
                z, penalty, inc = score(f, xs[split], selected[dep][es], arm)
                n = f.drop(columns=[c for c in ('final_score', 'rank', 'method', 'waveform_contribution',
                                                'time_contribution', 'sky_contribution') if c in f]).copy()
                n['OMC_baseline_waveform_score'], n['waveform_score'] = f.waveform_score, z
                n['ensemble_mass_penalty'], n['ensemble_mass_increment'] = penalty, inc
                for name, val in xs[split].items():
                    n['ensemble_' + name] = val
                for c in ('time_score', 'sky_raw_log_bf', 'idx_i', 'idx_j'):
                    assert np.array_equal(n[c], f[c])
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = n
                else:
                    mm = ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm[split])})
                    for method in mm:
                        for config, m in [('OMC_BASELINE', bm[split][method]), ('CANDIDATE', mm[method])]:
                            metrics.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(metrics))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))


def tests():
    p = np.array([[.1, .4, .5], [.6, .2, .2], [.2, .5, .3]])
    a = {'p': p, 'outside': np.zeros(3)}
    i, j = np.array([0, 1]), np.array([1, 2])
    x = evaluate.pair_features(a, i, j, np.array([.2, .3, .5]))
    y = evaluate.pair_features(a, j, i, np.array([.2, .3, .5]))
    for key in x:
        assert np.array_equal(x[key], y[key]), key
    assert np.allclose(np.stack([p] * 3).mean(0), p)
    frame = pd.DataFrame({'waveform_score': [1., 2.]})
    assert np.array_equal(score(frame, x, {'unchanged_baseline': True}, 'MIXTURE-BC')[0], frame.waveform_score)
    print('PASS: pair symmetry, equal-model mixture identity, unchanged-score identity', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=('init', 'run', 'test'), required=True)
    p.add_argument('--arm', choices=ARMS)
    args = p.parse_args()
    if args.phase == 'init':
        initialize(args.root)
    elif args.phase == 'test':
        tests()
    else:
        run(args.root, args.arm)
