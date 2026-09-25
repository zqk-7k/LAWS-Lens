#!/usr/bin/env python3
"""Recalibrate frozen physical profiles on new, explicit source/noise folds."""
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

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_bounded_information_runtime_20260909 as analytic
import mcwf_profile_mode_aware_20260909 as mode
h, n, r, d, cal = analytic.h, mode.n, mode.r, mode.d, mode.c
ROOT = DATA = None
KINDS = ('GLOBAL', 'EXPECTED', 'BOUNDED', 'STRENGTH')
PARENT = h.PARENT


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'calibration', 'predictions', 'tables', 'audit', 'scripts',
                 'logs', 'manifest', 'reports', 'figures'):
        (ROOT/name).mkdir(parents=True)
    contract = {'UTC': n.utc(), 'id': 'MCWF-NODUP-INDEPENDENT-PREDICTIVE-29',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'data': str(DATA), 'same_method_for_both_runs': True,
        'reason': 'Old profile-error calibration used only38-39 independent fit sources. Refit predictive uncertainty on registered new1024parents/run, without modifying waveforms, optimizer, centers or event quality.',
        'fold': 'Use explicit pre-generation metadata fold0=fit/fold1=tune; never rehash source IDs. Each source total weight1; images remain together. Noise blocks and parent4096s chunks disjoint across folds.',
        'frozen': ['encoders', 'R10eventquality', 'time', 'sky', 'outerweights', 'historicalscopes/results'],
        'arms': list(KINDS),
        'GLOBAL': 'Refit one normalized Student-t residual location/scale for active profiles; no additional event information.',
        'EXPECTED': 'Unchanged R18 expected derivative information width, only a predictive covariate, not a PE covariance.',
        'BOUNDED': 'Unchanged R21 bounded reference-prior mean/width; same64/128quadrature gates. No inference of public PE priors.',
        'STRENGTH': 'Inverse profile projection-strength proxy from fitted quadratures/local variance; not public or PSD-optimal network SNR.',
        'grid': {'df': [3, 5, 10, 30], 'slope_ridge': [.001, .01, .1, 1.]},
        'support': 'Minimum20 independent fit AND active tune sources. Conditional covariates outside fit range retain exact R10 density. No widening support after viewing outcomes.',
        'fit': 'Source-weighted proper normalized log-density loss. Constant arm uses unchanged fit_error; conditional arms use already gradient-tested bounded_information_runtime fit.',
        'selection': 'Each arm separately, source-disjoint tune NLL; tie smaller slope then stronger ridge. Only candidates satisfying the predictive Gate are eligible; all failed grids retained.',
        'gate': 'Bothruns independently: NLL<=R10; MAE and >10percent catastrophic count<=NN; source-bootstrap95percent coverage intervals contain.9 globally AND active; >=20 active tune sources.',
        'uncertainty_limit': 'Source bootstrap is conditional on16 fixed noise parent chunks/run. Also report chunk-level coverage diagnostic. It is not a fully independent noise or postselection confidence statement.',
        'scoring': 'Not performed by this script. A later scoring arm can replace one mass marginal and calibrate ONE waveform classifier, only after both-run predictive Gate; no old Mc/q heads or total-score mixture.',
        'references': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/gr-qc/0703086',
                       'https://arxiv.org/abs/1804.06788'],
        'real_PE_official_test_inputs': False}
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', contract)
    paths = [DATA/'contracts/BULK_NOISE_ACQUISITION_ADDENDUM.json',
             DATA/'contracts/FEATURE_PROFILE_CONTRACT.json',
             PARENT/'calibration/PROFILE_PREDICTIVE.json', Path(mode.__file__), Path(analytic.__file__)]
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', [{'path': str(p), 'sha256': n.sha(p),
        'bytes': p.stat().st_size} for p in paths])
    shutil.copy2(__file__, ROOT/'scripts/independent_predictive_calibration.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def inputs(dep):
    if not (DATA/f'contracts/{dep}_PROFILES_COMPLETE.json').exists():
        raise RuntimeError('Independent profile data not complete:'+dep)
    meta = pd.read_parquet(DATA/f'data/{dep}/event_metadata.parquet')
    original = np.load(DATA/f'predictions/{dep}/ensemble_mass.npy')
    records = [json.loads(p.read_text()) for p in sorted((DATA/f'profile_events/{dep}').glob('*.json'))]
    previous = json.loads((PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    active, center = mode.quality(original, records, previous['parent_coverage'])
    truth = np.log(meta.mc_det.to_numpy())
    parent = original.copy()
    parent[active] = cal.density(center[active], previous)
    bins = np.searchsorted(d.EDGES, truth, side='right').clip(1, 512)-1
    lp = np.log(parent[np.arange(len(parent)), bins].clip(1e-300))-np.log(np.diff(d.EDGES)[bins])
    lp[active] = cal.normalized_logpdf(truth[active], center[active], previous)
    widths = {key: np.full(len(meta), np.nan) for key in KINDS}
    centers = {key: center.copy() for key in KINDS}
    widths['GLOBAL'][active] = 1.
    for path in (DATA/f'profile_information/{dep}').glob('*.json'):
        row = json.loads(path.read_text()); index = int(row['row_index'])
        if row.get('information_valid'):
            widths['EXPECTED'][index] = row['information_logmc_width']
        if 'coefficients' in row and 'offsource_variance' in row:
            strength = np.sqrt(np.sum(np.asarray(row['coefficients'])**2 /
                               np.asarray(row['offsource_variance'])[:, None]))
            if np.isfinite(strength) and strength > 0:
                widths['STRENGTH'][index] = 1./strength
    for path in (DATA/f'profile_bounded_information/{dep}').glob('*.json'):
        row = json.loads(path.read_text()); index = int(row['row_index'])
        if row.get('valid'):
            widths['BOUNDED'][index] = row['width128']
            centers['BOUNDED'][index] = row['mean128']
    return meta, original, parent, lp, active, center, centers, widths


def cluster_coverage(p, truth, groups, mask):
    if not mask.any():
        return {'coverage': None, 'low': None, 'high': None, 'groups': 0}
    cov, lo, hi, count = h.coverage_interval(p, truth, groups, mask)
    return {'coverage': cov, 'low': lo, 'high': hi, 'groups': count}


def calibrate():
    outputs, grids, selected = [], [], {}
    previous = pd.read_csv(ROOT/'manifest/INPUT_SHA256.csv')
    if any(n.sha(Path(row.path)) != row.sha256 for row in previous.itertuples()):
        raise RuntimeError('Frozen inputs changed')
    for dep in n.DEPS:
        meta, original, parent, parent_lp, pa, oldcenter, centers, widths = inputs(dep)
        truth, fold = np.log(meta.mc_det.to_numpy()), meta.fold.to_numpy(int)
        groups = meta.source_uid.to_numpy(str)
        noise = pd.read_csv(DATA/f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        chunk = meta.noise_bank_index.map(noise.parent_file_gps).to_numpy()
        if set(groups[fold == 0]) & set(groups[fold == 1]) or set(chunk[fold == 0]) & set(chunk[fold == 1]):
            raise RuntimeError('New source or parent-noise chunk leakage')
        tune = fold == 1; weight = h.weights(groups[tune])
        ref_nll = -float(weight@parent_lp[tune])
        olderr = abs(r.mass_summary(original)[:, 0]-truth)
        for kind in KINDS:
            center, width = centers[kind], widths[kind]
            eligible = pa & np.isfinite(width) & (width > 0)
            fit = eligible & (fold == 0)
            key = dep+'_'+kind
            if len(np.unique(groups[fit])) < 20:
                selected[key] = {'pass': False, 'reason': 'Insufficient independent fit sources'}
                continue
            options = []
            for df in (3, 5, 10, 30):
                for ridge in ([0.] if kind == 'GLOBAL' else [.001, .01, .1, 1.]):
                    if kind == 'GLOBAL':
                        spec = cal.fit_error(truth[fit], center[fit], groups[fit], df)
                        active = eligible.copy()
                    else:
                        spec = analytic.fit(truth[fit], center[fit], width[fit], groups[fit], df, ridge)
                        active = eligible & (width >= spec['minimum_h']) & (width <= spec['maximum_h'])
                    pp, lp = parent.copy(), parent_lp.copy()
                    if kind == 'GLOBAL':
                        pp[active] = cal.density(center[active], spec)
                        lp[active] = cal.normalized_logpdf(truth[active], center[active], spec)
                    else:
                        pp[active] = h.density(center[active], width[active], spec)
                        lp[active] = h.logpdf(truth[active], center[active], width[active], spec)
                    if not np.isfinite(pp).all() or abs(pp.sum(1)-1).max() > 1e-10:
                        raise RuntimeError('Invalid normalized predictive mass')
                    error = abs(r.mass_summary(pp)[:, 0]-truth)
                    cv = cluster_coverage(pp, truth, groups, tune)
                    acv = cluster_coverage(pp, truth, groups, tune & active)
                    row = {'deployment': dep, 'kind': kind, 'df': df, 'ridge': ridge,
                        'fit_sources': len(np.unique(groups[fit])), 'active_tune_sources': acv['groups'],
                        'NLL': -float(weight@lp[tune]), 'R10_NLL': ref_nll,
                        'MAE': float(weight@error[tune]), 'NN_MAE': float(weight@olderr[tune]),
                        'catastrophic10percent': int((error[tune] > np.log(1.1)).sum()),
                        'NN_catastrophic10percent': int((olderr[tune] > np.log(1.1)).sum()),
                        'coverage90': cv['coverage'], 'coverage_low': cv['low'], 'coverage_high': cv['high'],
                        'active_coverage90': acv['coverage'], 'active_coverage_low': acv['low'],
                        'active_coverage_high': acv['high'], 'slope': spec.get('slope', 0.)}
                    row['PASS'] = bool(row['NLL'] <= ref_nll and row['MAE'] <= row['NN_MAE'] and
                        row['catastrophic10percent'] <= row['NN_catastrophic10percent'] and
                        acv['groups'] >= 20 and cv['low'] <= .9 <= cv['high'] and acv['low'] <= .9 <= acv['high'])
                    options.append((row, spec, pp, active)); grids.append(row)
            passing = [v for v in options if v[0]['PASS']]
            row, spec, pp, active = min(passing or options, key=lambda v: (v[0]['NLL'], v[0]['slope'], -v[0]['ridge']))
            selected[key] = {'pass': bool(passing), 'spec': spec, 'selection': row,
                'noise_chunk_coverage_diagnostic': cluster_coverage(pp, truth, chunk.astype(str), tune & active),
                'noise_chunk_limit': 'Descriptive alternative clustering; source parents span two noise chunks, not fully independent events.'}
            np.savez_compressed(ROOT/f'predictions/{key}_development.npz', p=pp, parent_p=parent,
                active=active, profile_active=pa, profile_center=oldcenter, center=center, width=width,
                fold=fold, truth=truth)
            outputs.append(row)
            print('INDEPENDENT_PREDICTIVE', key, selected[key], flush=True)
    n.write_csv(ROOT/'tables/PREDICTIVE_GRID.csv', grids)
    n.write_csv(ROOT/'tables/PREDICTIVE_SELECTION.csv', outputs)
    n.write_json(ROOT/'calibration/INDEPENDENT_PREDICTIVE.json', selected)
    n.write_json(ROOT/'contracts/PREDICTIVE_FROZEN.json', {'UTC': n.utc(),
        'both_run_pass_by_kind': {k: all(selected.get(dep+'_'+k, {}).get('pass', False) for dep in n.DEPS) for k in KINDS},
        'sha256': n.sha(ROOT/'calibration/INDEPENDENT_PREDICTIVE.json'), 'real_or_test_scored': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate'), required=True)
    args = parser.parse_args(); ROOT, DATA = args.root, args.data_root
    globals()[args.stage]()
