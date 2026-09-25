#!/usr/bin/env python3
"""Empirically calibrate profile errors conditional on waveform curvature."""
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
from scipy.optimize import minimize
from scipy.stats import t as student

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_mode_aware_20260909 as mode
n, r, d, cal = mode.n, mode.r, mode.d, mode.c
PARENT = P / 'results/mcwf_nodup_mode_profile_10_20260909T085830Z'
PROFILE = P / 'results/mcwf_nodup_lowmass_profile_07_20260909T082227Z'
ROOT = None


def weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1 / counts[inverse]
    return w / w.sum()


def scales(h, spec):
    return np.exp(spec['intercept'] + spec['slope'] * (np.log(h) - spec['logh_center']))


def logpdf(truth, centers, h, spec):
    scale = scales(h, spec)
    loc = centers + spec['location']
    norm = student.cdf((d.EDGES[-1] - loc) / scale, spec['df']) - student.cdf((d.EDGES[0] - loc) / scale, spec['df'])
    return student.logpdf((truth - loc) / scale, spec['df']) - np.log(scale) - np.log(norm.clip(1e-300))


def density(centers, h, spec, edges=None):
    edges = d.EDGES if edges is None else edges
    scale, loc = scales(h, spec), centers + spec['location']
    cdf = student.cdf((edges[None, :] - loc[:, None]) / scale[:, None], spec['df'])
    p = np.diff(cdf, axis=1).clip(0.)
    p /= p.sum(1, keepdims=True)
    return p


def fit(truth, centers, h, groups, df, ridge):
    w = weights(groups)
    mu = float(np.sum(w * np.log(h)))
    def objective(theta):
        spec = {'location': theta[0], 'intercept': theta[1], 'slope': theta[2], 'logh_center': mu, 'df': df}
        return -np.sum(w * logpdf(truth, centers, h, spec)) + .5 * ridge * theta[2]**2
    result = minimize(objective, [0., np.log(.005), .5], method='L-BFGS-B',
                      bounds=[(-.02, .02), (np.log(.0002), np.log(.2)), (0., 2.)],
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-7})
    if not result.success:
        raise RuntimeError('Conditional error fit failed: ' + result.message)
    return {'location': float(result.x[0]), 'intercept': float(result.x[1]), 'slope': float(result.x[2]),
            'logh_center': mu, 'df': df, 'ridge': ridge, 'minimum_h': float(h.min()), 'maximum_h': float(h.max()),
            'fit_sources': len(np.unique(groups)), 'fit_events': len(h), 'fit_objective': float(result.fun),
            'hessian_is_PE_covariance': False}


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'calibration', 'predictions', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest'):
        (ROOT / name).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-HETEROSCEDASTIC-PROFILE-16', 'UTC': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'same_both_runs': True,
        'motivation': 'One constant Student-t error scale pools precise and ambiguous waveform profiles. Test whether waveform-curvature-dependent empirical error calibration improves proper predictive loss.',
        'parent': str(PARENT), 'profiles': str(PROFILE),
        'new_feature': 'sqrt(inverse(-Hessian of profile power)[logMc,logMc]);finite difference steps .0008,.004,.004;positive definite and interior only.',
        'important': 'This is NOT a normalized likelihood Hessian or Fisher PE covariance. It is only an input to simulation-error calibration.',
        'density': 'Truncated Student-t(logMc; center=profile+location, scale=exp(intercept+slope*(log(h)-fitmean)));slope[0,2].',
        'grid': {'df': [3, 5, 10, 30], 'ridge_on_slope': [.001, .01, .1, 1.]},
        'fit': 'Only R10-quality-active,interior-positive-curvature events in source/noise fold0;one totalweight per source;atleast20sources.',
        'fallback': 'Keep exact R10 predictive density if quality fails,curvatureinvalid,or outside frozen fit curvature support. No narrowing unsupported rows.',
        'selection': 'Fold1 source-weighted proper log-density loss,then smaller slope andstronger ridge. No real/test orpublicPE input.',
        'predictive_gate': 'Bothruns:fold1NLL no worse than R10;MAE no worse than original NN;no additional >10percent point errors versus NN;sourcebootstrap95% coverage intervalcontains.90 globally AND in curvature-active subset;atleast20active tunesources.',
        'score_stage': 'Not authorized by this stage until predictivegatepasses. No new waveform rankings in this script.',
        'frozen': ['encoder', 'time', 'sky', 'outerweights', 'oldresults', 'scope', 'paper'],
        'oldMc_q_or_total_blend': False,
        'refs': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/1804.06788']})
    files = pd.read_csv(PARENT / 'manifest/INPUT_SHA256.csv').to_dict('records')
    for folder in (PROFILE / 'events', PARENT / 'calibration'):
        files.extend({'path': str(p), 'sha256': n.sha(p), 'bytes': p.stat().st_size} for p in sorted(folder.glob('*.json')))
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', files)
    shutil.copy2(__file__, ROOT / 'scripts/profile_heteroscedastic.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json')})


def coverage_interval(p, truth, groups, mask):
    pit = np.array([np.interp(t, d.EDGES, np.r_[0., pp.cumsum()]) for t, pp in zip(truth[mask], p[mask])])
    hit = ((pit >= .05) & (pit <= .95)).astype(float)
    names, inv = np.unique(groups[mask], return_inverse=True)
    v = np.bincount(inv, weights=hit) / np.bincount(inv)
    rng = np.random.default_rng(2026091016)
    draws = rng.integers(len(names), size=(5000, len(names)))
    lo, hi = np.quantile(v[draws].mean(1), [.025, .975])
    return float(v.mean()), float(lo), float(hi), len(names)


def calibrate():
    selected, grid, outputs = {}, [], []
    previous = json.loads((PARENT / 'calibration/PROFILE_PREDICTIVE.json').read_text())
    for dep in n.DEPS:
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment'] = dep
        fold, truth = r.source_fold(meta), np.log(meta.mc_det.to_numpy())
        groups = meta.source_uid.to_numpy(str)
        original = np.mean(cal.parent_predictions(dep), axis=0)
        parent_data = dict(np.load(PARENT / f'predictions/{dep}_development_ENSEMBLE.npz'))
        parent = parent_data['p']
        rows = [json.loads(path.read_text()) for path in sorted((PROFILE / 'events').glob(dep + '_*.json'))]
        hessian = np.full(len(meta), np.nan)
        centers = parent_data['profile_centers']
        for row in rows:
            if row.get('hessian_interior') and row.get('hessian_positive_definite'):
                hessian[int(row['row_index'])] = row['diagnostic_logmc_curvature_width_NOT_PE']
        eligible = parent_data['active'] & np.isfinite(hessian) & (hessian > 0)
        fit_mask = eligible & (fold == 0)
        if len(np.unique(groups[fit_mask])) < 20:
            selected[dep] = {'pass': False, 'reason': 'Less than20independentfit sources'}
            continue
        bins = np.searchsorted(d.EDGES, truth, side='right').clip(1, 512) - 1
        parent_log = np.log(parent[np.arange(len(parent)), bins].clip(1e-300)) - np.log(np.diff(d.EDGES)[bins])
        # R10 profile densities are continuous within the grid; use their exact
        # normalized density rather than comparing a continuous score to bin averages.
        pa = parent_data['active']
        parent_log[pa] = cal.normalized_logpdf(truth[pa], centers[pa], previous[dep]['spec'])
        tune = fold == 1
        wt = weights(groups[tune])
        ref_nll = -float(wt @ parent_log[tune])
        olderr = abs(r.mass_summary(original)[:, 0] - truth)
        options = []
        for df in (3, 5, 10, 30):
            for ridge in (.001, .01, .1, 1.):
                spec = fit(truth[fit_mask], centers[fit_mask], hessian[fit_mask], groups[fit_mask], df, ridge)
                active = eligible & (hessian >= spec['minimum_h']) & (hessian <= spec['maximum_h'])
                pp = parent.copy()
                pp[active] = density(centers[active], hessian[active], spec)
                lp = parent_log.copy()
                lp[active] = logpdf(truth[active], centers[active], hessian[active], spec)
                error = abs(r.mass_summary(pp)[:, 0] - truth)
                cov, lo, hi, nt = coverage_interval(pp, truth, groups, tune)
                acov, alo, ahi, na = coverage_interval(pp, truth, groups, tune & active)
                row = {'deployment': dep, 'df': df, 'ridge': ridge, 'slope': spec['slope'],
                    'fit_sources': spec['fit_sources'], 'tune_sources': nt, 'active_tune_sources': na,
                    'NLL': -float(wt @ lp[tune]), 'R10_NLL': ref_nll,
                    'MAE': float(wt @ error[tune]), 'NN_MAE': float(wt @ olderr[tune]),
                    'catastrophic10percent': int((error[tune] > np.log(1.1)).sum()),
                    'NN_catastrophic10percent': int((olderr[tune] > np.log(1.1)).sum()),
                    'coverage90': cov, 'coverage_low': lo, 'coverage_high': hi,
                    'active_coverage90': acov, 'active_coverage_low': alo, 'active_coverage_high': ahi}
                row['PASS'] = bool(row['NLL'] <= ref_nll and row['MAE'] <= row['NN_MAE'] and
                    row['catastrophic10percent'] <= row['NN_catastrophic10percent'] and
                    na >= 20 and lo <= .9 <= hi and alo <= .9 <= ahi)
                options.append((row, spec, pp, active))
                grid.append(row)
        passed = [x for x in options if x[0]['PASS']]
        row, spec, pp, active = min(passed or options, key=lambda x: (x[0]['NLL'], x[0]['slope'], -x[0]['ridge']))
        selected[dep] = {'pass': bool(passed), 'selection': row, 'spec': spec}
        np.savez_compressed(ROOT / f'predictions/{dep}_development.npz', p=pp, parent_p=parent,
                            active=active, profile_active=pa, curvature_width=hessian, profile_centers=centers,
                            fold=fold, truth=truth)
        outputs.append(row)
        print('HETEROSCEDASTIC_PREDICTIVE', dep, selected[dep], flush=True)
    n.write_csv(ROOT / 'tables/PREDICTIVE_GRID.csv', grid)
    n.write_csv(ROOT / 'tables/PREDICTIVE_SELECTION.csv', outputs)
    n.write_json(ROOT / 'calibration/PROFILE_HETEROSCEDASTIC.json', selected)
    n.write_json(ROOT / 'contracts/PREDICTIVE_FROZEN.json', {'UTC': n.utc(),
        'both_runs_pass': all(x['pass'] for x in selected.values()),
        'sha256': n.sha(ROOT / 'calibration/PROFILE_HETEROSCEDASTIC.json'), 'no_real_test_scored': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate'), required=True)
    args = parser.parse_args()
    ROOT = args.root
    globals()[args.stage]()
