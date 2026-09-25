#!/usr/bin/env python3
"""Conservative local profile refinement, independently versioned after08FAIL."""
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
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_predictive_calibration_20260909 as c
import mcwf_profile_predictive_evaluate_20260909 as application

n, r, d = c.n, c.r, c.d
ROOT = None
REGIONS = (.50, .68, .80, .90, .95, .99)
METHODS = ('NODUP-DIRECT-REPLAY', 'LOCAL-PROFILE-FIXED', 'LOCAL-PROFILE-VAL')


def quality(parent, rows, coverage):
    summary = r.mass_summary(parent)
    eligible = np.isfinite(summary[:, 0]) & (np.exp(summary[:, 0]) <= 15.)
    active, centers = np.zeros(len(parent), bool), np.full(len(parent), np.nan)
    for row in rows:
        i = int(row['row_index'])
        if not eligible[i]:
            continue
        value = float(row['logmc'])
        centers[i] = value
        lo, hi = np.interp([(1 - coverage) / 2, (1 + coverage) / 2], np.r_[0., parent[i].cumsum()], d.EDGES)
        # q=1 is the physical mass-ordering boundary, not unsupported q<.25.
        # No curvature-derived covariance is used at that boundary.
        active[i] = bool(row['best_run_converged'] and .251 < row['q'] <= 1. and
                         -.799 < row['chieff_equal'] < .799 and lo <= value <= hi)
    return active, centers


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independentdirectoryrequired')
    for folder in ('contracts', 'calibration', 'predictions', 'configs', 'tables', 'audit', 'reports',
                   'scripts', 'logs', 'manifest', 'results', 'figures', 'cache', 'profile_events'):
        (ROOT / folder).mkdir(parents=True)
    prior = P / 'results/mcwf_nodup_profile_predictive_08_20260909T083816Z'
    contract = json.loads((prior / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract.update(id='MCWF-NODUP-TRUSTED-LOCAL-PROFILE-09', UTC=n.utc(), previous_predictive_failure=str(prior),
        previous_not_rejudged=True, quality_grid=list(REGIONS),
        adaptive_simulation_development='08gavebetterNLL/MAEbutoneadditional>10percentfailureineachrun. Testboundedlocalrefinement. Existingtest/realresultsnotusedfornewthresholdselection.',
        quality='Bestoptimumconverged;rejectartificiallowerq=.25andspinboundaries. Physicalmassorderingq=1allowedbecauseuncertaintyisempiricallycalibrated,notHessian-derived. Parentprobabilityregionselectedonlyonsimulationfold1.',
        meaning='Replace the newmasspredictiononlywheretwo waveformestimatorsagree locally;otherwise retainparentdistribution.No total-scoreblend,nolegacyMc/qterm.',
        same_ensemble_everywhere=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.read_csv(prior / 'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__, ROOT / 'scripts/profile_trusted_local.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


def calibrate_predictive():
    if not (c.PROFILE / 'contracts/PROFILE_DEVELOPMENT_COMPLETE.json').exists():
        raise RuntimeError('Missingcompletedprofiledevelopment')
    selected, grid, summaries = {}, [], []
    for dep in n.DEPS:
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment'] = dep
        fold, truth, groups = r.source_fold(meta), np.log(meta.mc_det.to_numpy()), meta.source_uid.to_numpy(str)
        parent = np.mean(c.parent_predictions(dep), axis=0)
        rows = [json.loads(path.read_text()) for path in sorted((c.PROFILE / 'events').glob(dep + '_*.json'))]
        bins = np.searchsorted(d.EDGES, truth, side='right').clip(1, 512) - 1
        parent_logpdf = np.log(parent[np.arange(len(parent)), bins].clip(1e-300)) - np.log(np.diff(d.EDGES)[bins])
        parent_median = r.mass_summary(parent)[:, 0]
        tune = fold == 1
        parent_nll = -float(parent_logpdf[tune].mean())
        parent_error = abs(parent_median[tune] - truth[tune])
        gu, gn = pd.factorize(groups[tune], sort=True)
        rng = np.random.default_rng(2026091000)
        draws = rng.integers(len(gn), size=(5000, len(gn)))
        trials = []
        for coverage in REGIONS:
            active, centers = quality(parent, rows, coverage)
            fit = active & (fold == 0)
            if len(np.unique(groups[fit])) < 20:
                continue
            for df in (3, 5, 10, 30):
                spec = c.fit_error(truth[fit], centers[fit], groups[fit], df)
                spec['parent_coverage'] = coverage
                pp = parent.copy()
                pp[active] = c.density(centers[active], spec)
                med = r.mass_summary(pp)[:, 0]
                logpdf = parent_logpdf.copy()
                logpdf[active] = c.normalized_logpdf(truth[active], centers[active], spec)
                pit = np.array([np.interp(t, d.EDGES, np.r_[0., p.cumsum()]) for t, p in zip(truth, pp)])
                hit = ((pit[tune] >= .05) & (pit[tune] <= .95)).astype(float)
                per_source = np.bincount(gu, weights=hit) / np.bincount(gu)
                lo, hi = np.quantile(per_source[draws].mean(1), [.025, .975])
                error = abs(med[tune] - truth[tune])
                row = {'deployment': dep, 'parent_coverage': coverage, 'df': df, 'location': spec['location'],
                    'scale': spec['scale'], 'fit_sources': spec['fit_sources'], 'tune_sources': len(gn),
                    'NLL': -float(logpdf[tune].mean()), 'parent_NLL': parent_nll,
                    'MAE': float(error.mean()), 'parent_MAE': float(parent_error.mean()),
                    'catastrophic10percent': int((error > np.log(1.1)).sum()),
                    'parent_catastrophic10percent': int((parent_error > np.log(1.1)).sum()),
                    'coverage90': float(hit.mean()), 'coverage90_ci_low': float(lo), 'coverage90_ci_high': float(hi),
                    'activation_fraction': float(active.mean())}
                row['PASS'] = bool(row['NLL'] <= parent_nll and row['MAE'] <= row['parent_MAE'] and
                    row['catastrophic10percent'] <= row['parent_catastrophic10percent'] and lo <= .9 <= hi)
                trials.append((row, spec))
                grid.append(row)
        passing = [(a, b) for a, b in trials if a['PASS']]
        if not trials:
            selected[dep] = {'pass': False, 'reason': 'Insufficientsourcelevelsupport'}
            continue
        row, spec = min(passing or trials, key=lambda pair: (pair[0]['NLL'], pair[0]['parent_coverage'], -pair[0]['df']))
        selected[dep] = {'pass': bool(passing), 'selection': row, 'spec': spec}
        active, centers = quality(parent, rows, spec['parent_coverage'])
        pp = parent.copy()
        pp[active] = c.density(centers[active], spec)
        np.savez_compressed(ROOT / f'predictions/{dep}_development_ENSEMBLE.npz', p=pp, parent_p=parent,
                            active=active, profile_centers=centers, truth=truth, fold=fold)
        summaries.append(row)
        print('LOCAL_PROFILE_PREDICTIVE', dep, selected[dep], flush=True)
    n.write_csv(ROOT / 'tables/PROFILE_PREDICTIVE_GRID.csv', grid)
    n.write_csv(ROOT / 'tables/PROFILE_PREDICTIVE_SELECTION.csv', summaries)
    n.write_json(ROOT / 'calibration/PROFILE_PREDICTIVE.json', selected)
    n.write_json(ROOT / 'contracts/PREDICTIVE_FROZEN.json', {'UTC': n.utc(),
        'sha256': n.sha(ROOT / 'calibration/PROFILE_PREDICTIVE.json'), 'both_runs_pass': all(v['pass'] for v in selected.values()),
        'no_test_or_real_used_for_selection': True, 'status': n.STATUS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate_predictive', 'freeze_score', 'calibrate', 'evaluate', 'real'), required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    ROOT = c.ROOT = d.ROOT = r.ROOT = application.ROOT = application.score.ROOT = args.root
    application.WORKERS = args.workers
    c.quality_mask = quality
    application.score.METHODS = METHODS
    r.install()
    d.predict = application.predict
    n.METHODS, n.load_panel, n.infer = METHODS, application.score.load_panel, application.score.infer
    if args.stage in ('freeze', 'calibrate_predictive'):
        globals()[args.stage]()
    elif args.stage == 'freeze_score':
        application.freeze_score()
    elif args.stage == 'calibrate':
        application.score.calibrate()
    else:
        n.run(ROOT, args.stage)
