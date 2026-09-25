#!/usr/bin/env python3
"""Empirically calibrated profile predictions, never called full PE."""
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
import mcwf_subgrid_density_20260909 as d
n, r = d.n, d.reliability
PROFILE = P / 'results/mcwf_nodup_lowmass_profile_07_20260909T082227Z'
ROOT = None


def parent_predictions(dep):
    ps = []
    for slot in n.t.MODEL_SLOTS:
        with np.load(r.INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz') as a:
            ps.append(a['p'])
    return ps


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for folder in ('contracts', 'calibration', 'predictions', 'configs', 'tables', 'audit', 'reports',
                   'scripts', 'logs', 'manifest', 'results', 'figures', 'cache', 'profile_events'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {'id': 'MCWF-NODUP-PROFILE-PREDICTIVE-08',
        'UTC': n.utc(), 'status': n.STATUS, 'goal_achieved': False, 'same_both_runs': True,
        'adaptive_simulation_development': 'O3 rawprofilevalidation inspected:medianerror improvedbutnewcatastrophicpointoutliers. Raw07 isNOT retroactivelypassed;thisisaseparatequalitycontrolledpredictor.',
        'profile_source': str(PROFILE), 'parent': str(d.LOW),
        'per_event_activation': 'Own ensemble waveformmassmedian<=15Msun;doesnot dependon companionlabel,eventname,PEorofficialtable. Ifonlyoneimageeligibleonlythateventchanges.',
        'quality': 'Bestvaluehasconvergedoptimizer,notwithin1e-3ofq[.25,1]orchi[-.8,.8]boundary,profilemass insideparentcentralprobabilityregion. FailedfitskeepoldNEWmassdensity,notoldencoderregression.',
        'quality_grid': [.90, .95, .99], 'boundary_rationale': 'Ten timestheoriginaloptimizerparameterstoppingtolerance;boundarysolutionsaremisspecificationdiagnosticsnotreliablecurvature.',
        'probability': 'Studentt residual predictive density fortruthlogMc-profilelogMc,normalized onlog5..log200;CDFintegratedbinmasses. ThisisempiricalpredictionnotphysicalPEorfullstrainBayesfactor.',
        'errorfit': 'source/noisehashfold0,perindependentsourcetotalweight1;locationandscalemaximumlikelihood fordf3,5,10,30;location[-.1,.1],scale[.0002,.2]numericalbounds',
        'selection': 'Onlyfold1properNLL;requireNLLandabsoluteerrornotworse,noincreasein>10percentpointfailures,and90percentcoverageblockCIincludes.9. IfnoneHOLDthisarm;no realranking.',
        'uncertainty': '5000sourcebootstrapdraws,twoimageskepttogether;notpostselectionconfidenceorblindconfirmation',
        'future_score': 'ReplacemassmarginalinONEnewjointdensity;retainfrozenconditionaleta/chi. RecalibratejointBConsimulationdata;nooldMc/q terms,no totalalpha,time/skyunchanged.',
        'frozen': ['oldencoder', 'time_score', 'sky_raw_log_bf', 'outerweights', 'scope', 'oldrankings', 'paper'],
        'no_real_outcome_selection': True,
        'limitations': 'Alignedspinprofilecanmissprecession/highermodes;hardconditionalfallbackandfiniteerrorcalibrationneedexplicitaudits;no claimpublicPEuncertaintyachieved.',
        'reference': 'https://arxiv.org/abs/gr-qc/9402014'})
    shutil.copy2(__file__, ROOT / 'scripts/profile_predictive_calibration.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.read_csv(PROFILE / 'manifest/INPUT_SHA256.csv'))


def quality_mask(parent, rows, coverage):
    summary = r.mass_summary(parent)
    mask = np.isfinite(summary[:, 0]) & (np.exp(summary[:, 0]) <= 15.)
    active = np.zeros(len(parent), bool)
    centers = np.full(len(parent), np.nan)
    for row in rows:
        idx = int(row['row_index'])
        if not mask[idx]:
            continue
        value = float(row['logmc'])
        centers[idx] = value
        lo, hi = np.interp([(1 - coverage) / 2, (1 + coverage) / 2],
                           np.r_[0., parent[idx].cumsum()], d.EDGES)
        active[idx] = bool(row['best_run_converged'] and .251 < row['q'] < .999 and
                           -.799 < row['chieff_equal'] < .799 and lo <= value <= hi)
    return active, centers


def normalized_logpdf(truth, centers, spec):
    loc, scale, df = spec['location'], spec['scale'], spec['df']
    normalization = student.cdf((d.EDGES[-1] - centers - loc) / scale, df) - student.cdf((d.EDGES[0] - centers - loc) / scale, df)
    return student.logpdf((truth - centers - loc) / scale, df) - np.log(scale) - np.log(normalization.clip(1e-300))


def density(centers, spec, edges=None):
    edges = d.EDGES if edges is None else edges
    cdf = student.cdf((edges[None] - centers[:, None] - spec['location']) / spec['scale'], spec['df'])
    prob = np.diff(cdf, axis=1).clip(0.)
    prob /= prob.sum(1, keepdims=True)
    if not np.isfinite(prob).all() or abs(prob.sum(1) - 1).max() > 1e-12:
        raise RuntimeError('Invalid predictive probability mass')
    return prob


def fit_error(truth, centers, groups, df):
    weight = pd.Series(groups).map(1 / pd.Series(groups).value_counts()).to_numpy()
    weight /= weight.sum()
    residual = truth - centers
    start = [float(np.clip(np.median(residual), -.099, .099)), float(np.log(np.clip(np.median(abs(residual - np.median(residual))) * 1.4826, .001, .1)))]
    def objective(params):
        spec = {'location': params[0], 'scale': np.exp(params[1]), 'df': df}
        return -float(np.sum(weight * normalized_logpdf(truth, centers, spec)))
    opt = minimize(objective, start, method='L-BFGS-B', bounds=[(-.1, .1), (np.log(.0002), np.log(.2))])
    if not opt.success:
        raise RuntimeError('Residual fit failed: ' + opt.message)
    return {'location': float(opt.x[0]), 'scale': float(np.exp(opt.x[1])), 'df': int(df),
            'fit_sources': len(np.unique(groups)), 'fit_events': len(truth), 'NLL': float(opt.fun)}


def calibrate():
    if not (PROFILE / 'contracts/PROFILE_DEVELOPMENT_COMPLETE.json').exists():
        raise RuntimeError('Waitforcompletepredecessorpilot')
    selected, all_rows, curves = {}, [], []
    for dep in n.DEPS:
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment'] = dep
        fold = r.source_fold(meta)
        truth = np.log(meta.mc_det.to_numpy())
        groups = meta.source_uid.to_numpy(str)
        parent = np.mean(parent_predictions(dep), axis=0)
        rows = [json.loads(path.read_text()) for path in sorted((PROFILE / 'events').glob(dep + '_*.json'))]
        bins = np.searchsorted(d.EDGES, truth, side='right').clip(1, 512) - 1
        parent_logpdf = np.log(parent[np.arange(len(parent)), bins].clip(1e-300)) - np.log(np.diff(d.EDGES)[bins])
        parent_median = r.mass_summary(parent)[:, 0]
        parent_nll = -float(parent_logpdf[fold == 1].mean())
        tune_ids = np.flatnonzero(fold == 1)
        gu, gn = pd.factorize(groups[tune_ids], sort=True)
        rng = np.random.default_rng(2026090999)
        draws = rng.integers(len(gn), size=(5000, len(gn)))
        trials = []
        for coverage in (.9, .95, .99):
            active, centers = quality_mask(parent, rows, coverage)
            fit = active & (fold == 0)
            if len(np.unique(groups[fit])) < 20:
                continue
            for df in (3, 5, 10, 30):
                spec = fit_error(truth[fit], centers[fit], groups[fit], df)
                spec['parent_coverage'] = coverage
                pp = parent.copy()
                pp[active] = density(centers[active], spec)
                med = r.mass_summary(pp)[:, 0]
                logpdf = parent_logpdf.copy()
                logpdf[active] = normalized_logpdf(truth[active], centers[active], spec)
                pit = np.array([np.interp(t, d.EDGES, np.r_[0., p.cumsum()]) for t, p in zip(truth, pp)])
                hit = ((pit[tune_ids] >= .05) & (pit[tune_ids] <= .95)).astype(float)
                per_source = np.bincount(gu, weights=hit) / np.bincount(gu)
                lo, hi = np.quantile(per_source[draws].mean(1), [.025, .975])
                tune = fold == 1
                parent_error, new_error = abs(parent_median[tune] - truth[tune]), abs(med[tune] - truth[tune])
                row = {'deployment': dep, 'parent_coverage': coverage, 'df': df, 'location': spec['location'],
                    'scale': spec['scale'], 'fit_sources': spec['fit_sources'], 'tune_sources': len(gn),
                    'NLL': -float(logpdf[tune].mean()), 'parent_NLL': parent_nll,
                    'MAE': float(new_error.mean()), 'parent_MAE': float(parent_error.mean()),
                    'catastrophic10percent': int((new_error > np.log(1.1)).sum()),
                    'parent_catastrophic10percent': int((parent_error > np.log(1.1)).sum()),
                    'coverage90': float(hit.mean()), 'coverage90_ci_low': float(lo), 'coverage90_ci_high': float(hi),
                    'activation_fraction': float(active.mean())}
                row['PASS'] = bool(row['NLL'] <= parent_nll and row['MAE'] <= row['parent_MAE'] and
                    row['catastrophic10percent'] <= row['parent_catastrophic10percent'] and lo <= .9 <= hi)
                trials.append((row, spec))
                all_rows.append(row)
        passing = [(a, b) for a, b in trials if a['PASS']]
        candidates = passing or trials
        if not candidates:
            selected[dep] = {'pass': False, 'reason': 'Insufficientindependentfit support'}
            continue
        row, spec = min(candidates, key=lambda pair: (pair[0]['NLL'], abs(pair[0]['parent_coverage'] - .95), -pair[0]['df']))
        selected[dep] = {'pass': bool(passing), 'selection': row, 'spec': spec}
        active, centers = quality_mask(parent, rows, spec['parent_coverage'])
        pp = parent.copy()
        pp[active] = density(centers[active], spec)
        np.savez_compressed(ROOT / f'predictions/{dep}_development_ENSEMBLE.npz', p=pp, parent_p=parent,
                            active=active, profile_centers=centers, truth=truth, fold=fold)
        curves.append({'deployment': dep, **row})
        print('PROFILE_PREDICTIVE_SELECTION', dep, selected[dep], flush=True)
    n.write_csv(ROOT / 'tables/PROFILE_PREDICTIVE_GRID.csv', all_rows)
    n.write_csv(ROOT / 'tables/PROFILE_PREDICTIVE_SELECTION.csv', curves)
    n.write_json(ROOT / 'calibration/PROFILE_PREDICTIVE.json', selected)
    n.write_json(ROOT / 'contracts/PREDICTIVE_FROZEN.json', {'UTC': n.utc(),
        'sha256': n.sha(ROOT / 'calibration/PROFILE_PREDICTIVE.json'), 'both_runs_pass': all(v['pass'] for v in selected.values()),
        'no_test_or_real_used_for_selection': True, 'status': n.STATUS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate'), required=True)
    args = parser.parse_args()
    ROOT = args.root
    globals()[args.stage]()
