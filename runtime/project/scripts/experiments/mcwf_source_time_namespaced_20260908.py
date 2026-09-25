#!/usr/bin/env python3
"""Source-group time-prior audit and independent finite-population correction."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.integrate import trapezoid

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_temporal_response_20260908 as t
import mcwf_temporal_response_evaluate_20260908 as ev
dev, cf = t.dev, ev.cf
SOURCE = Path('/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR8/BBH_2.5PLUS_Any_Detected_SNR8_ImageParams.csv')
SOURCE_PARAMETERS = SOURCE.with_name(SOURCE.name.replace('ImageParams','SourceParams'))
ET_SOURCE_PARAMETERS = Path('/root/autodl-tmp/GW-LMC/ET/BBH/Any_Detected_SNR8/BBH_ET_Any_Detected_SNR8_SourceParams.csv')
MEMBERSHIP = P / 'results/time_snr_joint_evidence_phase05a_20260807_final/contract/global_system_split_membership_frozen.parquet'
V7 = P / 'results/real_noise_injection_v7_peak2s_formal_20260722'
KINDS = ('SOURCE-GROUP-TIME', 'SOURCE-GROUP-TIME-CONSERVATIVE')


def raw_prior():
    rows = []
    for r in pd.read_csv(SOURCE).itertuples():
        snr, delay = np.asarray(ast.literal_eval(r.img_snrs)), np.asarray(ast.literal_eval(r.img_delays_days))
        ids = np.flatnonzero(snr >= 8.)
        for k, i in enumerate(ids):
            for j in ids[k+1:]:
                dt = float(abs(delay[j] - delay[i]))
                if dt > 0 and np.isfinite(dt):
                    rows.append({'global_source_group_id': f'GW-LMC:2.5PLUS:source:{r.event_id}', 'source_id': int(r.event_id),
                                 'image_i': int(i), 'image_j': int(j), 'delay_days': dt})
    a = pd.DataFrame(rows)
    a['multiplicity'] = a.groupby('global_source_group_id').delay_days.transform('size')
    a['source_group_weight'] = 1. / a.multiplicity
    if not np.allclose(a.groupby('global_source_group_id').source_group_weight.sum(), 1.):
        raise RuntimeError('Source total weights not one')
    return a


def membership_audit(root):
    membership = pd.read_parquet(MEMBERSHIP)
    rows, plans = [], []
    forbidden = set()
    for dep in t.DEPS:
        for seed in t.SEEDS:
            lookup = membership[(membership.deployment == dep) & (membership.seed == seed)].set_index(['family', 'source_index'])
            if lookup.index.duplicated().any():
                raise RuntimeError('Ambiguous global ID mapping')
            for split in ('validation', 'test'):
                plan = dev.BASE.retained_event_plan(dep, seed, split)
                mapped = plan[['family', 'source_index']].drop_duplicates().join(
                    lookup[['global_source_group_id', 'global_lens_system_id', 'corrected_split']], on=['family', 'source_index'])
                if mapped.global_source_group_id.isna().any():
                    raise RuntimeError('Unmapped current source')
                mapped['global_source_group_id'] = mapped.global_source_group_id.str.replace('GW-LMC-source:', 'GW-LMC:ET:source:', regex=False)
                forbidden.update(mapped.global_source_group_id)
                mapped['deployment'], mapped['seed'], mapped['current_split'] = dep, seed, split
                plans.append(mapped)
                rows.append({'deployment': dep, 'seed': seed, 'split': split, 'systems': len(mapped),
                             'phase05_excluded_still_in_current': int((mapped.corrected_split == 'excluded').sum()),
                             'phase05_train_still_in_current': int((mapped.corrected_split == 'train').sum()),
                             'global_ids_complete': True})
    pd.concat(plans).to_parquet(root / 'tables/CURRENT_SOURCE_MEMBERSHIP.parquet', index=False)
    dev.csv_write(root / 'tables/CURRENT_SPLIT_AUDIT.csv', pd.DataFrame(rows))
    return forbidden


def cross_namespace_audit(root):
    a, b = pd.read_csv(SOURCE_PARAMETERS), pd.read_csv(ET_SOURCE_PARAMETERS)
    fields = ['m1_det','m2_det','m1_src','m2_src','ra','dec','a1','a2','tilt1','tilt2','theta_jn','psi']
    fields = [f for f in fields if f in a and f in b]
    if len(fields)<8:
        raise RuntimeError('Insufficient physical fingerprint fields')
    raw_rows = {'2p5': len(a), 'ET': len(b)}
    metadata_conflicts = []
    for name, frame in (('2p5', a), ('ET', b)):
        if (frame.groupby('event_id')[fields].nunique(dropna=False)>1).any().any():
            raise RuntimeError('Numeric source ID has conflicting physical parameters within '+name)
        # Redshift is audited separately: several ET duplicate rows have identical
        # masses/directions/spins but slightly different published redshift values.
        if 'z_source' in frame:
            for ident, part in frame.groupby('event_id'):
                if part.z_source.nunique(dropna=False)>1:
                    metadata_conflicts.append({'population':name,'event_id':int(ident),
                        'rows':len(part),'z_min':float(part.z_source.min()),'z_max':float(part.z_source.max()),
                        'z_from_mass_ratio':float(part.m1_det.iloc[0]/part.m1_src.iloc[0]-1.),
                        'identity_fields_equal':True,'redshift_used_in_time_score':False})
    dev.csv_write(root/'tables/SOURCE_REDSHIFT_METADATA_CONFLICTS.csv',pd.DataFrame(metadata_conflicts,
        columns=['population','event_id','rows','z_min','z_max','z_from_mass_ratio','identity_fields_equal','redshift_used_in_time_score']))
    a, b = a.drop_duplicates('event_id'), b.drop_duplicates('event_id')
    matched = a.merge(b,on='event_id',suffixes=('_2p5','_ET'),validate='one_to_one')
    same = np.ones(len(matched),bool)
    for field in fields:
        same &= np.isclose(matched[field+'_2p5'],matched[field+'_ET'],atol=1e-9,rtol=1e-10)
    def fingerprints(frame):
        values=frame[fields].round(8)
        return pd.util.hash_pandas_object(values,index=False).to_numpy(np.uint64)
    fa,fb=fingerprints(a),fingerprints(b)
    cross=np.intersect1d(fa,fb)
    if len(cross):
        raise RuntimeError('Cross-network physical matches need an explicit common-parent map before scoring')
    rows=[]
    for field in fields:
        diff=abs(matched[field+'_2p5']-matched[field+'_ET'])
        rows.append({'field':field,'same_numeric_id_rows':len(matched),'max_abs_difference':diff.max(),
                     'equal_within_tolerance':int(np.isclose(matched[field+'_2p5'],matched[field+'_ET'],atol=1e-9,rtol=1e-10).sum())})
    dev.csv_write(root/'tables/CROSS_NETWORK_ID_PHYSICAL_AUDIT.csv',pd.DataFrame(rows))
    audit={'2p5_sources':len(a),'ET_sources':len(b),'native_raw_rows':raw_rows,
           'core_identity_equal_duplicate_rows':{'2p5':raw_rows['2p5']-len(a),'ET':raw_rows['ET']-len(b)},
           'redshift_metadata_conflict_groups':len(metadata_conflicts),'same_numeric_event_ids':len(matched),
           'same_ID_and_all_physical_parameters':int(same.sum()),'cross_ID_matching_physical_fingerprints':len(cross),
           'fingerprint_fields':fields,'rounding_decimals':8,'raw_equal_ID_is_not_global_source_identity':True,
           'redshift_metadata_note':'Not silently corrected. Duplicate ET z_source conflicts retained in a separate table; neither published z_source nor an inferred replacement enters this time lookup. Identity uses the exactly matching mass, sky and spin fields.',
           'conclusion':'Do not exclude a 2.5PLUS system merely because an ET source has the same numeric event_id.',
           'upstream_phase05_ET_internal_grouping_not_changed':True}
    dev.json_write(root/'contracts/CROSS_NAMESPACE_IDENTITY_AUDIT.json',audit)


def interval_cumulative(x, schedule):
    start, end = schedule.start_gps.to_numpy(float), schedule.end_gps.to_numpy(float)
    if np.any(start[1:] < end[:-1]):
        raise RuntimeError('Exposure intervals overlap')
    cumulative = np.r_[0., np.cumsum(end - start)]
    idx = np.searchsorted(start, x, side='right') - 1
    safe = idx.clip(0, len(start) - 1)
    answer = cumulative[safe] + np.clip(x - start[safe], 0., end[safe] - start[safe])
    return np.where(idx < 0, 0., answer)


def exposure_acceptance(days, schedule):
    start, end, rate = [schedule[k].to_numpy(float) for k in ('start_gps', 'end_gps', 'weight_per_second')]
    denom = np.dot(end - start, rate)
    answer = []
    for d in np.atleast_1d(days):
        length = interval_cumulative(end + 86400 * d, schedule) - interval_cumulative(start + 86400 * d, schedule)
        answer.append(np.dot(rate, length) / denom)
    result = np.asarray(answer)
    if np.any(result < -1e-12) or np.any(result > 1 + 1e-12):
        raise RuntimeError('Invalid exposure acceptance')
    return result.clip(0, 1)


def choose_bandwidth(source):
    x = np.log10(source.delay_days.to_numpy(float))
    groups = source.global_source_group_id.to_numpy(str)
    w = source.source_group_weight.to_numpy(float)
    ids = np.unique(groups)
    def scale(a, weights, n):
        weights = weights / weights.sum()
        variance = np.dot(weights, (a - np.dot(weights, a))**2) * n / (n - 1)
        return np.sqrt(variance) * n**(-.2)
    rows = []
    for factor in (.5, 1., 2., 4.):
        fold = []
        for group in ids:
            fit = groups != group
            h = factor * scale(x[fit], w[fit], len(ids) - 1)
            delta = (x[~fit, None] - x[fit]) / h
            lp = logsumexp(-.5 * delta**2 + np.log(w[fit] / w[fit].sum()), axis=1) - np.log(h * np.sqrt(2 * np.pi))
            fold.append(float(np.average(lp, weights=w[~fit])))
        rows.append({'bandwidth_factor': factor, 'mean_leave_one_system_log_density': np.mean(fold),
                     'source_systems': len(ids), 'folds': fold})
    chosen = min(rows, key=lambda r: (-r['mean_leave_one_system_log_density'], abs(np.log(r['bandwidth_factor']))))
    return chosen['bandwidth_factor'] * scale(x, w, len(ids)), rows


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent new directory required')
    for name in ('contracts', 'tables', 'calibration', 'evaluation', 'reports', 'figures', 'scripts', 'logs', 'manifest'):
        (root / name).mkdir(parents=True)
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NAMESPACED-SOURCE-TIME-13', 'UTC': datetime.now(timezone.utc).isoformat(), 'status': t.STATUS,
        'goal_achieved': False, 'same_both_runs': True,
        'purpose': 'Audit and repair finite independent-population/time-prior provenance,not arbitrary reward shaping.',
        'raw_prior': str(SOURCE), 'membership': str(MEMBERSHIP),
        'source_unit': 'One total weight per independent source,uniform choice of detectable pair within source;not uniform all image-pairs. Use network-population namespaces and verify physical parameter fingerprints before testing global source overlap.',
        'invalid_previous': 'Round11 falsely merged numeric event_id across ET and2.5PLUS;its twelve alleged overlaps were false identifiers. Retain11asINVALID,do not treat its excluded22-system population as a valid correction. This round uses correctly namespaced34 external systems after physical identity audit.',
        'bandwidth': 'Weighted source-population log-delay variance with source-count correction;Scott factor based on independent sources,not bootstrap draws. Factor0.5/1/2/4 chosen by leave-one-source-out log predictive density. No injection test or real-candidate metric selection.',
        'signal_density': 'Smooth underlying external detectable-pair population first;then multiply exact acceptance of both arrivals in frozen H1L1 live intervals;normalize in log10 days.',
        'exposure': 'KEEP HISTORICAL schedule for this isolated finite-source comparison. O3 schedule includes O1/O2/O3a/O3b;this is an acknowledged mismatch to current official O3-only real scope,not a run-matched correction.',
        'background': 'Original frozen null density and lookup grid unchanged;no 2D SNR term.',
        'uncertainty': '2000 independent source-system bootstrap draws;all pairs from each source move together. Refit weighted population mixture at frozen bandwidth and renormalize exposure. Does not include cosmology/lens-family uncertainty.',
        'methods': list(KINDS),
        'conservative': 'Nearest point to zero in source-bootstrap95% pointwise logLR interval: lower bound ifpositive,upper bound ifnegative,zero ifintervalcontainszero. This is an uncertainty-discounted ranking score,not a Bayes factor.',
        'numerical_floor': '1e-300 only to keep logs finite;both signal and null zero/exposure impossible gives neutral0;no fitted score cap.',
        'weights': 'All original C-fixed per-seed W/T/S weights retained,no optimization based on PE or official overlap.',
        'frozen': ['OMC waveform score', 'all encoder models', 'sky_raw_log_bf', 'outer weights', 'scope', 'old results', 'paper'],
        'changed_channels': ['time'],
        'known_limits': 'Small external population;2.5PLUS SNR>=8 selection is not a full O3/O4a detector-response selection model;legacy time-scope mismatch remains;current O4a seed241 source1568 issue separately audited. Results exploratory,not independently confirmed.',
        'real': 'Adaptive development audit only. Real PE and official labels appended after density freeze,never used as score inputs.',
        'references': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html', 'https://github.com/LensedGW/GW-LMC']})
    original = t.protected() + [{'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size} for p in (SOURCE, MEMBERSHIP,SOURCE_PARAMETERS,ET_SOURCE_PARAMETERS)]
    for dep in t.DEPS:
        for name in ('time_delay_likelihood_ratio.json', 'h1l1_live_schedule.csv', 'time_delay_lensed_samples_days.npy'):
            f = V7 / dep / 'shared' / name
            original.append({'path': str(f), 'sha256': dev.sha(f), 'bytes': f.stat().st_size})
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(original))
    shutil.copy2(__file__, root / 'scripts/source_time_namespaced.py')
    dev.json_write(root / 'contracts/START_FREEZE.json', {'contract_sha256': dev.sha(root / 'contracts/ANALYSIS_CONTRACT.json'), 'code_sha256': dev.sha(Path(__file__))})
    cross_namespace_audit(root)
    forbidden = membership_audit(root)
    raw = raw_prior()
    raw['excluded_validation_or_test_source'] = raw.global_source_group_id.isin(forbidden)
    dev.csv_write(root / 'tables/EXTERNAL_PRIOR_ALL_PAIRS_AUDIT.csv', raw)
    source = raw[~raw.excluded_validation_or_test_source].copy()
    if source.global_source_group_id.nunique() < 15:
        raise RuntimeError('Insufficient independent external source support for this 1D pilot')
    h, cv = choose_bandwidth(source)
    dev.json_write(root / 'calibration/SOURCE_BANDWIDTH_LOO.json', {'bandwidth_log10_days': h, 'grid': cv})
    rows, configs, numerical = [], [], []
    x = np.log10(source.delay_days.to_numpy(float))
    groups = source.global_source_group_id.to_numpy(str)
    ids = np.unique(groups)
    weights = source.source_group_weight.to_numpy(float)
    rng = np.random.default_rng(202609920)
    bootstrap = rng.multinomial(len(ids), np.ones(len(ids)) / len(ids), size=2000)
    for dep in t.DEPS:
        shared = V7 / dep / 'shared'
        old = json.loads((shared / 'time_delay_likelihood_ratio.json').read_text())
        samples = np.load(shared / 'time_delay_lensed_samples_days.npy')
        _, counts = np.unique(samples, return_counts=True)
        schedule = pd.read_csv(shared / 'h1l1_live_schedule.csv').sort_values('start_gps')
        grid = np.asarray(old['log10_delay_grid'])
        acceptance = exposure_acceptance(10**grid, schedule)
        kernel = np.exp(-.5 * ((grid[:, None] - x) / h)**2) / (h * np.sqrt(2 * np.pi))
        group_kernel = np.stack([(kernel[:, groups == g] * weights[groups == g]).sum(1) for g in ids])
        population = group_kernel.mean(0)
        density = population * acceptance
        density /= trapezoid(density, grid)
        null = np.asarray(old['p_null_log10'])
        score = np.log(density.clip(1e-300)) - np.log(null.clip(1e-300))
        draws = (bootstrap @ group_kernel) / len(ids) * acceptance
        draws /= trapezoid(draws, grid, axis=1)[:, None]
        boot_score = np.log(draws.clip(1e-300)) - np.log(null.clip(1e-300))[None]
        low, high = np.quantile(boot_score, [.025, .975], axis=0)
        impossible = (acceptance == 0.) & (density == 0.)
        score[impossible], low[impossible], high[impossible] = 0., 0., 0.
        conservative = np.where(low > 0., low, np.where(high < 0., high, 0.))
        # Independent Monte Carlo verifies the analytic exposure operator.
        start, end, rate = [schedule[c].to_numpy(float) for c in ('start_gps', 'end_gps', 'weight_per_second')]
        pick = rng.choice(len(schedule), 200000, p=(end-start)*rate/np.dot(end-start, rate))
        first = start[pick] + rng.random(len(pick)) * (end[pick] - start[pick])
        for day in (0., .1, 1., 10., 100.):
            late = first + day * 86400
            idx = np.searchsorted(start, late, side='right') - 1
            good = (idx >= 0) & (late <= end[idx.clip(0, len(end)-1)])
            empirical = good.mean()
            exact = exposure_acceptance([day], schedule)[0]
            tol = 5 * np.sqrt(max(exact*(1-exact),1e-8)/len(pick)) + 1e-5
            if abs(empirical-exact) > tol:
                raise RuntimeError('Exact exposure differs from independent Monte Carlo')
            numerical.append({'deployment': dep, 'delay_days': day, 'exact_acceptance': exact, 'MC_acceptance': empirical, 'MC_n': len(pick), 'tolerance': tol, 'pass': True})
        detail = {'grid': grid, 'population_density': population, 'exposure_acceptance': acceptance,
                  'p_lens_log10': density, 'p_null_log10': null, 'logLR': score,
                  'bootstrap95_low': low, 'bootstrap95_high': high, 'conservative_logLR': conservative,
                  'independent_sources': len(ids), 'independent_raw_sources': raw.global_source_group_id.nunique(),
                  'source_rows': len(source), 'bandwidth_log10_days': h, 'bootstrap_system_draws': 2000,
                  'source_test_overlap': 0, 'heldout_or_real_selection': False, 'historical_schedule_runs': schedule.run.unique().tolist()}
        dev.json_write(root / f'calibration/{dep}.json', detail)
        dev.csv_write(root / f'tables/{dep}_TIME_LOOKUP_COMPARISON.csv', pd.DataFrame({'log10_days': grid, 'old_time_score': old['log_likelihood_ratio'], 'new_time_score': score, 'conservative_score': conservative, 'bootstrap95_low': low, 'bootstrap95_high': high, 'acceptance': acceptance}))
        rows.append({'deployment': dep, 'original_raw_pairs': len(raw), 'original_independent_sources': raw.global_source_group_id.nunique(),
                     'old_bootstrap_draws': len(samples), 'old_distinct_delays': len(counts), 'old_delay_neff': len(samples)**2/np.square(counts.astype(float)).sum(),
                     'old_bandwidth_nominal_N': 30000, 'excluded_current_validation_test_sources': raw[raw.excluded_validation_or_test_source].global_source_group_id.nunique(),
                     'remaining_independent_sources': len(ids), 'new_bandwidth_log10_days': h,
                     'all_source_total_weight_one': bool(np.allclose(source.groupby('global_source_group_id').source_group_weight.sum(),1.)),
                     'shared_old_prior_sources_with_current_validation_test': bool(raw.excluded_validation_or_test_source.any())})
        for kind in KINDS:
            for seed in t.SEEDS:
                configs.append({'method': kind, 'deployment': dep, 'seed': seed, 'gamma': 0., 'beta': 0., 'weights': cf.frozen_weights(dep, seed).tolist(), 'calibration_path': f'calibration/{dep}.json'})
    dev.csv_write(root / 'tables/TIME_PROVENANCE_AUDIT.csv', pd.DataFrame(rows))
    dev.csv_write(root / 'tables/EXPOSURE_OPERATOR_TESTS.csv', pd.DataFrame(numerical))
    dev.json_write(root / 'calibration/SELECTED.json', configs)
    dev.json_write(root / 'contracts/INTEGRATION_FROZEN.json', {'file': 'calibration/SELECTED.json', 'sha256': dev.sha(root / 'calibration/SELECTED.json'),
                   'UTC': datetime.now(timezone.utc).isoformat(), 'real_or_test_metrics_used': False,
                   'calibration_hashes': {dep: dev.sha(root / f'calibration/{dep}.json') for dep in t.DEPS}})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    initialize(args.root)
