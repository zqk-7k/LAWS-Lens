#!/usr/bin/env python3
"""Public single-event PE audit, strictly downstream of frozen real ranks.

BC and standardized posterior-center distance are descriptive marginal tests,
not shared-source evidence or lensing probabilities. No Hanabi is run here.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_evaluate_20260912 as evaluate


def posterior(path, group):
    with h5py.File(path, 'r') as f:
        key = group.rstrip('/')+'/posterior_samples'
        if key not in f:
            raise RuntimeError('Frozen posterior group missing: '+key)
        dataset = f[key]
        if isinstance(dataset, h5py.Dataset) and dataset.dtype.names:
            names = set(dataset.dtype.names)
            def get(*options):
                for name in options:
                    if name in names: return np.asarray(dataset[name], float)
                raise KeyError(options)
        elif isinstance(dataset, h5py.Group):
            names = set(dataset.keys())
            def get(*options):
                for name in options:
                    if name in names: return np.asarray(dataset[name], float)
                raise KeyError(options)
        else:
            raise RuntimeError('Unsupported public posterior schema')
        m1, m2 = get('mass_1', 'mass1'), get('mass_2', 'mass2')
        m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
        mc = (m1*m2)**.6/(m1+m2)**.2
        q = m2/m1
        chi = get('chi_eff', 'effective_spin')
        distance = get('luminosity_distance', 'luminosity_distance_mpc')
        if 'chirp_mass' in names:
            named = get('chirp_mass')
            if not np.allclose(mc, named, rtol=1e-4, atol=1e-6, equal_nan=True):
                raise RuntimeError('Detector-frame chirp-mass field does not match detector masses')
    values = dict(Mc=mc, q=q, chi_eff=chi, apparent_distance=distance)
    mask = np.isfinite(np.column_stack(list(values.values()))).all(1)
    mask &= (mc > 0) & (q > 0) & (q <= 1) & (abs(chi) <= 1) & (distance > 0)
    if mask.mean() < .99 or mask.sum() < 500:
        raise RuntimeError('Invalid/insufficient posterior samples')
    return {k: v[mask] for k, v in values.items()}, {'rows': len(mask), 'valid_rows': int(mask.sum()),
                                                  'invalid_fraction': float(1-mask.mean()), 'exact_group': group,
                                                  'mass_frame': 'detector, recomputed from mass_1,mass_2'}


def density(samples, grid, bounds=None):
    # Deterministic thinning limits KDE cost without selecting posterior modes.
    take = np.linspace(0, len(samples)-1, min(len(samples), 4096)).round().astype(int)
    x = samples[take]
    if np.std(x) <= 0:
        raise RuntimeError('Degenerate public posterior')
    kde = gaussian_kde(x)
    y = kde(grid)
    if bounds is not None:
        low, high = bounds
        y += kde(2*low-grid)+kde(2*high-grid)
    mass = np.maximum(y, 0.)
    if not np.isfinite(mass).all() or mass.sum() <= 0:
        raise RuntimeError('PE KDE invalid')
    return mass/mass.sum()


def run(root):
    if not (root/'contracts/REAL_RANKINGS_COMPLETE.json').exists():
        raise RuntimeError('Real ranking must be frozen first')
    marker = root/'contracts/PE_AUDIT_COMPLETE.json'
    if marker.exists():
        return
    status = json.loads((root/'PE_DOWNLOAD_STATUS.json').read_text())
    if status.get('state') != 'PE_INPUTS_VERIFIED' or status.get('verified') != 86:
        raise RuntimeError('All86 original PE files must pass publisher verification')
    manifest = pd.read_csv(root/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830/data/event_manifest.csv')
    restored = {r['event_name']: r for r in json.loads((root/'manifests/RESTORE_PE_ARIA2.json').read_text())}
    samples, audits, summaries = {}, [], []
    for event in manifest.itertuples():
        record = restored[event.event_name]; path = Path(record['path'])
        if s.sha(path) != record['sha256']:
            raise RuntimeError('Verified public PE input changed')
        values, audit = posterior(path, event.pe_result_samples_key)
        samples[event.event_name] = values
        audits.append({'event': event.event_name, 'path': str(path), 'sha256': record['sha256'], **audit})
        for name, value in values.items():
            summaries.append({'event': event.event_name, 'parameter': name, 'median': float(np.median(value)),
                              'mean': float(np.mean(value)), 'std': float(np.std(value, ddof=1)),
                              'q05': float(np.quantile(value, .05)), 'q95': float(np.quantile(value, .95)), 'n': len(value)})
    out = root/'results/PE_audit'; out.mkdir(parents=True, exist_ok=True)
    evaluate.csv(out/'posterior_event_summary.csv', pd.DataFrame(summaries))
    s.write(out/'POSTERIOR_GROUP_AND_INPUT_MANIFEST.json', audits)
    # Keep endpoint summaries aligned with the canonical, sorted pair key.
    event_names = sorted(manifest.event_name.tolist()); n = len(event_names)
    ii, jj = np.triu_indices(n, 1)
    pairs = pd.DataFrame({'pair_key': ['--'.join(sorted((event_names[i], event_names[j]))) for i, j in zip(ii, jj)]})
    numerical = []
    for parameter in ('Mc', 'q', 'chi_eff', 'apparent_distance'):
        values = [samples[name][parameter] for name in event_names]
        med = np.array([np.median(v) for v in values]); sd = np.array([np.std(v, ddof=1) for v in values])
        transformed = [np.log(v) for v in values] if parameter in ('Mc', 'apparent_distance') else values
        bounds = (0., 1.) if parameter == 'q' else (-1., 1.) if parameter == 'chi_eff' else None
        if bounds is None:
            lo, hi = min(v.min() for v in transformed), max(v.max() for v in transformed)
            margin = .05*(hi-lo); lo -= margin; hi += margin
        else:
            lo, hi = bounds
        grid = np.linspace(lo, hi, 4096)
        masses = np.stack([density(v, grid, bounds) for v in transformed])
        bc = (np.sqrt(masses)@np.sqrt(masses).T)[ii, jj].clip(0, 1)
        overlap = np.array([np.minimum(masses[i], masses[j]).sum() for i, j in zip(ii, jj)])
        pairs['BC_'+parameter] = bc; pairs['O_'+parameter] = overlap
        pairs['D_'+parameter] = abs(med[ii]-med[jj])/np.sqrt(sd[ii]**2+sd[jj]**2)
        for end, indexes in [('i', ii), ('j', jj)]:
            pairs[f'{parameter}_median_{end}'] = med[indexes]
        np.savez_compressed(out/f'{parameter}_marginal_density.npz', event_names=event_names, grid=grid,
                            probability_mass=masses, coordinate='log' if parameter in ('Mc', 'apparent_distance') else 'linear')
        numerical.append({'parameter': parameter, 'max_normalization_error': float(abs(masses.sum(1)-1).max()),
                          'grid_points': 4096, 'KDE_max_samples_per_event': 4096,
                          'BC_invariant_coordinate_change': 'log coordinate with correctly transformed density; common grid probability mass'})
    pairs['Dmax'] = pairs[['D_Mc', 'D_q', 'D_chi_eff']].max(axis=1)
    pairs['Dmax_le3'] = pairs.Dmax <= 3
    pairs['official_PO_FPP'] = np.nan; pairs['official_ML_or_Phazap_FPP'] = np.nan
    pairs['official_frontend_overlap'] = pd.NA; pairs['public_Hanabi_overlap'] = pd.NA
    pairs['official_stage'] = 'NO_VERIFIED_O4B_LENSING_PAIR_TABLE'
    pairs['Hanabi_conclusion'] = 'NOT_RUN_IN_THIS_EXPERIMENT; NO_VERIFIED_PUBLIC_O4B_PAIR_RESULT'
    pairs.to_parquet(out/'all_pair_PE_and_official_status.parquet', index=False)
    budgets, correlations = [], []
    ranks = root/'results/real/consensus'
    for path in ranks.glob('*_all_pairs.parquet'):
        original_hash = s.sha(path)
        frame = pd.read_parquet(path)
        merged = frame.drop(columns=['PE_audit_status']).merge(pairs, on='pair_key', validate='one_to_one', how='left')
        if merged.BC_Mc.isna().any():
            raise RuntimeError('Incomplete public PE pair join')
        merged['PE_audit_status'] = 'DESCRIPTIVE_SINGLE_EVENT_MARGINALS_ONLY'
        method = merged.method.iloc[0]
        dest = out/method; dest.mkdir(exist_ok=True)
        merged.to_parquet(dest/'all_pairs_with_PE.parquet', index=False)
        evaluate.csv(dest/'all_pairs_with_PE.csv', merged)
        for b in (10, 20, 50, 100):
            top = merged.head(b)
            evaluate.csv(dest/f'top{b}_with_PE.csv', top)
            budgets.append({'method': method, 'budget': b, 'BC_Mc_ge05': int((top.BC_Mc >= .5).sum()),
                            'D_Mc_le3': int((top.D_Mc <= 3).sum()), 'Dmax_le3': int(top.Dmax_le3.sum()),
                            'median_BC_Mc': float(top.BC_Mc.median()),
                            'catastrophic_Mc': int(((top.BC_Mc < .1) | (top.D_Mc > 5)).sum()),
                            'official_pair_FPP_coverage': 0, 'official_overlap_count': pd.NA,
                            'Hanabi_overlap_count': pd.NA, 'missing_is_not_zero_overlap': True})
        for name in ('waveform_score_mean', 'final_score_mean'):
            correlations.append({'method': method, 'score': name,
                                  'Spearman_BC_Mc': float(spearmanr(merged[name], merged.BC_Mc).statistic),
                                  'Spearman_negative_D_Mc': float(spearmanr(merged[name], -merged.D_Mc).statistic),
                                  'pair_independence_pvalue_not_reported': True})
        if s.sha(path) != original_hash:
            raise RuntimeError('PE audit modified frozen real ranks')
    evaluate.csv(root/'tables/real_PE_and_official_budget_summary.csv', pd.DataFrame(budgets))
    evaluate.csv(root/'tables/real_waveform_PE_correlations.csv', pd.DataFrame(correlations))
    s.write(marker, {'utc': s.now(), 'events': 86, 'pairs': len(pairs), 'numerical': numerical,
                     'no_PE_or_official_feedback_to_scores': True, 'no_Hanabi_run': True,
                     'distance_excluded_from_intrinsic_Dmax': True,
                     'official_missing_not_fabricated': True,
                     'scope': 'PE consistency is neither a lens label nor a joint Bayes factor'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
