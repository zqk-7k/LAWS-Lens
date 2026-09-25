#!/usr/bin/env python3
"""Post-freeze diagnostics only; never used to choose models or coefficients."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_evaluate_20260912 as ev
import o4b_hl_nso_inference_20260912 as inference
import o4b_hl_nso_multiscale_training_20260912 as models

SEEDS = (2026091221, 2026091222, 2026091223)
PRIMARY = 'NEW-SCORE-ONLY-POSITIVE-CANDIDATE'


def effective_rank(z):
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False)).clip(0)
    if eig.sum() <= 0:
        return 0., 0.
    p = eig/eig.sum(); p = p[p > 0]
    return float(np.exp(-np.sum(p*np.log(p)))), float(1/np.sum(p*p))


def injection(root):
    inference.guard(root, 'test')
    marker = root/'contracts/POSTFREEZE_DIAGNOSTICS_COMPLETE.json'
    if marker.exists():
        return
    sources = pd.read_parquet(root/'tables/SOURCE_SPLIT_v2.parquet')
    sources = sources.set_index(['family', 'source_index'])
    low, _, _, _ = models.initialize(root)
    mass_grid = np.exp(low.old.CENTERS)
    stratified, ood, correlations, embedding, mass_summary = [], [], [], [], []
    out = root/'results/diagnostics'; out.mkdir(exist_ok=True)
    for seed in SEEDS:
        data = root/f'results/injection/seed_{seed}'
        events = pd.read_parquet(data/'events.parquet')
        events['snr_bin'] = pd.cut(events.target_network_snr,
            [0, 8, 10, 12, 20, 60, np.inf], right=False,
            labels=['below8', '8-10', '10-12', '12-20', '20-60', '60plus']).astype(str)
        rows = sources.loc[list(zip(events.family, events.source_index))]
        m1, m2 = rows.mass_1_detector.to_numpy(), rows.mass_2_detector.to_numpy()
        truth = (m1*m2)**.6/(m1+m2)**.2
        if not np.isfinite(truth).all() or (truth <= 0).any():
            raise RuntimeError('Missing physical source truth for diagnostic')
        for path in data.glob('*_query_ranks.parquet'):
            f = pd.read_parquet(path).merge(events[['event_uid', 'snr_bin']],
                                            on='event_uid', validate='one_to_one')
            for column in ('family', 'snr_bin'):
                for value, part in f.groupby(column):
                    stratified.append({'seed': seed, 'method': f.method.iloc[0],
                        'stratum': column, 'value': value, 'n_queries': len(part),
                        'n_systems_represented': part.source_id.nunique(),
                        **{f'R{k}': float((part['rank'] <= k).mean()) for k in (1, 5, 10, 50)},
                        'median_rank': float(part['rank'].median()),
                        'candidate_catalog_unchanged': True})
        f = pd.read_parquet(data/'all_pair_scores.parquet')
        for population, mask in [('companion', f.is_true_pair), ('noncompanion', ~f.is_true_pair)]:
            part = f.loc[mask]
            ood.append({'seed': seed, 'population': population, 'n_pairs': len(part),
                **{key+'_fraction': float(part[key].mean()) for key in
                   ('ordered_ood', 'joint_ood', 'OMC_calibration_ood', 'joint_calibration_ood')},
                **{key+'_negative_fraction': float((part[key] < 0).mean()) for key in
                   ('FRT_penalty', 'OMC_penalty', 'joint_penalty')},
                **{key+'_at_cap_fraction': float((part[key].abs() >= 4-1e-10).mean())
                   for key in ('OMC_increment', 'joint_increment')}})
            columns = ['waveform_score', 'time_score', 'sky_raw_log_bf', 'joint_BC']
            corr = part[columns].corr(method='spearman')
            for a in columns:
                for b in columns:
                    correlations.append({'seed': seed, 'population': population,
                        'channel_i': a, 'channel_j': b, 'spearman': corr.loc[a, b],
                        'no_independent_pair_p_value': True})
        pred = root/f'predictions_o4b/seed_{seed}/test'
        for name in ('short', 'rnc'):
            z = np.load(pred/f'{name}.npz')['z']
            erank, participation = effective_rank(z)
            embedding.append({'seed': seed, 'component': name, 'events': len(z),
                'dimensions': z.shape[1], 'entropy_effective_rank': erank,
                'participation_rank': participation,
                'norm_min': float(np.linalg.norm(z, axis=1).min()),
                'norm_max': float(np.linalg.norm(z, axis=1).max())})
        for name in ('ordered', 'multirate'):
            p = np.load(pred/f'{name}.npz')['p']
            cdf = p.cumsum(1)
            indices = {q: (cdf >= q).argmax(1) for q in (.05, .25, .5, .75, .95)}
            median = mass_grid[indices[.5]]
            event = events[['event_uid', 'global_source_id', 'family', 'snr_bin']].copy()
            event['Mc_detector_truth'], event['Mc_predicted_median'] = truth, median
            event['relative_error'] = (median-truth)/truth
            event['coverage_50'] = (truth >= mass_grid[indices[.25]]) & (truth <= mass_grid[indices[.75]])
            event['coverage_90'] = (truth >= mass_grid[indices[.05]]) & (truth <= mass_grid[indices[.95]])
            event['truth_outside_mass_grid'] = (truth < mass_grid[0]) | (truth > mass_grid[-1])
            event.to_parquet(out/f'{seed}_{name}_predicted_mass_diagnostics.parquet', index=False)
            for column in ('all', 'family', 'snr_bin'):
                parts = [('all', event)] if column == 'all' else event.groupby(column)
                for value, part in parts:
                    mass_summary.append({'seed': seed, 'component': name, 'stratum': column, 'value': value,
                        'n_events': len(part), 'median_absolute_relative_error': float(part.relative_error.abs().median()),
                        'median_relative_bias': float(part.relative_error.median()),
                        'coverage50': float(part.coverage_50.mean()), 'coverage90': float(part.coverage_90.mean()),
                        'out_of_grid_fraction': float(part.truth_outside_mass_grid.mean()),
                        'network_predictive_intervals_not_Bayesian_PE': True})
    for name, records in [('retrieval_by_family_and_SNR', stratified), ('waveform_OOD_and_tail_audit', ood),
                          ('channel_correlations', correlations), ('embedding_effective_rank', embedding),
                          ('predicted_mass_calibration', mass_summary)]:
        ev.csv(root/f'tables/{name}.csv', pd.DataFrame(records))
    s.write(marker, {'utc': s.now(), 'selection_feedback': False,
                    'SNR_stratification_keeps_full_candidate_catalog': True,
                    'mass_intervals_are_network_predictions_not_public_PE': True})


def real_figures(root):
    marker = root/'contracts/REAL_TOP10_FIGURES_COMPLETE.json'
    if marker.exists():
        return
    if not (root/'contracts/PE_AUDIT_COMPLETE.json').exists():
        return
    top = pd.read_parquet(root/f'results/PE_audit/{PRIMARY}/all_pairs_with_PE.parquet').head(10)
    events = pd.read_parquet(root/'inference_inputs/real/events.parquet')
    raw = np.load(root/'inference_inputs/real/raw2s.npy', mmap_mode='r')
    lookup = {name: i for i, name in enumerate(events.event_uid)}
    out = root/'figures'; out.mkdir(exist_ok=True)
    time = np.arange(4096)/2048-1.75
    with PdfPages(out/'fig_real_top10_H1L1_waveforms_and_PE.pdf') as pdf:
        for row in top.itertuples():
            fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), gridspec_kw={'height_ratios': [2, 2, 1]})
            for detector, ax in enumerate(axes[:2]):
                for name, color in [(row.event_i, '#256d85'), (row.event_j, '#9c3f48')]:
                    ax.plot(time, raw[lookup[name], detector], lw=.6, alpha=.8, label=name, color=color)
                ax.set_xlim(-1.75, .25); ax.set_ylabel(['H1', 'L1'][detector]+' processed strain')
                ax.spines[['top', 'right']].set_visible(False)
            axes[0].legend(frameon=False, fontsize=8)
            axes[0].set_title(f'Consensus rank {row.consensus_rank}: {row.event_i} / {row.event_j}', fontsize=10)
            axes[1].set_xlabel('Seconds relative to catalog coalescence')
            bc = [row.BC_Mc, row.BC_q, row.BC_chi_eff, row.BC_apparent_distance]
            axes[2].bar(['Detector Mc', 'Mass ratio q', 'Effective spin', 'Apparent distance'], bc, color='#256d85')
            axes[2].set_ylim(0, 1); axes[2].set_ylabel('Marginal PE BC')
            fig.text(.05, .01, f'Preprocessed, scaled model inputs, not physical amplitude or best-fit waveform. Intrinsic Dmax={row.Dmax:.2f}; no lensing confirmation.', fontsize=8)
            fig.tight_layout(rect=[0, .03, 1, 1]); pdf.savefig(fig)
            if row.consensus_rank == 1:
                fig.savefig(out/'fig_real_rank1_waveform_PE.png', dpi=160)
            plt.close(fig)
    matrix = top[['BC_Mc', 'BC_q', 'BC_chi_eff', 'BC_apparent_distance']].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(4), ['Detector Mc', 'q', 'Effective spin', 'Apparent distance'])
    ax.set_yticks(range(len(top)), [f'Rank {x}' for x in top.consensus_rank])
    for i in range(len(top)):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='white' if matrix[i,j]<.5 else 'black')
    fig.colorbar(im, ax=ax, label='Descriptive marginal PE BC'); fig.tight_layout()
    fig.savefig(out/'fig_real_top10_PE_heatmap.pdf'); fig.savefig(out/'fig_real_top10_PE_heatmap.png', dpi=180); plt.close(fig)
    s.write(marker, {'utc': s.now(), 'no_parameter_or_rank_changes': True,
                    'plotted_strain': 'same whitened/scaled H1L1 model inputs, not unprocessed physical amplitude'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); injection(a.root); real_figures(a.root)
