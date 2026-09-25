#!/usr/bin/env python3
"""Read-only experiment ledger and corrected per-seed external audits."""
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
from scipy.stats import spearmanr

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_nodup_reliability_20260909 as r
n = r.n
ROUNDS = [
    (2, 'mcwf_nodup_mass_reliability_02_20260909T154100Z', 'Joint predictive reliability calibration'),
    (3, 'mcwf_nodup_subgrid_density_03_20260909T161600Z', 'Continuous mixture mass density'),
    (4, 'mcwf_nodup_latent_density_04_20260909T075600Z', 'Frozen base embedding conditioned mass density'),
    (5, 'mcwf_nodup_pair_response_05_20260909T080703Z', 'Symmetric waveform-response pair verifier'),
    (6, 'mcwf_nodup_large_density_06_20260909T081554Z', 'Expanded12288parent mass-density training'),
    (7, 'mcwf_nodup_lowmass_profile_07_20260909T082227Z', 'Low-frequency16s physical profile development'),
    (8, 'mcwf_nodup_profile_predictive_08_20260909T083816Z', 'Student-t predictive error calibration'),
    (9, 'mcwf_nodup_trusted_local_09_20260909T084714Z', 'Local trustworthy-profile quality controls'),
    (10, 'mcwf_nodup_mode_profile_10_20260909T085830Z', 'Multi-mode ambiguity safeguard'),
    (11, 'mcwf_nodup_conditional_profile_11_20260909T090510Z', 'INVALID inactive-fallback coefficient contamination'),
    (12, 'mcwf_nodup_isolated_profile_12_20260909T092250Z', 'Corrected exact immutable inactive fallback'),
    (13, 'mcwf_nodup_hierarchical_profile_13_20260909T093520Z', 'One intrinsic classifier, separated conditional BC features'),
    (14, 'mcwf_nodup_single_waveform_14_20260909T095000Z', 'One joint waveform classifier, no added cosine LR'),
    (15, 'mcwf_nodup_null_scale_15_20260909T095600Z', 'Fit-null quantile score alignment'),
    (16, 'mcwf_nodup_heteroscedastic_profile_16_20260909T100700Z', 'Observed-curvature predictive covariate'),
    (17, 'mcwf_nodup_phase_match_17_20260909T105358Z', 'PyCBC-verified point-template physical phase match'),
    (18, 'mcwf_nodup_expected_information_18_20260909T110338Z', 'Expected derivative information predictive calibration'),
    (19, 'mcwf_nodup_information_integrated_19_20260909T110744Z', 'Information-conditioned density integrated scoring'),
    (20, 'mcwf_nodup_minimal_prior_20_20260909T111401Z', 'Minimal global classifier with prior-corrected joint overlap'),
    (21, 'mcwf_nodup_bounded_information_21_20260909T111754Z', 'Reference-prior bounded expected-information development check'),
    (21, 'mcwf_nodup_bounded_information_21b_20260909T112100Z', 'Independent analytic-gradient numerical retry of round21'),
]
KEY = ('GW191103_012549', 'GW191105_143521')
ROOT = None


def csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8-sig')


def concat(parts):
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect():
    if ROOT.exists():
        raise RuntimeError('Audit snapshots must use independent directories')
    for folder in ('contracts', 'tables', 'reports', 'scripts', 'manifest', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    ledger, budgets, keys, correlations, sources, seed_errors = [], [], [], [], [], []
    retrieval, models, protected = [], [], []
    for number, name, description in ROUNDS:
        folder = P / 'results' / name
        if not folder.exists():
            ledger.append({'round': number, 'path': str(folder), 'description': description, 'exists': False})
            continue
        freeze = folder / 'contracts/PREDICTIVE_FROZEN.json'
        pred = json.loads(freeze.read_text()) if freeze.exists() else {}
        ledger.append({'round': number, 'path': str(folder), 'description': description, 'exists': True,
            'predictive_both_runs_pass': pred.get('both_runs_pass'),
            'simulation_evaluated': (folder/'contracts/EVALUATION_COMPLETE.json').exists(),
            'real_audited': (folder/'contracts/REAL_COMPLETE.json').exists(),
            'scientifically_valid_comparison': number != 11, 'adopted': False})
        for filename, dest in [('RETRIEVAL_SUMMARY.csv', retrieval), ('RETRIEVAL_PER_MODEL.csv', models)]:
            path = folder / 'tables' / filename
            if path.exists():
                a = pd.read_csv(path)
                a.insert(0, 'round', number)
                dest.append(a)
                sources.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
        old_seed = folder / 'tables/PER_SEED_PE_OFFICIAL_BUDGETS.csv'
        if old_seed.exists():
            frame = pd.read_csv(old_seed)
            bad = int(frame.seed.astype(str).eq('consensus').sum())
            seed_errors.append({'round': number, 'path': str(old_seed), 'rows': len(frame),
                'rows_incorrectly_labeled_consensus': bad,
                'fix': 'Recalculate using seed=actual_seed in budget_row; original file untouched'})
        for path in sorted(folder.glob('results/*/gwtc*/**/*_all_pairs.parquet')):
            rel = path.relative_to(folder/'results').parts
            if len(rel) not in (4, 5):
                continue
            method, dep = rel[:2]
            mode = 'waveform' if path.name.startswith('waveform') else 'fusion'
            if rel[2] == 'consensus':
                seed, rank = 'consensus', 'consensus_rank'
            elif rel[2].startswith('seed_') and len(rel) == 5 and rel[3] == 'real':
                seed, rank = rel[2][5:], 'rank'
            else:
                continue
            a = pd.read_parquet(path).sort_values(rank, kind='mergesort')
            expected = 1891 if dep == 'gwtc3' else 2701
            if len(a) != expected or a.pair_key.duplicated().any():
                raise RuntimeError('Real scope or pair uniqueness changed: '+str(path))
            for budget in (10, 20, 50, 100):
                row = n.dev.budget_row(a, method, dep, mode, budget, seed=seed)
                budgets.append({'round': number, **row, 'source_path': str(path)})
            if dep == 'gwtc3':
                selected = a[a.pair_key.astype(str).str.contains(KEY[0], regex=False) &
                             a.pair_key.astype(str).str.contains(KEY[1], regex=False)]
                if len(selected) != 1:
                    raise RuntimeError('Critical pair must appear once')
                row = selected.iloc[0]
                item = {'round': number, 'config': method, 'deployment': dep, 'mode': mode,
                        'seed': seed, 'rank': int(row[rank]), 'path': str(path)}
                for field in ('waveform_score', 'final_score', 'waveform_score_mean', 'final_score_mean',
                              'time_score', 'sky_raw_log_bf', 'pe_mc_bhattacharyya_coefficient',
                              'pe_mc_standardized_distance', 'pe_dmax_intrinsic',
                              'official_po_or_ml_fpp_below_0p01', 'official_any_pair_resolved_hanabi_overlap'):
                    if field in row:
                        item[field] = row[field]
                keys.append(item)
            if seed != 'consensus':
                for metric, sign in (('pe_mc_bhattacharyya_coefficient', 1), ('pe_mc_standardized_distance', -1)):
                    valid = a[['waveform_score', metric]].dropna()
                    rho = spearmanr(valid.waveform_score, sign*valid[metric]).statistic
                    correlations.append({'round': number, 'config': method, 'deployment': dep,
                        'seed': seed, 'mode': mode, 'quantity': metric, 'direction': sign,
                        'spearman_descriptive': rho, 'pairs': len(valid),
                        'independent_pair_pvalue_not_reported': True})
            sources.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    table = pd.DataFrame(budgets)
    guards = []
    for number, a in table[(table.method=='fusion') & table.budget.isin([10,20])].groupby('round'):
        for (dep, seed, budget), b in a.groupby(['deployment', 'seed', 'budget']):
            for baseline in ('NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED'):
                ref = b[b.config == baseline]
                if len(ref) != 1:
                    continue
                ref = ref.iloc[0]
                for _, row in b[~b.config.isin(['NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED'])].iterrows():
                    item = {'round': number, 'config': row.config, 'deployment': dep, 'seed': seed,
                            'budget': budget, 'baseline': baseline, 'all_PE_official_nonworse': True}
                    for metric in ('BC_mc_ge_0p5', 'median_BC_mc', 'Dmax_le_3', 'official_frontend', 'official_hanabi'):
                        delta = row[metric]-ref[metric]
                        item['delta_'+metric] = delta
                        item['all_PE_official_nonworse'] &= bool(delta >= -1e-12)
                    item['delta_catastrophic_mc'] = row.catastrophic_mc-ref.catastrophic_mc
                    item['all_PE_official_nonworse'] &= bool(item['delta_catastrophic_mc'] <= 0)
                    guards.append(item)
    retrieval_df = concat(retrieval)
    injection_guards = []
    for (number, dep, split, mode), a in retrieval_df.groupby(['round','deployment','split','mode']):
        for baseline in ('NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED'):
            ref = a[a.method == baseline]
            if len(ref) != 1:
                continue
            ref = ref.iloc[0]
            for _, row in a[~a.method.isin(['NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED'])].iterrows():
                delta_r = row.macro_r_at_10_mean-ref.macro_r_at_10_mean
                delta_ap = row.average_precision_mean-ref.average_precision_mean
                f50ratio = (row.false_at_recall_0p5_mean+1e-12)/(ref.false_at_recall_0p5_mean+1e-12)
                f90ratio = (row.false_at_recall_0p9_mean+1e-12)/(ref.false_at_recall_0p9_mean+1e-12)
                injection_guards.append({'round': number, 'deployment': dep, 'split': split, 'mode': mode,
                    'config': row.method, 'baseline': baseline, 'delta_R10': delta_r, 'delta_AP': delta_ap,
                    'F50_ratio': f50ratio, 'F90_ratio': f90ratio,
                    'existing_guard': delta_r >= -.02 and delta_ap >= -.005 and f50ratio <= 1.10 and f90ratio <= 1.10,
                    'strict_no_loss': delta_r >= -1e-12 and delta_ap >= -1e-12 and f50ratio <= 1+1e-12 and f90ratio <= 1+1e-12})
    prior = pd.read_csv(r.PRIOR / 'manifest/INPUT_SHA256.csv')
    for row in prior.drop_duplicates('path').to_dict('records'):
        path = Path(row['path'])
        actual = n.sha(path) if path.is_file() else None
        protected.append({'path': str(path), 'before_sha256': row['sha256'], 'after_sha256': actual,
                          'unchanged': actual == row['sha256']})
    if not all(row['unchanged'] for row in protected):
        raise RuntimeError('Protected historical input changed')
    csv(ROOT/'tables/EXPERIMENT_LEDGER.csv', ledger)
    csv(ROOT/'tables/RETRIEVAL_SUMMARY_ALL_ROUNDS.csv', retrieval_df)
    csv(ROOT/'tables/RETRIEVAL_PER_MODEL_ALL_ROUNDS.csv', concat(models))
    csv(ROOT/'tables/PE_OFFICIAL_BUDGETS_CORRECTED_SEEDS.csv', table)
    csv(ROOT/'tables/CRITICAL_PAIR_RANK_TRACE.csv', keys)
    csv(ROOT/'tables/PE_OFFICIAL_BOTH_BASELINE_GUARDS.csv', guards)
    csv(ROOT/'tables/INJECTION_BOTH_BASELINE_GUARDS.csv', injection_guards)
    csv(ROOT/'tables/WAVEFORM_PE_CORRELATIONS.csv', correlations)
    csv(ROOT/'tables/SEED_LABEL_REPORTING_BUG_AUDIT.csv', seed_errors)
    csv(ROOT/'tables/HISTORICAL_INPUT_HASH_VERIFICATION.csv', protected)
    csv(ROOT/'manifest/INPUT_SHA256.csv', sources)
    n.write_json(ROOT/'contracts/AUDIT_CONTRACT.json', {'UTC': n.utc(),
        'status': n.STATUS, 'goal_achieved': False, 'adoption': False,
        'purpose': 'Audit snapshot, not a success declaration or final selection',
        'rounds': [row[0] for row in ROUNDS], 'two_baselines_explicit': True,
        'seed_label_fix': 'Pass seed explicitly to budget_row; historical raw score/rank files unchanged',
        'protected_files': len(protected), 'protected_hash_failures': 0,
        'adaptive_development': True, 'official_overlap_not_lensing_truth': True,
        'real_or_reused_catalogs_not_independent_confirmation': True,
        'prespecified_external_metrics': ['McBC>=.5','medianMcBC','Mc catastrophic','Dmax<=3','official1percent','publishedHanabi overlap'],
        'injection_guard_reporting': 'Existing .02R10/.005AP/10percentF50F90 noninferiority and strict no-loss both shown; no changed pass thresholds.'})
    shutil.copy2(__file__, ROOT/'scripts/campaign_audit.py')
    print(json.dumps({'root': str(ROOT), 'rounds': len(ledger), 'corrected_budget_rows': len(table),
                      'critical_rank_rows': len(keys), 'historical_hash_failures': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.root
    collect()
