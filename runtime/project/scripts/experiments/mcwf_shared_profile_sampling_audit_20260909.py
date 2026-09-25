#!/usr/bin/env python3
"""Audit the exact R50 random-plus-neighbor negative sampling design."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as pilot
n = pilot.n
DATA = P/'results/mcwf_nodup_independent_bulk_development_22b_20260909T114300Z'


def inclusion(N, H, random_count, hard_count):
    if min(H-random_count, N-H) <= 0 or H-random_count < hard_count:
        raise RuntimeError('Unsupported finite-population sampling counts')
    k = np.arange(0, min(random_count, H-1)+1)
    mass = hypergeom.pmf(k, N-1, H-1, random_count)
    pi_n = random_count / N
    pi_h = pi_n + (1-pi_n)*hard_count*np.sum(mass/(H-k))
    if not 0 < pi_n <= pi_h <= 1:
        raise RuntimeError('Invalid inclusion probability')
    return float(pi_n), float(pi_h)


def unit():
    N, H, r, h = 9, 5, 2, 2
    ordinary, hard = inclusion(N, H, r, h)
    frequencies = np.zeros(N)
    populations = list(itertools.combinations(range(N), r))
    for first in populations:
        second_options = list(itertools.combinations(sorted(set(range(H))-set(first)), h))
        for second in second_options:
            frequencies[list(set(first)|set(second))] += 1/(len(populations)*len(second_options))
    expected = np.r_[np.full(H, hard), np.full(N-H, ordinary)]
    if not np.allclose(frequencies, expected, rtol=0, atol=1e-12):
        raise RuntimeError('Exact sampling enumeration failed')
    target = np.arange(1, N+1, dtype=float)
    estimate = np.dot(frequencies, target/expected)
    if not np.isclose(estimate, target.sum(), rtol=0, atol=1e-10):
        raise RuntimeError('Horvitz-Thompson expectation unit failed')
    return {'N': N, 'H': H, 'random_count': r, 'hard_count': h,
        'max_inclusion_error': float(abs(frequencies-expected).max()),
        'expected_HT_total_error': float(abs(estimate-target.sum())), 'passed': True}


def main(root, source):
    if root.exists():
        raise RuntimeError('Independent sampling audit required')
    for folder in ('contracts', 'tables', 'audit', 'scripts', 'manifest', 'reports'):
        (root/folder).mkdir(parents=True)
    n.write_json(root/'contracts/ANALYSIS_CONTRACT.json', {
        'UTC': n.utc(), 'id': 'MCWF-SHARED-PROFILE-SAMPLING-AUDIT-53',
        'purpose': 'Reconstruct exact R50 finite population and inclusion probabilities before any refit. No real candidate scoring.',
        'reference': 'https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446',
        'source_pair_target': 'Each unordered source pair has total population weight 1 across valid image combinations.',
        'selection': 'r uniform random null event pairs, followed by h uniform neighbor-null pairs excluding the first draw.',
        'pi_non_neighbor': 'r/N',
        'pi_neighbor': 'r/N + (1-r/N)*h*E[1/(H-K)], K~Hypergeom(N-1,H-1,r)',
        'HT_weight': '1/(source_pair_event_multiplicity * inclusion_probability)',
        'normalized_loss_limit': 'Class-normalized weighted mean is a ratio estimator, not exactly unbiased at finite sample size.',
        'hash_sampling_assumption': 'Independent fixed hash namespaces act as uniform random permutations, not a cryptographic proof of physical randomness.',
        'fitting_or_scoring': False, 'goal_achieved': False, 'status': n.STATUS})
    n.write_json(root/'audit/EXACT_ENUMERATION_UNIT.json', unit())
    plan = pd.read_parquet(source/'contracts/PAIR_PLAN.parquet')
    pilot.predictive.DATA = DATA
    rows, summaries, inputs = [], [], [source/'contracts/PAIR_PLAN.parquet', Path(pilot.__file__)]
    for dep in n.DEPS:
        meta, _, _, _, active, _, _, _ = pilot.predictive.inputs(dep)
        bank_path = DATA/f'data/{dep}/noise/noise_manifest.csv'
        banks = pd.read_csv(bank_path).set_index('bank_index')
        noise = meta.noise_bank_index.map(banks.parent_file_gps).to_numpy()
        profiles = {int(path.stem): json.loads(path.read_text())
                    for path in (DATA/f'profile_events/{dep}').glob('*.json')}
        inputs.extend([DATA/f'data/{dep}/event_metadata.parquet', bank_path,
                       DATA/f'predictions/{dep}/ensemble_mass.npy'])
        inputs.extend((DATA/f'profile_events/{dep}').glob('*.json'))
        for fold in (0, 1):
            ids = np.flatnonzero(active & meta.fold.eq(fold).to_numpy())
            true_population = []
            for uid, group in meta.iloc[ids].groupby('source_uid'):
                if len(group) == 2:
                    i, j = sorted(group.index.to_list())
                    true_population.append((pilot.digest(f'{pilot.SEED}/{dep}/{fold}/true/{uid}'), i, j))
            if len(true_population) < 24:
                raise RuntimeError('Too few positive source pairs')
            pi_true = 24 / len(true_population)
            population = [(int(i), int(j)) for a, i in enumerate(ids) for j in ids[a+1:]
                if meta.source_uid.iloc[i] != meta.source_uid.iloc[j] and noise[i] != noise[j]]
            neighbors = set()
            for i in ids:
                options = [(abs(profiles[int(i)]['logmc']-profiles[int(j)]['logmc']), int(j))
                    for j in ids if j != i and meta.source_uid.iloc[i] != meta.source_uid.iloc[j]
                    and noise[i] != noise[j]]
                for _, j in sorted(options)[:8]:
                    neighbors.add(tuple(sorted((int(i), j))))
            def source_key(i, j):
                return tuple(sorted((str(meta.source_uid.iloc[i]), str(meta.source_uid.iloc[j]))))
            multiplicity = Counter(source_key(i, j) for i, j in population)
            pi_n, pi_h = inclusion(len(population), len(neighbors), 48, 48)
            selected = plan[(plan.deployment == dep) & (plan.fold == fold)]
            expected_random = sorted((pilot.digest(f'{pilot.SEED}/{dep}/{fold}/null/{i}/{j}'), i, j)
                                     for i, j in population)[:48]
            random_set = {(i, j) for _, i, j in expected_random}
            expected_hard = sorted((pilot.digest(f'{pilot.SEED}/{dep}/{fold}/hard/{i}/{j}'), i, j)
                                   for i, j in neighbors-random_set)[:48]
            for kind, expected in [('true', sorted(true_population)[:24]),
                                   ('random_null', expected_random), ('hard_null', expected_hard)]:
                observed = selected[selected.kind == kind]
                if set(zip(observed.idx_i, observed.idx_j)) != {(i, j) for _, i, j in expected}:
                    raise RuntimeError('Frozen R50 sampling plan not reproduced')
            for row in selected.itertuples():
                true = row.kind == 'true'
                pair = (int(row.idx_i), int(row.idx_j))
                neighbor = pair in neighbors
                mult = 1 if true else multiplicity[source_key(*pair)]
                pi = pi_true if true else pi_h if neighbor else pi_n
                rows.append({'pair_id': row.pair_id, 'deployment': dep, 'fold': fold,
                    'kind': row.kind, 'source_i': row.source_i, 'source_j': row.source_j,
                    'neighbor_population': neighbor, 'source_pair_multiplicity': mult,
                    'inclusion_probability': pi, 'source_pair_weight': 1/mult,
                    'HT_weight': 1/(mult*pi)})
            total = sum(1/multiplicity[source_key(i, j)] for i, j in population)
            if not np.isclose(total, len(multiplicity), rtol=0, atol=1e-8):
                raise RuntimeError('Source-pair total-weight identity failed')
            summaries.append({'deployment': dep, 'fold': fold, 'active_events': len(ids),
                'null_event_pairs': len(population), 'neighbor_event_pairs': len(neighbors),
                'null_source_pairs': len(multiplicity), 'pi_non_neighbor': pi_n,
                'pi_neighbor': pi_h, 'true_source_pairs': len(true_population),
                'pi_true': pi_true, 'source_pair_total_error': abs(total-len(multiplicity))})
    n.write_csv(root/'tables/SAMPLING_POPULATION_SUMMARY.csv', summaries)
    pd.DataFrame(rows).to_parquet(root/'tables/SAMPLING_WEIGHTS.parquet', index=False)
    n.write_csv(root/'manifest/INPUT_SHA256.csv', [{'path': str(path), 'sha256': n.sha(path)}
        for path in sorted(set(inputs))])
    shutil.copy2(__file__, root/'scripts/shared_profile_sampling_audit.py')
    n.write_json(root/'contracts/SAMPLING_AUDIT_COMPLETE.json', {'UTC': n.utc(),
        'passed': True, 'pair_rows': len(rows), 'population_panels': len(summaries),
        'no_new_real_or_catalog_scores': True, 'goal_achieved': False})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print('SAMPLING_AUDIT_PASS', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.pilot_root)
