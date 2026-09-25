#!/usr/bin/env python3
"""Read-only R60 feasibility bound, not a scoring or ranking method."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_profile_catalog_audit_20260909 as audit
n = audit.n
FIELDS = ('BC_mc_ge_0p5', 'Dmax_le_3', 'official_frontend', 'official_hanabi')


def write_once(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)


def median_clauses(values, budget, threshold):
    # For sorted selected values a_k,a_(k+1), exclude a_k+a_(k+1)<2m.
    half = budget // 2
    for v in np.unique(values[values < threshold]):
        yield np.flatnonzero(values <= v), np.flatnonzero(values < 2 * threshold - v), half


def median_unit():
    rng = np.random.default_rng(2026090960)
    checks = 0
    for _ in range(30):
        values = np.round(rng.random(7), 2)
        for budget in (2, 4, 6):
            for subset in itertools.combinations(range(len(values)), budget):
                actual = np.median(values[list(subset)])
                for threshold in (actual, actual - 1e-8, actual + 1e-8, .5):
                    # Numerical tolerance is applied to the target, not the data.
                    target = threshold - 1e-12
                    legal = True
                    for lo, hi, half in median_clauses(values, budget, target):
                        a = len(set(subset) & set(lo))
                        b = len(set(subset) & set(hi))
                        legal &= a <= half - 1 or b <= half
                    if legal != (actual >= target):
                        raise RuntimeError('Median clause unit disagrees with exact median')
                    checks += 1
    return {'passed': True, 'exhaustive_subset_threshold_checks': checks,
            'includes_ties': True, 'seed': 2026090960}


def solve(values, indicators, catastrophic, inactive, critical, targets, budgets, timeout=120.):
    size = len(values)
    entries, lower, upper = [], [], []
    count = len(budgets) * size
    def add(ids, coefficients, low=-np.inf, high=np.inf):
        entries.append((np.asarray(ids, dtype=int), np.asarray(coefficients, dtype=float)))
        lower.append(low); upper.append(high)
    for bpos, budget in enumerate(budgets):
        offset = bpos * size
        add(np.arange(size) + offset, np.ones(size), budget, budget)
        for field, mask in indicators.items():
            idx = np.flatnonzero(mask)
            add(idx + offset, np.ones(len(idx)), targets[budget][field], np.inf)
        idx = np.flatnonzero(catastrophic)
        add(idx + offset, np.ones(len(idx)), high=targets[budget]['catastrophic_mc'])
        fixed = np.flatnonzero(inactive)
        for first, second in zip(fixed[:-1], fixed[1:]):
            add([offset + first, offset + second], [1., -1.], low=0.)
        if bpos == 0 and critical is not None:
            add([offset + critical], [1.], 0., 0.)
        for low_ids, high_ids, half in median_clauses(values, budget, targets[budget]['median_BC_mc'] - 1e-12):
            auxiliary = count
            count += 1
            # u=0 enforces count_low<=k-1; u=1 enforces count_high<=k.
            add(np.r_[offset + low_ids, auxiliary], np.r_[np.ones(len(low_ids)), -float(budget)], high=half - 1)
            add(np.r_[offset + high_ids, auxiliary], np.r_[np.ones(len(high_ids)), float(budget)], high=half + budget)
    for bpos in range(len(budgets) - 1):
        for i in range(size):
            add([bpos * size + i, (bpos + 1) * size + i], [1., -1.], high=0.)
    row, col, val = [], [], []
    for k, (ids, coefs) in enumerate(entries):
        row.extend([k] * len(ids)); col.extend(ids.tolist()); val.extend(coefs.tolist())
    matrix = coo_matrix((val, (row, col)), shape=(len(entries), count)).tocsc()
    start = time.monotonic()
    result = milp(np.zeros(count), integrality=np.ones(count), bounds=Bounds(0., 1.),
        constraints=LinearConstraint(matrix, lower, upper),
        options={'time_limit': timeout, 'mip_rel_gap': 0., 'presolve': True})
    info = {'solver_status': int(result.status), 'message': result.message,
            'seconds': time.monotonic() - start, 'binary_variables': count,
            'linear_constraints': len(entries), 'feasible_witness_verified': False}
    if result.x is None:
        info['conclusion'] = 'INFEASIBLE_OPTIMISTIC_SCOPE' if result.status == 2 else 'INCONCLUSIVE_SOLVER_LIMIT'
        return info, []
    x = result.x[:size * len(budgets)].reshape(len(budgets), size)
    if np.max(abs(x - np.rint(x))) > 1e-5:
        raise RuntimeError('Nonintegral solver witness')
    x = x > .5
    summary = []
    for k, budget in enumerate(budgets):
        selected = x[k]
        row = {'budget': budget, 'selected_pairs': int(selected.sum()),
               'active_pairs': int((selected & ~inactive).sum()),
               'median_BC_mc': float(np.median(values[selected])),
               'catastrophic_mc': int(catastrophic[selected].sum()),
               **{f: int(a[selected].sum()) for f, a in indicators.items()}}
        ok = selected.sum() == budget
        ok &= all(row[f] >= targets[budget][f] for f in indicators)
        ok &= row['catastrophic_mc'] <= targets[budget]['catastrophic_mc']
        ok &= row['median_BC_mc'] >= targets[budget]['median_BC_mc'] - 1e-12
        ok &= not np.any(np.diff(selected[inactive].astype(int)) > 0)
        if k == 0 and critical is not None:
            ok &= not selected[critical]
        if k:
            ok &= np.all(~x[k-1] | selected)
        if not ok:
            raise RuntimeError('Independent witness verification failed')
        summary.append(row)
    info['feasible_witness_verified'] = True
    info['conclusion'] = 'FEASIBLE_ONLY_WITH_UNCONSTRAINED_ORACLE_ORDER'
    # Deliberately return no selected pair identities, fitted score or ranking.
    return info, summary


def main(root, reference):
    if root.exists():
        raise RuntimeError('Independent R60 output required')
    for name in ('contracts', 'tables', 'audit', 'scripts', 'manifest', 'reports', 'logs'):
        (root / name).mkdir(parents=True)
    contract = {'UTC': n.utc(), 'id': 'MCWF-FROZEN-WAVEFORM-SCOPE-FEASIBILITY-60',
        'reference': str(reference), 'status': n.STATUS, 'goal_achieved': False,
        'type': 'Read-only post-development constraint audit; NOT model fitting, calibration, a new ranking, or an achievable performance claim.',
        'question': 'Even allowing arbitrary scores for every existing waveform-qualified pair, can nested Top10/20 budgets satisfy all per-model PE/official constraints while excluding the key pair from Top10?',
        'frozen': 'NODUP scope, inactive pair order, R55 eligibility, public audit values and existing budget thresholds.',
        'relaxation': 'Eligible pairs may move arbitrarily and differently per model/run. No common classifier, monotonic feature law, injection or consensus constraints. Feasibility is only an optimistic existence bound; it does not solve the scientific goal.',
        'exact_scope_reduction': 'Only the first20 inactive pairs can enter nested Top10/20 because their order is fixed. Retain all eligible pairs plus that inactive prefix; every other inactive binary is provably zero.',
        'constraints': {'budgets': [10, 20], 'fields': FIELDS,
            'median': 'Exact order-statistic disjunctions; median must not decrease, numerical tolerance1e-12.',
            'catastrophic_mc': 'BC<.1 or Dmc>5; cannot increase.',
            'key': audit.KEY, 'key_exclusion': 'O3 each seed Top10 only',
            'nested_sets': True, 'inactive_prefix': True},
        'solver': 'scipy.optimize.milp/HiGHS, binary feasibility, zero objective,120seconds/run/model; timeout is inconclusive, not infeasible.',
        'output_boundary': 'Only feasibility status, aggregate witness counts and input hashes; never export oracle selected identities, scores, ranking or model parameters.',
        'cannot_use_to_train_or_select': True, 'no_injection_or_real_rescoring': True,
        'not_independent_confirmation': True}
    write_once(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    shutil.copy2(__file__, root / 'scripts/waveform_scope_feasibility.py')
    write_once(root / 'audit/MEDIAN_UNIT.json', median_unit())
    zero = np.zeros(6, dtype=bool)
    unit_targets = {b: {'median_BC_mc': .5, 'catastrophic_mc': 0,
                       **{f: 0 for f in FIELDS}} for b in (2, 4)}
    solver_checks = []
    for inactive, expected in [(zero, True), (~zero, False)]:
        info, _ = solve(np.full(6, .5), {f: zero for f in FIELDS}, zero,
                        inactive, 0, unit_targets, (2, 4), 10.)
        if info['feasible_witness_verified'] != expected or (not expected and info['solver_status'] != 2):
            raise RuntimeError('Solver feasibility/prefix unit failed')
        solver_checks.append({'all_inactive': bool(inactive.all()), 'expected_feasible': expected, **info})
    write_once(root / 'audit/SOLVER_UNIT.json', solver_checks)
    paths = [Path(__file__), Path(audit.__file__), reference / 'audit/REFERENCE_AUDIT_FROZEN.json']
    for dep in n.DEPS:
        for seed in n.SEEDS:
            for method in ('NODUP-DIRECT-REPLAY', 'SHARED-PROFILE-MONOTONE-BOUNDARY'):
                paths.append(reference / f'results/{method}/{dep}/seed_{seed}/real/fusion_all_pairs.parquet')
    inputs = [{'path': str(p), 'sha256': n.sha(p)} for p in paths]
    n.write_csv(root / 'manifest/INPUT_SHA256.csv', inputs)
    write_once(root / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'runtime_sha256': n.sha(Path(__file__)), 'input_manifest_sha256': n.sha(root / 'manifest/INPUT_SHA256.csv')})
    results, budgets = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            frame = pd.read_parquet(reference / f'results/NODUP-DIRECT-REPLAY/{dep}/seed_{seed}/real/fusion_all_pairs.parquet').sort_values('rank').reset_index(drop=True)
            other = pd.read_parquet(reference / f'results/SHARED-PROFILE-MONOTONE-BOUNDARY/{dep}/seed_{seed}/real/fusion_all_pairs.parquet').set_index('pair_key').loc[frame.pair_key]
            for col in ('waveform_score', 'time_score', 'sky_raw_log_bf'):
                if col != 'waveform_score' and not np.array_equal(frame[col], other[col]):
                    raise RuntimeError('Frozen time/sky mismatch')
            bc = frame.pe_mc_bhattacharyya_coefficient.to_numpy(float)
            distance = frame.pe_mc_standardized_distance.to_numpy(float)
            if not np.isfinite(bc).all() or not np.isfinite(distance).all():
                raise RuntimeError('This exact median audit requires complete finite PE')
            inactive = ~other.shared_profile_eligible.to_numpy(bool)
            first = 'official_po_or_ml_fpp_below_0p01' if dep == 'gwtc3' else 'official_po_or_phazap_fpp_below_0p01'
            indicators = {'BC_mc_ge_0p5': bc >= .5,
                'Dmax_le_3': frame.pe_dmax_intrinsic.to_numpy(float) <= 3,
                'official_frontend': frame[first].fillna(False).to_numpy(bool),
                'official_hanabi': frame.official_any_pair_resolved_hanabi_overlap.fillna(False).to_numpy(bool)}
            catastrophic = (bc < .1) | (distance > 5)
            targets = {b: n.dev.budget_row(frame, 'NODUP', dep, 'fusion', b, seed) for b in (10, 20)}
            keep = ~inactive.copy()
            keep[np.flatnonzero(inactive)[:20]] = True
            critical = np.flatnonzero(frame.pair_key.to_numpy()[keep] == audit.KEY)
            if dep == 'gwtc3' and len(critical) != 1:
                raise RuntimeError('Missing critical pair')
            info, witness = solve(bc[keep], {f: a[keep] for f, a in indicators.items()}, catastrophic[keep], inactive[keep],
                int(critical[0]) if len(critical) else None, targets, (10, 20))
            results.append({'deployment': dep, 'seed': seed, 'pairs': len(frame),
                            'eligible_pairs': int((~inactive).sum()), 'pairs_after_exact_prefix_reduction': int(keep.sum()), **info})
            for b, target in targets.items():
                budgets.append({'deployment': dep, 'seed': seed, 'type': 'required_baseline',
                    **{f: target[f] for f in ('budget', 'median_BC_mc', 'catastrophic_mc', *FIELDS)}})
            budgets.extend({'deployment': dep, 'seed': seed, 'type': 'anonymous_oracle_witness', **r} for r in witness)
            print('SCOPE_FEASIBILITY', dep, seed, info['conclusion'], flush=True)
    for r in inputs:
        if n.sha(Path(r['path'])) != r['sha256']:
            raise RuntimeError('Read-only input changed')
    n.write_csv(root / 'tables/SCOPE_FEASIBILITY.csv', results)
    n.write_csv(root / 'tables/ANONYMOUS_AGGREGATE_WITNESS_BUDGETS.csv', budgets)
    write_once(root / 'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'AUDIT_COMPLETE',
        'goal_achieved': False, 'status': n.STATUS, 'new_rankings_produced': False,
        'oracle_identities_saved': False, 'historical_inputs_unchanged': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.reference_root)
