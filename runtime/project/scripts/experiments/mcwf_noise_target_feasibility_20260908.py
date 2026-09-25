#!/usr/bin/env python3
"""Audit objective feasibility, never create a candidate ranking or model input."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack
import mcwf_noise_context_20260908 as n


def audit(root):
    rows = []
    baseline = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    baseline = baseline[(baseline.seed.astype(str) == 'consensus') & (baseline.method == 'C_fixed')]
    for dep in n.e.old.DEPS:
        f = pd.read_parquet(root / f'audit/{dep}_external_reference.parquet')
        bc = f.pe_mc_bhattacharyya_coefficient.to_numpy(float)
        fail = (bc < .1) | (f.pe_mc_standardized_distance.to_numpy(float) > 5)
        okay = (~fail) & (f.pe_dmax_intrinsic.to_numpy(float) <= 3) & np.isfinite(bc)
        front = f['official_po_or_ml_fpp_below_0p01' if dep == 'gwtc3' else 'official_po_or_phazap_fpp_below_0p01'].fillna(False).to_numpy(bool)
        hanabi = f.official_any_pair_resolved_hanabi_overlap.fillna(False).to_numpy(bool)
        b = baseline[baseline.deployment == dep].set_index('budget')
        # A relaxation gives a rigorous infeasibility conclusion if the solver proves infeasible.
        # Feasibility of this relaxation is NOT proof a waveform-only model can achieve the target.
        arr = np.c_[bc[okay] >= .5, front[okay], hanabi[okay]].astype(float)
        values = bc[okay]
        k = len(arr)
        constraints, lower, upper = [], [], []
        for pos, budget in enumerate((10, 20)):
            offset = pos * k
            a = np.zeros(2 * k); a[offset:offset+k] = 1
            constraints.append(a); lower.append(budget); upper.append(budget)
            for c, name in enumerate(('BC_mc_ge_0p5', 'official_frontend', 'official_hanabi')):
                a = np.zeros(2 * k); a[offset:offset+k] = arr[:, c]
                constraints.append(a); lower.append(float(b.loc[budget, name])); upper.append(np.inf)
            a = np.zeros(2 * k); a[offset:offset+k] = values >= b.loc[budget, 'median_BC_mc']
            constraints.append(a); lower.append(budget / 2); upper.append(np.inf)
        for c, name in ((1, 'official_frontend'), (2, 'official_hanabi')):
            constraints.append(np.tile(arr[:, c], 2))
            lower.append(float(b.loc[[10, 20], name].sum()) + 1)
            upper.append(np.inf)
        dense = csr_matrix(np.stack(constraints))
        nested = hstack([eye(k), -eye(k)], format='csr')
        cons = [LinearConstraint(dense, lower, upper), LinearConstraint(nested, -np.inf, 0)]
        objective = -np.tile(arr[:, 0] + .01 * values, 2)
        res = milp(objective, integrality=np.ones(2*k), bounds=Bounds(0, 1), constraints=cons,
                   options={'time_limit': 30., 'mip_rel_gap': 0.})
        row = {'deployment': dep, 'all_pairs': len(f), 'noncatastrophic_Dmax_pass_pairs': k,
               'PE_eligible_official_frontend_pairs': int(front[okay].sum()),
               'PE_eligible_Hanabi_pairs': int(hanabi[okay].sum()),
               'solver_status': int(res.status), 'solver_message': res.message,
               'relaxation_proven_infeasible': bool(res.status == 2),
               'interpretation': 'Objective-only diagnostic. No proposed ordering, no fitted model, no PE/official inputs to a score.'}
        if res.x is not None:
            exact_medians_pass = True
            bc_gain = 0
            median_gain = 0.
            for pos, budget in enumerate((10, 20)):
                take = res.x[pos*k:(pos+1)*k] > .5
                median = np.median(values[take])
                exact_medians_pass &= median >= b.loc[budget, 'median_BC_mc'] - 1e-12
                row[f'hypothetical_Top{budget}_BC_count'] = int(arr[take, 0].sum())
                row[f'hypothetical_Top{budget}_median_BC'] = float(median)
                row[f'hypothetical_Top{budget}_front'] = int(arr[take, 1].sum())
                row[f'hypothetical_Top{budget}_Hanabi'] = int(arr[take, 2].sum())
                bc_gain += arr[take, 0].sum() - b.loc[budget, 'BC_mc_ge_0p5']
                median_gain += median - b.loc[budget, 'median_BC_mc']
            row['exact_external_budget_feasible_witness'] = bool(exact_medians_pass and (bc_gain > 0 or median_gain > 1e-10))
            row['witness_is_not_a_waveform_method'] = True
        rows.append(row)
    n.dev.csv_write(root / 'audit/EXTERNAL_OBJECTIVE_FEASIBILITY_NOT_A_RANKING.csv', pd.DataFrame(rows))
    n.dev.json_write(root / 'audit/EXTERNAL_OBJECTIVE_FEASIBILITY_NOT_A_RANKING.json', rows)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
