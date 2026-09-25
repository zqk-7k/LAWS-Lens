#!/usr/bin/env python3
"""Continuous profile overlap with one source-disjoint waveform calibration."""
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
import torch
from numpy.polynomial.legendre import leggauss
from scipy.stats import t as student

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_waveform_calibration_20260909 as base
n, r, mode, co, h = base.n, base.r, base.mode, base.co, base.h
cal, d = mode.c, mode.d
ROOT = DATA = AUDIT = None
COUNT = 8
ORIGINAL_PANEL = h.load_panel


def overlaps(dep, slot, mass, joint_path, dest):
    if dest.exists():
        return dict(np.load(dest))
    spec = json.loads((co.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    valid = np.isfinite(mass['p']).all(1)
    ids = np.flatnonzero(valid)
    pp, active, centers = mass['p'][valid], mass['active'][valid], mass['profile_centers'][valid]
    joint = np.load(joint_path)['joint'][valid].astype(float)
    norm = joint.sum(-1, keepdims=True)
    underflow = norm[..., 0] <= 1e-250
    joint /= norm.clip(1e-250)
    ck = torch.load(r.INTR/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt',
                    map_location='cpu', weights_only=False)
    prior = r.physical.original.interpolate_rows(ck['conditional_prior'], r.physical.old.CENTERS)
    joint[underflow] = np.broadcast_to(prior, joint.shape)[underflow]
    joint /= joint.sum(-1, keepdims=True)
    sqrt_conditional = torch.as_tensor(np.sqrt(joint), dtype=torch.float64, device='cuda')
    widths = np.diff(d.EDGES)
    nodes, weights = leggauss(COUNT)
    size = len(ids)
    jb = torch.zeros((size, size), dtype=torch.float64, device='cuda')
    mb = np.zeros((size, size), float)
    integral = np.zeros(size)
    for node, weight in zip(nodes, weights):
        x = d.EDGES[:-1]+widths*(node+1)/2
        density = pp/widths
        for index in np.flatnonzero(active):
            density[index] = np.exp(cal.normalized_logpdf(x, np.full(len(x), centers[index]), spec))
        root_mass = np.sqrt(density*widths*weight/2)
        integral += (root_mass**2).sum(1)
        mb += root_mass@root_mass.T
        z = (torch.as_tensor(root_mass, dtype=torch.float64, device='cuda')[:, :, None]*sqrt_conditional).reshape(size, -1)
        jb += z@z.T
        del z
    continuous_joint = jb.cpu().numpy()
    del jb, sqrt_conditional
    error = float(abs(integral-1).max())
    if error > 1e-6 or (continuous_joint-mb).max() > 1e-8:
        raise RuntimeError('Continuous quadrature normalization/data-processing failure')
    coarse_mass = np.sqrt(pp)@np.sqrt(pp).T
    excess = float((mb-coarse_mass).max())
    if excess > 1e-8:
        raise RuntimeError('Continuous BC exceeds binning upper bound')
    summary = r.mass_summary(mass['p'])
    for index in np.flatnonzero(mass['active'] & valid):
        center = mass['profile_centers'][index]+spec['location']
        lower, upper = student.cdf((d.EDGES[[0, -1]]-center)/spec['scale'], spec['df'])
        q16, q50, q84 = center+spec['scale']*student.ppf(lower+np.array([.16, .5, .84])*(upper-lower), spec['df'])
        summary[index, :2] = q50, max((q84-q16)/2, .001)
    result = {'massbc': np.full((len(valid), len(valid)), np.nan),
              'jointbc': np.full((len(valid), len(valid)), np.nan),
              'sm': summary, 'active': mass['active'], 'outside': mass.get('outside', np.zeros(len(valid)))}
    result['massbc'][np.ix_(ids, ids)] = mb.clip(1e-300, 1.)
    result['jointbc'][np.ix_(ids, ids)] = continuous_joint.clip(1e-300, 1.)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **result)
    n.write_json(dest.with_suffix('.json'), {'UTC': n.utc(), 'gauss_legendre_nodes_per_bin': COUNT,
        'max_probability_norm_error': error, 'mass_BC_upper_bound_violation': excess,
        'max_mass_BC_binning_bias': float((coarse_mass-mb).max()),
        'conditional_fallback_mass_max': float((pp*underflow).sum(1).max()),
        'joint_source_sha256': n.sha(joint_path), 'profile_spec_sha256': n.sha(co.PARENT/'calibration/PROFILE_PREDICTIVE.json'),
        'NN_information_not_upsampled': True, 'summary_quantiles': 'Continuous truncated Student-t active;unchanged original interpolation inactive;entropy remains original-bin diagnostic.',
        'no_new_PE_posterior': True})
    return result


def population(dep, seed):
    key = dep, seed
    if key in base.MEMO:
        return base.MEMO[key]
    slot = n.recipes()[key]['slot']
    path = DATA/f'predictions/{dep}/{slot}_parent.npz'
    with np.load(path) as file:
        p, outside = file['p'], file['outside']
    ensemble = np.load(DATA/f'predictions/{dep}/ensemble_mass.npy')
    records = [json.loads(f.read_text()) for f in sorted((DATA/f'profile_events/{dep}').glob('*.json'))]
    spec = json.loads((co.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    active, centers = mode.quality(ensemble, records, spec['parent_coverage'])
    p[active] = cal.density(centers[active], spec)
    mass = {'p': p, 'active': active, 'profile_centers': centers, 'outside': outside}
    base.MEMO[key] = overlaps(dep, slot, mass, path, ROOT/f'cache/{dep}_{seed}_development.npz')
    return base.MEMO[key]


def panel(dep, seed, split, catalog=None):
    frame = ORIGINAL_PANEL(dep, seed, split, catalog)
    slot = n.recipes()[dep, seed]['slot']
    tag = split if catalog is None else f'{split}_{catalog}'
    mass = co.mass(dep, seed, split, catalog)
    path = r.INTR/f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz' if split == 'development' else r.prediction_path(dep, seed, split, catalog)
    a = overlaps(dep, slot, mass, path, ROOT/f'cache/panels/{dep}_{seed}_{tag}.npz')
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    x = h.features(a, i, j)
    for k, name in enumerate(h.FEATURES):
        frame[name] = x[:, k]
    frame['continuous_joint_BC'] = a['jointbc'][i, j]
    frame['continuous_mass_BC'] = a['massbc'][i, j]
    return frame


def freeze():
    base.freeze()
    table = pd.read_csv(AUDIT/'tables/MASS_BC_QUADRATURE_AUDIT.csv')
    selected = table[(table.left.astype(str) == '8') & (table.right.astype(str) == '16')]
    if set(selected.deployment) != set(n.DEPS) or not selected.converged.all():
        raise RuntimeError('Development-only quadrature convergence not demonstrated')
    n.write_json(ROOT/'contracts/CONTINUOUS_OVERLAP_ADDENDUM.json', {
        'UTC': n.utc(), 'id': 'MCWF-NODUP-CONTINUOUS-OVERLAP-31', 'overrides_base': 'Only overlap integration and active predictive quantiles. Same23Bfit/tune,classifiergrid,quality,models and R10error scales.',
        'audit': str(AUDIT), 'audit_sha256': n.sha(AUDIT/'tables/MASS_BC_QUADRATURE_AUDIT.csv'),
        'nodes': COUNT, 'selection': 'Minimum quadrature with own probability norm error<=1e-6 and converged BC versus next rule in both runs/all listed development strata.',
        'continuous_definition': 'Active logMc density is the existing normalized Student-t;inactive mass density remains piecewise uniform in native512bins. Frozen conditional(eta,chi|Mc) is piecewise constant per original mass bin.',
        'BC_joint': 'Sum over mass bins and GL8nodes of quadratureweight*sqrt(p_i(logMc)*p_j(logMc))*sum_eta_chi sqrt(c_i*c_j). No independent extra Mc evidence.',
        'same_both_runs': True, 'no_predictive_width_change': True, 'NN_subbin_information_not_invented': True,
        'conditional_underflow': 'Same frozen simulated conditional prior fallback with affected probability mass audit.',
        'probability_gate': 'Before scoring each panel: norm error<=1e-6;jointBC<=massBC;continuousmassBC<=binnedmassBC within1e-8.',
        'quantiles': 'Exact truncated Student-t16/50/84percentiles for profile-active events;unchangedNNfallback. Existing .001logwidthfloor unchanged.',
        'rationale': 'Cauchy-Schwarz implies binning can only raise Bhattacharyya overlap. Reducing discretization error is not a new evidence channel or a change fitted to real PE.',
        'real_or_test_selection': False, 'time_sky_outer_weights_unchanged': True,
        'runtime_sha256': n.sha(Path(__file__))})
    shutil.copy2(__file__, ROOT/'scripts/continuous_overlap_calibration.py')


def run_stage(stage):
    base.ROOT, base.DATA = ROOT, DATA
    base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = ROOT
    r.install()
    co.score.METHODS = base.METHODS
    co.score.matrices = co.matrices
    base.population = population
    h.load_panel = panel
    n.METHODS, n.load_panel, n.infer = base.METHODS, panel, base.infer
    if stage == 'freeze':
        freeze()
    elif stage == 'calibrate':
        base.calibrate()
    else:
        # Preserve the frozen configuration file; repair only its legacy reader.
        def read_configs(root):
            receipt = json.loads((root/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
            path = root/receipt.get('file', 'configs/SELECTED_CONFIGURATIONS.json')
            if n.sha(path) != receipt['sha256']:
                raise RuntimeError('Frozen classifier configuration changed')
            return json.loads(path.read_text())
        n.selections = read_configs
        n.run(ROOT, stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args(); ROOT, DATA, AUDIT = args.root, args.data_root, args.audit_root
    run_stage(args.stage)
