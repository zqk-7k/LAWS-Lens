#!/usr/bin/env python3
"""One validation-selected predictive model and one waveform classifier."""
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

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_waveform_calibration_20260909 as base
import mcwf_independent_predictive_calibration_20260909 as pred
import mcwf_prior_constrained_information_20260909 as bounded
n, r, co, h, mode = base.n, base.r, base.co, base.h, base.mode
ROOT = DATA = PREDICTIVE = None
ORDER = ('GLOBAL', 'EXPECTED', 'STRENGTH', 'BOUNDED')
INFO = P/'results/mcwf_nodup_information_integrated_19_20260909T110744Z'
ORIGINAL_PANEL = h.load_panel


def freeze():
    base.freeze()
    n.write_json(ROOT/'contracts/PREDICTIVE_SELECTION_POLICY.json', {
        'UTC': n.utc(), 'id': 'MCWF-NODUP-INDEPENDENT-PREDICTIVE-SCORE-33',
        'overrides_base': 'Only predictive profile error model changes. One classifier, same data, hyperparameter grid, outer weights, quality and original neural conditionals as23B.',
        'predictive_source': str(PREDICTIVE), 'before_complete_predictive_results': not (PREDICTIVE/'contracts/PREDICTIVE_FROZEN.json').exists(),
        'selection': 'Only kinds passing frozenR29Gate in BOTH runs. Maximize minimum O3/O4a tune NLL improvement overR10,then mean improvement,then fixed complexity order.',
        'complexity_tie_order': list(ORDER), 'one_common_kind_for_both_runs': True,
        'no_passing_kind': 'HOLD;do not score a failed predictive model',
        'classification': 'Four predeclared GLOBAL/PARTITION x LINEAR/TREE controls. Each grid selected by independent tune logloss;no real/test/official selection.',
        'integration': 'Native512probability masses as23B,not a continuous-integration claim. R31 separately audits this discretization;report binning limitation.',
        'fallback': 'Exact R10 predictive mass outside selected covariate support or when local approximation is invalid. R10inactive events retain originalNN.',
        'real_information': 'Read previously frozen waveform-only information and profile records;no public PE in predictive construction.',
        'adaptive': True, 'new_locked_confirmation': False,
        'encoder_time_sky_outerweights_unchanged': True,
        'no_old_Mc_q': True, 'no_total_score_mixture': True,
        'script_sha256': n.sha(Path(__file__))})
    shutil.copy2(__file__, ROOT/'scripts/independent_predictive_scoring.py')


def prepare():
    path = ROOT/'configs/SELECTED_PREDICTIVE_KIND.json'
    if path.exists():
        return json.loads(path.read_text())
    receipt = json.loads((PREDICTIVE/'contracts/PREDICTIVE_FROZEN.json').read_text())
    source = PREDICTIVE/'calibration/INDEPENDENT_PREDICTIVE.json'
    if n.sha(source) != receipt['sha256']:
        raise RuntimeError('Predictive source hash mismatch')
    choices = json.loads(source.read_text())
    options = []
    for kind in ORDER:
        if receipt['both_run_pass_by_kind'].get(kind, False):
            gains = [choices[dep+'_'+kind]['selection']['R10_NLL']-
                     choices[dep+'_'+kind]['selection']['NLL'] for dep in n.DEPS]
            options.append({'kind': kind, 'minimum_gain': min(gains),
                            'mean_gain': float(np.mean(gains)), 'gains': gains})
    if not options:
        n.write_json(ROOT/'contracts/HOLD_NO_PREDICTIVE_KIND.json', {'UTC': n.utc(),
            'both_run_pass_by_kind': receipt['both_run_pass_by_kind']})
        raise RuntimeError('HOLD_NO_BOTH_RUN_PREDICTIVE_KIND')
    win = min(options, key=lambda row: (-row['minimum_gain'], -row['mean_gain'], ORDER.index(row['kind'])))
    result = {'UTC': n.utc(), 'selected': win, 'all_passing_options': options,
        'specs': {dep: choices[dep+'_'+win['kind']]['spec'] for dep in n.DEPS},
        'predictive_sha256': receipt['sha256'], 'real_or_test_used': False}
    n.write_json(path, result)
    n.write_json(ROOT/'contracts/PREDICTIVE_KIND_FROZEN.json', {'UTC': n.utc(),
        'file': str(path.relative_to(ROOT)), 'sha256': n.sha(path)})
    return result


def selected():
    receipt = json.loads((ROOT/'contracts/PREDICTIVE_KIND_FROZEN.json').read_text())
    path = ROOT/receipt['file']
    if n.sha(path) != receipt['sha256']:
        raise RuntimeError('Selected predictive model changed')
    return json.loads(path.read_text())


def covariates(dep, seed, split, catalog, mass):
    choice = selected()['selected']['kind']
    width = np.full(len(mass['p']), np.nan)
    center = mass['profile_centers'].copy()
    if choice == 'GLOBAL':
        width[mass['active']] = 1.
        return center, width
    tag = co.app.tag_for(seed, split, catalog)
    folder = INFO/f'information/{dep}/{tag}'
    for index in np.flatnonzero(mass['active']):
        path = folder/f'{index}.json'
        if not path.exists():
            raise RuntimeError('Missing frozen waveform information:'+str(path))
        row = json.loads(path.read_text())
        if choice == 'EXPECTED' and row.get('information_valid'):
            width[index] = row['information_logmc_width']
        elif choice == 'STRENGTH' and 'coefficients' in row and 'offsource_variance' in row:
            strength = np.sqrt(np.sum(np.asarray(row['coefficients'])**2 /
                                     np.asarray(row['offsource_variance'])[:, None]))
            if np.isfinite(strength) and strength > 0:
                width[index] = 1./strength
        elif choice == 'BOUNDED' and row.get('information_valid'):
            point = json.loads((co.PARENT/f'profile_events/{dep}/{tag}/{index}.json').read_text())
            m64, s64, _ = bounded.bounded_moments(row, point, 64)
            m128, s128, ess = bounded.bounded_moments(row, point, 128)
            valid = bool(np.isfinite([m128, s128]).all() and s128 > 0 and
                abs(s64/s128-1) <= .05 and abs(m64-m128) <= .001 and ess >= 10 and
                m128-5*s128 > pred.d.EDGES[0] and m128+5*s128 < pred.d.EDGES[-1])
            if valid:
                center[index], width[index] = m128, s128
    return center, width


def updated_mass(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    tag = co.app.tag_for(seed, split, catalog)
    dest = ROOT/f'predictions/{dep}/{slot}_{tag}.npz'
    if dest.exists():
        return dict(np.load(dest))
    if split == 'real' and not (ROOT/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Simulation evaluation must precede real audit')
    mass = co.mass(dep, seed, split, catalog)
    center, width = covariates(dep, seed, split, catalog, mass)
    choice = selected()
    kind, spec = choice['selected']['kind'], choice['specs'][dep]
    active = mass['active'] & np.isfinite(width) & (width > 0)
    if kind != 'GLOBAL':
        active &= (width >= spec['minimum_h']) & (width <= spec['maximum_h'])
    p = mass['p'].copy()
    if kind == 'GLOBAL':
        p[active] = pred.cal.density(center[active], spec)
    else:
        p[active] = pred.h.density(center[active], width[active], spec)
    if not np.array_equal(p[~active], mass['p'][~active], equal_nan=True):
        raise RuntimeError('Inactive predictive fallback changed')
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = {**mass, 'p': p, 'predictive_active': active, 'predictive_center': center, 'predictive_width': width}
    np.savez_compressed(dest, **result)
    n.write_json(dest.with_suffix('.json'), {'UTC': n.utc(), 'kind': kind,
        'predictive_active': int(active.sum()), 'profile_active': int(mass['active'].sum()),
        'fallback_exact': True, 'no_PE_input': True,
        'selected_model_sha256': n.sha(ROOT/'configs/SELECTED_PREDICTIVE_KIND.json')})
    return result


def overlaps(dep, slot, mass, joint_path, dest):
    if dest.exists():
        return dict(np.load(dest))
    p = mass['p']
    valid = np.isfinite(p).all(1)
    ids = np.flatnonzero(valid)
    joint = np.load(joint_path)['joint'][valid].astype(float)
    norm = joint.sum(-1, keepdims=True)
    zero = norm[..., 0] <= 1e-250
    joint /= norm.clip(1e-250)
    ck = torch.load(r.INTR/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt',
                    map_location='cpu', weights_only=False)
    prior = r.physical.original.interpolate_rows(ck['conditional_prior'], r.physical.old.CENTERS)
    joint[zero] = np.broadcast_to(prior, joint.shape)[zero]
    joint /= joint.sum(-1, keepdims=True)
    joint *= p[valid, :, None]
    if abs(joint.sum(-1)-p[valid]).max() > 1e-10:
        raise RuntimeError('Mass marginal changed')
    z = torch.as_tensor(np.sqrt(joint.reshape(len(ids), -1)), dtype=torch.float64, device='cuda')
    jb = (z@z.T).cpu().numpy().clip(1e-300, 1.)
    del z
    mb = (np.sqrt(p[valid])@np.sqrt(p[valid]).T).clip(1e-300, 1.)
    if (jb-mb).max() > 1e-8:
        raise RuntimeError('BC data-processing inequality failed')
    result = {'massbc': np.full((len(p), len(p)), np.nan),
        'jointbc': np.full((len(p), len(p)), np.nan), 'sm': r.mass_summary(p), 'active': mass['active']}
    result['massbc'][np.ix_(ids, ids)] = mb
    result['jointbc'][np.ix_(ids, ids)] = jb
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **result)
    n.write_json(dest.with_suffix('.json'), {'UTC': n.utc(),
        'selected_model_sha256': n.sha(ROOT/'configs/SELECTED_PREDICTIVE_KIND.json'),
        'conditional_fallback_mass_max': float((p[valid]*zero).sum(1).max()),
        'joint_input_sha256': n.sha(joint_path), 'native_mass_bins': 512})
    return result


def population(dep, seed):
    key = dep, seed
    if key in base.MEMO:
        return base.MEMO[key]
    slot = n.recipes()[key]['slot']
    parent_path = DATA/f'predictions/{dep}/{slot}_parent.npz'
    with np.load(parent_path) as file:
        p = file['p']
    kind = selected()['selected']['kind']
    updated = dict(np.load(PREDICTIVE/f'predictions/{dep}_{kind}_development.npz'))
    oldspec = json.loads((co.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    original_active = updated['profile_active']
    p[original_active] = pred.cal.density(updated['profile_center'][original_active], oldspec)
    active = updated['active']
    p[active] = updated['p'][active]
    mass = {'p': p, 'active': original_active}
    base.MEMO[key] = overlaps(dep, slot, mass, parent_path, ROOT/f'cache/{dep}_{seed}_development.npz')
    return base.MEMO[key]


def panel(dep, seed, split, catalog=None):
    frame = ORIGINAL_PANEL(dep, seed, split, catalog)
    mass = updated_mass(dep, seed, split, catalog)
    slot = n.recipes()[dep, seed]['slot']
    tag = split if catalog is None else f'{split}_{catalog}'
    a = overlaps(dep, slot, mass, r.prediction_path(dep, seed, split, catalog),
                 ROOT/f'cache/panels/{dep}_{seed}_{tag}.npz')
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    for k, name in enumerate(h.FEATURES):
        frame[name] = h.features(a, i, j)[:, k]
    frame['predictive_active_i'] = mass['predictive_active'][i]
    frame['predictive_active_j'] = mass['predictive_active'][j]
    return frame


def main():
    global ROOT, DATA, PREDICTIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--predictive-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'prepare', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT, DATA, PREDICTIVE = args.root, args.data_root, args.predictive_root
    base.ROOT, base.DATA = ROOT, DATA
    base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = ROOT
    r.install()
    co.score.METHODS = base.METHODS
    co.score.matrices = co.matrices
    if args.stage == 'freeze':
        freeze(); return
    if args.stage == 'prepare':
        prepare(); return
    selected()
    base.population = population
    h.load_panel = panel
    n.METHODS, n.load_panel, n.infer = base.METHODS, panel, base.infer
    def configs(root):
        receipt = json.loads((root/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
        path = root/receipt.get('file', 'configs/SELECTED_CONFIGURATIONS.json')
        if n.sha(path) != receipt['sha256']:
            raise RuntimeError('Waveform classifier changed')
        return json.loads(path.read_text())
    n.selections = configs
    if args.stage == 'calibrate':
        base.calibrate()
    else:
        n.run(ROOT, args.stage)


if __name__ == '__main__':
    main()
