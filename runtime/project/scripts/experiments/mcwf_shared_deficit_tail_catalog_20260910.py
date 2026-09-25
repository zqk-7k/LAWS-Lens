#!/usr/bin/env python3
"""R65: only attenuate waveform scores beyond a frozen true-source deficit tail."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_global_branch_calibration_plan_20260910 as cal
import mcwf_reference_regularized_catalog_20260910 as prior
import mcwf_shared_profile_catalog_audit_20260909 as audit
n = prior.n
BASE_EXPORT = n.public_frame
BASE_CSV = n.write_csv
METHODS = ('NODUP-DIRECT-REPLAY', 'R62-FROZEN-REPLAY', 'SHARED-DEFICIT-TAIL95-ONLY')
NEW = METHODS[-1]
ALPHA = .05
ROOT = REFERENCE = POPULATION = None


def path(root, method, dep, seed, split, catalog=None):
    folder = split if catalog is None else f'{split}_{catalog}'
    filename = 'fusion_all_pairs.parquet' if split == 'real' else 'pairs.parquet'
    return root / f'results/{method}/{dep}/seed_{seed}/{folder}/{filename}'


def tail_threshold(values):
    values = np.sort(np.asarray(values, dtype=float))
    order = int(math.ceil((len(values) + 1) * (1 - ALPHA)))
    if len(values) < 20 or order > len(values) or not np.isfinite(values).all():
        raise RuntimeError('Insufficient independent source tail support')
    return float(values[order - 1]), order


def interval(k, size):
    return (0. if k == 0 else float(beta.ppf(.025, k, size - k + 1)),
            1. if k == size else float(beta.ppf(.975, k + 1, size - k)))


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent tail-only output required')
    for name in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'logs', 'figures', 'results'):
        (ROOT / name).mkdir(parents=True)
    contract = {'UTC': cal.utc(), 'id': 'MCWF-SHARED-DEFICIT-TAIL95-CATALOG-65',
        'reference': str(REFERENCE), 'population': str(POPULATION), 'status': cal.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'methods': METHODS, 'primary': NEW,
        'motivation': 'Global branch affine calibration failed. Restrict waveform attenuation to genuine empirical shared-source incompatibility instead of replacing every eligible pair score, including compatible sources.',
        'previous_controls': 'R51 reject-only min applied to every eligible pair; R48 rejection used independent event predictive BC, not shared pair-fit D. Here the tail variable is the frozen actual shared-fit deficit.',
        'alpha': ALPHA, 'threshold_grid': None, 'threshold_selection': 'Run-specific fold0 eligible true-source census; sorted D at ceil((n+1)*.95). Same perrun cutoff for all3encoder models. No choices from real data, test, or official/PE labels.',
        'waveform_rule': 'If eligible and D>cutoff, use min(NODUP_waveform,R62_waveform); otherwise exact NODUP. This is one bounded waveform channel, not a total-score mixture or additional independent evidence.',
        'shared_D': 'Independent projection power minus shared projection power. Phenomenological20-580Hz16s waveform fit; NOT normalized likelihood, full PE, or Bayes factor.',
        'pilot_gate': 'Eachrun at least20fit and20tune eligible true systems. Tune true exceedance Clopper-Pearson95% lower bound <=.05 (not demonstrably overnominal); source-bootstrap95% lower percentile of HT null exceedance minus true exceedance >0. Report upper bound; passing does NOT prove true rejection<=5%. Bothruns mustpass.',
        'bootstrap': '2000source draws on full512-source fold1, fixed sampling weights; compare conditional E-tail rates. Shared-source image pairs move together. Noise-block/sampling-design uncertainty not included.',
        'limitations': 'Nominal tail inspired by split-conformal quantiles, but reused adaptive data and selected profile support mean no formal distribution-free real-GWTC coverage claim. No thresholds altered on failure.',
        'frozen': ['encoder and predictive checkpoints', 'all R62 waveform scores and deficit measurements', 'eligibility', 'time', 'sky', 'outer weights', 'source/noise scope', 'all historical results'],
        'deleted_stays_deleted': ['legacy encoder Mc/q scoring', 'PATH875 total blend'],
        'no_new_training_or_PE_or_Hanabi': True,
        'full_evaluation': 'After pilot PASS, replay original validation/test and3reused catalogs, bothruns3models; external ranking only after all injection evaluations. PE/official joined after ranking.',
        'full_goal_guards': 'Unchanged NODUP primary reference: eachmodel/panel waveform andfusion R10drop<=.02,APdrop<=.005,F50/F90<=1.1reference; report strict no-loss separately. Top10/20 PE and official budgets nonworse bothruns consensus AND eachmodel. Critical pair mustleaveTop10consensusandallmodels.',
        'historical_reference': 'PATH875 archive only; it is not an input to new waveform score. Passing NODUP comparison is not passing all historical versions.',
        'references': [{'url': 'https://arxiv.org/abs/2107.07511',
             'relation': 'Finite-sample empirical quantile convention; exchangeability is necessary for formal coverage.',
             'not_claimed': 'No conformal guarantee for adaptively reused or domain-shifted GWTC events.'},
            {'url': 'https://arxiv.org/abs/2104.09339', 'relation': 'Common-source model motivates checking joint compatibility, but this surrogate is not Hanabi evidence.'},
            {'url': 'https://www.jmlr.org/papers/v11/cawley10a.html', 'relation': 'Repeated validation limits confirmation claims.'}]}
    cal.write(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    inputs = [Path(__file__), Path(cal.__file__), Path(prior.__file__), Path(audit.__file__),
              REFERENCE / 'configs/SELECTED_CONFIGURATIONS.json', REFERENCE / 'contracts/ANALYSIS_CONTRACT.json',
              POPULATION / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet',
              POPULATION / 'contracts/ANALYSIS_CONTRACT.json', POPULATION / 'contracts/PILOT_GATE.json']
    # Real outcome files are hashed for immutability, not read for threshold selection.
    for method in (METHODS[0], 'REFERENCE-REGULARIZED-SINGLE-WF', n.BASELINE):
        inputs += list((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        inputs += list((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    inputs += [n.t.EXTERNAL / f'{dep}_external_reference.parquet' for dep in n.DEPS]
    for module in list(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if f and str(P / 'scripts/experiments') in f and Path(f).suffix == '.py':
            inputs.append(Path(f))
    cal.snapshot(ROOT, inputs)
    shutil.copy2(__file__, ROOT / 'scripts/shared_deficit_tail_catalog.py')
    cal.write(ROOT / 'contracts/START_FREEZE.json', {'UTC': cal.utc(), 'runtime_sha256': cal.sha(Path(__file__)),
        'contract_sha256': cal.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
        'manifest_sha256': cal.sha(ROOT / 'manifest/INPUT_SHA256.csv')})
    print('TAIL_ONLY_PROTOCOL_FROZEN', ROOT, flush=True)


def check():
    spec = json.loads((ROOT / 'contracts/START_FREEZE.json').read_text())
    for key, p in [('runtime_sha256', Path(__file__)), ('contract_sha256', ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
                   ('manifest_sha256', ROOT / 'manifest/INPUT_SHA256.csv')]:
        if cal.sha(p) != spec[key]:
            raise RuntimeError('Frozen tail protocol changed')
    hashes = pd.read_csv(ROOT / 'manifest/INPUT_SHA256.csv')
    for r in hashes.itertuples():
        if cal.sha(r.path) != r.sha256:
            raise RuntimeError('Historical input changed: ' + r.path)
    return len(hashes)


def calibrate():
    check()
    if (ROOT / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Tail threshold already frozen')
    c = json.loads((POPULATION / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data = Path(c['data'])
    pairs = pd.read_parquet(POPULATION / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet')
    refs = {(r['deployment'], r['seed']): r for r in n.selections(REFERENCE) if r['method'] == 'REFERENCE-REGULARIZED-SINGLE-WF'}
    configurations, rows, plans, thresholds = [], [], [], {}
    for offset, dep in enumerate(n.DEPS):
        f = pairs[(pairs.deployment == dep) & pairs.eligible].copy()
        fit = f[f.fold.eq(0) & f.kind.eq('true')]
        tune = f[f.fold.eq(1)].copy()
        if fit.source_i.duplicated().any() or tune.loc[tune.kind == 'true', 'source_i'].duplicated().any():
            raise RuntimeError('Tail calibration unit must be source doublet')
        cutoff, order = tail_threshold(fit.deficit)
        y = tune.kind.eq('true').to_numpy()
        flag = tune.deficit.to_numpy() > cutoff
        total, rejected = int(y.sum()), int(flag[y].sum())
        if total < 20:
            raise RuntimeError('Insufficient tune sources')
        lower, upper = interval(rejected, total)
        raw = tune.global_HT_weight.to_numpy()
        null_rate = float(np.average(flag[~y], weights=raw[~y]))
        true_rate = rejected / total
        meta = pd.read_parquet(data / f'data/{dep}/event_metadata.parquet')
        names = sorted(meta.source_uid[meta.fold == 1].unique())
        lookup = {v: k for k, v in enumerate(names)}
        gi, gj = tune.source_i.map(lookup).to_numpy(int), tune.source_j.map(lookup).to_numpy(int)
        rng = np.random.default_rng(2026091065 + offset)
        counts = rng.multinomial(len(names), np.full(len(names), 1. / len(names)), size=2000)
        weights = counts[:, gi] * counts[:, gj] * raw[None, :]
        weights[:, y] = counts[:, gi[y]]
        denominators = (weights[:, y].sum(1), weights[:, ~y].sum(1))
        if any(np.any(d <= 0) for d in denominators):
            raise RuntimeError('Degenerate conditional tail bootstrap')
        boot_delta = (weights[:, ~y] @ flag[~y]) / denominators[1] - (weights[:, y] @ flag[y]) / denominators[0]
        ci = np.quantile(boot_delta, [.025, .975])
        passed = bool(lower <= ALPHA and ci[0] > 0)
        rows.append({'deployment': dep, 'alpha': ALPHA, 'cutoff_D': cutoff, 'fit_sources': len(fit),
            'finite_sample_order': order, 'tune_true_sources': total, 'true_tail_count': rejected,
            'true_tail_rate': true_rate, 'true_tail_CP95_lower': lower, 'true_tail_CP95_upper': upper,
            'HT_null_tail_rate': null_rate, 'null_minus_true': null_rate - true_rate,
            'source_bootstrap_delta_lower': ci[0], 'source_bootstrap_delta_upper': ci[1], 'gate_pass': passed})
        thresholds[dep] = {'cutoff_D': cutoff, 'alpha': ALPHA, 'fit_sources': len(fit),
            'order': order, 'calibration_values': np.sort(fit.deficit.to_numpy()).tolist()}
        g = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j', 'deficit', 'global_HT_weight']].copy()
        g['cutoff_D'] = cutoff; g['tail_flag'] = g.deficit > cutoff
        plans.append(g)
        for seed in n.SEEDS:
            r = refs[dep, seed]
            for method in METHODS:
                configurations.append({**r, 'method': method, 'tail_calibration': thresholds[dep],
                    'same_rule_both_runs': True, 'tail_only': method == NEW})
        print('TAIL_VALIDATION', json.dumps(rows[-1], default=cal.plain), flush=True)
    cal.csv(ROOT / 'tables/TAIL_VALIDATION.csv', rows)
    pd.concat(plans).to_parquet(ROOT / 'tables/TAIL_FIT_TUNE_FLAGS.parquet', index=False)
    cal.write(ROOT / 'configs/TAIL_THRESHOLDS.json', thresholds)
    cal.write(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configurations)
    cal.write(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': cal.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': cal.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_and_test_not_used': True, 'no_hyperparameter_grid': True})
    cal.write(ROOT / 'contracts/PILOT_GATE.json', {'UTC': cal.utc(),
        'gate': 'PASS' if all(r['gate_pass'] for r in rows) else 'FAIL',
        'goal_achieved': False, 'status': cal.STATUS, 'not_a_coverage_guarantee': True})


def load_panel(dep, seed, split, catalog=None):
    def read(method):
        out = pd.read_parquet(path(REFERENCE, method, dep, seed, split, catalog))
        return out.sort_values(['idx_i', 'idx_j'], kind='stable').reset_index(drop=True)
    f = read('REFERENCE-REGULARIZED-SINGLE-WF')
    nd, archive = read(METHODS[0]), read(n.BASELINE)
    for other in (nd, archive):
        for col in ('idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf', 'embedding_only'):
            if not f[col].equals(other[col]):
                raise RuntimeError('Frozen source pair table changed')
    for tag, table in [('R62', f.copy()), ('NODUP', nd)]:
        for target, col in [('waveform', 'waveform_score'), ('ood', 'score_ood'), ('clip', 'score_clipped')]:
            f[f'{tag}_frozen_{target}'] = table[col].to_numpy(copy=True)
    f['PATH875_waveform'] = archive.waveform_score.to_numpy(float)
    f['PATH875_final_score'] = archive.final_score.to_numpy(float)
    f = f.drop(columns=[c for c in f if c.startswith(('pe_', 'official_')) or c in ('rank', 'seed')])
    return f


def infer(frame, config):
    method = config['method']
    original = frame.NODUP_frozen_waveform.to_numpy(float)
    candidate = frame.R62_frozen_waveform.to_numpy(float)
    active = frame.shared_profile_eligible.to_numpy(bool)
    d = frame.shared_profile_deficit.to_numpy(float)
    threshold = config['tail_calibration']['cutoff_D']
    flagged = active & (d > threshold)
    if method == METHODS[1]:
        z, ood, clipped = [frame['R62_frozen_' + c].to_numpy(copy=True) for c in ('waveform', 'ood', 'clip')]
        used = active
    else:
        z, ood, clipped = [frame['NODUP_frozen_' + c].to_numpy(copy=True) for c in ('waveform', 'ood', 'clip')]
        used = flagged & (candidate < original) if method == NEW else np.zeros(len(frame), bool)
        z[used] = candidate[used]
        ood[used] = frame.R62_frozen_ood.to_numpy(bool)[used]
        clipped[used] = frame.R62_frozen_clip.to_numpy(bool)[used]
    if not np.isfinite(z).all():
        raise RuntimeError('Invalid waveform score')
    if method == NEW and (np.any(z > original) or not np.array_equal(z[~flagged], original[~flagged])):
        raise RuntimeError('Tail-only noncompensatory rule failed')
    frame['deficit_tail_cutoff'] = threshold
    frame['deficit_tail_flag'] = flagged
    frame['deficit_tail_attenuated'] = used if method == NEW else False
    frame['deficit_tail_reference_waveform'] = candidate
    frame['shared_profile_used'] = used
    frame['shared_profile_candidate_waveform'] = z
    frame['NN_joint_BC_used_in_waveform'] = ~used | (config['features'] == 4)
    return z, ood, clipped


def export(frame, z, weights, method):
    out = BASE_EXPORT(frame, z, weights, method)
    fields = [c for c in frame if c.startswith(('shared_', 'deficit_tail_'))]
    fields += [c for c in ('sky_j50', 'sky_j90', 'NN_joint_BC_used_in_waveform') if c in frame]
    for c in fields:
        out[c] = frame[c].to_numpy(copy=True)
    return out


def save_csv(filename, values):
    if Path(filename).name == 'PER_SEED_PE_OFFICIAL_BUDGETS.csv':
        values = []
        for dep in n.DEPS:
            for seed in n.SEEDS:
                for method in (n.BASELINE, *METHODS):
                    f = pd.read_parquet(path(ROOT, method, dep, seed, 'real'))
                    values += [{**n.dev.budget_row(f, method, dep, 'fusion', b), 'seed': seed} for b in (10, 20, 50, 100)]
    BASE_CSV(filename, values)


def units():
    check()
    specs = {(c['deployment'], c['seed'], c['method']): c for c in n.selections(ROOT)}
    rows = []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            f = load_panel(dep, seed, 'validation')
            for method in METHODS:
                c = specs[dep, seed, method]
                old = infer(f.copy(), c)[0]
                poison = f.copy()
                for key in ('pe_mc_bhattacharyya_coefficient', 'official_frontend', 'old_Mc_weight',
                            'old_mass_score', 'old_q_score', 'PATH875_waveform', 'PATH875_final_score'):
                    poison[key] = 99999.
                new = infer(poison, {**c, 'alpha': -999., 'old_Mc_weight': 999.})[0]
                if not np.array_equal(old, new):
                    raise RuntimeError('Forbidden input affected waveform')
                reordered = f.iloc[::-1].copy()
                if not np.array_equal(infer(reordered, c)[0][::-1], old):
                    raise RuntimeError('Row order affects waveform')
                if method != NEW:
                    field = 'NODUP_frozen_waveform' if method == METHODS[0] else 'R62_frozen_waveform'
                    if not np.array_equal(old, f[field]):
                        raise RuntimeError('Historical score replay changed')
                rows.append({'deployment': dep, 'seed': seed, 'method': method,
                    'forbidden_fields_no_effect': True, 'order_invariant': True, 'historical_replay_exact': True})
    # Exercise both branches even where the original validation has no eligible pair.
    f = pd.DataFrame({'NODUP_frozen_waveform': [8., 8., 8., 8.], 'R62_frozen_waveform': [2., 2., 9., -2.],
        'NODUP_frozen_ood': False, 'NODUP_frozen_clip': False, 'R62_frozen_ood': False, 'R62_frozen_clip': False,
        'shared_profile_eligible': [True, True, True, False], 'shared_profile_deficit': [9., 11., 11., 11.]})
    c = {'method': NEW, 'tail_calibration': {'cutoff_D': 10.}, 'features': 3}
    if not np.array_equal(infer(f, c)[0], [8., 2., 8., 8.]):
        raise RuntimeError('Synthetic active/inactive/reward unit failed')
    cal.csv(ROOT / 'audit/PIPELINE_UNITS.csv', rows)
    cal.write(ROOT / 'contracts/PIPELINE_UNIT_PASS.json', {'UTC': cal.utc(), 'tests': len(rows),
        'synthetic_all_branches': True, 'PE_official_oldheads_totalblend_not_used': True})


def run(stage):
    check()
    if json.loads((ROOT / 'contracts/PILOT_GATE.json').read_text())['gate'] != 'PASS':
        raise RuntimeError('Tail pilot must pass')
    if not (ROOT / 'contracts/PIPELINE_UNIT_PASS.json').exists():
        raise RuntimeError('Unit tests required')
    n.METHODS = METHODS
    n.load_panel, n.infer, n.public_frame, n.write_csv = load_panel, infer, export, save_csv
    n.run(ROOT, stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'units', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT, REFERENCE, POPULATION = args.root, args.reference, args.population
    if args.stage in ('freeze', 'calibrate', 'units'):
        globals()[args.stage]()
    else:
        run(args.stage)
