#!/usr/bin/env python3
"""A single waveform LR with explicitly normalized quality partitions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_profile_single_waveform_20260909 as s
import mcwf_profile_mode_aware_20260909 as mode
h, co, r, n = s.h, s.co, s.r, s.n
ROOT = DATA = None
METHODS = ('NODUP-DIRECT-REPLAY', 'NEWDEV-GLOBAL-LINEAR', 'NEWDEV-GLOBAL-TREE',
           'NEWDEV-PARTITION-LINEAR', 'NEWDEV-PARTITION-TREE')
MEMO = {}
MODEL_CACHE = {}
torch.set_num_threads(2)


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts', 'configs', 'calibration', 'cache', 'predictions', 'tables', 'audit',
                   'scripts', 'logs', 'manifest', 'reports', 'results', 'figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {'UTC': n.utc(),
        'id': 'MCWF-NODUP-INDEPENDENT-WAVEFORM-CALIBRATION-23', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'same_both_runs': True,
        'data': str(DATA), 'new_calibration': '1024 new sourceparents/run;512fit/512tune;64new256snoiseblocks/run. Bothimages assignedsamefoldbeforedatageneration.',
        'frozen_prediction': 'Unmodified three MULTIRATE/conditionalmodels plus R10 profilequality/predictive Student-t. No new widthchosenfrompublicPE.',
        'features': list(s.FEATURES), 'methods': list(METHODS),
        'global': 'Exactly onebalanced-class waveform classifier logit,notseparateZcos+mass+q terms.',
        'partition': 'A=atleastoneeventpassesfrozenR10profilequality. FitseparateclassifiersconditionalonA=0/1,then add log[P(A|companion)/P(A|noncompanion)] fromfitonly. This equalsjoint-featureLR under exact densities,not a total-score mixture.',
        'why_partition_offset': 'A classbalancedconditionalclassifier estimates p(x|L,A)/p(x|N,A). Withoutthepartitionprobabilityratio its scoresdonothavea commonnullreferenceacrossA.',
        'empirical_not_exact': 'Classifier approximationandfinite-dataOOD/caps meanfinalscore remainsrankingproxy,notphysicalPE/lensingBayesfactor.',
        'fit': 'Oneweightpersourcepair. Nulls sharingtheidenticalnoiseblock excludedfromdensity fit/tune;source andnoiseblockcount reported. Balancedclasses aftersourcepairweights.',
        'minimum': 'Atleast20true sourcegroups infitandtune for each partition. Reused pair count notindependent count.',
        'grid': {'ridge': [.001, .01, .1, 1.], 'tree_leaves': [3, 7], 'iterations': 100,
                 'learning_rate': .05, 'minimum_leaf_rows': 20},
        'selection': 'Eachseed/run classifier grid by newtune proper balancedlogloss;tiesfewerleaves,strongerridge. Allfourarmsretained;no real/testmetricselects agridpoint.',
        'OOD': 'Perpartition fitfeaturebox;partitionoffset appliedbeforecap/positiveOOD0;commoncaplog(totalfitcompanions+1).',
        'frozen': ['encoder', 'time', 'sky', 'outerweights', 'scope', 'historicalranks', 'paper'],
        'forbidden': ['oldMc_q_head', 'totalblendalpha', 'eventIDscore', 'PE_or_official_scorefeatures'],
        'validation_guard': 'Reportcurrentwaveform/fusionguardsonarchivedBAYESTARvalidation;notusedtoalterclassifierhyperparameters.',
        'evaluation': 'BothNODUPandPATH875comparisons;reusedinjectioncatalogsnotblind;realPEandofficialauditsafterconfigfreezeonly.',
        'references': ['https://arxiv.org/abs/1506.02169', 'https://arxiv.org/abs/gr-qc/9402014']})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', pd.read_csv(r.PRIOR/'manifest/INPUT_SHA256.csv'))
    n.write_json(ROOT/'audit/PARTITION_ALGEBRA.json', {
        'pass': bool(np.isclose(np.log(.8/.3)+np.log(.25/.4), np.log((.8*.25)/(.3*.4)))),
        'independent_metadata_not_used': True, 'no_free_mixture_coefficient': True})
    shutil.copy2(__file__, ROOT/'scripts/independent_waveform_calibration.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def population(dep, seed):
    key = dep, seed
    if key in MEMO:
        return MEMO[key]
    path = ROOT/f'cache/{dep}_{seed}_development.npz'
    if path.exists():
        MEMO[key] = dict(np.load(path))
        return MEMO[key]
    slot = n.recipes()[dep, seed]['slot']
    parent = dict(np.load(DATA/f'predictions/{dep}/{slot}_parent.npz'))
    ensemble = np.load(DATA/f'predictions/{dep}/ensemble_mass.npy')
    profiles = [json.loads(p.read_text()) for p in sorted((DATA/f'profile_events/{dep}').glob('*.json'))]
    spec = json.loads((co.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    active, centers = mode.quality(ensemble, profiles, spec['parent_coverage'])
    mass = parent['p'].copy()
    mass[active] = mode.c.density(centers[active], spec)
    conditional = parent['joint'].astype(float)
    norm = conditional.sum(-1, keepdims=True)
    zero = norm[..., 0] <= 1e-250
    conditional /= norm.clip(1e-250)
    ckpath = r.INTR/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt'
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    cp = r.physical.original.interpolate_rows(ck['conditional_prior'], r.physical.old.CENTERS)
    conditional[zero] = np.broadcast_to(cp, conditional.shape)[zero]
    conditional /= conditional.sum(-1, keepdims=True)
    joint = mass[:, :, None]*conditional
    marginal_error = float(abs(joint.sum(-1)-mass).max())
    if marginal_error > 1e-10:
        raise RuntimeError('Marginal changed')
    z = torch.as_tensor(np.sqrt(joint.reshape(len(mass), -1)), dtype=torch.float64, device='cuda')
    jb = (z@z.T).cpu().numpy().clip(1e-300, 1.)
    mb = (np.sqrt(mass)@np.sqrt(mass).T).clip(1e-300, 1.)
    if (jb-mb).max() > 1e-8:
        raise RuntimeError('BC data processing inequality failed')
    result = {'massbc': mb, 'jointbc': jb, 'sm': r.mass_summary(mass), 'active': active}
    np.savez_compressed(path, **result)
    n.write_json(path.with_suffix('.json'), {'UTC': n.utc(), 'marginal_error': marginal_error,
        'active_events': int(active.sum()), 'conditional_fallback_mass_max': float((mass*zero).sum(1).max()),
        'R10_spec_sha256': n.sha(co.PARENT/'calibration/PROFILE_PREDICTIVE.json'), 'new_predictions_sha256': n.sha(path)})
    MEMO[key] = result
    return result


def panels(dep, seed):
    meta = pd.read_parquet(DATA/f'data/{dep}/event_metadata.parquet')
    fold, groups = meta.fold.to_numpy(), meta.source_uid.to_numpy(str)
    noise = meta.noise_bank_index.to_numpy(int)
    a = population(dep, seed)
    z = np.load(DATA/f'predictions/{dep}/{seed}_embedding.npy').astype(float)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    if set(groups[fold == 0]) & set(groups[fold == 1]) or set(noise[fold == 0]) & set(noise[fold == 1]):
        raise RuntimeError('Source/noise leakage')
    result = {}
    for side in (0, 1):
        ids = np.flatnonzero(fold == side)
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        good = noise[i] != noise[j]
        i, j = i[good], j[good]
        y = groups[i] == groups[j]
        x = np.column_stack([np.sum(z[i]*z[j], 1), h.features(a, i, j)])
        active = a['active'][i] | a['active'][j]
        codes = np.minimum(i//2, j//2)*len(meta)+np.maximum(i//2, j//2)
        _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        w = 1/counts[inverse]
        w[y] *= .5/w[y].sum()
        w[~y] *= .5/w[~y].sum()
        result[side] = x, y, w, active, i, j
    return result


def fit_one(dep, seed, kind, state, panel):
    xx, yy, weights, aa, _, _ = panel[0]
    vx, vy, vw, va, _, _ = panel[1]
    mask = np.ones(len(yy), bool) if state is None else aa == state
    vmask = np.ones(len(vy), bool) if state is None else va == state
    x, y, w = xx[mask], yy[mask], weights[mask].copy()
    test, label, tw = vx[vmask], vy[vmask], vw[vmask].copy()
    if min(y.sum(), label.sum()) < 20:
        raise RuntimeError('HOLD_INSUFFICIENT_SOURCE_SUPPORT')
    offset = 0. if state is None else float(np.log(weights[mask & yy].sum()/weights[yy].sum())-
        np.log(weights[mask & ~yy].sum()/weights[~yy].sum()))
    w[y] *= .5/w[y].sum(); w[~y] *= .5/w[~y].sum()
    tw[label] *= .5/tw[label].sum(); tw[~label] *= .5/tw[~label].sum()
    models, trials = [], []
    for ridge in (.001, .01, .1, 1.):
        if kind == 'LINEAR':
            options = [s.linear(x, y, w, ridge)]
        else:
            options = []
            for leaves in (3, 7):
                estimator = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05, max_leaf_nodes=leaves,
                    min_samples_leaf=20, l2_regularization=ridge, early_stopping=False,
                    monotonic_cst=[1, 1, 1, 1, 0, 0, 0], random_state=202609092)
                estimator.fit(x, y, sample_weight=w*len(x))
                options.append({'kind': kind, 'estimator': estimator, 'leaves': leaves})
        for model in options:
            logits = r.raw_predict(model, test)
            value = float(tw@(np.logaddexp(0., logits)-label*logits))
            models.append(model)
            trials.append({'ridge': ridge, 'leaves': model.get('leaves', 0), 'logloss': value})
    best = min(range(len(trials)), key=lambda k: (trials[k]['logloss'], trials[k]['leaves'], -trials[k]['ridge']))
    labelstate = 'global' if state is None else str(int(state))
    dest = ROOT/f'calibration/{dep}/{seed}/{kind}_{labelstate}.pkl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as file:
        pickle.dump(models[best], file)
    n.write_csv(dest.with_suffix('.GRID.csv'), trials)
    spec = {'file': str(dest.relative_to(ROOT)), 'sha256': n.sha(dest), 'kind': kind, 'state': state,
        'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(), 'partition_log_ratio': offset,
        'cap': float(np.log(yy.sum()+1)), 'fit_sources': int(y.sum()), 'tune_sources': int(label.sum()),
        'fit_null_rows': int((~y).sum()), 'selected': trials[best], 'same_noise_nulls_excluded': True}
    print('INDEPENDENT_CLASSIFIER', dep, seed, kind, labelstate, spec['fit_sources'], spec['tune_sources'], offset, trials[best], flush=True)
    return spec


def apply(x, spec):
    path = ROOT/spec['file']
    if str(path) not in MODEL_CACHE:
        if n.sha(path) != spec['sha256']:
            raise RuntimeError('Classifier hash changed')
        with path.open('rb') as file:
            MODEL_CACHE[str(path)] = pickle.load(file)
    raw = r.raw_predict(MODEL_CACHE[str(path)], x)+spec['partition_log_ratio']
    outside = ((x < np.asarray(spec['minimum'])-1e-7) | (x > np.asarray(spec['maximum'])+1e-7)).any(1)
    clipped = abs(raw) > spec['cap']
    value = raw.clip(-spec['cap'], spec['cap'])
    value[outside] = np.minimum(value[outside], 0.)
    return value, outside, clipped


def infer(frame, config):
    original = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    if config['method'] == METHODS[0]:
        return co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    x = frame[list(s.FEATURES)].to_numpy(float)
    if len(config['experts']) == 1:
        return apply(x, config['experts'][0])
    a = frame.profile_pair_active.to_numpy(bool)
    output = np.empty(len(x)), np.empty(len(x), bool), np.empty(len(x), bool)
    for spec in config['experts']:
        mask = a == spec['state']
        if mask.any():
            for dst, src in zip(output, apply(x[mask], spec)):
                dst[mask] = src
    return output


def calibrate():
    configs, rows = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            p = panels(dep, seed)
            original = h.isolated.ORIGINALS[dep, seed]
            base = {**original, 'method': METHODS[0]}
            configs.append(base)
            frame = h.load_panel(dep, seed, 'validation')
            zref = infer(frame, base)[0]
            archived = pd.read_parquet(r.PRIOR/f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(frame[['idx_i', 'idx_j']].to_numpy(), archived[['idx_i', 'idx_j']].to_numpy()) or abs(zref-archived.waveform_score.to_numpy()).max() > 1e-12:
                raise RuntimeError('Frozen NODUP replay changed')
            fm = n.cf.fast_metrics(frame, n.cf.channels(frame, zref)@np.asarray(original['weights']))
            wm = n.cf.fast_metrics(frame, zref)
            for partition in (False, True):
                for kind in ('LINEAR', 'TREE'):
                    experts = [fit_one(dep, seed, kind, state, p) for state in ((False, True) if partition else (None,))]
                    method = 'NEWDEV-'+('PARTITION-' if partition else 'GLOBAL-')+kind
                    config = {**original, 'method': method, 'experts': experts}
                    z, oo, cl = infer(frame, config)
                    metric = n.cf.fast_metrics(frame, n.cf.channels(frame, z)@np.asarray(original['weights']))
                    waveform = n.cf.fast_metrics(frame, z)
                    config['tune_guard'] = n.cf.guard(metric, fm) and n.cf.guard(waveform, wm)
                    config['tune_metrics'] = metric
                    bad = frame.copy()
                    bad['pair_key'], bad['pe_mc_bhattacharyya_coefficient'], bad['official_po_fpp'] = 'unused', -123., 1.
                    if not np.array_equal(z, infer(bad, {**config, 'alpha': -999., 'old_Mc_weight': 999.})[0]):
                        raise RuntimeError('Forbidden metadata influences score')
                    configs.append(config)
                    rows.append({'deployment': dep, 'seed': seed, 'method': method, 'guard': config['tune_guard'],
                        'OOD_fraction': float(oo.mean()), 'clipped_fraction': float(cl.mean()), **metric})
            MEMO.clear()
    n.write_csv(ROOT/'tables/VALIDATION_CALIBRATION_GUARDS.csv', rows)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'sha256': n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'), 'no_real_test_selection': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT, DATA = args.root, args.data_root
    s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = ROOT
    r.install()
    co.score.METHODS = METHODS
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, h.load_panel, infer
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
