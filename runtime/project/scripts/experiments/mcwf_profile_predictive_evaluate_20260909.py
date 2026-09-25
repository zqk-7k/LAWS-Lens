#!/usr/bin/env python3
"""Apply simulation-calibrated profile predictions with frozen time and sky."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_predictive_calibration_20260909 as cal
import mcwf_subgrid_evaluate_20260909 as score
import mcwf_multirate_deployment_20260908 as deployment
import mcwf_path_fresh_confirmation_20260908 as fresh
import mcwf_lowband_profile_20260908 as physical

d, n, r = cal.d, cal.n, cal.r
ROOT = None
METHODS = ('NODUP-DIRECT-REPLAY', 'PROFILE-JOINT-FIXED', 'PROFILE-JOINT-VAL')
WORKERS = 32
STATE = {}


def original(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    path = r.INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz' if split == 'development' else r.prediction_path(dep, seed, split, catalog)
    with np.load(path) as a:
        p = a['p'] if 'p' in a else a['joint'].sum(-1)
        ood = a['outside'] if 'outside' in a else p[:, :4].sum(1) + p[:, -4:].sum(1)
    return p, ood


def tag_for(seed, split, catalog):
    if split in ('development', 'real'):
        return split
    return f'catalog_{catalog}' if catalog is not None else f'{seed}_{split}'


def full20(raw, frequency, psd, v3):
    return v3.preprocess_24s(raw, frequency, psd, band_low_hz=20.)


def parent_ensemble(dep, seed, split, catalog=None):
    if split in ('development', 'real') or catalog is not None:
        return np.mean([original(dep, s, split, catalog)[0] for s in n.SEEDS], axis=0)
    cache = ROOT / f'cache/parent_ensemble/{dep}_{seed}_{split}.npz'
    if cache.exists():
        return np.load(cache)['p']
    # Apply every frozen predictor to THIS catalog, never concatenate events
    # belonging to another model seed's data split.
    x = d.event_features(dep, seed, split, catalog)
    probabilities = []
    own_error = None
    for slot in n.t.MODEL_SLOTS:
        cp = torch.load(d.LOW / f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
        model = d.mult.Predictor('MULTIRATE').cuda().eval()
        model.load_state_dict(cp['model'])
        pp, _ = d.mult.old.probability(d.mult.old.infer(model, (x - cp['mu']) / cp['sd']), cp['temperature'])
        if slot == n.recipes()[dep, seed]['slot']:
            own = original(dep, seed, split, catalog)[0]
            own_error = float(abs(pp - own).max())
            if own_error > 1e-4:
                raise RuntimeError('Frozenpredictorprobabilityreplaymismatch:' + str(own_error))
            pp = own
        probabilities.append(pp)
    p = np.mean(probabilities, axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, p=p)
    n.write_json(cache.with_suffix('.json'), {'same_catalog_all_models': True, 'own_replay_error': own_error,
        'replay_tolerance_probability_mass': 1e-4, 'oldcheckpoint_changed': False})
    return p


def prepare(dep, seed, split, catalog, event_ids):
    tag = tag_for(seed, split, catalog)
    folder = ROOT / f'cache/profile_inputs/{dep}/{tag}'
    if (folder / 'COMPLETE.json').exists():
        if not np.array_equal(np.load(folder / 'event_ids.npy'), event_ids):
            raise RuntimeError('Frozen profile activation scope changed')
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    x = d.event_features(dep, seed, split, catalog)[:, :27]
    if catalog is None:
        deployment.m.low_view = full20
        if split == 'real':
            if not (ROOT / 'contracts/EVALUATION_COMPLETE.json').exists():
                raise RuntimeError('Real rawreadrequirescompletedsimulationevaluation')
            if not (ROOT / 'contracts/INTEGRATION_FROZEN.json').exists():
                n.write_json(ROOT / 'contracts/INTEGRATION_FROZEN.json', {'UTC': n.utc(),
                    'configuration_sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
                    'data_replay_only': True, 'newtime_sky': False})
            raw = deployment.real_raw(ROOT, dep)
            _, events = n.dev.real_inputs(dep)
            valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
            inverse = np.full(len(valid), -1, int)
            inverse[valid] = np.arange(valid.sum())
            local = inverse[event_ids]
            if (local < 0).any():
                raise RuntimeError('Invalid strictscope event')
        else:
            raw = deployment.injection_raw(ROOT, dep, seed, split)
            local = event_ids
        freq, psds, ids, _ = n.t.contexts.psd_context(dep, seed, split)
        values = np.asarray(raw[local], np.float32)
        local_psd = np.asarray(psds[ids[local]])
        feature = x[local]
    else:
        fresh.init_worker(n.FRESH, dep, catalog)
        ctx = fresh.old._CTX
        src, v3 = ctx['source'], ctx['v3']
        source = n.FRESH / f'confirmation/{dep}/catalog_{catalog}'
        events = pd.read_parquet(source / 'event_manifest.parquet')
        systems = pd.read_parquet(source / 'source_systems.parquet').set_index('source_uid')
        values, local_psd = [], []
        for idx in event_ids:
            event = events.iloc[idx]
            row = systems.loc[event.source_uid]
            number = 1 if event.image == 'a' else 2
            clean, _ = src.detector_response(ctx['generator'], ctx['ifos'],
                src.source_parameters(row, event.gps_obs), src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
            b, offset = int(event.noise_bank_index), int(event.noise_offset_samples)
            scaled, _, _ = v3.scale_to_network_snr(np.asarray(clean, np.float32), event.target_network_snr, ctx['freq'], ctx['psds'][b])
            noise = np.asarray(ctx['refs'][b, :, offset:offset + v3.RAW_PADDED_SAMPLES], np.float32)
            raw = noise + v3.embed_signal_in_padded_window(scaled)
            old = v3.preprocess_24s(raw, ctx['freq'], ctx['psds'][b]).astype(np.float32)
            if not np.array_equal(old, np.load(source / f'events/{event.event_uid}_full24.npy')):
                raise RuntimeError('Frozenreusedcatalogstrainreplayfailed')
            values.append(full20(raw, ctx['freq'], ctx['psds'][b], v3))
            local_psd.append(ctx['psds'][b])
        values, local_psd, freq, feature = np.stack(values), np.stack(local_psd), ctx['freq'], x[event_ids]
    if values.shape != (len(event_ids), 2, 49152) or not np.isfinite(values).all():
        raise RuntimeError('Full20Hz contextshape/finitecheckfailed')
    for name, array in [('full20', values), ('frequency', freq), ('psd', local_psd), ('features', feature), ('event_ids', event_ids)]:
        np.save(folder / (name + '.npy'), array)
    n.write_json(folder / 'COMPLETE.json', {'UTC': n.utc(), 'events': len(event_ids),
        'all_old40Hz_peak2s_replayed': True, 'new_full20Hz_shape': list(values.shape),
        'new_SNR_scaling': False, 'inputs': {p.name: n.sha(p) for p in folder.glob('*.npy')}})
    return folder


def worker_init(folder):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _, _, v3 = deployment.m.expanded.modules()
    physical.base.STATE['v3'] = v3
    for key in ('full20', 'frequency', 'psd', 'features', 'event_ids'):
        STATE[key] = np.load(Path(folder) / (key + '.npy'), mmap_mode='r')


def fit_event(k):
    model = physical.LowBand(STATE['full20'][k], STATE['frequency'], STATE['psd'][k], 16)
    result = model.fit(physical.base.initial_peaks(STATE['features'][k]))
    result['row_index'] = int(STATE['event_ids'][k])
    result['best_run_converged'] = any(a['success'] and abs(a['value'] - result['projection_statistic']) < 1e-3 for a in result['optimizer_runs'])
    return result


def profiles(dep, seed, split, catalog, parent):
    if split == 'development':
        return [json.loads(path.read_text()) for path in sorted((cal.PROFILE / 'events').glob(dep + '_*.json'))]
    tag = tag_for(seed, split, catalog)
    out = ROOT / f'profile_events/{dep}/{tag}'
    out.mkdir(parents=True, exist_ok=True)
    sm = r.mass_summary(parent)
    ids = np.flatnonzero(np.isfinite(sm[:, 0]) & (np.exp(sm[:, 0]) <= 15.))
    if not len(ids):
        return []
    folder = prepare(dep, seed, split, catalog, ids)
    pending = [k for k, idx in enumerate(ids) if not (out / f'{idx}.json').exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(pending)), mp_context=mp.get_context('spawn'),
                                 initializer=worker_init, initargs=(folder,)) as pool:
            for count, result in enumerate(pool.map(fit_event, pending, chunksize=1), 1):
                n.write_json(out / f'{result["row_index"]}.json', result)
                if count % 16 == 0 or count == len(pending):
                    print('PROFILE_PANEL', dep, tag, count, len(pending), flush=True)
    return [json.loads((out / f'{idx}.json').read_text()) for idx in ids]


def predict(dep, seed, split, catalog=None):
    frozen = json.loads((ROOT / 'contracts/PREDICTIVE_FROZEN.json').read_text())
    if not frozen['both_runs_pass']:
        raise RuntimeError('Predictivevalidationfailed;no newrankingspermitted')
    slot = n.recipes()[dep, seed]['slot']
    tag = tag_for(seed, split, catalog)
    path = ROOT / f'predictions/{dep}/{slot}_{tag}.npz'
    if path.exists():
        return dict(np.load(path))
    calibration = ROOT / 'calibration/PROFILE_PREDICTIVE.json'
    if n.sha(calibration) != frozen['sha256']:
        raise RuntimeError('Predictivecalibrationchanged')
    spec = json.loads(calibration.read_text())[dep]['spec']
    p, outside = original(dep, seed, split, catalog)
    parent = parent_ensemble(dep, seed, split, catalog)
    rows = profiles(dep, seed, split, catalog, parent)
    active, centers = cal.quality_mask(parent, rows, spec['parent_coverage'])
    output = p.copy()
    output[active] = cal.density(centers[active], spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=output, outside=outside, active=active, profile_centers=centers)
    n.write_json(path.with_suffix('.json'), {'profiled': len(rows), 'active': int(active.sum()),
        'events': len(p), 'source_label_used_for_activation': False, 'old_legacy_regression_terms': False,
        'time_sky_changed': False, 'conditional_prediction_not_full_PE': True})
    return dict(np.load(path))


def freeze_score():
    score.freeze_score()
    n.write_json(ROOT / 'contracts/PROFILE_INTEGRATION_ADDENDUM.json', {'UTC': n.utc(),
        'profile': '20-580Hzfull24context,last16sprofile,originalpeak2sreplayexact',
        'activation': 'per-eventpredictedmedian<=15;no companionorPElabel',
        'seed_scope': 'Sameensembleactivationeverywhere:allthreefrozenpredictorsevaluatedonTHESAMEevents,includingseed-specificvalidation/testcatalogs',
        'reconstruction': 'Onlynewauxiliary20Hzcontext;nochange tohistoricalwaveforms,times,sky,maps,labels orSNR',
        'score': 'Samefixed/validationselected jointcalibration as03,no totalscoremixing',
        'files_named_low16s_by_legacy_replay_helper': 'Inthisrootonlytheycontain20-580Hzfull24/2048Hz;not old4096samplelow20-80Hzarray'})
    shutil.copy2(__file__, ROOT / 'scripts/profile_predictive_evaluate.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze_score', 'calibrate', 'evaluate', 'real'), required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    ROOT = cal.ROOT = d.ROOT = r.ROOT = score.ROOT = args.root
    WORKERS = args.workers
    score.METHODS = METHODS
    r.install()
    d.predict = predict
    n.METHODS, n.load_panel, n.infer = METHODS, score.load_panel, score.infer
    if args.stage == 'freeze_score':
        freeze_score()
    elif args.stage == 'calibrate':
        score.calibrate()
    else:
        n.run(ROOT, args.stage)
