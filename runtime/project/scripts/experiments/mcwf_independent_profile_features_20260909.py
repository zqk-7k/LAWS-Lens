#!/usr/bin/env python3
"""Frozen waveform predictions and physical profiles on new calibration data."""
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
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_profile_development_20260909 as generation
import mcwf_subgrid_density_20260909 as d
import mcwf_physical_lowmass_profile_20260909 as prof
import mcwf_profile_mode_aware_20260909 as mode
import mcwf_profile_expected_information_20260909 as info
import mcwf_prior_constrained_information_20260909 as bounded
n, r, mult = d.n, d.reliability, generation.mult
ROOT = None
torch.set_num_threads(2)


def freeze():
    contract = ROOT/'contracts/FEATURE_PROFILE_CONTRACT.json'
    if contract.exists():
        raise RuntimeError('Already frozen')
    n.write_json(contract, {'UTC': n.utc(), 'id': 'NODUP22-FROZEN-PREDICTIONS',
        'data_contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'models': 'Frozen MULTIRATE mass and conditional eta/chi models, three original slots; old base96D encoder only, never Mc/q heads.',
        'features': 'Exact original 2277 peak2s ordered match features plus27 low20-80Hz16s channels. No feature-model retraining.',
        'profiles': 'R7/R10 four-start continuous IMRPhenomD20-580Hz16s profile, unchanged optimizer and quality checks. Source selected ifeither image ensembleMc<=15; activation iseventwise.',
        'quality': 'Frozen R10 parent coverage perrun, mode separation.02 andpowergap4; exact same bothrun algorithm.',
        'information': 'R18 expected nuisance-projected local information andR21 bounded reference-prior moments; unchanged numerical convergence gates.',
        'selection': 'Only newexplicit source/noise-disjoint fit/tune folds. No useof realcandidatePE orofficialstage.',
        'no_full_PE_claim': True, 'no_total_blend': True, 'historical_time_sky_unchanged': True,
        'numerical_replay': 'Compare same originaldevelopment waveforms/PSD with frozen ordered features and predictions before new data feature generation.'})
    shutil.copy2(__file__, ROOT/'scripts/independent_profile_features.py')
    n.write_json(ROOT/'contracts/FEATURE_PROFILE_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(contract), 'code_sha256': n.sha(Path(__file__))})


def peak_features(raw, frequency, psds, ids, path):
    if path.with_suffix('.COMPLETE.json').exists():
        return np.load(path, mmap_mode='r')
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = mult.t.spectrum_bank()
    a = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                dtype=np.float32, shape=(len(raw), 27, 253))
    progress = path.with_suffix('.progress.json')
    rows = json.loads(progress.read_text()) if progress.exists() else []
    done = {v['psd_index'] for v in rows}
    for pid in np.unique(ids):
        if int(pid) in done:
            continue
        tick = time.perf_counter()
        take = np.flatnonzero(ids == pid)
        bank = mult.t.adaptive.whitened_bank(spectrum, frequency, psds[int(pid)])
        power = mult.t.fine.features(raw[take], bank)
        a[take] = power.reshape(len(take), 3, 253, 3, 3).transpose(0, 1, 3, 4, 2).reshape(len(take), 27, 253)
        if not np.isfinite(a[take]).all():
            raise RuntimeError('Nonfinite peak response')
        a.flush()
        rows.append({'psd_index': int(pid), 'events': len(take), 'seconds': time.perf_counter()-tick})
        n.write_json(progress, rows)
        if len(rows) % 8 == 0:
            print('NEW_PEAK_FEATURES', path.parent.name, len(rows), len(np.unique(ids)), flush=True)
    n.write_json(path.with_suffix('.COMPLETE.json'), {'UTC': n.utc(), 'shape': list(a.shape), 'sha256': n.sha(path)})
    return a


def numerical_test():
    dest = ROOT/'audit/FROZEN_FEATURE_REPLAY.json'
    if dest.exists():
        return
    rows = []
    for dep in n.DEPS:
        base = n.t.PREVIOUS/f'expanded_data/{dep}'
        meta = pd.read_parquet(base/'validation/event_metadata.parquet')
        raw = np.load(base/'validation/raw2s.npy', mmap_mode='r')
        freq, psds = np.load(base/'noise/frequency.npy'), np.load(base/'noise/psd.npy')
        indices = np.array([0, 1, 17, 48])
        pred = peak_features(raw[indices], freq, psds, meta.noise_bank_index.to_numpy(int)[indices], ROOT/f'audit/{dep}/peak_replay.npy')
        expected = d.mult.training_data(d.LOW, dep, 'validation', 'MULTIRATE')[0][indices, :27]
        difference = float(abs(pred-expected).max())
        if difference > 3e-3:
            raise RuntimeError('Frozen feature replay mismatch: '+str(difference))
        rows.append({'deployment': dep, 'max_abs_difference': difference, 'tolerance': .003})
    n.write_json(dest, {'passed': True, 'rows': rows, 'time_sky_not_used': True})


def features(dep):
    base = ROOT/f'data/{dep}'
    if not (base/'COMPLETE.json').exists():
        raise RuntimeError('New data not complete')
    meta = pd.read_parquet(base/'event_metadata.parquet')
    frequency = np.load(base/'noise/frequency.npy')
    psds = np.load(base/'noise/psd.npy')
    ids = meta.noise_bank_index.to_numpy(int)
    peak_features(np.load(base/'raw2s.npy', mmap_mode='r'), frequency, psds, ids, ROOT/f'features/{dep}/peak.npy')
    mult.grouped(ROOT, np.load(base/'low16s.npy', mmap_mode='r'), frequency, psds, ids, ROOT/f'features/{dep}/low.npy')


@torch.no_grad()
def predict(dep):
    x = np.concatenate([np.load(ROOT/f'features/{dep}/peak.npy'), np.load(ROOT/f'features/{dep}/low.npy')], 1)
    if x.shape[1:] != (54, 253):
        raise RuntimeError('Wrong model input')
    base = ROOT/f'data/{dep}'
    meta = pd.read_parquet(base/'event_metadata.parquet')
    raw = np.load(base/'raw2s.npy', mmap_mode='r')
    for seed, slot in zip(n.SEEDS, n.t.MODEL_SLOTS):
        path = ROOT/f'predictions/{dep}/{slot}_parent.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            cp_path = d.LOW/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt'
            cp = torch.load(cp_path, map_location='cpu', weights_only=False)
            model = d.mult.Predictor('MULTIRATE').cuda().eval()
            model.load_state_dict(cp['model'])
            mass, outside = d.mult.old.probability(d.mult.old.infer(model, (x-cp['mu'])/cp['sd']), cp['temperature'])
            ckpath = r.INTR/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt'
            ck = torch.load(ckpath, map_location='cpu', weights_only=False)
            if ck['mass_checkpoint_sha256'] != n.sha(cp_path):
                raise RuntimeError('Mass checkpoint provenance mismatch')
            head = d.intrinsic.Head(ck['conditional_dimensions']).cuda().eval()
            head.load_state_dict(ck['model'])
            joint = np.empty((len(x), 512, ck['conditional_dimensions']), np.float32)
            hi = np.searchsorted(d.CENTERS, d.mult.old.CENTERS, side='right').clip(1, 252)
            lo = hi-1
            w = ((d.mult.old.CENTERS-d.CENTERS[lo])/(d.CENTERS[hi]-d.CENTERS[lo])).clip(0, 1)
            for start in range(0, len(x), 64):
                sl = slice(start, start+64)
                rep = d.intrinsic.original.representations(x[sl], cp)
                residual = d.intrinsic.original.infer(head, ((rep-ck['mu'])/ck['sd']).reshape(-1, 103)).reshape(len(rep), 253, -1)
                conditional = softmax(residual/ck['temperature']+np.log(ck['conditional_prior'].clip(1e-12))[None], -1)
                conditional = (1-w[None, :, None])*conditional[:, lo]+w[None, :, None]*conditional[:, hi]
                joint[sl] = mass[sl, :, None]*conditional
            error = float(abs(joint.sum(-1)-mass).max())
            if error > 1e-6:
                raise RuntimeError('Mass marginal conservation')
            np.savez_compressed(path, p=mass, outside=outside, joint=joint)
            n.write_json(path.with_suffix('.json'), {'UTC': n.utc(), 'mass_checkpoint_sha256': n.sha(cp_path),
                'conditional_checkpoint_sha256': n.sha(ckpath), 'marginal_error': error, 'sha256': n.sha(path)})
            print('NEW_FROZEN_DENSITY', dep, slot, flush=True)
        ep = ROOT/f'predictions/{dep}/{seed}_embedding.npy'
        if not ep.exists():
            cp_path = n.dev.V7/dep/f'seed_{seed}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt'
            v7 = n.dev.MAINCODE.cbase.v7
            encoder, _ = v7.load_unified_model(cp_path)
            rows = []
            for values in DataLoader(v7.ArrayCatalog(raw), batch_size=64, shuffle=False, num_workers=0):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = encoder.base(values.cuda())
                rows.append(z.float().cpu().numpy())
            z = np.concatenate(rows)
            if z.shape != (len(meta), 96) or not np.isfinite(z).all():
                raise RuntimeError('Invalid frozen embedding')
            np.save(ep, z)
            n.write_json(ep.with_suffix('.json'), {'checkpoint_sha256': n.sha(cp_path),
                'head_executed': False, 'output_sha256': n.sha(ep)})
    parents = np.mean([np.load(ROOT/f'predictions/{dep}/{slot}_parent.npz')['p'] for slot in n.t.MODEL_SLOTS], axis=0)
    np.save(ROOT/f'predictions/{dep}/ensemble_mass.npy', parents)
    sm = r.mass_summary(parents)
    meta['parent_predicted_logmc'] = sm[:, 0]
    meta['parent_predicted_halfwidth'] = sm[:, 1]
    meta['deployment'] = dep
    meta['fit_tune_fold'] = meta.fold
    keep = set(meta.source_uid[np.exp(sm[:, 0]) <= 15.])
    selected = meta[meta.source_uid.isin(keep)].copy()
    selected.to_parquet(ROOT/f'contracts/{dep}_PROFILE_EVENT_PLAN.parquet', index=False)
    n.write_json(ROOT/f'contracts/{dep}_PREDICTIONS_COMPLETE.json', {'UTC': n.utc(),
        'events': len(meta), 'profile_events': len(selected), 'profile_sources': len(keep),
        'plan_sha256': n.sha(ROOT/f'contracts/{dep}_PROFILE_EVENT_PLAN.parquet')})


def init_worker(root, dep):
    generation.init(root, dep)
    ctx = generation.STATE
    ctx['oldraw'] = np.load(Path(root)/f'data/{dep}/raw2s.npy', mmap_mode='r')
    prof.replay.CTX.update(ctx)
    prof.profile.base.STATE['v3'] = ctx['v3']


def profile_event(job):
    tick = time.perf_counter()
    row, feature = job
    result = prof.fit_one((row, feature))
    result['wall_seconds'] = time.perf_counter()-tick
    return result


def profile(dep, workers):
    folder = ROOT/f'profile_events/{dep}'
    folder.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(ROOT/f'contracts/{dep}_PROFILE_EVENT_PLAN.parquet')
    x = np.load(ROOT/f'features/{dep}/peak.npy')
    jobs = [(row, x[int(row['row_index'])]) for row in meta.to_dict('records')
            if not (folder/f'{int(row["row_index"])}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker, initargs=(str(ROOT), dep)) as pool:
        for count, result in enumerate(pool.map(profile_event, jobs), 1):
            n.write_json(folder/f'{result["row_index"]}.json', result)
            if count % 32 == 0 or count == len(jobs):
                print('NEW_CALIBRATION_PROFILE', dep, count, len(jobs), flush=True)
    n.write_json(ROOT/f'contracts/{dep}_PROFILES_COMPLETE.json', {'UTC': n.utc(), 'events': len(meta),
        'fit_tune_only': True, 'real_or_test_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'features', 'predict', 'profile'), required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    ROOT = args.root
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'features':
        numerical_test()
        for dep in n.DEPS:
            features(dep)
    elif args.stage == 'predict':
        for dep in n.DEPS:
            predict(dep)
    else:
        for dep in n.DEPS:
            profile(dep, args.workers)
