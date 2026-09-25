#!/usr/bin/env python3
"""Extend the frozen low-band branch to all existing training parents."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_multirate_features_20260908 as low
import mcwf_multirate_train_20260908 as mult
import mcwf_temporal_response_evaluate_20260908 as ev
t, old, dev = low.t, low.old, low.dev
UPSTREAM = P / 'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
ADDITIONAL = t.PREVIOUS / 'additional_population'
KINDS = ('LARGE-CONTROL', 'MULTIRATE')
SEEDS = (202610021, 202610022, 202610023)
ORIGINAL_DATA = mult.training_data


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'scripts', 'logs', 'cache', 'features', 'models', 'predictions',
                 'calibration', 'evaluation', 'tables', 'reports', 'manifest', 'figures'):
        (root / name).mkdir(parents=True)
    contract = {
        'id': 'MCWF-LARGE-MULTIRATE-24', 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'same_algorithm_both_runs': True,
        'hypothesis': 'The low20-80Hz16s auxiliary currently sees4096parents; recover the same branch for8192existing additional parents without changing their original peak2s waveforms.',
        'arms': KINDS, 'training_seeds': SEEDS,
        'data': '12288independent simulated source parents,160offsource noise blocks,8views/source;512development sources/32noise blocks unchanged.',
        'control': 'Both arms warm the archived12288parent OMC mass predictor and receive identical15epoch source-balanced training. New lowband weights initially zero.',
        'population_comparison': 'Same model/loss/15epoch recipe as06;source-count changes the number ofoptimizer steps. Within24 both arms have identical populations andsteps.',
        'replay': 'Reconstruct original physical H1L1 detector response,PSD target-SNR scaling andactualnoise;require exact archivedfloat16 peak2s BEFORE accepting a new20-80Hz16s view.',
        'pilot': 'First64events perrun must pass bit-exactreplay;then process all65536additional views perrun. Existing06lowbranch reusedread-only.',
        'conditioning': 'Unchanged low06 whitening/filter/anti-alias/lag-featureoperator;no interpolation of old40Hzstrain to create missinglowfrequencies.',
        'selection': 'Same06 developmentCE/temperature andsimulationvalidation-only paircoefficient selection;bothCANDIDATE/RETRIEVAL reported;realmetrics onlyafterfreeze.',
        'data_audit': 'SourceUID intersections andGPSinterval overlap,not equality oflocalnoisebankindices acrossdifferentbanks.',
        'frozen': ['time_score', 'sky_raw_log_bf', 'C-fixed outerweights', 'scope', 'originalmodels', 'history', 'paper'],
        'changed_channels': ['waveform'], 'outer_weights_changed': False,
        'disk': {'minimum_free_GiB': 25, 'maximum_new_cache_GiB': 20},
        'adaptive_development': True, 'fresh_confirmation_required_before_upgrade': True,
        'limitations': 'Existing simulationpopulation reused;predictive mass densities are not physicalPE. No guarantee officialcandidate overlaps canimprove.'}
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    audits = []
    protected = t.protected()
    for dep in t.DEPS:
        base = t.PREVIOUS / f'expanded_data/{dep}'
        extra = ADDITIONAL / f'expanded_data/{dep}'
        train = pd.read_parquet(base / 'train/event_metadata.parquet')
        val = pd.read_parquet(base / 'validation/event_metadata.parquet')
        new = pd.read_parquet(extra / 'train/event_metadata.parquet')
        source_intersections = [len(set(a.source_uid) & set(b.source_uid))
                                for a, b in ((train, val), (new, val), (train, new))]
        first = pd.read_csv(base / 'noise/noise_manifest.csv')
        second = pd.read_csv(extra / 'noise/noise_manifest.csv')
        allnoise = pd.concat([first.assign(bank_origin='expanded'), second.assign(bank_origin='additional')], ignore_index=True)
        start, end = allnoise.start_gps.to_numpy(), allnoise.end_gps.to_numpy()
        collision = (start[:, None] < end[None, :]) & (start[None, :] < end[:, None])
        overlaps = int(np.triu(collision, 1).sum())
        if any(source_intersections) or overlaps:
            raise RuntimeError(f'Source/noise overlap: {dep}, {source_intersections}, {overlaps}')
        audits.append({'deployment': dep, 'train_sources': train.source_uid.nunique() + new.source_uid.nunique(),
                       'development_sources': val.source_uid.nunique(), 'source_intersections': source_intersections,
                       'total_train_noise_blocks': int((allnoise.split == 'train').sum()),
                       'development_noise_blocks': int((allnoise.split == 'validation').sum()),
                       'GPS_interval_overlaps': overlaps, 'all_noise_blocks': len(allnoise)})
        dev.csv_write(root / f'tables/{dep}_NOISE_SOURCE_INVENTORY.csv', allnoise)
        for path in (base/'train/event_metadata.parquet', base/'validation/event_metadata.parquet',
                     extra/'train/event_metadata.parquet', extra/'noise/noise_manifest.csv'):
            protected.append({'path': str(path), 'sha256': dev.sha(path), 'bytes': path.stat().st_size})
    dev.json_write(root / 'contracts/SOURCE_NOISE_AUDIT.json', audits)
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(protected))
    shutil.copy2(__file__, root / 'scripts/large_multirate.py')
    shutil.copy2(UPSTREAM / 'cache/lowband_aligned_spectra.npy', root / 'cache/lowband_aligned_spectra.npy')
    dev.json_write(root / 'contracts/START_FREEZE.json', {
        'contract_sha256': dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'), 'code_sha256': dev.sha(Path(__file__))})


def init_additional(dep):
    src, data, v3 = low.expanded.modules()
    import logging
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    logging.getLogger('bilby').setLevel(logging.ERROR)
    base = ADDITIONAL / f'expanded_data/{dep}'
    low.CTX.update(src=src, v3=v3, generator=src.build_waveform_generator(),
        ifos=[src.bilby.gw.detector.get_empty_interferometer(d) for d in ('H1', 'L1')],
        refs=np.load(base/'noise/reference.npy', mmap_mode='r'), freq=np.load(base/'noise/frequency.npy'),
        psds=np.load(base/'noise/psd.npy', mmap_mode='r'), oldraw=np.load(base/'train/raw2s.npy', mmap_mode='r'))


def raw(root, dep, workers):
    out = root / f'cache/additional/{dep}'
    if (out/'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(ADDITIONAL/f'expanded_data/{dep}/train/event_metadata.parquet')
    path = out / 'low16s.npy'
    values = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float32,
                                     shape=(len(meta), 2, low.LENGTH))
    progress = out / 'PROGRESS.parquet'
    rows = pd.read_parquet(progress).to_dict('records') if progress.exists() else []
    done = {int(r['row_index']) for r in rows}
    started = time.perf_counter()
    # The pilot is part of the same fixed training data, not a sample selected by outcomes.
    for start, stop in ((0, 64), (64, len(meta))):
        pending = meta.iloc[start:stop]
        pending = pending[~pending.row_index.isin(done)].to_dict('records')
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(pending))), mp_context=mp.get_context('spawn'),
                                 initializer=init_additional, initargs=(dep,)) as pool:
            for count, (idx, waveform, scales) in enumerate(pool.map(low.reconstruct, pending, chunksize=4), 1):
                values[idx] = waveform
                rows.append({'row_index': idx, 'old_peak2s_exact': True,
                             'pre_window_std_H1': float(scales[0]), 'pre_window_std_L1': float(scales[1])})
                done.add(idx)
                if count % 256 == 0 or count == len(pending):
                    if shutil.disk_usage(root).free < 25*2**30:
                        raise RuntimeError('HOLD_DISK_LIMIT')
                    values.flush()
                    pd.DataFrame(rows).to_parquet(progress, index=False)
                    print(json.dumps({'replay': dep, 'completed': len(rows), 'total': len(meta),
                                      'seconds': time.perf_counter()-started}), flush=True)
        if stop == 64:
            dev.json_write(out/'PILOT_PASS.json', {'pass': all(i in done for i in range(64)),
                                                'events': 64, 'old2s_exact': True})
    values.flush()
    if len(done) != len(meta):
        raise RuntimeError('Incomplete replay')
    dev.json_write(out/'COMPLETE.json', {'events': len(meta), 'source_parents': meta.source_uid.nunique(),
        'old2s_replay_all_exact': True, 'sha256': dev.sha(path), 'seconds': time.perf_counter()-started})


def feature_stage(root, dep):
    out = root / f'cache/additional/{dep}'
    if not (out/'COMPLETE.json').exists():
        raise RuntimeError('Missing raw replay receipt')
    base = ADDITIONAL / f'expanded_data/{dep}'
    meta = pd.read_parquet(base/'train/event_metadata.parquet')
    low.grouped(root, np.load(out/'low16s.npy', mmap_mode='r'), np.load(base/'noise/frequency.npy'),
        np.load(base/'noise/psd.npy', mmap_mode='r'), meta.noise_bank_index.to_numpy(int),
        root/f'features/additional/{dep}/train.npy')


def training_data(root, dep, split, kind):
    x, meta = ORIGINAL_DATA(UPSTREAM, dep, split, kind)
    meta = meta.assign(noise_origin='expanded')
    if split == 'validation':
        return x, meta
    coarse = np.load(ADDITIONAL/f'expanded_encoder/features/{dep}/train.npy', mmap_mode='r')
    fine = np.load(ADDITIONAL/f'features/{dep}/fine.npy', mmap_mode='r')
    extra = old.arrange(coarse, fine)
    if kind == 'MULTIRATE':
        extra = np.concatenate([extra, np.load(root/f'features/additional/{dep}/train.npy', mmap_mode='r')], 1)
    em = pd.read_parquet(ADDITIONAL/f'expanded_data/{dep}/train/event_metadata.parquet').assign(noise_origin='additional')
    # Give separate reference banks distinct labels; physical independence is GPS-audited above.
    em['noise_bank_index'] += 128
    return np.concatenate([x, extra]), pd.concat([meta, em], ignore_index=True)


def prediction(root, dep, kind, slot, seed, split):
    path = root/f'predictions/{kind}/{dep}/model_{slot}_eval_{seed}/{split}.npz'
    if path.exists():
        return np.load(path)
    cp = root/f'models/{kind}/{dep}/seed_{slot}/selected.pt'
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    x = t.old_features(dep, seed, split)
    if kind == 'MULTIRATE':
        file = 'real.npy' if split == 'real' else f'{seed}_{split}.npy'
        x = np.concatenate([x, np.load(UPSTREAM/f'features/{dep}/{file}')], 1)
    model = mult.Predictor(kind).cuda().eval()
    model.load_state_dict(ck['model'])
    prob, outside = old.probability(old.infer(model, (x-ck['mu'])/ck['sd']), ck['temperature'])
    if split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        a, b = np.full((len(full), 512), np.nan), np.full(len(full), np.nan)
        a[valid], b[valid] = prob, outside
        prob, outside = a, b
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=prob, outside=outside, checkpoint_sha256=dev.sha(cp))
    return np.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('initialize', 'raw', 'features', 'train', 'select', 'evaluate', 'real', 'assess'), required=True)
    parser.add_argument('--deployment', choices=t.DEPS)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    mult.KINDS = KINDS
    mult.training_data = training_data
    t.predict = prediction
    if args.stage == 'initialize':
        initialize(args.root)
    elif args.stage == 'raw':
        raw(args.root, args.deployment, args.workers)
    elif args.stage == 'features':
        feature_stage(args.root, args.deployment)
    elif args.stage == 'train':
        for dep in t.DEPS:
            for kind in KINDS:
                for slot, seed in zip(t.MODEL_SLOTS, SEEDS):
                    mult.train_one(args.root, dep, kind, slot, seed)
    elif args.stage == 'select':
        mult.select(args.root)
    else:
        getattr(ev, args.stage)(args.root)
