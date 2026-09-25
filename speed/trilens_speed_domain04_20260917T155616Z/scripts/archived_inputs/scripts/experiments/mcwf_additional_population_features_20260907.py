#!/usr/bin/env python3
"""Audit the new training cohort and reuse frozen waveform feature definitions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_expanded_encoder_20260907 as coarse
import mcwf_dense_features_20260907 as dense
import mcwf_fine_mass_features_20260907 as fine

dev = e.dev


def run(root, dep):
    child = root / 'additional_population'
    src = child / f'expanded_data/{dep}/train'
    if not (src / 'COMPLETE.json').exists():
        raise RuntimeError('Additional cohort generation incomplete')
    out = child / f'features/{dep}'
    out.mkdir(parents=True, exist_ok=True)
    n = pd.read_parquet(src / 'source_plan.parquet')
    t = pd.read_parquet(root / f'expanded_data/{dep}/train/source_plan.parquet')
    v = pd.read_parquet(root / f'expanded_data/{dep}/validation/source_plan.parquet')
    cols = ['m1_det', 'm2_det', 'a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi12', 'phijl', 'phase', 'ra', 'dec']
    hashes = [set(pd.util.hash_pandas_object(f[cols], index=False)) for f in (n, t, v)]
    overlaps = {'new_vs_previous_train_ids': len(set(n.source_uid) & set(t.source_uid)),
                'new_vs_validation_ids': len(set(n.source_uid) & set(v.source_uid)),
                'new_vs_previous_train_physics': len(hashes[0] & hashes[1]),
                'new_vs_validation_physics': len(hashes[0] & hashes[2])}
    noise = child / f'expanded_data/{dep}/noise'
    nb = pd.read_csv(noise / 'noise_manifest.csv')
    ob = pd.read_csv(root / f'expanded_data/{dep}/noise/noise_manifest.csv')
    collision = sum(bool(((r.start_gps < ob.end_gps + 16) & (r.end_gps > ob.start_gps - 16)).any()) for r in nb.itertuples())
    overlaps['source_noise_block_GPS_overlaps_with_guard16s'] = collision
    if any(overlaps.values()):
        raise RuntimeError('Source/noise overlap: ' + str(overlaps))
    dev.json_write(out / 'DATA_SCALING_GATE.json', {'pass': True, **overlaps,
        'additional_sources': len(n), 'combined_train_sources': len(t) + len(n), 'development_validation_sources': len(v),
        'additional_noise_blocks': len(nb), 'combined_train_noise_blocks': len(nb) + int((ob.split == 'train').sum()),
        'new_train_source_sha256': dev.sha(src / 'source_plan.parquet'),
        'old_validation_source_sha256': dev.sha(root / f'expanded_data/{dep}/validation/source_plan.parquet'),
        'new_noise_sha256': dev.sha(noise / 'noise_manifest.csv'),
        'limits': 'lens environments may recur; old validation is reused development, not fresh confirmation; O3 response epochs retain old cumulative calendar while noise is O3-only'})
    coarse.features(child, dep, 'train')
    meta = pd.read_parquet(src / 'event_metadata.parquet')
    raw = np.load(src / 'raw2s.npy', mmap_mode='r')
    frequency = np.load(noise / 'frequency.npy')
    psds = np.load(noise / 'psd.npy', mmap_mode='r')
    ids = meta.noise_bank_index.to_numpy(int)
    dense.grouped(root, raw, frequency, psds, ids, out / 'dense.npy')
    fine.grouped(root, raw, frequency, psds, ids, out / 'fine.npy')
    dev.json_write(out / 'COMPLETE.json', {'coarse_sha256': dev.sha(child / f'expanded_encoder/features/{dep}/train.npy'),
        'dense_sha256': dev.sha(out / 'dense.npy'), 'fine_sha256': dev.sha(out / 'fine.npy'), 'events': len(meta)})
    print(json.dumps({'additional_features_complete': dep, 'events': len(meta)}), flush=True)


def inputs(root, dep, split):
    from mcwf_fine_mass_density_20260907 import inputs as original
    old = original(root, dep, split)
    if split != 'train':
        return old
    child = root / 'additional_population'
    folder = child / f'features/{dep}'
    if not (folder / 'COMPLETE.json').exists():
        raise RuntimeError('Additional features incomplete')
    a = np.load(child / f'expanded_encoder/features/{dep}/train.npy', mmap_mode='r')
    b = np.load(folder / 'dense.npy', mmap_mode='r')
    c = np.load(folder / 'fine.npy', mmap_mode='r')
    merged = np.empty((len(old) + len(a), old.shape[1]), dtype=np.float32)
    merged[:len(old)] = old
    for start in range(0, len(a), 2048):
        s = slice(start, start + 2048)
        part = [x[s].reshape(len(x[s]), -1) for x in (a, b, c)]
        merged[len(old) + start:len(old) + start + len(part[0])] = np.concatenate(part, axis=1)
    return merged


def metadata(root, dep, split):
    old = pd.read_parquet(root / f'expanded_data/{dep}/{split}/event_metadata.parquet')
    old['training_cohort'] = 'original_expanded'
    if split != 'train':
        return old
    new = pd.read_parquet(root / f'additional_population/expanded_data/{dep}/train/event_metadata.parquet')
    new['training_cohort'] = 'additional_8192'
    new['noise_bank_index'] += 1000
    out = pd.concat([old, new], ignore_index=True)
    if out.source_uid.nunique() != 12288:
        raise RuntimeError('Unexpected global source denominator')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deployment', choices=e.DEPS, required=True)
    args = p.parse_args()
    run(args.root, args.deployment)
