#!/usr/bin/env python3
"""Reuse audited NEW-SCORE-ONLY component implementations on each new arm/run.

The archived adapter's gwtc5 directory key is only an internal storage alias.
No old model checkpoint is used to initialize these six deployments.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


def setup(root, run, arm):
    sys.path.insert(0, str(root/'scripts'))
    import unified_ab as u
    sys.path.insert(0, str(u.P))
    sys.path.insert(0, str(u.P/'scripts/experiments'))
    sys.path.insert(0, str(root/'scripts/archived_adapters'))
    import o4b_hl_nso_data_training_20260912 as support
    import o4b_hl_nso_multiscale_training_20260912 as models
    import o4b_hl_nso_rankncontrast_v2_20260912 as rnc
    current = root/'arms'/arm/run
    for name in ('contracts', 'logs', 'tables', 'scripts'):
        (current/name).mkdir(parents=True, exist_ok=True)
    shared = current/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    shared.mkdir(parents=True, exist_ok=True)
    for name in ('noise_reference_bank.npy', 'noise_psd_bank.npy', 'noise_psd_frequency.npy'):
        link = shared/name
        target = root/'plans'/run/name
        if link.exists():
            if link.resolve() != target:
                raise RuntimeError('Noise alias points to another deployment')
        else:
            link.symlink_to(target)
    scripts = current/'workspace/scripts'
    if not scripts.exists():
        scripts.symlink_to(u.P/'scripts', target_is_directory=True)
    models.SEEDS = rnc.SEEDS = u.SEEDS
    support.disk_guard = lambda _: u.guard(root)
    u.write(current/'contracts/UAB_COMPONENT_ADAPTER.json', {
        'actual_run': run, 'arm': arm, 'internal_directory_key': 'gwtc5',
        'internal_key_is_not_a_claim_of_O4b_data': True,
        'authoritative_contract': str(root/'contracts/ANALYSIS_CONTRACT.json'),
        'archive_descriptive_O4b_labels_overridden_by_this_contract': True,
        'short_initialized_from_scratch': True,
        'RNC_initialization': 'same arm/run/seed newly trained short encoder only',
        'other_architectures': 'same ordered Mc, multirate and conditional eta/chi implementation',
        'new_model_seeds': u.SEEDS, 'old_score_mixing': False,
        'not_full_experiment_results': True})
    return u, current, models, rnc


def export_aux(root, run, arm, split):
    u, current, _, _ = setup(root, run, arm)
    target = current/'auxiliary_data_v2'/split
    marker = target/'COMPLETE.json'
    if marker.exists():
        for name, value in json.loads(marker.read_text())['sha256'].items():
            if u.sha(target/name) != value:
                raise RuntimeError('Auxiliary input changed')
        return
    if not (root/'contracts'/f'{run}_aux_{split}_COMPLETE.json').exists():
        raise RuntimeError('Both-arm auxiliary generation incomplete')
    target.mkdir(parents=True, exist_ok=True)
    sources = pd.read_parquet(root/'plans'/run/'sources.parquet')
    sources = sources[(sources.role == 'aux') & (sources.split == split)].sort_values('source_index')
    n = len(sources)*(8 if split == 'train' else 2)
    raw = np.lib.format.open_memmap(target/'raw2s.npy', mode='w+', dtype=np.float32, shape=(n, 2, 4096))
    low = np.lib.format.open_memmap(target/'low16s.npy', mode='w+', dtype=np.float32, shape=(n, 2, 4096))
    offset, meta = 0, []
    for source in sources.to_dict('records'):
        folder = root/'data'/run/'aux'/split/source['source_uid']
        if not (folder/'COMPLETE.json').exists():
            raise RuntimeError('Auxiliary source incomplete')
        frame = pd.read_parquet(folder/'metadata.parquet')
        frame = frame[frame.arm == arm].copy()
        x = np.load(folder/f'{arm}_short.npy', mmap_mode='r')
        y = np.load(folder/f'{arm}_long.npy', mmap_mode='r')
        if len(frame) != len(x):
            raise RuntimeError('Auxiliary metadata/input length mismatch')
        raw[offset:offset+len(x)] = x
        low[offset:offset+len(y)] = y
        frame['row_index'] = np.arange(offset, offset+len(frame))
        frame['waveform_parent_uid'] = frame.source_uid
        # A source waveform and a reused lens environment are different units.
        frame['lens_system_group_id'] = frame.global_source_id
        frame['global_source_id'] = frame.source_uid
        frame['chirp_mass_detector'] = frame.mc_det
        meta.append(frame)
        offset += len(frame)
    if offset != n:
        raise RuntimeError('Incomplete auxiliary views')
    raw.flush(); low.flush()
    frame = pd.concat(meta, ignore_index=True)
    frame.to_parquet(target/'event_metadata.parquet', index=False)
    u.write(marker, {'utc': u.now(), 'events': n, 'waveform_parents': frame.source_uid.nunique(),
                     'independent_lens_groups': frame.lens_system_group_id.nunique(),
                     'sha256': {name: u.sha(target/name) for name in ('raw2s.npy', 'low16s.npy', 'event_metadata.parquet')}})


def main(args):
    u, current, models, rnc = setup(args.root, args.run, args.arm)
    u.verify(args.root)
    import torch
    torch.set_num_threads(2)
    if args.stage == 'export':
        export_aux(args.root, args.run, args.arm, args.split)
    elif args.stage == 'features':
        export_aux(args.root, args.run, args.arm, args.split)
        models.features(current, args.split)
    elif args.stage == 'rnc-prepare':
        rnc.prepare(current)
    elif args.stage == 'rnc':
        rnc.train(current, args.seed)
    else:
        models.train(current, args.stage, args.seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', choices=['O3', 'O4a', 'O4b'], required=True)
    p.add_argument('--arm', choices=['C_PHYSICAL'], required=True)
    p.add_argument('--stage', choices=['export', 'features', 'rnc-prepare', 'rnc', 'ordered', 'multirate', 'conditional'], required=True)
    p.add_argument('--split', choices=['train', 'validation'], default='validation')
    p.add_argument('--seed', type=int, default=2026091721)
    main(p.parse_args())
