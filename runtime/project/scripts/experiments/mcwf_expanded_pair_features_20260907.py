#!/usr/bin/env python3
"""Frozen retrained waveform latents for source-disjoint pair-verifier ablation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json
import numpy as np
import torch
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_joint_source_density_20260907 as source

dev = e.dev


@torch.no_grad()
def hidden(root, dep, ms, es, split):
    cp = root / f'joint_source_density/models/{dep}/seed_{ms}/validation_selected_model.pt'
    suffix = split if split in ('train', 'development', 'real') else f'{es}_{split}'
    path = root / f'expanded_pair_verifier/features/{dep}/seed_{ms}/{suffix}.npy'
    if path.exists():
        if json.loads(path.with_suffix('.json').read_text())['checkpoint_sha256'] != dev.sha(cp):
            raise RuntimeError('Changed source encoder')
        return np.load(path, mmap_mode='r')
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    if split in ('train', 'development'):
        name = 'validation' if split == 'development' else split
        raw = np.load(root / f'expanded_data/{dep}/{name}/raw2s.npy', mmap_mode='r')
        x = np.load(root / f'expanded_encoder/features/{dep}/{name}.npy')
    elif split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        raw = dev.TRAIN.make_window_view(np.asarray(full[valid], np.float32), 2)
        x = np.load(e.TRAINED / f'cache/deployment_event_psd/{dep}/real_features.npy')
    else:
        plan = dev.BASE.retained_event_plan(dep, es, split)
        full = dev.ORCH.event_array_for_plan(dep, es, split, plan)
        raw = dev.TRAIN.make_window_view(np.asarray(full, np.float32), 2)
        x = np.load(e.TRAINED / f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy')
    model = source.JointSource().cuda().eval()
    model.load_state_dict(ck['model'])
    values = []
    for start in range(0, len(x), 128):
        xx = torch.as_tensor((x[start:start + 128] - ck['mu']) / ck['sd'], dtype=torch.float32, device='cuda')
        rr = torch.as_tensor(np.asarray(raw[start:start + 128]), dtype=torch.float32, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            phase = model.phase.layers(xx)
            r = model.raw(rr)
            merged = model.merge(torch.cat([phase, r], dim=1))
        values.append(torch.cat([phase, r, merged], dim=1).float().cpu().numpy())
    a = np.concatenate(values)
    if a.shape != (len(x), 352) or not np.isfinite(a).all():
        raise RuntimeError('Invalid physical-waveform latent array')
    if split == 'real':
        padded = np.full((len(full), 352), np.nan, dtype=np.float32)
        padded[valid] = a
        a = padded
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)
    dev.json_write(path.with_suffix('.json'), {'checkpoint_sha256': dev.sha(cp),
        'shape': list(a.shape), 'split': split, 'event_seed': es, 'array_sha256': dev.sha(path)})
    print(json.dumps({'expanded_pair_features': str(path), 'events': len(a)}), flush=True)
    return np.load(path, mmap_mode='r')
