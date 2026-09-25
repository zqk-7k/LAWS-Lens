#!/usr/bin/env python3
"""Read-only encoder feature maps, before global temporal pooling."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import mcwf_pe_frontend_extension_v2_20260907 as e

dev, body = e.dev, e.body
torch.set_num_threads(2)


@torch.no_grad()
def features(root, dep, ms, es, split):
    cp = root / f'large_population_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt'
    suffix = split if split in ('train', 'development', 'real') else f'{es}_{split}'
    folder = root / f'temporal_pair/features/{dep}/seed_{ms}/{suffix}'
    if (folder / 'COMPLETE.json').exists():
        info = json.loads((folder / 'COMPLETE.json').read_text())
        if info['checkpoint_sha256'] != dev.sha(cp):
            raise RuntimeError('Frozen backbone changed')
        return np.load(folder / 'tokens.npy', mmap_mode='r'), np.load(folder / 'global.npy')
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Backbone not complete')
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    valid = None
    if split in ('train', 'development'):
        name = 'validation' if split == 'development' else split
        raw = np.load(root / f'expanded_data/{dep}/{name}/raw2s.npy', mmap_mode='r')
        x = np.load(root / f'expanded_encoder/features/{dep}/{name}.npy', mmap_mode='r')
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
    if len(x) != len(raw) or raw.shape[1:] != (2, 4096):
        raise RuntimeError('Input shape mismatch')
    folder.mkdir(parents=True, exist_ok=True)
    n = len(valid) if valid is not None else len(raw)
    tokens = np.lib.format.open_memmap(folder / 'tokens.npy', mode='w+', dtype=np.float16, shape=(n, 64, 128))
    glob = np.full((n, 192), np.nan, np.float32)
    if valid is not None:
        tokens[:] = np.nan
    indices = np.flatnonzero(valid) if valid is not None else np.arange(n)
    model = body.Encoder('RAW-PHASE-SOURCE').cuda().eval()
    model.load_state_dict(ck['model'])
    started = time.perf_counter()
    max_error = 0.
    for start in range(0, len(raw), 128):
        rr = torch.as_tensor(np.asarray(raw[start:start+128]), dtype=torch.float32, device='cuda')
        xx = torch.as_tensor((np.asarray(x[start:start+128])-ck['mu'])/ck['sd'], dtype=torch.float32, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model.raw.stem(rr)
            residual = y
            for k, block in enumerate(model.raw.blocks):
                y = block(y)
                if k % 3 == 2:
                    y = model.raw.res_act(y+model.raw.shortcuts[k](residual))
                    residual = y
            t = F.adaptive_avg_pool1d(y, 64).transpose(1, 2)
            weights = torch.softmax(model.raw.attn(y), dim=-1)
            readout = model.raw.head(torch.cat([(y*weights).sum(-1), y.mean(-1), y.amax(-1)], 1))
            readout = F.normalize(readout, dim=-1)
            if start == 0:
                max_error = float((readout-model.raw(rr)).abs().max())
                if max_error > .002:
                    raise RuntimeError('Feature-hook path changes backbone output')
            phase = model.phase.layers(xx)
            hidden = model.merge(torch.cat([phase, readout], 1))
            logits = model.phase.head(hidden)
            z = F.normalize(model.embedding(hidden), dim=-1)
        pp = torch.softmax(logits.float()/ck['temperature'], dim=-1)
        ids = indices[start:start+len(rr)]
        tokens[ids] = t.float().cpu().numpy().astype(np.float16)
        glob[ids] = torch.cat([z.float(), pp], -1).cpu().numpy()
    tokens.flush()
    if not np.isfinite(tokens[indices]).all() or not np.isfinite(glob[indices]).all():
        raise RuntimeError('Nonfinite valid waveform features')
    np.save(folder / 'global.npy', glob)
    info = {'checkpoint_sha256': dev.sha(cp), 'shape': list(tokens.shape),
            'global_shape': list(glob.shape), 'valid_events': len(indices),
            'backbone_output_max_error': max_error, 'seconds': time.perf_counter()-started,
            'token_sha256': dev.sha(folder / 'tokens.npy'), 'global_sha256': dev.sha(folder / 'global.npy'),
            'input': 'unchanged peak2s4096 H1L1,40-580Hz; noPE/time/sky features',
            'statistical_unit': 'event feature reused by all pair evaluations for this frozen model'}
    dev.json_write(folder / 'COMPLETE.json', info)
    print(json.dumps({'temporal_features': str(folder), **info}), flush=True)
    return tokens, glob


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    for dep in e.DEPS:
        for ms in body.MODEL_SEEDS:
            for split in ('train', 'development'):
                features(a.root, dep, ms, 0, split)
