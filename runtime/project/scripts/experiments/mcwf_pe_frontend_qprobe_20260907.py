#!/usr/bin/env python3
"""Simulation-only mass-ratio probe on frozen waveform representations."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax
import mcwf_pe_frontend_extension_v2_20260907 as e

dev, body, ev = e.dev, e.body, e.ev
torch.set_num_threads(2)
QGRID = np.linspace(.05, 1., 20)
EPOCHS = 50


def initialize(root):
    out = root / 'qprobe'
    p = out / 'contracts/TRAINING.json'
    if p.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    dev.json_write(p, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'mechanism': 'freeze current RNC encoder; learn mass-ratio predictive density from frozen raw/phase/merged waveform features',
        'feature_size': 352, 'network': '352->128 SiLU dropout0.1 ->20 q bins',
        'target': 'simulation q=m2/m1, Gaussian soft target sigma0.06 in linear q',
        'seed_mapping': dict(zip(map(str, dev.SEEDS), [202609071, 202609072, 202609073])),
        'optimizer': {'AdamW_lr': .001, 'weight_decay': .001, 'epochs': EPOCHS, 'source_batch': 64, 'views_per_source': 2, 'passes': 4},
        'selection': 'earliest minimum source-balanced validation q cross-entropy, temperature by validation CE',
        'unchanged': ['all old and RNC model parameters', 'Mc probabilities', 'time', 'sky', 'outer C-fixed weights'],
        'PE_and_official_labels_used_in_training': False,
        'trial_interpretation': 'adaptive waveform development, not physical PE or lens truth',
        'same_O3_O4a': True})
    (out / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, out / 'scripts' / Path(__file__).name)


def head():
    return nn.Sequential(nn.Linear(352, 128), nn.SiLU(), nn.Dropout(.1), nn.Linear(128, len(QGRID)))


@torch.no_grad()
def hidden_features(dep, ms, x, raw):
    cp = e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    model = body.Encoder('RAW-PHASE-SOURCE').cuda().eval()
    model.load_state_dict(ck['model'])
    values = []
    for start in range(0, len(x), 128):
        xx = torch.as_tensor((x[start:start+128] - ck['mu']) / ck['sd'], dtype=torch.float32, device='cuda')
        rr = torch.as_tensor(np.asarray(raw[start:start+128]), dtype=torch.float32, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            phase = model.phase.layers(xx)
            rw = model.raw(rr)
            merged = model.merge(torch.cat([phase, rw], dim=1))
        values.append(torch.cat([phase, rw, merged], dim=1).float().cpu().numpy())
    return np.concatenate(values)


def targets(meta):
    q = meta.mass_2_detector.to_numpy(float) / meta.mass_1_detector.to_numpy(float)
    if not np.all((q > 0) & (q <= 1)):
        raise RuntimeError('Invalid ordered component masses')
    p = softmax(-.5 * ((q[:, None] - QGRID) / .06)**2, axis=1).astype('float32')
    return q, p


def train(root, dep, ms, qs):
    initialize(root)
    out = root / f'qprobe/models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in ('train', 'validation'):
        feature = out / f'{split}_hidden.npy'
        if not feature.exists():
            x = np.load(e.TRAINED / f'cache/finelag/{dep}/{split}_features.npy')
            raw = np.load(e.TRAINED / f'cache/{dep}/{split}_raw2s.npy', mmap_mode='r')
            np.save(feature, hidden_features(dep, ms, x, raw))
        arrays[split] = np.load(feature)
    mu, sd = arrays['train'].mean(0), arrays['train'].std(0).clip(.05)
    tx = (arrays['train'] - mu) / sd
    vx = (arrays['validation'] - mu) / sd
    meta = pd.read_parquet(e.TRAINED / f'cache/{dep}/train_metadata.parquet')
    vm = pd.read_parquet(e.TRAINED / f'cache/{dep}/validation_metadata.parquet')
    tq, y = targets(meta)
    vq, vy = targets(vm)
    group, names = pd.factorize(meta.waveform_parent_uid, sort=True)
    members = [np.flatnonzero(group == k) for k in range(len(names))]
    prior = np.stack([y[group == k].mean(0) for k in range(len(names))]).mean(0)
    prior = prior.clip(1e-6); prior /= prior.sum()
    dev.TRAIN.seed_everything(qs)
    model = head().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    history, best = [], float('inf')
    xt = torch.as_tensor(tx, device='cuda')
    yt = torch.as_tensor(y, device='cuda')
    xv = torch.as_tensor(vx, device='cuda')
    yv = torch.as_tensor(vy, device='cuda')
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(qs + epoch)
        losses = []
        for repeat in range(4):
            order = rng.permutation(len(names))
            for first in range(0, len(order), 64):
                ids = np.concatenate([rng.choice(members[k], 2, replace=False) for k in order[first:first+64]])
                opt.zero_grad(set_to_none=True)
                loss = -(F.log_softmax(model(xt[ids]), -1) * yt[ids]).sum(-1).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        with torch.no_grad():
            l = model(xv)
            ce = float(-(F.log_softmax(l, -1) * yv).sum(-1).mean())
            pred = softmax(l.cpu().numpy(), 1) @ QGRID
        history.append({'epoch': epoch, 'training_ce': np.mean(losses), 'validation_ce': ce,
                        'validation_q_MAE': np.abs(pred-vq).mean(), 'seconds_elapsed': time.perf_counter()-start})
        if ce < best:
            best = ce
            torch.save({'model': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 'mu': mu, 'sd': sd,
                        'q_prior': prior, 'epoch': epoch, 'temperature': 1., 'frozen_encoder': str(e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'),
                        'probe_seed': qs, 'original_RNC_unchanged': True}, out / 'validation_selected_probe.pt')
    ck = torch.load(out / 'validation_selected_probe.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model']); model.eval()
    with torch.no_grad():
        logits = model(xv).cpu().numpy()
    temps = []
    for t in (.5, .75, 1., 1.25, 1.5, 2., 3., 4.):
        lp = logits.astype(float)/t
        lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
        temps.append({'temperature': t, 'ce': float(-(vy*lp).sum(-1).mean())})
    ck['temperature'] = min(temps, key=lambda r:(r['ce'], abs(r['temperature']-1)))['temperature']
    torch.save(ck, out / 'validation_selected_probe.pt')
    np.savez_compressed(out / 'validation_predictions.npz', p=softmax(logits/ck['temperature'], axis=1), q_true=vq, group=vm.waveform_parent_uid.to_numpy(str))
    dev.csv_write(out / 'history.csv', pd.DataFrame(history))
    dev.csv_write(out / 'temperatures.csv', pd.DataFrame(temps))
    pred = softmax(logits / ck['temperature'], 1) @ QGRID
    result = {'dep': dep, 'seed': ms, 'q_seed': qs, 'epoch': ck['epoch'], 'q_MAE': float(np.abs(pred-vq).mean()),
              'constant_prior_MAE': float(np.abs(prior @ QGRID-vq).mean()), 'temperature': ck['temperature'], 'seconds': time.perf_counter()-start}
    dev.json_write(out / 'COMPLETE.json', result)
    print(json.dumps({'probe_trained': result}), flush=True)


@torch.no_grad()
def prediction(root, dep, ms, es, split):
    folder = root / f'qprobe/predictions/{dep}/model_{ms}_eval_{es}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{split}.npy'
    if path.exists():
        return np.load(path)
    if split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        values = np.asarray(full[valid])
        feature = e.TRAINED / f'cache/deployment_event_psd/{dep}/real_features.npy'
    else:
        events = dev.BASE.retained_event_plan(dep, es, split)
        values = dev.ORCH.event_array_for_plan(dep, es, split, events)
        feature = e.TRAINED / f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy'
    h = hidden_features(dep, ms, np.load(feature), dev.TRAIN.make_window_view(np.asarray(values, dtype=np.float32), 2))
    ck = torch.load(root / f'qprobe/models/{dep}/seed_{ms}/validation_selected_probe.pt', weights_only=False, map_location='cpu')
    model = head().cuda().eval(); model.load_state_dict(ck['model'])
    l = model(torch.as_tensor((h-ck['mu'])/ck['sd'], dtype=torch.float32, device='cuda')).cpu().numpy()
    p = softmax(l/ck['temperature'], axis=1)
    if split == 'real':
        pp = np.full((len(full), len(QGRID)), np.nan)
        pp[valid] = p; p = pp
    np.save(path, p)
    return p


def run(root):
    for dep in e.DEPS:
        for ms, qs in zip(body.MODEL_SEEDS, (202609071, 202609072, 202609073)):
            train(root, dep, ms, qs)
    print('All six probes trained; original RNC checkpoints unchanged.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); run(a.root)
