#!/usr/bin/env python3
"""Three finite waveform-only architectures; no PE targets in model training."""
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
import mcwf_omc_ensemble_extension_20260907 as experiment
import mcwf_ordered_mass_predictor_20260907 as original

dev = experiment.dev
KINDS = ('WIDE', 'GLOBAL', 'ORDINAL')
SEEDS = (202609451, 202609452, 202609453)
EPOCHS = 50
torch.set_num_threads(2)


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 9, padding=4*dilation, dilation=dilation),
                                 nn.GroupNorm(8, width), nn.SiLU(), nn.Dropout(.1),
                                 nn.Conv1d(width, width, 9, padding=4*dilation, dilation=dilation),
                                 nn.GroupNorm(8, width))

    def forward(self, x):
        return F.silu(x + self.net(x))


class Predictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'ORDINAL':
            self.body = original.Predictor()
            return
        width = 96 if kind == 'WIDE' else 64
        self.register_buffer('coordinate', torch.linspace(-1, 1, 253)[None, None, :])
        self.first = nn.Sequential(nn.Conv1d(28, width, 9, padding=4), nn.GroupNorm(8, width), nn.SiLU())
        self.local = nn.Sequential(*(Block(width, d) for d in ((1, 2, 4, 8) if kind == 'WIDE' else (2, 4))))
        self.global_context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, dim_feedforward=128, dropout=.1,
                                       activation='gelu', batch_first=True, norm_first=True),
            num_layers=2, enable_nested_tensor=False) if kind == 'GLOBAL' else None
        self.last = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        if self.kind == 'ORDINAL':
            return self.body(x)
        y = self.local(self.first(torch.cat([x, self.coordinate.expand(len(x), -1, -1)], 1)))
        if self.global_context is not None:
            y = self.global_context(y.transpose(1, 2)).transpose(1, 2)
        return self.last(y).squeeze(1)


def initialize(root, kind):
    experiment.initialize(root)
    dev.json_write(root / 'contracts/ARCHITECTURE_ADDENDUM.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'kind': kind,
        'frozen_before_training': True, 'training_seeds': SEEDS, 'epochs': EPOCHS,
        'data': 'Exactly previous OMC 12288 training source parents,160noise blocks;512 development sourceparents,32noise blocks',
        'features': 'Exactly previous 27 x253 phase-template responses, no new time/sky/strain windows',
        'kind_definitions': {
            'WIDE': '96 channels,4 residual dilated convolution blocks,dilations1,2,4,8;mass-grid cross entropy',
            'GLOBAL': '64 channels,2 local residual blocks plus2 4head self-attention blocks;mass-grid cross entropy',
            'ORDINAL': 'Original48channel OMC architecture;loss CE + mean mass-CDF squared error (ranked probability score)'},
        'optimizer': 'AdamW lr2e-4 WD1e-4 cosine min1e-5,128sources x2views,gradclip5',
        'checkpoint_selection': 'minimum development CE, not real PE/official outcome',
        'temperature': 'development CE;grid .5,.75,1,1.25,1.5,2',
        'deployment': 'uniform mixture of3 trained distributions;same architecture for O3/O4a',
        'selection_disclosure': 'Later finite pair-score operating-point selection uses real PE/official development feedback, not blind.',
        'no_PE_official_training_or_score_inputs': True})
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)


def train(root, dep, ms, seed, kind):
    out = root / f'models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    if out.exists():
        raise RuntimeError('Incomplete run retained; use a new recovery directory')
    out.mkdir(parents=True)
    started = time.perf_counter()
    x, tm = original.data(experiment.PREVIOUS, dep, 'train')
    v, vm = original.data(experiment.PREVIOUS, dep, 'validation')
    assert not set(tm.source_uid) & set(vm.source_uid)
    assert not set(tm.noise_bank_index) & set(vm.noise_bank_index)
    group, names = pd.factorize(tm.source_uid, sort=True)
    vg, _ = pd.factorize(vm.source_uid, sort=True)
    truth, vt = np.log(tm.mc_det.to_numpy(float)), np.log(vm.mc_det.to_numpy(float))
    targets, vy = original.targets(truth), original.targets(vt)
    mu, sd = x.mean((0, 2), keepdims=True), x.std((0, 2), keepdims=True).clip(.01)
    x -= mu
    x /= sd
    v = (v - mu) / sd
    xx = torch.as_tensor(x, dtype=torch.float32, device='cuda')
    yy = torch.as_tensor(targets, device='cuda')
    members = [np.flatnonzero(group == k) for k in range(len(names))]
    dev.TRAIN.seed_everything(seed)
    model = Predictor(kind).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-5)
    history, best = [], float('inf')
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(names))
        losses = []
        model.train()
        for start in range(0, len(order), 128):
            ix = np.stack([rng.choice(members[k], 2, replace=False) for k in order[start:start+128]]).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(xx[ix]).float()
                loss = -(F.log_softmax(logits, -1) * yy[ix]).sum(-1).mean()
                if kind == 'ORDINAL':
                    loss = loss + ((F.softmax(logits, -1).cumsum(-1) - yy[ix].cumsum(-1)) ** 2).mean()
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        logits = original.infer(model, v)
        p = softmax(logits.astype(float), 1)
        ce = float(-(vy * np.log(p.clip(1e-30))).sum(-1).mean())
        row = {'epoch': epoch, 'training_loss': float(np.mean(losses)), 'validation_CE': ce,
               'logMc_MAE': float(abs(p @ original.LOG_CENTERS - vt).mean()),
               'seconds': time.perf_counter() - started}
        history.append(row)
        if ce < best:
            best = ce
            torch.save({'model': {k: z.detach().cpu().clone() for k, z in model.state_dict().items()},
                        'mu': mu, 'sd': sd, 'epoch': epoch, 'kind': kind, 'seed': seed},
                       out / 'validation_selected_model.pt')
        dev.csv_write(out / 'history.csv', pd.DataFrame(history))
        if epoch % 10 == 0:
            print(json.dumps({'training': [kind, dep, ms], **row}), flush=True)
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model'])
    logits = original.infer(model, v)
    grid = []
    for temp in (.5, .75, 1., 1.25, 1.5, 2.):
        p = softmax(logits.astype(float) / temp, 1)
        grid.append({'temperature': temp, 'CE': float(-(vy * np.log(p.clip(1e-30))).sum(-1).mean())})
    ck['temperature'] = min(grid, key=lambda r: (r['CE'], abs(r['temperature'] - 1)))['temperature']
    ck['prior'] = experiment.checkpoint(dep, ms)['prior']
    p, outside = original.probability(logits, ck['temperature'])
    np.savez_compressed(out / 'validation_predictions.npz', p=p, outside=outside, group=vg, truth=vt)
    torch.save(ck, out / 'validation_selected_model.pt')
    dev.csv_write(out / 'temperature_grid.csv', pd.DataFrame(grid))
    cdf = np.c_[np.zeros(len(p)), p.cumsum(1)]
    pit = np.array([np.interp(t, original.EDGES, c) for t, c in zip(vt, cdf)])
    report = {'kind': kind, 'deployment': dep, 'seed': seed, 'selected_epoch': ck['epoch'],
              'temperature': ck['temperature'], 'sources_train': len(names), 'sources_development': len(np.unique(vg)),
              'source_overlap': 0, 'noise_overlap': 0, 'central90_coverage': float(((pit >= .05) & (pit <= .95)).mean()),
              'logMc_MAE': float(abs(p @ original.CENTERS - vt).mean()),
              'seconds': time.perf_counter() - started, 'checkpoint_sha256': dev.sha(out / 'validation_selected_model.pt')}
    dev.json_write(out / 'COMPLETE.json', report)
    print(json.dumps({'training_complete': report}), flush=True)


def prediction(root, dep, ms, es, split):
    path = root / f'cache/predictions/{dep}/{es}_{split}_{ms}.npz'
    if path.exists():
        return dict(np.load(path))
    cp = root / f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    if split == 'development':
        a = dict(np.load(cp.parent / 'validation_predictions.npz'))
    else:
        ck = torch.load(cp, weights_only=False, map_location='cpu')
        coarse = np.load(experiment.old.TRAINED / f'cache/deployment_event_psd/{dep}' /
                         ('real_features.npy' if split == 'real' else f'{es}_{split}_features.npy'))
        fine = np.load(experiment.PREVIOUS / f'fine_mass_context/features/{dep}' /
                       ('real.npy' if split == 'real' else f'{es}_{split}.npy'))
        x = original.arrange(coarse, fine)
        model = Predictor(ck['kind']).cuda().eval()
        model.load_state_dict(ck['model'])
        p, outside = original.probability(original.infer(model, (x - ck['mu']) / ck['sd']), ck['temperature'])
        if split == 'real':
            full, events = dev.real_inputs(dep)
            valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
            pp, bb = np.full((len(full), 512), np.nan), np.full(len(full), np.nan)
            pp[valid], bb[valid] = p, outside
            p, outside = pp, bb
        a = {'p': p, 'outside': outside}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **a)
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kind', choices=KINDS, required=True)
    parser.add_argument('--phase', choices=('init', 'train', 'evaluate'), required=True)
    parser.add_argument('--arm', choices=experiment.ARMS, default='MIXTURE-PRIOR')
    args = parser.parse_args()
    if args.phase == 'init':
        initialize(args.root, args.kind)
    elif args.phase == 'train':
        for dep in experiment.old.DEPS:
            for ms, seed in zip(experiment.body.MODEL_SEEDS, SEEDS):
                train(args.root, dep, ms, seed, args.kind)
    else:
        experiment.raw_prediction = prediction
        experiment.run(args.root, args.arm)
