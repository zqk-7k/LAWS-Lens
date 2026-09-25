#!/usr/bin/env python3
"""Waveform time-frequency residual ablation, not posterior-supervised training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import fcntl
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from torch import nn
import torch.nn.functional as F
import mcwf_noise_context_20260908 as n

e, dev, mass, arch = n.e, n.dev, n.mass, n.arch
KINDS = ('TF-POWER', 'TF-COMPLEX')
SEEDS = (202609821, 202609822, 202609823)
EPOCHS = 40
torch.set_num_threads(2)


def initialize(root):
    contract = root / 'contracts/TF_RESIDUAL_ADDENDUM.json'
    if contract.exists():
        raise RuntimeError('Preserve existing addendum')
    dev.json_write(contract, {
        'utc': datetime.now(timezone.utc).isoformat(), 'code': 'MCWF-TF-RESIDUAL-EXPLORATORY',
        'hypothesis': 'Template maximum correlations discard time-frequency structure. Test raw2s spectrogram/complex-STFT residual mass inference.',
        'same_both_runs': True, 'input': 'Unchanged H1/L1 peak2s4096points,2048Hz,40-580Hz',
        'frozen_parent': str(n.GLOBAL), 'parent_frozen_during_training': True,
        'features': 'Per-detector mean/SD normalization;Hann256 STFT hop64,centered constant padding;normalized Fourier coefficients;retain40..576Hz bins inclusive.',
        'arms': {'TF-POWER': 'log1p(abs(STFT)^2),2channels',
                 'TF-COMPLEX': 'Same power plus real and imaginary coefficients/ sqrt(1+abs(STFT)^2),6channels'},
        'model': 'Conv2d16/32/64/64 stride2 kernels5/3/3/3,GroupNorm,SiLU;adaptive4x4 pooling;1024->128->253;zero-init final layer;residual bounded+-2nats added to frozen GLOBAL logits.',
        'training': {'epochs': EPOCHS, 'seeds': SEEDS, 'optimizer': 'AdamWlr5e-4,WD1e-3,cosine_min1e-5,clip5',
                     'batch': '128sources x2randomviews, same12288sources/160noiseblocks',
                     'checkpoint': 'minimum512source developmentCE,include zero-residualepoch0',
                     'target': 'simulated detector-frame Mc only', 'temperature': [.5, .75, 1, 1.25, 1.5, 2]},
        'calibration_and_score': 'Same frozen noise-context experiment ADD/REPLACE and mass-PRIOR/BC grids;no PSD input or PSD-OOD for this arm,existing mass-boundary guard.',
        'not_new_strain_generation': True, 'not_full_PE': True, 'no_new_time_sky': True,
        'development_data_reused': True, 'real_external_objective_selection_is_adaptive': True,
        'comparison': 'OMC main baseline and preserved O4 GLOBAL-PRIOR;same shared method acrossruns',
        'references': ['https://arxiv.org/abs/2011.10425', 'https://docs.pytorch.org/docs/stable/generated/torch.stft.html'],
        'references_scope': 'Motivation and numerical API;architecture and hyperparameters are explicitly exploratory engineering settings,not validated literature defaults.',
        'stop_boundary': 'No automatic adoption,paper update or claims of blinded real generalization'})
    for kind in KINDS:
        work = root / 'variants' / kind
        for p in ('contracts', 'audit', 'models', 'cache', 'trials', 'reports', 'tables', 'logs', 'scripts'):
            (work / p).mkdir(parents=True)
        for p in (contract, root / 'contracts/BASELINE_BUDGETS.csv'):
            shutil.copy2(p, work / 'contracts' / p.name)
        for dep in e.old.DEPS:
            shutil.copy2(root / f'audit/{dep}_external_reference.parquet', work / f'audit/{dep}_external_reference.parquet')
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)
    dev.json_write(root / 'contracts/TF_NUMERICAL_TESTS.json', tests())


def raw_development(dep, split):
    parts = [e.PREVIOUS / f'expanded_data/{dep}/{split}/raw2s.npy']
    if split == 'train':
        parts += [e.PREVIOUS / f'additional_population/expanded_data/{dep}/train/raw2s.npy']
    raw = np.concatenate([np.load(p, mmap_mode='r') for p in parts])
    if raw.shape[1:] != (2, 4096) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid raw waveform')
    return raw


def raw_deployment(dep, es, split):
    if split == 'real':
        full, events = dev.real_inputs(dep)
        full = full[events.strict_h1l1_preprocessing_pass.to_numpy(bool)]
    else:
        plan = dev.BASE.retained_event_plan(dep, es, split)
        full = dev.ORCH.event_array_for_plan(dep, es, split, plan)
    raw = dev.TRAIN.make_window_view(np.asarray(full, np.float32), 2)
    if raw.shape[1:] != (2, 4096) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid deployment raw waveform')
    return raw


def warm_logits(root, dep, ms, es, split):
    stem = split if split in ('train', 'development', 'real') else f'{es}_{split}'
    path = root / f'cache/TF_warm_logits/{dep}/{ms}_{stem}.npy'
    path.parent.mkdir(parents=True, exist_ok=True)
    cp = n.GLOBAL / f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    # Both TF arms share this immutable cache; serialize publication and reads.
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            provenance = json.loads(path.with_suffix('.json').read_text())
            if provenance['warm_sha256'] != dev.sha(cp) or provenance['logits_sha256'] != dev.sha(path):
                raise RuntimeError('Warm-cache provenance mismatch: ' + str(path))
            logits = np.load(path)
        else:
            logits = compute_warm_logits(dep, ms, es, split)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                np.save(handle, logits)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            dev.json_write(path.with_suffix('.json'), {'warm_sha256': dev.sha(cp), 'logits_sha256': dev.sha(path)})
        if logits.ndim != 2 or logits.shape[1] != 253 or not np.isfinite(logits).all():
            raise RuntimeError('Invalid warm-cache content: ' + str(path))
        return logits


def compute_warm_logits(dep, ms, es, split):
    cp = n.GLOBAL / f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    if split in ('train', 'development'):
        x, _ = mass.data(e.PREVIOUS, dep, 'train' if split == 'train' else 'validation')
    else:
        prefix = 'real' if split == 'real' else f'{es}_{split}'
        c = np.load(e.old.TRAINED / f'cache/deployment_event_psd/{dep}/{prefix}_features.npy')
        f = np.load(e.PREVIOUS / f'fine_mass_context/features/{dep}/{prefix}.npy')
        x = mass.arrange(c, f)
    model = arch.Predictor('GLOBAL').cuda().eval()
    model.load_state_dict(ck['model'])
    logits = mass.infer(model, (x - ck['mu']) / ck['sd']).astype(np.float32)
    return logits


class Predictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer('window', torch.hann_window(256))
        channels = 2 if kind == 'TF-POWER' else 6
        blocks = []
        for width, kernel in ((16, 5), (32, 3), (64, 3), (64, 3)):
            blocks += [nn.Conv2d(channels, width, kernel, stride=2, padding=kernel//2),
                       nn.GroupNorm(8, width), nn.SiLU()]
            channels = width
        self.conv = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 4)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(1024, 128), nn.SiLU(), nn.Dropout(.1), nn.Linear(128, 253))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def transform(self, raw):
        x = raw.float()
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        spec = torch.stft(x.flatten(0, 1), 256, hop_length=64, window=self.window,
                          normalized=True, center=True, pad_mode='constant', return_complex=True)
        spec = spec.reshape(len(x), 2, 129, -1)[:, :, 5:73]
        power = spec.abs().square()
        magnitude = torch.log1p(power)
        if self.kind == 'TF-POWER':
            return magnitude
        scale = (1 + power).sqrt()
        return torch.cat([magnitude, spec.real / scale, spec.imag / scale], 1)

    def forward(self, raw, warm):
        with torch.autocast('cuda', enabled=False):
            feature = self.transform(raw)
        return warm.float() + 2 * torch.tanh(self.head(self.conv(feature)).float())


def tests():
    torch.manual_seed(202609820)
    x = torch.randn(3, 2, 4096)
    model = Predictor('TF-POWER').eval()
    a = model.transform(x)
    b = model.transform(-x)
    assert a.shape == (3, 2, 68, 65) and torch.allclose(a, b, atol=1e-6)
    with torch.no_grad():
        resid = model.head(model.conv(a))
    assert torch.equal(resid, torch.zeros_like(resid))
    return {'pass': True, 'STFT_shape': list(a.shape), 'power_global_sign_invariant': True,
            'initial_residual_exact_zero': True, 'complex_phase_intentionally_not_invariant': True}


def cache_tests(root):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import patch
    import threading
    guard, calls = threading.Lock(), []
    expected = np.arange(759, dtype=np.float32).reshape(3, 253)
    def compute(*args):
        with guard:
            calls.append(args)
        time.sleep(.05)
        return expected.copy()
    with tempfile.TemporaryDirectory(prefix='mcwf_cache_unit_') as temporary:
        base = Path(temporary)
        checkpoint = base / 'parent/models/unit/seed_1/validation_selected_model.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'unit-test-provenance-only-not-a-checkpoint')
        with patch.object(n, 'GLOBAL', base / 'parent'), patch(__name__ + '.compute_warm_logits', side_effect=compute):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: warm_logits(base / 'work', 'unit', 1, 0, 'real'), range(4)))
        assert len(calls) == 1 and all(np.array_equal(v, expected) for v in results)
    dev.json_write(root / 'audit/CACHE_CONCURRENCY_UNIT_TEST.json', {
        'pass': True, 'concurrent_readers': 4, 'compute_calls': len(calls),
        'all_readers_identical': True, 'unit_data_only': True,
        'real_cache_replay_separate': 'audit/WARM_CACHE_INDEPENDENT_REPLAY.csv'})


@torch.no_grad()
def infer(model, raw, logits):
    model.eval(); output = []
    for start in range(0, len(raw), 256):
        x = torch.as_tensor(np.asarray(raw[start:start+256], np.float32), device='cuda')
        w = torch.as_tensor(logits[start:start+256], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(x, w)
        output.append(p.float().cpu().numpy())
    return np.concatenate(output)


def train(root, dep, ms, seed, kind):
    work = root / 'variants' / kind
    out = work / f'models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists(): return
    if out.exists(): raise RuntimeError('Incomplete run; do not overwrite')
    out.mkdir(parents=True)
    started = time.perf_counter()
    meta = mass.population.metadata(e.PREVIOUS, dep, 'train')
    vm = mass.population.metadata(e.PREVIOUS, dep, 'validation')
    if set(meta.source_uid) & set(vm.source_uid) or set(meta.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Source/noise split leakage')
    groups, names = pd.factorize(meta.source_uid, sort=True)
    vg, _ = pd.factorize(vm.source_uid, sort=True)
    members = [np.flatnonzero(groups == k) for k in range(len(names))]
    raw, vr = raw_development(dep, 'train'), raw_development(dep, 'validation')
    warm, vw = warm_logits(root, dep, ms, 0, 'train'), warm_logits(root, dep, ms, 0, 'development')
    assert len(raw) == len(warm) == len(meta)
    x = torch.as_tensor(raw, device='cuda')
    w = torch.as_tensor(warm, device='cuda')
    truth, vt = np.log(meta.mc_det.to_numpy(float)), np.log(vm.mc_det.to_numpy(float))
    y = torch.as_tensor(mass.targets(truth), device='cuda'); vy = mass.targets(vt)
    dev.TRAIN.seed_everything(seed)
    model = Predictor(kind).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    best, history = float('inf'), []
    for epoch in range(EPOCHS + 1):
        losses = []
        if epoch:
            order = np.random.default_rng(seed+epoch).permutation(len(names)); rng = np.random.default_rng(seed+epoch+1000)
            model.train()
            for start in range(0, len(order), 128):
                ix = np.stack([rng.choice(members[k], 2, replace=False) for k in order[start:start+128]]).reshape(-1)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x[ix], w[ix])
                    loss = -(F.log_softmax(logits, -1) * y[ix]).sum(-1).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite TF loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        logits = infer(model, vr, vw); pp = softmax(logits.astype(float), 1)
        ce = float(-(vy * np.log(pp.clip(1e-30))).sum(-1).mean())
        row = {'epoch': epoch, 'train_CE': float(np.mean(losses)) if losses else None,
               'development_CE': ce, 'logMc_MAE': float(abs(pp @ mass.LOG_CENTERS-vt).mean()),
               'seconds': time.perf_counter()-started}
        history.append(row)
        if ce < best:
            best = ce
            torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'kind': kind, 'epoch': epoch, 'seed': seed,
                        'warm_model': str(n.GLOBAL / f'models/{dep}/seed_{ms}/validation_selected_model.pt')},
                       out / 'validation_selected_model.pt')
        dev.csv_write(out / 'history.csv', pd.DataFrame(history))
        if epoch % 10 == 0: print(json.dumps({'TF_training': [kind, dep, seed], **row}), flush=True)
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model']); logits = infer(model, vr, vw)
    grid = []
    for t in (.5, .75, 1., 1.25, 1.5, 2.):
        pp = softmax(logits.astype(float)/t, 1)
        grid.append({'temperature': t, 'CE': float(-(vy*np.log(pp.clip(1e-30))).sum(-1).mean())})
    ck['temperature'] = min(grid, key=lambda r: (r['CE'], abs(r['temperature']-1)))['temperature']
    p, boundary = mass.probability(logits, ck['temperature'])
    np.savez_compressed(out / 'validation_predictions.npz', p=p, outside=boundary, group=vg, truth=vt)
    ck['warm_sha256'] = dev.sha(Path(ck['warm_model']))
    torch.save(ck, out / 'validation_selected_model.pt')
    dev.csv_write(out / 'temperature_grid.csv', pd.DataFrame(grid))
    pit = np.array([np.interp(t, mass.EDGES, c) for t,c in zip(vt, np.c_[np.zeros(len(p)),p.cumsum(1)])])
    dev.json_write(out / 'COMPLETE.json', {'kind': kind, 'deployment': dep, 'seed': seed,
        'selected_epoch': ck['epoch'], 'temperature': ck['temperature'], 'train_sources': len(names),
        'development_sources': len(np.unique(vg)), 'source_overlap': 0, 'noise_overlap': 0,
        'logMc_MAE': float(abs(p @ mass.CENTERS-vt).mean()),
        'central90_coverage': float(((pit>=.05)&(pit<=.95)).mean()),
        'seconds': time.perf_counter()-started, 'checkpoint_sha256': dev.sha(out / 'validation_selected_model.pt')})
    print(json.dumps({'TF_training_complete': [kind,dep,seed]}), flush=True)


def prediction(work, dep, ms, es, split):
    path = work / f'cache/predictions/{dep}/{es}_{split}_{ms}.npz'
    if path.exists(): return dict(np.load(path))
    cp = work / f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    assert (cp.parent/'COMPLETE.json').exists()
    if split == 'development':
        a = dict(np.load(cp.parent/'validation_predictions.npz'))
    else:
        root = work.parent.parent
        raw = raw_deployment(dep,es,split); warm = warm_logits(root,dep,ms,es,split)
        ck = torch.load(cp,weights_only=False,map_location='cpu')
        assert dev.sha(Path(ck['warm_model'])) == ck['warm_sha256']
        model = Predictor(ck['kind']).cuda().eval(); model.load_state_dict(ck['model'])
        p, outside = mass.probability(infer(model, raw, warm),ck['temperature'])
        if split == 'real':
            full,events = dev.real_inputs(dep); valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
            pp,bb=np.full((len(full),512),np.nan),np.full(len(full),np.nan)
            pp[valid],bb[valid]=p,outside;p,outside=pp,bb
        a={'p':p,'outside':outside,'checkpoint_sha256':dev.sha(cp)}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)
    return a


def configure():
    n.configure();e.raw_prediction=prediction


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('init','train','evaluate','test-cache'),required=True)
    p.add_argument('--kind',choices=KINDS,default='TF-POWER')
    p.add_argument('--arm',choices=('ADD-PRIOR','ADD-BC','REPLACE-PRIOR','REPLACE-BC'),default='ADD-PRIOR')
    a=p.parse_args()
    if a.phase=='init': initialize(a.root)
    elif a.phase=='train':
        for dep in e.old.DEPS:
            for ms,seed in zip(e.body.MODEL_SEEDS,SEEDS): train(a.root,dep,ms,seed,a.kind)
    elif a.phase=='test-cache': cache_tests(a.root)
    else:
        configure();e.run(a.root/'variants'/a.kind,a.arm)
