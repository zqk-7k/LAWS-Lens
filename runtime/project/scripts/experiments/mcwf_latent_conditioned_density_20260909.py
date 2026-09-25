#!/usr/bin/env python3
"""A single mass predictor conditioned on frozen raw-waveform embeddings."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_subgrid_density_20260909 as d
import mcwf_subgrid_evaluate_20260909 as score
n, mult = d.n, d.mult
ROOT = None
SEEDS = (2026090961, 2026090962, 2026090963)
WIDTH = 96


def raw_events(dep, seed, split, catalog=None):
    if split in ('train', 'development'):
        source = 'train' if split == 'train' else 'validation'
        return np.load(n.t.PREVIOUS / f'expanded_data/{dep}/{source}/raw2s.npy', mmap_mode='r'), None
    if split == 'real':
        full, events = n.dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        return np.asarray(full[valid, :, -4096:], np.float32), valid
    if catalog is None:
        plan = n.dev.BASE.retained_event_plan(dep, seed, split)
        return n.dev.ORCH.event_array_for_plan(dep, seed, split, plan)[..., -4096:], None
    folder = n.FRESH / f'confirmation/{dep}/catalog_{catalog}'
    events = pd.read_parquet(folder / 'event_manifest.parquet')
    return np.stack([np.load(folder / f'events/{uid}_full24.npy')[..., -4096:] for uid in events.event_uid]), None


@torch.no_grad()
def embeddings(dep, seed, split, catalog=None):
    tag = split if catalog is None else f'{split}_{catalog}'
    path = ROOT / f'embeddings/{dep}/{seed}_{tag}.npy'
    if path.exists():
        return np.load(path)
    cp = n.dev.V7 / dep / f'seed_{seed}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt'
    before = n.sha(cp)
    v7 = n.dev.MAINCODE.cbase.v7
    old, _ = v7.load_unified_model(cp)
    raw, valid = raw_events(dep, seed, split, catalog)
    if not np.isfinite(raw).all():
        raise RuntimeError('Invalid H1/L1 waveform data')
    rows = []
    for values in DataLoader(v7.ArrayCatalog(raw), batch_size=64, shuffle=False, num_workers=0):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = old.base(values.cuda())
        rows.append(z.float().cpu().numpy())
    z = np.concatenate(rows).astype(np.float32)
    if z.shape != (len(raw), WIDTH):
        raise RuntimeError('Embedding shape changed: ' + str(z.shape))
    if n.sha(cp) != before:
        raise RuntimeError('Frozen old encoder changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, z)
    n.write_json(path.with_suffix('.json'), {'checkpoint_sha256': before, 'output_sha256': n.sha(path),
        'shape': list(z.shape), 'head_executed': False, 'old_mass_or_q_outputs_read': False,
        'selection_inputs': 'waveform only', 'finite_raw': True})
    return z


class Predictor(d.Predictor):
    def __init__(self):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(WIDTH, 96), nn.SiLU(), nn.Linear(96, 96))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x, embedding):
        h = self.first(torch.cat([x, self.coordinate.expand(len(x), -1, -1)], 1))
        scale, bias = self.film(embedding).chunk(2, dim=-1)
        h = h * (1 + .5 * torch.tanh(scale[..., None])) + .5 * torch.tanh(bias[..., None])
        h = F.silu(h + self.middle(h))
        logits = self.last(h).squeeze(1)
        offset, logsigma = self.density(h).unbind(1)
        mean = self.mass_centers + .5 * d.DX * torch.tanh(offset)
        sigma = d.DX * torch.exp(logsigma.clamp(-2., 2.))
        return logits, mean, sigma


@torch.no_grad()
def inference(model, x, z):
    rows = [[], [], []]
    model.eval()
    for start in range(0, len(x), 128):
        a = torch.as_tensor(x[start:start + 128], dtype=torch.float32, device='cuda')
        b = torch.as_tensor(z[start:start + 128], dtype=torch.float32, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(a, b)
        for dst, src in zip(rows, out):
            dst.append(src.float().cpu().numpy())
    return [np.concatenate(v).astype(float) for v in rows]


def freeze():
    if ROOT.exists():
        raise RuntimeError('New directory required')
    for name in ('contracts', 'models', 'embeddings', 'predictions', 'calibration', 'configs', 'tables', 'audit',
                 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures', 'cache'):
        (ROOT / name).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {'id': 'MCWF-NODUP-LATENT-DENSITY-04', 'UTC': n.utc(),
        'status': n.STATUS, 'goal_achieved': False, 'same_both_runs': True,
        'hypothesis': 'The old regression values are badly biased but the underlying raw-waveform representation may contain information discarded by scalar cosine/template powers. Condition ONE mass density on that representation,not add old mass scores.',
        'input': 'Frozen old96D waveform embedding plus unchanged54x253 short/low-band template response channels',
        'old_network': 'Call base encoder directly;do not execute parameter_head or read its Mc/q outputs',
        'new_predictor': 'Shared local mass CNN conditioned by96->96->96 FiLM;scale1+0.5tanh andbias0.5tanh;zero-init modulation exactly recovers unconditioned density at epoch0',
        'density': 'Same continuous truncated Gaussian mixture andCDF mass conservation asSUBGRID03',
        'training': {'seeds': list(SEEDS), 'epochs': 30, 'optimizer': 'AdamW1e-4 WD1e-4 cosine1e-5 clip5',
                     'population': '4096parents96noiseblocks8views,512developmentparents32noiseblocks',
                     'selection': 'minimum simulateddevelopment continuousNLL,then temperature .5,.75,1,1.25,1.5,2 by NLL'},
        'mass_score': 'Only new joint mass/eta/spin compatibility;no old mass/q score,no raw-phase orOMC penalty,no alpha totalblend',
        'calibration': 'Same asSUBGRID03;two arms fixed jointgamma/beta and validation-selectedjointgamma/beta;outerweights unchanged',
        'frozen': ['oldencoder', 'time_score', 'sky_raw_log_bf', 'outerweights', 'scope', 'history', 'paper'],
        'no_PE_or_official_score_inputs': True, 'adaptive_development': True, 'not_blind_confirmation': True,
        'refs': ['https://arxiv.org/abs/1709.07871', 'https://www.microsoft.com/en-us/research/publication/mixture-density-networks/'],
        'ref_limits': 'Architecture and probability-representation precedent;not GW PE validation or evidence of optimal hyperparameters.'})
    shutil.copy2(__file__, ROOT / 'scripts/latent_conditioned_density.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(), 'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
        'script_sha256': n.sha(Path(__file__))})
    rows = pd.read_csv(P / 'results/mcwf_nodup_mass_reliability_02_20260909T154100Z/manifest/INPUT_SHA256.csv')
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', rows)
    n.write_json(ROOT / 'audit/ALGORITHM_BOUNDARY.json', {'no_legacy_head_execution': True, 'time_sky_changes': False,
        'no_global_total_mixture': True, 'pure_cosine_and_joint_density_are_correlated_features_not_independent_Bayes_factors': True})


def train(dep, slot, eval_seed, seed):
    out = ROOT / f'models/{dep}/seed_{slot}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    cp = torch.load(d.LOW / f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    x, meta = mult.training_data(d.LOW, dep, 'train', 'MULTIRATE')
    v, vm = mult.training_data(d.LOW, dep, 'validation', 'MULTIRATE')
    if set(meta.source_uid) & set(vm.source_uid) or set(meta.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Source/noise split violation')
    z, vz = embeddings(dep, eval_seed, 'train'), embeddings(dep, eval_seed, 'development')
    emu, esd = z.mean(0), z.std(0).clip(.01)
    z, vz = (z - emu) / esd, (vz - emu) / esd
    x, v = (x - cp['mu']) / cp['sd'], (v - cp['mu']) / cp['sd']
    truth, vt = np.log(meta.mc_det.to_numpy()), np.log(vm.mc_det.to_numpy())
    n.dev.TRAIN.seed_everything(seed)
    model = Predictor().cuda()
    missing, unexpected = model.load_state_dict(cp['model'], strict=False)
    if unexpected or any(not (k.startswith(('density.', 'film.')) or k == 'mass_centers') for k in missing):
        raise RuntimeError('Warm parameter mapping failure')
    control = d.Predictor().cuda().eval()
    control.load_state_dict({k: a for k, a in model.state_dict().items() if not k.startswith('film.')})
    with torch.no_grad():
        aa, bb = torch.as_tensor(v[:4], device='cuda'), torch.as_tensor(vz[:4], device='cuda')
        diff = max(float(abs(a - b).max()) for a, b in zip(model.eval()(aa, bb), control(aa)))
    if diff != 0:
        raise RuntimeError('Zero-conditioning does not reproduce matched density')
    n.write_json(out / 'WARM_TEST.json', {'max_difference': diff})
    del control
    xx, zz, yy = torch.as_tensor(x, device='cuda'), torch.as_tensor(z, device='cuda'), torch.as_tensor(truth, dtype=torch.float32, device='cuda')
    del x, z
    group, names = pd.factorize(meta.source_uid, sort=True)
    members = [np.flatnonzero(group == k) for k in range(len(names))]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30, eta_min=1e-5)
    history, best, start = [], float('inf'), time.perf_counter()
    for epoch in range(31):
        if epoch:
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(names))
            model.train()
            for lo in range(0, len(order), 128):
                idx = np.concatenate([rng.choice(members[k], 2, replace=False) for k in order[lo:lo + 128]])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(xx[idx], zz[idx])
                loss = d.nll(*outputs, yy[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError('Invalid density loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
            scheduler.step()
        a = inference(model, v, vz)
        value = d.numpy_nll(a, vt)
        history.append({'epoch': epoch, 'NLL': value, 'seconds': time.perf_counter() - start})
        if value < best:
            best = value
            torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'mu': cp['mu'], 'sd': cp['sd'], 'emu': emu, 'esd': esd, 'prior': cp['prior'],
                'epoch': epoch, 'seed': seed, 'evaluation_seed': eval_seed, 'NLL': best}, out / 'selected.pt')
        n.write_csv(out / 'HISTORY.csv', history)
        if epoch % 5 == 0:
            print('LATENT_DENSITY', dep, slot, history[-1], flush=True)
    ck = torch.load(out / 'selected.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    a = inference(model, v, vz)
    grid = [{'T': temp, 'NLL': d.numpy_nll(a, vt, temp)} for temp in (.5, .75, 1., 1.25, 1.5, 2.)]
    ck['temperature'] = min(grid, key=lambda r: (r['NLL'], abs(r['T'] - 1)))['T']
    torch.save(ck, out / 'selected.pt')
    pp, oo = d.probability(a, ck['temperature'])
    np.savez_compressed(out / 'development_predictions.npz', p=pp, outside=oo, truth=vt, group=pd.factorize(vm.source_uid, sort=True)[0])
    pit = np.array([np.interp(t, d.EDGES, np.r_[0., p.cumsum()]) for t, p in zip(vt, pp)])
    n.write_csv(out / 'TEMPERATURE_GRID.csv', grid)
    n.write_json(out / 'COMPLETE.json', {'deployment': dep, 'slot': slot, 'epoch': ck['epoch'], 'NLL': best,
        'temperature': ck['temperature'], 'central90coverage': float(((pit >= .05) & (pit <= .95)).mean()),
        'logMc_MAE': float(np.mean(abs(pp @ mult.old.CENTERS - vt))), 'oldhead_outputs_used': False,
        'trained_sources': len(names), 'development_sources': vm.source_uid.nunique(), 'seconds': time.perf_counter() - start,
        'checkpoint_sha256': n.sha(out / 'selected.pt')})


def predict(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    tag = split if catalog is None else f'{split}_{catalog}'
    path = ROOT / f'predictions/{dep}/{slot}_{tag}.npz'
    if path.exists():
        return dict(np.load(path))
    cp = torch.load(ROOT / f'models/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    x, z = d.event_features(dep, seed, split, catalog), embeddings(dep, seed, split, catalog)
    model = Predictor().cuda().eval()
    model.load_state_dict(cp['model'])
    p, oo = d.probability(inference(model, (x - cp['mu']) / cp['sd'], (z - cp['emu']) / cp['esd']), cp['temperature'])
    if split == 'real':
        full, events = n.dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp, o = np.full((len(full), 512), np.nan), np.full(len(full), np.nan)
        pp[valid], o[valid] = p, oo
        p, oo = pp, o
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, outside=oo)
    return {'p': p, 'outside': oo}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'train', 'freeze_score', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = d.ROOT = d.reliability.ROOT = score.ROOT = args.root
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'train':
        for dep in n.DEPS:
            for slot, es, seed in zip(n.t.MODEL_SLOTS, n.SEEDS, SEEDS):
                train(dep, slot, es, seed)
    else:
        score.METHODS = ('NODUP-DIRECT-REPLAY', 'LATENT-JOINT-FIXED', 'LATENT-JOINT-VAL')
        d.reliability.install()
        d.predict = predict
        n.METHODS, n.infer, n.load_panel = score.METHODS, score.infer, score.load_panel
        if args.stage in ('freeze_score', 'calibrate'):
            getattr(score, args.stage)()
        else:
            n.run(ROOT, args.stage)
