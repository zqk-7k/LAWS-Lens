#!/usr/bin/env python3
"""Explicit off-source PSD conditioning; identical waveform method in both runs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import softmax
import torch
from torch import nn
import torch.nn.functional as F
import mcwf_omc_ensemble_extension_20260907 as e
import mcwf_omc_architecture_extension_20260907 as arch
import mcwf_ordered_mass_predictor_20260907 as mass
import mcwf_finelag_eventpsd_20260907 as psd_api

dev = e.dev
GLOBAL = dev.PROJECT / 'results/mcwf_omc_ensemble_exploratory_20260907T141320Z/architectures/GLOBAL'
PRIOR_ROUND = dev.PROJECT / 'results/mcwf_intrinsic_grid_exploratory_20260907T151700Z'
KINDS = ('CONTROL', 'PSD', 'PSD-ROBUST')
SEEDS = (202609811, 202609812, 202609813)
EPOCHS = 30
FREQUENCIES = np.geomspace(40., 580., 32)
torch.set_num_threads(2)


def initialize(root):
    e.initialize(root)
    contract = {
        'code': 'MCWF-NOISE-CONTEXT-EXPLORATORY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'supersedes_initial_scaffolding_contract_only': 'EXPERIMENT_CONTRACT.json describes reused evaluation helpers; THIS document defines new training and scoring.',
        'hypothesis': 'Matched templates use event PSD, but the learned mass head lacks explicit noise context. Test whether conditioning on off-source PSD improves domain transfer.',
        'same_method_both_runs': True,
        'waveform_unchanged': 'H1/L1, peak2s,4096 points,2048Hz,40-580Hz. No new strain, sky or time input.',
        'context': '64 real values: log10 PSD at32 logarithmically spaced40-580Hz frequencies per H1/L1. This is waveform noise context, not PE, event ID or a fourth ranking channel.',
        'context_scaling': 'Mean/SD over unique training PSDs only, SD floor0.1dex. Context never normalized using real catalog.',
        'context_OOD': 'RMS standardized distance to nearest training PSD; threshold is q99 of unique development PSD distances. Candidate new evidence falls back to baseline for either OOD event.',
        'warm_start': str(GLOBAL),
        'kinds': {
            'CONTROL': 'Continue GLOBAL for30epochs, no PSD input. Controls extra optimization.',
            'PSD': 'Same warm GLOBAL;64->128->128 FiLM context; zero-init last layer; bounded gain/offset before local layers.',
            'PSD-ROBUST': 'PSD plus training-noise-group mean-loss EMA. exp(.25*EMA), normalized, clipped[.5,2], detached weights. Exploratory robust optimization, not exact group-DRO theorem.'},
        'training': {'epochs': EPOCHS, 'seeds': SEEDS, 'body_lr': 2e-5, 'new_context_lr': 1e-4,
                     'optimizer': 'AdamW WD1e-4, cosine schedules,gradclip5',
                     'batch': '128 independent source parents,2 random image/noise views per parent',
                     'data': '12288 existing independent waveform source parents;160 train noise blocks.512 development parents;32 disjoint noise blocks.',
                     'targets': 'simulated detector-frame log chirp mass only; no real PE or official labels',
                     'checkpoint': 'minimum development CE, including warm epoch0',
                     'temperature_grid': [.5, .75, 1., 1.25, 1.5, 2.]},
        'same_source_caveat': 'Independent waveform sources, not12288 independent GW-LMC lens environments.',
        'score_modes': ['ADD-PRIOR', 'ADD-BC', 'REPLACE-PRIOR', 'REPLACE-BC'],
        'gamma_grid': [0., .125, .25, .5, 1., 2., 4.],
        'beta_grid': [0., .0625, .125, .25, .5, 1., 2., 4.],
        'ADD': 'Z_OMC+gamma*finite-reference mass inconsistency penalty+beta*simulation-calibrated bounded predictive overlap',
        'REPLACE': 'Z_FRT+same new mass terms. Must pass latest OMC guardrails, not weaker FRT.',
        'support': 'Mass boundary>0.25 or PSD-context OOD disables the entire proposed update and returns current OMC, including replacement mode.',
        'predictive_mass_not_PE': True,
        'frozen': ['time', 'sky', 'outer C-fixed weights', 'strict scope', 'historical encoders/results', 'papers/Overleaf'],
        'O4_comparison': 'Preserve GLOBAL-PRIOR. Candidate must also pass its real-budget noninferiority checks.',
        'simulation_guard': {'max_R10_drop': .02, 'max_AP_drop': .005, 'max_F50_F90_ratio': 1.1},
        'real_development_target': 'No worsening of Top10/20 catastrophic mass count, BC>=.5,Dmax<=3,medianBC, frontend or Hanabi; PE improves somewhere and summed frontend and summed Hanabi counts each increase in EACH run; waveform-onlyTop10 PE noninferior.',
        'real_feedback_disclosure': 'Finite operating-point selection DOES use previously observed real PE/official outcomes as adaptive development objectives. Never describe as independent real validation. Official matches are not true lens labels.',
        'fresh_confirmation': 'Only after both runs pass development; previously used tests are reused development, not locked.',
        'references': ['https://arxiv.org/abs/2106.12594', 'https://arxiv.org/abs/2211.08801'],
        'reference_limit': 'Noise conditioning motivation only. This is not DINGO, full PE or importance-sampling-corrected evidence.',
        'final_status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'}
    dev.json_write(root / 'contracts/NOISE_CONTEXT_CONTRACT.json', contract)
    protected = pd.read_csv(root / 'manifest/PROTECTED_INPUT_SHA256.csv')
    rows = protected.to_dict('records')
    for folder in (GLOBAL / 'models', GLOBAL / 'trials/MIXTURE-PRIOR/diagnostic_export',
                   PRIOR_ROUND / 'reports', PRIOR_ROUND / 'tables', PRIOR_ROUND / 'contracts'):
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                rows.append({'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size})
    dev.csv_write(root / 'manifest/PROTECTED_INPUT_SHA256.csv', pd.DataFrame(rows).drop_duplicates('path'))
    for kind in KINDS:
        work = root / 'variants' / kind
        for part in ('contracts', 'audit', 'models', 'cache', 'trials', 'reports', 'tables', 'scripts', 'logs'):
            (work / part).mkdir(parents=True)
        for path in (root / 'contracts/BASELINE_BUDGETS.csv', root / 'contracts/NOISE_CONTEXT_CONTRACT.json'):
            shutil.copy2(path, work / 'contracts' / path.name)
        for dep in e.old.DEPS:
            shutil.copy2(root / f'audit/{dep}_external_reference.parquet', work / f'audit/{dep}_external_reference.parquet')
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)
    dev.json_write(root / 'contracts/NUMERICAL_TESTS.json', tests())
    print(json.dumps({'initialized': str(root)}), flush=True)


def compress_psd(frequency, psds):
    psds = np.asarray(psds, dtype=float)
    if psds.ndim != 3 or psds.shape[1] != 2 or len(frequency) != psds.shape[2]:
        raise ValueError('PSD must be [noise block,H1/L1,frequency]')
    if not np.isfinite(psds).all() or (psds <= 0).any() or not np.all(np.diff(frequency) > 0):
        raise ValueError('PSD is not finite/positive or frequencies unordered')
    if frequency[0] > FREQUENCIES[0] or frequency[-1] < FREQUENCIES[-1]:
        raise ValueError('PSD does not cover frozen band')
    return np.array([[np.interp(FREQUENCIES, frequency, np.log10(p)) for p in block]
                     for block in psds]).reshape(len(psds), 64).astype(np.float32)


def development_context(dep, split):
    parts = [e.PREVIOUS]
    if split == 'train':
        parts.append(e.PREVIOUS / 'additional_population')
    contexts, ids, unique, hashrows = [], [], [], []
    for part in parts:
        noise = part / f'expanded_data/{dep}/noise'
        m = pd.read_parquet(part / f'expanded_data/{dep}/{split}/event_metadata.parquet')
        base = compress_psd(np.load(noise / 'frequency.npy'), np.load(noise / 'psd.npy', mmap_mode='r'))
        used = m.noise_bank_index.to_numpy(int)
        contexts.append(base[used])
        ids.append(used + (1000 if len(parts) > 1 and part == parts[1] else 0))
        unique.append(base[np.unique(used)])
        for p in (noise / 'frequency.npy', noise / 'psd.npy', part / f'expanded_data/{dep}/{split}/event_metadata.parquet'):
            hashrows.append({'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size})
    return np.concatenate(contexts), np.concatenate(ids), np.concatenate(unique), hashrows


def deployment_context(dep, es, split):
    freq, psds, ids, hashvalue = psd_api.psd_context(dep, es, split)
    context = compress_psd(freq, psds)[ids]
    return context, ids, hashvalue


def context_fit(train_unique, validation_unique):
    mu, sd = train_unique.mean(0), train_unique.std(0).clip(.1)
    anchors = (train_unique - mu) / sd
    dist = cdist((validation_unique - mu) / sd, anchors) / np.sqrt(64)
    limit = float(np.quantile(dist.min(1), .99))
    return {'mu': mu, 'sd': sd, 'anchors': anchors, 'limit': limit}


def context_ood(context, fit):
    values, inverse = np.unique(context, axis=0, return_inverse=True)
    d = cdist((values - fit['mu']) / fit['sd'], fit['anchors']).min(1) / np.sqrt(64)
    return (d > fit['limit'])[inverse], d[inverse]


def audit(root):
    rows, hashes = [], []
    for dep in e.old.DEPS:
        tc, ti, tu, hh = development_context(dep, 'train')
        vc, vi, vu, vv = development_context(dep, 'validation')
        hashes.extend(hh + vv)
        if set(ti) & set(vi):
            raise RuntimeError('Train/development noise IDs overlap')
        fit = context_fit(tu, vu)
        np.savez_compressed(root / f'audit/{dep}_noise_context_fit.npz', **fit)
        for split, c, ids in (('train', tc, ti), ('development', vc, vi)):
            outside, distance = context_ood(c, fit)
            rows.append({'deployment': dep, 'split': split, 'seed': 0, 'events': len(c),
                         'independent_noise_blocks': len(np.unique(ids)), 'OOD_fraction': outside.mean(),
                         'median_distance': np.median(distance), 'q90_distance': np.quantile(distance, .9),
                         'max_distance': distance.max(), 'support_limit': fit['limit']})
        for split in ('validation', 'test', 'real'):
            for es in (dev.SEEDS[:1] if split == 'real' else dev.SEEDS):
                c, ids, hashvalue = deployment_context(dep, es, split)
                outside, distance = context_ood(c, fit)
                rows.append({'deployment': dep, 'split': split, 'seed': es, 'events': len(c),
                             'independent_noise_blocks': len(np.unique(ids)), 'OOD_fraction': outside.mean(),
                             'median_distance': np.median(distance), 'q90_distance': np.quantile(distance, .9),
                             'max_distance': distance.max(), 'support_limit': fit['limit'], 'PSD_sha256': hashvalue})
                dev.csv_write(root / f'audit/{dep}_{es}_{split}_PSD_coverage.csv', pd.DataFrame({
                    'row': np.arange(len(c)), 'noise_index': ids, 'nearest_train_distance': distance, 'OOD': outside}))
    dev.csv_write(root / 'tables/NOISE_DOMAIN_AUDIT.csv', pd.DataFrame(rows))
    dev.csv_write(root / 'manifest/NOISE_INPUT_SHA256.csv', pd.DataFrame(hashes).drop_duplicates('path'))
    dev.json_write(root / 'audit/NOISE_AUDIT_COMPLETE.json', {'pass': True, 'contexts': len(rows),
                   'no_real_PE_or_official_labels_read': True, 'support_fit_on_development_only': True})
    print(json.dumps({'noise_audit_complete': True, 'contexts': len(rows)}), flush=True)


class Predictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.body = arch.Predictor('GLOBAL')
        if kind != 'CONTROL':
            self.context = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Dropout(.1), nn.Linear(128, 128))
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)

    def forward(self, x, context):
        if self.kind == 'CONTROL':
            return self.body(x)
        b = self.body
        y = b.first(torch.cat([x, b.coordinate.expand(len(x), -1, -1)], 1))
        gain, shift = (.5 * torch.tanh(self.context(context))).chunk(2, -1)
        y = y * (1 + gain[..., None]) + shift[..., None]
        y = b.local(y)
        y = b.global_context(y.transpose(1, 2)).transpose(1, 2)
        return b.last(y).squeeze(1)


@torch.no_grad()
def infer(model, x, context):
    model.eval()
    output = []
    for start in range(0, len(x), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(torch.as_tensor(x[start:start+256], dtype=torch.float32, device='cuda'),
                      torch.as_tensor(context[start:start+256], dtype=torch.float32, device='cuda'))
        output.append(z.float().cpu().numpy())
    return np.concatenate(output)


def tests():
    p = np.ones((3, 2, 101)) * 1e-46
    freq = np.linspace(0, 1000, 101)
    a = compress_psd(freq, p)
    b = compress_psd(freq, p * 10)
    assert np.allclose(b - a, 1)
    m = Predictor('PSD').eval()
    x = torch.randn(2, 27, 253)
    with torch.no_grad():
        assert torch.allclose(m(x, torch.randn(2, 64)), m.body(x), atol=1e-6, rtol=1e-6)
    return {'pass': True, 'logPSD_scale_test': True, 'zero_FiLM_equals_warm_GLOBAL': True,
            'coordinates': FREQUENCIES.tolist()}


def train(root, kind, dep, ms, seed):
    work = root / 'variants' / kind
    out = work / f'models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    if out.exists():
        raise RuntimeError('Incomplete training retained; recovery must be explicit')
    out.mkdir(parents=True)
    started = time.perf_counter()
    x, tm = mass.data(e.PREVIOUS, dep, 'train')
    v, vm = mass.data(e.PREVIOUS, dep, 'validation')
    tc, ids, _, _ = development_context(dep, 'train')
    vc, vids, _, _ = development_context(dep, 'validation')
    if not np.array_equal(ids, tm.noise_bank_index.to_numpy(int)) or not np.array_equal(vids, vm.noise_bank_index.to_numpy(int)):
        raise RuntimeError('Context/event metadata misalignment')
    if set(tm.source_uid) & set(vm.source_uid) or set(ids) & set(vids):
        raise RuntimeError('Source/noise leakage')
    fit = dict(np.load(root / f'audit/{dep}_noise_context_fit.npz'))
    warm_path = GLOBAL / f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    warm = torch.load(warm_path, weights_only=False, map_location='cpu')
    mu, sd = warm['mu'], warm['sd']
    x -= mu
    x /= sd
    v = (v - mu) / sd
    tc, vc = (tc - fit['mu']) / fit['sd'], (vc - fit['mu']) / fit['sd']
    groups, names = pd.factorize(tm.source_uid, sort=True)
    vg, _ = pd.factorize(vm.source_uid, sort=True)
    ng, unique_noise = pd.factorize(ids, sort=True)
    members = [np.flatnonzero(groups == k) for k in range(len(names))]
    truth, vtruth = np.log(tm.mc_det.to_numpy(float)), np.log(vm.mc_det.to_numpy(float))
    y, vy = mass.targets(truth), mass.targets(vtruth)
    xx = torch.as_tensor(x, dtype=torch.float32, device='cuda')
    yy = torch.as_tensor(y, dtype=torch.float32, device='cuda')
    cc = torch.as_tensor(tc, dtype=torch.float32, device='cuda')
    gg = torch.as_tensor(ng, dtype=torch.long, device='cuda')
    dev.TRAIN.seed_everything(seed)
    model = Predictor(kind).cuda()
    model.body.load_state_dict(warm['model'])
    params = [{'params': model.body.parameters(), 'lr': 2e-5}]
    if kind != 'CONTROL':
        params.append({'params': model.context.parameters(), 'lr': 1e-4})
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-6)
    ema = torch.zeros(len(unique_noise), device='cuda')
    seen = torch.zeros(len(unique_noise), dtype=torch.bool, device='cuda')
    history, best = [], float('inf')
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS + 1):
        losses = []
        if epoch:
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(names))
            model.train()
            for start in range(0, len(order), 128):
                ix = np.stack([rng.choice(members[k], 2, replace=False) for k in order[start:start+128]]).reshape(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(xx[ix], cc[ix]).float()
                    loss_vector = -(F.log_softmax(logits, -1) * yy[ix]).sum(-1)
                    if kind == 'PSD-ROBUST':
                        with torch.no_grad():
                            for g in gg[ix].unique():
                                value = loss_vector.detach()[gg[ix] == g].mean()
                                ema[g] = .9 * ema[g] + .1 * value if seen[g] else value
                                seen[g] = True
                            weights = torch.exp(.25 * (ema - ema[seen].mean())).clamp(.5, 2.)
                            w = weights[gg[ix]]
                        loss = (w * loss_vector).sum() / w.sum()
                    else:
                        loss = loss_vector.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        logits = infer(model, v, vc)
        prob = softmax(logits.astype(float), 1)
        ce = float(-(vy * np.log(prob.clip(1e-30))).sum(-1).mean())
        row = {'epoch': epoch, 'training_loss': float(np.mean(losses)) if losses else None,
               'validation_CE': ce, 'logMc_MAE': float(abs(prob @ mass.LOG_CENTERS - vtruth).mean()),
               'seconds': time.perf_counter() - started}
        history.append(row)
        if ce < best:
            best = ce
            torch.save({'model': {k: z.detach().cpu().clone() for k, z in model.state_dict().items()},
                        'mu': mu, 'sd': sd, 'context_fit': fit, 'kind': kind, 'epoch': epoch,
                        'seed': seed, 'prior': warm['prior'], 'warm_sha256': dev.sha(warm_path)}, out / 'validation_selected_model.pt')
        dev.csv_write(out / 'history.csv', pd.DataFrame(history))
        if epoch % 5 == 0:
            print(json.dumps({'training': [kind, dep, seed], **row}), flush=True)
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model'])
    logits = infer(model, v, vc)
    grid = []
    for temperature in (.5, .75, 1., 1.25, 1.5, 2.):
        pp = softmax(logits.astype(float) / temperature, 1)
        grid.append({'temperature': temperature, 'CE': float(-(vy * np.log(pp.clip(1e-30))).sum(-1).mean())})
    ck['temperature'] = min(grid, key=lambda r: (r['CE'], abs(r['temperature'] - 1)))['temperature']
    p, edge = mass.probability(logits, ck['temperature'])
    original_context = vc * fit['sd'] + fit['mu']
    outside, distances = context_ood(original_context, fit)
    if kind == 'CONTROL':
        outside[:] = False
    np.savez_compressed(out / 'validation_predictions.npz', p=p, outside=np.where(outside, .5, edge),
                        boundary_mass=edge, noise_ood=outside, noise_distance=distances, group=vg, truth=vtruth)
    torch.save(ck, out / 'validation_selected_model.pt')
    dev.csv_write(out / 'temperature_grid.csv', pd.DataFrame(grid))
    # Whole PSD contexts are permuted only as a validation diagnostic, never as a trained new arm.
    perm = np.random.default_rng(seed + 888).permutation(len(vc))
    shuffled, _ = mass.probability(infer(model, v, vc[perm]), ck['temperature'])
    cdf = np.c_[np.zeros(len(p)), p.cumsum(1)]
    pit = np.array([np.interp(t, mass.EDGES, c) for t, c in zip(vtruth, cdf)])
    dev.json_write(out / 'COMPLETE.json', {'kind': kind, 'deployment': dep, 'seed': seed,
        'selected_epoch': ck['epoch'], 'temperature': ck['temperature'], 'train_sources': len(names),
        'train_noise_blocks': len(unique_noise), 'development_sources': len(np.unique(vg)),
        'source_overlap': 0, 'noise_overlap': 0, 'logMc_MAE': float(abs(p @ mass.CENTERS - vtruth).mean()),
        'central90_coverage': float(((pit >= .05) & (pit <= .95)).mean()),
        'shuffled_context_logMc_MAE': float(abs(shuffled @ mass.CENTERS - vtruth).mean()),
        'context_OOD_rate': float(outside.mean()), 'maximum_gpu_bytes': torch.cuda.max_memory_allocated(),
        'seconds': time.perf_counter() - started, 'checkpoint_sha256': dev.sha(out / 'validation_selected_model.pt')})
    print(json.dumps({'training_complete': [kind, dep, seed]}), flush=True)


def prediction(work, dep, ms, es, split):
    path = work / f'cache/predictions/{dep}/{es}_{split}_{ms}.npz'
    if path.exists():
        return dict(np.load(path))
    cp = work / f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Unfinished model')
    if split == 'development':
        a = dict(np.load(cp.parent / 'validation_predictions.npz'))
    else:
        ck = torch.load(cp, weights_only=False, map_location='cpu')
        stem = 'real' if split == 'real' else f'{es}_{split}'
        coarse = np.load(e.old.TRAINED / f'cache/deployment_event_psd/{dep}/{stem}_features.npy')
        fine = np.load(e.PREVIOUS / f'fine_mass_context/features/{dep}/{stem}.npy')
        x = mass.arrange(coarse, fine)
        context, _, context_hash = deployment_context(dep, es, split)
        if len(context) != len(x):
            raise RuntimeError('Context/feature row mismatch')
        fit = ck['context_fit']
        noise_ood, distance = context_ood(context, fit)
        if ck['kind'] == 'CONTROL':
            noise_ood[:] = False
        model = Predictor(ck['kind']).cuda().eval()
        model.load_state_dict(ck['model'])
        p, edge = mass.probability(infer(model, (x - ck['mu']) / ck['sd'],
                                       (context - fit['mu']) / fit['sd']), ck['temperature'])
        outside = np.where(noise_ood, .5, edge)
        if split == 'real':
            full, events = dev.real_inputs(dep)
            valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
            pp, oo = np.full((len(full), 512), np.nan), np.full(len(full), np.nan)
            dd, nn = np.full(len(full), np.nan), np.zeros(len(full), bool)
            pp[valid], oo[valid], dd[valid], nn[valid] = p, outside, distance, noise_ood
            p, outside, distance, noise_ood = pp, oo, dd, nn
        a = {'p': p, 'outside': outside, 'noise_distance': distance, 'noise_ood': noise_ood,
             'checkpoint_sha256': dev.sha(cp), 'context_sha256': context_hash}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **a)
    return a


OLD_SCORE = e.score
OLD_TARGET = e.batch_target
OLD_VALUES = e.values


def cached_values(work, dep, es, split):
    path = work / f'cache/pair_features/{dep}_{es}_{split}.npz'
    if path.exists():
        frame = pd.read_parquet(e.TRIAL / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        return frame, dict(np.load(path))
    frame, values = OLD_VALUES(work, dep, es, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return frame, values


def score(frame, x, spec, arm):
    if spec.get('unchanged_baseline'):
        return frame.waveform_score.to_numpy(float), np.zeros(len(frame)), np.zeros(len(frame))
    f = frame.copy()
    f['waveform_score'] = frame.waveform_score if arm.startswith('ADD-') else frame.FRT_baseline_waveform_score
    z, penalty, increment = e.evaluate.score(f, x, spec, arm.split('-')[-1])
    unavailable = np.asarray(x['ood'], dtype=bool)
    z[unavailable] = frame.waveform_score.to_numpy(float)[unavailable]
    penalty[unavailable], increment[unavailable] = 0., 0.
    return z, penalty, increment


def target(combos, pools, pe, baseline, dep):
    result = OLD_TARGET(combos, pools, pe, baseline, dep)
    if dep == 'gwtc4':
        budgets = pd.read_csv(GLOBAL / 'trials/MIXTURE-PRIOR/diagnostic_export/BUDGETS.csv')
        budgets = budgets[budgets.seed.astype(str).eq('consensus')]
        check = OLD_TARGET(combos, pools, pe, budgets, dep)
        result['GLOBAL_noninferior'] = check['noninferior']
        result['target_pass'] &= check['noninferior']
    else:
        result['GLOBAL_noninferior'] = np.ones(len(combos), bool)
    return result


def configure():
    e.raw_prediction = prediction
    e.values = cached_values
    e.score = score
    e.batch_target = target
    e.GAMMAS = (0., .125, .25, .5, 1., 2., 4.)
    e.BETAS = (0., .0625, .125, .25, .5, 1., 2., 4.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('init', 'audit', 'train', 'evaluate'), required=True)
    parser.add_argument('--kind', choices=KINDS, default='PSD')
    parser.add_argument('--deployment', choices=e.old.DEPS)
    parser.add_argument('--model-key', type=int)
    parser.add_argument('--arm', choices=('ADD-PRIOR', 'ADD-BC', 'REPLACE-PRIOR', 'REPLACE-BC'), default='ADD-PRIOR')
    args = parser.parse_args()
    if args.phase == 'init':
        initialize(args.root)
    elif args.phase == 'audit':
        audit(args.root)
    elif args.phase == 'train':
        for dep in ([args.deployment] if args.deployment else e.old.DEPS):
            for ms, seed in zip(e.body.MODEL_SEEDS, SEEDS):
                if args.model_key is None or ms == args.model_key:
                    train(args.root, args.kind, dep, ms, seed)
    else:
        configure()
        e.run(args.root / 'variants' / args.kind, args.arm)
