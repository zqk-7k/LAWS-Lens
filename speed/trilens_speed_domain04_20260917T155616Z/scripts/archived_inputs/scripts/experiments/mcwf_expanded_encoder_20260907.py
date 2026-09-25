#!/usr/bin/env python3
"""Controlled RNC encoder update using independent, expanded waveform data."""
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
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_finelag_features_20260907 as fine
import mcwf_rankncontrast_20260907 as rnc
import mcwf_mass_tf_20260905 as tf
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
torch.set_num_threads(2)
NEW_SEEDS = (202609131, 202609132, 202609133)
EPOCHS = 50
GAMMAS = (.125, .25, .5, 1., 2., 4.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)


def initialize(root):
    folder = root / 'expanded_encoder'
    path = folder / 'contracts/TRAINING.json'
    if path.exists():
        return
    contract = root / 'expanded_data/contracts/DATA_CONTRACT.json'
    dev.json_write(path, {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'single_controlled_change': 'new4096source training population and96independent run-matched noise blocks; same2s RAW-PHASE-SOURCE architecture, same576templates, same event-specificPSD and same CE+SupCon+RNC loss',
        'data_contract_sha256': dev.sha(contract), 'training_seeds': NEW_SEEDS,
        'validation': '512new source parents,1024views,32noise blocks disjoint from train and historical data',
        'normalization': 'keep original RNC phase-feature mean/SD and input preprocessor to preserve the warm-start function exactly',
        'optimizer': 'AdamW1e-4,weightdecay1e-4,cosine_min1e-5,50epochs,64sources*2views,onepass/epoch,clip5',
        'step_budget': '4096/64*50=3200updates vs old960/64*4*50=3000updates; source diversity changes, not a4x optimization-budget increase',
        'loss': 'soft logMc64bin CE+sourceSupCon+RNC(logMc)',
        'selection': 'earliest minimum newvalidation CE+0.2SupCon+0.2RNC,includingepoch0; temperature by CE only',
        'epoch_zero': 'no encoder improvement claim if unchanged model is selected; new temperature may differ and is separately disclosed',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'all oldmodel and result files'],
        'no_real_PE_or_official_training': True,
        'interpretation': 'adaptive development followed by a required new independent confirmation if all real-development targets are met',
    })
    (folder / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, folder / 'scripts' / Path(__file__).name)


def features(root, dep, split):
    folder = root / f'expanded_data/{dep}/{split}'
    if not (folder / 'COMPLETE.json').exists():
        raise RuntimeError('Physical waveform data are incomplete')
    path = root / f'expanded_encoder/features/{dep}/{split}.npy'
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.load(folder / 'raw2s.npy', mmap_mode='r')
    meta = pd.read_parquet(folder / 'event_metadata.parquet')
    noise = root / f'expanded_data/{dep}/noise'
    frequency = np.load(noise / 'frequency.npy')
    psds = np.load(noise / 'psd.npy', mmap_mode='r')
    spectra = np.load(e.TRAINED / 'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy')
    result = np.empty((len(raw), 3, 64, 3, 3), np.float32)
    audit = []
    for bi in sorted(meta.noise_bank_index.unique()):
        indices = np.flatnonzero(meta.noise_bank_index.to_numpy(int) == bi)
        start = time.perf_counter()
        bank = adaptive.whitened_bank(spectra, frequency, psds[bi])
        result[indices] = fine.features(raw[indices], bank)
        audit.append({'noise_bank_index': int(bi), 'events': len(indices), 'seconds': time.perf_counter()-start})
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite phase features')
    np.save(path, result)
    dev.csv_write(path.with_suffix('.audit.csv'), pd.DataFrame(audit))
    dev.json_write(path.with_suffix('.json'), {'feature_sha256': dev.sha(path), 'input_sha256': dev.sha(folder / 'raw2s.npy'),
        'psd_sha256': dev.sha(noise / 'psd.npy'), 'events': len(raw), 'seconds': sum(a['seconds'] for a in audit),
        'unit_test': fine.numerical_test()})
    print(json.dumps({'expanded_features': dep, 'split': split, 'events': len(raw), 'seconds': sum(a['seconds'] for a in audit)}), flush=True)


def data(root, dep, split, ck):
    folder = root / f'expanded_data/{dep}/{split}'
    meta = pd.read_parquet(folder / 'event_metadata.parquet')
    raw = np.load(folder / 'raw2s.npy', mmap_mode='r')
    x = np.load(root / f'expanded_encoder/features/{dep}/{split}.npy')
    x = (x-ck['mu']) / ck['sd']
    y = tf.target_prob(np.log(meta.mc_det.to_numpy(float)))
    group, names = pd.factorize(meta.source_uid, sort=True)
    return x, raw, y, meta, group, names


def train(root, dep, ms, seed):
    initialize(root)
    out = root / f'expanded_encoder/models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation'):
        features(root, dep, split)
    checkpoint = e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
    tx, raw, y, meta, group, names = data(root, dep, 'train', ck)
    vx, vr, vy, vm, vg, vnames = data(root, dep, 'validation', ck)
    if set(names) & set(vnames) or set(meta.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Source/noise split overlap')
    dev.json_write(out / 'SPLIT_AUDIT.json', {'train_sources': len(names), 'validation_sources': len(vnames),
        'source_overlap': 0, 'noise_overlap': 0, 'data_noise_audit': str(root / f'expanded_data/{dep}/noise/COMPLETE.json')})
    members = [np.flatnonzero(group == k) for k in range(len(names))]
    truth = np.log(meta.mc_det.to_numpy(float))
    vt = np.log(vm.mc_det.to_numpy(float))
    dev.TRAIN.seed_everything(seed)
    model = body.Encoder('RAW-PHASE-SOURCE').cuda()
    model.load_state_dict(ck['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    best, history, first = float('inf'), [], 0
    started = time.perf_counter()
    resume = out / 'resume.pt'
    if resume.exists():
        r = torch.load(resume, weights_only=False, map_location='cpu')
        model.load_state_dict(r['model'])
        opt.load_state_dict(r['optimizer'])
        scheduler.load_state_dict(r['scheduler'])
        best, history, first = r['best'], r['history'], r['epoch']+1
        torch.set_rng_state(r['rng'])
        torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first, EPOCHS+1):
        start = time.perf_counter()
        losses = []
        if epoch:
            rng = np.random.default_rng(seed+epoch)
            model.train()
            order = rng.permutation(len(names))
            for begin in range(0, len(order), 64):
                sources = order[begin:begin+64]
                indices = np.stack([rng.choice(members[k], 2, replace=False) for k in sources]).T.reshape(-1)
                xx = torch.as_tensor(tx[indices], dtype=torch.float32, device='cuda')
                rr = torch.as_tensor(np.asarray(raw[indices]), dtype=torch.float32, device='cuda')
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, z = model(xx, rr)
                    ce = -(F.log_softmax(logits.float(), -1)*torch.as_tensor(y[indices], device='cuda')).sum(-1).mean()
                    sc = body.source_contrastive(z, torch.as_tensor(np.tile(sources, 2), device='cuda'))
                    rank = rnc.rank_loss(z, torch.as_tensor(truth[indices], device='cuda'))
                    loss = ce+sc+rank
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite expanded-data training')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        logits, z = body.infer(model, vx, vr)
        p = softmax(logits.astype(float), 1)
        ce = float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())
        sc = float(body.source_contrastive(torch.as_tensor(z, device='cuda'), torch.as_tensor(vg, device='cuda')))
        rank = float(rnc.rank_loss(torch.as_tensor(z, device='cuda'), torch.as_tensor(vt, device='cuda')))
        criterion = ce+.2*sc+.2*rank
        row = {'epoch': epoch, 'training_loss': float(np.mean(losses)) if losses else None,
            'CE': ce, 'SupCon': sc, 'RNC': rank, 'criterion': criterion,
            'logMc_MAE': float(abs(p@tf.LOG_CENTERS-vt).mean()), 'seconds': time.perf_counter()-start}
        history.append(row)
        if criterion < best:
            best = criterion
            torch.save({**ck, 'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'epoch': epoch, 'criterion': criterion, 'new_training_seed': seed,
                'warm_sha256': dev.sha(checkpoint), 'training_population': str(root / 'expanded_data'),
                'training_contract_sha256': dev.sha(root / 'expanded_encoder/contracts/TRAINING.json')}, out / 'validation_selected_model.pt')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
            'best': best, 'history': history, 'epoch': epoch, 'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all()}, resume)
        dev.csv_write(out / 'history.csv', pd.DataFrame(history))
        if epoch % 10 == 0:
            print(json.dumps({'expanded_training': dep, 'seed': seed, **row}), flush=True)
    chosen = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(chosen['model'])
    logits, z = body.infer(model, vx, vr)
    grid = []
    for temperature in (.5, .75, 1., 1.25, 1.5, 2., 3., 4.):
        p = softmax(logits.astype(float)/temperature, 1)
        grid.append({'temperature': temperature, 'CE': float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())})
    chosen['temperature'] = min(grid, key=lambda r: (r['CE'], abs(r['temperature']-1)))['temperature']
    torch.save(chosen, out / 'validation_selected_model.pt')
    p = softmax(logits.astype(float)/chosen['temperature'], 1)
    prior = np.maximum(y.mean(0), 1e-6)
    prior /= prior.sum()
    np.savez_compressed(out / 'validation_predictions.npz', p=p, z=z, group=vg, truth=vy, prior=prior)
    dev.csv_write(out / 'temperature_grid.csv', pd.DataFrame(grid))
    report = {'deployment': dep, 'seed': seed, 'epoch': chosen['epoch'], 'temperature': chosen['temperature'],
        'logMc_MAE': float(abs(p@tf.LOG_CENTERS-vt).mean()), 'seconds': time.perf_counter()-started,
        'checkpoint_sha256': dev.sha(out / 'validation_selected_model.pt')}
    dev.json_write(out / 'COMPLETE.json', report)
    print(json.dumps({'expanded_training_done': report}), flush=True)


def prediction(root, dep, ms, es, split):
    cp = root / f'expanded_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Training incomplete')
    folder = root / f'expanded_encoder/predictions/{dep}/model_{ms}_eval_{es}'
    path = folder / f'{split}.npz'
    if path.exists():
        a = np.load(path)
        if str(a['checkpoint_sha256']) != dev.sha(cp):
            raise RuntimeError('Changed checkpoint')
        return a
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    if split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        raw = dev.TRAIN.make_window_view(np.asarray(full[valid], np.float32), 2)
        x = np.load(e.TRAINED / f'cache/deployment_event_psd/{dep}/real_features.npy')
    else:
        plan = dev.BASE.retained_event_plan(dep, es, split)
        full = dev.ORCH.event_array_for_plan(dep, es, split, plan)
        raw = dev.TRAIN.make_window_view(np.asarray(full, np.float32), 2)
        x = np.load(e.TRAINED / f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy')
    model = body.Encoder('RAW-PHASE-SOURCE').cuda().eval()
    model.load_state_dict(ck['model'])
    logits, z = body.infer(model, (x-ck['mu'])/ck['sd'], raw)
    p = softmax(logits.astype(float)/ck['temperature'], 1)
    if split == 'real':
        pp, zz = np.full((len(full), 64), np.nan), np.full((len(full), 128), np.nan)
        pp[valid], zz[valid] = p, z
        p, z = pp, zz
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, z=z, checkpoint_sha256=dev.sha(cp))
    return np.load(path)


def frame_features(root, dep, ms, es, split, prior):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = prediction(root, dep, ms, es, split)
    p, z = a['p'], a['z']
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    return f, {'mass': -np.log(np.sum(np.sqrt(p[i]*p[j]), 1).clip(1e-12)),
        'reference': np.log(np.sum(p[i]*p[j]/prior, 1).clip(1e-30)),
        'cosine': np.sum(z[i]*z[j], 1)}


def score(f, x, spec):
    penalty = np.minimum(np.log(tail.tail_probability(x['mass'], spec['reference_mass'])/.05), 0.)
    ood = (x['reference'] < spec['fit_min']) | (x['reference'] > spec['fit_max'])
    ref = x['reference'].clip(-4, 4)
    ref = np.where(ood, np.minimum(ref, 0), ref)
    return f.previous_waveform_score.to_numpy(float)+spec['gamma']*penalty+spec['beta']*ref, penalty, ref, ood


def evaluate(root):
    trial = root / 'trials/EXPANDED-SOURCE-RNC-ENCODER'
    if trial.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'calibration', 'evaluation', 'results', 'tables'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'formula': 'Zold+gamma*finite_Mc_reference_tail(newprediction)+beta*bounded_predictive_reference_overlap',
        'new_training_contract_sha256': dev.sha(root / 'expanded_encoder/contracts/TRAINING.json'),
        'gammas': GAMMAS, 'betas': BETAS, 'selection': 'BAYESTAR simulationvalidation only, original objective and FRT-relative guards',
        'reference': 'new512source independent developmentvalidation predictions; not a physical PE Bayes factor',
        'real_audit': 'adaptive development, not blindtest; independent source/noise confirmation required if target passes',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'historical outputs']})
    chosen, states, cache = {}, [], {}
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            a = np.load(root / f'expanded_encoder/models/{dep}/seed_{ms}/validation_predictions.npz')
            prior = a['prior']
            i, j = np.triu_indices(len(a['p']), 1)
            raw = np.log(np.sum(a['p'][i]*a['p'][j]/prior, 1).clip(1e-30))
            f, x = frame_features(root, dep, ms, es, 'validation', prior)
            cache[dep, es] = (f, x)
            y = f.is_true_pair.to_numpy(bool)
            cal = {'reference_mass': np.sort(x['mass'][y]).tolist(), 'fit_min': float(raw.min()),
                'fit_max': float(raw.max()), 'prior': prior.tolist(), 'deployment': dep, 'model_seed': ms, 'eval_seed': es}
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for gamma in GAMMAS:
                for beta in BETAS:
                    spec = {**cal, 'gamma': gamma, 'beta': beta}
                    z, _, _, ood = score(f, x, spec)
                    mm = ev.metrics(f, z, dep, es)
                    ok = ev.guard(mm, bm)
                    rows.append({'gamma': gamma, 'beta': beta, 'pass': ok, 'ood_fraction': float(ood.mean()),
                        **{a+'_'+k: v for a, b in mm.items() for k, v in b.items()}})
                    if ok:
                        choices.append((e.old_selection.objective(mm, gamma, beta), spec))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            state = {'deployment': dep, 'seed': es, 'eligible': len(choices)}
            if choices:
                spec = min(choices, key=lambda x: x[0])[1]
                chosen[dep, es] = spec
                dev.json_write(out / 'SELECTED_CONFIG.json', spec)
                state.update(gamma=spec['gamma'], beta=spec['beta'])
            states.append(state)
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'pass': len(chosen) == 6, 'real_used_to_select': False})
    print(json.dumps({'expanded_selection': states}), flush=True)
    if len(chosen) != 6:
        return
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            spec = chosen[dep, es]
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else frame_features(root, dep, ms, es, split, np.asarray(spec['prior']))
                z, penalty, reference, ood = score(f, x, spec)
                n = f.copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['expanded_mass_penalty'], n['expanded_reference'], n['expanded_ood'] = z, penalty, reference, ood
                for col in ('time_score', 'sky_raw_log_bf'):
                    assert np.array_equal(n[col], f[col])
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = n
                else:
                    bm, mm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es), ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm)})
                    for method in bm:
                        for config, m in [('FRT_BASELINE', bm[method]), ('CANDIDATE', mm[method])]:
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['features', 'train', 'evaluate'], required=True)
    p.add_argument('--deployment', choices=e.DEPS)
    a = p.parse_args()
    if a.stage == 'evaluate':
        evaluate(a.root)
    else:
        for dep in (a.deployment,) if a.deployment else e.DEPS:
            if a.stage == 'features':
                for split in ('validation', 'train'):
                    features(a.root, dep, split)
            else:
                for ms, seed in zip(body.MODEL_SEEDS, NEW_SEEDS):
                    train(a.root, dep, ms, seed)
