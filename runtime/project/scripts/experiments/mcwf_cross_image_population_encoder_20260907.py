#!/usr/bin/env python3
"""Paired-noise versus paired-image source-contrastive training ablation."""
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
import mcwf_additional_population_features_20260907 as population
import mcwf_expanded_encoder_20260907 as original_expanded

dev, body, ev = e.dev, e.body, e.ev
torch.set_num_threads(2)
NEW_SEEDS = (202609301, 202609302, 202609303)
EPOCHS = 17
GAMMAS = (.125, .25, .5, 1., 2., 4.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)


def initialize(root):
    folder = root / 'cross_image_population_encoder'
    path = folder / 'contracts/TRAINING.json'
    if path.exists():
        return
    contract = root / 'expanded_data/contracts/DATA_CONTRACT.json'
    dev.json_write(path, {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'single_controlled_change': 'each positive source pair contains one image a and one image b, rather than two arbitrary noise views; identical data, architecture, source batch order, epochs, optimizer and initialization as large_population_encoder',
        'paired_control_seeds': '301/302/303 reused for a paired ablation, not additional independent seeds',
        'cross_image_rule': 'sample one of the four noise views from each image, then flatten all a views followed by all b views; no real PE/official inputs',
        'data_contract_sha256': dev.sha(contract),
        'additional_data_contract_sha256': dev.sha(root / 'additional_population/expanded_data/contracts/DATA_CONTRACT.json'),
        'training_seeds': NEW_SEEDS,
        'validation': 'unchanged512 development source parents,1024views,32noise blocks disjoint from training; reused for adaptive development, not a new blind test',
        'normalization': 'keep original RNC phase-feature mean/SD and input preprocessor to preserve the warm-start function exactly',
        'optimizer': 'AdamW1e-4,weightdecay1e-4,cosine_min1e-5,17epochs,64sources*2views,onepass/epoch,clip5',
        'step_budget': '12288/64*17=3264 updates, exactly matched to larger-population random-view control',
        'loss': 'soft logMc64bin CE+sourceSupCon+RNC(logMc)',
        'selection': 'earliest minimum newvalidation CE+0.2SupCon+0.2RNC,includingepoch0; temperature by CE only',
        'epoch_zero': 'no encoder improvement claim if unchanged model is selected; new temperature may differ and is separately disclosed',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'all oldmodel and result files'],
        'no_real_PE_or_official_training': True,
        'interpretation': 'adaptive development followed by a required new independent confirmation if all real-development targets are met',
    })
    (folder / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, folder / 'scripts' / Path(__file__).name)


def data(root,dep,split,ck):
    meta=population.metadata(root,dep,split)
    a=np.load(root/f'expanded_data/{dep}/{split}/raw2s.npy',mmap_mode='r')
    x=np.load(root/f'expanded_encoder/features/{dep}/{split}.npy')
    if split=='train':
        b=np.load(root/f'additional_population/expanded_data/{dep}/train/raw2s.npy',mmap_mode='r')
        raw=np.concatenate([a,b],axis=0)
        nx=np.load(root/f'additional_population/expanded_encoder/features/{dep}/train.npy')
        x=np.concatenate([x,nx],axis=0)
    else:
        raw=a
    x=(x-ck['mu'])/ck['sd']
    y=tf.target_prob(np.log(meta.mc_det.to_numpy(float)))
    group,names=pd.factorize(meta.source_uid,sort=True)
    return x,raw,y,meta,group,names


def train(root, dep, ms, seed):
    initialize(root)
    out = root / f'cross_image_population_encoder/models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    gate_path = root / f'additional_population/features/{dep}/DATA_SCALING_GATE.json'
    coarse_path = root / f'additional_population/expanded_encoder/features/{dep}/train.npy'
    if not gate_path.exists() or not json.loads(gate_path.read_text())['pass'] or not coarse_path.exists():
        raise RuntimeError('Independent-source/noise audit and coarse features must complete first')
    for split in ('train','validation'):
        original_expanded.features(root,dep,split)
    checkpoint = e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
    tx, raw, y, meta, group, names = data(root, dep, 'train', ck)
    vx, vr, vy, vm, vg, vnames = data(root, dep, 'validation', ck)
    if set(names) & set(vnames) or set(meta.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Source/noise split overlap')
    dev.json_write(out / 'SPLIT_AUDIT.json', {'train_sources': len(names), 'validation_sources': len(vnames),
        'source_overlap': 0, 'noise_overlap': 0, 'data_noise_audit': str(root / f'additional_population/features/{dep}/DATA_SCALING_GATE.json')})
    members = [np.flatnonzero(group == k) for k in range(len(names))]
    image_members = [{im: m[meta.iloc[m].image.to_numpy() == im] for im in ('a', 'b')} for m in members]
    if not all(len(v['a']) == 4 and len(v['b']) == 4 for v in image_members):
        raise RuntimeError('Expected four noise views per image and two images per source')
    dev.json_write(out / 'POSITIVE_VIEW_AUDIT.json', {
        'sources': len(members), 'views_per_image': 4, 'images_per_positive_pair': 2,
        'previous_probability_same_image': 3 / 7, 'new_probability_same_image': 0,
        'metadata_sha256_roles': 'same source/noise population as paired control'})
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
                indices = np.stack([[rng.choice(image_members[k]['a']), rng.choice(image_members[k]['b'])] for k in sources]).T.reshape(-1)
                if not (np.all(meta.iloc[indices[:len(sources)]].image == 'a') and np.all(meta.iloc[indices[len(sources):]].image == 'b')):
                    raise RuntimeError('Positive view pairing changed')
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
        if epoch == 0:
            old = pd.read_csv(root / f'large_population_encoder/models/{dep}/seed_{ms}/history.csv').iloc[0]
            differences = {key: abs(float(row[key]) - float(old[key])) for key in ('CE', 'SupCon', 'RNC', 'criterion', 'logMc_MAE')}
            if max(differences.values()) > 1e-5:
                raise RuntimeError('Paired control initialization mismatch')
            dev.json_write(out / 'PAIRED_INITIALIZATION_AUDIT.json', {'pass': True, 'absolute_differences': differences})
        history.append(row)
        if criterion < best:
            best = criterion
            torch.save({**ck, 'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'epoch': epoch, 'criterion': criterion, 'new_training_seed': seed,
                'warm_sha256': dev.sha(checkpoint), 'training_population': str(root / 'expanded_data'),
                'training_contract_sha256': dev.sha(root / 'cross_image_population_encoder/contracts/TRAINING.json')}, out / 'validation_selected_model.pt')
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
    cp = root / f'cross_image_population_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Training incomplete')
    folder = root / f'cross_image_population_encoder/predictions/{dep}/model_{ms}_eval_{es}'
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--deployment',choices=e.DEPS,required=True)
    args=parser.parse_args()
    for ms,seed in zip(body.MODEL_SEEDS,NEW_SEEDS):
        train(args.root,args.deployment,ms,seed)

