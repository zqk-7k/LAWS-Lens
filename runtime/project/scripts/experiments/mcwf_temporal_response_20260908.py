#!/usr/bin/env python3
"""Template temporal-response consistency, independent from time-delay evidence."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax

PROJECT = Path('/root/autodl-tmp/gw-catalog')
PREVIOUS = PROJECT / 'results/mcwf_pe_frontend_extension_20260907T041300Z'
PAIRS = PREVIOUS / 'trials/ORDERED-MASS-COMPLETE-GRID-PRIOR/evaluation'
EXTERNAL = PROJECT / 'results/mcwf_noise_context_exploratory_20260908T014829Z/audit'
sys.path.insert(0, str(PROJECT / 'scripts/experiments'))
import mcwf_ordered_mass_predictor_20260907 as old
import mcwf_fine_mass_features_20260907 as fine
import mcwf_finelag_eventpsd_20260907 as contexts
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_pe_frontend_extension_v2_20260907 as e

dev = e.dev
DEPS = ('gwtc3', 'gwtc4')
SEEDS = (202607241, 202607242, 202607243)
MODEL_SLOTS = (202609061, 202609062, 202609063)
TRAIN_SEEDS = (202609851, 202609852, 202609853)
NFFT, MAX_LAG = 8192, 304
RADII = (32, 175)
STATUS = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
torch.set_num_threads(2)


def protected():
    paths = list(PAIRS.glob('*/*/*_pairs.parquet'))
    paths += list((PREVIOUS / 'ordered_mass_predictor/models').glob('*/*/*.pt'))
    paths += list(EXTERNAL.glob('*_external_reference.parquet'))
    return [{'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size} for p in sorted(paths)]


def initialize(root):
    if root.exists():
        raise RuntimeError('Fresh independent output directory required')
    for name in ('contracts', 'scripts', 'logs', 'features', 'models', 'predictions',
                 'calibration', 'tables', 'reports', 'manifest', 'evaluation', 'figures'):
        (root / name).mkdir(parents=True)
    contract = {
        'utc': datetime.now(timezone.utc).isoformat(), 'id': 'MCWF-TEMPORAL-RESPONSE-01',
        'goal_status': 'ACTIVE_UNACHIEVED', 'release_status': STATUS,
        'baseline': str(PAIRS), 'same_algorithm_both_runs': True,
        'frozen': ['2s4096 H1L1 40-580Hz inputs', 'original encoders', 'one-dimensional time-delay score',
                   'sky_raw_log_bf', 'outer per-seed C-fixed weights', 'scope', 'historical results'],
        'new_mechanism': 'Shape of the matched-filter time series around its peak,not frequency-bin consistency already tested',
        'formula': 'a=c(t_peak); K_ba(delta)=<q_a,shift(q_b,delta)>; residual=c(t_peak+delta)-K(delta)*a; Xi=sum_delta||residual||^2/sum_delta(2-||K||_F^2)',
        'radii_samples': RADII, 'sampling_rate': 2048,
        'lag_search_samples': MAX_LAG, 'features': 'H1/L1 Xi at two radii,9 q/spin combinations,253 Mc centers:36 additional channels',
        'no_probability_claim': 'This cropped,window-standardized,peak-maximized feature is NOT a chi-square p-value,optimal SNR,PE likelihood or lens Bayes factor',
        'mismatch_caution': 'Aligned templates can penalize true precessing signals; do not veto. Train on existing XPHM injections and audit by spin/tilt/mass/SNR.',
        'training': {'input': 'same12288train/512development sourceparents and160/32noiseblocks',
                     'variants': ['CONTINUE', 'TEMPORAL'], 'epochs': 15, 'learning_rate': 1e-4,
                     'optimizer': 'AdamW,weight_decay1e-4,cosine_min1e-5,clip5',
                     'batch': '128sourceparents,2views/source', 'seeds': TRAIN_SEEDS,
                     'warm_start': 'corresponding original OMC checkpoint; extra input weights initialized zero; original coordinate weight retained',
                     'checkpoint_selection': 'minimum simulated-development CE,including epoch0',
                     'temperature': [.5,.75,1.,1.25,1.5,2.]},
        'selection': 'Simulated validation only; real PE/official labels appended after config freeze; all variants retained',
        'score': 'retained OMC waveform + finite-source mass-conflict term + bounded simulated-calibrated prior-overlap term',
        'gamma_grid': [0,.125,.25,.5,1,2], 'beta_grid': [0,.125,.25,.5,1,2],
        'calibration_fit': 'even hash-assigned development noise banks,drop systems crossing noise halves',
        'calibration_audit': 'other development noise half; final coefficients selected on existing BAYESTAR validation',
        'guardrails': 'per-seed simulated validation R10>=baseline-.02;AP>=baseline-.005;F50/F90<=1.1baseline;zero-increment baseline always available',
        'priority': 'candidate:F50,F90,-AP,-R10,-R1,minimum_coefficient_norm;retrieval:-R10,-R1,-AP,F50,F90,minimum_coefficient_norm',
        'external_target': 'both runs Top10/20 Mc counts and medianBC nondecreasing;zero catastrophic;Dmax pass counts nondecreasing;official1percent and resolvedHanabi nondecreasing;official strict gain per run and PE strict gain per run',
        'disclosure': 'Real catalogs repeatedly inspected in earlier studies:adaptive development,not blind confirmation. Fresh independent injection confirmation required for promotion.',
        'not_labels': 'official candidates and publicHanabi overlap are not lensing truths',
        'references': ['https://link.aps.org/accepted/10.1103/PhysRevD.95.042001',
                       'https://pmc.ncbi.nlm.nih.gov/articles/PMC7430253/',
                       'https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.023038']}
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(protected()))
    shutil.copy2(__file__, root / 'scripts/timeprofile.py')
    dev.json_write(root / 'contracts/CONTRACT_HASH.json', {
        'sha256': dev.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'code_sha256': dev.sha(root / 'scripts/timeprofile.py')})


def spectrum_bank():
    coarse = np.load(e.TRAINED / 'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy', mmap_mode='r')
    extra = np.load(PREVIOUS / 'fine_mass_context/cache/fine_mass_spectra.npy', mmap_mode='r')
    h = np.empty((253, 3, 3, coarse.shape[-1]), np.complex64)
    h[::4] = coarse.reshape(64, 3, 3, -1)
    h[np.arange(253) % 4 != 0] = extra.reshape(189, 3, 3, -1)
    return h.reshape(2277, -1)


@torch.no_grad()
def extract(raw, bank, batch=12, normalize=True, maximize=True):
    device = 'cuda'
    kernel = torch.fft.rfft(torch.as_tensor(bank, device=device), n=NFFT)
    offsets = torch.arange(-max(RADII), max(RADII)+1, device=device)
    lags = torch.arange(-MAX_LAG, MAX_LAG+1, device=device) if maximize else torch.zeros(1, device=device, dtype=torch.long)
    # Output indices b,a: correlation of template a against filter b.
    K = torch.fft.irfft(kernel[:, :, None, :, :] * kernel[:, :, :, None, :].conj(), n=NFFT)
    K = K[..., offsets.remainder(NFFT)]
    gram = K[..., max(RADII)]
    gram_error = float((gram-torch.eye(2, device=device)).abs().max())
    if gram_error > 1e-4:
        raise RuntimeError(f'Nonorthonormal quadratures:{gram_error}')
    variance = (2-K.square().sum((-3, -2))).clamp_min(0)
    radii_masks = [offsets.abs() <= r for r in RADII]
    denom = torch.stack([variance[..., mask].sum(-1) for mask in radii_masks], -1)
    if denom.min() < 1e-5:
        raise RuntimeError('Degenerate autocorrelation normalizer')
    result, power_result = [], []
    for start in range(0, len(raw), batch):
        x = torch.as_tensor(np.array(raw[start:start+batch], dtype=np.float32), device=device)
        if x.shape[1:] != (2, 4096) or not torch.isfinite(x).all():
            raise RuntimeError('Need complete finite H1L1 peak2s')
        if normalize:
            x = (x-x.mean(-1, keepdim=True))/x.std(-1, keepdim=True).clamp_min(1e-6)
        spectrum = torch.fft.rfft(x, n=NFFT)
        values, powers = [], []
        for d in range(2):
            rv, pv = [], []
            for lo in range(0, len(bank), 64):
                sl = slice(lo, lo+64)
                c = torch.fft.irfft(spectrum[:, d, None, None, :]*kernel[None, sl, d].conj(), n=NFFT)
                search = c[..., lags.remainder(NFFT)]
                p = search.square().sum(-2)
                peak = p.argmax(-1)
                a = search.gather(-1, peak[..., None, None].expand(-1,-1,2,1)).squeeze(-1)
                at = (lags[peak][..., None]+offsets).remainder(NFFT)
                observed = c.gather(-1, at[..., None, :].expand(-1,-1,2,-1))
                predicted = torch.einsum('kbal,nka->nkbl', K[sl,d], a)
                residual = (observed-predicted).square().sum(-2)
                xi = torch.stack([residual[..., mask].sum(-1) for mask in radii_masks], -1)/denom[None, sl, d]
                if not torch.isfinite(xi).all():
                    raise RuntimeError('Nonfinite temporal residual')
                rv.append(xi.cpu().numpy())
                pv.append(p.gather(-1, peak[..., None]).squeeze(-1).cpu().numpy())
            values.append(np.concatenate(rv,1))
            powers.append(np.concatenate(pv,1))
        result.append(np.stack(values,1))
        power_result.append(np.stack(powers,1))
    return np.concatenate(result), np.concatenate(power_result), {'quadrature_gram_max_error': gram_error}


def numerical_tests(bank):
    small = bank[:5]
    exact = (2*small[:,:,0]+small[:,:,1]).astype(np.float32)
    xi, power, audit = extract(exact, small, normalize=False, maximize=False)
    own = np.arange(len(small))
    residual = float(xi[own,:,own].max())
    power_error = float(abs(power[own,:,own]-5).max())
    if residual > 2e-6 or power_error > 1e-4:
        raise RuntimeError(f'Pure template consistency failed:{residual},{power_error}')
    # Independent CPU correlation verifies indexing and finite-window response.
    raw = np.random.default_rng(202609850).normal(size=(3,2,4096)).astype(np.float32)
    xi, powers, _ = extract(raw, small[:2], normalize=False, maximize=False)
    cpu = np.zeros_like(xi)
    for n in range(len(raw)):
        for d in range(2):
            for k in range(2):
                q = small[k,d].astype(float)
                a = q @ raw[n,d]
                for s, radius in enumerate(RADII):
                    numerator, denominator = 0., 0.
                    for delta in range(-radius,radius+1):
                        shifted = np.roll(np.pad(q,((0,0),(0,4096))),delta,axis=-1)[:,:4096]
                        K = shifted @ q.T
                        r = shifted @ raw[n,d]-K@a
                        numerator += float(r@r)
                        denominator += max(0.,2-float((K*K).sum()))
                    cpu[n,d,k,s] = numerator/denominator
    error = float(abs(cpu-xi).max())
    if error > 2e-5:
        raise RuntimeError(f'CPU finite-window residual mismatch:{error}')
    return {'pass': True, 'pure_template_residual': residual, 'pure_template_power_error': power_error,
            'cpu_gpu_max_abs_difference': error, **audit,
            'limits': 'Algebra tests only;finite crop and maximizing peak prevent calibrated chi-square interpretation.'}


def grouped(root, raw, freq, psds, ids, path, original):
    if path.with_suffix('.COMPLETE.json').exists():
        spec = json.loads(path.with_suffix('.COMPLETE.json').read_text())
        if dev.sha(path) != spec['sha256']:
            raise RuntimeError('Cached feature hash mismatch')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial.npy')
    mode = 'r+' if partial.exists() else 'w+'
    out = np.lib.format.open_memmap(partial, mode=mode, dtype=np.float32, shape=(len(raw),36,253))
    audit_path = path.with_suffix('.progress.json')
    rows = json.loads(audit_path.read_text()) if audit_path.exists() else []
    done = {r['psd_index'] for r in rows}
    h = spectrum_bank()
    for pid in np.unique(ids):
        if int(pid) in done:
            continue
        if shutil.disk_usage(root).free < 25*2**30:
            raise RuntimeError('HOLD_DISK_LIMIT')
        take = np.flatnonzero(ids == pid)
        started = time.perf_counter()
        bank = adaptive.whitened_bank(h, freq, psds[int(pid)])
        if not (root/'contracts/NUMERICAL_TESTS.json').exists():
            dev.json_write(root/'contracts/NUMERICAL_TESTS.json',numerical_tests(bank))
        xi, power, audit = extract(raw[take],bank)
        previous_power = original[take,:18].reshape(len(take),2,3,3,253).transpose(0,1,4,2,3).reshape(len(take),2,2277)
        discrepancy = float(abs(np.log1p(power)-previous_power).max())
        if discrepancy > .003:
            raise RuntimeError(f'Original matching powers changed:{discrepancy}')
        ordered = np.log1p(xi).reshape(len(take),2,253,3,3,2).transpose(0,1,5,3,4,2).reshape(len(take),36,253)
        if not np.isfinite(ordered).all():
            raise RuntimeError('Nonfinite features')
        out[take] = ordered
        out.flush()
        rows.append({'psd_index': int(pid), 'events': len(take), 'seconds': time.perf_counter()-started,
                     'original_logpower_max_abs_difference': discrepancy, **audit})
        dev.json_write(audit_path,rows)
        print(json.dumps({'feature':str(path),'psd':int(pid),'done':len(rows),'total':len(np.unique(ids)),**rows[-1]}),flush=True)
    del out
    partial.rename(path)
    dev.csv_write(path.with_suffix('.audit.csv'),pd.DataFrame(rows))
    dev.json_write(path.with_suffix('.COMPLETE.json'), {'sha256':dev.sha(path),'shape':[len(raw),36,253],
                  'seconds':sum(r['seconds'] for r in rows)})


def development_features(root,dep,split):
    prefix = PREVIOUS/f'expanded_data/{dep}'
    metadata = pd.read_parquet(prefix/f'{split}/event_metadata.parquet')
    c = np.load(PREVIOUS/f'expanded_encoder/features/{dep}/{split}.npy',mmap_mode='r')
    f = np.load(PREVIOUS/f'fine_mass_context/features/{dep}/{split}.npy',mmap_mode='r')
    grouped(root,np.load(prefix/f'{split}/raw2s.npy',mmap_mode='r'),np.load(prefix/'noise/frequency.npy'),
            np.load(prefix/'noise/psd.npy'),metadata.noise_bank_index.to_numpy(int),
            root/f'features/{dep}/{split}.npy',old.arrange(c,f))
    if split=='train':
        prefix=PREVIOUS/f'additional_population/expanded_data/{dep}'
        metadata=pd.read_parquet(prefix/'train/event_metadata.parquet')
        c=np.load(PREVIOUS/f'additional_population/expanded_encoder/features/{dep}/train.npy',mmap_mode='r')
        f=np.load(PREVIOUS/f'additional_population/features/{dep}/fine.npy',mmap_mode='r')
        grouped(root,np.load(prefix/'train/raw2s.npy',mmap_mode='r'),np.load(prefix/'noise/frequency.npy'),
                np.load(prefix/'noise/psd.npy'),metadata.noise_bank_index.to_numpy(int),
                root/f'features/{dep}/additional.npy',old.arrange(c,f))


def old_features(dep,es,split):
    coarse=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}'/('real_features.npy' if split=='real' else f'{es}_{split}_features.npy'))
    refined=np.load(PREVIOUS/f'fine_mass_context/features/{dep}'/('real.npy' if split=='real' else f'{es}_{split}.npy'))
    return old.arrange(coarse,refined)


def deployment_features(root,dep,es,split):
    path=root/f'features/{dep}'/('real.npy' if split=='real' else f'{es}_{split}.npy')
    if path.with_suffix('.COMPLETE.json').exists():
        return path
    freq,psds,ids,_=contexts.psd_context(dep,es,split)
    if split=='real':
        full,events=dev.real_inputs(dep)
        raw=dev.TRAIN.make_window_view(np.asarray(full[events.strict_h1l1_preprocessing_pass.to_numpy(bool)],np.float32),2)
    else:
        plan=dev.BASE.retained_event_plan(dep,es,split)
        full=dev.ORCH.event_array_for_plan(dep,es,split,plan)
        raw=dev.TRAIN.make_window_view(np.asarray(full,np.float32),2)
    grouped(root,raw,freq,psds,ids,path,old_features(dep,es,split))
    return path


class Predictor(old.Predictor):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind
        if kind=='TEMPORAL':
            self.first[0]=nn.Conv1d(64,48,9,padding=4)


def warm_model(kind,checkpoint):
    model=Predictor(kind).cuda()
    state={k:v.clone() for k,v in checkpoint['model'].items()}
    if kind=='TEMPORAL':
        weight=torch.zeros_like(model.first[0].weight,device='cpu')
        weight[:,:27]=state['first.0.weight'][:,:27]
        weight[:,-1]=state['first.0.weight'][:,-1]
        state['first.0.weight']=weight
    model.load_state_dict(state)
    return model


def training_data(root,dep,split,kind):
    x,meta=old.data(PREVIOUS,dep,split)
    if kind=='TEMPORAL':
        other=np.load(root/f'features/{dep}/{split}.npy',mmap_mode='r')
        if split=='train':
            other=np.concatenate([other,np.load(root/f'features/{dep}/additional.npy',mmap_mode='r')])
        x=np.concatenate([x,other],1)
    return x,meta


def train(root,dep,kind,slot,seed):
    out=root/f'models/{kind}/{dep}/seed_{slot}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(PREVIOUS/f'ordered_mass_predictor/models/{dep}/seed_{slot}/validation_selected_model.pt',map_location='cpu',weights_only=False)
    x,tm=training_data(root,dep,'train',kind);v,vm=training_data(root,dep,'validation',kind)
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):
        raise RuntimeError('Training-development source/noise overlap')
    mu,sd=checkpoint['mu'].copy(),checkpoint['sd'].copy()
    if kind=='TEMPORAL':
        mu=np.concatenate([mu,x[:,27:].mean((0,2),keepdims=True)],1)
        sd=np.concatenate([sd,x[:,27:].std((0,2),keepdims=True).clip(.01)],1)
    x-=mu;x/=sd;v=(v-mu)/sd
    group,names=pd.factorize(tm.source_uid,sort=True)
    members=[np.flatnonzero(group==i) for i in range(len(names))]
    vg,_=pd.factorize(vm.source_uid,sort=True)
    truth=np.log(tm.mc_det.to_numpy(float));vt=np.log(vm.mc_det.to_numpy(float))
    y,vy=old.targets(truth),old.targets(vt)
    dev.TRAIN.seed_everything(seed)
    model=warm_model(kind,checkpoint)
    xx=torch.as_tensor(x,device='cuda');yy=torch.as_tensor(y,device='cuda')
    del x
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,15,eta_min=1e-5)
    best=float('inf');history=[];first=0;started=time.perf_counter()
    if (out/'resume.pt').exists():
        resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
        history,best,first=resume['history'],resume['best'],resume['epoch']+1
        torch.set_rng_state(resume['rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
    for epoch in range(first,16):
        losses=[]
        if epoch:
            rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));model.train()
            for start in range(0,len(order),128):
                ids=np.concatenate([rng.choice(members[k],2,replace=False) for k in order[start:start+128]])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(xx[ids]);loss=-(F.log_softmax(logits.float(),-1)*yy[ids]).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
        logits=old.infer(model,v);p=softmax(logits.astype(float),1)
        ce=float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())
        row={'epoch':epoch,'validationCE':ce,'trainCE':float(np.mean(losses)) if losses else None,
             'logMc_MAE':float(abs(p@old.LOG_CENTERS-vt).mean()),'seconds':time.perf_counter()-started}
        history.append(row)
        if ce<best:
            best=ce
            torch.save({'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},'mu':mu,'sd':sd,
                        'kind':kind,'seed':seed,'slot':slot,'epoch':epoch,'CE':ce,'prior':checkpoint['prior']},out/'selected.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'resume.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        print(json.dumps({'training':[kind,dep,seed],**row}),flush=True)
    ck=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    logits=old.infer(model,v);grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.):
        p=softmax(logits.astype(float)/t,1)
        grid.append({'temperature':t,'CE':float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())})
    ck['temperature']=min(grid,key=lambda a:(a['CE'],abs(a['temperature']-1)))['temperature']
    torch.save(ck,out/'selected.pt')
    p,boundary=old.probability(logits,ck['temperature'])
    np.savez_compressed(out/'development_predictions.npz',p=p,outside=boundary,group=vg,truth=vt)
    cdf=np.c_[np.zeros(len(p)),p.cumsum(1)]
    pit=np.array([np.interp(t,old.EDGES,c) for t,c in zip(vt,cdf)])
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.json_write(out/'COMPLETE.json',{'epoch':ck['epoch'],'temperature':ck['temperature'],
        'logMc_MAE':float(abs(p@old.CENTERS-vt).mean()),'central90coverage':float(((pit>=.05)&(pit<=.95)).mean()),
        'training_sources':len(names),'development_sources':len(np.unique(vg)),
        'source_noise_overlap':0,'sha256':dev.sha(out/'selected.pt'),'seconds':time.perf_counter()-started})


def predict(root,dep,kind,slot,es,split):
    path=root/f'predictions/{kind}/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if path.exists():return np.load(path)
    cp=root/f'models/{kind}/{dep}/seed_{slot}/selected.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    x=old_features(dep,es,split)
    if kind=='TEMPORAL':
        x=np.concatenate([x,np.load(deployment_features(root,dep,es,split))],1)
    model=Predictor(kind).cuda().eval();model.load_state_dict(ck['model'])
    p,ood=old.probability(old.infer(model,(x-ck['mu'])/ck['sd']),ck['temperature'])
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp,oo=np.full((len(full),512),np.nan),np.full(len(full),np.nan)
        pp[valid],oo[valid]=p,ood;p,ood=pp,oo
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=p,outside=ood,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['initialize','features','train','predict'],required=True)
    parser.add_argument('--deployment',choices=DEPS)
    args=parser.parse_args()
    if args.stage=='initialize':initialize(args.root);return
    for dep in ((args.deployment,) if args.deployment else DEPS):
        if args.stage=='features':
            for split in ('validation','train'):development_features(args.root,dep,split)
        if args.stage=='train':
            for kind in ('CONTINUE','TEMPORAL'):
                for slot,seed in zip(MODEL_SLOTS,TRAIN_SEEDS):train(args.root,dep,kind,slot,seed)
        if args.stage=='predict':
            for kind in ('CONTINUE','TEMPORAL'):
                for slot,es in zip(MODEL_SLOTS,SEEDS):
                    for split in ('validation','test','real'):predict(args.root,dep,kind,slot,es,split)


if __name__=='__main__':main()
