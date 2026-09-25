#!/usr/bin/env python3
"""Clean-pretrained, multi-noise-adapted full-24-s waveform pilot."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import EvaluationSet, MatchArrays, ground_truth_partner, load_match_arrays, peak_flip_channels, split_indices, zscore_channels
from matchgw.matching import similarity_matrix
from matchgw.models import NTXentLoss
from scripts.experiments.mainline_uncertainty_common import seed_everything
from scripts.real_search.physical_common import effective_rank, write_json


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare(x: np.ndarray, rng: np.random.Generator, train: bool) -> np.ndarray:
    y = peak_flip_channels(np.asarray(x, dtype=np.float32).copy())
    if train:
        y = np.roll(y, int(rng.integers(-128, 129)), axis=-1)
        y *= float(1 + rng.uniform(-0.1, 0.1))
        y += rng.normal(0, 0.003 * (y.std(axis=-1, keepdims=True) + 1e-8), size=y.shape)
    return zscore_channels(y)


class CleanPairs(Dataset):
    def __init__(self, arrays, source_ids, seed):
        self.arrays=arrays;self.source_ids=np.asarray(source_ids,dtype=np.int64);self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.source_ids)
    def __getitem__(self,item):
        idx=int(self.source_ids[item]);return torch.from_numpy(prepare(self.arrays.l1_pure[idx],self.rng,True)),torch.from_numpy(prepare(self.arrays.l2_pure[idx],self.rng,True))


class EpochMultiNoise(Dataset):
    def __init__(self, root: Path, clean_arrays, seed: int):
        self.a=np.load(root/'noisy_image_a.npy',mmap_mode='r');self.b=np.load(root/'noisy_image_b.npy',mmap_mode='r')
        ids=np.load(root/'source_index.npy');variants=np.load(root/'variant.npy')
        self.sources=np.unique(ids);self.rows={int(src):np.flatnonzero(ids==src)[np.argsort(variants[ids==src])] for src in self.sources}
        self.clean=clean_arrays;self.rng=np.random.default_rng(seed);self.epoch=0
    def set_epoch(self,epoch:int): self.epoch=int(epoch)
    def __len__(self): return len(self.sources)
    def __getitem__(self,item):
        src=int(self.sources[item]);choices=self.rows[src];row=int(choices[(self.epoch+item)%len(choices)])
        return tuple(torch.from_numpy(prepare(x,self.rng,True)) for x in (self.a[row],self.b[row],self.clean.l1_pure[src],self.clean.l2_pure[src]))


@torch.no_grad()
def evaluate(model,dataset,batch_size):
    model.eval();z=[]
    for x in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True):
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):z.append(model(x.cuda(non_blocking=True)).float().cpu().numpy())
    z=np.concatenate(z);s=similarity_matrix(z);truth=ground_truth_partner(dataset.meta);r=[]
    for i,j in enumerate(truth):
        if j>=0:r.append(1+int(np.sum(s[i]>s[i,j])))
    r=np.asarray(r);upper=s[np.triu_indices(len(s),1)]
    return {'r_at_1':float(np.mean(r<=1)),'r_at_10':float(np.mean(r<=10)),'median_rank':float(np.median(r)),'embedding_effective_rank':effective_rank(z),'cosine_median':float(np.median(upper)),'cosine_std':float(np.std(upper))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed-root',type=Path,required=True);p.add_argument('--family',choices=('SIS','PM'),required=True);p.add_argument('--seed',type=int,default=202607211);p.add_argument('--samples',type=int,default=600);p.add_argument('--variants-per-source',type=int,default=8);p.add_argument('--pretrain-epochs',type=int,default=20);p.add_argument('--adapt-epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--architecture',choices=('full','dual'),default='full');p.add_argument('--adaptation',choices=('moving','teacher'),default='moving');args=p.parse_args()
    v3=module_from(REPO/'scripts/experiments/20_real_noise_injection_v3_physical.py','v3');specmod=module_from(REPO/'scripts/experiments/26_spectrogram_encoder_pilot.py','specmod')
    cfg=v3.training_config(args.seed_root,args.family,args.seed,args.samples,args.adapt_epochs,args.batch_size);cfg.use_pure_aux=True
    arrays=load_match_arrays(cfg);parts=split_indices(args.samples,args.samples,cfg)
    pure=MatchArrays(arrays.l1_pure,arrays.l2_pure,arrays.unlensed_pure)
    clean_train=CleanPairs(arrays,parts['lensed']['train'],cfg.seed+3000)
    multiroot=args.seed_root/'data/real_noise_injections'/f'multinoise_{args.family.lower()}_train_v{args.variants_per_source}'
    adapt_train=EpochMultiNoise(multiroot,arrays,cfg.seed+3001)
    pure_val=EvaluationSet(pure,parts['lensed']['val'],parts['unlensed']['val'],cfg);noisy_val=EvaluationSet(arrays,parts['lensed']['val'],parts['unlensed']['val'],cfg)
    Encoder=specmod.TriggerAlignedDualScaleSpectrogramEncoder if args.architecture=='dual' else specmod.Full24SpectrogramEncoder
    seed_everything(cfg.seed+3000);model=Encoder(base_channels=24).cuda();loss_fn=NTXentLoss(.10);history=[]
    loader=DataLoader(clean_train,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0,pin_memory=True);opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4);best=None;bestkey=None
    for epoch in range(1,args.pretrain_epochs+1):
        t=time.perf_counter();model.train();ls=[]
        for a,b in loader:
            a=a.cuda(non_blocking=True);b=b.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=loss_fn(model(a),model(b))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append(float(loss.detach().cpu()))
        row={'phase':'clean_pretrain','epoch':epoch,'loss':float(np.mean(ls)),'epoch_s':float(time.perf_counter()-t)}
        if epoch==1 or epoch%2==0 or epoch==args.pretrain_epochs:
            m=evaluate(model,pure_val,args.batch_size);row.update({f'pure_val_{k}':v for k,v in m.items()});key=(m['r_at_10'],m['r_at_1'],-m['median_rank'])
            if bestkey is None or key>bestkey:bestkey=key;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row);print(json.dumps(row),flush=True)
    model.load_state_dict(best)
    teacher=None
    if args.adaptation=='teacher':
        teacher=copy.deepcopy(model).eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=1e-4);bestm=None;beststate=None;bestkey=None
    for epoch in range(1,args.adapt_epochs+1):
        adapt_train.set_epoch(epoch);loader=DataLoader(adapt_train,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=0,pin_memory=True);t=time.perf_counter();model.train();ls=[]
        for na,nb,ca,cb in loader:
            na=na.cuda(non_blocking=True);nb=nb.cuda(non_blocking=True);ca=ca.cuda(non_blocking=True);cb=cb.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                zna=model(na);znb=model(nb);ln=loss_fn(zna,znb)
                if teacher is None:
                    zca=model(ca);zcb=model(cb);lc=loss_fn(zca,zcb);la=.5*((1-(zna*zca).sum(-1)).mean()+(1-(znb*zcb).sum(-1)).mean());loss=ln+.25*lc+.75*la
                else:
                    with torch.no_grad():prototype=torch.nn.functional.normalize(teacher(ca)+teacher(cb),dim=-1)
                    target=torch.arange(len(prototype),device=prototype.device)
                    lc=.5*(torch.nn.functional.cross_entropy(zna@prototype.T/.10,target)+torch.nn.functional.cross_entropy(znb@prototype.T/.10,target))
                    la=.5*((1-(zna*prototype).sum(-1)).mean()+(1-(znb*prototype).sum(-1)).mean())
                    loss=.5*ln+lc
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append((float(loss.detach().cpu()),float(ln.detach().cpu()),float(lc.detach().cpu()),float(la.detach().cpu())))
        q=np.asarray(ls);row={'phase':'multinoise_adaptation','epoch':epoch,'loss':float(q[:,0].mean()),'loss_noise':float(q[:,1].mean()),'loss_clean':float(q[:,2].mean()),'loss_align':float(q[:,3].mean()),'epoch_s':float(time.perf_counter()-t)}
        if epoch==1 or epoch%2==0 or epoch==args.adapt_epochs:
            m=evaluate(model,noisy_val,args.batch_size);row.update({f'noisy_val_{k}':v for k,v in m.items()});key=(m['r_at_10'],m['r_at_1'],-m['median_rank'],m['embedding_effective_rank'])
            if bestkey is None or key>bestkey:bestkey=key;bestm=m;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row);print(json.dumps(row),flush=True)
    architecture_name=Encoder.__name__;out=args.seed_root/'waveform_gate'/f'{args.family.lower()}_{args.architecture}_{args.adaptation}_multinoise_curriculum_pilot';out.mkdir(parents=True,exist_ok=True);ck=out/'validation_selected_model.pt';torch.save({'model_state':beststate,'family':args.family,'architecture':architecture_name,'adaptation':args.adaptation,'objective':'clean pretrain + rotating independent real-noise adaptation'},ck);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    summary={'family':args.family,'architecture':architecture_name,'adaptation':args.adaptation,'n_unique_training_sources':len(adapt_train),'variants_per_source':args.variants_per_source,'validation_test_modified':False,'selection_split':'validation only','heldout_test_was_evaluated':False,'best_noisy_validation':bestm,'checkpoint':str(ck)};write_json(out/'multinoise_curriculum_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
