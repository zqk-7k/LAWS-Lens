#!/usr/bin/env python3
"""Noise-PSD-conditioned phase-feature Mc head; only simulated supervision."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_phasebank_20260905 as phase
import mcwf_mass_tf_20260905 as tf
import mcwf_mass_lr_20260905 as lr

FREQS=np.geomspace(40,580,16)


def psd_features(f,p):
    a=np.array([[np.interp(FREQS,f,ch) for ch in row] for row in p])
    if not np.all(np.isfinite(a)&(a>0)):raise ValueError("Invalid PSD")
    l=np.log(a);l-=l.mean(-1,keepdims=True)
    return l.reshape(len(a),-1).astype(np.float32)


def development_psd(root,dep,split):
    cache=root/f"cache/phasepsd/{dep}";cache.mkdir(parents=True,exist_ok=True)
    dest=cache/f"{split}_psd_features.npy"
    if dest.exists():return np.load(dest)
    bank=dev.OLD/f"data/noise_banks/{dep}"
    f=np.load(bank/f"noise_psd_frequency_{split}.npy")
    p=np.load(bank/f"noise_psd_{split}.npy",mmap_mode="r")
    features=psd_features(f,p);parts=[]
    for family in ("sis","pm"):
        m=pd.read_parquet(dev.OLD/f"data/development/{dep}/{family}_{split}_metadata.parquet")
        for image in ("a","b"):
            parts.append(features[m[f"{image}_noise_bank_index"].to_numpy(int)])
    out=np.concatenate(parts);np.save(dest,out);return out


def real_psd(root,dep):
    cache=root/f"cache/phasepsd/{dep}";cache.mkdir(parents=True,exist_ok=True)
    dest=cache/"real_psd_features.npy"
    if dest.exists(): return np.load(dest)
    _,events=dev.real_inputs(dep);v3=dev.BASE.v7.v3
    source=dev.MAIN/"cache/source_run" if dep=="gwtc3" else dev.PROJECT/"runs/real_gwtc34_lensing_search_20260629_full_o4"
    hdf=v3.HdfCache(source,max_files=4)
    out=np.full((len(events),32),np.nan,dtype=np.float32);audits=[];psds=[]
    for r in events.itertuples():
        if not r.strict_h1l1_preprocessing_pass:continue
        refs=[]
        for info in json.loads(r.detector_audit):
            data,start,duration=hdf.get(str(info["path"]))
            ref,t0=v3._extract_psd_reference(data,start,float(r.gps_time))
            if ref is None:raise RuntimeError("Reference no longer available")
            if abs(t0-float(info["psd_reference_start_gps"]))>1/4096:raise RuntimeError("PSD window changed")
            refs.append(ref)
            audits.append({"event_name":r.event_name,"detector":info["detector"],"source_path":info["path"],"psd_start_gps":t0})
        f,p=v3.estimate_psd(np.stack(refs))
        out[r.idx]=psd_features(f,p[None])[0];psds.append({"idx":r.idx,"psd":p})
    np.save(dest,out)
    np.savez_compressed(cache/"real_reference_PSDs.npz",frequency=f,psd=np.stack([a["psd"] for a in psds]),idx=np.array([a["idx"] for a in psds]))
    dev.csv_write(cache/"real_psd_source_manifest.csv",pd.DataFrame(audits))
    return out


class MassPSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(3*64*3*3+32,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.15),nn.Linear(256,128),nn.SiLU())
        self.head=nn.Linear(128,64)
    def forward(self,x):
        z=self.body(x);return self.head(z),F.normalize(z,dim=-1)


def train(root,dep,seed):
    contract=root/"contracts/ROUND4_PSD_CONDITIONED.json"
    if not contract.exists():
        dev.json_write(contract,{"rationale":"fixed median-PSD phase filters see a family of differently whitened event shapes; supply actual conditioning PSD to mass head",
            "input":"same2s4096 H1L1 phase-filter features plus 16 logPSD shape samples per detector; off-source conditioning, not a fourth ranking channel",
            "PSD_frequency_Hz":FREQS.tolist(),"PSD_units":"strain^2/Hz before logarithm; per-detector subtract mean to remove arbitrary amplitude scale",
            "training":"same isolated simulations; CE,50epochs,three frozen seeds; validationCE temperature only",
            "real_PSD":"recompute from exact frozen 256s off-source window used by old preprocessing; verify start matches audit",
            "time_sky_fusion_frozen":True,"PE_official_as_inputs":False,"reference":"https://arxiv.org/abs/2106.12594",
            "status":"exploratory developmental change; no claim of proper PE"})
    out=root/f"models/PHASEPSD/{dep}/seed_{seed}"
    if (out/"COMPLETE.json").exists():return
    out.mkdir(parents=True,exist_ok=True)
    x=np.load(root/f"cache/phasebank/{dep}/train_features.npy").reshape(-1,3*64*3*3)
    v=np.load(root/f"cache/phasebank/{dep}/validation_features.npy").reshape(-1,3*64*3*3)
    x=np.concatenate([x,development_psd(root,dep,"train")],axis=1)
    v=np.concatenate([v,development_psd(root,dep,"validation")],axis=1)
    y=np.load(root/f"cache/masstf/{dep}/train_targets.npy");vy=np.load(root/f"cache/masstf/{dep}/validation_targets.npy")
    mu=x.mean(0);sd=x.std(0).clip(.05);x=(x-mu)/sd;v=(v-mu)/sd
    dev.ORCH.TRAIN.seed_everything(seed);model=MassPSD().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=3e-5)
    history=[];best=np.inf
    for epoch in range(1,51):
        model.train();losses=[]
        for ids in np.array_split(np.random.default_rng(seed+epoch).permutation(len(x)),int(np.ceil(len(x)/96))):
            opt.zero_grad(set_to_none=True)
            logits,_=model(torch.as_tensor(x[ids],device="cuda"));yy=torch.as_tensor(y[ids],device="cuda")
            loss=-(F.log_softmax(logits,dim=-1)*yy).sum(-1).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        sched.step();logits,_=tf.infer_logits(model,v)
        lp=logits-np.logaddexp.reduce(logits,axis=-1,keepdims=True);ce=float(-(lp*vy).sum(-1).mean())
        history.append({"epoch":epoch,"train_ce":np.mean(losses),"validation_ce":ce,"validation_logmc_mae":float(np.mean(abs(np.exp(lp)@tf.LOG_CENTERS-vy@tf.LOG_CENTERS)))})
        if ce<best:
            best=ce;prior=y.mean(0)+1e-4;prior/=prior.sum()
            torch.save({"state":{k:v.cpu().clone() for k,v in model.state_dict().items()},"prior":prior,"mu":mu,"sd":sd,"epoch":epoch,"seed":seed,"deployment":dep,
                "bankpath":str(root/f"cache/phasebank/{dep}/quadrature_bank.npy")},out/"validation_selected_model.pt")
        dev.csv_write(out/"training_history.csv",pd.DataFrame(history))
    ck=torch.load(out/"validation_selected_model.pt",weights_only=False,map_location="cpu");model.load_state_dict(ck["state"])
    logits,emb=tf.infer_logits(model,v);grid=[]
    for temp in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        lp=logits/temp-np.logaddexp.reduce(logits/temp,axis=-1,keepdims=True);grid.append({"temperature":temp,"ce":float(-(vy*lp).sum(-1).mean())})
    ck["temperature"]=min(grid,key=lambda r:(r["ce"],abs(r["temperature"]-1)))["temperature"]
    torch.save(ck,out/"validation_selected_model.pt");dev.csv_write(out/"temperature_grid.csv",pd.DataFrame(grid))
    np.savez_compressed(out/"development_validation_predictions.npz",logits=logits,truth=vy,embedding=emb)
    dev.json_write(out/"COMPLETE.json",{"real_inputs_in_training":False,"checkpoint_sha256":dev.sha(out/"validation_selected_model.pt")})


class Encoder:
    def __init__(self,root,dep,seed):self.root=root;self.dep=dep;self.seed=seed
    @torch.no_grad()
    def __call__(self,checkpoint,full,split):
        ck=torch.load(checkpoint,weights_only=False,map_location="cpu");model=MassPSD().cuda().eval();model.load_state_dict(ck["state"])
        feat=phase.features(full,np.load(ck["bankpath"])).reshape(len(full),-1)
        if split=="real":
            _,events=dev.real_inputs(self.dep);pfeat=real_psd(self.root,self.dep)[events.strict_h1l1_preprocessing_pass.to_numpy(bool)]
        else:
            plan=dev.BASE.retained_event_plan(self.dep,self.seed,split)
            shared=dev.ORCH.SOURCE_ROOT/self.dep/"shared"
            f=np.load(shared/"noise_psd_frequency.npy");p=np.load(shared/"noise_psd_bank.npy",mmap_mode="r")
            pfeat=psd_features(f,p)[plan.parent_noise_bank.to_numpy(int)]
        if len(pfeat)!=len(full) or not np.isfinite(pfeat).all():raise RuntimeError("PSD/event alignment failure")
        x=np.concatenate([feat,pfeat],axis=1);x=(x-ck["mu"])/ck["sd"]
        logits,emb=tf.infer_logits(model,x);logits=logits.astype(float)/ck["temperature"]
        return np.exp(logits-np.logaddexp.reduce(logits,axis=-1,keepdims=True)),emb,ck


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--deployment",required=True)
    p.add_argument("--seed",type=int,default=tf.SEEDS[0]);p.add_argument("--eval-seed",type=int,default=dev.SEEDS[0]);p.add_argument("--phase",choices=("train","evaluate"),required=True);a=p.parse_args()
    if a.phase=="train":train(a.root,a.deployment,a.seed)
    else:tf.evaluate(a.root,a.deployment,a.seed,a.eval_seed,config="PHASEPSD-MASSLR",model_config="PHASEPSD",
                    mass_feature_builder=lr.MassLR(),context_encoder=Encoder(a.root,a.deployment,a.eval_seed))
