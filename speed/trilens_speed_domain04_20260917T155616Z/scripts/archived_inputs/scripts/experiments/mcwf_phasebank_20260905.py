#!/usr/bin/env python3
"""Simulation-trained mass head on phase-sensitive waveform-filter features.

The template bank is an approximate feature extractor, not a PE likelihood.
All weights/calibration use simulation development data, never public PE.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import signal
from pycbc.waveform import get_fd_waveform
import torch
from torch import nn
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf

CONFIG = "PHASEBANK"
LAGS = 304
STEP = 16


def initialize(root):
    path = root / "contracts/ROUND2_PHASEBANK.json"
    if path.exists(): return
    dev.json_write(path, {
        "round": "R2-PHASEBANK", "input": "unchanged peak2s H1/L1,4096samples,2048Hz",
        "model": "fixed physical quadrature-filter bank plus simulation-trained 64-bin predictive logMc head",
        "bank": {"waveform": "IMRPhenomD", "Mc": "64 logarithmic bin centers5-200Msun", "q": [.25,.5,1.], "chi_eff": [-.5,0.,.5],
                 "PSD": "per-detector median of development TRAIN noise PSD only", "frequency": [40,580],
                 "phase": "two orthonormal quadratures; no event PE parameters", "event_lag_s": [-LAGS/2048,LAGS/2048], "lag_step_s": STEP/2048},
        "limits": "approximate phase-sensitive features, no proper likelihood/evidence claim; no bank minimal-match guarantee; precession/higher-mode mismatch measured on independent development waveforms",
        "features": "max over lag of H1,L1 quadrature power plus near-coincident network power; log1p, mass/q/spin grid retained",
        "training": "same source-disjoint development bank as round1; 50epochs,AdamW3e-4,earliest minimum validationCE",
        "temperature": "same validationCE-only grid as MASS-TF",
        "scoring": "same frozen validation-only negative_guard and mass_cap grids/guardrails as round1; no reweighting time or sky",
        "seeds": list(tf.SEEDS), "real_PE_and_official_inputs": False,
        "references": ["https://pycbc.org/pycbc/latest/html/pycbc.filter.html", "https://arxiv.org/abs/2106.12594"],
    })


def make_bank(root, dep):
    out = root / f"cache/phasebank/{dep}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "quadrature_bank.npy"
    if path.exists(): return path
    source = dev.OLD / f"data/noise_banks/{dep}"
    psds = np.load(source / "noise_psd_train.npy", mmap_mode="r")
    pf = np.load(source / "noise_psd_frequency_train.npy")
    median = np.median(psds, axis=0)
    fs, n = 2048, 16*2048
    freq = np.fft.rfftfreq(n,1/fs)
    window = signal.windows.tukey(4096,.05)
    # Apply the same physical band, with zero-phase Butterworth amplitude response.
    sos = signal.butter(6,[40,580],fs=fs,btype="bandpass",output="sos")
    _, response = signal.sosfreqz(sos, worN=freq, fs=fs)
    filt = np.abs(response)**2
    bank, meta, gram_errors = [], [], []
    for mc in np.exp(tf.LOG_CENTERS):
        for q in (.25,.5,1.):
            for spin in (-.5,0.,.5):
                m1=mc*(1+q)**.2/q**.6
                hp,_=get_fd_waveform(approximant="IMRPhenomD",mass1=m1,mass2=m1*q,
                    spin1z=spin,spin2z=spin,delta_f=1/16,f_lower=30,f_final=580,distance=1000,inclination=0)
                hp.resize(len(freq))
                a=np.fft.irfft(np.asarray(hp),n=n)
                peak=int(np.argmax(np.abs(signal.hilbert(a))))
                destination=14*fs
                buffer=np.roll(a,destination-peak)
                h=np.fft.rfft(buffer)
                detectors=[]
                for d in range(2):
                    p=np.interp(freq,pf,median[d]); p=np.maximum(p,np.max(p)*1e-20)
                    white=np.fft.irfft(h/np.sqrt(p)*filt,n=n)
                    analytic=signal.hilbert(white)
                    offset=destination-int(1.75*fs)
                    u=analytic.real[offset:offset+4096]*window
                    v=analytic.imag[offset:offset+4096]*window
                    u-=u.mean(); v-=v.mean()
                    u/=np.linalg.norm(u)
                    v-=np.dot(u,v)*u; v/=np.linalg.norm(v)
                    gram_errors.append(max(abs(np.dot(u,v)),abs(np.dot(u,u)-1),abs(np.dot(v,v)-1)))
                    detectors.append(np.stack([u,v]).astype(np.float32))
                bank.append(detectors)
                meta.append({"mc":mc,"q":q,"chi_eff":spin})
    bank=np.stack(bank) # template,detector,quadrature,time
    np.save(path,bank)
    dev.csv_write(out/"bank_parameters.csv",pd.DataFrame(meta))
    dev.json_write(out/"BANK_TESTS.json", {"n_templates":len(bank),"finite":bool(np.isfinite(bank).all()),
        "max_quadrature_gram_error":max(gram_errors),"psd_source_sha256":dev.sha(source/"noise_psd_train.npy"),
        "source_parameters_from_PE":False,"bank_sha256":dev.sha(path)})
    return path


@torch.no_grad()
def features(x, bank, batch=48):
    templates=torch.as_tensor(bank,dtype=torch.float32,device="cuda")
    outputs=[]
    for start in range(0,len(x),batch):
        values=torch.as_tensor(np.asarray(x[start:start+batch,:,-4096:],dtype=np.float32),device="cuda")
        if values.shape[1:]!=(2,4096) or not torch.isfinite(values).all(): raise ValueError("Invalid H1L1")
        values=(values-values.mean(-1,keepdim=True))/values.std(-1,keepdim=True).clamp_min(1e-6)
        powers=[]
        for d in range(2):
            shifted=F.pad(values[:,d],(LAGS,LAGS)).unfold(-1,4096,STEP)
            kernel=templates[:,d].reshape(-1,4096).T
            with torch.autocast("cuda",dtype=torch.bfloat16):
                corr=shifted @ kernel
            power=corr.float().reshape(len(values),-1,len(bank),2).square().sum(-1)
            powers.append(power)
        # Allow at most one lag step on either side for H1/L1 travel-time difference.
        l1=F.max_pool1d(powers[1].transpose(1,2),3,stride=1,padding=1).transpose(1,2)
        net=(powers[0]+l1).amax(1)
        feat=torch.stack([p.amax(1) for p in powers]+[net],dim=1)
        feat=torch.log1p(feat).reshape(len(values),3,64,3,3)
        outputs.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


class MassPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Flatten(),nn.Linear(3*64*3*3,256),nn.LayerNorm(256),nn.SiLU(),
            nn.Dropout(.15),nn.Linear(256,128),nn.SiLU())
        self.head=nn.Linear(128,64)
    def forward(self,x):
        z=self.layers(x)
        return self.head(z),F.normalize(z,dim=-1)


def data(root,dep,bank):
    out=root/f"cache/phasebank/{dep}"
    if (out/"DATA_COMPLETE.json").exists(): return out
    tf.prepare_data(root,dep)
    paths=dev.TRAIN.cache_development_views(dev.OLD,dep,(2,))[2]
    for split in ("train","validation"):
        parts=[]
        for family in ("sis","pm"):
            for image in ("a","b"):
                x=np.load(paths[f"{family}_{split}_{image}"],mmap_mode="r")
                parts.append(features(x,bank))
        x=np.concatenate(parts)
        np.save(out/f"{split}_features.npy",x)
        print({"features":split,"shape":x.shape},flush=True)
    dev.json_write(out/"DATA_COMPLETE.json",{"order":"same as round1 target array","n_strain_created":0})
    return out


def train(root,dep,seed):
    initialize(root)
    out=root/f"models/{CONFIG}/{dep}/seed_{seed}"
    if (out/"COMPLETE.json").exists(): return
    out.mkdir(parents=True,exist_ok=True)
    bankpath=make_bank(root,dep); bank=np.load(bankpath)
    cache=data(root,dep,bank)
    x=np.load(cache/"train_features.npy"); v=np.load(cache/"validation_features.npy")
    labels=root/f"cache/masstf/{dep}"
    y=np.load(labels/"train_targets.npy"); vy=np.load(labels/"validation_targets.npy")
    mu=x.mean(0); sd=x.std(0).clip(.05)
    x=(x-mu)/sd; v=(v-mu)/sd
    dev.ORCH.TRAIN.seed_everything(seed)
    model=MassPhase().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=3e-5)
    history=[]; best=np.inf
    for epoch in range(1,51):
        t=time.perf_counter(); model.train(); losses=[]
        order=np.random.default_rng(seed+epoch).permutation(len(x))
        for ids in np.array_split(order,int(np.ceil(len(x)/96))):
            xx=torch.as_tensor(x[ids],device="cuda"); yy=torch.as_tensor(y[ids],device="cuda")
            opt.zero_grad(set_to_none=True)
            pred,_=model(xx)
            loss=-(F.log_softmax(pred,dim=-1)*yy).sum(-1).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        logits,_=tf.infer_logits(model,v)
        lp=logits-np.logaddexp.reduce(logits,axis=-1,keepdims=True)
        ce=float(-(vy*lp).sum(-1).mean())
        mae=float(np.mean(abs(np.exp(lp)@tf.LOG_CENTERS-vy@tf.LOG_CENTERS)))
        history.append({"epoch":epoch,"train_ce":np.mean(losses),"validation_ce":ce,"validation_logmc_mae":mae,"seconds":time.perf_counter()-t})
        if ce<best:
            best=ce; prior=y.mean(0)+1e-4; prior/=prior.sum()
            torch.save({"state":{k:v.cpu().clone() for k,v in model.state_dict().items()},"prior":prior,
                "mu":mu,"sd":sd,"bankpath":str(bankpath),"epoch":epoch,"seed":seed,"deployment":dep},out/"validation_selected_model.pt")
        dev.csv_write(out/"training_history.csv",pd.DataFrame(history))
        if epoch%5==0: print(history[-1],flush=True)
    ck=torch.load(out/"validation_selected_model.pt",weights_only=False,map_location="cpu"); model.load_state_dict(ck["state"])
    logits,emb=tf.infer_logits(model,v)
    grid=[]
    for temp in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        lp=logits/temp-np.logaddexp.reduce(logits/temp,axis=-1,keepdims=True)
        grid.append({"temperature":temp,"ce":float(-(vy*lp).sum(-1).mean())})
    ck["temperature"]=min(grid,key=lambda r:(r["ce"],abs(r["temperature"]-1)))["temperature"]
    torch.save(ck,out/"validation_selected_model.pt")
    dev.csv_write(out/"temperature_grid.csv",pd.DataFrame(grid))
    np.savez_compressed(out/"development_validation_predictions.npz",logits=logits,truth=vy,embedding=emb)
    dev.json_write(out/"COMPLETE.json",{"epoch":ck["epoch"],"temperature":ck["temperature"],"real_training_inputs":False,"checkpoint_sha256":dev.sha(out/"validation_selected_model.pt")})


@torch.no_grad()
def encode(checkpoint,full24):
    ck=torch.load(checkpoint,weights_only=False,map_location="cpu")
    model=MassPhase().cuda().eval(); model.load_state_dict(ck["state"])
    x=features(full24,np.load(ck["bankpath"]))
    logits,emb=tf.infer_logits(model,(x-ck["mu"])/ck["sd"])
    logits=logits.astype(float)/ck["temperature"]
    p=np.exp(logits-np.logaddexp.reduce(logits,axis=-1,keepdims=True))
    return p,emb,ck


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--deployment",choices=("gwtc3","gwtc4"),required=True)
    p.add_argument("--seed",type=int,default=tf.SEEDS[0]); p.add_argument("--eval-seed",type=int,default=dev.SEEDS[0])
    p.add_argument("--phase",choices=("train","evaluate"),required=True)
    a=p.parse_args()
    if a.phase=="train": train(a.root,a.deployment,a.seed)
    else: tf.evaluate(a.root,a.deployment,a.seed,a.eval_seed,config=CONFIG,encoder=encode)
