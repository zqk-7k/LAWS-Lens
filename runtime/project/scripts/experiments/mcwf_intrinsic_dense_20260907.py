#!/usr/bin/env python3
"""Structured 3-D template-response head; old fine-Mc marginal remains frozen."""
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
from scipy.ndimage import gaussian_filter

import mcwf_intrinsic_grid_20260907 as g
import mcwf_dense_features_20260907 as dense

g.Q=np.asarray(dense.QGRID);g.CHI=np.asarray(dense.SPINGRID);g.SHAPE=(253,len(g.Q),len(g.CHI))
dev,e,mass=g.dev,g.e,g.mass
SEEDS=(202609801,202609802,202609803)
N_M=27*253
OLD=np.array([k*7+l for k,q in enumerate(g.Q) for l,s in enumerate(g.CHI)
              if q in (.25,.5,1.) and s in (-.5,0.,.5)])
EXTRA=np.array([i for i in range(56) if i not in set(OLD)])


def grid(coarse,extra):
    n=len(coarse)
    out=np.empty((n,3,64,56),np.float32)
    out[...,OLD]=np.asarray(coarse).reshape(n,3,64,9)
    out[...,EXTRA]=np.asarray(extra).reshape(n,3,64,47)
    assert np.array_equal(out[...,OLD],np.asarray(coarse).reshape(n,3,64,9))
    return out.reshape(n,3,64,8,7)


def data(dep,split):
    x,meta=mass.data(e.PREVIOUS,dep,split)
    c=np.load(e.PREVIOUS/f'expanded_encoder/features/{dep}/{split}.npy',mmap_mode='r')
    d=np.load(e.PREVIOUS/f'mixture_density/features/{dep}/{split}_extra.npy',mmap_mode='r')
    z=grid(c,d)
    if split=='train':
        c=np.load(e.PREVIOUS/f'additional_population/expanded_encoder/features/{dep}/train.npy',mmap_mode='r')
        d=np.load(e.PREVIOUS/f'additional_population/features/{dep}/dense.npy',mmap_mode='r')
        z=np.concatenate([z,grid(c,d)],0)
    assert len(z)==len(x)==len(meta)
    return x,z,meta


def targets(t):
    choices=[]
    for v,c in zip(t.T,(mass.LOG_CENTERS,g.Q,g.CHI)):
        v=v.clip(c[0],c[-1]);hi=np.searchsorted(c,v,side='right').clip(1,len(c)-1);lo=hi-1
        f=(v-c[lo])/(c[hi]-c[lo]);choices.append(((lo,1-f),(hi,f)))
    ii,ww=[],[]
    for m,wm in choices[0]:
        for q,wq in choices[1]:
            for s,ws in choices[2]:
                ii.append((m*8+q)*7+s);ww.append(wm*wq*ws)
    ii,ww=np.stack(ii,1),np.stack(ww,1).astype(np.float32)
    assert np.allclose(ww.sum(1),1)
    return ii,ww


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=g.arch.Predictor('GLOBAL')
        for p in self.body.parameters():p.requires_grad_(False)
        axes=[torch.linspace(-1,1,n) for n in (64,8,7)]
        self.register_buffer('coordinates',torch.stack(torch.meshgrid(*axes,indexing='ij'))[None])
        self.first=nn.Sequential(nn.Conv3d(6,24,(5,3,3),padding=(2,1,1)),nn.GroupNorm(8,24),nn.SiLU())
        self.residual=nn.Sequential(nn.Conv3d(24,24,3,padding=1),nn.GroupNorm(8,24),nn.SiLU(),nn.Dropout3d(.05),
                                    nn.Conv3d(24,24,(5,3,3),padding=(4,1,1),dilation=(2,1,1)),nn.GroupNorm(8,24))
        self.last=nn.Conv3d(24,1,1)

    def forward(self,x,mass_temperature=1.,conditional_temperature=1.):
        xm=x[:,:N_M].reshape(-1,27,253)
        xd=x[:,N_M:].reshape(-1,3,64,8,7)
        m=F.log_softmax(self.body(xm).float()/mass_temperature,1)
        h=self.first(torch.cat([xd,self.coordinates.expand(len(x),-1,-1,-1,-1)],1))
        h=F.silu(h+self.residual(h))
        c=F.interpolate(self.last(h).float(),size=(253,8,7),mode='trilinear',align_corners=True).squeeze(1).flatten(2)
        c=F.log_softmax(c/conditional_temperature,2)
        return (m[:,:,None]+c).flatten(1)


def pack(x,z,ck):
    return np.concatenate([((x-ck['muM'])/ck['sdM']).reshape(len(x),-1),
                           ((z-ck['muD'])/ck['sdD']).reshape(len(x),-1)],1).astype(np.float32)


def initialize(root):
    e.initialize(root)
    comp=g.GLOBAL/'trials/MIXTURE-PRIOR/diagnostic_export'
    shutil.copytree(comp/'results/DIAGNOSTIC/gwtc4',root/'comparators/O4a_GLOBAL_PRIOR')
    dev.csv_write(root/'manifest/GLOBAL_PRIOR_PROTECTED.csv',pd.DataFrame([
        {'path':str(p),'sha256':dev.sha(p)} for parent in (comp,g.GLOBAL/'models') for p in sorted(parent.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts]))
    contract=json.loads((root.parent/'contracts/INTRINSIC_GRID_CONTRACT.json').read_text())
    contract.update({'utc':datetime.now(timezone.utc).isoformat(),'kind':'DENSE3D_CONDITIONAL','seeds':SEEDS,
        'representation':'frozen GLOBAL p(logMc) times learned p(q,chi|Mc,dense 3D waveform grid),253x8x7 discrete mass',
        'features':'existing same2s template responses:3detector/network x64mass x8q x7spin;plus original27x253 fine-Mc features',
        'new_network':'6(coordinates+responses)->24Conv3d/GN/SiLU,residual24Conv3d,dilatedmassConv3d->1;linear interpolation only of conditional logits along mass coordinate64to253',
        'not_new_physical_template_information':'all 3584 dense templates and1701 refinedMc templates were already calculated;this is a new structured readout,not new strain/PE',
        'frozen_neural_component':'parent GLOBAL waveform predictor and its complete mass marginal',
        'training':'40epochs AdamW2e-4 WD1e-4 cosine_min2e-5,128source x2views;train dense conditional head only',
        'q_grid':g.Q.tolist(),'chi_grid':g.CHI.tolist(),'edge_handling':'trilinear targets clipped to native grid;report boundary occupancy and out-of-grid truth;not calibrated PE beyond support',
        'why_added':'Prior reviewed implementations flattened dense q/spin responses. Preserve their physical neighborhood geometry as an explicit new control.'})
    dev.json_write(root/'contracts/INTRINSIC_GRID_CONTRACT.json',contract)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)


def train(root,dep,ms,seed):
    out=root/f'models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    x,z,tm=data(dep,'train');v,vz,vm=data(dep,'validation')
    assert not set(tm.source_uid)&set(vm.source_uid) and not set(tm.noise_bank_index)&set(vm.noise_bank_index)
    parent=torch.load(g.GLOBAL/f'models/{dep}/seed_{ms}/validation_selected_model.pt',weights_only=False,map_location='cpu')
    ck={'muM':parent['mu'],'sdM':parent['sd'],'muD':z.mean((0,2,3,4),keepdims=True),
        'sdD':z.std((0,2,3,4),keepdims=True).clip(.01),'mass_temperature':parent['temperature'],
        'kind':'DENSE3D_CONDITIONAL','seed':seed}
    x=pack(x,z,ck);v=pack(v,vz,ck);del z,vz
    tt,vt=g.truth(tm),g.truth(vm);ti,tw=targets(tt);vi,vw=targets(vt)
    group,names=pd.factorize(tm.source_uid,sort=True);vg,_=pd.factorize(vm.source_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(names))]
    dev.TRAIN.seed_everything(seed);model=Predictor().cuda();model.body.load_state_dict(parent['model'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40,eta_min=2e-5)
    xx=torch.as_tensor(x,device='cuda');ii=torch.as_tensor(ti,device='cuda');ww=torch.as_tensor(tw,device='cuda')
    history=[];best=float('inf');mt=ck['mass_temperature']
    for epoch in range(1,41):
        model.train();model.body.eval();rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));losses=[]
        for start in range(0,len(order),128):
            ix=np.stack([rng.choice(members[k],2,replace=False) for k in order[start:start+128]]).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lp=model(xx[ix],mt);loss=-(lp.gather(1,ii[ix])*ww[ix]).sum(1).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        scheduler.step();lp=g.infer(model,v,mt);nll=g.nll(lp,vi,vw)
        row={'epoch':epoch,'train_NLL':float(np.mean(losses)),'validation_NLL':nll,'seconds':time.perf_counter()-started};history.append(row)
        if nll<best:
            best=nll
            torch.save({**ck,'model':{k:p.detach().cpu().clone() for k,p in model.state_dict().items()},'epoch':epoch},out/'validation_selected_model.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch%10==0:print(json.dumps({'train':['DENSE3D',dep,seed],**row}),flush=True)
    ck=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['model'])
    temps=[{'temperature':t,'NLL':g.nll(g.infer(model,v,mt,t),vi,vw)} for t in (.5,.75,1.,1.25,1.5,2.,3.)]
    ct=min(temps,key=lambda r:(r['NLL'],abs(r['temperature']-1)))['temperature'];ck['conditional_temperature']=ct
    p=np.exp(g.infer(model,v,mt,ct)).astype(np.float32);p/=p.sum(1,keepdims=True)
    unique=~tm.source_uid.duplicated().to_numpy()
    prior=np.bincount(ti[unique].reshape(-1),weights=tw[unique].reshape(-1),minlength=np.prod(g.SHAPE)).reshape(g.SHAPE)
    prior=gaussian_filter(prior.astype(float),1,mode='nearest')+1e-5;prior/=prior.sum();ck['prior']=prior.astype(np.float32)
    torch.save(ck,out/'validation_selected_model.pt');np.savez_compressed(out/'development_predictions.npz',p=p,group=vg,truth=vt)
    marginal=p.reshape(-1,*g.SHAPE)
    errors=[abs(marginal.sum(axes)@centers-vt[:,k]).mean()
            for k,axes,centers in ((0,(2,3),mass.LOG_CENTERS),(1,(1,3),g.Q),(2,(1,2),g.CHI))]
    report={'kind':'DENSE3D_CONDITIONAL','deployment':dep,'seed':seed,'epoch':ck['epoch'],'conditional_temperature':ct,
            'development_joint_NLL':g.nll(np.log(p.clip(1e-30)),vi,vw),'logMc_MAE':float(errors[0]),
            'q_MAE':float(errors[1]),'chi_eff_MAE':float(errors[2]),'source_overlap':0,'noise_overlap':0,
            'training_sources':len(names),'development_sources':len(np.unique(vg)),
            'seconds':time.perf_counter()-started,'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt')}
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(temps));dev.json_write(out/'COMPLETE.json',report)
    print(json.dumps({'training_complete':report}),flush=True)


def prediction(root,dep,ms,es,split):
    cp=root/f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    if split=='development':return dict(np.load(cp.parent/'development_predictions.npz'))
    cache=root/f'cache/predictions/{dep}/{ms}_{es}_{split}.npz'
    if cache.exists():return dict(np.load(cache))
    name='real' if split=='real' else f'{es}_{split}'
    coarse=np.load(e.old.TRAINED/f'cache/deployment_event_psd/{dep}'/('real_features.npy' if split=='real' else f'{name}_features.npy'))
    fine=np.load(e.PREVIOUS/f'fine_mass_context/features/{dep}/{name}.npy')
    d=np.load(e.PREVIOUS/f'dense_context/features/{dep}/{name}.npy')
    x=mass.arrange(coarse,fine);z=grid(coarse,d)
    ck=torch.load(cp,weights_only=False,map_location='cpu');model=Predictor().cuda().eval();model.load_state_dict(ck['model'])
    p=np.exp(g.infer(model,pack(x,z,ck),ck['mass_temperature'],ck['conditional_temperature'])).astype(np.float32);p/=p.sum(1,keepdims=True)
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp=np.full((len(full),p.shape[1]),np.nan,np.float32);pp[valid]=p;p=pp
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,p=p);return {'p':p}


def tests():
    c=np.arange(2*3*64*9,dtype=np.float32).reshape(2,3,64,9)
    d=np.arange(2*3*64*47,dtype=np.float32).reshape(2,3,64,47)
    out=grid(c,d).reshape(2,3,64,56)
    assert np.array_equal(out[...,EXTRA],d)
    t=np.array([[np.log(20),.5,.1],[np.log(40),1.,-.5]])
    ii,w=targets(t);coords=np.array(np.meshgrid(mass.LOG_CENTERS,g.Q,g.CHI,indexing='ij')).reshape(3,-1).T
    assert np.allclose((coords[ii]*w[:,:,None]).sum(1),t,atol=1e-7)
    print('dense feature ordering and trilinear labels: PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('init','train','test','evaluate'),required=True)
    p.add_argument('--arm',choices=('JOINT-PRIOR','JOINT-BC','MASS-Q-PRIOR','MASS-PRIOR'),default='JOINT-PRIOR');a=p.parse_args()
    if a.phase=='init':initialize(a.root)
    elif a.phase=='test':tests()
    elif a.phase=='train':
        for dep in e.old.DEPS:
            for ms,seed in zip(e.body.MODEL_SEEDS,SEEDS):train(a.root,dep,ms,seed)
    else:
        g.prediction=prediction;g.run(a.root,a.arm)
