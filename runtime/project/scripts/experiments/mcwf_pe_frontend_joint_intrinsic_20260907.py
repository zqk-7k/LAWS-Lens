#!/usr/bin/env python3
"""Joint simulated Mc/q representation training, identical in O3 and O4a."""
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
import mcwf_pe_frontend_qprobe_20260907 as probe
import mcwf_mass_tf_20260905 as tf

dev, body = e.dev, e.body
torch.set_num_threads(2)
EPOCHS = 50


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = body.Encoder('RAW-PHASE-SOURCE')
        self.q_head = probe.head()

    def forward(self, features, strain):
        phase = self.encoder.phase.layers(features)
        raw = self.encoder.raw(strain)
        hidden = self.encoder.merge(torch.cat([phase, raw], dim=1))
        mass = self.encoder.phase.head(hidden)
        q = self.q_head(torch.cat([phase, raw, hidden], dim=1))
        return mass, q, F.normalize(self.encoder.embedding(hidden), dim=1)


def rank_loss(z, target, temperature=.1):
    z, target = z.float(), target.float()
    n = len(z)
    keep = ~torch.eye(n, dtype=torch.bool, device=z.device)
    d = torch.cdist(target, target, p=1)[keep].reshape(n, n-1)
    s = (-torch.cdist(z, z, p=2, compute_mode='donot_use_mm_for_euclid_dist')/temperature)[keep].reshape(n, n-1)
    order = torch.argsort(d, dim=1, descending=True, stable=True)
    dd, ss = torch.gather(d, 1, order), torch.gather(s, 1, order)
    right = torch.searchsorted((-dd).contiguous(), (-dd).contiguous(), right=True)-1
    return (torch.gather(torch.logcumsumexp(ss, dim=1), 1, right)-ss).mean()


def tests():
    torch.manual_seed(202609075)
    z = torch.randn(6, 4, requires_grad=True)
    t = torch.tensor([[0.,0.], [0.,0.], [.5,.3], [.5,.3], [1.,.2], [1.,.2]])
    calc = rank_loss(z,t)
    terms=[]
    for i in range(len(z)):
        for j in range(len(z)):
            if i == j: continue
            keep=[k for k in range(len(z)) if k != i and (t[i]-t[k]).abs().sum() >= (t[i]-t[j]).abs().sum()]
            terms.append(torch.logsumexp(-(z[i]-z[keep]).norm(dim=1)/.1,0)+(z[i]-z[j]).norm()/.1)
    direct = torch.stack(terms).mean()
    assert torch.allclose(calc,direct,atol=3e-6)
    g1=torch.autograd.grad(calc,z,retain_graph=True)[0];g2=torch.autograd.grad(direct,z)[0]
    assert torch.allclose(g1,g2,atol=3e-6)
    return {'pass':True,'loss_error':float(abs(calc-direct).detach()),'gradient_error':float((g1-g2).abs().max())}


def initialize(root):
    out=root/'joint_intrinsic'
    path=out/'contracts/TRAINING.json'
    if path.exists(): return
    out.mkdir(parents=True,exist_ok=True)
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'mechanism':'jointly refine complete waveform encoder for logMc and mass ratio q, not merely a frozen-feature q probe',
        'input':'same peak2s4096 H1L1 and same event-PSD phase features',
        'objective':'CE_Mc+CE_q+SupCon(source)+RNC((logMc/train_SD,0.5*q/train_SD))',
        'rank_target':'L1 distance in standardized 2D simulated intrinsic parameters, exact tie handling',
        'q_target':'20-bin Gaussian soft labels sigma0.06, reference only simulated m2/m1',
        'selection':'earliest minimum validation CE_Mc+CE_q+0.2*SupCon+0.2*RNC',
        'optimizer':{'name':'AdamW','lr':.0001,'weight_decay':.0001,'epochs':50,'source_batch':64,'views':2,'passes':4,'clip':5},
        'seeds':[202609081,202609082,202609083],'warm':'current frozen RNC models and separate q-probe initialization',
        'same_for_O3_O4a':True,'no_real_PE_labels':True,'no_official_labels':True,
        'time_sky_outer_weights_unchanged':True,'unit_tests':tests(),
        'reference':'https://arxiv.org/abs/2210.01189',
        'scientific_limit':'supervised simulation predictive distributions, not real PE or physical Bayes factors; all tolerances are declared development choices'})
    (out/'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__,out/'scripts'/Path(__file__).name)


@torch.no_grad()
def infer(model,x,raw):
    model.eval();ml,ql,zs=[],[],[]
    for start in range(0,len(x),128):
        xx=torch.as_tensor(x[start:start+128],dtype=torch.float32,device='cuda')
        rr=torch.as_tensor(np.array(raw[start:start+128],dtype=np.float32,copy=True),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            m,q,z=model(xx,rr)
        ml.append(m.float().cpu().numpy());ql.append(q.float().cpu().numpy());zs.append(z.float().cpu().numpy())
    return np.concatenate(ml),np.concatenate(ql),np.concatenate(zs)


def train(root,dep,ms,seed):
    initialize(root)
    out=root/f'joint_intrinsic/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists(): return
    out.mkdir(parents=True,exist_ok=True)
    cp=e.TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    qp=torch.load(root/f'qprobe/models/{dep}/seed_{ms}/validation_selected_probe.pt',weights_only=False,map_location='cpu')
    # Absorb probe normalization into its first affine layer before joint updates.
    qs=qp['model'].copy()
    w=qs['0.weight'].clone();b=qs['0.bias'].clone()
    sd=torch.as_tensor(qp['sd']);mu=torch.as_tensor(qp['mu'])
    qs['0.weight']=w/sd[None,:];qs['0.bias']=b-(w*mu[None,:]/sd[None,:]).sum(1)
    model=Encoder().cuda()
    model.encoder.load_state_dict(ck['model']);model.q_head.load_state_dict(qs)
    dev.TRAIN.seed_everything(seed)
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/train_metadata.parquet')
    vm=pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet')
    tq,tyq=probe.targets(meta);vq,vyq=probe.targets(vm)
    tmc=np.log(meta.chirp_mass_detector.to_numpy(float));vmc=np.log(vm.chirp_mass_detector.to_numpy(float))
    scales=np.array([tmc.std(),tq.std()]);weights=np.array([1.,.5])
    target=np.column_stack([tmc,tq])/scales*weights
    vtarget=np.column_stack([vmc,vq])/scales*weights
    x=(np.load(e.TRAINED/f'cache/finelag/{dep}/train_features.npy')-ck['mu'])/ck['sd']
    vx=(np.load(e.TRAINED/f'cache/finelag/{dep}/validation_features.npy')-ck['mu'])/ck['sd']
    raw=np.load(e.TRAINED/f'cache/{dep}/train_raw2s.npy',mmap_mode='r')
    vr=np.load(e.TRAINED/f'cache/{dep}/validation_raw2s.npy',mmap_mode='r')
    ym=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy')
    vym=np.load(body.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    group,names=pd.factorize(meta.waveform_parent_uid,sort=True)
    vg,_=pd.factorize(vm.waveform_parent_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(names))]
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    history=[];best=float('inf');start=time.perf_counter();start_epoch=1
    resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu')
        model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler'])
        history,best,start_epoch=r['history'],r['best'],r['epoch']+1
        torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(start_epoch,EPOCHS+1):
        tick=time.perf_counter();rng=np.random.default_rng(seed+epoch);model.train();losses=[]
        for repeat in range(4):
            order=rng.permutation(len(names))
            for first in range(0,len(order),64):
                sources=order[first:first+64]
                ids=np.stack([rng.choice(members[k],2,replace=False) for k in sources]).T.reshape(-1)
                xx=torch.as_tensor(x[ids],dtype=torch.float32,device='cuda')
                rr=torch.as_tensor(np.asarray(raw[ids]),dtype=torch.float32,device='cuda')
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    m,q,z=model(xx,rr)
                    cm=-(F.log_softmax(m.float(),-1)*torch.as_tensor(ym[ids],device='cuda')).sum(-1).mean()
                    cq=-(F.log_softmax(q.float(),-1)*torch.as_tensor(tyq[ids],device='cuda')).sum(-1).mean()
                    sc=body.source_contrastive(z,torch.as_tensor(np.tile(sources,2),device='cuda'))
                    rc=rank_loss(z,torch.as_tensor(target[ids],device='cuda'))
                    loss=cm+cq+sc+rc
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite joint loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        sched.step()
        m,q,z=infer(model,vx,vr)
        pm,pq=softmax(m.astype(float),1),softmax(q.astype(float),1)
        cm=float(-(vym*np.log(pm.clip(1e-30))).sum(-1).mean());cq=float(-(vyq*np.log(pq.clip(1e-30))).sum(-1).mean())
        sc=float(body.source_contrastive(torch.as_tensor(z,device='cuda'),torch.as_tensor(vg,device='cuda')))
        rc=float(rank_loss(torch.as_tensor(z,device='cuda'),torch.as_tensor(vtarget,device='cuda')))
        criterion=cm+cq+.2*sc+.2*rc
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'CE_Mc':cm,'CE_q':cq,'SupCon':sc,'RNC2D':rc,'criterion':criterion,
             'Mc_MAE_log':float(np.abs(pm@tf.LOG_CENTERS-vmc).mean()),'q_MAE':float(np.abs(pq@probe.QGRID-vq).mean()),'seconds':time.perf_counter()-tick}
        history.append(row)
        if criterion<best:
            best=criterion
            torch.save({**ck,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,
                        'new_seed':seed,'criterion':criterion,'warm_sha256':dev.sha(cp),'q_temperature':1.,'temperature':1.,'intrinsic_scales':scales,
                        'q_prior':qp['q_prior'],'feature_implementation':'joint_Mc_q_event_psd_v1'},out/'validation_selected_model.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,
                    'epoch':epoch,'best':best,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch%10==0: print(json.dumps({'training':dep,'seed':seed,**row}),flush=True)
    selected=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu')
    model.load_state_dict(selected['model']);m,q,z=infer(model,vx,vr)
    grids=[]
    for label,logits,y in [('Mc',m,vym),('q',q,vyq)]:
        rows=[]
        for t in (.5,.75,1.,1.25,1.5,2.,3.,4.):
            p=softmax(logits.astype(float)/t,1)
            rows.append({'label':label,'temperature':t,'CE':float(-(y*np.log(p.clip(1e-30))).sum(-1).mean())})
        pick=min(rows,key=lambda r:(r['CE'],abs(r['temperature']-1)))
        selected['temperature' if label=='Mc' else 'q_temperature']=pick['temperature'];grids.extend(rows)
    torch.save(selected,out/'validation_selected_model.pt')
    pm=softmax(m/selected['temperature'],1);pq=softmax(q/selected['q_temperature'],1)
    np.savez_compressed(out/'validation_predictions.npz',p=pm,q=pq,z=z,truth_m=vym,truth_q=vyq,group=vg)
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grids))
    result={'dep':dep,'seed':seed,'epoch':selected['epoch'],'seconds':time.perf_counter()-start,
            'q_MAE':float(np.abs(pq@probe.QGRID-vq).mean()),'constant_q_MAE':float(np.abs(qp['q_prior']@probe.QGRID-vq).mean()),
            'logMc_MAE':float(np.abs(pm@tf.LOG_CENTERS-vmc).mean()),'sha256':dev.sha(out/'validation_selected_model.pt')}
    dev.json_write(out/'COMPLETE.json',result);print(json.dumps({'trained_joint':result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,(202609081,202609082,202609083)):
            train(a.root,dep,ms,seed)
