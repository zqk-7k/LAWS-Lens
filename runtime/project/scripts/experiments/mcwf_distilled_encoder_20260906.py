#!/usr/bin/env python3
"""Train one shared source/mass/geometry-distillation recipe in O3/O4a."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_unified_waveform_20260906 as u

PREV=u.PREV
EPOCHS=50
DISTILL_WEIGHT=10.


def initialize(root,mass_local=False,hard_negatives=False):
    path=root/'contracts/DISTILLATION_TRAINING.json'
    if path.exists():
        if bool(json.loads(path.read_text()).get('mass_local',False))!=mass_local:
            raise RuntimeError('Training recipe differs from frozen contract')
        if bool(json.loads(path.read_text()).get('hard_negatives',False))!=hard_negatives:
            raise RuntimeError('Hard-negative recipe differs from frozen contract')
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Refuse overwrite')
    for d in ('contracts','scripts','models','cache','tables','audit','logs','reports','figures','manifest','results','evaluation','calibration'):
        (root/d).mkdir(parents=True,exist_ok=True)
    dev.json_write(path,{
        'code':'MCWF-UNIFIED-v6-HARDNEG' if hard_negatives else ('MCWF-UNIFIED-v5-MASSLOCAL' if mass_local else 'MCWF-UNIFIED-v4-DISTILLED'),'utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Previous shared scoring ablations could not retain O3/O4a injection guardrails. Train the new representation to preserve the old encoder geometry on simulation data while learning physical source/mass information.',
        'shared_architecture':'RAW-PHASE-SOURCE InceptionAttention+phasebank; exactly the same in both runs',
        'input':'unchanged H1L1 peak2s4096 at2048Hz; physical filtering/preprocessing unchanged',
        'data':'same source-disjoint simulation development arrays as previous six RAW models; no public PE or event identity inputs',
        'training_noise_limitation':'legacy O3 development includes O1/O2; this control isolates loss/training, not a claim of strictly O3-only training',
        'teacher':'frozen baseline encoder on simulation training/validation waveforms only; no real teacher embeddings for loss',
        'loss':'soft-label simulated logMc CE + source SupCon +10*mean_offdiagonal((z_student.z_studentT)-(z_teacher.z_teacherT))^2',
        'distillation_interpretation':'preserve similarity geometry, not reproduce teacher mass mistakes as labels; teacher is not a physical ground truth',
        'mass_local':mass_local,
        'mass_local_definition':'If enabled, geometry weights exp(-0.5*(delta_true_logMc/0.25)^2), off-diagonal normalized. Teacher geometry is trusted locally in known simulation mass, not between very different masses.',
        'mass_local_scale_log':.25,
        'hard_negatives':hard_negatives,
        'hard_negative_sampling':'If enabled: 32 uniform anchors plus 32 unique teacher-nearest source groups having abs(delta_true_logMc)>=0.4. Randomly draw from each anchor top16; deterministic tie order. All are training groups; no real pairs or PE labels. Remaining batch entries filled from the shuffled source order.',
        'warm_start':'corresponding previous RAW-PHASE-SOURCE checkpoint; all representation parameters trained',
        'epochs':EPOCHS,'lr':5e-5,'weight_decay':1e-4,'lr_min':5e-6,'batch_sources':64,'source_passes_per_epoch':4,
        'checkpoint':'minimum validation CE+0.2SupCon+10*geometry_MSE; earliest tie',
        'temperature':'same validation CE-only grid as previous model',
        'seeds':body.MODEL_SEEDS,'teacher_seeds':dev.SEEDS,'both_runs_required':True,
        'real_PE_role':'later explicit development audit, not blind confirmation',
        'frozen':['time','sky','outer C-fixed weights','scope','historical files'],
        'references':['https://arxiv.org/abs/1503.02531','https://arxiv.org/abs/2004.11362'],
        'reference_limit':'this pairwise-geometry loss is our declared engineering distillation control, not the exact objective of Hinton et al.',
    })
    shutil.copy2(PREV/'manifest/PROTECTED_INPUT_SHA256.csv',root/'manifest/PROTECTED_INPUT_SHA256.csv')
    for p in (PREV/'audit').glob('*pe_official*.parquet'):
        shutil.copy2(p,root/'audit'/p.name)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)


def geometry_loss(z,t,logmass=None):
    z=F.normalize(z.float(),dim=-1)
    t=F.normalize(t.float(),dim=-1)
    difference=(z@z.T-t@t.T).square()
    if logmass is not None:
        weight=torch.exp(-.5*((logmass[:,None]-logmass[None,:])/.25).square())
        weight=weight.masked_fill(torch.eye(len(z),dtype=torch.bool,device=z.device),0.)
        return (difference*weight).sum()/weight.sum().clamp_min(1e-6)
    return (difference.sum()-difference.diagonal().sum())/(len(z)*(len(z)-1))


def teacher_features(root,dep,es,split,raw):
    out=root/f'cache/teacher/{dep}/seed_{es}'
    out.mkdir(parents=True,exist_ok=True)
    path=out/f'{split}.npy'
    if path.exists():
        return np.load(path)
    checkpoint=dev.V7/dep/f'seed_{es}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt'
    model,_=dev.BASE.v7.load_unified_model(checkpoint)
    z,_=dev.BASE.v7.embed_catalog(model,dev.BASE.v7.ArrayCatalog(raw),batch_size=64)
    del model
    np.save(path,z)
    dev.json_write(out/f'{split}_PROVENANCE.json',{'teacher_checkpoint':str(checkpoint),'teacher_sha256':dev.sha(checkpoint),'rows':len(z),'real_data':False,'shape':list(z.shape)})
    return z


def train(root,dep,seed,mass_local=False,hard_negatives=False):
    initialize(root,mass_local=mass_local,hard_negatives=hard_negatives)
    out=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}'
    if (out/'COMPLETE.json').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    cache=PREV/f'cache/{dep}'
    raw=np.load(cache/'train_raw2s.npy',mmap_mode='r')
    vraw=np.load(cache/'validation_raw2s.npy',mmap_mode='r')
    meta=pd.read_parquet(cache/'train_metadata.parquet')
    vmeta=pd.read_parquet(cache/'validation_metadata.parquet')
    if set(meta.waveform_parent_uid)&set(vmeta.waveform_parent_uid):
        raise RuntimeError('Source leakage')
    groups,labels=pd.factorize(meta.waveform_parent_uid,sort=True)
    vg,_=pd.factorize(vmeta.waveform_parent_uid,sort=True)
    true_logmass=np.log(meta.chirp_mass_detector.to_numpy(float))
    val_logmass=np.log(vmeta.chirp_mass_detector.to_numpy(float))
    members=[np.flatnonzero(groups==i) for i in range(len(labels))]
    y=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy')
    vy=np.load(body.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    warm=PREV/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/validation_selected_model.pt'
    payload=torch.load(warm,map_location='cpu',weights_only=False)
    froot=body.PREVIOUS/f'cache/phasebank/{dep}'
    x=(np.load(froot/'train_features.npy')-payload['mu'])/payload['sd']
    vx=(np.load(froot/'validation_features.npy')-payload['mu'])/payload['sd']
    es=dev.SEEDS[body.MODEL_SEEDS.index(seed)]
    teacher=teacher_features(root,dep,es,'train',raw)
    vteacher=teacher_features(root,dep,es,'validation',vraw)
    if len(teacher)!=len(x) or len(vteacher)!=len(vx):
        raise RuntimeError('Teacher row order mismatch')
    hard=[]
    if hard_negatives:
        center=np.stack([teacher[idx].mean(0) for idx in members]).astype(float)
        center/=np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-12)
        group_mass=np.array([true_logmass[idx[0]] for idx in members])
        similarity=center@center.T
        similarity[np.abs(group_mass[:,None]-group_mass[None,:])<.4]=-np.inf
        hard=[np.argsort(-row,kind='stable')[:16] for row in similarity]
        if any(not np.isfinite(similarity[k,ids]).all() for k,ids in enumerate(hard)):
            raise RuntimeError('Not enough mass-separated training negatives')
        np.save(out/'training_hard_neighbor_ids.npy',np.asarray(hard))
    dev.TRAIN.seed_everything(seed)
    model=body.Encoder('RAW-PHASE-SOURCE').cuda()
    model.load_state_dict(payload['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=5e-6)
    history=[]
    best=float('inf')
    first=1
    resume=out/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        history,best,first=state['history'],state['best'],state['epoch']+1
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first,EPOCHS+1):
        started=time.perf_counter()
        model.train()
        rng=np.random.default_rng(seed+100000+epoch)
        losses=[]
        sampled=np.zeros(len(labels),dtype=int)
        for _ in range(4):
            order=rng.permutation(len(labels))
            for k in range(0,len(order),64):
                parents=order[k:k+64]
                if hard_negatives:
                    anchors=parents[:len(parents)//2].tolist()
                    picked=anchors.copy()
                    for anchor in anchors:
                        candidates=[int(v) for v in hard[anchor] if int(v) not in picked]
                        if candidates:
                            picked.append(int(rng.choice(candidates)))
                    for value in np.r_[parents,order]:
                        if len(picked)>=len(parents):
                            break
                        if int(value) not in picked:
                            picked.append(int(value))
                    parents=np.asarray(picked,dtype=int)
                sampled[parents]+=1
                idx=np.stack([rng.choice(members[j],2,replace=False) for j in parents]).T.reshape(-1)
                g=torch.as_tensor(np.tile(parents,2),device='cuda')
                a=torch.as_tensor(x[idx],dtype=torch.float32,device='cuda')
                b=torch.as_tensor(np.asarray(raw[idx]),dtype=torch.float32,device='cuda')
                target=torch.as_tensor(y[idx],device='cuda')
                t=torch.as_tensor(teacher[idx],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,z=model(a,b)
                    ce=-(F.log_softmax(logits.float(),-1)*target).sum(-1).mean()
                    sc=body.source_contrastive(z,g)
                    kd=geometry_loss(z,t,torch.as_tensor(true_logmass[idx],dtype=torch.float32,device='cuda') if mass_local else None)
                    loss=ce+sc+DISTILL_WEIGHT*kd
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite distillation loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5)
                optimizer.step()
                losses.append([ce.item(),sc.item(),kd.item()])
        scheduler.step()
        logits,z=body.infer(model,vx,vraw)
        lp=logits.astype(float)-np.logaddexp.reduce(logits.astype(float),axis=-1,keepdims=True)
        ce=float(-(vy*lp).sum(-1).mean())
        sc=float(body.source_contrastive(torch.as_tensor(z,device='cuda'),torch.as_tensor(vg,device='cuda')))
        kd=float(geometry_loss(torch.as_tensor(z,device='cuda'),torch.as_tensor(vteacher,device='cuda'),torch.as_tensor(val_logmass,dtype=torch.float32,device='cuda') if mass_local else None))
        criterion=ce+.2*sc+DISTILL_WEIGHT*kd
        row={'epoch':epoch,'train_ce':np.mean(losses,axis=0)[0],'train_supcon':np.mean(losses,axis=0)[1],'train_distillation':np.mean(losses,axis=0)[2],
            'validation_ce':ce,'validation_supcon':sc,'validation_distillation':kd,'criterion':criterion,'seconds':time.perf_counter()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'unique_sampled_sources':int((sampled>0).sum()),'max_source_batch_count':int(sampled.max())}
        history.append(row)
        if criterion<best:
            best=criterion
            selected={**payload,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,'criterion':criterion,
                'experiment':'MCWF-UNIFIED-v6-HARDNEG' if hard_negatives else ('MCWF-UNIFIED-v5-MASSLOCAL' if mass_local else 'MCWF-UNIFIED-v4-DISTILLED'),'hard_negatives':hard_negatives,'mass_local':mass_local,'distillation_weight':DISTILL_WEIGHT,'warm_start':str(warm),'warm_sha256':dev.sha(warm),
                'encoder_body_trained':True,'training_contract_sha256':dev.sha(root/'contracts/DISTILLATION_TRAINING.json')}
            torch.save(selected,out/'validation_selected_model.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'epoch':epoch,'history':history,'best':best},resume)
        dev.csv_write(out/'training_history.csv',pd.DataFrame(history))
        print(json.dumps({'deployment':dep,'seed':seed,**row}),flush=True)
    selected=torch.load(out/'validation_selected_model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'])
    logits,z=body.infer(model,vx,vraw)
    grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        lp=logits.astype(float)/t
        lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True)
        grid.append({'temperature':t,'ce':float(-(vy*lp).sum(-1).mean())})
    selected['temperature']=min(grid,key=lambda r:(r['ce'],abs(r['temperature']-1)))['temperature']
    torch.save(selected,out/'validation_selected_model.pt')
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    np.savez_compressed(out/'development_validation.npz',logits=logits,embedding=z,truth=vy,group=vg)
    dev.json_write(out/'COMPLETE.json',{'epochs':EPOCHS,'selected_epoch':selected['epoch'],'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt'),
        'same_training_across_runs':True,'real_PE_training':False,'n_train_source':len(labels)})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',choices=u.DEPS,required=True)
    p.add_argument('--seed',type=int,choices=body.MODEL_SEEDS,required=True)
    p.add_argument('--mass-local',action='store_true')
    p.add_argument('--hard-negatives',action='store_true')
    a=p.parse_args()
    if a.hard_negatives and not a.mass_local:
        raise ValueError('This ablation is defined relative to mass-local distillation')
    train(a.root,a.deployment,a.seed,mass_local=a.mass_local,hard_negatives=a.hard_negatives)
