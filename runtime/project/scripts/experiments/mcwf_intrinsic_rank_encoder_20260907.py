#!/usr/bin/env python3
"""Raw-waveform source encoder with a conditional intrinsic-density head."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
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
import mcwf_expanded_encoder_20260907 as expanded
import mcwf_mixture_density_20260907 as mixture
from scipy.special import expit
import mcwf_mass_tf_20260905 as tf

dev,body=e.dev,e.body
torch.set_num_threads(2)
SEEDS=(202609221,202609222,202609223)
EPOCHS=50



def physical_labels(meta):
    q=meta.m2_det.to_numpy(float)/meta.m1_det.to_numpy(float)
    chi=(meta.a1.to_numpy()*np.cos(meta.tilt1)+q*meta.a2.to_numpy()*np.cos(meta.tilt2))/(1+q)
    return np.column_stack([np.log(meta.mc_det),q,chi])


def reference_scale(warm,ck):
    a=np.load(warm.parent/'validation_predictions.npz')
    rng=np.random.default_rng(202609220);variances=[]
    for k in range(len(a['w'])):
        w=np.asarray(a['w'][k],float);w/=w.sum()
        pick=rng.choice(len(w),size=512,p=w)
        draws=a['m'][k,pick]+np.einsum('nij,nj->ni',
            np.linalg.cholesky(a['cov'][k,pick]),rng.standard_normal((512,3)))
        draws=draws*ck['ys']+ck['ym']
        draws[:,1]=expit(draws[:,1]);draws[:,2]=np.tanh(draws[:,2])
        variances.append(np.var(draws,axis=0,ddof=1))
    return np.sqrt(np.mean(variances,axis=0)).clip(1e-4)


def intrinsic_rank_loss(z,label,temperature=.1):
    z=z.float();label=label.float()
    if label.ndim!=2 or label.shape!=(len(z),3):
        raise ValueError('Three scaled intrinsic target coordinates required')
    n=len(z)
    if n<2:return z.sum()*0
    keep=~torch.eye(n,dtype=torch.bool,device=z.device)
    d=torch.cdist(label,label,p=1)[keep].reshape(n,n-1)
    s=(-torch.cdist(z,z,p=2,compute_mode='donot_use_mm_for_euclid_dist')/temperature)[keep].reshape(n,n-1)
    order=torch.argsort(d,dim=-1,descending=True,stable=True)
    sd,ss=torch.gather(d,1,order),torch.gather(s,1,order)
    right=torch.searchsorted((-sd).contiguous(),(-sd).contiguous(),right=True)-1
    return (torch.gather(torch.logcumsumexp(ss,-1),1,right)-ss).mean()


def rank_tests():
    torch.manual_seed(202609220);z=torch.randn(8,4,requires_grad=True)
    lab=torch.tensor([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,0.,0.],
        [0.,1.,0.],[0.,1.,0.],[1.,1.,1.],[1.,1.,1.]])
    actual=intrinsic_rank_loss(z,lab);terms=[]
    for i in range(len(z)):
        d=(lab-lab[i]).abs().sum(-1)
        for j in range(len(z)):
            if i==j:continue
            keep=(d>=d[j]);keep[i]=False
            terms.append(torch.logsumexp(-(z[i]-z[keep]).norm(dim=-1)/.1,0)+(z[i]-z[j]).norm()/.1)
    reference=torch.stack(terms).mean()
    ga=torch.autograd.grad(actual,z,retain_graph=True)[0]
    gb=torch.autograd.grad(reference,z)[0]
    if not torch.allclose(actual,reference,atol=3e-6,rtol=1e-6) or not torch.allclose(ga,gb,atol=3e-6,rtol=1e-5):
        raise RuntimeError('Intrinsic RNC direct definition/gradient mismatch')
    return {'loss_error':float(abs(actual-reference)),'gradient_error':float((ga-gb).abs().max()),
        'ties_included':True,'multivariate_L1':True}


class JointSource(body.Encoder):
    def __init__(self):
        super().__init__('RAW-PHASE-SOURCE')
        self.density_head=nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,40))

    def forward(self,features,strain):
        h=self.merge(torch.cat([self.phase.layers(features),self.raw(strain)],dim=-1))
        logits=self.phase.head(h);z=F.normalize(self.embedding(h),dim=-1)
        a=self.density_head(h).float().reshape(-1,4,10)
        cov=torch.zeros((*a.shape[:2],3,3),dtype=torch.float32,device=a.device)
        rows,cols=torch.tril_indices(3,3,device=a.device)
        cov[...,rows,cols]=a[...,4:]
        ii=torch.arange(3,device=a.device);cov[...,ii,ii]=F.softplus(cov[...,ii,ii])+.03
        return logits,z,(a[...,0],a[...,1:4],cov)


@torch.no_grad()
def infer(model,x,raw,batch=128):
    model.eval();outputs={k:[] for k in ('logits','z','w','m','cov')}
    for start in range(0,len(x),batch):
        xx=torch.as_tensor(x[start:start+batch],dtype=torch.float32,device='cuda')
        rr=torch.as_tensor(np.asarray(raw[start:start+batch]),dtype=torch.float32,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,z,(w,m,c)=model(xx,rr)
        out={'logits':logits.float(),'z':z.float(),'w':F.softmax(w,-1),'m':m,'cov':c@c.transpose(-1,-2)}
        for k,v in out.items():outputs[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in outputs.items()}


def initialize(root):
    out=root/'intrinsic_rank_encoder';path=out/'contracts/TRAINING.json'
    if path.exists():return
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a':True,
        'input':'unchangedH1L1peak2s4096points40-580Hz;rawInceptionAttention+576templatephasefeatures',
        'data':'expanded4096source/512sourcevalidationdataset;historical/source/noiseisolationcontractretained',
        'warm_start':'corresponding joint_source_density checkpoint;oldC-fixedandbaselineFRTcheckpointsuntouched',
        'architecture':'unchanged joint_source_density network;only ordered target geometry is changed',
        'targets':['logMc','logitq','atanhchieff'],'normalization':'oldphasefeaturenormalizationfrozen;3DtargetmeanSDfromtrainingonly',
        'training':'50epochs jointtraining,including epoch0 as eligible validation checkpoint;128sources*2views/step,onepass/epoch;AdamW1e-4,WD1e-4,cosine_min1e-5,clip5',
        'loss':'soft-binMcCE+sourceSupCon+RNC(logMc,q,chieff;predictive-widthscaledL1)+0.5*jointmixtureNLL',
        'selection':'earliestvalidationminimum CE+0.2SupCon+0.2RNC+0.5NLL;noPEorofficiallabels',
        'seeds':SEEDS,'epochs':EPOCHS,'mixture_variance_floor':.03,'rank_label_scale':'RMS calibrated predictive SD in physical logMc,q,chieff from512source validation; fixed beforetraining;notpublicPE;L1 target distance',
        'uncertainty_calibration':'masssoftmaxtemperaturebyvalidationCE;globalcovariancemultiplierbyvalidationjointNLL',
        'not_PE':'learnedconditionalpredictivedensityunderthesimulator;notfullstrainPEorproperlensingBF',
        'references':['https://arxiv.org/abs/2210.01189','https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf'],
        'frozen':['time','sky','outerweights','strictscope','historicalresults'],'fresh_confirmation_required':True})
    (out/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,out/'scripts'/Path(__file__).name)


def nll(a,y,temperature=1.):
    w=torch.as_tensor(a['w'],dtype=torch.float64,device='cuda')
    m=torch.as_tensor(a['m'],dtype=torch.float64,device='cuda')
    c=torch.as_tensor(a['cov'],dtype=torch.float64,device='cuda')*temperature
    d=torch.distributions.MultivariateNormal(m,covariance_matrix=c)
    yy=torch.as_tensor(y,dtype=torch.float64,device='cuda')
    return float(-torch.logsumexp(w.clamp_min(1e-30).log()+d.log_prob(yy[:,None]),-1).mean())


def train(root,dep,ms,seed):
    initialize(root);out=root/f'intrinsic_rank_encoder/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    warm=root/f'joint_source_density/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(warm,weights_only=False,map_location='cpu')
    x,raw,y,tm,group,names=expanded.data(root,dep,'train',ck)
    vx,vr,vy,vm,vg,vnames=expanded.data(root,dep,'validation',ck)
    if set(names)&set(vnames) or set(tm.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Source/noiseoverlap')
    yy=mixture.targets(tm);vv=mixture.targets(vm);ym,ys=yy.mean(0),yy.std(0).clip(1e-5)
    yy,vv=(yy-ym)/ys,(vv-ym)/ys
    label_scale=reference_scale(warm,ck)
    labels,vlabels=physical_labels(tm)/label_scale,physical_labels(vm)/label_scale
    dev.json_write(out/'LABEL_SCALE.json',{'scale':label_scale.tolist(),'reference_sha256':dev.sha(warm.parent/'validation_predictions.npz'),'unit_tests':rank_tests()})
    truth=np.log(tm.mc_det.to_numpy(float));vt=np.log(vm.mc_det.to_numpy(float))
    members=[np.flatnonzero(group==g) for g in range(len(names))]
    dev.TRAIN.seed_everything(seed);model=JointSource().cuda()
    missing,unexpected=model.load_state_dict(ck['model'],strict=False)
    if unexpected or any(not k.startswith('density_head.') for k in missing):raise RuntimeError('Warmstartmismatch')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    history,best,first=[],np.inf,1;resume=out/'resume.pt';start=time.perf_counter()
    if not resume.exists():
        a=infer(model,vx,vr);p=softmax(a['logits'].astype(float),1)
        ce=float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())
        sc=float(body.source_contrastive(torch.as_tensor(a['z'],device='cuda'),torch.as_tensor(vg,device='cuda')))
        rank=float(intrinsic_rank_loss(torch.as_tensor(a['z'],device='cuda'),torch.as_tensor(vlabels,device='cuda')))
        joint=nll(a,vv);best=ce+.2*sc+.2*rank+.5*joint
        history.append({'epoch':0,'CE':ce,'SupCon':sc,'RNC':rank,'jointNLL':joint,'criterion':best,
            'logMc_MAE':float(abs(p@tf.LOG_CENTERS-vt).mean())})
        torch.save({**ck,'epoch':0,'seed':seed,'criterion':best,'label_scale':label_scale,
            'warm_sha256':dev.sha(warm)},out/'validation_selected_model.pt')
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu');model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler'])
        history,best,first=r['history'],r['best'],r['epoch']+1;torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        for param in model.parameters():param.requires_grad_(True)
        model.train()
        rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));losses=[]
        for begin in range(0,len(order),128):
            source=order[begin:begin+128]
            idx=np.stack([rng.choice(members[g],2,replace=False) for g in source]).T.reshape(-1)
            xx=torch.as_tensor(x[idx],dtype=torch.float32,device='cuda');rr=torch.as_tensor(np.asarray(raw[idx]),dtype=torch.float32,device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,z,params=model(xx,rr)
                ce=-(F.log_softmax(logits.float(),-1)*torch.as_tensor(y[idx],device='cuda')).sum(-1).mean()
                sc=body.source_contrastive(z,torch.as_tensor(np.tile(source,2),device='cuda'))
                rank=intrinsic_rank_loss(z,torch.as_tensor(labels[idx],device='cuda'))
            joint=-mixture.logprob(params,torch.as_tensor(yy[idx],dtype=torch.float32,device='cuda')).mean()
            loss=ce+sc+rank+.5*joint
            if not torch.isfinite(loss):raise RuntimeError('Nonfinitejointtraining')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        sched.step();a=infer(model,vx,vr);p=softmax(a['logits'].astype(float),1)
        ce=float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())
        sc=float(body.source_contrastive(torch.as_tensor(a['z'],device='cuda'),torch.as_tensor(vg,device='cuda')))
        rank=float(intrinsic_rank_loss(torch.as_tensor(a['z'],device='cuda'),torch.as_tensor(vlabels,device='cuda')))
        joint=nll(a,vv);criterion=ce+.2*sc+.2*rank+.5*joint
        row={'epoch':epoch,'training_loss':float(np.mean(losses)),'CE':ce,'SupCon':sc,'RNC':rank,'jointNLL':joint,
            'criterion':criterion,'logMc_MAE':float(abs(p@tf.LOG_CENTERS-vt).mean())};history.append(row)
        if criterion<best:
            best=criterion
            torch.save({**ck,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'ym':ym,'ys':ys,'label_scale':label_scale,'epoch':epoch,'seed':seed,'criterion':criterion,'warm_sha256':dev.sha(warm)},out/'validation_selected_model.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
            'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch%10==0:print(json.dumps({'intrinsic_rank_training':dep,'seed':seed,**row}),flush=True)
    ck=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['model'])
    a=infer(model,vx,vr);grid=[]
    for temperature in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        p=softmax(a['logits'].astype(float)/temperature,1)
        grid.append({'temperature':temperature,'CE':float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean()),'NLL':nll(a,vv,temperature)})
    ck['temperature']=min(grid,key=lambda r:(r['CE'],abs(r['temperature']-1)))['temperature']
    ck['covariance_temperature']=min(grid,key=lambda r:(r['NLL'],abs(r['temperature']-1)))['temperature']
    a['cov']*=ck['covariance_temperature'];a['p']=softmax(a['logits'].astype(float)/ck['temperature'],1)
    np.savez_compressed(out/'validation_predictions.npz',**a,group=vg,truth=vv)
    torch.save(ck,out/'validation_selected_model.pt');dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    report={'deployment':dep,'seed':seed,'epoch':ck['epoch'],'temperature':ck['temperature'],
        'covariance_temperature':ck['covariance_temperature'],'logMc_MAE':float(abs(a['p']@tf.LOG_CENTERS-vt).mean()),
        'jointNLL':nll(a,vv),'seconds':time.perf_counter()-start,'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt'),
        'raw_encoder_retrained':ck['epoch']>0}
    dev.json_write(out/'COMPLETE.json',report);print(json.dumps({'intrinsic_rank_done':report}),flush=True)


def prediction(root,dep,ms,es,split):
    cp=root/f'intrinsic_rank_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent/'COMPLETE.json').exists():raise RuntimeError('Trainingincomplete')
    path=root/f'intrinsic_rank_encoder/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if str(a['checkpoint_sha256'])!=dev.sha(cp):raise RuntimeError('Changedcheckpoint')
        return a
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        raw=dev.TRAIN.make_window_view(np.asarray(full[valid],np.float32),2)
        x=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/real_features.npy')
    else:
        plan=dev.BASE.retained_event_plan(dep,es,split);full=dev.ORCH.event_array_for_plan(dep,es,split,plan)
        raw=dev.TRAIN.make_window_view(np.asarray(full,np.float32),2)
        x=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy')
    model=JointSource().cuda().eval();model.load_state_dict(ck['model']);a=infer(model,(x-ck['mu'])/ck['sd'],raw)
    a['cov']*=ck['covariance_temperature'];a['p']=softmax(a['logits'].astype(float)/ck['temperature'],1)
    if split=='real':
        old=a;a={k:np.full((len(full),*v.shape[1:]),np.nan) for k,v in old.items()}
        for k,v in old.items():a[k][valid]=v
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a,checkpoint_sha256=dev.sha(cp));return np.load(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,SEEDS):train(a.root,dep,ms,seed)
