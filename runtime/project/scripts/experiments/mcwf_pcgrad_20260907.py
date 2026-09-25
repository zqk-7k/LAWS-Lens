#!/usr/bin/env python3
"""Two-task PCGrad audit and initialization; no catalog or PE selection."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_finelag_encoder_20260907 as trainer

WARM=dev.PROJECT/'results/mcwf_unified_finelag_eventpsd_encoder_20260907'


def gradients(losses,parameters):
    parameters=list(parameters)
    a=torch.autograd.grad(losses[0],parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(losses[1],parameters,allow_unused=True)
    a=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,a)]
    b=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,b)]
    dot=sum((x.float()*y.float()).sum() for x,y in zip(a,b))
    na=sum(x.float().square().sum() for x in a)
    nb=sum(x.float().square().sum() for x in b)
    if not torch.isfinite(dot+na+nb):
        raise RuntimeError('Nonfinite task gradient')
    ca=torch.minimum(dot,torch.zeros_like(dot))/nb.clamp_min(1e-20)
    cb=torch.minimum(dot,torch.zeros_like(dot))/na.clamp_min(1e-20)
    # With two tasks, each projected gradient uses the other original gradient.
    for p,x,y in zip(parameters,a,b):
        p.grad=(x-ca*y)+(y-cb*x)
    return {'conflict':float(dot<0),'cosine':float(dot/(na*nb).sqrt().clamp_min(1e-20)),
            'ce_norm':float(na.sqrt()),'source_norm':float(nb.sqrt())}


def numerical_tests():
    p=torch.nn.Parameter(torch.tensor([1.,1.]))
    gradients((p.sum(),2*p.sum()),[p])
    assert torch.allclose(p.grad,torch.tensor([3.,3.]))
    p=torch.nn.Parameter(torch.tensor([1.,1.]))
    s=gradients((p[0],-p[0]+p[1]),[p])
    assert s['conflict']==1 and torch.allclose(p.grad,torch.tensor([.5,1.5]))
    p=torch.nn.Parameter(torch.tensor([1.,1.]))
    gradients((p[0],-p[0]),[p])
    assert torch.equal(p.grad,torch.zeros(2))
    return {'pass':True,'tests':['no-conflict=sum','two-task projection analytic','opposite-gradients finite zero']}


def audit(root):
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    torch.set_num_threads(2)
    for dep in ('gwtc3','gwtc4'):
        meta=pd.read_parquet(WARM/f'cache/{dep}/train_metadata.parquet')
        group,labels=pd.factorize(meta.waveform_parent_uid,sort=True)
        members=[np.flatnonzero(group==k) for k in range(len(labels))]
        raw=np.load(WARM/f'cache/{dep}/train_raw2s.npy',mmap_mode='r')
        features=np.load(WARM/f'cache/finelag/{dep}/train_features.npy')
        target=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy')
        for seed in body.MODEL_SEEDS:
            ck=torch.load(WARM/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/validation_selected_model.pt',weights_only=False,map_location='cpu')
            model=body.Encoder('RAW-PHASE-SOURCE').cuda().train()
            model.load_state_dict(ck['model'])
            dev.TRAIN.seed_everything(seed)
            rng=np.random.default_rng(seed)
            for trial in range(8):
                idsources=rng.choice(len(labels),64,replace=False)
                ids=np.stack([rng.choice(members[k],2,replace=False) for k in idsources]).T.reshape(-1)
                x=torch.as_tensor((features[ids]-ck['mu'])/ck['sd'],device='cuda')
                h=torch.as_tensor(np.asarray(raw[ids]),dtype=torch.float32,device='cuda')
                y=torch.as_tensor(target[ids],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    l,z=model(x,h)
                    ce=-(F.log_softmax(l.float(),-1)*y).sum(-1).mean()
                    sc=body.source_contrastive(z,torch.as_tensor(np.tile(idsources,2),device='cuda'))
                result=gradients((ce,sc),model.parameters())
                rows.append({'deployment':dep,'seed':seed,'batch':trial,**result})
            del model
    f=pd.DataFrame(rows)
    dev.csv_write(root/'TASK_GRADIENT_AUDIT.csv',f)
    summary=f.groupby('deployment').conflict.mean().to_dict()
    dev.json_write(root/'AUDIT.json',{'numerical_tests':numerical_tests(),'conflict_fraction':summary,
        'proceed':all(v>0 for v in summary.values()),'source':'training only; no test or real PE used',
        'reference':'https://arxiv.org/abs/2001.06782'})
    print(json.dumps(summary),flush=True)


def initialize(root,audit_root):
    report=json.loads((audit_root/'AUDIT.json').read_text())
    if not report['proceed']:
        raise RuntimeError('No measured gradient-conflict support in both runs')
    if root.exists():
        raise RuntimeError('New independent PCGrad output required')
    trainer.initialize(root)
    contract=json.loads((root/'contracts/FINE_LAG_TRAINING.json').read_text())
    contract.update(code='MCWF-FINELAG-EVENTPSD-PCGRAD',created_utc=datetime.now(timezone.utc).isoformat(),
        changed_mechanism='Only optimizer gradient combination: two-task PCGrad replaces CE+SupCon gradient sum. Same architecture,input,features,seeds and validation selection.',
        gradient_rule='g1p=g1-min(g1.g2,0)/||g2||^2*g2; g2p=g2-min(g1.g2,0)/||g1||^2*g1; update with g1p+g2p before norm clip',
        warm_root=str(WARM),gradient_mode='pcgrad',feature_implementation='event_psd_sample_resolved_fft_v1',
        audit_root=str(audit_root),audit_sha256=dev.sha(audit_root/'AUDIT.json'),
        reference='https://arxiv.org/abs/2001.06782',no_guaranteed_GW_benefit=True)
    dev.json_write(root/'contracts/FINE_LAG_TRAINING.json',contract)
    for dep in ('gwtc3','gwtc4'):
        (root/'cache').mkdir(exist_ok=True)
        (root/f'cache/{dep}').symlink_to((WARM/f'cache/{dep}').resolve(),target_is_directory=True)
        out=root/f'cache/finelag/{dep}'
        out.mkdir(parents=True)
        for name in ('train_features.npy','validation_features.npy'):
            (out/name).symlink_to((WARM/f'cache/finelag/{dep}/{name}').resolve())
        metadata=json.loads((WARM/f'cache/finelag/{dep}/COMPLETE.json').read_text())
        metadata.update(warm_root=str(WARM),gradient_mode='pcgrad')
        dev.json_write(out/'COMPLETE.json',metadata)
    out=root/'cache/adaptive_psd'
    out.mkdir(parents=True)
    for name in ('unwhitened_aligned_template_spectra.npy','SPECTRA_SOURCE.json'):
        shutil.copy2(WARM/'cache/adaptive_psd'/name,out/name)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    print(root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit-root',type=Path)
    a=p.parse_args()
    if a.audit_root:
        initialize(a.root,a.audit_root)
    else:
        audit(a.root)
