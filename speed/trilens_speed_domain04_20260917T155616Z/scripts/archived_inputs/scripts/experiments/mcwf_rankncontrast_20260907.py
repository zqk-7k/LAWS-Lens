#!/usr/bin/env python3
"""Exact tie-aware rank contrast on continuous simulated log chirp mass."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import mcwf_development_20260905 as dev
import mcwf_finelag_encoder_20260907 as trainer

WARM=dev.PROJECT/'results/mcwf_unified_finelag_eventpsd_encoder_20260907'


def rank_loss(z,label,temperature=.1):
    z=z.float()
    label=label.float().reshape(-1)
    n=len(z)
    if n<2:
        return z.sum()*0
    keep=~torch.eye(n,dtype=torch.bool,device=z.device)
    d=(label[:,None]-label[None,:]).abs()[keep].reshape(n,n-1)
    s=(-torch.cdist(z,z,p=2,compute_mode='donot_use_mm_for_euclid_dist')/temperature)[keep].reshape(n,n-1)
    order=torch.argsort(d,dim=-1,descending=True,stable=True)
    sorted_d=torch.gather(d,1,order)
    sorted_s=torch.gather(s,1,order)
    right=torch.searchsorted((-sorted_d).contiguous(),(-sorted_d).contiguous(),right=True)-1
    denominator=torch.gather(torch.logcumsumexp(sorted_s,dim=-1),1,right)
    return (denominator-sorted_s).mean()


def tests():
    torch.manual_seed(202609074)
    z=torch.randn(8,4,requires_grad=True)
    label=torch.tensor([0.,0.,.1,.1,.4,.4,1.,1.])
    value=rank_loss(z,label)
    direct=[]
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                continue
            selected=[k for k in range(len(z)) if i!=k and abs(float(label[i]-label[k]))>=abs(float(label[i]-label[j]))]
            terms=-(z[i]-z[selected]).norm(dim=-1)/.1
            direct.append(torch.logsumexp(terms,0)+(z[i]-z[j]).norm()/.1)
    reference=torch.stack(direct).mean()
    if not torch.allclose(value,reference,atol=3e-6,rtol=1e-6):
        raise RuntimeError('RNC sorted/tie implementation differs from defining sum')
    a=torch.autograd.grad(value,z,retain_graph=True)[0]
    b=torch.autograd.grad(reference,z)[0]
    if not torch.allclose(a,b,atol=3e-6,rtol=1e-5):
        raise RuntimeError('RNC gradient mismatch')
    return {'pass':True,'loss_difference':float(abs(value-reference).detach()),
            'gradient_max_abs_difference':float((a-b).abs().max()),'label_ties_included':True}


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent RNC training directory required')
    trainer.initialize(root)
    contract=json.loads((root/'contracts/FINE_LAG_TRAINING.json').read_text())
    contract.update(code='MCWF-FINELAG-EVENTPSD-RNC',created_utc=datetime.now(timezone.utc).isoformat(),
        changed_mechanism='Add continuous logMc rank contrast to the same CE+source-SupCon representation training, warm start from fine-lag/event-PSD models.',
        objective='CE + source-SupCon + RNC(logMc)',selection='earliest minimum validation CE+0.2*SupCon+0.2*RNC',
        rnc_definition='For each anchor/positive, normalize feature similarity over all non-self samples with equal or larger absolute simulated logMc distance; average over directed non-self pairs.',
        rnc_feature='negative L2 distance of existing unit-normalized128D embeddings, temperature0.1',
        difference_from_published_default='Published reference uses unnormalized features and default temperature2; our existing unit-norm geometry retains the established0.1 contrast temperature. Same ordering objective, not a reproduction claim.',
        reference='https://arxiv.org/abs/2210.01189',reference_code='https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py',
        warm_root=str(WARM),rankncontrast=True,feature_implementation='event_psd_sample_resolved_fft_v1',
        rationale='Source-ID contrastive learning alone need not order different-source representations by intrinsic mass. Explicitly constrain continuous mass ordering using simulation labels only.',
        same_across_runs=True,numerical_test=tests(),real_PE_training=False)
    dev.json_write(root/'contracts/FINE_LAG_TRAINING.json',contract)
    dev.json_write(root/'contracts/RNC_TRAINING.json',contract)
    for dep in ('gwtc3','gwtc4'):
        (root/'cache').mkdir(exist_ok=True)
        (root/f'cache/{dep}').symlink_to((WARM/f'cache/{dep}').resolve(),target_is_directory=True)
        out=root/f'cache/finelag/{dep}'
        out.mkdir(parents=True)
        for name in ('train_features.npy','validation_features.npy'):
            (out/name).symlink_to((WARM/f'cache/finelag/{dep}/{name}').resolve())
        metadata=json.loads((WARM/f'cache/finelag/{dep}/COMPLETE.json').read_text())
        metadata.update(warm_root=str(WARM),rankncontrast=True)
        dev.json_write(out/'COMPLETE.json',metadata)
    out=root/'cache/adaptive_psd'
    out.mkdir(parents=True)
    for name in ('unwhitened_aligned_template_spectra.npy','SPECTRA_SOURCE.json'):
        shutil.copy2(WARM/'cache/adaptive_psd'/name,out/name)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    print(root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();initialize(a.root)
