#!/usr/bin/env python3
"""Sample-resolved, zero-padded quadrature correlations for 2-s inputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body

LAGS=304
NFFT=8192
FS=2048


@torch.no_grad()
def features(x,bank,batch=12,step=1,network_radius=21):
    templates=torch.as_tensor(bank,dtype=torch.float32,device='cuda')
    kernels=torch.fft.rfft(templates,n=NFFT)
    lags=torch.arange(-LAGS,LAGS+1,step,device='cuda')
    indices=lags.remainder(NFFT)
    output=[]
    for start in range(0,len(x),batch):
        values=torch.as_tensor(np.asarray(x[start:start+batch,:,-4096:],dtype=np.float32),device='cuda')
        if values.shape[1:]!=(2,4096) or not torch.isfinite(values).all():
            raise RuntimeError('Invalid two-detector 2-s waveform')
        values=(values-values.mean(-1,keepdim=True))/values.std(-1,keepdim=True).clamp_min(1e-6)
        data=torch.fft.rfft(values,n=NFFT)
        powers=[]
        for d in range(2):
            blocks=[]
            for k in range(0,len(bank),64):
                product=data[:,d,None,None,:]*kernels[None,k:k+64,d,:,:].conj()
                corr=torch.fft.irfft(product,n=NFFT)[...,indices]
                blocks.append(corr.square().sum(2))
            powers.append(torch.cat(blocks,1))
        radius=int(network_radius//step)
        l1=F.max_pool1d(powers[1],2*radius+1,stride=1,padding=radius)
        net=(powers[0]+l1).amax(-1)
        f=torch.stack([p.amax(-1) for p in powers]+[net],1)
        output.append(torch.log1p(f).reshape(len(values),3,64,3,3).cpu().numpy())
    return np.concatenate(output)


def numerical_test():
    rng=np.random.default_rng(202609071)
    values=rng.normal(size=(2,4096))
    kernel=rng.normal(size=4096)
    spectrum=np.fft.rfft(values,n=NFFT)*np.fft.rfft(kernel,n=NFFT).conj()
    fft=np.fft.irfft(spectrum,n=NFFT)
    lags=np.arange(-LAGS,LAGS+1)
    padded=np.pad(values,((0,0),(LAGS,LAGS)))
    dot=np.stack([padded[:,s:s+4096]@kernel for s in range(2*LAGS+1)],1)
    error=float(np.max(np.abs(dot-fft[:,lags%NFFT])))
    if error>1e-9:
        raise RuntimeError(f'FFT/direct lag convention mismatch: {error}')
    return {'float64_direct_dot_max_abs_error':error,'zero_padding':NFFT,'input_length':4096,
            'time_step_seconds':1/FS,'lag_range_seconds':[-LAGS/FS,LAGS/FS],
            'H1L1_light_travel_time_seconds':.010012846152267725,
            'network_lag_radius_samples':21,'network_radius_rule':'ceil(light_travel_time*2048)'}


def prepare(root,dep):
    cache=body.prepare(root,dep)
    out=root/f'cache/finelag/{dep}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return out
    checkpoint=body.PREVIOUS.parent/'main_o3_mcwf_encoder_v2_20260906T153600Z'/f'models/RAW-PHASE-SOURCE/{dep}/seed_{body.MODEL_SEEDS[0]}/validation_selected_model.pt'
    ck=torch.load(checkpoint,weights_only=False,map_location='cpu')
    bankpath=Path(ck['bankpath'])
    if not bankpath.is_absolute():
        bankpath=dev.PROJECT/bankpath
    bank=np.load(bankpath)
    stats=[]
    for split in ('train','validation'):
        path=out/f'{split}_features.npy'
        if path.exists():
            continue
        x=np.load(cache/f'{split}_raw2s.npy',mmap_mode='r')
        start=time.perf_counter()
        z=features(x,bank)
        np.save(path,z)
        row={'deployment':dep,'split':split,'rows':len(x),'seconds':time.perf_counter()-start,'feature_sha256':dev.sha(path)}
        stats.append(row)
        print(json.dumps(row),flush=True)
    dev.json_write(out/'COMPLETE.json',{'bankpath':str(bankpath),'bank_sha256':dev.sha(bankpath),'splits':stats,'numerical_test':numerical_test()})
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',choices=('gwtc3','gwtc4'),required=True)
    a=p.parse_args()
    prepare(a.root,a.deployment)


if __name__=='__main__':
    main()
