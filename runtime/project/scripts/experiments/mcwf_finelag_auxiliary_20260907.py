#!/usr/bin/env python3
"""Recompute auxiliary validation waveform features without changing source/noise splits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_conditional_waveform_calibration_20260906 as c

p=argparse.ArgumentParser()
p.add_argument('--trained-root',type=Path,required=True)
a=p.parse_args()
out=a.trained_root/'auxiliary_validation'
for dep in ('gwtc3','gwtc4'):
    actual_meta=pd.read_parquet(a.trained_root/f'cache/{dep}/validation_metadata.parquet')
    for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
        src=c.SOURCE/dep/f'seed_{es}'
        dest=out/dep/f'seed_{es}'
        dest.mkdir(parents=True,exist_ok=False)
        f=pd.read_parquet(src/'pairs.parquet')
        meta=pd.read_parquet(src/'event_metadata.parquet')
        for col in ('waveform_parent_uid','image'):
            if not np.array_equal(meta[col].to_numpy(),actual_meta[col].to_numpy()):
                raise RuntimeError('Auxiliary event ordering mismatch')
        model=a.trained_root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}'
        ck=torch.load(model/'validation_selected_model.pt',map_location='cpu',weights_only=False)
        saved=np.load(model/'development_validation.npz')
        logp=saved['logits'].astype(float)/ck['temperature']
        logp-=np.logaddexp.reduce(logp,axis=-1,keepdims=True)
        x=ev.features(f,np.exp(logp),saved['embedding'])
        for name,key in (('new_encoder_cosine','cosine'),('new_mass_similarity','mass'),('new_mass_predictive_BC','predictive_BC'),('new_mass_pred_logmc_i','mean_i'),('new_mass_pred_logmc_j','mean_j')):
            f[name]=x[key]
        f.to_parquet(dest/'pairs.parquet',index=False)
        meta.to_parquet(dest/'event_metadata.parquet',index=False)
        dev.json_write(dest/'PROVENANCE.json',{'source_sha256':dev.sha(src/'pairs.parquet'),
            'checkpoint_sha256':dev.sha(model/'validation_selected_model.pt'),'event_order_exact':True,
            'source_noise_labels_frozen':True,'public_PE_inputs':False})
print(out,flush=True)
