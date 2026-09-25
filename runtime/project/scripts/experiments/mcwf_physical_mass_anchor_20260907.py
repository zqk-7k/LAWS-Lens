#!/usr/bin/env python3
"""Equal neural/physical mass-distribution ensemble, calibrated on simulation."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as tf
import mcwf_finelag_encoder_20260907 as trainer

WARM=dev.PROJECT/'results/mcwf_unified_independent_mass_encoder_20260907'
GRID=(.5,1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,1024.)


def profile(features,temperature):
    power=np.expm1(np.asarray(features[:,2],dtype=float)).reshape(len(features),64,9)
    if not np.isfinite(power).all():
        raise RuntimeError('Nonfinite physical phase features')
    logits=logsumexp((power-power.max((1,2),keepdims=True))/float(temperature),axis=-1)-np.log(9.)
    return np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))


def combine(neural,features,spec):
    physical=profile(features,spec['temperature'])
    combined=.5*np.asarray(neural)+.5*physical
    combined/=combined.sum(-1,keepdims=True)
    return combined


def run(root):
    if root.exists():
        raise RuntimeError('Independent anchored-mass output required')
    trainer.initialize(root)
    cfg=json.loads((WARM/'contracts/FINE_LAG_TRAINING.json').read_text())
    cfg.update(code='MCWF-FINELAG-PHYSICAL-MASS-ANCHOR',created_utc=datetime.now(timezone.utc).isoformat(),
        changed_mechanism='Equal predictive mixture of trained phase-only mass head and tempered network quadrature profile. Source encoder/embedding unchanged from independently trained fine-lag event-PSD encoder.',
        physical_profile='softmax_Mc(logsumexp_q,spin(network_quadrature_power/T)-log9); fixed576 IMRPhenomD template grid',
        mixture_weights=[.5,.5],temperature_grid=GRID,
        selection='minimum mixed predictive CE on source-disjoint development validation; tie lower temperature',
        interpretation='physics-anchored predictive mass distribution, NOT a PE posterior or Bayes factor; profile uses maximization over arrival time/phase and incomplete intrinsic grid',
        scientific_rationale='neural class priors can dominate weak/noisy waveform shapes; enforce explicit contribution of the measured coherent phase-match profile without real PE supervision',
        references=['https://pycbc.org/pycbc/latest/html/filter.html','https://arxiv.org/abs/1706.04599'],
        waveform_only=True,source_root=str(WARM))
    dev.json_write(root/'contracts/FINE_LAG_TRAINING.json',cfg)
    dev.json_write(root/'contracts/PHYSICAL_MASS_ANCHOR.json',cfg)
    rows=[]
    for dep in ('gwtc3','gwtc4'):
        (root/'cache').mkdir(exist_ok=True)
        (root/f'cache/{dep}').symlink_to((WARM/f'cache/{dep}').resolve(),target_is_directory=True)
        out=root/f'cache/finelag/{dep}'
        out.mkdir(parents=True)
        for name in ('train_features.npy','validation_features.npy','COMPLETE.json'):
            (out/name).symlink_to((WARM/f'cache/finelag/{dep}/{name}').resolve())
        x=np.load(WARM/f'cache/finelag/{dep}/validation_features.npy')
        for ms in body.MODEL_SEEDS:
            source=WARM/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}'
            ck=torch.load(source/'validation_selected_model.pt',weights_only=False,map_location='cpu')
            data=np.load(source/'development_validation.npz')
            logits=data['logits'].astype(float)/ck['temperature']
            neural=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
            y=data['truth']
            grid=[]
            for t in GRID:
                p=combine(neural,x,{'temperature':t})
                ce=float(-(y*np.log(p.clip(1e-30))).sum(-1).mean())
                grid.append({'temperature':t,'validation_ce':ce,
                    'validation_logMc_mae':float(np.mean(abs(p@tf.LOG_CENTERS-y@tf.LOG_CENTERS)))})
            selected=min(grid,key=lambda r:(r['validation_ce'],r['temperature']))
            cp=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}'
            cp.mkdir(parents=True)
            ck['physical_mass_anchor']={'temperature':selected['temperature'],'mixture_neural_weight':.5}
            ck['mass_anchor_source_checkpoint_sha256']=dev.sha(source/'validation_selected_model.pt')
            torch.save(ck,cp/'validation_selected_model.pt')
            p=combine(neural,x,ck['physical_mass_anchor'])
            # Compatibility data stores logits such that the established temperature step recovers p.
            np.savez_compressed(cp/'development_validation.npz',logits=np.log(p.clip(1e-30))*ck['temperature'],
                embedding=data['embedding'],truth=y,group=data['group'])
            dev.csv_write(cp/'physical_temperature_grid.csv',pd.DataFrame(grid))
            dev.json_write(cp/'COMPLETE.json',{'checkpoint_sha256':dev.sha(cp/'validation_selected_model.pt'),
                'source_and_mass_networks_frozen_from':str(WARM),'selected_temperature':selected['temperature'],
                'new_PE_inputs':False})
            rows.append({'deployment':dep,'seed':ms,**selected})
    out=root/'cache/adaptive_psd'
    out.mkdir(parents=True)
    for name in ('unwhitened_aligned_template_spectra.npy','SPECTRA_SOURCE.json'):
        shutil.copy2(WARM/'cache/adaptive_psd'/name,out/name)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    dev.csv_write(root/'tables/MASS_ANCHOR_VALIDATION.csv',pd.DataFrame(rows))
    dev.json_write(root/'contracts/ALL_SIX_TRAINED.json',{'pass':True,'six_trained_networks_reused':True,
        'new_parameter_fitting':'simulation-only physical-profile temperature, not new encoder training',
        'feature_implementation':'event_psd_sample_resolved_fft_v1'})
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    run(a.root)
