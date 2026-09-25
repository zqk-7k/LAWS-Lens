#!/usr/bin/env python3
"""Uniform predictive and kernel ensemble across three controlled RNC recipes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_large_population_encoder_20260907 as random_views
import mcwf_large_population_hard_batch_20260907 as hard_batch
import mcwf_cross_image_population_encoder_20260907 as cross_image

dev,body=e.dev,e.body
PARENTS=(('large_population_encoder',random_views),('large_population_hard_batch',hard_batch),
         ('cross_image_population_encoder',cross_image))


def combine(items):
    p=np.mean([np.asarray(a['p'],float) for a in items],axis=0)
    zz=[]
    for a in items:
        z=np.asarray(a['z'],float)
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        zz.append(z)
    z=np.concatenate(zz,axis=1)/np.sqrt(len(items))
    return p,z


def initialize(root):
    folder=root/'training_recipe_ensemble'
    if (folder/'contracts/RECIPE.json').exists():return
    dev.json_write(folder/'contracts/RECIPE.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a':True,
        'models':[x[0] for x in PARENTS],'weights':[1/3]*3,
        'mass_pool':'arithmetic average of the three simulation-calibrated64bin predictive distributions',
        'source_kernel':'concatenate separatelyL2normalized128D embeddings and divide bysqrt3;dot product is arithmeticmean of parentcosines;no latent-axis alignment assumption',
        'selection_of_parent_recipe':'fixed uniform combination of random-view, mass-neighbor-batch and cross-image-view controls; no per-run family choice or real-label weighting',
        'data':'same12288sources/160noiseblocks,512development sources/32noiseblocks',
        'seeds':'each of three paired initialization indices now uses three correlated recipe models;not nine independent seeds',
        'input':'unchanged H1L1 peak2s4096points40-580Hz',
        'cost':'three new RNC model inferences per seed plus frozen old waveform pipeline; cache event embeddings before pair scoring',
        'no_retraining':True,'no_PE_official_ID_inputs':True,
        'limits':'predictive mixture and discriminative kernel,not fullPE or proper lensingBF;real-budget comparisons remain adaptive development',
        'frozen':['time','sky','scope','outerweights','historicalresults'],
        'fresh_confirmation_required':True})
    (folder/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,folder/'scripts'/Path(__file__).name)


def prepare(root,dep,ms):
    initialize(root)
    out=root/f'training_recipe_ensemble/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return out
    hashes=[];items=[]
    for name,module in PARENTS:
        folder=root/f'{name}/models/{dep}/seed_{ms}'
        if not (folder/'COMPLETE.json').exists():raise RuntimeError('Parent training incomplete')
        hashes.append(dev.sha(folder/'validation_selected_model.pt'))
        items.append(np.load(folder/'validation_predictions.npz'))
    for a in items[1:]:
        if not np.array_equal(items[0]['group'],a['group']) or not np.array_equal(items[0]['truth'],a['truth']):
            raise RuntimeError('Parent development row alignment mismatch')
    p,z=combine(items)
    if not np.allclose(np.linalg.norm(z,axis=1),1.,atol=1e-10):raise RuntimeError('Kernel norm mismatch')
    n=min(50,len(z));expected=0.
    for a in items:
        zz=np.asarray(a['z'][:n],float);zz/=np.linalg.norm(zz,axis=1,keepdims=True)
        expected=expected+zz@zz.T/3
    error=float(abs(z[:n]@z[:n].T-expected).max())
    if error>1e-10:raise RuntimeError('Mean-kernel identity failed')
    out.mkdir(parents=True,exist_ok=True)
    cp=out/'validation_selected_model.pt'
    torch.save({'kind':'uniform_training_recipe_ensemble','parent_sha256':hashes,
                'parent_folders':[x[0] for x in PARENTS],'new_training':False},cp)
    np.savez_compressed(out/'validation_predictions.npz',p=p,z=z,group=items[0]['group'],truth=items[0]['truth'])
    dev.json_write(out/'COMPLETE.json',{'parent_sha256':hashes,'checkpoint_sha256':dev.sha(cp),
        'source_kernel_identity_max_abs_error':error,'new_training':False})
    return out


def prediction(root,dep,ms,es,split):
    folder=prepare(root,dep,ms)
    cp=folder/'validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    current=[dev.sha(root/f'{name}/models/{dep}/seed_{ms}/validation_selected_model.pt') for name,_ in PARENTS]
    if current!=ck['parent_sha256']:raise RuntimeError('Changed parent checkpoints')
    out=root/f'training_recipe_ensemble/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if out.exists():
        a=np.load(out)
        if str(a['checkpoint_sha256'])!=dev.sha(cp):raise RuntimeError('Changed aggregate contract')
        return a
    items=[module.prediction(root,dep,ms,es,split) for _,module in PARENTS]
    if len({a['p'].shape for a in items})!=1 or len({a['z'].shape for a in items})!=1:
        raise RuntimeError('Parent deployment row count mismatch')
    if any(not np.array_equal(np.isfinite(items[0]['p']),np.isfinite(a['p'])) for a in items[1:]):
        raise RuntimeError('Detector-availability mask mismatch')
    p,z=combine(items)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,p=p,z=z,checkpoint_sha256=dev.sha(cp),parent_sha256=np.array(current))
    return np.load(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    for dep in e.DEPS:
        for ms in body.MODEL_SEEDS:prepare(args.root,dep,ms)
