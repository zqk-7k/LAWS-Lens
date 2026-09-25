#!/usr/bin/env python3
"""Pair-excluded local waveform-density correction, explicitly transductive."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_seed_plateau_20260907 as shared
import mcwf_runwise_development_20260907 as rw

dev,body,ev=e.dev,e.body,e.ev
BETAS=(0.,.125,.25,.5,1.,2.)
NEIGHBORS=(5,10,20)


def background(f,k):
    ids=np.unique(np.r_[f.idx_i,f.idx_j]);n=len(ids)
    if len(f)!=n*(n-1)//2:raise RuntimeError('Strict-scope complete graph required')
    i,j=np.searchsorted(ids,f.idx_i),np.searchsorted(ids,f.idx_j)
    s=np.full((n,n),-np.inf);v=f.waveform_score.to_numpy(float)
    if not np.isfinite(v).all():raise RuntimeError('Invalid frozen waveform values')
    s[i,j]=s[j,i]=v
    if not np.isfinite(s[~np.eye(n,dtype=bool)]).all():raise RuntimeError('Missing or duplicate pairs')
    k=min(k,n-2)
    ordered=np.sort(s,axis=1)[:,::-1]
    topk=ordered[:,:k].sum(1);topk1=ordered[:,:k+1].sum(1);boundary=ordered[:,k-1]
    # Each tested pair is removed from its own reference, without consulting labels.
    ri=np.where(v>=boundary[i],(topk1[i]-v)/k,topk[i]/k)
    rj=np.where(v>=boundary[j],(topk1[j]-v)/k,topk[j]/k)
    ref=(topk/k).mean()
    return .5*(ri+rj)-ref


def tests():
    rng=np.random.default_rng(202609151);i,j=np.triu_indices(12,1)
    f=pd.DataFrame({'idx_i':i,'idx_j':j,'waveform_score':rng.normal(size=len(i))})
    v=background(f,5);order=rng.permutation(len(f));p=rng.permutation(12)
    g=f.iloc[order].copy();g.idx_i=p[g.idx_i];g.idx_j=p[g.idx_j]
    error=float(abs(background(g,5)-v[order]).max())
    shift=f.copy();shift.waveform_score+=4.1
    shift_error=float(abs(background(shift,5)-v).max())
    if max(error,shift_error)>1e-12:raise RuntimeError('Local-density invariance failed')
    return {'event_permutation_error':error,'translation_error':shift_error,'uses_exact_pair_scope':True}


def run(root,k):
    trial=root/'trials'/f'LOCAL-WAVEFORM-DENSITY-K{k}'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a':True,'neighbors':k,'beta_grid':BETAS,
        'formula':'Zwf_new=Z_FRT-beta*(0.5*(mean_topK_excluding_j(row_i)+mean_topK_excluding_i(row_j))-catalog_mean_topK)',
        'input':'only frozen waveform scores on exactstrict-scopecompletepairgraph',
        'rationale':'test waveform hubness; a pair should be evaluated relative to unrelated high-similarity alternatives',
        'reference':'https://arxiv.org/abs/1710.04087',
        'reference_limit':'CSLS motivateslocaldensitycorrection,butthispair-excludedsingle-catalogvariantisnotthatpapersGWresult',
        'catalog_dependence':'EXPLICITtransductivewaveformstatistic;unlikepair-localZ_FRT,usesallstrict-scopecatalogwaveformvalues;noextraout-of-scopeevents',
        'not_Bayes_factor':True,'not_PE':True,'no_new_encoder_training':True,
        'no_real_PE_official_or_ID_scoring_features':True,'validation_selection':'originaldeterministicobjectiveandguards',
        'frozen':['time','sky','outerweights','scope','historicaloutputs'],'fresh_confirmation_required':True,'tests':tests()})
    data,chosen,states={},{},[]
    for dep in e.DEPS:
        for es in dev.SEEDS:
            f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/validation_pairs.parquet');r=background(f,k)
            data[dep,es,'validation']=f,r;bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows,options=[],[]
            for beta in BETAS:
                m=ev.metrics(f,f.waveform_score.to_numpy(float)-beta*r,dep,es);ok=ev.guard(m,bm)
                rows.append({'beta':beta,'pass':ok,**{a+'_'+key:v for a,b in m.items() for key,v in b.items()}})
                if ok:options.append((e.old_selection.objective(m,beta,k),beta))
            beta=min(options,key=lambda x:x[0])[1];chosen[dep,es]=beta
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            dev.json_write(out/'SELECTED_CONFIG.json',{'neighbors':k,'beta':beta})
            states.append({'deployment':dep,'seed':es,'neighbors':k,'beta':beta})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'local_density_selection':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            for split in ('validation','test','real'):
                if split=='validation':f,r=data[dep,es,split]
                else:
                    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet');r=background(f,k)
                z=f.waveform_score.to_numpy(float)-chosen[dep,es]*r
                n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z;n['waveform_local_background']=r
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm,mm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es),ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in mm:
                        for config,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for k in NEIGHBORS:run(a.root,k)
