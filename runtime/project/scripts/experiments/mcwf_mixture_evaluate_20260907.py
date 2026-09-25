#!/usr/bin/env python3
"""Finite validation-selected mixture-density waveform evidence experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_mixture_density_20260907 as mdn

dev,body,ev=e.dev,e.body,e.ev
BETAS=(0.,.0625,.125,.25,.5,1.,2.)
KINDS=('full','conditional','quadratic_joint')


def features(a,i,j):
    joint=mdn.pair_overlap(a,i,j)
    mass=mdn.pair_overlap(a,i,j,(0,))
    return {'full':joint,'mass':mass,'conditional':joint-mass}


def poly(x):
    return np.column_stack([x[:,0],x[:,1],x[:,0]**2,x[:,0]*x[:,1],x[:,1]**2])


def fit(root,dep,ms):
    path=root/f'mixture_density/calibration/{dep}/seed_{ms}/fit.json'
    if path.exists():return json.loads(path.read_text())
    a=np.load(root/f'mixture_density/models/{dep}/seed_{ms}/validation_predictions.npz')
    i,j=np.triu_indices(len(a['w']),1)
    y=a['group'][i]==a['group'][j]
    x=features(a,i,j)
    weights=np.where(y,.5/y.sum(),.5/(~y).sum())
    spec={'positive_source_pairs':int(y.sum()),'source_count':len(np.unique(a['group'])),
        'noise_blocks':32,'independent_noise_count_is_not_pair_count':True,
        'checkpoint_sha256':dev.sha(root/f'mixture_density/models/{dep}/seed_{ms}/validation_selected_model.pt')}
    for kind in ('full','conditional'):
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x[kind],y.astype(float),sample_weight=weights)
        p=np.clip(iso.y_thresholds_,1/(y.sum()+2),1-1/(y.sum()+2))
        spec[kind]={'knots':iso.X_thresholds_.tolist(),'loglr':(np.log(p)-np.log1p(-p)).tolist(),
            'minimum':float(x[kind].min()),'maximum':float(x[kind].max())}
    xx=np.column_stack([x['mass'],x['conditional']])
    mean=np.average(xx,axis=0,weights=weights);sd=np.sqrt(np.average((xx-mean)**2,axis=0,weights=weights)).clip(1e-6)
    model=LogisticRegression(C=1,max_iter=1000).fit(poly((xx-mean)/sd),y,sample_weight=weights*y.sum()*2)
    spec['quadratic_joint']={'mean':mean.tolist(),'sd':sd.tolist(),'coef':model.coef_[0].tolist(),
        'intercept':float(model.intercept_[0]),'minimum':xx.min(0).tolist(),'maximum':xx.max(0).tolist()}
    path.parent.mkdir(parents=True,exist_ok=True)
    dev.json_write(path,spec)
    np.savez_compressed(path.with_suffix('.npz'),**x,y=y,i=i,j=j)
    return spec


def get(root,dep,ms,es,split):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    path=root/f'mixture_density/pair_features/{dep}/seed_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if not np.array_equal(a['i'],f.idx_i) or not np.array_equal(a['j'],f.idx_j):
            raise RuntimeError('Pair ordering changed')
        return f,{k:a[k] for k in ('full','mass','conditional')}
    a=mdn.prediction(root,dep,ms,es,split)
    x=features(a,f.idx_i.to_numpy(int),f.idx_j.to_numpy(int))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**x,i=f.idx_i.to_numpy(int),j=f.idx_j.to_numpy(int))
    return f,x


def score(f,x,spec):
    kind=spec['kind'];cal=spec[kind]
    if kind=='quadratic_joint':
        xx=np.column_stack([x['mass'],x['conditional']])
        raw=poly((xx-np.array(cal['mean']))/np.array(cal['sd']))@np.array(cal['coef'])+cal['intercept']
        ood=((xx<np.array(cal['minimum']))|(xx>np.array(cal['maximum']))).any(1)
    else:
        raw=np.interp(x[kind],cal['knots'],cal['loglr'])
        ood=(x[kind]<cal['minimum'])|(x[kind]>cal['maximum'])
    increment=np.clip(np.where(ood,np.minimum(raw,0),raw),-4,4)
    return f.waveform_score.to_numpy(float)+spec['beta']*increment,increment,ood


def run(root):
    trial=root/'trials/JOINT-INTRINSIC-MIXTURE-DENSITY'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a':True,'formula':'Z_FRT+beta*bounded_LR(joint_intrinsic_predictive_overlap)',
        'predictive_targets':'logMc,logitq,atanhchieff;4componentfullcovarianceMDN;waveform-only2sphysicaltemplates',
        'development_calibration':'512newsource validation companions1pair/source,class-balancednegativepairs;32noiseblocks,not524k independentunits',
        'kinds':KINDS,'betas':BETAS,'cap':4,'quadratic_regularization_C':1,
        'kind_interpretation':'full3Dproductintegral;conditional=joint-minus1DMcproductintegral;quadraticjoint=class-balancedlogisticratio onthetwofeatures',
        'not_independent_Bayes_factors':True,'not_astrophysical_PE':True,
        'selection':'BAYESTAR validation only, frozen original objective and per-seed noninferiorityguards',
        'frozen':['time','sky','outerC-fixedweights','scope','allhistoricaloutputs'],'no_real_PE_or_official_inputs':True,
        'real_audit':'adaptive development,notblindvalidation','newconfirmation_required':True})
    selected,cache,states={},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            cal=fit(root,dep,ms);f,x=get(root,dep,ms,es,'validation');cache[dep,es]=f,x
            bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows,choices=[],[]
            for kind in KINDS:
                for beta in BETAS:
                    spec={**cal,'kind':kind,'beta':beta};z,*_=score(f,x,spec);mm=ev.metrics(f,z,dep,es)
                    ok=ev.guard(mm,bm)
                    rows.append({'kind':kind,'beta':beta,'pass':ok,**{a+'_'+k:v for a,b in mm.items() for k,v in b.items()}})
                    if ok:choices.append((e.old_selection.objective(mm,0,beta)+(kind,),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True)
            dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows));spec=min(choices,key=lambda a:a[0])[1]
            selected[dep,es]=spec;dev.json_write(out/'SELECTED_CONFIG.json',spec)
            states.append({'deployment':dep,'seed':es,'kind':spec['kind'],'beta':spec['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'mixture_selected':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else get(root,dep,ms,es,split)
                z,inc,ood=score(f,x,selected[dep,es]);n=f.copy()
                n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z
                n['joint_intrinsic_increment']=inc;n['joint_intrinsic_ood']=ood
                for k,v in x.items():n['joint_predictive_'+k]=v
                for col in ('time_score','sky_raw_log_bf'):assert np.array_equal(n[col],f[col])
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True)
                n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm,mm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es),ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in bm:
                        for config,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:
                            rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
