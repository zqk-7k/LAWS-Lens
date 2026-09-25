#!/usr/bin/env python3
"""Waveform-only pair and detector consistency; frozen time/sky and weights."""
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
import torch
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_detector_mass_density_20260907 as detector
import mcwf_finite_reference_tail_20260907 as tail

dev,body,ev=e.dev,e.body,e.ev
GAMMAS=(0.,.125,.25,.5,1.,2.,4.)
QUALITIES=(0.,.125,.25,.5,1.,2.)


def values(root,dep,ms,es,split):
    cp=root/f'detector_mass_density/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    if split=='development':
        a=np.load(cp.parent/'validation_predictions.npz');i,j=np.triu_indices(len(a['p']),1)
        f=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':a['group'][i]==a['group'][j]})
    else:
        a=detector.prediction(root,dep,ms,es,split);f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    mass=-np.log(np.sqrt(a['p'][i]*a['p'][j]).sum(1).clip(1e-15,1))
    self_evidence=np.log((a['pH']*a['pL']/ck['prior']).sum(1).clip(1e-300))
    quality=np.maximum(-np.minimum(self_evidence[i],self_evidence[j]),0.)
    outside=np.maximum(a['outside_H'],a['outside_L'])
    ood=(outside[i]>.25)|(outside[j]>.25)
    return f,{'mass':mass,'quality':quality,'self_i':self_evidence[i],'self_j':self_evidence[j],'ood':ood}


def score(f,x,spec):
    mass=np.minimum(np.log(tail.tail_probability(x['mass'],spec['mass_reference'])/.05),0.)
    quality=np.minimum(np.log(tail.tail_probability(x['quality'],spec['quality_reference'])/.05),0.)
    mass=np.where(x['ood'],0.,mass);quality=np.where(x['ood'],0.,quality)
    return f.waveform_score.to_numpy(float)+spec['gamma']*mass+spec['quality_weight']*quality,mass,quality


def run(root):
    trial=root/'trials/DETECTOR-RESOLVED-MASS-CONSISTENCY'
    if trial.exists():raise RuntimeError('Independenttrialrequired')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a':True,
        'new_model':'one sharedH1/L1single-detectorMcmixturedensityhead perrun/seed;peak2s3584templatefeatures',
        'pooled_mass':'temperedprior-correctedpredictiveaggregation;notrigorousnetworkPE dueunmarginalizedsharednuisancecoupling',
        'quality':'max(0,-min(logsum(pHi*pLi/prior),logsum(pHj*pLj/prior)))',
        'formula':'Z_FRT+gamma*pairMc-tailpenalty+quality_weight*H1L1self-consistencytailpenalty',
        'reference':'512sourceparents,onecompanionpair perparent;eventqualitiesjointlyreducedtopairminimum,not1024independentsources',
        'gamma_grid':GAMMAS,'quality_grid':QUALITIES,'alpha':.05,'no_positive_reward':True,
        'OOD':'ifanydetectorloses>25percentpredictivemassoutside5-200Msun,newpenaltiesneutral;originalFRTunchanged',
        'selection':'originalBAYESTARvalidationpriorityandguards','real_used_to_select':False,
        'frozen':['time','sky','outerweights','scope','historicaloutputs'],'no_PE_official_ID_inputs':True,
        'fresh_confirmation_required':True,'not_FPP_PE_or_lensing_Bayesfactor':True})
    selected,cache,states={},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            df,dx=values(root,dep,ms,es,'development');y=df.is_true_pair.to_numpy(bool)
            cal={'mass_reference':np.sort(dx['mass'][y]).tolist(),'quality_reference':np.sort(dx['quality'][y]).tolist(),
                'independent_sources':int(y.sum()),'shared_noise_blocks':32}
            f,x=values(root,dep,ms,es,'validation');cache[dep,es]=f,x;bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows,choices=[],[]
            for gamma in GAMMAS:
                for weight in QUALITIES:
                    spec={**cal,'gamma':gamma,'quality_weight':weight};z,*_=score(f,x,spec);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,bm)
                    rows.append({'gamma':gamma,'quality_weight':weight,'pass':ok,**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                    if ok:choices.append((e.old_selection.objective(m,gamma,weight),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            spec=min(choices,key=lambda c:c[0])[1];selected[dep,es]=spec;dev.json_write(out/'SELECTED_CONFIG.json',spec)
            states.append({'deployment':dep,'seed':es,'gamma':spec['gamma'],'quality_weight':spec['quality_weight']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'detector_consistency_selected':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else values(root,dep,ms,es,split)
                z,mass,quality=score(f,x,selected[dep,es]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score
                n['waveform_score']=z;n['detector_mass_penalty']=mass;n['detector_quality_penalty']=quality
                for key,v in x.items():n['detector_'+key]=v
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
