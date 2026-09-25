#!/usr/bin/env python3
"""Frozen joint-predictive deep ensemble with validation-only integration."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_joint_intrinsics_evidence_20260908 as joint
import mcwf_joint_fusion_retune_20260908 as refit
ev,t,dev,cf,model=joint.ev,joint.t,joint.dev,joint.cf,joint.model
UPSTREAM=joint.UPSTREAM
MASS_CROSS_CACHE=P/'results/mcwf_multirate_integration_exploratory_20260908T151518Z/raw_predictions'


def cross_prediction(root,dep,slot,seed,split,kind):
    path=root/f'predictions/{kind}/{dep}/{slot}_{seed}_{split}.npz'
    if path.exists():return dict(np.load(path))
    ck=torch.load(root/f'models/{kind}/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    cp=torch.load(ck['mass_checkpoint'],map_location='cpu',weights_only=False)
    if dev.sha(Path(ck['mass_checkpoint']))!=ck['mass_checkpoint_sha256']:
        raise RuntimeError('Frozen mass checkpoint changed')
    # The earlier integration run owns cross-seed mass predictions. Never create
    # missing files in the already sealed MULTIRATE directory.
    mass_path=MASS_CROSS_CACHE/f'{dep}/{slot}_{seed}_{split}.npz'
    if not mass_path.exists():raise RuntimeError('Missing read-only cross-seed mass predictions: '+str(mass_path))
    mass=dict(np.load(mass_path))
    filename='real.npy' if split=='real' else f'{seed}_{split}.npy'
    x=np.concatenate([t.old_features(dep,seed,split),np.load(model.UPSTREAM/f'features/{dep}'/filename)],1)
    rep=model.original.representations(x,cp)
    head=model.Head(ck['conditional_dimensions']).cuda().eval();head.load_state_dict(ck['model'])
    residual=model.original.infer(head,((rep-ck['mu'])/ck['sd']).reshape(-1,103)).reshape(len(rep),253,-1)
    conditional=softmax(residual/ck['temperature']+np.log(ck['conditional_prior'].clip(1e-12))[None],-1)
    hi=np.searchsorted(model.old.LOG_CENTERS,model.old.CENTERS,side='right').clip(1,252);lo=hi-1
    w=((model.old.CENTERS-model.old.LOG_CENTERS[lo])/(model.old.LOG_CENTERS[hi]-model.old.LOG_CENTERS[lo])).clip(0,1)
    conditional=(1-w[None,:,None])*conditional[:,lo]+w[None,:,None]*conditional[:,hi]
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        out=np.full((len(full),512,ck['conditional_dimensions']),np.nan,np.float32);out[valid]=conditional;conditional=out
    density=mass['p'][:,:,None]*conditional;valid=np.isfinite(mass['p']).all(1)
    error=float(abs(density[valid].sum(-1)-mass['p'][valid]).max())
    if error>1e-6:raise RuntimeError('Cross-prediction mass marginal mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,joint=density.astype(np.float32),p=mass['p'],outside=mass['outside'])
    dev.json_write(path.with_suffix('.json'),{'mass_marginal_error':error,'mass_prediction_path':str(mass_path),
        'mass_prediction_sha256':dev.sha(mass_path),'mass_model_sha256':ck['mass_checkpoint_sha256'],'kind':kind})
    return dict(np.load(path))


def initialize(root):
    if root.exists():raise RuntimeError('Fresh directory required')
    for name in ('contracts','tables','models','calibration','evaluation','reports','figures','scripts','logs','manifest','cache'):
        (root/name).mkdir(parents=True)
    contract={
        'id':'MCWF-JOINT-PREDICTIVE-ENSEMBLE-15','UTC':datetime.now(timezone.utc).isoformat(),
        'status':t.STATUS,'goal_achieved':False,'same_both_runs':True,'upstream':str(UPSTREAM),
        'mechanism':'Arithmetic mean of three frozen joint predictive probabilities BEFORE computing overlap;not mean ranks or selection of the best real-catalog seed.',
        'kinds':list(model.KINDS),'ensemble_slots':list(t.MODEL_SLOTS),
        'uncertainty_note':'The three catalog/calibration evaluation seeds share the same frozen three-model ensemble. Their SD is not three independent retrained-ensemble variance.',
        'head_models':'No new training. Reuse conditional chi and conditional eta+chi models from10;every new cross-model prediction is stored only in this directory.',
        'mass_marginal':'Ensemble mass marginal must equal arithmetic mean of frozen mass marginals within1e-6;all probability rows normalize to1. Prior arrays must agree acrossslots.',
        'score':'Full joint logBhattacharyya proxy calibrated with source/noise-disjoint fit/audit halves as12. Not a physical PE or Bayes factor.',
        'inner_grid':{'gamma':list(joint.GAMMA),'beta':list(joint.BETA),'integration':'ADD to OMC;unchanged OMC explicit'},
        'outer_grid':'FIXED originalC-fixed,orPOSITIVE simplex0.05 withallweights>=.05,plus exact normalizedC-fixed. Originaltime/skyraw unchanged.',
        'priorities':['candidate:F50,F90,-AP,-R10,-R1','retrieval:-R10,-R1,-AP,F50,F90'],
        'tie':'InnerunchangedOMCthenleastgamma^2+beta^2;outerclosestnormalizedC-fixedthenlexW/T/S.',
        'guards':'Same perseed waveform+fusion guardrails relative original retainedOMC. No realcatalog PE/official labels or event IDs enter these selections.',
        'frozen':['allneuralmodels','Z_time','sky_raw_log_bf','scope','history','paper'],
        'changed_channels':[],'outer_weights_changed':True,
        'external':'Adaptive development only. Families motivated by earlier failures;numeric selection is simulation-only. Fresh source/noise confirmation required if jointly successful.',
        'references':['https://arxiv.org/abs/1612.01474'],
        'reference_scope':'General motivation for averaging independently trained predictive distributions;not evidence this particular GW model is physically calibrated.'}
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    protected=t.protected()
    for kind in model.KINDS:
        for dep in t.DEPS:
            for slot in t.MODEL_SLOTS:
                p=UPSTREAM/f'models/{kind}/{dep}/seed_{slot}/selected.pt'
                dest=root/p.relative_to(UPSTREAM);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
                protected.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(protected))
    shutil.copy2(__file__,root/'scripts/joint_ensemble.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),
        'code_sha256':dev.sha(Path(__file__))})


def features(root,dep,slot,seed,split,kind):
    # Model probabilities for real events do not depend on the catalog seed.
    cache_seed=t.SEEDS[0] if split=='real' else seed
    path=root/f'cache/{kind}/{dep}/ensemble_{cache_seed}_{split}.npz'
    if path.exists():return dict(np.load(path))
    average=None;mass=None;outside=[];prior=None;inputs=[]
    for s in t.MODEL_SLOTS:
        upstream=UPSTREAM/f'predictions/{kind}/{dep}/{s}_{cache_seed}_{split}.npz'
        model.KIND=kind
        if upstream.exists():a=dict(np.load(upstream));inputs.append(str(upstream))
        else:
            a=cross_prediction(root,dep,s,cache_seed,split,kind)
            inputs.append(str(root/f'predictions/{kind}/{dep}/{s}_{cache_seed}_{split}.npz'))
        ck=torch.load(root/f'models/{kind}/{dep}/seed_{s}/selected.pt',map_location='cpu',weights_only=False)
        cp=model.original.interpolate_rows(ck['conditional_prior'],model.old.CENTERS)
        this_prior=ck['prior'][:,None]*cp
        if prior is not None and not np.allclose(prior,this_prior,atol=1e-12,rtol=1e-8):
            raise RuntimeError('Ensemble members do not share a common training prior')
        prior=this_prior
        if average is None:average=np.asarray(a['joint'],dtype=np.float64)/3.;mass=np.asarray(a['p'],dtype=np.float64)/3.
        else:average+=a['joint']/3.;mass+=a['p']/3.
        outside.append(a['outside'])
    good=np.isfinite(mass).all(1)
    marginal_error=float(abs(average[good].sum(-1)-mass[good]).max())
    norm_error=float(abs(average[good].sum((1,2))-1.).max())
    if max(marginal_error,norm_error)>1e-6:raise RuntimeError('Ensemble marginal/normalization changed')
    z=torch.as_tensor(average,device='cuda',dtype=torch.float64).flatten(1)
    bc=(z.sqrt()@z.sqrt().T).cpu().numpy().clip(1e-15,1.)
    weighted=z/torch.as_tensor(np.sqrt(prior).reshape(1,-1),device='cuda')
    bf=(weighted@weighted.T).cpu().numpy().clip(1e-300)
    result={'joint_logbc':np.log(bc),'joint_bc':bc,'joint_logbf':np.log(bf),'outside':np.mean(outside,axis=0)}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**result)
    dev.json_write(root/f'tables/ensemble_replay/{kind}_{dep}_{cache_seed}_{split}.json',
        {'mass_marginal_error':marginal_error,'probability_normalization_error':norm_error,'input_predictions':inputs,
         'input_sha256':[dev.sha(Path(p)) for p in inputs],'scope':'valid waveform events;invalidrows remainNaN,notzero'})
    return result


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    configs,innergrid,outergrid,audits=[],[],[],[]
    grid=np.asarray([(i/20,j/20,(20-i-j)/20) for i in range(1,20) for j in range(1,20-i)])
    def metric_key(r,policy):
        return ((r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1'])
                if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9']))
    for kind in model.KINDS:
        for dep in t.DEPS:
            specs,audit=joint.calibration(root,dep,t.MODEL_SLOTS[0],kind);audits+=audit;spec=specs['BC']
            for slot,seed in zip(t.MODEL_SLOTS,t.SEEDS):
                f,pen,inc,_=joint.values(root,dep,slot,seed,'validation',kind,spec)
                z=f.waveform_score.to_numpy(float);w=cf.frozen_weights(dep,seed);normalized=w/w.sum()
                bm,bwm=cf.fast_metrics(f,cf.channels(f,z)@w),cf.fast_metrics(f,z)
                rows=[{'gamma':0.,'beta':0.,'unchanged_OMC':True,'pass':True,**bm}]
                for gamma in joint.GAMMA:
                    for beta in joint.BETA:
                        wf=z+gamma*pen+beta*inc
                        mm,wm=cf.fast_metrics(f,cf.channels(f,wf)@w),cf.fast_metrics(f,wf)
                        row={'gamma':gamma,'beta':beta,'unchanged_OMC':False,'pass':cf.guard(mm,bm) and cf.guard(wm,bwm),**mm}
                        rows.append(row);innergrid.append({'kind':kind,'deployment':dep,'seed':seed,**row})
                for policy in ('CANDIDATE','RETRIEVAL'):
                    chosen=min((r for r in rows if r['pass']),key=lambda r:(*metric_key(r,policy),not r['unchanged_OMC'],r['gamma']**2+r['beta']**2,r['gamma'],r['beta']))
                    c={'method':kind.replace('CONDITIONAL','ENSEMBLE')+'-FIXED-'+policy,'deployment':dep,'slot':slot,'seed':seed,
                       'kind':kind,'statistic':'BC','integration':'ADD','gamma':chosen['gamma'],'beta':chosen['beta'],
                       'unchanged_OMC':chosen['unchanged_OMC'],'calibration':spec,'weights':w.tolist()}
                    configs.append(c)
                    wf=z if c['unchanged_OMC'] else z+c['gamma']*pen+c['beta']*inc
                    wm=cf.fast_metrics(f,wf);candidates=[]
                    points=np.vstack([grid,normalized]) if not np.any(np.all(np.isclose(grid,normalized,atol=1e-12),axis=1)) else grid
                    for weights in points:
                        m=cf.fast_metrics(f,cf.channels(f,wf)@weights)
                        row={'weights':weights.tolist(),'pass':cf.guard(m,bm) and cf.guard(wm,bwm),**m}
                        candidates.append(row);outergrid.append({'kind':kind,'deployment':dep,'seed':seed,'inner_policy':policy,**row})
                    for outer in ('CANDIDATE','RETRIEVAL'):
                        choice=min((r for r in candidates if r['pass']),key=lambda r:(*metric_key(r,outer),float(np.square(np.asarray(r['weights'])-normalized).sum()),*r['weights']))
                        configs.append({**c,'method':kind.replace('CONDITIONAL','ENSEMBLE')+'-REFIT-'+policy+'-'+outer,'weights':choice['weights']})
    dev.csv_write(root/'tables/VALIDATION_INNER_GRID.csv',pd.DataFrame(innergrid));dev.csv_write(root/'tables/VALIDATION_OUTER_GRID.csv',pd.DataFrame(outergrid))
    dev.csv_write(root/'tables/CALIBRATION_AUDIT.csv',pd.DataFrame(audits))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{**{k:v for k,v in c.items() if k not in ('calibration','weights')},
        **dict(zip(('lambda_waveform','lambda_time','lambda_sky'),c['weights']))} for c in configs]))
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'real_or_test_selection':False,'outer_weights_changed':True})


def scored(root,c,split):
    return joint.scored(root,c,split)


def evaluate(root):
    rows,checks=[],[]
    for c in ev.load_selections(root):
        for split in ('validation','test'):
            f,z,extra=scored(root,c,split);base=cf.read(c['deployment'],c['seed'],split)
            out=f.copy();out['retained_OMC_waveform']=f.waveform_score;out['waveform_score']=z
            out['final_score']=cf.channels(f,z)@np.asarray(c['weights'])
            for key,value in extra.items():out[key]=value
            oldweights=cf.frozen_weights(c['deployment'],c['seed'])
            for mode,score,old in [('fusion',out.final_score.to_numpy(float),cf.channels(base,base.waveform_score)@oldweights),('waveform',z,base.waveform_score.to_numpy(float))]:
                m,b=cf.fast_metrics(f,score),cf.fast_metrics(base,old);reference=dev.BASE.full_metrics(f,score)
                for key in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                    if abs(m[key]-reference[key])>1e-10:raise RuntimeError('Metric replay failed')
                rows.append({'method':c['method'],'deployment':c['deployment'],'seed':c['seed'],'split':split,'mode':mode,'guard_pass':cf.guard(m,b),**m})
            dest=root/f'evaluation/{c["method"]}/{c["deployment"]}/seed_{c["seed"]}';dest.mkdir(parents=True,exist_ok=True)
            out.to_parquet(dest/f'{split}_pairs.parquet',index=False)
            for col in ('time_score','sky_raw_log_bf'):
                if not np.array_equal(out[col],base[col]):raise RuntimeError('Frozen physicalscore changed')
            checks.append({'method':c['method'],'deployment':c['deployment'],'seed':c['seed'],'split':split,'time_sky_raw_max_difference':0})
    a=pd.DataFrame(rows);dev.csv_write(root/'tables/RETRIEVAL_PER_SEED.csv',a)
    columns=['macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9']
    s=a.groupby(['method','deployment','split','mode'])[columns].agg(['mean','std']).reset_index();s.columns=['_'.join(c).rstrip('_') for c in s.columns]
    dev.csv_write(root/'tables/RETRIEVAL_SUMMARY.csv',s);dev.csv_write(root/'tables/FROZEN_CHANNEL_REPLAY.csv',pd.DataFrame(checks))
    dev.json_write(root/'contracts/INJECTION_COMPARISON_COMPLETE.json',{'rows':len(a),'independent_confirmation':False,'raw_time_sky_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True,
        choices=['initialize','select','evaluate','real','assess']);a=p.parse_args()
    joint.features=features;ev.scored=scored
    if a.stage in ('initialize','select','evaluate'):globals()[a.stage](a.root)
    else:getattr(ev,a.stage)(a.root)
