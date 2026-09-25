#!/usr/bin/env python3
"""Replace, rather than duplicate, mass evidence; test predictive averaging."""
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

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_multirate_train_20260908 as trained
import mcwf_temporal_response_evaluate_20260908 as ev
t,dev,cf=trained.t,trained.dev,ev.cf
UPSTREAM=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
KINDS=('SINGLE','ENSEMBLE')
GAMMAS=(0.,.125,.25,.5,1.,2.,4.)
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


def raw_prediction(root,dep,slot,es,split):
    cp=UPSTREAM/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt'
    if split=='development':return dict(np.load(cp.parent/'development_predictions.npz'))
    path=root/f'raw_predictions/{dep}/{slot}_{es}_{split}.npz'
    if path.exists():return dict(np.load(path))
    archived=UPSTREAM/f'predictions/MULTIRATE/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if archived.exists():a=dict(np.load(archived))
    else:
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        low=UPSTREAM/f'features/{dep}'/('real.npy' if split=='real' else f'{es}_{split}.npy')
        x=np.concatenate([t.old_features(dep,es,split),np.load(low)],1)
        model=trained.Predictor('MULTIRATE').cuda().eval();model.load_state_dict(ck['model'])
        p,b=t.old.probability(t.old.infer(model,(x-ck['mu'])/ck['sd']),ck['temperature'])
        if split=='real':
            full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
            pp,bb=np.full((len(full),512),np.nan),np.full(len(full),np.nan);pp[valid],bb[valid]=p,b;p,b=pp,bb
        a={'p':p,'outside':b}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)
    return a


def predict(root,dep,kind,slot,es,split):
    path=root/f'predictions/{kind}/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if path.exists():return np.load(path)
    sources=[raw_prediction(root,dep,s,es,split) for s in (t.MODEL_SLOTS if kind=='ENSEMBLE' else (slot,))]
    p=np.mean([a['p'] for a in sources],0);outside=np.mean([a['outside'] for a in sources],0)
    finite=np.isfinite(p).all(1)
    if not np.allclose(p[finite].sum(1),1.,atol=1e-6):raise RuntimeError('Predictive probability not normalized')
    extra={k:sources[0][k] for k in ('group','truth') if k in sources[0]}
    for a in sources[1:]:
        for k in extra:
            if not np.array_equal(a[k],extra[k]):raise RuntimeError('Ensemble source ordering mismatch')
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,outside=outside,**extra)
    return np.load(path)


def initialize(root):
    if root.exists():raise RuntimeError('Fresh independent directory required')
    for n in ('contracts','scripts','logs','models','predictions','raw_predictions','calibration','evaluation','tables','reports','manifest','figures'):(root/n).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-MULTIRATE-INTEGRATION-07','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,
        'upstream':str(UPSTREAM),'goal_achieved':False,'same_both_runs':True,
        'rationale':'06 improved injection discrimination and O3 mass agreement but not both-run PE+official target. A second mass correction added on top of retained mass evidence can count correlated waveform evidence twice. Compare an explicit replacement and predictive averaging,not changes to time/sky.',
        'factorial':{'predictive_model':list(KINDS),'mass_integration':['ADD','REPLACE']},
        'ADD':'Z_OMC + gamma*new_finite_mass_tail + beta*new_bounded_mass_increment',
        'REPLACE':'Z_FRT + gamma*new_finite_mass_tail + beta*new_bounded_mass_increment;remove old OMC mass layer,not the waveform-shape/FRT evidence',
        'unchanged_OMC_always_available':True,
        'ensemble':'arithmetic mean of three mass predictive probabilities,not sum of log-evidences;no extra independent systems claimed',
        'neural_training':False,'input':'same frozen peak2s and20-80Hz16s waveform features;no PE or event-ID input',
        'gamma_grid':GAMMAS,'beta_grid':BETAS,
        'calibration':'same source/noise-disjoint development halves and bounded isotonic rule as06',
        'selection':'same per-seed simulated validation candidate/retrieval priority;guardbothwaveform/fusion;ties:baseline first,then coefficient norm andlexical',
        'frozen':['all checkpoints','time','sky_raw_log_bf','outerweights','scope','historical outputs','paper'],
        'real_feedback':'adaptive development record,notblind confirmation;real PE/official not used to select current coefficients',
        'reference':'https://arxiv.org/abs/1612.01474','reference_limit':'predictive-averaging motivation,not a calibration or GW-lensing result'})
    rows=t.protected()
    rows += [{'path':str(f),'sha256':dev.sha(f),'bytes':f.stat().st_size} for f in (UPSTREAM/'models/MULTIRATE').glob('*/*/selected.pt')]
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows));shutil.copy2(__file__,root/'scripts/multirate_integration.py')
    audit=[]
    for dep in t.DEPS:
        oldcfg=json.loads((t.PAIRS.parent/f'contracts/{dep}_SELECTED.json').read_text())['configs']
        for slot,es in zip(t.MODEL_SLOTS,t.SEEDS):
            f=cf.read(dep,es,'validation');c=oldcfg[str(es)]
            value=f.FRT_baseline_waveform_score+c['gamma']*f.mass_penalty+c['beta']*f.mass_increment
            err=float(abs(value-f.waveform_score).max())
            if err>1e-10:raise RuntimeError('Cannot exactly decompose retained mass layer')
            audit.append({'deployment':dep,'seed':es,'old_gamma':c['gamma'],'old_beta':c['beta'],'replay_max_abs':err})
            for kind in KINDS:
                a=predict(root,dep,kind,slot,0,'development')
                folder=root/f'models/{kind}/{dep}/seed_{slot}';folder.mkdir(parents=True)
                np.savez_compressed(folder/'development_predictions.npz',**dict(a))
                paths=[UPSTREAM/f'models/MULTIRATE/{dep}/seed_{s}/selected.pt' for s in (t.MODEL_SLOTS if kind=='ENSEMBLE' else (slot,))]
                priors=[torch.load(p,map_location='cpu',weights_only=False)['prior'] for p in paths]
                if not all(np.array_equal(priors[0],p) for p in priors):raise RuntimeError('Ensemble prior mismatch')
                torch.save({'prior':priors[0],'upstream_models':[{'path':str(p),'sha256':dev.sha(p)} for p in paths],
                    'not_a_trainable_checkpoint':True},folder/'selected.pt')
    dev.csv_write(root/'tables/RETAINED_MASS_LAYER_REPLAY.csv',pd.DataFrame(audit))
    dev.json_write(root/'contracts/FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def selected_score(f,p,i,c):
    if c.get('baseline_only'):return f.waveform_score.to_numpy(float)
    anchor=f.FRT_baseline_waveform_score if c['integration']=='REPLACE' else f.waveform_score
    return anchor.to_numpy(float)+c['gamma']*p+c['beta']*i


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    choices=[];grid=[];audits=[]
    for dep in t.DEPS:
        for kind in KINDS:
            for slot,es in zip(t.MODEL_SLOTS,t.SEEDS):
                spec,audit=ev.simulated_calibration(root,dep,kind,slot);audits+=audit
                f,p,i,_=ev.pair_values(root,dep,kind,slot,es,'validation',spec)
                w=cf.frozen_weights(dep,es);old=f.waveform_score.to_numpy(float)
                bm,bwm=cf.fast_metrics(f,cf.channels(f,old)@w),cf.fast_metrics(f,old)
                for mode in ('ADD','REPLACE'):
                    rows=[{'baseline_only':True,'gamma':0.,'beta':0.,'pass':True,**bm}]
                    for gamma in GAMMAS:
                        for beta in BETAS:
                            z=selected_score(f,p,i,{'integration':mode,'gamma':gamma,'beta':beta})
                            mm,wm=cf.fast_metrics(f,cf.channels(f,z)@w),cf.fast_metrics(f,z)
                            row={'baseline_only':False,'gamma':gamma,'beta':beta,'pass':cf.guard(mm,bm) and cf.guard(wm,bwm),**mm}
                            rows.append(row)
                            grid.append({'deployment':dep,'kind':kind,'integration':mode,'seed':es,**row,**{'waveform_'+k:v for k,v in wm.items()}})
                    for policy in ('CANDIDATE','RETRIEVAL'):
                        def key(r):
                            primary=(r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1']) if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'])
                            return (*primary,not r['baseline_only'],r['gamma']**2+r['beta']**2,r['gamma'],r['beta'])
                        win=min((r for r in rows if r['pass']),key=key)
                        choices.append({'deployment':dep,'kind':kind,'slot':slot,'seed':es,'method':kind+'-'+mode+'-'+policy,
                            'integration':mode,'gamma':win['gamma'],'beta':win['beta'],'baseline_only':win['baseline_only'],
                            'weights':w.tolist(),'calibration':spec})
    dev.csv_write(root/'tables/VALIDATION_GRID.csv',pd.DataFrame(grid));dev.csv_write(root/'tables/CALIBRATION_AUDIT.csv',pd.DataFrame(audits))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in c.items() if k not in ('weights','calibration')} for c in choices]))
    dev.json_write(root/'calibration/SELECTED.json',choices)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'UTC':datetime.now(timezone.utc).isoformat(),'file':'calibration/SELECTED.json',
        'sha256':dev.sha(root/'calibration/SELECTED.json'),'real_or_test_selection':False})


def scored(root,c,split):
    if c['method']=='OMC':
        f=cf.read(c['deployment'],c['seed'],split);return f,f.waveform_score.to_numpy(float),{}
    f,p,i,x=ev.pair_values(root,c['deployment'],c['kind'],c['slot'],c['seed'],split,c['calibration'])
    return f,selected_score(f,p,i,c),{**x,'penalty':p,'increment':i}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['initialize','select','evaluate','real','assess'],required=True);a=p.parse_args()
    t.predict=predict;ev.scored=scored
    if a.stage in ('initialize','select'):globals()[a.stage](a.root)
    else:getattr(ev,a.stage)(a.root)
