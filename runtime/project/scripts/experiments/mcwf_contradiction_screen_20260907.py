#!/usr/bin/env python3
"""Screen new waveform rewards at the existing five-percent mass-conflict rule."""
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
import mcwf_mixture_evaluate_20260907 as mdn
import mcwf_concordant_dense_20260907 as dense
import mcwf_finite_reference_tail_20260907 as tail
import mcwf_seed_plateau_20260907 as shared
import mcwf_runwise_development_20260907 as rw

dev,body,ev=e.dev,e.body,e.ev
BETAS=(0.,.0625,.125,.25,.5,1.,2.,4.)


def get(root,dep,ms,es,split,feature):
    family,kind=feature.split(':')
    if family=='MDN':
        f,x=mdn.get(root,dep,ms,es,split);cal={**mdn.fit(root,dep,ms),'kind':kind,'beta':1.}
        _,inc,ood=mdn.score(f,x,cal)
    else:
        f,x=dense.get(root,dep,ms,es,split,kind)
        cal=json.loads((root/f'dense_profile/{dep}/FIT.json').read_text())[kind]
        value=x['profile'];raw=np.interp(value,cal['knots'],cal['loglr'])
        ood=(value<cal['fit_min'])|(value>cal['fit_max'])
        inc=np.clip(np.where(ood,np.minimum(raw,0),raw),-4,4)
    return f,{'inc':inc,'ood':ood,'old_mass':f.waveform_abs_delta_logmc_std.to_numpy(float),
        'new_mass':-np.log(f.new_mass_predictive_BC.to_numpy(float).clip(1e-12))},cal


def score(f,x,spec):
    a=tail.tail_probability(x['old_mass'],spec['old_reference']);b=tail.tail_probability(x['new_mass'],spec['new_reference'])
    gate=(a>=.05)&(b>=.05)
    inc=np.maximum(x['inc'],0)*gate
    if spec['mode']=='full':inc+=np.minimum(x['inc'],0)
    return f.waveform_score.to_numpy(float)+spec['beta']*inc,inc,gate


def run(root,feature,mode):
    code=f'CONTRADICTION-SCREEN-{feature.replace(":","-").upper()}-{mode.upper()}'
    trial=root/'trials'/code
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','configs','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_global_recipe_O3_O4a':[feature,mode],
        'formula':'Z_FRT+beta*(positive_increment*no_mass_conflict + optionalnegative_increment)',
        'no_mass_conflict':'oldpointMcgap and RNCpredictiveMcBC finite-reference tail ranks BOTH>=0.05',
        'fixed_alpha':.05,'alpha_source':'unchangedoriginalFRTfivepercentcontradictionlevel;notselectedonrealdata',
        'rationale':'absenceofdetectedconflict permitsboundednewreward;tailrankisNOT aprobabilityofphysicalagreement,so it neednotattenuateeverycompatiblepaircontinuously',
        'beta_grid':BETAS,'negative_retained':mode=='full','old_FRT_penalty_always_retained':True,
        'simulation_selection':'all originalvalidation/reused-testguards;all eligiblebetas retained',
        'real_feedback':'explicitadaptiveDEVELOPMENTselection underfrozengoals,notblindvalidation',
        'no_PE_official_ID_features':True,'frozen':['time','sky','outerweights','scope','targets','historicaloutputs'],
        'fresh_confirmation_required':True,'no_conformal_FPP_claim':True})
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv');baseline=baseline[baseline.seed.astype(str)=='consensus']
    selected,data,rows,combos={},{},[],[]
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True);keys=pe.pair_key.to_list();pools=[]
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            fs,xs,bm={},{},{}
            for split in ('validation','test','real'):
                fs[split],xs[split],cal=get(root,dep,ms,es,split,feature)
                if split!='real':bm[split]=ev.metrics(fs[split],fs[split].waveform_score.to_numpy(float),dep,es)
            data[dep,es]=fs,xs,bm;y=fs['validation'].is_true_pair.to_numpy(bool)
            ref={'feature':feature,'mode':mode,'calibration':cal,
                'old_reference':np.sort(xs['validation']['old_mass'][y]).tolist(),'new_reference':np.sort(xs['validation']['new_mass'][y]).tolist()}
            pool=[]
            for beta in BETAS:
                spec={**ref,'beta':beta};z,*_=score(fs['validation'],xs['validation'],spec)
                vm=ev.metrics(fs['validation'],z,dep,es);vp=ev.guard(vm,bm['validation']);tp=False
                if vp:
                    z,*_=score(fs['test'],xs['test'],spec);tp=ev.guard(ev.metrics(fs['test'],z,dep,es),bm['test'])
                rows.append({'deployment':dep,'seed':es,'beta':beta,'validation_pass':vp,'reused_test_pass':tp,
                    **{a+'_'+k:v for a,b in vm.items() for k,v in b.items()}})
                if vp and tp:
                    z,*_=score(fs['real'],xs['real'],spec)
                    pool.append({'id':f'B{beta:g}','spec':spec,'rank_arrays':shared.rank_arrays(fs['real'],z,dep,es,keys)})
            dev.json_write(trial/f'configs/{dep}_{es}_ELIGIBLE.json',{'configs':[c['spec'] for c in pool]});pools.append(pool)
        dev.csv_write(trial/'tables/VALIDATION_GRID.csv',pd.DataFrame(rows));qualified=[]
        for ci,combo in enumerate(itertools.product(*pools)):
            result=rw.target(shared.fast_budgets(combo,pe,dep),baseline,dep);ids=[c['id'] for c in combo]
            combos.append({'deployment':dep,'combination':ci,'seed1':ids[0],'seed2':ids[1],'seed3':ids[2],**{k:v for k,v in result.items() if k!='details'}})
            if result['target_pass']:
                key=(-result['BC_gain']-result['Dmax_gain'],-min(min(c['front_delta'],c['hanabi_delta']) for c in result['details']),
                    -result['front_gain']-result['hanabi_gain'],sum(c['spec']['beta'] for c in combo),ids)
                qualified.append((key,combo,result))
        dev.csv_write(trial/'tables/ALL_COMBINATIONS.csv',pd.DataFrame(combos))
        if qualified:
            _,combo,result=min(qualified,key=lambda x:x[0]);selected[dep]={es:c['spec'] for es,c in zip(dev.SEEDS,combo)}
            dev.json_write(trial/f'contracts/{dep}_SELECTED.json',{'configs':selected[dep],'development_target':result,'qualifying':len(qualified)})
        print(json.dumps({'contradiction_screen':code,'deployment':dep,'pools':[len(p) for p in pools],'qualifying':len(qualified)}),flush=True)
    dev.json_write(trial/'contracts/SEARCH_COMPLETE.json',{'both_runs_qualify':len(selected)==2,'selected':selected,'fresh_confirmation_done':False})
    if len(selected)!=2:return
    metrics,guards=[],[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            fs,xs,bm=data[dep,es]
            for split,f in fs.items():
                z,inc,gate=score(f,xs[split],selected[dep][es]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score
                n['waveform_score']=z;n['screened_increment']=inc;n['no_predicted_mass_conflict']=gate
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    mm=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm[split])})
                    for method in mm:
                        for config,m in [('FRT_BASELINE',bm[split][method]),('CANDIDATE',mm[method])]:metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for feature in ('MDN:full','MDN:conditional','DENSE:full','DENSE:conditional_q_spin'):
        for mode in ('full','positive_only'):run(a.root,feature,mode)
