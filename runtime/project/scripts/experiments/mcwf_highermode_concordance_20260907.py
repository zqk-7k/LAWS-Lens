#!/usr/bin/env python3
"""Finite development search for mass-concordant intrinsic density increments."""
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
import mcwf_highermode_evaluate_20260907 as mix
import mcwf_finite_reference_tail_20260907 as tail
import mcwf_seed_plateau_20260907 as shared
import mcwf_runwise_development_20260907 as rw

dev,body,ev=e.dev,e.body,e.ev
BETAS=(0.,.0625,.125,.25,.5,1.,2.,4.)
POWERS=(1.,2.,4.)


def get(root,dep,ms,es,split):
    f,x=mix.get(root,dep,ms,es,split)
    return f,{**x,'old_mass':f.waveform_abs_delta_logmc_std.to_numpy(float),
        'RNC_mass':-np.log(f.new_mass_predictive_BC.to_numpy(float).clip(1e-12))}


def score(f,x,spec):
    _,inc,ood=mix.score(f,x,{**spec,'beta':1.})
    a=tail.tail_probability(x['old_mass'],spec['old_mass_reference'])
    b=tail.tail_probability(x['RNC_mass'],spec['RNC_mass_reference'])
    c=tail.tail_probability(-x['mass'],spec['highermode_mass_reference'])
    gate=np.minimum(np.minimum(a,b),c)**spec['power']
    weighted=inc*gate if spec['gate_mode']=='both_signs' else np.minimum(inc,0)+np.maximum(inc,0)*gate
    return f.waveform_score.to_numpy(float)+spec['beta']*weighted,weighted,gate,ood


def run(root,kind,mode):
    code=f'HIGHER-MODE-CONCORDANCE-{kind.upper()}-{mode.upper()}'
    trial=root/'trials'/code
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','configs','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_global_kind_and_gate_for_O3_O4a':[kind,mode],
        'formula':'Z_FRT+beta*MDN_increment*mass_gate;positive_only variant attenuates only positive increment',
        'mass_gate':'min(p_old_Mc_gap_tail,p_RNC_Mc_BC_tail,p_XPHM_predictive_Mc_product_tail)^power;each finite reference has1companionpair/source',
        'rationale':'new q/spin evidence must not overwhelm a waveform mass disagreement already identified by the frozen models',
        'beta_grid':BETAS,'power_grid':POWERS,
        'simulation_plateau':'baseline+4 best validation points per power passing all frozen reused-test guards;maximum13perseed',
        'real_feedback':'explicitly ADAPTIVE DEVELOPMENT budget selection within simulation-eligible plateau,notvalidation-only and notblindtest',
        'tie_order':['PEcountgains','minimumofficialgain','totalofficialgain','minimumcoefficientchange','fixedlexical'],
        'no_real_or_official_features':True,'no_eventID_features':True,'fresh_confirmation_required':True,
        'frozen':['time','sky','outerweights','scope','thresholds','historicalresults']})
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv')
    baseline=baseline[baseline.seed.astype(str)=='consensus']
    selected,data,rows,combinations={},{},[],[]
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        keys=pe.pair_key.to_list();pools=[]
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            fs,xs,bm={},{},{}
            for split in ('validation','test','real'):
                fs[split],xs[split]=get(root,dep,ms,es,split)
                if split!='real':bm[split]=ev.metrics(fs[split],fs[split].waveform_score.to_numpy(float),dep,es)
            data[dep,es]=fs,xs,bm
            y=fs['validation'].is_true_pair.to_numpy(bool)
            ref={**mix.fit(root,dep,ms),'kind':kind,'gate_mode':mode,
                'old_mass_reference':np.sort(xs['validation']['old_mass'][y]).tolist(),
                'RNC_mass_reference':np.sort(xs['validation']['RNC_mass'][y]).tolist()}
            reference=np.load(root/f'highermode_density/calibration/{dep}/seed_{ms}/fit.npz')
            ref['highermode_mass_reference']=np.sort(-reference['mass'][reference['y']]).tolist()
            eligible=[]
            for power in POWERS:
                for beta in BETAS:
                    if beta==0 and power!=POWERS[0]:continue
                    spec={**ref,'power':power,'beta':beta};z,*_=score(fs['validation'],xs['validation'],spec)
                    vm=ev.metrics(fs['validation'],z,dep,es);vp=ev.guard(vm,bm['validation']);tp=False
                    if vp:
                        z,*_=score(fs['test'],xs['test'],spec)
                        tp=ev.guard(ev.metrics(fs['test'],z,dep,es),bm['test'])
                    ident=f'P{power:g}_B{beta:g}'
                    rows.append({'deployment':dep,'seed':es,'id':ident,'power':power,'beta':beta,
                        'validation_pass':vp,'reused_test_pass':tp,
                        **{a+'_'+k:v for a,b in vm.items() for k,v in b.items()}})
                    if vp and tp:eligible.append({'id':ident,'spec':spec,'objective':e.old_selection.objective(vm,beta,power)})
            eligible.sort(key=lambda c:(c['objective'],c['id']))
            pool=[next(c for c in eligible if c['spec']['beta']==0)]
            for power in POWERS:pool.extend([c for c in eligible if c['spec']['power']==power and c['spec']['beta']>0][:4])
            dev.json_write(trial/f'configs/{dep}_{es}_PLATEAU.json',{'eligible':len(eligible),'retained':pool})
            for c in pool:
                z,*_=score(fs['real'],xs['real'],c['spec'])
                c['rank_arrays']=shared.rank_arrays(fs['real'],z,dep,es,keys)
            pools.append(pool)
        dev.csv_write(trial/'tables/VALIDATION_GRID.csv',pd.DataFrame(rows))
        qualified=[]
        for ci,combo in enumerate(itertools.product(*pools)):
            budgets=shared.fast_budgets(combo,pe,dep);result=rw.target(budgets,baseline,dep)
            ids=[c['id'] for c in combo]
            combinations.append({'deployment':dep,'combination':ci,'seed1':ids[0],'seed2':ids[1],'seed3':ids[2],
                **{k:v for k,v in result.items() if k!='details'}})
            if result['target_pass']:
                balance=min(min(c['front_delta'],c['hanabi_delta']) for c in result['details'])
                key=(-result['BC_gain']-result['Dmax_gain'],-balance,-result['front_gain']-result['hanabi_gain'],
                    sum(c['spec']['beta'] for c in combo),ids)
                qualified.append((key,combo,result))
        dev.csv_write(trial/'tables/ALL_COMBINATIONS.csv',pd.DataFrame(combinations))
        if qualified:
            _,combo,result=min(qualified,key=lambda x:x[0])
            selected[dep]={es:c['spec'] for es,c in zip(dev.SEEDS,combo)}
            dev.json_write(trial/f'contracts/{dep}_SELECTED.json',{'configs':selected[dep],'development_target':result,'qualifying':len(qualified)})
        print(json.dumps({'highermode_concordance':code,'deployment':dep,'pools':[len(p) for p in pools],
            'qualifying':len(qualified)}),flush=True)
    dev.json_write(trial/'contracts/SEARCH_COMPLETE.json',{'both_runs_qualify':len(selected)==2,'selected':selected,
        'fresh_confirmation_done':False})
    if len(selected)!=2:return
    metrics,guards=[],[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            fs,xs,bm=data[dep,es]
            for split,f in fs.items():
                z,inc,gate,ood=score(f,xs[split],selected[dep][es]);n=f.copy()
                n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z
                n['mixture_increment']=inc;n['mass_concordance_gate']=gate;n['mixture_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True)
                n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    mm=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm[split])})
                    for method in mm:
                        for config,m in [('FRT_BASELINE',bm[split][method]),('CANDIDATE',mm[method])]:
                            metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for kind in mix.KINDS:
        for mode in ('positive_only','both_signs'):run(a.root,kind,mode)
