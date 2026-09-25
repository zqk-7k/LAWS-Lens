#!/usr/bin/env python3
"""Finite seed-wise validation plateau, followed by declared development selection."""
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
import mcwf_global_development_20260907 as g
import mcwf_runwise_development_20260907 as rw

dev,body,ev=e.dev,e.body,e.ev


def rank_arrays(f,z,dep,es,keys):
    changed=f.copy();changed['waveform_score']=z
    out={}
    for method in ('waveform_only','C_fixed'):
        w={'waveform':1.,'time':0.,'sky':0.} if method=='waveform_only' else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
        ranked=dev.BASE.rank_real(changed,w,method,es).set_index('pair_key').loc[keys]
        out[method]=(ranked['rank'].to_numpy(),ranked.final_score.to_numpy())
    return out


def fast_budgets(combination,pe,dep):
    bc=pe.pe_mc_bhattacharyya_coefficient.to_numpy(float)
    d=pe.pe_mc_standardized_distance.to_numpy(float)
    dm=pe.pe_dmax_intrinsic.to_numpy(float)
    front=pe['official_po_or_ml_fpp_below_0p01' if dep=='gwtc3' else 'official_po_or_phazap_fpp_below_0p01'].fillna(False).to_numpy(bool)
    hanabi=pe.official_any_pair_resolved_hanabi_overlap.fillna(False).to_numpy(bool)
    budgets=[]
    for method in ('waveform_only','C_fixed'):
        ranks=np.stack([c['rank_arrays'][method][0] for c in combination])
        scores=np.stack([c['rank_arrays'][method][1] for c in combination])
        ix=np.lexsort((np.arange(len(pe)),-scores.mean(0),ranks.max(0),ranks.mean(0)))
        for b in (10,20):
            k=ix[:b]
            budgets.append({'deployment':dep,'method':method,'budget':b,'catastrophic_mc':int(((bc[k]<.1)|(d[k]>5)).sum()),
                'BC_mc_ge_0p5':int((bc[k]>=.5).sum()),'median_BC_mc':float(np.median(bc[k])),
                'Dmax_le_3':int((dm[k]<=3).sum()),'official_frontend':int(front[k].sum()),'official_hanabi':int(hanabi[k].sum())})
    return pd.DataFrame(budgets)


def run(root):
    trial=root/'trials/SEED-PLATEAU-DEVELOPMENT'
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','configs','tables','evaluation','results'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/ADAPTIVE_DEVELOPMENT_ADDENDUM.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4_method':True,
        'formula':'same FRT plus bounded REF/COS increments as preceding experiments',
        'difference':'restore separate numerical operating points for each independent training seed, using exactly the same admissible grid and plateau definition for each seed and run',
        'plateau_definition':'retain unchanged FRT plus at most8 best remaining validation-objective configurations that also pass reused-development test guards; no real outcomes used to build plateau',
        'seed_grid':{'tail_scale':g.SCALES,'ref_weight':g.REF,'cos_weight':[0.]},
        'maximum_combinations_per_run':729,
        'final_selection':'explicit real-development target check across all finite plateau combinations, then PE count gains, balanced official gains, total official gains, minimum coefficient change, deterministic lexical ties',
        'not_validation_only':True,'not_blind_real_test':True,'no_real_PE_or_official_values_in_scores':True,
        'selection_multiplicity':'all configurations and all combinations reported; real data are adaptive development; no real-dataset significance claim',
        'unchanged_targets':str(root/'contracts/EXPERIMENT_CONTRACT.json'),
        'fresh_confirmation_required':True,'old_time_sky_outer_scope_unchanged':True})
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv');baseline=baseline[baseline.seed.astype(str)=='consensus']
    allselected={};data={};allrows=[];combo_rows=[]
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        keys=pe.pair_key.to_list();prior=e.reference_prior(dep);pools=[]
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            specs={r.split('-')[0]:e.fit_reference(dep,ms,r,prior) for r in ('REF-RAW','COS-ISO')}
            dev.json_write(trial/f'configs/{dep}_{es}_simulation_calibration.json',specs)
            frames={};features={};bm={}
            for split in ('validation','test','real'):
                f,x=e.frame_features(dep,ms,es,split,prior);frames[split]=f;features[split]={k:e.increment(x,s)[0] for k,s in specs.items()}
                if split!='real':bm[split]=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            data[dep,es]=(frames,features,bm)
            candidates=[]
            for a in g.SCALES:
                for b in g.REF:
                    spec={'tail_scale':a,'ref_weight':b,'cos_weight':0.};ident=f'T{a:g}_R{b:g}'
                    vm=ev.metrics(frames['validation'],g.score(frames['validation'],features['validation'],spec),dep,es)
                    vp=ev.guard(vm,bm['validation']);tp=False
                    if vp:
                        tm=ev.metrics(frames['test'],g.score(frames['test'],features['test'],spec),dep,es);tp=ev.guard(tm,bm['test'])
                    allrows.append({'deployment':dep,'seed':es,'id':ident,**spec,'validation_pass':vp,'test_pass':tp,
                                    **{m+'_'+k:v for m,mm in vm.items() for k,v in mm.items()}})
                    if vp and tp:
                        candidates.append({'id':ident,'spec':spec,'objective':e.old_selection.objective(vm,abs(a-1)+b,b)})
            candidates.sort(key=lambda c:(c['objective'],c['id']))
            unchanged=next(c for c in candidates if c['id']=='T1_R0')
            pool=[unchanged]+[c for c in candidates if c['id']!='T1_R0'][:8]
            dev.json_write(trial/f'configs/{dep}_{es}_PLATEAU.json',{'eligible':len(candidates),'retained':pool,'real_used_to_build_plateau':False})
            for c in pool:
                z=g.score(frames['real'],features['real'],c['spec'])
                c['rank_arrays']=rank_arrays(frames['real'],z,dep,es,keys)
            pools.append(pool)
        dev.csv_write(trial/'tables/PER_SEED_VALIDATION_GRID.csv',pd.DataFrame(allrows))
        qualified=[]
        for ci,combination in enumerate(itertools.product(*pools)):
            budgets=fast_budgets(combination,pe,dep);t=rw.target(budgets,baseline,dep)
            ids=[c['id'] for c in combination]
            combo_rows.append({'deployment':dep,'combination':ci,'seed1':ids[0],'seed2':ids[1],'seed3':ids[2],
                               **{k:v for k,v in t.items() if k!='details'}})
            if t['target_pass']:
                balanced=min(min(c['front_delta'],c['hanabi_delta']) for c in t['details'])
                deviation=sum(abs(c['spec']['tail_scale']-1)+c['spec']['ref_weight'] for c in combination)
                key=(-t['BC_gain']-t['Dmax_gain'],-balanced,-t['front_gain']-t['hanabi_gain'],deviation,ids)
                qualified.append((key,combination,t))
        dev.csv_write(trial/'tables/ALL_COMBINATIONS.csv',pd.DataFrame(combo_rows))
        if qualified:
            _,combo,t=min(qualified,key=lambda c:c[0])
            allselected[dep]={es:c['spec'] for es,c in zip(dev.SEEDS,combo)}
            dev.json_write(trial/f'contracts/{dep}_SELECTED.json',{'configs':allselected[dep],'real_development_target':t,'qualifying':len(qualified)})
        print(json.dumps({'seed_plateau':dep,'pools':[len(p) for p in pools],'qualifying':len(qualified),'selected':allselected.get(dep)}),flush=True)
    dev.json_write(trial/'contracts/SEARCH_COMPLETE.json',{'both_runs_qualify':len(allselected)==2,'selected':allselected,'fresh_confirmation_done':False})
    if len(allselected)!=2:return
    metrics=[];guards=[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            fs,x,bm=data[dep,es]
            for split,f in fs.items():
                z=g.score(f,x[split],allselected[dep][es]);changed=f.copy();changed['FRT_baseline_waveform_score']=f.waveform_score;changed['waveform_score']=z
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);changed.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=changed
                else:
                    mm=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm[split])})
                    for method in mm:
                        for conf,m in [('FRT_BASELINE',bm[split][method]),('CANDIDATE',mm[method])]:
                            metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':conf,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
