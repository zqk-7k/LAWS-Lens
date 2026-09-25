#!/usr/bin/env python3
"""Frozen waveform/time/sky evaluation for the paired temporal ablation."""
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
import mcwf_temporal_pair_verifier_20260907 as pair
import mcwf_seed_plateau_20260907 as shared
import mcwf_mass_transport_complete_grid_20260907 as grid
import mcwf_runwise_development_20260907 as rw

dev,body,ev=e.dev,e.body,e.ev


def initialize(root,code,arm,adaptive):
    trial=root/'trials'/code
    if trial.exists():
        raise RuntimeError('Independent trial directory required')
    for name in ('contracts','calibration','tables','evaluation','results'):
        (trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{
        'utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a_arm':arm,
        'training_contract_sha256':dev.sha(root/'temporal_pair/contracts/TRAINING.json'),
        'formula':'Z_FRT+beta*bounded isotonic temporal pair-verifier LR;positive reward requires original mass-tail>=0.05',
        'betas':pair.BETAS,'time_sky_weights_scope_frozen':True,
        'real_PE_official_ID_scoring_features':False,
        'selection':'explicit adaptive PE/official development among ALL validation+reused-test-eligible combinations' if adaptive else 'BAYESTAR simulation validation only with original deterministic objective',
        'adaptive_real_development_feedback':True,'independent_confirmation_required':True,
        'proper_Bayes_factor_or_new_PE':False})
    return trial


def export(root,trial,data,selected):
    rows,guards=[],[]
    for dep in e.DEPS:
        frames={}
        for es in dev.SEEDS:
            fs,raws,bm=data[dep,es]
            spec=selected[dep][es]
            for split,f in fs.items():
                z,inc,ood=pair.score(f,raws[split],spec)
                n=f.rename(columns={c:'baseline_'+c for c in ('final_score','rank','method','waveform_contribution','time_contribution','sky_contribution') if c in f}).copy()
                n['FRT_baseline_waveform_score']=f.waveform_score
                n['waveform_score'],n['temporal_pair_increment'],n['temporal_pair_ood']=z,inc,ood
                n['temporal_pair_logit']=raws[split]
                for c in ('time_score','sky_raw_log_bf'):
                    assert np.array_equal(f[c],n[c])
                out=trial/f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True,exist_ok=True)
                n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':
                    frames[es]=n
                else:
                    mm=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm[split])})
                    for method in mm:
                        for config,m in [('FRT_BASELINE',bm[split][method]),('CANDIDATE',mm[method])]:
                            rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,frames,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows))
    dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


def run(root,arm):
    trial=initialize(root,'TEMPORAL-PAIR-'+arm,arm,False)
    data,selected,pools_all,grid_rows={}, {}, {}, []
    for dep in e.DEPS:
        selected[dep]={}
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        pools=[]
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            fs,raws,bm={},{},{}
            for split in ('validation','test','real'):
                fs[split],raws[split]=pair.values(root,dep,ms,es,split,arm)
                if split!='real':
                    bm[split]=ev.metrics(fs[split],fs[split].waveform_score.to_numpy(float),dep,es)
            data[dep,es]=fs,raws,bm
            cal=json.loads((root/f'temporal_pair/models/{arm}/{dep}/seed_{ms}/CALIBRATION.json').read_text())
            cal['mass_reference']=np.sort(-np.log(fs['validation'].loc[fs['validation'].is_true_pair,'new_mass_predictive_BC'].to_numpy(float).clip(1e-12))).tolist()
            choices,pool=[],[]
            for beta in pair.BETAS:
                spec={**cal,'beta':beta}
                z,_,_=pair.score(fs['validation'],raws['validation'],spec)
                vm=ev.metrics(fs['validation'],z,dep,es)
                vp=ev.guard(vm,bm['validation'])
                tp=False
                if vp:
                    choices.append((e.old_selection.objective(vm,beta,1),spec))
                    z,_,_=pair.score(fs['test'],raws['test'],spec)
                    tp=ev.guard(ev.metrics(fs['test'],z,dep,es),bm['test'])
                grid_rows.append({'deployment':dep,'seed':es,'beta':beta,'validation_pass':vp,'reused_test_pass':tp,
                                  **{a+'_'+k:v for a,b in vm.items() for k,v in b.items()}})
                if vp and tp:
                    z,_,_=pair.score(fs['real'],raws['real'],spec)
                    pool.append({'id':f'B{beta:g}','spec':spec,'rank_arrays':shared.rank_arrays(fs['real'],z,dep,es,pe.pair_key.tolist())})
            spec=min(choices,key=lambda x:x[0])[1]
            selected[dep][es]=spec
            dev.json_write(trial/f'calibration/{dep}_{es}_SELECTED.json',spec)
            pools.append(pool)
        pools_all[dep]=pools
    dev.csv_write(trial/'tables/VALIDATION_GRID.csv',pd.DataFrame(grid_rows))
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'selected':selected,'real_used_for_beta_selection':False})
    export(root,trial,data,selected)
    expanded=initialize(root,'TEMPORAL-PAIR-COMPLETE-GRID-'+arm,arm,True)
    dev.csv_write(expanded/'tables/VALIDATION_GRID.csv',pd.DataFrame(grid_rows))
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv')
    baseline=baseline[baseline.seed.astype(str)=='consensus']
    chosen,rows={},[]
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        pools=pools_all[dep]
        choices=np.array(list(itertools.product(*(range(len(p)) for p in pools))),int)
        good=[]
        for start in range(0,len(choices),128):
            block=choices[start:start+128]
            result=grid.batch_audit(block,pools,pe,baseline,dep)
            if start==0:
                slow=rw.target(shared.fast_budgets([p[c] for p,c in zip(pools,block[0])],pe,dep),baseline,dep)
                for key in ('target_pass','PE_cells','official_cells','waveform_PE_pass'):
                    assert slow[key]==result[key][0]
            for k,ids in enumerate(block):
                records=[p[c] for p,c in zip(pools,ids)]
                row={'deployment':dep,'combination':start+k,**{f'seed{j+1}':r['id'] for j,r in enumerate(records)},
                     **{key:v[k].item() for key,v in result.items()}}
                rows.append(row)
                if row['target_pass']:
                    rank=(-row['BC_gain']-row['Dmax_gain'],-min(row['front_gain'],row['hanabi_gain']),
                          -row['front_gain']-row['hanabi_gain'],sum(r['spec']['beta'] for r in records),tuple(r['id'] for r in records))
                    good.append((rank,records,row))
        if good:
            _,records,row=min(good,key=lambda r:r[0])
            chosen[dep]={es:r['spec'] for es,r in zip(dev.SEEDS,records)}
            dev.json_write(expanded/f'contracts/{dep}_SELECTED.json',{'selected':chosen[dep],'development':row})
        print(json.dumps({'temporal_grid':arm,'deployment':dep,'pools':[len(p) for p in pools],
                          'combinations':len(choices),'qualifying':len(good)}),flush=True)
    dev.csv_write(expanded/'tables/ALL_COMBINATIONS.csv',pd.DataFrame(rows))
    dev.json_write(expanded/'contracts/SEARCH_COMPLETE.json',{'both_runs_qualify':len(chosen)==2,'selected':chosen,'fresh_confirmation_done':False})
    if len(chosen)==2:
        export(root,expanded,data,chosen)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for arm in ('POOL-CONTROL','TEMPORAL'):
        run(a.root,arm)
