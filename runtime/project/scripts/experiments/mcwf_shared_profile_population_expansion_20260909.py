#!/usr/bin/env python3
"""Bounded R56 simulation-only expansion of shared-profile calibration."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as physical
import mcwf_shared_profile_sampling_audit_20260909 as sampling
import mcwf_shared_joint_feature_pilot_20260909 as classifier
import mcwf_shared_profile_monotone_boundary_20260909 as boundary
n=physical.n
RANDOM=128
HARD=64
ROOT=DATA=PILOT=REFERENCE=None


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent R56 output required')
    for name in ('contracts','configs','cache','pairs','tables','audit','scripts','logs','manifest','reports'):
        (ROOT/name).mkdir(parents=True)
    contract={'UTC':n.utc(),'id':'MCWF-SHARED-PROFILE-POPULATION-EXPANSION-56',
        'status':n.STATUS,'goal_achieved':False,'adaptive_development':True,
        'data':str(DATA),'reuse_physical_pilot':str(PILOT),'reference':str(REFERENCE),
        'hypothesis':'R50 uses only24 positive systems per run/fold. Census the existing quality-qualified positives and enlarge negative calibration without increasing model complexity.',
        'sampling':{'true':'all existing quality-qualified source doublets',
            'random_null':RANDOM,'hard_null':HARD,'hash_seed':physical.SEED,
            'design':'same hash permutations as R50; expanded random prefix then hard prefix excluding random. This is not a new independent data realization.',
            'neighbors':'8 nearest frozen profile-logMc neighbors per event, excluding shared source/noise parent'},
        'weights':'Null inverse exact two-stage inclusion probability divided by full quality-qualified source-pair image multiplicity; true census weight1. Each source pair has total weight1 BEFORE additional deployed-power eligibility. Subsequent conditioning targets its eligible image mass, not a new equal-source conditional population.',
        'fit':'fold0 only, finite/converged and inside frozen R51 power eligibility; one monotone logistic classifier with cosine/logminpower/-log1pD, HT class-balanced loss',
        'selection':'fold1 HT class-balanced deployed-policy proper logloss; ridge .001/.01/.1/1, ties stronger ridge; no catalog or real outcomes',
        'deployment_policy':'Exactly R55 low-D saturation, fixed R51 power/deficit supports, caps and outer weights. No new support or positive cap selected.',
        'gate':'Mean tune loss across3encoders nonworse than frozen R55 in full HT population, random-draw source-pair diagnostic AND neighbor-population source-pair diagnostic separately in BOTH runs; strict gain in at least1. No selecting an O3-only or O4a-only winner.',
        'on_fail':'Retain all results; no catalog or real scoring for this calibration.',
        'unchanged':['all encoders','time','sky','outerweights','event scope','source/noise folds','all historical results'],
        'no_old_Mc_q_score_or_total_blend':True,'no_real_PE_or_official_inputs':True,
        'no_full_PE_or_Bayes_factor_claim':True,
        'independence_limits':'R22B source parents and noise-parent folds are isolated; GW-LMC lens environments reused, per-image SNR target scaling retained. Not a new lens population or new locked confirmation.',
        'resources':'24 CPU workers, completed exact physical pairs reused; do not regenerate or train encoders',
        'references':['https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446','https://arxiv.org/abs/2104.09339'],
        'reference_limits':'Unequal-probability weights and shared-source hypothesis motivate the design; neither proves this surrogate optimal or its score a physical Bayes factor.'}
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',contract)
    n.write_json(ROOT/'audit/INCLUSION_ENUMERATION_UNIT.json',sampling.unit())
    physical.predictive.DATA=DATA
    plans, inventories, inputs=[],[],[]
    configs=json.loads((REFERENCE/'configs/SELECTED_CONFIGURATIONS.json').read_text())
    configs=[r for r in configs if r['method']==boundary.NEW]
    n.write_json(ROOT/'configs/FROZEN_REFERENCE_CONFIGS.json',configs)
    inputs.extend([REFERENCE/'configs/SELECTED_CONFIGURATIONS.json',Path(physical.__file__),
        Path(sampling.__file__),Path(classifier.__file__),Path(classifier.base.__file__),Path(boundary.__file__)])
    for dep in n.DEPS:
        meta,_,_,_,active,_,_,_=physical.predictive.inputs(dep)
        noise_path=DATA/f'data/{dep}/noise/noise_manifest.csv'
        noise=pd.read_csv(noise_path).set_index('bank_index')
        chunk=meta.noise_bank_index.map(noise.parent_file_gps)
        if set(meta.source_uid[meta.fold==0])&set(meta.source_uid[meta.fold==1]) or set(chunk[meta.fold==0])&set(chunk[meta.fold==1]):
            raise RuntimeError('Global source or noise-parent leak')
        profiles={int(p.stem):json.loads(p.read_text()) for p in (DATA/f'profile_events/{dep}').glob('*.json')}
        selected=set()
        for fold in (0,1):
            ids=np.flatnonzero(active&meta.fold.eq(fold).to_numpy())
            truth=[]
            for uid,g in meta.iloc[ids].groupby('source_uid'):
                if len(g)==2:
                    i,j=sorted(g.index.to_list())
                    truth.append((physical.digest(f'{physical.SEED}/{dep}/{fold}/true/{uid}'),i,j))
            def source_key(i,j):
                return tuple(sorted((str(meta.source_uid.iloc[i]),str(meta.source_uid.iloc[j]))))
            population=[(int(i),int(j)) for pos,i in enumerate(ids) for j in ids[pos+1:]
                if meta.source_uid.iloc[i]!=meta.source_uid.iloc[j] and chunk.iloc[i]!=chunk.iloc[j]]
            multiplicity=Counter(source_key(i,j) for i,j in population)
            random=sorted((physical.digest(f'{physical.SEED}/{dep}/{fold}/null/{i}/{j}'),i,j) for i,j in population)[:RANDOM]
            random_set={(i,j) for _,i,j in random}
            neighbors=set()
            for i in ids:
                nearby=sorted((abs(profiles[i]['logmc']-profiles[j]['logmc']),int(j)) for j in ids
                    if j!=i and meta.source_uid.iloc[i]!=meta.source_uid.iloc[j] and chunk.iloc[i]!=chunk.iloc[j])
                for _,j in nearby[:8]:
                    neighbors.add(tuple(sorted((int(i),j))))
            hard=sorted((physical.digest(f'{physical.SEED}/{dep}/{fold}/hard/{i}/{j}'),i,j)
                for i,j in neighbors-random_set)[:HARD]
            if len(random)!=RANDOM or len(hard)!=HARD:
                raise RuntimeError('Insufficient fixed negative sample support')
            pi_n,pi_h=sampling.inclusion(len(population),len(neighbors),RANDOM,HARD)
            total=sum(1/multiplicity[source_key(i,j)] for i,j in population)
            if not np.isclose(total,len(multiplicity),rtol=0,atol=1e-8):
                raise RuntimeError('Source-pair population total failed')
            for kind,group in [('true',sorted(truth)),('random_null',random),('hard_null',hard)]:
                for digest,i,j in group:
                    neighbor=(i,j) in neighbors
                    mult=1 if kind=='true' else multiplicity[source_key(i,j)]
                    pi=1. if kind=='true' else pi_h if neighbor else pi_n
                    plans.append({'deployment':dep,'fold':fold,'kind':kind,'idx_i':i,'idx_j':j,
                        'pair_id':f'{dep}_{fold}_{i}_{j}','selection_hash':digest,
                        'source_i':meta.source_uid.iloc[i],'source_j':meta.source_uid.iloc[j],
                        'noise_parent_i':chunk.iloc[i],'noise_parent_j':chunk.iloc[j],
                        'neighbor_population':neighbor,'source_pair_multiplicity':mult,
                        'inclusion_probability':pi,'source_pair_weight':1/mult,'HT_weight':1/(mult*pi),
                        'profile_logMc_gap':abs(profiles[i]['logmc']-profiles[j]['logmc'])})
                    selected.update((i,j))
            inventories.append({'deployment':dep,'fold':fold,'active_events':len(ids),
                'true_source_census':len(truth),'random_null':len(random),'hard_null':len(hard),
                'null_population':len(population),'neighbor_population':len(neighbors),
                'source_pair_population':len(multiplicity),'pi_n':pi_n,'pi_h':pi_h})
        meta.loc[sorted(selected)].to_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet',index=False)
        inputs.extend([DATA/f'data/{dep}/event_metadata.parquet',noise_path,
            DATA/f'predictions/{dep}/ensemble_mass.npy'])
        inputs.extend(DATA/f'predictions/{dep}/{seed}_embedding.npy' for seed in n.SEEDS)
        inputs.extend(DATA/f'profile_events/{dep}/{i}.json' for i in selected)
    plan=pd.DataFrame(plans)
    if plan.pair_id.duplicated().any():
        raise RuntimeError('Duplicate calibration pair')
    plan.to_parquet(ROOT/'contracts/PAIR_PLAN.parquet',index=False)
    reused=[]
    for row in plan.itertuples():
        source=PILOT/f'pairs/{row.pair_id}.json'
        if source.exists():
            record=json.loads(source.read_text())
            if record['status']!='COMPLETE':
                raise RuntimeError('Cannot reuse failed physical pair')
            target=ROOT/f'pairs/{row.pair_id}.json'
            shutil.copy2(source,target)
            if n.sha(source)!=n.sha(target):
                raise RuntimeError('Physical result reuse hash mismatch')
            inputs.append(source)
            reused.append({'source':str(source),'target':str(target),'sha256':n.sha(source)})
    for dep in n.DEPS:
        for row in pd.read_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet').itertuples():
            source=PILOT/f'cache/{dep}/{row.row_index}.npz'
            if source.exists():
                receipt=json.loads(source.with_suffix('.json').read_text())
                if receipt['sha256']!=n.sha(source):
                    raise RuntimeError('Existing waveform cache corrupt')
                target=ROOT/f'cache/{dep}/{row.row_index}.npz'
                target.parent.mkdir(exist_ok=True,parents=True)
                shutil.copy2(source,target)
                n.write_json(target.with_suffix('.json'),{**receipt,'path':str(target),'reused_from':str(source)})
                inputs.append(source)
    n.write_csv(ROOT/'tables/PLANNING_INVENTORY.csv',inventories)
    n.write_csv(ROOT/'audit/REUSED_PHYSICAL_MEASUREMENTS.csv',reused)
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p)} for p in sorted(set(inputs))])
    shutil.copy2(__file__,ROOT/'scripts/shared_profile_population_expansion.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'runtime_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'pair_plan_sha256':n.sha(ROOT/'contracts/PAIR_PLAN.parquet'),
        'reference_configs_sha256':n.sha(ROOT/'configs/FROZEN_REFERENCE_CONFIGS.json')})
    print(pd.DataFrame(inventories).to_string(index=False),flush=True)
    print('EXPANSION_FROZEN',len(plan),'reused',len(reused),'new',len(plan)-len(reused),flush=True)


def check():
    r=json.loads((ROOT/'contracts/START_FREEZE.json').read_text())
    paths={'runtime_sha256':Path(__file__),'contract_sha256':ROOT/'contracts/ANALYSIS_CONTRACT.json',
        'pair_plan_sha256':ROOT/'contracts/PAIR_PLAN.parquet',
        'reference_configs_sha256':ROOT/'configs/FROZEN_REFERENCE_CONFIGS.json'}
    if any(n.sha(path)!=r[name] for name,path in paths.items()):
        raise RuntimeError('Frozen expansion protocol changed')
    physical.ROOT,physical.DATA=ROOT,DATA


def fit():
    if (ROOT/'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Do not overwrite fitted pilot')
    receipt=json.loads((ROOT/'contracts/MEASUREMENT_COMPLETE.json').read_text())
    if receipt['failures']:
        raise RuntimeError('All planned physical measurements must complete')
    pairs=pd.read_parquet(ROOT/'tables/PAIR_RESULTS.parquet')
    references={(r['deployment'],r['seed']):r for r in json.loads((ROOT/'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    rows,grid,predictions,selected=[],[],[],{}
    for dep in n.DEPS:
        f=pairs[pairs.deployment==dep].reset_index(drop=True)
        ref=references[dep,n.SEEDS[0]]
        mask=f.minimum_independent_power.between(*ref['minimum_power_support'])&f.any_shared_converged
        f=f[mask].reset_index(drop=True)
        train=f.fold.eq(0).to_numpy(); tune=f.fold.eq(1).to_numpy()
        y=f.kind.eq('true').to_numpy(float)
        fitw=classifier.base.balanced(y[train],f.HT_weight.to_numpy()[train])
        tunew=classifier.base.balanced(y[tune],f.HT_weight.to_numpy()[tune])
        for seed in n.SEEDS:
            ref=references[dep,seed]
            e=np.load(DATA/f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            e/=np.linalg.norm(e,axis=1,keepdims=True)
            cosine=np.einsum('ij,ij->i',e[f.idx_i],e[f.idx_j])
            x=np.column_stack([cosine,np.log(f.minimum_independent_power),-np.log1p(f.deficit)])
            options=[]
            for ridge in classifier.base.RIDGES:
                spec=classifier.fit(x[train],y[train],fitw,ridge)
                config={**ref,'shared_classifier':spec}
                z=boundary.predict_policy(x,config,True)
                loss=classifier.base.loss(y[tune],z[tune],tunew)
                options.append((loss,-ridge,config))
                grid.append({'deployment':dep,'seed':seed,'ridge':ridge,'tune_HT_policy_logloss':loss})
            winner=min(options,key=lambda a:(a[0],a[1]))[2]
            selected[f'{dep}/{seed}']=winner
            masks={'full_population':tune,'random_draw':tune&f.kind.ne('hard_null').to_numpy(),
                'neighbor_population':tune&(f.neighbor_population.to_numpy()|(y==1))}
            for arm,config in [('R55-FROZEN',ref),('EXPANDED-HT-PROFILE',winner)]:
                z=boundary.predict_policy(x,config,True)
                for diagnostic,take in masks.items():
                    weights=f.HT_weight.to_numpy() if diagnostic=='full_population' else f.source_pair_weight.to_numpy()
                    w=classifier.base.balanced(y[take],weights[take])
                    rows.append({'deployment':dep,'seed':seed,'arm':arm,'diagnostic':diagnostic,
                        'logloss':classifier.base.loss(y[take],z[take],w),'pairs':int(take.sum()),
                        'positive_sources':int(y[take].sum())})
                out=f[['pair_id','deployment','fold','kind','source_i','source_j','neighbor_population',
                    'HT_weight','source_pair_weight','deficit','minimum_independent_power']].copy()
                out['seed']=seed;out['arm']=arm;out['waveform_score']=z
                predictions.extend(out.to_dict('records'))
            print('EXPANSION_FIT',dep,seed,flush=True)
    metric=pd.DataFrame(rows)
    summary=metric.groupby(['deployment','arm','diagnostic']).logloss.agg(['mean','std']).reset_index()
    comparison=summary.pivot(index=['deployment','diagnostic'],columns='arm',values='mean')
    delta=comparison['EXPANDED-HT-PROFILE']-comparison['R55-FROZEN']
    comparison['delta']=delta
    passed=bool((delta<=1e-12).all() and (delta<-1e-12).any())
    n.write_csv(ROOT/'tables/CALIBRATION_METRICS_PER_SEED.csv',rows)
    n.write_csv(ROOT/'tables/CALIBRATION_SUMMARY.csv',summary)
    n.write_csv(ROOT/'tables/PILOT_GATE_COMPARISON.csv',comparison.reset_index())
    n.write_csv(ROOT/'tables/RIDGE_GRID.csv',grid)
    pd.DataFrame(predictions).to_parquet(ROOT/'tables/CLASSIFIER_PREDICTIONS.parquet',index=False)
    n.write_json(ROOT/'configs/EXPANDED_CLASSIFIERS.json',selected)
    n.write_json(ROOT/'contracts/PILOT_GATE.json',{'UTC':n.utc(),'gate':'PASS' if passed else 'FAIL',
        'catalog_scoring_allowed':passed,'goal_achieved':False,'status':n.STATUS,
        'no_real_or_catalog_outcomes_read':True,'same_method_both_runs':True})
    print(comparison.to_string(),flush=True)
    print('EXPANSION_PILOT_GATE','PASS' if passed else 'FAIL',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','data-root','pilot-root','reference-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','prepare','unit','measure','fit'),required=True)
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    ROOT,DATA,PILOT,REFERENCE=args.root,args.data_root,args.pilot_root,args.reference_root
    if args.stage=='freeze':
        freeze()
    else:
        check()
        if args.stage=='prepare': physical.prepare(args.workers)
        elif args.stage=='unit': physical.unit()
        elif args.stage=='measure': physical.run(args.workers)
        else: fit()
