#!/usr/bin/env python3
"""Frozen shared-profile waveform classifier, complete catalog evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as p
import mcwf_shared_profile_classifier_20260909 as classifier
import mcwf_profile_conditional_20260909 as co
n=co.n
BASE=P/'results/mcwf_nodup_nomix_01_20260909T071108Z'
PROFILE=co.PARENT
ROOT=PILOT=None
METHODS=('NODUP-DIRECT-REPLAY','SHARED-PROFILE-SINGLE-WF','SHARED-PROFILE-REJECT-ONLY')
RAW_EXPORT=n.public_frame
RAW_CSV=n.write_csv
FIELDS=('shared_profile_deficit','shared_profile_relative_deficit','shared_profile_minimum_power',
        'shared_profile_candidate_waveform','shared_profile_available','shared_profile_eligible',
        'shared_profile_used','shared_profile_OOD','shared_profile_any_optimizer_converged',
        'NN_joint_BC_used_in_waveform','shared_profile_scope')


def original_configs():
    return {(v['deployment'],int(v['seed'])):v for v in json.loads(
        (BASE/'configs/SELECTED_CONFIGURATIONS.json').read_text())if v['method']=='NODUP-DIRECT'}


def panel_tag(seed,split,catalog=None):
    return co.app.tag_for(seed,split,catalog)


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output directory required')
    if json.loads((PILOT/'contracts/PILOT_GATE.json').read_text())['gate']!='PASS':
        raise RuntimeError('Shared-profile pilot must pass first')
    for folder in ('contracts','configs','calibration','pairs','tables','audit','results','reports','scripts','logs','manifest'):
        (ROOT/folder).mkdir(parents=True)
    contract={'UTC':n.utc(),'id':'MCWF-NODUP-SHARED-PROFILE-CATALOG-51','status':n.STATUS,
        'goal_achieved':False,'same_method_both_runs':True,'pilot_root':str(PILOT),
        'shared_statistic':'Exactly R50:three sorted starts,450evaluations/start,300independentrefinement/image;no numerical budget reduction.',
        'pair_scope':'Every unordered pair with BOTH frozen R10 quality-valid endpoints,not selected by labels,ranks,PE or official tables.',
        'classifier':'Frozen R50 per-run/per-encoderseed balanced logistic classifier. No retraining and no extra old Mc/q regression score.',
        'confidence_cap_grid':[2.,4.,8.,16.,None],
        'cap_selection':'R50 tune random-null balanced proper logloss only;tie smaller finitecap. Hard null,oldtest,new catalog and real outcomes do not choose cap.',
        'support':'Minimum independent profile power within R50 FIT true/random range. Outside:exactNODUP. Beyond fitted consistency range,positive reward set0;negative retained within selected symmetriccap. No publicPEinputs.',
        'SINGLE_WF':'Replace entire waveform score by ONE shared-profile classifier where eligible; no additive massLR or total-score mixture. Else exactNODUP.',
        'REJECT_ONLY':'Same classifier and eligibility; min(originalNODUP waveform,newwaveform). Conservative score bound,not Bayes-factor multiplication or a total-score blend.',
        'arms':'Both prespecified,complete delivery;no choosing an arm by real PE or official overlap. SINGLE_WF primary waveform replacement;REJECT_ONLY conservative sensitivity.',
        'fixed':['allencodercheckpoints','one-dimensionaltime','sky','outerweights','catalogscope','splits','oldresults','paper'],
        'numerical_failure':'Any failed pair retained and blocks score evaluation. Optimizer nonconvergence explicitly flagged and exactNODUP fallback;not silently dropped.',
        'evaluation':'All originalvalidation/test and threeSep8reusedcatalogs perrun,all3frozenencoders. Reusedcatalogs are not newlyblind confirmation.',
        'real_gate':'All injection pair measurements and complete injection evaluations must finish before real measurements/ranking. PE/officialjoinedonlyafterranking.',
        'uncertainty':'Individual models,mean/SD;paired system-bootstrap R1/R10 and pairmetrics;rawmaxpair aggregation unchanged.',
        'goal_guards':'CurrentNODUP is primary reference;PATH875 historicalreference retained. No goal achieved merely by moving critical pair;bothrunPE/official and fullinjection guards checked.',
        'resources':'24CPUworkers;deduplicate event/profile input across encoderseeds;reuse frozen full20Hzcache,do not regenerate strain.',
        'no_Hanabi_or_complete_PE_claim':True,'adaptive_development':True}
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',contract)
    plans=[];inputs=[];inventories=[]
    for dep in n.DEPS:
        seen={}
        for file in sorted((PROFILE/f'predictions/{dep}').glob('*.npz')):
            tag=file.stem.split('_',1)[1]
            if tag=='development':
                continue
            with np.load(file)as data:
                active=data['active'].copy()
            inputs.append({'path':str(file),'sha256':n.sha(file),'bytes':file.stat().st_size})
            if tag in seen:
                if not np.array_equal(seen[tag],active):
                    raise RuntimeError('Encoderseeds disagree on supposedly shared event-quality scope')
                continue
            seen[tag]=active
            ids=np.flatnonzero(active)
            folder=PROFILE/f'cache/profile_inputs/{dep}/{tag}'
            if len(ids):
                receipt=json.loads((folder/'COMPLETE.json').read_text())
                for name,expected in receipt['inputs'].items():
                    path=folder/name
                    if n.sha(path)!=expected:
                        raise RuntimeError('Frozen cached waveform changed:'+str(path))
                    inputs.append({'path':str(path),'sha256':expected,'bytes':path.stat().st_size})
                cache_ids=np.load(folder/'event_ids.npy')
                if not set(ids).issubset(set(cache_ids)):
                    raise RuntimeError('Active profile event absent from original rawcache')
            for index in ids:
                path=PROFILE/f'profile_events/{dep}/{tag}/{index}.json'
                if not path.exists():
                    raise RuntimeError('Active event missing its frozen fitted profile')
                inputs.append({'path':str(path),'sha256':n.sha(path),'bytes':path.stat().st_size})
            i,j=np.triu_indices(len(ids),1)
            for a,b in zip(ids[i],ids[j]):
                plans.append({'deployment':dep,'tag':tag,'scope':'real'if tag=='real'else'injection',
                    'idx_i':int(a),'idx_j':int(b),'pair_id':f'{dep}/{tag}/{a}_{b}'})
            inventories.append({'deployment':dep,'tag':tag,'events':len(active),'active_events':len(ids),'pairs':len(i)})
    pd.DataFrame(plans).to_parquet(ROOT/'contracts/PAIR_PLAN.parquet',index=False)
    n.write_csv(ROOT/'tables/MEASUREMENT_INVENTORY.csv',inventories)
    for path in [PILOT/'contracts/ANALYSIS_CONTRACT.json',PILOT/'contracts/SELECTED_PILOT_CLASSIFIERS.json',
                 PILOT/'tables/PAIR_RESULTS.parquet',PILOT/'tables/CLASSIFIER_PREDICTIONS.parquet',
                 BASE/'configs/SELECTED_CONFIGURATIONS.json',Path(p.__file__),Path(classifier.__file__)]:
        inputs.append({'path':str(path),'sha256':n.sha(path),'bytes':path.stat().st_size})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',pd.DataFrame(inputs).drop_duplicates('path').to_dict('records'))
    shutil.copy2(__file__,ROOT/'scripts/shared_profile_catalog.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'runtime_sha256':n.sha(Path(__file__)),'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'pair_plan_sha256':n.sha(ROOT/'contracts/PAIR_PLAN.parquet'),'pairs':len(plans)})
    print('SHARED_PROFILE_CATALOG_FROZEN',len(plans),flush=True)


def calibrate():
    if (ROOT/'contracts/CONFIGURATIONS_FROZEN.json').exists():
        raise RuntimeError('Already calibrated')
    configs=original_configs()
    specs=json.loads((PILOT/'contracts/SELECTED_PILOT_CLASSIFIERS.json').read_text())
    pairs=pd.read_parquet(PILOT/'tables/PAIR_RESULTS.parquet')
    predictions=pd.read_parquet(PILOT/'tables/CLASSIFIER_PREDICTIONS.parquet')
    selections=[];grids=[]
    for dep in n.DEPS:
        reference=pairs[pairs.deployment.eq(dep)&pairs.fold.eq(0)&pairs.kind.ne('hard_null')]
        low,high=reference.minimum_independent_power.min(),reference.minimum_independent_power.max()
        dlo,dhi=reference.deficit.min(),reference.deficit.max()
        for seed in n.SEEDS:
            common=configs[dep,seed]
            spec=specs[f'{dep}/{seed}/SHARED-PROFILE-DEFICIT']
            tune=predictions[predictions.deployment.eq(dep)&predictions.seed.eq(seed)&predictions.arm.eq('SHARED-PROFILE-DEFICIT')&predictions.fold.eq(1)&predictions.kind.ne('hard_null')]
            y,w0=classifier.base_weights(tune);w=classifier.balanced(y,w0)
            logits=tune.logit.to_numpy(float)
            options=[]
            for cap in (2.,4.,8.,16.,None):
                value=classifier.loss(y,logits if cap is None else logits.clip(-cap,cap),w)
                row={'deployment':dep,'seed':seed,'cap':cap,'tune_logloss':value}
                options.append(row);grids.append(row)
            win=min(options,key=lambda row:(row['tune_logloss'],float('inf')if row['cap']is None else row['cap']))
            for method in METHODS:
                selections.append({**common,'method':method,'shared_classifier':spec,'shared_cap':win['cap'],
                    'minimum_power_support':[float(low),float(high)],'deficit_support':[float(dlo),float(dhi)],
                    'cap_selection':win,'no_gamma_beta_used_when_shared_score_replaces_waveform':True})
    n.write_csv(ROOT/'tables/CAP_CALIBRATION_GRID.csv',grids)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',selections)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection':False,'outerweights_frozen':True})
    print('SHARED_PROFILE_CAPS_FROZEN',flush=True)


@lru_cache(maxsize=3)
def context(dep,tag):
    folder=PROFILE/f'cache/profile_inputs/{dep}/{tag}'
    arrays={key:np.load(folder/f'{key}.npy',mmap_mode='r')for key in ('full20','frequency','psd','event_ids')}
    arrays['indices']={int(index):k for k,index in enumerate(arrays['event_ids'])}
    return arrays


def measure_one(row):
    start=time.perf_counter()
    dep,tag=row['deployment'],row['tag']
    ctx=context(dep,tag);models=[];profiles=[]
    for index in (row['idx_i'],row['idx_j']):
        k=ctx['indices'][int(index)]
        models.append(p.physical.LowBand(ctx['full20'][k],ctx['frequency'],ctx['psd'][k],16))
        profiles.append(json.loads((PROFILE/f'profile_events/{dep}/{tag}/{index}.json').read_text()))
    differences=[float(m.evaluate(p.point(profile))-profile['projection_statistic'])for m,profile in zip(models,profiles)]
    if max(abs(v)for v in differences)>1e-6:
        raise RuntimeError('Frozen single-profile projection does not replay')
    result=p.coherent(models,[p.point(profile)for profile in profiles])
    result.update(pair_id=row['pair_id'],deployment=dep,tag=tag,idx_i=row['idx_i'],idx_j=row['idx_j'],
        status='COMPLETE',profile_replay_max_difference=max(abs(v)for v in differences),
        seconds=time.perf_counter()-start)
    return result


def measure(scope,workers):
    if scope=='real'and not(ROOT/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Real measurement must wait for complete injection evaluation')
    if not(ROOT/'contracts/CONFIGURATIONS_FROZEN.json').exists():
        raise RuntimeError('Calibrate and freeze before measurements')
    frozen=json.loads((ROOT/'contracts/START_FREEZE.json').read_text())
    if n.sha(Path(__file__))!=frozen['runtime_sha256']:
        raise RuntimeError('Frozen production runtime changed')
    plan=pd.read_parquet(ROOT/'contracts/PAIR_PLAN.parquet')
    plan=plan[plan.scope.eq(scope)]
    jobs=[row for row in plan.to_dict('records')if not(ROOT/f'pairs/{row["pair_id"]}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=p.init_compute)as pool:
        tasks={pool.submit(measure_one,row):row for row in jobs}
        for k,task in enumerate(as_completed(tasks),1):
            row=tasks[task]
            try:
                result=task.result()
            except Exception:
                result={**row,'status':'FAIL','traceback':traceback.format_exc()}
            n.write_json(ROOT/f'pairs/{row["pair_id"]}.json',result)
            if k%32==0 or k==len(tasks):
                print('SHARED_CATALOG_MEASUREMENT',scope,k,len(tasks),result['status'],flush=True)
    records=[json.loads((ROOT/f'pairs/{row.pair_id}.json').read_text())for row in plan.itertuples()]
    failures=sum(row['status']!='COMPLETE'for row in records)
    if failures:
        n.write_json(ROOT/f'contracts/{scope.upper()}_MEASUREMENT_FAIL.json',{'UTC':n.utc(),'failures':failures})
        raise RuntimeError('Failed physical pairs must be audited before scoring')
    pd.DataFrame(records).to_parquet(ROOT/f'tables/{scope}_shared_profile_pairs.parquet',index=False)
    n.write_json(ROOT/f'contracts/{scope.upper()}_MEASUREMENT_COMPLETE.json',{'UTC':n.utc(),
        'pairs':len(records),'failures':0,'status':n.STATUS,'goal_achieved':False})


@lru_cache(maxsize=24)
def measured(dep,tag):
    rows=[json.loads(path.read_text())for path in (ROOT/f'pairs/{dep}/{tag}').glob('*.json')]
    return {(int(row['idx_i']),int(row['idx_j'])):row for row in rows}


def archive_file(method,dep,seed,split,catalog):
    if split=='real':
        return BASE/f'results/{method}/{dep}/seed_{seed}/real/fusion_all_pairs.parquet'
    tag=split if catalog is None else f'{split}_{catalog}'
    return BASE/f'results/{method}/{dep}/seed_{seed}/{tag}/pairs.parquet'


def load_panel(dep,seed,split,catalog=None):
    frame=pd.read_parquet(archive_file('NODUP-DIRECT',dep,seed,split,catalog))
    frame=frame.drop(columns=[c for c in frame if c.startswith(('pe_','official_'))])
    other=pd.read_parquet(archive_file(n.BASELINE,dep,seed,split,catalog))
    other=other.set_index(['idx_i','idx_j']).loc[pd.MultiIndex.from_frame(frame[['idx_i','idx_j']])]
    for col in ('time_score','sky_raw_log_bf'):
        if not np.array_equal(frame[col].to_numpy(),other[col].to_numpy()):
            raise RuntimeError('Frozen time or sky differs between baseline archives')
    frame['NODUP_frozen_waveform']=frame.waveform_score.to_numpy(float)
    frame['NODUP_frozen_ood']=frame.score_ood.to_numpy(bool)
    frame['NODUP_frozen_clip']=frame.score_clipped.to_numpy(bool)
    frame['PATH875_waveform']=other.waveform_score.to_numpy(float)
    frame['PATH875_final_score']=other.final_score.to_numpy(float)
    tag=panel_tag(seed,split,catalog)
    active=co.mass(dep,seed,split,catalog)['active']
    lookup=measured(dep,tag)
    available=active[frame.idx_i.to_numpy(int)]&active[frame.idx_j.to_numpy(int)]
    frame['shared_profile_available']=available
    frame['shared_profile_scope']=tag
    frame['shared_profile_any_optimizer_converged']=False
    for col in ('shared_profile_deficit','shared_profile_relative_deficit','shared_profile_minimum_power'):
        frame[col]=np.nan
    for position in np.flatnonzero(available):
        i,j=int(frame.idx_i.iloc[position]),int(frame.idx_j.iloc[position])
        row=lookup.get(tuple(sorted((i,j))))
        if row is None or row['status']!='COMPLETE':
            raise RuntimeError('Qualified pair missing a complete physical measurement')
        for col,key in [('shared_profile_deficit','deficit'),('shared_profile_relative_deficit','relative_deficit'),
                        ('shared_profile_minimum_power','minimum_independent_power'),
                        ('shared_profile_any_optimizer_converged','any_shared_converged')]:
            frame.loc[frame.index[position],col]=row[key]
    return frame


def infer(frame,config):
    original=frame.NODUP_frozen_waveform.to_numpy(float)
    ood=frame.NODUP_frozen_ood.to_numpy(bool).copy()
    clipped=frame.NODUP_frozen_clip.to_numpy(bool).copy()
    available=frame.shared_profile_available.to_numpy(bool)
    eligible=available&frame.shared_profile_any_optimizer_converged.to_numpy(bool)
    power=frame.shared_profile_minimum_power.to_numpy(float)
    lo,hi=config['minimum_power_support']
    eligible&=(power>=lo)&(power<=hi)
    candidate=original.copy()
    outside=np.zeros(len(frame),bool)
    positions=np.flatnonzero(eligible)
    if len(positions):
        x=np.column_stack([frame.embedding_only.to_numpy(float)[positions],np.log(power[positions]),
                           -np.log1p(frame.shared_profile_deficit.to_numpy(float)[positions])])
        logits=classifier.predict(x,config['shared_classifier'])
        if not np.isfinite(logits).all():
            raise RuntimeError('Nonfinite shared waveform score')
        cap=config['shared_cap']
        bounded=logits if cap is None else logits.clip(-cap,cap)
        d=frame.shared_profile_deficit.to_numpy(float)[positions]
        dlo,dhi=config['deficit_support']
        outside[positions]=(d<dlo)|(d>dhi)
        bounded=np.where(outside[positions],np.minimum(bounded,0.),bounded)
        candidate[positions]=bounded
        if config['method']!=METHODS[0]:
            ood[positions]=outside[positions]
            clipped[positions]=(bounded!=logits)
    if config['method']==METHODS[0]:
        result=original.copy();used=np.zeros(len(frame),bool)
    elif config['method']==METHODS[1]:
        result=candidate;used=eligible
    else:
        result=np.minimum(original,candidate);used=eligible&(candidate<original)
    if not np.array_equal(result[~eligible],original[~eligible]):
        raise RuntimeError('Untouched waveform region changed')
    if config['method']==METHODS[2]and np.any(result>original):
        raise RuntimeError('Rejection-only control increased a waveform score')
    frame['shared_profile_candidate_waveform']=candidate
    frame['shared_profile_eligible']=eligible
    frame['shared_profile_used']=used
    frame['shared_profile_OOD']=available&(~eligible|outside)
    frame['NN_joint_BC_used_in_waveform']=~used
    if config['method']!=METHODS[0]:
        ood|=available&~eligible
    return result,ood,clipped


def export(frame,z,weights,method):
    out=RAW_EXPORT(frame,z,weights,method)
    for col in FIELDS:
        if col in frame:
            out[col]=frame[col].to_numpy()
    return out


def write_csv(path,rows):
    if Path(path).name=='PER_SEED_PE_OFFICIAL_BUDGETS.csv':
        corrected=[]
        for dep in n.DEPS:
            for seed in n.SEEDS:
                for method in (n.BASELINE,*METHODS):
                    frame=pd.read_parquet(ROOT/f'results/{method}/{dep}/seed_{seed}/real/fusion_all_pairs.parquet')
                    for budget in (10,20,50,100):
                        corrected.append({**n.dev.budget_row(frame,method,dep,'fusion',budget),'seed':seed})
        rows=corrected
    return RAW_CSV(path,rows)


def evaluate(stage):
    required='INJECTION_MEASUREMENT_COMPLETE.json'if stage=='evaluate'else'REAL_MEASUREMENT_COMPLETE.json'
    if not(ROOT/'contracts'/required).exists():
        raise RuntimeError('Complete all planned physical measurements first')
    n.METHODS=METHODS;n.load_panel=load_panel;n.infer=infer;n.public_frame=export;n.write_csv=write_csv
    n.run(ROOT,stage)


def unit():
    plan=pd.read_parquet(ROOT/'contracts/PAIR_PLAN.parquet')
    first=plan[plan.tag.str.endswith('_validation')].iloc[0].to_dict()
    path=ROOT/f'pairs/{first["pair_id"]}.json'
    if not path.exists():
        p.init_compute()
        n.write_json(path,measure_one(first))
    measured.cache_clear()
    specs={(v['deployment'],v['seed'],v['method']):v for v in n.selections(ROOT)}
    checks=[]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            frame=load_panel(dep,seed,'validation')
            for method in METHODS:
                config=specs[dep,seed,method]
                z,_,_=infer(frame,config)
                poisoned=frame.copy()
                for column in ('joint_BC','joint_logbc','pe_mc_bhattacharyya_coefficient',
                               'official_po_fpp','old_mc_gap','old_q_gap'):
                    poisoned[column]=12345.
                other=infer(poisoned,{**config,'alpha':-999.})[0]
                if not np.array_equal(z,other):
                    raise RuntimeError('Unused old mass/PE/official/totalblend fields changed score')
                inactive=~frame.shared_profile_eligible.to_numpy(bool)
                if not np.array_equal(z[inactive],frame.NODUP_frozen_waveform.to_numpy()[inactive]):
                    raise RuntimeError('Exact fallback unit failed')
                if method==METHODS[0]and not np.array_equal(z,frame.NODUP_frozen_waveform.to_numpy()):
                    raise RuntimeError('NODUP unit replay failed')
                out=export(frame,z,config['weights'],method)
                for column in ('time_score','sky_raw_log_bf'):
                    if not np.array_equal(frame[column],out[column]):
                        raise RuntimeError('Frozen physical channel changed')
                checks.append({'deployment':dep,'seed':seed,'method':method,
                    'pairs':len(frame),'shared_eligible':int((~inactive).sum()),
                    'unused_field_poison_difference':float(np.max(np.abs(z-other))),
                    'inactive_waveform_max_difference':0.,'time_sky_max_difference':0.})
    n.write_csv(ROOT/'audit/PIPELINE_REPLAY_AND_POISON_UNITS.csv',checks)
    n.write_json(ROOT/'contracts/PIPELINE_UNIT_PASS.json',{'UTC':n.utc(),'tests':len(checks),
        'real_or_test_read':False,'passed':True})
    print('SHARED_CATALOG_PIPELINE_UNIT_PASS',len(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','unit','measure','evaluate','real'),required=True)
    parser.add_argument('--scope',choices=('injection','real'),default='injection')
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args();ROOT,PILOT=args.root,args.pilot_root
    if args.stage=='measure':
        measure(args.scope,args.workers)
    elif args.stage in ('evaluate','real'):
        evaluate(args.stage)
    else:
        globals()[args.stage]()
