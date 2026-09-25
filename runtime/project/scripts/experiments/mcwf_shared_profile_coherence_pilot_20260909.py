#!/usr/bin/env python3
"""Direct shared-intrinsic waveform-profile pilot, not Bayesian joint PE."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_inspiral_profile_pilot_20260909 as pilot
import mcwf_independent_predictive_calibration_20260909 as predictive
features, n, physical = pilot.features, pilot.n, pilot.prof.profile
ROOT=DATA=None
SEED=2026090950
COUNTS={'true':24,'random_null':48,'hard_null':48}
BOUNDS=((float(np.log(5.)),float(np.log(200.))),(.25,1.),(-.8,.8))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts','cache','pairs','tables','audit','scripts','logs','manifest','reports'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{
        'UTC':n.utc(),'id':'MCWF-NODUP-SHARED-PROFILE-COHERENCE-PILOT-50',
        'status':n.STATUS,'data':str(DATA),'same_both_runs':True,'pilot_seed':SEED,
        'physical_question':'Can both measured waveforms share logMc,q,equal aligned spin, allowing each image independent phase/amplitude/arrival nuisances?',
        'statistic':'D=P_independent-P_shared>=0. Also D/P_independent. P is inherited20-580Hz16s conditioned quadrature projection,NOT a normalized likelihood or Bayes factor.',
        'independent':'Retain frozen separate optima; maximize each image over the union of joint-search trials and one additional300-evaluation independent refinement per image.',
        'shared':'Maximize P_i(theta)+P_j(theta) from sorted separate optima and their midpoint;450Nelder-Mead evaluations per start,full frozenphysicalbounds,xatol/fatol1e-4. Include separate refinement results in shared candidate set.',
        'symmetry':'Identical deterministic coordinate-sorted starts under event swap. Independent refinement starting points frozen before either refinement.',
        'nuisances':'Each image independently maximizes H1 lag+/-304samples and H1-L1 lag within21samples,detector quadrature amplitudes. Does not impose sky or time-delay evidence a second time.',
        'limits':'Equal-spin aligned dominant-mode approximation;finite multimode optimizer;processed nonstationary-noise data,not exactGaussianlikelihood. No chi-square significance theorem or Hanabi claim.',
        'sampling':{'per_run_fold':COUNTS,'true':'SHA256 order among R10both-quality-validsourcepairs',
            'random_null':'SHA256 order among distinctsource and distinctparent-noisechunk activeeventpairs',
            'hard_null':'For each event,8 nearest profile-logMc neighbors meeting same exclusions,then SHA256 order,disjointfromrandom-nullpairs'},
        'units':'Explicit fit/tune global-source and noise-parent chunk split from R22B;never rehash folds. Null pairs within a fold share events;not independent draws.',
        'classifier':'ONE balanced logistic waveform classifier per original encoderseed;compare cosine/logminimumprofilepower/-absprofilelogMcgap with cosine/logminimumprofilepower/-log1pD. No other massLR or oldheads added.',
        'ridge_grid':[.001,.01,.1,1.],'monotonic':'Cosine and consistency feature nonnegative coefficient;strength unconstrained.',
        'null_training':'Random null only;hard null is separately held-out-within-tune diagnostic,not fitted as though it were uniform catalog background.',
        'weights':'True pair1;random-null pair1/sqrt(endpointdegreesproduct),then classesbalanced. Same weights for old/new feature comparison.',
        'pilot_expansion_gate':'All planned pairs finite and all units pass;newproperbalancedtune logloss nonworse than comparator for random AND hardnull separately inbothruns,averagedover3frozenencoders;strictgainin at leastone run. This is a development feasibility gate,not final retrieval adoption.',
        'selection':'Choose ridge independently per encoder and run by balanced tune random-null logloss;tie uses larger ridge. Hard-null tune does not select ridge. No zero-loss or ranking-based rescaling.',
        'no_real_test_read':True,'no_PE_or_official_inputs':True,'no_encoder_training':True,
        'frozen':['time','sky','outerweights','NODUP','PATH875','scope','allhistoricalresults'],
        'adaptive_development':True,'goal_achieved':False,
        'references':['https://pycbc.org/pycbc/latest/html/inference/examples/hierarchical.html',
            'https://arxiv.org/abs/2104.09339','https://arxiv.org/abs/gr-qc/0703086'],
        'reference_limits':'Shared-source model motivates sharedintrinsics;our maximization and empiricalcalibration are NOT the cited marginalizedBayesian analysis.'})
    predictive.DATA=DATA
    plans=[];inputs=[];inventory=[]
    for dep in n.DEPS:
        meta,_,_,_,active,_,_,_=predictive.inputs(dep)
        if not np.array_equal(meta.index.to_numpy(),np.arange(len(meta))) or not np.array_equal(meta.row_index.to_numpy(),np.arange(len(meta))):
            raise RuntimeError('Event index and frozen row_index must agree')
        profiles={int(p.stem):json.loads(p.read_text())for p in (DATA/f'profile_events/{dep}').glob('*.json')}
        noise=pd.read_csv(DATA/f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        chunk=meta.noise_bank_index.map(noise.parent_file_gps)
        if set(meta.source_uid[meta.fold==0])&set(meta.source_uid[meta.fold==1])or set(chunk[meta.fold==0])&set(chunk[meta.fold==1]):
            raise RuntimeError('Source or parent-noise split leak')
        selected_events=set()
        for fold in (0,1):
            ids=np.flatnonzero(active&meta.fold.eq(fold).to_numpy())
            sub=meta.iloc[ids]
            true=[]
            for uid,g in sub.groupby('source_uid'):
                if len(g)==2:
                    i,j=sorted(g.index.to_list())
                    true.append((digest(f'{SEED}/{dep}/{fold}/true/{uid}'),i,j))
            true=sorted(true)[:COUNTS['true']]
            null=[]
            for k,i in enumerate(ids):
                for j in ids[k+1:]:
                    if meta.source_uid.iloc[i]!=meta.source_uid.iloc[j]and chunk.iloc[i]!=chunk.iloc[j]:
                        null.append((digest(f'{SEED}/{dep}/{fold}/null/{i}/{j}'),int(i),int(j)))
            random=sorted(null)[:COUNTS['random_null']]
            random_set={(i,j)for _,i,j in random}
            neighbors=set()
            for i in ids:
                opts=[(abs(profiles[i]['logmc']-profiles[j]['logmc']),int(j))for j in ids
                      if j!=i and meta.source_uid.iloc[i]!=meta.source_uid.iloc[j]and chunk.iloc[i]!=chunk.iloc[j]]
                for _,j in sorted(opts)[:8]:
                    neighbors.add(tuple(sorted((int(i),j))))
            hard=sorted((digest(f'{SEED}/{dep}/{fold}/hard/{i}/{j}'),i,j)for i,j in neighbors-random_set)[:COUNTS['hard_null']]
            for kind,group in [('true',true),('random_null',random),('hard_null',hard)]:
                if len(group)!=COUNTS[kind]:
                    raise RuntimeError('Insufficient predefined pilot support')
                for order,i,j in group:
                    a,b=profiles[i],profiles[j]
                    plans.append({'deployment':dep,'fold':fold,'kind':kind,'idx_i':i,'idx_j':j,
                        'pair_id':f'{dep}_{fold}_{i}_{j}','selection_hash':order,
                        'source_i':meta.source_uid.iloc[i],'source_j':meta.source_uid.iloc[j],
                        'noise_parent_i':chunk.iloc[i],'noise_parent_j':chunk.iloc[j],
                        'profile_logMc_gap':abs(a['logmc']-b['logmc']),
                        'mc_truth_i':meta.mc_det.iloc[i],'mc_truth_j':meta.mc_det.iloc[j],
                        'truth_is_for_summary_only':True})
                    selected_events.update((i,j))
            inventory.append({'deployment':dep,'fold':fold,'active_events':len(ids),
                'true':len(true),'random_null':len(random),'hard_null':len(hard)})
        meta.loc[sorted(selected_events)].to_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet',index=False)
        for i in selected_events:
            p=DATA/f'profile_events/{dep}/{i}.json'
            inputs.append({'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size})
        for p in (DATA/f'data/{dep}/event_metadata.parquet',DATA/f'data/{dep}/noise/noise_manifest.csv',
                  DATA/f'predictions/{dep}/ensemble_mass.npy',
                  *(DATA/f'predictions/{dep}/{seed}_embedding.npy'for seed in n.SEEDS)):
            inputs.append({'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size})
    pd.DataFrame(plans).to_parquet(ROOT/'contracts/PAIR_PLAN.parquet',index=False)
    n.write_csv(ROOT/'tables/PLANNING_INVENTORY.csv',inventory)
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',inputs)
    shutil.copy2(__file__,ROOT/'scripts/shared_profile_coherence_pilot.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'pair_plan_sha256':n.sha(ROOT/'contracts/PAIR_PLAN.parquet'),'script_sha256':n.sha(Path(__file__))})
    print('SHARED_PROFILE_PLAN',len(plans),flush=True)


def init_prepare(data,dep):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    features.init_worker(data,dep)


def prepare_event(job):
    root,dep,row=job
    path=Path(root)/f'cache/{dep}/{int(row["row_index"])}.npz'
    tick=time.perf_counter()
    raw,freq,psd,scale=pilot.raw_for(row)
    full=pilot.generation.STATE['v3'].preprocess_24s(raw,freq,psd,band_low_hz=20.)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as f:
        np.savez_compressed(f,full20=full,frequency=freq,psd=psd)
    return {'deployment':dep,'row_index':int(row['row_index']),'path':str(path),
        'sha256':n.sha(path),'old_peak2s_bitexact':True,'unchanged_SNR_scale':scale,
        'seconds':time.perf_counter()-tick}


def prepare(workers):
    for dep in n.DEPS:
        meta=pd.read_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet')
        jobs=[]
        for row in meta.to_dict('records'):
            path=ROOT/f'cache/{dep}/{int(row["row_index"])}.npz'
            receipt=path.with_suffix('.json')
            if path.exists() or receipt.exists():
                if not(path.exists() and receipt.exists()):
                    raise RuntimeError(f'Incomplete cache retained for explicit recovery: {path}')
                if n.sha(path)!=json.loads(receipt.read_text())['sha256']:
                    raise RuntimeError(f'Corrupt cached input: {path}')
            else:
                jobs.append((str(ROOT),dep,row))
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=init_prepare,initargs=(str(DATA),dep))as pool:
            for k,out in enumerate(pool.map(prepare_event,jobs),1):
                n.write_json(ROOT/f'cache/{dep}/{out["row_index"]}.json',out)
                if k%32==0 or k==len(jobs):
                    print('PREPARED',dep,k,len(jobs),flush=True)
    expected=sum(len(pd.read_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet'))for dep in n.DEPS)
    records=[json.loads(p.read_text())for p in (ROOT/'cache').glob('gwtc*/*.json')]
    if len(records)!=expected or any(n.sha(Path(r['path']))!=r['sha256']for r in records):
        raise RuntimeError('Incomplete or corrupt prepared inputs')
    n.write_json(ROOT/'contracts/PREPARED_INPUTS_COMPLETE.json',{'UTC':n.utc(),'events':expected,'old_2s_bitexact':True})


def model(root,dep,index):
    with np.load(Path(root)/f'cache/{dep}/{index}.npz')as a:
        return physical.LowBand(a['full20'],a['frequency'],a['psd'],16)


def point(row):
    return np.asarray([row['logmc'],row['q'],row['chieff_equal']],float)


class SharedObjective:
    def __init__(self,models):
        self.models=models;self.cache={}

    def values(self,x):
        key=tuple(map(float,x))
        if key not in self.cache:
            self.cache[key]=tuple(m.evaluate(np.asarray(x))for m in self.models)
        return self.cache[key]

    def __call__(self,x):
        return -sum(self.values(x))


def optimize(fn,start,budget):
    simplex=np.tile(start,(4,1))
    for k,step in enumerate((.004,.035,.035)):
        simplex[k+1,k]+=step if start[k]+step<=BOUNDS[k][1]else-step
    result=minimize(fn,start,method='Nelder-Mead',bounds=BOUNDS,
        options={'maxfev':budget,'xatol':1e-4,'fatol':1e-4,'initial_simplex':simplex})
    return {'x':result.x.tolist(),'value':float(-result.fun),'success':bool(result.success),
            'nfev':int(result.nfev),'message':str(result.message)}


def coherent(models,separate,budget=450,refine_budget=300):
    objective=SharedObjective(models)
    starts=sorted(set(tuple(x)for x in [*separate,(separate[0]+separate[1])/2]))
    for x in starts:
        objective.values(x)
    runs=[optimize(objective,np.asarray(x),budget)for x in starts]
    # Freeze both refinement starts before evaluating either extra optimization.
    keys=sorted(objective.cache);scores=np.asarray([objective.cache[k]for k in keys])
    initial=[np.asarray(keys[int(np.argmax(scores[:,e]))])for e in (0,1)]
    independent=[]
    for e in (0,1):
        independent.append(optimize(lambda x:-objective.values(x)[e],initial[e],refine_budget))
    keys=sorted(objective.cache);scores=np.asarray([objective.cache[k]for k in keys])
    best=int(np.argmax(scores.sum(1)))
    separate_power=scores.max(0);joint_power=float(scores[best].sum())
    deficit=float(separate_power.sum()-joint_power)
    if not np.isfinite(scores).all()or min(separate_power)<=0 or deficit < -1e-9:
        raise RuntimeError('Nonfinite or negative shared-profile deficit')
    return {'shared_parameters':list(keys[best]),'shared_power':joint_power,
        'independent_power':separate_power.tolist(),'deficit':max(deficit,0.),
        'relative_deficit':max(deficit,0.)/float(separate_power.sum()),
        'minimum_independent_power':float(separate_power.min()),
        'evaluation_points':len(keys),'shared_optimizer_runs':runs,
        'independent_refinements':independent,'any_shared_converged':any(x['success']for x in runs)}


def init_compute():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    physical.base.STATE['v3']=pilot.generation.expanded.modules()[2]


def work(job):
    root,data,row=job
    before=time.perf_counter()
    dep=row['deployment'];i,j=int(row['idx_i']),int(row['idx_j'])
    profiles=[json.loads((Path(data)/f'profile_events/{dep}/{k}.json').read_text())for k in (i,j)]
    models=[model(root,dep,k)for k in (i,j)]
    replay=[float(m.evaluate(point(p))-p['projection_statistic'])for m,p in zip(models,profiles)]
    if max(abs(x)for x in replay)>1e-6:
        raise RuntimeError('Frozen event profile optimum cannot be reproduced')
    result=coherent(models,[point(p)for p in profiles])
    result.update(pair_id=row['pair_id'],deployment=dep,idx_i=i,idx_j=j,
        profile_replay_max_difference=max(abs(x)for x in replay),
        independent_gain_from_stored=(np.asarray(result['independent_power'])-
            np.asarray([p['projection_statistic']for p in profiles])).tolist(),
        seconds=time.perf_counter()-before)
    return result


def unit():
    init_compute()
    rows=[]
    for dep in n.DEPS:
        plan=pd.read_parquet(ROOT/'contracts/PAIR_PLAN.parquet')
        row=plan[(plan.deployment==dep)&(plan.kind=='true')].iloc[0]
        models=[model(ROOT,dep,int(k))for k in (row.idx_i,row.idx_j)]
        profiles=[json.loads((DATA/f'profile_events/{dep}/{int(k)}.json').read_text())for k in (row.idx_i,row.idx_j)]
        errors=[float(m.evaluate(point(p))-p['projection_statistic'])for m,p in zip(models,profiles)]
        if max(abs(x)for x in errors)>1e-6:
            raise RuntimeError('Profile data/operator replay fails')
        theta=np.asarray([np.log(10.),.6,.1])
        for number,m in enumerate(models):
            bank=m.template(theta)
            m.raw=(2.+number)*bank[:,0]+(.5-number)*bank[:,1]
            m.data=np.fft.rfft(m.raw,m.nfft)
        a=coherent(models,[theta,theta],budget=60,refine_budget=60)
        b=coherent(models[::-1],[theta,theta],budget=60,refine_budget=60)
        if a['relative_deficit']>1e-10 or abs(a['deficit']-b['deficit'])>1e-8:
            raise RuntimeError('Synthetic shared source or swap algebra failure')
        rows.append({'deployment':dep,'old_profile_max_difference':max(abs(x)for x in errors),
            'synthetic_same_source_relative_deficit':a['relative_deficit'],'swap_deficit_difference':abs(a['deficit']-b['deficit'])})
    n.write_csv(ROOT/'audit/OPERATOR_AND_SHARED_ALGEBRA.csv',rows)
    n.write_json(ROOT/'contracts/UNIT_PASS.json',{'UTC':n.utc(),'passed':True,'real_or_test_read':False})


def run(workers):
    if not (ROOT/'contracts/UNIT_PASS.json').exists():
        raise RuntimeError('Unit tests required')
    plan=pd.read_parquet(ROOT/'contracts/PAIR_PLAN.parquet')
    jobs=[(str(ROOT),str(DATA),row)for row in plan.to_dict('records')if not(ROOT/f'pairs/{row["pair_id"]}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_compute)as pool:
        tasks={pool.submit(work,j):j[2]for j in jobs}
        for k,task in enumerate(as_completed(tasks),1):
            row=tasks[task]
            try:
                output=task.result();output['status']='COMPLETE'
            except Exception:
                output={'pair_id':row['pair_id'],'status':'FAIL','traceback':traceback.format_exc()}
            n.write_json(ROOT/f'pairs/{row["pair_id"]}.json',output)
            if k%16==0 or k==len(tasks):
                print('SHARED_PROFILE_PAIRS',k,len(tasks),output['status'],flush=True)
    summarize()


def summarize():
    plan=pd.read_parquet(ROOT/'contracts/PAIR_PLAN.parquet')
    records=[json.loads(p.read_text())for p in (ROOT/'pairs').glob('*.json')]
    if len(records)!=len(plan):
        raise RuntimeError('Incomplete joint-profile pilot')
    frame=plan.merge(pd.DataFrame(records),on='pair_id',suffixes=('','_computed'),validate='one_to_one')
    frame.to_parquet(ROOT/'tables/PAIR_RESULTS.parquet',index=False)
    fail=int(frame.status.ne('COMPLETE').sum())
    summaries=[]
    for (dep,fold,kind),g in frame[frame.status.eq('COMPLETE')].groupby(['deployment','fold','kind']):
        summaries.append({'deployment':dep,'fold':fold,'kind':kind,'pairs':len(g),
            'median_deficit':g.deficit.median(),'q90_deficit':g.deficit.quantile(.9),
            'median_relative_deficit':g.relative_deficit.median(),'seconds_median':g.seconds.median(),
            'seconds_p90':g.seconds.quantile(.9),'shared_optimizer_converged':g.any_shared_converged.mean()})
    n.write_csv(ROOT/'tables/PROFILE_STATISTIC_SUMMARY.csv',summaries)
    n.write_json(ROOT/'contracts/MEASUREMENT_COMPLETE.json',{'UTC':n.utc(),'pairs':len(frame),
        'failures':fail,'all_finite':fail==0,'automatic_expansion':False,
        'next':'Fit frozen small classifiers only after all numerical checks;no real ranking at this stage.',
        'status':n.STATUS,'goal_achieved':False})
    print('SHARED_PROFILE_MEASUREMENT_COMPLETE',len(frame),fail,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','prepare','unit','run','summarize'),required=True)
    parser.add_argument('--workers',type=int,default=20)
    args=parser.parse_args();ROOT,DATA=args.root,args.data_root
    if args.stage in ('prepare','run'):
        globals()[args.stage](args.workers)
    else:
        globals()[args.stage]()
