#!/usr/bin/env python3
"""R57: prespecified joint-feature control on the enlarged simulation panel."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_joint_feature_pilot_20260909 as classifier
import mcwf_shared_profile_monotone_boundary_20260909 as boundary
n=classifier.n
ARMS=('R55-FROZEN','EXPANDED-HT-PROFILE','EXPANDED-HT-JOINT')


def write_once(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2)


def freeze(root,expansion):
    if root.exists(): raise RuntimeError('Independent R57 output required')
    if (expansion/'contracts/MEASUREMENT_COMPLETE.json').exists():
        raise RuntimeError('Freeze joint control before expanded physical outcomes complete')
    for name in ('contracts','configs','tables','audit','scripts','manifest','reports','logs'):
        (root/name).mkdir(parents=True)
    source=json.loads((expansion/'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract={'UTC':n.utc(),'id':'MCWF-EXPANDED-JOINT-CALIBRATION-57','source':str(expansion),
        'status':n.STATUS,'goal_achieved':False,'adaptive_development':True,
        'arms':ARMS,'joint_feature':'log BC of the frozen learned joint Mc/eta/chi event probability; not public PE and not old encoder Mc/q differences',
        'reason':'R52/R54 joint-feature controls failed on the small hard-null validation. R56 extends positive source census and negative sampling; test complementary correlated features with adequate provenance, without claiming prior failures passed.',
        'fixed':source['unchanged']+['R55 cap/power/deficit boundary policy'],
        'fit':'R56 fold0, deployed power eligibility, same exact HT/source-pair weights, one4D monotone conditional logistic fit',
        'selection':'R56 fold1 full HT deployed-policy balanced logloss; same ridge .001/.01/.1/1, ties stronger ridge; no catalog/real scoring here',
        'no_independent_LR_addition':True,'no_old_encoder_Mc_q_or_total_blend':True,
        'gate':'For each new arm, all full/random/neighbor diagnostics nonworse than R55 separately in BOTH runs averaged over3fixed encoders; at leastone strictgain.',
        'common_arm_selection':'Among passing arms: largest worst-run relative full-population logloss gain vsR55, then largest average gain, then simpler EXPANDED-HT-PROFILE. Rule freezes now, before new measurements finish.',
        'on_fail':'No catalog or real scoring; retain failed controls.',
        'future_catalog_gate':'Same inherited R51 per-model/per-catalog injection and consensus+per-seed PE/official guards; no relaxation.',
        'no_new_independent_confirmation':True}
    write_once(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    data=Path(source['data'])
    inputs=[expansion/'contracts/ANALYSIS_CONTRACT.json',expansion/'contracts/START_FREEZE.json',
            expansion/'contracts/PAIR_PLAN.parquet',expansion/'configs/FROZEN_REFERENCE_CONFIGS.json',
            Path(classifier.__file__),Path(classifier.base.__file__),Path(boundary.__file__)]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            slot=n.recipes()[dep,seed]['slot']
            inputs.extend([data/f'predictions/{dep}/{slot}_parent.npz',data/f'predictions/{dep}/{seed}_embedding.npy'])
    n.write_csv(root/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p)} for p in sorted(set(inputs))])
    shutil.copy2(__file__,root/'scripts/shared_expanded_joint_calibration.py')
    write_once(root/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'runtime_sha256':n.sha(Path(__file__)),'contract_sha256':n.sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    print('EXPANDED_JOINT_CONTROL_FROZEN',root,flush=True)


def fit(root,expansion):
    frozen=json.loads((root/'contracts/START_FREEZE.json').read_text())
    if frozen['runtime_sha256']!=n.sha(Path(__file__)):
        raise RuntimeError('Frozen R57 runtime changed')
    if (root/'contracts/PILOT_GATE.json').exists(): raise RuntimeError('Fit already finalized')
    if not (expansion/'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Finish R56 measurement and primary fit first; its failure does not authorize skipping this prespecified control')
    for row in pd.read_csv(root/'manifest/INPUT_SHA256.csv').itertuples():
        if n.sha(Path(row.path))!=row.sha256: raise RuntimeError('Frozen input changed')
    contract=json.loads((expansion/'contracts/ANALYSIS_CONTRACT.json').read_text())
    data=Path(contract['data'])
    paths=[expansion/'tables/PAIR_RESULTS.parquet',expansion/'configs/EXPANDED_CLASSIFIERS.json',
           expansion/'tables/CALIBRATION_METRICS_PER_SEED.csv']
    n.write_csv(root/'manifest/SEALED_EXPANDED_INPUTS.csv',[{'path':str(p),'sha256':n.sha(p)} for p in paths])
    frame=pd.read_parquet(paths[0])
    refs={(c['deployment'],c['seed']):c for c in json.loads((expansion/'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    simple=json.loads(paths[1].read_text())
    rows,grids,predictions,normalization=[],[],[],[]
    specs={}
    for dep in n.DEPS:
        f=frame[frame.deployment==dep].reset_index(drop=True)
        ref=refs[dep,n.SEEDS[0]]
        eligible=f.minimum_independent_power.between(*ref['minimum_power_support'])&f.any_shared_converged
        f=f[eligible].reset_index(drop=True)
        train=f.fold.eq(0).to_numpy(); tune=f.fold.eq(1).to_numpy()
        y=f.kind.eq('true').to_numpy(float)
        fitw=classifier.base.balanced(y[train],f.HT_weight.to_numpy()[train])
        tunew=classifier.base.balanced(y[tune],f.HT_weight.to_numpy()[tune])
        for seed in n.SEEDS:
            e=np.load(data/f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            e/=np.linalg.norm(e,axis=1,keepdims=True)
            cosine=np.einsum('ij,ij->i',e[f.idx_i],e[f.idx_j])
            slot=n.recipes()[dep,seed]['slot']
            with np.load(data/f'predictions/{dep}/{slot}_parent.npz') as a:
                joint=a['joint'].astype(float).reshape(len(e),-1)
            sums=joint.sum(axis=1)
            if not np.isfinite(joint).all() or np.any(joint<0) or np.any(sums<=0):
                raise RuntimeError('Invalid frozen joint event distribution')
            normalization.append({'deployment':dep,'seed':seed,'mass_error':float(abs(sums-1).max()),'cells':joint.shape[1]})
            joint=np.sqrt(joint/sums[:,None])
            bc=np.array([np.dot(joint[int(i)],joint[int(j)]) for i,j in zip(f.idx_i,f.idx_j)]).clip(1e-300,1.)
            x0=np.column_stack([cosine,np.log(f.minimum_independent_power),-np.log1p(f.deficit)])
            x=np.column_stack([x0,np.log(bc)])
            options=[]
            for ridge in classifier.base.RIDGES:
                spec=classifier.fit(x[train],y[train],fitw,ridge)
                config={**refs[dep,seed],'shared_classifier':spec}
                z=boundary.predict_policy(x,config,True)
                loss=classifier.base.loss(y[tune],z[tune],tunew)
                grids.append({'deployment':dep,'seed':seed,'ridge':ridge,'tune_HT_policy_logloss':loss})
                options.append((loss,-ridge,config))
            winner=min(options,key=lambda a:(a[0],a[1]))[2]
            specs[f'{dep}/{seed}']=winner
            diagnostics={'full_population':tune,'random_draw':tune&f.kind.ne('hard_null').to_numpy(),
                'neighbor_population':tune&(f.neighbor_population.to_numpy()|(y==1))}
            for arm,config in [(ARMS[0],refs[dep,seed]),(ARMS[1],simple[f'{dep}/{seed}']),(ARMS[2],winner)]:
                z=boundary.predict_policy(x if arm==ARMS[2] else x0,config,True)
                for diagnostic,take in diagnostics.items():
                    w0=f.HT_weight.to_numpy() if diagnostic=='full_population' else f.source_pair_weight.to_numpy()
                    w=classifier.base.balanced(y[take],w0[take])
                    rows.append({'deployment':dep,'seed':seed,'arm':arm,'diagnostic':diagnostic,
                        'logloss':classifier.base.loss(y[take],z[take],w),'pairs':int(take.sum())})
                out=f[['pair_id','deployment','fold','kind','source_i','source_j','HT_weight',
                    'source_pair_weight','neighbor_population']].copy()
                out['seed']=seed;out['arm']=arm;out['learned_joint_BC']=bc;out['waveform_score']=z
                predictions.extend(out.to_dict('records'))
            print('EXPANDED_JOINT_FIT',dep,seed,flush=True)
    metrics=pd.DataFrame(rows)
    prior=pd.read_csv(expansion/'tables/CALIBRATION_METRICS_PER_SEED.csv')
    keys=['deployment','seed','arm','diagnostic']
    replay=metrics[metrics.arm!=ARMS[2]].merge(prior,on=keys,validate='one_to_one',suffixes=('','_R56'))
    if np.max(abs(replay.logloss-replay.logloss_R56))>1e-12:
        raise RuntimeError('R56 calibration replay changed')
    summary=metrics.groupby(['deployment','arm','diagnostic']).logloss.agg(['mean','std']).reset_index()
    comparison=summary.pivot(index=['deployment','diagnostic'],columns='arm',values='mean')
    passing=[]
    for arm in ARMS[1:]:
        delta=comparison[arm]-comparison[ARMS[0]]
        comparison[arm+'_delta']=delta
        if (delta<=1e-12).all() and (delta<-1e-12).any():
            gains=(-delta/comparison[ARMS[0]]).xs('full_population',level='diagnostic')
            passing.append({'arm':arm,'worst_run_relative_gain':float(gains.min()),'mean_relative_gain':float(gains.mean())})
    winner=min(passing,key=lambda a:(-a['worst_run_relative_gain'],-a['mean_relative_gain'],ARMS.index(a['arm']))) if passing else None
    n.write_csv(root/'tables/ALL_CALIBRATION_METRICS.csv',metrics)
    n.write_csv(root/'tables/PILOT_GATE_COMPARISON.csv',comparison.reset_index())
    n.write_csv(root/'tables/RIDGE_GRID.csv',grids)
    n.write_csv(root/'audit/JOINT_PROBABILITY_NORMALIZATION.csv',normalization)
    pd.DataFrame(predictions).to_parquet(root/'tables/PREDICTIONS.parquet',index=False)
    write_once(root/'configs/EXPANDED_JOINT_CLASSIFIERS.json',specs)
    write_once(root/'contracts/PILOT_GATE.json',{'UTC':n.utc(),'gate':'PASS' if winner else 'FAIL',
        'passing_arms':passing,'selected_common_arm':winner,'goal_achieved':False,'status':n.STATUS,
        'no_catalog_or_real_scoring':True,'R56_replay_max_difference':float(abs(replay.logloss-replay.logloss_R56).max())})
    print(comparison.to_string(),flush=True)
    print('EXPANDED_JOINT_GATE','PASS' if winner else 'FAIL',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expansion-root',type=Path,required=True)
    p.add_argument('--stage',choices=('freeze','fit'),required=True)
    a=p.parse_args()
    globals()[a.stage](a.root,a.expansion_root)
