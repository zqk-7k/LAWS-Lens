#!/usr/bin/env python3
"""Pre-outcome, simulation-only check against a cheap strength-scaled mass gap."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_classifier_20260909 as c
n=c.n


def freeze(root):
    target=root/'contracts/STRENGTH_CONTROL_CONTRACT.json'
    if target.exists() or (root/'tables/PAIR_RESULTS.parquet').exists():
        raise RuntimeError('Freeze supplemental control before complete pilot outcomes')
    n.write_json(target,{'UTC':n.utc(),'supplemental_only':True,'does_not_change_R50_gate':True,
        'physical_motivation':'At fixed waveform shape Fisher information scales with squared signal amplitude; use projection power as approximate strength covariate,not calibrated optimalSNR.',
        'control_consistency':'-log1p(min(P_i,P_j)*(logMc_i-logMc_j)^2)',
        'other_inputs':'Same frozen cosine and log minimum independent profile power.',
        'comparison':'Same R50 fit/tune pairs,ridges,balanced proper logloss,monotonic slopes; no hard-null fitting;no real/test data.',
        'interpretation':'If cheap scaled gap matches full shared fit,do not attribute gain specifically to joint q/spin information.',
        'references':['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/gr-qc/0703086'],
        'limits':'This approximation is not a Fisher covariance or PE posterior.',
        'runtime_sha256':n.sha(Path(__file__)),'status':n.STATUS,'goal_achieved':False})
    shutil.copy2(__file__,root/'scripts/shared_profile_strength_control.py')


def run(root,data):
    contract=json.loads((root/'contracts/STRENGTH_CONTROL_CONTRACT.json').read_text())
    if contract['runtime_sha256']!=n.sha(Path(__file__)):
        raise RuntimeError('Supplemental frozen code changed')
    if not (root/'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Report main fixed gate first')
    frame=pd.read_parquet(root/'tables/PAIR_RESULTS.parquet')
    configs={};outputs=[]
    for dep in n.DEPS:
        panel=frame[frame.deployment.eq(dep)].reset_index(drop=True)
        fit=panel.fold.eq(0)&panel.kind.ne('hard_null')
        tune=panel.fold.eq(1)&panel.kind.ne('hard_null')
        y,w0=c.base_weights(panel[fit]);w=c.balanced(y,w0)
        ty,tw0=c.base_weights(panel[tune]);tw=c.balanced(ty,tw0)
        strength=panel.minimum_independent_power.to_numpy(float)
        scaled=-np.log1p(strength*panel.profile_logMc_gap.to_numpy(float)**2)
        for seed in n.SEEDS:
            embedding=np.load(data/f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            embedding/=np.linalg.norm(embedding,axis=1)[:,None]
            cosine=np.einsum('ij,ij->i',embedding[panel.idx_i],embedding[panel.idx_j])
            x=np.column_stack([cosine,np.log(strength),scaled])
            choices=[]
            for ridge in c.RIDGES:
                spec=c.fit(x[fit],y,w,ridge)
                choices.append((c.loss(ty,c.predict(x[tune],spec),tw),-ridge,spec))
            selected=min(choices,key=lambda item:(item[0],item[1]))[2]
            configs[f'{dep}/{seed}']=selected
            z=c.predict(x,selected)
            for kind in ('random_null','hard_null'):
                take=panel.fold.eq(1)&panel.kind.isin(('true',kind))
                py,pw0=c.base_weights(panel[take]);pw=c.balanced(py,pw0)
                outputs.append({'deployment':dep,'seed':seed,'null_kind':kind,
                    'arm':'STRENGTH-SCALED-PROFILE-GAP','balanced_logloss':c.loss(py,z[take],pw),
                    'balanced_ROC_AUC':roc_auc_score(py,z[take],sample_weight=pw)})
    n.write_json(root/'contracts/STRENGTH_CONTROL_CLASSIFIERS.json',configs)
    n.write_csv(root/'tables/STRENGTH_CONTROL_PER_SEED.csv',outputs)
    original=pd.read_csv(root/'tables/CLASSIFIER_METRICS_PER_SEED.csv')
    combined=pd.concat([original,pd.DataFrame(outputs)],ignore_index=True)
    summary=combined.groupby(['deployment','null_kind','arm']).balanced_logloss.agg(['mean','std']).reset_index()
    n.write_csv(root/'tables/STRENGTH_CONTROL_COMPARISON.csv',summary.to_dict('records'))
    n.write_json(root/'contracts/STRENGTH_CONTROL_COMPLETE.json',{'UTC':n.utc(),
        'main_gate_unchanged':True,'real_or_test_read':False,'goal_achieved':False,'status':n.STATUS})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','run'),required=True)
    args=parser.parse_args()
    if args.stage=='freeze':
        freeze(args.root)
    else:
        run(args.root,args.data_root)
