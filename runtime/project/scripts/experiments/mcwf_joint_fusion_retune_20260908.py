#!/usr/bin/env python3
"""Validation-only fusion adjustment after changing waveform evidence."""
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

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
t,dev,cf=ev.t,ev.dev,ev.cf
UPSTREAM=P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
WAVEFORMS=('OMC','JOINT-CHI-BC-ADD-CANDIDATE','JOINT-CHI-BC-ADD-RETRIEVAL',
           'JOINT-ETA-CHI-BC-ADD-CANDIDATE','JOINT-ETA-CHI-BC-ADD-RETRIEVAL')


def waveform(dep,seed,split,name):
    f=cf.read(dep,seed,split)
    if name=='OMC':return f,f.waveform_score.to_numpy(float)
    filename='real_fusion_pairs.parquet' if split=='real' else f'{split}_pairs.parquet'
    path=UPSTREAM/f'evaluation/{name}/{dep}/seed_{seed}'/filename
    upstream=pd.read_parquet(path)
    key=['pair_key'] if split=='real' else ['idx_i','idx_j']
    a=upstream.set_index(key).loc[f.set_index(key).index]
    if len(a)!=len(f) or a.index.duplicated().any():raise RuntimeError('Pair alignment failed')
    for col in ('time_score','sky_raw_log_bf'):
        if not np.array_equal(a[col],f[col]):raise RuntimeError('Upstream changed time orsky')
    return f,a.waveform_score.to_numpy(float)


def initialize(root):
    if root.exists():raise RuntimeError('Fresh result directory required')
    for name in ('contracts','tables','calibration','evaluation','reports','figures','scripts','logs','manifest'):(root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-JOINT-FUSION-RETUNE-14','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,'goal_achieved':False,
        'same_both_runs':True,'upstream':str(UPSTREAM),'waveform_controls':list(WAVEFORMS),
        'reason':'A changed waveform evidence scale can require new validation-only fusion weights. Include old-waveform retuning control to distinguish representation gains from fusion gains.',
        'no_new_model':'All neural parameters,predictions,calibrations and waveform increments frozen from12. No new learned feature orphysical time/sky formula.',
        'family_selection_disclosure':'Mechanism family is motivated by prior adaptive real-catalog development,not a newblind hypothesis. Every numeric weight is then selected using simulation validation only.',
        'weights':'Nonnegative simplex0.05step231points,plus exact normalizedC-fixedvectorifoffgrid. POSITIVErequiresall>=0.05;NONNEGATIVEallowszeroandmustreportactivechannels.',
        'priorities':['candidate:F50,F90,-AUPRC,-R10,-R1','retrieval:-R10,-R1,-AUPRC,F50,F90'],
        'tie':'Closest normalizedC-fixedvector,thenfixedlexicographicW/T/S. No manual choices.',
        'guards':'Same retainedOMC baseline R10-.02,AP-.005,F50/F90<=1.1,perseed andbothwaveform/fusion;retain unchangedOMCifno candidatepasses.',
        'frozen':['neuralmodels','waveformperpairvalueswithinconfiguration','time_score','sky_raw_log_bf','scope','oldweightsandresults','paper'],
        'changed_channels':[],'outer_weights_changed':True,
        'real':'PE/official joinedonlyafterselection;no true labelsorIDs inranking. Official overlapnotlensedtruth. Independentfreshsource/noiseconfirmationrequiredforgoalcompletion.'})
    protected=t.protected()
    for name in WAVEFORMS[1:]:
        for path in (UPSTREAM/f'evaluation/{name}').glob('*/*/*_pairs.parquet'):
            protected.append({'path':str(path),'sha256':dev.sha(path),'bytes':path.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(protected));shutil.copy2(__file__,root/'scripts/joint_fusion_retune.py')
    dev.json_write(root/'contracts/FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    gridbase=np.asarray([(i/20,j/20,(20-i-j)/20) for i in range(21) for j in range(21-i)])
    if len(gridbase)!=231:raise RuntimeError('Simplex count')
    configs,grids=[] ,[]
    for dep in t.DEPS:
        for seed in t.SEEDS:
            fixed=cf.frozen_weights(dep,seed);fixed=fixed/fixed.sum()
            base=cf.read(dep,seed,'validation');old=base.waveform_score.to_numpy(float)
            bm,bwm=cf.fast_metrics(base,cf.channels(base,old)@fixed),cf.fast_metrics(base,old)
            for name in WAVEFORMS:
                f,z=waveform(dep,seed,'validation',name);channels=cf.channels(f,z);wm=cf.fast_metrics(f,z)
                grid=np.vstack([gridbase,fixed]) if not np.any(np.all(np.isclose(gridbase,fixed,atol=1e-12),axis=1)) else gridbase
                rows=[]
                for w in grid:
                    m=cf.fast_metrics(f,channels@w)
                    row={'waveform':w[0],'time':w[1],'sky':w[2], 'positive':bool((w>=.05-1e-12).all()),
                         'pass':cf.guard(m,bm) and cf.guard(wm,bwm),**m}
                    rows.append(row);grids.append({'deployment':dep,'seed':seed,'waveform_model':name,**row})
                for constraint in ('POSITIVE','NONNEGATIVE'):
                    allowed=[r for r in rows if r['pass'] and (constraint=='NONNEGATIVE' or r['positive'])]
                    for policy in ('CANDIDATE','RETRIEVAL'):
                        def key(r):
                            primary=(r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1']) if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'])
                            w=np.array([r['waveform'],r['time'],r['sky']]);return (*primary,float(np.square(w-fixed).sum()),*w)
                        chosen=min(allowed,key=key) if allowed else None
                        weights=[chosen[k] for k in ('waveform','time','sky')] if chosen else fixed.tolist()
                        configs.append({'method':'REFIT-'+name+'-'+constraint+'-'+policy,'deployment':dep,'seed':seed,
                                        'waveform_model':name if chosen else 'OMC','constraint':constraint,'priority':policy,'guard_fallback':not bool(chosen),
                                        'weights':weights,'active_channels':[k for k,w in zip(('waveform','time','sky'),weights) if w>0],
                                        'gamma':0.,'beta':0.})
    dev.csv_write(root/'tables/VALIDATION_GRID.csv',pd.DataFrame(grids))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{**{k:v for k,v in c.items() if k!='weights'},
        **dict(zip(('lambda_waveform','lambda_time','lambda_sky'),c['weights']))} for c in configs]))
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'real_or_test_weight_selection':False,'outer_weights_changed':True})


def scored(root,c,split):
    f,z=waveform(c['deployment'],c['seed'],split,c.get('waveform_model','OMC'))
    return f,z,{}


def evaluate(root):
    rows=[];integrity=[]
    for c in ev.load_selections(root):
        for split in ('validation','test'):
            f,z,_=scored(root,c,split);base=cf.read(c['deployment'],c['seed'],split)
            out=f.copy();out['retained_OMC_waveform']=base.waveform_score;out['waveform_score']=z
            out['final_score']=cf.channels(f,z)@np.asarray(c['weights'])
            oldweights=cf.frozen_weights(c['deployment'],c['seed'])
            for mode,score,old in [('fusion',out.final_score.to_numpy(float),cf.channels(base,base.waveform_score)@oldweights),
                                   ('waveform',z,base.waveform_score.to_numpy(float))]:
                m,b=cf.fast_metrics(f,score),cf.fast_metrics(base,old)
                reference=dev.BASE.full_metrics(f,score)
                for key in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                    if abs(m[key]-reference[key])>1e-10:raise RuntimeError('Metricreplay')
                rows.append({'method':c['method'],'deployment':c['deployment'],'seed':c['seed'],'split':split,'mode':mode,'guard_pass':cf.guard(m,b),**m})
            dest=root/f'evaluation/{c["method"]}/{c["deployment"]}/seed_{c["seed"]}';dest.mkdir(parents=True,exist_ok=True)
            out.to_parquet(dest/f'{split}_pairs.parquet',index=False)
            for col in ('time_score','sky_raw_log_bf'):
                if not np.array_equal(out[col],base[col]):raise RuntimeError('Frozenphysicalscorechanged')
            integrity.append({'method':c['method'],'deployment':c['deployment'],'seed':c['seed'],'split':split,'time_sky_raw_max_difference':0,'fusion_weights_changed':c['method']!='OMC'})
    a=pd.DataFrame(rows);dev.csv_write(root/'tables/RETRIEVAL_PER_SEED.csv',a)
    columns=['macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9']
    summary=a.groupby(['method','deployment','split','mode'])[columns].agg(['mean','std']).reset_index();summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    dev.csv_write(root/'tables/RETRIEVAL_SUMMARY.csv',summary);dev.csv_write(root/'tables/FROZEN_CHANNEL_REPLAY.csv',pd.DataFrame(integrity))
    dev.json_write(root/'contracts/INJECTION_COMPARISON_COMPLETE.json',{'rows':len(a),'independent_confirmation':False,'raw_time_sky_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['initialize','select','evaluate','real','assess'],required=True);a=p.parse_args()
    ev.scored=scored
    if a.stage in ('initialize','select','evaluate'):globals()[a.stage](a.root)
    else:getattr(ev,a.stage)(a.root)
