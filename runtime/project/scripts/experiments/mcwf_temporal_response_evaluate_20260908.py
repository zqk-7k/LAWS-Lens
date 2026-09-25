#!/usr/bin/env python3
"""Validation-only integration of temporal-response mass predictors."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

PROJECT=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(PROJECT/'scripts/experiments'))
import mcwf_temporal_response_20260908 as t
import mcwf_ordered_mass_evaluate_20260907 as mass
import mcwf_finite_reference_tail_20260907 as tail

_spec=importlib.util.spec_from_file_location('calfuse_frozen',PROJECT/'results/mcwf_calfuse_01_exploratory_20260908T090318Z/scripts/calfuse.py')
cf=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(cf)
dev=t.dev
COEFFICIENTS=(0.,.125,.25,.5,1.,2.)


def simulated_calibration(root,dep,kind,slot):
    folder=root/f'models/{kind}/{dep}/seed_{slot}'
    a=np.load(folder/'development_predictions.npz')
    cp=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    _,meta=t.old.data(t.PREVIOUS,dep,'validation')
    groups=meta.source_uid.to_numpy(str)
    banks=sorted(meta.noise_bank_index.unique(),key=lambda x:hashlib.sha256(f'202609850:{dep}:{x}'.encode()).hexdigest())
    fold=meta.noise_bank_index.map({x:i%2 for i,x in enumerate(banks)}).to_numpy(int)
    mixed={g for g in np.unique(groups) if len(np.unique(fold[groups==g]))>1}
    fold[np.isin(groups,list(mixed))]=-1
    cohorts={}
    for side in (0,1):
        ids=np.flatnonzero(fold==side)
        i,j=np.triu_indices(len(ids),1);i,j=ids[i],ids[j]
        y=groups[i]==groups[j]
        if y.sum()<30:raise RuntimeError('Insufficient source/noise-disjoint calibration positives')
        x=mass.pair_features(a,i,j,cp['prior'])
        cohorts[side]=(y,x)
    y,x=cohorts[0]
    weight=np.where(y,.5/y.sum(),.5/(~y).sum())
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x['prior_overlap'],y,sample_weight=weight)
    p=iso.y_thresholds_.clip(1/(y.sum()+2),1-1/(y.sum()+2))
    spec={'mass_reference':np.sort(-np.log(x['bc'][y])).tolist(),
          'knots':iso.X_thresholds_.tolist(),'loglr':(np.log(p)-np.log1p(-p)).tolist(),
          'minimum':float(x['prior_overlap'].min()),'maximum':float(x['prior_overlap'].max()),
          'n_independent_source_positive':int(y.sum()),'prior':cp['prior'].tolist(),
          'fit_noise_banks':[int(v) for v in np.unique(meta.noise_bank_index[fold==0])],
          'audit_noise_banks':[int(v) for v in np.unique(meta.noise_bank_index[fold==1])],
          'discard_cross_noise_sources':len(mixed),'model_hash':dev.sha(folder/'selected.pt')}
    metrics=[]
    for side,(yy,xx) in cohorts.items():
        pp=tail.tail_probability(-np.log(xx['bc']),spec['mass_reference'])
        raw=np.interp(xx['prior_overlap'],spec['knots'],spec['loglr'])
        odds=1/(1+np.exp(-raw))
        w=np.where(yy,.5/yy.sum(),.5/(~yy).sum())
        metrics.append({'deployment':dep,'kind':kind,'slot':slot,'cohort':'fit' if side==0 else 'noise_disjoint_audit',
                        'true_systems':int(yy.sum()),'null_pairs':int((~yy).sum()),
                        'true_mass_tail_below_0p05':float((pp[yy]<.05).mean()),
                        'balanced_Brier':float(np.sum(w*(odds-yy)**2)),
                        'model_development_reuse':True})
    path=root/f'calibration/{kind}/{dep}/slot_{slot}.json'
    dev.json_write(path,spec)
    return spec,metrics


def pair_values(root,dep,kind,slot,es,split,spec):
    frame=cf.read(dep,es,split)
    a=t.predict(root,dep,kind,slot,es,split)
    i,j=frame.idx_i.to_numpy(int),frame.idx_j.to_numpy(int)
    x=mass.pair_features(a,i,j,np.asarray(spec['prior']))
    if not all(np.isfinite(v).all() for v in x.values()):
        raise RuntimeError('Missing event predictive density in strict pair scope')
    p=tail.tail_probability(-np.log(x['bc']),spec['mass_reference'])
    penalty=np.minimum(np.log(p/.05),0.)
    overlap=x['prior_overlap'];raw=np.interp(overlap,spec['knots'],spec['loglr'])
    outside=x['ood']|(overlap<spec['minimum'])|(overlap>spec['maximum'])
    increment=np.where(outside|(p<.05),np.minimum(raw,0.),raw).clip(-4,4)
    penalty=np.where(x['ood'],0.,penalty);increment=np.where(x['ood'],0.,increment)
    return frame,penalty,increment,{**x,'calibration_ood':outside,'tail_p':p}


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    choices=[];grid=[];audit=[]
    for dep in t.DEPS:
        for kind in ('CONTINUE','TEMPORAL'):
            for slot,es in zip(t.MODEL_SLOTS,t.SEEDS):
                spec,rows=simulated_calibration(root,dep,kind,slot);audit+=rows
                frame,penalty,increment,x=pair_values(root,dep,kind,slot,es,'validation',spec)
                w=cf.frozen_weights(dep,es)
                old=frame.waveform_score.to_numpy(float)
                b=cf.fast_metrics(frame,cf.channels(frame,old)@w)
                bw=cf.fast_metrics(frame,old)
                rows=[]
                for gamma in COEFFICIENTS:
                    for beta in COEFFICIENTS:
                        z=old+gamma*penalty+beta*increment
                        m=cf.fast_metrics(frame,cf.channels(frame,z)@w)
                        wm=cf.fast_metrics(frame,z)
                        ok=cf.guard(m,b) and cf.guard(wm,bw)
                        rows.append({'gamma':gamma,'beta':beta,'pass':ok,**m})
                        grid.append({'deployment':dep,'kind':kind,'seed':es,'gamma':gamma,'beta':beta,'guard_pass':ok,
                                     **m,**{'waveform_'+k:v for k,v in wm.items()}})
                for policy in ('CANDIDATE','RETRIEVAL'):
                    def key(r):
                        if policy=='CANDIDATE':
                            m=(r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1'])
                        else:
                            m=(-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'])
                        return (*m,r['gamma']**2+r['beta']**2,r['gamma'],r['beta'])
                    win=min((r for r in rows if r['pass']),key=key)
                    choices.append({'deployment':dep,'kind':kind,'slot':slot,'seed':es,'method':kind+'-'+policy,
                                    'gamma':win['gamma'],'beta':win['beta'],'weights':w.tolist(),'calibration':spec})
                print(json.dumps({'selected':dep,'kind':kind,'seed':es}),flush=True)
    dev.csv_write(root/'tables/VALIDATION_GRID.csv',pd.DataFrame(grid))
    dev.csv_write(root/'tables/CALIBRATION_AUDIT.csv',pd.DataFrame(audit))
    dev.json_write(root/'calibration/SELECTED.json',choices)
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in c.items() if k not in ('weights','calibration')} for c in choices]))
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'sha256':dev.sha(root/'calibration/SELECTED.json'),'file':'calibration/SELECTED.json',
        'real_or_test_read_for_selection':False,'outer_weights_changed':False})


def load_selections(root):
    freeze=json.loads((root/'contracts/INTEGRATION_FROZEN.json').read_text())
    if dev.sha(root/freeze['file'])!=freeze['sha256']:raise RuntimeError('Configuration hash mismatch')
    out=json.loads((root/freeze['file']).read_text())
    for dep in t.DEPS:
        for es in t.SEEDS:
            out.append({'deployment':dep,'seed':es,'method':'OMC','weights':cf.frozen_weights(dep,es).tolist(),'gamma':0.,'beta':0.})
    return out


def scored(root,c,split):
    dep,es=c['deployment'],c['seed']
    if c['method']=='OMC':
        f=cf.read(dep,es,split);return f,f.waveform_score.to_numpy(float),{}
    f,penalty,increment,x=pair_values(root,dep,c['kind'],c['slot'],es,split,c['calibration'])
    z=f.waveform_score.to_numpy(float)+c['gamma']*penalty+c['beta']*increment
    return f,z,{**x,'penalty':penalty,'increment':increment}


def evaluate(root):
    choices=load_selections(root);rows=[];integrity=[]
    for c in choices:
        for split in ('validation','test'):
            f,z,x=scored(root,c,split);w=np.asarray(c['weights'])
            base=f.waveform_score.to_numpy(float)
            out=f.copy();out['retained_OMC_waveform']=base;out['waveform_score']=z
            out['final_score']=cf.channels(f,z)@w
            for k,v in x.items():out['temporal_'+k]=v
            for mode,s,b in [('fusion',out.final_score.to_numpy(float),cf.channels(f,base)@w),('waveform',z,base)]:
                m=cf.fast_metrics(f,s);bm=cf.fast_metrics(f,b)
                reference=dev.BASE.full_metrics(f,s)
                for k in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                    if abs(m[k]-reference[k])>1e-10:raise RuntimeError('Metric replay failed:'+k)
                rows.append({'method':c['method'],'deployment':c['deployment'],'seed':c['seed'],'split':split,
                             'mode':mode,'guard_pass':cf.guard(m,bm),**m})
            for col in ('time_score','sky_raw_log_bf'):
                if not np.array_equal(f[col],out[col]):raise RuntimeError('Changed frozen channel')
            dest=root/f'evaluation/{c["method"]}/{c["deployment"]}/seed_{c["seed"]}'
            dest.mkdir(parents=True,exist_ok=True)
            out.to_parquet(dest/f'{split}_pairs.parquet',index=False)
            integrity.append({'method':c['method'],'deployment':c['deployment'],'seed':c['seed'],'split':split,'time_sky_max_abs_difference':0})
        print('EVALUATED',c['method'],c['deployment'],c['seed'],flush=True)
    m=pd.DataFrame(rows);dev.csv_write(root/'tables/RETRIEVAL_PER_SEED.csv',m)
    cols=['macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9']
    summary=m.groupby(['method','deployment','split','mode'])[cols].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    dev.csv_write(root/'tables/RETRIEVAL_SUMMARY.csv',summary)
    dev.csv_write(root/'tables/FROZEN_CHANNEL_REPLAY.csv',pd.DataFrame(integrity))
    dev.json_write(root/'contracts/INJECTION_COMPARISON_COMPLETE.json',{'utc':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'independent_confirmation':False})


def real(root):
    if not (root/'contracts/INJECTION_COMPARISON_COMPLETE.json').exists():raise RuntimeError('Evaluate injection first')
    choices=load_selections(root);budgets=[];correlations=[]
    for dep in t.DEPS:
        ref=pd.read_parquet(t.EXTERNAL/f'{dep}_external_reference.parquet')
        for method in sorted({c['method'] for c in choices}):
            frames={'fusion':[],'waveform':[]}
            for c in [x for x in choices if x['deployment']==dep and x['method']==method]:
                f,z,x=scored(root,c,'real');out=f.copy();out['retained_OMC_waveform']=out.waveform_score;out['waveform_score']=z
                for k,v in x.items():out['temporal_'+k]=v
                for mode,w in (('fusion',c['weights']),('waveform',[1.,0.,0.])):
                    rank=dev.BASE.rank_real(out,cf.weights_dict(w),method+'_'+mode,c['seed'])
                    columns=[k for k in ref if k not in rank or k=='pair_key']
                    rank=rank.merge(ref[columns],on='pair_key',validate='one_to_one')
                    if len(rank)!=len(f):raise RuntimeError('External join changed scope')
                    check=rank.set_index('pair_key').loc[f.pair_key]
                    for col in ('time_score','sky_raw_log_bf'):
                        if not np.array_equal(check[col],f[col]):raise RuntimeError('Frozen real channel changed')
                    if abs(check.final_score.to_numpy()-cf.channels(f,z)@np.asarray(w)).max()>1e-10:
                        raise RuntimeError('Real fusion replay failed')
                    dest=root/f'evaluation/{method}/{dep}/seed_{c["seed"]}'
                    rank.to_parquet(dest/f'real_{mode}_pairs.parquet',index=False)
                    dev.csv_write(dest/f'real_{mode}_top100.csv',rank.head(100))
                    budgets += [dev.budget_row(rank,method,dep,mode,b,c['seed']) for b in (10,20,50,100)]
                    frames[mode].append(rank)
                for col in ('pe_mc_bhattacharyya_coefficient','pe_mc_standardized_distance'):
                    correlations.append({'deployment':dep,'method':method,'seed':c['seed'],'PE_statistic':col,
                        'spearman':float(spearmanr(rank.waveform_score,rank[col],nan_policy='omit').statistic),
                        'meaning':'descriptive dependent pairs,not an iid significance test'})
            for mode,ranks in frames.items():
                result=dev.BASE.consensus_real(ranks,method+'_'+mode)
                columns=[k for k in ref if k not in result or k=='pair_key']
                result=result.merge(ref[columns],on='pair_key',validate='one_to_one')
                dest=root/f'evaluation/{method}/{dep}'
                result.to_parquet(dest/f'consensus_{mode}_all_pairs.parquet',index=False)
                dev.csv_write(dest/f'consensus_{mode}_top100.csv',result.head(100))
                budgets += [dev.budget_row(result,method,dep,mode,b) for b in (10,20,50,100)]
            print('REAL AUDIT',dep,method,flush=True)
    dev.csv_write(root/'tables/PE_OFFICIAL_BUDGETS.csv',pd.DataFrame(budgets))
    dev.csv_write(root/'tables/WAVEFORM_PE_CORRELATIONS.csv',pd.DataFrame(correlations))
    dev.json_write(root/'contracts/REAL_AUDIT_COMPLETE.json',{'rank_frozen_before_PE_join':True,'Hanabi_not_run':True,'scope':{'gwtc3':62,'gwtc4':74}})


def assess(root):
    a=pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
    a=a[(a.seed.astype(str)=='consensus')&(a.method=='fusion')&(a.budget.isin([10,20]))]
    m=pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv');rows=[]
    for method in sorted(set(a.config)-{'OMC'}):
        for dep in t.DEPS:
            b=a[(a.config=='OMC')&(a.deployment==dep)].set_index('budget')
            n=a[(a.config==method)&(a.deployment==dep)].set_index('budget').loc[b.index]
            strictcols=['BC_mc_ge_0p5','median_BC_mc','D_mc_le_3','Dmax_le_3','official_frontend','official_hanabi']
            no_loss=all((n[k]>=b[k]-1e-12).all() for k in strictcols) and (n.catastrophic_mc<=b.catastrophic_mc).all()
            pe_gain=(n.BC_mc_ge_0p5>b.BC_mc_ge_0p5).any() or (n.median_BC_mc>b.median_BC_mc+1e-6).any()
            official_gain=(n.official_frontend>b.official_frontend).any()
            injection=m[(m.method==method)&(m.deployment==dep)&(m.split=='test')].guard_pass.all()
            rows.append({'configuration':method,'deployment':dep,'no_loss_top10_top20':bool(no_loss),
                         'PE_gain':bool(pe_gain),'official_gain':bool(official_gain),'all_seed_injection_guard':bool(injection),
                         'run_target_pass':bool(no_loss and pe_gain and official_gain and injection)})
    table=pd.DataFrame(rows);dev.csv_write(root/'tables/DEVELOPMENT_TARGET_CHECK.csv',table)
    winners=table.groupby('configuration').run_target_pass.all()
    protect=pd.read_csv(root/'manifest/INPUT_SHA256.csv');checks=[]
    for r in protect.itertuples():
        current=dev.sha(Path(r.path));checks.append({'path':r.path,'before':r.sha256,'after':current,'unchanged':current==r.sha256})
    dev.csv_write(root/'manifest/PROTECTED_RECHECK.csv',pd.DataFrame(checks))
    if not all(r['unchanged'] for r in checks):raise RuntimeError('Protected file hash changed')
    dev.json_write(root/'contracts/ROUND_ASSESSMENT.json',{'development_candidates_requiring_fresh_confirmation':winners[winners].index.tolist(),
        'fresh_confirmation_completed':False,'goal_achieved':False,'status':t.STATUS,'historical_unchanged':True,
        'next':'Fresh independent source/noise confirmation if target candidate exists;otherwise retain negative result and continue a scientifically distinct mechanism.'})
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['select','evaluate','real','assess'],required=True)
    args=parser.parse_args();globals()[args.stage](args.root)
