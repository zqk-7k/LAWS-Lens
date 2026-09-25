#!/usr/bin/env python3
"""Read-only recomputation of archived scores; no inference or parameter selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

P=Path('/root/autodl-tmp/gw-catalog/results')
A=P/'mcwf_unified_path875_devconf_20260908T181500Z'
B=P/'o4b_hl_bayestar_new_score_only_20260912T072746Z'
COLS=['R1','R10','AP','F50','F90','R1_pessimistic','R10_pessimistic','tied_query_fraction','F50_threshold','F90_threshold']
INPUTS={}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(p):
    INPUTS[str(p)]=sha(p)
    if p.suffix=='.json':return json.loads(p.read_text())
    if p.suffix=='.csv':return pd.read_csv(p)
    return pd.read_parquet(p)


def csv(root,name,f):
    f.to_csv(root/name,index=False,encoding='utf-8-sig')


def metrics(f,v):
    v=np.asarray(v,float);y=f.is_true_pair.to_numpy(bool)
    ii,jj=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    n=int(f.event_count.iloc[0]);mat=np.full((n,n),-np.inf)
    mat[ii,jj]=v;mat[jj,ii]=v
    rows=np.r_[ii[y],jj[y]];ss=np.r_[v[y],v[y]]
    ranks=1+(mat[rows]>ss[:,None]).sum(1)
    worst=(mat[rows]>=ss[:,None]).sum(1)
    families=np.tile(f.true_pair_family.to_numpy(str)[y],2)
    mean=lambda mask:float(np.mean([mask[families==fam].mean() for fam in np.unique(families)]))
    out={'R1':mean(ranks<=1),'R10':mean(ranks<=10),'AP':float(average_precision_score(y,v)),
         'R1_pessimistic':mean(worst<=1),'R10_pessimistic':mean(worst<=10),
         'tied_query_fraction':float(np.mean(worst!=ranks))}
    order=np.argsort(-v,kind='stable');tp=np.cumsum(y[order]);fp=np.cumsum(~y[order])
    for target,name in [(0.5,'F50'),(0.9,'F90')]:
        idx=np.searchsorted(tp/y.sum(),target,side='left')
        out[name]=float(fp[idx]);out[name+'_threshold']=int(np.sum((v>=v[order[idx]])&~y))
    return out


def aggregate(f,group,models):
    avg=f.groupby(group+models,dropna=False)[COLS].mean().reset_index()
    out=avg.groupby(group,dropna=False)[COLS].agg(['mean','std']).reset_index()
    out.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in out.columns]
    return out


def o3o4(root):
    contract=read(A/'independent_confirmation/contracts/FRESH_FREEZE.json')
    recipes={(r['deployment'],r['slot']):r for r in contract['recipes']}
    rows=[];weights=[];budget=[];checks=[];val=[]
    for dep in ['gwtc3','gwtc4']:
        real=[]
        for slot in [202609061,202609062,202609063]:
            r=recipes[dep,slot];seed=r['seed'];w=np.asarray(r['upstream_weights']);joint=r['joint_config']
            weights.append({'deployment':dep,'seed':seed,'waveform':w[0],'time':w[1],'sky':w[2],'gamma':joint.get('gamma'),'beta':joint.get('beta')})
            for cs in [202609941,202609942,202609943]:
                parent=A/f'independent_confirmation/confirmation/{dep}/catalog_{cs}/model_{slot}'
                f=read(parent/'PATH25_pairs.parquet');base=read(parent/'OMC_pairs.parquet')
                z=f.retained_OMC_waveform.to_numpy()+joint['gamma']*f.joint_penalty.to_numpy()+joint['beta']*f.joint_increment.to_numpy()
                channels=np.c_[z,f.time_score,f.sky_raw_log_bf];score=channels@w
                oldw=np.asarray(r['old_normalized_weights']);alpha=r['alpha']
                expected=alpha*score+(1-alpha)*(np.c_[base.waveform_score,base.time_score,base.sky_raw_log_bf]@oldw)
                checks.append({'deployment':dep,'slot':slot,'catalog':cs,'mix_replay_error':float(np.max(np.abs(expected-f.final_score))),
                               'time_unchanged':np.array_equal(f.time_score,base.time_score),'sky_unchanged':np.array_equal(f.sky_raw_log_bf,base.sky_raw_log_bf)})
                for name,v in [('waveform',z),('fusion',score),('time',f.time_score),('sky',f.sky_raw_log_bf),('OMC_waveform',base.waveform_score),('OMC_fusion',base.final_score)]:
                    rows.append({'deployment':dep,'slot':slot,'catalog':cs,'method':name,**metrics(f,v)})
            parent=A/f'development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/{dep}/seed_{seed}'
            f=read(parent/'real_fusion_pairs.parquet')
            z=f.upstream_joint_waveform.to_numpy();con=np.c_[z,f.time_score,f.sky_raw_log_bf]*w
            f['audit_score']=con.sum(1)
            for i,name in enumerate(['wf','time','sky']):f['audit_'+name]=con[:,i]
            f=f.sort_values(['audit_score','pair_key'],ascending=[False,True],kind='stable');f['audit_rank']=np.arange(1,len(f)+1)
            f['audit_seed']=seed;real.append(f)
            f=read(parent/'validation_pairs.parquet')
            for mode,v in [('waveform',f.upstream_joint_waveform),('fusion',np.c_[f.upstream_joint_waveform,f.time_score,f.sky_raw_log_bf]@w)]:
                val.append({'deployment':dep,'slot':slot,'method':mode,**metrics(f,v)})
        allf=pd.concat(real)
        means=allf.groupby('pair_key').agg(score_mean=('audit_score','mean'),rank_mean=('audit_rank','mean'),rank_max=('audit_rank','max'),wf=('audit_wf','mean'),time=('audit_time','mean'),sky=('audit_sky','mean')).reset_index()
        extra=allf.drop_duplicates('pair_key')
        keep=['pair_key','event_i','event_j']+[c for c in extra if c.startswith('pe_') or c.startswith('official_')]
        means=means.merge(extra[keep],on='pair_key',validate='one_to_one').sort_values(['rank_mean','rank_max','score_mean','pair_key'],ascending=[True,True,False,True],kind='stable')
        means['rank']=np.arange(1,len(means)+1)
        csv(root,f'{dep}_PATH1_all_pairs_audited.csv',means)
        front='official_po_or_ml_fpp_below_0p01' if dep=='gwtc3' else 'official_po_or_phazap_fpp_below_0p01'
        for k in [10,20,50,100]:
            t=means.head(k);bc=t.pe_mc_bhattacharyya_coefficient;dm=t.pe_mc_standardized_distance
            budget.append({'deployment':dep,'budget':k,'BCmc_ge05':int((bc>=.5).sum()),'Dmax_le3':int((t.pe_dmax_intrinsic<=3).sum()),'catastrophic':int(((bc<.1)|(dm>5)).sum()),'frontend':int(t[front].sum()),'Hanabi':int(t.official_any_pair_resolved_hanabi_overlap.sum())})
    f=pd.DataFrame(rows);csv(root,'PATH1_injection_per_catalog_model.csv',f)
    csv(root,'PATH1_injection_summary.csv',aggregate(f,['deployment','method'],['slot']))
    csv(root,'PATH1_validation_summary.csv',aggregate(pd.DataFrame(val),['deployment','method'],['slot']))
    csv(root,'PATH1_real_budget.csv',pd.DataFrame(budget));csv(root,'PATH1_weights.csv',pd.DataFrame(weights));csv(root,'PATH875_replay_checks.csv',pd.DataFrame(checks))
    guards=[]
    for keys,t in f.groupby(['deployment','slot','catalog']):
        for mode in ['waveform','fusion']:
            x=t[t.method==mode].iloc[0];b=t[t.method=='OMC_'+mode].iloc[0]
            guards.append({'deployment':keys[0],'slot':keys[1],'catalog':keys[2],'mode':mode,'pass':bool(x.R10>=b.R10-.02-1e-12 and x.AP>=b.AP-.005-1e-12 and x.F50<=1.1*b.F50+1e-12 and x.F90<=1.1*b.F90+1e-12)})
    csv(root,'PATH1_per_catalog_guards.csv',pd.DataFrame(guards))
    means=[]
    for keys,t in f.groupby(['deployment','slot']):
        for mode in ['waveform','fusion']:
            x=t[t.method==mode][COLS].mean();b=t[t.method=='OMC_'+mode][COLS].mean()
            means.append({'deployment':keys[0],'slot':keys[1],'mode':mode,'pass':bool(x.R10>=b.R10-.02-1e-12 and x.AP>=b.AP-.005-1e-12 and x.F50<=b.F50*1.1+1e-12 and x.F90<=b.F90*1.1+1e-12)})
    csv(root,'PATH1_model_mean_guards.csv',pd.DataFrame(means))
    from astropy.time import Time
    calendar=[]
    for dep in ['gwtc3','gwtc4']:
        events=pd.concat([read(A/f'independent_confirmation/confirmation/{dep}/catalog_{cs}/event_manifest.parquet') for cs in [202609941,202609942,202609943]])
        dates=Time(events.gps_obs.to_numpy(),format='gps').utc.isot
        calendar.append({'deployment':dep,'events':len(events),'gps_min':float(events.gps_obs.min()),'gps_max':float(events.gps_obs.max()),
                         'utc_min':str(min(dates)),'utc_max':str(max(dates)),
                         'events_before_2019_04_01':int(np.sum(dates<'2019-04-01T00:00:00'))})
    csv(root,'SIMULATED_EVENT_CALENDAR.csv',pd.DataFrame(calendar))
    return {'checks':len(checks),'max_mix_error':max(x['mix_replay_error'] for x in checks),'catalog_guards_pass':sum(x['pass'] for x in guards),'catalog_guard_count':len(guards),
            'model_mean_guards_pass':sum(x['pass'] for x in means),'model_mean_guard_count':len(means)}


def o4b(root):
    rows=[]
    for seed in [2026091221,2026091222,2026091223]:
        f=read(B/f'results/injection/seed_{seed}/all_pair_scores.parquet')
        for name,col in [('waveform','waveform_score'),('fusion','final_score_POSITIVE'),('time','time_score'),('sky','sky_raw_log_bf')]:
            rows.append({'seed':seed,'method':name,**metrics(f,f[col])})
    f=pd.DataFrame(rows);csv(root,'O4b_450_per_seed.csv',f);csv(root,'O4b_450_summary.csv',aggregate(f,['method'],['seed']))
    f=read(B/'results/PE_audit/NEW-SCORE-ONLY-POSITIVE-CANDIDATE/all_pairs_with_PE.parquet').sort_values('consensus_rank')
    csv(root,'O4b_top20_audited.csv',f.head(20))
    b=[]
    for k in [10,20,50,100]:
        t=f.head(k);b.append({'budget':k,'BCmc_ge05':int((t.BC_Mc>=.5).sum()),'Dmax_le3':int((t.Dmax<=3).sum()),'median_BC':float(t.BC_Mc.median()),'catastrophic':int(((t.BC_Mc<.1)|(t.D_Mc>5)).sum())})
    csv(root,'O4b_real_budget_audited.csv',pd.DataFrame(b))


def main(root):
    root.mkdir(parents=True,exist_ok=False)
    a=o3o4(root);o4b(root)
    changes=[p for p,h in INPUTS.items() if sha(Path(p))!=h]
    csv(root,'INPUT_SHA256.csv',pd.DataFrame([{'path':p,'sha256':h} for p,h in INPUTS.items()]))
    (root/'AUDIT_STATUS.json').write_text(json.dumps({'read_only_archive_audit':True,'original_inputs_unchanged':not changes,'input_changes':changes,'PATH1':a,
       'O4b_190_exact_subcatalog_manifest':'not available here; cannot certify the claimed500subcatalog numbers',
       'does_not_validate_physics_or_independent_generalization':True},indent=2))
    print(json.dumps(a),flush=True)
    for name in ['PATH1_injection_summary.csv','PATH1_validation_summary.csv','PATH1_real_budget.csv','O4b_450_summary.csv','O4b_real_budget_audited.csv']:
        print(name,pd.read_csv(root/name).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
