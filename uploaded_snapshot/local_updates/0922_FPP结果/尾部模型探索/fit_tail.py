from pathlib import Path
import hashlib,json,platform,time,warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import genpareto

BASE=Path(__file__).resolve().parent
DATA=next((BASE.parent/'extracted').iterdir())
OUT=BASE/'tables';OUT.mkdir(exist_ok=True)
QS=[.9,.925,.95,.975,.98]
TARGETS=[.01,.005,.001,.0005,.00025]
KEY='final_score_POSITIVE'
fits=[];pred=[];checks=[];deletions=[];curves=[];input_hashes=[];audits=[]
def track(p):
    input_hashes.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
def fit(x,q,model='GPD'):
    x=np.asarray(x,float);u=float(np.quantile(x,q));y=x[x>u]-u
    assert len(y)>=30 and np.isfinite(y).all()
    if model=='EXP':xi=0.;scale=float(y.mean());gap=0.
    else:
        options=[]
        for start in [-.2,0.,.2]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                xi,loc,scale=genpareto.fit(y,start,floc=0,scale=float(y.mean()))
            lp=genpareto.logpdf(y,xi,loc=0,scale=scale)
            if np.isfinite(lp).all() and scale>0:options.append((float(-lp.sum()),float(xi),float(scale)))
        assert options,'All fitting starts failed'
        options.sort();_,xi,scale=options[0];gap=options[-1][0]-options[0][0]
    ys=np.sort(y);cdf=genpareto.cdf(ys,xi,scale=scale);n=len(y)
    ks=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
    theo=genpareto.ppf((np.arange(n)+.5)/n,xi,scale=scale)
    return dict(model=model,q=q,u=u,n=len(x),m=len(y),tail_weight=len(y)/len(x),xi=xi,scale=scale,
                endpoint=u-scale/xi if xi<0 else np.inf,descriptive_KS=float(ks),
                QQ_RMSE=float(np.sqrt(np.mean((ys-theo)**2))),optimizer_NLL_spread=float(gap),background_max=float(x.max()))
def probability(f,s):
    s=np.asarray(s,float)
    assert np.all(s>=f['u'])
    return f['tail_weight']*genpareto.sf(s-f['u'],f['xi'],scale=f['scale'])
def inversion(f,p):return float(f['u']+genpareto.isf(p/f['tail_weight'],f['xi'],scale=f['scale']))
def empirical(x,s):return int(np.count_nonzero(np.asarray(x)>=s))

started=time.time()
for run in ['O3','O4a','O4b']:
    files=[DATA/f'background/{run}/mean_S_calibration_pairs.csv',DATA/f'background/{run}/calibration_events.csv',DATA/f'tables/{run}/injection_mean_S_FPP.parquet',DATA/f'tables/{run}/real_GWTC_Top20_FPP.csv']
    for p in files:track(p)
    bg=pd.read_csv(files[0]);events=pd.read_csv(files[1]);test=pd.read_parquet(files[2]);real=pd.read_csv(files[3]).sort_values('consensus_rank')
    single=test[test.event_i.str.contains('unlensed') & test.event_j.str.contains('unlensed')]
    assert len(bg)==4005 and len(events)==90 and len(single)==4005 and not single.is_true_pair.any()
    assert set(events.event_uid).isdisjoint(set(single.event_i)|set(single.event_j))
    assert len(set(single.event_i)|set(single.event_j))==90
    x=bg[KEY].to_numpy();rs=real[KEY].to_numpy()
    sources=sorted(set(bg.source_i)|set(bg.source_j));noise=sorted(set(bg.noise_i)|set(bg.noise_j))
    assert len(sources)==90 and len(noise)==6
    audits.append({'run':run,'background_sources':90,'background_pairs':4005,'background_noise_blocks':6,'test_singletons':90,'test_singleton_pairs':4005})
    for q in QS:
        for model in ['GPD','EXP']:
            f=fit(x,q,model);fits.append(dict(run=run,**f))
            ps=probability(f,rs)
            for (_,r),p in zip(real.iterrows(),ps):
                pred.append(dict(run=run,rank=int(r.consensus_rank),pair=r.pair_key,S=r[KEY],q=q,model=model,
                    estimated_tail_probability=float(p),empirical_count=int(r.background_exceedances),empirical_FPP=r.conditional_FPP,
                    beyond_background_max=bool(r[KEY]>max(x)),outside_fitted_support=bool(r[KEY]>=f['endpoint']),
                    fitted_endpoint=f['endpoint']))
            for target in TARGETS:
                s=inversion(f,target)
                for name,frame in [('test_singletons',single),('test_all_unrelated',test[~test.is_true_pair])]:
                    k=empirical(frame[KEY],s);expected=len(frame)*target
                    checks.append(dict(run=run,q=q,model=model,target_tail_probability=target,threshold_S=s,
                        check_set=name,n_pairs=len(frame),observed_exceedances=k,expected_exceedances=expected,
                        observed_fraction=k/len(frame),observed_expected_ratio=k/expected))
            grid=np.linspace(f['u'],max(float(rs.max()),float(x.max()))+.1,160)
            for s,p in zip(grid,probability(f,grid)):
                curves.append(dict(run=run,q=q,model=model,S=s,estimated_tail_probability=p,empirical_count=empirical(x,s)))
        # Leave one noise parent out at each threshold, not independent pair resampling.
        for block in noise:
            sub=bg[(bg.noise_i!=block)&(bg.noise_j!=block)]
            f=fit(sub[KEY],q)
            for (_,r),p in zip(real.iterrows(),probability(f,rs)):
                deletions.append(dict(run=run,kind='noise_parent',deleted=block,q=q,rank=int(r.consensus_rank),
                    n_pairs=len(sub),xi=f['xi'],scale=f['scale'],u=f['u'],estimated_tail_probability=p,
                    outside_fitted_support=bool(r[KEY]>=f['endpoint'])))
    for source in sources:
        sub=bg[(bg.source_i!=source)&(bg.source_j!=source)]
        f=fit(sub[KEY],.95)
        for (_,r),p in zip(real.iterrows(),probability(f,rs)):
            deletions.append(dict(run=run,kind='source',deleted=source,q=.95,rank=int(r.consensus_rank),n_pairs=len(sub),
                xi=f['xi'],scale=f['scale'],u=f['u'],estimated_tail_probability=p,outside_fitted_support=bool(r[KEY]>=f['endpoint'])))
    # Deleting only the most extreme pair is an influence diagnostic, not an exclusion decision.
    f=fit(np.delete(x,np.argmax(x)),.95)
    for (_,r),p in zip(real.iterrows(),probability(f,rs)):
        deletions.append(dict(run=run,kind='maximum_pair',deleted=bg.iloc[np.argmax(x)].pair_key,q=.95,rank=int(r.consensus_rank),
            n_pairs=len(x)-1,xi=f['xi'],scale=f['scale'],u=f['u'],estimated_tail_probability=p,outside_fitted_support=bool(r[KEY]>=f['endpoint'])))
    print(run,'complete',round(time.time()-started,1),'s',flush=True)

for name,rows in [('tail_fits',fits),('candidate_predictions_all',pred),('heldout_checks',checks),('deletion_sensitivity',deletions),('tail_curves',curves)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
pf=pd.DataFrame(pred);de=pd.DataFrame(deletions);summary=[]
for run in ['O3','O4a','O4b']:
    for rank in range(1,21):
        ps=pf[(pf.run==run)&(pf['rank']==rank)];ref=ps[(ps.q==.95)&(ps.model=='GPD')].iloc[0]
        row={k:ref[k] for k in ['run','rank','pair','S','empirical_count','empirical_FPP','beyond_background_max']}
        row['GPD_q95_probability']=ref.estimated_tail_probability
        row['EXP_q95_probability']=ps[(ps.q==.95)&(ps.model=='EXP')].iloc[0].estimated_tail_probability
        for label,v in [('threshold',ps[ps.model=='GPD'].estimated_tail_probability),('source_delete',de[(de.run==run)&(de['rank']==rank)&(de.kind=='source')].estimated_tail_probability),('noise_delete_q95',de[(de.run==run)&(de['rank']==rank)&(de.kind=='noise_parent')&(de.q==.95)].estimated_tail_probability),('noise_delete_all_q',de[(de.run==run)&(de['rank']==rank)&(de.kind=='noise_parent')].estimated_tail_probability)]:
            row[label+'_min']=v.min();row[label+'_max']=v.max();row[label+'_ratio']=v.max()/v.min() if v.min()>0 else np.inf
        row['maximum_pair_delete_probability']=de[(de.run==run)&(de['rank']==rank)&(de.kind=='maximum_pair')].iloc[0].estimated_tail_probability
        summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'candidate_Top20_summary.csv',index=False)
for r in input_hashes:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
audit={'analysis':'Existing-background GPD tail exploration','seconds':time.time()-started,'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'python':platform.python_version(),'inputs':input_hashes,'design':audits,'thresholds':QS,'reference_threshold':.95,'source_changes':False,'new_simulations':False,'paper_changed':False,'uncertainty':'Deletion and threshold sensitivity only, not confidence intervals','fit_failed':False,'status':'EXPLORATORY_NO_ADOPTION'}
(BASE/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(summary).query('run=="O4b" and rank<=10').to_string(index=False))
