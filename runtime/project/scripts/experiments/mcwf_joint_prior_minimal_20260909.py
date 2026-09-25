#!/usr/bin/env python3
"""One two-feature waveform ratio using a prior-corrected joint overlap."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_information_predictive_scoring_20260909 as integration
s, h, co, r, n = integration.s, integration.h, integration.co, integration.r, integration.n
ROOT = None
INFO = P/'results/mcwf_nodup_information_integrated_19_20260909T110744Z'
METHODS = ('NODUP-DIRECT-REPLAY','PRIOR-NN-LINEAR','PRIOR-NN-TREE',
           'PRIOR-INFORMATION-LINEAR','PRIOR-INFORMATION-TREE')
CACHE = {}


def overlap(dep,seed,split,catalog=None):
    key = dep,seed,split,catalog
    if key in CACHE:
        return CACHE[key]
    slot=n.recipes()[dep,seed]['slot']
    tag=co.app.tag_for(seed,split,catalog)
    dest=ROOT/f'overlap/{dep}/{slot}_{tag}.npz'
    if dest.exists():
        CACHE[key]=dict(np.load(dest))
        return CACHE[key]
    original_path=(r.INTR/f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz'
        if split=='development' else r.prediction_path(dep,seed,split,catalog))
    with np.load(original_path) as file:
        original=file['joint'].astype(float)
    info_path=INFO/f'predictions/{dep}/{slot}_{tag}.npz'
    if not info_path.exists():
        raise RuntimeError('Information integration panel must complete first:'+str(info_path))
    information=dict(np.load(info_path))
    valid=np.isfinite(information['p']).all(1)
    ck=torch.load(r.INTR/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt',
                  map_location='cpu',weights_only=False)
    cond=r.physical.original.interpolate_rows(ck['conditional_prior'],r.physical.old.CENTERS)
    prior=ck['prior'][:,None]*cond
    if not np.isfinite(prior).all() or (prior<=0).any() or abs(prior.sum()-1)>1e-5:
        raise RuntimeError('Invalid frozen source prior')
    original=original[valid]
    conditional=original/original.sum(-1,keepdims=True).clip(1e-250)
    missing=original.sum(-1)<=1e-250
    conditional[missing]=np.broadcast_to(cond,conditional.shape)[missing]
    conditional/=conditional.sum(-1,keepdims=True)
    updated=information['p'][valid,:,None]*conditional
    result={}
    unit=[]
    for kind, joint in [('NN',original),('INFORMATION',updated)]:
        total=joint.sum((1,2))
        if np.max(abs(total-1))>1e-5 or (joint<0).any() or not np.isfinite(joint).all():
            raise RuntimeError('Invalid event joint predictive mass')
        x=torch.as_tensor((joint/np.sqrt(prior)).reshape(len(joint),-1),dtype=torch.float64,device='cuda')
        raw=(x@x.T).cpu().numpy().clip(1e-300)
        out=np.full((len(valid),len(valid)),np.nan)
        ids=np.flatnonzero(valid)
        out[np.ix_(ids,ids)]=np.log(raw)
        result[kind]=out
        for i,j in [(0,0),(0,min(1,len(joint)-1)),(len(joint)-1,len(joint)-1)]:
            expected=float(np.sum(joint[i]*joint[j]/prior))
            error=abs(raw[i,j]-expected)/max(expected,1e-300)
            if error>1e-8:
                raise RuntimeError('Prior-overlap direct-sum unit failed')
            unit.append({'kind':kind,'i':int(ids[i]),'j':int(ids[j]),'relative_error':error})
    # The prior paired with itself is neutral. Same broad posterior is not
    # automatically strong evidence merely because its shape matches itself.
    neutral=float(np.sum(prior*prior/prior))
    if abs(neutral-1)>1e-5:
        raise RuntimeError('Prior-neutral overlap unit failed')
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,**result)
    n.write_csv(dest.with_name(dest.stem+'_UNITS.csv'),unit)
    n.write_json(dest.with_suffix('.json'),{'UTC':n.utc(),'prior_self_overlap':neutral,
        'source_prior':'Frozen original conditional-model training prior,not fitted to current real events',
        'conditional_fallback_mass_max':float(np.sum(information['p'][valid]*missing,axis=1).max()),
        'public_PE_read':False,'not_physical_PE_Bayes_factor':True})
    CACHE[key]=result
    return result


def load_panel(dep,seed,split,catalog=None):
    f=co.BASE_PANEL(dep,seed,split,catalog)
    a=overlap(dep,seed,split,catalog)
    ii,jj=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    for kind in ('NN','INFORMATION'):
        f['joint_prior_log_overlap_'+kind]=a[kind][ii,jj]
    return f


def development(dep,seed,kind):
    meta=pd.read_parquet(n.t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
    meta['deployment']=dep
    fold=r.source_fold(meta)
    groups=meta.source_uid.to_numpy(str)
    z=np.load(s.EMB/f'{dep}/{seed}_development.npy').astype(float)
    z/=np.linalg.norm(z,axis=1,keepdims=True)
    a=overlap(dep,seed,'development')[kind]
    out={}
    for side in (0,1):
        ids=np.flatnonzero(fold==side)
        i,j=np.triu_indices(len(ids),1)
        i,j=ids[i],ids[j]
        y=groups[i]==groups[j]
        keys=pd.Series([':'.join(sorted((u,v))) for u,v in zip(groups[i],groups[j])])
        w=keys.map(1/keys.value_counts()).to_numpy()
        w[y]*=.5/w[y].sum()
        w[~y]*=.5/w[~y].sum()
        x=np.c_[np.sum(z[i]*z[j],axis=1),a[i,j]]
        assert np.isfinite(x).all()
        out[side]=x,y,w
    assert not set(groups[fold==0])&set(groups[fold==1])
    assert not set(meta.noise_bank_index[fold==0])&set(meta.noise_bank_index[fold==1])
    return out


def linear(x,y,w,ridge):
    mu=np.sum(w[:,None]*x,axis=0)
    sd=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0)).clip(1e-6)
    z=(x-mu)/sd
    def loss(theta):
        logits=theta[0]+z@theta[1:]
        error=w*(expit(logits)-y)
        return float(np.sum(w*(np.logaddexp(0.,logits)-y*logits))+.5*ridge*np.sum(theta[1:]**2)),np.r_[error.sum(),z.T@error+ridge*theta[1:]]
    fit=minimize(loss,np.zeros(3),jac=True,method='L-BFGS-B',bounds=[(None,None),(0.,None),(0.,None)],
                 options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    if not fit.success:
        raise RuntimeError(fit.message)
    return {'kind':'LINEAR','mu':mu,'sd':sd,'theta':fit.x}


def fit_classifier(dep,seed,prior,kind):
    parts=development(dep,seed,prior)
    x,y,w=parts[0]
    xx,yy,ww=parts[1]
    options,trials=[],[]
    for ridge in (.001,.01,.1,1.):
        models=[linear(x,y,w,ridge)] if kind=='LINEAR' else []
        if kind=='TREE':
            for leaves in (3,7):
                m=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=leaves,
                    min_samples_leaf=20,l2_regularization=ridge,early_stopping=False,
                    monotonic_cst=[1,1],random_state=202609092)
                m.fit(x,y,sample_weight=w*len(x))
                models.append({'kind':'TREE','estimator':m,'leaves':leaves})
        for model in models:
            logits=r.raw_predict(model,xx)
            value=float(np.sum(ww*(np.logaddexp(0.,logits)-yy*logits)))
            options.append(model)
            trials.append({'ridge':ridge,'leaves':model.get('leaves',0),'logloss':value})
    win=min(range(len(trials)),key=lambda i:(trials[i]['logloss'],trials[i]['leaves'],-trials[i]['ridge']))
    dest=ROOT/f'calibration/{prior}/{kind}/{dep}/{seed}.pkl'
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as file:
        pickle.dump(options[win],file)
    n.write_csv(dest.with_name(dest.stem+'_GRID.csv'),trials)
    return {'file':str(dest.relative_to(ROOT)),'sha256':n.sha(dest),'kind':kind,
        'minimum':x.min(0).tolist(),'maximum':x.max(0).tolist(),'cap':float(np.log(int(y.sum())+1)),
        'fit_sources':int(y.sum()),'tune_sources':int(yy.sum()),'selected':trials[win]}


def infer(f,c):
    original=h.isolated.ORIGINALS[c['deployment'],c['seed']]
    if c['method']==METHODS[0]:
        return co.BASE_INFER(f,{**original,'method':METHODS[0]})
    x=f[['embedding_only','joint_prior_log_overlap_'+c['prior_representation']]].to_numpy(float)
    return r.apply_classifier(x,c['waveform_calibrator'])


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    if not (INFO/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Information simulation evaluation required')
    for folder in ('contracts','configs','calibration','overlap','tables','audit','reports','scripts','logs','manifest','results','figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'id':'MCWF-NODUP-MINIMAL-PRIOR-20','UTC':n.utc(),
        'status':n.STATUS,'goal_achieved':False,'adaptive_development':True,'same_both_runs':True,
        'hypothesis':'BC rewards similar broad distributions. Prior-corrected full joint overlap preserves localization-volume information;test a parsimonious2feature classifier.',
        'statistic':'log sum_b p_i(b)p_j(b)/pi_training(b), b=(logMc,eta,chi),common bin probabilities',
        'prior':'Unchanged conditional-model checkpoint prior(Mc)*prior(eta,chi|Mc),not estimated from real catalog',
        'representations':['originalNNjoint','R19expected-information Mc with unchanged conditional eta/chi'],
        'features':['frozenbaseembeddingcosine','single_joint_prior_corrected_overlap'],
        'score':'One global classifier logit. No separateZcos,no oldMc/q,no separate massBC or conditionalBC terms,no totalblend.',
        'fit_tune':'Source/noise fold0 fit andfold1properbalancedlogloss;one source-pair totalweight;LINEAR/TREE3/7leaves100iters;ridge.001,.01,.1,1.',
        'limits':'Same feature-box positiveOOD0 and caplog(nfittrue+1). Frozen outerweights,time,sky.',
        'methods':list(METHODS),'real_selection':False,'eventIDveto':False,
        'important':'Neural/empirical predictive densities are not exact PE posteriors. This is a prior-corrected compatibility proxy,not a proper astrophysical Bayes factor.',
        'references':['https://arxiv.org/abs/2104.09339','https://arxiv.org/abs/1506.02169'],
        'all_arms_reported':True,'historical_outputs_modified':False})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',pd.read_csv(INFO/'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__,ROOT/'scripts/joint_prior_minimal.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),'script_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    configs,rows=[],[]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old=h.isolated.ORIGINALS[dep,seed]
            base={**old,'method':METHODS[0]}
            configs.append(base)
            f=load_panel(dep,seed,'validation')
            ref=infer(f,base)[0]
            fm=n.cf.fast_metrics(f,n.cf.channels(f,ref)@np.asarray(old['weights']))
            wm=n.cf.fast_metrics(f,ref)
            for prior in ('NN','INFORMATION'):
                for kind in ('LINEAR','TREE'):
                    spec=fit_classifier(dep,seed,prior,kind)
                    c={**old,'method':f'PRIOR-{prior}-{kind}','prior_representation':prior,'waveform_calibrator':spec}
                    z,oo,cl=infer(f,c)
                    m=n.cf.fast_metrics(f,n.cf.channels(f,z)@np.asarray(old['weights']))
                    ww=n.cf.fast_metrics(f,z)
                    c.update(tune_guard=n.cf.guard(m,fm) and n.cf.guard(ww,wm),tune_metrics=m)
                    poisoned=f.copy()
                    poisoned['pair_key']='ignored'
                    poisoned['pe_mc_bhattacharyya_coefficient']=-99.
                    poisoned['official_po_fpp']=1.
                    assert np.array_equal(z,infer(poisoned,{**c,'alpha':-999.,'old_mc_weight':999.})[0])
                    configs.append(c)
                    rows.append({'deployment':dep,'seed':seed,'method':c['method'],'tune_guard':c['tune_guard'],
                        'forbidden_input_delta':0.,'proper_logloss':spec['selected']['logloss'],**m})
                    print('MINIMAL_PRIOR_SELECTION',dep,seed,c['method'],c['tune_guard'],flush=True)
    n.write_csv(ROOT/'tables/PRIOR_SELECTION.csv',rows)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    args=parser.parse_args()
    ROOT=r.ROOT=co.ROOT=co.score.ROOT=args.root
    r.install()
    # R10 pair features are only a read-only carrier for exact baseline replay.
    co.score.matrices=co.matrices
    co.score.METHODS=METHODS
    n.METHODS,n.load_panel,n.infer=METHODS,load_panel,infer
    old_export,old_consensus=n.public_frame,n.dev.BASE.consensus_real
    fields=['joint_prior_log_overlap_NN','joint_prior_log_overlap_INFORMATION']
    def export(frame,z,weights,method):
        out=old_export(frame,z,weights,method)
        for field in fields:
            out[field]=frame[field].to_numpy()
        return out
    def consensus(items,method):
        out=old_consensus(items,method)
        extra=pd.concat(items).groupby('pair_key')[fields].mean().add_suffix('_mean').reset_index()
        return out.merge(extra,on='pair_key',validate='one_to_one')
    n.public_frame,n.dev.BASE.consensus_real=export,consensus
    if args.stage in ('freeze','calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT,args.stage)
