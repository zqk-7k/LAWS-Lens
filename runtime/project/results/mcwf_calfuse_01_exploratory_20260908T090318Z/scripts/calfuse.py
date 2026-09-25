#!/usr/bin/env python3
"""Frozen-score calibration/fusion experiment. No real-label model selection."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path('/root/autodl-tmp/gw-catalog')
PREVIOUS = PROJECT / 'results/mcwf_pe_frontend_extension_20260907T041300Z'
PAIRS = PREVIOUS / 'trials/ORDERED-MASS-COMPLETE-GRID-PRIOR/evaluation'
NOISE = PROJECT / 'results/mcwf_noise_context_exploratory_20260908T014829Z'
SEEDS = (202607241, 202607242, 202607243)
DEPS = ('gwtc3', 'gwtc4')
FAMILIES = ('OMC', 'AFFINE', 'JOINT')
REGULARIZATION = (0.01, 0.1, 1.0)
STATUS = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
sys.path.insert(0, str(PROJECT / 'scripts/experiments'))
import mcwf_development_20260905 as dev


def dump(path, value):
    dev.json_write(path, value)


def csv(path, value):
    dev.csv_write(path, pd.DataFrame(value))


def read(dep, seed, split):
    return pd.read_parquet(PAIRS / dep / f'seed_{seed}/{split}_pairs.parquet')


def frozen_weights(dep, seed):
    w = dev.BASE.FROZEN_V93_WEIGHTS[dep][seed]
    return np.array([w['waveform'], w['time'], w['sky']], float)


def weights_dict(w):
    return dict(zip(('waveform', 'time', 'sky'), map(float, w)))


def weight_grid(dep, seed):
    w = [np.array([a, b, 20-a-b], float)/20
         for a in range(21) for b in range(21-a)]
    old = frozen_weights(dep, seed)
    w.append(old / old.sum())
    unique = {tuple(np.round(a, 12)): a for a in w}
    return np.array([unique[k] for k in sorted(unique)])


def channels(frame, waveform):
    return np.column_stack((waveform, frame.time_score, frame.sky_raw_log_bf))


def features(frame, family):
    if family == 'AFFINE':
        return frame[['waveform_score']].to_numpy(float)
    if family == 'JOINT':
        return np.column_stack((frame.previous_waveform_score,
            frame.FRT_baseline_waveform_score-frame.previous_waveform_score,
            frame.waveform_score-frame.FRT_baseline_waveform_score))
    raise ValueError(family)


def balanced_weights(frame, plan):
    groups = plan.set_index('idx').system_id.astype(str)
    left = groups.loc[frame.idx_i].to_numpy()
    right = groups.loc[frame.idx_j].to_numpy()
    keys = np.array(['--'.join(sorted((a, b))) for a, b in zip(left, right)])
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    w = 1.0/count[inverse]
    y = frame.is_true_pair.to_numpy(bool)
    assert y.any() and (~y).any()
    for value in (False, True):
        w[y == value] *= 0.5/w[y == value].sum()
    return w


def split_validation(plan, seed):
    banks = sorted(plan.parent_noise_bank.unique(), key=lambda x:
        hashlib.sha256(f'202609081:{seed}:{x}'.encode()).hexdigest())
    assignment = {x: k % 2 for k, x in enumerate(banks)}
    p = plan.copy()
    p['calibration_fold'] = p.parent_noise_bank.map(assignment)
    counts = p.groupby('system_id').calibration_fold.nunique()
    good = set(counts[counts == 1].index)
    p['cross_noise_system_removed'] = ~p.system_id.isin(good)
    p.loc[p.cross_noise_system_removed, 'calibration_fold'] = -1
    return p


def subset(frame, ids):
    return frame[frame.idx_i.isin(ids) & frame.idx_j.isin(ids)].copy().reset_index(drop=True)


def fit_logit(frame, plan, family, regularization):
    x, y = features(frame, family), frame.is_true_pair.to_numpy(float)
    w = balanced_weights(frame, plan)
    mu = np.sum(x*w[:, None], axis=0)
    sd = np.sqrt(np.sum((x-mu)**2*w[:, None], axis=0))
    sd = np.maximum(sd, 1e-6)
    z = (x-mu)/sd
    def objective(a):
        eta = a[0]+z @ a[1:]
        loss = np.sum(w*(np.logaddexp(0., eta)-y*eta))
        loss += 0.5*regularization*np.dot(a[1:], a[1:])
        residual = w*(expit(eta)-y)
        gradient = np.r_[residual.sum(), z.T @ residual+regularization*a[1:]]
        return loss, gradient
    result = minimize(objective, np.zeros(1+x.shape[1]), jac=True,
        method='L-BFGS-B', bounds=[(None, None)]+[(0., None)]*x.shape[1],
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9})
    if not result.success:
        raise RuntimeError(str(result.message))
    independent_positive = plan.set_index('idx').loc[
        frame.loc[y.astype(bool), 'idx_i'], 'system_id'].nunique()
    return {'family': family, 'regularization': regularization,
        'mu': mu.tolist(), 'sd': sd.tolist(), 'coef': result.x[1:].tolist(),
        'intercept': float(result.x[0]), 'minimum': x.min(0).tolist(),
        'maximum': x.max(0).tolist(), 'fit_positive_systems': int(independent_positive),
        'score_cap': float(np.log(independent_positive+1)),
        'optimizer_success': True, 'objective': float(result.fun),
        'meaning': 'Balanced simulated-class predictive logit,not proper lensing BF',
        'support_policy': 'Feature-box OOD returns unchanged OMC; within support clip logit to +/-log(n_positive_systems+1)'}


def apply(frame, spec):
    if spec['family'] == 'OMC':
        return frame.waveform_score.to_numpy(float).copy(), np.zeros(len(frame), bool), np.zeros(len(frame), bool)
    x = features(frame, spec['family'])
    eta = spec['intercept']+((x-spec['mu'])/spec['sd']) @ np.asarray(spec['coef'])
    active = np.asarray(spec['coef']) > 1e-8
    ood = (((x < np.asarray(spec['minimum'])-1e-7) |
             (x > np.asarray(spec['maximum'])+1e-7)) & active).any(1)
    clipped = np.abs(eta) > spec['score_cap']
    score = eta.clip(-spec['score_cap'], spec['score_cap'])
    score[ood] = frame.waveform_score.to_numpy(float)[ood]
    if not np.isfinite(score).all():
        raise RuntimeError('Nonfinite calibrated score')
    return score, ood, clipped & ~ood


def fast_metrics(frame, scores, multiplicity=None):
    ii, jj = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    y = frame.is_true_pair.to_numpy(bool)
    n = int(frame.event_count.iloc[0])
    matrix = np.full((n, n), -np.inf)
    matrix[ii, jj] = scores
    matrix[jj, ii] = scores
    pos = np.flatnonzero(y)
    rank_i = 1+(matrix[ii[pos]] > scores[pos, None]).sum(1)
    rank_j = 1+(matrix[jj[pos]] > scores[pos, None]).sum(1)
    # Preserve historical optimistic ties, with pessimistic ranks separately audited.
    worst_i = (matrix[ii[pos]] >= scores[pos, None]).sum(1)
    worst_j = (matrix[jj[pos]] >= scores[pos, None]).sum(1)
    family = frame.true_pair_family.to_numpy(str)[pos]
    event_mult = np.ones(n) if multiplicity is None else np.asarray(multiplicity)
    qweight = (event_mult[ii[pos]]+event_mult[jj[pos]])/2
    values, worst = {}, []
    for name in sorted(set(family)):
        use = family == name
        ww = qweight[use]
        if ww.sum() == 0: continue
        for k in (1, 5, 10):
            value = (((rank_i[use] <= k)+(rank_j[use] <= k).astype(float))/2)
            values[f'{name.lower()}_r_at_{k}'] = float(np.average(value, weights=ww))
        worst.append(np.average(((worst_i[use] <= 10)+(worst_j[use] <= 10).astype(float))/2, weights=ww))
    for k in (1, 5, 10):
        values[f'macro_r_at_{k}'] = float(np.mean([v for name, v in values.items() if name.endswith(f'_r_at_{k}')]))
    values['pessimistic_macro_r_at_10'] = float(np.mean(worst))
    pair_weight = event_mult[ii]*event_mult[jj]
    pair_weight[y] = qweight
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(y[order]*pair_weight[order])
    fp = np.cumsum((~y[order])*pair_weight[order])
    for target, name in ((.5, '0p5'), (.9, '0p9')):
        loc = min(int(np.searchsorted(tp, target*tp[-1])), len(tp)-1)
        values[f'false_at_recall_{name}'] = float(fp[loc])
    values['average_precision'] = float(average_precision_score(y, scores, sample_weight=pair_weight))
    if multiplicity is None:
        values['roc_auc'] = float(roc_auc_score(y, scores))
        for b in (10, 20, 50, 100, 200, 500):
            a = y[order[:b]]
            values[f'top_{b}_true'] = int(a.sum())
            values[f'top_{b}_false'] = int((~a).sum())
            values[f'top_{b}_precision'] = float(a.mean())
    return values


def guard(candidate, baseline):
    return (candidate['macro_r_at_10'] >= baseline['macro_r_at_10']-.02-1e-12
        and candidate['average_precision'] >= baseline['average_precision']-.005-1e-12
        and candidate['false_at_recall_0p5'] <= 1.1*baseline['false_at_recall_0p5']+1e-12
        and candidate['false_at_recall_0p9'] <= 1.1*baseline['false_at_recall_0p9']+1e-12)


def select_grid(table, policy, old, enforce_guard=True):
    candidates = [r for r in table if r['guard_pass']] if enforce_guard else list(table)
    if not candidates:
        return None
    def key(r):
        distance = float(np.linalg.norm(np.asarray(r['weights'])-old/old.sum()))
        if policy == 'VF50':
            metrics = (r['false_at_recall_0p5'], r['false_at_recall_0p9'],
                       -r['average_precision'], -r['macro_r_at_10'], -r['macro_r_at_1'])
        else:
            metrics = (-r['macro_r_at_10'], -r['macro_r_at_1'],
                       -r['average_precision'], r['false_at_recall_0p5'], r['false_at_recall_0p9'])
        return (*metrics, distance, tuple(r['weights']))
    return min(candidates, key=key)


def initialize(root):
    if root.exists(): raise RuntimeError('New directory required')
    for folder in ('contracts','scripts','logs','tables','audit','configs','results','figures','reports','manifest'):
        (root/folder).mkdir(parents=True)
    shutil.copy2(__file__, root/'scripts/calfuse.py')
    contract = {
        'code': 'MCWF-CALFUSE-01', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'baseline': str(PAIRS), 'status': 'REGISTERED_BEFORE_NEW_CONFIGURATION_SELECTION',
        'frozen': ['all encoder checkpoints','2s 4096-point H1L1 inputs','time_score','sky_raw_log_bf',
                   'Nside512 corrected maps','strict O3 62 and O4a 74 event scope','all historical results'],
        'changes': ['waveform scalar/three-component calibration','validation-selected outer weights'],
        'family': {'OMC':'unchanged current waveform','AFFINE':'positive-slope balanced logistic calibration of complete OMC score',
                   'JOINT':'nonnegative joint logistic of original waveform, FRT increment, OMC increment; replaces summation,does not add a fourth channel'},
        'fit_tune': 'hash parent noise banks with seed 202609081:evaluation_seed:bank, alternating fit/tune; remove whole systems spanning both; no fit/tune source or parent-noise overlap',
        'regularization_grid': REGULARIZATION, 'regularization_choice': 'minimum balanced tune log loss, then larger regularization',
        'weighted_fit':'class mass0.5 each; each source-pair total weight equal within class; correlated null pairs not independent observations',
        'min_fit_positive_systems': 10,
        'OOD':'inactive features ignored; outside fit box returns old OMC; within support cap abs(logit) at log(n_positive_systems+1)',
        'fusion': ['original C-fixed weights','VF50','VR10'],
        'weight_grid':'nonnegative simplex0.05,deduplicate proportional vectors,include original normalized vector',
        'allow_zero_channel_weight': True,
        'selection':'Only tune simulation labels; paired R10>=base-.02,AP>=base-.005,F50/F90<=1.1base; if no grid eligible return unchanged baseline. VF50=F50,F90,-AP,-R10,-R1. VR10=-R10,-R1,-AP,F50,F90. Exact ties nearest normalized original then lexicographic.',
        'additional_controls':'OMC full-validation VF50/VR10 require no calibration fit,report separately from common tune cohort',
        'protocol_identical_across_runs': True,
        'real_labels_used_for_selection': False,
        'prior_real_adaptation':'Inherited OMC and data were adaptively developed with real PE/official outcomes. This is NOT new independent confirmation.',
        'test':'previously inspected comparison set; opened after configs locked; not a pristine heldout test',
        'target':'Both runs PE/Mc and official Top10/20 overlap improve without injection guard violation; report failures,never force success',
        'target_exact':'No increase catastrophic, no decrease BC>=.5,Dmax<=3,medianBC at either budget; increase BC count sum or median sum; frontend and Hanabi count sums each increase with neither budget decreasing; all-seed test guard and waveform-only guard required',
        'uncertainty':'10000 system bootstrap R1/R10; 1000 paired source-block weighted pair metrics; 100 tune system bootstrap fusion choices. These do not adjust adaptive selection or establish independent run-to-run stability.',
        'sky_and_time':'Read-only provenance/domain audit,no map regeneration or time lookup change this round',
        'references': ['https://scikit-learn.org/stable/modules/calibration.html','https://arxiv.org/abs/1506.02169','https://arxiv.org/abs/1508.03634'],
        'final_status': STATUS}
    dump(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    inherited = pd.read_csv(NOISE/'manifest/FINAL_HISTORICAL_HASH_RECHECK.csv')
    rows = inherited[['path','sha256']].to_dict('records')
    for p in PAIRS.rglob('*.parquet'):
        rows.append({'path':str(p),'sha256':dev.sha(p)})
    for p in (Path(dev.__file__),Path(dev.BASE.__file__),Path(dev.BASE.v7.__file__)):
        rows.append({'path':str(p),'sha256':dev.sha(p)})
        shutil.copy2(p, root/'scripts'/p.name)
    protected = pd.DataFrame(rows).drop_duplicates('path')
    for r in protected.itertuples():
        assert dev.sha(Path(r.path)) == r.sha256, r.path
    csv(root/'manifest/PROTECTED_INPUTS.csv', protected)
    inventory, cross = [], []
    for dep in DEPS:
        for seed in SEEDS:
            plans = {split:dev.BASE.retained_event_plan(dep,seed,split) for split in ('validation','test')}
            for split,p in plans.items():
                csv(root/f'audit/{dep}_{seed}_{split}_event_plan.csv', p)
                inventory.append({'deployment':dep,'seed':seed,'split':split,'events':len(p),
                    'systems':p.system_id.nunique(),'parent_noise_banks':p.parent_noise_bank.nunique(),
                    'gps_min':p.gps_obs.min(),'gps_max':p.gps_obs.max(),
                    'system_ID_semantics':'inherited family:source_index,not independently re-proven GW-LMC global-system identity'})
            a,b=plans['validation'],plans['test']
            source=len(set(a.system_id)&set(b.system_id));noise=len(set(a.parent_noise_bank)&set(b.parent_noise_bank))
            assert source==0 and noise==0
            p=split_validation(a,seed)
            csv(root/f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv',p)
            fit=p[p.calibration_fold==0];tune=p[p.calibration_fold==1]
            assert not set(fit.system_id)&set(tune.system_id)
            assert not set(fit.parent_noise_bank)&set(tune.parent_noise_bank)
            counts=[int((q.groupby('system_id').size()==2).sum()) for q in (fit,tune)]
            if min(counts)<10: raise RuntimeError('Insufficient independent calibration positives')
            cross.append({'deployment':dep,'seed':seed,'validation_test_system_overlap':source,
                'validation_test_parent_noise_overlap':noise,'fit_tune_system_overlap':0,'fit_tune_noise_overlap':0,
                'fit_events':len(fit),'tune_events':len(tune),'fit_positive_systems':counts[0],
                'tune_positive_systems':counts[1],'excluded_cross_noise_events':int(p.cross_noise_system_removed.sum())})
    csv(root/'audit/INJECTION_INVENTORY.csv',inventory)
    csv(root/'audit/DATA_SPLIT_AUDIT.csv',cross)
    dump(root/'contracts/INITIALIZATION_COMPLETE.json',{'protected_files':len(protected),'all_pass':True})
    print('INITIALIZED',root,flush=True)


def calibrate(root):
    if (root/'contracts/CONFIGURATIONS_FROZEN.json').exists(): raise RuntimeError('Already frozen')
    choices, allgrids, calibration_table, replay = [], [], [], []
    for dep in DEPS:
        for seed in SEEDS:
            frame=read(dep,seed,'validation')
            p=pd.read_csv(root/f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv')
            fit=subset(frame,p[p.calibration_fold==0].idx)
            tune=subset(frame,p[p.calibration_fold==1].idx)
            old=frozen_weights(dep,seed)
            base=fast_metrics(tune,channels(tune,tune.waveform_score)@old)
            ref=dev.BASE.full_metrics(tune,channels(tune,tune.waveform_score)@old)
            for k in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                assert abs(ref[k]-base[k])<1e-10,(k,ref[k],base[k])
            replay.append({'deployment':dep,'seed':seed,'metric_replay_pass':True})
            for family in FAMILIES:
                spec={'family':'OMC'}
                if family!='OMC':
                    options=[]
                    for reg in REGULARIZATION:
                        s=fit_logit(fit,p,family,reg)
                        z,ood,clip=apply(tune,s)
                        w=balanced_weights(tune,p); y=tune.is_true_pair.to_numpy(float)
                        loss=float(np.sum(w*(np.logaddexp(0,z)-y*z)))
                        brier=float(np.sum(w*(expit(z)-y)**2))
                        row={'deployment':dep,'seed':seed,'family':family,'regularization':reg,
                            'tune_logloss':loss,'tune_Brier':brier,'tune_OOD':float(ood.mean()),
                            'tune_clip':float(clip.mean()),'coef':json.dumps(s['coef']),
                            'fit_positive_systems':s['fit_positive_systems']}
                        calibration_table.append(row)
                        options.append(((loss,-reg),s))
                    spec=min(options,key=lambda a:a[0])[1]
                dump(root/f'configs/{dep}_{seed}_{family}_CALIBRATION.json',spec)
                z,_,_=apply(tune,spec)
                original_id=f'{family}-FIXED'
                choices.append({'deployment':dep,'seed':seed,'method':original_id,
                    'family':family,'policy':'FIXED','calibration':spec,'weights':old.tolist(),
                    'cohort':'tune','fallback_to_baseline':False})
                table=[]
                for k,w in enumerate(weight_grid(dep,seed)):
                    m=fast_metrics(tune,channels(tune,z)@w)
                    table.append({'grid_id':k,'weights':w.tolist(),'guard_pass':guard(m,base),**m})
                allgrids += [{'deployment':dep,'seed':seed,'family':family,'cohort':'tune',
                    **{k:v for k,v in r.items() if k!='weights'},
                    **dict(zip(('w_waveform','w_time','w_sky'),r['weights']))} for r in table]
                for policy in ('VF50','VR10'):
                    win=select_grid(table,policy,old)
                    fallback=win is None
                    choices.append({'deployment':dep,'seed':seed,'method':f'{family}-{policy}',
                        'family':family,'policy':policy,'calibration':{'family':'OMC'} if fallback else spec,
                        'weights':old.tolist() if fallback else win['weights'],'cohort':'tune',
                        'fallback_to_baseline':fallback,'validation_metrics':base if fallback else win})
            fullbase=fast_metrics(frame,channels(frame,frame.waveform_score)@old)
            table=[]
            for k,w in enumerate(weight_grid(dep,seed)):
                m=fast_metrics(frame,channels(frame,frame.waveform_score)@w)
                table.append({'grid_id':k,'weights':w.tolist(),'guard_pass':guard(m,fullbase),**m})
            allgrids += [{'deployment':dep,'seed':seed,'family':'OMC-FULL','cohort':'full-validation',
                **{k:v for k,v in r.items() if k!='weights'},
                **dict(zip(('w_waveform','w_time','w_sky'),r['weights']))} for r in table]
            for policy in ('VF50','VR10'):
                win=select_grid(table,policy,old);assert win is not None
                choices.append({'deployment':dep,'seed':seed,'method':f'OMC-FULL-{policy}',
                    'family':'OMC','policy':policy,'calibration':{'family':'OMC'},
                    'weights':win['weights'],'cohort':'full-validation','fallback_to_baseline':False,
                    'validation_metrics':win})
            print('FROZEN',dep,seed,flush=True)
    dump(root/'configs/SELECTED_CONFIGURATIONS.json',choices)
    csv(root/'tables/VALIDATION_GRID.csv',allgrids)
    csv(root/'tables/CALIBRATION_GRID.csv',calibration_table)
    csv(root/'audit/METRIC_REPLAY.csv',replay)
    csv(root/'tables/SELECTED_WEIGHTS.csv',[{k:v for k,v in c.items() if k not in ('calibration','weights','validation_metrics')}
        |dict(zip(('w_waveform','w_time','w_sky'),c['weights'])) for c in choices])
    digest=dev.sha(root/'configs/SELECTED_CONFIGURATIONS.json')
    dump(root/'contracts/CONFIGURATIONS_FROZEN.json',{'sha256':digest,
        'file':'configs/SELECTED_CONFIGURATIONS.json','UTC':datetime.now(timezone.utc).isoformat(),
        'real_PE_or_official_tables_read_by_calibration':False,'test_scores_read_by_calibration':False})


def selections(root):
    frozen=json.loads((root/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    assert dev.sha(root/frozen['file'])==frozen['sha256']
    return json.loads((root/frozen['file']).read_text())


def evaluate(root):
    if (root/'contracts/COMPARISON_COMPLETE.json').exists():raise RuntimeError('Already evaluated')
    metrics,delta,dist,correlations=[] ,[],[],[]
    for c in selections(root):
        dep,seed,method=c['deployment'],c['seed'],c['method']
        for split in ('validation','test'):
            f=read(dep,seed,split);z,ood,clip=apply(f,c['calibration'])
            combined=channels(f,z)@np.asarray(c['weights'])
            old=channels(f,f.waveform_score)@frozen_weights(dep,seed)
            base=fast_metrics(f,old)
            for mode,score in (('fusion',combined),('waveform_only',z)):
                m=fast_metrics(f,score)
                b=base if mode=='fusion' else fast_metrics(f,f.waveform_score.to_numpy(float))
                metrics.append({'deployment':dep,'seed':seed,'method':method,'split':split,'mode':mode,
                    'n_events':int(f.event_count.iloc[0]),'n_true_pairs':int(f.is_true_pair.sum()),
                    'OOD_rate':float(ood.mean()),'clip_rate':float(clip.mean()),'guard_pass':guard(m,b),**m})
                delta.append({'deployment':dep,'seed':seed,'method':method,'split':split,'mode':mode,
                    **{k:m[k]-b[k] for k in m}})
            out=f.copy()
            out['OMC_baseline_waveform_score']=f.waveform_score
            out['waveform_score']=z;out['final_score']=combined
            for key,value in zip(('waveform','time','sky'),c['weights']):out[f'{key}_contribution']=value*out[{'waveform':'waveform_score','time':'time_score','sky':'sky_raw_log_bf'}[key]]
            out['calibration_ood']=ood;out['calibration_clipped']=clip
            assert np.array_equal(out.time_score,f.time_score) and np.array_equal(out.sky_raw_log_bf,f.sky_raw_log_bf)
            dest=root/f'results/{method}/{dep}/seed_{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            out.to_parquet(dest/f'{split}_pairs.parquet',index=False)
            for label,mask in (('companion',f.is_true_pair.to_numpy(bool)),('null',~f.is_true_pair.to_numpy(bool))):
                for name,values in (('OMC',f.waveform_score.to_numpy(float)),('calibrated_waveform',z),('time',f.time_score.to_numpy(float)),('sky',f.sky_raw_log_bf.to_numpy(float))):
                    x=values[mask]
                    dist.append({'deployment':dep,'seed':seed,'method':method,'split':split,'population':label,'channel':name,
                        'n':len(x),'mean':x.mean(),'std':x.std(ddof=1),'median':np.median(x),
                        **{f'q{q}':float(np.percentile(x,q)) for q in (1,25,75,90,95,99)},'min':x.min(),'max':x.max()})
            if method=='OMC-FIXED':
                values={'original_waveform':f.previous_waveform_score.to_numpy(float),
                    'FRT_increment':(f.FRT_baseline_waveform_score-f.previous_waveform_score).to_numpy(float),
                    'OMC_increment':(f.waveform_score-f.FRT_baseline_waveform_score).to_numpy(float),
                    'OMC_total':f.waveform_score.to_numpy(float),'time':f.time_score.to_numpy(float),'sky':f.sky_raw_log_bf.to_numpy(float)}
                for label,mask in (('companion',f.is_true_pair.to_numpy(bool)),('null',~f.is_true_pair.to_numpy(bool))):
                    for a,va in values.items():
                        for b,vb in values.items():
                            valid=np.std(va[mask])>0 and np.std(vb[mask])>0
                            correlations.append({'deployment':dep,'seed':seed,'split':split,'population':label,
                                'feature_i':a,'feature_j':b,'spearman':float(spearmanr(va[mask],vb[mask]).statistic) if valid else None})
        print('EVALUATED',dep,seed,method,flush=True)
    csv(root/'tables/RETRIEVAL_PER_SEED.csv',metrics)
    csv(root/'tables/RETRIEVAL_PAIRED_DELTAS.csv',delta)
    csv(root/'tables/SCORE_DISTRIBUTIONS.csv',dist)
    csv(root/'tables/CHANNEL_CORRELATIONS.csv',correlations)
    m=pd.DataFrame(metrics)
    keys=['deployment','method','split','mode']
    nums=[k for k in m if k not in keys+['seed'] and pd.api.types.is_numeric_dtype(m[k])]
    summary=m.groupby(keys)[nums].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    csv(root/'tables/RETRIEVAL_SUMMARY.csv',summary)
    dump(root/'contracts/COMPARISON_COMPLETE.json',{'rows':len(metrics),'fresh_blind_test':False})


def real_audit(root):
    assert (root/'contracts/COMPARISON_COMPLETE.json').exists()
    if (root/'contracts/REAL_AUDIT_COMPLETE.json').exists():raise RuntimeError('Already audited')
    choices=selections(root);budgets,correlation,integrity=[],[],[]
    references={dep:pd.read_parquet(NOISE/f'audit/{dep}_external_reference.parquet') for dep in DEPS}
    for dep in DEPS:
        for method in sorted(set(c['method'] for c in choices)):
            ranks={'fusion':[],'waveform_only':[]}
            for c in [x for x in choices if x['deployment']==dep and x['method']==method]:
                seed=c['seed'];f=read(dep,seed,'real');z,ood,clip=apply(f,c['calibration'])
                out=f.copy();out['OMC_baseline_waveform_score']=out.waveform_score
                out['waveform_score']=z;out['calibration_ood']=ood;out['calibration_clipped']=clip
                for mode,w in (('fusion',c['weights']),('waveform_only',[1.,0.,0.])):
                    rr=dev.BASE.rank_real(out,weights_dict(w),method+'_'+mode,seed)
                    ref=references[dep];cols=[k for k in ref if k not in rr or k=='pair_key']
                    rr=rr.merge(ref[cols],on='pair_key',validate='one_to_one')
                    dest=root/f'results/{method}/{dep}/seed_{seed}'
                    dest.mkdir(parents=True,exist_ok=True)
                    rr.to_parquet(dest/f'real_{mode}_pairs.parquet',index=False)
                    csv(dest/f'real_{mode}_top100.csv',rr.head(100))
                    ranks[mode].append(rr)
                    budgets += [dev.budget_row(rr,method,dep,mode,b,seed) for b in (10,20,50,100)]
                    if mode=='fusion':
                        check=rr.set_index('pair_key').loc[f.pair_key]
                        assert np.array_equal(check.time_score.to_numpy(),f.time_score.to_numpy())
                        assert np.array_equal(check.sky_raw_log_bf.to_numpy(),f.sky_raw_log_bf.to_numpy())
                        error=np.max(abs(check.final_score.to_numpy()-channels(f,z)@np.asarray(w)))
                        assert error<1e-10
                        integrity.append({'deployment':dep,'seed':seed,'method':method,'time_sky_exact':True,'max_score_error':error})
                joined=ranks['waveform_only'][-1]
                for col in ('pe_mc_bhattacharyya_coefficient','pe_mc_standardized_distance'):
                    valid=joined[col].notna()
                    correlation.append({'deployment':dep,'seed':seed,'method':method,'PE_statistic':col,
                        'spearman':float(spearmanr(joined.loc[valid,'waveform_score'],joined.loc[valid,col]).statistic),
                        'interpretation':'dependent real pairs,descriptive only,no iid p-value'})
            for mode,frames in ranks.items():
                result=dev.BASE.consensus_real(frames,method+'_'+mode)
                ref=references[dep];cols=[k for k in ref if k not in result or k=='pair_key']
                result=result.merge(ref[cols],on='pair_key',validate='one_to_one')
                flags=pd.concat(frames).groupby('pair_key')[['calibration_ood','calibration_clipped']].mean()
                result=result.merge(flags,on='pair_key',validate='one_to_one')
                dest=root/f'results/{method}/{dep}'
                result.to_parquet(dest/f'consensus_{mode}_all_pairs.parquet',index=False)
                csv(dest/f'consensus_{mode}_top100.csv',result.head(100))
                budgets += [dev.budget_row(result,method,dep,mode,b) for b in (10,20,50,100)]
            print('REAL_AUDIT',dep,method,flush=True)
    csv(root/'tables/PE_OFFICIAL_BUDGETS.csv',budgets)
    csv(root/'tables/WAVEFORM_PE_CORRELATIONS.csv',correlation)
    csv(root/'audit/REAL_SCORE_INTEGRITY.csv',integrity)
    dump(root/'contracts/REAL_AUDIT_COMPLETE.json',{'ranks_frozen_before_PE_join':True,
        'official_candidates_are_lens_truth':False,'Hanabi_run':False,'real_scope':{'gwtc3':62,'gwtc4':74}})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['init','calibrate','evaluate','real'],required=True)
    a=p.parse_args()
    {'init':initialize,'calibrate':calibrate,'evaluate':evaluate,'real':real_audit}[a.phase](a.root)


if __name__=='__main__':main()
