#!/usr/bin/env python3
"""One joint intrinsic overlap plus cosine; no separately weighted mass factors."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
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

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_quality_state_joint_calibration_20260909 as state_model
ctrl, engine, base, n, r, co, h = (getattr(state_model, k) for k in ('ctrl','engine','base','n','r','co','h'))
ROOT = DATA = PREDICTIVE = SELECTION = None
METHODS = ('NODUP-DIRECT-REPLAY', 'JOINT2-GLOBAL-LINEAR', 'JOINT2-GLOBAL-TREE',
           'JOINT2-STATE-LINEAR', 'JOINT2-STATE-TREE')
CACHE = {}
FIELDS = ('actual_waveform_joint_BC', 'single_waveform_classifier_used', 'waveform_quality_state')


def freeze():
    base.freeze()
    n.write_json(ROOT/'contracts/MINIMAL_JOINT_ADDENDUM.json', {
        'UTC': n.utc(), 'id': 'MCWF-NODUP-MINIMAL-JOINT-CLASSIFIER-36',
        'overrides_base': 'Exactlytwofeatures:embeddingcosine and logBC of ONE normalizedjoint(Mc,eta,chi)predictivedistribution.',
        'removed_inputs': ['separatelyweightedmassBC','conditionalBCratio','normalizedmassdifference','pooledwidth','absolutemeanmass'],
        'hypothesis': 'Separate free coefficients can let conditional agreement compensate for incompatible mass.Dont let that factorization choose separate importance.',
        'predictive': 'SameR33BOUNDED model selected only bynewdevelopmentfit/tune;sameNNconditionals.',
        'population': 'Newindependent512fit/512tune sources perrun;source-pairtotalweight1;identicalnoise nullpairs excluded.',
        'models': ['nonnegative2featurelogistic','smallmonotonic2featuretree'],
        'grid': {'ridge': [.001,.01,.1,1.], 'leaves': [3,7], 'iterations': 100,
                 'learning_rate': .05, 'min_leaf': 20},
        'selection': 'Balancedproper tune logloss,thenfewerleaves,thenstrongerridge. No real/test selection.',
        'state': 'globalor0/1/2profilequality states;conditionalLR includes fit logP(state|L)/P(state|N).',
        'minimum_true_sources_per_state': 20, 'OOD': 'Fitfeatureboxpositive0',
        'cap': 'log(1+totalindependentfittruepairs)',
        'score': 'ONEwaveformclassifierlogit. NoaddedoldcosLR,Mc/qhead,tailpenalty,orold/newtotalscoremixture.',
        'frozen': ['encoders','time','sky','scope','outerweights','oldresults'],
        'old_gamma_beta_unused': True, 'no_true_PE_or_official_fields': True,
        'adaptive_development': True, 'new_blind_confirmation': False,
        'status': n.STATUS, 'script_sha256': n.sha(Path(__file__))})
    shutil.copy2(__file__, ROOT/'scripts/minimal_joint_waveform_classifier.py')


def linear(x, y, w, ridge):
    mu = w@x
    sd = np.sqrt(w@((x-mu)**2)).clip(1e-6)
    a = (x-mu)/sd
    def objective(theta):
        logits = theta[0]+a@theta[1:]
        residual = w*(expit(logits)-y)
        loss = w@(np.logaddexp(0., logits)-y*logits)+.5*ridge*(theta[1:]@theta[1:])
        return float(loss), np.r_[residual.sum(), a.T@residual+ridge*theta[1:]]
    fit = minimize(objective, np.zeros(3), jac=True, method='L-BFGS-B',
        bounds=[(None,None),(0.,None),(0.,None)],
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fit.success:
        raise RuntimeError(fit.message)
    return {'kind': 'LINEAR', 'mu': mu, 'sd': sd, 'theta': fit.x}


def features(frame):
    return np.column_stack([frame.embedding_only.to_numpy(float), frame.joint_logbc.to_numpy(float)])


def fit_one(dep, seed, kind, state, panels):
    xx, yy, ww, ss = panels[0]
    vx, vy, vw, vs = panels[1]
    mask = np.ones(len(yy), bool) if state is None else ss == state
    vm = np.ones(len(vy), bool) if state is None else vs == state
    x, y, w = xx[mask], yy[mask], ww[mask].copy()
    test, label, tw = vx[vm], vy[vm], vw[vm].copy()
    if min(y.sum(), label.sum()) < 20:
        raise RuntimeError('HOLD_INSUFFICIENT_TRUE_SOURCE_SUPPORT')
    offset = 0. if state is None else float(np.log(ww[mask & yy].sum()/ww[yy].sum())-
        np.log(ww[mask & ~yy].sum()/ww[~yy].sum()))
    w[y] *= .5/w[y].sum(); w[~y] *= .5/w[~y].sum()
    tw[label] *= .5/tw[label].sum(); tw[~label] *= .5/tw[~label].sum()
    models, trials = [], []
    for ridge in (.001,.01,.1,1.):
        for leaves in ((0,) if kind == 'LINEAR' else (3,7)):
            if kind == 'LINEAR':
                model = linear(x, y, w, ridge)
            else:
                estimator = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05,
                    max_leaf_nodes=leaves, min_samples_leaf=20, l2_regularization=ridge,
                    early_stopping=False, monotonic_cst=[1,1], random_state=202609092)
                estimator.fit(x, y, sample_weight=w*len(x))
                model = {'kind': 'TREE', 'estimator': estimator, 'leaves': leaves}
            z = r.raw_predict(model, test)
            value = float(tw@(np.logaddexp(0.,z)-label*z))
            models.append(model)
            trials.append({'ridge': ridge, 'leaves': leaves, 'logloss': value})
    win = min(range(len(trials)), key=lambda k: (trials[k]['logloss'], trials[k]['leaves'], -trials[k]['ridge']))
    labelstate = 'global' if state is None else str(state)
    path = ROOT/f'calibration/{dep}/{seed}/{kind}_{labelstate}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        pickle.dump(models[win], file)
    n.write_csv(path.with_suffix('.GRID.csv'), trials)
    spec = {'file': str(path.relative_to(ROOT)), 'sha256': n.sha(path), 'state': state,
        'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(), 'partition_log_ratio': offset,
        'cap': float(np.log(yy.sum()+1)), 'fit_sources': int(y.sum()),
        'tune_sources': int(label.sum()), 'selected': trials[win]}
    print('MINIMAL_JOINT_CALIBRATED', dep, seed, kind, labelstate, trials[win], flush=True)
    return spec


def apply(x, spec):
    path = ROOT/spec['file']
    if str(path) not in CACHE:
        if n.sha(path) != spec['sha256']:
            raise RuntimeError('Classifier changed')
        with path.open('rb') as file:
            CACHE[str(path)] = pickle.load(file)
    raw = r.raw_predict(CACHE[str(path)], x)+spec['partition_log_ratio']
    outside = ((x < np.asarray(spec['minimum'])-1e-7)|(x > np.asarray(spec['maximum'])+1e-7)).any(1)
    z = raw.clip(-spec['cap'], spec['cap'])
    z[outside] = np.minimum(z[outside],0.)
    return z, outside, abs(raw) > spec['cap']


def infer(frame, config):
    old = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    if config['method'] == METHODS[0]:
        return co.BASE_INFER(frame, {**old, 'method': METHODS[0]})
    x, state = features(frame), state_model.states(frame)
    result = np.empty(len(frame)), np.empty(len(frame),bool), np.empty(len(frame),bool)
    for spec in config['experts']:
        mask = np.ones(len(frame),bool) if spec['state'] is None else state == spec['state']
        if mask.any():
            for dest, src in zip(result, apply(x[mask], spec)):
                dest[mask] = src
    return result


def calibrate():
    configs, audits, units = [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            pop = base.population(dep,seed)
            panels = {}
            for side, (x,y,w,_,i,j) in base.panels(dep,seed).items():
                panels[side] = (np.column_stack([x[:,0],x[:,1]+x[:,2]]), y, w,
                    pop['active'][i].astype(int)+pop['active'][j].astype(int))
            old = h.isolated.ORIGINALS[dep,seed]
            frame = ctrl.panel(dep,seed,'validation')
            cbase = {**old,'method': METHODS[0]}
            reference = infer(frame,cbase)[0]
            archived = pd.read_parquet(r.PRIOR/f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(reference,archived.waveform_score.to_numpy()):
                raise RuntimeError('NODUP waveform changed')
            configs.append(cbase)
            fm = n.cf.fast_metrics(frame,n.cf.channels(frame,reference)@np.asarray(old['weights']))
            wm = n.cf.fast_metrics(frame,reference)
            for partition in (False,True):
                for kind in ('LINEAR','TREE'):
                    experts = [fit_one(dep,seed,kind,s,panels) for s in ((0,1,2) if partition else (None,))]
                    method = 'JOINT2-'+('STATE-' if partition else 'GLOBAL-')+kind
                    config = {**old,'method': method,'experts': experts}
                    z,oo,cl = infer(frame,config)
                    metric = n.cf.fast_metrics(frame,n.cf.channels(frame,z)@np.asarray(old['weights']))
                    waveform = n.cf.fast_metrics(frame,z)
                    config['tune_guard'] = n.cf.guard(metric,fm) and n.cf.guard(waveform,wm)
                    config['tune_metrics'] = metric
                    poison = frame.copy()
                    for field in ('pair_key','pe_mc_bhattacharyya_coefficient','official_po_fpp',
                                  'profile_log_mass_BC','profile_log_conditional_BC','profile_negative_mass_distance',
                                  'profile_mean_logmass','profile_log_pooled_width'):
                        poison[field] = -999.
                    changed = {**config,'gamma':-999.,'beta':999.,'alpha':-999.,'old_Mc_weight':999.}
                    if not np.array_equal(z,infer(poison,changed)[0]):
                        raise RuntimeError('Forbidden or removed feature changed score')
                    configs.append(config)
                    audits.append({'deployment':dep,'seed':seed,'method':method,'tune_guard':config['tune_guard'],
                        'OOD_fraction':float(oo.mean()),'clip_fraction':float(cl.mean()),**metric})
                    units.append({'deployment':dep,'seed':seed,'method':method,'baseline_delta':0.,
                        'removed_feature_delta':0.,'PE_official_delta':0.,'old_coeff_delta':0.})
            base.MEMO.clear()
    n.write_csv(ROOT/'tables/VALIDATION_GUARDS.csv',audits)
    n.write_csv(ROOT/'audit/INVARIANCE.csv',units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_or_test_selection':True})


def install_export():
    old_export, old_consensus = n.public_frame,n.dev.BASE.consensus_real
    def export(frame,z,weights,method):
        out = old_export(frame,z,weights,method)
        use = method not in (METHODS[0],n.BASELINE)
        out['actual_waveform_joint_BC'] = frame.joint_BC.to_numpy() if use else frame.parent_joint_BC.to_numpy()
        if method == n.BASELINE:
            out['actual_waveform_joint_BC'] = np.nan
        out['single_waveform_classifier_used'] = use
        out['waveform_quality_state'] = state_model.states(frame)
        return out
    def consensus(items,method):
        out = old_consensus(items,method)
        extra = pd.concat(items).groupby('pair_key')[list(FIELDS)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra,on='pair_key',validate='one_to_one')
    n.public_frame,n.dev.BASE.consensus_real = export,consensus


def main():
    global ROOT,DATA,PREDICTIVE,SELECTION
    parser=argparse.ArgumentParser()
    for name in ('root','data-root','predictive-root','selection-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','prepare','calibrate','evaluate','real'),required=True)
    args=parser.parse_args()
    ROOT,DATA,PREDICTIVE,SELECTION=args.root,args.data_root,args.predictive_root,args.selection_root
    ctrl.ROOT,ctrl.SELECTION=ROOT,SELECTION
    engine.ROOT,engine.DATA,engine.PREDICTIVE=ROOT,DATA,PREDICTIVE
    base.ROOT,base.DATA=ROOT,DATA
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT
    r.install()
    install_export()
    co.score.matrices=co.matrices
    base.population=engine.population
    n.METHODS,n.load_panel,n.infer=METHODS,ctrl.panel,infer
    if args.stage=='freeze':
        freeze()
    elif args.stage=='prepare':
        ctrl.prepare()
    elif args.stage=='calibrate':
        calibrate()
    else:
        n.run(ROOT,args.stage)


if __name__=='__main__':
    main()
