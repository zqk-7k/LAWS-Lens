#!/usr/bin/env python3
"""Integrate a simulation-selected spectral covariate into one joint density."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import numpy as np

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_profile_spectral_consistency_v2_20260909 as spectral
import mcwf_quality_state_joint_calibration_20260909 as state
engine,base,n,r,co,h=state.engine,state.base,state.n,state.r,state.co,state.h
ROOT=DATA=PREDICTIVE=SPECTRAL=PARENT=None
WORKERS=24
METHODS=('NODUP-DIRECT-REPLAY','SPECTRAL-JOINTSTATE-GLOBAL','SPECTRAL-JOINTSTATE-REJECT-ONLY')
FIELDS=state.FIELDS+('spectral_fraction_i','spectral_fraction_j','spectral_update_i','spectral_update_j')
WORK={}


def freeze():
    receipt=json.loads((SPECTRAL/'contracts/PREDICTIVE_FROZEN.json').read_text())
    src=SPECTRAL/'calibration/SPECTRAL_PREDICTIVE.json'
    if not receipt['both_runs_pass']or n.sha(src)!=receipt['sha256']:
        raise RuntimeError('Both-run predictive gate or hash failed')
    base.freeze()
    n.write_json(ROOT/'contracts/SPECTRAL_SCORE_CONTRACT.json',{
        'UTC':n.utc(),'id':'MCWF-NODUP-SPECTRAL-JOINT-SCORE-44','status':n.STATUS,
        'overrides_base':'No seven-feature classifier;same R35 quality-state jointBC calibration. Only mass-predictive scale covariate changes.',
        'spectral_source':str(SPECTRAL),'spectral_sha256':receipt['sha256'],
        'exponent':receipt['selected_exponent'],'bins':receipt['selected_bins'],
        'selection':'Already frozen by independent simulated predictive loss before scoring. No new candidate-specific hyperparameter search.',
        'data':str(DATA),'predictive_source':str(PREDICTIVE),'R29_scoring_parent':str(PARENT),
        'density':'Replace R29 mass marginal only where R29 active AND new spectral-width in independent fit support. Keep frozen NN eta/chi conditional. ONE jointBC,not another mass score.',
        'empirical_not_PE':'PyCBC spectral fraction on conditioned16s data with flat PSD;not formal search chi-square,pvalue or optimalSNR.',
        'integration':'Same512massbins as R35;no claim of continuous integration.',
        'fallback':'Exact R29 selected prediction for unavailable/OOD newcovariate. R29 retains its own exact R10/NN fallback.',
        'fit':'Same independent source/noise fitfold0;source-pairweights,three R10 profile states,statefrequencyoffset,one isotonic jointBC calibration.',
        'fixed':['encoders','profile optima','cosine calibration','gamma','beta','outerweights','time','sky','scope','oldresults'],
        'no_old_encoder_Mc_q':True,'no_total_blend':True,'same_both_runs':True,
        'methods':list(METHODS),'no_real_PE_or_official_selection':True,
        'real_only_after_simulation_evaluation':True,'adaptive_development':True,
        'new_blind_confirmation':False,'goal_achieved':False})
    cfg=json.loads(src.read_text());e=receipt['selected_exponent'];bins=receipt['selected_bins']
    selected={'UTC':n.utc(),'selected':{'kind':'SPECTRAL_BOUNDED','exponent':e,'bins':bins},
        'specs':{dep:cfg[f'{dep}_B{bins}_E{e:g}']['spec']for dep in n.DEPS},
        'source':str(src),'predictive_sha256':receipt['sha256'],'real_or_test_used':False}
    dest=ROOT/'configs/SELECTED_PREDICTIVE_KIND.json'
    n.write_json(dest,selected)
    n.write_json(ROOT/'contracts/PREDICTIVE_KIND_FROZEN.json',{'UTC':n.utc(),
        'file':str(dest.relative_to(ROOT)),'sha256':n.sha(dest)})
    paths=[Path(__file__),Path(spectral.__file__),Path(state.__file__),src,
           PARENT/'configs/SELECTED_PREDICTIVE_KIND.json']
    n.write_csv(ROOT/'manifest/SPECTRAL_INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),
        'bytes':p.stat().st_size}for p in paths])
    shutil.copy2(__file__,ROOT/'scripts/spectral_joint_scoring.py')


def worker_init(folder):
    co.app.worker_init(folder)
    WORK.update(co.app.STATE)


def event(job):
    index,row=job
    model=spectral.replay.prof.profile.LowBand(WORK['full20'][index],WORK['frequency'],WORK['psd'][index],16)
    result=spectral.statistic(model,np.array([row['logmc'],row['q'],row['chieff_equal']]))
    result['row_index']=int(row['row_index'])
    return result


def fractions(dep,seed,split,catalog,active):
    tag=co.app.tag_for(seed,split,catalog)
    folder=co.PARENT/f'cache/profile_inputs/{dep}/{tag}'
    out=ROOT/f'spectral_events/{dep}/{tag}';out.mkdir(parents=True,exist_ok=True)
    quality=np.full(len(active),np.nan)
    if not active.any():
        return quality
    ids=np.load(folder/'event_ids.npy')
    if set(np.flatnonzero(active))-set(ids.tolist()):
        raise RuntimeError('Missing frozen waveform inputs')
    jobs=[]
    for k,idx in enumerate(ids):
        if active[idx]and not (out/f'{idx}.json').exists():
            row=json.loads((co.PARENT/f'profile_events/{dep}/{tag}/{idx}.json').read_text())
            jobs.append((k,row))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(WORKERS,len(jobs)),mp_context=mp.get_context('spawn'),
            initializer=worker_init,initargs=(str(folder),))as pool:
            for row in pool.map(event,jobs):
                n.write_json(out/f'{row["row_index"]}.json',row)
        print('SPECTRAL_PANEL',dep,tag,len(jobs),flush=True)
    bins=engine.selected()['selected']['bins']
    for idx in np.flatnonzero(active):
        row=json.loads((out/f'{idx}.json').read_text())
        if row['spectral_valid']:
            quality[idx]=row[f'fraction_{bins}']
    return quality


def updated_mass(dep,seed,split,catalog=None):
    slot=n.recipes()[dep,seed]['slot'];tag=co.app.tag_for(seed,split,catalog)
    dest=ROOT/f'predictions/{dep}/{slot}_{tag}.npz'
    if dest.exists():
        return dict(np.load(dest))
    if split=='real'and not (ROOT/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Simulation evaluation required before real waveform processing')
    path=PARENT/f'predictions/{dep}/{slot}_{tag}.npz'
    if not path.exists():
        raise RuntimeError('Missing frozen R29 panel prediction:'+str(path))
    mass=dict(np.load(path));choice=engine.selected();spec=choice['specs'][dep]
    quality=fractions(dep,seed,split,catalog,mass['predictive_active'])
    width=mass['predictive_width']*quality**choice['selected']['exponent']
    active=mass['predictive_active']&np.isfinite(width)&(width>=spec['minimum_h'])&(width<=spec['maximum_h'])
    p=mass['p'].copy()
    p[active]=spectral.h.density(mass['predictive_center'][active],width[active],spec)
    if not np.array_equal(p[~active],mass['p'][~active],equal_nan=True):
        raise RuntimeError('R29 fallback changed')
    valid=np.isfinite(p).all(1)
    if abs(p[valid].sum(1)-1).max()>1e-10:
        raise RuntimeError('Mass predictive normalization failed')
    result={**mass,'p':p,'spectral_active':active,'spectral_fraction':quality,
            'spectral_width':width,'predictive_active':active}
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,**result)
    n.write_json(dest.with_suffix('.json'),{'UTC':n.utc(),'parent_sha256':n.sha(path),
        'new_active':int(active.sum()),'R29_active':int(mass['predictive_active'].sum()),
        'exact_fallback':True,'selected_model_sha256':n.sha(ROOT/'configs/SELECTED_PREDICTIVE_KIND.json')})
    return result


def population(dep,seed):
    key=dep,seed
    if key in base.MEMO:
        return base.MEMO[key]
    slot=n.recipes()[key]['slot'];path=DATA/f'predictions/{dep}/{slot}_parent.npz'
    p=np.load(path)['p'].copy()
    previous=dict(np.load(PREDICTIVE/f'predictions/{dep}_BOUNDED_development.npz'))
    oldspec=json.loads((co.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    active=previous['profile_active']
    p[active]=engine.pred.cal.density(previous['profile_center'][active],oldspec)
    p[previous['active']]=previous['p'][previous['active']]
    c=engine.selected()['selected'];new=dict(np.load(SPECTRAL/f'predictions/{dep}_B{c["bins"]}_E{c["exponent"]:g}.npz'))
    p[new['active']]=new['p'][new['active']]
    mass={'p':p,'active':active}
    base.MEMO[key]=engine.overlaps(dep,slot,mass,path,ROOT/f'cache/{dep}_{seed}_development.npz')
    return base.MEMO[key]


def panel(dep,seed,split,catalog=None):
    frame=state.ctrl.panel(dep,seed,split,catalog)
    mass=updated_mass(dep,seed,split,catalog)
    for suffix,idx in [('i',frame.idx_i.to_numpy(int)),('j',frame.idx_j.to_numpy(int))]:
        frame['spectral_fraction_'+suffix]=mass['spectral_fraction'][idx]
        frame['spectral_update_'+suffix]=mass['spectral_active'][idx]
    return frame


def main():
    global ROOT,DATA,PREDICTIVE,SPECTRAL,PARENT,WORKERS
    parser=argparse.ArgumentParser()
    for arg in ('root','data-root','predictive-root','spectral-root','parent-root'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args();ROOT,DATA,PREDICTIVE,SPECTRAL,PARENT=args.root,args.data_root,args.predictive_root,args.spectral_root,args.parent_root
    WORKERS=args.workers
    state.ROOT,state.DATA,state.PREDICTIVE=ROOT,DATA,PREDICTIVE
    state.ctrl.ROOT=ROOT
    engine.ROOT,engine.DATA,engine.PREDICTIVE=ROOT,DATA,PREDICTIVE
    base.ROOT,base.DATA=ROOT,DATA
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT
    r.install()
    state.METHODS=METHODS;state.FIELDS=FIELDS
    state.install_export()
    co.score.matrices=co.matrices
    engine.updated_mass=updated_mass;engine.population=population;base.population=population
    n.METHODS,n.load_panel,n.infer=METHODS,panel,state.infer
    if args.stage=='freeze':
        freeze()
    elif args.stage=='calibrate':
        state.calibrate()
    else:
        n.run(ROOT,args.stage)


if __name__=='__main__':
    main()
