#!/usr/bin/env python3
"""Re-test genuine pre-normalization scale, preserving the invalid earlier run."""
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
import mcwf_noise_scale_20260908 as experiment
import mcwf_multirate_train_20260908 as matched
from mcwf_temporal_response_20260908 import protected

dev=experiment.dev
REPLAY=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'


def source_data(root,dep,split):
    return matched.training_data(REPLAY,dep,split,'SMALL-CONTROL')


def scales(root,dep,split,seed=0):
    path=root/f'features/{dep}'/(f'{seed}_{split}.npy' if seed else f'{split}.npy')
    if path.exists():return np.load(path)
    if split in ('development','train'):
        part='validation' if split=='development' else 'train'
        source=REPLAY/f'cache/{dep}/{part}/PROGRESS.parquet'
        progress=pd.read_parquet(source).sort_values('row_index')
        _,meta=source_data(root,dep,part)
        if not np.array_equal(progress.row_index,meta.row_index):raise RuntimeError('Reconstructed scale/event alignment failed')
        if not progress.old_peak2s_exact.all():raise RuntimeError('Original input was not exactly reproduced')
        value=progress[['pre_window_normalization_std_H1','pre_window_normalization_std_L1']].to_numpy(np.float32)
        provenance={'source':str(source),'sha256':dev.sha(source),'before_make_window_view':True}
    else:
        if split=='real':
            full,events=dev.real_inputs(dep);full=full[events.strict_h1l1_preprocessing_pass.to_numpy(bool)]
        else:
            plan=dev.BASE.retained_event_plan(dep,seed,split);full=dev.ORCH.event_array_for_plan(dep,seed,split,plan)
        raw=np.asarray(full,np.float32)[...,-4096:]
        if not np.isfinite(raw).all():raise RuntimeError('Incomplete strict waveform')
        value=raw.std(-1,ddof=0)
        provenance={'source':'frozen full24 prepared arrays, last4096 samples BEFORE make_window_view','before_make_window_view':True}
    if np.any(value<=0) or not np.isfinite(value).all():raise RuntimeError('Invalid genuine pre-window scale')
    result=np.log(value).astype(np.float32)
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,result)
    dev.json_write(path.with_suffix('.sha256.json'),{'sha256':dev.sha(path),'shape':list(result.shape),**provenance})
    stats=[]
    for d,name in enumerate(('H1','L1')):
        stats.append({'deployment':dep,'seed':seed,'split':split,'detector':name,'events':len(value),
            'mean_raw_std':float(value[:,d].mean()),'log_std_SD':float(result[:,d].std()),
            'q01_raw_std':float(np.quantile(value[:,d],.01)),'median_raw_std':float(np.median(value[:,d])),
            'q99_raw_std':float(np.quantile(value[:,d],.99)),'pre_normalization_provenance':True})
    dev.csv_write(path.with_suffix('.audit.csv'),pd.DataFrame(stats))
    return result


def initialize(root):
    if root.exists():raise RuntimeError('Fresh independent output required')
    for n in ('contracts','scripts','logs','models','features','predictions','calibration','evaluation','tables','reports','manifest','figures'):(root/n).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-GENUINE-NOISE-SCALE-09','UTC':datetime.now(timezone.utc).isoformat(),'status':experiment.STATUS,'goal_achieved':False,
        'invalid_previous':'MCWF-NOISE-SCALE-02 used already z-scored raw2s and was invalidated;all originaloutputs preserved. This is a new experiment,not a correction of its numeric results.',
        'same_both_runs':True,'source_recovery':str(REPLAY),
        'data':'same4096train/512development sourceparents and96/32noiseblocks as SMALL-CONTROL,oldOMC warmstart;no 20Hz lowbandfeatures in this experiment',
        'input':'27frozenpeak2s responsechannels+log(std_H1),log(std_L1) BEFORE extra2s normalization,using exactreconstructed preparedfull24 arrays',
        'scale_definition':'ddof0 std oflast4096samples inPSDwhitened40-580Hz,antialiasprepared24s;notthealreadywindow-standardized raw2sbank',
        'not_SNR':'These are signal/noise-scale descriptors,not optimal SNR,PE network SNR or magnification',
        'isolation':'same source/noise-disjoint splits,labels onlysimulation;matchedsmall-data control availablein06',
        'training':'unchanged SCALE predictor/training code,30inputchannels withcoordinate,extra2weightszero,15epochs,threeoriginal training seeds;only correctedinputandmatcheddata subset',
        'OOD':'per-detector pre-window scale outside development min/max makes newincrementneutral;frozenbefore realdata',
        'selection':'same source/noise-half simulated calibration,validation-only candidate/retrieval priorities andguards',
        'frozen':['encoder checkpoints','peak2s waveform scores asbaseline','time','sky_raw_log_bf','outerC-fixedweights','scope','history','paper'],
        'real':'adaptive development,notblindconfirmation;PE/official appendedafterfreeze;notlabels',
        'references':['https://arxiv.org/abs/2106.12594'],'reference_scope':'noise-conditioned inference motivation only;this is not DINGO or fullPE'})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(protected()))
    shutil.copy2(__file__,root/'scripts/correct_scale.py');shutil.copy2(experiment.__file__,root/'scripts/frozen_scale_training.py')
    audits=[]
    for dep in experiment.DEPS:
        for split in ('train','development'):
            s=scales(root,dep,split)
            if np.min(s.std(0))<1e-3:raise RuntimeError('Scale descriptors still resemble post-normalization constants')
            audits.append({'deployment':dep,'split':split,'log_scale_SD_H1':float(s[:,0].std()),'log_scale_SD_L1':float(s[:,1].std()),'input_pre_normalization_pass':True})
    dev.csv_write(root/'tables/INPUT_SCALE_PROVENANCE_PASS.csv',pd.DataFrame(audits))
    dev.json_write(root/'contracts/FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),
        'source_sha256':dev.sha(Path(__file__)),'unchanged_training_sha256':dev.sha(Path(experiment.__file__))})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['initialize','train','select','evaluate','real','assess'],required=True);a=parser.parse_args()
    # Process-local API substitution leaves the archived invalid run untouched.
    experiment.scales=scales;experiment.old.data=source_data;experiment.ev.t=experiment
    if a.stage=='initialize':initialize(a.root)
    elif a.stage=='train':
        for dep in experiment.DEPS:
            for slot,seed in zip(experiment.MODEL_SLOTS,experiment.TRAIN_SEEDS):experiment.train(a.root,dep,slot,seed)
    elif a.stage=='select':experiment.select(a.root)
    else:getattr(experiment.ev,a.stage)(a.root)
