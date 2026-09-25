#!/usr/bin/env python3
"""Probe input alignment sensitivity on already examined simulation failures."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import json
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as b
import mcwf_mass_tf_20260905 as tf
root=dev.PROJECT/'results/mcwf_timing_sensitivity_probe_20260907'
root.mkdir(exist_ok=False)
rows=[]
for dep,es,ids in [('gwtc3',202607242,[6,41]),('gwtc3',202607243,[9,44]),('gwtc4',202607243,[70,105])]:
    ms=b.MODEL_SEEDS[dev.SEEDS.index(es)]
    plan=dev.BASE.retained_event_plan(dep,es,'test')
    full=dev.ORCH.event_array_for_plan(dep,es,'test',plan)[ids,:,-4096:]
    checkpoint=dev.PROJECT/f'results/main_o3_mcwf_encoder_v2_20260906T153600Z/models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    means=[]
    pp=[]
    zz=[]
    for shift in (-128,-64,-8,-4,0,4,8,64,128):
        p,z,ck=b.encode(checkpoint,np.roll(full,shift,axis=-1))
        lp=p@tf.LOG_CENTERS
        row={'deployment':dep,'eval_seed':es,'shift_samples':shift,'pred_logMc_i':lp[0],'pred_logMc_j':lp[1],
             'BC_pred':np.sqrt(p[0]*p[1]).sum(),'cosine':np.sum(z[0]*z[1])}
        rows.append(row)
        pp.append(p)
        zz.append(z)
    avg=np.mean(pp,axis=0)
    row={'deployment':dep,'eval_seed':es,'shift_samples':'mean9','pred_logMc_i':(avg@tf.LOG_CENTERS)[0],
         'pred_logMc_j':(avg@tf.LOG_CENTERS)[1],'BC_pred':np.sqrt(avg[0]*avg[1]).sum()}
    rows.append(row)
dev.csv_write(root/'ALREADY_EXAMINED_FAILURE_TIMING_PROBE.csv',pd.DataFrame(rows))
print(pd.DataFrame(rows).to_string(index=False),flush=True)
