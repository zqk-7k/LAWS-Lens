#!/usr/bin/env python3
"""Recompute frozen old waveform evidence from the exact new-model inputs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as d
import mcwf_encoder_body_20260906 as body
import mcwf_unified_waveform_20260906 as u
import mcwf_fresh_confirmation_20260906 as old

torch.set_num_threads(2)
out=d.PROJECT/'results/mcwf_unified_deployment_input_audit_20260907'
out.mkdir(exist_ok=False)
rows=[]
for dep in u.DEPS:
    for ms,es in zip(body.MODEL_SEEDS,d.SEEDS):
        for split in ('validation','test'):
            plan=d.BASE.retained_event_plan(dep,es,split)
            full=d.ORCH.event_array_for_plan(dep,es,split,plan)
            frame=u.frame(dep,ms,es,split)
            newview=d.TRAIN.make_window_view(full,2)
            oldview=np.stack([d.BASE.v7.pilot.prepare(v,None,False) for v in full])
            recompute=old.old_waveform(dep,es,full,frame)
            fields=[c for c in recompute if 'waveform' in c]
            target='waveform_score' if 'waveform_score' in recompute else 'waveform_lr_score'
            if target not in recompute:
                raise RuntimeError(fields)
            residual=recompute[target].to_numpy(float)-frame.previous_waveform_score.to_numpy(float)
            row={'deployment':dep,'model_seed':ms,'eval_seed':es,'split':split,
                 'n_events':len(plan),'n_pairs':len(frame),'input_shape':str(full.shape),
                 'max_preprocess_abs_delta':float(np.max(np.abs(newview-oldview))),
                 'max_old_waveform_abs_delta':float(np.max(np.abs(residual))),
                 'old_waveform_delta_q99':float(np.quantile(np.abs(residual),.99)),
                 'allclose_old_waveform_1e-5':bool(np.allclose(recompute[target],frame.previous_waveform_score,atol=1e-5,rtol=1e-5)),
                 'idx_within_plan':bool(frame[['idx_i','idx_j']].to_numpy().max()<len(plan)),
                 'array_input_root':str(d.ORCH.INPUT_ROOT)}
            rows.append(row)
            print(json.dumps(row),flush=True)
            dest=out/dep/f'seed_{es}'
            dest.mkdir(parents=True,exist_ok=True)
            plan.to_parquet(dest/f'{split}_event_plan.parquet',index=False)
            pd.DataFrame({'idx_i':frame.idx_i,'idx_j':frame.idx_j,'old_stored':frame.previous_waveform_score,
                          'old_recomputed':recompute[target],'delta':residual}).to_parquet(dest/f'{split}_score_check.parquet',index=False)
            d.csv_write(out/'INPUT_AND_FROZEN_SCORE_AUDIT.csv',pd.DataFrame(rows))
d.json_write(out/'AUDIT_GATE.json',{'pass':all(r['allclose_old_waveform_1e-5'] and r['max_preprocess_abs_delta']==0 and r['idx_within_plan'] for r in rows),'rows':rows,'claim':'Tests waveform row mapping and preprocessing equivalence only, not physical calibration.'})
