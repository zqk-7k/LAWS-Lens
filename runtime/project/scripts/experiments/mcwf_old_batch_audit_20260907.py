#!/usr/bin/env python3
"""Separate row/preprocessing identity from mixed-precision batch effects."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_development_20260905 as d
import mcwf_unified_waveform_20260906 as u
root=d.PROJECT/'results/mcwf_unified_old_batch_audit_20260907'
root.mkdir(exist_ok=False)
rows=[]
for dep,es in [('gwtc3',202607241),('gwtc3',202607243),('gwtc4',202607241),('gwtc4',202607243)]:
    ms=(202609061,202609062,202609063)[d.SEEDS.index(es)]
    plan=d.BASE.retained_event_plan(dep,es,'test')
    full=d.ORCH.event_array_for_plan(dep,es,'test',plan)
    f=u.frame(dep,ms,es,'test')
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    checkpoint=d.V7/dep/f'seed_{es}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt'
    model,_=d.BASE.v7.load_unified_model(checkpoint)
    cal=json.loads((d.V7/dep/f'seed_{es}/results/waveform_channel_calibration_v7.json').read_text())
    computed=[]
    for bs in (16,24,32):
        z,p=d.BASE.v7.embed_catalog(model,d.BASE.v7.ArrayCatalog(full),batch_size=bs)
        g=f.copy()
        g['waveform_embedding_cosine']=np.sum(z[i]*z[j],axis=1)
        g['waveform_abs_delta_logmc_std']=np.abs(p[i,0]-p[j,0])
        g['waveform_abs_delta_logitq_std']=np.abs(p[i,1]-p[j,1])
        g=d.BASE.v7.apply_waveform_channel(g,cal)
        score=g.waveform_score.to_numpy()
        computed.append(score)
        rows.append({'deployment':dep,'eval_seed':es,'batch_size':bs,'max_abs_delta_stored':float(np.max(abs(score-f.previous_waveform_score))),
            'q99_abs_delta_stored':float(np.quantile(abs(score-f.previous_waveform_score),.99)),
            'max_cosine_delta_stored':float(np.max(abs(g.waveform_embedding_cosine-f.waveform_embedding_cosine))),
            'max_mass_gap_delta_stored':float(np.max(abs(g.waveform_abs_delta_logmc_std-f.waveform_abs_delta_logmc_std)))})
    devrow={'deployment':dep,'eval_seed':es,'batch_size':'batch_range','max_abs_delta_stored':float(np.ptp(computed,axis=0).max())}
    rows.append(devrow)
    del model
d.csv_write(root/'BATCH_RECOMPUTATION_AUDIT.csv',pd.DataFrame(rows))
d.json_write(root/'INTERPRETATION.json',{'source_arrays':'same family/image/source_index arrays as frozen BAYESTAR baseline',
    'preprocessing_difference':'legacy uses one global polarity flip across H1L1; new RAW uses independent detector peak polarity; both use the same2s4096samples. This deliberate old/new preprocessing difference is identical for O3 and O4a, not event misalignment.',
    'baseline_policy':'stored waveform evidence is never replaced by these diagnostic recomputations; new source/noise confirmation uses a fixed batch16 old baseline for both arms',
    'initial_audit_note':'initial equality gate compared different old/new polarity conventions and is retained as originally emitted; not evidence of shuffled events',
    'precision':'legacy inference uses bfloat16; batch-size variability is measured rather than assumed absent'})
print(pd.DataFrame(rows).to_string(index=False),flush=True)
