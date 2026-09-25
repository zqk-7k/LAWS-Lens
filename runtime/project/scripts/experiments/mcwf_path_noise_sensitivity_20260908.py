#!/usr/bin/env python3
"""Frozen data-quality sensitivity, not a replacement confirmation set."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import shutil
import numpy as np
import pandas as pd
P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_path_fresh_confirmation_v2_20260908 as run
import mcwf_summarize_20260905 as bootstrap
dev,t=run.dev,run.t


def freeze(root):
    path=root/'contracts/NONOVERLAP_SENSITIVITY_FREEZE.json'
    if path.exists():
        if json.loads(path.read_text())['code_sha256']!=dev.sha(Path(__file__)):raise RuntimeError('Code changed')
        return
    remove=[]
    for dep in t.DEPS:
        noise=pd.read_csv(root/f'confirmation/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        for cs in run.CATALOGS:
            events=pd.read_parquet(root/f'confirmation/{dep}/catalog_{cs}/event_manifest.parquet')
            events['noise_start']=noise.loc[events.noise_bank_index,'start_gps'].to_numpy()+events.noise_offset_samples/4096
            for uid,g in events.groupby('source_uid'):
                if len(g)==2 and abs(float(g.noise_start.iloc[0]-g.noise_start.iloc[1]))<26:
                    remove.append({'deployment':dep,'catalog_seed':cs,'source_uid':uid})
    dev.json_write(path,{'UTC':datetime.now(timezone.utc).isoformat(),'code_sha256':dev.sha(Path(__file__)),
        'rule':'Exclude both images of true source if actual two26snoise intervals overlap.Induce complete subcatalog and remapindices;no rescoring orcalibration.',
        'excluded':remove,'pre_data_quality_defined':True,'excluded_by_PE_or_performance':False,
        'primary_test_remains_full':True,'no_reselection':True,'additional_diagnostic_not_new_independent_data':True})
    shutil.copy2(__file__,root/'scripts/path_noise_sensitivity.py')


def evaluate(root):
    run.verify(root);freeze(root)
    contract=json.loads((root/'contracts/NONOVERLAP_SENSITIVITY_FREEZE.json').read_text())
    rows=[]
    for dep in t.DEPS:
        for cs in run.CATALOGS:
            cat=root/f'confirmation/{dep}/catalog_{cs}'
            events=pd.read_parquet(cat/'event_manifest.parquet')
            excluded={r['source_uid'] for r in contract['excluded'] if r['deployment']==dep and r['catalog_seed']==cs}
            keep=np.flatnonzero(~events.source_uid.isin(excluded).to_numpy())
            mapping={int(k):int(i) for i,k in enumerate(keep)}
            for slot in t.MODEL_SLOTS:
                selected=[]
                for name in ('OMC','PATH25'):
                    f=pd.read_parquet(cat/f'model_{slot}/{name}_pairs.parquet')
                    f=f[f.idx_i.isin(keep)&f.idx_j.isin(keep)].copy()
                    f['idx_i']=f.idx_i.map(mapping).astype(int);f['idx_j']=f.idx_j.map(mapping).astype(int)
                    f['event_count']=len(keep)
                    if len(f)!=len(keep)*(len(keep)-1)//2:raise RuntimeError('Incomplete induced subcatalog')
                    selected.append(f)
                b,c=selected
                for mode in ('waveform','fusion'):
                    col='waveform_score' if mode=='waveform' else 'final_score'
                    bm,cm=dev.BASE.full_metrics(b,b[col]),dev.BASE.full_metrics(c,c[col])
                    ci,_=bootstrap.ranks_bootstrap(c,c[col].to_numpy(float),b[col].to_numpy(float),cs+slot,repeats=10000)
                    rows.append({'deployment':dep,'catalog_seed':cs,'model_seed':slot,'mode':mode,'removed_systems':len(excluded),
                        'remaining_events':len(keep),'point_guard_pass':run.cf.guard(cm,bm),**cm,**ci,
                        **{'baseline_'+k:bm[k] for k in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9')}})
    f=pd.DataFrame(rows);dev.csv_write(root/'tables/NONOVERLAP_SENSITIVITY.csv',f)
    metrics=['macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9']
    a=f.groupby(['deployment','model_seed','mode'])[metrics+['baseline_'+k for k in metrics]].mean().reset_index()
    a['mean_guard_pass']=[run.cf.guard(row,{k:row['baseline_'+k] for k in metrics}) for row in a.to_dict('records')]
    dev.csv_write(root/'tables/NONOVERLAP_MODEL_GUARDS.csv',a)
    dev.json_write(root/'contracts/NONOVERLAP_SENSITIVITY_COMPLETE.json',{'excluded_systems':len(contract['excluded']),
        'model_run_mode_mean_guards_pass':bool(a.mean_guard_pass.all()),'primary_test_not_changed':True,'no_reselection':True,
        'additional_check_not_independent_of_primary':True})
    print(json.dumps({'nonoverlap_guard_pass':bool(a.mean_guard_pass.all()),'excluded':len(contract['excluded'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['freeze','evaluate'],required=True)
    a=p.parse_args();globals()[a.stage](a.root)
