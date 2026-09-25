#!/usr/bin/env python3
"""Read-only replay of the selected predictive evidence and OOD policy."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_path_release_v3_20260908 as r
import mcwf_joint_intrinsics_evidence_20260908 as joint
dev,t,cf=r.dev,r.t,r.cf
JOINT=P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'


def run(root):
    if (root/'contracts/MECHANISM_AUDIT_COMPLETE.json').exists():raise RuntimeError('Already complete')
    selected=json.loads((root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json').read_text())
    outer=json.loads((Path(selected['upstream'])/'calibration/SELECTED.json').read_text())
    inner=json.loads((JOINT/'calibration/SELECTED.json').read_text())
    summaries=[]
    for cfg in [c for c in outer if c['method']==selected['upstream_method']]:
        dep,seed=cfg['deployment'],cfg['seed']
        c=next(x for x in inner if x['deployment']==dep and x['seed']==seed and x['method']==cfg['waveform_model'])
        for split in ('validation','test','real'):
            cache=JOINT/f"cache/{c['kind']}/{dep}/{c['slot']}_{seed}_{split}.npz"
            if not cache.exists():raise RuntimeError('Do not create upstream cache during audit')
            f,z,a=joint.scored(JOINT,c,split)
            name='real_fusion_pairs.parquet' if split=='real' else split+'_pairs.parquet'
            expected=pd.read_parquet(Path(selected['upstream'])/f"evaluation/{cfg['method']}/{dep}/seed_{seed}/{name}")
            keys=['pair_key'] if split=='real' else ['idx_i','idx_j']
            expected=expected.set_index(keys).reindex(f.set_index(keys).index)
            error=float(np.max(abs(z-expected.waveform_score.to_numpy(float))))
            if error>1e-10:raise RuntimeError('Frozen joint evidence replay mismatch')
            audit=f[[c for c in ('pair_key','event_i','event_j','idx_i','idx_j','is_true_pair') if c in f]].copy()
            for k,v in a.items():audit[k]=v
            audit['joint_waveform_score']=z
            dest=root/f'mechanism_audit/{dep}/seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            audit.to_parquet(dest/f'{split}_joint_features.parquet',index=False)
            populations=[('all_real',np.ones(len(f),bool))] if split=='real' else [('true',f.is_true_pair.to_numpy(bool)),('null',~f.is_true_pair.to_numpy(bool))]
            for label,take in populations:
                summaries.append({'deployment':dep,'seed':seed,'split':split,'population':label,'n_pairs':int(take.sum()),
                    'joint_ood_fraction':float(np.mean(a['joint_ood'][take])),
                    'joint_tail_below_0p05_fraction':float(np.mean(a['joint_tail_probability'][take]<.05)),
                    'positive_increment_fraction':float(np.mean(a['increment'][take]>0)),
                    'increment_abs4_fraction':float(np.mean(abs(a['increment'][take])>=4)),
                    'median_joint_BC':float(np.median(a['joint_BC'][take])),
                    'max_replay_error':error})
            spoof=f.copy()
            for column in spoof:
                if column.startswith(('pe_','official_')) or column in ('event_i','event_j','pair_key','is_true_pair'):
                    spoof[column]=0
            w=np.asarray(cfg['weights'],float);w/=w.sum()
            original,weights=r.pathmodel.mix(f,z,w,dep,seed,selected['alpha'])
            altered,_=r.pathmodel.mix(spoof,z,w,dep,seed,selected['alpha'])
            if not np.array_equal(original,altered):raise RuntimeError('Audit labels altered score')
    dev.csv_write(root/'tables/JOINT_MECHANISM_OOD_SUMMARY.csv',pd.DataFrame(summaries))
    shutil.copy2(__file__,root/'scripts/path_mechanism_audit.py')
    dev.json_write(root/'contracts/MECHANISM_AUDIT_COMPLETE.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'selected_evidence_replay_pass':True,'PE_official_ID_columns_not_pair_score_inputs':True,
        'global_selection_still_used_real_outcomes':True,'upstream_cache_read_only':True,'code_sha256':dev.sha(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
