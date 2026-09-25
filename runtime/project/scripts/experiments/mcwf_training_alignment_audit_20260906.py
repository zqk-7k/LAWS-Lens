#!/usr/bin/env python3
"""Read-only checks that cached soft mass labels follow metadata rows."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as tf
import mcwf_unified_waveform_20260906 as u


def run(output):
    rows=[]
    for dep in u.DEPS:
        for split in ('train','validation'):
            meta=pd.read_parquet(u.PREV/f'cache/{dep}/{split}_metadata.parquet')
            raw=np.load(u.PREV/f'cache/{dep}/{split}_raw2s.npy',mmap_mode='r')
            y=np.load(body.PREVIOUS/f'cache/masstf/{dep}/{split}_targets.npy')
            x=np.load(body.PREVIOUS/f'cache/phasebank/{dep}/{split}_features.npy',mmap_mode='r')
            mass=meta.chirp_mass_detector.to_numpy(float)
            expected=(meta.mass_1_detector.to_numpy()*meta.mass_2_detector.to_numpy())**.6/(meta.mass_1_detector.to_numpy()+meta.mass_2_detector.to_numpy())**.2
            pred=np.exp(y@tf.LOG_CENTERS)
            row={'deployment':dep,'split':split,'n_rows':len(meta),'n_sources':meta.waveform_parent_uid.nunique(),
                'raw_shape':list(raw.shape),'target_shape':list(y.shape),'feature_shape':list(x.shape),
                'chirpmass_relative_error_max':float(np.max(np.abs(expected/mass-1))),
                'target_mass_Spearman':float(spearmanr(pred,mass).statistic),
                'target_logmass_absolute_error_q99':float(np.quantile(np.abs(np.log(pred/mass)),.99)),
                'target_sum_max_error':float(np.max(np.abs(y.sum(1)-1))),
                'label_range_Msun':[float(mass.min()),float(mass.max())]}
            if not (len(raw)==len(y)==len(x)==len(meta)) or row['target_mass_Spearman']<.999 or row['chirpmass_relative_error_max']>1e-8:
                raise RuntimeError(str(row))
            rows.append(row)
    dev.json_write(output,{'row_alignment_pass':True,'no_original_files_modified':True,'tables':rows})
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
