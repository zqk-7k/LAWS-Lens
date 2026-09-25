#!/usr/bin/env python3
"""Read-only source/noise identity and feature provenance checks."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_intrinsic_grid_20260907 as g

e,dev=g.e,g.dev


def run(root):
    sources=[];manifest=[];seen=set()
    def record(p):
        if p in seen:return
        seen.add(p);manifest.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    for dep in e.old.DEPS:
        t=g.mass.population.metadata(e.PREVIOUS,dep,'train')
        v=g.mass.population.metadata(e.PREVIOUS,dep,'validation')
        physics=['m1_det','m2_det','a1','a2','tilt1','tilt2','theta_jn','phi12','phijl','phase','ra','dec']
        th=set(pd.util.hash_pandas_object(t[physics],index=False));vh=set(pd.util.hash_pandas_object(v[physics],index=False))
        source_overlap=len(set(t.source_uid)&set(v.source_uid));noise_overlap=len(set(t.noise_bank_index)&set(v.noise_bank_index))
        assert source_overlap==noise_overlap==len(th&vh)==0
        original=pd.read_csv(e.PREVIOUS/f'expanded_data/{dep}/noise/noise_manifest.csv')
        extra=pd.read_csv(e.PREVIOUS/f'additional_population/expanded_data/{dep}/noise/noise_manifest.csv')
        fit=pd.concat([original[original.split=='train'],extra]);val=original[original.split=='validation']
        assert len(val)>0
        gps_overlap=sum(bool(((r.start_gps<val.end_gps+16)&(r.end_gps>val.start_gps-16)).any()) for r in fit.itertuples())
        assert gps_overlap==0
        sources.append({'deployment':dep,'training_sources':t.source_uid.nunique(),'development_sources':v.source_uid.nunique(),
                        'training_noise_blocks':t.noise_bank_index.nunique(),'development_noise_blocks':v.noise_bank_index.nunique(),
                        'source_overlap':source_overlap,'exact_physical_source_overlap':len(th&vh),'noise_id_overlap':noise_overlap,
                        'noise_GPS_overlap_with_16s_guard':gps_overlap,'lens_environment_reuse_allowed':True,
                        'source_draws_not_independent_new_lens_environments':True})
        for folder in (e.PREVIOUS/f'expanded_data/{dep}',e.PREVIOUS/f'additional_population/expanded_data/{dep}'):
            for name in ('source_plan.parquet','event_metadata.parquet','noise_manifest.csv'):
                for p in folder.rglob(name):record(p)
        for folder in (e.PREVIOUS/f'expanded_encoder/features/{dep}',e.PREVIOUS/f'fine_mass_context/features/{dep}',
                       e.PREVIOUS/f'mixture_density/features/{dep}',e.PREVIOUS/f'dense_context/features/{dep}',
                       e.PREVIOUS/f'additional_population/expanded_encoder/features/{dep}',
                       e.PREVIOUS/f'additional_population/features/{dep}',
                       e.old.TRAINED/f'cache/deployment_event_psd/{dep}'):
            for p in folder.glob('*.npy'):record(p)
        p=e.PREVIOUS/f'additional_population/features/{dep}/DATA_SCALING_GATE.json'
        record(p);target=root/'audit/source_provenance';target.mkdir(exist_ok=True)
        shutil.copy2(p,target/f'{dep}_PARENT_DATA_SCALING_GATE.json')
    dev.csv_write(root/'audit/SOURCE_NOISE_DISJOINT_RECHECK.csv',pd.DataFrame(sources))
    dev.csv_write(root/'manifest/FROZEN_FEATURE_INPUT_SHA256.csv',pd.DataFrame(manifest))
    dev.json_write(root/'audit/DATA_PROVENANCE_LIMITS.json',{
        'pass_source_noise_development_isolation':True,'new_data_generated_this_round':False,
        'O3_response_epochs':'Inherited cumulative O1-O3 response calendar; noise O3-only; unchanged in this waveform-only study',
        'image_strength':'Inherited per-image optimal target-SNR scaling,not new response-derived ratio validation',
        'BAYESTAR':'Inherited conditional fast-sky products,not new full-PE posterior runs',
        'ensemble_dependence':'One uniform three-model ensemble shared across three frozen evaluation seeds; SD is not three independently retrained ensembles',
        'external_results':'Adaptive real development,not blind confirmation'})
    print(pd.DataFrame(sources).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
