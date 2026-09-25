#!/usr/bin/env python3
"""Audit actual source parameters and global GPS blocks before confirmation claims."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import mcwf_development_20260905 as dev
import mcwf_new_confirmation_20260906 as fresh


def audit(root):
    fresh.verify(root)
    output=root/'confirmation'
    oldmass,newsource,newnoise,oldnoise,files=[],[],[],[],[]
    for dep in ('gwtc3','gwtc4'):
        banks=(dev.ORCH.SOURCE_ROOT/dep/'shared/physical_h1l1_source_bank',
               dev.OLD/f'data/source_banks/{dep}')
        for bank in banks:
            paths=sorted(bank.glob('*/*metadata.parquet'))
            if not paths:
                raise RuntimeError('Missing historical source metadata: '+str(bank))
            for path in paths:
                f=pd.read_parquet(path)
                oldmass.append(f[['mass_1_detector','mass_2_detector']].to_numpy(float))
                files.append({'path':str(path),'sha256':dev.sha(path),'rows':len(f)})
        for name in ('confirmation','confirmation_global_noise_fixed'):
            base=fresh.PREVIOUS/name/dep
            paths=sorted(base.glob('catalog_*/source_systems.parquet'))
            if not paths:
                raise RuntimeError('Missing previously opened source catalogs: '+str(base))
            for path in paths:
                f=pd.read_parquet(path)
                oldmass.append(f[['m1_det','m2_det']].to_numpy(float))
                files.append({'path':str(path),'sha256':dev.sha(path),'rows':len(f)})
        excluded=fresh.exclusions(dep)
        oldnoise.extend(excluded.assign(deployment=dep).to_dict('records'))
        path=output/f'{dep}/noise/noise_manifest.csv'
        f=pd.read_csv(path)
        newnoise.append(f.assign(deployment=dep))
        files.append({'path':str(path),'sha256':dev.sha(path),'rows':len(f)})
        for seed in fresh.CATALOGS:
            path=output/f'{dep}/catalog_{seed}/source_systems.parquet'
            f=pd.read_parquet(path)
            newsource.append(f.assign(deployment=dep,catalog_seed=seed))
            files.append({'path':str(path),'sha256':dev.sha(path),'rows':len(f)})
    sources=pd.concat(newsource,ignore_index=True)
    noise=pd.concat(newnoise,ignore_index=True)
    oldmass=np.concatenate(oldmass)
    masses=sources[['m1_det','m2_det']].to_numpy(float)
    if not np.isfinite(masses).all() or not np.isfinite(oldmass).all():
        raise RuntimeError('Nonfinite source parameters')
    distance,_=cKDTree(oldmass).query(masses,p=np.inf)
    conflicts=[]
    for i,a in noise.iterrows():
        for j,b in noise.iloc[i+1:].iterrows():
            if a.start_gps<b.end_gps+16 and a.end_gps>b.start_gps-16:
                conflicts.append({'type':'fresh_vs_fresh_16s_guard','left':int(i),'right':int(j)})
        for b in oldnoise:
            if a.start_gps<b['end']+16 and a.end_gps>b['start']-16:
                conflicts.append({'type':'fresh_vs_development_16s_guard','left':int(i),'right':b})
    uid_repeats=int(sources.source_uid.duplicated().sum())
    mass_repeats=int(sources.duplicated(['m1_det','m2_det']).sum())
    old_matches=int((distance<1e-10).sum())
    record={'fresh_sources':len(sources),'fresh_noise_blocks':len(noise),
        'historical_metadata_rows':len(oldmass),'historical_noise_intervals':len(oldnoise),
        'source_uid_duplicates':uid_repeats,'fresh_mass_pair_duplicates':mass_repeats,
        'old_mass_pair_matches_at_1e_minus10_Msun':old_matches,
        'minimum_old_mass_pair_Linf_distance_Msun':float(distance.min()),
        'global_GPS_conflicts':conflicts,
        'source_noise_independence_pass':not(conflicts or uid_repeats or mass_repeats or old_matches),
        'lens_environment_independence_claimed':False,
        'interpretation':'Different component masses rule out identical full source parents. This does not establish independent lens environments or astrophysical population rates.',
        'sky_limitation':'conditional BAYESTAR measurement simulator, not localization of the exact injected non-Gaussian strain'}
    dev.json_write(output/'GLOBAL_SOURCE_NOISE_INDEPENDENCE.json',record)
    dev.csv_write(output/'INDEPENDENCE_INPUT_MANIFEST.csv',pd.DataFrame(files))
    dev.csv_write(output/'GLOBAL_NOISE_BLOCKS.csv',noise)
    if not record['source_noise_independence_pass']:
        raise RuntimeError('Confirmation source/noise independence failed')
    print(record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root)
