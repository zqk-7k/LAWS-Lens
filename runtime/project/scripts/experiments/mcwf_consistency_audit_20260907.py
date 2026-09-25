#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_unified_waveform_20260906 as u
import mcwf_unified_tail_20260906 as tail
import mcwf_conditional_waveform_calibration_20260906 as conditional


def run(root):
    model_info=root/'contracts/DISTILLED_MODEL_PROVENANCE.json'
    provenance=json.loads(model_info.read_text()) if model_info.exists() else {}
    trained=Path(provenance.get('training_root',provenance.get('trained_root',u.PREV)))
    rows=[]
    shapes=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            checkpoint=trained/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            shape=sorted((k,list(v.shape),str(v.dtype)) for k,v in ck['model'].items())
            shape += sorted(('independent_mass.'+k,list(v.shape),str(v.dtype)) for k,v in ck.get('independent_mass_component',{}).items())
            shape_hash=hashlib.sha256(json.dumps(shape).encode()).hexdigest()
            shapes.append(shape_hash)
            spec=json.loads((root/f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json').read_text())
            real=pd.read_parquet(root/f'evaluation/{dep}/seed_{es}/real_pairs.parquet')
            old=dev.real_frame(dep,es)
            for column in ('time_score','sky_raw_log_bf','idx_i','idx_j','pair_key'):
                if not np.array_equal(real[column],old[column]):
                    raise RuntimeError('Frozen real channel/pair changed: '+column)
            contribution=real.waveform_score.to_numpy()-real.previous_waveform_score.to_numpy()
            rows.append({'deployment':dep,'model_seed':ms,'eval_seed':es,'shape_hash':shape_hash,'recipe':spec['recipe'],'gamma':spec['gamma'],
                'checkpoint_sha256':dev.sha(checkpoint),'real_pairs':len(real),'changed_pair_fraction':float((contribution!=0).mean()),
                'waveform_delta_min':float(contribution.min()),'waveform_delta_max':float(contribution.max()),'time_sky_pair_order_unchanged':True,
                'selected_epoch':ck['epoch'],'network_kind':ck['config']})
            rows[-1].update(independent_mass_component='independent_mass_component' in ck,
                physical_mass_anchor='physical_mass_anchor' in ck,
                physical_mass_temperature=ck.get('physical_mass_anchor',{}).get('temperature'),
                mass_selected_epoch=ck.get('independent_mass_epoch'),feature_implementation=ck.get('feature_implementation','legacy_coarse_phasebank'))
    f=pd.DataFrame(rows)
    if len(set(shapes))!=1 or f.recipe.nunique()!=1 or not f.gamma.gt(0).all() or not f.changed_pair_fraction.gt(0).all():
        raise RuntimeError('Shared-method/new-model-participation audit failed')
    dev.csv_write(root/'tables/METHOD_CONSISTENCY_AUDIT.csv',f)
    implementation=(conditional.score if f.recipe.iloc[0]=='conditional_monotone_waveform'
                    else tail.score if f.recipe.iloc[0].startswith('companion_tail') else u.frozen_score)
    dev.json_write(root/'contracts/METHOD_CONSISTENCY_AUDIT.json',{'pass':True,'all_six_same_parameter_shapes':True,'all_six_same_score_formula':True,
        'both_runs_new_encoder_active':True,'O4_old_only_fallback':False,'time_sky_exactly_frozen':True,
        'formula_implementation_sha256':hashlib.sha256(inspect.getsource(implementation).encode()).hexdigest(),
        'parameters_may_differ':'run-matched trained weights, validation normalization/calibration and numeric hyperparameters; never formula or input channels'})
    print(f.to_json(orient='records'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    run(a.root)
