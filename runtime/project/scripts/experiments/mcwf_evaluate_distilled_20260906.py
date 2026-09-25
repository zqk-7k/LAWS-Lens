#!/usr/bin/env python3
"""Predict, calibrate on validation, then audit the shared distilled model."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u

TRAINED=dev.PROJECT/'results/mcwf_unified_v4_distilled_20260906'


def features(split):
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            out=TRAINED/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}'
            out.mkdir(parents=True,exist_ok=True)
            path=out/f'{split}_NEW-PHYSICAL_pairs.parquet'
            if path.exists():
                continue
            f=dev.real_frame(dep,es) if split=='real' else pd.read_parquet(dev.BAY/f'results/{dep}/seed_{es}/{split}_pair_scores_bayestar_sky.parquet')
            p,z=ev.input_prediction(TRAINED,dep,ms,es,'RAW-PHASE-SOURCE',split)
            x=ev.features(f,p,z)
            f=f.copy()
            f['previous_waveform_score']=f.waveform_score
            for name,key in (('new_encoder_cosine','cosine'),('new_mass_similarity','mass'),('new_mass_predictive_BC','predictive_BC'),('new_mass_pred_logmc_i','mean_i'),('new_mass_pred_logmc_j','mean_j')):
                f[name]=x[key]
            f.to_parquet(path,index=False)
            dev.json_write(out/f'{split}_FEATURE_PROVENANCE.json',{'new_waveform_score_not_fitted_here':True,'checkpoint_sha256':dev.sha(TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'),
                'frame_sha256':dev.sha(path),'waveform_column_still_baseline':True})
            print(json.dumps({'features':split,'deployment':dep,'seed':es}),flush=True)


def main():
    global TRAINED
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recipe',choices=('blend','negative','conservative'),required=True)
    p.add_argument('--trained-root',type=Path)
    a=p.parse_args()
    if a.trained_root:
        TRAINED=a.trained_root
    if not (TRAINED/'contracts/ALL_SIX_TRAINED.json').exists():
        raise RuntimeError('All six shared-model training runs must finish first')
    features('validation')
    u.PREV=TRAINED
    u.fit(a.root,negative_only=a.recipe!='blend',conservative_strength=a.recipe=='conservative')
    dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{
        'training_root':str(TRAINED),'training_contract_sha256':dev.sha(TRAINED/'contracts/DISTILLATION_TRAINING.json'),
        'six_models_newly_trained':True,'new_epochs_each':50,'old_run_specific_fallback':False,
        'feature_cache_NEW_PHYSICAL_filename':'compatibility name; cache waveform_score is baseline until this shared calibration is applied'})
    for p in (Path(__file__),Path(u.__file__),TRAINED/'scripts/mcwf_distilled_encoder_20260906.py'):
        shutil.copy2(p,a.root/'scripts'/p.name)
    if not json.loads((a.root/'contracts/VALIDATION_GATE.json').read_text())['pass']:
        return
    for split in ('test','real'):
        features(split)
    u.evaluate(a.root)


if __name__=='__main__':
    main()
