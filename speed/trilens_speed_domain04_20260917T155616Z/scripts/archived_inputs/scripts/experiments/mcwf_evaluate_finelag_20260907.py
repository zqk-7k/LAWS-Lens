#!/usr/bin/env python3
"""Frozen-model, validation-only calibration of sample-resolved waveform features."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
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
import mcwf_finelag_encoder_20260907 as fine
IMPLEMENTATION='sample_resolved_zero_padded_fft_v1'


def features(trained,split):
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            out=trained/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}'
            out.mkdir(parents=True,exist_ok=True)
            path=out/f'{split}_NEW-PHYSICAL_pairs.parquet'
            if path.exists():
                continue
            f=dev.real_frame(dep,es) if split=='real' else pd.read_parquet(dev.BAY/f'results/{dep}/seed_{es}/{split}_pair_scores_bayestar_sky.parquet')
            p,z=ev.input_prediction(trained,dep,ms,es,'RAW-PHASE-SOURCE',split)
            x=ev.features(f,p,z)
            f=f.copy()
            f['previous_waveform_score']=f.waveform_score
            for name,key in (('new_encoder_cosine','cosine'),('new_mass_similarity','mass'),('new_mass_predictive_BC','predictive_BC'),('new_mass_pred_logmc_i','mean_i'),('new_mass_pred_logmc_j','mean_j')):
                f[name]=x[key]
            f.to_parquet(path,index=False)
            dev.json_write(out/f'{split}_FEATURE_PROVENANCE.json',{'feature_implementation':IMPLEMENTATION,
                'frame_sha256':dev.sha(path),'waveform_score_is_frozen_baseline':True,
                'checkpoint_sha256':dev.sha(trained/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt')})
            print(json.dumps({'features':split,'deployment':dep,'seed':es}),flush=True)


def main():
    global IMPLEMENTATION
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--trained-root',type=Path,required=True)
    p.add_argument('--recipe',choices=('negative','blend'),default='negative')
    p.add_argument('--event-psd',action='store_true')
    p.add_argument('--cosine-only',action='store_true',help='Representation-only ablation; no separate predicted-mass gap in the score')
    a=p.parse_args()
    if not (a.trained_root/'contracts/ALL_SIX_TRAINED.json').exists():
        raise RuntimeError('Six training runs must finish before evaluation')
    if a.root.exists():
        raise RuntimeError('Independent result root required')
    if a.cosine_only:
        u.MASS_GRID=(0.,)
    u.initialize(a.root,negative_only=a.recipe=='negative')
    if a.event_psd:
        import mcwf_finelag_eventpsd_20260907 as psd
        IMPLEMENTATION=psd.IMPLEMENTATION
        ev.input_prediction=psd.input_prediction
    training_contract=json.loads((a.trained_root/'contracts/FINE_LAG_TRAINING.json').read_text())
    cfg=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    cfg.update(code='MCWF-FINELAG-'+a.recipe.upper(),new_training_this_round=True,
        trained_models='six50epochfine-lag models with shared architecture/objective',
        training_objective=training_contract['objective'],
        cosine_only=a.cosine_only,
        mass_gap_in_score=not a.cosine_only,
        waveform_feature_implementation=IMPLEMENTATION,
        additional_reference='https://pycbc.org/pycbc/latest/html/filter.html')
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',cfg)
    provenance={'training_root':str(a.trained_root),'training_contract_sha256':dev.sha(a.trained_root/'contracts/FINE_LAG_TRAINING.json'),
        'six_models_newly_trained':True,'new_epochs_each':50,'feature_implementation':IMPLEMENTATION,
        'not_distillation':'legacy provenance filename for shared evaluation interface',
        'training_objective':training_contract['objective'],
        'rankncontrast':bool(training_contract.get('rankncontrast',False))}
    dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',provenance)
    body.encode=fine.encode
    features(a.trained_root,'validation')
    u.PREV=a.trained_root
    u.fit(a.root,negative_only=a.recipe=='negative')
    for path in (Path(__file__),Path(fine.__file__),Path(fine.fine.__file__)):
        shutil.copy2(path,a.root/'scripts'/path.name)
    if a.event_psd:
        shutil.copy2(psd.__file__,a.root/'scripts'/Path(psd.__file__).name)
    if json.loads((a.root/'contracts/VALIDATION_GATE.json').read_text())['pass']:
        for split in ('test','real'):
            features(a.trained_root,split)
        u.evaluate(a.root)


if __name__=='__main__':
    main()
