#!/usr/bin/env python3
"""Fine-lag mass-compatibility calibration, with fixed five-percent tolerance."""
import argparse
import json
from pathlib import Path
import shutil
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_unified_waveform_20260906 as u
import mcwf_unified_tail_20260906 as tail
import mcwf_finelag_encoder_20260907 as fine
import mcwf_evaluate_finelag_20260907 as evaluate

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--trained-root',type=Path,required=True)
p.add_argument('--event-psd',action='store_true')
a=p.parse_args()
if a.root.exists() or not (a.trained_root/'contracts/ALL_SIX_TRAINED.json').exists():
    raise RuntimeError('Require independent output and completed six-model training')
body.encode=fine.encode
implementation='sample_resolved_zero_padded_fft_v1'
if a.event_psd:
    import mcwf_finelag_eventpsd_20260907 as psd
    evaluate.ev.input_prediction=psd.input_prediction
    implementation=psd.IMPLEMENTATION
    evaluate.IMPLEMENTATION=implementation
evaluate.features(a.trained_root,'validation')
u.PREV=a.trained_root
tail.ALPHA=(.05,)
tail.GAMMAS=tuple(2.**k for k in range(-6,2))
tail.fit(a.root)
cfg=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
cfg.update(code='MCWF-FINELAG-MASS-COMPATIBILITY',new_training_this_round=True,
    trained_models='six50epoch fine-lag shared-architecture models',
    feature_implementation=implementation,
    calibration='same fixed5percent predictive-mass-disagreement tail used in archived control, no real PE tuning',
    rationale_addendum='The direct phase-bank calibration failed one validation seed. Use the already-defined uncertainty-aware mass-compatibility channel on the numerically repaired feature extractor, preserving old morphology evidence and all guardrails.')
dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',cfg)
dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{
    'training_root':str(a.trained_root),'training_contract_sha256':dev.sha(a.trained_root/'contracts/FINE_LAG_TRAINING.json'),
    'six_models_newly_trained':True,'new_epochs_each':50,'feature_implementation':implementation,
    'not_distillation':'legacy metadata filename only; actual objective CE+SupCon'})
training_contract=json.loads((a.trained_root/'contracts/FINE_LAG_TRAINING.json').read_text())
cfg.update(training_contract=str(a.trained_root/'contracts/FINE_LAG_TRAINING.json'),
    training_code=training_contract['code'],
    training_objective=training_contract['objective'],
    rationale_addendum='Apply the already specified fixed-five-percent predictive-mass compatibility rule to this frozen waveform model; report all simulation and real-development failures.')
dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',cfg)
provenance_path=a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json'
provenance=json.loads(provenance_path.read_text())
provenance.update(training_objective=training_contract['objective'],
    rankncontrast=bool(training_contract.get('rankncontrast',False)),
    not_distillation='legacy provenance filename only; actual objective is recorded in training_objective')
dev.json_write(provenance_path,provenance)
if (a.trained_root/'contracts/INDEPENDENT_MASS_COMPONENT.json').exists():
    cfg.update(code='MCWF-FINELAG-INDEPENDENT-MASS-COMPATIBILITY',
        trained_models='six trained fine-lag event-PSD source encoders, each with an independently trained50epoch phase-only mass network',
        changed_mechanism='mass prediction no longer shares the raw/source representation; source embedding retained from the preceding shared-method encoder')
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',cfg)
    dev.json_write(a.root/'contracts/MASS_COMPONENT_PROVENANCE.json',{
        'contract_sha256':dev.sha(a.trained_root/'contracts/INDEPENDENT_MASS_COMPONENT.json'),
        'mass_objective':'simulation soft-logMc CE only','source_objective':'earlier CE+SupCon',
        'mass_epochs':50,'raw_source_body_retrained_in_mass_substep':False,
        'source_encoder_is_new_not_original_C_fixed':True})
if (a.trained_root/'contracts/PHYSICAL_MASS_ANCHOR.json').exists():
    cfg.update(code='MCWF-FINELAG-PHYSICAL-MASS-ANCHOR-COMPATIBILITY',
        new_training_this_round=False,
        trained_models='six fine-lag event-PSD source encoders plus six independent trained mass heads; simulation-calibrated equal physical/neural predictive mixture',
        waveform_mass_distribution='0.5*p_new_neural+0.5*p_tempered_physical_profile')
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',cfg)
    dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{
        'training_root':str(a.trained_root),'training_contract_sha256':dev.sha(a.trained_root/'contracts/FINE_LAG_TRAINING.json'),
        'six_models_newly_trained':False,'six_previously_trained_new_encoders_reused':True,
        'feature_implementation':implementation,'not_distillation':'calibrated physical/neural mass-distribution ensemble',
        'physical_mass_anchor_sha256':dev.sha(a.trained_root/'contracts/PHYSICAL_MASS_ANCHOR.json')})
for path in (Path(__file__),Path(fine.__file__),Path(fine.fine.__file__)):
    shutil.copy2(path,a.root/'scripts'/path.name)
if json.loads((a.root/'contracts/VALIDATION_GATE.json').read_text())['pass']:
    for split in ('test','real'):
        evaluate.features(a.trained_root,split)
    u.evaluate(a.root,score_function=tail.score)
