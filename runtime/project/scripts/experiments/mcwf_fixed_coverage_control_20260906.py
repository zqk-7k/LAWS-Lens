#!/usr/bin/env python3
"""Freeze the companion-tail tolerance; no real-PE-derived thresholds."""
import argparse
import json
from pathlib import Path
import shutil
import mcwf_development_20260905 as dev
import mcwf_unified_tail_20260906 as tail
import mcwf_unified_waveform_20260906 as u
import mcwf_evaluate_distilled_20260906 as distilled_eval


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--trained-root',type=Path)
    a=p.parse_args()
    tail.ALPHA=(.05,)
    if a.trained_root:
        if not (a.trained_root/'contracts/ALL_SIX_TRAINED.json').exists():
            raise RuntimeError('All six common-model training tasks required')
        distilled_eval.TRAINED=a.trained_root
        distilled_eval.features('validation')
        u.PREV=a.trained_root
    tail.fit(a.root)
    dev.json_write(a.root/'contracts/FIXED_COVERAGE_CONTROL.json',{
        'name':'fixed-five-percent-companion-tail',
        'alpha':.05,
        'why':'Previous alpha selection on a small validation set could choose excessive false-dismissal tolerance. Hold tolerance fixed while preserving the same score, gamma grid and selection in both runs.',
        'interpretation':'Empirical validation quantile, not an exact finite-sample conformal guarantee or physical 5% false-lensing probability.',
        'prior_trials_remain_failed':True,
        'locked_data_reused_for_development':True,
        'trained_root':str(u.PREV),
        'no_time_sky_or_outer_weight_changes':True,
    })
    shutil.copy2(__file__,a.root/'scripts'/Path(__file__).name)
    if a.trained_root:
        dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{
            'training_root':str(u.PREV),
            'training_contract_sha256':dev.sha(u.PREV/'contracts/DISTILLATION_TRAINING.json'),
            'six_models_newly_trained':True,'new_epochs_each':50,
            'old_run_specific_fallback':False})
    if not json.loads((a.root/'contracts/VALIDATION_GATE.json').read_text())['pass']:
        return
    if a.trained_root:
        for split in ('test','real'):
            distilled_eval.features(split)
    u.evaluate(a.root,score_function=tail.score)


if __name__=='__main__':
    main()
