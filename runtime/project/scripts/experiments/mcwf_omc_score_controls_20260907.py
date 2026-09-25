#!/usr/bin/env python3
"""Separate removal of duplicate mass veto and bounded agreement-only controls."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import mcwf_omc_ensemble_extension_20260907 as e
import mcwf_omc_single_extension_20260907 as single


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--mode', choices=('REPLACE_FRT', 'ADD_POSITIVE'), required=True)
    p.add_argument('--phase', choices=('init', 'evaluate'), required=True)
    a = p.parse_args()
    if a.phase == 'init':
        e.initialize(a.root)
        e.dev.json_write(a.root / 'contracts/SCORE_STRUCTURE_ADDENDUM.json', {
            'utc': datetime.now(timezone.utc).isoformat(), 'model_root': str(a.model_root),
            'mode': a.mode, 'single_seed_matched_models': True,
            'difference': 'REPLACE_FRT removes previous neural mass penalties and uses original Cfixed waveform LR plus the new single mass model; ADD_POSITIVE retains all current OMC score and adds only supported capped positive new mass evidence.',
            'reason': 'Test potential duplicate mass vetoes separately from corroborating evidence; neither is a product of independent Bayes factors.',
            'gamma_grid': [0.] if a.mode == 'ADD_POSITIVE' else e.GAMMAS,
            'beta_grid': [0., .0625, .125, .25, .5, 1., 2., 4.] if a.mode == 'ADD_POSITIVE' else e.BETAS,
            'same_O3_O4a': True, 'time_sky_outer_weights_frozen': True,
            'no_PE_official_or_ID_inputs': True, 'adaptive_real_selection': True,
            'acceptance_unchanged': str(a.root / 'contracts/EXPERIMENT_CONTRACT.json')})
        shutil.copy2(__file__, a.root / 'scripts' / Path(__file__).name)
    else:
        single.configure(a.root, a.model_root)
        cal = e.calibrate
        e.calibrate = lambda root, dep, es: {**cal(root, dep, es), 'score_mode': a.mode}
        if a.mode == 'ADD_POSITIVE':
            e.GAMMAS = (0.,)
            e.BETAS = (0., .0625, .125, .25, .5, 1., 2., 4.)
        e.run(a.root, 'MIXTURE-PRIOR')
