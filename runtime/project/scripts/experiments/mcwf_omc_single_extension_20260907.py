#!/usr/bin/env python3
"""Seed-matched architecture controls versus uniform predictive ensembles."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import mcwf_omc_ensemble_extension_20260907 as e
import mcwf_omc_architecture_extension_20260907 as architecture

DEV = e.dev


def configure(root, model_root):
    native_calibrate = e.calibrate
    native_ensemble = e.ensemble

    def single_values(output, dep, es, split):
        if es not in DEV.SEEDS:
            raise RuntimeError('Explicit model/evaluation seed required')
        ms = dict(zip(DEV.SEEDS, e.body.MODEL_SEEDS))[es]
        return architecture.prediction(model_root, dep, ms, es, split)

    def single_calibrate(output, dep, es):
        # Reuse the exact calibrator through an isolated per-seed output directory.
        separate = output / f'seed_calibration/{dep}/{es}'
        old_prediction = e.ensemble
        e.ensemble = lambda unused_root, dep0, es0, split: single_values(output, dep0, es, split)
        try:
            return native_calibrate(separate, dep, es)
        finally:
            e.ensemble = old_prediction

    e.ensemble = single_values
    e.calibrate = single_calibrate


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--phase', choices=('init', 'evaluate'), required=True)
    p.add_argument('--arm', choices=e.ARMS, default='MIXTURE-PRIOR')
    a = p.parse_args()
    if a.phase == 'init':
        e.initialize(a.root)
        DEV.json_write(a.root / 'contracts/SINGLE_MODEL_ADDENDUM.json', {
            'utc': datetime.now(timezone.utc).isoformat(), 'model_root': str(a.model_root),
            'override': 'Replace uniform three-model mixture by one seed-matched predictive model; all mass features/calibration/grid/acceptance unchanged',
            'training_seeds': architecture.SEEDS, 'evaluation_seed_to_model': dict(zip(DEV.SEEDS, e.body.MODEL_SEEDS)),
            'no_new_training': True, 'ensemble_word_in_trial_name_is_legacy_identifier': True,
            'no_PE_official_scoring_inputs': True, 'adaptive_real_selection': True,
            'model_hashes': [{ 'path': str(p), 'sha256': DEV.sha(p)} for p in sorted(a.model_root.glob('models/*/*/validation_selected_model.pt'))]})
        shutil.copy2(__file__, a.root / 'scripts' / Path(__file__).name)
    else:
        configure(a.root, a.model_root)
        e.run(a.root, a.arm)
