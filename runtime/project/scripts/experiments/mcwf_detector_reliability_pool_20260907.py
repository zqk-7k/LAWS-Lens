#!/usr/bin/env python3
"""Validation-calibrated detector-information pooling, not network PE."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_detector_mass_density_20260907 as detector

dev, body = e.dev, e.body
MODE = 'loglinear'
POWERS = (0., .5, 1., 2., 4.)
SCALES = (.5, 1., 1.5, 2.)


def root_name():
    return 'detector_reliability_pool_' + MODE


def pool(a, prior, power, scale):
    p = np.stack([a['pH'], a['pL']]).astype(float)
    logp = np.log(p.clip(1e-300))
    information = np.sum(p * (logp - np.log(prior)[None, None, :]), axis=2).clip(0.)
    weights = softmax(power * np.log(information + .01), axis=0).clip(.05, .95)
    weights /= weights.sum(0, keepdims=True)
    if MODE == 'loglinear':
        logits = np.log(prior)[None, :] + scale * np.sum(weights[:, :, None] * (logp - np.log(prior)[None, None, :]), axis=0)
    else:
        logits = scale * np.log(np.sum(weights[:, :, None] * p, axis=0).clip(1e-300))
    probability = softmax(logits, axis=1)
    outside = weights[0] * a['outside_H'] + weights[1] * a['outside_L']
    return probability, outside, weights, information


def fit(root, dep, ms):
    folder = root / f'{root_name()}/models/{dep}/seed_{ms}'
    if (folder / 'COMPLETE.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    contract = root / root_name() / 'contracts/RECIPE.json'
    if not contract.exists():
        dev.json_write(contract, {'created_utc': datetime.now(timezone.utc).isoformat(),
            'same_O3_O4a': True, 'mode': MODE,
            'parent': 'frozen shared H1/L1 single-detector 4Gaussian(logMc) predictor',
            'input': 'unchanged peak2s4096points40-580Hz,3584 physical templates,per-event PSD',
            'information': 'KL(p_detector || training_prior),learned-predictive information only,NOT publicPE',
            'weights': 'softmax(power*log(KL+0.01)),clipped[0.05,0.95] andrenormalized;no detectorID-dependent coefficient',
            'loglinear': 'prior*(p_H/prior)^(scale*w_H)*(p_L/prior)^(scale*w_L),renormalized',
            'mixture': '(w_H*p_H+w_L*p_L)^scale,renormalized',
            'power_grid': POWERS, 'scale_grid': SCALES,
            'selection': '512development-source predictive logMc CE only;same complete grid foreachrun/seed',
            'calibration_limit': 'reuses the parent-model validation sample;source-unseen training but not a fresh blind validation',
            'physical_limit': 'predictive pooling is not full coherent PE;shared nuisance correlations and detector-specific glitches are not rigorously marginalized',
            'frozen': ['time', 'sky', 'outerweights', 'scope', 'oldmodels', 'oldresults'],
            'PE_official_ID_inputs': False, 'fresh_confirmation_required': True})
        dest = root / root_name() / 'scripts'
        dest.mkdir(exist_ok=True)
        shutil.copy2(__file__, dest / Path(__file__).name)
    cp = root / f'detector_mass_density/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    a = np.load(cp.parent / 'validation_predictions.npz')
    truth = a['truth']
    bins = np.searchsorted(detector.EDGES, truth, side='right') - 1
    if ((bins < 0) | (bins >= len(detector.CENTERS))).any():
        raise RuntimeError('Truth outside the fixed mass grid')
    rows = []
    for power in POWERS:
        for scale in SCALES:
            p, _, _, _ = pool(a, ck['prior'], power, scale)
            rows.append({'power': power, 'scale': scale,
                'CE': float(-np.log(p[np.arange(len(p)), bins].clip(1e-300)).mean()),
                'logMc_MAE': float(abs(p @ detector.CENTERS - truth).mean())})
    best = min(rows, key=lambda x: (x['CE'], x['power'], abs(x['scale'] - 1.5)))
    p, outside, weights, information = pool(a, ck['prior'], best['power'], best['scale'])
    if MODE == 'loglinear':
        equal, *_ = pool(a, ck['prior'], 0., 2 * ck['eta'])
        error = float(np.max(abs(equal - a['p'])))
        if error > 1e-10:
            raise RuntimeError('Uniform-weight pooling does not reproduce the existing detector model')
    else:
        error = None
    swap = {'pH': a['pL'], 'pL': a['pH'], 'outside_H': a['outside_L'], 'outside_L': a['outside_H']}
    swapped, *_ = pool(swap, ck['prior'], best['power'], best['scale'])
    if not np.allclose(p, swapped, rtol=0., atol=1e-12):
        raise RuntimeError('Detector exchange invariance failed')
    payload = {**best, 'prior': ck['prior'], 'parent_sha256': dev.sha(cp), 'mode': MODE,
               'uniform_parent_max_abs_error': error, 'detector_swap_max_abs_error': float(abs(p - swapped).max())}
    torch.save(payload, folder / 'validation_selected_model.pt')
    np.savez_compressed(folder / 'validation_predictions.npz', p=p, outside=outside,
        group=a['group'], truth=truth, detector_weights=weights.T, detector_information=information.T)
    dev.csv_write(folder / 'POOL_GRID.csv', pd.DataFrame(rows))
    dev.json_write(folder / 'COMPLETE.json', {'deployment': dep, 'seed': ms, **best,
        'uniform_parent_error': error, 'checkpoint_sha256': dev.sha(folder / 'validation_selected_model.pt')})
    print(json.dumps({'detector_reliability_pool': MODE, 'deployment': dep, 'seed': ms, **best}), flush=True)


def prediction(root, dep, ms, es, split):
    cp = root / f'{root_name()}/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Pooling fit incomplete')
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    path = root / f'{root_name()}/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a = np.load(path)
        if str(a['checkpoint_sha256']) != dev.sha(cp):
            raise RuntimeError('Changed pooling configuration')
        return a
    a = detector.prediction(root, dep, ms, es, split)
    p, outside, weights, information = pool(a, ck['prior'], ck['power'], ck['scale'])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, outside=outside, detector_weights=weights.T,
        detector_information=information.T, checkpoint_sha256=dev.sha(cp))
    return np.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for MODE in ('loglinear', 'mixture'):
        for dep in e.DEPS:
            for ms in body.MODEL_SEEDS:
                fit(args.root, dep, ms)
