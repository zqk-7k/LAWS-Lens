#!/usr/bin/env python3
"""Waveform-only heterogeneous predictive mass pool, not a PE posterior."""
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
from scipy.special import ndtr, softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_fine_mass_only_20260907 as fine
import mcwf_highermode_density_20260907 as higher
import mcwf_mass_tf_20260905 as tf

dev, body = e.dev, e.body
MODE = 'mixture'


def root_name():
    return 'predictive_mass_pool_' + MODE


def convert_rnc(p):
    centers = np.asarray(tf.LOG_CENTERS, float)
    edges = np.r_[centers[0] - .5 * (centers[1] - centers[0]),
                  .5 * (centers[:-1] + centers[1:]),
                  centers[-1] + .5 * (centers[-1] - centers[-2])]
    cumulative = np.concatenate([np.zeros((len(p), 1)), np.cumsum(p, axis=1)], axis=1)
    values = np.stack([np.interp(fine.EDGES, edges, row, left=0., right=1.) for row in cumulative])
    mass = np.diff(values, axis=1).clip(1e-300)
    return mass / mass.sum(1, keepdims=True)


def convert_higher(a, ck):
    m = a['m'][..., 0] * ck['ys'][0] + ck['ym'][0]
    s = np.sqrt(a['cov'][..., 0, 0]) * ck['ys'][0]
    cdf = (a['w'][..., None] * ndtr((fine.EDGES[None, None, :] - m[..., None]) / s[..., None])).sum(1)
    mass = np.diff(cdf, axis=1).clip(1e-300)
    retained = mass.sum(1)
    return mass / retained[:, None], 1 - retained


def components(root, dep, ms, es, split):
    hcp = root / f'highermode_density/models/{dep}/seed_{ms}/validation_selected_model.pt'
    hck = torch.load(hcp, weights_only=False, map_location='cpu')
    if split == 'development':
        rnc = np.load(root / f'expanded_calibration/development/{dep}/seed_{es}/frozen_RNC_predictions.npz')
        a = np.load(root / f'fine_mass_only/models/{dep}/seed_{ms}/validation_predictions.npz')
        h = np.load(hcp.parent / 'validation_predictions.npz')
    else:
        rnc = e.predictions(dep, ms, es, split)
        a = fine.prediction(root, dep, ms, es, split)
        h = higher.prediction(root, dep, ms, es, split)
    hp, ho = convert_higher(h, hck)
    p = np.stack([convert_rnc(rnc['p']), a['p'], hp])
    return p, np.stack([np.zeros(len(hp)), a['outside'], ho])


def pool(p, weights, temperature):
    indices = np.flatnonzero(np.array(weights) > 0)
    weights = np.array(weights)[indices]
    if MODE == 'mixture':
        value = np.log(np.einsum('m,mnb->nb', weights, p[indices]).clip(1e-300))
    else:
        value = np.einsum('m,mnb->nb', weights, np.log(p[indices].clip(1e-300)))
    return softmax(value / temperature, axis=1)


def fit(root, dep, ms, es):
    out = root / f'{root_name()}/models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    path = root / root_name() / 'contracts/RECIPE.json'
    if not path.exists():
        dev.json_write(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
            'same_O3_O4a': True, 'pool': MODE,
            'components': ['frozen RNC mass classification', 'fine-Mc mass-only mixture', 'higher-mode 3D mixture Mc marginal'],
            'purpose': 'test complementary waveform predictors; not multiplying independent observations',
            'selection': '512-source development validation CE only; no PE or official labels',
            'pool_weight_grid': '3-simplex step0.25 including vertices',
            'temperature_grid': [.5, .75, 1., 1.25, 1.5],
            'RNC_conversion': 'piecewise-constant probability mass on midpoint logMc cells, then CDF rebin to512',
            'input': 'unchanged peak2s4096points40-580Hz',
            'limitation': 'model selection reused development validation; not fresh confirmation or physical PE',
            'reference': 'https://arxiv.org/abs/1612.01474',
            'frozen': ['time', 'sky', 'outer weights', 'scope', 'historical outputs']})
        dest = root / root_name() / 'scripts'
        dest.mkdir(exist_ok=True)
        shutil.copy2(__file__, dest / Path(__file__).name)
    p, outside = components(root, dep, ms, es, 'development')
    original = np.load(root / f'fine_mass_only/models/{dep}/seed_{ms}/validation_predictions.npz')
    truth = original['truth']
    bins = np.searchsorted(fine.EDGES, truth, side='right') - 1
    if ((bins < 0) | (bins >= 512)).any():
        raise RuntimeError('Training targets outside frozen mass support')
    rows = []
    for i in range(5):
        for j in range(5 - i):
            weights = np.array([i, j, 4 - i - j], float) / 4
            for t in (.5, .75, 1., 1.25, 1.5):
                pp = pool(p, weights, t)
                ce = float(-np.log(pp[np.arange(len(pp)), bins].clip(1e-300)).mean())
                rows.append({'w_rnc': weights[0], 'w_fine': weights[1], 'w_higher': weights[2],
                             'temperature': t, 'CE': ce})
    best = min(rows, key=lambda r: (r['CE'], abs(r['temperature'] - 1), -r['w_rnc'], r['w_fine']))
    weights = [best['w_rnc'], best['w_fine'], best['w_higher']]
    pp = pool(p, weights, best['temperature'])
    fcp = root / f'fine_mass_only/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(fcp, weights_only=False, map_location='cpu')
    checkpoint = {'weights': weights, 'temperature': best['temperature'], 'prior': ck['prior'],
                  'pool_mode': MODE, 'model_seed': ms,
                  'component_sha256': [dev.sha(e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'),
                                       dev.sha(fcp), dev.sha(root / f'highermode_density/models/{dep}/seed_{ms}/validation_selected_model.pt')]}
    torch.save(checkpoint, out / 'validation_selected_model.pt')
    np.savez_compressed(out / 'validation_predictions.npz', p=pp, truth=truth,
                        group=original['group'], outside=np.einsum('m,mn->n', weights, outside))
    dev.csv_write(out / 'POOL_GRID.csv', pd.DataFrame(rows))
    result = {'deployment': dep, 'model_seed': ms, **best,
              'Mc_MAE': float(abs(pp @ fine.CENTERS - truth).mean()),
              'checkpoint_sha256': dev.sha(out / 'validation_selected_model.pt')}
    dev.json_write(out / 'COMPLETE.json', result)
    print(json.dumps({'predictive_pool': MODE, **result}), flush=True)


def prediction(root, dep, ms, es, split):
    cp = root / f'{root_name()}/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    path = root / f'{root_name()}/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a = np.load(path)
        if str(a['checkpoint_sha256']) != dev.sha(cp):
            raise RuntimeError('Changed predictive pool')
        return a
    p, outside = components(root, dep, ms, es, split)
    pp = pool(p, ck['weights'], ck['temperature'])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=pp, outside=np.einsum('m,mn->n', ck['weights'], outside),
                        checkpoint_sha256=dev.sha(cp))
    return np.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for MODE in ('mixture', 'geometric'):
        for dep in e.DEPS:
            for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
                fit(args.root, dep, ms, es)
