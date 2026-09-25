#!/usr/bin/env python3
"""Shrinkage whitening of frozen waveform embeddings, without PE features."""
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
import torch.nn.functional as F
from sklearn.covariance import OAS
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e

dev, body, ev = e.dev, e.body, e.ev
MODES = ('training_total', 'training_within_source', 'unlabeled_catalog')
POWERS = (.25, .5, 1.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)
torch.set_num_threads(2)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def effective_rank(cov):
    values = np.linalg.eigvalsh(cov).clip(1e-30)
    values /= values.sum()
    return float(np.exp(-np.sum(values*np.log(values))))


def train_metric(root, dep, ms):
    path = root / f'embedding_geometry/{dep}/seed_{ms}/metric.npz'
    if path.exists():
        return np.load(path)
    checkpoint = e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
    h = np.load(root / f'qprobe/models/{dep}/seed_{ms}/train_hidden.npy')[:, -128:]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = F.normalize(F.linear(torch.as_tensor(h, dtype=torch.float32, device='cuda'),
            ck['model']['embedding.weight'].cuda(), ck['model']['embedding.bias'].cuda()), dim=-1).float().cpu().numpy()
    z = unit(z.astype(float))
    meta = pd.read_parquet(e.TRAINED / f'cache/{dep}/train_metadata.parquet')
    group, names = pd.factorize(meta.waveform_parent_uid, sort=True)
    means = np.stack([z[group == k].mean(0) for k in range(len(names))])
    center = means.mean(0)
    total = OAS(assume_centered=True).fit(z-center)
    within = OAS(assume_centered=True).fit(z-means[group])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, center=center, training_total=total.covariance_,
                        training_within_source=within.covariance_)
    dev.json_write(path.with_suffix('.json'), {'source_count': len(names), 'event_count': len(z),
        'effective_rank_total': effective_rank(total.covariance_), 'shrinkage_total': total.shrinkage_,
        'shrinkage_within': within.shrinkage_, 'checkpoint_sha256': dev.sha(checkpoint),
        'independent_sample_caveat': '8views/source and repeatednoise blocks; covariance fit is a metric, not a significance estimate'})
    return np.load(path)


def transform(z, metric, mode, power):
    z = unit(np.asarray(z, float))
    valid = np.isfinite(z).all(1)
    output = np.full_like(z, np.nan)
    values = z[valid]
    if mode == 'unlabeled_catalog':
        center = values.mean(0)
        cov = OAS(assume_centered=True).fit(values-center).covariance_
    else:
        center, cov = metric['center'], metric[mode]
    eigenvalues, vectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, eigenvalues.max()*1e-4)
    projection = (vectors*eigenvalues[None]**(-.5*power)) @ vectors.T
    output[valid] = unit((values-center) @ projection)
    if not np.isfinite(output[valid]).all():
        raise RuntimeError('Invalid transformed embedding')
    return output


def calibration(dep, ms, metric, mode, power):
    a = np.load(e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/development_validation.npz')
    z = transform(a['embedding'], metric, mode, power)
    i, j = np.triu_indices(len(z), 1)
    raw = np.sum(z[i]*z[j], 1)
    y = a['group'][i] == a['group'][j]
    w = np.where(y, .5/y.sum(), .5/(~y).sum())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(raw, y.astype(float), sample_weight=w)
    floor = 1/(y.sum()+2)
    p = np.clip(iso.y_thresholds_, floor, 1-floor)
    return {'mode': mode, 'power': power, 'knots': iso.X_thresholds_.tolist(),
        'loglr': (np.log(p)-np.log1p(-p)).tolist(), 'fit_min': float(raw.min()), 'fit_max': float(raw.max()),
        'positive_source_count': int(y.sum())}


def pair_raw(dep, ms, es, split, metric, spec):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = e.predictions(dep, ms, es, split)
    z = transform(a['z'], metric, spec['mode'], spec['power'])
    return f, np.sum(z[f.idx_i.to_numpy(int)]*z[f.idx_j.to_numpy(int)], 1)


def score(f, raw, spec):
    inc = np.interp(raw, spec['knots'], spec['loglr'])
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    inc = np.where(ood, np.minimum(inc, 0.), inc).clip(-4, 4)
    return f.waveform_score.to_numpy(float)+spec['beta']*inc, inc, ood


def run(root):
    trial = root / 'trials/EMBEDDING-SHRINKAGE-GEOMETRY'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'mechanism': 'whiten redundant embedding covariance using shrinkage; no new encoder, PE, or official input',
        'modes': MODES, 'powers': POWERS, 'beta_grid': BETAS,
        'transductive_disclosure': 'unlabeled_catalog mode uses all available waveform embeddings in that catalog to fit covariance; it is NOT pair-local or frozen global calibration. The same operation is applied to simulation calibration/validation/test and real catalogs.',
        'frozen_training_modes': 'training_total and training_within_source use only training embeddings for center and covariance',
        'calibration': 'balanced true/null isotonic on480disjoint developmentvalidation events; bounded[-4,4]; positiveOODdisabled',
        'selection': 'BAYESTAR simulation validation only, original deterministic priority and FRT-relative guards',
        'frozen': ['waveform model parameters', 'time', 'sky', 'outerweights', 'scope', 'old outputs'],
        'interpretation': 'adaptive development; real audit cannot be labeled blind confirmation',
        'references': ['https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html'],
    })
    chosen, metrics, states = {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            metric = train_metric(root, dep, ms)
            metrics[dep, ms] = metric
            choices, rows = [], []
            for mode in MODES:
                for power in POWERS:
                    cal = calibration(dep, ms, metric, mode, power)
                    f, raw = pair_raw(dep, ms, es, 'validation', metric, cal)
                    bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
                    for beta in BETAS:
                        spec = {**cal, 'beta': beta}
                        z, _, ood = score(f, raw, spec)
                        mm = ev.metrics(f, z, dep, es)
                        ok = ev.guard(mm, bm)
                        rows.append({'mode': mode, 'power': power, 'beta': beta, 'pass': ok,
                            'ood_fraction': float(ood.mean()), **{a+'_'+k: v for a, b in mm.items() for k, v in b.items()}})
                        if ok:
                            choices.append((e.old_selection.objective(mm, beta, power)+(mode,), spec))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            spec = min(choices, key=lambda x: x[0])[1]
            chosen[dep, es] = spec
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'mode': spec['mode'], 'power': spec['power'], 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'embedding_geometry_selected': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, raw = pair_raw(dep, ms, es, split, metrics[dep, ms], chosen[dep, es])
                z, inc, ood = score(f, raw, chosen[dep, es])
                new = f.copy()
                new['FRT_baseline_waveform_score'] = f.waveform_score
                new['waveform_score'], new['geometry_raw'], new['geometry_increment'], new['geometry_ood'] = z, raw, inc, ood
                for col in ('time_score', 'sky_raw_log_bf'):
                    assert np.array_equal(new[col], f[col])
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                new.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = new
                else:
                    bm, mm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es), ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm)})
                    for method in bm:
                        for config, m in [('FRT_BASELINE', bm[method]), ('CANDIDATE', mm[method])]:
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    run(a.root)
