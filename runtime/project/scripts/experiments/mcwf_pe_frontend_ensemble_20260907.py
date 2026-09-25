#!/usr/bin/env python3
"""Source-calibrated deep-ensemble mass evidence, with no PE features."""
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
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
torch.set_num_threads(2)
GAMMA = (.25, .5, 1., 2., 4.)
BETA = (0., .0625, .125, .25, .5, 1., 2.)


def prepare(root, split):
    for dep in e.DEPS:
        for es in dev.SEEDS:
            folder = root / f'cache/ensemble/{dep}/seed_{es}'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{split}.npz'
            if path.exists():
                continue
            original = e.predictions(dep, body.MODEL_SEEDS[0], dev.SEEDS[0], 'real') if split == 'real' else None
            if split == 'real':
                full, events = dev.real_inputs(dep)
                valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
                values = np.asarray(full[valid])
                feature = e.TRAINED / f'cache/deployment_event_psd/{dep}/real_features.npy'
            else:
                events = dev.BASE.retained_event_plan(dep, es, split)
                values = dev.ORCH.event_array_for_plan(dep, es, split, events)
                feature = e.TRAINED / f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy'
            x = np.load(feature)
            raw = dev.TRAIN.make_window_view(np.asarray(values, dtype=np.float32), 2)
            probs, embeddings, hashes = [], [], []
            for ms in body.MODEL_SEEDS:
                checkpoint = e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
                ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
                model = body.Encoder('RAW-PHASE-SOURCE').cuda().eval()
                model.load_state_dict(ck['model'])
                l, z = body.infer(model, (x - ck['mu']) / ck['sd'], raw)
                p = softmax(l.astype(float) / ck['temperature'], axis=1)
                if split == 'real':
                    pp = np.full((len(full), 64), np.nan)
                    zz = np.full((len(full), 128), np.nan)
                    pp[valid], zz[valid] = p, z
                    p, z = pp, zz
                probs.append(p)
                embeddings.append(z)
                hashes.append(dev.sha(checkpoint))
                del model
            np.savez_compressed(path, p=np.stack(probs), z=np.stack(embeddings), checkpoint_sha256=np.array(hashes), feature_sha256=dev.sha(feature))
            print(json.dumps({'ensemble_predictions': dep, 'eval_seed': es, 'split': split}), flush=True)


def pooled_prob(p, pooling):
    if pooling == 'mixture':
        return np.mean(p, axis=0)
    lp = np.log(np.maximum(p, 1e-12)).mean(0)
    return softmax(lp, axis=1)


def pair_features(root, dep, es, split, pooling):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    saved = np.load(root / f'cache/ensemble/{dep}/seed_{es}/{split}.npz')
    p = pooled_prob(saved['p'], pooling)
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    prior = e.reference_prior(dep)
    bc = np.sum(np.sqrt(p[i] * p[j]), axis=1)
    overlap = np.log(np.maximum(np.sum(p[i] * p[j] / prior, axis=1), 1e-30))
    return f, {'a': -np.log(bc.clip(1e-12)), 'overlap': overlap, 'p': p}


def score(f, x, spec):
    pt = tail.tail_probability(x['a'], spec['reference'])
    penalty = np.minimum(np.log(pt / .05), 0.)
    ref = np.clip(x['overlap'], -4., 4.)
    ood = (x['overlap'] < spec['reference_overlap_min']) | (x['overlap'] > spec['reference_overlap_max'])
    ref = np.where(ood, np.minimum(ref, 0.), ref)
    zw = f.previous_waveform_score.to_numpy(float) + spec['gamma'] * penalty + spec['beta'] * ref
    return zw, penalty, ref, ood


def run(root, pooling):
    trial = root / 'trials' / ('ENSEMBLE-' + pooling.upper())
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for d in ('contracts', 'calibration', 'evaluation', 'results', 'tables'):
        (trial / d).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'pooling': pooling,
        'same_both_runs': True, 'gamma_grid': GAMMA, 'beta_grid': BETA,
        'formula': 'Zold+gamma*finite_companion_tail(BC(pool(p_i),pool(p_j)))+beta*clip(log_predictive_reference_overlap,-4,4)',
        'difference_from_baseline': 'all three frozen new encoder predictions, rather than one, determine predictive mass agreement; old encoder and outer weights remain per seed',
        'no_new_training': True, 'no_PE_labels_or_event_id_features': True,
        'purpose': 'reduce stochastic predictive mass instability; geometric pooling is a declared log-opinion-pool sensitivity, not an exact PE posterior',
        'selection': 'same validation-only guardrails and lexicographic F50/F90/AP/R10 objective',
        'real_results': 'adaptive development, not blind confirmation',
    })
    prepare(root, 'validation')
    choices, status = {}, []
    for dep in e.DEPS:
        for es in dev.SEEDS:
            f, x = pair_features(root, dep, es, 'validation', pooling)
            y = f.is_true_pair.to_numpy(bool)
            ref = x['a'][y]
            base = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, options = [], []
            for gamma in GAMMA:
                for beta in BETA:
                    cfg = {'recipe': 'ensemble_finite_reference', 'pooling': pooling, 'gamma': gamma, 'beta': beta,
                           'reference': np.sort(ref).tolist(), 'reference_overlap_min': float(x['overlap'].min()),
                           'reference_overlap_max': float(x['overlap'].max()), 'deployment': dep, 'eval_seed': es}
                    zw, *_ = score(f, x, cfg)
                    m = ev.metrics(f, zw, dep, es)
                    passed = ev.guard(m, base)
                    rows.append({'gamma': gamma, 'beta': beta, 'pass': passed, **{method + '_' + k: v for method, mm in m.items() for k, v in mm.items()}})
                    if passed:
                        options.append((e.old_selection.objective(m, gamma, beta), cfg))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            st = {'deployment': dep, 'seed': es, 'eligible': len(options)}
            if options:
                choices[dep, es] = min(options, key=lambda p: p[0])[1]
                dev.json_write(out / 'SELECTED_CONFIG.json', choices[dep, es])
                st.update(gamma=choices[dep, es]['gamma'], beta=choices[dep, es]['beta'], sha256=dev.sha(out / 'SELECTED_CONFIG.json'))
            status.append(st)
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': status, 'pass': len(choices) == 6})
    print(json.dumps({'selected_ensemble': pooling, 'status': status}), flush=True)
    if len(choices) != 6:
        return
    prepare(root, 'test')
    prepare(root, 'real')
    metrics, guards = [], []
    for dep in e.DEPS:
        real = {}
        for es in dev.SEEDS:
            for split in ('validation', 'test', 'real'):
                f, x = pair_features(root, dep, es, split, pooling)
                zw, penalty, ref, ood = score(f, x, choices[dep, es])
                changed = f.copy()
                changed['FRT_baseline_waveform_score'] = f.waveform_score
                changed['waveform_score'] = zw
                changed['ensemble_mass_penalty'] = penalty
                changed['ensemble_reference_overlap'] = ref
                changed['ensemble_ood'] = ood
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                changed.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = changed
                else:
                    b = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
                    n = ev.metrics(f, zw, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(n, b)})
                    for method in b:
                        for c, mm in (('FRT_BASELINE', b[method]), ('CANDIDATE', n[method])):
                            metrics.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': c, **mm})
        pe = pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet')
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pe)
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(metrics))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pooling', choices=('mixture', 'geometric'), required=True)
    a = p.parse_args()
    run(a.root, a.pooling)
