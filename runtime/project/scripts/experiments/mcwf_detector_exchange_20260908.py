#!/usr/bin/env python3
"""Detector-order invariant mass inference from unchanged waveform responses."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_temporal_response_20260908 as t
import mcwf_temporal_response_evaluate_20260908 as ev

dev, old, cf = t.dev, t.old, ev.cf
KIND = 'EXCHANGE'
PERM = list(range(9, 18)) + list(range(9)) + list(range(18, 27))
torch.set_num_threads(2)


class Predictor(old.Predictor):
    def __init__(self, kind=KIND):
        super().__init__()
        self.register_buffer('input_mu', torch.zeros(1, 27, 1))
        self.register_buffer('input_sd', torch.ones(1, 27, 1))

    def exchange(self, x):
        physical = x * self.input_sd + self.input_mu
        return (physical[:, PERM] - self.input_mu) / self.input_sd

    def forward(self, x):
        logits = super().forward(torch.cat([x, self.exchange(x)], 0)).float()
        a, b = F.log_softmax(logits, -1).chunk(2)
        return torch.logaddexp(a, b) - np.log(2.)


def warm_model(kind, checkpoint):
    model = Predictor().cuda()
    missing, extra = model.load_state_dict(checkpoint['model'], strict=False)
    if set(missing) != {'input_mu', 'input_sd'} or extra:
        raise RuntimeError(f'Warm-start state mismatch:{missing}:{extra}')
    model.input_mu.copy_(torch.as_tensor(checkpoint['mu'], device='cuda'))
    model.input_sd.copy_(torch.as_tensor(checkpoint['sd'], device='cuda'))
    return model


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent directory required')
    for name in ('contracts', 'scripts', 'logs', 'models', 'predictions', 'calibration', 'evaluation', 'tables', 'reports', 'manifest', 'figures'):
        (root / name).mkdir(parents=True)
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-DETECTOR-EXCHANGE-04', 'utc': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'same_both_runs': True,
        'baseline': str(t.PAIRS),
        'mechanism': 'Use shared mass-response CNN on both detector feature orders and average predictive probabilities;mass should not depend on the presentation order of detector-conditioned waveform evidence',
        'inputs': 'Exact existing27x253 phase-response features;swapH9/L9,retain9networkpowers;no new signal-scale descriptor',
        'formula': 'p_sym(Mc|x)=0.5*p_theta(Mc|x)+0.5*p_theta(Mc|exchange(x));normalize raw features with frozen training means/SD separately for both presentations',
        'physical_scope': 'Permutation of evidence records,not a claim H1/L1 have identical PSD,response or data quality;these remain encoded in each detector matched-response shape',
        'network_limit': 'Frozen network-max feature is retained;the symmetry test does not rederive arrival-time boundary searches',
        'model': 'same original48channel OMC CNN shared parameters for both orders;warm original;15epochs same12288train/512development,160/32noiseblocks',
        'training_seeds': t.TRAIN_SEEDS,
        'optimizer': 'same continued-training control:AdamW1e-4,WD1e-4,128sourcesx2views,clip5,cosine_min1e-5',
        'control': str(P / 'results/mcwf_temporal_response_exploratory_20260908T142000Z/models/CONTINUE'),
        'epoch0': 'symmetrizing predictions changes epoch0 output;not claimed to equal old OMC. Eligible under minimum simulated-development CE.',
        'temperature': [.5, .75, 1., 1.25, 1.5, 2.],
        'integration': 'same finite-source mass conflict and bounded simulated isotonically calibrated prior-overlap increment;gamma,beta grid0,.125,.25,.5,1,2',
        'selection': 'simulated validation only,CANDIDATE and RETRIEVAL frozen priorities,original per-seed guardrails;real joined after selection',
        'frozen': ['historical models', 'waveform input', 'time', 'sky_raw_log_bf', 'outer C-fixed weights', 'scope', 'paper'],
        'not_PE': True, 'not_lensing_Bayes_factor': True,
        'disclosure': 'Adaptive real-catalog development history,not blind confirmation;official overlaps are not lens truths',
        'reference': 'https://arxiv.org/abs/1703.06114',
        'reference_scope': 'Permutation-invariance motivation only;this is two-order averaging,not a reproduction of the Deep Sets architecture or results'})
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(t.protected()))
    shutil.copy2(__file__, root / 'scripts/detector_exchange.py')
    dev.json_write(root / 'contracts/FREEZE.json', {
        'contract_sha256': dev.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'code_sha256': dev.sha(root / 'scripts/detector_exchange.py')})
    model = Predictor().double().eval()
    model.input_mu.copy_(torch.arange(27).reshape(1, 27, 1) / 27)
    model.input_sd.copy_(1 + torch.arange(27).reshape(1, 27, 1) / 54)
    g = torch.Generator().manual_seed(202609864)
    x = torch.randn(4, 27, 253, generator=g, dtype=torch.float64)
    with torch.no_grad():
        y, other = model(x), model(model.exchange(x))
    # Internal logits cast tofloat32 intentionally matches the training logsumexp.
    error = float((y-other).abs().max())
    if error > 1e-6:
        raise RuntimeError(f'Detector exchange invariance failed:{error}')
    dev.json_write(root / 'contracts/UNIT_TEST.json', {'pass': True, 'max_log_probability_difference': error,
        'probability_sum_error': float((y.exp().sum(1)-1).abs().max()),
        'real_or_test_inputs': False})


def select(root):
    if (root / 'contracts/INTEGRATION_FROZEN.json').exists():
        raise RuntimeError('Already frozen')
    choices, grid, audits = [], [], []
    for dep in t.DEPS:
        for slot, es in zip(t.MODEL_SLOTS, t.SEEDS):
            spec, audit = ev.simulated_calibration(root, dep, KIND, slot)
            audits += audit
            f, penalty, increment, _ = ev.pair_values(root, dep, KIND, slot, es, 'validation', spec)
            w = cf.frozen_weights(dep, es)
            base = f.waveform_score.to_numpy(float)
            bm, bwm = cf.fast_metrics(f, cf.channels(f, base) @ w), cf.fast_metrics(f, base)
            rows = []
            for gamma in ev.COEFFICIENTS:
                for beta in ev.COEFFICIENTS:
                    z = base + gamma*penalty + beta*increment
                    m, wm = cf.fast_metrics(f, cf.channels(f, z) @ w), cf.fast_metrics(f, z)
                    good = cf.guard(m, bm) and cf.guard(wm, bwm)
                    row = {'gamma': gamma, 'beta': beta, 'pass': good, **m}
                    rows.append(row)
                    grid.append({'deployment': dep, 'seed': es, **row, **{'waveform_'+k: v for k, v in wm.items()}})
            for policy in ('CANDIDATE', 'RETRIEVAL'):
                def key(r):
                    if policy == 'CANDIDATE':
                        primary = (r['false_at_recall_0p5'], r['false_at_recall_0p9'], -r['average_precision'], -r['macro_r_at_10'], -r['macro_r_at_1'])
                    else:
                        primary = (-r['macro_r_at_10'], -r['macro_r_at_1'], -r['average_precision'], r['false_at_recall_0p5'], r['false_at_recall_0p9'])
                    return (*primary, r['gamma']**2+r['beta']**2, r['gamma'], r['beta'])
                win = min((r for r in rows if r['pass']), key=key)
                choices.append({'deployment': dep, 'kind': KIND, 'slot': slot, 'seed': es, 'method': KIND+'-'+policy,
                    'gamma': win['gamma'], 'beta': win['beta'], 'weights': w.tolist(), 'calibration': spec})
    dev.csv_write(root / 'tables/VALIDATION_GRID.csv', pd.DataFrame(grid))
    dev.csv_write(root / 'tables/CALIBRATION_AUDIT.csv', pd.DataFrame(audits))
    dev.csv_write(root / 'tables/SELECTED_COEFFICIENTS.csv', pd.DataFrame([{k: v for k, v in c.items() if k not in ('calibration', 'weights')} for c in choices]))
    dev.json_write(root / 'calibration/SELECTED.json', choices)
    dev.json_write(root / 'contracts/INTEGRATION_FROZEN.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'file': 'calibration/SELECTED.json',
        'sha256': dev.sha(root / 'calibration/SELECTED.json'), 'real_test_used_to_select': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['initialize', 'train', 'select', 'evaluate', 'real', 'assess'], required=True)
    a = p.parse_args()
    # Process-local model factory replacement; frozen modules/files are not edited.
    t.Predictor = Predictor
    t.warm_model = warm_model
    if a.stage == 'initialize':
        initialize(a.root)
    elif a.stage == 'train':
        for dep in t.DEPS:
            for slot, seed in zip(t.MODEL_SLOTS, t.TRAIN_SEEDS):
                t.train(a.root, dep, KIND, slot, seed)
    elif a.stage == 'select':
        select(a.root)
    else:
        getattr(ev, a.stage)(a.root)
