#!/usr/bin/env python3
"""Time-only correction, explicit channel attribution and historical replay."""
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

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_source_time_audit_20260908 as science
ev, dev, t, cf = science.ev, science.dev, science.t, science.cf
AUDIT = []


def pair_delays(frame, dep, seed, split):
    if split != 'real':
        plan = dev.BASE.retained_event_plan(dep, seed, split).set_index('idx')
        i = plan.loc[frame.idx_i, 'gps_obs'].to_numpy(float)
        j = plan.loc[frame.idx_j, 'gps_obs'].to_numpy(float)
        return abs(i - j) / 86400.
    # Real pairs carry delay_days in the immutable C-fixed table.
    if 'delta_t_days' in frame:
        return frame.delta_t_days.to_numpy(float)
    if 'time_delay_days' in frame:
        return frame.time_delay_days.to_numpy(float)
    if 'dt_days' in frame:
        return frame.dt_days.to_numpy(float)
    _, events = dev.real_inputs(dep)
    lookup = events.set_index('event_name').gps_time
    if lookup.index.duplicated().any():
        raise RuntimeError('Ambiguous real event GPS mapping')
    first, second = frame.event_i.map(lookup), frame.event_j.map(lookup)
    if first.isna().any() or second.isna().any():
        raise RuntimeError('Missing exact event GPS;never invert a time score')
    return abs(first.to_numpy(float)-second.to_numpy(float))/86400.


def scored(root, config, split):
    dep, seed = config['deployment'], config['seed']
    original = cf.read(dep, seed, split)
    waveform = original.waveform_score.to_numpy(float)
    if config['method'] == 'OMC':
        return original, waveform, {}
    frozen = json.loads((root / 'contracts/INTEGRATION_FROZEN.json').read_text())
    path = root / config['calibration_path']
    if dev.sha(path) != frozen['calibration_hashes'][dep]:
        raise RuntimeError('Time calibration changed after freeze')
    cal = json.loads(path.read_text())
    delays = pair_delays(original, dep, seed, split)
    values = np.log10(np.maximum(delays, 1e-12))
    historical = json.loads((science.V7 / dep / 'shared/time_delay_likelihood_ratio.json').read_text())
    replay = np.interp(values, historical['log10_delay_grid'], historical['log_likelihood_ratio']).astype(np.float32)
    error = float(abs(replay - original.time_score.to_numpy(float)).max())
    if error > 1e-5:
        raise RuntimeError(f'Historical time lookup replay mismatch: {dep}/{seed}/{split}:{error}')
    column = 'conservative_logLR' if config['method'].endswith('CONSERVATIVE') else 'logLR'
    new = np.interp(values, cal['grid'], cal[column])
    outside = (values < min(cal['grid'])) | (values > max(cal['grid']))
    new = np.where(outside, 0., new)
    if not np.isfinite(new).all():
        raise RuntimeError('Nonfinite new time score')
    frame = original.copy()
    frame['historical_time_score'] = original.time_score
    frame['time_score'] = new
    extra = {'actual_delay_days': delays, 'time_score_change': new - original.time_score.to_numpy(float),
             'source_time_bootstrap95_low': np.interp(values, cal['grid'], cal['bootstrap95_low']),
             'source_time_bootstrap95_high': np.interp(values, cal['grid'], cal['bootstrap95_high']), 'time_grid_ood': outside}
    AUDIT.append({'method': config['method'], 'deployment': dep, 'seed': seed, 'split': split,
                  'pairs': len(frame), 'historical_lookup_max_difference': error,
                  'waveform_max_difference': float(abs(frame.waveform_score-original.waveform_score).max()),
                  'sky_max_difference': float(abs(frame.sky_raw_log_bf-original.sky_raw_log_bf).max()),
                  'time_changed_max_abs': float(abs(new-original.time_score.to_numpy(float)).max()),
                  'time_changed_median_abs': float(np.median(abs(new-original.time_score.to_numpy(float)))),
                  'time_only_change': True, 'grid_ood_fraction': float(outside.mean())})
    return frame, waveform, extra


def evaluate(root):
    choices = ev.load_selections(root)
    rows = []
    for c in choices:
        for split in ('validation', 'test'):
            frame, waveform, extra = scored(root, c, split)
            baseline = cf.read(c['deployment'], c['seed'], split)
            w = np.asarray(c['weights'])
            out = frame.copy()
            out['retained_OMC_waveform'] = baseline.waveform_score
            out['final_score'] = cf.channels(frame, waveform) @ w
            for k, v in extra.items():
                out[k] = v
            for mode, score, base in (
                ('fusion', out.final_score.to_numpy(float), cf.channels(baseline, baseline.waveform_score.to_numpy(float)) @ w),
                ('waveform', waveform, baseline.waveform_score.to_numpy(float))):
                metric, reference = cf.fast_metrics(frame, score), dev.BASE.full_metrics(frame, score)
                for key in ('macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'):
                    if abs(metric[key] - reference[key]) > 1e-10:
                        raise RuntimeError('Independent metric replay failed')
                rows.append({'method': c['method'], 'deployment': c['deployment'], 'seed': c['seed'], 'split': split,
                             'mode': mode, 'guard_pass': cf.guard(metric, cf.fast_metrics(baseline, base)), **metric})
            dest = root / f'evaluation/{c["method"]}/{c["deployment"]}/seed_{c["seed"]}'
            dest.mkdir(parents=True, exist_ok=True)
            out.to_parquet(dest / f'{split}_pairs.parquet', index=False)
    frame = pd.DataFrame(rows)
    dev.csv_write(root / 'tables/RETRIEVAL_PER_SEED.csv', frame)
    columns = ['macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9']
    summary = frame.groupby(['method', 'deployment', 'split', 'mode'])[columns].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns]
    dev.csv_write(root / 'tables/RETRIEVAL_SUMMARY.csv', summary)
    dev.json_write(root / 'contracts/INJECTION_COMPARISON_COMPLETE.json', {'UTC': datetime.now(timezone.utc).isoformat(),
                   'rows': len(frame), 'time_changed': True, 'waveform_sky_changed': False, 'independent_confirmation': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['evaluate', 'real', 'assess'], required=True)
    a = p.parse_args()
    ev.scored = scored
    snapshot = a.root / 'scripts/source_time_evaluate.py'
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    if a.stage == 'evaluate':
        evaluate(a.root)
    else:
        getattr(ev, a.stage)(a.root)
    if AUDIT:
        dev.csv_write(a.root / f'tables/EXPLICIT_CHANNEL_CHANGE_{a.stage}.csv', pd.DataFrame(AUDIT))
