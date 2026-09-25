#!/usr/bin/env python3
"""Read-only, pair-aligned integrity audit for the final profile controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_nodup_reliability_20260909 as r
n = r.n
ARMS = {
    'R48': ('mcwf_nodup_quality_rejection_48_20260909T155053Z', 'QUALITY-PROFILE-GLOBAL-REJECT95'),
    'R49': ('mcwf_nodup_paired_quality_backoff_49_20260909T160024Z', 'PAIRED-QUALITY-GLOBAL-BACKOFF'),
}


def aligned(path):
    frame = pd.read_parquet(path).sort_values(['idx_i', 'idx_j']).reset_index(drop=True)
    if frame.duplicated(['idx_i', 'idx_j']).any():
        raise RuntimeError('Duplicate pair identity: '+str(path))
    return frame


def metrics(frame):
    return {mode: n.cf.fast_metrics(frame, frame[column].to_numpy(float))
            for mode, column in [('fusion', 'final_score'), ('waveform', 'waveform_score')]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if root.exists():
        raise RuntimeError('Independent audit output required')
    for folder in ('contracts', 'tables', 'scripts', 'manifest'):
        (root/folder).mkdir(parents=True)
    rows, gates, inputs, keyrows, configs = [], [], [], [], []
    for label, (dirname, method) in ARMS.items():
        source = P/'results'/dirname
        configuration = json.loads((source/'configs/SELECTED_CONFIGURATIONS.json').read_text())
        lookup = {(c['deployment'], int(c['seed']), c['method']): c for c in configuration}
        for (dep, seed, arm), cfg in lookup.items():
            if arm != method:
                continue
            old = lookup[dep, seed, 'NODUP-DIRECT-REPLAY']
            equal = {field: cfg[field] == old[field] for field in ('weights', 'gamma', 'beta')}
            if not all(equal.values()):
                raise RuntimeError('Frozen coefficient changed')
            configs.append({'arm': label, 'deployment': dep, 'seed': seed,
                'weights': json.dumps(cfg['weights']), 'gamma': cfg['gamma'], 'beta': cfg['beta'],
                **{key+'_unchanged': value for key, value in equal.items()}})
        paths = sorted((source/f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        paths += sorted((source/f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
        for path in paths:
            rel = path.relative_to(source/f'results/{method}')
            dep, seedstr, panel = rel.parts[:3]
            seed = int(seedstr.split('_')[1])
            basepath = source/'results/NODUP-DIRECT-REPLAY'/rel
            frame, baseline = aligned(path), aligned(basepath)
            if not frame[['idx_i','idx_j']].equals(baseline[['idx_i','idx_j']]):
                raise RuntimeError('Changed candidate scope')
            frozen = ['time_score', 'sky_raw_log_bf', 'embedding_only']
            frozen += [c for c in ('delta_t_days','sky_bc','sky_j50','sky_j90','time_contribution','sky_contribution') if c in frame]
            errors = {}
            for column in frozen:
                if not frame[column].equals(baseline[column]):
                    raise RuntimeError('Frozen score/data changed: '+column+' '+str(path))
                errors[column+'_max_delta'] = 0.
            isreal = panel == 'real'
            if isreal:
                auditcols = [c for c in baseline if c.startswith('pe_') or c.startswith('official_')]
                if not frame[auditcols].equals(baseline[auditcols]):
                    raise RuntimeError('PE or official audit fields changed')
            used = frame.profile_state.to_numpy(int) == 2
            if label == 'R49' and not np.array_equal(frame.waveform_score.to_numpy()[~used], baseline.waveform_score.to_numpy()[~used]):
                raise RuntimeError('Inactive exact NODUP backoff failed')
            reference = frame.quality_reference_waveform.to_numpy(float)
            lower = frame.profile_reference_waveform.to_numpy(float)
            flag = frame.profile_reference_true_tail.to_numpy(float) < .05
            expected = np.where(used & flag & (lower < reference), lower, reference)
            if label == 'R49':
                expected = np.where(used, expected, baseline.waveform_score.to_numpy(float))
            rule_delta = float(np.max(np.abs(expected-frame.waveform_score.to_numpy(float))))
            config = lookup[dep, seed, method]
            final = n.cf.channels(frame, frame.waveform_score.to_numpy(float)) @ np.asarray(config['weights'])
            final_delta = float(np.max(np.abs(final-frame.final_score.to_numpy(float))))
            if max(rule_delta, final_delta) > 1e-12:
                raise RuntimeError('Rule/final-score reconstruction failed')
            rows.append({'arm': label, 'deployment': dep, 'seed': seed, 'panel': panel,
                'pairs': len(frame), 'both_valid_pairs': int(used.sum()),
                'profile_lower_envelope_pairs': int((used & flag & (lower < reference)).sum()),
                'waveform_changed_pairs': int((frame.waveform_score.to_numpy()!=baseline.waveform_score.to_numpy()).sum()),
                'rule_max_delta': rule_delta, 'final_reconstruction_max_delta': final_delta,
                'PE_official_unchanged': True if isreal else None, **errors})
            for src in (path, basepath):
                inputs.append({'path': str(src), 'sha256': n.sha(src), 'bytes': src.stat().st_size})
            if isreal:
                key = frame.pair_key.eq('GW191103_012549--GW191105_143521')
                for record in frame.loc[key].to_dict('records'):
                    keyrows.append({'arm': label, 'deployment': dep, 'training_seed': seed, **record})
                continue
            candidate_metrics = metrics(frame)
            for name in ('NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED'):
                other = aligned(source/'results'/name/rel)
                references = metrics(other)
                for mode, value in candidate_metrics.items():
                    reference_metrics = references[mode]
                    keymetrics = ('macro_r_at_1', 'macro_r_at_10', 'average_precision',
                                  'false_at_recall_0p5', 'false_at_recall_0p9')
                    gates.append({'arm': label, 'deployment': dep, 'seed': seed, 'panel': panel,
                        'mode': mode, 'baseline': name, 'existing_guard': n.cf.guard(value, reference_metrics),
                        **{k+'_delta': value[k]-reference_metrics[k] for k in keymetrics}})
    n.write_csv(root/'tables/PAIR_ALIGNED_FROZEN_CHANNEL_AUDIT.csv', rows)
    n.write_csv(root/'tables/BASELINE_SPECIFIC_INJECTION_GUARDS.csv', gates)
    n.write_csv(root/'tables/FROZEN_COEFFICIENTS.csv', configs)
    n.write_csv(root/'tables/CRITICAL_PAIR_PER_SEED_COMPLETE.csv', keyrows)
    n.write_csv(root/'manifest/INPUT_SHA256.csv', inputs)
    grouped = pd.DataFrame(gates)
    grouped['split'] = np.where(grouped.panel.str.startswith('sept8_reused'), 'sept8_reused', grouped.panel)
    summary = grouped.groupby(['arm','deployment','baseline','split','mode']).agg(
        passed=('existing_guard','sum'), total=('existing_guard','size')).reset_index()
    n.write_csv(root/'tables/GUARD_COUNTS_BY_REFERENCE.csv', summary.to_dict('records'))
    n.write_json(root/'contracts/FINAL_PROFILE_INTEGRITY.json', {'UTC': n.utc(),
        'pair_aligned_panels': len(rows), 'frozen_channel_failures': 0, 'rule_failures': 0,
        'scope_PE_official_failures': 0, 'outer_weights_gamma_beta_unchanged': True,
        'score_rule_tolerance': 1e-12, 'frozen_channels_tolerance': 0,
        'stage': 'read-only, post-freeze adaptive-development audit;not parameter selection',
        'public_PE_is_not_a_prediction_input': True,
        'not_a_universal_noninferiority_declaration': True,
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})
    shutil.copy2(__file__, root/'scripts/final_profile_integrity.py')
    print('FINAL_INTEGRITY_COMPLETE', len(rows), len(inputs), flush=True)


if __name__ == '__main__':
    main()
