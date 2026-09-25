#!/usr/bin/env python3
"""Expose actual predictive inputs without changing frozen score functions."""
import argparse
from pathlib import Path
import runpy
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('predictive', 'joint'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--predictive-root', type=Path, required=True)
    parser.add_argument('--selection-root', type=Path)
    parser.add_argument('--stage', choices=('prepare', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    import mcwf_independent_waveform_calibration_20260909 as base
    n, r, co, h = base.n, base.r, base.co, base.h
    old_install, old_run = r.install, n.run
    fields = ('embedding_only',)+h.FEATURES+('predictive_active_i', 'predictive_active_j',
        'actual_waveform_joint_BC', 'new_prediction_used_in_score', 'single_classifier_used')

    def install():
        old_install()
        old_export, old_consensus = n.public_frame, n.dev.BASE.consensus_real
        def export(frame, z, weights, method):
            out = old_export(frame, z, weights, method)
            for name in fields[:-3]:
                out[name] = frame[name].to_numpy() if name in frame else np.nan
            newbc = np.exp(frame.profile_log_mass_BC.to_numpy()+frame.profile_log_conditional_BC.to_numpy())
            replay, archived = method == 'NODUP-DIRECT-REPLAY', method == n.BASELINE
            if archived:
                out['actual_waveform_joint_BC'] = np.nan
                out['new_prediction_used_in_score'] = 0.
            elif replay:
                out['actual_waveform_joint_BC'] = frame.parent_joint_BC.to_numpy()
                out['new_prediction_used_in_score'] = 0.
            elif method == 'NEWPROFILE-PAIRED-MIN':
                use = frame._audit_new_prediction_used.to_numpy(bool)
                out['actual_waveform_joint_BC'] = np.where(use, newbc, frame.parent_joint_BC.to_numpy())
                out['new_prediction_used_in_score'] = use.astype(float)
            else:
                out['actual_waveform_joint_BC'] = newbc
                out['new_prediction_used_in_score'] = 1.
            out['single_classifier_used'] = float(args.engine == 'predictive' and not (archived or replay))
            return out
        def consensus(items, method):
            out = old_consensus(items, method)
            extra = pd.concat(items).groupby('pair_key')[list(fields)].mean().add_suffix('_mean').reset_index()
            # Inherited export may already aggregate embedding_only.
            extra = extra.drop(columns=[c for c in extra if c != 'pair_key' and c in out])
            return out.merge(extra, on='pair_key', validate='one_to_one')
        n.public_frame, n.dev.BASE.consensus_real = export, consensus

    def run(root, stage):
        original = n.infer
        def infer(frame, config):
            result = original(frame, config)
            if config['method'] == 'NEWPROFILE-PAIRED-MIN':
                old = h.isolated.ORIGINALS[config['deployment'], config['seed']]
                reference = co.BASE_INFER(frame, {**old, 'method': 'NODUP-DIRECT-REPLAY'})[0]
                used = frame.profile_active_i.to_numpy(bool) & frame.profile_active_j.to_numpy(bool) & (result[0] < reference)
                frame['_audit_new_prediction_used'] = used
            return result
        n.infer = infer
        return old_run(root, stage)
    r.install, n.run = install, run
    receipt = args.root/'contracts/ACTUAL_SCORE_EXPORT_ADDENDUM.json'
    if not receipt.exists():
        n.write_json(receipt, {'UTC': n.utc(), 'reporting_only': True,
            'engine': args.engine, 'raw_score_functions_unchanged': True,
            'actual_waveform_joint_BC': 'Original neural BC for NODUPreplay/inactive lower-envelope;new predictive BC for actualupdate;unknown for archivedPATHblend is NaN.',
            'legacy_joint_BC': 'Retained diagnostic column;use explicitly namedactualcolumnforattribution.',
            'seed_budget': 'Write a separate explicit-seed budget table;do not mutate inherited old mislabeled table.',
            'script_sha256': n.sha(Path(__file__))})
        shutil.copy2(__file__, args.root/'scripts/selected_score_audit_runtime.py')
    filename = 'mcwf_independent_predictive_scoring_20260909.py' if args.engine == 'predictive' else 'mcwf_selected_predictive_joint_control_20260909.py'
    sys.argv = [filename, '--root', str(args.root), '--data-root', str(args.data_root),
                '--predictive-root', str(args.predictive_root), '--stage', args.stage]
    if args.selection_root:
        sys.argv += ['--selection-root', str(args.selection_root)]
    runpy.run_path(str(P/'scripts/experiments'/filename), run_name='__main__')
    if args.stage == 'real':
        rows = []
        for path in sorted(args.root.glob('results/*/gwtc*/seed_*/real/*_all_pairs.parquet')):
            method, dep, seed = path.relative_to(args.root/'results').parts[:3]
            mode = 'fusion' if path.name.startswith('fusion') else 'waveform'
            frame = pd.read_parquet(path).sort_values('rank', kind='mergesort')
            for budget in (10, 20, 50, 100):
                rows.append(n.dev.budget_row(frame, method, dep, mode, budget, seed=int(seed[5:])))
        n.write_csv(args.root/'tables/PER_SEED_PE_OFFICIAL_BUDGETS_EXPLICIT.csv', rows)


if __name__ == '__main__':
    main()
