#!/usr/bin/env python3
"""Add actual classifier features and explicit seed labels without score changes."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('base', 'continuous'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path)
    parser.add_argument('--stage', choices=('calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    import mcwf_independent_waveform_calibration_20260909 as base
    n, r, h, co = base.n, base.r, base.h, base.co
    install_original = r.install
    fields = h.FEATURES+('continuous_joint_BC', 'continuous_mass_BC',
                        'actual_waveform_joint_BC', 'classifier_features_used')
    def install():
        install_original()
        original_export, original_consensus = n.public_frame, n.dev.BASE.consensus_real
        def export(frame, z, weights, method):
            result = original_export(frame, z, weights, method)
            for name in h.FEATURES+('continuous_joint_BC', 'continuous_mass_BC'):
                if name in frame:
                    result[name] = frame[name].to_numpy()
                else:
                    result[name] = np.nan
            replay, archived = method == 'NODUP-DIRECT-REPLAY', method == n.BASELINE
            if archived:
                result['actual_waveform_joint_BC'] = np.nan
            elif replay:
                result['actual_waveform_joint_BC'] = frame.parent_joint_BC.to_numpy()
            else:
                result['actual_waveform_joint_BC'] = np.exp(
                    frame.profile_log_mass_BC.to_numpy()+frame.profile_log_conditional_BC.to_numpy())
            result['classifier_features_used'] = float(not (replay or archived))
            return result
        def consensus(items, method):
            result = original_consensus(items, method)
            extra = pd.concat(items).groupby('pair_key')[list(fields)].mean().add_suffix('_mean').reset_index()
            return result.merge(extra, on='pair_key', validate='one_to_one')
        n.public_frame, n.dev.BASE.consensus_real = export, consensus
    r.install = install
    contract = args.root/'contracts/OUTPUT_FEATURE_AUDIT_ADDENDUM.json'
    if not contract.exists():
        n.write_json(contract, {'UTC': n.utc(), 'reporting_only': True, 'no_score_changes': True,
            'engine': args.engine, 'runtime_sha256': n.sha(Path(__file__)),
            'purpose': 'Export the seven actual single-waveform classifier inputs, continuous-overlap fields when applicable, and distinguish replay diagnostic features from actual replay inputs.',
            'legacy_joint_BC_column': 'Inherited profile diagnostic in base exporter;actual_waveform_joint_BC explicitly identifies the evidence used. NODUPreplay uses parent joint, not profile joint.',
            'seed_label_repair': 'After real stage write separate PER_SEED_PE_OFFICIAL_BUDGETS_EXPLICIT.csv with actual seed argument;leave legacy mislabeled summary untouched.',
            'config_reader': 'Canonical filename fallback only when receipt lacksfile;SHAchecked,receiptnevermodified.'})
        shutil.copy2(__file__, args.root/'scripts/waveform_audit_export_runtime.py')
    if args.engine == 'continuous':
        import mcwf_continuous_overlap_calibration_20260909 as engine
        engine.ROOT, engine.DATA, engine.AUDIT = args.root, args.data_root, args.audit_root
        engine.run_stage(args.stage)
    else:
        base.ROOT, base.DATA = args.root, args.data_root
        base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
        r.install()
        co.score.METHODS = base.METHODS
        co.score.matrices = co.matrices
        n.METHODS, n.load_panel, n.infer = base.METHODS, h.load_panel, base.infer
        def selections(root):
            record = json.loads((root/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
            path = root/record.get('file', 'configs/SELECTED_CONFIGURATIONS.json')
            if n.sha(path) != record['sha256']:
                raise RuntimeError('Frozen configuration changed')
            return json.loads(path.read_text())
        n.selections = selections
        if args.stage == 'calibrate':
            base.calibrate()
        else:
            n.run(args.root, args.stage)
    if args.stage == 'real':
        rows = []
        for path in sorted(args.root.glob('results/*/gwtc*/seed_*/real/*_all_pairs.parquet')):
            method, dep, seed_name = path.relative_to(args.root/'results').parts[:3]
            seed = int(seed_name[5:])
            mode = 'fusion' if path.name.startswith('fusion') else 'waveform'
            data = pd.read_parquet(path).sort_values('rank', kind='mergesort')
            for budget in (10, 20, 50, 100):
                rows.append(n.dev.budget_row(data, method, dep, mode, budget, seed=seed))
        n.write_csv(args.root/'tables/PER_SEED_PE_OFFICIAL_BUDGETS_EXPLICIT.csv', rows)


if __name__ == '__main__':
    main()
