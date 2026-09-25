#!/usr/bin/env python3
"""Wait for each frozen run's inputs, without changing calibration rules."""
import argparse
from pathlib import Path
import runpy
import shutil
import sys
import time

P = Path('/root/autodl-tmp/gw-catalog')
SCRIPTS = P/'scripts/experiments'
sys.path.insert(0, str(SCRIPTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('base', 'continuous', 'predictive'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path)
    args = parser.parse_args()
    import mcwf_independent_waveform_calibration_20260909 as base
    n = base.n
    started = time.monotonic()
    receipt = args.root/'contracts/PER_RUN_READY_SCHEDULING.json'
    if receipt.exists():
        raise RuntimeError('Scheduling receipt already exists; inspect before resuming')
    n.write_json(receipt, {'UTC': n.utc(), 'scheduling_only': True,
        'engine': args.engine, 'data': str(args.data_root), 'maximum_wait_seconds': 7200,
        'per_run_requirement': 'data, predictions and all planned physical profiles complete before fitting that run',
        'no_analysis_changes': True, 'time_sky_weights_inputs_unchanged': True,
        'script_sha256': n.sha(Path(__file__))})
    shutil.copy2(__file__, args.root/'scripts/ready_calibration_runtime.py')

    def wait(dep):
        paths = [args.data_root/f'data/{dep}/COMPLETE.json',
                 args.data_root/f'contracts/{dep}_PREDICTIONS_COMPLETE.json',
                 args.data_root/f'contracts/{dep}_PROFILES_COMPLETE.json']
        while not all(path.exists() for path in paths):
            if time.monotonic()-started > 7200:
                raise RuntimeError('HOLD_INPUT_WAIT_LIMIT:'+dep)
            if shutil.disk_usage(args.data_root).free < 20*2**30:
                raise RuntimeError('HOLD_DISK_LIMIT')
            print('WAIT_FROZEN_RUN_INPUTS', dep, [str(p) for p in paths if not p.exists()], flush=True)
            time.sleep(20)
        dest = args.root/f'audit/{dep}_READY_INPUTS.json'
        if not dest.exists():
            n.write_json(dest, {'UTC': n.utc(), 'deployment': dep,
                'files': [{'path': str(p), 'sha256': n.sha(p)} for p in paths]})

    if args.engine == 'predictive':
        import mcwf_independent_predictive_calibration_20260909 as engine
        original = engine.inputs
        def inputs(dep):
            wait(dep)
            return original(dep)
        engine.inputs = inputs
        engine.ROOT, engine.DATA = args.root, args.data_root
        engine.calibrate()
    else:
        original = base.panels
        def panels(dep, seed):
            wait(dep)
            return original(dep, seed)
        base.panels = panels
        sys.argv = ['waveform_audit_export_runtime.py', '--engine', args.engine,
                    '--root', str(args.root), '--data-root', str(args.data_root),
                    '--stage', 'calibrate']
        if args.audit_root is not None:
            sys.argv += ['--audit-root', str(args.audit_root)]
        runpy.run_path(str(SCRIPTS/'mcwf_waveform_audit_export_runtime_20260909.py'), run_name='__main__')


if __name__ == '__main__':
    main()
