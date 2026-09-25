#!/usr/bin/env python3
"""Finish frozen actual-data quadrature ladder; never launches bulk training."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main(root, oracle):
    project_scripts = Path('/root/autodl-tmp/gw-catalog/scripts/experiments')
    spec = importlib.util.spec_from_file_location('actual_base', root/'scripts/et3_physical_trigger_repair_20260915_v2.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    lock = root/'contracts/DATA_GATE_RUNNER_STARTED.json'
    if lock.exists():
        raise RuntimeError('Runner already started; inspect status before any restart')
    base.write(lock, {'utc': base.utc(), 'pid': os.getpid(), 'code_sha256': base.sha(__file__),
        'actions': ['wait_for_existing_q64', 'audit32_64', 'q128_if_needed', 'pixel_and_ranking_audits'],
        'no_model_or_prior_changes': True, 'no_bulk_or_encoder_training': True,
        'maximum_wait_for_existing_stage_seconds': 10800})
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    script_names = ['et3_pixel_coverage_audit_20260915.py', 'et3_rank_stability_20260915.py',
                    'et3_plot_actual_sky_20260915_v2.py']
    for name in script_names:
        shutil.copy2(project_scripts/name, root/'scripts'/name)
    start = time.monotonic()
    while not (root/'contracts/DATA_SKY_Q64_RESULT.json').exists():
        if time.monotonic()-start > 10800:
            raise RuntimeError('Existing q64 did not finish within controller wait bound')
        time.sleep(30)
    result = json.loads((root/'contracts/DATA_SKY_Q64_RESULT.json').read_text())
    if result['maps_complete'] != 24:
        raise RuntimeError('q64 missing/failed maps; retain logs, no silent omission')
    g64 = base.audit(root, 32, 64)
    highest, gates = 64, {'Q32_Q64': g64}
    if not g64['pass']:
        source = oracle/'build/q128'
        complete = json.loads((source/'COMPLETE.json').read_text())
        if complete['upstream_tests'] != 'PASS' or base.sha(source/'core.abi3.so') != complete['sha256']:
            raise RuntimeError('q128 isolated core verification failed')
        destination = root/'build/q128'
        if destination.exists():
            raise RuntimeError('q128 destination exists; no overwrite or accidental duplicate run')
        shutil.copytree(source, destination)
        base.write(root/'contracts/Q128_CORE_PROVENANCE.json', {
            'source': str(source), 'sha256': complete['sha256'],
            'reused_identical_compiled_quadrature_not_reused_sky_maps': True})
        command = [sys.executable, '-B', '-u', str(root/'scripts/et3_data_sky_pilot_20260915.py'),
                   '--root', str(root), '--maps', '128']
        base.write(root/'contracts/Q128_LAUNCH.json', {'utc': base.utc(), 'command': command,
            'threads_per_map': 4, 'workers': 2, 'frozen_gate': g64,
            'scientific_configuration_unchanged': True})
        subprocess.run(command, check=True)
        gates['Q64_Q128'] = base.audit(root, 64, 128)
        completed = json.loads((root/'contracts/DATA_SKY_Q128_RESULT.json').read_text())
        if completed['maps_complete'] == 24:
            highest = 128
    commands = [
        ['et3_pixel_coverage_audit_20260915.py', '--q', str(highest)],
        ['et3_rank_stability_20260915.py', '--qa', '32', '--qb', '64'],
        ['et3_plot_actual_sky_20260915_v2.py', '--q', str(highest)]]
    if highest == 128:
        commands.append(['et3_rank_stability_20260915.py', '--qa', '64', '--qb', '128'])
    for name, *args in commands:
        subprocess.run([sys.executable, '-B', str(root/'scripts'/name), '--root', str(root), *args], check=True)
    final = {'utc': base.utc(), 'state': base.FINAL, 'highest_complete_quadrature': highest,
        'quadrature_gates': gates, 'numerical_ladder_passed': any(v['pass'] for v in gates.values()),
        'pixel_gate': json.loads((root/'contracts'/f'PIXEL_GATE_Q{highest}.json').read_text()),
        'coverage': json.loads((root/'contracts'/f'DESCRIPTIVE_COVERAGE_Q{highest}.json').read_text()),
        'no_formal_population_coverage_claim': True, 'bulk_started': False,
        'encoder_training_started': False, 'heldout_test_opened': False,
        'oracle_intrinsics_used': False, 'full_BBH_PE': False}
    base.write(root/'contracts/DATA_DRIVEN_FINAL_RESULT.json', final)
    base.write(root/'STATUS.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root, args.oracle)
    except Exception:
        path = args.root/'contracts/DATA_GATE_RUNNER_FAILURE.json'
        if not path.exists():
            path.write_text(json.dumps({'state': 'HOLD_RUNNER_ERROR', 'traceback': traceback.format_exc()}, indent=2)+'\n')
        raise
