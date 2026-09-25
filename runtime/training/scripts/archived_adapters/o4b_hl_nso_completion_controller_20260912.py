#!/usr/bin/env python3
"""Bounded O4b completion sequence with explicit failure states and no adoption."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import o4b_hl_nso_data_training_20260912 as s

SEEDS = (2026091221, 2026091222, 2026091223)


def state(root, name, **extra):
    s.write(root/'COMPLETION_STATUS.json', {'utc': s.now(), 'state': name,
                                          'complete_results': False, **extra})


def wait_files(root, paths, label, timeout_hours=48):
    start = time.monotonic()
    while not all(p.exists() for p in paths):
        state(root, label, missing=[str(p) for p in paths if not p.exists()])
        s.disk_guard(root)
        for name in ('RUN_STATUS.json', 'RNC_V2_RUN_STATUS.json', 'SKY_SCORE_STATUS.json'):
            path = root/name
            if path.exists() and 'FAIL' in str(json.loads(path.read_text()).get('state', '')):
                raise RuntimeError('Upstream failure: '+name)
        if time.monotonic()-start > timeout_hours*3600:
            raise RuntimeError('HOLD_WAIT_TIMEOUT:'+label)
        time.sleep(30)


def command(root, filename, *args):
    path = root/'scripts'/filename
    if not path.exists():
        raise RuntimeError('Required completion implementation missing:'+filename)
    return [sys.executable, '-B', '-u', str(path), '--root', str(root), *map(str, args)]


def launch(root, label, cmd):
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    folder = root/'logs/completion'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{label}_{stamp}.log'
    f = path.open('a')
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '2',
                                                                     'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    f.close()
    return proc, path, cmd, time.monotonic()


def finish(root, label, item):
    proc, log, cmd, start = item
    rc = proc.wait()
    record = {'utc': s.now(), 'returncode': rc, 'command': cmd, 'log': str(log),
              'wall_seconds': time.monotonic()-start}
    s.write(root/'contracts/completion_tasks'/f'{label}_{log.stem}.json', record)
    if rc:
        raise RuntimeError('Task failed: '+label+'; '+str(log))


def run(root, label, cmd):
    state(root, label)
    finish(root, label, launch(root, label, cmd))


def main(root):
    run(root, 'REGISTER_SCORE_PROTOCOL', command(root, 'o4b_hl_nso_score_20260912.py', '--stage', 'register'))
    for seed in SEEDS:
        required = [root/f'models/short_encoder/seed_{seed}/summary.json',
                    root/f'rankncontrast_component_v2/models/RAW-PHASE-SOURCE/gwtc5/seed_{seed}/COMPLETE.json',
                    root/f'ordered_mass_predictor/models/gwtc5/seed_{seed}/COMPLETE.json',
                    root/f'models/MULTIRATE/gwtc5/seed_{seed}/COMPLETE.json',
                    root/f'models/CONDITIONAL-ETA-CHI/gwtc5/seed_{seed}/COMPLETE.json']
        wait_files(root, required, f'WAIT_ALL_WAVEFORM_COMPONENTS_{seed}')
        for split in ('development', 'validation'):
            run(root, f'INFER_{seed}_{split}', command(root, 'o4b_hl_nso_inference_20260912.py', '--seed', seed, '--split', split))
        wait_files(root, [root/f'sky_pair_scores/seed_{seed}/validation/COMPLETE.json'], f'WAIT_VALIDATION_SKY_{seed}')
        run(root, f'CALIBRATE_{seed}', command(root, 'o4b_hl_nso_score_20260912.py', '--stage', 'calibrate', '--seed', seed))
    run(root, 'FREEZE_BEFORE_TEST', command(root, 'o4b_hl_nso_score_20260912.py', '--stage', 'freeze'))
    maps = []
    for seed in SEEDS:
        maps.append((seed, launch(root, f'BAYESTAR_TEST_{seed}', command(root, 'o4b_hl_nso_bayestar_20260912.py',
                                                                    '--seed', seed, '--split', 'test', '--workers', 4))))
    for seed in SEEDS:
        run(root, f'INFER_TEST_{seed}', command(root, 'o4b_hl_nso_inference_20260912.py', '--seed', seed, '--split', 'test'))
    for seed, task in maps:
        state(root, 'WAIT_TEST_MAPS', seed=seed)
        finish(root, f'BAYESTAR_TEST_{seed}', task)
        run(root, f'TEST_SKY_SCORE_{seed}', command(root, 'o4b_hl_nso_sky_scores_20260912.py', '--seed', seed, '--split', 'test'))
    run(root, 'EVALUATE_LOCKED_TEST', command(root, 'o4b_hl_nso_evaluate_20260912.py', '--stage', 'injection'))
    run(root, 'REAL_HL_PREPROCESS', command(root, 'o4b_hl_nso_real_inputs_20260912.py'))
    run(root, 'REAL_SKY_SCORE', command(root, 'o4b_hl_nso_sky_scores_20260912.py', '--seed', SEEDS[0], '--split', 'real'))
    for seed in SEEDS:
        run(root, f'INFER_REAL_{seed}', command(root, 'o4b_hl_nso_inference_20260912.py', '--seed', seed, '--split', 'real'))
    run(root, 'REAL_FROZEN_RANKINGS', command(root, 'o4b_hl_nso_evaluate_20260912.py', '--stage', 'real'))
    run(root, 'SKY_CONVERGENCE_AUDIT', command(root, 'o4b_hl_nso_convergence_20260912.py'))
    run(root, 'REPORT_WITH_PENDING_PE', command(root, 'o4b_hl_nso_delivery_20260912.py', '--stage', 'report'))
    start = time.monotonic()
    while True:
        pe = root/'PE_DOWNLOAD_STATUS.json'
        status = json.loads(pe.read_text()) if pe.exists() else {}
        if status.get('state') == 'PE_INPUTS_VERIFIED':
            break
        if str(status.get('state', '')).startswith('HOLD'):
            raise RuntimeError('Public PE restoration failed; rankings preserved without invented PE values')
        state(root, 'WAIT_PUBLIC_PE_FOR_FINAL_AUDIT', pe_status=status)
        s.disk_guard(root)
        if time.monotonic()-start > 48*3600:
            raise RuntimeError('HOLD_PE_DOWNLOAD_TIMEOUT')
        time.sleep(60)
    run(root, 'PUBLIC_PE_AUDIT', command(root, 'o4b_hl_nso_pe_audit_20260912.py'))
    run(root, 'FINAL_DELIVERY', command(root, 'o4b_hl_nso_delivery_20260912.py', '--stage', 'package'))
    s.write(root/'COMPLETION_STATUS.json', {'utc': s.now(), 'state': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
                                          'complete_results': True, 'paper_or_historical_adoption': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    try:
        main(args.root)
    except BaseException:
        state(args.root, 'HOLD_COMPLETION_ERROR', traceback=traceback.format_exc())
        raise
