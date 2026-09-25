"""Resumable one-way completion, including evaluation, real audit and delivery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import traceback

import uab_completion as c
STATUS_LOCK = threading.Lock()


def status(stage, **data):
    record = dict(stage=stage, utc=c.U.now(), controller_pid=os.getpid(),
                  continuation_directory=str(c.OUT), complete_results=False, **data)
    with STATUS_LOCK:
        c.write(c.OUT/'RUN_STATUS.json', record)
        c.write(c.ROOT/'RUN_STATUS.json', record)


def task(label, script, *args):
    receipt = c.OUT/'contracts/tasks'/f'{label}.json'
    if receipt.exists() and json.loads(receipt.read_text())['exit_code'] == 0:
        return
    c.U.guard(c.ROOT)
    log = c.OUT/'logs'/f'{label}_{time.time_ns()}.log'
    cmd = [sys.executable, '-B', '-u', str(c.OUT/'scripts'/script), '--root', str(c.ROOT), '--out', str(c.OUT), *args]
    start = time.monotonic()
    with log.open('x') as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        c.write(c.OUT/'contracts/tasks'/f'{label}_RUNNING.json', dict(pid=proc.pid, log=str(log), command=cmd, utc=c.U.now()))
        status(label, child_pid=proc.pid, log=str(log))
        code = proc.wait()
    c.write(receipt, dict(command=cmd, log=str(log), exit_code=code, elapsed_seconds=time.monotonic()-start,
                          utc=c.U.now(), script_sha256=c.U.sha(c.OUT/'scripts'/script)))
    if code:
        raise RuntimeError(label+' failed: '+str(log))


def running_maps(run, arm, split):
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd = path.read_bytes().split(b'\0')
            if not any(x.endswith(b'/uab_completion.py') for x in cmd):
                continue
            text = [x.decode(errors='replace') for x in cmd]
            def value(flag):
                return text[text.index(flag)+1] if flag in text else None
            if value('--stage') == 'maps' and value('--run') == run and value('--arm') == arm and value('--split') == split:
                matches.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return matches


def maps(run, arm, split, workers=12):
    dest = c.OUT/'maps'/run/arm/split
    while not (dest/'COMPLETE.json').exists() and running_maps(run, arm, split):
        status('WAIT_EXISTING_MAP_JOB', run=run, arm=arm, split=split, pids=running_maps(run, arm, split))
        time.sleep(20)
    task(f'MAPS_{run}_{arm}_{split}', 'uab_completion.py', '--stage', 'maps', '--run', run,
         '--arm', arm, '--split', split, '--workers', str(workers))


def validation():
    task('CORE_AUDIT', 'uab_completion.py', '--stage', 'audit')
    task('SCORE_UNIT_TESTS', 'uab_scoring.py', '--stage', 'tests')
    task('ORDERING_UNIT_TESTS', 'uab_real.py', '--stage', 'tests')
    task('EVALUATION_CONTRACT', 'uab_diagnostics.py', '--stage', 'contract')
    task('REAL_SCOPE_FREEZE', 'uab_real.py', '--stage', 'inventory')
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            maps(run, arm, 'validation')
            task(f'SKY_{run}_{arm}_validation', 'uab_completion.py', '--stage', 'sky-pairs', '--run', run, '--arm', arm, '--split', 'validation')
            for seed in c.U.SEEDS:
                for split in ('development', 'validation'):
                    marker = c.deployment(run, arm)/f'predictions_o4b/seed_{seed}/{split}/COMPLETE.json'
                    while not marker.exists():
                        info = json.loads((c.OUT/'contracts/INFERENCE_PREPARATION_STATUS.json').read_text())
                        if info['stage'].startswith('HOLD'):
                            raise RuntimeError('Validation inference failed: '+str(info))
                        if info['stage'] == 'ALL_VALIDATION_WAVEFORM_INFERENCE_COMPLETE':
                            raise RuntimeError('Inference controller ended but prediction absent')
                        status('WAIT_VALIDATION_INFERENCE', run=run, arm=arm, seed=seed, split=split)
                        time.sleep(10)
                task(f'CALIBRATE_{run}_{arm}_{seed}', 'uab_scoring.py', '--stage', 'calibrate', '--run', run, '--arm', arm, '--seed', str(seed))
    task('FINAL_SCORE_FREEZE', 'uab_scoring.py', '--stage', 'freeze')


def generate(run):
    label = 'GENERATE_TEST_'+run
    receipt = c.OUT/'contracts/tasks'/f'{label}.json'
    if receipt.exists() and json.loads(receipt.read_text())['exit_code'] == 0:
        return
    c.gate('test'); c.U.guard(c.ROOT)
    log = c.OUT/'logs'/f'{label}_{time.time_ns()}.log'
    command = [sys.executable, '-B', '-u', str(c.ROOT/'scripts/unified_ab.py'), '--root', str(c.ROOT),
               '--stage', 'generate', '--run', run, '--role', 'main', '--split', 'test', '--workers', '4']
    begin = time.monotonic()
    with log.open('x') as f:
        proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
        code = proc.wait()
    c.write(receipt, dict(command=command, exit_code=code, log=str(log), elapsed_seconds=time.monotonic()-begin, utc=c.U.now()))
    if code:
        raise RuntimeError(label+' failed: '+str(log))


def evaluate_seed(run, arm, seed):
    for split in ('validation', 'test'):
        for stage in ('catalog', 'subcatalogs'):
            task(f'EVAL_{stage}_{run}_{arm}_{seed}_{split}', 'uab_evaluate.py', '--stage', stage,
                 '--run', run, '--arm', arm, '--seed', str(seed), '--split', split)


def finish():
    validation()
    status('GENERATING_THREE_RUN_TEST_AFTER_FREEZE')
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(generate, c.U.RUNS))
    with ThreadPoolExecutor(max_workers=3) as map_pool:
        map_jobs = {(run, arm): map_pool.submit(maps, run, arm, 'test', 6)
                    for run in c.U.RUNS for arm in c.U.ARMS}
        for run in c.U.RUNS:
            for arm in c.U.ARMS:
                for seed in c.U.SEEDS:
                    task(f'INFER_{run}_{arm}_{seed}_test', 'uab_completion.py', '--stage', 'infer', '--run', run,
                         '--arm', arm, '--seed', str(seed), '--split', 'test')
                map_jobs[(run, arm)].result()
                task(f'SKY_{run}_{arm}_test', 'uab_completion.py', '--stage', 'sky-pairs', '--run', run, '--arm', arm, '--split', 'test')
                with ThreadPoolExecutor(max_workers=3) as pool:
                    list(pool.map(lambda seed: evaluate_seed(run, arm, seed), c.U.SEEDS))
    task('SUMMARIZE_INJECTION', 'uab_evaluate.py', '--stage', 'summarize')
    for run in c.U.RUNS:
        task('REAL_PREPARE_'+run, 'uab_real.py', '--stage', 'prepare', '--run', run)
        task('REAL_SKY_'+run, 'uab_real.py', '--stage', 'sky', '--run', run)
        for arm in c.U.ARMS:
            for seed in c.U.SEEDS:
                task(f'INFER_{run}_{arm}_{seed}_real', 'uab_completion.py', '--stage', 'infer', '--run', run,
                     '--arm', arm, '--seed', str(seed), '--split', 'real')
            task(f'REAL_RANK_{run}_{arm}', 'uab_evaluate.py', '--stage', 'real', '--run', run, '--arm', arm)
        task('REAL_PE_'+run, 'uab_real.py', '--stage', 'pe', '--run', run)
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            for split in ('validation', 'test'):
                task(f'RESOLUTION_{run}_{arm}_{split}', 'uab_diagnostics.py', '--stage', 'resolution', '--run', run,
                     '--arm', arm, '--split', split)
    task('RESOLUTION_SUMMARY', 'uab_diagnostics.py', '--stage', 'summarize')
    task('POPULATION_MODEL_AUDIT', 'uab_diagnostics.py', '--stage', 'population')
    task('FINAL_REPORT_PACKAGE', 'uab_delivery.py', '--stage', 'deliver')
    record = json.loads((c.OUT/'contracts/FINAL_DELIVERY.json').read_text())
    c.write(c.ROOT/'RUN_STATUS.json', record)
    c.write(c.OUT/'RUN_STATUS.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); c.initialize(a.root, a.out)
    (c.OUT/'contracts/tasks').mkdir(exist_ok=True)
    with (c.OUT/'contracts/COMPLETION_CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            finish()
        except Exception:
            record = dict(stage='HOLD_COMPLETION_ERROR', utc=c.U.now(), error=traceback.format_exc(),
                          complete_results=False, continuation_directory=str(c.OUT))
            c.write(c.OUT/'contracts'/f'FAILURE_{time.time_ns()}.json', record)
            c.write(c.OUT/'RUN_STATUS.json', record)
            c.write(c.ROOT/'RUN_STATUS.json', record)
            raise
