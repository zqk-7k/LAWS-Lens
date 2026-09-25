#!/usr/bin/env python3
"""Watch frozen transitive runtime files; pause owned processes on mismatch."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import psutil

P = Path('/root/autodl-tmp/gw-catalog')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--owned-pids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    root = args.root
    expected = {}

    def add(path, digest):
        if path in expected and expected[path] != digest:
            raise RuntimeError('Different frozen hashes for '+path)
        expected[path] = digest

    with (args.pilot_root/'manifest/RUNTIME_DEPENDENCIES.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            add(row['source'], row['sha256'])
    with (root/'manifest/INPUT_SHA256.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['path'].endswith('.py'):
                add(row['path'], row['sha256'])
    frozen = json.loads((root/'contracts/COMPLETION_DRIVER_FROZEN.json').read_text())
    for path, digest in frozen['scripts'].items():
        add(path, digest)

    def mismatches():
        failed = []
        for path, digest in expected.items():
            if not Path(path).is_file() or sha(Path(path)) != digest:
                failed.append(path)
        return failed

    def hold(failed):
        write(root/'audit/HOLD_RUNTIME_HASH_CHANGE.json', {'UTC': utc(), 'changed_paths': failed,
            'action': 'SIGSTOP only this experiment measurement/driver trees; no deletion or automatic repair.'})
        seen = set()
        for pid in args.owned_pids:
            try:
                parent = psutil.Process(pid)
                family = [parent]+parent.children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            for process in family:
                if process.pid in seen or process.pid == os.getpid():
                    continue
                try:
                    process.send_signal(signal.SIGSTOP)
                    seen.add(process.pid)
                except psutil.NoSuchProcess:
                    pass
        raise RuntimeError('Frozen runtime mismatch; owned calculations paused')

    failed = mismatches()
    if failed:
        hold(failed)
    records = []
    for name, digest in expected.items():
        path = Path(name)
        dest = root/'scripts/runtime_dependencies'/path.relative_to(P)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha(dest) != digest:
                raise RuntimeError('Runtime snapshot conflict')
        else:
            shutil.copy2(path, dest)
        records.append({'source': name, 'sha256': digest, 'snapshot': str(dest.relative_to(root))})
    write(root/'manifest/TRANSITIVE_RUNTIME_SHA256.json', {'UTC': utc(), 'files': records})
    shutil.copy2(__file__, root/'scripts/shared_profile_runtime_guard.py')
    checks = 0
    with (root/'logs/RUNTIME_HASH_WATCH.jsonl').open('x') as stream:
        while True:
            failed = mismatches()
            if failed:
                hold(failed)
            checks += 1
            stream.write(json.dumps({'UTC': utc(), 'files': len(expected), 'failures': 0})+'\n')
            stream.flush()
            complete = root/'contracts/FROZEN_COMPUTATIONS_COMPLETE.json'
            driver_failure = root/'logs/COMPLETION_DRIVER_FAIL.json'
            if complete.exists() or driver_failure.exists():
                write(root/'audit/TRANSITIVE_RUNTIME_FINAL_CHECK.json', {'UTC': utc(), 'files': len(expected),
                    'checks': checks, 'hash_failures': 0, 'computations_complete': complete.exists(),
                    'driver_failure': driver_failure.exists()})
                break
            time.sleep(30)
    print('TRANSITIVE_RUNTIME_WATCH_COMPLETE', len(expected), checks, flush=True)


if __name__ == '__main__':
    main()
