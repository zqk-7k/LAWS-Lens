#!/usr/bin/env python3
"""Execute a frozen script with process resource receipts, without GNU time."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    if a.receipt.exists():
        raise RuntimeError('Existing resource receipt must not be overwritten')
    digest = hashlib.sha256(a.script.read_bytes()).hexdigest()
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    sys.argv = [str(a.script), *a.arguments]
    error = None
    try:
        runpy.run_path(str(a.script), run_name='__main__')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        with a.receipt.open('x') as stream:
            json.dump({'start_UTC': started, 'end_UTC': datetime.now(timezone.utc).isoformat(),
                'wall_seconds': time.monotonic() - tick, 'user_CPU_seconds': usage.ru_utime,
                'system_CPU_seconds': usage.ru_stime, 'peak_process_RSS_KiB': usage.ru_maxrss,
                'script': str(a.script), 'script_sha256': digest, 'argv': a.arguments,
                'error': error, 'scope': 'This process only; not whole workflow or GPU peak.',
                'no_scientific_configuration_changed': True}, stream, indent=2)


if __name__ == '__main__':
    main()
