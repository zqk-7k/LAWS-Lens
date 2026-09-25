#!/usr/bin/env python3
"""Capture actual process start times and allocation, without changing jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    rows = []
    for pid in args.pids:
        p = psutil.Process(pid)
        rows.append({'pid': pid, 'command': p.cmdline(), 'create_time_unix': p.create_time(),
            'create_time_UTC': datetime.fromtimestamp(p.create_time(), timezone.utc).isoformat()})
    constraints = {}
    for name in ('cpu.max', 'memory.max', 'memory.current', 'memory.peak'):
        path = Path('/sys/fs/cgroup')/name
        constraints[name] = path.read_text().strip() if path.exists() else None
    with (args.root/'audit/RESOURCE_PROCESS_PROVENANCE.json').open('x') as stream:
        json.dump({'UTC': datetime.now(timezone.utc).isoformat(), 'processes': rows,
            'logical_CPUs_visible': psutil.cpu_count(), 'physical_CPUs_visible': psutil.cpu_count(logical=False),
            'host_RAM_bytes': psutil.virtual_memory().total, 'cgroup_constraints': constraints,
            'disk_usage_bytes': psutil.disk_usage(str(args.root))._asdict(),
            'note': 'Cgroup/host resources include other jobs. Monitor RSS is a sum over processes, not unique physical RAM. Pair seconds are elapsed wall time, not CPU seconds.'}, stream, indent=2)
    shutil.copy2(__file__, args.root/'scripts/shared_profile_resource_metadata.py')
    print('RESOURCE_PROCESS_PROVENANCE_CAPTURED', len(rows), flush=True)


if __name__ == '__main__':
    main()
