"""Sample allocated resource use, without retaining process command lines."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


def run(root, out):
    target = out/'tables/RESOURCE_SAMPLES.csv'
    with target.open('a') as f:
        columns = ['utc', 'stage', 'cgroup_memory_GiB', 'cgroup_peak_GiB', 'disk_free_GiB', 'gpu_memory_MiB', 'gpu_utilization_percent']
        writer = csv.DictWriter(f, columns)
        if target.stat().st_size == 0: writer.writeheader()
        while True:
            try:
                status = json.loads((out/'RUN_STATUS.json').read_text())
                stage = status.get('stage', status.get('state', 'UNKNOWN'))
            except (OSError, ValueError): stage = 'STATUS_UPDATE'
            if stage == 'FINAL_REPORT_PACKAGE' or stage.startswith('HOLD_'):
                break
            value = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            parts = value.stdout.strip().split(',')
            mem = Path('/sys/fs/cgroup/memory.current')
            peak = Path('/sys/fs/cgroup/memory.peak')
            writer.writerow(dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                cgroup_memory_GiB=int(mem.read_text())/2**30,
                cgroup_peak_GiB=int(peak.read_text())/2**30 if peak.exists() else '',
                disk_free_GiB=shutil.disk_usage(root).free/2**30,
                gpu_memory_MiB=parts[0] if len(parts) == 2 else '', gpu_utilization_percent=parts[1] if len(parts) == 2 else ''))
            f.flush()
            time.sleep(15)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.out)
