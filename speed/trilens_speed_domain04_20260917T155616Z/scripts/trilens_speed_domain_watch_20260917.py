"""Observe the bounded speed job and package its terminal state."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    import psutil
    root = args.root
    records = []
    begin = time.time()
    while time.time()-begin < 11100:
        alive = False
        processes = []
        try:
            parent = psutil.Process(args.pid)
            alive = (parent.status() != psutil.STATUS_ZOMBIE and
                     any('trilens_speed_domain_20260917.py' in s for s in parent.cmdline()))
            if alive:
                processes = [parent]+parent.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        rss = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not alive:
            break
        row = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   process_tree_rss_MiB=rss/2**20, process_count=len(processes),
                   available_RAM_GiB=psutil.virtual_memory().available/2**30,
                   free_disk_GiB=shutil.disk_usage(root).free/2**30,
                   load1=os.getloadavg()[0])
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'], capture_output=True,
                                 text=True, timeout=10, check=True).stdout.strip().splitlines()
            row['whole_GPU_memory_MiB'] = float(gpu[0].split(',')[0])
            row['whole_GPU_utilization_percent'] = float(gpu[0].split(',')[1])
        except Exception:
            row['whole_GPU_memory_MiB'] = None
            row['whole_GPU_utilization_percent'] = None
        records.append(row)
        with (root/'tables/resource_timeline.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(records)
        time.sleep(15)
    if args.log.exists():
        shutil.copy2(args.log, root/'logs/controller.log')
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    if not (root/'contracts/FINAL_STATUS.json').exists():
        (root/'contracts/WATCH_HOLD.json').write_text(json.dumps(dict(
            status='HOLD_MISSING_TERMINAL_STATUS', job_pid=args.pid, samples=len(records)), indent=2))
        return 2
    report = Path(__file__).with_name('trilens_speed_domain_report_20260917.py')
    with (root/'logs/report_and_package.log').open('x') as out:
        result = subprocess.run([sys.executable, '-B', str(report), '--root', str(root)],
                                stdout=out, stderr=subprocess.STDOUT)
    print(json.dumps(dict(root=str(root), report_exit_code=result.returncode)), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
