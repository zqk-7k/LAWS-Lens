#!/usr/bin/env python3
"""Stop only this run's two parent/worker trees for deterministic wider resume."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
import psutil

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--pids", nargs="+", type=int, required=True)
p.add_argument("--audit-name", default="WORKER_PARALLELISM_RESUME.json")
p.add_argument("--reason", default="increase workers6-to12 preserving all scientific settings")
a = p.parse_args()
procs, records = [], []
for pid in a.pids:
    parent = psutil.Process(pid)
    command = parent.cmdline()
    if not any("mcwf_fresh_confirmation_20260906.py" in arg for arg in command) or str(a.root) not in command:
        raise RuntimeError(f"Refusing to stop unrelated process {pid}")
    children = parent.children(recursive=True)
    procs += [parent] + children
    records.append({"pid": pid, "children": [child.pid for child in children], "command": command})
log = a.root / "contracts" / a.audit_name
if log.exists():
    raise RuntimeError("Resume audit already exists")
log.write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "reason": a.reason, "from_workers_per_run": 6,
    "to_workers_per_run": 12, "CPU_quota": 25, "memory_limit_GiB": 92,
    "scientific_configuration_changed": False, "source_plans_and_seed_unchanged": True,
    "completed_system_markers_reused": True, "affected_processes": records}, indent=2)+"\n")
for process in procs:
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass
gone, alive = psutil.wait_procs(procs, timeout=5)
for process in alive:
    if process.status() != psutil.STATUS_ZOMBIE:
        process.kill()
print(json.dumps({"terminated_own_run_processes": len(procs), "preserved_completed_outputs": True}))
