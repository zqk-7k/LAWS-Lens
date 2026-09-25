#!/usr/bin/env python3
"""Run an experiment with a unique durable log and resource summary."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
if not a.command:raise SystemExit('Command required')
a.log.parent.mkdir(parents=True,exist_ok=True)
start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
with a.log.open('x',encoding='utf-8') as handle:
    proc=subprocess.Popen(a.command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        handle.write(line);handle.flush();sys.stdout.write(line);sys.stdout.flush()
    code=proc.wait()
r=resource.getrusage(resource.RUSAGE_CHILDREN)
summary={'command':a.command,'start_utc':utc,'end_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,
         'wall_seconds':time.perf_counter()-start,'user_seconds':r.ru_utime,'system_seconds':r.ru_stime,'maxrss_kib':r.ru_maxrss}
with a.log.with_suffix('.runtime.json').open('x',encoding='utf-8') as h:json.dump(summary,h,indent=2)
raise SystemExit(code)
