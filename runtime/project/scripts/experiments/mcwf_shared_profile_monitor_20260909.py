#!/usr/bin/env python3
"""Read-only process/resource observation, with append-only measurement receipts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil


def main(root, pid):
    output=root/'logs/RESOURCE_MONITOR.jsonl'
    with output.open('x') as stream:
        while True:
            tick=datetime.now(timezone.utc).isoformat()
            try:
                parent=psutil.Process(pid)
                processes=[parent]+parent.children(recursive=True)
                active=parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                processes=[]; active=False
            rss=cpu=0.; count=0
            for process in processes:
                try:
                    rss+=process.memory_info().rss
                    times=process.cpu_times(); cpu+=times.user+times.system; count+=1
                except psutil.NoSuchProcess:
                    continue
            receipts=[]
            for path in (root/'pairs').glob('*.json'):
                try:
                    receipts.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    pass
            record={'UTC':tick,'parent_pid':pid,'parent_alive':active,
                'processes':count,'summed_RSS_GiB':rss/1024**3,'live_process_cpu_seconds':cpu,
                'disk_free_GiB':psutil.disk_usage(str(root)).free/1024**3,
                'receipts':len(receipts),'failed_receipts':sum(x.get('status')=='FAIL'for x in receipts)}
            stream.write(json.dumps(record)+'\n'); stream.flush()
            if not active:
                break
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args();main(args.root,args.pid)
