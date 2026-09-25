"""Finite release validation entry point. Never trains, tunes, or publishes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import uuid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['paper','models','features','injections','metrics-sky'])
    ap.add_argument('--release', type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args()
    root = a.release.resolve()
    common = [sys.executable,'-B',str(root/'scripts/portable_run.py'),'--release',str(root)]
    if a.stage=='paper':
        command = [sys.executable,'-B',str(root/'scripts/verify_paper.py'),'--release',str(root)]
    elif a.stage=='metrics-sky':
        script = root/'supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/scripts/verify_environment.py'
        command = common+[str(script),'--source-root',str(root/'runtime/training'),
            '--project-root',str(root/'runtime/project'),'--completion-root',str(root/'runtime/completion'),
            '--output',str(root/'verification/replay_metrics_sky')]
    else:
        script = {'models':'verify_models_full_context.py','features':'verify_features.py',
                  'injections':'verify_injections.py'}[a.stage]
        command = common+[str(root/'scripts'/script),'--release',str(root),
                          '--output-tag','replay_'+a.stage]
    logs = root/'logs/replay'
    logs.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logfile = logs / f'{a.stage}_{stamp}_{uuid.uuid4().hex[:8]}.log'
    with logfile.open('x') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    print(a.stage, 'PASS' if result.returncode==0 else 'FAIL', 'log:',logfile)
    raise SystemExit(result.returncode)

if __name__=='__main__':
    main()
