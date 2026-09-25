"""Run finite release-extension checks without training, tuning or publication."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['figures','all-subsets','test-real-models','recovery','all'])
    p.add_argument('--release',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args()
    root=a.release.resolve()
    scripts=root/'extension/scripts'
    tag='extension_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=root/'verification'/tag
    out.mkdir(parents=True,exist_ok=False)
    stages=['figures','all-subsets','test-real-models','recovery'] if a.stage=='all' else [a.stage]
    report={'status':'RUNNING','stages':[],'training':False,'retuning':False,'publication':False}
    guarded=[sys.executable,'-B',str(root/'scripts/portable_run.py'),'--release',str(root)]
    for stage in stages:
        if stage=='figures':
            commands=[[sys.executable,'-B',str(scripts/'rebuild_figures.py'),'--paper',str(root/'uploaded_snapshot/paper_snapshot'),
                       '--output',str(out/'figures')],
                      [sys.executable,'-B',str(scripts/'check_figure_rendering.py'),'--figures',str(out/'figures'),
                       '--output',str(out/'figure_rendering')]]
        else:
            name={'all-subsets':'replay_all_subsets.py','test-real-models':'replay_test_real_models.py',
                  'recovery':'replay_recovery.py'}[stage]
            commands=[guarded+[str(scripts/name),'--release',str(root),'--output-tag',tag+'/'+stage]]
        for number,command in enumerate(commands):
            log=out/f'{stage}_{number}.log'
            with log.open('x') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            report['stages'].append({'stage':stage,'step':number,'returncode':result.returncode,
                                    'log':str(log.relative_to(root)),'command':command})
            (out/'EXECUTION.json').write_text(json.dumps(report,indent=2))
            print(stage,number,'PASS' if result.returncode==0 else 'FAIL',flush=True)
            if result.returncode:
                report['status']='FAIL'
                (out/'EXECUTION.json').write_text(json.dumps(report,indent=2))
                raise SystemExit(result.returncode)
    report['status']='PASS'
    (out/'EXECUTION.json').write_text(json.dumps(report,indent=2))
    print(out/'EXECUTION.json',flush=True)

if __name__=='__main__':
    main()
