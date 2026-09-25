"""Verify and replay actual core+extension archives in a fresh extraction directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--attempt',default='delivered_r2_replay')
    a=p.parse_args()
    root=a.release
    packages=root/'packages'
    index=json.loads((packages/'RELEASE_PACKAGE_INDEX_R2.json').read_text())
    dest=root/'verification'/a.attempt
    dest.mkdir(exist_ok=False)
    report={'status':'RUNNING','archives':[],'stages':[],
            'extraction_root':str(dest),
            'scope':'Fresh extraction and isolated locked environment on same physical server; not a second host.'}
    def save():
        (packages/'DELIVERED_EXTENSION_REPLAY.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in index['packages']:
        path=packages/row['file']
        if sha(path)!=row['sha256']:
            raise RuntimeError('Packaged SHA mismatch')
        report['archives'].append({'file':row['file'],'sha256_pass':True})
        if row['role'] in ('C_NATIVE_MAPS','WHEELHOUSE'):
            continue
        with tarfile.open(path,'r|*') as archive:
            for member in archive:
                relative=Path(member.name)
                if not member.isfile() or relative.is_absolute() or '..' in relative.parts or (dest/relative).exists():
                    raise RuntimeError('Unsafe or overlapping archived member')
                archive.extract(member,dest,filter='data')
        print('EXTRACTED',row['role'],flush=True)
        save()
    all_checks=[]
    for manifest in (dest/'package_manifests').glob('*.json'):
        values=json.loads(manifest.read_text())
        # Each modular archive carries an ordinary list of content hashes.
        if isinstance(values,list):
            for entry in values:
                filename=entry.get('path')
                if filename and 'sha256' in entry:
                    passed=sha(dest/filename)==entry['sha256']
                    all_checks.append(passed)
                    if not passed:
                        raise RuntimeError('Extracted file hash mismatch')
    report['extracted_file_hash_checks']=len(all_checks)
    report['extracted_hash_failures']=sum(not x for x in all_checks)
    save()
    for stage in ('figures','all-subsets','test-real-models','recovery'):
        start=time.monotonic()
        log=packages/('REPLAY_'+a.attempt+'_'+stage+'.log')
        with log.open('x') as stream:
            proc=subprocess.run([sys.executable,'-B',str(dest/'extension/scripts/reproduce_extension.py'),stage,
                                 '--release',str(dest)],stdout=stream,stderr=subprocess.STDOUT)
        report['stages'].append({'stage':stage,'returncode':proc.returncode,'wall_seconds':time.monotonic()-start,
                                'log':log.name,'status':'PASS' if proc.returncode==0 else 'FAIL'})
        print(stage,report['stages'][-1]['status'],flush=True)
        save()
        if proc.returncode:
            report['status']='FAIL'
            save()
            raise RuntimeError('Actual delivered extension failed')
    report['status']='PASS'
    guards=[]
    for name in ('replay_all_subsets','replay_test_real_models','replay_recovery'):
        path=dest/f'verification/PORTABILITY_{name}.json'
        item=json.loads(path.read_text())
        guards.append({'script':name,'status':item['status'],'violations':item['violations']})
        if item['status']!='PASS' or item['violations']:
            report['status']='FAIL'
    report['historical_read_isolation']=guards
    changed=[]
    for manifest in (dest/'package_manifests').glob('*.json'):
        values=json.loads(manifest.read_text())
        if isinstance(values,list):
            for entry in values:
                filename=entry.get('path')
                if filename and 'sha256' in entry and sha(dest/filename)!=entry['sha256']:
                    changed.append(filename)
    report['extracted_payload_changed_by_replay']=changed
    if changed or report['status']!='PASS':
        report['status']='FAIL'
        save()
        raise RuntimeError('Post-replay integrity/isolation check failed')
    report['does_not_certify']=['Full public-corpus download','Full population retraining','New-host execution',
                               'Original layouts for 11 figures','Licenses or real-candidate significance']
    save()
    index['delivered_archive_replay']='PASS'
    index['delivered_archive_replay_report']='DELIVERED_EXTENSION_REPLAY.json'
    (packages/'RELEASE_PACKAGE_INDEX_R2.json').write_text(json.dumps(index,indent=2)+'\n')
    rows=''.join(r['sha256']+'  '+r['file']+'\n' for r in index['packages'])
    for name in ('RELEASE_PACKAGE_INDEX_R2.json','DELIVERED_EXTENSION_REPLAY.json','READ_FIRST_R2_CN.md'):
        rows+=sha(packages/name)+'  '+name+'\n'
    (packages/'SHA256SUMS_R2.txt').write_text(rows)
    print('DELIVERED_R2_PASS',flush=True)

if __name__=='__main__':
    main()
