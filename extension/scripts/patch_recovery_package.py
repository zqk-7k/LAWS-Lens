"""Preserve a failed archive attempt and package the isolated diagnostic loader."""
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import sys
import re

p=argparse.ArgumentParser()
p.add_argument('--release',required=True,type=Path)
a=p.parse_args()
root=a.release
packages=root/'packages'
sys.path.insert(0,str(root/'extension/scripts'))
from package_extension import bundle,sha
old=json.loads((packages/'RELEASE_PACKAGE_INDEX_R2.json').read_text())
failed=json.loads((packages/'DELIVERED_EXTENSION_REPLAY.json').read_text())
if failed['status']!='FAIL':
    raise RuntimeError('Expected preserved failed packaging attempt')
previous=root/'extension/evidence/first_packaged_recovery_import_failure'
previous.mkdir(exist_ok=False)
shutil.copy2(packages/'DELIVERED_EXTENSION_REPLAY.json',previous/'REPORT.json')
for path in packages.glob('REPLAY_*.log'):
    shutil.copy2(path,previous/path.name)
for path in (root/'verification/delivered_r2_replay/verification').glob('extension_*/recovery_0.log'):
    shutil.copy2(path,previous/'RECOVERY_TRACEBACK.log')
repair={'kind':'Packaging dependency isolation, not scientific algorithm change',
    'failure':'A spawned recovery worker imported legacy experiment globals while loading posterior map diagnostics.',
    'fix':'Compile the exact frozen raster_probability/posterior_metrics function AST and two constants only. No numerical function edits.',
    'scope':'The original signal generation, complete bank, matched-filter selection, BAYESTAR and map data stay unchanged.',
    'guard':'Historical project reads remain blocked in spawned workers.',
    'extra_check':'MOC UNIQ integer identifiers must match exactly; area, entropy and truth-coverage diagnostics also compared.',
    'failed_archive_preserved':True,'historical_results_modified':False}
(root/'extension/reports/RECOVERY_IMPORT_REPAIR.json').write_text(json.dumps(repair,indent=2)+'\n')
diagnostic=root/'verification/diagnostic_loader_selftest/REPORT.json'
if json.loads(diagnostic.read_text())['status']!='PASS':
    raise RuntimeError('Isolated diagnostic selftest not passed')
shutil.copy2(diagnostic,root/'extension/evidence/DIAGNOSTIC_LOADER_SELFTEST.json')
shutil.copy2(__file__,root/'extension/scripts/patch_recovery_package.py')
patterns=[re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
          re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
          re.compile(r'''(?i)\b[A-Za-z_]*(?:password|passwd|access_token|api_token|secret_key|zenodo_token|ssh_pass|github_token|auth_token)[A-Za-z_]*["']?\s*[=:]\s*["']([^"'\n]{6,})["']'''),
          re.compile(r'https?://[^\s/:@]+:[^\s/@]{6,}@')]
scan={'files':0,'findings':[],'scope':'Patched extension plus new fixtures, textual formats',
      'full_security_certification':False}
for folder in (root/'extension',root/'fixtures/test_real',root/'fixtures/recovery'):
    for path in folder.rglob('*'):
        if not path.is_file() or path.suffix not in ('.py','.md','.json','.jsonl','.csv','.txt','.log'):
            continue
        scan['files']+=1
        for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
            for pattern in patterns:
                match=pattern.search(line)
                if match:
                    scan['findings'].append({'path':str(path.relative_to(root)),'line':number,
                        'digest_only':hashlib.sha256(match.group(0).encode()).hexdigest()})
(root/'extension/reports/EXTENSION_SECRET_SCAN.json').write_text(json.dumps(scan,indent=2)+'\n')
if scan['findings']:
    raise RuntimeError('Secret-pattern scan failed')
paths=[p for p in (root/'extension').rglob('*') if p.is_file() and not p.is_symlink()
       and '__pycache__' not in p.parts and p.suffix!='.pyc']
replacement=bundle(root,paths,'reproduction_extension_r2p1','LAWS_Lens_v1_DRAFT_reproduction_extension_r2p1_20260924.tar.gz')
prior=next(x for x in old['packages'] if x['role']=='reproduction_extension_r2')
old['packages']=[replacement if x['role']=='reproduction_extension_r2' else x for x in old['packages']]
old['superseded_not_recommended']=[{'file':prior['file'],'reason':'Recovery worker diagnostic import isolation failed; preserved, do not extract with r2p1.'}]
old['total_bytes']=sum(x['bytes'] for x in old['packages'])
old['delivered_archive_replay']='PENDING_PATCHED_REPLAY'
(packages/'RELEASE_PACKAGE_INDEX_R2.json').write_text(json.dumps(old,indent=2)+'\n')
(packages/(replacement['file']+'.sha256')).write_text(replacement['sha256']+'  '+replacement['file']+'\n')
print(json.dumps(replacement),flush=True)
