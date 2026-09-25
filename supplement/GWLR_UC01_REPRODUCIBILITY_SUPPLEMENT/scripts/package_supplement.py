"""Package immutable compact metadata and an optional offline wheel archive."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile

root=Path(sys.argv[1])
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

assert json.loads((root/'reports/REPRODUCIBILITY_FINAL_STATUS.json').read_text())['clean_environment_verified']
wheelhouse=Path('/root/autodl-tmp/gwlr_uc01_wheelhouse_20260919')
built=root/'environment/built_wheels'; built.mkdir(exist_ok=True)
for p in list(wheelhouse.glob('astroplan-*.whl'))+list(wheelhouse.glob('pims-*.whl')):
    shutil.copy2(p,built/p.name)
notices=[]
for row in json.loads((root/'environment/WHEEL_ARTIFACTS.json').read_text()):
    p=wheelhouse/row['filename']
    assert sha(p)==row['sha256']
    with zipfile.ZipFile(p) as z:
        notices.append(dict(wheel=p.name, license_files=[n for n in z.namelist()
            if any(x in Path(n).name.lower() for x in ('license','copying','notice'))]))
(root/'environment/THIRD_PARTY_NOTICE_LOCATIONS.json').write_text(json.dumps(notices,indent=2))

# Wheels are already compressed. A tar wrapper avoids another expensive compression.
wheel_tar=root/'package/GWLR_UC01_verified_wheelhouse.tar'
with tarfile.open(wheel_tar,'x') as tar:
    for row in json.loads((root/'environment/WHEEL_ARTIFACTS.json').read_text()):
        tar.add(wheelhouse/row['filename'],arcname='wheelhouse/'+row['filename'],recursive=False)
    tar.add(root/'environment/WHEEL_ARTIFACTS.json',arcname='WHEEL_ARTIFACTS.json')
    tar.add(root/'environment/requirements.linux-x86_64-cp312-cu128.lock',arcname='requirements.lock')
wheel_hash=sha(wheel_tar)
wheel_tar.with_suffix('.tar.sha256').write_text(wheel_hash+'  '+wheel_tar.name+'\n')

pub=json.loads((root/'publication/ARCHIVE_UPLOAD_PLAN.json').read_text())
pub['optional_offline_software_bundle']=dict(path=str(wheel_tar),filename=wheel_tar.name,
    bytes=wheel_tar.stat().st_size,sha256=wheel_hash,public_url=None,
    license_review_required=True,mandatory_for_publication=False,
    alternative='hash lock plus official package download sources; locally built wheels included in supplement')
(root/'publication/ARCHIVE_UPLOAD_PLAN.json').write_text(json.dumps(pub,ensure_ascii=False,indent=2)+'\n')

files=[]
for p in sorted(root.rglob('*')):
    relative=p.relative_to(root)
    if not p.is_file() or relative.parts[0] in ('package','reconstruction_assets','manifest'):
        continue
    if p.name=='NUMERICAL_OUTPUTS.npz' or '__pycache__' in relative.parts:
        continue
    if p.suffix in ('.py','.md','.json','.txt','.csv','.log','.lock'):
        text=p.read_text(errors='replace')
        if re.search(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=]{30,}',text):
            raise RuntimeError('Private-key material in release: '+str(relative))
    files.append(p)
manifest=root/'manifest'; manifest.mkdir(exist_ok=True)
rows=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
with (manifest/'FILES_SHA256.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(rows)
(manifest/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in rows))
files.extend([manifest/'FILES_SHA256.csv',manifest/'SHA256SUMS.txt'])
bundle=root/'package/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT_20260919.tar.gz'
with tarfile.open(bundle,'x:gz',compresslevel=6) as tar:
    for p in files:
        tar.add(p,arcname='GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/'+str(p.relative_to(root)),recursive=False)
digest=sha(bundle)
bundle.with_suffix(bundle.suffix+'.sha256').write_text(digest+'  '+bundle.name+'\n')
expected={r['path']:r['sha256'] for r in rows}
verified=0
with tarfile.open(bundle) as tar:
    for m in tar:
        key=str(Path(m.name).relative_to('GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT'))
        if key in expected:
            assert hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()==expected[key]
            verified+=1
assert verified==len(expected)
report=dict(state='PASS',bundle=str(bundle),bytes=bundle.stat().st_size,sha256=digest,
    member_count=len(files),verified_internal_files=verified,
    wheel_archive=str(wheel_tar),wheel_archive_bytes=wheel_tar.stat().st_size,
    wheel_archive_sha256=wheel_hash,publicly_published=False)
(root/'package/FINAL_DELIVERY.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
