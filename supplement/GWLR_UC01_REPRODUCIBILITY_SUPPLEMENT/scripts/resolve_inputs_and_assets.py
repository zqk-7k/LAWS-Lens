"""Complete archived acquisition routes and package missing reconstruction inputs."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile

import pandas as pd
import requests

P = Path('/root/autodl-tmp/gw-catalog')
R = P/'results/gwlr_unified_c_physical_20260918T134500Z_r1'
DEST = Path(sys.argv[1])
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def save(name, obj):
    (DEST/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

source = DEST/'inputs/RAW_INPUT_ACQUISITION.json'
records = json.loads(source.read_text())
uses = json.loads((DEST/'inputs/INPUT_USES.json').read_text())
freeze = {r['path']: r['sha256'] for r in json.loads(
    (R/'completion/contracts/REAL_INPUT_MANIFEST_FREEZE.json').read_text())['files']}
repairs = []
for r in records:
    if not r['exists']:
        p = P/'runs/real_gwtc34_lensing_search_20260629_full_o4'/Path(r['original_path']).relative_to(P)
        if not p.is_file():
            raise RuntimeError('Missing actual file: '+str(p))
        old = r['original_path']
        r.update(original_path=str(p), resolved_path=str(p.resolve()), exists=True,
                 bytes=p.stat().st_size, sha256=sha(p))
        for u in uses:
            if u['path'] == old:
                u['path'] = str(p)
        repairs.append({'before': old, 'after': str(p), 'reason': 'O4a relative path anchored at frozen run root'})
    if r['original_path'] in freeze and r['sha256'] != freeze[r['original_path']]:
        raise RuntimeError('Changed frozen input: '+r['original_path'])

archive = P/'results/gwtc5_o4b_scheme_c_exact_20260831_20260831T093631Z/incoming/IGWN-GWTC5p0-29ebe06b7_25-Archived_Skymaps.tar.gz'
url = 'https://zenodo.org/api/records/20348005/files/'+archive.name+'/content'
response = requests.get('https://zenodo.org/api/records/20348005', timeout=60)
response.raise_for_status()
release = response.json()
save('provenance/GWTC5_PE_RELEASE_20348005.json', release)
remote = next(f for f in release['files'] if f['key'] == archive.name)
with archive.open('rb') as f:
    md5 = hashlib.file_digest(f, 'md5').hexdigest()
assert remote['checksum'] == 'md5:'+md5
with tarfile.open(archive) as tar:
    members = {Path(m.name).name: m for m in tar if m.isfile()}
    for r in records:
        if r['acquisition_status'] == 'NEEDS_SOURCE_RESOLUTION':
            m = members[r['filename']]
            stream = tar.extractfile(m)
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == r['sha256']
            r.update(acquisition_status='VERIFIED_PUBLISHER_ARCHIVE_MEMBER', acquisition_urls=[url],
                archive_member=m.name, download_bytes=archive.stat().st_size,
                download_sha256=sha(archive) if not any('download_sha256' in x for x in records) else
                    next(x['download_sha256'] for x in records if 'download_sha256' in x),
                publisher_checksum=remote['checksum'],
                url_evidence=['https://zenodo.org/records/20348005', 'exact local tar member SHA256 verified'])

# Include the public O3 machine tables behind the derived official audit columns.
o3 = P/'results/main_o3official_cfixed_v1_20260904_20260904T072435Z'
for name in ('O3b_lensing_datafile_Figure1_ML_PO_FPPs.csv', 'o3b_lensing_datafile_Table1.csv',
             'O3b_lensing_datafile_Table2.xml.gz'):
    p = o3/'data'/name
    r = dict(input_id='official_'+name, original_path=str(p), resolved_path=str(p.resolve()),
        filename=name, kinds='official_machine_table', runs='O3', events='', roles='post_ranking_audit_only',
        exists=p.is_file(), bytes=p.stat().st_size, sha256=sha(p),
        acquisition_urls=['https://zenodo.org/api/records/7693837/files/'+name+'/content'],
        url_evidence=['https://zenodo.org/records/7693837'], acquisition_status='PUBLISHER_RECORD',
        target_relative_path='inputs/official/'+name)
    records.append(r)
shutil.copy2(o3/'contracts/OFFICIAL_FOLLOWUP_PROVENANCE.json', DEST/'provenance/OFFICIAL_O3_PROVENANCE.json')
save('provenance/PATH_RESOLUTION_AUDIT.json', repairs)

assets = DEST/'reconstruction_assets'
assets.mkdir(exist_ok=False)
asset_manifest = []
blocked = []
def copy(p, relative):
    if p.suffix == '.py':
        text = p.read_text(errors='replace')
        if re.search(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', text) or re.search(
            r'''(?i)(?:password|passwd|secret_key)\s*=\s*['"][^'"]{6,}['"]''', text):
            blocked.append(str(p))
            return
    target = assets/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    shutil.copy2(p, target)
    asset_manifest.append({'original_path': str(p), 'archive_path': str(relative),
                           'bytes': target.stat().st_size, 'sha256': sha(target)})

for r in records:
    if r['acquisition_status'] == 'PROJECT_DERIVED_FROZEN_PRODUCT':
        relative = Path(r['target_relative_path'])
        copy(Path(r['original_path']), relative)
        r['bundled_path'] = str(relative)
        r['bundled_archive'] = 'GWLR_UC01_reconstruction_inputs_and_code.tar.gz'
for folder in ('scripts', 'matchgw'):
    for p in sorted((P/folder).rglob('*.py')):
        copy(p, Path('project')/p.relative_to(P))
for p in sorted((R/'shared_time').rglob('*')):
    if p.is_file() and p.stat().st_size < 100*2**20:
        copy(p, Path('training/shared_time')/p.relative_to(R/'shared_time'))
for p in sorted((R/'plans').rglob('*')):
    if p.is_file() and p.suffix != '.npy':
        copy(p, Path('training/plans')/p.relative_to(R/'plans'))
for p in (P/'pyproject.toml',):
    copy(p, Path('project')/p.name)
copy(R/'bank/templates.csv', Path('training/bank/templates.csv'))
copy(R/'bank/COMPLETE.json', Path('training/bank/COMPLETE.json'))

for run in ('O3', 'O4a', 'O4b'):
    trigger = sorted((R/f'timings/pilot/{run}').glob('*/TRIGGERS.npz'))[0]
    for p in trigger.parent.iterdir():
        if p.is_file():
            copy(p, Path('training')/p.relative_to(R))
    source = sorted((R/f'data/{run}/main/validation').glob('*/C_PHYSICAL_short.npy'))[0]
    copy(source, Path('training')/source.relative_to(R))
    # Models/calibrations are already in the reviewed result archive.

save('inputs/RAW_INPUT_ACQUISITION.json', records)
save('inputs/INPUT_USES.json', uses)
pd.DataFrame([{**r, 'acquisition_urls': json.dumps(r['acquisition_urls']),
               'url_evidence': json.dumps(r['url_evidence'])} for r in records]).to_csv(
    DEST/'inputs/RAW_INPUT_ACQUISITION.csv', index=False, encoding='utf-8-sig')
save('inputs/RECONSTRUCTION_ASSETS.json', asset_manifest)
save('reports/SOURCE_CODE_CREDENTIAL_SCAN.json', {'blocked_paths': blocked,
    'actual_secrets_not_saved': True, 'not_a_complete_security_certification': True})
content = {r['sha256']: r['bytes'] for r in records}
save('reports/INPUT_INVENTORY_FINAL.json', dict(files=len(records), unique_contents=len(content),
    unique_content_bytes=sum(content.values()), missing=sum(not r['exists'] for r in records),
    unresolved=sum(r['acquisition_status']=='NEEDS_SOURCE_RESOLUTION' for r in records),
    real_events={'O3':62, 'O4a':74, 'O4b':86}, noise_parents_per_run=32,
    noise_parent_split={'train':20, 'validation':6, 'test':6},
    source_code_blocked=len(blocked), public_archive_published=False,
    raw_data_full_redownload_performed=False))
(DEST/'package').mkdir(exist_ok=True)
destination = DEST/'package/GWLR_UC01_reconstruction_inputs_and_code.tar.gz'
with tarfile.open(destination, 'x:gz', compresslevel=1) as tar:
    for row in asset_manifest:
        tar.add(assets/row['archive_path'], arcname=row['archive_path'], recursive=False)
digest = sha(destination)
destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
print(json.dumps({'files':len(records), 'unique_contents':len(content),
    'archive':str(destination), 'bytes':destination.stat().st_size, 'sha256':digest,
    'blocked_code_files': blocked}), flush=True)
