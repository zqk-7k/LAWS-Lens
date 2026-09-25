"""Record the original O4a release and its exact selected members."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import pandas as pd
import requests

root = Path(sys.argv[1])
project = Path('/root/autodl-tmp/gw-catalog')
source = project/'results/main_o3_mcwf_dev_v1_20260905T135000Z/source_data/official_O4'
response = requests.get('https://zenodo.org/api/records/18163632', timeout=60)
response.raise_for_status()
metadata = response.json()
(root/'provenance/O4A_OFFICIAL_RELEASE_18163632.json').write_text(json.dumps(metadata, indent=2))
remote = next(f for f in metadata['files'] if f['key']=='GWTC4_Lensing_DataRelease.zip')
url = 'https://zenodo.org/api/records/18163632/files/GWTC4_Lensing_DataRelease.zip/content'
records = json.loads((root/'inputs/RAW_INPUT_ACQUISITION.json').read_text())
for i, p in enumerate(sorted((source/'GWTC4_Lensing_DataRelease').rglob('*'))):
    if not p.is_file() or p.suffix not in ('.csv', '.json'):
        continue
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    records.append(dict(input_id='official_o4_'+str(i), original_path=str(p),
        resolved_path=str(p.resolve()), filename=p.name, kinds='official_machine_table',
        runs='O4a', events='', roles='post_ranking_audit_only', exists=True,
        bytes=p.stat().st_size, sha256=digest, acquisition_urls=[url],
        url_evidence=['https://zenodo.org/records/18163632',
            'archived extraction relative path; full publisher zip not re-downloaded in this audit'],
        acquisition_status='PUBLISHER_ZIP_MEMBER_ROUTE', archive_format='zip',
        archive_member=str(p.relative_to(source)), download_bytes=remote['size'],
        download_checksum=remote['checksum'].split(':',1)[1],
        download_checksum_algorithm=remote['checksum'].split(':',1)[0],
        target_relative_path='inputs/official_o4/'+str(p.relative_to(source))))
for name in ('GWTC41_OFFICIAL_NAME_MAPPING.csv', 'GWTC-4.1_gwosc_catalog.json'):
    p = source/name
    shutil.copy2(p, root/'provenance'/name)
shutil.copy2(source.parent.parent/'scripts/mcwf_official_o4_20260905.py',
             root/'provenance/O4_OFFICIAL_JOIN_REFERENCE.py')
(root/'inputs/RAW_INPUT_ACQUISITION.json').write_text(json.dumps(records, indent=2)+'\n')
pd.DataFrame([{**r, 'acquisition_urls': json.dumps(r['acquisition_urls']),
               'url_evidence': json.dumps(r['url_evidence'])} for r in records]).to_csv(
    root/'inputs/RAW_INPUT_ACQUISITION.csv', index=False, encoding='utf-8-sig')
report = json.loads((root/'reports/INPUT_INVENTORY_FINAL.json').read_text())
report.update(files=len(records), unique_contents=len({r['sha256'] for r in records}),
    unique_content_bytes=sum({r['sha256']:r['bytes'] for r in records}.values()),
    official_o4_zip_full_download_recheck=False)
(root/'reports/INPUT_INVENTORY_FINAL.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
