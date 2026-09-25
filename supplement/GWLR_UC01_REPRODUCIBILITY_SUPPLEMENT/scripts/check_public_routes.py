"""Bounded acquisition checks; do not download the full public strain/PE corpus."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
import requests

root = Path(sys.argv[1])
rows = json.loads((root/'inputs/RAW_INPUT_ACQUISITION.json').read_text())
chosen = {}
for r in rows:
    if not r['acquisition_urls']:
        continue
    url = r['acquisition_urls'][0]
    host = urlsplit(url).hostname
    group = (r['runs'], r['kinds'], host)
    if r['kinds'] in ('GW_LMC_catalog', 'official_machine_table'):
        group = (r['input_id'],)
    chosen.setdefault(group, r)
results = []
for r in chosen.values():
    url = r['acquisition_urls'][0]
    try:
        if r['kinds'] == 'GW_LMC_catalog':
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            digest = hashlib.sha256(response.content).hexdigest()
            passed = digest == r['sha256']
            mode = 'FULL_DOWNLOAD_SHA256'
        else:
            response = requests.head(url, allow_redirects=True, timeout=30)
            passed = response.status_code == 200
            digest, mode = None, 'HEAD_ONLY_NOT_CONTENT_VERIFICATION'
        result = dict(input_id=r['input_id'], url=url, status=response.status_code,
                      check=mode, sha256=digest, passed=passed)
    except Exception as exc:
        result = dict(input_id=r['input_id'], url=url, passed=False, error=type(exc).__name__+': '+str(exc))
    results.append(result)
    print(r['input_id'], result.get('status'), result['passed'], flush=True)
(root/'reports/PUBLIC_ROUTE_CHECKS.json').write_text(json.dumps({'checks':results,
    'all_files_redownloaded':False, 'scope':'all three GW-LMC CSV content hashes; stratified public URL availability only'}, indent=2)+'\n')
