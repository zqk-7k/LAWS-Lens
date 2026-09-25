"""Read-only raw-input provenance collection for GWLR-UC-01."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
R = P/'results/gwlr_unified_c_physical_20260918T134500Z_r1'
BASE = P/'results/gwlr_unified_snr_ab_20260917T113500Z_r4'
O3 = P/'results/main_o3official_cfixed_v1_20260904_20260904T072435Z'
O4B = P/'results/o4b_hl_bayestar_new_score_only_20260912T072746Z'
DEST = Path(sys.argv[1])
for name in ('inputs', 'provenance', 'scripts', 'reports', 'publication'):
    (DEST/name).mkdir(exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*2**20), b''):
            h.update(b)
    return h.hexdigest()

def output(name, value):
    (DEST/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

urls, metadata_rows = {}, []
def remember(url, source):
    if not isinstance(url, str) or not url.startswith('https://'):
        return
    parsed = urlsplit(url)
    if parsed.username or parsed.password or any(x in parsed.query.lower() for x in ('token=', 'signature=', 'credential=')):
        return
    parts = unquote(parsed.path).rstrip('/').split('/')
    key = parts[-2] if parts[-1] == 'content' else parts[-1]
    urls.setdefault(key, []).append({'url': url, 'source': str(source)})

def walk(obj, source):
    if isinstance(obj, dict):
        if any(isinstance(v, str) and v.startswith('https://') for v in obj.values()):
            metadata_rows.append((source, obj))
        for v in obj.values():
            walk(v, source)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, source)
    else:
        remember(obj, source)

candidates = [P/'results/gwtc_download_links_20260702/gwtc_download_links_all.csv']
candidates += list((P/'runs/gwtc4p1_data_completion_20260628/data').glob('*.csv'))
candidates += list((O3/'manifests').glob('*.csv'))
candidates += list((O3/'data').glob('*.csv'))
candidates += list((O4B/'manifests').glob('*.json'))
candidates += list((O4B/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830/data').glob('*.csv'))
candidates += list((P/'data/gwtc4p1/api_json').rglob('*.json'))
for name in ('real_gwtc_lensing_search_20260625', 'real_gwtc34_lensing_search_20260629_full_o4'):
    candidates += list((P/'runs'/name/'data').glob('*.csv'))
for f in sorted(set(candidates)):
    if not f.exists() or f.stat().st_size > 30*2**20:
        continue
    try:
        if f.suffix == '.csv':
            with f.open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            walk(rows, f)
        else:
            walk(json.loads(f.read_text()), f)
    except (ValueError, UnicodeError):
        continue

real = json.loads((R/'completion/contracts/REAL_INPUT_MANIFEST.json').read_text())
uses = []
def add(path, kind, **fields):
    if not path:
        return
    p = Path(path)
    if not p.is_absolute():
        p = P/p
    uses.append(dict(path=str(p), kind=kind, **fields))

for run, entries in real['runs'].items():
    for e in entries:
        common = dict(run=run, event=e['event_name'], gps=e['gps_time'], role='real_audit',
                      detectors=e['detectors'], posterior_group=e['pe_group'])
        add(e['pe_path'], 'PE_posterior', **common)
        add(e['map_path'], 'real_sky_source', **common)
        for d in e.get('detector_audit', []):
            add(d['path'], 'strain', **common, detector=d['detector'],
                psd_reference_start_gps=d['psd_reference_start_gps'])

# O4b raw strain/PSD files are in the frozen acquisition manifest.
strain_manifest = O4B/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830/data/strain_gwosc_download_manifest.csv'
frame = pd.read_csv(strain_manifest)
names = {e['event_name'] for e in real['runs']['O4b']}
for row in frame[frame.event_name.isin(names)].to_dict('records'):
    for key in ('local_path', 'psd_local_path'):
        add(strain_manifest.parent.parent/row[key], 'strain', run='O4b', event=row['event_name'],
            detector=row['detector'], role='real_audit', psd_reference_start_gps=row['psd_reference_start_gps'])

noise_rows = []
for run in ('O3', 'O4a', 'O4b'):
    noise = pd.read_parquet(R/f'plans/{run}/noise_plan.parquet')
    for row in noise.to_dict('records'):
        common = dict(run=run, event=row.get('source_event', ''), role='injection_noise',
            split=row['split'], noise_parent_uid=row['parent_uid'],
            noise_start_gps=row.get('reference_start_gps', row.get('start_gps')),
            noise_duration_s=256, noise_bank_index=row['noise_bank_index'])
        found = []
        if run == 'O4b':
            found = [(d, row[d.lower()+'_path']) for d in ('H1', 'L1')]
        else:
            for _, meta in metadata_rows:
                if meta.get('event_name') != row['source_event'] or meta.get('detector') not in ('H1', 'L1'):
                    continue
                for key in ('path', 'local_path', 'source_local_path'):
                    value = str(meta.get(key, ''))
                    if value.endswith(('.hdf5', '.h5')) and ('strain' in value):
                        paths = [Path(value)] if value.startswith('/') else [P/value,
                            P/'runs/real_gwtc_lensing_search_20260625'/value,
                            P/'runs/real_gwtc34_lensing_search_20260629_full_o4'/value]
                        found.extend((meta['detector'], str(f)) for f in paths if f.exists())
        for detector in ('H1', 'L1'):
            choices = sorted({p for d, p in found if d == detector and Path(p).exists()})
            if not choices:
                noise_rows.append({**common, 'detector': detector, 'status': 'RAW_PATH_UNRESOLVED'})
                continue
            path = choices[0]
            add(path, 'strain', **common, detector=detector)
            noise_rows.append({**common, 'detector': detector, 'path': path, 'status': 'FOUND'})
    add(R/f'plans/{run}/live_schedule.csv', 'frozen_calendar', run=run, role='injection_exposure')
    for name in ('noise_reference_bank.npy', 'noise_psd_bank.npy', 'noise_psd_frequency.npy'):
        add(R/f'plans/{run}/{name}', 'frozen_noise_product', run=run, role='reconstruction_exact_input')

gwlmc = Path('/root/autodl-tmp/GW-LMC')
commit = subprocess.check_output(['git', '-C', str(gwlmc), 'rev-parse', 'HEAD'], text=True).strip()
for p in sorted((gwlmc/'2.5PLUS/BBH/Any_Detected_SNR1').glob('*.csv')):
    remember('https://raw.githubusercontent.com/LensedGW/GW-LMC/'+commit+'/'+str(p.relative_to(gwlmc)), 'GW-LMC git HEAD')
    add(p, 'GW_LMC_catalog', role='source_lens_image_parameters', source_commit=commit)
output('provenance/GWLMC_GIT.json', {'commit': commit,
    'repository': 'https://github.com/LensedGW/GW-LMC',
    'branch_input': '2.5PLUS/BBH/Any_Detected_SNR1',
    'status': subprocess.check_output(['git', '-C', str(gwlmc), 'status', '--porcelain'], text=True)})
for p in [O3/'data/official_followup_stage_contract.csv', Path(real['official4_path'])]:
    add(p, 'frozen_official_audit_table', role='post_ranking_audit_only')

output('inputs/INPUT_USES.json', uses)
pd.DataFrame(noise_rows).to_csv(DEST/'inputs/NOISE_BLOCKS_AND_RAW_FILES.csv', index=False, encoding='utf-8-sig')
unique = {}
for item in uses:
    unique.setdefault(item['path'], []).append(item)
records = []
for i, (name, usage) in enumerate(sorted(unique.items())):
    p = Path(name)
    rows = urls.get(p.name, [])
    record = dict(input_id=f'input_{i:04d}', original_path=name, resolved_path=str(p.resolve()),
        filename=p.name, kinds=';'.join(sorted({u['kind'] for u in usage})),
        runs=';'.join(sorted({u.get('run', '') for u in usage} - {''})),
        events=';'.join(sorted({u.get('event', '') for u in usage} - {''})),
        roles=';'.join(sorted({u['role'] for u in usage})), exists=p.is_file(),
        acquisition_urls=list(dict.fromkeys(r['url'] for r in rows)),
        url_evidence=list(dict.fromkeys(r['source'] for r in rows)),
        bytes=p.stat().st_size if p.is_file() else None,
        sha256=sha(p) if p.is_file() else None,
        target_relative_path=f'inputs/input_{i:04d}/{p.name}')
    record['acquisition_status'] = 'URL_FROM_ARCHIVED_MANIFEST' if rows else 'NEEDS_SOURCE_RESOLUTION'
    if any(u['kind'].startswith('frozen_') for u in usage):
        record['acquisition_status'] = 'PROJECT_DERIVED_FROZEN_PRODUCT'
    records.append(record)
    if i % 100 == 0:
        print(f'Hashed {i}/{len(unique)} inputs', flush=True)
output('inputs/RAW_INPUT_ACQUISITION.json', records)
pd.DataFrame([{**r, 'acquisition_urls': json.dumps(r['acquisition_urls']),
               'url_evidence': json.dumps(r['url_evidence'])} for r in records]).to_csv(
                   DEST/'inputs/RAW_INPUT_ACQUISITION.csv', index=False, encoding='utf-8-sig')
output('provenance/URL_INDEX.json', urls)
for p in [R/'contracts/ANALYSIS_CONTRACT.json', R/'contracts/PROTECTED_INPUTS.json',
          R/'contracts/PLAN_FREEZE.json', R/'contracts/FINAL_SCORE_FREEZE.json',
          R/'completion/contracts/REAL_INPUT_MANIFEST.json', R/'completion/contracts/REAL_INPUT_MANIFEST_FREEZE.json',
          BASE/'contracts/NOISE_CALENDAR_AUDIT.json']:
    shutil.copy2(p, DEST/'provenance'/p.name)
report = {'files': len(records), 'file_bytes': sum(r['bytes'] or 0 for r in records),
    'missing': [r['original_path'] for r in records if not r['exists']],
    'url_unresolved': [r['original_path'] for r in records if r['acquisition_status'] == 'NEEDS_SOURCE_RESOLUTION'],
    'noise_unresolved': [r for r in noise_rows if r['status'] != 'FOUND'],
    'real_events': {run: len(events) for run, events in real['runs'].items()}}
output('reports/INPUT_INVENTORY_STATUS.json', report)
print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}), flush=True)
