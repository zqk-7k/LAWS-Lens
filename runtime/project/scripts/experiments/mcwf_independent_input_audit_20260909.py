#!/usr/bin/env python3
"""Read-only checks on transport invariance and independent noise folds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root, args.output
    if out.exists():
        raise RuntimeError('Independent audit output required')
    out.mkdir(parents=True)
    references = {dep: np.load(root/f'data/{dep}/noise/reference.npy', mmap_mode='r')
                  for dep in ('gwtc3', 'gwtc4')}
    transport = json.loads((root/'contracts/LAZY_HDF_TRANSPORT_ADDENDUM.json').read_text())
    rows = []
    for item in transport['previous_reference_rows']:
        actual = hashlib.sha256(np.asarray(references[item['deployment']][item['bank_index']]).tobytes()).hexdigest()
        rows.append({**item, 'actual_sha256': actual, 'unchanged': actual == item['sha256']})
    pd.DataFrame(rows).to_csv(out/'TRANSPORT_REFERENCE_INVARIANCE.csv', index=False)
    prior = json.loads((root/'audit/PROFILE_PARALLELISM_RESUME.json').read_text())
    profiles = []
    for item in prior['unchanged_completed_files']:
        actual = digest(item['path'])
        profiles.append({**item, 'actual_sha256': actual, 'unchanged': actual == item['sha256']})
    pd.DataFrame(profiles).to_csv(out/'PROFILE_RESUME_INVARIANCE.csv', index=False)
    inventory = []
    for dep in references:
        folder = root/f'data/{dep}/noise'
        complete = json.loads((folder/'COMPLETE.json').read_text())
        noise = pd.read_csv(folder/'noise_manifest.csv')
        if len(noise) != 64 or sorted(noise.bank_index) != list(range(64)):
            raise RuntimeError('Incomplete independent noise inventory')
        if set(noise.parent_file_gps[noise.fold == 0]) & set(noise.parent_file_gps[noise.fold == 1]):
            raise RuntimeError('Parent-noise fold leakage')
        excluded = pd.read_csv(root/f'audit/{dep}_EXCLUDED_NOISE.csv')
        overlap = 0
        for row in noise.itertuples():
            overlap += int(((row.start_gps < excluded.end+16) &
                            (row.end_gps > excluded.start-16)).sum())
        gps = [float(v['GPS']) for v in json.loads((root/'contracts/GWOSC_ALL_EVENTS.json').read_text())['events'].values()
               if v.get('GPS') is not None]
        known_overlap = sum(any(row.start_gps < x+128 and row.end_gps > x-128 for x in gps)
                            for row in noise.itertuples())
        ref_ok = digest(folder/'reference.npy') == complete['reference_sha256']
        psd_ok = digest(folder/'psd.npy') == complete['psd_sha256']
        finite = bool(np.isfinite(references[dep]).all())
        record = {'deployment': dep, 'noise_blocks': len(noise), 'parent_file_pairs': noise.parent_file_gps.nunique(),
            'historical_guard_overlaps': overlap, 'known_event_window_overlaps': known_overlap,
            'parent_noise_fold_intersection': 0, 'reference_sha256_matches': ref_ok,
            'PSD_sha256_matches': psd_ok, 'finite_all': finite}
        record['PASS'] = overlap == 0 and known_overlap == 0 and ref_ok and psd_ok and finite
        inventory.append(record)
    pd.DataFrame(inventory).to_csv(out/'INDEPENDENT_NOISE_INVENTORY.csv', index=False)
    result = {'UTC': datetime.now(timezone.utc).isoformat(),
        'reference_rows_checked': len(rows), 'reference_failures': sum(not r['unchanged'] for r in rows),
        'profile_files_checked': len(profiles), 'profile_failures': sum(not r['unchanged'] for r in profiles),
        'noise_audit_pass': all(row['PASS'] for row in inventory),
        'scores_or_predictions_changed': False}
    (out/'AUDIT_RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if result['reference_failures'] or result['profile_failures'] or not result['noise_audit_pass']:
        raise RuntimeError('Independent input audit failed')


if __name__ == '__main__':
    main()
