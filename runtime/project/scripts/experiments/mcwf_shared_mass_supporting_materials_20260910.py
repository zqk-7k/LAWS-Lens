#!/usr/bin/env python3
"""Archive existing implementation dependencies and R72 provenance, not data caches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main(root):
    target = root / 'scripts/dependencies'
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    inputs = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in inputs.itertuples():
        file = Path(r.path)
        if file.suffix != '.py' or not file.is_relative_to(P):
            continue
        if sha(file) != r.sha256:
            raise RuntimeError('Protected code changed')
        dest = target / file.relative_to(P)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
        if sha(dest) != r.sha256:
            raise RuntimeError('Dependency copy mismatch')
        rows.append({'source': str(file), 'relative_destination': str(dest.relative_to(root)), 'sha256': r.sha256})
    prior = P / 'results/mcwf_shared_mass_population_72_20260910T015630Z'
    for folder in ('contracts', 'tables', 'audit', 'reports', 'scripts'):
        for file in (prior / folder).rglob('*'):
            if not file.is_file() or file.suffix in ('.npy', '.npz', '.pyc') or file.is_symlink():
                continue
            dest = root / 'provenance/R72' / file.relative_to(prior)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            digest = sha(file)
            if sha(dest) != digest:
                raise RuntimeError('Provenance copy mismatch')
            rows.append({'source': str(file), 'relative_destination': str(dest.relative_to(root)), 'sha256': digest})
    shutil.copy2(__file__, root / 'scripts/shared_mass_supporting_materials.py')
    pd.DataFrame(rows).to_csv(root / 'manifest/SUPPORTING_MATERIALS_SHA256.csv', index=False, encoding='utf-8-sig')
    with (root / 'contracts/SUPPORTING_MATERIALS_COMPLETE.json').open('x', encoding='utf-8') as stream:
        json.dump({'UTC': datetime.now(timezone.utc).isoformat(), 'files': len(rows), 'data_or_scoring_changed': False,
            'credentials_and_raw_strain_excluded': True, 'reproduction': 'Dependencies retain project-relative paths. Large raw/cache/checkpoint inputs remain referenced by immutable manifests, not redistributed.'}, stream, indent=2)
    print('SUPPORTING_MATERIALS_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    main(p.parse_args().root)
