#!/usr/bin/env python3
"""Small review archive, alongside untouched full result archives."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs=2, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    chosen = []
    allowed = {'contracts', 'configs', 'tables', 'reports', 'figures',
               'scripts', 'audit', 'manifest', 'logs'}
    for root in args.roots:
        source_manifest = {
            row['path']: row['sha256'] for row in csv.DictReader(
                io.StringIO((root/'manifest/OUTPUT_SHA256.csv').read_text(encoding='utf-8-sig')))
        }
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(root)
            if relative.parts[0] not in allowed and str(relative) != 'README_CN.md':
                continue
            digest = sha(path)
            if str(relative) in source_manifest:
                assert digest == source_manifest[str(relative)], path
            chosen.append((path, f'{root.name}/{relative}', digest))
    contract = {
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'type': 'review_only_compact_subset',
        'full_results_remain_at': [str(r) for r in args.roots],
        'excluded': 'Per-configuration full pair parquet/CSV result directories; available in both full archives',
        'included': 'All configurations, summaries, Top100 combined tables, PE/official budgets, reports, figures, audits, logs and scripts',
        'files': [{'name': name, 'sha256': digest} for _, name, digest in chosen],
    }
    payload = json.dumps(contract, indent=2, ensure_ascii=False).encode('utf-8')
    with tarfile.open(args.output, 'w:gz') as archive:
        header = tarfile.TarInfo('REVIEW_BUNDLE_CONTRACT.json')
        header.size = len(payload)
        archive.addfile(header, io.BytesIO(payload))
        for path, name, _ in chosen:
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(args.output, 'r:gz') as archive:
        for _, name, expected in chosen:
            h = hashlib.sha256()
            with archive.extractfile(name) as handle:
                for block in iter(lambda: handle.read(8 << 20), b''):
                    h.update(block)
            assert h.hexdigest() == expected, name
    digest = sha(args.output)
    args.output.with_suffix('.gz.sha256').write_text(f'{digest}  {args.output.name}\n')
    result = {'archive': str(args.output), 'bytes': args.output.stat().st_size,
              'sha256': digest, 'files_verified': len(chosen), 'all_pass': True}
    args.output.with_suffix('.gz.verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
