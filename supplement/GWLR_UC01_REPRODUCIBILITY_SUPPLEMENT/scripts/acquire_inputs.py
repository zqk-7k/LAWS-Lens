"""Hash-checked public input acquisition; defaults to listing, never overwrites."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

def sha(p, algorithm='sha256'):
    with p.open('rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()

def download(url, target, expected, algorithm='sha256'):
    if target.exists():
        if sha(target, algorithm) != expected:
            raise RuntimeError('Existing file has wrong hash; refusing overwrite: '+str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix+'.part')
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={'User-Agent': 'GWLR-UC01-repro/1.0',
        **({'Range': f'bytes={offset}-'} if offset else {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        resume = offset > 0 and response.status == 206
        if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Unexpected Content-Range')
        with part.open('ab' if resume else 'wb') as stream:
            shutil.copyfileobj(response, stream, length=2**20)
    if sha(part, algorithm) != expected:
        raise RuntimeError('Downloaded checksum mismatch: '+url)
    part.rename(target)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-gib', type=float, default=200)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    if args.ids:
        names = set(args.ids)
        rows = [r for r in rows if r['input_id'] in names]
        if {r['input_id'] for r in rows} != names:
            raise RuntimeError('Unknown input ID')
    total = sum(r.get('download_bytes', r['bytes']) for r in rows)
    # This conservative sum includes shared archives more than once.
    for r in rows:
        print(r['input_id'], r['filename'], r['acquisition_status'])
    if not args.execute:
        print('LIST ONLY. Use --execute and --ids to acquire selected inputs.')
        return
    seen_archives = {}
    expected = 0
    for r in rows:
        key = r.get('download_sha256', r.get('download_checksum', r['sha256']))
        if key not in seen_archives:
            expected += r.get('download_bytes', r['bytes'])
            seen_archives[key] = True
    if expected > args.max_gib*2**30:
        raise RuntimeError('Download selection exceeds --max-gib')
    for r in rows:
        relative = Path(r['target_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe manifest target')
        target = args.destination/relative
        if target.exists():
            if sha(target) != r['sha256']:
                raise RuntimeError('Existing input changed: '+str(target))
            continue
        if r['acquisition_status'] == 'PROJECT_DERIVED_FROZEN_PRODUCT':
            raise RuntimeError('First extract '+r['bundled_archive']+' into destination; missing '+str(target))
        urls = r['acquisition_urls']
        if not urls:
            raise RuntimeError('No public acquisition route: '+r['input_id'])
        if 'archive_member' in r:
            digest = r.get('download_sha256', r.get('download_checksum'))
            algorithm = 'sha256' if 'download_sha256' in r else r['download_checksum_algorithm']
            archive = args.destination/'downloads'/(digest+('.zip' if r.get('archive_format')=='zip' else '.tar.gz'))
            download(urls[0], archive, digest, algorithm)
            target.parent.mkdir(parents=True, exist_ok=True)
            if r.get('archive_format') == 'zip':
                with zipfile.ZipFile(archive) as bundle, bundle.open(r['archive_member']) as stream, target.open('xb') as f:
                    shutil.copyfileobj(stream, f)
            else:
                with tarfile.open(archive) as bundle:
                    member = bundle.getmember(r['archive_member'])
                    if not member.isfile():
                        raise RuntimeError('Not an ordinary archive member')
                    with bundle.extractfile(member) as stream, target.open('xb') as f:
                        shutil.copyfileobj(stream, f)
            if sha(target) != r['sha256']:
                raise RuntimeError('Archive member hash mismatch: '+str(target))
        else:
            download(urls[0], target, r['sha256'])
        print('VERIFIED', r['input_id'], flush=True)

if __name__ == '__main__':
    main()
