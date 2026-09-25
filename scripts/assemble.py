"""Create an independent, unpublished paper release workspace."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

P = Path('/root/autodl-tmp/gw-catalog')
UPLOAD = P/'release_staging/laws_lens_paper_v1_20260924/20260924T130225Z'
C = P/'results/gwlr_unified_c_physical_20260918T134500Z_r1'
REPRO = P/'results/gwlr_uc01_reproducibility_20260919T154000Z'
OUT = Path(sys.argv[1])

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def extract(bundle, dest):
    with tarfile.open(bundle) as tar:
        for member in tar:
            rel = Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe member: '+member.name)
            if member.isdir():
                continue
            target = dest/rel
            with tar.extractfile(member) as stream:
                data = stream.read()
            if target.exists():
                if sha(target) != hashlib.sha256(data).hexdigest():
                    raise ValueError('Conflicting members: '+str(rel))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(data)

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ('audit', 'scripts', 'reports', 'logs', 'verification', 'packages', 'contracts'):
        (OUT/name).mkdir()
    manifest = json.loads((UPLOAD/'extracted/FILE_MANIFEST.json').read_text())
    files = manifest['files']
    failures = [r['path'] for r in files if sha(UPLOAD/'extracted'/r['path']) != r['sha256']]
    save(OUT/'audit/UPLOAD_RECHECK.json', dict(files=len(files), failures=failures,
        transfer_record=json.loads((UPLOAD/'REMOTE_UPLOAD_VERIFICATION.json').read_text())))
    if failures:
        raise ValueError('Upload changed')
    shutil.copytree(UPLOAD/'extracted', OUT/'uploaded_snapshot')
    protected = [dict(path=str(UPLOAD/'extracted'/r['path']), sha256=r['sha256']) for r in files]
    bundles = [
        ('C_REVIEWED', C/'package/GWLR_UC_01_deliverables_reviewed.tar.gz',
         'bd5e31e619b0de31a7d605351fc06ff6be5cb828b59d5d0d667269d58e1865a5', 'runtime'),
        ('C_NATIVE_MAPS', C/'package/GWLR_UC_01_native_BAYESTAR_maps.tar.gz',
         '1287ef0c4a7850a2c3052a2c1c133cc89a96e65dbc16acdcf6f8e5d41ee90583', None),
        ('RECONSTRUCTION', REPRO/'package/GWLR_UC01_reconstruction_inputs_and_code.tar.gz', None, 'runtime'),
        ('ENVIRONMENT_AND_INPUTS', REPRO/'package/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT_20260919.tar.gz', None, 'supplement'),
        ('WHEELHOUSE', REPRO/'package/GWLR_UC01_verified_wheelhouse.tar', None, None),
        ('DOMAIN04', P/'packages/trilens_speed_domain04_20260917T155616Z_deliverables.tar.gz', None, 'speed'),
    ]
    inventory = []
    for role, p, expected, dest in bundles:
        if expected is None:
            expected = Path(str(p)+'.sha256').read_text().split()[0]
        actual = sha(p)
        if expected != actual:
            raise ValueError('Archive hash mismatch: '+str(p))
        protected.append(dict(path=str(p), sha256=actual))
        inventory.append(dict(role=role, original_path=str(p), bytes=p.stat().st_size,
                              sha256=actual, extracted_to=dest, publication_authorized=False))
        print(role, p.stat().st_size, actual, flush=True)
        if dest:
            extract(p, OUT/dest)
    save(OUT/'contracts/RELEASE_INPUTS.json', inventory)
    save(OUT/'audit/PROTECTED_INPUTS.json', protected)
    save(OUT/'contracts/RELEASE_STATE.json', dict(
        release_name='LAWS-Lens paper release v1.0.0 DRAFT',
        baseline='GWLR-UC-01/C_PHYSICAL', paper_commit=manifest['paper_commit'],
        publication_authorized=False, repository_created=False, zenodo_created=False,
        license='PENDING_AUTHOR_AND_THIRD_PARTY_REVIEW', status='INTERNAL_VALIDATION',
        missing_author_decisions=['repository/account', 'code/model/data licenses', 'ORCID', 'publication time'],
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('ASSEMBLY_COMPLETE', OUT, flush=True)

if __name__ == '__main__':
    main()
