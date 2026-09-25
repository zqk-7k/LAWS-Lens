"""Create non-overlapping extension archives and a combined private release index."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def verify(path,expected):
    seen=set()
    with tarfile.open(path,'r|*') as archive:
        for member in archive:
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise RuntimeError('Invalid or duplicate archive member')
            if hashlib.file_digest(archive.extractfile(member),'sha256').hexdigest()!=expected[member.name]:
                raise RuntimeError('Archive content hash mismatch')
            seen.add(member.name)
    if seen!=set(expected):
        raise RuntimeError('Missing archive member')
    return len(seen)

def bundle(root,paths,role,name):
    rows=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)]
    data=json.dumps(rows,indent=2).encode()
    manifest='package_manifests/'+role+'.json'
    expected={r['path']:r['sha256'] for r in rows}
    expected[manifest]=hashlib.sha256(data).hexdigest()
    dest=root/'packages'/name
    with tarfile.open(dest,'x:gz',compresslevel=1) as archive:
        for path in sorted(paths):
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
        member=tarfile.TarInfo(manifest)
        member.size=len(data)
        member.mode=0o644
        archive.addfile(member,io.BytesIO(data))
    count=verify(dest,expected)
    return {'role':role,'file':name,'bytes':dest.stat().st_size,'sha256':sha(dest),
            'members':count,'hash_failures':[],'all_expected_present':True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--core',required=True,type=Path)
    a=p.parse_args()
    root=a.release
    target=root/'packages'
    target.mkdir(exist_ok=False)
    scan=json.loads((root/'extension/reports/EXTENSION_SECRET_SCAN.json').read_text())
    if scan['findings']:
        raise RuntimeError('Secret scan not cleared')
    index=json.loads((a.core/'packages/RELEASE_PACKAGE_INDEX_FINAL.json').read_text())
    packages=[]
    for row in index['packages']:
        src=a.core/'packages'/row['file']
        if sha(src)!=row['sha256']:
            raise RuntimeError('Core archive changed')
        dst=target/row['file']
        shutil.copy2(src,dst)
        if sha(dst)!=row['sha256']:
            raise RuntimeError('Core archive copy mismatch')
        packages.append({**row,'origin':'r1 immutable archive, unchanged bytes'})
        print('CORE_VERIFIED',row['file'],flush=True)
    paths=[p for p in (root/'extension').rglob('*') if p.is_file() and not p.is_symlink()
           and '__pycache__' not in p.parts and p.suffix!='.pyc']
    packages.append(bundle(root,paths,'reproduction_extension_r2','LAWS_Lens_v1_DRAFT_reproduction_extension_r2_20260924.tar.gz'))
    paths=[p for prefix in ('fixtures/test_real','fixtures/recovery') for p in (root/prefix).rglob('*')
           if p.is_file() and not p.is_symlink()]
    paths.append(root/'inventory/REPLAY_MODEL_FREEZE.json')
    packages.append(bundle(root,paths,'test_real_fixtures_r2','LAWS_Lens_v1_DRAFT_test_real_fixtures_r2_20260924.tar.gz'))
    combined={'status':'HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION','revision':'r2','public_upload':False,
              'main_scientific_version':'GWLR-UC-01/C_PHYSICAL','paper_commit':'afb86ff5c8f212aec579592f0d7439f3f909d931',
              'packages':packages,'total_bytes':sum(r['bytes'] for r in packages),
              'entrypoint':'extension/README_CN.md','validation_report':'extension/reports/RELEASE_COMPLETION_R2_CN.md',
              'delivered_archive_replay':'PENDING','scientific_payload_changes':False,
              'assembly_note':'Extract four r1 modular packages plus two r2 extensions together into a new empty directory. Maps and wheelhouse are separate artifacts.'}
    (target/'RELEASE_PACKAGE_INDEX_R2.json').write_text(json.dumps(combined,indent=2)+'\n')
    shutil.copy2(root/'extension/README_CN.md',target/'READ_FIRST_R2_CN.md')
    for row in packages:
        (target/(row['file']+'.sha256')).write_text(row['sha256']+'  '+row['file']+'\n')
    (target/'SHA256SUMS_R2.txt').write_text(''.join(row['sha256']+'  '+row['file']+'\n' for row in packages))
    print(json.dumps(combined),flush=True)

if __name__=='__main__':
    main()
