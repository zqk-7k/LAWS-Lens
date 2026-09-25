"""Build unpublished modular bundles and verify every regular member."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def verify(bundle, expected):
    seen=set();fail=[]
    with tarfile.open(bundle,'r|*') as tar:
        for m in tar:
            if not m.isfile():
                raise ValueError('Unexpected non-file member')
            if m.name not in expected:
                raise ValueError('Unexpected file: '+m.name)
            if m.name in seen:
                raise ValueError('Duplicate archive member')
            seen.add(m.name)
            with tar.extractfile(m) as f:
                h=hashlib.file_digest(f,'sha256').hexdigest()
            if h!=expected[m.name]:
                fail.append(m.name)
    if seen!=set(expected) or fail:
        raise ValueError('Member verification failed')
    return dict(members=len(seen),hash_failures=fail,all_expected_present=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True)
    ap.add_argument('--resume', action='store_true')
    a=ap.parse_args();R=a.release;out=R/'packages';out.mkdir(exist_ok=True)
    quarantine=set(json.loads((R/'contracts/PUBLICATION_QUARANTINE.json').read_text())['paths'])
    groups=[('code_models_scores',['runtime']),('paper_fpp_speed',['uploaded_snapshot','speed']),
            ('validation_fixtures',['fixtures']),
            ('release_audit',['audit','contracts','reports','scripts','supplement','logs','README.md'])]
    inventory=[]
    for label,folders in groups:
        bundle=out/f'LAWS_Lens_v1_DRAFT_{label}_20260924.tar.gz'
        man_name=f'package_manifests/{label}.json'
        if bundle.exists() and a.resume:
            with tarfile.open(bundle) as tar:
                content=tar.extractfile(man_name).read()
            manifest=json.loads(content)
            expected={r['path']:r['sha256'] for r in manifest}
            expected[man_name]=hashlib.sha256(content).hexdigest()
            check=verify(bundle,expected)
            row=dict(role=label,file=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),**check)
            inventory.append(row);print(json.dumps(row),flush=True)
            continue
        paths=[]
        for name in folders:
            source=R/name
            for p in ([source] if source.is_file() else source.rglob('*')):
                if not p.is_file() or p.is_symlink():
                    continue
                rel=p.relative_to(R)
                if any(s in rel.parts for s in ('.git','.secrets','__pycache__')) or p.suffix=='.pyc' or str(rel) in quarantine:
                    continue
                paths.append(p)
        manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
        content=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
        man_name=f'package_manifests/{label}.json'
        expected={r['path']:r['sha256'] for r in manifest}
        expected[man_name]=hashlib.sha256(content).hexdigest()
        bundle=out/f'LAWS_Lens_v1_DRAFT_{label}_20260924.tar.gz'
        with tarfile.open(bundle,'x:gz',compresslevel=1) as tar:
            for p in sorted(set(paths)):
                tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
            info=tarfile.TarInfo(man_name);info.size=len(content);info.mode=0o644
            tar.addfile(info,io.BytesIO(content))
        check=verify(bundle,expected)
        row=dict(role=label,file=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),**check)
        inventory.append(row);print(json.dumps(row),flush=True)
    inputs=json.loads((R/'contracts/RELEASE_INPUTS.json').read_text())
    for role in ('C_NATIVE_MAPS','WHEELHOUSE'):
        row=next(x for x in inputs if x['role']==role)
        source=Path(row['original_path']);target=out/source.name
        if target.exists() and not a.resume:
            raise ValueError('Existing artifact, refusing overwrite')
        if not target.exists():
            shutil.copy2(source,target)
        if sha(target)!=row['sha256']:
            raise ValueError('Copied archive mismatch')
        if role=='C_NATIVE_MAPS':
            expected=json.loads(source.with_name('GWLR_UC_01_native_BAYESTAR_maps.tar.manifest.json').read_text())
        else:
            artifacts=json.loads((R/'supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/environment/WHEEL_ARTIFACTS.json').read_text())
            expected={'wheelhouse/'+r['filename']:r['sha256'] for r in artifacts}
            env=R/'supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/environment'
            expected['WHEEL_ARTIFACTS.json']=sha(env/'WHEEL_ARTIFACTS.json')
            expected['requirements.lock']=sha(env/'requirements.linux-x86_64-cp312-cu128.lock')
        check=verify(target,expected)
        item=dict(role=role,file=target.name,bytes=target.stat().st_size,sha256=row['sha256'],**check)
        inventory.append(item);print(json.dumps(item),flush=True)
    save(out/'RELEASE_PACKAGE_INDEX.json',dict(status='INTERNAL_DRAFT_NOT_PUBLIC',packages=inventory,
        total_bytes=sum(x['bytes'] for x in inventory),public_upload=False,
        missing_items='../reports/MISSING_ITEMS.json'))
    (out/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['file']+'\n' for r in inventory))
    for r in inventory:
        (out/(r['file']+'.sha256')).write_text(r['sha256']+'  '+r['file']+'\n')
    print('PACKAGES_VERIFIED',sum(r['members'] for r in inventory),flush=True)

if __name__=='__main__':
    main()
