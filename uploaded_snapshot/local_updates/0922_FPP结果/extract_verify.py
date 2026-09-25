from pathlib import Path,PurePosixPath
import tarfile,hashlib,json
base=Path(__file__).resolve().parent
archive=next((base/'archives').glob('*.tar.gz'))
dest=base/'extracted'
assert not dest.exists(), 'Do not overwrite extraction'
with tarfile.open(archive) as t:
    members=t.getmembers(); seen=set(); roots=set()
    for m in members:
        p=PurePosixPath(m.name)
        assert not p.is_absolute() and '..' not in p.parts and ':' not in m.name and '\\' not in m.name
        assert m.isfile() or m.isdir()
        assert all(s.rstrip(' .')==s for s in p.parts)
        assert m.name.casefold() not in seen
        seen.add(m.name.casefold());roots.add(p.parts[0])
    assert len(roots)==1
    dest.mkdir()
    t.extractall(dest,filter='data')
root=dest/next(iter(roots))
manifest=json.loads((root/'manifests/SHA256SUMS.json').read_text())
print(type(manifest).__name__,str(manifest)[:500])
rows=manifest if isinstance(manifest,list) else manifest.get('files',manifest)
if isinstance(rows,dict):rows=[{'path':k,'sha256':v} for k,v in rows.items()]
for row in rows:
    p=root/row.get('path',row.get('relative_path',''))
    assert p.resolve().is_relative_to(root.resolve())
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    assert h==row['sha256'],str(p)
result={'archive':archive.name,'verified_manifest_files':len(rows),'archive_files':sum(m.isfile() for m in members),'hash_failures':0,'root':str(root)}
(base/'audit/EXTRACTION_VERIFIED.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
