"""Preflight archive paths and extract without overwriting."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

root = Path(__file__).resolve().parent
archive = next((root/'archives').glob('*.tar.gz'))
dest = (root/'extracted').resolve()
assert not dest.exists(), 'Do not overwrite previous extraction'
with tarfile.open(archive, 'r:gz') as tf:
    members = tf.getmembers()
    seen = set()
    for m in members:
        p = PurePosixPath(m.name)
        assert not p.is_absolute() and '..' not in p.parts
        assert '\\' not in m.name and ':' not in m.name
        assert m.isfile() or m.isdir(), 'Links/special files not allowed'
        target = dest.joinpath(*p.parts).resolve()
        assert target.is_relative_to(dest)
        key = str(target).casefold()
        assert key not in seen, 'Duplicate or case collision'
        seen.add(key)
    total = sum(m.size for m in members)
    assert total < 2 * 1024**3
    tf.extractall(dest, filter='data')
report = {'members': len(members), 'files': sum(m.isfile() for m in members), 'uncompressed_bytes': total, 'safe_path_preflight': True}
with (root/'audit/EXTRACTION.json').open('x', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report))
for f in dest.rglob('*'):
    if f.is_file(): print(f.relative_to(dest))
