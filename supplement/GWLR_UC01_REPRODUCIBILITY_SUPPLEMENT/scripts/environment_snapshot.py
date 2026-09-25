"""Snapshot the C dependency closure without changing its production environment."""
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(sys.argv[1])
OUT = ROOT / 'environment'
OUT.mkdir(parents=True, exist_ok=False)
roots = ['numpy', 'scipy', 'pandas', 'pyarrow', 'torch', 'scikit-learn',
         'matplotlib', 'h5py', 'healpy', 'bilby', 'lalsuite', 'pycbc',
         'ligo.skymap', 'astropy', 'gwpy', 'tqdm', 'psutil', 'threadpoolctl',
         'requests', 'packaging', 'setuptools', 'wheel', 'pip']
dists = {canonicalize_name(d.metadata['Name']): d for d in md.distributions()}
selected = {}
queue = [(name, '') for name in roots]
seen = set()
edges = []
while queue:
    raw, extra = queue.pop()
    name = canonicalize_name(raw)
    if (name, extra) in seen:
        continue
    seen.add((name, extra))
    d = dists[name]
    selected[name] = d.version
    for entry in d.requires or []:
        req = Requirement(entry)
        if req.marker and not req.marker.evaluate({'extra': extra}):
            continue
        dep = canonicalize_name(req.name)
        if dep not in dists:
            raise RuntimeError(f'Missing production dependency: {name}: {entry}')
        if dists[dep].version not in req.specifier:
            raise RuntimeError(f'Conflict in C closure: {name}: {entry}: {dists[dep].version}')
        edges.append({'parent': name, 'requirement': entry})
        queue.append((dep, ''))
        queue.extend((dep, x) for x in req.extras)
(OUT / 'requirements.production-closure.txt').write_text(''.join(
    f'{name}=={version}\n' for name, version in sorted(selected.items())))
(OUT / 'DEPENDENCY_GRAPH.json').write_text(json.dumps({'roots': roots,
    'selected': selected, 'edges': edges, 'excluded_unrelated_faiss': 'faiss-cpu' not in selected}, indent=2))
for filename, command in [('production-pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze', '--all']),
                          ('production-pip-check.txt', [sys.executable, '-m', 'pip', 'check']),
                          ('nvidia-smi.txt', ['nvidia-smi']), ('lscpu.txt', ['lscpu'])]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / filename).write_text(result.stdout)
(OUT / 'SYSTEM.json').write_text(json.dumps({'python': sys.version,
    'executable': sys.executable, 'platform': platform.platform(),
    'libc': platform.libc_ver(), 'system_site_packages_in_production': True,
    'verification_scope': 'clean rebuild, numerical smoke tests, frozen score replay; not full retraining'}, indent=2))
print(json.dumps({'output': str(ROOT), 'packages_in_closure': len(selected), 'roots': roots}))
