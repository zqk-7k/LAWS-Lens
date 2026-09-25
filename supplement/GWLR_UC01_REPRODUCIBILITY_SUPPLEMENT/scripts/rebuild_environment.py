"""Build exact production dependency wheels and verify an isolated installation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
out = root/'environment'
env = Path('/root/autodl-tmp/gwlr_uc01_clean_env_20260919')
python = str(env/'bin/python')
wheels = Path('/root/autodl-tmp/gwlr_uc01_wheelhouse_20260919')
wheels.mkdir(exist_ok=True)
requirements = (out/'requirements.production-closure.txt').read_text().splitlines()
standard = [r for r in requirements if not r.startswith('torch==')]
(out/'requirements.non-torch.txt').write_text('\n'.join(standard)+'\n')
commands = []
def run(command, name):
    commands.append(command)
    print(name, flush=True)
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'TMPDIR': str(wheels)})
    (out/'BUILD_COMMANDS.json').write_text(json.dumps(commands, indent=2))
    if result.returncode:
        raise RuntimeError(f'{name} failed; see log')

run([python, '-m', 'pip', 'wheel', '--no-deps', '--index-url',
     'https://pypi.tuna.tsinghua.edu.cn/simple', '--wheel-dir', str(wheels),
     '-r', str(out/'requirements.non-torch.txt')], '01-build-wheels')
torch = next(r for r in requirements if r.startswith('torch=='))
run([python, '-m', 'pip', 'download', '--no-deps', '--index-url',
     'https://download.pytorch.org/whl/cu128', '--dest', str(wheels), torch], '02-torch-wheel')
# Hash the exact wheel artifacts; locally built wheels must travel with this lock.
from packaging.utils import parse_wheel_filename, canonicalize_name
entries = {}
artifacts = []
for f in sorted(wheels.glob('*.whl')):
    name, version, _, _ = parse_wheel_filename(f.name)
    digest = hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()
    entries[canonicalize_name(name)] = f'{name}=={version} --hash=sha256:{digest}'
    artifacts.append({'filename': f.name, 'bytes': f.stat().st_size, 'sha256': digest})
expected = {r.split('==')[0] for r in requirements}
if set(entries) != expected:
    raise RuntimeError(f'Wheel closure mismatch: {set(entries)^expected}')
lock = out/'requirements.linux-x86_64-cp312-cu128.lock'
lock.write_text('# Exact offline wheel lock for Linux x86_64, CPython 3.12.3, CUDA 12.8.\n'+
    '# Install with --no-index --find-links WHEELHOUSE --require-hashes.\n'+
    '\n'.join(entries[k] for k in sorted(entries))+'\n')
(out/'WHEEL_ARTIFACTS.json').write_text(json.dumps(artifacts, indent=2))
run([python, '-m', 'pip', 'install', '--no-index', '--find-links', str(wheels),
     '--require-hashes', '-r', str(lock)], '03-offline-install')
run([python, '-m', 'pip', 'check'], '04-clean-pip-check')
(out/'CLEAN_BUILD_STATUS.json').write_text(json.dumps({'state': 'PASS',
    'python': python, 'system_site_packages': False, 'packages': len(entries),
    'offline_install_from_hashed_wheels': True, 'scientific_validation': 'PENDING'}, indent=2))
print('CLEAN_BUILD_PASS', flush=True)
