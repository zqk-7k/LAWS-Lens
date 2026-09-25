"""Validate mirrored torch against the official SHA and finish offline install."""
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit
import requests
from packaging.utils import canonicalize_name, parse_wheel_filename

root = Path(sys.argv[1])
out = root/'environment'
wheels = Path('/root/autodl-tmp/gwlr_uc01_wheelhouse_20260919')
name = 'torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'
source = Path('/root/autodl-tmp/gwlr_uc01_torch_nju')/name
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag=='a':
            self.links.append(dict(attrs).get('href',''))
response = requests.get('https://download.pytorch.org/whl/cu128/torch/', timeout=60)
response.raise_for_status()
parser = Links(); parser.feed(response.text)
link = next(x for x in parser.links if unquote(urlsplit(x).path).endswith('/'+name))
expected = urlsplit(link).fragment.removeprefix('sha256=')
assert len(expected)==64
deadline=time.monotonic()+600
while not source.exists() or source.with_suffix(source.suffix+'.aria2').exists():
    if time.monotonic()>deadline:
        raise RuntimeError('Torch mirror transfer did not finish within bounded wait')
    time.sleep(2)
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
actual=sha(source)
assert actual==expected, (actual, expected)
(out/'TORCH_OFFICIAL_HASH_CHECK.json').write_text(json.dumps(dict(state='PASS',
    official_index=response.url, official_link=link,
    transport_url='https://mirrors.nju.edu.cn/pytorch/whl/cu128/'+name,
    expected_sha256=expected, actual_sha256=actual), indent=2))
shutil.copy2(source, wheels/name)
requirements=(out/'requirements.production-closure.txt').read_text().splitlines()
selected={line.split('==')[0]:line.split('==')[1] for line in requirements}
entries={}; artifacts=[]
for f in sorted(wheels.glob('*.whl')):
    package, version, _, _=parse_wheel_filename(f.name)
    package=canonicalize_name(package)
    assert selected[package]==str(version)
    digest=sha(f)
    entries[package]=f'{package}=={version} --hash=sha256:{digest}'
    artifacts.append(dict(filename=f.name, bytes=f.stat().st_size, sha256=digest))
assert set(entries)==set(selected)
lock=out/'requirements.linux-x86_64-cp312-cu128.lock'
lock.write_text('# Linux x86_64 / Python 3.12.3 / CUDA 12.8. Use the accompanying wheelhouse.\n'+
    '\n'.join(entries[k] for k in sorted(entries))+'\n')
(out/'WHEEL_ARTIFACTS.json').write_text(json.dumps(artifacts, indent=2)+'\n')
python='/root/autodl-tmp/gwlr_uc01_clean_env_20260919/bin/python'
commands=[[python,'-m','pip','install','--no-index','--find-links',str(wheels),
           '--require-hashes','-r',str(lock)],[python,'-m','pip','check']]
for cmd,name in zip(commands, ('03-offline-install','04-clean-pip-check')):
    print(name, flush=True)
    with (out/(name+'.log')).open('w') as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
(out/'OFFLINE_INSTALL_COMMANDS.json').write_text(json.dumps(commands,indent=2))
(out/'CLEAN_BUILD_STATUS.json').write_text(json.dumps(dict(state='PASS',
    python=python, system_site_packages=False, packages=len(entries),
    offline_install_from_hashed_wheels=True, scientific_validation='PENDING'),indent=2)+'\n')
print('CLEAN_BUILD_PASS',flush=True)
