#!/usr/bin/env python3
"""Snapshot dependencies and failure provenance without touching old results."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(root, first, reference, bank, data_sky):
    project = Path('/root/autodl-tmp/gw-catalog')
    records = json.loads((root/'manifest/PROTECTED_BEFORE.json').read_text())
    copied = []
    for row in records:
        p = Path(row['path'])
        if not p.is_file() or p.suffix != '.py':
            continue
        if digest(p) != row['sha256']:
            raise RuntimeError(f'Protected source changed: {p}')
        relative = p.relative_to(project)
        target = root/'dependency_sources'/relative
        if target.exists():
            if digest(target) != row['sha256']:
                raise RuntimeError('Duplicate source snapshot conflict')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied.append({'original': str(p), 'snapshot': str(target.relative_to(root)), 'sha256': row['sha256']})
    write(root/'contracts/DEPENDENCY_SOURCE_SNAPSHOT.json', copied)
    packages = sorted({(d.metadata['Name'], d.version) for d in importlib.metadata.distributions()})
    write(root/'contracts/ENVIRONMENT_VERSIONS.json', {'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'packages': [{'name': n, 'version': v} for n, v in packages],
        'not_a_container_image_or_wheel_hash_lock': True,
        'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'memory_quota': Path('/sys/fs/cgroup/memory.max').read_text().strip()})
    versions = {'codename': 'ET-PHYS-SKY-PILOT-01', 'production_release': False,
        'historical_HL_unchanged': True, 'roots': [
            {'path': str(first), 'role': 'preserved failed precision attempt, not a science result'},
            {'path': str(reference), 'role': 'corrected raw strain and dense q32 oracle reference; q64 deliberately interrupted'},
            {'path': str(root), 'role': 'bandlimited-SNR numerical control and consolidated audited delivery'},
            {'path': str(bank), 'role': 'data-driven template recovery, no injected intrinsic truth in selection'},
            {'path': str(data_sky), 'role': 'actual noisy-strain BAYESTAR maps using recovered templates'}]}
    write(root/'contracts/VERSION_RELATIONS.json', versions)
    failure_folder = root/'preserved_failures'
    failure_folder.mkdir(exist_ok=False)
    for label, source in [('precision_attempt', first), ('dense_reference', reference)]:
        destination = failure_folder/label
        destination.mkdir()
        for part in ('contracts', 'tables'):
            if (source/part).exists():
                shutil.copytree(source/part, destination/part)
        if (source/'STATUS.json').exists():
            shutil.copy2(source/'STATUS.json', destination/'STATUS_AS_RECEIVED.json')
    for filename in ('et3_physical_trigger_repair_20260915T041500Z.log',
                     'et3_physical_trigger_repair_20260915T041500Z_r2.log',
                     'et3_data_template_pilot_20260915T034329Z.log'):
        p = project/'logs'/filename
        if p.is_file():
            shutil.copy2(p, failure_folder/filename)
    write(failure_folder/'INTERPRETATION.json', {
        'initial_precision_attempt': 'float32/complex64 strain FFT incompatible with PyCBC complex128 template; all inputs explicitly promoted in independent r2',
        'dense_reference': 'all q32 completed; q64 stopped only after dense-vs-bandlimited comparison inputs existed; partial q64 is not counted as successful',
        'status_as_received_is_historical_not_current_process_status': True,
        'old_files_modified': False})
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    print(json.dumps({'source_snapshots': len(copied), 'packages': len(packages), 'root': str(root)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('root', 'first', 'reference', 'bank', 'data-sky'):
        parser.add_argument('--'+flag, type=Path, required=True)
    a = parser.parse_args()
    main(a.root, a.first, a.reference, a.bank, a.data_sky)
