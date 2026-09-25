"""Packaging-only repair: distinguish detection signatures from credentials."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile

import pandas as pd

import uab_completion as c
import uab_delivery as original

TOKENS = [b'-----BEGIN '+b'RSA PRIVATE KEY-----',
          b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----', b'sshpass '+b'-p ']


def scan(path):
    content = path.read_bytes()
    if not any(token in content for token in TOKENS):
        return None
    # The only exemption is the exact frozen detector-signature assignment.
    frozen = json.loads((c.ROOT/'contracts/FINAL_SCORE_FREEZE.json').read_text())['files']
    allowed = c.OUT/'scripts/uab_delivery.py'
    expected = {row['path']: row['sha256'] for row in frozen}
    if path != allowed or c.U.sha(path) != expected.get(str(allowed)):
        raise RuntimeError('Credential-like content: '+str(path))
    tree = ast.parse(content.decode('utf-8'))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'forbidden' for t in node.targets)]
    if len(matches) != 1 or ast.literal_eval(matches[0].value) != TOKENS:
        raise RuntimeError('Unrecognized detector-signature assignment')
    node = matches[0]
    lines = content.splitlines(keepends=True)
    remaining = b''.join(lines[:node.lineno-1]+lines[node.end_lineno:])
    if any(token in remaining for token in TOKENS):
        raise RuntimeError('Additional credential-like content beyond detection signatures')
    return dict(path=str(path), sha256=c.U.sha(path),
                exception='exact frozen byte-literal detection rule, not credential material',
                all_other_content_scanned=True)


def verify_archive(path, expected):
    count = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected:
                continue
            h = hashlib.sha256(); stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(2**20), b''):
                h.update(block)
            if h.hexdigest() != expected[member.name]:
                raise RuntimeError('Archive hash mismatch: '+member.name)
            count += 1
    if count != len(expected):
        raise RuntimeError('Archive file count mismatch')
    return count


def package():
    files = {}
    def add(path, name):
        if path.is_file(): files[name] = path
    for folder in ('contracts', 'scripts', 'tables', 'reports', 'figures'):
        for path in (c.OUT/folder).rglob('*'):
            if path.is_file() and not path.name.endswith('.lock') and '__pycache__' not in str(path):
                add(path, 'completion/'+str(path.relative_to(c.OUT)))
    add(c.OUT/'README_CN.md', 'README_CN.md')
    for folder in ('real_PE', 'real_sky'):
        for path in (c.OUT/folder).rglob('*'):
            add(path, 'completion/'+str(path.relative_to(c.OUT)))
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            root = c.deployment(run, arm)
            for folder in ('calibration', 'evaluation', 'sky_pair_scores', 'real_ranking'):
                for path in (root/folder).rglob('*'):
                    add(path, 'completion/'+str(path.relative_to(c.OUT)))
    for folder in ('contracts', 'manifests', 'reports', 'scripts', 'plans'):
        for path in (c.ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix not in ('.npy', '.pyc') and not path.name.endswith('.lock'):
                add(path, 'training/'+str(path.relative_to(c.ROOT)))
    for row in pd.read_csv(c.OUT/'contracts/TRAINED_MODEL_MANIFEST.csv').itertuples():
        path = Path(row.path); add(path, 'training/'+str(path.relative_to(c.ROOT)))
    for path in (c.OUT/'logs').glob('*.log'):
        add(path, 'completion/logs/'+path.name)
    exceptions = []
    for path in files.values():
        if path.suffix in ('.json', '.csv', '.md', '.py', '.log', '.txt'):
            result = scan(path)
            if result: exceptions.append(result)
    scan_record = c.OUT/'contracts/PACKAGE_SECURITY_SCAN.json'
    c.write(scan_record, dict(state='PASS', known_signature_exceptions=exceptions,
                            other_credential_markers=0, actual_key_material_included=False))
    add(scan_record, 'completion/contracts/'+scan_record.name)
    records = [dict(path=name, sha256=c.U.sha(path), bytes=path.stat().st_size) for name,path in sorted(files.items())]
    manifest = c.OUT/'manifests/DELIVERABLE_SHA256.csv'
    original.csv(manifest, pd.DataFrame(records)); add(manifest, 'manifest/'+manifest.name)
    target = c.ROOT/'package'; target.mkdir(exist_ok=True)
    archive = target/'GWLR_UAB_01_20260918_deliverables.tar.gz'
    maps = target/'GWLR_UAB_01_20260918_native_BAYESTAR_maps.tar.gz'
    if archive.exists() or maps.exists():
        raise RuntimeError('Never overwrite an existing archive')
    mapfiles = [x for x in (c.OUT/'maps').rglob('*') if x.is_file() and '.tmp.' not in x.name]
    import shutil
    upper = sum(x.stat().st_size for x in files.values())+sum(x.stat().st_size for x in mapfiles)
    if shutil.disk_usage(c.ROOT).free-upper < 25*2**30:
        raise RuntimeError('Insufficient space for conservative uncompressed archive bound')
    with tarfile.open(archive, 'w:gz', compresslevel=5, dereference=True) as tar:
        for name,path in sorted(files.items()): tar.add(path, arcname=name, recursive=False)
    verified = verify_archive(archive, {row['path']:row['sha256'] for row in records})
    sha = c.U.sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha+'  '+archive.name+'\n')
    map_records = [dict(path=str(x.relative_to(c.OUT)), sha256=c.U.sha(x), bytes=x.stat().st_size) for x in mapfiles]
    native_manifest = c.OUT/'manifests/NATIVE_MAP_SHA256.json'; c.write(native_manifest, map_records)
    with tarfile.open(maps, 'w:gz', compresslevel=2) as tar:
        for path in mapfiles: tar.add(path, arcname=str(path.relative_to(c.OUT)), recursive=False)
        tar.add(native_manifest, arcname='manifest/'+native_manifest.name)
    mapsha = c.U.sha(maps)
    maps.with_suffix(maps.suffix+'.sha256').write_text(mapsha+'  '+maps.name+'\n')
    map_verified = verify_archive(maps, {row['path']:row['sha256'] for row in map_records})
    return dict(deliverables=str(archive), deliverables_sha256=sha, internal_files_verified=verified,
        native_maps=str(maps), native_maps_sha256=mapsha, map_files_verified=map_verified)


def main(root, out):
    c.initialize(root, out)
    c.write(out/'contracts/PACKAGING_ONLY_REPAIR.json', dict(utc=c.U.now(),
        reason='security scanner matched its own byte-literal detection signatures',
        original_delivery_script_preserved=True, frozen_scientific_files_changed=False,
        scores_weights_ranks_or_temperature_changed=False, old_failure_log_preserved=True))
    c.write(out/'RUN_STATUS.json', dict(stage='FINAL_REPORT_PACKAGE', utc=c.U.now(), packaging_only_repair=True))
    original.validate(); original.report()
    archived = package()
    record = dict(state=original.FINAL, complete_results=True, utc=c.U.now(), results=str(out),
        report=str(out/'reports/GWLR_UAB_01_COMPLETE_METHOD_AND_RESULTS_CN.md'),
        read_first=str(out/'reports/READ_FIRST_FINAL_AUDIT_LIMITS_CN.md'),
        historical_results_overwritten=False, paper_changed=False, automatically_adopted=False,
        A_NEUTRAL_complete=True, B_CUE_complete=True, all_three_runs_complete=True, **archived)
    c.write(out/'contracts/FINAL_DELIVERY.json', record)
    c.write(out/'RUN_STATUS.json', record); c.write(root/'RUN_STATUS.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.root, a.out)
