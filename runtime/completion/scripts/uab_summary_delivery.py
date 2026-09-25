"""Build a small reader bundle without duplicating the full scientific archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(out):
    delivery_path = out/'contracts/FINAL_DELIVERY.json'
    delivery = json.loads(delivery_path.read_text())
    if not delivery.get('complete_results'):
        raise RuntimeError('Full delivery must complete first')
    files = {}
    for folder in ('reports', 'figures'):
        for path in (out/folder).rglob('*'):
            if path.is_file():
                files[str(path.relative_to(out))] = path
    table_names = (
        'retrieval_summary_190.csv', 'retrieval_summary_450.csv',
        'retrieval_metrics_per_seed.csv', 'paired_arm_difference_intervals.csv',
        'cluster_intervals.csv', 'selected_weights.csv', 'SNR_only_diagnostic.csv',
        'real_PE_official_budget_summary.csv', 'real_top50_both_arms_all_runs.csv',
        'sky_resolution_summary.csv', 'sky_resolution_retrieval.csv',
        'independent_metric_recheck.csv', 'real_resolution_decision_audit.csv',
        'real_top50_resolution_sensitivity.csv', 'population_calendar_noise_audit.csv',
        'Mc_prediction_embedding_diagnostics.csv', 'sky_truth_coverage_and_runtime.csv',
        'HISTORICAL_HASH_AUDIT.csv',
    )
    for name in table_names:
        path = out/'tables'/name
        if not path.is_file():
            raise FileNotFoundError(path)
        files['tables/'+name] = path
    for name in ('FINAL_DELIVERY.json', 'COMPLETENESS_AUDIT.json',
                 'PACKAGE_SECURITY_SCAN.json', 'PACKAGING_ONLY_REPAIR.json',
                 'INDEPENDENT_METRIC_RECHECK.json', 'REAL_DECISION_RESOLUTION_AUDIT.json'):
        path = out/'contracts'/name
        if not path.is_file():
            raise FileNotFoundError(path)
        files['contracts/'+name] = path
    for key in ('deliverables', 'native_maps'):
        path = Path(delivery[key]+'.sha256')
        files['archive_checksums/'+path.name] = path
    size = sum(path.stat().st_size for path in files.values())
    if size > 30*2**20:
        raise RuntimeError('Reader bundle exceeds local disk budget')
    records = {name: sha(path) for name, path in sorted(files.items())}
    manifest = out/'manifests/READER_BUNDLE_SHA256.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(records, indent=2)+'\n')
    files['manifest/READER_BUNDLE_SHA256.json'] = manifest
    target = out.parent/'package/GWLR_UAB_01_20260918_reader_bundle.tar.gz'
    if target.exists():
        raise FileExistsError(target)
    with tarfile.open(target, 'w:gz', compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname='GWLR_UAB_01/'+name, recursive=False)
    count = 0
    with tarfile.open(target, 'r:gz') as archive:
        for member in archive:
            key = member.name.removeprefix('GWLR_UAB_01/')
            if key in records:
                if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != records[key]:
                    raise RuntimeError('Bundle verification failed: '+key)
                count += 1
    if count != len(records):
        raise RuntimeError('Missing bundled file')
    digest = sha(target)
    target.with_suffix(target.suffix+'.sha256').write_text(digest+'  '+target.name+'\n')
    print(json.dumps(dict(archive=str(target), sha256=digest, verified_files=count,
                         bytes=target.stat().st_size, uncompressed_bytes=size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
