"""Independent brute-count checks and compact, self-verifying delivery archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd


RUNS = {'O3': 62, 'O4a': 74, 'O4b': 86}
MODELS = ['2026091721', '2026091722', '2026091723', 'mean_S']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def verify(root, failed_attempt=None):
    status = json.loads((root/'RUN_STATUS.json').read_text())
    if not status['conditional_tables_complete'] or status['annual_FAR_usable']:
        raise RuntimeError('Unexpected completion state')
    checks = []
    for run, n_real in RUNS.items():
        for model in MODELS:
            label = 'mean_S' if model == 'mean_S' else 'seed_'+model
            background = pd.read_parquet(root/f'background/{run}/{model}_calibration_pairs.parquet')
            scores = background.final_score_POSITIVE.to_numpy()
            if len(scores) != 4005:
                raise RuntimeError('Wrong background size')
            for domain, prefix, expected in [('injection', 'injection', 101025),
                                              ('real', 'real_GWTC', n_real*(n_real-1)//2)]:
                path = root/f'tables/{run}/{prefix}_{label}_FPP.parquet'
                f = pd.read_parquet(path)
                if len(f) != expected or f.pair_key.duplicated().any():
                    raise RuntimeError('Incorrect table scope: '+str(path))
                if domain == 'injection' and int(f.is_true_pair.sum()) != 180:
                    raise RuntimeError('Incorrect true-pair count')
                indices = np.arange(len(f)) if domain == 'real' else np.unique(
                    np.r_[np.linspace(0, len(f)-1, 257).astype(int), np.flatnonzero(f.is_true_pair)])
                queries = f.final_score_POSITIVE.to_numpy()[indices]
                # No reuse of the production searchsorted implementation.
                counted = np.array([np.count_nonzero(scores >= value) for value in queries])
                if not np.array_equal(counted, f.background_exceedances.to_numpy()[indices]):
                    raise RuntimeError('Independent brute-force exceedance count differs')
                if not np.allclose(f.background_exceedances/len(scores), f.conditional_FPP, rtol=0, atol=1e-15):
                    raise RuntimeError('FPP denominator mismatch')
                if not np.array_equal(f.zero_exceedance_unresolved, f.background_exceedances == 0):
                    raise RuntimeError('Zero exceedance flag mismatch')
                if not f.FAR_per_year.isna().all() or not f.validated_GWTC_FPP.isna().all():
                    raise RuntimeError('Unvalidated FAR or real FPP was populated')
                if domain == 'real':
                    if not np.array_equal(f.consensus_rank, np.arange(1, len(f)+1)):
                        raise RuntimeError('Real rank order changed')
                    draws = np.load(root/f'background/{run}/{model}_conditional_catalogs.npz')
                    maximum = draws['max_scores']
                    counted_catalogs = np.array([np.count_nonzero(maximum >= v) for v in queries])
                    if not np.allclose(counted_catalogs/len(maximum), f.conditional_catalog_FPP_mc):
                        raise RuntimeError('Catalog tail count mismatch')
                    if not np.allclose(f.conditional_expected_false_pairs, expected*f.conditional_FPP):
                        raise RuntimeError('Conditional catalog burden mismatch')
                checks.append(dict(run=run, model=model, domain=domain, rows=len(f),
                                   independently_counted_queries=len(indices), passed=True))
        for prefix in ['injection', 'real_GWTC']:
            frames = [pd.read_parquet(root/f'tables/{run}/{prefix}_seed_{m}_FPP.parquet')
                      for m in MODELS[:3]]
            mean = pd.read_parquet(root/f'tables/{run}/{prefix}_mean_S_FPP.parquet')
            aligned = [f.set_index('pair_key').loc[mean.pair_key] for f in frames]
            expected = np.mean([f.final_score_POSITIVE.to_numpy() for f in aligned], axis=0)
            if not np.allclose(expected, mean.final_score_POSITIVE, rtol=1e-12, atol=1e-12):
                raise RuntimeError('Mean-score identity failed')
    protected = pd.read_csv(root/'manifests/PROTECTED_INPUT_SHA256.csv')
    for row in protected.to_dict('records'):
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('Historical input changed')
    write(root/'manifests/INDEPENDENT_CHECK.json', dict(
        passed=True, checks=checks, protected_files_unchanged=len(protected),
        statistical_meaning='Numerical verification only, not validation of real-domain transfer'))
    if failed_attempt is not None:
        dest = root/'provenance/failed_attempt_r1'
        dest.mkdir(parents=True, exist_ok=False)
        paths = [failed_attempt/'logs/build.log', failed_attempt/'scripts/build_fpp_tables.py',
                 failed_attempt/'contracts/ANALYSIS_CONTRACT.json', *failed_attempt.glob('FAILURE_*.json')]
        for p in paths:
            shutil.copy2(p, dest/p.name)
        write(dest/'REPAIR_NOTE.json', dict(
            original_directory=str(failed_attempt), removed_or_overwritten=False,
            cause='Historical real COMPLETE.json seals all_seed_rankings, not raw per-seed files',
            fix='Verify each raw seed score/channel against hash-sealed all_seed_rankings; also verify consensus identity',
            statistical_rules_changed=False, failed_large_partial_tables_left_in_original_directory=True))
    (root/'README_CN.md').write_text(
        '# GWLR-FPP-01\n\n先读 reports/FINAL_FPP_AND_FAR_AUDIT_CN.md。\n\n'
        '本包为既有C背景条件FPP表，不是新生成的总体背景，也不是已验证的真实GWTC FPP。'
        '年度FAR=NO-GO，数值空缺不是0。模型、权重、排名和论文均未改动。\n\n'
        'tables/{run}/injection_mean_S_FPP.csv 与 real_GWTC_mean_S_FPP.csv 为两张主表；'
        '逐seed表、真实Top10/20/50、背景曲线、源噪声隔离、FAR审计均在包内。\n'
        '注入逐seed完整表为Parquet；逐seed真对表与汇总另有CSV。\n'
        'SHA256SUMS.json覆盖打包前的所有有效载荷文件，可离线核对。\n', encoding='utf-8')
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'manifests/SHA256SUMS.json', files)
    package = root.parent/(root.name+'_deliverables.tar.gz')
    if package.exists():
        raise RuntimeError('Refuse package overwrite')
    with tarfile.open(package, 'x:gz') as out:
        out.add(root, arcname=root.name)
    with tarfile.open(package, 'r:gz') as archive:
        members = archive.getmembers()
        for name, digest in files.items():
            stream = archive.extractfile(root.name+'/'+name)
            h = hashlib.sha256()
            for data in iter(lambda: stream.read(1024*1024), b''):
                h.update(data)
            if h.hexdigest() != digest:
                raise RuntimeError('Package member hash mismatch: '+name)
    digest = sha(package)
    package.with_suffix(package.suffix+'.sha256').write_text(digest+'  '+package.name+'\n')
    delivery = dict(package=str(package), sha256=digest, bytes=package.stat().st_size,
        package_members=len(members), internally_verified_files=len(files), internal_hash_failures=0,
        state=status['state'], report=str(root/'reports/FINAL_FPP_AND_FAR_AUDIT_CN.md'),
        no_new_population_background=True, not_validated_real_FPP=True, annual_FAR_usable=False)
    write(root.parent/(root.name+'_DELIVERY.json'), delivery)
    print(json.dumps(delivery, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--failed-attempt', type=Path)
    a = p.parse_args()
    verify(a.root, a.failed_attempt)
