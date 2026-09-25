#!/usr/bin/env python3
"""Audit source identity of calibration and reused catalog populations."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_catalog_20260909 as app
n = app.n
DATA = P/'results/mcwf_nodup_independent_bulk_development_22b_20260909T114300Z'
FIELDS = ['m1_det', 'm2_det', 'a1', 'a2', 'tilt1', 'tilt2', 'phi12', 'phijl', 'theta_jn']


def fingerprints(frame):
    values = frame[FIELDS].to_numpy(dtype='<f8')
    if not np.isfinite(values).all():
        raise RuntimeError('Missing source physical parameters')
    return {hashlib.sha256(row.tobytes()).hexdigest() for row in values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if (root/'audit/POPULATION_IDENTITY_AUDIT.json').exists():
        raise RuntimeError('Population audit already exists')
    inputs, rows, intersections = [], [], []
    for dep in n.DEPS:
        bank = DATA/f'data/{dep}/source_plan.parquet'
        inputs.append({'path': str(bank), 'sha256': n.sha(bank)})
        development = pd.read_parquet(bank)
        populations = {f'development_fold{k}': development[development.fold.eq(k)] for k in (0, 1)}
        for catalog in n.CATALOGS:
            path = n.FRESH/f'confirmation/{dep}/catalog_{catalog}/source_systems.parquet'
            frame = pd.read_parquet(path)
            populations[f'reused_catalog_{catalog}'] = frame
            inputs.append({'path': str(path), 'sha256': n.sha(path)})
        for label, frame in populations.items():
            rows.append({'deployment': dep, 'population': label, 'rows': len(frame),
                'unique_source_uid': frame.source_uid.nunique(),
                'unique_intrinsic_fingerprint': len(fingerprints(frame)),
                'unique_lens_environments': frame.gwlmc_environment_id.dropna().nunique()})
            if frame.source_uid.nunique()!=len(frame) or len(fingerprints(frame))!=len(frame):
                raise RuntimeError('Repeated waveform source within population')
        for (left, a), (right, b) in itertools.combinations(populations.items(), 2):
            uid_overlap = len(set(a.source_uid)&set(b.source_uid))
            waveform_overlap = len(fingerprints(a)&fingerprints(b))
            lens_overlap = len(set(a.gwlmc_environment_id.dropna())&set(b.gwlmc_environment_id.dropna()))
            intersections.append({'deployment': dep, 'left': left, 'right': right,
                'shared_source_uid': uid_overlap, 'shared_intrinsic_fingerprint': waveform_overlap,
                'shared_lens_environment': lens_overlap})
    n.write_csv(root/'tables/POPULATION_IDENTITY_COUNTS.csv', rows)
    n.write_csv(root/'tables/POPULATION_INTERSECTIONS.csv', intersections)
    n.write_csv(root/'manifest/POPULATION_INPUT_SHA256.csv', inputs)
    failed = any(row['shared_source_uid'] or row['shared_intrinsic_fingerprint'] for row in intersections)
    n.write_json(root/'audit/POPULATION_IDENTITY_AUDIT.json', {'UTC': n.utc(),
        'waveform_source_identity_pass': not failed, 'fingerprint_fields': FIELDS,
        'lens_environment_overlap_reported_separately': True,
        'not_independent_lens_population_validation': True,
        'noise_population_independence_not_inferred_from_unique_source_ids': True,
        'bootstrap_interpretation': 'Conditional on fixed noise and existing catalogs; shared physical source fingerprints checked across catalogs, model draws remain shared.',
        'known_limitations': [
            'Coverage-oriented source/SNR proposal, not an astrophysical rate sample.',
            'Per-image target SNR scaling; not response-derived magnification-ratio validation.',
            'O3 noise with inherited cumulative O1-O3 response calendar.',
            'Frozen conditional BAYESTAR uses simulated matched-filter summaries; not complete PE of the exact non-Gaussian injection strain.',
            'All reused catalogs are adaptive-development data, not newly blinded confirmation.'],
        'status': n.STATUS})
    shutil.copy2(__file__, root/'scripts/shared_profile_population_audit.py')
    if failed:
        raise RuntimeError('Source identity intersection found; bootstrap interpretation requires review')
    print('POPULATION_IDENTITY_PASS', len(rows), len(intersections), flush=True)


if __name__ == '__main__':
    main()
