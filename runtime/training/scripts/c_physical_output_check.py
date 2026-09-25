"""Read-only provenance checks for all C validation and test sky products."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def main(root):
    out = root/'completion'
    report = out/'contracts/PHYSICAL_OUTPUT_RECHECK_C.json'
    if report.exists():
        raise RuntimeError('Refuse to overwrite an existing physical output audit')
    if not (out/'contracts/INJECTION_EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Complete injection evaluation before this audit')
    records, isolation = [], []
    for run in ('O3', 'O4a', 'O4b'):
        frames = {}
        for split in ('validation', 'test'):
            path = out/'maps'/run/'C_PHYSICAL'/split/'events.parquet'
            frame = pd.read_parquet(path)
            frames[split] = frame
            assert len(frame) == 450 and frame.event_uid.nunique() == 450
            assert frame.source_uid.nunique() == 270 and frame.moc_path.nunique() == 450
            assert frame.groupby('source_uid').physical_strain_scale_factor.nunique().eq(1).all()
            assert frame.groupby('source_uid').common_distance_mpc.nunique().eq(1).all()
            assert not frame.separate_image_rescaling.any()
            assert frame.raw_shared_with_waveform.all() and frame.same_noisy_strain.all()
            assert not frame.truth_used_for_template_selection.any()
            assert frame.map_shared_across_model_seeds.all()
            assert frame.status.eq('PASS').all() and not frame.fallback_used.any()
            assert frame.ordering.eq('NESTED').all() and frame.coordinate_frame.eq('ICRS').all()
            scale_error = abs(frame.optimal_network_snr/frame.unscaled_optimal_network_snr
                              /frame.physical_strain_scale_factor-1)
            assert scale_error.max() < 1e-10
            normalization_error = abs(frame.sky_normalization-1)
            assert normalization_error.max() < 1e-5
            for row in frame.itertuples(index=False):
                assert digest(row.raw_strain_path) == row.raw_strain_sha256
                assert digest(row.moc_path) == row.moc_sha256
            checks = [item for values in frame.checks for item in values]
            maxima = {key: max(item[key] for item in checks) for key in (
                'matched_filter_relative', 'self_injection_relative', 'horizon_quadrature_relative')}
            assert max(maxima.values()) <= 1e-6
            records.append(dict(run=run, split=split, events=len(frame),
                source_parents=frame.source_uid.nunique(), noise_parents=frame.noise_parent_uid.nunique(),
                maximum_relative_amplitude_identity_error=float(scale_error.max()),
                maximum_map_normalization_error=float(normalization_error.max()),
                matched_filter_checks=maxima, all_raw_and_MOC_hashes_match=True,
                same_amplitude_and_distance_within_source=True, same_noisy_strain_provenance=True,
                template_selection_uses_truth=False, fallback_events=0, events_manifest_sha256=digest(path)))
        overlap = {key: sorted(set(frames['validation'][key]) & set(frames['test'][key]), key=str)
                   for key in ('event_uid', 'source_uid', 'global_source_id', 'noise_parent_uid')}
        assert not any(overlap.values()), (run, overlap)
        isolation.append(dict(run=run, validation_test_intersections=overlap))
    result = dict(state='PASS', post_freeze_read_only=True, script_sha256=digest(__file__),
        events=2700, raw_files_hash_checked=2700, native_maps_hash_checked=2700,
        scope='all validation/test products; does not add a new coverage or adoption gate',
        limitations=['provenance and numerical identities, not independent full BBH PE',
                     'does not establish nominal posterior coverage or astrophysical representativeness'],
        records=records, isolation=isolation)
    with report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
