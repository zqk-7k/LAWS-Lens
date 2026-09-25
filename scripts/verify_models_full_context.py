"""Replay all five frozen components, three runs and seeds, on fixed fixtures."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--output-tag', default='all_components_full_context')
    a = ap.parse_args()
    R = a.release
    runtime = R/'runtime'
    sys.path[:0] = [str(runtime/'training/scripts/archived_adapters'),
                   str(runtime/'project/scripts/experiments'), str(runtime/'project'),
                   str(runtime/'training/scripts')]
    import numpy as np
    import pandas as pd
    import torch
    import o4b_hl_nso_inference_20260912 as inf
    import o4b_hl_nso_multiscale_training_20260912 as models
    import mcwf_finelag_eventpsd_20260907 as psd
    import mcwf_finelag_encoder_20260907 as trainer
    import mcwf_mass_tf_20260905 as tf
    torch.set_num_threads(2)
    models.P = runtime/'project'
    low, mult, cond = models.modules()
    fixtures = R/'fixtures/inference_full_context'
    output = R/'verification'/a.output_tag
    output.mkdir(parents=True, exist_ok=False)
    records, inputs = [], []
    load = torch.load
    for run in ('O3', 'O4a', 'O4b'):
        root = output/run
        root.mkdir()
        fixture = fixtures/run
        if a.baseline:
            old = a.baseline/f'completion/deployments/{run}/C_PHYSICAL'
            e = pd.read_parquet(old/'inference_inputs/validation/events.parquet')
            selected = np.r_[np.flatnonzero(e.family.ne('unlensed'))[:2],
                             np.flatnonzero(e.family.eq('unlensed'))[:2]]
            fixture.mkdir(parents=True, exist_ok=False)
            e.to_parquet(fixture/'events.parquet', index=False)
            (fixture/'selected_rows.json').write_text(json.dumps(selected.tolist()))
            for src, dest in [('inference_inputs/validation/raw2s.npy', 'raw2s.npy'),
                              ('inference_inputs/validation/low16s.npy', 'low16s.npy'),
                              ('inference_features/validation/short.npy', 'short.npy'),
                              ('inference_features/validation/long.npy', 'long.npy'),
                              ('inference_features/validation/rnc.npy', 'rnc.npy')]:
                np.save(fixture/dest, np.load(old/src, mmap_mode='r'))
                inputs.append(dict(original=str(old/src), fixture=str((fixture/dest).relative_to(R)),
                                   selected_rows=selected.tolist(), sha256=sha(fixture/dest)))
            for seed in (2026091721, 2026091722, 2026091723):
                dest = fixture/f'expected/{seed}'
                dest.mkdir(parents=True)
                for kind in ('short', 'rnc', 'ordered', 'multirate', 'joint'):
                    with np.load(old/f'predictions_o4b/seed_{seed}/validation/{kind}.npz') as z:
                        np.savez_compressed(dest/f'{kind}.npz', **{
                            k: v[selected] if v.ndim and v.shape[0] == len(e) else v for k, v in z.items()})
        events = pd.read_parquet(fixture/'events.parquet')
        selected = np.array(json.loads((fixture/'selected_rows.json').read_text()), int)
        archive = runtime/f'training/arms/C_PHYSICAL/{run}'
        for folder in ('models', 'ordered_mass_predictor', 'rankncontrast_component_v2'):
            (root/folder).symlink_to(archive/folder, target_is_directory=True)
        rout = root/'rankncontrast_component_v2'
        trainer.b.PREVIOUS = rout
        inf.rnc_adapter.setup = lambda unused: (rout, psd, trainer, tf)
        inf.models.initialize = lambda unused: (low, mult, cond, root/'operator')
        inf.features = lambda unused, seed, split: (fixture, events, fixture, fixture/'rnc.npy')
        inf.short_module = lambda unused: inf.s.load(runtime/'project/scripts/real_search/37_unified_intrinsic_multitask_pilot.py', 'release_short')
        # Relocate embedded mass-checkpoint references only; bytes and hash remain unchanged.
        def relocate(path, *args, **kwargs):
            path = Path(path)
            marker = '/models/MULTIRATE/'
            if marker in str(path):
                path = root/'models/MULTIRATE'/str(path).split(marker, 1)[1]
            return load(path, *args, **kwargs)
        torch.load = relocate
        # The archived infer function hashes the referenced file before torch.load.
        original_sha = inf.s.sha
        def relocated_sha(path):
            marker = '/models/MULTIRATE/'
            if marker in str(path):
                path = root/'models/MULTIRATE'/str(path).split(marker, 1)[1]
            return sha(path)
        inf.s.sha = relocated_sha
        for seed in (2026091721, 2026091722, 2026091723):
            inf.infer(root, seed, 'validation')
            for kind in ('short', 'rnc', 'ordered', 'multirate', 'joint'):
                got = root/f'predictions_o4b/seed_{seed}/validation/{kind}.npz'
                expected = fixture/f'expected/{seed}/{kind}.npz'
                errors = {}
                with np.load(got) as g, np.load(expected) as ref:
                    for key in ref.files:
                        actual = g[key][selected] if g[key].ndim and g[key].shape[0] == len(events) else g[key]
                        delta = float(np.max(np.abs(actual.astype(float)-ref[key])))
                        errors[key] = delta
                        np.testing.assert_allclose(actual, ref[key], rtol=3e-5, atol=3e-6)
                    full_path = fixture/'EXPECTED_FULL_ARRAYS.json'
                    full_exact = None
                    if full_path.exists():
                        full = json.loads(full_path.read_text())
                        full_exact = all(hashlib.sha256(g[key].tobytes()).hexdigest() ==
                            full[f'{seed}/{kind}/{key}']['array_sha256'] for key in g.files)
                        if not full_exact:
                            raise RuntimeError('Full validation prediction hashes differ; investigate hardware/batching before accepting replay.')
                records.append(dict(run=run, seed=seed, component=kind, forward_events=len(events), compared_events=len(selected),
                                    status='PASS', maximum_absolute_errors=errors, full_validation_arrays_exact=full_exact))
            print(run, seed, 'PASS', flush=True)
        torch.load, inf.s.sha = load, original_sha
    modules = sorted(set(str(Path(m.__file__).resolve()) for m in sys.modules.values()
                         if getattr(m, '__file__', None) and '/gw-catalog/' in str(m.__file__)))
    external = [p for p in modules if not Path(p).is_relative_to(R)]
    (output/'REPORT.json').write_text(json.dumps(dict(status='PASS', comparisons=records,
        components=45, compared_events_per_run=4, validation_forward_events_per_run=450, all_catalog_inference=False,
        input_level='frozen 2s arrays and template-matching features; not raw-strain complete inference',
        rtol=3e-5, atol=3e-6, external_project_modules=external,
        fixture_inputs=inputs), indent=2))

if __name__ == '__main__':
    main()
