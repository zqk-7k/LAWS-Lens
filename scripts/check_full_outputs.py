"""Compare all validation predictions to the immutable server baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for run in ('O3','O4a','O4b'):
        expected = {}
        for seed in (2026091721,2026091722,2026091723):
            for kind in ('short','rnc','ordered','multirate','joint'):
                actual = a.release/f'verification/portable_models/{run}/predictions_o4b/seed_{seed}/validation/{kind}.npz'
                old = a.baseline/f'completion/deployments/{run}/C_PHYSICAL/predictions_o4b/seed_{seed}/validation/{kind}.npz'
                with np.load(actual) as g, np.load(old) as ref:
                    for key in ref.files:
                        maximum = float(np.max(abs(g[key].astype(float)-ref[key])))
                        np.testing.assert_allclose(g[key],ref[key],rtol=3e-5,atol=3e-6)
                        h = hashlib.sha256(ref[key].tobytes()).hexdigest()
                        same = hashlib.sha256(g[key].tobytes()).hexdigest()==h
                        expected[f'{seed}/{kind}/{key}'] = dict(shape=list(ref[key].shape),dtype=str(ref[key].dtype),array_sha256=h)
                        rows.append(dict(run=run,seed=seed,component=kind,array=key,max_abs=maximum,exact=same,status='PASS'))
        (a.release/f'fixtures/inference_full_context/{run}/EXPECTED_FULL_ARRAYS.json').write_text(json.dumps(expected,indent=2))
    (a.release/'verification/FULL_VALIDATION_PREDICTION_CHECK.json').write_text(json.dumps(dict(status='PASS',
        validation_events_per_run=450,model_implementations=9,components=45,arrays=rows,
        all_exact=all(x['exact'] for x in rows),all_test_and_real_events=False),indent=2))
    print('FULL_VALIDATION_OUTPUTS_PASS',len(rows),flush=True)

if __name__=='__main__':
    main()
