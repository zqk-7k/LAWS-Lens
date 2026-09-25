"""Compare freshly generated features with archived event arrays, read only."""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'results/trilens_scaling_20260917_10to1891_r2/scripts'))
import trilens_scale_20260917 as run


def main(root):
    run.ROOT = root
    chosen, rows, recipes = run.init()
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    run.js(root/'contracts/AUDIT_SCOPE.json', dict(not_timing_acceptance=True,
        change_models=False, change_calibration=False, purpose='Find exact feature replay discrepancy'))
    F, B = run.F, run.B
    raw, refs = B.read_raw(rows)
    psd = [F.dev.BASE.v7.v3.estimate_psd(item) for item in refs]
    frequency, psds = psd[0][0], np.stack([item[1] for item in psd])
    full = np.stack([F.dev.BASE.v7.v3.preprocess_24s(w,frequency,psds[k]).astype(np.float32) for k,w in enumerate(raw)])
    lowviews = np.stack([F.low.low_view(w,frequency,psds[k],F.dev.BASE.v7.v3) for k,w in enumerate(raw)])
    coarse, fine, low, short = B.all_features(F,full,lowviews,frequency,psds)
    slots = rows.native_slot.to_numpy(int)
    expected = dict(
        coarse=np.load(F.e.TRAINED/'cache/deployment_event_psd/gwtc3/real_features.npy')[slots],
        fine=np.load(F.t.PREVIOUS/'fine_mass_context/features/gwtc3/real.npy')[slots],
        low=np.load(F.LOW/'features/gwtc3/real.npy')[slots])
    results=[]
    for name, actual in [('coarse',coarse),('fine',fine),('low',low)]:
        np.save(root/'results'/f'{name}.npy', actual)
        error=abs(actual-expected[name]).reshape(len(rows),-1)
        for k,row in rows.iterrows():
            results.append(dict(event=row.event_name,feature=name,
                max_abs=float(error[k].max()),nonzero=int(np.count_nonzero(error[k])),
                dtype=str(actual.dtype),expected_dtype=str(expected[name].dtype)))
    pd.DataFrame(results).to_csv(root/'tables/FEATURE_REPLAY.csv',index=False)
    run.finish(False,'READ_ONLY_FEATURE_DIAGNOSTIC')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
