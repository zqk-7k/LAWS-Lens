"""Check archived noise-bank samples against original public strain files."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd

root=Path(sys.argv[1])
R=Path('/root/autodl-tmp/gw-catalog/results/gwlr_unified_c_physical_20260918T134500Z_r1')
frame=pd.read_csv(root/'inputs/NOISE_BLOCKS_AND_RAW_FILES.csv')
rows=[]
for run in ('O3','O4a','O4b'):
    bank=np.load(R/f'plans/{run}/noise_reference_bank.npy',mmap_mode='r')
    for row in frame[frame.run==run].to_dict('records'):
        with h5py.File(row['path']) as f:
            data=f['strain/Strain']
            start=float(f['meta/GPSstart'][()])
            fs=round(1/float(data.attrs['Xspacing']))
            index=int(round((row['noise_start_gps']-start)*fs))
            if index<0 or index+bank.shape[-1]>len(data):
                rows.append(dict(run=run, parent=row['noise_parent_uid'], detector=row['detector'],
                    passed=False, reason='slice outside chosen raw file'))
                continue
            raw=np.asarray(data[index:index+bank.shape[-1]],dtype=bank.dtype)
            ref=np.asarray(bank[int(row['noise_bank_index']),('H1','L1').index(row['detector'])])
            rows.append(dict(run=run,parent=row['noise_parent_uid'],detector=row['detector'],
                passed=bool(np.array_equal(raw,ref)),maximum_absolute_error=float(np.max(abs(raw-ref))),
                samples=len(raw),sample_rate=fs))
pd.DataFrame(rows).to_csv(root/'reports/NOISE_BANK_RAW_RECONSTRUCTION.csv',index=False)
(root/'reports/NOISE_BANK_RAW_RECONSTRUCTION.json').write_text(json.dumps(dict(
    checks=len(rows), passed=sum(r['passed'] for r in rows), failed=[r for r in rows if not r['passed']]),indent=2))
print(json.dumps(dict(checks=len(rows),passed=sum(r['passed'] for r in rows),
                     failed=sum(not r['passed'] for r in rows))),flush=True)
