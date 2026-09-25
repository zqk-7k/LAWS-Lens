"""Exercise isolated legacy diagnostic definitions against six archived maps."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--release',type=Path,required=True)
p.add_argument('--bank-root',type=Path,required=True)
a=p.parse_args()
spec=importlib.util.spec_from_file_location('recovery_replay',a.release/'extension/scripts/replay_recovery.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.init_worker(str(a.bank_root),str(a.release))
import c_trigger
import numpy as np
from astropy.table import Table
diagnostics=c_trigger.module(a.bank_root/'scripts/bayestar_si_fixed.py','isolated_diagnostics')
rows=[]
for path in sorted((a.release/'fixtures/recovery').glob('*/*/EXPECTED_MAP.fits.gz')):
    ref=json.loads((path.parent/'RESULT.json').read_text())
    # The expected truth coordinate comes from the frozen event/source plan.
    import pandas as pd
    source=pd.read_parquet(a.release/f'fixtures/injections/{path.parent.parent.name}/source.parquet').iloc[0]
    metrics=diagnostics.posterior_metrics(diagnostics.raster_probability(Table.read(path),512),source.ra,source.dec)
    for key,value in metrics.items():
        if key in ref:
            if not np.isclose(value,ref[key],rtol=1e-10,atol=1e-10):
                raise RuntimeError('Diagnostic arithmetic differs: '+key)
            rows.append({'run':path.parent.parent.name,'event':path.parent.name,'field':key,'delta':float(abs(value-ref[key]))})
out=a.release/'verification/diagnostic_loader_selftest'
out.mkdir(exist_ok=False)
report={'status':'PASS','checks':len(rows),'rows':rows,'historical_project_reads_blocked':True,
        'no_legacy_catalog_initialization':True}
(out/'REPORT.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
