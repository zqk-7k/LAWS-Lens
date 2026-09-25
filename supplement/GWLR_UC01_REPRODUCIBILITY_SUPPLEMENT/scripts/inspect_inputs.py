import json
from pathlib import Path
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
R = P/'results/gwlr_unified_c_physical_20260918T134500Z_r1'

for path in sorted((R/'plans').rglob('*')):
    if path.suffix not in ('.parquet', '.csv', '.json'):
        continue
    print('FILE', path)
    if path.suffix == '.json':
        print(path.read_text()[:2200])
    else:
        f = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        print('SHAPE', f.shape, 'COLUMNS', list(f.columns))
        print(f.head(1).to_json(orient='records'))
for path in sorted((R/'completion').glob('real_inputs/**/*')):
    if path.is_file() and path.suffix in ('.json', '.csv', '.parquet'):
        print('REAL', path, path.stat().st_size)
print('ENV', (P/'.venvs/v10p1_pe/pyvenv.cfg').read_text())
print('RECEIPT', next((R/'contracts/tasks').glob('*.json')).read_text())
print('COMPLETION DIRECTORIES', [x.name for x in (R/'completion').iterdir()])
