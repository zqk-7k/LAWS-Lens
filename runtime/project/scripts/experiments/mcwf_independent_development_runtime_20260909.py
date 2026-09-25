#!/usr/bin/env python3
"""Resume unchanged noise selection with larger read cache and diagnostics."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_profile_development_20260909 as g

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    g.ROOT = args.root
    _, data, v3 = g.expanded.modules()
    original_init, original_get = v3.HdfCache.__init__, v3.HdfCache.get
    counter = [0]
    def init(self, source_run, max_files=8):
        return original_init(self, source_run, max_files=128)
    def get(self, path):
        counter[0] += 1
        tick = time.perf_counter()
        result = original_get(self, path)
        if counter[0] % 50 == 0:
            print('NOISE_READ_DIAGNOSTIC', counter[0], path, round(time.perf_counter()-tick, 4), flush=True)
        return result
    v3.HdfCache.__init__, v3.HdfCache.get = init, get
    selector = data.select_valid_noise_references
    def select(*arguments, **keywords):
        chosen, counts = selector(*arguments, **keywords)
        print('NOISE_SELECTION_COUNTS', counts, flush=True)
        return chosen, counts
    data.select_valid_noise_references = select
    record = g.ROOT/'contracts/READ_CACHE_RUNTIME_ADDENDUM.json'
    if not record.exists():
        g.n.write_json(record, {'UTC': g.n.utc(), 'change': 'Read cache8to128 andread/selectiondiagnosticloggingonly.',
            'reason': 'First31noiseblockswritten;nextselection longrunning. Preservecheckpointandallinputs.',
            'unchanged': ['RNGpernoiseindex', 'selectiondistribution', 'finitecriterion', 'PSDmethod', 'sourceplan', 'globalexclusions', 'workers'],
            'rejections_column_note': 'Original generator storeslen(rejected_dict)=6 categories,notnumberofrejectedtrials. Do notinterpretthatcolumnasanattemptcount. Newstdoutrecordsfulldict.',
            'code_sha256': g.n.sha(Path(__file__))})
        shutil.copy2(__file__, g.ROOT/'scripts/independent_development_runtime.py')
    for dep in g.n.DEPS:
        g.generate(dep, args.workers)
