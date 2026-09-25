#!/usr/bin/env python3
"""R58: explicitly bounded refinement of a boundary-selected regularizer."""
import argparse
import json
from pathlib import Path
import shutil
import sys

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_expanded_joint_calibration_20260909 as joint
n=joint.n
GRID=(1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.1,.3,1.)


def freeze(root,previous,expansion):
    if root.exists(): raise RuntimeError('Independent R58 directory required')
    gate=json.loads((previous/'contracts/PILOT_GATE.json').read_text())
    specs=json.loads((previous/'configs/EXPANDED_JOINT_CLASSIFIERS.json').read_text())
    if gate['gate']!='FAIL' or not all(c['shared_classifier']['ridge']==.001 for c in specs.values()):
        raise RuntimeError('Expected documented failure and all winners at lower grid boundary')
    for name in ('contracts','configs','tables','audit','scripts','manifest','reports','logs'):
        (root/name).mkdir(parents=True)
    contract=json.loads((previous/'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract.update(UTC=n.utc(),id='MCWF-SHARED-REGULARIZATION-PATH-58',
        previous_failure=str(previous),expanded_measurements=str(expansion),ridge_grid=GRID,
        motivation='All6R57 validation-selected ridges equal the minimum .001, with lower full-population loss at smaller tested ridge. Extend the numeric regularization path, not PE-selected score offsets.',
        changed='Only the ridge candidate grid. Same data/features/support/caps/fitloss/selection/diagnostic gates.',
        wrapper_in_memory_override='Only joint.classifier.base.RIDGES=GRID; original source files unmodified.',
        original_failures_retained=True,real_or_catalog_selection=False,
        stopping_boundary='This finite grid only. No automatic further extension or relaxing loss tolerance.')
    joint.write_once(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    paths=[previous/'contracts/PILOT_GATE.json',previous/'configs/EXPANDED_JOINT_CLASSIFIERS.json',
        previous/'tables/RIDGE_GRID.csv',Path(joint.__file__),Path(__file__)]
    import pandas as pd
    paths.extend(Path(row.path) for row in pd.read_csv(previous/'manifest/INPUT_SHA256.csv').itertuples())
    paths.extend(Path(row.path) for row in pd.read_csv(previous/'manifest/SEALED_EXPANDED_INPUTS.csv').itertuples())
    n.write_csv(root/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p)} for p in sorted(set(paths))])
    shutil.copy2(__file__,root/'scripts/shared_regularization_path.py')
    shutil.copy2(joint.__file__,root/'scripts/shared_expanded_joint_calibration.py')
    joint.write_once(root/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'runtime_sha256':n.sha(Path(joint.__file__)),'wrapper_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    print('REGULARIZATION_PATH_FROZEN',GRID,flush=True)


def fit(root,expansion):
    frozen=json.loads((root/'contracts/START_FREEZE.json').read_text())
    if frozen['wrapper_sha256']!=n.sha(Path(__file__)) or frozen['contract_sha256']!=n.sha(root/'contracts/ANALYSIS_CONTRACT.json'):
        raise RuntimeError('Frozen regularization wrapper or contract changed')
    joint.classifier.base.RIDGES=GRID
    joint.fit(root,expansion)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--expansion-root',type=Path,required=True)
    p.add_argument('--stage',choices=('freeze','fit'),required=True)
    a=p.parse_args()
    if a.stage=='freeze': freeze(a.root,a.previous_root,a.expansion_root)
    else: fit(a.root,a.expansion_root)
