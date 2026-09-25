"""Replay frozen test/real predictions without training or ranking changes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

RUNS=('O3','O4a','O4b')
SEEDS=(2026091721,2026091722,2026091723)
KINDS=('short','rnc','ordered','multirate','joint')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def prepare(root, core, baseline):
    import numpy as np
    import pandas as pd
    shutil.copytree(core/'runtime',root/'runtime',symlinks=False)
    paths=[]
    for run in RUNS:
        deployment=baseline/f'completion/deployments/{run}/C_PHYSICAL'
        for split in ('test','real'):
            inputs=baseline/f'completion/real_inputs/{run}' if split=='real' else deployment/'inference_inputs/test'
            features=deployment/f'inference_features/{split}'
            fixture=root/f'fixtures/test_real/{run}/{split}'
            fixture.mkdir(parents=True,exist_ok=False)
            for source,name in [(inputs/'events.parquet','events.parquet'),(inputs/'raw2s.npy','raw2s.npy'),
                                (inputs/'low16s.npy','low16s.npy'),(features/'short.npy','short.npy'),
                                (features/'long.npy','long.npy'),(features/'rnc.npy','rnc.npy')]:
                shutil.copy2(source,fixture/name)
                digest=sha(fixture/name)
                if digest!=sha(source):
                    raise ValueError('Copy mismatch')
                paths.append({'source':str(source),'fixture':str((fixture/name).relative_to(root)),'sha256':digest})
            expected={}
            for seed in SEEDS:
                for kind in KINDS:
                    source=deployment/f'predictions_o4b/seed_{seed}/{split}/{kind}.npz'
                    with np.load(source) as arrays:
                        for key in arrays.files:
                            expected[f'{seed}/{kind}/{key}']={'shape':list(arrays[key].shape),
                                'dtype':str(arrays[key].dtype),'array_sha256':hashlib.sha256(arrays[key].tobytes()).hexdigest()}
            (fixture/'EXPECTED_FULL_ARRAYS.json').write_text(json.dumps(expected,indent=2))
            print('PREPARED',run,split,len(pd.read_parquet(fixture/'events.parquet')),flush=True)
    (root/'inventory/TEST_REAL_INPUTS.json').write_text(json.dumps(paths,indent=2))
    finish_fixtures(root, core, baseline)

def finish_fixtures(root, core, baseline):
    optional=[]
    for run in RUNS:
        for split in ('test','real'):
            original=baseline/f'completion/real_inputs/{run}' if split=='real' else baseline/f'completion/deployments/{run}/C_PHYSICAL/inference_inputs/test'
            source=original/'short_raw2s.npy'
            if source.is_file():
                target=root/f'fixtures/test_real/{run}/{split}/short_raw2s.npy'
                if target.exists():
                    if sha(target)!=sha(source):
                        raise ValueError('Existing short-input fixture differs')
                else:
                    shutil.copy2(source,target)
                optional.append({'source':str(source),'fixture':str(target.relative_to(root)),'sha256':sha(target)})
    (root/'inventory/OPTIONAL_SHORT_INPUTS.json').write_text(json.dumps(optional,indent=2))
    shutil.copy2(core/'contracts/MODEL_ARTIFACTS.json',root/'inventory/REPLAY_MODEL_FREEZE.json')

def replay(root, tag):
    runtime=root/'runtime'
    sys.path[:0]=[str(runtime/'training/scripts/archived_adapters'),str(runtime/'project/scripts/experiments'),
                  str(runtime/'project'),str(runtime/'training/scripts')]
    import numpy as np
    import pandas as pd
    import torch
    import o4b_hl_nso_inference_20260912 as inf
    import o4b_hl_nso_multiscale_training_20260912 as models
    import mcwf_finelag_eventpsd_20260907 as psd
    import mcwf_finelag_encoder_20260907 as trainer
    import mcwf_mass_tf_20260905 as tf
    torch.set_num_threads(2)
    frozen=json.loads((root/'inventory/REPLAY_MODEL_FREEZE.json').read_text())
    def checked_model_guard(unused, split):
        if split not in ('test','real'):
            raise ValueError('Replay scope changed')
        for model in frozen:
            if sha(root/model['path'])!=model['sha256']:
                raise RuntimeError('Frozen model changed')
    inf.guard=checked_model_guard
    models.P=runtime/'project'
    low,mult,cond=models.modules()
    output=root/'verification'/tag
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    original_load=torch.load
    original_sha=inf.s.sha
    for run in RUNS:
        destination=output/run
        destination.mkdir()
        archive=runtime/f'training/arms/C_PHYSICAL/{run}'
        for folder in ('models','ordered_mass_predictor','rankncontrast_component_v2'):
            (destination/folder).symlink_to(archive/folder,target_is_directory=True)
        trainer.b.PREVIOUS=destination/'rankncontrast_component_v2'
        inf.rnc_adapter.setup=lambda unused:(destination/'rankncontrast_component_v2',psd,trainer,tf)
        inf.models.initialize=lambda unused:(low,mult,cond,destination/'operator')
        inf.short_module=lambda unused:inf.s.load(runtime/'project/scripts/real_search/37_unified_intrinsic_multitask_pilot.py','release_short')
        def relocate(path):
            marker='/models/MULTIRATE/'
            return destination/'models/MULTIRATE'/str(path).split(marker,1)[1] if marker in str(path) else Path(path)
        torch.load=lambda path,*args,**kwargs:original_load(relocate(path),*args,**kwargs)
        inf.s.sha=lambda path:sha(relocate(path))
        for split in ('test','real'):
            fixture=root/f'fixtures/test_real/{run}/{split}'
            events=pd.read_parquet(fixture/'events.parquet')
            expected=json.loads((fixture/'EXPECTED_FULL_ARRAYS.json').read_text())
            inf.features=lambda unused,seed,requested:(fixture,events,fixture,fixture/'rnc.npy')
            for seed in SEEDS:
                inf.infer(destination,seed,split)
                for kind in KINDS:
                    prediction=destination/f'predictions_o4b/seed_{seed}/{split}/{kind}.npz'
                    fields=[]
                    with np.load(prediction) as arrays:
                        expected_keys={k.split('/',2)[2] for k in expected if k.startswith(f'{seed}/{kind}/')}
                        if set(arrays.files)!=expected_keys:
                            raise RuntimeError('Prediction fields missing or added')
                        for key in arrays.files:
                            reference=expected[f'{seed}/{kind}/{key}']
                            digest=hashlib.sha256(arrays[key].tobytes()).hexdigest()
                            ok=digest==reference['array_sha256'] and list(arrays[key].shape)==reference['shape'] and str(arrays[key].dtype)==reference['dtype']
                            fields.append({'field':key,'exact':ok,'actual':digest,'expected':reference['array_sha256']})
                    records.append({'run':run,'split':split,'seed':seed,'component':kind,'events':len(events),'fields':fields,'passed':all(x['exact'] for x in fields)})
                print(run,split,seed,'PASS' if all(r['passed'] for r in records) else 'MISMATCH',flush=True)
                (output/'PROGRESS.json').write_text(json.dumps(records,indent=2))
    torch.load,inf.s.sha=original_load,original_sha
    report={'status':'PASS' if all(r['passed'] for r in records) else 'FAIL','components':len(records),
            'all_arrays_exact':all(r['passed'] for r in records),'records':records,
            'input_boundary':'Frozen 2s input and matching features; not fresh strain-to-feature generation.',
            'trained_or_retuned':False,'rank_tables_modified':False}
    (output/'REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    if report['status']!='PASS':
        raise RuntimeError('Prediction replay mismatch; original results preserved')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--core',type=Path)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--finish-fixtures',action='store_true')
    p.add_argument('--output-tag',default='test_real_prediction_replay')
    args=p.parse_args()
    if args.finish_fixtures:
        if not args.core or not args.baseline:
            p.error('--finish-fixtures requires --core and --baseline')
        finish_fixtures(args.release,args.core,args.baseline)
    elif args.prepare_only:
        if not args.core or not args.baseline:
            p.error('--prepare-only requires --core and --baseline')
        prepare(args.release,args.core,args.baseline)
    else:
        replay(args.release,args.output_tag)
