#!/usr/bin/env python3
"""Inventory all shared-method trials, including failures and uncompleted stages."""
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
import mcwf_development_20260905 as dev

stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=dev.PROJECT/f'results/mcwf_goal_ledger_{stamp}'
out.mkdir(exist_ok=False)
rows=[]
metrics=[]
budgets=[]
for root in sorted((dev.PROJECT/'results').glob('mcwf_unified_*')):
    contract=root/'contracts/UNIFIED_METHOD.json'
    if not contract.exists():
        continue
    conf=json.loads(contract.read_text())
    val=root/'contracts/VALIDATION_GATE.json'
    gate=root/'contracts/DEVELOPMENT_GATE.json'
    v=json.loads(val.read_text()) if val.exists() else {}
    g=json.loads(gate.read_text()) if gate.exists() else {}
    status=('DEVELOPMENT_PASS_NOT_FINAL' if g.get('development_pass') else 'DEVELOPMENT_FAIL') if g else (
        'VALIDATION_FAIL' if v and not v.get('pass') else 'NO_COMPLETED_DEVELOPMENT_GATE')
    rows.append({'root':str(root),'code':conf.get('code'),'status':status,'validation_pass':v.get('pass'),
        'development_pass':g.get('development_pass'),'feature_implementation':conf.get('waveform_feature_implementation',conf.get('feature_implementation','legacy_coarse_phasebank')),
        'contract_sha256':dev.sha(contract),'real_is_reused_development':True,'not_lensing_confirmation':True})
    for name,dest in [('RETRIEVAL_PER_SEED.csv',metrics),('PE_OFFICIAL_ALL.csv',budgets)]:
        path=root/'tables'/name
        if path.exists():
            f=pd.read_csv(path)
            f.insert(0,'trial_root',str(root))
            f.insert(1,'trial_code',conf.get('code'))
            dest.append(f)
dev.csv_write(out/'ALL_TRIAL_STATUS.csv',pd.DataFrame(rows))
if metrics:
    dev.csv_write(out/'ALL_REUSED_RETRIEVAL_PER_SEED.csv',pd.concat(metrics,ignore_index=True))
if budgets:
    dev.csv_write(out/'ALL_PE_OFFICIAL_BUDGETS.csv',pd.concat(budgets,ignore_index=True))
dev.json_write(out/'READ_ME.json',{'utc':stamp,'purpose':'internal exhaustive audit, not final delivery or adopted result',
    'trials':len(rows),'policy':'failed and incomplete trials retained; original v9.3/C-fixed/ET/paper not modified',
    'missing_results':'validation failure means no candidate test or real ranking was authorized in that trial; never silently substitute baseline metrics',
    'real_PE':'repeated development feedback, not blind validation; public candidate overlap is not a true-lensing label'})
print(out,flush=True)
