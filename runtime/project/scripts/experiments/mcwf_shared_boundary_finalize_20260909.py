#!/usr/bin/env python3
"""Finalize R55 export: preserve draft, repair LaTeX escapes, include AP CI."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main(root,source):
    original=root/'reports/R55_BOUNDARY_CORRECTION_COMPLETE_CN.md'
    text=original.read_bytes().decode('utf-8')
    count=text.count('\r')
    fixed=text.replace('\r',r'\r')
    if any(ord(c)<32 and c not in '\n\t' for c in fixed):
        raise RuntimeError('Unexpected report control character')
    target=root/'reports/R55_FINAL_COMPLETE_REPORT_CN.md'
    fixed=fixed.replace('`tables/PRIMARY_BOOTSTRAP_DELTA_CI.csv`','`tables/PRIMARY_BOOTSTRAP_DELTA_CI_WITH_AUPRC.csv`')
    fixed+='\n## 导出校验附注\n\n本文件为排版修正版：原报告的 LaTeX `\\rm` 转义错误已修复，原报告保留，不改任何数值。完整的主要差值CI另存 `tables/PRIMARY_BOOTSTRAP_DELTA_CI_WITH_AUPRC.csv`，包含 average_precision 字段。\n'
    with target.open('x',encoding='utf-8') as f: f.write(fixed)
    ci=pd.read_csv(root/'uncertainty_summary/tables/PAIRED_METHOD_DELTA_CI.csv')
    ci=ci[(ci.method=='SHARED-PROFILE-MONOTONE-BOUNDARY')&(ci.baseline=='NODUP-DIRECT-REPLAY')&
          (ci['mode']=='fusion')&ci.quantity.isin(['R1','R10','average_precision','F50','F90'])]
    if set(ci.quantity)!=set(['R1','R10','average_precision','F50','F90']):
        raise RuntimeError('Primary confidence interval export incomplete')
    with (root/'tables/PRIMARY_BOOTSTRAP_DELTA_CI_WITH_AUPRC.csv').open('x',encoding='utf-8-sig') as f:
        ci.to_csv(f,index=False)
    copied=[]
    dependencies=source/'scripts/runtime_dependencies'
    for path in sorted(dependencies.rglob('*.py')):
        target=root/'scripts/runtime_dependencies'/path.relative_to(dependencies)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): raise RuntimeError('Dependency target already exists')
        shutil.copy2(path,target)
        if sha(path)!=sha(target): raise RuntimeError('Dependency copy mismatch')
        copied.append({'source':str(path),'target':str(target),'sha256':sha(path)})
    shutil.copy2(__file__,root/'scripts/shared_boundary_finalize.py')
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'goal_achieved':False,
        'status':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'authoritative_report':'reports/R55_FINAL_COMPLETE_REPORT_CN.md',
        'original_report_preserved':True,'latex_control_characters_repaired':count,
        'AUPRC_interval_exported':True,'interval_rows':len(ci),'copied_dependencies':copied,
        'numeric_scores_changed':False}
    with (root/'contracts/FINAL_EXPORT_COMPLETE.json').open('x') as f: json.dump(receipt,f,indent=2)
    print('FINAL_EXPORT_COMPLETE',len(ci),len(copied),count)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    a=p.parse_args()
    main(a.root,a.source_root)
