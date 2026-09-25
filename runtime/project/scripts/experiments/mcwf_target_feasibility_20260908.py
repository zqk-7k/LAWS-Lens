#!/usr/bin/env python3
"""Anonymous category-count feasibility, never a ranking or training input."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import Bounds,LinearConstraint,milp

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_20260908 as t
dev=t.dev
EXTERNAL=P/'results/mcwf_noise_context_exploratory_20260908T014829Z/audit'
BASE={'gwtc3':{'mc':[6,12],'median':[.766621,.730264],'official':[5,9],'hanabi':[5,7]},
      'gwtc4':{'mc':[10,19],'median':[.809014,.848349],'official':[7,15],'hanabi':[5,11]}}


def run(root):
    if root.exists():raise RuntimeError('Independent directory required')
    for name in ('contracts','tables','reports','manifest','scripts'):(root/name).mkdir(parents=True)
    rows=[];inventories=[];hashes=[]
    for dep,b in BASE.items():
        path=EXTERNAL/f'{dep}_external_reference.parquet';f=pd.read_parquet(path)
        hashes.append({'path':str(path),'sha256':dev.sha(path)})
        if f.pair_key.duplicated().any():raise RuntimeError('Duplicate pair')
        bc=f.pe_mc_bhattacharyya_coefficient.to_numpy(float)
        d=f.pe_mc_standardized_distance.to_numpy(float)
        dm=f.pe_dmax_intrinsic.to_numpy(float)
        off=f['official_po_or_ml_fpp_below_0p01' if dep=='gwtc3' else 'official_po_or_phazap_fpp_below_0p01'].fillna(False).to_numpy(bool)
        han=f.official_any_pair_resolved_hanabi_overlap.fillna(False).to_numpy(bool)
        eligible=np.isfinite(bc)&np.isfinite(d)&np.isfinite(dm)&(dm<=3)&(bc>=.1)&(d<=5)
        inventories.append({'deployment':dep,'pairs':len(f),'eligible_PE_no_catastrophe':int(eligible.sum()),
          'official1pct':int(off.sum()),'hanabi_table':int(han.sum()),'eligible_official':int((eligible&off).sum()),
          'eligible_hanabi':int((eligible&han).sum()),'eligible_Mc_ge0p5_official':int((eligible&(bc>=.5)&off).sum())})
        for official_gain_budget in (10,20):
            for pe_mode in ('COUNT10','COUNT20','MEDIAN10','MEDIAN20'):
                med=list(b['median'])
                if pe_mode.startswith('MEDIAN'):med[0 if pe_mode.endswith('10') else 1]+=.0001
                bits=np.stack([bc>=.5,bc>=med[0],bc>=med[1],off,han],1)[eligible].astype(int)
                keys,counts=np.unique(bits,axis=0,return_counts=True);n=len(keys)
                a=[];lo=[];hi=[]
                def constraint(coeff,lower=-np.inf,upper=np.inf):a.append(coeff);lo.append(lower);hi.append(upper)
                constraint(np.r_[np.ones(n),np.zeros(n)],10,10)
                constraint(np.ones(2*n),20,20)
                for k in range(n):
                    c=np.zeros(2*n);c[k]=c[n+k]=1;constraint(c,upper=counts[k])
                for j,B in enumerate((10,20)):
                    scope=np.r_[np.ones(n),np.zeros(n)] if B==10 else np.ones(2*n)
                    mc=b['mc'][j]+int(pe_mode==f'COUNT{B}')
                    # B/2+1 members above the threshold suffice to protect the exact median.
                    for feature,minimum in ((0,mc),(1+j,B//2+1),(3,b['official'][j]+int(official_gain_budget==B)),(4,b['hanabi'][j])):
                        constraint(np.tile(keys[:,feature],2)*scope,minimum)
                result=milp(np.zeros(2*n),integrality=np.ones(2*n),bounds=Bounds(np.zeros(2*n),np.tile(counts,2)),
                    constraints=LinearConstraint(np.asarray(a),lo,hi),options={'time_limit':30.})
                row={'deployment':dep,'official_gain_budget':official_gain_budget,'PE_gain_mode':pe_mode,
                    'category_count':n,'solver_status':int(result.status),'sufficient_construction_exists':bool(result.success)}
                if result.success:
                    x=np.rint(result.x).astype(int)
                    if np.any(np.asarray(a)@x<np.asarray(lo)-1e-8) or np.any(np.asarray(a)@x>np.asarray(hi)+1e-8):raise RuntimeError('Integer constraints')
                    for j,B in enumerate((10,20)):
                        use=x[:n] if B==10 else x[:n]+x[n:]
                        for k,col in ((0,'Mc_ge0p5'),(3,'official'),(4,'hanabi')):row[f'Top{B}_{col}']=int(use@keys[:,k])
                rows.append(row)
    dev.csv_write(root/'tables/ANONYMOUS_ELIGIBILITY_COUNTS.csv',pd.DataFrame(inventories))
    dev.csv_write(root/'tables/ANONYMOUS_NESTED_BUDGET_FEASIBILITY.csv',pd.DataFrame(rows))
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(hashes))
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{'id':'MCWF-TARGET-FEASIBILITY-AUDIT','UTC':datetime.now(timezone.utc).isoformat(),
      'purpose':'Descriptive logical feasibility only;not score selection,training,or an oracle ranking.',
      'outputs':'Anonymous category counts;no selected event IDs,rankings,or pair scores exported.',
      'limits':'A feasible construction does not imply waveform data can identify it. An infeasible sufficient-median construction does not prove the true target impossible. Official overlap is not lensed truth.',
      'historical_changes':False,'ranking_changes':False,'goal_achieved':False,'status':t.STATUS})
    import shutil
    shutil.copy2(__file__,root/'scripts/target_feasibility.py')
    lines=['# 目标可行性：只读匿名计数审计','',
      '这不是候选排序，也不是模型。只按PE和官方标签组成匿名类别，检查是否存在嵌套的Top10/20数量组合同时满足既定保护条件。没有导出任何解对应的事件身份。',
      '可行仅说明目标未被现有目录的标签组合排除，不能证明当前波形能辨认这些配对，更不能把求解器用于排名。不可行只针对更严格的充分中位数条件，不能证明真实目标不可能。','',
      '| Run | 官方增加预算 | PE增加方式 | 匿名数量组合可行 |','|---|---:|---|---|']
    lines += [f'|{r["deployment"]}|{r["official_gain_budget"]}|{r["PE_gain_mode"]}|{r["sufficient_construction_exists"]}|' for r in rows]
    lines+=['','未重新训练、未改波形/时间/天空、未修改任何权重与排名。',t.STATUS]
    (root/'reports/FEASIBILITY_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'inventory':inventories,'feasibility':rows}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
