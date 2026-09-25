from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re,json,subprocess,importlib.util,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WT=PROJECT/'0909结果/论文Methods更新_20260911/overleaf_worktree'
def old(p):return subprocess.check_output(['git','show','86ed5d7:'+p],cwd=WT).decode('utf-8')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if __name__=='__main__':
    main=(WT/'main.tex').read_text(encoding='utf-8')
    assert old('main.tex').split(r'\maketitle')[0]==main.split(r'\maketitle')[0]
    p='tables/o4b/o4b_main_top10.tex';new=(WT/p).read_text(encoding='utf-8')
    rows=lambda text:[s for s in text.splitlines() if re.match(r'^\d+ &',s)]
    assert len(rows(old(p)))==len(rows(new))==10
    predicted=pd.read_csv(BASE.parent/'tables/candidate_Top50_all_bounds.csv')
    for a,b in zip(rows(old(p)),rows(new)):
        assert a.split(' & ')[:5]==b.split(' & ')[:5]
        fields=b.removesuffix(r'\\').strip().split(' & ');rank=int(fields[0])
        r=predicted[(predicted.run=='O4b')&(predicted.B==4)&(predicted['rank']==rank)].iloc[0]
        assert fields[5]==f'{r.empirical_count}/4005'
        assert np.isclose(float(fields[6]),r.estimated_tail_percent,atol=5.1e-6,rtol=0)
    for p in ['tables/c_current/metrics190.tex','tables/c_current/metrics450.tex','tables/c_current/real_budget.tex','tables/c_current/fpp_efficiency.tex','tables/c_current/O3_top20.tex','tables/c_current/O4a_top20.tex','tables/c_current/O4b_top20.tex']:
        assert old(p).replace('\r\n','\n')==(WT/p).read_text(encoding='utf-8').replace('\r\n','\n')
    audit=json.loads((BASE.parent/'AUDIT.json').read_text(encoding='utf-8'))
    for p,h in audit['input_hashes_unchanged'].items():
        with Path(p).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    for run in ['O3','O4a','O4b']:
        tt=(WT/'tables/c_current/tail_candidates.tex').read_text(encoding='utf-8').split(r'\label{tab:gpd-'+run+'}')[1].split(r'\end{table}')[0]
        rr=rows(tt);assert len(rr)==20
        for line in rr:
            vals=line.removesuffix(r'\\').strip().split(' & ');rank=int(vals[0])
            pp=predicted[(predicted.run==run)&(predicted['rank']==rank)].sort_values('B')
            np.testing.assert_allclose([float(z) for z in vals[3:]],pp.estimated_tail_percent,atol=5.1e-6,rtol=0)
    (BASE/'compile').mkdir(exist_ok=True)
    a=module('build_base',PROJECT/'0919_C主结果论文修订/finalize_and_compile.py')
    b=module('build_finish',PROJECT/'0921_全文叙述精炼修订/verify_and_compile.py')
    a.ROOT=BASE;a.WT=WT;a.QA=BASE/'compile';b.ROOT=BASE;b.WT=WT
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(lambda s:b.finish_compile(a.compile_one(s)),['main','supplementary']))
    result={'base_commit':'86ed5d7','abstract_unchanged':True,'recall_PE_official_ranks_unchanged':True,'main_table_numbers_checked':10,'supplementary_model_values_checked':180,'input_hashes_unchanged':len(audit['input_hashes_unchanged']),'build':reports}
    (BASE/'QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
