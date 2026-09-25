from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re,json,subprocess,importlib.util,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WT=PROJECT/'0909结果/论文Methods更新_20260911/overleaf_worktree'
def old(p):return subprocess.check_output(['git','show','1c22d35:'+p],cwd=WT).decode('utf-8')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if __name__=='__main__':
    main=(WT/'main.tex').read_text(encoding='utf-8');before=old('main.tex')
    assert before.split(r'\abstract{')[1].split(r'\maketitle')[0]==main.split(r'\abstract{')[1].split(r'\maketitle')[0]
    p='tables/o4b/o4b_main_top10.tex';new=(WT/p).read_text(encoding='utf-8')
    rows=lambda text:[s for s in text.splitlines() if re.match(r'^\d+ &',s)]
    for a,b in zip(rows(old(p)),rows(new)):
        assert [x.strip() for x in a.removesuffix(r'\\').split(' & ')]==[x.strip() for x in b.split(' & ')[:6]]
    audit=json.loads((BASE.parent/'尾部模型探索/AUDIT.json').read_text(encoding='utf-8'))
    for i in audit['inputs']:
        with Path(i['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==i['sha256']
    pp=pd.read_csv(BASE.parent/'尾部模型探索/tables/candidate_predictions_all.csv')
    dd=pd.read_csv(BASE.parent/'尾部模型探索/tables/deletion_sensitivity.csv')
    for rank in range(2,10):
        d=dd[(dd.run=='O4b')&(dd['rank']==rank)&(dd.q==.95)&dd.kind.isin(['source','noise_parent'])]
        assert d.outside_fitted_support.any(),(rank,dd.kind.unique())
    q=pp[(pp.run=='O4b')&(pp.model=='GPD')&(pp['rank']==1)]
    assert q.outside_fitted_support.all() and len(q)==5
    (BASE/'compile').mkdir(exist_ok=True)
    a=module('build_base',PROJECT/'0919_C主结果论文修订/finalize_and_compile.py')
    b=module('build_finish',PROJECT/'0921_全文叙述精炼修订/verify_and_compile.py')
    a.ROOT=BASE;a.WT=WT;a.QA=BASE/'compile';b.ROOT=BASE;b.WT=WT
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(lambda s:b.finish_compile(a.compile_one(s)),['main','supplementary']))
    report={'base_commit':'1c22d35','abstract_unchanged':True,'all_12_inputs_hash_unchanged':True,'main_candidate_original_columns_unchanged':True,'GPD_not_replacement_for_empirical_FPP':True,'build':reports}
    (BASE/'QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
