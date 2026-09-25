from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re,json,subprocess,importlib.util,hashlib
import pandas as pd
BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WT=PROJECT/'0909结果/论文Methods更新_20260911/overleaf_worktree'
def old(p):return subprocess.check_output(['git','show','3ebd7a5:'+p],cwd=WT).decode('utf-8')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if __name__=='__main__':
    main=(WT/'main.tex').read_text(encoding='utf-8');before=old('main.tex')
    start=r'\subsection*{Candidate lensed pairs in O4b}';end=r'\section*{Discussion}'
    assert before.split(start)[0]==main.split(start)[0]
    assert before.split(end,1)[1]==main.split(end,1)[1]
    p='tables/o4b/o4b_main_top10.tex';new=(WT/p).read_text(encoding='utf-8')
    rows=lambda text:[s for s in text.splitlines() if re.match(r'^\d+ &',s)]
    original,updated=rows(old(p)),rows(new)
    df=pd.read_csv(next((BASE.parent/'extracted').iterdir())/'tables/O4b/real_GWTC_Top10_FPP.csv')
    assert len(updated)==10
    for a,b,(_,r) in zip(original,updated,df.iterrows()):
        assert a.split(' & ')[:5]==b.split(' & ')[:5]
        k=int(r.background_exceedances)
        expected=('---' if k==0 else f'{100*r.conditional_FPP:.4f}'+r'\%')+f' ({k}/4005)'
        assert b.split(' & ')[5].strip().removesuffix(r'\\').strip()==expected
    assert r'D_{\max}' not in new
    assert r'D_{\max}' in (WT/'tables/c_current/O4b_top20.tex').read_text(encoding='utf-8')
    assert subprocess.check_output(['git','diff','--name-only'],cwd=WT).decode().splitlines()==['main.tex',p]
    (BASE/'compile').mkdir(exist_ok=True)
    a=module('build_base',PROJECT/'0919_C主结果论文修订/finalize_and_compile.py')
    b=module('build_finish',PROJECT/'0921_全文叙述精炼修订/verify_and_compile.py')
    a.ROOT=BASE;a.WT=WT;a.QA=BASE/'compile';b.ROOT=BASE;b.WT=WT
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(lambda s:b.finish_compile(a.compile_one(s)),['main','supplementary']))
    assert all(not x['warnings'] for x in reports),reports
    report={'base_commit':'3ebd7a5','only_O4b_subsection_and_main_table_changed':True,'rank_scores_PE_counts_unchanged':True,'Dmax_retained_in_supplement':True,'figures_unchanged':True,'build':reports}
    (BASE/'QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
