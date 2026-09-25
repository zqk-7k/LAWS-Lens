from pathlib import Path
import shutil,json,importlib.util
from concurrent.futures import ThreadPoolExecutor
BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WT=PROJECT/'0909结果/论文Methods更新_20260911/overleaf_worktree'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if __name__=='__main__':
    target=WT/'figures/fig_background_diagnostics.pdf'
    backup=BASE/'before/figures';backup.mkdir(parents=True,exist_ok=True)
    if not (backup/target.name).exists():shutil.copy2(target,backup/target.name)
    shutil.copy2(BASE/'figures/Fig4_FPP_final.pdf',target)
    data=WT/'source_data/c_current/conditional_fpp';data.mkdir(parents=True,exist_ok=True)
    for p in (BASE/'source_data').glob('*.csv'):shutil.copy2(p,data/p.name)
    a=module('compile_base',PROJECT/'0919_C主结果论文修订/finalize_and_compile.py')
    b=module('compile_finish',PROJECT/'0921_全文叙述精炼修订/verify_and_compile.py')
    a.ROOT=BASE;a.WT=WT;a.QA=BASE/'compile';a.QA.mkdir(exist_ok=True);b.ROOT=BASE;b.WT=WT
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(lambda stem:b.finish_compile(a.compile_one(stem)),['main','supplementary']))
    (BASE/'compile/BUILD_REPORT.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(reports,ensure_ascii=False))
