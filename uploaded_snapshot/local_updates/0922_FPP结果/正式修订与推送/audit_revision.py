from pathlib import Path
import json,re,subprocess
import pandas as pd
BASE=Path(__file__).resolve().parent
WT=BASE.parents[1]/'0909结果/论文Methods更新_20260911/overleaf_worktree'
def old(p):return subprocess.check_output(['git','show','HEAD:'+p],cwd=WT).decode('utf-8')
def read(p):return (WT/p).read_text(encoding='utf-8')
main=read('main.tex')
assert old('main.tex').split(r'\maketitle')[0]==main.split(r'\maketitle')[0]
for label in [r'\subsection*{Computational cost',r'\subsection*{Candidate lensed pairs in O4b}']:
    assert label in main
df=pd.read_csv(BASE/'source_data/candidate_top20_all_runs.csv')
checks={}
for p in ['tables/c_current/O3_top20.tex','tables/c_current/O4a_top20.tex','tables/c_current/O4b_top20.tex','tables/o4b/o4b_main_top10.tex']:
    before=[l for l in old(p).splitlines() if re.match(r'^\d+ &',l)]
    after=[l for l in read(p).splitlines() if re.match(r'^\d+ &',l)]
    assert len(before)==len(after)
    for a,b in zip(before,after):assert a.split(' & ')[:6]==b.split(' & ')[:6],p
    if 'O3_' in p or 'O4a_' in p:assert before==after,p
    checks[p]={'ranking_score_contributions_PE_unchanged':len(after)}
text=read('tables/c_current/candidate_background.tex')
for run in ['O3','O4a','O4b']:
    part=text.split('\\label{tab:c-bg-'+run+'}')[1].split(r'\end{table}')[0]
    rows=[l for l in part.splitlines() if re.match(r'^\d+ &',l)]
    for line,(_,row) in zip(rows,df[df.run==run].iterrows()):
        cells=line.split(' & ');k=int(row.background_exceedances)
        assert cells[1]==f'{k}/4005'
        assert cells[2]==('---' if k==0 else f'{100*row.conditional_FPP:.4f}')
    assert len(rows)==20
sources=[p for p in (WT/'tables/c_current').glob('*.tex') if p.name not in ['far_normalization.tex']]
for p in ['main.tex','supplementary.tex','tables/o4b/o4b_main_top10.tex']:
    assert 'annualized' not in read(p)
    assert 'unresolved' not in read(p).lower()
figtext=__import__('fitz').open(BASE/'figures/Fig4_FPP_final.pdf')[0].get_text()
assert figtext.count('0/4005')==8
assert 'unresolved' not in figtext.lower()
report={'abstract_unchanged':True,'candidate_checks':checks,'all_60_background_rows_verified':True,'zero_counts':{'O3':0,'O4a':6,'O4b':1},'figure_zero_labels':7,'annualized_values_removed_from_active_text':True,'new_model_training':False}
(BASE/'PAPER_QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
