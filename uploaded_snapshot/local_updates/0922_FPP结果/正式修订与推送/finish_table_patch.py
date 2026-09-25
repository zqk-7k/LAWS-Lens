from pathlib import Path
import re,json,difflib
import pandas as pd
BASE=Path(__file__).resolve().parent
p=BASE.parents[1]/'0909结果/论文Methods更新_20260911/overleaf_worktree/tables/c_current/O4b_top20.tex'
old=p.read_text(encoding='utf-8');s=old
df=pd.read_csv(next((BASE.parent/'extracted').iterdir())/'tables/O4b/real_GWTC_Top20_FPP.csv')
s=re.sub(r'\\caption\{.*?\}\n(?=\\label)',lambda m:r'\caption{\textbf{O4b consensus candidates '+('1--10' if '1--10.' in m[0] else '11--20')+r': 0/4005 indicates that no background pair reached the candidate score.} $BC$ order is detector-frame chirp mass, mass ratio, effective spin and apparent distance. Contributions and $S$ are model means. Conditional FPP uses the validation background and is expressed in percent. No O4b individual-pair results are included in the published lensing searches used for comparison.}'+'\n',s)
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('Rank &'):lines[i]=line.split(' & ')[0]+' & '+ ' & '.join(line.split(' & ')[1:6])+r' & $k/N_{\rm bg}$ & FPP (\%) \\'
    m=re.match(r'^(\d+) &',line)
    if m:
        r=df.loc[df.consensus_rank==int(m[1])].iloc[0];k=int(r.background_exceedances);f='---' if k==0 else f'{r.conditional_FPP*100:.4f}'
        lines[i]=' & '.join(line.split(' & ')[:6])+f' & {k}/4005 & {f} '+r'\\'
new='\n'.join(lines)+'\n'
diff=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),n=3))[2:]
diff=[re.sub(r'^@@.*@@\n$','@@\n',l) for l in diff]
print(json.dumps('*** Begin Patch\n*** Update File: '+p.as_posix()+'\n'+''.join(diff)+'*** End Patch'))
