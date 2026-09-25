from pathlib import Path
import difflib,json,re,shutil
import pandas as pd
BASE=Path(__file__).resolve().parent
WT=BASE.parents[1]/'0909结果/论文Methods更新_20260911/overleaf_worktree'
DATA=next((BASE.parent/'extracted').iterdir())
changes={}
def read(p):return (WT/p).read_text(encoding='utf-8')
def region(s,start,end,new):
    a=s.index(start);b=s.index(end,a)
    return s[:a]+new.strip()+'\n\n'+s[b:]
s=read('main.tex')
s=region(s,r'\subsection*{Conditional false-pair rates',r'\subsection*{Computational cost',r'''
\subsection*{Background probabilities at candidate score thresholds}

We compared pair scores with an unrelated background constructed from 90 isolated validation sources in each run. Their 4{,}005 pairs define the conditional false-positive probability (FPP) as the fraction reaching or exceeding a given score. The comparison uses the mean score across three models for both candidates and background pairs. Higher score thresholds retain fewer background pairs, at the cost of recovering fewer simulated lensed pairs (Fig.~\ref{fig:background-diagnostics}a--c).

At conditional FPP below 1\%, the complete test catalogs retain 134, 133 and 145 of the 180 lensed pairs in O3, O4a and O4b, corresponding to pair recalls of 74.44\%, 73.89\% and 80.56\%. They also retain 870, 965 and 1{,}008 unrelated pairs, respectively. These counts describe recovery in the global pair list rather than retrieval within ten candidates for an individual query. Thus, a low background fraction still leaves unrelated pairs for subsequent assessment when many pairs are searched.

For the real Top-20 lists, nonzero conditional FPP estimates lie below 1\% in every run (Fig.~\ref{fig:background-diagnostics}d--f). The O4a ranks 1--6 and O4b rank~1 have no background exceedances. Their scores extend beyond the largest value in the available reference set, rather than providing a measured probability of zero. The background comparison remains conditional on the simulated population and does not establish the probability that a real candidate is lensed.

\begin{figure*}[tbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_background_diagnostics.pdf}
\caption{\textbf{Conditional FPP at score thresholds and real-candidate scores.} (a--c)~The empirical background fraction versus three-model mean score $S$ in O3, O4a and O4b. Each reference set contains all 4{,}005 pairs of 90 isolated validation sources. Circles and squares mark thresholds recovering 50\% and 90\% of the 180 test lensed pairs, including ties. The score axes use an inverse-hyperbolic-sine scale with linear width 2 and untransformed tick labels. Dashed lines mark 1\%. Dotted lines mark one background count, not confidence bounds. Curves stop at the largest background score without extrapolation. (d--f)~Nonzero conditional FPP at the frozen Top-20 mean scores. Open circles correspond to 1--19 background exceedances and filled circles to at least 20. Counts below the rank axes have no probability ordinate. These empirical functions share sources and noise and are not shown with independent-pair confidence intervals. The consensus ordering is unchanged.}
\label{fig:background-diagnostics}
\end{figure*}
''')
s=region(s,'The rank-1 pair has a conditional annualized',r'\section*{Discussion}',r'''
The rank-1 score exceeds every score among the 4{,}005 validation background pairs. The other nine Top-10 candidates each have one exceedance, corresponding to a conditional FPP of 0.0250\% (Fig.~\ref{fig:background-diagnostics}f). Together with the waveform and posterior comparisons, these counts motivate further examination of the shortlisted pairs. They do not confirm or exclude lensing, because the reference population and detection selection have not been validated as a background for real O4b events. Joint Bayesian analysis is needed to assess a common-source interpretation. The Supplementary Information provides the complete Top-20 comparisons.
''')
s=region(s,'For a real pair, we evaluate a separate background fraction',r'\subsection*{Event selection',r'''
For background comparison, we use the mean score $\overline S$ across the three models for both simulated and real pairs. In each run, the reference set $\mathcal B$ consists of all 4{,}005 unordered pairs of 90 isolated validation sources, excluding every image of a simulated lensed system. At threshold $s$, the empirical conditional FPP is
\begin{equation}
\widehat{\mathrm{FPP}}(s)=\frac{k(s)}{N_{\rm bg}},\qquad
k(s)=\sum_{(a,b)\in\mathcal B}\mathbf 1[\overline S_{ab}\ge s],\qquad
N_{\rm bg}=4{,}005.
\label{eq:conditional-fpp}
\end{equation}
The same reference maps test scores and real-candidate mean scores to background counts. This calculation differs from averaging model-specific FPPs and does not use consensus rank as a score threshold. Ties are included. For zero exceedances, the candidate tables retain the count and leave the FPP entry blank with a dash, rather than assigning a probability or upper limit.

Validation sources also informed calibration and model selection, so these reference tails are not an independent calibration of real-catalog probabilities. Test sources and noise parents are disjoint from validation. The reference pairs share 90 sources and six noise parents, and their number is not an independent sample size. The Supplementary Information reports source- and noise-parent deletion sensitivities, which are not confidence intervals. Differences in source selection and detector networks limit transfer to real candidates. No effective background search exposure has been established, so the fractions are not converted into annual FAR or lensing probabilities.
''')
s=s.replace('Background fractions are evaluated separately for each model, not by treating consensus rank as a score threshold.','The displayed conditional FPP compares candidate mean scores with background mean scores, not consensus ranks.')
assert read('main.tex').split(r'\maketitle')[0]==s.split(r'\maketitle')[0]
changes['main.tex']=s
s=read('supplementary.tex')
s=region(s,r'\section{Conditional annualized false-pair rates',r'\section{Runtime measurements',r'''
\section{Conditional FPP for simulated and real pairs}
\label{sec:supp-candidate-background}

The reference set in each run contains 90 isolated validation sources and all their 4{,}005 unordered pairs. Images from lensed systems do not enter this reference. We average the three model scores for each pair before comparing scores, using the same operation for reference, test and real pairs. Equation~(\ref{eq:supp-conditional-fpp}) defines the empirical mapping,
\begin{equation}
\widehat{\mathrm{FPP}}(s)=\frac{\#\{(a,b)\in\mathcal B:\overline S_{ab}\ge s\}}{4{,}005}.
\label{eq:supp-conditional-fpp}
\end{equation}
It is not the average of three model-specific FPPs. The mean score need not decrease with consensus rank, which is still determined from within-model ranks. Source Data include the mean-score curve at every distinct reference score, without smoothing or tail extrapolation.

The complete test catalog contains 180 lensed pairs and 100{,}845 unrelated pairs among 450 events. Table~\ref{tab:c-fpp-efficiency} reports the pairs retained below conditional FPP 1\%. The singleton-only check uses the 4{,}005 pairs among the 90 test isolated sources. Its exceedance fraction is evaluated at the same threshold and is not assumed to equal 1\%. Pair recall here is different from query Recall@10.

\input{tables/c_current/fpp_efficiency.tex}

The validation set was used for fitting and selection of calibration parameters. The reference is therefore a development-set background, not an independent real-population significance calibration. The test sources and noise parents are disjoint from validation, but the test catalogs were previously examined. Neither reuse of three models nor the large number of pairs creates additional independent observing exposure.

Tables~\ref{tab:c-bg-O3}--\ref{tab:c-bg-O4b} retain integer counts for every real Top-20 pair. Nonzero estimates use $k/4{,}005$. Zero-count FPP cells contain a dash, without a probability upper bound. All Top-10 counts are below 20. The O4a ranks 1--6 and O4b rank~1 have zero counts. The count grid is finite, and absence of an exceedance does not establish zero false-positive risk.

The sensitivity columns recompute the fraction after deleting one reference source, or all events associated with one noise parent, at a time. The resulting minimum--maximum ranges quantify dependence on the available reference set, not confidence intervals. The pairs share 90 sources and six noise parents and cannot be treated as 4{,}005 independent observations.

The current background does not establish effective search exposure, so no annual FAR is reported. Nor is the mapping a probability that a candidate is lensed or a calibration of consensus-rank selection. The controlled source population and amplitude selection differ from an astrophysical event population. In addition, 75 of the 86 real O4b sky maps include Virgo, whereas injection localization uses H1/L1. These differences limit transferring the reference fractions to real-candidate significance.

\input{tables/c_current/candidate_background.tex}
\FloatBarrier
''')
changes['supplementary.tex']=s
real={r:pd.read_csv(DATA/f'tables/{r}/real_GWTC_Top20_FPP.csv').sort_values('consensus_rank') for r in ['O3','O4a','O4b']}
for run in real:
    p=f'tables/c_current/{run}_top20.tex'
    s=read(p).replace('Conditional annualized false-pair rates are listed separately','Conditional FPP and background counts are listed separately')
    changes[p]=s
p='tables/o4b/o4b_main_top10.tex';s=read(p)
a=s.index(r'\caption{');b=s.index(r'\label{',a)
s=s[:a]+r'''\caption{\textbf{O4b consensus Top-10 and background counts: 0/4005 indicates that no background pair reached the candidate score.} All 3{,}655 pairs of 86 events are ranked. $S$ and weighted waveform/time/sky contributions are three-model means. $BC$ gives marginal overlaps for detector-frame chirp mass, mass ratio, effective spin and apparent distance. $D_{\max}$ excludes distance. Conditional FPP compares the mean candidate score with the mean-score background of isolated validation sources and is expressed in percent. It is not a lensing probability or annual FAR.}
'''+s[b:]
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('Rank &'):lines[i]=r'Rank & Candidate pair & $S$ & wf / time / sky & $BC$: $\mathcal M_c/q/\chi_{\rm eff}/d_L^{\rm app}$ & $D_{\max}$ & $k/N_{\rm bg}$ & FPP (\%) \\'
    m=re.match(r'^(\d+) &',line)
    if m:
        r=real['O4b'].iloc[int(m[1])-1];k=int(r.background_exceedances)
        f='---' if k==0 else f'{100*r.conditional_FPP:.4f}'
        lines[i]=' & '.join(line.split(' & ')[:6])+f' & {k}/4005 & {f} '+r'\\'
changes[p]='\n'.join(lines)+'\n'
tables=[]
for run,frame in real.items():
    tables.append(r'''\begin{table}[p]
\centering\small
\caption{\textbf{RUN Top-20 background comparison: 0/4005 indicates that no background pair reached the candidate score.} Conditional FPP is expressed in percent. Deletion ranges are minimum--maximum sensitivities, not confidence intervals. Candidate ordering remains frozen.}
\label{tab:c-bg-RUN}
\begin{tabular}{rrrrr}
\toprule
Rank & $k/N_{\rm bg}$ & FPP (\%) & Source deletion (\%) & Noise deletion (\%) \\
\midrule
'''.replace('RUN',run))
    for _,r in frame.iterrows():
        k=int(r.background_exceedances);f='---' if k==0 else f'{r.conditional_FPP*100:.4f}'
        ranges=['---' if k==0 else f'{r[x+"_min"]*100:.4f}--{r[x+"_max"]*100:.4f}' for x in ['source_leave_one_out','noise_leave_one_out']]
        tables.append(f'{int(r.consensus_rank)} & {k}/4005 & {f} & '+ ' & '.join(ranges)+r' \\'+'\n')
    tables.append('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
changes['tables/c_current/candidate_background.tex']=''.join(tables)
changes['tables/c_current/fpp_efficiency.tex']=r'''\begin{table}[tbp]
\centering\small
\caption{Pair recovery at conditional FPP below 1\%. Each run has 180 lensed and 100{,}845 unrelated test pairs. The final column uses only the 4{,}005 test singleton pairs. Thresholds are mapped through the validation background, not fitted to achieve a test fraction of 1\%.}
\label{tab:c-fpp-efficiency}
\begin{tabular}{lrrrrr}
\toprule
Run & Lensed retained & Pair recall (\%) & Unrelated retained & Unrelated (\%) & Singleton (\%) \\
\midrule
O3 & 134/180 & 74.44 & 870 & 0.8627 & 0.8489 \\
O4a & 133/180 & 73.89 & 965 & 0.9569 & 1.4232 \\
O4b & 145/180 & 80.56 & 1008 & 0.9996 & 0.9488 \\
\bottomrule
\end{tabular}
\end{table}
'''
patch='*** Begin Patch\n'
for p,new in changes.items():
    dest=WT/p
    if dest.exists():
        old=read(p)
        if old==new:continue
        backup=BASE/'before'/p;backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copy2(dest,backup)
        diff=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),n=3))[2:]
        diff=[re.sub(r'^@@.*@@\n$','@@\n',line) for line in diff]
        patch+='*** Update File: '+dest.as_posix()+'\n'+''.join(diff)
    else:patch+='*** Add File: '+dest.as_posix()+'\n'+''.join('+'+x+'\n' for x in new.splitlines())
print(json.dumps(patch+'*** End Patch'))
