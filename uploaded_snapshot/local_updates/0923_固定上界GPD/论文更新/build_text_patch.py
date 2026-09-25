from pathlib import Path
import difflib,json,re
import pandas as pd
BASE=Path(__file__).resolve().parent
WT=BASE.parents[1]/'0909结果/论文Methods更新_20260911/overleaf_worktree'
TAB=BASE.parent/'tables'
F=pd.read_csv(TAB/'fits.csv');P=pd.read_csv(TAB/'candidate_Top50_all_bounds.csv')
H=pd.read_csv(TAB/'heldout_checks.csv');D=pd.read_csv(TAB/'deletion_summary.csv')
patch=[]
def edit(path,new):
    p=WT/path;old=p.read_text(encoding='utf-8')
    if old==new:return
    diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=2,lineterm=''))[2:]
    patch.append('*** Update File: '+p.as_posix()+'\n'+'\n'.join('@@' if l.startswith('@@') else l for l in diff)+'\n')
def fmt(p):return f'{p:.5f}' if p>=.001 else f'{p:.8f}'

path='tables/o4b/o4b_main_top10.tex';t=(WT/path).read_text(encoding='utf-8')
cap=r'''\caption{\textbf{O4b consensus Top-10 and background comparison: 0/4005 indicates that no background pair reached the candidate score.} All 3{,}655 pairs of 86 events are ranked. $S$ and weighted waveform/time/sky contributions are three-model means. $BC$ gives marginal overlaps for detector-frame chirp mass, mass ratio, effective spin and apparent distance. Background counts compare candidate mean scores with 4{,}005 pairs of isolated validation sources. Model FPP is estimated from a generalized Pareto tail above the 95th background percentile, with fixed upper score bound $B=4$. These percentages depend on the tail assumption and reference population, and are neither lensing probabilities nor annual FAR. Endpoint and reference-deletion sensitivities are reported in the Supplementary Information.}'''
t=re.sub(r'\\caption\{.*?\}\n\\label',lambda m:cap+'\n\\label',t,count=1,flags=re.S)
t=t.replace(r'\shortstack{Conditional FPP\\(background count)} & \shortstack{GPD tail (\%)\\exploratory}',r'\shortstack{Background\\count} & \shortstack{Model FPP (\%)\\$B=4$}')
lines=t.splitlines()
for i,line in enumerate(lines):
    if not re.match(r'^\d+ &',line):continue
    fields=line.removesuffix(r'\\').strip().split(' & ');rank=int(fields[0])
    r=P[(P.run=='O4b')&(P.B==4)&(P['rank']==rank)].iloc[0]
    lines[i]=' & '.join(fields[:5]+[f'{r.empirical_count}/4005',fmt(r.estimated_tail_percent)])+r' \\'
edit(path,'\n'.join(lines)+'\n')

tables=[]
for run in ['O3','O4a','O4b']:
    s=[r'\begin{table}[p]',r'\centering',
       r'\caption{\textbf{'+run+r' Top-20 empirical counts and finite-tail model estimates.} Model FPP values are percentages fitted above the 95th background percentile. The endpoint $B=4$ defines the displayed model, and $B=4.5$ and $5$ assess endpoint sensitivity. These settings are imposed assumptions, not confidence limits or independently measured score maxima. Ranks are the consensus ordering.}',
       r'\label{tab:gpd-'+run+'}',r'\fontsize{8}{10}\selectfont',r'\begin{tabular}{rrrrrr}',r'\toprule',r'Rank & $S$ & Background count & $B=4$ (\%) & $B=4.5$ (\%) & $B=5$ (\%) \\',r'\midrule']
    for rank in range(1,21):
        rr=P[(P.run==run)&(P['rank']==rank)].sort_values('B');r=rr.iloc[0]
        s.append(f'{rank} & {r.S:.3f} & {r.empirical_count}/4005 & '+' & '.join(fmt(p) for p in rr.estimated_tail_percent)+r' \\')
    s += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    tables.extend(s)
edit('tables/c_current/tail_candidates.tex','\n'.join(tables)+'\n')
s=[r'\begin{table}[p]',r'\centering',r'\caption{\textbf{GPD parameters under fixed upper score bounds.} Each fit uses 201 excesses above the 95th percentile of 4{,}005 validation pairs. The last column is the mean conditional tail log density of the existing test excesses at the same threshold. It is a descriptive predictive score, not a calibrated significance test. None of these test scores exceeds the three specified bounds.}',r'\label{tab:gpd-diagnostics}',r'\small',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Run & Bound $B$ & Threshold $u$ & Shape $\xi$ & Scale $\sigma$ & Test mean log density \\',r'\midrule']
for r in F.itertuples():s.append(f'{r.run} & {r.B:g} & {r.u:.4f} & {r.xi:.4f} & {r.scale:.4f} & {r.heldout_mean_tail_logpdf:.4f}'+r' \\')
s += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[p]',r'\centering',r'\caption{\textbf{Test-background counts at model probability thresholds.} The tail model has $B=4$. Entries are exceedances among 4{,}005 pairs of test isolated sources. Expected counts are nominal model probabilities multiplied by 4{,}005, irrespective of dependence. These previously examined data do not provide new blind confirmation.}',r'\label{tab:gpd-heldout}',r'\begin{tabular}{lrrrrr}',r'\toprule',r' & 1\% & 0.5\% & 0.1\% & 0.05\% & 0.025\% \\',r'\midrule',r'Expected & 40.05 & 20.03 & 4.01 & 2.00 & 1.00 \\']
for run in ['O3','O4a','O4b']:
    rr=H[(H.run==run)&(H.B==4)].sort_values('target_probability',ascending=False)
    s.append(run+' & '+' & '.join(str(v) for v in rr.observed)+r' \\')
s += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[p]',r'\centering',r'\caption{\textbf{Rank-1 model FPP sensitivity to reference deletion.} Each source deletion removes all pairs containing that source, and each noise deletion removes all pairs involving the corresponding parent block. The 95th-percentile threshold and GPD parameters are refitted while the upper bound remains fixed. Ranges are minimum--maximum percentages, not confidence intervals. Each run has 90 source deletions and six noise-parent deletions per endpoint.}',r'\label{tab:gpd-fixed-deletion}',r'\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Run & Bound $B$ & Source deletion (\%) & Noise deletion (\%) \\',r'\midrule']
for run in ['O3','O4a','O4b']:
    for B in [4,4.5,5]:
        rr=D[(D.run==run)&(D.B==B)].set_index('unit')
        vals=[fmt(rr.loc[k,'rank1_min_pct'])+'--'+fmt(rr.loc[k,'rank1_max_pct']) for k in ['source','noise']]
        s.append(f'{run} & {B:g} & '+' & '.join(vals)+r' \\')
s += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
edit('tables/c_current/tail_diagnostics.tex','\n'.join(s)+'\n')

path='supplementary.tex';t=(WT/path).read_text(encoding='utf-8')
t=t.replace('Table~\\ref{tab:c-fpp-efficiency} reports the pairs retained below conditional FPP 1\\%.','Table~\\ref{tab:c-fpp-efficiency} reports the pairs retained below empirical conditional FPP 1\\%.')
t=t.replace('Nonzero estimates use $k/4{,}005$. Zero-count FPP cells contain a dash, without a probability upper bound.','Nonzero empirical estimates use $k/4{,}005$. Zero-count empirical FPP cells contain a dash, without a probability upper bound. Model estimates are presented separately in Tables~\\ref{tab:gpd-O3}--\\ref{tab:gpd-O4b}.')
start=t.index(r'\subsection{Exploratory modelling of the high-score tail}')
end=t.index(r'\section{Runtime measurements on a common O3 sample}',start)
new=r'''\subsection{Finite-tail modelling of background scores}
\label{sec:supp-gpd}

The model estimates the background probability of reaching a candidate score by fitting a GPD to excesses above the 95th reference percentile. Each run supplies 201 excesses from 4{,}005 pairs, with its own threshold and fitted parameters. The upper score bound is fixed at $B=4$, while $B=4.5$ and $5$ quantify sensitivity to that assumption.

For excess $y=s-u$, let $E=B-u$ and write the conditional density as
\begin{equation}
f(y\mid a,E)=\frac{a}{E}\left(1-\frac{y}{E}\right)^{a-1},\qquad 0<y<E,\quad a>0.
\end{equation}
This is a GPD with shape $\xi=-1/a$ and scale $\sigma=E/a$. For $m$ observed excesses, the pairwise log likelihood is
\begin{equation}
\ell(a)=m\log a-m\log E+(a-1)\sum_{j=1}^{m}\log(1-y_j/E).
\end{equation}
Its maximum gives $\widehat a=-m/\sum_j\log(1-y_j/E)$. Multiplication of the conditional survival function by $m/4{,}005$ gives the model FPP reported in the candidate tables. The analytic estimate was checked against scalar numerical optimization and the SciPy GPD density and survival functions. Pairwise dependence makes this a product-likelihood point estimate, not an independent-pair uncertainty analysis.

The bound specifies a finite-tail model rather than an established maximum of the score or unrelated population. These endpoint assumptions were examined after the candidate scores were available, so they are not an independently selected significance model. Candidate scores enter the probability mapping but not the likelihood used to estimate $a$. All reported Top-50 scores lie below $B=4$, and values at or beyond an assumed bound are treated as outside model support.

Endpoint sensitivity is greatest for the O4a rank-1 pair, whose score is 3.9374 (Fig.~\ref{fig:supp-tail-endpoints}). Its estimated FPP is 0.00072850\% at $B=4$, 0.02606\% at $B=4.5$ and 0.05488\% at $B=5$. The O3 rank-1 estimates are 0.20373\%, 0.22438\% and 0.23987\%, respectively, and the O4b estimates are 0.03042\%, 0.04620\% and 0.05923\%. The variation describes dependence on the imposed endpoint rather than a confidence interval. Complete Top-20 estimates appear in Tables~\ref{tab:gpd-O3}--\ref{tab:gpd-O4b}.

Reference deletion refits the threshold and shape after removing each source, each noise parent, or the highest-scoring background pair. The three runs and three bounds give 873 deletion fits, summarized for rank~1 in Table~\ref{tab:gpd-fixed-deletion}. All retain the specified endpoint, so continued coverage of a candidate after deletion follows from that constraint. It does not validate the endpoint or its extrapolated probability.

The existing test reference contains 90 isolated sources and 4{,}005 pairs per run, disjoint from validation in source and noise-parent identity. Table~\ref{tab:gpd-heldout} compares observed counts with the counts expected at fixed model probability thresholds. The $B=4$ fits give higher reference log likelihoods and test mean tail log densities than $B=4.5$ or $5$ in each run. These descriptive comparisons use previously examined data and do not establish calibration in the extreme candidate tail.

Additional fits with freely estimated endpoints assess how well the background constrains a finite score limit. Fits above the 90th, 95th and 97.5th percentiles cover three model scores and their mean in each run, giving 36 fits. At the 95th percentile, the mean-score endpoints are 2.3755, 3.2924 and 2.8458 in O3, O4a and O4b. Three O3 and two O4a test pairs exceed those fitted endpoints. All 30 real Top-10 scores also exceed the highest background percentile used in the predictive screening checks. Thus the background diagnostics do not establish the assumed endpoint or validate real-candidate significance. Model FPP remains conditional on the reference population and endpoint assumption, and is not annual FAR or a posterior probability of lensing.

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/supp_tail_endpoint_diagnostics.pdf}
\caption{\textbf{Dependence of rank-1 model FPP on the fixed upper score bound.} Columns show O3, O4a and O4b. Each point uses the same 4{,}005-pair mean-score background, a 95th-percentile tail threshold and a separately fitted GPD shape under the indicated bound. Lines connect the three imposed settings and do not represent independent repetitions or confidence intervals. Vertical axes show percentages on logarithmic scales, with separate ranges to resolve each run's response. The O4a estimate is particularly sensitive because its candidate score lies close to the bound of 4.}
\label{fig:supp-tail-endpoints}
\end{figure}

\input{tables/c_current/tail_diagnostics.tex}
\input{tables/c_current/tail_candidates.tex}
\FloatBarrier

'''
t=t[:start]+new+t[end:]
edit(path,t)
print('*** Begin Patch\n'+''.join(patch)+'*** End Patch')
