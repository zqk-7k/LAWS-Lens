"""Existing-data tail model: figures and table/text patch, no refitting or ranking."""
from pathlib import Path
import json, shutil, hashlib, math, difflib, sys
import numpy as np
import pandas as pd
from scipy.stats import genpareto
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import fitz

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WT=PROJECT/'0909结果/论文Methods更新_20260911/overleaf_worktree'
TAIL=BASE.parent/'尾部模型探索/tables'
RAW=next((BASE.parent/'extracted').iterdir())
OUT=BASE/'figures'
OUT.mkdir(exist_ok=True)
RUNS=['O3','O4a','O4b']
F=pd.read_csv(TAIL/'tail_fits.csv')
P=pd.read_csv(TAIL/'candidate_predictions_all.csv')
D=pd.read_csv(TAIL/'deletion_sensitivity.csv')
V=pd.read_csv(TAIL/'test_support_violations.csv')
H=pd.read_csv(TAIL/'heldout_checks.csv')
Q95=F.query('model=="GPD" and q==0.95').set_index('run')
PURPLE='#7756A3';TEAL='#1B8A78';ORANGE='#B65C39';GRAY='#737373'
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8.5,
 'axes.labelweight':'bold','axes.titlesize':9,'axes.titleweight':'bold',
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'legend.frameon':False,'legend.fontsize':7.5,
 'pdf.fonttype':42,'svg.fonttype':'none'})

def pred(run):return P.query('run==@run and model=="GPD" and q==0.95').sort_values('rank')
def tail(bg,s):return len(bg)-np.searchsorted(np.sort(bg),np.asarray(s),side='left')
def probtxt(p):return f'{100*p:.5f}'
def empirical(k):return ('---' if k==0 else f'{100*k/4005:.4f}'+r'\%')+f' ({k}/4005)'

fig=plt.figure(figsize=(7.2047244094,7.0866141732))
grid=fig.add_gridspec(2,3,left=.105,right=.98,bottom=.23,top=.83,wspace=.34,hspace=.52)
upper=[];lower=[]
for c,run in enumerate(RUNS):
    bg=pd.read_csv(RAW/f'background/{run}/mean_S_calibration_pairs.csv').final_score_POSITIVE.to_numpy()
    inj=pd.read_parquet(RAW/f'tables/{run}/injection_mean_S_FPP.parquet')
    r=pred(run);f=Q95.loc[run]
    assert len(bg)==4005 and len(r)==20
    np.testing.assert_array_equal(tail(bg,r.S),r.empirical_count)
    ax=fig.add_subplot(grid[0,c]);upper.append(ax)
    x=np.unique(bg);ax.step(x,tail(bg,x)/len(bg),where='pre',color=PURPLE,lw=1.35)
    sx=np.linspace(f.u,genpareto.isf(5e-5/f.tail_weight,f.xi,scale=f.scale)+f.u,250)
    py=f.tail_weight*genpareto.sf(sx-f.u,f.xi,scale=f.scale)
    assert np.all(py>0) and np.all(np.diff(py)<=0)
    ax.plot(sx,py,color=ORANGE,ls='--',lw=1.2)
    true=np.sort(inj.loc[inj.is_true_pair,'final_score_POSITIVE'])[::-1]
    for q,m in [(.5,'o'),(.9,'s')]:
        t=true[math.ceil(q*len(true))-1]
        ax.plot(t,tail(bg,[t])[0]/4005,m,color=PURPLE,ms=4.2,mec='white',mew=.5,zorder=4)
    ax.set_xscale('asinh',linear_width=2)
    ax.set_xlim(2*np.sinh(np.arcsinh(x.min()/2)-.07),2*np.sinh(np.arcsinh(max(x.max(),sx.max())/2)+.14))
    ticks=[v for v in [-300,-30,-10,-3,0,3,30] if ax.get_xlim()[0]<v<ax.get_xlim()[1]]
    ax.set_xticks(ticks,[str(v) for v in ticks]);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yscale('log');ax.set_ylim(4e-5,1.45)
    ax.set_yticks([.0001,.001,.01,.1,1],['0.01','0.1','1','10','100'])
    ax.set_title(run,pad=8);ax.set_xlabel('Score threshold S')
    if c==0:ax.set_ylabel('Background tail (%)')
    ax.text(-.20,1.05,chr(97+c),transform=ax.transAxes,fontsize=10,fontweight='bold')
    ax=fig.add_subplot(grid[1,c]);lower.append(ax)
    k=r.empirical_count.to_numpy();ranks=r['rank'].to_numpy()
    for mask,filled in [((k>0)&(k<20),False),(k>=20,True)]:
        ax.scatter(ranks[mask],k[mask]/4005,s=19,marker='o',facecolors=TEAL if filled else 'white',edgecolors=TEAL,lw=.9,zorder=4)
    ok=~r.outside_fitted_support
    ax.scatter(r.loc[ok,'rank'],r.loc[ok,'estimated_tail_probability'],s=14,marker='D',facecolors='none',edgecolors=ORANGE,lw=.8,zorder=3)
    ax.set_yscale('log');ax.set_ylim(5e-6,.025)
    assert r.loc[ok,'estimated_tail_probability'].min()>5e-6
    ax.set_yticks([.00001,.0001,.001,.01],['0.001','0.01','0.1','1'])
    ax.set_xlim(.3,20.7);ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('Candidate rank',labelpad=47)
    if c==0:ax.set_ylabel('Background tail (%)')
    ax.text(-.20,1.05,chr(100+c),transform=ax.transAxes,fontsize=10,fontweight='bold')
    for rank in ranks[k==0]:
        ax.text(rank,-.16,'0/4005',transform=ax.get_xaxis_transform(),rotation=90,rotation_mode='anchor',ha='right',va='center',fontsize=6,clip_on=False)
for ax in upper+lower:
    ax.axhline(.01,color=GRAY,ls='--',lw=.65)
    ax.axhline(1/4005,color='#BBBBBB',ls=':',lw=.65)
    ax.yaxis.set_minor_locator(NullLocator())
handles=[Line2D([],[],color=PURPLE,lw=1.3,label='Empirical background'),Line2D([],[],color=ORANGE,ls='--',lw=1.2,label='GPD fit (exploratory)'),Line2D([],[],color=GRAY,ls='--',lw=.7,label='1% threshold'),Line2D([],[],color=PURPLE,marker='o',ls='none',ms=4,label='50% pair recall'),Line2D([],[],color=PURPLE,marker='s',ls='none',ms=4,label='90% pair recall')]
leg1=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.955),ncol=3,columnspacing=1.4,handlelength=1.6)
fig.suptitle('0/4005 indicates that no background pair reached the candidate score',fontsize=8.5,fontweight='bold',y=.995)
leg2=fig.legend(handles=[Line2D([],[],color=TEAL,marker='o',mfc='white',ls='none',ms=4,label='Empirical: 1–19 counts'),Line2D([],[],color=TEAL,marker='o',ls='none',ms=4,label='Empirical: ≥20 counts'),Line2D([],[],color=ORANGE,marker='D',mfc='none',ls='none',ms=4,label='GPD estimate')],loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,columnspacing=1.2)
fig.canvas.draw();ren=fig.canvas.get_renderer()
assert leg1.get_window_extent(ren).y0>max(a.get_tightbbox(ren).y1 for a in upper)
assert leg2.get_window_extent(ren).y1<min(a.get_tightbbox(ren).y0 for a in lower)
fig.savefig(OUT/'Fig4_empirical_and_GPD.pdf')
fig.savefig(OUT/'Fig4_empirical_and_GPD.svg')
fig.savefig(OUT/'Fig4_empirical_and_GPD.png',dpi=600)
fig.savefig(OUT/'Fig4_empirical_and_GPD.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(7.2047244094,3.0708661417))
fig.subplots_adjust(left=.095,right=.98,bottom=.21,top=.78,wspace=.35)
for c,(ax,run) in enumerate(zip(axs,RUNS)):
    f=F.query('run==@run and model=="GPD"').sort_values('q')
    ax.plot(f.q*100,f.endpoint,'o-',color=ORANGE,ms=4,lw=1.2)
    r=pred(run).iloc[0]
    ax.axhline(r.S,color=PURPLE,ls='--',lw=1.1)
    ax.axhline(V.query('run==@run').iloc[0].test_singleton_max_score,color=TEAL,ls=':',lw=1.2)
    ax.set_title(run);ax.set_xlabel('Threshold percentile');ax.set_xticks([90,95,98])
    if c==0:ax.set_ylabel('Score S')
    ax.text(-.20,1.06,chr(97+c),transform=ax.transAxes,fontweight='bold',fontsize=10)
fig.legend(handles=[Line2D([],[],color=ORANGE,marker='o',lw=1.2,label='Fitted upper endpoint'),Line2D([],[],color=PURPLE,ls='--',label='Real rank 1 score'),Line2D([],[],color=TEAL,ls=':',label='Maximum test background score')],loc='upper center',bbox_to_anchor=(.53,.98),ncol=3,handlelength=2,columnspacing=1.2)
fig.savefig(OUT/'Tail_endpoint_diagnostics.pdf')
fig.savefig(OUT/'Tail_endpoint_diagnostics.svg')
fig.savefig(OUT/'Tail_endpoint_diagnostics.png',dpi=600)
plt.close(fig)

# Export exact source tables. These files add provenance without altering inputs.
dest=WT/'source_data/c_current/tail_model';dest.mkdir(parents=True,exist_ok=True)
for name in ['tail_fits.csv','candidate_predictions_all.csv','candidate_Top20_summary.csv','heldout_checks.csv','test_support_violations.csv','deletion_sensitivity.csv']:
    shutil.copy2(TAIL/name,dest/name)
for src,tgt in [('Fig4_empirical_and_GPD.pdf','fig_background_diagnostics.pdf'),('Tail_endpoint_diagnostics.pdf','supp_tail_endpoint_diagnostics.pdf')]:
    target=WT/'figures'/tgt
    if target.exists():
        backup=BASE/'before';backup.mkdir(exist_ok=True)
        if not (backup/tgt).exists():shutil.copy2(target,backup/tgt)
    shutil.copy2(OUT/src,target)

if '--figures-only' in sys.argv:
    print('Figures regenerated; text and table sources unchanged.')
    sys.exit(0)

patches=[]
def edit(path,new):
    p=WT/path
    if p.exists():
        old=p.read_text(encoding='utf-8')
        if old==new:return
        diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=2,lineterm=''))[2:]
        patches.append('*** Update File: '+p.as_posix()+'\n'+'\n'.join('@@' if l.startswith('@@') else l for l in diff)+'\n')
    else:patches.append('*** Add File: '+p.as_posix()+'\n'+''.join('+'+l+'\n' for l in new.splitlines()))

# Main table adds an explicitly model-based column. Existing first six columns unchanged.
path='tables/o4b/o4b_main_top10.tex';t=(WT/path).read_text(encoding='utf-8')
t=t.replace('Additional posterior diagnostics are provided in the Supplementary Information.',r'The last column gives exploratory generalized Pareto distribution (GPD) tail estimates in percent, fitted above the 95th background percentile. A dagger denotes a score beyond the fitted upper endpoint, not zero probability. These model estimates are unstable under reference deletion and do not replace the empirical counts. Additional diagnostics are provided in the Supplementary Information.')
t=t.replace(r'\fontsize{8}{10.5}',r'\fontsize{7}{9}').replace(r'\tabcolsep}{3pt}',r'\tabcolsep}{2pt}').replace('rlrllc@{}','rlrllcc@{}')
t=t.replace(r'(background count)} \\',r'(background count)} & \shortstack{GPD tail (\%)\\exploratory} \\')
rows=[];pr=pred('O4b').set_index('rank')
for line in t.splitlines():
    if line and line.split(' & ')[0].isdigit():
        r=pr.loc[int(line.split(' & ')[0])]
        value=r'---$^\dagger$' if r.outside_fitted_support else probtxt(r.estimated_tail_probability)
        line=line.removesuffix(r'\\')+' & '+value+r' \\'
    rows.append(line)
edit(path,'\n'.join(rows)+'\n')

main=(WT/'main.tex').read_text(encoding='utf-8')
anchor=r'\begin{figure*}[tbp]'+'\n'+r'\centering'+'\n'+r'\includegraphics[width=\textwidth]{figures/fig_background_diagnostics.pdf}'
addition=r'''We also fitted generalized Pareto distributions to the high-score background tail to investigate whether a continuous model could distinguish candidates sharing the same empirical count. The fits assign different tail estimates to such candidates, but their upper endpoints depend on the reference sample. In O4b, rank~1 exceeds the fitted endpoint for all five thresholds examined. Existing test backgrounds also contain scores beyond the central fitted endpoints in O3 and O4a. We therefore show these estimates as an exploratory comparison, not a replacement for the empirical FPP (Supplementary Information).

'''
assert main.count(anchor)==1
main=main.replace(anchor,addition+anchor)
start=main.index(r'\caption{\textbf{Conditional FPP at score thresholds')
end=main.index('\n'+r'\label{fig:background-diagnostics}',start)
main=main[:start]+r'''\caption{\textbf{Empirical background probabilities and exploratory tail models.} (a--c)~Background exceedance fractions versus three-model mean score $S$ in O3, O4a and O4b. Each reference set contains 4{,}005 pairs of 90 isolated validation sources. Purple steps show empirical conditional FPP. Orange dashed curves show GPD fits above the 95th score percentile. Circles and squares mark thresholds recovering 50\% and 90\% of the 180 test lensed pairs. Score axes use an inverse-hyperbolic-sine scale with linear width 2. Grey dashed lines mark 1\%, and dotted lines mark one background count. (d--f)~Empirical conditional FPP and exploratory GPD estimates for the frozen Top-20. Open circles denote 1--19 exceedances, filled circles at least 20, and orange diamonds model estimates. Scores beyond fitted support have no diamond, and counts below rank axes have no probability ordinate. Fits have finite upper endpoints and are sensitive to threshold and reference deletion. No confidence interval is inferred from these dependent pairs. Empirical counts, candidate scores and consensus ranks are unchanged.}'''+main[end:]
old=main[main.index('The rank-1 score exceeds all'):main.index('\n\\section*{Discussion}')].strip()
new=r'''The rank-1 score exceeds all 4{,}005 reference background scores. Ranks 2--9 each have one background exceedance and rank~10 has two, giving empirical conditional FPP values of 0.0250\% and 0.0499\%, respectively (Fig.~\ref{fig:background-diagnostics}f). A continuous GPD fit distinguishes ranks 2--9, with exploratory tail estimates from 0.00202\% to 0.14395\%. These values are not monotonic in consensus rank because the background comparison uses mean scores rather than mean ranks.

The tail fit does not resolve the significance of rank~1. Its score of 2.985 exceeds the central fitted endpoint of 2.846. Removing the highest-scoring background pair moves ranks 1--9 outside the fitted support, showing that this extrapolation depends on very few reference pairs. We therefore retain the empirical counts as the primary background result. Waveform, sky and posterior compatibility identify these pairs as candidates for joint Bayesian analysis, but neither the finite background nor its fitted tail establishes a strongly lensed source. The Supplementary Information provides the complete Top-20 comparisons and model diagnostics.'''
assert old.startswith('The rank-1') and len(old)<2000
main=main.replace(old,new)
anchor='Validation sources also informed calibration and model selection, so these reference tails are not an independent calibration of real-catalog probabilities.'
method=r'''To explore a continuous tail mapping, we fitted a generalized Pareto distribution to excess scores above a threshold $u$. For $m$ excesses among $N_{\rm bg}$ reference pairs, the model tail is
\begin{equation}
p_{\rm GPD}(s)=\frac{m}{N_{\rm bg}}\left[1+\xi\frac{s-u}{\sigma}\right]^{-1/\xi},\qquad s\ge u,
\label{eq:gpd-tail}
\end{equation}
within the fitted support $1+\xi(s-u)/\sigma>0$. Here $\sigma>0$ is the scale and $\xi$ is the shape. The limit $\xi=0$ gives an exponential tail. The central threshold is the 95th reference percentile, with four additional thresholds used for sensitivity checks. Negative fitted shapes imply a finite endpoint $u-\sigma/\xi$. Scores at or beyond that endpoint are flagged rather than assigned a reported probability of zero. These exploratory fits do not alter model scores or candidate selection. Fitting, deletion checks and comparisons with existing test backgrounds are detailed in the Supplementary Information.

'''
assert main.count(anchor)==1
main=main.replace(anchor,method+anchor)
edit('main.tex',main)

# Supplement: reproducible diagnostics, no unvalidated confidence intervals.
section=r'''\subsection{Exploratory modelling of the high-score tail}
\label{sec:supp-gpd}

The empirical probability changes only when a threshold crosses an observed reference score. We therefore explored a generalized Pareto tail model to distinguish scores within the same empirical step. This analysis reused the existing mean-score backgrounds without retraining, adding simulations or changing rankings. The 95th-percentile central threshold was fixed in the analysis plan before fitting. Thresholds at the 90th, 92.5th, 97.5th and 98th percentiles assess its sensitivity. They retain 401, 301, 101 and 81 excesses, respectively, compared with 201 at the central threshold.

For each threshold we maximized the marginal GPD log likelihood of the excesses with location fixed at zero. Three starting shape parameters ($-0.2$, 0 and 0.2) were used, retaining the solution with the largest likelihood. An exponential tail with scale equal to the mean excess provides a shape-zero comparison. The 30 fits across three runs and five thresholds used SciPy 1.17.1. Pairwise dependence is not modelled by this likelihood, so it provides point fits rather than independent-pair uncertainty estimates. All fit parameters and candidate predictions are supplied as Source Data.

All central GPD fits have negative shape parameters (Table~\ref{tab:gpd-diagnostics}), which impose finite upper score endpoints. The test background exceeds these endpoints in three O3 pairs and two O4a pairs. These observations show that the fitted endpoints cannot be treated as established limits of the unrelated population. For O4b, rank~1 exceeds the endpoints at all five thresholds, even though its gap from the 98th-percentile endpoint is only about 0.0013 in score (Fig.~\ref{fig:supp-tail-endpoints}). No probability is reported for a score outside fitted support.

Deletion diagnostics refit the central model after removing each of the 90 reference sources and after removing all events associated with each of the six noise parents. Noise-parent deletion was also repeated at all five thresholds, giving 360 deletion fits in total across the three runs. A separate diagnostic removes the single largest reference pair per run. In O4b, deleting that pair moves ranks 1--9 beyond the refitted endpoint. Every rank from 2 to 9 also falls outside support in at least one central source- or noise-deletion fit. These checks demonstrate sensitivity to the sparse upper tail. Their ranges are not confidence intervals.

The existing test background provides a further check using all 4{,}005 pairs of its 90 isolated sources. At nominal model tail probabilities of 1\% and 0.5\%, the O4b central fit predicts 40.05 and 20.025 exceedances, compared with 39 and 21 observed. At 0.025\%, it predicts about one and observes none. Corresponding counts for all runs are provided in Table~\ref{tab:gpd-heldout}. Agreement at moderate thresholds does not establish the accuracy of smaller tail probabilities. These test catalogs had already been examined and do not provide new blind confirmation.

The exponential comparison gives a positive estimate for O4b rank~1, but its descriptive tail discrepancy is larger than that of the central GPD fit (Kolmogorov--Smirnov distances 0.146 and 0.074, respectively). These distances are fit diagnostics, not calibrated goodness-of-fit tests for dependent pairs. We did not choose an unbounded model merely to populate a zero-count candidate. The GPD values in Tables~\ref{tab:gpd-O3}--\ref{tab:gpd-O4b} are exploratory estimates alongside the empirical counts, not validated real-catalog FPP, annual FAR or lensing probabilities.

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{figures/supp_tail_endpoint_diagnostics.pdf}
\caption{\textbf{Sensitivity of the fitted upper score endpoint.} GPD fits above five background percentiles have negative shapes in all three runs. Orange points show their fitted endpoints, with connecting lines as visual guides. Purple dashed lines show the real rank-1 score and teal dotted lines show the maximum score among 4{,}005 existing test singleton pairs. A score above the endpoint is outside the fitted support, not evidence for zero background probability. Points are deterministic fits to the same reference set, not independent repetitions or confidence limits.}
\label{fig:supp-tail-endpoints}
\end{figure}

\input{tables/c_current/tail_diagnostics.tex}
\input{tables/c_current/tail_candidates.tex}
\FloatBarrier

'''
supp=(WT/'supplementary.tex').read_text(encoding='utf-8')
anchor=r'\section{Runtime measurements on a common O3 sample}'
assert supp.count(anchor)==1
supp=supp.replace(anchor,section+anchor)
edit('supplementary.tex',supp)

diag=[r'\begin{table}[p]',r'\centering',r'\caption{\textbf{Central GPD fits and finite-support checks.} Fits use 201 excesses above the 95th percentile of each 4{,}005-pair validation background. The last column counts existing test singleton pairs above the fitted endpoint. Parameters are point estimates, not uncertainty bounds.}',r'\label{tab:gpd-diagnostics}',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Run & Threshold $u$ & Shape $\xi$ & Scale $\sigma$ & Endpoint & Test pairs beyond \\',r'\midrule']
for run in RUNS:
    f=Q95.loc[run];v=V.query('run==@run and q==0.95').iloc[0]
    diag.append(f'{run} & {f.u:.4f} & {f.xi:.4f} & {f.scale:.4f} & {f.endpoint:.4f} & {int(v.test_singleton_pairs_outside_support)}'+r' \\')
diag += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[p]',r'\centering',r'\caption{\textbf{Existing test-background checks at central GPD thresholds.} Entries are observed counts among 4{,}005 test singleton pairs. Expected counts are the nominal model probabilities multiplied by 4{,}005. These are previously examined test data, not new independent confirmation.}',r'\label{tab:gpd-heldout}',r'\begin{tabular}{lrrrrr}',r'\toprule',r' & 1\% & 0.5\% & 0.1\% & 0.05\% & 0.025\% \\',r'\midrule',r'Expected & 40.05 & 20.03 & 4.01 & 2.00 & 1.00 \\']
for run in RUNS:
    h=H.query('run==@run and model=="GPD" and q==0.95 and check_set=="test_singletons"').sort_values('target_tail_probability',ascending=False)
    diag.append(run+' & '+' & '.join(str(int(x)) for x in h.observed_exceedances)+r' \\')
diag += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
edit('tables/c_current/tail_diagnostics.tex','\n'.join(diag)+'\n')
tables=[]
for run in RUNS:
    tables += [r'\begin{table}[p]',r'\centering',r'\caption{\textbf{'+run+r' Top-20 empirical and exploratory model tails.} Ranks refer to the frozen candidate list. Empirical FPP retains background counts. GPD values and threshold ranges are percentages, not confidence intervals. Ranges include only fits whose support contains the score. The last column counts thresholds for which the score is outside support, out of five. A dagger denotes no reported probability because the score is outside fitted support.}',r'\label{tab:gpd-'+run+'}',r'\fontsize{8}{10}\selectfont',r'\begin{tabular}{rrrrrr}',r'\toprule',r'Rank & $S$ & Empirical FPP (count) & GPD (\%) & Threshold range (\%) & Outside / 5 \\',r'\midrule']
    for _,r in pred(run).iterrows():
        qs=P[(P.run==run)&(P.model=='GPD')&(P['rank']==r['rank'])]
        valid=qs.loc[~qs.outside_fitted_support,'estimated_tail_probability']
        model=r'---$^\dagger$' if r.outside_fitted_support else probtxt(r.estimated_tail_probability)
        ran=r'---$^\dagger$' if len(valid)==0 else probtxt(valid.min())+'--'+probtxt(valid.max())
        tables.append(f'{int(r["rank"])} & {r.S:.3f} & '+empirical(int(r.empirical_count))+' & '+model+' & '+ran+f' & {int(qs.outside_fitted_support.sum())}/5'+r' \\')
    tables += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
edit('tables/c_current/tail_candidates.tex','\n'.join(tables)+'\n')
edit('source_data/c_current/tail_model/README.md',"# Exploratory GPD tail modelling\n\nThe files retain all threshold, candidate and deletion results, including failures. Probabilities equal to zero in raw fit outputs can indicate a score outside a negative-shape GPD's finite support. Use outside_fitted_support and the endpoint fields; do not interpret those zeros as measured risk. Threshold/deletion ranges are sensitivity diagnostics, not confidence intervals. The underlying scores, empirical counts, rankings and model fits of TriLens are unchanged. Candidate tables report probabilities in percent; CSV probability fields are fractions. No annual FAR is inferred.\n")

checks=[]
for file in OUT.glob('*.pdf'):
    with fitz.open(file) as doc:
        for page in doc:
            spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
            assert min(s['size'] for s in spans)>=5
            assert not [s for s in spans if s['bbox'][0]<-.5 or s['bbox'][1]<-.5 or s['bbox'][2]>page.rect.width+.5 or s['bbox'][3]>page.rect.height+.5]
        checks.append({'file':file.name,'pages':len(doc),'minimum_text_pt':min(s['size'] for s in spans)})
(BASE/'FIGURE_QA.json').write_text(json.dumps({'pdf_checks':checks,'backend':'Python','intervals':'None; fixed point fits with sensitivity in SI','rank_changes':False,'finite_support_zeros_not_plotted':True},indent=2),encoding='utf-8')
print(json.dumps('*** Begin Patch\n'+''.join(patches)+'*** End Patch',ensure_ascii=False))
