"""Review-only quantitative figures. Style-only inheritance; no annualization.

All 4005 background pairs enter every tail count. All requested Top-20 rows
are displayed. Mean-score tails are not means of per-model probabilities.
Zero counts are labels below rank ticks, not probability ordinates.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import findfont
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
import fitz

BASE=Path(__file__).resolve().parent
ROOT=next((BASE.parent/'extracted').iterdir())
OUT=BASE/'figures';SRC=BASE/'source_data';TAB=BASE/'tables'
for p in (OUT,SRC,TAB):p.mkdir(exist_ok=True)
RUNS=['O3','O4a','O4b'];PURPLE='#7756A3';TEAL='#1B8A78';ORANGE='#B65C39';GRAY='#737373'
assert Path(findfont('Arial',fallback_to_default=False)).exists()
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8.5,
 'axes.labelweight':'bold','axes.titlesize':9,'axes.titleweight':'bold',
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'legend.frameon':False,'legend.fontsize':7.5,
 'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'custom',
 'mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
def read(p):return pd.read_csv(p)
def pct(x,d=4):return f'{float(x)*100:.{d}f}%'
def fmt(x,d=3):return 'NA' if pd.isna(x) else f'{float(x):.{d}f}'
def flag(v):return str(v).lower() in ['true','1','1.0']
def tail(bg,s):return len(bg)-np.searchsorted(np.sort(bg),np.asarray(s),side='left')
sources=[];audit_panels=[];DATA={};OPS=[];CURVES=[];CANDS=[]
for run in RUNS:
    paths=[ROOT/f'background/{run}/mean_S_calibration_pairs.csv',ROOT/f'tables/{run}/injection_mean_S_FPP.parquet',ROOT/f'tables/{run}/real_GWTC_Top20_FPP.csv']
    for p in paths:
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        sources.append({'file':str(p),'sha256':h})
    bg=read(paths[0]);inj=pd.read_parquet(paths[1]);real=read(paths[2]).sort_values('consensus_rank')
    assert len(bg)==4005 and len(inj)==101025 and len(real)==20
    assert real.consensus_rank.tolist()==list(range(1,21))
    for frame in [inj,real]:
        k=tail(bg.final_score_POSITIVE,frame.final_score_POSITIVE)
        np.testing.assert_array_equal(k,frame.background_exceedances)
        np.testing.assert_allclose(k/4005,frame.conditional_FPP,atol=1e-15)
    # Every distinct observed threshold retained; count ties using >=.
    unique=np.unique(bg.final_score_POSITIVE)
    curve=pd.DataFrame({'run':run,'threshold_S':unique,'exceedances':tail(bg.final_score_POSITIVE,unique)})
    curve['conditional_FPP']=curve.exceedances/4005
    assert (np.diff(curve.threshold_S)>0).all() and (np.diff(curve.conditional_FPP)<=0).all()
    assert (curve.conditional_FPP>0).all(), 'Log curves stop at the largest observed background score'
    CURVES.append(curve)
    true=np.sort(inj.loc[inj.is_true_pair,'final_score_POSITIVE'].to_numpy())[::-1]
    for q in [.5,.9]:
        threshold=true[math.ceil(q*len(true))-1]
        OPS.append({'run':run,'recovery_target':q,'threshold_S':threshold,'true_retained':int((true>=threshold).sum()),'conditional_FPP':int(tail(bg.final_score_POSITIVE,[threshold])[0])/4005})
    CANDS.append(real)
    DATA[run]=(bg,inj,real,curve)
pd.concat(CURVES).to_csv(SRC/'score_FPP_curves.csv',index=False)
pd.DataFrame(OPS).to_csv(SRC/'recovery_operating_points.csv',index=False)
pd.concat(CANDS).to_csv(SRC/'candidate_top20_all_runs.csv',index=False)
op=pd.DataFrame(OPS)

fig=plt.figure(figsize=(7.2047244094,6.6929133858))
grid=fig.add_gridspec(2,3,left=.102,right=.985,bottom=.225,top=.855,wspace=.32,hspace=.51,height_ratios=[1,1.04])
upper=[];lower=[]
for col,run in enumerate(RUNS):
    bg,inj,real,curve=DATA[run]
    ax=fig.add_subplot(grid[0,col]);upper.append(ax)
    ax.step(curve.threshold_S,curve.conditional_FPP,where='pre',color=PURPLE,lw=1.35)
    for q,m in [(.5,'o'),(.9,'s')]:
        p=op.query('run==@run and recovery_target==@q').iloc[0]
        assert p.conditional_FPP>0
        ax.plot(p.threshold_S,p.conditional_FPP,m,color=PURPLE,ms=4.4,mec='white',mew=.5,zorder=4)
    ax.axhline(.01,color=GRAY,ls='--',lw=.8)
    ax.axhline(1/4005,color='#BBBBBB',ls=':',lw=.65)
    ax.set_xscale('asinh',linear_width=2)
    lo,hi=curve.threshold_S.min(),curve.threshold_S.max()
    ax.set_xlim(2*np.sinh(np.arcsinh(lo/2)-.07),2*np.sinh(np.arcsinh(hi/2)+.14))
    xt=[v for v in [-300,-30,-10,-3,0,3,30] if ax.get_xlim()[0]<v<ax.get_xlim()[1]]
    ax.set_xticks(xt,[str(v) for v in xt]);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yscale('log');ax.set_ylim(.00018,1.45)
    ax.set_yticks([.001,.01,.1,1],['0.1','1','10','100'])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Score threshold S');ax.set_title(run,pad=8)
    if col==0:ax.set_ylabel('Conditional FPP (%)')
    ax.text(-.18,1.05,chr(97+col),transform=ax.transAxes,fontsize=10,fontweight='bold')
    ax=fig.add_subplot(grid[1,col]);lower.append(ax)
    k=real.background_exceedances.to_numpy();p=real.conditional_FPP.to_numpy();ranks=real.consensus_rank.to_numpy()
    for mask,filled in [((k>0)&(k<20),False),(k>=20,True)]:
        ax.scatter(ranks[mask],p[mask],s=17,marker='o',facecolors=TEAL if filled else 'white',edgecolors=TEAL,linewidths=.9,zorder=3)
    ax.axhline(.01,color=GRAY,ls='--',lw=.8)
    ax.axhline(1/4005,color='#BBBBBB',ls=':',lw=.65)
    ax.set_yscale('log');ax.set_ylim(.00018,.025)
    ax.set_yticks([.0003,.001,.003,.01],['0.03','0.1','0.3','1'])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(.3,20.7);ax.set_xticks([1,5,10,15,20])
    ax.set_xlabel('Candidate rank',labelpad=48)
    if col==0:ax.set_ylabel('Conditional FPP (%)')
    ax.text(-.18,1.06,chr(100+col),transform=ax.transAxes,fontsize=10,fontweight='bold')
    zero_labels=[]
    for rank in ranks[k==0]:
        zero_labels.append(ax.text(rank,-.16,'0/4005',transform=ax.get_xaxis_transform(),
            rotation=90,rotation_mode='anchor',ha='right',va='center',fontsize=6.0,color='#333333',clip_on=False))
    fig.canvas.draw()
    ren=fig.canvas.get_renderer()
    for first,second in zip(zero_labels,zero_labels[1:]):
        assert not first.get_window_extent(ren).overlaps(second.get_window_extent(ren))
    audit_panels.extend([{'panel':chr(97+col),'run':run,'quantity':'Mean-score empirical tail','background_pairs':4005,'uncertainty':'No CI; deterministic empirical function of shared finite background'},
                         {'panel':chr(100+col),'run':run,'quantity':'All frozen Top20 mean-score tails','rows':20,'zero_label_count':int((k==0).sum()),'low_nonzero_count':int(((k>0)&(k<20)).sum()),'uncertainty':'Tail counts encoded; no iid error bars; zero labels are outside probability coordinates'}])
top_handles=[Line2D([],[],color=PURPLE,lw=1.4,label='Mean-score background'),Line2D([],[],color=PURPLE,marker='o',ls='none',ms=4,label='50% pair recall'),Line2D([],[],color=PURPLE,marker='s',ls='none',ms=4,label='90% pair recall'),Line2D([],[],color=GRAY,ls='--',lw=.8,label='1% threshold')]
legend1=fig.legend(handles=top_handles,loc='upper center',bbox_to_anchor=(.53,.95),ncol=4,columnspacing=1.1,handlelength=1.6)
fig.suptitle('0/4005 indicates that no background pair reached the candidate score',
             fontsize=8.5,fontweight='bold',y=.99)
bottom_handles=[Line2D([],[],color=TEAL,marker='o',mfc='white',ls='none',ms=4,label='1–19 exceedances'),Line2D([],[],color=TEAL,marker='o',ls='none',ms=4,label='20 or more')]
legend2=fig.legend(handles=bottom_handles,loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,columnspacing=1.5,handlelength=1.2)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
assert legend1.get_window_extent(renderer).y0>max(a.get_tightbbox(renderer).y1 for a in upper)
assert legend2.get_window_extent(renderer).y1<min(a.get_tightbbox(renderer).y0 for a in lower)
fig.savefig(OUT/'Fig4_FPP_final.pdf')
fig.savefig(OUT/'Fig4_FPP_final.svg')
fig.savefig(OUT/'Fig4_FPP_final.png',dpi=600)
fig.savefig(OUT/'Fig4_FPP_final.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)

# Table 1: the operating point is computed from mean scores, not model SD.
eff=read(ROOT/'summaries/injection_efficiency_at_FPP.csv')
eff=eff.query('model=="mean_S" and FPP_threshold==0.01').copy()
eff.to_csv(SRC/'efficiency_at_one_percent.csv',index=False)
fig,ax=plt.subplots(figsize=(183/25.4,58/25.4));ax.axis('off')
rows=[[r.run,f'{int(r.true_pairs_retained)}/180',pct(r.true_pair_recovery_fraction,2),f'{int(r.false_pairs_retained):,}',pct(r.noncompanion_tail_fraction,4),pct(r.test_singleton_tail_fraction,4)] for _,r in eff.iterrows()]
headers=['Run','Lensed pairs\nretained','Pair recall','Unrelated pairs\nretained','Test unrelated\ntail fraction','Test singleton\ntail fraction']
tbl=ax.table(cellText=rows,colLabels=headers,cellLoc='center',colLoc='center',bbox=[0,.32,1,.63],colWidths=[.07,.18,.15,.2,.2,.2])
tbl.auto_set_font_size(False);tbl.set_fontsize(8)
for (r,c),cell in tbl.get_celld().items():
    cell.set_edgecolor('#D3D3D3');cell.set_linewidth(.4)
    if r==0:cell.set_facecolor('#EEEAF3');cell.get_text().set_weight('bold')
fig.text(.035,.12,'Condition: mean-score FPP < 1%. Test: 180 lensed pairs and 100,845 unrelated pairs per run.',fontsize=7.4)
fig.text(.035,.06,'Singleton-only test check: 4,005 pairs per run. Pair recall is not query Recall@10.',fontsize=7.4)
fig.subplots_adjust(left=.035,right=.985,bottom=.04,top=.99)
for ext in ['pdf','svg','png']:fig.savefig(TAB/f'Table_efficiency_FPP1_review.{ext}',dpi=600)
plt.close(fig)

def official(r,run):
    if run=='O4b':return 'Unavailable','Unavailable;\nno new Hanabi'
    second='official_ml_fpp' if run=='O3' else 'official_phazap_fpp'
    nums=f'{r.official_po_fpp:.5f}\n{r[second]:.5f}'
    st=('Pass' if flag(r.official_frontend) else 'No pass')+' 1% screen\n'
    st+='Hanabi matched*' if flag(r.public_Hanabi_overlap) else 'No public match'
    return nums,st
pages=[]
with PdfPages(TAB/'Tables_candidates_Top20_final.pdf') as pdf:
    for run in RUNS:
        real=DATA[run][2];present=[]
        for _,r in real.iterrows():
            nums,st=official(r,run)
            present.append([int(r.consensus_rank),r.event_i+'\n'+r.event_j,fmt(r.score_mean), '\n'.join(f'{r[x]:+.3f}' for x in ['wf_contribution_mean','time_contribution_mean','sky_contribution_mean']),f'{r.BC_Mc:.3f} / {r.BC_q:.3f}\n{r.BC_chi_eff:.3f} / {r.BC_apparent_distance:.3f}',fmt(r.Dmax),f'{int(r.background_exceedances)}/4005','—' if r.zero_exceedance_unresolved else pct(r.conditional_FPP),nums,st])
        for start in [0,10]:
            fig,ax=plt.subplots(figsize=(297/25.4,210/25.4));ax.axis('off')
            fig.subplots_adjust(left=.028,right=.98,bottom=.09,top=.89)
            h=['Rank','Candidate pair','S','wf / time / sky','BC: $M_c$ / $q$\n'+r'$\chi_{\rm eff}$ / $d_L^{\rm app}$','Dmax','Background\nexceedances','TriLens\nconditional FPP','Official FPP\nPO / '+('ML' if run=='O3' else 'Phazap' if run=='O4a' else 'NA'),'Published\nfollow-up']
            widths=[.045,.18,.06,.095,.125,.06,.085,.10,.09,.16];widths=np.asarray(widths)/sum(widths)
            table=ax.table(cellText=present[start:start+10],colLabels=h,cellLoc='center',colLoc='center',colWidths=widths,bbox=[0,.08,1,.9])
            table.auto_set_font_size(False);table.set_fontsize(8.0)
            for (row,col),cell in table.get_celld().items():
                cell.set_edgecolor('#D4D4D4');cell.set_linewidth(.35)
                if row==0:cell.set_facecolor('#EEEAF3');cell.get_text().set_weight('bold')
                elif row%2==0:cell.set_facecolor('#FAFAFA')
            fig.text(.029,.949,f'{run} candidate pairs | Ranks {start+1}–{start+10}',fontsize=14,fontweight='bold')
            fig.text(.029,.915,'0/4005 indicates that no background pair reached the candidate score',fontsize=10,color='#333333')
            notes=['S and weighted contributions are model means; candidate order remains the frozen consensus ranking.',
                   'TriLens conditional FPP = k/4005 for nonzero counts. It is not a lensing probability or annual FAR.',
                   'BC: detector-frame chirp mass, mass ratio, effective spin and apparent luminosity distance. Dmax is descriptive.',
                   '* Hanabi matched: a published pair result is available, not a new analysis or lensing confirmation. Official FPPs are fractions.']
            for n,line in enumerate(notes):fig.text(.03,.108-n*.018,line,fontsize=8)
            fig.canvas.draw();ren=fig.canvas.get_renderer()
            for (row,col),cell in table.get_celld().items():
                tb=cell.get_text().get_window_extent(ren);cb=cell.get_window_extent(ren)
                assert tb.width<cb.width-2 and tb.height<cb.height-2,(run,start,row,col,tb.width,cb.width)
            pdf.savefig(fig)
            pagefile=TAB/f'{run}_Top{start+1}_{start+10}_review.png';fig.savefig(pagefile,dpi=300)
            pages.append(str(pagefile.name));plt.close(fig)

# Numeric QA of exported vector pages.
pdfchecks=[]
for p in [OUT/'Fig4_FPP_final.pdf',TAB/'Table_efficiency_FPP1_review.pdf',TAB/'Tables_candidates_Top20_final.pdf']:
    with fitz.open(p) as doc:
        for i,page in enumerate(doc):
            spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
            minfont=min(s['size'] for s in spans)
            outside=[s['text'] for s in spans if s['bbox'][0]<-.2 or s['bbox'][1]<-.2 or s['bbox'][2]>page.rect.width+.2 or s['bbox'][3]>page.rect.height+.2]
            assert not outside and minfont>=5,(p.name,i,outside,minfont)
            pdfchecks.append({'file':p.name,'page':i+1,'min_font_pt':minfont,'fonts':sorted({s['font'] for s in spans}),'outside_page':outside})
qa={'inputs':sources,'panels':audit_panels,'pdf_checks':pdfchecks,'candidate_table_pages':pages,'backend':'Python','plot_size_mm':[183,170],'rank_changes':False,'paper_pushed':False,'zero_policy':'Rank-aligned 0/4005 labels outside probability axes; no upper limit assigned','all_60_candidates_displayed':True}
(BASE/'FIGURE_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'figure':str(OUT/'Fig4_FPP_final.png'),'pdf_pages':len(pdfchecks),'source_hashes':len(sources),'table_pages':len(pages)},ensure_ascii=False))
