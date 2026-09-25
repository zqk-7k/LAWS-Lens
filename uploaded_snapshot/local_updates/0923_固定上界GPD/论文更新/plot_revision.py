from pathlib import Path
import json, math, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WT=PROJECT/'0909结果/论文Methods更新_20260911/overleaf_worktree'
TAB=BASE.parent/'tables'
RAW=PROJECT/'0922_GWLR_TAIL02诊断/extracted/gwlr_tail02_20260922T113500Z_r2'
INJ=PROJECT/'0922_FPP结果/extracted/gwlr_fpp_injection_real_01_20260922T061000Z_r2'
OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
F=pd.read_csv(TAB/'fits.csv');P=pd.read_csv(TAB/'candidate_Top50_all_bounds.csv')
RUNS=['O3','O4a','O4b'];PURPLE='#7756A3';TEAL='#1B8A78';ORANGE='#B65C39';GRAY='#737373'
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8.5,'axes.labelweight':'bold','axes.titlesize':9,'axes.titleweight':'bold','xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none'})
def count(bg,s):return len(bg)-np.searchsorted(np.sort(bg),np.asarray(s),side='left')
def survival(f,s):
    s=np.asarray(s);assert np.all((s>=f.u)&(s<f.B))
    return f.tail_mass*((f.B-s)/(f.B-f.u))**f.a
def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=300)
    fig.savefig(OUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})

fig=plt.figure(figsize=(7.2047244094,7.0866141732))
grid=fig.add_gridspec(2,3,left=.105,right=.98,bottom=.23,top=.83,wspace=.34,hspace=.52)
upper=[];lower=[];source=[]
for c,run in enumerate(RUNS):
    bg=pd.read_csv(RAW/f'inputs/{run}_fit_scores.csv').mean_S.to_numpy()
    inj=pd.read_parquet(INJ/f'tables/{run}/injection_mean_S_FPP.parquet')
    r=P[(P.run==run)&(P.B==4)&(P['rank']<=20)].sort_values('rank')
    f=F[(F.run==run)&(F.B==4)].iloc[0]
    assert len(bg)==4005 and len(r)==20
    np.testing.assert_array_equal(count(bg,r.S),r.empirical_count)
    np.testing.assert_allclose(survival(f,r.S),r.estimated_tail_probability)
    ax=fig.add_subplot(grid[0,c]);upper.append(ax)
    x=np.unique(bg);p=count(bg,x)/4005
    ax.step(x,p,where='pre',color=PURPLE,lw=1.35)
    sx=np.linspace(f.u,f.B-(f.B-f.u)*(5e-6/f.tail_mass)**(1/f.a),350)
    py=survival(f,sx);assert np.all(py>0) and np.all(np.diff(py)<=0)
    ax.plot(sx,py,color=ORANGE,ls='--',lw=1.2)
    source.extend({'run':run,'series':'Fixed_endpoint_4','S':float(a),'probability':float(b)} for a,b in zip(sx,py))
    source.extend({'run':run,'series':'Empirical','S':float(a),'probability':float(b)} for a,b in zip(x,p))
    true=np.sort(inj.loc[inj.is_true_pair,'final_score_POSITIVE'])[::-1]
    for q,m in [(.5,'o'),(.9,'s')]:
        t=true[math.ceil(q*len(true))-1];p0=count(bg,[t])[0]/4005
        ax.plot(t,p0,m,color=PURPLE,ms=4.2,mec='white',mew=.5,zorder=4)
        source.append(dict(run=run,series=f'Recall_{q}',S=float(t),probability=float(p0)))
    ax.set_xscale('asinh',linear_width=2)
    ax.set_xlim(2*np.sinh(np.arcsinh(x.min()/2)-.07),4.5)
    ticks=[v for v in [-300,-30,-10,-3,0,3] if ax.get_xlim()[0]<v<ax.get_xlim()[1]]
    ax.set_xticks(ticks,[str(v) for v in ticks]);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yscale('log');ax.set_ylim(5e-6,1.45)
    ax.set_yticks([.00001,.0001,.001,.01,.1,1],['0.001','0.01','0.1','1','10','100'])
    ax.set_title(run,pad=8);ax.set_xlabel('Score threshold S')
    if c==0:ax.set_ylabel('Background tail (%)')
    ax.text(-.20,1.05,chr(97+c),transform=ax.transAxes,fontsize=10,fontweight='bold')
    ax=fig.add_subplot(grid[1,c]);lower.append(ax)
    k=r.empirical_count.to_numpy();ranks=r['rank'].to_numpy()
    for mask,filled in [((k>0)&(k<20),False),(k>=20,True)]:
        ax.scatter(ranks[mask],k[mask]/4005,s=19,marker='o',facecolors=TEAL if filled else 'white',edgecolors=TEAL,lw=.9,zorder=4)
    assert np.all(r.estimated_tail_probability>5e-6)
    ax.scatter(ranks,r.estimated_tail_probability,s=14,marker='D',facecolors='none',edgecolors=ORANGE,lw=.8,zorder=3)
    ax.set_yscale('log');ax.set_ylim(5e-6,.025)
    ax.set_yticks([.00001,.0001,.001,.01],['0.001','0.01','0.1','1'])
    ax.set_xlim(.3,20.7);ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('Candidate rank',labelpad=47)
    if c==0:ax.set_ylabel('Background tail (%)')
    ax.text(-.20,1.05,chr(100+c),transform=ax.transAxes,fontsize=10,fontweight='bold')
    for rank in ranks[k==0]:ax.text(rank,-.16,'0/4005',transform=ax.get_xaxis_transform(),rotation=90,rotation_mode='anchor',ha='right',va='center',fontsize=6,clip_on=False)
for ax in upper+lower:
    ax.axhline(.01,color=GRAY,ls='--',lw=.65)
    ax.axhline(1/4005,color='#BBBBBB',ls=':',lw=.65)
    ax.yaxis.set_minor_locator(NullLocator())
handles=[Line2D([],[],color=PURPLE,lw=1.3,label='Empirical background'),Line2D([],[],color=ORANGE,ls='--',lw=1.2,label='GPD model (upper bound 4)'),Line2D([],[],color=GRAY,ls='--',lw=.7,label='1% threshold'),Line2D([],[],color=PURPLE,marker='o',ls='none',ms=4,label='50% pair recall'),Line2D([],[],color=PURPLE,marker='s',ls='none',ms=4,label='90% pair recall')]
leg1=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.955),ncol=3,columnspacing=1.1,handlelength=1.6)
fig.suptitle('0/4005 indicates that no background pair reached the candidate score',fontsize=8.5,fontweight='bold',y=.995)
leg2=fig.legend(handles=[Line2D([],[],color=TEAL,marker='o',mfc='white',ls='none',ms=4,label='Empirical: 1–19 counts'),Line2D([],[],color=TEAL,marker='o',ls='none',ms=4,label='Empirical: ≥20 counts'),Line2D([],[],color=ORANGE,marker='D',mfc='none',ls='none',ms=4,label='GPD model')],loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,columnspacing=1.2)
fig.canvas.draw();ren=fig.canvas.get_renderer()
assert leg1.get_window_extent(ren).y0>max(a.get_tightbbox(ren).y1 for a in upper)
assert leg2.get_window_extent(ren).y1<min(a.get_tightbbox(ren).y0 for a in lower)
save(fig,'Fig4_fixed_endpoint4');plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(7.2047244094,3.2283464567))
fig.subplots_adjust(left=.105,right=.98,bottom=.22,top=.84,wspace=.39)
for c,(ax,run) in enumerate(zip(axs,RUNS)):
    r=P[(P.run==run)&(P['rank']==1)].sort_values('B')
    assert len(r)==3
    ax.plot(r.B,r.estimated_tail_percent,'o-',color=ORANGE,ms=4,lw=1.2)
    ax.set_yscale('log');ax.set_xticks([4,4.5,5]);ax.set_xlim(3.88,5.12)
    ax.set_title(run);ax.set_xlabel('Fixed upper bound')
    if c==0:ax.set_ylabel('Rank 1 model FPP (%)')
    ax.text(-.20,1.07,chr(97+c),transform=ax.transAxes,fontweight='bold',fontsize=10)
    lo=r.estimated_tail_percent.min();hi=r.estimated_tail_percent.max()
    ax.set_ylim(lo/1.4,hi*1.4)
    ax.set_yticks(r.estimated_tail_percent,[f'{v:.5f}' for v in r.estimated_tail_percent])
    ax.yaxis.set_minor_locator(NullLocator())
save(fig,'Supp_fixed_endpoint_sensitivity');plt.close(fig)

sd=WT/'source_data/c_current/fixed_endpoint_gpd';sd.mkdir(parents=True,exist_ok=True)
for p in TAB.glob('*.csv'):shutil.copy2(p,sd/p.name)
pd.DataFrame(source).to_csv(sd/'figure4_curves.csv',index=False)
for name in ['gpd_fits.csv','HELDOUT_ENDPOINT_DIAGNOSTIC.csv','gates.csv']:
    shutil.copy2(RAW/'results'/name,sd/('tail02_'+name))
for src,tgt in [('Fig4_fixed_endpoint4.pdf','fig_background_diagnostics.pdf'),('Supp_fixed_endpoint_sensitivity.pdf','supp_tail_endpoint_diagnostics.pdf')]:
    target=WT/'figures'/tgt
    backup=BASE/'before';backup.mkdir(exist_ok=True)
    if not (backup/tgt).exists():shutil.copy2(target,backup/tgt)
    shutil.copy2(OUT/src,target)
print('Figures and source tables updated; scores and ranks unchanged.')
