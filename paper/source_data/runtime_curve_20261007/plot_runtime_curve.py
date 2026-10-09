"""Plot initial processing and cached pair-scoring times for five O3 sample sizes.

Points give medians and bars span the technical repetitions: three initial
processing measurements and nine cached calculations at each size.
Export a 183 x 83 mm figure in PDF, SVG, PNG and TIFF formats.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,default=ROOT/'extracted')
parser.add_argument('--output-dir',type=Path,default=ROOT/'figures')
args=parser.parse_args()
DATA=args.data_dir
OUT=args.output_dir; OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((DATA/'DELIVERY_SHA256.json').read_text(encoding='utf-8'))
for item in manifest:
    p=DATA/item['path']
    assert p.stat().st_size==item['bytes'],item['path']
    assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256'],item['path']
summary=pd.read_csv(DATA/'measurement/tables/runtime_summary.csv')
raw=pd.read_csv(DATA/'measurement/tables/scaling_timings.csv')
assert summary.n_events.tolist()==[5,15,30,46,62]
assert (summary.n_pairs==summary.n_events*(summary.n_events-1)//2).all()
assert len(raw)==75
recomputed=[]
for row in summary.itertuples():
    for boundary,field,count in [('PRODUCTS_READY_TOTAL','first_pass',3),
                                  ('EVENT_CACHE_PAIR_RANK','cached_pair',9)]:
        values=raw.loc[(raw.n_events==row.n_events)&(raw.boundary==boundary),'wall_s'].to_numpy()
        assert len(values)==count and np.isfinite(values).all() and (values>0).all()
        assert raw.loc[(raw.n_events==row.n_events)&(raw.boundary==boundary),'n_pairs'].eq(row.n_pairs).all()
        for label,value in [('median',np.median(values)),('min',values.min()),('max',values.max())]:
            assert np.isclose(value,getattr(row,f'{field}_{label}_s'),atol=1e-12,rtol=1e-12)
        recomputed.append({'n_events':row.n_events,'n_pairs':row.n_pairs,
            'boundary':boundary,'n_repeats':count,'median_s':np.median(values),
            'min_s':values.min(),'max_s':values.max()})
pd.DataFrame(recomputed).to_csv(OUT/'recomputed_plot_data.csv',index=False)
checks=json.loads((DATA/'measurement/results/checks.json').read_text())
status=json.loads((DATA/'measurement/contracts/FINAL_STATUS.json').read_text())
assert status['checks_passed']==status['checks_total']==len(checks)==540

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
    'legend.fontsize':8,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(7.2047244094,3.2677165354))
fig.subplots_adjust(left=.085,right=.97,bottom=.18,top=.84,wspace=.35)
color='#147967'
for i,(ax,field,ylabel) in enumerate(zip(axes,['first_pass','cached_pair'],
           ['Initial processing time (s)','Pair calculation time (s)'])):
    x=summary.n_pairs.to_numpy()
    y=summary[f'{field}_median_s'].to_numpy()
    lo=summary[f'{field}_min_s'].to_numpy(); hi=summary[f'{field}_max_s'].to_numpy()
    assert np.all(x>0) and np.all(y>0) and np.all(lo>0)
    ax.plot(x,y,color=color,marker='o',ms=4.3,lw=1.4,label='LAWS-Lens',zorder=3)
    ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt='none',ecolor=color,
                capsize=2.5,elinewidth=.85,zorder=2)
    ax.annotate(f'{y[-1]:.3f} s' if i==0 else f'{y[-1]:.4f} s',
                (x[-1],y[-1]),xytext=(.98,.54) if i==0 else (-10,-25),
                textcoords=ax.transAxes if i==0 else 'offset points',
                ha='right',va='center',fontsize=8,color='black',
                arrowprops=dict(arrowstyle='-',lw=.6,color='black',shrinkA=2,shrinkB=5))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(7,2800); ax.set_ylim((2,160) if i==0 else (.012,1.8))
    ax.set_xticks([10,100,1000],['10','100','1,000'])
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2,5,10,20,50,100] if i==0 else [.02,.05,.1,.2,.5,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(direction='out',length=3,width=.7)
    ax.tick_params(which='minor',length=1.7,width=.5)
    ax.grid(axis='y',which='major',color='#E7E9EC',linewidth=.55,zorder=0)
    ax.set_xlabel('Number of event pairs',labelpad=5)
    ax.set_ylabel(ylabel,labelpad=5)
    ax.text(-.17,1.015,'ab'[i],transform=ax.transAxes,fontsize=10,fontweight='bold')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',
           bbox_to_anchor=(.53,.98),handlelength=2.2,frameon=False)
fig.savefig(OUT/'fig_runtime_scaling.pdf',facecolor='white')
fig.savefig(OUT/'fig_runtime_scaling.svg',facecolor='white')
fig.savefig(OUT/'fig_runtime_scaling.png',dpi=300,facecolor='white')
fig.savefig(OUT/'fig_runtime_scaling.tiff',dpi=600,facecolor='white')
plt.close(fig)
audit={'manifest_files_verified':len(manifest),'raw_timing_rows':len(raw),
       'plot_points_per_panel':5,'event_counts':summary.n_events.tolist(),
       'recomputed_summary_matches':True,'first_repeats_per_size':3,'cached_repeats_per_size':9,
       'error_bars':'observed minimum and maximum','old_measurements_used':False,
       'server_checks_recorded':len(checks),'inference_rerun_locally':False,
       'measurement_implementation':'DOMAIN04/PATH1','paper_modified':False}
(ROOT/'LOCAL_VERIFICATION.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit))
