"""Data-equivalent replacements for eleven missing plotting entry points.

This is not a recovered author script or a pixel-identical paper reconstruction.
Frozen scores, model selection, candidate ranks and paper files are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

RUNS = ('O3', 'O4a', 'O4b')
METHODS = ('waveform-new', 'time-only', 'sky-only', 'three-channel')
COLORS = ('#257c9b', '#b36820', '#68724c', '#b54363')
LABELS = ('Waveform', 'Time delay', 'Sky position', 'Three-channel score')
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 160})


class Replay:
    def __init__(self, paper, output):
        self.paper, self.output = paper, output
        self.data = paper / 'source_data/c_current'
        self.output.mkdir(parents=True, exist_ok=False)
        self.sources, self.artists, self.checks, self.figures = {}, [], [], []

    def read(self, name, *, relative=None, **kwargs):
        path = self.paper / relative if relative else self.data / name
        self.sources[str(path.relative_to(self.paper))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, **kwargs)

    def record(self, figure, panel, label, x, y, rule):
        xx, yy = np.asarray(x), np.asarray(y, dtype=float)
        self.artists.append({'figure': figure, 'panel': panel, 'label': label,
                             'x': xx.tolist(), 'y': [None if not np.isfinite(v) else float(v) for v in yy],
                             'rule': rule})

    def finish(self, fig, name, meaning):
        fig.savefig(self.output / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(self.output / (name + '.png'), bbox_inches='tight')
        plt.close(fig)
        self.figures.append({'file': name + '.pdf', 'meaning': meaning,
                             'recovered_original_script': False, 'pixel_identity_claimed': False})

    def check(self, name, ok, **values):
        self.checks.append({'name': name, 'passed': bool(ok), **values})
        if not ok:
            raise AssertionError(name)

    def distributions(self):
        name = 'fig2_channel_distributions'
        data = self.read('fig2_density_curves.csv', keep_default_na=False)
        columns = ['waveform_score','time_score','sky_raw_log_bf','final_score_POSITIVE']
        available = data['column'].unique().tolist()
        if set(columns) != set(available):
            raise ValueError(f'Unexpected score columns: {available}')
        groups = [('null','Unrelated pairs','#666666'),('SIS','No subhalos','#287ba4'),('PM','With subhalos','#b35f38')]
        self.check('density groups preserve literal null string', set(data.group)=={x[0] for x in groups})
        fig, axes = plt.subplots(3,4,figsize=(12,6.3),layout='constrained')
        for i, run in enumerate(RUNS):
            for j, col in enumerate(columns):
                ax = axes[i,j]
                for group, label, color in groups:
                    frame = data[(data.run==run)&(data.column==col)&(data.group==group)].sort_values('x')
                    y = frame.density.to_numpy()/frame.density.max()
                    ax.plot(frame.x,y,label=label,color=color,lw=1.3)
                    self.record(name, f'{run}/{col}',group,frame.x,y,'Frozen density / within-curve maximum; asinh x display')
                ax.set_xscale('asinh',linear_width=2)
                ax.set_ylim(0,1.05)
                ax.set_yticks([])
                ax.grid(alpha=.12)
                if i==0:
                    ax.set_title(['Waveform score','Time-delay evidence','Sky-position evidence','Three-channel score'][j])
                if j==0:
                    ax.set_ylabel(run + ' noise')
                if i==2:
                    ax.set_xlabel(['Z_wf','Z_time','Z_sky','S'][j])
        axes[0,0].legend(loc='upper left',fontsize=6)
        self.finish(fig,name,'Unit-peak frozen density curves; not new density fitting')

    def retrieval(self, n):
        name = 'fig_retrieval_o3_o4a_o4b' if n==190 else 'fig_retrieval_native_catalogs'
        roc, recall = self.read('roc_per_model.csv'), self.read('recall_per_model.csv')
        nnull = 17885 if n==190 else 100845
        fig, axes = plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for j,run in enumerate(RUNS):
            for method,color,label in zip(METHODS,COLORS,LABELS):
                r = roc[(roc.run==run)&(roc.n==n)&(roc.method==method)]
                self.check(f'{name}/{run}/{method}/three models',r.seed.nunique()==3)
                avg = r.groupby('fpr').value.mean().dropna()
                avg = avg[avg.index >= 1/nnull]
                axes[0,j].plot(avg.index,avg.values,color=color,label=label)
                self.record(name,run+'/ROC',method,avg.index,avg.values,'Equal model means; NaN/unattainable FPR excluded; no extrapolation below 1/null_count')
                r = recall[(recall.run==run)&(recall.n==n)&(recall.method==method)]
                avg = r.groupby('k').value.mean().dropna()
                axes[1,j].plot(avg.index,avg.values,color=color,label=label)
                self.record(name,run+'/Recall',method,avg.index,avg.values,'Equal means of three model curves; subset means already frozen per model')
            axes[0,j].set(xscale='log',xlim=(1/nnull,1),ylim=(0,1.03),title=run,xlabel='Pair false-positive rate')
            axes[1,j].set(xscale='log',xlim=(1,n-1),ylim=(0,1.03),xlabel='Rank cutoff, K')
            axes[1,j].set_xticks([1,5,10,50,n-1],labels=[1,5,10,50,n-1])
            top=axes[0,j].secondary_xaxis('top',functions=(lambda v:v*nnull,lambda v:v/nnull))
            top.set_xlabel('False-pair count')
            top=axes[1,j].secondary_xaxis('top',functions=(lambda v:100*v/(n-1),lambda v:v*(n-1)/100))
            top.set_xlabel('Catalog searched (%)')
            for ax in axes[:,j]:
                ax.grid(alpha=.15)
        axes[0,0].set_ylabel('True-positive rate')
        axes[1,0].set_ylabel('Recall of lensed counterparts')
        axes[0,1].legend(fontsize=6,loc='lower right')
        self.finish(fig,name,f'Frozen {n}-event curves, averaging models only')

    def attribution_and_topb(self):
        table=self.read('retrieval_metrics_per_seed.csv')
        test=table[(table.split=='test')&(table.arm=='C_PHYSICAL')]
        fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
        name='fig_supp_current_attribution'
        order=['waveform-short','waveform-FRT','waveform-OMC','waveform-new']
        for ax,run in zip(axes,RUNS):
            selected=test[(test.run==run)&test.method.isin(order)]
            avg=selected.groupby('method').macro_r_at_10.mean().reindex(order)
            sd=selected.groupby('method').macro_r_at_10.std().reindex(order)
            ax.bar(range(4),avg,yerr=sd,color=COLORS,capsize=3)
            ax.set(title=run,ylim=(0,1),xticks=range(4),xticklabels=['Short','+R','+M','+J'])
            self.record(name,run,'R@10',range(4),avg,'Three model mean, sample SD error bars; cumulative branches')
            self.record(name,run,'SD',range(4),sd,'Sample standard deviation across three models')
        axes[0].set_ylabel('Waveform-only R@10')
        self.finish(fig,name,'Cumulative waveform ablation, not independent additive causal effects')
        fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
        name='fig_supp_current_topb'
        budgets=[10,20,50,100,200,500]
        for ax,run in zip(axes,RUNS):
            for method,color,label in [(METHODS[0],COLORS[0],LABELS[0]),(METHODS[-1],COLORS[-1],LABELS[-1])]:
                group=test[(test.run==run)&(test.method==method)]
                y=[group[f'top{b}_precision'].mean() for b in budgets]
                ax.plot(budgets,y,'o-',label=label,color=color)
                self.record(name,run,method,budgets,y,'Mean frozen Top-B precision across three models; 450-event catalogs')
            ax.set(xscale='log',title=run,ylim=(0,1),xlabel='Pair budget, B')
            ax.set_xticks([10,50,100,500],labels=[10,50,100,500])
        axes[0].set_ylabel('Precision within Top-B')
        axes[1].legend(fontsize=7)
        self.finish(fig,name,'Fixed candidate-budget precision')
        name='fig_supp_current_seed_stability'
        fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
        for ax,run in zip(axes,RUNS):
            for method,color,label in [(METHODS[0],COLORS[0],LABELS[0]),(METHODS[-1],COLORS[-1],LABELS[-1])]:
                group=test[(test.run==run)&(test.method==method)].sort_values('seed')
                self.check(f'{run}/{method}/seed plot count',len(group)==3)
                ax.plot([1,2,3],group.macro_r_at_10,'o-',color=color,label=label)
                self.record(name,run,method,[1,2,3],group.macro_r_at_10,'One value per frozen model seed, sorted seed ascending')
            ax.set(title=run,ylim=(.5,1),xticks=[1,2,3],xlabel='Model realization')
        axes[0].set_ylabel('450-event R@10')
        axes[1].legend(fontsize=7)
        self.finish(fig,name,'Observed three-model variability, not population confidence intervals')

    def sky(self):
        name='fig_supp_current_sky_audit'
        table=self.read('sky_area_before_after_summary.csv').set_index('run').loc[list(RUNS)]
        events=self.read('sky_area_before_after_events.csv')
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for j,(field,label) in enumerate([('HPD90_coverage','Coverage of nominal 90% region'),('A90_median_deg2','Median 90% area (deg2)')]):
            for k,(prefix,color) in enumerate([('raw','#777777'),('calibrated','#277d91')]):
                y=table[f'{prefix}_{field}'].values
                axes[j].bar(np.arange(3)+(k-.5)*.3,y,width=.28,color=color,label=prefix.capitalize()+' maps')
                self.record(name,label,prefix,RUNS,y,'Frozen summary; coverage is source-weighted, area uses event median')
                if j==1:
                    med=events.groupby('run')[f'{prefix}_A90_deg2'].median().reindex(RUNS).values
                    self.check(f'{prefix} area median recomputation',np.allclose(y,med,rtol=1e-12,atol=1e-9))
            axes[j].set_xticks(range(3),labels=RUNS)
            axes[j].set_ylabel(label)
        axes[0].axhline(.9,color='black',ls='--',lw=.8)
        axes[0].set_ylim(.65,1)
        axes[1].legend(fontsize=7)
        self.finish(fig,name,'Frozen raw/calibrated test-map summary; no temperature reselection')

    def injection(self):
        name='fig_supp_current_injection_design'
        events=self.read('test_events.csv')
        fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for j,run in enumerate(RUNS):
            group=events[events.run==run]
            self.check(run+'/test unique event count',len(group)==450 and group.event_uid.nunique()==450)
            for family,color,label in [('SIS','#287ba4','No subhalos'),('PM','#b35f38','With subhalos'),('unlensed','#777777','Isolated')]:
                subset=group[group.family==family]
                x=np.sort(subset.optimal_network_snr.to_numpy())
                y=np.arange(1,len(x)+1)/len(x)
                axes[0,j].step(x,y,where='post',color=color,label=label)
                self.record(name,run+'/event ECDF',family,x,y,'Unweighted event ECDF of frozen optimal network SNR')
                if family!='unlensed':
                    pairs=subset.groupby('source_uid').optimal_network_snr.agg(['min','max','size'])
                    self.check(run+'/'+family+'/doublets',len(pairs)==90 and pairs['size'].eq(2).all())
                    axes[1,j].scatter(pairs['min'],pairs['max'],s=9,alpha=.65,color=color)
                    self.record(name,run+'/image SNR',family,pairs['min'],pairs['max'],'Per-source min/max of two image SNRs; not independently target-rescaled')
            axes[0,j].set(title=run,xscale='log',ylim=(0,1),xlabel='Optimal network SNR')
            axes[1,j].set(xscale='log',yscale='log',xlabel='Weaker-image optimal SNR')
            for ax in axes[:,j]:
                ax.grid(alpha=.15)
        axes[0,0].set_ylabel('Cumulative event fraction')
        axes[1,0].set_ylabel('Stronger-image optimal SNR')
        axes[0,1].legend(fontsize=7)
        self.finish(fig,name,'C common-system-amplitude injection diagnostics; no new population generation')

    def real(self):
        summary=self.read('real_PE_official_budget_summary.csv')
        name='fig_supp_current_catalog_pe'
        fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
        for ax,run in zip(axes,RUNS):
            pairs=self.read(run+'_real_all.csv').sort_values('consensus_rank')
            if 'method' in pairs:
                pairs=pairs[pairs.method=='three-channel']
            selected=summary[(summary.run==run)&(summary.method=='three-channel')].sort_values('budget')
            for field,label,color in [('Mc_BC_ge05','Mass overlap >= 0.5',COLORS[0]),('Dmax_le3','Dmax <= 3',COLORS[-1])]:
                counts=[]
                for row in selected.itertuples():
                    head=pairs.head(int(row.budget))
                    count=int((head.BC_Mc>=.5).sum()) if field=='Mc_BC_ge05' else int((head.Dmax<=3).sum())
                    counts.append(count)
                self.check(run+'/'+field+'/budget counts',np.array_equal(counts,selected[field].values))
                y=selected[field]/selected.budget
                ax.plot(selected.budget,y,'o-',label=label,color=color)
                self.record(name,run,field,selected.budget,y,'Recomputed count from unchanged consensus rank / candidate budget')
            ax.set(title=run,xscale='log',ylim=(.5,1.02),xlabel='Real-pair budget')
            ax.set_xticks([10,20,50,100],labels=[10,20,50,100])
        axes[0].set_ylabel('Fraction satisfying PE screen')
        axes[1].legend(fontsize=7)
        self.finish(fig,name,'Descriptive PE budget audit; no PE-based reranking')

    def candidate_panels(self):
        name='fig4_current_consensus_pe'
        guide=self.read('fig5_rank1_PE_morphology_guides.csv')
        fig,axes=plt.subplots(2,3,figsize=(14,8),gridspec_kw={'width_ratios':[2.4,1,1.8]},layout='constrained')
        for i,run in enumerate(RUNS[:2]):
            top=self.read(f'fig5_{run}_top10.csv').sort_values('consensus_rank')
            original=self.read(f'{run}_real_all.csv').sort_values('consensus_rank')
            if 'method' in original:
                original=original[original.method=='three-channel']
            self.check(run+'/display rank mapping',top.pair_key.tolist()==original.head(10).pair_key.tolist())
            y=np.arange(10)
            labels=[f'{row.consensus_rank:02d} {row.event_i} / {row.event_j}'+(' *' if row.official_frontend else '') for row in top.itertuples()]
            axes[i,0].barh(y,top.score_mean,color=COLORS[0])
            axes[i,0].set(yticks=y,yticklabels=labels,xlabel='Mean score, S',title=run+' consensus Top-10')
            axes[i,0].invert_yaxis()
            axes[i,0].tick_params(axis='y',labelsize=6)
            self.record(name,run+'/ranking','S',top.consensus_rank,top.score_mean,'Frozen consensus order; star means public screening FPP < 1%')
            bc=top[['BC_Mc','BC_q','BC_chi_eff']].values
            axes[i,1].imshow(bc,aspect='auto',vmin=0,vmax=1,cmap='viridis')
            axes[i,1].set(xticks=[0,1,2],xticklabels=['Mc','q','chi_eff'],yticks=y,yticklabels=range(1,11),title='Marginal PE overlap')
            for r in range(10):
                for c in range(3):
                    axes[i,1].text(c,r,f'{bc[r,c]:.2f}',ha='center',va='center',fontsize=6,color='black' if bc[r,c]>.6 else 'white')
            for c,label in enumerate(['Mc','q','chi_eff']):
                self.record(name,run+'/PE',label,top.consensus_rank,bc[:,c],'Unmodified public PE BC values')
            waves=self.read(f'fig5_{run}_rank1_waveforms.csv') if run=='O4a' else self.read('',relative='source_data/current_waveform_display/fig4_gwtc3_rank1_waveforms.csv')
            times=waves.iloc[:,0].values
            mask=(times>=-.75)&(times<=.05)
            for k,col in enumerate(waves.columns[1:]):
                amplitude=waves[col].values
                scale=max(float(np.max(np.abs(amplitude[mask]))),np.finfo(float).tiny)
                plotted=amplitude[mask]/scale*.35+(3-k)
                axes[i,2].plot(times[mask],plotted,lw=.5,color=COLORS[0 if k<2 else 1])
                self.record(name,run+'/strain',col,times[mask],plotted,'Display-only max normalization in [-0.75,0.05] s and fixed track offsets; no time or phase alignment fitted')
                event=col.split('__')[0]
                gg=guide[event].values
                gt=guide.iloc[:,0].values
                gm=(gt>=-.75)&(gt<=.05)
                gp=gg[gm]/max(float(np.max(np.abs(gg[gm]))),np.finfo(float).tiny)*.3+(3-k)
                axes[i,2].plot(gt[gm],gp,lw=.6,color='#444444',alpha=.65)
                self.record(name,run+'/PE guide',col,gt[gm],gp,'Frozen morphology guide, separately normalized for display, not a new likelihood fit')
            axes[i,2].set(yticks=[3,2,1,0],yticklabels=['1 H1','1 L1','2 H1','2 L1'],xlabel='Time from trigger / guide reference (s)',title='Rank 1 waveforms')
        self.finish(fig,name,'Independent rendering of frozen candidate tables and display traces; display normalization documented')

    def timing(self):
        name='fig_runtime_scaling'
        table=self.read('',relative='source_data/runtime_comparison/runtime_summary.csv')
        raw=self.read('',relative='source_data/runtime_comparison/scaling_timings.csv')
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        for j,(field,label) in enumerate([('first_pass_s','Initial processing time (s)'),('cached_pair_median_s','Pair calculation time (s)')]):
            for method,color in [('TriLens',COLORS[0]),('Phazap',COLORS[1])]:
                group=table[table.method==method].sort_values('n_pairs')
                axes[j].loglog(group.n_pairs,group[field],'o-',color=color,label='LAWS-Lens' if method=='TriLens' else method)
                self.record(name,label,method,group.n_pairs,group[field],'Archived DOMAIN04 measurements, not retimed or converted to same-input equivalence')
                if j==1:
                    for row in group.itertuples():
                        times=raw[(raw.method==method)&(raw.n_pairs==row.n_pairs)&(raw.boundary=='EVENT_CACHE_PAIR_RANK')].wall_s
                        self.check(f'{method}/{row.n_pairs}/cached timing median',len(times)==row.pair_repeats and np.isclose(times.median(),getattr(row,field),rtol=1e-12))
            axes[j].set(xlabel='Number of event pairs',ylabel=label)
            axes[j].grid(alpha=.15)
        axes[0].legend()
        self.finish(fig,name,'Frozen speed benchmark with its original input and hardware boundary, no new speedup claim')

    def run(self):
        self.distributions()
        self.retrieval(190)
        self.retrieval(450)
        self.attribution_and_topb()
        self.sky()
        self.injection()
        self.real()
        self.candidate_panels()
        self.timing()
        self.check('eleven reconstructed data figures',len(self.figures)==11)
        self.check('all original source hashes unchanged',all(hashlib.sha256((self.paper/name).read_bytes()).hexdigest()==digest for name,digest in self.sources.items()))
        (self.output/'PLOTTED_VALUES.json').write_text(json.dumps(self.artists,indent=2,allow_nan=False))
        report={'status':'PASS','scope':'Data-equivalent replacement entries, NOT original-script or pixel-identical reproduction',
                'original_paper_modified':False,'sources':self.sources,'figures':self.figures,'checks':self.checks,
                'remaining_original_script_gap':True}
        (self.output/'REPORT.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':'PASS','figures':len(self.figures),'checks':len(self.checks)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    Replay(a.paper,a.output).run()
