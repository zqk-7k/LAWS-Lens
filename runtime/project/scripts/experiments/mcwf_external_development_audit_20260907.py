#!/usr/bin/env python3
"""Frozen real-catalog descriptive audits. Nothing here selects a model."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as tf
import mcwf_new_confirmation_20260906 as fresh

KNOWN=('GW190924_021846--GW191105_143521','GW190412--GW191204_171526',
       'GW190924_021846--GW190930_133541')


def o4_extra_pe(out):
    source=dev.PROJECT/'results/gwtc_sky_ordering_corrected_v94_20260830/gwtc4/real_candidate_consensus_with_pe_v93.parquet'
    f=pd.read_parquet(source)
    f['pair_key']=['--'.join(sorted((str(a),str(b)))) for a,b in zip(f.event_i,f.event_j)]
    rename={'event_i':'pe_source_event_i','event_j':'pe_source_event_j'}
    for old,new in (('chirp_mass','mc'),('mass_ratio','q'),('chi_eff','chi_eff')):
        for which in ('i','j'):
            rename[f'{old}_median_{which}']=f'pe_{new}_median_{which}'
            rename[f'{old}_sigma_{which}_q16_q84']=f'pe_{new}_sigma_{which}'
        rename[f'{old}_standardized_posterior_distance']=f'pe_{new}_standardized_distance'
        rename[f'{old}_wasserstein_distance']=f'pe_{new}_wasserstein_distance'
        rename[f'{old}_bhattacharyya_coefficient']=f'pe_{new}_bhattacharyya_coefficient'
    result=f[['pair_key',*rename]].rename(columns=rename)
    morph=dev.module(dev.PROJECT/'scripts/experiments/sky_morphology_one_vs_two_stage_exploratory.py','mcwf_external_distance_reader')
    names=set(f.event_i)|set(f.event_j)
    distances=morph.load_distance_posteriors('gwtc4',names)
    rows=[]
    for row in f.itertuples():
        if row.event_i not in distances or row.event_j not in distances:
            raise RuntimeError('Frozen O4a posterior group lacks distance samples')
        metrics=dev.MAINCODE.posterior_metrics(distances[row.event_i],distances[row.event_j])
        rows.append({'pair_key':row.pair_key,**{'pe_dl_app_'+k:v for k,v in metrics.items()}})
    result=result.merge(pd.DataFrame(rows),on='pair_key',validate='one_to_one')
    result.to_parquet(out/'tables/gwtc4_full_frozen_PE_extension.parquet',index=False)
    dev.json_write(out/'tables/gwtc4_PE_EXTENSION_PROVENANCE.json',{
        'intrinsic_source':str(source),'intrinsic_source_sha256':dev.sha(source),
        'distance_reader':'same exact manifest sky_map_group as historical morphology audit; up to30000 fixed evenly indexed samples',
        'distance_not_used_for_intrinsic_gate':True,'new_PE_sampling':False,
        'width_scale':'(q84-q16)/2 in both deployments'})
    return result


def enrich(f,extra):
    if extra is None:
        return f
    indexed=extra.set_index('pair_key').loc[f.pair_key].reset_index()
    for col in extra:
        if col.startswith('pe_') and col in f:
            if not np.allclose(f[col].to_numpy(float),indexed[col].to_numpy(float),equal_nan=True,atol=1e-10,rtol=1e-10):
                raise RuntimeError('Frozen PE extension disagrees with existing metric: '+col)
    flip=indexed.pe_source_event_i.to_numpy()!=f.event_i.to_numpy()
    if not np.all(np.where(flip,indexed.pe_source_event_j,indexed.pe_source_event_i)==f.event_i.to_numpy()):
        raise RuntimeError('PE event orientation mismatch')
    added=[c for c in extra if c not in f and c not in ('pe_source_event_i','pe_source_event_j')]
    result=f.copy()
    for c in added:
        value=indexed[c].to_numpy()
        if c.endswith('_i') and c[:-2]+'_j' in indexed:
            value=np.where(flip,indexed[c[:-2]+'_j'],value)
        elif c.endswith('_j') and c[:-2]+'_i' in indexed:
            value=np.where(flip,indexed[c[:-2]+'_i'],value)
        result[c]=value
    return result


def run(root,output_name):
    conf=fresh.verify(root)
    trained=Path(conf['training_root'])
    out=root/output_name
    if out.exists():
        raise RuntimeError('External audit already exists; preserve it')
    out.mkdir()
    for d in ('tables','figures','waveform_inputs'):
        (out/d).mkdir()
    font=font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7})
    correlations,known,toprows,changes=[],[],[],[]
    for dep in ('gwtc3','gwtc4'):
        extra=o4_extra_pe(out) if dep=='gwtc4' else None
        _,events=dev.real_inputs(dep)
        scopes=set()
        for variant in ('BASELINE','UNIFIED'):
            for method in ('waveform_only','C_fixed'):
                path=root/f'results/{variant}/{dep}/consensus_{method}_all_pairs_pe_official.parquet'
                f=enrich(pd.read_parquet(path),extra)
                scopes.update(f.event_i);scopes.update(f.event_j)
                dev.csv_write(out/f'tables/{dep}_{variant}_{method}_all_pairs.csv',f)
                for cutoff in (10,20,50,100):
                    head=f.nsmallest(cutoff,'consensus_rank').copy()
                    head.insert(0,'budget',cutoff);head.insert(0,'variant',variant);head.insert(0,'deployment',dep)
                    toprows.append(head)
                score=f.waveform_score_mean.to_numpy(float)
                bc=f.pe_mc_bhattacharyya_coefficient.to_numpy(float)
                distance=f.pe_mc_standardized_distance.to_numpy(float)
                keep=np.isfinite(score)&np.isfinite(bc)&np.isfinite(distance)
                correlations.append({'deployment':dep,'variant':variant,'method':method,'pairs':int(keep.sum()),
                    'score_vs_BC_spearman':float(spearmanr(score[keep],bc[keep]).statistic),
                    'score_vs_negative_D_spearman':float(spearmanr(score[keep],-distance[keep]).statistic),
                    'p_value_not_reported':'pairs share events; descriptive correlation only'})
                old=pd.read_parquet(root/f'results/BASELINE/{dep}/consensus_{method}_all_pairs_pe_official.parquet')
                bad=old.loc[(old.consensus_rank<=100)&((old.pe_mc_bhattacharyya_coefficient<.1)|(old.pe_mc_standardized_distance>5)),'pair_key']
                for key in sorted(set(KNOWN)|set(bad)):
                    rows=f.loc[f.pair_key.eq(key)]
                    entry={'deployment':dep,'variant':variant,'method':method,'pair_key':key,
                        'status':'IN_SCOPE' if len(rows) else 'OUTSIDE_FROZEN_SCOPE'}
                    if len(rows):
                        entry.update(rows.iloc[0].to_dict())
                    known.append(entry)
                if variant=='UNIFIED':
                    joined=f.merge(old[['pair_key','consensus_rank','final_score_mean','waveform_score_mean']],on='pair_key',suffixes=('_new','_old'),validate='one_to_one')
                    joined.insert(0,'deployment',dep)
                    changes.append(joined)
        dev.csv_write(out/f'tables/{dep}_scope.csv',events[events.event_name.isin(scopes)])
        allfull,events=dev.real_inputs(dep)
        valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        raw=np.full((len(events),2,4096),np.nan,dtype=np.float32)
        raw[valid]=dev.TRAIN.make_window_view(np.asarray(allfull[valid],dtype=np.float32),2)
        index=dict(zip(events.event_name.astype(str),events.idx.astype(int)))
        f=enrich(pd.read_parquet(root/f'results/UNIFIED/{dep}/consensus_C_fixed_all_pairs_pe_official.parquet'),extra).nsmallest(10,'consensus_rank')
        probs=[]
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            p=np.load(trained/f'cache/predictions/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/real.npz')['p']
            probs.append(p)
        pmean=np.mean(probs,axis=0)
        for row in f.itertuples():
            i,j=index[str(row.event_i)],index[str(row.event_j)]
            name=f'{dep}_top{row.consensus_rank:02d}'
            np.savez_compressed(out/f'waveform_inputs/{name}.npz',event_i=row.event_i,event_j=row.event_j,
                waveform_i=raw[i],waveform_j=raw[j],sample_rate=2048,
                prediction_i=pmean[i],prediction_j=pmean[j],logMc_centers=tf.LOG_CENTERS)
            fig,axes=plt.subplots(2,2,figsize=(9,5.4),layout='constrained')
            t=np.arange(4096)/2048-2
            for d,ax in enumerate(axes[0]):
                for idx,label,color in ((i,row.event_i,'#287d8e'),(j,row.event_j,'#a54f6e')):
                    y=raw[idx,d].astype(float)
                    y=(y-y.mean())/max(y.std(),1e-12)
                    ax.plot(t,y,lw=.55,alpha=.8,color=color,label=label)
                ax.set(xlabel='Time relative to window end (s)',ylabel='Input / RMS',title=('H1','L1')[d],xlim=(-2,0))
            axes[0,0].legend(loc='upper left',fontsize=6,frameon=False)
            ax=axes[1,0]
            ax.plot(np.exp(tf.LOG_CENTERS),pmean[i],color='#287d8e',lw=1)
            ax.plot(np.exp(tf.LOG_CENTERS),pmean[j],color='#a54f6e',lw=1)
            ax.set(xscale='log',xlabel='Detector-frame chirp mass (solar masses)',ylabel='Probability per log-mass bin',
                title='Network predictive distributions (not PE)')
            ax=axes[1,1]
            for k,which,color in ((0,'i','#287d8e'),(1,'j','#a54f6e')):
                median=getattr(row,'pe_mc_median_'+which)
                sigma=getattr(row,'pe_mc_sigma_'+which)
                ax.errorbar(median,k,xerr=sigma,fmt='o',color=color,capsize=3)
            ax.set(yticks=[0,1],yticklabels=[row.event_i,row.event_j],ylim=(-.6,1.7),xlabel='Detector-frame chirp mass (solar masses)',title='Public PE median and frozen width scale')
            ax.text(.02,.97,f'BC = {row.pe_mc_bhattacharyya_coefficient:.3f}; D = {row.pe_mc_standardized_distance:.2f}',
                transform=ax.transAxes,va='top',fontsize=8)
            fig.suptitle(f'{dep}: consensus rank {row.consensus_rank} | frozen waveform development audit',fontsize=10)
            fig.savefig(out/f'figures/{name}_waveform_mass_audit.pdf')
            fig.savefig(out/f'figures/{name}_waveform_mass_audit.png',dpi=180)
            plt.close(fig)
        del raw,allfull
    dev.csv_write(out/'tables/REAL_WAVEFORM_MASS_CORRELATIONS.csv',pd.DataFrame(correlations))
    dev.csv_write(out/'tables/KNOWN_AND_TOP100_CATASTROPHIC_PAIRS.csv',pd.DataFrame(known))
    dev.csv_write(out/'tables/TOP_B_ALL_PE_OFFICIAL.csv',pd.concat(toprows,ignore_index=True))
    dev.csv_write(out/'tables/ALL_RANK_CHANGES.csv',pd.concat(changes,ignore_index=True))
    dev.json_write(out/'INTERPRETATION.json',{'candidate_was_frozen_before_this_audit':True,'parameter_selection':False,
        'real_data_role':'reused development audit, not independent blind confirmation',
        'official_candidates_are_not_true_lens_labels':True,'Hanabi_run_by_this_experiment':False,
        'waveform_plots':'same2s4096 input window; RMS-normalized only for display; time axis is relative to window end, not asserted geocentric merger time',
        'predictive_curves':'mean of three network categorical predictions for display, not Bayesian PE',
        'PE_intervals':'published marginal median plus/minus frozen pe_mc_sigma audit scale; O3 uses(q84-q16)/2, not a standard deviation or exact median-centered68percent interval',
        'font':font,'time_sky_and_weights_changed':False})
    fresh.verify(root)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-name',default='external_development_audit_v3')
    a=p.parse_args();run(a.root,a.output_name)
