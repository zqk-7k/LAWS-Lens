#!/usr/bin/env python3
"""Read-only physical and predictive audit of the declared failure pair."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_nodup_reliability_20260909 as r
import mcwf_conditional_intrinsics_20260908 as intr
n=r.n
PAIR='GW191103_012549--GW191105_143521'
LOW=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
PROFILE=P/'results/mcwf_nodup_mode_profile_10_20260909T085830Z'
STRENGTH=P/'results/mcwf_nodup_strength_integrated_28_20260909T121700Z'
PAIRED=P/'results/mcwf_nodup_paired_profile_26_20260909T120700Z'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    if root.exists():
        raise RuntimeError('New output directory required')
    for d in ('contracts','tables','figures','scripts','manifest','reports','data'):
        (root/d).mkdir(parents=True)
    inputs={}
    def remember(p):
        inputs[str(p)]={'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}
        return p
    external=pd.read_parquet(remember(n.t.EXTERNAL/'gwtc3_external_reference.parquet'))
    ext=external.set_index('pair_key').loc[PAIR]
    scopes=pd.read_parquet(remember(PAIRED/'results/NODUP-DIRECT-REPLAY/gwtc3/seed_202607241/real/fusion_all_pairs.parquet'))
    case=scopes.set_index('pair_key').loc[PAIR]
    indices=[int(case.idx_i),int(case.idx_j)]
    events=[PAIR.split('--')[0],PAIR.split('--')[1]]
    edges=np.asarray(intr.old.EDGES)
    mass_edges=np.exp(edges); grid=.5*(mass_edges[:-1]+mass_edges[1:])
    collections={'NODUP neural':[],'Physical profile':[],'Strength conditioned':[]}
    events_table=[]
    for seed,slot in zip(n.SEEDS,n.t.MODEL_SLOTS):
        files={
            'NODUP neural':LOW/f'predictions/MULTIRATE/gwtc3/model_{slot}_eval_{seed}/real.npz',
            'Physical profile':PROFILE/f'predictions/gwtc3/{slot}_real.npz',
            'Strength conditioned':STRENGTH/f'predictions/gwtc3/{slot}_real.npz'}
        for method,path in files.items():
            prob=np.load(remember(path))['p'][indices]
            collections[method].append(prob)
            for k,p in enumerate(prob):
                quantiles=np.exp(np.interp([.05,.16,.5,.84,.95],np.r_[0.,p.cumsum()],edges))
                events_table.append({'method':method,'seed':seed,'event':events[k],
                    **{label:v for label,v in zip(('q05','q16','median','q84','q95'),quantiles)},
                    'predictive_mass_BC':float(np.sqrt(prob[0]*prob[1]).sum()),
                    'not_public_PE_posterior':True})
    pd.DataFrame(events_table).to_csv(root/'tables/PREDICTIVE_EVENT_SUMMARY.csv',index=False,encoding='utf-8-sig')
    pdfs={name:np.asarray(values) for name,values in collections.items()}
    np.savez_compressed(root/'data/PREDICTIVE_DENSITIES.npz',log_mass_edges=edges,**{k.replace(' ','_'):v for k,v in pdfs.items()})
    profiles=[]
    for i,event in zip(indices,events):
        row=json.loads(remember(PROFILE/f'profile_events/gwtc3/real/{i}.json').read_text())
        profiles.append({'event':event,**row})
    n.write_json(root/'data/PHYSICAL_PROFILE_DIAGNOSTIC.json',profiles)
    rows=[]
    arms=[('PATH875 reference',PAIRED,'PATH875-ARCHIVED'),
          ('NODUP no blend',PAIRED,'NODUP-DIRECT-REPLAY'),
          ('Paired conservative',PAIRED,'PAIRED-PROFILE-GLOBAL'),
          ('Strength active tree',STRENGTH,'INFORMATION-ACTIVE-TREE')]
    for label,folder,method in arms:
        con=pd.read_parquet(remember(folder/f'results/{method}/gwtc3/consensus/fusion_all_pairs.parquet')).set_index('pair_key').loc[PAIR]
        for seed in n.SEEDS:
            frame=pd.read_parquet(remember(folder/f'results/{method}/gwtc3/seed_{seed}/real/fusion_all_pairs.parquet'))
            rr=frame.set_index('pair_key').loc[PAIR]
            rows.append({'scheme':label,'seed':seed,'consensus_rank':int(con.consensus_rank),
                'direct_rank':int(rr['rank']),**{c:float(rr[c]) for c in
                    ('waveform_score','time_score','sky_raw_log_bf','waveform_contribution','time_contribution','sky_contribution','final_score')},
                'public_mc_BC':float(ext.pe_mc_bhattacharyya_coefficient),
                'public_mc_D':float(ext.pe_mc_standardized_distance)})
    changes=pd.DataFrame(rows)
    for col in ('time_score','sky_raw_log_bf','time_contribution','sky_contribution'):
        fixed=changes[changes.scheme!='PATH875 reference'].groupby('seed')[col].agg(lambda a:float(a.max()-a.min()))
        if fixed.max()!=0:
            raise RuntimeError('Frozen physical channel changed')
    n.write_csv(root/'tables/KEY_PAIR_SCORE_AND_RANK.csv',changes)
    public={k:(v.item() if isinstance(v,np.generic) else v) for k,v in ext.to_dict().items() if k.startswith(('pe_','official_'))}
    n.write_json(root/'data/PUBLIC_AUDIT_VALUES.json',public)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#12699a','#c24d40']
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,(name,array) in zip(axs[0],pdfs.items()):
        for k in (0,1):
            for p in array[:,k]:
                ax.plot(grid,p/np.diff(mass_edges),color=colors[k],alpha=.24,lw=.8)
            ax.plot(grid,array[:,k].mean(0)/np.diff(mass_edges),color=colors[k],label=events[k],lw=1.5)
        ax.set(xlim=(8.8,10.8),xlabel='Detector-frame chirp mass [solar masses]',ylabel='Predictive density',title=name)
    axs[0,0].legend(fontsize=7,loc='upper left')
    ax=axs[1,0]
    for k,side in enumerate(('i','j')):
        median=float(ext['pe_mc_median_'+side]); sigma=float(ext['pe_mc_sigma_'+side])
        ax.errorbar(median,k,xerr=sigma,fmt='o',capsize=5,color=colors[k],label=events[k])
    ax.set(yticks=[0,1],yticklabels=['GW191103','GW191105'],xlabel='Detector-frame chirp mass [solar masses]',title='Public PE: median and reported width')
    ax.text(.03,.86,f'BC = {float(ext.pe_mc_bhattacharyya_coefficient):.3f}\nD = {float(ext.pe_mc_standardized_distance):.2f}',transform=ax.transAxes)
    ax=axs[1,1]
    for pos,(label,_,_) in enumerate(arms):
        a=changes[changes.scheme==label]
        ax.scatter(np.repeat(pos,len(a)),a.direct_rank,color='#426b61',s=20)
        ax.scatter(pos,a.consensus_rank.iloc[0],marker='D',color='#9b4c8f',s=34)
    ax.set(xticks=range(4),xticklabels=['PATH875','NODUP','Paired\nconservative','Strength\nactive tree'],ylabel='Rank (smaller is higher)',title='Individual seeds and consensus')
    ax.invert_yaxis()
    ax=axs[1,2]
    for k,col in enumerate(('waveform_contribution','time_contribution','sky_contribution')):
        means=changes.groupby('scheme',sort=False)[col].mean().reindex([a[0] for a in arms]).to_numpy()
        ax.bar(np.arange(4)+(k-1)*.24,means,.24,label=col.split('_')[0])
    ax.set(xticks=range(4),xticklabels=['PATH875','NODUP','Paired','Strength'],ylabel='Mean contribution across fixed models',title='All three contributions; no new total blend')
    ax.legend(fontsize=8)
    fig.savefig(root/'figures/CRITICAL_PAIR_PREDICTIVE_AND_PUBLIC_PE.pdf')
    fig.savefig(root/'figures/CRITICAL_PAIR_PREDICTIVE_AND_PUBLIC_PE.png',dpi=170)
    plt.close(fig)
    folder=PROFILE/'cache/profile_inputs/gwtc3/real'
    ids=np.load(remember(folder/'event_ids.npy'))
    data=np.load(remember(folder/'full20.npy'),mmap_mode='r')
    pair=np.asarray([data[np.flatnonzero(ids==i)[0]] for i in indices])
    # Display-only robust scaling; score inputs remain untouched.
    scale=np.median(abs(pair[:,:,:8*2048]-np.median(pair[:,:,:8*2048],axis=-1,keepdims=True)),axis=-1)/.6744897501960817
    fig,axs=plt.subplots(2,2,figsize=(12,5),layout='constrained')
    for det in (0,1):
        for column,seconds in ((0,16),(1,2)):
            for k in (0,1):
                view=pair[k,det,-seconds*2048:]/scale[k,det]
                t=np.arange(len(view))/2048
                axs[det,column].plot(t,view,lw=.5,alpha=.65,color=colors[k],label=events[k])
            axs[det,column].set(xlabel='Time from displayed window start [s]',ylabel=f'{("H1","L1")[det]} / robust display scale',title=f'Last {seconds} s; existing 20 Hz conditioning')
    axs[0,0].legend(fontsize=8)
    fig.savefig(root/'figures/CRITICAL_PAIR_CONDITIONED_WAVEFORMS.pdf')
    fig.savefig(root/'figures/CRITICAL_PAIR_CONDITIONED_WAVEFORMS.png',dpi=170)
    plt.close(fig)
    n.write_csv(root/'manifest/INPUT_SHA256.csv',list(inputs.values()))
    n.write_json(root/'contracts/READONLY_CRITICAL_AUDIT.json',{'UTC':n.utc(),'pair':PAIR,
        'read_only':True,'no_score_changed':True,'no_goal_success_claim':True,
        'density_display':'Solid lines average three model predictive densities for display only. They are not full PE posterior, and averaging is not a scoring rule.',
        'public_display':'Only published-audit median and width used; no fabricated public posterior PDF.',
        'waveform_display':'Existing20Hzconditionedlast16s/2s;individual robustMAD display scaling;not a new encoder input or likelihood.',
        'physics_channels_exact':True,'official_overlap_not_lensing_truth':True})
    shutil.copy2(__file__,root/'scripts/critical_pair_explanation.py')
    print('CRITICAL_PAIR_EXPLANATION_COMPLETE',root,flush=True)


if __name__=='__main__':
    main()
