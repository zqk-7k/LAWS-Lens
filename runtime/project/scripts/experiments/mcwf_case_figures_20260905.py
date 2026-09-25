#!/usr/bin/env python3
"""Visual audit: frozen waveform inputs, public PE, and learned mass distributions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf

p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();root=a.root
pairs=[("GW191126_115259","GW191129_134029"),("GW191105_143521","GW191126_115259"),
       ("GW200316_215756","GW200322_091133"),("GW191113_071753","GW200316_215756")]
full,events=dev.real_inputs("gwtc3");index=events.set_index("event_name").idx.to_dict()
manifest=pd.read_csv(dev.MAIN/"cache/source_run/data/event_manifest.csv").set_index("event_name")
sample={n:dev.MAINCODE.load_pe_samples(manifest.loc[n])["chirp_mass"] for n in set(sum((list(p) for p in pairs),[]))}
post=[np.load(root/f"results/PHASEBANK/gwtc3/model_{m}_eval_{e}/real_mass_predictions.npz")["probability"] for m,e in zip((202609051,202609052,202609053),dev.SEEDS)]
colors=("#376F95","#AF4157");time=np.arange(4096)/2048-1.75
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":7,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(4,3,figsize=(11,9.4))
for row,(n1,n2) in enumerate(pairs):
    for detector in range(2):
        ax=axes[row,detector]
        for name,color in zip((n1,n2),colors):
            x=np.asarray(full[index[name],detector,-4096:],dtype=float);x=(x-x.mean())/x.std()
            ax.plot(time,x,lw=.35,color=color,alpha=.7,label=name)
        ax.axvline(0,color="black",lw=.5,ls="--");ax.set_xlim(-1.75,.25)
        ax.set_title(f"{('H1','L1')[detector]}: {n1}\n{n2}")
        ax.set_xlabel("Seconds relative to each event GPS");ax.set_ylabel("Whitened, normalized input")
    ax=axes[row,2]
    lo=min(np.quantile(sample[n],[.001])[0] for n in (n1,n2))
    hi=max(np.quantile(sample[n],[.999])[0] for n in (n1,n2))
    for name in (n1,n2):
        probability=np.mean([p[index[name]] for p in post],axis=0)
        upper=min(int(np.searchsorted(np.cumsum(probability),.995)),63)
        hi=max(hi,float(np.exp(tf.LOG_EDGES[upper+1])))
    lo=max(5,min(lo,np.exp(tf.LOG_CENTERS[0])));hi=min(200,max(hi,20))
    grid=np.geomspace(lo,hi,2000)
    for name,color in zip((n1,n2),colors):
        density=gaussian_kde(np.log(sample[name]))(np.log(grid))/grid
        ax.plot(grid,density,color=color,lw=1,label=name+" public PE")
        pmean=np.mean([p[index[name]] for p in post],axis=0)
        widths=np.diff(np.exp(tf.LOG_EDGES))
        ax.step(np.exp(tf.LOG_CENTERS),pmean/widths,where="mid",color=color,ls="--",lw=1,label=name+" waveform prediction")
    ax.set_xscale("log");ax.set_xlim(lo,hi);ax.set_xlabel("Detector-frame Mc (solar masses)");ax.set_ylabel("Probability density")
    ax.set_title("Public PE vs simulation-calibrated predictive density")
    if row==0:ax.legend(fontsize=5.5,frameon=False)
fig.suptitle("Waveform / Mc diagnostic: predictions are not full PE",fontsize=10)
fig.tight_layout(rect=(0,0,1,.97));fig.savefig(root/"figures/fig_O3_problem_pairs_waveforms_and_Mc.pdf",bbox_inches="tight")
fig.savefig(root/"figures/fig_O3_problem_pairs_waveforms_and_Mc.png",dpi=180,bbox_inches="tight");plt.close(fig)
dev.json_write(root/"figures/PROBLEM_PAIR_FIGURE_CONTRACT.json",{"pairs":pairs,"waveform_input":"same preprocessed peak2s, separately centered at eventGPS; not raw physical strain amplitude", "PE":"same exact manifest posterior group, detector-frame Mc", "prediction":"mean of 3 simulated-supervision PHASEBANK predictive distributions; no real PE labels", "x_limits":"include public PE 99.9th percentile and predictive 99.5th percentile, bounded to model mass range", "role":"visual development audit only"})
