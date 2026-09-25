from __future__ import annotations
import importlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
base=importlib.import_module('scripts.experiments.17_waveform_reranker_existing')
aux=importlib.import_module('scripts.experiments.21_observable_aux_reranker')
GROUPS={
 'time':['delta_time'],
 'sky':['sky_sep'],
 'time_sky':['delta_time','sky_sep'],
 'mass_sky':['chirp_diff','q_diff','sky_sep'],
}

def make(obs,scores,ens,gt,names):
    cands=aux.topk_union(scores,ens,50); obs=obs.reset_index(drop=True)
    ra=obs.ra.to_numpy(); dec=obs.dec.to_numpy(); t=obs.geocent_time.to_numpy(); cm=obs.chirp_mass.to_numpy(); q=obs.mass_ratio.to_numpy()
    X=[]; y=[]; a=[]; c=[]
    for i,row in enumerate(cands):
      for j in row:
        feat=[]
        for name in names:
          if name=='delta_time': feat.append(float(np.log1p(abs(t[i]-t[j]))))
          elif name=='sky_sep': feat.append(float(aux.angular_sep(ra[i],dec[i],ra[j],dec[j])))
          elif name=='chirp_diff': feat.append(float(abs(np.log(cm[i]/cm[j]))))
          elif name=='q_diff': feat.append(float(abs(q[i]-q[j])))
        X.append(feat); y.append(1 if int(gt[i])==int(j) else 0); a.append(i); c.append(j)
    return np.asarray(X,dtype=np.float32), np.asarray(y,dtype=np.int8), np.asarray(a,dtype=np.int32), np.asarray(c,dtype=np.int32)

def run(fam,mode,out):
    od=out/fam/mode; od.mkdir(parents=True,exist_ok=True)
    _,vs,_,vsc,vens,vgt,cfg=base.load_score_sets(fam,'val',od/'val')
    _,ts,_,tsc,tens,tgt,_=base.load_score_sets(fam,'test',od/'test')
    vobs=aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root,fam,vs['lensed']['val'],vs['unlensed']['val']),mode,seed=333+abs(hash((fam,mode)))%10000)
    tobs=aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root,fam,ts['lensed']['test'],ts['unlensed']['test']),mode,seed=444+abs(hash((fam,mode)))%10000)
    rows=[]
    for g,names in GROUPS.items():
      Xv,yv,av,cv=make(vobs,vsc,vens,vgt,names); Xt,yt,at,ct=make(tobs,tsc,tens,tgt,names)
      clf=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.06,max_leaf_nodes=15,l2_regularization=1e-4,class_weight='balanced',random_state=42)
      clf.fit(Xv,yv); pv=clf.predict_proba(Xv)[:,1]; pt=clf.predict_proba(Xt)[:,1]
      r={'family':fam,'mode':mode,'group':g,'features':'+'.join(names),'feature_count':len(names),'val_auc':float(roc_auc_score(yv,pv)),**aux.eval_ranker(pt,at,ct,tgt)}
      rows.append(r); print(r,flush=True)
    return rows

def main():
    out=Path('runs/et10000_minimal_aux_selected'); results=[]
    for mode in ['realistic','rough']:
      for fam in ['SIS','PM']:
        results.extend(run(fam,mode,out))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/'minimal_aux_selected_summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False),flush=True)
if __name__=='__main__': main()
