from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

m=importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base=m.base
SRC_ROOT=Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
CACHE_ROOT=Path('runs/ligo_sis_grid18_rank_fusion_20260604')
OUT_ROOT=Path('runs/ligo_sis_union_top100_candidate_rerank_20260605')
EPS=1e-8
K_EACH=100
CHUNK_ROWS=64
FEATURE_KEYS=['wf','wf_z','rr','sky','sky_z','dt','dt_z','wf_rank_log','sky_rank_log','time_rank_log','sky_time_z','wf_sky_z','wf_time_z']


def load_model(job):
    _, train_ds, _, _, _, _=m.load_pack(job,'train',SRC_ROOT/'train',False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=m.SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    ckpt=torch.load(SRC_ROOT/'ligo_noisy_sis'/'grid_skymap_cnn.pt',map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model,device

def load_split(job,split,model,device):
    _, ds, raw, time_obs, gt, scores=m.load_pack(job,split,OUT_ROOT/split,True)
    cache=CACHE_ROOT/f'{split}_prob.npy'
    prob=np.load(cache) if cache.exists() else m.predict_maps(model,ds,device)
    ranks=base.row_ranks(scores)
    return time_obs,gt,scores.astype(np.float32),ranks.astype(np.int32),prob.astype(np.float32)

def zrow(x):
    x=x.astype(np.float32,copy=True); finite=np.isfinite(x)
    if not finite.all():
        for i in range(x.shape[0]):
            ok=finite[i]; fill=float(np.min(x[i,ok])) if ok.any() else 0.0; x[i,~ok]=fill
    return (x-x.mean(axis=1,keepdims=True))/np.maximum(x.std(axis=1,keepdims=True),1e-6)

def rank_desc(mat):
    order=np.argsort(-mat,axis=1); ranks=np.empty_like(order,dtype=np.int32); rr=np.arange(1,mat.shape[1]+1,dtype=np.int32); ranks[np.arange(mat.shape[0])[:,None],order]=rr; return ranks

def dt_mat(time_obs,rows):
    n=len(time_obs); allc=np.arange(n,dtype=np.int32); out=np.empty((len(rows),n),np.float32)
    for i,a in enumerate(rows): out[i]=-m.log1p_delta_time_obs(time_obs,np.full(n,int(a),dtype=np.int32),allc)
    return out

def cube(time_obs,gt,scores,wf_ranks,prob,rows):
    wf=scores[rows].astype(np.float32)
    sky=np.log(np.maximum(prob[rows]@prob.T,EPS)).astype(np.float32)
    dt=dt_mat(time_obs,rows)
    sr=rank_desc(sky); tr=rank_desc(dt)
    feats={'wf':wf,'wf_z':zrow(wf),'rr':(1.0/np.maximum(wf_ranks[rows],1)).astype(np.float32),'sky':sky,'sky_z':zrow(sky),'dt':dt,'dt_z':zrow(dt),'wf_rank_log':-np.log1p(wf_ranks[rows].astype(np.float32)),'sky_rank_log':-np.log1p(sr.astype(np.float32)),'time_rank_log':-np.log1p(tr.astype(np.float32))}
    feats['sky_time_z']=feats['sky_z']*feats['dt_z']; feats['wf_sky_z']=feats['wf_z']*feats['sky_z']; feats['wf_time_z']=feats['wf_z']*feats['dt_z']
    return feats

def candidates(feats,a,pos,row_pos):
    cand=set()
    for key in ['wf','sky','dt']:
        s=feats[key][row_pos].copy(); s[int(a)]=-np.inf
        top=np.argpartition(-s,K_EACH)[:K_EACH]
        cand.update(map(int,top))
    cand.discard(int(a))
    return np.array(sorted(cand),dtype=np.int32)

def take(feats,rp,cols): return np.column_stack([feats[k][rp,cols] for k in FEATURE_KEYS]).astype(np.float32)

def train_data(time_obs,gt,scores,wf_ranks,prob):
    valid=np.flatnonzero(gt>=0).astype(np.int32)
    feats=cube(time_obs,gt,scores,wf_ranks,prob,valid)
    Xs=[]; ys=[]; groups=[]
    for rp,a in enumerate(valid):
        pos=int(gt[a]); cand=candidates(feats,a,pos,rp)
        if pos not in set(map(int,cand)):
            cand=np.concatenate([cand,np.array([pos],dtype=np.int32)])
        y=(cand==pos).astype(np.int8)
        Xs.append(take(feats,rp,cand)); ys.append(y); groups.append(len(cand))
    X=np.vstack(Xs); y=np.concatenate(ys)
    rng=np.random.default_rng(57001); order=rng.permutation(len(y))
    return X[order],y[order]

def eval_model(clf,time_obs,gt,scores,wf_ranks,prob,score_all=False):
    valid=np.flatnonzero(gt>=0).astype(np.int32); n=len(time_obs); ranks=[]; in_cand=[]
    for st in range(0,len(valid),CHUNK_ROWS):
        rows=valid[st:st+CHUNK_ROWS]; feats=cube(time_obs,gt,scores,wf_ranks,prob,rows)
        for rp,a in enumerate(rows):
            pos=int(gt[a]); cand=candidates(feats,a,pos,rp); in_cand.append(pos in set(map(int,cand)))
            if score_all:
                cols=np.arange(n,dtype=np.int32)
            else:
                cols=cand
                if pos not in set(map(int,cols)):
                    # True target not in candidate set: put it after all candidates.
                    ranks.append(len(cols)+1); continue
            pred=clf.predict_proba(take(feats,rp,cols))[:,1]
            if score_all:
                pred[int(a)] = -np.inf
                true=pred[pos]; ranks.append(int(1+np.sum(pred>true)))
            else:
                true=pred[np.where(cols==pos)[0][0]]; ranks.append(int(1+np.sum(pred>true)))
    r=np.asarray(ranks); return {'candidate_recall':float(np.mean(in_cand)),'r@1':float(np.mean(r<=1)),'r@5':float(np.mean(r<=5)),'r@10':float(np.mean(r<=10)),'r@50':float(np.mean(r<=50)),'median_rank':float(np.median(r)),'valid':int(len(valid))}

def main():
    OUT_ROOT.mkdir(parents=True,exist_ok=True); job=m.JOBS[0]; model,device=load_model(job)
    val_time,val_gt,val_scores,val_ranks,val_prob=load_split(job,'val',model,device)
    test_time,test_gt,test_scores,test_ranks,test_prob=load_split(job,'test',model,device)
    X,y=train_data(val_time,val_gt,val_scores,val_ranks,val_prob)
    print('TRAIN',X.shape,'pos',int(y.sum()),'neg',int((y==0).sum()),flush=True)
    models={
      'hgb_union_top100':HistGradientBoostingClassifier(max_iter=220,learning_rate=0.045,max_leaf_nodes=15,l2_regularization=1e-3,class_weight='balanced',random_state=57),
      'logreg_union_top100':LogisticRegression(max_iter=1000,class_weight='balanced',n_jobs=4,C=0.5),
    }
    rows=[]
    for name,clf in models.items():
        clf.fit(X,y)
        met=eval_model(clf,test_time,test_gt,test_scores,test_ranks,test_prob,score_all=False)
        row={'method':name,'mode':'candidate_only','features':','.join(FEATURE_KEYS),**met}; rows.append(row); print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT_ROOT/'summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__': main()
