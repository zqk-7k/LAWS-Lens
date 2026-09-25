from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

m = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base = m.base
SRC_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
CACHE_ROOT = Path('runs/ligo_sis_grid18_rank_fusion_20260604')
OUT_ROOT = Path('runs/ligo_sis_candidate_recall_diagnostic_20260605')
EPS=1e-8


def load_test():
    job=m.JOBS[0]
    _, train_ds, _, _, _, _ = m.load_pack(job,'train',SRC_ROOT/'train',False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=m.SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    ckpt=torch.load(SRC_ROOT/'ligo_noisy_sis'/'grid_skymap_cnn.pt',map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    _, ds, raw, time_obs, gt, scores = m.load_pack(job,'test',OUT_ROOT/'test',True)
    cache=CACHE_ROOT/'test_prob.npy'
    prob=np.load(cache) if cache.exists() else m.predict_maps(model,ds,device)
    return time_obs, gt, scores.astype(np.float32), prob.astype(np.float32)


def dt_rows(time_obs, rows):
    n=len(time_obs); allc=np.arange(n,dtype=np.int32); out=np.empty((len(rows),n),np.float32)
    for i,a in enumerate(rows):
        out[i]=-m.log1p_delta_time_obs(time_obs, np.full(n,int(a),dtype=np.int32), allc)
    return out


def ranks_from_score(score, rows, gt):
    s=score.copy()
    for i,a in enumerate(rows): s[i,int(a)] = -np.inf
    true=s[np.arange(len(rows)), gt[rows].astype(int)]
    return 1+np.sum(s>true[:,None],axis=1)


def top_set_contains(score, rows, gt, k):
    hit=[]
    for i,a in enumerate(rows):
        s=score[i].copy(); s[int(a)] = -np.inf
        top=np.argpartition(-s, min(k, len(s)-1))[:k]
        hit.append(int(gt[a]) in set(map(int,top)))
    return np.array(hit)


def union_contains(mats, rows, gt, k_each):
    hit=[]
    n=mats[0].shape[1]
    kk=min(k_each,n-1)
    for i,a in enumerate(rows):
        cand=set()
        for mat in mats:
            s=mat[i].copy(); s[int(a)] = -np.inf
            top=np.argpartition(-s, kk)[:kk]
            cand.update(map(int,top))
        hit.append(int(gt[a]) in cand)
    return np.array(hit)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    time_obs, gt, scores, prob = load_test()
    valid=np.flatnonzero(gt>=0).astype(np.int32)
    wf=scores[valid]
    sky=np.log(np.maximum(prob[valid]@prob.T, EPS)).astype(np.float32)
    dt=dt_rows(time_obs, valid)
    mats={'waveform':wf,'sky_overlap':sky,'trigger_time':dt}
    rows=[]
    for name,mat in mats.items():
        r=ranks_from_score(mat,valid,gt)
        row={'method':name,'r@1':float(np.mean(r<=1)),'r@5':float(np.mean(r<=5)),'r@10':float(np.mean(r<=10)),'r@50':float(np.mean(r<=50)),'r@100':float(np.mean(r<=100)),'r@500':float(np.mean(r<=500)),'median_rank':float(np.median(r))}
        rows.append(row); print(row, flush=True)
    for k in [5,10,20,50,100,200,500,1000]:
        hit=union_contains([wf,sky,dt],valid,gt,k)
        row={'method':f'union_top{k}_each_oracle_candidate_recall','r@candidate':float(hit.mean()),'candidate_max_size':int(3*k)}
        rows.append(row); print(row, flush=True)
    # simple best possible if candidate union top k each and oracle ranks true first inside candidate.
    pd.DataFrame(rows).to_csv(OUT_ROOT/'summary.csv',index=False)

if __name__=='__main__': main()
