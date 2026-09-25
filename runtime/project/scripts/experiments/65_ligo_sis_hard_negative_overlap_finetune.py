from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

m = importlib.import_module('scripts.experiments.58_ligo_sis_expected_angular_grid18_skymap_rerank')
base = m.base

OUT_ROOT = Path('runs/ligo_sis_hard_negative_overlap_finetune_20260605')
SRC_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
PRETRAINED_CKPT = SRC_ROOT / 'ligo_noisy_sis' / 'grid_skymap_cnn.pt'
EPS = 1e-8
EPOCHS = 6
BATCH_SIZE = 96
LR = 1.5e-4
PAIR_MARGIN = 0.12
PAIR_LAMBDA = 0.035
ANGULAR_LAMBDA = 0.20
TOP_HARD = 80
NEG_PER_POS = 500
CHUNK_ROWS = 64


def cfg_job():
    return m.JOBS[0]


def load_pack(job, split, need_scores=True):
    return m.load_pack(job, split, OUT_ROOT / split, need_scores)


def predict_prob(model, ds, device):
    return m.predict_maps(model, ds, device).astype(np.float32)


def hard_negatives_from_prob(prob, gt):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    neg = np.empty(len(valid), dtype=np.int32)
    for start in range(0, len(valid), 128):
        rows = valid[start:start+128]
        ov = np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)
        for rp, a in enumerate(rows):
            p = int(gt[a])
            s = ov[rp]
            s[int(a)] = -np.inf
            s[p] = -np.inf
            topn = min(TOP_HARD, len(s) - 2)
            top = np.argpartition(-s, topn)[:topn]
            # Pick one of the strongest false overlaps, not always the first, to reduce overfitting.
            neg[start + rp] = int(top[(start + rp) % len(top)])
    return valid, neg


class HardPairSkyDataset(Dataset):
    def __init__(self, ds, raw, gt, anchors, negs):
        self.ds = ds
        self.y = m.soft_skymap(raw)
        self.unit = base.unit_vectors(raw).astype(np.float32)
        self.gt = gt.astype(np.int64)
        self.anchors = anchors.astype(np.int64)
        self.negs = negs.astype(np.int64)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        a = int(self.anchors[idx])
        p = int(self.gt[a])
        n = int(self.negs[idx])
        return (
            self.ds[a], torch.from_numpy(self.y[a]), torch.from_numpy(self.unit[a]),
            self.ds[p], torch.from_numpy(self.y[p]), torch.from_numpy(self.unit[p]),
            self.ds[n], torch.from_numpy(self.y[n]), torch.from_numpy(self.unit[n]),
        )


def train_model(job, train_ds, train_raw, train_gt, val_ds, val_raw, out_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = m.SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_prob = predict_prob(model, train_ds, device)
    anchors, negs = hard_negatives_from_prob(base_prob, train_gt)
    pd.DataFrame({'anchor': anchors, 'hard_negative': negs, 'partner': train_gt[anchors]}).to_csv(out_dir / 'train_hard_negatives.csv', index=False)

    loader = DataLoader(HardPairSkyDataset(train_ds, train_raw, train_gt, anchors, negs), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_true = m.soft_skymap(val_raw)
    center_unit = torch.from_numpy(m.CENTER_UNIT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = {'score': -1e9, 'state': None, 'epoch': 0, 'val_mean_err': 999.0, 'val_kl': 999.0}
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses=[]; pair_losses=[]; kl_losses=[]
        for xa, ya, ua, xp, yp, up, xn, yn, un in loader:
            xa=xa.to(device); ya=ya.to(device); ua=ua.to(device)
            xp=xp.to(device); yp=yp.to(device); up=up.to(device)
            xn=xn.to(device); yn=yn.to(device); un=un.to(device)
            la=model(xa); lp=model(xp); ln=model(xn)
            logpa=F.log_softmax(la, dim=1); logpp=F.log_softmax(lp, dim=1); logpn=F.log_softmax(ln, dim=1)
            pa=torch.softmax(la, dim=1); pp=torch.softmax(lp, dim=1); pn=torch.softmax(ln, dim=1)
            loss_kl=(F.kl_div(logpa, ya, reduction='batchmean') + F.kl_div(logpp, yp, reduction='batchmean') + 0.35*F.kl_div(logpn, yn, reduction='batchmean'))/2.35
            va=F.normalize(pa @ center_unit, dim=1); vp=F.normalize(pp @ center_unit, dim=1); vn=F.normalize(pn @ center_unit, dim=1)
            loss_ang=(torch.mean(1.0-torch.sum(va*ua,dim=1)) + torch.mean(1.0-torch.sum(vp*up,dim=1)) + 0.35*torch.mean(1.0-torch.sum(vn*un,dim=1)))/2.35
            log_pos=torch.log(torch.sum(pa*pp, dim=1)+EPS)
            log_neg=torch.log(torch.sum(pa*pn, dim=1)+EPS)
            loss_pair=F.margin_ranking_loss(log_pos, log_neg, torch.ones_like(log_pos), margin=PAIR_MARGIN)
            loss=loss_kl + ANGULAR_LAMBDA*loss_ang + PAIR_LAMBDA*loss_pair
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach().cpu())); pair_losses.append(float(loss_pair.detach().cpu())); kl_losses.append(float(loss_kl.detach().cpu()))
        sched.step()
        val_prob = predict_prob(model, val_ds, device)
        val_kl = float(np.mean(np.sum(val_true*(np.log(np.maximum(val_true,EPS))-np.log(np.maximum(val_prob,EPS))), axis=1)))
        err = m.map_center_error(val_prob, val_raw)
        # Select for a balance: keep sky error low while rewarding less entropy.
        entropy = -np.sum(val_prob*np.log(np.maximum(val_prob, EPS)), axis=1) / np.log(val_prob.shape[1])
        score = -float(err.mean()) - 0.05*float(entropy.mean())
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'loss_kl': float(np.mean(kl_losses)), 'loss_pair': float(np.mean(pair_losses)), 'val_kl': val_kl, 'val_mean_err': float(err.mean()), 'val_median_err': float(np.median(err)), 'val_lt1': float(np.mean(err<1.0)), 'val_entropy_norm': float(entropy.mean()), 'select_score': score}
        history.append(row); print('HARD_PAIR_EPOCH', row, flush=True)
        if score > best['score']:
            best = {'score': score, 'state': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'epoch': epoch, 'val_mean_err': row['val_mean_err'], 'val_kl': val_kl}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'best': best, 'history': history, 'pair_margin': PAIR_MARGIN, 'pair_lambda': PAIR_LAMBDA}, out_dir/'grid_skymap_cnn.pt')
    pd.DataFrame(history).to_csv(out_dir/'grid_skymap_cnn_history.csv', index=False)
    return model, device, history, best


def map_overlap(prob, a, c):
    return np.log(np.sum(prob[a]*prob[c], axis=1)+EPS).astype(np.float32)


def feature_matrix(time_obs, prob, scores, ranks, a, c):
    return np.column_stack([
        m.log1p_delta_time_obs(time_obs, a, c),
        map_overlap(prob, a, c),
        scores[a, c].astype(np.float32),
        (1.0/np.maximum(ranks[a, c], 1)).astype(np.float32),
    ]).astype(np.float32)


def train_reranker(time_obs, gt, prob, scores, ranks):
    rng=np.random.default_rng(65001); valid=np.flatnonzero(gt>=0).astype(np.int32); n=len(time_obs)
    pos_a=valid; pos_c=gt[valid].astype(np.int32)
    neg_a=np.repeat(pos_a, NEG_PER_POS); neg_c=rng.integers(0,n,size=len(neg_a),dtype=np.int32)
    bad=(neg_c==neg_a)|(neg_c==gt[neg_a])
    while bad.any():
        neg_c[bad]=rng.integers(0,n,size=int(bad.sum()),dtype=np.int32); bad=(neg_c==neg_a)|(neg_c==gt[neg_a])
    a=np.concatenate([pos_a,neg_a]); c=np.concatenate([pos_c,neg_c]); y=np.concatenate([np.ones(len(pos_a),dtype=np.int8),np.zeros(len(neg_a),dtype=np.int8)])
    X=feature_matrix(time_obs,prob,scores,ranks,a,c); order=rng.permutation(len(y)); return X[order], y[order]


def eval_full(clf, time_obs, gt, prob, scores, ranks):
    valid=np.flatnonzero(gt>=0).astype(np.int32); n=len(time_obs); out=[]
    for start in range(0,len(valid),CHUNK_ROWS):
        rows=valid[start:start+CHUNK_ROWS]; a=np.repeat(rows,n).astype(np.int32); c=np.tile(np.arange(n,dtype=np.int32),len(rows))
        pred=clf.predict_proba(feature_matrix(time_obs,prob,scores,ranks,a,c))[:,1].reshape(len(rows),n)
        pred[np.arange(len(rows)),rows]=-np.inf; true=pred[np.arange(len(rows)),gt[rows].astype(int)]
        out.extend((1+np.sum(pred>true[:,None],axis=1)).tolist())
    r=np.asarray(out); return {'r@1':float(np.mean(r<=1)),'r@5':float(np.mean(r<=5)),'r@10':float(np.mean(r<=10)),'r@50':float(np.mean(r<=50)),'r@100':float(np.mean(r<=100)),'r@500':float(np.mean(r<=500)),'median_rank':float(np.median(r)),'valid':int(len(valid))}


def pair_quality(prob, gt):
    rng=np.random.default_rng(65002); valid=np.flatnonzero(gt>=0).astype(np.int32); pos=np.sum(prob[valid]*prob[gt[valid].astype(int)],axis=1)
    n=len(prob); a=rng.choice(valid,size=200000,replace=True); c=rng.integers(0,n,size=len(a),dtype=np.int32); bad=(c==a)|(c==gt[a])
    while bad.any():
        c[bad]=rng.integers(0,n,size=int(bad.sum()),dtype=np.int32); bad=(c==a)|(c==gt[a])
    neg=np.sum(prob[a]*prob[c],axis=1); pp=rng.choice(pos,size=len(neg),replace=True)
    auc=float(np.mean(pp>neg)+0.5*np.mean(pp==neg)); ent=-np.sum(prob*np.log(np.maximum(prob,EPS)),axis=1)/np.log(prob.shape[1])
    return {'entropy_norm_mean':float(np.mean(ent)),'pos_overlap_mean':float(np.mean(pos)),'neg_overlap_mean':float(np.mean(neg)),'overlap_ratio':float(np.mean(pos)/max(float(np.mean(neg)),EPS)),'overlap_auc_sampled':auc}


def run():
    OUT_ROOT.mkdir(parents=True, exist_ok=True); job=cfg_job(); out_dir=OUT_ROOT/'ligo_noisy_sis'
    _, train_ds, train_raw, _, train_gt, _ = load_pack(job,'train',False)
    _, val_ds, val_raw, val_time, val_gt, val_scores = load_pack(job,'val',True)
    _, test_ds, test_raw, test_time, test_gt, test_scores = load_pack(job,'test',True)
    model, device, history, best = train_model(job, train_ds, train_raw, train_gt, val_ds, val_raw, out_dir)
    val_prob=predict_prob(model,val_ds,device); test_prob=predict_prob(model,test_ds,device)
    val_ranks=base.row_ranks(val_scores); test_ranks=base.row_ranks(test_scores)
    X,y=train_reranker(val_time.reset_index(drop=True), val_gt, val_prob, val_scores.astype(np.float32), val_ranks.astype(np.int32))
    clf=HistGradientBoostingClassifier(max_iter=320,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1e-4,class_weight='balanced',random_state=65)
    clf.fit(X,y); pv=clf.predict_proba(X[:200000])[:,1]
    row={'detector':'LIGO','data_mode':'noisy','family':'SIS','method':'hard_negative_overlap_finetune','sky_model':'SkyMapCNN_grid18_hard_overlap_finetune','sky_best_epoch':int(best['epoch']),'sky_val_mean_error_rad':float(best['val_mean_err']),'sky_val_kl':float(best['val_kl']),'sample_auc':float(roc_auc_score(y[:len(pv)],pv)),**pair_quality(test_prob,test_gt),**eval_full(clf,test_time.reset_index(drop=True),test_gt,test_prob,test_scores.astype(np.float32),test_ranks.astype(np.int32))}
    pd.DataFrame([row]).to_csv(out_dir/'summary.csv',index=False); pd.DataFrame([row]).to_csv(OUT_ROOT/'summary.csv',index=False)
    print(row, flush=True)

if __name__=='__main__':
    run()
