from __future__ import annotations

import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matchgw import MatchRunConfig
from matchgw.catalog import catalog_system_report
from matchgw.data import EvaluationSet, MatchArrays, ground_truth_partner, load_match_arrays, pad_or_trim, peak_flip, split_indices, to_channels, zscore
from matchgw.matching import evaluate_scores, similarity_matrix, tune_matching
from matchgw.models import NTXentLoss
from matchgw.pipeline import build_model, default_candidate_params, default_tuning_grid, _amp_dtype, _device, _loader_kwargs
from matchgw.rerank import calibrated_candidate_report, candidate_feature_frame, fit_pair_calibrator


class PairDistillDataset(Dataset):
    # 返回 noisy 正样本对，以及同一事件的 pure waveform，供 frozen teacher 给出稳定目标。
    def __init__(self, noisy: MatchArrays, pure: MatchArrays, lensed_idx: np.ndarray, unlensed_idx: np.ndarray, cfg: MatchRunConfig) -> None:
        self.noisy, self.pure, self.cfg = noisy, pure, cfg
        self.items = [("L", int(i)) for i in lensed_idx] + [("U", int(i)) for i in unlensed_idx]
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.items)

    def _prep(self, x: np.ndarray, train: bool) -> torch.Tensor:
        y = pad_or_trim(x, self.cfg.target_len, self.cfg.stride).copy()
        if self.cfg.aug_flip:
            y = peak_flip(y)
        if train and self.cfg.aug_roll > 0:
            y = np.roll(y, int(self.rng.integers(-self.cfg.aug_roll, self.cfg.aug_roll + 1)))
        if train and self.cfg.aug_scale > 0:
            y = y * float(1.0 + self.rng.uniform(-self.cfg.aug_scale, self.cfg.aug_scale))
        if train and self.cfg.aug_noise > 0:
            y = y + self.rng.normal(0.0, self.cfg.aug_noise * (float(y.std()) + 1e-8), size=y.shape)
        return torch.from_numpy(to_channels(zscore(y), self.cfg.use_hilbert))

    def __getitem__(self, idx: int):
        kind, i = self.items[idx]
        if kind == "L":
            return (
                self._prep(self.noisy.l1[i], True), self._prep(self.noisy.l2[i], True),
                self._prep(self.pure.l1[i], False), self._prep(self.pure.l2[i], False),
            )
        x1 = self._prep(self.noisy.unlensed[i], True)
        x2 = self._prep(self.noisy.unlensed[i], True)
        p = self._prep(self.pure.unlensed[i], False)
        return x1, x2, p, p


def load_teacher(teacher_dir: Path, cfg: MatchRunConfig, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(teacher_dir / 'model.pt', map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def embed_eval_model(model, arrays, l_idx, u_idx, cfg, cpu):
    ds = EvaluationSet(arrays, l_idx, u_idx, cfg)
    device = _device(cpu)
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, **_loader_kwargs(cfg, device))
    chunks=[]
    for x in dl:
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=bool(cfg.amp and device.type=='cuda')):
            chunks.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(chunks).astype('float32'), ds.meta


def run(args):
    t0=time.perf_counter()
    cfg=MatchRunConfig(data_root=args.data_root, model_type=args.model_type, data_mode='noisy', out_dir=args.out_dir,
        backbone='inceptiontime', lensed_limit=args.lensed_limit, unlensed_limit=args.unlensed_limit,
        epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_memory, amp=args.amp,
        target_len=args.target_len, stride=args.stride, lr=args.lr, width_scale=args.width_scale,
        d_model=args.d_model, emb_dim=args.emb_dim, use_hilbert=args.use_hilbert,
        aug_roll=args.aug_roll, aug_scale=args.aug_scale, aug_noise=args.aug_noise,
        candidate_topk=args.candidate_topk, p_low=args.p_low, p_high=args.p_high, calibration_iters=args.calibration_iters)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    noisy=load_match_arrays(cfg)
    pure_cfg=MatchRunConfig(**{**asdict(cfg), 'data_mode':'pure', 'aug_roll':0, 'aug_scale':0.0, 'aug_noise':0.0})
    pure=load_match_arrays(pure_cfg)
    splits=split_indices(len(noisy.l1), len(noisy.unlensed), cfg)
    device=_device(args.cpu)
    teacher=load_teacher(args.teacher_dir, pure_cfg, device)
    student=build_model(cfg).to(device)
    if cfg.amp and device.type=='cuda':
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision('high')
    ds=PairDistillDataset(noisy, pure, splits['lensed']['train'], splits['unlensed']['train'], cfg)
    dl=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,drop_last=True,**_loader_kwargs(cfg,device,train=True))
    ntx=NTXentLoss(cfg.tau)
    opt=torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist=[]; train_t0=time.perf_counter()
    for ep in range(1,cfg.epochs+1):
        ep_t0=time.perf_counter(); losses=[]; closses=[]; dlosses=[]
        student.train()
        for xa, xb, pa, pb in dl:
            xa=xa.to(device,non_blocking=True); xb=xb.to(device,non_blocking=True)
            pa=pa.to(device,non_blocking=True); pb=pb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type=device.type,dtype=_amp_dtype(cfg),enabled=bool(cfg.amp and device.type=='cuda')):
                ta=teacher(pa).float(); tb=teacher(pb).float()
            with torch.autocast(device_type=device.type,dtype=_amp_dtype(cfg),enabled=bool(cfg.amp and device.type=='cuda')):
                za=student(xa).float(); zb=student(xb).float()
                contrast=ntx(za,zb)
                distill=0.5*((1-F.cosine_similarity(za,ta,dim=-1)).mean()+(1-F.cosine_similarity(zb,tb,dim=-1)).mean())
                loss=contrast + args.distill_weight*distill
            loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu())); closses.append(float(contrast.detach().cpu())); dlosses.append(float(distill.detach().cpu()))
        row={'epoch':ep,'loss':float(np.mean(losses)),'contrast':float(np.mean(closses)),'distill':float(np.mean(dlosses)),'epoch_s':time.perf_counter()-ep_t0}
        hist.append(row); print(json.dumps(row), flush=True)
    train_s=time.perf_counter()-train_t0
    val_emb,val_meta=embed_eval_model(student,noisy,splits['lensed']['val'],splits['unlensed']['val'],cfg,args.cpu)
    val_scores=similarity_matrix(val_emb); val_gt=ground_truth_partner(val_meta)
    best,val_stats=tune_matching(val_scores,val_gt,default_tuning_grid(cfg),metric=cfg.tune_for)
    cal=fit_pair_calibrator(candidate_feature_frame(val_scores,val_gt,default_candidate_params(cfg)),cfg)
    test_emb,test_meta=embed_eval_model(student,noisy,splits['lensed']['test'],splits['unlensed']['test'],cfg,args.cpu)
    test_scores=similarity_matrix(test_emb); test_gt=ground_truth_partner(test_meta)
    test=evaluate_scores(test_scores,test_gt,**best)
    cand,cand_stats=calibrated_candidate_report(test_scores,test_gt,default_candidate_params(cfg),cal,cfg)
    cat1,cat1_stats=catalog_system_report(cand,test_meta,cfg.p_high,threshold_name='tier1')
    cat12,cat12_stats=catalog_system_report(cand,test_meta,cfg.p_low,threshold_name='tier12')
    res={'method':'hybrid_noisy_contrastive_plus_pure_teacher_distill','distill_weight':args.distill_weight,'config':{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},'teacher_dir':str(args.teacher_dir),'best_params':best,'val':val_stats,'test':test,'test_candidates':cand_stats,'test_catalog_tier1':cat1_stats,'test_catalog_tier12':cat12_stats,'history':hist,'timing':{'train_s':train_s,'total_s':time.perf_counter()-t0}}
    torch.save({'model':student.state_dict(),'config':res['config'],'teacher_dir':str(args.teacher_dir)},cfg.out_dir/'model.pt')
    pd.DataFrame(hist).to_csv(cfg.out_dir/'history.csv',index=False)
    cand.to_csv(cfg.out_dir/'test_candidates.csv',index=False); cat1.to_csv(cfg.out_dir/'test_catalog_systems_tier1.csv',index=False); cat12.to_csv(cfg.out_dir/'test_catalog_systems_tier12.csv',index=False)
    with open(cfg.out_dir/'summary.json','w',encoding='utf-8') as f: json.dump(res,f,indent=2,ensure_ascii=False)
    print(json.dumps(res,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--teacher-dir',type=Path,required=True)
    ap.add_argument('--model-type',choices=['SIS','PM'],required=True); ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--lensed-limit',type=int,default=10000); ap.add_argument('--unlensed-limit',type=int,default=10000)
    ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--eval-batch-size',type=int,default=512)
    ap.add_argument('--num-workers',type=int,default=4); ap.add_argument('--pin-memory',action='store_true'); ap.add_argument('--amp',action='store_true')
    ap.add_argument('--target-len',type=int,default=8192); ap.add_argument('--stride',type=int,default=2); ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--width-scale',type=float,default=2.0); ap.add_argument('--d-model',type=int,default=256); ap.add_argument('--emb-dim',type=int,default=128); ap.add_argument('--use-hilbert',action='store_true')
    ap.add_argument('--aug-roll',type=int,default=32); ap.add_argument('--aug-scale',type=float,default=0.03); ap.add_argument('--aug-noise',type=float,default=0.0)
    ap.add_argument('--distill-weight',type=float,default=0.2)
    ap.add_argument('--candidate-topk',type=int,default=10); ap.add_argument('--p-low',type=float,default=0.20); ap.add_argument('--p-high',type=float,default=0.80); ap.add_argument('--calibration-iters',type=int,default=600)
    ap.add_argument('--cpu',action='store_true')
    run(ap.parse_args())
if __name__=='__main__': main()
