#!/usr/bin/env python3
"""Complete development summaries and uncertainty, without choosing a winner."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import mcwf_development_20260905 as dev


def registry(root):
    for dep in ("gwtc3","gwtc4"):
        for seed in dev.SEEDS:
            yield {"config":"CFIX-baseline","deployment":dep,"model_seed":seed,"eval_seed":seed,
                   "path":dev.BAY/f"results/{dep}/seed_{seed}/test_pair_scores_bayestar_sky.parquet"}
        for parent in (root/"results").iterdir():
            if not parent.is_dir(): continue
            if parent.name.startswith("OLD-"):
                path=parent/dep/"test_pair_scores.parquet"
                if path.exists(): yield {"config":parent.name,"deployment":dep,"model_seed":202609031,"eval_seed":dev.SEEDS[0],"path":path}
            elif parent.name in ("MASSTF","PHASEBANK","PHASEBANK-MASSLR","PHASEPSD-MASSLR"):
                for modeldir in (parent/dep).glob("model_*_eval_*"):
                    _,m,_,e=modeldir.name.split("_")
                    for rule in ("mass_only","negative_guard","mass_cap"):
                        path=modeldir/f"test_{rule}_pairs.parquet"
                        if path.exists(): yield {"config":f"{parent.name}-{rule}","deployment":dep,"model_seed":int(m),"eval_seed":int(e),"path":path}
            elif parent.name in ("PHASEBANK-MCREPLACED-point","PHASEBANK-MCREPLACED-uncertainty","PHASEBANK-BOUNDED-SAFE","PHASEBANK-BOUNDED-PHYS","PHASEPSD-BOUNDED-SAFE","PHASEPSD-BOUNDED-PHYS"):
                for modeldir in (parent/dep).glob("model_*_eval_*"):
                    _,m,_,e=modeldir.name.split("_")
                    path=modeldir/"test_pairs.parquet"
                    if path.exists(): yield {"config":parent.name,"deployment":dep,"model_seed":int(m),"eval_seed":int(e),"path":path}


def ranks_bootstrap(frame,score,baseline,seed,repeats=10000):
    ranks=dev.BASE.v7.query_rank_rows(frame,score,"candidate")
    base=dev.BASE.v7.query_rank_rows(frame,baseline,"baseline")
    assert ranks.system_id.equals(base.system_id)
    out={};rng=np.random.default_rng(seed)
    for k in (1,10):
        samples=[];references=[]
        for fam in ("SIS","PM"):
            sel=ranks.family.eq(fam)
            s=ranks.loc[sel,["system_id"]].copy()
            s["candidate"]=(ranks.loc[sel,"query_rank"]<=k).to_numpy(float)
            s["baseline"]=(base.loc[sel,"query_rank"]<=k).to_numpy(float)
            g=s.groupby("system_id")[["candidate","baseline"]].mean()
            counts=s.groupby("system_id").size()
            if not counts.eq(2).all(): raise RuntimeError("Directed queries not paired")
            ids=rng.integers(0,len(g),size=(repeats,len(g)))
            samples.append(g.candidate.to_numpy()[ids].mean(1));references.append(g.baseline.to_numpy()[ids].mean(1))
        boot=np.mean(samples,axis=0);bboot=np.mean(references,axis=0)
        out[f"r{k}_system_ci_low"],out[f"r{k}_system_ci_high"]=np.quantile(boot,[.025,.975])
        out[f"delta_r{k}_system_ci_low"],out[f"delta_r{k}_system_ci_high"]=np.quantile(boot-bboot,[.025,.975])
    return out,ranks


def pair_bootstrap(frame,score,baseline,plan,seed,repeats=2000):
    """Paired source-block weighted bootstrap; no iid-pair confidence claims."""
    groups,group_idx=np.unique(plan.system_id.astype(str).to_numpy(),return_inverse=True)
    families=np.array([str(plan.iloc[np.flatnonzero(group_idx==g)[0]].family) for g in range(len(groups))])
    rng=np.random.default_rng(seed)
    counts=np.zeros((repeats,len(groups)),dtype=np.int16)
    for fam in np.unique(families):
        ids=np.flatnonzero(families==fam)
        counts[:,ids]=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=repeats)
    i=group_idx[frame.idx_i.to_numpy(int)];j=group_idx[frame.idx_j.to_numpy(int)]
    labels=frame.is_true_pair.to_numpy(bool)
    result=[]
    for scores in (score,baseline):
        order=np.argsort(-scores,kind="stable");ys=labels[order]
        ends=np.r_[np.flatnonzero(np.diff(np.asarray(scores)[order])!=0),len(order)-1]
        outputs=[]
        for start in range(0,repeats,40):
            c=counts[start:start+40]
            weights=(c[:,i].astype(float)*c[:,j])
            weights[:,i==j]=c[:,i[i==j]]
            w=weights[:,order]
            tp=np.cumsum(w*ys,axis=1);fp=np.cumsum(w*(~ys),axis=1)
            total=tp[:,-1].clip(1)
            tpe=tp[:,ends];fpe=fp[:,ends]
            prec=np.divide(tpe,tpe+fpe,out=np.zeros_like(tpe),where=(tpe+fpe)>0)
            dr=np.diff(np.pad(tpe/total[:,None],((0,0),(1,0))),axis=1)
            ap=(prec*dr).sum(1)
            f50=fp[np.arange(len(w)),np.argmax(tp>=.5*total[:,None],axis=1)]
            f90=fp[np.arange(len(w)),np.argmax(tp>=.9*total[:,None],axis=1)]
            outputs.append(np.stack([ap,f50,f90],axis=-1))
        result.append(np.concatenate(outputs))
    out={}
    for column,k in enumerate(("AP","F50","F90")):
        out[f"{k}_block_ci_low"],out[f"{k}_block_ci_high"]=np.quantile(result[0][:,column],[.025,.975])
        out[f"delta_{k}_block_ci_low"],out[f"delta_{k}_block_ci_high"]=np.quantile(result[0][:,column]-result[1][:,column],[.025,.975])
    return out


def run(root,bootstrap=False):
    rows=[];invariance=[];paircis=[]
    rankroot=root/"audit/query_ranks";rankroot.mkdir(parents=True,exist_ok=True)
    for entry in registry(root):
        config,dep,m,e=[entry[k] for k in ("config","deployment","model_seed","eval_seed")]
        frame=pd.read_parquet(entry["path"])
        baseline=pd.read_parquet(dev.BAY/f"results/{dep}/seed_{e}/test_pair_scores_bayestar_sky.parquet")
        for col in ("idx_i","idx_j","is_true_pair","time_score","sky_raw_log_bf"):
            if not np.array_equal(frame[col].to_numpy(),baseline[col].to_numpy()): raise RuntimeError(f"Frozen mismatch {config} {col}")
        invariance.append({**{k:v for k,v in entry.items() if k!="path"},"time_sky_pair_order_label_exact":True,
            "path":str(entry["path"]),"sha256":dev.sha(entry["path"])})
        for method in ("waveform_only","C_fixed"):
            s=frame.waveform_score.to_numpy() if method=="waveform_only" else dev.BASE.score_vector(frame,dev.BASE.FROZEN_V93_WEIGHTS[dep][e])
            bs=baseline.waveform_score.to_numpy() if method=="waveform_only" else dev.BASE.score_vector(baseline,dev.BASE.FROZEN_V93_WEIGHTS[dep][e])
            metrics=dev.BASE.full_metrics(frame,s);bm=dev.BASE.full_metrics(baseline,bs)
            row={"config":config,"deployment":dep,"model_seed":m,"eval_seed":e,"method":method,"test_role":"reused_development_comparator",**metrics}
            for k in ("macro_r_at_1","macro_r_at_10","average_precision","false_at_recall_0p5","false_at_recall_0p9"):
                row[f"delta_{k}"]=metrics[k]-bm[k]
            row["development_guardrail_pass"]=(row["delta_macro_r_at_10"]>=-.02 and row["delta_average_precision"]>=-.005 and metrics["false_at_recall_0p5"]<=1.1*bm["false_at_recall_0p5"] and metrics["false_at_recall_0p9"]<=1.1*bm["false_at_recall_0p9"])
            ci,rank=ranks_bootstrap(frame,s,bs,e)
            row.update(ci)
            rank.to_parquet(rankroot/f"{config}_{dep}_{m}_{method}.parquet",index=False)
            rows.append(row)
            if bootstrap and config in ("CFIX-baseline","PHASEBANK-mass_cap","PHASEBANK-MASSLR-negative_guard","PHASEBANK-BOUNDED-SAFE","PHASEBANK-BOUNDED-PHYS","PHASEPSD-BOUNDED-SAFE","PHASEPSD-BOUNDED-PHYS"):
                plan=dev.BASE.retained_event_plan(dep,e,"test")
                paircis.append({"config":config,"deployment":dep,"model_seed":m,"eval_seed":e,"method":method,
                    **pair_bootstrap(frame,s,bs,plan,e)})
        print({"summarized":config,"deployment":dep,"seed":m},flush=True)
    table=pd.DataFrame(rows)
    dev.csv_write(root/"tables/ALL_INJECTION_METRICS_PER_SEED.csv",table)
    columns=["macro_r_at_1","macro_r_at_10","average_precision","false_at_recall_0p5","false_at_recall_0p9",
        "top_10_precision","top_50_precision","top_100_precision","top_200_precision"]
    aggregate=table.groupby(["config","deployment","method"])[columns].agg(["mean","std","count"]).reset_index()
    aggregate.columns=["_".join(filter(None,c)) if isinstance(c,tuple) else c for c in aggregate.columns]
    dev.csv_write(root/"tables/ALL_INJECTION_METRICS_SUMMARY.csv",aggregate)
    dev.csv_write(root/"audit/TIME_SKY_LABEL_SCOPE_INVARIANCE.csv",pd.DataFrame(invariance))
    if paircis: dev.csv_write(root/"tables/PRIMARY_PAIRED_SOURCE_BLOCK_BOOTSTRAP.csv",pd.DataFrame(paircis))
    dev.json_write(root/"contracts/UNCERTAINTY_METHOD.json",{
        "R1_R10":"10000 system-level stratified resamples; both directed queries together; fixed candidate catalog and trained model",
        "pair_metrics":"2000 paired source-block weighted bootstrap replicates, stratified SIS-slot/PM-slot/unlensed; null edges m_i*m_j, true within-system edge m_i; weighted empirical-catalog uncertainty, not new-catalog population coverage",
        "seed_SD":"three independently initialized trained heads, with corresponding frozen baseline deployment; report individually",
        "real_PE":"descriptive, shared events not iid pairs; no ordinary pair-level p-values",
        "old_test_reuse":"development analysis, not confirmatory significance"})
    budgets=[];correlations=[]
    for path in sorted((root/"results").glob("*/gwtc*/consensus_*_all_pairs_pe_official.parquet")):
        config=path.parent.parent.name;dep=path.parent.name
        if "-model" in config: continue
        method="waveform_only" if "waveform_only" in path.name else "C_fixed"
        frame=pd.read_parquet(path)
        pe=dev.pe_audit(root,dep)
        # Replace placeholder audit columns only in the new consolidated deliverable.
        plain=frame.drop(columns=[c for c in frame if c.startswith("official_") or c=="public_hanabi_table_overlap"])
        off=[c for c in pe if c.startswith("official_") or c=="pair_key"]
        frame=plain.merge(pe[off],on="pair_key",how="left",validate="one_to_one")
        out=root/f"final_tables/{dep}/{config}/{method}";out.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(out/"all_pairs.parquet",index=False)
        dev.csv_write(out/"top100.csv",frame.head(100))
        for b in (10,20,50,100): budgets.append(dev.budget_row(frame,config,dep,method,b))
        for label,col,sign in (("BC_Mc","pe_mc_bhattacharyya_coefficient",1),("minus_D_Mc","pe_mc_standardized_distance",-1)):
            mask=np.isfinite(frame[col])&np.isfinite(frame.waveform_score_mean)
            correlations.append({"config":config,"deployment":dep,"method":method,"target":label,"n_pairs":int(mask.sum()),
                "spearman_descriptive_only":stats.spearmanr(frame.loc[mask,"waveform_score_mean"],sign*frame.loc[mask,col]).statistic})
    dev.csv_write(root/"tables/ALL_REAL_PE_OFFICIAL_BUDGETS.csv",pd.DataFrame(budgets))
    dev.csv_write(root/"tables/ALL_REAL_WAVEFORM_PE_CORRELATIONS.csv",pd.DataFrame(correlations))
    forensic=pd.read_csv(root/"audit/PE_CONDITIONED_SIGNAL_MATCHED_FILTER_DIAGNOSTIC.csv")
    forensic["usable_as_SNR_comparison"]=forensic.window.eq("full32")
    forensic["assessment"]=np.where(forensic.window.eq("full32"),"PE-conditioned approximate audit only; no scoring inputs",
        "INVALID: hard masking raw colored strain causes spectral leakage and large edge artifacts; neither SNR loss nor physical evidence")
    dev.csv_write(root/"audit/PE_CONDITIONED_SIGNAL_FORENSICS_QUALIFIED.csv",forensic)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--bootstrap",action="store_true");a=p.parse_args();run(a.root,a.bootstrap)
