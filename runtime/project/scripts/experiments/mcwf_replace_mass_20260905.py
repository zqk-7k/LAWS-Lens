#!/usr/bin/env python3
"""Replace only the failed mass estimate in the waveform composite.

Existing embedding cosine and q estimate are preserved. The waveform lookup is
refit on simulation validation; time/sky/fusion remain frozen. No real PE input.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf


def run(root,dep,model_seed,eval_seed):
    contract=root/"contracts/ROUND2B_MASS_REPLACEMENT.json"
    if not contract.exists():
        dev.json_write(contract,{
            "purpose":"replace old network mass estimate, not compensate it with another positive prior-overlap score",
            "frozen":"old embedding cosine, q difference, time, sky, C-fixed fusion",
            "mass_model":"simulation-selected PHASEBANK head",
            "variants":["point:absolute predicted logMc difference", "uncertainty:absolute logMc difference / max(combined predicted std,0.15)"],
            "calibration":"same validation-only feature standardization, mc/q grid and KDE protocol as previous waveform exploration",
            "selection":"macroR10,AP,F50,R1; bandwidth AP,R10,F50",
            "real_PE_or_official_data_in_selection":False,
            "interpretation":"exploratory reuse-test comparison; not fresh blind confirmation"})
    source=root/f"results/PHASEBANK/{dep}/model_{model_seed}_eval_{eval_seed}"
    metrics=[]; budgets=[]
    for variant in ("point","uncertainty"):
        config=f"PHASEBANK-MCREPLACED-{variant}"
        out=root/f"results/{config}/{dep}/model_{model_seed}_eval_{eval_seed}"
        if (out/"COMPLETE.json").exists(): continue
        out.mkdir(parents=True,exist_ok=True)
        for split in ("validation","test","real"):
            if split=="real":
                ref=dev.real_frame(dep,eval_seed)
                p=np.load(source/"real_mass_predictions.npz")["probability"]
            else:
                ref=pd.read_parquet(dev.BAY/f"results/{dep}/seed_{eval_seed}/{split}_pair_scores_bayestar_sky.parquet")
                p=np.load(source/f"{split}_mass_predictions.npz")["probability"]
            mean=p@tf.LOG_CENTERS; var=np.maximum(p@tf.LOG_CENTERS**2-mean**2,0)
            i=ref.idx_i.to_numpy(int);j=ref.idx_j.to_numpy(int)
            delta=abs(mean[i]-mean[j])
            if variant=="uncertainty": delta/=np.maximum(np.sqrt(var[i]+var[j]),.15)
            components={"cosine":ref.waveform_embedding_cosine.to_numpy(),
                "mc_similarity":-delta,"q_similarity":-ref.waveform_abs_delta_logitq_std.to_numpy(),
                "delta_mc":delta,"delta_q":ref.waveform_abs_delta_logitq_std.to_numpy()}
            if split=="validation":
                calibration,grid=dev.ORCH.fit_waveform_calibration(ref,components)
                dev.json_write(out/"waveform_calibration.json",calibration)
                dev.csv_write(out/"component_grid.csv",grid)
            raw,score=dev.ORCH.apply_waveform_calibration(components,calibration)
            frame=dev.ORCH.enrich_waveform_frame(ref,components,raw,score)
            assert np.array_equal(frame.time_score,ref.time_score) and np.array_equal(frame.sky_raw_log_bf,ref.sky_raw_log_bf)
            frame.to_parquet(out/f"{split}_pairs.parquet",index=False)
            if split=="real":
                budgets.extend(dev.save_evaluation(root,f"{config}-model{model_seed}",dep,{eval_seed:frame},dev.pe_audit(root,dep)))
            else:
                for method,s in (("waveform_only",score),("C_fixed",dev.BASE.score_vector(frame,dev.BASE.FROZEN_V93_WEIGHTS[dep][eval_seed]))):
                    metrics.append({"config":config,"deployment":dep,"model_seed":model_seed,"eval_seed":eval_seed,
                        "split":split,"method":method,**dev.BASE.full_metrics(frame,s)})
        dev.csv_write(out/"retrieval_pair_metrics.csv",pd.DataFrame([r for r in metrics if r["config"]==config]))
        dev.json_write(out/"COMPLETE.json",{"time_sky_identical":True,"real_PE_in_selection":False})
    if budgets:
        dev.csv_write(root/f"tables/PHASEBANK_REPLACEMENT_{dep}_{model_seed}_BUDGET.csv",pd.DataFrame(budgets))
        print(pd.DataFrame(budgets).query("budget==10 and seed=='consensus'").to_string(index=False),flush=True)
    if metrics:
        print(pd.DataFrame(metrics).query("split=='test'")[["config","method","macro_r_at_10","average_precision","false_at_recall_0p5","false_at_recall_0p9"]].to_string(index=False),flush=True)
    if not budgets and not metrics:
        print("All requested mass-replacement evaluations already complete; outputs preserved",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--deployment",required=True)
    p.add_argument("--seed",type=int,default=tf.SEEDS[0]);p.add_argument("--eval-seed",type=int,default=dev.SEEDS[0]);a=p.parse_args()
    run(a.root,a.deployment,a.seed,a.eval_seed)
