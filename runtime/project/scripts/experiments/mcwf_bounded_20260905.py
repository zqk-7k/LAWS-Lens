#!/usr/bin/env python3
"""Bounded waveform mass guard with full validation false-burden guardrails."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev


def validation_truth(dep,ev,frame):
    plan=dev.BASE.retained_event_plan(dep,ev,"validation")
    source=dev.ORCH.SOURCE_ROOT/dep/"shared/physical_h1l1_source_bank"
    stores={f:pd.read_parquet(source/f"{f}_data_0222/physical_source_pair_metadata.parquet") for f in ("SIS","PM")}
    stores["U"]=pd.read_parquet(source/"Unlensed_data_0222/physical_unlensed_source_metadata.parquet")
    logmc=[]
    for row in plan.itertuples():
        f="U" if row.tag=="U" else row.family
        r=stores[f].iloc[int(row.source_index)]
        a,b=float(r.mass_1_detector),float(r.mass_2_detector)
        logmc.append(np.log((a*b)**.6/(a+b)**.2))
    logmc=np.asarray(logmc)
    delta=abs(logmc[frame.idx_i.to_numpy(int)]-logmc[frame.idx_j.to_numpy(int)])
    if not np.allclose(delta[frame.is_true_pair.to_numpy(bool)],0): raise RuntimeError("Truth/source-index mapping failed")
    return delta


def apply(frame,gamma,cap):
    f=frame.copy()
    f["waveform_score"]=frame.waveform_score.to_numpy()+gamma*np.clip(frame.mass_predictive_overlap.to_numpy(),-cap,0)
    f["bounded_mass_waveform_penalty"]=f.waveform_score-frame.waveform_score
    return f


def run(root,dep,source_config="PHASEBANK-MASSLR",output_prefix="PHASEBANK"):
    contract=root/("contracts/ROUND3_BOUNDED_MASS_GUARD.json" if output_prefix=="PHASEBANK" else f"contracts/ROUND4_{output_prefix}_BOUNDED_GUARD.json")
    if not contract.exists():
        dev.json_write(contract,{
            "rationale":"unbounded mass penalties reduced some PE catastrophes but increased injection F90; retain failed R2 and test finite influence",
            "model":f"same simulation-trained {output_prefix}; no new training during score selection",
            "formula":"new_Zwf = old_Zwf + gamma * clip(MassLR,-cap,0)",
            "gamma_grid":[0.,.25,.5,1.,2.,4.],"cap_grid":[.25,.5,1.,2.,4.],
            "selection_guardrails":"both waveform-only AND frozen three-channel validation: R10>=base-.02,AP>=base-.005,F50<=1.1*base,F90<=1.1*base",
            "variant_SAFE":"among admissible configurations minimize frozen three-channel F50,F90,then maximizeAP,R10,then smallest maximum penalty",
            "variant_PHYS":"among admissible configurations minimize mean of Top10 and Top50 absolute injected true logMc difference in frozen three-channel ranking,then F50,F90,AP,R10,smallest maximum penalty",
            "truth_for_PHYS":"only synthetic validation detector-frame injected masses; no test masses, no real PE or official candidates",
            "data":"same reused development comparator, not a newly blind test",
            "time_sky_weights_scope_frozen":True,"seeds":[202609051,202609052,202609053],
            "OOD":"mass LR boundary interpolation as archived; bounded negative-only effect; never add positive evidence",
            "selection_known_limits":"finite validation size; test guardrail failures must still be reported"})
    budgets=[];metrics=[];states=[]
    for m,e in zip((202609051,202609052,202609053),dev.SEEDS):
        src=root/f"results/{source_config}/{dep}/model_{m}_eval_{e}"
        val=pd.read_parquet(src/"validation_baseline_pairs.parquet")
        weights=dev.BASE.FROZEN_V93_WEIGHTS[dep][e]
        b_wf=dev.BASE.full_metrics(val,val.waveform_score.to_numpy())
        b_all=dev.BASE.full_metrics(val,dev.BASE.score_vector(val,weights))
        truth_delta=validation_truth(dep,e,val)
        grid=[]
        for gamma in (0.,.25,.5,1.,2.,4.):
            for cap in ((.25,) if gamma==0 else (.25,.5,1.,2.,4.)):
                f=apply(val,gamma,cap);sw=f.waveform_score.to_numpy();sa=dev.BASE.score_vector(f,weights)
                mw=dev.BASE.full_metrics(f,sw);ma=dev.BASE.full_metrics(f,sa)
                guard=all(mt["macro_r_at_10"]>=base["macro_r_at_10"]-.02 and mt["average_precision"]>=base["average_precision"]-.005 and
                    mt["false_at_recall_0p5"]<=1.1*base["false_at_recall_0p5"] and mt["false_at_recall_0p9"]<=1.1*base["false_at_recall_0p9"]
                    for mt,base in ((mw,b_wf),(ma,b_all)))
                order=np.argsort(-sa,kind="stable")
                phys=float(np.mean([truth_delta[order[:b]].mean() for b in (10,50)]))
                grid.append({"gamma":gamma,"cap":cap,"admissible":guard,"simulation_validation_mass_loss":phys,
                    **{f"wf_{k}":v for k,v in mw.items()},**ma})
        admissible=[r for r in grid if r["admissible"]]
        def secondary(r):return (r["false_at_recall_0p5"],r["false_at_recall_0p9"],-r["average_precision"],-r["macro_r_at_10"],r["gamma"]*r["cap"],r["gamma"],r["cap"])
        selected={"SAFE":min(admissible,key=secondary),"PHYS":min(admissible,key=lambda r:(r["simulation_validation_mass_loss"],)+secondary(r))}
        selection=root/f"results/{output_prefix}-BOUNDED/{dep}/model_{m}_eval_{e}"
        selection.mkdir(parents=True,exist_ok=True)
        dev.csv_write(selection/"validation_grid.csv",pd.DataFrame(grid))
        dev.json_write(selection/"selected_validation_config.json",selected)
        for variant,row in selected.items():
            config=f"{output_prefix}-BOUNDED-{variant}"
            out=root/f"results/{config}/{dep}/model_{m}_eval_{e}"
            if (out/"COMPLETE.json").exists(): continue
            out.mkdir(parents=True,exist_ok=True)
            states.append({"config":config,"deployment":dep,"model_seed":m,"eval_seed":e,"gamma":row["gamma"],"cap":row["cap"],"n_validation_admissible":len(admissible)})
            for split in ("validation","test","real"):
                if split=="real":
                    ref=dev.real_frame(dep,e).copy()
                    cached=pd.read_parquet(root/f"results/{source_config}-negative_guard-model{m}/{dep}/seed_{e}_C_fixed.parquet")
                    key=cached.set_index("pair_key").mass_predictive_overlap
                    ref["mass_predictive_overlap"]=ref.pair_key.map(key)
                    if ref.mass_predictive_overlap.isna().any():raise RuntimeError("Missing waveform mass feature")
                else: ref=pd.read_parquet(src/f"{split}_baseline_pairs.parquet")
                f=apply(ref,row["gamma"],row["cap"])
                assert np.array_equal(f.time_score,ref.time_score) and np.array_equal(f.sky_raw_log_bf,ref.sky_raw_log_bf)
                f.to_parquet(out/f"{split}_pairs.parquet",index=False)
                if split=="real": budgets.extend(dev.save_evaluation(root,f"{config}-model{m}",dep,{e:f},dev.pe_audit(root,dep)))
                else:
                    for method,s in (("waveform_only",f.waveform_score.to_numpy()),("C_fixed",dev.BASE.score_vector(f,weights))):
                        metrics.append({"config":config,"deployment":dep,"model_seed":m,"eval_seed":e,"split":split,"method":method,**dev.BASE.full_metrics(f,s)})
            dev.csv_write(out/"retrieval_pair_metrics.csv",pd.DataFrame([r for r in metrics if r["config"]==config and r["model_seed"]==m]))
            dev.json_write(out/"COMPLETE.json",{"selection_sha256":dev.sha(selection/"selected_validation_config.json"),"time_sky_frozen":True,"real_PE_used_in_selection":False})
    for variant in ("SAFE","PHYS"):
        config=f"{output_prefix}-BOUNDED-{variant}"
        frames={e:pd.read_parquet(root/f"results/{config}/{dep}/model_{m}_eval_{e}/real_pairs.parquet") for m,e in zip((202609051,202609052,202609053),dev.SEEDS)}
        r=dev.save_evaluation(root,config+"-three-seed",dep,frames,dev.pe_audit(root,dep));budgets.extend(r)
    label="BOUNDED" if output_prefix=="PHASEBANK" else f"{output_prefix}_BOUNDED"
    if states:dev.csv_write(root/f"tables/{label}_{dep}_SELECTED_CONFIGS.csv",pd.DataFrame(states))
    if metrics:dev.csv_write(root/f"tables/{label}_{dep}_METRICS.csv",pd.DataFrame(metrics))
    dev.csv_write(root/f"tables/{label}_{dep}_PE_BUDGETS.csv",pd.DataFrame(budgets))
    print(pd.DataFrame(budgets).query("budget==10 and seed=='consensus'").to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--deployment",required=True);a=p.parse_args();run(a.root,a.deployment)
