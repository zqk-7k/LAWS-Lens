#!/usr/bin/env python3
"""R77 corrected compatibility gate, using one deficit definition end-to-end."""
import sys
from pathlib import Path
import json
import hashlib
import tarfile
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/gw-catalog/scripts/experiments")
import mcwf_r76_compatibility_gate as r76

r76.OUT = r76.P / "results/mcwf_r77b_compatibility_gate_20260911T020000Z"

def select_corrected():
    out = r76.OUT
    out.mkdir(parents=True, exist_ok=False)
    for d in ("contracts","configs","tables","results","reports","figures",
              "manifest","scripts","logs"):
        (out/d).mkdir()
    contract = {
        "id": "MCWF-R77-COMPATIBILITY-GATE",
        "status": r76.STATUS, "goal_achieved": False,
        "question": "Does a validation-selected monotone gate on the measured full shared-parameter deficit improve physical waveform calibration?",
        "frozen": ["R75 encoder", "R75 Z_wf input", "Z_time", "Z_sky",
                   "outer weights", "scope/splits", "all historical outputs"],
        "formula": "w_new=w_old for w_old<=0 or D_full<=tau; otherwise w_new=w_old*exp(-gamma*(D_full-tau)/max(tau,1))",
        "deficit": "D_full_common_denominator is used both for threshold calibration and application; no D_mass/D_full mixing",
        "grid": {"gamma": list(r76.GAMMAS), "tau": "q50/q75/q90 of positive fit-fold D_full per deployment"},
        "selection": "Per seed, simulation fold-1 weighted log-loss only; all real PE/public columns excluded",
        "guard": "The selected gate must be no worse than identity on fold 1; gamma=0 is the deterministic fallback",
        "interpretation": "Adaptive development, not independent confirmation; no paper/v9.3 overwrite"
    }
    r76.write(out/"contracts/ANALYSIS_CONTRACT.json", contract)
    files=[Path("/root/autodl-tmp/gw-catalog/scripts/experiments/mcwf_r77_compatibility_gate.py"),
           r76.R75/"contracts/FINAL_AUDIT.json",
           r76.R73/"tables/PREDICTIONS.parquet",
           r76.R73/"configs/ALL_CONDITIONAL_MODELS.json"]
    pd.DataFrame([{"path":str(p),"sha256":r76.sha(p)} for p in files]).to_csv(
        out/"manifest/INPUT_SHA256.csv", index=False)
    chosen=[]; grid=[]; preds=[]
    for dep in r76.DEPS:
      for seed in r76.SEEDS:
        f=r76.load_dev(dep,seed)
        pos=f[(f.fold==0)&(f.kind=="true")].D_full_common_denominator.dropna()
        taus=[float(np.quantile(pos,q)) for q in (.5,.75,.9)]
        tune=f.fold==1
        y=f.loc[tune,"kind"].eq("true").to_numpy(float)
        wt=f.loc[tune,"global_balanced_weight"].to_numpy(float) if "global_balanced_weight" in f else np.ones(int(tune.sum()))
        best=None
        for ti,tau in enumerate(taus):
          for gamma in r76.GAMMAS:
            ss=r76.gate_score(f.waveform_score,f.D_full_common_denominator,tau,gamma)
            loss=float(np.average(r76.logloss(y,ss[tune]),weights=wt))
            row={"deployment":dep,"seed":seed,"tau_index":ti,"tau":tau,"gamma":gamma,
                 "fold1_logloss":loss}
            grid.append(row)
            key=(loss,0 if gamma==0 else 1,gamma,ti)
            if best is None or key<best[0]: best=(key,row)
        row=best[1]
        identity=min(x["fold1_logloss"] for x in grid
                     if x["deployment"]==dep and x["seed"]==seed and x["gamma"]==0)
        row["identity_logloss"]=identity
        row["not_worse_than_identity"]=bool(row["fold1_logloss"]<=identity+1e-12)
        if not row["not_worse_than_identity"]:
            row["tau_index"]=0; row["tau"]=taus[0]; row["gamma"]=0.; row["fold1_logloss"]=identity
        chosen.append(row)
        ss=r76.gate_score(f.waveform_score,f.D_full_common_denominator,row["tau"],row["gamma"])
        for fold in (0,1):
          m=f.fold==fold; z=f.loc[m].copy()
          z["r77_waveform"]=ss[m]; z["seed"]=seed; z["deployment"]=dep; z["fold_eval"]=fold
          keep=["pair_id","deployment","seed","fold_eval","kind","D_mass",
                "D_full_common_denominator","waveform_score","r77_waveform"]
          preds.append(z[[x for x in keep if x in z]])
        print("R77_SELECT",dep,seed,row,flush=True)
    pd.DataFrame(grid).to_csv(out/"tables/CALIBRATION_GRID.csv",index=False)
    pd.DataFrame(chosen).to_csv(out/"configs/SELECTED_GATE.csv",index=False)
    pd.concat(preds,ignore_index=True).to_parquet(out/"tables/SIMULATION_PREDICTIONS.parquet",index=False)
    return chosen

def finish(chosen):
    # Reuse only the frozen-panel and real-audit mechanics from R76.
    r76.score_all(chosen)
    inj=pd.read_csv(r76.OUT/"tables/INJECTION_FUSION_AND_REAL_AUDIT.csv")
    fusion=inj[(inj.get("mode","")=="fusion") & (inj.get("panel","")!="real_top50")]
    if len(fusion):
        fusion.groupby(["deployment","method","mode"])[
            ["R@1","R@5","R@10","AUPRC","F50","F90"]].agg(["mean","std"]).to_csv(
                r76.OUT/"tables/INJECTION_FUSION_SUMMARY.csv")
    r76.report(chosen)
    text=(r76.OUT/"reports/R76_REPORT_CN.md").read_text(encoding="utf-8")
    text=text.replace("R76","R77").replace("R76_GATE","R77_GATE")
    text += "\n\nR77 correction: threshold and applied deficit are both D_full_common_denominator. R76's D_mass/D_full mismatch is explicitly invalidated and is not used.\n"
    (r76.OUT/"reports/R77_REPORT_CN.md").write_text(text,encoding="utf-8")
    pkg = r76.P / "packages/mcwf_r77b_compatibility_gate_20260911T020000Z.tar.gz"
    items=[]
    for p in sorted(r76.OUT.rglob("*")):
      if (not p.is_file() or p.suffix in (".npy",".npz",".h5",".hdf5",".pt",".pth")
          or p.name == "SIMULATION_PREDICTIONS.parquet"):
        continue
      h=hashlib.sha256(p.read_bytes()).hexdigest()
      items.append((p,r76.OUT.name+"/"+str(p.relative_to(r76.OUT)),h))
    with (r76.OUT/"manifest/DELIVERY_SHA256SUMS.txt").open("w") as f:
      for _,a,h in items:f.write(h+"  "+a+"\n")
    with tarfile.open(pkg,"w:gz") as t:
      for p,a,_ in items:t.add(p,arcname=a)
      t.add(r76.OUT/"manifest/DELIVERY_SHA256SUMS.txt",
            arcname=r76.OUT.name+"/manifest/DELIVERY_SHA256SUMS.txt")
    h=hashlib.sha256(pkg.read_bytes()).hexdigest()
    (Path(str(pkg)+".sha256")).write_text(h+"  "+pkg.name+"\n")
    print("R77_PACKAGE",pkg,flush=True)

if __name__=="__main__":
    finish(select_corrected())
