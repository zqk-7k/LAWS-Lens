#!/usr/bin/env python3
"""Read-only scientific checks; do not refit or select any configuration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf


def source_mapping(root):
    summary=[]
    for dep in ("gwtc3","gwtc4"):
        bank=dev.ORCH.SOURCE_ROOT/dep/"shared/physical_h1l1_source_bank"
        stores={f:pd.read_parquet(bank/f"{f}_data_0222/physical_source_pair_metadata.parquet") for f in ("SIS","PM")}
        stores["unlensed"]=pd.read_parquet(bank/"Unlensed_data_0222/physical_unlensed_source_metadata.parquet")
        for seed in dev.SEEDS:
            metadata=pd.read_parquet(dev.V7/dep/f"seed_{seed}/data/real_noise_injections/compact_injection_metadata.parquet").set_index(["family","sample_index"])
            if not metadata.index.is_unique:raise RuntimeError("Ambiguous injection sample index")
            for split in ("validation","test"):
                plan=dev.BASE.retained_event_plan(dep,seed,split)
                records=[]
                for r in plan.itertuples():
                    family="unlensed" if r.tag=="U" else r.family
                    source=stores[family].iloc[int(r.source_index)]
                    mixed=metadata.loc[(family,int(r.source_index))]
                    image=2 if r.tag=="L2" else 1
                    ok=int(source.gwlmc_row)==int(mixed.gwlmc_row)
                    noiseok=int(r.parent_noise_bank)==int(mixed[f"image{image}_noise_bank_index"])
                    records.append({"event_idx":r.idx,"family":family,"source_index":r.source_index,
                        "source_gwlmc_row":int(source.gwlmc_row),"injection_gwlmc_row":int(mixed.gwlmc_row),
                        "source_gwlmc_event_id":int(source.gwlmc_event_id),"source_mapping_equal":ok,
                        "noise_parent_equal":noiseok,
                        "chirp_mass_detector":float((source.mass_1_detector*source.mass_2_detector)**.6/(source.mass_1_detector+source.mass_2_detector)**.2)})
                frame=pd.DataFrame(records)
                dev.csv_write(root/f"audit/source_mapping/{dep}_{seed}_{split}.csv",frame)
                if not frame.source_mapping_equal.all() or not frame.noise_parent_equal.all():
                    raise RuntimeError(f"Injection provenance mismatch {dep} {seed} {split}")
                summary.append({"deployment":dep,"seed":seed,"split":split,"events":len(frame),"source_mapping_mismatches":0,"noise_mapping_mismatches":0})
    dev.csv_write(root/"audit/SOURCE_TRUTH_AND_NOISE_MAPPING_SUMMARY.csv",pd.DataFrame(summary))


def predictive_coverage(root):
    rows=[]
    for path in (root/"models").glob("*/gwtc*/seed_*/development_validation_predictions.npz"):
        model=path.parents[2].name;dep=path.parents[1].name;seed=int(path.parent.name.split("_")[1])
        checkpoint=torch.load(path.parent/"validation_selected_model.pt",map_location="cpu",weights_only=False)
        data=np.load(path);l=data["logits"].astype(float)/checkpoint["temperature"]
        prob=np.exp(l-np.logaddexp.reduce(l,axis=-1,keepdims=True))
        truth=data["truth"]@tf.LOG_CENTERS
        position=np.clip((truth-tf.LOG_EDGES[0])/np.diff(tf.LOG_EDGES)[0],0,64-1e-9)
        bins=position.astype(int);cum=np.c_[np.zeros(len(prob)),np.cumsum(prob,axis=1)]
        pit=cum[np.arange(len(prob)),bins]+(position-bins)*prob[np.arange(len(prob)),bins]
        embedding=data["embedding"].astype(float);values=np.linalg.eigvalsh(np.cov(embedding,rowvar=False));values=np.maximum(values,0)
        values=values/values.sum();effective=float(np.exp(-np.sum(values[values>0]*np.log(values[values>0]))))
        rows.append({"model":model,"deployment":dep,"seed":seed,"validation_images":len(truth),
            "temperature":checkpoint["temperature"],"logMc_mae":float(abs(prob@tf.LOG_CENTERS-truth).mean()),
            "central50_coverage":float(((pit>=.25)&(pit<=.75)).mean()),
            "central90_coverage":float(((pit>=.05)&(pit<=.95)).mean()),"embedding_effective_rank":effective,
            "role":"temperature-selected development validation, not independent PE coverage"})
    dev.csv_write(root/"tables/PREDICTIVE_VALIDATION_COVERAGE_AND_RANK.csv",pd.DataFrame(rows))


def support_audit(root):
    rows=[]
    for name in ("PHASEBANK-MASSLR","PHASEPSD-MASSLR"):
        for directory in (root/"results"/name).glob("gwtc*/model_*_eval_*"):
            dep=directory.parent.name;parts=directory.name.split("_");m,e=int(parts[1]),int(parts[3])
            lookup=json.loads((directory/"mass_distance_likelihood_ratio.json").read_text());grid=np.asarray(lookup["score_grid"])
            for split in ("validation","test","real"):
                p=np.load(directory/f"{split}_mass_predictions.npz")["probability"]
                frame=dev.real_frame(dep,e) if split=="real" else pd.read_parquet(directory/f"{split}_baseline_pairs.parquet")
                means=p@tf.LOG_CENTERS;u=-abs(means[frame.idx_i.to_numpy(int)]-means[frame.idx_j.to_numpy(int)])
                rows.append({"config":name,"deployment":dep,"model_seed":m,"eval_seed":e,"split":split,
                    "n_pairs":len(frame),"lookup_min":grid.min(),"lookup_max":grid.max(),"raw_min":u.min(),"raw_max":u.max(),
                    "outside_lookup_fraction":float(((u<grid.min())|(u>grid.max())).mean()),
                    "boundary_behavior":"frozen constant endpoint interpolation; no positive mass reward in bounded variants"})
    dev.csv_write(root/"tables/MASS_LR_SUPPORT_AUDIT.csv",pd.DataFrame(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
    source_mapping(args.root);predictive_coverage(args.root);support_audit(args.root)
    dev.json_write(args.root/"audit/FINAL_DIAGNOSTICS_COMPLETE.json",{"source_mapping":"PASS","no_refit":True,"no_selection":True})
