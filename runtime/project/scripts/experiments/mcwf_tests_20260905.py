#!/usr/bin/env python3
"""Focused numerical and contract tests for the experimental waveform module."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
import mcwf_development_20260905 as dev
import mcwf_phasebank_20260905 as phase
import mcwf_mass_tf_20260905 as tf
import mcwf_bounded_20260905 as bounded


def run(root):
    checks=[]
    def check(name,condition,detail):
        checks.append({"name":name,"passed":bool(condition),"detail":detail})
        if not condition:raise AssertionError(name)
    bank=np.load(root/"cache/phasebank/gwtc3/quadrature_bank.npy")
    g=np.einsum("tdqx,tdrx->tdqr",bank,bank)
    check("quadrature_orthonormal",np.max(abs(g-np.eye(2)))<2e-5,float(np.max(abs(g-np.eye(2)))))
    x=np.random.default_rng(195).normal(size=(3,2,4096)).astype(np.float32)
    f=phase.features(x,bank,batch=3);ff=phase.features(-x,bank,batch=3)
    check("phase_features_sign_invariance",np.array_equal(f,ff),float(np.max(abs(f-ff))))
    check("finite_2s_input_features",np.isfinite(f).all() and f.shape==(3,3,64,3,3),list(f.shape))
    rejected=False
    try:
        bad=x.copy();bad[0,1,1]=np.nan;phase.features(bad,bank)
    except ValueError:rejected=True
    check("missing_detector_nan_rejected",rejected,"no silent zero filling")
    p=np.random.default_rng(6).dirichlet(np.ones(64),size=3);prior=np.ones(64)/64
    pairs=pd.DataFrame({"idx_i":[0,0,1],"idx_j":[1,2,2]})
    a=tf.mass_overlap(pairs,p,prior);b=tf.mass_overlap(pairs.rename(columns={"idx_i":"idx_j","idx_j":"idx_i"}),p,prior)
    check("mass_overlap_pair_symmetry",np.allclose(a,b,atol=1e-14),float(np.max(abs(a-b))))
    base=pd.DataFrame({"waveform_score":[-2.,0.,3.,4.],"mass_predictive_overlap":[-100.,-.5,0.,2.]})
    out=bounded.apply(base,2.,.25)
    diff=out.waveform_score-base.waveform_score
    check("bounded_never_adds_positive_evidence",bool((diff<=0).all()),diff.tolist())
    check("bounded_maximum_influence",bool((diff>=-.5).all()),diff.tolist())
    check("zero_gamma_exact_baseline",np.array_equal(bounded.apply(base,0.,1.).waveform_score,base.waveform_score),"bitwise score identity")
    for dep in ("gwtc3","gwtc4"):
        for seed in dev.SEEDS:
            f=pd.read_parquet(dev.BAY/f"results/{dep}/seed_{seed}/test_pair_scores_bayestar_sky.parquet")
            s=f.waveform_score.to_numpy()
            ap=dev.BASE.full_metrics(f,s)["average_precision"]
            check(f"frozen_AP_{dep}_{seed}",abs(ap-average_precision_score(f.is_true_pair,s))<1e-12,ap)
            ranks=dev.BASE.v7.query_rank_rows(f,s,"test")
            count=ranks.groupby("system_id").size()
            check(f"bootstrap_system_unit_{dep}_{seed}",count.eq(2).all(),{"systems":len(count),"queries":len(ranks)})
    dev.csv_write(root/"audit/FOCUSED_TEST_RESULTS.csv",pd.DataFrame(checks))
    dev.json_write(root/"audit/FOCUSED_TEST_SUMMARY.json",{"n_tests":len(checks),"all_passed":True})
    print(f"{len(checks)} tests passed",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();run(a.root)
