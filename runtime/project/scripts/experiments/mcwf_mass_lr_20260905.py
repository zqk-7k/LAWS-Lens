#!/usr/bin/env python3
"""Validation-calibrated mass-consistency verifier inside waveform channel."""
import argparse
from pathlib import Path
import numpy as np
import mcwf_development_20260905 as dev
import mcwf_mass_tf_20260905 as tf
import mcwf_phasebank_20260905 as phase


class MassLR:
    def __init__(self): self.lookup=None
    def __call__(self,frame,p,prior,split,out):
        means=p@tf.LOG_CENTERS
        raw=-abs(means[frame.idx_i.to_numpy(int)]-means[frame.idx_j.to_numpy(int)])
        if split=="validation":
            labels=frame.is_true_pair.to_numpy(bool)
            self.lookup=dev.ORCH.PHYS.fit_score_likelihood_ratio(raw[labels],raw[~labels],bandwidth_scale=1.)
            dev.json_write(out/"mass_distance_likelihood_ratio.json",self.lookup)
        if self.lookup is None: raise RuntimeError("Validation lookup not frozen")
        return dev.ORCH.PHYS.apply_score_likelihood_ratio(raw,self.lookup)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--deployment",required=True)
    p.add_argument("--seed",type=int,default=tf.SEEDS[0]);p.add_argument("--eval-seed",type=int,default=dev.SEEDS[0]);a=p.parse_args()
    contract=a.root/"contracts/ROUND2C_MASSLR.json"
    if not contract.exists():
        dev.json_write(contract,{
            "purpose":"empirical waveform-derived mass consistency, not a broad astrophysical prior-overlap reward",
            "features":"negative absolute difference of PHASEBANK predicted logMc means",
            "density":"same project one-dimensional signal/null KDE; fixed bandwidth_scale1, fitted on simulation validation",
            "score":"log p(feature|companion)/p(feature|null); empirical discriminative score, not physical PE Bayes factor",
            "waveform_variants":"same negative_guard and mass_cap gamma grid; same validation guardrails and tie rules as round1",
            "frozen":"embedding, phase head, time/sky, scope, initial fusion",
            "real_PE_official_in_scoring_or_selection":False,
            "out_of_support":"frozen boundary interpolation of existing LR implementation; report OOD explicitly",
            "data_role":"development, not fresh locked confirmation"})
    tf.evaluate(a.root,a.deployment,a.seed,a.eval_seed,config="PHASEBANK-MASSLR",encoder=phase.encode,
                model_config="PHASEBANK",mass_feature_builder=MassLR())
