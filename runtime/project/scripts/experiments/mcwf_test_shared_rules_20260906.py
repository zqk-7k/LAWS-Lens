#!/usr/bin/env python3
"""No real data needed: check score symmetry, negative-only and schemas."""
import numpy as np
import pandas as pd
import torch
import mcwf_unified_waveform_20260906 as u
import mcwf_unified_tail_20260906 as tail
import mcwf_unified_contradiction_20260906 as bc
import mcwf_distilled_encoder_20260906 as distilled


def run():
    f=pd.DataFrame({'previous_waveform_score':[-3.,-1.,0.,1.,3.],
        'new_mass_predictive_BC':[1.,.9,.5,.1,.01]})
    spec={'threshold':.2,'scale':.3,'gamma':.25,'validation_max':4.,'neutral_floor':True}
    score,*_=tail.score(f,spec)
    assert np.all(score<=f.previous_waveform_score)
    assert np.array_equal(score[:3],f.previous_waveform_score.to_numpy()[:3])
    assert (score[3:]>=0).all()
    assert np.isfinite(score).all()
    model_spec={'x_thresholds':[-30.,-1.,0.],'p_thresholds':[.0001,.4,.99],'gamma':1.,'validation_min':-5.,'validation_max':0.}
    score,*_=bc.score(f,model_spec)
    assert np.all(score<=f.previous_waveform_score)
    rng=np.random.default_rng(202609060)
    z=torch.tensor(rng.normal(size=(16,128)),dtype=torch.float32,requires_grad=True)
    t=torch.tensor(rng.normal(size=(16,128)),dtype=torch.float32)
    loss=distilled.geometry_loss(z,t)
    perm=torch.randperm(16)
    assert torch.allclose(loss,distilled.geometry_loss(z[perm],t[perm]),atol=1e-7)
    assert distilled.geometry_loss(t,t)<1e-12
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    masses=torch.linspace(1.,5.,16)
    local=distilled.geometry_loss(z,t,masses)
    assert torch.allclose(local,distilled.geometry_loss(z[perm],t[perm],masses[perm]),atol=1e-7)
    assert distilled.geometry_loss(t,t,masses)<1e-12
    local.backward()
    assert torch.isfinite(z.grad).all()
    p=rng.dirichlet(np.ones(64),12)
    sim=np.sqrt(p[:,None]*p[None,:]).sum(-1)
    assert np.allclose(sim,sim.T) and np.allclose(np.diag(sim),1.)
    assert all(np.isfinite(v).all() for v in (sim,score))
    print('PASS: symmetry, finite scores, neutral floor, no positive boost, global/local loss permutation, finite gradient')


if __name__=='__main__':
    run()
