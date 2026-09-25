#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_physical_mass_anchor_20260907 as anchor


def run(root):
    rng=np.random.default_rng(202609073)
    power=rng.uniform(1,100,(5,3,64,3,3))
    x=np.log1p(power)
    neural=rng.dirichlet(np.ones(64),size=5)
    p=anchor.profile(x,8.)
    shifted=np.log1p(power+100.)
    assert np.allclose(p,anchor.profile(shifted,8.),atol=1e-13)
    assert np.allclose(p.sum(-1),1.) and (p>=0).all()
    mixed=anchor.combine(neural,x,{'temperature':8.})
    assert np.allclose(mixed,.5*(p+neural),atol=1e-13)
    rows=[]
    for dep in ('gwtc3','gwtc4'):
        for ms in body.MODEL_SEEDS:
            path=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
            current=torch.load(path,weights_only=False,map_location='cpu')
            previous=torch.load(anchor.WARM/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt',weights_only=False,map_location='cpu')
            assert all(torch.equal(v,previous['model'][k]) for k,v in current['model'].items())
            assert all(torch.equal(v,previous['independent_mass_component'][k]) for k,v in current['independent_mass_component'].items())
            cache=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/development_validation.npz'
            old=anchor.WARM/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/development_validation.npz'
            a=np.load(cache);b=np.load(old)
            assert np.array_equal(a['embedding'],b['embedding']) and np.array_equal(a['group'],b['group'])
            rows.append({'deployment':dep,'seed':ms,'source_weights_identical':True,'neural_mass_weights_identical':True,
                'auxiliary_source_embedding_and_group_identical':True,'checkpoint_sha256':dev.sha(path)})
    dev.csv_write(root/'audit/ANCHOR_IDENTITY_TESTS.csv',pd.DataFrame(rows))
    dev.json_write(root/'audit/ANCHOR_UNIT_TESTS.json',{'pass':True,'normalization':True,'positive_probabilities':True,
        'physical_power_offset_invariance':True,'half_neural_half_physical_identity':True,
        'six_seed_model_identity':True,'no_real_PE_fields_in_profile':True,
        'interpretation':'predictive distribution only; not posterior evidence or full strain PE'})
    print('ANCHOR_UNIT_TESTS_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
