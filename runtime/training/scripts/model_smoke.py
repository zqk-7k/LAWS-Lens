#!/usr/bin/env python3
"""Short-encoder forward/backward check on new data, never a formal model."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import unified_ab as u


def main(root):
    sys.path.insert(0, str(u.P))
    m = u.module(root/'scripts/short_trainer.py', 'uab_model_smoke')
    m.REPO = u.P; m.INPUT_SAMPLES = 4096
    torch.set_num_threads(2)
    rows = []
    for run in u.RUNS:
        inputs = []
        for p in sorted((root/'data'/run/'main/validation').glob('*/A_NEUTRAL_short.npy')):
            inputs.extend(np.load(p))
        x = np.stack([m.prepare(a, None, False) for a in inputs[:8]])
        m.seed_everything(2026091700)
        model = m.PhysicsRegularizedEncoder(m.build_base_encoder('inception_attention', None)).cuda()
        model.train()
        z, mu = model(torch.as_tensor(x, device='cuda'), return_parameters=True)
        loss = (z[:, 0]**2).mean()+mu.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert torch.isfinite(z).all() and torch.isfinite(mu).all()
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
        rows.append({'run': run, 'input_shape': list(x.shape), 'embedding_shape': list(z.shape),
                     'parameter_shape': list(mu.shape), 'finite_nonzero_gradients': True,
                     'no_checkpoint_saved_or_reused': True})
        del model; torch.cuda.empty_cache()
    u.write(root/'contracts/MODEL_SMOKE_PASS.json', {'utc': u.now(), 'checks': rows,
                                                   'not_formal_training_or_retrieval': True})
    print(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
