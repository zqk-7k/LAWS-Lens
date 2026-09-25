#!/usr/bin/env python3
"""Synthetic-only tests of public PE ingestion and descriptive audit algebra."""
import argparse
from pathlib import Path
import tempfile

import h5py
import numpy as np

import o4b_hl_nso_pe_audit_20260912 as pe
import o4b_hl_nso_diagnostics_20260912 as diagnostic
import o4b_hl_nso_data_training_20260912 as s


def run(root):
    rng = np.random.default_rng(2026091291)
    n = 1024
    values = np.zeros(n, dtype=[(name, 'f8') for name in
        ('mass_1', 'mass_2', 'chirp_mass', 'chi_eff', 'luminosity_distance')])
    values['mass_1'] = rng.uniform(29., 31., n)
    values['mass_2'] = rng.uniform(19., 21., n)
    values['chirp_mass'] = (values['mass_1']*values['mass_2'])**.6/(values['mass_1']+values['mass_2'])**.2
    values['chi_eff'] = rng.uniform(-.1, .1, n)
    values['luminosity_distance'] = rng.uniform(400., 600., n)
    (root/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root/'tmp') as folder:
        path = Path(folder)/'synthetic.hdf5'
        with h5py.File(path, 'w') as f:
            f.create_dataset('FROZEN_GROUP/posterior_samples', data=values)
        result, audit = pe.posterior(path, 'FROZEN_GROUP')
        assert np.allclose(result['Mc'], values['chirp_mass'])
        assert np.allclose(result['q'], values['mass_2']/values['mass_1'])
        try:
            pe.posterior(path, 'OTHER_GROUP')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Must not silently choose another posterior group')
    grid = np.linspace(0., 1., 256)
    p = pe.density(rng.beta(30., 2., n), grid, (0., 1.))
    q = pe.density(rng.beta(2., 30., n), grid, (0., 1.))
    assert abs(p.sum()-1) < 1e-12
    assert abs(np.sqrt(p*p).sum()-1) < 1e-12
    bc = np.sqrt(p*q).sum()
    assert 0 <= bc <= 1
    erank, participation = diagnostic.effective_rank(rng.normal(size=(500, 8)))
    assert 1 <= erank <= 8 and 1 <= participation <= 8
    zero_rank = diagnostic.effective_rank(np.ones((10, 4)))
    assert zero_rank == (0., 0.)
    s.write(root/'contracts/POSTPROCESS_SYNTHETIC_UNIT_TESTS.json', {
        'utc': s.now(), 'state': 'PASS', 'exact_group_required': True,
        'mass_frame_reconstruction_pass': True, 'normalized_BC_bounds_pass': True,
        'rank_constant_input_pass': True, 'real_PE_or_locked_test_read': False,
        'scope': 'software algebra only; does not certify scientific posterior coverage'})
    print('POSTPROCESS_SYNTHETIC_UNIT_TESTS_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
