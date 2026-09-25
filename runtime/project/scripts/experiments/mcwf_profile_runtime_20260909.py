#!/usr/bin/env python3
"""Recorded implementation fix for zero eligible profile events."""
import json
from pathlib import Path
import runpy
import sys
import numpy as np

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_predictive_calibration_20260909 as cal

original = cal.density


def density(centers, spec, edges=None):
    edges = cal.d.EDGES if edges is None else edges
    if len(centers) == 0:
        return np.empty((0, len(edges) - 1), dtype=float)
    return original(centers, spec, edges)


if __name__ == '__main__':
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    receipt = root / 'audit/EMPTY_PROFILE_IMPLEMENTATION_FIX.json'
    sample = np.array([2.3, 2.4])
    spec = dict(location=0., scale=.005, df=3)
    assert np.array_equal(density(sample, spec), original(sample, spec))
    assert density(np.empty(0), spec).shape == (0, 512)
    if not receipt.exists():
        cal.n.write_json(receipt, {'UTC': cal.n.utc(), 'reason': 'Noeligibleeventsisanormalfallbackcase;maxonemptyarraywasinvalid.',
            'nonempty_probability_bit_exact': True, 'empty_shape': [0, 512],
            'scientific_configuration_changed': False, 'script_sha256': cal.n.sha(Path(__file__)),
            'previous_failure_log': 'logs/calibrate.stderr.log'})
    cal.density = density
    runpy.run_path(str(P / 'scripts/experiments/mcwf_profile_mode_aware_20260909.py'), run_name='__main__')
