#!/usr/bin/env python3
"""Check selection and fallback without reading real candidates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/gw-catalog/scripts/experiments')
import mcwf_independent_predictive_scoring_20260909 as engine


class Rules(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mcwf_predictive_unit_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        engine.ROOT = self.root/'output'
        engine.PREDICTIVE = self.root/'source'
        for folder in ('contracts', 'configs'):
            (engine.ROOT/folder).mkdir(parents=True, exist_ok=True)
        for folder in ('contracts', 'calibration'):
            (engine.PREDICTIVE/folder).mkdir(parents=True, exist_ok=True)

    def source(self, passes=True):
        improvements = {'GLOBAL': [0.1, 0.0], 'EXPECTED': [0.02, 0.02],
                        'STRENGTH': [0.05, 0.005], 'BOUNDED': [0.0, 0.0]}
        data = {}
        for kind, values in improvements.items():
            for dep, gain in zip(engine.n.DEPS, values):
                data[dep+'_'+kind] = {'spec': {'ignored': kind},
                    'selection': {'R10_NLL': 0., 'NLL': -gain}}
        source = engine.PREDICTIVE/'calibration/INDEPENDENT_PREDICTIVE.json'
        source.write_text(json.dumps(data))
        receipt = {'sha256': engine.n.sha(source),
                   'both_run_pass_by_kind': {key: passes and key != 'GLOBAL' for key in improvements}}
        (engine.PREDICTIVE/'contracts/PREDICTIVE_FROZEN.json').write_text(json.dumps(receipt))

    def test_minimum_run_gain_not_best_single_run(self):
        self.source()
        out = engine.prepare()
        self.assertEqual(out['selected']['kind'], 'EXPECTED')
        self.assertFalse(out['real_or_test_used'])

    def test_all_failed_stops(self):
        self.source(False)
        with self.assertRaisesRegex(RuntimeError, 'HOLD_NO_BOTH_RUN'):
            engine.prepare()
        self.assertFalse((engine.ROOT/'configs/SELECTED_PREDICTIVE_KIND.json').exists())

    def test_selected_hash_change_rejected(self):
        self.source()
        engine.prepare()
        path = engine.ROOT/'configs/SELECTED_PREDICTIVE_KIND.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            engine.selected()

    def test_conditional_out_of_support_keeps_exact_mass(self):
        (engine.ROOT/'configs/SELECTED_PREDICTIVE_KIND.json').write_text('{}')
        original = np.full((3, 512), 1/512.)
        mass = {'p': original.copy(), 'active': np.array([True, True, False]),
                'profile_centers': np.log([10., 10., 30.]), 'outside': np.zeros(3)}
        spec = {'minimum_h': .001, 'maximum_h': .1, 'intercept': np.log(.005),
                'slope': .5, 'logh_center': np.log(.01), 'location': 0., 'df': 3}
        choice = {'selected': {'kind': 'EXPECTED'}, 'specs': {'gwtc3': spec}}
        with patch.object(engine.co, 'mass', return_value=mass), \
             patch.object(engine.n, 'recipes', return_value={('gwtc3', 1): {'slot': 2}}), \
             patch.object(engine, 'selected', return_value=choice), \
             patch.object(engine, 'covariates', return_value=(mass['profile_centers'], np.array([.01, .2, .01]))):
            result = engine.updated_mass('gwtc3', 1, 'validation')
        np.testing.assert_array_equal(result['p'][1:], original[1:])
        self.assertFalse(np.array_equal(result['p'][0], original[0]))
        np.testing.assert_allclose(result['p'].sum(1), 1., atol=1e-12)
        np.testing.assert_array_equal(result['predictive_active'], [True, False, False])


if __name__ == '__main__':
    unittest.main()
