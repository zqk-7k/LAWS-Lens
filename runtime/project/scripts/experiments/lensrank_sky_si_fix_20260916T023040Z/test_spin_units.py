"""Unit and numerical regression tests for the isolated spin-unit correction."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import bilby
import lal
import numpy as np
import pandas as pd

SOURCE = Path(__file__).with_name("bayestar_injection_sky_full_experiment.py")


def function():
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "spin_components")
    scope = {"pd": pd, "F_LOW_HZ": 20.0}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["spin_components"]


def source(**changes):
    values = dict(m1_det=37., m2_det=19., theta_jn=1.1, phijl=0.9,
                  tilt1=0.7, tilt2=1.8, phi12=1.4, a1=0.6, a2=0.4, phase=0.3)
    values.update(changes)
    return pd.Series(values)


class SpinUnitTests(unittest.TestCase):
    def test_interface_receives_kg_exactly_once(self):
        row = source()
        before = row.copy(deep=True)
        with patch.object(bilby.gw.conversion, "bilby_to_lalsimulation_spins",
                          return_value=(0., 0., 0., 0., 0., 0., 0.)) as call:
            function()(row)
        args = call.call_args.args
        self.assertEqual(args[7], 37. * lal.MSUN_SI)
        self.assertEqual(args[8], 19. * lal.MSUN_SI)
        self.assertEqual(args[9], 20.)
        pd.testing.assert_series_equal(row, before)

    def test_agrees_with_direct_lal_si_reference(self):
        for masses in [(37., 19.), (6., 5.), (180., 80.)]:
            for tilts in [(0.7, 1.8), (0., np.pi)]:
                with self.subTest(masses=masses, tilts=tilts):
                    r = source(m1_det=masses[0], m2_det=masses[1],
                               tilt1=tilts[0], tilt2=tilts[1])
                    args = [r[k] for k in ["theta_jn", "phijl", "tilt1", "tilt2",
                                           "phi12", "a1", "a2"]]
                    expected = bilby.gw.conversion.bilby_to_lalsimulation_spins(
                        *args, r.m1_det * lal.MSUN_SI, r.m2_det * lal.MSUN_SI, 20., r.phase)
                    angle, spins = function()(r)
                    np.testing.assert_allclose([angle, *spins], expected, rtol=0., atol=0.)
                    np.testing.assert_allclose(np.linalg.norm(spins[:3]), r.a1, atol=1e-12)
                    np.testing.assert_allclose(np.linalg.norm(spins[3:]), r.a2, atol=1e-12)

    def test_nonspinning_limit(self):
        r = source(a1=0., a2=0.)
        angle, spins = function()(r)
        self.assertEqual(angle, r.theta_jn)
        np.testing.assert_array_equal(spins, np.zeros(6))

    def test_precessing_regression_detects_old_bug(self):
        r = source()
        args = [r[k] for k in ["theta_jn", "phijl", "tilt1", "tilt2", "phi12", "a1", "a2"]]
        wrong = bilby.gw.conversion.bilby_to_lalsimulation_spins(
            *args, r.m1_det, r.m2_det, 20., r.phase)
        angle, spins = function()(r)
        self.assertGreater(np.max(np.abs(np.asarray([angle, *spins]) - wrong)), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
