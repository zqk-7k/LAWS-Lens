import unittest
import numpy as np
import pandas as pd
from fpp_core import ConditionalNull, tail_counts, assert_disjoint, catalog_queries, conservative_cut, interval_union
from build_fpp_tables import verify_sealed_real


class FPPTests(unittest.TestCase):
    def setUp(self):
        self.null = ConditionalNull([1., 2., 3.], [0, 0, 1], [1, 2, 2], ['a', 'b', 'c'], ['x', 'y', 'z'])

    def test_ge_ties(self):
        np.testing.assert_array_equal(tail_counts([1, 2, 2, 3], [0, 2, 3, 4]), [4, 3, 1, 0])

    def test_monotonic(self):
        self.assertTrue(np.all(np.diff(self.null.query(np.arange(5), False)['conditional_FPP']) <= 0))

    def test_same_score_same_fpp(self):
        self.assertEqual(self.null.query([2], False)['conditional_FPP'][0], 2/3)

    def test_zero_unresolved(self):
        out = self.null.query([4], False)
        self.assertTrue(out['zero_exceedance_unresolved'][0])
        self.assertEqual(out['tail_distinct_sources'][0], 0)

    def test_prefix_ties(self):
        null = ConditionalNull([1, 2, 2], [0, 0, 1], [1, 2, 2], ['a', 'b', 'c'], ['x', 'y', 'z'])
        self.assertEqual(null.query([2], False)['tail_distinct_sources'][0], 3)

    def test_group_sensitivity(self):
        r = self.null.query([2])
        self.assertEqual(r['noise_leave_one_out_min'][0], 0.)
        self.assertEqual(r['noise_leave_one_out_max'][0], 1.)

    def test_duplicate_source_rejected(self):
        with self.assertRaises(ValueError):
            ConditionalNull([1], [0], [1], ['a', 'a'], ['x', 'y'])

    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):
            ConditionalNull([1, 2, 3], [0, 0, 0], [1, 1, 2], ['a', 'b', 'c'], ['x', 'y', 'z'])

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            tail_counts([1, np.nan], [2])
        with self.assertRaises(ValueError):
            tail_counts([1, 2], [np.inf])

    def test_disjoint(self):
        a, b = pd.DataFrame({'source': ['a']}), pd.DataFrame({'source': ['b']})
        self.assertEqual(assert_disjoint(a, b, ['source']), {'source': 0})
        with self.assertRaises(ValueError):
            assert_disjoint(a, a, ['source'])

    def test_catalog_no_replacement(self):
        ids, maxima = self.null.catalogs(3, 10, 71)
        self.assertTrue(all(len(set(row)) == 3 for row in ids))
        np.testing.assert_array_equal(maxima, np.full(10, 3.))

    def test_catalog_probability_not_pair_probability(self):
        result = catalog_queries(np.full(10, 3.), [1/3], [3.], 3)
        self.assertEqual(result['conditional_catalog_FPP_mc'][0], 1.)
        self.assertEqual(result['conditional_expected_false_pairs'][0], 1.)

    def test_conservative_tail(self):
        for alpha in [.1, .2, .4, .9]:
            values = np.array([1, 1, 2, 2, 3])
            self.assertLessEqual(np.mean(values >= conservative_cut(values, alpha)), alpha)

    def test_calendar_union(self):
        self.assertEqual(interval_union([(0, 2), (1, 3), (4, 5)]), [[0., 3.], [4., 5.]])

    def test_mean_score_not_mean_fpp(self):
        a, b, q = np.array([0, 4, 10]), np.array([1, 2, 9]), 3.
        mean_tail = tail_counts((a+b)/2, [q])[0]/3
        average_tail = (tail_counts(a, [q])[0]+tail_counts(b, [q])[0])/6
        self.assertNotEqual(mean_tail, average_tail)

    def test_real_sealed_score_identity(self):
        raw = pd.DataFrame(dict(pair_key=['a--b'], final_score_POSITIVE=[3.],
            waveform_contribution_POSITIVE=[1.], time_contribution_POSITIVE=[1.],
            sky_contribution_POSITIVE=[1.], waveform_score=[2.], time_score=[4.], sky_raw_log_bf=[5.]))
        sealed = pd.DataFrame(dict(pair_key=['a--b'], seed=[1], method=['three-channel'],
            score=[3.], wf_contribution=[1.], time_contribution=[1.], sky_contribution=[1.],
            waveform_score=[2.], time_score=[4.], sky_raw_log_bf=[5.]))
        verify_sealed_real(raw, sealed, 1)
        raw.loc[0, 'final_score_POSITIVE'] = 4.
        with self.assertRaises(RuntimeError):
            verify_sealed_real(raw, sealed, 1)


if __name__ == '__main__':
    unittest.main()
