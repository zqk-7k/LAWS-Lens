import unittest
import numpy as np
from tail_audit import fit_tail, survival, factor_range


class TailTests(unittest.TestCase):
    def test_exponential_recovery(self):
        rng=np.random.default_rng(220922)
        x=rng.exponential(2.,50000)
        model=fit_tail(x,np.quantile(x,.9))
        self.assertLess(abs(model['shape']),.06)
        self.assertLess(abs(model['scale']-2),.2)

    def test_threshold_normalization(self):
        x=np.random.default_rng(4).exponential(size=1000)
        u=np.quantile(x,.9);m=fit_tail(x,u)
        self.assertAlmostEqual(survival(m,[u])[0],np.mean(x>u))

    def test_monotonic_and_bounded_endpoint(self):
        m=dict(threshold=1.,shape=-.2,scale=1.,tail_fraction=.1)
        y=survival(m,np.linspace(1,7,100))
        self.assertTrue(np.all(np.diff(y)<=0))
        self.assertEqual(y[-1],0.)

    def test_no_fit_without_support(self):
        self.assertIsNone(fit_tail(np.arange(100),95.))
        self.assertIsNone(fit_tail(np.ones(100),.5))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):fit_tail([1,np.nan],0)
        with self.assertRaises(ValueError):
            survival(dict(threshold=1.,shape=0.,scale=1.,tail_fraction=.1),[0.])

    def test_zero_not_hidden_in_stability(self):
        self.assertEqual(factor_range([[.1,.01],[.2,0]]),np.inf)
        self.assertAlmostEqual(factor_range([[.1,.01],[.2,.02]]),2.)

    def test_pair_dependency_deleted_by_nodes(self):
        i,j=np.triu_indices(6,1)
        mask=(i!=2)&(j!=2)
        self.assertEqual(mask.sum(),10)
        self.assertFalse(np.any((i[mask]==2)|(j[mask]==2)))


if __name__=='__main__':unittest.main()
