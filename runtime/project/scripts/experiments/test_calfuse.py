"""Regression tests for statistics, support handling and blocked data split."""
import unittest
import numpy as np
import pandas as pd
import calfuse as c


class TestCalfuse(unittest.TestCase):
    def fixture(self):
        i,j=np.triu_indices(12,1)
        y=(j==i+1)&(i%2==0)
        f=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':y.astype(int),
            'true_pair_family':np.where(y,np.where(i<6,'SIS','PM'),'background'),
            'event_count':12})
        p=pd.DataFrame({'idx':np.arange(12),'system_id':[f'g{k//2}' for k in range(12)],
            'parent_noise_bank':np.arange(12)//2})
        return f,p

    def test_historical_metrics_exact_including_ties(self):
        f,p=self.fixture()
        for scores in (np.random.default_rng(77).normal(size=len(f)),np.zeros(len(f)),
                       np.round(np.random.default_rng(88).normal(size=len(f)))):
            a,b=c.fast_metrics(f,scores),c.dev.BASE.full_metrics(f,scores)
            for k in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                self.assertAlmostEqual(a[k],b[k],places=12)

    def test_grid_deduplicates_and_includes_original(self):
        for dep in c.DEPS:
            for seed in c.SEEDS:
                grid=c.weight_grid(dep,seed)
                self.assertTrue(np.allclose(grid.sum(1),1))
                self.assertTrue((grid>=0).all())
                self.assertEqual(len({tuple(np.round(w,12)) for w in grid}),len(grid))
                old=c.frozen_weights(dep,seed);old/=old.sum()
                self.assertTrue(np.any(np.max(abs(grid-old),axis=1)<1e-12))

    def test_blocking_removes_whole_cross_noise_sources(self):
        f,p=self.fixture()
        p.loc[1,'parent_noise_bank']=5
        out=c.split_validation(p,202607241)
        fit=out[out.calibration_fold==0];tune=out[out.calibration_fold==1]
        self.assertFalse(set(fit.system_id)&set(tune.system_id))
        self.assertFalse(set(fit.parent_noise_bank)&set(tune.parent_noise_bank))
        self.assertTrue((out.groupby('system_id').calibration_fold.nunique()==1).all())

    def test_balanced_class_weights(self):
        f,p=self.fixture();w=c.balanced_weights(f,p)
        self.assertAlmostEqual(w[f.is_true_pair==1].sum(),.5)
        self.assertAlmostEqual(w[f.is_true_pair==0].sum(),.5)

    def test_positive_fit_and_ood_fallback(self):
        f,p=self.fixture();f['waveform_score']=f.is_true_pair*2-1.
        spec=c.fit_logit(f,p,'AFFINE',.1)
        self.assertGreater(spec['coef'][0],0)
        out=f.copy();out.loc[0,'waveform_score']=100.
        score,ood,clipped=c.apply(out,spec)
        self.assertTrue(ood[0]);self.assertEqual(score[0],100.)
        self.assertFalse(clipped[0])
        self.assertTrue(np.isfinite(score).all())

    def test_tie_rule_uses_distance_then_lexicographic(self):
        old=np.array([.5,.25,.5]);w=old/old.sum()
        metric={'guard_pass':True,'false_at_recall_0p5':2,'false_at_recall_0p9':4,
            'average_precision':.5,'macro_r_at_10':.8,'macro_r_at_1':.4}
        a={**metric,'weights':w.tolist()};b={**metric,'weights':[1.,0.,0.]}
        self.assertEqual(c.select_grid([b,a],'VF50',old)['weights'],w.tolist())


if __name__=='__main__':unittest.main(verbosity=2)
