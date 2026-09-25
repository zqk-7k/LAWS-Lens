#!/usr/bin/env python3
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import mcwf_conditional_waveform_calibration_20260906 as c


class CalibrationTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'previous_waveform_score':[2.,2.,-1.],
            'new_encoder_cosine':[-10.,0.,10.],'new_mass_similarity':[-1.,-1.,-1.],
            'new_mass_predictive_BC':[.1,.1,.1],'waveform_embedding_cosine':[.8]*3,
            'waveform_abs_delta_logmc_std':[.2]*3,'waveform_abs_delta_logitq_std':[.1]*3})

    def test_neutral_mixture_bounds(self):
        f=self.frame()
        model={'kind':'nonnegative_logistic','center':np.zeros(4),'scale':np.ones(4),
               'coef':np.array([0.,1.,0.,0.]),'intercept':0.}
        for gamma in (.05,.5,.95):
            spec={'bounds':[[-100.]*4,[100.]*4],'calibrator_path':'mock','calibrator_sha256':'mock',
                  'gamma':gamma,'residual_mode':'neutral_mixture'}
            with patch.object(c,'load_model',return_value=model):
                score,pred,residual,ood=c.score(f,spec)
            self.assertTrue(np.isfinite(score).all())
            self.assertTrue((residual<=0).all())
            self.assertTrue((residual>=np.log1p(-gamma)-1e-12).all())
            self.assertTrue(np.allclose(score,f.previous_waveform_score+residual))
            self.assertEqual(score[2],f.previous_waveform_score.iloc[2])
            self.assertFalse(ood.any())

    def test_old_component_signs(self):
        x=c.design(self.frame(),c.EXTENDED_COLUMNS)
        np.testing.assert_array_equal(x[:,-2:],np.tile([-.2,-.1],(3,1)))

    def test_swapping_pair_endpoints_does_not_change_inputs(self):
        f=self.frame().assign(new_mass_pred_logmc_i=2.,new_mass_pred_logmc_j=3.)
        swapped=f.assign(new_mass_pred_logmc_i=f.new_mass_pred_logmc_j,
                         new_mass_pred_logmc_j=f.new_mass_pred_logmc_i)
        np.testing.assert_array_equal(c.design(f,c.EXTENDED_COLUMNS),c.design(swapped,c.EXTENDED_COLUMNS))


if __name__=='__main__':
    unittest.main()
