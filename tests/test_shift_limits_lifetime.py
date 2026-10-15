# Code style: AI-code. AI-generated checks against analytic decay probabilities.
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('lifetime',Path(__file__).parents[1]/'utils'/'shift_limits_lifetime.py')
lifetime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifetime)


class LifetimeWeightingTest(unittest.TestCase):
    def test_survival_probability_keeps_full_generated_denominator(self):
        # Midpoint quadrature of a known source exponential. At the target
        # lifetime the exact selected probability is exp(-threshold/target).
        source,target,threshold=10.,2.,1.
        quantiles=(np.arange(8192)+.5)/8192
        lengths=-source*np.log1p(-quantiles)
        data={'proper_length_mm':lengths,'category':np.where(lengths>threshold,0,-1)}
        weights=lifetime.lifetime_weights(data,source,target)
        estimate=lifetime.acceptance(data,weights)
        self.assertAlmostEqual(float(weights.mean()),1.,delta=1.e-6)
        ratio=target/source
        self.assertAlmostEqual(float(np.square(weights).mean()),1/(ratio*(2-ratio)),delta=1.e-6)
        self.assertAlmostEqual(estimate['value'],np.exp(-threshold/target),delta=.001)
        self.assertLess(estimate['selected_ess'],len(lengths))

    def test_same_lifetime_matches_binomial_acceptance(self):
        data={'proper_length_mm':np.array([.1,1.,3.,10.]),'category':np.array([-1,0,-1,1])}
        weights=lifetime.lifetime_weights(data,2.,2.)
        self.assertTrue(np.array_equal(weights,np.ones(4)))
        estimate=lifetime.acceptance(data,weights)
        self.assertEqual(estimate['value'],.5)
        self.assertEqual(estimate['selected_ess'],2.)
        self.assertAlmostEqual(estimate['error'],np.sqrt(.25/3))


if __name__=='__main__':unittest.main()
