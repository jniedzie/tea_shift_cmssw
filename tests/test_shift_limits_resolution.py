# Code style: AI-code. AI-generated invariance checks for fine likelihood bins.
import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from shift_limits_resolution import resolution_boundaries,interpolate_region
from shift_limits_profile import profile_q,poisson_deviance_array,poisson_deviance_half


class ResolutionTest(unittest.TestCase):
    def test_grid_follows_resolution_without_mc_counts(self):
        edges=resolution_boundaries()
        self.assertEqual((edges[0],edges[-1]),(0,1000))
        self.assertTrue(all(b>a for a,b in zip(edges,edges[1:])))
        self.assertTrue(all((b-a)/10<=max(.1,.02*a/10)+1.e-10 for a,b in zip(edges,edges[1:])))
        self.assertGreater(len(edges),100)

    def test_interpolation_preserves_rate_and_has_support(self):
        for power in [0.,1.,2.]:
            values=interpolate_region(1234.,117,1000,power)
            self.assertAlmostEqual(math.fsum(values),1234.,places=10)
            self.assertTrue(np.all(values>0))
        self.assertGreater(interpolate_region(1.,117,1000,2.)[0],interpolate_region(1.,117,1000,2.)[-1])

    def test_splitting_identical_shapes_cannot_improve_the_limit(self):
        coarse=profile_q(2.,[10.,20.],[1000.,2000.])
        fine=profile_q(2.,[5.,5.,10.,10.],[500.,500.,1000.,1000.])
        self.assertAlmostEqual(coarse,fine,places=10)

    def test_vectorized_deviance_keeps_small_signal_precision(self):
        deltas=np.array([-0.7,-1.e-6,1.e-10,1.e-4,.1,2.])
        expected=np.array([poisson_deviance_half(float(x)) for x in deltas])
        np.testing.assert_allclose(poisson_deviance_array(deltas),expected,rtol=1.e-12,atol=0.)


if __name__=='__main__':unittest.main()
