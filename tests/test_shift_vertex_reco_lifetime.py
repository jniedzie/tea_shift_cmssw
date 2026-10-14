# Code style: AI-code. AI-generated analytic reconstructed-length checks.
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from shift_vertex_reco_lifetime import reconstructed_proper_length_mm, reconstructed_vertex_distance_m


class ProperLengthTests(unittest.TestCase):
    names = ['pair_pt', 'pair_phi', 'pair_pz', 'pair_vx', 'pair_vy', 'pair_vz']

    def test_backward_beam_sign_units_and_negative_estimates(self):
        x = np.asarray([[0, 0, -100, 0, 0, 14700], [0, 0, -100, 0, 0, 14900]])
        np.testing.assert_allclose(reconstructed_proper_length_mm(x, self.names, np.asarray([10, 10]), (0,0,14800)), [100,-100])

    def test_transverse_projection_and_translation(self):
        x = np.asarray([[10, np.pi/2, 0, 100, 7, 3]])
        value = reconstructed_proper_length_mm(x, self.names, np.asarray([2]), (100,2,3))
        np.testing.assert_allclose(value, [10])
        x[:,3:] += 50
        np.testing.assert_allclose(reconstructed_proper_length_mm(x, self.names, np.asarray([2]), (150,52,53)), value)

    def test_zero_nonfinite_momentum_and_empty_are_explicit(self):
        x = np.asarray([[0,0,0,0,0,0], [np.nan,0,1,0,0,0]])
        self.assertTrue(np.isnan(reconstructed_proper_length_mm(x,self.names,np.ones(2),(0,0,0))).all())
        self.assertEqual(reconstructed_proper_length_mm(np.empty((0,6)),self.names,np.empty(0),(0,0,0)).shape,(0,))


class VertexDistanceTests(unittest.TestCase):
    names = ['pair_vx', 'pair_vy', 'pair_vz']

    def test_metres_and_three_dimensional_distance(self):
        x = np.asarray([[0,0,14800], [0,0,0], [300,400,14800], [0,0,-2000]])
        np.testing.assert_allclose(reconstructed_vertex_distance_m(x,self.names,(0,0,14800)),
                                   [0.,148.,5.,168.])

    def test_target_translation_and_invalid_coordinates(self):
        x = np.asarray([[3.,4.,0.], [np.nan,0.,0.]])
        value = reconstructed_vertex_distance_m(x,self.names,(0,0,0))
        self.assertAlmostEqual(value[0],.05)
        self.assertTrue(np.isnan(value[1]))
        np.testing.assert_allclose(reconstructed_vertex_distance_m(x+10,self.names,(10,10,10)),
                                   value,equal_nan=True)
        self.assertEqual(reconstructed_vertex_distance_m(np.empty((0,3)),self.names,(0,0,0)).shape,(0,))


if __name__ == '__main__':
    unittest.main()
