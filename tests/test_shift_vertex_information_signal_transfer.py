# Code style: AI-code. AI-generated transfer denominator and native-weight tests.
import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('transfer',Path(__file__).parents[1]/'utils/shift_vertex_information_signal_transfer.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SignalTransferTests(unittest.TestCase):
    def test_empty_support_not_efficiency_zero(self):
        self.assertIsNone(m.efficiency(0,0)['efficiency'])
    def test_signed_weights_and_unknowns_stay_separate(self):
        s=m.class_stats(np.array([.8,.2,.7,.9]),.5,np.array([1,1,0,-1]),np.array([2.,-1.,4.,3.]))
        self.assertEqual(s['genuine']['passed'],1);self.assertEqual(s['genuine']['total'],2)
        self.assertEqual(s['genuine']['native_sumw'],1.);self.assertEqual(s['genuine']['accepted_native_sumw'],2.)
        self.assertEqual(s['unknown']['total'],1);self.assertEqual(s['accidental']['passed'],1)
    def test_boundary_wilson_intervals(self):
        for passed in (0,20):
            r=m.efficiency(passed,20);self.assertLessEqual(r['wilson95'][0],r['efficiency']);self.assertGreaterEqual(r['wilson95'][1],r['efficiency'])
if __name__=='__main__':unittest.main()
