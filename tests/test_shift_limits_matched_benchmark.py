# Code style: AI-code. AI-generated checks of matched GEN/RECO diagnostic accounting.
import math
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from shift_limits_pheno_comparison import (
    ideal_pair_pointing,weighted_response_summary,benchmark_admission_probabilities,
    reference_counting_thresholds,fixed_response_coupling,source_grid_rows,
)


class MatchedBenchmarkTest(unittest.TestCase):
    def test_two_forward_muons_and_strict_annulus_energy_boundaries(self):
        good=[[2.,0.,-160.,40.],[-2.,0.,-160.,40.]]
        variants=[good,[[1.,0.,-160.,40.],good[1]],[[7.5,0.,-160.,40.],good[1]],
                  [[2.,0.,160.,40.],good[1]],[[2.,0.,-160.,20.],good[1]]]
        np.testing.assert_array_equal(ideal_pair_pointing(variants,20.),[True,False,False,False,False])
        self.assertTrue(ideal_pair_pointing(good))
        with self.assertRaises(ValueError):ideal_pair_pointing(good,inner_m=8.)

    def test_admission_indices_join_original_identity_without_event_number_assumption(self):
        ids=[[10,1,2],[10,1,5],[10,1,31]]
        self.assertEqual(benchmark_admission_probabilities(
            dict(selected_indices=[0,2],sampling_probabilities=[1.,.1]),ids),
            {(10,1,2):1.,(10,1,31):.1})
        for indices,probabilities in [([0,0],[1.,1.]),([3],[.1]),([1],[0.])]:
            with self.assertRaises(ValueError):benchmark_admission_probabilities(
                dict(selected_indices=indices,sampling_probabilities=probabilities),ids)

    def test_weighted_response_uses_full_original_denominator_and_event_variance(self):
        summary=weighted_response_summary([0.,10.,0.,0.])
        self.assertEqual(summary['efficiency'],2.5)  # An HT estimate must not be clipped.
        self.assertEqual(summary['standard_error'],2.5)
        self.assertEqual(summary['effective_events'],1.)
        empty=weighted_response_summary([0.,0.,0.])
        self.assertTrue(empty['zero_observed_response_unknown'])
        self.assertIsNone(empty['effective_events'])

    def test_background_proxy_and_zero_observation_are_distinct_prescriptions(self):
        low=reference_counting_thresholds(2.,300.)
        high=reference_counting_thresholds(2.,3000.)
        self.assertAlmostEqual(high['smooth_background_events']/low['smooth_background_events'],10.)
        self.assertAlmostEqual(high['SensCalc_style_one_sided_Gaussian90_proxy']/
                               low['SensCalc_style_one_sided_Gaussian90_proxy'],math.sqrt(10.))
        self.assertAlmostEqual(math.exp(-low['exact_zero_observation_Poisson90']),.1)
        self.assertEqual(fixed_response_coupling(3.,300.,100.,.01),.1)
        self.assertIsNone(fixed_response_coupling(3.,300.,100.,0.))
        with self.assertRaises(ValueError):fixed_response_coupling(3.,300.,100.,float('nan'))


class SourceGridAccountingTest(unittest.TestCase):
    def row(self):
        return dict(mass_gev=.25,meson_fragmentation={'channels':{'eta-decay':{'sigma_over_epsilon_squared_pb':8.}}},
                    FR_sigma_over_epsilon_squared_pb=2.,inclusive_sigma_over_epsilon_squared_pb=10.,
                    BR_mumu=.4,dimuon_sigma_over_epsilon_squared_pb=4.)

    def test_inclusive_source_closure_prevents_double_counted_mechanisms(self):
        row=self.row()
        self.assertEqual(source_grid_rows({'masses':[row]}),[row])
        bad=copy.deepcopy(row);bad['inclusive_sigma_over_epsilon_squared_pb']=18.
        with self.assertRaisesRegex(ValueError,'channels do not close'):
            source_grid_rows({'masses':[bad]})

    def test_branching_once_and_ordered_mass_support(self):
        bad=self.row();bad['dimuon_sigma_over_epsilon_squared_pb']*=bad['BR_mumu']
        with self.assertRaisesRegex(ValueError,'exactly once'):source_grid_rows({'masses':[bad]})
        with self.assertRaisesRegex(ValueError,'strictly increasing'):
            source_grid_rows({'masses':[self.row(),self.row()]})


if __name__=='__main__':unittest.main()
