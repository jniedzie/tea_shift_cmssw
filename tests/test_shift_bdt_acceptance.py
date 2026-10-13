# Code style: AI-code. AI-generated regression test.
import importlib.util
from pathlib import Path
import unittest

import numpy as np


def module(name):
    path=Path(__file__).resolve().parents[1]/'utils'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


diagnostic=module('shift_bdt_acceptance')
statistics=module('shift_bdt_acceptance_statistics')


class AcceptanceTest(unittest.TestCase):
    def test_rejection_complements_counts_weights_and_intervals(self):
        kept=dict(empirical=dict(pairs=107,passed_pairs=37,efficiency=37/107,ci95_pair_wilson=[.2,.5]),
            weighted=dict(weight_sum=10.,selected_weight_sum=3.,efficiency=.3,signed_mc_acceptance_ratio=.3,ci95_event_bootstrap=[.1,.5]))
        rejected=diagnostic.complement(kept)
        self.assertEqual(rejected['empirical']['passed_pairs'],70)
        self.assertAlmostEqual(rejected['empirical']['efficiency'],70/107)
        self.assertEqual(rejected['empirical']['ci95_pair_wilson'],[.5,.8])
        self.assertEqual(rejected['weighted']['selected_weight_sum'],7.)
        self.assertAlmostEqual(rejected['weighted']['signed_mc_acceptance_ratio'],.7)
        self.assertEqual(kept['empirical']['passed_pairs'],37)

    def test_signed_rejection_keeps_probability_unavailable(self):
        rejected=diagnostic.complement(dict(weighted=dict(weight_sum=2.,selected_weight_sum=3.,efficiency=None,
            signed_mc_acceptance_ratio=1.5,ci95_event_bootstrap=[None,None],available=False)))['weighted']
        self.assertEqual(rejected['selected_weight_sum'],-1.)
        self.assertEqual(rejected['signed_mc_acceptance_ratio'],-.5)
        self.assertIsNone(rejected['efficiency'])
        self.assertFalse(rejected['available'])

    def test_topologies_and_other_are_exclusive(self):
        value=diagnostic.topology_names([0,0,2,2,1],[2,3,2,3,3])
        self.assertEqual(value.tolist(),['Near-Both','Near-Far','Both-Both','Both-Far','Other'])

    def test_exact_uint64_identity_parity(self):
        ids=np.asarray([[3,1,2**63+3]],dtype=np.uint64)
        diagnostic.assert_alignment({'ids':ids},{'ids':ids.copy()},['ids'])
        changed=ids.copy();changed[0,2]+=np.uint64(1)
        with self.assertRaises(ValueError):diagnostic.assert_alignment({'ids':changed},{'ids':ids},['ids'])

    def test_shapes_preserve_flow_and_signed_weights(self):
        h=diagnostic.shape_histogram([-1,.5,1.,2.],[True,False,True,True],[2.,-3.,4.,5.],[0.,1.,2.])
        self.assertEqual(h['before'],[2.,-3.,4.,5.])
        self.assertEqual(h['after'],[2.,0.,4.,5.])
        self.assertEqual(sum(h['before']),8.)
        self.assertEqual(h['after_sumw2'],[4.,0.,16.,25.])

    def test_shapes_reject_nonfinite_or_wrong_edges(self):
        with self.assertRaises(ValueError):diagnostic.shape_histogram([np.nan],[True],[1.],[0.,1.])
        with self.assertRaises(ValueError):diagnostic.shape_histogram([0.],[True],[1.],[1.,0.])

    def test_paired_identical_models_have_exact_zero_uncertainty(self):
        a=np.asarray([True,False,True])
        r=statistics.paired_acceptance_difference(a,a,[('dy',1),('dy',1),('dy',2)],repeats=100)
        self.assertEqual(r['difference'],0.)
        self.assertEqual(r['ci95_event_bootstrap'],[0.,0.])

    def test_pair_correlations_are_preserved(self):
        # One event contributes two positive differences; another one negative.
        r=statistics.paired_acceptance_difference([1,1,0],[0,0,1],[('dy',1),('dy',1),('dy',2)],repeats=200)
        self.assertAlmostEqual(r['difference'],1/3)
        self.assertEqual(r['events'],2)
        self.assertEqual(r['ci95_event_bootstrap'],[-1.,1.])
        self.assertEqual(r['first_only_pairs'],2)

    def test_signed_weights_do_not_define_probability(self):
        r=statistics.paired_acceptance_difference([1,0],[0,1],[('a',1),('a',2)],weights=[2.,-1.])
        self.assertFalse(r['probability_available'])
        self.assertIsNone(r['difference'])
        self.assertEqual(r['signed_difference_ratio'],3.)

    def test_empty_and_single_event_do_not_invent_interval(self):
        r=statistics.paired_acceptance_difference([],[],[])
        self.assertIsNone(r['difference'])
        r=statistics.paired_acceptance_difference([1],[0],[('qcd',1)])
        self.assertEqual(r['difference'],1.)
        self.assertEqual(r['ci95_event_bootstrap'],[None,None])

    def test_event_histogram_error_adds_correlated_pairs_first(self):
        variance=statistics.event_histogram_variance([0,0,0,1],[('a',1),('a',1),('a',2),('a',2)],[1.,2.,4.,5.],2)
        np.testing.assert_array_equal(variance,[25.,25.])

    def test_concentrated_weights_define_probability_without_support(self):
        r=statistics.paired_acceptance_difference([1]*30,[0]*30,
            [('dy',i) for i in range(30)],weights=[1000.]+[1.]*29,repeats=30)
        self.assertTrue(r['probability_available'])
        self.assertFalse(r['supported'])
        self.assertLess(r['event_weight_effective_n'],2.)
        self.assertEqual(r['ci95_event_bootstrap'],[1.,1.])

    def test_many_pairs_from_one_event_do_not_establish_support(self):
        r=statistics.paired_acceptance_difference([1]*30,[0]*30,[('dy',1)]*30)
        self.assertTrue(r['probability_available'])
        self.assertFalse(r['supported'])
        self.assertEqual(r['event_weight_effective_n'],1.)
        self.assertEqual(r['ci95_event_bootstrap'],[None,None])

    def test_support_floor_is_separate_from_bootstrap_availability(self):
        r=statistics.paired_acceptance_difference([1]*20,[0]*20,
            [('dy',i) for i in range(20)],repeats=0)
        self.assertTrue(r['supported'])
        self.assertEqual(r['support_minimum'],20)
        self.assertEqual(r['event_weight_effective_n'],20.)
        self.assertEqual(r['ci95_event_bootstrap'],[None,None])
        self.assertFalse(statistics.paired_support(19,20,20.,True))
        self.assertFalse(statistics.paired_support(20,19,20.,True))
        self.assertFalse(statistics.paired_support(20,20,19.9,True))
        self.assertFalse(statistics.paired_support(20,20,20.,False))
        with self.assertRaises(ValueError):
            statistics.paired_acceptance_difference([],[],[],minimum=0)


if __name__=='__main__':unittest.main()
