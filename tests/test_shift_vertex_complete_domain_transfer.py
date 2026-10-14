# Code style: AI-code. AI-generated fixed-cut transfer accounting tests.
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from shift_vertex_complete_domain_transfer import rates, selected, seed_range


class TransferAccountingTest(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 1, 0, 0])
        self.passed = np.array([True, False, True, False])

    def test_fake_metric_is_rejection(self):
        result = rates(self.y, np.array([True, True, False, False]), np.ones(4, dtype=bool))
        self.assertEqual(result['genuine_retention']['efficiency'], 1.)
        self.assertEqual(result['fake_rejection']['efficiency'], 1.)

    def test_process_mask_preserves_denominator(self):
        result = rates(self.y, self.passed, np.array([True, False, False, True]))
        for value in result.values():
            self.assertEqual(value['n'], 1)
            self.assertEqual(value['efficiency'], 1.)
            self.assertFalse(value['supported_n50'])

    def test_empty_class_is_undefined(self):
        result = rates(self.y, self.passed, self.y == 1)['fake_rejection']
        self.assertEqual(result['n'], 0)
        self.assertIsNone(result['efficiency'])
        self.assertIsNone(result['wilson95'])

    def test_sparse_interval_retains_uncertainty(self):
        result = rates(self.y, self.passed, np.ones(4, dtype=bool))['genuine_retention']
        low, high = result['wilson95']
        self.assertLess(low, .1)
        self.assertGreater(high, .9)
        self.assertEqual(result['accepted'], 1)

    def test_fullsize_locked_families_only(self):
        protocol = {'recipes': [{'family': 'a', 'fraction': 1.},
                                {'family': 'a', 'fraction': .5},
                                {'family': 'b', 'fraction': 1.}]}
        lock = {'winners': {'bdt': {'rejection': 'a', 'acceptance': None}}}
        self.assertEqual(selected(protocol, lock), [protocol['recipes'][0]])

    def test_duplicate_selection_roles_do_not_duplicate_models(self):
        protocol = {'recipes': [{'family': 'a', 'fraction': 1.}]}
        lock = {'winners': {'bdt': {'rejection': 'a', 'acceptance': 'a'}}}
        self.assertEqual(len(selected(protocol, lock)), 1)

    def test_identical_seed_range_roundoff_is_nonnegative(self):
        for value in (.9391891891891891, .7797716150081566, .1):
            center, lower, upper = seed_range([value, value, value])
            self.assertAlmostEqual(center, value)
            self.assertGreaterEqual(lower, 0.)
            self.assertGreaterEqual(upper, 0.)


if __name__ == '__main__':
    unittest.main()
