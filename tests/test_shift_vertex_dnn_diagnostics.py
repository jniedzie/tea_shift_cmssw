# Code style: AI-code. AI-generated frozen DNN diagnostic boundary tests.
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from shift_vertex_dnn_diagnostics import acceptance, checked, frozen_statistics, selected_models


class DnnDiagnosticBoundary(unittest.TestCase):
    def test_models_are_only_unique_validation_winners_and_fixed_references(self):
        lock = dict(selection_uses_test=False, rejection_winner='nn_a', flatness_winner='nn_a',
                    acceptance_envelope_winner=None, thresholds={'nn_unselected': .2})
        self.assertEqual(selected_models(lock), ['reference_uniform_bdt_3p0_s71', 'expanded_d3_s0_n400', 'nn_a'])
        lock['selection_uses_test'] = True
        with self.assertRaisesRegex(ValueError, 'Test-based'):
            selected_models(lock)

    def test_hash_tampering_fails_before_loading_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.npz'
            path.write_bytes(b'original')
            ref = {'path': str(path), 'sha256': hashlib.sha256(b'original').hexdigest()}
            self.assertEqual(checked(ref), path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed pinned'):
                checked(ref)

    def test_weight_probability_availability_is_not_statistical_support(self):
        source = Path(__file__).resolve().parents[2] / 'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier/bdt_performance.py'
        perf = frozen_statistics(source)
        passed = np.arange(40) < 30
        groups = [('dy', 1, 1, i) for i in range(40)]
        weights = np.ones(40)
        weights[0] = 1000
        result = acceptance(perf, passed, groups, weights, repeats=0)
        self.assertTrue(result['empirical']['supported'])
        self.assertTrue(result['weighted']['available'])
        self.assertFalse(result['weighted']['supported'])
        self.assertLess(result['weighted']['event_weight_effective_n'], 2)
        result = acceptance(perf, passed, [('dy', 1, 1, 1)] * 40, np.ones(40), repeats=0)
        self.assertFalse(result['empirical']['supported'])
        self.assertFalse(result['weighted']['supported'])

    def test_empty_genuine_population_has_no_efficiency_or_roc(self):
        source = Path(__file__).resolve().parents[2] / 'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier/bdt_performance.py'
        perf = frozen_statistics(source)
        result = acceptance(perf, np.array([], bool), [], np.array([]), repeats=0)
        self.assertIsNone(result['empirical']['efficiency'])
        self.assertIsNone(result['weighted']['efficiency'])
        roc = perf.roc_summary(np.zeros(30), np.linspace(0, 1, 30), [('qcd', 1, 1, i) for i in range(30)], repeats=0)
        self.assertIsNone(roc['auc'])
        self.assertFalse(roc['supported'])


if __name__ == '__main__':
    unittest.main()
