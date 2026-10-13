# Code style: AI-code. AI-generated regression test.
"""Physics/provenance boundaries of the retained-pair label audit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'utils/shift_bdt_label_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class LabelAuditTest(unittest.TestCase):
    def test_source_partition_handles_zero_pair_files_and_rejects_truncated_counts(self):
        sources = [{'pairs':0}, {'pairs':2}, {'pairs':0}, {'pairs':1}]
        self.assertEqual(audit.locate_source(sources, 2, 3), (3, 0))
        with self.assertRaises(ValueError):
            audit.locate_source(sources, 2, 4)
        with self.assertRaises(ValueError):
            audit.locate_source(sources, 3, 3)

    def test_sampling_correction_is_applied_exactly_once(self):
        self.assertEqual(audit.corrected_weight(1., .1, 10., 50., 100.), 5.)
        self.assertEqual(audit.corrected_weight(-1., .1, -10., 50., 100.), -5.)
        with self.assertRaises(ValueError):
            audit.corrected_weight(1., .1, 100., 50., 100.)
        with self.assertRaises(ValueError):
            audit.corrected_weight(1., 0., 10., 50., 100.)

    def test_ancestry_preserves_leg_identity_and_rejects_cycles(self):
        mothers = [-1, 0, 1, 1]
        self.assertEqual(audit.ancestor_indices(mothers, 2), [2, 1, 0])
        self.assertEqual(audit.ancestor_indices(mothers, 3), [3, 1, 0])
        with self.assertRaises(ValueError):
            audit.ancestor_indices([1, 0], 1)

    def test_common_origin_residual_is_not_a_new_label_or_fit_cut(self):
        reco = np.array([[0.,0.,100.], [0.,0.,999.]])
        truth = np.array([[[0.,0.,0.], [0.,0.,.004]], [[0.,0.,0.], [0.,0.,0.]]])
        labels = np.array([1,-1]); before = labels.copy()
        result = audit.residual_summary(reco, truth, labels, np.array(['qcd','qcd']))
        self.assertEqual(result['all']['finite_position_pairs'], 1)
        self.assertAlmostEqual(result['all']['signed_z_cm_quantiles_10_50_90'][1], 99.998)
        self.assertFalse(result['selection_applied'])
        np.testing.assert_array_equal(labels, before)

    def test_nonfinite_truth_is_explicitly_unavailable_for_residuals(self):
        report = audit.residual_summary(np.zeros((1,3)), np.full((1,2,3), np.nan), np.array([1]), np.array(['dy']))
        self.assertEqual(report['all']['nominal_genuine_pairs'], 1)
        self.assertEqual(report['all']['finite_position_pairs'], 0)
        self.assertIsNone(report['all']['distance_cm_quantiles_10_50_90'])

    def test_residuals_keep_quoted_uncertainty_and_frozen_cut_separate(self):
        report = audit.residual_summary(np.array([[0.,0.,-54.], [0.,0.,1.]]), np.zeros((2,2,3)),
            np.array([1,1]), np.array(['qcd','dy']), z_errors=np.array([33.,0.]), scores=np.array([.6,.2]), threshold=.44)
        uncertainty = report['all']['quoted_z_uncertainty']
        self.assertEqual(uncertainty['finite_positive_pairs'], 1)
        self.assertEqual(uncertainty['invalid_error_pairs'], 1)
        self.assertEqual(uncertainty['within_two_quoted_sigma'], 1)
        self.assertEqual(report['unchanged_benchmark_cut']['before']['nominal_genuine_pairs'], 2)
        self.assertEqual(report['unchanged_benchmark_cut']['after']['nominal_genuine_pairs'], 1)

    def test_angular_matches_do_not_promote_unknown_hit_labels(self):
        labels = dict(y=np.array([-1]), gen_index=np.array([[-1,-1]]),
            matched_layers=np.array([[2,5]]), match_purity=np.array([[1.,1.]]),
            sim_track_id=np.array([[-1,14]]), genPartIdx=np.array([[4.,5.]]))
        result = audit.optional_match_audit(labels, np.array(['qcd']))
        self.assertEqual(result['overall']['leg_states'], {'insufficient_layer_or_purity_gate':1, 'accepted_simtrack_without_gen_link':1})
        self.assertEqual(result['overall']['unknown_pairs_with_both_angle_matches'], 1)
        self.assertEqual(labels['y'][0], -1)


if __name__ == '__main__':
    unittest.main()
