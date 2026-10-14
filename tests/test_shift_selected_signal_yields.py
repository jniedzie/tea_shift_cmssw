# Code style: AI-code. AI-generated frozen-response yield accounting tests.
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_baseline_sensitivity import _selected_signal_yield_rows, render_selected_signal_yields
from shift_vertex_efficiency import summarize_vertex_efficiency


class SelectedSignalYieldTest(unittest.TestCase):
    def fixture(self, zero=False, variable=False):
        ids = np.array([[17, 1, 2**60+i] for i in range(100)], dtype=np.uint64)
        weights = np.arange(1.,101.) if variable else np.ones(100)
        pair_ids = ids[[0, 1, 1, 2]]
        labels = np.array([1, 1, 1, 0], dtype=np.int8)
        scores = np.zeros(4) if zero else np.array([.8,.9,.7,.99])
        response = summarize_vertex_efficiency(ids, weights, pair_ids, labels, scores, .5)
        point = dict(mass_gev=12., proper_ctau_mm=10., full_GEN_events=100,
                     native_sumw=float(weights.sum()), native_sumw2=float((weights*weights).sum()),
                     full_GEN_event_ids_sha256="a"*64,gen_audit={"path":"audit.json","sha256":"b"*64})
        report = dict(threshold=.5,selection_uses_signal=False,
                      signal_points={"point":dict(point,response=response)})
        theory = dict(schema="shift-vertex-theory-yield-inputs-v1",complete=True,
                      branching_included_in_sigma=True,epsilon_correction_already_in_sigma=True,
                      full_GEN_events=100,fixed_classifier_cut=.5,
                      points={"point":dict(point,epsilon=1e-4,physical_selected_sigma_pb=2.,
                          physical_selected_sigma_stat_error_pb=.2,physical_inclusive_sigma_pb=20.,physical_br_mumu=.1)})
        return report,theory

    def test_full_gen_event_or_normalization_and_br_once(self):
        report,theory=self.fixture();result=_selected_signal_yield_rows(report,theory,3.)["point"]
        self.assertEqual(result["selected_qualified_signal_parents"],2)
        self.assertEqual(result["expected_selected_events"],120.)
        self.assertEqual(result["produced_dimuons"],6000.)
        self.assertEqual(result["rate_mc_standard_error_events"],12.)
        # Three passing genuine pairs share only two parents; fake pair excluded.
        self.assertEqual(report["signal_points"]["point"]["response"]["classes"]["genuine"]["selected_pairs"],3)

    def test_zero_mc_has_nonzero_exact95_upper(self):
        report,theory=self.fixture(zero=True);r=_selected_signal_yield_rows(report,theory,3.)["point"]
        self.assertEqual(r["expected_selected_events"],0.)
        self.assertAlmostEqual(r["selected_event_mc_interval95"][1],6000*(1-.05**.01))
        self.assertGreater(r["selected_event_mc_interval95"][1],0.)
        self.assertEqual(r["response_asymptotic_standard_error_events"],0.)

    def test_native_weighted_response_not_candidate_fraction(self):
        report,theory=self.fixture(variable=True);r=_selected_signal_yield_rows(report,theory,3.)["point"]
        self.assertAlmostEqual(r["expected_selected_events"],6000*3/5050)
        self.assertIsNone(r["selected_event_mc_interval95"])

    def test_reject_mismatched_gen_population(self):
        report,theory=self.fixture();theory["points"]["point"]["full_GEN_events"]=99
        with self.assertRaisesRegex(ValueError,"population count"):_selected_signal_yield_rows(report,theory,3.)

    def test_reject_swapped_gen_identities(self):
        report,theory=self.fixture();theory["points"]["point"]["full_GEN_event_ids_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"identities"):_selected_signal_yield_rows(report,theory,3.)

    def test_reject_weight_denominator_or_cut_change(self):
        report,theory=self.fixture();bad=copy.deepcopy(report)
        bad["signal_points"]["point"]["response"]["full_parent_selected"]["weighted"]["denominator_sumw"]=50.
        with self.assertRaisesRegex(ValueError,"denominator"):_selected_signal_yield_rows(bad,theory,3.)
        theory["fixed_classifier_cut"]=.6
        with self.assertRaisesRegex(ValueError,"cuts differ"):_selected_signal_yield_rows(report,theory,3.)

    def test_reject_unapplied_branching_convention(self):
        report,theory=self.fixture();theory["branching_included_in_sigma"]=False
        with self.assertRaisesRegex(ValueError,"already include"):_selected_signal_yield_rows(report,theory,3.)

    def test_existing_output_refused_before_inputs_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"existing"):
                render_selected_signal_yields("missing-response.json","missing-theory.json",directory)


if __name__ == "__main__":
    unittest.main()
