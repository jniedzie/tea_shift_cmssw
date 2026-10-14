# Code style: AI-code. AI-generated independent signal-map accounting tests.
"""Synthetic full-GEN corruption and lifetime-estimator checks; no ROOT reads."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "tea_shift_cmssw/utils"))
sys.path.insert(0, str(WORKSPACE / "shift_cmssw_workflow/scripts"))

from dark_photon_model import HBAR_C_GEV_MM
from dark_photon_scan import LifetimeEvent, LifetimeProposal, estimate_lifetime_acceptance
from shift_vertex_efficiency import summarize_vertex_efficiency
from shift_vertex_efficiency_map import align_gen, event_masks, precision_forecast, render


class SignalMapTests(unittest.TestCase):
    def test_prompt_only_zero_candidate_response_can_be_reported(self):
        stats = summarize_vertex_efficiency(
            self.ids, self.weights, np.empty((0, 3), dtype=np.uint64),
            np.empty(0, dtype=np.int8), np.empty(0), .5)
        name = 'info_global_rich_s71'
        report = dict(models=[name], points={'prompt': dict(
            mass_gev=15., proper_ctau_mm=1e-7, response_available=True,
            models={name: stats})})
        with tempfile.TemporaryDirectory() as directory:
            render(report, Path(directory))
            for filename in ('mass_proper_lifetime_response.pdf',
                             'realized_proper_length_profiles.pdf', 'local_lifetime_reweighting.pdf'):
                self.assertGreater((Path(directory) / filename).stat().st_size, 1000)
        self.assertEqual(stats['full_parent_selected']['empirical']['efficiency'], 0.)
        self.assertIsNone(stats['conditional_event_retention']['empirical']['efficiency'])

    def setUp(self):
        self.ids = np.asarray([[9, 2, 2**53 + i] for i in range(4)], dtype=np.uint64)
        self.weights = np.ones(4)
        self.inventory = dict(events=4, event_ids=self.ids.tolist(), native_sumw=4.,
                              native_sumw2=4., proper_ctau_mm=100.)
        self.audit = dict(events=4, runtime_validated=True, sum_weights=4.,
                          sum_weights_squared=4., expected_proper_length_mm=100.,
                          event_rows=[dict(id=identity.tolist(), weight=1.,
                              signal_decays=[dict(proper_length_mm=length, mass_gev=15.)])
                              for identity, length in zip(self.ids, [0., 25., 100., 400.])])
        self.contract = dict(width_authority=dict(ctau_mm=100.), filtering="none",
                             generated_mass_support_gev=[14., 16.])

    def aligned(self, inventory=None, audit=None, contract=None):
        return align_gen("synthetic", inventory or self.inventory, audit or self.audit,
                         contract or self.contract, self.ids, self.weights)

    def test_exact_large_integer_id_alignment(self):
        np.testing.assert_array_equal(self.aligned(), [0., 25., 100., 400.])
        changed = copy.deepcopy(self.inventory)
        changed["event_ids"][1][2] += 1
        with self.assertRaisesRegex(ValueError, "population"):
            self.aligned(inventory=changed)

    def test_incomplete_reordered_or_unvalidated_audit_rejected(self):
        cases = []
        audit = copy.deepcopy(self.audit)
        audit["event_rows"] = audit["event_rows"][:-1]
        cases.append(audit)
        audit = copy.deepcopy(self.audit)
        audit["event_rows"].reverse()
        cases.append(audit)
        audit = copy.deepcopy(self.audit)
        audit["runtime_validated"] = False
        cases.append(audit)
        for audit in cases:
            with self.subTest(audit=audit), self.assertRaisesRegex(ValueError, "population"):
                self.aligned(audit=audit)

    def test_native_weight_and_denominator_corruption_rejected(self):
        for key in ("sum_weights", "sum_weights_squared"):
            audit = copy.deepcopy(self.audit)
            audit[key] -= .1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unit weights"):
                self.aligned(audit=audit)
        audit = copy.deepcopy(self.audit)
        audit["event_rows"][0]["weight"] = .5
        with self.assertRaisesRegex(ValueError, "unit weights"):
            self.aligned(audit=audit)
        for key in ("native_sumw", "native_sumw2"):
            inventory = copy.deepcopy(self.inventory)
            inventory[key] = 3.
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unit weights"):
                self.aligned(inventory=inventory)

    def test_decay_and_generation_contract_corruption_rejected(self):
        for field, value in (("proper_length_mm", -1.), ("proper_length_mm", np.nan),
                             ("mass_gev", 17.)):
            audit = copy.deepcopy(self.audit)
            audit["event_rows"][0]["signal_decays"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.aligned(audit=audit)
        audit = copy.deepcopy(self.audit)
        audit["event_rows"][0]["signal_decays"] *= 2
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            self.aligned(audit=audit)
        for contract in (dict(self.contract, filtering="muon_filter"),
                         dict(self.contract, width_authority=dict(ctau_mm=101.))):
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, "contract"):
                self.aligned(contract=contract)

    def test_parent_or_accounting_keeps_zero_candidate_and_unknown_parents(self):
        pair_ids = self.ids[[0, 0, 1, 2]]
        labels = np.asarray([1, 1, 0, -1])
        scores = np.asarray([.5, .9, .99, .99])
        result = summarize_vertex_efficiency(self.ids, self.weights, pair_ids, labels, scores, .5)
        baseline, selected = event_masks(self.ids, pair_ids, labels == 1, scores, .5)
        np.testing.assert_array_equal(baseline, [True, False, False, False])
        np.testing.assert_array_equal(selected, baseline)
        self.assertEqual(result["counts"]["parents_without_candidates"], 1)
        self.assertEqual(result["counts"]["parents_selected_by_genuine_pair"], 1)
        self.assertEqual(result["full_parent_selected"]["empirical"]["efficiency"], .25)
        self.assertEqual(result["conditional_event_retention"]["empirical"]["efficiency"], 1.)
        self.assertEqual(result["classes"]["unknown"]["selected_pairs"], 1)
        self.assertIsNone(result["genuine_pair_retention"]["empirical"]["wilson95"])

    def test_duplicate_or_orphan_parents_fail_closed(self):
        duplicate = self.ids.copy()
        duplicate[1] = duplicate[0]
        for parents, pairs, reason in ((duplicate, self.ids[:1], "Duplicate"),
                                       (self.ids[:3], self.ids[3:], "Orphan")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                summarize_vertex_efficiency(parents, np.ones(len(parents)), pairs,
                                            np.asarray([1]), np.asarray([.9]), .5)

    def test_single_anchor_identity_full_denominator_and_support(self):
        proposal = LifetimeProposal((100.,), (4,), "synthetic-fixed-context")
        events = [LifetimeEvent(str(i), length, 0, i < 2, proposal.context_id)
                  for i, length in enumerate(self.aligned())]
        result = estimate_lifetime_acceptance(proposal, events, HBAR_C_GEV_MM / 100., 4)
        self.assertAlmostEqual(result["acceptance"]["estimate"], .5)
        self.assertAlmostEqual(result["acceptance"]["effective_sample_size"], 2.)
        self.assertAlmostEqual(result["density_normalization"]["estimate"], 1.)
        self.assertEqual(result["proposal"]["counts"], (4,))
        self.assertEqual(result["precision_goals"]["min_selected_ess"], 100)
        self.assertFalse(result["precision_goals_passed"])
        self.assertFalse(result["physical_sensitivity_validated"])
        with self.assertRaisesRegex(ValueError, "full frozen GEN ledger"):
            estimate_lifetime_acceptance(proposal, events[:-1], HBAR_C_GEV_MM / 100., 4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            estimate_lifetime_acceptance(proposal, events[:3] + events[:1], HBAR_C_GEV_MM / 100., 4)

    def test_boundary_precision_forecast_is_unknown_not_zero(self):
        self.assertIsNone(precision_forecast(.1, 1.)["genuine_for_retention_95_halfwidth"])
        self.assertIsNone(precision_forecast(0., None)["gen_for_relative_response_se"])


if __name__ == "__main__":
    unittest.main()
