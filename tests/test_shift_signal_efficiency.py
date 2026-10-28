# Code style: AI-code. AI-generated checks for signal MC-statistics accounting.
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from array import array

import ROOT
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "utils/shift_signal_efficiency.py"
SPEC = importlib.util.spec_from_file_location("signal_efficiency", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class RootObjects:
    def __init__(self, objects):
        self.objects = objects

    def Get(self, name):
        return self.objects.get(name)


def histogram(name, values, lower=0, upper=None):
    hist = ROOT.TH1D(name, "", len(values), lower, upper or len(values))
    hist.SetDirectory(0)
    hist.Sumw2()
    for index, count in enumerate(values, 1):
        for _ in range(count):
            hist.Fill(hist.GetBinCenter(index))
    return hist


class EfficiencyDiagnosticTest(unittest.TestCase):
    def test_weighted_response_counts_events_and_preserves_full_gen(self):
        result = DIAGNOSTIC.weighted_response_statistics([.5, .1, 1.], np.array([False, True, True]), 10)
        self.assertEqual(result['raw_passing_events'], 2)
        self.assertEqual(result['passing_sumw'], 11.)
        self.assertEqual(result['passing_sumw2'], 101.)
        self.assertAlmostEqual(result['effective_passing_events'], 121. / 101.)
        self.assertEqual(result['original_population_efficiency'], 1.1)  # Do not clip a noisy HT estimate.
        self.assertIsNone(result['projected_total_gen_events'])
        zero = DIAGNOSTIC.weighted_response_statistics([.1, 1.], np.array([False, False]), 10)
        self.assertIsNone(zero['poisson_mc_standard_error'])
        self.assertEqual(zero['status'], 'bounded_followup_required')
        stable = DIAGNOSTIC.weighted_response_statistics([1.] * 100, np.ones(100, dtype=bool), 200)
        self.assertEqual(stable['projected_total_gen_events'], 6000)

    def test_weighted_response_rejects_bad_probabilities_and_exposure(self):
        for probability in (0., 1.1, float('nan')):
            with self.assertRaises(ValueError):
                DIAGNOSTIC.weighted_response_statistics([probability], np.array([True]), 2)
        with self.assertRaises(ValueError):
            DIAGNOSTIC.weighted_response_statistics([1., 1.], np.array([True, True]), 1)

    def test_grouped_nano_response_agrees_with_weighted_histograms(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            (campaign / 'nanoAOD_merged').mkdir()
            (campaign / 'histograms_merged').mkdir()
            root = ROOT.TFile(str(campaign / 'nanoAOD_merged/ntuple_0.root'), 'RECREATE')
            tree = ROOT.TTree('Events', '')
            integers = {name: array('i', [1]) for name in ('run', 'luminosityBlock', 'event', 'nShiftDimuonVertex')}
            doubles = {name: array('d', [1.]) for name in ('shiftOriginalGenWeight', 'shiftSamplingProbability', 'shiftSamplingGenWeight')}
            vectors = {name: ROOT.std.vector('int')() for name in ('ShiftDimuonVertex_topologyMin', 'ShiftDimuonVertex_topologyMax')}
            for name, value in integers.items(): tree.Branch(name, value, name + '/I')
            for name, value in doubles.items(): tree.Branch(name, value, name + '/D')
            for name, value in vectors.items(): tree.Branch(name, value)
            for i, probability in enumerate((.5, 1.)):
                integers['event'][0] = i + 1
                integers['nShiftDimuonVertex'][0] = i + 1
                doubles['shiftSamplingProbability'][0] = probability
                doubles['shiftSamplingGenWeight'][0] = 1. / probability
                for vector in vectors.values():
                    vector.clear()
                    for j in range(i + 1): vector.push_back(2)
                tree.Fill()
            tree.Write(); root.Close()
            path = campaign / 'histograms_merged/ntuple_0.root'
            root = ROOT.TFile(str(path), 'RECREATE')
            ROOT.TObjString(json.dumps(dict(denominator_kind='conditional_signal_events', full_source_sumw=10,
                histogram_weight='native_gen_weight/reco_inclusion_probability'))).Write('ShiftFullGenNormalization')
            root.mkdir('event').cd()
            for category in ('', 'Both-Both'):
                hist = ROOT.TH1D('Event_nShiftDimuonVertex' + category, '', 10, 0, 10)
                hist.Fill(1, 2.); hist.Fill(2, 1.); hist.Write()
            root.Close()
            result = DIAGNOSTIC.measure_weighted_histogram(path)
            self.assertEqual(result['both_both']['raw_passing_events'], 2)
            self.assertEqual(result['both_both']['passing_sumw'], 3.)
            self.assertEqual(result['both_both']['passing_sumw2'], 5.)
            self.assertEqual(result['both_both']['original_population_efficiency'], .3)

    def setUp(self):
        ROOT.gROOT.SetBatch(True)
        self.settings = dict(rebin=1, minimum_cm=0, maximum_cm=10)

    def shape(self, values, generated=100, relative_error=0.1, multiple=0):
        hist = histogram("shape", values)
        source = RootObjects({"dimuon/ShiftDimuonVertexBoth-Both_vz": hist})
        counts = dict(generated_events=generated, multiple_dimuon_events=multiple)
        return DIAGNOSTIC.occupancy_plan(source, counts, self.settings, relative_error, .9)

    def test_zero_observations_preserve_unknown_acceptance(self):
        low, high = DIAGNOSTIC.interval(0, 100, .95)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0)
        self.assertLess(high, .05)

    def test_efficiency_counts_events_rather_than_pairs(self):
        hist = histogram("multiplicity", [92, 7, 1, 0])
        source = RootObjects({"event/Event_nShiftDimuonVertexBoth-Both": hist})
        counts = DIAGNOSTIC.count_events(source, "Both-Both")
        self.assertEqual(counts["generated_events"], 100)
        self.assertEqual(counts["selected_events"], 8)
        self.assertEqual(counts["multiple_dimuon_events"], 1)

    def test_weighted_inputs_cannot_claim_binomial_efficiencies(self):
        hist = histogram("weighted", [1, 0])
        hist.Fill(1.5, 2.)
        with self.assertRaisesRegex(ValueError, "unit weights"):
            DIAGNOSTIC.unit_count_histogram(RootObjects({"h": hist}), "h")

    def test_no_visible_yield_requires_pilot_not_infinite_production(self):
        plan = self.shape([0, 0, 0])
        self.assertIsNone(plan["conservative_target_total_events"])
        self.assertGreaterEqual(plan["pilot_total_events"], 10000)
        self.assertGreater(plan["visible_efficiency_upper_95"], 0)

    def test_multiple_pairs_require_event_correlations(self):
        with self.assertRaisesRegex(ValueError, "covariance"):
            self.shape([10, 10], multiple=1)

    def test_tighter_bin_error_needs_more_events(self):
        loose = self.shape([10] * 10, generated=1000, relative_error=.2)
        tight = self.shape([10] * 10, generated=1000, relative_error=.05)
        self.assertGreater(tight["conservative_target_total_events"], loose["conservative_target_total_events"])

    def test_sparse_shape_cannot_be_defined_by_only_two_occupied_bins(self):
        plan = self.shape([0, 1, 0, 1], generated=500)
        self.assertGreaterEqual(plan["conservative_target_total_events"], 250000)
        self.assertLess(plan["pilot_total_events"], plan["conservative_target_total_events"])
        self.assertEqual(plan["status"], "pilot_first_sparse_shape")

    def test_rare_outlier_does_not_set_the_dominant_shape_budget(self):
        plan = self.shape([1, 0, 99, 0, 0], generated=1000)
        self.assertEqual(plan["core_bin_indices"], [2])
        self.assertEqual(plan["bin_events"], [1, 0, 99, 0, 0])
        self.assertEqual(plan["unobserved_visible_bin_indices"], [1, 3, 4])
        self.assertLess(plan["conservative_target_total_events"], 20000)

    def test_missing_credit_inventory_does_not_invent_pending_events(self):
        row = dict(point="test", both_both=dict(generated_events=1000),
                   vertex_z_plan=self.shape([10, 10], generated=1000))
        DIAGNOSTIC.apply_credit([row], None)
        self.assertEqual(row["production_credit"]["completed_total_events"], 1000)
        self.assertEqual(row["production_credit"]["independent_inflight_events"], 0)
        self.assertEqual(row["vertex_z_plan"]["planned_next_stage_additional_events"],
                         row["vertex_z_plan"]["planned_next_stage_total_events"] - 1000)

    def test_active_z_binning_is_read_from_existing_plot(self):
        settings = DIAGNOSTIC.plot_settings()
        self.assertEqual(settings["rebin"], 10)
        self.assertIsNone(settings["minimum_cm"])
        self.assertIsNone(settings["maximum_cm"])

    def test_automatic_limits_use_the_recorded_histogram_domain(self):
        self.settings.update(minimum_cm=None, maximum_cm=None)
        result = self.shape([1, 2, 3])
        self.assertEqual(result["visible_selected_events"], 6)
        self.assertEqual(result["edges_cm"], [0., 1., 2., 3.])

    def test_equal_passing_targets_scale_with_efficiency(self):
        efficient = DIAGNOSTIC.event_statistics_plan(dict(
            generated_events=1000, selected_events=100, efficiency_interval_95=[.08, .12]))
        inefficient = DIAGNOSTIC.event_statistics_plan(dict(
            generated_events=1000, selected_events=30, efficiency_interval_95=[.02, .04]))
        self.assertEqual(efficient["nominal_target_total_events"], 30000)
        self.assertEqual(inefficient["nominal_target_total_events"], 100000)
        self.assertGreater(inefficient["conservative_target_total_events"], 100000)

    def test_zero_passes_have_finite_pilot_and_no_invented_final_target(self):
        plan = DIAGNOSTIC.event_statistics_plan(dict(
            generated_events=100, selected_events=0, efficiency_interval_95=[0., .0362]))
        self.assertIsNone(plan["nominal_target_total_events"])
        self.assertIsNone(plan["conservative_target_total_events"])
        self.assertEqual(plan["planned_next_stage_total_events"], 10100)
        self.assertEqual(plan["status"], "zero_pass_pilot")

    def test_sparse_event_measurement_is_remeasured_before_full_allocation(self):
        plan = DIAGNOSTIC.event_statistics_plan(dict(
            generated_events=1320, selected_events=3, efficiency_interval_95=[.0004, .0066]))
        self.assertEqual(plan["nominal_target_total_events"], 1320000)
        self.assertEqual(plan["pilot_total_events"], 44000)
        self.assertLess(plan["planned_next_stage_total_events"], plan["nominal_target_total_events"])

    def test_event_allocation_credits_pending_independent_events(self):
        import json
        import tempfile
        counts = dict(generated_events=1000, selected_events=100, efficiency_interval_95=[.08, .12])
        row = dict(point="p", both_both=counts, vertex_z_plan=self.shape([10, 10], generated=1000),
                   event_statistics_plan=DIAGNOSTIC.event_statistics_plan(counts))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credit.json"
            path.write_text(json.dumps({"points": {"p": {"completed_total_events": 2000,
                                                       "independent_inflight_events": 3000}}}))
            DIAGNOSTIC.apply_credit([row], path)
        self.assertEqual(row["event_statistics_plan"]["additional_events_after_completed"], 3000)
        self.assertEqual(row["event_statistics_plan"]["additional_events_if_inflight_succeeds"], 0)
        self.assertTrue(row["event_statistics_plan"]["credit_inventory_supplied"])

    def test_new_model_credit_does_not_reuse_old_histogram_population(self):
        import json
        import tempfile
        counts = dict(generated_events=1000, selected_events=100, efficiency_interval_95=[.08, .12])
        row = dict(point="p", both_both=counts, vertex_z_plan=self.shape([10, 10], generated=1000),
                   event_statistics_plan=DIAGNOSTIC.event_statistics_plan(counts))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credit.json"
            path.write_text(json.dumps({"points": {"p": {"completed_total_events": 200,
                "reference_histogram_is_allocation_population": False}}}))
            DIAGNOSTIC.apply_credit([row], path)
        self.assertEqual(row["production_credit"]["completed_total_events"], 200)
        self.assertEqual(row["event_statistics_plan"]["additional_events_after_completed"], 4800)

    def test_partial_processing_is_not_detector_inefficiency(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "histograms_merged").mkdir()
            (campaign / "cross_sections.json").write_text(json.dumps({"full_gen": {
                "accepted_events": 1000, "sum_weights": 1000, "sum_weights_squared": 1000,
                "source_collisions": 500000}}))
            result = DIAGNOSTIC.normalization_context(campaign / "histograms_merged/h.root", 100, 10)
        self.assertEqual(result["processing_fraction"], .1)
        self.assertEqual(result["selected_per_full_gen_available"], .01)
        self.assertEqual(result["source_collisions"], 500000)

    def test_storage_round_respects_cap_and_prioritizes_long_flights(self):
        rows = [dict(point=name, mass_gev=.3, mean_lab_flight_m=flight,
                     both_both=dict(efficiency=.01, selected_events=10),
                     production_credit=dict(completed_total_events=0, independent_inflight_events=0))
                for name, flight in (("short", 5), ("long", 1000))]
        densities = {"points": {name: dict(reserved_bytes_per_gen=100., fixed_bytes_per_point=0.)
                                for name in ("short", "long")}}
        plan = DIAGNOSTIC.storage_capped_round(rows, densities, 50000, base_passes=1, priority_passes=10)
        allocated = {p["point"]: p for p in plan["points"]}
        self.assertLessEqual(plan["reserved_bytes"], 50000)
        self.assertGreater(allocated["long"]["additional_gen_events"], allocated["short"]["additional_gen_events"])
        self.assertFalse(plan["production_admitted"])

    def test_storage_round_never_credits_old_population_or_invents_zero_acceptance(self):
        row = dict(point="zero", mass_gev=.3, mean_lab_flight_m=1000,
                   both_both=dict(efficiency=0., selected_events=0),
                   production_credit=dict(completed_total_events=200, independent_inflight_events=0))
        plan = DIAGNOSTIC.storage_capped_round([row], {"points": {"zero": {
            "reserved_bytes_per_gen": 100., "fixed_bytes_per_point": 1000.}}}, 10000)
        self.assertEqual(plan["additional_gen_events"], 0)
        self.assertEqual(plan["points"][0]["allocation_status"], "zero_efficiency_requires_separate_exploration")


if __name__ == "__main__":
    unittest.main()
