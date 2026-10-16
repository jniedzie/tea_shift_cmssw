# Code style: AI-code. AI-generated checks for signal MC-statistics accounting.
import importlib.util
from pathlib import Path
import unittest

import ROOT


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
        self.assertIsInstance(settings["rebin"], int)
        self.assertGreater(settings["rebin"], 0)
        self.assertLess(settings["minimum_cm"], settings["maximum_cm"])


if __name__ == "__main__":
    unittest.main()
