# Code style: AI-code. AI-generated independent counting and exact-CLs checks.
import importlib.util
from array import array
import math
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("baseline", Path(__file__).resolve().parents[1] /
                                           "utils/shift_baseline_sensitivity.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class BaselineCountingTest(unittest.TestCase):
    def test_one_event_selects_only_one_candidate_and_ties_use_index(self):
        self.assertEqual(baseline.choose_candidate([15., 15., 15.2], [1, 1, 1], [1, 1, 1], 15., .1), 0)

    def test_only_reconstructed_os_valid_finite_candidates(self):
        self.assertEqual(baseline.choose_candidate([15., 15., math.nan, 15.1], [0, 1, 1, 1], [1, 0, 1, 1], 15., .1), 3)
        self.assertIsNone(baseline.choose_candidate([30.], [1], [1], 15., .1))

    def test_partial_mc_retains_full_generated_denominator(self):
        sample = dict(full_gen_sumw=200., selected_cross_section_pb=2.)
        counts = dict(events=180, windows={"15.0": dict(sumw=20., sumw2=20., events=20, ess=20.)})
        row = baseline.sample_prediction(sample, counts, 15., 3.)
        self.assertAlmostEqual(row["events"], 600.)
        self.assertEqual(row["full_gen_sumw"], 200.)

    def test_native_nano_arrays_count_each_event_once_with_its_weight(self):
        import ROOT
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nano.root"
            output = ROOT.TFile(str(path), "RECREATE")
            tree = ROOT.TTree("Events", "Events")
            run, lumi, event, n = [array("i", [value]) for value in (1, 1, 1, 2)]
            mass, os, valid, weight = array("f", [15., 15.1]), array("i", [1, 1]), array("i", [1, 1]), array("d", [2.])
            for name, buffer in (("run", run), ("luminosityBlock", lumi), ("event", event), ("nShiftDimuonVertex", n)):
                tree.Branch(name, buffer, name + "/I")
            for name, buffer, kind in (("ShiftDimuonVertex_mass", mass, "F"), ("ShiftDimuonVertex_isOS", os, "I"),
                                       ("ShiftDimuonVertex_dcaValid", valid, "I")):
                tree.Branch(name, buffer, name + "[nShiftDimuonVertex]/" + kind)
            tree.Branch("genWeight", weight, "genWeight/D")
            tree.Fill()
            event[0], os[0], os[1], weight[0] = 2, 0, 0, 3.
            tree.Fill()
            tree.Write()
            output.Close()
            counts = baseline.count_sample(dict(files=[str(path)], weight_branch="genWeight"), [15.], .1)
            self.assertEqual(counts["events"], 2)
            self.assertEqual(counts["sumw"], 5.)
            self.assertEqual(counts["windows"]["15.0"]["events"], 1)
            self.assertEqual(counts["windows"]["15.0"]["sumw"], 2.)
            self.assertEqual(counts["windows"]["15.0"]["sumw2"], 4.)


class ExactClsTest(unittest.TestCase):
    def test_signal_only_predictions_do_not_invent_background_or_limits(self):
        sample = dict(name="signal", full_gen_sumw=200., selected_cross_section_pb=2.,
                      mass_gev=15., epsilon=1.e-6, partial=True, unit_gen_weights=True)
        counts = dict(events=180, windows={"15.0": dict(sumw=20., sumw2=20., events=20, ess=20.)})
        inputs = dict(masses_gev=[15.], backgrounds=[], signals=[sample], window_half_width_fraction=.1)
        report = baseline.screen(inputs, {"signal": counts}, [.3], signal_only=True)
        point = report["scenarios"][0]["points"][0]
        self.assertIsNone(point["background"])
        self.assertNotIn("signal_events_90", point)
        self.assertAlmostEqual(point["signal"]["events"], 60.)
        self.assertAlmostEqual(point["signal"]["generated_dimuon_events"], 600.)
        self.assertAlmostEqual(point["unprocessed_signal_yield_maximum"], 60.)

    def test_empty_constant_weight_floor_uses_exact_binomial_bound(self):
        sample = dict(full_gen_sumw=1000., selected_cross_section_pb=2.)
        counts = dict(events=100, sumw=200., sumw2=400.)
        expected = .3 * 1000. * 2. * 200. * (1. - .1 ** .01) / 1000.
        self.assertAlmostEqual(baseline.empty_mc_planning_floor(sample, counts, .3), expected)
        counts["sumw2"] = 500.
        self.assertIsNone(baseline.empty_mc_planning_floor(sample, counts, .3))

    def test_zero_background_check_is_ln_ten(self):
        row = baseline.median_expected_limit(0.)
        self.assertEqual(row["observed_median"], 0)
        self.assertAlmostEqual(row["signal_events_90"], math.log(10.), places=10)

    def test_background_median_is_integer_not_fractional_asimov(self):
        row = baseline.median_expected_limit(2.7)
        self.assertEqual(row["observed_median"], 3)
        self.assertAlmostEqual(math.exp(baseline.log_cls(row["signal_events_90"], 2.7, 3)), .1, places=11)
        with self.assertRaises(ValueError):
            baseline.log_cls(1., 2.7, 2.7)

    def test_cls_has_correct_order_and_null(self):
        self.assertAlmostEqual(baseline.log_cls(0., 10., 10), 0.)
        self.assertLess(baseline.log_cls(5., 10., 10), baseline.log_cls(1., 10., 10))

    def test_empty_mc_cell_does_not_become_background_free_limit(self):
        empty = dict(events=10, windows={"15.0": dict(events=0, sumw=0., sumw2=0., ess=0.)})
        signal = dict(events=10, windows={"15.0": dict(events=3, sumw=3., sumw2=3., ess=3.)})
        samples = [dict(name=name, full_gen_sumw=10., selected_cross_section_pb=1.) for name in ("background", "signal")]
        samples[1].update(mass_gev=15., epsilon=1.e-6)
        report = baseline.screen(dict(masses_gev=[15.], backgrounds=samples[:1], signals=samples[1:],
                                      window_half_width_fraction=.1), {"background": empty, "signal": signal}, [3.])
        point = report["scenarios"][0]["points"][0]
        self.assertFalse(point["supported"])
        self.assertNotIn("signal_events_90", point)


if __name__ == "__main__":
    unittest.main()
