# Code style: AI-code. AI-generated regression test.
"""Bounded runtime check, enabled with SHIFT_WEIGHT_TEST_EXECUTABLE."""

from array import array
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("SHIFT_WEIGHT_TEST_EXECUTABLE"),
                     "Set the built histogrammer for the bounded runtime test")
class SamplingWeightRuntimeTest(unittest.TestCase):
    def setUp(self):
        import ROOT
        self.ROOT = ROOT
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def input(self, weights, weight_type="D", weight_branch="shiftSamplingGenWeight"):
        output = self.ROOT.TFile(str(self.scratch / "input.root"), "RECREATE")
        clone = self.ROOT.TTree("Events", "Synthetic empty SHIFT collections")
        buffers = {}
        for collection in ("ShiftMuon", "ShiftDimuonVertex", "GenPart"):
            count = buffers[f"n{collection}"] = array("i", [0])
            values = buffers[f"{collection}_pt"] = array("f", [0.0])
            clone.Branch(f"n{collection}", count, f"n{collection}/I")
            clone.Branch(f"{collection}_pt", values, f"{collection}_pt[n{collection}]/F")
        native = array("f", [7.0])
        clone.Branch("genWeight", native, "genWeight/F")
        value = array("d" if weight_type == "D" else "f", [0.0])
        if weights is not None:
            clone.Branch(weight_branch, value, f"{weight_branch}/{weight_type}")
        for index in range(3):
            if weights is not None:
                value[0] = weights[index]
            clone.Fill()
        clone.Write()
        output.Close()
        return self.scratch / "input.root"

    def run_histograms(self, weights, weight_type="D", weight_branch="shiftSamplingGenWeight",
                       truth_diagnostics=True, data_mode=False):
        source = self.input(weights, weight_type, weight_branch)
        config = self.scratch / "config.py"
        config.write_text(
            "from shift_histogrammer_config import *\nnEvents = 3\n"
            f"weightsBranchName = {weight_branch!r}\n"
            f"enableTruthDiagnostics = {truth_diagnostics!r}\n"
            f"recoDataMode = {data_mode!r}\n"
            "# Synthetic fixture has no detector-diagnostic scalar branches.\n"
            "defaultHistParams = tuple(entry for entry in defaultHistParams "
            "if not (entry[0] == 'Event' and entry[1].startswith('ShiftRecoDiag_')))\n"
        )
        output = self.scratch / "histograms.root"
        result = subprocess.run(
            [os.environ["SHIFT_WEIGHT_TEST_EXECUTABLE"], "--config", str(config),
             "--input_path", str(source), "--output_hists_path", str(output)],
            capture_output=True, text=True, timeout=180,
        )
        return result, output

    def test_signed_zero_weights_reach_histograms_and_cutflow_once(self):
        result, output = self.run_histograms([2.5, -1.0, 0.0])
        self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        self.assertAlmostEqual(root.Get("cutFlow").GetBinContent(1), 1.5)
        self.assertEqual(root.Get("rawEventsCutFlow").GetBinContent(1), 3)
        histogram = root.Get("event/Event_nShiftMuon")
        self.assertTrue(histogram)
        self.assertAlmostEqual(histogram.Integral(0, histogram.GetNbinsX() + 1), 1.5)
        root.Close()

    def test_double_weights_reach_histograms_and_cutflow_without_float_rounding(self):
        weights = [1.0 / 3.0, -1.0 / 7.0, 0.0]
        result, output = self.run_histograms(weights)
        self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        histogram = root.Get("event/Event_nShiftMuon")
        measured = histogram.Integral(0, histogram.GetNbinsX() + 1)
        expected = math.fsum(weights)
        self.assertTrue(math.isclose(measured, expected, rel_tol=2e-13, abs_tol=0))
        self.assertTrue(math.isclose(root.Get("cutFlow").GetBinContent(1), expected, rel_tol=2e-13, abs_tol=0))
        root.Close()

    def test_recovery_weights_are_used_with_truth_plots_on_and_off(self):
        for truth_diagnostics in (False, True):
            with self.subTest(truth_diagnostics=truth_diagnostics):
                result, output = self.run_histograms([2.5, -1.0, 0.0], weight_branch="shiftRecoveryGenWeight",
                    truth_diagnostics=truth_diagnostics)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
                root = self.ROOT.TFile.Open(str(output))
                self.assertEqual(root.Get("cutFlow").GetBinContent(1), 1.5)
                self.assertEqual(root.Get("rawEventsCutFlow").GetBinContent(1), 3)
                histogram = root.Get("event/Event_nShiftMuon")
                self.assertEqual(histogram.Integral(0, histogram.GetNbinsX() + 1), 1.5)
                root.Close()

    def test_tiny_double_recovery_weights_remain_positive(self):
        weights = [1e-100, 2e-100, 0.0]
        result, output = self.run_histograms(weights, weight_branch="shiftRecoveryGenWeight",
            truth_diagnostics=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        histogram = root.Get("event/Event_nShiftMuon")
        measured = histogram.Integral(0, histogram.GetNbinsX() + 1)
        self.assertGreater(measured, 0)
        self.assertTrue(math.isclose(measured, math.fsum(weights), rel_tol=2e-13, abs_tol=0))
        self.assertTrue(math.isclose(root.Get("cutFlow").GetBinContent(1), math.fsum(weights), rel_tol=2e-13, abs_tol=0))
        self.assertTrue(math.isclose(sum(histogram.GetBinError(index) ** 2 for index in range(histogram.GetNbinsX() + 2)),
                                    math.fsum(weight * weight for weight in weights), rel_tol=2e-13, abs_tol=0))
        root.Close()

    def test_explicit_data_mode_uses_unit_weights(self):
        result, output = self.run_histograms([2.5, -1.0, 0.0], weight_branch="shiftRecoveryGenWeight",
            truth_diagnostics=False, data_mode=True)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        self.assertEqual(root.Get("cutFlow").GetBinContent(1), 3)
        histogram = root.Get("event/Event_nShiftMuon")
        self.assertEqual(histogram.Integral(0, histogram.GetNbinsX() + 1), 3)
        root.Close()

    def test_missing_sampling_weight_refuses_output(self):
        result, output = self.run_histograms(None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot read configured event weight", result.stdout + result.stderr)
        self.assertFalse(output.exists())

    def test_float_generator_weight_is_supported(self):
        result, output = self.run_histograms([2.5, -1.0, 0.0], weight_type="F")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        self.assertAlmostEqual(root.Get("cutFlow").GetBinContent(1), 1.5)
        self.assertEqual(root.Get("rawEventsCutFlow").GetBinContent(1), 3)
        root.Close()


    def efficiency_input(self, events=None, weight_branch="shiftSamplingGenWeight"):
        output = self.ROOT.TFile(str(self.scratch / "efficiency_input.root"), "RECREATE")
        tree = self.ROOT.TTree("Events", "Synthetic Jpsi reconstruction outcomes")
        source = (Path(__file__).resolve().parents[1] / "libs/user_extensions/src/ShiftHistogramsFiller.cpp").read_text()
        fields = set(re.findall(r'GetAs<(?:float|int)>\("([^"]+)"\)', source))
        fields.update(("pt", "eta", "phi", "mass", "pz", "vx", "vy", "vz", "statusFlags",
                       "dca", "dcaValid", "isOS", "chi2", "normalizedChi2"))
        buffers = {}
        for collection in ("GenPart", "ShiftMuon", "ShiftDimuonVertex"):
            count = buffers[f"n{collection}"] = array("i", [0])
            tree.Branch(f"n{collection}", count, f"n{collection}/I")
            variables = fields | {"pdgId", "status", "genPartIdxMother"} if collection == "GenPart" else fields
            for field in sorted(variables):
                integer = field in ("pdgId", "status", "statusFlags", "genPartIdxMother", "charge")
                value = buffers[f"{collection}_{field}"] = array("i" if integer else "f", [0] * 3)
                kind = "I" if integer else "F"
                tree.Branch(f"{collection}_{field}", value, f"{collection}_{field}[n{collection}]/{kind}")
        weight = array("d", [1.0])
        native = array("f", [7.0])
        tree.Branch(weight_branch, weight, f"{weight_branch}/D")
        tree.Branch("genWeight", native, "genWeight/F")
        defaults = {
            "GenPart_pdgId": [443, 13, -13], "GenPart_status": [2, 1, 1],
            "GenPart_genPartIdxMother": [-1, 0, 0], "GenPart_pt": [4.0, 2.0, 3.0],
            "GenPart_eta": [1.0, 1.0, 1.0], "GenPart_phi": [0.1, 0.5, -0.5],
            "GenPart_pz": [4.0, 2.0, 3.0], "GenPart_vx": [1.0] * 3,
            "GenPart_vy": [1.0] * 3, "GenPart_vz": [10.0] * 3,
            "ShiftMuon_hitGenPartIdx": [1.0, 2.0], "ShiftMuon_genPartIdx": [1.0, 2.0],
            "ShiftMuon_topology": [2.0, 2.0], "ShiftMuon_charge": [-1, 1],
            "ShiftMuon_pt": [2.0, 3.0], "ShiftMuon_pz": [2.0, 3.0],
            "ShiftMuon_eta": [1.0, 1.0], "ShiftMuon_phi": [0.5, -0.5],
            "ShiftMuon_vx": [1.0, 1.0], "ShiftMuon_vy": [1.0, 1.0], "ShiftMuon_vz": [10.0, 10.0],
            "ShiftDimuonVertex_muonIdx1": [0.0], "ShiftDimuonVertex_muonIdx2": [1.0],
            "ShiftDimuonVertex_topologyMin": [2.0], "ShiftDimuonVertex_topologyMax": [2.0],
            "ShiftDimuonVertex_pt": [4.0], "ShiftDimuonVertex_pz": [4.0],
            "ShiftDimuonVertex_eta": [1.0], "ShiftDimuonVertex_phi": [0.1],
            "ShiftDimuonVertex_mass": [3.0], "ShiftDimuonVertex_vx": [1.0],
            "ShiftDimuonVertex_vy": [1.0], "ShiftDimuonVertex_vz": [10.0],
        }
        if events is None:
            events = (
                {"weight": 1.0, "reconstructed": False},
                {"weight": 3.0, "reconstructed": True},
            )
        for event in events:
            for values in buffers.values():
                for index in range(len(values)):
                    values[index] = 0
            for key, values in {**defaults, **event.get("branches", {})}.items():
                for index, value in enumerate(values):
                    buffers[key][index] = value
            weight[0] = event["weight"]
            buffers["nGenPart"][0] = 3
            buffers["nShiftMuon"][0] = 2 if event["reconstructed"] else 0
            buffers["nShiftDimuonVertex"][0] = 1 if event["reconstructed"] else 0
            tree.Fill()
        tree.Write()
        output.Close()
        return self.scratch / "efficiency_input.root"

    def test_recovery_double_weights_reach_truth_efficiencies_and_diagnostics(self):
        first, second = 1.0 / 3.0, 1.0 / 7.0
        source = self.efficiency_input(events=(
            {"weight": first, "reconstructed": False},
            {"weight": second, "reconstructed": True},
        ), weight_branch="shiftRecoveryGenWeight")
        config = self.scratch / "recovery_efficiency.py"
        config.write_text(
            "from shift_histogrammer_config import *\nnEvents = 2\n"
            "weightsBranchName = 'shiftRecoveryGenWeight'\n"
            "enableTruthDiagnostics = True\nrecoDataMode = False\n"
            "defaultHistParams = (('Event', 'nShiftMuon', 10, 0, 10, 'event'),)\n"
            "irregularDefaultHistParams = ()\n"
        )
        output = self.scratch / "recovery_efficiency.root"
        result = subprocess.run([os.environ["SHIFT_WEIGHT_TEST_EXECUTABLE"], "--config", str(config),
                                "--input_path", str(source), "--output_hists_path", str(output)],
                               capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        integral = lambda histogram: histogram.Integral(0, histogram.GetNbinsX() + 1)
        for name, expected in (
            ("efficiency/ShiftMuonEfficiency_pt_total", 2 * (first + second)),
            ("efficiency/ShiftMuonEfficiency_pt_pass", 2 * second),
            ("efficiency/ShiftDimuonVertexEfficiency_pt_total", first + second),
            ("efficiency/ShiftDimuonVertexEfficiency_pt_pass", second),
            ("truth_diagnostics/TruthDiagnostics_hitMatched", 2 * second),
        ):
            histogram = root.Get(name)
            self.assertTrue(histogram, name)
            self.assertTrue(math.isclose(integral(histogram), expected, rel_tol=2e-13, abs_tol=0), name)
        self.assertTrue(math.isclose(root.Get("cutFlow").GetBinContent(1), first + second, rel_tol=2e-13, abs_tol=0))
        root.Close()

    def test_unequal_weights_correct_muon_dimuon_efficiencies_and_diagnostic_shapes(self):
        source = self.efficiency_input()
        recorded = {}
        for mode in ("weighted", "legacy"):
            config = self.scratch / f"efficiency_{mode}.py"
            config.write_text(
                "from shift_histogrammer_config import *\nnEvents = 2\n"
                "defaultHistParams = (('Event', 'nShiftMuon', 10, 0, 10, 'event'),)\n"
                "irregularDefaultHistParams = ()\n"
                + ("weightsBranchName = 'genWeight'\n" if mode == "legacy" else "")
            )
            path = self.scratch / f"efficiency_{mode}.root"
            result = subprocess.run([os.environ["SHIFT_WEIGHT_TEST_EXECUTABLE"], "--config", str(config),
                                     "--input_path", str(source), "--output_hists_path", str(path)],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
            root = self.ROOT.TFile.Open(str(path))
            values = {}
            for prefix in ("ShiftMuonEfficiency", "ShiftDimuonVertexEfficiency"):
                total = root.Get(f"efficiency/{prefix}_pt_total")
                passed = root.Get(f"efficiency/{prefix}_pt_pass")
                self.assertTrue(total and passed, prefix)
                integral = lambda histogram: histogram.Integral(0, histogram.GetNbinsX() + 1)
                values[prefix] = (integral(passed), integral(total),
                                  sum(total.GetBinError(index) ** 2 for index in range(total.GetNbinsX() + 2)),
                                  sum(passed.GetBinError(index) ** 2 for index in range(passed.GetNbinsX() + 2)))
            diagnostic = root.Get("truth_diagnostics/TruthDiagnostics_hitMatched")
            values["hit_matches"] = integral(diagnostic)
            self.assertEqual(root.Get("rawEventsCutFlow").GetBinContent(1), 2)
            recorded[mode] = values
            root.Close()
        self.assertEqual(recorded["weighted"]["ShiftMuonEfficiency"][:2], (6.0, 8.0))
        self.assertEqual(recorded["weighted"]["ShiftDimuonVertexEfficiency"][:2], (3.0, 4.0))
        self.assertAlmostEqual(recorded["weighted"]["ShiftMuonEfficiency"][2], 20.0)
        self.assertAlmostEqual(recorded["weighted"]["ShiftMuonEfficiency"][3], 18.0)
        self.assertAlmostEqual(recorded["weighted"]["ShiftDimuonVertexEfficiency"][2], 10.0)
        self.assertAlmostEqual(recorded["weighted"]["ShiftDimuonVertexEfficiency"][3], 9.0)
        self.assertEqual(recorded["legacy"]["ShiftMuonEfficiency"][:2], (2.0, 4.0))
        self.assertEqual(recorded["legacy"]["ShiftDimuonVertexEfficiency"][:2], (1.0, 2.0))
        self.assertEqual(recorded["weighted"]["hit_matches"], 6.0)
        self.assertEqual(recorded["legacy"]["hit_matches"], 2.0)

    def test_dimuon_opening_angle_uses_generated_signed_momenta_and_hit_matching(self):
        good_pair = {"ShiftDimuonVertex_isOS": [1.0], "ShiftDimuonVertex_dcaValid": [1.0],
                     "ShiftDimuonVertex_dca": [10.0]}
        # Keep eta unchanged while varying pz to distinguish signed momentum
        # directions from an angle reconstructed through eta or deltaR.
        events = (
            # The failed reconstruction still contributes at its generated angle.
            {"weight": 1.0, "reconstructed": False, "branches": {
                "GenPart_pt": [4.0, 1.0, 1.0], "GenPart_phi": [0.1, 0.0, 0.6],
                "GenPart_pz": [4.0, 0.0, 0.0]}},
            # Reco directions differ deliberately: the numerator uses the truth angle.
            {"weight": 3.0, "reconstructed": True, "branches": {
                **good_pair, "GenPart_pt": [4.0, 1.0, 1.0],
                "GenPart_phi": [0.1, 0.0, 0.02], "GenPart_pz": [4.0, -100.0, -100.0],
                "ShiftMuon_phi": [0.0, 1.0], "ShiftMuon_pz": [0.0, 0.0]}},
            # Legacy angular association must not count as successful hit matching.
            {"weight": 5.0, "reconstructed": True, "branches": {
                **good_pair, "GenPart_pt": [4.0, 1.0, 1.0],
                "GenPart_phi": [0.1, 0.0, 0.4], "GenPart_pz": [4.0, 0.0, 0.0],
                "ShiftMuon_hitGenPartIdx": [-1.0, -1.0]}},
            # About 1e-9 radians: acos of the rounded cosine would give zero.
            {"weight": 7.0, "reconstructed": True, "branches": {
                "GenPart_pt": [4.0, 1.0, 1.0], "GenPart_phi": [0.1, 0.0, 1e-7],
                "GenPart_pz": [4.0, -100.0, -100.0], "ShiftMuon_topology": [0.0, 2.0],
                "ShiftDimuonVertex_topologyMin": [0.0], "ShiftDimuonVertex_topologyMax": [2.0]}},
            # Opposite signed longitudinal momenta give an angle near pi.
            {"weight": 11.0, "reconstructed": True, "branches": {
                **good_pair, "GenPart_pt": [4.0, 2.0, 3.0],
                "GenPart_phi": [0.1, 0.0, math.pi], "GenPart_pz": [4.0, -20.0, 30.0],
                "ShiftMuon_topology": [2.0, 3.0], "ShiftDimuonVertex_topologyMin": [2.0],
                "ShiftDimuonVertex_topologyMax": [3.0]}},
            # A muon pair from a Z is outside the existing J/psi denominator.
            {"weight": 13.0, "reconstructed": True, "branches": {
                **good_pair, "GenPart_pdgId": [23, 13, -13]}},
        )
        source = self.efficiency_input(events)
        categories = ("", "Good", "Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other")
        edges = (0.0, 1e-10, 1e-8, 1e-3, 0.5, 1.0, 2.0, 3.0, math.nextafter(math.pi, math.inf))
        for mode in ("weighted", "legacy"):
            with self.subTest(mode=mode):
                config = self.scratch / f"opening_angle_{mode}.py"
                config.write_text(
                    "from shift_histogrammer_config import *\n"
                    f"nEvents = {len(events)}\n"
                    "defaultHistParams = (('Event', 'nShiftMuon', 10, 0, 10, 'event'),)\n"
                    "irregularDefaultHistParams = ()\n"
                    "irregularHistParams = tuple(entry for entry in irregularHistParams "
                    "if entry[1] not in ('openingAngle_total', 'openingAngle_pass')) + tuple(\n"
                    f"    (prefix, f'openingAngle_{{kind}}', {edges!r}, 'efficiency')\n"
                    "    for prefix in dimuonEfficiencyPrefixes for kind in ('total', 'pass'))\n"
                    + ("weightsBranchName = 'genWeight'\n" if mode == "legacy" else "")
                )
                path = self.scratch / f"opening_angle_{mode}.root"
                result = subprocess.run(
                    [os.environ["SHIFT_WEIGHT_TEST_EXECUTABLE"], "--config", str(config),
                     "--input_path", str(source), "--output_hists_path", str(path)],
                    capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
                root = self.ROOT.TFile.Open(str(path))
                try:
                    weights = (1.0, 3.0, 5.0, 7.0, 11.0) if mode == "weighted" else (1.0,) * 5
                    # Bin 1 (exact parallel) and flow bins remain empty. Tiny but
                    # nonzero, forward, unmatched, failed and antiparallel pairs
                    # occupy five distinct bins.
                    expected_total = [0.0, weights[3], weights[1], weights[2], weights[0],
                                      0.0, 0.0, weights[4]]
                    category_pass = {
                        "": {2: weights[3], 3: weights[1], 8: weights[4]},
                        "Good": {3: weights[1], 8: weights[4]},
                        "Near-Both": {2: weights[3]}, "Near-Far": {},
                        "Both-Both": {3: weights[1]}, "Both-Far": {8: weights[4]}, "Other": {},
                    }
                    for category in categories:
                        prefix = f"ShiftDimuonVertex{category}Efficiency"
                        total = root.Get(f"efficiency/{prefix}_openingAngle_total")
                        passed = root.Get(f"efficiency/{prefix}_openingAngle_pass")
                        self.assertTrue(total and passed, prefix)
                        self.assertEqual(total.GetNbinsX(), len(edges) - 1)
                        self.assertEqual([total.GetBinContent(index) for index in range(1, len(edges))],
                                         expected_total, prefix)
                        self.assertEqual([passed.GetBinContent(index) for index in range(1, len(edges))],
                                         [category_pass[category].get(index, 0.0)
                                          for index in range(1, len(edges))], prefix)
                        for histogram in (total, passed):
                            self.assertEqual(histogram.GetBinContent(0), 0.0, prefix)
                            self.assertEqual(histogram.GetBinContent(len(edges)), 0.0, prefix)
                        pt_total = root.Get(f"efficiency/{prefix}_pt_total")
                        pt_pass = root.Get(f"efficiency/{prefix}_pt_pass")
                        integral = lambda histogram: histogram.Integral(0, histogram.GetNbinsX() + 1)
                        self.assertEqual(integral(total), integral(pt_total), prefix)
                        self.assertEqual(integral(passed), integral(pt_pass), prefix)
                        for index, content in enumerate(expected_total, start=1):
                            self.assertAlmostEqual(total.GetBinError(index) ** 2, content ** 2)
                        for index, content in category_pass[category].items():
                            self.assertAlmostEqual(passed.GetBinError(index) ** 2, content ** 2)
                    self.assertEqual(root.Get("rawEventsCutFlow").GetBinContent(1), len(events))
                finally:
                    root.Close()


if __name__ == "__main__":
    unittest.main()
