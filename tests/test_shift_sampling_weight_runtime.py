"""Bounded runtime check, enabled with SHIFT_WEIGHT_TEST_EXECUTABLE."""

from array import array
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

    def input(self, weights, weight_type="D"):
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
            clone.Branch("shiftSamplingGenWeight", value, f"shiftSamplingGenWeight/{weight_type}")
        for index in range(3):
            if weights is not None:
                value[0] = weights[index]
            clone.Fill()
        clone.Write()
        output.Close()
        return self.scratch / "input.root"

    def run_histograms(self, weights, weight_type="D"):
        source = self.input(weights, weight_type)
        config = self.scratch / "config.py"
        config.write_text(
            "from shift_histogrammer_config import *\nnEvents = 3\n"
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

    def test_double_to_float_boundary_agrees_with_single_rounding(self):
        weights = [1.0 / 3.0, -1.0 / 7.0, 0.0]
        result, output = self.run_histograms(weights)
        self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
        root = self.ROOT.TFile.Open(str(output))
        histogram = root.Get("event/Event_nShiftMuon")
        measured = histogram.Integral(0, histogram.GetNbinsX() + 1)
        rounded = sum(array("f", weights))
        self.assertAlmostEqual(measured, rounded, places=8)
        self.assertLess(abs(measured - sum(weights)), 6e-8)
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


    def efficiency_input(self):
        output = self.ROOT.TFile(str(self.scratch / "efficiency_input.root"), "RECREATE")
        tree = self.ROOT.TTree("Events", "Two unequal-weight Jpsi outcomes")
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
        tree.Branch("shiftSamplingGenWeight", weight, "shiftSamplingGenWeight/D")
        tree.Branch("genWeight", native, "genWeight/F")
        for key, values in {
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
        }.items():
            for index, value in enumerate(values):
                buffers[key][index] = value
        buffers["nGenPart"][0] = 3
        tree.Fill()  # W/p=1: truth dimuon exists, no reconstructed candidates.
        weight[0] = 3.0
        buffers["nShiftMuon"][0] = 2
        buffers["nShiftDimuonVertex"][0] = 1
        tree.Fill()  # W/p=3: both truth muons and their pair are reconstructed.
        tree.Write()
        output.Close()
        return self.scratch / "efficiency_input.root"

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


if __name__ == "__main__":
    unittest.main()
