"""Bounded reconstructed-pair checks using the built histogrammer."""

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
class DimuonKinematicsRuntimeTest(unittest.TestCase):
    def setUp(self):
        import ROOT
        self.ROOT = ROOT
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def run_histograms(self, events, truth=False):
        root = self.ROOT.TFile(str(self.scratch / "input.root"), "RECREATE")
        tree = self.ROOT.TTree("Events", "Synthetic reconstructed muon pairs")
        source = (Path(__file__).resolve().parents[1] /
                  "libs/user_extensions/src/ShiftHistogramsFiller.cpp").read_text()
        fields = set(re.findall(r'GetAs<(?:float|int)>\("([^"]+)"\)', source))
        fields.update(("pt", "pz", "eta", "phi", "mass", "vx", "vy", "vz", "topology",
                       "probability", "originCompatibilityNormalizedChi2", "refitStatus",
                       "dca", "dcaValid"))
        buffers = {}
        collections = ("ShiftMuon", "ShiftDimuonVertex", "GenPart") if truth else (
            "ShiftMuon", "ShiftDimuonVertex")
        for collection in collections:
            count = buffers[f"n{collection}"] = array("i", [0])
            tree.Branch(f"n{collection}", count, f"n{collection}/I")
            for field in sorted(fields):
                value = buffers[f"{collection}_{field}"] = array("f", [0.0] * 2)
                tree.Branch(f"{collection}_{field}", value,
                            f"{collection}_{field}[n{collection}]/F")
        weight = array("d", [1.0])
        tree.Branch("shiftSamplingGenWeight", weight, "shiftSamplingGenWeight/D")
        for event in events:
            for name, values in buffers.items():
                for index in range(len(values)):
                    values[index] = 0
            buffers["nShiftMuon"][0] = 2
            buffers["nShiftDimuonVertex"][0] = 1
            defaults = {
                "ShiftMuon_pt": [3.0, 4.0], "ShiftMuon_phi": [0.0, math.pi / 2],
                "ShiftMuon_eta": [0.0, 0.0], "ShiftMuon_topology": [2.0, 2.0],
                "ShiftMuon_charge": [-1.0, 1.0], "ShiftMuon_hitGenPartIdx": [-1.0, -1.0],
                "ShiftMuon_genPartIdx": [-1.0, -1.0],
                "ShiftDimuonVertex_muonIdx1": [0.0], "ShiftDimuonVertex_muonIdx2": [1.0],
                "ShiftDimuonVertex_topologyMin": [2.0], "ShiftDimuonVertex_topologyMax": [2.0],
                "ShiftDimuonVertex_pt": [5.0], "ShiftDimuonVertex_mass": [3.1],
                "ShiftDimuonVertex_vx": [3.0], "ShiftDimuonVertex_vy": [4.0],
                "ShiftDimuonVertex_isOS": [1.0], "ShiftDimuonVertex_dcaValid": [1.0],
                "ShiftDimuonVertex_dca": [1.0],
            }
            defaults.update(event.get("branches", {}))
            for name, values in defaults.items():
                for index, value in enumerate(values):
                    buffers[name][index] = value
            weight[0] = event.get("weight", 1.0)
            tree.Fill()
        tree.Write()
        root.Close()
        config = self.scratch / "config.py"
        config.write_text(
            "from shift_histogrammer_config import *\n"
            "defaultHistParams = ()\nirregularDefaultHistParams = ()\n"
            f"enableTruthDiagnostics = {truth}\n"
            + ("" if truth else
               "extraEventCollections = {name: value for name, value in extraEventCollections.items() "
               "if not name.startswith('Gen')}\n")
        )
        output = self.scratch / "histograms.root"
        result = subprocess.run(
            [os.environ["SHIFT_WEIGHT_TEST_EXECUTABLE"], "--config", str(config),
             "--input_path", str(self.scratch / "input.root"),
             "--output_hists_path", str(output)], capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
        histograms = self.ROOT.TFile.Open(str(output))
        self.addCleanup(histograms.Close)
        return histograms

    def histogram(self, root, variable, category=""):
        histogram = root.Get(f"dimuon/ShiftDimuonVertex{category}_{variable}")
        self.assertTrue(histogram, f"{category}/{variable}")
        return histogram

    def integral(self, histogram):
        return histogram.Integral(0, histogram.GetNbinsX() + 1)

    def test_known_geometry_truth_free_and_category_membership(self):
        root = self.run_histograms([{}])
        for category in ("", "Good", "Both-Both"):
            angle = self.histogram(root, "openingAngle", category)
            self.assertEqual(self.integral(angle), 1.0)
            self.assertEqual(angle.GetBinContent(angle.FindBin(math.pi / 2)), 1.0)
            for variable, expected in (("ptAsymmetry", 1.0 / 7.0), ("momentumAsymmetry", 1.0 / 7.0),
                                       ("minMuonPt", 3.0), ("maxMuonPt", 4.0), ("vertexR", 5.0),
                                       ("deltaPhi", math.pi / 2), ("deltaR", math.pi / 2)):
                histogram = self.histogram(root, variable, category)
                self.assertEqual(histogram.GetBinContent(histogram.FindBin(expected)), 1.0, variable)
        for category in ("Near-Both", "Near-Far", "Both-Far", "Other"):
            self.assertEqual(self.integral(self.histogram(root, "openingAngle", category)), 0.0)
        self.assertEqual(self.integral(self.histogram(root, "massOS")), 1.0)
        self.assertEqual(self.integral(self.histogram(root, "massSS")), 0.0)
        self.assertEqual(self.integral(self.histogram(root, "refittedMass")), 0.0)
        self.assertEqual(self.integral(self.histogram(root, "constrainedMass")), 0.0)
        self.assertEqual(root.Get("dimuon/ShiftDimuonVertex_massVsOpeningAngle").GetEntries(), 1.0)

    def test_parallel_antiparallel_phi_wrap_and_small_angle(self):
        root = self.run_histograms([
            {"branches": {"ShiftMuon_phi": [0.0, 0.0]}},
            {"branches": {"ShiftMuon_phi": [0.0, math.pi]}},
            {"branches": {"ShiftMuon_phi": [math.pi - 0.01, -math.pi + 0.01]}},
            {"branches": {"ShiftMuon_phi": [0.0, 1e-5]}},
        ])
        angle = self.histogram(root, "openingAngle")
        self.assertEqual(angle.GetBinContent(angle.FindBin(0.0)), 2.0)
        self.assertEqual(angle.GetBinContent(angle.FindBin(math.pi)), 1.0)
        self.assertEqual(angle.GetBinContent(angle.GetNbinsX() + 1), 0.0)
        fine = self.histogram(root, "openingAngleFine")
        # The float-valued input lies very close to a bin edge after wrapping.
        self.assertAlmostEqual(fine.GetMean(), (0.02 + 1e-5) / 3.0, delta=1e-6)
        self.assertEqual(fine.GetBinContent(fine.GetNbinsX() + 1), 1.0)

    def test_bad_pair_indices_zero_and_nonfinite_momenta_are_skipped(self):
        root = self.run_histograms([
            {"branches": {"ShiftDimuonVertex_muonIdx1": [-1.0]}},
            {"branches": {"ShiftDimuonVertex_muonIdx2": [2.0]}},
            {"branches": {"ShiftDimuonVertex_muonIdx2": [0.0]}},
            {"branches": {"ShiftMuon_pt": [0.0, 4.0]}},
            {"branches": {"ShiftMuon_pz": [math.nan, 0.0]}},
        ])
        self.assertEqual(self.integral(self.histogram(root, "openingAngle")), 0.0)

    def test_longitudinal_momenta_and_valid_constrained_states(self):
        root = self.run_histograms([{"branches": {
            "ShiftMuon_pz": [30.0, 40.0],
            "ShiftMuon_constrainedValid": [1.0, 1.0],
            "ShiftMuon_constrainedPt": [3.0, 4.0],
            "ShiftMuon_constrainedPhi": [0.0, math.pi / 2],
            "ShiftMuon_constrainedPz": [30.0, 40.0],
            "ShiftDimuonVertex_pz": [70.0],
        }}])
        angle = math.acos(1200.0 / (math.hypot(3.0, 30.0) * math.hypot(4.0, 40.0)))
        self.assertAlmostEqual(self.histogram(root, "openingAngle").GetMean(), angle, places=6)
        self.assertAlmostEqual(self.histogram(root, "constrainedOpeningAngle").GetMean(), angle, places=6)
        rapidity = math.asinh(70.0 / math.hypot(3.1, 5.0))
        self.assertAlmostEqual(self.histogram(root, "rapidity").GetMean(), rapidity, places=6)

    def test_all_topologies_fit_validity_and_signed_weights(self):
        events = []
        for low, high in ((0, 2), (0, 3), (2, 2), (2, 3), (0, 0)):
            events.append({"weight": 2.5, "branches": {
                "ShiftDimuonVertex_topologyMin": [float(low)],
                "ShiftDimuonVertex_topologyMax": [float(high)],
                "ShiftDimuonVertex_constrainedValid": [1.0],
                "ShiftDimuonVertex_constrainedMass": [3.0],
                "ShiftDimuonVertex_refitStatus": [1.0],
                "ShiftDimuonVertex_refittedMass": [3.0],
            }})
        events.append({"weight": -1.0, "branches": {"ShiftDimuonVertex_isOS": [0.0]}})
        root = self.run_histograms(events, truth=True)
        self.assertEqual(self.integral(self.histogram(root, "openingAngle")), 11.5)
        for category in ("Near-Both", "Near-Far", "Both-Far", "Other"):
            self.assertEqual(self.integral(self.histogram(root, "openingAngle", category)), 2.5)
        self.assertEqual(self.integral(self.histogram(root, "openingAngle", "Both-Both")), 1.5)
        self.assertEqual(self.integral(self.histogram(root, "openingAngle", "Good")), 12.5)
        self.assertEqual(self.integral(self.histogram(root, "constrainedMass")), 12.5)
        self.assertEqual(self.integral(self.histogram(root, "refittedMass")), 12.5)
        self.assertEqual(self.integral(self.histogram(root, "massSS")), -1.0)


if __name__ == "__main__":
    unittest.main()
