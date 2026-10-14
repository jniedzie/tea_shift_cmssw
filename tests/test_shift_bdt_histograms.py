# Code style: AI-code. AI-generated regression test.
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from array import array

import numpy as np
import ROOT
import uproot

REPOSITORY = Path(__file__).resolve().parents[1]
EXECUTABLE = os.environ.get("SHIFT_BDT_TEST_EXECUTABLE")


def run(input_path, output_path, *, data=False, require_score=True, truth_diagnostics=None,
        weight_branch="shiftSamplingGenWeight"):
    if truth_diagnostics is None:
        truth_diagnostics = not data
    configuration = Path(str(output_path) + ".py")
    configuration.write_text(
        "from shift_histogrammer_config import *\n"
        "defaultHistParams = (('Event', 'nShiftMuon', 10, 0, 10, 'event'),)\n"
        "irregularDefaultHistParams = ()\n"
        f"enableTruthDiagnostics = {truth_diagnostics}\nrequireBdtScore = {require_score}\n"
        f"recoDataMode = {data}\nweightsBranchName = {weight_branch!r}\n"
        + ("extraEventCollections = {n: c for n, c in extraEventCollections.items() if not n.startswith('Gen')}\n"
           if data else "")
    )
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = ":".join((str(REPOSITORY / "configs"),
        str(REPOSITORY / "tea/pylibs"), str(REPOSITORY / "bin"), environment.get("PYTHONPATH", "")))
    build = Path(EXECUTABLE).parent
    environment["LD_LIBRARY_PATH"] = ":".join(
        [str(build / "tea/libs" / name) for name in ("core", "extensions", "histogramming")]
        + [environment.get("LD_LIBRARY_PATH", "")])
    result = subprocess.run([EXECUTABLE, "--config", str(configuration),
        "--input_path", str(input_path), "--output_hists_path", str(output_path)],
        capture_output=True, text=True, env=environment, timeout=180)
    if result.returncode:
        raise ValueError(result.stderr + result.stdout[-3000:])
    with uproot.open(output_path) as source:
        return json.loads(str(source["ShiftDimuonVertex_bdtHistogramMetadata"]))


def fixture(path, *, omit_score=False, score_type="D", nonfinite=False, data=False, zero_pairs=False,
            fractional_weight=False, wrong_score_count=False, invalid_reference=False,
            weight_branch="shiftSamplingGenWeight", weights=None):
    file = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("Events", "Events")
    counts = {name: array("i", [0]) for name in ("nShiftMuon", "nShiftDimuonVertex", "nGenPart")}
    for name, buffer in counts.items():
        tree.Branch(name, buffer, name + "/I")
    fields = {
        "ShiftDimuonVertex_muonIdx1": "I", "ShiftDimuonVertex_muonIdx2": "I",
        "ShiftDimuonVertex_mass": "D", "ShiftDimuonVertex_vx": "D",
        "ShiftDimuonVertex_vy": "D", "ShiftDimuonVertex_vz": "D",
        "ShiftDimuonVertex_topologyMin": "I", "ShiftDimuonVertex_topologyMax": "I",
        "ShiftDimuonVertex_isOS": "I", "ShiftDimuonVertex_dcaValid": "I",
        "ShiftDimuonVertex_dca": "D",
    }
    if not omit_score:
        fields["ShiftDimuonVertex_bdtScore"] = score_type
    if not data:
        fields.update({
            "ShiftMuon_hitGenPartIdx": "I", "ShiftMuon_hitTruthPurity": "D",
            "ShiftMuon_hitTruthMatchedLayers": "I", "ShiftMuon_hitSimTrackId": "I",
            "GenPart_pdgId": "I", "GenPart_status": "I", "GenPart_genPartIdxMother": "I",
            "GenPart_vx": "D", "GenPart_vy": "D", "GenPart_vz": "D",
        })
    # Ordinary diagnostics share the same run. Supply their scalar inputs too.
    cpp = (REPOSITORY / "libs/user_extensions/src/ShiftHistogramsFiller.cpp").read_text()
    ordinary = set(re.findall(r'GetAs<(?:float|int)>\("([^"+]+)"\)', cpp))
    ordinary.update(("pt", "pz", "eta", "phi", "mass", "vx", "vy", "vz", "topology",
                     "originCompatibilityNormalizedChi2", "probability"))
    for collection in ("ShiftMuon", "ShiftDimuonVertex") + (() if data else ("GenPart",)):
        for variable in ordinary:
            fields.setdefault(collection + "_" + variable, "F")
    if data:
        fields = {name: kind for name, kind in fields.items()
                  if name not in ("ShiftMuon_hitGenPartIdx", "ShiftMuon_genPartIdx")}
    buffers = {}
    for name, kind in fields.items():
        count = "nShiftDimuonVertex" if name.startswith("ShiftDimuonVertex_") else (
            "nShiftMuon" if name.startswith("ShiftMuon_") else "nGenPart")
        if name == "ShiftDimuonVertex_bdtScore" and wrong_score_count:
            count = "nShiftMuon"
        buffers[name] = array({"I": "i", "D": "d", "F": "f"}[kind], [0] * 8)
        tree.Branch(name, buffers[name], f"{name}[{count}]/{kind}")
    weight = array("d", [0])
    if not data:
        tree.Branch(weight_branch, weight, f"{weight_branch}/D")
    for event_index, (pairs, event_weight) in enumerate(((0, -2), (2, 0), (1, 3), (1, -1), (1, 2))):
        counts["nShiftMuon"][0] = counts["nGenPart"][0] = 2
        counts["nShiftDimuonVertex"][0] = pairs = 0 if zero_pairs else pairs
        weight[0] = weights[event_index] if weights is not None else (
            event_weight + (.123456789123 if fractional_weight else 0.))
        for index in range(2):
            for name in fields:
                buffers[name][index] = 0
            for name, value in {
                "ShiftDimuonVertex_muonIdx2": 7 if invalid_reference else 1, "ShiftDimuonVertex_mass": 3.1,
                "ShiftDimuonVertex_vx": 1, "ShiftDimuonVertex_vy": 2,
                "ShiftDimuonVertex_vz": -12000, "ShiftDimuonVertex_topologyMin": 2,
                "ShiftDimuonVertex_topologyMax": 2, "ShiftDimuonVertex_isOS": 1,
                "ShiftDimuonVertex_dcaValid": 1, "ShiftDimuonVertex_dca": 2,
            }.items():
                buffers[name][index] = value
            if not omit_score:
                buffers["ShiftDimuonVertex_bdtScore"][index] = (
                    float("nan") if nonfinite else (float(index) if event_index == 1 else 0.4403188643890805))
            if not data:
                values = {
                    "ShiftMuon_hitGenPartIdx": index, "ShiftMuon_hitTruthPurity": 0.0 if event_index == 4 else 1.0,
                    "ShiftMuon_hitTruthMatchedLayers": 3, "ShiftMuon_hitSimTrackId": index,
                    "GenPart_pdgId": 13 if index == 0 else -13, "GenPart_status": 1,
                    "GenPart_genPartIdxMother": -1, "GenPart_vx": 0, "GenPart_vy": 0,
                    "GenPart_vz": float(index) if event_index == 2 else 0.0,
                }
                for name, value in values.items():
                    buffers[name][index] = value
            for collection in ("ShiftMuon", "GenPart") if not data else ("ShiftMuon",):
                for variable, value in (("pt", 3. + index), ("pz", -100.), ("eta", -3.), ("phi", .2)):
                    buffers[collection + "_" + variable][index] = value
        tree.Fill()
    tree.Write()
    ROOT.TObjString('{"model_sha256":"frozen-model","score_selection_applied":false}').Write(
        "ShiftDimuonVertex_bdtScoreMetadata")
    file.Close()


@unittest.skipUnless(EXECUTABLE, "Set SHIFT_BDT_TEST_EXECUTABLE to the bounded validation build")
class BdtHistogramTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.input = self.directory / "input.root"
        self.output = self.directory / "hist.root"

    def test_all_pairs_weights_once_and_mc_subsets_partition(self):
        fixture(self.input)
        receipt = run(str(self.input), self.output)
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["scope"], "input_file")
        self.assertEqual((receipt["events"], receipt["pairs"]), (5, 5))
        self.assertEqual(receipt["label_pairs"], {"Genuine": 3, "Accidental": 1, "Unknown": 1})
        self.assertEqual(receipt["pair_sumw"], 4)
        self.assertEqual(receipt["pair_sumw2"], 14)
        self.assertEqual(receipt["event_sumw"], 2)
        with uproot.open(self.output) as source:
            inclusive = source["dimuon_bdt/ShiftDimuonVertex_bdtScore"]
            raw = source["dimuon_bdt/ShiftDimuonVertex_bdtScoreUnweighted"]
            self.assertEqual(np.sum(inclusive.values(flow=True)), 4)
            self.assertEqual(np.sum(inclusive.variances(flow=True)), 14)
            self.assertEqual(np.sum(raw.values(flow=True)), 5)
            self.assertEqual(raw.values(flow=True)[0], 0)
            self.assertEqual(raw.values(flow=True)[-1], 1)  # Exact score 1 remains in ROOT overflow.
            self.assertEqual(np.sum(source["dimuon_bdt/ShiftDimuonVertex_bdtScoreVsMass"].values(flow=True)), 4)
            self.assertEqual(np.sum(source["dimuon_bdt/ShiftDimuonVertexBoth-Both_bdtScore"].values(flow=True)), 4)
            self.assertIn("cutFlow", source)
            self.assertEqual(source["rawEventsCutFlow"].values()[0], 5)
            self.assertEqual(str(source["ShiftDimuonVertex_bdtScoreMetadata"]),
                             '{"model_sha256":"frozen-model","score_selection_applied":false}')
            self.assertEqual(source["dimuon_bdt/ShiftDimuonVertex_bdtScoreVsMass"].axis(0).member("fTitle"),
                             "m_{#mu#mu}^{reco} [GeV]")

    def test_missing_score_fails_even_zero_pairs(self):
        fixture(self.input, omit_score=True, zero_pairs=True)
        with self.assertRaisesRegex(ValueError, "Missing required saved BDT score"):
            run(str(self.input), self.output)
        self.assertFalse(self.output.exists())

    def test_float_score_is_rejected(self):
        fixture(self.input, score_type="F")
        with self.assertRaisesRegex(ValueError, "must be a Double"):
            run(str(self.input), self.output)

    def test_invalid_score_stops_before_writing_histograms(self):
        fixture(self.input, nonfinite=True)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            run(str(self.input), self.output)
        self.assertFalse(self.output.exists())

    def test_data_path_reads_no_truth_and_uses_unit_weight(self):
        fixture(self.input, data=True)
        with uproot.open(self.input) as source:
            self.assertNotIn("ShiftMuon_hitGenPartIdx", source["Events"])
            self.assertNotIn("GenPart_pdgId", source["Events"])
        receipt = run(str(self.input), self.output, data=True)
        self.assertEqual(receipt["pairs"], 5)
        self.assertEqual(receipt["pair_sumw"], 5)
        with uproot.open(self.output) as source:
            self.assertEqual(np.sum(source["dimuon_bdt/ShiftDimuonVertex_bdtScoreGenuine"].values(flow=True)), 0)
        self.assertFalse(receipt["truth_diagnostics"])

    def test_zero_pairs_creates_empty_complete_histograms(self):
        fixture(self.input, zero_pairs=True)
        receipt = run(str(self.input), self.output)
        self.assertEqual((receipt["events"], receipt["pairs"]), (5, 0))
        self.assertTrue(receipt["complete"])

    def test_unscored_input_keeps_ordinary_histogramming(self):
        fixture(self.input, omit_score=True)
        receipt = run(str(self.input), self.output, require_score=False)
        self.assertFalse(receipt["score_available"])
        with uproot.open(self.output) as source:
            self.assertEqual(source["rawEventsCutFlow"].values()[0], 5)

    def test_wrong_score_array_count_is_rejected(self):
        fixture(self.input, wrong_score_count=True)
        with self.assertRaisesRegex(ValueError, "counted by nShiftDimuonVertex"):
            run(str(self.input), self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_pair_reference_is_rejected(self):
        fixture(self.input, invalid_reference=True)
        with self.assertRaisesRegex(ValueError, "Invalid retained BDT dimuon reference"):
            run(str(self.input), self.output)
        self.assertFalse(self.output.exists())

    def test_double_event_weight_is_not_truncated(self):
        fixture(self.input, fractional_weight=True)
        receipt = run(str(self.input), self.output)
        weights = (0.123456789123, 0.123456789123, 3.123456789123,
                   -0.876543210877, 2.123456789123)
        self.assertAlmostEqual(receipt["pair_sumw"], sum(weights), places=12)
        with uproot.open(self.output) as source:
            histogram = source["dimuon_bdt/ShiftDimuonVertex_bdtScore"]
            self.assertAlmostEqual(np.sum(histogram.values(flow=True)), sum(weights), places=12)
            self.assertAlmostEqual(np.sum(histogram.variances(flow=True)), sum(w*w for w in weights), places=12)

    def test_truth_plots_off_preserves_recovery_weights_in_bdt_histograms(self):
        for weights in ((-2.123456789123, 0.123456789123, 3.123456789123,
                         -0.876543210877, 2.123456789123),
                        (1e-100, 2e-100, 3e-100, 4e-100, 5e-100)):
            with self.subTest(weights=weights):
                fixture(self.input, weight_branch="shiftRecoveryGenWeight", weights=weights)
                receipt = run(str(self.input), self.output, truth_diagnostics=False,
                              weight_branch="shiftRecoveryGenWeight")
                pairs = (weights[1], weights[1], *weights[2:])
                expected = math.fsum(pairs)
                self.assertFalse(receipt["truth_diagnostics"])
                self.assertEqual(receipt["weights_branch"], "shiftRecoveryGenWeight")
                self.assertEqual(receipt["weight_application_count"], 1)
                self.assertTrue(math.isclose(receipt["event_sumw"], math.fsum(weights), rel_tol=2e-13, abs_tol=0))
                self.assertTrue(math.isclose(receipt["pair_sumw"], expected, rel_tol=2e-13, abs_tol=0))
                with uproot.open(self.output) as source:
                    for name in ("ShiftDimuonVertex_bdtScore", "ShiftDimuonVertex_bdtScoreVsMass",
                                 "ShiftDimuonVertexBoth-Both_bdtScore"):
                        histogram = source["dimuon_bdt/" + name]
                        self.assertGreater(np.sum(histogram.values(flow=True)), 0)
                        self.assertTrue(math.isclose(np.sum(histogram.values(flow=True)), expected,
                                                     rel_tol=2e-13, abs_tol=0), name)
                        self.assertTrue(math.isclose(np.sum(histogram.variances(flow=True)),
                            math.fsum(value * value for value in pairs), rel_tol=2e-13, abs_tol=0), name)
                    self.assertTrue(math.isclose(source["cutFlow"].values()[0], math.fsum(weights),
                                                 rel_tol=2e-13, abs_tol=0))
                    self.assertEqual(np.sum(source["dimuon_bdt/ShiftDimuonVertex_bdtScoreGenuine"].values(flow=True)), 0)


if __name__ == "__main__":
    unittest.main()
