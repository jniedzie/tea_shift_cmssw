import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from array import array

import numpy as np
import ROOT
import uproot

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "utils"))
from shift_bdt_histogrammer import run

CLASSIFIER = REPOSITORY.parent / "CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier"


def fixture(path, *, omit_score=False, score_type="D", nonfinite=False, data=False, zero_pairs=False):
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
    buffers = {}
    for name, kind in fields.items():
        count = "nShiftDimuonVertex" if name.startswith("ShiftDimuonVertex_") else (
            "nShiftMuon" if name.startswith("ShiftMuon_") else "nGenPart")
        buffers[name] = array({"I": "i", "D": "d", "F": "f"}[kind], [0] * 8)
        tree.Branch(name, buffers[name], f"{name}[{count}]/{kind}")
    weight = array("d", [0])
    if not data:
        tree.Branch("shiftSamplingGenWeight", weight, "shiftSamplingGenWeight/D")
    for event_index, (pairs, event_weight) in enumerate(((0, -2), (2, 0), (1, 3), (1, -1), (1, 2))):
        counts["nShiftMuon"][0] = counts["nGenPart"][0] = 2
        counts["nShiftDimuonVertex"][0] = pairs = 0 if zero_pairs else pairs
        weight[0] = event_weight
        for index in range(2):
            for name in fields:
                buffers[name][index] = 0
            for name, value in {
                "ShiftDimuonVertex_muonIdx2": 1, "ShiftDimuonVertex_mass": 3.1,
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
        tree.Fill()
    tree.Write()
    file.Close()


class BdtHistogramTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.input = self.directory / "input.root"
        self.output = self.directory / "hist.root"

    def test_all_pairs_weights_once_and_mc_subsets_partition(self):
        fixture(self.input)
        receipt = run(str(self.input), self.output, CLASSIFIER, chunk_events=2)
        self.assertTrue(receipt["complete"])
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
            self.assertNotIn("cutFlow", source)
            self.assertNotIn("rawEventsCutFlow", source)

    def test_missing_score_fails_even_zero_pairs(self):
        fixture(self.input, omit_score=True, zero_pairs=True)
        with self.assertRaisesRegex(ValueError, "Missing required saved BDT score"):
            run(str(self.input), self.output, CLASSIFIER)
        self.assertFalse(self.output.exists())

    def test_float_score_is_rejected(self):
        fixture(self.input, score_type="F")
        with self.assertRaisesRegex(ValueError, "must be Double"):
            run(str(self.input), self.output, CLASSIFIER)

    def test_invalid_score_retains_failure_receipt_without_publishing(self):
        fixture(self.input, nonfinite=True)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            run(str(self.input), self.output, CLASSIFIER)
        self.assertFalse(self.output.exists())
        self.assertTrue(Path(str(self.output) + ".incomplete").exists())
        self.assertFalse(json.loads(Path(str(self.output) + ".json").read_text())["complete"])

    def test_data_path_reads_no_truth_and_uses_unit_weight(self):
        fixture(self.input, data=True)
        receipt = run(str(self.input), self.output, self.directory / "absent", sample_kind="data")
        self.assertEqual(receipt["pairs"], 5)
        self.assertEqual(receipt["pair_sumw"], 5)
        with uproot.open(self.output) as source:
            self.assertNotIn("dimuon_bdt/ShiftDimuonVertex_bdtScoreGenuine", source)

    def test_zero_pairs_creates_empty_complete_histograms(self):
        fixture(self.input, zero_pairs=True)
        receipt = run(str(self.input), self.output, CLASSIFIER)
        self.assertEqual((receipt["events"], receipt["pairs"]), (5, 0))
        self.assertTrue(receipt["complete"])

    def test_existing_output_not_overwritten(self):
        fixture(self.input)
        self.output.write_text("original")
        with self.assertRaises(FileExistsError):
            run(str(self.input), self.output, CLASSIFIER)
        self.assertEqual(self.output.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
