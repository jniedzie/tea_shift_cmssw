import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import ROOT
import uproot

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "utils"))
from prepare_shift_bdt_plots import bin_scale, prepare


def histograms(path, weighted, raw):
    output = ROOT.TFile(str(path), "RECREATE")
    output.mkdir("dimuon_bdt").cd()
    for name, value in (("bdtScore", weighted), ("bdtScoreUnweighted", raw)):
        histogram = ROOT.TH1D("ShiftDimuonVertex_" + name, "", 100, 0, 1)
        histogram.Sumw2()
        histogram.Fill(0.5, value)
        histogram.Write()
    output.Close()


class BdtPlotPreparationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.first = self.directory / "first.root"
        self.second = self.directory / "second.root"
        histograms(self.first, 10, 2)
        histograms(self.second, 20, 3)
        self.manifest_path = self.directory / "inputs.json"
        self.output = self.directory / "combined"
        self.manifest = {
            "schema": "shift-dimuon-bdt-bin-inputs-v1", "complete": True,
            "samples": [
                {"name": "qcd_low", "process": "qcd", "label": "QCD", "histogram_path": str(self.first),
                 "cross_section_pb": 100, "full_gen_sumw": 1000},
                {"name": "qcd_high", "process": "qcd", "label": "QCD", "histogram_path": str(self.second),
                 "cross_section_pb": 2, "full_gen_sumw": 100},
            ],
        }

    def run_preparation(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return prepare(self.manifest_path, self.output)

    def test_exposure_scale_before_process_sum_and_raw_counts_preserved(self):
        receipt = self.run_preparation()
        self.assertTrue(receipt["complete"])
        self.assertEqual(len(receipt["samples"]), 1)
        self.assertEqual([row["applied_shape_scale"] for row in receipt["inputs"]], [0.1, 0.02])
        with uproot.open(self.output / "qcd.root") as source:
            weighted = source["dimuon_bdt/ShiftDimuonVertex_bdtScore"]
            raw = source["dimuon_bdt/ShiftDimuonVertex_bdtScoreUnweighted"]
            self.assertAlmostEqual(np.sum(weighted.values(flow=True)), 1.4)
            self.assertAlmostEqual(np.sum(weighted.variances(flow=True)), 1.16)
            self.assertEqual(np.sum(raw.values(flow=True)), 5)

    def test_selected_sum_is_not_an_implicit_denominator(self):
        self.manifest["samples"][0].pop("full_gen_sumw")
        self.manifest["samples"][0]["selected_sumw"] = 10
        with self.assertRaises(KeyError):
            self.run_preparation()
        self.assertFalse(json.loads((self.output / "plot_inputs.json").read_text())["complete"])

    def test_signal_mass_lifetime_samples_stay_separate(self):
        for index, row in enumerate(self.manifest["samples"]):
            row["process"] = "signal"
            row["name"] = f"signal_m{index}_tau"
        receipt = self.run_preparation()
        self.assertEqual(len(receipt["samples"]), 2)
        self.assertEqual([row["applied_shape_scale"] for row in receipt["inputs"]], [1, 1])

    def test_duplicate_input_rejected(self):
        self.manifest["samples"][1]["name"] = self.manifest["samples"][0]["name"]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.run_preparation()

    def test_incomplete_input_manifest_rejected(self):
        self.manifest["complete"] = False
        with self.assertRaisesRegex(ValueError, "complete BDT"):
            self.run_preparation()
        self.assertFalse(self.output.exists())

    def test_checksum_mismatch_rejected(self):
        self.manifest["samples"][0]["histogram_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_preparation()

    def test_nonpositive_full_gen_exposure_rejected(self):
        self.manifest["samples"][0]["full_gen_sumw"] = 0
        with self.assertRaisesRegex(ValueError, "full GEN exposure"):
            self.run_preparation()

    def test_existing_output_directory_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_preparation()


if __name__ == "__main__":
    unittest.main()
