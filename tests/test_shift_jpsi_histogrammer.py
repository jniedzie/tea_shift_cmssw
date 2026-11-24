# Code style: AI-code. AI-generated regression test.
"""Exercise the unchanged SHIFT histogram names and rejected skim inputs."""

from array import array
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("SHIFT_JPSI_HISTOGRAMMER"), "Set the built SHIFT histogrammer")
class ShiftJpsiHistogrammerTest(unittest.TestCase):
    def setUp(self):
        import ROOT

        self.root = ROOT
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def skim(self, events, *, mc=False, extra_branch=False, seal=True, count_kind="i", **metadata):
        source = self.scratch / "input.root"
        output = self.root.TFile.Open(str(source), "RECREATE")
        tree = self.root.TTree("Events", "Synthetic candidate fixtures")
        run = array("I", [380000])
        lumi = array("I", [8])
        event = array("Q", [0])
        count = array(count_kind, [0])
        tree.Branch("run", run, "run/i")
        tree.Branch("luminosityBlock", lumi, "luminosityBlock/i")
        tree.Branch("event", event, "event/l")
        tree.Branch("nShiftDimuonVertex", count, "nShiftDimuonVertex/" + ("i" if count_kind == "I" else "I"))
        float_fields = "mass pt pz eta phi vx vy vz chi2 normalizedChi2 dca".split()
        int_fields = "topologyMin topologyMax isOS dcaValid".split()
        buffers = {}
        for fields, buffer_type, leaf_type in ((float_fields, "f", "F"), (int_fields, "i", "I")):
            for field in fields:
                name = f"ShiftDimuonVertex_{field}"
                buffers[field] = array(buffer_type, [0] * 4)
                tree.Branch(name, buffers[field], f"{name}[nShiftDimuonVertex]/{leaf_type}")
        if extra_branch:
            forbidden = array("f", [10.0])
            tree.Branch("unapprovedCentralTrackPt", forbidden, "unapprovedCentralTrackPt/F")
        weight = array("d", [0.0])
        if mc:
            tree.Branch("shiftSamplingGenWeight", weight, "shiftSamplingGenWeight/D")
        for event_id, (pairs, event_weight) in enumerate(events, start=1):
            event[0] = event_id
            count[0] = len(pairs)
            weight[0] = event_weight
            for index, (mass, topology) in enumerate(pairs):
                for field in float_fields:
                    buffers[field][index] = 1.0
                buffers["mass"][index] = mass
                buffers["topologyMin"][index], buffers["topologyMax"][index] = topology
                buffers["isOS"][index] = 1
                buffers["dcaValid"][index] = 1
            tree.Fill()
        tree.Write()
        if seal:
            fields = {
                "schema": "shift-jpsi-skim-v1",
                "input_kind": "mc" if mc else "collision-data",
                "mass_min_gev": 2.6,
                "mass_max_gev": 3.6,
                "endpoints": "inclusive",
                "original_events": len(events) + 1,
                "selected_events": len(events),
                "selected_pairs": sum(len(pairs) for pairs, _ in events),
                "weight_branch": "shiftSamplingGenWeight" if mc else "",
            }
            fields.update(metadata)
            self.root.TObjString(json.dumps(fields)).Write("ShiftJpsiSkimMetadata")
        output.Close()
        return source

    def histogram(self, source, *, input_override=None):
        config = self.scratch / "config.py"
        output = self.scratch / "histograms.root"
        output.unlink(missing_ok=True)
        config.write_text(
            "import shift_paths\n"
            f"shift_paths.histogram_input_path = {str(source)!r}\n"
            f"shift_paths.histogram_output_path = {str(output)!r}\n"
            "from shift_histogrammer_config import *\n"
        )
        paths = ["--input_path", str(input_override)] if input_override else []
        result = subprocess.run(
            [os.environ["SHIFT_JPSI_HISTOGRAMMER"], "--config", str(config), *paths],
            capture_output=True, text=True, timeout=60,
        )
        return result, output

    def histogram_rejected(self, source, message, **mode):
        result, output = self.histogram(source, **mode)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertFalse(output.exists())

    def test_binned_pair_distributions_and_signed_mc_weights(self):
        events = [([(2.6000001430511475, (2, 2)), (3.2, (1, 1))], -2.0),
                  ([(3.5999999046325684, (2, 2))], 5.0)]
        for mc in (False, True):
            with self.subTest(mc=mc):
                result, output = self.histogram(self.skim(events, mc=mc, count_kind="I"))
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
                root = self.root.TFile.Open(str(output))
                try:
                    for name in ("mass", "pt", "logMass", "vertexR", "p_variable"):
                        self.assertAlmostEqual(root.Get(f"dimuon/ShiftDimuonVertex_{name}").Integral(), 1 if mc else 3)
                    self.assertAlmostEqual(root.Get("dimuon/ShiftDimuonVertexBoth-Both_mass").Integral(), 3 if mc else 2)
                    self.assertAlmostEqual(root.Get("dimuon/ShiftDimuonVertexOther_mass").Integral(), -2 if mc else 1)
                    self.assertEqual(root.Get("rawEventsCutFlow").GetBinContent(1), 2)
                    self.assertFalse(root.Get("dimuon/ShiftDimuonVertex_refittedMass"))
                finally:
                    root.Close()

    def test_empty_skim_retains_the_existing_histogram_names(self):
        result, output = self.histogram(self.skim([]))
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        root = self.root.TFile.Open(str(output))
        try:
            self.assertEqual(root.Get("dimuon/ShiftDimuonVertexBoth-Both_mass").Integral(), 0)
        finally:
            root.Close()

    def test_rejects_unapproved_data_branches(self):
        events = [([(3.1, (2, 2))], 1.0)]
        self.histogram_rejected(self.skim(events, extra_branch=True), "unexpected or missing data branches")

    def test_rejects_replacing_the_configured_skim_at_launch(self):
        events = [([(3.1, (2, 2))], 1.0)]
        skim = self.scratch / "skim.root"
        self.skim(events).rename(skim)
        raw = self.skim(events, seal=False)
        self.histogram_rejected(skim, "Set the skim input path in shift_paths.py", input_override=raw)

    def test_rejects_mismatched_seal_and_weight(self):
        from run_shift_jpsi_control import skim_counts, verify_merge

        events = [([(3.1, (2, 2))], 1.0)]
        with self.assertRaisesRegex(ValueError, "Skim metadata disagrees"):
            skim_counts(self.skim(events, mass_min_gev=2.5), 2, "collision-data", "")
        with self.assertRaisesRegex(ValueError, "Skim metadata disagrees"):
            skim_counts(self.skim(events, mc=True), 2, "mc", "genWeight")

        source = self.skim(events, selected_pairs=2)
        counts = skim_counts(source, 2, "collision-data", "")
        result, output = self.histogram(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaisesRegex(ValueError, "disagrees with the exact skim pair count"):
            verify_merge(output, *counts)
