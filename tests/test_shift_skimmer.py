# Code style: AI-code. AI-generated native skimmer regression test.
from array import array
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import ROOT
except ImportError:
    ROOT = None


TENANT = Path(__file__).resolve().parents[1]
BINARY_PATH = os.environ.get("SHIFT_JPSI_SKIMMER")
BINARY = Path(BINARY_PATH).expanduser().resolve() if BINARY_PATH else None
DEFINITION = runpy.run_path(TENANT / "configs/shift_jpsi_dimuon.py")["jpsi_dimuon"]
COLLECTION = DEFINITION["collection"]
COUNT = f"n{COLLECTION}"
FLOAT_FIELDS = ("mass", "pt", "pz", "eta", "phi", "vx", "vy", "vz", "chi2", "normalizedChi2", "dca")
INT_FIELDS = ("topologyMin", "topologyMax", "isOS", "dcaValid")
EVENT_BASE = 2**63 + 12345


def float_around(value, steps=0):
    bits, = struct.unpack("I", struct.pack("f", value))
    number, = struct.unpack("f", struct.pack("I", bits + steps))
    return number


def create_input(path, mc=False, invalid=False):
    output = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("Events", "Events")
    scalars = {
        "run": (array("I", [357900]), "i"),
        "luminosityBlock": (array("I", [42]), "i"),
        "event": (array("Q", [0]), "l"),
        COUNT: (array("i" if mc else "I", [0]), "I" if mc else "i"),
    }
    if mc:
        scalars["genWeight"] = (array("f", [0]), "F")
        scalars["shiftSamplingGenWeight"] = (array("d", [0]), "D")
    for name, (buffer, leaf_code) in scalars.items():
        tree.Branch(name, buffer, f"{name}/{leaf_code}")

    candidates = {}
    for fields, array_type, leaf_code in ((FLOAT_FIELDS, "f", "F"), (INT_FIELDS + ("muonIdx1",), "i", "I")):
        for field in fields:
            name = f"{COLLECTION}_{field}"
            candidates[field] = array(array_type, [0] * 4)
            tree.Branch(name, candidates[field], f"{name}[{COUNT}]/{leaf_code}")
    lower, upper = DEFINITION["mass"]
    events = (
        ((1.2, 800), (float_around(lower, 1), 12), (float_around(upper, 1), 900), (float_around(upper), 35)),
        ((float_around(lower), 700), (9.0, 750)),
        ((float("nan") if invalid else 3.1, 56),),
        (),
    )
    for event_index, pairs in enumerate(events):
        scalars["event"][0][0] = EVENT_BASE + event_index
        scalars[COUNT][0][0] = len(pairs)
        if mc:
            scalars["genWeight"][0][0] = event_index - 1
            scalars["shiftSamplingGenWeight"][0][0] = event_index - 6.125
        for index, (mass, pt) in enumerate(pairs):
            for field in FLOAT_FIELDS:
                candidates[field][index] = pt
            for field in INT_FIELDS + ("muonIdx1",):
                candidates[field][index] = index + 1
            candidates["mass"][index] = mass
        tree.Fill()
    tree.Write()

    side_tree = ROOT.TTree("Runs", "Runs")
    original_count = array("q", [4])
    side_tree.Branch("genEventCount", original_count, "genEventCount/L")
    side_tree.Fill()
    side_tree.Write()
    output.Close()


class ShiftSkimmerTest(unittest.TestCase):
    def setUp(self):
        if ROOT is None:
            self.skipTest("Requires the native ROOT runtime")
        if BINARY is None:
            self.skipTest("Set SHIFT_JPSI_SKIMMER to the built SHIFT skimmer")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.source = self.directory / "input.root"
        self.dest = self.directory / "skim.root"

    def skim(self, kind="collision-data"):
        return subprocess.run(
            [str(BINARY), "--config", str(TENANT / "configs/shift_skimmer_config.py"),
             "--input_path", str(self.source), "--output_trees_path", str(self.dest), "--input_kind", kind],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def check_skim(self, mc=False):
        create_input(self.source, mc=mc)
        result = self.skim("mc" if mc else "collision-data")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.dest.stat().st_mode & 0o777, 0o600)
        with ROOT.TFile.Open(str(self.dest), "READ") as output:
            self.assertEqual({key.GetName() for key in output.GetListOfKeys()}, {"Events", "ShiftJpsiSkimMetadata"})
            tree = output.Get("Events")
            weights = {"genWeight", "shiftSamplingGenWeight"} if mc else set()
            expected = {"run", "luminosityBlock", "event", COUNT} | weights
            for fields, kind in ((FLOAT_FIELDS, "Float_t"), (INT_FIELDS, "Int_t")):
                for field in fields:
                    name = f"{COLLECTION}_{field}"
                    expected.add(name)
                    leaf = tree.GetLeaf(name)
                    self.assertEqual(leaf.GetTypeName(), kind)
                    self.assertEqual(leaf.GetLeafCount().GetName(), COUNT)
            self.assertEqual({branch.GetName() for branch in tree.GetListOfBranches()}, expected)
            self.assertEqual(tree.GetLeaf(COUNT).GetTypeName(), "Int_t" if mc else "UInt_t")
            lower, upper = DEFINITION["mass"]
            expected_events = (
                (EVENT_BASE, (float_around(lower, 1), float_around(upper)), (12, 35), (2, 4), -6.125),
                (EVENT_BASE + 2, (float_around(3.1),), (56,), (1,), -4.125),
            )
            self.assertEqual(tree.GetEntries(), len(expected_events))
            for row, (identity, masses, points, topologies, weight) in zip(tree, expected_events):
                self.assertEqual((row.run, row.luminosityBlock, row.event), (357900, 42, identity))
                self.assertEqual(getattr(row, COUNT), len(masses))
                self.assertEqual(list(getattr(row, f"{COLLECTION}_mass")), list(masses))
                self.assertEqual(list(getattr(row, f"{COLLECTION}_pt")), list(points))
                self.assertEqual(list(getattr(row, f"{COLLECTION}_topologyMin")), list(topologies))
                if mc:
                    self.assertEqual(row.shiftSamplingGenWeight, weight)
            self.assertEqual(json.loads(str(output.Get("ShiftJpsiSkimMetadata").GetString())), {
                "schema": "shift-jpsi-skim-v1", "input_kind": "mc" if mc else "collision-data",
                "mass_min_gev": lower, "mass_max_gev": upper, "endpoints": "inclusive", "original_events": 4,
                "selected_events": 2, "selected_pairs": 3, "weight_branch": "shiftSamplingGenWeight" if mc else "",
            })

    def test_collision_data_round_trip(self):
        self.check_skim()

    def test_mc_native_weight_round_trip(self):
        self.check_skim(mc=True)

    def test_invalid_input_leaves_no_visible_output(self):
        create_input(self.source, invalid=True)
        result = self.skim()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical candidate mass is nonfinite", result.stdout)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.directory.glob(".shift_jpsi_pending_*")), [])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} /path/to/shift_skimmer")
    BINARY = Path(sys.argv.pop()).resolve(strict=True)
    unittest.main()
