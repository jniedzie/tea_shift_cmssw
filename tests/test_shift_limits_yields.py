# Code style: AI-code. AI-generated checks of exclusive event accounting.
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_limits_yields import CATEGORIES, count_sample, event_category, predict


class LimitYieldTests(unittest.TestCase):
    def test_topologies_partition_all_pairs(self):
        categories = [event_category(low, high) for low in range(5) for high in range(low, 5)]
        self.assertEqual(len(categories), 15)
        self.assertEqual(categories.count("Other"), 11)
        for name in CATEGORIES[:-1]:
            self.assertEqual(categories.count(name), 1)
        with self.assertRaises(ValueError):
            event_category(3, 0)

    def test_multiple_pairs_count_once_without_selection_branches(self):
        import ROOT
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.root"
            output = ROOT.TFile(str(path), "RECREATE")
            tree = ROOT.TTree("Events", "Events")
            run, lumi, event, pairs = array('I', [1]), array('I', [1]), array('Q', [0]), array('I', [0])
            weight, minimum, maximum = array('f', [0]), array('i', [0, 0]), array('i', [0, 0])
            for name, value, leaf in (("run", run, "i"), ("luminosityBlock", lumi, "i"),
                                      ("event", event, "l"), ("nShiftDimuonVertex", pairs, "i"),
                                      ("genWeight", weight, "F")):
                tree.Branch(name, value, name + "/" + leaf)
            tree.Branch("ShiftDimuonVertex_topologyMin", minimum, "ShiftDimuonVertex_topologyMin[nShiftDimuonVertex]/I")
            tree.Branch("ShiftDimuonVertex_topologyMax", maximum, "ShiftDimuonVertex_topologyMax[nShiftDimuonVertex]/I")
            for identity, native_weight, number in ((1, 2., 2), (2, 3., 1), (3, 4., 0)):
                event[0], weight[0], pairs[0] = identity, native_weight, number
                minimum[0], maximum[0] = (0, 2) if identity == 1 else (2, 2)
                minimum[1], maximum[1] = 0, 3
                tree.Fill()
            tree.Write(); output.Close()
            sample = dict(name="toy", files=[str(path)], weight_branch="genWeight",
                          full_gen_sumw=20., selected_cross_section_pb=2.)
            counts = count_sample(sample)
            self.assertEqual(counts['events'], 3)
            self.assertEqual(counts['multiple_dimuon_events'], 1)
            self.assertEqual(counts['no_dimuon_events'], 1)
            self.assertEqual(counts['categories']['Near-Both']['sumw'], 2.)
            self.assertEqual(counts['categories']['Near-Far']['sumw'], 0.)
            yields = predict(sample, counts, 3000.)
            self.assertEqual(yields['yields']['Near-Both'], 600.)
            self.assertEqual(sum(yields['yields'].values()), 1500.)


if __name__ == '__main__':
    unittest.main()
