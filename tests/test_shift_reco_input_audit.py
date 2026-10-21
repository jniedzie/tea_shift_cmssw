# Code style: AI-code. AI-generated independent CSC hit-association tests.
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_reco_input_audit import (associate_csc_hits, json_safe_report, validate_selected_association,
                                    CSC_RECHIT_COLLECTION, CSC_SEGMENT_COLLECTION)


def fixture():
    hits = [dict(layer=layer, position_cm=[1., 2., 0.], error_cm2=[.01, 0., .01]) for layer in range(1, 5)]
    crossings = [dict(layer=layer, track_id=7, position_cm=[0., 2., -1.], momentum_GeV=[1., 0., 1.], particle_type=13)
                 for layer in range(1, 5)]
    return hits, crossings


class HitAssociationTest(unittest.TestCase):
    def test_projection_and_exact_coordinate_duplicates(self):
        hits, truth = fixture()
        hits.append(copy.deepcopy(hits[0]))
        result = associate_csc_hits(hits, truth)
        self.assertEqual((result["track"], result["total"], result["matched"]), (7, 4, 4))
        self.assertEqual(result["purity"], 1.)

    def test_electron_shower_ambiguity_is_not_filtered(self):
        hits, truth = fixture()
        truth.append(dict(layer=1, track_id=8, position_cm=[1., 2., 0.], momentum_GeV=[0., 0., 1.], particle_type=11))
        result = associate_csc_hits(hits, truth)
        self.assertEqual((result["track"], result["ambiguous"], result["purity"]), (7, 1, .75))
        truth.append(dict(truth[-1], layer=2))
        self.assertEqual(associate_csc_hits(hits, truth)["track"], -1)

    def test_multiple_measured_positions_are_ambiguous_not_nearest(self):
        hits, truth = fixture()
        hits.extend([dict(hits[0], position_cm=[10., 2., 0.]), dict(hits[1], position_cm=[20., 2., 0.])])
        result = associate_csc_hits(hits, truth)
        self.assertEqual((result["track"], result["matched"], result["ambiguous"]), (-1, 2, 2))

    def test_missing_short_and_invalid_measurements_cannot_match(self):
        hits, truth = fixture()
        for measured, crossings in ((hits, []), (hits[:2], truth),
                                    ([dict(h, error_cm2=[0., 0., .01]) for h in hits], truth),
                                    (hits, [dict(t, momentum_GeV=[1., 0., 0.]) for t in truth])):
            self.assertEqual(associate_csc_hits(measured, crossings)["track"], -1)

    def test_tied_votes_and_chi2_boundary(self):
        hits, truth = fixture()
        for index in (2, 3):
            truth[index]["track_id"] = 8
        self.assertEqual(associate_csc_hits(hits, truth)["track"], -1)
        hits, truth = fixture()
        for crossing in truth:
            crossing.update(position_cm=[1.5, 2., 0.], momentum_GeV=[0., 0., 1.])
        self.assertEqual(associate_csc_hits(hits, truth)["track"], 7)
        for crossing in truth:
            crossing["position_cm"][0] = 1.500001
        self.assertEqual(associate_csc_hits(hits, truth)["track"], -1)

    def test_selected_identity_counts_fail_closed_and_purity_is_quantized(self):
        result = dict(track=7, total=39, matched=33, ambiguous=4, purity=33 / 39)
        stored = dict(hitSimTrackId=7, hitTruthLayers=39, hitTruthMatchedLayers=33,
                      hitTruthAmbiguousLayers=4, hitTruthPurity=.84619140625)
        proof = validate_selected_association(result, stored)
        self.assertTrue(proof["identity_and_layer_counts_exact"])
        for key in ("hitSimTrackId", "hitTruthLayers", "hitTruthMatchedLayers", "hitTruthAmbiguousLayers"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "association mismatch"):
                validate_selected_association(result, dict(stored, **{key: stored[key] + 1}))
        with self.assertRaisesRegex(ValueError, "quantized"):
            validate_selected_association(result, dict(stored, hitTruthPurity=.9))

    def test_nonfinite_raw_fields_are_retained_and_cannot_cast_votes(self):
        hits, truth = fixture()
        hits[0]["position_cm"][0] = float("nan")
        hits[1]["error_cm2"][0] = float("inf")
        result = associate_csc_hits(hits, truth)
        self.assertEqual((result["track"], result["matched"]), (-1, 2))
        report = json_safe_report(dict(raw_tracks=[dict(p_GeV=float("-inf"), hits=hits, association=result)]))
        json.dumps(report, allow_nan=False)
        self.assertEqual(len(report["raw_tracks"]), 1)
        self.assertEqual(len(report["raw_tracks"][0]["hits"]), 4)
        self.assertEqual(report["nonfinite_field_values"], [
            dict(path="/raw_tracks/0/p_GeV", value="-inf"),
            dict(path="/raw_tracks/0/hits/0/position_cm/0", value="nan"),
            dict(path="/raw_tracks/0/hits/1/error_cm2/0", value="+inf")])
        self.assertIsNone(report["raw_tracks"][0]["p_GeV"])


@unittest.skipUnless(importlib.util.find_spec("DataFormats"), "Requires native CMSSW FWLite")
class NativeDictionaryTest(unittest.TestCase):
    def test_mandatory_product_wrappers(self):
        import ROOT
        from DataFormats.FWLite import Handle
        ROOT.gSystem.Load("libFWCoreFWLite")
        ROOT.FWLiteEnabler.enable()
        for typename in (CSC_RECHIT_COLLECTION, CSC_SEGMENT_COLLECTION,
                         "edm::DetSetVector<StripDigiSimLink>", "std::vector<reco::Track>",
                         "std::vector<SimTrack>", "std::vector<SimVertex>", "std::vector<PSimHit>"):
            with self.subTest(typename=typename):
                Handle(typename)


if __name__ == "__main__":
    unittest.main()
