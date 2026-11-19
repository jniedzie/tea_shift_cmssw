# Code style: AI-code. AI-generated conservative truth-label recovery tests.
"""Truth-label failure modes; no classifier inference or model tuning."""

import copy
import math
import unittest

from utils.shift_classifier_labels import recover_pair_label


class TruthRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.event = dict(run=7, luminosityBlock=2, event=123, original_request_sha256="a" * 64,
                          ShiftMuon_hitSimTrackId=[21, 22],
                          ShiftMuon_hitTruthMatchedLayers=[6, 6],
                          ShiftMuon_hitTruthPurity=[1.0, 1.0],
                          ShiftMuon_hitGenPartIdx=[-1, -1],
                          GenPart_pdgId=[], GenPart_status=[])
        self.truth = dict(event_id=[7, 2, 123], original_request_sha256="a" * 64, tracks=[
            dict(track_id=21, pdg_id=13, vertex_index=4, vertex_cm=[0., 0., 100.]),
            dict(track_id=22, pdg_id=-13, vertex_index=4, vertex_cm=[0., 0., 100.])])

    def recover(self, truth=None, label=-1, reason="unknown_match"):
        return recover_pair_label(self.event, 0, 1, label, reason, truth)

    def test_known_genuine_preserved_even_with_bad_reconstructed_vertex(self):
        self.event["ShiftDimuonVertex_vz"] = [-9000.]
        self.event["ShiftDimuonVertex_normalizedChi2"] = [1.e9]
        y, why, detail = self.recover(label=1, reason="common_vertex")
        self.assertEqual((y, why), (1, "common_vertex"))
        self.assertFalse(detail["recovered"])

    def test_known_negative_preserved(self):
        self.assertEqual(self.recover(self.truth, 0, "different_vertices")[:2],
                         (0, "different_vertices"))

    def test_no_simulation_vertex_remains_unknown(self):
        y, why, detail = self.recover()
        self.assertEqual((y, why), (-1, "unknown_match"))
        self.assertEqual(detail["status"], "simulation_vertex_unavailable")

    def test_exact_duplicate_without_gen_is_negative(self):
        self.event["ShiftMuon_hitSimTrackId"] = [21, 21]
        self.assertEqual(self.recover()[:2], (0, "same_hit_sim_track"))

    def test_weak_or_rejected_hit_association_never_recovers(self):
        for branch, value in [("ShiftMuon_hitTruthPurity", .74),
                              ("ShiftMuon_hitTruthPurity", math.nan),
                              ("ShiftMuon_hitTruthMatchedLayers", 2),
                              ("ShiftMuon_hitSimTrackId", -1)]:
            event = copy.deepcopy(self.event)
            event[branch][0] = value
            result = recover_pair_label(event, 0, 1, -1, "unknown_match", self.truth)
            self.assertEqual(result[0], -1)

    def test_original_simulation_vertices_recover_true_and_false(self):
        self.assertEqual(self.recover(self.truth)[:2], (1, "common_hit_sim_vertex"))
        self.truth["tracks"][1]["vertex_cm"][2] += 1.
        self.assertEqual(self.recover(self.truth)[:2], (0, "different_hit_sim_vertices"))

    def test_tolerance_gap_stays_unknown(self):
        self.truth["tracks"][1]["vertex_cm"][2] += .05
        self.assertEqual(self.recover(self.truth)[0], -1)

    def test_wrong_event_and_duplicate_record_fail_closed(self):
        self.truth["event_id"][2] += 1
        with self.assertRaisesRegex(ValueError, "different run/lumi/event"):
            self.recover(self.truth)
        self.truth["event_id"][2] -= 1
        self.truth["tracks"].append(self.truth["tracks"][0])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.recover(self.truth)

    def test_same_event_numbers_from_another_response_cannot_relabel(self):
        self.truth["original_request_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "original detector request"):
            self.recover(self.truth)
        self.truth["original_request_sha256"] = "a" * 64
        del self.event["original_request_sha256"]
        with self.assertRaisesRegex(ValueError, "original detector request"):
            self.recover(self.truth)

    def test_missing_nonmuon_and_invalid_vertex_do_not_recover(self):
        variants = []
        missing = copy.deepcopy(self.truth)
        missing["tracks"].pop()
        variants.append(missing)
        nonmuon = copy.deepcopy(self.truth)
        nonmuon["tracks"][0]["pdg_id"] = 211
        variants.append(nonmuon)
        nonfinite = copy.deepcopy(self.truth)
        nonfinite["tracks"][0]["vertex_cm"][0] = math.nan
        variants.append(nonfinite)
        invalid = copy.deepcopy(self.truth)
        invalid["tracks"][0]["vertex_index"] = -1
        variants.append(invalid)
        for truth in variants:
            self.assertEqual(self.recover(truth)[0], -1)


if __name__ == "__main__":
    unittest.main()
