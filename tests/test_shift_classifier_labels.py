# Code style: AI-code. AI-generated conservative truth-label recovery tests.
"""Truth-label failure modes; no classifier inference or model tuning."""

import copy
import math
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from utils.shift_classifier_labels import (recover_pair_label, pair_task_truth, task_pair_label,
    task_labels_for_data, refine_frozen_label_reasons, TASK_LABEL_SCHEMA)


class TruthRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.event = dict(run=7, luminosityBlock=2, event=123, original_request_sha256="a" * 64,
                          ShiftMuon_hitSimTrackId=[21, 22],
                          ShiftMuon_hitTruthMatchedLayers=[6, 6],
                          ShiftMuon_hitTruthPurity=[1.0, 1.0],
                          ShiftMuon_hitGenPartIdx=[-1, -1],
                          GenPart_pdgId=[], GenPart_status=[])
        self.truth = dict(event_id=[7, 2, 123], original_request_sha256="a" * 64, tracks=[
            dict(track_id=21, pdg_id=13, vertex_index=4, vertex_cm=[0., 0., 100.],
                 origin_semantics="validated_physical_muon_birth", physical_muon_id="muon-a"),
            dict(track_id=22, pdg_id=-13, vertex_index=4, vertex_cm=[0., 0., 100.],
                 origin_semantics="validated_physical_muon_birth", physical_muon_id="muon-b")])

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

    def test_raw_transport_vertices_do_not_certify_physical_birth(self):
        del self.truth["tracks"][0]["origin_semantics"]
        y, _, detail = self.recover(self.truth)
        self.assertEqual(y, -1)
        self.assertEqual(detail["status"], "physical_birth_unqualified")

    def test_distinct_track_ids_do_not_by_themselves_mean_distinct_muons(self):
        del self.truth["tracks"][0]["physical_muon_id"]
        self.assertEqual(self.recover(self.truth)[0], -1)
        self.truth["tracks"][0]["physical_muon_id"] = "muon-b"
        self.assertEqual(self.recover(self.truth)[:2], (0, "same_physical_muon"))

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


class SeparateTaskTest(unittest.TestCase):
    setUp = TruthRecoveryTest.setUp
    def test_duplicate_and_different_birth_are_separate_targets(self):
        expected = [('common_vertex',1,1,1),('different_vertices',0,1,0),
                    ('same_muon_twice',0,0,-1),('same_hit_sim_track',0,0,-1),
                    ('same_physical_muon',0,0,-1)]
        for reason,original,identity,origin in expected:
            self.assertEqual(task_pair_label(original,reason,'distinct-muons')[0],identity)
            self.assertEqual(task_pair_label(original,reason,'vertex-origin')[0],origin)

    def test_ambiguous_old_negatives_and_missing_birth_are_not_guessed(self):
        for task in ('distinct-muons','vertex-origin'):
            self.assertEqual(task_pair_label(0,'frozen_old_label',task)[0],-1)
            self.assertEqual(task_pair_label(-1,'unknown_vertex',task)[0],-1)
        self.assertEqual(task_pair_label(-1,'vertex_tolerance_gap','distinct-muons')[0],1)
        self.assertEqual(task_pair_label(-1,'vertex_tolerance_gap','vertex-origin')[0],-1)
        with self.assertRaisesRegex(ValueError,'contradicts'):
            task_pair_label(1,'different_vertices','vertex-origin')

    def test_bad_fit_of_common_origin_stays_positive_for_both_tasks(self):
        self.event['ShiftDimuonVertex_vz']=[-1e9]
        self.event['ShiftDimuonVertex_normalizedChi2']=[1e9]
        result=pair_task_truth(self.event,0,1,1,'common_vertex')
        self.assertEqual((result['distinct_muons_y'],result['vertex_origin_y']),(1,1))

    def test_identity_certificate_does_not_certify_birth(self):
        for track in self.truth['tracks']:
            del track['origin_semantics']
            track['identity_semantics']='validated_physical_muon_identity'
        result=pair_task_truth(self.event,0,1,-1,'unknown_match',self.truth)
        self.assertEqual((result['distinct_muons_y'],result['vertex_origin_y']),(1,-1))
        self.truth['tracks'][1]['physical_muon_id']='muon-a'
        result=pair_task_truth(self.event,0,1,-1,'unknown_match',self.truth)
        self.assertEqual((result['distinct_muons_y'],result['vertex_origin_y']),(0,-1))
        self.assertEqual(result['vertex_origin_reason'],'unknown_excluded_duplicate_muons')

    def test_raw_simtrack_ids_and_vertices_stay_unknown_for_new_tasks(self):
        for track in self.truth['tracks']:del track['origin_semantics']
        result=pair_task_truth(self.event,0,1,-1,'unknown_match',self.truth)
        self.assertEqual((result['distinct_muons_y'],result['vertex_origin_y']),(-1,-1))
        self.truth['original_request_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'original detector request'):
            pair_task_truth(self.event,0,1,-1,'unknown_match',self.truth)

    def test_explicit_task_arrays_require_schema_and_cannot_reverse_known_truth(self):
        data=dict(y=np.array([1,-1]),label_reason=np.array(['common_vertex','unknown_match']),
                  distinct_muons_y=np.array([1,1]),distinct_muons_reason=np.array(['distinct_common_origin','certified_distinct_physical_muons']),
                  task_label_schema=np.array([TASK_LABEL_SCHEMA]*2))
        original=data['y'].copy();labels,_=task_labels_for_data(data,'distinct-muons')
        np.testing.assert_array_equal(labels,[1,1]);np.testing.assert_array_equal(data['y'],original)
        data['distinct_muons_y'][0]=0;data['distinct_muons_reason'][0]='duplicate_same_muon'
        with self.assertRaisesRegex(ValueError,'contradicts'):
            task_labels_for_data(data,'distinct-muons')
        duplicate=dict(y=np.array([0]),label_reason=np.array(['same_muon_twice']),vertex_origin_y=np.array([1]),
                       vertex_origin_reason=np.array(['common_origin']),task_label_schema=np.array([TASK_LABEL_SCHEMA]))
        with self.assertRaisesRegex(ValueError,'contradicts'):
            task_labels_for_data(duplicate,'vertex-origin')


class ArchivedReasonTest(unittest.TestCase):
    def fixture(self,directory):
        directory=Path(directory);names=np.array(['x'+str(i)for i in range(119)])
        reco=dict(X=np.zeros((1,119)),ids=np.array([[7,2,123]]),pair_index=np.array([0]),
                  muon_indices=np.array([[0,1]]),source_index=np.array([0]),process=np.array(['qcd']),feature_names=names)
        np.savez_compressed(directory/'reco.npz',**reco);np.savez_compressed(directory/'labels.npz',y=np.array([0]),reason=np.array(['different_vertices']))
        sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        manifest=dict(reco_sha256=sha(directory/'reco.npz'),labels_sha256=sha(directory/'labels.npz'),
                      inputs=[dict(index=0,path='/eos/home-j/jniedzie/source.root')])
        (directory/'inputs.json').write_text(json.dumps(manifest))
        return dict(X=np.zeros((1,121)),ids=reco['ids'],pair_index=reco['pair_index'],muon_indices=reco['muon_indices'],
                    process=reco['process'],feature_names=np.r_[names,['pointing1','pointing2']],y=np.array([0]),
                    label_reason=np.array(['frozen_old_label']),nano_path=np.array(['/eos/user/j/jniedzie/source.root']))

    def test_refinement_preserves_binary_y_and_checks_exact_source_and_features(self):
        with tempfile.TemporaryDirectory()as directory:
            data=self.fixture(directory);refined,proof=refine_frozen_label_reasons(data,directory)
            self.assertEqual(proof['refined_rows'],1);self.assertTrue(proof['original_binary_y_unchanged'])
            self.assertEqual(refined['label_reason'][0],'different_vertices');self.assertEqual(data['label_reason'][0],'frozen_old_label')
            self.assertEqual(task_labels_for_data(refined,'distinct-muons')[0][0],1)
            data['muon_indices']=np.array([[-1,-1]])
            _,proof=refine_frozen_label_reasons(data,directory);self.assertEqual(proof['unavailable_muon_index_rows'],1)
            data['muon_indices']=np.array([[1,0]])
            with self.assertRaisesRegex(ValueError,'muon-index parity'):refine_frozen_label_reasons(data,directory)
            data['muon_indices']=np.array([[-1,-1]])
            data['X'][0,5]=1
            with self.assertRaisesRegex(ValueError,'parity failed'):refine_frozen_label_reasons(data,directory)
            data['X'][0,5]=0;data['nano_path']=np.array(['/eos/user/j/jniedzie/other.root'])
            _,proof=refine_frozen_label_reasons(data,directory);self.assertEqual(proof['refined_rows'],0)


if __name__ == "__main__":
    unittest.main()
