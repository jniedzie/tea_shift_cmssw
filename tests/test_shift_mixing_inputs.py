# Code style: AI-code. AI-generated regression test.
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_mixing import (CAMPAIGN, SCHEMA, event_identity, event_partition,
                                 first_noncopy_parent, iter_pool, truth_census,
                                 truth_chain, validate_weights, REQUIRED_MUON_FIELDS,
                                 build_inventory)


class MixingInputContractTest(unittest.TestCase):
    def test_other_detector_source_version_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "V10 natural-decay"):
            build_inventory("shift_detector_representative_20261005_v8")

    def test_modified_event_ledger_fails_against_its_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "sampling_events.jsonl.gz"
            with gzip.open(ledger, "wt") as stream:
                stream.write('{"event_id":[1,1,1]}\n')
            digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
            (root / "sampling_plan.json").write_text(json.dumps({
                "sampling_ledger": {"path": ledger.name, "sha256": digest}}))
            (root / "manifest.json").write_text(json.dumps({
                "schema": "shift-gen-to-nano-plan-v1", "eos_output": "/eos/" + CAMPAIGN,
                "complete_muon_decays": True, "correction": "retain natural branching ratios",
                "detector_sampling": {"ledger": ledger.name}}))
            with gzip.open(ledger, "wt") as stream:
                stream.write('{"event_id":[1,1,2]}\n')
            with patch("shift_mixing.full_gen_weight_sums", return_value={"qcd_0to1": 100.}):
                with self.assertRaisesRegex(RuntimeError, "ledger differs"):
                    build_inventory(CAMPAIGN, root)

    def test_pool_covers_the_existing_reco_only_classifier_contract(self):
        classifier = Path(__file__).resolve().parents[2] / "CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier"
        if not (classifier / "features.py").is_file():
            self.skipTest("Shared classifier checkout unavailable")
        sys.path.insert(0, str(classifier))
        from features import MUON_KINEMATICS, MUON_QUALITY
        self.assertTrue(set(MUON_KINEMATICS + MUON_QUALITY).issubset(REQUIRED_MUON_FIELDS))

    def test_partition_is_an_event_identity_function(self):
        identity = event_identity(CAMPAIGN, "qcd_0to1", [1, 2, 3])
        self.assertEqual(event_partition(identity), event_partition(identity))
        self.assertIn(event_partition(identity), ("train", "validation", "test"))
        self.assertNotEqual(identity, event_identity(CAMPAIGN, "qcd_1to2", [1, 2, 3]))
        with self.assertRaises(ValueError):
            event_identity(CAMPAIGN, "qcd_0to1", [1, 2, 3.5])

    def test_signed_small_native_weights_survive(self):
        values = dict(shiftOriginalGenWeight=-1e-20, shiftSamplingProbability=0.125,
                      shiftSamplingWeight=8., shiftSamplingGenWeight=-8e-20)
        ledger = dict(native_weight=-1e-20, sampling_probability=0.125,
                      inverse_sampling_probability=8., corrected_weight=-8e-20)
        validate_weights(values, ledger)
        values["shiftSamplingGenWeight"] = 0.
        with self.assertRaisesRegex(RuntimeError, "differs"):
            validate_weights(values, ledger)

    def test_reapplying_the_sampling_weight_fails_closed(self):
        values = dict(shiftOriginalGenWeight=1., shiftSamplingProbability=0.1,
                      shiftSamplingWeight=10., shiftSamplingGenWeight=100.)
        ledger = dict(native_weight=1., sampling_probability=0.1,
                      inverse_sampling_probability=10., corrected_weight=10.)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            validate_weights(values, ledger)

    def test_ancestry_retains_resonance_and_generator_indices(self):
        gen = dict(pdgId=[13, 13, 100443, 21], genPartIdxMother=[1, 2, 3, -1],
                   vx=[0., 0., 0., 0.], vy=[0., 0., 0., 0.], vz=[5500., 5500., 5500., 5500.])
        chain = truth_chain(0, gen)
        self.assertEqual([row["pdgId"] for row in chain], [13, 13, 100443, 21])
        self.assertEqual([row["index"] for row in chain], [0, 1, 2, 3])
        self.assertEqual(truth_chain(-1, gen), [])

    def test_cyclic_ancestry_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Cyclic"):
            truth_chain(0, dict(pdgId=[13, 443], genPartIdxMother=[1, 0]))

    def test_parent_census_distinguishes_same_pdg_different_particles(self):
        def muon(index, parent_index):
            chain = [{"pdgId": 13 if index == 0 else -13, "index": index},
                     {"pdgId": 333, "index": parent_index}]
            return {"truth": {"gen_match": chain, "hit_match": []}}
        pair = dict(muonIdx1=0, muonIdx2=1, topologyMin=2, topologyMax=2)
        distinct = dict(muons=[muon(0, 5), muon(1, 6)], stored_pairs=[pair])
        common = dict(muons=[muon(0, 5), muon(1, 5)], stored_pairs=[pair])
        counts = truth_census([distinct, common])["counts"]["gen_match"]["both_both_pairs"]
        self.assertEqual(counts, {"different_parents": 1, "common_parent_abs_pdg_333": 1})
        self.assertEqual(first_noncopy_parent([
            {"pdgId": 13, "index": 0}, {"pdgId": 13, "index": 1},
            {"pdgId": 100443, "index": 2}])["index"], 2)

    def test_reader_keeps_empty_events(self):
        event = dict(schema=SCHEMA, event_key="source:event", muons=[], stored_pairs=[],
                     nShiftMuon=0, detector_weight=10.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(json.dumps(event) + "\n")
            self.assertEqual(list(iter_pool(path)), [event])
            with gzip.open(path, "wt") as stream:
                stream.write('{"schema":"wrong"}\n')
            with self.assertRaisesRegex(RuntimeError, "schema"):
                list(iter_pool(path))


if __name__ == "__main__":
    unittest.main()
