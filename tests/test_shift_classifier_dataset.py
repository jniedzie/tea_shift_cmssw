# Code style: AI-code. AI-generated ownership and EOS-routing regression checks.
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "utils" / "shift_classifier_dataset.py"
SPEC = importlib.util.spec_from_file_location("shift_classifier_dataset", MODULE)
DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET)
sys.path.insert(0, str(MODULE.parent))


class OwnershipTests(unittest.TestCase):
    def test_reweights_and_transitive_shared_sources_cannot_cross_folds(self):
        rows = [{"source_group_keys": keys} for keys in
                (["gen-a"], ["gen-a", "bank-x"], ["bank-x", "gen-b"], ["gen-c"])]
        groups = DATASET.ownership_components(rows)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])
        reverse = DATASET.ownership_components(rows[::-1])
        self.assertEqual(groups, reverse[::-1])

    def test_missing_ownership_is_not_silently_file_independence(self):
        with self.assertRaisesRegex(ValueError, "Missing original source ownership"):
            DATASET.ownership_components([{"path": "replay.root"}])

    def test_cms_archive_and_user_alias_use_the_correct_instances(self):
        self.assertEqual(DATASET.xrootd_url("/eos/cms/store/test.root"),
                         "root://eoscms.cern.ch//eos/cms/store/test.root")
        self.assertEqual(DATASET.xrootd_url("/eos/home-j/jniedzie/test.root"),
                         "root://eosuser.cern.ch//eos/user/j/jniedzie/test.root")
        with self.assertRaises(ValueError):
            DATASET.xrootd_url("/tmp/test.root")


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = np.array(["feature" + str(i) for i in range(121)])
        self.sm = self.sample(160, "qcd", "SM", "recent")
        self.sm["y"] = np.array([0] * 80 + [1] * 80, dtype=np.int8)
        self.sm["original_y"] = self.sm["y"].copy()
        self.sm["group_id"] = np.array(["sm-group-" + str(i) for i in range(160)])
        self.sm["event_owner_id"] = np.array(["sm-event-" + str(i) for i in range(160)])
        self.sm["cache_row"] = np.arange(160)
        self.signal = self.sample(4, "signal", "m15_prompt", "signal_control")
        self.signal["ids"][:, 0] = 200
        self.old = {k: np.concatenate((self.sm[k], self.signal[k])) for k in self.signal if k != "feature_names"}
        self.old["feature_names"] = self.names
        self.old["direct443"] = np.zeros(164, dtype=bool)
        self.old["direct443"][99] = True
        self.new = self.sample(2, "signal", "m20_prompt", "current_BSM")
        self.new.update(y=np.array([-1, 1], dtype=np.int8), original_y=np.array([-1, 1], dtype=np.int8),
                        label_recovered=np.array([False, False]), label_status=np.array(["unknown", "known_bad_fit"]),
                        truth_leg_gen_state=np.array([["absent", "absent"], ["muon", "muon"]]))
        self.new["nano_path"] = np.full(2, "positive.root")
        self.new["ids"][:, 0] = 300
        old_ref = self.save("old.npz", self.old)
        sm_ref = self.save("sm.npz", self.sm)
        inventory = dict(simulation_only=True, do_not_read_collision_data=True, rows=[
            dict(path="positive.root", nano_path="positive.root", point="m20_prompt", sha256="nano-a", expected_events=2,
                 source_group_keys=["GEN_SHA256:new"], source_records=[dict(run_number=300, gen_sha256="new")]),
            dict(path="zero.root", nano_path="zero.root", point="m30_prompt", sha256="nano-b", expected_events=3,
                 source_group_keys=["GEN_SHA256:zero"], source_records=[])])
        inventory_ref = self.save("inventory.json", inventory)
        expanded = dict(full_current_data_sha256=old_ref["sha256"], source_proofs=[dict(cache_source_index=0, nano_path="oldsignal.root", nano_sha256="old-nano")], rows=[
            dict(full_current_cache_row=160+i, expanded_cache_source_index=0, identity=self.signal["ids"][i].tolist(), pair_index=0,
                 event_owner_id="old-event-"+str(i), source_group_keys=["old-gen-"+str(i)], original_source_ownership_qualified=True) for i in range(4)])
        expanded_ref = self.save("expanded.json", expanded)
        ownership = dict(source_manifest_sha256={"current_bsm.json": inventory_ref["sha256"], "expanded_signal_source_proof.json": expanded_ref["sha256"]},
            path_point_groups=[dict(path=r["path"], point=r["point"], group_id="new-group-"+str(i)) for i, r in enumerate(inventory["rows"])],
            expanded_cache_groups=[dict(cache_row=160+i, group_id="old-group-"+str(i)) for i in range(4)])
        ownership_ref = self.save("ownership.json", ownership)
        preparation = dict(dataset=sm_ref, global_groups_sha256=ownership_ref["sha256"], omitted_cache_rows=[999], current_nano_source_set_rows=[])
        new_ref = self.save("new.npz", self.new)
        receipt = dict(complete=True, failures=[], start=0, stop=2, candidates=2, data_sha256=new_ref["sha256"], inventory_sha256=inventory_ref["sha256"], files=[
            dict(inventory_index=0, path="positive.root", input_sha256="nano-a", events=2, pairs=2),
            dict(inventory_index=1, path="zero.root", input_sha256="nano-b", events=3, pairs=0)])
        self.manifest = dict(simulation_only=True, do_not_read_collision_data=True, cached_sm=sm_ref,
            sm_preparation=self.save("preparation.json", preparation), old_cache=old_ref, expanded_signal_proof=expanded_ref,
            global_ownership=ownership_ref, inventory=inventory_ref,
            shards=[dict(receipt=self.save("receipt.json", receipt), data=new_ref, readback_verified=True)])

    def sample(self, n, process, point, domain):
        return dict(X=np.zeros((n, 121)), feature_names=self.names, y=np.ones(n, dtype=np.int8), weight=np.arange(n, dtype=float)+1,
            mass=np.full(n, 15.), process=np.full(n, process), point=np.full(n, point), category=np.full(n, "Both-Both"), domain=np.full(n, domain),
            ids=np.column_stack((np.full(n, 100, dtype=np.uint64), np.ones(n, dtype=np.uint64), np.arange(1, n+1, dtype=np.uint64))),
            pair_index=np.zeros(n, dtype=np.int64), label_reason=np.full(n, "frozen"), nano_path=np.full(n, process+".root"))

    def save(self, name, value):
        path = self.root / name
        if name.endswith(".npz"):
            np.savez_compressed(path, **value)
        else:
            path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=DATASET.sha256(path))

    def assemble(self):
        manifest = self.save("manifest.json", self.manifest)
        return DATASET.assemble_cohort(manifest["path"], self.root / "assembled")

    def test_complete_coverage_preserves_unknown_and_genuine_bad_fit(self):
        report = self.assemble()
        data = dict(np.load(self.root / "assembled/data.npz", allow_pickle=False))
        self.assertEqual(report["inventory_files"], 2)
        self.assertEqual(report["candidates"], 166)
        self.assertEqual(data["y"][-2:].tolist(), [-1, 1])
        self.assertEqual(data["label_status"][-1], "known_bad_fit")
        np.testing.assert_array_equal(data["weight"][:160], self.sm["weight"])
        self.assertEqual(report["original_source_quarantine"], [999])
        self.assertEqual(report["ownership_precision"]["original_gen_event_namespace"], 2)
        self.assertTrue(data["direct443"][99])
        self.assertFalse(data["direct443"][-1])
        self.assertEqual(report["inventory_physical_points"], ["m20_prompt", "m30_prompt"])
        self.assertEqual(report["current_BSM_point_census"]["m30_prompt"]["observed_raw_pairs"], 0)
        self.assertIn("unknown", report["current_BSM_point_census"]["m30_prompt"]["observation_status"])
        self.assertEqual(len(report["fold_readiness"]), 4)

    def test_duplicate_shard_is_not_additional_exposure(self):
        self.manifest["shards"].append(self.manifest["shards"][0])
        with self.assertRaisesRegex(ValueError, "coverage overlaps"):
            self.assemble()
        self.assertFalse((self.root / "assembled").exists())

    def test_unverified_readback_cannot_enter_cohort(self):
        self.manifest["shards"][0]["readback_verified"] = False
        with self.assertRaisesRegex(ValueError, "unverified shard"):
            self.assemble()

    def test_changed_local_payload_fails_before_assembly(self):
        self.new["X"][0, 0] = 42
        np.savez_compressed(self.root / "new.npz", **self.new)
        with self.assertRaisesRegex(ValueError, "digest differs"):
            self.assemble()

    def test_separate_empty_shard_still_closes_its_input_coverage(self):
        receipt = json.loads((self.root / "receipt.json").read_text())
        zero = dict(receipt, start=1, stop=2, candidates=0, files=receipt["files"][1:])
        zero.pop("data_sha256")
        positive = dict(receipt, stop=1, files=receipt["files"][:1])
        self.manifest["shards"] = [
            dict(receipt=self.save("positive.json", positive), data=self.manifest["shards"][0]["data"], readback_verified=True),
            dict(receipt=self.save("zero.json", zero), data=None, readback_verified=True)]
        report = self.assemble()
        self.assertEqual(report["inventory_files"], 2)
        self.assertEqual(report["shards"][1]["candidates"], 0)

    def test_recovery_cannot_reverse_a_known_genuine_label(self):
        self.new["y"][1] = 0
        new_ref = self.save("new.npz", self.new)
        receipt = json.loads((self.root / "receipt.json").read_text())
        receipt["data_sha256"] = new_ref["sha256"]
        self.manifest["shards"][0].update(data=new_ref, receipt=self.save("receipt.json", receipt))
        with self.assertRaisesRegex(ValueError, "changed a known truth label"):
            self.assemble()

    def test_quarantined_source_row_is_rejected_even_with_matching_hashes(self):
        preparation = json.loads((self.root / "preparation.json").read_text())
        preparation["omitted_cache_rows"] = [0]
        self.manifest["sm_preparation"] = self.save("preparation.json", preparation)
        with self.assertRaisesRegex(ValueError, "Quarantined original-source"):
            self.assemble()


if __name__ == "__main__":
    unittest.main()
