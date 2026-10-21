# Code style: AI-code. AI-generated data-preservation regression tests.
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import shift_nano_integrity as integrity


def receipt():
    return dict(nano_inputs=[dict(sha256="a" * 64, events=1)],
        merged_nano=dict(sha256="b" * 64),
        merged_nano_integrity=dict(schema="shift-nano-all-branch-equivalence-v1", complete=True,
            events=1, event_branches=1, input_sha256=["a" * 64], merged_sha256="b" * 64,
            all_branch_inventory_sha256="c" * 64, edm_metadata_baskets_preserved=True,
            edm_metadata_values_preserved=True,
            trees={name: dict(entries=1, branches=1) for name in integrity.REQUIRED_TREES}))


class RetirementProofTest(unittest.TestCase):
    def test_proof_binds_all_original_and_retained_bytes(self):
        good = receipt()
        self.assertEqual(integrity.validate_retirement_proof(good), good["merged_nano_integrity"])
        for edit in (lambda r: r["nano_inputs"][0].update(sha256="d" * 64),
                     lambda r: r["merged_nano"].update(sha256="d" * 64),
                     lambda r: r["merged_nano"].pop("sha256")):
            bad = copy.deepcopy(good)
            edit(bad)
            with self.assertRaisesRegex(ValueError, "different file bytes"):
                integrity.validate_retirement_proof(bad)

    def test_ids_only_incomplete_and_missing_metadata_proofs_fail(self):
        for key in ("schema", "complete", "edm_metadata_baskets_preserved", "edm_metadata_values_preserved"):
            bad = receipt()
            bad["merged_nano_integrity"].pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "complete all-branch"):
                integrity.validate_retirement_proof(bad)
        for tree in integrity.REQUIRED_TREES:
            bad = receipt()
            bad["merged_nano_integrity"]["trees"].pop(tree)
            with self.subTest(tree=tree), self.assertRaisesRegex(ValueError, "omits"):
                integrity.validate_retirement_proof(bad)

    def test_mismatched_event_count_and_partial_branch_inventory_fail(self):
        for field, value in (("events", 2), ("event_branches", 0),
                             ("all_branch_inventory_sha256", "")):
            bad = receipt()
            bad["merged_nano_integrity"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                integrity.validate_retirement_proof(bad)
        bad = receipt()
        bad["merged_nano_integrity"]["trees"]["Events"]["branches"] = 2
        with self.assertRaisesRegex(ValueError, "inconsistent Events"):
            integrity.validate_retirement_proof(bad)


@unittest.skipUnless(importlib.util.find_spec("uproot") and importlib.util.find_spec("ROOT"),
                     "Run the bounded ROOT fixtures inside the stock CMSSW runtime")
class NanoValuePreservationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def nano(self, name, event=1, change_tree=None, change_branch=None, drop_tree=None):
        import numpy as np
        import uproot
        path = self.base / name
        columns = dict(Events=dict(event=[event], pt=[3.5], nativeWeight=[1.25]),
            Runs=dict(run=[10], generated=[123]),
            LuminosityBlocks=dict(run=[10], luminosityBlock=[2]),
            MetaData=dict(history=[17]), ParameterSets=dict(parameter=[23]))
        if change_tree:
            columns[change_tree][change_branch][0] += 1
        if drop_tree:
            del columns[drop_tree]
        with uproot.recreate(path) as target:
            for tree, values in columns.items():
                arrays = {key: np.array(value, dtype="float64" if key in ("pt", "nativeWeight") else "uint64")
                          for key, value in values.items()}
                target.mktree(tree, {key: value.dtype for key, value in arrays.items()})
                target[tree].extend(arrays)
        return path

    def test_split_merge_preserves_every_tree_and_value(self):
        import ROOT
        first, second = self.nano("first.root", event=1), self.nano("second.root", event=2)
        merged = self.base / "merged.root"
        merger = ROOT.TFileMerger(False)
        merger.OutputFile(str(merged), "RECREATE")
        merger.AddFile(str(first))
        merger.AddFile(str(second))
        self.assertTrue(merger.Merge())
        proof = integrity.compare_all_branches([first, second], merged)
        self.assertEqual(proof["events"], 2)
        self.assertEqual(proof["trees"]["Runs"]["entries"], 2)
        self.assertTrue(proof["edm_metadata_baskets_preserved"])

    def test_non_weight_event_value_changes_are_rejected(self):
        original = self.nano("original.root")
        changed = self.nano("changed.root", change_tree="Events", change_branch="pt")
        with self.assertRaisesRegex(ValueError, "changed branch values"):
            integrity.compare_all_branches([original], changed)

    def test_run_lumi_and_persisted_metadata_changes_are_rejected(self):
        original = self.nano("original.root")
        for tree, branch in (("Runs", "generated"), ("LuminosityBlocks", "luminosityBlock"),
                             ("MetaData", "history"), ("ParameterSets", "parameter")):
            changed = self.nano(tree + ".root", change_tree=tree, change_branch=branch)
            with self.subTest(tree=tree), self.assertRaisesRegex(ValueError, "changed branch values"):
                integrity.compare_all_branches([original], changed)

    def test_omitted_metadata_tree_is_rejected(self):
        original = self.nano("original.root")
        changed = self.nano("changed.root", drop_tree="ParameterSets")
        with self.assertRaisesRegex(ValueError, "Missing retained Nano tree"):
            integrity.compare_all_branches([original], changed)


@unittest.skipUnless(importlib.util.find_spec("uproot"), "Requires uproot and awkward")
class ReplayValuePreservationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def nano(self, name, edit=None, reverse_tree=None, pt_type="float32",
             normalization='{"full_source_trials":2000}', drop_tree=None, empty_tree=None):
        import numpy as np
        import uproot
        # Large IDs and tiny Double weights must survive without a float cast
        # or a tolerance that hides physics changes.
        columns = dict(
            Events=dict(event=np.array([2**63 + 1, 2**63 + 2], dtype="uint64"),
                        pt=np.array([3.5, 7.25], dtype=pt_type),
                        nativeWeight=np.array([4.953869585087369e-13, 1.4637661928892926e-15])),
            Runs=dict(run=np.array([10, 11], dtype="uint32"), generated=np.array([2000, 3000])),
            LuminosityBlocks=dict(run=np.array([10, 11], dtype="uint32"),
                                 luminosityBlock=np.array([2, 3], dtype="uint32")),
            MetaData=dict(history=np.array([17])), ParameterSets=dict(parameter=np.array([23])))
        if edit:
            tree, branch, row, value = edit
            columns[tree][branch][row] = value
        if reverse_tree:
            columns[reverse_tree] = {key: values[::-1] for key, values in columns[reverse_tree].items()}
        if drop_tree:
            del columns[drop_tree]
        if empty_tree:
            columns[empty_tree] = {key: values[:0] for key, values in columns[empty_tree].items()}
        path = self.base / name
        with uproot.recreate(path) as target:
            for tree, arrays in columns.items():
                target.mktree(tree, {key: values.dtype for key, values in arrays.items()})
                target[tree].extend(arrays)
            if normalization is not None:
                target["shiftSamplingNormalization"] = normalization
        return path

    def test_replay_allows_labeled_edm_provenance_changes_but_not_retirement(self):
        baseline = self.nano("baseline.root")
        replay = self.nano("replay.root", edit=("MetaData", "history", 0, 99))
        proof = integrity.compare_replay_values(baseline, replay)
        self.assertTrue(proof["complete"])
        self.assertEqual(proof["events"], 2)
        self.assertEqual(proof["baseline_sha256"], integrity.sha256(baseline))
        self.assertEqual(proof["replay_sha256"], integrity.sha256(replay))
        self.assertEqual(proof["string_metadata_objects"], ["shiftSamplingNormalization"])
        self.assertFalse(proof["edm_metadata"]["MetaData"]["basket_inventory_equal"])
        self.assertTrue(proof["edm_metadata"]["ParameterSets"]["basket_inventory_equal"])
        self.assertFalse(proof["edm_metadata_byte_equivalence_claimed"])
        self.assertFalse(proof["retirement_authorized"])
        with self.assertRaisesRegex(ValueError, "changed branch values"):
            integrity.compare_all_branches([baseline], replay)
        merged_receipt = receipt()
        merged_receipt["merged_nano_integrity"] = proof
        with self.assertRaisesRegex(ValueError, "complete all-branch"):
            integrity.validate_retirement_proof(merged_receipt)

    def test_physics_tiny_weights_run_and_lumi_changes_fail(self):
        baseline = self.nano("baseline.root")
        for edit in (("Events", "pt", 1, 8.), ("Events", "nativeWeight", 0, 0.),
                     ("Events", "nativeWeight", 1, 1.4637661928892928e-15),
                     ("Runs", "generated", 0, 2001), ("LuminosityBlocks", "luminosityBlock", 1, 4)):
            changed = self.nano("changed.root", edit=edit)
            with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, "changed ordered typed"):
                integrity.compare_replay_values(baseline, changed)

    def test_order_and_typed_schema_are_exact(self):
        baseline = self.nano("baseline.root")
        for arguments in (dict(reverse_tree="Events"), dict(reverse_tree="Runs"),
                          dict(reverse_tree="LuminosityBlocks"), dict(pt_type="float64")):
            replay = self.nano("replay.root", **arguments)
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "changed ordered typed"):
                integrity.compare_replay_values(baseline, replay)

    def test_changed_or_missing_normalization_fails(self):
        baseline = self.nano("baseline.root")
        for normalization in (None, '{"full_source_trials":2}'):
            replay = self.nano("replay.root", normalization=normalization)
            with self.subTest(normalization=normalization), self.assertRaisesRegex(ValueError, "string metadata"):
                integrity.compare_replay_values(baseline, replay)

    def test_missing_edm_metadata_fails(self):
        baseline = self.nano("baseline.root")
        for name in integrity.EDM_METADATA_TREES:
            replay = self.nano("replay.root", drop_tree=name)
            with self.subTest(tree=name), self.assertRaisesRegex(ValueError, "Missing retained Nano tree"):
                integrity.compare_replay_values(baseline, replay)

    def test_parameter_set_delta_is_provenance_but_empty_metadata_fails(self):
        baseline = self.nano("baseline.root")
        replay = self.nano("replay.root", edit=("ParameterSets", "parameter", 0, 101))
        proof = integrity.compare_replay_values(baseline, replay)
        self.assertFalse(proof["edm_metadata"]["ParameterSets"]["basket_inventory_equal"])
        for name in integrity.EDM_METADATA_TREES:
            replay = self.nano("empty.root", empty_tree=name)
            with self.subTest(tree=name), self.assertRaisesRegex(ValueError, "empty EDM metadata"):
                integrity.compare_replay_values(baseline, replay)


if __name__ == "__main__":
    unittest.main()
