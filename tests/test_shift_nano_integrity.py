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


if __name__ == "__main__":
    unittest.main()
