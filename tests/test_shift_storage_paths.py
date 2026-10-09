import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
sys.path.insert(0, str(CONFIGS))
import shift_storage_paths as storage


class StorageNamingTest(unittest.TestCase):
    def test_process_bin_names_preserve_boundaries(self):
        self.assertEqual(storage.process_bin_directory("qcd", "0to1"), "qcd_pThat0to1GeV")
        self.assertEqual(storage.process_bin_directory("jpsi", "2to5"), "jpsi_pThat2to5GeV")
        self.assertEqual(storage.process_bin_directory("dy", "0.211317to0.5"), "dy_mHat0p211317to0p5GeV")
        self.assertEqual(storage.process_bin_directory("dy", "0p211317to0p5"), "dy_mHat0p211317to0p5GeV")

    def test_open_bins_have_one_name(self):
        self.assertEqual(storage.normalize_bin("20to-1"), "20toinf")
        self.assertEqual(storage.normalize_bin("20toinf"), "20toinf")
        self.assertEqual(storage.normalize_bin("020.000to100.00"), "20to100")

    def test_decimal_conversion_keeps_exact_digits(self):
        self.assertEqual(storage.normalize_bin("0.000000000000000001to0.000000000000000002"),
                         "0p000000000000000001to0p000000000000000002")

    def test_campaign_requires_a_valid_full_date(self):
        actual = storage.tier_directory("/eos/base", "qcd", "0to1", "detector_20261007_v10", "nanoAOD")
        self.assertEqual(str(actual), "/eos/base/qcd_pThat0to1GeV/detector_20261007_v10/nanoAOD")
        for name in ("detector_2023_v10", "detector_20260230_v10", "detector_20261007_v10/escape"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                storage.campaign_directory("/eos/base", "qcd", "0to1", name)

    def test_invalid_bins_are_refused(self):
        for value in ("1to1", "2to1", "-1to2", "1to2to3", "../0to1", "1e-3to1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage.normalize_bin(value)


class StorageResolverTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "map.json"
        self.old = "/eos/user/j/jniedzie/shift_cmssw/ntuple_production/campaign/qcd_0to1/job0000000/nano.root"
        self.new = "/eos/user/j/jniedzie/shift_cmssw/qcd_pThat0to1GeV/campaign_20261007/nanoAOD/nano_job0000000.root"
        self.record = dict(schema="shift-storage-path-map-v1", complete=True, paths={self.old: self.new},
                           support_prefixes={"/eos/user/j/jniedzie/shift_cmssw/production_dags":
                                             "/eos/user/j/jniedzie/shift_cmssw_support/production_dags"})
        self.write()

    def write(self):
        self.manifest.write_text(json.dumps(self.record))

    def resolve(self, path):
        return storage.resolve_storage_path(path, self.manifest)

    def test_old_receipt_resolves_without_modifying_it(self):
        receipt = self.root / "receipt.json"
        receipt.write_text(json.dumps(dict(path=self.old, sha256="a" * 64)))
        before = receipt.read_bytes()
        self.assertEqual(self.resolve(json.loads(before)["path"]), self.new)
        self.assertEqual(receipt.read_bytes(), before)

    def test_eos_mount_alias_and_remote_pfn(self):
        local_old = self.old.replace("/eos/user/j", "/eos/home-j")
        self.assertEqual(self.resolve(local_old), self.new.replace("/eos/user/j", "/eos/home-j"))
        self.assertEqual(self.resolve("root://eosuser.cern.ch/" + self.old), "root://eosuser.cern.ch/" + self.new)

    def test_support_prefix_is_component_bounded(self):
        path = "/eos/user/j/jniedzie/shift_cmssw/production_dags/gen/receipts/bin/part.json"
        self.assertEqual(self.resolve(path), path.replace("shift_cmssw/", "shift_cmssw_support/"))
        unrelated = path.replace("production_dags/", "production_dags_extra/")
        self.assertEqual(self.resolve(unrelated), unrelated)

    def test_more_specific_support_prefix_wins(self):
        self.record["support_prefixes"]["/eos/user/j/jniedzie/shift_cmssw/production_dags/special"] = "/eos/user/j/jniedzie/shift_cmssw_support/special"
        self.write()
        self.assertEqual(self.resolve("/eos/user/j/jniedzie/shift_cmssw/production_dags/special/a.json"),
                         "/eos/user/j/jniedzie/shift_cmssw_support/special/a.json")

    def test_incomplete_map_refused(self):
        self.record["complete"] = False
        self.write()
        with self.assertRaisesRegex(RuntimeError, "not validated and complete"):
            self.resolve(self.old)

    def test_archived_directory_mapping_is_exact(self):
        old = "/eos/user/j/jniedzie/shift_cmssw/jpsi/old/configs/step1"
        new = "/eos/user/j/jniedzie/shift_cmssw/jpsi_pThat1to5GeV/old_20260921/metadata/configs/step1"
        self.record["directories"] = {old: new}
        self.write()
        self.assertEqual(self.resolve(old), new)
        self.assertEqual(self.resolve(old + "/unlisted_cfg.py"), old + "/unlisted_cfg.py")

    def test_explicit_files_take_precedence_over_directory_entries(self):
        self.record["directories"] = {self.old: self.new + "_directory"}
        self.write()
        self.assertEqual(self.resolve(self.old), self.new)

    def test_logical_output_alias_resolves_mount_and_remote_paths(self):
        old = self.old.replace("/campaign/", "/campaign_bdt_v1/")
        new = self.new.replace("campaign_20261007", "campaign_bdt_v1_20261009")
        self.record["aliases"] = {old: new}
        self.write()
        self.assertEqual(self.resolve(old), new)
        self.assertEqual(self.resolve(old.replace("/eos/user/j", "/eos/home-j")),
                         new.replace("/eos/user/j", "/eos/home-j"))
        self.assertEqual(self.resolve("root://eosuser.cern.ch/" + old), "root://eosuser.cern.ch/" + new)
        self.assertEqual(self.record["paths"], {self.old: self.new})

    def test_physical_paths_take_precedence_over_logical_aliases(self):
        self.record["aliases"] = {self.old: self.new + "_alias"}
        self.write()
        self.assertEqual(self.resolve(self.old), self.new)

    def test_alias_mapping_types_are_refused(self):
        for aliases in ([], {self.old: 17}):
            self.record["aliases"] = aliases
            self.write()
            with self.subTest(aliases=aliases), self.assertRaisesRegex(RuntimeError, "Invalid storage"):
                self.resolve(self.old)

    def test_duplicate_alias_keys_are_refused(self):
        self.manifest.write_text('{"schema":"shift-storage-path-map-v1","complete":true,"paths":{},'
                                 '"aliases":{"/tmp/old.root":"/tmp/new.root","/tmp/old.root":"/tmp/other.root"}}')
        with self.assertRaisesRegex(ValueError, "Duplicate storage mapping"):
            self.resolve(self.old)

    def test_conflicting_alias_mount_spellings_are_refused(self):
        self.record["aliases"] = {self.old: self.new,
            self.old.replace("/eos/user/j", "/eos/home-j"): self.new + "_other"}
        self.write()
        with self.assertRaisesRegex(RuntimeError, "Conflicting EOS alias"):
            self.resolve(self.old)

    def test_missing_explicit_map_refused(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            storage.resolve_storage_path(self.old, self.root / "missing.json")

    def test_missing_default_is_compatible(self):
        with patch.object(storage, "DEFAULT_MIGRATION_MANIFEST", self.root / "missing.json"), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(storage.resolve_storage_path(self.old), self.old)

    def test_incomplete_default_blocks_shift_eos_but_keeps_unrelated_inputs_readable(self):
        self.record["complete"] = False
        self.write()
        with patch.object(storage, "DEFAULT_MIGRATION_MANIFEST", self.manifest), patch.dict("os.environ", {}, clear=True):
            for path in ("/tmp/source.root", "/afs/work/campaign/manifest.json",
                         "/eos/user/j/jniedzie/shift_cmssw_extra/source.root"):
                with self.subTest(path=path):
                    self.assertEqual(storage.resolve_storage_path(path), path)
            for path in (self.old, self.old.replace("/eos/user/j", "/eos/home-j"),
                         "/eos/user/j/jniedzie/shift_cmssw_support/runtime.tar.gz"):
                with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "not validated and complete"):
                    storage.resolve_storage_path(path)

    def test_explicit_local_map_still_checks_completion(self):
        self.record["complete"] = False
        self.record["paths"] = {"/tmp/old.root": "/tmp/new.root"}
        self.write()
        with self.assertRaisesRegex(RuntimeError, "not validated and complete"):
            self.resolve("/tmp/old.root")

    def test_default_scope_does_not_bypass_traversal_validation(self):
        with self.assertRaisesRegex(ValueError, "without traversal"):
            storage.resolve_storage_path("/tmp/../source.root")

    def test_env_override_uses_selected_map(self):
        with patch.dict("os.environ", {"SHIFT_STORAGE_MIGRATION_MANIFEST": str(self.manifest)}):
            self.assertEqual(storage.resolve_storage_path(self.old), self.new)

    def test_conflicting_alias_mappings_are_refused(self):
        self.record["paths"][self.old.replace("/eos/user/j", "/eos/home-j")] = self.new + "_other"
        self.write()
        with self.assertRaisesRegex(RuntimeError, "Conflicting EOS alias"):
            self.resolve(self.old)

    def test_duplicate_json_keys_are_refused(self):
        self.manifest.write_text('{"schema":"shift-storage-path-map-v1","complete":true,"complete":false,"paths":{}}')
        with self.assertRaisesRegex(ValueError, "Duplicate storage mapping"):
            self.resolve(self.old)

    def test_map_replacement_invalidates_cached_record(self):
        self.assertEqual(self.resolve(self.old), self.new)
        self.record["paths"][self.old] = self.new + "_replaced"
        self.write()
        self.assertEqual(self.resolve(self.old), self.new + "_replaced")

    def test_mapping_traversal_is_refused(self):
        self.record["paths"][self.old] = "/eos/user/j/jniedzie/../unexpected.root"
        self.write()
        with self.assertRaisesRegex(ValueError, "without traversal"):
            self.resolve(self.old)


if __name__ == "__main__":
    unittest.main()
