# Code style: AI-code. AI-generated regression test.
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
        retention_default = patch.object(storage, "DEFAULT_RETENTION_MANIFEST",
                                         self.root / "absent_retention.json")
        retention_default.start()
        self.addCleanup(retention_default.stop)
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


class StorageRetentionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.migration = self.root / "migration.json"
        self.retention = self.root / "retention.json"
        self.base = "/eos/user/j/jniedzie/shift_cmssw/"
        self.campaign = self.base + "jpsi_pThat1to5GeV/old_20260921"
        self.archive = ("/eos/user/j/jniedzie/shift_cmssw_support/legacy_references/"
                        "jpsi_pThat1to5GeV/old_20260921")
        self.old = self.base + "jpsi/old/samples/step4/file.root"
        self.alias = self.old.replace("/old/", "/logical_old/")
        self.dated = self.campaign + "/nanoAOD/file.root"
        self.migration_record = dict(schema="shift-storage-path-map-v1", complete=True,
                                     paths={self.old: self.dated}, aliases={self.alias: self.dated})
        self.record = dict(schema="shift-storage-retention-map-v1", complete=True,
                           archive_prefixes={self.campaign: self.archive}, deleted_paths=[])
        self.migration.write_text(json.dumps(self.migration_record))
        self.write()
        for setting, value in (("DEFAULT_MIGRATION_MANIFEST", self.migration),
                               ("DEFAULT_RETENTION_MANIFEST", self.retention)):
            selected = patch.object(storage, setting, value)
            selected.start()
            self.addCleanup(selected.stop)
        environment = patch.dict("os.environ", {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write(self):
        self.retention.write_text(json.dumps(self.record))

    def test_legacy_alias_and_dated_paths_resolve_without_rewriting_migration(self):
        before = self.migration.read_bytes()
        expected = self.archive + "/nanoAOD/file.root"
        for path in (self.old, self.alias, self.dated):
            with self.subTest(path=path):
                self.assertEqual(storage.resolve_storage_path(path), expected)
        self.assertEqual(self.migration.read_bytes(), before)

    def test_archive_retains_local_mount_and_remote_pfn_spelling(self):
        expected = self.archive + "/nanoAOD/file.root"
        for path in (self.old, self.alias, self.dated):
            with self.subTest(path=path):
                self.assertEqual(storage.resolve_storage_path(path.replace("/eos/user/j", "/eos/home-j")),
                                 expected.replace("/eos/user/j", "/eos/home-j"))
                self.assertEqual(storage.resolve_storage_path("root://eosuser.cern.ch/" + path),
                                 "root://eosuser.cern.ch/" + expected)

    def test_archived_config_directory_resolves_after_exact_migration_mapping(self):
        old = self.base + "jpsi/old/configs/step1"
        self.migration_record["directories"] = {old: self.campaign + "/metadata/legacy/configs/step1"}
        self.migration.write_text(json.dumps(self.migration_record))
        self.assertEqual(storage.resolve_storage_path(old), self.archive + "/metadata/legacy/configs/step1")

    def test_longest_archive_prefix_wins_with_component_boundaries(self):
        self.record["archive_prefixes"][self.campaign + "/nanoAOD"] = self.archive + "/special"
        self.write()
        self.assertEqual(storage.resolve_storage_path(self.dated), self.archive + "/special/file.root")
        adjacent = self.campaign + "_extra/nanoAOD/file.root"
        self.assertEqual(storage.resolve_storage_path(adjacent), adjacent)
        self.assertEqual(storage.resolve_storage_path(self.campaign + "/nanoAOD_extra/file.root"),
                         self.archive + "/nanoAOD_extra/file.root")

    def test_tombstone_blocks_legacy_alias_dated_mount_and_pfn_before_archive(self):
        self.record["deleted_paths"] = [self.dated]
        self.write()
        for path in (self.old, self.alias, self.dated,
                     self.dated.replace("/eos/user/j", "/eos/home-j"),
                     "root://eosuser.cern.ch/" + self.old):
            with self.subTest(path=path), self.assertRaisesRegex(FileNotFoundError, "data was retired"):
                storage.resolve_storage_path(path)
        self.assertEqual(storage.resolve_storage_path(self.campaign + "/nanoAOD/another.root"),
                         self.archive + "/nanoAOD/another.root")

    def test_tombstone_also_checks_original_canonical_input(self):
        self.record["deleted_paths"] = [self.old]
        self.write()
        with self.assertRaisesRegex(FileNotFoundError, "data was retired"):
            storage.resolve_storage_path(self.old)

    def test_gen_pilot_tombstone_does_not_retire_campaign_metadata(self):
        pilot = self.base + "qcd_pThat1to2GeV/pilot_20260921/GEN_pilot/gen.root"
        metadata = pilot.replace("GEN_pilot/gen.root", "metadata/normalization_complete.json")
        self.record["deleted_paths"] = [pilot]
        self.write()
        with self.assertRaises(FileNotFoundError):
            storage.resolve_storage_path(pilot)
        self.assertEqual(storage.resolve_storage_path(metadata), metadata)

    def test_current_outputs_and_archive_paths_are_unchanged(self):
        for path in (self.base + "jpsi_pThat1to2GeV/detector_20261007_v10/nanoAOD/current.root",
                     self.archive + "/nanoAOD/file.root"):
            with self.subTest(path=path):
                self.assertEqual(storage.resolve_storage_path(path), path)

    def test_incomplete_overlay_blocks_only_shift_inputs(self):
        self.record["complete"] = False
        self.write()
        for path in (self.old, self.dated,
                     self.base + "jpsi_pThat1to2GeV/current_20261007/nanoAOD/file.root",
                     "/eos/home-j/jniedzie/shift_cmssw_support/runtime.tar.gz"):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "retention is not validated"):
                storage.resolve_storage_path(path)
        for path in ("/tmp/source.root", "/afs/work/manifest.json",
                     self.base.replace("shift_cmssw/", "shift_cmssw_extra/") + "file.root"):
            with self.subTest(path=path):
                self.assertEqual(storage.resolve_storage_path(path), path)
                self.assertEqual(storage.resolve_storage_path(path, self.migration, self.retention), path)

    def test_custom_migration_maps_opt_out_of_default_retention(self):
        self.record["complete"] = False
        self.write()
        self.assertEqual(storage.resolve_storage_path(self.old, self.migration), self.dated)
        with patch.dict("os.environ", {"SHIFT_STORAGE_MIGRATION_MANIFEST": str(self.migration)}):
            self.assertEqual(storage.resolve_storage_path(self.old), self.dated)

    def test_custom_migration_can_explicitly_opt_into_retention(self):
        self.assertEqual(storage.resolve_storage_path(self.old, self.migration, self.retention),
                         self.archive + "/nanoAOD/file.root")
        self.record["deleted_paths"] = [self.dated]
        self.write()
        with self.assertRaises(FileNotFoundError):
            storage.resolve_storage_path(self.old, self.migration, self.retention)

    def test_absent_default_overlay_preserves_migration_behavior(self):
        self.retention.unlink()
        self.assertEqual(storage.resolve_storage_path(self.old), self.dated)

    def test_dated_paths_can_be_archived_without_a_migration_map(self):
        self.migration.unlink()
        self.assertEqual(storage.resolve_storage_path(self.dated), self.archive + "/nanoAOD/file.root")

    def test_missing_explicit_retention_map_is_refused_for_shift_inputs(self):
        with self.assertRaisesRegex(RuntimeError, "retention map is missing"):
            storage.resolve_storage_path(self.old, self.migration, self.root / "missing.json")

    def test_retention_schema_and_mapping_types_are_refused(self):
        for change in ({"schema": "unknown"}, {"archive_prefixes": []}, {"deleted_paths": {}},
                       {"archive_prefixes": {self.campaign: 17}}, {"deleted_paths": [17]}):
            record = dict(self.record, **change)
            self.retention.write_text(json.dumps(record))
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                storage.resolve_storage_path(self.old)

    def test_retention_rejects_traversal_relative_and_out_of_scope_paths(self):
        for prefixes, deleted in (({"relative": self.archive}, []),
                                  ({self.campaign: self.archive + "/../escape"}, []),
                                  ({self.campaign: "/tmp/archive"}, []),
                                  ({"/tmp/source": self.archive}, []),
                                  ({}, ["/tmp/deleted.root"]),
                                  ({}, [self.base + "../deleted.root"])):
            self.record.update(archive_prefixes=prefixes, deleted_paths=deleted)
            self.write()
            with self.subTest(prefixes=prefixes, deleted=deleted), self.assertRaises(ValueError):
                storage.resolve_storage_path(self.old)

    def test_duplicate_json_retention_keys_are_refused(self):
        self.retention.write_text('{"schema":"shift-storage-retention-map-v1",'
                                  '"complete":true,"complete":false,"archive_prefixes":{},"deleted_paths":[]}')
        with self.assertRaisesRegex(ValueError, "Duplicate storage mapping key"):
            storage.resolve_storage_path(self.old)

    def test_duplicate_canonical_prefixes_and_tombstones_are_refused(self):
        local = self.campaign.replace("/eos/user/j", "/eos/home-j")
        self.record["archive_prefixes"][local] = self.archive
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate canonical storage retention prefix"):
            storage.resolve_storage_path(self.old)
        self.record.update(archive_prefixes={}, deleted_paths=[self.dated, self.dated.replace("/eos/user/j", "/eos/home-j")])
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate retired storage path"):
            storage.resolve_storage_path(self.old)

    def test_retention_replacement_invalidates_cached_record(self):
        self.assertEqual(storage.resolve_storage_path(self.old), self.archive + "/nanoAOD/file.root")
        self.record["archive_prefixes"][self.campaign] = self.archive + "/updated"
        self.write()
        self.assertEqual(storage.resolve_storage_path(self.old), self.archive + "/updated/nanoAOD/file.root")


if __name__ == "__main__":
    unittest.main()
