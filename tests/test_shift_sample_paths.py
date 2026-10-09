import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
import hashlib
from unittest.mock import patch

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
sys.path.insert(0, str(CONFIGS))
sys.path.insert(0, str(CONFIGS.parent / 'tea/pylibs/logger'))
from shift_sample_paths import latest_merged_sample, validated_nano_files


class MergedSampleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        migration_default = patch('shift_storage_paths.DEFAULT_MIGRATION_MANIFEST',
                                  self.directory / 'missing_storage_map.json')
        migration_default.start()
        self.addCleanup(migration_default.stop)
        retention_default = patch('shift_storage_paths.DEFAULT_RETENTION_MANIFEST',
                                  self.directory / 'missing_retention_map.json')
        retention_default.start()
        self.addCleanup(retention_default.stop)

    def complete(self):
        path = self.directory / 'ntuple_sampling_complete_200events.root'
        path.touch()
        path.with_suffix('.json').write_text(json.dumps(dict(
            status='validated', output=str(path), sha256='a' * 64)))
        return path

    def test_complete_merge_hash(self):
        path = self.complete()
        self.assertEqual(latest_merged_sample(self.directory), (str(path), 'a' * 12))

    def test_legacy_name(self):
        path = self.directory / 'ntuple_0_1234567-dirty-abcdef12.root'
        path.touch()
        self.assertEqual(latest_merged_sample(self.directory), (str(path), '1234567-dirty-abcdef12'))

    def test_complete_supersedes_newer_legacy(self):
        complete = self.complete()
        (self.directory / 'ntuple_0_1234567.root').touch()
        self.assertEqual(latest_merged_sample(self.directory)[0], str(complete))

    def test_missing_record_rejected(self):
        (self.directory / 'ntuple_sampling_complete_200events.root').touch()
        with self.assertRaisesRegex(RuntimeError, 'missing its validation'):
            latest_merged_sample(self.directory)

    def test_mismatched_record_rejected(self):
        path = self.complete()
        record = path.with_suffix('.json')
        value = json.loads(record.read_text())
        value['output'] = '/another/file.root'
        record.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'Invalid merged-sample'):
            latest_merged_sample(self.directory)

    def test_relocated_merge_keeps_its_original_sidecar(self):
        original = self.complete()
        relocated = self.directory / 'nanoAOD_merged'
        relocated.mkdir()
        target = relocated / original.name
        sidecar = original.with_suffix('.json')
        original_record = sidecar.read_bytes()
        original.rename(target)
        sidecar.rename(target.with_suffix('.json'))
        migration = self.directory / 'map.json'
        migration.write_text(json.dumps(dict(schema='shift-storage-path-map-v1',
            complete=True, paths={str(original): str(target)})))
        with patch.dict('os.environ', {'SHIFT_STORAGE_MIGRATION_MANIFEST': str(migration)}):
            self.assertEqual(latest_merged_sample(relocated), (str(target), 'a' * 12))
        self.assertEqual(target.with_suffix('.json').read_bytes(), original_record)

    def test_unknown_files_rejected(self):
        (self.directory / 'ntuple_unknown.root').touch()
        with self.assertRaisesRegex(RuntimeError, 'No versioned or validated'):
            latest_merged_sample(self.directory)

    def test_full_config_books_refit_diagnostics(self):
        sys.path.insert(0, str(CONFIGS.parent / 'tea/pylibs/logger'))
        self.addCleanup(sys.path.remove, str(CONFIGS.parent / 'tea/pylibs/logger'))
        config = runpy.run_path(str(CONFIGS / 'shift_histogrammer_config.py'),
                                init_globals={'recoDataMode': True})
        names = {f'{entry[0]}_{entry[1]}' for entry in config['histParams']}
        self.assertIn('VertexRefitDiagnostics_massErr', names)
        self.assertIn('VertexRefitDiagnostics_massRelativeErr', names)


class CompleteNanoInventoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        migration_default = patch('shift_storage_paths.DEFAULT_MIGRATION_MANIFEST',
                                  self.root / 'missing_storage_map.json')
        migration_default.start()
        self.addCleanup(migration_default.stop)
        retention_default = patch('shift_storage_paths.DEFAULT_RETENTION_MANIFEST',
                                  self.root / 'missing_retention_map.json')
        retention_default.start()
        self.addCleanup(retention_default.stop)
        self.campaign = 'shift_detector_representative_20261005_v8'
        self.base = '/eos/home-j/jniedzie/shift_cmssw'
        self.eos = '/eos/user/j/jniedzie/shift_cmssw/ntuple_production/' + self.campaign
        self.tiers = ('GEN', 'SIM', 'DIGIHLT', 'RECO', 'NANO')
        self.manifest = dict(jobs=2, events=2, strata={'qcd_0to1': 1, 'jpsi_20toinf': 1}, eos_output=self.eos)
        self.write(self.root / 'sampling_plan.json', dict(parent_exposure='full GEN'))
        self.manifest['detector_sampling'] = dict(plan_sha256=hashlib.sha256(
            (self.root / 'sampling_plan.json').read_bytes()).hexdigest())
        self.manifest['sources'] = [dict(index=i, stratum=name, events=1)
                                    for i, name in enumerate(self.manifest['strata'])]
        self.write(self.root / 'manifest.json', self.manifest)
        (self.root / 'all_jobs.txt').write_text('00000 0 1 0\n00001 0 1 1\n')
        (self.root / 'results').mkdir()
        self.final = dict(complete=True, jobs=2, events=2,
                          tier_events={name: {tier: 1 for tier in self.tiers} for name in self.manifest['strata']})
        self.write(self.root / 'production_complete.json', self.final)
        self.rows = [dict(job=job, source_stratum=name, events=1, complete=True,
                          nano_path=f'{self.eos}/{name}/job{job:07d}/nano.root',
                          validated_tier_events={tier: 1 for tier in self.tiers},
                          exit_code=0, nano_bytes=128, report_sha256='a' * 64)
                     for job, name in enumerate(self.manifest['strata'])]
        self.refresh()

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def refresh(self):
        for job, row in enumerate(self.rows):
            self.write(self.root / 'results' / f'status{job}.json', row)

    def files(self):
        return validated_nano_files(self.base, self.campaign, self.root)

    def test_all_bins_get_unique_conventional_outputs(self):
        files = self.files()
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0][0].endswith('qcd_0to1/job0000000/nano.root'))
        self.assertEqual(files[1][2], self.base + '/jpsi_pThat20toinfGeV/' + self.campaign + '/histograms/histograms_job0000001.root')

    def test_relocated_inputs_preserve_original_receipt_validation(self):
        original_rows = [(self.root / 'results' / f'status{job}.json').read_bytes() for job in range(2)]
        original_plan = (self.root / 'manifest.json').read_bytes()
        original_sampling = (self.root / 'sampling_plan.json').read_bytes()
        migration = self.root / 'map.json'
        destinations = [
            self.base.replace('/eos/home-j', '/eos/user/j') + '/qcd_pThat0to1GeV/' + self.campaign + '/nanoAOD/nano_job0000000.root',
            self.base.replace('/eos/home-j', '/eos/user/j') + '/jpsi_pThat20toinfGeV/' + self.campaign + '/nanoAOD/nano_job0000001.root',
        ]
        migration.write_text(json.dumps(dict(schema='shift-storage-path-map-v1', complete=True,
            paths={row['nano_path']: destination for row, destination in zip(self.rows, destinations)})))
        with patch.dict('os.environ', {'SHIFT_STORAGE_MIGRATION_MANIFEST': str(migration)}):
            files = self.files()
            self.assertEqual([row[0] for row in files],
                [path.replace('/eos/user/j', '/eos/home-j') for path in destinations])
            self.assertEqual([(self.root / 'results' / f'status{job}.json').read_bytes()
                for job in range(2)], original_rows)
            # A rewritten receipt must still fail even when it names the new file.
            self.rows[0]['nano_path'] = destinations[0]
            self.refresh()
            with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
                self.files()
        self.assertEqual((self.root / 'manifest.json').read_bytes(), original_plan)
        self.assertEqual((self.root / 'sampling_plan.json').read_bytes(), original_sampling)

    def test_terminal_failed_production_reports_failure_instead_of_waiting(self):
        (self.root / 'production_complete.json').unlink()
        self.write(self.root / 'live_status.json', dict(health='failed', terminal=True,
                   nano_jobs_done=1, nano_jobs_expected=2, failed_nano_jobs=[1]))
        with self.assertRaisesRegex(RuntimeError, '1/2 Nano jobs completed; 1 failed.*--allow-incomplete'):
            self.files()

    def test_sharded_receipts_keep_complete_inventory_validation(self):
        group = self.root / 'results' / 'g0'
        group.mkdir()
        for job in range(2):
            (self.root / 'results' / f'status{job}.json').rename(group / f'status{job}.json')
        self.assertEqual(len(self.files()), 2)
        self.rows[1]['complete'] = False
        self.write(group / 'status1.json', self.rows[1])
        with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
            self.files()

    def test_missing_completion_record_is_refused(self):
        (self.root / 'production_complete.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'not marked complete.*Wait for'):
            self.files()

    def test_partial_inventory_uses_only_validated_receipts(self):
        (self.root / 'production_complete.json').unlink()
        (self.root / 'results' / 'status1.json').unlink()
        files = validated_nano_files(self.base, self.campaign, self.root, allow_incomplete=True)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0][0].endswith('qcd_0to1/job0000000/nano.root'))

    def test_partial_inventory_skips_failed_attempt_but_rejects_bad_success(self):
        (self.root / 'production_complete.json').unlink()
        self.rows[1].update(complete=False, exit_code=1, nano_path=None, nano_bytes=None)
        self.refresh()
        with patch('Logger.warn') as warn:
            files = validated_nano_files(self.base, self.campaign, self.root, allow_incomplete=True)
        warn.assert_called_once_with('Nano production incomplete: 1/2 jobs ready.')
        self.assertEqual(len(files), 1)
        self.rows[0]['validated_tier_events']['NANO'] = 0
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
            validated_nano_files(self.base, self.campaign, self.root, allow_incomplete=True)

    def test_partial_flag_does_not_skip_failure_in_marked_complete_production(self):
        self.rows[1].update(complete=False, exit_code=1)
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
            validated_nano_files(self.base, self.campaign, self.root, allow_incomplete=True)

    def test_changed_sampling_plan_is_refused(self):
        (self.root / 'sampling_plan.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Sampling plan does not match'):
            self.files()

    def test_duplicate_job_is_refused(self):
        (self.root / 'all_jobs.txt').write_text('00000 0 1 0\n00001 0 1 0\n')
        with self.assertRaisesRegex(RuntimeError, 'unique planned job'):
            self.files()

    def test_missing_receipt_is_refused(self):
        (self.root / 'results/status1.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'receipt is missing.*status1'):
            self.files()

    def test_wrong_canonical_path_is_refused(self):
        self.rows[0]['nano_path'] = '/another/nano.root'
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
            self.files()

    def test_receipt_with_wrong_job_is_refused(self):
        self.rows[1]['job'] = 0
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
            self.files()

    def test_flat_final_tier_counts_are_refused(self):
        self.final['tier_events'] = {tier: 2 for tier in self.tiers}
        self.write(self.root / 'production_complete.json', self.final)
        with self.assertRaisesRegex(RuntimeError, 'all process/bin tiers'):
            self.files()

    def test_unvalidated_payload_is_refused(self):
        self.rows[0]['validated_tier_events']['SIM'] = 0
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'Unvalidated Nano receipt'):
            self.files()


if __name__ == '__main__':
    unittest.main()
