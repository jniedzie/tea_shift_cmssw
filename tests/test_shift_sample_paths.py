# Code style: AI-code. AI-generated regression test.
import copy
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
from shift_sample_paths import latest_merged_sample, validated_nano_files, validated_bsm_histogram_files
import shift_sample_paths


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



class SignalHistogramSampleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.contract = self.directory / "contract.json"
        self.contract.write_text(json.dumps(dict(mass_gev=15, epsilon=0.001,
                                                 requested_events=2, run_number=10)))
        self.histogram = self.directory / "histograms.root"
        self.histogram.write_bytes(b"retained histogram payload")
        self.report = dict(
            complete=True, input_event_ids=[[10, 1, 1], [10, 1, 2]],
            native_event_weight_sum=2.0, histogram_bytes=self.histogram.stat().st_size,
            histogram_sha256=hashlib.sha256(self.histogram.read_bytes()).hexdigest(),
        )
        self.point = dict(
            point="m15_prompt", detector_directory=str(self.directory),
            signal_contract=str(self.contract),
            signal_contract_sha256=hashlib.sha256(self.contract.read_bytes()).hexdigest(),
            events=2, detector_count=2, mass_gev=15, epsilon=0.001,
            target_mean_lab_flight_m=None,
        )
        self.plan = dict(schema="shift-dark-photon-grid-plan-v1", prepared=True,
                         points=[self.point])
        self.plan_path = self.directory / "plan.json"
        self.refresh()

    def refresh(self):
        self.plan_path.write_text(json.dumps(self.plan))
        (self.directory / "histogram_report.json").write_text(json.dumps(self.report))

    def test_complete_signal_retains_full_gen_exposure_and_unvalidated_rate(self):
        rows = validated_bsm_histogram_files(self.plan_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["initial_weight_sum"], 2)
        self.assertEqual(rows[0]["events"], 2)
        self.assertFalse(rows[0]["normalization_ready"])
        self.assertFalse(rows[0]["physics_valid"])
        self.assertIsNone(rows[0]["mean_lab_flight_m"])

    def test_unfinished_signal_cannot_silently_leave_a_partial_plot(self):
        self.report["complete"] = False
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            validated_bsm_histogram_files(self.plan_path)

    def test_missing_zero_reconstruction_event_is_not_a_full_gen_sample(self):
        self.report["input_event_ids"] = [[10, 1, 1]]
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "full GEN"):
            validated_bsm_histogram_files(self.plan_path)

    def test_changed_histogram_or_contract_is_rejected(self):
        for path in (self.histogram, self.contract):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    validated_bsm_histogram_files(self.plan_path)
                path.write_bytes(original)

    def test_duplicate_grid_point_is_rejected(self):
        self.plan["points"].append(dict(self.point))
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validated_bsm_histogram_files(self.plan_path)

    def test_plot_label_cannot_change_the_underlying_model(self):
        self.point["mass_gev"] = 30
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "model labels"):
            validated_bsm_histogram_files(self.plan_path)

    def test_refreshed_histograms_keep_the_original_signal_plan(self):
        refreshed = self.directory / "refreshed" / self.point["point"]
        refreshed.mkdir(parents=True)
        (refreshed / "histograms.root").write_bytes(self.histogram.read_bytes())
        (refreshed / "histogram_report.json").write_text(json.dumps(self.report))
        rows = validated_bsm_histogram_files(self.plan_path, refreshed.parent)
        self.assertEqual(rows[0]["histogram_path"], str(refreshed / "histograms.root"))
        self.assertEqual(rows[0]["events"], self.point["events"])


class BsmCampaignHistogramTest(unittest.TestCase):
    """Synthetic sealed campaigns; no real EOS or ROOT event data are used."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / 'data'
        self.validation = self.root / 'validation'
        self.deployment = self.validation / 'study' / 'replacement'
        self.campaign = '20261012-bsm-fixture'
        self.model_hash = 'b' * 64
        self.plan = dict(schema='shift-independent-signal-repartition-plan-v1', prepared=True,
                         submitted=False, campaign=self.campaign, points=[], events=8)
        self.inventory = dict(schema='shift-independent-signal-GEN-inventory-v1', complete=True,
                              each_parent_ledger_counted_once=True, points={})
        self.scoring = dict(model_sha256=self.model_hash)
        self.reports = {}
        self.histograms = {}
        for index, (mass, flight) in enumerate(((15, None), (15, 5), (30, None), (30, 5))):
            name = f'm{mass}_' + ('prompt' if flight is None else 'lab_5m')
            directory = self.base / ('darkphoton_' + name) / self.campaign
            epsilon = .001 if flight is None else 1.e-8
            ctau = .001 if flight is None else 10.
            run = 100 + index
            parent = dict(events=2, event_ids=[[run, 1, 1], [run, 1, 2]], native_sumw=2.,
                          native_sumw2=2., mass_gev=mass, epsilon=epsilon,
                          proper_ctau_mm=ctau, run_number=run, gen_sha256='c' * 64,
                          contract_sha256='d' * 64, sample_role='validation_only')
            self.inventory['points'][name] = parent
            self.plan['points'].append(dict(point=name, events=2, mass_gev=mass, epsilon=epsilon,
                proper_ctau_mm=ctau, run_number=run, target_mean_lab_flight_m=flight,
                final_output_base=str(directory)))
            histogram = directory / 'histograms_merged' / 'ntuple_0.root'
            histogram.parent.mkdir(parents=True)
            histogram.write_bytes(('synthetic histogram ' + name).encode())
            self.histograms[name] = histogram
            grouped = dict(path=str(directory / 'nanoAOD_merged_unscored/ntuple_0.root'),
                           sha256='e' * 64, bytes=128)
            union = dict(schema='shift-independent-signal-union-v1', complete=True, point=name,
                events=2, event_ids=copy.deepcopy(parent['event_ids']), parent_gen=copy.deepcopy(parent),
                native_sumw=2., native_sumw2=2., parent_gen_ledger_count=1,
                histogramming_started_after_complete_grouping=True, grouped=grouped)
            marker = dict(schema='shift-merged-dimuon-bdt-complete-v1', complete=True, events=2,
                model_sha256=self.model_hash, source=grouped['path'], source_sha256=grouped['sha256'],
                selection_applied=False, published=dict(histograms=dict(path=str(histogram),
                    bytes=histogram.stat().st_size, sha256=self.digest(histogram))))
            self.reports[name] = dict(schema='shift-independent-signal-point-complete-v1', complete=True,
                point=name, events=2, final_output_base=str(directory), union=union,
                score_histograms=dict(complete=True, marker=marker))
        self.freeze()

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')

    def freeze(self):
        for name, value in (('plan.json', self.plan), ('gen_inventory.json', self.inventory),
                            ('scoring_components.json', self.scoring)):
            self.write(self.deployment / name, value)
        self.write(self.deployment / 'seal.json', dict(campaign=self.campaign, files={
            str((self.deployment / name).resolve()): self.digest(self.deployment / name)
            for name in ('plan.json', 'gen_inventory.json', 'scoring_components.json')}))
        self.plan_hash = self.digest(self.deployment / 'plan.json')
        self.seal_hash = self.digest(self.deployment / 'seal.json')
        for name, report in self.reports.items():
            report.update(plan_sha256=self.plan_hash, seal_sha256=self.seal_hash)
            report['union'].update(plan_sha256=self.plan_hash, seal_sha256=self.seal_hash)
            report['score_histograms']['marker']['plan_sha256'] = self.seal_hash
            self.publish(name)

    def publish(self, name):
        report = self.reports[name]
        directory = self.base / ('darkphoton_' + name) / self.campaign
        self.write(self.deployment / 'completed_points' / (name + '.json'), report)
        self.write(directory / 'metadata/grouping/union.json', report['union'])
        self.write(directory / 'metadata/bdt_grouped/complete.json', report['score_histograms']['marker'])

    def files(self, masses=None, flights=None, **kwargs):
        return shift_sample_paths.validated_bsm_campaign_histograms(
            self.base, self.campaign, masses, flights,
            validation_directory=self.validation, **kwargs)

    def test_all_points_preserve_full_gen_weights_and_unvalidated_rates(self):
        rows = self.files()
        self.assertEqual({row['name'] for row in rows}, set(self.reports))
        self.assertTrue(all(row['events'] == row['initial_weight_sum'] == 2 for row in rows))
        self.assertTrue(all(row['physics_valid'] is False and row['normalization_ready'] is False
                            for row in rows))

    def test_original_stage_schema_keeps_sealed_full_gen_validation(self):
        self.plan['schema'] = 'shift-bdt-independent-signal-stage-v1'
        for row in self.plan['points']:
            prefix = row.pop('final_output_base')
            row['chunks'] = [dict(nano=prefix + '/nanoAOD/chunk_000.root')]
            self.reports[row['point']].pop('final_output_base')
        self.freeze()
        for name, report in self.reports.items():
            report.pop('plan_sha256')
            report['union'].pop('plan_sha256')
            self.publish(name)
        rows = self.files([15], [None, 5])
        self.assertEqual({row['name'] for row in rows}, {'m15_prompt', 'm15_lab_5m'})
        self.assertTrue(all(row['initial_weight_sum'] == 2 for row in rows))
        self.reports['m15_prompt']['union']['event_ids'].pop()
        self.publish('m15_prompt')
        with self.assertRaisesRegex(RuntimeError, 'GEN'):
            self.files([15], [None])

    def test_mass_flight_prompt_and_empty_selection(self):
        self.assertEqual({row['name'] for row in self.files([15], [None, 5])},
                         {'m15_prompt', 'm15_lab_5m'})
        self.assertEqual({row['name'] for row in self.files([15], [None])}, {'m15_prompt'})
        self.assertEqual({row['name'] for row in self.files(None, [5])}, {'m15_lab_5m', 'm30_lab_5m'})
        self.assertEqual(self.files([], None), [])
        self.assertEqual(self.files(None, []), [])
        self.assertEqual(shift_sample_paths.validated_bsm_campaign_histograms(self.base, None), [])

    def test_unselected_failed_point_does_not_block_requested_complete_points(self):
        self.reports['m30_lab_5m']['complete'] = False
        self.publish('m30_lab_5m')
        with patch('Logger.warn') as warn:
            self.assertEqual(len(self.files([15], [None, 5])), 2)
            warn.assert_not_called()
        with patch('Logger.warn') as warn:
            rows = self.files([30], [None, 5])
            self.assertEqual({row['name'] for row in rows}, {'m30_prompt'})
            warn.assert_called_once()
            self.assertIn('m30_lab_5m', warn.call_args.args[0])

    def test_missing_selected_report_or_publication_warns_and_keeps_available_curves(self):
        report = self.reports['m15_lab_5m']
        directory = Path(report['final_output_base'])
        paths = (self.deployment / 'completed_points/m15_lab_5m.json',
                 directory / 'metadata/grouping/union.json',
                 directory / 'metadata/bdt_grouped/complete.json', self.histograms['m15_lab_5m'])
        for path in paths:
            before = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(path=path), patch('Logger.warn') as warn:
                    rows = self.files([15], [None, 5])
                    self.assertEqual({row['name'] for row in rows}, {'m15_prompt'})
                    warn.assert_called_once()
                    self.assertIn('m15_lab_5m', warn.call_args.args[0])
            finally:
                path.write_bytes(before)

    def test_all_selected_incomplete_points_warn_and_return_no_curves(self):
        for name in ('m15_prompt', 'm15_lab_5m'):
            self.reports[name]['complete'] = False
            self.publish(name)
        with patch('Logger.warn') as warn:
            self.assertEqual(self.files([15], [None, 5]), [])
            self.assertEqual(warn.call_count, 2)
            messages = [call.args[0] for call in warn.call_args_list]
            self.assertTrue(any('m15_prompt' in message for message in messages))
            self.assertTrue(any('m15_lab_5m' in message for message in messages))

    def test_unfinished_local_grouping_scoring_or_marker_warns_and_keeps_available_curves(self):
        original = copy.deepcopy(self.reports['m15_lab_5m'])
        for stage in ('union', 'scoring', 'marker'):
            report = copy.deepcopy(original)
            if stage == 'union':
                report['union']['complete'] = False
            elif stage == 'scoring':
                report['score_histograms']['complete'] = False
            else:
                report['score_histograms']['marker']['complete'] = False
            self.reports['m15_lab_5m'] = report
            self.publish('m15_lab_5m')
            with self.subTest(stage=stage), patch('Logger.warn') as warn:
                rows = self.files([15], [None, 5])
                self.assertEqual({row['name'] for row in rows}, {'m15_prompt'})
                warn.assert_called_once()
                self.assertIn('m15_lab_5m', warn.call_args.args[0])

    def test_unknown_requested_values_and_cross_combination_are_rejected(self):
        for masses, flights in (([999], None), (None, [99]), ([15, 30], [None, 99])):
            with self.subTest(masses=masses, flights=flights), self.assertRaisesRegex(RuntimeError, 'absent'):
                self.files(masses, flights)
        self.plan['points'] = [row for row in self.plan['points'] if row['point'] != 'm30_lab_5m']
        self.inventory['points'].pop('m30_lab_5m')
        self.plan['events'] = 6
        self.freeze()
        # Both values separately exist, but the explicitly requested Cartesian point does not.
        with self.assertRaisesRegex(RuntimeError, 'points are absent'):
            self.files([15, 30], [None, 5])
        self.assertEqual(len(self.files()), 3)

    def test_invalid_selector_values_are_rejected(self):
        for masses, flights in (([True], None), ([0], None), ([float('nan')], None),
                                ('15', None), (None, [0]), (None, [float('inf')]), (None, 'prompt')):
            with self.subTest(masses=masses, flights=flights), self.assertRaises(RuntimeError):
                self.files(masses, flights)

    def test_changed_sealed_provenance_files_are_rejected(self):
        for name in ('plan.json', 'gen_inventory.json', 'scoring_components.json'):
            path = self.deployment / name
            before = path.read_bytes()
            path.write_bytes(before + b' ')
            try:
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'checksum|SHA|hash'):
                    self.files([15], [None])
            finally:
                path.write_bytes(before)

    def test_model_labels_cannot_disagree_with_the_full_gen_parent(self):
        original = copy.deepcopy(self.plan['points'][0])
        for key, value in (('mass_gev', 25), ('epsilon', .002), ('proper_ctau_mm', .002), ('run_number', 999)):
            self.plan['points'][0] = dict(original, **{key: value})
            self.freeze()
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'model|GEN'):
                self.files()
        self.plan['points'][0] = original

    def test_changed_gen_coverage_weights_and_grouping_are_rejected(self):
        original = copy.deepcopy(self.reports['m15_prompt'])
        mutations = (lambda report: report['union']['event_ids'].reverse(),
            lambda report: report['union']['event_ids'].pop(),
            lambda report: report['union'].update(native_sumw=1.),
            lambda report: report['union'].update(native_sumw2=1.),
            lambda report: report['union'].update(native_sumw=float('nan')),
            lambda report: report['union'].update(parent_gen_ledger_count=2),
            lambda report: report['union'].update(histogramming_started_after_complete_grouping=False),
            lambda report: report['union']['parent_gen'].update(gen_sha256='f' * 64),
            lambda report: report.update(plan_sha256='f' * 64),
            lambda report: report['union'].update(seal_sha256='f' * 64))
        for mutate in mutations:
            self.reports['m15_prompt'] = copy.deepcopy(original)
            mutate(self.reports['m15_prompt'])
            self.publish('m15_prompt')
            with self.subTest(mutation=mutate), self.assertRaises(RuntimeError):
                self.files([15], [None])

    def test_duplicate_expected_gen_id_and_cross_point_identity_are_rejected(self):
        original = copy.deepcopy(self.inventory['points']['m15_prompt'])
        point = self.inventory['points']['m15_prompt']
        point['event_ids'][1] = point['event_ids'][0]
        self.reports['m15_prompt']['union']['parent_gen'] = copy.deepcopy(point)
        self.reports['m15_prompt']['union']['event_ids'] = copy.deepcopy(point['event_ids'])
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'GEN'):
            self.files([15], [None])
        self.inventory['points']['m15_prompt'] = original
        self.reports['m15_prompt']['union']['parent_gen'] = copy.deepcopy(original)
        self.reports['m15_prompt']['union']['event_ids'] = copy.deepcopy(original['event_ids'])
        other = self.inventory['points']['m30_prompt']
        other['event_ids'] = copy.deepcopy(original['event_ids'])
        other['run_number'] = original['run_number']
        next(row for row in self.plan['points'] if row['point'] == 'm30_prompt')['run_number'] = original['run_number']
        self.reports['m30_prompt']['union']['parent_gen'] = copy.deepcopy(other)
        self.reports['m30_prompt']['union']['event_ids'] = copy.deepcopy(other['event_ids'])
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'GEN'):
            self.files([15, 30], [None])

    def test_marker_model_source_seal_and_selection_must_match(self):
        original = copy.deepcopy(self.reports['m15_prompt'])
        mutations = (lambda marker: marker.update(model_sha256='f' * 64),
                     lambda marker: marker.update(source_sha256='f' * 64),
                     lambda marker: marker.update(source='/unrelated/grouped.root'),
                     lambda marker: marker.update(plan_sha256=self.plan_hash),
                     lambda marker: marker.update(selection_applied=True),
                     lambda marker: marker.update(events=1),
                     lambda marker: marker['published']['histograms'].update(path='/unrelated/histograms.root'))
        for mutate in mutations:
            self.reports['m15_prompt'] = copy.deepcopy(original)
            mutate(self.reports['m15_prompt']['score_histograms']['marker'])
            self.publish('m15_prompt')
            with self.subTest(mutation=mutate), self.assertRaises(RuntimeError):
                self.files([15], [None])

    def test_real_grouping_and_completion_markers_must_match_local_report(self):
        directory = Path(self.reports['m15_prompt']['final_output_base'])
        for path in (directory / 'metadata/grouping/union.json',
                     directory / 'metadata/bdt_grouped/complete.json'):
            value = json.loads(path.read_text())
            value['complete'] = False
            self.write(path, value)
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                self.files([15], [None])
            self.publish('m15_prompt')

    def test_histogram_bytes_and_cached_copy_are_verified(self):
        histogram = self.histograms['m15_prompt']
        before = histogram.read_bytes()
        histogram.write_bytes(before + b'changed')
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            self.files([15], [None])
        histogram.write_bytes(before)
        cache = self.root / 'cache'
        row = self.files([15], [None], cache_directory=cache)[0]
        copied = Path(row['histogram_path'])
        self.assertEqual(copied.parent, cache)
        self.assertEqual(copied.read_bytes(), before)
        self.assertEqual(histogram.read_bytes(), before)
        copied.write_bytes(b'corrupt cache')
        with self.assertRaisesRegex(RuntimeError, 'Cached|checksum'):
            self.files([15], [None], cache_directory=cache)

    def test_duplicate_points_and_ambiguous_campaign_plans_are_rejected(self):
        original = copy.deepcopy(self.plan['points'])
        self.plan['points'].append(copy.deepcopy(original[0]))
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            self.files()
        self.plan['points'] = original
        self.freeze()
        duplicate = self.validation / 'other'
        for name in ('plan.json', 'gen_inventory.json', 'scoring_components.json'):
            path = duplicate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.deployment / name).read_bytes())
        self.write(duplicate / 'seal.json', dict(campaign=self.campaign, files={str((duplicate / name).resolve()):
            self.digest(duplicate / name) for name in ('plan.json', 'gen_inventory.json', 'scoring_components.json')}))
        with self.assertRaisesRegex(RuntimeError, 'ambiguous|exactly one|found 2'):
            self.files([15], [None])

    def test_unrelated_non_object_or_partial_plan_does_not_block_campaign_discovery(self):
        self.write(self.validation / 'unrelated' / 'plan.json', ['an unrelated inventory'])
        partial = self.validation / 'partial' / 'plan.json'
        partial.parent.mkdir()
        partial.write_text('{"schema":')
        self.assertEqual({row['name'] for row in self.files([15], [None])}, {'m15_prompt'})

    def test_corrupted_requested_plan_cannot_fall_back_to_finished_payloads(self):
        (self.deployment / 'plan.json').write_text('{"schema":')
        with self.assertRaises(RuntimeError):
            self.files([15], [None])

    def test_missing_or_invalid_frozen_model_hash_cannot_match_an_invalid_marker(self):
        for value in (None, '', 'not-a-sha256'):
            if value is None:
                self.scoring.pop('model_sha256', None)
            else:
                self.scoring['model_sha256'] = value
            for report in self.reports.values():
                marker = report['score_histograms']['marker']
                if value is None:
                    marker.pop('model_sha256', None)
                else:
                    marker['model_sha256'] = value
            self.freeze()
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.files([15], [None])

    def test_missing_or_invalid_grouped_source_hash_cannot_match_an_invalid_marker(self):
        original = copy.deepcopy(self.reports['m15_prompt'])
        for value in (None, '', 'not-a-sha256'):
            report = copy.deepcopy(original)
            if value is None:
                report['union']['grouped'].pop('sha256')
                report['score_histograms']['marker'].pop('source_sha256')
            else:
                report['union']['grouped']['sha256'] = value
                report['score_histograms']['marker']['source_sha256'] = value
            self.reports['m15_prompt'] = report
            self.publish('m15_prompt')
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.files([15], [None])


if __name__ == '__main__':
    unittest.main()
