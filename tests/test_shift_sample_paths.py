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


class StreamingHistogramInputsTest(unittest.TestCase):
    def test_published_campaign_resolves_without_working_directory_dependence(self):
        path = shift_sample_paths.campaign_histogram_inventory('20261016_mu_enriched_weekend_v1')
        self.assertIsNone(path)
        self.assertIsNone(shift_sample_paths.campaign_histogram_inventory('legacy_20260921'))
        self.assertEqual(shift_sample_paths.campaign_histogram_inventory('/tmp/snapshot.json'),
                         '/tmp/snapshot.json')

    def test_periodic_versions_select_newest_complete_merge_and_reject_unknown_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            older = directory/'ntuple_ready_20261019T100000Z_abcdef123456.root'
            newer = directory/'ntuple_ready_20261019T120000Z_abcdef123456.root'
            for path in (older, newer, directory/'ntuple_0.root'):
                path.write_bytes(b'fixture')
            self.assertEqual(shift_sample_paths.single_root_file(directory), str(newer))
            (directory/'other.root').write_bytes(b'unknown')
            with self.assertRaisesRegex(RuntimeError, 'exactly one ROOT'):
                shift_sample_paths.single_root_file(directory)

    def test_reference_uses_full_gen_weight_sum_instead_of_selected_cutflow(self):
        import ROOT
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'reference.root'
            source = ROOT.TFile(str(path), 'RECREATE')
            cutflow = ROOT.TH1D('cutFlow', '', 1, 0, 1)
            cutflow.SetBinContent(1, 17.)
            cutflow.Write()
            ROOT.TObjString(json.dumps(dict(schema='shift-preserved-full-GEN-histogram-v1',
                stratum='jpsi_10to20', full_source_sumw=.0009177813540578304,
                cross_section_pb=23.906403461436227, no_second_filter_efficiency=True,
                histogram_weight='native_gen_weight/reco_inclusion_probability'))).Write('ShiftFullGenNormalization')
            source.Close()
            inventory = directory / 'inputs.json'
            inventory.write_text(json.dumps(dict(schema='shift-streaming-plot-inputs-v1', samples=[],
                reference_samples=[dict(stratum='jpsi_10to20', path=str(path))])))
            rows = shift_sample_paths.streaming_reference_histogram_inputs(inventory)
            self.assertEqual(rows['jpsi_10to20']['full_source_sumw'], .0009177813540578304)
            self.assertEqual(rows['jpsi_10to20']['cross_section_pb'], 23.906403461436227)
            own = shift_sample_paths.histogram_gen_normalization(str(path), 'jpsi_10to20')
            self.assertEqual(own['initial_weight_sum'], .0009177813540578304)
            with self.assertRaisesRegex(ValueError, 'Invalid full GEN'):
                shift_sample_paths.histogram_gen_normalization(str(path), 'jpsi_0to1')

    def test_missing_declared_reference_does_not_select_a_legacy_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inventory = directory / 'inputs.json'
            inventory.write_text(json.dumps(dict(schema='shift-streaming-plot-inputs-v1', samples=[],
                reference_samples=[dict(stratum='dy_20to-1', path=str(directory/'missing.root'))])))
            with patch('Logger.warn') as warn:
                rows = shift_sample_paths.streaming_reference_histogram_inputs(inventory)
            self.assertEqual(rows, {'dy_20to-1': None})
            warn.assert_called_once()

    def test_missing_file_warns_without_hiding_ready_snapshot(self):
        import ROOT
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            missing = directory / 'missing.root'
            ready = directory / 'ready.root'
            source = ROOT.TFile(str(ready), 'RECREATE')
            ROOT.TObjString(json.dumps(dict(schema='shift-streaming-normalization-v1',
                stratum='qcd_20toinf', closed_cohorts=True, full_source_trials=100,
                sigma_nd_pb=7.e10, no_second_filter_efficiency=True,
                histogram_weight='native_gen_weight/reco_inclusion_probability'))).Write('ShiftStreamingNormalization')
            source.Close()
            inventory = directory / 'inputs.json'
            inventory.write_text(json.dumps(dict(schema='shift-streaming-plot-inputs-v1',
                samples=[dict(stratum='qcd_0to1', path=str(missing)),
                         dict(stratum='qcd_20toinf', path=str(ready))])))
            with patch('Logger.warn') as warn:
                rows = shift_sample_paths.streaming_histogram_inputs(inventory)
            self.assertEqual([row['stratum'] for row in rows], ['qcd_20toinf'])
            self.assertEqual(rows[0]['full_source_trials'], 100)
            warn.assert_called_once()
            self.assertIn(str(missing), warn.call_args.args[0])

    def test_raw_counts_cannot_be_used_as_a_rate_histogram(self):
        import ROOT
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'raw.root'
            source = ROOT.TFile(str(path), 'RECREATE')
            ROOT.TObjString(json.dumps(dict(schema='shift-streaming-normalization-v1',
                stratum='qcd_0to1', closed_cohorts=True, full_source_trials=100,
                sigma_nd_pb=7.e10, no_second_filter_efficiency=True,
                histogram_weight='one_per_reconstructed_entry'))).Write('ShiftStreamingNormalization')
            source.Close()
            inventory = directory / 'inputs.json'
            inventory.write_text(json.dumps(dict(schema='shift-streaming-plot-inputs-v1',
                samples=[dict(stratum='qcd_0to1', path=str(path), raw_path=str(path))])))
            with self.assertRaisesRegex(ValueError, 'weight policy'):
                shift_sample_paths.streaming_histogram_inputs(inventory)
            self.assertEqual(len(shift_sample_paths.streaming_histogram_inputs(inventory, raw=True)), 1)


class AvailableBsmHistogramsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.campaign = '20261013-partial-histograms'

    def histogram(self, point):
        directory = self.base / ('darkphoton_' + point) / self.campaign / 'histograms_merged'
        directory.mkdir(parents=True)
        path = directory / 'available.root'
        path.write_bytes(b'available partial histogram')
        return path

    def inputs(self, masses=None, flights=None, epsilons=None):
        with patch('shift_sample_paths._bsm_campaign_records', side_effect=AssertionError('Production validation')):
            return shift_sample_paths.available_bsm_campaign_histograms(self.base, self.campaign, masses, flights,
                                                                      epsilons=epsilons)

    def test_epsilon_points_and_lifetime_points_have_independent_selectors(self):
        self.histogram('m15_prompt')
        epsilon = self.histogram('m15_epsilon_1e-06')
        self.histogram('m15_epsilon_0p01')
        self.histogram('m30p5_epsilon_3p1622777e-08')
        self.assertEqual(len(self.inputs()), 4)
        rows = self.inputs([15], [], [1.e-6])
        self.assertEqual(rows, [dict(name='m15_epsilon_1e-06', histogram_path=str(epsilon),
                                    mass_gev=15., mean_lab_flight_m=None, epsilon=1.e-6)])
        self.assertEqual([row['name'] for row in self.inputs([15], [None], [])], ['m15_prompt'])
        self.assertEqual(self.inputs(None, [], []), [])
        self.assertEqual(self.inputs([30.5], [], [3.1622777e-8])[0]['epsilon'], 3.1622777e-8)

    def test_missing_epsilon_point_warns_without_hiding_ready_point(self):
        self.histogram('m15_epsilon_1e-06')
        with patch('Logger.warn') as warn:
            rows = self.inputs([15], [], [1.e-6, 1.e-4])
        self.assertEqual([row['name'] for row in rows], ['m15_epsilon_1e-06'])
        warn.assert_called_once()
        self.assertIn('epsilon 0.0001', warn.call_args.args[0])

    def test_files_need_no_validation_or_completion_records(self):
        path = self.histogram('m15_lab_100m')
        # A stale/incomplete record is irrelevant to ordinary plotting too.
        (path.parent.parent / 'complete.json').write_text('{not a completion record')
        rows = self.inputs([15], [100])
        self.assertEqual(rows, [dict(name='m15_lab_100m', histogram_path=str(path),
                                    mass_gev=15., mean_lab_flight_m=100.)])

    def test_unfinished_points_warn_and_do_not_hide_available_points(self):
        path = self.histogram('m15_prompt')
        (self.base / 'darkphoton_m15_lab_5m' / self.campaign / 'histograms_merged').mkdir(parents=True)
        with patch('Logger.warn') as warn:
            rows = self.inputs([15, 30], [None, 5])
        self.assertEqual([row['histogram_path'] for row in rows], [str(path)])
        self.assertEqual(warn.call_count, 3)

    def test_selectors_and_decimal_point_names(self):
        self.histogram('m15_prompt')
        self.histogram('m15_lab_1000m')
        self.histogram('m30p5_lab_0p5m')
        self.assertEqual([row['name'] for row in self.inputs(None, [None])], ['m15_prompt'])
        self.assertEqual([row['name'] for row in self.inputs([15], None)], ['m15_prompt', 'm15_lab_1000m'])
        self.assertEqual(self.inputs([30.5], [.5])[0]['mean_lab_flight_m'], .5)
        self.assertEqual(len(self.inputs()), 3)

    def test_disabled_or_unavailable_campaign_does_not_need_validation(self):
        self.assertEqual(self.inputs([], None), [])
        self.assertEqual(self.inputs(None, []), [])
        self.assertEqual(shift_sample_paths.available_bsm_campaign_histograms(self.base, None), [])
        with patch('Logger.warn'):
            self.assertEqual(self.inputs([15], [5]), [])
            self.assertEqual(shift_sample_paths.available_bsm_campaign_histograms(self.base / 'missing', self.campaign), [])

    def test_one_alias_can_select_different_runs_for_different_points(self):
        prompt = self.histogram('m15_prompt')
        self.campaign = '20261014-same-settings-more-lifetimes'
        displaced = self.histogram('m15_lab_1000m')
        for path in (prompt, displaced):
            campaign_directory = path.parent.parent
            (campaign_directory.parent / 'latest').symlink_to(campaign_directory.name,
                                                            target_is_directory=True)
        self.campaign = 'latest'
        rows = self.inputs([15], [None, 1000])
        self.assertEqual([row['name'] for row in rows], ['m15_prompt', 'm15_lab_1000m'])
        self.assertEqual([Path(row['histogram_path']).resolve() for row in rows], [prompt, displaced])

    def test_alias_keeps_ready_histograms_until_new_output_is_published(self):
        previous = self.histogram('m15_prompt')
        point_directory = previous.parent.parent.parent
        alias = point_directory / 'latest'
        alias.symlink_to(self.campaign, target_is_directory=True)
        self.campaign = '20261014-same-settings-more-events'
        pending = point_directory / self.campaign
        pending.mkdir()
        self.campaign = 'latest'
        self.assertEqual(Path(self.inputs([15], [None])[0]['histogram_path']).resolve(), previous)
        self.campaign = pending.name
        ready = self.histogram('m15_prompt')
        alias.unlink()
        alias.symlink_to(self.campaign, target_is_directory=True)
        self.campaign = 'latest'
        self.assertEqual(Path(self.inputs([15], [None])[0]['histogram_path']).resolve(), ready)

    def test_canonical_ctau_names_keep_lab_selectors_without_duplicate_legacy_links(self):
        point = 'm15_ctau_62p1784451mm'
        histogram = self.histogram(point)
        directory = histogram.parent.parent.parent
        (directory / 'sample.json').write_text(json.dumps(dict(
            schema='shift-dark-photon-sample-v1', point=point, mass_gev=15.,
            proper_ctau_mm=62.17844507957349, epsilon=1.0955803943963588e-7,
            target_mean_lab_flight_m=5., selector_kind='mean_lab_flight')))
        (self.base / 'darkphoton_m15_lab_5m').symlink_to(directory.name, target_is_directory=True)
        rows = self.inputs([15], [5], [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], point)
        self.assertEqual(rows[0]['mean_lab_flight_m'], 5.)
        self.assertEqual(rows[0]['proper_ctau_mm'], 62.17844507957349)
        self.assertEqual(rows[0]['histogram_path'], str(histogram))

    def test_ctau_named_epsilon_population_keeps_its_physical_parameters(self):
        point = 'm15_ctau_0p746325638mm'
        histogram = self.histogram(point)
        (histogram.parent.parent.parent / 'sample.json').write_text(json.dumps(dict(
            schema='shift-dark-photon-sample-v1', point=point, mass_gev=15.,
            proper_ctau_mm=.7463256382295375, epsilon=1.e-6,
            target_mean_lab_flight_m=None, selector_kind='epsilon')))
        rows = self.inputs([15], [], [1.e-6])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['epsilon'], 1.e-6)
        self.assertEqual(rows[0]['proper_ctau_mm'], .7463256382295375)

    def test_ctau_histogram_can_plot_without_optional_sample_metadata(self):
        histogram = self.histogram('m15_ctau_7p46324902em7mm')
        with patch('Logger.warn') as warn:
            rows = self.inputs([15])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['histogram_path'], str(histogram))
        self.assertAlmostEqual(rows[0]['proper_ctau_mm'], 7.46324902e-7)
        self.assertTrue(warn.called)

    def test_physical_lab_rename_keeps_partial_histograms_and_parameters(self):
        old_point = 'm15_ctau_62p1784451mm'
        old_histogram = self.histogram(old_point)
        old_directory = old_histogram.parent.parent.parent
        metadata = dict(schema='shift-dark-photon-sample-v1', point=old_point, mass_gev=15.,
                        proper_ctau_mm=62.17844507957349, epsilon=1.0955803943963588e-7,
                        target_mean_lab_flight_m=5., selector_kind='mean_lab_flight')
        (old_directory / 'sample.json').write_text(json.dumps(metadata))
        before = self.inputs([15], [5], [])

        # The physical directory moves; no production receipt or alias is needed.
        new_point = 'm15_lab_5m'
        new_directory = old_directory.with_name('darkphoton_' + new_point)
        old_directory.rename(new_directory)
        metadata['point'] = new_point
        (new_directory / 'sample.json').write_text(json.dumps(metadata))
        (self.base / 'darkphoton_m15_lab_30m' / self.campaign).mkdir(parents=True)
        with patch('Logger.warn') as warn:
            after = self.inputs([15], [5, 30], [])
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]['name'], new_point)
        for parameter in ('mass_gev', 'mean_lab_flight_m', 'proper_ctau_mm'):
            self.assertEqual(after[0][parameter], before[0][parameter])
        self.assertEqual(Path(after[0]['histogram_path']).read_bytes(), b'available partial histogram')
        self.assertTrue(warn.called)

    def test_lab_named_epsilon_sample_uses_recorded_mean_flight_and_physical_parameters(self):
        point = 'm15_lab_0p06m'
        histogram = self.histogram(point)
        (histogram.parent.parent.parent / 'sample.json').write_text(json.dumps(dict(
            schema='shift-dark-photon-sample-v1', point=point, mass_gev=15.,
            proper_ctau_mm=.7463256382295375, epsilon=1.e-6,
            target_mean_lab_flight_m=None, mean_lab_flight_m=.060014743,
            selector_kind='epsilon')))
        rows = self.inputs([15], [], [1.e-6])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['epsilon'], 1.e-6)
        self.assertEqual(rows[0]['mean_lab_flight_m'], .060014743)
        self.assertEqual(rows[0]['proper_ctau_mm'], .7463256382295375)

    def test_scientific_lab_name_needs_no_sample_metadata(self):
        histogram = self.histogram('m15_lab_4p2em7m')
        rows = self.inputs([15], [4.2e-7], [])
        self.assertEqual(rows[0]['histogram_path'], str(histogram))
        self.assertEqual(rows[0]['mean_lab_flight_m'], 4.2e-7)

    def test_old_ctau_alias_does_not_duplicate_physical_lab_sample(self):
        histogram = self.histogram('m15_lab_5m')
        (self.base / 'darkphoton_m15_ctau_62p1784451mm').symlink_to(
            histogram.parent.parent.parent.name, target_is_directory=True)
        rows = self.inputs([15], [5], [])
        self.assertEqual([row['name'] for row in rows], ['m15_lab_5m'])


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

    def prepare_partial_point(self):
        name = 'm15_lab_5m'
        row = next(row for row in self.plan['points'] if row['point'] == name)
        parent = self.inventory['points'][name]
        row['chunks'] = [dict(chunk=index, skip=index, count=1,
            nano=row['final_output_base'] + f'/nanoAOD/chunk_{index:03d}.root') for index in range(2)]
        self.freeze()
        (self.deployment / 'completed_points' / (name + '.json')).unlink()
        source_path = self.deployment / 'statuses' / (name + '_000.json')
        status = dict(schema='shift-independent-signal-chunk-v1', complete=True, point=name,
            chunk=0, events=1, native_sumw=1., selection_applied=False,
            expected_event_ids=parent['event_ids'][:1], parent_gen_sha256=parent['gen_sha256'],
            plan_sha256=self.plan_hash, seal_sha256=self.seal_hash, nano_sha256='a' * 64,
            published=dict(nanoAOD=dict(path=row['chunks'][0]['nano'], sha256='a' * 64)))
        self.write(source_path, status)
        prepared = self.validation / 'bsm_plot_inputs_fixture' / name
        grouped = prepared / 'nanoAOD_merged' / 'ntuple_0.root'
        grouped.parent.mkdir(parents=True)
        grouped.write_bytes(b'audited merged partial Nano payload')
        histogram = prepared / 'histograms_merged' / 'ntuple_0.root'
        histogram.parent.mkdir()
        histogram.write_bytes(b'partial histogram payload')
        receipt = dict(row, schema='shift-bsm-plot-histogram-v1', point=name, campaign=self.campaign,
            complete=True, production_complete=False, partial=True, selection_applied=False,
            events=1, full_gen_events=2, parent_gen=copy.deepcopy(parent),
            plan_sha256=self.plan_hash, seal_sha256=self.seal_hash,
            input_event_ids=copy.deepcopy(parent['event_ids'][:1]),
            missing_event_ids=copy.deepcopy(parent['event_ids'][1:]),
            native_event_weight_sum=1., native_event_weight_sum2=1.,
            grouped_nano_path=str(grouped), grouped_nano_sha256=self.digest(grouped),
            histogram_path=str(histogram), histogram_bytes=histogram.stat().st_size,
            histogram_sha256=self.digest(histogram), source_chunks=[dict(chunk=0,
                receipt_path=str(source_path), receipt_sha256=self.digest(source_path),
                nano_path=row['chunks'][0]['nano'], nano_sha256='a' * 64)])
        receipt_path = prepared / 'point_receipt.json'
        self.write(receipt_path, receipt)
        return receipt_path, receipt

    def test_partial_point_is_plotted_with_the_original_full_gen_denominator(self):
        _, receipt = self.prepare_partial_point()
        with patch('Logger.warn') as warn:
            rows = self.files([15], [None, 5])
            warn.assert_not_called()
        partial = next(row for row in rows if row['name'] == 'm15_lab_5m')
        self.assertEqual(partial['histogram_path'], receipt['histogram_path'])
        self.assertEqual(partial['events'], 1)
        self.assertEqual(partial['generated_events'], 2)
        self.assertEqual(partial['initial_weight_sum'], 2.)
        self.assertFalse(partial['production_complete'])

    def test_partial_point_cannot_change_parent_subset_or_native_weights(self):
        path, original = self.prepare_partial_point()
        mutations = (lambda item: item['parent_gen'].update(native_sumw=1.),
            lambda item: item.update(full_gen_events=1),
            lambda item: item.update(plan_sha256='f' * 64),
            lambda item: item.update(seal_sha256='f' * 64),
            lambda item: item.update(native_event_weight_sum=2.),
            lambda item: item.update(native_event_weight_sum2=2.),
            lambda item: item.update(input_event_ids=[[999, 1, 1]]),
            lambda item: item.update(missing_event_ids=[]),
            lambda item: item['source_chunks'].append(copy.deepcopy(item['source_chunks'][0])),
            lambda item: item['source_chunks'][0].update(nano_sha256='f' * 64),
            lambda item: item.update(selection_applied=True))
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            self.write(path, changed)
            with self.subTest(mutation=mutate), self.assertRaises(RuntimeError):
                self.files([15], [5])

    def test_partial_point_payloads_and_source_receipts_are_checksum_verified(self):
        _, receipt = self.prepare_partial_point()
        paths = (Path(receipt['grouped_nano_path']), Path(receipt['histogram_path']),
                 Path(receipt['source_chunks'][0]['receipt_path']))
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b'corrupt')
            try:
                with self.subTest(path=path), self.assertRaises(RuntimeError):
                    self.files([15], [5])
            finally:
                path.write_bytes(original)

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
