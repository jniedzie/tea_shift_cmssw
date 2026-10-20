# Code style: AI-code. AI-generated focused tests for measured-only reconstruction policies.
import unittest
import json
import ctypes
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

import shift_reco_audit as audit


class PairedReconstructionTests(unittest.TestCase):
    def test_upstream_event_draws_keep_zero_originals_and_source_strata(self):
        parents = [('fileA', 1), ('fileA', 2), ('fileB', 1), ('fileB', 2)]
        events = {key: {'source_context': key[0]} for key in parents}
        def rows(offset):
            return [dict(event=parents[0], row_index=0, metrics={'mass_relative': 1.+offset}),
                    dict(event=parents[2], row_index=7, metrics={'mass_relative': 3.+offset})]
        modes = {'etaOff': dict(events=events, pairs=rows(0.)),
                 'etaOn': dict(events=dict(events), pairs=rows(5.))}
        result = audit.upstream_event_statistics(modes, 'pairs', 'mass_relative', bootstrap=80)
        self.assertEqual(result['originals'], 4)
        self.assertEqual(result['source_strata'], {'fileA': 2, 'fileB': 2})
        self.assertEqual(result['arms']['etaOff']['zero_metric_originals'], 2)
        self.assertEqual(result['arms']['etaOn']['contributing_originals'], 2)
        comparison = result['paired_delta_vs_reference']['etaOn']
        self.assertAlmostEqual(comparison['estimate']['median'], 5.)
        self.assertAlmostEqual(comparison['interval68']['median']['low'], 5.)
        self.assertAlmostEqual(comparison['interval68']['median']['high'], 5.)
        self.assertFalse(result['arms']['etaOn']['quantile_resolved'])

    def test_upstream_singleton_source_strata_do_not_fabricate_precision(self):
        events = {('fileA', 1): {'source_context': 'fileA'}, ('fileB', 1): {'source_context': 'fileB'}}
        rows = [dict(event=key, row_index=0, metrics={'mass_relative': value})
                for key, value in zip(events, (.1, .2))]
        result = audit.upstream_event_statistics({'etaOff': dict(events=events, pairs=rows)},
                                                 'pairs', 'mass_relative', bootstrap=20)
        self.assertEqual(result['arms']['etaOff']['contributing_originals'], 2)
        self.assertEqual(result['arms']['etaOff']['resampleable_contributing_originals'], 0)
        self.assertNotIn('interval68', result['arms']['etaOff'])

    def test_upstream_statistics_keep_own_changed_and_empty_populations(self):
        events = {('source', i): {'source_context': 'source'} for i in range(4)}
        first = [dict(event=('source', 0), row_index=0, metrics={'p_relative': .1}),
                 dict(event=('source', 0), row_index=1, metrics={'p_relative': -.1})]
        # Different row numbers do not suppress or force-match either arm.
        changed = [dict(event=('source', 1), row_index=90, metrics={'p_relative': .3})]
        modes = {'etaOff': dict(events=events, muons=first),
                 'etaOn': dict(events=events, muons=changed)}
        result = audit.upstream_event_statistics(modes, 'muons', 'p_relative', bootstrap=40)
        self.assertEqual(result['arms']['etaOff']['finite_rows'], 2)
        self.assertEqual(result['arms']['etaOn']['finite_rows'], 1)
        self.assertEqual(result['arms']['etaOff']['contributing_originals'], 1)
        self.assertNotIn('interval68', result['arms']['etaOff'])
        self.assertNotIn('interval68', result['paired_delta_vs_reference']['etaOn'])
        modes['etaOn']['muons'] = []
        result = audit.upstream_event_statistics(modes, 'muons', 'p_relative', bootstrap=40)
        self.assertIsNone(result['arms']['etaOn']['estimate'])
        self.assertEqual(result['arms']['etaOn']['bootstrap_empty_draws'], 40)
        self.assertNotIn('interval68', result['arms']['etaOn'])

    def test_upstream_wrong_charge_and_orientation_remain_in_closure(self):
        truth = dict(pdgId=13, momentum=[0., 0., 10.])
        result = audit.upstream_muon_residuals([0., 0., -10.], 1, truth)
        self.assertTrue(result['wrong_charge'])
        self.assertTrue(result['direction_disagrees'])
        self.assertAlmostEqual(result['metrics']['production_p_relative'], 0.)
        self.assertAlmostEqual(result['metrics']['production_pz_relative'], -2.)
        self.assertAlmostEqual(result['metrics']['production_qoverp_relative'], 2.)
        self.assertAlmostEqual(result['metrics']['production_angle_rad'], np.pi)
        rest = audit.upstream_muon_residuals([0., 0., 10.], -1, dict(pdgId=13, momentum=[0., 0., 0.]))
        self.assertIsNone(rest['direction_disagrees'])
        self.assertEqual(rest['metrics'], {})
        self.assertIsNone(audit.upstream_muon_residuals([0., 0., 1.], -1, {})['wrong_charge'])
        invalid = audit.upstream_muon_residuals([np.nan, 0., 1.], 1, truth)
        self.assertTrue(invalid['wrong_charge'])
        self.assertEqual(invalid['metrics'], {})
        self.assertEqual(audit.upstream_report_values({'failed_state': [np.nan, np.inf, 3.]}),
                         {'failed_state': [None, None, 3.]})

    def test_upstream_pair_reference_requires_distinct_positive_simtracks_and_GEN(self):
        first = dict(matched_production=True, hit_simtrack_id=7, hit_gen_index=0, charge=1)
        second = dict(matched_production=True, hit_simtrack_id=8, hit_gen_index=1, charge=1)
        self.assertEqual(audit.upstream_pair_reference_status(first, second), 'distinct_hit_associated_muons')
        self.assertEqual(audit.upstream_pair_reference_status(first, dict(second, hit_simtrack_id=7)), 'same_simtrack_split')
        self.assertEqual(audit.upstream_pair_reference_status(first, dict(second, hit_simtrack_id=-1)), 'missing_hit_simtrack_identity')
        self.assertEqual(audit.upstream_pair_reference_status(first, dict(second, hit_gen_index=0)), 'same_hit_associated_GEN_muon')
        self.assertEqual(audit.upstream_pair_reference_status(first, dict(second, matched_production=False)), 'missing_or_nonmuon_hit_reference')

    def test_upstream_plot_points_exclude_singletons_and_never_request_width(self):
        arm = dict(contributing_originals=1, estimate={'median': .1, 'sigma68': 9.},
                   interval68={'median': dict(error_down=.02, error_up=.03)})
        report = dict(sources=[{'source': dict(sample='jpsi_test', process='JPsi')}], groups=[dict(
            name='sample/jpsi_test', categories={'Both-Both': {'metrics': {'pairs': {
                'production_mass_relative': {'arms': {'etaOff': arm, 'etaOn': dict(arm, contributing_originals=2)}}}}}})])
        points = audit.upstream_plot_summary_points(report, 'Both-Both', 'pairs', 'production_mass_relative')
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['mode'], 'etaOn')
        self.assertEqual(points[0]['value'], .1)
        self.assertEqual(points[0]['error_up'], .03)

    def test_upstream_case_cells_show_absolute_error_changes_without_missing_fallback(self):
        def pair(mass, vertex):
            return dict(category='Both-Both', canonical={'mass': 3.},
                        metrics=dict(production_mass_relative=mass, production_vertex_z_cm=vertex))
        case = dict(arms={'etaOff': {'pairs': [pair(-.2, 300.)]}, 'etaOn': {'pairs': [pair(-.1, -100.)]}})
        cells = audit.upstream_event_case_cells(case, 'Both-Both', ['etaOff', 'etaOn'])
        self.assertAlmostEqual(cells['delta_abs_mass_error_pp'], -10.)
        self.assertAlmostEqual(cells['delta_abs_z_error_m'], -2.)
        case['arms']['etaOn']['pairs'] = []
        self.assertIsNone(audit.upstream_event_case_cells(case, 'Both-Both', ['etaOff', 'etaOn'])['delta_abs_mass_error_pp'])
        case['arms']['etaOn']['pairs'] = [pair(0., 0.), pair(.2, 100.)]
        self.assertIsNone(audit.upstream_event_case_cells(case, 'Both-Both', ['etaOff', 'etaOn'])['etaOn']['mass_gev'])

    def test_upstream_native_source_integrity_does_not_protect_fitted_tracks(self):
        key = ('context', 1, 1, 1)
        event = dict(gen_sha256='gen', weights={'genWeight': -.25}, weight_types={'genWeight': 'Float_t'},
                     raw_input_sha256='AOD', source_context='context', protected_muons_sha256='before')
        baseline = dict(events={key: event}, muons=[], pairs=[])
        changed = dict(events={key: dict(event, protected_muons_sha256='after')}, muons=[], pairs=[])
        audit.validate_upstream_populations({'etaOff': baseline, 'etaOn': changed})
        for field, value in (('weights', {'genWeight': .25}), ('weight_types', {}),
                             ('raw_input_sha256', 'other'), ('gen_sha256', 'other')):
            altered = dict(changed, events={key: dict(changed['events'][key], **{field: value})})
            with self.assertRaisesRegex(RuntimeError, 'GEN/source/native weights'):
                audit.validate_upstream_populations({'etaOff': baseline, 'etaOn': altered})
        row = dict(event=key, row_index=0)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            audit.validate_upstream_populations({'etaOff': baseline,
                'etaOn': dict(changed, muons=[row, dict(row)])})

    def test_upstream_physical_units_have_declared_tail_thresholds(self):
        relative = audit.upstream_scalar_statistics([.1, .6, 2.], 'mass_relative')
        vertex = audit.upstream_scalar_statistics([10., 200., 600.], 'production_vertex_z_cm')
        self.assertAlmostEqual(relative['tail_abs_gt_100pct'], 1./3.)
        self.assertAlmostEqual(vertex['tail_abs_gt_1m'], 2./3.)
        self.assertNotIn('tail100', vertex)
        self.assertNotIn('tail_abs_gt_50pct', vertex)

    def test_upstream_weight_concentration_keeps_signs_and_original_units(self):
        positive = audit.upstream_weight_accounting([1., 9.])
        self.assertAlmostEqual(positive['effective_originals_signed'], 100./82.)
        self.assertEqual(positive['nonzero_absolute_range'], 9.)
        self.assertTrue(positive['positive_weight_CDF_allowed'])
        signed = audit.upstream_weight_accounting([-1., 1.])
        self.assertEqual(signed['effective_originals_signed'], 0.)
        self.assertEqual(signed['effective_originals_absolute'], 2.)
        self.assertFalse(signed['positive_weight_CDF_allowed'])
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            audit.upstream_weight_accounting([1., np.inf])

    def test_upstream_route_is_closed_before_any_outcome_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/'manifest.json'; release = Path(directory)/'release.json'
            manifest.write_text('{}'); release.write_text(json.dumps({'released_for_analysis': False}))
            with patch.object(audit, 'read_upstream_reconstruction') as reader:
                with self.assertRaisesRegex(RuntimeError, 'not released'):
                    audit.run_upstream_replay(manifest, release)
                reader.assert_not_called()

    def test_upstream_case_migrations_retain_zeros_and_signed_weight_accounting(self):
        parents = [('context', 1, 1, index) for index in (1, 2, 3)]
        events = {key: dict(gen_sha256='fixed', weights={'genWeight': weight},
            weight_types={'genWeight': 'Float_t'}, raw_input_sha256='AOD', source_context='context',
            native_EDM_double_weight=weight) for key, weight in zip(parents, (-2., .5, 1.))}
        def muon(parent, index):
            return dict(event=parent, row_index=index, charge=1, algorithm=1, category='Both',
                matched_production=True, wrong_charge=False, direction_disagrees=False, metrics={})
        def pair(parent):
            return dict(event=parent, row_index=0, legs=[0, 1], category='Both-Both',
                distinct_hit_associated_muons=True, shared_production_vertex=True,
                reference_status='distinct_hit_associated_muons', metrics={})
        modes = {'etaOff': dict(events=events, muons=[muon(parents[0], i) for i in range(2)], pairs=[pair(parents[0])]),
                 'etaOn': dict(events=dict(events), muons=[muon(parents[1], i) for i in range(2)], pairs=[pair(parents[1])])}
        result = audit.upstream_population_report(modes)
        self.assertEqual(len(result['event_cases']), 3)
        self.assertEqual(result['transitions_vs_reference']['etaOn']['pairs'], {'1 -> 0': 1, '0 -> 1': 1, '0 -> 0': 1})
        self.assertEqual(result['transitions_vs_reference']['etaOn']['category/Both-Both']['0 -> 0'], 1)
        self.assertEqual(result['arms']['etaOff']['zero_selected_pair_originals'], 2)
        self.assertEqual(result['arms']['etaOn']['native_EDM_double_sumw'], -.5)
        self.assertEqual(result['arms']['etaOn']['persisted_weights']['genWeight']['sumw2'], 5.25)
        self.assertEqual(result['event_cases'][2]['arms']['etaOn']['pairs'], [])
        modes['etaOn']['pairs'][0]['legs'] = [0, 0]
        with self.assertRaisesRegex(RuntimeError, 'repeated muon legs'):
            audit.validate_upstream_populations(modes)

    def test_upstream_full_route_gates_sources_and_retains_pair_migrations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            def write(name, value):
                path = directory/name
                path.write_text(json.dumps(value))
                return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            native = {mode: write(mode+'.root', mode) for mode in ('etaOff', 'etaOn')}
            ids = [[1, 1, 1], [1, 1, 2]]
            declared = dict(source='sourceA', sample='jpsi_test', kernel='V10 engineering', parent_GEN_sha256='full GEN',
                full_GEN_accounting={'parent_gen_sha256': 'full GEN', 'full_gen_accepted_events': 100},
                original_AOD_reference={'sha256': 'AOD'}, selected_for_replay=[
                    dict(event_id=identity, native_EDM_double_weight=weight) for identity, weight in zip(ids, (-.25, .5))])
            cohort = write('cohort.json', {'sources': [declared]})
            gate = write('gate.json', dict(passed=True, original_equivalence=True, outputs=[dict(source='sourceA', mode=mode, **record)
                                                               for mode, record in native.items()]))
            source = dict(source_name='sourceA', source_context='context', process='JPsi', sample='jpsi_test',
                kernel='V10 engineering', raw_input_sha256='AOD', expected_ids=ids,
                expected_native_EDM_double_weights=[-.25, .5], reference='saved_hit_associated_GEN_engineering_only', modes=native)
            manifest = write('manifest.json', dict(cohort=cohort, native_gate=gate,
                                                    mode_order=['etaOff', 'etaOn'], sources=[source]))
            release = write('release.json', dict(released_for_analysis=True, manifest_sha256=manifest['sha256'],
                cohort_sha256=cohort['sha256'], analysis_source_sha256=audit.native_file_checksum(audit.__file__)))
            parents = [('context',)+tuple(identity) for identity in ids]
            def data(active):
                events = {key: dict(gen_sha256='GEN'+str(i), weights={'genWeight': weight},
                    weight_types={'genWeight': 'Float_t'}, source_context='context')
                    for i, (key, weight) in enumerate(zip(parents, (-.25, .5)))}
                muons = [dict(event=parents[active], row_index=i, charge=(-1 if i else 1), algorithm=1,
                    category='Both', matched_production=True, wrong_charge=False, direction_disagrees=False,
                    metrics={'production_p_relative': .1}) for i in range(2)]
                pair = dict(event=parents[active], row_index=0, legs=[0, 1], category='Both-Both',
                    distinct_hit_associated_muons=True, shared_production_vertex=True,
                    reference_status='distinct_hit_associated_muons', metrics={'production_mass_relative': .2})
                return dict(events=events, muons=muons, pairs=[pair])
            records = [(data(i), {'bytes': native[mode]['bytes']}) for i, mode in enumerate(native)]
            with patch.object(audit, 'read_upstream_reconstruction', side_effect=records):
                report = audit.run_upstream_replay(manifest['path'], release['path'], bootstrap=12)
            self.assertEqual(report['population']['transitions_vs_reference']['etaOn']['pairs'],
                             {'1 -> 0': 1, '0 -> 1': 1})
            self.assertEqual(report['population']['arms']['etaOn']['native_EDM_double_sumw'], .25)
            self.assertEqual(report['groups'][0]['categories']['Both-Both']['native_counts']['etaOn']['pairs'], 1)
            json.dumps(report, allow_nan=False)
            figures = audit.plot_upstream_report(report, directory/'figures')
            self.assertEqual(set(figures), {'Both-Both'})
            pdf = Path(figures['Both-Both']['path'])
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF'))
            self.assertEqual(figures['Both-Both']['pages'], 2)
            self.assertEqual(figures['Both-Both']['sha256'], audit.native_file_checksum(pdf))
            # An all-zero-weight source has undefined effective N, but still has valid cases to show.
            weights = report['population']['arms']['etaOff']['original_native_weights_by_source']
            for source_name in weights:
                weights[source_name] = audit.upstream_weight_accounting([0., 0.])
            zero_figures = audit.plot_upstream_report(report, directory/'zero_weight_figures')
            self.assertTrue(Path(zero_figures['Both-Both']['path']).read_bytes().startswith(b'%PDF'))
            # A historical failed gate needs a separate exact root qualification, never an implicit bypass.
            historical = write('historical.json', {'passed': False})
            qualification = write('qualification.json', dict(schema='shift-eta45-engineering-baseline-qualification-v1',
                released_for_downstream=True, passed_for_scope=True, original_full_raw_equality_passed=False,
                refs=[historical, cohort], mismatching_original=[1, 1, 1]))
            gate = write('gate.json', dict(passed=True, original_equivalence=False, comparison_qualification=qualification,
                outputs=[dict(source='sourceA', mode=mode, **record) for mode, record in native.items()]))
            manifest = write('manifest.json', dict(cohort=cohort, native_gate=gate, reference_directory=str(directory),
                comparison_qualification=qualification, mode_order=['etaOff', 'etaOn'], sources=[source]))
            release = write('release.json', dict(released_for_analysis=True, manifest_sha256=manifest['sha256'],
                cohort_sha256=cohort['sha256'], analysis_source_sha256=audit.native_file_checksum(audit.__file__)))
            with patch.object(audit, 'read_upstream_reconstruction') as reader:
                with self.assertRaisesRegex(RuntimeError, 'exact explicit engineering qualification'):
                    audit.run_upstream_replay(manifest['path'], release['path'], bootstrap=12)
                reader.assert_not_called()
            release = write('release.json', dict(released_for_analysis=True, manifest_sha256=manifest['sha256'],
                cohort_sha256=cohort['sha256'], comparison_qualification_sha256=qualification['sha256'],
                analysis_source_sha256=audit.native_file_checksum(audit.__file__)))
            records = [(data(i), {'bytes': native[mode]['bytes']}) for i, mode in enumerate(native)]
            with patch.object(audit, 'read_upstream_reconstruction', side_effect=records):
                qualified = audit.run_upstream_replay(manifest['path'], release['path'], bootstrap=12)
            self.assertFalse(qualified['original_equivalence'])
            self.assertFalse(qualified['comparison_qualification']['record']['original_full_raw_equality_passed'])
            self.assertFalse(json.loads(Path(historical['path']).read_text())['passed'])
            gate = write('gate.json', dict(passed=True, original_equivalence=True,
                outputs=[dict(source='sourceA', mode=mode, **record) for mode, record in native.items()]))
            # A manifest cannot replace the original source's exact double ledger.
            source['expected_native_EDM_double_weights'] = [.25, .5]
            manifest = write('manifest.json', dict(cohort=cohort, native_gate=gate,
                mode_order=['etaOff', 'etaOn'], sources=[source]))
            release = write('release.json', dict(released_for_analysis=True, manifest_sha256=manifest['sha256'],
                cohort_sha256=cohort['sha256'], analysis_source_sha256=audit.native_file_checksum(audit.__file__)))
            with patch.object(audit, 'read_upstream_reconstruction') as reader:
                with self.assertRaisesRegex(RuntimeError, 'source/weight/reference'):
                    audit.run_upstream_replay(manifest['path'], release['path'], bootstrap=12)
                reader.assert_not_called()
            # Changing a file/context label cannot count the same physical GEN original twice.
            source['expected_native_EDM_double_weights'] = [-.25, .5]
            second_declared = dict(declared, source='sourceB', original_AOD_reference={'sha256': 'AOD2'})
            second = dict(source, source_name='sourceB', source_context='different context', raw_input_sha256='AOD2')
            cohort = write('cohort.json', {'sources': [declared, second_declared]})
            gate = write('gate.json', dict(passed=True, original_equivalence=True, outputs=[dict(source=name, mode=mode, **record)
                for name in ('sourceA', 'sourceB') for mode, record in native.items()]))
            manifest = write('manifest.json', dict(cohort=cohort, native_gate=gate,
                mode_order=['etaOff', 'etaOn'], sources=[source, second]))
            release = write('release.json', dict(released_for_analysis=True, manifest_sha256=manifest['sha256'],
                cohort_sha256=cohort['sha256'], analysis_source_sha256=audit.native_file_checksum(audit.__file__)))
            with patch.object(audit, 'read_upstream_reconstruction') as reader:
                with self.assertRaisesRegex(RuntimeError, 'physical GEN original across source files'):
                    audit.run_upstream_replay(manifest['path'], release['path'], bootstrap=12)
                reader.assert_not_called()

    def test_detector_union_keeps_lost_new_and_charge_changed_pairs(self):
        first, second, third = ((1, 1, 1, 0, i) for i in range(3))
        pair = audit.detector_pair_identity(first, second)
        new_pair = audit.detector_pair_identity(second, third)
        event = dict(gen_sha256='fixed', weights={'genWeight': .25}, raw_input_sha256='same hits')
        baseline = dict(events={(1, 1, 1): event},
            muons={first: dict(charge=-1, topology=2), second: dict(charge=1, topology=2)},
            pairs={pair: dict(is_os=True, category='Both-Both')})
        changed = dict(events={(1, 1, 1): dict(event)},
            muons={second: dict(charge=-1, topology=1), third: dict(charge=1, topology=2)},
            pairs={new_pair: dict(is_os=True, category='Near-Both')})
        result = audit.detector_population_union({'reference': baseline, 'changed': changed})
        self.assertEqual(result['muon_common_all'], [second])
        self.assertEqual(result['pair_common_all'], [])
        self.assertEqual(set(result['pair_union']), {pair, new_pair})
        self.assertEqual(result['modes']['changed']['muons']['disappeared'], [first])
        self.assertEqual(result['modes']['changed']['muons']['appeared'], [third])
        self.assertEqual(result['modes']['changed']['muons']['transitions']['charge'], {'1 -> -1': 1})
        # Fitted multiplicities may change, but the raw input/GEN/weight contract may not.
        for field, value in (('gen_sha256', 'bad'), ('raw_input_sha256', 'new hits'), ('weights', {})):
            altered = dict(changed, events={(1, 1, 1): dict(event, **{field: value})})
            with self.assertRaisesRegex(RuntimeError, 'raw inputs, GEN or native weights'):
                audit.detector_population_union({'reference': baseline, 'changed': altered})
        self.assertEqual(pair, audit.detector_pair_identity(second, first))
        with self.assertRaisesRegex(ValueError, 'same event'):
            audit.detector_pair_identity(first, (2, 1, 1, 0, 1))

    def test_detector_decomposition_exposes_cancellation_of_fit_bias(self):
        value = audit.detector_momentum_decomposition(8., 10., 20., 20.)
        self.assertTrue(value['valid'])
        self.assertEqual(value['detector_error_gev'], -2.)
        self.assertEqual(value['material_return_error_gev'], 2.)
        self.assertEqual(value['total_error_gev'], 0.)
        self.assertAlmostEqual(value['detector_log_ratio']+value['material_return_log_ratio'],
                               value['total_log_ratio'])
        self.assertFalse(audit.detector_momentum_decomposition(8., -1., 20., 20.)['valid'])

    def surface_state(self, parameters, covariance=None):
        parameters = np.asarray(parameters, dtype=float)
        direction = np.asarray([parameters[1], parameters[2], 1.])
        momentum = direction/np.linalg.norm(direction)/abs(parameters[0])
        return dict(valid=True, surface=(123, 'native'), local_pz_sign=1, momentum=momentum, parameters=parameters,
                    covariance=np.eye(5) if covariance is None else covariance)

    def test_detector_paired_statistics_do_not_use_a_six_mode_intersection(self):
        result = audit.detector_baseline_statistics(
            {'reference': [1., 2., 3.], 'failed_control': [np.nan, np.nan, np.nan],
             'corrected': [2., 3., 4.]}, ['a', 'b', 'c'], bootstrap=40)
        paired = result['paired_vs_reference']
        self.assertEqual(paired['failed_control']['entries'], 0)
        self.assertEqual(paired['corrected']['entries'], 3)
        self.assertEqual(result['all_retained']['corrected']['entries'], 3)
        self.assertAlmostEqual(paired['corrected']['policies']['corrected'][
            'paired_delta_vs_canonical']['median']['value'], 1.)

    def test_detector_surface_reference_must_use_the_same_chart(self):
        fitted = self.surface_state([.1, 0., 0., 0., 0.])
        truth = dict(fitted, surface=(456, 'native'))
        result = audit.detector_surface_residuals(fitted, truth)
        self.assertTrue(result['measured_valid'])
        self.assertFalse(result['same_surface'])
        self.assertFalse(result['truth_valid'])
        self.assertEqual(result['metrics'], {})

    def test_detector_wrong_orientation_keeps_angle_but_not_local_pull_coverage(self):
        fitted = self.surface_state([.1, .01, .02, 0., 0.])
        truth = dict(fitted, local_pz_sign=-1, momentum=-fitted['momentum'])
        result = audit.detector_surface_residuals(fitted, truth)
        self.assertTrue(result['measured_valid'])
        self.assertTrue(result['truth_valid'])
        self.assertTrue(result['covariance_valid'])
        self.assertFalse(result['orientation_agrees'])
        self.assertFalse(result['pull_reference_valid'])
        self.assertAlmostEqual(result['metrics']['angle_error'], np.pi)
        self.assertNotIn('qoverp_pull', result['metrics'])

    def test_detector_pulls_use_full_correlated_fitted_covariance(self):
        covariance = np.diag([.0001, .01, .01, .04, .09])
        covariance[0, 3] = covariance[3, 0] = .001
        truth = self.surface_state([.1, .01, .01, 0., 0.])
        fitted = self.surface_state([.12, .02, .03, .4, -.3], covariance)
        result = audit.detector_surface_residuals(fitted, truth)
        self.assertTrue(result['covariance_valid'])
        self.assertAlmostEqual(result['metrics']['qoverp_pull'], 2.)
        self.assertAlmostEqual(result['metrics']['x_pull'], 2.)
        # q/p and x each differ by2sigma with correlation1/2; their joint contribution is16/3.
        self.assertAlmostEqual(result['metrics']['mahalanobis2'], 16./3.+.01+.04+1.)
        self.assertAlmostEqual(result['metrics']['p_relative'], 5./6.-1.)

    def test_detector_bad_covariance_never_replaces_a_measured_state(self):
        truth = self.surface_state([.1, 0., 0., 0., 0.])
        covariance = np.eye(5); covariance[0, 1] = covariance[1, 0] = 2.
        fitted = self.surface_state([.2, .01, 0., 0., 0.], covariance)
        result = audit.detector_surface_residuals(fitted, truth)
        self.assertTrue(result['measured_valid'])
        self.assertFalse(result['covariance_valid'])
        self.assertAlmostEqual(result['metrics']['p_relative'], -.5)
        self.assertNotIn('qoverp_pull', result['metrics'])
        unavailable = audit.detector_surface_residuals(fitted, dict(truth, valid=False))
        self.assertTrue(unavailable['measured_valid'])
        self.assertFalse(unavailable['truth_valid'])
        self.assertEqual(unavailable['metrics'], {})

    def test_detector_charge_error_stays_visible_in_same_surface_closure(self):
        truth = self.surface_state([.1, .01, .02, 0., 0.])
        fitted = self.surface_state([-.1, .01, .02, 0., 0.])
        result = audit.detector_surface_residuals(fitted, truth)
        self.assertAlmostEqual(result['metrics']['p_relative'], 0.)
        self.assertAlmostEqual(result['metrics']['qoverp_relative'], -2.)
        self.assertAlmostEqual(result['metrics']['angle_error'], 0.)

    def test_detector_pair_closure_uses_corresponding_surface_truth(self):
        first = self.surface_state([.01, .001, .002, 0., 0.])
        second = self.surface_state([-.02, -.001, .003, 1., 2.])
        result = audit.detector_pair_surface_closure(first, second, first, second)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['mass_error_gev'], 0.)
        self.assertAlmostEqual(result['opening_error'], 0.)
        self.assertFalse(audit.detector_pair_surface_closure(first, second, first,
                                                           dict(second, valid=False))['valid'])

    def test_detector_pull_coverage_does_not_treat_two_legs_as_two_parents(self):
        result = audit.detector_pull_coverage([.5, 3.], ['one event', 'one event'], bootstrap=40)
        self.assertEqual(result['physical_parents'], 1)
        self.assertEqual(result['one_sigma']['covered'], 1)
        self.assertEqual(result['one_sigma']['fraction'], .5)
        self.assertNotIn('interval68', result['one_sigma'])

    def test_detector_manifest_is_closed_until_explicit_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps({'released_for_analysis': False}))
            with patch.object(audit, 'read_detector_audit') as reader, patch.object(audit, 'native_file_checksum') as checksum:
                with self.assertRaisesRegex(RuntimeError, 'not been released'):
                    audit.run_detector_audit(path)
                reader.assert_not_called()
                checksum.assert_not_called()

    def test_detector_report_allows_pair_disappearance_and_retains_counters(self):
        context = 'fixedGEN'
        event_key = (context, 1, 1, 1)
        first, second = (event_key+(0, i) for i in range(2))
        pair_key = audit.detector_pair_identity(first, second)
        event = dict(gen_sha256='fixed', weights={'genWeight': 2.}, raw_input_sha256='input',
                     sample='DP5', process='DarkPhoton', counters={'inputCandidates': 4, 'retained': 2})
        def row(charge):
            measured = dict(Available=1, FitAccepted=1, CanonicalAccepted=1, NativeStatus=1,
                            TargetStatus=1, SnapshotKind=1, Variant=0, Iteration=1,
                            Charge=charge, LocalPzSign=1, InputHashLo=1, InputHashHi=2,
                            UsedHashLo=3, UsedHashHi=4)
            return dict(charge=charge, topology=2, raw_signature=(3.,), audit=measured,
                        truth_audit=dict(Status=1, Charge=charge, LocalPzSign=1, Span=.1, LowerKind=0, UpperKind=1),
                        truth={'valid': True}, production={'pdgId': -13*charge},
                        evaluation=dict(measured_valid=True, truth_valid=True, covariance_valid=True,
                                        pull_reference_valid=True),
                        metrics={'p_relative': .1, 'qoverp_pull': .5}, original_fallback=False,
                        legacy_hit_association_agrees=True)
        baseline = dict(events={event_key: event}, muons={first: row(-1), second: row(1)},
                        pairs={pair_key: dict(is_os=True, category='Both-Both',
                                             metrics={'production_mass_relative': .2})})
        changed = dict(events={event_key: dict(event, counters={'inputCandidates': 4, 'retained': 1})},
                       muons={first: row(-1)}, pairs={})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            protocol = directory/'protocol.json'
            protocol.write_text(json.dumps({'ordered_modes': [{'name': 'reference'}, {'name': 'changed'}]}))
            manifest = directory/'manifest.json'
            manifest.write_text(json.dumps(dict(released_for_analysis=True,
                protocol=dict(path=str(protocol), sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()),
                mode_order=['reference', 'changed'], sources=[dict(sample='DP5', process='DarkPhoton',
                    source_context=context, raw_input_sha256='input', expected_ids=[[1, 1, 1]],
                    modes={name: {'path': name, 'sha256': 'checked'} for name in ('reference', 'changed')})])))
            with patch.object(audit, 'native_file_checksum', return_value='checked'), patch.object(
                    audit, 'read_detector_audit', side_effect=[(baseline, {}), (changed, {})]):
                report = audit.run_detector_audit(manifest, bootstrap=40)
        group = report['groups'][0]
        self.assertEqual(group['telemetry']['changed']['counts']['event_retained'], 1)
        self.assertEqual(group['population']['modes']['changed']['pairs']['disappeared'], [pair_key])
        self.assertEqual(group['pair_metrics']['production_mass_relative'][
            'paired_vs_reference']['changed']['entries'], 0)
        self.assertEqual(group['muon_metrics']['p_relative'][
            'paired_vs_reference']['changed']['entries'], 1)
        ledger = group['selected_case_ledger']
        self.assertEqual(len(ledger['reference']['muons']), 2)
        self.assertEqual(len(ledger['changed']['muons']), 1)
        self.assertEqual(ledger['changed']['pairs'], [])
        self.assertEqual(ledger['reference']['muons'][0]['metrics'], baseline['muons'][first]['metrics'])
        json.dumps(report, allow_nan=False)

    def test_detector_reference_category_keeps_a_common_pair_after_category_migration(self):
        first, second = ((1, 1, 1, 0, i) for i in range(2))
        key = audit.detector_pair_identity(first, second)
        muons = {x: {'metrics': {'p_relative': .1}} for x in key}
        baseline = dict(muons=muons, pairs={key: dict(category='Both-Both', metrics={'mass': .2})})
        changed = dict(muons=muons, pairs={key: dict(category='Near-Both', metrics={'mass': .3})})
        result = audit.detector_category_statistics({'reference': baseline, 'changed': changed}, bootstrap=40)
        old = result['Both-Both']
        self.assertEqual(old['pair_metrics']['mass']['paired_vs_reference']['changed']['entries'], 1)
        self.assertEqual(old['pair_metrics']['mass']['all_retained']['changed']['entries'], 0)
        self.assertEqual(result['Near-Both']['native_counts']['changed']['pairs'], 1)

    def test_detector_surface_pair_ledger_labels_duplicate_and_unavailable_truth(self):
        keys = [(1, 1, 1, 0, i) for i in range(3)]
        muons = {key: dict(charge=1, topology=2, original_fallback=False, audit={},
            truth_audit={'SimTrackId': track}, evaluation={}, legacy_hit_association_agrees=None,
            metrics={}) for key, track in zip(keys, (7, 7, 8))}
        pairs = {audit.detector_pair_identity(keys[a], keys[b]): dict(category='Both-Both', is_os=False,
            metrics={'detector_pair_mass_relative': .1} if available else {})
            for a, b, available in ((0, 1, True), (0, 2, True), (1, 2, False))}
        ledger = audit.detector_selected_case_ledger({'mode': dict(muons=muons, pairs=pairs)})
        self.assertEqual([row['surface_proxy_simtrack_relation'] for row in ledger['mode']['pairs']],
                         ['same', 'distinct', 'unknown'])
        self.assertEqual(len(pairs), 3)
        json.dumps(ledger, allow_nan=False)

    def test_detector_comparison_allows_state_updates_but_protects_the_population(self):
        events = {(1, 1, 1): dict(muons=2, pairs=1, gen_sha256='a', weights={'genWeight': 2.},
                                 protected_muons_sha256='original fitted states')}
        pairs = {(1, 1, 1, 0): dict(signature=(0, 1, 2, 2, 4, 5), canonical={'mass': 3.})}
        updated_events = {(1, 1, 1): dict(events[(1, 1, 1)], protected_muons_sha256='new fitted states')}
        updated_pairs = {(1, 1, 1, 0): dict(pairs[(1, 1, 1, 0)], canonical={'mass': 4.})}
        audit.validate_detector_replay(events, pairs, updated_events, updated_pairs)
        for changed in (dict(gen_sha256='changed'), dict(weights={'genWeight': 3.}), dict(muons=1)):
            altered = {(1, 1, 1): dict(updated_events[(1, 1, 1)], **changed)}
            with self.assertRaisesRegex(RuntimeError, 'counts, GEN or native weights'):
                audit.validate_detector_replay(events, pairs, altered, updated_pairs)
        with self.assertRaisesRegex(RuntimeError, 'membership'):
            audit.validate_detector_replay(events, pairs, {}, {})
        altered_pairs = {(1, 1, 1, 0): dict(updated_pairs[(1, 1, 1, 0)], signature=(0, 1, 2, 2, 4, 6))}
        with self.assertRaisesRegex(RuntimeError, 'identities, topology or hit association'):
            audit.validate_detector_replay(events, pairs, updated_events, altered_pairs)

    def test_native_failure_status_is_retained_in_coverage(self):
        original = dict(mass=3., pt=1., pz=20., vx=2., vy=4., vz=14000.)
        measured = {'refitStatus': -1.}
        measured.update({'refitted'+n: 0. for n in ('Mass', 'Pt', 'Pz', 'Vx', 'Vy', 'Vz')})
        pair = dict(canonical=original, measured=measured, truth={}, signature=(0, 1, 2, 2), category=(2, 2))
        events = {(1, 1, 1): dict(muons=2, pairs=1, gen_sha256='a', weights={'genWeight': 1.})}
        stamp = {'pair_branches': list(measured)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(hypotheses={'joint': dict(prefix='refitted', status='refitStatus', muon_prefix='vertexRefitted')},
                sources=[dict(baseline='a.root', sample='test', process='test', source_context='a')])))
            with patch.object(audit, 'read_native_reconstruction', return_value=(events, {(1, 1, 1, 0): pair}, stamp)):
                report = audit.compare_native_reconstruction(path, bootstrap=20)['Both-Both']
        counts = report['coverage'][0]['counts']
        self.assertEqual(counts['joint_status_-1'], 1)
        self.assertEqual(counts['joint_used'], 0)
        self.assertNotIn('joint_association_failure', counts)

    def test_remote_checksum_reads_all_raw_bytes_and_closes_on_failure(self):
        payload = b'root'+bytes(range(256))*5000
        class Remote:
            closed = False
            fail = False
            def IsZombie(self): return False
            def GetSize(self): return len(payload)
            def ReadBuffer(self, buffer, offset, count):
                ctypes.memmove(buffer, payload[offset:offset+count], count)
                return self.fail
            def Close(self): self.closed = True
        source = Remote()
        with patch.object(audit.ROOT.TFile, 'Open', return_value=source):
            self.assertEqual(audit.native_file_checksum('root://example/file.root'), hashlib.sha256(payload).hexdigest())
        self.assertTrue(source.closed)
        source.closed = False; source.fail = True
        with patch.object(audit.ROOT.TFile, 'Open', return_value=source):
            with self.assertRaisesRegex(RuntimeError, 'checksum read failed'):
                audit.native_file_checksum('root://example/file.root')
        self.assertTrue(source.closed)

    def test_native_alternative_falls_back_as_one_complete_state(self):
        original = dict(mass=3., pt=1., pz=20., vx=2., vy=4., vz=14000.)
        alternative = dict(mass=2., pt=3., pz=22., vx=4., vy=6., vz=14500.)
        self.assertEqual(audit.native_alternative_state(original, alternative, 1), (alternative, True))
        for status in (0, -1, -4):
            self.assertEqual(audit.native_alternative_state(original, alternative, status), (original, False))
        alternative['vz'] = float('nan')
        self.assertEqual(audit.native_alternative_state(original, alternative, 1), (original, False))

    def test_native_comparison_rejects_changed_canonical_rows_and_gen(self):
        original = dict(mass=3., pt=1., pz=20., vx=2., vy=4., vz=14000.)
        pairs = {(1, 1, 1, 0): dict(canonical=original, measured={}, truth={}, signature=(0, 1, 2, 2), category=(2, 2))}
        events = {(1, 1, 1): dict(muons=2, pairs=1, gen_sha256='a', weights={'genWeight': 1.})}
        stamp = dict(pair_branches=['positionOnly'+n for n in ('Status', 'Mass', 'Pt', 'Pz', 'Vx', 'Vy', 'Vz')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(hypotheses={'position': {'prefix': 'positionOnly', 'status': 'positionOnlyStatus'}},
                sources=[dict(baseline='a.root', alternatives={'position': 'b.root'}, sample='test', process='test', source_context='a')])) )
            changed = {(1, 1, 1, 0): dict(pairs[(1, 1, 1, 0)], canonical=dict(original, mass=4.))}
            with patch.object(audit, 'read_native_reconstruction', side_effect=[(events, pairs, stamp), (events, changed, stamp)]):
                with self.assertRaisesRegex(RuntimeError, 'protected canonical'):
                    audit.compare_native_reconstruction(path, bootstrap=20)
            altered_events = {(1, 1, 1): dict(events[(1, 1, 1)], gen_sha256='b')}
            with patch.object(audit, 'read_native_reconstruction', side_effect=[(events, pairs, stamp), (altered_events, pairs, stamp)]):
                with self.assertRaisesRegex(RuntimeError, 'GEN or pair membership'):
                    audit.compare_native_reconstruction(path, bootstrap=20)
            with patch.object(audit, 'read_native_reconstruction', side_effect=[(events, pairs, stamp),
                             (events, pairs, {'pair_branches': []})]):
                with self.assertRaisesRegex(RuntimeError, 'branches missing'):
                    audit.compare_native_reconstruction(path, bootstrap=20)

    def measured(self):
        data = {}
        for name in ('mass', 'pt', 'pz', 'vx', 'vy', 'vz'):
            data['pair_'+name] = np.ones(4)
            data['pair_refitted'+name[0].upper()+name[1:]] = np.full(4, 2.)
        for leg in ('mu1', 'mu2'):
            for field in ('pt', 'pz', 'phi'):
                data[leg+'_'+field] = np.ones(4)
                data[leg+'_vertexRefitted'+field[0].upper()+field[1:]] = np.full(4, 2.)
            data[leg+'_px'] = np.ones(4)
            data[leg+'_py'] = np.zeros(4)
            data[leg+'_vertexRefitStatus'] = np.ones(4)
            data[leg+'_vertexRefitIdx'] = np.arange(4)
        data['pair_index'] = np.arange(4)
        data['pair_refitStatus'] = np.array([1., 1., 1., -3.])
        data['pair_refittedMinQoverPSignificance'] = np.array([1., 2., 3., 8.])
        return data

    def test_near_collinear_boosted_mass_is_stable(self):
        first = np.array([[0., 0., 1.e8], [0., 0., 0.]])
        second = first.copy()
        np.testing.assert_allclose(audit.stable_pair_mass(first, second), 2.*audit.MUON_MASS, rtol=1.e-14)

    def test_joint_rules_are_measured_only_and_preserve_every_pair(self):
        measured = self.measured()
        states, choices = audit.paired_policy_states(measured)
        np.testing.assert_array_equal(states['joint_sig2']['mass'], [1., 2., 2., 1.])
        np.testing.assert_array_equal(states['joint_sig3']['mass'], [1., 1., 2., 1.])
        measured['truth_pair_mass'] = np.array([100., 0., np.nan, -1.])
        measured['sample'] = np.array(['a', 'b', 'c', 'd'])
        repeated, _ = audit.paired_policy_states(measured)
        for policy in states:
            np.testing.assert_array_equal(repeated[policy]['mass'], states[policy]['mass'])
            self.assertEqual(len(states[policy]['mass']), 4)
        measured['pair_refittedMass'][2] = np.nan
        states, choices = audit.paired_policy_states(measured)
        self.assertEqual(states['joint']['mass'][2], 1.)
        self.assertTrue(choices['joint']['failed_alternative'][2])

    def test_joint_association_requires_both_legs_to_point_to_this_pair(self):
        measured = self.measured()
        measured['mu2_vertexRefitIdx'][1] = 99
        states, choices = audit.paired_policy_states(measured)
        self.assertEqual(states['joint']['mass'][1], measured['pair_mass'][1])
        self.assertFalse(choices['joint']['association_valid'][1])
        del measured['mu1_vertexRefitStatus']
        states, choices = audit.paired_policy_states(measured)
        np.testing.assert_array_equal(states['joint']['mass'], measured['pair_mass'])

    def test_counterfactual_identity_for_exact_measured_states(self):
        state = dict(pt=np.array([3.]), pz=np.array([17.]), vx=np.array([.1]), vy=np.array([.2]), vz=np.array([400.]),
                     mu1_pt=np.array([1.]), mu1_phi=np.array([.2]), mu1_pz=np.array([8.]),
                     mu2_pt=np.array([2.]), mu2_phi=np.array([-.1]), mu2_pz=np.array([9.]))
        first = audit.momentum_vector(state['mu1_pt'], state['mu1_phi'], state['mu1_pz'])
        second = audit.momentum_vector(state['mu2_pt'], state['mu2_phi'], state['mu2_pz'])
        state['mass'] = audit.stable_pair_mass(first, second)
        truth = {'truth_pair_mass': state['mass'], 'truth_pair_pt': state['pt'], 'truth_pair_pz': state['pz'],
                 'truth_pair_p': np.hypot(state['pt'], state['pz']), 'truth_pair_shared_vertex': np.array([1.]),
                 'truth_pair_openingAngle': np.arctan2(np.linalg.norm(np.cross(first, second), axis=1), np.sum(first*second, axis=1))}
        for axis in ('x', 'y', 'z'):
            truth['truth_pair_v'+axis] = state['v'+axis]
        for leg in ('mu1', 'mu2'):
            for field in ('pt', 'phi', 'pz'):
                truth['truth_'+leg+'_'+field] = state[leg+'_'+field]
        result = audit.reconstruction_residuals(state, truth)
        for values in result.values():
            np.testing.assert_allclose(values, 0., atol=1.e-14)

    def test_paired_parent_draws_cancel_identical_estimators(self):
        values = np.array([-.8, -.1, .1, 1., 2., -.2])
        groups = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
        result = audit.paired_parent_statistics({'canonical': values, 'copy': values.copy()}, groups, bootstrap=30)
        self.assertEqual(result['physical_parents'], 3)
        delta = result['policies']['copy']['paired_delta_vs_canonical']
        for item in delta.values():
            self.assertEqual(item, {'value': 0., 'low': 0., 'high': 0.})

    def test_one_physical_event_has_no_population_error_band(self):
        result = audit.paired_parent_statistics({'canonical': [1., 2.]}, ['same', 'same'], bootstrap=30)
        self.assertFalse(result['quantile_resolved'])
        self.assertNotIn('interval68', result['policies']['canonical'])

    def test_whole_points_and_processes_stay_out_of_development(self):
        sources = [dict(index=6,process='DarkPhoton',sample='darkphoton_m5_lab_5m'),
                   dict(index=7,process='DarkPhoton',sample='darkphoton_m5_prompt'),
                   dict(index=8,process='DarkPhoton',sample='darkphoton_m2_prompt'),
                   dict(index=9,process='DY',sample='dy_mHat2to5GeV')]
        data = dict(source_index=np.array([6, 7, 7, 7, 8, 9]), split_bucket=np.array([1, 1, 6, 9, 1, 1]),
                    initial_canary_peeked=np.zeros(6, dtype=bool), prior_development_peeked=np.zeros(6, dtype=bool))
        masks = audit.paired_development_masks(data, sources)
        np.testing.assert_array_equal(masks['development'], [False, True, False, False, False, False])
        np.testing.assert_array_equal(masks['event_holdout'], [False, False, False, True, False, False])
        np.testing.assert_array_equal(masks['point_holdout'], [True, False, False, False, True, False])
        np.testing.assert_array_equal(masks['process_holdout'], [False, False, False, False, False, True])

    def test_reindexed_canaries_and_rehashed_old_development_never_enter_event_holdout(self):
        sources = [dict(index=42, process='DarkPhoton', sample='darkphoton_m2_prompt'),
                   dict(index=43, process='DarkPhoton', sample='darkphoton_m5_prompt')]
        data = dict(source_index=np.array([42, 43, 43]), split_bucket=np.array([9, 9, 9]),
                    initial_canary_peeked=np.array([True, False, False]),
                    prior_development_peeked=np.array([True, True, False]))
        masks = audit.paired_development_masks(data, sources)
        np.testing.assert_array_equal(masks['development'], [True, False, False])
        np.testing.assert_array_equal(masks['point_holdout'], [False, False, False])
        np.testing.assert_array_equal(masks['event_holdout'], [False, False, True])

    def test_development_slices_before_outcome_evaluation_and_validation_requires_freeze(self):
        data = self.measured()
        data.update(source_index=np.array([6, 7, 7, 9]), split_bucket=np.array([1, 1, 9, 1]),
                    initial_canary_peeked=np.zeros(4, dtype=bool), prior_development_peeked=np.zeros(4, dtype=bool),
                    pair_topologyMin=np.full(4, 2), pair_topologyMax=np.full(4, 2),
                    truth_pair_matched=np.ones(4), truth_pair_mass=np.ones(4),
                    truth_pair_same_parent=np.ones(4), truth_pair_parent_pdgId=np.full(4,32),
                    truth_pair_shared_vertex=np.ones(4), truth_pair_vx=np.zeros(4), truth_pair_vy=np.zeros(4),
                    physical_parent_key=np.array(['a', 'b', 'c', 'd']))
        sources = [dict(index=6, process='DarkPhoton', sample='darkphoton_m2_prompt'),
                   dict(index=7, process='DarkPhoton', sample='darkphoton_m5_prompt'),
                   dict(index=9, process='DY', sample='dy_mHat2to5GeV')]
        def outcomes(state, truth):
            self.assertEqual(len(state['mass']), 1)
            np.testing.assert_array_equal(truth['physical_parent_key'], ['b'])
            return {'mass': np.array([.1])}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.savez(base/'pairs.npz', **data)
            (base/'extraction.json').write_text(json.dumps(dict(sources=sources, complete=True)))
            (base/'protocol.json').write_text(json.dumps(dict(created_before_full_outcome_analysis=True)))
            with patch.object(audit, 'reconstruction_residuals', side_effect=outcomes) as evaluation:
                result = audit.run_paired_audit(base/'pairs.npz', base/'extraction.json', base/'protocol.json', bootstrap=20)
                self.assertEqual(evaluation.call_count, len(audit.PAIRED_POLICIES))
                self.assertEqual(result['groups'][0]['matched_both_both'], 1)
                evaluation.reset_mock()
                with self.assertRaisesRegex(ValueError, 'requires a frozen policy'):
                    audit.run_paired_audit(base/'pairs.npz', base/'extraction.json', base/'protocol.json', mode='validation')
                evaluation.assert_not_called()


class ActualBirthReferenceTests(unittest.TestCase):
    def track(self, number=1, pdg=13, momentum=(3., 0., 4.), vertex=None):
        if vertex is None:
            vertex = dict(index=0, parent_G4_track_id=-1, process_subtype=0,
                          position_cm=[0., 0., -10000.], time_ns=2.)
        return dict(track_id=number, pdg=pdg, GEN_barcode=1000+number,
                    momentum_GeV=list(momentum), energy_GeV=float(np.hypot(np.linalg.norm(momentum), audit.MUON_MASS)),
                    vertex=vertex, matched_GEN_muon=None, saved_parent_chain=[],
                    missing_saved_parent_track_id=None, CSC_station_hits={})

    def record(self, tracks=None):
        tracks = [self.track()] if tracks is None else tracks
        return dict(schema='shift-material-scout-SIM-audit-v1', passed=True, checks={'identities': True},
                    complete_GEN={'sha256': 'a'*64}, events=[dict(event_id=[1, 2, 3], native_weight=.25,
                    actual_saved_muons=len(tracks), muons=tracks)])

    def references(self, tracks=None):
        return audit.read_actual_birth_reference(self.record(tracks), 'source:case')[(1, 2, 3)]

    def test_all_actual_muons_are_indexed_without_GEN_or_CSC_selection(self):
        reference = self.references([self.track(), self.track(2, -13)])
        self.assertEqual(set(reference['tracks']), {1, 2})
        matched = audit.actual_birth_muon_reference(dict(hitSimTrackId=2, charge=1), reference)
        self.assertEqual(matched['status'], 'matched')
        self.assertFalse(matched['wrong_charge'])
        self.assertEqual(matched['reference']['GEN_barcode'], 1002)

    def test_no_barcode_or_angular_fallback_for_a_missing_actual_identity(self):
        reference = self.references()
        self.assertEqual(audit.actual_birth_muon_reference(dict(hitSimTrackId=1001, charge=-1), reference)[
            'status'], 'no_saved_actual_muon')
        self.assertEqual(audit.actual_birth_muon_reference(dict(hitSimTrackId=-1, charge=-1), reference)[
            'status'], 'missing_hit_identity')

    def test_wrong_measured_charge_and_orientation_remain_in_closure(self):
        reference = self.references()
        matched = audit.actual_birth_muon_reference(dict(hitSimTrackId=1, charge=1), reference)
        self.assertEqual(matched['status'], 'matched')
        self.assertTrue(matched['wrong_charge'])
        metrics = audit.actual_birth_muon_metrics(dict(pt=3., phi=np.pi, pz=-4.), matched['reference'])
        self.assertAlmostEqual(metrics['p_relative'], 0.)
        self.assertAlmostEqual(metrics['angle_error_rad'], np.pi)
        self.assertAlmostEqual(metrics['pz_relative'], -2.)

    def test_non_muon_and_invalid_actual_birth_states_remain_coverage(self):
        hadron = self.track(pdg=211)
        matched = audit.actual_birth_muon_reference(dict(hitSimTrackId=1, charge=1), self.references([hadron]))
        self.assertEqual(matched['status'], 'non_muon_reference')
        invalid = self.track(); invalid['energy_GeV'] = float('nan')
        matched = audit.actual_birth_muon_reference(dict(hitSimTrackId=1, charge=-1), self.references([invalid]))
        self.assertEqual(matched['status'], 'invalid_actual_birth_state')
        vertex_invalid = self.track(); vertex_invalid['vertex']['position_cm'][0] = float('nan')
        matched = audit.actual_birth_muon_reference(dict(hitSimTrackId=1, charge=-1), self.references([vertex_invalid]))
        self.assertEqual(matched['status'], 'matched')  # momentum reference survives missing vertex truth
        self.assertIsNone(matched['reference']['vertex'])
        self.assertEqual(matched['reference']['vertex_status'], 'invalid')

    def test_duplicate_actual_ids_are_rejected_but_ids_reset_per_event(self):
        with self.assertRaisesRegex(ValueError, 'SimTrack ID'):
            self.references([self.track(), self.track()])
        record = self.record(); record['events'].append(dict(record['events'][0], event_id=[1, 2, 4]))
        references = audit.read_actual_birth_reference(record, 'source:case')
        self.assertEqual(len(references), 2)
        first, second = (references[e]['tracks'][1] for e in references)
        with self.assertRaisesRegex(ValueError, 'source/event context'):
            audit.actual_birth_pair_reference(first, second)
        with self.assertRaisesRegex(ValueError, 'source/event context'):
            audit.actual_birth_pair_reference(first, dict(first, context='another source/case', track_id=2))
        record['events'][1]['event_id'] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'Duplicate actual birth event'):
            audit.read_actual_birth_reference(record, 'source:case')

    def test_same_SimTrack_split_is_not_a_physical_two_muon_pair(self):
        reference = self.references()['tracks'][1]
        result = audit.actual_birth_pair_reference(reference, reference)
        self.assertEqual(result['status'], 'same_simtrack_split')
        self.assertNotIn('kinematic_birth_mass_gev', result)
        self.assertEqual(audit.actual_birth_pair_reference(reference, None)['status'], 'unmatched_leg')

    def test_rest_muon_is_valid_with_undefined_relative_momentum_and_angle(self):
        refs = self.references([self.track(1, momentum=(0., 0., 0.)), self.track(2, -13)])['tracks']
        self.assertTrue(refs[1]['valid'])
        metrics = audit.actual_birth_muon_metrics(dict(pt=0., phi=0., pz=1.), refs[1])
        self.assertNotIn('p_relative', metrics)
        self.assertNotIn('angle_error_rad', metrics)
        pair = audit.actual_birth_pair_reference(refs[1], refs[2])
        self.assertEqual(pair['status'], 'distinct_actual_muons')
        self.assertIsNone(pair['opening_angle'])
        self.assertGreater(pair['kinematic_birth_mass_gev'], 2.*audit.MUON_MASS)

    def test_shared_vertex_requires_physical_position_time_and_actual_links(self):
        first, second = (self.track(1), self.track(2, -13))
        def pair():
            refs = self.references([first, second])['tracks']
            return audit.actual_birth_pair_reference(refs[1], refs[2])
        self.assertTrue(pair()['shared_vertex'])  # common primary vertex, parent=-1
        second['vertex']['position_cm'][0] = 1.e-4
        self.assertFalse(pair()['shared_vertex'])
        second['vertex']['position_cm'][0] = 0.
        second['vertex']['time_ns'] += 1.e-4
        self.assertFalse(pair()['shared_vertex'])
        second['vertex']['time_ns'] = 2.
        second['vertex']['index'] = 7
        self.assertFalse(pair()['shared_vertex'])  # equal coordinates alone are insufficient
        first['vertex']['parent_G4_track_id'] = second['vertex']['parent_G4_track_id'] = 9
        self.assertTrue(pair()['shared_vertex'])  # separate preassigned vertex indices

    def test_separate_births_keep_stable_kinematics_without_a_vertex_truth(self):
        first, second = self.track(1, momentum=(0., 0., 1.e5)), self.track(2, -13, (0., 0., 1.e5))
        second['vertex']['position_cm'][2] += 100.
        refs = self.references([first, second])['tracks']
        result = audit.actual_birth_pair_reference(refs[1], refs[2])
        self.assertFalse(result['shared_vertex'])
        self.assertNotIn('vertex_cm', result)
        self.assertAlmostEqual(result['kinematic_birth_mass_gev'], 2.*audit.MUON_MASS)

    def test_sparse_rows_do_not_become_a_population_resolution(self):
        repeated = [dict(parent='one original', metrics={'p_relative': float(i)}) for i in range(30)]
        result = audit.actual_birth_group_statistics(repeated, bootstrap=20)['p_relative']
        self.assertFalse(result['population_summary_available'])
        self.assertIsNone(result['statistics'])
        independent = [dict(parent=str(i), metrics={'p_relative': i*.01}) for i in range(20)]
        result = audit.actual_birth_group_statistics(independent, bootstrap=20)['p_relative']
        self.assertTrue(result['population_summary_available'])
        self.assertEqual(result['statistics']['physical_parents'], 20)

    def test_native_Float32_weight_rounding_is_explicit_not_a_tolerance(self):
        native = 8.216943565012035e-09
        saved = dict(genWeight=float(np.float32(native)), shiftOriginalGenWeight=native,
                     shiftSamplingProbability=.25, shiftSamplingWeight=4., shiftSamplingGenWeight=native/.25,
                     shiftDecayProposalDensity=1., shiftDecayProposalWeight=1., shiftProposalGenWeight=native/.25)
        self.assertNotEqual(saved['genWeight'], native)
        audit.actual_birth_check_weights(saved, native, .25)
        with self.assertRaisesRegex(RuntimeError, 'serialized weight'):
            audit.actual_birth_check_weights(dict(saved, genWeight=native), native, .25)
        with self.assertRaisesRegex(RuntimeError, 'exact accounting'):
            audit.actual_birth_check_weights(dict(saved, shiftOriginalGenWeight=float(np.float32(native))), native, .25)
        with self.assertRaisesRegex(RuntimeError, 'exact accounting'):
            audit.actual_birth_check_weights(dict(saved, shiftDecayProposalWeight=2.), native, .25)
        missing = dict(saved); missing.pop('shiftProposalGenWeight')
        with self.assertRaisesRegex(RuntimeError, 'exact accounting'):
            audit.actual_birth_check_weights(missing, native, .25)

    def test_terminal_publication_binds_both_payloads_and_source_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sample = dict(id='case', event_ids=[[1, 2, 3]], native_weights=[.25], probability=1.)
            (base/'request.json').write_text(json.dumps(dict(source=dict(source_id='source', gen_sha256='a'*64), sample=sample)))
            (base/'nano.root').write_bytes(b'unit fixture; no native event data')
            record = self.record()
            record['complete_GEN']['sha256'] = 'd'*64  # timed/sliced GEN differs from original raw GEN
            record['SIM'] = {'sha256': 'e'*64}
            packed = audit.gzip.compress(json.dumps(record).encode())
            with audit.tarfile.open(base/'evidence.tar.gz', 'w:gz') as archive:
                member = audit.tarfile.TarInfo('material_audit.json.gz'); member.size = len(packed)
                archive.addfile(member, audit.io.BytesIO(packed))
                # The embedded report is deliberately prepublication, and is not terminal authority.
                raw = b'{"complete":false}'
                member = audit.tarfile.TarInfo('report.json'); member.size = len(raw)
                archive.addfile(member, audit.io.BytesIO(raw))
            publications = {n: dict(url='unit://'+n, bytes=(base/n).stat().st_size,
                                   sha256=audit.native_file_checksum(base/n)) for n in ('nano.root', 'evidence.tar.gz')}
            terminal = dict(complete=True, exit_code=0, source_id='source', original_source_sha256='a'*64,
                            request_sha256=audit.native_file_checksum(base/'request.json'),
                            published=list(publications.values()), removed_scratch_inputs=[
                                dict(path='timed_gen.root', sha256='d'*64), dict(path='step1.root', sha256='e'*64)])
            (base/'complete.json').write_text(json.dumps(terminal))
            case = dict(source_id='source', case_id='case', source={'gen_sha256':'a'*64}, sample=sample,
                nano_path=str(base/'nano.root'), material_evidence_path=str(base/'evidence.tar.gz'),
                request_path=str(base/'request.json'), request_sha256=terminal['request_sha256'],
                terminal_result_path=str(base/'complete.json'), terminal_result_sha256=audit.native_file_checksum(base/'complete.json'),
                publications=publications, expected_event_ids=[[1, 2, 3]], expected_native_EDM_double_weights=[.25])
            self.assertEqual(len(audit.actual_birth_case_inputs(case)), 1)
            changed = dict(case, expected_native_EDM_double_weights=[.5])
            with self.assertRaisesRegex(RuntimeError, 'request source, membership or weight'):
                audit.actual_birth_case_inputs(changed)
            terminal['published'][0] = dict(terminal['published'][0], sha256='b'*64)
            (base/'complete.json').write_text(json.dumps(terminal))
            case['terminal_result_sha256'] = audit.native_file_checksum(base/'complete.json')
            with self.assertRaisesRegex(RuntimeError, 'share the terminal publication'):
                audit.actual_birth_case_inputs(case)

    def test_exact_cohort_and_source_release_precede_every_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base/'manifest.json'
            manifest.write_text(json.dumps(dict(schema='shift-corrected-QCD-offline-birth-join-frozen-cohort-v1',
                                                kernel={'pileup':'on'})))
            release = base/'release.json'
            permission = dict(outcome_analysis_released=True, cohort_sha256='wrong',
                              analysis_source_sha256=audit.native_file_checksum(audit.__file__))
            release.write_text(json.dumps(permission))
            with patch.object(audit, 'actual_birth_case_inputs') as reader:
                with self.assertRaisesRegex(ValueError, 'exact frozen cohort'):
                    audit.run_actual_birth_audit(manifest, release)
                permission['cohort_sha256'] = audit.native_file_checksum(manifest)
                permission['analysis_source_sha256'] = 'wrong source'
                release.write_text(json.dumps(permission))
                with self.assertRaisesRegex(ValueError, 'exact frozen cohort'):
                    audit.run_actual_birth_audit(manifest, release)
                permission['analysis_source_sha256'] = audit.native_file_checksum(audit.__file__)
                release.write_text(json.dumps(permission))
                with self.assertRaisesRegex(ValueError, 'no pileup'):
                    audit.run_actual_birth_audit(manifest, release)
                reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
