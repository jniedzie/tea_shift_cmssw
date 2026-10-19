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


if __name__ == '__main__':
    unittest.main()
