# Code style: AI-code. AI-generated frozen DNN diagnostic boundary tests.
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from shift_vertex_dnn_diagnostics import acceptance, checked, frozen_statistics, selected_models, validate_model_origin


class DnnDiagnosticBoundary(unittest.TestCase):
    def test_combined_bundle_preserves_real_model_origin_and_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            origin = base / 'round3'
            origin.mkdir()
            (origin / 'source_snapshot').mkdir()
            payload = base / 'data.bin'
            payload.write_bytes(b'pinned data')
            ref = lambda p: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            source = origin / 'source_snapshot/trainer.py'
            source.write_text('# frozen source\n')
            splits = origin / 'splits.npz'
            np.savez(splits, ids=np.array([[1, 1, 9]], dtype=np.uint64), expanded_split=np.array([2]))
            recipe = {'name': 'chosen', 'mode': 'quality_relative'}
            protocol = {key: ref(payload) for key in ('cache_manifest', 'reco', 'labels', 'bdt_scores')}
            protocol.update(recipes=[recipe], splits=ref(splits), sources={'trainer.py': ref(source)})
            path = origin / 'protocol.json'
            path.write_text(json.dumps(protocol))
            result = dict(recipe=recipe, protocol=ref(path), validation={'threshold': .4}, model=ref(payload), history=ref(payload))
            (origin / 'chosen_validation.json').write_text(json.dumps(result))
            self.assertEqual(validate_model_origin('chosen', result, protocol)['protocol'], ref(path))
            changed = dict(result, recipe=dict(recipe, mode='all'))
            with self.assertRaisesRegex(ValueError, 'recipe'):
                validate_model_origin('chosen', changed, protocol)
            other = base / 'other_split.npz'
            np.savez(other, ids=np.array([[1, 1, 9]], dtype=np.uint64), expanded_split=np.array([0]))
            with self.assertRaisesRegex(ValueError, 'split differs'):
                validate_model_origin('chosen', result, dict(protocol, splits=ref(other)))
            source.write_text('# tampered snapshot\n')
            with self.assertRaisesRegex(ValueError, 'source snapshot'):
                validate_model_origin('chosen', result, protocol)

    def test_models_are_only_unique_validation_winners_and_fixed_references(self):
        lock = dict(selection_uses_test=False, rejection_winner='nn_a', flatness_winner='nn_a',
                    acceptance_envelope_winner=None, thresholds={'nn_unselected': .2})
        self.assertEqual(selected_models(lock), ['reference_uniform_bdt_3p0_s71', 'expanded_d3_s0_n400', 'nn_a'])
        self.assertEqual(selected_models(lock, include_flat_bdt=True),
                         ['reference_uniform_bdt_3p0_s71', 'expanded_d3_s0_n400', 'expanded_d3_s3_n400', 'nn_a'])
        lock['selection_uses_test'] = True
        with self.assertRaisesRegex(ValueError, 'Test-based'):
            selected_models(lock)

    def test_hash_tampering_fails_before_loading_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.npz'
            path.write_bytes(b'original')
            ref = {'path': str(path), 'sha256': hashlib.sha256(b'original').hexdigest()}
            self.assertEqual(checked(ref), path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed pinned'):
                checked(ref)

    def test_weight_probability_availability_is_not_statistical_support(self):
        source = Path(__file__).resolve().parents[2] / 'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier/bdt_performance.py'
        perf = frozen_statistics(source)
        passed = np.arange(40) < 30
        groups = [('dy', 1, 1, i) for i in range(40)]
        weights = np.ones(40)
        weights[0] = 1000
        result = acceptance(perf, passed, groups, weights, repeats=0)
        self.assertTrue(result['empirical']['supported'])
        self.assertTrue(result['weighted']['available'])
        self.assertFalse(result['weighted']['supported'])
        self.assertLess(result['weighted']['event_weight_effective_n'], 2)
        result = acceptance(perf, passed, [('dy', 1, 1, 1)] * 40, np.ones(40), repeats=0)
        self.assertFalse(result['empirical']['supported'])
        self.assertFalse(result['weighted']['supported'])

    def test_empty_genuine_population_has_no_efficiency_or_roc(self):
        source = Path(__file__).resolve().parents[2] / 'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier/bdt_performance.py'
        perf = frozen_statistics(source)
        result = acceptance(perf, np.array([], bool), [], np.array([]), repeats=0)
        self.assertIsNone(result['empirical']['efficiency'])
        self.assertIsNone(result['weighted']['efficiency'])
        roc = perf.roc_summary(np.zeros(30), np.linspace(0, 1, 30), [('qcd', 1, 1, i) for i in range(30)], repeats=0)
        self.assertIsNone(roc['auc'])
        self.assertFalse(roc['supported'])


if __name__ == '__main__':
    unittest.main()
