# Code style: AI-code. AI-generated signal cache integrity regression tests.
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('signal_cache', ROOT / 'utils/shift_vertex_signal_cache.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SignalCacheTests(unittest.TestCase):
    def fixture(self):
        ids = [[123, 1, 2**60 + 1], [123, 1, 2**60 + 2]]
        parent = dict(event_ids=ids, events=2, full_gen_denominator={'accepted_events': 2},
                      validation_sha256='audit', native_sumw=2., native_sumw2=2.)
        source = dict(point='p', gen_audit={'sha256': 'audit'},
                      nano={'path': '/fixture.root', 'sha256': 'nano', 'bytes': 1})
        inventory = dict(complete=True, each_parent_ledger_counted_once=True,
                         sample_role='validation_only', points={'p': parent})
        audit = dict(runtime_validated=True, events=2, event_ids=ids,
                     event_rows=[dict(id=identity, weight=1.) for identity in ids])
        union = dict(complete=True, events=2, parent_gen_ledger_count=1,
                     point='p', parent_gen=parent, event_ids=ids)
        marker = dict(complete=True, selection_applied=False, events=2, pairs=1,
                      published={'nano': source['nano']})
        receipt = dict(schema='shift-independent-signal-point-complete-v1',
                       complete=True, events=2, point='p', sample_role='validation_only',
                       union=union, score_histograms={'marker': marker})
        return source, receipt, inventory, audit

    def test_exact_uint64_and_complete_coverage(self):
        source, receipt, inv, audit = self.fixture()
        parent, ids, weights, pairs = m.validate_point(source, receipt, inv, audit)
        self.assertEqual(ids.dtype, np.dtype('uint64'))
        self.assertEqual(int(ids[0, 2]), 2**60 + 1)
        m.validate_event_readback(ids, weights, np.array([1, 0]), ids, weights, pairs)

    def test_partial_receipt_rejected(self):
        for field, value in [('complete', False), ('partial', True), ('production_complete', False)]:
            source, receipt, inv, audit = self.fixture()
            receipt[field] = value
            with self.assertRaisesRegex(ValueError, 'Incomplete point'):
                m.validate_point(source, receipt, inv, audit)

    def test_missing_swapped_and_duplicate_parent_rejected(self):
        source, receipt, inv, audit = self.fixture()
        _, ids, weights, pairs = m.validate_point(source, receipt, inv, audit)
        for changed in (ids[:1], ids[::-1], ids[[0, 0]]):
            with self.assertRaisesRegex(ValueError, 'identit'):
                m.validate_event_readback(changed, weights, np.array([1, 0]), ids, weights, pairs)

    def test_float_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer'):
            m.identities(np.array([[1., 1., float(2**60 + 1)]]))

    def test_weights_and_candidate_count_rejected(self):
        source, receipt, inv, audit = self.fixture()
        _, ids, weights, pairs = m.validate_point(source, receipt, inv, audit)
        with self.assertRaisesRegex(ValueError, 'weights'):
            m.validate_event_readback(ids, weights * 2, np.array([1, 0]), ids, weights, pairs)
        with self.assertRaisesRegex(ValueError, 'count'):
            m.validate_event_readback(ids, weights, np.array([0, 0]), ids, weights, pairs)

    def test_swapped_receipt_nano_or_gen_audit_rejected(self):
        for target in ('nano', 'gen_audit'):
            source, receipt, inv, audit = copy.deepcopy(self.fixture())
            source = copy.deepcopy(source)
            source[target]['sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                m.validate_point(source, receipt, inv, audit)

    def test_full_denominator_and_ledger_rejected(self):
        source, receipt, inv, audit = self.fixture()
        receipt['union']['parent_gen_ledger_count'] = 2
        with self.assertRaisesRegex(ValueError, 'ledger'):
            m.validate_point(source, receipt, inv, audit)
        source, receipt, inv, audit = self.fixture()
        inv['points']['p']['full_gen_denominator']['accepted_events'] = 3
        with self.assertRaisesRegex(ValueError, 'denominator'):
            m.validate_point(source, receipt, inv, audit)

    def prepare_fixture(self, directory, excluded):
        def store(name, value):
            path = directory / name
            path.write_text(json.dumps(value))
            return dict(path=str(path), sha256=m.digest(path))
        source, receipt, inv, audit = self.fixture()
        source['gen_audit'] = store('audit.json', audit)
        inv['points']['p']['validation_sha256'] = source['gen_audit']['sha256']
        source['gen_inventory'] = store('inventory.json', inv)
        source['receipt'] = store('receipt.json', receipt)
        exclusion = directory / 'excluded.npz'
        np.savez(exclusion, ids=np.asarray(excluded, dtype=np.uint64).reshape(-1, 3))
        sources = {name: store(name, {}) for name in m.SOURCE_ORDER}
        return dict(schema='shift-complete-signal-cache-inputs-v1', sample_kind='simulation_only',
                    sample_role='validation_only', sources=sources, points=[source],
                    exclude_events=[dict(path=str(exclusion), sha256=m.digest(exclusion))])

    def test_manifest_rejects_prior_physical_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare_fixture(Path(directory), [[123, 1, 2**60 + 1]])
            with self.assertRaisesRegex(ValueError, 'previously inspected'):
                m.prepare_manifest(manifest)

    def test_metadata_only_performs_no_nano_read_and_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = self.prepare_fixture(directory, [[9, 1, 1]])
            path = directory / 'manifest.json'
            path.write_text(json.dumps(manifest))
            output = directory / 'output'
            result = m.extract_cache(path, output, validate_only=True)
            self.assertTrue(result['metadata_validated'])
            self.assertFalse(result['nano_readback_performed'])
            self.assertFalse(output.exists())
            manifest['points'].append(copy.deepcopy(manifest['points'][0]))
            with self.assertRaisesRegex(ValueError, 'Repeated signal point'):
                m.prepare_manifest(manifest)

    def run_synthetic_read(self, pair_count):
        import awkward as ak
        frozen = ROOT / 'plots/DNN_vertex_20261013/fresh_signal_transfer/cache/source_snapshot'
        sources = {name: dict(path=str((frozen / name).resolve()), sha256=m.digest(frozen / name))
                   for name in m.SOURCE_ORDER if name != 'shift_vertex_reco_information.py'}
        rich_path = ROOT / 'utils/shift_vertex_reco_information.py'
        sources[rich_path.name] = dict(path=str(rich_path), sha256=m.digest(rich_path))
        with m.frozen_readers(sources) as modules, tempfile.TemporaryDirectory() as temporary:
            f, rich, label = modules['features'], modules['shift_vertex_reco_information'], modules['labels']
            source, receipt, inv, audit = self.fixture()
            _, ids, weights, _ = m.validate_point(source, receipt, inv, audit)
            a = {}
            for key in set(f.RECO_BRANCHES + rich.BRANCHES):
                if key.startswith('ShiftMuon_'):
                    a[key] = ak.Array([[1., 2.], []])
                elif key.startswith('ShiftDimuonVertex_'):
                    a[key] = ak.Array([[1.] * pair_count, []])
            a.update(run=ids[:, 0], luminosityBlock=ids[:, 1], event=ids[:, 2],
                     nShiftMuon=np.array([2, 0]), nShiftDimuonVertex=np.array([pair_count, 0]),
                     genWeight=weights)
            for key in rich.EVENT[1:]:
                a[key] = np.array([0, 0])
            for key, value in [('muonIdx1', 0), ('muonIdx2', 1), ('topologyMin', 0),
                               ('topologyMax', 2), ('isOS', 1), ('dcaValid', 1)]:
                a['ShiftDimuonVertex_' + key] = ak.Array([[value] * pair_count, []])
            truth = {key: ak.Array([[0, 0], []]) for key in label.TRUTH_BRANCHES}
            truth.update(GenPart_pdgId=ak.Array([[13, -13, 32], []]),
                         GenPart_status=ak.Array([[1, 1, 3], []]),
                         GenPart_genPartIdxMother=ak.Array([[2, 2, -1], []]),
                         ShiftMuon_hitGenPartIdx=ak.Array([[0, 1], []]),
                         ShiftMuon_hitTruthPurity=ak.Array([[1., 1.], []]),
                         ShiftMuon_hitTruthMatchedLayers=ak.Array([[3, 3], []]),
                         ShiftMuon_hitSimTrackId=ak.Array([[10, 11], []]))
            for axis in ('vx', 'vy', 'vz'):
                truth['GenPart_' + axis] = ak.Array([[0., 0., 0.], []])
            calls = []
            class Tree:
                num_entries = 2
                def arrays(self, branches, **kwargs):
                    calls.append(tuple(branches))
                    values = truth if tuple(branches) == label.TRUTH_BRANCHES else a
                    return {key: values[key] for key in branches}
            class File:
                def __enter__(self): return {'Events': Tree()}
                def __exit__(self, *args): return False
            fake_uproot = types.SimpleNamespace(open=lambda *args, **kwargs: File(),
                source=types.SimpleNamespace(file=types.SimpleNamespace(MultithreadedFileSource=object)))
            path = Path(temporary) / 'fixture.root'
            path.write_bytes(b'synthetic fixture; no ROOT payload')
            source['nano'] = dict(path=str(path), sha256=m.digest(path), bytes=path.stat().st_size)
            with patch.dict('sys.modules', uproot=fake_uproot):
                reco, labels, events, row = m.extract_point(source, inv['points']['p'], ids,
                                                           weights, pair_count, modules, 0)
            self.assertEqual(reco['X'].shape, (pair_count, 119))
            self.assertEqual(len(events['ids']), 2)
            self.assertEqual(row['zero_pair_events'], 2 - int(pair_count > 0))
            np.testing.assert_array_equal(events['ids'], ids)
            self.assertFalse(any(key in label.TRUTH_BRANCHES for key in calls[0]))
            self.assertEqual(calls[1], label.TRUTH_BRANCHES)
            if pair_count:
                self.assertEqual(labels['y'].tolist(), [1])
                self.assertEqual(labels['genuine_signal_associated'].tolist(), [True])
                np.testing.assert_array_equal(reco['X'][:, :50], f.extract_reco(a)['X'])

    def test_synthetic_nonempty_read_separates_truth_and_preserves_X50(self):
        self.run_synthetic_read(1)

    def test_synthetic_empty_point_keeps_all_parents(self):
        self.run_synthetic_read(0)


if __name__ == '__main__':
    unittest.main()
