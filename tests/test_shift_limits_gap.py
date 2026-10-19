# Code style: AI-code. AI-generated synthetic candidate-cache regression tests.
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import shift_limits_gap as gap
import shift_limits_gap_analysis as analysis
import shift_limits_gap_checks as checks


class CandidateCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='shift-gap-', dir='/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        self.yields = self.root / 'yields.json'
        self.joint = self.root / 'cuts/data/qcd_0to1.npz'
        self.joint.parent.mkdir(parents=True)
        np.savez(self.joint, ids=np.array([[1, 2, 3]]), weight=np.array([1.]))
        self.joint.with_suffix('.json').write_text(json.dumps({'sha256': gap.digest(self.joint)}))
        self.nano = self.root / 'background.root'
        self.nano.write_bytes(b'synthetic background')
        backgrounds = {'qcd_0to1': {'source': {'files': [str(self.nano)]},
                                   'cross_section_pb': 2., 'full_gen_sumw': 1.}}
        self.yields.write_text(json.dumps({'backgrounds': backgrounds,
                                          'signals': {'m2_prompt': {}, 'm5_prompt': {}, 'm10_prompt': {}}}))
        self.parents = {}
        for name in ('m2_prompt', 'm5_prompt', 'm10_prompt'):
            parent = self.root / ('darkphoton_' + name) / 'latest'
            self.parents[name] = parent
            (parent / 'nanoAOD_merged').mkdir(parents=True)
            (parent / 'nanoAOD_merged/0.root').write_bytes(b'synthetic signal')
            (parent / 'metadata/grouping').mkdir(parents=True)
            (parent / 'metadata/grouping/union.json').write_text(json.dumps(
                {'complete': True, 'events': 1, 'native_sumw': 1.}))
            (parent / 'cross_sections.json').write_text(json.dumps(
                {'full_gen': {'accepted_events': 1, 'sum_weights': 1.},
                 'physical_selected_cross_sections_pb': [{'value_pb': .5}],
                 'epsilon': .001, 'proper_ctau_mm': 0.}))
        paths = SimpleNamespace(base_path=str(self.root), limits_cut_output_path=str(self.root / 'cuts'))
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(sys.modules, {'shift_paths': paths}).start()
        self.config = SimpleNamespace(audited_yields_path=str(self.yields), output_path=str(self.output))
        event = SimpleNamespace(run=1, luminosityBlock=2, event=3, genWeight=1.,
                                nShiftDimuonVertex=1, ShiftDimuonVertex_topologyMin=[2],
                                ShiftDimuonVertex_topologyMax=[2])
        tree = mock.MagicMock()
        tree.__iter__.side_effect = lambda: iter([event])
        tree.GetListOfBranches.return_value = []
        file = mock.Mock()
        file.IsZombie.return_value = False
        file.Get.return_value = tree
        root = SimpleNamespace(gROOT=mock.Mock(), TFile=SimpleNamespace(Open=mock.Mock(return_value=file)))
        with mock.patch.dict(sys.modules, {'ROOT': root}), mock.patch.object(
                gap, 'candidate', side_effect=lambda *args: dict(
                    topologyMin=2., topologyMax=2., both_hit_matched=False)):
            self.collect()

    def collect(self):
        with contextlib.redirect_stdout(io.StringIO()):
            gap.collect_candidates(self.config)

    def assert_rejected_without_writing(self, reason):
        paths = sorted((self.output / 'candidates').glob('*.json'))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        # Reuse and rejection must work without loading ROOT or reading a tree.
        with mock.patch.dict(sys.modules, {'ROOT': None}):
            with self.assertRaisesRegex(ValueError, reason):
                self.collect()
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_unchanged_cache_is_reused_without_root_or_writes(self):
        paths = sorted((self.output / 'candidates').glob('*.json'))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        with mock.patch.dict(sys.modules, {'ROOT': None}):
            self.collect()
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_current_signal_authority_is_checked_before_reuse(self):
        rate = self.parents['m2_prompt'] / 'cross_sections.json'
        value = json.loads(rate.read_text())
        value['epsilon'] = .002
        rate.write_text(json.dumps(value))
        self.assert_rejected_without_writing('authority')

    def test_current_signal_union_is_checked_before_reuse(self):
        union = self.parents['m2_prompt'] / 'metadata/grouping/union.json'
        union.write_text(union.read_text() + '\n')
        self.assert_rejected_without_writing('authority')

    def test_current_yields_authority_is_checked_before_reuse(self):
        self.yields.write_text(self.yields.read_text() + '\n')
        self.assert_rejected_without_writing('authority')

    def test_current_background_selection_authority_is_checked(self):
        np.savez(self.joint, ids=np.array([[1, 2, 4]]), weight=np.array([1.]))
        self.joint.with_suffix('.json').write_text(json.dumps({'sha256': gap.digest(self.joint)}))
        self.assert_rejected_without_writing('authority')

    def test_source_size_and_mtime_are_checked_for_signals_and_backgrounds(self):
        for path in (self.nano, self.parents['m2_prompt'] / 'nanoAOD_merged/0.root'):
            original = path.read_bytes()
            stat = path.stat()
            for change in ('size', 'mtime'):
                with self.subTest(path=path, change=change):
                    if change == 'size':
                        path.write_bytes(original + b' changed')
                    else:
                        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
                    self.assert_rejected_without_writing('source_file_stats')
                    path.write_bytes(original)
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_missing_or_changed_algorithm_and_selection_are_rejected(self):
        path = self.output / 'candidates/m2_prompt.json'
        original = json.loads(path.read_text())
        for key in ('algorithm_sha256', 'selection'):
            for missing in (True, False):
                with self.subTest(key=key, missing=missing):
                    value = dict(original)
                    if missing:
                        value.pop(key)
                    else:
                        value[key] = 'old'
                    path.write_text(json.dumps(value))
                    self.assert_rejected_without_writing(key)
        path.write_text(json.dumps(original))

    def test_background_candidate_ids_weights_and_duplicates_are_checked(self):
        path = self.output / 'candidates/qcd_0to1.json'
        original = path.read_text()
        for change in ('id', 'weight', 'duplicate'):
            with self.subTest(change=change):
                value = json.loads(original)
                if change == 'duplicate':
                    value['candidates'] *= 2
                else:
                    value['candidates'][0][change] = [1, 2, 4] if change == 'id' else 2.
                path.write_text(json.dumps(value))
                self.assert_rejected_without_writing('selection identities/weights')


class DiagnosticTest(unittest.TestCase):
    def test_missing_snapshot_names_required_manual_input_without_root(self):
        with tempfile.TemporaryDirectory(prefix='shift-gap-', dir='/tmp') as directory:
            with mock.patch.dict(sys.modules, {'ROOT': None}):
                with self.assertRaisesRegex(FileNotFoundError, 'stream_snapshot.json.*manually prepared'):
                    analysis.stream_candidates(Path(directory))

    def test_ideal_count_and_efficiency_follow_selected_weighted_histogram(self):
        muons = [dict(pt=1., pz=0., phi=phi, mass=0., pdgId=13) for phi in (0., np.pi)]
        accepted = dict(both_hit_matched=True, same_truth_particle=False,
                        matched_gen_muons=muons, matched_gen_mass=2., mass=2., weight=2.)
        # Matching and distinct-particle cuts remove these rows from the histogram.
        rows = [accepted, dict(accepted, weight=.5), dict(accepted, both_hit_matched=False),
                dict(accepted, same_truth_particle=True)]
        with tempfile.TemporaryDirectory(prefix='shift-gap-', dir='/tmp') as directory:
            output = Path(directory)
            (output / 'candidates').mkdir()
            signal = dict(candidates=rows, full_gen_sumw=20., sigma_pb=1.,
                          kind='signal', authority={'epsilon': .001})
            (output / 'candidates/m2_prompt.json').write_text(json.dumps(signal))
            (output / 'comparison.json').write_text(json.dumps({'records': []}))
            config = SimpleNamespace(output_path=directory, luminosity=30000.)
            with mock.patch.dict(sys.modules, {'shift_limits_book_match_config': config}), \
                    mock.patch.object(checks, 'expected_limit', return_value=(1., 0.)) as limit, \
                    contextlib.redirect_stdout(io.StringIO()):
                checks.main()
            item = json.loads((output / 'comparison.json').read_text())['records'][0]
            self.assertEqual(item['selected_mc'], 2.5)
            self.assertEqual(item['efficiency'], 2.5 / 20.)
            self.assertEqual(item['confidence_levels']['0.9']['selected_events_at_limit'],
                             item['selected_mc'] * config.luminosity / signal['full_gen_sumw'])
            np.testing.assert_array_equal(limit.call_args.args[0],
                                          analysis.histogram(rows, 'ideal_mass') * 30000. / 20.)


if __name__ == '__main__':
    unittest.main()
