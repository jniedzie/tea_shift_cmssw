# Code style: AI-code. AI-generated locked-development plotting tests.
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import shift_vertex_complete_development as diagnostic


def synthetic():
    recipes = []
    for index in range(9):
        algorithm = 'bdt' if index < 3 else 'dnn'
        for fraction in (.25, .5, 1.):
            for seed in (71, 72, 73):
                recipes.append(dict(name=f'{algorithm}{index}_f{fraction}_s{seed}', family=f'{algorithm}{index}',
                                    algorithm=algorithm, fraction=fraction, seed=seed))
    chosen = [spec for spec in recipes if spec['family'] in ('bdt0', 'dnn3')]
    table = dict(ids=np.asarray([[1, 3, 2**53 + i] for i in range(12)], dtype=np.uint64),
        process=np.asarray(['dy'] * 4 + ['jpsi'] * 6 + ['qcd'] * 2),
        category=np.asarray([diagnostic.CATEGORIES[i % 5] for i in range(12)]),
        y=np.asarray([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8),
        nuisance=np.column_stack((np.arange(12), np.arange(12) * 10, np.arange(12) * 2)),
        evaluation_weight=np.ones(12))
    metrics = {}
    edges = {'mass': [4., 8.], 'vertex_z': [40., 80.], 'vertex_radius': [8., 16.]}
    for spec in chosen:
        score = np.asarray([.8, .5, .6, .1, .9, .4, .7, .2, .3, .8, .1, .4])
        table[spec['name']] = score
        passed = score >= .5
        row = dict(threshold=.5, auc=float(roc_auc_score(table['y'], score)),
            common_vertex=diagnostic.empirical(table['y'] == 1, passed),
            accidental=diagnostic.empirical(table['y'] == 0, passed), per_process_flatness={},
            per_process={}, categories={}, flatness={})
        row['rejection'] = 1-row['accidental']['efficiency']
        for process in ('dy', 'jpsi', 'qcd'):
            mask = table['process'] == process
            row['per_process'][process] = {key: diagnostic.empirical(mask & (table['y'] == label), passed)
                                          for label, key in ((1, 'common_vertex'), (0, 'accidental'))}
        for category in set(table['category']):
            row['categories'][category] = dict(common_vertex=diagnostic.empirical(
                (table['category'] == category) & (table['y'] == 1), passed, minimum=50))
        for column, axis in enumerate(diagnostic.AXES):
            assignments = np.searchsorted(edges[axis], table['nuisance'][:, column], side='right')
            row['flatness'][axis] = dict(interior_edges=edges[axis], bins=[
                diagnostic.empirical((table['y'] == 1) & (assignments == i), passed) for i in range(3)])
        metrics[spec['name']] = row
    return recipes, chosen, table, metrics


def write_study(root):
    recipes, chosen, table, metrics = synthetic()
    (root / 'fits').mkdir()
    protocol = dict(recipes=recipes, domain='historical_proxy')
    diagnostic.write(root / 'protocol.json', protocol)
    protocol_ref = diagnostic.reference(root / 'protocol.json')
    refs = {}
    for spec in recipes:
        fit = dict(recipe=spec, protocol=protocol_ref, validation=dict(threshold=.5),
                   training_counts=dict(pairs=int(100 * spec['fraction'])))
        diagnostic.write(root / 'fits' / (spec['name'] + '.json'), fit)
        refs[spec['name']] = diagnostic.reference(root / 'fits' / (spec['name'] + '.json'))
    lock = dict(protocol=protocol_ref, selection_uses_test=False, domain='historical_proxy',
        winners=dict(bdt=dict(rejection='bdt0', acceptance=None), dnn=dict(rejection='dnn3', acceptance=None)),
        validation_fits=refs, thresholds={spec['name']: .5 for spec in recipes})
    diagnostic.write(root / 'selection_locked.json', lock)
    selection_ref = diagnostic.reference(root / 'selection_locked.json')
    diagnostic.write(root / 'development_metrics.json', dict(metrics=metrics, selection=selection_ref,
                      selection_uses_test=False, domain='historical_proxy'))
    np.savez_compressed(root / 'development_scores.npz', **table)
    diagnostic.write(root / 'completion.json', dict(complete=True, fits=81, domain='historical_proxy',
        protocol=protocol_ref, selection=selection_ref, scores=diagnostic.reference(root / 'development_scores.npz'),
        metrics=diagnostic.reference(root / 'development_metrics.json')))


class CompleteDevelopmentTests(unittest.TestCase):
    def test_no_score_access_before_completion(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(diagnostic.np, 'load', side_effect=AssertionError('Early score read')):
            with self.assertRaises(FileNotFoundError):
                diagnostic.load_completed(Path(temporary))

    def test_completed_lock_and_absent_genuine_qcd_remain_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_study(root)
            report, table = diagnostic.load_completed(root)
            self.assertEqual(len(report['models']), 18)
            self.assertIsNone(report['winners']['bdt']['acceptance'])
            self.assertEqual(report['class_counts']['qcd']['genuine'], 0)
            self.assertEqual(report['class_counts']['pooled']['accidental'], 6)
            for row in report['models'].values():
                self.assertFalse(row['populations']['qcd']['roc_defined'])
                self.assertIsNone(row['populations']['qcd']['genuine']['efficiency'])
                self.assertEqual(row['threshold'], .5)
            self.assertEqual(table['ids'].dtype, np.uint64)

    def test_tampered_score_artifact_fails_before_array_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_study(root)
            with (root / 'development_scores.npz').open('ab') as stream:
                stream.write(b'changed')
            with patch.object(diagnostic.np, 'load', side_effect=AssertionError('Unverified array read')):
                with self.assertRaisesRegex(ValueError, 'Completion input changed'):
                    diagnostic.load_completed(root)

    def test_fixed_cuts_counts_and_report_bins_are_reconciled(self):
        _, specs, table, metrics = synthetic()
        cuts = {spec['name']: .5 for spec in specs}
        name = specs[0]['name']
        changed = copy.deepcopy(metrics)
        changed[name]['threshold'] = .6
        with self.assertRaisesRegex(ValueError, 'threshold differs'):
            diagnostic.validate_table(table, specs, changed, cuts)
        changed = copy.deepcopy(metrics)
        changed[name]['common_vertex']['passed'] -= 1
        with self.assertRaisesRegex(ValueError, 'count differs'):
            diagnostic.validate_table(table, specs, changed, cuts)
        changed = copy.deepcopy(metrics)
        changed[specs[-1]['name']]['flatness']['mass']['interior_edges'] = [4., 9.]
        with self.assertRaisesRegex(ValueError, 'Reporting bins changed'):
            diagnostic.validate_table(table, specs, changed, cuts)

    def test_rounded_ids_and_unlocked_score_columns_are_rejected(self):
        _, specs, table, metrics = synthetic()
        cuts = {spec['name']: .5 for spec in specs}
        changed = dict(table, ids=table['ids'].astype(float))
        with self.assertRaisesRegex(ValueError, 'uint64'):
            diagnostic.validate_table(changed, specs, metrics, cuts)
        changed = dict(table, unselected_candidate=np.zeros(12))
        with self.assertRaisesRegex(ValueError, 'Score columns'):
            diagnostic.validate_table(changed, specs, metrics, cuts)

    def test_pdf_pages_use_saved_cuts_without_rank_or_model_access(self):
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_study(root)
            report, table = diagnostic.load_completed(root)
            out = root / 'plots'
            out.mkdir()
            diagnostic.render(report, table, out)
            expected = {'development_roc.pdf': 3, 'development_truth_scores.pdf': 2,
                        'development_genuine_retention.pdf': 3, 'development_learning_curves.pdf': 2}
            self.assertEqual({path.name for path in out.iterdir()}, set(expected))
            for name, pages in expected.items():
                self.assertEqual(len(PdfReader(out / name).pages), pages)
            text = PdfReader(out / 'development_roc.pdf').pages[0].extract_text()
            self.assertIn('ROC undefined', text)
            self.assertIn('Historical proxy', text)


if __name__ == '__main__':
    unittest.main()
