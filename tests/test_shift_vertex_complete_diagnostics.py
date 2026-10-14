# Code style: AI-code. AI-generated independent learning-curve integrity tests.
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
import shift_vertex_complete_diagnostics as diagnostic


def fixture():
    recipes, results = [], {}
    for index in range(9):
        family = ('bdt' if index < 3 else 'dnn') + str(index)
        for seed in diagnostic.SEEDS:
            for fraction in diagnostic.FRACTIONS:
                name = f'{family}_{seed}_{fraction}'
                recipe = dict(name=name, family=family, algorithm='BDT' if index < 3 else 'DNN',
                              seed=seed, fraction=fraction, config={'capacity': index})
                recipes.append(recipe)
                rejection = .4 + .01 * index
                if index in (0, 3) and fraction < 1:
                    rejection = .8
                profile = {axis: dict(bins=[dict(supported=True, efficiency=.9)])
                           for axis in ('mass', 'vertex_z', 'vertex_radius')}
                validation = dict(auc=.8, rejection=rejection, threshold=.5,
                    common_vertex=dict(n=100, passed=90, efficiency=.9),
                    accidental=dict(n=100, passed=100 * (1-rejection), efficiency=1-rejection),
                    flatness=profile, per_process_flatness={},
                    dependence={label: dict(joint_dcor=.1) for label in ('common_vertex', 'accidental')},
                    maximum_marginal_efficiency_span=.1, process_efficiency_span=.02,
                    minimum_supported_category_efficiency=.85, category_efficiency_span=.1,
                    categories={})
                counts = dict(pairs=int(400 * fraction), genuine=int(300 * fraction), fake=int(100 * fraction))
                results[name] = dict(recipe=recipe, training_counts=counts, validation=validation,
                    training_at_validation_cut=dict(auc=.85, genuine_retention=.92, accidental_rejection=.85),
                    optimizer=dict(epochs_completed=10, best_epoch=8, updates_per_epoch=2,
                                   total_updates=20, cap_reached=False, stopped_early=True),
                    fit_seconds=1., model=dict(path='/unopened/nonexistent.pkl', sha256='model'),
                    history=dict(path='/unopened/nonexistent_history.json', sha256='history'))
    return dict(recipes=recipes, acceptance_envelope=diagnostic.BOUNDS), results


class CompleteDiagnosticsTests(unittest.TestCase):
    def test_select_only_full_fraction_and_preserve_all_seed_curves(self):
        protocol, results = fixture()
        report = diagnostic.aggregate(protocol, results)
        self.assertEqual(report['winners']['bdt']['rejection'], 'bdt2')
        self.assertEqual(report['winners']['dnn']['rejection'], 'dnn8')
        self.assertEqual(len(report['families']['bdt0']['fits']), 9)
        self.assertGreater(report['families']['bdt0']['fits'][0]['rejection'],
                           report['families']['bdt2']['full_training']['minimum_rejection'])

    def test_all_seeds_must_pass_and_missing_dependence_is_unknown(self):
        protocol, results = fixture()
        results['bdt2_73_1.0']['validation']['dependence']['accidental']['joint_dcor'] = None
        report = diagnostic.aggregate(protocol, results)
        self.assertEqual(report['winners']['bdt']['rejection'], 'bdt2')
        self.assertEqual(report['winners']['bdt']['acceptance'], 'bdt1')
        for name, result in results.items():
            if result['recipe']['algorithm'] == 'DNN':
                result['validation']['category_efficiency_span'] = .2
        self.assertIsNone(diagnostic.aggregate(protocol, results)['winners']['dnn']['acceptance'])

    def test_incomplete_changed_duplicate_and_inconsistent_subsets_rejected(self):
        protocol, results = fixture()
        missing = dict(results)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, '81'):
            diagnostic.aggregate(protocol, missing)
        changed = copy.deepcopy(results)
        changed[next(iter(changed))]['recipe']['config']['capacity'] = 999
        with self.assertRaisesRegex(ValueError, 'recipe differs'):
            diagnostic.aggregate(protocol, changed)
        changed = copy.deepcopy(results)
        changed[next(iter(changed))]['training_counts'].update(pairs=99, genuine=74)
        with self.assertRaisesRegex(ValueError, 'subset counts'):
            diagnostic.aggregate(protocol, changed)
        changed_protocol, changed = copy.deepcopy((protocol, results))
        changed_protocol['recipes'][0]['fraction'] = .5
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            diagnostic.aggregate(changed_protocol, changed)

    def test_recipe_cannot_change_with_training_size(self):
        protocol, results = fixture()
        protocol['recipes'][0]['config']['capacity'] = 99
        with self.assertRaisesRegex(ValueError, 'Family changes'):
            diagnostic.aggregate(protocol, results)

    def test_json_provenance_and_lock_fail_closed_without_payload_reads(self):
        protocol, results = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'fits').mkdir()
            diagnostic.write(root / 'protocol.json', protocol)
            ref = diagnostic.reference(root / 'protocol.json')
            for name, result in results.items():
                result['protocol'] = ref
                diagnostic.write(root / 'fits' / (name + '.json'), result)
            report = diagnostic.build_report(root)
            self.assertTrue(report['no_model_or_test_data_read'])
            lock = dict(protocol=ref, winners=copy.deepcopy(report['winners']), selection_uses_test=False,
                        validation_fits=report['fit_references'],
                        thresholds={name: result['validation']['threshold'] for name, result in results.items()})
            diagnostic.write(root / 'selection.json', lock)
            diagnostic.build_report(root, root / 'selection.json')
            lock['winners']['bdt']['rejection'] = 'bdt0'
            (root / 'selection.json').write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                diagnostic.build_report(root, root / 'selection.json')
            lock['winners'] = report['winners']
            lock['thresholds'][next(iter(results))] = .6
            (root / 'selection.json').write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'thresholds'):
                diagnostic.build_report(root, root / 'selection.json')
            (root / 'protocol.json').write_text(json.dumps(protocol) + '\n')
            with self.assertRaisesRegex(ValueError, 'another protocol'):
                diagnostic.build_report(root)
            with self.assertRaisesRegex(ValueError, 'Changed JSON'):
                diagnostic.read_reference(ref)

    def test_four_page_pdf_is_rendered_from_json_only(self):
        from pypdf import PdfReader
        protocol, results = fixture()
        report = diagnostic.aggregate(protocol, results)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            diagnostic.render(report, output)
            pdf = PdfReader(output / 'validation_learning_curves.pdf')
            self.assertEqual(len(pdf.pages), 4)
            self.assertIn('Accidental rejection', pdf.pages[0].extract_text())
            self.assertIn('Training', pdf.pages[3].extract_text())


if __name__ == '__main__':
    unittest.main()
