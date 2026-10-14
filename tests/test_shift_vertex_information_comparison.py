# Code style: AI-code. AI-generated validation comparison integrity tests.
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'utils/shift_vertex_information_comparison.py'
SPEC = importlib.util.spec_from_file_location('information_comparison', SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture():
    families = ['global_raw50', 'global_rich', 'shared_raw50', 'shared_rich',
                'heads_raw50', 'heads_rich', 'heads_rich_weak', 'heads_rich_strong',
                'shared_rich_strong', 'global_muon', 'shared_rich_accept10', 'shared_rich_accept30']
    recipes = [dict(name=f'{name}_{seed}', family=name, seed=seed) for name in families for seed in (71, 72)]
    bounds = dict(minimum_supported_genuine_retention=.8, maximum_supported_marginal_span=.15,
                  maximum_process_efficiency_span=.05, maximum_class_joint_dcor=.3,
                  maximum_category_efficiency_span=.15)
    val = dict(flatness={v: {'bins': [{'efficiency': .9, 'supported': True}]} for v in ('mass', 'vertex_z', 'vertex_radius')},
               per_process_flatness={}, maximum_marginal_efficiency_span=.1, process_efficiency_span=.02,
               dependence={label: {'joint_dcor': .1} for label in ('common_vertex', 'accidental')},
               minimum_supported_category_efficiency=.85, category_efficiency_span=.08,
               rejection=.4, auc=.7, common_vertex={'efficiency': .9}, threshold=.3)
    results = {r['name']: dict(recipe=r, validation=copy.deepcopy(val)) for r in recipes}
    return dict(recipes=recipes, acceptance_envelope=bounds), results


class ComparisonTests(unittest.TestCase):
    def test_all_fits_and_both_seeds_required(self):
        protocol, results = fixture()
        results.pop(next(iter(results)))
        with self.assertRaisesRegex(ValueError, 'All24'): M.aggregate(protocol, results)

    def test_worst_seed_ranking_and_hard_gate(self):
        protocol, results = fixture()
        for seed in (71, 72): results[f'global_rich_{seed}']['validation']['rejection'] = .6
        results['heads_rich_71']['validation']['rejection'] = .9
        results['heads_rich_72']['validation']['rejection'] = .59
        self.assertEqual(M.aggregate(protocol, results)['robust_family'], 'global_rich')
        results['global_rich_72']['validation']['category_efficiency_span'] = .151
        report = M.aggregate(protocol, results)
        self.assertEqual(report['robust_family'], 'heads_rich')
        self.assertEqual(report['families']['global_rich']['seeds'][72]['failed_gates'], ['category_span'])

    def test_contrast_units_and_no_unsupported_gate_pass(self):
        protocol, results = fixture()
        results['global_rich_71']['validation']['rejection'] = .53
        report = M.aggregate(protocol, results)
        self.assertAlmostEqual(report['paired_seed_contrasts'][0]['rejection_change_pp'], 13.)
        entry = results['global_rich_71']['validation']
        entry['minimum_supported_category_efficiency'] = None
        self.assertIn('minimum_category', M.failures(entry, protocol['acceptance_envelope']))

    def test_pdf_render(self):
        protocol, results = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); M.render(M.aggregate(protocol, results), path)
            self.assertEqual(len(list(path.glob('*.pdf'))), 2)
            for pdf in path.glob('*.pdf'): self.assertTrue(pdf.read_bytes().startswith(b'%PDF'))


if __name__ == '__main__': unittest.main()
