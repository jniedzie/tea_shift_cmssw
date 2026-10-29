# Code style: AI-code. AI-generated checks of stratified response plot claims.
import importlib.util
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('sm_response_plot',
    Path(__file__).parents[1]/'utils/plot_shift_sm_response.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResponseProfileTest(unittest.TestCase):
    def summary(self):
        groups = []
        for split, count, passed in [('development', 100, 90), ('validation', 20, 2)]:
            stats = dict(events=count, positive_originals=passed, pairs=passed)
            groups.append(dict(stratum='qcd_20toinf', study_split=split,
                totals={c: dict(stats) for c in module.CATEGORIES},
                efficiency_maps={f: dict(edges=[0., 10.], cells={
                    '0': {c: dict(stats) for c in module.CATEGORIES}})
                    for f in module.FEATURE_LABELS}))
        return dict(schema='shift-sm-measured-response-summary-v1', events=120,
                    original_event_is_error_unit=True, groups=groups)

    def test_validation_profiles_do_not_include_development_successes(self):
        for feature in module.FEATURE_LABELS:
            edges, pooled = module.pool_response(self.summary(), feature, 'validation')
            self.assertEqual(pooled['qcd']['events'], 20)
            self.assertEqual(pooled['qcd']['categories']['both_both_good']['positive'], 2)

    def test_missing_feature_population_cannot_disappear_from_a_map(self):
        summary = self.summary()
        summary['groups'][1]['efficiency_maps']['best_pair_miss_cm']['cells'] = {}
        with self.assertRaisesRegex(ValueError, 'cover group totals'):
            module.pool_response(summary, 'best_pair_miss_cm', 'validation')

    @unittest.skipUnless(shutil.which('pdftotext'), 'Requires PDF text readback')
    def test_energy_render_uses_requested_axis_and_split(self):
        with tempfile.TemporaryDirectory(prefix='sm-response-profile-') as directory:
            path = Path(directory)/'energy.pdf'
            module.plot(self.summary(), path, 'best_pair_min_energy_GeV', 'validation')
            text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
        self.assertIn('20 originals', text)
        self.assertIn('validation only', text)
        self.assertIn('Lower muon energy', text)
        self.assertNotIn('Smallest known incoming', text)


@unittest.skipUnless(shutil.which('pdftotext'), 'Requires PDF text readback')
class StratifiedPlotTest(unittest.TestCase):
    def summary(self):
        report = dict(observed_events=12, population_events=400,
            forecast_scope='existing selected census',
            cells=[dict(region='credible_shallow_pointing', sampled_events=12,
                targets=dict(inclusive=dict(pairs=4),
                    both_both=dict(pairs=2, positive_originals=2),
                    both_both_good=dict(pairs=1, positive_originals=1)))],
            planning_forecasts=[dict(stratum='qcd_20toinf', control_floor=floor,
                expected_admission_fraction=.15) for floor in (.1, .01, .001)])
        return dict(stratified_benchmark=report,
            groups=[dict(tails=dict(both_both=dict(
                mass_gt_20=dict(positive_originals=1),
                pt_gt_10=dict(positive_originals=2))))])
    def render(self, summary):
        with tempfile.TemporaryDirectory(prefix='sm-response-plot-test-') as directory:
            path = Path(directory)/'response.pdf'
            receipt = module.plot_benchmark(summary, path)
            text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
        return receipt, text

    def test_counts_are_derived_and_planning_is_not_presented_as_physics(self):
        receipt, text = self.render(self.summary())
        self.assertIn('Reconstructed pairs: 4; Both-Both: 2; Both-Both + Good: 1', text)
        self.assertIn('mass > 20 GeV: 1; pair pT > 10 GeV: 2', text)
        self.assertIn('Raw counts are not population efficiencies', text)
        self.assertIn('production default is unchanged', text)
        self.assertFalse(receipt['production_policy_qualified'])

    def test_completed_larger_comparison_replaces_the_old_outlier_description(self):
        summary = self.summary()
        summary['decay_pilot'] = dict(
            arms={arm: dict(events=78, categories=dict(both_both_good=dict(
                mass_gt20=dict(raw_pairs=0, effective_originals=0.))))
                for arm in ['ordinary', 'mixture']},
            muons=dict(ordinary=0, mixture=34),
            positive_pairs=[dict(mass_GeV=17.3, final_weight=2.3e-17)],
            comparison_description='78 originals drawn from 572 saved events; 156 completed responses',
            notes=['Three good pairs; no observed target tails.',
                   'Zero observed tails do not establish zero rate.'])
        receipt, text = self.render(summary)
        self.assertIn('156 completed responses', text)
        self.assertIn('no observed target tails', text)
        self.assertNotIn('136 GeV entry', text)
        self.assertFalse(receipt['production_policy_qualified'])
        self.assertFalse(receipt['physical_yield_estimate'])

    def test_parent_opportunity_page_is_not_a_daughter_yield_claim(self):
        summary = self.summary()
        summary['reserve_opportunities'] = dict(cohorts=[dict(stratum='qcd_20toinf',
            original_events=400, strict_parent_tail_events=2, unclassified_original_pions=1)],
            largest_strict_parent_pair_mass_per_original_GeV=[25., 30.])
        receipt, text = self.render(summary)
        self.assertEqual(text.count('\f'), 2)
        self.assertIn('parent mass differs from dimuon mass', text)
        self.assertIn('repeated draws of one original are not independent events', text)
        self.assertIn('these counts are not a physical event rate', text)
        self.assertTrue(receipt['reserve_opportunities_included'])
        self.assertFalse(receipt['physical_yield_estimate'])

    def test_decay_result_reports_weighted_limit_and_bad_momentum(self):
        summary = self.summary()
        summary['decay_pilot'] = dict(
            arms={arm: dict(events=12, categories=dict(both_both_good=dict(
                mass_gt20=dict(raw_pairs=count, effective_originals=effective))))
                for arm, count, effective in [('ordinary', 0, 0.), ('mixture', 2, 1.002)]},
            muons=dict(ordinary=0, mixture=10),
            positive_pairs=[dict(mass_GeV=20.73, final_weight=7.16e-20),
                            dict(mass_GeV=136.13, final_weight=8.49e-23)])
        receipt, text = self.render(summary)
        self.assertIn('1.002', text)
        self.assertIn('inflated reconstructed momentum', text)
        self.assertIn('No orders-of-magnitude gain is established', text)
        self.assertIn('Zero ordinary pairs do not establish zero rate', text)
        self.assertTrue(receipt['decay_pilot_included'])
        self.assertFalse(receipt['production_policy_qualified'])


class CandidateComparisonTest(unittest.TestCase):
    def statistics(self, count, passed):
        return dict(events=count, positive_originals=passed, pairs=passed,
                    effective_pair_contributions=float(passed))

    def summary(self):
        groups = []
        for name, count, muons, passed in [('candidate', 10, 4, 2), ('controls', 20, 1, 0)]:
            stats = self.statistics(count, passed)
            groups.append(dict(region_group=name, rows=count, muon_events=muons,
                targets={category: dict(stats) for category in ('inclusive', *module.CATEGORIES)},
                tails={category: {tail: self.statistics(count, 0) for tail in
                    ('mass_gt_20', 'mass_gt_50', 'pt_gt_10', 'pt_gt_20')}
                    for category in module.CATEGORIES}))
        histograms = []
        for group in ('candidate', 'controls'):
            for category in module.CATEGORIES:
                for field in ('mass', 'pt'):
                    histograms.append(dict(stratum='qcd_20toinf', region_group=group,
                        category=category, field=field, edges=[0., 10., 20., 50., 100.],
                        raw_counts=[0, 2 if group == 'candidate' else 0, 0, 0, 0, 0]))
        return dict(schema='shift-sm-candidate-response-analysis-v1',
            scope=dict(complete_sources=5, total_sources=385, complete_events=30,
                       completion_conditioned=True), pooled_groups=groups,
            groups=[dict(copy.deepcopy(group), stratum='qcd_20toinf', total_sim_seconds=0.)
                    for group in groups], histograms=histograms)

    def test_incomplete_response_cannot_be_presented_as_completed(self):
        report = self.summary()
        report['scope']['complete_events'] += 1
        with self.assertRaisesRegex(ValueError, 'do not cover completed'):
            module.candidate_comparison_input(report)
        report = self.summary()
        report['pooled_groups'][0]['targets']['both_both']['positive_originals'] = 11
        with self.assertRaisesRegex(ValueError, 'do not describe'):
            module.candidate_comparison_input(report)
        report = self.summary()
        report['groups'][0]['rows'] -= 1
        with self.assertRaisesRegex(ValueError, 'Generator-bin counts'):
            module.candidate_comparison_input(report)

    def test_flow_bins_survive_pooling_and_different_axes_fail(self):
        report = self.summary()
        additional = copy.deepcopy(report['histograms'][0])
        additional.update(stratum='qcd_10to20', raw_counts=[1, 3, 0, 0, 0, 4])
        report['histograms'].append(additional)
        key = ('candidate', 'both_both', 'mass')
        pooled = module.pooled_candidate_histograms(report)
        self.assertEqual(pooled[key]['raw_counts'], [1, 5, 0, 0, 0, 4])
        report['histograms'][-1]['edges'][1] = 12.
        with self.assertRaisesRegex(ValueError, 'different axes'):
            module.pooled_candidate_histograms(report)

    def test_kinematic_entries_cannot_disappear_from_the_report(self):
        report = self.summary()
        report['histograms'][0]['raw_counts'][1] = 1
        with tempfile.TemporaryDirectory(prefix='sm-missing-pair-') as directory:
            with self.assertRaisesRegex(ValueError, 'do not cover retained pairs'):
                module.plot_candidate_comparison(report, Path(directory)/'response.pdf')

    @unittest.skipUnless(shutil.which('pdftotext'), 'Requires PDF text readback')
    def test_partial_pairs_are_visible_but_do_not_enter_efficiency_counts(self):
        report = self.summary()
        report['partial_groups'] = [dict(stratum='all_qcd', region_group='candidate', rows=2,
            muon_events=2, targets=dict(inclusive=dict(positive_originals=2), both_both=dict(pairs=2)))]
        report['partial_positive_originals'] = [dict(source_id='supply_qcd_20toinf_00032',
            gen_sha256='a'*64, event_id=[1, 1, number], region_group='candidate',
            rates_and_intervals_withheld=True, pair_counts=dict(both_both=1),
            retained_pairs=[dict(topologyMin=2, topologyMax=2, isOS=1, dcaValid=1,
                                 mass=mass, pt=1.2, dca=distance)])
            for number, mass, distance in [(576, 3.119, 219.36), (61, .971, 96.05)]]
        pairs = module.partial_both_both_pairs(report)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(any(row['good'] for row in pairs))
        with tempfile.TemporaryDirectory(prefix='sm-partial-pairs-') as directory:
            path = Path(directory)/'response.pdf'
            receipt = module.plot_candidate_comparison(report, path)
            text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
        self.assertEqual(receipt['pages'], 4)
        self.assertEqual(receipt['complete_events'], 30)
        self.assertIn('First Both-Both pairs from partial source cohorts', text)
        self.assertIn('Good DCA limit: 50 cm', text)
        self.assertIn('Good pairs in these observations: 0/2', text)
        self.assertIn('excluded from the efficiency plots', text)
        report['partial_positive_originals'].pop()
        with self.assertRaisesRegex(ValueError, 'does not cover the raw observed'):
            module.partial_both_both_pairs(report)

    @unittest.skipUnless(shutil.which('pdftotext'), 'Requires PDF text readback')
    def test_render_keeps_raw_and_weighted_claims_separate(self):
        with tempfile.TemporaryDirectory(prefix='sm-candidate-response-') as directory:
            path = Path(directory)/'response.pdf'
            receipt = module.plot_candidate_comparison(self.summary(), path)
            text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
        self.assertEqual(text.count('\f'), 3)
        self.assertIn('30 originals from 5 complete source cohorts / 385 planned cohorts', text)
        self.assertIn('not population efficiencies', text)
        self.assertIn('Zero observed tails do not establish zero physical rate', text)
        self.assertIn('rate normalization remains unqualified', text)
        self.assertFalse(receipt['physical_yield_estimate'])
        self.assertFalse(receipt['production_policy_qualified'])

    @unittest.skipUnless(shutil.which('pdftotext'), 'Requires PDF text readback')
    def test_empty_completed_snapshot_does_not_invent_an_efficiency(self):
        report = self.summary()
        report['scope'].update(complete_sources=0, complete_events=0)
        for group in report['pooled_groups']+report['groups']:
            group.update(rows=0, muon_events=0)
            group['targets'] = {category: self.statistics(0, 0) for category in
                                ('inclusive', *module.CATEGORIES)}
            group['tails'] = {category: {tail: self.statistics(0, 0) for tail in
                ('mass_gt_20', 'mass_gt_50', 'pt_gt_10', 'pt_gt_20')} for category in module.CATEGORIES}
        report['histograms'] = []
        with tempfile.TemporaryDirectory(prefix='sm-empty-response-') as directory:
            path = Path(directory)/'response.pdf'
            module.plot_candidate_comparison(report, path)
            text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
        self.assertIn('No complete current source draws yet', text)
        self.assertIn('No reconstruction efficiency is estimated', text)


if __name__ == '__main__':
    unittest.main()
