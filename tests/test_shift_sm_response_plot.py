# Code style: AI-code. AI-generated checks of stratified response plot claims.
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('sm_response_plot',
    Path(__file__).parents[1]/'utils/plot_shift_sm_response.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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


if __name__ == '__main__':
    unittest.main()
