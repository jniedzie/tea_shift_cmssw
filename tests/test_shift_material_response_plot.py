# Code style: AI-code. AI-generated response-report scope regression tests.
import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('response_plot',
    Path(__file__).parents[1]/'utils'/'plot_shift_material_response.py')
response_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(response_plot)


def summary():
    target = dict(positive_originals=1, baseline_effective_count=1.,
        statistics_sufficient=False, precision_per_CPU_gain_estimate=2.,
        precision_per_CPU_gain_interval=[1.1, 3.])
    shape = dict(edges=[0., 20., 50.], reference=[1., 2.], candidate=[1., 2.],
                 reference_variance=[1., 4.], candidate_variance=[1., 4.])
    return dict(shapes={name: copy.deepcopy(shape) for name in ('mass', 'pt')},
        all_bins=dict(originals=13, bootstrap={'confidence': .95},
                      targets={'mass_gt20': target}, tail_improvement_proven=True))


@unittest.skipUnless(shutil.which('pdftotext'), 'PDF text readback needs pdftotext')
class MaterialResponsePlotTest(unittest.TestCase):
    def render(self, report):
        with tempfile.TemporaryDirectory(prefix='shift-response-plot-test-') as directory:
            path = Path(directory)/'report.pdf'
            response_plot.plot(report, path)
            self.assertGreater(path.stat().st_size, 1000)
            return subprocess.check_output(['pdftotext', str(path), '-'], text=True)

    def test_mass_mechanism_plot_cannot_claim_a_gain_even_when_metrics_are_positive(self):
        report = summary()
        report.update(mechanism_study=True, reference_label='GEN-mass-selected originals',
            sampling_design='All thirteen OS GEN mass>20 originals in forty sources.',
            source_population_scope='GEN-mass-selected physical mechanism test.',
            reconstructed_tail_rate_scope='Cannot estimate inclusive reconstructed-tail rates; excluded GEN originals can migrate.')
        text = self.render(report)
        self.assertIn('GEN-mass-selected originals', text)
        self.assertIn('forty sources', text)
        self.assertIn('Cannot estimate inclusive reconstructed-tail rates', text)
        self.assertNotIn('Conditional gain supported', text)
        self.assertNotIn('Random reconstruction reference', text)
        self.assertNotIn('gain 2', text)

    def test_empty_tail_shapes_are_labelled_instead_of_shown_on_arbitrary_log_scale(self):
        report = summary()
        for shape in report['shapes'].values():
            for key in ('reference', 'candidate', 'reference_variance', 'candidate_variance'):
                shape[key] = [0., 0.]
        text = self.render(report)
        self.assertEqual(text.count('No reconstructed OS Both-Both pairs'), 2)

    def test_tail_control_design_and_old_summary_remain_readable(self):
        report = summary()
        report.update(sampling_design='GEN tail certainty + independent 1% controls.',
                      reference_label='Declared reconstruction sample, weighted')
        text = self.render(report)
        self.assertIn('independent 1% controls', text)
        self.assertIn('Declared reconstruction sample, weighted', text)
        self.assertNotIn('Random reconstruction reference', text)
        self.assertIn('Reconstruction reference, weighted', self.render(summary()))


if __name__ == '__main__':
    unittest.main()
