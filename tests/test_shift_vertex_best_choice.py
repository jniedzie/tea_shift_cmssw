# Code style: AI-code. AI-generated independent choice gates and lock guards.
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from shift_vertex_best_choice import gate_failures,evaluate,evaluate_signal,selected_signal_yield
from shift_vertex_complete_diagnostics import BOUNDS


class ChoiceTests(unittest.TestCase):
    def metrics(self):
        profile={axis:{'bins':[{'efficiency':.9,'supported':True}]} for axis in ('mass','vertex_z','vertex_radius')}
        return dict(flatness=profile,per_process_flatness={},maximum_marginal_efficiency_span=.05,
            process_efficiency_span=.01,minimum_supported_category_efficiency=.86,category_efficiency_span=.07,
            dependence={label:{'joint_dcor':.1}for label in ('common_vertex','accidental')})

    def test_proper_length_is_an_additional_gate_and_undefined_fails(self):
        life=dict(minimum_supported_retention=.87,maximum_supported_span=.07,score_length_dcor={'0':.12,'1':.15})
        self.assertEqual(gate_failures(self.metrics(),life,BOUNDS),[])
        life['maximum_supported_span']=.2
        self.assertIn('reco_length_span',gate_failures(self.metrics(),life,BOUNDS))
        life['score_length_dcor']['1']=None
        self.assertIn('reco_length_dcor',gate_failures(self.metrics(),life,BOUNDS))

    def test_no_signal_cache_or_pickle_access_before_selection_lock(self):
        with tempfile.TemporaryDirectory()as temporary:
            with patch('shift_vertex_best_choice.load',side_effect=AssertionError('Cache read before lock')):
                with patch('shift_vertex_best_choice.pickle.load',side_effect=AssertionError('Model read before lock')):
                    with self.assertRaises(FileNotFoundError):evaluate(Path(temporary),Path(temporary)/'signal')

    def test_extended_signal_cannot_run_without_a_locked_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('shift_vertex_best_choice.load', side_effect=AssertionError('Cache opened before lock')):
                with patch('shift_vertex_best_choice.pickle.load', side_effect=AssertionError('Model opened before lock')):
                    with self.assertRaises(FileNotFoundError):
                        evaluate_signal(root / 'choice', root / 'cache', root / 'response')
            self.assertFalse((root / 'response').exists())

    def test_extended_signal_preserves_existing_response_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            response = root / 'response'
            response.mkdir()
            evidence = response / 'evaluation.json'
            evidence.write_text('frozen evidence\n')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                evaluate_signal(root / 'choice', root / 'cache', response)
            self.assertEqual(evidence.read_text(), 'frozen evidence\n')


class YieldTests(unittest.TestCase):
    def test_full_parent_efficiency_and_inverse_fb_units(self):
        response = {'weighted': dict(numerator_sumw=130., denominator_sumw=1000.,
                                     efficiency=.13, standard_error=.01)}
        result = selected_signal_yield(response, .0028088487087559208, 3.)
        self.assertAlmostEqual(result['expected_events'], 1.0954509964148091)
        self.assertAlmostEqual(result['mc_standard_error_events'], .08426546126267763)

    def test_native_weight_ratio(self):
        response = {'weighted': dict(numerator_sumw=4., denominator_sumw=20.,
                                     efficiency=.2, standard_error=.05)}
        self.assertEqual(selected_signal_yield(response, 2., 3.)['expected_events'], 1200.)

    def test_zero_yield_and_invalid_denominator(self):
        response = {'weighted': dict(numerator_sumw=0., denominator_sumw=100.,
                                     efficiency=0., standard_error=0.)}
        self.assertEqual(selected_signal_yield(response, 1., 3.)['expected_events'], 0.)
        response['weighted']['denominator_sumw'] = 0.
        with self.assertRaises(ValueError): selected_signal_yield(response, 1., 3.)

    def test_refuses_changed_efficiency(self):
        response = {'weighted': dict(numerator_sumw=10., denominator_sumw=100.,
                                     efficiency=.9, standard_error=.01)}
        with self.assertRaises(ValueError): selected_signal_yield(response, 1., 3.)


if __name__=='__main__':unittest.main()
