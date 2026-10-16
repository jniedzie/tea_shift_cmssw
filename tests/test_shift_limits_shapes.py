# Code style: AI-code. AI-generated physical mass-binning accounting checks.
import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from shift_limits_shapes import merge_background_bins, rebin, weighted_shape, clipped_boundary_segments, boundary_outline_segments, draw_mass_improvement
from shift_limits import digest, reference_cross_section
from shift_limits_cuts import cut_mask


class MassShapesTest(unittest.TestCase):
    def test_sparse_mass_tail_is_merged_without_losing_events(self):
        values = [1.,8.,10.,0.,0.,1.,0.]
        boundaries = merge_background_bins(values,values,10.)
        bins = rebin(values,boundaries)
        self.assertEqual(boundaries[0],0)
        self.assertEqual(boundaries[-1],len(values))
        self.assertAlmostEqual(sum(bins),sum(values))
        self.assertTrue(all(value>=10 for value in bins))

    def test_insufficient_whole_sample_stays_one_bin(self):
        self.assertEqual(merge_background_bins([0.,1.,0.],[0.,1.,0.],10.),[0,3])

    def test_shape_conditioning_matches_an_equivalent_flat_likelihood(self):
        signal = dict(reference_cross_section_pb=.01,yields={'a':10.},binned_yields={'a':[1.,9.]})
        background = {'qcd':dict(yields={'a':1.e9},binned_yields={'a':[9.e8,1.e8]})}
        shaped = reference_cross_section(signal,background,('a',))
        flat = reference_cross_section(dict(reference_cross_section_pb=.01,yields={'x':1.,'y':9.}),
            {'qcd':dict(yields={'x':9.e8,'y':1.e8})},('x','y'))
        self.assertAlmostEqual(shaped,flat)
        self.assertLess(shaped,reference_cross_section(
            dict(reference_cross_section_pb=.01,yields={'a':10.}),
            {'qcd':dict(yields={'a':1.e9})},('a',)))

    def test_weighted_response_keeps_zero_reconstruction_events_in_denominator(self):
        data = dict(ids=np.zeros((4,3),dtype=np.uint64),category=np.array([0,-1,-1,-1]),
                    mass_gev=np.array([2.,math.nan,math.nan,math.nan]))
        report = dict(luminosity_pb=3000.,mass_binning={'a':dict(fine_bin_boundaries=[0,1000])})
        values,ess = weighted_shape(data,np.array([2.,1.,1.,1.]),report,('a',))
        self.assertEqual(values['a'],[1500.])
        self.assertEqual(ess['a'],1.)

    def test_exclusion_islands_stop_at_the_coupling_coverage(self):
        points = [(2.,.05),(5.,.2),(10.,.03),(20.,.09),(30.,.3)]
        segments = clipped_boundary_segments(points,.1)
        self.assertEqual(len(segments),2)
        self.assertEqual(segments[0][0],[2.,.05])
        self.assertAlmostEqual(segments[0][-1][1],.1)
        self.assertAlmostEqual(segments[1][0][1],.1)
        self.assertAlmostEqual(segments[1][-1][1],.1)
        self.assertTrue(all(epsilon<=.1 for segment in segments for mass,epsilon in segment))

    def test_no_contour_is_drawn_across_a_missing_mass(self):
        segments = clipped_boundary_segments([(2.,.05),None,(10.,.03),(20.,.09)],.1)
        self.assertEqual(segments,[[[10.,.03],[20.,.09]]])

    def test_full_axis_connects_mass_anchors_before_clipping(self):
        points = [(2.,.05),(5.,.2),(10.,.03),(20.,.09),(30.,.38),(40.,1.07)]
        segments = clipped_boundary_segments(points,1.)
        self.assertEqual(len(segments),1)
        self.assertIn([5.,.2],segments[0])
        self.assertAlmostEqual(segments[0][-1][1],1.)
        self.assertTrue(30.<segments[0][-1][0]<40.)

    def test_reco_cuts_require_both_momenta_and_downstream_vertex(self):
        data = dict(mass=np.ones(4),minimum_muon_p=np.array([30.,19.,30.,30.]),
                    distance_m=np.array([11.,11.,11.,math.nan]),
                    downstream_m=np.array([10.,10.,-10.,10.]))
        self.assertEqual(cut_mask(data,20.,10.).tolist(),[True,False,False,False])

    def test_baseline_does_not_silently_add_vertex_cuts(self):
        data = dict(mass=np.ones(1),minimum_muon_p=np.array([math.nan]),
                    distance_m=np.array([math.nan]),downstream_m=np.array([math.nan]))
        self.assertEqual(cut_mask(data,0.,0.).tolist(),[True])

    def test_outline_corner_does_not_change_boundary_or_shading(self):
        boundary = [[[2.,.05],[5.,.2],[10.,.03],[39.,1.]]]
        outline = boundary_outline_segments(boundary,2.,1.)
        self.assertEqual(outline[0][0],[2.,1.])
        self.assertEqual(outline[0][1:],boundary[0])
        self.assertEqual(boundary[0][0],[2.,.05])

    def test_outline_does_not_bridge_missing_low_mass_coverage(self):
        boundary = [[[10.,.03],[39.,1.]]]
        self.assertEqual(boundary_outline_segments(boundary,2.,1.),boundary)


class CountingProfileInputTest(unittest.TestCase):
    def test_missing_explicit_input_fails_before_drawing(self):
        with self.assertRaisesRegex(ValueError, '--counting-profile'):
            draw_mass_improvement(None, None, [], None)

    def test_explicit_input_preserves_values_and_requires_matching_yields(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            yields = directory / 'yields.json'
            yields.write_text('audited full-GEN yields')
            profile = directory / 'chosen_predictions.json'
            config = SimpleNamespace(audited_yields_path=yields, output_path=directory)
            saved = dict(yields_sha256=digest(yields),
                         records=[dict(point='m2_prompt', expected95_pb=12.)])
            profile.write_text(json.dumps(saved))
            results = [dict(point='m2_prompt', channel='combined', confidence_level=.95,
                            status='ok', mass_gev=2., expected_limit_cross_section_pb=3.)]
            axes = [Mock(), Mock()]
            figure, plotting = Mock(), Mock()
            plotting.subplots.return_value = figure, axes
            with patch('shift_limits_lifetime.plotting', return_value=plotting), \
                    patch('shift_limits_lifetime.finish') as finish:
                draw_mass_improvement(config, {}, results, profile)
            self.assertEqual(axes[0].plot.call_args_list[0].args[:2], ([2.], [12.]))
            self.assertEqual(axes[1].plot.call_args.args[1].tolist(), [4.])
            finish.assert_called_once()
            saved['yields_sha256'] = 'a' * 64
            profile.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'same audited yields'):
                draw_mass_improvement(config, {}, results, profile)
            profile.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Counting profile input'):
                draw_mass_improvement(config, {}, results, profile)

    def test_cli_requires_input_before_loading_study_configuration(self):
        script = Path(__file__).resolve().parents[1] / 'utils/shift_limits_shapes.py'
        for stage in ('preview', 'publish'):
            with self.subTest(stage=stage):
                result = subprocess.run([sys.executable, str(script), '--stage', stage,
                    '--config', 'missing_study_configuration'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('--counting-profile is required', result.stderr)


if __name__=='__main__':
    unittest.main()
