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
from shift_limits_shapes import (merge_background_bins, rebin, weighted_shape, clipped_boundary_segments,
                                boundary_outline_segments, draw_mass_improvement,
                                source_variation_templates, source_templates_at_lumi)
from shift_limits import digest, reference_cross_section
from shift_limits_cuts import cut_mask


class SourceVariationTemplatesTest(unittest.TestCase):
    def theory(self):
        return dict(variations={name:dict(available=True) for name in ('Lower','Central','Upper')},
            records=[dict(event_id=[10,1,i],weights=dict(zip(('Lower','Central','Upper'),weights)))
                     for i,weights in enumerate(((.5,1.,2.),(2.,1.,.25),(1.,1.,1.)),1)])

    def events(self):
        return [dict(id=[10,1,1],weight=2.,dimuons=[dict(category='Both-Both',mass_gev=1.5),
                                                    dict(category='Both-Both',mass_gev=5.)]),
                dict(id=[10,1,2],weight=1.,dimuons=[dict(category='Other',mass_gev=-1.)]),
                dict(id=[10,1,3],weight=3.,dimuons=[])]

    def test_density_weights_preserve_flows_raw_counts_and_cross_bin_correlations(self):
        result=source_variation_templates(self.events(),self.theory(),[0.,1.,2.],('Both-Both','Other'))
        sums=np.array(result['weighted_sums'])
        self.assertEqual(result['raw_MC_events'],3)
        self.assertEqual(result['raw_MC_events_with_dimuon'],2)
        self.assertEqual(result['raw_MC_dimuon_pairs'],[2,1])
        self.assertEqual(list(sums[:,0,2]),[1.,2.,4.])
        self.assertEqual(list(sums[:,0,3]),[1.,2.,4.])
        self.assertEqual(list(sums[:,1,0]),[2.,1.,.25])
        cross=next(row['values'] for row in result['covariance'] if row['cells']==[0,2,0,3])
        self.assertEqual(cross[1][1],4.)
        self.assertEqual(cross[1][2],8.)

    def test_shared_source_collision_is_not_independent_raw_MC(self):
        events=self.events()[:2]
        for event in events:
            event['dimuons']=[dict(category='Both-Both',mass_gev=1.5)]
            event['source_collision_id']=['same-bank',9]
        result=source_variation_templates(events,self.theory(),[0.,1.,2.],('Both-Both',))
        self.assertEqual(result['response_clusters_with_dimuon'],1)
        self.assertEqual(result['entry_sumw2'][1][0][2],5.)
        self.assertEqual(result['cluster_sumw2'][1][0][2],9.)
        self.assertAlmostEqual(result['cluster_sumw2'][2][0][2],4.25**2)

    def test_lumi_factor_uses_original_GEN_once_and_does_not_scale_raw_counts(self):
        result=source_variation_templates(self.events(),self.theory(),[0.,1.,2.],('Both-Both','Other'))
        normalized=source_templates_at_lumi(result,cross_section_pb=10.,full_gen_sumw=100.,luminosity_pb=300.)
        self.assertEqual(normalized['raw_MC_events'],3)
        self.assertEqual(normalized['original_full_gen_sumw'],100.)
        self.assertEqual(normalized['common_lumi_factor'],30.)
        self.assertEqual(np.array(normalized['events_normalized_to_lumi'])[1].sum(),150.)
        self.assertEqual(normalized['response_variance_normalized_to_lumi'][1][0][2],3600.)
        doubled=source_templates_at_lumi(result,cross_section_pb=10.,full_gen_sumw=100.,luminosity_pb=600.)
        self.assertEqual(doubled['events_normalized_to_lumi'][1][0][2],120.)
        self.assertEqual(doubled['response_variance_normalized_to_lumi'][1][0][2],14400.)
        self.assertFalse(normalized['extra_integrated_variation_rate_factor'])

    def test_missing_reused_or_unsupported_source_weights_fail(self):
        import copy
        for change in ('duplicate','missing','unavailable','missing_weight'):
            theory=copy.deepcopy(self.theory())
            if change=='duplicate':theory['records'].append(copy.deepcopy(theory['records'][0]))
            elif change=='missing':theory['records'].pop()
            elif change=='unavailable':theory['variations']['Upper']['available']=False
            else:theory['records'][0]['weights']['Upper']=None
            with self.subTest(change=change),self.assertRaises(ValueError):
                source_variation_templates(self.events(),theory,[0.,1.,2.],('Both-Both','Other'))
        events=self.events();events.append(events[0])
        with self.assertRaises(ValueError):
            source_variation_templates(events,self.theory(),[0.,1.,2.],('Both-Both','Other'))


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
