# Code style: AI-code. AI-generated regression tests for unsupported limit gaps.
import copy
from pathlib import Path
import sys
import unittest

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ANALYSIS/'bin'),str(ANALYSIS/'utils')]
from shift_limits_current_plots import draw_regions, supported_contour_rows
from shift_limits_profile import expected_limit, profile_q
from shift_dp_progress_plots import signal_strength_contours, strength_surface, strength_contour_segments, strength_interpolator, clip_strength_domain


class RecordingAxis:
    def __init__(self):
        self.lines = []
        self.points = []

    def plot(self, x, y, **options):
        self.lines.append((list(x), list(y)))

    def scatter(self, x, y, **options):
        self.points.append((list(x), list(y), options))

    def fill_between(self, *args, **kwargs):
        pass


def boundary(mass):
    return dict(mass_gev=mass, mode='zero_background', epsilon_support=[1.e-6, 1.e-3],
                excluded_intervals=[[1.e-5, 1.e-4]], roots=[1.e-5, 1.e-4],
                upper_prompt_continuation=False)


class SupportedRegionsTest(unittest.TestCase):
    def check_gap(self, middle):
        axis = RecordingAxis()
        rows = [boundary(.9), middle, boundary(1.5)]
        saved = copy.deepcopy(rows)
        draw_regions(axis, rows, 'zero_background', 'blue', 'test')
        self.assertEqual(rows, saved)
        self.assertTrue(any(x == [.9, 1.5] for x, y in axis.lines))
        self.assertFalse(any(len(x)>1 and min(x)==max(x) for x,y in axis.lines))
        return axis

    def test_isolated_anchor_is_omitted_from_continuous_curve(self):
        middle = boundary(1.02)
        middle.update(epsilon_support=[3.e-6, 3.e-6],
                      excluded_intervals=[[3.e-6, 3.e-6]], roots=[])
        axis = self.check_gap(middle)
        self.assertEqual(axis.points, [])

    def test_unexcluded_mass_is_omitted_without_vertical_lines(self):
        middle = boundary(1.02)
        middle['excluded_intervals'] = []
        self.check_gap(middle)

    def test_connected_supported_neighbours_remain_connected(self):
        axis = RecordingAxis()
        draw_regions(axis, [boundary(.9), boundary(1.5)], 'zero_background', 'blue', 'test')
        self.assertTrue(any(x == [.9, 1.5] for x, y in axis.lines))

    def test_interpolation_island_without_generated_bracket_is_omitted(self):
        row=boundary(1.02)
        row.update(confidence_level=.9, epsilon_grid=[1.e-6,1.e-5,1.e-4,1.e-3],
                   q_grid=[0.,5.,5.,0.])
        endpoints=[dict(mass_gev=1.02,epsilon=e) for e in [1.e-6,1.e-3]]
        self.assertEqual(supported_contour_rows([row],endpoints),[])
        anchors=endpoints+[dict(mass_gev=1.02,epsilon=1.e-4)]
        self.assertEqual(len(supported_contour_rows([row],anchors)),1)
        row['channel']='Both-Both'
        anchors[-1]['counts']={'Both-Both': {'sumw': 0.}}
        self.assertEqual(supported_contour_rows([row],anchors),[])

    def test_r_scales_inversely_with_model_signal(self):
        signal_limit, threshold = expected_limit([1.], [0.], .9)
        predicted = 2 * signal_limit
        r = signal_limit / predicted
        self.assertAlmostEqual(r, .5)
        self.assertGreater(profile_q(1., [predicted], [0.]), threshold)
        self.assertAlmostEqual(profile_q(r, [predicted], [0.]), threshold, places=6)

    def test_signal_strength_map_omits_prompt_roots_beyond_generated_support(self):
        row=boundary(1.5)
        row.update(confidence_level=.9,channel='Both-Both',roots=[.016],
                   excluded_intervals=[[.016,.03]],epsilon_support=[1.e-6,.03],
                   epsilon_grid=[1.e-6,.001,.016,.03],q_grid=[0.,0.,2.7055,5.])
        signals=[dict(mass_gev=1.5,epsilon=e) for e in [1.e-6,.001]]
        self.assertEqual(signal_strength_contours(dict(signals=signals),dict(records=[row])),[])
        signals.append(dict(mass_gev=1.5,epsilon=.03))
        self.assertEqual(len(signal_strength_contours(dict(signals=signals),dict(records=[row]))),1)
        signals[-1]['counts']={'Both-Both':dict(sumw=0.)}
        self.assertEqual(signal_strength_contours(dict(signals=signals),dict(records=[row])),[])


class StrengthSurfaceTest(unittest.TestCase):
    def test_log_plane_and_no_extrapolation(self):
        import numpy as np
        rows = [dict(mode='background', mass_gev=m, epsilon=e, r=m*e)
                for m,e in [(1.,.001),(10.,.001),(1.,.1),(10.,.01)]]
        edges, centres, z = strength_surface(rows, 'background', bins=12)
        xx, yy = np.meshgrid(*[np.log10(c) for c in centres])
        finite = np.isfinite(z)
        self.assertTrue(finite.any())
        self.assertTrue((~finite).any())
        np.testing.assert_allclose(z[finite], (xx+yy)[finite], atol=1.e-12)

    def test_single_point_mass_column_does_not_blank_neighbouring_hull(self):
        import numpy as np
        rows = [dict(mode='background', mass_gev=m, epsilon=e, r=e/1.e-4)
                for m,e in [(1.,1.e-5),(1.,1.e-3),(10.,1.e-5),(10.,1.e-3),(3.,1.e-4)]]
        _, _, z = strength_surface(rows, 'background', bins=30)
        self.assertTrue(np.isfinite(z).all())
        segments = strength_contour_segments(dict(records=rows), 'background')
        self.assertEqual(len(segments), 1)
        field, _, _ = strength_interpolator(rows, 'background')
        np.testing.assert_allclose(field(np.log10(segments[0])), 0., atol=1.e-12)
        np.testing.assert_allclose(segments[0][:, 1], 1.e-4, rtol=.002)

    def test_zero_response_is_not_a_finite_interpolation_vertex(self):
        rows = [dict(mode='background', mass_gev=m, epsilon=e, r=r)
                for m,e,r in [(1.,.001,1.),(10.,.001,0.),(1.,.1,3.)]]
        with self.assertRaises(ValueError):
            strength_surface(rows, 'background')

    def test_contours_respect_model_domain_without_closing_support_edges(self):
        import numpy as np
        domain=dict(mass_boundary_gev=12.,below_maximum_epsilon=.03,above_maximum_epsilon=.1)
        parts=clip_strength_domain([np.array([[5.,.02],[20.,.08]])],domain)
        self.assertEqual(len(parts),2)
        self.assertAlmostEqual(parts[0][-1,1],.03)
        self.assertAlmostEqual(parts[1][0,0],12.)


if __name__ == '__main__':
    unittest.main()
