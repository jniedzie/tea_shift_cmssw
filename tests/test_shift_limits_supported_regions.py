# Code style: AI-code. AI-generated regression tests for unsupported limit gaps.
import copy
from pathlib import Path
import sys
import unittest

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ANALYSIS/'bin'),str(ANALYSIS/'utils')]
from shift_limits_current_plots import draw_regions, supported_contour_rows
from shift_limits_profile import expected_limit, profile_q
from shift_dp_progress_plots import signal_strength_contours


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


if __name__ == '__main__':
    unittest.main()
