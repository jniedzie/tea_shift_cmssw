# Code style: AI-code. AI-generated regression tests for unsupported limit gaps.
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from shift_limits_current_plots import draw_regions
from shift_limits_profile import expected_limit, profile_q


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
        self.assertFalse(any(min(x) < 1.02 < max(x) for x, y in axis.lines))
        return axis

    def test_excluded_isolated_anchor_interrupts_boundary(self):
        middle = boundary(1.02)
        middle.update(epsilon_support=[3.e-6, 3.e-6],
                      excluded_intervals=[[3.e-6, 3.e-6]], roots=[])
        axis = self.check_gap(middle)
        self.assertTrue(any(x == [1.02] and y == [3.e-6] for x, y, options in axis.points))

    def test_unexcluded_mass_interrupts_boundary(self):
        middle = boundary(1.02)
        middle['excluded_intervals'] = []
        self.check_gap(middle)

    def test_connected_supported_neighbours_remain_connected(self):
        axis = RecordingAxis()
        draw_regions(axis, [boundary(.9), boundary(1.5)], 'zero_background', 'blue', 'test')
        self.assertTrue(any(x == [.9, 1.5] for x, y in axis.lines))

    def test_r_scales_inversely_with_model_signal(self):
        signal_limit, threshold = expected_limit([1.], [0.], .9)
        predicted = 2 * signal_limit
        r = signal_limit / predicted
        self.assertAlmostEqual(r, .5)
        self.assertGreater(profile_q(1., [predicted], [0.]), threshold)
        self.assertAlmostEqual(profile_q(r, [predicted], [0.]), threshold, places=6)


if __name__ == '__main__':
    unittest.main()
