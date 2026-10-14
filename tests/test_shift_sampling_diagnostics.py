# Code style: AI-code. AI-generated statistical and source-identity checks.
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "sampling", Path(__file__).parents[1] / "utils" / "shift_sampling_diagnostics.py")
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


class SamplingDiagnosticsTest(unittest.TestCase):
    def test_effective_count_survives_tiny_weights(self):
        self.assertAlmostEqual(sampling.effective_count([1e-250, 2e-250]), 1.8)
        self.assertEqual(sampling.effective_count([0.0, 0.0]), 0.0)

    def test_unequal_weights_are_not_counted_as_equal_events(self):
        self.assertAlmostEqual(sampling.effective_count([99, 1]), 10000 / 9802)

    def test_threshold_is_strict_and_counts_sources_once(self):
        events = [{"mass": 20, "weight": 1}, {"mass": 21, "weight": 1}]
        self.assertEqual(sampling.tail_counts(events, "mass", 20), (1, 1.0))

    def test_repeated_sources_are_rejected(self):
        row = {"event_id": [1, 1, 1], "final_weight": 1, "actual_OS_muon_pairs": []}
        with self.assertRaisesRegex(ValueError, "Repeated source"):
            sampling.source_events([row, row])

    def test_invalid_weights_are_rejected(self):
        for weight in (-1, float("inf"), float("nan")):
            row = {"event_id": [1, 1, 1], "final_weight": weight, "actual_OS_muon_pairs": []}
            with self.assertRaisesRegex(ValueError, "Invalid source weight"):
                sampling.source_events([row])


if __name__ == "__main__":
    unittest.main()
