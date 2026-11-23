# Code style: AI-code. AI-generated regression tests for data exposure labels.
"""Use synthetic aggregate counts to check luminosity on both data figures."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.plot_shift_data_control_region import MASS_EDGES, render_control_region


def sample(luminosity, selection):
    return dict(selection=selection, streams=["Muon0"], closed_lumisections=50,
                recorded_luminosity_pb_inverse=luminosity,
                all_bins=[0] * 19 + [6], both_both_bins=[0] * 19 + [2])


class ControlRegionRenderTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib") and shutil.which("pdftotext"),
                         "Matplotlib and pdftotext are required for PDF readback")
    def test_luminosity_is_visible_in_both_figures(self):
        aggregate = dict(mass_bin_edges_gev=list(MASS_EDGES),
                         samples=[sample(11.440983444, "CSC tight"), sample(None, "Unknown exposure")])
        with tempfile.TemporaryDirectory() as directory:
            all_path, both_path = render_control_region(aggregate, directory)
            for path, count in ((all_path, 6), (both_path, 2)):
                output = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                                        capture_output=True, text=True).stdout
                self.assertIn("11.4410 pb⁻¹ recorded CMS luminosity", output)
                self.assertIn("Processed luminosity unknown", output)
                self.assertIn(f"{count} pairs in the window", output)

    def test_undeclared_or_invalid_luminosity_cannot_publish(self):
        absent = sample(None, "Test")
        del absent["recorded_luminosity_pb_inverse"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots"
            for item in (absent, sample(0.0, "Test"), sample(float("nan"), "Test")):
                with self.assertRaises(ValueError):
                    render_control_region(dict(mass_bin_edges_gev=list(MASS_EDGES), samples=[item]), output)
                self.assertFalse(output.exists())

    def test_out_of_window_and_inconsistent_aggregates_cannot_publish(self):
        wider_edges = [2.5, *MASS_EDGES[1:]]
        invalid_category = sample(1.0, "Test")
        invalid_category["both_both_bins"][0] = 1
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots"
            for edges, item in ((wider_edges, sample(1.0, "Test")), (list(MASS_EDGES), invalid_category)):
                with self.assertRaises(ValueError):
                    render_control_region(dict(mass_bin_edges_gev=edges, samples=[item]), output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
