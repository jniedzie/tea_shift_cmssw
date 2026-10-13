"""Shape matching after density conversion, with physical SM bin weights."""

from array import array
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import ROOT

REPOSITORY = Path(__file__).resolve().parents[1]
for directory in ("tea/pylibs/plotting", "tea/pylibs/logger"):
    sys.path.insert(0, str(REPOSITORY / directory))

from Histogram import Histogram
from HistogramNormalizer import NormalizationType
from HistogramPlotter import HistogramPlotter
from Legend import Legend
from Sample import Sample, SampleType


class HistogramShapeNormalizationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        ROOT.gROOT.SetBatch(True)

    def shape_plot(self, backgrounds, signal, *, density=True, normalization=NormalizationType.to_background):
        """Run the real stack builder; parent GEN sums differ from cutflows."""
        edges = array("d", (0., 1., 11.))
        samples, histograms = [], []
        rows = [(f"background{index}", values, cross_section, SampleType.background)
                for index, (values, cross_section) in enumerate(backgrounds)]
        rows.append(("signal", signal, -1., SampleType.signal))
        for name, values, cross_section, sample_type in rows:
            path = self.directory / (name + ".root")
            with ROOT.TFile.Open(str(path), "RECREATE") as file:
                cutflow = ROOT.TH1D("cutFlow", "", 1, 0, 1)
                cutflow.SetBinContent(1, 7.)
                cutflow.Write()
            sample = Sample(name=name, file_path=str(path), type=sample_type,
                            cross_section=cross_section, initial_weight_sum=100.)
            samples.append(sample)
            raw = ROOT.TH1D("model_mass", "", len(edges) - 1, edges)
            raw.SetDirectory(0)
            raw.Sumw2()
            for index, value in enumerate(values, 1):
                if value:
                    raw.Fill(raw.GetBinCenter(index), value)
            histogram = Histogram(name="model_mass", norm_type=normalization,
                                  scale_by_bin_width=density)
            histogram.set_hist(raw)
            histograms.append((histogram, sample))
        config = SimpleNamespace(
            samples=samples, histograms=[Histogram(name="model_mass", norm_type=normalization)],
            luminosity=2., output_path=str(self.directory / "plots"), show_ratio_plots=False,
            custom_stacks_order=[sample.name for sample in samples],
            legends={SampleType.background: Legend(.1, .1, .5, .5, "f"),
                     SampleType.signal: Legend(.1, .1, .5, .5, "l")},
        )
        plotter = HistogramPlotter(config)
        for file in plotter.normalizer.input_files.values():
            self.addCleanup(file.Close)
        plotter.histosamples = histograms
        plotter.setupLegends()
        plotter.buildStacks()
        return plotter, {sample.name: histogram.hist for histogram, sample in plotter.histosamples}

    def test_different_width_populations_match_area_not_sum_of_densities(self):
        plotter, histograms = self.shape_plot([([100., 0.], 10.)], [0., 10.])
        background, signal = histograms["background0"], histograms["signal"]
        self.assertEqual(plotter.total_backgrounds_integral["model_mass"], 20.)
        self.assertEqual(background.Integral("width"), 20.)
        self.assertEqual(signal.Integral("width"), 20.)
        self.assertEqual(signal.GetBinContent(2), 2.)

    def test_full_gen_exposure_and_cross_section_scale_each_background_before_sum(self):
        plotter, histograms = self.shape_plot(
            [([10., 0.], 10.), ([0., 8.], 20.)], [0., 4.])
        # Parent sums are 100; selected cutflows of 7 must not replace them.
        self.assertAlmostEqual(histograms["background0"].Integral("width"), 2.)
        self.assertAlmostEqual(histograms["background1"].Integral("width"), 3.2)
        self.assertAlmostEqual(plotter.total_backgrounds_integral["model_mass"], 5.2)
        self.assertAlmostEqual(histograms["signal"].Integral("width"), 5.2)

    def test_count_plots_retain_ordinary_bin_sum(self):
        plotter, histograms = self.shape_plot([([100., 0.], 10.)], [0., 10.], density=False)
        self.assertEqual(plotter.total_backgrounds_integral["model_mass"], 20.)
        self.assertEqual(histograms["signal"].Integral(), 20.)
        self.assertEqual(histograms["signal"].GetBinContent(2), 20.)

    def test_signed_signal_weights_keep_their_sign_after_area_matching(self):
        _, histograms = self.shape_plot([([100., 0.], 10.)], [-2., 5.])
        signal = histograms["signal"]
        self.assertAlmostEqual(signal.Integral("width"), 20.)
        self.assertLess(signal.GetBinContent(1), 0.)
        self.assertGreater(signal.GetBinContent(2), 0.)
        self.assertAlmostEqual(signal.GetBinContent(1), -40. / 3.)
        self.assertAlmostEqual(signal.GetBinContent(2), 10. / 3.)

    def test_zero_signed_area_remains_unscaled_without_division_by_zero(self):
        _, histograms = self.shape_plot([([100., 0.], 10.)], [-1., 1.])
        signal = histograms["signal"]
        self.assertEqual(signal.Integral("width"), 0.)
        self.assertEqual(signal.GetBinContent(1), -1.)
        self.assertEqual(signal.GetBinContent(2), .1)

    def test_other_normalization_modes_keep_the_existing_density_integral(self):
        # The fix is specific to background shape matching, not rate scaling.
        plotter, histograms = self.shape_plot(
            [([0., 100.], 10.)], [0., 10.], normalization=NormalizationType.none)
        self.assertEqual(plotter.total_backgrounds_integral["model_mass"], 10.)
        self.assertEqual(histograms["signal"].GetBinContent(2), 1.)


if __name__ == "__main__":
    unittest.main()
