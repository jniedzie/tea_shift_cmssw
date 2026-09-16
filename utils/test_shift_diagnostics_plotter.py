"""Run with python3 -m unittest discover -s utils -p test_shift_diagnostics_plotter.py."""
import importlib.util
import argparse
from array import array
import subprocess
from pathlib import Path
import tempfile
import unittest

import ROOT

spec = importlib.util.spec_from_file_location("plotter", Path(__file__).with_name("shift_diagnostics_plotter.py"))
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


class ComparisonTests(unittest.TestCase):
  def test_labels_and_duplicate_recipes(self):
    selections = [("", 44, "ca5af1024788_control_v3"),
                  ("", 45, "ca5af1024788-dirty-12345678_materialField_v3"),
                  ("", 46, "ca5af1024788")]
    self.assertEqual(plotter.comparison_labels(selections), ["control v3", "materialField v3", "v46"])
    selections.append(("", 47, "abcdef123456_control_v3"))
    self.assertEqual(plotter.comparison_labels(selections)[::3], ["control v3 (v44)", "control v3 (v47)"])

  def test_wide_bins_preserve_core_resolution_despite_outliers(self):
    edges = array("d", [-1.e9, -1024., -512., -256., -128., -64., -32., -16., -8., -4., -2.] +
                  [-1. + 0.02 * index for index in range(101)] +
                  [2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 1.e9])
    hist = ROOT.TH1D("wide", "", len(edges) - 1, edges)
    hist.SetDirectory(0)
    values = [-0.2 + 0.01 * index for index in range(41)] * 5 + [-40., 900.]
    for value in values:
      hist.Fill(value)
    with tempfile.TemporaryDirectory() as directory:
      path = str(Path(directory) / "tails.root")
      with ROOT.TFile(path, "RECREATE") as output:
        hist.Write()
      with ROOT.TFile(path) as source:
        hist = source.Get("wide")
        plotter.require_complete_resolution(hist)
        self.assertEqual(hist.Integral(), len(values))
        median, sigma68 = plotter.robust_resolution_summary(hist)
        self.assertAlmostEqual(median, 0., delta=0.02)
        self.assertAlmostEqual(sigma68, 0.14, delta=0.02)
        self.assertGreater(hist.GetStdDev(), 50.)

  def test_legacy_overflow_is_not_reported_as_complete(self):
    hist = ROOT.TH1D("legacy_tails", "", 50, -1, 1)
    hist.SetDirectory(0)
    hist.Sumw2()
    hist.Fill(2., 1.)
    hist.Fill(2., -1.)
    with self.assertRaisesRegex(RuntimeError, "Rerun shift_histogrammer"):
      plotter.require_complete_resolution(hist)

  def test_every_requested_series_is_drawn(self):
    class Source:
      def __init__(self, hist):
        self.hist = hist

      def Get(self, name):
        return self.hist

    hist = ROOT.TH1D("complete", "", 50, -1, 1)
    hist.SetDirectory(0)
    for _ in range(10):
      for value in (-0.1, 0., 0.1):
        hist.Fill(value)
    source = Source(hist)
    for count in (2, 4, 9):
      comparison = [(f"recipe {i}", source) for i in range(count)]
      canvas = plotter.comparison_canvas(f"test_{count}", count, 1600)
      objects = plotter.draw_scale_resolution_summary(canvas, plotter.SUMMARY_CANVAS_SPECS[0], source, comparison)
      graphs = [obj for obj in objects if obj.InheritsFrom("TGraphErrors")]
      self.assertEqual(len(graphs), count * len(plotter.SUMMARY_CANVAS_SPECS[0]["variables"]))
      self.assertTrue(all(graph.GetN() == 5 for graph in graphs))
      self.assertEqual(len(plotter.comparison_styles(count)), count)
      canvas.Close()

  def test_four_version_pdfs_include_every_label(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      labels = ["control v3", "material v3", "field v3", "materialField v3"]
      for version, label in enumerate(labels, 1):
        path = root / f"v{version}_abcdef123456_{label.replace(' ', '_')}" / "histograms.root"
        path.parent.mkdir()
        with ROOT.TFile(str(path), "RECREATE") as output:
          output.mkdir("resolution").cd()
          for summary in plotter.SUMMARY_CANVAS_SPECS:
            for category, _, _ in summary["categories"]:
              for variables in summary["variables"]:
                for variable in variables:
                  name = f"{summary['histogram_prefix']}{category}_{variable}"
                  hist = ROOT.TH1D(name, name, 200, -100, 100)
                  for _ in range(10):
                    for value in (-20., 0.1, 3. * version):
                      hist.Fill(value)
                  hist.Write()
          output.mkdir("efficiency").cd()
          for object_name, categories in (("ShiftMuon", plotter.MUON_EFFICIENCY_TYPES),
                                          ("ShiftDimuonVertex", plotter.DIMUON_EFFICIENCY_TYPES)):
            for category, _, _ in categories:
              for variable in plotter.EFFICIENCY_VARIABLES:
                for suffix, count in (("pass", version), ("total", 10)):
                  name = f"{plotter.efficiency_prefix(object_name, category)}_{variable}_{suffix}"
                  hist = ROOT.TH1D(name, name, 10, -1, 1)
                  for _ in range(count):
                    hist.Fill(0.1)
                  hist.Write()
      args = argparse.Namespace(compare_versions=[1, 2, 3, 4], histograms_dir=str(root), output_dir=str(root))
      plotter.run_version_comparison(args)
      pdfs = list((root / "comparisons/v1_vs_v2_vs_v3_vs_v4").glob("*.pdf"))
      self.assertEqual(len(pdfs), 6)
      for pdf in pdfs:
        text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        for label in labels:
          self.assertIn(label, text)


if __name__ == "__main__":
  unittest.main()
