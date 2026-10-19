# Code style: AI-code.
"""Run with python3 -m unittest discover -s utils -p test_shift_diagnostics_plotter.py."""
import importlib.util
import argparse
from array import array
from contextlib import redirect_stdout
import io
import subprocess
from pathlib import Path
import tempfile
import unittest

import ROOT

spec = importlib.util.spec_from_file_location("plotter", Path(__file__).with_name("shift_diagnostics_plotter.py"))
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


OPENING_ANGLE_EDGES = array("d", [0., .001, .002, .003, .005, .01, .02, .03, .05,
                                 .075, .1, .15, .2, .3, .5, 1., 1.5, 2., 3.15])


class EfficiencySource:
  def __init__(self, object_name, include_opening_angle=True):
    self.histograms = {}
    self.requests = []
    for variable in plotter.efficiency_variables(object_name):
      if variable == "openingAngle" and not include_opening_angle:
        continue
      for suffix, count in (("pass", 5), ("total", 10)):
        name = f"{plotter.efficiency_prefix(object_name, '')}_{variable}_{suffix}"
        hist = (ROOT.TH1D(name, name, len(OPENING_ANGLE_EDGES) - 1, OPENING_ANGLE_EDGES)
                if variable == "openingAngle" else ROOT.TH1D(name, name, 10, -1, 1))
        hist.SetDirectory(0)
        for _ in range(count):
          hist.Fill(0. if variable == "openingAngle" else 0.1)
        self.histograms["efficiency/" + name] = hist

  def Get(self, name):
    self.requests.append(name)
    return self.histograms.get(name)


class ComparisonTests(unittest.TestCase):
  def write_histogram_fixture(self, root, version, label, include_opening_angle=True):
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
          for variable in plotter.efficiency_variables(object_name):
            if variable == "openingAngle" and not include_opening_angle:
              continue
            for suffix, count in (("pass", version), ("total", 10)):
              name = f"{plotter.efficiency_prefix(object_name, category)}_{variable}_{suffix}"
              hist = (ROOT.TH1D(name, name, len(OPENING_ANGLE_EDGES) - 1, OPENING_ANGLE_EDGES)
                      if variable == "openingAngle" else ROOT.TH1D(name, name, 10, -1, 1))
              for _ in range(count):
                hist.Fill(0.1)
              hist.Write()
    return path

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
        self.write_histogram_fixture(root, version, label)
      args = argparse.Namespace(compare_versions=[1, 2, 3, 4], histograms_dir=str(root), output_dir=str(root))
      plotter.run_version_comparison(args)
      pdfs = list((root / "comparisons/v1_vs_v2_vs_v3_vs_v4").glob("*.pdf"))
      self.assertEqual(len(pdfs), 6)
      for pdf in pdfs:
        text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        for label in labels:
          self.assertIn(label, text)
        if pdf.name == "shiftdimuonvertex_efficiency_inclusive.pdf":
          self.assertIn("opening angle (rad)", text)
        elif pdf.name == "shiftmuon_efficiency_inclusive.pdf":
          self.assertNotIn("opening angle", text)

  def test_legacy_comparison_warns_without_drawing_a_partial_angle_comparison(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.write_histogram_fixture(root, 1, "current")
      self.write_histogram_fixture(root, 2, "legacy", include_opening_angle=False)
      args = argparse.Namespace(compare_versions=[1, 2], histograms_dir=str(root), output_dir=str(root))
      messages = io.StringIO()
      with redirect_stdout(messages):
        plotter.run_version_comparison(args)
      self.assertIn("opening-angle comparison unavailable; missing histograms for legacy", messages.getvalue())
      output = root / "comparisons/v1_vs_v2"
      self.assertEqual(len(list(output.glob("*.pdf"))), 6)
      text = subprocess.check_output(
          ["pdftotext", str(output / "shiftdimuonvertex_efficiency_inclusive.pdf"), "-"], text=True)
      self.assertIn("Opening-angle efficiency unavailable", text)
      self.assertIn("Missing histograms: legacy", text)
      self.assertIn("10.0%", text)
      self.assertIn("20.0%", text)

  def test_incomplete_opening_angle_pair_fails_before_any_comparison_pdf(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = self.write_histogram_fixture(root, 1, "incomplete")
      self.write_histogram_fixture(root, 2, "complete")
      with ROOT.TFile(str(path), "UPDATE") as output:
        output.GetDirectory("efficiency").Delete("ShiftDimuonVertexEfficiency_openingAngle_total;*")
      args = argparse.Namespace(compare_versions=[1, 2], histograms_dir=str(root), output_dir=str(root))
      with self.assertRaisesRegex(RuntimeError, "missing or inconsistent efficiency .*openingAngle"):
        plotter.run_version_comparison(args)
      self.assertFalse((root / "comparisons").exists())


class EfficiencyTests(unittest.TestCase):
  def test_angle_panel_preserves_truth_bins_and_single_muon_panels(self):
    for object_name, rows, expected in (("ShiftDimuonVertex", 4, 6), ("ShiftMuon", 3, 5)):
      with self.subTest(object_name=object_name):
        source = EfficiencySource(object_name)
        canvas = ROOT.TCanvas("test_" + object_name, "", 1100, rows * 400)
        canvas.Divide(2, rows)
        objects = plotter.draw_efficiencies(canvas, source, object_name, [("", "Inclusive", ROOT.kBlack)])
        efficiencies = [obj for obj in objects if obj.InheritsFrom("TEfficiency")]
        self.assertEqual(len(efficiencies), expected)
        self.assertEqual(canvas.GetPad(1).GetLogx(), 0)
        if object_name == "ShiftDimuonVertex":
          angle = efficiencies[-1]
          self.assertIn("openingAngle", angle.GetName())
          self.assertEqual(angle.GetTotalHistogram().GetNbinsX(), len(OPENING_ANGLE_EDGES) - 1)
          self.assertEqual(angle.GetTotalHistogram().GetBinContent(1), 10)
          self.assertEqual(angle.GetPassedHistogram().GetBinContent(1), 5)
          self.assertEqual(canvas.GetPad(6).GetLogx(), 1)
          frame = canvas.GetPad(6).GetListOfPrimitives().FindObject("hframe")
          self.assertEqual(frame.GetXaxis().GetTitle(), "Generator #mu#mu opening angle (rad)")
          self.assertAlmostEqual(frame.GetXaxis().GetXmin(), 0.0001)
          self.assertTrue(canvas.GetPad(7).GetListOfPrimitives().FindObject("TPave"))
        else:
          self.assertFalse(any("openingAngle" in name for name in source.requests))
          self.assertTrue(canvas.GetPad(6).GetListOfPrimitives().FindObject("TPave"))
        canvas.Close()

  def test_legacy_dimuon_plot_warns_and_keeps_the_existing_five_panels(self):
    source = EfficiencySource("ShiftDimuonVertex", include_opening_angle=False)
    canvas = ROOT.TCanvas("test_legacy_dimuon", "", 1100, 1600)
    canvas.Divide(2, 4)
    messages = io.StringIO()
    with redirect_stdout(messages):
      objects = plotter.draw_efficiencies(canvas, source, "ShiftDimuonVertex", [("", "Inclusive", ROOT.kBlack)])
    self.assertIn("ShiftDimuonVertexEfficiency_openingAngle' was not found", messages.getvalue())
    self.assertEqual(sum(obj.InheritsFrom("TEfficiency") for obj in objects), 5)
    self.assertEqual(canvas.GetPad(6).GetLogx(), 0)
    notice = canvas.GetPad(6).GetListOfPrimitives().FindObject("TPave")
    self.assertEqual(notice.GetListOfLines().First().GetTitle(), "Opening-angle efficiency unavailable")
    canvas.Close()


if __name__ == "__main__":
  unittest.main()


class ResponseWidthTests(unittest.TestCase):
  def test_uniform_width_and_known_flow_probability(self):
    histogram = ROOT.TH1D("uniform_response", "", 100, 0., 1.)
    histogram.SetDirectory(0)
    for index in range(100):
      histogram.Fill((index+.5)/100.)
    self.assertAlmostEqual(plotter.response_summary(histogram)["width"][0], .341345, places=6)
    self.assertAlmostEqual(plotter.response_summary(histogram)["bias"][0], .5, delta=.01)
    for _ in range(10):
      histogram.Fill(-1.)
    self.assertAlmostEqual(plotter.response_summary(histogram)["width"][0], .3754795, places=6)
    self.assertAlmostEqual(plotter.response_summary(histogram)["bias"][0], .45, delta=.01)
    for _ in range(20):
      histogram.Fill(-1.)
    self.assertIsNone(plotter.response_summary(histogram))

  def test_sparse_width_is_undefined(self):
    histogram = ROOT.TH1D("sparse_response", "", 10, -1., 1.)
    histogram.SetDirectory(0)
    histogram.Fill(0.)
    self.assertIsNone(plotter.response_summary(histogram))

  def test_process_render_uses_only_its_local_inputs(self):
    import types
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source_path = root / "process.root"
      with ROOT.TFile(str(source_path), "RECREATE") as source:
        for kind in ("Muon", "Dimuon"):
          source.mkdir(kind.lower()).cd()
          name = "Shift" + kind + ("Vertex" if kind == "Dimuon" else "") + "_pt"
          histogram = ROOT.TH1D(name, name, 10, 0., 10.)
          for _ in range(30):
            histogram.Fill(3.)
          histogram.Write()
          source.cd()
        source.mkdir("response").cd()
        for kind in ("Muon", "Dimuon"):
          for state in ("", "Constrained"):
            name = kind + "Response" + state + "_ptVspt"
            histogram = ROOT.TH2D(name, name, 10, 0., 10., 100, -1., 1.)
            for index in range(30):
              for x in (3., 4., 6.):
                histogram.Fill(x, index/100.)
            histogram.Write()
      settings = types.ModuleType("shift_plotter_config")
      settings.reconstructionReportCategories = {"Muon": {}, "Dimuon": {}}
      settings.reconstructionResponseOverview = ("pt",)
      settings.reconstructionResponseTitles = {"pt": "relative pT"}
      settings.reconstructionResponseAxes = {"pt": "True pT [GeV]"}
      settings.reconstructionResponseVariables = {"pt": "Relative pT residual"}
      settings.reconstructionResponseMinimumEntries = 20
      settings.reconstructionResponseDisplayRanges = {"pt": (-1., 1.)}
      settings.reconstructionResponseLogY = True
      settings.reconstructionDiagnosticLabels = {"pt": "pT [GeV]"}
      settings.reconstructionDiagnosticVariables = {"Muon": ("pt",), "Dimuon": ("pt",)}
      category = [("", "inclusive", "Inclusive")]
      with patch.object(plotter, "reconstruction_plot_settings", return_value=vars(settings)), \
           patch.object(plotter, "MUON_RESOLUTION_TYPES", category), \
           patch.object(plotter, "DIMUON_RESOLUTION_TYPES", category):
        plotter.draw_process_responses(["QCD="+str(source_path), "DY="+str(source_path)], str(root / "plots"))
      pdfs = list((root / "plots").glob("*.pdf"))
      self.assertEqual({path.name for path in pdfs}, {"muon_inclusive.pdf", "dimuon_inclusive.pdf"})
      for path in pdfs:
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        self.assertEqual(text.count("\f"), 5)
        for expected in ("Direct resolution", "Direct median bias", "QCD", "DY", "Constrained", "Unconstrained", "order-statistic"):
          self.assertIn(expected, text)
      with ROOT.TFile(str(source_path)) as source:
        canvas = ROOT.TCanvas("response_errors_test", "", 900, 600)
        retained = plotter.draw_response_panel(canvas, [("QCD", source, ROOT.kBlue)], "Muon", "", "", "pt", "pt", "width", vars(settings), {})
        graphs = [item for item in retained if item.InheritsFrom("TGraphAsymmErrors")]
        self.assertEqual(len(graphs), 2)
        self.assertEqual([graph.GetN() for graph in graphs], [2, 1])
        self.assertGreater(graphs[0].GetErrorYlow(0), 0.)
        self.assertGreater(graphs[0].GetErrorYhigh(0), 0.)
        multigraph = next(item for item in retained if item.InheritsFrom("TMultiGraph"))
        self.assertEqual(multigraph.GetGraphDrawOption(graphs[0]), "LP")
        canvas.Clear()
        canvas.Divide(2, 3)
        retained = []
        for index, state in enumerate(("", "Constrained")):
          retained.extend(plotter.draw_response_panel(canvas.cd(index+1), [("QCD", source, ROOT.kBlue)], "Muon", "", state, "pt", "pt", "width", vars(settings), {}))
        plotter.align_response_panels(canvas, 0)
        pairs = [item for item in retained if item.InheritsFrom("TMultiGraph")]
        self.assertEqual(pairs[0].GetHistogram().GetMinimum(), pairs[1].GetHistogram().GetMinimum())
        self.assertEqual(pairs[0].GetHistogram().GetMaximum(), pairs[1].GetHistogram().GetMaximum())
        canvas.Close()

  def test_binomial_rank_inverts_the_exact_count_distribution(self):
    import math
    for total, p, tail in ((20, .158655, .0793275), (100, .5, .158655), (40, .841345, .9206725)):
      rank = plotter.response_binomial_rank(total, p, tail)
      cdf = lambda k: sum(math.comb(total, j)*p**j*(1-p)**(total-j) for j in range(k+1))
      self.assertGreaterEqual(cdf(rank), tail-1.e-12)
      self.assertLess(cdf(rank-1), tail)

  def test_single_bin_precision_and_flow_are_not_hidden(self):
    histogram = ROOT.TH1D("one_bin_response", "", 10, -5., 5.)
    histogram.SetDirectory(0)
    for _ in range(1000):
      histogram.Fill(.1)
    summary = plotter.response_summary(histogram)
    value, down, up = summary["width"]
    self.assertAlmostEqual(value-down, 0.)
    self.assertGreater(up, 0.)
    median, down, up = summary["bias"]
    self.assertAlmostEqual(median-down, 0.)
    self.assertAlmostEqual(median+up, 1.)
    for _ in range(1000):
      histogram.Fill(-10.)
    self.assertIsNone(plotter.response_summary(histogram))

  def test_gaussian_width_and_interval_scale(self):
    histogram = ROOT.TH1D("gaussian_response", "", 2000, -10., 10.)
    histogram.SetDirectory(0)
    random = ROOT.TRandom3(1447)
    for _ in range(10000):
      histogram.Fill(random.Gaus(0., 1.))
    summary = plotter.response_summary(histogram)
    width, down, up = summary["width"]
    self.assertAlmostEqual(width, 1., delta=.035)
    self.assertGreater(min(down, up), .005)
    self.assertLess(max(down, up), .06)
    self.assertAlmostEqual(summary["bias"][0], 0., delta=.035)
