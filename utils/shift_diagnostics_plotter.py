#!/usr/bin/env python3

import argparse
from array import array
import glob
import math
import os
import re

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleSize(0.060, "XY")
ROOT.gStyle.SetLabelSize(0.050, "XY")
ROOT.gStyle.SetLabelOffset(0.012, "XY")
ROOT.gStyle.SetTitleOffset(1.15, "X")
ROOT.gStyle.SetTitleOffset(1.35, "Y")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMS_DP_QOVERPT_DATA = os.path.join(PROJECT_DIR, "data", "cms_dp_2015_015_qoverpt_digitized.txt")

# Rebin factors for all histograms on the corresponding resolution canvas.
MUON_RESOLUTION_REBIN = 1
DIMUON_RESOLUTION_REBIN = 2

# The canvases are split into relatively small pads, so ROOT's defaults clip
# long axis titles and the outermost tick labels.
PAD_LEFT_MARGIN = 0.19
PAD_RIGHT_MARGIN = 0.17
PAD_BOTTOM_MARGIN = 0.21
PAD_TOP_MARGIN = 0.10

MUON_CORRELATIONS = [
    "RecoVsGenMuon_eta",
    "RecoVsGenMuon_minv",
    "RecoVsGenMuon_phi",
    "RecoVsGenMuon_pt",
    "RecoVsGenMuon_pz",
    "RecoVsGenMuon_vx",
    "RecoVsGenMuon_vy",
    "RecoVsGenMuon_vz",
]
DIMUON_CORRELATIONS = [
    "RecoVsGenDimuon_eta",
    "RecoVsGenDimuon_minv",
    "RecoVsGenDimuon_phi",
    "RecoVsGenDimuon_pt",
    "RecoVsGenDimuon_pz",
    "RecoVsGenDimuon_vx",
    "RecoVsGenDimuon_vy",
    "RecoVsGenDimuon_vz",
]
MUON_RESOLUTION_TYPES = [
    ("BothEndcaps", "both_endcaps", "Both Endcaps"),
    ("NearEndcapOnly", "near_endcap_only", "Near Endcap Only"),
    ("FarEndcapOnly", "far_endcap_only", "Far Endcap Only"),
    ("NearEndcapAndBarrel", "near_endcap_and_barrel", "Near Endcap + Barrel"),
    ("Unclassified", "unclassified", "Unclassified"),
]
DIMUON_RESOLUTION_TYPES = [
    ("", "inclusive", "Inclusive"),
    ("Both-Both", "both_both", "Both Endcaps + Both Endcaps"),
    ("Near-Both", "near_both", "Near Endcap Only + Both Endcaps"),
    ("Both-Far", "both_far", "Both Endcaps + Far Endcap Only"),
    ("Near-Far", "near_far", "Near Endcap Only + Far Endcap Only"),
    ("Other", "other", "Other Topologies"),
]
DIMUON_SUMMARY_TYPES = [
    ("Both-Both", "both_both", "Both + Both"),
    ("Near-Both", "near_both", "Both + Near"),
    ("Both-Far", "both_far", "Both + Far"),
    ("Near-Far", "near_far", "Near + Far"),
    ("Other", "other", "Other"),
]
QOVERPT_SHIFT_STYLES = {
    "BothEndcaps": (ROOT.kRed, "Both Endcaps"),
    "SingleEndcap": (ROOT.kBlue, "Single Endcap"),
}
# The q/pT topology samples have very different occupancies. These fixed
# factors keep the populous categories detailed while suppressing empty-bin
# fluctuations in the rarer unclassified sample.
QOVERPT_SHIFT_REBIN = {
    "BothEndcaps": 5,
    "SingleEndcap": 5,
    "Unclassified": 20,
}
QOVERPT_REFERENCE_STYLES = [
    ("prompt_tracker_muon", ROOT.kBlack, 1, "prompt #mu, tracker+muon"),
    ("prompt_muon_only", ROOT.TColor.GetColor("#E69F00"), 1, "prompt #mu, muon-only"),
    ("displaced_muon_only", ROOT.TColor.GetColor("#009E73"), 1,
     "displaced #mu (displaced-muon reconstruction)"),
    ("standard_muon_only_displaced", ROOT.TColor.GetColor("#CC79A7"), 1,
     "displaced #mu (standard-muon reconstruction)"),
]
MUON_RESOLUTION_VARIABLES = [
    ("eta", "constrainedEta"),
    ("phi", "constrainedPhi"),
    ("pz", "constrainedPz"),
    ("vz", "constrainedVz"),
    ("pt", "constrainedPt"),
    
]
DIMUON_RESOLUTION_VARIABLES = [
    ("eta", "constrainedEta"),
    ("phi", "constrainedPhi"),
    ("pz", "constrainedPz"),
    ("vz", "constrainedVz"),
    ("pt", "constrainedPt"),
    ("minv", "constrainedMinv"),
]

MUON_SUMMARY_VARIABLES = [
    variables for variables in MUON_RESOLUTION_VARIABLES if variables[0] not in ("vx", "vy")
]
DIMUON_SUMMARY_VARIABLES = [
    variables for variables in DIMUON_RESOLUTION_VARIABLES if variables[0] not in ("vx", "vy")
]

EFFICIENCY_VARIABLES = ["pt", "pz", "eta", "phi", "vz"]
MUON_EFFICIENCY_TITLES = {
    "pt": "Generator p^{#mu}_{T} (GeV)",
    "pz": "Generator p^{#mu}_{z} (GeV)",
    "eta": "Generator #eta^{#mu}",
    "phi": "Generator #phi^{#mu}",
    "vz": "Generator v^{#mu}_{z} (cm)",
}
DIMUON_EFFICIENCY_TITLES = {
    "pt": "Generator p^{#mu#mu}_{T} (GeV)",
    "pz": "Generator p^{#mu#mu}_{z} (GeV)",
    "eta": "Generator #eta^{#mu#mu}",
    "phi": "Generator #phi^{#mu#mu}",
    "vz": "Generator v^{#mu#mu}_{z} (cm)",
}
MUON_EFFICIENCY_TYPES = [
    ("", "Inclusive", ROOT.kBlack),
    ("BothEndcaps", "Both Endcaps", ROOT.kGreen + 2),
    ("NearEndcapOnly", "Near Endcap Only", ROOT.kViolet + 1),
    ("FarEndcapOnly", "Far Endcap Only", ROOT.kCyan + 2),
    ("NearEndcapAndBarrel", "Near Endcap + Barrel", ROOT.kBlue + 1),
    ("Unclassified", "Unclassified", ROOT.kGray + 2),
]
DIMUON_EFFICIENCY_TYPES = [
    ("", "Inclusive", ROOT.kBlack),
    ("Both-Both", "Both Endcaps + Both Endcaps", ROOT.kGreen + 2),
    ("Near-Both", "Both Endcaps + Near Endcap Only", ROOT.kViolet + 1),
    ("Both-Far", "Both Endcaps + Far Endcap Only", ROOT.kCyan + 2),
    ("Near-Far", "Near Endcap Only + Far Endcap Only", ROOT.kBlue + 1),
    ("Other", "Other topologies", ROOT.kGray + 2),
]


def muon_resolution_names(muon_type, constrained):
  variable_index = 1 if constrained else 0
  return [f"MuonResolution{muon_type}_{variables[variable_index]}" for variables in MUON_RESOLUTION_VARIABLES]


MUON_RESOLUTION_CANVASES = [{
    "names": muon_resolution_names(muon_type, constrained),
    "canvas_name": f"canvas_muon_resolutions_{slug}{'_constrained' if constrained else ''}",
    "canvas_title": f"{display_name} Muon Resolutions ({'Constrained' if constrained else 'Unconstrained'})",
    "output_name": f"muon_resolutions_{slug}{'_constrained' if constrained else ''}.pdf",
} for muon_type, slug, display_name in MUON_RESOLUTION_TYPES for constrained in (False, True)]


def dimuon_resolution_names(dimuon_type, constrained):
  variable_index = 1 if constrained else 0
  return [f"DimuonResolution{dimuon_type}_{variables[variable_index]}" for variables in DIMUON_RESOLUTION_VARIABLES]


DIMUON_RESOLUTION_CANVASES = [{
    "names": dimuon_resolution_names(dimuon_type, constrained),
    "canvas_name": f"canvas_dimuon_resolutions_{slug}{'_constrained' if constrained else ''}",
    "canvas_title": f"{display_name} Dimuon Resolutions ({'Constrained' if constrained else 'Unconstrained'})",
    "output_name": f"dimuon_resolutions_{slug}{'_constrained' if constrained else ''}.pdf",
} for dimuon_type, slug, display_name in DIMUON_RESOLUTION_TYPES for constrained in (False, True)]

SUMMARY_CANVAS_SPECS = [
    {
        "object_name": "Muon",
        "categories": MUON_RESOLUTION_TYPES,
        "variables": MUON_SUMMARY_VARIABLES,
        "histogram_prefix": "MuonResolution",
        "canvas_name": "canvas_muon_scale_resolution_summary",
        "canvas_title": "Muon Scale and Resolution Summary",
        "output_name": "muon_scale_resolution_summary.pdf",
    },
    {
        "object_name": "Dimuon",
        # The inclusive category would duplicate the same events represented
        # by these mutually exclusive labeled topology bins.
        "categories": DIMUON_SUMMARY_TYPES,
        "variables": DIMUON_SUMMARY_VARIABLES,
        "histogram_prefix": "DimuonResolution",
        "canvas_name": "canvas_dimuon_scale_resolution_summary",
        "canvas_title": "Dimuon Scale and Resolution Summary",
        "output_name": "dimuon_scale_resolution_summary.pdf",
    },
]

TITLES = {
    "RecoVsGenMuon_eta": "Muon #eta",
    "RecoVsGenMuon_minv": "Muon m_{inv} (GeV)",
    "RecoVsGenMuon_phi": "Muon #phi",
    "RecoVsGenMuon_pt": "Muon p_{T} (GeV)",
    "RecoVsGenMuon_pz": "Muon p_{z} (GeV)",
    "RecoVsGenMuon_vx": "Muon v_{x} (cm)",
    "RecoVsGenMuon_vy": "Muon v_{y} (cm)",
    "RecoVsGenMuon_vz": "Muon v_{z} (cm)",
    "RecoVsGenDimuon_pt": "J/#psi p_{T} (GeV)",
    "RecoVsGenDimuon_eta": "J/#psi #eta",
    "RecoVsGenDimuon_phi": "J/#psi #phi",
    "RecoVsGenDimuon_minv": "J/#psi m_{inv} (GeV)",
    "RecoVsGenDimuon_pz": "J/#psi p_{z} (GeV)",
    "RecoVsGenDimuon_vx": "J/#psi v_{x} (cm)",
    "RecoVsGenDimuon_vy": "J/#psi v_{y} (cm)",
    "RecoVsGenDimuon_vz": "J/#psi v_{z} (cm)",
}

# These labels mirror the quantities filled in ShiftHistogramsFiller::FillResolutionPlots.
RESOLUTION_TITLES = {}

MUON_RESOLUTION_LABELS = {
    "eta": "#eta",
    "phi": "#phi",
    "pt": "p_{T}",
    "pz": "p_{z}",
    "vx": "v_{x}",
    "vy": "v_{y}",
    "vz": "v_{z}",
}
for muon_type, _, _ in MUON_RESOLUTION_TYPES:
  for (variable, constrained_variable) in MUON_RESOLUTION_VARIABLES:
    quantity = MUON_RESOLUTION_LABELS[variable]
    RESOLUTION_TITLES[f"MuonResolution{muon_type}_{variable}"] = (
        f"({quantity}^{{reco}} - {quantity}^{{gen}}) / {quantity}^{{gen}}")
    RESOLUTION_TITLES[f"MuonResolution{muon_type}_{constrained_variable}"] = (
        f"({quantity}^{{reco, constrained}} - {quantity}^{{gen}}) / {quantity}^{{gen}}")

DIMUON_RESOLUTION_LABELS = {
    "eta": "#eta_{#mu#mu}",
    "minv": "m_{#mu#mu}",
    "phi": "#phi_{#mu#mu}",
    "pt": "p_{T,#mu#mu}",
    "pz": "p_{z,#mu#mu}",
    "vx": "v_{x,#mu#mu}",
    "vy": "v_{y,#mu#mu}",
    "vz": "v_{z,#mu#mu}",
}
for dimuon_type, _, _ in DIMUON_RESOLUTION_TYPES:
  for variable, constrained_variable in DIMUON_RESOLUTION_VARIABLES:
    quantity = DIMUON_RESOLUTION_LABELS[variable]
    RESOLUTION_TITLES[f"DimuonResolution{dimuon_type}_{variable}"] = (
        f"({quantity}^{{reco}} - {quantity}^{{gen}}) / {quantity}^{{gen}}")
    RESOLUTION_TITLES[f"DimuonResolution{dimuon_type}_{constrained_variable}"] = (
        f"({quantity}^{{reco, constrained}} - {quantity}^{{gen}}) / {quantity}^{{gen}}")

RESOLUTION_X_RANGES = {}
for resolution_canvas in MUON_RESOLUTION_CANVASES:
  for name in resolution_canvas["names"]:
    if name.endswith(("_vx", "_vy", "_constrainedVx", "_constrainedVy")):
      RESOLUTION_X_RANGES[name] = (-5000.0, 5000.0)
for resolution_canvas in DIMUON_RESOLUTION_CANVASES:
  for name in resolution_canvas["names"]:
    if name.endswith(("_vx", "_vy", "_constrainedVx", "_constrainedVy")):
      RESOLUTION_X_RANGES[name] = (-5000.0, 5000.0)

HISTOGRAM_FILE_PATTERN = re.compile(r"v([1-9][0-9]*)_([0-9a-f]{7,40}(?:-dirty-[0-9a-f]{8})?)")


def histogram_version(path):
  match = HISTOGRAM_FILE_PATTERN.fullmatch(os.path.basename(os.path.dirname(path)))
  if not match:
    raise ValueError(f"histogram file '{path}' must be located in 'vN_<hash>/histograms.root'")
  return int(match.group(1)), match.group(2)


def latest_histogram_file(histograms_dir):
  candidates = []
  for input_path in glob.glob(os.path.join(histograms_dir, "v*_*", "histograms.root")):
    try:
      version, provenance_tag = histogram_version(input_path)
    except ValueError:
      continue
    candidates.append((version, os.path.basename(os.path.dirname(input_path)), input_path, provenance_tag))

  if not candidates:
    raise RuntimeError(f"No 'vN_<hash>/histograms.root' files found in '{histograms_dir}'")

  latest_version = max(candidate[0] for candidate in candidates)
  latest_candidates = [candidate for candidate in candidates if candidate[0] == latest_version]
  if len(latest_candidates) != 1:
    names = ", ".join(sorted(candidate[1] for candidate in latest_candidates))
    raise RuntimeError(f"Multiple histogram files claim version v{latest_version}: {names}")

  version, _, path, provenance_tag = latest_candidates[0]
  return path, version, provenance_tag


def parse_rebin_specs(specs, dimensions):
  result = {}
  for spec in specs:
    try:
      name, factors_text = spec.split("=", 1)
      factors = tuple(int(value) for value in factors_text.split(","))
    except ValueError as error:
      raise argparse.ArgumentTypeError(f"invalid rebin specification '{spec}'") from error
    if len(factors) == 1 and dimensions == 2:
      factors *= 2
    if len(factors) != dimensions or any(factor < 1 for factor in factors):
      raise argparse.ArgumentTypeError(f"'{spec}' needs {dimensions} positive integer factor(s)")
    result[name] = factors
  return result


def set_axes_titles(hist, x_title, y_title):
  hist.SetTitle("")
  hist.GetXaxis().SetTitle(x_title)
  hist.GetYaxis().SetTitle(y_title)
  for axis in (hist.GetXaxis(), hist.GetYaxis()):
    axis.SetLabelSize(0.050)
    axis.SetLabelOffset(0.012)
    axis.SetTitleSize(0.060)
    axis.SetNdivisions(505)
  hist.GetXaxis().SetTitleOffset(1.15)
  hist.GetYaxis().SetTitleOffset(1.35)


def set_pad_margins(right_margin=0.06):
  ROOT.gPad.SetLeftMargin(PAD_LEFT_MARGIN)
  ROOT.gPad.SetRightMargin(right_margin)
  ROOT.gPad.SetBottomMargin(PAD_BOTTOM_MARGIN)
  ROOT.gPad.SetTopMargin(PAD_TOP_MARGIN)


def annotate_entries(hist):
  label = ROOT.TLatex()
  label.SetNDC(True)
  label.SetTextFont(42)
  label.SetTextSize(0.040)
  label.SetTextAlign(31)
  label.DrawLatex(0.88, 0.92, f"Entries: {hist.GetEntries():.0f}")
  return label


def occupied_range_2d(hist, margin_fraction=0.20):
  x_bins = []
  y_bins = []
  for x_bin in range(1, hist.GetNbinsX() + 1):
    for y_bin in range(1, hist.GetNbinsY() + 1):
      # Include bins whose weighted contents cancel to zero but which still
      # have a non-zero statistical uncertainty.
      if hist.GetBinContent(x_bin, y_bin) != 0 or hist.GetBinError(x_bin, y_bin) != 0:
        x_bins.append(x_bin)
        y_bins.append(y_bin)
  if not x_bins:
    return None

  def with_margin(axis, first_bin, last_bin):
    lower = axis.GetBinLowEdge(first_bin)
    upper = axis.GetBinUpEdge(last_bin)
    span = upper - lower
    # A single occupied bin still gets a useful visible neighborhood.
    margin = margin_fraction * max(span, axis.GetBinWidth(first_bin))
    return max(axis.GetXmin(), lower - margin), min(axis.GetXmax(), upper + margin)

  return (
      with_margin(hist.GetXaxis(), min(x_bins), max(x_bins)),
      with_margin(hist.GetYaxis(), min(y_bins), max(y_bins)),
  )


def draw_diag_line(hist):
  x_axis = hist.GetXaxis()
  y_axis = hist.GetYaxis()
  xmin, xmax = x_axis.GetBinLowEdge(x_axis.GetFirst()), x_axis.GetBinUpEdge(x_axis.GetLast())
  ymin, ymax = y_axis.GetBinLowEdge(y_axis.GetFirst()), y_axis.GetBinUpEdge(y_axis.GetLast())
  lo, hi = max(xmin, ymin), min(xmax, ymax)
  if lo >= hi:
    return None
  line = ROOT.TLine(lo, lo, hi, hi)
  line.SetLineColor(ROOT.kRed + 1)
  line.SetLineStyle(2)
  line.SetLineWidth(2)
  line.Draw("same")
  return line


def draw_zero_line(hist):
  ROOT.gPad.Update()
  line = ROOT.TLine(0.0, ROOT.gPad.GetUymin(), 0.0, ROOT.gPad.GetUymax())
  line.SetLineColor(ROOT.kRed + 1)
  line.SetLineStyle(2)
  line.SetLineWidth(2)
  # DrawClone gives each pad an independent ROOT-owned line. Without the clone,
  # PyROOT can discard the earlier transient lines while processing later pads.
  return line.DrawClone("same")


def double_sided_crystal_ball(values, parameters):
  # PyROOT exposes a pointer-like parameter buffer whose apparent length can be
  # larger than the TF1 parameter count, so access only the seven values used.
  normalization, mean, sigma = parameters[0], parameters[1], parameters[2]
  alpha_left, n_left = parameters[3], parameters[4]
  alpha_right, n_right = parameters[5], parameters[6]
  if sigma <= 0:
    return 0.0
  t = (values[0] - mean) / sigma
  if -alpha_left <= t <= alpha_right:
    return normalization * math.exp(-0.5 * t * t)
  if t < -alpha_left:
    b = n_left / alpha_left - alpha_left
    # Combine the two powers in log space.  Evaluating them separately can
    # overflow during the minimizer's parameter scans even though their
    # product is finite.
    exponent = -0.5 * alpha_left * alpha_left + n_left * (math.log(n_left / alpha_left) - math.log(b - t))
    return normalization * math.exp(exponent)
  b = n_right / alpha_right - alpha_right
  exponent = -0.5 * alpha_right * alpha_right + n_right * (math.log(n_right / alpha_right) - math.log(b + t))
  return normalization * math.exp(exponent)


def left_sided_crystal_ball(values, parameters):
  normalization, mean, sigma = parameters[0], parameters[1], parameters[2]
  alpha, n = parameters[3], parameters[4]
  if sigma <= 0:
    return 0.0
  t = (values[0] - mean) / sigma
  if t >= -alpha:
    return normalization * math.exp(-0.5 * t * t)
  b = n / alpha - alpha
  exponent = -0.5 * alpha * alpha + n * (math.log(n / alpha) - math.log(b - t))
  return normalization * math.exp(exponent)


def right_sided_crystal_ball(values, parameters):
  normalization, mean, sigma = parameters[0], parameters[1], parameters[2]
  alpha, n = parameters[3], parameters[4]
  if sigma <= 0:
    return 0.0
  t = (values[0] - mean) / sigma
  if t <= alpha:
    return normalization * math.exp(-0.5 * t * t)
  b = n / alpha - alpha
  exponent = -0.5 * alpha * alpha + n * (math.log(n / alpha) - math.log(b + t))
  return normalization * math.exp(exponent)


def fit_result_is_reliable(fit_result):
  result = fit_result.Get()
  return (int(fit_result) == 0 and result and result.IsValid() and result.CovMatrixStatus() == 3)


def configure_core_parameters(fit, hist, mean_seed, sigma_seed, fit_low, fit_high):
  bin_width = hist.GetXaxis().GetBinWidth(1)
  fit.SetParLimits(0, 0.0, max(10.0 * hist.GetMaximum(), 1.0))
  fit.SetParLimits(1, fit_low, fit_high)
  fit.SetParLimits(2, 0.1 * bin_width, fit_high - fit_low)
  fit.SetParameter(0, hist.GetMaximum())
  fit.SetParameter(1, mean_seed)
  fit.SetParameter(2, sigma_seed)


def fit_quality(fit_result):
  result = fit_result.Get()
  return result.Chi2() / max(result.Ndf(), 1)


def fit_resolution(hist, name):
  if hist.GetEntries() < 10 or hist.GetMaximum() <= 0:
    return None, "none"

  x_axis = hist.GetXaxis()
  fit_low, fit_high = x_axis.GetXmin(), x_axis.GetXmax()
  median, central_68_half_width = robust_resolution_summary(hist)
  bin_width = x_axis.GetBinWidth(1)
  sigma_seed = max(central_68_half_width, bin_width)

  # First fit only the populated core. This supplies stable mean and width
  # seeds before the tail parameters are introduced.
  core_low = max(fit_low, median - 2.0 * sigma_seed)
  core_high = min(fit_high, median + 2.0 * sigma_seed)
  gaussian = ROOT.TF1(f"fit_gaussian_{name}", "gaus", core_low, core_high)
  configure_core_parameters(gaussian, hist, median, sigma_seed, core_low, core_high)
  gaussian_result = hist.Fit(gaussian, "QRS0N")
  if int(gaussian_result) == 0:
    mean_seed = gaussian.GetParameter(1)
    sigma_seed = abs(gaussian.GetParameter(2))
  else:
    mean_seed = median

  candidates = []

  double_cb = ROOT.TF1(f"fit_double_cb_{name}", double_sided_crystal_ball, fit_low, fit_high, 7)
  double_cb.SetParNames("N", "#mu", "#sigma", "#alpha_{L}", "n_{L}", "#alpha_{R}", "n_{R}")
  configure_core_parameters(double_cb, hist, mean_seed, sigma_seed, fit_low, fit_high)
  double_cb.SetParameter(3, 1.5)
  double_cb.SetParameter(4, 3.0)
  double_cb.SetParameter(5, 1.5)
  double_cb.SetParameter(6, 3.0)
  for index in (3, 5):
    double_cb.SetParLimits(index, 0.05, 10.0)
  for index in (4, 6):
    double_cb.SetParLimits(index, 1.01, 100.0)
  double_result = hist.Fit(double_cb, "QRS0N")
  if fit_result_is_reliable(double_result):
    candidates.append((0, fit_quality(double_result), double_cb, "DSCB"))

  for priority, callback, model_name in (
      (1, left_sided_crystal_ball, "left-tail CB"),
      (1, right_sided_crystal_ball, "right-tail CB"),
  ):
    model_id = model_name.replace("-", "_").replace(" ", "_")
    fit = ROOT.TF1(f"fit_{model_id}_{name}", callback, fit_low, fit_high, 5)
    fit.SetParNames("N", "#mu", "#sigma", "#alpha", "n")
    configure_core_parameters(fit, hist, mean_seed, sigma_seed, fit_low, fit_high)
    fit.SetParameter(3, 1.5)
    fit.SetParameter(4, 3.0)
    fit.SetParLimits(3, 0.05, 10.0)
    fit.SetParLimits(4, 1.01, 100.0)
    fit_result = hist.Fit(fit, "QRS0N")
    if fit_result_is_reliable(fit_result):
      candidates.append((priority, fit_quality(fit_result), fit, model_name))

  if fit_result_is_reliable(gaussian_result):
    candidates.append((2, fit_quality(gaussian_result), gaussian, "Gaussian"))

  if not candidates:
    return None, "none converged"

  # Prefer the most expressive reliable family; use fit quality to select
  # between the equally complex left- and right-tail alternatives.
  _, _, fit, model_name = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
  fit.SetLineColor(ROOT.kBlue + 1)
  fit.SetLineWidth(2)
  fit.SetNpx(500)
  fit.Draw("same")
  return fit, model_name


def robust_resolution_summary(hist):
  probabilities = array("d", (0.16, 0.50, 0.84))
  quantiles = array("d", (0.0, 0.0, 0.0))
  hist.GetQuantiles(len(probabilities), quantiles, probabilities)
  return quantiles[1], 0.5 * (quantiles[2] - quantiles[0])


def draw_scale_resolution_summary(canvas, spec, input_file):
  """Compare unconstrained and constrained direct summaries by topology."""
  objects = []
  categories = spec["categories"]
  legend_graphs = None
  for pad_index, variables in enumerate(spec["variables"], 1):
    canvas.cd(pad_index)
    ROOT.gPad.SetLeftMargin(0.20)
    ROOT.gPad.SetRightMargin(0.06)
    ROOT.gPad.SetBottomMargin(0.32)
    ROOT.gPad.SetTopMargin(0.16)

    frame = ROOT.TH1D(
        f"frame_{spec['canvas_name']}_{variables[0]}",
        "",
        len(categories),
        0.5,
        len(categories) + 0.5,
    )
    frame.SetDirectory(0)
    frame.SetStats(False)
    for bin_index, (_, _, display_name) in enumerate(categories, 1):
      frame.GetXaxis().SetBinLabel(bin_index, display_name)

    summaries = {
        "unconstrained": (array("d"), array("d"), array("d"), array("d")),
        "constrained": (array("d"), array("d"), array("d"), array("d")),
    }
    for bin_index, (category, _, _) in enumerate(categories, 1):
      for strategy, variable, x_offset in (
          ("unconstrained", variables[0], -0.12),
          ("constrained", variables[1], 0.12),
      ):
        name = f"{spec['histogram_prefix']}{category}_{variable}"
        hist = input_file.Get(f"resolution/{name}")
        if not hist:
          print(f"Warning: histogram '{name}' was not found")
          continue
        if hist.Integral(1, hist.GetNbinsX()) <= 0.0:
          print(f"Warning: histogram '{name}' has no in-range entries")
          continue

        scale = 1.0 + hist.GetMean()
        resolution = hist.GetStdDev()
        if not math.isfinite(scale) or not math.isfinite(resolution):
          print(f"Warning: histogram '{name}' has a non-finite direct summary")
          continue
        x_values, y_values, x_errors, y_errors = summaries[strategy]
        x_values.append(bin_index + x_offset)
        y_values.append(scale)
        x_errors.append(0.0)
        y_errors.append(resolution)

    reference_value = 1.0
    all_points = [
        (value, error)
        for _, y_values, _, y_errors in summaries.values()
        for value, error in zip(y_values, y_errors)
    ]
    lows = [value - error for value, error in all_points] + [reference_value]
    highs = [value + error for value, error in all_points] + [reference_value]
    y_min = min(lows)
    y_max = max(highs)
    span = max(y_max - y_min, 0.05 * max(abs(y_min), abs(y_max), 1.0))
    frame.SetMinimum(y_min - 0.18 * span)
    frame.SetMaximum(y_max + 0.18 * span)
    set_axes_titles(
        frame,
        f"{spec['object_name']} topology",
        "RECO / GEN scale",
    )
    frame.GetXaxis().SetLabelSize(0.050)
    frame.GetXaxis().SetLabelOffset(0.015)
    frame.GetXaxis().LabelsOption("d")
    frame.GetXaxis().SetTitleOffset(2.45)
    frame.GetYaxis().SetTitleOffset(1.45)
    frame.Draw("AXIS")

    unity = ROOT.TLine(0.5, reference_value, len(categories) + 0.5, reference_value)
    unity.SetLineColor(ROOT.kBlack)
    unity.SetLineStyle(2)
    unity.SetLineWidth(2)
    unity.Draw("same")

    graphs = []
    for strategy, color, marker_style in (
        ("unconstrained", ROOT.kBlue + 1, 20),
        ("constrained", ROOT.kOrange + 7, 21),
    ):
      x_values, y_values, x_errors, y_errors = summaries[strategy]
      graph = ROOT.TGraphErrors(len(x_values), x_values, y_values, x_errors, y_errors)
      graph.SetName(f"graph_{spec['canvas_name']}_{variables[0]}_{strategy}")
      graph.SetMarkerStyle(marker_style)
      graph.SetMarkerSize(1.15)
      graph.SetMarkerColor(color)
      graph.SetLineColor(color)
      graph.SetLineWidth(2)
      graph.Draw("P SAME")
      graphs.append(graph)
    if legend_graphs is None:
      legend_graphs = graphs

    quantity = (MUON_RESOLUTION_LABELS if spec["object_name"] == "Muon"
                else DIMUON_RESOLUTION_LABELS)[variables[0]]
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.060)
    label.DrawLatex(0.22, 0.89, quantity)
    objects.extend((frame, unity, *graphs, label))

  canvas.cd(len(spec["variables"]) + 1)
  ROOT.gPad.SetLeftMargin(0.06)
  ROOT.gPad.SetRightMargin(0.06)
  ROOT.gPad.SetBottomMargin(0.06)
  ROOT.gPad.SetTopMargin(0.06)
  legend = ROOT.TLegend(0.10, 0.30, 0.90, 0.70)
  legend.SetBorderSize(0)
  legend.SetFillStyle(0)
  legend.SetTextFont(42)
  legend.SetTextSize(0.060)
  legend.SetHeader("Scale and resolution", "C")
  legend.AddEntry(legend_graphs[0], "Unconstrained: 1 + mean #pm RMS", "pe")
  legend.AddEntry(legend_graphs[1], "Constrained: 1 + mean #pm RMS", "pe")
  legend.Draw()
  objects.append(legend)

  canvas.Update()
  return objects


def set_resolution_y_range(hist, fit, margin_fraction=0.15):
  x_axis = hist.GetXaxis()
  y_min = 0.0
  y_max = 0.0
  for bin_index in range(x_axis.GetFirst(), x_axis.GetLast() + 1):
    content = hist.GetBinContent(bin_index)
    error = hist.GetBinError(bin_index)
    y_min = min(y_min, content - error)
    y_max = max(y_max, content + error)

  if fit:
    y_min = min(y_min, fit.GetMinimum(fit.GetXmin(), fit.GetXmax()))
    y_max = max(y_max, fit.GetMaximum(fit.GetXmin(), fit.GetXmax()))

  span = max(y_max - y_min, 1.0)
  hist.SetMinimum(y_min - margin_fraction * span if y_min < 0.0 else 0.0)
  hist.SetMaximum(y_max + margin_fraction * span)
  ROOT.gPad.Modified()
  ROOT.gPad.Update()


def draw_2d(canvas, names, input_file, rebin_factors):
  objects = []
  for pad, name in enumerate(names, 1):
    canvas.cd(pad)
    set_pad_margins(PAD_RIGHT_MARGIN)
    hist = input_file.Get(f"correlations/{name}")
    if not hist:
      print(f"Warning: histogram '{name}' was not found")
      continue
    x_factor, y_factor = rebin_factors.get(name, (1, 1))
    if x_factor > 1 or y_factor > 1:
      hist.Rebin2D(x_factor, y_factor)
    set_axes_titles(hist, f"RECO {TITLES[name]}", f"GEN {TITLES[name]}")
    visible_range = occupied_range_2d(hist)
    if visible_range:
      hist.GetXaxis().SetRangeUser(*visible_range[0])
      hist.GetYaxis().SetRangeUser(*visible_range[1])
    hist.Draw("COLZ")
    objects.extend(filter(None, (draw_diag_line(hist), annotate_entries(hist))))
  canvas.Update()
  return objects


def draw_resolutions(canvas, names, input_file, rebin_factor):
  objects = []
  for pad, name in enumerate(names, 1):
    canvas.cd(pad)
    set_pad_margins()
    hist = input_file.Get(f"resolution/{name}")
    if not hist:
      print(f"Warning: histogram '{name}' was not found")
      continue
    if rebin_factor > 1:
      hist.Rebin(rebin_factor)
    set_axes_titles(hist, RESOLUTION_TITLES[name], "Entries")
    if name in RESOLUTION_X_RANGES:
      hist.GetXaxis().SetRangeUser(*RESOLUTION_X_RANGES[name])
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.Draw("E1")
    if hist.Integral(1, hist.GetNbinsX()) <= 0.0:
      label = ROOT.TLatex()
      label.SetNDC(True)
      label.SetTextFont(42)
      label.SetTextSize(0.040)
      label.DrawLatex(PAD_LEFT_MARGIN, 0.92, "No in-range entries")
      objects.extend(filter(None, (draw_zero_line(hist), label, annotate_entries(hist))))
      continue
    fit, fit_model = fit_resolution(hist, name)
    set_resolution_y_range(hist, fit)
    median, central_68_half_width = robust_resolution_summary(hist)
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.034)
    text_x = 0.24
    if fit:
      label.DrawLatex(text_x, 0.85, f"fit #mu = {fit.GetParameter(1):.2g}")
      label.DrawLatex(text_x, 0.80, f"fit #sigma = {abs(fit.GetParameter(2)):.2g}")
    label.DrawLatex(text_x, 0.75, f"median = {median:.2g}")
    label.DrawLatex(text_x, 0.70, f"#sigma_{{68}} = {central_68_half_width:.2g}")
    fit_description = f"{fit_model}, converged" if fit else fit_model
    label.DrawLatex(PAD_LEFT_MARGIN, 0.92, fit_description)
    objects.extend(filter(None, (fit, draw_zero_line(hist), label, annotate_entries(hist))))
  canvas.Update()
  return objects


def load_qoverpt_reference_graphs(path):
  """Load and unit-normalize the approximate CMS-DP raster digitization."""
  points = {key: (array("d"), array("d")) for key, _, _, _ in QOVERPT_REFERENCE_STYLES}

  with open(path, encoding="utf-8") as data_file:
    for line in data_file:
      if not line.strip() or line.startswith("#"):
        continue
      values = line.split()
      if len(values) != 5:
        raise ValueError(f"malformed q/pT reference row: {line.rstrip()}")
      x_value = float(values[0])
      for column, (key, _, _, _) in enumerate(QOVERPT_REFERENCE_STYLES, 1):
        y_value = float(values[column])
        if math.isfinite(y_value):
          points[key][0].append(x_value)
          points[key][1].append(y_value)

  graphs = {}
  for key, color, line_style, _ in QOVERPT_REFERENCE_STYLES:
    x_values, y_values = points[key]
    # The digitized values are event counts sampled at the original 0.02-wide
    # bin centers.  Missing values were below the visible logarithmic frame;
    # normalize the available digitized area over the displayed x range.
    bin_width = min(
        (x_values[index] - x_values[index - 1]
         for index in range(1, len(x_values))
         if x_values[index] > x_values[index - 1]),
        default=0.02,
    )
    area = sum(y_values) * bin_width
    if area <= 0.0:
      raise ValueError(f"q/pT reference curve '{key}' has no positive area")
    for index in range(len(y_values)):
      y_values[index] /= area
    graph = ROOT.TGraph(len(x_values), x_values, y_values)
    graph.SetName(f"cms_dp_2015_015_{key}")
    graph.SetLineColor(color)
    graph.SetLineStyle(line_style)
    graph.SetLineWidth(2)
    graphs[key] = graph
  return graphs


def draw_qoverpt_comparison(canvas, input_file, reference_path):
  """Overlay the SHIFT fits and the four CMS-DP-2015-015 references."""
  canvas.cd()
  canvas.SetLogy(True)
  canvas.SetLeftMargin(0.15)
  canvas.SetRightMargin(0.04)
  canvas.SetBottomMargin(0.17)
  canvas.SetTopMargin(0.34)

  objects = []
  shift_curves = []
  for muon_type, (color, display_name) in QOVERPT_SHIFT_STYLES.items():
    for constrained in (False,):
      variable = "constrainedQOverPt" if constrained else "qOverPt"
      name = f"MuonResolution{muon_type}_{variable}"
      source_hist = input_file.Get(f"resolution/{name}")
      if not source_hist:
        print(f"Warning: histogram '{name}' was not found")
        continue
      hist = source_hist.Clone(f"draw_{name}")
      hist.SetDirectory(0)
      rebin_factor = QOVERPT_SHIFT_REBIN[muon_type]
      source_bin_count = hist.GetNbinsX()
      hist = hist.Rebin(rebin_factor, f"draw_{name}_rebin{rebin_factor}")
      hist.SetDirectory(0)
      print(f"q/pT rebin {name}: factor={rebin_factor}, "
            f"bins={source_bin_count}->{hist.GetNbinsX()}, "
            f"width={hist.GetBinWidth(1):g}")
      area = hist.Integral(1, hist.GetNbinsX(), "width")
      if area <= 0.0:
        print(f"Warning: histogram '{name}' has no entries in [-2, 2]")
        continue
      hist.Scale(1.0 / area)
      x_values = array("d", (hist.GetBinCenter(index) for index in range(1, hist.GetNbinsX() + 1)))
      y_values = array("d", (hist.GetBinContent(index) for index in range(1, hist.GetNbinsX() + 1)))
      curve = ROOT.TGraph(hist.GetNbinsX(), x_values, y_values)
      curve.SetName(f"curve_{name}")
      curve.SetLineColor(color)
      curve.SetLineStyle(2)
      curve.SetLineWidth(4)
      shift_curves.append((muon_type, curve, display_name))

  reference_graphs = load_qoverpt_reference_graphs(reference_path)
  shift_y_max = max(
      (curve.GetY()[index] for _, curve, _ in shift_curves for index in range(curve.GetN())),
      default=1.0,
  )
  reference_y_max = max(
      (reference_graphs[key].GetY()[index]
       for key in reference_graphs
       for index in range(reference_graphs[key].GetN())),
      default=1.0,
  )
  y_max = max(shift_y_max, reference_y_max)
  frame = canvas.DrawFrame(-2.0, 5.0e-3, 2.0, max(10.0, 3.0 * y_max))
  set_axes_titles(
      frame,
      "[(q/p_{T})_{reco} - (q/p_{T})_{gen}] / (q/p_{T})_{gen}",
      "Unit-normalized density",
  )
  for axis in (frame.GetXaxis(), frame.GetYaxis()):
    axis.SetLabelSize(0.032)
    axis.SetTitleSize(0.040)
  frame.GetXaxis().SetTitleOffset(1.25)
  frame.GetYaxis().SetTitleOffset(1.45)
  frame.GetXaxis().SetNdivisions(510)
  objects.append(frame)

  for key, _, _, _ in QOVERPT_REFERENCE_STYLES:
    reference_graphs[key].Draw("L SAME")
  for _, curve, _ in shift_curves:
    curve.Draw("L SAME")

  legend = ROOT.TLegend(0.08, 0.69, 0.96, 0.96)
  legend.SetNColumns(2)
  legend.SetBorderSize(0)
  legend.SetFillColor(ROOT.kWhite)
  legend.SetFillStyle(1001)
  legend.SetTextFont(42)
  legend.SetTextSize(0.027)
  shift_by_type = {muon_type: (curve, label) for muon_type, curve, label in shift_curves}
  cms_entries = [(reference_graphs[key], label) for key, _, _, label in QOVERPT_REFERENCE_STYLES]
  shift_entries = [shift_by_type[key] for key in (
      "BothEndcaps", "SingleEndcap", "Unclassified"
  ) if key in shift_by_type]
  legend.AddEntry(0, "CMS", "")
  legend.AddEntry(0, "SHIFT", "")
  for row in range(max(len(cms_entries), len(shift_entries))):
    for entries in (cms_entries, shift_entries):
      if row < len(entries):
        curve, label = entries[row]
        legend.AddEntry(curve, label, "l")
      else:
        legend.AddEntry(0, "", "")
  legend.Draw()

  canvas.RedrawAxis()
  canvas.Update()
  objects.extend([*[curve for _, curve, _ in shift_curves], *reference_graphs.values(), legend])
  return objects


def efficiency_prefix(object_name, category):
  return f"{object_name}{category}Efficiency"


def draw_efficiencies(canvas, input_file, object_name, categories):
  objects = []
  legend_entries = []
  is_dimuon = object_name == "ShiftDimuonVertex"
  efficiency_titles = DIMUON_EFFICIENCY_TITLES if is_dimuon else MUON_EFFICIENCY_TITLES
  for pad_index, variable in enumerate(EFFICIENCY_VARIABLES, 1):
    canvas.cd(pad_index)
    ROOT.gPad.SetLeftMargin(0.25 if is_dimuon else 0.22)
    ROOT.gPad.SetRightMargin(0.07)
    ROOT.gPad.SetBottomMargin(0.23)
    ROOT.gPad.SetTopMargin(0.08)
    ROOT.gPad.SetLogy(is_dimuon)
    curves = []
    frame = None
    for category, label, color in categories:
      prefix = efficiency_prefix(object_name, category)
      passed = input_file.Get(f"efficiency/{prefix}_{variable}_pass")
      total = input_file.Get(f"efficiency/{prefix}_{variable}_total")
      if not passed or not total:
        print(f"Warning: efficiency pair '{prefix}_{variable}' was not found")
        continue
      if not ROOT.TEfficiency.CheckConsistency(passed, total):
        print(f"Warning: inconsistent efficiency pair '{prefix}_{variable}'")
        continue
      if frame is None:
        x_axis = total.GetXaxis()
        y_min, y_max = (1.e-3, 1.3) if is_dimuon else (0.0, 1.08)
        frame = ROOT.gPad.DrawFrame(x_axis.GetXmin(), y_min, x_axis.GetXmax(), y_max)
        set_axes_titles(frame, efficiency_titles[variable], "Reconstruction efficiency")
        if is_dimuon:
          frame.GetYaxis().SetTitleOffset(1.10)
        objects.append(frame)
        for efficiency_value in ((1.0, 0.1, 0.01) if is_dimuon else (1.0, 0.5)):
          line = ROOT.TLine(x_axis.GetXmin(), efficiency_value, x_axis.GetXmax(), efficiency_value)
          line.SetLineColor(ROOT.kGray + 1)
          line.SetLineStyle(2 if efficiency_value == 1.0 else 3)
          line.SetLineWidth(1)
          line.Draw("same")
          objects.append(line)
      efficiency = ROOT.TEfficiency(passed, total)
      efficiency.SetName(f"{prefix}_{variable}_efficiency")
      efficiency.SetStatisticOption(ROOT.TEfficiency.kFCP)
      graph = efficiency.CreateGraph()
      graph.SetLineColor(color)
      graph.SetMarkerColor(color)
      graph.SetMarkerStyle(20)
      graph.SetMarkerSize(0.75)
      graph.Draw("P SAME")
      curves.append((efficiency, graph, label))
      if pad_index == 1:
        # Include underflow and overflow so this is the efficiency for the
        # complete selected truth sample, independent of the displayed range.
        first_bin = 0
        last_bin = total.GetNbinsX() + 1
        passed_count = passed.Integral(first_bin, last_bin)
        total_count = total.Integral(first_bin, last_bin)
        total_efficiency = passed_count / total_count if total_count else 0.0
        legend_label = f"{label} ( #varepsilon_{{tot}} = {100.0 * total_efficiency:.2g}% ) "
        legend_entries.append((graph, legend_label))
    if not curves:
      continue
    objects.extend(item for efficiency, graph, _ in curves for item in (efficiency, graph))

  # Reserve the pad immediately after the last variable entirely for the
  # legend so the layout remains correct when variables are added or removed.
  canvas.cd(len(EFFICIENCY_VARIABLES) + 1)
  ROOT.gPad.SetLeftMargin(0.04)
  ROOT.gPad.SetRightMargin(0.04)
  ROOT.gPad.SetTopMargin(0.04)
  ROOT.gPad.SetBottomMargin(0.04)
  legend = ROOT.TLegend(0.06, 0.15, 0.94, 0.85)
  legend.SetNColumns(1)
  legend.SetBorderSize(0)
  legend.SetFillStyle(0)
  legend.SetTextFont(42)
  legend.SetTextSize(0.045 if is_dimuon else 0.055)
  for graph, label in legend_entries:
    legend.AddEntry(graph, label, "pe")
  legend.Draw()
  objects.append(legend)
  canvas.Update()
  return objects


def parse_arguments():
  parser = argparse.ArgumentParser(description="Plot SHIFT reconstruction diagnostics")
  parser.add_argument(
      "--input",
      help="versioned input ROOT file (default: highest-version file in --histograms-dir)",
  )
  parser.add_argument(
      "--histograms-dir",
      default=f"{PROJECT_DIR}/plots/",
      help="directory searched for histograms_vN_<hash>.root files",
  )
  parser.add_argument(
      "--output-dir",
      default=f"{PROJECT_DIR}/plots",
      help="parent directory for versioned plot directories",
  )
  parser.add_argument(
      "--rebin-2d",
      action="append",
      default=[],
      metavar="HIST=XFACTOR[,YFACTOR]",
      help="rebin one 2D histogram; one factor applies to both axes",
  )
  return parser.parse_args()


def main():
  # set batch mode
  ROOT.gROOT.SetBatch(True)

  args = parse_arguments()
  try:
    if args.input:
      input_path = args.input
      version, provenance_tag = histogram_version(input_path)
    else:
      input_path, version, provenance_tag = latest_histogram_file(args.histograms_dir)
  except (RuntimeError, ValueError) as error:
    raise SystemExit(f"error: {error}") from error

  print(f"Selected histograms v{version}_{provenance_tag}: {input_path}")
  try:
    correlation_rebin = parse_rebin_specs(args.rebin_2d, 2)
  except argparse.ArgumentTypeError as error:
    raise SystemExit(f"error: {error}") from error

  known_correlations = set(MUON_CORRELATIONS + DIMUON_CORRELATIONS)
  unknown = set(correlation_rebin) - known_correlations
  if unknown:
    raise SystemExit(f"error: unknown histogram(s) in rebin options: {', '.join(sorted(unknown))}")

  input_file = ROOT.TFile.Open(input_path, "READ")
  if not input_file or input_file.IsZombie():
    raise SystemExit(f"error: could not open input ROOT file '{input_path}'")
  output_dir = f"{args.output_dir}/v{version}_{provenance_tag}"
  os.makedirs(output_dir, exist_ok=True)
  print(f"Output directory: {output_dir}")

  correlation_canvases = [
      (ROOT.TCanvas("canvas_muon_correlations", "Muon Correlations", 900, 1600), 2, 4),
      (ROOT.TCanvas("canvas_dimuon_correlations", "Dimuon Correlations", 900, 1600), 2, 4),
  ]
  muon_resolution_canvases = [
      (ROOT.TCanvas(spec["canvas_name"], spec["canvas_title"], 900, 1600), 2, 4) for spec in MUON_RESOLUTION_CANVASES
  ]
  dimuon_resolution_canvases = [
      (ROOT.TCanvas(spec["canvas_name"], spec["canvas_title"], 900, 1600), 2, 4) for spec in DIMUON_RESOLUTION_CANVASES
  ]
  summary_canvases = [
      (ROOT.TCanvas(spec["canvas_name"], spec["canvas_title"], 1100, 1600), 2, 4)
      for spec in SUMMARY_CANVAS_SPECS
  ]
  qoverpt_canvas = (ROOT.TCanvas("canvas_muon_qoverpt_comparison", "Muon q/pT Resolution Comparison", 1400, 1600), 1, 1)
  efficiency_canvases = []
  for object_name, categories, output_stem in (
      ("ShiftMuon", MUON_EFFICIENCY_TYPES, "shiftmuon_efficiency"),
      ("ShiftDimuonVertex", DIMUON_EFFICIENCY_TYPES, "shiftdimuonvertex_efficiency"),
  ):
    efficiency_canvases.append((
        ROOT.TCanvas(f"canvas_{output_stem}", f"{object_name} efficiency", 1100, 1200),
        object_name,
        categories,
        f"{output_stem}.pdf",
    ))
  canvases = correlation_canvases + muon_resolution_canvases + dimuon_resolution_canvases + summary_canvases
  for canvas, columns, rows in canvases:
    canvas.Divide(columns, rows)

  # Keep references alive until all canvases have been serialized by PyROOT.
  drawn_objects = []
  drawn_objects += draw_2d(correlation_canvases[0][0], MUON_CORRELATIONS, input_file, correlation_rebin)
  drawn_objects += draw_2d(correlation_canvases[1][0], DIMUON_CORRELATIONS, input_file, correlation_rebin)
  # Summarize the native-bin histograms before the detailed resolution plots
  # apply their display rebinning in place.
  for summary_spec, (canvas, _, _) in zip(SUMMARY_CANVAS_SPECS, summary_canvases):
    drawn_objects += draw_scale_resolution_summary(canvas, summary_spec, input_file)
  for canvas_spec, (canvas, _, _) in zip(MUON_RESOLUTION_CANVASES, muon_resolution_canvases):
    drawn_objects += draw_resolutions(canvas, canvas_spec["names"], input_file, MUON_RESOLUTION_REBIN)
  for canvas_spec, (canvas, _, _) in zip(DIMUON_RESOLUTION_CANVASES, dimuon_resolution_canvases):
    drawn_objects += draw_resolutions(canvas, canvas_spec["names"], input_file, DIMUON_RESOLUTION_REBIN)
  drawn_objects += draw_qoverpt_comparison(qoverpt_canvas[0], input_file, CMS_DP_QOVERPT_DATA)
  for canvas, object_name, categories, _ in efficiency_canvases:
    canvas.Divide(2, 3)
    drawn_objects += draw_efficiencies(canvas, input_file, object_name, categories)

  output_names = [
      "muon_correlations.pdf",
      "dimuon_correlations.pdf",
      *[spec["output_name"] for spec in MUON_RESOLUTION_CANVASES],
      *[spec["output_name"] for spec in DIMUON_RESOLUTION_CANVASES],
      *[spec["output_name"] for spec in SUMMARY_CANVAS_SPECS],
  ]
  for (canvas, _, _), output_name in zip(canvases, output_names):
    canvas.SaveAs(f"{output_dir}/{output_name}")
  qoverpt_canvas[0].SaveAs(f"{output_dir}/muon_qoverpt_resolution_comparison.pdf")
  for canvas, _, _, output_name in efficiency_canvases:
    canvas.SaveAs(f"{output_dir}/{output_name}")
  input_file.Close()


if __name__ == "__main__":
  main()
