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
ROOT.gStyle.SetTitleSize(0.055, "XY")
ROOT.gStyle.SetLabelSize(0.045, "XY")
ROOT.gStyle.SetTitleOffset(1.05, "X")
ROOT.gStyle.SetTitleOffset(1.20, "Y")

# Rebin factors for all histograms on the corresponding resolution canvas.
MUON_RESOLUTION_REBIN = 1
DIMUON_RESOLUTION_REBIN = 2

# The canvases are split into relatively small pads, so ROOT's defaults clip
# long axis titles and the outermost tick labels.
PAD_LEFT_MARGIN = 0.17
PAD_RIGHT_MARGIN = 0.17
PAD_BOTTOM_MARGIN = 0.18
PAD_TOP_MARGIN = 0.10


MUON_CORRELATIONS = [
	"RecoVsGenMuon_eta", "RecoVsGenMuon_minv", "RecoVsGenMuon_phi", "RecoVsGenMuon_pt",
	"RecoVsGenMuon_pz", "RecoVsGenMuon_vx", "RecoVsGenMuon_vy", "RecoVsGenMuon_vz",
]
DIMUON_CORRELATIONS = [
	"RecoVsGenDimuon_eta", "RecoVsGenDimuon_minv", "RecoVsGenDimuon_phi", "RecoVsGenDimuon_pt",
	"RecoVsGenDimuon_pz", "RecoVsGenDimuon_vx", "RecoVsGenDimuon_vy", "RecoVsGenDimuon_vz",
]
MUON_RESOLUTIONS = [
  "MuonResolution_eta", "MuonResolution_phi", "MuonResolution_pt", "MuonResolution_pz",
  "MuonResolution_vx", "MuonResolution_vy", "MuonResolution_vz",
]
DIMUON_RESOLUTIONS = [
	"DimuonResolution_eta", "DimuonResolution_minv", "DimuonResolution_phi",
	"DimuonResolution_pt", "DimuonResolution_pz", "DimuonResolution_vx", "DimuonResolution_vy", "DimuonResolution_vz",
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
RESOLUTION_TITLES = {
	"MuonResolution_eta": "(#eta^{reco} - #eta^{gen}) / #eta^{gen}",
	"MuonResolution_phi": "(#phi^{reco} - #phi^{gen}) / #phi^{gen}",
	"MuonResolution_pt": "(p_{T}^{reco} - p_{T}^{gen}) / p_{T}^{gen}",
	"MuonResolution_pz": "(p_{z}^{reco} - p_{z}^{gen}) / p_{z}^{gen}",
	"MuonResolution_vx": "(v_{x}^{reco} - v_{x}^{gen}) / v_{x}^{gen}",
	"MuonResolution_vy": "(v_{y}^{reco} - v_{y}^{gen}) / v_{y}^{gen}",
	"MuonResolution_vz": "(v_{z}^{reco} - v_{z}^{gen}) / v_{z}^{gen}",
	"DimuonResolution_eta": "(#eta_{#mu#mu}^{reco} - #eta_{#mu#mu}^{gen}) / #eta_{#mu#mu}^{gen}",
	"DimuonResolution_phi": "(#phi_{#mu#mu}^{reco} - #phi_{#mu#mu}^{gen}) / #phi_{#mu#mu}^{gen}",
	"DimuonResolution_pt": "(p_{T,#mu#mu}^{reco} - p_{T,#mu#mu}^{gen}) / p_{T,#mu#mu}^{gen}",
	"DimuonResolution_pz": "(p_{z,#mu#mu}^{reco} - p_{z,#mu#mu}^{gen}) / p_{z,#mu#mu}^{gen}",
	"DimuonResolution_minv": "(m_{#mu#mu}^{reco} - m_{#mu#mu}^{gen}) / m_{#mu#mu}^{gen}",
	"DimuonResolution_vx": "(v_{x,#mu#mu}^{reco} - v_{x,#mu#mu}^{gen}) / v_{x,#mu#mu}^{gen}",
	"DimuonResolution_vy": "(v_{y,#mu#mu}^{reco} - v_{y,#mu#mu}^{gen}) / v_{y,#mu#mu}^{gen}",
	"DimuonResolution_vz": "(v_{z,#mu#mu}^{reco} - v_{z,#mu#mu}^{gen}) / v_{z,#mu#mu}^{gen}",
}

RESOLUTION_X_RANGES = {
	"MuonResolution_vx": (-5000.0, 5000.0),
	"MuonResolution_vy": (-5000.0, 5000.0),
	"DimuonResolution_vx": (-5000.0, 5000.0),
	"DimuonResolution_vy": (-5000.0, 5000.0),
}


HISTOGRAM_FILE_PATTERN = re.compile(
	r"v([1-9][0-9]*)_([0-9a-f]{7,40}(?:-dirty-[0-9a-f]{8})?)"
)


def histogram_version(path):
	match = HISTOGRAM_FILE_PATTERN.fullmatch(os.path.basename(os.path.dirname(path)))
	if not match:
		raise ValueError(
			f"histogram file '{path}' must be located in 'vN_<hash>/histograms.root'"
		)
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
		raise RuntimeError(
			f"No 'vN_<hash>/histograms.root' files found in '{histograms_dir}'"
		)

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
			raise argparse.ArgumentTypeError(
				f"'{spec}' needs {dimensions} positive integer factor(s)"
			)
		result[name] = factors
	return result


def set_axes_titles(hist, x_title, y_title):
	hist.SetTitle("")
	hist.GetXaxis().SetTitle(x_title)
	hist.GetYaxis().SetTitle(y_title)


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
		exponent = -0.5 * alpha_left * alpha_left + n_left * (
			math.log(n_left / alpha_left) - math.log(b - t)
		)
		return normalization * math.exp(exponent)
	b = n_right / alpha_right - alpha_right
	exponent = -0.5 * alpha_right * alpha_right + n_right * (
		math.log(n_right / alpha_right) - math.log(b + t)
	)
	return normalization * math.exp(exponent)


def fit_resolution(hist, name):
	if hist.GetEntries() < 10 or hist.GetMaximum() <= 0:
		return None, None
	x_axis = hist.GetXaxis()
	fit = ROOT.TF1(f"fit_{name}", double_sided_crystal_ball, x_axis.GetXmin(), x_axis.GetXmax(), 7)
	fit.SetParNames("N", "#mu", "#sigma", "#alpha_{L}", "n_{L}", "#alpha_{R}", "n_{R}")
	sigma_seed = max(hist.GetRMS(), x_axis.GetBinWidth(1))
	fit.SetParameters(hist.GetMaximum(), hist.GetMean(), sigma_seed, 1.5, 3.0, 1.5, 3.0)
	fit.SetParLimits(0, 0.0, max(10.0 * hist.GetMaximum(), 1.0))
	fit.SetParLimits(1, x_axis.GetXmin(), x_axis.GetXmax())
	fit.SetParLimits(2, 0.1 * x_axis.GetBinWidth(1), x_axis.GetXmax() - x_axis.GetXmin())
	for index in (3, 5):
		fit.SetParLimits(index, 0.05, 10.0)
	for index in (4, 6):
		fit.SetParLimits(index, 1.01, 100.0)
	fit_result = hist.Fit(fit, "QRS0")
	fit.SetLineColor(ROOT.kBlue + 1)
	fit.SetLineWidth(2)
	fit.Draw("same")
	return fit, int(fit_result)


def robust_resolution_summary(hist):
	probabilities = array("d", (0.16, 0.50, 0.84))
	quantiles = array("d", (0.0, 0.0, 0.0))
	hist.GetQuantiles(len(probabilities), quantiles, probabilities)
	return quantiles[1], 0.5 * (quantiles[2] - quantiles[0])


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
		fit, fit_status = fit_resolution(hist, name)
		median, central_68_half_width = robust_resolution_summary(hist)
		label = ROOT.TLatex()
		label.SetNDC(True)
		label.SetTextFont(42)
		label.SetTextSize(0.031)
		if fit:
			label.DrawLatex(0.17, 0.85, f"fit #mu = {fit.GetParameter(1):.4g}")
			label.DrawLatex(0.17, 0.80, f"fit #sigma = {abs(fit.GetParameter(2)):.4g}")
			if fit_status:
				label.DrawLatex(0.17, 0.65, f"fit status = {fit_status}")
		label.DrawLatex(0.17, 0.75, f"median = {median:.4g}")
		label.DrawLatex(0.17, 0.70, f"#sigma_{{68}} = {central_68_half_width:.4g}")
		objects.extend(filter(None, (fit, draw_zero_line(hist), label, annotate_entries(hist))))
	canvas.Update()
	return objects


def parse_arguments():
	parser = argparse.ArgumentParser(description="Plot SHIFT reconstruction diagnostics")
	project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	parser.add_argument(
		"--input",
		help="versioned input ROOT file (default: highest-version file in --histograms-dir)",
	)
	parser.add_argument(
		"--histograms-dir", default=f"{project_dir}/plots/",
		help="directory searched for histograms_vN_<hash>.root files",
	)
	parser.add_argument(
		"--output-dir", default=f"{project_dir}/plots",
		help="parent directory for versioned plot directories",
	)
	parser.add_argument(
		"--rebin-2d", action="append", default=[], metavar="HIST=XFACTOR[,YFACTOR]",
		help="rebin one 2D histogram; one factor applies to both axes",
	)
	return parser.parse_args()


def main():
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

	canvases = [
		(ROOT.TCanvas("canvas_muon_correlations", "Muon Correlations", 900, 1600), 2, 4),
		(ROOT.TCanvas("canvas_dimuon_correlations", "Dimuon Correlations", 900, 1600), 2, 4),
		(ROOT.TCanvas("canvas_muon_resolutions", "Muon Resolutions", 900, 1600), 2, 4),
		(ROOT.TCanvas("canvas_dimuon_resolutions", "Dimuon Resolutions", 900, 1600), 2, 4),
	]
	for canvas, columns, rows in canvases:
		canvas.Divide(columns, rows)

	# Keep references alive until all canvases have been serialized by PyROOT.
	drawn_objects = []
	drawn_objects += draw_2d(canvases[0][0], MUON_CORRELATIONS, input_file, correlation_rebin)
	drawn_objects += draw_2d(canvases[1][0], DIMUON_CORRELATIONS, input_file, correlation_rebin)
	drawn_objects += draw_resolutions(canvases[2][0], MUON_RESOLUTIONS, input_file, MUON_RESOLUTION_REBIN)
	drawn_objects += draw_resolutions(canvases[3][0], DIMUON_RESOLUTIONS, input_file, DIMUON_RESOLUTION_REBIN)

	output_names = [
		"muon_correlations.pdf",
		"dimuon_correlations.pdf",
		"muon_resolutions.pdf",
		"dimuon_resolutions.pdf",
	]
	for (canvas, _, _), output_name in zip(canvases, output_names):
		canvas.SaveAs(f"{output_dir}/{output_name}")
	input_file.Close()


if __name__ == "__main__":
	main()
