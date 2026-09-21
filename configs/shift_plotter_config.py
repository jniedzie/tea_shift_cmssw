import ROOT
from Sample import Sample, SampleType
from Legend import Legend
from Histogram import Histogram
from HistogramNormalizer import NormalizationType
from CmsLabelsManager import CmsLabel

from shift_paths import base_path, sample, campaign

samples = (
    Sample(
        name="jpsi",
        file_path="../plots/v51_ca5af1024788/histograms.root",
        type=SampleType.background,
        cross_section=5.856e+05,  # pb
        line_alpha=0.0,
        fill_color=ROOT.kRed - 2,
        fill_alpha=0.7,
        marker_size=0.0,
        legend_description="J/#Psi",
    ),
    Sample(
        name="qcd",
        file_path="../plots/v50_fcbd49caa94d/histograms.root",
        type=SampleType.background,
        cross_section=4.023e+10,  # pb
        line_alpha=0.0,
        fill_color=ROOT.kGreen+2,
        fill_alpha=0.7,
        marker_size=0.0,
        legend_description="QCD",
    ),
)

output_path = "../plots/plots/"

dimuon_categories = ["", "Both-Both"]
histograms = []
# fmt: off
for category in dimuon_categories:
  # name        title         logx   logy   norm_type                  rebin  xmin  xmax  ymin  ymax  xlabel         ylabel
  histograms.extend([
      Histogram(f"dimuon/ShiftDimuonVertex{category}_pt"   , "", False, True , NormalizationType.to_lumi,     5, None, None, None, None, "p_{T}^{#mu#mu} [GeV]", "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_mass" , "", False, True , NormalizationType.to_lumi,     5, None, None, None, None, "m_{#mu#mu} [GeV]"    , "# events"),
  ])
# fmt: on
luminosity = 300000.0  # pb^-1 (Run 3)

legends = {
    SampleType.background: Legend(0.7, 0.8, 0.85, 0.85, "f"),
}

plotting_options = {
    SampleType.background: "hist",
    SampleType.signal: "nostack hist",
    SampleType.data: "nostack e",
}

canvas_size = (800, 600)
show_ratio_plots = False
# ratio_limits = (0.7, 1.3)  # Optional override; limits are automatic by default.
