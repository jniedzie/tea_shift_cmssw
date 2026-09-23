import ROOT
from Sample import Sample, SampleType
from Legend import Legend
from Histogram import Histogram
from HistogramNormalizer import NormalizationType
from CmsLabelsManager import CmsLabel
from shift_sample_paths import single_root_file

from shift_paths import base_path, sample, campaign, cross_sections, jpsi_campaign_base, qcd_campaign_base

samples = []

jpsi_colors = {
    "1to2": ROOT.TColor.GetColor("#D62728"),  # red
    "2to5": ROOT.TColor.GetColor("#FF7F0E"),  # orange
    "5to10": ROOT.TColor.GetColor("#BCBD22"),  # olive
    "10to20": ROOT.TColor.GetColor("#9467BD"),  # purple
    "20to-1": ROOT.TColor.GetColor("#E377C2"),  # pink
}

qcd_colors = {
    "1to2": ROOT.TColor.GetColor("#1F77B4"),  # blue
    "2to5": ROOT.TColor.GetColor("#17BECF"),  # cyan
    "5to10": ROOT.TColor.GetColor("#2CA02C"),  # green
    "10to20": ROOT.TColor.GetColor("#8C564B"),  # brown
    "20to-1": ROOT.TColor.GetColor("#4D4D4D"),  # dark gray
}

for pt_bin in cross_sections["jpsi"]:
  samples.append(
      Sample(
          name=f"jpsi_{pt_bin}",
          file_path=single_root_file(f"{base_path}/jpsi/{jpsi_campaign_base.format(pt_bin)}/histograms"),
          type=SampleType.background,
          cross_section=cross_sections["jpsi"][pt_bin],
          line_alpha=0.0,
          fill_color=jpsi_colors[pt_bin],
          fill_alpha=0.7,
          marker_size=0.0,
          legend_description=f"J/#Psi pThat {pt_bin} GeV",
      )
  )

for pt_bin in cross_sections["qcd"]:
  samples.append(
      Sample(
          name=f"qcd_{pt_bin}",
          file_path=single_root_file(f"{base_path}/qcd/{qcd_campaign_base.format(pt_bin)}/histograms"),
          type=SampleType.background,
          cross_section=cross_sections["qcd"][pt_bin],
          line_alpha=0.0,
          fill_color=qcd_colors[pt_bin],
          fill_alpha=0.7,
          marker_size=0.0,
          legend_description=f"QCD pThat {pt_bin} GeV",
      )
  )

output_path = "../plots/plots/"

muon_categories = ["", "BothEndcaps"]
dimuon_categories = ["", "Both-Both"]

histograms = []
# fmt: off

for category in muon_categories:
  histograms.extend([
    Histogram(f"event/Event_nShiftMuon{category}", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "n_{#mu}"            , "# events"),

    Histogram(f"muon/ShiftMuon{category}_pt"   , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "p_{T}^{#mu} [GeV]", "# events"),
    Histogram(f"muon/ShiftMuon{category}_pz"   , "", False, True , NormalizationType.to_lumi,     1, None, 200 , None, None, "p_{Z}^{#mu} [GeV]", "# events"),
    Histogram(f"muon/ShiftMuon{category}_eta"  , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "#eta^{#mu}"       , "# events"),
    Histogram(f"muon/ShiftMuon{category}_phi"  , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "#phi^{#mu}"       , "# events"),

    Histogram(f"muon/ShiftMuon{category}_vx" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "v_{x}^{#mu} [cm]"    , "# events"),
    Histogram(f"muon/ShiftMuon{category}_vy" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "v_{y}^{#mu} [cm]"    , "# events"),
    Histogram(f"muon/ShiftMuon{category}_vz" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "v_{z}^{#mu} [cm]"    , "# events"),
    Histogram(f"muon/ShiftMuon{category}_dz" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "d_{z}^{#mu} [cm]"    , "# events"),

    Histogram(f"muon/ShiftMuon{category}_nCSCHits" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "n_{CSCHits}^{#mu}"    , "# events"),
    Histogram(f"muon/ShiftMuon{category}_nDTHits"  , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "n_{DTHits}^{#mu}"    , "# events"),
    Histogram(f"muon/ShiftMuon{category}_nRPCHits" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "n_{RPCHits}^{#mu}"    , "# events"),
  ])

for category in dimuon_categories:
  # name        title         logx   logy   norm_type                  rebin  xmin  xmax  ymin  ymax  xlabel         ylabel
  histograms.extend([
      Histogram(f"event/Event_nShiftDimuonVertex{category}", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "n_{#mu#mu}"        , "# events"),

      Histogram(f"dimuon/ShiftDimuonVertex{category}_pt"   , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "p_{T}^{#mu#mu} [GeV]", "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_pz"   , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "p_{Z}^{#mu#mu} [GeV]", "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_eta"  , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "#eta^{#mu#mu}"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_phi"  , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "#phi^{#mu#mu}"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_mass" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "m_{#mu#mu} [GeV]"    , "# events"),

      Histogram(f"dimuon/ShiftDimuonVertex{category}_vx" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "v_{x}^{#mu#mu} [GeV]"    , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_vy" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "v_{y}^{#mu#mu} [GeV]"    , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_vz" , "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "v_{z}^{#mu#mu} [GeV]"    , "# events"),

      Histogram(f"dimuon/ShiftDimuonVertex{category}_chi2", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "#chi^{2}"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_normalizedChi2", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "#chi^{2}/ndof"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_dca", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "DCA [cm]"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_dcaValid", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "DCA valid"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_isOS", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "isOS"       , "# events"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_genPid", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "generator mother", "# dimuons"),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_genPidDifferentMothers", "", False, True , NormalizationType.to_lumi,     1, None, None, None, None, "generator mother", "# dimuon mothers"),

  ])
# fmt: on
luminosity = 300000.0  # pb^-1 (Run 3)

legends = {
    SampleType.background: Legend(0.7, 0.6, 0.8, 0.9, "f"),
}

plotting_options = {
    SampleType.background: "hist",
    SampleType.signal: "nostack hist",
    SampleType.data: "nostack e",
}

canvas_size = (800, 600)
show_ratio_plots = False
# ratio_limits = (0.7, 1.3)  # Optional override; limits are automatic by default.
