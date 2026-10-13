# Code style: human-code.
import ROOT
from Sample import Sample, SampleType
from Legend import Legend
from Histogram import Histogram, Histogram2D
from HistogramNormalizer import NormalizationType
from CmsLabelsManager import CmsLabel
from Logger import warn
from shift_sample_paths import (
    single_root_file,
    full_gen_weight_sums,
    validated_bsm_campaign_histograms,
)
from shift_storage_paths import tier_directory
from shift_plot_binning import muon_variable_binning, dimuon_variable_binning
from shift_plot_labels import apply_reconstruction_labels

from shift_paths import (
    base_path,
    cross_sections,
    campaign,
    bsm_campaign,
    bsm_masses_to_plot,
    bsm_mean_lab_flights_to_plot,
    histogram_cache_directory,
)


samples = []
normalization_sums = full_gen_weight_sums(campaign)

jpsi_colors = {
    "0to1": ROOT.TColor.GetColor("#FEC44F"),  # gold
    "1to2": ROOT.TColor.GetColor("#FE9929"),  # amber
    "2to5": ROOT.TColor.GetColor("#EC7014"),  # orange
    "5to10": ROOT.TColor.GetColor("#EF3B2C"),  # coral red
    "10to20": ROOT.TColor.GetColor("#CB181D"),  # red
    "20toinf": ROOT.TColor.GetColor("#99000D"),  # dark red
}

qcd_colors = {
    "0to1": ROOT.TColor.GetColor("#74C476"),  # light green
    "1to2": ROOT.TColor.GetColor("#41AE76"),  # green
    "2to5": ROOT.TColor.GetColor("#1FA3A3"),  # teal
    "5to10": ROOT.TColor.GetColor("#3690C0"),  # cyan blue
    "10to20": ROOT.TColor.GetColor("#2171B5"),  # blue
    "20toinf": ROOT.TColor.GetColor("#084594"),  # dark blue
}

dy_colors = {
    "0.211317to0.5": ROOT.TColor.GetColor("#D9D9D9"),  # light gray
    "0.5to1": ROOT.TColor.GetColor("#BDBDBD"),  # silver gray
    "1to2": ROOT.TColor.GetColor("#969696"),  # medium gray
    "2to5": ROOT.TColor.GetColor("#737373"),  # slate gray
    "5to10": ROOT.TColor.GetColor("#525252"),  # dark gray
    "10to20": ROOT.TColor.GetColor("#303030"),  # charcoal
    "20to-1": ROOT.TColor.GetColor("#000000"),  # black
}


def get_bin_label(bin_name, process):
    low, high = bin_name.split("to")
    variable = "m_{#mu#mu}^{gen}" if process == "dy" else "#hat{p}_{T}"
    interval = f"[{low}, #infty)" if high in ("-1", "inf") else f"[{low}, {high}]"
    return f"{variable} #in {interval} GeV"


for process, colors, label in (
    ("dy", dy_colors, "DY"),
    ("jpsi", jpsi_colors, "J/#Psi"),
    ("qcd", qcd_colors, "QCD"),
):
    if process not in cross_sections:
        continue

    for bin_name, cross_section in sorted(
        cross_sections[process].items(),
        key=lambda item: float(item[0].split("to")[0]),
        reverse=True,
    ):
        directory = tier_directory(
            base_path, process, bin_name, campaign, "histograms_merged"
        )
        file_path = single_root_file(
            directory, allow_missing=True, cache_directory=histogram_cache_directory
        )
        if file_path is None:
            warn(
                f"Skipping sample {process}_{bin_name}: ROOT input directory is missing, "
                f"inaccessible, or contains no ROOT files: '{directory}'"
            )
            continue
        samples.append(
            Sample(
                name=f"{process}_{bin_name}",
                file_path=file_path,
                type=SampleType.background,
                cross_section=cross_section,
                initial_weight_sum=(
                    normalization_sums[f"{process}_{bin_name}"]
                    if normalization_sums is not None
                    else -1
                ),
                line_alpha=0.0,
                fill_color=colors[bin_name],
                fill_alpha=0.7,
                marker_size=0.0,
                legend_description=f"{label} {get_bin_label(bin_name, process)}",
            )
        )


bsm_inputs = validated_bsm_campaign_histograms(
    base_path,
    bsm_campaign,
    bsm_masses_to_plot,
    bsm_mean_lab_flights_to_plot,
    cache_directory=histogram_cache_directory,
)
mass_colors = (ROOT.kMagenta + 1, ROOT.kAzure + 2, ROOT.kOrange + 7)
masses = sorted({point["mass_gev"] for point in bsm_inputs})
for point in bsm_inputs:
    mass = point["mass_gev"]

    flight = point["mean_lab_flight_m"]
    lifetime_label = "prompt" if flight is None else f"#LT L #GT #approx {flight:g} m"

    samples.append(
        Sample(
            name=point["name"],
            file_path=point["histogram_path"],
            type=SampleType.signal,
            initial_weight_sum=point["initial_weight_sum"],
            line_color=mass_colors[masses.index(point["mass_gev"]) % len(mass_colors)],
            line_style=1 if flight is None else 2 if flight <= 5 else 3,
            line_width=3,
            fill_alpha=0.0,
            marker_size=0.0,
            legend_description=f"A' {point['mass_gev']:g} GeV, {lifetime_label}",
        )
    )

# Keep both the stack (bottom to top) and legend in process/bin order.
# Otherwise the shared plotter sorts all processes together by cross section.
custom_stacks_order = [sample.name for sample in samples]

output_path = f"../plots/SM_BSM/{bsm_campaign}/" if bsm_inputs else "../plots/plots/"

# Expand automatic axes only when needed to keep distributions clear of legends.
auto_adjust_axes_for_legend = True

muon_categories = ["", "BothEndcaps"]
dimuon_categories = ["", "Both-Both"]

histograms = []
# fmt: off

for category in muon_categories:
  histograms.extend([
    Histogram(f"event/Event_nShiftMuon{category}"    , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "n_{#mu}"          , "# events"      ),

    Histogram(f"muon/ShiftMuon{category}_pt"         , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{T}^{#mu} [GeV]", "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_pz"         , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{Z}^{#mu} [GeV]", "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_pt_variable", "", False, True, NormalizationType.to_lumi, 10, None, None, None, None, "p_{T}^{#mu} [GeV]", "# events / GeV", scale_by_bin_width=True),
    Histogram(f"muon/ShiftMuon{category}_pz_variable", "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{Z}^{#mu} [GeV]", "# events / GeV", scale_by_bin_width=True),
    Histogram(f"muon/ShiftMuon{category}_p_variable" , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p^{#mu} [GeV]"    , "# events / GeV", scale_by_bin_width=True),
    Histogram(f"muon/ShiftMuon{category}_eta"        , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "#eta^{#mu}"       , "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_phi"        , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "#phi^{#mu}"       , "# events"      ),

    Histogram(f"muon/ShiftMuon{category}_vx"         , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "v_{x}^{#mu} [cm]" , "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_vy"         , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "v_{y}^{#mu} [cm]" , "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_vz"         , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "v_{z}^{#mu} [cm]" , "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_dz"         , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "d_{z}^{#mu} [cm]" , "# events"      ),

    Histogram(f"muon/ShiftMuon{category}_nCSCHits"   , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "n_{CSCHits}^{#mu}", "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_nDTHits"    , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "n_{DTHits}^{#mu}" , "# events"      ),
    Histogram(f"muon/ShiftMuon{category}_nRPCHits"   , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "n_{RPCHits}^{#mu}", "# events"      ),
  ])

for category in dimuon_categories:
  # name        title         logx   logy   norm_type                  rebin  xmin  xmax  ymin  ymax  xlabel         ylabel
  histograms.extend([
      Histogram(f"event/Event_nShiftDimuonVertex{category}"                 , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "n_{#mu#mu}"          , "# events"        ),

      Histogram(f"dimuon/ShiftDimuonVertex{category}_pt"                    , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{T}^{#mu#mu} [GeV]", "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_pz"                    , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{Z}^{#mu#mu} [GeV]", "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_pt_variable"           , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{T}^{#mu#mu} [GeV]", "# events / GeV"  , scale_by_bin_width=True      ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_pz_variable"           , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p_{Z}^{#mu#mu} [GeV]", "# events / GeV"  , scale_by_bin_width=True      ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_p_variable"            , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "p^{#mu#mu} [GeV]"    , "# events / GeV"  , scale_by_bin_width=True      ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_eta"                   , "", False, True, NormalizationType.to_lumi,  2, None, None, None, None, "#eta^{#mu#mu}"       , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_phi"                   , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "#phi^{#mu#mu}"       , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_mass"                  , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "m_{#mu#mu} [GeV]"    , "# events"        , allow_legend_x_extension=True),

      Histogram(f"dimuon/ShiftDimuonVertex{category}_vx"                    , "", False, True, NormalizationType.to_lumi,  5, None, None, None, None, "v_{x}^{#mu#mu} [cm]" , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_vy"                    , "", False, True, NormalizationType.to_lumi,  5, None, None, None, None, "v_{y}^{#mu#mu} [cm]" , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_vz"                    , "", False, True, NormalizationType.to_lumi, 20, None, None, None, None, "v_{z}^{#mu#mu} [cm]" , "# events"        ),

      Histogram(f"dimuon/ShiftDimuonVertex{category}_chi2"                  , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "#chi^{2}"            , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_normalizedChi2"        , "", False, True, NormalizationType.to_lumi,  2, None, None, None, None, "#chi^{2}/ndof"       , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_dca"                   , "", False, True, NormalizationType.to_lumi,  4, None, None, None, None, "DCA [cm]"            , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_isOS"                  , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "isOS"                , "# events"        ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_genPid"                , "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "generator mother"    , "# dimuons"       ),
      Histogram(f"dimuon/ShiftDimuonVertex{category}_genPidDifferentMothers", "", False, True, NormalizationType.to_lumi,  1, None, None, None, None, "generator mother"    , "# dimuon mothers"),

  ])
# fmt: on

# Compare the two momentum hypotheses and locate category-dependent failures.
for category in muon_categories:
    for variable, label in (
        ("constrainedPt", "p_{T}^{#mu, constrained} [GeV]"),
        ("constrainedPz", "p_{z}^{#mu, constrained} [GeV]"),
        ("constrainedEta", "#eta^{#mu, constrained}"),
        ("constrainedPhi", "#phi^{#mu, constrained}"),
        ("constrainedVz", "v_{z}^{#mu, constrained} [cm]"),
        ("linePcaZ", "z_{PCA}^{#mu} [cm]"),
        ("trackVz", "z_{track}^{#mu} [cm]"),
        ("nGEMHits", "n_{GEM hits}^{#mu}"),
        ("nME0Hits", "n_{ME0 hits}^{#mu}"),
    ):
        histograms.append(
            Histogram(
                name=f"muon/ShiftMuon{category}_{variable}",
                log_y=True,
                x_label=label,
                y_label="# muons",
            )
        )

# All quantities use reconstruction only. Mass alternatives are filled only
# for successful fits; SS mass provides a separate combinatorial control.
for category in dimuon_categories:
    for variable, label, rebin, upper in (
        ("openingAngle", "#alpha_{#mu#mu} [rad]", 1, None),
        ("openingAngleFine", "#alpha_{#mu#mu} [rad]", 1, None),
        ("constrainedOpeningAngle", "#alpha_{#mu#mu}^{constrained} [rad]", 1, None),
        ("deltaPhi", "|#Delta#phi_{#mu#mu}| [rad]", 1, None),
        ("deltaEta", "|#Delta#eta_{#mu#mu}|", 1, None),
        ("deltaR", "#Delta R_{#mu#mu}", 1, None),
        ("ptAsymmetry", "|p_{T,1}-p_{T,2}|/(p_{T,1}+p_{T,2})", 1, None),
        ("momentumAsymmetry", "|p_{1}-p_{2}|/(p_{1}+p_{2})", 1, None),
        ("minMuonPt", "min(p_{T,1},p_{T,2}) [GeV]", 2, None),
        ("maxMuonPt", "max(p_{T,1},p_{T,2}) [GeV]", 2, None),
        ("rapidity", "y_{#mu#mu}", 2, None),
        ("vertexR", "r_{vertex}^{#mu#mu} [cm]", 2, None),
        ("massOS", "m_{#mu^{+}#mu^{-}} [GeV]", 5, None),
        ("massSS", "m_{#mu^{#pm}#mu^{#pm}} [GeV]", 5, None),
        ("constrainedMass", "m_{#mu#mu}^{constrained} [GeV]", 1, None),
        ("refittedMass", "m_{#mu#mu}^{refit} [GeV]", 1, None),
        ("probability", "vertex fit probability", 1, None),
        ("originCompatibilityNormalizedChi2", "#chi^{2}_{origin}/3", 2, None),
        ("constrainedValid", "valid constrained vertex", 1, None),
        ("refitStatus", "joint vertex refit status", 1, None),
        ("dcaValid", "valid two-track DCA", 1, None),
    ):
        histograms.append(
            Histogram(
                name=f"dimuon/ShiftDimuonVertex{category}_{variable}",
                log_y=True,
                rebin=rebin,
                x_max=upper,
                x_label=label,
                y_label="# dimuons",
                allow_legend_x_extension=upper is not None,
            )
        )

# These distributions are already filled, but were absent from this plotter.
for name, label in (
    ("muon/ShiftMuon_topology", "muon topology"),
    ("muon/ShiftMuon_recoAlgorithm", "muon reconstruction algorithm"),
    ("dimuon/ShiftDimuonVertex_topologyCategory", "dimuon topology"),
    (
        "detector_diagnostics/DetectorDiagnostics_dtAttachmentFraction",
        "DT attachment fraction",
    ),
    (
        "detector_diagnostics/DetectorDiagnostics_trackerAttachmentFraction",
        "tracker attachment fraction",
    ),
    (
        "detector_diagnostics/DetectorDiagnostics_timingMeasurements",
        "timing measurements per muon",
    ),
    (
        "detector_diagnostics/DetectorDiagnostics_timingDeltaChi2",
        "#Delta#chi^{2}_{muon timing}",
    ),
    (
        "detector_diagnostics/DetectorDiagnostics_combinedTimingDeltaChi2",
        "#Delta#chi^{2}_{combined timing}",
    ),
    ("target_diagnostics/TargetDiagnostics_constrainedStatus", "target fit status"),
    (
        "target_diagnostics/TargetDiagnostics_targetForwardStatus",
        "forward propagation status",
    ),
    (
        "vertex_refit_diagnostics/VertexRefitDiagnostics_massErr",
        "#sigma(m_{#mu#mu}^{refit}) [GeV]",
    ),
    (
        "vertex_refit_diagnostics/VertexRefitDiagnostics_massRelativeErr",
        "#sigma(m_{#mu#mu}^{refit})/m_{#mu#mu}^{refit}",
    ),
    (
        "vertex_refit_diagnostics/VertexRefitDiagnostics_minQoverPSignificance",
        "min |q/p|/#sigma(q/p)",
    ),
):
    histograms.append(
        Histogram(name=name, log_y=True, x_label=label, y_label="# entries")
    )

# Rebin only recorded source bins and display densities, so changing a bin's
# width does not itself create a rising tail or a false bump. The same edges
# apply to the inclusive and Both-Endcaps/Both-Both selections and all samples.
variable_histogram_rebin = 20  # Merge this many adjacent display bins; 1 keeps them.

for histogram in histograms:
    if histogram.name.startswith("muon/ShiftMuon"):
        rules, object_label = muon_variable_binning, "# muons"
    elif histogram.name.startswith("dimuon/ShiftDimuonVertex"):
        rules, object_label = dimuon_variable_binning, "# dimuons"
    else:
        continue
    variable = histogram.name.rsplit("_", 1)[-1]
    if variable not in rules:
        continue
    histogram.bin_edges = rules[variable]
    histogram.rebin = variable_histogram_rebin
    histogram.scale_by_bin_width = True
    unit = next(
        (unit for unit in ("GeV", "cm", "rad") if f"[{unit}]" in histogram.x_label),
        "unit x",
    )
    histogram.y_label = f"{object_label} / {unit}"
    if variable == "openingAngleFine":
        # Resolve the boosted peak here; the full angular plots retain all
        # recorded tails, including the antiparallel endpoint.
        histogram.x_max = 0.2
        histogram.allow_legend_x_extension = True
    # if variable in ("openingAngle", "constrainedOpeningAngle",
    #                 "originCompatibilityNormalizedChi2"):
    #     histogram.show_full_x_range = True

apply_reconstruction_labels(histograms)

histograms2D = []
for category in dimuon_categories:
    for variable, label in (
        ("massVsOpeningAngle", "#alpha_{#mu#mu} [rad]"),
        ("massVsPtAsymmetry", "|p_{T,1}-p_{T,2}|/(p_{T,1}+p_{T,2})"),
    ):
        histograms2D.append(
            Histogram2D(
                name=f"dimuon/ShiftDimuonVertex{category}_{variable}",
                log_z=True,
                norm_type=NormalizationType.none,
                x_label=label,
                y_label="m_{#mu#mu} [GeV]",
                z_label="weighted dimuons",
            )
        )
canvas_size_2Dhists = (800, 700)

luminosity = 0.01 * 300000.0  # pb^-1 (1% of Run 3)

show_cms_labels = True
cms_label = CmsLabel.pas_sim
lumi_label_value = luminosity
label_outside_axes = True
beam_label = "pp @ 13.6 TeV"

legends = {
    SampleType.background: Legend(0.58, 0.2, 0.9, 0.9, "f", text_size=16),
}

plotting_options = {
    SampleType.background: "hist",
    SampleType.signal: "nostack hist",
    SampleType.data: "nostack e",
}

output_formats = ["pdf"]
canvas_size = (800, 600)
categorical_canvas_size = (1200, 850)
categorical_legend_outside = True
show_ratio_plots = False
# ratio_limits = (0.7, 1.3)  # Optional override; limits are automatic by default.

if bsm_inputs:
    # Match each BSM histogram integral to its SM histogram integral.
    for histogram in histograms:
        histogram.norm_type = NormalizationType.to_background
    canvas_size = (1400, 850)
    plot_margins = {"left": 0.10, "right": 0.37, "top": 0.12, "bottom": 0.14}
    legends = {
        SampleType.background: Legend(
            0.65, 0.48, 0.99, 0.90, "f", title="SM", text_size=16
        ),
        SampleType.signal: Legend(
            0.65,
            0.12,
            0.99,
            0.45,
            "l",
            title="BSM (normalized to SM integral)",
            text_size=17,
        ),
    }
