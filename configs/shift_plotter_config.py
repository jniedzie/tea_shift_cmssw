# Code style: human-code.

# Quick dimuon reruns only need the first group. Enable the others as needed.
plot_dimuons = True
plot_muons = False
plot_reco = False
plot_diagnostics = False
plot_2d = False

muon_categories = ["", "BothEndcaps"]
dimuon_categories = ["", "Both-Both"]
variable_histogram_rebin = 1  # 1 disables extra coarsening of variable-bin plots.

# Optional J/ψ shape comparison using the published merged ROOT files.
jpsi_data_histogram = globals().get("jpsi_data_histogram")
jpsi_mc_histograms = globals().get("jpsi_mc_histograms", {})

from Histogram import Histogram, Histogram2D
from HistogramNormalizer import NormalizationType
from shift_plot_samples import prepare_plot_samples
from shift_plot_style import (
    apply_plot_style,
    apply_2d_plot_style,
    plot_legends,
    luminosity,
    show_cms_labels,
    cms_label,
    label_outside_axes,
    beam_label,
    plotting_options,
    categorical_canvas_size,
    categorical_legend_outside,
)
from shift_paths import (
    base_path,
    cross_sections,
    campaign,
    bsm_campaign,
    bsm_masses_to_plot,
    bsm_mean_lab_flights_to_plot,
    bsm_epsilons_to_plot,
    histogram_cache_directory,
)

if jpsi_data_histogram:
    bsm_inputs = []
    output_path = globals().get("jpsi_output_path", "../plots/data/jpsi_control/")
    normalization = NormalizationType.to_data
else:
    from shift_plot_style import lumi_label_value

    samples, bsm_inputs = prepare_plot_samples(
        base_path,
        cross_sections,
        campaign,
        bsm_campaign,
        bsm_masses_to_plot,
        bsm_mean_lab_flights_to_plot,
        bsm_epsilons_to_plot,
        histogram_cache_directory,
    )
    output_path = (
        f"../plots/physics/SM_BSM/{bsm_campaign}/"
        if bsm_inputs
        else f"../plots/physics/SM/{campaign}/"
    )
    normalization = (
        NormalizationType.to_background if bsm_inputs else NormalizationType.to_lumi
    )
histograms = []

# fmt: off

if plot_muons:
    for category in muon_categories:
        histograms.extend([
            # name                                                                                   title log_x  log_y           norm           rebin x_min x_max y_min y_max
            Histogram(f"event/Event_nShiftMuon{category}"    , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_pt"         , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_pz"         , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_pt_variable", "", False, True, normalization, 1, None, None, None, None, scale_by_bin_width=True),
            Histogram(f"muon/ShiftMuon{category}_pz_variable", "", False, True, normalization, 1, None, None, None, None, scale_by_bin_width=True),
            Histogram(f"muon/ShiftMuon{category}_p_variable" , "", False, True, normalization, 1, None, None, None, None, scale_by_bin_width=True),
            Histogram(f"muon/ShiftMuon{category}_eta"        , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_phi"        , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_vx"         , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_vy"         , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_vz"         , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_dz"         , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_nCSCHits"   , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_nDTHits"    , "", False, True, normalization, 1, None, None, None, None),
            Histogram(f"muon/ShiftMuon{category}_nRPCHits"   , "", False, True, normalization, 1, None, None, None, None),
        ])

if plot_dimuons:
    for category in dimuon_categories:
        histograms.extend([
            # name                                                                                   title log_x  log_y           norm           rebin x_min x_max y_min y_max
            Histogram(f"event/Event_nShiftDimuonVertex{category}"            , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pt"               , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pz"               , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pt_variable"      , "", False, True          , normalization,  5, None  , None   , 1e-3, None, scale_by_bin_width=True      ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pz_variable"      , "", False, True          , normalization,  5, None  , None   , 1e-3, None, scale_by_bin_width=True      ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_p_variable"       , "", False, True          , normalization,  5, None  , None   , 1e-3, None, scale_by_bin_width=True      ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_eta"              , "", False, True          , normalization,  5, None  , None   , 1e-3, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_phi"              , "", False, True          , normalization,  2, None  , None   , 1e-1, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_mass"             , "", False, True          , normalization, 20, None  , 350    , 1e-3, None, allow_legend_x_extension=True),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_logMass"          , "", False, True          , normalization,  5, None  , None   , 1e-1, None, de_loggify_x_axis=True       ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_vx"               , "", False, True          , normalization,  5, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_vy"               , "", False, True          , normalization,  5, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_vz"               , "", False, not bsm_inputs, normalization, 10, None  , None   , 1e-1, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_chi2"             , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_normalizedChi2"   , "", False, True          , normalization,  2, None  , None   , None, None),
            
            Histogram(f"dimuon/ShiftDimuonVertex{category}_openingAngle"     , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_openingAngleFine" , "", False, True          , normalization,  1, None  , 0.2    , None, None, allow_legend_x_extension=True),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_deltaPhi"         , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_deltaEta"         , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_deltaR"           , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_ptAsymmetry"      , "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_momentumAsymmetry", "", False, True          , normalization,  1, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_minMuonPt"        , "", False, True          , normalization,  2, None  , None   , None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_maxMuonPt"        , "", False, True          , normalization,  2, None  , None   , None, None),
            
            
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_dca"                              , "", False, True          , normalization,  4, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_isOS"                             , "", False, True          , normalization,  1, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_rapidity"                         , "", False, True          , normalization,  2, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_vertexR"                          , "", False, True          , normalization,  2, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_massOS"                           , "", False, True          , normalization,  5, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_massSS"                           , "", False, True          , normalization,  5, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_probability"                      , "", False, True          , normalization,  1, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_originCompatibilityNormalizedChi2", "", False, True          , normalization,  2, None, None, None, None),
        ])

        # The current signal files use different bins and counts per bin need a shared axis.
        if not bsm_inputs:
            histograms.extend([
                Histogram(f"dimuon/ShiftDimuonVertex{category}_ptLog", log_x=True, log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvDca", log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvDcaLog", log_x=True, log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCos", log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCosLog", log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCosFine", log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCosFineNegative", log_y=True, norm_type=normalization, show_full_x_range=True, legend_outside=True),
            ])

# Alternative fit states and reconstruction checks, separate from pair properties.
if plot_reco:
    for category in muon_categories:
        histograms.extend([
            # name                                        title  log_x  log_y  norm           rebin  x_min  x_max  y_min  y_max
            Histogram(f"muon/ShiftMuon{category}_constrainedPt" , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_constrainedPz" , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_constrainedEta", ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_constrainedPhi", ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_constrainedVz" , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_linePcaZ"      , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_trackVz"       , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_nGEMHits"      , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"muon/ShiftMuon{category}_nME0Hits"      , ""   , False, True , normalization,     1, None , None , None , None),
        ])
    for category in dimuon_categories:
        histograms.extend([
            # name                                                           title  log_x  log_y  norm           rebin  x_min  x_max  y_min  y_max
            Histogram(f"dimuon/ShiftDimuonVertex{category}_constrainedOpeningAngle", ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_constrainedMass"        , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_refittedMass"           , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_constrainedValid"       , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_refitStatus"            , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_dcaValid"               , ""   , False, True , normalization,     1, None , None , None , None),
        ])

if plot_diagnostics:
    for category in dimuon_categories:
        histograms.extend([
            # name                                                          title  log_x  log_y  norm           rebin  x_min  x_max  y_min  y_max
            Histogram(f"dimuon/ShiftDimuonVertex{category}_genPid"                , ""   , False, True , normalization,     1, None , None , None , None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_genPidDifferentMothers", ""   , False, True , normalization,     1, None , None , None , None),
        ])
    histograms.extend([
        # name                                                                     title  log_x  log_y  norm           rebin  x_min  x_max  y_min  y_max
        Histogram("muon/ShiftMuon_topology"                                              , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("muon/ShiftMuon_recoAlgorithm"                                         , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("dimuon/ShiftDimuonVertex_topologyCategory"                            , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("detector_diagnostics/DetectorDiagnostics_dtAttachmentFraction"        , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("detector_diagnostics/DetectorDiagnostics_trackerAttachmentFraction"   , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("detector_diagnostics/DetectorDiagnostics_timingMeasurements"          , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("detector_diagnostics/DetectorDiagnostics_timingDeltaChi2"             , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("detector_diagnostics/DetectorDiagnostics_combinedTimingDeltaChi2"     , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("target_diagnostics/TargetDiagnostics_constrainedStatus"               , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("target_diagnostics/TargetDiagnostics_targetForwardStatus"             , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("vertex_refit_diagnostics/VertexRefitDiagnostics_massErr"              , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("vertex_refit_diagnostics/VertexRefitDiagnostics_massRelativeErr"      , ""   , False, True , normalization,     1, None , None , None , None),
        Histogram("vertex_refit_diagnostics/VertexRefitDiagnostics_minQoverPSignificance", ""   , False, True , normalization,     1, None , None , None , None),
    ])

histograms2D = []
if plot_2d and not jpsi_data_histogram:
    for category in dimuon_categories:
        histograms2D.extend([
            Histogram2D(f"dimuon/ShiftDimuonVertex{category}_massVsOpeningAngle", log_x=False, log_y=False, log_z=True, norm_type=NormalizationType.none),
            Histogram2D(f"dimuon/ShiftDimuonVertex{category}_massVsPtAsymmetry" , log_x=False, log_y=False, log_z=True, norm_type=NormalizationType.none),
        ])

# fmt: on

if plot_dimuons and jpsi_data_histogram:
    for category in dimuon_categories:
        for variable in (
            "dca",
            "dcaValid",
            "isOS",
            "rapidity",
            "vertexR",
            "massOS",
            "massSS",
        ):
            histograms.append(
                Histogram(
                    f"dimuon/ShiftDimuonVertex{category}_{variable}",
                    norm_type=normalization,
                )
            )

apply_plot_style(histograms, variable_rebin=variable_histogram_rebin)
apply_2d_plot_style(histograms2D)
legends = plot_legends(bool(bsm_inputs))

if jpsi_data_histogram:
    from CmsLabelsManager import CmsLabel
    from Legend import Legend
    from Sample import SampleType
    from shift_plot_samples import prepare_jpsi_plot

    samples, histograms, luminosity = prepare_jpsi_plot(
        jpsi_data_histogram, jpsi_mc_histograms, histograms
    )
    cms_label = CmsLabel.private_data_sim
    beam_label = f"{luminosity:.12g} pb^{{-1}} recorded, pp at 13.6 TeV"
    legends = {
        SampleType.data: Legend(
            0.62, 0.79, 0.94, 0.91, "pe", title="J/#psi control region"
        ),
        SampleType.background: Legend(
            0.62, 0.44, 0.94, 0.78, "f", title="MC shape scaled to data"
        ),
    }
    for histogram in histograms:
        histogram.rebin = 1
        histogram.bin_edges = None
        histogram.x_max = None
        histogram.log_y = False
        histogram.scale_by_bin_width = False
        histogram.y_label = (
            "Dimuons / bin" if histogram.name.startswith("dimuon/") else "Events / bin"
        )

# Keep the stack (bottom to top) and legend in the same process/bin order.
custom_stacks_order = [sample.name for sample in samples]

canvas_size = (1400, 850) if bsm_inputs else (800, 600)
if bsm_inputs:
    plot_margins = {"left": 0.10, "right": 0.37, "top": 0.12, "bottom": 0.14}
canvas_size_2Dhists = (800, 700)
auto_adjust_axes_for_legend = True
show_signal_stat_uncertainty = True
show_ratio_plots = False
output_formats = ["pdf"]
