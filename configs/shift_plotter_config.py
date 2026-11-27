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
from Sample import SampleType
from shift_plot_samples import prepare_jpsi_cr_data, prepare_plot_samples
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
    plot_jpsi_control_region,
    jpsi_cr_category,
    jpsi_cr_base_path,
    jpsi_cr_data_histogram,
)

if plot_jpsi_control_region:
    from shift_plot_style import lumi_label_value

    dimuon_categories = ["" if jpsi_cr_category == "All" else jpsi_cr_category]
    plot_muons = plot_reco = plot_diagnostics = plot_2d = False
    samples, bsm_inputs = prepare_plot_samples(
        jpsi_cr_base_path,
        cross_sections,
        campaign,
        None,
        None,
        None,
        None,
        histogram_cache_directory,
        jpsi_control_region=jpsi_cr_category,
    )
    output_path = f"../plots/physics/Jpsi_CR/{campaign}/"
    if jpsi_cr_category == "All":
        output_path += "All/"
    samples.sort(key=lambda sample: sample.name.startswith("jpsi_"))
    normalization = NormalizationType.to_lumi
elif jpsi_data_histogram:
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
        f"../plots/physics/SM_BSM/{bsm_campaign}/no_analysis_selections/"
        if bsm_inputs
        else f"../plots/physics/SM/{campaign}/no_analysis_selections/"
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
            Histogram(f"event/Event_nShiftDimuonVertex{category}"            , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pt"               , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pz"               , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pt_variable"      , "", False, True          , normalization,  5, None, None, 1e-3, None, scale_by_bin_width=True      ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_pz_variable"      , "", False, True          , normalization,  5, None, None, 1e-3, None, scale_by_bin_width=True      ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_p_variable"       , "", False, True          , normalization,  5, None, None, 1e-3, None, scale_by_bin_width=True      ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_eta"              , "", False, True          , normalization,  5, None, None, 1e-3, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_phi"              , "", False, True          , normalization,  2, None, None, 1e-1, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_mass"             , "", False, False         , normalization,  2, 0   , 10  , 1e-3, None, legend_outside=not bsm_inputs),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_logMass"          , "", False, False         , normalization,  4, None, None, 1e-1, None, de_loggify_x_axis=True       ),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_vx"               , "", False, True          , normalization,  5, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_vy"               , "", False, True          , normalization,  5, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_vz"               , "", False, not bsm_inputs, normalization, 10, None, None, 1e-1, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_chi2"             , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_normalizedChi2"   , "", False, True          , normalization,  2, None, None, None, None),
            
            Histogram(f"dimuon/ShiftDimuonVertex{category}_openingAngle"     , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_openingAngleFine" , "", False, True          , normalization,  1, None, 0.2 , None, None, allow_legend_x_extension=True),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_deltaPhi"         , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_deltaEta"         , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_deltaR"           , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_ptAsymmetry"      , "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_momentumAsymmetry", "", False, True          , normalization,  1, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_minMuonPt"        , "", False, True          , normalization,  2, None, None, None, None),
            Histogram(f"dimuon/ShiftDimuonVertex{category}_maxMuonPt"        , "", False, True          , normalization,  2, None, None, None, None),
            
            
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_dca"                              , "", False, True          , normalization,  4, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_isOS"                             , "", False, True          , normalization,  1, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_rapidity"                         , "", False, True          , normalization,  2, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_vertexR"                          , "", False, True          , normalization,  2, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_massOS"                           , "", False, True          , normalization,  5, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_massSS"                           , "", False, True          , normalization,  5, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_probability"                      , "", False, True          , normalization,  1, None, None, None, None),
            # Histogram(f"dimuon/ShiftDimuonVertex{category}_originCompatibilityNormalizedChi2", "", False, True          , normalization,  2, None, None, None, None),
        ])

        # The dated refresh uses the same pointing and log bins for SM and signal.
        if not bsm_inputs or bsm_campaign == "dimuon_logs_20261124":
            histograms.extend([
                Histogram(f"dimuon/ShiftDimuonVertex{category}_ptLog"                    , log_x=True, log_y=True             , norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvDca"                    , log_y=True, norm_type=normalization, show_full_x_range=True , legend_outside=True   ),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvDcaLog"                 , log_x=True, log_y=True             , norm_type=normalization, show_full_x_range=True, legend_outside=True),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCos"            , log_y=True, norm_type=normalization, show_full_x_range=True , legend_outside=True   ),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCosLog"         , log_y=True, norm_type=normalization, show_full_x_range=True , legend_outside=True   ),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCosFine"        , log_y=True, norm_type=normalization, show_full_x_range=True , legend_outside=True   ),
                Histogram(f"dimuon/ShiftDimuonVertex{category}_pvPointingCosFineNegative", log_y=True, norm_type=normalization, show_full_x_range=True , legend_outside=True   ),
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

if plot_dimuons and jpsi_data_histogram and not plot_jpsi_control_region:
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

if plot_jpsi_control_region:
    from CmsLabelsManager import CmsLabel
    from Legend import Legend
    from Logger import info

    # The compact dimuon skim does not carry these observables.
    missing_from_skim = {
        "openingAngle",
        "openingAngleFine",
        "deltaPhi",
        "deltaEta",
        "deltaR",
        "ptAsymmetry",
        "momentumAsymmetry",
        "minMuonPt",
        "maxMuonPt",
        "massSS",
    }
    histograms = [
        hist
        for hist in histograms
        if hist.name.split("_", 1)[-1] not in missing_from_skim
    ]
    data, data_histograms, recorded_lumi = prepare_jpsi_cr_data(
        jpsi_cr_data_histogram, histograms, jpsi_cr_category
    )
    if jpsi_cr_data_histogram is not None and not data_histograms:
        raise ValueError(
            "The selected J/ψ control region has no usable data histograms"
        )
    legends[SampleType.background].title = "J/#psi control region"
    if data is not None:
        recorded = "Recorded CMS lumi unknown"
        if recorded_lumi is not None:
            recorded = f"{recorded_lumi:.12g} pb^{{-1}} recorded CMS lumi"
        info(
            f"J/ψ CR data: {recorded}. Absolute SHIFT MC normalization is unqualified."
        )
        if data_histograms:
            normalization = (
                NormalizationType.to_data if samples else NormalizationType.none
            )
            cms_label = CmsLabel.private_data_sim if samples else CmsLabel.private_data
            del lumi_label_value
            beam_label = recorded
            legends[SampleType.data] = Legend(
                0.71, 0.77, 0.98, 0.86, "pe", title="J/#psi control region", text_size=16
            )
            legends[SampleType.background] = Legend(
                0.71,
                0.2,
                0.98,
                0.76,
                "f",
                title="MC shape scaled to data",
                text_size=16,
            )
            samples.append(data)
            histograms = data_histograms
    for histogram in histograms:
        histogram.norm_type = normalization
        if histogram.name.endswith("_logMass"):
            histogram.x_min = 0.40  # Display the full 2.6–3.6 GeV window on log10(m).
            histogram.x_max = 0.57
            histogram.y_min = None
            histogram.legend_outside = True
elif jpsi_data_histogram:
    from CmsLabelsManager import CmsLabel
    from Legend import Legend
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
        if not histogram.name.endswith("_logMass"):
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
has_data = any(sample.type == SampleType.data for sample in samples)
show_ratio_plots = has_data and any(sample.type == SampleType.background for sample in samples)
if has_data:
    plotting_options = {**plotting_options, SampleType.data: "nostack p e1 x0"}
if show_ratio_plots and plot_jpsi_control_region:
    canvas_size = (1200, 850)
    plot_margins = {"left": 0.13, "right": 0.32, "top": 0.10, "bottom": 0.13}
output_formats = ["pdf"]
