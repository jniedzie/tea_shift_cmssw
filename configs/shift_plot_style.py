# Code style: human-code.
"""Axis labels, display conventions and reconstruction report settings."""

from Legend import Legend
from Sample import SampleType
from CmsLabelsManager import CmsLabel
from shift_plot_binning import muon_variable_binning, dimuon_variable_binning
from shift_plot_labels import apply_reconstruction_labels

muon_axis_labels = {
    "pt": ("p_{T}^{#mu} [GeV]", "# events"),
    "pz": ("p_{Z}^{#mu} [GeV]", "# events"),
    "pt_variable": ("p_{T}^{#mu} [GeV]", "# events / GeV"),
    "pz_variable": ("p_{Z}^{#mu} [GeV]", "# events / GeV"),
    "p_variable": ("p^{#mu} [GeV]", "# events / GeV"),
    "eta": ("#eta^{#mu}", "# events"),
    "phi": ("#phi^{#mu}", "# events"),
    "vx": ("v_{x}^{#mu} [cm]", "# events"),
    "vy": ("v_{y}^{#mu} [cm]", "# events"),
    "vz": ("v_{z}^{#mu} [cm]", "# events"),
    "dz": ("d_{z}^{#mu} [cm]", "# events"),
    "nCSCHits": ("n_{CSCHits}^{#mu}", "# events"),
    "nDTHits": ("n_{DTHits}^{#mu}", "# events"),
    "nRPCHits": ("n_{RPCHits}^{#mu}", "# events"),
    "constrainedPt": ("p_{T}^{#mu, constrained} [GeV]", "# muons"),
    "constrainedPz": ("p_{z}^{#mu, constrained} [GeV]", "# muons"),
    "constrainedEta": ("#eta^{#mu, constrained}", "# muons"),
    "constrainedPhi": ("#phi^{#mu, constrained}", "# muons"),
    "constrainedVz": ("v_{z}^{#mu, constrained} [cm]", "# muons"),
    "linePcaZ": ("z_{PCA}^{#mu} [cm]", "# muons"),
    "trackVz": ("z_{track}^{#mu} [cm]", "# muons"),
    "nGEMHits": ("n_{GEM hits}^{#mu}", "# muons"),
    "nME0Hits": ("n_{ME0 hits}^{#mu}", "# muons"),
}

dimuon_axis_labels = {
    "pt": ("p_{T}^{#mu#mu} [GeV]", "# events"),
    "pz": ("p_{Z}^{#mu#mu} [GeV]", "# events"),
    "pt_variable": ("p_{T}^{#mu#mu} [GeV]", "# events / GeV"),
    "pz_variable": ("p_{Z}^{#mu#mu} [GeV]", "# events / GeV"),
    "p_variable": ("p^{#mu#mu} [GeV]", "# events / GeV"),
    "eta": ("#eta^{#mu#mu}", "# events"),
    "phi": ("#phi^{#mu#mu}", "# events"),
    "mass": ("m_{#mu#mu} [GeV]", "# events"),
    "vx": ("v_{x}^{#mu#mu} [cm]", "# events"),
    "vy": ("v_{y}^{#mu#mu} [cm]", "# events"),
    "vz": ("v_{z}^{#mu#mu} [cm]", "# events"),
    "chi2": ("#chi^{2}", "# events"),
    "normalizedChi2": ("#chi^{2}/ndof", "# events"),
    "dca": ("DCA [cm]", "# events"),
    "isOS": ("isOS", "# events"),
    "genPid": ("generator mother", "# dimuons"),
    "genPidDifferentMothers": ("generator mother", "# dimuon mothers"),
    "openingAngle": ("#alpha_{#mu#mu} [rad]", "# dimuons"),
    "openingAngleFine": ("#alpha_{#mu#mu} [rad]", "# dimuons"),
    "constrainedOpeningAngle": ("#alpha_{#mu#mu}^{constrained} [rad]", "# dimuons"),
    "deltaPhi": ("|#Delta#phi_{#mu#mu}| [rad]", "# dimuons"),
    "deltaEta": ("|#Delta#eta_{#mu#mu}|", "# dimuons"),
    "deltaR": ("#Delta R_{#mu#mu}", "# dimuons"),
    "ptAsymmetry": ("|p_{T,1}-p_{T,2}|/(p_{T,1}+p_{T,2})", "# dimuons"),
    "momentumAsymmetry": ("|p_{1}-p_{2}|/(p_{1}+p_{2})", "# dimuons"),
    "minMuonPt": ("min(p_{T,1},p_{T,2}) [GeV]", "# dimuons"),
    "maxMuonPt": ("max(p_{T,1},p_{T,2}) [GeV]", "# dimuons"),
    "rapidity": ("y_{#mu#mu}", "# dimuons"),
    "vertexR": ("r_{vertex}^{#mu#mu} [cm]", "# dimuons"),
    "massOS": ("m_{#mu^{+}#mu^{-}} [GeV]", "# dimuons"),
    "massSS": ("m_{#mu^{#pm}#mu^{#pm}} [GeV]", "# dimuons"),
    "constrainedMass": ("m_{#mu#mu}^{constrained} [GeV]", "# dimuons"),
    "refittedMass": ("m_{#mu#mu}^{refit} [GeV]", "# dimuons"),
    "probability": ("vertex fit probability", "# dimuons"),
    "originCompatibilityNormalizedChi2": ("#chi^{2}_{origin}/3", "# dimuons"),
    "constrainedValid": ("valid constrained vertex", "# dimuons"),
    "refitStatus": ("joint vertex refit status", "# dimuons"),
    "dcaValid": ("valid two-track DCA", "# dimuons"),
    "logMass": ("log_{10}(m_{#mu#mu} / GeV)", "# dimuons"),
}

other_axis_labels = {
    "event/Event_nShiftMuon": ("n_{#mu}", "# events"),
    "event/Event_nShiftDimuonVertex": ("n_{#mu#mu}", "# events"),
    "muon/ShiftMuon_topology": ("muon topology", "# entries"),
    "muon/ShiftMuon_recoAlgorithm": ("muon reconstruction algorithm", "# entries"),
    "dimuon/ShiftDimuonVertex_topologyCategory": ("dimuon topology", "# entries"),
    "detector_diagnostics/DetectorDiagnostics_dtAttachmentFraction": ("DT attachment fraction", "# entries"),
    "detector_diagnostics/DetectorDiagnostics_trackerAttachmentFraction": ("tracker attachment fraction", "# entries"),
    "detector_diagnostics/DetectorDiagnostics_timingMeasurements": ("timing measurements per muon", "# entries"),
    "detector_diagnostics/DetectorDiagnostics_timingDeltaChi2": ("#Delta#chi^{2}_{muon timing}", "# entries"),
    "detector_diagnostics/DetectorDiagnostics_combinedTimingDeltaChi2": ("#Delta#chi^{2}_{combined timing}", "# entries"),
    "target_diagnostics/TargetDiagnostics_constrainedStatus": ("target fit status", "# entries"),
    "target_diagnostics/TargetDiagnostics_targetForwardStatus": ("forward propagation status", "# entries"),
    "vertex_refit_diagnostics/VertexRefitDiagnostics_massErr": ("#sigma(m_{#mu#mu}^{refit}) [GeV]", "# entries"),
    "vertex_refit_diagnostics/VertexRefitDiagnostics_massRelativeErr": ("#sigma(m_{#mu#mu}^{refit})/m_{#mu#mu}^{refit}", "# entries"),
    "vertex_refit_diagnostics/VertexRefitDiagnostics_minQoverPSignificance": ("min |q/p|/#sigma(q/p)", "# entries"),
}

muon_axis_labels["topology"] = other_axis_labels["muon/ShiftMuon_topology"]
muon_axis_labels["recoAlgorithm"] = other_axis_labels["muon/ShiftMuon_recoAlgorithm"]
dimuon_axis_labels["topologyCategory"] = other_axis_labels["dimuon/ShiftDimuonVertex_topologyCategory"]


def apply_plot_style(histograms, variable_rebin=1):
    """Apply labels and density binning; keep each plot's log axes and limits."""
    for histogram in histograms:
        name = histogram.name
        if name.startswith("muon/ShiftMuon"):
            # Momentum variables contain an underscore in their suffix.
            variable = name.split("_", 1)[1]
            histogram.x_label, histogram.y_label = muon_axis_labels[variable]
            rules, object_label = muon_variable_binning, "# muons"
        elif name.startswith("dimuon/ShiftDimuonVertex"):
            variable = name.split("_", 1)[1]
            histogram.x_label, histogram.y_label = dimuon_axis_labels[variable]
            if variable == "logMass" and histogram.de_loggify_x_axis:
                histogram.x_label = "m_{#mu#mu} (GeV)"
            rules, object_label = dimuon_variable_binning, "# dimuons"
        else:
            label_key = name
            if name.startswith("event/Event_nShiftDimuonVertex"):
                label_key = "event/Event_nShiftDimuonVertex"
            elif name.startswith("event/Event_nShiftMuon"):
                label_key = "event/Event_nShiftMuon"
            histogram.x_label, histogram.y_label = other_axis_labels[label_key]
            continue

        if name.endswith("_variable"):
            histogram.scale_by_bin_width = True
            histogram.rebin *= variable_rebin
            continue
        if variable not in rules:
            continue
        histogram.bin_edges = None if variable_rebin == 1 else rules[variable]
        histogram.rebin *= variable_rebin
        histogram.scale_by_bin_width = True
        unit = next(
            (unit for unit in ("GeV", "cm", "rad") if f"[{unit}]" in histogram.x_label),
            "unit x",
        )
        histogram.y_label = f"{object_label} / {unit}"

    apply_reconstruction_labels(histograms)


def apply_2d_plot_style(histograms):
    for histogram in histograms:
        variable = histogram.name.rsplit("_", 1)[-1]
        x_variable = {"massVsOpeningAngle": "openingAngle", "massVsPtAsymmetry": "ptAsymmetry"}[variable]
        histogram.x_label = dimuon_axis_labels[x_variable][0]
        histogram.y_label = "m_{#mu#mu} [GeV]"
        histogram.z_label = "weighted dimuons"


def plot_legends(compare_bsm):
    if compare_bsm:
        return {
            SampleType.background: Legend(0.65, 0.48, 0.99, 0.90, "f", title="SM", text_size=16),
            SampleType.signal: Legend(0.65, 0.12, 0.99, 0.45, "l",
                                      title="BSM (normalized to SM integral)", text_size=17),
        }
    return {SampleType.background: Legend(0.58, 0.2, 0.9, 0.9, "f", text_size=16)}


luminosity = 0.01 * 300000.0  # pb^-1 (1% of Run 3)
show_cms_labels = True
cms_label = CmsLabel.pas_sim
lumi_label_value = luminosity
label_outside_axes = True
beam_label = "pp @ 13.6 TeV"
plotting_options = {
    SampleType.background: "hist",
    SampleType.signal: "nostack hist",
    SampleType.data: "nostack e",
}
categorical_canvas_size = (1200, 850)
categorical_legend_outside = True

# Truth-binned reconstruction response; rendered by shift_diagnostics_plotter.
# Keep these settings literal so diagnostics can read them without campaign I/O.
reconstructionResponseAxes = {
    "pt": "True p_{T} [GeV]",
    "pz": "True p_{z} [GeV]",
    "p": "True |p| [GeV]",
    "mass": "True dimuon mass [GeV]",
    "vz": "True production z [cm]",
    "decayR": "True production radius [cm]",
    "flight": "True parent flight [cm]",
    "ctau": "True parent proper decay length [cm]",
    "openingAngle": "True dimuon opening angle [rad]",
}
reconstructionResponseVariables = {
    "pt": "(p_{T}^{reco}-p_{T}^{true})/p_{T}^{true}",
    "pz": "(p_{z}^{reco}-p_{z}^{true})/p_{z}^{true}",
    "p": "(|p|^{reco}-|p|^{true})/|p|^{true}",
    "mass": "(m^{reco}-m^{true})/m^{true}",
    "eta": "#eta^{reco}-#eta^{true}",
    "phi": "#phi^{reco}-#phi^{true} [rad]",
    "vx": "x^{reco}-x^{true} [cm]",
    "vy": "y^{reco}-y^{true} [cm]",
    "vz": "z^{reco}-z^{true} [cm]",
    "openingAngle": "#alpha^{reco}-#alpha^{true} [rad]",
}
reconstructionResponseMinimumEntries = 20
reconstructionDiagnosticVariables = {
    "Muon": (
        "pt",
        "pz",
        "eta",
        "phi",
        "vx",
        "vy",
        "vz",
        "nCSCHits",
        "nDTHits",
        "nRPCHits",
        "dz",
        "linePcaZ",
        "trackVz",
    ),
    "Dimuon": (
        "pt",
        "pz",
        "eta",
        "phi",
        "mass",
        "vx",
        "vy",
        "vz",
        "normalizedChi2",
        "dca",
        "isOS",
    ),
}

reconstructionResponseDisplayRanges = {
    "pt": (-5.0, 5.0),
    "pz": (-5.0, 5.0),
    "p": (-5.0, 5.0),
    "mass": (-5.0, 5.0),
    "eta": (-15.0, 15.0),
    "phi": (-3.142, 3.142),
    "openingAngle": (-3.142, 3.142),
    "vx": (-5000.0, 5000.0),
    "vy": (-5000.0, 5000.0),
    "vz": (-30000.0, 30000.0),
}
reconstructionResponseLogY = True
# Keep the core readable; the plotter prints each process's fraction outside.
reconstructionReviewResidualRanges = {
    "pt": (-1.0, 1.5),
    "p": (-1.0, 1.5),
    "mass": (-1.0, 1.5),
    "vx": (-70.0, 70.0),
    "vy": (-70.0, 70.0),
    "vz": (-2500.0, 2500.0),
}
reconstructionDiagnosticLabels = {
    "pt": "p_{T} [GeV]",
    "pz": "p_{z} [GeV]",
    "eta": "#eta",
    "phi": "#phi [rad]",
    "mass": "Dimuon mass [GeV]",
    "vx": "x [cm]",
    "vy": "y [cm]",
    "vz": "z [cm]",
    "dz": "Track d_{z} [cm]",
    "linePcaZ": "Track-axis PCA z [cm]",
    "trackVz": "Track reference z [cm]",
    "nCSCHits": "CSC hits",
    "nDTHits": "DT hits",
    "nRPCHits": "RPC hits",
    "normalizedChi2": "#chi^{2}/ndof",
    "dca": "Muon-pair DCA [cm]",
    "isOS": "Opposite-charge pair",
}

# Category reports keep fit states side by side and put diagonal comparisons first.
reconstructionReportCategories = {
    "Muon": {
        "BothEndcaps": "Both",
        "NearEndcapOnly": "Near",
        "FarEndcapOnly": "Far",
        "NearEndcapAndBarrel": "Near+Barrel",
        "Unclassified": "Unclassified",
    },
    "Dimuon": {
        "": "Inclusive",
        "Both-Both": "Both-Both",
        "Near-Both": "Near-Both",
        "Both-Far": "Both-Far",
        "Near-Far": "Near-Far",
        "Other": "Other",
    },
}
reconstructionResponseOverview = ("mass", "pt", "pz", "p", "vz", "openingAngle")
reconstructionResponseTitles = {
    "pt": "relative p_{T}",
    "pz": "relative p_{z}",
    "p": "relative |p|",
    "mass": "relative mass",
    "eta": "#Delta#eta",
    "phi": "#Delta#phi [rad]",
    "vx": "#Deltax [cm]",
    "vy": "#Deltay [cm]",
    "vz": "#Deltaz [cm]",
    "openingAngle": "#Delta#alpha [rad]",
}
