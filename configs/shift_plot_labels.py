"""Stored reconstruction codes from ShiftMuonTableProducer and its fit helpers.

Keep numerical codes in the labels so failures remain traceable to NanoAOD.
The forward target code -1 also represents an unavailable diagnostic.
"""

MUON_TOPOLOGY_LABELS = {
    0: "0: near endcap only",
    1: "1: near endcap + barrel",
    2: "2: both endcaps",
    3: "3: far endcap only",
    4: "4: unclassified",
}
MUON_ALGORITHM_LABELS = {
    0: "0: displaced standalone",
    1: "1: strict traversing",
    2: "2: ordinary cosmic",
}
VERTEX_REFIT_LABELS = {
    -32: "-32: invalid charge",
    -31: "-31: transport call limit",
    -30: "-30: invalid derivative",
    -27: "-27: iteration limit",
    -26: "-26: no descent",
    -25: "-25: singular fit",
    -23: "-23: forward covariance",
    -22: "-22: forward transport",
    -21: "-21: invalid input",
    -20: "-20: wrong propagator",
    -4: "-4: no convergence",
    -3: "-3: invalid covariance/update",
    -2: "-2: transport failed",
    -1: "-1: missing detector state",
    0: "0: disabled/unpaired",
    1: "1: valid",
}
TARGET_FIT_LABELS = {
    -11: "-11: path integration failed",
    -10: "-10: no detector refit",
    -9: "-9: curvature sign crossing",
    -8: "-8: forward fit failed",
    -7: "-7: numerical covariance",
    -6: "-6: target update failed",
    -5: "-5: residual covariance",
    -4: "-4: target uncertainty/slopes",
    -3: "-3: backward transport failed",
    -1: "-1: invalid detector state",
    0: "0: disabled/not attempted",
    1: "1: Kalman target update",
    2: "2: forward likelihood",
    3: "3: path integration",
    4: "4: forward moment estimator",
}
TARGET_FORWARD_LABELS = {
    -10: "-10: unresolved derivative",
    -7: "-7: iteration limit",
    -6: "-6: no descent",
    -5: "-5: singular/invalid fit matrix",
    -4: "-4: derivative setup",
    -3: "-3: prediction/target covariance",
    -2: "-2: forward transport failed",
    -1: "-1: unavailable/invalid input",
    1: "1: valid",
}


def apply_reconstruction_labels(histograms):
    for histogram in histograms:
        name = histogram.name
        variable = name.rsplit("_", 1)[-1]
        if name.startswith("muon/ShiftMuon") and variable == "topology":
            histogram.bin_labels = MUON_TOPOLOGY_LABELS
        elif name.startswith("muon/ShiftMuon") and variable == "recoAlgorithm":
            histogram.bin_labels = MUON_ALGORITHM_LABELS
        elif name.startswith("dimuon/ShiftDimuonVertex") and variable == "refitStatus":
            histogram.bin_labels = VERTEX_REFIT_LABELS
        elif name == "target_diagnostics/TargetDiagnostics_constrainedStatus":
            histogram.bin_labels = TARGET_FIT_LABELS
        elif name == "target_diagnostics/TargetDiagnostics_targetForwardStatus":
            histogram.bin_labels = TARGET_FORWARD_LABELS
