# Code style: human-code.
# ============================================================
# Default single-muon histograms
# ============================================================

from Logger import info
from shift_paths import base_path, campaign, sample
from shift_extra_collections import extraEventCollections
from shift_sample_paths import latest_merged_sample
from shift_plot_binning import (
    linear_bin_edges,
    dimuon_opening_angle_edges,
    dimuon_fine_opening_angle_edges,
    dimuon_origin_chi2_edges,
)
from shift_storage_paths import resolve_storage_path
import re
import glob
import os

# Truth is used only for MC performance evaluation, never reco selection.
enableTruthDiagnostics = True

# Optional truth-binned response maps for the reconstruction study.
enableResolutionMaps = False
resolutionMapsOnly = False
resolutionMapAxes = {
    "pt": [0., .5, 1., 2., 3., 5., 10., 20., 30., 50., 100., 200.],
    "pz": [-5000., -2000., -1000., -500., -200., -100., -50., -20., -5., 0., 100., 1000.],
    "p": [0., 5., 10., 20., 50., 100., 200., 500., 1000., 2000., 5000.],
    "mass": [0., .211, .3, .5, 1., 2., 3., 4., 5., 10., 15., 20., 30., 50., 70., 100., 200.],
    "vz": [-20000., -10000., -1000., 0., 1000., 5000., 10000., 13000., 14000., 14700., 14800., 14900., 15000., 20000., 40000.],
    "decayR": [0., .01, .1, 1., 10., 50., 100., 300., 1000., 10000., 100000.],
    "flight": [0., .01, .1, 1., 10., 100., 500., 1000., 3000., 10000., 30000., 100000., 1000000.],
    "ctau": [0., 1.e-8, 1.e-6, 1.e-4, .01, .1, 1., 10., 100., 1000., 10000., 100000.],
    "openingAngle": [0., .0001, .0003, .001, .003, .01, .03, .1, .3, 1., 2., 3.142],
}
resolutionMapResiduals = {
    "pt": (-1.e9, 1.e9), "pz": (-1.e9, 1.e9), "p": (-1.e9, 1.e9), "mass": (-1.e9, 1.e9),
    "eta": (-20., 20.), "phi": (-3.142, 3.142), "openingAngle": (-3.142, 3.142),
    "vx": (-1.e6, 1.e6), "vy": (-1.e6, 1.e6), "vz": (-1.e6, 1.e6),
}


def book_resolution_maps():
    maps = []
    for kind, categories in (("Muon", muonCategories), ("Dimuon", dimuonCategories)):
        for category in categories:
            for state in ("", "Constrained"):
                for variable, (low, high) in resolutionMapResiduals.items():
                    if kind == "Muon" and variable in ("mass", "openingAngle"):
                        continue
                    # Fine core and wide tails. Flow counts remain explicit.
                    core = 1.
                    if variable in ("vx", "vy"):
                        core = 500.
                    if variable == "vz":
                        core = 3000.
                    if variable in ("eta", "phi"):
                        core = .2
                    if variable == "openingAngle":
                        core = .01
                    residual_edges = [core * i / 100. for i in range(-100, 101)]
                    tail = core * 1.15
                    while tail < high:
                        residual_edges.extend((-tail, tail))
                        tail *= 1.15
                    residual_edges = sorted([low, high] + residual_edges)
                    for axis, edges in resolutionMapAxes.items():
                        name = f"{kind}Response{category}{state}_{variable}Vs{axis}"
                        maps.append((name, edges, residual_edges, "response"))
    return tuple(maps)

# Saved scores are diagnostics only; unscored Nano inputs remain supported.
enableBdtDiagnostics = True
requireBdtScore = False
bdtTruthDiagnostics = True

# The same application can run the optional event-mixing preparation stage.
runEventMixing = False
eventMixing = {
    "pool": None, "inventory": None, "output": None,
    "config_json": None, "fit": None,
    "partners": 20, "bootstrap": 100, "seed": 20261009,
    "extract": False, "campaign": campaign, "production_directory": None,
    "workers": 8, "executor": "process",
    # Optional source-event uncertainty and coverage checks stay in utils.
    "diagnostics": False,
}


def run_mixing_stage(settings=None):
    import sys
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from utils.shift_mixing import run_event_mixing
    return run_event_mixing(eventMixing if settings is None else settings)

# fmt: off
defaultHistParams = (
    ("ShiftMuon"        , "topology"     , 5, -0.5, 4.5, "muon"  ),
    ("ShiftMuon"        , "recoAlgorithm", 3, -0.5, 2.5, "muon"  ),
    ("ShiftDimuonVertex", "topologyMin"  , 5, -0.5, 4.5, "dimuon"),
    ("ShiftDimuonVertex", "topologyMax"  , 5, -0.5, 4.5, "dimuon"),
)
# fmt: on

# fmt: off
defaultHistParams += (
    ("ShiftMuon", "simDTHits"                                               ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "simPixelHits"                                            ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "simStripHits"                                            ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "simGEMHits"                                              ,   20,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchValid"                                       ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("ShiftMuon", "trackerValidHits"                                        ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "combinedTrackValid"                                      ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("ShiftMuon", "combinedTrackerHits"                                     ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "combinedTargetPt"                                        ,  200,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "combinedTargetPz"                                        ,  200, -1000,   100, "detector_diagnostics"),
    ("ShiftMuon", "combinedTargetDca"                                       ,  100,    -1,   500, "detector_diagnostics"),
    ("ShiftMuon", "simHcalHits"                                             ,   50,    -2,    48, "detector_diagnostics"),
    ("ShiftMuon", "simHBHEHits"                                             ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "simHFHits"                                               ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "simHOHits"                                               ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "simHcalEnergy"                                           ,  100,    -2,     2, "detector_diagnostics"),
    ("ShiftMuon", "simZDCHits"                                              ,   50,    -2,    48, "detector_diagnostics"),
    ("ShiftMuon", "simZDCEnergy"                                            ,  100,    -2,     2, "detector_diagnostics"),
    ("ShiftMuon", "simZDCFirstTime"                                         ,  120,  -600,     0, "detector_diagnostics"),
    ("ShiftMuon", "nCompatibleDTSegments"                                   ,   20,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "nPropagatedDTSegments"                                   ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "nPropagatedPixelHits"                                    ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "nPropagatedStripHits"                                    ,  200,     0,   200, "detector_diagnostics"),
    ("ShiftMuon", "nCompatiblePixelHits"                                    ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "nCompatibleStripHits"                                    ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "nAddedDTRefitHits"                                       ,   20,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "nAddedTrackerRefitHits"                                  ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "nAddedDTTruthChamberMatches"                             ,   20,    -1,    19, "detector_diagnostics"),
    ("ShiftMuon", "minDTResidual"                                           ,  200,    -1,   999, "detector_diagnostics"),
    ("ShiftMuon", "minDTEstimatorChi2"                                      ,  200,    -1,   999, "detector_diagnostics"),
    ("ShiftMuon", "minTrackerResidual"                                      ,  200,    -1,   199, "detector_diagnostics"),
    ("ShiftMuon", "minTrackerEstimatorChi2"                                 ,  200,    -1,   999, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchRawValid"                                    ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchValid"                                       ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchLineDistance"                                ,  200,    -1,   999, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchAxisAngle"                                   ,  160, -0.01,  1.59, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedEcalRecHits"                                     ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedHBHERecHits"                                     ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedHFRecHits"                                       ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedHORecHits"                                       ,   50,     0,    50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedZDCRecHits"                                      ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "matchedEcalEnergy"                                       ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "matchedHBHEEnergy"                                       ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "matchedHFEnergy"                                         ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "matchedHOEnergy"                                         ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHBHEIds"                                         ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHBHERecHits"                                     ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "nValidCrossedHBHETimes"                                  ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "crossedHBHEEnergy"                                       ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "hbhe3x3Energy"                                           ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHBHEEnergy"                                    ,  100,     0,    10, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHBHETime"                                      ,  160, -1000,   200, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHOIds"                                           ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHORecHits"                                       ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "nValidCrossedHOTimes"                                    ,   30,     0,    30, "detector_diagnostics"),
    ("ShiftMuon", "hcalAssociationDirection"                                ,    4,  -0.5,   3.5, "detector_diagnostics"),
    ("ShiftMuon", "crossedHOEnergy"                                         ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "ho3x3Energy"                                             ,  100,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHOEnergy"                                      ,  100,     0,    10, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHOTime"                                        ,  160, -1000,   200, "detector_diagnostics"),
    ("ShiftMuon", "matchedZDCEnergy"                                        ,  100,     0,    10, "detector_diagnostics"),
    ("ShiftMuon", "nCaloTimingMeasurements"                                 ,   10,     0,    10, "detector_diagnostics"),
    ("ShiftMuon", "caloTimingDirectionSign"                                 ,    3,  -1.5,   1.5, "detector_diagnostics"),
    ("ShiftMuon", "caloTimingDeltaChi2"                                     ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "combinedTimingDirectionSign"                             ,    3,  -1.5,   1.5, "detector_diagnostics"),
    ("ShiftMuon", "nCombinedTimingMeasurements"                             ,   20,     0,    20, "detector_diagnostics"),
    ("ShiftMuon", "combinedTimingDeltaChi2"                                 ,  100,     0,   100, "detector_diagnostics"),
    ("ShiftMuon", "combinedTimingAgreesWithMuon"                            ,    3,  -1.5,   1.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_enableDTMeasurement"                       ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_dtNavigationMode"                          ,    3,  -0.5,   2.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_enableGEMMeasurement"                      ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_trackerMode"                               ,    3,  -0.5,   2.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_enableHcalDiagnostics"                     ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_enableZDCDiagnostics"                      ,    2,  -0.5,   1.5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nDTSimHits"                                ,  100,     0,   100, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nDTSegments"                               ,   50,     0,    50, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nGEMSimHits"                               ,   20,     0,    20, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nGEMSegments"                              ,   20,     0,    20, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nGeneralTracks"                            ,   50,     0,    50, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTrackerSeeds"                             ,   30,     0,    30, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTrackerTrackCandidates"                   ,   30,     0,    30, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTrackerRawTracks"                         ,   30,     0,    30, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTrackerSelectedTracks"                    ,   30,     0,    30, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTrackerLHCTrackCandidates"                ,   30,     0,    30, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTrackerLHCTracks"                         ,   30,     0,    30, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nSignalMuonSimTracks"                      ,   10,     0,    10, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nSignalMuonWithTrackerSimHits"             ,   10,     0,    10, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nSignalMuonWithMuonSystemSimHits"          ,   10,     0,    10, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nSignalMuonWithTrackerAndMuonSystemSimHits",   10,     0,    10, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nDSATrackerMatches"                        ,   10,     0,    10, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nTraversingTrackerMatches"                 ,   10,     0,    10, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nSignalMuonHcalSimHits"                    ,   50,    -2,    48, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_signalMuonHcalSimEnergy"                   ,  100,    -2,     2, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nSignalMuonZDCSimHits"                     ,   50,    -2,    48, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_signalMuonZDCSimEnergy"                    ,  100,    -2,     2, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_signalMuonZDCFirstTime"                    ,  120,  -600,     0, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nPixelRecHits"                             ,  200,    -1,   199, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nStripMatchedRecHits"                      ,  500,    -1,   499, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nStripRphiRecHits"                         , 3000,    -1,  2999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nStripRphiUnmatchedRecHits"                , 2000,    -1,  1999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nStripStereoRecHits"                       , 1500,    -1,  1499, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nStripStereoUnmatchedRecHits"              , 1000,    -1,   999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHBHEQIE11Digis"                           ,  200,     0, 20000, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHFQIE10Digis"                             ,  200,     0,  5000, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHODigis"                                  ,  200,     0,  3000, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nZDCDigis"                                 ,   30,    -1,    29, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nEcalBarrelRecHits"                        ,  200,    -1,   199, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nEcalEndcapRecHits"                        ,  200,    -1,   199, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHBHERecHits"                              ,  200,    -1,  3999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHFRecHits"                                ,  200,    -1,  1999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHORecHits"                                ,  200,    -1,  2999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nZDCRecHits"                               ,   50,    -1,    49, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nZDCRecHitsAboveThreshold"                 ,   30,    -1,    29, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_ZDCPositiveRecEnergy"                      ,  100,    -1,    20, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_ZDCMaxRecEnergy"                           ,  100,    -1,     5, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_ZDCMaxRecTime"                             ,  160,  -600,   200, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHBHERecHitsAboveThreshold"                ,  200,    -1,  3999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_HBHEPositiveRecEnergy"                     ,  200,    -1,   200, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHFRecHitsAboveThreshold"                  ,  200,    -1,  1999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_HFPositiveRecEnergy"                       ,  200,    -1,   200, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nEcalBarrelRecHitsAboveThreshold"          ,  200,    -1,   399, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_EcalBarrelPositiveRecEnergy"               ,  200,    -1,   100, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nEcalEndcapRecHitsAboveThreshold"          ,  200,    -1,   399, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_EcalEndcapPositiveRecEnergy"               ,  200,    -1,   100, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nHORecHitsAboveThreshold"                  ,  200,    -1,  2999, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_HOPositiveRecEnergy"                       ,  200,    -1,   200, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nBCM1FSimHits"                             ,   50,    -1,    49, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nBHMSimHits"                               ,   50,    -1,    49, "detector_diagnostics"),
    ("Event"    , "ShiftRecoDiag_nPLTSimHits"                               ,   50,    -1,    49, "detector_diagnostics"),
)
# fmt: on

muonCategories = [
    "NearEndcapOnly",
    "NearEndcapAndBarrel",
    "BothEndcaps",
    "FarEndcapOnly",
    "Unclassified",
]

for name in ["GenMuon", "ShiftMuon"] + [
    f"ShiftMuon{category}" for category in muonCategories
]:
    # fmt: off
    defaultHistParams += (
        ("Event"  , f"n{name}",   10,      0,    10, "event"),
        (f"{name}", "pt"      ,  500,      0,    50, "muon" ),
        (f"{name}", "pz"      ,  500,  -2000,   100, "muon" ),
        (f"{name}", "eta"     , 1000,    -10,    10, "muon" ),
        (f"{name}", "phi"     ,  100,   -3.2,   3.2, "muon" ),
        (f"{name}", "vx"      , 1000,   -300,   300, "muon" ),
        (f"{name}", "vy"      , 1000,   -300,   300, "muon" ),
        (f"{name}", "vz"      , 1000, -20000, 40000, "muon" ),
    )
    # fmt: on

    if "Gen" in name:
        continue

    # fmt: off
    defaultHistParams += (
        (f"{name}", "constrainedPt" ,  500,      0,    50, "muon"),
        (f"{name}", "constrainedPz" ,  500,  -2000,   100, "muon"),
        (f"{name}", "constrainedEta",  100,    -10,    10, "muon"),
        (f"{name}", "constrainedPhi",  100,   -3.2,   3.2, "muon"),
        (f"{name}", "constrainedVx" , 1000,   -300,   300, "muon"),
        (f"{name}", "constrainedVy" , 1000,   -300,   300, "muon"),
        (f"{name}", "constrainedVz" , 1000, -20000, 40000, "muon"),
        (f"{name}", "nCSCHits"      ,  200,      0,   200, "muon"),
        (f"{name}", "nDTHits"       ,  100,      0,   100, "muon"),
        (f"{name}", "nRPCHits"      ,  100,      0,   100, "muon"),
        (f"{name}", "nGEMHits"      ,  100,      0,   100, "muon"),
        (f"{name}", "nME0Hits"      ,  100,      0,   100, "muon"),
        (f"{name}", "dz"            ,  500, -20000, 40000, "muon"),
        (f"{name}", "linePcaZ"      ,  100, -20000, 20000, "muon"),
        (f"{name}", "trackVz"       ,  100, -20000, 20000, "muon"),
        (f"{name}", "genPartIdx"    ,   30,    -10,    20, "muon"),
    )
    # fmt: on

# ============================================================
# Default dimuon histograms
# ============================================================

dimuonCategories = [
    "",
    "Good",
    "Near-Both",
    "Near-Far",
    "Both-Both",
    "Both-Far",
    "Other",
]

for name in dimuonCategories:
    # fmt: off
    defaultHistParams += (
        ("Event"                   , f"nShiftDimuonVertex{name}",   10,      0,    10, "event" ),
        (f"ShiftDimuonVertex{name}", "pt"                       , 2000,      0,   200, "dimuon"),
        (f"ShiftDimuonVertex{name}", "pz"                       ,  200,  -1000,   100, "dimuon"),
        (f"ShiftDimuonVertex{name}", "eta"                      , 1000,    -10,    10, "dimuon"),
        (f"ShiftDimuonVertex{name}", "phi"                      ,  100,   -3.2,   3.2, "dimuon"),
        (f"ShiftDimuonVertex{name}", "mass"                     , 2000,      0,   200, "dimuon"),
        (f"ShiftDimuonVertex{name}", "vx"                       , 1000,   -300,   300, "dimuon"),
        (f"ShiftDimuonVertex{name}", "vy"                       , 1000,   -300,   300, "dimuon"),
        (f"ShiftDimuonVertex{name}", "vz"                       , 1000, -20000, 20000, "dimuon"),
        (f"ShiftDimuonVertex{name}", "chi2"                     , 1000,     -1,   100, "dimuon"),
        (f"ShiftDimuonVertex{name}", "normalizedChi2"           , 5000,     -1,    50, "dimuon"),
        (f"ShiftDimuonVertex{name}", "dca"                      , 1000,      0,  1500, "dimuon"),
        (f"ShiftDimuonVertex{name}", "dcaValid"                 ,   20,    -10,    10, "dimuon"),
        (f"ShiftDimuonVertex{name}", "isOS"                     ,   20,    -10,    10, "dimuon"),
        (f"ShiftDimuonVertex{name}", "genIsOS"                  ,   20,    -10,    10, "dimuon"),
        (f"ShiftDimuonVertex{name}", "sameGenMuon"              ,   20,    -10,    10, "dimuon"),
    )
    # fmt: on

# ============================================================
# Custom 1D histograms
# ============================================================

# fmt: off
histParams = (
    ("DetectorDiagnostics", "dtAttachmentFraction"     ,  51,      0,  1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "trackerAttachmentFraction",  51,      0,  1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "dtTruthChamberPurity"     ,  51,      0,  1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "timingMeasurements"       ,  20,      0,    20, "detector_diagnostics"),
    ("DetectorDiagnostics", "timingDeltaChi2"          , 100,      0,   100, "detector_diagnostics"),
    ("DetectorDiagnostics", "combinedTimingDeltaChi2"  , 100,      0,   100, "detector_diagnostics"),
    ("DetectorDiagnostics", "hbheValidTimeFraction"    ,  51,      0,  1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "hoValidTimeFraction"      ,  51,      0,  1.02, "detector_diagnostics"),
    (f"GenDimuon"         , "pt"                       , 200,      0,    20, "dimuon"              ),
    (f"GenDimuon"         , "pz"                       , 200,  -1000,   100, "dimuon"              ),
    (f"GenDimuon"         , "eta"                      , 100,    -10,    10, "dimuon"              ),
    (f"GenDimuon"         , "phi"                      , 100,   -3.2,   3.2, "dimuon"              ),
    (f"GenDimuon"         , "mass"                     , 100,      0,    10, "dimuon"              ),
    (f"GenDimuon"         , "vx"                       , 100,   -300,   300, "dimuon"              ),
    (f"GenDimuon"         , "vy"                       , 100,   -300,   300, "dimuon"              ),
    (f"GenDimuon"         , "vz"                       , 100, -20000, 20000, "dimuon"              ),
    ("ShiftDimuonVertex"  , "topologyCategory"         ,   5,   -0.5,   4.5, "dimuon"              ),
)
# fmt: on

# Reconstructed pair observables use the indexed ShiftMuon states, without
# truth matching or additional cuts. Angular spectra retain the full physical
# range; fine source bins are merged only when rendering sparsely populated tails.
for category in dimuonCategories:
    for variable, bins, low, high in (
        ("openingAngle", 630, 0.0, 3.15),
        ("openingAngleFine", 400, 0.0, 0.2),
        ("constrainedOpeningAngle", 630, 0.0, 3.15),
        ("deltaPhi", 315, 0.0, 3.15),
        ("deltaEta", 400, 0.0, 20.0),
        ("deltaR", 400, 0.0, 20.0),
        ("ptAsymmetry", 101, 0.0, 1.01),
        ("momentumAsymmetry", 101, 0.0, 1.01),
        ("minMuonPt", 500, 0.0, 50.0),
        ("maxMuonPt", 500, 0.0, 50.0),
        ("rapidity", 400, -10.0, 10.0),
        ("vertexR", 450, 0.0, 450.0),
        ("massOS", 5000, 0.0, 100.0),
        ("massSS", 5000, 0.0, 100.0),
        ("constrainedMass", 1000, 0.0, 100.0),
        ("refittedMass", 1000, 0.0, 100.0),
        ("probability", 101, 0.0, 1.01),
        ("originCompatibilityNormalizedChi2", 500, 0.0, 50.0),
        ("constrainedValid", 2, -0.5, 1.5),
        ("refitStatus", 34, -32.5, 1.5),
    ):
        histParams += ((f"ShiftDimuonVertex{category}", variable, bins, low, high, "dimuon"),)

# MC-only origin diagnostic. Each dimuon contributes exactly one categorical
# entry: its common immediate non-muon mother species, "different mothers",
# or "unmatched". ShiftHistogramsFiller assigns human-readable bin labels.
for name in dimuonCategories:
    histParams += ((f"ShiftDimuonVertex{name}", "genPid", 13, -0.5, 12.5, "dimuon"),)
    # When the two reconstructed muons have different immediate mothers, the
    # filler records one entry per mother in the reduced 11-species scheme.
    histParams += (
        (
            f"ShiftDimuonVertex{name}",
            "genPidDifferentMothers",
            11,
            -0.5,
            10.5,
            "dimuon",
        ),
    )


def resolution_bin_edges(core_half_width, core_bins, tail_limit=1.0e9):
    """Fine linear core plus symmetric logarithmic tails, without axis growth."""
    core_step = 2.0 * core_half_width / core_bins
    core = [-core_half_width + index * core_step for index in range(core_bins + 1)]
    positive_tail = []
    edge = 2.0 * core_half_width
    while edge < tail_limit:
        positive_tail.append(edge)
        edge *= 2.0
    positive_tail.append(tail_limit)
    negative_tail = [-edge for edge in reversed(positive_tail)]
    return negative_tail + core + positive_tail


relativeResolutionBinEdges = resolution_bin_edges(1.0, 100)
qOverPtResolutionBinEdges = resolution_bin_edges(2.0, 200)
vertexResolutionBinEdges = resolution_bin_edges(5000.0, 200)
irregularHistParams = ()

# Retain the original fine peak resolution, then widen before the sparse
# tails begin to fluctuate. These remain companions to the fixed-bin spectra.
muonPtVariableBinEdges = (
    tuple(index / 10.0 for index in range(61))
    + tuple(6.5 + 0.5 * index for index in range(8))
    + tuple(11.0 + index for index in range(10))
    + tuple(22.0 + 2.0 * index for index in range(15))
)
muonPzVariableBinEdges = (
    linear_bin_edges(
        (-2000.0, -800.0, 100.0), (-800.0, -400.0, 40.0),
        (-400.0, -200.0, 10.0), (-200.0, -100.0, 5.0), (-100.0, 100.0, 2.0)
    )
)
muonPVariableBinEdges = linear_bin_edges(
    (0.0, 120.0, 2.0), (120.0, 200.0, 5.0), (200.0, 400.0, 10.0),
    (400.0, 800.0, 40.0), (800.0, 2000.0, 100.0)
)

# The dimuon spectrum becomes sparse earlier than the single-muon spectrum.
dimuonPtVariableBinEdges = (
    tuple(index / 10.0 for index in range(41))
    + tuple(4.25 + 0.25 * index for index in range(8))
    + tuple(6.5 + 0.5 * index for index in range(8))
    + tuple(11.0 + index for index in range(10))
    + tuple(22.0 + 2.0 * index for index in range(40))
    + tuple(105.0 + 5.0 * index for index in range(20))
)
dimuonPzVariableBinEdges = linear_bin_edges(
    (-1000.0, -400.0, 50.0), (-400.0, -200.0, 20.0),
    (-200.0, -100.0, 10.0), (-100.0, 100.0, 2.5)
)
dimuonPVariableBinEdges = linear_bin_edges(
    (0.0, 120.0, 2.5), (120.0, 200.0, 10.0), (200.0, 400.0, 20.0),
    (400.0, 1000.0, 50.0)
)

# These are filled explicitly from reconstructed pairs, so they can retain
# fine core bins and extend their axes without allocating fine empty tails.
dimuon_source_variable_edges = {
    "openingAngle": dimuon_opening_angle_edges,
    "constrainedOpeningAngle": dimuon_opening_angle_edges,
    "openingAngleFine": dimuon_fine_opening_angle_edges,
    "originCompatibilityNormalizedChi2": dimuon_origin_chi2_edges,
}
histParams = tuple(
    row for row in histParams
    if not (row[0].startswith("ShiftDimuonVertex") and row[1] in dimuon_source_variable_edges)
)
for category in dimuonCategories:
    irregularHistParams += tuple(
        (f"ShiftDimuonVertex{category}", variable, edges, "dimuon")
        for variable, edges in dimuon_source_variable_edges.items()
    )

irregularDefaultHistParams = ()
for name in ["GenMuon", "ShiftMuon"] + [
    f"ShiftMuon{category}" for category in muonCategories
]:
    irregularDefaultHistParams += (
        (name, "pt_variable", muonPtVariableBinEdges, "muon"),
        (name, "pz_variable", muonPzVariableBinEdges, "muon"),
        (name, "p_variable", muonPVariableBinEdges, "muon"),
    )
for name in [f"ShiftDimuonVertex{category}" for category in dimuonCategories]:
    irregularDefaultHistParams += (
        (name, "pt_variable", dimuonPtVariableBinEdges, "dimuon"),
        (name, "pz_variable", dimuonPzVariableBinEdges, "dimuon"),
        (name, "p_variable", dimuonPVariableBinEdges, "dimuon"),
    )


def add_resolution_histograms(prefix, variables, bin_edges):
    global irregularHistParams
    irregularHistParams += tuple(
        (prefix, variable, bin_edges, "resolution") for variable in variables
    )


for name in dimuonCategories:
    prefix = f"DimuonResolution{name}"
    add_resolution_histograms(
        prefix,
        (
            "pt",
            "pz",
            "eta",
            "phi",
            "minv",
            "vz",
            "constrainedPt",
            "constrainedPz",
            "constrainedEta",
            "constrainedPhi",
            "constrainedMinv",
            "constrainedVz",
        ),
        relativeResolutionBinEdges,
    )
    add_resolution_histograms(
        prefix, ("vx", "vy", "constrainedVx", "constrainedVy"), vertexResolutionBinEdges
    )
    # fmt: off
    histParams += (
        (f"DimuonResolution{name}", "deltaEta"           , 2400,    -12,    12, "resolution"),
        (f"DimuonResolution{name}", "deltaPhi"           , 2000, -3.142, 3.142, "resolution"),
        (f"DimuonResolution{name}", "constrainedDeltaEta", 2400,    -12,    12, "resolution"),
        (f"DimuonResolution{name}", "constrainedDeltaPhi", 2000, -3.142, 3.142, "resolution"),
    )
    # fmt: on

for name in muonCategories:
    prefix = f"MuonResolution{name}"
    add_resolution_histograms(
        prefix,
        (
            "pt",
            "pz",
            "eta",
            "phi",
            "vz",
            "constrainedPt",
            "constrainedPz",
            "constrainedEta",
            "constrainedPhi",
            "constrainedVz",
        ),
        relativeResolutionBinEdges,
    )
    add_resolution_histograms(
        prefix, ("qOverPt", "constrainedQOverPt"), qOverPtResolutionBinEdges
    )
    add_resolution_histograms(
        prefix, ("vx", "vy", "constrainedVx", "constrainedVy"), vertexResolutionBinEdges
    )
    # fmt: off
    histParams += (
        (f"MuonResolution{name}", "deltaEta"           , 2400,    -12,    12, "resolution"),
        (f"MuonResolution{name}", "deltaPhi"           , 2000, -3.142, 3.142, "resolution"),
        (f"MuonResolution{name}", "constrainedDeltaEta", 2400,    -12,    12, "resolution"),
        (f"MuonResolution{name}", "constrainedDeltaPhi", 2000, -3.142, 3.142, "resolution"),
    )
    # fmt: on

add_resolution_histograms(
    "MuonResolutionSingleEndcap", ("qOverPt",), qOverPtResolutionBinEdges
)

# ============================================================
# Generator-coordinate reconstruction efficiencies
# ============================================================


def frange(start, stop, step):
    values = []
    current = start
    while current <= stop:
        values.append(round(current, 10))
        current += step
    return tuple(values)


muonEfficiencyBinning = {
    "pt": (
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.3,
        2.6,
        3.0,
        3.5,
        4.5,
        6.0,
        8.0,
    ),
    "pz": frange(-300, 50, 10),
    "eta": frange(-5, -3, 0.10),
    "phi": frange(-3.2, 3.2, 0.4),
    "vz": (
        14500,
        14600,
        14680,
        14720,
        14750,
        14775,
        14800,
        14825,
        14850,
        14880,
        14920,
        15000,
        15100,
    ),
}

dimuonEfficiencyBinning = {
    "pt": (0, 1.0, 2.0, 3.0, 6.0, 10.0),
    "pz": (-400, -300, -150, -125, -100, -50, -20, 20, 40),
    "eta": frange(-8, -3, 0.5),
    "phi": frange(-3.2, 3.2, 0.8),
    "vz": (14500, 14600, 14700, 14750, 14800, 14850, 14900, 15000, 15100),
    # Lab-frame angle between the generated daughter momenta, in radians.
    # Resolve boosted pairs finely while retaining the full physical range.
    "openingAngle": (
        0.0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
        0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.15,
    ),
}

muonEfficiencyPrefixes = ["ShiftMuonEfficiency"] + [
    f"ShiftMuon{category}Efficiency" for category in muonCategories
]

dimuonEfficiencyPrefixes = [
    f"ShiftDimuonVertex{category}Efficiency" for category in dimuonCategories
]

for prefixes, binning in (
    (muonEfficiencyPrefixes, muonEfficiencyBinning),
    (dimuonEfficiencyPrefixes, dimuonEfficiencyBinning),
):
    for prefix in prefixes:
        for variable, binEdges in binning.items():
            irregularHistParams += (
                (prefix, f"{variable}_total", binEdges, "efficiency"),
                (prefix, f"{variable}_pass", binEdges, "efficiency"),
            )

# ============================================================
# 2D histograms
# ============================================================

histParams2D = ()
for category in dimuonCategories:
    histParams2D += (
        (f"ShiftDimuonVertex{category}_massVsOpeningAngle", 100, 0.0, 0.2, 100, 0.0, 10.0, "dimuon"),
        (f"ShiftDimuonVertex{category}_massVsPtAsymmetry", 100, 0.0, 1.01, 100, 0.0, 10.0, "dimuon"),
    )

for name in ["RecoVsGenMuon", "RecoVsGenDimuon"]:
    # fmt: off
    histParams2D += (
        (f"{name}_pt"  , 1000,    0,    60, 1000,    0,    60, "correlations"),
        (f"{name}_pz"  ,  500, -500,   100,  500, -500,   100, "correlations"),
        (f"{name}_eta" , 1000,  -10,    10, 1000,  -10,    10, "correlations"),
        (f"{name}_phi" ,  100, -3.2,   3.2,  100, -3.2,   3.2, "correlations"),
        (f"{name}_minv",  100,    0,    10,  100,    0,    10, "correlations"),
        (f"{name}_vx"  ,  200, -300,   300,  200,  -10,    10, "correlations"),
        (f"{name}_vy"  ,  200, -300,   300,  200,  -10,    10, "correlations"),
        (f"{name}_vz"  ,  200,    0, 20000, 2000,    0, 20000, "correlations"),
    )
    # fmt: on

# ============================================================
# Other stuff
# ============================================================


def latest_versioned_sample():
    samples_dir = resolve_storage_path(f"{base_path}/{sample}/{campaign}/samples/step4_merged")
    input_path, provenance_tag = latest_merged_sample(samples_dir)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    version_pattern = re.compile(r"^v([0-9]+)_([^/_]+)(?:_([^/]+))?$")
    latest_version = 0
    existing_version_for_hash = None
    for output_path in glob.glob(f"{project_dir}/plots/v*_*"):
        if not os.path.isdir(output_path):
            continue
        dir_name = os.path.basename(output_path)
        match = version_pattern.fullmatch(dir_name)
        if not match:
            continue
        version_number = int(match.group(1))
        latest_version = max(latest_version, version_number)
        output_hash = match.group(2)
        output_variant = match.group(3) or ""
        if output_hash == provenance_tag and not output_variant:
            if existing_version_for_hash is None:
                existing_version_for_hash = version_number
            else:
                existing_version_for_hash = max(
                    existing_version_for_hash, version_number
                )

    sample_version = (
        existing_version_for_hash
        if existing_version_for_hash is not None
        else latest_version + 1
    )
    return input_path, sample_version, provenance_tag


nEvents = -1

if globals().get("recoDataMode", False):
    # Input/output are supplied by the command line; do not scan an MC campaign.
    input_path, sample_version, provenance_tag = "", 0, "data"
else:
    # input_path, sample_version, provenance_tag = latest_versioned_sample()
    # The submitter supplies one exact Nano file per job for all process/bins.
    input_path, sample_version, provenance_tag = "", 0, campaign
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

inputFilePath = input_path
histogramsOutputFilePath = (
    f"{project_dir}/plots/v{sample_version}_{provenance_tag}/histograms.root"
)

info(f"Selected sample v{sample_version}_{provenance_tag}: {inputFilePath}")
info(f"Histogram output: {histogramsOutputFilePath}")

# weightsBranchName = "genWeight"
weightsBranchName = "shiftSamplingGenWeight"
# Use W/p once; retain the full parent GEN exposure/trial denominator from
# validation/ntuple_production_20261005_v8/sampling_plan.json, not Nano Runs.
eventsTreeNames = [
    "Events",
]
specialBranchSizes = {
    "Particle": "Event_numberP",
}

# Independent truth coverage and angular-tail diagnostics.
for variable in ("hitMatched", "legacyMatched", "dimuonHitMatched"):
    histParams += (("TruthDiagnostics", variable, 2, -0.5, 1.5, "truth_diagnostics"),)
for category in muonCategories:
    for variable, bins, lo, hi in (
        ("directionReversed", 2, -0.5, 1.5),
        ("chargeMisidentified", 2, -0.5, 1.5),
        ("axisDeltaEta", 400, -2.0, 2.0),
        ("axisDeltaPhi", 400, -3.142, 3.142),
        ("constrainedEtaPull", 400, -20.0, 20.0),
        ("constrainedPhiPull", 400, -20.0, 20.0),
    ):
        histParams += (
            (f"MuonResolution{category}", variable, bins, lo, hi, "truth_diagnostics"),
        )
for variable, lo, hi in (
    ("constrainedEtaErr", 0.0, 1.0),
    ("constrainedPhiErr", 0.0, 1.0),
    ("targetPredictedEtaErr", 0.0, 1.0),
    ("targetPredictedPhiErr", 0.0, 1.0),
    ("targetResidualX", -1000.0, 1000.0),
    ("targetResidualY", -1000.0, 1000.0),
    ("targetPullX", -20.0, 20.0),
    ("targetPullY", -20.0, 20.0),
):
    histParams += (("TargetDiagnostics", variable, 200, lo, hi, "target_diagnostics"),)
histParams += (
    ("TargetDiagnostics", "constrainedStatus", 16, -11.5, 4.5, "target_diagnostics"),
)
histParams += (
    ("TargetDiagnostics", "targetForwardStatus", 12, -10.5, 1.5, "target_diagnostics"),
)
histParams += (
    (
        "TargetDiagnostics",
        "targetForwardIterations",
        66,
        -1.5,
        64.5,
        "target_diagnostics",
    ),
)
for variable, lo, hi in (
    ("massErr", 0.0, 10.0),
    ("massRelativeErr", 0.0, 5.0),
    ("minQoverPSignificance", 0.0, 20.0),
):
    histParams += (
        ("VertexRefitDiagnostics", variable, 200, lo, hi, "vertex_refit_diagnostics"),
    )
for category in dimuonCategories:
    histParams += (
        (
            f"DimuonResolution{category}",
            "refittedMassPull",
            400,
            -20.0,
            20.0,
            "truth_diagnostics",
        ),
    )
    histParams2D += (
        (
            f"DimuonResolution{category}_refittedMassPullVsCurvature",
            40,
            0.0,
            20.0,
            200,
            -20.0,
            20.0,
            "truth_diagnostics",
        ),
        (
            f"DimuonResolution{category}_refittedMassScaleVsCurvature",
            40,
            0.0,
            20.0,
            100,
            0.0,
            5.0,
            "truth_diagnostics",
        ),
    )
histParams2D += (
    (
        "VertexRefitDiagnostics_massVsCurvature",
        40,
        0.0,
        20.0,
        100,
        0.0,
        10.0,
        "vertex_refit_diagnostics",
    ),
)

# Physical target-constrained values are valid only after a successful target
# fit. The explicit filler checks constrainedValid; default branch filling
# would otherwise turn uninitialized zeros into apparent physical peaks.
def is_constrained_muon_physical_histogram(row):
    return (
        row[0].startswith("ShiftMuon")
        and row[1].startswith("constrained")
        and row[1] not in ("constrainedValid", "constrainedStatus")
    )


histParams += tuple(
    row for row in defaultHistParams if is_constrained_muon_physical_histogram(row)
)
defaultHistParams = tuple(
    row for row in defaultHistParams if not is_constrained_muon_physical_histogram(row)
)
irregularHistParams += tuple(
    row for row in irregularDefaultHistParams if is_constrained_muon_physical_histogram(row)
)
irregularDefaultHistParams = tuple(
    row for row in irregularDefaultHistParams if not is_constrained_muon_physical_histogram(row)
)

# Custom fills retain native Double scores and signed event weights once.
BDT_CATEGORIES = tuple(dimuonCategories)
BDT_LABEL_SUFFIXES = ("", "Genuine", "Accidental", "Unknown")
BDT_SCORE_BINS = (100, 0.0, 1.0)
BDT_MAP_AXES = {
    "Mass": (100, 0.0, 100.0, "m_{#mu#mu}^{reco} [GeV]"),
    "Vz": (100, -20000.0, 20000.0, "v_{z}^{reco} [cm]"),
    "VertexR": (100, 0.0, 300.0, "r_{vertex}^{reco} [cm]"),
}


def bdt_histogram_name(category="", label="", unweighted=False):
    return f"ShiftDimuonVertex{category}_bdtScore{label}" + ("Unweighted" if unweighted else "")


def bdt_histogram_path(category="", label="", unweighted=False):
    return "dimuon_bdt/" + bdt_histogram_name(category, label, unweighted)


if enableBdtDiagnostics:
    for category in BDT_CATEGORIES:
        for label in BDT_LABEL_SUFFIXES:
            for unweighted in (False, True):
                histParams += ((f"ShiftDimuonVertex{category}",
                                "bdtScore" + label + ("Unweighted" if unweighted else ""),
                                *BDT_SCORE_BINS, "dimuon_bdt"),)
        for axis, (bins, low, high, title) in BDT_MAP_AXES.items():
            histParams2D += ((bdt_histogram_name(category) + "Vs" + axis,
                              bins, low, high, *BDT_SCORE_BINS, "dimuon_bdt"),)

if enableResolutionMaps:
    irregularHistParams2D = book_resolution_maps()
