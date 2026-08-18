# ============================================================
# Default single-muon histograms
# ============================================================

from Logger import info
from shift_paths import base_path, campaign, sample
from shift_extra_collections import extraEventCollections
import re
import glob
import os

defaultHistParams = (
    ("ShiftMuon",  "topology", 5, -.5, 4.5, "muon"),
    ("ShiftMuon",  "recoAlgorithm", 3, -.5, 2.5, "muon"),
    ("ShiftDimuonVertex", "topologyMin", 5, -.5, 4.5, "dimuon"),
    ("ShiftDimuonVertex", "topologyMax", 5, -.5, 4.5, "dimuon"),
)

defaultHistParams += (
    ("ShiftMuon", "simDTHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "simPixelHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "simStripHits", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "simGEMHits", 20, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchValid", 2, -.5, 1.5, "detector_diagnostics"),
    ("ShiftMuon", "trackerValidHits", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "combinedTrackValid", 2, -.5, 1.5, "detector_diagnostics"),
    ("ShiftMuon", "combinedTrackerHits", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "combinedTargetPt", 200, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "combinedTargetPz", 200, -1000, 100, "detector_diagnostics"),
    ("ShiftMuon", "combinedTargetDca", 100, -1, 500, "detector_diagnostics"),
    ("ShiftMuon", "simHcalHits", 50, -2, 48, "detector_diagnostics"),
    ("ShiftMuon", "simHBHEHits", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "simHFHits", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "simHOHits", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "simHcalEnergy", 100, -2, 2, "detector_diagnostics"),
    ("ShiftMuon", "simZDCHits", 50, -2, 48, "detector_diagnostics"),
    ("ShiftMuon", "simZDCEnergy", 100, -2, 2, "detector_diagnostics"),
    ("ShiftMuon", "simZDCFirstTime", 120, -600, 0, "detector_diagnostics"),
    ("ShiftMuon", "nCompatibleDTSegments", 20, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "nPropagatedDTSegments", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "nPropagatedPixelHits", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "nPropagatedStripHits", 200, 0, 200, "detector_diagnostics"),
    ("ShiftMuon", "nCompatiblePixelHits", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "nCompatibleStripHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "nAddedDTRefitHits", 20, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "nAddedTrackerRefitHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "nAddedDTTruthChamberMatches", 20, -1, 19, "detector_diagnostics"),
    ("ShiftMuon", "minDTResidual", 200, -1, 999, "detector_diagnostics"),
    ("ShiftMuon", "minDTEstimatorChi2", 200, -1, 999, "detector_diagnostics"),
    ("ShiftMuon", "minTrackerResidual", 200, -1, 199, "detector_diagnostics"),
    ("ShiftMuon", "minTrackerEstimatorChi2", 200, -1, 999, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchRawValid", 2, -.5, 1.5, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchValid", 2, -.5, 1.5, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchLineDistance", 200, -1, 999, "detector_diagnostics"),
    ("ShiftMuon", "trackerMatchAxisAngle", 160, -0.01, 1.59, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedEcalRecHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedHBHERecHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedHFRecHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedHORecHits", 50, 0, 50, "detector_diagnostics"),
    ("ShiftMuon", "nMatchedZDCRecHits", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "matchedEcalEnergy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "matchedHBHEEnergy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "matchedHFEnergy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "matchedHOEnergy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHBHEIds", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHBHERecHits", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "nValidCrossedHBHETimes", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "crossedHBHEEnergy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "hbhe3x3Energy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHBHEEnergy", 100, 0, 10, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHBHETime", 160, -1000, 200, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHOIds", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "nCrossedHORecHits", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "nValidCrossedHOTimes", 30, 0, 30, "detector_diagnostics"),
    ("ShiftMuon", "hcalAssociationDirection", 4, -0.5, 3.5, "detector_diagnostics"),
    ("ShiftMuon", "crossedHOEnergy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "ho3x3Energy", 100, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHOEnergy", 100, 0, 10, "detector_diagnostics"),
    ("ShiftMuon", "maxCrossedHOTime", 160, -1000, 200, "detector_diagnostics"),
    ("ShiftMuon", "matchedZDCEnergy", 100, 0, 10, "detector_diagnostics"),
    ("ShiftMuon", "nCaloTimingMeasurements", 10, 0, 10, "detector_diagnostics"),
    ("ShiftMuon", "caloTimingDirectionSign", 3, -1.5, 1.5, "detector_diagnostics"),
    ("ShiftMuon", "caloTimingDeltaChi2", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "combinedTimingDirectionSign", 3, -1.5, 1.5, "detector_diagnostics"),
    ("ShiftMuon", "nCombinedTimingMeasurements", 20, 0, 20, "detector_diagnostics"),
    ("ShiftMuon", "combinedTimingDeltaChi2", 100, 0, 100, "detector_diagnostics"),
    ("ShiftMuon", "combinedTimingAgreesWithMuon", 3, -1.5, 1.5, "detector_diagnostics"),

    ("Event", "ShiftRecoDiag_enableDTMeasurement", 2, -.5, 1.5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_dtNavigationMode", 3, -.5, 2.5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_enableGEMMeasurement", 2, -.5, 1.5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_trackerMode", 3, -.5, 2.5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_enableHcalDiagnostics", 2, -.5, 1.5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_enableZDCDiagnostics", 2, -.5, 1.5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nDTSimHits", 100, 0, 100, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nDTSegments", 50, 0, 50, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nGEMSimHits", 20, 0, 20, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nGEMSegments", 20, 0, 20, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nGeneralTracks", 50, 0, 50, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTrackerSeeds", 30, 0, 30, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTrackerTrackCandidates", 30, 0, 30, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTrackerRawTracks", 30, 0, 30, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTrackerSelectedTracks", 30, 0, 30, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTrackerLHCTrackCandidates", 30, 0, 30, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTrackerLHCTracks", 30, 0, 30, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nSignalMuonSimTracks", 10, 0, 10, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nSignalMuonWithTrackerSimHits", 10, 0, 10, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nSignalMuonWithMuonSystemSimHits", 10, 0, 10, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nSignalMuonWithTrackerAndMuonSystemSimHits", 10, 0, 10, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nDSATrackerMatches", 10, 0, 10, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nTraversingTrackerMatches", 10, 0, 10, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nSignalMuonHcalSimHits", 50, -2, 48, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_signalMuonHcalSimEnergy", 100, -2, 2, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nSignalMuonZDCSimHits", 50, -2, 48, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_signalMuonZDCSimEnergy", 100, -2, 2, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_signalMuonZDCFirstTime", 120, -600, 0, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nPixelRecHits", 200, -1, 199, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nStripMatchedRecHits", 500, -1, 499, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nStripRphiRecHits", 3000, -1, 2999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nStripRphiUnmatchedRecHits", 2000, -1, 1999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nStripStereoRecHits", 1500, -1, 1499, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nStripStereoUnmatchedRecHits", 1000, -1, 999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHBHEQIE11Digis", 200, 0, 20000, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHFQIE10Digis", 200, 0, 5000, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHODigis", 200, 0, 3000, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nZDCDigis", 30, -1, 29, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nEcalBarrelRecHits", 200, -1, 199, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nEcalEndcapRecHits", 200, -1, 199, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHBHERecHits", 200, -1, 3999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHFRecHits", 200, -1, 1999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHORecHits", 200, -1, 2999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nZDCRecHits", 50, -1, 49, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nZDCRecHitsAboveThreshold", 30, -1, 29, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_ZDCPositiveRecEnergy", 100, -1, 20, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_ZDCMaxRecEnergy", 100, -1, 5, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_ZDCMaxRecTime", 160, -600, 200, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHBHERecHitsAboveThreshold", 200, -1, 3999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_HBHEPositiveRecEnergy", 200, -1, 200, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHFRecHitsAboveThreshold", 200, -1, 1999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_HFPositiveRecEnergy", 200, -1, 200, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nEcalBarrelRecHitsAboveThreshold", 200, -1, 399, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_EcalBarrelPositiveRecEnergy", 200, -1, 100, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nEcalEndcapRecHitsAboveThreshold", 200, -1, 399, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_EcalEndcapPositiveRecEnergy", 200, -1, 100, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nHORecHitsAboveThreshold", 200, -1, 2999, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_HOPositiveRecEnergy", 200, -1, 200, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nBCM1FSimHits", 50, -1, 49, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nBHMSimHits", 50, -1, 49, "detector_diagnostics"),
    ("Event", "ShiftRecoDiag_nPLTSimHits", 50, -1, 49, "detector_diagnostics"),

)

muonCategories = ["NearEndcapOnly", "NearEndcapAndBarrel", "BothEndcaps", "FarEndcapOnly", "Unclassified"]

for name in ["GenMuon", "ShiftMuon"] + [f"ShiftMuon{category}" for category in muonCategories]:
  defaultHistParams += (
      ("Event", f"n{name}", 10, 0, 10, "event"),
      (f"{name}", "pt", 200, 0, 20, "muon"),
      (f"{name}", "pz", 200, -1000, 100, "muon"),
      (f"{name}", "eta", 100, -10, 10, "muon"),
      (f"{name}", "phi", 100, -3.2, 3.2, "muon"),
      (f"{name}", "vx", 100, -300, 300, "muon"),
      (f"{name}", "vy", 100, -300, 300, "muon"),
      (f"{name}", "vz", 100, -20000, 20000, "muon"),
  )

  if "Gen" in name:
    continue

  defaultHistParams += (
      (f"{name}", "constrainedPt", 200, 0, 20, "muon"),
      (f"{name}", "constrainedPz", 200, -1000, 100, "muon"),
      (f"{name}", "constrainedEta", 100, -10, 10, "muon"),
      (f"{name}", "constrainedPhi", 100, -3.2, 3.2, "muon"),
      (f"{name}", "constrainedVx", 100, -300, 300, "muon"),
      (f"{name}", "constrainedVy", 100, -300, 300, "muon"),
      (f"{name}", "constrainedVz", 100, -20000, 20000, "muon"),

      (f"{name}", "nCSCHits", 20, 0, 20, "muon"),
      (f"{name}", "nDTHits", 20, 0, 20, "muon"),
      (f"{name}", "nRPCHits", 20, 0, 20, "muon"),
      (f"{name}", "nGEMHits", 20, 0, 20, "muon"),
      (f"{name}", "nME0Hits", 20, 0, 20, "muon"),

      (f"{name}", "dz", 100, -20000, 20000, "muon"),
      (f"{name}", "linePcaZ", 100, -20000, 20000, "muon"),
      (f"{name}", "trackVz", 100, -20000, 20000, "muon"),
      (f"{name}", "genPartIdx", 30, -10, 20, "muon"),
  )

# ============================================================
# Default dimuon histograms
# ============================================================

dimuonCategories = ["", "Good", "Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other"]

for name in dimuonCategories:
  defaultHistParams += (
      ("Event", f"nShiftDimuonVertex{name}", 10, 0, 10, "event"),
      (f"ShiftDimuonVertex{name}", "pt", 200, 0, 20, "dimuon"),
      (f"ShiftDimuonVertex{name}", "pz", 200, -1000, 100, "dimuon"),
      (f"ShiftDimuonVertex{name}", "eta", 100, -10, 10, "dimuon"),
      (f"ShiftDimuonVertex{name}", "phi", 100, -3.2, 3.2, "dimuon"),
      (f"ShiftDimuonVertex{name}", "mass", 100, 0, 10, "dimuon"),
      (f"ShiftDimuonVertex{name}", "vx", 100, -300, 300, "dimuon"),
      (f"ShiftDimuonVertex{name}", "vy", 100, -300, 300, "dimuon"),
      (f"ShiftDimuonVertex{name}", "vz", 100, -20000, 20000, "dimuon"),
      (f"ShiftDimuonVertex{name}", "chi2", 100, -1, 100, "dimuon"),
      (f"ShiftDimuonVertex{name}", "normalizedChi2", 100, -1, 100, "dimuon"),
      (f"ShiftDimuonVertex{name}", "dca", 100, 0, 1500, "dimuon"),
      (f"ShiftDimuonVertex{name}", "dcaValid", 20, -10, 10, "dimuon"),

      (f"ShiftDimuonVertex{name}", "isOS", 20, -10, 10, "dimuon"),
      (f"ShiftDimuonVertex{name}", "genIsOS", 20, -10, 10, "dimuon"),
      (f"ShiftDimuonVertex{name}", "sameGenMuon", 20, -10, 10, "dimuon"),
  )

# ============================================================
# Custom 1D histograms
# ============================================================

histParams = (
    ("DetectorDiagnostics", "dtAttachmentFraction", 51, 0, 1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "trackerAttachmentFraction", 51, 0, 1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "dtTruthChamberPurity", 51, 0, 1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "timingMeasurements", 20, 0, 20, "detector_diagnostics"),
    ("DetectorDiagnostics", "timingDeltaChi2", 100, 0, 100, "detector_diagnostics"),
    ("DetectorDiagnostics", "combinedTimingDeltaChi2", 100, 0, 100, "detector_diagnostics"),
    ("DetectorDiagnostics", "hbheValidTimeFraction", 51, 0, 1.02, "detector_diagnostics"),
    ("DetectorDiagnostics", "hoValidTimeFraction", 51, 0, 1.02, "detector_diagnostics"),
    (f"GenDimuon", "pt", 200, 0, 20, "dimuon"),
    (f"GenDimuon", "pz", 200, -1000, 100, "dimuon"),
    (f"GenDimuon", "eta", 100, -10, 10, "dimuon"),
    (f"GenDimuon", "phi", 100, -3.2, 3.2, "dimuon"),
    (f"GenDimuon", "mass", 100, 0, 10, "dimuon"),
    (f"GenDimuon", "vx", 100, -300, 300, "dimuon"),
    (f"GenDimuon", "vy", 100, -300, 300, "dimuon"),
    (f"GenDimuon", "vz", 100, -20000, 20000, "dimuon"),

    ("ShiftDimuonVertex", "topologyCategory", 5, -.5, 4.5, "dimuon"),
)

for name in dimuonCategories:
  histParams += (
      (f"DimuonResolution{name}", "pt", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "pz", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "eta", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "phi", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "minv", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "vx", 100, -5000, 5000, "resolution"),
      (f"DimuonResolution{name}", "vy", 100, -5000, 5000, "resolution"),
      (f"DimuonResolution{name}", "vz", 50, -1, 1, "resolution"),

      (f"DimuonResolution{name}", "constrainedPt", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "constrainedPz", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "constrainedEta", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "constrainedPhi", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "constrainedMinv", 50, -1, 1, "resolution"),
      (f"DimuonResolution{name}", "constrainedVx", 100, -5000, 5000, "resolution"),
      (f"DimuonResolution{name}", "constrainedVy", 100, -5000, 5000, "resolution"),
      (f"DimuonResolution{name}", "constrainedVz", 50, -1, 1, "resolution"),
  )

for name in muonCategories:
  histParams += (
      # Signed-curvature residual used in CMS-DP-2015-015.  With 200 bins over
      # [-2, 2], every bin has the same 0.02 width as the published reference.
      (f"MuonResolution{name}", "qOverPt", 200, -2, 2,     "resolution"),
      (f"MuonResolution{name}", "pt", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "pz", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "eta", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "phi", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "vx", 100, -5000, 5000,     "resolution"),
      (f"MuonResolution{name}", "vy", 100, -5000, 5000,     "resolution"),
      (f"MuonResolution{name}", "vz", 50, -1, 1,     "resolution"),

      (f"MuonResolution{name}", "constrainedPt", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "constrainedQOverPt", 200, -2, 2,     "resolution"),
      (f"MuonResolution{name}", "constrainedPz", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "constrainedEta", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "constrainedPhi", 50, -1, 1,     "resolution"),
      (f"MuonResolution{name}", "constrainedVx", 100, -5000, 5000,     "resolution"),
      (f"MuonResolution{name}", "constrainedVy", 100, -5000, 5000,     "resolution"),
      (f"MuonResolution{name}", "constrainedVz", 50, -1, 1,     "resolution"),
  )

histParams += (
    ("MuonResolutionSingleEndcap", "qOverPt", 200, -2, 2, "resolution"),
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
    "pt":  (0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.3, 2.6, 3., 3.5, 4.5, 6., 8.),
    "pz":  frange(-300, 50, 10),
    "eta": frange(-5, -3, 0.10),
    "phi": frange(-3.2, 3.2, 0.4),
    "vz":  (14500, 14600, 14680, 14720, 14750, 14775, 14800, 14825, 14850, 14880, 14920, 15000, 15100),
}

dimuonEfficiencyBinning = {
    "pt":  (0, 1., 2., 3., 6., 10.),
    "pz":  (-400, -300, -150, -125, -100, -50, -20, 20, 40),
    "eta": frange(-8, -3, 0.5),
    "phi": frange(-3.2, 3.2, 0.8),
    "vz":  (14500, 14600, 14700, 14750, 14800, 14850, 14900, 15000, 15100),
}

muonEfficiencyPrefixes = ["ShiftMuonEfficiency"] + [f"ShiftMuon{category}Efficiency" for category in muonCategories]

dimuonEfficiencyPrefixes = [f"ShiftDimuonVertex{category}Efficiency" for category in dimuonCategories]

irregularHistParams = ()
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

for name in ["RecoVsGenMuon", "RecoVsGenDimuon"]:

  histParams2D += (
      (f"{name}_pt", 1000, 0, 60, 1000, 0, 60, "correlations"),
      (f"{name}_pz", 500, -500, 100, 500, -500, 100, "correlations"),
      (f"{name}_eta", 1000, -10, 10, 1000, -10, 10, "correlations"),
      (f"{name}_phi", 100, -3.2, 3.2, 100, -3.2, 3.2, "correlations"),
      (f"{name}_minv", 100, 0, 10, 100, 0, 10, "correlations"),
      (f"{name}_vx", 200, -300, 300, 200, -10, 10, "correlations"),
      (f"{name}_vy", 200, -300, 300, 200, -10, 10, "correlations"),
      (f"{name}_vz", 200, 0, 20000, 2000, 0, 20000, "correlations"),
  )

# ============================================================
# Other stuff
# ============================================================


def latest_versioned_sample():
  samples_dir = f"{base_path}/{sample}/{campaign}/samples/step4_merged"
  sample_pattern = re.compile(r"ntuple_0_([0-9a-f]{7,40}(?:-dirty-[0-9a-f]{8})?)\.root")
  samples = []
  for input_path in glob.glob(f"{samples_dir}/ntuple_0_*.root"):
    file_name = os.path.basename(input_path)
    match = sample_pattern.fullmatch(file_name)
    if match:
      samples.append((os.stat(input_path).st_mtime_ns, file_name, input_path, match.group(1)))

  if not samples:
    raise RuntimeError(f"No versioned ntuple_0_<hash>.root files found in '{samples_dir}'")

  samples.sort()
  _, _, input_path, provenance_tag = samples[-1]

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
        existing_version_for_hash = max(existing_version_for_hash, version_number)

  sample_version = existing_version_for_hash if existing_version_for_hash is not None else latest_version + 1
  return input_path, sample_version, provenance_tag


nEvents = -1

input_path, sample_version, provenance_tag = latest_versioned_sample()
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

inputFilePath = input_path
histogramsOutputFilePath = f"{project_dir}/plots/v{sample_version}_{provenance_tag}/histograms.root"

info(f"Selected sample v{sample_version}_{provenance_tag}: {inputFilePath}")
info(f"Histogram output: {histogramsOutputFilePath}")

weightsBranchName = "genWeight"
eventsTreeNames = ["Events",]
specialBranchSizes = {
    "Particle": "Event_numberP",
}
