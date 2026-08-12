# ============================================================
# Default single-muon histograms
# ============================================================

defaultHistParams = ()
for name in ["GenMuon", "ShiftMuon", "ShiftMuonDoubleTraversing", "ShiftMuonTraversing", "ShiftMuonDSA", "ShiftMuonCosmic"]:
  defaultHistParams += (
    ("Event"     , f"n{name}"       , 10  , 0     , 10    , "event"),
    (f"{name}"   , "pt"             , 200 , 0     , 20    , "muon"),
    (f"{name}"   , "pz"             , 200 , -1000 , 100   , "muon"),
    (f"{name}"   , "eta"            , 100 , -10   , 10    , "muon"),
    (f"{name}"   , "phi"            , 100 , -3.2  , 3.2   , "muon"),
    (f"{name}"   , "vx"             , 100 , -300  , 300   , "muon"),
    (f"{name}"   , "vy"             , 100 , -300  , 300   , "muon"),
    (f"{name}"   , "vz"             , 100 , -20000, 20000 , "muon"),
  )
   
  if "Gen" in name:
    continue
  
  defaultHistParams += (
    (f"{name}"   , "constrainedPt"             , 200 , 0     , 20    , "muon"),
    (f"{name}"   , "constrainedPz"             , 200 , -1000 , 100   , "muon"),
    (f"{name}"   , "constrainedEta"            , 100 , -10   , 10    , "muon"),
    (f"{name}"   , "constrainedPhi"            , 100 , -3.2  , 3.2   , "muon"),
    (f"{name}"   , "constrainedVx"             , 100 , -300  , 300   , "muon"),
    (f"{name}"   , "constrainedVy"             , 100 , -300  , 300   , "muon"),
    (f"{name}"   , "constrainedVz"             , 100 , -20000, 20000 , "muon"),
    
    (f"{name}"   , "dz"             , 100 , -20000, 20000 , "muon"),
    (f"{name}"   , "linePcaZ"       , 100 , -20000, 20000 , "muon"),
    (f"{name}"   , "trackVz"        , 100 , -20000, 20000 , "muon"),  
    (f"{name}"   , "genPartIdx"     , 30  , -10   , 20    , "muon"),
  )

# ============================================================
# Default dimuon histograms
# ============================================================

for name in ["ShiftDimuonVertex", "GoodShiftDimuonVertex"]:
  defaultHistParams += (
    ("Event"     , f"n{name}"                     , 10  , 0     , 10    , "event"),
    (f"{name}"   , "pt"                           , 200 , 0     , 20    , "dimuon"),
    (f"{name}"   , "pz"                           , 200 , -1000 , 100   , "dimuon"),
    (f"{name}"   , "eta"                          , 100 , -10   , 10    , "dimuon"),
    (f"{name}"   , "phi"                          , 100 , -3.2  , 3.2   , "dimuon"),
    (f"{name}"   , "mass"                         , 100 , 0     , 10    , "dimuon"),
    (f"{name}"   , "vx"                           , 100 , -300  , 300   , "dimuon"),
    (f"{name}"   , "vy"                           , 100 , -300  , 300   , "dimuon"),
    (f"{name}"   , "vz"                           , 100 , -20000, 20000 , "dimuon"),
  )
   
  if "Gen" in name:
    continue 
  
  defaultHistParams += (
    (f"{name}"   , "chi2"           , 100 , -1    , 100   , "dimuon"),
    (f"{name}"   , "normalizedChi2" , 100 , -1    , 100   , "dimuon"),
    (f"{name}"   , "dca"            , 100 , 0     , 1500  , "dimuon"),
    (f"{name}"   , "dcaValid"       , 20  ,-10    , 10    , "dimuon"),
    
    (f"{name}"   , "isOS"           , 20  , -10   , 10    , "dimuon"),
    (f"{name}"   , "genIsOS"        , 20  ,-10    , 10    , "dimuon"),
    (f"{name}"   , "sameGenMuon"    , 20  ,-10    , 10    , "dimuon"),
  )

# ============================================================
# Custom 1D histograms
# ============================================================

histParams = (
  (f"GenDimuon"   , "pt"                           , 200 , 0     , 20    , "dimuon"),
  (f"GenDimuon"   , "pz"                           , 200 , -1000 , 100   , "dimuon"),
  (f"GenDimuon"   , "eta"                          , 100 , -10   , 10    , "dimuon"),
  (f"GenDimuon"   , "phi"                          , 100 , -3.2  , 3.2   , "dimuon"),
  (f"GenDimuon"   , "mass"                         , 100 , 0     , 10    , "dimuon"),
  (f"GenDimuon"   , "vx"                           , 100 , -300  , 300   , "dimuon"),
  (f"GenDimuon"   , "vy"                           , 100 , -300  , 300   , "dimuon"),
  (f"GenDimuon"   , "vz"                           , 100 , -20000, 20000 , "dimuon"),
  
  ("DimuonResolution" , "pt"    , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "pz"    , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "eta"   , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "phi"   , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "minv"  , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "vx"    , 100   , -5000, 5000,     "resolution"),
  ("DimuonResolution" , "vy"    , 100   , -5000, 5000,     "resolution"),
  ("DimuonResolution" , "vz"    , 50    , -1  , 1   ,     "resolution"),
)

for name in ["DoubleTraversing", "Traversing", "DSA", "Cosmic"]:
  histParams += (
    (f"MuonResolution{name}"   , "pt"    , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "pz"    , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "eta"   , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "phi"   , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "vx"    , 100   , -5000, 5000  ,     "resolution"),
    (f"MuonResolution{name}"   , "vy"    , 100   , -5000, 5000  ,     "resolution"),
    (f"MuonResolution{name}"   , "vz"    , 50    , -1  , 1      ,     "resolution"),
    
    (f"MuonResolution{name}"   , "constrainedPt"    , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "constrainedPz"    , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "constrainedEta"   , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "constrainedPhi"   , 50    , -1  , 1      ,     "resolution"),
    (f"MuonResolution{name}"   , "constrainedVx"    , 100   , -5000, 5000  ,     "resolution"),
    (f"MuonResolution{name}"   , "constrainedVy"    , 100   , -5000, 5000  ,     "resolution"),
    (f"MuonResolution{name}"   , "constrainedVz"    , 50    , -1  , 1      ,     "resolution"),
  )

# ============================================================
# 2D histograms
# ============================================================

histParams2D = ()

for name in ["RecoVsGenMuon", "RecoVsGenDimuon"]:

  histParams2D += (
    (f"{name}_pt"   , 1000  , 0     , 60    , 1000  , 0   , 60    , "correlations"),
    (f"{name}_pz"   , 500   , -500  , 100   , 500   , -500, 100   , "correlations"),
    (f"{name}_eta"  , 1000  , -10   , 10    , 1000  , -10 , 10    , "correlations"),
    (f"{name}_phi"  , 100   , -3.2  , 3.2   , 100   , -3.2, 3.2   , "correlations"),
    (f"{name}_minv" , 100   , 0     , 10    , 100   , 0   , 10    , "correlations"),
    (f"{name}_vx"   , 200   , -300  , 300   , 200   , -10 , 10    , "correlations"),
    (f"{name}_vy"   , 200   , -300  , 300   , 200   , -10 , 10    , "correlations"),
    (f"{name}_vz"   , 200   , 0     , 20000 , 2000  , 0   , 20000 , "correlations"),
  )

# ============================================================
# Other stuff
# ============================================================

import glob
import os
import re

from shift_extra_collections import extraEventCollections
from shift_paths import base_path, campaign, sample
from Logger import info


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
  return input_path, len(samples), provenance_tag

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
