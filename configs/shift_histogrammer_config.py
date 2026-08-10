# ============================================================
# Default single-muon histograms
# ============================================================

defaultHistParams = ()
for name in ["GenMuon", "ShiftMuon", "LooseShiftMuon"]:
  defaultHistParams += (
    ("Event"     , f"n{name}"       , 10  , 0     , 10    , "event"),
    (f"{name}"   , "pt"             , 200 , 0     , 20    , "muon"),
    (f"{name}"   , "pz"             , 200 , -1000 , 100   , "muon"),
    (f"{name}"   , "eta"            , 100 , -10   , 10    , "muon"),
    (f"{name}"   , "phi"            , 100 , -3.2  , 3.2   , "muon"),
    (f"{name}"   , "vx"             , 100 , -1000 , 1000  , "muon"),
    (f"{name}"   , "vy"             , 100 , -1000 , 1000  , "muon"),
    (f"{name}"   , "vz"             , 100 , -20000, 20000 , "muon"),
  )
   
  if "Gen" in name:
    continue
  
  defaultHistParams += (
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
    (f"{name}"   , "vx"                           , 100 , -1000 , 1000  , "dimuon"),
    (f"{name}"   , "vy"                           , 100 , -1000 , 1000  , "dimuon"),
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
  ("LooseShiftMuon"   , "minv"  , 200   ,  0  , 10  ,     "dimuon"),
  
  (f"GenDimuon"   , "pt"                           , 200 , 0     , 20    , "dimuon"),
  (f"GenDimuon"   , "pz"                           , 200 , -1000 , 100   , "dimuon"),
  (f"GenDimuon"   , "eta"                          , 100 , -10   , 10    , "dimuon"),
  (f"GenDimuon"   , "phi"                          , 100 , -3.2  , 3.2   , "dimuon"),
  (f"GenDimuon"   , "mass"                         , 100 , 0     , 10    , "dimuon"),
  (f"GenDimuon"   , "vx"                           , 100 , -1000 , 1000  , "dimuon"),
  (f"GenDimuon"   , "vy"                           , 100 , -1000 , 1000  , "dimuon"),
  (f"GenDimuon"   , "vz"                           , 100 , -20000, 20000 , "dimuon"),

  ("MuonResolution"   , "pt"    , 50    , -1  , 1   ,     "resolution"),
  ("MuonResolution"   , "pz"    , 50    , -1  , 1   ,     "resolution"),
  ("MuonResolution"   , "eta"   , 50    , -1  , 1   ,     "resolution"),
  ("MuonResolution"   , "phi"   , 50    , -1  , 1   ,     "resolution"),
  ("MuonResolution"   , "vx"    , 100   , -5000, 5000,     "resolution"),
  ("MuonResolution"   , "vy"    , 100   , -5000, 5000,     "resolution"),
  ("MuonResolution"   , "vz"    , 50    , -1  , 1   ,     "resolution"),
  
  ("DimuonResolution" , "pt"    , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "pz"    , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "eta"   , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "phi"   , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "minv"  , 50    , -1  , 1   ,     "resolution"),
  ("DimuonResolution" , "vx"    , 100   , -5000, 5000,     "resolution"),
  ("DimuonResolution" , "vy"    , 100   , -5000, 5000,     "resolution"),
  ("DimuonResolution" , "vz"    , 50    , -1  , 1   ,     "resolution"),
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
    (f"{name}_vx"   , 200   , -400  , 400   , 200   , -10 , 10    , "correlations"),
    (f"{name}_vy"   , 200   , -400  , 400   , 200   , -10 , 10    , "correlations"),
    (f"{name}_vz"   , 200   , 0     , 20000 , 2000  , 0   , 20000 , "correlations"),
  )

# ============================================================
# Other stuff
# ============================================================

from shift_extra_collections import extraEventCollections

nEvents = -1

inputFilePath = "/eos/home-j/jniedzie/shift_cmssw/jpsi/Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_10k_beamB/samples/step4_merged/ntuple_0_2f9aeb2ab027.root"
histogramsOutputFilePath = "../test_hists_10k.root"

weightsBranchName = "genWeight"
eventsTreeNames = ["Events",]
specialBranchSizes = {
  "Particle": "Event_numberP",
}
