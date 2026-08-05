## specify how many events to run on (and how often to print current event number)
nEvents = -1

# specify input/output paths 
# inputFilePath = "/user/jniedzie/shift_cmssw/CMSSW_14_0_24_patch1/src/test_run4d126/step4_nano.root"
inputFilePath = "/user/jniedzie/shift_cmssw/samples_tmp/jpsi/test_run/samples/step4/events_NanoAOD_part_0000.root"
histogramsOutputFilePath = "../test_hists.root"

extraEventCollections = {
    "GenJPsi": {
        "inputCollections": ["GenPart"],
        "pdgId": (443, 443),
        "status": (2, 2),
    },
    "GenMuonPlus": {
        "inputCollections": ["GenPart"],
        "pdgId": (13, 13),
    },
    "GenMuonMinus": {
        "inputCollections": ["GenPart"],
        "pdgId": (-13, -13),
    },
    "GenMuon": {
        "inputCollections": ["GenMuonPlus", "GenMuonMinus"],
        # "pt": (3., 9999999.),
        "status": (1, 1),
    },
    "LoosePATMuons": {
        "inputCollections": ("Muon",),
        "pt": (3., 9999999.),
        "eta": (-2.5, 2.5),
        "looseId": True,
    },
    "LooseDSAMuons": {
        "inputCollections": ("DSAMuon",),
        "pt": (3., 9999999.),
        "eta": (-2.5, 2.5),
        "displacedID": (1, 9999999.),
    },
    "LooseShiftMuons": {
        "inputCollections": ("ShiftMuon",),
        "vz": (5000., 9999999.),
        # "pt": (3., 9999999.),
        # "eta": (-2.5, 2.5),
    },
    "GoodShiftDimuonVertex": {
        "inputCollections": ("ShiftDimuonVertex",),
        "isOS": (1, 1),
        # "chi2": (0., 10.),
        "dcaValid": (1, 1),
        "dca": (50., 9999999.),
    },
}

defaultHistParams = (
#  collection      variable          bins    xmin     xmax     dir
  ("GenJPsi"     , "pt"                     , 400,    0,       200,     ""  ),
  ("GenJPsi"     , "eta"                    , 100,    -2.5,    2.5,     ""  ),
  
  ("Event", "nGenMuon", 10, 0, 10, ""),
  
  ("Event", "nShiftMuon", 10, 0, 10, ""),
  ("ShiftMuon"   , "pt"                     , 400,    0,       200,     ""  ),
  ("ShiftMuon"   , "pz"                     , 400,    -10000,       10000,     ""  ),
  ("ShiftMuon"   , "eta"                    , 100,    -10,      10,     ""  ),
  ("ShiftMuon"   , "vx"                     , 100,    -1000,  1000,     ""  ),
  ("ShiftMuon"   , "vy"                     , 100,    -1000,  1000,     ""  ),
  ("ShiftMuon"   , "vz"                     , 100,    -20000,  20000,     ""  ),
  ("ShiftMuon"   , "dz"                     , 100,    -20000,  20000,     ""  ),
  ("ShiftMuon"   , "linePcaZ"               , 100,    -20000,  20000,     ""  ),
  ("ShiftMuon"   , "trackVz"                , 100,    -20000,  20000,     ""  ),
  ("ShiftMuon"   , "genPartIdx"             , 30,    -10,      20,     ""  ),
  
  ("LooseShiftMuons"   , "pt"               , 400,    0,       200,     ""  ),
  ("LooseShiftMuons"   , "eta"              , 100,    -10,      10,     ""  ),
  ("LooseShiftMuons"   , "vx"               , 100,    -1000,  1000,     ""  ),
  ("LooseShiftMuons"   , "vy"               , 100,    -1000,  1000,     ""  ),
  ("LooseShiftMuons"   , "vz"               , 100,    -20000,  20000,     ""  ),
  ("LooseShiftMuons"   , "dz"                     , 100,    -20000,  20000,     ""  ),
  ("LooseShiftMuons"   , "linePcaZ"               , 100,    -20000,  20000,     ""  ),
  ("LooseShiftMuons"   , "trackVz"                , 100,    -20000,  20000,     ""  ),
  
  ("Event"               , "nShiftDimuonVertex", 10, 0, 10, ""),
  ("ShiftDimuonVertex"   , "x"              , 100,    -1000,  1000,     ""  ),
  ("ShiftDimuonVertex"   , "y"              , 100,    -1000,  1000,     ""  ),
  ("ShiftDimuonVertex"   , "z"              , 100,    -20000,  20000,     ""  ),
  ("ShiftDimuonVertex"   , "mass"           , 100,    0,       10,     ""  ),
  ("ShiftDimuonVertex"   , "pt"             , 400,    0,       200,     ""  ),
  ("ShiftDimuonVertex"   , "eta"            , 100,    -10,      10,     ""  ),
  ("ShiftDimuonVertex"   , "chi2"           , 100,    -1,     100,     ""  ),
  ("ShiftDimuonVertex"   , "normalizedChi2" , 100,    -1,     100,     ""  ),
  ("ShiftDimuonVertex"   , "isOS"           , 20,    -10,     10,     ""  ),
  ("ShiftDimuonVertex"   , "dca"            , 100,    0,     1500,     ""  ),
  ("ShiftDimuonVertex"   , "dcaValid"       , 20,    -10,     10,     ""  ),
  ("ShiftDimuonVertex"   , "sameGenMuon"     , 20,    -10,     10,     ""  ),
  ("ShiftDimuonVertex"   , "genIsOS"     , 20,    -10,     10,     ""  ),
  
  
  ("Event"               , "nGoodShiftDimuonVertex", 10, 0, 10, ""),
  ("GoodShiftDimuonVertex"   , "x"              , 100,    -1000,  1000,     ""  ),
  ("GoodShiftDimuonVertex"   , "y"              , 100,    -1000,  1000,     ""  ),
  ("GoodShiftDimuonVertex"   , "z"              , 100,    -20000,  20000,     ""  ),
  ("GoodShiftDimuonVertex"   , "mass"           , 100,    0,       10,     ""  ),
  ("GoodShiftDimuonVertex"   , "pt"             , 400,    0,       200,     ""  ),
  ("GoodShiftDimuonVertex"   , "eta"            , 100,    -10,      10,     ""  ),
  ("GoodShiftDimuonVertex"   , "chi2"           , 100,    -1,     100,     ""  ),
  ("GoodShiftDimuonVertex"   , "normalizedChi2" , 100,    -1,     100,     ""  ),
  ("GoodShiftDimuonVertex"   , "isOS"           , 20,    -10,     10,     ""  ),
  ("GoodShiftDimuonVertex"   , "dca"            , 100,    0,     1500,     ""  ),
  ("GoodShiftDimuonVertex"   , "dcaValid"       , 20,    -10,     10,     ""  ),
  
  ("GenMuon"      , "vx"   , 1000, -100,     100,     ""),  
  ("GenMuon"      , "vy"   , 1000, -100,     100,     ""),  
  ("GenMuon"      , "vz"   , 1000, -20000, 20000,     ""),
  ("GenMuon"      , "pt"  , 1000,    0,      200,     ""),
  ("GenMuon"      , "pz"  , 1000,    -10000,      10000,     ""),
  ("GenMuon"      , "eta" , 1000,   -10,       10,     ""),
  ("GenMuon"      , "phi" , 1000,   -3.2,     3.2,     ""),
  
)

histParams = (
#  collection  variable         bins  xmin    xmax    dir
  ("GenMuon"      , "minv", 1000,    0,      10,     ""),
  ("LooseDSAMuon" , "minv", 1000,  0,       10,     ""),
  ("LoosePATMuon" , "minv", 1000,  0,       10,     ""),
  ("LooseShiftMuons", "minv", 200,  0,       10,     ""),
)

histParams2D = (
  ("RecoVsGenMuon_vx", 100, -1000, 1000, 100, -1000, 1000),
  ("RecoVsGenMuon_vy", 100, -1000, 1000, 100, -1000, 1000),
  ("RecoVsGenMuon_vz", 100, -20000, 20000, 100, -20000, 20000),
  ("RecoVsGenMuon_pt", 100, 0, 200, 100, 0, 200),
  ("RecoVsGenMuon_eta", 100, -10, 10, 100, -10, 10),
  ("RecoVsGenMuon_phi", 100, -3.2, 3.2, 100, -3.2, 3.2),
)

# specify name of the branch containing event weights
weightsBranchName = "genWeight"

eventsTreeNames = ["Events",]
specialBranchSizes = {
  "Particle": "Event_numberP",
}