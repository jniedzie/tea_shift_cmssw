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
}

defaultHistParams = (
#  collection      variable          bins    xmin     xmax     dir
  ("Event"       , "nMuon"         , 50,     0,       50,      ""  ),
  ("GenJPsi"     , "pt"            , 400,    0,       200,     ""  ),
  ("GenJPsi"     , "eta"           , 100,    -2.5,    2.5,     ""  ),
  
  ("ShiftMuon"   , "pt"            , 400,    0,       200,     ""  ),
  ("ShiftMuon"   , "eta"           , 100,    -10,      10,     ""  ),
  ("ShiftMuon"   , "vx"            , 100,    -1000,  1000,     ""  ),
  ("ShiftMuon"   , "vy"            , 100,    -1000,  1000,     ""  ),
  ("ShiftMuon"   , "vz"            , 100,    -20000,  20000,     ""  ),
  
  ("LooseShiftMuons"   , "pt"            , 400,    0,       200,     ""  ),
  ("LooseShiftMuons"   , "eta"           , 100,    -10,      10,     ""  ),
  ("LooseShiftMuons"   , "vx"            , 100,    -1000,  1000,     ""  ),
  ("LooseShiftMuons"   , "vy"            , 100,    -1000,  1000,     ""  ),
  ("LooseShiftMuons"   , "vz"            , 100,    -20000,  20000,     ""  ),
)

histParams = (
#  collection  variable         bins  xmin    xmax    dir
  ("GenMuon"      , "minv", 1000,    0,      10,     ""),
  ("GenMuon"      , "x"   , 1000, -100,     100,     ""),  
  ("GenMuon"      , "y"   , 1000, -100,     100,     ""),  
  ("GenMuon"      , "z"   , 1000, -20000, 20000,     ""),
  ("GenMuon"      , "logZ", 100,  -10   , 10   ,     ""),
  ("GenMuon"      , "pt"  , 1000,    0,      100,     ""),
  ("GenMuon"      , "eta" , 1000,   -10,       10,     ""),
  ("GenMuon"      , "phi" , 1000,   -3.2,     3.2,     ""),
  ("LooseDSAMuon" , "minv", 1000,  0,       10,     ""),
  ("LoosePATMuon" , "minv", 1000,  0,       10,     ""),
  ("LooseShiftMuons", "minv", 200,  0,       10,     ""),
)


# specify name of the branch containing event weights
weightsBranchName = "genWeight"

eventsTreeNames = ["Events",]
specialBranchSizes = {
  "Particle": "Event_numberP",
}