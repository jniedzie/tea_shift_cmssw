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
    "GenMuon": {
        "inputCollections": ["GenPart"],
        "pdgId": (13, -13),
        "status": (1, 1),
    },
    "LoosePATMuons": {
        "inputCollections": ("Muon",),
        "pt": (3., 9999999.),
        "eta": (-2.5, 2.5),
        "looseId": True,
    },
}

defaultHistParams = (
#  collection      variable          bins    xmin     xmax     dir
  ("Event"       , "nMuon"         , 50,     0,       50,      ""  ),
  ("GenJPsi"     , "pt"            , 400,    0,       200,     ""  ),
  ("GenJPsi"     , "eta"           , 100,    -2.5,    2.5,     ""  ),
)

histParams = (
#  collection  variable         bins  xmin    xmax    dir
  ("Muon"     , "minv", 1000,  0,      10,     ""),
  ("GenMuon"  , "minv", 1000, -3.5,    3.5,    ""),
)


# specify name of the branch containing event weights
weightsBranchName = "genWeight"

eventsTreeNames = ["Events",]
specialBranchSizes = {
  "Particle": "Event_numberP",
}