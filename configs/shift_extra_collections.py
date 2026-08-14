extraEventCollections = {
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
        "status": (1, 1),
    },
    "ShiftMuonNearEndcapOnly": {
        "inputCollections": ("ShiftMuon",),
        "topology": 0,
    },
    "ShiftMuonNearEndcapAndBarrel": {
        "inputCollections": ("ShiftMuon",),
        "topology": 1,
    },
    "ShiftMuonBothEndcaps": {
        "inputCollections": ("ShiftMuon",),
        "topology": 2,
    },
    "ShiftMuonFarEndcapOnly": {
        "inputCollections": ("ShiftMuon",),
        "topology": 3,
    },
    "ShiftMuonUnclassified": {
        "inputCollections": ("ShiftMuon",),
        "topology": 4,
    },
    "GoodShiftDimuonVertex": {
        "inputCollections": ("ShiftDimuonVertex",),
        "isOS": (1, 1),
        # "chi2": (0., 10.),
        "dcaValid": (1, 1),
        "dca": (50., 9999999.),
    },
}
