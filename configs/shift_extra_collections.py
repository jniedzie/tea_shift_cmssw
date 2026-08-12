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
    "ShiftMuonDoubleTraversing": {
        "inputCollections": ("ShiftMuon",),
        "quality": 3,
    },
    "ShiftMuonTraversing": {
        "inputCollections": ("ShiftMuon",),
        "quality": 2,
    },
    "ShiftMuonDSA": {
        "inputCollections": ("ShiftMuon",),
        "quality": 1,
    },
    "ShiftMuonCosmic": {
        "inputCollections": ("ShiftMuon",),
        "quality": 0,
    },
    "GoodShiftDimuonVertex": {
        "inputCollections": ("ShiftDimuonVertex",),
        "isOS": (1, 1),
        # "chi2": (0., 10.),
        "dcaValid": (1, 1),
        "dca": (50., 9999999.),
    },
}
