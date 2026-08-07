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
    "LooseShiftMuon": {
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