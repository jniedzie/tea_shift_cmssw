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
    "ShiftDimuonVertexGood": {
        "inputCollections": ("ShiftDimuonVertex",),
        "isOS": (1, 1),
        # "chi2": (0., 10.),
        "dcaValid": (1, 1),
        "dca": (50., 9999999.),
    },
}

for topologyMin in range(0, 5):
  for topologyMax in range(topologyMin, 5):
    extraEventCollections[f"ShiftDimuonVertexMin{topologyMin}-Max{topologyMax}"] = {
        "inputCollections": ("ShiftDimuonVertex",),
        "topologyMin": topologyMin,
        "topologyMax": topologyMax,
    }


extraEventCollections[f"ShiftDimuonVertexNear-Both"] = {"inputCollections": ("ShiftDimuonVertexMin0-Max2",)}
extraEventCollections[f"ShiftDimuonVertexNear-Far"] = {"inputCollections": ("ShiftDimuonVertexMin0-Max3",)}
extraEventCollections[f"ShiftDimuonVertexBoth-Both"] = {"inputCollections": ("ShiftDimuonVertexMin2-Max2",)}
extraEventCollections[f"ShiftDimuonVertexBoth-Far"] = {"inputCollections": ("ShiftDimuonVertexMin2-Max3",)}
extraEventCollections[f"ShiftDimuonVertexOther"] = {
  "inputCollections": (
    "ShiftDimuonVertexMin0-Max0",
    "ShiftDimuonVertexMin0-Max1",
    "ShiftDimuonVertexMin0-Max4",
    "ShiftDimuonVertexMin1-Max1",
    "ShiftDimuonVertexMin1-Max2",
    "ShiftDimuonVertexMin1-Max3",
    "ShiftDimuonVertexMin1-Max4",
    "ShiftDimuonVertexMin2-Max4",
    "ShiftDimuonVertexMin3-Max3",
    "ShiftDimuonVertexMin3-Max4",
    "ShiftDimuonVertexMin4-Max4",
  )
}
